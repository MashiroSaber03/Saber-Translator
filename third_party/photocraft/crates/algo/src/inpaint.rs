//! Hole filling for Content-Aware Fill and Spot Healing: patch-based completion, proximity match
//! and texture synthesis.
//!
//! * [`complete`], content-aware fill: the multi-scale "space-time completion" EM scheme of
//!   Y. Wexler, E. Shechtman, M. Irani, *Space-Time Completion of Video*, IEEE TPAMI 2007, with the
//!   nearest-neighbour field computed by C. Barnes, E. Shechtman, A. Finkelstein, D. B. Goldman,
//!   *PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing*, SIGGRAPH 2009
//!   (random init, propagation, exponentially shrinking random search). Each EM iteration finds, for
//!   every patch overlapping the hole, its most similar source patch, then re-estimates hole pixels
//!   by a similarity-weighted vote of the overlapping matches. Coarse to fine over an image pyramid
//!   (down to where the hole is about four patches across): the coarsest level starts by filling
//!   the hole from its edge inwards, ring by ring, each pixel from the patch that best matches what
//!   is known around it (rather than from a smooth membrane, which draws the fill towards flat
//!   areas); every finer level starts from the coarser level's matches scaled up and voted with the
//!   finer level's own pixels, so the texture stays sharp. Where neighbouring patches agree on their
//!   source (the usual case), the vote copies the source pixels exactly; only along the seams
//!   between copied pieces does it blend.
//!
//!   This is what Photoshop's Content-Aware Fill does, measured as a black box on Photoshop 25.4
//!   (see `content_aware`): with colour adaptation off its fill is pieces of the image copied pixel
//!   for pixel, typically 15–40 px across, joined by seams a few pixels wide; with colour
//!   adaptation the copies come with a gain and bias. That is the patch transform of S. Darabi,
//!   E. Shechtman, C. Barnes, D. B. Goldman, P. Sen, *Image Melding: Combining Inconsistent Images
//!   using Patch-based Synthesis*, SIGGRAPH 2012: each source patch may be scaled and offset per
//!   channel to match its target, within limits ([`Adapt`]), and is voted in adjusted.
//! * [`best_offset`], proximity match: the single displacement whose surrounding ring best matches the
//!   ring around the hole (sum of squared differences), i.e. an automatic Healing Brush source.
//! * [`synthesize`], create texture: patch-based texture synthesis in the style of A. Efros,
//!   W. Freeman, *Image Quilting for Texture Synthesis and Transfer*, SIGGRAPH 2001 (blocks picked from
//!   random candidates by overlap error, joined along minimum-error boundary cuts, so the texture is
//!   copied, never averaged).
//!
//! All functions take interleaved `w × h × ch` normalised floats (any colour model/depth) and a
//! `w × h` hole mask, and return a full buffer equal to the input outside the hole. They are
//! deterministic (fixed-seed hash RNG, results independent of thread count).

use std::borrow::Cow;

use crate::par;
use crate::poisson::membrane_fill;

mod proximity_integral;
use proximity_integral::HoleIntegral;

/// Below this many cells a pass runs on the calling thread (thread hand-off costs more than it
/// saves). Every pass gives the same result either way.
const PAR_MIN: usize = 64 * 64;

/// [`PAR_MIN`] for the passes that match or vote a patch per cell.
const PAR_MIN_PATCHES: usize = 16 * 16;

/// Largest level (in cells) whose source patches' moments [`Solver::prepare`] keeps.
const SOURCE_MOMENTS_MAX: usize = 1 << 20;

/// SplitMix64: tiny deterministic RNG.
#[derive(Clone)]
pub(crate) struct Rng(u64);
impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Skip `n` numbers (the same state as after `n` calls of [`Rng::next`]).
    pub(crate) fn skip(&mut self, n: u64) {
        self.0 = self.0.wrapping_add(n.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    }
    /// Uniform integer in `lo..=hi`.
    pub(crate) fn range(&mut self, lo: i32, hi: i32) -> i32 {
        if hi <= lo {
            return lo;
        }
        lo + (self.next() % (hi - lo + 1) as u64) as i32
    }
}

/// How far a copied patch may be adjusted to its target, per channel: `gain · source + bias`, the
/// gain within `gain` and the bias within `±bias` (Image Melding's photometric transform). The
/// gain is the ratio of the patches' standard deviations, the bias what then matches their means.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Adapt {
    pub gain: (f32, f32),
    pub bias: f32,
}

impl Adapt {
    /// Copies are taken as they are.
    pub const NONE: Adapt = Adapt { gain: (1.0, 1.0), bias: 0.0 };

    pub fn is_none(&self) -> bool {
        self.gain == (1.0, 1.0) && self.bias == 0.0
    }
}

impl Default for Adapt {
    fn default() -> Self {
        Self::NONE
    }
}

/// Parameters for [`complete`].
#[derive(Clone, Debug, PartialEq)]
pub struct CompleteParams {
    /// Patch radius (patches are `2r+1` square).
    pub patch_radius: usize,
    /// EM iterations at the coarsest level (finer levels use fewer, down to `min_em_iters`).
    pub em_iters: usize,
    pub min_em_iters: usize,
    /// PatchMatch passes per EM iteration.
    pub pm_iters: usize,
    /// Levels where the hole is wider than this (in that level's pixels) only *refine* the
    /// scaled-up match field (one local PatchMatch pass and one vote), which keeps big holes
    /// tractable: the structure is decided at coarse scale, fine levels just add detail.
    pub refine_extent: usize,
    /// Colour adaptation of the copies.
    pub adapt: Adapt,
    pub seed: u64,
}

impl Default for CompleteParams {
    fn default() -> Self {
        Self { patch_radius: 3, em_iters: 10, min_em_iters: 3, pm_iters: 2, refine_extent: 192, adapt: Adapt::NONE, seed: 1 }
    }
}

/// What a pixel is to [`complete_masked`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Px {
    /// Not part of the image (a gap, or outside a transformed copy): ignored.
    Void,
    /// Known, part of the context the fill must agree with, but not copied from.
    Known,
    /// Known and may be copied from.
    Source,
    /// To be filled.
    Hole,
}

/// One pyramid level. Level 0 borrows the caller's buffers until it is solved.
struct Level<'a> {
    w: usize,
    h: usize,
    img: Cow<'a, [f32]>,
    px: Cow<'a, [Px]>,
}

impl Level<'_> {
    const EMPTY: Level<'static> = Level { w: 0, h: 0, img: Cow::Borrowed(&[]), px: Cow::Borrowed(&[]) };
}

/// The half-open box `[x0, x1) × [y0, y1)` of a level's grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Bounds {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
}

impl Bounds {
    /// Grown by `r` on every side, within a `w × h` grid.
    fn grow(self, r: usize, w: usize, h: usize) -> Bounds {
        Bounds { x0: self.x0.saturating_sub(r), y0: self.y0.saturating_sub(r), x1: self.x1.saturating_add(r).min(w), y1: self.y1.saturating_add(r).min(h) }
    }

    /// The same box on the next coarser level (whose cells are 2 × 2 of these).
    fn half(self) -> Bounds {
        Bounds { x0: self.x0 / 2, y0: self.y0 / 2, x1: self.x1.div_ceil(2), y1: self.y1.div_ceil(2) }
    }

    fn width(&self) -> usize {
        self.x1.saturating_sub(self.x0)
    }

    fn height(&self) -> usize {
        self.y1.saturating_sub(self.y0)
    }

    fn area(&self) -> usize {
        self.width() * self.height()
    }
}

/// Half-size level: a cell is a hole if any of its pixels is; else it averages its non-void
/// pixels, and may be copied from only if all of them may.
fn downsample(l: &Level, ch: usize) -> Level<'static> {
    let (w, h) = (l.w.div_ceil(2), l.h.div_ceil(2));
    let mut img = vec![0.0f32; w * h * ch];
    let mut px = vec![Px::Void; w * h];
    let (lw, lh, limg, lpx) = (l.w, l.h, &*l.img, &*l.px);
    let row = |y: usize, irow: &mut [f32], prow: &mut [Px]| {
        for (x, p) in prow.iter_mut().enumerate() {
            let mut n = 0.0f32;
            let o = x * ch;
            let (mut hole, mut all_source, mut any) = (false, true, false);
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let (sx, sy) = (x * 2 + dx, y * 2 + dy);
                if sx >= lw || sy >= lh {
                    continue;
                }
                let si = sy * lw + sx;
                match lpx[si] {
                    Px::Hole => hole = true,
                    Px::Void => all_source = false,
                    k => {
                        any = true;
                        all_source &= k == Px::Source;
                        for c in 0..ch {
                            irow[o + c] += limg[si * ch + c];
                        }
                        n += 1.0;
                    }
                }
            }
            if n > 0.0 {
                for c in 0..ch {
                    irow[o + c] /= n;
                }
            }
            *p = if hole {
                Px::Hole
            } else if !any {
                Px::Void
            } else if all_source {
                Px::Source
            } else {
                Px::Known
            };
        }
    };
    par::chunks2_mut(&mut img, w * ch, &mut px, w, w * h >= PAR_MIN, row);
    Level { w, h, img: Cow::Owned(img), px: Cow::Owned(px) }
}

/// Summed-area table of a boolean grid (`(w+1) × (h+1)`). Not inlined: in [`synthesize`] it made
/// Create Texture's candidate loop slower (measured).
#[inline(never)]
fn integral(w: usize, h: usize, m: &[bool]) -> Vec<u32> {
    let mut s = vec![0u32; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row = 0u32;
        for x in 0..w {
            row += u32::from(m[y * w + x]);
            s[(y + 1) * (w + 1) + x + 1] = s[y * (w + 1) + x + 1] + row;
        }
    }
    s
}

/// Count of set cells in the clipped window `[x0, x1) × [y0, y1)`.
fn window_count(s: &[u32], gw: usize, gh: usize, x0: i32, y0: i32, x1: i32, y1: i32) -> u32 {
    let (x0, y0) = (x0.max(0) as usize, y0.max(0) as usize);
    let (x1, y1) = ((x1.max(0) as usize).min(gw), (y1.max(0) as usize).min(gh));
    if x1 <= x0 || y1 <= y0 {
        return 0;
    }
    let w1 = gw + 1;
    s[y1 * w1 + x1] + s[y0 * w1 + x0] - s[y0 * w1 + x1] - s[y1 * w1 + x0]
}

fn hole_bbox(w: usize, h: usize, hole: &[bool]) -> Option<(usize, usize, usize, usize)> {
    let mut b: Option<(usize, usize, usize, usize)> = None;
    for y in 0..h {
        for x in 0..w {
            if hole[y * w + x] {
                b = Some(match b {
                    None => (x, y, x + 1, y + 1),
                    Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1)),
                });
            }
        }
    }
    b
}

/// Bounds of the hole pixels of a role grid.
fn hole_bounds(w: usize, h: usize, px: &[Px]) -> Option<Bounds> {
    // Per row the first and last hole column; the union of the rows' spans.
    let rows: Vec<Option<(usize, usize)>> = par::map(h, w * h >= PAR_MIN, |y| {
        let row = px.get(y * w..(y + 1) * w)?;
        let x0 = row.iter().position(|k| *k == Px::Hole)?;
        let x1 = row.iter().rposition(|k| *k == Px::Hole).unwrap_or(x0) + 1;
        Some((x0, x1))
    });
    let mut b: Option<Bounds> = None;
    for (y, span) in rows.into_iter().enumerate() {
        if let Some((x0, x1)) = span {
            b = Some(match b {
                None => Bounds { x0, y0: y, x1, y1: y + 1 },
                Some(b) => Bounds { x0: b.x0.min(x0), y0: b.y0, x1: b.x1.max(x1), y1: y + 1 },
            });
        }
    }
    b
}

/// Valid source centres: the whole `2r+1` patch inside the grid and copyable. Found per row
/// (each pixel's horizontal run), then per column of runs. `None` when cancelled.
fn valid_sources(w: usize, h: usize, r: usize, px: &[Px], ctl: &photocraft_raster::Interrupt) -> Option<Vec<bool>> {
    const BAND: usize = 64;
    let side = 2 * r + 1;
    let par = w * h >= PAR_MIN;
    let mut run = vec![false; w * h];
    if w < side || h < side {
        return Some(run);
    }
    // Per row: is the run of `side` pixels centred on each one all copyable (a sliding count)?
    par::chunks_mut(&mut run, w, par, |y, out| {
        if y % BAND == 0 && ctl.cancelled() {
            return;
        }
        let row = &px[y * w..(y + 1) * w];
        let copyable = |x: usize| usize::from(row[x] == Px::Source);
        let mut count: usize = (0..side).map(copyable).sum();
        out[r] = count == side;
        for (x, o) in out.iter_mut().enumerate().take(w - r).skip(r + 1) {
            count = count + copyable(x + r) - copyable(x - r - 1);
            *o = count == side;
        }
    });
    if ctl.cancelled() {
        return None;
    }
    // Per column, in bands of rows: how many such runs end each row in a row; a centre is valid
    // where `side` of them stack up to `r` rows below it.
    let mut valid = vec![false; w * h];
    par::chunks_mut(&mut valid, BAND * w, par, |band, out| {
        if ctl.cancelled() {
            return;
        }
        let (y0, rows) = (band * BAND, out.len() / w);
        let mut stack = vec![0usize; w];
        for yy in y0.saturating_sub(r)..(y0 + rows + r).min(h) {
            for (x, n) in stack.iter_mut().enumerate() {
                *n = if run[yy * w + x] { *n + 1 } else { 0 };
            }
            // `yy` is the bottom row of centre `y`'s patch.
            if yy < y0 + r || yy < 2 * r {
                continue;
            }
            let y = yy - r;
            for (v, n) in out[(y - y0) * w..(y - y0 + 1) * w].iter_mut().zip(&stack).take(w - r).skip(r) {
                *v = *n >= side;
            }
        }
    });
    if ctl.cancelled() {
        return None;
    }
    Some(valid)
}

/// Columns per block of [`Sources`]' index.
const SOURCE_BLOCK: usize = 64;

/// The valid source centres in row-major order, indexed by counts per row and per block of
/// [`SOURCE_BLOCK`] columns (so the `k`-th is found without listing them all).
struct Sources {
    /// Valid centres before each row (`h + 1` entries).
    row_start: Vec<usize>,
    /// Per row, the valid centres in it before each block.
    block_start: Vec<u32>,
    blocks: usize,
}

impl Sources {
    fn new(valid: &[bool], w: usize) -> Sources {
        let h = valid.len().checked_div(w).unwrap_or(0);
        let blocks = w.div_ceil(SOURCE_BLOCK);
        let mut block_start = vec![0u32; h * blocks];
        // Per row: the block prefix, and the row's count.
        let counts: Vec<usize> = par::map(h, valid.len() >= PAR_MIN, |y| valid[y * w..(y + 1) * w].iter().filter(|v| **v).count());
        par::chunks_mut(&mut block_start, blocks, valid.len() >= PAR_MIN, |y, out| {
            let mut n = 0u32;
            for (b, o) in out.iter_mut().enumerate() {
                *o = n;
                n += valid[y * w + b * SOURCE_BLOCK..y * w + ((b + 1) * SOURCE_BLOCK).min(w)].iter().filter(|v| **v).count() as u32;
            }
        });
        let mut row_start = Vec::with_capacity(counts.len() + 1);
        let mut total = 0usize;
        row_start.push(0);
        for c in counts {
            total += c;
            row_start.push(total);
        }
        Sources { row_start, block_start, blocks }
    }

    fn len(&self) -> usize {
        self.row_start.last().copied().unwrap_or(0)
    }

    /// The `k`-th valid centre in row-major order (`k < len`).
    fn nth(&self, k: usize, valid: &[bool], w: usize) -> (i32, i32) {
        let y = self.row_start.partition_point(|&s| s <= k).saturating_sub(1);
        let left = k.saturating_sub(self.row_start.get(y).copied().unwrap_or(0));
        let row_blocks = self.block_start.get(y * self.blocks..(y + 1) * self.blocks).unwrap_or(&[]);
        let b = row_blocks.partition_point(|&s| s as usize <= left).saturating_sub(1);
        let mut left = left.saturating_sub(row_blocks.get(b).copied().unwrap_or(0) as usize);
        let x0 = b * SOURCE_BLOCK;
        for (x, v) in valid.iter().skip(y * w + x0).take(w.saturating_sub(x0)).enumerate() {
            if *v {
                if left == 0 {
                    return ((x0 + x) as i32, y as i32);
                }
                left -= 1;
            }
        }
        (0, 0)
    }

    /// All of them, in row-major order.
    fn all(&self, valid: &[bool], w: usize) -> Vec<(i32, i32)> {
        let mut out = Vec::with_capacity(self.len());
        for (i, v) in valid.iter().enumerate() {
            if *v {
                out.push(((i % w) as i32, (i / w) as i32));
            }
        }
        out
    }
}

/// Per cell of `tb` (the hole's bounds `hb` grown by the patch radius, where all of them lie): is
/// it a target centre (its patch overlaps the hole)?
fn targets_around(w: usize, r: usize, px: &[Px], hb: Bounds, tb: Bounds) -> Vec<bool> {
    // Hole counts summed over the hole's bounds: every hole pixel lies inside them.
    let (bw, bh) = (hb.width(), hb.height());
    let mut s = vec![0u32; (bw + 1) * (bh + 1)];
    for y in 0..bh {
        let mut row = 0u32;
        for x in 0..bw {
            row += u32::from(px[(hb.y0 + y) * w + hb.x0 + x] == Px::Hole);
            s[(y + 1) * (bw + 1) + x + 1] = s[y * (bw + 1) + x + 1] + row;
        }
    }
    let mut targets = vec![false; tb.area()];
    let ri = r as i32;
    let (ox, oy) = (hb.x0 as i32, hb.y0 as i32);
    for y in tb.y0..tb.y1 {
        for x in tb.x0..tb.x1 {
            let k = (y - tb.y0) * tb.width() + (x - tb.x0);
            let (x, y) = (x as i32, y as i32);
            targets[k] = window_count(&s, bw, bh, x - ri - ox, y - ri - oy, x + ri + 1 - ox, y + ri + 1 - oy) > 0;
        }
    }
    targets
}

/// A finished coarser level: its size, pixels and match field (the source centre per cell of
/// its target box, `i32::MIN` where there is none; empty when it had nothing to copy).
struct Coarser {
    w: usize,
    h: usize,
    img: Vec<f32>,
    field: Vec<(i32, i32)>,
    tbox: Bounds,
}

impl Coarser {
    /// The match of cell `(x, y)` (`i32::MIN` outside the target box).
    fn field_at(&self, x: usize, y: usize) -> (i32, i32) {
        let t = self.tbox;
        if x < t.x0 || x >= t.x1 || y < t.y0 || y >= t.y1 {
            return (i32::MIN, i32::MIN);
        }
        self.field.get((y - t.y0) * t.width() + (x - t.x0)).copied().unwrap_or((i32::MIN, i32::MIN))
    }
}

/// Most channels whose moments the colour adaptation fits (the rest are copied as they are).
const MOM_CH: usize = 8;

/// Per-level PatchMatch/EM state.
struct Solver<'a> {
    w: usize,
    h: usize,
    ch: usize,
    r: i32,
    img: &'a mut Vec<f32>,
    px: &'a [Px],
    /// Valid source centres: the whole patch is inside the grid and may be copied from.
    valid: Vec<bool>,
    /// The hole's bounds and the targets' (the hole's grown by the patch radius): nothing outside
    /// them changes or is matched. The match field, its costs and these per-target tables are
    /// kept over `tbox`, row-major.
    hole_box: Bounds,
    tbox: Bounds,
    /// Target centres (the patch overlaps the hole).
    targets: Vec<bool>,
    /// Per cell of `tbox`: the whole patch is inside the grid and not void (so its distances
    /// need no bounds checks and its own moments can be kept).
    full: Vec<bool>,
    /// With colour adaptation, per cell of `tbox` and channel (at most [`MOM_CH`]): `Σt, Σt²` of
    /// the patch, for full patches, as of the last [`Solver::costs`].
    tmom: Vec<[f32; 2]>,
    /// With colour adaptation where it pays: per source centre and channel, `Σs, Σs²` of its
    /// patch (source pixels never change within a level). Empty otherwise.
    smom: Vec<[f32; 2]>,
    adapt: Adapt,
    /// Cancellation: checked per row band; a cancelled solve's output is discarded.
    ctl: photocraft_raster::Interrupt<'a>,
}

/// A channel deviation under which a patch counts as flat for [`fit`] (about 5 levels of 255).
const FLAT: f32 = 0.02;

/// Gain and bias of one channel for target moments `(Σt, Σt²)` and source moments `(Σs, Σs²)`
/// over `n` samples, within the limits.
#[inline]
pub(crate) fn fit(a: Adapt, n: f32, st: f32, stt: f32, ss: f32, sss: f32) -> (f32, f32) {
    let (mt, ms) = (st / n, ss / n);
    let vt = (stt / n - mt * mt).max(0.0);
    let vs = (sss / n - ms * ms).max(0.0);
    let g = if vs > 1e-10 { (vt / vs).sqrt() } else { 1.0 };
    let (mut lo, mut hi) = a.gain;
    // Nearly flat patches of about the same colour (their means within the bias of each other):
    // the ratio of their deviations is noise, so match the mean first, keeping the gain where the
    // bias can still bring the source's mean to the target's. Gained about zero, a gain at its
    // limit moved bright means further than the bias could bring back, and each channel by a
    // different amount: a tinted blotch in a smooth gradient. Textured patches, and flat ones
    // whose colours differ more (Content-Aware Move's Color), are fitted as before.
    if vt.max(vs) < FLAT * FLAT && (mt - ms).abs() <= a.bias && ms > 1e-6 {
        let (glo, ghi) = ((mt - a.bias) / ms, (mt + a.bias) / ms);
        if glo.max(lo) <= ghi.min(hi) {
            (lo, hi) = (glo.max(lo), ghi.min(hi));
        }
    }
    let g = g.clamp(lo, hi);
    let b = (mt - g * ms).clamp(-a.bias, a.bias);
    (g, b)
}

impl Solver<'_> {
    #[inline]
    fn is_valid(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.w && (y as usize) < self.h && self.valid[y as usize * self.w + x as usize]
    }

    /// Mark the full target patches (`full`), and keep the source moments where they pay off.
    fn prepare(&mut self, keep_source_moments: bool) {
        let (w, h, r, tb) = (self.w, self.h, self.r as usize, self.tbox);
        let side = 2 * r + 1;
        let mut full = vec![false; tb.area()];
        let px = self.px;
        par::chunks_mut(&mut full, tb.width(), tb.area() >= PAR_MIN_PATCHES, |ty, out| {
            let y = tb.y0 + ty;
            if y < r || y + r >= h {
                return;
            }
            for (tx, f) in out.iter_mut().enumerate() {
                let x = tb.x0 + tx;
                if x < r || x + r >= w {
                    continue;
                }
                *f = (y - r..=y + r).all(|yy| px[yy * w + x - r..yy * w + x - r + side].iter().all(|k| *k != Px::Void));
            }
        });
        self.full = full;
        if self.adapt.is_none() {
            return;
        }
        let c8 = self.ch.min(MOM_CH);
        self.tmom = vec![[0.0; 2]; tb.area() * c8];
        if !keep_source_moments {
            return;
        }
        let (ch, img, valid, ctl) = (self.ch, &*self.img, &self.valid, &self.ctl);
        let mut smom = vec![[0.0f32; 2]; w * h * c8];
        par::chunks_mut(&mut smom, w * c8, w * h >= PAR_MIN_PATCHES, |y, out| {
            if ctl.cancelled() {
                return;
            }
            for x in 0..w {
                if valid[y * w + x] {
                    patch_moments(img, w, ch, c8, side, x - r, y - r, &mut out[x * c8..(x + 1) * c8]);
                }
            }
        });
        self.smom = smom;
    }

    /// Distance between the target patch at `t` (its pixels inside the grid and not void; `k` its
    /// index in the `tbox` tables) and the source patch at `s`, per sample; with colour adaptation
    /// the source is fitted first.
    #[inline]
    fn dist(&self, t: (i32, i32), k: usize, s: (i32, i32), cutoff: f32) -> f32 {
        let full = self.full[k];
        if !self.adapt.is_none() {
            return if full { self.dist_adapted_full(k, t, s) } else { self.dist_adapted(t, s) };
        }
        if full { self.dist_full(t, s, cutoff) } else { self.dist_clipped(t, s, cutoff) }
    }

    /// [`Solver::dist`] without adaptation, of a patch that may be clipped.
    fn dist_clipped(&self, t: (i32, i32), s: (i32, i32), cutoff: f32) -> f32 {
        let (w, h, ch, r) = (self.w as i32, self.h as i32, self.ch, self.r);
        let full = ((2 * r + 1) * (2 * r + 1)) as f32 * ch as f32;
        let mut sum = 0.0f32;
        let mut n = 0usize;
        for dy in -r..=r {
            let ty = t.1 + dy;
            if ty < 0 || ty >= h {
                continue;
            }
            let sy = s.1 + dy;
            for dx in -r..=r {
                let tx = t.0 + dx;
                if tx < 0 || tx >= w || self.px[ty as usize * self.w + tx as usize] == Px::Void {
                    continue;
                }
                let ti = (ty as usize * self.w + tx as usize) * ch;
                let si = (sy as usize * self.w + (s.0 + dx) as usize) * ch;
                for c in 0..ch {
                    let d = self.img[ti + c] - self.img[si + c];
                    sum += d * d;
                }
                n += ch;
            }
            // Early out: the mean can only be ≥ sum / (full patch size).
            if sum > cutoff * full {
                return f32::INFINITY;
            }
        }
        if n == 0 { f32::INFINITY } else { sum / n as f32 }
    }

    /// [`Solver::dist_clipped`] of a full patch: the same sums, in the same order. It gives up
    /// (infinity) as soon as the sum shows the patch is worse than `cutoff`, which the sum, only
    /// growing, can show after any pixel: the caller only asks whether it is better.
    fn dist_full(&self, t: (i32, i32), s: (i32, i32), cutoff: f32) -> f32 {
        let (w, ch, r) = (self.w, self.ch, self.r);
        let side = (2 * r + 1) as usize;
        let full = ((2 * r + 1) * (2 * r + 1)) as f32 * ch as f32;
        let (span, limit) = (side * ch, cutoff * full);
        let mut sum = 0.0f32;
        for dy in -r..=r {
            let ti = ((t.1 + dy) as usize * w + (t.0 - r) as usize) * ch;
            let si = ((s.1 + dy) as usize * w + (s.0 - r) as usize) * ch;
            for (tp, sp) in self.img[ti..ti + span].chunks_exact(ch).zip(self.img[si..si + span].chunks_exact(ch)) {
                for (a, b) in tp.iter().zip(sp) {
                    let d = a - b;
                    sum += d * d;
                }
                if sum > limit {
                    return f32::INFINITY;
                }
            }
        }
        sum / (side * side * ch) as f32
    }

    /// Moments of the target patch at `t` and the source patch at `s` per channel:
    /// `[Σt, Σt², Σs, Σs², Σts]`, and the sample count.
    fn moments(&self, t: (i32, i32), s: (i32, i32), m: &mut [[f32; 5]]) -> usize {
        let (w, h, ch, r) = (self.w as i32, self.h as i32, self.ch, self.r);
        m.iter_mut().for_each(|v| *v = [0.0; 5]);
        let mut n = 0usize;
        for dy in -r..=r {
            let ty = t.1 + dy;
            if ty < 0 || ty >= h {
                continue;
            }
            let sy = s.1 + dy;
            for dx in -r..=r {
                let tx = t.0 + dx;
                if tx < 0 || tx >= w || self.px[ty as usize * self.w + tx as usize] == Px::Void {
                    continue;
                }
                let ti = (ty as usize * self.w + tx as usize) * ch;
                let si = (sy as usize * self.w + (s.0 + dx) as usize) * ch;
                for (c, mc) in m.iter_mut().enumerate().take(ch) {
                    let (a, b) = (self.img[ti + c], self.img[si + c]);
                    mc[0] += a;
                    mc[1] += a * a;
                    mc[2] += b;
                    mc[3] += b * b;
                    mc[4] += a * b;
                }
                n += 1;
            }
        }
        n
    }

    /// [`Solver::moments`] of a full target patch (`k` in the `tbox` tables): the target's own
    /// sums kept from [`Solver::costs`], the source's from [`Solver::prepare`] when kept, each
    /// summed in the same order as there.
    fn moments_full(&self, k: usize, t: (i32, i32), s: (i32, i32), m: &mut [[f32; 5]]) {
        let (w, ch, r) = (self.w, self.ch, self.r);
        let c8 = m.len();
        let side = (2 * r + 1) as usize;
        let tm = &self.tmom[k * c8..(k + 1) * c8];
        let sm = self.smom.get((s.1 as usize * w + s.0 as usize) * c8..(s.1 as usize * w + s.0 as usize + 1) * c8);
        let mut acc = [[0.0f32; 3]; MOM_CH];
        for dy in -r..=r {
            let ti = ((t.1 + dy) as usize * w + (t.0 - r) as usize) * ch;
            let si = ((s.1 + dy) as usize * w + (s.0 - r) as usize) * ch;
            let (trow, srow) = (&self.img[ti..ti + side * ch], &self.img[si..si + side * ch]);
            if sm.is_some() {
                for (tp, sp) in trow.chunks_exact(ch).zip(srow.chunks_exact(ch)) {
                    for c in 0..c8 {
                        acc[c][2] += tp[c] * sp[c];
                    }
                }
            } else {
                for (tp, sp) in trow.chunks_exact(ch).zip(srow.chunks_exact(ch)) {
                    for c in 0..c8 {
                        let (a, b) = (tp[c], sp[c]);
                        acc[c][0] += b;
                        acc[c][1] += b * b;
                        acc[c][2] += a * b;
                    }
                }
            }
        }
        for (c, mc) in m.iter_mut().enumerate() {
            let [ss, sss] = match sm {
                Some(sm) => sm[c],
                None => [acc[c][0], acc[c][1]],
            };
            *mc = [tm[c][0], tm[c][1], ss, sss, acc[c][2]];
        }
    }

    fn dist_adapted(&self, t: (i32, i32), s: (i32, i32)) -> f32 {
        let mut m = [[0.0f32; 5]; MOM_CH];
        let ch = self.ch.min(MOM_CH);
        let n = self.moments(t, s, &mut m[..ch]);
        if n == 0 {
            return f32::INFINITY;
        }
        adapted_cost(self.adapt, n as f32, &m[..ch])
    }

    fn dist_adapted_full(&self, k: usize, t: (i32, i32), s: (i32, i32)) -> f32 {
        let mut m = [[0.0f32; 5]; MOM_CH];
        let ch = self.ch.min(MOM_CH);
        self.moments_full(k, t, s, &mut m[..ch]);
        let side = (2 * self.r + 1) as usize;
        adapted_cost(self.adapt, (side * side) as f32, &m[..ch])
    }

    /// Gain and bias per channel for the match `t → s` (identity without adaptation).
    fn transform(&self, t: (i32, i32), k: usize, s: (i32, i32), out: &mut [(f32, f32)]) {
        if self.adapt.is_none() {
            out.iter_mut().for_each(|v| *v = (1.0, 0.0));
            return;
        }
        let mut m = [[0.0f32; 5]; MOM_CH];
        let ch = self.ch.min(MOM_CH);
        let n = if self.full[k] {
            self.moments_full(k, t, s, &mut m[..ch]);
            let side = (2 * self.r + 1) as usize;
            side * side
        } else {
            self.moments(t, s, &mut m[..ch])
        };
        for (c, o) in out.iter_mut().enumerate() {
            *o = if n == 0 || c >= ch { (1.0, 0.0) } else { fit(self.adapt, n as f32, m[c][0], m[c][1], m[c][2], m[c][3]) };
        }
    }

    /// One PatchMatch step at target `t` (`k` in the target box) whose match is `cur`: propagation
    /// from the matches of the neighbours visited just before it (`left` in its row, `up` in its
    /// column, in this pass's direction `step`), then random search in exponentially shrinking
    /// windows around the best (two numbers from `rng` per window).
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn pm_step(
        &self,
        t: (i32, i32),
        k: usize,
        cur: ((i32, i32), f32),
        left: Option<(i32, i32)>,
        up: Option<(i32, i32)>,
        step: i32,
        rng: &mut Rng,
        search_radius: i32,
    ) -> ((i32, i32), f32) {
        let (mut best, mut bc) = cur;
        if let Some(c) = left {
            let cand = (c.0 + step, c.1);
            if self.is_valid(cand.0, cand.1) && cand != best {
                let d = self.dist(t, k, cand, bc);
                if d < bc {
                    (best, bc) = (cand, d);
                }
            }
        }
        if let Some(c) = up {
            let cand = (c.0, c.1 + step);
            if self.is_valid(cand.0, cand.1) && cand != best {
                let d = self.dist(t, k, cand, bc);
                if d < bc {
                    (best, bc) = (cand, d);
                }
            }
        }
        let mut rad = search_radius;
        while rad >= 1 {
            let cand = (best.0 + rng.range(-rad, rad), best.1 + rng.range(-rad, rad));
            if self.is_valid(cand.0, cand.1) && cand != best {
                let d = self.dist(t, k, cand, bc);
                if d < bc {
                    (best, bc) = (cand, d);
                }
            }
            rad /= 2;
        }
        (best, bc)
    }

    /// One PatchMatch pass over all target rows, in bands of the grid's rows: each band is
    /// visited in raster order (backwards on odd passes), propagating within itself, its random
    /// numbers seeded by its index. `nnf` and `cost` are over the target box. The bands run in
    /// parallel; when there are fewer of them than threads, each is also swept in wavefronts of
    /// blocks (a cell depends only on its neighbours before it in its row and its column), which
    /// visits every cell with the same neighbours and random numbers as the raster order.
    fn patchmatch(&self, nnf: &mut [(i32, i32)], cost: &mut [f32], pass: usize, seed: u64, search_radius: i32) {
        const BAND: usize = 16;
        let tb = self.tbox;
        let tw = tb.width();
        if tw == 0 || tb.height() == 0 || nnf.len() < tb.area() || cost.len() < tb.area() {
            return;
        }
        let reverse = pass % 2 == 1;
        let step: i32 = if reverse { -1 } else { 1 };
        let band_seed = |bi: usize| seed ^ (bi as u64).wrapping_mul(0x2545_F491_4F6C_DD1D) ^ ((pass as u64) << 40);
        // The bands crossing the target box: index, first row, rows in the box.
        let mut bands = Vec::new();
        let mut y = tb.y0;
        while y < tb.y1 {
            let bi = y / BAND;
            let end = ((bi + 1) * BAND).min(tb.y1);
            bands.push((bi, y, end - y));
            y = end;
        }
        let threads = par::threads();
        if tb.area() >= PAR_MIN_PATCHES * 8 && !bands.is_empty() && bands.len() < threads {
            self.patchmatch_wavefront(nnf, cost, &bands, reverse, &band_seed, search_radius);
            return;
        }
        let band_fn = |bi: usize, ys: usize, nn: &mut [(i32, i32)], co: &mut [f32]| {
            if self.ctl.cancelled() {
                return;
            }
            let rows = nn.len() / tw;
            let mut rng = Rng::new(band_seed(bi));
            for ry in 0..rows {
                if self.ctl.cancelled() {
                    return;
                }
                let ly = if reverse { rows - 1 - ry } else { ry };
                let y = ys + ly;
                for rx in 0..tw {
                    let lx = if reverse { tw - 1 - rx } else { rx };
                    let li = ly * tw + lx;
                    let k = (y - tb.y0) * tw + lx;
                    if !self.targets[k] {
                        continue;
                    }
                    // Neighbours outside the target box hold no match: nothing to try.
                    let nx = lx as i32 - step;
                    let left = (nx >= 0 && (nx as usize) < tw).then(|| nn[ly * tw + nx as usize]);
                    let nly = ly as i32 - step;
                    let up = (nly >= 0 && (nly as usize) < rows).then(|| nn[nly as usize * tw + lx]);
                    let t = ((tb.x0 + lx) as i32, y as i32);
                    (nn[li], co[li]) = self.pm_step(t, k, (nn[li], co[li]), left, up, step, &mut rng, search_radius);
                }
            }
        };
        // The target box's rows, band by band.
        let mut parts = Vec::with_capacity(bands.len());
        let (mut nn_rest, mut co_rest) = (&mut nnf[..tb.area()], &mut cost[..tb.area()]);
        for &(bi, ys, rows) in &bands {
            let (nn, nr) = std::mem::take(&mut nn_rest).split_at_mut(rows * tw);
            let (co, cr) = std::mem::take(&mut co_rest).split_at_mut(rows * tw);
            parts.push((bi, ys, nn, co));
            (nn_rest, co_rest) = (nr, cr);
        }
        let par = tb.area() >= PAR_MIN_PATCHES && parts.len() > 1;
        par::each(parts, par, |(bi, ys, nn, co)| band_fn(bi, ys, nn, co));
    }

    /// [`Solver::patchmatch`] swept in wavefronts: the cells of each band in blocks of a row, a
    /// block once the block before it in its row and the one above it (in the pass's direction)
    /// are done. A target's random numbers are those it would draw in raster order (each target
    /// draws the same count).
    fn patchmatch_wavefront(
        &self,
        nnf: &mut [(i32, i32)],
        cost: &mut [f32],
        bands: &[(usize, usize, usize)],
        reverse: bool,
        band_seed: &(dyn Fn(usize) -> u64 + Sync),
        search_radius: i32,
    ) {
        use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};

        let tb = self.tbox;
        let tw = tb.width();
        let step: i32 = if reverse { -1 } else { 1 };
        let pack = |(x, y): (i32, i32)| (u64::from(x as u32) << 32) | u64::from(y as u32);
        let unpack = |v: u64| ((v >> 32) as u32 as i32, v as u32 as i32);
        // Random numbers each target draws.
        let mut draws = 0u64;
        let mut rad = search_radius;
        while rad >= 1 {
            draws += 2;
            rad /= 2;
        }
        // Targets before each column of each row of the box.
        let mut before = vec![0u32; tb.height() * (tw + 1)];
        for (ry, row) in before.chunks_mut(tw + 1).enumerate() {
            for lx in 0..tw {
                row[lx + 1] = row[lx] + u32::from(self.targets[ry * tw + lx]);
            }
        }
        // Targets visited before each of a band's rows (in the pass's order).
        let row_base: Vec<Vec<u64>> = bands
            .iter()
            .map(|&(_, ys, rows)| {
                let mut base = Vec::with_capacity(rows);
                let mut n = 0u64;
                for pr in 0..rows {
                    base.push(n);
                    let ly = if reverse { rows - 1 - pr } else { pr };
                    n += u64::from(before[(ys + ly - tb.y0) * (tw + 1) + tw]);
                }
                base
            })
            .collect();
        let block = (tw / 8).clamp(4, 32);
        let blocks = tw.div_ceil(block);
        let nn: Vec<AtomicU64> = nnf[..tb.area()].iter().map(|m| AtomicU64::new(pack(*m))).collect();
        let co: Vec<AtomicU32> = cost[..tb.area()].iter().map(|c| AtomicU32::new(c.to_bits())).collect();
        let sweep = |(j, pr, b): (usize, usize, usize)| {
            let (bi, ys, rows) = bands[j];
            let ly = if reverse { rows - 1 - pr } else { pr };
            let y = ys + ly;
            let ry = y - tb.y0;
            let (pc0, pc1) = (b * block, ((b + 1) * block).min(tw));
            // The targets visited before this block, in raster order.
            let first = if reverse { tw - 1 - pc0 } else { pc0 };
            let row = &before[ry * (tw + 1)..(ry + 1) * (tw + 1)];
            let in_row = if reverse { row[tw] - row[first + 1] } else { row[first] };
            let mut rng = Rng::new(band_seed(bi));
            rng.skip((row_base[j][pr] + u64::from(in_row)) * draws);
            for pc in pc0..pc1 {
                let lx = if reverse { tw - 1 - pc } else { pc };
                let k = ry * tw + lx;
                if !self.targets[k] {
                    continue;
                }
                let nx = lx as i32 - step;
                let left = (nx >= 0 && (nx as usize) < tw).then(|| unpack(nn[ry * tw + nx as usize].load(Relaxed)));
                let nly = ly as i32 - step;
                let up = (nly >= 0 && (nly as usize) < rows).then(|| unpack(nn[(ys + nly as usize - tb.y0) * tw + lx].load(Relaxed)));
                let cur = (unpack(nn[k].load(Relaxed)), f32::from_bits(co[k].load(Relaxed)));
                let t = ((tb.x0 + lx) as i32, y as i32);
                let (m, c) = self.pm_step(t, k, cur, left, up, step, &mut rng, search_radius);
                nn[k].store(pack(m), Relaxed);
                co[k].store(c.to_bits(), Relaxed);
            }
        };
        // Wavefront `d`: block `b` of (visited) row `pr` with `pr + b = d`, in every band. Each
        // wave ends before the next starts.
        let longest = bands.iter().map(|b| b.2).max().unwrap_or(0);
        for d in 0..longest + blocks - 1 {
            if self.ctl.cancelled() {
                return;
            }
            let wave: Vec<(usize, usize, usize)> = bands
                .iter()
                .enumerate()
                .flat_map(|(j, &(_, _, rows))| (0..rows).filter(move |&pr| pr <= d && d - pr < blocks).map(move |pr| (j, pr, d - pr)))
                .collect();
            par::each(wave, true, sweep);
        }
        for (m, a) in nnf.iter_mut().zip(&nn) {
            *m = unpack(a.load(Relaxed));
        }
        for (c, a) in cost.iter_mut().zip(&co) {
            *c = f32::from_bits(a.load(Relaxed));
        }
    }

    /// Recompute every target's match cost (hole pixels change between iterations), and with
    /// colour adaptation the full target patches' own moments.
    fn costs(&mut self, nnf: &[(i32, i32)], cost: &mut [f32]) -> Result<(), photocraft_raster::Cancelled> {
        let (w, ch, r, tb) = (self.w, self.ch, self.r as usize, self.tbox);
        let par = tb.area() >= PAR_MIN_PATCHES;
        self.ctl.check()?;
        if !self.adapt.is_none() {
            let c8 = ch.min(MOM_CH);
            let side = 2 * r + 1;
            let mut tmom = std::mem::take(&mut self.tmom);
            let (img, full, ctl) = (&*self.img, &self.full, &self.ctl);
            par::chunks_mut(&mut tmom, tb.width() * c8, par, |ty, out| {
                if ctl.cancelled() {
                    return;
                }
                for tx in 0..tb.width() {
                    if full[ty * tb.width() + tx] {
                        patch_moments(img, w, ch, c8, side, tb.x0 + tx - r, tb.y0 + ty - r, &mut out[tx * c8..(tx + 1) * c8]);
                    }
                }
            });
            self.tmom = tmom;
        }
        let this = &*self;
        par::chunks_mut(cost, tb.width(), par, |ty, co| {
            if this.ctl.cancelled() {
                return;
            }
            let y = tb.y0 + ty;
            for (tx, c) in co.iter_mut().enumerate() {
                let k = ty * tb.width() + tx;
                if this.targets[k] {
                    *c = this.dist(((tb.x0 + tx) as i32, y as i32), k, nnf[k], f32::INFINITY);
                }
            }
        });
        self.ctl.check()
    }

    /// Re-estimate hole pixels by a similarity-weighted vote of all overlapping patch matches
    /// (each copy adjusted by its gain and bias).
    fn vote(&mut self, nnf: &[(i32, i32)], cost: &[f32]) {
        let (w, h, ch, r) = (self.w, self.h, self.ch, self.r);
        let (tb, hb) = (self.tbox, self.hole_box);
        // A cancelled solve is thrown away.
        if self.ctl.cancelled() {
            return;
        }
        let used = |k: usize| self.targets[k] && cost[k].is_finite();
        // σ² from the 75th percentile of match costs (Wexler et al. use a robust scale like this).
        let cs: Vec<f32> = (0..tb.area()).filter(|&k| used(k)).map(|k| cost[k]).collect();
        let mut cs = cs;
        let sigma2 = if cs.is_empty() {
            1.0
        } else {
            let k = (cs.len() * 3 / 4).min(cs.len() - 1);
            let (_, v, _) = cs.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
            v.max(1e-6)
        };
        drop(cs);
        // Each target's weight and photometric transform (computed once, used by every pixel it
        // covers).
        let adapted = !self.adapt.is_none();
        let par = tb.area() >= PAR_MIN_PATCHES;
        // A weight (zero for a target that doesn't vote) and its match's centre (as a pixel
        // index) per target.
        let mut wt = vec![0.0f32; tb.area()];
        let mut src = vec![0isize; tb.area()];
        par::chunks2_mut(&mut wt, tb.width(), &mut src, tb.width(), par, |ty, wrow, srow| {
            for (tx, (o, sc)) in wrow.iter_mut().zip(srow.iter_mut()).enumerate() {
                let k = ty * tb.width() + tx;
                if used(k) {
                    *o = (-cost[k] / (2.0 * sigma2)).exp().max(1e-8);
                    *sc = nnf[k].1 as isize * w as isize + nnf[k].0 as isize;
                }
            }
        });
        let mut gb: Vec<(f32, f32)> = Vec::new();
        if adapted {
            gb = vec![(1.0f32, 0.0f32); tb.area() * ch];
            let this = &*self;
            par::chunks_mut(&mut gb, tb.width() * ch, par, |ty, out| {
                if this.ctl.cancelled() {
                    return;
                }
                let y = tb.y0 + ty;
                for tx in 0..tb.width() {
                    let k = ty * tb.width() + tx;
                    if used(k) {
                        this.transform(((tb.x0 + tx) as i32, y as i32), k, nnf[k], &mut out[tx * ch..(tx + 1) * ch]);
                    }
                }
            });
        }
        // The vote, into a buffer over the hole's bounds (every pixel reads the old pixels).
        let votes = Votes { w, h, ch, r: r as usize, tbox: tb, img: &*self.img, wt: &wt, src: &src, gb: &gb };
        let (px, ctl) = (self.px, &self.ctl);
        let mut out = vec![0.0f32; hb.area() * ch];
        let mut got = vec![false; hb.area()];
        let row = |yy: usize, orow: &mut [f32], grow: &mut [bool]| {
            if ctl.cancelled() {
                return;
            }
            let y = hb.y0 + yy;
            for x in hb.x0..hb.x1 {
                if px[y * w + x] != Px::Hole {
                    continue;
                }
                let o = (x - hb.x0) * ch;
                let dst = &mut orow[o..o + ch];
                grow[x - hb.x0] = match (ch, adapted) {
                    (1, false) => votes.pixel::<1, false>(x, y, dst),
                    (3, false) => votes.pixel::<3, false>(x, y, dst),
                    (4, false) => votes.pixel::<4, false>(x, y, dst),
                    (1, true) => votes.pixel::<1, true>(x, y, dst),
                    (3, true) => votes.pixel::<3, true>(x, y, dst),
                    (4, true) => votes.pixel::<4, true>(x, y, dst),
                    _ => votes.pixel_any(x, y, adapted, dst),
                };
            }
        };
        par::chunks2_mut(&mut out, hb.width() * ch, &mut got, hb.width(), hb.area() >= PAR_MIN_PATCHES, row);
        for yy in 0..hb.height() {
            for xx in 0..hb.width() {
                if got[yy * hb.width() + xx] {
                    let (d, s) = (((hb.y0 + yy) * w + hb.x0 + xx) * ch, (yy * hb.width() + xx) * ch);
                    self.img[d..d + ch].copy_from_slice(&out[s..s + ch]);
                }
            }
        }
    }
}

/// What [`Solver::vote`] votes with: per target (over the target box) its weight (zero when it
/// doesn't vote), its match's centre as a pixel index, and with colour adaptation its gain and
/// bias per channel.
struct Votes<'a> {
    w: usize,
    h: usize,
    ch: usize,
    r: usize,
    tbox: Bounds,
    img: &'a [f32],
    wt: &'a [f32],
    src: &'a [isize],
    gb: &'a [(f32, f32)],
}

impl Votes<'_> {
    /// Hole pixel `(x, y)`'s vote into `out` (`CH` channels): the weighted mean of what the
    /// matches of the patches covering it put there, summed patch by patch in raster order of
    /// their centres. False when no patch votes.
    #[inline(always)]
    fn pixel<const CH: usize, const ADAPT: bool>(&self, x: usize, y: usize, out: &mut [f32]) -> bool {
        let (w, r, tb) = (self.w, self.r, self.tbox);
        let mut acc = [0.0f32; CH];
        let mut wsum = 0.0f32;
        for ty in y.saturating_sub(r)..=(y + r).min(self.h - 1) {
            let dy = ty as isize - y as isize;
            let krow = (ty - tb.y0) * tb.width();
            for tx in x.saturating_sub(r)..=(x + r).min(w - 1) {
                let k = krow + tx - tb.x0;
                let wt = self.wt[k];
                if wt == 0.0 {
                    continue;
                }
                // The pixel of the match's patch that lands here.
                let si = (self.src[k] - dy * w as isize - (tx as isize - x as isize)) as usize * CH;
                let s = &self.img[si..si + CH];
                for c in 0..CH {
                    let v = if ADAPT {
                        let (g, b) = self.gb[k * CH + c];
                        g * s[c] + b
                    } else {
                        s[c]
                    };
                    acc[c] += v * wt;
                }
                wsum += wt;
            }
        }
        if wsum > 0.0 {
            for (d, a) in out.iter_mut().zip(&acc) {
                *d = a / wsum;
            }
            return true;
        }
        false
    }

    /// [`Votes::pixel`] for any number of channels.
    fn pixel_any(&self, x: usize, y: usize, adapted: bool, out: &mut [f32]) -> bool {
        let (w, r, tb, ch) = (self.w, self.r, self.tbox, self.ch);
        let mut acc = vec![0.0f32; ch];
        let mut wsum = 0.0f32;
        for ty in y.saturating_sub(r)..=(y + r).min(self.h - 1) {
            let dy = ty as isize - y as isize;
            let krow = (ty - tb.y0) * tb.width();
            for tx in x.saturating_sub(r)..=(x + r).min(w - 1) {
                let k = krow + tx - tb.x0;
                let wt = self.wt[k];
                if wt == 0.0 {
                    continue;
                }
                let si = (self.src[k] - dy * w as isize - (tx as isize - x as isize)) as usize * ch;
                for (c, a) in acc.iter_mut().enumerate() {
                    let v = if adapted {
                        let (g, b) = self.gb[k * ch + c];
                        g * self.img[si + c] + b
                    } else {
                        self.img[si + c]
                    };
                    *a += v * wt;
                }
                wsum += wt;
            }
        }
        if wsum > 0.0 {
            for (d, a) in out.iter_mut().zip(&acc) {
                *d = a / wsum;
            }
            return true;
        }
        false
    }
}

/// `Σv, Σv²` per channel (the first `out.len()`) of the `side × side` patch with top-left
/// `(x0, y0)`, summed row by row, pixel by pixel, as [`Solver::moments`] does.
#[allow(clippy::too_many_arguments)]
fn patch_moments(img: &[f32], w: usize, ch: usize, c8: usize, side: usize, x0: usize, y0: usize, out: &mut [[f32; 2]]) {
    out.iter_mut().for_each(|v| *v = [0.0; 2]);
    for y in y0..y0 + side {
        let row = &img[(y * w + x0) * ch..(y * w + x0 + side) * ch];
        for p in row.chunks_exact(ch) {
            for (c, o) in out.iter_mut().enumerate().take(c8) {
                let a = p[c];
                o[0] += a;
                o[1] += a * a;
            }
        }
    }
}

/// The adapted distance from the moments `[Σt, Σt², Σs, Σs², Σts]` of each channel over `n`
/// samples: what is left of the target after the source is fitted to it, per sample.
#[inline]
fn adapted_cost(a: Adapt, nf: f32, m: &[[f32; 5]]) -> f32 {
    let mut sum = 0.0f32;
    for mc in m {
        let (g, b) = fit(a, nf, mc[0], mc[1], mc[2], mc[3]);
        // Σ (t − g s − b)², expanded.
        sum += mc[1] + g * g * mc[3] + nf * b * b - 2.0 * g * mc[4] - 2.0 * b * mc[0] + 2.0 * g * b * mc[2];
    }
    sum.max(0.0) / (nf * m.len() as f32)
}

/// `Σ (t − s)²` over a target patch's known pixels (offsets `offs` from its centre, samples
/// `tv`, `CH` per pixel) against the source patch centred at pixel `base`, summed in that order;
/// `None` once the sum reaches `limit` (it only grows).
#[inline(always)]
fn known_sse<const CH: usize>(image: &[f32], offs: &[isize], tv: &[f32], base: isize, limit: f32) -> Option<f32> {
    let mut e = 0.0f32;
    for (o, t) in offs.iter().zip(tv.as_chunks::<CH>().0) {
        let si = (base + o) as usize * CH;
        let s = &image[si..si + CH];
        for c in 0..CH {
            let d = t[c] - s[c];
            e += d * d;
        }
        if e >= limit {
            return None;
        }
    }
    Some(e)
}

/// [`known_sse`] of [`LANES`] sources at once, each against its own limit (summed in step: the
/// sums are independent, so the processor overlaps them).
#[inline(always)]
fn known_sse_lanes<const CH: usize>(image: &[f32], offs: &[isize], tv: &[f32], bases: [isize; LANES], limits: [f32; LANES]) -> [Option<f32>; LANES] {
    let mut e = [0.0f32; LANES];
    for (o, t) in offs.iter().zip(tv.as_chunks::<CH>().0) {
        for l in 0..LANES {
            let si = (bases[l] + o) as usize * CH;
            let s = &image[si..si + CH];
            for c in 0..CH {
                let d = t[c] - s[c];
                e[l] += d * d;
            }
        }
        // Sums only grow: once all of them reached their limits, none can win.
        if (0..LANES).all(|l| e[l] >= limits[l]) {
            break;
        }
    }
    std::array::from_fn(|l| (e[l] < limits[l]).then_some(e[l]))
}

/// Sources [`known_sse_lanes`] sums at once.
const LANES: usize = 4;

/// [`known_sse`] for any number of channels.
fn known_sse_any(image: &[f32], ch: usize, offs: &[isize], tv: &[f32], base: isize, limit: f32) -> Option<f32> {
    let mut e = 0.0f32;
    for (o, t) in offs.iter().zip(tv.chunks_exact(ch.max(1))) {
        let si = (base + o) as usize * ch;
        for (tc, sc) in t.iter().zip(&image[si..si + ch]) {
            let d = tc - sc;
            e += d * d;
        }
        if e >= limit {
            return None;
        }
    }
    Some(e)
}

/// The best source so far of [`onion_peel`]'s search.
#[derive(Clone, Copy)]
struct Best {
    /// Error per sample, and the place in the sources.
    v: f32,
    j: usize,
    /// Sums of squared errors from which `sum / n` is above `v`, and at least `v`.
    worse: f32,
    equal: f32,
}

/// The least sum `e ≥ 0` with `e / n > v` (`strict`) or `e / n ≥ v` in float arithmetic (the
/// quotient only grows with `e`); infinity for an infinite `v`.
fn least_sum(v: f32, n: f32, strict: bool) -> f32 {
    let holds = |e: f32| if strict { e / n > v } else { e / n >= v };
    if !v.is_finite() || !n.is_finite() || n <= 0.0 {
        return f32::INFINITY;
    }
    let mut e = (v * n).max(0.0);
    while e > 0.0 && holds(e.next_down()) {
        e = e.next_down();
    }
    while !holds(e) && e < f32::INFINITY {
        e = e.next_up();
    }
    e
}

/// Fill the hole from its edge inwards, one ring at a time: each pixel takes the centre of the
/// source patch that best matches what is already known around it (the first in `sources` of
/// those that match best). `hb` bounds the hole.
///
/// The search is exhaustive, but most sources are given up early (#2771, #2805): the target's
/// known pixels are gathered once, the search starts from where the pixel's filled neighbours
/// took their sources (or every 16th source at the hole's edge), and a source's error stops being
/// added up once it can only lose (it only grows). So it finds the same source as a plain scan in
/// order, sooner.
#[allow(clippy::too_many_arguments)]
fn onion_peel(
    w: usize,
    h: usize,
    ch: usize,
    r: i32,
    img: &mut [f32],
    px: &[Px],
    hb: Bounds,
    sources: &[(i32, i32)],
    ctl: &photocraft_raster::Interrupt,
) -> Result<(), photocraft_raster::Cancelled> {
    const NONE: u32 = u32::MAX;
    let mut known: Vec<bool> = px.iter().map(|k| matches!(k, Px::Known | Px::Source)).collect();
    // Each source centre's place in `sources`, and the source each filled pixel took.
    let mut index = vec![NONE; w * h];
    for (j, &(x, y)) in sources.iter().enumerate() {
        index[y as usize * w + x as usize] = j as u32;
    }
    let mut took = vec![NONE; w * h];
    let neighbours = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)];
    loop {
        ctl.check()?;
        let mut front: Vec<usize> = Vec::new();
        for y in hb.y0..hb.y1 {
            for x in hb.x0..hb.x1 {
                let i = y * w + x;
                if known[i] || px[i] != Px::Hole {
                    continue;
                }
                let (x, y) = (x as i32, y as i32);
                let edge = (-1..=1).any(|dy: i32| {
                    (-1..=1).any(|dx: i32| {
                        let (nx, ny) = (x + dx, y + dy);
                        nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h && known[ny as usize * w + nx as usize]
                    })
                });
                if edge {
                    front.push(i);
                }
            }
        }
        if front.is_empty() {
            break;
        }
        let (known_now, took_now) = (&known, &took);
        let image = &*img;
        let best = |&i: &usize| -> (Vec<f32>, u32) {
            let (x, y) = ((i % w) as i32, (i / w) as i32);
            // The known pixels of the target patch, in the plain scan's order: each one's offset
            // from the patch centre (in pixels) and its samples (gathered once, #2805).
            let side = (2 * r + 1) as usize;
            let mut offs: Vec<isize> = Vec::with_capacity(side * side);
            let mut tv: Vec<f32> = Vec::with_capacity(side * side * ch);
            for dy in -r..=r {
                for dx in -r..=r {
                    let (tx, ty) = (x + dx, y + dy);
                    if tx >= 0 && ty >= 0 && (tx as usize) < w && (ty as usize) < h && known_now[ty as usize * w + tx as usize] {
                        let ti = (ty as usize * w + tx as usize) * ch;
                        offs.push(dy as isize * w as isize + dx as isize);
                        tv.extend_from_slice(&image[ti..ti + ch]);
                    }
                }
            }
            // The best so far: its error per sample, its place in `sources` (the first of equals
            // wins, as in a scan in order), and the sums of squares at which a source surely loses
            // to it (`e / n` above the best) or ties it (at least the best).
            let mut best = Best { v: f32::INFINITY, j: usize::MAX, worse: f32::INFINITY, equal: f32::INFINITY };
            if !offs.is_empty() {
                let nf = offs.len() as f32;
                let base_of = |j: usize| sources.get(j).map_or(0, |&(sx, sy)| sy as isize * w as isize + sx as isize);
                // The sum at which source `j` can't win: a tie loses to an earlier source.
                let limit_of = |j: usize, best: &Best| if j > best.j { best.equal } else { best.worse };
                // Source `j`'s whole sum `e` (it did not reach its limit): does it win?
                let adopt = |j: usize, e: f32, best: &mut Best| {
                    let v = e / nf;
                    if v < f32::INFINITY && (v < best.v || (v == best.v && j < best.j)) {
                        *best = Best { v, j, worse: least_sum(v, nf, true), equal: least_sum(v, nf, false) };
                    }
                };
                let consider = |j: usize, best: &mut Best| {
                    if j >= sources.len() {
                        return;
                    }
                    let (base, limit) = (base_of(j), limit_of(j, best));
                    let e = match ch {
                        1 => known_sse::<1>(image, &offs, &tv, base, limit),
                        3 => known_sse::<3>(image, &offs, &tv, base, limit),
                        4 => known_sse::<4>(image, &offs, &tv, base, limit),
                        _ => known_sse_any(image, ch, &offs, &tv, base, limit),
                    };
                    if let Some(e) = e {
                        adopt(j, e, best);
                    }
                };
                // A close bound early: where the filled neighbours' sources point (shifted as in
                // PatchMatch's propagation), or at the hole's edge, where none is filled yet,
                // every 16th source.
                for (dx, dy) in neighbours {
                    let (qx, qy) = (x + dx, y + dy);
                    if qx < 0 || qy < 0 || qx as usize >= w || qy as usize >= h {
                        continue;
                    }
                    let Some(&(sx, sy)) = sources.get(took_now[qy as usize * w + qx as usize] as usize) else { continue };
                    let (cx, cy) = (sx - dx, sy - dy);
                    if cx >= 0 && cy >= 0 && (cx as usize) < w && (cy as usize) < h && index[cy as usize * w + cx as usize] != NONE {
                        consider(index[cy as usize * w + cx as usize] as usize, &mut best);
                    }
                }
                if best.j == usize::MAX {
                    for j in (0..sources.len()).step_by(16) {
                        consider(j, &mut best);
                    }
                }
                // The scan in order, `LANES` sources at a time (their sums are independent, so they
                // overlap), each against its limit from the best before them: the best only gets
                // better, so a source given up would have lost to it anyway.
                let mut j = 0;
                while j + LANES <= sources.len() {
                    if j.is_multiple_of(4096) && j > 0 && ctl.cancelled() {
                        break;
                    }
                    let bases: [isize; LANES] = std::array::from_fn(|l| base_of(j + l));
                    let limits: [f32; LANES] = std::array::from_fn(|l| limit_of(j + l, &best));
                    let sums = match ch {
                        1 => known_sse_lanes::<1>(image, &offs, &tv, bases, limits),
                        3 => known_sse_lanes::<3>(image, &offs, &tv, bases, limits),
                        4 => known_sse_lanes::<4>(image, &offs, &tv, bases, limits),
                        _ => std::array::from_fn(|l| known_sse_any(image, ch, &offs, &tv, bases[l], limits[l])),
                    };
                    for (l, e) in sums.into_iter().enumerate() {
                        if let Some(e) = e {
                            adopt(j + l, e, &mut best);
                        }
                    }
                    j += LANES;
                }
                for j in j..sources.len() {
                    consider(j, &mut best);
                }
            }
            let (sx, sy) = sources.get(best.j).copied().unwrap_or((0, 0));
            let si = (sy as usize * w + sx as usize) * ch;
            (image[si..si + ch].to_vec(), if best.j == usize::MAX { NONE } else { best.j as u32 })
        };
        let vals: Vec<(Vec<f32>, u32)> = par::map(front.len(), front.len() * sources.len() >= 4096, |k| best(&front[k]));
        ctl.check()?;
        for (&i, (v, j)) in front.iter().zip(vals) {
            img[i * ch..(i + 1) * ch].copy_from_slice(&v);
            known[i] = true;
            took[i] = j;
        }
    }
    Ok(())
}

/// Content-aware completion of `hole` (see module docs), copying from any pixel outside it. Returns
/// `None` when there is no fully-known patch to copy from (the caller can fall back to
/// [`membrane_fill`]).
pub fn complete(w: usize, h: usize, ch: usize, img: &[f32], hole: &[bool], p: &CompleteParams) -> Option<Vec<f32>> {
    // Never cancelled, so never `Err`.
    complete_with(w, h, ch, img, hole, p, &photocraft_raster::Interrupt::NONE).unwrap_or(None)
}

/// [`complete`] that can be cancelled (checked per EM iteration, PatchMatch pass and row band)
/// and reports progress (by pyramid level). `Err(Cancelled)` when `ctl` was cancelled.
pub fn complete_with(
    w: usize,
    h: usize,
    ch: usize,
    img: &[f32],
    hole: &[bool],
    p: &CompleteParams,
    ctl: &photocraft_raster::Interrupt,
) -> Result<Option<Vec<f32>>, photocraft_raster::Cancelled> {
    assert_eq!(hole.len(), w * h);
    let px: Vec<Px> = hole.iter().map(|h| if *h { Px::Hole } else { Px::Source }).collect();
    complete_masked(w, h, ch, img, &px, p, ctl)
}

/// [`complete_with`] with every pixel's role given ([`Px`]): the hole, the context, what may be
/// copied, and what isn't image at all.
pub fn complete_masked(
    w: usize,
    h: usize,
    ch: usize,
    img: &[f32],
    px: &[Px],
    p: &CompleteParams,
    ctl: &photocraft_raster::Interrupt,
) -> Result<Option<Vec<f32>>, photocraft_raster::Cancelled> {
    Ok(complete_cow(w, h, ch, Cow::Borrowed(img), px, p, ctl)?.ok())
}

/// [`complete_masked`] that fills `img` itself instead of a copy of it (only the hole changes):
/// `Err(img)`, untouched, when there is nothing to copy from.
pub fn complete_masked_owned(
    w: usize,
    h: usize,
    ch: usize,
    img: Vec<f32>,
    px: &[Px],
    p: &CompleteParams,
    ctl: &photocraft_raster::Interrupt,
) -> Result<Result<Vec<f32>, Vec<f32>>, photocraft_raster::Cancelled> {
    Ok(complete_cow(w, h, ch, Cow::Owned(img), px, p, ctl)?.map_err(Cow::into_owned))
}

/// The completion, on the caller's pixels or a copy of them: `Err` gives them back when there is
/// nothing to copy from.
fn complete_cow<'a>(
    w: usize,
    h: usize,
    ch: usize,
    img: Cow<'a, [f32]>,
    px: &'a [Px],
    p: &CompleteParams,
    ctl: &photocraft_raster::Interrupt,
) -> Result<Result<Vec<f32>, Cow<'a, [f32]>>, photocraft_raster::Cancelled> {
    assert_eq!(img.len(), w * h * ch);
    assert_eq!(px.len(), w * h);
    let Some(hb0) = hole_bounds(w, h, px) else { return Ok(Ok(img.into_owned())) };
    if ch == 0 {
        return Ok(Ok(Vec::new()));
    }
    let r = p.patch_radius.max(1);
    let psz = 2 * r + 1;
    // Pyramid: shrink until the hole is about four patches across (a deeper pyramid decides the
    // structure where the texture has averaged out, and fills flat).
    let mut levels = vec![Level { w, h, img, px: Cow::Borrowed(px) }];
    let mut boxes = vec![hb0];
    let mut ext = hb0.width().max(hb0.height());
    let mut exts = vec![ext];
    while let Some(l) = levels.last() {
        ctl.check()?;
        if ext <= 4 * psz || l.w / 2 < 3 * psz || l.h / 2 < 3 * psz {
            break;
        }
        let d = downsample(l, ch);
        levels.push(d);
        boxes.push(boxes[boxes.len() - 1].half());
        ext = ext.div_ceil(2);
        exts.push(ext);
    }
    let nlev = levels.len();
    // The coarser level's result.
    let mut prev: Option<Coarser> = None;
    let total_px: usize = levels.iter().map(|l| l.w * l.h).sum();
    let mut done_px = 0usize;
    let ri = r as i32;
    for li in (0..nlev).rev() {
        ctl.check()?;
        let Level { w: lw, h: lh, img: limg, px: lpx } = std::mem::replace(&mut levels[li], Level::EMPTY);
        let hb = boxes[li];
        let tb = hb.grow(r, lw, lh);
        let valid = valid_sources(lw, lh, r, &lpx, ctl).ok_or(photocraft_raster::Cancelled)?;
        let sources = Sources::new(&valid, lw);
        // Nothing to copy from at the finest level: the caller falls back (and gets its pixels
        // back as they were).
        if li == 0 && sources.len() == 0 {
            return Ok(Err(limg));
        }
        // The level's own pixels to fill (at level 0 the caller's, or a copy made in parallel).
        let mut limg = match limg {
            Cow::Owned(v) => v,
            Cow::Borrowed(b) => {
                const CHUNK: usize = 1 << 16;
                let mut v = vec![0.0f32; b.len()];
                par::chunks_mut(&mut v, CHUNK, true, |i, d| d.copy_from_slice(&b[i * CHUNK..i * CHUNK + d.len()]));
                v
            }
        };
        let targets = targets_around(lw, r, &lpx, hb, tb);
        ctl.check()?;
        // Initialise the hole: at the coarsest level peeled from its edge inwards (or a membrane
        // where nothing can be copied), else the coarser level's pixels, refined below by a vote of
        // its scaled-up matches.
        let coarser = prev.take();
        match &coarser {
            Some(c) => {
                let (pw, ph) = (c.w, c.h);
                for y in hb.y0..hb.y1 {
                    for x in hb.x0..hb.x1 {
                        if lpx[y * lw + x] == Px::Hole {
                            let (sx, sy) = ((x / 2).min(pw - 1), (y / 2).min(ph - 1));
                            let (d, s) = ((y * lw + x) * ch, (sy * pw + sx) * ch);
                            limg[d..d + ch].copy_from_slice(&c.img[s..s + ch]);
                        }
                    }
                }
            }
            None if sources.len() > 0 => onion_peel(lw, lh, ch, ri, &mut limg, &lpx, hb, &sources.all(&valid, lw), ctl)?,
            None => {
                let unknown: Vec<bool> = lpx.iter().map(|k| matches!(k, Px::Hole | Px::Void)).collect();
                let m = membrane_fill(lw, lh, ch, &limg, &unknown);
                for (i, k) in lpx.iter().enumerate() {
                    if *k == Px::Hole {
                        limg[i * ch..(i + 1) * ch].copy_from_slice(&m[i * ch..(i + 1) * ch]);
                    }
                }
            }
        }
        if sources.len() == 0 {
            prev = Some(Coarser { w: lw, h: lh, img: limg, field: Vec::new(), tbox: tb });
            continue;
        }
        let refine_only = li + 1 < nlev && exts[li] > p.refine_extent;
        let n_targets = targets.iter().filter(|t| **t).count();
        // Source moments are worth keeping when there are few sources for the matches tried.
        let keep_source_moments = lw * lh <= SOURCE_MOMENTS_MAX && sources.len() <= n_targets.saturating_mul(if refine_only { 4 } else { 64 });
        let mut solver = Solver {
            w: lw,
            h: lh,
            ch,
            r: ri,
            img: &mut limg,
            px: &lpx,
            valid,
            targets,
            hole_box: hb,
            tbox: tb,
            full: Vec::new(),
            tmom: Vec::new(),
            smom: Vec::new(),
            adapt: p.adapt,
            ctl: *ctl,
        };
        solver.prepare(keep_source_moments);
        ctl.check()?;
        // NNF init: the coarser level's matches scaled up where they land on a valid source,
        // random otherwise.
        let mut rng = Rng::new(p.seed ^ (li as u64) << 20);
        let tw = tb.width();
        let mut nnf = vec![(0i32, 0i32); tb.area()];
        let mut upsampled = false;
        for y in tb.y0..tb.y1 {
            if y % 32 == 0 {
                ctl.check()?;
            }
            for x in tb.x0..tb.x1 {
                let k = (y - tb.y0) * tw + (x - tb.x0);
                if !solver.targets[k] {
                    continue;
                }
                let mut cand = None;
                if let Some(c) = &coarser
                    && !c.field.is_empty()
                {
                    let (cx, cy) = ((x / 2).min(c.w - 1), (y / 2).min(c.h - 1));
                    let (sx, sy) = c.field_at(cx, cy);
                    if sx != i32::MIN {
                        // The same displacement, twice as long.
                        let c = (x as i32 + 2 * (sx - cx as i32), y as i32 + 2 * (sy - cy as i32));
                        if solver.is_valid(c.0, c.1) {
                            cand = Some(c);
                        }
                    }
                }
                upsampled |= cand.is_some();
                nnf[k] = cand.unwrap_or_else(|| sources.nth((rng.next() % sources.len() as u64) as usize, &solver.valid, lw));
            }
        }
        // The coarser level has given all it had.
        drop(coarser);
        let mut cost = vec![f32::INFINITY; tb.area()];
        // A finer level starts from the coarse matches voted with its own (sharp) pixels.
        if upsampled {
            solver.costs(&nnf, &mut cost)?;
            solver.vote(&nnf, &cost);
        }
        let em = if li + 1 == nlev {
            p.em_iters
        } else if refine_only {
            1
        } else {
            (p.em_iters >> (nlev - 1 - li)).max(p.min_em_iters)
        };
        let pm_iters = if refine_only { 1 } else { p.pm_iters };
        let full_search = (lw.max(lh)) as i32;
        for it in 0..em {
            // Costs change as hole pixels are re-estimated: refresh before each PatchMatch.
            solver.costs(&nnf, &mut cost)?;
            // Wide random search on the coarsest level and on each level's first iteration,
            // local refinement afterwards.
            let radius = if refine_only {
                2
            } else if li + 1 == nlev || it == 0 {
                full_search
            } else {
                (4 * psz as i32).min(full_search)
            };
            for pass in 0..pm_iters {
                ctl.check()?;
                solver.patchmatch(&mut nnf, &mut cost, pass + it * pm_iters, p.seed.wrapping_add((li * 1000 + it) as u64), radius);
            }
            ctl.check()?;
            solver.vote(&nnf, &cost);
            ctl.check()?;
        }
        // A vote that saw the cancellation stopped part-way: never hand its pixels on.
        ctl.check()?;
        // Keep the match field (unmatched cells marked) for the next level (none after the last).
        let field: Vec<(i32, i32)> =
            if li == 0 { Vec::new() } else { nnf.iter().zip(&solver.targets).map(|(n, t)| if *t { *n } else { (i32::MIN, i32::MIN) }).collect() };
        // Finer levels cost ~4× the previous one: weight progress by pixel count.
        done_px += lw * lh;
        ctl.progress(done_px as f32 / total_px.max(1) as f32);
        prev = Some(Coarser { w: lw, h: lh, img: limg, field, tbox: tb });
    }
    Ok(prev.map(|c| c.img).ok_or(Cow::Owned(Vec::new())))
}

/// Proximity match: the displacement `(dx, dy)` (within `max_radius`) whose surroundings best match the
/// ring of `ring` pixels around the hole, with the displaced hole fully inside the grid and outside the
/// hole. `None` if no displacement fits.
pub fn best_offset(w: usize, h: usize, ch: usize, img: &[f32], hole: &[bool], ring: usize, max_radius: i32) -> Option<(i32, i32)> {
    let (bx0, by0, bx1, by1) = hole_bbox(w, h, hole)?;
    // No valid translation can move farther than the source image's extent.
    // Bound arbitrary public API inputs before constructing i32 search ranges:
    // usize -> i32 truncation and signed increments otherwise overflow or hang.
    let extent = w.max(h).min((i32::MAX / 4) as usize) as i32;
    let max_radius = max_radius.max(0).min(extent);
    if max_radius == 0 {
        return None;
    }
    let ring = ring.max(1).min(extent as usize) as i32;
    // Every set cell lies in the hole bounds. Counts outside that rectangle are zero,
    // so source rejection needs a table for the hole, not for the whole photo.
    let hs = HoleIntegral::new(w, hole, (bx0, by0, bx1, by1))?;
    // Sample points: the hole's ring (dilated minus hole), subsampled for speed.
    let mut pts = Vec::new();
    let (x0, y0) = (bx0 as i32 - ring, by0 as i32 - ring);
    let (x1, y1) = (bx1 as i32 + ring, by1 as i32 + ring);
    for y in y0.max(0)..y1.min(h as i32) {
        for x in x0.max(0)..x1.min(w as i32) {
            if hole[y as usize * w + x as usize] {
                continue;
            }
            if hs.count(x - ring, y - ring, x + ring + 1, y + ring + 1) > 0 {
                pts.push((x, y));
            }
        }
    }
    let stride = (pts.len() / 2000).max(1);
    let pts: Vec<(i32, i32)> = pts.into_iter().step_by(stride).collect();
    let eval = |dx: i32, dy: i32, cutoff: f32| -> Option<f32> {
        let (hx0, hy0, hx1, hy1) = (x0 + dx, y0 + dy, x1 + dx, y1 + dy);
        let inside = hx0 >= 0 && hy0 >= 0 && hx1 <= w as i32 && hy1 <= h as i32;
        if (dx, dy) == (0, 0) || !inside || hs.count(hx0, hy0, hx1, hy1) != 0 {
            return None;
        }
        let mut e = 0.0f32;
        let n = pts.len().max(1) as f32;
        let penalty = 1.0 + 1e-3 * ((dx as f32).hypot(dy as f32) / max_radius as f32);
        for (i, &(x, y)) in pts.iter().enumerate() {
            let (a, b) = ((y as usize * w + x as usize) * ch, ((y + dy) as usize * w + (x + dx) as usize) * ch);
            for c in 0..ch {
                let d = img[a + c] - img[b + c];
                e += d * d;
            }
            // SSD terms are nonnegative. The partial score is a lower bound on the
            // final score; use the same division and penalty to keep float rounding
            // and strict tie ordering unchanged. Check in batches to keep this cheap.
            if i % 32 == 31 && cutoff.is_finite() && e / n * penalty >= cutoff {
                return None;
            }
        }
        // Mild preference for nearer sources on ties.
        Some(e / n * penalty)
    };
    // Coarse grid over the search window, then a full-resolution refinement around the winner.
    let step = (max_radius / 24).max(1);
    let mut best: Option<((i32, i32), f32)> = None;
    let consider = |dx: i32, dy: i32, best: &mut Option<((i32, i32), f32)>| {
        if let Some(e) = eval(dx, dy, best.map_or(f32::INFINITY, |(_, error)| error))
            && best.is_none_or(|(_, be)| e < be)
        {
            *best = Some(((dx, dy), e));
        }
    };
    let mut dy = -max_radius;
    while dy <= max_radius {
        let mut dx = -max_radius;
        while dx <= max_radius {
            consider(dx, dy, &mut best);
            dx += step;
        }
        dy += step;
    }
    if step > 1
        && let Some(((bx, by), _)) = best
    {
        for dy in (by - step)..=(by + step) {
            for dx in (bx - step)..=(bx + step) {
                consider(dx, dy, &mut best);
            }
        }
    }
    best.map(|b| b.0)
}

/// Create texture: fill the hole with blocks of surrounding texture (see module docs).
pub fn synthesize(w: usize, h: usize, ch: usize, img: &[f32], hole: &[bool], block: usize, seed: u64) -> Vec<f32> {
    let mut out = img.to_vec();
    let Some((bx0, by0, bx1, by1)) = hole_bbox(w, h, hole) else { return out };
    let b = block.max(4) as i32;
    let ov = (b / 3).max(1);
    let hs = integral(w, h, hole);
    // Candidate block origins: whole block (with overlap) inside the grid and outside the hole.
    let mut cands = Vec::new();
    for y in ov..(h as i32 - b - ov) {
        for x in ov..(w as i32 - b - ov) {
            if window_count(&hs, w, h, x - ov, y - ov, x + b + ov, y + b + ov) == 0 {
                cands.push((x, y));
            }
        }
    }
    if cands.is_empty() {
        return membrane_fill(w, h, ch, img, hole);
    }
    let mut filled: Vec<bool> = hole.iter().map(|x| !x).collect();
    let mut rng = Rng::new(seed);
    let mut by = by0 as i32;
    while by < by1 as i32 {
        let mut bx = bx0 as i32;
        while bx < bx1 as i32 {
            if window_count(&hs, w, h, bx, by, bx + b, by + b) > 0 {
                // Pick the candidate whose block best agrees with already-known pixels around/inside.
                let mut best = (cands[0], f32::INFINITY);
                for _ in 0..48 {
                    let c = cands[(rng.next() % cands.len() as u64) as usize];
                    let (mut e, mut n) = (0.0f32, 0u32);
                    for y in (by - ov)..(by + b + ov) {
                        for x in (bx - ov)..(bx + b + ov) {
                            if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 || !filled[y as usize * w + x as usize] {
                                continue;
                            }
                            let (a, s) = ((y as usize * w + x as usize) * ch, ((y - by + c.1) as usize * w + (x - bx + c.0) as usize) * ch);
                            for k in 0..ch {
                                let d = out[a + k] - img[s + k];
                                e += d * d;
                            }
                            n += 1;
                        }
                    }
                    let e = if n == 0 { 0.0 } else { e / n as f32 };
                    if e < best.1 {
                        best = (c, e);
                    }
                }
                let c = best.0;
                let (x0, y0) = ((bx - ov).max(0), (by - ov).max(0));
                let (x1, y1) = ((bx + b + ov).min(w as i32), (by + b + ov).min(h as i32));
                let src = |x: i32, y: i32| ((y - by + c.1) as usize * w + (x - bx + c.0) as usize) * ch;
                // Squared difference between what is there and the block, where both exist.
                let err = |x: i32, y: i32| {
                    let i = y as usize * w + x as usize;
                    if !(hole[i] && filled[i]) {
                        return 0.0;
                    }
                    let (a, s) = (i * ch, src(x, y));
                    (0..ch).map(|q| (out[a + q] - img[s + q]).powi(2)).sum::<f32>()
                };
                // Minimum-error boundary cuts through the overlaps with the blocks to the left and
                // above: the block is copied right of and below them, untouched pixels are kept
                // on the other side. Copying (rather than averaging) keeps the texture's grain.
                let left = min_cut((x0..(bx + ov).min(x1)).len(), (y0..y1).len(), |u, v| err(x0 + u as i32, y0 + v as i32));
                let top = min_cut((y0..(by + ov).min(y1)).len(), (x0..x1).len(), |u, v| err(x0 + v as i32, y0 + u as i32));
                for y in y0..y1 {
                    for x in x0..x1 {
                        let i = y as usize * w + x as usize;
                        if !hole[i] {
                            continue;
                        }
                        let (u, v) = ((x - x0) as usize, (y - y0) as usize);
                        let keep = filled[i] && (left.get(v).is_some_and(|s| u < *s) || top.get(u).is_some_and(|s| v < *s));
                        if !keep {
                            let (a, s) = (i * ch, src(x, y));
                            out[a..a + ch].copy_from_slice(&img[s..s + ch]);
                        }
                        filled[i] = true;
                    }
                }
            }
            bx += b;
        }
        by += b;
    }
    out
}

/// The minimum-error cut through a band `across` wide and `along` long (Efros & Freeman's
/// boundary cut, by dynamic programming): for each step along the band, the offset across it
/// where the cut runs (`err(across, along)` is the cost of a pixel). Pixels before the cut keep
/// what is there. Empty when there is no cost anywhere (nothing to cut through).
fn min_cut(across: usize, along: usize, err: impl Fn(usize, usize) -> f32) -> Vec<usize> {
    if across == 0 || along == 0 {
        return Vec::new();
    }
    // Cumulative cost, step by step along the band; the cut may move by one pixel per step.
    let mut acc: Vec<f32> = (0..along).flat_map(|v| (0..across).map(move |u| (u, v))).map(|(u, v)| err(u, v)).collect();
    if acc.iter().all(|x| *x == 0.0) {
        return Vec::new();
    }
    for v in 1..along {
        for u in 0..across {
            let lo = u.saturating_sub(1);
            let hi = (u + 1).min(across - 1);
            let m = (lo..=hi).map(|k| acc[(v - 1) * across + k]).fold(f32::INFINITY, f32::min);
            acc[v * across + u] += m;
        }
    }
    let argmin = |v: usize, lo: usize, hi: usize| (lo..=hi).min_by(|a, b| acc[v * across + a].total_cmp(&acc[v * across + b])).unwrap_or(lo);
    let mut cut = vec![0; along];
    let mut u = argmin(along - 1, 0, across - 1);
    for v in (0..along).rev() {
        if v + 1 < along {
            u = argmin(v, u.saturating_sub(1), (u + 1).min(across - 1));
        }
        cut[v] = u;
    }
    cut
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mean is matched first: on a nearly flat bright patch with a noisy deviation ratio the
    /// gain stays where the bias can still reach the target's mean.
    #[test]
    fn fit_matches_the_mean_before_the_contrast() {
        let a = Adapt { gain: (0.9, 1.1), bias: 0.04 };
        // Target mean 0.70, source mean 0.70; deviations 0.01 vs 0.004 (ratio 2.5).
        let n = 25.0;
        let (mt, ms, sdt, sds) = (0.70f32, 0.70f32, 0.01f32, 0.004f32);
        let (g, b) = fit(a, n, mt * n, (sdt * sdt + mt * mt) * n, ms * n, (sds * sds + ms * ms) * n);
        assert!((g * ms + b - mt).abs() < 1e-5, "mean {} vs {mt} (g {g}, b {b})", g * ms + b);
        // Where the mean can't be reached anyway, the limits hold as before.
        let (g, b) = fit(a, n, 0.2 * n, 0.04 * n, 0.9 * n, 0.81 * n);
        assert!((a.gain.0..=a.gain.1).contains(&g) && b.abs() <= a.bias + 1e-6);
    }

    /// Smooth gradient plus deterministic fine noise.
    fn texture(w: usize, h: usize) -> Vec<f32> {
        let mut rng = Rng::new(7);
        (0..w * h)
            .flat_map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                let n = (rng.next() % 1000) as f32 / 1000.0 * 0.04 - 0.02;
                [0.3 + 0.004 * x + n, 0.5 + 0.002 * y + n, 0.4 + n]
            })
            .collect()
    }

    fn disc(w: usize, h: usize, cx: f32, cy: f32, r: f32) -> Vec<bool> {
        (0..w * h).map(|i| ((i % w) as f32 - cx).hypot((i / w) as f32 - cy) < r).collect()
    }

    fn hole_rmse(a: &[f32], b: &[f32], hole: &[bool], ch: usize) -> f32 {
        let (mut e, mut n) = (0.0f32, 0);
        for (i, hh) in hole.iter().enumerate() {
            if *hh {
                for c in 0..ch {
                    e += (a[i * ch + c] - b[i * ch + c]).powi(2);
                    n += 1;
                }
            }
        }
        (e / n as f32).sqrt()
    }

    #[test]
    fn completion_fills_blemish_from_texture() {
        let (w, h, ch) = (96, 96, 3);
        let clean = texture(w, h);
        let hole = disc(w, h, 48.0, 48.0, 9.0);
        let mut dirty = clean.clone();
        for (i, hh) in hole.iter().enumerate() {
            if *hh {
                dirty[i * ch..i * ch + ch].copy_from_slice(&[0.02, 0.02, 0.02]);
            }
        }
        let out = complete(w, h, ch, &dirty, &hole, &CompleteParams::default()).unwrap();
        let before = hole_rmse(&dirty, &clean, &hole, ch);
        let after = hole_rmse(&out, &clean, &hole, ch);
        assert!(after < 0.06 && after < before * 0.15, "rmse before {before} after {after}");
        // Outside the hole untouched.
        assert!(out.iter().zip(&dirty).zip(hole.iter().flat_map(|h| [*h; 3])).all(|((a, b), h)| h || a == b));
        // Deterministic.
        assert_eq!(out, complete(w, h, ch, &dirty, &hole, &CompleteParams::default()).unwrap());
    }

    #[test]
    fn completion_continues_structure() {
        // Vertical stripes: the filled hole should keep the stripes (not a blur).
        let (w, h, ch) = (80, 80, 1);
        let clean: Vec<f32> = (0..w * h).map(|i| if (i % w / 4) % 2 == 0 { 0.2 } else { 0.8 }).collect();
        let hole = disc(w, h, 40.0, 40.0, 7.0);
        let dirty: Vec<f32> = clean.iter().zip(&hole).map(|(c, h)| if *h { 0.5 } else { *c }).collect();
        let out = complete(w, h, ch, &dirty, &hole, &CompleteParams::default()).unwrap();
        let err = hole_rmse(&out, &clean, &hole, ch);
        assert!(err < 0.15, "{err}");
    }

    #[test]
    fn proximity_finds_matching_offset() {
        let (w, h, ch) = (80, 60, 1);
        // Periodic pattern with period 10 in x: a match is any multiple-of-10 displacement.
        let img: Vec<f32> = (0..w * h).map(|i| ((i % w) as f32 * std::f32::consts::TAU / 10.0).sin() * 0.4 + 0.5).collect();
        let hole = disc(w, h, 40.0, 30.0, 5.0);
        let (dx, dy) = best_offset(w, h, ch, &img, &hole, 3, 20).unwrap();
        assert_eq!(dx.rem_euclid(10), 0, "dx {dx} dy {dy}");
        assert!((dx, dy) != (0, 0));
    }

    #[test]
    fn proximity_search_clamps_hostile_radii_without_overflow_or_hanging() {
        let (w, h, ch) = (24, 20, 1);
        let img: Vec<f32> = (0..w * h).map(|i| (i % w) as f32 / w as f32).collect();
        let hole = disc(w, h, 12.0, 10.0, 2.0);
        // User-supplied values used to overflow i32 loops, cast a huge usize
        // ring into a negative number, or take billions of search iterations.
        let result = best_offset(w, h, ch, &img, &hole, usize::MAX, i32::MAX);
        if let Some((dx, dy)) = result {
            assert!(dx.abs() <= w as i32 && dy.abs() <= h as i32);
        }
        assert_eq!(best_offset(w, h, ch, &img, &hole, 2, -1), None);
        assert_eq!(best_offset(w, h, ch, &img, &hole, 2, 0), None);
    }

    /// Structure, fine noise and flat patches (so that some matches tie exactly), `ch` channels.
    fn busy(w: usize, h: usize, ch: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        (0..w * h)
            .flat_map(|i| {
                let (x, y) = (i % w, i / w);
                let flat = (x / 9 + y / 7) % 5 == 0;
                let n = (rng.next() % 1000) as f32 / 1000.0;
                (0..ch)
                    .map(move |c| if flat { 0.25 + 0.1 * c as f32 } else { (0.2 + 0.004 * x as f32 + 0.1 * ((x + 2 * y + c) % 6) as f32 + 0.05 * n).min(1.0) })
            })
            .collect()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// Roles: a hole, a band that is context only, a corner that is no image.
    fn roles(w: usize, h: usize, hole: &[bool]) -> Vec<Px> {
        (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                if hole[i] {
                    Px::Hole
                } else if x < 4 && y < 6 {
                    Px::Void
                } else if (w * 3 / 4..w * 3 / 4 + 3).contains(&x) {
                    Px::Known
                } else {
                    Px::Source
                }
            })
            .collect()
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn results_do_not_depend_on_the_thread_count() {
        // Big enough for the parallel passes, and for PatchMatch to sweep in wavefronts where
        // there are fewer bands than threads (never with one thread).
        let (w, h) = (300, 260);
        for (ch, adapt) in [(3, Adapt::NONE), (4, Adapt { gain: (0.9, 1.1), bias: 0.04 })] {
            let img = busy(w, h, ch, 3);
            let hole = disc(w, h, 140.0, 130.0, 40.0);
            let px = roles(w, h, &hole);
            let p = CompleteParams { adapt, ..Default::default() };
            let run = |threads: usize| {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
                bits(&pool.install(|| complete_masked(w, h, ch, &img, &px, &p, &photocraft_raster::Interrupt::NONE)).unwrap().unwrap())
            };
            let one = run(1);
            assert_eq!(one, run(3), "{ch} channels, 3 threads");
            assert_eq!(one, run(16), "{ch} channels, 16 threads");
        }
    }

    /// The onion peel as first written: every source in order, every known pixel summed.
    fn onion_peel_plain(w: usize, h: usize, ch: usize, r: i32, img: &mut [f32], px: &[Px], sources: &[(i32, i32)]) {
        let mut known: Vec<bool> = px.iter().map(|k| matches!(k, Px::Known | Px::Source)).collect();
        loop {
            let front: Vec<usize> = (0..w * h)
                .filter(|&i| {
                    let (x, y) = ((i % w) as i32, (i / w) as i32);
                    !known[i]
                        && px[i] == Px::Hole
                        && (-1..=1).any(|dy: i32| {
                            (-1..=1).any(|dx: i32| {
                                let (nx, ny) = (x + dx, y + dy);
                                nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h && known[ny as usize * w + nx as usize]
                            })
                        })
                })
                .collect();
            if front.is_empty() {
                break;
            }
            let vals: Vec<Vec<f32>> = front
                .iter()
                .map(|&i| {
                    let (x, y) = ((i % w) as i32, (i / w) as i32);
                    let mut best = (f32::INFINITY, (0i32, 0i32));
                    for &(sx, sy) in sources {
                        let (mut e, mut n) = (0.0f32, 0usize);
                        for dy in -r..=r {
                            for dx in -r..=r {
                                let (tx, ty) = (x + dx, y + dy);
                                if tx < 0 || ty < 0 || tx as usize >= w || ty as usize >= h || !known[ty as usize * w + tx as usize] {
                                    continue;
                                }
                                let (ti, si) = ((ty as usize * w + tx as usize) * ch, ((sy + dy) as usize * w + (sx + dx) as usize) * ch);
                                for c in 0..ch {
                                    let d = img[ti + c] - img[si + c];
                                    e += d * d;
                                }
                                n += 1;
                            }
                        }
                        if n > 0 && e / (n as f32) < best.0 {
                            best = (e / n as f32, (sx, sy));
                        }
                    }
                    let si = (best.1.1 as usize * w + best.1.0 as usize) * ch;
                    img[si..si + ch].to_vec()
                })
                .collect();
            for (&i, v) in front.iter().zip(vals) {
                img[i * ch..(i + 1) * ch].copy_from_slice(&v);
                known[i] = true;
            }
        }
    }

    /// #2805's case (upstream's own speed-up of the onion peel, superseded here): repeating
    /// texture with many exact ties, two holes; the result must be the plain scan's to the bit.
    #[test]
    fn onion_peel_picks_exactly_what_the_plain_scan_picks() {
        let (w, h, ch, r) = (48, 40, 3, 3);
        let mut img = texture(w, h);
        for y in 0..h {
            for x in 6..w / 2 {
                for c in 0..ch {
                    img[(y * w + x) * ch + c] = img[(y * w + x % 6) * ch + c];
                }
            }
        }
        let none = photocraft_raster::Interrupt::NONE;
        for hole in [disc(w, h, 20.0, 18.0, 7.5), disc(w, h, 34.0, 22.0, 5.0)] {
            let px: Vec<Px> = hole.iter().map(|h| if *h { Px::Hole } else { Px::Source }).collect();
            let near = integral(w, h, &hole);
            let sources: Vec<(i32, i32)> = (r..h as i32 - r)
                .flat_map(|y| (r..w as i32 - r).map(move |x| (x, y)))
                .filter(|&(x, y)| window_count(&near, w, h, x - r, y - r, x + r + 1, y + r + 1) == 0)
                .collect();
            let (mut fast, mut plain) = (img.clone(), img.clone());
            onion_peel(w, h, ch, r, &mut fast, &px, hole_bounds(w, h, &px).unwrap(), &sources, &none).unwrap();
            onion_peel_plain(w, h, ch, r, &mut plain, &px, &sources);
            assert!(fast.iter().zip(&plain).all(|(a, b)| a.to_bits() == b.to_bits()));
            assert_ne!(fast, img);
        }
    }

    #[test]
    fn onion_peel_takes_the_plain_scans_sources() {
        let none = photocraft_raster::Interrupt::NONE;
        for (w, h, ch, seed) in [(44, 36, 1, 1), (50, 40, 3, 2), (40, 40, 4, 3)] {
            let mut img = busy(w, h, ch, seed);
            // Also an image of a few exact values, where many sources tie.
            if seed == 3 {
                img.iter_mut().for_each(|v| *v = (*v * 3.0).round() / 3.0);
            }
            let hole = disc(w, h, w as f32 * 0.45, h as f32 * 0.5, 7.5);
            let px = roles(w, h, &hole);
            let valid = valid_sources(w, h, 3, &px, &none).unwrap();
            let sources = Sources::new(&valid, w).all(&valid, w);
            let (mut a, mut b) = (img.clone(), img.clone());
            onion_peel(w, h, ch, 3, &mut a, &px, hole_bounds(w, h, &px).unwrap(), &sources, &none).unwrap();
            onion_peel_plain(w, h, ch, 3, &mut b, &px, &sources);
            assert_eq!(bits(&a), bits(&b), "{w}×{h}×{ch}");
        }
    }

    #[test]
    fn least_sum_is_where_the_quotient_crosses() {
        let mut rng = Rng::new(5);
        for i in 0..4000 {
            let n = (1 + rng.next() % 49) as f32;
            let v = match i % 4 {
                0 => 0.0,
                1 => (rng.next() % 1000) as f32 * 1e-9,
                _ => (rng.next() % 1_000_000) as f32 / 1_000_000.0,
            };
            for strict in [true, false] {
                let holds = |e: f32| if strict { e / n > v } else { e / n >= v };
                let e = least_sum(v, n, strict);
                assert!(holds(e), "{v} {n} {strict}: {e}");
                assert!(e == 0.0 || !holds(e.next_down()), "{v} {n} {strict}: {e} is not the least");
            }
        }
        assert_eq!(least_sum(f32::INFINITY, 4.0, true), f32::INFINITY);
    }

    #[test]
    fn valid_sources_are_the_wholly_copyable_patches() {
        let (w, h, r) = (37, 29, 2);
        let mut rng = Rng::new(9);
        let px: Vec<Px> = (0..w * h)
            .map(|_| match rng.next() % 23 {
                0 => Px::Hole,
                1 => Px::Void,
                2 => Px::Known,
                _ => Px::Source,
            })
            .collect();
        let valid = valid_sources(w, h, r, &px, &photocraft_raster::Interrupt::NONE).unwrap();
        let source: Vec<bool> = px.iter().map(|k| *k == Px::Source).collect();
        let ss = integral(w, h, &source);
        let (ri, full) = (r as i32, ((2 * r + 1) * (2 * r + 1)) as u32);
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let inside = x >= ri && y >= ri && x + ri < w as i32 && y + ri < h as i32;
                let want = inside && window_count(&ss, w, h, x - ri, y - ri, x + ri + 1, y + ri + 1) == full;
                assert_eq!(valid[y as usize * w + x as usize], want, "({x}, {y})");
            }
        }
        // Counting per row and block finds the centres in order.
        let sources = Sources::new(&valid, w);
        let list = sources.all(&valid, w);
        assert!(list.len() > 10 && list.len() == sources.len());
        for (k, p) in list.iter().enumerate() {
            assert_eq!(sources.nth(k, &valid, w), *p);
        }
        // And on a grid wider than a block, with empty blocks and rows.
        let (w, h) = (300, 9);
        let wide: Vec<bool> = (0..w * h).map(|i| (i % w) % 7 == 3 && (i % w) / 64 != 2 && i / w != 4).collect();
        let sources = Sources::new(&wide, w);
        let list = sources.all(&wide, w);
        for (k, p) in list.iter().enumerate() {
            assert_eq!(sources.nth(k, &wide, w), *p);
        }
    }

    #[test]
    fn full_patches_measure_as_clipped_ones() {
        let (w, h, ch, r) = (46, 38, 4, 3);
        let mut img = busy(w, h, ch, 4);
        let hole = disc(w, h, 20.0, 18.0, 6.0);
        let px = roles(w, h, &hole);
        let none = photocraft_raster::Interrupt::NONE;
        let hb = hole_bounds(w, h, &px).unwrap();
        let tb = hb.grow(r, w, h);
        let valid = valid_sources(w, h, r, &px, &none).unwrap();
        let picks: Vec<(i32, i32)> = Sources::new(&valid, w).all(&valid, w).iter().step_by(37).copied().collect();
        assert!(picks.len() > 5);
        for adapt in [Adapt::NONE, Adapt { gain: (0.8, 1.25), bias: 0.08 }] {
            for keep in [false, true] {
                let targets = targets_around(w, r, &px, hb, tb);
                let mut s = Solver {
                    w,
                    h,
                    ch,
                    r: r as i32,
                    img: &mut img,
                    px: &px,
                    valid: valid.clone(),
                    targets,
                    hole_box: hb,
                    tbox: tb,
                    full: Vec::new(),
                    tmom: Vec::new(),
                    smom: Vec::new(),
                    adapt,
                    ctl: none,
                };
                s.prepare(keep);
                let nnf = vec![picks[0]; tb.area()];
                let mut cost = vec![f32::INFINITY; tb.area()];
                s.costs(&nnf, &mut cost).unwrap();
                let mut checked = 0;
                for k in 0..tb.area() {
                    if !s.targets[k] || !s.full[k] {
                        continue;
                    }
                    let t = ((tb.x0 + k % tb.width()) as i32, (tb.y0 + k / tb.width()) as i32);
                    for &src in &picks {
                        if adapt.is_none() {
                            let exact = s.dist_clipped(t, src, f32::INFINITY);
                            assert_eq!(s.dist_full(t, src, f32::INFINITY).to_bits(), exact.to_bits());
                            for cutoff in [exact * 0.5, exact, exact * 2.0] {
                                let (a, b) = (s.dist_full(t, src, cutoff), s.dist_clipped(t, src, cutoff));
                                assert_eq!(a < cutoff, b < cutoff);
                            }
                        } else {
                            let (mut m1, mut m2) = ([[0.0f32; 5]; 4], [[0.0f32; 5]; 4]);
                            s.moments_full(k, t, src, &mut m1);
                            assert_eq!(s.moments(t, src, &mut m2), (2 * r + 1) * (2 * r + 1));
                            assert_eq!(m1.map(|c| c.map(f32::to_bits)), m2.map(|c| c.map(f32::to_bits)));
                            assert_eq!(s.dist_adapted_full(k, t, src).to_bits(), s.dist_adapted(t, src).to_bits());
                        }
                        checked += 1;
                    }
                }
                assert!(checked > 100, "{checked}");
            }
        }
    }

    #[test]
    fn owned_completion_fills_in_place_or_gives_the_pixels_back() {
        let (w, h, ch) = (64, 52, 3);
        let img = busy(w, h, ch, 6);
        let hole = disc(w, h, 30.0, 26.0, 8.0);
        let px = roles(w, h, &hole);
        let (p, none) = (CompleteParams::default(), photocraft_raster::Interrupt::NONE);
        let borrowed = complete_masked(w, h, ch, &img, &px, &p, &none).unwrap().unwrap();
        let owned = complete_masked_owned(w, h, ch, img.clone(), &px, &p, &none).unwrap().unwrap();
        assert_eq!(bits(&owned), bits(&borrowed));
        // Nothing may be copied: no completion, and the pixels come back as they were.
        let known: Vec<Px> = px.iter().map(|k| if *k == Px::Source { Px::Known } else { *k }).collect();
        assert!(complete_masked(w, h, ch, &img, &known, &p, &none).unwrap().is_none());
        let back = complete_masked_owned(w, h, ch, img.clone(), &known, &p, &none).unwrap().unwrap_err();
        assert_eq!(bits(&back), bits(&img));
        // Cancelled, it stops.
        let yes = || true;
        assert!(complete_masked_owned(w, h, ch, img, &px, &p, &photocraft_raster::Interrupt::cancel_only(&yes)).is_err());
    }

    #[test]
    fn synthesis_fills_hole_with_plausible_values() {
        let (w, h, ch) = (64, 64, 3);
        let clean = texture(w, h);
        let hole = disc(w, h, 32.0, 32.0, 6.0);
        let out = synthesize(w, h, ch, &clean, &hole, 8, 3);
        let err = hole_rmse(&out, &clean, &hole, ch);
        assert!(err < 0.15, "{err}");
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
