//! The Content-Aware Move Tool as a command: move or duplicate the selected pixels, and let
//! content-aware fill close the gaps.
//!
//! The selection is the content, `offset` how far the user dragged it. At the new place the content
//! is pasted, fitted to its new surroundings by `color`, and a band along its edge, inside the
//! selection, is re-synthesised from the surroundings so that it merges in. In `move` mode the place
//! the content left is then filled content-aware (`extend` keeps it). That fill never samples the
//! moved content, so the object doesn't come back as a ghost; the edge band may sample it (it is the
//! context the band blends towards), but not the content's old place.
//!
//! Measured on Photoshop 25.4 (a 60 px selection holding a ring-shaped object, dragged 150 px):
//! * **Structure** 3 to 7 give the same result, bit for bit: the content is copied exactly up to
//!   about 8 px from the selection's edge, and that band blends into the surroundings. 2 widens the
//!   band; 1 re-synthesises the content too, so even its middle is no longer an exact copy.
//! * **Color** 0 pastes the content as it is; higher values bring its level towards the new
//!   surroundings by up to about `0.04 · color` (Patch Tool, content 42 and 102 levels brighter:
//!   3 shifts it 32 and 31 levels, 10 shifts it 35 and 95), a limited gain and bias
//!   ([`photocraft_algo::content_aware::color_amount`]).
//!
//! The selection moves with the content, as in Photoshop. A soft selection counts wherever it is
//! above zero (as the Patch Tool's); the band does the blending. The fills are PatchMatch
//! completions and can take seconds on large selections, so the command runs as a background job
//! (#210): with progress, cancellable, and the document unchanged until it applies. The Patch
//! Tool's Content-Aware mode lands its content the same way ([`land`]).
//!
//! **Transform On Drop** (the options-bar checkbox): after the drop a transform box lets the user
//! scale and rotate the content before the blend is computed. `scaleX` / `scaleY` (%) and
//! `rotate` (degrees, clockwise) scale and turn the content about the centre of the selection's
//! bounds before `offset` moves it, as `select.transformSelection` does; the selection is
//! transformed with it. A transformed patch may reach past the canvas edge (Photoshop lets the
//! box go there): what lands outside is dropped.

use photocraft_algo::content_aware::{FillOptions, color_amount, color_level, fill_with, level_transform};
use photocraft_algo::transform::{Homography, Interp};

use super::patch::{coverage, offset, selection_area, target_layer};
use super::*;

const CMD: &str = "paint.contentAwareMove";

/// Largest selection, in pixels of its bounding box (as the Patch Tool's). Each fill window is the
/// selection plus a margin; at this size one window holds about 37 M pixels and the completion's
/// working set stays within a few GB.
const MAX_PIXELS: u64 = 16 << 20;

/// Cap on the sampling margin around the content (the window grows with the selection).
const MAX_MARGIN: i32 = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// The content leaves its place, which is filled from the surroundings.
    Move,
    /// The content is duplicated; its place stays.
    Extend,
}

/// A pixel target and a selection to move.
pub(super) fn enabled(s: &Session) -> std::result::Result<(), String> {
    has_pixel_layer(s)?;
    let d = s.active().ok_or("no document open")?;
    if d.doc.selection.is_some() { Ok(()) } else { Err("select the area to move first".into()) }
}

/// Optional whole-number param in `lo..=hi`; absent or null gives `default`.
fn level(p: &Value, k: &str, default: u8, lo: u8, hi: u8) -> Result<u8> {
    match p.get(k) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => match v.as_f64() {
            Some(x) if x.is_finite() && (f64::from(lo)..=f64::from(hi)).contains(&x) => Ok(x.round() as u8),
            _ => Err(bad(CMD, format!("`{k}` must be a number {lo}..{hi}"))),
        },
    }
}

/// The pixels a step reads and writes: a layer (or, with `None`, the channel `p` targets), and
/// whether reading samples the visible composite (Sample All Layers).
#[derive(Clone, Copy, Debug)]
pub(super) struct Target {
    pub id: Option<LayerId>,
    pub all_layers: bool,
}

/// What a move does, checked before any work starts.
#[derive(Clone, Copy, Debug)]
struct Plan {
    target: Target,
    canvas: Rect,
    /// The selected area (where the content comes from).
    area: Rect,
    offset: (i32, i32),
    /// Transform On Drop: where a source point lands (scale and rotation about the area's centre,
    /// then the offset); `None` for a plain move by `offset`.
    map: Option<Homography>,
    /// Where the content lands (the moved area, or the transformed one's bounds), on the canvas.
    dest: Rect,
    mode: Mode,
    structure: u8,
    color: u8,
    /// Most pixels the fills may sample around the content ([`MAX_MARGIN`], less for a preview).
    margin: i32,
}

/// Transform On Drop's `scaleX` / `scaleY` (%) and `rotate` (degrees): the map from a source point
/// to where it lands, about the centre of `area` and then moved by `offset`. `None` when the params
/// leave the content as it is (a plain move).
fn placement(p: &Value, area: Rect, (dx, dy): (i32, i32)) -> Result<Option<Homography>> {
    let num = |k: &str, default: f64| -> Result<f64> {
        match p.get(k) {
            None | Some(Value::Null) => Ok(default),
            Some(v) => v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| bad(CMD, format!("`{k}` must be a finite number"))),
        }
    };
    let scale = |k: &str| -> Result<f64> {
        let v = num(k, 100.0)?;
        // 1% to 10000% either way (a negative scale flips): beyond that the content collapses to
        // nothing or outgrows any selection the move can take.
        if (1.0..=10_000.0).contains(&v.abs()) { Ok(v / 100.0) } else { Err(bad(CMD, format!("`{k}` must be 1..10000 % (or -10000..-1 to flip)"))) }
    };
    let (sx, sy) = (scale("scaleX")?, scale("scaleY")?);
    let deg = num("rotate", 0.0)?;
    if deg.abs() > 36_000.0 {
        return Err(bad(CMD, "`rotate` must be -36000..36000 degrees"));
    }
    let deg = deg.rem_euclid(360.0);
    if sx == 1.0 && sy == 1.0 && deg == 0.0 {
        return Ok(None);
    }
    let (cx, cy) = (f64::from(area.x0 + area.x1) / 2.0, f64::from(area.y0 + area.y1) / 2.0);
    let t = |x: f64, y: f64| Homography([1.0, 0.0, x, 0.0, 1.0, y, 0.0, 0.0, 1.0]);
    let (sa, ca) = deg.to_radians().sin_cos();
    let rot = Homography([ca, -sa, 0.0, sa, ca, 0.0, 0.0, 0.0, 1.0]);
    let scale = Homography([sx, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 1.0]);
    Ok(Some(t(cx + f64::from(dx), cy + f64::from(dy)).mul(&rot).mul(&scale).mul(&t(-cx, -cy))))
}

/// The pixel bounds of `r` mapped by `h`.
fn mapped_bounds(h: &Homography, r: Rect) -> Rect {
    let corners = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)].map(|(x, y)| h.apply(f64::from(x), f64::from(y)));
    let (mut x0, mut y0, mut x1, mut y1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for (x, y) in corners {
        (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
    }
    if !(x0.is_finite() && y0.is_finite() && x1.is_finite() && y1.is_finite()) {
        return Rect::EMPTY;
    }
    // Rounded inwards by a hair so an exact scale doesn't grow the bounds by a pixel.
    let clamp = |v: f64| v.clamp(-1e8, 1e8) as i32;
    Rect::new(clamp((x0 + 1e-6).floor()), clamp((y0 + 1e-6).floor()), clamp((x1 - 1e-6).ceil()), clamp((y1 - 1e-6).ceil()))
}

/// `src` resampled (bilinear, edges clamped) through `inv` (destination → source) over `rect`:
/// the content as it lands. Works on float samples, so at any depth.
fn resample(src: &Region, inv: &Homography, rect: Rect) -> Region {
    let (sw, sh, n) = (src.rect.width() as usize, src.rect.height() as usize, src.ch);
    let mut out = Region { rect, ch: n, data: vec![0.0; rect.width() as usize * rect.height() as usize * n] };
    if sw == 0 || sh == 0 {
        return out;
    }
    let px = |x: usize, y: usize, c: usize| src.data.get((y * sw + x) * n + c).copied().unwrap_or(0.0);
    for (i, dst) in out.data.chunks_exact_mut(n).enumerate() {
        let (qx, qy) = (rect.x0 + (i % rect.width() as usize) as i32, rect.y0 + (i / rect.width() as usize) as i32);
        let (sx, sy) = inv.apply(f64::from(qx) + 0.5, f64::from(qy) + 0.5);
        let fx = (sx - 0.5 - f64::from(src.rect.x0)).clamp(0.0, (sw - 1) as f64);
        let fy = (sy - 0.5 - f64::from(src.rect.y0)).clamp(0.0, (sh - 1) as f64);
        if !(fx.is_finite() && fy.is_finite()) {
            continue;
        }
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(sw - 1), (y0 + 1).min(sh - 1));
        let (tx, ty) = ((fx - x0 as f64) as f32, (fy - y0 as f64) as f32);
        for (c, d) in dst.iter_mut().enumerate() {
            let top = px(x0, y0, c) * (1.0 - tx) + px(x1, y0, c) * tx;
            let bottom = px(x0, y1, c) * (1.0 - tx) + px(x1, y1, c) * tx;
            *d = top * (1.0 - ty) + bottom * ty;
        }
    }
    out
}

/// The selection's shape as it lands over `rect`: `cov` (coverage over `area`) sampled through
/// `inv` (destination → source), nearest, set wherever it is above zero (as [`mask`]).
fn landed_mask(cov: &[f32], area: Rect, inv: &Homography, rect: Rect) -> Vec<bool> {
    let w = rect.width() as usize;
    (0..w * rect.height() as usize)
        .map(|i| {
            let (qx, qy) = (rect.x0 + (i % w) as i32, rect.y0 + (i / w) as i32);
            let (sx, sy) = inv.apply(f64::from(qx) + 0.5, f64::from(qy) + 0.5);
            let (x, y) = (sx.floor(), sy.floor());
            if !(f64::from(area.x0)..f64::from(area.x1)).contains(&x) || !(f64::from(area.y0)..f64::from(area.y1)).contains(&y) {
                return false;
            }
            let j = (y as i32 - area.y0) as usize * area.width() as usize + (x as i32 - area.x0) as usize;
            cov.get(j).is_some_and(|c| *c > 0.0)
        })
        .collect()
}

fn plan(s: &Session, p: &Value, margin: i32) -> Result<Plan> {
    let mode = match string(p, "mode", "move") {
        "move" => Mode::Move,
        "extend" => Mode::Extend,
        o => return Err(bad(CMD, format!("unknown mode `{o}` (move|extend)"))),
    };
    let structure = level(p, "structure", 4, 1, 7)?;
    let color = level(p, "color", 0, 0, 10)?;
    let (dx, dy) = offset(CMD, p)?;
    let id = target_layer(s, p)?;
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let canvas = d.doc.bounds();
    let sel = d.doc.selection.as_ref().ok_or_else(|| bad(CMD, "select the area to move first"))?;
    let area = selection_area(sel, canvas);
    if area.is_empty() {
        return Err(bad(CMD, "the selection is empty"));
    }
    let pixels = u64::from(area.width()) * u64::from(area.height());
    if pixels > MAX_PIXELS {
        return Err(bad(CMD, format!("the selection is too large to move ({pixels} pixels, at most {MAX_PIXELS})")));
    }
    let map = placement(p, area, (dx, dy))?;
    let dest = match &map {
        None => {
            if (dx, dy) == (0, 0) {
                return Err(bad(CMD, "drag the selection to where the content should go"));
            }
            // Content dropped beyond the edge would be lost: keep it on the canvas, as the Patch
            // Tool does.
            let dest = area.translate(dx, dy);
            if !canvas.contains_rect(&dest) {
                return Err(bad(CMD, "the moved selection must stay inside the canvas"));
            }
            dest
        }
        Some(h) => {
            let landed = mapped_bounds(h, area);
            let pixels = u64::from(landed.width()) * u64::from(landed.height());
            if pixels > MAX_PIXELS {
                return Err(bad(CMD, format!("the transformed selection is too large to move ({pixels} pixels, at most {MAX_PIXELS})")));
            }
            let dest = landed.intersect(&canvas);
            if dest.is_empty() {
                return Err(bad(CMD, "the transformed selection must land on the canvas"));
            }
            dest
        }
    };
    let all_layers = flag(p, "sampleAllLayers", false) && targets_pixels(p);
    Ok(Plan { target: Target { id, all_layers }, canvas, area, offset: (dx, dy), map, dest, mode, structure, color, margin })
}

/// Width in pixels of the edge band re-synthesised at the new place, for Structure 1..7 and content
/// `min_side` pixels across. Photoshop gives the same result for 3 to 7: the band is about a patch
/// wide whatever the content's size; 2 and 1 widen it (1 so far that the content itself is
/// re-synthesised).
pub(super) fn band_width(structure: u8, min_side: u32) -> usize {
    let side = min_side as f32;
    let w = match structure {
        0 | 1 => (side * 0.3).max(12.0),
        2 => (side * 0.18).max(9.0),
        _ => 6.0,
    };
    (w.round() as usize).min(128)
}

/// The cells of a `w × h` mask whose whole `(2r+1)²` neighbourhood, as far as it lies on the grid,
/// is set (a square erosion in O(w·h) through a summed-area table). The grid's border doesn't erode:
/// a selection reaching the canvas edge keeps its content there.
fn erode(w: usize, h: usize, m: &[bool], r: usize) -> Vec<bool> {
    if r == 0 {
        return m.to_vec();
    }
    let w1 = w + 1;
    let mut sat = vec![0u32; w1 * (h + 1)];
    for y in 0..h {
        let mut row = 0u32;
        for x in 0..w {
            row += u32::from(m.get(y * w + x).copied().unwrap_or(false));
            sat[(y + 1) * w1 + x + 1] = sat[y * w1 + x + 1] + row;
        }
    }
    let mut out = vec![false; w * h];
    for y in 0..h {
        let (y0, y1) = (y.saturating_sub(r), (y + r + 1).min(h));
        for x in 0..w {
            if !m.get(y * w + x).copied().unwrap_or(false) {
                continue;
            }
            let (x0, x1) = (x.saturating_sub(r), (x + r + 1).min(w));
            let set = sat[y1 * w1 + x1] + sat[y0 * w1 + x0] - sat[y0 * w1 + x1] - sat[y1 * w1 + x0];
            out[y * w + x] = set as usize == (x1 - x0) * (y1 - y0);
        }
    }
    out
}

/// Where a fill samples from around `r`: Photoshop's Content-Aware window
/// ([`crate::fill_cmds::sampling_window`]), at most [`MAX_MARGIN`] pixels around `r`.
pub(super) fn window(r: Rect, canvas: Rect) -> Rect {
    window_within(r, canvas, MAX_MARGIN)
}

/// [`window`] with at most `margin` pixels around `r`.
fn window_within(r: Rect, canvas: Rect, margin: i32) -> Rect {
    crate::fill_cmds::sampling_window(r, canvas).intersect(&r.inflate(margin))
}

/// The pixels a step works on: the targeted surface, or with Sample All Layers the visible
/// composite (encoded in the target's format).
pub(super) fn read(doc: &mut Document, t: Target, p: &Value, rect: Rect) -> Result<Region> {
    let (surf, _) = crate::channel_cmds::target_surface(doc, t.id, p)?;
    if t.all_layers {
        let fmt = surf.format();
        return Ok(composite_region(doc, None, SampleLayers::All, rect, fmt));
    }
    Ok(Region::read(surf, rect))
}

/// Write `out` over `mask` into the target, replacing the pixels (the result already holds what
/// should show there). With the transparency lock, alpha is kept and transparent pixels stay as
/// they are. Returns the written rectangle. Shared with the Remove Tool.
pub(super) fn replace(doc: &mut Document, t: Target, p: &Value, out: &Region, mask: &[bool]) -> Result<Rect> {
    let (surf, lock) = crate::channel_cmds::target_surface(doc, t.id, p)?;
    let fmt = surf.format();
    let a = alpha_index(&fmt);
    let rect = out.rect;
    let mut px = surf.read_region(rect);
    let n = out.ch;
    if px.len() != out.data.len() || mask.len() * n != px.len() {
        return Err(EngineError::Other("internal error: fill buffers disagree in size".into()));
    }
    for ((dst, src), m) in px.chunks_exact_mut(n).zip(out.data.chunks_exact(n)).zip(mask) {
        if !*m {
            continue;
        }
        match a.filter(|_| lock) {
            Some(ai) => {
                if dst.get(ai).is_some_and(|v| *v > 0.0) {
                    for (c, (d, s)) in dst.iter_mut().zip(src).enumerate() {
                        if c != ai {
                            *d = *s;
                        }
                    }
                }
            }
            None => dst.copy_from_slice(src),
        }
    }
    surf.write_region(rect, &px);
    Ok(rect)
}

/// Selection coverage over `rect` as a mask (set wherever it is above zero).
pub(super) fn mask(sel: &Surface, rect: Rect) -> Vec<bool> {
    coverage(sel, rect).into_iter().map(|c| c > 0.0).collect()
}

/// Fill options: the fill's own colour adaptation (Edit › Content-Aware Fill's default), seeded so
/// the result is reproducible.
pub(super) fn fill_options() -> FillOptions {
    FillOptions { seed: 0x00ca_4e00, ..FillOptions::default() }
}

/// A ring of `w` pixels just outside `m` (within the grid).
fn outer_ring(gw: usize, gh: usize, m: &[bool], w: usize) -> Vec<bool> {
    let inside: Vec<bool> = m.iter().map(|v| !*v).collect();
    erode(gw, gh, &inside, w).iter().zip(m).map(|(keep, sel)| !*keep && !*sel).collect()
}

/// Content landing at `dst_area`: the pixels of the window around it, with `selected` (the
/// selection's shape there) to be replaced by the content found `shift` away (`content` = that
/// window, so `content` pixel = window pixel + `shift`). The content's middle is copied, its level
/// fitted to the new surroundings by `color`; a `band_width(structure)` band inside the selection's
/// edge is re-synthesised from the surroundings and that middle, never sampling `avoid`. Writes the
/// result into the target and returns the rectangle it changed.
#[allow(clippy::too_many_arguments)]
pub(super) fn land(
    doc: &mut Document,
    t: Target,
    p: &Value,
    win: Rect,
    selected: &[bool],
    content: &Region,
    avoid: &[bool],
    structure: u8,
    color: u8,
    stage: (f32, f32),
    ctx: &crate::jobs::JobCtx,
    label: &str,
) -> Result<Rect> {
    let (w, h) = (win.width() as usize, win.height() as usize);
    let here = read(doc, t, p, win)?;
    let n = here.ch;
    let (sx0, sy0, sx1, sy1) = selected.iter().enumerate().filter(|(_, s)| **s).fold((w, h, 0, 0), |b, (i, _)| {
        let (x, y) = (i % w, i / w);
        (b.0.min(x), b.1.min(y), b.2.max(x + 1), b.3.max(y + 1))
    });
    let side = (sx1.saturating_sub(sx0)).min(sy1.saturating_sub(sy0)) as u32;
    let core = erode(w, h, selected, band_width(structure, side));
    ctx.check()?;
    // Colour: the content's surroundings (where it came from) against the new ones, both just
    // outside the selection's shape.
    let ring = outer_ring(w, h, selected, 6);
    let pick = |r: &Region| -> Vec<f32> { ring.iter().enumerate().filter(|(_, k)| **k).flat_map(|(i, _)| r.data[i * n..(i + 1) * n].to_vec()).collect() };
    let gb = level_transform(n, &pick(content), &pick(&here), color_amount(color));
    let fmt = crate::channel_cmds::target_surface(doc, t.id, p)?.0.format();
    let alpha = alpha_index(&fmt);
    let mut buf = here;
    for (i, _) in core.iter().enumerate().filter(|(_, c)| **c) {
        for (c, &(g, b)) in gb.iter().enumerate() {
            let v = content.data[i * n + c];
            // Colour channels are fitted; alpha is copied.
            buf.data[i * n + c] = if Some(c) == alpha { v } else { g * v + b };
        }
    }
    clamp_samples(&fmt, &mut buf.data);
    // Structure: the band between the kept core and the selection's edge comes from the
    // surroundings and the kept core, never from what `avoid` marks.
    let band: Vec<bool> = selected.iter().zip(&core).map(|(m, c)| *m && !*c).collect();
    if band.iter().any(|b| *b) {
        let allowed: Vec<bool> = avoid.iter().map(|o| !*o).collect();
        let opts = FillOptions { color_adaptation: if color == 0 { color_level("default") } else { color_amount(color) }, ..fill_options() };
        buf.data = ctx.stage(stage.0, stage.1, label, |ctl| fill_with(w, h, n, &buf.data, &band, &allowed, &opts, ctl)).map_err(|_| EngineError::Cancelled)?;
    }
    replace(doc, t, p, &buf, selected)
}

/// The new target surface, the rectangle it changed and (Transform On Drop) the transformed
/// selection: the content pasted at the offset, then (`move`) its old place filled. Works on `doc`,
/// a copy of the document.
fn run_move(doc: &mut Document, sel: &Surface, plan: &Plan, p: &Value, ctx: &crate::jobs::JobCtx, label: &str) -> Result<(Surface, Rect, Option<Surface>)> {
    let (dx, dy) = plan.offset;
    let opts = fill_options();
    let split = if plan.mode == Mode::Move { 0.4 } else { 1.0 };
    let cancelled = |_| EngineError::Cancelled;
    let inv = match &plan.map {
        Some(h) => Some(h.inverse().ok_or_else(|| bad(CMD, "the transform collapses the selection"))?),
        None => None,
    };
    // The selection's coverage over the area, which the transformed shape is sampled from.
    let cov = if inv.is_some() { coverage(sel, plan.area) } else { Vec::new() };

    // 1. The content at its new place: the content with its original surroundings laid over the
    // window there; never sampling the content's old place.
    let win = window_within(plan.dest, plan.canvas, plan.margin);
    let (content, moved) = match &inv {
        None => {
            let mut content = read(doc, plan.target, p, win.translate(-dx, -dy))?;
            content.rect = win;
            (content, mask(sel, win.translate(-dx, -dy)))
        }
        Some(inv) => {
            // Only the content and a ring around it (the colour fit's reference) matter: read
            // the source behind that, a few pixels past the area, and resample it.
            let src = mapped_bounds(inv, plan.dest.inflate(8)).intersect(&plan.area.inflate(plan.margin)).intersect(&plan.canvas);
            let src = if src.is_empty() { plan.area } else { src };
            (resample(&read(doc, plan.target, p, src)?, inv, win), landed_mask(&cov, plan.area, inv, win))
        }
    };
    let old_place = mask(sel, win);
    let mut damage = land(doc, plan.target, p, win, &moved, &content, &old_place, plan.structure, plan.color, (0.0, split), ctx, label)?;

    // 2. Move: fill the place the content left (where the content didn't land on it).
    if plan.mode == Mode::Move {
        ctx.check()?;
        let win = window_within(plan.area, plan.canvas, plan.margin);
        let (w, h) = (win.width() as usize, win.height() as usize);
        let img = read(doc, plan.target, p, win)?;
        let landed = match &inv {
            None => mask(sel, win.translate(-dx, -dy)),
            Some(inv) => landed_mask(&cov, plan.area, inv, win),
        };
        let hole: Vec<bool> = mask(sel, win).iter().zip(&landed).map(|(s, l)| *s && !*l).collect();
        if hole.iter().any(|h| *h) {
            let allowed: Vec<bool> = landed.iter().map(|l| !*l).collect();
            let data = ctx.stage(split, 1.0, label, |ctl| fill_with(w, h, img.ch, &img.data, &hole, &allowed, &opts, ctl)).map_err(cancelled)?;
            damage = damage.union(&replace(doc, plan.target, p, &Region { rect: win, ch: img.ch, data }, &hole)?);
        }
    }
    let surf = crate::channel_cmds::target_surface(doc, plan.target.id, p)?.0.clone();
    // The selection follows the transformed content, as Select › Transform Selection moves it.
    let moved_sel = match &plan.map {
        Some(h) => Some(crate::transform_cmds::warp_gray(sel, h, Interp::Bilinear)?),
        None => None,
    };
    Ok((surf, damage, moved_sel))
}

pub(super) fn content_aware_move(s: &mut Session, p: &Value) -> Result<Value> {
    let plan = plan(s, p, MAX_MARGIN)?;
    let label = if plan.mode == Mode::Move { "Content-Aware Move" } else { "Content-Aware Extend" };
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let doc = d.doc.clone();
    let p = p.clone();
    crate::jobs::run(
        s,
        label,
        true,
        {
            let p = p.clone();
            move |ctx| {
                ctx.progress(0.0, label);
                let mut work = (*doc).clone();
                let sel = doc.selection.as_ref().ok_or_else(|| bad(CMD, "select the area to move first"))?;
                run_move(&mut work, sel, &plan, &p, ctx, label)
            }
        },
        move |s, (surf, damage, moved_sel)| {
            let (dx, dy) = plan.offset;
            s.edit(label, |doc, _| {
                *crate::channel_cmds::target_surface(doc, plan.target.id, &p)?.0 = surf;
                // The selection follows the content, ready for another drag.
                if let Some(moved) = moved_sel {
                    doc.selection = Some(moved).filter(|s| !s.content_bounds().is_empty());
                } else if let Some(sel) = doc.selection.as_ref() {
                    doc.selection = Some(crate::layer_multi_cmds::shift_surface(sel, dx, dy));
                }
                Ok(())
            })?;
            if let Some(st) = s.active_mut() {
                st.last_damage = Some(damage);
            }
            let mode = if plan.mode == Mode::Move { "move" } else { "extend" };
            let mut out = json!({ "damage": damage_json(damage), "offset": [dx, dy], "mode": mode });
            if plan.map.is_some() {
                out["transformed"] = json!(true);
            }
            Ok(out)
        },
    )
}

/// Largest selection (bounding box, in pixels) the live preview computes.
const PREVIEW_MAX_PIXELS: u64 = 256 * 256;
/// Sampling margin of a preview: a fraction of the command's, so a preview is a small solve.
const PREVIEW_MARGIN: i32 = 48;

/// A live preview of `paint.contentAwareMove` that can run on any thread: the document as of
/// [`preview`], with the move's params. Cheaper than the command (a small sampling margin, and
/// only small selections), so the result is close to, not the same as, what the command makes.
pub struct ContentAwarePreview {
    doc: std::sync::Arc<Document>,
    plan: Plan,
    params: Value,
}

impl ContentAwarePreview {
    /// The active document with the move applied (nothing is recorded, the session is not
    /// touched) and the rectangle it changed. Stops with `Cancelled` once `ctx` is cancelled.
    pub fn run(&self, ctx: &crate::jobs::JobCtx) -> Result<(Document, Rect)> {
        let mut work = (*self.doc).clone();
        let sel = self.doc.selection.as_ref().ok_or_else(|| bad(CMD, "select the area to move first"))?;
        let (_, damage, _) = run_move(&mut work, sel, &self.plan, &self.params, ctx, "Content-Aware Move preview")?;
        Ok((work, damage))
    }
}

/// Prepare the preview of `paint.contentAwareMove` with params `p` on the active document. Errors
/// like the command does, and for a selection larger than a preview takes.
pub fn preview(s: &Session, p: &Value) -> Result<ContentAwarePreview> {
    let plan = plan(s, p, PREVIEW_MARGIN)?;
    let pixels = u64::from(plan.area.width()) * u64::from(plan.area.height());
    if pixels > PREVIEW_MAX_PIXELS {
        return Err(bad(CMD, format!("the selection is too large to preview ({pixels} pixels, at most {PREVIEW_MAX_PIXELS})")));
    }
    let d = s.active().ok_or(EngineError::NoDocument)?;
    Ok(ContentAwarePreview { doc: d.doc.clone(), plan, params: p.clone() })
}
