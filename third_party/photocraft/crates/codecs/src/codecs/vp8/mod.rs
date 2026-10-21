//! A clean-room VP8 key-frame encoder (the lossy WebP bitstream), implemented from RFC 6386.
//!
//! One intra frame: 16×16 macroblocks in raster order, each predicted from the frame's own
//! reconstruction with one of the four whole-block luma modes or the sixteen 4×4 sub-block
//! modes (`B_PRED`), chroma with one of four modes; the residual goes through the 4×4 DCT
//! (and the WHT of the luma DCs for whole-block modes), one frame-wide quantizer derived from the
//! quality, and the boolean entropy coder with the specification's default token probabilities,
//! refined per frame from the frame's own token statistics. Macroblocks without coefficients
//! are skipped. The loop filter is left to the decoder at a level matched to the quantizer.
//!
//! What it does not do (yet): rate-distortion optimized mode decisions and quantization,
//! segments, multiple token partitions, or SIMD. It is deterministic, allocation-bounded and
//! never panics: dimensions are checked up front and every index is derived from them.

mod bool_encoder;
mod predict;
mod tables;
mod transform;

use bool_encoder::BoolEncoder;
use predict::{Plane, SubEdges, predict_block, predict_sub};
use tables::*;
use transform::{fdct, fwht, idct, iwht};

/// The largest dimension a VP8 frame header can hold (14 bits).
pub const MAX_DIMENSION: u32 = 16383;

/// A planar 4:2:0 picture: `y` is `width × height`, `u` and `v` are `ceil(w/2) × ceil(h/2)`.
pub struct Yuv {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

impl Yuv {
    /// Converts interleaved 8-bit RGB (`stride` bytes per pixel, 3 or 4) to studio-range BT.601
    /// Y'CbCr with the chroma averaged over each 2 × 2 block, the convention WebP decoders invert.
    pub fn from_rgb(rgb: &[u8], width: u32, height: u32, stride: usize) -> Yuv {
        let (w, h) = (width as usize, height as usize);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let mut y = vec![0u8; w * h];
        let mut u = vec![0u8; cw * ch];
        let mut v = vec![0u8; cw * ch];
        let px = |x: usize, yy: usize| -> (i32, i32, i32) {
            let i = (yy.min(h - 1) * w + x.min(w - 1)) * stride;
            (i32::from(rgb[i]), i32::from(rgb[i + 1]), i32::from(rgb[i + 2]))
        };
        for yy in 0..h {
            for x in 0..w {
                let (r, g, b) = px(x, yy);
                y[yy * w + x] = (((66 * r + 129 * g + 25 * b + 128) >> 8) + 16) as u8;
            }
        }
        for cy in 0..ch {
            for cx in 0..cw {
                let (mut r, mut g, mut b) = (0, 0, 0);
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let (pr, pg, pb) = px(cx * 2 + dx, cy * 2 + dy);
                    r += pr;
                    g += pg;
                    b += pb;
                }
                let (r, g, b) = ((r + 2) >> 2, (g + 2) >> 2, (b + 2) >> 2);
                u[cy * cw + cx] = (((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128).clamp(0, 255) as u8;
                v[cy * cw + cx] = (((112 * r - 94 * g - 18 * b + 128) >> 8) + 128).clamp(0, 255) as u8;
            }
        }
        Yuv { width, height, y, u, v }
    }
}

/// Encoder settings.
#[derive(Debug, Clone, Copy)]
pub struct Params {
    /// 0 (smallest) … 100 (best); maps to the frame quantizer.
    pub quality: u8,
}

/// Why a frame cannot be encoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Vp8Error {
    /// A dimension is 0 or above [`MAX_DIMENSION`].
    Dimensions,
    /// The plane buffers do not match the dimensions.
    Planes,
}

impl std::fmt::Display for Vp8Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Vp8Error::Dimensions => write!(f, "VP8 frames are 1 to {MAX_DIMENSION} pixels per side"),
            Vp8Error::Planes => write!(f, "plane sizes do not match the dimensions"),
        }
    }
}

/// Quality 0..=100 → quantizer index 0..=127 (0 is finest). A gentle curve: quality 90 ≈ 11,
/// 75 ≈ 28, 50 ≈ 60, 25 ≈ 92.
pub fn quantizer_index(quality: u8) -> u8 {
    let q = f64::from(quality.min(100));
    let t = (100.0 - q) / 100.0;
    (t.powf(1.1) * 127.0).round().clamp(0.0, 127.0) as u8
}

/// Dequantization factors for one quantizer index (RFC 6386 §14.1 and dixie's `dequant_init`).
#[derive(Debug, Clone, Copy)]
struct Quant {
    y_dc: i32,
    y_ac: i32,
    y2_dc: i32,
    y2_ac: i32,
    uv_dc: i32,
    uv_ac: i32,
}

impl Quant {
    fn new(qi: u8) -> Quant {
        let qi = usize::from(qi.min(127));
        Quant {
            y_dc: DC_QLOOKUP[qi],
            y_ac: AC_QLOOKUP[qi],
            y2_dc: DC_QLOOKUP[qi] * 2,
            y2_ac: (AC_QLOOKUP[qi] * 155 / 100).max(8),
            uv_dc: DC_QLOOKUP[qi].min(132),
            uv_ac: AC_QLOOKUP[qi],
        }
    }
}

/// Quantizes one coefficient: round to nearest for DC, with a small dead zone for AC (fewer
/// ±1 coefficients cost far fewer bits than they are worth in distortion).
#[inline]
fn quantize(coeff: i32, factor: i32, dc: bool) -> i32 {
    let bias = if dc { factor / 2 } else { factor * 7 / 16 };
    let q = (coeff.abs() + bias) / factor;
    let q = q.min(2048);
    if coeff < 0 { -q } else { q }
}

/// Block types of the token probability table (RFC 6386 §13.3).
const TYPE_Y_AFTER_Y2: usize = 0;
const TYPE_Y2: usize = 1;
const TYPE_UV: usize = 2;
const TYPE_Y_WITH_DC: usize = 3;

/// Which coefficient token codes `v` and, for the categories, the extra bits to send.
fn token_of(v: i32) -> (u8, Option<(usize, u32)>) {
    let a = v.unsigned_abs() as i32;
    match a {
        0 => (DCT_0, None),
        1 => (DCT_1, None),
        2 => (DCT_2, None),
        3 => (DCT_3, None),
        4 => (DCT_4, None),
        5..=6 => (DCT_CAT1, Some((0, (a - CAT_BASE[0]) as u32))),
        7..=10 => (DCT_CAT2, Some((1, (a - CAT_BASE[1]) as u32))),
        11..=18 => (DCT_CAT3, Some((2, (a - CAT_BASE[2]) as u32))),
        19..=34 => (DCT_CAT4, Some((3, (a - CAT_BASE[3]) as u32))),
        35..=66 => (DCT_CAT5, Some((4, (a - CAT_BASE[4]) as u32))),
        _ => (DCT_CAT6, Some((5, (a - CAT_BASE[5]) as u32))),
    }
}

/// One coded block's quantized coefficients in zig-zag order, from `first` (1 for Y after Y2).
struct Block {
    coeffs: [i32; 16],
    first: usize,
    /// One past the last non-zero zig-zag position (`first` when the block is empty).
    eob: usize,
}

impl Block {
    fn from_raster(raster: &[i32; 16], first: usize) -> Block {
        let mut coeffs = [0i32; 16];
        for (pos, &idx) in ZIGZAG.iter().enumerate() {
            coeffs[pos] = raster[idx];
        }
        let eob = (first..16).rev().find(|&p| coeffs[p] != 0).map_or(first, |p| p + 1);
        Block { coeffs, first, eob }
    }

    fn has_coeffs(&self) -> bool {
        self.eob > self.first
    }
}

/// Token-count statistics, `[type][band][ctx][token]`, from which the frame's probabilities are
/// derived (and the per-node branch counts needed to turn them into tree probabilities).
#[derive(Clone)]
struct Stats {
    /// Per tree node (11 per context): how often the bool was 0 and how often 1.
    node: Vec<[[[[u32; 2]; 11]; 3]; 8]>,
}

impl Stats {
    fn new() -> Stats {
        Stats { node: vec![[[[[0; 2]; 11]; 3]; 8]; 4] }
    }
}

/// Everything about a block's tokens the two passes share: the branch path of each token in the
/// coefficient tree. Walked once per block to count branches, once to write them.
fn tree_path(value: u8, start: usize) -> Vec<(usize, bool)> {
    fn find(node: usize, value: u8, path: &mut Vec<(usize, bool)>) -> bool {
        for bit in [false, true] {
            let i = node + usize::from(bit);
            let Some(&entry) = COEFF_TREE.get(i) else { continue };
            path.push((node >> 1, bit));
            if entry <= 0 {
                if (-entry) as u8 == value {
                    return true;
                }
            } else if find(entry as usize, value, path) {
                return true;
            }
            path.pop();
        }
        false
    }
    let mut path = Vec::with_capacity(8);
    find(start, value, &mut path);
    path
}

/// One bool of a block's token sequence.
enum Event {
    /// A coefficient-tree branch: `(type, band, ctx, node, bit)`, coded with `probs[type][band][ctx][node]`.
    Node(usize, usize, usize, usize, bool),
    /// The extra bits of a category token: `(category, bits)`.
    Extra(usize, u32),
    /// The sign of a non-zero coefficient (`true` = negative).
    Sign(bool),
}

/// Visits every bool of a block's token sequence in coding order.
fn for_each_token_bool(block: &Block, ty: usize, ctx0: usize, mut f: impl FnMut(Event)) {
    let mut ctx = ctx0;
    let mut prev_zero = false;
    let mut pos = block.first;
    while pos < block.eob {
        let v = block.coeffs[pos];
        let (token, cat) = token_of(v);
        let band = COEFF_BANDS[pos];
        for (node, bit) in tree_path(token, if prev_zero { 2 } else { 0 }) {
            f(Event::Node(ty, band, ctx, node, bit));
        }
        if let Some((c, bits)) = cat {
            f(Event::Extra(c, bits));
        }
        if v != 0 {
            f(Event::Sign(v < 0));
        }
        ctx = match v.abs() {
            0 => 0,
            1 => 1,
            _ => 2,
        };
        prev_zero = v == 0;
        pos += 1;
    }
    if block.eob < 16 {
        // End of block (never directly after a zero: the coded run ends with a non-zero value).
        let band = COEFF_BANDS[block.eob];
        for (node, bit) in tree_path(DCT_EOB, 0) {
            f(Event::Node(ty, band, ctx, node, bit));
        }
    }
}

/// A macroblock's decisions and coefficients.
struct Macroblock {
    y_mode: u8,
    b_modes: [u8; 16],
    uv_mode: u8,
    /// Y2 (whole-block modes only), 16 Y, 4 U, 4 V.
    y2: Option<Block>,
    y: Vec<Block>,
    u: Vec<Block>,
    v: Vec<Block>,
    skip: bool,
}

/// Non-zero contexts: 4 Y (by column for "above", by row for "left"), 2 U, 2 V, 1 Y2.
type Ctx = [u8; 9];

struct Encoder<'a> {
    src: &'a Yuv,
    mb_cols: usize,
    mb_rows: usize,
    q: Quant,
    rec_y: Plane,
    rec_u: Plane,
    rec_v: Plane,
    mbs: Vec<Macroblock>,
}

impl<'a> Encoder<'a> {
    fn new(src: &'a Yuv, qi: u8) -> Encoder<'a> {
        let mb_cols = (src.width as usize).div_ceil(16);
        let mb_rows = (src.height as usize).div_ceil(16);
        Encoder {
            src,
            mb_cols,
            mb_rows,
            q: Quant::new(qi),
            rec_y: Plane::new(mb_cols * 16, mb_rows * 16),
            rec_u: Plane::new(mb_cols * 8, mb_rows * 8),
            rec_v: Plane::new(mb_cols * 8, mb_rows * 8),
            mbs: Vec::with_capacity(mb_cols * mb_rows),
        }
    }

    /// Source luma at (`x`, `y`), the frame's last row and column extended over the padding.
    #[inline]
    fn src_y(&self, x: usize, y: usize) -> i32 {
        let (w, h) = (self.src.width as usize, self.src.height as usize);
        i32::from(self.src.y[y.min(h - 1) * w + x.min(w - 1)])
    }

    #[inline]
    fn src_c(&self, plane: &[u8], x: usize, y: usize) -> i32 {
        let (w, h) = ((self.src.width as usize).div_ceil(2), (self.src.height as usize).div_ceil(2));
        i32::from(plane[y.min(h - 1) * w + x.min(w - 1)])
    }

    /// Encodes one 4×4 block: residual → coefficients → quantized (in raster order, with the
    /// quantization factors) → the dequantized coefficients the decoder will see.
    fn code_block(residual: &[i32; 16], dc_factor: i32, ac_factor: i32, with_dc: bool) -> ([i32; 16], [i32; 16]) {
        let coeffs = fdct(residual);
        let mut quantized = [0i32; 16];
        let mut dequantized = [0i32; 16];
        for i in 0..16 {
            let factor = if i == 0 { dc_factor } else { ac_factor };
            if i == 0 && !with_dc {
                continue;
            }
            quantized[i] = quantize(coeffs[i], factor, i == 0);
            dequantized[i] = quantized[i] * factor;
        }
        (quantized, dequantized)
    }

    /// Sum of squared differences between a predicted block and the source.
    fn sse(pred: &[u8], src: &[i32]) -> u64 {
        pred.iter()
            .zip(src)
            .map(|(&p, &s)| {
                let d = i64::from(p) - i64::from(s);
                (d * d) as u64
            })
            .sum()
    }

    // Sub-block indices address several arrays by geometry; ranges read better than iterators here.
    #[allow(clippy::needless_range_loop)]
    fn encode_macroblock(&mut self, mbx: usize, mby: usize) {
        let (x0, y0) = (mbx * 16, mby * 16);
        // Source luma of this macroblock.
        let mut src = [0i32; 256];
        for r in 0..16 {
            for c in 0..16 {
                src[r * 16 + c] = self.src_y(x0 + c, y0 + r);
            }
        }
        // Whole-block luma modes: the least squared prediction error wins.
        let mut pred16 = [0u8; 256];
        let mut best_mode = DC_PRED;
        let mut best_sse = u64::MAX;
        let mut best_pred = [0u8; 256];
        for mode in [DC_PRED, V_PRED, H_PRED, TM_PRED] {
            predict_block(&self.rec_y, x0, y0, 16, mode, &mut pred16);
            let e = Self::sse(&pred16, &src);
            if e < best_sse {
                best_sse = e;
                best_mode = mode;
                best_pred = pred16;
            }
        }
        // Rate-aware comparison with B_PRED: each sub-block mode costs about three bits, and
        // the whole-block path codes the DCs once in Y2. lambda scales distortion per bit with
        // the quantizer (squared error units).
        let lambda = u64::from((self.q.y_ac * self.q.y_ac) as u32) / 8;
        let whole_cost = best_sse + 3 * lambda;

        // B_PRED: sub-blocks in raster order, each predicted from the reconstruction so far.
        let saved_y = self.rec_y.data.clone();
        let mut b_modes = [B_DC_PRED; 16];
        let mut b_blocks: Vec<Block> = Vec::with_capacity(16);
        let mut b_sse = 0u64;
        {
            for b in 0..16 {
                let (bx, by) = (x0 + (b & 3) * 4, y0 + (b >> 2) * 4);
                let edges = SubEdges::gather(&self.rec_y, x0, y0, b, self.mb_cols);
                let mut sub_src = [0i32; 16];
                for r in 0..4 {
                    for c in 0..4 {
                        sub_src[r * 4 + c] = src[((b >> 2) * 4 + r) * 16 + (b & 3) * 4 + c];
                    }
                }
                let mut best = (u64::MAX, B_DC_PRED, [0u8; 16]);
                let mut pred = [0u8; 16];
                for mode in 0..10u8 {
                    predict_sub(&edges, mode, &mut pred);
                    let e = Self::sse(&pred, &sub_src);
                    if e < best.0 {
                        best = (e, mode, pred);
                    }
                }
                b_modes[b] = best.1;
                b_sse += best.0;
                let residual: [i32; 16] = std::array::from_fn(|i| sub_src[i] - i32::from(best.2[i]));
                let (quantized, dequantized) = Self::code_block(&residual, self.q.y_dc, self.q.y_ac, true);
                let recon = idct(&dequantized);
                for r in 0..4 {
                    for c in 0..4 {
                        self.rec_y.set(bx + c, by + r, (i32::from(best.2[r * 4 + c]) + recon[r * 4 + c]).clamp(0, 255) as u8);
                    }
                }
                b_blocks.push(Block::from_raster(&quantized, 0));
            }
        }
        let b_cost = b_sse + 50 * lambda;

        let (y_mode, y2, y_blocks) = if b_cost < whole_cost {
            (B_PRED, None, b_blocks)
        } else {
            // Whole-block mode: undo the B_PRED reconstruction, code 16 blocks + Y2.
            self.rec_y.data = saved_y;
            let mut dcs = [0i32; 16];
            let mut ac_blocks: Vec<[i32; 16]> = Vec::with_capacity(16);
            let mut deq_blocks: Vec<[i32; 16]> = Vec::with_capacity(16);
            for b in 0..16 {
                let mut residual = [0i32; 16];
                for r in 0..4 {
                    for c in 0..4 {
                        let i = ((b >> 2) * 4 + r) * 16 + (b & 3) * 4 + c;
                        residual[r * 4 + c] = src[i] - i32::from(best_pred[i]);
                    }
                }
                let coeffs = fdct(&residual);
                dcs[b] = coeffs[0];
                let mut quantized = [0i32; 16];
                let mut dequantized = [0i32; 16];
                for i in 1..16 {
                    quantized[i] = quantize(coeffs[i], self.q.y_ac, false);
                    dequantized[i] = quantized[i] * self.q.y_ac;
                }
                ac_blocks.push(quantized);
                deq_blocks.push(dequantized);
            }
            // The DCs through the WHT, quantized as Y2, and back through the inverse WHT so the
            // reconstruction uses exactly the decoder's DC values.
            let y2_coeffs = fwht(&dcs);
            let mut y2_q = [0i32; 16];
            let mut y2_deq = [0i32; 16];
            for i in 0..16 {
                let factor = if i == 0 { self.q.y2_dc } else { self.q.y2_ac };
                y2_q[i] = quantize(y2_coeffs[i], factor, i == 0);
                y2_deq[i] = y2_q[i] * factor;
            }
            let dc_rec = iwht(&y2_deq);
            for b in 0..16 {
                deq_blocks[b][0] = dc_rec[b];
                let recon = idct(&deq_blocks[b]);
                let (bx, by) = (x0 + (b & 3) * 4, y0 + (b >> 2) * 4);
                for r in 0..4 {
                    for c in 0..4 {
                        let i = ((b >> 2) * 4 + r) * 16 + (b & 3) * 4 + c;
                        self.rec_y.set(bx + c, by + r, (i32::from(best_pred[i]) + recon[r * 4 + c]).clamp(0, 255) as u8);
                    }
                }
            }
            let blocks = ac_blocks.iter().map(|q| Block::from_raster(q, 1)).collect();
            (best_mode, Some(Block::from_raster(&y2_q, 0)), blocks)
        };

        // Chroma: one mode for both planes.
        let (cx0, cy0) = (mbx * 8, mby * 8);
        let mut src_u = [0i32; 64];
        let mut src_v = [0i32; 64];
        for r in 0..8 {
            for c in 0..8 {
                src_u[r * 8 + c] = self.src_c(&self.src.u, cx0 + c, cy0 + r);
                src_v[r * 8 + c] = self.src_c(&self.src.v, cx0 + c, cy0 + r);
            }
        }
        let mut uv_mode = DC_PRED;
        let mut uv_best = u64::MAX;
        let mut pred_u = [0u8; 64];
        let mut pred_v = [0u8; 64];
        let mut pu = [0u8; 64];
        let mut pv = [0u8; 64];
        for mode in [DC_PRED, V_PRED, H_PRED, TM_PRED] {
            predict_block(&self.rec_u, cx0, cy0, 8, mode, &mut pu);
            predict_block(&self.rec_v, cx0, cy0, 8, mode, &mut pv);
            let e = Self::sse(&pu, &src_u) + Self::sse(&pv, &src_v);
            if e < uv_best {
                uv_best = e;
                uv_mode = mode;
                pred_u = pu;
                pred_v = pv;
            }
        }
        let mut u_blocks = Vec::with_capacity(4);
        let mut v_blocks = Vec::with_capacity(4);
        for (plane_src, pred, blocks, is_u) in [(&src_u, &pred_u, &mut u_blocks, true), (&src_v, &pred_v, &mut v_blocks, false)] {
            for b in 0..4 {
                let mut residual = [0i32; 16];
                for r in 0..4 {
                    for c in 0..4 {
                        let i = ((b >> 1) * 4 + r) * 8 + (b & 1) * 4 + c;
                        residual[r * 4 + c] = plane_src[i] - i32::from(pred[i]);
                    }
                }
                let (quantized, dequantized) = Self::code_block(&residual, self.q.uv_dc, self.q.uv_ac, true);
                let recon = idct(&dequantized);
                let (bx, by) = (cx0 + (b & 1) * 4, cy0 + (b >> 1) * 4);
                for r in 0..4 {
                    for c in 0..4 {
                        let i = ((b >> 1) * 4 + r) * 8 + (b & 1) * 4 + c;
                        let value = (i32::from(pred[i]) + recon[r * 4 + c]).clamp(0, 255) as u8;
                        if is_u {
                            self.rec_u.set(bx + c, by + r, value);
                        } else {
                            self.rec_v.set(bx + c, by + r, value);
                        }
                    }
                }
                blocks.push(Block::from_raster(&quantized, 0));
            }
        }

        let skip = y2.as_ref().is_none_or(|b| !b.has_coeffs())
            && y_blocks.iter().all(|b| !b.has_coeffs())
            && u_blocks.iter().all(|b| !b.has_coeffs())
            && v_blocks.iter().all(|b| !b.has_coeffs());
        self.mbs.push(Macroblock { y_mode, b_modes, uv_mode, y2, y: y_blocks, u: u_blocks, v: v_blocks, skip });
    }

    /// Runs `f` over every token bool of the frame in coding order, with the non-zero contexts
    /// maintained exactly as the decoder does (RFC 6386 §13.3; skipped macroblocks reset them,
    /// leaving Y2 alone for `B_PRED` macroblocks).
    fn walk_tokens(&self, mut f: impl FnMut(&Block, usize, usize), mut on_mb: impl FnMut(usize, &Macroblock)) {
        let mut above: Vec<Ctx> = vec![[0; 9]; self.mb_cols];
        for mby in 0..self.mb_rows {
            let mut left: Ctx = [0; 9];
            for (mbx, a) in above.iter_mut().enumerate() {
                let mb = &self.mbs[mby * self.mb_cols + mbx];
                on_mb(mby * self.mb_cols + mbx, mb);
                if mb.skip {
                    for i in 0..8 {
                        left[i] = 0;
                        a[i] = 0;
                    }
                    if mb.y_mode != B_PRED {
                        left[8] = 0;
                        a[8] = 0;
                    }
                    continue;
                }
                if let Some(y2) = &mb.y2 {
                    let ctx = usize::from(left[8] + a[8]);
                    f(y2, TYPE_Y2, ctx);
                    let t = u8::from(y2.has_coeffs());
                    left[8] = t;
                    a[8] = t;
                }
                let ty = if mb.y2.is_some() { TYPE_Y_AFTER_Y2 } else { TYPE_Y_WITH_DC };
                for (i, b) in mb.y.iter().enumerate() {
                    let (ai, li) = (i & 3, i >> 2);
                    let ctx = usize::from(left[li] + a[ai]);
                    f(b, ty, ctx);
                    let t = u8::from(b.has_coeffs());
                    left[li] = t;
                    a[ai] = t;
                }
                for (base, blocks) in [(4usize, &mb.u), (6usize, &mb.v)] {
                    for (i, b) in blocks.iter().enumerate() {
                        let (ai, li) = (base + (i & 1), base + (i >> 1));
                        let ctx = usize::from(left[li] + a[ai]);
                        f(b, TYPE_UV, ctx);
                        let t = u8::from(b.has_coeffs());
                        left[li] = t;
                        a[ai] = t;
                    }
                }
            }
        }
    }

    /// Token probabilities for this frame: the defaults, replaced where the frame's own branch
    /// counts give a clearly better estimate (sent as updates in the header).
    fn choose_probs(&self, stats: &Stats) -> [[[[u8; 11]; 3]; 8]; 4] {
        let mut probs = DEFAULT_COEFF_PROBS;
        for (t, plane) in probs.iter_mut().enumerate() {
            for (b, band) in plane.iter_mut().enumerate() {
                for (c, ctx) in band.iter_mut().enumerate() {
                    for (n, p) in ctx.iter_mut().enumerate() {
                        let [zeros, ones] = stats.node[t][b][c][n];
                        let total = zeros + ones;
                        if total < 24 {
                            continue;
                        }
                        // Probability of a 0 branch, kept inside 1..=255.
                        let est = ((u64::from(zeros) * 256 + u64::from(total) / 2) / u64::from(total)).clamp(1, 255) as u8;
                        // Only worth the update bits when the estimate moves the coding cost.
                        let cost = |pr: u8| -> f64 {
                            let p0 = f64::from(pr) / 256.0;
                            -(f64::from(zeros) * p0.log2() + f64::from(ones) * (1.0 - p0).log2())
                        };
                        let update_bits = 8.0 + 1.0;
                        if cost(*p) - cost(est) > update_bits * 2.0 {
                            *p = est;
                        }
                    }
                }
            }
        }
        probs
    }

    fn write_frame(&self, probs: &[[[[u8; 11]; 3]; 8]; 4], loop_filter_level: u8, qi: u8) -> Vec<u8> {
        // First partition: frame header, then per-macroblock prediction records.
        let mut h = BoolEncoder::new();
        h.put_flag(false); // color_space: BT.601-like
        h.put_flag(false); // clamping_type: the decoder clamps
        h.put_flag(false); // segmentation_enabled
        h.put_flag(false); // filter_type: normal
        h.put_literal(6, u32::from(loop_filter_level.min(63)));
        h.put_literal(3, 0); // sharpness_level
        h.put_flag(false); // loop_filter_adj_enable
        h.put_literal(2, 0); // one token partition
        h.put_literal(7, u32::from(qi.min(127))); // y_ac_qi
        for _ in 0..5 {
            h.put_flag(false); // no quantizer deltas
        }
        h.put_flag(true); // refresh_entropy_probs (meaningless for a single frame)
        for t in 0..4 {
            for b in 0..8 {
                for c in 0..3 {
                    for n in 0..11 {
                        let changed = probs[t][b][c][n] != DEFAULT_COEFF_PROBS[t][b][c][n];
                        h.put(COEFF_UPDATE_PROBS[t][b][c][n], changed);
                        if changed {
                            h.put_literal(8, u32::from(probs[t][b][c][n]));
                        }
                    }
                }
            }
        }
        h.put_flag(true); // mb_no_skip_coeff
        let skipped = self.mbs.iter().filter(|m| m.skip).count() as u64;
        let total = self.mbs.len().max(1) as u64;
        let prob_skip_false = (((total - skipped) * 256 + total / 2) / total).clamp(1, 255) as u8;
        h.put_literal(8, u32::from(prob_skip_false));

        for mby in 0..self.mb_rows {
            for mbx in 0..self.mb_cols {
                let mb = &self.mbs[mby * self.mb_cols + mbx];
                h.put(prob_skip_false, mb.skip);
                h.put_tree(KF_YMODE_TREE, &KF_YMODE_PROB, mb.y_mode);
                if mb.y_mode == B_PRED {
                    for b in 0..16 {
                        let above = if b >= 4 {
                            mb.b_modes[b - 4]
                        } else if mby == 0 {
                            B_DC_PRED
                        } else {
                            self.sub_mode_of(&self.mbs[(mby - 1) * self.mb_cols + mbx], b + 12)
                        };
                        let left = if b & 3 != 0 {
                            mb.b_modes[b - 1]
                        } else if mbx == 0 {
                            B_DC_PRED
                        } else {
                            self.sub_mode_of(&self.mbs[mby * self.mb_cols + mbx - 1], b + 3)
                        };
                        h.put_tree(BMODE_TREE, &KF_BMODE_PROBS[usize::from(above)][usize::from(left)], mb.b_modes[b]);
                    }
                }
                h.put_tree(UV_MODE_TREE, &KF_UV_MODE_PROB, mb.uv_mode);
            }
        }
        let first = h.finish();

        // Token partition.
        let mut t = BoolEncoder::new();
        self.walk_tokens(
            |block, ty, ctx0| {
                for_each_token_bool(block, ty, ctx0, |ev| match ev {
                    Event::Node(ty, band, ctx, node, bit) => t.put(probs[ty][band][ctx][node], bit),
                    Event::Extra(cat, bits) => {
                        let p = PCAT[cat];
                        for (i, &prob) in p.iter().enumerate() {
                            let shift = p.len() - 1 - i;
                            t.put(prob, (bits >> shift) & 1 != 0);
                        }
                    }
                    Event::Sign(negative) => t.put_flag(negative),
                });
            },
            |_, _| {},
        );
        let tokens = t.finish();

        // Frame tag, start code, dimensions, partitions.
        let mut out = Vec::with_capacity(10 + first.len() + tokens.len());
        let size = (first.len() as u32).min((1 << 19) - 1);
        // Bit 0 = 0: key frame; bits 1-3 = 0: version 0; bit 4: show_frame; then the size.
        let tag: u32 = (1 << 4) | (size << 5);
        out.extend_from_slice(&tag.to_le_bytes()[..3]);
        out.extend_from_slice(&[0x9d, 0x01, 0x2a]);
        out.extend_from_slice(&(self.src.width as u16).to_le_bytes());
        out.extend_from_slice(&(self.src.height as u16).to_le_bytes());
        out.extend_from_slice(&first);
        out.extend_from_slice(&tokens);
        out
    }

    /// The sub-block mode context a neighbouring macroblock provides for sub-block `b` (an
    /// index into the neighbour's 16 sub-blocks): its own sub-block mode when it is `B_PRED`,
    /// else the sub-block equivalent of its whole-block mode (RFC 6386 §11.3).
    fn sub_mode_of(&self, mb: &Macroblock, b: usize) -> u8 {
        match mb.y_mode {
            B_PRED => mb.b_modes[b.min(15)],
            V_PRED => B_VE_PRED,
            H_PRED => B_HE_PRED,
            TM_PRED => B_TM_PRED,
            _ => B_DC_PRED,
        }
    }
}

/// Encodes `yuv` as one VP8 key frame (the payload of a WebP `VP8 ` chunk).
pub fn encode_frame(yuv: &Yuv, params: Params) -> Result<Vec<u8>, Vp8Error> {
    if yuv.width == 0 || yuv.height == 0 || yuv.width > MAX_DIMENSION || yuv.height > MAX_DIMENSION {
        return Err(Vp8Error::Dimensions);
    }
    let (w, h) = (yuv.width as usize, yuv.height as usize);
    if yuv.y.len() != w * h || yuv.u.len() != w.div_ceil(2) * h.div_ceil(2) || yuv.v.len() != yuv.u.len() {
        return Err(Vp8Error::Planes);
    }
    let qi = quantizer_index(params.quality);
    let mut enc = Encoder::new(yuv, qi);
    for mby in 0..enc.mb_rows {
        for mbx in 0..enc.mb_cols {
            enc.encode_macroblock(mbx, mby);
        }
    }
    // Pass 1 over the tokens: branch statistics → this frame's probabilities.
    let mut stats = Stats::new();
    enc.walk_tokens(
        |block, ty, ctx0| {
            for_each_token_bool(block, ty, ctx0, |ev| {
                if let Event::Node(ty, band, ctx, node, bit) = ev {
                    stats.node[ty][band][ctx][node][usize::from(bit)] += 1;
                }
            });
        },
        |_, _| {},
    );
    let probs = enc.choose_probs(&stats);
    // The loop filter smooths block edges in proportion to the quantizer; none at the finest.
    let loop_filter_level = if qi == 0 { 0 } else { (u32::from(qi) / 2 + 4).min(63) as u8 };
    Ok(enc.write_frame(&probs, loop_filter_level, qi))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantizer_curve_is_monotone_and_spans_the_range() {
        assert_eq!(quantizer_index(100), 0);
        assert_eq!(quantizer_index(0), 127);
        let mut last = 0;
        for q in (0..=100).rev() {
            let qi = quantizer_index(q);
            assert!(qi >= last, "quality {q}");
            last = qi;
        }
        assert!((8..=14).contains(&quantizer_index(90)), "{}", quantizer_index(90));
        assert!((24..=32).contains(&quantizer_index(75)), "{}", quantizer_index(75));
    }

    #[test]
    fn tokens_cover_every_magnitude() {
        for v in [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 18, 19, 34, 35, 66, 67, 1000, 2048] {
            let (tok, extra) = token_of(v);
            let base = match extra {
                None => i32::from(tok),
                Some((c, bits)) => {
                    assert!(bits < (1 << PCAT[c].len()), "{v}");
                    CAT_BASE[c] + bits as i32
                }
            };
            assert_eq!(base, v);
            assert_eq!(token_of(-v).0, tok);
        }
    }

    #[test]
    fn block_eob_is_after_the_last_nonzero() {
        let mut raster = [0i32; 16];
        let b = Block::from_raster(&raster, 1);
        assert_eq!(b.eob, 1);
        assert!(!b.has_coeffs());
        raster[ZIGZAG[5]] = 3;
        let b = Block::from_raster(&raster, 0);
        assert_eq!(b.eob, 6);
        assert!(b.has_coeffs());
        raster[ZIGZAG[15]] = -1;
        assert_eq!(Block::from_raster(&raster, 0).eob, 16);
    }

    #[test]
    fn rejects_bad_dimensions() {
        let y = Yuv { width: 0, height: 4, y: vec![], u: vec![], v: vec![] };
        assert_eq!(encode_frame(&y, Params { quality: 80 }).unwrap_err(), Vp8Error::Dimensions);
        let y = Yuv { width: 4, height: 4, y: vec![0; 15], u: vec![0; 4], v: vec![0; 4] };
        assert_eq!(encode_frame(&y, Params { quality: 80 }).unwrap_err(), Vp8Error::Planes);
    }

    #[test]
    fn frame_header_is_well_formed() {
        let rgb: Vec<u8> = (0..(20 * 13 * 3)).map(|i| (i * 7 % 256) as u8).collect();
        let yuv = Yuv::from_rgb(&rgb, 20, 13, 3);
        let frame = encode_frame(&yuv, Params { quality: 75 }).unwrap();
        assert_eq!(frame[0] & 1, 0, "key frame");
        assert_eq!((frame[0] >> 4) & 1, 1, "shown");
        assert_eq!(&frame[3..6], &[0x9d, 0x01, 0x2a]);
        assert_eq!(u16::from_le_bytes([frame[6], frame[7]]) & 0x3fff, 20);
        assert_eq!(u16::from_le_bytes([frame[8], frame[9]]) & 0x3fff, 13);
        let first_size = (u32::from(frame[0]) | (u32::from(frame[1]) << 8) | (u32::from(frame[2]) << 16)) >> 5;
        assert!((10 + first_size as usize) < frame.len());
    }
}
