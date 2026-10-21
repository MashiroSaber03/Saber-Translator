//! Intra prediction (RFC 6386 §12): whole-macroblock luma and chroma modes, and the ten 4×4
//! sub-block modes. Predictions read the frame's *unfiltered* reconstruction, which is what the
//! decoder predicts from too (the loop filter runs on the finished frame).

use super::tables::*;

/// A plane of the reconstruction with the virtual border VP8 defines: the row above the frame
/// reads as 127, the column left of it as 129, and the pixel above-left of the frame as 127.
pub struct Plane {
    pub data: Vec<u8>,
    pub width: usize,
}

impl Plane {
    pub fn new(width: usize, height: usize) -> Self {
        Plane { data: vec![0; width * height], width }
    }

    #[inline]
    pub fn at(&self, x: isize, y: isize) -> u8 {
        if y < 0 {
            127
        } else if x < 0 {
            129
        } else {
            self.data.get(y as usize * self.width + x as usize).copied().unwrap_or(127)
        }
    }

    /// The row above an `n`-wide block at (`x`, `y`), plus the four pixels to its upper right
    /// (sub-block modes use them; `avail_right` says whether those exist in the frame yet).
    pub fn above(&self, x: usize, y: usize, n: usize, out: &mut [u8]) {
        for (i, o) in out.iter_mut().enumerate().take(n) {
            *o = self.at((x + i) as isize, y as isize - 1);
        }
    }

    pub fn left(&self, x: usize, y: usize, n: usize, out: &mut [u8]) {
        for (i, o) in out.iter_mut().enumerate().take(n) {
            *o = self.at(x as isize - 1, (y + i) as isize);
        }
    }

    /// The pixel above-left of (`x`, `y`).
    pub fn corner(&self, x: usize, y: usize) -> u8 {
        self.at(x as isize - 1, y as isize - 1)
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: u8) {
        if let Some(p) = self.data.get_mut(y * self.width + x) {
            *p = v;
        }
    }
}

#[inline]
fn clamp255(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Whole-block prediction of an `n × n` block at (`x`, `y`) in `plane` (16 for luma, 8 for chroma)
/// with one of DC / V / H / TM, written row-major into `out`.
pub fn predict_block(plane: &Plane, x: usize, y: usize, n: usize, mode: u8, out: &mut [u8]) {
    let mut above = [0u8; 16];
    let mut left = [0u8; 16];
    plane.above(x, y, n, &mut above);
    plane.left(x, y, n, &mut left);
    let top_row = y == 0;
    let left_col = x == 0;
    match mode {
        V_PRED => {
            for r in 0..n {
                out[r * n..r * n + n].copy_from_slice(&above[..n]);
            }
        }
        H_PRED => {
            for r in 0..n {
                out[r * n..r * n + n].fill(left[r]);
            }
        }
        TM_PRED => {
            let p = i32::from(plane.corner(x, y));
            for r in 0..n {
                for c in 0..n {
                    out[r * n + c] = clamp255(i32::from(left[r]) + i32::from(above[c]) - p);
                }
            }
        }
        _ => {
            // DC: the average of the visible edge pixels; 128 in the top-left corner.
            let shift = if n == 16 { 4 } else { 3 };
            let v = match (top_row, left_col) {
                (true, true) => 128,
                (true, false) => {
                    let s: u32 = left[..n].iter().map(|&v| u32::from(v)).sum();
                    ((s + (1 << (shift - 1))) >> shift) as u8
                }
                (false, true) => {
                    let s: u32 = above[..n].iter().map(|&v| u32::from(v)).sum();
                    ((s + (1 << (shift - 1))) >> shift) as u8
                }
                (false, false) => {
                    let s: u32 = above[..n].iter().chain(&left[..n]).map(|&v| u32::from(v)).sum();
                    ((s + (1 << shift)) >> (shift + 1)) as u8
                }
            };
            out[..n * n].fill(v);
        }
    }
}

#[inline]
fn avg2(a: u8, b: u8) -> u8 {
    ((u16::from(a) + u16::from(b) + 1) >> 1) as u8
}

#[inline]
fn avg3(a: u8, b: u8, c: u8) -> u8 {
    ((u16::from(a) + 2 * u16::from(b) + u16::from(c) + 2) >> 2) as u8
}

/// The edge pixels a 4×4 sub-block prediction reads: `above[0..8]` (the four above and the four
/// above-right), `left[0..4]`, and the corner `p` above-left.
pub struct SubEdges {
    pub above: [u8; 8],
    pub left: [u8; 4],
    pub p: u8,
}

impl SubEdges {
    /// Edges of sub-block `b` (0..16, raster) of the macroblock at (`mx`, `my`) in luma pixels.
    /// The above-right pixels of the right-column sub-blocks come from the row above the
    /// macroblock (positions (-1, 16..20)), replicated from (-1, 15) for the rightmost macroblock
    /// (RFC 6386 §12.3); for the top macroblock row they read 127 like every above pixel.
    pub fn gather(plane: &Plane, mx: usize, my: usize, b: usize, mb_cols: usize) -> SubEdges {
        let bx = mx + (b & 3) * 4;
        let by = my + (b >> 2) * 4;
        let mut above = [0u8; 8];
        let mut left = [0u8; 4];
        plane.above(bx, by, 4, &mut above);
        plane.left(bx, by, 4, &mut left);
        let p = plane.corner(bx, by);
        if b & 3 == 3 {
            // Right column: the above-right pixels are those above the macroblock's right edge.
            let last_col = mx / 16 + 1 == mb_cols;
            for i in 0..4 {
                let ax = if last_col { mx as isize + 15 } else { mx as isize + 16 + i as isize };
                above[4 + i] = plane.at(ax, my as isize - 1);
            }
        } else {
            for i in 0..4 {
                above[4 + i] = plane.at((bx + 4 + i) as isize, by as isize - 1);
            }
        }
        SubEdges { above, left, p }
    }
}

/// Predicts one 4×4 sub-block with `mode` into `out` (row-major), RFC 6386 §12.3.
// Written as the specification's position-by-position assignments, which index on purpose.
#[allow(clippy::needless_range_loop)]
pub fn predict_sub(e: &SubEdges, mode: u8, out: &mut [u8; 16]) {
    let a = &e.above;
    let l = &e.left;
    // E: the nine edge pixels from bottom-left, through the corner, to the right.
    let ed = [l[3], l[2], l[1], l[0], e.p, a[0], a[1], a[2], a[3]];
    // avg3 centred on E[i], avg2 of E[i] and E[i + 1].
    let e3 = |i: usize| avg3(ed[i - 1], ed[i], ed[i + 1]);
    let e2 = |i: usize| avg2(ed[i], ed[i + 1]);
    // The same over the above row (A[-1] is P).
    let ap = |i: isize| if i < 0 { e.p } else { a[i as usize] };
    let a3 = |i: isize| avg3(ap(i - 1), ap(i), ap(i + 1));
    let a2 = |i: isize| avg2(ap(i), ap(i + 1));
    let lp = |i: isize| if i < 0 { e.p } else { l[i as usize] };
    let l3 = |i: isize| avg3(lp(i - 1), lp(i), lp(i + 1));
    let l2 = |i: isize| avg2(lp(i), lp(i + 1));
    let mut b = [[0u8; 4]; 4];
    match mode {
        B_TM_PRED => {
            for r in 0..4 {
                for c in 0..4 {
                    b[r][c] = clamp255(i32::from(l[r]) + i32::from(a[c]) - i32::from(e.p));
                }
            }
        }
        B_VE_PRED => {
            for c in 0..4 {
                let v = a3(c as isize);
                for r in 0..4 {
                    b[r][c] = v;
                }
            }
        }
        B_HE_PRED => {
            let rows = [l3(0), l3(1), l3(2), avg3(l[2], l[3], l[3])];
            for r in 0..4 {
                b[r] = [rows[r]; 4];
            }
        }
        B_LD_PRED => {
            b[0][0] = a3(1);
            b[0][1] = a3(2);
            b[1][0] = b[0][1];
            b[0][2] = a3(3);
            b[1][1] = b[0][2];
            b[2][0] = b[0][2];
            b[0][3] = a3(4);
            b[1][2] = b[0][3];
            b[2][1] = b[0][3];
            b[3][0] = b[0][3];
            b[1][3] = a3(5);
            b[2][2] = b[1][3];
            b[3][1] = b[1][3];
            b[2][3] = a3(6);
            b[3][2] = b[2][3];
            b[3][3] = avg3(a[6], a[7], a[7]);
        }
        B_RD_PRED => {
            b[3][0] = e3(1);
            b[3][1] = e3(2);
            b[2][0] = b[3][1];
            b[3][2] = e3(3);
            b[2][1] = b[3][2];
            b[1][0] = b[3][2];
            b[3][3] = e3(4);
            b[2][2] = b[3][3];
            b[1][1] = b[3][3];
            b[0][0] = b[3][3];
            b[2][3] = e3(5);
            b[1][2] = b[2][3];
            b[0][1] = b[2][3];
            b[1][3] = e3(6);
            b[0][2] = b[1][3];
            b[0][3] = e3(7);
        }
        B_VR_PRED => {
            b[3][0] = e3(2);
            b[2][0] = e3(3);
            b[3][1] = e3(4);
            b[1][0] = b[3][1];
            b[2][1] = e2(4);
            b[0][0] = b[2][1];
            b[3][2] = e3(5);
            b[1][1] = b[3][2];
            b[2][2] = e2(5);
            b[0][1] = b[2][2];
            b[3][3] = e3(6);
            b[1][2] = b[3][3];
            b[2][3] = e2(6);
            b[0][2] = b[2][3];
            b[1][3] = e3(7);
            b[0][3] = e2(7);
        }
        B_VL_PRED => {
            b[0][0] = a2(0);
            b[1][0] = a3(1);
            b[2][0] = a2(1);
            b[0][1] = b[2][0];
            b[1][1] = a3(2);
            b[3][0] = b[1][1];
            b[2][1] = a2(2);
            b[0][2] = b[2][1];
            b[3][1] = a3(3);
            b[1][2] = b[3][1];
            b[2][2] = a2(3);
            b[0][3] = b[2][2];
            b[3][2] = a3(4);
            b[1][3] = b[3][2];
            b[2][3] = a3(5);
            b[3][3] = a3(6);
        }
        B_HD_PRED => {
            b[3][0] = e2(0);
            b[3][1] = e3(1);
            b[2][0] = e2(1);
            b[3][2] = b[2][0];
            b[2][1] = e3(2);
            b[3][3] = b[2][1];
            b[2][2] = e2(2);
            b[1][0] = b[2][2];
            b[2][3] = e3(3);
            b[1][1] = b[2][3];
            b[1][2] = e2(3);
            b[0][0] = b[1][2];
            b[1][3] = e3(4);
            b[0][1] = b[1][3];
            b[0][2] = e3(5);
            b[0][3] = e3(6);
        }
        B_HU_PRED => {
            b[0][0] = l2(0);
            b[0][1] = l3(1);
            b[0][2] = l2(1);
            b[1][0] = b[0][2];
            b[0][3] = l3(2);
            b[1][1] = b[0][3];
            b[1][2] = l2(2);
            b[2][0] = b[1][2];
            b[1][3] = avg3(l[2], l[3], l[3]);
            b[2][1] = b[1][3];
            b[2][2] = l[3];
            b[2][3] = l[3];
            b[3] = [l[3]; 4];
        }
        _ => {
            // B_DC_PRED: the average of the four above and four left pixels.
            let s: u32 = a[..4].iter().chain(&l[..]).map(|&v| u32::from(v)).sum();
            b = [[((s + 4) >> 3) as u8; 4]; 4];
        }
    }
    for r in 0..4 {
        out[r * 4..r * 4 + 4].copy_from_slice(&b[r]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn border_values() {
        let p = Plane::new(16, 16);
        assert_eq!(p.at(-1, -1), 127);
        assert_eq!(p.at(5, -1), 127);
        assert_eq!(p.at(-1, 5), 129);
        assert_eq!(p.corner(0, 0), 127);
        assert_eq!(p.corner(0, 16), 129);
        assert_eq!(p.corner(16, 0), 127);
    }

    #[test]
    fn dc_prediction_edge_cases() {
        let mut p = Plane::new(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                p.set(x, y, if y < 16 { 10 } else { 200 });
            }
        }
        let mut out = [0u8; 256];
        predict_block(&p, 0, 0, 16, DC_PRED, &mut out);
        assert!(out.iter().all(|&v| v == 128), "top-left is 128");
        predict_block(&p, 16, 0, 16, DC_PRED, &mut out);
        assert!(out.iter().all(|&v| v == 10), "top row averages the left column only");
        predict_block(&p, 0, 16, 16, DC_PRED, &mut out);
        assert!(out.iter().all(|&v| v == 10), "left column averages the above row only");
        predict_block(&p, 16, 16, 16, DC_PRED, &mut out);
        assert!(out.iter().all(|&v| v == (10 + 200) / 2));
        predict_block(&p, 16, 16, 16, TM_PRED, &mut out);
        // L + A - P = 200 + 10 - 10 = 200 everywhere.
        assert!(out.iter().all(|&v| v == 200));
    }

    #[test]
    fn sub_block_modes_are_in_range_and_dc_averages() {
        let mut p = Plane::new(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                p.set(x, y, ((x * 7 + y * 13) % 256) as u8);
            }
        }
        for b in 0..16 {
            let e = SubEdges::gather(&p, 16, 16, b, 2);
            let mut out = [0u8; 16];
            for mode in 0..10u8 {
                predict_sub(&e, mode, &mut out);
            }
            predict_sub(&e, B_DC_PRED, &mut out);
            let s: u32 = e.above[..4].iter().chain(&e.left).map(|&v| u32::from(v)).sum();
            assert!(out.iter().all(|&v| u32::from(v) == (s + 4) >> 3));
        }
        // Rightmost macroblock: the above-right pixels replicate (-1, 15).
        let e = SubEdges::gather(&p, 16, 16, 3, 2);
        assert!(e.above[4..].iter().all(|&v| v == p.at(31, 15)));
        // Top row: everything above is 127.
        let e = SubEdges::gather(&p, 0, 0, 3, 2);
        assert!(e.above.iter().all(|&v| v == 127));
        assert_eq!(e.p, 127);
    }
}
