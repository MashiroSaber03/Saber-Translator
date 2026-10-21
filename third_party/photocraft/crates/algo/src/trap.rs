//! Image › Trap (prepress). Trapping spreads each ink slightly into neighbouring inked areas so
//! that, if the press misregisters the plates, the colours still overlap at an edge instead of
//! leaving a white gap. We spread every ink by `width` px (grayscale dilation) but only *under*
//! already-inked pixels, so internal colour edges get a trap band while the object-vs-paper edge is
//! left alone. Clean-room approximation of Photoshop's Trap, which works on a flattened CMYK image.
//!
//! Operates on an interleaved buffer of `ch` channels per pixel (CMYK = inks 0..4, plus alpha).

use crate::photo_util::par_rows;

/// Prefix/suffix block maxima, with clipped edge windows and the original zero floor.
/// One comparison per sample in each direction, independent of the trap width.
fn max_line(out: &mut [f32], radius: usize, get: impl Fn(usize) -> f32) {
    let len = out.len();
    if len == 0 {
        return;
    }
    let r = radius.min(len - 1);
    let mut suffix = Vec::new();
    if r <= 2 || suffix.try_reserve_exact(len).is_err() {
        for (x, v) in out.iter_mut().enumerate() {
            *v = (x.saturating_sub(r)..=x.saturating_add(r).min(len - 1)).fold(0.0f32, |m, i| m.max(get(i)));
        }
        return;
    }
    suffix.resize(len, 0.0f32);
    let block = r.saturating_mul(2).saturating_add(1);
    let mut acc = 0.0f32;
    let mut negative_zero = false;
    for (i, v) in out.iter_mut().enumerate() {
        if i % block == 0 {
            acc = 0.0;
        }
        let value = get(i);
        negative_zero |= value.to_bits() == (-0.0f32).to_bits();
        acc = acc.max(value);
        *v = acc;
    }
    // f32::max may choose either sign of equal zero depending on operand order.
    // Preserve the original left-to-right fold for these uncommon float lines.
    if negative_zero {
        for (x, v) in out.iter_mut().enumerate() {
            *v = (x.saturating_sub(r)..=x.saturating_add(r).min(len - 1)).fold(0.0f32, |m, i| m.max(get(i)));
        }
        return;
    }
    acc = 0.0;
    for (i, v) in suffix.iter_mut().enumerate().rev() {
        if i == len - 1 || (i + 1) % block == 0 {
            acc = 0.0;
        }
        acc = acc.max(get(i));
        *v = acc;
    }
    for x in 0..len {
        let (lo, hi) = (x.saturating_sub(r), x.saturating_add(r).min(len - 1));
        let value = if lo == 0 {
            out.get(hi).copied().unwrap_or(0.0)
        } else if hi == len - 1 && lo / block == hi / block {
            suffix.get(lo).copied().unwrap_or(0.0)
        } else {
            suffix.get(lo).copied().unwrap_or(0.0).max(out.get(hi).copied().unwrap_or(0.0))
        };
        if let Some(v) = out.get_mut(x) {
            *v = value;
        }
    }
}

/// A channel's square dilation, transposed so each vertical scan writes a contiguous row.
fn dilate_channel(src: &[f32], w: usize, h: usize, ch: usize, c: usize, r: usize) -> Vec<f32> {
    if r <= 2 {
        return dilate_channel_direct(src, w, h, ch, c, r);
    }
    let mut tmp = vec![0.0f32; w * h];
    par_rows(&mut tmp, w, 1, |y, row| {
        max_line(row, r, |x| src.get((y * w + x) * ch + c).copied().unwrap_or(0.0));
    });
    let mut out = vec![0.0f32; w * h];
    par_rows(&mut out, h, 1, |x, column| {
        max_line(column, r, |y| tmp.get(y * w + x).copied().unwrap_or(0.0));
    });
    out
}

fn dilate_channel_direct(src: &[f32], w: usize, h: usize, ch: usize, c: usize, r: usize) -> Vec<f32> {
    let get = |buf: &[f32], x: usize, y: usize| buf[(y * w + x) * ch + c];
    // Horizontal pass.
    let mut tmp = vec![0.0f32; w * h];
    par_rows(&mut tmp, w, 1, |y, row| {
        for (x, o) in row.iter_mut().enumerate() {
            let lo = x.saturating_sub(r);
            let hi = (x + r).min(w - 1);
            let mut m = 0.0f32;
            for xx in lo..=hi {
                m = m.max(get(src, xx, y));
            }
            *o = m;
        }
    });
    // Vertical pass over tmp.
    let mut out = vec![0.0f32; w * h];
    par_rows(&mut out, w, 1, |y, row| {
        let lo = y.saturating_sub(r);
        let hi = (y + r).min(h - 1);
        for (x, o) in row.iter_mut().enumerate() {
            let mut m = 0.0f32;
            for yy in lo..=hi {
                m = m.max(tmp[yy * w + x]);
            }
            *o = m;
        }
    });
    out
}

/// Trap an interleaved CMYK(+alpha) buffer in place. `width` is the trap in px (>=1). Inks 0..4 are
/// spread; any alpha channel (`ch > 4`) is untouched. A pixel keeps its original ink where it has
/// no ink at all (paper), so only internal colour edges gain a trap band.
pub fn trap(px: &mut [f32], w: usize, h: usize, ch: usize, width: usize) {
    if width == 0 || w == 0 || h == 0 || ch < 4 || w.checked_mul(h).and_then(|n| n.checked_mul(ch)) != Some(px.len()) {
        return;
    }
    let inks = 4.min(ch);
    let dil: Vec<Vec<f32>> = (0..inks).map(|c| dilate_channel(px, w, h, ch, c, width)).collect();
    par_rows(px, w, ch, |y, row| {
        for (x, pixel) in row.chunks_exact_mut(ch).enumerate() {
            let i = if width <= 2 { y * w + x } else { x * h + y };
            // Printed where any ink is present.
            let inked = (0..inks).any(|c| pixel[c] > 1e-4);
            if !inked {
                continue;
            }
            for c in 0..inks {
                pixel[c] = pixel[c].max(dil[c][i]);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two solid ink blocks meeting at x=4: cyan (C) on the left, magenta (M) on the right.
    fn two_blocks() -> (Vec<f32>, usize, usize) {
        let (w, h, ch) = (8usize, 4usize, 4usize);
        let mut px = vec![0.0f32; w * h * ch];
        for y in 0..h {
            for x in 0..w {
                let p = &mut px[(y * w + x) * ch..][..ch];
                if x < 4 {
                    p[0] = 1.0; // cyan
                } else {
                    p[1] = 1.0; // magenta
                }
            }
        }
        (px, w, h)
    }

    #[test]
    fn traps_the_internal_colour_edge() {
        let (mut px, w, h) = two_blocks();
        let ch = 4;
        trap(&mut px, w, h, ch, 1);
        // At the seam, cyan and magenta now overlap (both inks > 0) in a 1px band on each side.
        let at = |x: usize, y: usize, c: usize| px[(y * w + x) * ch + c];
        assert!(at(3, 0, 1) > 0.5, "magenta spread left into the cyan edge");
        assert!(at(4, 0, 0) > 0.5, "cyan spread right into the magenta edge");
        // Deep inside each block is unchanged (single ink).
        assert_eq!(at(0, 0, 1), 0.0, "far cyan has no magenta");
        assert_eq!(at(7, 0, 0), 0.0, "far magenta has no cyan");
    }

    #[test]
    fn leaves_paper_untouched() {
        let (w, h, ch) = (6usize, 3usize, 4usize);
        let mut px = vec![0.0f32; w * h * ch];
        // One cyan pixel in the middle, rest paper.
        px[(w + 3) * ch] = 1.0;
        let before = px.clone();
        trap(&mut px, w, h, ch, 1);
        // Paper pixels (no ink) stay paper — no ink bleeds into white.
        for i in 0..w * h {
            if before[i * ch] == 0.0 {
                assert_eq!(px[i * ch], 0.0, "paper pixel {i} gained cyan");
            }
        }
    }

    #[test]
    fn width_zero_is_identity() {
        let (mut px, w, h) = two_blocks();
        let orig = px.clone();
        trap(&mut px, w, h, 4, 0);
        assert_eq!(px, orig);
    }
}

#[cfg(test)]
mod linear_tests {
    use super::*;
    #[test]
    fn all_clipped_windows_match_the_direct_maximum() {
        let mut seed = 1234567u32;
        for len in 1..90 {
            let values: Vec<f32> = (0..len)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0, 0.0, -2.0, 0.25, 1.0, 2.0][seed as usize % 9]
                })
                .collect();
            for radius in [0, 1, 2, 3, 5, 16, 50, 128, usize::MAX] {
                let mut out = vec![0.0; len];
                max_line(&mut out, radius, |i| values[i]);
                for (x, v) in out.iter().enumerate() {
                    let want = (x.saturating_sub(radius)..=x.saturating_add(radius).min(len - 1)).fold(0.0f32, |m, i| m.max(values[i]));
                    assert_eq!(v.to_bits(), want.to_bits(), "len {len} radius {radius} x {x}");
                }
            }
        }
    }
    #[test]
    fn trap_preserves_paper_alpha_and_spare_channels_exactly() {
        for (w, h, ch) in [(1, 31, 5), (47, 1, 4), (37, 29, 4), (37, 29, 5), (23, 19, 8)] {
            for radius in [1, 3, 10, 128] {
                let src: Vec<f32> = (0..w * h * ch).map(|i| if (i / ch) % 5 == 0 { 0.0 } else { ((i * 71 + i / ch * 103) % 65536) as f32 / 65535.0 }).collect();
                let mut want = src.clone();
                for y in 0..h {
                    for x in 0..w {
                        let index = (y * w + x) * ch;
                        if !(0..4).any(|c| src[index + c] > 1e-4) {
                            continue;
                        }
                        for c in 0..4 {
                            let mut m = 0.0f32;
                            for yy in y.saturating_sub(radius)..=y.saturating_add(radius).min(h - 1) {
                                for xx in x.saturating_sub(radius)..=x.saturating_add(radius).min(w - 1) {
                                    m = m.max(src[(yy * w + xx) * ch + c]);
                                }
                            }
                            want[index + c] = src[index + c].max(m);
                        }
                    }
                }
                let mut got = src;
                trap(&mut got, w, h, ch, radius);
                assert_eq!(got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), want.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
            }
        }
    }
    #[test]
    fn malformed_buffer_is_left_alone() {
        let mut pixels = vec![0.5; 20];
        let original = pixels.clone();
        trap(&mut pixels, usize::MAX, 2, 5, 10);
        assert_eq!(pixels, original);
    }
}
