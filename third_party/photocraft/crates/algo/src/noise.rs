//! Add Noise, Median, Dust & Scratches.

use photocraft_geom::Rect;

use crate::image::Image;
use crate::{Ctx, Distribution};

/// Deterministic hash → `[0, 1)` from document coordinates, so results do not
/// depend on tiling.
#[inline]
pub(crate) fn hash01(x: i32, y: i32, c: u32, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343) ^ (y as u32).wrapping_mul(0xd816_3841) ^ c.wrapping_mul(0xcb1a_b31f) ^ seed.wrapping_mul(0x9e37_79b9);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^= h >> 16;
    (h >> 8) as f32 / (1u32 << 24) as f32
}

/// Amount 100 % spans ±50 % of the range (uniform); Gaussian uses the same
/// amount as ~2σ.
#[allow(clippy::too_many_arguments)]
pub(crate) fn add(src: &Image, out: Rect, ctx: &Ctx, amount: f32, dist: Distribution, mono: bool, seed: u32) -> Vec<f32> {
    let n = src.ch;
    let cc = if ctx.alpha { n - 1 } else { n };
    let a = amount.max(0.0) / 100.0 * 0.5;
    let mut res = src.crop(out);
    let w = out.width() as usize;
    for (i, px) in res.chunks_exact_mut(n).enumerate() {
        if ctx.alpha && px[n - 1] <= 0.0 {
            continue;
        }
        let (x, y) = (out.x0 + (i % w) as i32, out.y0 + (i / w) as i32);
        for (c, pv) in px.iter_mut().enumerate().take(cc) {
            let ch = if mono { 0 } else { c as u32 };
            let v = match dist {
                Distribution::Uniform => (hash01(x, y, ch, seed) - 0.5) * 2.0 * a,
                Distribution::Gaussian => {
                    let u1 = hash01(x, y, ch * 2 + 101, seed).max(1e-7);
                    let u2 = hash01(x, y, ch * 2 + 102, seed);
                    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos() * a * 0.5
                }
            };
            // Integer storage clamps on write; float surfaces keep the value.
            *pv += v;
        }
    }
    res
}

/// Median over a disc of `radius`; with `threshold`, a pixel is replaced only
/// when it differs from the median by more than `threshold` levels.
pub(crate) fn median(src: &Image, out: Rect, radius: f32, threshold: Option<f32>) -> Vec<f32> {
    let r = radius.max(0.0).round() as i32;
    if r == 0 {
        return src.crop(out);
    }
    let t = threshold.map(|t| t / 255.0);
    if r >= 2 {
        return median_ranked(src, out, r, t);
    }
    median_direct(src, out, r, t)
}

/// Half-width of the median disc on each row `dy` in `-r..=r`: the largest `dx` with
/// `dx^2 + dy^2 <= r^2 + r`.
fn disc_rows(r: i32) -> Vec<i32> {
    (-r..=r)
        .map(|dy| {
            let lim = r * r + r - dy * dy;
            let mut hw = (lim as f64).sqrt() as i32;
            while hw * hw > lim {
                hw -= 1;
            }
            while (hw + 1) * (hw + 1) <= lim {
                hw += 1;
            }
            hw
        })
        .collect()
}

/// [`median`] with a sliding histogram, at any depth: every sample of a channel's window is
/// replaced by its rank among the window's distinct values (its level for 8-bit samples), so the
/// histogram of the disc slides from pixel to pixel, removing one end and adding the other end of
/// each disc row (or column) per step, and the median rank is tracked as it moves (Huang). Ranks are counted
/// in fine bins and in blocks of bins, so the median crosses empty stretches of a large rank range
/// a block at a time. A pixel costs O(radius) instead of a selection over O(radius^2) samples, and
/// the value picked is the sample the selection picks (ranks follow `total_cmp`), so the output is
/// identical.
fn median_ranked(src: &Image, out: Rect, r: i32, t: Option<f32>) -> Vec<f32> {
    let n = src.ch;
    let win = out.inflate(r);
    let (ww, wh) = (win.width() as usize, win.height() as usize);
    let rows = disc_rows(r);
    let count: usize = rows.iter().map(|&hw| (2 * hw + 1) as usize).sum();
    let mid = count / 2;
    let (ow, oh) = (out.width() as usize, out.height() as usize);
    let ru = r as usize;
    let mut res = vec![0.0f32; ow * oh * n];
    let mut plane = vec![0u32; ww * wh];
    let mut values = Vec::new();
    let (mut hist, mut blocks) = (Vec::new(), Vec::new());
    for c in 0..n {
        // Window samples of this channel; outside the source reads as 0, as `Image::get` does.
        values.clear();
        values.extend((win.y0..win.y1).flat_map(|y| (win.x0..win.x1).map(move |x| (x, y))).map(|(x, y)| src.get(x, y, c)));
        let eight_bit = values.iter().all(|&v| {
            let k = (v * 255.0).round();
            (0.0..=255.0).contains(&k) && (k / 255.0).to_bits() == v.to_bits()
        });
        let levels: Vec<f32> = if eight_bit {
            for (p, &v) in plane.iter_mut().zip(&values) {
                *p = (v * 255.0).round() as u32;
            }
            (0..256).map(|k| k as f32 / 255.0).collect()
        } else {
            let mut sorted = values.clone();
            sorted.sort_unstable_by(|a, b| a.total_cmp(b));
            sorted.dedup_by(|a, b| a.to_bits() == b.to_bits());
            for (p, v) in plane.iter_mut().zip(&values) {
                *p = sorted.partition_point(|s| s.total_cmp(v).is_lt()) as u32;
            }
            sorted
        };
        let k = levels.len();
        // Block size: about sqrt(k) bins, a power of two.
        let shift = (usize::BITS - k.isqrt().max(1).leading_zeros()) as usize;
        let bs = 1usize << shift;
        hist.clear();
        hist.resize(k, 0u32);
        blocks.clear();
        blocks.resize(k.div_ceil(bs), 0u32);
        hist.fill(0);
        blocks.fill(0);
        // Disc of output pixel (0, 0), centred on window pixel (r, r).
        for (j, &hw) in rows.iter().enumerate() {
            for &q in &plane[j * ww + ru - hw as usize..=j * ww + ru + hw as usize] {
                hist[q as usize] += 1;
                blocks[q as usize >> shift] += 1;
            }
        }
        // `m` is the median rank and `below` the number of samples ranked under it.
        let (mut m, mut below) = (0usize, 0usize);
        // The disc visits the rows in a serpentine (left to right, down one, right to left, ...),
        // so every move swaps one end of each disc row or column and the histogram is never
        // rebuilt. The disc is symmetric, so a column's half-height is the row's half-width.
        for oy in 0..oh {
            for step in 0..ow {
                let ox = if oy % 2 == 0 { step } else { ow - 1 - step };
                // Window pixel at the disc's centre, and the two ends of each row or column.
                let (cx, cy) = (ox + ru, oy + ru);
                let moves = rows.iter().enumerate().map(|(j, &hw)| {
                    let hw = hw as usize;
                    if step == 0 {
                        // Down from (cx, cy - 1): column cx - r + j loses its top, gains a bottom.
                        let x = cx + j - ru;
                        ((cy - 1 - hw) * ww + x, (cy + hw) * ww + x)
                    } else if oy % 2 == 0 {
                        // Right from (cx - 1, cy).
                        let y = (cy + j - ru) * ww;
                        (y + cx - 1 - hw, y + cx + hw)
                    } else {
                        // Left from (cx + 1, cy).
                        let y = (cy + j - ru) * ww;
                        (y + cx + 1 + hw, y + cx - hw)
                    }
                });
                if oy > 0 || step > 0 {
                    for (gone, new) in moves {
                        let (gone, new) = (plane[gone] as usize, plane[new] as usize);
                        hist[gone] -= 1;
                        blocks[gone >> shift] -= 1;
                        hist[new] += 1;
                        blocks[new >> shift] += 1;
                        below -= usize::from(gone < m);
                        below += usize::from(new < m);
                    }
                }
                while below > mid {
                    if m & (bs - 1) == 0 && below - blocks[(m >> shift) - 1] as usize > mid {
                        below -= blocks[(m >> shift) - 1] as usize;
                        m -= bs;
                    } else {
                        m -= 1;
                        below -= hist[m] as usize;
                    }
                }
                loop {
                    if m & (bs - 1) == 0 && below + blocks[m >> shift] as usize <= mid {
                        below += blocks[m >> shift] as usize;
                        m += bs;
                    } else if below + hist[m] as usize <= mid {
                        below += hist[m] as usize;
                        m += 1;
                    } else {
                        break;
                    }
                }
                let mv = levels[m];
                let o = src.get(out.x0 + ox as i32, out.y0 + oy as i32, c);
                res[(oy * ow + ox) * n + c] = match t {
                    Some(t) if (o - mv).abs() <= t => o,
                    _ => mv,
                };
            }
        }
    }
    res
}

/// [`median`] by selecting the middle of every disc's samples.
fn median_direct(src: &Image, out: Rect, r: i32, t: Option<f32>) -> Vec<f32> {
    let n = src.ch;
    let offs: Vec<(i32, i32)> = (-r..=r).flat_map(|dy| (-r..=r).map(move |dx| (dx, dy))).filter(|(dx, dy)| dx * dx + dy * dy <= r * r + r).collect();
    let mut vals = Vec::with_capacity(offs.len());
    let mut res = Vec::with_capacity(out.width() as usize * out.height() as usize * n);
    for y in out.y0..out.y1 {
        for x in out.x0..out.x1 {
            for c in 0..n {
                vals.clear();
                vals.extend(offs.iter().map(|(dx, dy)| src.get(x + dx, y + dy, c)));
                let mid = vals.len() / 2;
                let (_, m, _) = vals.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
                let m = *m;
                let o = src.get(x, y, c);
                res.push(match t {
                    Some(t) if (o - m).abs() <= t => o,
                    _ => m,
                });
            }
        }
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Random samples: 8-bit levels, 16-bit levels, or floats with ties, signed zeros, values
    /// outside 0..1 and a NaN.
    fn noisy(rect: Rect, kind: u32) -> Image {
        let mut img = Image::new(rect, 4);
        let mut s = 0x2545_f491u32 ^ kind;
        for (i, v) in img.data.iter_mut().enumerate() {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *v = match kind {
                // Full range in one half, a few levels (many ties) in the other.
                0 if i % 2 == 0 => (s % 256) as f32 / 255.0,
                0 => (s % 4 * 85) as f32 / 255.0,
                1 => (s % 65536) as f32 / 65535.0,
                _ => [0.0, -0.0, 0.25, 1.5, -2.0, (s % 1000) as f32 / 997.0][(s % 6) as usize],
            };
        }
        if kind == 2 {
            img.data[100] = f32::NAN;
        }
        img
    }

    #[test]
    fn median_histogram_matches_the_direct_selection() {
        let src_rect = Rect::new(-10, -8, 36, 30);
        // The output reaches past the source edge, where samples read as 0.
        let out = Rect::new(-14, -3, 30, 34);
        for kind in 0..3 {
            let img = noisy(src_rect, kind);
            for r in [2, 3, 7, 16] {
                for t in [None, Some(0.0), Some(10.0 / 255.0), Some(1.0)] {
                    let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
                    let fast = bits(median_ranked(&img, out, r, t));
                    assert_eq!(fast, bits(median_direct(&img, out, r, t)), "kind {kind} r {r} threshold {t:?}");
                }
            }
        }
    }
}
