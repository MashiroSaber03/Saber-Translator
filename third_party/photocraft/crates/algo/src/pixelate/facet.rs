//! Reuse each 3x3 quadrant at its four output pixels. Keep the original
//! nine-sample addition order: reassociating the sums can change variance ties.
use crate::fxutil::{MAXC, luma};
use crate::{Ctx, image::Image};
use photocraft_geom::Rect;

#[derive(Clone, Copy, Default)]
struct Quadrant {
    variance: f32,
    mean: [f32; MAXC],
}

pub(super) fn run(src: &Image, out: Rect, ctx: &Ctx) -> Option<Vec<f32>> {
    let n = src.ch;
    if n == 0 || n > MAXC || out.is_empty() {
        return None;
    }
    let win = Rect::new(out.x0.checked_sub(2)?, out.y0.checked_sub(2)?, out.x1.checked_add(2)?, out.y1.checked_add(2)?);
    if win.intersect(&src.rect) != win {
        return None;
    }
    let (w, h) = (win.width() as usize, win.height() as usize);
    let len = w.checked_mul(h)?;
    // A normal 256px tile needs 264 KiB. Bound scratch for explicit tile overrides.
    if len > 1024 * 1024 {
        return None;
    }
    let stride = (src.rect.width() as usize).checked_mul(n)?;
    let offset = (win.x0.abs_diff(src.rect.x0) as usize).checked_mul(n)?;
    let row = |y: i32| -> Option<&[f32]> {
        let start = (y.abs_diff(src.rect.y0) as usize).checked_mul(stride)?.checked_add(offset)?;
        src.data.get(start..start.checked_add(w.checked_mul(n)?)?)
    };
    let mut lum = Vec::new();
    lum.try_reserve_exact(len).ok()?;
    for y in win.y0..win.y1 {
        lum.extend(row(y)?.chunks_exact(n).map(|px| luma(ctx, px)));
    }
    let qw = w.checked_sub(2)?;
    let mut quadrants: [Vec<Quadrant>; 3] = std::array::from_fn(|_| Vec::new());
    for line in &mut quadrants {
        line.try_reserve_exact(qw).ok()?;
        line.resize(qw, Quadrant::default());
    }
    let build_row = |y: usize, line: &mut [Quadrant]| -> Option<()> {
        let rows = [
            row(win.y0.checked_add(i32::try_from(y).ok()?)?)?,
            row(win.y0.checked_add(i32::try_from(y + 1).ok()?)?)?,
            row(win.y0.checked_add(i32::try_from(y + 2).ok()?)?)?,
        ];
        for (x, q) in line.iter_mut().enumerate() {
            let mut sum = [0.0; MAXC];
            let (mut s, mut s2) = (0.0, 0.0);
            for (dy, pixels) in rows.iter().enumerate() {
                let start = x.checked_mul(n)?;
                let samples = pixels.get(start..start.checked_add(3 * n)?)?;
                let lights = lum.get((y + dy) * w + x..(y + dy) * w + x + 3)?;
                for (pixel, &l) in samples.chunks_exact(n).zip(lights) {
                    for (total, value) in sum.iter_mut().zip(pixel) {
                        *total += *value;
                    }
                    s += l;
                    s2 += l * l;
                }
            }
            for v in &mut sum {
                *v /= 9.0;
            }
            *q = Quadrant { variance: s2 / 9.0 - (s / 9.0).powi(2), mean: sum };
        }
        Some(())
    };
    build_row(0, &mut quadrants[0])?;
    build_row(1, &mut quadrants[1])?;
    let width = out.width() as usize;
    let height = out.height() as usize;
    let mut result = Vec::new();
    result.try_reserve_exact(width.checked_mul(height)?.checked_mul(n)?).ok()?;
    for y in 0..height {
        build_row(y + 2, &mut quadrants[2])?;
        for x in 0..width {
            let mut best = Quadrant { variance: f32::MAX, mean: [0.0; MAXC] };
            // Preserve quadrant order, including exact ties and non-finite variances.
            for q in [quadrants[0].get(x)?, quadrants[0].get(x + 2)?, quadrants[2].get(x)?, quadrants[2].get(x + 2)?] {
                if q.variance < best.variance {
                    best = *q;
                }
            }
            result.extend(best.mean.iter().take(n));
        }
        quadrants.rotate_left(1);
    }
    Some(result)
}
