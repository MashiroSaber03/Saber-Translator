//! Mosaic, Emboss, Find Edges, Solarize/Invert (per pixel), Desaturate.

use photocraft_color::PixelFormat;
use photocraft_geom::Rect;
use photocraft_raster::{from_rgba, to_rgba};

use crate::Ctx;
use crate::fxutil::subtractive;
use crate::image::Image;

fn fmt(ctx: &Ctx) -> PixelFormat {
    PixelFormat::new(ctx.mode, photocraft_color::SampleType::F32, ctx.alpha)
}

fn luma(ctx: &Ctx, px: &[f32]) -> f32 {
    let c = to_rgba(&fmt(ctx), px);
    0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
}

/// Replaces the colour with gray `g` in the pixel's own model (alpha kept).
fn set_gray(ctx: &Ctx, px: &mut [f32], g: f32) {
    let a = if ctx.alpha { px[px.len() - 1] } else { 1.0 };
    let v = from_rgba(&fmt(ctx), [g, g, g, a]);
    px.copy_from_slice(&v[..px.len()]);
}

/// Solarize: inverts the highlights. CMYK samples are ink, so there the
/// channels below 50% ink (the light ones) invert.
pub(crate) fn solarize(src: &Image, out: Rect, ctx: &Ctx) -> Vec<f32> {
    if subtractive(ctx) {
        per_pixel(src, out, ctx, |v| if v < 0.5 { 1.0 - v } else { v })
    } else {
        per_pixel(src, out, ctx, |v| if v > 0.5 { 1.0 - v } else { v })
    }
}

/// Colour channels through `f` (alpha untouched).
pub(crate) fn per_pixel(src: &Image, out: Rect, ctx: &Ctx, f: impl Fn(f32) -> f32) -> Vec<f32> {
    let n = src.ch;
    let cc = if ctx.alpha { n - 1 } else { n };
    let mut res = src.crop(out);
    for px in res.chunks_exact_mut(n) {
        for v in px.iter_mut().take(cc) {
            *v = f(*v);
        }
    }
    res
}

pub(crate) fn desaturate(src: &Image, out: Rect, ctx: &Ctx) -> Vec<f32> {
    let n = src.ch;
    let mut res = src.crop(out);
    for px in res.chunks_exact_mut(n) {
        let g = luma(ctx, px);
        set_gray(ctx, px, g);
    }
    res
}

/// Mosaic cells anchored at the bounds' origin; each cell becomes its
/// (premultiplied) average over the part inside the bounds.
pub(crate) fn mosaic(src: &Image, out: Rect, ctx: &Ctx, cell: f32) -> Vec<f32> {
    let n = src.ch;
    let width = out.width() as usize;
    let height = out.height() as usize;
    if width == 0 || height == 0 || n == 0 {
        return Vec::new();
    }
    let cs = cell.max(1.0).round() as i32;
    let b = ctx.bounds;
    let Some(row_stride) = width.checked_mul(n) else { return Vec::new() };
    let Some(len) = row_stride.checked_mul(height) else { return Vec::new() };
    let mut res = vec![0.0f32; len];
    let mut acc = vec![0.0f32; n];
    let cx0 = (out.x0 - b.x0).div_euclid(cs);
    let cx1 = (out.x1 - 1 - b.x0).div_euclid(cs);
    let cy0 = (out.y0 - b.y0).div_euclid(cs);
    let cy1 = (out.y1 - 1 - b.y0).div_euclid(cs);

    // Average each cell once per tile, rather than once per output scanline.
    for cy in cy0..=cy1 {
        for cx in cx0..=cx1 {
            let cell_rect = Rect::new(b.x0 + cx * cs, b.y0 + cy * cs, b.x0 + (cx + 1) * cs, b.y0 + (cy + 1) * cs);
            let r = cell_rect.intersect(&b);
            acc.fill(0.0);
            let mut count = 0.0;
            if n == 4
                && let Some((sum, samples)) = mosaic_sum4(src, r, ctx.alpha)
            {
                acc.copy_from_slice(&sum);
                count = samples;
            } else {
                for yy in r.y0..r.y1 {
                    for xx in r.x0..r.x1 {
                        let a = if ctx.alpha { src.get(xx, yy, n - 1) } else { 1.0 };
                        for (c, v) in acc.iter_mut().enumerate() {
                            let s = src.get(xx, yy, c);
                            *v += if ctx.alpha && c < n - 1 { s * a } else { s };
                        }
                        count += 1.0;
                    }
                }
            }
            if count > 0.0 {
                for v in acc.iter_mut() {
                    *v /= count;
                }
            }
            if ctx.alpha {
                let a = acc[n - 1];
                for v in acc.iter_mut().take(n - 1) {
                    *v = if a > 1e-7 { *v / a } else { 0.0 };
                }
            }

            // Use the full cell for output clipping, preserving partial cells
            // when the requested output extends beyond the averaging bounds.
            let write = cell_rect.intersect(&out);
            let first_row = write.y0.abs_diff(out.y0) as usize;
            let first_pixel = write.x0.abs_diff(out.x0) as usize;
            for row in res.chunks_exact_mut(row_stride).skip(first_row).take(write.height() as usize) {
                for px in row.chunks_exact_mut(n).skip(first_pixel).take(write.width() as usize) {
                    px.copy_from_slice(&acc);
                }
            }
        }
    }
    res
}

/// Read each four-channel pixel once, keeping independent channel sums in registers.
/// Preserve the original row/pixel addition order and alpha arithmetic.
fn mosaic_sum4(src: &Image, r: Rect, alpha: bool) -> Option<([f32; 4], f32)> {
    if r.intersect(&src.rect) != r {
        return None;
    }
    let stride = (src.rect.width() as usize).checked_mul(4)?;
    let offset = (r.x0.abs_diff(src.rect.x0) as usize).checked_mul(4)?;
    let width = (r.width() as usize).checked_mul(4)?;
    let mut sum = [0.0; 4];
    let mut count = 0.0;
    for y in r.y0..r.y1 {
        let start = (y.abs_diff(src.rect.y0) as usize).checked_mul(stride)?.checked_add(offset)?;
        let row = src.data.get(start..start.checked_add(width)?)?;
        for pixel in row.as_chunks::<4>().0 {
            let a = if alpha { pixel[3] } else { 1.0 };
            for c in 0..3 {
                sum[c] += if alpha { pixel[c] * a } else { pixel[c] };
            }
            sum[3] += pixel[3];
            count += 1.0;
        }
    }
    Some((sum, count))
}

/// Emboss: gray relief from luminance differences along the light angle.
pub(crate) fn emboss(src: &Image, out: Rect, ctx: &Ctx, angle: f32, height: f32, amount: f32) -> Vec<f32> {
    let n = src.ch;
    let (s, c) = angle.to_radians().sin_cos();
    let h = height.max(1.0);
    let (dx, dy) = ((c * h).round() as i32, (-s * h).round() as i32);
    let k = amount / 100.0;
    let mut res = src.crop(out);
    let mut tmp = vec![0.0f32; n];
    let lum_at = |x: i32, y: i32, tmp: &mut Vec<f32>| {
        for (ci, t) in tmp.iter_mut().enumerate() {
            *t = src.get(x, y, ci);
        }
        luma(ctx, tmp)
    };
    let w = out.width() as usize;
    for (i, px) in res.chunks_exact_mut(n).enumerate() {
        let (x, y) = (out.x0 + (i % w) as i32, out.y0 + (i / w) as i32);
        let g = 0.5 + k * (lum_at(x + dx, y + dy, &mut tmp) - lum_at(x - dx, y - dy, &mut tmp));
        set_gray(ctx, px, g.clamp(0.0, 1.0));
    }
    res
}

/// Find Edges: Sobel magnitude per colour channel, dark edges on white (CMYK samples are ink, so
/// there the edge strength is the ink itself). At full strength, as Photoshop's: a 60-level step
/// darkens by 240 (Photoshop 27.11; a quarter of that was 4× too pale).
pub(crate) fn find_edges(src: &Image, out: Rect, ctx: &Ctx) -> Vec<f32> {
    let n = src.ch;
    let cc = if ctx.alpha { n - 1 } else { n };
    let sub = subtractive(ctx);
    let mut res = src.crop(out);
    let w = out.width() as usize;
    for (i, px) in res.chunks_exact_mut(n).enumerate() {
        let (x, y) = (out.x0 + (i % w) as i32, out.y0 + (i / w) as i32);
        for (c, v) in px.iter_mut().enumerate().take(cc) {
            let g = |dx: i32, dy: i32| src.get(x + dx, y + dy, c);
            let gx = (g(1, -1) + 2.0 * g(1, 0) + g(1, 1)) - (g(-1, -1) + 2.0 * g(-1, 0) + g(-1, 1));
            let gy = (g(-1, 1) + 2.0 * g(0, 1) + g(1, 1)) - (g(-1, -1) + 2.0 * g(0, -1) + g(1, -1));
            let e = (gx * gx + gy * gy).sqrt().clamp(0.0, 1.0);
            *v = if sub { e } else { 1.0 - e };
        }
    }
    res
}
