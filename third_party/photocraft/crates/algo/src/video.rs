//! Filter › Video: De-Interlace and NTSC Colors.

use photocraft_color::ColorMode;
use photocraft_geom::Rect;

use crate::Ctx;
use crate::fxutil::{MAXC, rgba, set_rgba, xy};
use crate::image::Image;

/// De-Interlace: removes one field (odd = lines 1, 3, 5… counted from the
/// bounds' top, i.e. even offsets) and rebuilds it from the neighbouring
/// kept lines by duplication or (premultiplied) interpolation.
pub(crate) fn deinterlace(src: &Image, out: Rect, ctx: &Ctx, eliminate_even: bool, interpolate: bool) -> Vec<f32> {
    let n = src.ch;
    let b = ctx.bounds;
    let mut res = src.crop(out);
    let mut acc = [0.0f32; MAXC];
    for (i, px) in res.chunks_exact_mut(n).enumerate() {
        let (x, y) = xy(out, i);
        if !b.contains(x, y) {
            continue;
        }
        // 1-based line number parity: offset 0 is line 1 (odd).
        let odd_line = (y - b.y0) % 2 == 0;
        if odd_line == eliminate_even {
            continue;
        }
        let above = (y > b.y0).then_some(y - 1);
        let below = (y + 1 < b.y1).then_some(y + 1);
        let (rows, cnt) = match (interpolate, above, below) {
            (true, Some(a), Some(c)) => ([a, c], 2),
            (_, Some(a), _) => ([a, a], 1),
            (_, None, Some(c)) => ([c, c], 1),
            _ => continue,
        };
        acc[..n].fill(0.0);
        let k = 1.0 / cnt as f32;
        for &yy in &rows[..cnt] {
            let a = if ctx.alpha { src.get(x, yy, n - 1) } else { 1.0 };
            for (c, v) in acc.iter_mut().enumerate().take(n) {
                let s = src.get(x, yy, c);
                *v += if ctx.alpha && c < n - 1 { s * a } else { s } * k;
            }
        }
        crate::fxutil::unpremul_px(&mut acc[..n], ctx.alpha);
        px.copy_from_slice(&acc[..n]);
    }
    res
}

/// Composite-signal limits (fractions of the 7.5–100 IRE luma range): broadcast-safe
/// is −20…120 IRE, i.e. Y ± C within these bounds.
const NTSC_MAX: f32 = (120.0 - 7.5) / 92.5;
const NTSC_MIN: f32 = (-20.0 - 7.5) / 92.5;

/// NTSC Colors: scales chroma (in YIQ) so the composite signal Y ± C stays
/// within broadcast limits; hue and luma are kept.
pub(crate) fn ntsc(src: &Image, out: Rect, ctx: &Ctx) -> Vec<f32> {
    let n = src.ch;
    let native = ctx.mode == ColorMode::Rgb;
    let mut res = src.crop(out);
    for px in res.chunks_exact_mut(n) {
        let c = if native { [px[0], px[1], px[2], 1.0] } else { rgba(ctx, px) };
        let yy = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
        let ii = 0.595_716 * c[0] - 0.274_453 * c[1] - 0.321_263 * c[2];
        let qq = 0.211_456 * c[0] - 0.522_591 * c[1] + 0.311_135 * c[2];
        let chroma = (ii * ii + qq * qq).sqrt();
        if chroma <= 1e-7 {
            continue;
        }
        let mut k: f32 = 1.0;
        if yy + chroma > NTSC_MAX {
            k = k.min(((NTSC_MAX - yy) / chroma).max(0.0));
        }
        if yy - chroma < NTSC_MIN {
            k = k.min(((yy - NTSC_MIN) / chroma).max(0.0));
        }
        if k >= 1.0 {
            continue;
        }
        let (i2, q2) = (ii * k, qq * k);
        let rgb = [yy + 0.956_3 * i2 + 0.621_0 * q2, yy - 0.272_1 * i2 - 0.647_4 * q2, yy - 1.107_0 * i2 + 1.704_6 * q2];
        if native {
            px[..3].copy_from_slice(&rgb);
        } else {
            set_rgba(ctx, px, [rgb[0], rgb[1], rgb[2], c[3]]);
        }
    }
    res
}
