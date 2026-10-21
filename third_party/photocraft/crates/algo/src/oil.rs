//! Oil Paint: flow-guided smoothing plus a lit bristle relief.
//!
//! The approach follows published non-photorealistic rendering work rather
//! than any product: the structure tensor of the luminance (Brox et al.) gives
//! a smooth stroke direction field, and a line integral convolution (Cabral &
//! Leedom 1993) smears colour along it, as in flow-based abstraction (Kyprianidis
//! & Döllner 2008). Convolving white noise along the same field produces the
//! bristle streaks, which together with the smoothed luminance form a height
//! map shaded by a directional light.

use photocraft_color::ColorMode;
use photocraft_geom::Rect;

use crate::Ctx;
use crate::fxutil::{MAXC, gauss_blur_n, ncol, premul_window, unpremul_px};
use crate::image::Image;
use crate::noise::hash01;

pub(crate) struct OilSpec {
    pub stylization: f32,
    pub cleanliness: f32,
    pub scale: f32,
    pub bristle_detail: f32,
    pub lighting: bool,
    pub angle: f32,
    pub shine: f32,
}

fn stroke_len(stylization: f32) -> f32 {
    1.5 + stylization.clamp(0.1, 10.0) * 2.0
}
fn tensor_sigma(scale: f32) -> f32 {
    0.8 + scale.clamp(0.1, 10.0) * 0.6
}

/// Pixels read beyond an output tile.
pub(crate) fn reach(stylization: f32, scale: f32) -> f32 {
    stroke_len(stylization) + 3.0 * tensor_sigma(scale) + 3.0 * 2.5 + 4.0
}

pub(crate) fn oil_paint(src: &Image, out: Rect, ctx: &Ctx, spec: &OilSpec) -> Vec<f32> {
    let n = src.ch;
    let win = src.rect;
    let (ww, wh) = (win.width() as usize, win.height() as usize);
    let alpha = ctx.alpha;
    let cc = ncol(ctx, n);
    // 1. Premultiplied colour, cleaned (pre-smoothed) by `cleanliness`.
    let mut p = premul_window(src, win, alpha);
    gauss_blur_n(&mut p, ww, wh, n, spec.cleanliness.clamp(0.0, 10.0) * 0.25);
    // 2. Luminance-like intensity for the structure tensor (channel mean works in every model).
    let lum: Vec<f32> = p.chunks_exact(n).map(|px| px[..cc].iter().sum::<f32>() / cc as f32).collect();
    // 3. Smoothed structure tensor (E, F, G).
    let mut efg = vec![0.0f32; ww * wh * 3];
    for y in 1..wh.saturating_sub(1) {
        for x in 1..ww - 1 {
            let l = |dx: isize, dy: isize| lum[((y as isize + dy) as usize) * ww + (x as isize + dx) as usize];
            let gx = (l(1, -1) + 2.0 * l(1, 0) + l(1, 1)) - (l(-1, -1) + 2.0 * l(-1, 0) + l(-1, 1));
            let gy = (l(-1, 1) + 2.0 * l(0, 1) + l(1, 1)) - (l(-1, -1) + 2.0 * l(0, -1) + l(1, -1));
            let o = (y * ww + x) * 3;
            efg[o] = gx * gx;
            efg[o + 1] = gx * gy;
            efg[o + 2] = gy * gy;
        }
    }
    gauss_blur_n(&mut efg, ww, wh, 3, tensor_sigma(spec.scale));
    // 4. Stroke direction = minor eigenvector (along edges).
    let tangent: Vec<(f32, f32)> = efg
        .as_chunks::<3>()
        .0
        .iter()
        .map(|t| {
            let (e, f, g) = (t[0], t[1], t[2]);
            let l1 = 0.5 * (e + g + ((e - g) * (e - g) + 4.0 * f * f).sqrt());
            let (tx, ty) = (l1 - e, -f);
            let m = (tx * tx + ty * ty).sqrt();
            // Near-isotropic structure has no stable direction; treat as flat.
            if m < 1e-6 || (l1 - (e + g - l1)) < 1e-6 {
                // Flat areas: gentle diagonal strokes.
                (std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2)
            } else {
                (tx / m, ty / m)
            }
        })
        .collect();
    // 5. Line integral convolution over `out` grown by 1 (for the relief gradient).
    let r1 = out.inflate(1).intersect(&win);
    let (rw, rh) = (r1.width() as usize, r1.height() as usize);
    let len = stroke_len(spec.stylization);
    let h = (len / 10.0).max(1.0);
    let steps = (len / h).ceil() as i32;
    let mut col = vec![0.0f32; rw * rh * n];
    let mut bristle = vec![0.0f32; rw * rh];
    let sample = |x: f32, y: f32, acc: &mut [f32], k: f32| {
        let x = x.clamp(0.0, (ww - 1) as f32);
        let y = y.clamp(0.0, (wh - 1) as f32);
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(ww - 1), (y0 + 1).min(wh - 1));
        let (ax, ay) = (x - x0 as f32, y - y0 as f32);
        let w00 = (1.0 - ax) * (1.0 - ay) * k;
        let w10 = ax * (1.0 - ay) * k;
        let w01 = (1.0 - ax) * ay * k;
        let w11 = ax * ay * k;
        let (i00, i10, i01, i11) = ((y0 * ww + x0) * n, (y0 * ww + x1) * n, (y1 * ww + x0) * n, (y1 * ww + x1) * n);
        for c in 0..acc.len() {
            acc[c] += p[i00 + c] * w00 + p[i10 + c] * w10 + p[i01 + c] * w01 + p[i11 + c] * w11;
        }
    };
    let mut acc = [0.0f32; MAXC];
    for ry in 0..rh {
        for rx in 0..rw {
            let (dx, dy) = ((r1.x0 - win.x0) as f32 + rx as f32, (r1.y0 - win.y0) as f32 + ry as f32);
            acc[..n].fill(0.0);
            let mut wsum = 0.0;
            let mut nsum = 0.0;
            sample(dx, dy, &mut acc[..n], 1.0);
            wsum += 1.0;
            nsum += hash01(win.x0 + dx as i32, win.y0 + dy as i32, 71, 0);
            for dir in [1.0f32, -1.0] {
                let (mut qx, mut qy) = (dx, dy);
                let (mut px, mut py) = (0.0f32, 0.0f32);
                for k in 1..=steps {
                    let ti = (qy.round().clamp(0.0, (wh - 1) as f32) as usize) * ww + qx.round().clamp(0.0, (ww - 1) as f32) as usize;
                    let (mut tx, mut ty) = tangent[ti];
                    if k == 1 {
                        tx *= dir;
                        ty *= dir;
                    } else if tx * px + ty * py < 0.0 {
                        tx = -tx;
                        ty = -ty;
                    }
                    px = tx;
                    py = ty;
                    qx += tx * h;
                    qy += ty * h;
                    let wk = 1.0 - k as f32 / (steps + 1) as f32;
                    sample(qx, qy, &mut acc[..n], wk);
                    wsum += wk;
                    nsum += wk * hash01(win.x0 + qx.round() as i32, win.y0 + qy.round() as i32, 71, 0);
                }
            }
            let o = (ry * rw + rx) * n;
            for c in 0..n {
                col[o + c] = acc[c] / wsum;
            }
            bristle[ry * rw + rx] = nsum / wsum - 0.5;
        }
    }
    // 6. Relief lighting.
    let (ls, lc) = spec.angle.to_radians().sin_cos();
    let elev = 45f32.to_radians();
    let light = [lc * elev.cos(), -ls * elev.cos(), elev.sin()];
    let half = {
        let v = [light[0], light[1], light[2] + 1.0];
        let m = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / m, v[1] / m, v[2] / m]
    };
    let amp = spec.bristle_detail.clamp(0.0, 10.0) / 10.0;
    let shine = spec.shine.clamp(0.0, 10.0) / 10.0;
    let sub = ctx.mode == ColorMode::Cmyk;
    let lab = ctx.mode == ColorMode::Lab;
    let height = |rx: usize, ry: usize| -> f32 {
        let o = (ry * rw + rx) * n;
        let l = col[o..o + cc].iter().sum::<f32>() / cc as f32;
        let l = if sub { 1.0 - l } else { l };
        l * 2.0 + bristle[ry * rw + rx] * amp * 6.0
    };
    let flat_spec = half[2].powf(40.0);
    let mut res = Vec::with_capacity(out.width() as usize * out.height() as usize * n);
    for y in out.y0..out.y1 {
        for x in out.x0..out.x1 {
            let (rx, ry) = ((x - r1.x0) as usize, (y - r1.y0) as usize);
            let o = (ry * rw + rx) * n;
            let mut px = [0.0f32; MAXC];
            px[..n].copy_from_slice(&col[o..o + n]);
            unpremul_px(&mut px[..n], alpha);
            // Painting restyles colour, not coverage: keep the layer's own alpha (no faded
            // canvas edges from smearing in transparent margins).
            if alpha {
                px[n - 1] = src.get(x, y, n - 1);
            }
            if spec.lighting {
                let hx = height((rx + 1).min(rw - 1), ry) - height(rx.saturating_sub(1), ry);
                let hy = height(rx, (ry + 1).min(rh - 1)) - height(rx, ry.saturating_sub(1));
                let nv = [-hx * 0.5, -hy * 0.5, 1.0];
                let m = (nv[0] * nv[0] + nv[1] * nv[1] + 1.0).sqrt();
                let ndl = (nv[0] * light[0] + nv[1] * light[1] + nv[2] * light[2]) / m;
                let ndh = ((nv[0] * half[0] + nv[1] * half[1] + nv[2] * half[2]) / m).max(0.0);
                let shade = 1.0 + (ndl - light[2]) * 1.2;
                let spec_v = shine * (ndh.powf(40.0) - flat_spec).max(0.0) * 0.8;
                let chans = if lab { 1 } else { cc };
                for v in px.iter_mut().take(chans) {
                    *v = if sub { 1.0 - ((1.0 - *v) * shade + spec_v) } else { *v * shade + spec_v };
                }
            }
            res.extend_from_slice(&px[..n]);
        }
    }
    res
}
