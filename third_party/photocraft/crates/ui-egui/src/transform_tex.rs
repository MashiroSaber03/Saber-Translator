//! Textures for the Free Transform live preview (#91).
//!
//! The moving pixels are uploaded at full resolution (up to the GPU's texture limit) and drawn on
//! a mesh through the box's homography. What the preview samples depends on how big a source
//! pixel ends up on screen (view zoom × transform scale):
//!
//! - shrinking (< ½ screen px per source px): a box-filtered level of a lazily built mip chain, so
//!   zoomed-out previews neither shimmer nor alias;
//! - around 1:1 or enlarging: the full-resolution level, bilinear;
//! - Nearest Neighbor interpolation, or a view zoomed in to ≥ 200% on pixels that aren't
//!   enlarged by the transform: the full-resolution level with nearest filtering, so pixels stay
//!   crisp like the canvas around them.
//!
//! Levels and filter variants are uploaded on first use and cached for the session.

use std::sync::{Arc, Mutex, PoisonError};

use egui::{Color32, ColorImage, TextureHandle, TextureId, TextureOptions};

/// One uploaded texture: its mip level, filter and the uv extent covering the source.
struct Level {
    level: u32,
    nearest: bool,
    tex: TextureHandle,
    uv: [f32; 2],
}

pub struct PreviewTextures {
    name: String,
    /// CPU copies of the levels built so far (level 0 = full resolution), premultiplied.
    images: Mutex<Vec<Arc<ColorImage>>>,
    /// uv extent of level 0 that covers the source exactly (< 1 when level 0 was padded).
    base_uv: [f32; 2],
    uploaded: Mutex<Vec<Level>>,
}

/// A source pixel shown at this many screen pixels or more counts as "zoomed in" (nearest).
const CRISP_ZOOM: f32 = 2.0;

impl PreviewTextures {
    /// `image` covers the source with `base_uv` of its extent; `name` keys the texture in egui.
    pub fn new(ctx: &egui::Context, name: String, image: ColorImage, base_uv: [f32; 2]) -> Self {
        let image = Arc::new(image);
        let tex = ctx.load_texture(format!("{name}-0-linear"), Arc::clone(&image), TextureOptions::LINEAR);
        let base_uv = [base_uv[0].clamp(0.0, 1.0), base_uv[1].clamp(0.0, 1.0)];
        Self { name, images: Mutex::new(vec![image]), base_uv, uploaded: Mutex::new(vec![Level { level: 0, nearest: false, tex, uv: base_uv }]) }
    }

    /// Size of the full-resolution level in texels.
    pub fn size(&self) -> [usize; 2] {
        self.images.lock().unwrap_or_else(PoisonError::into_inner).first().map_or([0, 0], |i| i.size)
    }

    /// The level and filter for a source pixel shown `screen_px` screen pixels wide, where the
    /// transform itself scales by `scale` and the view zoom is `zoom`.
    pub fn choose(zoom: f32, scale: f32, nearest_interpolation: bool) -> (u32, bool) {
        let s = zoom * scale;
        if !s.is_finite() || s <= 0.0 {
            return (0, false);
        }
        // Nearest Neighbor is point sampling: downsampled mip levels have already averaged
        // neighbouring texels, so using one would blur a pixel-art transform preview.
        if nearest_interpolation {
            return (0, true);
        }
        if s < 0.5 {
            // log2(1/s) levels down, at most 12 (4096× smaller).
            return ((1.0 / s).log2().floor().clamp(0.0, 12.0) as u32, false);
        }
        let crisp = zoom >= CRISP_ZOOM && scale <= 1.05;
        (0, crisp)
    }

    /// Texture and uv extent to draw with (uploading a level / filter on first use).
    pub fn pick(&self, ctx: &egui::Context, zoom: f32, scale: f32, nearest_interpolation: bool) -> (TextureId, [f32; 2]) {
        let (level, nearest) = Self::choose(zoom, scale, nearest_interpolation);
        let mut up = self.uploaded.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(l) = up.iter().find(|l| l.level == level && l.nearest == nearest) {
            return (l.tex.id(), l.uv);
        }
        let Some((img, uv)) = self.level_image(level) else {
            // Can't build it (shouldn't happen): fall back to the first upload.
            return up.first().map_or((TextureId::default(), [1.0, 1.0]), |l| (l.tex.id(), l.uv));
        };
        let opts = if nearest { TextureOptions::NEAREST } else { TextureOptions::LINEAR };
        let tex = ctx.load_texture(format!("{}-{level}-{}", self.name, if nearest { "nearest" } else { "linear" }), img, opts);
        let id = tex.id();
        up.push(Level { level, nearest, tex, uv });
        (id, uv)
    }

    /// Level `n` of the mip chain (built on demand by 2×2 box filtering) and its uv extent.
    pub(crate) fn level_image(&self, n: u32) -> Option<(Arc<ColorImage>, [f32; 2])> {
        let mut imgs = self.images.lock().unwrap_or_else(PoisonError::into_inner);
        while imgs.len() <= n as usize {
            let prev = imgs.last()?;
            if prev.size[0] <= 1 && prev.size[1] <= 1 {
                break;
            }
            let next = Arc::new(halve(prev));
            imgs.push(next);
        }
        let k = (imgs.len() - 1).min(n as usize);
        let img = imgs.get(k)?.clone();
        let base = imgs.first()?.size;
        // Level k covers size·2^k base texels (the last row/column may be padding).
        let f = (1u64 << k) as f32;
        let uv = [self.base_uv[0] * base[0] as f32 / (img.size[0] as f32 * f).max(1.0), self.base_uv[1] * base[1] as f32 / (img.size[1] as f32 * f).max(1.0)];
        Some((img, [uv[0].min(1.0), uv[1].min(1.0)]))
    }
}

/// 2×2 box filter of a premultiplied image (odd edges repeat the last row/column).
fn halve(img: &ColorImage) -> ColorImage {
    let [w, h] = img.size;
    let (nw, nh) = (w.div_ceil(2).max(1), h.div_ceil(2).max(1));
    let mut out = vec![Color32::TRANSPARENT; nw * nh];
    let px = |x: usize, y: usize| img.pixels.get(y.min(h.saturating_sub(1)) * w + x.min(w.saturating_sub(1))).copied().unwrap_or_default();
    for (y, row) in out.chunks_mut(nw).enumerate() {
        for (x, o) in row.iter_mut().enumerate() {
            let q = [px(2 * x, 2 * y), px(2 * x + 1, 2 * y), px(2 * x, 2 * y + 1), px(2 * x + 1, 2 * y + 1)];
            let avg = |c: fn(&Color32) -> u8| ((q.iter().map(|p| u32::from(c(p))).sum::<u32>() + 2) / 4) as u8;
            *o = Color32::from_rgba_premultiplied(avg(Color32::r), avg(Color32::g), avg(Color32::b), avg(Color32::a));
        }
    }
    ColorImage::new([nw, nh], out)
}

/// Reads `b` of `surf` as premultiplied texels, at full resolution when it fits in `max_side`
/// (else box-filtered down by the smallest integer factor that fits). Returns the image and the
/// uv extent of it that covers `b`.
/// A layer mask applied to the preview texels: the texel's alpha is scaled by
/// `1 − density × (1 − value)`, as the compositor applies a pixel mask.
pub struct PreviewMask {
    pub surface: photocraft_raster::Surface,
    pub density: f32,
}

pub fn read_surface(surf: &photocraft_raster::Surface, mask: Option<&PreviewMask>, b: photocraft_geom::Rect, max_side: usize) -> (ColorImage, [f32; 2]) {
    let (w, h) = (b.width() as usize, b.height() as usize);
    if w == 0 || h == 0 {
        return (ColorImage::new([1, 1], vec![Color32::TRANSPARENT]), [1.0, 1.0]);
    }
    let k = w.max(h).div_ceil(max_side.max(1)).max(1);
    let (tw, th) = (w.div_ceil(k), h.div_ceil(k));
    let mut px = vec![Color32::TRANSPARENT; tw * th];
    // One band of `k` source rows per output row, in parallel on native targets.
    let band = |ty: usize, out: &mut [Color32]| {
        let y0 = b.y0 + (ty * k) as i32;
        let y1 = (y0 + k as i32).min(b.y1);
        let rows = (y1 - y0).max(0) as usize;
        let mut buf = vec![[0u8; 4]; w * rows];
        surf.read_rgba8_into(photocraft_geom::Rect::new(b.x0, y0, b.x1, y1), &mut buf);
        if let Some(m) = mask {
            let v = m.surface.read_region(photocraft_geom::Rect::new(b.x0, y0, b.x1, y1));
            let ch = m.surface.channels().max(1);
            for (p, mv) in buf.iter_mut().zip(v.chunks_exact(ch)) {
                let k = (1.0 - m.density * (1.0 - mv.first().copied().unwrap_or(1.0))).clamp(0.0, 1.0);
                p[3] = (f32::from(p[3]) * k).round() as u8;
            }
        }
        if k == 1 {
            for (o, p) in out.iter_mut().zip(&buf) {
                *o = if p[3] == 255 { Color32::from_rgb(p[0], p[1], p[2]) } else { Color32::from_rgba_unmultiplied(p[0], p[1], p[2], p[3]) };
            }
            return;
        }
        for (tx, o) in out.iter_mut().enumerate() {
            let (x0, x1) = (tx * k, ((tx + 1) * k).min(w));
            let mut acc = [0u32; 4];
            let mut n = 0u32;
            for r in 0..rows {
                for p in buf.get(r * w + x0..r * w + x1).unwrap_or(&[]) {
                    let c = Color32::from_rgba_unmultiplied(p[0], p[1], p[2], p[3]);
                    for (a, v) in acc.iter_mut().zip(c.to_array()) {
                        *a += u32::from(v);
                    }
                    n += 1;
                }
            }
            let n = n.max(1);
            let v = acc.map(|a| ((a + n / 2) / n) as u8);
            *o = Color32::from_rgba_premultiplied(v[0], v[1], v[2], v[3]);
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        px.par_chunks_mut(tw).enumerate().for_each(|(ty, out)| band(ty, out));
    }
    #[cfg(target_arch = "wasm32")]
    for (ty, out) in px.chunks_mut(tw).enumerate() {
        band(ty, out);
    }
    let uv = [w as f32 / (tw * k) as f32, h as f32 / (th * k) as f32];
    (ColorImage::new([tw, th], px), uv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_and_filter_follow_the_on_screen_size() {
        assert_eq!(PreviewTextures::choose(1.0, 1.0, false), (0, false));
        assert_eq!(PreviewTextures::choose(3.0, 1.0, false), (0, true), "zoomed in: crisp texels");
        assert_eq!(PreviewTextures::choose(3.0, 2.0, false), (0, false), "enlarged by the transform: smooth");
        assert_eq!(PreviewTextures::choose(1.0, 1.0, true), (0, true));
        assert_eq!(PreviewTextures::choose(0.25, 1.0, false), (2, false));
        assert_eq!(PreviewTextures::choose(0.3, 1.0, false), (1, false));
        // Nearest Neighbor must use the full-resolution texels, even when zoomed far out.
        // A mipmap would average adjacent opaque/transparent pixels before sampling.
        assert_eq!(PreviewTextures::choose(0.25, 1.0, true), (0, true));
        assert_eq!(PreviewTextures::choose(1.0 / 4096.0, 1.0, true), (0, true));
        assert_eq!(PreviewTextures::choose(f32::NAN, 1.0, false), (0, false));
        assert_eq!(PreviewTextures::choose(1e-9, 1.0, false).0, 12);
    }

    #[test]
    fn full_resolution_unless_over_the_texture_limit() {
        let mut s = photocraft_raster::Surface::new(photocraft_color::PixelFormat::RGBA8);
        // A one-pixel checkerboard: any downsampling would turn it grey.
        for y in 0..300 {
            for x in 0..500 {
                if (x + y) % 2 == 0 {
                    s.fill_rect(photocraft_geom::Rect::new(x, y, x + 1, y + 1), &[1.0, 1.0, 1.0, 1.0]);
                }
            }
        }
        let r = photocraft_geom::Rect::new(0, 0, 500, 300);
        let (img, uv) = read_surface(&s, None, r, 8192);
        assert_eq!(img.size, [500, 300]);
        assert_eq!(uv, [1.0, 1.0]);
        assert_eq!(img.pixels[0], Color32::WHITE);
        assert_eq!(img.pixels[1].a(), 0);
        // Over the limit: box-filtered (grey-ish average), uv covers the padded edge.
        let (small, uv) = read_surface(&s, None, r, 256);
        assert_eq!(small.size, [250, 150]);
        assert!((100..=155).contains(&small.pixels[0].a()), "{:?}", small.pixels[0]);
        assert_eq!(uv, [1.0, 1.0]);
        let (odd, uv) = read_surface(&s, None, photocraft_geom::Rect::new(0, 0, 301, 3), 256);
        assert_eq!(odd.size, [151, 2]);
        assert!(uv[0] < 1.0 && uv[1] < 1.0);
        // Empty rect: a placeholder, no panic.
        assert_eq!(read_surface(&s, None, photocraft_geom::Rect::new(5, 5, 5, 9), 256).0.size, [1, 1]);
    }

    #[test]
    fn mip_levels_are_built_on_demand() {
        let ctx = egui::Context::default();
        let img = ColorImage::new([5, 3], vec![Color32::WHITE; 15]);
        let t = PreviewTextures::new(&ctx, "t".into(), img, [1.0, 1.0]);
        assert_eq!(t.size(), [5, 3]);
        let (_, uv) = t.pick(&ctx, 0.5, 0.5, false);
        // Level 2 is 2×1 texels covering 8×4 base texels.
        assert!((uv[0] - 5.0 / 8.0).abs() < 1e-6 && (uv[1] - 3.0 / 4.0).abs() < 1e-6, "{uv:?}");
        let (a, _) = t.pick(&ctx, 1.0, 1.0, false);
        let (b, _) = t.pick(&ctx, 4.0, 1.0, false);
        assert_ne!(a, b, "a nearest-filtered copy for zoomed-in views");
        assert_eq!(t.pick(&ctx, 4.0, 1.0, false).0, b, "cached");
        // Asking for more levels than exist stops at 1×1.
        let _ = t.pick(&ctx, 1e-6, 1.0, false);
    }
}
