//! Mode-native flattening of plain CMYK layer stacks (#3778).
//!
//! The compositor works in display RGB, so flattening a CMYK document through it reads every ink
//! value through the profile and separates it again: even a lone Background comes back with
//! different ink. When every visible layer is a plain raster layer in Normal mode (no masks,
//! effects, clipping or advanced blending), the composite needs no colour conversion at all:
//! Normal mixes ink values linearly, as Photoshop does in CMYK, so it is worked out on the stored
//! values and an opaque layer keeps them exactly.

use photocraft_color::blend::BlendMode;
use photocraft_color::{ColorMode, PixelFormat};
use photocraft_doc::{Document, Layer, LayerContent};
use photocraft_geom::Rect;
use photocraft_raster::Surface;

/// Process channels of a CMYK pixel.
const INKS: usize = 4;

/// A layer the native path can composite: a visible raster layer the compositor would only blend
/// Normally at its opacity.
fn plain(layer: &Layer) -> Option<&Surface> {
    let LayerContent::Raster(s) = &layer.content else { return None };
    let simple = layer.blend == BlendMode::Normal
        && !layer.clipped
        && layer.mask.as_ref().is_none_or(|m| !m.enabled)
        && layer.vector_mask.as_ref().is_none_or(|v| !v.enabled)
        && !crate::effects::has_effects(layer)
        && layer.excluded_channels == 0
        && layer.blend_if.is_default()
        && layer.advanced.is_default()
        && layer.video.is_none()
        && s.format().mode == ColorMode::Cmyk;
    simple.then_some(s)
}

/// The composite of `doc` over `rect` in `fmt`, worked out in ink values, or `None` when the
/// document isn't CMYK or a visible layer needs the full compositor (see the module docs).
/// `paper` flattens onto white paper (no ink) and makes the result opaque.
pub fn flatten_cmyk(doc: &Document, rect: Rect, fmt: PixelFormat, paper: bool) -> Option<Surface> {
    if doc.mode != ColorMode::Cmyk || fmt.mode != ColorMode::Cmyk || doc.layers.iter().any(|l| l.clipped) {
        return None;
    }
    let mut stack = Vec::new();
    for layer in doc.layers.iter().filter(|l| l.visible) {
        let alpha = (layer.opacity * layer.fill_opacity).clamp(0.0, 1.0);
        stack.push((plain(layer)?, alpha));
    }
    let mut out = Surface::new(fmt);
    if rect.is_empty() {
        return Some(out);
    }
    let n = fmt.channels();
    let w = rect.width() as usize;
    let rows = crate::band_rows_for(rect.width(), 0);
    let (mut acc, mut src, mut data) = (Vec::new(), Vec::new(), Vec::new());
    let mut y = rect.y0;
    while y < rect.y1 {
        let band = Rect::new(rect.x0, y, rect.x1, y.saturating_add(rows).min(rect.y1));
        // Straight C, M, Y, K, alpha per pixel, starting transparent.
        acc.clear();
        acc.resize(w * band.height() as usize, [0.0f32; INKS + 1]);
        for &(surf, opacity) in &stack {
            if opacity <= 0.0 || (surf.content_bounds().intersect(&band).is_empty() && surf.default_pixel().iter().all(|v| *v == 0.0)) {
                continue;
            }
            let sf = surf.format();
            let sn = sf.channels();
            surf.read_region_into(band, &mut src);
            for (d, s) in acc.iter_mut().zip(src.chunks_exact(sn)) {
                let sa = if sf.alpha { s[INKS] } else { 1.0 } * opacity;
                if sa <= 0.0 {
                    continue;
                }
                let oa = sa + d[INKS] * (1.0 - sa);
                let keep = d[INKS] * (1.0 - sa);
                for c in 0..INKS {
                    // An opaque pixel replaces what is below exactly (no rounding through the mix).
                    d[c] = if sa >= 1.0 { s[c] } else { (s[c] * sa + d[c] * keep) / oa };
                }
                d[INKS] = oa;
            }
        }
        data.clear();
        data.resize(acc.len() * n, 0.0);
        for (p, o) in acc.iter().zip(data.chunks_exact_mut(n)) {
            // White paper carries no ink, so flattening onto it scales the ink by coverage.
            let a = p[INKS];
            let k = if paper && a < 1.0 { a } else { 1.0 };
            for c in 0..INKS {
                o[c] = p[c] * k;
            }
            if fmt.alpha {
                o[INKS] = if paper { 1.0 } else { a };
            }
        }
        out.write_region(band, &data);
        y = band.y1;
    }
    out.prune();
    Some(out)
}
