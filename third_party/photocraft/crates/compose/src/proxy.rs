//! Downsampled proxy documents: every pixel surface scaled down by an integer factor `k`, so a
//! composite of the proxy approximates the document's composite at 1/k resolution for ~1/k² of
//! the work. Used for interactive previews and for thumbnails of large documents (navigator,
//! channel thumbnails, histograms, file previews), where a full-resolution composite of a 200 MP
//! document would cost seconds and gigabytes.

use photocraft_doc::{Document, Layer, LayerContent, Size};
use photocraft_geom::Rect;
use photocraft_raster::Surface;

/// Nearest-neighbour downsample of a surface by integer factor `k` (document coordinates / k).
pub fn downsample(s: &Surface, k: u32) -> Surface {
    let mut out = Surface::with_default(s.format(), &s.default_pixel());
    let b = s.content_bounds();
    if b.is_empty() || k <= 1 {
        return if k <= 1 { s.clone() } else { out };
    }
    // Surfaces span at most ±2^30 px; larger factors reduce everything to one pixel anyway.
    let k = k.min(1 << 20) as i32;
    let bpp = s.format().bytes_per_pixel();
    let (x0, x1) = (b.x0.div_euclid(k), b.x1.div_euclid(k) + i32::from(b.x1.rem_euclid(k) != 0));
    let (y0, y1) = (b.y0.div_euclid(k), b.y1.div_euclid(k) + i32::from(b.y1.rem_euclid(k) != 0));
    let w = x1.saturating_sub(x0).max(0) as usize;
    let mut row = vec![0u8; w * bpp];
    for oy in y0..y1 {
        let src = s.to_interleaved(Rect::new(x0.saturating_mul(k), oy.saturating_mul(k), x1.saturating_mul(k), oy.saturating_mul(k).saturating_add(1)));
        for (i, px) in row.chunks_exact_mut(bpp).enumerate() {
            // Saturated edges (a surface at the i32 limits) can make `src` shorter than expected.
            if let Some(from) = (i * k as usize * bpp).checked_add(bpp).and_then(|end| src.get(i * k as usize * bpp..end)) {
                px.copy_from_slice(from);
            }
        }
        out.write_interleaved(Rect::new(x0, oy, x1, oy + 1), &row);
    }
    out
}

fn shrink_layer(l: &mut Layer, k: u32) {
    if let Some(m) = &mut l.mask {
        m.surface = downsample(&m.surface, k);
    }
    if let Some(fc) = &mut l.fill_cache {
        fc.surface = downsample(&fc.surface, k);
    }
    match &mut l.content {
        LayerContent::Raster(s) => *s = downsample(s, k),
        LayerContent::Group(g) => {
            if let Some(ab) = &mut g.artboard {
                let divisor = i64::from(k);
                // Pixel x samples source x*k: both half-open edges round up, including negatives.
                let edge = |v: i32| {
                    let v = i64::from(v);
                    (v.div_euclid(divisor) + i64::from(v.rem_euclid(divisor) != 0)) as i32
                };
                ab.rect = Rect::new(edge(ab.rect.x0), edge(ab.rect.y0), edge(ab.rect.x1), edge(ab.rect.y1));
            }
            for c in &mut g.children {
                shrink_layer(c, k);
            }
        }
        LayerContent::Text(t) => t.cache = t.cache.as_ref().map(|s| downsample(s, k)),
        LayerContent::Shape(sh) => sh.cache = sh.cache.as_ref().map(|s| downsample(s, k)),
        LayerContent::Smart(so) => so.cache = so.cache.as_ref().map(|s| downsample(s, k)),
        LayerContent::Adjustment(_) | LayerContent::Fill(_) => {}
    }
}

/// A copy of `doc` scaled down by `k` (same layer ids, so adjustments can be swapped in). The
/// selection scales with it, so a filter previewed on the proxy stays inside the (feathered)
/// selection exactly as the full-size result does.
pub fn proxy_document(doc: &Document, k: u32) -> Document {
    let mut p = doc.clone();
    if k <= 1 {
        return p;
    }
    p.size = Size::new(doc.size.width.div_ceil(k), doc.size.height.div_ceil(k));
    for l in &mut p.layers {
        shrink_layer(l, k);
    }
    // Spot channels are printed under Multichannel composites.
    for c in &mut p.channels {
        c.surface = downsample(&c.surface, k);
    }
    p.selection = p.selection.as_ref().map(|s| downsample(s, k));
    if let Some(mask) = &mut p.quick_mask {
        mask.surface = downsample(&mask.surface, k);
    }
    p
}

/// Whether a proxy composite is a faithful reduction of `doc`: layer effects (whose sizes are in
/// document pixels) and nonempty vector masks (paths in document coordinates) would not scale
/// with it. Empty vector masks have constant coverage, including inversion and density.
/// Nonempty paths can still anchor fill layouts even when their mask density is zero.
pub fn proxy_faithful(doc: &Document) -> bool {
    doc.walk().iter().all(|(_, _, l)| !crate::effects::has_effects(l) && l.vector_mask.as_ref().is_none_or(|v| !v.enabled || v.path.is_empty()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_color::{Color, ColorMode, PixelFormat, SampleType};
    use photocraft_doc::vector::{Knot, Path, Subpath, VectorMask};
    use photocraft_doc::{Effect, Fill, GradientStyle, LayerMask};

    fn two_colors(size: Size, depth: SampleType) -> Document {
        let mut doc = Document::new("mask", size, ColorMode::Rgb, depth);
        let format = doc.pixel_format();
        for (name, color) in [("blue", [0.0, 0.0, 1.0, 1.0]), ("red", [1.0, 0.0, 0.0, 1.0])] {
            doc.layers.push(Layer::new(name, LayerContent::Raster(Surface::with_default(format, &color))));
        }
        doc
    }

    fn assert_same_pixels(actual: &crate::Buffer, expected: &crate::Buffer) {
        assert_same_pixels_within(actual, expected, 1e-6);
    }

    fn assert_same_pixels_within(actual: &crate::Buffer, expected: &crate::Buffer, tolerance: f32) {
        assert_eq!(actual.rect, expected.rect);
        assert_eq!(actual.px.len(), expected.px.len());
        for (a, b) in actual.px.iter().zip(&expected.px) {
            assert!(a.iter().zip(b).all(|(a, b)| (a - b).abs() <= tolerance), "{a:?} != {b:?}");
        }
    }

    fn u16_coverage(value: f32) -> f32 {
        // Feathered masks use the existing U16 combined-mask surface, including its default.
        (value.clamp(0.0, 1.0) * 65535.0 + 0.5).floor() / 65535.0
    }

    fn opacity_reference(doc: &Document, coverage: f32) -> Document {
        let mut reference = doc.clone();
        reference.layers[1].vector_mask = None;
        reference.layers[1].opacity *= coverage;
        reference
    }

    #[test]
    fn downsample_picks_every_kth_pixel() {
        let mut s = Surface::new(PixelFormat::RGBA8);
        s.fill_rect(Rect::new(0, 0, 8, 8), &[1.0, 0.0, 0.0, 1.0]);
        s.write_pixel(4, 4, &[0.0, 0.0, 1.0, 1.0]);
        let d = downsample(&s, 4);
        assert_eq!(d.pixel(0, 0), vec![1.0, 0.0, 0.0, 1.0]);
        assert_eq!(d.pixel(1, 1), vec![0.0, 0.0, 1.0, 1.0]);
        assert_eq!(d.pixel(2, 2)[3], 0.0);
    }

    #[test]
    fn downsample_handles_negative_coordinates() {
        let mut s = Surface::new(PixelFormat::RGBA8);
        s.fill_rect(Rect::new(-8, -8, 0, 0), &[0.0, 1.0, 0.0, 1.0]);
        let d = downsample(&s, 4);
        assert_eq!(d.pixel(-1, -1), vec![0.0, 1.0, 0.0, 1.0]);
        assert_eq!(d.pixel(0, 0)[3], 0.0);
    }

    #[test]
    fn proxy_keeps_structure_and_ids() {
        let doc = Document::with_background("d", Size::new(4000, 3000), ColorMode::Rgb, SampleType::U8, Color::WHITE);
        let p = proxy_document(&doc, 4);
        assert_eq!((p.size.width, p.size.height), (1000, 750));
        assert_eq!(p.layers[0].id, doc.layers[0].id);
        assert_eq!(p.layers[0].surface().unwrap().pixel(999, 749), vec![1.0; 4]);
        assert!(proxy_faithful(&doc));
    }

    #[test]
    fn proxy_scales_the_selection() {
        let mut doc = Document::with_background("d", Size::new(64, 64), ColorMode::Rgb, SampleType::U8, Color::WHITE);
        let mut sel = Surface::new(PixelFormat::GRAY8);
        sel.fill_rect(Rect::new(0, 0, 32, 64), &[1.0]);
        doc.selection = Some(sel);
        let p = proxy_document(&doc, 4);
        let s = p.selection.as_ref().expect("selection kept");
        assert_eq!((s.pixel(7, 15)[0], s.pixel(8, 0)[0]), (1.0, 0.0));
        assert!(proxy_document(&Document::new("n", Size::new(8, 8), ColorMode::Rgb, SampleType::U8), 4).selection.is_none());
    }

    #[test]
    fn proxy_scales_quick_mask_at_every_depth() {
        for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
            let mut doc = Document::new("mask", Size::new(16, 12), ColorMode::Rgb, depth);
            let mut surface = Surface::new(PixelFormat::new(ColorMode::Grayscale, depth, false));
            surface.fill_rect(Rect::new(0, 0, 8, 12), &[1.0]);
            let mut mask = photocraft_doc::AlphaChannel::new("Quick Mask", surface);
            mask.opacity = 0.25;
            doc.quick_mask = Some(mask);
            for k in [1, 2, 4] {
                let proxy = proxy_document(&doc, k);
                let mask = proxy.quick_mask.as_ref().unwrap();
                assert_eq!(mask.surface.pixel((8 / k - 1) as i32, 0), vec![1.0]);
                assert_eq!(mask.surface.pixel((8 / k) as i32, 0), vec![0.0], "depth {depth:?}, factor {k}");
                assert_eq!(mask.opacity, 0.25);
                assert_eq!(mask.surface.format().sample, depth);
            }
            assert_eq!(doc.quick_mask.as_ref().unwrap().surface.pixel(7, 0), vec![1.0], "source stays intact");
        }
    }

    #[test]
    fn proxy_accepts_empty_paths_but_keeps_geometry_and_effect_guards() {
        let mut doc = two_colors(Size::new(32, 16), SampleType::F32);
        doc.layers[1].vector_mask = Some(VectorMask::new(Path::new(vec![Subpath::default()])));
        assert!(proxy_faithful(&doc));
        // A lone knot has no fillable segments, but still carries document coordinates.
        doc.layers[1].vector_mask.as_mut().unwrap().path.subpaths[0].knots.push(Knot::corner(12.0, 8.0));
        assert!(!proxy_faithful(&doc));
        doc.layers[1].vector_mask.as_mut().unwrap().enabled = false;
        assert!(proxy_faithful(&doc));
        doc.layers[1].vector_mask = Some(VectorMask::new(Path::default()));
        doc.layers[1].effects.items.push(Effect::default_drop_shadow());
        doc.layers[1].effects.enabled = true;
        assert!(!proxy_faithful(&doc));
        doc.layers[1].effects.items[0].set_enabled(false);
        assert!(proxy_faithful(&doc));
    }

    #[test]
    fn empty_vector_masks_keep_constant_coverage_at_every_depth() {
        for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
            let mut doc = two_colors(Size::new(32, 16), depth);
            for path in [Path::default(), Path::new(vec![Subpath::default()])] {
                for inverted in [false, true] {
                    for density in [0.0, 0.25, 1.0] {
                        for feather in [0.0, 4.0] {
                            let mut mask = VectorMask::new(path.clone());
                            mask.path.inverted = inverted;
                            mask.density = density;
                            mask.feather = feather;
                            doc.layers[1].vector_mask = Some(mask);
                            assert!(proxy_faithful(&doc));
                            let coverage = if inverted { 1.0 - density } else { 1.0 };
                            let coverage = if feather > 0.0 { u16_coverage(coverage) } else { coverage };
                            let expected = [coverage, 0.0, 1.0 - coverage, 1.0];
                            let full = crate::render(&doc, doc.bounds());
                            for pixel in &full.px {
                                assert!(pixel.iter().zip(expected).all(|(a, b)| (a - b).abs() < 1e-6), "{depth:?} {inverted} {density} {feather}: {pixel:?}");
                            }
                            let proxy = proxy_document(&doc, 4);
                            assert_eq!(proxy.layers[1].vector_mask, doc.layers[1].vector_mask);
                            let reference = proxy_document(&opacity_reference(&doc, coverage), 4);
                            assert_same_pixels(&crate::render(&proxy, proxy.bounds()), &crate::render(&reference, reference.bounds()));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn empty_vector_mask_multiplies_a_feathered_pixel_mask_on_the_proxy() {
        let mut doc = two_colors(Size::new(32, 16), SampleType::F32);
        let mut surface = Surface::with_default(PixelFormat::new(ColorMode::Grayscale, SampleType::F32, false), &[0.25]);
        surface.fill_rect(Rect::new(0, 0, 16, 16), &[0.75]);
        doc.layers[1].mask = Some(LayerMask { surface, density: 0.5, feather: 2.0, ..LayerMask::reveal_all() });
        let mut mask = VectorMask::new(Path { inverted: true, ..Path::default() });
        mask.density = 0.75;
        mask.feather = 8.0;
        doc.layers[1].vector_mask = Some(mask);
        assert!(proxy_faithful(&doc));
        let reference = opacity_reference(&doc, 0.25);
        for k in [1, 2, 4] {
            let (actual, expected) = (proxy_document(&doc, k), proxy_document(&reference, k));
            // Q(0.25 * pixel) versus 0.25 * Q(pixel): U16 rounding differs by at most
            // 0.625 / 65535. Allow one U16 level plus a small float arithmetic allowance.
            assert_same_pixels_within(&crate::render(&actual, actual.bounds()), &crate::render(&expected, expected.bounds()), 1.0 / 65535.0 + 1e-6);
        }
    }

    #[test]
    fn empty_vector_masks_keep_the_pixel_mask_gradient_frame() {
        let mut doc = two_colors(Size::new(32, 16), SampleType::F32);
        doc.layers[1].content = LayerContent::Fill(Fill::gradient(vec![(0.0, Color::BLACK), (1.0, Color::WHITE)], 0.0, 1.0, GradientStyle::Linear, false));
        let mut pixel = LayerMask::hide_all();
        pixel.surface.fill_rect(Rect::new(8, 4, 24, 12), &[1.0]);
        doc.layers[1].mask = Some(pixel);
        for inverted in [false, true] {
            let mut mask = VectorMask::new(Path { inverted, ..Path::default() });
            mask.density = 0.5;
            mask.feather = 3.0;
            doc.layers[1].vector_mask = Some(mask);
            assert!(proxy_faithful(&doc));
            assert_eq!(crate::fill_frame(&doc.layers[1], doc.bounds()), Rect::new(8, 4, 24, 12));
            let actual = proxy_document(&doc, 4);
            assert_eq!(crate::fill_frame(&actual.layers[1], actual.bounds()), Rect::new(2, 1, 6, 3));
            let reference = proxy_document(&opacity_reference(&doc, u16_coverage(if inverted { 0.5 } else { 1.0 })), 4);
            assert_same_pixels(&crate::render(&actual, actual.bounds()), &crate::render(&reference, reference.bounds()));
        }
    }

    #[test]
    fn zero_density_geometry_still_blocks_the_proxy_and_anchors_a_gradient() {
        let mut doc = Document::new("gradient", Size::new(64, 32), ColorMode::Rgb, SampleType::F32);
        let fill = Fill::gradient(vec![(0.0, Color::BLACK), (1.0, Color::WHITE)], 0.0, 1.0, GradientStyle::Linear, false);
        let mut layer = Layer::new("gradient", LayerContent::Fill(fill));
        let path = Path::new(vec![Subpath::polygon(&[(16.0, 8.0), (48.0, 8.0), (48.0, 24.0), (16.0, 24.0)])]);
        let mut mask = VectorMask::new(path);
        mask.density = 0.0;
        layer.vector_mask = Some(mask);
        doc.layers.push(layer);
        assert!(!proxy_faithful(&doc));
        assert_eq!(crate::fill_frame(&doc.layers[0], doc.bounds()), Rect::new(16, 8, 48, 24));
        // Density zero reveals all pixels, but the unscaled path would still move the fill.
        let forced_proxy = proxy_document(&doc, 4);
        let exact = crate::render_reduced(&doc, 16, 8);
        let wrong = crate::render(&forced_proxy, forced_proxy.bounds());
        assert!(exact.px.iter().zip(&wrong.px).any(|(a, b)| (a[0] - b[0]).abs() > 0.1));
    }

    #[test]
    fn large_thumbnail_keeps_inverted_empty_mask_coverage() {
        // Default-only surfaces exercise the large-document branch without allocating 17 MP.
        let mut doc = two_colors(Size::new(4200, 4200), SampleType::F32);
        let mut mask = VectorMask::new(Path { inverted: true, ..Path::default() });
        mask.density = 0.75;
        mask.feather = 8.0;
        doc.layers[1].vector_mask = Some(mask);
        assert!(doc.size.area() > crate::PROXY_THUMBNAIL_PIXELS);
        assert!(proxy_faithful(&doc));
        let actual = crate::thumbnail_buffer(&doc, 4);
        assert_eq!(actual.rect, Rect::new(0, 0, 4, 4));
        let coverage = u16_coverage(0.25);
        assert_eq!(actual.px, vec![[coverage, 0.0, 1.0 - coverage, 1.0]; 16]);
    }
}
