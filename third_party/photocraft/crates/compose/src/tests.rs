use super::*;
use photocraft_color::{Color, ColorMode, PixelFormat, SampleType};
use photocraft_doc::adjust::{CurvePoint, LevelsChannel};
use photocraft_doc::{Adjustment, Document, Fill, Layer, LayerContent, LayerMask};
use photocraft_geom::Size;

const E: f32 = 2.0 / 255.0;

fn close4(a: [f32; 4], b: [f32; 4]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= E)
}

fn doc_white(w: u32, h: u32) -> Document {
    Document::with_background("t", Size::new(w, h), ColorMode::Rgb, SampleType::U8, Color::WHITE)
}

fn solid_layer(name: &str, rect: Rect, rgba: [f32; 4]) -> Layer {
    let mut l = Layer::raster(name, PixelFormat::RGBA8);
    l.surface_mut().unwrap().fill_rect(rect, &rgba);
    l
}

fn px(doc: &Document, x: i32, y: i32) -> [f32; 4] {
    render(doc, Rect::from_xywh(x, y, 1, 1)).px[0]
}

#[test]
fn background_only() {
    let d = doc_white(8, 8);
    assert!(close4(px(&d, 3, 3), [1.0; 4]));
    // outside canvas the background layer has no pixels
    assert!(close4(px(&d, 20, 3), [0.0; 4]));
}

#[test]
fn normal_layer_over_background() {
    let mut d = doc_white(8, 8);
    d.layers.push(solid_layer("red", Rect::new(0, 0, 4, 8), [1.0, 0.0, 0.0, 1.0]));
    assert!(close4(px(&d, 1, 1), [1.0, 0.0, 0.0, 1.0]));
    assert!(close4(px(&d, 5, 1), [1.0; 4]));
}

// The pre-optimisation Normal formula, independent of the production helper. Its operation
// order matters: an algebraically equivalent premultiplied mix can round differently.
fn original_normal_rgb(backdrop: [f32; 4], source: [f32; 4], opacity: f32) -> [f32; 4] {
    let ab = backdrop[3];
    let as_ = source[3] * opacity;
    if as_ <= 0.0 {
        return backdrop;
    }
    let ao = as_ + ab * (1.0 - as_);
    if ao <= 0.0 {
        return [0.0; 4];
    }
    let kb = ab * (1.0 - as_);
    let inv = 1.0 / ao;
    [(kb * backdrop[0] + as_ * source[0]) * inv, (kb * backdrop[1] + as_ * source[1]) * inv, (kb * backdrop[2] + as_ * source[2]) * inv, ao]
}

#[test]
fn normal_rgb_kernel_preserves_scalar_bits() {
    let subnormal = f32::from_bits(1);
    let backdrops = [[0.0, -0.0, 2.0, 0.0], [0.2, 0.4, 0.6, 0.5], [2.0, -0.25, 4.0, 1.0], [-1.0, 0.01, 1.0, -0.5], [1.0, -0.0, f32::MIN_POSITIVE, subnormal]];
    let sources = [
        [0.9, 0.1, 0.3, 1.0],
        [0.25, 2.0, -1.0, 0.5],
        [subnormal, 1e-20, 1.0, subnormal],
        [0.5, 0.2, 0.7, -0.0],
        [0.5, 0.2, 0.7, -0.5],
        [f32::NAN, 0.2, 0.7, 0.0],
    ];
    for b in backdrops {
        for s in sources {
            for opacity in [-1.0, -0.0, 0.0, 0.5, 1.0, 2.0] {
                let want = original_normal_rgb(b, s, opacity);
                let got = psblend::composite_normal_rgb(b, s, opacity);
                assert_eq!(got.map(f32::to_bits), want.map(f32::to_bits), "{b:?} / {s:?} / {opacity}");
            }
        }
    }
}

fn normal_rgb_test_buffers(fmt: PixelFormat) -> (Buffer, Buffer) {
    let rect = Rect::new(-3, -2, 5, 0);
    let mut bottom = Surface::new(fmt);
    let mut top = Surface::new(fmt);
    for y in rect.y0..rect.y1 {
        for x in rect.x0..rect.x1 {
            let i = ((y - rect.y0) * rect.width() as i32 + x - rect.x0) as usize;
            let t = i as f32 / 16.0;
            let b = [2.0 * t, 0.4, -0.25 + t, [0.0, 0.25, 0.75, 1.0][i % 4]];
            let s = [0.9 - t, 0.1 + t, 0.3, [0.0, -0.0, 0.5, 1.0][i % 4]];
            bottom.write_pixel(x, y, &photocraft_raster::from_rgba(&fmt, b));
            top.write_pixel(x, y, &photocraft_raster::from_rgba(&fmt, s));
        }
    }
    (surface_to_buffer(&bottom, rect), surface_to_buffer(&top, rect))
}

// Keep the old buffer dispatch as an oracle for the specialised loop and all of its fallbacks.
fn original_blend_into_g(backdrop: &mut Buffer, src: &Buffer, mode: BlendMode, opacity: f32, gamma: f32) {
    let rect = backdrop.rect;
    let w = rect.width() as i32;
    for (i, b) in backdrop.px.iter_mut().enumerate() {
        let mut s = src.px[i];
        if s[3] <= 0.0 {
            continue;
        }
        let mut mode = mode;
        if mode == BlendMode::Dissolve {
            let x = rect.x0 + (i as i32 % w);
            let y = rect.y0 + (i as i32 / w);
            s[3] = if dissolve_noise(x, y) < s[3] * opacity { 1.0 } else { 0.0 };
            mode = BlendMode::Normal;
            *b = blend::composite(mode, *b, s, 1.0);
            continue;
        }
        *b = blend::composite_gamma(mode, *b, s, opacity, gamma);
    }
}

fn assert_buffer_bits(got: &Buffer, want: &Buffer) {
    assert_eq!(got.rect, want.rect);
    assert_eq!(got.px.len(), want.px.len());
    for (i, (got, want)) in got.px.iter().zip(&want.px).enumerate() {
        assert_eq!(got.map(f32::to_bits), want.map(f32::to_bits), "pixel {i}: {got:?} vs {want:?}");
    }
}

#[test]
fn normal_rgb_buffer_dispatch_and_fallbacks_preserve_bits() {
    for fmt in [PixelFormat::RGBA8, PixelFormat::RGBA16, PixelFormat::RGBA32F] {
        let (bottom, top) = normal_rgb_test_buffers(fmt);
        for lab in [false, true] {
            psblend::with_lab_mix(lab, || {
                psblend::with_hdr(fmt.sample == SampleType::F32, || {
                    for mode in BlendMode::LAYER_MODES {
                        for gamma in [1.0, psblend::TEXT_GAMMA] {
                            for opacity in [0.0, 0.5, 1.0] {
                                let mut got = bottom.clone();
                                let mut want = bottom.clone();
                                original_blend_into_g(&mut want, &top, mode, opacity, gamma);
                                blend_into_g(&mut got, &top, mode, opacity, gamma);
                                assert_buffer_bits(&got, &want);
                            }
                        }
                    }
                });
            });
        }
    }
}

#[test]
fn normal_rgb_buffer_preserves_alpha_skip_and_nested_scopes() {
    let rect = Rect::new(-1, -1, 2, 0);
    let bottom = Buffer::filled(rect, [-0.0, 0.4, 2.0, 0.75]);
    let top = Buffer { rect, px: vec![[f32::NAN, 0.3, 0.1, -0.5], [0.2, 0.3, 0.1, -0.0], [0.2, 0.3, 0.1, f32::NAN]] };
    psblend::with_lab_mix(true, || {
        psblend::with_hdr(true, || {
            psblend::with_lab_mix(false, || {
                psblend::with_hdr(false, || {
                    let mut got = bottom.clone();
                    let mut want = bottom.clone();
                    original_blend_into_g(&mut want, &top, BlendMode::Normal, -1.0, 1.0);
                    blend_into_g(&mut got, &top, BlendMode::Normal, -1.0, 1.0);
                    assert_buffer_bits(&got, &want);
                    assert_eq!(got.px[0].map(f32::to_bits), bottom.px[0].map(f32::to_bits));
                    assert_eq!(got.px[1].map(f32::to_bits), bottom.px[1].map(f32::to_bits));
                });
                assert!(psblend::HDR.with(|h| h.get()));
            });
            assert!(psblend::LAB_MIX.with(|l| l.get()));
            assert!(psblend::HDR.with(|h| h.get()));
            let mut got = bottom.clone();
            let mut want = bottom.clone();
            original_blend_into_g(&mut want, &top, BlendMode::Normal, 0.5, 1.0);
            blend_into_g(&mut got, &top, BlendMode::Normal, 0.5, 1.0);
            assert_buffer_bits(&got, &want);
        });
    });
}

#[test]
fn opacity_and_fill_multiply() {
    let mut d = doc_white(4, 4);
    let mut l = solid_layer("k", Rect::new(0, 0, 4, 4), [0.0, 0.0, 0.0, 1.0]);
    l.opacity = 0.5;
    l.fill_opacity = 0.5;
    d.layers.push(l);
    let p = px(&d, 0, 0);
    assert!((p[0] - 0.75).abs() <= E, "{p:?}");
}

#[test]
fn hidden_layers_skipped() {
    let mut d = doc_white(4, 4);
    let mut l = solid_layer("k", Rect::new(0, 0, 4, 4), [0.0, 0.0, 0.0, 1.0]);
    l.visible = false;
    d.layers.push(l);
    assert!(close4(px(&d, 0, 0), [1.0; 4]));
}

#[test]
fn every_blend_mode_matches_reference_on_opaque_pixels() {
    let backdrop = [0.6, 0.3, 0.2, 1.0];
    let source = [0.2, 0.7, 0.5, 1.0];
    for mode in BlendMode::LAYER_MODES {
        if mode == BlendMode::Dissolve {
            continue;
        }
        let mut d = Document::new("b", Size::new(2, 2), ColorMode::Rgb, SampleType::F32);
        let mut bottom = Layer::raster("b", PixelFormat::RGBA32F);
        bottom.surface_mut().unwrap().fill_rect(Rect::new(0, 0, 2, 2), &backdrop);
        let mut top = Layer::raster("t", PixelFormat::RGBA32F);
        top.surface_mut().unwrap().fill_rect(Rect::new(0, 0, 2, 2), &source);
        top.blend = mode;
        d.layers = vec![bottom, top];
        let got = px(&d, 0, 0);
        // A 32-bit document: Add / Divide don't clip at 1 (Divide gives 0.6 / 0.2 = 3 in red).
        psblend::HDR.with(|h| h.set(true));
        let want = blend::blend_rgb(mode, [0.6, 0.3, 0.2], [0.2, 0.7, 0.5]);
        psblend::HDR.with(|h| h.set(false));
        for i in 0..3 {
            assert!((got[i] - want[i]).abs() < 1e-5, "{mode:?}: {got:?} vs {want:?}");
        }
        assert!((got[3] - 1.0).abs() < 1e-6);
    }
}

/// One opaque pixel: `top` in `mode` over `bottom`, in a document of `depth`.
fn blend_px(depth: SampleType, mode: BlendMode, bottom: [f32; 3], top: [f32; 3]) -> [f32; 4] {
    let mut d = Document::new("b", Size::new(1, 1), ColorMode::Rgb, depth);
    let fmt = d.pixel_format();
    let mut b = Layer::raster("b", fmt);
    b.surface_mut().unwrap().fill_rect(Rect::new(0, 0, 1, 1), &photocraft_raster::from_rgba(&fmt, [bottom[0], bottom[1], bottom[2], 1.0]));
    let mut t = Layer::raster("t", fmt);
    t.surface_mut().unwrap().fill_rect(Rect::new(0, 0, 1, 1), &photocraft_raster::from_rgba(&fmt, [top[0], top[1], top[2], 1.0]));
    t.blend = mode;
    d.layers = vec![b, t];
    px(&d, 0, 0)
}

#[test]
fn non_separable_modes_keep_values_above_one_in_32_bit() {
    use photocraft_color::blend::{blend_rgb_within, lum};
    let modes = [BlendMode::Hue, BlendMode::Saturation, BlendMode::Color, BlendMode::Luminosity];
    // Hue over a grey of 2: the grey (the backdrop has no saturation), not ClipColor dividing
    // by the rounding noise of `max − lum` (Hue over (4, 3.9999, 4) gave 27.7 in blue).
    for bg in [[2.0, 2.0, 2.0], [2.0, 2.0, 1.999_99], [4.0, 3.999_9, 4.0]] {
        for m in modes {
            for top in [[3.0, 0.25, 0.125], [2.0, 2.0, 2.0], [4.0, 3.999_9, 4.0]] {
                let got = blend_px(SampleType::F32, m, bg, top);
                let want = blend_rgb_within(m, bg, top, f32::MAX);
                assert!(got.iter().zip(want).all(|(g, w)| (g - w).abs() <= 1e-4 * w.abs().max(1.0)), "{m:?} {bg:?} {top:?}: {got:?} vs {want:?}");
                assert!(got.iter().all(|v| v.is_finite() && *v >= -1e-5), "{m:?} {bg:?} {top:?}: {got:?}");
            }
        }
    }
    // A saturated colour keeps its brightness above 1 in 32-bit, and is clipped into 0..1
    // (exactly as before) in 8/16-bit.
    let (bg, top) = ([0.5, 0.5, 0.5], [1.0, 0.0, 0.0]);
    let hdr = blend_px(SampleType::F32, BlendMode::Color, bg, top);
    assert!((hdr[0] - 1.2).abs() < 1e-4 && (lum([hdr[0], hdr[1], hdr[2]]) - 0.5).abs() < 1e-4, "{hdr:?}");
    for depth in [SampleType::U8, SampleType::U16] {
        for m in modes {
            let got = blend_px(depth, m, bg, top);
            let want = photocraft_color::blend::blend_rgb(m, bg, top);
            assert!(got.iter().zip(want).all(|(g, w)| (g - w).abs() <= 1.0 / 255.0), "{m:?} {depth:?}: {got:?} vs {want:?}");
        }
        let got = blend_px(depth, BlendMode::Color, bg, top);
        assert!(got[0] <= 1.0 && got[0] > 0.9, "{depth:?}: {got:?}");
    }
    // A 32-bit document's in-range colours that stay in range are unchanged.
    for m in modes {
        let got = blend_px(SampleType::F32, m, [0.6, 0.3, 0.2], [0.2, 0.7, 0.5]);
        let want = photocraft_color::blend::blend_rgb(m, [0.6, 0.3, 0.2], [0.2, 0.7, 0.5]);
        assert!(got.iter().zip(want).all(|(g, w)| (g - w).abs() <= 1e-5), "{m:?}: {got:?} vs {want:?}");
    }
}

#[test]
fn layer_mask_hides_pixels() {
    let mut d = doc_white(4, 4);
    let mut l = solid_layer("k", Rect::new(0, 0, 4, 4), [0.0, 0.0, 0.0, 1.0]);
    let mut m = LayerMask::reveal_all();
    m.surface.fill_rect(Rect::new(0, 0, 2, 4), &[0.0]);
    l.mask = Some(m);
    d.layers.push(l);
    assert!(close4(px(&d, 0, 0), [1.0; 4]));
    assert!(close4(px(&d, 3, 0), [0.0, 0.0, 0.0, 1.0]));
}

#[test]
fn disabled_mask_is_ignored() {
    let mut d = doc_white(2, 2);
    let mut l = solid_layer("k", Rect::new(0, 0, 2, 2), [0.0, 0.0, 0.0, 1.0]);
    let mut m = LayerMask::hide_all();
    m.enabled = false;
    l.mask = Some(m);
    d.layers.push(l);
    // CMYK displays through the built-in CMYK profile: 100 % K alone is a dark neutral
    // (as in any real CMYK profile), not pure black.
    let p = px(&d, 0, 0);
    assert!(p[0] < 0.3 && (p[0] - p[1]).abs() < 0.05 && (p[1] - p[2]).abs() < 0.05 && p[3] == 1.0, "{p:?}");
}

#[test]
fn clipping_mask_restricts_to_base_alpha() {
    let mut d = doc_white(8, 8);
    d.layers.push(solid_layer("base", Rect::new(0, 0, 4, 8), [0.0, 0.0, 1.0, 1.0]));
    let mut clip = solid_layer("clip", Rect::new(0, 0, 8, 8), [1.0, 0.0, 0.0, 1.0]);
    clip.clipped = true;
    d.layers.push(clip);
    assert!(close4(px(&d, 1, 1), [1.0, 0.0, 0.0, 1.0]), "inside base: clipped layer visible");
    assert!(close4(px(&d, 6, 1), [1.0; 4]), "outside base: clipped layer hidden");
}

#[test]
fn hidden_base_hides_clipping_group() {
    let mut d = doc_white(4, 4);
    let mut base = solid_layer("base", Rect::new(0, 0, 4, 4), [0.0, 0.0, 1.0, 1.0]);
    base.visible = false;
    d.layers.push(base);
    let mut clip = solid_layer("clip", Rect::new(0, 0, 4, 4), [1.0, 0.0, 0.0, 1.0]);
    clip.clipped = true;
    d.layers.push(clip);
    assert!(close4(px(&d, 1, 1), [1.0; 4]));
}

#[test]
fn isolated_group_differs_from_pass_through() {
    // A Multiply layer inside a group: pass-through multiplies with the background,
    // an isolated (Normal) group multiplies only with its own (empty) contents.
    let make = |blend: BlendMode| {
        let mut d = doc_white(2, 2);
        d.layers[0].surface_mut().unwrap().fill_rect(Rect::new(0, 0, 2, 2), &[0.5, 0.5, 0.5, 1.0]);
        let mut m = solid_layer("m", Rect::new(0, 0, 2, 2), [0.5, 0.5, 0.5, 1.0]);
        m.blend = BlendMode::Multiply;
        let mut g = Layer::group("g", vec![m]);
        g.blend = blend;
        d.layers.push(g);
        px(&d, 0, 0)
    };
    let pass = make(BlendMode::PassThrough);
    let iso = make(BlendMode::Normal);
    assert!((pass[0] - 0.25).abs() <= E, "{pass:?}");
    assert!((iso[0] - 0.5).abs() <= E, "{iso:?}");
}

/// psd-tools passthrough_fill_adjustment.psd: below 100% fill a pass-through group is isolated,
/// so an adjustment inside it no longer changes the layers beneath.
#[test]
fn pass_through_group_below_full_fill_is_isolated() {
    let make = |fill: f32| {
        let mut d = doc_white(2, 2);
        let mut g = Layer::group("g", vec![Layer::new("inv", LayerContent::Adjustment(Adjustment::Invert))]);
        g.blend = BlendMode::PassThrough;
        g.fill_opacity = fill;
        d.layers.push(g);
        px(&d, 0, 0)
    };
    assert!(close4(make(1.0), [0.0, 0.0, 0.0, 1.0]), "full fill: the invert reaches the background");
    assert!(close4(make(0.4), [1.0; 4]), "isolated: nothing beneath to invert");
}

/// psd-tools layer_mask_data.psd: a mask feather blurs the mask edge (CPU and the GPU's
/// combined mask share `masks::combined_mask`).
#[test]
fn mask_feather_blurs_the_edge() {
    let mut d = doc_white(64, 8);
    let mut l = solid_layer("k", Rect::new(0, 0, 64, 8), [0.0, 0.0, 0.0, 1.0]);
    let mut m = LayerMask::hide_all();
    m.surface.fill_rect(Rect::new(0, 0, 32, 8), &[1.0]);
    l.mask = Some(m);
    d.layers.push(l.clone());
    assert!(px(&d, 31, 4)[0] < 0.01 && px(&d, 32, 4)[0] > 0.99, "sharp without feather");
    if let Some(m) = &mut d.layers[1].mask {
        m.feather = 8.0;
    }
    let (a, b) = (px(&d, 31, 4)[0], px(&d, 32, 4)[0]);
    assert!(a > 0.3 && a < 0.5 && b > 0.5 && b < 0.7, "soft edge around 50%: {a} {b}");
    assert!(px(&d, 4, 4)[0] < 0.01 && px(&d, 60, 4)[0] > 0.99, "far from the edge unchanged");
}

#[test]
fn group_opacity_applies_once() {
    let mut d = doc_white(2, 2);
    let a = solid_layer("a", Rect::new(0, 0, 2, 2), [0.0, 0.0, 0.0, 1.0]);
    let b = solid_layer("b", Rect::new(0, 0, 2, 2), [0.0, 0.0, 0.0, 1.0]);
    let mut g = Layer::group("g", vec![a, b]);
    g.blend = BlendMode::Normal;
    g.opacity = 0.5;
    d.layers.push(g);
    assert!((px(&d, 0, 0)[0] - 0.5).abs() <= E);
}

#[test]
fn pass_through_group_opacity_mixes() {
    let mut d = doc_white(2, 2);
    let a = solid_layer("a", Rect::new(0, 0, 2, 2), [0.0, 0.0, 0.0, 1.0]);
    let mut g = Layer::group("g", vec![a]);
    g.opacity = 0.25;
    d.layers.push(g);
    assert!((px(&d, 0, 0)[0] - 0.75).abs() <= E);

    // Opaque backdrops hide straight-alpha interpolation errors.
    let mut transparent = Document::new("coverage", Size::new(2, 2), ColorMode::Rgb, SampleType::F32);
    let mut red = Layer::raster("HDR red", transparent.pixel_format());
    red.surface_mut().unwrap().fill_rect(transparent.bounds(), &[2.0, 0.0, 0.0, 1.0]);
    let mut group = Layer::group("half", vec![red]);
    group.opacity = 0.5;
    transparent.layers.push(group);
    let actual = px(&transparent, 0, 0);
    assert!(close4(actual, [2.0, 0.0, 0.0, 0.5]), "{actual:?}");
}

#[test]
fn invert_adjustment_layer() {
    let mut d = doc_white(2, 2);
    d.layers[0].surface_mut().unwrap().fill_rect(Rect::new(0, 0, 2, 2), &[0.2, 0.4, 0.6, 1.0]);
    d.layers.push(Layer::new("inv", LayerContent::Adjustment(Adjustment::Invert)));
    assert!(close4(px(&d, 0, 0), [0.8, 0.6, 0.4, 1.0]));
}

#[test]
fn adjustment_opacity_and_mask() {
    let mut d = doc_white(4, 1);
    d.layers[0].surface_mut().unwrap().fill_rect(Rect::new(0, 0, 4, 1), &[0.0, 0.0, 0.0, 1.0]);
    let mut adj = Layer::new("inv", LayerContent::Adjustment(Adjustment::Invert));
    adj.opacity = 0.5;
    let mut m = LayerMask::reveal_all();
    m.surface.fill_rect(Rect::new(2, 0, 4, 1), &[0.0]);
    adj.mask = Some(m);
    d.layers.push(adj);
    assert!((px(&d, 0, 0)[0] - 0.5).abs() <= E);
    assert!(px(&d, 3, 0)[0].abs() <= E);
}

#[test]
fn clipped_adjustment_only_affects_base() {
    let mut d = doc_white(4, 1);
    d.layers.push(solid_layer("base", Rect::new(0, 0, 2, 1), [0.0, 0.0, 0.0, 1.0]));
    let mut adj = Layer::new("inv", LayerContent::Adjustment(Adjustment::Invert));
    adj.clipped = true;
    d.layers.push(adj);
    assert!(close4(px(&d, 0, 0), [1.0; 4]), "base inverted to white");
    assert!(close4(px(&d, 3, 0), [1.0; 4]), "background untouched (white)");
}

#[test]
fn threshold_posterize_levels_curves() {
    let mut buf = Buffer::filled(Rect::new(0, 0, 1, 1), [0.3, 0.3, 0.3, 1.0]);
    adjust::apply(&Adjustment::Threshold { level: 0.5 }, &mut buf);
    assert!(close4(buf.px[0], [0.0, 0.0, 0.0, 1.0]));

    let mut buf = Buffer::filled(Rect::new(0, 0, 1, 1), [0.3, 0.6, 0.9, 1.0]);
    adjust::apply(&Adjustment::Posterize { levels: 2 }, &mut buf);
    assert!(close4(buf.px[0], [0.0, 1.0, 1.0, 1.0]));

    let lv = LevelsChannel { in_black: 0.2, in_white: 0.8, ..Default::default() };
    assert!((adjust::levels(&lv, 0.5) - 0.5).abs() < 1e-6);
    assert!((adjust::levels(&lv, 0.2)).abs() < 1e-6);
    assert!((adjust::levels(&lv, 0.9) - 1.0).abs() < 1e-6);

    let identity = adjust::curve_lut(&[CurvePoint { input: 0.0, output: 0.0 }, CurvePoint { input: 1.0, output: 1.0 }]);
    for (i, v) in identity.iter().enumerate().step_by(97) {
        assert!((v - i as f32 / (identity.len() - 1) as f32).abs() < 1e-4);
    }
    let s = adjust::curve_lut(&[
        CurvePoint { input: 0.0, output: 0.0 },
        CurvePoint { input: 0.25, output: 0.15 },
        CurvePoint { input: 0.75, output: 0.85 },
        CurvePoint { input: 1.0, output: 1.0 },
    ]);
    assert!(s.windows(2).all(|w| w[1] >= w[0] - 1e-6), "monotone");
    assert!(s[s.len() / 4] < 0.25 && s[3 * s.len() / 4] > 0.75);
}

#[test]
fn hue_saturation_roundtrips_and_desaturates() {
    let c = [0.8, 0.3, 0.1];
    let (h, s, l) = adjust::rgb_to_hsl(c);
    let back = adjust::hsl_to_rgb(h, s, l);
    for i in 0..3 {
        assert!((back[i] - c[i]).abs() < 1e-5);
    }
    let mut buf = Buffer::filled(Rect::new(0, 0, 1, 1), [0.8, 0.3, 0.1, 1.0]);
    adjust::apply(
        &Adjustment::HueSaturation { hue: 0.0, saturation: -100.0, lightness: 0.0, colorize: false, ranges: photocraft_doc::adjust::HueRange::defaults() },
        &mut buf,
    );
    let p = buf.px[0];
    assert!((p[0] - p[1]).abs() < 1e-5 && (p[1] - p[2]).abs() < 1e-5);
    let mut buf = Buffer::filled(Rect::new(0, 0, 1, 1), [1.0, 0.0, 0.0, 1.0]);
    adjust::apply(
        &Adjustment::HueSaturation { hue: 120.0, saturation: 0.0, lightness: 0.0, colorize: false, ranges: photocraft_doc::adjust::HueRange::defaults() },
        &mut buf,
    );
    assert!(close4(buf.px[0], [0.0, 1.0, 0.0, 1.0]));
}

#[test]
fn solid_and_gradient_fill_layers() {
    let mut d = doc_white(10, 1);
    d.layers.push(Layer::new("fill", LayerContent::Fill(Fill::Solid(Color::rgb(0.0, 0.0, 1.0)))));
    assert!(close4(px(&d, 5, 0), [0.0, 0.0, 1.0, 1.0]));

    let g = Fill::gradient(vec![(0.0, Color::BLACK), (1.0, Color::WHITE)], 0.0, 1.0, photocraft_doc::GradientStyle::Linear, false);
    let patterns = pattern::PreparedPatterns::new(&[], pattern::PREPARED_PATTERN_BYTES);
    let buf = render_fill(&g, Rect::new(0, 0, 10, 1), Rect::new(0, 0, 10, 1), &patterns, None);
    // tile independence: a 1px render of the right edge equals the full render
    let one = render_fill(&g, Rect::new(9, 0, 10, 1), Rect::new(0, 0, 10, 1), &patterns, None);
    assert_eq!(one.px[0], buf.px[9]);
    assert!(buf.px[0][0] < buf.px[9][0], "left dark, right light");
}

fn dithered_gradient(stops: Vec<(f32, Color)>, angle: f32) -> Fill {
    let mut f = Fill::gradient(stops, angle, 1.0, photocraft_doc::GradientStyle::Linear, false);
    if let Fill::Gradient { dither, .. } = &mut f {
        *dither = true;
    }
    f
}

/// #2755: a dithered gradient fill under a bright Color Dodge gradient. Dodge divides by
/// `1 − cs`, so the dither's fraction of a level, left in float, came out as speckle several
/// levels high (a black channel lifted to 4/255: coloured fringes). On an 8-bit document the
/// dithered pixels sit on levels, as a painted gradient's do, so a channel at 0 stays 0.
#[test]
fn dithered_gradient_fills_under_color_dodge_keep_black_channels() {
    let mut d = doc_white(128, 32);
    // Red to blue: green is 0 everywhere.
    let bottom = dithered_gradient(vec![(0.0, Color::rgb(1.0, 0.0, 0.0)), (1.0, Color::rgb(0.0, 0.0, 1.0))], 0.0);
    d.layers.push(Layer::new("bottom", LayerContent::Fill(bottom)));
    let top = dithered_gradient(vec![(0.0, Color::rgb(0.88, 0.88, 0.88)), (1.0, Color::rgb(0.97, 0.97, 0.97))], 90.0);
    let mut l = Layer::new("top", LayerContent::Fill(top));
    l.blend = BlendMode::ColorDodge;
    d.layers.push(l);
    for (i, p) in flatten(&d).px.iter().enumerate() {
        assert_eq!(p[1], 0.0, "pixel {i}: green lifted off black to {} levels", p[1] * 255.0);
    }
}

/// Dithered gradient pixels land on the depth's levels; float documents and undithered
/// gradients keep their float values.
#[test]
fn dithered_gradient_fill_is_rounded_to_the_document_depth() {
    let stops = vec![(0.0, Color::rgb(0.1, 0.4, 0.0)), (1.0, Color::rgb(0.9, 0.2, 1.0))];
    let rect = Rect::new(0, 0, 64, 8);
    let f = dithered_gradient(stops.clone(), 0.0);
    for depth in [SampleType::U8, SampleType::U16] {
        let q = adjustment_quantum(depth).unwrap();
        for p in gradient_fill::render_quantized(&f, rect, rect, Some(q)) {
            for v in &p[..3] {
                assert_eq!((v * q).round() / q, *v, "{depth:?}: {v} is between levels");
            }
        }
    }
    assert_eq!(gradient_fill::render_quantized(&f, rect, rect, None), gradient_fill::render(&f, rect, rect));
    let plain = Fill::gradient(stops, 0.0, 1.0, photocraft_doc::GradientStyle::Linear, false);
    assert_eq!(gradient_fill::render_quantized(&plain, rect, rect, Some(255.0)), gradient_fill::render(&plain, rect, rect));
}

#[test]
fn dissolve_coverage_matches_opacity() {
    let mut d = Document::new("d", Size::new(64, 64), ColorMode::Rgb, SampleType::U8);
    let mut l = solid_layer("k", Rect::new(0, 0, 64, 64), [0.0, 0.0, 0.0, 1.0]);
    l.blend = BlendMode::Dissolve;
    l.opacity = 0.3;
    d.layers.push(l);
    let b = flatten(&d);
    let covered = b.px.iter().filter(|p| p[3] > 0.5).count() as f32 / b.px.len() as f32;
    assert!((covered - 0.3).abs() < 0.05, "{covered}");
    // deterministic
    assert_eq!(flatten(&d), b);
}

#[test]
fn render_is_tile_independent() {
    // Rendering a sub-rect must equal the same region of a full render.
    let mut d = doc_white(300, 300);
    let mut l = solid_layer("a", Rect::new(20, 20, 280, 280), [0.1, 0.5, 0.9, 0.7]);
    l.blend = BlendMode::Overlay;
    d.layers.push(l);
    let full = flatten(&d);
    let sub = render(&d, Rect::new(250, 250, 270, 262));
    for y in 250..262 {
        for x in 250..270 {
            assert_eq!(full.get(x, y), sub.get(x, y));
        }
    }
}

/// A Lab Invert layer in a 16-bit document keeps a grey neutral (the compositor passes the
/// document's depth to `adjust::apply_in`).
#[test]
fn a_16_bit_lab_invert_layer_keeps_greys_neutral() {
    let mut d = Document::new("lab", Size::new(2, 2), ColorMode::Lab, SampleType::U16);
    let mut bg = Layer::raster("bg", d.pixel_format());
    bg.surface_mut().unwrap().fill_rect(Rect::new(0, 0, 2, 2), &[0.25, 128.0 / 255.0, 128.0 / 255.0, 1.0]);
    d.layers.push(bg);
    d.layers.push(Layer::new("inv", LayerContent::Adjustment(Adjustment::Invert)));
    let p = px(&d, 1, 1);
    let lab = photocraft_color::convert::srgb_to_lab([p[0], p[1], p[2]]);
    assert!((lab[0] - 75.0).abs() < 0.1 && lab[1].abs() < 0.05 && lab[2].abs() < 0.05, "{lab:?}");
}

#[test]
fn cmyk_document_renders_via_rgb() {
    let mut d = Document::new("c", Size::new(2, 2), ColorMode::Cmyk, SampleType::U8);
    let mut l = Layer::raster("k", d.pixel_format());
    l.surface_mut().unwrap().fill_rect(Rect::new(0, 0, 2, 2), &[0.0, 0.0, 0.0, 1.0, 1.0]);
    d.layers.push(l);
    // CMYK displays through the built-in CMYK profile: 100 % K alone is a dark neutral
    // (as in any real CMYK profile), not pure black.
    let p = px(&d, 0, 0);
    assert!(p[0] < 0.3 && (p[0] - p[1]).abs() < 0.05 && (p[1] - p[2]).abs() < 0.05 && p[3] == 1.0, "{p:?}");
}

/// Transparent pixels convert to colour in CMYK (K = 100%) and Lab (a/b = 0.5). A flattened
/// surface must still treat them as empty, or its content spans the whole canvas.
#[test]
fn flatten_to_surface_leaves_transparent_pixels_empty_in_every_mode() {
    let modes = [
        (ColorMode::Rgb, [SampleType::U8, SampleType::U16, SampleType::F32].as_slice()),
        (ColorMode::Grayscale, &[SampleType::U8, SampleType::U16, SampleType::F32]),
        (ColorMode::Cmyk, &[SampleType::U8, SampleType::U16]),
        (ColorMode::Lab, &[SampleType::U8, SampleType::U16]),
    ];
    for (mode, samples) in modes {
        for &sample in samples {
            let mut d = Document::new("t", Size::new(300, 200), mode, sample);
            let fmt = PixelFormat::new(mode, sample, true);
            let mut l = Layer::raster("dot", fmt);
            let opaque: Vec<f32> = photocraft_raster::from_rgba(&fmt, [0.9, 0.4, 0.1, 1.0]);
            l.surface_mut().unwrap().fill_rect(Rect::new(120, 80, 140, 90), &opaque);
            d.layers.push(l);
            let s = flatten_to_surface(&d, fmt, None);
            assert_eq!(s.content_bounds(), Rect::new(120, 80, 140, 90), "{mode:?} {sample:?}");
            // Over a background everything is content, as before.
            let fmt_opaque = PixelFormat::new(mode, sample, false);
            let s = flatten_to_surface(&d, fmt_opaque, Some([1.0, 1.0, 1.0]));
            assert_eq!(s.rgba(5, 5)[3], 1.0, "{mode:?} {sample:?}: background stays opaque");
        }
    }
}

#[test]
fn sixteen_bit_and_float_layers_composite() {
    for fmt in [PixelFormat::RGBA16, PixelFormat::RGBA32F] {
        let mut d = Document::new("x", Size::new(2, 2), ColorMode::Rgb, fmt.sample);
        let mut l = Layer::raster("a", fmt);
        l.surface_mut().unwrap().fill_rect(Rect::new(0, 0, 2, 2), &[0.25, 0.5, 0.75, 1.0]);
        d.layers.push(l);
        assert!(close4(px(&d, 1, 1), [0.25, 0.5, 0.75, 1.0]), "{fmt:?}");
    }
}

/// `top` blended with `mode` over an opaque `bg`, both solid, in a 4×4 RGB document of `depth`.
fn two_solid_layers(depth: SampleType, mode: BlendMode, bg: [f32; 3], top: [f32; 3]) -> Document {
    let mut d = Document::new("hdr", Size::new(4, 4), ColorMode::Rgb, depth);
    let fmt = d.pixel_format();
    let mut b = Layer::raster("bg", fmt);
    b.surface_mut().unwrap().fill_rect(Rect::new(0, 0, 4, 4), &photocraft_raster::from_rgba(&fmt, [bg[0], bg[1], bg[2], 1.0]));
    let mut t = Layer::raster("top", fmt);
    t.surface_mut().unwrap().fill_rect(Rect::new(0, 0, 4, 4), &photocraft_raster::from_rgba(&fmt, [top[0], top[1], top[2], 1.0]));
    t.blend = mode;
    d.layers.push(b);
    d.layers.push(t);
    d
}

#[test]
fn float_documents_add_and_divide_past_white() {
    // 32-bit: Linear Dodge (Add) and Divide keep HDR values (adding light never darkens).
    let (bg, top) = ([2.0, 0.5, 4.0], [0.5, 0.25, 0.5]);
    for (mode, want) in [(BlendMode::LinearDodge, [2.5, 0.75, 4.5]), (BlendMode::Divide, [4.0, 2.0, 8.0]), (BlendMode::Multiply, [1.0, 0.125, 2.0])] {
        let d = two_solid_layers(SampleType::F32, mode, bg, top);
        for buf in [flatten(&d), render_tiled(&d, d.bounds(), 2)] {
            for p in &buf.px {
                assert!(close4(*p, [want[0], want[1], want[2], 1.0]), "{mode:?}: {p:?}");
            }
        }
    }
    // The flag doesn't leak into later renders on this thread.
    let d = two_solid_layers(SampleType::U16, BlendMode::LinearDodge, [0.8, 0.5, 0.6], top);
    assert!(close4(flatten(&d).px[0], [1.0, 0.75, 1.0, 1.0]));
}

#[test]
fn integer_documents_clip_add_and_divide_at_white() {
    let (bg, top) = ([0.8, 0.5, 0.6], [0.5, 0.25, 0.5]);
    for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
        let f32_doc = depth == SampleType::F32;
        for (mode, clipped, hdr) in [(BlendMode::LinearDodge, [1.0, 0.75, 1.0], [1.3, 0.75, 1.1]), (BlendMode::Divide, [1.0, 1.0, 1.0], [1.6, 2.0, 1.2])] {
            let want = if f32_doc { hdr } else { clipped };
            let p = flatten(&two_solid_layers(depth, mode, bg, top)).px[5];
            assert!(close4(p, [want[0], want[1], want[2], 1.0]), "{mode:?} {depth:?}: {p:?}");
        }
    }
}

#[test]
fn thumbnail_dimensions() {
    let d = doc_white(400, 200);
    let t = thumbnail(&d, 100);
    assert_eq!((t.width, t.height), (100, 50));
    assert_eq!(&t.pixels[0..4], &[255, 255, 255, 255]);
    let small = doc_white(10, 5);
    let t = thumbnail(&small, 100);
    assert_eq!((t.width, t.height), (10, 5));
}

#[test]
fn buffer_over_background() {
    let b = Buffer::filled(Rect::new(0, 0, 1, 1), [0.0, 0.0, 0.0, 0.5]);
    let o = b.over_background([1.0, 1.0, 1.0]);
    assert!(close4(o.px[0], [0.5, 0.5, 0.5, 1.0]));
    assert_eq!(b.to_rgba8().pixels, vec![0, 0, 0, 128]);
}

// ---------- layer effects ----------

use photocraft_doc::{Effect, FxCommon, FxPaint, Glow, GlowSource, GlowTechnique, Gradient, GradientStyle, Satin, Shadow, StrokeFx, StrokePosition};

fn fx_doc(effects: Vec<Effect>) -> Document {
    let mut d = doc_white(40, 40);
    let mut l = solid_layer("sq", Rect::new(10, 10, 30, 30), [1.0, 0.0, 0.0, 1.0]);
    l.effects.items = effects;
    d.layers.push(l);
    d
}

fn stroke(size: f32, position: StrokePosition) -> Effect {
    Effect::Stroke(StrokeFx {
        common: FxCommon::new(photocraft_color::BlendMode::Normal, 1.0),
        size,
        position,
        paint: FxPaint::Color(Color::rgb(0.0, 0.0, 1.0)),
    })
}

#[test]
fn outside_stroke_width() {
    let d = fx_doc(vec![stroke(3.0, StrokePosition::Outside)]);
    assert!(close4(px(&d, 8, 20), [0.0, 0.0, 1.0, 1.0]));
    assert!(close4(px(&d, 7, 20), [0.0, 0.0, 1.0, 1.0]));
    assert!(close4(px(&d, 5, 20), [1.0; 4]), "{:?}", px(&d, 5, 20));
    assert!(close4(px(&d, 20, 20), [1.0, 0.0, 0.0, 1.0]), "interior untouched");
}

#[test]
fn outside_stroke_at_zero_fill_leaves_the_interior_clear() {
    // Fill 0 % + an Outside stroke is the classic "outline only" look: the stroke lies outside
    // the layer's pixels, so the interior shows the backdrop, not the stroke colour.
    for (depth, fmt) in [(SampleType::U8, PixelFormat::RGBA8), (SampleType::U16, PixelFormat::RGBA16), (SampleType::F32, PixelFormat::RGBA32F)] {
        let mut d = Document::with_background("t", Size::new(40, 40), ColorMode::Rgb, depth, Color::WHITE);
        let mut l = Layer::raster("sq", fmt);
        l.surface_mut().unwrap().fill_rect(Rect::new(10, 10, 30, 30), &[1.0, 0.0, 0.0, 1.0]);
        l.effects.items = vec![stroke(3.0, StrokePosition::Outside)];
        l.fill_opacity = 0.0;
        d.layers.push(l);
        assert!(close4(px(&d, 20, 20), [1.0; 4]), "{depth:?}: interior shows the white backdrop: {:?}", px(&d, 20, 20));
        assert!(close4(px(&d, 11, 20), [1.0; 4]), "{depth:?}: just inside the edge too: {:?}", px(&d, 11, 20));
        assert!(close4(px(&d, 8, 20), [0.0, 0.0, 1.0, 1.0]), "{depth:?}: the stroke itself stays: {:?}", px(&d, 8, 20));
        // Partial fill: the interior is the layer at that fill over the backdrop, still no stroke.
        d.layers[1].fill_opacity = 0.5;
        assert!(close4(px(&d, 20, 20), [1.0, 0.5, 0.5, 1.0]), "{depth:?}: {:?}", px(&d, 20, 20));
        // 100 % fill is unchanged.
        d.layers[1].fill_opacity = 1.0;
        assert!(close4(px(&d, 20, 20), [1.0, 0.0, 0.0, 1.0]), "{depth:?}: {:?}", px(&d, 20, 20));
    }
}

#[test]
fn inside_and_center_strokes() {
    let d = fx_doc(vec![stroke(2.0, StrokePosition::Inside)]);
    assert!(close4(px(&d, 10, 20), [0.0, 0.0, 1.0, 1.0]));
    assert!(close4(px(&d, 11, 20), [0.0, 0.0, 1.0, 1.0]));
    assert!(close4(px(&d, 13, 20), [1.0, 0.0, 0.0, 1.0]));
    assert!(close4(px(&d, 9, 20), [1.0; 4]));
    let d = fx_doc(vec![stroke(4.0, StrokePosition::Center)]);
    assert!(close4(px(&d, 8, 20), [0.0, 0.0, 1.0, 1.0]));
    assert!(close4(px(&d, 11, 20), [0.0, 0.0, 1.0, 1.0]));
    assert!(close4(px(&d, 13, 20), [1.0, 0.0, 0.0, 1.0]));
}

#[test]
fn master_switch_and_per_effect_enable() {
    let mut d = fx_doc(vec![stroke(3.0, StrokePosition::Outside)]);
    d.layers[1].effects.enabled = false;
    assert!(close4(px(&d, 8, 20), [1.0; 4]));
    let mut d = fx_doc(vec![stroke(3.0, StrokePosition::Outside)]);
    if let Effect::Stroke(s) = &mut d.layers[1].effects.items[0] {
        s.common.enabled = false;
    }
    assert!(close4(px(&d, 8, 20), [1.0; 4]));
}

#[test]
fn color_overlay_ignores_fill_opacity_but_not_opacity() {
    let mut d = fx_doc(vec![Effect::ColorOverlay { common: FxCommon::new(photocraft_color::BlendMode::Normal, 1.0), color: Color::rgb(0.0, 1.0, 0.0) }]);
    d.layers[1].fill_opacity = 0.0;
    assert!(close4(px(&d, 20, 20), [0.0, 1.0, 0.0, 1.0]), "overlay shows at fill 0");
    d.layers[1].opacity = 0.5;
    assert!(close4(px(&d, 20, 20), [0.5, 1.0, 0.5, 1.0]), "{:?}", px(&d, 20, 20));
}

fn shadow(distance: f32, angle: f32) -> Shadow {
    Shadow {
        common: FxCommon::new(photocraft_color::BlendMode::Normal, 1.0),
        color: Color::BLACK,
        angle,
        use_global_light: false,
        distance,
        spread: 1.0,
        size: 0.0,
        contour: photocraft_doc::Contour::Linear,
        anti_alias: false,
        noise: 0.0,
        knocks_out: true,
    }
}

#[test]
fn drop_shadow_falls_away_from_light() {
    // Light from the top (90°): shadow below the square.
    let d = fx_doc(vec![Effect::DropShadow(shadow(5.0, 90.0))]);
    assert!(close4(px(&d, 20, 32), [0.0, 0.0, 0.0, 1.0]), "{:?}", px(&d, 20, 32));
    assert!(close4(px(&d, 20, 8), [1.0; 4]));
    // Light from the right (0°, 3 o'clock): shadow to the left.
    let d = fx_doc(vec![Effect::DropShadow(shadow(5.0, 0.0))]);
    assert!(close4(px(&d, 8, 20), [0.0, 0.0, 0.0, 1.0]));
    assert!(close4(px(&d, 32, 20), [1.0; 4]));
}

#[test]
fn group_effects_reach_past_children_effects() {
    // A child's outside stroke reaches further than its group's drop shadow: the group's shadow
    // follows the stroke, not the child's pixels alone (#3274).
    let mut d = doc_white(48, 48);
    let mut sq = solid_layer("sq", Rect::new(14, 14, 34, 34), [1.0, 0.0, 0.0, 1.0]);
    sq.effects.items = vec![stroke(6.0, StrokePosition::Outside)];
    let mut grp = Layer::group("g", vec![sq]);
    grp.effects.items = vec![Effect::DropShadow(shadow(2.0, 90.0))];
    d.layers.push(grp);
    assert!(close4(px(&d, 9, 24), [0.0, 0.0, 1.0, 1.0]), "stroke left of the pixels: {:?}", px(&d, 9, 24));
    assert!(close4(px(&d, 24, 38), [0.0, 0.0, 1.0, 1.0]), "stroke below the pixels: {:?}", px(&d, 24, 38));
    assert!(close4(px(&d, 24, 41), [0.0, 0.0, 0.0, 1.0]), "shadow below the stroke: {:?}", px(&d, 24, 41));
    assert!(close4(px(&d, 24, 43), [1.0; 4]), "nothing past the shadow: {:?}", px(&d, 24, 43));
}

#[test]
fn drop_shadow_uses_global_light_and_knockout() {
    let mut d = fx_doc(vec![Effect::DropShadow(Shadow { use_global_light: true, ..shadow(5.0, 0.0) })]);
    d.global_light.angle = 90.0;
    assert!(close4(px(&d, 20, 32), [0.0, 0.0, 0.0, 1.0]));
    // Knock-out: with fill 0 the shadow does not show through the layer.
    d.layers[1].fill_opacity = 0.0;
    assert!(close4(px(&d, 20, 28), [1.0; 4]), "{:?}", px(&d, 20, 28));
    if let Effect::DropShadow(s) = &mut d.layers[1].effects.items[0] {
        s.knocks_out = false;
    }
    assert!(close4(px(&d, 20, 28), [0.0, 0.0, 0.0, 1.0]));
}

#[test]
fn soft_shadow_is_blurred_and_bounded() {
    let d = fx_doc(vec![Effect::DropShadow(Shadow { spread: 0.0, size: 6.0, ..shadow(0.0, 90.0) })]);
    let edge = px(&d, 30, 20); // just outside the right edge
    let far = px(&d, 39, 20);
    assert!(edge[0] < 0.8 && edge[0] > 0.2, "{edge:?}");
    assert!(far[0] > 0.99, "{far:?}");
}

#[test]
fn inner_shadow_only_inside() {
    let d = fx_doc(vec![Effect::InnerShadow(Shadow { knocks_out: false, ..shadow(4.0, 90.0) })]);
    // Top rows of the square are shadowed (light from the top pushes the
    // outside's shadow down into the shape).
    assert!(close4(px(&d, 20, 11), [0.0, 0.0, 0.0, 1.0]), "{:?}", px(&d, 20, 11));
    assert!(close4(px(&d, 20, 25), [1.0, 0.0, 0.0, 1.0]));
    assert!(close4(px(&d, 20, 5), [1.0; 4]));
}

fn glow(technique: GlowTechnique, source: GlowSource) -> Glow {
    Glow {
        common: FxCommon::new(photocraft_color::BlendMode::Normal, 1.0),
        paint: FxPaint::Color(Color::rgb(0.0, 1.0, 0.0)),
        technique,
        spread: 0.0,
        size: 4.0,
        contour: photocraft_doc::Contour::Linear,
        anti_alias: false,
        range: 0.5,
        jitter: 0.0,
        noise: 0.0,
        source,
    }
}

#[test]
fn outer_and_inner_glow_regions() {
    let d = fx_doc(vec![Effect::OuterGlow(glow(GlowTechnique::Precise, GlowSource::Edge))]);
    let near = px(&d, 9, 20);
    assert!(near[1] > 0.7 && near[0] < 0.3, "{near:?}");
    assert!(close4(px(&d, 3, 20), [1.0; 4]));
    assert!(close4(px(&d, 20, 20), [1.0, 0.0, 0.0, 1.0]));
    let d = fx_doc(vec![Effect::InnerGlow(glow(GlowTechnique::Softer, GlowSource::Edge))]);
    let edge = px(&d, 10, 20);
    assert!(edge[1] > 0.3, "{edge:?}");
    assert!(close4(px(&d, 20, 20), [1.0, 0.0, 0.0, 1.0]));
    assert!(close4(px(&d, 5, 20), [1.0; 4]));
}

// A Precise inner glow from the centre is brightest in the middle and fades towards the edge,
// like the Softer one; it used to paint nothing at all (#966).
#[test]
fn precise_inner_glow_from_center_paints_the_middle() {
    let d = fx_doc(vec![Effect::InnerGlow(glow(GlowTechnique::Precise, GlowSource::Center))]);
    let (center, edge) = (px(&d, 20, 20), px(&d, 10, 20));
    assert!(center[1] > 0.9 && center[0] < 0.1, "green in the middle: {center:?}");
    assert!(edge[0] > center[0] && edge[1] < center[1], "fades towards the edge: {edge:?} vs {center:?}");
    assert!(close4(px(&d, 5, 20), [1.0; 4]));
}

// A glow gradient runs along the glow, not across the canvas: the first stop hugs the shape on
// every side and later stops lie further out, opaque until the glow fades (#443).
#[test]
fn glow_gradient_follows_the_shape() {
    let red_to_blue = Gradient { stops: vec![(0.0, Color::rgb(1.0, 0.0, 0.0)), (1.0, Color::rgb(0.0, 0.0, 1.0))], ..Gradient::default() };
    let g = Glow { paint: FxPaint::Gradient(red_to_blue), technique: GlowTechnique::Softer, size: 8.0, ..glow(GlowTechnique::Softer, GlowSource::Edge) };
    let d = fx_doc(vec![Effect::OuterGlow(g.clone())]);
    let near = px(&d, 9, 20);
    for p in [px(&d, 30, 20), px(&d, 20, 9), px(&d, 20, 30)] {
        assert!((0..4).all(|c| (p[c] - near[c]).abs() < 1.0 / 255.0), "{p:?} vs {near:?}");
    }
    assert!(near[0] > 0.8 && near[2] < 0.2 && near[1] < 0.1, "red at the edge: {near:?}");
    let out = px(&d, 5, 20);
    assert!(out[2] > out[0] && out[1] < 0.9, "bluer and fading further out: {out:?}");
    // An inner glow's gradient also starts at the edge.
    let d = fx_doc(vec![Effect::InnerGlow(g)]);
    let (edge, top) = (px(&d, 10, 20), px(&d, 20, 10));
    assert!((0..4).all(|c| (edge[c] - top[c]).abs() < 1.0 / 255.0), "{edge:?} vs {top:?}");
    assert!(edge[0] > 0.8 && edge[2] < 0.2, "{edge:?}");
}

#[test]
fn gradient_overlay_follows_angle_and_reverse() {
    let g = Gradient { stops: vec![(0.0, Color::BLACK), (1.0, Color::WHITE)], angle: 0.0, ..Gradient::default() };
    let d = fx_doc(vec![Effect::GradientOverlay { common: FxCommon::new(photocraft_color::BlendMode::Normal, 1.0), gradient: g.clone(), dither: false }]);
    let (l, r) = (px(&d, 10, 20), px(&d, 29, 20));
    assert!(l[0] < 0.1 && r[0] > 0.9, "{l:?} {r:?}");
    let d = fx_doc(vec![Effect::GradientOverlay {
        common: FxCommon::new(photocraft_color::BlendMode::Normal, 1.0),
        gradient: Gradient { reverse: true, style: GradientStyle::Linear, ..g },
        dither: false,
    }]);
    assert!(px(&d, 10, 20)[0] > 0.9);
}

#[test]
fn satin_and_bevel_stay_inside_shape() {
    let satin = Effect::Satin(Satin {
        common: FxCommon::new(photocraft_color::BlendMode::Multiply, 1.0),
        color: Color::BLACK,
        angle: 19.0,
        distance: 4.0,
        size: 4.0,
        contour: photocraft_doc::Contour::Linear,
        anti_alias: true,
        invert: false,
    });
    let d = fx_doc(vec![
        satin,
        Effect::BevelEmboss(photocraft_doc::Bevel {
            enabled: true,
            style: photocraft_doc::BevelStyle::InnerBevel,
            technique: photocraft_doc::BevelTechnique::Smooth,
            depth: 1.0,
            up: true,
            size: 4.0,
            soften: 0.0,
            angle: 90.0,
            altitude: 30.0,
            use_global_light: false,
            gloss_contour: photocraft_doc::Contour::Linear,
            highlight: FxCommon::new(photocraft_color::BlendMode::Screen, 0.75),
            highlight_color: Color::WHITE,
            shadow: FxCommon::new(photocraft_color::BlendMode::Multiply, 0.75),
            shadow_color: Color::BLACK,
            contour: None,
            texture: None,
        }),
    ]);
    for (x, y) in [(5, 5), (35, 20), (20, 35)] {
        assert!(close4(px(&d, x, y), [1.0; 4]), "({x},{y}) {:?}", px(&d, x, y));
    }
    // Light from the top: the top bevel edge is brighter than the bottom one.
    let (top, bot) = (px(&d, 20, 11), px(&d, 20, 28));
    assert!(top[1] > bot[1], "{top:?} {bot:?}");
}

#[test]
fn clipped_layer_effects_are_clipped_to_base() {
    let mut d = doc_white(40, 40);
    d.layers.push(solid_layer("base", Rect::new(10, 10, 30, 30), [1.0, 0.0, 0.0, 1.0]));
    let mut c = solid_layer("clip", Rect::new(20, 10, 40, 30), [0.0, 1.0, 0.0, 1.0]);
    c.clipped = true;
    c.effects.items.push(stroke(3.0, StrokePosition::Outside));
    d.layers.push(c);
    assert!(close4(px(&d, 18, 20), [0.0, 0.0, 1.0, 1.0]), "stroke inside base");
    assert!(close4(px(&d, 20, 8), [1.0; 4]), "stroke clipped outside base");
}

#[test]
fn group_effects_apply_to_group_shape() {
    let mut d = doc_white(40, 40);
    let mut g = Layer::group(
        "g",
        vec![solid_layer("a", Rect::new(10, 10, 20, 30), [1.0, 0.0, 0.0, 1.0]), solid_layer("b", Rect::new(20, 10, 30, 30), [1.0, 0.0, 0.0, 1.0])],
    );
    g.effects.items.push(stroke(2.0, StrokePosition::Outside));
    d.layers.push(g);
    assert!(close4(px(&d, 9, 20), [0.0, 0.0, 1.0, 1.0]));
    assert!(close4(px(&d, 20, 20), [1.0, 0.0, 0.0, 1.0]), "no stroke at the seam between children");
}

#[test]
fn effects_render_identically_in_tiles() {
    for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
        let mut d = fx_doc(vec![Effect::DropShadow(Shadow { spread: 0.0, size: 5.0, ..shadow(4.0, 120.0) }), stroke(2.0, StrokePosition::Outside)]);
        d.depth = depth;
        for layer in &mut d.layers {
            if let Some(surface) = layer.surface_mut() {
                *surface = surface.convert(photocraft_color::PixelFormat { sample: depth, ..surface.format() });
            }
        }
        let full = flatten(&d);
        // Compare every boundary pixel: a few sampled points miss application-halo regressions.
        for tile in [1, 7, 33] {
            assert_eq!(render_tiled(&d, d.bounds(), tile), full, "{depth:?}, tile {tile}");
        }
    }
}

#[test]
fn parallel_tiles_match_single_pass() {
    // A document exercising masks, clipping, groups, adjustments, fills and effects.
    let mut d = doc_white(97, 61);
    let mut a = solid_layer("a", Rect::new(5, 5, 70, 50), [1.0, 0.2, 0.1, 0.8]);
    let mut m = LayerMask::reveal_all();
    m.surface.fill_rect(Rect::new(0, 0, 40, 61), &[0.25]);
    m.density = 0.8;
    a.mask = Some(m);
    a.effects.items.push(stroke(3.0, StrokePosition::Outside));
    d.layers.push(a);
    let mut clip = solid_layer("clip", Rect::new(30, 0, 97, 61), [0.0, 1.0, 0.0, 1.0]);
    clip.clipped = true;
    clip.blend = photocraft_color::BlendMode::Multiply;
    d.layers.push(clip);
    let mut g = Layer::group("g", vec![solid_layer("in", Rect::new(50, 20, 90, 60), [0.2, 0.3, 0.9, 1.0])]);
    g.opacity = 0.6;
    d.layers.push(g);
    d.layers.push(Layer::new("inv", LayerContent::Adjustment(Adjustment::Invert)));
    let mut f = Layer::new("fill", LayerContent::Fill(Fill::Solid(Color::rgb(0.5, 0.5, 0.0))));
    f.opacity = 0.3;
    d.layers.push(f);
    let r = d.bounds();
    let whole = render_tiled(&d, r, 10_000);
    for tile in [7, 16, 33] {
        assert_eq!(render_tiled(&d, r, tile), whole, "tile {tile}");
    }
    assert_eq!(render(&d, r), whole);
}

#[test]
fn vector_mask_combines_with_pixel_mask() {
    use photocraft_doc::{Path, Subpath, VectorMask};
    let mut d = doc_white(8, 8);
    let mut l = solid_layer("k", Rect::new(0, 0, 8, 8), [0.0, 0.0, 0.0, 1.0]);
    // Vector mask reveals x in 0..4 (and half of column 4); the pixel mask hides rows 0..2.
    l.vector_mask = Some(VectorMask::new(Path::new(vec![Subpath::polygon(&[(0.0, 0.0), (4.5, 0.0), (4.5, 8.0), (0.0, 8.0)])])));
    let mut m = LayerMask::reveal_all();
    m.surface.fill_rect(Rect::new(0, 0, 8, 2), &[0.0]);
    l.mask = Some(m);
    d.layers.push(l);
    assert!(close4(px(&d, 1, 4), [0.0, 0.0, 0.0, 1.0]));
    assert!(close4(px(&d, 6, 4), [1.0; 4]));
    assert!(close4(px(&d, 4, 4), [0.5, 0.5, 0.5, 1.0]));
    assert!(close4(px(&d, 1, 1), [1.0; 4]));
    // Tiled rendering gives the same result.
    let whole = render_tiled(&d, d.bounds(), 3);
    assert_eq!(whole.px, render_tiled(&d, d.bounds(), 256).px);
    // Density and disabling.
    let top = d.layers.len() - 1;
    let vm = d.layers[top].vector_mask.as_mut().unwrap();
    vm.density = 0.5;
    assert!(close4(px(&d, 6, 4), [0.5, 0.5, 0.5, 1.0]));
    d.layers[top].vector_mask.as_mut().unwrap().enabled = false;
    assert!(close4(px(&d, 6, 4), [0.0, 0.0, 0.0, 1.0]));
    // Without a pixel mask the vector mask alone applies.
    d.layers[top].mask = None;
    d.layers[top].vector_mask.as_mut().unwrap().enabled = true;
    d.layers[top].vector_mask.as_mut().unwrap().density = 1.0;
    assert!(close4(px(&d, 1, 1), [0.0, 0.0, 0.0, 1.0]));
    assert!(close4(px(&d, 6, 1), [1.0; 4]));
}

#[test]
fn effect_maps_are_cached_and_invalidated_by_pixel_changes() {
    let mut doc = doc_white(64, 64);
    let mut l = solid_layer("fx", Rect::new(16, 16, 48, 48), [1.0, 0.0, 0.0, 1.0]);
    l.effects.items.push(photocraft_doc::Effect::default_drop_shadow());
    doc.layers.push(l);
    let patterns = pattern::PreparedPatterns::new(&doc.patterns, pattern::PREPARED_PATTERN_BYTES);
    let cx = Ctx::for_doc(&doc, &patterns);
    let a = effect_maps(&doc.layers[1], doc.bounds(), &cx);
    let b = effect_maps(&doc.layers[1], doc.bounds(), &cx);
    assert!(std::sync::Arc::ptr_eq(&a, &b), "second request hits the cache");
    let first = flatten(&doc);
    // Editing the layer's pixels changes its tiles, so the maps are rebuilt.
    doc.layers[1].surface_mut().unwrap().fill_rect(Rect::new(8, 8, 20, 20), &[0.0, 0.0, 1.0, 1.0]);
    let patterns = pattern::PreparedPatterns::new(&doc.patterns, pattern::PREPARED_PATTERN_BYTES);
    let cx = Ctx::for_doc(&doc, &patterns);
    let c = effect_maps(&doc.layers[1], doc.bounds(), &cx);
    assert!(!std::sync::Arc::ptr_eq(&a, &c), "pixel edit invalidates");
    // Cached rendering equals a fresh build (tiled and full renders agree too).
    let again = flatten(&doc);
    let region = doc.bounds();
    let tiled = render_tiled(&doc, region, 16);
    for (p, q) in again.px.iter().zip(&tiled.px) {
        assert!(close4(*p, *q));
    }
    assert_ne!(first.px, again.px);

    // Tileless masks differ only in their defaults; reusing their effect maps leaves a ghost shadow.
    let mut masked = fx_doc(vec![Effect::DropShadow(shadow(5.0, 90.0))]);
    masked.layers[1].mask = Some(photocraft_doc::LayerMask::reveal_all());
    assert!(close4(px(&masked, 20, 32), [0.0, 0.0, 0.0, 1.0]));
    masked.layers[1].mask = Some(photocraft_doc::LayerMask::hide_all());
    let warm = flatten(&masked);
    let mut cold = masked.clone();
    cold.layers[1].id = Layer::raster("uncached", cold.pixel_format()).id;
    assert!(warm.px == flatten(&cold).px, "default-only mask changes must invalidate cached effects");
    assert!(close4(px(&masked, 20, 32), [1.0; 4]));
}

// ---------- Transparency Shapes Layer off on tall documents (#1909) ----------

fn bevel_fx(style: photocraft_doc::BevelStyle, soften: f32) -> Effect {
    Effect::BevelEmboss(photocraft_doc::Bevel {
        enabled: true,
        style,
        technique: photocraft_doc::BevelTechnique::Smooth,
        depth: 1.0,
        up: true,
        size: 5.0,
        soften,
        angle: 120.0,
        altitude: 30.0,
        use_global_light: false,
        gloss_contour: photocraft_doc::Contour::Linear,
        highlight: FxCommon::new(photocraft_color::BlendMode::Screen, 0.75),
        highlight_color: Color::WHITE,
        shadow: FxCommon::new(photocraft_color::BlendMode::Multiply, 0.75),
        shadow_color: Color::BLACK,
        contour: None,
        texture: None,
    })
}

/// A tall page of layers with Transparency Shapes Layer off, one effect stack each (every local
/// kind), most masked to a soft-edged block so their effect shapes vary down the page.
fn shapeless_fx_doc(depth: SampleType) -> Document {
    use photocraft_doc::BevelStyle;
    let mut d = Document::with_background("t", Size::new(48, 640), ColorMode::Rgb, depth, Color::WHITE);
    let satin = Effect::Satin(Satin {
        common: FxCommon::new(photocraft_color::BlendMode::Multiply, 1.0),
        color: Color::BLACK,
        angle: 19.0,
        distance: 4.0,
        size: 5.0,
        contour: photocraft_doc::Contour::Linear,
        anti_alias: true,
        invert: false,
    });
    let stacks = vec![
        vec![stroke(3.0, StrokePosition::Outside)],
        vec![stroke(2.0, StrokePosition::Inside), Effect::DropShadow(Shadow { size: 6.0, spread: 0.3, noise: 0.2, ..shadow(5.0, 120.0) })],
        vec![stroke(4.0, StrokePosition::Center)],
        vec![Effect::InnerShadow(Shadow { size: 4.0, spread: 0.0, ..shadow(3.0, 30.0) })],
        vec![Effect::OuterGlow(Glow { size: 6.0, spread: 0.2, noise: 0.1, ..glow(GlowTechnique::Softer, GlowSource::Edge) })],
        vec![Effect::InnerGlow(glow(GlowTechnique::Softer, GlowSource::Center)), satin],
        vec![bevel_fx(BevelStyle::Emboss, 2.0)],
        vec![bevel_fx(BevelStyle::PillowEmboss, 0.0), bevel_fx(BevelStyle::OuterBevel, 1.0)],
        vec![stroke(3.0, StrokePosition::Outside), Effect::InnerGlow(glow(GlowTechnique::Softer, GlowSource::Edge))],
    ];
    let mask_fmt = PixelFormat::new(ColorMode::Grayscale, depth, false);
    for (i, fx) in stacks.into_iter().enumerate() {
        let y = 24 + i as i32 * 66;
        let mut l = Layer::raster(format!("fx{i}"), d.pixel_format());
        let c = [0.2 + 0.08 * i as f32, 0.5, 1.0 - 0.1 * i as f32, 1.0];
        l.surface_mut().unwrap().fill_rect(Rect::new(10, y, 38, y + 18), &photocraft_raster::from_rgba(&d.pixel_format(), c));
        l.effects.items = fx;
        l.advanced.transparency_shapes = false;
        l.opacity = 0.8;
        // The last layer stays unmasked: its shape is the whole canvas, edged only at its sides.
        if i < 8 {
            let mut m = photocraft_raster::Surface::new(mask_fmt);
            m.fill_rect(Rect::new(4 + i as i32, y - 12, 44, y + 30), &[1.0]);
            m.fill_rect(Rect::new(4 + i as i32, y + 30, 30, y + 34), &[0.5]);
            l.mask = Some(LayerMask { surface: m, enabled: true, linked: true, density: 1.0, feather: 0.0 });
        }
        d.layers.push(l);
    }
    // A tsly-off stroked layer inside a group whose own effects reach much further: the group's
    // effect shape composites its children over the group's whole region, far past any band.
    let mut inner = Layer::raster("inner", d.pixel_format());
    inner.surface_mut().unwrap().fill_rect(Rect::new(12, 300, 30, 310), &photocraft_raster::from_rgba(&d.pixel_format(), [0.1, 0.9, 0.3, 1.0]));
    inner.effects.items = vec![stroke(3.0, StrokePosition::Outside)];
    inner.advanced.transparency_shapes = false;
    let mut m = photocraft_raster::Surface::new(mask_fmt);
    m.fill_rect(Rect::new(8, 240, 36, 380), &[1.0]);
    inner.mask = Some(LayerMask { surface: m, enabled: true, linked: true, density: 1.0, feather: 0.0 });
    let mut g = Layer::group("g", vec![inner]);
    g.effects.items = vec![Effect::DropShadow(Shadow { size: 40.0, spread: 0.1, ..shadow(90.0, 70.0) }), stroke(120.0, StrokePosition::Outside)];
    g.opacity = 0.6;
    d.layers.push(g);
    // A clipping mask: a tsly-off stroked layer clipped to a base with large effects of its own.
    let mut base = Layer::raster("base", d.pixel_format());
    base.surface_mut().unwrap().fill_rect(Rect::new(6, 420, 42, 470), &photocraft_raster::from_rgba(&d.pixel_format(), [0.9, 0.9, 0.2, 1.0]));
    base.effects.items = vec![Effect::OuterGlow(Glow { size: 150.0, ..glow(GlowTechnique::Softer, GlowSource::Edge) })];
    let mut clip = Layer::raster("clip", d.pixel_format());
    clip.surface_mut().unwrap().fill_rect(Rect::new(14, 430, 34, 450), &photocraft_raster::from_rgba(&d.pixel_format(), [0.6, 0.1, 0.6, 1.0]));
    clip.effects.items = vec![stroke(2.0, StrokePosition::Center), Effect::InnerShadow(Shadow { size: 3.0, spread: 0.0, ..shadow(2.0, 45.0) })];
    clip.advanced.transparency_shapes = false;
    clip.clipped = true;
    d.layers.push(base);
    d.layers.push(clip);
    d
}

#[test]
fn shapeless_effects_render_the_same_in_bands_and_parts() {
    for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
        let d = shapeless_fx_doc(depth);
        // Bands and parts first, from an empty cache: maps a band builds (a group's, say) must not
        // borrow from a whole render's.
        purge_effect_cache();
        let mut banded = Vec::new();
        render_bands(&d, d.bounds(), 64, |b| -> Result<(), ()> {
            banded.extend_from_slice(&b.px);
            Ok(())
        })
        .unwrap();
        purge_effect_cache();
        let parts: Vec<(Rect, Buffer)> =
            [Rect::new(3, 290, 41, 333), Rect::new(0, 0, 48, 20), Rect::new(20, 600, 48, 640), Rect::new(0, 296, 20, 312), Rect::new(10, 425, 40, 445)]
                .into_iter()
                .map(|r| (r, render(&d, r)))
                .collect();
        let whole = render(&d, d.bounds());
        assert_eq!(banded.len(), whole.px.len());
        let diff = banded.iter().zip(&whole.px).position(|(a, b)| a != b);
        assert!(diff.is_none(), "{depth:?}: band render differs at {diff:?}");
        // A small part (an edit's damage) and a tiled render agree with the whole render too.
        for (part, p) in &parts {
            let part = *part;
            for y in part.y0..part.y1 {
                for x in part.x0..part.x1 {
                    assert_eq!(p.get(x, y), whole.get(x, y), "{depth:?}: part {part:?} differs at ({x},{y})");
                }
            }
        }
        assert!(render_tiled(&d, d.bounds(), 16).px == whole.px, "{depth:?}: tiled render differs");
    }
}

#[test]
fn shapeless_effect_maps_cover_only_the_band_being_rendered() {
    // Transparency Shapes Layer off makes an effect region the whole canvas: on a 1920 × 103675
    // page each stroked type layer's maps took ~2.4 GB, for every layer at once (#1909). Rendering
    // a band builds them over that band (plus the effects' reach) only.
    let d = shapeless_fx_doc(SampleType::U8);
    let patterns = pattern::PreparedPatterns::new(&d.patterns, pattern::PREPARED_PATTERN_BYTES);
    let cx = Ctx::for_doc(&d, &patterns);
    let l = &d.layers[9];
    let whole = effect_maps(l, Rect::new(0, 256, 48, 320), &cx).bytes();
    cx.set_fx_clip(Rect::new(0, 256, 48, 320));
    let band = effect_maps(l, Rect::new(0, 256, 48, 320), &cx).bytes();
    assert!(band * 4 < whole, "band maps {band} B vs whole-canvas maps {whole} B");
    // The next band lets the previous band's maps go.
    cx.set_fx_clip(Rect::new(0, 320, 48, 384));
    let _ = effect_maps(l, Rect::new(0, 320, 48, 384), &cx);
    assert_eq!(cx.fx_maps.lock().unwrap().len(), 2, "the unclipped maps and this band's");
    // Composited over more than the band (a group's effect shape renders its children over the
    // group's whole region): the whole region.
    let full = effect_region(l, Rect::new(0, 320, 48, 384), &cx).0;
    assert_eq!(
        effect_region(l, Rect::new(0, 200, 48, 500), &cx),
        (layer_bounds(l, cx.canvas).inflate(effects::margin(l)).intersect(&cx.canvas.inflate(effects::margin(l))), false)
    );
    assert!(full.height() < 100, "{full:?}");
    // Effects whose maps aren't local keep the whole region.
    let mut g = l.clone();
    g.effects.items = vec![Effect::OuterGlow(glow(GlowTechnique::Precise, GlowSource::Edge))];
    assert_eq!(
        effect_region(&g, Rect::new(0, 320, 48, 384), &cx),
        (layer_bounds(&g, cx.canvas).inflate(effects::margin(&g)).intersect(&cx.canvas.inflate(effects::margin(&g))), false)
    );
}

// ---------- PSD-fidelity effect semantics (fitted on Photoshop composites) ----------

#[test]
fn stroke_edges_use_the_chamfer_when_narrow_and_the_exact_distance_when_wide() {
    // #3646: a narrow stroke's outer edge keeps the 5 × 5 chamfer, whose sub-pixel seeding
    // Photoshop matches at small offsets (see `stroke_outside_exact`); only a wide one — where the
    // chamfer's octagon facets (see `a_wide_outside_stroke_is_round_not_a_polygon`) — uses the
    // exact Euclidean transform.
    // Square 10..30, outside stroke 3: (9, 7) is offset (1, 3) from the corner pixel. Photoshop
    // measures √5 + 1 = 3.236 there, so coverage is 3 + 1 − 3.236 = 0.764 (the exact distance is
    // 3.162, coverage 0.838).
    let d = fx_doc(vec![stroke(3.0, StrokePosition::Outside)]);
    let a = 1.0 - px(&d, 9, 7)[0]; // blue stroke over white: red channel drops by coverage
    assert!((a - 0.764).abs() < 0.01, "{a}");
    // (2, 2) is the same in both metrics (two diagonal steps).
    let a = 1.0 - px(&d, 8, 8)[0];
    assert!((a - 1.0).abs() < 0.01, "{a}");
}

#[test]
fn the_stroke_metric_switches_at_the_fitted_threshold() {
    // The widest outside stroke any Photoshop oracle matches with the chamfer is 18 px
    // (ag-psd read-write/strokes): its octagon deviates 0.48 px from a circle there; 20 px keeps
    // it. Wider strokes (the 60 px #3646 case) are round.
    assert!(!crate::effects::stroke_outside_exact(18.0));
    assert!(crate::effects::stroke_outside_exact(crate::effects::STROKE_CHAMFER_MAX));
    assert!(crate::effects::stroke_outside_exact(60.0));
}

#[test]
fn a_wide_outside_stroke_is_round_not_a_polygon() {
    // A point source dilated by 60 px must stay a circle. The 5 × 5 chamfer's octagonal level set
    // made the boundary swing ~2.5 px at this radius (~10 px at 245 px) — the "low-poly" #3646.
    let mut d = doc_white(170, 170);
    let mut l = solid_layer("dot", Rect::new(85, 85, 86, 86), [1.0, 0.0, 0.0, 1.0]);
    l.effects.items = vec![stroke(60.0, StrokePosition::Outside)];
    d.layers.push(l);
    let buf = render(&d, Rect::new(0, 0, 170, 170));
    let at = |x: i32, y: i32| buf.px[(y as usize) * 170 + x as usize];
    let (mut lo, mut hi) = (f64::MAX, 0.0f64);
    let stroke_at = |x: i32, y: i32| x >= 0 && y >= 0 && x < 170 && y < 170 && at(x, y)[2] > 0.5 && at(x, y)[0] < 0.5;
    for k in 0..72 {
        let th = (k as f64) * std::f64::consts::TAU / 72.0;
        let (dx, dy) = (th.cos(), th.sin());
        let mut r = 0.0f64;
        let mut t = 0.0f64;
        while t < 84.0 {
            if stroke_at((85.0 + dx * t).round() as i32, (85.0 + dy * t).round() as i32) {
                r = t;
            }
            t += 0.25;
        }
        lo = lo.min(r);
        hi = hi.max(r);
    }
    assert!(hi - lo <= 2.0, "the 60 px stroke boundary swings {lo}..{hi} px — a polygon, not a circle (#3646)");
}

#[test]
fn inside_stroke_respects_partial_edge_coverage() {
    // A 25 %-alpha edge column puts the edge 0.75 px further out: a 3 px inside stroke then
    // reaches 75 % into the fourth column, and the edge column takes the stroke colour at 25 %.
    let mut d = doc_white(40, 40);
    let mut l = solid_layer("sq", Rect::new(10, 10, 30, 30), [1.0, 0.0, 0.0, 1.0]);
    l.surface_mut().unwrap().fill_rect(Rect::new(10, 10, 11, 30), &[1.0, 0.0, 0.0, 0.25]);
    l.effects.items = vec![stroke(3.0, StrokePosition::Inside)];
    d.layers.push(l);
    let edge = px(&d, 10, 20);
    assert!(close4(edge, [0.75, 0.75, 1.0, 1.0]), "edge = blue stroke at 25 % over white: {edge:?}");
    let p = px(&d, 13, 20);
    assert!((p[2] - 0.75).abs() < 0.02 && (p[0] - 0.25).abs() < 0.02, "{p:?}");
}

#[test]
fn emboss_shades_the_composited_layer() {
    // Emboss styles paint over the layer as composited (for type: mixed at the text gamma), so a
    // Normal white highlight `k` on an edge pixel keeps (1 − R) = (1 − k)(1 − C) whatever C is:
    // the ratio of type to raster results equals that of their plain composites.
    let emboss = Effect::BevelEmboss(photocraft_doc::Bevel {
        enabled: true,
        style: photocraft_doc::BevelStyle::Emboss,
        technique: photocraft_doc::BevelTechnique::Smooth,
        depth: 1.0,
        up: true,
        size: 6.0,
        soften: 0.0,
        angle: 90.0,
        altitude: 30.0,
        use_global_light: false,
        gloss_contour: photocraft_doc::Contour::Linear,
        highlight: FxCommon::new(photocraft_color::BlendMode::Normal, 1.0),
        highlight_color: Color::WHITE,
        shadow: FxCommon::new(photocraft_color::BlendMode::Multiply, 0.0),
        shadow_color: Color::BLACK,
        contour: None,
        texture: None,
    });
    let make = |text: bool, fx: bool| {
        let mut d = doc_white(40, 40);
        let mut l = solid_layer("sq", Rect::new(10, 10, 30, 30), [0.2, 0.1, 0.6, 1.0]);
        l.surface_mut().unwrap().fill_rect(Rect::new(10, 10, 30, 11), &[0.2, 0.1, 0.6, 0.5]);
        if text {
            let t = photocraft_doc::TextLayer { cache: l.surface().cloned(), ..Default::default() };
            l = Layer::new("t", LayerContent::Text(t));
        }
        if fx {
            l.effects.items = vec![emboss.clone()];
        }
        d.layers.push(l);
        px(&d, 20, 10)
    };
    let (ct, cr, rt, rr) = (make(true, false), make(false, false), make(true, true), make(false, true));
    assert!(rr[1] > cr[1] + 0.05, "the top edge is lit: {rr:?} vs {cr:?}");
    assert!((ct[1] - cr[1]).abs() > 0.02, "text gamma changes the edge: {ct:?} {cr:?}");
    for c in 0..3 {
        let want = (1.0 - ct[c]) / (1.0 - cr[c]);
        assert!(((1.0 - rt[c]) / (1.0 - rr[c]) - want).abs() < 2e-3, "channel {c}: {rt:?} {rr:?} {ct:?} {cr:?}");
    }
}

#[test]
fn interior_effects_keep_the_layer_alpha() {
    // A colour overlay replaces a half-transparent pixel's colour without adding coverage.
    let mut d = Document::new("t", Size::new(40, 40), ColorMode::Rgb, SampleType::U8);
    let mut l = solid_layer("sq", Rect::new(10, 10, 30, 30), [1.0, 0.0, 0.0, 0.5]);
    l.effects.items = vec![Effect::ColorOverlay { common: FxCommon::new(photocraft_color::BlendMode::Normal, 1.0), color: Color::rgb(0.0, 1.0, 0.0) }];
    d.layers.push(l);
    assert!(close4(px(&d, 20, 20), [0.0, 1.0, 0.0, 0.5]), "{:?}", px(&d, 20, 20));
}

#[test]
fn upper_stroke_blends_with_the_backdrop_not_the_lower_stroke() {
    // Multiply yellow (3 px) listed above a Normal blue (8 px): the inner ring multiplies the
    // white backdrop (stays yellow), the outer ring is blue.
    let yellow = Effect::Stroke(StrokeFx {
        common: FxCommon::new(photocraft_color::BlendMode::Multiply, 1.0),
        size: 3.0,
        position: StrokePosition::Outside,
        paint: FxPaint::Color(Color::rgb(1.0, 1.0, 0.0)),
    });
    let d = fx_doc(vec![yellow, stroke(8.0, StrokePosition::Outside)]);
    assert!(close4(px(&d, 8, 20), [1.0, 1.0, 0.0, 1.0]), "{:?}", px(&d, 8, 20));
    assert!(close4(px(&d, 4, 20), [0.0, 0.0, 1.0, 1.0]), "{:?}", px(&d, 4, 20));
}

#[test]
fn shadow_knockout_needs_see_through_fill() {
    // At 100 % fill an anti-aliased edge is not attenuated twice: the result equals no knockout.
    let mut base = Document::new("t", Size::new(40, 40), ColorMode::Rgb, SampleType::U8);
    base.layers.push(solid_layer("bg", Rect::new(0, 0, 40, 40), [1.0; 4]));
    let mut l = solid_layer("sq", Rect::new(10, 10, 30, 30), [1.0, 0.0, 0.0, 1.0]);
    l.surface_mut().unwrap().fill_rect(Rect::new(10, 10, 11, 30), &[1.0, 0.0, 0.0, 0.5]);
    l.effects.items = vec![Effect::DropShadow(Shadow { distance: 0.0, spread: 1.0, size: 4.0, ..shadow(0.0, 0.0) })];
    base.layers.push(l);
    let on = px(&base, 10, 20);
    if let Effect::DropShadow(s) = &mut base.layers[1].effects.items[0] {
        s.knocks_out = false;
    }
    assert!(close4(on, px(&base, 10, 20)), "{on:?}");
}

#[test]
fn shadow_knockout_beneath_a_layer_in_a_blending_mode() {
    // Layer Knocks Out Drop Shadow keeps the shadow out from under the layer's shape. In Normal
    // mode an opaque layer covers it anyway; in a mode that blends with what's beneath, the shadow
    // would show through: under a white Exclusion square on white, it read as white (white over
    // the black shadow) instead of black (white over white), so a logo's Exclusion bar with a
    // drop shadow came out blank. A red Multiply square came out black instead of red.
    for (blend, colour, inside, depth) in [
        (photocraft_color::BlendMode::Exclusion, [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]),
        (photocraft_color::BlendMode::Multiply, [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]),
    ]
    .into_iter()
    .flat_map(|c| SampleType::ALL.map(|depth| (c, depth)))
    .map(|((b, c, i), depth)| (b, c, i, depth))
    {
        let mut d = Document::new("t", Size::new(40, 40), ColorMode::Rgb, depth);
        d.layers.push(solid_layer("bg", Rect::new(0, 0, 40, 40), [1.0; 4]));
        let mut l = solid_layer("sq", Rect::new(10, 10, 30, 30), colour);
        l.blend = blend;
        l.effects.items = vec![Effect::DropShadow(shadow(0.0, 90.0))];
        d.layers.push(l);
        assert!(close4(px(&d, 20, 20), inside), "{blend:?} {depth:?}: {:?}", px(&d, 20, 20));
        // Not knocked out, the shadow shows through the blend, as in Photoshop with the box unticked.
        if let Effect::DropShadow(s) = &mut d.layers[1].effects.items[0] {
            s.knocks_out = false;
        }
        assert!(!close4(px(&d, 20, 20), inside), "{blend:?} {depth:?} without knockout: {:?}", px(&d, 20, 20));
    }
}

#[test]
fn linked_pattern_overlay_anchors_at_the_effects_reference_point() {
    use photocraft_doc::pattern::Pattern;
    // 2 × 1 tile: red, blue. Anchored at x = 11 (reference point), x = 11 is red, 12 blue.
    let mut tile = photocraft_raster::Surface::new(PixelFormat::RGBA8);
    tile.fill_rect(Rect::new(0, 0, 1, 1), &[1.0, 0.0, 0.0, 1.0]);
    tile.fill_rect(Rect::new(1, 0, 2, 1), &[0.0, 0.0, 1.0, 1.0]);
    let pat = Pattern::new("rb", tile, 2, 1);
    let overlay = Effect::PatternOverlay {
        common: FxCommon::new(photocraft_color::BlendMode::Normal, 1.0),
        name: "rb".into(),
        id: pat.id.clone(),
        scale: 1.0,
        angle: 0.0,
        link: true,
        phase: (0.0, 0.0),
    };
    let mut d = fx_doc(vec![overlay]);
    d.patterns.push(pat);
    // Without a reference point the layer's top-left (10) anchors the tiling.
    assert!(close4(px(&d, 10, 20), [1.0, 0.0, 0.0, 1.0]), "{:?}", px(&d, 10, 20));
    d.layers[1].effects.reference = Some((11.0, 0.0));
    assert!(close4(px(&d, 11, 20), [1.0, 0.0, 0.0, 1.0]), "{:?}", px(&d, 11, 20));
    assert!(close4(px(&d, 12, 20), [0.0, 0.0, 1.0, 1.0]), "{:?}", px(&d, 12, 20));
}

#[test]
fn channel_restrictions_keep_the_backdrop() {
    // Blue left out (Photoshop's Advanced Blending › Channels: R, G only).
    let mut d = doc_white(8, 8);
    let mut l = solid_layer("dark", Rect::new(0, 0, 8, 8), [0.2, 0.3, 0.4, 1.0]);
    l.excluded_channels = 0b100;
    d.layers.push(l);
    assert!(close4(px(&d, 1, 1), [0.2, 0.3, 1.0, 1.0]));
    // Adjustment layers honour it too.
    let mut inv = Layer::new("inv", LayerContent::Adjustment(Adjustment::Invert));
    inv.excluded_channels = 0b001;
    d.layers.push(inv);
    assert!(close4(px(&d, 1, 1), [0.2, 0.7, 0.0, 1.0]));
    // CMYK documents composite in display RGB: no exact equivalent, ignored.
    assert_eq!(channel_weights(&d.layers[1], ColorMode::Cmyk), None);
    assert_eq!(channel_weights(&d.layers[1], ColorMode::Rgb), Some([1.0, 1.0, 0.0]));
}

fn range(black: [u8; 2], white: [u8; 2]) -> photocraft_doc::BlendRange {
    photocraft_doc::BlendRange { black, white }
}

const FULL: photocraft_doc::BlendRange = photocraft_doc::BlendRange::FULL;

#[test]
fn blend_if_this_layer_hides_by_the_layers_own_value() {
    // Mid-grey backdrop; the layer is black on the left, white on the right.
    let mut d = doc_white(8, 8);
    d.layers.push(solid_layer("grey", Rect::new(0, 0, 8, 8), [0.5, 0.5, 0.5, 1.0]));
    let mut l = solid_layer("bw", Rect::new(0, 0, 4, 8), [0.0, 0.0, 0.0, 1.0]);
    l.surface_mut().unwrap().fill_rect(Rect::new(4, 0, 8, 8), &[1.0, 1.0, 1.0, 1.0]);
    // Gray › This Layer: black point at 50 hides the blacks.
    l.blend_if.set(0, [range([50, 50], [255, 255]), FULL]);
    d.layers.push(l);
    assert!(close4(px(&d, 1, 1), [0.5, 0.5, 0.5, 1.0]), "{:?}", px(&d, 1, 1));
    assert!(close4(px(&d, 6, 1), [1.0, 1.0, 1.0, 1.0]));
    // White point at 200 hides the whites as well.
    d.layers[2].blend_if.set(0, [range([50, 50], [200, 200]), FULL]);
    assert!(close4(px(&d, 6, 1), [0.5, 0.5, 0.5, 1.0]));
    // Back to the defaults: everything shows again.
    d.layers[2].blend_if.set(0, [FULL, FULL]);
    assert!(d.layers[2].blend_if.is_default());
    assert!(close4(px(&d, 1, 1), [0.0, 0.0, 0.0, 1.0]));
}

#[test]
fn blend_if_underlying_layer_hides_by_the_backdrop_value() {
    // Backdrop black on the left, white on the right; a red layer over all of it.
    let mut d = doc_white(8, 8);
    d.layers.push(solid_layer("black", Rect::new(0, 0, 4, 8), [0.0, 0.0, 0.0, 1.0]));
    let mut l = solid_layer("red", Rect::new(0, 0, 8, 8), [1.0, 0.0, 0.0, 1.0]);
    // Gray › Underlying Layer: white point at 128 = only over the darks (sky-replacement style).
    l.blend_if.set(0, [FULL, range([0, 0], [128, 128])]);
    d.layers.push(l);
    assert!(close4(px(&d, 1, 1), [1.0, 0.0, 0.0, 1.0]));
    assert!(close4(px(&d, 6, 1), [1.0, 1.0, 1.0, 1.0]));
}

/// An 8-bit backdrop at level `k` counts as `k` for a hard Underlying black point at `k`, also
/// over a semi-transparent backdrop, where compositing leaves the value a few millionths off the
/// level (19.999998 for 20 before #4395); the GPU compositor agrees.
#[test]
fn blend_if_hard_point_includes_its_own_level() {
    for k in 0..=255u8 {
        let v = f32::from(k) / 255.0;
        let mut d = Document::new("t", Size::new(2, 1), ColorMode::Rgb, SampleType::U8);
        d.layers.push(solid_layer("under", Rect::new(0, 0, 2, 1), [v, v, v, 184.0 / 255.0]));
        let mut l = solid_layer("top", Rect::new(0, 0, 2, 1), [1.0, 0.0, 0.0, 1.0]);
        l.blend_if.set(1, [FULL, range([k, k], [255, 255])]);
        d.layers.push(l);
        assert!(close4(px(&d, 0, 0), [1.0, 0.0, 0.0, 1.0]), "level {k} is hidden by a black point at {k}");
    }
}

#[test]
fn blend_if_split_point_fades_like_opacity() {
    // A split black point (Alt-drag) at 50/150 shows a value-100 layer at half strength, which
    // is exactly the layer at 50% opacity, also over a semi-transparent backdrop.
    let v = 100.0 / 255.0;
    let mut d = doc_white(4, 4);
    d.layers[0].surface_mut().unwrap().fill_rect(Rect::new(0, 0, 4, 4), &[0.0, 0.2, 1.0, 0.5]);
    let mut l = solid_layer("v", Rect::new(0, 0, 4, 4), [v, v, v, 1.0]);
    l.blend = BlendMode::Multiply;
    let mut half = l.clone();
    half.opacity = 0.5;
    l.blend_if.set(0, [range([50, 150], [255, 255]), FULL]);
    let mut d2 = d.clone();
    d.layers.push(l);
    d2.layers.push(half);
    let (a, b) = (px(&d, 1, 1), px(&d2, 1, 1));
    assert!(a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4), "{a:?} vs {b:?}");
}

#[test]
fn blend_if_per_channel_ranges() {
    // Blue › This Layer: hide pixels whose blue is above 100.
    let mut d = doc_white(8, 8);
    let mut l = solid_layer("c", Rect::new(0, 0, 4, 8), [1.0, 0.0, 0.0, 1.0]);
    l.surface_mut().unwrap().fill_rect(Rect::new(4, 0, 8, 8), &[0.0, 0.0, 1.0, 1.0]);
    l.blend_if.set(3, [range([0, 0], [100, 100]), FULL]);
    d.layers.push(l);
    assert!(close4(px(&d, 1, 1), [1.0, 0.0, 0.0, 1.0]));
    assert!(close4(px(&d, 6, 1), [1.0, 1.0, 1.0, 1.0]));
    // Red › Underlying Layer: the backdrop's red (255) is above 254 → hidden everywhere.
    d.layers[1].blend_if = Default::default();
    d.layers[1].blend_if.set(1, [FULL, range([0, 0], [254, 254])]);
    assert!(close4(px(&d, 1, 1), [1.0; 4]));
}

#[test]
fn blend_if_on_adjustment_and_clipped_layers() {
    let mut d = doc_white(8, 8);
    d.layers.push(solid_layer("dark", Rect::new(0, 0, 4, 8), [0.1, 0.1, 0.1, 1.0]));
    // Invert, but only where the backdrop is dark (Underlying white point at 128).
    let mut inv = Layer::new("inv", LayerContent::Adjustment(Adjustment::Invert));
    inv.blend_if.set(0, [FULL, range([0, 0], [128, 128])]);
    d.layers.push(inv);
    assert!(close4(px(&d, 1, 1), [0.9, 0.9, 0.9, 1.0]));
    assert!(close4(px(&d, 6, 1), [1.0; 4]));
    // A clipped layer judges "underlying" by its clipping base.
    let mut d = doc_white(8, 8);
    let mut base = solid_layer("base", Rect::new(0, 0, 8, 8), [0.0, 0.0, 0.0, 1.0]);
    base.surface_mut().unwrap().fill_rect(Rect::new(4, 0, 8, 8), &[0.9, 0.9, 0.9, 1.0]);
    let mut clip = solid_layer("clip", Rect::new(0, 0, 8, 8), [0.0, 1.0, 0.0, 1.0]);
    clip.clipped = true;
    clip.blend_if.set(0, [FULL, range([128, 128], [255, 255])]);
    d.layers.push(base);
    d.layers.push(clip);
    assert!(close4(px(&d, 1, 1), [0.0, 0.0, 0.0, 1.0]));
    assert!(close4(px(&d, 6, 1), [0.0, 1.0, 0.0, 1.0]));
}

#[test]
fn blend_if_modes() {
    // Grayscale documents: the single channel's entry (or the gray entry) applies.
    let mut d = Document::with_background("g", Size::new(4, 4), ColorMode::Grayscale, SampleType::U8, Color::WHITE);
    let mut l = Layer::raster("k", d.pixel_format());
    l.surface_mut().unwrap().fill_rect(Rect::new(0, 0, 4, 4), &[0.0, 1.0]);
    l.blend_if.set(1, [range([10, 10], [255, 255]), FULL]);
    d.layers.push(l);
    assert!(close4(px(&d, 1, 1), [1.0; 4]), "{:?}", px(&d, 1, 1));
    // CMYK/Lab composite in display RGB: kept for round trip, not applied.
    assert!(blend_if_active(&d.layers[1], ColorMode::Rgb));
    assert!(blend_if_active(&d.layers[1], ColorMode::Grayscale));
    assert!(!blend_if_active(&d.layers[1], ColorMode::Cmyk));
    assert!(!blend_if_active(&d.layers[1], ColorMode::Lab));
    assert!(!blend_if_active(&d.layers[0], ColorMode::Rgb));
}

#[test]
fn effect_maps_built_inside_parallel_tiles_do_not_deadlock() {
    // A map large enough for the blur to go multi-threaded, built while rayon renders tiles that
    // all wait on the same map (a rayon-parallel blur deadlocked here).
    let mut d = doc_white(900, 700);
    let mut l = solid_layer("fx", Rect::new(100, 100, 800, 600), [0.2, 0.4, 0.9, 1.0]);
    l.effects.items = vec![photocraft_doc::Effect::default_drop_shadow()];
    if let photocraft_doc::Effect::DropShadow(s) = &mut l.effects.items[0] {
        s.size = 30.0;
    }
    d.layers.push(l);
    let out = render(&d, d.bounds());
    assert_eq!(out.px.len(), 900 * 700);
}

#[test]
fn adjustment_results_are_rounded_to_the_document_depth() {
    for (depth, q) in [(SampleType::U8, Some(255.0f32)), (SampleType::U16, Some(32768.0)), (SampleType::F32, None)] {
        let mut d = Document::with_background("q", Size::new(4, 4), ColorMode::Rgb, depth, Color::rgb(0.3, 0.6, 0.9));
        d.layers.push(Layer::new("lv", LayerContent::Adjustment(Adjustment::Exposure { exposure: 0.37, offset: 0.0, gamma: 1.0 })));
        let p = px(&d, 1, 1);
        match q {
            Some(q) => assert!(p.iter().all(|v| ((v * q).round() - v * q).abs() < 1e-3), "{depth:?} {p:?}"),
            None => assert!(p[..3].iter().any(|v| ((v * 255.0).round() - v * 255.0).abs() > 1e-3), "{p:?}"),
        }
    }
}

// Ground truth captured from Adobe Photoshop 2026 (27.10.0): a 0..255 ramp pushed through modern
// (non-legacy) Brightness/Contrast via `executeAction("BrgC", ... useLegacy=false)`. These pin our
// reverse-engineered curves (see crates/compose/src/adjust.rs modern_* and log/devlog.md).
#[test]
fn modern_contrast_matches_photoshop() {
    // (contrast, [out at x = 0,16,32,64,96,128,160,192,224,255])
    let cases: [(f32, [u8; 10]); 4] = [
        (50.0, [0, 11, 23, 52, 87, 128, 169, 204, 233, 255]),
        (-50.0, [0, 21, 41, 76, 105, 128, 151, 180, 215, 255]),
        (100.0, [0, 5, 14, 40, 78, 128, 178, 216, 242, 255]),
        (-25.0, [0, 19, 37, 70, 101, 128, 155, 186, 220, 255]),
    ];
    let xs = [0usize, 16, 32, 64, 96, 128, 160, 192, 224, 255];
    for (c, out) in cases {
        for (i, &x) in xs.iter().enumerate() {
            let got = adjust::modern_contrast(x as f32 / 255.0, c) * 255.0;
            let err = (got - out[i] as f32).abs();
            assert!(err <= 2.0, "contrast {c} x={x}: got {got:.1} want {} (err {err:.1})", out[i]);
        }
    }
}

#[test]
fn modern_brightness_matches_photoshop() {
    let cases: [(f32, [u8; 10]); 2] = [(50.0, [0, 22, 44, 88, 132, 171, 203, 228, 246, 255]), (-50.0, [0, 12, 23, 47, 70, 93, 118, 148, 186, 255])];
    let xs = [0usize, 16, 32, 64, 96, 128, 160, 192, 224, 255];
    for (b, out) in cases {
        for (i, &x) in xs.iter().enumerate() {
            let got = adjust::modern_brightness(x as f32 / 255.0, b) * 255.0;
            let err = (got - out[i] as f32).abs();
            // The brightness roll-off is a spline; our fit is close but approximate at the top.
            assert!(err <= 6.0, "brightness {b} x={x}: got {got:.1} want {} (err {err:.1})", out[i]);
        }
    }
    // Endpoints and the zero slider are exact.
    assert_eq!(adjust::modern_brightness(0.0, 120.0), 0.0);
    assert_eq!(adjust::modern_brightness(1.0, -120.0), 1.0);
    assert_eq!(adjust::modern_brightness(0.37, 0.0), 0.37);
    assert_eq!(adjust::modern_contrast(0.37, 0.0), 0.37);
}

// Levels ground truth from Adobe Photoshop 2026 (adjustLevels on a 0..255 ramp). Verifies our
// `levels()` matches the real app, including the gamma>1 soft shadow toe (initial slope 2^gamma).
#[test]
fn levels_matches_photoshop() {
    let ident = |g: f32| LevelsChannel { in_black: 0.0, in_white: 1.0, gamma: g, out_black: 0.0, out_white: 1.0 };
    // gamma 2.0 (lifts shadows): x -> Photoshop out. The deep shadow is the toe-bounded region.
    let g2: [(usize, u8); 8] = [(1, 4), (8, 30), (16, 55), (32, 90), (64, 128), (128, 181), (192, 221), (224, 239)];
    for (x, ps) in g2 {
        let got = (adjust::levels(&ident(2.0), x as f32 / 255.0) * 255.0).round();
        assert!((got - ps as f32).abs() <= 4.0, "levels gamma 2.0 x={x}: got {got} want {ps}");
    }
    // gamma 0.5 (darkens midtones) is exact: out = (x/255)^2.
    for x in [0usize, 32, 64, 128, 192, 255] {
        let got = (adjust::levels(&ident(0.5), x as f32 / 255.0) * 255.0).round();
        let want = ((x as f32 / 255.0).powi(2) * 255.0).round();
        assert_eq!(got, want, "levels gamma 0.5 x={x}");
    }
    // Endpoints pinned; identity is identity. (The gamma>1 toe's soft-min leaves white within a
    // few parts in 1e5 of 1.0 — invisible at any bit depth; exact at 8-bit.)
    assert_eq!(adjust::levels(&ident(2.5), 0.0), 0.0);
    assert_eq!((adjust::levels(&ident(2.5), 1.0) * 255.0).round(), 255.0);
    assert!((adjust::levels(&ident(2.5), 1.0) - 1.0).abs() < 5e-4);
    assert!((adjust::levels(&ident(1.0), 0.37) - 0.37).abs() < 1e-6);
}

// Photoshop (psd-tools levels_grayscale, 8-bit): input range 44..214, gamma 1.78. The stretch
// is rounded to whole levels before the gamma (45 → 1.5 → 2, 47 → 4.5 → 5), so the shadows
// step unevenly: 45, 46, 47, 48, 52, 69 → 7, 10, 17, 20, 37, 88.
#[test]
fn levels_work_on_whole_levels() {
    let ch = LevelsChannel { in_black: 44.0 / 255.0, in_white: 214.0 / 255.0, gamma: 1.78, out_black: 0.0, out_white: 1.0 };
    let adj = Adjustment::Levels {
        master: LevelsChannel::default(),
        per_channel: [ch.clone(), ch.clone(), ch.clone()],
        space: Default::default(),
        black: LevelsChannel::default(),
    };
    let luts = adjust::tone_luts_q(&adj, Some(255.0));
    let at = |v: u8| {
        let x = f32::from(v) / 255.0 * 4095.0;
        let (i, f) = (x.floor() as usize, x.fract());
        let y = luts[0][i] * (1.0 - f) + luts[0][(i + 1).min(4095)] * f;
        (y * 255.0).round()
    };
    for (v, ps) in [(44u8, 0.0f32), (45, 7.0), (46, 10.0), (47, 17.0), (48, 20.0), (52, 37.0), (69, 88.0), (100, 137.0), (214, 255.0)] {
        assert!((at(v) - ps).abs() <= 1.0, "levels {v}: got {} want {ps}", at(v));
    }
    // 16-bit and float stay smooth (no 8-bit staircase).
    for q in [Some(32768.0), None] {
        let a = adjust::levels_q(&ch, 0.2, q);
        let b = adjust::levels_q(&ch, 0.2 + 1.0 / 2048.0, q);
        assert!(b > a && b - a < 2.0 / 255.0, "{q:?}: {a} {b}");
    }
    // Hostile quantum values fall back to the continuous curve.
    for q in [Some(0.0), Some(f32::NAN), Some(-3.0)] {
        assert_eq!(adjust::levels_q(&ch, 0.4, q), adjust::levels(&ch, 0.4));
    }
}

// The oracle corpus' rgb32 levels.psd: input 15..230, gamma 1.3, output 10..245. In a
// 32-bit document neither range clips and the gamma is a plain power curve mirrored below black:
// 0.0497 → 3 (the clipped curve gives 10), 0.9473 → 255 (245). Integer documents still clip.
#[test]
fn levels_dont_clip_in_32_bit() {
    let ch = LevelsChannel { in_black: 15.0 / 255.0, in_white: 230.0 / 255.0, gamma: 1.3, out_black: 10.0 / 255.0, out_white: 245.0 / 255.0 };
    let adj = Adjustment::Levels { master: ch, per_channel: Default::default(), space: Default::default(), black: LevelsChannel::default() };
    let level = |depth, v| {
        let mut b = Buffer::filled(Rect::new(0, 0, 1, 1), [v, v, v, 1.0]);
        adjust::apply_depth(&adj, &mut b, adjust::Transfer::Srgb, Some(depth));
        (b.px[0][0] * 255.0).round()
    };
    for (v, want) in [(0.0497, 3.0), (0.0976, 32.0), (0.4508, 140.0), (0.9473, 255.0)] {
        assert_eq!(level(SampleType::F32, v), want, "{v}");
    }
    assert!(level(SampleType::F32, 0.0) < 0.0);
    assert!(level(SampleType::F32, 1.0) > 255.0);
    assert_eq!(level(SampleType::U8, 0.0), 10.0);
    assert_eq!(level(SampleType::U8, 1.0), 245.0);
}

// Exposure linearises RGB documents through a 2.2 power, not the sRGB curve: Photoshop lifts 76
// to 134 with an offset of 0.1738 (psd-tools adjustment_nested_composition_4).
#[test]
fn exposure_offset_uses_gamma_2_2_in_rgb() {
    let mut b = Buffer::filled(Rect::new(0, 0, 1, 1), [76.0 / 255.0, 76.0 / 255.0, 76.0 / 255.0, 1.0]);
    adjust::apply_with(&Adjustment::Exposure { exposure: 0.0, offset: 0.1738, gamma: 1.0 }, &mut b, adjust::Transfer::Srgb);
    assert_eq!((b.px[0][0] * 255.0).round(), 134.0);
    assert_eq!(adjust::Transfer::Gamma(1.732).for_exposure(), adjust::Transfer::Gamma(1.732));
}

// 32-bit samples are linear light: +1 stop doubles them as stored, where an 8-bit document goes
// through its tone curve first (photoshop corpus rgb32 and gray32 exposure.psd).
#[test]
fn exposure_scales_32_bit_samples_as_stored() {
    let plus_one_stop = Adjustment::Exposure { exposure: 1.0, offset: 0.0, gamma: 1.0 };
    for (depth, fmt, want) in [(SampleType::F32, PixelFormat::RGBA32F, 0.4), (SampleType::U8, PixelFormat::RGBA8, 0.2744)] {
        let mut d = Document::new("e", Size::new(1, 1), ColorMode::Rgb, depth);
        let mut l = Layer::raster("px", fmt);
        l.surface_mut().unwrap().fill_rect(Rect::new(0, 0, 1, 1), &[0.2, 0.2, 0.2, 1.0]);
        d.layers = vec![l, Layer::new("exp", LayerContent::Adjustment(plus_one_stop.clone()))];
        let got = px(&d, 0, 0)[0];
        assert!((got - want).abs() < 2e-3, "{depth:?}: {got}");
    }
    for mode in [ColorMode::Rgb, ColorMode::Grayscale] {
        assert_eq!(adjust::Transfer::for_document(mode, SampleType::F32), adjust::Transfer::Gamma(1.0));
    }
}

#[test]
fn float_adjustments_preserve_hdr_range() {
    let identity = vec![photocraft_doc::adjust::CurvePoint { input: 0.0, output: 0.0 }, photocraft_doc::adjust::CurvePoint { input: 1.0, output: 1.0 }];
    let levels = LevelsChannel { in_black: 0.25, in_white: 0.75, gamma: 2.0, out_black: -0.1, out_white: 1.5 };
    let adjustments = [
        (Adjustment::Exposure { exposure: 0.0, offset: 0.0, gamma: 1.0 }, [4.0, 2.0, 0.5]),
        (Adjustment::Exposure { exposure: 1.0, offset: 0.0, gamma: 1.0 }, [8.0, 4.0, 1.0]),
        (
            Adjustment::Levels { master: Default::default(), per_channel: Default::default(), space: Default::default(), black: Default::default() },
            [4.0, 2.0, 0.5],
        ),
        (
            Adjustment::Levels { master: levels, per_channel: Default::default(), space: Default::default(), black: Default::default() },
            [4.0, 2.0, 0.5].map(|v| -0.1 + ((v - 0.25_f32) / 0.5).sqrt() * 1.6),
        ),
        (
            Adjustment::Curves {
                master: identity.clone(),
                per_channel: std::array::from_fn(|_| identity.clone()),
                space: Default::default(),
                black: Vec::new(),
                pencil: false,
            },
            [4.0, 2.0, 0.5],
        ),
        (
            Adjustment::Curves { master: Vec::new(), per_channel: Default::default(), space: Default::default(), black: Vec::new(), pencil: false },
            [4.0, 2.0, 0.5],
        ),
    ];
    for (adj, want) in adjustments {
        let mut d = Document::new("HDR", Size::new(1, 1), ColorMode::Rgb, SampleType::F32);
        let mut l = Layer::raster("pixels", PixelFormat::RGBA32F);
        l.surface_mut().unwrap().fill_rect(Rect::new(0, 0, 1, 1), &[4.0, 2.0, 0.5, 0.75]);
        d.layers = vec![l, Layer::new("adjustment", LayerContent::Adjustment(adj.clone()))];
        let got = px(&d, 0, 0);
        for (a, b) in got[..3].iter().zip(want) {
            assert!((a - b).abs() < 1e-5, "{adj:?}: {got:?} vs {want:?}");
        }
        assert_eq!(got[3], 0.75);
    }
}

#[test]
fn float_tone_adjustments_preserve_negative_and_untouched_channels() {
    let identity = vec![photocraft_doc::adjust::CurvePoint { input: 0.0, output: 0.0 }, photocraft_doc::adjust::CurvePoint { input: 1.0, output: 1.0 }];
    let edited = vec![photocraft_doc::adjust::CurvePoint { input: 0.0, output: 0.0 }, photocraft_doc::adjust::CurvePoint { input: 1.0, output: 0.5 }];
    for (adj, want) in [
        (
            Adjustment::Levels { master: Default::default(), per_channel: Default::default(), space: Default::default(), black: Default::default() },
            [-0.1, 4.0, 0.5],
        ),
        (
            Adjustment::Curves {
                master: identity.clone(),
                per_channel: [identity.clone(), Vec::new(), edited],
                space: Default::default(),
                black: Vec::new(),
                pencil: false,
            },
            [-0.1, 4.0, 0.25],
        ),
    ] {
        let mut b = Buffer::filled(Rect::new(0, 0, 1, 1), [-0.1, 4.0, 0.5, 0.75]);
        adjust::apply_depth(&adj, &mut b, adjust::Transfer::Gamma(1.0), Some(SampleType::F32));
        for (a, b) in b.px[0][..3].iter().zip(want) {
            assert!((a - b).abs() < 1e-5, "{adj:?}: {:?} vs {want:?}", b);
        }
        assert_eq!(b.px[0][3], 0.75);
    }
}

#[test]
fn float_exposure_stops_round_trip_without_losing_highlights() {
    let input = [4.0, -2.0, 0.5, 0.75];
    let mut b = Buffer::filled(Rect::new(0, 0, 1, 1), input);
    for exposure in [1.0, -1.0] {
        adjust::apply_depth(&Adjustment::Exposure { exposure, offset: 0.0, gamma: 1.0 }, &mut b, adjust::Transfer::Gamma(1.0), Some(SampleType::F32));
    }
    assert_eq!(b.px[0], input);
}

#[test]
fn float_exposure_preserves_signed_gamma_and_offset() {
    let mut b = Buffer::filled(Rect::new(0, 0, 1, 1), [-0.5, 0.0, 4.0, 0.75]);
    adjust::apply_depth(&Adjustment::Exposure { exposure: 1.0, offset: -0.25, gamma: 2.0 }, &mut b, adjust::Transfer::Gamma(1.0), Some(SampleType::F32));
    for (got, want) in b.px[0].iter().zip([-1.25_f32.sqrt(), -0.5, 7.75_f32.sqrt(), 0.75]) {
        assert!((got - want).abs() < 1e-6, "{:?}", b.px[0]);
    }
}

#[test]
fn float_levels_apply_channel_before_master_outside_unit_range() {
    let channel = LevelsChannel { in_black: 0.5, in_white: 1.5, ..Default::default() };
    let master = LevelsChannel { out_black: -0.1, out_white: 1.5, ..Default::default() };
    let adj = Adjustment::Levels { master, per_channel: std::array::from_fn(|_| channel.clone()), space: Default::default(), black: Default::default() };
    let mut b = Buffer::filled(Rect::new(0, 0, 1, 1), [4.0, 2.0, -0.1, 1.0]);
    adjust::apply_depth(&adj, &mut b, adjust::Transfer::Gamma(1.0), Some(SampleType::F32));
    for (got, want) in b.px[0].iter().zip([5.5, 2.3, -1.06, 1.0]) {
        assert!((got - want).abs() < 1e-6, "{:?}", b.px[0]);
    }
}

// A 30° Reflected gradient fill on a 4 × 4 canvas renders as Photoshop's (its end point snaps to
// the corner: t = |x − y| / 4) at every depth.
#[test]
fn small_gradient_fill_matches_photoshop_at_all_depths() {
    let stops = vec![(0.0, Color::rgb(0.0, 0.0, 0.0)), (1.0, Color::rgb(1.0, 1.0, 1.0))];
    for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
        let mut d = Document::new("g", Size::new(4, 4), ColorMode::Rgb, depth);
        let fill = Fill::gradient(stops.clone(), 30.0, 1.0, photocraft_doc::GradientStyle::Reflected, false);
        d.layers.push(Layer::new("g", LayerContent::Fill(fill)));
        let out = flatten(&d);
        for y in 0..4usize {
            for x in 0..4usize {
                let want = (x as f32 - y as f32).abs() / 4.0;
                let got = out.px[y * 4 + x][0];
                assert!((got - want).abs() < 1.0 / 255.0, "{depth:?} ({x},{y}): {got} vs {want}");
            }
        }
    }
}

/// A shape layer's gradient fill must render the same pixels as the compositor's gradient
/// fill layers, for every non-default gradient field (midpoints, opacity stops, centre offset,
/// dither, unsorted stops) and all five styles, in both frames: "Align with layer" on lays the
/// gradient out over the shape's bounds, off over the canvas. `photocraft-vector` cannot
/// depend on this crate, so it carries a copy of this ramp and geometry — this test is what
/// keeps the two in step (only interior pixels, where the shape's coverage is exactly 1).
#[test]
fn shape_layer_gradients_match_fill_layers() {
    use photocraft_doc::GradientStyle;
    let canvas = Rect::new(0, 0, 64, 48);
    // rect(x, y, w, h): the shape spans (16, 8)–(64, 48), so its whole-pixel bounds sit strictly
    // inside the canvas — "Align with layer" then picks a different frame than the canvas.
    let path = photocraft_vector::shapes::rect(16.0, 8.0, 48.0, 40.0);
    let layer_frame = Rect::new(16, 8, 64, 48);
    let grad = |style: GradientStyle, midpoints: Vec<f32>, opacity_stops: Vec<(f32, f32)>, offset: (f32, f32), dither: bool, unsorted: bool, align: bool| {
        let mut stops = vec![(0.0, Color::BLACK), (0.5, Color::rgb(1.0, 0.25, 0.5)), (1.0, Color::WHITE)];
        if unsorted {
            stops.reverse();
        }
        Fill::Gradient { stops, angle: 30.0, scale: 1.7, style, reverse: false, opacity_stops, midpoints, offset, dither, align }
    };
    let styles = [GradientStyle::Linear, GradientStyle::Radial, GradientStyle::Angle, GradientStyle::Reflected, GradientStyle::Diamond];
    // (name, midpoints, opacity stops, centre offset, dither, unsorted stops) per case.
    type Case = (&'static str, Vec<f32>, Vec<(f32, f32)>, (f32, f32), bool, bool);
    let cases: Vec<Case> = vec![
        ("plain", vec![], vec![], (0.0, 0.0), false, false),
        ("midpoints", vec![0.25, 0.9], vec![], (0.0, 0.0), false, false),
        ("opacity stops", vec![], vec![(0.0, 1.0), (0.5, 0.2), (1.0, 0.9)], (0.0, 0.0), false, false),
        ("offset", vec![], vec![], (0.3, -0.2), false, false),
        ("dither", vec![], vec![], (0.0, 0.0), true, false),
        ("all of them", vec![0.75], vec![(0.0, 1.0), (1.0, 0.4)], (0.1, 0.1), true, true),
    ];
    for style in styles {
        for (name, mids, opac, offset, dither, unsorted) in &cases {
            for align in [true, false] {
                let f = grad(style, mids.clone(), opac.clone(), *offset, *dither, *unsorted, align);
                let frame = if align { layer_frame } else { canvas };
                let want = gradient_fill::render(&f, canvas, frame);
                let sh = photocraft_doc::vector::ShapeLayer { path: path.clone(), fill: Some(f), ..Default::default() };
                let s = photocraft_vector::render_shape(&sh, PixelFormat::RGBA8, canvas);
                let mut worst = 0.0f32;
                for y in 9..47 {
                    for x in 17..63 {
                        let p = want[(y as usize) * canvas.width() as usize + x as usize];
                        for (got, want_ch) in (0..4usize).map(|ch| (s.sample_channel(x, y, ch), p[ch])) {
                            worst = worst.max((got - want_ch).abs());
                        }
                    }
                }
                assert!(worst <= 1.5 / 255.0, "{style:?} {name} align={align}: worst delta {worst}");
            }
        }
    }
}

/// A tall document with soft content, a translucent region and an adjustment.
fn tall_doc(w: u32, h: u32) -> Document {
    let mut d = doc_white(w, h);
    let mut l = Layer::raster("grad", PixelFormat::RGBA8);
    let s = l.surface_mut().unwrap();
    for y in 0..h as i32 {
        s.fill_rect(Rect::new(0, y, w as i32 / 2, y + 1), &[y as f32 / h as f32, 0.3, 1.0 - y as f32 / h as f32, 1.0]);
    }
    s.fill_rect(Rect::new(w as i32 / 2, 0, w as i32, h as i32 / 3), &[0.9, 0.1, 0.1, 0.4]);
    d.layers.push(l);
    d.layers.push(Layer::new("inv", LayerContent::Adjustment(Adjustment::Invert)));
    d
}

#[test]
fn render_tiled_in_place_matches_one_tile_with_many_bands() {
    // Enough bands for the band-parallel path, and few for the tile-parallel one.
    let d = tall_doc(37, 900);
    let whole = render_tiled(&d, d.bounds(), 10_000);
    for tile in [5, 64, 300] {
        assert_eq!(render_tiled(&d, d.bounds(), tile), whole, "tile {tile}");
    }
    let r = Rect::new(3, 17, 30, 811);
    assert_eq!(render_tiled(&d, r, 4).px, render_tiled(&d, r, 10_000).px);
    assert!(render_tiled(&d, Rect::new(5, 5, 5, 900), 4).px.is_empty());
}

#[test]
fn render_bands_cover_the_rect_in_order() {
    let d = tall_doc(20, 700);
    let r = Rect::new(2, 10, 18, 690);
    let mut rows = Vec::new();
    let mut px = Vec::new();
    render_bands(&d, r, 100, |b| -> Result<(), ()> {
        assert_eq!((b.rect.x0, b.rect.x1), (r.x0, r.x1));
        rows.push((b.rect.y0, b.rect.y1));
        px.extend(b.px);
        Ok(())
    })
    .unwrap();
    // 100 rows round up to one 256-row tile band.
    assert_eq!(rows, vec![(10, 266), (266, 522), (522, 690)]);
    assert_eq!(px, render(&d, r).px);
    // The sink's error stops the render.
    let mut n = 0;
    assert_eq!(
        render_bands(&d, r, 1, |_| {
            n += 1;
            Err("stop")
        }),
        Err("stop")
    );
    assert_eq!(n, 1);
    assert_eq!(band_rows_for(14_000, 0) % RENDER_TILE, 0);
    assert!(band_rows_for(1, 0) > 0 && band_rows_for(0, -5) > 0);
}

/// The previous thumbnail: flatten, then area-average (premultiplied).
fn reference_reduce(doc: &Document, w: usize, h: usize) -> Vec<[f32; 4]> {
    let full = flatten(doc);
    let (fw, fh) = (full.rect.width() as usize, full.rect.height() as usize);
    let mut out = Vec::new();
    for ty in 0..h {
        let (y0, y1) = (ty * fh / h, ((ty + 1) * fh / h).max(ty * fh / h + 1).min(fh));
        for tx in 0..w {
            let (x0, x1) = (tx * fw / w, ((tx + 1) * fw / w).max(tx * fw / w + 1).min(fw));
            let mut acc = [0.0f32; 4];
            for y in y0..y1 {
                for x in x0..x1 {
                    let p = full.px[y * fw + x];
                    for c in 0..3 {
                        acc[c] += p[c] * p[3];
                    }
                    acc[3] += p[3];
                }
            }
            let n = ((y1 - y0) * (x1 - x0)).max(1) as f32;
            out.push(if acc[3] > 0.0 { [acc[0] / acc[3], acc[1] / acc[3], acc[2] / acc[3], acc[3] / n] } else { [0.0; 4] });
        }
    }
    out
}

#[test]
fn reduced_render_matches_the_full_composite_averaged() {
    let d = tall_doc(53, 1000);
    for (w, h, band) in [(10, 190, 0), (7, 33, 256), (53, 999, 512), (1, 1, 256)] {
        let got = render_reduced_in_bands(&d, d.bounds(), w, h, None, band);
        assert_eq!((got.rect.width(), got.rect.height()), (w, h));
        assert_eq!(got.px, reference_reduce(&d, w as usize, h as usize), "{w}x{h} band {band}");
    }
    // Full size is the composite itself; an empty document reduces to transparency.
    assert_eq!(render_reduced(&d, 53, 1000).px, flatten(&d).px);
    let empty = Document::new("e", Size::new(0, 0), ColorMode::Rgb, SampleType::U8);
    assert_eq!(render_reduced(&empty, 4, 4).px, vec![[0.0; 4]]);
}

#[test]
fn reduced_damage_matches_the_whole_reduction() {
    // A reduced canvas texture updates only what a stroke touched: those pixels must equal the
    // whole reduction's, for uneven factors, damage on span edges and bands smaller than the area.
    let d = tall_doc(53, 1000);
    for (w, h, band) in [(10, 190, 0), (7, 33, 256), (53, 999, 64), (53, 1000, 0), (1, 1, 256)] {
        let all = render_reduced_in_bands(&d, d.bounds(), w, h, None, band);
        for dmg in [Rect::new(0, 0, 1, 1), Rect::new(5, 17, 6, 18), Rect::new(12, 300, 40, 701), Rect::new(-9, 990, 80, 2000), Rect::new(0, 0, 53, 1000)] {
            let part = render_reduced_in_bands(&d, d.bounds(), w, h, Some(dmg), band);
            let r = part.rect;
            assert!(!r.is_empty() && r.x0 >= 0 && r.y0 >= 0 && r.x1 as u32 <= w && r.y1 as u32 <= h, "{w}x{h} {dmg:?} -> {r:?}");
            for y in r.y0..r.y1 {
                for x in r.x0..r.x1 {
                    assert_eq!(part.get(x, y), all.get(x, y), "{w}x{h} band {band} {dmg:?} at {x},{y}");
                }
            }
            // Exactly the output pixels whose source area meets the damage.
            let span = |t: u32, f: u32, n: u32| (t * f / n, ((t + 1) * f / n).max(t * f / n + 1).min(f));
            let d = dmg.intersect(&Rect::new(0, 0, 53, 1000));
            for y in 0..h {
                for x in 0..w {
                    let ((x0, x1), (y0, y1)) = (span(x, 53, w), span(y, 1000, h));
                    let meets = !Rect::new(x0 as i32, y0 as i32, x1 as i32, y1 as i32).intersect(&d).is_empty();
                    assert_eq!(r.contains(x as i32, y as i32), meets, "{w}x{h} {dmg:?} at {x},{y}");
                }
            }
        }
        assert!(render_reduced_in_bands(&d, d.bounds(), w, h, Some(Rect::new(60, 0, 70, 10)), band).rect.is_empty());
    }
}

#[test]
fn reduced_render_of_an_area_past_the_canvas() {
    // The Crop tool shows layer pixels beyond the canvas: rendered in place, they must equal the
    // same pixels on a canvas that holds them (adjustment layers apply out there too).
    let paint = |d: &mut Document, dx: i32, dy: i32| {
        let mut l = Layer::raster("big", PixelFormat::RGBA8);
        let s = l.surface_mut().unwrap();
        for y in -10..40 {
            for x in -20..60 {
                let a = if x < 0 || y >= 30 { 0.6 } else { 1.0 };
                s.fill_rect(Rect::new(x + dx, y + dy, x + dx + 1, y + dy + 1), &[(x + 20) as f32 / 80.0, (y + 10) as f32 / 50.0, 0.5, a]);
            }
        }
        d.layers.push(l);
        d.layers.push(Layer::new("inv", LayerContent::Adjustment(Adjustment::Invert)));
    };
    let mut d = Document::new("d", Size::new(40, 30), ColorMode::Rgb, SampleType::U8);
    paint(&mut d, 0, 0);
    let mut whole = Document::new("w", Size::new(80, 50), ColorMode::Rgb, SampleType::U8);
    paint(&mut whole, 20, 10);
    let area = Rect::new(-20, -10, 60, 40);
    for (w, h) in [(80, 50), (8, 5), (27, 13), (1, 1)] {
        let got = render_reduced_rect(&d, area, w, h);
        assert_eq!(got.rect, Rect::new(0, 0, w as i32, h as i32));
        assert_eq!(got.px, render_reduced(&whole, w, h).px, "{w}x{h}");
    }
    // Past every layer there is nothing to show.
    assert!(render_reduced_rect(&d, Rect::new(100, 100, 120, 110), 20, 10).px.iter().all(|p| p[3] == 0.0));
}

#[test]
fn large_documents_thumbnail_from_a_proxy() {
    // 4200 × 4200 (17.6 MP) is above PROXY_THUMBNAIL_PIXELS: the proxy path must still give a
    // thumbnail of the right size whose colours match the exact reduction.
    let mut d = doc_white(4200, 4200);
    d.layers.push(solid_layer("red", Rect::new(0, 0, 2100, 4200), [1.0, 0.0, 0.0, 1.0]));
    let t = thumbnail(&d, 100);
    assert_eq!((t.width, t.height), (100, 100));
    let at = |x: usize, y: usize| t.pixels[(y * 100 + x) * 4..(y * 100 + x) * 4 + 4].to_vec();
    assert_eq!(at(10, 50), vec![255, 0, 0, 255]);
    assert_eq!(at(90, 50), vec![255, 255, 255, 255]);
    // Effects don't scale with a proxy: such documents reduce the exact composite.
    assert!(proxy::proxy_faithful(&d));
    d.layers[1].effects.items.push(photocraft_doc::Effect::default_drop_shadow());
    assert!(!proxy::proxy_faithful(&d));
}

/// Layers whose effect maps are built with nested Rayon work (a gradient fill's rows render in
/// parallel), each carrying a different effect.
#[cfg(not(target_arch = "wasm32"))]
fn nested_parallel_fx_doc() -> Document {
    use photocraft_doc::{Bevel, BevelStyle, BevelTechnique, Contour, Effect, FxCommon, FxPaint, Glow, GlowSource, GlowTechnique, StrokeFx, StrokePosition};
    let mut d = doc_white(512, 512);
    let g = |a: f32| Fill::gradient(vec![(0.0, Color::BLACK), (1.0, Color::rgb(0.2, 0.5, 0.9))], a, 1.0, photocraft_doc::GradientStyle::Linear, false);
    let mut shadow = Layer::new("shadow", LayerContent::Fill(g(0.0)));
    shadow.effects.items = vec![Effect::default_drop_shadow()];
    let mut stroke = Layer::new("stroke", LayerContent::Fill(g(45.0)));
    stroke.opacity = 0.6;
    stroke.effects.items = vec![Effect::Stroke(StrokeFx {
        common: FxCommon::new(BlendMode::Normal, 1.0),
        size: 3.0,
        position: StrokePosition::Inside,
        paint: FxPaint::Color(Color::rgb(1.0, 0.0, 0.0)),
    })];
    let mut glow = Layer::new("glow", LayerContent::Fill(g(90.0)));
    glow.opacity = 0.5;
    glow.effects.items = vec![Effect::InnerGlow(Glow {
        common: FxCommon::new(BlendMode::Screen, 0.75),
        paint: FxPaint::Color(Color::rgb(1.0, 1.0, 0.7)),
        technique: GlowTechnique::Softer,
        spread: 0.0,
        size: 9.0,
        contour: Contour::Linear,
        anti_alias: false,
        range: 0.5,
        jitter: 0.0,
        noise: 0.0,
        source: GlowSource::Edge,
    })];
    // A chiselled bevel: its distance transforms run on threads of their own, inside the tile that builds them.
    let mut chisel = solid_layer("chisel", Rect::new(60, 60, 450, 450), [0.8, 0.6, 0.2, 1.0]);
    chisel.effects.items = vec![Effect::BevelEmboss(Bevel {
        enabled: true,
        style: BevelStyle::InnerBevel,
        technique: BevelTechnique::ChiselHard,
        depth: 1.5,
        up: true,
        size: 9.0,
        soften: 0.0,
        angle: 120.0,
        altitude: 32.0,
        use_global_light: false,
        gloss_contour: Contour::Linear,
        highlight: FxCommon::new(BlendMode::Screen, 0.75),
        highlight_color: Color::WHITE,
        shadow: FxCommon::new(BlendMode::Multiply, 0.75),
        shadow_color: Color::BLACK,
        contour: None,
        texture: None,
    })];
    // Unaffected layers whose rows also render in parallel keep both workers stealing.
    let base = Layer::new("base", LayerContent::Fill(g(30.0)));
    let mut top = Layer::new("top", LayerContent::Fill(g(60.0)));
    top.opacity = 0.3;
    d.layers.extend([base, shadow, stroke, glow, chisel, top]);
    d
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn effect_maps_with_nested_rayon_work_do_not_deadlock() {
    // #276: a tile building a layer's effect maps inside a once-init ran Rayon work; waiting on
    // it, its worker stole another tile of the same layer, which then blocked on the init held
    // further up its own stack. Eight threads (two rarely steal across that wait) and 64 tiles make that stealing likely; the cache is
    // purged every round so the maps are rebuilt. A watchdog turns a hang into a failure.
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let Ok(pool) = rayon::ThreadPoolBuilder::new().num_threads(8).build() else { return };
        let d = nested_parallel_fx_doc();
        let first = pool.install(|| render_tiled(&d, d.bounds(), 64));
        for _ in 0..100 {
            purge_effect_cache();
            let out = pool.install(|| render_tiled(&d, d.bounds(), 64));
            assert!(out.px == first.px, "renders differ between rounds");
        }
        let _ = tx.send(());
    });
    match rx.recv_timeout(std::time::Duration::from_secs(120)) {
        Ok(()) => {}
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => panic!("effect-map rendering deadlocked (#276)"),
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => panic!("render thread panicked"),
    }
}

// Photo Filter multiplies D50 XYZ (relative to white) by the filter colour's, then restores the
// encoded luminosity like the Luminosity blend mode; 32-bit documents keep luminance Y instead.
// Samples from the photoshop corpus photo-filter.psd (Warming Filter (85), density 40 %, Preserve
// Luminosity): pure blue gains red, and white stays white (#442).
#[test]
fn photo_filter_matches_photoshop() {
    let filter = |color: [f32; 3], depth, v: [f32; 3]| {
        let adj = Adjustment::PhotoFilter { color, density: 0.4, preserve_luminosity: true };
        let mut b = Buffer::filled(Rect::new(0, 0, 1, 1), [v[0], v[1], v[2], 1.0]);
        adjust::apply_depth(&adj, &mut b, adjust::Transfer::for_document(ColorMode::Rgb, depth), Some(depth));
        [b.px[0][0], b.px[0][1], b.px[0][2]].map(|c| c * 255.0)
    };
    let warming = [236.0 / 255.0, 138.0 / 255.0, 0.0];
    for depth in [SampleType::U8, SampleType::U16] {
        for (v, ps) in [
            ([1.0, 0.0, 0.0], [239.0, 7.0, 7.0]),
            ([0.0, 1.0, 0.0], [69.0, 220.0, 0.0]),
            ([0.0, 0.0, 1.0], [46.0, 0.0, 130.0]),
            ([1.0, 1.0, 0.0], [255.0, 240.0, 80.0]),
            ([0.5019, 0.5019, 0.5019], [140.0, 124.0, 116.0]),
            ([0.949, 0.949, 0.949], [255.0, 238.0, 229.0]),
            ([1.0, 1.0, 1.0], [255.0, 255.0, 255.0]),
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ] {
            let got = filter(warming, depth, v);
            assert!(got.iter().zip(ps).all(|(g, p)| (g - p).abs() <= 1.0), "{depth:?} {v:?}: got {got:?} want {ps:?}");
        }
    }
    // The 32-bit file picks the same numbers as linear values (Lab 81.5, 9.41, 82.19). Its fit is
    // looser (mean 0.9/255 over the file, 6.9/255 with a plain RGB multiply).
    let warming32 = [0.966_445_6, 0.761_806_3, 0.0];
    for (v, ps) in [([0.2159, 0.2159, 0.2159], [66.0, 53.0, 40.0]), ([1.0, 1.0, 0.0], [255.0, 251.0, 0.0]), ([1.0, 0.0, 0.0], [255.0, 0.0, 0.0])] {
        let got = filter(warming32, SampleType::F32, v);
        assert!(got.iter().zip(ps).all(|(g, p)| (g - p).abs() <= 2.5), "F32 {v:?}: got {got:?} want {ps:?}");
    }
}

#[test]
fn rendering_restores_the_lab_mix_flag_it_found() {
    // #1112: a render set LAB_MIX for its document, then cleared it. A nested rayon job (a
    // gradient fill, say) can let a thread render another tile in the middle of an outer Lab
    // tile, and clearing the flag made the rest of that outer tile mix in sRGB.
    for mode in [ColorMode::Lab, ColorMode::Rgb] {
        let doc = Document::with_background("t", Size::new(8, 8), mode, SampleType::U8, Color::WHITE);
        // Single-tile and multi-tile paths (the latter also prepares effect maps under the flag).
        for tile in [RENDER_TILE, 4] {
            for outer in [true, false] {
                psblend::LAB_MIX.with(|l| l.set(outer));
                let _ = render_tiled(&doc, doc.bounds(), tile);
                let after = psblend::LAB_MIX.with(|l| l.get());
                psblend::LAB_MIX.with(|l| l.set(false));
                assert_eq!(after, outer, "{mode:?} with tile {tile} inside a tile that had LAB_MIX = {outer}");
            }
        }
    }
}

/// Vibrance of an 8-bit sRGB colour (0–255 in and out).
fn vibrance_255(c: [f32; 3], vibrance: f32, saturation: f32) -> [f32; 3] {
    let mut buf = Buffer::filled(Rect::new(0, 0, 1, 1), [c[0] / 255.0, c[1] / 255.0, c[2] / 255.0, 1.0]);
    adjust::apply(&Adjustment::Vibrance { vibrance, saturation }, &mut buf);
    let p = buf.px[0];
    [p[0] * 255.0, p[1] * 255.0, p[2] * 255.0]
}

fn assert_near_255(got: [f32; 3], want: [f32; 3], tol: f32, what: &str) {
    assert!(got.iter().zip(want).all(|(g, w)| (g - w).abs() <= tol), "{what}: got {got:?}, Photoshop {want:?}");
}

// Photoshop 25.4 references (16-bit runs of 8-bit colours in an sRGB document).

#[test]
fn vibrance_saturation_matches_photoshop() {
    // −100 greys to 0.288 R + 0.712 G in linear light: blue has no weight, yellow stays bright.
    for (c, g) in [([251.0, 201.0, 0.0], 217.0), ([255.0, 0.0, 0.0], 146.0), ([0.0, 255.0, 0.0], 219.4), ([0.0, 0.0, 255.0], 0.0), ([255.0, 255.0, 0.0], 255.0)]
    {
        assert_near_255(vibrance_255(c, 0.0, -100.0), [g; 3], 0.5, "saturation -100");
    }
    for (c, s, want) in [
        ([255.0, 128.0, 0.0], -50.0, [220.7, 155.2, 129.4]),
        ([224.0, 176.0, 144.0], -50.0, [208.6, 183.9, 169.9]),
        ([64.0, 128.0, 192.0], 50.0, [0.0, 134.3, 218.7]),
        ([224.0, 176.0, 144.0], 100.0, [251.3, 158.5, 51.7]),
        ([32.0, 96.0, 64.0], 100.0, [0.0, 106.8, 32.8]),
    ] {
        assert_near_255(vibrance_255(c, 0.0, s), want, 0.5, "saturation");
    }
}

#[test]
fn negative_vibrance_matches_photoshop() {
    for (c, v, want) in [
        ([255.0, 0.0, 0.0], -100.0, [255.0, 137.0, 137.0]),
        ([128.0, 0.0, 0.0], -100.0, [128.0, 65.7, 65.7]),
        ([224.0, 176.0, 144.0], -100.0, [213.2, 197.4, 188.8]),
        ([64.0, 128.0, 192.0], -100.0, [124.4, 149.0, 185.0]),
        ([255.0, 128.0, 0.0], -50.0, [255.0, 152.0, 99.1]),
        ([251.0, 201.0, 0.0], -50.0, [251.0, 208.1, 97.4]),
        ([32.0, 96.0, 64.0], -50.0, [50.1, 91.7, 68.6]),
    ] {
        assert_near_255(vibrance_255(c, v, 0.0), want, 0.6, "vibrance");
    }
    // Vibrance applies first, then Saturation.
    assert_near_255(vibrance_255([64.0, 128.0, 192.0], 50.0, -50.0), [86.5, 117.8, 158.4], 3.0, "vibrance then saturation");
}

#[test]
fn positive_vibrance_follows_photoshop_closely() {
    // A fit, not exact: within a few levels, saturated colours untouched, skin damped.
    for (c, v, want, tol) in [
        ([255.0, 128.0, 0.0], 50.0, [255.0, 128.0, 0.0], 0.5),
        ([144.0, 160.0, 176.0], 100.0, [121.2, 152.6, 180.5], 1.5),
        ([64.0, 128.0, 192.0], 50.0, [50.6, 125.1, 192.8], 4.0),
        ([224.0, 176.0, 144.0], 50.0, [224.9, 173.4, 138.0], 3.0),
        ([224.0, 176.0, 144.0], 100.0, [226.6, 166.8, 122.4], 5.0),
    ] {
        assert_near_255(vibrance_255(c, v, 0.0), want, tol, "vibrance");
    }
}

#[test]
fn clipped_brightness_and_desaturation_whiten_a_lighter_color_logo() {
    // постер.psd: a yellow logo in Lighter Color over the same yellow, with Brightness/Contrast +150
    // and Vibrance › Saturation −100 clipped to it. Photoshop shows it white (254); 0.5.0 left it
    // yellow (its Saturation −100 greyed (255, 255, 0) to 128, darker than the yellow beneath).
    let yellow = [251.0 / 255.0, 201.0 / 255.0, 0.0, 1.0];
    let mut d = doc_white(2, 1);
    d.layers[0].surface_mut().unwrap().fill_rect(Rect::new(0, 0, 2, 1), &yellow);
    let mut logo = solid_layer("logo", Rect::new(0, 0, 1, 1), yellow);
    logo.blend = BlendMode::LighterColor;
    d.layers.push(logo);
    for adj in [Adjustment::BrightnessContrast { brightness: 150.0, contrast: 0.0, legacy: false }, Adjustment::Vibrance { vibrance: 0.0, saturation: -100.0 }]
    {
        let mut l = Layer::new("adj", LayerContent::Adjustment(adj));
        l.clipped = true;
        d.layers.push(l);
    }
    let p = px(&d, 0, 0);
    assert!(p[..3].iter().all(|v| *v * 255.0 >= 252.0), "logo whitened: {p:?}");
    assert!(close4(px(&d, 1, 0), yellow), "the yellow beside it is untouched");
}

#[test]
fn outer_glow_does_not_paint_through_a_zero_fill_layer() {
    use photocraft_doc::{Contour, Effect, FxCommon, FxPaint, Glow, GlowSource, GlowTechnique};

    let background = [0.18, 0.42, 0.85, 1.0];
    let rect = Rect::new(24, 24, 72, 72);
    for technique in [GlowTechnique::Softer, GlowTechnique::Precise] {
        let mut d = doc_white(96, 96);
        let bounds = d.bounds();
        d.layers[0].surface_mut().unwrap().fill_rect(bounds, &background);
        let mut layer = solid_layer("glow", rect, [1.0, 0.92, 0.25, 1.0]);
        layer.fill_opacity = 0.0;
        layer.effects.items.push(Effect::OuterGlow(Glow {
            common: FxCommon::new(BlendMode::Normal, 1.0),
            paint: FxPaint::Color(Color::rgb(1.0, 0.94, 0.28)),
            technique,
            spread: 0.5,
            size: 12.0,
            contour: Contour::Linear,
            anti_alias: false,
            range: 0.5,
            jitter: 0.0,
            noise: 0.0,
            source: GlowSource::Edge,
        }));
        d.layers.push(layer);

        let center = px(&d, 48, 48);
        assert!(close4(center, background), "{technique:?}: glow leaked into fully transparent fill: {center:?}");
        let distant = px(&d, 5, 5);
        assert!(close4(distant, background), "{technique:?}: changed distant background: {distant:?}");
        let halo = px(&d, 22, 48);
        assert!(halo[0] > background[0] + 0.1, "{technique:?}: exterior halo disappeared: {halo:?}");

        d.layers[1].fill_opacity = 1.0;
        let opaque = px(&d, 48, 48);
        assert!(opaque[0] > 0.9 && opaque[1] > 0.8, "{technique:?}: opaque fill disappeared: {opaque:?}");
    }
}

// Lab documents: Invert mirrors L, a and b; Threshold splits on L with a neutral result; Gradient
// Map follows L. Rendered through the document compositor at every depth.
#[test]
fn lab_documents_evaluate_invert_threshold_and_gradient_map_on_lab() {
    use photocraft_color::convert::{lab_to_srgb, srgb_to_lab};
    let c = lab_to_srgb([30.0, 12.0, -18.0]);
    for depth in SampleType::ALL {
        let render = |adj: Adjustment| {
            let mut d = Document::with_background("t", Size::new(4, 4), ColorMode::Lab, depth, Color::rgb(c[0], c[1], c[2]));
            d.layers.push(Layer::new("adj", LayerContent::Adjustment(adj)));
            let p = flatten(&d).px[5];
            srgb_to_lab([p[0], p[1], p[2]])
        };
        let near = |got: [f32; 3], want: [f32; 3], tol: f32| got.iter().zip(want).all(|(g, w)| (g - w).abs() <= tol);
        let inv = render(Adjustment::Invert);
        // 8-bit bytes become 255 - v (a, b one step past the mirror); 16 bits mirror them exactly.
        let want = if depth == SampleType::U8 { [70.0, -13.0, 17.0] } else { [70.0, -12.0, 18.0] };
        assert!(near(inv, want, 0.5), "{depth:?}: {inv:?}");
        let th = render(Adjustment::Threshold { level: 0.25 });
        assert!(near(th, [100.0, 0.0, 0.0], 0.5), "{depth:?}: L 30 is above level 64: {th:?}");
        let gm = render(Adjustment::GradientMap { stops: vec![(0.0, [0.0; 3]), (1.0, [1.0; 3])], reverse: false, dither: false });
        let want = srgb_to_lab([0.3; 3]);
        assert!(near(gm, want, 1.0), "{depth:?}: L 30 → 30 % of the way: {gm:?} vs {want:?}");
    }
}

/// `adj` over a one-colour Lab document (background `rgb`) at `depth`, read back as Lab.
fn lab_doc_adjusted(rgb: [f32; 3], adj: Adjustment, depth: SampleType) -> [f32; 3] {
    let mut doc = Document::with_background("lab", Size::new(4, 4), ColorMode::Lab, depth, Color::rgb(rgb[0], rgb[1], rgb[2]));
    doc.layers.push(Layer::new("adj", LayerContent::Adjustment(adj)));
    let p = px(&doc, 1, 1);
    photocraft_color::convert::srgb_to_lab([p[0], p[1], p[2]])
}

fn near3(got: [f32; 3], want: [f32; 3], tol: f32, what: &str) {
    assert!(got.iter().zip(want).all(|(g, w)| (g - w).abs() <= tol), "{what}: got {got:?}, want {want:?}");
}

/// A muted, in-gamut colour (L ≈ 46, a ≈ 10, b ≈ -20) and its Lab coordinates.
const MUTED: [f32; 3] = [0.40, 0.42, 0.60];

#[test]
fn lab_documents_invert_and_posterize_l_a_and_b() {
    let lab = photocraft_color::convert::srgb_to_lab(MUTED);
    for depth in SampleType::ALL {
        // Invert: L -> 100 - L and a, b -> their opposites (-a - 1 on the 8-bit 0..255 grid).
        let inv = lab_doc_adjusted(MUTED, Adjustment::Invert, depth);
        let step = if depth == SampleType::U8 { 1.0 } else { 0.0 };
        near3(inv, [100.0 - lab[0], -lab[1] - step, -lab[2] - step], 1.0, &format!("{depth:?} invert"));
        // An RGB invert would land elsewhere: lightness 100 - L is the Lab rule.
        let rgb_inv = photocraft_color::convert::srgb_to_lab(MUTED.map(|v| 1.0 - v));
        assert!((rgb_inv[0] - inv[0]).abs() > 2.0, "{depth:?}: {rgb_inv:?} vs {inv:?}");
        // Posterize, three levels: the same steps on L, a and b (here all land on the middle
        // step: L ≈ 50 and a, b ≈ -1, a neutral grey).
        let post = lab_doc_adjusted(MUTED, Adjustment::Posterize { levels: 3 }, depth);
        let n = [lab[0] / 100.0, (lab[1] + 128.0) / 255.0, (lab[2] + 128.0) / 255.0].map(|v| adjust::posterize(v, 3));
        let want = [n[0] * 100.0, n[1] * 255.0 - 128.0, n[2] * 255.0 - 128.0];
        assert!(want[0] > 45.0 && want[1].abs() < 2.0 && want[2].abs() < 2.0, "{want:?}");
        near3(post, want, 1.0, &format!("{depth:?} posterize"));
    }
}

#[test]
fn lab_documents_brightness_contrast_moves_lightness_only() {
    let lab = photocraft_color::convert::srgb_to_lab(MUTED);
    for depth in SampleType::ALL {
        for legacy in [true, false] {
            let got = lab_doc_adjusted(MUTED, Adjustment::BrightnessContrast { brightness: 40.0, contrast: 20.0, legacy }, depth);
            assert!(got[0] > lab[0] + 5.0, "{depth:?} legacy {legacy}: {got:?}");
            near3([lab[0], got[1], got[2]], lab, 0.75, &format!("{depth:?} legacy {legacy}: a, b kept"));
        }
    }
}

#[test]
fn lab_documents_threshold_on_lightness_with_a_neutral_result() {
    for depth in SampleType::ALL {
        // A saturated yellow is light (L ≈ 97): white, not a coloured result.
        let yellow = lab_doc_adjusted([1.0, 1.0, 0.0], Adjustment::Threshold { level: 0.5 }, depth);
        near3(yellow, [100.0, 0.0, 0.0], 0.5, &format!("{depth:?} yellow"));
        // Pure blue has L ≈ 30: black.
        let blue = lab_doc_adjusted([0.0, 0.0, 1.0], Adjustment::Threshold { level: 0.5 }, depth);
        near3(blue, [0.0, 0.0, 0.0], 0.5, &format!("{depth:?} blue"));
        // Mid grey L 53.4 against level 128 (n 0.502): Lab L decides, not RGB luminance.
        let level = 0.52;
        let grey = lab_doc_adjusted([0.5; 3], Adjustment::Threshold { level }, depth);
        near3(grey, [100.0, 0.0, 0.0], 0.5, &format!("{depth:?} grey"));
    }
}

#[test]
fn lab_documents_gradient_map_follows_lightness() {
    let stops = vec![(0.0, [0.0, 0.0, 1.0]), (1.0, [1.0, 1.0, 0.0])];
    // A grey and a saturated colour of the same L map to the same colour.
    let red = [0.9, 0.1, 0.1];
    let l = photocraft_color::convert::srgb_to_lab(red)[0];
    let grey = photocraft_color::convert::lab_to_srgb([l, 0.0, 0.0]);
    for depth in SampleType::ALL {
        for reverse in [false, true] {
            let adj = Adjustment::GradientMap { stops: stops.clone(), reverse, dither: false };
            let a = lab_doc_adjusted(red, adj.clone(), depth);
            let b = lab_doc_adjusted(grey, adj.clone(), depth);
            near3(a, b, 1.0, &format!("{depth:?} reverse {reverse}"));
            let t = if reverse { 1.0 - l / 100.0 } else { l / 100.0 };
            let want = photocraft_color::convert::srgb_to_lab([t, t, 1.0 - t]);
            near3(a, want, 1.5, &format!("{depth:?} reverse {reverse}"));
        }
    }
    // In RGB documents the map still follows RGB luminance: red and grey differ there.
    let mut buf = Buffer::filled(Rect::new(0, 0, 1, 1), [red[0], red[1], red[2], 1.0]);
    let adj = Adjustment::GradientMap { stops, reverse: false, dither: false };
    adjust::apply_in(&adj, &mut buf, adjust::Transfer::Srgb, None, ColorMode::Rgb);
    let mut grey_buf = Buffer::filled(Rect::new(0, 0, 1, 1), [grey[0], grey[1], grey[2], 1.0]);
    adjust::apply_in(&adj, &mut grey_buf, adjust::Transfer::Srgb, None, ColorMode::Rgb);
    assert!((buf.px[0][0] - grey_buf.px[0][0]).abs() > 0.05, "{:?} {:?}", buf.px[0], grey_buf.px[0]);
}

#[test]
fn render_refuses_rects_beyond_the_pixel_cap() {
    let doc = Document::new("c", Size::new(4, 4), ColorMode::Rgb, SampleType::U8);
    // 2^32 - 1 px wide: the width wraps negative as an i32, which once took the single-tile path.
    for tile in [crate::RENDER_TILE, 1 << 20] {
        let b = crate::render_tiled(&doc, Rect::new(i32::MIN, 0, i32::MAX, 1), tile);
        assert!(b.px.is_empty() && b.rect.is_empty());
    }
}

// The effect-map cache key covers Transparency Shapes Layer and Blend If: both change the shape
// the maps are built from (a layer's own, or a styled group's children composite) without
// touching pixels, masks or effects, and stale maps kept stroking a hole the layer no longer
// had. Each case renders with the cache warm, edits, renders again and compares with a cold
// render. (A purge by a concurrent test only makes the warm render cold, never a false failure.)
#[test]
fn effect_cache_follows_transparency_shapes_layer() {
    let mut d = doc_white(32, 32);
    let mut l = solid_layer("holed", Rect::new(0, 0, 32, 32), [0.9, 0.2, 0.2, 1.0]);
    l.surface_mut().unwrap().fill_rect(Rect::new(12, 12, 20, 20), &[0.0, 0.0, 0.0, 0.0]);
    l.effects.items = vec![stroke(2.0, StrokePosition::Outside)];
    assert!(l.advanced.transparency_shapes);
    d.layers.push(l);
    let on = render(&d, d.bounds());
    d.layers[1].advanced.transparency_shapes = false;
    let warm = render(&d, d.bounds());
    purge_effect_cache();
    let cold = render(&d, d.bounds());
    assert!(on.px != cold.px, "the stroke should stop rimming the hole");
    assert!(warm.px == cold.px, "stale effect maps after turning Transparency Shapes Layer off");
}

#[test]
fn effect_cache_follows_a_group_childs_blend_if() {
    let mut d = doc_white(32, 32);
    let mut child = solid_layer("bw", Rect::new(4, 4, 16, 28), [0.0, 0.0, 0.0, 1.0]);
    child.surface_mut().unwrap().fill_rect(Rect::new(16, 4, 28, 28), &[1.0, 1.0, 1.0, 1.0]);
    let mut g = Layer::group("g", vec![child]);
    g.effects.items = vec![Effect::DropShadow(Shadow { size: 2.0, spread: 0.0, ..shadow(3.0, 90.0) })];
    d.layers.push(g);
    let before = render(&d, d.bounds());
    let LayerContent::Group(grp) = &mut d.layers[1].content else { panic!("not a group") };
    // Gray › This Layer: black point at 128 hides the black half.
    grp.children[0].blend_if.set(0, [range([128, 128], [255, 255]), FULL]);
    let warm = render(&d, d.bounds());
    purge_effect_cache();
    let cold = render(&d, d.bounds());
    assert!(before.px != cold.px, "Blend If should change the group's shadow");
    assert!(warm.px == cold.px, "stale group effect maps after a child's Blend If changed");
}

#[test]
fn shape_layer_bounds_includes_vector_bounds_clipped_to_canvas_and_preserves_effects() {
    let canvas = Rect::new(0, 0, 100, 100);
    let subpath = photocraft_doc::vector::Subpath::polygon(&[(-20.0, -20.0), (50.0, -20.0), (50.0, 50.0), (-20.0, 50.0)]);
    let path = photocraft_doc::vector::Path { subpaths: vec![subpath], ..Default::default() };
    let sh = photocraft_doc::vector::ShapeLayer {
        path,
        fill: Some(photocraft_doc::Fill::Solid(photocraft_doc::Color::BLACK)),
        stroke: Some(photocraft_doc::vector::ShapeStroke { width: 4.0, opacity: 1.0, ..Default::default() }),
        ..Default::default()
    };
    let mut layer = Layer::new("shape", LayerContent::Shape(sh));

    // Vector bounds reach outside canvas (-24..54), but layer_bounds clips vector bounds to canvas:
    let b = layer_bounds(&layer, canvas);
    assert_eq!(b, Rect::new(0, 0, 54, 54), "shape vector bounds must clip to canvas and not reach past it");

    // When the shape has layer effects, layer_bounds preserves effects union handling:
    layer.effects.items = vec![Effect::DropShadow(Shadow { size: 5.0, spread: 0.0, ..shadow(5.0, 90.0) })];
    let b_fx = layer_bounds(&layer, canvas);
    assert!(!b_fx.is_empty());
    assert!(b_fx.contains_rect(&b));
}
