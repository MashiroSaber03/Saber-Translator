//! Composite oracle: the merged image stored in a PSD must match our
//! compositor's rendering of the imported document.

mod common;

use common::*;
use photocraft_color::{BlendMode, ColorMode, SampleType};
use photocraft_io::*;
use photocraft_psd::{PixelData, PsdBuilder, PsdFile};

const TOL: f32 = 1.0 / 255.0 + 1e-4;

fn check_export_oracle(d: &photocraft_doc::Document, tol: f32) {
    let r = export(d, "x.psd", &ExportOptions::default()).unwrap();
    let f = PsdFile::from_bytes(&r.bytes).unwrap();
    let merged = merged_composite(&f).unwrap();
    let d2 = import("x.psd", &r.bytes).unwrap().document;
    let mut ours = photocraft_compose::flatten(&d2).px;
    if d.mode == ColorMode::Cmyk {
        // The merged CMYK image is the RGB composite separated through the CMYK profile, which
        // gamut-maps colours that blend modes push outside CMYK: project ours the same way.
        let fmt = d.pixel_format();
        for p in &mut ours {
            *p = photocraft_raster::to_rgba(&fmt, &photocraft_raster::from_rgba(&fmt, *p));
        }
    }
    let m = max_diff(&ours, &merged);
    assert!(m <= tol, "max diff {m} > {tol}");
}

macro_rules! oracle {
    ($name:ident, $mode:expr, $depth:expr, $f:expr, $tol:expr) => {
        #[test]
        fn $name() {
            check_export_oracle(&gen_doc($mode, $depth, $f), $tol);
        }
    };
}

oracle!(oracle_rgb8, ColorMode::Rgb, SampleType::U8, Features::ALL, TOL);
oracle!(oracle_rgb16, ColorMode::Rgb, SampleType::U16, Features::ALL, TOL);
oracle!(oracle_rgb32, ColorMode::Rgb, SampleType::F32, Features::ALL, TOL);
oracle!(oracle_gray8, ColorMode::Grayscale, SampleType::U8, Features::ALL, TOL);
oracle!(oracle_gray16, ColorMode::Grayscale, SampleType::U16, Features::ALL, TOL);
// CMYK/Lab merged data is converted from the RGB composite (CMYK through the built-in
// profile), so the stored merged image differs by a quantization round trip through the model.
oracle!(oracle_cmyk8, ColorMode::Cmyk, SampleType::U8, Features::PIXELS, 3.0 / 255.0);
// Lab: one 8-bit Lab quantization step is amplified up to ~15/255 near the sRGB gamut
// edge (the sRGB toe has slope 12.92), so the tolerance is wider.
oracle!(oracle_lab8, ColorMode::Lab, SampleType::U8, Features::PIXELS, 16.0 / 255.0);

/// Builds a PSD with the *builder* (independent writer) whose merged image
/// is produced by photocraft-compose, then checks import + flatten against it.
fn builder_oracle(depth: u16, mode: ColorMode) {
    let sample = if depth == 16 { SampleType::U16 } else { SampleType::U8 };
    let d = gen_doc(mode, sample, Features { groups: false, masks: true, adjustments: false, fills: false, all_blends: true, extras: false });
    let flat = photocraft_compose::flatten(&d);
    let to_pixels = |px: &[[f32; 4]]| -> PixelData {
        let fmt = d.pixel_format();
        let vals: Vec<f32> = px
            .iter()
            .flat_map(|p| {
                let v = photocraft_raster::from_rgba(&fmt, *p);
                v.into_iter()
            })
            .collect();
        match (mode, depth) {
            (ColorMode::Grayscale, 8) => PixelData::GrayA8(vals.iter().map(|v| (v * 255.0).round() as u8).collect()),
            (ColorMode::Grayscale, _) => PixelData::GrayA16(vals.iter().map(|v| (v * 65535.0).round() as u16).collect()),
            (_, 8) => PixelData::Rgba8(vals.iter().map(|v| (v * 255.0).round() as u8).collect()),
            _ => PixelData::Rgba16(vals.iter().map(|v| (v * 65535.0).round() as u16).collect()),
        }
    };
    let mut b = PsdBuilder::new(d.size.width, d.size.height).depth(depth).color_mode(match mode {
        ColorMode::Grayscale => photocraft_psd::ColorMode::Grayscale,
        _ => photocraft_psd::ColorMode::Rgb,
    });
    for l in &d.layers {
        let s = l.surface().unwrap();
        let r = s.content_bounds();
        let buf: Vec<[f32; 4]> = if r.is_empty() {
            Vec::new()
        } else {
            let fmt = s.format();
            s.read_region(r).chunks_exact(fmt.channels()).map(|p| photocraft_raster::to_rgba(&fmt, p)).collect()
        };
        let mut spec = photocraft_psd::LayerSpec::new(l.name.clone(), r.x0, r.y0, r.width(), r.height(), to_pixels(&buf));
        spec.blend_mode = photocraft_psd::BlendMode::from_key(l.blend.psd_key());
        spec.opacity = (l.opacity * 255.0).round() as u8;
        spec.fill_opacity = Some((l.fill_opacity * 255.0).round() as u8);
        spec.visible = l.visible;
        spec.clipping = l.clipped;
        if let Some(m) = &l.mask {
            let mr = m.surface.content_bounds();
            let data = m.surface.read_region(mr).iter().map(|v| (v * 255.0).round() as u8).collect();
            spec.mask = Some(photocraft_psd::MaskSpec {
                rect: photocraft_psd::Rect { top: mr.y0, left: mr.x0, bottom: mr.y1, right: mr.x1 },
                data,
                default_color: (m.surface.default_pixel()[0] * 255.0).round() as u8,
                disabled: !m.enabled,
            });
        }
        b.push_layer(spec);
    }
    b.composite(to_pixels(&flat.px));
    let bytes = b.to_bytes().unwrap();
    let f = PsdFile::from_bytes(&bytes).unwrap();
    let merged = merged_composite(&f).unwrap();
    let d2 = import("b.psd", &bytes).unwrap().document;
    let ours = photocraft_compose::flatten(&d2).px;
    let m = max_diff(&ours, &merged);
    // Density/feather of masks are not written by the builder; the generated
    // doc uses them only on a disabled mask, so results must match.
    assert!(m <= 2.0 / 255.0, "builder oracle {mode:?} {depth}: max diff {m}");
}

#[test]
fn builder_oracle_rgb8() {
    builder_oracle(8, ColorMode::Rgb);
}
#[test]
fn builder_oracle_rgb16() {
    builder_oracle(16, ColorMode::Rgb);
}
#[test]
fn builder_oracle_gray8() {
    builder_oracle(8, ColorMode::Grayscale);
}
#[test]
fn builder_oracle_gray16() {
    builder_oracle(16, ColorMode::Grayscale);
}

#[test]
fn every_blend_mode_single_layer_oracle() {
    for m in BlendMode::LAYER_MODES {
        if m == BlendMode::Dissolve {
            continue;
        }
        let mut d = photocraft_doc::Document::new("b", photocraft_geom::Size::new(8, 8), ColorMode::Rgb, SampleType::U8);
        let fmt = d.pixel_format();
        d.layers.push(raster("bg", fmt, d.bounds(), 1, false));
        let mut top = raster("top", fmt, photocraft_geom::Rect::new(2, 2, 7, 7), 9, true);
        top.blend = m;
        top.opacity = g(190);
        d.layers.push(top);
        check_export_oracle(&d, TOL);
    }
}

#[test]
fn hidden_and_clipped_affect_composite() {
    let d = gen_doc(ColorMode::Rgb, SampleType::U8, Features::PIXELS);
    let f = document_to_psd(&d);
    let merged = merged_composite(&f).unwrap();
    let mut d2 = d.clone();
    for l in &mut d2.layers {
        l.visible = true;
    }
    let other = photocraft_compose::flatten(&d2).px;
    assert!(max_diff(&other, &merged) > 0.0, "unhiding a layer must change the composite");
}

#[test]
fn merged_alpha_written_when_transparent() {
    let mut d = photocraft_doc::Document::new("t", photocraft_geom::Size::new(4, 4), ColorMode::Rgb, SampleType::U8);
    let fmt = d.pixel_format();
    d.layers.push(raster("half", fmt, photocraft_geom::Rect::new(0, 0, 2, 4), 3, true));
    let f = document_to_psd(&d);
    assert_eq!(f.header.channels, 4);
    assert!(f.merged_has_alpha());
    let merged = merged_composite(&f).unwrap();
    assert_eq!(merged[3][3], 0.0);

    // U8 rounding must not erase transparency that is representable in the exported U16 plane.
    let mut near = photocraft_doc::Document::new("near opaque", photocraft_geom::Size::new(1, 1), ColorMode::Rgb, SampleType::U16);
    let mut layer = photocraft_doc::Layer::raster("pixel", near.pixel_format());
    layer.surface_mut().unwrap().fill_rect(near.bounds(), &[1.0, 0.0, 0.0, 65500.0 / 65535.0]);
    near.layers.push(layer);
    let out = export(&near, "near.psd", &ExportOptions::default()).unwrap();
    let file = PsdFile::from_bytes(&out.bytes).unwrap();
    assert_eq!(file.header.channels, 4);
    assert!(file.merged_has_alpha());
    let samples = file.decode_merged().unwrap();
    assert_eq!(u16::from_be_bytes([samples[6], samples[7]]), 65500);
}
