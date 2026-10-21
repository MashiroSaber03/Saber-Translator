//! Tests for the second filter batch (pixelate, stylize, render, blur gallery, video…).

use super::*;
use photocraft_color::{PixelFormat, SampleType};
use std::sync::Arc;

fn fmt(s: SampleType) -> PixelFormat {
    PixelFormat::new(ColorMode::Rgb, s, true)
}

const R: Rect = Rect { x0: 0, y0: 0, x1: 40, y1: 30 };

fn pattern(s: SampleType, r: Rect) -> Surface {
    let mut surf = Surface::new(fmt(s));
    let mut v = Vec::new();
    for y in r.y0..r.y1 {
        for x in r.x0..r.x1 {
            v.extend_from_slice(&[((x * 7 + y * 3) % 64) as f32 / 63.0, ((x * x + y) % 50) as f32 / 49.0, (y % 9) as f32 / 8.0, 1.0]);
        }
    }
    surf.write_region(r, &v);
    surf
}

fn flat(s: SampleType, r: Rect, px: [f32; 4]) -> Surface {
    let mut surf = Surface::new(fmt(s));
    surf.fill_rect(r, &px);
    surf
}

fn run(s: &Surface, p: &FilterParams) -> Surface {
    let area = output_area(p, s.content_bounds(), R, None);
    apply(s, p, area, R, None)
}

fn map_image(v: [f32; 4]) -> Arc<Image> {
    let r = Rect::new(0, 0, 8, 8);
    Arc::new(Image { rect: r, ch: 4, data: (0..64).flat_map(|_| v).collect() })
}

const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

fn new_filters() -> Vec<FilterParams> {
    vec![
        FilterParams::ColorHalftone { max_radius: 4.0, angles: [108.0, 162.0, 90.0, 45.0] },
        FilterParams::Crystallize { cell_size: 6.0, seed: 1 },
        FilterParams::Facet,
        FilterParams::Fragment,
        FilterParams::Mezzotint { kind: MezzotintType::MediumLines, seed: 2 },
        FilterParams::Mezzotint { kind: MezzotintType::GrainyDots, seed: 2 },
        FilterParams::Pointillize { cell_size: 5.0, seed: 3, background: WHITE },
        FilterParams::Diffuse { mode: DiffuseMode::Normal, seed: 4 },
        FilterParams::Diffuse { mode: DiffuseMode::Anisotropic, seed: 4 },
        FilterParams::Extrude { kind: ExtrudeType::Blocks, size: 6.0, depth: 20.0, level_based: false, solid_front: false, mask_incomplete: false, seed: 5 },
        FilterParams::Extrude { kind: ExtrudeType::Pyramids, size: 6.0, depth: 30.0, level_based: true, solid_front: false, mask_incomplete: true, seed: 5 },
        FilterParams::OilPaint { stylization: 3.0, cleanliness: 4.0, scale: 1.0, bristle_detail: 5.0, lighting: true, angle: -60.0, shine: 2.0 },
        FilterParams::Tiles { count: 4, max_offset: 20.0, fill: TileFill::Background, foreground: BLACK, background: WHITE, seed: 6 },
        FilterParams::TraceContour { level: 128.0, upper: true },
        FilterParams::Wind { method: WindMethod::Blast, from_right: false, seed: 7 },
        FilterParams::Wind { method: WindMethod::Stagger, from_right: true, seed: 7 },
        FilterParams::Displace { horizontal: 10.0, vertical: 10.0, stretch: true, undefined: UndefinedAreas::Wrap, map: Some(map_image([0.8, 0.3, 0.5, 1.0])) },
        FilterParams::Shear { points: vec![[0.0, 0.0], [0.5, 0.3], [1.0, 0.0]], undefined: UndefinedAreas::Repeat },
        FilterParams::ZigZag { amount: 50.0, ridges: 4.0, style: ZigZagStyle::PondRipples },
        FilterParams::Fibers { variance: 16.0, strength: 4.0, seed: 8, foreground: BLACK, background: WHITE },
        FilterParams::LensFlare { brightness: 100.0, center_x: 0.3, center_y: 0.3, lens: LensType::Zoom },
        FilterParams::LightingEffects {
            lights: vec![Light::default()],
            gloss: 0.0,
            metallic: 0.0,
            exposure: 0.0,
            ambience: 10.0,
            texture: TextureChannel::Luminance,
            height: 50.0,
            white_is_high: true,
        },
        FilterParams::Relight { angle: 45.0, elevation: 40.0, intensity: 40.0, ambient: 55.0, warmth: 0.0, softness: 25.0 },
        FilterParams::ReduceNoise { strength: 6.0, preserve_details: 60.0, reduce_color_noise: 45.0, sharpen_details: 25.0, remove_jpeg_artifact: true },
        FilterParams::SmartBlur { radius: 3.0, threshold: 40.0, quality: BlurQuality::High, mode: SmartBlurMode::Normal },
        FilterParams::SmartBlur { radius: 3.0, threshold: 25.0, quality: BlurQuality::Low, mode: SmartBlurMode::OverlayEdge },
        FilterParams::LensBlur {
            radius: 4.0,
            blades: 6,
            curvature: 20.0,
            rotation: 10.0,
            depth: DepthSource::None,
            focal_distance: 0.0,
            invert_depth: false,
            brightness: 30.0,
            threshold: 200.0,
            noise: 5.0,
            distribution: Distribution::Uniform,
            monochromatic: false,
            seed: 9,
            depth_map: None,
        },
        FilterParams::ShapeBlur { radius: 3.0, shape: BlurShape::Star },
        FilterParams::TiltShift { blur: 6.0, center_x: 0.5, center_y: 0.5, angle: 0.0, focus: 0.1, transition: 0.2 },
        FilterParams::IrisBlur { pins: vec![IrisPin { blur: 6.0, ..IrisPin::default() }] },
        FilterParams::FieldBlur { pins: vec![FieldPin { x: 0.2, y: 0.2, blur: 0.0 }, FieldPin { x: 0.8, y: 0.8, blur: 6.0 }] },
        FilterParams::SpinBlur { pins: vec![SpinPin { blur_angle: 30.0, ..SpinPin::default() }] },
        FilterParams::PathBlur { paths: vec![BlurPath { points: vec![[0.1, 0.2], [0.9, 0.8]], speed: 8.0, taper: 30.0 }] },
        FilterParams::Custom {
            kernel: vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 5.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            scale: 1.0,
            offset: 0.0,
        },
        FilterParams::HsbHsl { input: HsbModel::Rgb, output: HsbModel::Hsl },
        FilterParams::ColorToAlpha { color: WHITE, transparency_threshold: 0.0, opacity_threshold: 1.0 },
        FilterParams::ColorToAlpha { color: [0.5, 0.25, 0.75, 1.0], transparency_threshold: 0.1, opacity_threshold: 0.9 },
        FilterParams::DeInterlace { eliminate_even: false, interpolate: true },
        FilterParams::NtscColors,
    ]
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn mean_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len().max(1) as f32
}

#[test]
fn new_filters_are_tile_independent() {
    let s = pattern(SampleType::U8, R);
    for p in new_filters() {
        let area = output_area(&p, s.content_bounds(), R, None);
        let a = apply_tiled(&s, &p, area, R, None, 256, None);
        let b = apply_tiled(&s, &p, area, R, None, 7, None);
        let big = area.union(&R);
        // Running-sum blurs may round differently by one level depending on where a window starts.
        let d = max_diff(&a.read_region(big), &b.read_region(big));
        assert!(d <= 1.5 / 255.0, "{} differs between tile sizes by {d}", p.label());
    }
}

#[test]
fn new_filters_change_pixels_and_respect_empty_selection() {
    let s = pattern(SampleType::U16, R);
    let none = Surface::new(PixelFormat::GRAY8);
    for p in new_filters() {
        let out = run(&s, &p);
        assert!(out != s, "{} did nothing", p.label());
        let area = output_area(&p, s.content_bounds(), R, None);
        let masked = apply(&s, &p, area, R, Some(&none));
        assert_eq!(masked.read_region(area), s.read_region(area), "{} ignored an empty selection", p.label());
    }
}

#[test]
fn new_filters_agree_across_bit_depths() {
    // Threshold-type filters flip where quantized inputs straddle a threshold; compare them on average only.
    let thresholdy = |p: &FilterParams| {
        matches!(
            p,
            FilterParams::Facet
                | FilterParams::OilPaint { .. }
                | FilterParams::Mezzotint { .. }
                | FilterParams::TraceContour { .. }
                | FilterParams::SmartBlur { .. }
                | FilterParams::ColorHalftone { .. }
                | FilterParams::Diffuse { .. }
                | FilterParams::Wind { .. }
        ) || matches!(p, FilterParams::Extrude { level_based: true, .. })
    };
    for p in new_filters() {
        let outs: Vec<Vec<f32>> = [SampleType::U8, SampleType::U16, SampleType::F32]
            .iter()
            .map(|&st| {
                let s = pattern(st, R);
                // Clamp float results the way integer storage does, so HDR overshoot does not count.
                run(&s, &p).read_region(R).into_iter().map(|v| v.clamp(0.0, 1.0)).collect()
            })
            .collect();
        for k in 1..3 {
            let (mx, mn) = (max_diff(&outs[0], &outs[k]), mean_diff(&outs[0], &outs[k]));
            if thresholdy(&p) {
                assert!(mn < 0.06, "{}: mean depth difference {mn}", p.label());
            } else {
                assert!(mx < 0.06 && mn < 0.01, "{}: depth difference max {mx} mean {mn}", p.label());
            }
        }
    }
}

#[test]
fn new_filters_work_in_other_colour_modes() {
    for mode in [ColorMode::Grayscale, ColorMode::Cmyk, ColorMode::Lab] {
        let f = PixelFormat::new(mode, SampleType::U8, true);
        let mut s = Surface::new(f);
        let src = pattern(SampleType::F32, R);
        for y in R.y0..R.y1 {
            for x in R.x0..R.x1 {
                let px = photocraft_raster::from_rgba(&f, photocraft_raster::to_rgba(&src.format(), &src.pixel(x, y)));
                s.write_pixel(x, y, &px);
            }
        }
        for p in new_filters() {
            let out = run(&s, &p);
            assert_eq!(out.format(), f);
            assert!(out.read_region(R).iter().all(|v| v.is_finite()), "{} {mode:?}", p.label());
        }
    }
}

#[test]
fn new_params_serde_roundtrip() {
    for p in new_filters() {
        let j = serde_json::to_string(&p).unwrap();
        let back: FilterParams = serde_json::from_str(&j).unwrap();
        // Caller-supplied maps are not serialized.
        let strip = |p: &FilterParams| match p.clone() {
            FilterParams::Displace { horizontal, vertical, stretch, undefined, .. } => {
                FilterParams::Displace { horizontal, vertical, stretch, undefined, map: None }
            }
            other => other,
        };
        assert_eq!(back, strip(&p), "{j}");
    }
}

#[test]
fn identities_at_zero_strength() {
    let s = pattern(SampleType::U8, R);
    let cases = vec![
        FilterParams::Displace { horizontal: 0.0, vertical: 0.0, stretch: true, undefined: UndefinedAreas::Wrap, map: Some(map_image([0.9, 0.1, 0.0, 1.0])) },
        FilterParams::Displace { horizontal: 50.0, vertical: 50.0, stretch: true, undefined: UndefinedAreas::Wrap, map: None },
        FilterParams::Shear { points: vec![[0.0, 0.0], [1.0, 0.0]], undefined: UndefinedAreas::Wrap },
        FilterParams::ZigZag { amount: 0.0, ridges: 5.0, style: ZigZagStyle::AroundCenter },
        FilterParams::Custom { kernel: (0..25).map(|i| if i == 12 { 1.0 } else { 0.0 }).collect(), scale: 1.0, offset: 0.0 },
        FilterParams::HsbHsl { input: HsbModel::Hsb, output: HsbModel::Hsb },
        FilterParams::ReduceNoise { strength: 0.0, preserve_details: 60.0, reduce_color_noise: 0.0, sharpen_details: 0.0, remove_jpeg_artifact: false },
        FilterParams::LensBlur {
            radius: 0.0,
            blades: 6,
            curvature: 0.0,
            rotation: 0.0,
            depth: DepthSource::None,
            focal_distance: 0.0,
            invert_depth: false,
            brightness: 0.0,
            threshold: 255.0,
            noise: 0.0,
            distribution: Distribution::Uniform,
            monochromatic: false,
            seed: 0,
            depth_map: None,
        },
        FilterParams::ShapeBlur { radius: 0.0, shape: BlurShape::Heart },
        FilterParams::TiltShift { blur: 0.0, center_x: 0.5, center_y: 0.5, angle: 30.0, focus: 0.1, transition: 0.2 },
        FilterParams::IrisBlur { pins: vec![IrisPin { blur: 0.0, ..IrisPin::default() }] },
        FilterParams::FieldBlur { pins: vec![FieldPin { blur: 0.0, ..FieldPin::default() }] },
        FilterParams::SpinBlur { pins: vec![SpinPin { blur_angle: 0.0, ..SpinPin::default() }] },
        FilterParams::PathBlur { paths: vec![BlurPath { speed: 0.0, ..BlurPath::default() }] },
    ];
    for p in cases {
        let out = run(&s, &p);
        let d = max_diff(&out.read_region(R), &s.read_region(R));
        assert!(d <= 0.5 / 255.0, "{} is not identity ({d})", p.label());
    }
}

#[test]
fn flat_images_stay_flat() {
    let c = [0.3, 0.55, 0.8, 1.0];
    let s = flat(SampleType::F32, R, c);
    for p in [
        FilterParams::Crystallize { cell_size: 6.0, seed: 1 },
        FilterParams::Facet,
        FilterParams::OilPaint { stylization: 3.0, cleanliness: 4.0, scale: 1.0, bristle_detail: 0.0, lighting: true, angle: -60.0, shine: 2.0 },
        FilterParams::ReduceNoise { strength: 10.0, preserve_details: 0.0, reduce_color_noise: 100.0, sharpen_details: 0.0, remove_jpeg_artifact: false },
        FilterParams::SmartBlur { radius: 3.0, threshold: 25.0, quality: BlurQuality::High, mode: SmartBlurMode::Normal },
        FilterParams::LensBlur {
            radius: 5.0,
            blades: 5,
            curvature: 50.0,
            rotation: 0.0,
            depth: DepthSource::None,
            focal_distance: 0.0,
            invert_depth: false,
            brightness: 0.0,
            threshold: 255.0,
            noise: 0.0,
            distribution: Distribution::Uniform,
            monochromatic: false,
            seed: 0,
            depth_map: None,
        },
        FilterParams::ShapeBlur { radius: 4.0, shape: BlurShape::Ring },
        FilterParams::TiltShift { blur: 5.0, center_x: 0.5, center_y: 0.5, angle: 0.0, focus: 0.0, transition: 0.1 },
        FilterParams::Wind { method: WindMethod::Blast, from_right: false, seed: 0 },
        FilterParams::DeInterlace { eliminate_even: true, interpolate: true },
    ] {
        let out = run(&s, &p);
        let px = out.pixel(20, 15);
        for k in 0..4 {
            assert!((px[k] - c[k]).abs() < 2e-3, "{}: {px:?}", p.label());
        }
    }
}

#[test]
fn mezzotint_is_binary_and_tracks_brightness() {
    let s = flat(SampleType::F32, R, [0.25, 0.5, 0.75, 1.0]);
    let out = run(&s, &FilterParams::Mezzotint { kind: MezzotintType::FineDots, seed: 1 });
    let v = out.read_region(R);
    let mut mean = [0.0f32; 3];
    for px in v.as_chunks::<4>().0 {
        for k in 0..3 {
            assert!(px[k] == 0.0 || px[k] == 1.0);
            mean[k] += px[k] / 1200.0;
        }
    }
    assert!((mean[0] - 0.25).abs() < 0.06 && (mean[1] - 0.5).abs() < 0.06 && (mean[2] - 0.75).abs() < 0.06, "{mean:?}");
}

#[test]
fn halftone_covers_by_ink() {
    let angles = [108.0, 162.0, 90.0, 45.0];
    let w = run(&flat(SampleType::F32, R, WHITE), &FilterParams::ColorHalftone { max_radius: 4.0, angles });
    assert!(w.read_region(R).iter().all(|&v| (v - 1.0).abs() < 1e-6), "white paper stays white");
    let b = run(&flat(SampleType::F32, R, BLACK), &FilterParams::ColorHalftone { max_radius: 4.0, angles });
    let mean_b: f32 = b.read_region(R).as_chunks::<4>().0.iter().map(|p| p[0]).sum::<f32>() / 1200.0;
    assert!(mean_b < 0.05, "black is fully inked: {mean_b}");
    let g = run(&flat(SampleType::F32, R, [0.5, 0.5, 0.5, 1.0]), &FilterParams::ColorHalftone { max_radius: 4.0, angles });
    let mean_g: f32 = g.read_region(R).as_chunks::<4>().0.iter().map(|p| p[1]).sum::<f32>() / 1200.0;
    assert!((mean_g - 0.5).abs() < 0.12, "mid gray ≈ half coverage: {mean_g}");
}

#[test]
fn pointillize_shows_canvas() {
    let s = flat(SampleType::F32, R, [0.0, 0.0, 1.0, 1.0]);
    let out = run(&s, &FilterParams::Pointillize { cell_size: 10.0, seed: 3, background: [1.0, 0.0, 0.0, 1.0] });
    let v = out.read_region(R);
    assert!(v.as_chunks::<4>().0.iter().any(|p| p[0] > 0.99 && p[2] < 0.01), "canvas visible between dots");
    assert!(v.as_chunks::<4>().0.iter().any(|p| p[2] > 0.9), "dots take the image colour");
}

#[test]
fn diffuse_only_moves_pixels_locally() {
    let s = pattern(SampleType::F32, R);
    let out = run(&s, &FilterParams::Diffuse { mode: DiffuseMode::Normal, seed: 1 });
    for y in 1..29 {
        for x in 1..39 {
            let p = out.pixel(x, y);
            let found = (-1..=1).any(|dy| (-1..=1).any(|dx| s.pixel(x + dx, y + dy) == p));
            assert!(found, "({x},{y})");
        }
    }
    let dark = run(&s, &FilterParams::Diffuse { mode: DiffuseMode::DarkenOnly, seed: 1 });
    let l = |p: Vec<f32>| 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
    for y in 1..29 {
        for x in 1..39 {
            assert!(l(dark.pixel(x, y)) <= l(s.pixel(x, y)) + 1e-6);
        }
    }
}

#[test]
fn trace_contour_draws_line_at_level_crossing() {
    let mut s = flat(SampleType::F32, R, [0.2, 0.2, 0.2, 1.0]);
    s.fill_rect(Rect::new(20, 0, 40, 30), &[0.8, 0.8, 0.8, 1.0]);
    let up = run(&s, &FilterParams::TraceContour { level: 128.0, upper: true });
    assert_eq!(up.pixel(20, 10)[0], 0.0, "line on the bright side");
    assert_eq!(up.pixel(19, 10)[0], 1.0);
    assert_eq!(up.pixel(30, 10)[0], 1.0);
    let lo = run(&s, &FilterParams::TraceContour { level: 128.0, upper: false });
    assert_eq!(lo.pixel(19, 10)[0], 0.0, "line on the dark side");
}

#[test]
fn trace_contour_off_canvas_does_not_panic() {
    let s = flat(SampleType::F32, Rect::new(100, 100, 120, 120), [0.8, 0.8, 0.8, 1.0]);
    let p = FilterParams::TraceContour { level: 128.0, upper: true };
    let _ = apply_tiled(&s, &p, s.content_bounds(), R, None, 64, None);
}

#[test]
fn tiles_leave_gaps_with_fill() {
    let s = flat(SampleType::F32, R, [0.0, 0.0, 1.0, 1.0]);
    let out =
        run(&s, &FilterParams::Tiles { count: 3, max_offset: 40.0, fill: TileFill::Foreground, foreground: [0.0, 1.0, 0.0, 1.0], background: WHITE, seed: 2 });
    let v = out.read_region(R);
    assert!(v.as_chunks::<4>().0.iter().any(|p| p[1] > 0.99), "foreground gaps");
    assert!(v.as_chunks::<4>().0.iter().filter(|p| p[2] > 0.99).count() > 600, "mostly tiles");
}

#[test]
fn extrude_blocks_and_pyramids_shade_cells() {
    let s = pattern(SampleType::F32, R);
    let blocks = run(
        &s,
        &FilterParams::Extrude { kind: ExtrudeType::Blocks, size: 8.0, depth: 40.0, level_based: false, solid_front: true, mask_incomplete: false, seed: 1 },
    );
    // A solid front face is uniform: find two equal neighbours somewhere in the middle.
    let same = (5..35).any(|x| blocks.pixel(x, 12) == blocks.pixel(x + 1, 12));
    assert!(same);
    let flat_s = flat(SampleType::F32, R, [0.5, 0.5, 0.5, 1.0]);
    let pyr = run(
        &flat_s,
        &FilterParams::Extrude {
            kind: ExtrudeType::Pyramids,
            size: 10.0,
            depth: 100.0,
            level_based: false,
            solid_front: false,
            mask_incomplete: false,
            seed: 1,
        },
    );
    // Light from the top-left: the left face is brighter than the right face.
    assert!(pyr.pixel(11, 15)[0] > pyr.pixel(18, 15)[0], "{:?} {:?}", pyr.pixel(11, 15), pyr.pixel(18, 15));
}

#[test]
fn fibers_lie_between_colours_and_are_seeded() {
    let s = flat(SampleType::F32, R, WHITE);
    let p = FilterParams::Fibers { variance: 16.0, strength: 4.0, seed: 1, foreground: [1.0, 0.0, 0.0, 1.0], background: [0.0, 0.0, 1.0, 1.0] };
    let a = run(&s, &p);
    assert_eq!(a, run(&s, &p));
    for px in a.read_region(R).as_chunks::<4>().0 {
        assert!((px[0] + px[2] - 1.0).abs() < 1e-5 && px[1].abs() < 1e-6);
    }
    let b = run(&s, &FilterParams::Fibers { variance: 16.0, strength: 4.0, seed: 2, foreground: [1.0, 0.0, 0.0, 1.0], background: [0.0, 0.0, 1.0, 1.0] });
    assert_ne!(a, b);
}

#[test]
fn lens_flare_is_brightest_at_its_centre() {
    let s = flat(SampleType::F32, R, [0.2, 0.2, 0.2, 1.0]);
    let out = run(&s, &FilterParams::LensFlare { brightness: 100.0, center_x: 0.25, center_y: 0.5, lens: LensType::Prime105 });
    assert!(out.pixel(10, 15)[0] > 0.9);
    assert!(out.pixel(10, 15)[0] > out.pixel(38, 2)[0]);
    assert!(out.pixel(38, 2)[0] >= 0.2 - 1e-6, "screen never darkens");
}

#[test]
fn lighting_point_light_falls_off() {
    let s = flat(SampleType::F32, R, [0.8, 0.8, 0.8, 1.0]);
    let light = Light { kind: LightKind::Point, x: 0.1, y: 0.5, z: 0.2, radius: 1.0, intensity: 50.0, ..Light::default() };
    let out = run(
        &s,
        &FilterParams::LightingEffects {
            lights: vec![light],
            gloss: -100.0,
            metallic: 0.0,
            exposure: 0.0,
            ambience: 0.0,
            texture: TextureChannel::None,
            height: 0.0,
            white_is_high: true,
        },
    );
    assert!(out.pixel(4, 15)[0] > out.pixel(36, 15)[0] + 0.1, "{:?} {:?}", out.pixel(4, 15), out.pixel(36, 15));
    // A spot light does not reach outside its cone.
    let spot = Light { kind: LightKind::Spot, x: 0.5, y: 0.5, z: 0.3, target_x: 0.5, target_y: 0.5, cone: 20.0, ..Light::default() };
    let out = run(
        &s,
        &FilterParams::LightingEffects {
            lights: vec![spot],
            gloss: 0.0,
            metallic: 0.0,
            exposure: 0.0,
            ambience: 0.0,
            texture: TextureChannel::None,
            height: 0.0,
            white_is_high: true,
        },
    );
    assert!(out.pixel(20, 15)[0] > 0.3 && out.pixel(1, 1)[0] < 1e-4, "{:?} {:?}", out.pixel(20, 15), out.pixel(1, 1));
}

fn variance(s: &Surface, r: Rect, c: usize) -> f32 {
    let v: Vec<f32> = s.read_region(r).as_chunks::<4>().0.iter().map(|p| p[c]).collect();
    let m = v.iter().sum::<f32>() / v.len() as f32;
    v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32
}

fn noisy(st: SampleType) -> Surface {
    let s = flat(st, R, [0.5, 0.5, 0.5, 1.0]);
    run(&s, &FilterParams::AddNoise { amount: 20.0, distribution: Distribution::Gaussian, monochromatic: false, seed: 4 })
}

#[test]
fn reduce_noise_lowers_noise() {
    let s = noisy(SampleType::F32);
    let inner = Rect::new(8, 8, 32, 22);
    let out = run(
        &s,
        &FilterParams::ReduceNoise { strength: 10.0, preserve_details: 0.0, reduce_color_noise: 100.0, sharpen_details: 0.0, remove_jpeg_artifact: false },
    );
    for c in 0..3 {
        assert!(variance(&out, inner, c) < variance(&s, inner, c) * 0.5, "channel {c}");
    }
}

#[test]
fn smart_blur_keeps_edges_and_smooths_noise() {
    let mut s = noisy(SampleType::F32);
    s.fill_rect(Rect::new(20, 0, 40, 30), &[1.0, 1.0, 1.0, 1.0]);
    let out = run(&s, &FilterParams::SmartBlur { radius: 3.0, threshold: 60.0, quality: BlurQuality::High, mode: SmartBlurMode::Normal });
    assert!((out.pixel(20, 15)[0] - 1.0).abs() < 1e-5 && out.pixel(19, 15)[0] < 0.8, "edge kept");
    assert!(variance(&out, Rect::new(4, 4, 16, 26), 0) < variance(&s, Rect::new(4, 4, 16, 26), 0));
    let edges = run(&s, &FilterParams::SmartBlur { radius: 3.0, threshold: 60.0, quality: BlurQuality::High, mode: SmartBlurMode::EdgeOnly });
    assert_eq!(edges.pixel(20, 15)[0], 1.0);
    assert_eq!(edges.pixel(30, 15)[0], 0.0);
}

#[test]
fn shaped_blurs_spread_a_point_into_the_shape() {
    let mut s = Surface::new(fmt(SampleType::F32));
    s.fill_rect(R, &BLACK);
    s.write_pixel(20, 15, &WHITE);
    let ring = run(&s, &FilterParams::ShapeBlur { radius: 6.0, shape: BlurShape::Ring });
    assert_eq!(ring.pixel(20, 15)[0], 0.0, "ring has a hole");
    assert!(ring.pixel(25, 15)[0] > 0.0);
    let lb = run(
        &s,
        &FilterParams::LensBlur {
            radius: 6.0,
            blades: 3,
            curvature: 0.0,
            rotation: 0.0,
            depth: DepthSource::None,
            focal_distance: 0.0,
            invert_depth: false,
            brightness: 100.0,
            threshold: 200.0,
            noise: 0.0,
            distribution: Distribution::Uniform,
            monochromatic: false,
            seed: 0,
            depth_map: None,
        },
    );
    // A triangle iris pointing up: the bokeh reaches farther below the point than above it... and specular boost brightens.
    let sum: f32 = lb.read_region(R).as_chunks::<4>().0.iter().map(|p| p[0]).sum();
    assert!(sum > 1.5, "highlight bloomed: {sum}");
    assert!(lb.pixel(20, 15 + 5)[0] != lb.pixel(20, 15 - 5)[0], "triangular, not circular");
}

#[test]
fn lens_blur_depth_map_keeps_focal_plane_sharp() {
    let s = pattern(SampleType::F32, R);
    // Depth: left half 0 (in focus at focal 0), right half 1.
    let mut d = Image::new(R, 1);
    for y in 0..30 {
        for x in 20..40 {
            d.data[(y * 40 + x) as usize] = 1.0;
        }
    }
    let p = FilterParams::LensBlur {
        radius: 5.0,
        blades: 6,
        curvature: 100.0,
        rotation: 0.0,
        depth: DepthSource::LayerMask,
        focal_distance: 0.0,
        invert_depth: false,
        brightness: 0.0,
        threshold: 255.0,
        noise: 0.0,
        distribution: Distribution::Uniform,
        monochromatic: false,
        seed: 0,
        depth_map: Some(Arc::new(d)),
    };
    let out = run(&s, &p);
    assert_eq!(out.pixel(5, 10), s.pixel(5, 10), "focal plane untouched");
    assert_ne!(out.pixel(30, 10), s.pixel(30, 10), "far plane blurred");
}

#[test]
fn gallery_blurs_follow_their_geometry() {
    let s = pattern(SampleType::F32, R);
    let ts = run(&s, &FilterParams::TiltShift { blur: 6.0, center_x: 0.5, center_y: 0.5, angle: 0.0, focus: 0.1, transition: 0.2 });
    assert_eq!(ts.pixel(10, 15), s.pixel(10, 15), "band centre sharp");
    assert_ne!(ts.pixel(10, 1), s.pixel(10, 1), "far rows blurred");
    let ir = run(&s, &FilterParams::IrisBlur { pins: vec![IrisPin { blur: 6.0, radius_x: 0.3, radius_y: 0.3, ..IrisPin::default() }] });
    assert_eq!(ir.pixel(20, 15), s.pixel(20, 15));
    assert_ne!(ir.pixel(1, 1), s.pixel(1, 1));
    let sp = run(&s, &FilterParams::SpinBlur { pins: vec![SpinPin { radius_x: 0.3, radius_y: 0.3, blur_angle: 40.0, ..SpinPin::default() }] });
    assert_eq!(sp.pixel(1, 1), s.pixel(1, 1), "outside the spin ellipse");
    assert_ne!(sp.pixel(24, 15), s.pixel(24, 15));
    // Horizontal path: vertical stripes get blurred, horizontal ones survive.
    let mut stripes = Surface::new(fmt(SampleType::F32));
    for y in 0..30 {
        for x in 0..40 {
            let v = if y % 2 == 0 { 1.0 } else { 0.0 };
            stripes.write_pixel(x, y, &[v, v, v, 1.0]);
        }
    }
    let pb = run(&stripes, &FilterParams::PathBlur { paths: vec![BlurPath { points: vec![[0.0, 0.5], [1.0, 0.5]], speed: 6.0, taper: 0.0 }] });
    assert!((pb.pixel(20, 10)[0] - 1.0).abs() < 1e-5 && pb.pixel(20, 11)[0] < 1e-5);
}

#[test]
fn custom_kernel_shifts_and_offsets() {
    let s = pattern(SampleType::F32, R);
    let mut k = vec![0.0; 25];
    k[13] = 2.0; // right neighbour
    let out = run(&s, &FilterParams::Custom { kernel: k, scale: 2.0, offset: 0.0 });
    assert_eq!(out.pixel(10, 10)[0], s.pixel(11, 10)[0]);
    let mut id = vec![0.0; 25];
    id[12] = 1.0;
    let off = run(&flat(SampleType::F32, R, [0.2, 0.2, 0.2, 1.0]), &FilterParams::Custom { kernel: id, scale: 1.0, offset: 51.0 });
    assert!((off.pixel(5, 5)[0] - 0.4).abs() < 1e-5);
}

#[test]
fn hsb_filter_encodes_and_decodes() {
    let s = flat(SampleType::F32, R, [1.0, 0.0, 0.0, 1.0]);
    let hsb = run(&s, &FilterParams::HsbHsl { input: HsbModel::Rgb, output: HsbModel::Hsb });
    assert_eq!(hsb.pixel(3, 3), vec![0.0, 1.0, 1.0, 1.0]);
    let back = run(&hsb, &FilterParams::HsbHsl { input: HsbModel::Hsb, output: HsbModel::Rgb });
    assert_eq!(back.pixel(3, 3), s.pixel(3, 3));
}

#[test]
fn deinterlace_rebuilds_a_field() {
    let mut s = Surface::new(fmt(SampleType::F32));
    for y in 0..30 {
        let v = if y % 2 == 0 { 0.8 } else { 0.2 };
        s.fill_rect(Rect::new(0, y, 40, y + 1), &[v, v, v, 1.0]);
    }
    // Eliminate odd lines (offsets 0, 2…): the kept 0.2 lines fill in.
    let out = run(&s, &FilterParams::DeInterlace { eliminate_even: false, interpolate: false });
    for y in 0..30 {
        assert!((out.pixel(10, y)[0] - 0.2).abs() < 1e-6, "y={y}");
    }
    let even = run(&s, &FilterParams::DeInterlace { eliminate_even: true, interpolate: true });
    assert!((even.pixel(10, 5)[0] - 0.8).abs() < 1e-6);
}

#[test]
fn ntsc_limits_saturated_colours_only() {
    let gray = flat(SampleType::F32, R, [0.5, 0.5, 0.5, 1.0]);
    assert_eq!(run(&gray, &FilterParams::NtscColors).pixel(3, 3), gray.pixel(3, 3));
    let yellow = flat(SampleType::F32, R, [1.0, 1.0, 0.0, 1.0]);
    let y = run(&yellow, &FilterParams::NtscColors).pixel(3, 3);
    assert!(y[2] > 0.05, "yellow desaturated: {y:?}");
    let l = |p: &[f32]| 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
    assert!((l(&y) - l(&[1.0, 1.0, 0.0])).abs() < 1e-3, "luma kept");
}

#[test]
fn displace_shifts_by_map_and_shear_by_curve() {
    let s = pattern(SampleType::F32, R);
    let v = 0.5 + 3.0 / 64.0;
    let out = run(
        &s,
        &FilterParams::Displace { horizontal: 25.0, vertical: 0.0, stretch: false, undefined: UndefinedAreas::Wrap, map: Some(map_image([v, 0.5, 0.5, 1.0])) },
    );
    for x in 0..30 {
        assert_eq!(out.pixel(x, 7), s.pixel(x + 3, 7), "x={x}");
    }
    let sh = run(&s, &FilterParams::Shear { points: vec![[0.0, 0.1], [1.0, 0.1]], undefined: UndefinedAreas::Wrap });
    assert_eq!(sh.pixel(10, 4), s.pixel(8, 4));
}

#[test]
fn wind_streaks_downwind_from_edges() {
    let mut s = flat(SampleType::F32, R, BLACK);
    s.fill_rect(Rect::new(10, 0, 12, 30), &WHITE);
    let right = run(&s, &FilterParams::Wind { method: WindMethod::Blast, from_right: false, seed: 1 });
    let left_side: f32 = (0..30).map(|y| right.pixel(8, y)[0]).sum();
    let right_side: f32 = (0..30).map(|y| right.pixel(14, y)[0]).sum();
    assert!(right_side > left_side + 0.5, "{left_side} {right_side}");
}

#[test]
fn oil_paint_smooths_along_strokes() {
    let s = noisy(SampleType::F32);
    let out = run(&s, &FilterParams::OilPaint { stylization: 5.0, cleanliness: 5.0, scale: 2.0, bristle_detail: 0.0, lighting: false, angle: 0.0, shine: 0.0 });
    let inner = Rect::new(8, 8, 32, 22);
    assert!(variance(&out, inner, 0) < variance(&s, inner, 0) * 0.5);
}

#[test]
fn cancelled_filters_return_none_and_progress_never_regresses() {
    use std::sync::atomic::{AtomicU32, Ordering};
    let s = pattern(SampleType::U8, R);
    let p = FilterParams::GaussianBlur { radius: 3.0 };
    let area = output_area(&p, s.content_bounds(), R, None);
    let yes = || true;
    assert!(apply_tiled_with(&s, &p, area, R, None, 16, Some(R), &photocraft_raster::Interrupt::cancel_only(&yes)).is_none());
    // Tile workers report progress from their own threads, so values can arrive out of order;
    // consumers keep the high-water mark (as `JobCtx::progress` does with `fetch_max`).
    // Non-negative f32s order like their bits, so `fetch_max` on the bits needs no lock.
    let high = AtomicU32::new(0);
    let calls = AtomicU32::new(0);
    let no = || false;
    let progress = |f: f32| {
        assert!((0.0..=1.0).contains(&f), "progress in range: {f}");
        high.fetch_max(f.to_bits(), Ordering::Relaxed);
        calls.fetch_add(1, Ordering::Relaxed);
    };
    let out = apply_tiled_with(&s, &p, area, R, None, 16, Some(R), &photocraft_raster::Interrupt::new(&no, &progress)).unwrap();
    assert_eq!(f32::from_bits(high.load(Ordering::Relaxed)), 1.0);
    assert!(calls.load(Ordering::Relaxed) >= 1);
    assert_eq!(out.read_region(R), apply_tiled(&s, &p, area, R, None, 16, Some(R)).read_region(R));
}
