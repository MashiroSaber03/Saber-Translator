use super::*;
use photocraft_color::{PixelFormat, SampleType};

fn fmt(s: SampleType) -> PixelFormat {
    PixelFormat::new(ColorMode::Rgb, s, true)
}

/// Deterministic test image over `r`.
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

const R: Rect = Rect { x0: 0, y0: 0, x1: 40, y1: 30 };

fn run(s: &Surface, p: &FilterParams) -> Surface {
    let area = output_area(p, s.content_bounds(), R, None);
    apply(s, p, area, R, None)
}

fn all_filters() -> Vec<FilterParams> {
    vec![
        FilterParams::GaussianBlur { radius: 2.5 },
        FilterParams::GaussianBlur { radius: 9.0 },
        FilterParams::BoxBlur { radius: 2.0 },
        FilterParams::BoxBlur { radius: 6.0 },
        FilterParams::MotionBlur { angle: 30.0, distance: 7.0 },
        FilterParams::RadialBlur { amount: 20.0, method: RadialMethod::Spin, quality: RadialQuality::Good, center_x: 0.5, center_y: 0.5 },
        FilterParams::RadialBlur { amount: 30.0, method: RadialMethod::Zoom, quality: RadialQuality::Good, center_x: 0.3, center_y: 0.6 },
        FilterParams::SurfaceBlur { radius: 3.0, threshold: 20.0 },
        FilterParams::UnsharpMask { amount: 150.0, radius: 1.5, threshold: 2.0 },
        FilterParams::SmartSharpen { amount: 100.0, radius: 1.0, reduce_noise: 20.0 },
        FilterParams::HighPass { radius: 3.0 },
        FilterParams::AddNoise { amount: 25.0, distribution: Distribution::Gaussian, monochromatic: false, seed: 3 },
        FilterParams::Median { radius: 2.0 },
        FilterParams::DustAndScratches { radius: 2.0, threshold: 10.0 },
        FilterParams::Minimum { radius: 1.0, preserve: Preserve::Squareness },
        FilterParams::Maximum { radius: 2.0, preserve: Preserve::Squareness },
        FilterParams::Minimum { radius: 2.5, preserve: Preserve::Roundness },
        FilterParams::Maximum { radius: 3.0, preserve: Preserve::Roundness },
        FilterParams::Offset { horizontal: 7, vertical: -3, undefined: UndefinedAreas::Wrap },
        FilterParams::Mosaic { cell_size: 6.0 },
        FilterParams::Emboss { angle: 135.0, height: 2.0, amount: 100.0 },
        FilterParams::FindEdges,
        FilterParams::Solarize,
        FilterParams::Invert,
        FilterParams::Desaturate,
        FilterParams::Twirl { angle: 90.0 },
        FilterParams::Pinch { amount: 50.0 },
        FilterParams::Spherize { amount: -60.0, mode: SpherizeMode::HorizontalOnly },
        FilterParams::Wave {
            generators: 3,
            wavelength_min: 5.0,
            wavelength_max: 20.0,
            amplitude_min: 1.0,
            amplitude_max: 4.0,
            wave_type: WaveType::Triangle,
            undefined: UndefinedAreas::Repeat,
            seed: 9,
        },
        FilterParams::Ripple { amount: 200.0, size: RippleSize::Small },
        FilterParams::PolarCoordinates { mode: PolarMode::RectangularToPolar },
        FilterParams::PolarCoordinates { mode: PolarMode::PolarToRectangular },
    ]
}

#[test]
fn radial_blur_quality_defaults_for_older_serialized_parameters() {
    let params = serde_json::json!({
        "filter": "radialBlur",
        "amount": 20.0,
        "method": "spin",
        "centerX": 0.5,
        "centerY": 0.5
    });
    let decoded: FilterParams = serde_json::from_value(params).unwrap();
    assert_eq!(decoded, FilterParams::RadialBlur { amount: 20.0, method: RadialMethod::Spin, quality: RadialQuality::Good, center_x: 0.5, center_y: 0.5 });
}

#[test]
fn every_filter_is_tile_independent() {
    let s = pattern(SampleType::U8, R);
    for p in all_filters() {
        let area = output_area(&p, s.content_bounds(), R, None);
        let a = apply_tiled(&s, &p, area, R, None, 256, None);
        let b = apply_tiled(&s, &p, area, R, None, 7, None);
        let big = area.union(&R);
        assert!(a.to_interleaved(big) == b.to_interleaved(big), "{} differs between tile sizes", p.label());
    }
}

#[test]
fn every_filter_changes_something_and_respects_zero_selection() {
    let s = pattern(SampleType::U16, R);
    let none = Surface::new(PixelFormat::GRAY8);
    for p in all_filters() {
        let out = run(&s, &p);
        assert!(out != s, "{} did nothing", p.label());
        let area = output_area(&p, s.content_bounds(), R, None);
        let masked = apply(&s, &p, area, R, Some(&none));
        assert_eq!(masked.read_region(area), s.read_region(area), "{} ignored an empty selection", p.label());
    }
}

#[test]
fn half_selection_mixes_halfway() {
    let s = flat(SampleType::F32, R, [0.2, 0.4, 0.6, 1.0]);
    let sel = Surface::with_default(PixelFormat::new(ColorMode::Grayscale, SampleType::F32, false), &[0.5]);
    let out = apply(&s, &FilterParams::Invert, R, R, Some(&sel));
    let p = out.pixel(5, 5);
    assert!((p[0] - 0.5).abs() < 1e-5 && (p[2] - 0.5).abs() < 1e-5, "{p:?}");
}

#[test]
fn gaussian_identity_at_tiny_radius() {
    let s = pattern(SampleType::U8, R);
    let out = run(&s, &FilterParams::GaussianBlur { radius: 0.01 });
    assert_eq!(out.to_interleaved(R), s.to_interleaved(R));
}

#[test]
fn blur_preserves_flat_colour_and_mass() {
    let s = flat(SampleType::F32, R, [0.3, 0.6, 0.9, 1.0]);
    let out = run(&s, &FilterParams::GaussianBlur { radius: 3.0 });
    let p = out.pixel(20, 15);
    assert!((p[0] - 0.3).abs() < 1e-5 && (p[3] - 1.0).abs() < 1e-5);
    // Transparency mass (sum of alpha) is preserved when blurring into the margin.
    let small = flat(SampleType::F32, Rect::new(10, 10, 20, 20), [1.0, 0.0, 0.0, 1.0]);
    for p in [FilterParams::GaussianBlur { radius: 2.0 }, FilterParams::BoxBlur { radius: 3.0 }] {
        let out = run(&small, &p);
        let big = Rect::new(-10, -10, 40, 40);
        let sum: f32 = out.read_region(big).as_chunks::<4>().0.iter().map(|px| px[3]).sum();
        assert!((sum - 100.0).abs() < 0.01, "{}: {sum}", p.label());
        // No colour bleeding from transparent pixels.
        let edge = out.pixel(9, 15);
        assert!(edge[3] > 0.0 && (edge[0] - 1.0).abs() < 1e-4, "{edge:?}");
    }
}

#[test]
fn blur_keeps_16bit_precision() {
    // A shallow 16-bit ramp (1/65535 per pixel) must stay a ramp after blurring.
    let mut s = Surface::new(fmt(SampleType::U16));
    let r = Rect::new(0, 0, 200, 1);
    let v: Vec<f32> = (0..200).flat_map(|x| [x as f32 / 65535.0, 0.0, 0.0, 1.0]).collect();
    s.write_region(r, &v);
    let out = apply(&s, &FilterParams::GaussianBlur { radius: 2.0 }, r, r, None);
    for x in 20..180 {
        let got = (out.pixel(x, 0)[0] * 65535.0).round() as i32;
        assert_eq!(got, x, "x={x}");
    }
}

#[test]
fn float_surfaces_keep_hdr_values() {
    let s = flat(SampleType::F32, R, [4.0, 2.0, 0.5, 1.0]);
    let out = run(&s, &FilterParams::GaussianBlur { radius: 2.0 });
    assert!((out.pixel(20, 15)[0] - 4.0).abs() < 1e-4);
}

#[test]
fn gaussian_sigma_matches_radius() {
    // A single bright pixel blurs into a Gaussian with sigma = radius.
    let mut s = Surface::new(fmt(SampleType::F32));
    s.write_pixel(20, 15, &[1.0, 1.0, 1.0, 1.0]);
    let out = apply(&s, &FilterParams::GaussianBlur { radius: 2.0 }, R, R, None);
    let a0 = out.pixel(20, 15)[3];
    let a2 = out.pixel(22, 15)[3];
    // exp(-d²/2σ²) at d = σ = 2 → e^-0.5
    assert!((a2 / a0 - (-0.5f32).exp()).abs() < 1e-3, "{}", a2 / a0);
}

#[test]
fn unsharp_threshold_and_contrast() {
    let s = pattern(SampleType::F32, R);
    let none = run(&s, &FilterParams::UnsharpMask { amount: 200.0, radius: 2.0, threshold: 255.0 });
    assert_eq!(none.read_region(R), s.read_region(R), "threshold 255 disables sharpening");
    let sharp = run(&s, &FilterParams::UnsharpMask { amount: 200.0, radius: 2.0, threshold: 0.0 });
    let var = |surf: &Surface| {
        let v = surf.read_region(Rect::new(5, 5, 35, 25));
        let reds: Vec<f32> = v.as_chunks::<4>().0.iter().map(|p| p[0]).collect();
        let m = reds.iter().sum::<f32>() / reds.len() as f32;
        reds.iter().map(|r| (r - m).powi(2)).sum::<f32>()
    };
    assert!(var(&sharp) > var(&s));
}

#[test]
fn high_pass_of_flat_is_mid_gray() {
    let s = flat(SampleType::F32, R, [0.9, 0.1, 0.4, 1.0]);
    let p = run(&s, &FilterParams::HighPass { radius: 4.0 }).pixel(20, 15);
    assert!((p[0] - 0.5).abs() < 1e-5 && (p[1] - 0.5).abs() < 1e-5);
}

#[test]
fn noise_is_deterministic_monochrome_and_centred() {
    let s = flat(SampleType::F32, R, [0.5, 0.5, 0.5, 1.0]);
    let p = FilterParams::AddNoise { amount: 50.0, distribution: Distribution::Uniform, monochromatic: true, seed: 1 };
    let a = run(&s, &p);
    assert_eq!(a, run(&s, &p));
    let v = a.read_region(R);
    let mut mean = 0.0;
    for px in v.as_chunks::<4>().0 {
        assert_eq!(px[0], px[1]);
        assert_eq!(px[1], px[2]);
        assert!((px[0] - 0.5).abs() <= 0.25 + 1e-6);
        mean += px[0];
    }
    mean /= (v.len() / 4) as f32;
    assert!((mean - 0.5).abs() < 0.03, "{mean}");
    let other_seed = run(&s, &FilterParams::AddNoise { amount: 50.0, distribution: Distribution::Uniform, monochromatic: true, seed: 2 });
    assert_ne!(a, other_seed);
}

#[test]
fn median_and_dust_remove_a_speck() {
    let mut s = flat(SampleType::F32, R, [0.2, 0.2, 0.2, 1.0]);
    s.write_pixel(10, 10, &[1.0, 1.0, 1.0, 1.0]);
    let m = run(&s, &FilterParams::Median { radius: 1.0 });
    assert!((m.pixel(10, 10)[0] - 0.2).abs() < 1e-6);
    let keep = run(&s, &FilterParams::DustAndScratches { radius: 1.0, threshold: 255.0 });
    assert!((keep.pixel(10, 10)[0] - 1.0).abs() < 1e-6);
    let remove = run(&s, &FilterParams::DustAndScratches { radius: 1.0, threshold: 50.0 });
    assert!((remove.pixel(10, 10)[0] - 0.2).abs() < 1e-6);
}

#[test]
fn minimum_maximum_spread() {
    let mut s = flat(SampleType::F32, R, [0.5, 0.5, 0.5, 1.0]);
    s.write_pixel(10, 10, &[1.0, 1.0, 1.0, 1.0]);
    let mx = run(&s, &FilterParams::Maximum { radius: 1.0, preserve: Preserve::Squareness });
    assert!((mx.pixel(11, 11)[0] - 1.0).abs() < 1e-6);
    assert!((mx.pixel(12, 10)[0] - 0.5).abs() < 1e-6);
    let mn = run(&s, &FilterParams::Minimum { radius: 1.0, preserve: Preserve::Squareness });
    assert!((mn.pixel(10, 10)[0] - 0.5).abs() < 1e-6);
}

#[test]
fn maximum_roundness_grows_a_point_into_a_disc() {
    // One white pixel on black: Maximum spreads it over the structuring element itself.
    for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
        let mut s = flat(depth, R, [0.0, 0.0, 0.0, 1.0]);
        s.write_pixel(20, 15, &[1.0, 1.0, 1.0, 1.0]);
        for radius in [1.0f32, 2.0, 3.5, 6.0] {
            let round = run(&s, &FilterParams::Maximum { radius, preserve: Preserve::Roundness });
            let square = run(&s, &FilterParams::Maximum { radius, preserve: Preserve::Squareness });
            let rs = radius.round() as i32;
            for y in R.y0..R.y1 {
                for x in R.x0..R.x1 {
                    let (dx, dy) = (x - 20, y - 15);
                    let in_disc = (dx * dx + dy * dy) as f32 <= radius * radius;
                    assert_eq!(round.pixel(x, y)[0] > 0.5, in_disc, "{depth:?} r {radius} at {dx},{dy}");
                    assert_eq!(square.pixel(x, y)[0] > 0.5, dx.abs() <= rs && dy.abs() <= rs, "{depth:?} r {radius} at {dx},{dy}");
                }
            }
        }
    }
    // Radius 1: the four neighbours, not the diagonals (a 3×3 square would take them).
    let mut s = flat(SampleType::U8, R, [0.0, 0.0, 0.0, 1.0]);
    s.write_pixel(20, 15, &[1.0, 1.0, 1.0, 1.0]);
    let r1 = run(&s, &FilterParams::Maximum { radius: 1.0, preserve: Preserve::Roundness });
    assert_eq!(r1.pixel(21, 15)[0], 1.0);
    assert_eq!(r1.pixel(21, 16)[0], 0.0);
}

#[test]
fn minimum_maximum_roundness_match_brute_force() {
    // Exhaustive disc search on a textured image, both operators, a few radii (incl. fractional).
    let s = pattern(SampleType::F32, R);
    let img = s.read_region(R);
    let at = |x: i32, y: i32, c: usize| -> Option<f32> { R.contains(x, y).then(|| img[((y * 40 + x) * 4) as usize + c]) };
    for radius in [1.0f32, 1.5, 2.9, 4.0] {
        let ri = radius.floor() as i32;
        for max in [false, true] {
            let p = if max {
                FilterParams::Maximum { radius, preserve: Preserve::Roundness }
            } else {
                FilterParams::Minimum { radius, preserve: Preserve::Roundness }
            };
            let got = run(&s, &p);
            for y in R.y0 + ri..R.y1 - ri {
                for x in R.x0 + ri..R.x1 - ri {
                    for c in 0..3 {
                        let mut want = if max { f32::MIN } else { f32::MAX };
                        for dy in -ri..=ri {
                            for dx in -ri..=ri {
                                if (dx * dx + dy * dy) as f32 <= radius * radius
                                    && let Some(v) = at(x + dx, y + dy, c)
                                {
                                    want = if max { want.max(v) } else { want.min(v) };
                                }
                            }
                        }
                        assert!((got.pixel(x, y)[c] - want).abs() < 1e-6, "r {radius} max {max} at {x},{y} c{c}");
                    }
                }
            }
        }
    }
}

#[test]
fn running_extreme_windows() {
    let v = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let mut out = Vec::new();
    for k in 1..=v.len() {
        for max in [false, true] {
            crate::other::running_extreme(&v, k, max, &mut out);
            let want: Vec<f32> =
                v.windows(k).map(|w| w.iter().copied().fold(if max { f32::MIN } else { f32::MAX }, |a, b| if max { a.max(b) } else { a.min(b) })).collect();
            assert_eq!(out, want, "k {k} max {max}");
        }
    }
}

#[test]
fn minimum_maximum_old_params_default_to_squareness() {
    let p: FilterParams = serde_json::from_str(r#"{"filter":"minimum","radius":3}"#).unwrap();
    assert_eq!(p, FilterParams::Minimum { radius: 3.0, preserve: Preserve::Squareness });
    let p: FilterParams = serde_json::from_str(r#"{"filter":"maximum","radius":2,"preserve":"roundness"}"#).unwrap();
    assert_eq!(p, FilterParams::Maximum { radius: 2.0, preserve: Preserve::Roundness });
}

#[test]
fn offset_wrap_repeat_transparent() {
    let s = pattern(SampleType::U8, R);
    let full = run(&s, &FilterParams::Offset { horizontal: 40, vertical: 30, undefined: UndefinedAreas::Wrap });
    assert_eq!(full.to_interleaved(R), s.to_interleaved(R));
    let t = run(&s, &FilterParams::Offset { horizontal: 5, vertical: 0, undefined: UndefinedAreas::Transparent });
    assert_eq!(t.pixel(2, 3)[3], 0.0);
    assert_eq!(t.pixel(15, 3), s.pixel(10, 3));
    let rp = run(&s, &FilterParams::Offset { horizontal: 5, vertical: 0, undefined: UndefinedAreas::Repeat });
    assert_eq!(rp.pixel(2, 3), s.pixel(0, 3));
}

#[test]
fn mosaic_cells_are_uniform() {
    let s = pattern(SampleType::F32, R);
    let m = run(&s, &FilterParams::Mosaic { cell_size: 5.0 });
    assert_eq!(m.pixel(0, 0), m.pixel(4, 4));
    assert_ne!(m.pixel(4, 4), m.pixel(5, 4));
}

#[test]
fn stylize_basics() {
    let s = flat(SampleType::F32, R, [0.8, 0.3, 0.1, 1.0]);
    let e = run(&s, &FilterParams::Emboss { angle: 45.0, height: 3.0, amount: 100.0 }).pixel(20, 15);
    assert!((e[0] - 0.5).abs() < 1e-5 && (e[1] - 0.5).abs() < 1e-5, "{e:?}");
    let f = run(&s, &FilterParams::FindEdges).pixel(20, 15);
    assert!((f[0] - 1.0).abs() < 1e-6);
    let so = run(&s, &FilterParams::Solarize).pixel(3, 3);
    assert!((so[0] - 0.2).abs() < 1e-6 && (so[1] - 0.3).abs() < 1e-6);
    let d = run(&s, &FilterParams::Desaturate).pixel(3, 3);
    assert!((d[0] - d[1]).abs() < 1e-6 && (d[1] - d[2]).abs() < 1e-6);
}

/// CMYK samples are ink (1 = full ink): Find Edges leaves flat areas white (no ink, not 100% ink)
/// and Solarize inverts the channels below 50% ink, as Photoshop does (#4222). Alpha is untouched.
#[test]
fn find_edges_and_solarize_treat_cmyk_as_ink() {
    for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
        let f = PixelFormat::new(ColorMode::Cmyk, depth, true);
        let mut s = Surface::new(f);
        s.fill_rect(R, &[0.1, 0.1, 0.1, 0.1, 1.0]);
        let e = run(&s, &FilterParams::FindEdges).pixel(20, 15);
        assert!(e[..4].iter().all(|v| v.abs() < 1e-6) && (e[4] - 1.0).abs() < 1e-6, "{depth:?} flat CMYK -> white: {e:?}");
        let mut s = Surface::new(f);
        s.fill_rect(R, &[0.1, 0.2, 0.8, 0.9, 0.6]);
        let so = run(&s, &FilterParams::Solarize).pixel(3, 3);
        let want = [0.9, 0.8, 0.8, 0.9, 0.6];
        assert!(so.iter().zip(want).all(|(a, b)| (a - b).abs() < 0.01), "{depth:?} solarize CMYK: {so:?}");
    }
    // RGB keeps the light-polarity behaviour: flat -> white, channels above 50% invert.
    let s = flat(SampleType::U8, R, [0.1, 0.2, 0.8, 1.0]);
    let e = run(&s, &FilterParams::FindEdges).pixel(20, 15);
    assert!(e[..3].iter().all(|v| (v - 1.0).abs() < 1e-6), "{e:?}");
    let so = run(&s, &FilterParams::Solarize).pixel(3, 3);
    assert!(so.iter().zip([0.1, 0.2, 0.2, 1.0]).all(|(a, b)| (a - b).abs() < 0.01), "{so:?}");
}

#[test]
fn distortions_are_identity_at_zero() {
    let s = pattern(SampleType::U8, R);
    for p in [
        FilterParams::Twirl { angle: 0.0 },
        FilterParams::Pinch { amount: 0.0 },
        FilterParams::Spherize { amount: 0.0, mode: SpherizeMode::Normal },
        FilterParams::Ripple { amount: 0.0, size: RippleSize::Large },
    ] {
        let out = run(&s, &p);
        assert_eq!(out.to_interleaved(R), s.to_interleaved(R), "{}", p.label());
    }
}

#[test]
fn twirl_keeps_centre_and_outside() {
    let s = pattern(SampleType::F32, Rect::new(0, 0, 40, 40));
    let b = Rect::new(0, 0, 40, 40);
    let out = apply(&s, &FilterParams::Twirl { angle: 180.0 }, b, b, None);
    assert_eq!(out.pixel(0, 0), s.pixel(0, 0), "corner is outside the twirl circle");
    assert_ne!(out.pixel(20, 12), s.pixel(20, 12));
}

#[test]
fn motion_blur_spreads_along_angle_only() {
    let mut s = flat(SampleType::F32, R, [0.0, 0.0, 0.0, 1.0]);
    for y in 0..30 {
        s.write_pixel(20, y, &[1.0, 1.0, 1.0, 1.0]);
    }
    let h = run(&s, &FilterParams::MotionBlur { angle: 0.0, distance: 6.0 });
    assert!(h.pixel(22, 15)[0] > 0.05);
    let v = run(&s, &FilterParams::MotionBlur { angle: 90.0, distance: 6.0 });
    assert!(v.pixel(22, 15)[0] < 1e-4);
}

#[test]
fn params_serde_roundtrip() {
    for p in all_filters() {
        let j = serde_json::to_string(&p).unwrap();
        let back: FilterParams = serde_json::from_str(&j).unwrap();
        assert_eq!(back, p, "{j}");
    }
    let p: FilterParams = serde_json::from_str(r#"{"filter":"unsharpMask","amount":50,"radius":1,"threshold":0}"#).unwrap();
    assert_eq!(p, FilterParams::UnsharpMask { amount: 50.0, radius: 1.0, threshold: 0.0 });
}

#[test]
fn halos_and_output_area() {
    assert_eq!(FilterParams::GaussianBlur { radius: 2.0 }.halo(), Halo::Radius(7));
    assert_eq!(FilterParams::Twirl { angle: 1.0 }.halo(), Halo::Bounds);
    let content = Rect::new(10, 10, 20, 20);
    assert_eq!(output_area(&FilterParams::Invert, content, R, None), content.inflate(0));
    assert_eq!(output_area(&FilterParams::Twirl { angle: 1.0 }, content, R, None), R);
    assert_eq!(output_area(&FilterParams::BoxBlur { radius: 1.0 }, content, R, Some(Rect::new(0, 0, 12, 12))), Rect::new(8, 8, 12, 12));
    assert!(output_area(&FilterParams::Invert, Rect::EMPTY, R, None).is_empty());
    // Fibers renders without reading the layer: an empty layer still fills the bounds (#5241).
    let fibers = FilterParams::Fibers { variance: 16.0, strength: 4.0, seed: 0, foreground: [0.0; 4], background: [1.0; 4] };
    assert_eq!(output_area(&fibers, Rect::EMPTY, R, None), R);
    assert_eq!(output_area(&fibers, content, R, Some(Rect::new(0, 0, 12, 12))), Rect::new(0, 0, 12, 12));
}

#[test]
fn other_colour_modes_work() {
    for mode in [ColorMode::Grayscale, ColorMode::Cmyk, ColorMode::Lab] {
        let f = PixelFormat::new(mode, SampleType::U8, true);
        let mut s = Surface::new(f);
        let px: Vec<f32> = vec![0.4; f.channels() - 1].into_iter().chain([1.0]).collect();
        s.fill_rect(R, &px);
        for p in [
            FilterParams::GaussianBlur { radius: 1.0 },
            FilterParams::Desaturate,
            FilterParams::Emboss { angle: 0.0, height: 1.0, amount: 100.0 },
            FilterParams::Twirl { angle: 50.0 },
        ] {
            let out = run(&s, &p);
            assert_eq!(out.format(), f);
        }
    }
}

#[test]
fn layer_filters_repeat_the_canvas_edge() {
    // A fully opaque layer covering the canvas stays opaque at the edge (no transparent fade-in)
    // and doesn't grow past the canvas; without an extent, samples beyond the data are transparent.
    for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
        let canvas = Rect::new(0, 0, 40, 30);
        let s = flat(depth, canvas, [0.2, 0.6, 0.4, 1.0]);
        let p = FilterParams::GaussianBlur { radius: 6.0 };
        let area = output_area(&p, s.content_bounds(), canvas, None);
        let clamped = apply_in(&s, &p, area, canvas, None, canvas);
        assert_eq!(clamped.content_bounds(), canvas, "{depth:?}: grew past the canvas");
        for (x, y) in [(0, 0), (39, 0), (0, 29), (39, 29), (20, 0)] {
            let px = clamped.pixel(x, y);
            assert!((px[3] - 1.0).abs() < 1e-3, "{depth:?}: alpha faded at ({x},{y}): {}", px[3]);
            assert!((px[1] - 0.6).abs() < 2e-3, "{depth:?}: colour changed at ({x},{y})");
        }
        let faded = apply(&s, &p, area, canvas, None);
        assert!(faded.pixel(0, 0)[3] < 0.9, "{depth:?}: unclamped path should read transparency");
    }
}

/// Photoshop 27.11's Polar Coordinates turn counter-clockwise from 12 o'clock: just left of the
/// top centre, Rectangular to Polar samples the source's left edge (clockwise would take its right).
#[test]
fn polar_coordinates_turn_counter_clockwise() {
    let r = Rect::new(0, 0, 64, 64);
    let mut s = flat(SampleType::U8, r, [0.0, 0.0, 1.0, 1.0]);
    s.fill_rect(Rect::new(0, 0, 32, 64), &[1.0, 0.0, 0.0, 1.0]);
    let p = FilterParams::PolarCoordinates { mode: PolarMode::RectangularToPolar };
    let out = apply(&s, &p, r, r, None);
    let left_of_top = out.rgba(28, 12);
    assert!(left_of_top[0] > 0.9 && left_of_top[2] < 0.1, "{left_of_top:?}");
    // The corners outside the circle repeat the source's edge instead of going transparent.
    assert!(out.rgba(0, 0)[3] > 0.99, "{:?}", out.rgba(0, 0));
}

/// `mix_selection` before #4533, one selection lookup per pixel: the oracle for the per-tile read.
fn mix_selection_reference(data: &mut [f32], t: Rect, sel: &Surface, orig: &crate::image::Image, alpha: bool) {
    let w = t.width() as usize;
    for (i, px) in data.chunks_exact_mut(orig.ch.max(1)).enumerate() {
        let (x, y) = (t.x0 + (i % w) as i32, t.y0 + (i / w) as i32);
        let k = sel.pixel(x, y)[0].clamp(0.0, 1.0);
        if k >= 1.0 {
            continue;
        }
        if k <= 0.0 {
            for (c, v) in px.iter_mut().enumerate() {
                *v = orig.get(x, y, c);
            }
            continue;
        }
        if alpha {
            let a = px.len() - 1;
            let original_alpha = orig.get(x, y, a);
            let filtered_alpha = px[a];
            if original_alpha != filtered_alpha {
                let mixed_alpha = original_alpha + (filtered_alpha - original_alpha) * k;
                let original_weight = original_alpha * (1.0 - k);
                let filtered_weight = filtered_alpha * k;
                for (c, v) in px[..a].iter_mut().enumerate() {
                    *v = if mixed_alpha > 0.0 { (orig.get(x, y, c) * original_weight + *v * filtered_weight) / mixed_alpha } else { 0.0 };
                }
                px[a] = mixed_alpha;
                continue;
            }
        }
        for (c, v) in px.iter_mut().enumerate() {
            let o = orig.get(x, y, c);
            *v = o + (*v - o) * k;
        }
    }
}

/// Reading the selection once per tile gives exactly the per-pixel result (#4533): a soft
/// selection with fully selected, unselected and partial pixels, with and without alpha, on a
/// tile that crosses the selection's tile boundaries.
#[test]
fn mix_selection_reads_the_selection_once_per_tile() {
    let noise = |seed: u32, n: usize| -> Vec<f32> {
        (0..n as u32)
            .map(|i| {
                let mut h = seed.wrapping_mul(0x9e37_79b9) ^ i.wrapping_mul(0x85eb_ca6b);
                h ^= h >> 13;
                h = h.wrapping_mul(0xc2b2_ae35);
                (h >> 16) as f32 / 65535.0
            })
            .collect()
    };
    let t = Rect::new(200, 180, 330, 300);
    let n = (t.width() * t.height()) as usize;
    let mut sel = Surface::new(PixelFormat::GRAY8);
    sel.fill_rect(Rect::new(150, 150, 260, 250), &[1.0]);
    let soft = Rect::new(260, 150, 330, 250);
    sel.write_region(soft, &noise(5, (soft.width() * soft.height()) as usize));
    for (alpha, fmt) in [(true, PixelFormat::RGBA8), (false, PixelFormat { alpha: false, ..PixelFormat::RGBA8 })] {
        let ch = fmt.channels();
        let orig = crate::image::Image { rect: t, ch, data: noise(1, n * ch) };
        let filtered = noise(2, n * ch);
        let (mut a, mut b) = (filtered.clone(), filtered);
        mix_selection(&mut a, t, &sel, &orig, alpha);
        mix_selection_reference(&mut b, t, &sel, &orig, alpha);
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "alpha {alpha}: results differ");
    }
}
