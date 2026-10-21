use super::*;
use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_raster::Surface;

fn fixture(rect: Rect, ctx: &Ctx, pattern: u32) -> Image {
    let n = PixelFormat::new(ctx.mode, SampleType::F32, ctx.alpha).channels();
    let mut image = Image::new(rect, n);
    for (i, pixel) in image.data.chunks_exact_mut(n).enumerate() {
        let (x, y) = xy(rect, i);
        let seed = (x as u32).wrapping_mul(0x9e37_79b9) ^ (y as u32).wrapping_mul(0x85eb_ca6b);
        for (c, sample) in pixel.iter_mut().enumerate() {
            *sample = match pattern {
                0 => 0.5,
                1 => {
                    if x + y > -4 {
                        0.9
                    } else {
                        0.1
                    }
                }
                2 => (seed.rotate_left(c as u32 * 7) % 65536) as f32 / 65535.0,
                3 => (seed.rotate_left(c as u32 * 7) % 65536) as f32 / 16384.0 - 1.0,
                _ => match seed.wrapping_add(c as u32) % 7 {
                    0 => f32::NAN,
                    1 => f32::INFINITY,
                    2 => f32::NEG_INFINITY,
                    _ => 0.5,
                },
            };
        }
        if ctx.alpha && seed.is_multiple_of(5) {
            pixel[n - 1] = 0.0;
        }
    }
    image
}

fn assert_bits(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "sample {i}: {a} != {b}");
    }
}

#[test]
fn facet_reuse_matches_scalar_in_every_model() {
    let out = Rect::new(-17, -9, 20, 18);
    for mode in [
        ColorMode::Rgb,
        ColorMode::Grayscale,
        ColorMode::Cmyk,
        ColorMode::Lab,
        ColorMode::Bitmap,
        ColorMode::Duotone,
        ColorMode::Indexed,
        ColorMode::Multichannel,
    ] {
        for alpha in [false, true] {
            let ctx = Ctx { bounds: out, mode, alpha };
            for pattern in 0..5 {
                let src = fixture(out.inflate(3), &ctx, pattern);
                assert_bits(&facet(&src, out, &ctx), &reference::facet(&src, out, &ctx));
            }
        }
    }
}

#[test]
fn facet_partial_halo_and_thin_regions_match_scalar() {
    for out in [Rect::new(-3, 7, -2, 8), Rect::new(-5, 0, 30, 1), Rect::new(9, -8, 10, 29)] {
        let ctx = Ctx { bounds: out, mode: ColorMode::Rgb, alpha: true };
        for halo in [0, 1, 2, 3] {
            let src = fixture(out.inflate(halo), &ctx, 2);
            assert_bits(&facet(&src, out, &ctx), &reference::facet(&src, out, &ctx));
        }
    }
}

#[test]
fn halftone_cache_matches_scalar_screen_angles_radii_and_models() {
    let out = Rect::new(-17, -9, 20, 18);
    for mode in [
        ColorMode::Rgb,
        ColorMode::Grayscale,
        ColorMode::Cmyk,
        ColorMode::Lab,
        ColorMode::Bitmap,
        ColorMode::Duotone,
        ColorMode::Indexed,
        ColorMode::Multichannel,
    ] {
        for alpha in [false, true] {
            let ctx = Ctx { bounds: Rect::new(-23, -15, 29, 31), mode, alpha };
            for pattern in 0..5 {
                let src = fixture(ctx.bounds, &ctx, pattern);
                for radius in [1.0, 2.0, 3.75, 8.0, 64.0] {
                    for angles in [[108.0, 162.0, 90.0, 45.0], [0.0, -17.5, 359.99, 720.0]] {
                        assert_bits(&color_halftone(&src, out, &ctx, radius, angles), &reference::halftone(&src, out, &ctx, radius, angles));
                    }
                }
            }
        }
    }
}

#[test]
fn halftone_unusual_coordinates_and_params_keep_direct_semantics() {
    for out in [Rect::new(16_777_211, -3, 16_777_220, 4), Rect::new(-3, 4, -2, 5)] {
        let ctx = Ctx { bounds: out, mode: ColorMode::Rgb, alpha: true };
        let src = fixture(out.inflate(2), &ctx, 2);
        for radius in [f32::NAN, f32::INFINITY, -10.0, 1.0] {
            for angles in [[f32::NAN, 0.0, f32::INFINITY, 45.0], [0.0; 4]] {
                assert_bits(&color_halftone(&src, out, &ctx, radius, angles), &reference::halftone(&src, out, &ctx, radius, angles));
            }
        }
    }
    assert!(screen::Screen::new(Rect::new(0, 0, 100_000, 100_000), (0.0, 0.0), (0.0, 1.0), 1.0).is_none());
    assert!(screen::Screen::new(Rect::new(8_000_000, 0, 8_000_010, 10), (0.0, 0.0), (0.0, 1.0), 1.0).is_none());
    let mut screen = screen::Screen::new(Rect::new(0, 0, 1, 1), (0.0, 0.0), (0.0, 1.0), 1.0).unwrap();
    assert_eq!(screen.radius(-100.0, 0.0, || 3.0), 3.0);
    assert_eq!(screen.radius(f32::NAN, 0.0, || 4.0), 4.0);
    assert_eq!(screen.radius(0.0, 0.0, || 5.0), 5.0);
    assert_eq!(screen.radius(0.0, 0.0, || panic!("cached dot recomputed")), 5.0);
}

#[test]
fn tiled_filters_match_scalar_at_all_depths_with_soft_selection() {
    use crate::{FilterParams, apply_tiled};
    let bounds = Rect::new(-17, -9, 26, 24);
    for mode in [ColorMode::Rgb, ColorMode::Grayscale, ColorMode::Cmyk, ColorMode::Lab] {
        for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
            for alpha in [false, true] {
                let ctx = Ctx { bounds, mode, alpha };
                let src = fixture(bounds, &ctx, 2);
                let mut surface = Surface::new(PixelFormat::new(mode, depth, alpha));
                surface.write_region(bounds, &src.data);
                let mut selection = Surface::new(PixelFormat::new(ColorMode::Grayscale, SampleType::U8, false));
                let coverage: Vec<_> = (0..bounds.width() * bounds.height()).map(|i| (i % 7) as f32 / 6.0).collect();
                selection.write_region(bounds, &coverage);
                for params in [FilterParams::Facet, FilterParams::ColorHalftone { max_radius: 3.75, angles: [108.0, 162.0, 90.0, 45.0] }] {
                    for extent in [None, Some(bounds)] {
                        assert!(matches!(params.halo(), crate::Halo::Radius(_)));
                        let halo = match params.halo() {
                            crate::Halo::Radius(r) => r,
                            _ => 0,
                        };
                        let image = match extent {
                            Some(e) => Image::read_clamped(&surface, bounds.inflate(halo), e),
                            None => Image::read(&surface, bounds.inflate(halo)),
                        };
                        let mut expected_data = match params {
                            FilterParams::Facet => reference::facet(&image, bounds, &ctx),
                            FilterParams::ColorHalftone { max_radius, angles } => reference::halftone(&image, bounds, &ctx, max_radius, angles),
                            _ => Vec::new(),
                        };
                        crate::mix_selection(&mut expected_data, bounds, &selection, &image, ctx.alpha);
                        let mut expected = surface.clone();
                        expected.write_region(bounds, &expected_data);
                        for tile in [1, 7, 17, 256] {
                            let got = apply_tiled(&surface, &params, bounds, bounds, Some(&selection), tile, extent);
                            assert_bits(&got.read_region(bounds), &expected.read_region(bounds));
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn extended_channels_and_large_screen_offsets_match_scalar() {
    let out = Rect::new(16_777_211, -3, 16_777_220, 4);
    let ctx = Ctx { bounds: Rect::new(0, -10, out.x1 + 10, 10), mode: ColorMode::Multichannel, alpha: true };
    for n in 5..=MAXC {
        let mut src = Image::new(out.inflate(3), n);
        for (i, sample) in src.data.iter_mut().enumerate() {
            *sample = (i * 73 % 257) as f32 / 256.0;
        }
        assert_bits(&facet(&src, out, &ctx), &reference::facet(&src, out, &ctx));
        assert_bits(&color_halftone(&src, out, &ctx, 1.0, [0.0; 4]), &reference::halftone(&src, out, &ctx, 1.0, [0.0; 4]));
    }
}

#[test]
fn facet_scratch_rejects_large_or_incomplete_storage() {
    let out = Rect::new(0, 0, 10_000, 10_000);
    let ctx = Ctx { bounds: out, mode: ColorMode::Rgb, alpha: true };
    let src = Image { rect: out.inflate(3), ch: 4, data: Vec::new() };
    assert!(facet::run(&src, out, &ctx).is_none());
    let out = Rect::new(0, 0, 2, 2);
    let src = Image { rect: out.inflate(3), ch: 4, data: Vec::new() };
    assert!(facet::run(&src, out, &ctx).is_none());
}
