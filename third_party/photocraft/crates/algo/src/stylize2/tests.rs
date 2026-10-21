use super::*;
use crate::{FilterParams, apply_tiled};
use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_raster::Surface;

fn image(rect: Rect, n: usize, kind: usize) -> Image {
    let mut img = Image::new(rect, n);
    let mut s = 0x2545_f491u32;
    for (i, v) in img.data.iter_mut().enumerate() {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        *v = match kind {
            0 => (s % 256) as f32 / 255.0,
            1 => ((i / n) % 5) as f32 / 4.0,
            2 => 0.0,
            3 => (s % 65536) as f32 / 65535.0,
            _ => [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0, -2.0, 3.5, 0.5][s as usize % 7],
        };
    }
    img
}
fn bits(v: Vec<f32>) -> Vec<u32> {
    v.into_iter().map(f32::to_bits).collect()
}
const MODELS: [ColorMode; 8] =
    [ColorMode::Rgb, ColorMode::Grayscale, ColorMode::Cmyk, ColorMode::Lab, ColorMode::Bitmap, ColorMode::Duotone, ColorMode::Indexed, ColorMode::Multichannel];

#[test]
fn swept_blocks_match_the_pixel_search() {
    let bounds = Rect::new(-13, -9, 31, 26);
    for mode in MODELS {
        for alpha in [false, true] {
            let n = PixelFormat::new(mode, SampleType::F32, alpha).channels();
            let ctx = Ctx { bounds, mode, alpha };
            for kind in 0..5 {
                let src = image(bounds.inflate(20), n, kind);
                for (size, depth) in [(2.0, 0.0), (2.0, 255.0), (3.5, 190.0), (8.0, 64.0), (22.0, 255.0), (255.0, 100.0)] {
                    for flags in 0..8 {
                        let spec = ExtrudeSpec {
                            kind: ExtrudeType::Blocks,
                            size,
                            depth,
                            level_based: flags & 1 != 0,
                            solid_front: flags & 2 != 0,
                            mask_incomplete: flags & 4 != 0,
                            seed: 17,
                        };
                        let out = Rect::new(-17, -2, 24, 29);
                        assert_eq!(
                            bits(extrude(&src, out, &ctx, &spec)),
                            bits(extrude_direct(&src, out, &ctx, &spec)),
                            "{mode:?} alpha {alpha} kind {kind} size {size} depth {depth} flags {flags}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn streak_scatter_matches_the_pixel_search() {
    let bounds = Rect::new(-17, -9, 39, 14);
    for mode in MODELS {
        for alpha in [false, true] {
            let n = PixelFormat::new(mode, SampleType::F32, alpha).channels();
            let ctx = Ctx { bounds, mode, alpha };
            for kind in 0..5 {
                let src = image(bounds, n, kind);
                for method in [WindMethod::Wind, WindMethod::Blast, WindMethod::Stagger] {
                    for right in [false, true] {
                        for seed in [0, 1, 777, u32::MAX] {
                            let out = bounds.inflate(3);
                            assert_eq!(
                                bits(wind(&src, out, &ctx, method, right, seed)),
                                bits(wind_direct(&src, out, &ctx, method, right, seed)),
                                "{mode:?} alpha {alpha} kind {kind} {method:?} right {right} seed {seed}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn thin_and_partial_windows_match() {
    for bounds in [Rect::new(0, 0, 1, 1), Rect::new(-5, -6, -4, 31), Rect::new(3, 8, 43, 9)] {
        let ctx = Ctx { bounds, mode: ColorMode::Rgb, alpha: true };
        for src in [image(bounds, 4, 0), image(bounds.inflate(2), 4, 1)] {
            let out = bounds.inflate(2);
            for right in [false, true] {
                assert_eq!(bits(wind(&src, out, &ctx, WindMethod::Blast, right, 17)), bits(wind_direct(&src, out, &ctx, WindMethod::Blast, right, 17)));
            }
            for kind in [ExtrudeType::Blocks, ExtrudeType::Pyramids] {
                let spec = ExtrudeSpec { kind, size: 8.0, depth: 255.0, level_based: true, solid_front: false, mask_incomplete: false, seed: 17 };
                assert_eq!(bits(extrude(&src, out, &ctx, &spec)), bits(extrude_direct(&src, out, &ctx, &spec)));
            }
        }
    }
}

#[test]
fn block_bounds_cover_float_rounding_and_ties() {
    for offset in [0, -(1 << 22), (1 << 22) - 80] {
        let bounds = Rect::new(offset, offset, offset + 64, offset + 49);
        let ctx = Ctx { bounds, mode: ColorMode::Rgb, alpha: true };
        let src = image(bounds, 4, 1);
        for size in [2.0, 7.0, 19.0] {
            let spec = ExtrudeSpec { kind: ExtrudeType::Blocks, size, depth: 255.0, level_based: true, solid_front: true, mask_incomplete: false, seed: 42 };
            assert!(extrude::blocks(&src, bounds, &ctx, &spec).is_some());
            assert_eq!(bits(extrude(&src, bounds, &ctx, &spec)), bits(extrude_direct(&src, bounds, &ctx, &spec)));
        }
    }
}

#[test]
fn optional_scratch_rejects_large_and_unsupported_geometry() {
    let bounds = Rect::new(0, 0, 1, 1);
    let ctx = Ctx { bounds, mode: ColorMode::Rgb, alpha: true };
    let src = image(bounds, 4, 0);
    let spec = ExtrudeSpec { kind: ExtrudeType::Blocks, size: 2.0, depth: 255.0, level_based: false, solid_front: true, mask_incomplete: false, seed: 42 };
    assert!(extrude::blocks(&src, Rect::new(0, 0, 4096, 4096), &ctx, &spec).is_none());
    assert!(wind::streaks(&src, Rect::new(0, 0, 1_048_577, 1), &ctx, WindMethod::Blast, false, 0).is_none());
    let far = Ctx { bounds: Rect::new(10_000_000, 0, 10_000_010, 10), ..ctx };
    assert!(extrude::blocks(&src, bounds, &far, &spec).is_none());
}

#[test]
fn storage_selection_and_tile_sizes_match_original_kernels() {
    let bounds = Rect::new(-11, -7, 32, 24);
    let area = Rect::new(-9, -5, 29, 21);
    for mode in [ColorMode::Rgb, ColorMode::Grayscale, ColorMode::Cmyk, ColorMode::Lab] {
        for sample in [SampleType::U8, SampleType::U16, SampleType::F32] {
            let fmt = PixelFormat::new(mode, sample, true);
            let ctx = Ctx { bounds, mode, alpha: true };
            let img = image(bounds, fmt.channels(), 3);
            let mut s = Surface::new(fmt);
            s.write_region(bounds, &img.data);
            let mut selection = Surface::new(PixelFormat::new(ColorMode::Grayscale, SampleType::F32, false));
            let mask: Vec<f32> = (0..bounds.width() as usize * bounds.height() as usize).map(|i| (i % 5) as f32 / 4.0).collect();
            selection.write_region(bounds, &mask);
            let cases = [
                FilterParams::Extrude {
                    kind: ExtrudeType::Blocks,
                    size: 7.0,
                    depth: 170.0,
                    level_based: true,
                    solid_front: false,
                    mask_incomplete: false,
                    seed: 17,
                },
                FilterParams::Wind { method: WindMethod::Blast, from_right: true, seed: 17 },
            ];
            for p in cases {
                let halo = match p {
                    FilterParams::Extrude { .. } => 70,
                    _ => 40,
                };
                let src = Image::read_clamped(&s, area.inflate(halo), bounds);
                let mut raw = match p {
                    FilterParams::Extrude { kind, size, depth, level_based, solid_front, mask_incomplete, seed } => {
                        extrude_direct(&src, area, &ctx, &ExtrudeSpec { kind, size, depth, level_based, solid_front, mask_incomplete, seed })
                    }
                    FilterParams::Wind { method, from_right, seed } => wind_direct(&src, area, &ctx, method, from_right, seed),
                    _ => continue,
                };
                crate::mix_selection(&mut raw, area, &selection, &src, fmt.alpha);
                let mut want = s.clone();
                want.write_region(area, &raw);
                for tile in [1, 7, 17, 256] {
                    let got = apply_tiled(&s, &p, area, bounds, Some(&selection), tile, Some(bounds));
                    assert_eq!(bits(got.read_region(bounds)), bits(want.read_region(bounds)), "{mode:?} {sample:?} {} tile {tile}", p.label());
                }
            }
        }
    }
}
