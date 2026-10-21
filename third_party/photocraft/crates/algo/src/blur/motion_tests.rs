use super::*;
use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_raster::Surface;

fn assert_samples_equal(actual: &[f32], expected: &[f32], case: &str) {
    assert_eq!(actual.len(), expected.len(), "{case}");
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()), "{case}, sample {i}: {a:?} != {b:?}");
    }
}

fn patterned_image(rect: Rect, ch: usize, alpha: bool) -> Image {
    let mut data = Vec::new();
    for y in 0..rect.height() as usize {
        for x in 0..rect.width() as usize {
            for c in 0..ch {
                let k = x * 37 + y * 13 + c * 29;
                let value = if alpha && c + 1 == ch { [0.0, 1.0, 0.125, 0.875][k % 4] } else { (k % 256) as f32 / 255.0 };
                data.push(value);
            }
        }
    }
    Image { rect, ch, data }
}

#[test]
fn motion_identity_preserves_signed_zero_and_transparent_colours() {
    let rect = Rect::new(-1, -1, 2, 2);
    let src = Image { rect, ch: 4, data: [-0.0, 8.0, -2.0, 0.0].repeat(9) };
    let ctx = Ctx { bounds: rect, mode: ColorMode::Rgb, alpha: true };
    for distance in [-0.49, -0.0, 0.0, 0.49] {
        assert_samples_equal(&motion(&src, rect, &ctx, 17.0, distance), &src.data, &format!("distance={distance}"));
    }
    let blurred = motion(&src, rect, &ctx, 31.0, 2.0);
    assert!(blurred.iter().all(|value| *value == 0.0));
}

#[test]
fn motion_default_kernel_retains_current_upstream_rows() {
    // PR 902 deliberately merges bilinear taps and changes summation order.
    // The default kernel follows that implementation at every distance.
    for (x0, y0) in [(0, 0), (-17, 31), (16_000, -16_000), (1 << 20, -(1 << 20))] {
        let rect = Rect::new(x0, y0, x0 + 11, y0 + 9);
        for ch in [1, 2, 3, 4, 5, 6] {
            for alpha in [false, true] {
                let src = patterned_image(rect, ch, alpha);
                let ctx = Ctx { bounds: rect, mode: ColorMode::Multichannel, alpha };
                let out = Rect::new(x0 + 2, y0 + 1, x0 + 7, y0 + 6);
                for (angle, distance) in [(0.0, 1.0), (90.0, 2.5), (30.0, 10.0), (-45.0, -63.5), (89.75, 64.0), (37.0, 127.0)] {
                    assert_samples_equal(
                        &motion(&src, out, &ctx, angle, distance),
                        &motion_rows(&src, out, &ctx, angle, distance),
                        &format!("upstream rows: origin=({x0},{y0}), ch={ch}, alpha={alpha}, angle={angle}, distance={distance}"),
                    );
                }
            }
        }
    }
}

fn row_surface(surface: &Surface, params: &FilterParams, area: Rect, bounds: Rect, selection: Option<&Surface>, extent: Option<Rect>) -> Surface {
    let FilterParams::MotionBlur { angle, distance } = params else { panic!("motion-only oracle") };
    let crate::Halo::Radius(reach) = params.halo() else { panic!("motion uses a local halo") };
    let src = match extent {
        Some(e) => Image::read_clamped(surface, area.inflate(reach), e),
        None => Image::read(surface, area.inflate(reach)),
    };
    let format = surface.format();
    let ctx = Ctx { bounds, mode: format.mode, alpha: format.alpha };
    let mut data = motion_rows(&src, area, &ctx, *angle, *distance);
    if let Some(selection) = selection {
        crate::mix_selection(&mut data, area, selection, &src, format.alpha);
    }
    let mut result = surface.clone();
    result.write_region(area, &data);
    result.prune();
    result
}

#[test]
fn motion_pipeline_matches_upstream_rows_across_formats_edges_selections_and_tiles() {
    let rect = Rect::new(-13, 9, 16, 30);
    let mut selection = Surface::new(PixelFormat::GRAY8);
    let mut coverage = Vec::new();
    for y in 0..rect.height() as usize {
        for x in 0..rect.width() as usize {
            coverage.push([0.0, 0.25, 0.5, 0.75, 1.0][(x + y * 3) % 5]);
        }
    }
    selection.write_region(rect, &coverage);
    for mode in [ColorMode::Grayscale, ColorMode::Rgb, ColorMode::Cmyk, ColorMode::Lab] {
        for sample in [SampleType::U8, SampleType::U16, SampleType::F32] {
            for alpha in [false, true] {
                let format = PixelFormat::new(mode, sample, alpha);
                let mut source = Surface::new(format);
                let mut image = patterned_image(rect, source.channels(), alpha);
                if sample == SampleType::F32 {
                    for pixel in image.data.chunks_exact_mut(image.ch) {
                        for value in pixel.iter_mut().take(image.ch - usize::from(alpha)) {
                            *value = *value * 4.0 - 1.0;
                        }
                    }
                }
                source.write_region(rect, &image.data);
                for params in [
                    FilterParams::MotionBlur { angle: 0.0, distance: 2.5 },
                    FilterParams::MotionBlur { angle: 32.0, distance: 11.0 },
                    FilterParams::MotionBlur { angle: -90.0, distance: -7.25 },
                ] {
                    for extent in [None, Some(rect)] {
                        for mask in [None, Some(&selection)] {
                            let expected = row_surface(&source, &params, rect, rect, mask, extent);
                            for tile in [5, 17, 256] {
                                let actual = crate::apply_tiled(&source, &params, rect, rect, mask, tile, extent);
                                assert_eq!(
                                    actual.to_interleaved(rect.inflate(2)),
                                    expected.to_interleaved(rect.inflate(2)),
                                    "mode={mode:?}, sample={sample:?}, alpha={alpha}, params={params:?}, extent={extent:?}, selection={}, tile={tile}",
                                    mask.is_some()
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
