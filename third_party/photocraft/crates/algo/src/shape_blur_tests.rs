//! The pre-optimization prefix implementation remains an exact scalar oracle.
//! A separate point-by-point convolution also checks normalization and sampling.
use super::*;
use crate::{FilterParams, apply_tiled, kernel};
use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_raster::Surface;

const SHAPES: [BlurShape; 9] = [
    BlurShape::Circle,
    BlurShape::Ring,
    BlurShape::Square,
    BlurShape::Diamond,
    BlurShape::Triangle,
    BlurShape::Hexagon,
    BlurShape::Star,
    BlurShape::Heart,
    BlurShape::Cross,
];

#[test]
fn binary_kernel_sampling_including_fractional_square_is_preserved() {
    for (shape, radius, count) in [
        (BlurShape::Circle, 3.0, 29.0),
        (BlurShape::Ring, 3.0, 20.0),
        (BlurShape::Square, 3.0, 49.0),
        (BlurShape::Square, 2.5, 49.0),
        (BlurShape::Diamond, 3.0, 25.0),
        (BlurShape::Triangle, 3.0, 18.0),
        (BlurShape::Cross, 3.0, 13.0),
    ] {
        assert_eq!(Spans::new(radius, shape_inside(shape)).count, count);
    }
}

fn image(rect: Rect, n: usize, alpha: bool) -> Image {
    let data = (0..rect.width() as usize * rect.height() as usize)
        .flat_map(|i| (0..n).map(move |c| if alpha && c + 1 == n { (i % 7) as f32 / 6.0 } else { ((i * 31 + c * 13) % 97) as f32 / 96.0 }))
        .collect();
    Image { rect, ch: n, data }
}

fn exact(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()), "sample {i}: {a} != {b}");
    }
}

#[test]
fn scalar_prefix_exact_all_shapes_radii_channels_subrects() {
    for shape in SHAPES {
        for radius in [0.0, 0.49, 0.5, 0.75, 1.0, 1.1, 2.5, 5.0, 25.0, 100.0] {
            let spans = Spans::new(radius, shape_inside(shape));
            for n in 1..=MAXC {
                for alpha in [false, true] {
                    let src = image(Rect::new(-5, -3, 9, 7), n, alpha);
                    let ctx = Ctx { bounds: src.rect, mode: ColorMode::Multichannel, alpha };
                    for out in [src.rect, Rect::new(-2, 1, 3, 2), Rect::new(-8, -7, 12, 10), Rect::EMPTY] {
                        exact(&shape_blur(&src, out, &ctx, radius, shape), &conv_spans(&src, out, &ctx, &spans));
                    }
                }
            }
        }
    }
}

#[test]
fn point_convolution_independent_of_prefix_and_span_accumulation() {
    for shape in SHAPES {
        for radius in [0.0f32, 0.5, 1.0, 1.25, 2.5, 5.0, 7.75] {
            let src = image(Rect::new(-3, -2, 4, 3), 4, true);
            let ctx = Ctx { bounds: src.rect, mode: ColorMode::Rgb, alpha: true };
            let out = Rect::new(-5, -4, 6, 5);
            let actual = shape_blur(&src, out, &ctx, radius, shape);
            let inside = shape_inside(shape);
            let ri = radius.ceil() as i32;
            let mut points: Vec<(i32, i32)> = (-ri..=ri)
                .flat_map(|y| (-ri..=ri).map(move |x| (x, y)))
                .filter(|&(x, y)| radius >= 0.5 && inside(x as f32 / radius, y as f32 / radius))
                .collect();
            if points.is_empty() {
                points.push((0, 0));
            }
            let mut reference = Vec::new();
            for y in out.y0..out.y1 {
                for x in out.x0..out.x1 {
                    let mut sums = [0.0f64; 4];
                    for &(dx, dy) in &points {
                        let a = src.get(x + dx, y + dy, 3);
                        for (c, sum) in sums.iter_mut().enumerate() {
                            let v = src.get(x + dx, y + dy, c);
                            *sum += f64::from(if c < 3 { v * a } else { v });
                        }
                    }
                    let mut px = sums.map(|s| (s / points.len() as f64) as f32);
                    unpremul_px(&mut px, true);
                    reference.extend(px);
                }
            }
            // Finite f32 inputs are exactly representable in these short f64 sums.
            exact(&actual, &reference);
        }
    }
}

#[test]
fn surfaces_exact_depth_models_alpha_selection_and_tile_boundaries() {
    let bounds = Rect::new(-2, -1, 263, 5);
    let area = Rect::new(0, 0, 260, 4);
    let mut selection = Surface::new(PixelFormat::new(ColorMode::Grayscale, SampleType::F32, false));
    let coverage: Vec<f32> = (0..area.width() as usize * area.height() as usize).map(|i| (i % 5) as f32 / 4.0).collect();
    selection.write_region(area, &coverage);
    for shape in SHAPES {
        for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
            for mode in [ColorMode::Rgb, ColorMode::Grayscale, ColorMode::Cmyk, ColorMode::Lab, ColorMode::Multichannel] {
                for alpha in [false, true] {
                    let fmt = PixelFormat::new(mode, depth, alpha);
                    let n = fmt.channels();
                    let mut surface = Surface::new(fmt);
                    surface.write_region(bounds, &image(bounds, n, alpha).data);
                    let ctx = Ctx { bounds, mode, alpha };
                    for radius in [2.5, 5.0, 25.0] {
                        let p = FilterParams::ShapeBlur { radius, shape };
                        let reach = radius.ceil() as i32 + 1;
                        for extent in [None, Some(bounds)] {
                            let src = match extent {
                                None => Image::read(&surface, area.inflate(reach)),
                                Some(e) => Image::read_clamped(&surface, area.inflate(reach), e),
                            };
                            let spans = Spans::new(radius, shape_inside(shape));
                            let mut expected = conv_spans(&src, area, &ctx, &spans);
                            crate::mix_selection(&mut expected, area, &selection, &src, alpha);
                            let mut reference = surface.clone();
                            reference.write_region(area, &expected);
                            for tile in [17, 256] {
                                let actual = apply_tiled(&surface, &p, area, bounds, Some(&selection), tile, extent);
                                exact(&actual.read_region(bounds), &reference.read_region(bounds));
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn largest_supported_radius_and_fraction_keep_scalar_kernel() {
    let src = image(Rect::new(-1, -1, 2, 2), 4, true);
    let ctx = Ctx { bounds: src.rect, mode: ColorMode::Rgb, alpha: true };
    for shape in SHAPES {
        for radius in [999.25, 1000.0] {
            let spans = Spans::new(radius, shape_inside(shape));
            exact(&shape_blur(&src, src.rect, &ctx, radius, shape), &conv_spans(&src, src.rect, &ctx, &spans));
        }
    }
}

#[test]
fn degenerate_hdr_and_nonfinite_samples_keep_scalar_behavior() {
    for rect in [Rect::EMPTY, Rect::new(0, 0, 1, 1), Rect::new(0, 0, 1, 7), Rect::new(0, 0, 9, 1)] {
        for alpha in [false, true] {
            let mut src = image(rect, 4, alpha);
            for (i, v) in src.data.iter_mut().enumerate() {
                if !alpha || i % 4 != 3 {
                    *v = [f32::MAX, -f32::MAX, 1e-30, -17.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN][i % 7];
                } else {
                    *v = [0.0, 1e-8, 1e-7, 1.1e-7, 1.0][i % 5];
                }
            }
            let ctx = Ctx { bounds: rect, mode: ColorMode::Lab, alpha };
            for shape in SHAPES {
                let spans = Spans::new(2.5, shape_inside(shape));
                exact(&shape_blur(&src, rect, &ctx, 2.5, shape), &conv_spans(&src, rect, &ctx, &spans));
            }
        }
    }
    for alpha in [f32::NEG_INFINITY, -1.0, f32::NAN, 1e-30, f32::INFINITY, f32::MAX] {
        let rect = Rect::new(0, 0, 3, 2);
        let src = Image { rect, ch: 4, data: [1e20, -17.0, 0.0, alpha].repeat(6) };
        let ctx = Ctx { bounds: rect, mode: ColorMode::Rgb, alpha: true };
        for shape in SHAPES {
            let spans = Spans::new(2.5, shape_inside(shape));
            exact(&shape_blur(&src, rect, &ctx, 2.5, shape), &conv_spans(&src, rect, &ctx, &spans));
        }
    }
}

#[test]
fn hostile_shape_inputs_are_bounded_and_do_not_panic() {
    let rect = Rect::new(0, 0, 1, 1);
    let ctx = Ctx { bounds: rect, mode: ColorMode::Rgb, alpha: true };
    for src in [
        Image { rect, ch: 0, data: vec![] },
        Image { rect, ch: 9, data: vec![0.0; 9] },
        Image { rect, ch: 4, data: vec![] },
        Image { rect: Rect::new(i32::MIN, i32::MIN, i32::MAX, i32::MAX), ch: 4, data: vec![] },
    ] {
        assert!(kernel(&FilterParams::ShapeBlur { radius: 5.0, shape: BlurShape::Circle }, &src, rect, &ctx).is_empty());
    }
    let src = image(rect, 4, true);
    for radius in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0, f32::MAX] {
        let p = FilterParams::ShapeBlur { radius, shape: BlurShape::Circle };
        assert!(matches!(p.halo(), crate::Halo::Radius(1..=1001)));
        assert_eq!(kernel(&p, &src, rect, &ctx).len(), 4);
    }
    assert!(shape_blur(&src, Rect::new(i32::MIN, i32::MIN, i32::MAX, i32::MAX), &ctx, 5.0, BlurShape::Circle).is_empty());
}

#[test]
fn shape_tile_overrides_and_cancel_preserve_input() {
    for r in [Rect::new(0, 0, 4, 4), Rect::new(i32::MAX - 8, i32::MAX - 8, i32::MAX - 4, i32::MAX - 4)] {
        let mut s = Surface::new(PixelFormat::RGBA8);
        s.fill_rect(r, &[0.25, 0.5, 0.75, 1.0]);
        let before = s.read_region(r);
        let p = FilterParams::ShapeBlur { radius: 0.0, shape: BlurShape::Circle };
        for tile in [i32::MIN, -1, 0, 256, i32::MAX] {
            exact(&apply_tiled(&s, &p, r, r, None, tile, Some(r)).read_region(r), &before);
        }
        let ctl = photocraft_raster::Interrupt::cancel_only(&|| true);
        assert!(crate::apply_tiled_with(&s, &p, r, r, None, 256, Some(r), &ctl).is_none());
        exact(&s.read_region(r), &before);
    }
}

#[test]
#[ignore = "36 MP direct-kernel memory regression (about 1.7 GiB); run explicitly"]
fn direct_36mp_image_with_small_output_is_not_rejected() {
    let rect = Rect::new(0, 0, 6000, 6000);
    let src = Image { rect, ch: 4, data: vec![0.25; 36_000_000 * 4] };
    let ctx = Ctx { bounds: rect, mode: ColorMode::Rgb, alpha: true };
    let result = shape_blur(&src, Rect::new(3000, 3000, 3001, 3001), &ctx, 5.0, BlurShape::Circle);
    assert_eq!(result, vec![0.25; 4]);
}
