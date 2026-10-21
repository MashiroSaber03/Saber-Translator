//! Regression oracles for Mosaic's cell-once traversal.
use super::*;
use photocraft_color::{PixelFormat, SampleType};

// Retain the previous traversal as a value oracle, including its alpha math.
fn legacy_mosaic(src: &Image, out: Rect, ctx: &Ctx, cell: f32) -> Vec<f32> {
    let n = src.ch;
    let cs = cell.max(1.0).round() as i32;
    let b = ctx.bounds;
    let mut res = Vec::with_capacity(out.width() as usize * out.height() as usize * n);
    let mut cache: Option<(i32, i32, Vec<f32>)> = None;
    for y in out.y0..out.y1 {
        for x in out.x0..out.x1 {
            let (cx, cy) = ((x - b.x0).div_euclid(cs), (y - b.y0).div_euclid(cs));
            if cache.as_ref().is_none_or(|(a, bb, _)| (*a, *bb) != (cx, cy)) {
                let r = Rect::new(b.x0 + cx * cs, b.y0 + cy * cs, b.x0 + (cx + 1) * cs, b.y0 + (cy + 1) * cs).intersect(&b);
                let mut acc = vec![0.0f32; n];
                let mut count = 0.0;
                for yy in r.y0..r.y1 {
                    for xx in r.x0..r.x1 {
                        let a = if ctx.alpha { src.get(xx, yy, n - 1) } else { 1.0 };
                        for (c, v) in acc.iter_mut().enumerate() {
                            let s = src.get(xx, yy, c);
                            *v += if ctx.alpha && c < n - 1 { s * a } else { s };
                        }
                        count += 1.0;
                    }
                }
                if count > 0.0 {
                    for v in acc.iter_mut() {
                        *v /= count;
                    }
                }
                if ctx.alpha {
                    let a = acc[n - 1];
                    for v in acc.iter_mut().take(n - 1) {
                        *v = if a > 1e-7 { *v / a } else { 0.0 };
                    }
                }
                cache = Some((cx, cy, acc));
            }
            res.extend_from_slice(&cache.as_ref().map(|c| c.2.clone()).unwrap_or_default());
        }
    }
    res
}

fn image_pattern(rect: Rect, ch: usize) -> Image {
    let mut seed = 0x72e1905u32;
    let data = (0..rect.width() as usize * rect.height() as usize * ch)
        .map(|i| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            if i % ch == ch - 1 && i % 11 == 0 { 0.0 } else { (seed >> 8) as f32 / 16777215.0 }
        })
        .collect();
    Image { rect, ch, data }
}

#[test]
fn mosaic_matches_legacy_for_partial_cells_and_offset_bounds() {
    for b in [Rect::new(0, 0, 47, 61), Rect::new(-5, 7, 42, 68)] {
        for ch in 1..=5 {
            for alpha in [false, true] {
                for rect in [b, Rect::new(b.x0 + 3, b.y0 + 5, b.x1 - 6, b.y1 - 8)] {
                    let src = image_pattern(rect, ch);
                    let ctx = Ctx { bounds: b, mode: ColorMode::Rgb, alpha };
                    for output in [
                        b,
                        Rect::new(b.x0 + 4, b.y0 + 2, b.x1 - 3, b.y1 - 4),
                        Rect::new(b.x0 + 9, b.y0 + 9, b.x0 + 16, b.y0 + 17),
                        Rect::new(b.x0 - 3, b.y0 - 4, b.x1 + 6, b.y1 + 8),
                        Rect::new(b.x1 + 2, b.y1 - 3, b.x1 + 12, b.y1 + 5),
                        Rect::new(b.x0, b.y0, b.x0, b.y1),
                        Rect::new(b.x0, b.y0, b.x1, b.y0),
                    ] {
                        for size in [0.0, 1.0, 1.5, 2.0, 7.0, 13.0, 50.0, 80.0, 180.0] {
                            let expected = legacy_mosaic(&src, output, &ctx, size);
                            let got = stylize::mosaic(&src, output, &ctx, size);
                            assert_eq!(got.len(), expected.len());
                            assert!(
                                got.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()),
                                "Mosaic differs: bounds={b:?}, source={rect:?}, output={output:?}, channels={ch}, alpha={alpha}, cell={size}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn mosaic_preserves_depth_color_alpha_and_tile_independence() {
    let b = Rect::new(13, -7, 92, 54);
    for sample in [SampleType::U8, SampleType::U16, SampleType::F32] {
        for mode in [ColorMode::Grayscale, ColorMode::Rgb, ColorMode::Cmyk, ColorMode::Lab] {
            for alpha in [false, true] {
                let fmt = PixelFormat::new(mode, sample, alpha);
                let pattern = image_pattern(b, fmt.channels());
                let mut input = Surface::new(fmt);
                input.write_region(b, &pattern.data);
                let src = Image::read(&input, b);
                let ctx = Ctx { bounds: b, mode, alpha };
                for cell_size in [6.0, 13.0, 50.0, 180.0] {
                    let params = FilterParams::Mosaic { cell_size };
                    let mut expected = Surface::new(fmt);
                    expected.write_region(b, &legacy_mosaic(&src, b, &ctx, cell_size));
                    let bytes = expected.to_interleaved(b);
                    for tile in [17, 256] {
                        for extent in [None, Some(b)] {
                            let got = apply_tiled(&input, &params, b, b, None, tile, extent);
                            assert_eq!(
                                got.to_interleaved(b),
                                bytes,
                                "Mosaic differs: depth={sample:?}, mode={mode:?}, alpha={alpha}, cell={cell_size}, tile={tile}, extent={extent:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn mosaic_handles_zero_channels() {
    let rect = Rect::new(0, 0, 3, 3);
    let src = Image { rect, ch: 0, data: Vec::new() };
    for alpha in [false, true] {
        let ctx = Ctx { bounds: rect, mode: ColorMode::Rgb, alpha };
        assert!(stylize::mosaic(&src, rect, &ctx, 2.0).is_empty());
    }
}

#[test]
fn mosaic_row_reads_preserve_hdr_alpha_and_partial_source_bits() {
    let bounds = Rect::new(-7, 3, 30, 26);
    for rect in [bounds, Rect::new(-3, 7, 23, 21)] {
        let mut src = image_pattern(rect, 4);
        for (i, pixel) in src.data.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            pixel[0] = pixel[0] * 8.0 - 4.0;
            pixel[1] = if i.is_multiple_of(3) { -0.0 } else { pixel[1] * 16.0 };
            pixel[2] = if i.is_multiple_of(7) { f32::MIN_POSITIVE } else { -pixel[2] };
            pixel[3] = if i.is_multiple_of(5) { 0.0 } else { pixel[3] };
        }
        for alpha in [false, true] {
            let ctx = Ctx { bounds, mode: ColorMode::Rgb, alpha };
            for cell in [2.0, 4.0, 10.0, 50.0, 180.0] {
                let expected = legacy_mosaic(&src, bounds, &ctx, cell);
                let got = stylize::mosaic(&src, bounds, &ctx, cell);
                assert_eq!(got.len(), expected.len());
                assert!(got.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()));
            }
        }
    }
}

#[test]
#[ignore = "24 MP release performance comparison; run explicitly with --release --ignored --nocapture"]
fn mosaic_24mp_release_comparison() {
    use std::time::Instant;
    let rect = Rect::new(0, 0, 6000, 4000);
    let src = image_pattern(rect, 4);
    let ctx = Ctx { bounds: rect, mode: ColorMode::Rgb, alpha: true };
    for cell in [50.0, 180.0] {
        let start = Instant::now();
        let expected = legacy_mosaic(&src, rect, &ctx, cell);
        let old_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let got = stylize::mosaic(&src, rect, &ctx, cell);
        let new_ms = start.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(expected.len(), got.len());
        assert!(expected.iter().zip(&got).all(|(a, b)| a.to_bits() == b.to_bits()));
        eprintln!("Mosaic 24 MP, cell {cell}: legacy {old_ms:.3} ms, cell-once {new_ms:.3} ms, {:.2}x kernel speedup", old_ms / new_ms);
    }
}
