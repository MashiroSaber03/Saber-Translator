use super::*;
use photocraft_color::{ColorMode, PixelFormat, SampleType};
#[test]
fn shared_interpolation_matches_original_at_all_depths_and_edges() {
    let bounds = Rect::new(-13, -9, 39, 26);
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
        for sample in [SampleType::U8, SampleType::U16, SampleType::F32] {
            for alpha in [false, true] {
                let fmt = PixelFormat::new(mode, sample, alpha);
                let n = fmt.channels();
                let mut s = Surface::new(fmt);
                let data: Vec<f32> = (0..bounds.width() as usize * bounds.height() as usize * n)
                    .map(|i| if i % 17 == 0 { 0.0 } else { ((i * 103 + i / n * 71) % 65536) as f32 / 65535.0 })
                    .collect();
                s.write_region(bounds, &data);
                for edge in [EdgeMode::Transparency, EdgeMode::Extension, EdgeMode::Color([0.2, 0.5, 0.7, 0.4])] {
                    for ca in [false, true] {
                        let map = |x: f64, y: f64| {
                            if (x as i32 + y as i32) % 13 == 0 {
                                return None;
                            }
                            let p = [x * 0.83 - 3.7, y * 1.14 + 1.3];
                            Some(Sample { p, pr: if ca { [p[0] - 0.9, p[1] + 0.3] } else { p }, pb: if ca { [p[0] + 0.7, p[1] - 0.4] } else { p }, gain: 1.17 })
                        };
                        let out = bounds.inflate(3);
                        let a = remap_impl(&s, bounds, out, edge, &map, true).read_region(out);
                        let b = remap_impl(&s, bounds, out, edge, &map, false).read_region(out);
                        assert_eq!(
                            a.iter().zip(&b).position(|(a, b)| a.to_bits() != b.to_bits()),
                            None,
                            "{mode:?} {sample:?} alpha {alpha} edge {edge:?} ca {ca}"
                        );
                    }
                }
            }
        }
    }
}
#[test]
fn shared_sampler_rejects_bad_windows_and_nonfinite_points() {
    let win = Rect::new(0, 0, 3, 3);
    assert!(sample_channels(&[], win, win, 4, [1.0, 1.0]).is_none());
    assert!(sample_channels(&[0.0; 36], win, win, 4, [f64::NAN, 1.0]).is_none());
    assert!(sample_channels(&[], Rect::EMPTY, win, 4, [1.0, 1.0]).is_none());
}
