use super::*;
use photocraft_color::{ColorMode, PixelFormat, SampleType};

#[test]
fn all_rotation_tables_keep_the_direct_bits() {
    for angle in [0.0, 0.1, 15.0, 73.7, 180.0, 360.0, f32::NAN, f32::INFINITY] {
        let p = SpinPin { blur_angle: angle, ..Default::default() };
        let plans = spin_plan::prepare(&[p]).unwrap();
        let theta = angle.clamp(0.0, 360.0).to_radians();
        for count in 2..=64 {
            for j in 0..count {
                let phi = theta * (j as f32 / (count - 1) as f32 - 0.5);
                let (a, b) = phi.sin_cos();
                let (c, d) = plans[0].rotation(count, j).unwrap();
                assert_eq!((a.to_bits(), b.to_bits()), (c.to_bits(), d.to_bits()));
            }
        }
    }
    assert!(spin_plan::prepare(&vec![SpinPin::default(); 121]).is_none());
}
#[test]
fn prepared_spin_matches_the_original_kernel() {
    let bounds = Rect::new(-17, -11, 58, 45);
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
            let n = PixelFormat::new(mode, SampleType::F32, alpha).channels();
            let ctx = Ctx { bounds, mode, alpha };
            let mut src = Image::new(bounds, n);
            for (i, v) in src.data.iter_mut().enumerate() {
                *v = ((i * 71 + i / n * 107) % 65536) as f32 / 65535.0;
            }
            for (i, px) in src.data.chunks_exact_mut(n).enumerate() {
                if i % 11 == 0 {
                    px[n - 1] = 0.0;
                }
            }
            for angle in [0.0, 0.1, 15.0, 91.0, 360.0, f32::NAN] {
                let pins = [
                    SpinPin { x: 0.2, y: 0.3, radius_x: 0.25, radius_y: 0.5, angle: 31.0, blur_angle: angle },
                    SpinPin { x: 0.7, y: 0.6, radius_x: 0.8, radius_y: 0.6, angle: -15.0, blur_angle: angle },
                ];
                let out = Rect::new(-19, -8, 55, 49);
                let a = spin(&src, out, &ctx, &pins);
                let b = spin_impl::<0>(&src, out, &ctx, &pins, false);
                let first = a.iter().zip(&b).position(|(a, b)| a.to_bits() != b.to_bits());
                assert_eq!(first, None, "{mode:?} alpha {alpha} angle {angle}");
            }
        }
    }
}
