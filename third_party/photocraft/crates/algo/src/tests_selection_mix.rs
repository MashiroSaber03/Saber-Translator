use super::*;
use photocraft_color::{PixelFormat, SampleType};

fn selection(coverage: f32) -> Surface {
    Surface::with_default(PixelFormat::new(ColorMode::Grayscale, SampleType::F32, false), &[coverage])
}

#[test]
fn feathered_color_to_alpha_preserves_appearance_over_the_key() {
    let r = Rect::new(0, 0, 1, 1);
    let filter = FilterParams::ColorToAlpha { color: [1.0; 4], transparency_threshold: 0.0, opacity_threshold: 1.0 };
    for sample in SampleType::ALL {
        for mode in [ColorMode::Rgb, ColorMode::Grayscale] {
            for alpha in [0.5, 1.0] {
                let format = PixelFormat::new(mode, sample, true);
                let mut s = Surface::new(format);
                let mut p = vec![0.5; format.channels()];
                *p.last_mut().unwrap() = alpha;
                s.write_pixel(0, 0, &p);
                let original = s.pixel(0, 0);
                let a = format.channels() - 1;
                for coverage in [0.0, 0.25, 0.5, 0.75, 1.0] {
                    let out = apply(&s, &filter, r, r, Some(&selection(coverage)));
                    let actual = out.pixel(0, 0);
                    let tolerance = if sample == SampleType::U8 { 2.0 / 255.0 } else { 1e-4 };
                    for c in 0..a {
                        let before = original[c] * original[a] + 1.0 - original[a];
                        let after = actual[c] * actual[a] + 1.0 - actual[a];
                        assert!(
                            (before - after).abs() < tolerance,
                            "{sample:?}/{mode:?}, alpha={alpha}, coverage={coverage}: {original:?} -> {actual:?}, over white {before} -> {after}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn feathered_offset_ignores_hidden_colour_of_the_original_pixel() {
    let r = Rect::new(0, 0, 2, 1);
    let filter = FilterParams::Offset { horizontal: 1, vertical: 0, undefined: UndefinedAreas::Wrap };
    for sample in SampleType::ALL {
        let mut s = Surface::new(PixelFormat::new(ColorMode::Rgb, sample, true));
        s.write_pixel(0, 0, &[1.0, 0.0, 0.0, 0.0]);
        s.write_pixel(1, 0, &[0.0, 0.0, 1.0, 1.0]);
        let out = apply(&s, &filter, r, r, Some(&selection(0.5)));
        let p = out.pixel(0, 0);
        assert_eq!(&p[..3], &[0.0, 0.0, 1.0], "{sample:?}: hidden red contaminated the selected blue: {p:?}");
        assert!((p[3] - 0.5).abs() < 1.0 / 255.0);
    }
}

#[test]
fn feathered_blurs_do_not_reintroduce_hidden_colour() {
    let r = Rect::new(0, 0, 3, 3);
    let mut s = Surface::new(PixelFormat::RGBA32F);
    s.fill_rect(r, &[1.0, 0.0, 0.0, 0.0]);
    s.write_pixel(1, 1, &[0.0, 0.0, 1.0, 1.0]);
    for filter in [FilterParams::BoxBlur { radius: 1.0 }, FilterParams::MotionBlur { angle: 0.0, distance: 3.0 }] {
        let full = apply(&s, &filter, r, r, None);
        let out = apply(&s, &filter, r, r, Some(&selection(0.5)));
        let p = out.pixel(0, 1);
        assert!(p[3] > 0.0, "{filter:?}: expected blur to spread coverage");
        assert!((p[3] - full.pixel(0, 1)[3] * 0.5).abs() < 1e-6, "{filter:?}: selection should halve the new coverage");
        assert!(p[0].abs() < 1e-6 && (p[2] - 1.0).abs() < 1e-6, "{filter:?}: hidden red reappeared: {p:?}");
    }
}

#[test]
fn selection_mixing_keeps_hdr_range_and_exact_endpoints() {
    let r = Rect::new(0, 0, 1, 1);
    let original = Image { rect: r, ch: 4, data: vec![-2.0, 4.0, 0.5, 0.25] };
    let filtered = vec![8.0, -4.0, 0.75, 1.0];
    let mut data = filtered.clone();
    mix_selection(&mut data, r, &selection(0.5), &original, true);
    for (&actual, expected) in data.iter().zip([6.0, -2.4, 0.7, 0.625]) {
        assert!((actual - expected).abs() < 1e-6, "{data:?}");
    }
    for coverage in [0.0, 1.0] {
        let mut data = filtered.clone();
        mix_selection(&mut data, r, &selection(coverage), &original, true);
        assert_eq!(data, if coverage == 0.0 { &original.data } else { &filtered }.clone());
    }
}

#[test]
fn selection_mixing_without_alpha_keeps_every_channel_independent() {
    let r = Rect::new(0, 0, 1, 1);
    // One channel is a mask; three/four channels can be RGB or CMYK without alpha.
    for ch in [1, 3, 4] {
        let original = Image { rect: r, ch, data: vec![0.25; ch] };
        let mut data = vec![0.75; ch];
        mix_selection(&mut data, r, &selection(0.25), &original, false);
        assert_eq!(data, vec![0.375; ch]);
    }
}
