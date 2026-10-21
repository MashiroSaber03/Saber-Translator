//! Alpha / spot channels, channel options and the Quick Mask through PSD.

use photocraft_color::{Color, ColorMode, PixelFormat, SampleType};
use photocraft_doc::{AlphaChannel, ColorIndicates, Document};
use photocraft_geom::{Rect, Size};
use photocraft_io::{document_to_psd, psd_to_document};
use photocraft_raster::Surface;

fn channel(depth: SampleType, x1: i32) -> Surface {
    let mut s = Surface::new(PixelFormat::new(ColorMode::Grayscale, depth, false));
    s.fill_rect(Rect::new(0, 0, x1, 8), &[1.0]);
    s
}

#[test]
fn channel_options_spot_and_quick_mask_roundtrip() {
    for depth in SampleType::ALL {
        let mut d = Document::with_background("c", Size::new(16, 8), ColorMode::Rgb, depth, Color::WHITE);
        let mut a = AlphaChannel::new("Alpha 1", channel(depth, 4));
        a.color = Color::rgb(0.0, 0.0, 1.0);
        a.opacity = 0.7;
        a.indicates = ColorIndicates::SelectedAreas;
        d.channels.push(a);
        d.channels.push(AlphaChannel { spot: Some((Color::rgb(1.0, 0.0, 1.0), 0.25)), ..AlphaChannel::new("PANTONE X", channel(depth, 8)) });
        d.quick_mask = Some(AlphaChannel::new("Quick Mask", channel(depth, 12)));
        let (back, warnings) = psd_to_document(&document_to_psd(&d));
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(back.channels.len(), 2, "{depth:?}");
        let (a, s) = (&back.channels[0], &back.channels[1]);
        assert_eq!((a.name.as_str(), a.color, a.indicates), ("Alpha 1", Color::rgb(0.0, 0.0, 1.0), ColorIndicates::SelectedAreas));
        assert!((a.opacity - 0.7).abs() < 1e-6);
        assert_eq!(a.surface.sample_channel(2, 2, 0), 1.0);
        assert_eq!(a.surface.sample_channel(6, 2, 0), 0.0);
        assert_eq!(s.name, "PANTONE X");
        assert_eq!(s.spot, Some((Color::rgb(1.0, 0.0, 1.0), 0.25)));
        let q = back.quick_mask.as_ref().expect("quick mask restored");
        assert_eq!(q.surface.sample_channel(10, 2, 0), 1.0);
        assert_eq!(q.surface.sample_channel(14, 2, 0), 0.0);
        // Re-exporting doesn't duplicate the display-info resources.
        let f2 = document_to_psd(&back);
        assert_eq!(f2.resources.iter().filter(|r| r.id == 1077).count(), 1);
        assert_eq!(f2.resources.iter().filter(|r| r.id == 1022).count(), 1);
    }

    let mut full = Document::with_background("full", Size::new(1, 1), ColorMode::Rgb, SampleType::U8, Color::WHITE);
    full.channels = (0..53).map(|i| AlphaChannel::new(format!("Alpha {i}"), Surface::new(PixelFormat::GRAY8))).collect();
    full.quick_mask = Some(AlphaChannel::new("Quick Mask", Surface::new(PixelFormat::GRAY8)));
    let out = photocraft_io::export(&full, "full.psd", &Default::default()).unwrap();
    assert!(out.warnings.iter().any(|warning| warning.contains("Quick Mask")), "{:?}", out.warnings);
    let file = photocraft_psd::PsdFile::from_bytes(&out.bytes).unwrap();
    assert_eq!(file.header.channels, 56);
    assert!(file.resources.iter().all(|resource| resource.id != 1022));
}
