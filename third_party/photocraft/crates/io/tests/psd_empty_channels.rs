mod common;
use common::Features;
use photocraft_color::{ColorMode, SampleType};
use photocraft_io::{ExportOptions, export};
use photocraft_psd::{Compression, PsdFile};

#[test]
fn empty_layer_and_mask_channels_use_raw_without_a_payload() {
    for mode in [ColorMode::Rgb, ColorMode::Grayscale] {
        for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
            for ext in ["out.psd", "out.psb"] {
                let mut doc = common::gen_doc(mode, depth, Features::ALL);
                let mut masked = common::raster("Uniform mask", doc.pixel_format(), photocraft_geom::Rect::new(0, 0, 3, 3), 1, true);
                masked.mask = Some(photocraft_doc::LayerMask {
                    surface: photocraft_raster::Surface::new(photocraft_color::PixelFormat::new(ColorMode::Grayscale, depth, false)),
                    enabled: true,
                    linked: true,
                    density: 1.0,
                    feather: 0.0,
                });
                doc.layers.push(masked);
                let bytes = export(&doc, ext, &ExportOptions::default()).unwrap().bytes;
                let file = PsdFile::from_bytes(&bytes).unwrap();
                let mut empty = 0;
                let mut empty_masks = 0;
                for layer in file.layers() {
                    for channel in &layer.channels {
                        let rect = layer.channel_rect(channel.id);
                        if rect.width() == 0 || rect.height() == 0 {
                            empty += 1;
                            if channel.id == -2 {
                                empty_masks += 1;
                            }
                            assert_eq!(channel.compression, Some(Compression::Raw), "{mode:?} {depth:?} {ext}: {} channel {}", layer.name(), channel.id);
                            assert!(channel.data.is_empty());
                        }
                    }
                }
                assert!(empty > 0);
                assert!(empty_masks > 0);
            }
        }
    }
}
