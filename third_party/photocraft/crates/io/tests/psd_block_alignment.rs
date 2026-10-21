//! Every layer-level tagged block of an exported PSD/PSB is a multiple of 4 bytes long.
//!
//! Photoshop pads layer-level blocks to 4 inside the length field; all of its own files do (567
//! fixtures checked), and a regenerated block that stopped at an even but unaligned size left the
//! layer record's extra data unaligned: a Pattern Fill's `PtFl` came out 2 bytes short of the
//! boundary for half of all pattern names.

use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_doc::{Document, Fill, Layer, LayerContent, Pattern};
use photocraft_geom::{Rect, Size};
use photocraft_io::*;
use photocraft_psd::PsdFile;
use photocraft_raster::Surface;

fn pattern(name: &str) -> Pattern {
    let mut s = Surface::new(PixelFormat::RGBA8);
    s.fill_rect(Rect::new(0, 0, 8, 8), &[0.8, 0.8, 0.8, 1.0]);
    s.fill_rect(Rect::new(0, 0, 4, 4), &[0.2, 0.4, 0.9, 1.0]);
    Pattern { id: "059289fb-0000-4000-8000-000000000001".into(), name: name.into(), width: 8, height: 8, surface: s }
}

/// One Pattern Fill layer per name; the name's length moves the descriptor's size by two bytes
/// per character, so consecutive lengths cover every residue modulo 4.
fn doc(depth: SampleType, names: &[&str]) -> Document {
    let mut d = Document::new("p", Size::new(40, 24), ColorMode::Rgb, depth);
    for name in names {
        let p = pattern(name);
        let fill = Fill::Pattern { name: (*name).into(), scale: 1.0, id: p.id.clone(), angle: 0.0, link: true, phase: (0.0, 0.0) };
        d.patterns.push(p);
        d.layers.push(Layer::new(format!("Fill {name}"), LayerContent::Fill(fill)));
    }
    d
}

#[test]
fn layer_blocks_are_four_byte_aligned() {
    let names = ["A", "Ab", "Abc", "Abcd", "Dots", "Checker", "Wood grain"];
    for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
        for path in ["x.psd", "x.psb"] {
            let bytes = export(&doc(depth, &names), path, &ExportOptions::default()).expect("export").bytes;
            let file = PsdFile::from_bytes(&bytes).expect("reparse");
            let mut fills = 0;
            for layer in file.layers() {
                for b in &layer.blocks {
                    let key = String::from_utf8_lossy(&b.key).into_owned();
                    assert_eq!(b.data.len() % 4, 0, "{depth:?} {path}: layer {:?} block {key} is {} bytes", layer.name(), b.data.len());
                }
                fills += usize::from(layer.block(b"PtFl").is_some());
            }
            assert_eq!(fills, names.len(), "{depth:?} {path}: every pattern fill keeps its PtFl block");
        }
    }
}
