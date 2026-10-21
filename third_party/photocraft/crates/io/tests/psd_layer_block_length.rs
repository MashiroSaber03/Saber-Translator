//! The global `Lr16` / `Lr32` block of a 16/32-bit export declares the exact length of the layer
//! data, and its padding to 4 bytes follows outside that length (#4654).
//!
//! Photoshop writes these blocks this way. Readers that check the block length exactly (Unity's
//! PSD Importer, "Incorrect length for InfoLayers") reject a block whose length counts the padding.

mod common;

use common::Features;
use photocraft_color::{ColorMode, SampleType};
use photocraft_io::*;
use photocraft_psd::PsdFile;

fn be(b: &[u8], at: usize, long: bool) -> usize {
    if long { u64::from_be_bytes(b[at..at + 8].try_into().unwrap()) as usize } else { u32::from_be_bytes(b[at..at + 4].try_into().unwrap()) as usize }
}

/// Walks the raw file to the first global block and returns (key, declared length, the bytes
/// after the data up to the next block or the end of the section).
fn first_global_block(bytes: &[u8]) -> ([u8; 4], usize, Vec<u8>) {
    let psb = be(bytes, 4, false) >> 16 == 2;
    let mut p = 26;
    p += 4 + be(bytes, p, false); // color mode data
    p += 4 + be(bytes, p, false); // image resources
    let lw = if psb { 8 } else { 4 };
    let end = p + lw + be(bytes, p, psb);
    p += lw;
    p += lw + be(bytes, p, psb); // classic layer info
    p += 4 + be(bytes, p, false); // global layer mask
    let key: [u8; 4] = bytes[p + 4..p + 8].try_into().unwrap();
    let len = be(bytes, p + 8, psb);
    let data_end = p + 8 + if psb { 8 } else { 4 } + len;
    let next = (data_end..end).find(|&q| bytes.get(q..q + 4).is_some_and(|s| s == b"8BIM" || s == b"8B64")).unwrap_or(end);
    (key, len, bytes[data_end..next].to_vec())
}

#[test]
fn lr16_lr32_length_excludes_padding() {
    let mut unaligned = 0;
    for (depth, key) in [(SampleType::U16, *b"Lr16"), (SampleType::F32, *b"Lr32")] {
        for mode in [ColorMode::Rgb, ColorMode::Grayscale] {
            for features in [Features::PIXELS, Features::ALL] {
                let doc = common::gen_doc(mode, depth, features);
                for path in ["x.psd", "x.psb"] {
                    let name = format!("{depth:?} {mode:?} {path}");
                    let bytes = export(&doc, path, &ExportOptions::default()).expect("export").bytes;
                    let file = PsdFile::from_bytes(&bytes).expect("reparse");
                    let body = file.layer_info.as_ref().expect("layers").unpadded_len(file.header.version).unwrap() as usize;
                    let (k, len, after) = first_global_block(&bytes);
                    assert_eq!(k, key, "{name}: first global block");
                    assert_eq!(len, body, "{name}: declared length vs layer data");
                    assert_eq!(after, vec![0; (4 - body % 4) % 4], "{name}: padding after the block");
                    unaligned += usize::from(!body.is_multiple_of(4));
                }
            }
        }
    }
    // Only a body that is not already a multiple of 4 tells padding inside the length apart from
    // padding after it.
    assert!(unaligned > 0, "no case exercises the padding");
}
