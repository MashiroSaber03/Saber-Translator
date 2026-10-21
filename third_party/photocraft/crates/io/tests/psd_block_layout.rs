//! #200: PSD export must write tagged blocks that re-parse strictly. Third-party files
//! re-saved by PhotoCraft had `soLD` data under the `PlLd` key, global blocks padded only
//! to even, a layer info body not padded to 4 and, in PSB, `cinf`/`lnkE` read and written
//! with 4-byte lengths; psd-tools rejected or misaligned all of them.

mod common;

use photocraft_io::*;
use photocraft_psd::descriptor::{Descriptor, VersionedDescriptor};
use photocraft_psd::testgen;
use photocraft_psd::{Compression, LayerRecord, PsdFile, TaggedBlock, Version};

/// `PlLd` data per the Adobe spec ("Placed Layer"): `plcL`, version 3, Pascal uuid, page,
/// total pages, anti-alias policy, layer type, 8 transform doubles, warp version, warp
/// descriptor.
fn plld(uuid: &str) -> Vec<u8> {
    let mut d = b"plcL".to_vec();
    d.extend(3u32.to_be_bytes());
    d.push(uuid.len() as u8);
    d.extend(uuid.as_bytes());
    for v in [1u32, 1, 16, 2] {
        d.extend(v.to_be_bytes());
    }
    for v in [0.0f64, 0.0, 4.0, 0.0, 4.0, 3.0, 0.0, 3.0] {
        d.extend(v.to_be_bytes());
    }
    d.extend(0u32.to_be_bytes());
    d.extend(VersionedDescriptor::new(Descriptor::new("warp")).to_bytes());
    // Photoshop pads the data to 4 inside the block length.
    while !d.len().is_multiple_of(4) {
        d.push(0);
    }
    d
}

/// `SoLd` data per the Adobe spec ("Placed Layer Data"): `soLD`, version 4, descriptor.
fn sold(uuid: &str) -> Vec<u8> {
    let mut d = b"soLD".to_vec();
    d.extend(4u32.to_be_bytes());
    use photocraft_psd::descriptor::Value;
    // The strict check wants what Photoshop needs to place the layer: its file id and transform.
    let quad = [0.0, 0.0, 4.0, 0.0, 4.0, 4.0, 0.0, 4.0].map(Value::Double).to_vec();
    let desc = Descriptor::new("null").with("Idnt", Value::Text(photocraft_psd::descriptor::UnicodeString::new_nul(uuid))).with("Trnf", Value::List(quad));
    d.extend(VersionedDescriptor::new(desc).to_bytes());
    while !d.len().is_multiple_of(4) {
        d.push(0);
    }
    d
}

fn reexport(f: &PsdFile) -> Vec<u8> {
    let bytes = f.to_bytes().unwrap();
    let imp = import("in.psd", &bytes).unwrap();
    export(&imp.document, "out.psd", &ExportOptions::default()).unwrap().bytes
}

#[test]
fn placed_layer_blocks_keep_their_keys() {
    let uuid = "5b0c0b42-1b9c-11e0-a5b7-d0b1b2c3d4e5";
    let mut f = testgen::small(Version::Psd, Compression::Raw);
    // Photoshop's order: PlLd before SoLd. The importer keeps SoLd as the principal block.
    f.layers_mut().push(LayerRecord {
        name: b"smart".to_vec(),
        blocks: vec![TaggedBlock::unicode_name("smart"), TaggedBlock::new(*b"PlLd", plld(uuid)), TaggedBlock::new(*b"SoLd", sold(uuid))],
        ..Default::default()
    });
    for b in f.layers().last().unwrap().blocks.iter() {
        b.check_structure().unwrap();
    }
    let out = reexport(&f);
    assert_eq!(common::strict_block_errors(&out), Vec::<String>::new());
    let back = PsdFile::from_bytes(&out).unwrap();
    let rec = back.layers().iter().find(|l| l.block(b"SoLd").is_some()).unwrap();
    // Before the fix the SoLd data overwrote the PlLd block.
    assert_eq!(rec.block(b"PlLd").unwrap().data, plld(uuid));
    assert_eq!(rec.block(b"SoLd").unwrap().data, sold(uuid));
}

#[test]
fn check_structure_rejects_swapped_placed_data() {
    let uuid = "u";
    assert!(TaggedBlock::new(*b"PlLd", sold(uuid)).check_structure().is_err());
    assert!(TaggedBlock::new(*b"SoLd", plld(uuid)).check_structure().is_err());
    let mut short = plld(uuid);
    short.truncate(short.len() - 3);
    assert!(TaggedBlock::new(*b"PlLd", short).check_structure().is_err());
    let mut long = sold(uuid);
    long.extend([1, 2, 3, 4, 5]);
    assert!(TaggedBlock::new(*b"SoLd", long).check_structure().is_err());
    // Bad linked-file chain: a length past the end of the block.
    let mut lnk = 100u64.to_be_bytes().to_vec();
    lnk.extend(b"liFD");
    assert!(TaggedBlock::new(*b"lnk2", lnk).check_structure().is_err());
    // Unchecked keys and empty data never panic.
    for key in [b"PlLd", b"SoLd", b"SoLE", b"lnk2", b"lfx2", b"zzzz"] {
        let _ = TaggedBlock::new(*key, Vec::new()).check_structure();
    }
}

#[test]
fn global_blocks_are_padded_to_four() {
    let mut f = testgen::small(Version::Psd, Compression::Raw);
    // A 77-byte global block (Content Credentials `CAI ` has this shape in the wild),
    // stored with Photoshop's 4-byte padding.
    let mut cai = TaggedBlock::new(*b"CAI ", vec![7; 77]);
    cai.padding = Some(vec![0; 3]);
    f.global_blocks.push(cai);
    f.global_blocks.push(TaggedBlock::new(*b"zEnd", vec![1, 2, 3, 4]));
    let out = reexport(&f);
    assert_eq!(common::strict_block_errors(&out), Vec::<String>::new());
    let back = PsdFile::from_bytes(&out).unwrap();
    assert_eq!(back.global_block(b"CAI ").unwrap().data, vec![7; 77]);
    assert!(back.global_block(b"zEnd").is_some());
}

#[test]
fn deep_documents_pad_the_layer_info_block() {
    for depth in [photocraft_color::SampleType::U16, photocraft_color::SampleType::F32] {
        let mut d = photocraft_doc::Document::new("d", photocraft_geom::Size::new(5, 3), photocraft_color::ColorMode::Rgb, depth);
        let fmt = d.pixel_format();
        // Odd sizes so the unpadded layer info length is not a multiple of 4.
        d.layers.push(common::raster("a", fmt, photocraft_geom::Rect::new(0, 0, 5, 3), 1, true));
        d.layers.push(common::raster("b", fmt, photocraft_geom::Rect::new(1, 1, 3, 1), 2, true));
        let out = export(&d, "out.psd", &ExportOptions::default()).unwrap().bytes;
        assert_eq!(common::strict_block_errors(&out), Vec::<String>::new(), "{depth:?}");
    }
}

#[test]
fn odd_layer_blocks_carry_their_pad_inside_the_length() {
    let mut f = testgen::small(Version::Psd, Compression::Raw);
    let mut odd = TaggedBlock::new(*b"zOdd", vec![1, 2, 3]);
    odd.padding = Some(vec![0]);
    f.layers_mut()[0].blocks.push(odd);
    let out = reexport(&f);
    assert_eq!(common::strict_block_errors(&out), Vec::<String>::new());
}

#[test]
fn psb_long_length_keys_are_read_and_written_with_eight_bytes() {
    let mut f = testgen::small(Version::Psb, Compression::Raw);
    let mut cinf = TaggedBlock::new(*b"cinf", {
        let mut d = 16u32.to_be_bytes().to_vec();
        d.extend(Descriptor::new("null").to_bytes());
        d
    });
    cinf.signature = *b"8B64";
    f.global_blocks.push(cinf.clone());
    let bytes = f.to_bytes().unwrap();
    // The length field is 8 bytes: the 4 bytes after the key are the high half (0).
    let at = bytes.windows(8).position(|w| w == b"8B64cinf").unwrap();
    assert_eq!(&bytes[at + 8..at + 16], &(cinf.data.len() as u64).to_be_bytes());
    let back = PsdFile::from_bytes(&bytes).unwrap();
    assert!(back.layer_mask_trailing.is_empty());
    assert_eq!(back.global_block(b"cinf").unwrap().data, cinf.data);
    assert_eq!(back.to_bytes().unwrap(), bytes);
}
