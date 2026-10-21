//! CR3 preview extraction never mistakes encoded sensor data for a JPEG.
use photocraft_raw::testgen::cr3_preview;
use photocraft_raw::{RawFormat, embedded_preview, embedded_previews, identify, orientation};

fn jpeg() -> Vec<u8> {
    vec![0xff, 0xd8, 0xff, 0xc0, 0, 11, 8, 0, 20, 0, 32, 1, 1, 0x11, 0, 0xff, 0xd9]
}

fn position(b: &[u8], marker: &[u8]) -> usize {
    b.windows(marker.len()).position(|w| w == marker).unwrap()
}

#[test]
fn full_size_jpeg_and_orientation_are_read_from_their_boxes() {
    let j = jpeg();
    let b = cr3_preview(&j, 6);
    assert_eq!(identify(&b), Some(RawFormat::Cr3));
    let p = embedded_preview(&b).unwrap();
    assert_eq!((p.width, p.height), (32, 20));
    assert_eq!(p.jpeg, j);
    assert_eq!(embedded_previews(&b).len(), 1);
    assert_eq!(orientation(&b), Some(6));
}

#[test]
fn variable_sizes_and_32_bit_chunk_offsets_work() {
    let mut b = cr3_preview(&jpeg(), 1);
    let at = position(&b, b"stsz") + 8;
    // Add a single variable-size entry and repair ancestor lengths/offset.
    let len = u32::from_be_bytes(b[at..at + 4].try_into().unwrap());
    b[at..at + 4].fill(0);
    b.splice(at + 8..at + 8, len.to_be_bytes());
    for marker in [b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsz"] {
        let at = position(&b, marker) - 4;
        let n = u32::from_be_bytes(b[at..at + 4].try_into().unwrap());
        b[at..at + 4].copy_from_slice(&(n + 4).to_be_bytes());
    }
    let at = position(&b, b"co64");
    let off = u64::from_be_bytes(b[at + 12..at + 20].try_into().unwrap()) + 4;
    b[at..at + 4].copy_from_slice(b"stco");
    b[at + 12..at + 16].copy_from_slice(&(off as u32).to_be_bytes());
    assert_eq!(embedded_preview(&b).unwrap().jpeg, jpeg());
}

#[test]
fn broken_tables_and_raw_codec_tracks_are_not_used() {
    let good = cr3_preview(&jpeg(), 1);
    for marker in [b"stsz", b"co64", b"stsc", b"JPEG"] {
        let mut b = good.clone();
        let at = position(&b, marker);
        b[at..at + 4].copy_from_slice(b"xxxx");
        assert!(embedded_preview(&b).is_none());
    }
    let mut b = good.clone();
    let at = position(&b, b"co64") + 12;
    b[at..at + 8].copy_from_slice(&u64::MAX.to_be_bytes());
    assert!(embedded_preview(&b).is_none());
    let mut b = good;
    let at = position(&b, b"stsz") + 12;
    b[at..at + 4].copy_from_slice(&u32::MAX.to_be_bytes());
    assert!(embedded_preview(&b).is_none());
}

#[test]
fn truncations_and_corrupted_box_lengths_never_panic() {
    let b = cr3_preview(&jpeg(), 8);
    for n in 0..b.len() {
        let _ = embedded_previews(&b[..n]);
        let _ = orientation(&b[..n]);
    }
    for marker in [b"ftyp", b"moov", b"trak", b"mdat"] {
        for len in [0u32, 1, 2, 7, u32::MAX] {
            let mut corrupt = b.clone();
            let at = position(&b, marker) - 4;
            corrupt[at..at + 4].copy_from_slice(&len.to_be_bytes());
            let _ = embedded_previews(&corrupt);
        }
    }
}

#[test]
fn extended_mdat_length_is_supported() {
    let mut b = cr3_preview(&jpeg(), 1);
    let at = position(&b, b"mdat") - 4;
    let size = u32::from_be_bytes(b[at..at + 4].try_into().unwrap()) as u64 + 8;
    b[at..at + 4].copy_from_slice(&1u32.to_be_bytes());
    b.splice(at + 8..at + 8, size.to_be_bytes());
    let off = position(&b, b"co64") + 12;
    let n = u64::from_be_bytes(b[off..off + 8].try_into().unwrap()) + 8;
    b[off..off + 8].copy_from_slice(&n.to_be_bytes());
    assert_eq!(embedded_preview(&b).unwrap().jpeg, jpeg());
}

#[test]
fn prvw_is_used_when_no_jpeg_track_exists() {
    fn bx(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        [(body.len() as u32 + 8).to_be_bytes().as_slice(), kind, body].concat()
    }
    let ftyp = bx(b"ftyp", b"crx \0\0\0\x01crx isom");
    let mut body = vec![0; 12];
    body.extend_from_slice(&(jpeg().len() as u32).to_be_bytes());
    body.extend(jpeg());
    let mut uuid = vec![0xea, 0xf4, 0x2b, 0x5e, 0x1c, 0x98, 0x4b, 0x88, 0xb9, 0xfb, 0xb7, 0xdc, 0x40, 0x6e, 0x4d, 0x16];
    uuid.extend([0; 8]);
    uuid.extend(bx(b"PRVW", &body));
    let b = [ftyp, bx(b"uuid", &uuid)].concat();
    assert_eq!(embedded_preview(&b).unwrap().jpeg, jpeg());
    assert_eq!(orientation(&b), None);
}

#[test]
fn deeply_nested_containers_are_bounded() {
    let mut b = vec![0; 8];
    for _ in 0..1000 {
        b = [((b.len() + 8) as u32).to_be_bytes().as_slice(), b"moov", &b].concat();
    }
    let ftyp = [24u32.to_be_bytes().as_slice(), b"ftypcrx \0\0\0\x01crx isom"].concat();
    assert!(embedded_previews(&[ftyp, b].concat()).is_empty());
}
