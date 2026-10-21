//! EXIF in TIFF: an EXIF block's Exif and GPS directories and IFD0's descriptive tags are
//! written as the TIFF's own directories (IFD0 pointing to them, as cameras and Photoshop write
//! them) and read back into a block.

use photocraft_codecs::*;

/// A classic TIFF stream builder for EXIF blocks: directories of (tag, type, count, value bytes).
struct Block {
    le: bool,
}

type Entry = (u16, u16, u32, Vec<u8>);

impl Block {
    fn u16(&self, v: u16) -> [u8; 2] {
        if self.le { v.to_le_bytes() } else { v.to_be_bytes() }
    }
    fn u32(&self, v: u32) -> [u8; 4] {
        if self.le { v.to_le_bytes() } else { v.to_be_bytes() }
    }
    fn rational(&self, n: u32, d: u32) -> Vec<u8> {
        [self.u32(n), self.u32(d)].concat()
    }
    /// IFD0 (with pointers to `exif` and `gps` appended) followed by the two directories.
    fn build(&self, mut ifd0: Vec<Entry>, exif: Vec<Entry>, gps: Vec<Entry>) -> Vec<u8> {
        let size = |d: &[Entry]| 2 + 12 * d.len() + 4 + d.iter().filter(|e| e.3.len() > 4).map(|e| e.3.len().next_multiple_of(2)).sum::<usize>();
        ifd0.push((34665, 4, 1, vec![0; 4]));
        ifd0.push((34853, 4, 1, vec![0; 4]));
        let at_exif = 8 + size(&ifd0);
        let at_gps = at_exif + size(&exif);
        let n = ifd0.len();
        ifd0[n - 2].3 = self.u32(at_exif as u32).to_vec();
        ifd0[n - 1].3 = self.u32(at_gps as u32).to_vec();
        let mut out = if self.le { b"II".to_vec() } else { b"MM".to_vec() };
        out.extend(self.u16(42));
        out.extend(self.u32(8));
        for d in [ifd0, exif, gps] {
            let start = out.len();
            let mut extra = start + 2 + 12 * d.len() + 4;
            let mut values = Vec::new();
            out.extend(self.u16(d.len() as u16));
            for (tag, ty, count, v) in &d {
                out.extend(self.u16(*tag));
                out.extend(self.u16(*ty));
                out.extend(self.u32(*count));
                if v.len() <= 4 {
                    let mut inline = v.clone();
                    inline.resize(4, 0);
                    out.extend(inline);
                } else {
                    out.extend(self.u32(extra as u32));
                    values.extend(v);
                    if v.len() % 2 == 1 {
                        values.push(0);
                    }
                    extra += v.len().next_multiple_of(2);
                }
            }
            out.extend(self.u32(0));
            out.extend(values);
        }
        out
    }
}

/// The entries (tag, type, count, value bytes in the block's order) of IFD0 and of the
/// directories its tags `34665` and `34853` point to.
fn dirs(b: &[u8]) -> (bool, Vec<Entry>, Vec<Entry>, Vec<Entry>) {
    let le = &b[..2] == b"II";
    let u16 = |at: usize| if le { u16::from_le_bytes([b[at], b[at + 1]]) } else { u16::from_be_bytes([b[at], b[at + 1]]) };
    let u32 = |at: usize| {
        let s = [b[at], b[at + 1], b[at + 2], b[at + 3]];
        if le { u32::from_le_bytes(s) } else { u32::from_be_bytes(s) }
    };
    let size = |ty: u16| match ty {
        3 | 8 => 2,
        4 | 9 | 11 | 13 => 4,
        5 | 10 | 12 => 8,
        _ => 1,
    };
    let read = |at: usize| -> Vec<Entry> {
        (0..usize::from(u16(at)))
            .map(|i| {
                let e = at + 2 + 12 * i;
                let (tag, ty, count) = (u16(e), u16(e + 2), u32(e + 4));
                let len = count as usize * size(ty);
                let at = if len <= 4 { e + 8 } else { u32(e + 8) as usize };
                (tag, ty, count, b[at..at + len].to_vec())
            })
            .collect()
    };
    let ifd0 = read(u32(4) as usize);
    let sub = |tag: u16| ifd0.iter().find(|e| e.0 == tag).map(|e| read(u32_of(&e.3, le) as usize)).unwrap_or_default();
    let (exif, gps) = (sub(34665), sub(34853));
    (le, ifd0.into_iter().filter(|e| !matches!(e.0, 34665 | 34853)).collect(), exif, gps)
}

fn u32_of(v: &[u8], le: bool) -> u32 {
    let s = [v[0], v[1], v[2], v[3]];
    if le { u32::from_le_bytes(s) } else { u32::from_be_bytes(s) }
}

/// The entries of `d` with their values in little-endian order, sorted by tag.
fn normalized(d: &[Entry], le: bool) -> Vec<(u16, u16, u32, Vec<u8>)> {
    let swap = |ty: u16, v: &[u8]| -> Vec<u8> {
        let n = match ty {
            3 | 8 => 2,
            4 | 5 | 9 | 10 | 11 | 13 => 4,
            12 => 8,
            _ => 1,
        };
        if le || n == 1 { v.to_vec() } else { v.chunks(n).flat_map(|c| c.iter().rev().copied()).collect() }
    };
    let mut v: Vec<_> = d.iter().map(|e| (e.0, e.1, e.2, swap(e.1, &e.3))).collect();
    v.sort_by_key(|e| e.0);
    v
}

fn camera_block(le: bool) -> Vec<u8> {
    let b = Block { le };
    let ascii = |s: &str| [s.as_bytes(), b"\0"].concat();
    b.build(
        vec![(271, 2, 4, ascii("Cam")), (272, 2, 9, ascii("Model X1")), (315, 2, 16, ascii("A. Photographer")), (274, 3, 1, b.u16(1).to_vec())],
        vec![
            (33434, 5, 1, b.rational(1, 125)),
            (34855, 3, 1, b.u16(200).to_vec()),
            (36867, 2, 20, ascii("2026:10:13 09:30:00")),
            (37386, 5, 1, b.rational(50, 1)),
            (37500, 7, 6, b"maker!".to_vec()),
        ],
        vec![(1, 2, 2, ascii("N")), (2, 5, 3, [b.rational(51, 1), b.rational(30, 1), b.rational(0, 1)].concat())],
    )
}

#[test]
fn exif_round_trips_through_a_tiff_in_either_byte_order() {
    for le in [true, false] {
        let block = camera_block(le);
        let mut img = Image::from_raw(4, 3, ChannelLayout::Rgb, SampleType::U8, vec![90; 36]).unwrap();
        img.meta.exif = Some(block.clone());
        let tiff = encode(&img, Format::Tiff, &EncodeOptions::default()).unwrap();
        assert!(fidelity_warnings(&img, Format::Tiff).iter().all(|w| *w != FidelityWarning::ExifDropped));
        let back = decode(&tiff).unwrap().meta.exif.expect("EXIF read back");
        let (ble, b0, bexif, bgps) = dirs(&back);
        let (ole, o0, oexif, ogps) = dirs(&block);
        assert_eq!(normalized(&bexif, ble), normalized(&oexif, ole), "Exif directory (le {le})");
        assert_eq!(normalized(&bgps, ble), normalized(&ogps, ole), "GPS directory (le {le})");
        // IFD0's descriptive tags come along; its orientation is the image's own.
        let descriptive: Vec<_> = o0.into_iter().filter(|e| e.0 != 274).collect();
        assert_eq!(normalized(&b0, ble), normalized(&descriptive, ole), "IFD0 (le {le})");
    }
}

#[test]
fn a_tiff_without_exif_directories_reads_no_exif() {
    let mut img = Image::from_raw(2, 2, ChannelLayout::Gray, SampleType::U8, vec![0; 4]).unwrap();
    img.meta.text = vec![("Artist".into(), "A. Photographer".into())];
    let tiff = encode(&img, Format::Tiff, &EncodeOptions::default()).unwrap();
    assert_eq!(decode(&tiff).unwrap().meta.exif, None);
}

#[test]
fn no_exif_is_written_without_embed_metadata() {
    let mut img = Image::from_raw(2, 2, ChannelLayout::Gray, SampleType::U8, vec![0; 4]).unwrap();
    img.meta.exif = Some(camera_block(true));
    let tiff = encode(&img, Format::Tiff, &EncodeOptions { embed_metadata: false, ..Default::default() }).unwrap();
    assert_eq!(decode(&tiff).unwrap().meta.exif, None);
}
