//! The parallel RLE/decode paths must not change output bytes: a document exported by this
//! branch re-exports byte-equal after a decode/encode cycle, and matches main's exporter
//! (verified by the corpus round-trip floors in `corpus.rs`; here the byte-equality of the
//! codec layer itself is pinned).

use photocraft_psd::compression::{Compression, PlaneLayout, decode_planes, encode_planes};
use photocraft_psd::header::Version;

#[test]
fn encode_decode_planes_round_trip_exactly() {
    let mut data: u64 = 0x0123_4567_89ab_cdef;
    let mut next = move || {
        data ^= data << 13;
        data ^= data >> 7;
        data ^= data << 17;
        data
    };
    for (version, planes, w, h, depth) in
        [(Version::Psd, 1usize, 640usize, 480usize, 8u16), (Version::Psb, 1, 15000, 10000, 8), (Version::Psd, 4, 513, 65, 8), (Version::Psb, 3, 64, 2049, 16)]
    {
        let layout = PlaneLayout { planes, width: w, height: h, depth, version };
        let n = layout.decoded_len().expect("size");
        let pattern: Vec<u8> = if h % 2 == 0 { vec![42u8; n] } else { (0..n).map(|i| ((i as u64 ^ next()) % 251) as u8).collect() };
        let encoded = encode_planes(Compression::Rle, &pattern, &layout).expect("encode");
        let decoded = decode_planes(Compression::Rle, &encoded, &layout).expect("decode");
        assert_eq!(decoded, pattern, "{version:?} {planes}x{w}x{h} d{depth}");
    }
}
