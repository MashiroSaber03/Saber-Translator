//! Cryptomatte (specification 1.2): the hash function is checked against the official
//! test vectors from the reference repository, synthetic files carry ID/coverage streams
//! with header attributes and a manifest, and the info/decode tests check the listing,
//! the manifest round trip and the per-pixel pairs. Malformed inputs must error or
//! degrade, never panic.

use exr::meta::attribute::{AttributeValue, Text};
use exr::prelude::{
    AnyChannel, AnyChannels, Blocks, Compression, Encoding, FlatSamples, ImageAttributes, IntegerBounds, Layer, LayerAttributes, LineOrder, WritableImage,
};
use photocraft_codecs::{Limits, cryptomatte_id, cryptomatte_key, cryptomatte_layers, cryptomatte_preview_color, decode_cryptomatte};

// -------------------------------------------------------------------------------------------
// The official hash test vectors (cryptomatte_utilities_tests.py, CryptoHashing).

#[test]
// The values are quoted verbatim from the reference test suite; their extra digits are
// intentional documentation, not a precision mistake.
#[allow(clippy::excessive_precision)]
fn mm3_hash_matches_the_official_vectors() {
    // The values are the f32 IDs the reference implementation produces (single precision).
    let vectors: &[(&str, f64)] = &[
        ("hello", 6.0705627102400005616e-17),
        ("cube", -4.08461912519e+15),
        ("sphere", 2.79018604383e+15),
        ("plane", 3.66557617593e-11),
        ("равнина", -1.3192631212399999468e-25),
        ("mädchen", 6.2361298211599995797e+25),
    ];
    for (name, want) in vectors {
        let got = cryptomatte_id(name);
        assert_eq!(got.to_bits(), (*want as f32).to_bits(), "ID bits of {name:?}");
    }
}

#[test]
fn every_id_is_a_finite_normal_float() {
    // The exponent fix keeps IDs away from denormals, infinity and NaN, whatever the name.
    for name in ["", "a", "hello", "CryptoAsset", "very long object name with ünïcödé"] {
        let id = cryptomatte_id(name);
        assert!(id.is_finite() && id.abs() >= f32::MIN_POSITIVE, "{name:?} -> {id}");
    }
}

#[test]
fn key_is_seven_hex_digits_of_the_id() {
    let key = cryptomatte_key("CryptoAsset");
    assert_eq!(key.len(), 7);
    assert!(key.chars().all(|c| c.is_ascii_hexdigit()));
    let full = format!("{:08x}", cryptomatte_id("CryptoAsset").to_bits());
    assert_eq!(key, full[..7]);
}

#[test]
fn preview_color_shifts_the_id_bits() {
    let id = cryptomatte_id("cube");
    let [r, g, b] = cryptomatte_preview_color(id);
    assert_eq!((r, g, b), (0.0, f32::from_bits(id.to_bits() << 8), f32::from_bits(id.to_bits() << 16)));
}

// -------------------------------------------------------------------------------------------
// Synthetic Cryptomatte files.

fn text(s: &str) -> Text {
    Text::new_or_panic(s)
}

/// A 4x4 EXR with one Cryptomatte layer `crypto_asset` of two streams: pixel (x, y) holds
/// `cube` (coverage 0.6) and `sphere` (coverage 0.25) where the streams' values say so.
fn cryptomatte_file(name: &str) -> Vec<u8> {
    let (w, h) = (4usize, 4usize);
    let cube = cryptomatte_id("cube");
    let sphere = cryptomatte_id("sphere");
    // Stream 00: cube everywhere at 0.6; stream 01: sphere only in the first column.
    let s0: Vec<f32> = vec![cube; w * h];
    let s0c: Vec<f32> = vec![0.6; w * h];
    let s1: Vec<f32> = (0..w * h).map(|p| if p % w == 0 { sphere } else { 0.0 }).collect();
    let s1c: Vec<f32> = (0..w * h).map(|p| if p % w == 0 { 0.25 } else { 0.0 }).collect();
    let mk = |n: String, v: Vec<f32>| AnyChannel::new(n.as_str(), FlatSamples::F32(v));
    let list = vec![mk(format!("{name}00.red"), s0), mk(format!("{name}00.green"), s0c), mk(format!("{name}01.red"), s1), mk(format!("{name}01.green"), s1c)];
    let key = cryptomatte_key("CryptoAsset"); // attributes always use the canonical name
    let manifest = format!("{{\"cube\":\"{:08x}\",\"sphere\":\"{:08x}\"}}", cube.to_bits(), sphere.to_bits());
    let mut attrs = LayerAttributes::named(name);
    attrs.other.insert(text(&format!("cryptomatte/{key}/name")), AttributeValue::Text(text("CryptoAsset")));
    attrs.other.insert(text(&format!("cryptomatte/{key}/hash")), AttributeValue::Text(text(&key)));
    attrs.other.insert(text(&format!("cryptomatte/{key}/conversion")), AttributeValue::Text(text("uint32_to_float32")));
    attrs.other.insert(text(&format!("cryptomatte/{key}/manifest")), AttributeValue::Text(text(&manifest)));
    let layer = Layer::new(
        (w, h),
        attrs,
        Encoding { compression: Compression::Uncompressed, blocks: Blocks::ScanLines, line_order: LineOrder::Increasing },
        AnyChannels::sort(list.into_iter().collect()),
    );
    let image = exr::image::Image::from_layers(ImageAttributes::new(IntegerBounds::new((0, 0), (w, h))), vec![layer]);
    let mut out = std::io::Cursor::new(Vec::new());
    image.write().to_buffered(&mut out).expect("synthetic cryptomatte exr writes");
    out.into_inner()
}

/// The same layer, but with the short `.r`/`.g` channel suffixes Nuke also matches.
fn cryptomatte_file_short_suffix() -> Vec<u8> {
    let (w, h) = (4usize, 4usize);
    let id = cryptomatte_id("cube");
    let mk = |n: String, v: Vec<f32>| AnyChannel::new(n.as_str(), FlatSamples::F32(v));
    let list = vec![mk("crypto_asset00.r".into(), vec![id; w * h]), mk("crypto_asset00.g".into(), vec![1.0; w * h])];
    let key = cryptomatte_key("CryptoAsset");
    let mut attrs = LayerAttributes::named("crypto_asset");
    attrs.other.insert(text(&format!("cryptomatte/{key}/name")), AttributeValue::Text(text("CryptoAsset")));
    let layer = Layer::new(
        (w, h),
        attrs,
        Encoding { compression: Compression::Uncompressed, blocks: Blocks::ScanLines, line_order: LineOrder::Increasing },
        AnyChannels::sort(list.into_iter().collect()),
    );
    let image = exr::image::Image::from_layers(ImageAttributes::new(IntegerBounds::new((0, 0), (w, h))), vec![layer]);
    let mut out = std::io::Cursor::new(Vec::new());
    image.write().to_buffered(&mut out).expect("synthetic cryptomatte exr writes");
    out.into_inner()
}

#[test]
fn layers_list_name_streams_and_manifest() {
    let bytes = cryptomatte_file("crypto_asset");
    let found = cryptomatte_layers(&bytes, &Limits::default()).unwrap();
    assert_eq!(found.len(), 1);
    let layer = &found[0];
    assert_eq!(layer.part, 0);
    assert_eq!(layer.name, "CryptoAsset");
    assert_eq!(layer.key, cryptomatte_key("CryptoAsset"));
    assert_eq!(layer.conversion.as_deref(), Some("uint32_to_float32"));
    assert_eq!(layer.channels, ["crypto_asset00", "crypto_asset01"]);
    // The manifest maps names to their hash IDs, hex bits -> float value.
    assert_eq!(layer.manifest.len(), 2);
    let cube = cryptomatte_id("cube");
    let sphere = cryptomatte_id("sphere");
    assert!(layer.manifest.contains(&(cube, "cube".to_string())));
    assert!(layer.manifest.contains(&(sphere, "sphere".to_string())));
    assert_eq!(layer.name_of(cube), Some("cube"));
}

#[test]
fn decode_gives_the_pairs_per_pixel_sorted_by_coverage() {
    let bytes = cryptomatte_file("crypto_asset");
    let buf = decode_cryptomatte(&bytes, "CryptoAsset", &Limits::default()).unwrap();
    assert_eq!((buf.width, buf.height), (4, 4));
    let cube = cryptomatte_id("cube");
    let sphere = cryptomatte_id("sphere");
    for y in 0..4 {
        // First column: cube (0.6) over sphere (0.25), sorted by coverage.
        let both = buf.at(0, y);
        assert_eq!(both, &[(cube, 0.6), (sphere, 0.25)], "column 0, row {y}");
        // Other columns: cube only.
        for x in 1..4 {
            assert_eq!(buf.at(x, y), &[(cube, 0.6)], "({x}, {y})");
        }
    }
}

#[test]
fn short_channel_suffixes_decode_too() {
    let bytes = cryptomatte_file_short_suffix();
    let buf = decode_cryptomatte(&bytes, "CryptoAsset", &Limits::default()).unwrap();
    let cube = cryptomatte_id("cube");
    assert_eq!(buf.at(3, 3), &[(cube, 1.0)]);
}

/// A 4x4 layer `crypto` of one RGBA set `crypto00.<r>/<g>/<b>/<a>` with the given suffixes:
/// rank 0 (red/green) holds `cube` at 0.6 everywhere, rank 1 (blue/alpha) holds `sphere`
/// at 0.4 everywhere, the second (ID, coverage) pair the specification stores per set.
fn cryptomatte_file_rgba(suffixes: [&str; 4]) -> Vec<u8> {
    let (w, h) = (4usize, 4usize);
    let [r, g, b, a] = suffixes;
    let mk = |s: &str, v: f32| AnyChannel::new(format!("crypto00.{s}").as_str(), FlatSamples::F32(vec![v; w * h]));
    let list = vec![mk(r, cryptomatte_id("cube")), mk(g, 0.6), mk(b, cryptomatte_id("sphere")), mk(a, 0.4)];
    let key = cryptomatte_key("crypto");
    let mut attrs = LayerAttributes::named("crypto");
    attrs.other.insert(text(&format!("cryptomatte/{key}/name")), AttributeValue::Text(text("crypto")));
    let layer = Layer::new(
        (w, h),
        attrs,
        Encoding { compression: Compression::Uncompressed, blocks: Blocks::ScanLines, line_order: LineOrder::Increasing },
        AnyChannels::sort(list.into_iter().collect()),
    );
    let image = exr::image::Image::from_layers(ImageAttributes::new(IntegerBounds::new((0, 0), (w, h))), vec![layer]);
    let mut out = std::io::Cursor::new(Vec::new());
    image.write().to_buffered(&mut out).expect("synthetic cryptomatte exr writes");
    out.into_inner()
}

#[test]
fn blue_alpha_rank_of_a_set_decodes_too() {
    // Each RGBA set holds two ranks: (red, green) and (blue, alpha). Both must come back.
    let bytes = cryptomatte_file_rgba(["red", "green", "blue", "alpha"]);
    let found = cryptomatte_layers(&bytes, &Limits::default()).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].channels, ["crypto00"], "one set, listed once");
    let buf = decode_cryptomatte(&bytes, "crypto", &Limits::default()).unwrap();
    let (cube, sphere) = (cryptomatte_id("cube"), cryptomatte_id("sphere"));
    for y in 0..4 {
        for x in 0..4 {
            assert_eq!(buf.at(x, y), &[(cube, 0.6), (sphere, 0.4)], "({x}, {y})");
        }
    }
}

#[test]
fn uppercase_channel_suffixes_decode_both_ranks() {
    // Blender and others write `.R/.G/.B/.A`; suffixes match regardless of case.
    for suffixes in [["R", "G", "B", "A"], ["Red", "Green", "Blue", "Alpha"], ["r", "g", "b", "a"]] {
        let bytes = cryptomatte_file_rgba(suffixes);
        let buf = decode_cryptomatte(&bytes, "crypto", &Limits::default()).unwrap();
        let (cube, sphere) = (cryptomatte_id("cube"), cryptomatte_id("sphere"));
        assert_eq!(buf.at(2, 1), &[(cube, 0.6), (sphere, 0.4)], "{suffixes:?}");
    }
}

#[test]
fn unknown_layer_name_is_unsupported() {
    let bytes = cryptomatte_file("crypto_asset");
    let e = decode_cryptomatte(&bytes, "CryptoMaterial", &Limits::default()).unwrap_err();
    assert!(e.to_string().contains("no Cryptomatte layer"), "{e}");
}

#[test]
fn plain_exr_has_no_cryptomatte_layers() {
    // A normal RGB file (the multi-part fixture style) lists nothing.
    let (w, h) = (2usize, 2usize);
    let mk = |n: &str, v: f32| AnyChannel::new(n, FlatSamples::F32(vec![v; w * h]));
    let list = vec![mk("R", 0.5), mk("G", 0.5), mk("B", 0.5)];
    let layer = Layer::new(
        (w, h),
        LayerAttributes::named("beauty"),
        Encoding { compression: Compression::Uncompressed, blocks: Blocks::ScanLines, line_order: LineOrder::Increasing },
        AnyChannels::sort(list.into_iter().collect()),
    );
    let image = exr::image::Image::from_layers(ImageAttributes::new(IntegerBounds::new((0, 0), (w, h))), vec![layer]);
    let mut out = std::io::Cursor::new(Vec::new());
    image.write().to_buffered(&mut out).expect("synthetic exr writes");
    assert!(cryptomatte_layers(&out.into_inner(), &Limits::default()).unwrap().is_empty());
}

#[test]
fn a_broken_manifest_degrades_to_empty_not_a_panic() {
    let mut bytes = cryptomatte_file("crypto_asset");
    // Corrupt the embedded manifest text (keep the attribute length: overwrite with '#').
    let needle = b"{\"cube\":";
    let at = bytes.windows(needle.len()).position(|w| w == needle).expect("manifest json");
    for b in &mut bytes[at..at + 8] {
        *b = b'#';
    }
    let found = cryptomatte_layers(&bytes, &Limits::default()).unwrap();
    assert_eq!(found.len(), 1);
    assert!(found[0].manifest.is_empty(), "a broken manifest yields no entries");
    // The samples still decode.
    let buf = decode_cryptomatte(&bytes, "CryptoAsset", &Limits::default()).unwrap();
    assert_eq!(buf.at(1, 0).len(), 1);
}

#[test]
fn hostile_input_never_panics() {
    let bytes = cryptomatte_file("crypto_asset");
    for end in (8..bytes.len()).step_by(7) {
        let cut = &bytes[..end];
        let _ = cryptomatte_layers(cut, &Limits::default());
        let _ = decode_cryptomatte(cut, "CryptoAsset", &Limits::default());
        let _ = decode_cryptomatte(cut, "nothing", &Limits::default());
    }
    // Tiny limits refuse before allocating.
    let e = cryptomatte_layers(&bytes, &Limits { max_alloc: 8, ..Limits::default() }).unwrap_err();
    assert!(e.to_string().contains("exceed"), "{e}");
}
