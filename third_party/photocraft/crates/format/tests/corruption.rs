//! Damaged or hostile bundles must fail cleanly, never panic.

mod common;
use common::*;
use photocraft_color::{ColorMode, SampleType};
use photocraft_format::*;
use proptest::prelude::*;
use std::io::Read;

fn sample() -> Vec<u8> {
    save_to_bytes(&rich_doc(ColorMode::Rgb, SampleType::U8), &SaveOptions::default()).unwrap()
}

fn rebuild(bytes: &[u8], f: impl Fn(&str, Vec<u8>) -> Option<Vec<u8>>) -> Vec<u8> {
    let r = zip::ZipReader::new(bytes).unwrap();
    let mut z = zip::ZipWriter::new();
    for e in &r.entries {
        if let Some(d) = f(&e.name, r.read(e, usize::MAX).unwrap()) {
            z.add(&e.name, &d).unwrap();
        }
    }
    z.finish().unwrap()
}

#[test]
fn older_bundles_without_text_metadata_still_load() {
    let mut doc = rich_doc(ColorMode::Rgb, SampleType::U8);
    let bytes = save_to_bytes(&doc, &SaveOptions::default()).unwrap();
    let old = rebuild(&bytes, |name, data| {
        if name != "manifest.json" {
            return Some(data);
        }
        let mut manifest: serde_json::Value = serde_json::from_slice(&data).unwrap();
        assert!(manifest["document"]["metadata"].as_object_mut().unwrap().remove("text").is_some());
        Some(serde_json::to_vec(&manifest).unwrap())
    });
    doc.metadata.text.clear();
    assert_eq!(load_from_bytes(&old).unwrap(), doc);
}

#[test]
fn truncation_never_panics_and_fails() {
    let b = sample();
    let step = (b.len() / 200).max(1);
    for cut in (0..b.len()).step_by(step) {
        assert!(load_from_bytes(&b[..cut]).is_err(), "cut at {cut}");
    }
}

#[test]
fn byte_flips_detected() {
    let b = sample();
    let mut rng = common::Rng::new(9);
    let mut failures = 0;
    for _ in 0..300 {
        let mut c = b.clone();
        let i = (rng.next() as usize) % c.len();
        c[i] ^= 0x20;
        if load_from_bytes(&c).is_err() {
            failures += 1;
        }
    }
    // Some flips hit unused header fields; most must be caught (CRC + hashes).
    assert!(failures > 200, "{failures}");
}

#[test]
fn missing_tile_is_error() {
    let b = rebuild(&sample(), |n, d| (!n.starts_with("tiles/")).then_some(d));
    assert!(matches!(load_from_bytes(&b), Err(FormatError::Corrupt(_))));
}

#[test]
fn missing_manifest_is_error() {
    let b = rebuild(&sample(), |n, d| (n != "manifest.json").then_some(d));
    assert!(load_from_bytes(&b).is_err());
}

#[test]
fn swapped_tile_content_fails_hash_check() {
    let b = sample();
    let r = zip::ZipReader::new(&b).unwrap();
    let tiles: Vec<_> = r.entries.iter().filter(|e| e.name.starts_with("tiles/")).map(|e| e.name.clone()).collect();
    let other = r.read_by_name(&tiles[1], usize::MAX).unwrap();
    let first = tiles[0].clone();
    let b2 = rebuild(&b, |n, d| Some(if n == first { other.clone() } else { d }));
    let e = load_from_bytes(&b2).unwrap_err();
    assert!(e.to_string().contains("hash"), "{e}");
}

#[test]
fn bad_manifest_json() {
    let b = rebuild(&sample(), |n, d| Some(if n == "manifest.json" { b"{not json".to_vec() } else { d }));
    assert!(matches!(load_from_bytes(&b), Err(FormatError::Json(_))));
}

/// Hostile nesting fails with an error instead of overflowing the stack; nesting as deep as the
/// deepest groups a bundle may hold (three JSON levels each) parses, and is then rejected for
/// lacking a format version.
#[test]
fn over_deep_manifest_fails_cleanly() {
    let nested = |n: usize| format!("{}{}", "[".repeat(n), "]".repeat(n)).into_bytes();
    let with_manifest = |m: Vec<u8>| rebuild(&sample(), |n, d| Some(if n == "manifest.json" { m.clone() } else { d }));
    let e = load_from_bytes(&with_manifest(nested(100_000))).unwrap_err();
    assert!(matches!(e, FormatError::LimitExceeded(_)), "{e}");
    let e = load_from_bytes(&with_manifest(nested(3 * MAX_GROUP_DEPTH))).unwrap_err();
    assert!(matches!(e, FormatError::Corrupt(_)), "{e}");
    // Brackets inside strings (escaped quotes included) don't count.
    let e = load_from_bytes(&with_manifest(format!(r#"{{"x":"{}\"{}"}}"#, "[".repeat(100_000), "{".repeat(10)).into_bytes())).unwrap_err();
    assert!(matches!(e, FormatError::Corrupt(_)), "{e}");
}

#[test]
fn too_new_version_rejected() {
    let b = rebuild(&sample(), |n, d| {
        Some(if n == "manifest.json" {
            let mut v: serde_json::Value = serde_json::from_slice(&d).unwrap();
            v["format_version"] = 999.into();
            serde_json::to_vec(&v).unwrap()
        } else {
            d
        })
    });
    assert!(matches!(load_from_bytes(&b), Err(FormatError::TooNew { found: 999, .. })));
}

#[test]
fn invalid_hash_strings_rejected() {
    let b = rebuild(&sample(), |n, d| {
        Some(if n == "manifest.json" {
            let s = String::from_utf8(d).unwrap();
            let i = s.find("\"hash\": \"").unwrap() + 9;
            let mut s = s.into_bytes();
            s[i] = b'/';
            s
        } else {
            d
        })
    });
    assert!(load_from_bytes(&b).is_err());
}

#[test]
fn manifest_size_limit() {
    let b = sample();
    let opts = LoadOptions { max_manifest_bytes: 100, ..Default::default() };
    assert!(matches!(load_from_bytes_with(&b, &opts), Err(FormatError::LimitExceeded(_))));
}

#[test]
fn total_size_limit() {
    let b = sample();
    let opts = LoadOptions { max_total_bytes: 1000, ..Default::default() };
    assert!(matches!(load_from_bytes_with(&b, &opts), Err(FormatError::LimitExceeded(_))));
}

/// A hostile manifest may name one stored tile at very many coordinates. The load decodes and
/// charges it once, each reference only clones the shared tile, and the budget still bounds the
/// distinct bytes (#3315).
#[test]
fn many_references_to_one_tile_load_within_the_budget() {
    let mut doc = photocraft_doc::Document::new("d", photocraft_doc::Size::new(256, 256), ColorMode::Rgb, SampleType::U8);
    let mut l = photocraft_doc::Layer::raster("solid", doc.pixel_format());
    l.surface_mut().unwrap().fill_rect(photocraft_geom::Rect::new(0, 0, 256, 256), &[1.0, 0.0, 0.0, 1.0]);
    doc.layers.push(l);
    let bytes = save_to_bytes(&doc, &SaveOptions::default()).unwrap();
    let hostile = rebuild(&bytes, |name, data| {
        if name != "manifest.json" {
            return Some(data);
        }
        let mut m: serde_json::Value = serde_json::from_slice(&data).unwrap();
        let tiles = m["document"]["layers"][0]["content"]["surface"]["tiles"].as_array_mut().unwrap();
        let hash = tiles[0]["hash"].clone();
        *tiles = (0..256 * 256).map(|i| serde_json::json!({ "tx": i % 256, "ty": i / 256, "hash": hash })).collect();
        Some(serde_json::to_vec(&m).unwrap())
    });
    let one_tile = 256 * 256 * 4;
    let back = load_from_bytes_with(&hostile, &LoadOptions { max_total_bytes: one_tile, ..Default::default() }).unwrap();
    let s = back.layers[0].surface().unwrap();
    assert_eq!(s.tile_count(), 256 * 256);
    let first = s.tiles().next().unwrap().1;
    assert!(s.tiles().all(|(_, t)| std::sync::Arc::ptr_eq(t, first)));
    // The one distinct tile is still charged against the budget.
    let opts = LoadOptions { max_total_bytes: one_tile - 1, ..Default::default() };
    assert!(matches!(load_from_bytes_with(&hostile, &opts), Err(FormatError::LimitExceeded(_))));
}

#[test]
fn directory_bundle_resave_repairs_tampered_objects() {
    let doc = rich_doc(ColorMode::Rgb, SampleType::U8);
    let dir = temp_dir("tamper");
    let mut writer = PcraftWriter::new();
    writer.save_dir(&doc, &dir, &SaveOptions::default()).unwrap();

    for object_dir in ["tiles", "blobs"] {
        let object = std::fs::read_dir(dir.join(object_dir)).unwrap().next().unwrap().unwrap().path();
        let original = std::fs::read(&object).unwrap();
        let mut decoder = ruzstd::decoding::StreamingDecoder::new(&original[..]).unwrap();
        let mut decoded = Vec::new();
        decoder.read_to_end(&mut decoded).unwrap();
        if let Some(first) = decoded.first_mut() {
            *first ^= 1;
        } else {
            decoded.push(1);
        }
        let hash_mismatched = ruzstd::encoding::compress_to_vec(decoded.as_slice(), ruzstd::encoding::CompressionLevel::Fastest);

        for damaged in [hash_mismatched, b"garbage".to_vec()] {
            std::fs::write(&object, damaged).unwrap();
            assert!(load_path(&dir).is_err());

            let stats = writer.save_dir(&doc, &dir, &SaveOptions::default()).unwrap();
            if object_dir == "tiles" {
                assert_eq!(stats.tiles_written, 1);
            } else {
                assert_eq!(stats.blobs_written, 1);
            }
            assert_eq!(load_path(&dir).unwrap(), doc);
        }
    }

    std::fs::remove_dir_all(dir).unwrap();
}

/// Damages one object in place without changing its length or modification time, the case a
/// size-and-mtime cache alone would trust. Returns the damaged file's path.
fn damage_in_place_keeping_size_and_mtime(dir: &std::path::Path) -> std::path::PathBuf {
    let object = std::fs::read_dir(dir.join("tiles")).unwrap().next().unwrap().unwrap().path();
    let original = std::fs::read(&object).unwrap();
    let mtime = std::fs::metadata(&object).unwrap().modified().unwrap();
    // Flip one byte of the compressed payload; try positions until the bundle no longer loads.
    for at in (original.len() / 2..original.len()).chain(0..original.len() / 2) {
        let mut damaged = original.clone();
        damaged[at] ^= 0x5a;
        std::fs::write(&object, &damaged).unwrap();
        std::fs::File::options().write(true).open(&object).unwrap().set_modified(mtime).unwrap();
        if load_path(dir).is_err() {
            assert_eq!(std::fs::metadata(&object).unwrap().len(), original.len() as u64);
            assert_eq!(std::fs::metadata(&object).unwrap().modified().unwrap(), mtime);
            return object;
        }
    }
    panic!("no single-byte change made the bundle fail to load");
}

#[test]
fn same_size_same_mtime_tamper_is_repaired_on_the_next_save() {
    let doc = rich_doc(ColorMode::Rgb, SampleType::U8);
    let dir = temp_dir("tamper-same-size");
    let mut writer = PcraftWriter::new();
    writer.save_dir(&doc, &dir, &SaveOptions::default()).unwrap();
    // A second save trusts every object it verified in the first one.
    writer.save_dir(&doc, &dir, &SaveOptions::default()).unwrap();

    damage_in_place_keeping_size_and_mtime(&dir);
    // The sample bundle is far below DEFAULT_REVERIFY_BUDGET, so the rolling re-check reads
    // every cached object again (and on Unix the change time differs as well).
    let stats = writer.save_dir(&doc, &dir, &SaveOptions::default()).unwrap();
    assert_eq!(stats.tiles_written, 1);
    assert_eq!(load_path(&dir).unwrap(), doc);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn rolling_reverify_reaches_every_object_with_a_tiny_budget() {
    let doc = rich_doc(ColorMode::Rgb, SampleType::U8);
    let dir = temp_dir("tamper-rolling");
    let mut writer = PcraftWriter::new();
    writer.save_dir(&doc, &dir, &SaveOptions::default()).unwrap();
    let objects = ["tiles", "blobs"].iter().map(|d| std::fs::read_dir(dir.join(d)).unwrap().count()).sum::<usize>();

    // One object per save: the oldest-checked one.
    writer.set_reverify_budget(1);
    damage_in_place_keeping_size_and_mtime(&dir);
    let mut repaired = false;
    for _ in 0..=objects {
        let stats = writer.save_dir(&doc, &dir, &SaveOptions::default()).unwrap();
        if stats.tiles_written == 1 {
            repaired = true;
            break;
        }
    }
    assert!(repaired, "a damaged object was not re-verified within {objects} saves");
    assert_eq!(load_path(&dir).unwrap(), doc);

    // A zero budget re-reads nothing whose signature is unchanged; a fresh writer still verifies
    // everything on its first save.
    let mut fresh = PcraftWriter::new();
    fresh.set_reverify_budget(0);
    damage_in_place_keeping_size_and_mtime(&dir);
    assert_eq!(fresh.save_dir(&doc, &dir, &SaveOptions::default()).unwrap().tiles_written, 1);
    assert_eq!(load_path(&dir).unwrap(), doc);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn deflated_zip_entries_are_readable() {
    // Re-zip with DEFLATE (as a user's zip tool might) — must still load.
    use std::io::Write;
    let b = sample();
    let r = zip::ZipReader::new(&b).unwrap();
    let mut out = Vec::new();
    let mut central = Vec::new();
    for e in &r.entries {
        let data = r.read(e, usize::MAX).unwrap();
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(&data).unwrap();
        let comp = enc.finish().unwrap();
        let crc = zip::crc32(&data);
        let off = out.len() as u32;
        let name = e.name.as_bytes();
        out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        out.extend_from_slice(&[20, 0, 0, 0, 8, 0, 0, 0, 0x21, 0]);
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&(comp.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&[0, 0]);
        out.extend_from_slice(name);
        out.extend_from_slice(&comp);
        central.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        central.extend_from_slice(&[20, 0, 20, 0, 0, 0, 8, 0, 0, 0, 0x21, 0]);
        central.extend_from_slice(&crc.to_le_bytes());
        central.extend_from_slice(&(comp.len() as u32).to_le_bytes());
        central.extend_from_slice(&(data.len() as u32).to_le_bytes());
        central.extend_from_slice(&(name.len() as u16).to_le_bytes());
        central.extend_from_slice(&[0; 12]);
        central.extend_from_slice(&off.to_le_bytes());
        central.extend_from_slice(name);
    }
    let cd_off = out.len() as u32;
    let n = r.entries.len() as u16;
    out.extend_from_slice(&central);
    out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    out.extend_from_slice(&[0; 4]);
    out.extend_from_slice(&n.to_le_bytes());
    out.extend_from_slice(&n.to_le_bytes());
    out.extend_from_slice(&(central.len() as u32).to_le_bytes());
    out.extend_from_slice(&cd_off.to_le_bytes());
    out.extend_from_slice(&[0, 0]);
    assert_eq!(load_from_bytes(&out).unwrap(), rich_doc_ids_from(&b));
}

fn rich_doc_ids_from(b: &[u8]) -> photocraft_doc::Document {
    load_from_bytes(b).unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 300, .. ProptestConfig::default() })]

    #[test]
    fn random_bytes_never_panic(data in proptest::collection::vec(any::<u8>(), 0..4096)) {
        let _ = load_from_bytes(&data);
        let _ = read_manifest(&data);
    }

    #[test]
    fn random_manifest_never_panics(json in "\\PC{0,300}") {
        let mut z = zip::ZipWriter::new();
        z.add("manifest.json", json.as_bytes()).unwrap();
        let _ = load_from_bytes(&z.finish().unwrap());
    }

    #[test]
    fn random_tile_payload_never_panics(data in proptest::collection::vec(any::<u8>(), 0..512), which in 0usize..8) {
        let b = sample_cached();
        let r = zip::ZipReader::new(b).unwrap();
        let tiles: Vec<String> = r.entries.iter().filter(|e| e.name.starts_with("tiles/")).map(|e| e.name.clone()).collect();
        let target = tiles[which % tiles.len()].clone();
        let b2 = rebuild(b, |n, d| Some(if n == target { data.clone() } else { d }));
        prop_assert!(load_from_bytes(&b2).is_err());
    }
}

fn sample_cached() -> &'static [u8] {
    static S: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    S.get_or_init(sample)
}

/// Moves the first tile of the first top-level layer that has pixels to `(tx, ty)`; returns that
/// layer's index.
fn move_first_tile(v: &mut serde_json::Value, tx: i64, ty: i64) -> Option<usize> {
    let layers = v["document"]["layers"].as_array_mut()?;
    let (i, tile) = layers.iter_mut().enumerate().find_map(|(i, l)| l["content"]["surface"]["tiles"].get_mut(0).map(|t| (i, t)))?;
    tile["tx"] = tx.into();
    tile["ty"] = ty.into();
    Some(i)
}

#[test]
fn tile_coordinates_outside_the_i32_pixel_range_are_rejected() {
    // #938: a tile index whose rectangle overflows i32 loaded fine and panicked later, in
    // `TileCoord::rect`, wherever the layer's bounds were asked for (wrapping in release).
    let with_tile_at = |tx: i64, ty: i64| {
        let layer = std::cell::Cell::new(None);
        let b = rebuild(&sample(), |n, d| {
            Some(if n == "manifest.json" {
                let mut v: serde_json::Value = serde_json::from_slice(&d).unwrap();
                layer.set(move_first_tile(&mut v, tx, ty));
                serde_json::to_vec(&v).unwrap()
            } else {
                d
            })
        });
        (b, layer.get().expect("the sample has a raster layer with a tile"))
    };
    let (lo, hi) = (i64::from(i32::MIN / 256), i64::from(i32::MAX / 256 - 1));
    for (tx, ty) in [(hi + 1, 0), (0, hi + 1), (lo - 1, 0), (0, lo - 1), (i64::from(i32::MAX), 0), (0, i64::from(i32::MIN))] {
        let err = load_from_bytes(&with_tile_at(tx, ty).0).unwrap_err();
        assert!(err.to_string().contains("outside the coordinate range"), "({tx}, {ty}): {err}");
    }
    // The extreme valid indices still load, and the layer's bounds can be computed.
    for (tx, ty) in [(hi, hi), (lo, lo), (hi, lo)] {
        let (b, i) = with_tile_at(tx, ty);
        let doc = load_from_bytes(&b).unwrap_or_else(|e| panic!("({tx}, {ty}): {e}"));
        let s = doc.layers[i].surface().expect("raster layer");
        assert!(s.tiles().any(|(c, _)| (i64::from(c.tx), i64::from(c.ty)) == (tx, ty)), "({tx}, {ty})");
        assert!(!s.content_bounds().is_empty());
    }
}

#[test]
fn an_unlimited_blob_budget_still_loads_blobs() {
    // `max_blob_bytes + 1` overflowed (debug panic; release read zero bytes, failing every blob).
    let doc = rich_doc(ColorMode::Rgb, SampleType::U8);
    let bytes = save_to_bytes(&doc, &SaveOptions::default()).unwrap();
    let opts = LoadOptions { max_blob_bytes: usize::MAX, ..LoadOptions::default() };
    assert_eq!(load_from_bytes_with(&bytes, &opts).unwrap(), doc);
}

/// Rewrites the first pattern's declared size (its tiles stay as saved).
fn with_pattern_size(w: u64, h: u64) -> Vec<u8> {
    rebuild(sample_cached(), |n, d| {
        if n != "manifest.json" {
            return Some(d);
        }
        let mut v: serde_json::Value = serde_json::from_slice(&d).unwrap();
        let p = &mut v["document"]["patterns"][0];
        assert!(p.is_object(), "the sample has a pattern");
        p["width"] = w.into();
        p["height"] = h.into();
        Some(serde_json::to_vec(&v).unwrap())
    })
}

#[test]
fn pattern_sizes_from_the_file_are_checked() {
    // A pattern can claim any size without tiles; painting it allocated per pixel (abort).
    let huge = load_from_bytes(&with_pattern_size(65_535, 65_535));
    assert!(matches!(huge, Err(FormatError::LimitExceeded(_))), "{huge:?}");
    let empty = load_from_bytes(&with_pattern_size(0, 24));
    assert!(matches!(empty, Err(FormatError::Corrupt(_))), "{empty:?}");
    assert!(load_from_bytes(&with_pattern_size(32, 24)).is_ok());
}

/// A canvas is a few bytes of manifest JSON; one past the budget is refused before any work (#2935).
/// The default budget is the largest canvas the app creates (300000 × 300000, as `file.new`,
/// Canvas Size and PSB import allow), so everything a user can save reopens.
#[test]
fn declared_canvas_outside_the_budget_is_refused() {
    let with_size = |w: u64, h: u64| {
        rebuild(&sample(), |name, data| {
            if name != "manifest.json" {
                return Some(data);
            }
            let mut manifest: serde_json::Value = serde_json::from_slice(&data).unwrap();
            manifest["document"]["size"] = serde_json::json!({ "width": w, "height": h });
            Some(serde_json::to_vec(&manifest).unwrap())
        })
    };
    for (w, h) in [(300_001, 1), (1, 300_001), (300_001, 300_001), (u64::from(u32::MAX), u64::from(u32::MAX))] {
        let r = load_from_bytes(&with_size(w, h)).map(|d| d.size);
        assert!(matches!(r, Err(FormatError::LimitExceeded(_))), "{w}x{h}: {r:?}");
    }
    assert!(load_from_bytes(&with_size(0, 64)).is_err());
    // The largest creatable canvas loads, without allocating anything for its area: the layers
    // keep only the tiles the bundle holds.
    let edge = with_size(300_000, 300_000);
    let small = load_from_bytes(&sample()).unwrap();
    let big = load_from_bytes(&edge).unwrap();
    assert_eq!(big.size, photocraft_geom::Size::new(300_000, 300_000));
    let tiles = |d: &photocraft_doc::Document| d.layers.iter().filter_map(|l| l.surface()).map(|s| s.tile_count()).sum::<usize>();
    assert_eq!(tiles(&big), tiles(&small));
    // A caller may lower the budget.
    let opts = LoadOptions { max_canvas_pixels: 1 << 20, ..Default::default() };
    assert!(matches!(load_from_bytes_with(&edge, &opts), Err(FormatError::LimitExceeded(_))));
    let opts = LoadOptions { max_canvas_side: 65_536, ..Default::default() };
    assert!(matches!(load_from_bytes_with(&with_size(65_537, 1), &opts), Err(FormatError::LimitExceeded(_))));
}
