//! Type-layer bytes we write match what Photoshop 2026 writes for the same settings, checked
//! against Photoshop-authored files in `corpus/photoshop` (feature `corpus`; fetched by
//! `cargo xtask corpus --all`, a missing corpus fails):
//! - EngineData `AntiAlias` numbers for every anti-aliasing mode;
//! - the four f32 values that end a `TySh` block (zeros unwarped, text-space bounds warped);
//! - the version info resource (1057).

#![cfg(feature = "corpus")]

use std::path::{Path, PathBuf};

use photocraft_psd::PsdFile;
use photocraft_psd::descriptor::Value as D;
use photocraft_text::engine_data::Value as E;

fn corpus(rel: &str) -> PsdFile {
    let path: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/photoshop").join(rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e} (run `cargo xtask corpus --all`)", path.display()));
    PsdFile::from_bytes(&bytes).unwrap()
}

fn first_tysh(f: &PsdFile) -> Vec<u8> {
    f.layers().iter().find_map(|l| l.block(b"TySh")).map(|b| b.data.clone()).unwrap()
}

fn engine_antialias(tysh: &[u8]) -> Option<f64> {
    let t = photocraft_text::psd::parse_tysh(tysh)?;
    photocraft_text::psd::engine_data(&t.text)?.path(&["EngineDict", "AntiAlias"]).and_then(E::as_f64)
}

fn rect(t: &photocraft_text::psd::TySh, key: &str) -> [f32; 4] {
    let Some(D::Descriptor(d)) = t.text.get(key) else { panic!("no {key}") };
    ["Left", "Top ", "Rght", "Btom"].map(|k| match d.get(k) {
        Some(D::UnitFloat { value, .. }) => *value as f32,
        other => panic!("{k}: {other:?}"),
    })
}

#[test]
fn engine_data_antialias_matches_photoshop() {
    for name in ["none", "crisp", "strong", "smooth", "sharp"] {
        let raw = first_tysh(&corpus(&format!("text/antialias-{name}.psd")));
        let mut layer = photocraft_text::psd::text_layer_from_tysh(&raw, 72.0).unwrap();
        layer.psd_raw = None;
        let ours = photocraft_text::psd::build_tysh(&layer, 72.0, None);
        assert_eq!(engine_antialias(&ours), engine_antialias(&raw), "{name}");
    }
}

#[test]
fn tysh_trailing_bounds_match_photoshop() {
    for (file, warped) in [("text/warp-arc-50.psd", true), ("text/antialias-smooth.psd", false)] {
        let raw = first_tysh(&corpus(file));
        let theirs = photocraft_text::psd::parse_tysh(&raw).unwrap();
        let mut layer = photocraft_text::psd::text_layer_from_tysh(&raw, 72.0).unwrap();
        assert_eq!(layer.warp.is_some(), warped, "{file}");
        layer.psd_raw = None;
        let ours = photocraft_text::psd::parse_tysh(&photocraft_text::psd::build_tysh(&layer, 72.0, Some(rect(&theirs, "bounds")))).unwrap();
        assert_eq!(ours.bounds, theirs.bounds, "{file}");
        if warped {
            assert_eq!(theirs.bounds, rect(&theirs, "bounds"), "{file}");
        } else {
            assert_eq!(theirs.bounds, [0.0; 4], "{file}");
        }
    }
}

#[test]
fn version_info_matches_photoshop() {
    for file in ["text/antialias-smooth.psd", "text/warp-arc-50.psd"] {
        let theirs = corpus(file);
        assert_eq!(photocraft_psd::resources::version_info_resource(true).data, theirs.resource(1057).unwrap().data, "{file}");
    }
}
