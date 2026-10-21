//! Local PDN oracles are not bundled. Set PDN_TEST_DIR to a directory containing PDNs and their
//! exported PNGs, then run `cargo test -p photocraft-io --test pdn_corpus -- --ignored --nocapture`.

use std::path::{Path, PathBuf};

fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            files(&path, out);
        } else if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("pdn")) {
            out.push(path);
        }
    }
}

#[test]
#[ignore = "requires PDN_TEST_DIR with real Paint.NET files and matching PNG exports"]
fn real_pdn_layers_and_native_saves() {
    let root = std::env::var_os("PDN_TEST_DIR").expect("set PDN_TEST_DIR");
    let mut paths = Vec::new();
    files(Path::new(&root), &mut paths);
    paths.sort();
    assert!(!paths.is_empty(), "no PDN files found");
    let mut worst = 0.0f32;
    for path in &paths {
        let doc = photocraft_io::import(&path.to_string_lossy(), &std::fs::read(path).unwrap()).unwrap_or_else(|e| panic!("{}: {e}", path.display())).document;
        let flat = photocraft_compose::flatten(&doc);
        let native = photocraft_io::export(&doc, "copy.pcraft", &Default::default()).unwrap();
        let back = photocraft_io::import("copy.pcraft", &native.bytes).unwrap().document;
        assert_eq!(doc.layers, back.layers, "{}", path.display());
        assert_eq!(flat.px, photocraft_compose::flatten(&back).px);
        let png = path.with_extension("png");
        let oracle = photocraft_io::import("oracle.png", &std::fs::read(&png).expect("matching PNG export")).unwrap().document;
        assert_eq!(doc.size, oracle.size);
        let expected = photocraft_compose::flatten(&oracle);
        let error = flat
            .px
            .iter()
            .zip(&expected.px)
            .flat_map(|(a, b)| (0..4).map(move |c| if c == 3 { (a[3] - b[3]).abs() } else { (a[c] * a[3] - b[c] * b[3]).abs() }))
            .fold(0.0f32, f32::max);
        eprintln!("{}: {} layers, max PNG error {:.3}/255", path.display(), doc.layers.len(), error * 255.0);
        worst = worst.max(error);
    }
    // Paint.NET 4.x quantized intermediate layers to 8 bits; our compositor keeps floats.
    assert!(worst <= 3.0 / 255.0, "max error {worst}");
    eprintln!("{} real PDNs passed", paths.len());
}
