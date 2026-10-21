//! Layered TIFF against the real-file corpora (feature `corpus`; `cargo xtask test-corpus`):
//! every PSD that imports is written as a layered TIFF and read back, and the result must equal
//! the PSD round trip of the same document: the same layer structure (count, names, kinds) and
//! the same render (within one 8-bit step). The layers go through the same PSD mapping in both
//! directions, so any difference is a bug in the layer-data transcoder or the TIFF glue, not in
//! the mapping. Documents a TIFF stores flat (a lone Background, Multichannel) only compare the
//! render. A panic anywhere is a failure (Rule 9).
//!
//! The byte-order conversion is exercised for real: the TIFF encoder writes Intel order, so
//! every block of every corpus file (effects, type, smart objects, adjustments, paths, …) is
//! converted to little-endian on the way out and back on the way in. Blocks the transcoder
//! doesn't know are dropped with a warning; the test lists the keys it saw dropped, and fails
//! if any of them is one `photocraft-io` maps (a loss of fidelity we can fix).
#![cfg(feature = "corpus")]

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

use photocraft_doc::{Document, Layer};
use photocraft_io::*;

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect(&p, out);
        } else if p.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("psd") || e.eq_ignore_ascii_case("psb")) {
            out.push(p);
        }
    }
}

fn structure(layers: &[Layer], out: &mut Vec<String>, depth: usize) {
    for l in layers {
        out.push(format!(
            "{}{} [{}] {:?} {:.3} mask={} fx={}",
            " ".repeat(depth),
            l.name,
            l.content.kind_name(),
            l.blend,
            l.opacity,
            l.mask.is_some(),
            l.effects.items.len()
        ));
        if let photocraft_doc::LayerContent::Group(g) = &l.content {
            structure(&g.children, out, depth + 1);
        }
    }
}

fn max_diff(a: &[[f32; 4]], b: &[[f32; 4]]) -> f32 {
    a.iter().zip(b).map(|(p, q)| (0..4).map(|c| (p[c] * p[3] - q[c] * q[3]).abs()).fold(0.0, f32::max)).fold(0.0, f32::max)
}

/// Keys the io crate reads from layer records or global blocks: dropping one of these loses
/// something a user can see.
const MAPPED: &[&[u8; 4]] = &[
    b"lfx2", b"lmfx", b"lrFX", b"TySh", b"SoLd", b"SoLE", b"PlLd", b"vmsk", b"vsms", b"vogk", b"vscg", b"vstk", b"SoCo", b"GdFl", b"PtFl", b"levl", b"curv",
    b"hue2", b"brit", b"thrs", b"post", b"expA", b"vibA", b"blnc", b"mixr", b"grdm", b"phfl", b"selc", b"blwh", b"clrL", b"luni", b"lsct", b"lyid", b"lclr",
    b"iOpa", b"clbl", b"infx", b"knko", b"lspf", b"fxrp", b"brst", b"shmd", b"Patt", b"FEid", b"FXid", b"FMsk", b"artb", b"artd", b"cinf",
];

#[test]
fn corpus_round_trips_through_layered_tiff() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus");
    assert!(root.join("psd").is_dir(), "corpus missing: run `cargo xtask corpus --all`");
    let mut files = Vec::new();
    for sub in ["psd", "psd-tools", "photoshop"] {
        collect(&root.join(sub), &mut files);
    }
    files.sort();
    assert!(!files.is_empty());
    let (mut tested, mut crashes, mut failures) = (0usize, Vec::new(), Vec::new());
    let mut dropped: std::collections::BTreeMap<String, usize> = Default::default();
    for path in &files {
        let name = path.strip_prefix(&root).unwrap_or(path).to_string_lossy().replace('\\', "/");
        let Ok(bytes) = std::fs::read(path) else { continue };
        let Ok(imported) = import(&name, &bytes) else { continue };
        let doc = imported.document;
        // Only modes a TIFF holds; everything else is saved flat by design. Multichannel documents
        // have no layers at all (their channels are the image): nothing here to test.
        if !matches!(doc.pixel_format().mode, photocraft_color::ColorMode::Grayscale | photocraft_color::ColorMode::Rgb | photocraft_color::ColorMode::Cmyk)
            || doc.mode == photocraft_color::ColorMode::Multichannel
        {
            continue;
        }
        if doc.size.width.max(doc.size.height) > 4096 {
            continue;
        }
        tested += 1;
        let res = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
            let psd = export(&doc, "x.psd", &ExportOptions::default()).map_err(|e| format!("psd export: {e}"))?;
            let oracle = import("x.psd", &psd.bytes).map_err(|e| format!("psd re-import: {e}"))?.document;
            let out = export(&doc, "x.tif", &ExportOptions { tiff_layers: true, ..Default::default() }).map_err(|e| format!("export: {e}"))?;
            for w in &out.warnings {
                if let Some(rest) = w.strip_prefix("layer data: ")
                    && let Some((key, _)) = rest.split_once(" block (")
                {
                    *dropped.entry(key.to_string()).or_default() += 1;
                    if MAPPED.iter().any(|k| &key.as_bytes() == k) {
                        return Err(format!("mapped block dropped: {w}"));
                    }
                }
            }
            let back = import("x.tif", &out.bytes).map_err(|e| format!("re-import: {e}"))?.document;
            let flat = !tiff_layers::would_write_layers(&doc);
            let (mut a, mut b) = (Vec::new(), Vec::new());
            structure(&oracle.layers, &mut a, 0);
            structure(&back.layers, &mut b, 0);
            if a != b && !flat {
                return Err(format!("layer structure changed:\n  {}\n  {}", a.join("\n  "), b.join("\n  ")));
            }
            let expected: Document = {
                let mut d = oracle.clone();
                d.channels.clear();
                d
            };
            let x = photocraft_compose::flatten(&expected).px;
            let y = photocraft_compose::flatten(&back).px;
            let diff = max_diff(&x, &y);
            if diff > 1.0 / 255.0 + 1e-5 {
                return Err(format!("render differs by {diff}"));
            }
            Ok(())
        }));
        match res {
            Ok(Ok(())) => {}
            Ok(Err(e)) => failures.push(format!("{name}: {e}")),
            Err(_) => crashes.push(name.clone()),
        }
    }
    println!("layered TIFF round trips: {} tested, {} failed, {} crashed", tested, failures.len(), crashes.len());
    println!("blocks dropped on the little-endian path: {dropped:?}");
    for f in &failures {
        println!("FAIL {f}");
    }
    assert!(crashes.is_empty(), "crashes: {crashes:?}");
    assert!(failures.is_empty(), "{} failures (see above)", failures.len());
    assert!(tested > 100, "only {tested} files tested");
}
