//! Re-render oracle over the Photoshop corpus (`corpus/photoshop`): our own Photoshop-authored
//! PSDs from https://github.com/storytold/photocraft-corpus, fetched at a pinned commit by
//! `cargo xtask corpus --all`. Feature `corpus` (run with `cargo xtask test-corpus`); a missing
//! corpus fails.
//!
//! The io corpus test (`crates/io/tests/corpus.rs`) flattens imported documents, which composite
//! Photoshop's own cached pixels for smart objects and type layers. This test throws those caches
//! away: every smart object is re-rendered from its embedded source through our smart-filter
//! stack, and every type layer is re-laid-out and re-rasterised by our text engine; the flatten is
//! then compared with Photoshop's merged composite. It is the failure map for smart filters and
//! the text engine. Results are printed per file and per feature group; the pass count must not
//! fall below the floor (raise it as features land, never lower it). Type re-renders only match
//! with the corpus fonts installed (see the corpus README), so the floor counts on none of them.
#![cfg(feature = "corpus")]

use std::path::{Path, PathBuf};

use photocraft_doc::{Document, LayerContent};
use photocraft_psd::PsdFile;

/// Within two 8-bit steps of Photoshop (same tolerance as the io corpus).
const PASS_TOL: f32 = 2.0 / 255.0;
/// Re-rendered files that match Photoshop's merged composite (of 86 with smart objects or type).
const PASS_FLOOR: usize = 5;

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect(&p, out);
        } else if p.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("psd")) {
            out.push(p);
        }
    }
}

/// Premultiplied max difference over pixels visible in either image (1.0 if sizes differ).
fn max_diff(a: &[[f32; 4]], b: &[[f32; 4]]) -> (f32, f32) {
    if a.len() != b.len() {
        return (1.0, 100.0);
    }
    let (mut m, mut bad) = (0.0f32, 0usize);
    for (p, q) in a.iter().zip(b) {
        let mut pm = 0.0f32;
        for c in 0..4 {
            let (x, y) = if c < 3 { (p[c] * p[3], q[c] * q[3]) } else { (p[c], q[c]) };
            pm = pm.max((x - y).abs());
        }
        if pm > PASS_TOL {
            bad += 1;
        }
        m = m.max(pm);
    }
    (m, 100.0 * bad as f32 / a.len().max(1) as f32)
}

/// Re-renders every smart object and type layer of `doc`; returns how many were re-rendered.
fn rerender(doc: &mut Document) -> Result<usize, String> {
    let snapshot = doc.clone();
    let ids: Vec<_> = snapshot.walk().into_iter().map(|(_, _, l)| l.id).collect();
    let mut n = 0;
    for id in ids {
        let Some(l) = doc.layer_mut(id) else { continue };
        match &mut l.content {
            LayerContent::Smart(_) => {
                if photocraft_engine::smart_cmds::refresh_layer(&snapshot, l).map_err(|e| e.to_string())? {
                    n += 1;
                } else {
                    return Err("smart object source unavailable".into());
                }
            }
            LayerContent::Text(t) => {
                photocraft_engine::type_cmds::refresh(&snapshot, t);
                n += 1;
            }
            _ => {}
        }
    }
    Ok(n)
}

#[test]
fn photoshop_oracles_rerendered() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/photoshop");
    assert!(root.is_dir(), "{} is missing: run `cargo xtask corpus --all`", root.display());
    let mut files = Vec::new();
    collect(&root, &mut files);
    files.sort();
    let mut groups: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
    let (mut pass, mut total, mut errors) = (0, 0, 0);
    eprintln!("{:<60} {:>7} {:>9} {:>8}  status", "file (re-rendered)", "layers", "max_err", "bad_px%");
    for p in &files {
        let name = p.strip_prefix(&root).unwrap_or(p).display().to_string();
        let bytes = std::fs::read(p).unwrap_or_default();
        let Ok(file) = PsdFile::from_bytes(&bytes) else { continue };
        let Ok(imp) = photocraft_io::import(&name, &bytes) else {
            eprintln!("{name:<60} IMPORT-ERROR");
            errors += 1;
            continue;
        };
        let mut doc = imp.document;
        let n = match rerender(&mut doc) {
            Ok(0) => continue,
            Ok(n) => n,
            Err(e) => {
                eprintln!("{name:<60} RERENDER-ERROR {e}");
                errors += 1;
                continue;
            }
        };
        let Ok(merged) = photocraft_io::merged_composite(&file) else { continue };
        let ours = photocraft_compose::flatten(&doc).px;
        let (m, bad) = max_diff(&ours, &merged);
        let group = name.split(['/', '\\']).next().unwrap_or(".").to_string();
        let g = groups.entry(group).or_default();
        g.1 += 1;
        total += 1;
        let status = if m <= PASS_TOL {
            pass += 1;
            g.0 += 1;
            "PASS"
        } else {
            "DIFF"
        };
        eprintln!("{name:<60} {n:>7} {m:>9.4} {bad:>7.2}%  {status}");
    }
    eprintln!("photoshop re-render: {total} files with smart objects or type: {pass} pass (<= 2/255), {errors} errors");
    for (g, (p, n)) in &groups {
        eprintln!("photoshop re-render:   {g:<16} {p:>4} / {n:<4} pass");
    }
    assert!(pass >= PASS_FLOOR, "re-render oracle pass count {pass} fell below the floor {PASS_FLOOR}");
}
