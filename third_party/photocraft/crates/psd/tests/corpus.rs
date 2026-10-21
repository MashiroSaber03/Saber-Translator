//! Corpus test (feature `corpus`; run with `cargo xtask test-corpus`): iterates
//! `corpus/psd/**/*.{psd,psb}` and `corpus/psd-tools/**/*.{psd,psb}` at the workspace root
//! (gitignored, fetched and verified by `cargo xtask corpus --all`). A missing corpus fails.
#![cfg(feature = "corpus")]

use std::path::{Path, PathBuf};

use photocraft_psd::PsdFile;

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

/// Files known to be invalid upstream, with the reason. They are reported but
/// do not fail the test.
const KNOWN_BAD: &[(&str, &str)] = &[("group-divider-blend-mode.psd", "psd-tools fixture stripped to 1906 bytes: merged image data missing")];

#[test]
fn corpus_parse_and_byte_stable() {
    for dir in ["corpus/psd", "corpus/psd-tools"] {
        parse_and_byte_stable(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(dir));
    }
}

fn parse_and_byte_stable(root: &Path) {
    assert!(root.is_dir(), "{} is missing: run `cargo xtask corpus --all`", root.display());
    let mut files = Vec::new();
    collect(root, &mut files);
    files.sort();
    let mut failures = Vec::new();
    for p in &files {
        let name = p.strip_prefix(root).unwrap_or(p).display().to_string();
        let bytes = match std::fs::read(p) {
            Ok(b) => b,
            Err(e) => {
                failures.push(format!("{name}: read error {e}"));
                continue;
            }
        };
        match PsdFile::from_bytes(&bytes) {
            Err(e) => {
                if let Some((_, why)) = KNOWN_BAD.iter().find(|(f, _)| name.ends_with(f)) {
                    eprintln!("KNOWN {name}: parse error: {e} ({why})");
                } else {
                    eprintln!("FAIL  {name}: parse error: {e}");
                    failures.push(format!("{name}: parse: {e}"));
                }
            }
            Ok(f) => match f.to_bytes() {
                Ok(out) if out == bytes => eprintln!("ok    {name} ({} layers)", f.layers().len()),
                Ok(out) => {
                    let first = out.iter().zip(&bytes).position(|(a, b)| a != b).unwrap_or(out.len().min(bytes.len()));
                    eprintln!("DIFF  {name}: first difference at {first} (len {} vs {})", out.len(), bytes.len());
                    failures.push(format!("{name}: not byte stable (first diff at {first})"));
                }
                Err(e) => failures.push(format!("{name}: write: {e}")),
            },
        }
    }
    eprintln!("corpus {}: {} files, {} failures", root.display(), files.len(), failures.len());
    assert!(failures.is_empty(), "corpus failures:\n{}", failures.join("\n"));
}
