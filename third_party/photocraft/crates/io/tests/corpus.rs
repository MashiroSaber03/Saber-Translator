//! Real-file corpora (feature `corpus`; run with `cargo xtask test-corpus`). The corpora are
//! gitignored, never committed, and fetched at pinned commits and sha256-verified by
//! `cargo xtask corpus --all` (pins: `xtask/src/corpus_pins.rs`). A missing corpus fails.
//!
//! - `corpus/psd/**/*.{psd,psb}`: a hand-picked mix of small ag-psd and psd-tools samples
//!   (manifest `xtask/psd-corpus.sha256`).
//! - `corpus/psd-tools/**/*.{psd,psb}`: the complete psd-tools test set (MIT).
//! - `corpus/photoshop/**/*.psd`: our own Photoshop-authored oracles from
//!   https://github.com/storytold/photocraft-corpus (smart filters, layer-style effect shapes,
//!   the text engine, adjustments in every mode and depth), reported per feature group. Smart
//!   objects and type layers composite Photoshop's cached pixels here; the engine's
//!   `photoshop_oracles` test re-renders them with our smart-filter stack and text engine.
//!
//! For each file: parse → document → flatten, compared with the file's own
//! merged composite (Photoshop's rendering) as the oracle; then document →
//! PSD → document → flatten, compared with the first flatten (our own export
//! must not change what the document looks like). Prints a per-file table.
//! A panic anywhere is caught and reported as CRASH (Rule 9).
//!
//! Oracles: files without layers (Bitmap, Indexed, plain flat files) compare
//! our import with the merged image as decoded by the psd crate. When
//! Photoshop wrote no real merged image (Maximize Compatibility off: blank
//! image data) or the merged image would decode through our own model code
//! (Multichannel), the embedded thumbnail (resource 1036, Photoshop's own
//! render, JPEG, ≤ 160 px) is the oracle: our composite is box-downsampled to
//! its size and both are lightly blurred; PASS (thumbnail) needs a mean error
//! ≤ 3/255 with ≤ 2 % of pixels off by more than 24/255. That bar is strict
//! (calibrated on files that match their merged image, many thumbnails being
//! stale or resampled differently), so a thumbnail PASS is real evidence while
//! a thumbnail DIFF may be the thumbnail's fault. A file SKIPs only when no
//! oracle exists (no real composite and no usable thumbnail).
//!
//! Differences are expected where features are not yet rendered (effects, text engine, smart
//! filters, knockout); the run asserts no crashes, and that the oracle pass count and the export
//! round-trip count do not fall below the source's floors. Raise the floors when they improve;
//! never lower them.
//! Set `PHOTOCRAFT_CORPUS_STRICT=1` to also fail on import/export errors
//! (files listed in `KNOWN_BAD` excepted).
//!
//! Every exported file must also pass `common::strict_block_errors` (#200): each tagged block
//! re-parses strictly, with the padding Photoshop writes, so other readers (psd-tools) stay aligned.
//!
//! `*_mutations_never_panic` truncates and corrupts every corpus file and
//! asserts import + flatten return (Ok or Err) without panicking.
#![cfg(feature = "corpus")]

mod common;

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::time::Instant;

use photocraft_io::*;
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

const PASS_TOL: f32 = 2.0 / 255.0;
/// Export → re-import must render within one 8-bit step of the imported document.
const ROUNDTRIP_TOL: f32 = 1.0 / 255.0 + 1e-5;

/// Dissolve block size and tolerance (see [`dissolve_matches`]).
const DISSOLVE_BLOCK: i32 = 16;
const DISSOLVE_TOL: f32 = 0.1;

/// Oracle for documents with Dissolve layers. Photoshop's dissolve decides each pixel with a
/// position-only pseudo-random threshold (observed on psd-tools dissolve.psd: where dissolved
/// layers of equal opacity overlap, a pixel shows the top layer or nothing, never a lower one),
/// but its generator is not public and can't be recovered from one binary pattern; ours uses
/// another hash with the same rule. So outside the dissolve layers' bounds pixels must match
/// as usual (≤ 2/255), and inside them the premultiplied colour averaged over 16 × 16 blocks
/// (density and colour) must agree within 0.1 (binomial noise of a 50 % dissolve is about
/// 0.044 per block difference).
fn dissolve_matches(doc: &photocraft_doc::Document, ours: &[[f32; 4]], ps: &[[f32; 4]]) -> bool {
    let canvas = doc.bounds();
    let regions: Vec<_> = doc
        .walk()
        .into_iter()
        .filter(|(_, _, l)| l.visible && l.blend == photocraft_color::BlendMode::Dissolve)
        .map(|(_, _, l)| photocraft_compose::layer_bounds(l, canvas).intersect(&canvas))
        .filter(|r| !r.is_empty())
        .collect();
    let (w, h) = (canvas.width() as i32, canvas.height() as i32);
    if regions.is_empty() || ours.len() != ps.len() || ours.len() != (w as usize) * (h as usize) {
        return false;
    }
    let pm = |p: &[f32; 4], c: usize| if c < 3 { p[c] * p[3] } else { p[3] };
    let inside = |x: i32, y: i32| regions.iter().any(|r| r.contains(x + canvas.x0, y + canvas.y0));
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            if !inside(x, y) && (0..4).any(|c| (pm(&ours[i], c) - pm(&ps[i], c)).abs() > PASS_TOL) {
                return false;
            }
        }
    }
    for by in (0..h).step_by(DISSOLVE_BLOCK as usize) {
        for bx in (0..w).step_by(DISSOLVE_BLOCK as usize) {
            let (x1, y1) = ((bx + DISSOLVE_BLOCK).min(w), (by + DISSOLVE_BLOCK).min(h));
            let n = ((x1 - bx) * (y1 - by)) as f32;
            for c in 0..4 {
                let (mut a, mut b) = (0.0, 0.0);
                for y in by..y1 {
                    for x in bx..x1 {
                        let i = (y * w + x) as usize;
                        a += pm(&ours[i], c);
                        b += pm(&ps[i], c);
                    }
                }
                if ((a - b) / n).abs() > DISSOLVE_TOL {
                    return false;
                }
            }
        }
    }
    true
}

thread_local! {
    /// Files whose export has tagged blocks that fail `common::strict_block_errors`.
    static BLOCK_ERRORS: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Files known to be invalid upstream: their parse error is expected.
const KNOWN_BAD: &[(&str, &str)] = &[("group-divider-blend-mode.psd", "psd-tools fixture stripped to 1906 bytes: merged image data missing")];

/// One corpus directory with its own floors.
struct Source {
    label: &'static str,
    /// Location, relative to the workspace root.
    dir: &'static str,
    /// Path components naming a feature group for the per-group totals (0: no groups).
    group_depth: usize,
    /// Files whose flatten matches their oracle (merged image, or thumbnail when there is none).
    pass_floor: usize,
    /// Files whose export → re-import renders the same as the import.
    roundtrip_floor: usize,
}

/// Hand-picked mix in `corpus/psd` (170 files: 134 vs the merged image + 12 vs the thumbnail).
const MIXED: Source = Source { label: "io corpus", dir: "corpus/psd", group_depth: 0, pass_floor: 146, roundtrip_floor: 169 };

/// The full psd-tools test set (309 files at the pinned commit; see `xtask/psd-tools-corpus.sha256`):
/// 226 vs the merged image + 10 vs the thumbnail (Advanced Blending knockouts: +7).
const PSD_TOOLS: Source = Source { label: "psd-tools corpus", dir: "corpus/psd-tools", group_depth: 0, pass_floor: 236, roundtrip_floor: 308 };

/// Our Photoshop-authored oracles (256 files; https://github.com/storytold/photocraft-corpus),
/// grouped by feature (`smart-filters`, `effects`, `text`, `adjustments/<mode><bits>`).
const PHOTOSHOP: Source = Source { label: "photoshop oracles", dir: "corpus/photoshop", group_depth: 2, pass_floor: 145, roundtrip_floor: 258 };

/// The corpus directory; a missing corpus fails the test.
fn locate(src: &Source) -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(src.dir);
    assert!(root.is_dir(), "{}: {} is missing: run `cargo xtask corpus --all`", src.label, root.display());
    root
}

/// Feature group of a corpus-relative path: its first `depth` directories.
fn group_of(name: &str, depth: usize) -> String {
    let parts: Vec<&str> = name.split(['/', '\\']).collect();
    let n = depth.min(parts.len().saturating_sub(1));
    if n == 0 { ".".into() } else { parts[..n].join("/") }
}

fn files_in(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect(root, &mut files);
    files.sort();
    files
}

fn panic_message(e: &(dyn std::any::Any + Send)) -> String {
    e.downcast_ref::<&str>().map(|s| (*s).to_string()).or_else(|| e.downcast_ref::<String>().cloned()).unwrap_or_else(|| "panic".into())
}

/// Thumbnail oracle: mean absolute error and the share of pixels off by more than
/// `THUMB_BAD`, after box-downsampling our composite (over white) to the JPEG thumbnail's size.
const THUMB_MEAN_TOL: f32 = 3.0 / 255.0;
const THUMB_BAD: f32 = 24.0 / 255.0;
const THUMB_BAD_FRAC: f32 = 0.02;

/// Photoshop's embedded thumbnail (resource 1036: a 28-byte header, then JFIF) as RGB floats.
fn thumbnail(file: &PsdFile) -> Option<(u32, u32, Vec<[f32; 3]>)> {
    let r = file.resource(photocraft_psd::resources::ids::THUMBNAIL)?;
    let format = u32::from_be_bytes(r.data.get(..4)?.try_into().ok()?);
    if format != 1 {
        return None;
    }
    let img = photocraft_codecs::decode(r.data.get(28..)?).ok()?;
    let rgba = img.to_rgba8();
    Some((img.width(), img.height(), rgba.as_chunks::<4>().0.iter().map(|p| [0, 1, 2].map(|c| f32::from(p[c]) / 255.0)).collect()))
}

/// 3×3 box blur of an RGB plane (edges clamp): absorbs the sub-pixel offsets between our box
/// downsampling and Photoshop's thumbnail resampling, and JPEG ringing.
fn blur3(v: &[[f32; 3]], w: usize, h: usize) -> Vec<[f32; 3]> {
    let mut out = vec![[0.0; 3]; v.len()];
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f32; 3];
            for dy in [-1i64, 0, 1] {
                for dx in [-1i64, 0, 1] {
                    let (sx, sy) = ((x as i64 + dx).clamp(0, w as i64 - 1) as usize, (y as i64 + dy).clamp(0, h as i64 - 1) as usize);
                    for c in 0..3 {
                        acc[c] += v[sy * w + sx][c];
                    }
                }
            }
            out[y * w + x] = acc.map(|a| a / 9.0);
        }
    }
    out
}

/// Compares `ours` (canvas-size straight RGBA) with the thumbnail: our composite over white is
/// box-downsampled to the thumbnail's size, both are lightly blurred, and the result is the mean
/// absolute error and the share of pixels off by more than `THUMB_BAD`. `None` without a
/// thumbnail, or when it is not smaller than the canvas (tiny documents: not a real reduction).
fn thumbnail_oracle(file: &PsdFile, ours: &[[f32; 4]]) -> Option<(f32, f32)> {
    let (tw, th, thumb) = thumbnail(file)?;
    let (w, h) = (file.header.width as usize, file.header.height as usize);
    let (tw, th) = (tw as usize, th as usize);
    if tw < 8 || th < 8 || tw > w || th > h || ours.len() != w * h || thumb.len() != tw * th {
        return None;
    }
    let mut small = Vec::with_capacity(tw * th);
    for ty in 0..th {
        for tx in 0..tw {
            let (x0, x1) = (tx * w / tw, ((tx + 1) * w / tw).max(tx * w / tw + 1).min(w));
            let (y0, y1) = (ty * h / th, ((ty + 1) * h / th).max(ty * h / th + 1).min(h));
            let mut acc = [0.0f32; 3];
            for y in y0..y1 {
                for p in &ours[y * w + x0..y * w + x1] {
                    for c in 0..3 {
                        acc[c] += p[c] * p[3] + 1.0 - p[3];
                    }
                }
            }
            let n = ((x1 - x0) * (y1 - y0)) as f32;
            small.push(acc.map(|a| a / n));
        }
    }
    let (a, b) = (blur3(&small, tw, th), blur3(&thumb, tw, th));
    let (mut sum, mut bad) = (0.0f32, 0usize);
    for (p, t) in a.iter().zip(&b) {
        let d: [f32; 3] = std::array::from_fn(|c| (p[c] - t[c]).abs());
        sum += (d[0] + d[1] + d[2]) / 3.0;
        bad += usize::from(d.iter().any(|v| *v > THUMB_BAD));
    }
    let n = (tw * th) as f32;
    Some((sum / n, bad as f32 / n))
}

enum Outcome {
    Pass,
    /// Matched the embedded thumbnail (no real merged image to compare with).
    ThumbPass,
    Diff,
    Skip,
    Error,
}

/// Oracle + round trip for one file. Returns the oracle outcome and whether the export round trip
/// rendered the same (`None` when it could not run).
fn check_file(name: &str, bytes: &[u8]) -> (Outcome, Option<f32>) {
    let known_bad = KNOWN_BAD.iter().any(|(f, _)| name.ends_with(f));
    let file = match PsdFile::from_bytes(bytes) {
        Ok(f) => f,
        Err(e) => {
            let tag = if known_bad { "KNOWN-BAD" } else { "PARSE-ERROR" };
            eprintln!("{name:<60} {:>6} {:>9} {:>8}  {tag} {e}", "-", "-", "-");
            return (if known_bad { Outcome::Skip } else { Outcome::Error }, None);
        }
    };
    let imp = match import(name, bytes) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("{name:<60} IMPORT-ERROR {e}");
            return (Outcome::Error, None);
        }
    };
    let doc = &imp.document;
    // Export must always succeed, re-parse and re-import.
    let reimported = match export(doc, "x.psd", &ExportOptions::default()) {
        Ok(r) => {
            if let Err(e) = PsdFile::from_bytes(&r.bytes) {
                eprintln!("{name:<60} REEXPORT-PARSE-ERROR {e}");
                return (Outcome::Error, None);
            }
            // Every exported tagged block must re-parse strictly (#200).
            let errs = common::strict_block_errors(&r.bytes);
            if !errs.is_empty() {
                eprintln!("{name:<60} EXPORT-BLOCK-ERROR {}", errs.join("; "));
                BLOCK_ERRORS.with(|b| b.borrow_mut().push(format!("{name}: {}", errs.join("; "))));
            }
            match import(name, &r.bytes) {
                Ok(i) => i.document,
                Err(e) => {
                    eprintln!("{name:<60} REIMPORT-ERROR {e}");
                    return (Outcome::Error, None);
                }
            }
        }
        Err(e) => {
            eprintln!("{name:<60} EXPORT-ERROR {e}");
            return (Outcome::Error, None);
        }
    };
    let ours = photocraft_compose::flatten(doc).px;
    // Our own export must not change how the document renders.
    let again = photocraft_compose::flatten(&reimported).px;
    let rt = if again.len() == ours.len() { common::max_diff(&ours, &again) } else { f32::INFINITY };
    let layers = doc.layer_count();
    // Oracle choice: the merged image (Photoshop's composite; for files without layers it is the
    // whole image, decoded independently by the psd crate). When Photoshop wrote no real merged
    // image (Maximize Compatibility off) or it decodes through our own model code (Multichannel),
    // its embedded thumbnail is the oracle at thumbnail size.
    let use_thumb = file.has_real_merged_data() == Some(false) || file.header.color_mode == photocraft_psd::ColorMode::Multichannel;
    let merged = if use_thumb { None } else { merged_composite(&file).ok() };
    let Some(merged) = merged else {
        let notes: Vec<&str> = imp.warnings.iter().map(String::as_str).take(2).collect();
        return match thumbnail_oracle(&file, &ours) {
            Some((mean, frac)) => {
                let ok = mean <= THUMB_MEAN_TOL && frac <= THUMB_BAD_FRAC;
                let status = if ok { "PASS (thumbnail)" } else { "DIFF (thumbnail)" };
                eprintln!("{name:<60} {layers:>6} {mean:>9.4} {:>7.2}%  {status} {}", 100.0 * frac, notes.join(" | "));
                (if ok { Outcome::ThumbPass } else { Outcome::Diff }, Some(rt))
            }
            None => {
                eprintln!("{name:<60} {layers:>6} {:>9} {:>8}  SKIP (no real composite and no thumbnail)", "-", "-");
                (Outcome::Skip, Some(rt))
            }
        };
    };
    // Without an alpha channel in the merged image, Photoshop flattened the document over white.
    let ours = if file.merged_has_alpha() {
        ours
    } else {
        ours.iter()
            .map(|p| {
                let w = 1.0 - p[3];
                [p[0] * p[3] + w, p[1] * p[3] + w, p[2] * p[3] + w, 1.0]
            })
            .collect()
    };
    let m = common::max_diff(&ours, &merged);
    let bad = ours.iter().zip(&merged).filter(|(a, b)| (0..4).any(|c| (a[c] * a[3] - b[c] * b[3]).abs() > PASS_TOL)).count();
    let pct = 100.0 * bad as f32 / ours.len().max(1) as f32;
    let (outcome, status) = if m <= PASS_TOL {
        (Outcome::Pass, "PASS")
    } else if dissolve_matches(doc, &ours, &merged) {
        (Outcome::Pass, "PASS (dissolve metric)")
    } else {
        (Outcome::Diff, "DIFF")
    };
    let notes: Vec<&str> = imp.warnings.iter().map(String::as_str).take(2).collect();
    eprintln!("{name:<60} {layers:>6} {:>9.4} {:>7.2}%  {status} {}", m, pct, notes.join(" | "));
    if std::env::var_os("CORPUS_THUMB_CALIBRATE").is_some()
        && let Some((mean, frac)) = thumbnail_oracle(&file, &ours)
    {
        eprintln!("THUMBCAL {status} {mean:.4} {frac:.4} {name}");
    }
    (outcome, Some(rt))
}

fn run_oracle(src: &Source) {
    let root = locate(src);
    let files = files_in(&root);
    BLOCK_ERRORS.with(|b| b.borrow_mut().clear());
    let mut groups: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
    let (mut pass, mut thumb_pass, mut diff, mut skipped, mut errors) = (0, 0, 0, 0, 0);
    let (mut rt_same, mut rt_diff, mut crashes) = (0, Vec::new(), Vec::new());
    eprintln!("{:<60} {:>6} {:>9} {:>8}  status", "file", "layers", "max_err", "bad_px%");
    for p in &files {
        let name = p.strip_prefix(&root).unwrap_or(p).display().to_string();
        let bytes = std::fs::read(p).unwrap_or_default();
        let (outcome, rt) = match catch_unwind(AssertUnwindSafe(|| check_file(&name, &bytes))) {
            Ok(r) => r,
            Err(e) => {
                let msg = panic_message(e.as_ref());
                eprintln!("{name:<60} CRASH {msg}");
                crashes.push(format!("{name}: {msg}"));
                continue;
            }
        };
        let g = groups.entry(group_of(&name, src.group_depth)).or_default();
        g.1 += 1;
        if matches!(outcome, Outcome::Pass | Outcome::ThumbPass) {
            g.0 += 1;
        }
        match outcome {
            Outcome::Pass => pass += 1,
            Outcome::ThumbPass => thumb_pass += 1,
            Outcome::Diff => diff += 1,
            Outcome::Skip => skipped += 1,
            Outcome::Error => errors += 1,
        }
        match rt {
            Some(rt) if rt <= ROUNDTRIP_TOL => rt_same += 1,
            Some(rt) => rt_diff.push(format!("{name} ({rt:.4})")),
            None => {}
        }
    }
    let label = src.label;
    eprintln!(
        "{label}: {} files: {} pass ({pass} vs the merged image <= 2/255, {thumb_pass} vs the thumbnail), {diff} differ, {skipped} skipped, {errors} errors, {} crashes",
        files.len(),
        pass + thumb_pass,
        crashes.len()
    );
    let pass = pass + thumb_pass;
    if src.group_depth > 0 {
        for (g, (p, n)) in &groups {
            eprintln!("{label}:   {g:<28} {p:>4} / {n:<4} pass");
        }
    }
    eprintln!("{label}: export -> re-import renders the same for {rt_same} files; differs for {}: {}", rt_diff.len(), rt_diff.join(", "));
    let block_errors = BLOCK_ERRORS.with(|b| b.borrow().clone());
    eprintln!("{label}: exported files whose tagged blocks fail the strict re-parse: {}", block_errors.len());
    if std::env::var_os("PHOTOCRAFT_CORPUS_STRICT").is_some() {
        assert_eq!(errors, 0);
    }
    assert!(crashes.is_empty(), "{label}: panics (Rule 9): {crashes:?}");
    assert!(block_errors.is_empty(), "{label}: exported tagged blocks fail the strict re-parse (#200): {block_errors:?}");
    assert!(pass >= src.pass_floor, "{label}: oracle pass count {pass} fell below the floor {}", src.pass_floor);
    assert!(rt_same >= src.roundtrip_floor, "{label}: export round trip {rt_same} fell below the floor {}: {rt_diff:?}", src.roundtrip_floor);
}

/// Deterministic xorshift for the mutation sweep.
fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Truncations and byte corruptions of every corpus file: import (and flatten when the document
/// is small) must return without panicking. Files are mutated in their first 64 KB, where the
/// header, resources and layer records live.
fn run_mutations(src: &Source) {
    let root = locate(src);
    let files = files_in(&root);
    let (mut runs, mut panics, mut slow) = (0usize, Vec::new(), Vec::new());
    for p in &files {
        let name = p.strip_prefix(&root).unwrap_or(p).display().to_string();
        let Ok(bytes) = std::fs::read(p) else { continue };
        let mut seed = 0x9E37_79B9_7F4A_7C15u64 ^ bytes.len() as u64;
        let span = bytes.len().clamp(1, 64 * 1024);
        let mut variants: Vec<(String, Vec<u8>)> = Vec::new();
        for k in 1..8 {
            let cut = bytes.len() * k / 8;
            variants.push((format!("truncate@{cut}"), bytes[..cut].to_vec()));
        }
        for _ in 0..24 {
            let mut m = bytes.clone();
            let flips = 1 + (next(&mut seed) % 4) as usize;
            for _ in 0..flips {
                let at = (next(&mut seed) as usize) % span;
                if let Some(b) = m.get_mut(at) {
                    *b = match next(&mut seed) % 3 {
                        0 => 0xFF,
                        1 => 0x00,
                        _ => *b ^ (1 << (next(&mut seed) % 8)),
                    };
                }
            }
            variants.push(("mutate".into(), m));
        }
        for (what, data) in variants {
            runs += 1;
            let t = Instant::now();
            let r = catch_unwind(AssertUnwindSafe(|| {
                if let Ok(imp) = import(&name, &data) {
                    let b = imp.document.bounds();
                    if i64::from(b.width()) * i64::from(b.height()) <= 4_000_000 {
                        let _ = photocraft_compose::flatten(&imp.document);
                    }
                }
            }));
            let secs = t.elapsed().as_secs_f32();
            if secs > 20.0 {
                slow.push(format!("{name} {what}: {secs:.1}s"));
            }
            if let Err(e) = r {
                let msg = panic_message(e.as_ref());
                eprintln!("CRASH {name} {what}: {msg}");
                panics.push(format!("{name} {what}: {msg}"));
            }
        }
    }
    eprintln!("{}: {runs} mutated imports, {} panics, {} slow (> 20 s): {slow:?}", src.label, panics.len(), slow.len());
    assert!(panics.is_empty(), "{}: mutated files panicked (Rule 9): {panics:?}", src.label);
}

#[test]
fn corpus_import_flatten_oracle() {
    run_oracle(&MIXED);
}

#[test]
fn psd_tools_import_flatten_oracle() {
    run_oracle(&PSD_TOOLS);
}

#[test]
fn photoshop_oracle_corpus() {
    run_oracle(&PHOTOSHOP);
}

#[test]
fn psd_tools_mutations_never_panic() {
    run_mutations(&PSD_TOOLS);
}
