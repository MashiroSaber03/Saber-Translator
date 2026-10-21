//! `cargo xtask scorecard [--check]`: generate `docs/scorecard.md` (#221).
//!
//! Sources, all committed, so the output is deterministic and CI can check it:
//! - `perf/budgets.toml` + `perf/baseline.json`: the performance table (the baseline's full-run
//!   numbers against the budgets). A fresh `cargo xtask perf` result in `target/` is not used:
//!   publish it with `cargo xtask perf --update-baseline`.
//! - `crates/io/tests/corpus.rs`: the corpus floors (`Source { .. }` constants).
//! - `scorecard/*.toml`: per-area checklists (id, target, status, issue, note).
//! - the prefs audit: `Preferences` fields that no code outside `prefs.rs` / `prefs_ui.rs` reads.
//! - counted from the tree: never-crash attribute coverage, `docs/parity-checklist.md`'s live count.
//!
//! `--check` fails when `docs/scorecard.md` differs from what would be generated.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::Value;

use crate::perf;

pub const OUT: &str = "docs/scorecard.md";
const CORPUS_RS: &str = "crates/io/tests/corpus.rs";
const PREFS_RS: &str = "crates/engine/src/prefs.rs";

// ---- checklists ------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ItemStatus {
    Done,
    Partial,
    Missing,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Item {
    pub id: String,
    pub target: String,
    pub status: ItemStatus,
    pub issue: Option<u32>,
    pub note: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Checklist {
    pub area: String,
    pub title: String,
    #[serde(default)]
    pub order: u32,
    #[serde(default)]
    pub item: Vec<Item>,
}

impl Checklist {
    pub fn counts(&self) -> (usize, usize, usize) {
        let n = |s| self.item.iter().filter(|i| i.status == s).count();
        (n(ItemStatus::Done), n(ItemStatus::Partial), n(ItemStatus::Missing))
    }
}

pub fn parse_checklist(name: &str, text: &str) -> Result<Checklist, String> {
    let c: Checklist = toml::from_str(text).map_err(|e| format!("{name}: {e}"))?;
    for i in &c.item {
        if i.id.trim().is_empty() || i.target.trim().is_empty() {
            return Err(format!("{name}: every item needs an id and a target"));
        }
        if i.target.contains('|') || i.note.as_deref().is_some_and(|n| n.contains('|')) {
            return Err(format!("{name}: {}: `|` breaks the Markdown table", i.id));
        }
    }
    Ok(c)
}

fn load_checklists(root: &Path) -> Result<Vec<Checklist>, String> {
    let dir = root.join("scorecard");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| format!("read {}: {e}", dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "toml"))
        .collect();
    files.sort();
    let mut out = Vec::new();
    let mut ids = BTreeSet::new();
    for f in files {
        let name = format!("scorecard/{}", f.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
        let text = std::fs::read_to_string(&f).map_err(|e| format!("read {name}: {e}"))?;
        let c = parse_checklist(&name, &text)?;
        for i in &c.item {
            if !ids.insert(i.id.clone()) {
                return Err(format!("{name}: duplicate item id {}", i.id));
            }
        }
        out.push(c);
    }
    out.sort_by(|a, b| a.order.cmp(&b.order).then_with(|| a.area.cmp(&b.area)));
    Ok(out)
}

// ---- corpus floors ---------------------------------------------------------------------------

#[derive(Debug, Default, PartialEq)]
pub struct Corpus {
    pub label: String,
    pub dir: String,
    pub files: Option<u32>,
    pub pass_floor: Option<u32>,
    pub roundtrip_floor: Option<u32>,
    pub doc: String,
}

/// Every `const X: Source = Source { .. };` in `corpus.rs`, with its doc comment. Fields are
/// read by name, so new fields or new sources (more corpora) need no change here.
pub fn parse_corpus(src: &str) -> Vec<Corpus> {
    let lines: Vec<&str> = src.lines().collect();
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim_start();
        if !(t.starts_with("const ") && t.contains(": Source")) {
            continue;
        }
        // The literal may continue on the next lines.
        let mut lit = String::new();
        for l in lines.iter().skip(i) {
            lit.push_str(l.trim());
            lit.push(' ');
            if l.contains("};") {
                break;
            }
        }
        let Some(body) = lit.split_once("Source {").map(|(_, b)| b.split('}').next().unwrap_or("")) else { continue };
        let mut c = Corpus::default();
        for field in body.split(',') {
            let Some((k, v)) = field.split_once(':') else { continue };
            let v = v.trim().trim_matches('"').to_string();
            match k.trim() {
                "label" => c.label = v,
                "dir" | "default_dir" => c.dir = v,
                "pass_floor" => c.pass_floor = v.parse().ok(),
                "roundtrip_floor" => c.roundtrip_floor = v.parse().ok(),
                _ => {}
            }
        }
        let mut doc: Vec<&str> =
            lines.iter().take(i).rev().take_while(|l| l.trim_start().starts_with("///")).map(|l| l.trim_start().trim_start_matches('/').trim()).collect();
        doc.reverse();
        c.doc = doc.join(" ");
        c.files = c
            .doc
            .split_whitespace()
            .collect::<Vec<_>>()
            .windows(2)
            .find(|w| w.get(1).is_some_and(|x| x.starts_with("files")))
            .and_then(|w| w.first()?.trim_start_matches('(').parse().ok());
        out.push(c);
    }
    out
}

// ---- prefs audit -----------------------------------------------------------------------------

/// A preference: `section.field` (`section` is the `Preferences` field holding a section
/// struct), or a top-level `Preferences` field when `section` is `None`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PrefField {
    pub section: Option<String>,
    pub field: String,
    /// Methods of the section type (in `prefs.rs`) that read this field through `self.field`:
    /// calling `section.method()` reads it too.
    pub helpers: BTreeSet<String>,
}

impl PrefField {
    pub fn path(&self) -> String {
        match &self.section {
            Some(s) => format!("{s}.{}", self.field),
            None => self.field.clone(),
        }
    }
}

/// `pub struct` name → its `pub` fields `(name, type)`, for structs written at column 0.
fn structs(src: &str) -> BTreeMap<String, Vec<(String, String)>> {
    let mut out = BTreeMap::new();
    let mut cur: Option<(String, Vec<(String, String)>)> = None;
    for line in src.lines() {
        if let Some(rest) = line.strip_prefix("pub struct ")
            && let Some(name) = rest.strip_suffix(" {")
        {
            cur = Some((name.trim().to_string(), Vec::new()));
            continue;
        }
        if line == "}" {
            if let Some((n, f)) = cur.take() {
                out.insert(n, f);
            }
            continue;
        }
        if let Some((_, fields)) = cur.as_mut()
            && let Some(rest) = line.trim_start().strip_prefix("pub ")
            && let Some((name, ty)) = rest.split_once(':')
            && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            fields.push((name.to_string(), ty.trim().trim_end_matches(',').trim().to_string()));
        }
    }
    out
}

/// Top-level blocks (`header {` at column 0 up to the next `}` at column 0): `(header, body)`.
fn blocks<'a>(src: &'a str, starts: &str) -> Vec<(&'a str, String)> {
    let mut out = Vec::new();
    let mut cur: Option<(&str, String)> = None;
    for line in src.lines() {
        if cur.is_none() && line.starts_with(starts) && line.ends_with('{') {
            cur = Some((line.trim_end_matches('{').trim(), String::new()));
            continue;
        }
        if line == "}" {
            if let Some(b) = cur.take() {
                out.push(b);
            }
            continue;
        }
        if let Some((_, body)) = cur.as_mut() {
            body.push_str(line);
            body.push('\n');
        }
    }
    out
}

/// Methods of `impl <ty>` blocks → the `self.` fields each one reads.
fn helper_methods(src: &str, ty: &str) -> BTreeMap<String, BTreeSet<String>> {
    let mut out: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (_, body) in blocks(src, "impl ").into_iter().filter(|(h, _)| *h == format!("impl {ty}")) {
        let mut name: Option<String> = None;
        for line in body.lines() {
            let t = line.trim_start();
            if let Some(rest) = t.strip_prefix("pub fn ").or_else(|| t.strip_prefix("fn ")) {
                name = Some(rest.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect());
            }
            if let Some(n) = &name {
                let a = accesses(line);
                out.entry(n.clone()).or_default().extend(a.pairs.iter().filter_map(|p| p.strip_prefix("self.")).map(str::to_string));
            }
        }
    }
    out
}

/// The leaves of `pub struct Preferences` in `prefs.rs`, with the section helper methods that
/// read each one.
pub fn pref_fields(src: &str) -> Vec<PrefField> {
    let s = structs(src);
    let mut out = Vec::new();
    for (name, ty) in s.get("Preferences").cloned().unwrap_or_default() {
        match s.get(&ty) {
            Some(sub) => {
                let helpers = helper_methods(src, &ty);
                out.extend(sub.iter().map(|(f, _)| PrefField {
                    section: Some(name.clone()),
                    field: f.clone(),
                    helpers: helpers.iter().filter(|(_, used)| used.contains(f)).map(|(m, _)| m.clone()).collect(),
                }))
            }
            None => out.push(PrefField { section: None, field: name, helpers: BTreeSet::new() }),
        }
    }
    out
}

/// The engine code inside `prefs.rs`: its `impl Session` blocks (e.g. `apply_prefs`, which
/// pushes preferences into live state). Everything else there is the model itself.
pub fn session_code(src: &str) -> String {
    blocks(src, "impl Session").into_iter().map(|(_, b)| b).collect()
}

fn ident_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// Field accesses in one source file: every `.ident` and every `a.b` pair.
#[derive(Default)]
pub struct Accesses {
    singles: BTreeSet<String>,
    pairs: BTreeSet<String>,
    mentions_prefs: bool,
}

pub fn accesses(src: &str) -> Accesses {
    let b = src.as_bytes();
    let mut a = Accesses { mentions_prefs: src.contains("prefs"), ..Default::default() };
    for (i, &c) in b.iter().enumerate() {
        if c != b'.' {
            continue;
        }
        let end = b.iter().skip(i + 1).take_while(|&&x| ident_char(x)).count();
        let Some(name) = src.get(i + 1..i + 1 + end).filter(|n| n.bytes().next().is_some_and(|x| x.is_ascii_alphabetic() || x == b'_')) else { continue };
        a.singles.insert(name.to_string());
        let start = b.iter().take(i).rev().take_while(|&&x| ident_char(x)).count();
        if let Some(prev) = src.get(i - start..i).filter(|p| !p.is_empty()) {
            a.pairs.insert(format!("{prev}.{name}"));
        }
    }
    a
}

/// Whether some file reads `f`: `section.field` appears, or both `.section` and `.field` appear
/// in the same file (a section bound to a variable first). Top-level fields: `.field` in a file
/// that mentions `prefs`. Generic names (`enabled`) can be counted as read by accident, never
/// the other way round, so the count is a lower bound.
pub fn is_read(f: &PrefField, files: &[Accesses]) -> bool {
    let reads = |a: &Accesses, s: &str, name: &str| a.pairs.contains(&format!("{s}.{name}")) || (a.singles.contains(s) && a.singles.contains(name));
    files.iter().any(|a| match &f.section {
        Some(s) => reads(a, s, &f.field) || f.helpers.iter().any(|m| reads(a, s, m)),
        None => a.mentions_prefs && a.singles.contains(&f.field),
    })
}

/// `src` without its top-level `#[cfg(test)]` items (a `mod tests { .. }` block, a `mod x;`
/// line or a test helper function), so unit tests don't count as reads.
pub fn strip_tests(src: &str) -> String {
    let mut out = String::new();
    let mut lines = src.lines();
    while let Some(line) = lines.next() {
        if line != "#[cfg(test)]" {
            out.push_str(line);
            out.push('\n');
            continue;
        }
        // Further attributes, then the item: one line (`mod x;`) or a block up to `}` at column 0.
        for item in lines.by_ref() {
            if item.starts_with("#[") {
                continue;
            }
            if !item.trim_end().ends_with(';') {
                for l in lines.by_ref() {
                    if l == "}" {
                        break;
                    }
                }
            }
            break;
        }
    }
    out
}

/// Source files the audit scans: `.rs` under `crates/` and `apps/`, minus the prefs model
/// (`prefs.rs` and its tests), tests, examples, benches and fuzz targets, with `#[cfg(test)]`
/// items removed. `prefs_ui.rs` is scanned: the dialog is schema-driven and never names a field,
/// so its field accesses are the appliers (canvas colours, autosave, history log).
fn audit_sources(root: &Path) -> Vec<(PathBuf, String)> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            if p.is_dir() {
                if !matches!(name.as_str(), "tests" | "examples" | "benches" | "fuzz" | "target") && !name.starts_with('.') {
                    walk(&p, out);
                }
            } else if name.ends_with(".rs") && name != "tests.rs" {
                out.push(p);
            }
        }
    }
    let mut files = Vec::new();
    walk(&root.join("crates"), &mut files);
    walk(&root.join("apps"), &mut files);
    files.sort();
    let model = root.join(PREFS_RS);
    let prefs_dir = root.join("crates/engine/src/prefs");
    files
        .into_iter()
        .filter(|p| *p != model && !p.starts_with(&prefs_dir))
        .filter_map(|p| {
            let s = std::fs::read_to_string(&p).ok()?;
            Some((p, strip_tests(&s.replace("\r\n", "\n"))))
        })
        .collect()
}

/// Preferences that nothing reads (sorted paths).
pub fn prefs_audit(root: &Path) -> Result<(usize, Vec<String>), String> {
    let src = std::fs::read_to_string(root.join(PREFS_RS)).map_err(|e| format!("read {PREFS_RS}: {e}"))?;
    let fields = pref_fields(&src);
    if fields.is_empty() {
        return Err(format!("{PREFS_RS}: no `pub struct Preferences` fields found (did the struct move?)"));
    }
    let mut files: Vec<Accesses> = audit_sources(root).iter().map(|(_, s)| accesses(s)).collect();
    files.push(accesses(&session_code(&strip_tests(&src))));
    let unread: Vec<String> = fields.iter().filter(|f| !is_read(f, &files)).map(PrefField::path).collect();
    Ok((fields.len(), unread))
}

// ---- counted from the tree -------------------------------------------------------------------

const NEVER_CRASH: &str = "deny(clippy::unwrap_used";

/// Crate roots (`crates/*/src/lib.rs`; an app's `lib.rs`, else its `main.rs`) and the ones
/// missing the never-crash attribute.
fn never_crash(root: &Path) -> (usize, Vec<String>) {
    let mut roots = Vec::new();
    for (dir, lib_only) in [("crates", true), ("apps", false)] {
        let Ok(rd) = std::fs::read_dir(root.join(dir)) else { continue };
        let mut pkgs: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.join("Cargo.toml").exists()).collect();
        pkgs.sort();
        for p in pkgs {
            let lib = p.join("src/lib.rs");
            let main = p.join("src/main.rs");
            let r = if lib.exists() {
                lib
            } else if !lib_only && main.exists() {
                main
            } else {
                continue;
            };
            roots.push(r);
        }
    }
    let missing = roots
        .iter()
        .filter(|r| !std::fs::read_to_string(r).is_ok_and(|s| s.contains(NEVER_CRASH)))
        .map(|r| r.strip_prefix(root).unwrap_or(r).to_string_lossy().replace('\\', "/"))
        .collect();
    (roots.len(), missing)
}

/// `**626 of 626 menu items live (100.0%).**` from `docs/parity-checklist.md`.
fn parity_line(root: &Path) -> Option<String> {
    let s = std::fs::read_to_string(root.join("docs/parity-checklist.md")).ok()?;
    s.lines().find(|l| l.contains("menu items live")).map(|l| l.trim().trim_matches('*').trim_end_matches('.').to_string())
}

// ---- rendering -------------------------------------------------------------------------------

fn status_word(s: ItemStatus) -> &'static str {
    match s {
        ItemStatus::Done => "done",
        ItemStatus::Partial => "partial",
        ItemStatus::Missing => "missing",
    }
}

pub fn render_checklist(c: &Checklist) -> String {
    let mut s = String::from("| Id | Target | Status | Issue | Note |\n|---|---|---|---|---|\n");
    for i in &c.item {
        s.push_str(&format!("| {} | {} | {} | {} | {} |\n", i.id, i.target, status_word(i.status), perf::issue_link(i.issue), i.note.as_deref().unwrap_or("")));
    }
    s
}

/// Everything `render` needs, gathered from the tree.
pub struct Inputs {
    pub budgets: perf::Budgets,
    pub baseline: Option<Value>,
    pub corpora: Vec<Corpus>,
    pub checklists: Vec<Checklist>,
    pub prefs: (usize, Vec<String>),
    pub never_crash: (usize, Vec<String>),
    pub parity: Option<String>,
    pub panic_hunt: bool,
}

pub fn gather(root: &Path) -> Result<Inputs, String> {
    let budgets = perf::parse_budgets(&std::fs::read_to_string(root.join(perf::BUDGETS)).map_err(|e| format!("read {}: {e}", perf::BUDGETS))?)?;
    let baseline = match std::fs::read_to_string(root.join(perf::BASELINE)) {
        Ok(t) => Some(serde_json::from_str(&t).map_err(|e| format!("{}: {e}", perf::BASELINE))?),
        Err(_) => None,
    };
    let corpora = parse_corpus(&std::fs::read_to_string(root.join(CORPUS_RS)).map_err(|e| format!("read {CORPUS_RS}: {e}"))?);
    Ok(Inputs {
        budgets,
        baseline,
        corpora,
        checklists: load_checklists(root)?,
        prefs: prefs_audit(root)?,
        never_crash: never_crash(root),
        parity: parity_line(root),
        panic_hunt: root.join("crates/engine/tests/panic_hunt.rs").exists(),
    })
}

fn checklist_row(c: &Checklist, key: &str) -> String {
    let (d, p, m) = c.counts();
    format!("| [{}](#{}) | {d} | {p} | {m} | {key} |\n", c.title, anchor(&c.title))
}

/// GitHub's heading anchor: lower-case, spaces to `-`, punctuation dropped.
pub fn anchor(title: &str) -> String {
    title
        .to_lowercase()
        .chars()
        .filter_map(|c| {
            if c == ' ' {
                Some('-')
            } else if c.is_alphanumeric() || c == '-' {
                Some(c)
            } else {
                None
            }
        })
        .collect()
}

pub fn render(inp: &Inputs) -> String {
    let outcomes = perf::outcomes_from_baseline(&inp.budgets, inp.baseline.as_ref());
    let (pass, over, tracked, nm) = perf::counts(&outcomes);
    let not_run = outcomes.iter().filter(|o| o.status == perf::Status::NotRun).count();
    let full = inp.baseline.as_ref().and_then(|b| b.get("modes")?.get("full"));
    let machine = inp.baseline.as_ref().and_then(|b| b.get("machine"));
    let ms = |k: &str| machine.and_then(|m| m.get(k)).and_then(Value::as_str).unwrap_or("unknown").to_string();
    let base_date = full.and_then(|f| f.get("date")).and_then(Value::as_str).map(|d| d.get(..10).unwrap_or(d).to_string());
    let (unread_n, total_prefs) = (inp.prefs.1.len(), inp.prefs.0);
    let (roots, missing_nc) = (&inp.never_crash.0, &inp.never_crash.1);

    let mut s = String::from(
        "# PhotoCraft scorecard\n\nGenerated by `cargo xtask scorecard` from `perf/budgets.toml`, `perf/baseline.json`, \
`crates/io/tests/corpus.rs`, `scorecard/*.toml` and the source tree. Do not edit by hand: change the sources and \
regenerate (CI runs `cargo xtask scorecard --check`). Checklist statuses are verified against the code; numbers \
here supersede estimates elsewhere.\n\n## Summary\n\n| Area | Done | Partial | Missing | Key numbers |\n|---|---:|---:|---:|---|\n",
    );
    let perf_key = match &base_date {
        Some(d) => format!(
            "{} scenarios: {pass} within budget, {over} over budget or failing, {tracked} tracked only, {not_run} not run, {nm} not measurable yet (baseline {d})",
            outcomes.len()
        ),
        None => format!("{} scenarios, not run yet (no baseline); {nm} not measurable yet", outcomes.len()),
    };
    s.push_str(&format!("| [Performance](#performance) | {pass} | {over} | {} | {perf_key} |\n", nm + not_run));
    let corpus_key = inp
        .corpora
        .iter()
        .map(|c| {
            let of = c.files.map_or_else(String::new, |f| format!("/{f}"));
            format!("{}: oracle ≥ {}{of}, round trip ≥ {}{of}", c.label, c.pass_floor.unwrap_or(0), c.roundtrip_floor.unwrap_or(0))
        })
        .collect::<Vec<_>>()
        .join("; ");
    for c in &inp.checklists {
        let key = match c.area.as_str() {
            "file" => corpus_key.clone(),
            "ui" => format!("settings that do nothing: {unread_n} of {total_prefs} (target 0)"),
            "automation" => inp.parity.clone().unwrap_or_default(),
            "reliability" => format!(
                "never-crash lints on {} of {roots} crates; panic_hunt {}",
                roots - missing_nc.len(),
                if inp.panic_hunt { "present" } else { "missing" }
            ),
            _ => String::new(),
        };
        s.push_str(&checklist_row(c, &key));
    }

    // Performance.
    s.push_str("\n## Performance\n\n");
    s.push_str(&format!(
        "Budgets: `perf/budgets.toml` (targets from #209, #210, #211). Numbers: the committed baseline \
(`perf/baseline.json`, full run), updated only by `cargo xtask perf --update-baseline`. The nightly \
`perf-nightly` workflow runs `cargo xtask perf` on a fixed macOS runner and fails on a broken enforced \
budget or a p50 regression over {:.0} % against a baseline from the same machine class. \"Over budget\" \
rows are targets the code doesn't meet yet: they are reported, and only their regressions fail.\n\n",
        inp.budgets.settings.regression_pct
    ));
    match (&base_date, full) {
        (Some(d), Some(f)) => {
            let load = perf::fmt_load(f.get("load_avg_start"));
            let commit = f.get("commit").and_then(Value::as_str).map_or_else(|| "unknown".into(), |c| c.get(..10).unwrap_or(c).to_string());
            let ram = machine.and_then(|m| m.get("ram_bytes")).and_then(Value::as_u64).map_or_else(|| "?".into(), |b| format!("{} GB", b >> 30));
            s.push_str(&format!(
                "Baseline: {d}, commit `{commit}`, {} ({ram}), GPU {}, {}; load average {load} (1/5/15 min) at the start of the run.\n\n",
                ms("cpu"),
                ms("gpu_adapter"),
                ms("os_version")
            ));
        }
        _ => s.push_str("Baseline: **not run** (no `perf/baseline.json` full run yet).\n\n"),
    }
    s.push_str(&perf::render_table(&outcomes, false));

    // Corpora.
    for c in &inp.checklists {
        s.push_str(&format!("\n## {}\n\n", c.title));
        match c.area.as_str() {
            "file" => {
                s.push_str("Corpus floors (`crates/io/tests/corpus.rs`): the enforced minimums; the corpus tests (cargo feature `corpus`, `cargo xtask test-corpus`, always run in CI) fail below them.\n\n| Corpus | Directory | Files | Oracle pass floor | Round-trip floor |\n|---|---|---:|---:|---:|\n");
                for k in &inp.corpora {
                    let n = |v: Option<u32>| v.map_or_else(|| "–".into(), |x| x.to_string());
                    s.push_str(&format!("| {} | `{}` | {} | {} | {} |\n", k.label, k.dir, n(k.files), n(k.pass_floor), n(k.roundtrip_floor)));
                }
                s.push('\n');
            }
            "ui" => {
                s.push_str(&format!(
                    "**Settings that do nothing: {unread_n} of {total_prefs}** (target 0, #204). Counted by the prefs audit \
(`xtask/src/scorecard.rs`): a `Preferences` field counts as read when code outside the prefs model (`prefs.rs` \
except its `impl Session` blocks; tests excluded; the schema-driven dialog never names a field) accesses \
`section.field`, or `.section` and `.field` in the same file, or calls a section method that reads `self.field`. \
It is a heuristic lower bound: a generic name can look read when it isn't.\n\n"
                ));
                if !inp.prefs.1.is_empty() {
                    s.push_str(&format!("Unread: {}.\n\n", inp.prefs.1.iter().map(|p| format!("`{p}`")).collect::<Vec<_>>().join(", ")));
                }
            }
            "automation" => {
                if let Some(p) = &inp.parity {
                    s.push_str(&format!("Menu wiring (`docs/parity-checklist.md`): {p}. This counts dispatch, not behaviour.\n\n"));
                }
            }
            "reliability" => {
                s.push_str(&format!("Never-crash attribute on {} of {roots} crate roots", roots - missing_nc.len()));
                if missing_nc.is_empty() {
                    s.push_str(".\n\n");
                } else {
                    s.push_str(&format!("; missing: {}.\n\n", missing_nc.iter().map(|m| format!("`{m}`")).collect::<Vec<_>>().join(", ")));
                }
            }
            _ => {}
        }
        s.push_str(&render_checklist(c));
    }
    s
}

pub fn run(root: &Path, rest: &[&str]) -> Result<(), String> {
    let check = rest.contains(&"--check");
    if let Some(bad) = rest.iter().find(|a| **a != "--check") {
        return Err(format!("scorecard: unknown option `{bad}`"));
    }
    let inp = gather(root)?;
    let text = render(&inp);
    let path = root.join(OUT);
    if check {
        let current = std::fs::read_to_string(&path).unwrap_or_default().replace("\r\n", "\n");
        if current != text {
            return Err(format!("{OUT} is stale: run `cargo xtask scorecard` and commit the result"));
        }
        println!("{OUT} is up to date");
        return Ok(());
    }
    std::fs::write(&path, &text).map_err(|e| format!("write {OUT}: {e}"))?;
    println!("wrote {OUT}: settings that do nothing {} of {}", inp.prefs.1.len(), inp.prefs.0);
    if let Ok(t) = std::fs::read_to_string(crate::root().join("target/perf/results.json"))
        && let Ok(v) = serde_json::from_str::<Value>(&t)
    {
        println!(
            "note: target/perf/results.json ({} run, {}) is not in the scorecard; publish it with `cargo xtask perf --update-baseline`",
            v.get("mode").and_then(Value::as_str).unwrap_or("?"),
            v.get("date").and_then(Value::as_str).unwrap_or("?")
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PREFS: &str = r#"
pub struct General {
    pub beep_when_done: bool,
    /// doc
    pub animated_zoom: bool,
}

pub struct Interface {
    pub ui_scale: UiScale,
    pub grid_colors: Colors,
    pub grid_size: Size,
}

impl Interface {
    pub fn colors(&self) -> [u8; 2] {
        match self.grid_colors {
            _ => [0, 1],
        }
    }
    fn size(&self) -> f32 {
        self.grid_size.px()
    }
}

impl Default for Interface {
    fn default() -> Self {
        Self { ui_scale: 1, grid_colors: x, grid_size: y }
    }
}

impl Session {
    pub fn apply_prefs(&mut self) {
        let n = self.prefs().general.beep_when_done;
    }
}

#[derive(Default)]
pub struct Preferences {
    pub general: General,
    pub interface: Interface,
    pub shortcuts: BTreeMap<String, String>,
    pub workspace_locked: bool,
}
"#;

    #[test]
    fn pref_fields_walk_sections() {
        let fields = pref_fields(PREFS);
        let f: Vec<String> = fields.iter().map(PrefField::path).collect();
        assert_eq!(
            f,
            [
                "general.beep_when_done",
                "general.animated_zoom",
                "interface.ui_scale",
                "interface.grid_colors",
                "interface.grid_size",
                "shortcuts",
                "workspace_locked"
            ]
        );
        // Section helper methods: `colors()` reads grid_colors, `size()` reads grid_size.
        let helpers: Vec<Vec<String>> = fields.iter().map(|f| f.helpers.iter().cloned().collect()).collect();
        assert_eq!(helpers[3], ["colors"]);
        assert_eq!(helpers[4], ["size"]);
        assert!(helpers[2].is_empty());
        assert!(pref_fields("pub struct Other {\n    pub x: u8,\n}\n").is_empty());
        // Only `impl Session` counts as engine code inside prefs.rs.
        let code = session_code(PREFS);
        assert!(code.contains("general.beep_when_done") && !code.contains("grid_colors"));
    }

    #[test]
    fn prefs_audit_rules() {
        let fields = pref_fields(PREFS);
        let files = [
            accesses(&session_code(PREFS)),
            // Reads through a section helper.
            accesses("let [a, b] = app.session.prefs().interface.colors();"),
            // Section bound first, field read later in the same file.
            accesses("let i = &prefs.interface; let s = i.ui_scale;"),
            // A top-level field read in a file that mentions prefs.
            accesses("let p = session.prefs(); p.shortcuts.get(id)"),
            // Not prefs: `.workspace_locked` in a file that never mentions prefs doesn't count.
            accesses("state.workspace_locked = true;"),
        ];
        let unread: Vec<String> = fields.iter().filter(|f| !is_read(f, &files)).map(PrefField::path).collect();
        assert_eq!(unread, ["general.animated_zoom", "interface.grid_size", "workspace_locked"]);
        // A word inside an identifier or a string is not an access.
        let a = accesses("let animated_zoom = 1; \"general.animated_zoom\".len(); x.animated_zoomed");
        assert!(a.pairs.contains("general.animated_zoom"), "string paths look like accesses; documented lower bound");
        assert!(a.singles.contains("animated_zoom"), "and so do their parts");
        assert!(!a.singles.contains("general"), "`let animated_zoom` and `\"general` are not accesses");
        assert!(a.singles.contains("animated_zoomed"));
    }

    #[test]
    fn test_items_are_stripped() {
        let src = "fn a() { p.general.x }\n#[cfg(test)]\nfn helper() {\n    p.general.y\n}\n\nfn b() { p.general.z }\n#[cfg(test)]\nmod more;\n#[cfg(test)]\n#[allow(clippy::unreachable)]\nmod tests {\n    fn t() { p.general.w }\n}\n";
        let s = strip_tests(src);
        assert!(s.contains("general.x") && s.contains("general.z"));
        assert!(!s.contains("general.y") && !s.contains("general.w") && !s.contains("mod more"));
    }

    #[test]
    fn corpus_floors_parse() {
        let src = r#"
/// Hand-picked mix in `corpus/psd` (170 files: 121 vs the merged image + 12 vs the thumbnail).
const MIXED: Source = Source { label: "io corpus", env: "PHOTOCRAFT_CORPUS", default_dir: "corpus/psd", pass_floor: 133, roundtrip_floor: 169 };

/// The full psd-tools test set (309 files at the pinned commit):
/// 202 vs the merged image.
const PSD_TOOLS: Source =
    Source { label: "psd-tools corpus", env: "X", default_dir: "corpus/psd-tools", pass_floor: 212, roundtrip_floor: 307, extra: 5 };

/// Ours (256 files; https://github.com/storytold/photocraft-corpus), grouped by feature.
const PHOTOSHOP: Source = Source { label: "photoshop oracles", dir: "corpus/photoshop", group_depth: 2, pass_floor: 74, roundtrip_floor: 256 };
"#;
        let c = parse_corpus(src);
        assert_eq!(c.len(), 3);
        assert_eq!(
            (c[0].label.as_str(), c[0].dir.as_str(), c[0].files, c[0].pass_floor, c[0].roundtrip_floor),
            ("io corpus", "corpus/psd", Some(170), Some(133), Some(169))
        );
        assert_eq!((c[1].files, c[1].pass_floor, c[1].roundtrip_floor), (Some(309), Some(212), Some(307)));
        assert_eq!((c[2].dir.as_str(), c[2].files, c[2].pass_floor, c[2].roundtrip_floor), ("corpus/photoshop", Some(256), Some(74), Some(256)));
        assert!(parse_corpus("const X: u32 = 1;").is_empty());
    }

    #[test]
    fn checklists_validate() {
        let ok = "area = \"tools\"\ntitle = \"Tools\"\norder = 3\n[[item]]\nid = \"T-1\"\ntarget = \"Pencil\"\nstatus = \"partial\"\nissue = 213\n";
        let c = parse_checklist("t.toml", ok).unwrap();
        assert_eq!(c.counts(), (0, 1, 0));
        assert!(render_checklist(&c).contains("| T-1 | Pencil | partial | [#213](https://github.com/storytold/photocraft/issues/213) |  |"));
        assert!(parse_checklist("t.toml", &ok.replace("partial", "almost")).is_err(), "status must be done|partial|missing");
        assert!(parse_checklist("t.toml", &ok.replace("Pencil", "a | b")).is_err());
        assert!(parse_checklist("t.toml", &format!("{ok}bogus = 1\n")).is_err());
    }

    #[test]
    fn anchors() {
        assert_eq!(anchor("Workspace and UI"), "workspace-and-ui");
        assert_eq!(anchor("File compatibility"), "file-compatibility");
    }

    #[test]
    fn staleness_check_and_determinism() {
        let root = crate::root();
        let inp = gather(&root).unwrap_or_else(|e| panic!("{e}"));
        let a = render(&inp);
        assert_eq!(a, render(&gather(&root).unwrap()), "rendering is deterministic");
        // The committed file is current (what `--check` enforces in CI).
        let committed = std::fs::read_to_string(root.join(OUT)).unwrap_or_default().replace("\r\n", "\n");
        assert!(committed == a, "{OUT} is stale: run `cargo xtask scorecard`");
        // Every scenario and every checklist area shows up.
        for sc in &inp.budgets.scenario {
            assert!(a.contains(&format!("| {} |", sc.id)), "{} missing from the scorecard", sc.id);
        }
    }
}
