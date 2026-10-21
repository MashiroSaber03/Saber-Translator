//! Every preference does something, or the Preferences dialog hides it (issue #204).
//!
//! Walks every key of `Preferences::default()` serialised to JSON (`section.key`) and checks
//! the workspace sources: each one is either read outside `prefs.rs` or listed in
//! `prefs::HIDDEN_UNTIL_IMPLEMENTED`, never both. A setting that appears in the dialog but that
//! nothing reads is a control that silently does nothing.
//!
//! "Read" is a source scan (no type information), so it is deliberately simple:
//! - a member access `.<snake_key>` in a non-test source file that also names the section
//!   (`<snake_section>`), e.g. `p.file_handling.recent_files` or `let fh = &p.file_handling;
//!   … fh.autosave`;
//! - or the JSON path as a string literal (`"fileHandling.autosave"`), for code that reads
//!   preferences by path;
//! - or `self.<snake_key>` in an inherent `impl` of the section's struct in `prefs.rs` (helpers
//!   such as `TransparencyAndGamut::colors`).

use photocraft_engine::prefs::{HIDDEN_UNTIL_IMPLEMENTED, Preferences};
use std::path::{Path, PathBuf};

fn snake(camel: &str) -> String {
    let mut out = String::new();
    for c in camel.chars() {
        if c.is_ascii_uppercase() {
            out.push('_');
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

fn is_ident(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// `needle` occurs in `text` as a whole identifier (after `prefix`, e.g. "." or "self.").
fn has_ident(text: &str, prefix: &str, ident: &str) -> bool {
    let needle = format!("{prefix}{ident}");
    text.match_indices(&needle).any(|(i, _)| {
        let before = text[..i].chars().next_back();
        let after = text[i + needle.len()..].chars().next();
        (!prefix.is_empty() || before.is_none_or(|c| !is_ident(c))) && after.is_none_or(|c| !is_ident(c))
    })
}

/// Production source of a file: everything before its unit-test module.
fn production(text: &str) -> &str {
    // Windows checkouts may use CRLF; test-only reads must stay excluded.
    let cut = ["#[cfg(test)]\nmod ", "#[cfg(test)]\r\nmod ", "#[cfg(test)]\npub mod ", "#[cfg(test)]\r\npub mod "]
        .into_iter()
        .filter_map(|marker| text.find(marker))
        .min()
        .unwrap_or(text.len());
    &text[..cut]
}

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        if p.is_dir() {
            if !matches!(name.as_str(), "tests" | "examples" | "benches" | "target") {
                rust_sources(&p, out);
            }
        } else if name.ends_with(".rs") && name != "tests.rs" && !name.ends_with("_tests.rs") {
            out.push(p);
        }
    }
}

/// `pub struct Name { … }` field names, and the bodies of `impl Name { … }` blocks, in prefs.rs.
fn struct_fields(src: &str, name: &str) -> Vec<String> {
    let Some(start) = src.find(&format!("pub struct {name} {{")) else { return Vec::new() };
    let body = &src[start..];
    let end = body.find("\n}").unwrap_or(body.len());
    body[..end].lines().filter_map(|l| l.trim().strip_prefix("pub ")?.split_once(':').map(|(f, _)| f.trim().to_string())).collect()
}

fn inherent_impls(src: &str, name: &str) -> String {
    let marker = format!("\nimpl {name} {{");
    src.match_indices(&marker)
        .map(|(i, _)| {
            let body = &src[i + 1..];
            body[..body.find("\n}").unwrap_or(body.len())].to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn every_preference_is_read_or_hidden() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let prefs_rs = std::fs::read_to_string(root.join("crates/engine/src/prefs.rs")).unwrap();
    let mut files = Vec::new();
    rust_sources(&root.join("crates"), &mut files);
    rust_sources(&root.join("apps"), &mut files);
    let sources: Vec<String> = files
        .iter()
        .filter(|p| !p.ends_with("engine/src/prefs.rs"))
        .filter_map(|p| std::fs::read_to_string(p).ok())
        .map(|t| production(&t).to_string())
        .collect();
    assert!(sources.len() > 100, "found only {} source files under {}", sources.len(), root.display());

    // Section key → struct name, from `pub <field>: <Type>,` in `struct Preferences`.
    let section_types: Vec<(String, String)> = {
        let start = prefs_rs.find("pub struct Preferences {").unwrap();
        let body = &prefs_rs[start..];
        body[..body.find("\n}").unwrap()]
            .lines()
            .filter_map(|l| {
                let (f, ty) = l.trim().strip_prefix("pub ")?.split_once(':')?;
                Some((f.trim().to_string(), ty.trim().trim_end_matches(',').to_string()))
            })
            .collect()
    };

    let json = serde_json::to_value(Preferences::default()).unwrap();
    let mut paths = Vec::new();
    for (section, v) in json.as_object().unwrap() {
        match v.as_object() {
            Some(o) if !o.is_empty() => paths.extend(o.keys().map(|k| (section.clone(), Some(k.clone())))),
            _ => paths.push((section.clone(), None)),
        }
    }

    let mut unread = Vec::new();
    let mut stale = Vec::new();
    for (section, key) in &paths {
        let section_rs = if section == "type" { "type_".to_string() } else { snake(section) };
        let (path, read) = match key {
            None => (section.clone(), sources.iter().any(|t| has_ident(t, ".", &section_rs))),
            Some(k) => {
                let field = snake(k);
                let path = format!("{section}.{k}");
                let ty = section_types.iter().find(|(f, _)| *f == section_rs).map(|(_, t)| t.clone()).unwrap_or_default();
                let in_helpers =
                    !ty.is_empty() && struct_fields(&prefs_rs, &ty).contains(&field) && has_ident(&inherent_impls(&prefs_rs, &ty), "self.", &field);
                let read =
                    in_helpers || sources.iter().any(|t| (has_ident(t, ".", &field) && has_ident(t, "", &section_rs)) || t.contains(&format!("\"{path}\"")));
                (path, read)
            }
        };
        let hidden = HIDDEN_UNTIL_IMPLEMENTED.contains(&path.as_str());
        if !read && !hidden {
            unread.push(path);
        } else if read && hidden {
            stale.push(path);
        }
    }
    for h in HIDDEN_UNTIL_IMPLEMENTED {
        assert!(
            paths.iter().any(|(s, k)| k.as_ref().map(|k| format!("{s}.{k}")).as_deref() == Some(*h)),
            "HIDDEN_UNTIL_IMPLEMENTED lists `{h}`, which is not a preference"
        );
    }
    assert!(
        unread.is_empty(),
        "these preferences are shown in Edit › Preferences but nothing reads them: wire them up or add them to prefs::HIDDEN_UNTIL_IMPLEMENTED: {unread:#?}"
    );
    assert!(stale.is_empty(), "these preferences are read now: remove them from prefs::HIDDEN_UNTIL_IMPLEMENTED so the dialog shows them: {stale:#?}");
}

#[test]
fn the_scanner_finds_reads() {
    assert!(has_ident("p.file_handling.recent_files.clone()", ".", "recent_files"));
    assert!(!has_ident("p.file_handling.recent_files_x", ".", "recent_files"));
    assert!(has_ident("let fh = &p.file_handling;", "", "file_handling"));
    assert!(!has_ident("my_file_handling", "", "file_handling"));
    assert_eq!(snake("recentFileCount"), "recent_file_count");
    assert_eq!(production("fn a() {}\n#[cfg(test)]\nmod tests {\n p.x.y\n}"), "fn a() {}\n");
}

#[test]
fn production_excludes_test_modules_with_lf_or_crlf() {
    for newline in ["\n", "\r\n"] {
        for visibility in ["", "pub "] {
            let prefix = format!("// Préférences{newline}fn a() {{}}{newline}");
            let source = format!(
                "{prefix}#[cfg(test)]{newline}{visibility}mod tests {{{newline}    #[test]{newline}    fn hidden_setting() {{ let _ = \"type.smartQuotes\"; }}{newline}}}{newline}"
            );
            assert_eq!(production(&source), prefix, "newline={newline:?}, visibility={visibility:?}");
        }
    }
}

#[test]
fn production_keeps_sources_without_a_test_module() {
    for source in ["", "fn a() {}\n", "// Préférences\r\nfn a() {}\r\n"] {
        assert_eq!(production(source), source);
    }
}
