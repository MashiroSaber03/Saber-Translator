use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Key {
    context: String,
    source: String,
}

#[derive(Debug)]
struct Language {
    code: String,
    catalog: Option<PathBuf>,
}

pub fn run(root: &Path) -> Result<(), String> {
    let i18n_dir = root.join("crates/ui-egui/src/i18n");
    let registry_path = i18n_dir.join("mod.rs");
    let registry = fs::read_to_string(&registry_path).map_err(|e| format!("{}: {e}", registry_path.display()))?;
    let languages = registered_languages(&registry)?;

    let english_keys = source_keys(root)?;
    let mut catalog_texts = BTreeMap::new();
    for language in &languages {
        if let Some(path) = &language.catalog {
            let full_path = i18n_dir.join(path);
            let text = fs::read_to_string(&full_path).map_err(|e| format!("{}: {e}", full_path.display()))?;
            catalog_texts.insert(language.code.clone(), (full_path, text));
        }
    }
    let mut translations = BTreeMap::new();
    for language in &languages {
        let entries = if language.catalog.is_some() {
            let (full_path, text) = catalog_texts.get(&language.code).ok_or_else(|| format!("missing source for registered language `{}`", language.code))?;
            parse_catalog(full_path, text)?
        } else {
            BTreeSet::new()
        };
        translations.insert(language.code.clone(), entries);
    }

    if english_keys.is_empty() {
        return Err("no source-derived English UI translation keys were found".into());
    }
    print!("{}", format_report(&languages, &english_keys, &translations));
    Ok(())
}

fn source_keys(root: &Path) -> Result<BTreeSet<Key>, String> {
    let ui_sources = source_files(&root.join("crates/ui-egui/src"))?;
    let engine_sources = source_files(&root.join("crates/engine/src"))?;
    let mut keys = tl_keys(&ui_sources);
    keys.extend(plural_keys(&ui_sources));

    let catalog_path = root.join("crates/ui-egui/src/menu_catalog.rs");
    let catalog = fs::read_to_string(&catalog_path).map_err(|e| format!("{}: {e}", catalog_path.display()))?;
    keys.extend(menu_catalog_keys(&catalog));

    let ui_commands = ui_sources.iter().find(|(path, _)| path.ends_with("menus.rs")).ok_or_else(|| "could not find crates/ui-egui/src/menus.rs".to_string())?;
    keys.extend(ui_command_keys(&ui_commands.1));
    keys.extend(engine_command_labels(&engine_sources));
    Ok(keys)
}

fn format_report(languages: &[Language], english_keys: &BTreeSet<Key>, translations: &BTreeMap<String, BTreeSet<Key>>) -> String {
    let mut report = String::from("UI translation coverage (translated source keys / source keys)\n");
    report.push_str(&format!("{:<12} {:>12} {:>9}\n", "language", "translated", "coverage"));
    for language in languages {
        let translated =
            if language.code == "en" { english_keys.len() } else { translations.get(&language.code).map_or(0, |keys| keys.intersection(english_keys).count()) };
        let coverage = 100.0 * translated as f64 / english_keys.len() as f64;
        report.push_str(&format!("{:<12} {:>5}/{:<6} {:>8.2}%\n", language.code, translated, english_keys.len(), coverage));
    }
    report
}

fn registered_languages(source: &str) -> Result<Vec<Language>, String> {
    let registry = source
        .split("pub static LANGUAGES")
        .nth(1)
        .ok_or_else(|| "could not find the LANGUAGES registry".to_string())?
        .split("impl LangInfo")
        .next()
        .ok_or_else(|| "could not find the end of the LANGUAGES registry".to_string())?;
    let mut languages = Vec::new();
    let mut codes = HashSet::new();
    for (index, section) in registry.split("LangInfo {").skip(1).enumerate() {
        let block = section.split("},").next().unwrap_or(section);
        let code = quoted_field(block, "code:").ok_or_else(|| format!("LANGUAGES entry {} has no code", index + 1))?;
        if !codes.insert(code.clone()) {
            return Err(format!("duplicate registered language code `{code}`"));
        }
        let catalog = if code == "en" {
            None
        } else {
            // `catalog!("xx")` embeds `xx.tsv` (packed by ui-egui's build.rs).
            let marker = "catalog!(\"";
            let start = block.find(marker).ok_or_else(|| format!("language `{code}` has no catalog! entry"))? + marker.len();
            let rest = block.get(start..).ok_or_else(|| format!("language `{code}` has a malformed catalog path"))?;
            let end = rest.find("\")").ok_or_else(|| format!("language `{code}` has a malformed catalog path"))?;
            let file = format!("{}.tsv", rest.get(..end).ok_or_else(|| format!("language `{code}` has an invalid catalog path"))?);
            let path = Path::new(&file);
            if path.components().count() != 1 || path.extension().and_then(|ext| ext.to_str()) != Some("tsv") {
                return Err(format!("language `{code}` has unsafe catalog path `{file}`"));
            }
            Some(path.to_path_buf())
        };
        languages.push(Language { code, catalog });
    }
    if !languages.iter().any(|language| language.code == "en") {
        return Err("LANGUAGES registry does not contain English (`en`)".into());
    }
    languages.sort_by(|a, b| a.code.cmp(&b.code));
    Ok(languages)
}

fn quoted_field(block: &str, field: &str) -> Option<String> {
    let start = block.find(field)? + field.len();
    let rest = block.get(start..)?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest.get(..end)?.to_string())
}

fn source_files(root: &Path) -> Result<Vec<(PathBuf, String)>, String> {
    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(path) = stack.pop() {
        let entries = fs::read_dir(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("{}: {e}", path.display()))?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
                let stem = path.file_stem().and_then(|name| name.to_str()).unwrap_or_default();
                if stem == "tests" || stem.ends_with("_tests") {
                    continue;
                }
                let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
                files.push((path, text));
            }
        }
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(files)
}

fn tl_keys(sources: &[(PathBuf, String)]) -> BTreeSet<Key> {
    let mut keys = BTreeSet::new();
    for (path, source) in sources {
        if path.ends_with("i18n/mod.rs") {
            continue;
        }
        let code = before_test_module(source);
        let mut rest = code;
        while let Some(at) = rest.find("tl!(\"") {
            rest = rest.get(at + 5..).unwrap_or("");
            let bytes = rest.as_bytes();
            let mut end = 0;
            while end < bytes.len() && !(bytes[end] == b'"' && (end == 0 || bytes[end - 1] != b'\\')) {
                end += 1;
            }
            if let Some(raw) = rest.get(..end) {
                let source = unescape_rust_string(raw);
                if rest.get(end + 1..end + 2) == Some(")") {
                    keys.insert(Key { context: String::new(), source });
                }
            }
        }
    }
    keys
}

fn plural_keys(sources: &[(PathBuf, String)]) -> BTreeSet<Key> {
    let mut keys = BTreeSet::new();
    for (path, source) in sources {
        if path.ends_with("i18n/mod.rs") {
            continue;
        }
        let code = before_test_module(source);
        let mut rest = code;
        while let Some(at) = rest.find("trn(") {
            rest = rest.get(at + 4..).unwrap_or("");
            let Some(end) = call_end(rest) else { break };
            let args = rest.get(..end).unwrap_or("");
            // `trn(lang, n, one, other)`: the forms are the last two literals, so a string in the
            // count expression (`r["files"].len()`) is not mistaken for one.
            let forms = quoted_strings(args);
            if let [.., one, other] = forms.as_slice() {
                keys.insert(Key { context: "@plural".into(), source: format!("{one}|{other}") });
            }
            rest = rest.get(end + 1..).unwrap_or("");
        }
    }
    keys
}

fn menu_catalog_keys(source: &str) -> BTreeSet<Key> {
    let mut keys = BTreeSet::new();
    for line in source.lines().map(str::trim) {
        let Some(entry) = line.strip_prefix("(&[") else { continue };
        let Some((path, item)) = entry.split_once("],") else { continue };
        for segment in quoted_strings(path) {
            keys.insert(Key { context: String::new(), source: segment });
        }
        if let Some(label) = quoted_strings(item).first().filter(|label| label.as_str() != "---") {
            keys.insert(Key { context: String::new(), source: label.clone() });
        }
    }
    keys
}

fn ui_command_keys(source: &str) -> BTreeSet<Key> {
    let mut keys = BTreeSet::new();
    for line in source.lines().map(str::trim) {
        if !line.starts_with("(\"") {
            continue;
        }
        let Some((before_path, path_and_rest)) = line.split_once("&[") else { continue };
        let Some((path, _)) = path_and_rest.split_once(']') else { continue };
        if let Some(label) = quoted_strings(before_path).get(1) {
            keys.insert(Key { context: String::new(), source: label.clone() });
        }
        keys.extend(quoted_strings(path).into_iter().map(|source| Key { context: String::new(), source }));
    }
    keys
}

fn engine_command_labels(sources: &[(PathBuf, String)]) -> BTreeSet<Key> {
    let mut keys = BTreeSet::new();
    let mut command_macros = HashSet::new();
    for (_, source) in sources {
        let mut rest = before_test_module(source);
        while let Some(at) = rest.find("macro_rules!") {
            rest = rest.get(at + "macro_rules!".len()..).unwrap_or("");
            let name = rest.trim_start().split(|c: char| !c.is_ascii_alphanumeric() && c != '_').next().unwrap_or("");
            if name.is_empty() {
                continue;
            }
            let next_definition = rest.find("macro_rules!").unwrap_or(rest.len());
            let definition = rest.get(..next_definition).unwrap_or(rest);
            if definition.contains("CommandSpec {") && definition.contains("label:") {
                command_macros.insert(name.to_string());
            }
        }
    }
    for (_, source) in sources {
        let code = before_test_module(source);
        let mut rest = code;
        while let Some(at) = rest.find("CommandSpec {") {
            rest = rest.get(at + "CommandSpec {".len()..).unwrap_or("");
            let Some(end) = balanced_end(rest, '{', '}') else { break };
            let Some(body) = rest.get(..end) else { break };
            let mut fields = split_arguments(body).into_iter();
            let label_field = fields.find_map(|field| field.trim().strip_prefix("label:").map(str::trim));
            let menu_field = split_arguments(body).into_iter().find_map(|field| field.trim().strip_prefix("menu:").map(str::trim));
            if let (Some(label_field), Some(menu_field)) = (label_field, menu_field)
                && let Some(label) = quoted_strings(label_field).first()
                && (menu_field.starts_with("&[") || menu_field.starts_with('['))
                && let Some(path) = menu_field.split_once('[').and_then(|(_, rest)| rest.split_once(']').map(|(path, _)| path))
                && !path.trim().is_empty()
            {
                keys.insert(Key { context: String::new(), source: label.clone() });
                keys.extend(quoted_strings(path).into_iter().map(|source| Key { context: String::new(), source }));
            }
            rest = rest.get(end + 1..).unwrap_or("");
        }
        for name in &command_macros {
            let marker = format!("{name}!(");
            let mut rest = code;
            while let Some(at) = rest.find(&marker) {
                rest = rest.get(at + marker.len()..).unwrap_or("");
                let Some(end) = call_end(rest) else { break };
                let Some(args) = rest.get(..end) else { continue };
                if let Some(label) = menu_command_label(args) {
                    keys.insert(Key { context: String::new(), source: label });
                }
                rest = rest.get(end + 1..).unwrap_or("");
            }
        }
    }
    keys
}

fn before_test_module(source: &str) -> &str {
    let marker = "#[cfg(test)]";
    let mut offset = 0;
    while let Some(relative) = source.get(offset..).and_then(|rest| rest.find(marker)) {
        let start = offset + relative;
        let after_attribute = source.get(start + marker.len()..).unwrap_or("").trim_start();
        if after_attribute.strip_prefix("mod").is_some_and(|rest| rest.chars().next().is_some_and(char::is_whitespace)) {
            return source.get(..start).unwrap_or(source);
        }
        offset = start + marker.len();
    }
    source
}

fn menu_command_label(args: &str) -> Option<String> {
    let args = split_arguments(args);
    let menu = args.get(2)?.trim();
    if !(menu.starts_with("&[") || menu.starts_with('[')) || quoted_strings(menu).is_empty() {
        return None;
    }
    quoted_strings(args.get(1)?.trim()).first().cloned()
}

fn split_arguments(args: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0_usize;
    let mut in_string = false;
    let mut escaped = false;
    let mut start = 0;
    for (index, ch) in args.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                if let Some(arg) = args.get(start..index) {
                    out.push(arg);
                }
                start = index + ch.len_utf8();
            }
            _ => {}
        }
    }
    if let Some(arg) = args.get(start..) {
        out.push(arg);
    }
    out
}

fn call_end(source: &str) -> Option<usize> {
    balanced_end(source, '(', ')')
}

fn balanced_end(source: &str, open: char, close: char) -> Option<usize> {
    let mut depth = 1_usize;
    let mut in_string = false;
    let mut escaped = false;
    for (index, ch) in source.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            value if value == open => depth += 1,
            value if value == close => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

fn quoted_strings(source: &str) -> Vec<String> {
    let mut strings = Vec::new();
    let mut rest = source;
    while let Some(at) = rest.find('"') {
        rest = rest.get(at + 1..).unwrap_or("");
        let bytes = rest.as_bytes();
        let mut end = 0;
        while end < bytes.len() && !(bytes[end] == b'"' && (end == 0 || bytes[end - 1] != b'\\')) {
            end += 1;
        }
        if let Some(raw) = rest.get(..end) {
            strings.push(unescape_rust_string(raw));
        }
        rest = rest.get(end.saturating_add(1)..).unwrap_or("");
    }
    strings
}

fn unescape_rust_string(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            out.push(ch);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('t') => out.push('\t'),
            Some('\\') => out.push('\\'),
            Some('"') => out.push('"'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

fn parse_catalog(path: &Path, text: &str) -> Result<BTreeSet<Key>, String> {
    let mut keys = BTreeSet::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let mut columns = line.split('\t');
        let parsed = match (columns.next(), columns.next(), columns.next(), columns.next()) {
            (Some(context), Some(source), Some(translation), None) if !source.is_empty() && !translation.is_empty() => {
                Key { context: unescape_tsv(context), source: unescape_tsv(source) }
            }
            _ => return Err(format!("{}:{line_no}: expected `context<TAB>source<TAB>translation`", path.display(), line_no = index + 1)),
        };
        if !keys.insert(Key { context: parsed.context.clone(), source: parsed.source.clone() }) {
            return Err(format!("{}:{line_no}: duplicate key {:?} {:?}", path.display(), parsed.context, parsed.source, line_no = index + 1));
        }
    }
    Ok(keys)
}

fn unescape_tsv(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut chars = source.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            out.push(ch);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('\\') => out.push('\\'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn known_keys(values: &[&str]) -> BTreeSet<Key> {
        values.iter().map(|value| Key { context: String::new(), source: (*value).to_string() }).collect()
    }

    #[test]
    fn unused_legacy_catalog_keys_do_not_affect_coverage() {
        let catalog = parse_catalog(Path::new("xx.tsv"), "\tOld key\tÜbersetzung\n");
        assert_eq!(catalog, Ok(known_keys(&["Old key"])));
    }

    #[test]
    fn rejects_duplicate_keys() {
        let error = parse_catalog(Path::new("xx.tsv"), "\tOpen\tOuvrir\n\tOpen\tOuvert\n").err();
        assert!(error.is_some_and(|message| message.contains("duplicate key")));
    }

    #[test]
    fn registered_languages_and_report_are_stable() {
        let registry = r#"
            pub static LANGUAGES: [LangInfo; 3] = [
                LangInfo { code: "zh", source: catalog!("zh") },
                LangInfo { code: "en", source: b"" },
                LangInfo { code: "fr", source: catalog!("fr") },
            ];
            impl LangInfo {}
        "#;
        let languages = registered_languages(registry);
        assert!(languages.is_ok());
        let languages = languages.unwrap_or_default();
        let english_keys = BTreeSet::from([Key { context: String::new(), source: "Open".into() }, Key { context: String::new(), source: "Save".into() }]);
        let translations = BTreeMap::from([("fr".to_string(), BTreeSet::from([Key { context: String::new(), source: "Open".into() }]))]);
        assert_eq!(
            format_report(&languages, &english_keys, &translations),
            "UI translation coverage (translated source keys / source keys)\n\
             language       translated  coverage\n\
             en               2/2        100.00%\n\
             fr               1/2         50.00%\n\
             zh               0/2          0.00%\n"
        );
    }

    #[test]
    fn gathers_keys_from_ui_and_engine_sources() {
        let catalog = menu_catalog_keys(
            r#"(&["File", "Export"], "Export As…", Some("Cmd+Shift+S"), "file.exportAs"),
               (&["File"], "---", None, "---"),"#,
        );
        assert!(catalog.contains(&Key { context: String::new(), source: "File".into() }));
        assert!(catalog.contains(&Key { context: String::new(), source: "Export".into() }));
        assert!(catalog.contains(&Key { context: String::new(), source: "Export As…".into() }));
        assert!(!catalog.contains(&Key { context: String::new(), source: "---".into() }));

        let ui = ui_command_keys(
            r#"("view.zoomIn", "Zoom In", &["View"], Some("Cmd+=")),
               ("view.fit", "Fit", &[], None),"#,
        );
        assert!(ui.contains(&Key { context: String::new(), source: "Zoom In".into() }));
        assert!(ui.contains(&Key { context: String::new(), source: "View".into() }));

        let engine = vec![(
            PathBuf::from("commands.rs"),
            r#"
            macro_rules! spec { ($id:literal, $label:literal, $menu:expr) => { CommandSpec { id: $id, label: $label, menu: $menu } }; }
            spec!("file.open", "Open", &["File"]);
            spec!("brush.get", "Get Brush", &[]);
            CommandSpec { id: "layer.new", label: "New Layer", menu: &["Layer"] }
        "#
            .into(),
        )];
        let labels = engine_command_labels(&engine);
        assert!(labels.contains(&Key { context: String::new(), source: "Open".into() }));
        assert!(labels.contains(&Key { context: String::new(), source: "New Layer".into() }));
        assert!(labels.contains(&Key { context: String::new(), source: "Layer".into() }));
        assert!(!labels.contains(&Key { context: String::new(), source: "Get Brush".into() }));
    }

    #[test]
    fn plural_keys_come_from_source_calls() {
        let sources = vec![(PathBuf::from("ui.rs"), r#"trn(lang, count, "{n} layer", "{n} layers");"#.into())];
        assert!(plural_keys(&sources).contains(&Key { context: "@plural".into(), source: "{n} layer|{n} layers".into() }));
        // A string literal in the count expression is not one of the forms.
        let sources = vec![(PathBuf::from("ui.rs"), r#"trn(lang, r["files"].len() as u64, "{n} file", "{n} files");"#.into())];
        let keys = plural_keys(&sources);
        assert!(keys.contains(&Key { context: "@plural".into(), source: "{n} file|{n} files".into() }), "{keys:?}");
        assert_eq!(keys.len(), 1, "{keys:?}");
    }

    #[test]
    fn source_scanner_ignores_test_modules_with_lf_and_crlf() {
        for newline in ["\n", "\r\n"] {
            let source = format!("tl!(\"production key\");{newline}#[cfg(test)]{newline}mod tests {{{newline}tl!(\"test fixture key\");{newline}}}{newline}");
            let sources = vec![(PathBuf::from("ui.rs"), source)];
            let keys = tl_keys(&sources);
            assert!(keys.contains(&Key { context: String::new(), source: "production key".into() }));
            assert!(!keys.contains(&Key { context: String::new(), source: "test fixture key".into() }));
        }
    }

    #[test]
    fn source_key_set_is_covered_by_complete_catalogs() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap_or(Path::new("."));
        let keys = source_keys(root).unwrap_or_default();
        for (code, path) in [
            ("ja", "ja.tsv"),
            ("zh-hant", "zh-hant.tsv"),
            ("es", "es.tsv"),
            ("ru", "ru.tsv"),
            ("cs", "cs.tsv"),
            ("de", "de.tsv"),
            ("pt-br", "pt-br.tsv"),
            ("it", "it.tsv"),
            ("el", "el.tsv"),
            ("uk", "uk.tsv"),
            ("nl", "nl.tsv"),
            ("hu", "hu.tsv"),
            ("tr", "tr.tsv"),
            ("vi", "vi.tsv"),
            ("sr", "sr.tsv"),
        ] {
            let catalog_path = root.join("crates/ui-egui/src/i18n").join(path);
            let text = fs::read_to_string(&catalog_path).unwrap_or_default();
            let translations = parse_catalog(&catalog_path, &text).unwrap_or_default();
            let missing: Vec<_> = keys.difference(&translations).collect();
            assert!(missing.is_empty(), "{code}: missing source keys: {missing:#?}");
        }
    }
    #[test]
    fn plural_forms_follow_count_expressions_and_dynamic_forms_are_not_literal_keys() {
        let sources = vec![(
            PathBuf::from("ui.rs"),
            r#"trn(lang, r["files"].as_array().map_or(0, Vec::len) as u64, "Saved {n} file", "Saved {n} files");
               trn(lang, count, one, other);"#
                .into(),
        )];
        assert_eq!(plural_keys(&sources), BTreeSet::from([Key { context: "@plural".into(), source: "Saved {n} file|Saved {n} files".into() }]));
    }
}
