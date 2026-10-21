//! Batch output names are planned before opening or writing files. Names substituted from an
//! input are literal: braces in a source filename must never be interpreted as template tokens.

use super::{EngineError, Result, Value, join, stem};

fn bad(message: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: "file.automate.batch".into(), msg: message.into() }
}

enum Part<'a> {
    Literal(&'a str),
    Name,
    Serial,
    Date,
}

fn parse(template: &str) -> Result<Vec<Part<'_>>> {
    let mut parts = Vec::new();
    let mut rest = template;
    while let Some(at) = rest.find(['{', '}']) {
        parts.push(Part::Literal(&rest[..at]));
        rest = &rest[at..];
        if rest.starts_with('}') {
            return Err(bad("naming: unmatched closing brace"));
        }
        let end = rest.find('}').ok_or_else(|| bad("naming: unclosed token"))?;
        parts.push(match &rest[1..end] {
            "name" => Part::Name,
            "serial" => Part::Serial,
            "date" => Part::Date,
            token => return Err(bad(format!("naming: unknown token {{{token}}}; use {{name}}, {{serial}} or {{date}}"))),
        });
        rest = &rest[end + 1..];
    }
    parts.push(Part::Literal(rest));
    Ok(parts)
}

fn expand(parts: &[Part<'_>], name: &str, serial: u32, date: &str) -> String {
    let mut out = String::new();
    for part in parts {
        match part {
            Part::Literal(s) => out.push_str(s),
            Part::Name => out.push_str(name),
            Part::Serial => out.push_str(&format!("{serial:03}")),
            Part::Date => out.push_str(date),
        }
    }
    out
}

/// An absent or empty template preserves the old filename path exactly. An explicit template
/// generates one component per input; the export format supplies the extension separately.
pub(super) fn plan(inputs: &[String], output: &str, format: &str, params: &Value) -> Result<Option<Vec<String>>> {
    let start = match params.get("startingSerial") {
        None => 1,
        Some(n) => {
            n.as_u64().and_then(|n| u32::try_from(n).ok()).filter(|&n| n > 0).ok_or_else(|| bad("startingSerial must be an integer from 1 to 4294967295"))?
        }
    };
    let template = match params.get("naming") {
        None => "",
        Some(Value::String(s)) => s,
        Some(_) => return Err(bad("naming must be a string")),
    };
    if template.is_empty() {
        return Ok(None);
    }
    if template.len() > 255 {
        return Err(bad("naming must be at most 255 bytes"));
    }
    let parts = parse(template)?;
    let has_serial = parts.iter().any(|p| matches!(p, Part::Serial));
    let timestamp = crate::analysis_cmds::now_iso();
    let date = timestamp.get(..10).unwrap_or("");
    if date.is_empty() && parts.iter().any(|p| matches!(p, Part::Date)) {
        return Err(bad("naming: the current UTC date is unavailable"));
    }
    let mut outputs = Vec::with_capacity(inputs.len());
    for (index, path) in inputs.iter().enumerate() {
        let serial = if has_serial {
            u32::try_from(index).ok().and_then(|index| start.checked_add(index)).ok_or_else(|| bad("startingSerial plus the input count exceeds 4294967295"))?
        } else {
            start
        };
        let base = expand(&parts, &stem(path), serial, date);
        if base.is_empty() || base.ends_with(['.', ' ']) {
            return Err(bad("naming must produce a nonempty filename stem without a trailing dot or space"));
        }
        let ext =
            if format == "same" { path.rsplit('.').next().unwrap_or("png").to_ascii_lowercase() } else { format.trim_start_matches('.').to_ascii_lowercase() };
        let name = format!("{base}.{ext}");
        if name.len() > 255 || name.chars().any(|c| c.is_control() || "/\\<>:\"|?*".contains(c)) {
            return Err(bad("naming must produce a filename of at most 255 bytes without path separators or reserved characters"));
        }
        outputs.push(join(output, &name));
    }
    // Renaming within the source folder could replace a later input before it is read. Resolve
    // existing paths so aliases such as `source/./001.png` cannot bypass the check. The default
    // (no template) path keeps its existing in-place behavior.
    #[cfg(not(target_arch = "wasm32"))]
    {
        let sources: std::collections::HashSet<_> = inputs.iter().filter_map(|p| std::fs::canonicalize(p).ok()).collect();
        for output in &outputs {
            if std::fs::canonicalize(output).ok().is_some_and(|p| sources.contains(&p)) {
                return Err(bad(format!("naming would replace an input file: {output}; choose a different destination or template")));
            }
        }
    }
    Ok(Some(outputs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substituted_names_are_literal_and_serials_do_not_wrap_or_truncate() {
        let parts = parse("{date}_{name}_{serial}").unwrap();
        assert_eq!(expand(&parts, "photo{serial}", 7, "2026-10-13"), "2026-10-13_photo{serial}_007");
        assert_eq!(expand(&parts, "photo", 1000, "2026-10-13"), "2026-10-13_photo_1000");
    }
}
