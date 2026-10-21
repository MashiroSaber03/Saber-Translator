//! Edit › Check Spelling… and glyph insertion (Window › Glyphs).
//!
//! `edit.checkSpelling` checks type layers against the bundled English dictionary
//! (`photocraft_text::spell`, SCOWL size 50) plus the user dictionary in the preferences
//! (`userDictionary`). Headless, it lists misspellings with suggestions and changes, changes all,
//! or adds words to the dictionary; the UI's dialog drives the same actions. Offsets are
//! character indices.
//!
//! `type.insertText` inserts a string (a glyph) into a type layer at a character position or
//! over a range.

use std::collections::HashSet;

use photocraft_doc::{Document, LayerContent, LayerId};
use photocraft_text::spell::Dictionary;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, layer_param};
use crate::type_cmds::{check_text_unlocked, norm_text, refresh, replace_text};
use crate::{EngineError, Result, Session};

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

fn byte_at(text: &str, ci: usize) -> usize {
    text.char_indices().nth(ci).map_or(text.len(), |(b, _)| b)
}

fn char_at(text: &str, bi: usize) -> usize {
    text[..bi.min(text.len())].chars().count()
}

/// Type layers top to bottom (Layers panel order), skipping fully locked ones.
fn type_layers(doc: &Document) -> Vec<LayerId> {
    doc.walk().into_iter().rev().filter(|(p, _, l)| matches!(l.content, LayerContent::Text(_)) && !doc.locks_at(p).all).map(|(_, _, l)| l.id).collect()
}

fn has_type(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    if type_layers(&d.doc).is_empty() { Err("the document has no type layers".into()) } else { Ok(()) }
}

fn text(doc: &Document, id: LayerId) -> Option<&str> {
    match &doc.layer(id)?.content {
        LayerContent::Text(t) => Some(&t.text),
        _ => None,
    }
}

/// The user dictionary plus `ignore` words, lowercased.
fn accepted(s: &Session, p: &Value) -> HashSet<String> {
    let mut set: HashSet<String> = s.prefs().user_dictionary.iter().map(|w| w.to_lowercase()).collect();
    if let Some(a) = p.get("ignore").and_then(Value::as_array) {
        set.extend(a.iter().filter_map(Value::as_str).map(str::to_lowercase));
    }
    set
}

fn layers_param(s: &Session, p: &Value) -> Vec<LayerId> {
    let Some(st) = s.active() else { return Vec::new() };
    let all = type_layers(&st.doc);
    if p.get("allLayers").and_then(Value::as_bool) == Some(false) || p.get("layer").is_some() {
        let active = layer_param(s, p).ok();
        all.into_iter().filter(|id| Some(*id) == active).collect()
    } else {
        all
    }
}

fn list(s: &Session, p: &Value) -> Result<Value> {
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let dict = Dictionary::english();
    let ok = accepted(s, p);
    let max = p.get("suggestions").and_then(Value::as_u64).unwrap_or(5) as usize;
    let mut out = Vec::new();
    for id in layers_param(s, p) {
        let Some(t) = text(&st.doc, id) else { continue };
        for m in dict.misspellings(t, &ok) {
            out.push(
                json!({ "layer": id.0, "start": char_at(t, m.start), "end": char_at(t, m.end), "word": m.word, "suggestions": dict.suggest(&m.word, max) }),
            );
        }
    }
    Ok(json!({ "misspellings": out, "count": out.len() }))
}

/// Replaces one occurrence (verified to still be `word` when given).
fn change(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "edit.checkSpelling";
    let id = LayerId(p.get("layer").and_then(Value::as_u64).ok_or_else(|| bad(cmd, "`change` needs `layer`"))?);
    let start = p.get("start").and_then(Value::as_u64).ok_or_else(|| bad(cmd, "`change` needs `start`"))? as usize;
    let end = p.get("end").and_then(Value::as_u64).ok_or_else(|| bad(cmd, "`change` needs `end`"))? as usize;
    let to = p.get("replace").and_then(Value::as_str).ok_or_else(|| bad(cmd, "`change` needs `replace`"))?.to_string();
    let doc = s.active().ok_or(EngineError::NoDocument)?.doc.clone();
    let t = text(&doc, id).ok_or_else(|| bad(cmd, format!("layer {} is not a type layer", id.0)))?;
    check_text_unlocked(&doc, id)?;
    let (a, b) = (byte_at(t, start), byte_at(t, end.max(start)));
    if let Some(w) = p.get("word").and_then(Value::as_str)
        && &t[a..b] != w
    {
        return Err(bad(cmd, format!("the text at {start}..{end} is {:?}, not {w:?}", &t[a..b])));
    }
    s.edit("Check Spelling", |doc, _| {
        let snapshot = doc.clone();
        if let Some(LayerContent::Text(t)) = doc.layer_mut(id).map(|l| &mut l.content) {
            replace_text(t, a, b, &to);
            refresh(&snapshot, t);
        }
        Ok(())
    })?;
    Ok(json!({ "changed": 1, "layer": id.0, "start": start, "end": start + to.chars().count() }))
}

/// Replaces every whole-word, case-sensitive occurrence of `word` in the checked layers.
fn change_all(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "edit.checkSpelling";
    let word = p.get("word").and_then(Value::as_str).filter(|w| !w.is_empty()).ok_or_else(|| bad(cmd, "`changeAll` needs `word`"))?.to_string();
    let to = p.get("replace").and_then(Value::as_str).ok_or_else(|| bad(cmd, "`changeAll` needs `replace`"))?.to_string();
    let doc = s.active().ok_or(EngineError::NoDocument)?.doc.clone();
    let plan: Vec<(LayerId, Vec<(usize, usize)>)> = layers_param(s, p)
        .into_iter()
        .filter_map(|id| {
            let t = text(&doc, id)?;
            let hits: Vec<(usize, usize)> = photocraft_text::spell::words(t).into_iter().filter(|(a, b)| t[*a..*b] == word).collect();
            (!hits.is_empty()).then_some((id, hits))
        })
        .collect();
    let count: usize = plan.iter().map(|(_, h)| h.len()).sum();
    if count > 0 {
        s.edit("Check Spelling", |doc, _| {
            let snapshot = doc.clone();
            for (id, hits) in &plan {
                if let Some(LayerContent::Text(t)) = doc.layer_mut(*id).map(|l| &mut l.content) {
                    for (a, b) in hits.iter().rev() {
                        replace_text(t, *a, *b, &to);
                    }
                    refresh(&snapshot, t);
                }
            }
            Ok(())
        })?;
    }
    Ok(json!({ "changed": count, "layers": plan.len() }))
}

fn add_word(s: &mut Session, p: &Value) -> Result<Value> {
    let word = p
        .get("word")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|w| !w.is_empty() && !w.contains(char::is_whitespace))
        .ok_or_else(|| bad("edit.checkSpelling", "`add` needs a single `word`"))?
        .to_string();
    let added = s.edit_prefs(|pr| {
        if pr.user_dictionary.iter().any(|w| w.eq_ignore_ascii_case(&word)) {
            false
        } else {
            pr.user_dictionary.push(word.clone());
            pr.user_dictionary.sort_by_key(|w| w.to_lowercase());
            true
        }
    });
    Ok(json!({ "added": added, "word": word, "userDictionary": s.prefs().user_dictionary }))
}

fn remove_word(s: &mut Session, p: &Value) -> Result<Value> {
    let word = p.get("word").and_then(Value::as_str).unwrap_or("").to_string();
    let removed = s.edit_prefs(|pr| {
        let n = pr.user_dictionary.len();
        pr.user_dictionary.retain(|w| !w.eq_ignore_ascii_case(&word));
        n != pr.user_dictionary.len()
    });
    Ok(json!({ "removed": removed, "userDictionary": s.prefs().user_dictionary }))
}

fn check_spelling(s: &mut Session, p: &Value) -> Result<Value> {
    match p.get("action").and_then(Value::as_str).unwrap_or("list") {
        "list" | "check" => list(s, p),
        "change" => change(s, p),
        "changeAll" => change_all(s, p),
        "add" | "addToDictionary" => add_word(s, p),
        "removeFromDictionary" => remove_word(s, p),
        "suggest" => {
            let w = p.get("word").and_then(Value::as_str).unwrap_or("");
            let ok = accepted(s, p);
            let d = Dictionary::english();
            Ok(
                json!({ "word": w, "correct": d.check(w, &ok), "suggestions": d.suggest(w, p.get("suggestions").and_then(Value::as_u64).unwrap_or(5) as usize) }),
            )
        }
        other => Err(bad("edit.checkSpelling", format!("unknown action {other:?}"))),
    }
}

fn insert_text(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "type.insertText";
    let id = layer_param(s, p)?;
    let new = p.get("text").and_then(Value::as_str).filter(|t| !t.is_empty()).ok_or_else(|| bad(cmd, "missing `text`"))?.to_string();
    let new = norm_text(&new);
    let doc = s.active().ok_or(EngineError::NoDocument)?.doc.clone();
    let t = text(&doc, id).ok_or_else(|| bad(cmd, format!("layer {} is not a type layer", id.0)))?;
    check_text_unlocked(&doc, id)?;
    let n = t.chars().count();
    let (a, b) = match (p.get("range").and_then(Value::as_array), p.get("at").and_then(Value::as_u64)) {
        (Some(r), _) if r.len() == 2 => {
            let x = r[0].as_u64().unwrap_or(0) as usize;
            let y = r[1].as_u64().unwrap_or(x as u64) as usize;
            (x.min(y).min(n), x.max(y).min(n))
        }
        (_, Some(at)) => (at as usize, at as usize),
        _ => (n, n),
    };
    let (a, b) = (a.min(n), b.min(n));
    let (ba, bb) = (byte_at(t, a), byte_at(t, b));
    let label = p.get("label").and_then(Value::as_str).unwrap_or("Insert Glyph").to_string();
    s.edit(&label, |doc, _| {
        let snapshot = doc.clone();
        if let Some(LayerContent::Text(t)) = doc.layer_mut(id).map(|l| &mut l.content) {
            replace_text(t, ba, bb, &new);
            refresh(&snapshot, t);
        }
        Ok(())
    })?;
    let caret = a + new.chars().count();
    Ok(json!({ "layer": id.0, "caret": caret }))
}

fn has_text_layer(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    match d.active_layer.and_then(|id| d.doc.layer(id)).map(|l| &l.content) {
        Some(LayerContent::Text(_)) => Ok(()),
        _ => Err("the active layer is not a type layer".into()),
    }
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "edit.checkSpelling",
            label: "Check Spelling…",
            menu: &["Edit"],
            shortcut: None,
            params: r#"{"action":"list|change|changeAll|addToDictionary|removeFromDictionary|suggest"="list","allLayers":bool=true,"layer":id? (check one layer),"ignore":[word]? (Ignore All for this check),"suggestions":n=5; change: "layer","start","end" (chars),"word"? (verified),"replace"; changeAll: "word","replace"; addToDictionary/removeFromDictionary/suggest: "word"} → list: {"misspellings":[{"layer","start","end","word","suggestions"}],"count"}"#,
            enabled: has_type,
            journal: true,
            run: check_spelling,
        },
        CommandSpec {
            id: "type.insertText",
            label: "Insert Glyph",
            menu: &[],
            shortcut: None,
            params: r#"{"layer":id?,"text":str,"at":char? (default: end of text),"range":[startChar,endChar]? (replaced),"label":str?} → {"layer","caret"}"#,
            enabled: has_text_layer,
            journal: true,
            run: insert_text,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session(text: &str) -> (Session, u64) {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 300, "height": 100, "background": "white"})).unwrap();
        let r = s.execute("type.create", json!({"x": 10, "y": 50, "text": text, "size": 12})).unwrap();
        (s, r["layer"].as_u64().unwrap())
    }

    fn text_of(s: &Session, id: u64) -> String {
        text(&s.active().unwrap().doc, LayerId(id)).unwrap().to_string()
    }

    #[test]
    fn all_locked_layer_rejects_insert_text_and_spelling_change_without_history() {
        // #4188: Lock All stopped type.edit (#3132) but not these two text edits.
        let (mut s, id) = session("Teh");
        // A second, unlocked type layer keeps Check Spelling enabled.
        s.execute("type.create", json!({"x": 10, "y": 90, "text": "zzz", "size": 12})).unwrap();
        s.execute("layer.setProps", json!({"layer": id, "locked": true})).unwrap();
        let history = s.active().unwrap().history.past_len();
        for (cmd, params) in [
            ("type.insertText", json!({"layer": id, "text": "b"})),
            ("edit.checkSpelling", json!({"action": "change", "layer": id, "start": 0, "end": 3, "word": "Teh", "replace": "The"})),
        ] {
            let result = s.execute(cmd, params);
            assert!(matches!(&result, Err(EngineError::Other(message)) if message.contains("locked")), "{cmd}: {result:?}");
            assert_eq!(text_of(&s, id), "Teh", "{cmd}");
            assert_eq!(s.active().unwrap().history.past_len(), history, "{cmd}");
        }
        s.execute("layer.setProps", json!({"layer": id, "locked": false})).unwrap();
        s.execute("type.insertText", json!({"layer": id, "text": "b"})).unwrap();
        assert_eq!(text_of(&s, id), "Tehb");
    }

    #[test]
    fn lists_changes_and_learns_words() {
        let (mut s, id) = session("Teh quick brwon fox, teh end. Photocraft");
        let r = s.execute("edit.checkSpelling", json!({})).unwrap();
        let words: Vec<&str> = r["misspellings"].as_array().unwrap().iter().map(|m| m["word"].as_str().unwrap()).collect();
        assert_eq!(words, vec!["Teh", "brwon", "teh", "Photocraft"]);
        let first = &r["misspellings"][0];
        assert_eq!((first["start"].as_u64(), first["end"].as_u64()), (Some(0), Some(3)));
        assert_eq!(first["suggestions"][0], "The");
        // Change one occurrence (verified), then all of another.
        s.execute("edit.checkSpelling", json!({"action": "change", "layer": id, "start": 0, "end": 3, "word": "Teh", "replace": "The"})).unwrap();
        assert!(s.execute("edit.checkSpelling", json!({"action": "change", "layer": id, "start": 0, "end": 3, "word": "Teh", "replace": "X"})).is_err());
        let r = s.execute("edit.checkSpelling", json!({"action": "changeAll", "word": "brwon", "replace": "brown"})).unwrap();
        assert_eq!(r["changed"], 1);
        assert_eq!(text_of(&s, id), "The quick brown fox, teh end. Photocraft");
        // One undo step per change.
        s.execute("edit.undo", json!({})).unwrap();
        assert!(text_of(&s, id).contains("brwon"));
        s.execute("edit.redo", json!({})).unwrap();
        // Ignore All and Add to Dictionary.
        let r = s.execute("edit.checkSpelling", json!({"ignore": ["teh"]})).unwrap();
        assert_eq!(r["count"], 1);
        s.execute("edit.checkSpelling", json!({"action": "addToDictionary", "word": "Photocraft"})).unwrap();
        assert_eq!(s.prefs().user_dictionary, vec!["Photocraft".to_string()]);
        let r = s.execute("edit.checkSpelling", json!({})).unwrap();
        assert_eq!(r["count"], 1);
        s.execute("edit.checkSpelling", json!({"action": "removeFromDictionary", "word": "photocraft"})).unwrap();
        assert!(s.prefs().user_dictionary.is_empty());
        assert!(s.execute("edit.checkSpelling", json!({"action": "bogus"})).is_err());
        let r = s.execute("edit.checkSpelling", json!({"action": "suggest", "word": "recieve"})).unwrap();
        assert_eq!(r["correct"], false);
    }

    #[test]
    fn disabled_without_type() {
        let mut s = Session::new();
        assert!(!s.is_enabled("edit.checkSpelling"));
        s.execute("file.new", json!({"width": 10, "height": 10})).unwrap();
        assert!(!s.is_enabled("edit.checkSpelling"));
        assert!(!s.is_enabled("type.insertText"));
    }

    #[test]
    fn insert_glyph_at_caret_and_over_range() {
        let (mut s, id) = session("ab");
        s.execute("type.insertText", json!({"layer": id, "text": "→", "at": 1})).unwrap();
        assert_eq!(text_of(&s, id), "a→b");
        let r = s.execute("type.insertText", json!({"layer": id, "text": "€", "range": [0, 2]})).unwrap();
        assert_eq!((text_of(&s, id).as_str(), r["caret"].as_u64()), ("€b", Some(1)));
        s.execute("type.insertText", json!({"text": "!"})).unwrap();
        assert_eq!(text_of(&s, id), "€b!");
        assert!(s.execute("type.insertText", json!({"text": ""})).is_err());
    }

    #[test]
    fn insert_crlf_reports_caret_in_normalized_text() {
        let (mut s, id) = session("");
        let inserted = s.execute("type.insertText", json!({"layer": id, "text": "x\r\ny"})).unwrap();

        assert_eq!(inserted["caret"], 3);
        assert_eq!(s.execute("type.info", json!({"layer": id})).unwrap()["text"], "x\ny");
        assert_eq!(s.execute("type.caret", json!({"layer": id, "index": inserted["caret"]})).unwrap()["index"], inserted["caret"]);
    }
}
