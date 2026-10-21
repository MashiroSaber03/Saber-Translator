//! Character Styles and Paragraph Styles (Window › Character Styles / Paragraph Styles).
//!
//! `type.characterStyle.*` and `type.paragraphStyle.*`: new, duplicate, delete, rename, set
//! (Style Options), apply, redefine (from the selected text), clear override and list. The style
//! model is [`photocraft_doc::TextStyles`]; text runs keep fully resolved attributes plus a style
//! reference, so whatever differs from the referenced styles is a local override ("+").
//!
//! Text targets: `layer` (one type layer) or `layers` (several), default the selected type
//! layers; `range: [startChar, endChar]` restricts a single layer to part of its text.
//! Changing a style's definition re-renders every type layer that uses it, keeping overrides.

use std::collections::HashSet;

use photocraft_doc::text::{CharStyle, ParagraphRun, ParagraphStyle, TextRun};
use photocraft_doc::text_styles::{self as ts, CharacterStyleDef, ParagraphStyleDef, StyleAttrs};
use photocraft_doc::{Document, LayerContent, LayerId, TextLayer, TextStyles};
use serde_json::{Map, Value, json};

use crate::commands::CommandSpec;
use crate::type_cmds::{apply_char_props, apply_para_props, refresh};
use crate::{EngineError, Result, Session};

const FAMILY: &str = photocraft_text::fonts::DEFAULT_FAMILY;

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

fn exhausted_id(paragraph: bool) -> EngineError {
    EngineError::Other(format!("{} style id space exhausted", if paragraph { "paragraph" } else { "character" }))
}

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

fn text_of(doc: &Document, id: LayerId) -> Option<&TextLayer> {
    match &doc.layer(id)?.content {
        LayerContent::Text(t) => Some(t),
        _ => None,
    }
}

/// Type layers a style command acts on: `layer`, `layers`, else the selected type layers.
fn targets(s: &Session, p: &Value) -> Vec<LayerId> {
    let Some(st) = s.active() else { return Vec::new() };
    let ids: Vec<LayerId> = if let Some(id) = p.get("layer").and_then(Value::as_u64) {
        vec![LayerId(id)]
    } else if let Some(a) = p.get("layers").and_then(Value::as_array) {
        a.iter().filter_map(Value::as_u64).map(LayerId).collect()
    } else {
        st.selected_layers()
    };
    ids.into_iter().filter(|id| text_of(&st.doc, *id).is_some()).collect()
}

fn has_target(s: &Session) -> std::result::Result<(), String> {
    has_doc(s)?;
    if targets(s, &Value::Null).is_empty() { Err("select a type layer".into()) } else { Ok(()) }
}

fn byte_at(text: &str, ci: usize) -> usize {
    text.char_indices().nth(ci).map_or(text.len(), |(b, _)| b)
}

/// `range` in characters → bytes (whole text when absent).
fn range_of(text: &str, p: &Value) -> (usize, usize) {
    match p.get("range").and_then(Value::as_array) {
        Some(r) if r.len() == 2 => {
            let a = r[0].as_u64().unwrap_or(0) as usize;
            let b = r[1].as_u64().unwrap_or(a as u64) as usize;
            (byte_at(text, a.min(b)), byte_at(text, a.max(b)))
        }
        _ => (0, text.len()),
    }
}

fn id_param(cmd: &str, p: &Value) -> Result<Option<u32>> {
    crate::commands::u32_id_param(cmd, p, "id")
}

// ---------- attribute parameters ----------

/// A character style with every field changed from the default, to see which fields a set of
/// Character-panel keys touches.
fn perturbed_char() -> CharStyle {
    use photocraft_doc::text::{Caps, FontFeature, FontVariation, Kerning};
    CharStyle {
        font_family: "\u{1}".into(),
        font_style: "\u{1}".into(),
        postscript_name: Some("\u{1}".into()),
        weight: 1,
        italic: true,
        size_pt: -1.0,
        color: photocraft_doc::Color::rgba(0.123, 0.456, 0.789, 0.5),
        tracking: -12345.0,
        leading_pt: Some(-1.0),
        baseline_shift_pt: -1.0,
        horizontal_scale: -1.0,
        vertical_scale: -1.0,
        underline: true,
        strikethrough: true,
        faux_bold: true,
        faux_italic: true,
        kerning: Kerning::Off,
        kern: -12345.0,
        caps: Caps::AllCaps,
        ligatures: false,
        discretionary_ligatures: true,
        features: vec![FontFeature { tag: "\u{1}".into(), value: 9 }],
        variations: vec![FontVariation { axis: "\u{1}".into(), value: -1.0 }],
        language: Some("\u{1}".into()),
        style_sheet: None,
    }
}

fn perturbed_para() -> ParagraphStyle {
    use photocraft_doc::text::{TextAlign, TextDirection};
    ParagraphStyle {
        align: TextAlign::JustifyAll,
        first_line_indent_pt: -1.0,
        start_indent_pt: -1.0,
        end_indent_pt: -1.0,
        space_before_pt: -1.0,
        space_after_pt: -1.0,
        auto_leading: -1.0,
        direction: TextDirection::Rtl,
        hyphenate: true,
        style_sheet: None,
    }
}

/// Fields set by Character/Paragraph-panel keys (`size`, `font`, `align`, …): the fields whose
/// value is the same whatever they were before.
fn touched<T: serde::Serialize + Default + Clone>(perturbed: T, apply: impl Fn(&mut T)) -> StyleAttrs {
    let (mut a, mut b) = (T::default(), perturbed);
    apply(&mut a);
    apply(&mut b);
    let obj = |x: &T| match serde_json::to_value(x) {
        Ok(Value::Object(m)) => m,
        _ => Map::new(),
    };
    let all_b = obj(&b);
    obj(&a).into_iter().filter(|(k, v)| k != "style_sheet" && all_b.get(k) == Some(v)).collect()
}

/// Splits `attrs` into character and paragraph attribute sets. Keys may be model field names
/// (`size_pt`, `align`: values as in `type.info`) or the Character/Paragraph panel keys of
/// `type.setStyle` (`size`, `font`, `align`, …).
fn parse_attrs(cmd: &str, v: &Value) -> Result<(StyleAttrs, StyleAttrs)> {
    let Some(obj) = v.as_object() else {
        return if v.is_null() { Ok((Map::new(), Map::new())) } else { Err(bad(cmd, "`attrs` must be an object")) };
    };
    let char_keys: HashSet<String> = ts::char_attrs(&CharStyle::default()).keys().cloned().collect();
    let para_keys: HashSet<String> = ts::para_attrs(&ParagraphStyle::default()).keys().cloned().collect();
    let (mut c, mut pa, mut friendly) = (Map::new(), Map::new(), Map::new());
    for (k, x) in obj {
        // Names shared by the model and the panels (`color`, `align`, `caps`, …) are read as
        // model values when they parse as such, else as panel values ("#ff0000", "center").
        let one: StyleAttrs = std::iter::once((k.clone(), x.clone())).collect();
        if char_keys.contains(k) && ts::validate::<CharStyle>(&one).is_ok() {
            c.insert(k.clone(), x.clone());
        } else if para_keys.contains(k) && ts::validate::<ParagraphStyle>(&one).is_ok() {
            pa.insert(k.clone(), x.clone());
        } else {
            friendly.insert(k.clone(), x.clone());
        }
    }
    if !friendly.is_empty() {
        let f = Value::Object(friendly.clone());
        let fc = touched(perturbed_char(), |st| {
            apply_char_props(st, &f);
        });
        let fp = touched(perturbed_para(), |st| {
            apply_para_props(st, &f);
        });
        if fc.is_empty() && fp.is_empty() {
            return Err(bad(cmd, format!("unknown style attributes: {}", friendly.keys().cloned().collect::<Vec<_>>().join(", "))));
        }
        c.extend(fc);
        pa.extend(fp);
    }
    ts::validate::<CharStyle>(&c).map_err(|e| bad(cmd, e))?;
    ts::validate::<ParagraphStyle>(&pa).map_err(|e| bad(cmd, e))?;
    Ok((c, pa))
}

// ---------- restyling ----------

/// What to do to the targeted text while re-resolving against new styles.
#[derive(Clone, Copy, Default)]
struct Op {
    /// Set the character style of the range (`Some(None)` = "None").
    set_char: Option<Option<u32>>,
    /// Set the paragraph style of paragraphs touching the range (`Some(None)` = Basic).
    set_para: Option<Option<u32>>,
    /// Drop character overrides in the range.
    clear_char: bool,
    /// Drop paragraph overrides of paragraphs touching the range.
    clear_para: bool,
}

/// Re-resolves every run and paragraph of `t` against `new` (keeping overrides made against
/// `old`), applying `op` to bytes `a..b`. Returns whether anything changed.
fn restyle(t: &mut TextLayer, old: &TextStyles, new: &TextStyles, range: Option<(usize, usize)>, op: Op) -> bool {
    let before = (t.runs.clone(), t.paragraphs.clone());
    let text = t.text.clone();
    let (a, b) = range.unwrap_or((0, text.len()));
    let whole = range.is_none();
    // Paragraphs.
    let old_paras = t.paragraph_runs();
    let starts: Vec<usize> = old_paras
        .iter()
        .scan(0, |acc, r| {
            let s = *acc;
            *acc += r.len;
            Some(s)
        })
        .collect();
    let para_at = |off: usize| old_paras[starts.iter().rposition(|&s| s <= off).unwrap_or(0)].style.clone();
    let mut paras: Vec<(std::ops::Range<usize>, ParagraphStyle, Option<u32>)> = Vec::new(); // range, new style, old ref
    for pr in photocraft_text::layout::split_paragraphs(&text) {
        let old_p = para_at(pr.start);
        let touches = whole || (pr.start < b && pr.end > a) || (a == b && a >= pr.start && (a < pr.end || pr.end == text.len()));
        let new_ref = match op.set_para {
            Some(r) if touches => r,
            _ => old_p.style_sheet,
        };
        let np = if op.clear_para && touches { new.resolve_para(new_ref) } else { new.restyle_para(old, &old_p, new_ref) };
        paras.push((pr, np, old_p.style_sheet));
    }
    // Character runs, split at paragraph boundaries and the range edges.
    let mut cuts: Vec<usize> = paras.iter().map(|p| p.0.start).collect();
    cuts.extend([a, b, text.len()]);
    let mut runs: Vec<(usize, CharStyle)> = Vec::new();
    let mut at = 0;
    for r in t.char_runs() {
        let (rs, re) = (at, at + r.len);
        at = re;
        let mut edges: Vec<usize> = cuts.iter().copied().filter(|&c| c > rs && c < re).collect();
        edges.sort_unstable();
        edges.dedup();
        let mut s0 = rs;
        for e in edges.into_iter().chain(std::iter::once(re)) {
            if e <= s0 && rs != re {
                continue;
            }
            let pi = paras.iter().rposition(|p| p.0.start <= s0).unwrap_or(0);
            let (old_pref, new_pref) = (paras[pi].2, paras[pi].1.style_sheet);
            let inside = whole || (s0 >= a && e <= b && a < b);
            let chr = match op.set_char {
                Some(c) if inside => c,
                _ => r.style.style_sheet,
            };
            let st = if op.clear_char && inside {
                new.resolve_char(new_pref, chr, FAMILY)
            } else {
                new.restyle_char(old, &r.style, old_pref, new_pref, chr, FAMILY)
            };
            match runs.last_mut() {
                Some(last) if last.1 == st => last.0 += e - s0,
                _ => runs.push((e - s0, st)),
            }
            s0 = e;
        }
    }
    let mut pruns: Vec<(usize, ParagraphStyle)> = Vec::new();
    for (r, st, _) in paras {
        match pruns.last_mut() {
            Some(last) if last.1 == st => last.0 += r.len(),
            _ => pruns.push((r.len(), st)),
        }
    }
    if runs.is_empty() {
        let first = t.char_runs().into_iter().next().map(|r| r.style).unwrap_or_default();
        runs.push((0, new.restyle_char(old, &first, None, pruns.first().and_then(|p| p.1.style_sheet), op.set_char.unwrap_or(first.style_sheet), FAMILY)));
    }
    t.runs = runs.into_iter().map(|(len, style)| TextRun { len, style }).collect();
    t.paragraphs = pruns.into_iter().map(|(len, style)| ParagraphRun { len, style }).collect();
    t.sync_summary();
    (t.runs.clone(), t.paragraphs.clone()) != before
}

/// Replaces the document's styles with `new` and re-resolves every type layer (as one history
/// step), applying `op` to `target` (layer + optional byte range).
fn commit(s: &mut Session, label: &str, new: TextStyles, target: &[Target], op: Op) -> Result<usize> {
    s.edit(label, |doc, _| Ok(apply_styles(doc, &new, target, op)))
}

// Shared by history-backed commands and isolated preview documents.
fn apply_styles(doc: &mut Document, new: &TextStyles, target: &[Target], op: Op) -> usize {
    let old = doc.text_styles.clone();
    doc.text_styles = new.clone();
    let snapshot = doc.clone();
    let ids: Vec<LayerId> = doc.walk().into_iter().filter(|(_, _, l)| matches!(l.content, LayerContent::Text(_))).map(|(_, _, l)| l.id).collect();
    let mut changed = 0;
    for id in ids {
        let tgt = target.iter().find(|(t, _)| *t == id);
        let Some(LayerContent::Text(t)) = doc.layer_mut(id).map(|l| &mut l.content) else { continue };
        let did = match tgt {
            Some((_, r)) => restyle(t, &old, new, *r, op),
            None => restyle(t, &old, new, Some((0, 0)), Op::default()),
        };
        if did {
            refresh(&snapshot, t);
            changed += 1;
        }
    }
    changed
}

/// A type layer and an optional byte range in it.
type Target = (LayerId, Option<(usize, usize)>);

/// Target list with byte ranges (`range` applies only to a single layer).
fn target_ranges(s: &Session, p: &Value) -> Result<Vec<Target>> {
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let ids = targets(s, p);
    if ids.is_empty() {
        return Err(EngineError::Other("no type layer to apply the style to".into()));
    }
    Ok(ids
        .into_iter()
        .map(|id| {
            let r = (p.get("range").is_some() && p.get("layers").is_none()).then(|| text_of(&st.doc, id).map(|t| range_of(&t.text, p))).flatten();
            (id, r)
        })
        .collect())
}

/// The run and paragraph at the start of the target (for new/redefine), with the layer and the
/// run's byte range.
fn sample(s: &Session, p: &Value) -> Option<(CharStyle, ParagraphStyle, LayerId, (usize, usize))> {
    let st = s.active()?;
    let id = *targets(s, p).first()?;
    let t = text_of(&st.doc, id)?;
    let (a, _) = range_of(&t.text, p);
    let mut at = 0;
    let mut run = None;
    for r in t.char_runs() {
        let (s0, e0) = (at, at + r.len);
        at = e0;
        if a < e0 || e0 == t.text.len() {
            run = Some((r.style, (s0, e0)));
            break;
        }
    }
    let (run, span) = run?;
    let mut at = 0;
    let paras = t.paragraph_runs();
    let para = paras.iter().find(|r| {
        let hit = a < at + r.len || at + r.len == t.text.len();
        at += r.len;
        hit
    })?;
    Some((run, para.style.clone(), id, span))
}

/// Character / paragraph style state of the target: (char id, char override, para id, para
/// override), over the whole target (mixed ids → None).
fn current(s: &Session, p: &Value) -> Option<Value> {
    let st = s.active()?;
    let styles = &st.doc.text_styles;
    let tr = target_ranges(s, p).ok()?;
    let (mut chars, mut paras) = (Vec::new(), Vec::new());
    let (mut c_over, mut p_over) = (false, false);
    for (id, r) in &tr {
        let t = text_of(&st.doc, *id)?;
        let (a, b) = r.unwrap_or((0, t.text.len()));
        let pr = t.paragraph_runs();
        let mut at = 0;
        let mut para_spans = Vec::new();
        for x in &pr {
            let (s0, e0) = (at, at + x.len);
            at = e0;
            if (s0 < b && e0 > a) || (a == b && a >= s0 && (a < e0 || e0 == t.text.len())) || t.text.is_empty() {
                paras.push(x.style.style_sheet.unwrap_or(0));
                p_over |= !styles.para_overrides(&x.style).is_empty();
            }
            para_spans.push((s0, e0, x.style.style_sheet));
        }
        let mut at = 0;
        for x in t.char_runs() {
            let (s0, e0) = (at, at + x.len);
            at = e0;
            if (s0 < b && e0 > a) || (a == b && a >= s0 && a <= e0) || t.text.is_empty() {
                let pref = para_spans.iter().rev().find(|p| p.0 <= s0).and_then(|p| p.2);
                chars.push(x.style.style_sheet.unwrap_or(0));
                c_over |= !styles.char_overrides(&x.style, pref, FAMILY).is_empty();
            }
        }
    }
    let one = |v: &Vec<u32>| (!v.is_empty() && v.iter().all(|x| *x == v[0])).then(|| v[0]);
    Some(json!({ "character": one(&chars), "characterOverride": c_over, "paragraph": one(&paras), "paragraphOverride": p_over }))
}

// ---------- commands ----------

fn styles(s: &Session) -> Result<TextStyles> {
    Ok(s.active().ok_or(EngineError::NoDocument)?.doc.text_styles.clone())
}

fn char_json(d: &CharacterStyleDef, styles: &TextStyles) -> Value {
    json!({ "id": d.id, "name": d.name, "attrs": d.attrs, "resolved": styles.resolve_char(None, Some(d.id), FAMILY) })
}

fn para_json(d: &ParagraphStyleDef, styles: &TextStyles) -> Value {
    json!({ "id": d.id, "name": d.name, "paragraphAttrs": d.para_attrs, "characterAttrs": d.char_attrs, "resolved": { "paragraph": styles.resolve_para(Some(d.id)), "character": styles.resolve_char(Some(d.id), None, FAMILY) } })
}

fn list(s: &Session, p: &Value, paragraph: bool) -> Result<Value> {
    let st = styles(s)?;
    let cur = current(s, p);
    Ok(if paragraph {
        let mut v = vec![para_json(&st.basic, &st)];
        v.extend(st.paragraph.iter().map(|d| para_json(d, &st)));
        json!({ "styles": v, "current": cur })
    } else {
        let mut v = vec![json!({ "id": 0, "name": ts::NO_CHARACTER_STYLE, "attrs": {} })];
        v.extend(st.character.iter().map(|d| char_json(d, &st)));
        json!({ "styles": v, "current": cur })
    })
}

fn new_style(s: &mut Session, p: &Value, paragraph: bool) -> Result<Value> {
    let cmd = if paragraph { "type.paragraphStyle.new" } else { "type.characterStyle.new" };
    let mut st = styles(s)?;
    let (mut ca, mut pa) = parse_attrs(cmd, p.get("attrs").unwrap_or(&Value::Null))?;
    // With a type layer targeted, the style starts from the text's formatting (as Photoshop).
    let from_text = p.get("fromSelection").and_then(Value::as_bool).unwrap_or(true);
    if from_text && let Some((run, para, _, _)) = sample(s, p) {
        if paragraph {
            let base_c = st.resolve_char(None, None, FAMILY);
            let mut c = ts::diff_attrs(&CharStyle { style_sheet: None, ..run }, &base_c);
            c.extend(ca);
            ca = c;
            let mut pp = ts::diff_attrs(&para, &st.resolve_para(None));
            pp.extend(pa);
            pa = pp;
        } else {
            let base_c = st.resolve_char(para.style_sheet, None, FAMILY);
            let mut c = ts::diff_attrs(&run, &base_c);
            c.extend(ca);
            ca = c;
        }
    }
    let name = p
        .get("name")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| st.unique_name(paragraph, if paragraph { "Paragraph Style" } else { "Character Style" }));
    let id = if paragraph {
        let id = st.next_para_id().ok_or_else(|| exhausted_id(true))?;
        st.paragraph.push(ParagraphStyleDef { id, name: name.clone(), para_attrs: pa, char_attrs: ca });
        id
    } else {
        let id = st.next_char_id().ok_or_else(|| exhausted_id(false))?;
        st.character.push(CharacterStyleDef { id, name: name.clone(), attrs: ca });
        id
    };
    let apply = p.get("apply").and_then(Value::as_bool).unwrap_or(false);
    let (target, op) = if apply {
        let op = if paragraph { Op { set_para: Some(Some(id)), ..Op::default() } } else { Op { set_char: Some(Some(id)), ..Op::default() } };
        (target_ranges(s, p)?, op)
    } else {
        (Vec::new(), Op::default())
    };
    commit(s, if paragraph { "New Paragraph Style" } else { "New Character Style" }, st, &target, op)?;
    Ok(json!({ "id": id, "name": name }))
}

fn lookup_id(cmd: &str, st: &TextStyles, p: &Value, paragraph: bool, allow_default: bool) -> Result<u32> {
    let id = id_param(cmd, p)?.ok_or_else(|| bad(cmd, "missing `id`"))?;
    let ok = if paragraph { st.para_style(id).is_some() } else { st.char_style(id).is_some() };
    if !(ok || (allow_default && id == 0)) {
        return Err(bad(cmd, format!("no style with id {id}")));
    }
    if id == 0 && !allow_default {
        return Err(bad(cmd, if paragraph { "Basic Paragraph can't be changed this way" } else { "\"None\" is not a style" }));
    }
    Ok(id)
}

fn duplicate(s: &mut Session, p: &Value, paragraph: bool) -> Result<Value> {
    let cmd = if paragraph { "type.paragraphStyle.duplicate" } else { "type.characterStyle.duplicate" };
    let mut st = styles(s)?;
    let id = lookup_id(cmd, &st, p, paragraph, paragraph)?;
    let (new_id, name) = if paragraph {
        let src = st.para_style(id).cloned().unwrap_or_default();
        let name = format!("{} copy", src.name);
        let nid = st.next_para_id().ok_or_else(|| exhausted_id(true))?;
        let pos = st.paragraph.iter().position(|d| d.id == id).map_or(0, |i| i + 1);
        st.paragraph.insert(pos, ParagraphStyleDef { id: nid, name: name.clone(), ..src });
        (nid, name)
    } else {
        let src = st.char_style(id).cloned().unwrap_or_default();
        let name = format!("{} copy", src.name);
        let nid = st.next_char_id().ok_or_else(|| exhausted_id(false))?;
        let pos = st.character.iter().position(|d| d.id == id).map_or(0, |i| i + 1);
        st.character.insert(pos, CharacterStyleDef { id: nid, name: name.clone(), ..src });
        (nid, name)
    };
    commit(s, "Duplicate Style", st, &[], Op::default())?;
    Ok(json!({ "id": new_id, "name": name }))
}

fn delete(s: &mut Session, p: &Value, paragraph: bool) -> Result<Value> {
    let cmd = if paragraph { "type.paragraphStyle.delete" } else { "type.characterStyle.delete" };
    let st = styles(s)?;
    let id = lookup_id(cmd, &st, p, paragraph, false)?;
    // Text keeps its look: references are dropped and the attributes become overrides.
    let mut n = 0;
    s.edit("Delete Style", |doc, _| {
        if paragraph {
            doc.text_styles.paragraph.retain(|d| d.id != id);
        } else {
            doc.text_styles.character.retain(|d| d.id != id);
        }
        let snapshot = doc.clone();
        let ids: Vec<LayerId> = doc.walk().into_iter().map(|(_, _, l)| l.id).collect();
        for lid in ids {
            let Some(LayerContent::Text(t)) = doc.layer_mut(lid).map(|l| &mut l.content) else { continue };
            let mut hit = false;
            if paragraph {
                for r in &mut t.paragraphs {
                    if r.style.style_sheet == Some(id) {
                        r.style.style_sheet = None;
                        hit = true;
                    }
                }
            } else {
                for r in &mut t.runs {
                    if r.style.style_sheet == Some(id) {
                        r.style.style_sheet = None;
                        hit = true;
                    }
                }
            }
            if hit {
                refresh(&snapshot, t);
                n += 1;
            }
        }
        Ok(())
    })?;
    Ok(json!({ "deleted": id, "layers": n }))
}

fn rename(s: &mut Session, p: &Value, paragraph: bool) -> Result<Value> {
    let cmd = if paragraph { "type.paragraphStyle.rename" } else { "type.characterStyle.rename" };
    let mut st = styles(s)?;
    let id = lookup_id(cmd, &st, p, paragraph, false)?;
    let name = p.get("name").and_then(Value::as_str).map(str::trim).filter(|n| !n.is_empty()).ok_or_else(|| bad(cmd, "missing `name`"))?.to_string();
    if paragraph {
        if let Some(d) = st.para_style_mut(id) {
            d.name = name.clone();
        }
    } else if let Some(d) = st.char_style_mut(id) {
        d.name = name.clone();
    }
    commit(s, "Rename Style", st, &[], Op::default())?;
    Ok(json!({ "id": id, "name": name }))
}

fn prepare_options(mut st: TextStyles, p: &Value, paragraph: bool) -> Result<(TextStyles, u32)> {
    let cmd = if paragraph { "type.paragraphStyle.set" } else { "type.characterStyle.set" };
    let id = lookup_id(cmd, &st, p, paragraph, paragraph)?;
    let (ca, pa) = parse_attrs(cmd, p.get("attrs").unwrap_or(&Value::Null))?;
    let replace = p.get("replace").and_then(Value::as_bool).unwrap_or(false);
    let clear: Vec<String> =
        p.get("clear").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default();
    let name = p.get("name").and_then(Value::as_str).map(str::trim).filter(|n| !n.is_empty()).map(str::to_string);
    let merge = |dst: &mut StyleAttrs, src: StyleAttrs| {
        if replace {
            dst.clear();
        }
        dst.extend(src);
        for k in &clear {
            dst.remove(k);
        }
    };
    if paragraph {
        let d = st.para_style_mut(id).ok_or_else(|| bad(cmd, "no such style"))?;
        merge(&mut d.char_attrs, ca);
        if replace {
            d.para_attrs.clear();
        }
        d.para_attrs.extend(pa);
        for k in &clear {
            d.para_attrs.remove(k);
        }
        if let Some(n) = name.filter(|_| id != 0) {
            d.name = n;
        }
    } else {
        if !pa.is_empty() {
            return Err(bad(cmd, "character styles have no paragraph attributes"));
        }
        let d = st.char_style_mut(id).ok_or_else(|| bad(cmd, "no such style"))?;
        merge(&mut d.attrs, ca);
        if let Some(n) = name {
            d.name = n;
        }
    }
    Ok((st, id))
}

/// Apply Style Options to an isolated document, using exactly the command's resolution and
/// rendering logic. The source document and the caller's session/history remain untouched.
pub fn preview_options(doc: &Document, p: &Value, paragraph: bool) -> Result<Document> {
    let (new, _) = prepare_options(doc.text_styles.clone(), p, paragraph)?;
    let mut shown = doc.clone();
    apply_styles(&mut shown, &new, &[], Op::default());
    Ok(shown)
}

fn set_options(s: &mut Session, p: &Value, paragraph: bool) -> Result<Value> {
    let (st, id) = prepare_options(styles(s)?, p, paragraph)?;
    let layers = commit(s, "Style Options", st, &[], Op::default())?;
    let mut v = list(s, &json!({}), paragraph)?;
    v["layers"] = json!(layers);
    v["id"] = json!(id);
    Ok(v)
}

fn apply(s: &mut Session, p: &Value, paragraph: bool) -> Result<Value> {
    let cmd = if paragraph { "type.paragraphStyle.apply" } else { "type.characterStyle.apply" };
    let st = styles(s)?;
    let id = match p.get("id") {
        None | Some(Value::Null) => 0,
        Some(_) => lookup_id(cmd, &st, p, paragraph, true)?,
    };
    let target = target_ranges(s, p)?;
    let clear = p.get("clearOverrides").and_then(Value::as_bool).unwrap_or(false);
    let r = Some(id).filter(|i| *i != 0);
    let op = if paragraph {
        Op { set_para: Some(r), clear_para: clear, clear_char: clear, ..Op::default() }
    } else {
        Op { set_char: Some(r), clear_char: clear, ..Op::default() }
    };
    let n = commit(s, if paragraph { "Apply Paragraph Style" } else { "Apply Character Style" }, st, &target, op)?;
    Ok(json!({ "id": id, "layers": n, "current": current(s, p) }))
}

fn clear_override(s: &mut Session, p: &Value, paragraph: bool) -> Result<Value> {
    let st = styles(s)?;
    let target = target_ranges(s, p)?;
    let op = if paragraph { Op { clear_para: true, clear_char: true, ..Op::default() } } else { Op { clear_char: true, ..Op::default() } };
    let n = commit(s, "Clear Override", st, &target, op)?;
    Ok(json!({ "layers": n, "current": current(s, p) }))
}

fn redefine(s: &mut Session, p: &Value, paragraph: bool) -> Result<Value> {
    let cmd = if paragraph { "type.paragraphStyle.redefine" } else { "type.characterStyle.redefine" };
    let mut st = styles(s)?;
    let (run, para, layer, span) = sample(s, p).ok_or_else(|| bad(cmd, "select type to redefine the style from"))?;
    let id = match id_param(cmd, p)? {
        Some(_) => lookup_id(cmd, &st, p, paragraph, paragraph)?,
        None if paragraph => para.style_sheet.unwrap_or(0),
        None => run.style_sheet.ok_or_else(|| bad(cmd, "the selected text has no character style"))?,
    };
    if paragraph {
        let char_keys: HashSet<String> = run.style_sheet.and_then(|c| st.char_style(c)).map(|d| d.attrs.keys().cloned().collect()).unwrap_or_default();
        let c_over: StyleAttrs = st.char_overrides(&run, Some(id).filter(|i| *i != 0), FAMILY).into_iter().filter(|(k, _)| !char_keys.contains(k)).collect();
        let base_para = st.resolve_para(Some(id));
        let p_over = ts::diff_attrs(&para, &base_para);
        let d = st.para_style_mut(id).ok_or_else(|| bad(cmd, "no such style"))?;
        d.char_attrs.extend(c_over);
        d.para_attrs.extend(p_over);
    } else {
        let over = st.char_overrides(&CharStyle { style_sheet: Some(id), ..run }, para.style_sheet, FAMILY);
        let d = st.char_style_mut(id).ok_or_else(|| bad(cmd, "no such style"))?;
        d.attrs.extend(over);
    }
    // The sampled run gets the style (its overrides are now part of it); others follow.
    let target = [(layer, Some(span))];
    let op = if paragraph { Op { set_para: Some(Some(id).filter(|i| *i != 0)), ..Op::default() } } else { Op { set_char: Some(Some(id)), ..Op::default() } };
    let n = commit(s, if paragraph { "Redefine Paragraph Style" } else { "Redefine Character Style" }, st, &target, op)?;
    Ok(json!({ "id": id, "layers": n, "current": current(s, p) }))
}

macro_rules! target_doc {
    () => {
        r#""layer":id? | "layers":[id]? (default: the selected type layers),"range":[startChar,endChar]? (one layer; default all text)"#
    };
}
macro_rules! attrs_doc {
    () => {
        r##""attrs":{model fields as in type.info ("size_pt":36,"font_family":"Inter","align":"Center",…) or Character/Paragraph panel keys as in type.setStyle ("size":36,"font":"Inter","color":"#rrggbb","align":"center",…)}"##
    };
}

macro_rules! style_specs {
    ($paragraph:expr, $prefix:literal, $noun:literal) => {
        vec![
            CommandSpec {
                id: concat!($prefix, ".new"),
                label: concat!("New ", $noun, " Style"),
                menu: &[],
                shortcut: None,
                params: concat!(r#"{"name":str?,"#, attrs_doc!(), r#"?,"fromSelection":bool=true (start from the targeted text's formatting),"apply":bool=false,"#, target_doc!(), r#"} → {"id","name"}"#),
                enabled: has_doc,
                journal: true,
                run: |s, p| new_style(s, p, $paragraph),
            },
            CommandSpec { id: concat!($prefix, ".duplicate"), label: "Duplicate Style", menu: &[], shortcut: None, params: r#"{"id":u32}"#, enabled: has_doc, journal: true, run: |s, p| duplicate(s, p, $paragraph) },
            CommandSpec { id: concat!($prefix, ".delete"), label: "Delete Style", menu: &[], shortcut: None, params: r#"{"id":u32} (text keeps its formatting as overrides)"#, enabled: has_doc, journal: true, run: |s, p| delete(s, p, $paragraph) },
            CommandSpec { id: concat!($prefix, ".rename"), label: "Rename Style", menu: &[], shortcut: None, params: r#"{"id":u32,"name":str}"#, enabled: has_doc, journal: true, run: |s, p| rename(s, p, $paragraph) },
            CommandSpec {
                id: concat!($prefix, ".set"),
                label: concat!($noun, " Style Options"),
                menu: &[],
                shortcut: None,
                params: concat!(r#"{"id":u32 (paragraph: 0 = Basic Paragraph),"name":str?,"#, attrs_doc!(), r#","replace":bool=false (drop attributes not given),"clear":[field]?} (text using the style updates, overrides kept)"#),
                enabled: has_doc,
                journal: true,
                run: |s, p| set_options(s, p, $paragraph),
            },
            CommandSpec {
                id: concat!($prefix, ".apply"),
                label: concat!("Apply ", $noun, " Style"),
                menu: &[],
                shortcut: None,
                params: concat!(r#"{"id":u32|0|null (0/null: None / Basic Paragraph),"clearOverrides":bool=false (Alt-click),"#, target_doc!(), "}"),
                enabled: has_target,
                journal: true,
                run: |s, p| apply(s, p, $paragraph),
            },
            CommandSpec {
                id: concat!($prefix, ".redefine"),
                label: concat!("Redefine ", $noun, " Style"),
                menu: &[],
                shortcut: None,
                params: concat!(r#"{"id":u32? (default: the targeted text's style),"#, target_doc!(), "} (the style takes the formatting of the start of the targeted text)"),
                enabled: has_target,
                journal: true,
                run: |s, p| redefine(s, p, $paragraph),
            },
            CommandSpec { id: concat!($prefix, ".clearOverride"), label: "Clear Override", menu: &[], shortcut: None, params: concat!("{", target_doc!(), "}"), enabled: has_target, journal: true, run: |s, p| clear_override(s, p, $paragraph) },
            CommandSpec {
                id: concat!($prefix, ".list"),
                label: concat!("List ", $noun, " Styles"),
                menu: &[],
                shortcut: None,
                params: concat!("{", target_doc!(), r#"} → {"styles":[{"id","name",attributes,"resolved"}],"current":{"character":id|null (mixed),"characterOverride":bool,"paragraph":id|null,"paragraphOverride":bool}}"#),
                enabled: has_doc,
                journal: false,
                run: |s, p| list(s, p, $paragraph),
            },
        ]
    };
}

pub fn specs() -> Vec<CommandSpec> {
    let mut v = style_specs!(false, "type.characterStyle", "Character");
    v.extend(style_specs!(true, "type.paragraphStyle", "Paragraph"));
    v
}

#[cfg(test)]
mod tests;
