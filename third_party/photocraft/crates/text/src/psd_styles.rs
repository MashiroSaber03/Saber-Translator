//! Named character and paragraph styles in the PSD text engine data.
//!
//! Every type layer's `EngineData` carries the document's style sheets in its `ResourceDict`
//! (and `DocumentResources`): `StyleSheetSet` (character style sheets: `Name`,
//! `StyleSheetData`) and `ParagraphSheetSet` (`Name`, `DefaultStyleSheet`, `Properties`), with
//! `TheNormalStyleSheet` / `TheNormalParagraphSheet` naming the defaults. Runs refer to a sheet
//! from their `StyleSheet` / `ParagraphSheet` dictionaries.
//!
//! How Photoshop links a run to a named sheet isn't in Adobe's published PSD spec. We read both
//! a `Parent` index and a `Name` on the run's sheet, and write `Parent` (plus the run's fully
//! resolved attributes, so the text looks the same whatever a reader makes of the reference).
//! Named sheets are written with fully resolved attributes too. Unverified against Photoshop.

use photocraft_doc::text::{CharStyle, ParagraphStyle};
use photocraft_doc::{TextLayer, TextStyles};

use crate::engine_data::Value as E;
use crate::psd::{char_style, engine_data, para_style, paragraph_properties, parse_tysh, postscript_for, style_sheet_data, write_tysh};

/// Style sheets read from one type layer.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PsdStyleSheets {
    /// The default ("Normal") character sheet, resolved.
    pub normal_char: CharStyle,
    pub normal_para: ParagraphStyle,
    /// Named character sheets other than the normal one: (set index, name, resolved style).
    pub char_sheets: Vec<(usize, String, CharStyle)>,
    /// Named paragraph sheets: (set index, name, paragraph style, character style of its
    /// `DefaultStyleSheet` when that isn't the normal sheet).
    pub para_sheets: Vec<(usize, String, ParagraphStyle, Option<CharStyle>)>,
    /// Per `StyleRun` entry: the character sheet index it refers to, if any.
    pub run_char: Vec<Option<usize>>,
    /// Per `StyleRun` entry: its length in UTF-8 bytes of the layer text (entries don't map
    /// one to one to model runs: kerning moves pair data between characters).
    pub run_char_lens: Vec<usize>,
    /// Per `ParagraphRun` entry: the paragraph sheet index it refers to, if any.
    pub run_para: Vec<Option<usize>>,
}

fn k_of(dpi: f32) -> f32 {
    72.0 / if dpi.is_finite() && dpi > 0.0 { dpi } else { 72.0 }
}

/// A sheet index from EngineData. A negative or overflowing index is not a slot (#4429).
fn sheet_index(v: i64) -> Option<usize> {
    usize::try_from(v).ok()
}

fn names(set: Option<&E>) -> Vec<String> {
    set.and_then(E::as_array).map(|a| a.iter().map(|s| s.get("Name").and_then(E::as_str).unwrap_or("").to_string()).collect()).unwrap_or_default()
}

/// Index a run's sheet dictionary refers to (`Parent` index, else `Name`), skipping `normal`.
fn reference(sheet: Option<&E>, set_names: &[String], normal: usize) -> Option<usize> {
    let s = sheet?;
    let i = s.get("Parent").and_then(E::as_i64).and_then(sheet_index).or_else(|| {
        let n = s.get("Name").and_then(E::as_str)?;
        set_names.iter().position(|x| x == n)
    })?;
    (i != normal && i < set_names.len()).then_some(i)
}

/// Adds to `sheet[key]` (a dictionary) every entry of `resolved` it lacks; existing entries win.
fn fill_missing(sheet: &mut E, key: &str, resolved: E) {
    let E::Dict(all) = resolved else { return };
    if sheet.get(key).is_none() {
        sheet.set(key, E::dict());
    }
    let Some(E::Dict(have)) = sheet.get_mut(key) else { return };
    for (k, v) in all {
        if !have.iter().any(|(h, _)| *h == k) {
            have.push((k, v));
        }
    }
}

/// Reads the style sheets of a `TySh` block. None when it has no engine data.
pub fn read_style_sheets(tysh: &[u8], dpi: f32) -> Option<PsdStyleSheets> {
    let t = parse_tysh(tysh)?;
    let e = engine_data(&t.text)?;
    let k = k_of(dpi);
    let res = e.get("ResourceDict").or_else(|| e.get("DocumentResources"))?;
    let fonts: Vec<String> = names(res.get("FontSet"));
    let sset = res.get("StyleSheetSet").and_then(E::as_array).unwrap_or(&[]);
    let pset = res.get("ParagraphSheetSet").and_then(E::as_array).unwrap_or(&[]);
    let normal_c = res.get("TheNormalStyleSheet").and_then(E::as_i64).and_then(sheet_index).unwrap_or(0);
    let normal_p = res.get("TheNormalParagraphSheet").and_then(E::as_i64).and_then(sheet_index).unwrap_or(0);
    let empty = E::dict();
    let base_data = sset.get(normal_c).and_then(|s| s.get("StyleSheetData"));
    let normal_char = char_style(None, base_data.unwrap_or(&empty), &fonts, k);
    let base_props = pset.get(normal_p).and_then(|s| s.get("Properties"));
    let normal_para = para_style(None, base_props.unwrap_or(&empty), k);
    let mut out = PsdStyleSheets { normal_char, normal_para, ..Default::default() };
    for (i, s) in sset.iter().enumerate() {
        if i == normal_c {
            continue;
        }
        let name = s.get("Name").and_then(E::as_str).unwrap_or("").to_string();
        let data = s.get("StyleSheetData").unwrap_or(&empty);
        out.char_sheets.push((i, name, char_style(base_data, data, &fonts, k)));
    }
    for (i, s) in pset.iter().enumerate() {
        if i == normal_p {
            continue;
        }
        let name = s.get("Name").and_then(E::as_str).unwrap_or("").to_string();
        let props = s.get("Properties").unwrap_or(&empty);
        let dss = s.get("DefaultStyleSheet").and_then(E::as_i64).and_then(sheet_index).filter(|i| *i != normal_c);
        let cs = dss.and_then(|d| sset.get(d)).map(|d| char_style(base_data, d.get("StyleSheetData").unwrap_or(&empty), &fonts, k));
        out.para_sheets.push((i, name, para_style(base_props, props, k), cs));
    }
    let snames = names(res.get("StyleSheetSet"));
    let pnames = names(res.get("ParagraphSheetSet"));
    if let Some(a) = e.path(&["EngineDict", "StyleRun", "RunArray"]).and_then(E::as_array) {
        out.run_char = a.iter().map(|r| reference(r.get("StyleSheet"), &snames, normal_c)).collect();
        let ed_text = e.path(&["EngineDict", "Editor", "Text"]).and_then(E::as_str).unwrap_or("");
        out.run_char_lens = crate::psd::utf16_to_byte_lengths(ed_text, &crate::psd::arr_f(e.path(&["EngineDict", "StyleRun", "RunLengthArray"])));
    }
    if let Some(a) = e.path(&["EngineDict", "ParagraphRun", "RunArray"]).and_then(E::as_array) {
        out.run_para = a.iter().map(|r| reference(r.get("ParagraphSheet"), &pnames, normal_p)).collect();
    }
    Some(out)
}

fn remove_key(d: &mut E, key: &str) {
    if let E::Dict(items) = d {
        items.retain(|(k, _)| k != key);
    }
}

/// Whether a `TySh` block carries named sheets beyond the normal ones.
pub fn has_named_sheets(tysh: &[u8]) -> bool {
    let Some(e) = parse_tysh(tysh).and_then(|t| engine_data(&t.text)) else {
        return false;
    };
    let Some(res) = e.get("ResourceDict") else {
        return false;
    };
    let n = |k: &str| res.get(k).and_then(E::as_array).map_or(0, <[E]>::len);
    n("StyleSheetSet") > 1 || n("ParagraphSheetSet") > 1
}

/// Rewrites the style sheets of a `TySh` block from the document's styles and the layer's run
/// references. The normal sheets are kept; other sheets are replaced by `styles`.
pub fn write_style_sheets(tysh: &[u8], layer: &TextLayer, styles: &TextStyles, dpi: f32) -> Option<Vec<u8>> {
    let mut t = parse_tysh(tysh)?;
    let mut e = engine_data(&t.text)?;
    let k = k_of(dpi);
    for key in ["ResourceDict", "DocumentResources"] {
        let Some(rd) = e.get_mut(key) else { continue };
        let mut fonts = names(rd.get("FontSet"));
        let normal_c = rd.get("TheNormalStyleSheet").and_then(E::as_i64).and_then(sheet_index).unwrap_or(0);
        let normal_p = rd.get("TheNormalParagraphSheet").and_then(E::as_i64).and_then(sheet_index).unwrap_or(0);
        let keep_c = rd.get("StyleSheetSet").and_then(E::as_array).and_then(|a| a.get(normal_c).or(a.first())).cloned();
        let keep_p = rd.get("ParagraphSheetSet").and_then(E::as_array).and_then(|a| a.get(normal_p).or(a.first())).cloned();
        let mut font_index = |s: &CharStyle| {
            let ps = postscript_for(s);
            match fonts.iter().position(|f| *f == ps) {
                Some(i) => i,
                None => {
                    fonts.push(ps);
                    fonts.len() - 1
                }
            }
        };
        let mut sset: Vec<E> = keep_c.into_iter().collect();
        for d in &styles.character {
            let st = styles.resolve_char(None, Some(d.id), crate::fonts::DEFAULT_FAMILY);
            let fi = font_index(&st);
            sset.push(E::Dict(vec![
                ("Name".into(), E::String(d.name.clone())),
                ("Parent".into(), E::Int(0)),
                ("StyleSheetData".into(), style_sheet_data(&st, fi, k)),
            ]));
        }
        let mut pset: Vec<E> = keep_p.into_iter().collect();
        for d in &styles.paragraph {
            let p = styles.resolve_para(Some(d.id));
            pset.push(E::Dict(vec![
                ("Name".into(), E::String(d.name.clone())),
                ("DefaultStyleSheet".into(), E::Int(0)),
                ("Parent".into(), E::Int(0)),
                ("Properties".into(), paragraph_properties(&p, k)),
            ]));
        }
        rd.set("StyleSheetSet", E::Array(sset));
        rd.set("ParagraphSheetSet", E::Array(pset));
        rd.set("TheNormalStyleSheet", E::Int(0));
        rd.set("TheNormalParagraphSheet", E::Int(0));
        rd.set("FontSet", E::Array(fonts.iter().map(|f| crate::psd::font_entry(f)).collect()));
    }
    let char_fonts = e.path(&["ResourceDict", "FontSet"]).or_else(|| e.path(&["DocumentResources", "FontSet"])).map(|f| names(Some(f))).unwrap_or_default();
    let char_ix = |id: Option<u32>| id.and_then(|id| styles.character.iter().position(|d| d.id == id)).map(|i| i as i64 + 1);
    let para_ix = |id: Option<u32>| id.and_then(|id| styles.paragraph.iter().position(|d| d.id == id)).map(|i| i as i64 + 1);
    let runs = layer.char_runs();
    // Each entry lies inside one model run: find it by the entry's start offset.
    let ed_text = e.path(&["EngineDict", "Editor", "Text"]).and_then(E::as_str).unwrap_or("").to_string();
    let entry_lens = crate::psd::utf16_to_byte_lengths(&ed_text, &crate::psd::arr_f(e.path(&["EngineDict", "StyleRun", "RunLengthArray"])));
    let run_at = |off: usize| {
        let mut end = 0usize;
        runs.iter()
            .find(|r| {
                end = end.saturating_add(r.len);
                off < end
            })
            .or(runs.last())
    };
    if let Some(E::Array(a)) = e.get_mut("EngineDict").and_then(|d| d.get_mut("StyleRun")).and_then(|r| r.get_mut("RunArray")) {
        let mut off = 0usize;
        for (i, r) in a.iter_mut().enumerate() {
            let start = off;
            off = off.saturating_add(entry_lens.get(i).copied().unwrap_or(0));
            if let Some(sheet) = r.get_mut("StyleSheet") {
                remove_key(sheet, "Parent");
                remove_key(sheet, "Name");
                let run = run_at(start);
                if let Some(ix) = char_ix(run.and_then(|r| r.style.style_sheet)) {
                    sheet.set("Parent", E::Int(ix));
                }
                // The run's reference may be dropped or renumbered: write what it inherited
                // too, so it keeps its look whatever sheet it now points to (#3795).
                if let Some(run) = run {
                    let fi = char_fonts.iter().position(|f| *f == postscript_for(&run.style)).unwrap_or(0);
                    fill_missing(sheet, "StyleSheetData", style_sheet_data(&run.style, fi, k));
                }
            }
        }
    }
    let paras = layer.paragraph_runs();
    if let Some(E::Array(a)) = e.get_mut("EngineDict").and_then(|d| d.get_mut("ParagraphRun")).and_then(|r| r.get_mut("RunArray")) {
        for (i, r) in a.iter_mut().enumerate() {
            if let Some(sheet) = r.get_mut("ParagraphSheet") {
                remove_key(sheet, "Parent");
                remove_key(sheet, "Name");
                if let Some(ix) = para_ix(paras.get(i).and_then(|p| p.style.style_sheet)) {
                    sheet.set("Parent", E::Int(ix));
                }
                if let Some(p) = paras.get(i) {
                    fill_missing(sheet, "Properties", paragraph_properties(&p.style, k));
                }
            }
        }
    }
    let raw = crate::engine_data::write(&e);
    {
        let item = t.text.items.iter_mut().find(|(k, _)| k.is("EngineData"))?;
        item.1 = photocraft_psd::descriptor::Value::RawData(raw)
    }
    Some(write_tysh(&t))
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_doc::text::{TextAlign, TextRun};
    use photocraft_doc::text_styles::{CharacterStyleDef, ParagraphStyleDef, diff_attrs};

    #[test]
    fn a_negative_sheet_index_is_not_a_slot_and_infinite_dpi_uses_72() {
        assert_eq!(sheet_index(-1), None);
        assert_eq!(sheet_index(2), Some(2));
        assert_eq!(k_of(f32::INFINITY), 1.0);
        assert_eq!(k_of(144.0), 0.5);
    }

    #[test]
    fn named_sheets_round_trip() {
        let mut styles = TextStyles::default();
        styles.character.push(CharacterStyleDef {
            id: 4,
            name: "Emphasis".into(),
            attrs: diff_attrs(&CharStyle { size_pt: 30.0, underline: true, ..Default::default() }, &CharStyle::default()),
        });
        styles.paragraph.push(ParagraphStyleDef {
            id: 2,
            name: "Centered".into(),
            para_attrs: diff_attrs(&ParagraphStyle { align: TextAlign::Center, ..Default::default() }, &ParagraphStyle::default()),
            ..Default::default()
        });
        let base = CharStyle { font_family: crate::fonts::DEFAULT_FAMILY.into(), ..Default::default() };
        let emph = styles.resolve_char(Some(2), Some(4), crate::fonts::DEFAULT_FAMILY);
        let mut layer = TextLayer {
            text: "plain bold".into(),
            runs: vec![TextRun { len: 6, style: base.clone() }, TextRun { len: 4, style: emph }],
            paragraphs: vec![photocraft_doc::text::ParagraphRun { len: 10, style: styles.resolve_para(Some(2)) }],
            ..Default::default()
        };
        layer.sync_summary();
        let tysh = crate::psd::build_tysh(&layer, 72.0, None);
        assert!(!has_named_sheets(&tysh));
        let out = write_style_sheets(&tysh, &layer, &styles, 72.0).unwrap();
        assert!(has_named_sheets(&out));
        let r = read_style_sheets(&out, 72.0).unwrap();
        assert_eq!(r.char_sheets.len(), 1);
        assert_eq!(r.char_sheets[0].1, "Emphasis");
        assert_eq!((r.char_sheets[0].2.size_pt, r.char_sheets[0].2.underline), (30.0, true));
        assert_eq!(r.para_sheets[0].1, "Centered");
        assert_eq!(r.para_sheets[0].2.align, TextAlign::Center);
        assert_eq!(r.run_char, vec![None, Some(1)]);
        assert_eq!(r.run_para, vec![Some(1)]);
        // The text itself still reads back the same.
        let back = crate::psd::text_layer_from_tysh(&out, 72.0).unwrap();
        assert_eq!(back.text, layer.text);
        assert_eq!(back.char_runs()[1].style.size_pt, 30.0);
        // No styles: the sets shrink back to the normal sheets.
        let plain = write_style_sheets(&out, &layer, &TextStyles::default(), 72.0).unwrap();
        assert!(!has_named_sheets(&plain));
        assert!(read_style_sheets(&plain, 72.0).unwrap().run_char.iter().all(Option::is_none));
    }

    #[test]
    fn malformed_input() {
        assert!(read_style_sheets(&[], 72.0).is_none());
        assert!(read_style_sheets(&[0u8; 60], 72.0).is_none());
        assert!(!has_named_sheets(b"junk"));
        assert!(write_style_sheets(&[1, 2, 3], &TextLayer::default(), &TextStyles::default(), 72.0).is_none());
    }
}
