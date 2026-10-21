//! Character and paragraph styles ⇄ the PSD text engine's style sheets (see
//! `photocraft_text::psd_styles`). Import merges the named sheets of every type layer into the
//! document's [`TextStyles`] by name and sets the runs' style references; export rewrites each
//! type layer's sheets from the document's styles. Files without named styles are left as they
//! are (byte-exact).

use std::collections::HashMap;

use photocraft_doc::text_styles::{CharacterStyleDef, ParagraphStyleDef, diff_attrs};
use photocraft_doc::{Document, LayerContent, LayerId, TextLayer, TextStyles};

/// Builds `doc.text_styles` from the type layers' engine data and links the runs.
pub fn import(doc: &mut Document) -> Vec<String> {
    let mut warnings = Vec::new();
    let dpi = doc.resolution_dpi;
    let mut styles = std::mem::take(&mut doc.text_styles);
    let mut char_ids: HashMap<String, u32> = styles.character.iter().map(|d| (d.name.clone(), d.id)).collect();
    let mut para_ids: HashMap<String, u32> = styles.paragraph.iter().map(|d| (d.name.clone(), d.id)).collect();
    let ids: Vec<LayerId> = doc.walk().into_iter().filter(|(_, _, l)| matches!(l.content, LayerContent::Text(_))).map(|(_, _, l)| l.id).collect();
    for id in ids {
        let Some(LayerContent::Text(t)) = doc.layer_mut(id).map(|l| &mut l.content) else { continue };
        let Some(raw) = t.psd_raw.clone() else { continue };
        if !photocraft_text::psd_styles::has_named_sheets(&raw) {
            continue;
        }
        let Some(sh) = photocraft_text::psd_styles::read_style_sheets(&raw, dpi) else { continue };
        let mut cmap: HashMap<usize, u32> = HashMap::new();
        for (ix, name, st) in &sh.char_sheets {
            let id = if let Some(id) = char_ids.get(name).copied() {
                id
            } else {
                let Some(id) = styles.next_char_id() else {
                    warnings.push(format!("character style {:?} was not imported: style id space exhausted", name));
                    continue;
                };
                let mut attrs = diff_attrs(st, &sh.normal_char);
                attrs.remove("postscript_name");
                styles.character.push(CharacterStyleDef { id, name: name.clone(), attrs });
                char_ids.insert(name.clone(), id);
                id
            };
            cmap.insert(*ix, id);
        }
        let mut pmap: HashMap<usize, u32> = HashMap::new();
        for (ix, name, p, cs) in &sh.para_sheets {
            let id = if let Some(id) = para_ids.get(name).copied() {
                id
            } else {
                let Some(id) = styles.next_para_id() else {
                    warnings.push(format!("paragraph style {:?} was not imported: style id space exhausted", name));
                    continue;
                };
                let mut char_attrs = cs.as_ref().map(|c| diff_attrs(c, &sh.normal_char)).unwrap_or_default();
                char_attrs.remove("postscript_name");
                styles.paragraph.push(ParagraphStyleDef { id, name: name.clone(), para_attrs: diff_attrs(p, &sh.normal_para), char_attrs });
                para_ids.insert(name.clone(), id);
                id
            };
            pmap.insert(*ix, id);
        }
        link(t, &sh.run_char, &sh.run_char_lens, &cmap, &sh.run_para, &pmap);
    }
    doc.text_styles = styles;
    warnings
}

fn link(
    t: &mut TextLayer,
    run_char: &[Option<usize>],
    run_char_lens: &[usize],
    cmap: &HashMap<usize, u32>,
    run_para: &[Option<usize>],
    pmap: &HashMap<usize, u32>,
) {
    if run_char_lens.len() == run_char.len() && !run_char.is_empty() {
        // Each model run takes the sheet of the `StyleRun` entry its first character is in.
        let (mut off, mut i, mut end) = (0usize, 0usize, run_char_lens.first().copied().unwrap_or(0));
        for r in t.runs.iter_mut() {
            while off >= end && i + 1 < run_char_lens.len() {
                i += 1;
                end = end.saturating_add(run_char_lens.get(i).copied().unwrap_or(0));
            }
            r.style.style_sheet = run_char.get(i).copied().flatten().and_then(|i| cmap.get(&i).copied());
            off = off.saturating_add(r.len);
        }
    } else if t.runs.len() == run_char.len() {
        for (r, ix) in t.runs.iter_mut().zip(run_char) {
            r.style.style_sheet = ix.and_then(|i| cmap.get(&i).copied());
        }
    }
    if t.paragraphs.len() == run_para.len() {
        for (r, ix) in t.paragraphs.iter_mut().zip(run_para) {
            r.style.style_sheet = ix.and_then(|i| pmap.get(&i).copied());
        }
    }
}

/// The `TySh` data to write for a type layer, with its style sheets regenerated from
/// `styles`; None when neither the document nor the data has named styles (write as is).
pub fn export_tysh(data: &[u8], t: &TextLayer, styles: &TextStyles, dpi: f32) -> Option<Vec<u8>> {
    let named = !styles.character.is_empty() || !styles.paragraph.is_empty();
    if !named && !photocraft_text::psd_styles::has_named_sheets(data) {
        return None;
    }
    photocraft_text::psd_styles::write_style_sheets(data, t, styles, dpi)
}
