use super::*;

fn session(text: &str) -> (Session, u64) {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 400, "height": 200, "background": "transparent"})).unwrap();
    let r = s.execute("type.create", json!({"x": 10, "y": 50, "text": text, "size": 12})).unwrap();
    (s, r["layer"].as_u64().unwrap())
}

fn layer(s: &Session, id: u64) -> TextLayer {
    text_of(&s.active().unwrap().doc, LayerId(id)).unwrap().clone()
}

fn run_at(t: &TextLayer, ci: usize) -> CharStyle {
    let b = byte_at(&t.text, ci);
    let mut at = 0;
    for r in t.char_runs() {
        if b < at + r.len {
            return r.style;
        }
        at += r.len;
    }
    t.char_runs().last().unwrap().style.clone()
}

#[test]
fn character_style_lifecycle() {
    let (mut s, id) = session("Hello world");
    let r = s.execute("type.characterStyle.new", json!({"attrs": {"size": 30, "underline": true}, "fromSelection": false})).unwrap();
    let cs = r["id"].as_u64().unwrap();
    assert_eq!(r["name"], "Character Style 1");
    let l = s.execute("type.characterStyle.list", json!({})).unwrap();
    assert_eq!(l["styles"][0]["name"], "None");
    assert_eq!(l["styles"][1]["attrs"], json!({"size_pt": 30.0, "underline": true}));

    // Apply to "world": its runs take the style; "Hello " keeps None.
    s.execute("type.characterStyle.apply", json!({"id": cs, "layer": id, "range": [6, 11]})).unwrap();
    let t = layer(&s, id);
    let w = run_at(&t, 7);
    assert_eq!((w.size_pt, w.underline, w.style_sheet), (30.0, true, Some(cs as u32)));
    assert_eq!(run_at(&t, 0).style_sheet, None);
    let cur = s.execute("type.characterStyle.list", json!({"layer": id, "range": [6, 11]})).unwrap()["current"].clone();
    assert_eq!((cur["character"].as_u64(), cur["characterOverride"].as_bool()), (Some(cs), Some(false)));

    // A local change is an override ("+"); redefining the style changes other text too.
    s.execute("type.setStyle", json!({"layer": id, "range": [6, 8], "italic": true})).unwrap();
    let cur = s.execute("type.characterStyle.list", json!({"layer": id, "range": [6, 8]})).unwrap()["current"].clone();
    assert_eq!(cur["characterOverride"], true);
    s.execute("type.characterStyle.set", json!({"id": cs, "attrs": {"size_pt": 40.0}})).unwrap();
    let t = layer(&s, id);
    assert_eq!((run_at(&t, 6).size_pt, run_at(&t, 6).italic), (40.0, true));
    assert_eq!((run_at(&t, 9).size_pt, run_at(&t, 9).italic), (40.0, false));
    assert_eq!(run_at(&t, 0).size_pt, 12.0);

    // Clear Override drops the italic but keeps the style.
    s.execute("type.characterStyle.clearOverride", json!({"layer": id, "range": [6, 11]})).unwrap();
    let t = layer(&s, id);
    assert!(!run_at(&t, 6).italic);
    assert_eq!(run_at(&t, 6).style_sheet, Some(cs as u32));

    // Rename, duplicate, delete (text keeps its look).
    s.execute("type.characterStyle.rename", json!({"id": cs, "name": "Emphasis"})).unwrap();
    let d = s.execute("type.characterStyle.duplicate", json!({"id": cs})).unwrap();
    assert_eq!(d["name"], "Emphasis copy");
    s.execute("type.characterStyle.delete", json!({"id": cs})).unwrap();
    let t = layer(&s, id);
    assert_eq!((run_at(&t, 7).size_pt, run_at(&t, 7).style_sheet), (40.0, None));
    let l = s.execute("type.characterStyle.list", json!({})).unwrap();
    assert_eq!(l["styles"].as_array().unwrap().len(), 2);

    // Each command is one undo step.
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(run_at(&layer(&s, id), 7).style_sheet, Some(cs as u32));
}

#[test]
fn new_style_from_selection_and_redefine() {
    let (mut s, id) = session("Title");
    s.execute("type.setStyle", json!({"layer": id, "size": 48, "color": "#ff0000"})).unwrap();
    let r = s.execute("type.characterStyle.new", json!({"layer": id, "apply": true})).unwrap();
    let cs = r["id"].as_u64().unwrap() as u32;
    let st = s.active().unwrap().doc.text_styles.char_style(cs).unwrap().clone();
    assert_eq!(st.attrs.get("size_pt"), Some(&json!(48.0)));
    assert_eq!(run_at(&layer(&s, id), 0).style_sheet, Some(cs));
    // Change the text, then redefine the style from it: no override remains.
    s.execute("type.setStyle", json!({"layer": id, "size": 60})).unwrap();
    s.execute("type.characterStyle.redefine", json!({"layer": id})).unwrap();
    let cur = s.execute("type.characterStyle.list", json!({"layer": id})).unwrap()["current"].clone();
    assert_eq!((cur["character"].as_u64(), cur["characterOverride"].as_bool()), (Some(u64::from(cs)), Some(false)));
    assert_eq!(s.active().unwrap().doc.text_styles.char_style(cs).unwrap().attrs.get("size_pt"), Some(&json!(60.0)));
}

#[test]
fn paragraph_styles_and_basic_paragraph() {
    let (mut s, id) = session("One\nTwo");
    // A fresh layer is Basic Paragraph with no paragraph override.
    let cur = s.execute("type.paragraphStyle.list", json!({"layer": id})).unwrap()["current"].clone();
    assert_eq!((cur["paragraph"].as_u64(), cur["paragraphOverride"].as_bool()), (Some(0), Some(false)));
    let r =
        s.execute("type.paragraphStyle.new", json!({"name": "Head", "fromSelection": false, "attrs": {"align": "center", "size": 24, "weight": 700}})).unwrap();
    let ps = r["id"].as_u64().unwrap() as u32;
    s.execute("type.paragraphStyle.apply", json!({"id": ps, "layer": id, "range": [0, 1]})).unwrap();
    let t = layer(&s, id);
    let p = t.paragraph_runs();
    assert_eq!(p[0].style.style_sheet, Some(ps));
    assert_eq!(p[0].style.align, photocraft_doc::text::TextAlign::Center);
    assert_eq!(p.last().unwrap().style.style_sheet, None);
    // The first paragraph's characters are on the style's basis.
    assert_eq!((run_at(&t, 0).size_pt, run_at(&t, 0).weight), (24.0, 700));
    assert_eq!(run_at(&t, 5).size_pt, 12.0);
    // Redefining Basic Paragraph moves the second paragraph only.
    s.execute("type.paragraphStyle.set", json!({"id": 0, "attrs": {"spaceBefore": 6}})).unwrap();
    let t = layer(&s, id);
    let p = t.paragraph_runs();
    assert_eq!(p.last().unwrap().style.space_before_pt, 6.0);
    assert_eq!(p[0].style.space_before_pt, 6.0, "Head inherits Basic Paragraph for unset attributes");
    // Basic Paragraph can't be deleted or renamed; bad ids and attributes are rejected.
    assert!(s.execute("type.paragraphStyle.delete", json!({"id": 0})).is_err());
    assert!(s.execute("type.paragraphStyle.rename", json!({"id": 0, "name": "x"})).is_err());
    assert!(s.execute("type.characterStyle.apply", json!({"id": 99})).is_err());
    assert!(s.execute("type.characterStyle.new", json!({"attrs": {"bogus": 1}})).is_err());
    assert!(s.execute("type.characterStyle.new", json!({"attrs": {"size_pt": "big"}})).is_err());
    // Delete keeps the look.
    s.execute("type.paragraphStyle.delete", json!({"id": ps})).unwrap();
    let t = layer(&s, id);
    assert_eq!(t.paragraph_runs()[0].style.align, photocraft_doc::text::TextAlign::Center);
    assert_eq!(t.paragraph_runs()[0].style.style_sheet, None);
}

#[test]
fn styles_survive_typing_and_save() {
    let (mut s, id) = session("ab");
    let r = s.execute("type.characterStyle.new", json!({"attrs": {"size": 20}, "fromSelection": false})).unwrap();
    let cs = r["id"].as_u64().unwrap() as u32;
    s.execute("type.characterStyle.apply", json!({"id": cs, "layer": id})).unwrap();
    // Typed text inherits the run (and its style reference).
    s.execute("type.edit", json!({"layer": id, "replace": {"start": 2, "end": 2, "text": "cd"}})).unwrap();
    let t = layer(&s, id);
    assert_eq!(run_at(&t, 3).style_sheet, Some(cs));
    // .pcraft round trip keeps the styles and references.
    let doc = s.active().unwrap().doc.clone();
    let bytes = photocraft_format::save_to_bytes(&doc, &photocraft_format::SaveOptions::default()).unwrap();
    let back = photocraft_format::load_from_bytes(&bytes).unwrap();
    assert_eq!(back.text_styles, doc.text_styles);
    let LayerContent::Text(t2) = &back.walk().into_iter().find(|(_, _, l)| matches!(l.content, LayerContent::Text(_))).unwrap().2.content else { panic!() };
    assert_eq!(run_at(t2, 0).style_sheet, Some(cs));
}

#[test]
fn disabled_states() {
    let mut s = Session::new();
    assert!(!s.is_enabled("type.characterStyle.new"));
    s.execute("file.new", json!({"width": 10, "height": 10})).unwrap();
    assert!(s.is_enabled("type.characterStyle.new"));
    assert!(!s.is_enabled("type.characterStyle.apply"));
    assert!(s.execute("type.characterStyle.list", json!({})).unwrap()["current"].is_null());
}

#[test]
fn shared_attribute_names_accept_both_spellings() {
    let (mut s, _) = session("x");
    let r = s.execute("type.characterStyle.new", json!({"fromSelection": false, "attrs": {"color": "#ff0000", "caps": "all", "kerning": "Off"}})).unwrap();
    let id = r["id"].as_u64().unwrap() as u32;
    let st = s.active().unwrap().doc.text_styles.resolve_char(None, Some(id), FAMILY);
    assert_eq!(st.color.to_rgb(), [1.0, 0.0, 0.0]);
    assert_eq!(st.caps, photocraft_doc::text::Caps::AllCaps);
    assert_eq!(st.kerning, photocraft_doc::text::Kerning::Off);
    let r = s.execute("type.paragraphStyle.new", json!({"fromSelection": false, "attrs": {"align": "Center", "direction": "rtl"}})).unwrap();
    let p = s.active().unwrap().doc.text_styles.resolve_para(Some(r["id"].as_u64().unwrap() as u32));
    assert_eq!((p.align, p.direction), (photocraft_doc::text::TextAlign::Center, photocraft_doc::text::TextDirection::Rtl));
}

#[test]
fn exhausted_style_ids_return_errors_without_history_changes() {
    let (mut s, _) = session("x");
    let mut doc = (*s.active().unwrap().doc).clone();
    doc.text_styles.character.push(CharacterStyleDef { id: u32::MAX, name: "Max".into(), ..Default::default() });
    doc.text_styles.paragraph.push(ParagraphStyleDef { id: u32::MAX, name: "Max".into(), ..Default::default() });
    s.active_mut().unwrap().doc = std::sync::Arc::new(doc);
    let past = s.active().unwrap().history.past_len();
    let revision = s.active().unwrap().revision;

    assert!(s.execute("type.characterStyle.new", json!({"fromSelection": false})).unwrap_err().to_string().contains("character style id space exhausted"));
    assert!(s.execute("type.characterStyle.duplicate", json!({"id": u32::MAX})).unwrap_err().to_string().contains("character style id space exhausted"));
    assert!(s.execute("type.paragraphStyle.new", json!({"fromSelection": false})).unwrap_err().to_string().contains("paragraph style id space exhausted"));
    assert!(s.execute("type.paragraphStyle.duplicate", json!({"id": u32::MAX})).unwrap_err().to_string().contains("paragraph style id space exhausted"));
    assert_eq!(s.active().unwrap().history.past_len(), past);
    assert_eq!(s.active().unwrap().revision, revision);
    assert_eq!(s.active().unwrap().doc.text_styles.next_char_id(), None);
    assert_eq!(s.active().unwrap().doc.text_styles.next_para_id(), None);
}

#[test]
fn isolated_options_preview_matches_commit_with_multiple_layers_and_local_overrides() {
    for paragraph in [false, true] {
        for depth in [8, 16, 32] {
            let mut s = Session::new();
            s.execute("file.new", json!({"width":400,"height":200,"depth":depth,"background":"transparent"})).unwrap();
            let a = s.execute("type.create", json!({"x":10,"y":50,"text":"Hello world","size":24})).unwrap()["layer"].as_u64().unwrap();
            let b = s.execute("type.create", json!({"x":10,"y":100,"text":"Second layer","size":24})).unwrap()["layer"].as_u64().unwrap();
            let prefix = if paragraph { "type.paragraphStyle" } else { "type.characterStyle" };
            let id = s.execute(&format!("{prefix}.new"), json!({"fromSelection":false,"attrs":{"color":"#ff0000"}})).unwrap()["id"].as_u64().unwrap();
            for layer in [a, b] {
                s.execute(&format!("{prefix}.apply"), json!({"id":id,"layer":layer})).unwrap();
            }
            s.execute("type.setStyle", json!({"layer":a,"range":[6,11],"color":"#0000ff"})).unwrap();
            let doc = s.active().unwrap().doc.clone();
            let revision = s.active().unwrap().revision;
            let p = json!({"id":id,"attrs":{"color":"#00ff00"}});
            let shown = preview_options(&doc, &p, paragraph).unwrap();
            assert!(std::sync::Arc::ptr_eq(&s.active().unwrap().doc, &doc));
            assert_eq!(s.active().unwrap().revision, revision);
            assert_eq!(text_of(&shown, LayerId(a)).unwrap().char_runs().last().unwrap().style.color.to_rgb(), [0.0, 0.0, 1.0]);
            assert_eq!(text_of(&shown, LayerId(b)).unwrap().color.to_rgb(), [0.0, 1.0, 0.0]);
            s.execute(&format!("{prefix}.set"), p).unwrap();
            let committed = &s.active().unwrap().doc;
            assert_eq!(shown.text_styles, committed.text_styles);
            for layer in [a, b] {
                let expected = text_of(&shown, LayerId(layer)).unwrap();
                let actual = text_of(committed, LayerId(layer)).unwrap();
                assert_eq!(expected.runs, actual.runs);
                assert_eq!(expected.paragraphs, actual.paragraphs);
                assert_eq!(expected.psd_raw, actual.psd_raw);
            }
            assert_eq!(photocraft_compose::flatten(&shown).get(20, 40), photocraft_compose::flatten(committed).get(20, 40));
        }
    }
}

#[test]
fn preview_options_validates_without_mutating_source() {
    let (s, _) = session("Text");
    let doc = s.active().unwrap().doc.clone();
    for paragraph in [false, true] {
        for p in [json!({}), json!({"id":"bad"}), json!({"id":u64::MAX}), json!({"id":999}), json!({"id":0,"attrs":3})] {
            assert!(preview_options(&doc, &p, paragraph).is_err());
        }
    }
    let before = doc.text_styles.clone();
    let p = json!({"id":0,"attrs":{"color":"#112233"}});
    assert!(preview_options(&doc, &p, false).is_err());
    let shown = preview_options(&doc, &p, true).unwrap();
    assert_ne!(shown.text_styles, before);
    assert_eq!(doc.text_styles, before);
}
