//! Direct formatting applies only the requested properties to every selected Type layer.

use photocraft_doc::text::{AntiAlias, Caps, CharStyle, Kerning, Orientation, TextAlign};
use photocraft_doc::{LayerContent, LayerId, TextLayer};
use photocraft_engine::Session;
use serde_json::{Value, json};

fn text(s: &Session, id: LayerId) -> TextLayer {
    let LayerContent::Text(t) = &s.active().unwrap().doc.layer(id).unwrap().content else { panic!("not type") };
    t.clone()
}

fn select(s: &mut Session, ids: &[LayerId]) {
    for (i, id) in ids.iter().enumerate() {
        s.execute("layer.select", json!({"layer": id.0, "mode": if i == 0 { "replace" } else { "add" }})).unwrap();
    }
}

fn fixture() -> (Session, Vec<LayerId>, LayerId) {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 400, "height": 240, "background": "white"})).unwrap();
    let raster = s.active().unwrap().active_layer.unwrap();
    let mut ids = Vec::new();
    for (i, (size, weight, italic, kern)) in
        [(50, 400, false, 0), (30, 700, false, 0), (20, 400, true, -100), (10, 400, false, -50), (18, 600, false, 25)].into_iter().enumerate()
    {
        let id = s
            .execute(
                "type.create",
                json!({"text": "AéBC", "x": 20 + i * 55, "y": 80, "size": size, "font": "Inter", "weight": weight, "italic": italic, "kerning": kern}),
            )
            .unwrap()["layer"]
            .as_u64()
            .unwrap();
        s.execute("type.setStyle", json!({"layer": id, "range": [1, 3], "tracking": 30 + i, "underline": i % 2 == 0})).unwrap();
        ids.push(LayerId(id));
    }
    (s, ids, raster)
}

fn assert_styles(actual: &TextLayer, expected: &TextLayer) {
    assert_eq!(actual.runs, expected.runs);
    assert_eq!(actual.paragraphs, expected.paragraphs);
    assert_eq!(actual.text, expected.text);
    assert_eq!(actual.transform, expected.transform);
    assert_eq!(actual.shape, expected.shape);
}

#[test]
fn selected_type_properties_preserve_each_layers_other_attributes_and_one_history_step() {
    let (mut s, ids, raster) = fixture();
    type StyleCase<'a> = (&'a [usize], Value, fn(&mut CharStyle));
    let cases: [StyleCase<'_>; 4] = [
        (&[0, 2], json!({"font": "New Family"}), |st| {
            st.font_family = "New Family".into();
            st.postscript_name = None;
        }),
        (&[1, 3], json!({"kerning": 50}), |st| {
            st.kerning = Kerning::Off;
            st.kern = 50.0;
        }),
        (&[0, 1, 2, 3], json!({"font": "Another Family"}), |st| {
            st.font_family = "Another Family".into();
            st.postscript_name = None;
        }),
        (&[1, 2, 3], json!({"size": 35}), |st| st.size_pt = 35.0),
    ];
    for (indices, params, change) in cases {
        let mut selected: Vec<_> = indices.iter().map(|i| ids[*i]).collect();
        selected.push(raster); // The primary layer need not be Type.
        select(&mut s, &selected);
        let target = (s.active().unwrap().active_layer, s.active().unwrap().selected_layers());
        let before: Vec<_> = ids.iter().map(|id| text(&s, *id)).collect();
        let steps = s.active().unwrap().history.entries().len();
        s.execute("type.setStyle", params).unwrap();
        assert_eq!(s.active().unwrap().history.entries().len(), steps + 1);
        assert_eq!((s.active().unwrap().active_layer, s.active().unwrap().selected_layers()), target);
        for (i, id) in ids.iter().enumerate() {
            let mut expected = before[i].clone();
            if indices.contains(&i) {
                for run in &mut expected.runs {
                    change(&mut run.style);
                }
            }
            let actual = text(&s, *id);
            assert_styles(&actual, &expected);
            if indices.contains(&i) {
                assert_ne!(actual.psd_raw, before[i].psd_raw, "editable PSD data refreshed");
                let damage = s.active().unwrap().last_damage.unwrap();
                for t in [&before[i], &actual] {
                    let bounds = t.cache.as_ref().unwrap().tile_bounds();
                    assert_eq!(damage.intersect(&bounds), bounds);
                }
            }
        }
        assert!(s.undo());
        for (id, old) in ids.iter().zip(&before) {
            assert_styles(&text(&s, *id), old);
        }
        assert!(s.redo());
        assert_eq!((s.active().unwrap().active_layer, s.active().unwrap().selected_layers()), target);
    }
}

#[test]
fn explicit_type_layer_ranges_and_batch_paragraph_properties_keep_their_scope() {
    let (mut s, ids, raster) = fixture();
    select(&mut s, &ids[..4]);
    let before: Vec<_> = ids.iter().map(|id| text(&s, *id)).collect();
    s.execute("type.setStyle", json!({"layer": ids[0].0, "range": [1, 3], "fauxBold": true})).unwrap();
    let changed = text(&s, ids[0]);
    assert!(!changed.runs[0].style.faux_bold);
    assert!(changed.runs[1].style.faux_bold);
    assert!(!changed.runs[2].style.faux_bold);
    for i in 1..ids.len() {
        assert_styles(&text(&s, ids[i]), &before[i]);
    }
    assert!(s.undo());
    s.execute("type.setStyle", json!({"layers": [ids[0].0, ids[1].0, ids[0].0, raster.0], "caps": "all", "fauxItalic": true, "horizontalScale": 120, "align": "center", "firstLineIndent": 7, "hyphenate": false})).unwrap();
    for (i, id) in ids.iter().enumerate() {
        let mut expected = before[i].clone();
        if i < 2 {
            for run in &mut expected.runs {
                run.style.caps = Caps::AllCaps;
                run.style.faux_italic = true;
                run.style.horizontal_scale = 1.2;
            }
            for para in &mut expected.paragraphs {
                para.style.align = TextAlign::Center;
                para.style.first_line_indent_pt = 7.0;
                para.style.hyphenate = false;
            }
        }
        assert_styles(&text(&s, *id), &expected);
    }
}

#[test]
fn shown_type_metrics_convert_per_target_coalesce_and_fail_atomically() {
    let (mut s, ids, _) = fixture();
    s.execute("type.edit", json!({"layer": ids[0].0, "transform": [2, 0, 0, 2, 20, 80]})).unwrap();
    s.execute("type.edit", json!({"layer": ids[1].0, "transform": [0.5, 0, 0, 0.5, 75, 80]})).unwrap();
    select(&mut s, &ids[..2]);
    let before = [text(&s, ids[0]), text(&s, ids[1])];
    let steps = s.active().unwrap().history.entries().len();
    for size in [24, 30] {
        s.execute("type.setStyle", json!({"metricsAsShown": true, "size": size, "leading": 40, "coalesce": "batch-size"})).unwrap();
    }
    for (i, scale) in [2.0, 0.5].into_iter().enumerate() {
        let t = text(&s, ids[i]);
        assert!(t.runs.iter().all(|r| r.style.size_pt == 30.0 / scale && r.style.leading_pt == Some(40.0 / scale)));
        assert_eq!(t.transform, before[i].transform);
    }
    assert_eq!(s.active().unwrap().history.entries().len(), steps + 1);
    assert!(s.undo());
    for i in 0..2 {
        assert_styles(&text(&s, ids[i]), &before[i]);
    }
    assert!(s.redo());
    let unchanged = [text(&s, ids[0]), text(&s, ids[1])];
    let revision = s.active().unwrap().revision;
    let steps = s.active().unwrap().history.entries().len();
    // The second target would exceed the existing 1296pt limit after conversion.
    assert!(s.execute("type.setStyle", json!({"metricsAsShown": true, "size": 700})).is_err());
    assert_eq!(s.active().unwrap().revision, revision);
    assert_eq!(s.active().unwrap().history.entries().len(), steps);
    for i in 0..2 {
        assert_styles(&text(&s, ids[i]), &unchanged[i]);
    }
    s.execute("type.setStyle", json!({"size": 22, "leading": 26})).unwrap();
    for id in &ids[..2] {
        assert!(text(&s, *id).runs.iter().all(|r| r.style.size_pt == 22.0 && r.style.leading_pt == Some(26.0)));
    }
}

#[test]
fn direct_type_formatting_menu_routes_batch_without_enabling_structural_commands() {
    let (mut s, ids, raster) = fixture();
    s.execute("type.saveDefaultTypeStyles", json!({})).unwrap();
    let defaults = text(&s, ids[4]);
    select(&mut s, &[ids[0], ids[1], raster]);
    for (command, params) in [
        ("type.antiAlias.crisp", json!({})),
        ("type.orientation.vertical", json!({})),
        ("type.openType.fractions", json!({"on": true})),
        ("type.warpText", json!({"style": "arc", "bend": 15})),
        ("type.loadDefaultTypeStyles", json!({})),
    ] {
        assert!(s.is_enabled(command), "{command} supports selected Type peers");
        assert!(!s.is_enabled("type.convertToShape"));
        let before: Vec<_> = ids.iter().map(|id| text(&s, *id)).collect();
        let steps = s.active().unwrap().history.entries().len();
        s.execute(command, params).unwrap();
        assert_eq!(s.active().unwrap().history.entries().len(), steps + 1);
        for (i, id) in ids.iter().enumerate() {
            let actual = text(&s, *id);
            let mut expected = before[i].clone();
            if i < 2 {
                match command {
                    "type.antiAlias.crisp" => expected.antialias = AntiAlias::Crisp,
                    "type.orientation.vertical" => expected.orientation = Orientation::Vertical,
                    "type.openType.fractions" => {
                        for run in &mut expected.runs {
                            run.style.features.push(photocraft_doc::text::FontFeature { tag: "frac".into(), value: 1 });
                        }
                    }
                    "type.warpText" => {
                        assert_eq!(actual.warp.as_ref().unwrap().value, 15.0);
                        expected.warp = actual.warp.clone();
                    }
                    _ => {
                        expected.runs = vec![photocraft_doc::text::TextRun { len: expected.text.len(), style: defaults.runs[0].style.clone() }];
                        expected.paragraphs = defaults.paragraphs.clone();
                    }
                }
            }
            assert_styles(&actual, &expected);
            assert_eq!(actual.antialias, expected.antialias);
            assert_eq!(actual.orientation, expected.orientation);
            assert_eq!(actual.warp, expected.warp);
        }
        assert!(s.undo());
    }
}

#[test]
fn invalid_explicit_type_targets_leave_the_whole_batch_unchanged() {
    let (mut s, ids, _) = fixture();
    let before = text(&s, ids[0]);
    let revision = s.active().unwrap().revision;
    for params in [json!({"layers": [ids[0].0, u64::MAX], "size": 40}), json!({"layers": [ids[0].0, "bad"], "size": 40})] {
        assert!(s.execute("type.setStyle", params).is_err());
        assert_eq!(s.active().unwrap().revision, revision);
        assert_styles(&text(&s, ids[0]), &before);
    }
}
