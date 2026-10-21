use super::*;
use photocraft_doc::{Layer, TextLayer};
use photocraft_geom::Rect;

/// A document with a raster layer "photo", a togglable raster "badge" and a text layer "title".
fn session() -> (Session, LayerId, LayerId, LayerId) {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
    let ids = s
        .edit("setup", |doc, _| {
            let fmt = doc.pixel_format();
            let mut photo = Layer::raster("photo", fmt);
            photo.surface_mut().unwrap().fill_rect(Rect::new(0, 0, 64, 48), &photocraft_raster::from_rgba(&fmt, [0.5, 0.5, 0.5, 1.0]));
            let mut badge = Layer::raster("badge", fmt);
            badge.surface_mut().unwrap().fill_rect(Rect::new(0, 0, 10, 10), &photocraft_raster::from_rgba(&fmt, [1.0, 0.0, 0.0, 1.0]));
            let title = Layer::new("title", LayerContent::Text(TextLayer { text: "Old".into(), ..Default::default() }));
            let (p, b, t) = (photo.id, badge.id, title.id);
            doc.layers.push(photo);
            doc.layers.push(badge);
            doc.layers.push(title);
            Ok((p, b, t))
        })
        .unwrap();
    (s, ids.0, ids.1, ids.2)
}

fn doc(s: &Session) -> &Document {
    &s.active().unwrap().doc
}

fn text_of(s: &Session, id: LayerId) -> String {
    match &doc(s).layer(id).unwrap().content {
        LayerContent::Text(t) => t.text.clone(),
        _ => panic!("not a text layer"),
    }
}

fn tmp(name: &str) -> String {
    let d = std::env::temp_dir().join(format!("pc-vars-{}-{name}", std::process::id()));
    let _ = std::fs::create_dir_all(&d);
    d.to_string_lossy().into_owned()
}

#[test]
fn define_apply_visibility_and_text() {
    let (mut s, _photo, badge, title) = session();
    s.execute(
        "image.variables.define",
        json!({"defs": [
            {"name": "showBadge", "layer": badge.0, "type": "visibility"},
            {"name": "headline", "layer": title.0, "type": "textReplacement"},
        ]}),
    )
    .unwrap();
    s.execute(
        "image.variables.dataSets",
        json!({"dataSets": [
            {"name": "A", "values": [
                {"variable": "showBadge", "kind": "visibility", "value": false},
                {"variable": "headline", "kind": "text", "value": "Hello"},
            ]},
        ]}),
    )
    .unwrap();
    assert!(doc(&s).layer(badge).unwrap().visible);
    assert_eq!(text_of(&s, title), "Old");

    // document.inspect surfaces the variables for agents.
    let insp = s.execute("document.inspect", json!({})).unwrap();
    assert_eq!(insp["variables"]["defs"].as_array().unwrap().len(), 2);
    assert_eq!(insp["variables"]["dataSets"][0], "A");

    let r = s.execute("image.applyDataSet", json!({"name": "A"})).unwrap();
    assert_eq!(r["applied"], "A");
    assert!(!doc(&s).layer(badge).unwrap().visible, "visibility applied");
    assert_eq!(text_of(&s, title), "Hello", "text applied");
    assert_eq!(doc(&s).variables.active, Some(0));

    // One history step, undoable back to the original.
    assert_eq!(s.active().unwrap().history.entries().last().map(|e| e.as_str()), Some("Apply Data Set \"A\""));
    assert!(s.undo());
    assert!(doc(&s).layer(badge).unwrap().visible);
    assert_eq!(text_of(&s, title), "Old");
}

#[test]
fn listed_pixel_methods_can_be_used_to_redefine_variables() {
    for method in ["fit", "fill", "asIs", "conform"] {
        let (mut s, photo, _badge, _title) = session();
        let defined = s
            .execute(
                "image.variables.define",
                json!({"defs": [{"name": "photo", "layer": photo.0, "type": "pixelReplacement", "method": method, "align": "bottomRight", "clip": true}]}),
            )
            .unwrap();
        let expected = doc(&s).variables.defs.clone();
        let listed = s.execute("variables.list", json!({})).unwrap();
        s.execute("image.variables.define", json!({"defs": listed["defs"]})).unwrap();
        assert_eq!(doc(&s).variables.defs, expected, "{method}");
        assert_eq!(defined["defs"][0]["method"], method);
        assert_eq!(listed["defs"][0]["method"], method);
    }
}

#[test]
fn bad_params_are_errors() {
    let (mut s, _p, badge, _t) = session();
    assert!(s.execute("image.variables.define", json!({"defs": [{"name": "x", "layer": u64::MAX, "type": "visibility"}]})).is_err());
    assert!(s.execute("image.variables.define", json!({"defs": [{"name": "x", "layer": badge.0, "type": "nope"}]})).is_err());
    assert!(s.execute("image.applyDataSet", json!({"name": "missing"})).is_err());
}

#[test]
fn csv_import_and_apply() {
    let (mut s, _p, badge, title) = session();
    s.execute(
        "image.variables.define",
        json!({"defs": [
            {"name": "showBadge", "layer": badge.0, "type": "visibility"},
            {"name": "headline", "layer": title.0, "type": "textReplacement"},
        ]}),
    )
    .unwrap();
    let dir = tmp("csv");
    let csv = format!("{dir}/sets.csv");
    std::fs::write(&csv, "DataSet,showBadge,headline\nrow-on,true,On Sale\nrow-off,false,Sold Out\n").unwrap();
    let r = s.execute("file.import.variableDataSets", json!({"path": csv})).unwrap();
    assert_eq!(r["imported"], 2);

    s.execute("image.applyDataSet", json!({"name": "row-off"})).unwrap();
    assert!(!doc(&s).layer(badge).unwrap().visible);
    assert_eq!(text_of(&s, title), "Sold Out");

    s.execute("image.applyDataSet", json!({"name": "row-on"})).unwrap();
    assert!(doc(&s).layer(badge).unwrap().visible);
    assert_eq!(text_of(&s, title), "On Sale");
}

#[test]
fn csv_fields_preserve_unicode_and_quoting() {
    for (line, delimiter, expected) in [
        ("Café,Привет,日本語,👩‍🎨,e\u{301}", b',', vec!["Café", "Привет", "日本語", "👩‍🎨", "e\u{301}"]),
        ("\"Café, Привет\",\"日本語 \"\"quoted\"\" 👩‍🎨\",", b',', vec!["Café, Привет", "日本語 \"quoted\" 👩‍🎨", ""]),
        ("Café;\"Привет;日本語\";👩‍🎨", b';', vec!["Café", "Привет;日本語", "👩‍🎨"]),
        ("Café\t\"Привет\t日本語\"\t👩‍🎨", b'\t', vec!["Café", "Привет\t日本語", "👩‍🎨"]),
        ("plain,\"with, comma\",\"a\"\"b\",,", b',', vec!["plain", "with, comma", "a\"b", "", ""]),
        ("", b',', vec![""]),
    ] {
        assert_eq!(split_csv(line, delimiter), expected, "{line:?}");
    }
}

#[test]
fn csv_import_preserves_unicode_headers_names_and_text() {
    let dir = tmp("unicode-text");
    let csv = format!("{dir}/sets.csv");
    for delimiter in [",", ";", "\t"] {
        let (mut s, _photo, badge, title) = session();
        s.execute(
            "image.variables.define",
            json!({"defs": [
                {"name": "показать", "layer": badge.0, "type": "visibility"},
                {"name": "заголовок café", "layer": title.0, "type": "textReplacement"},
            ]}),
        )
        .unwrap();
        let text = "Café, Привет; 日本語\t👩‍🎨 e\u{301} \"quoted\"";
        std::fs::write(
            &csv,
            format!(
                "DataSet{delimiter}показать{delimiter}заголовок café\r\nНабор 日本語{delimiter}true{delimiter}Café\r\n\"Строка \"\"été\"\"\"{delimiter}false{delimiter}\"{}\"\r\n",
                text.replace('"', "\"\"")
            ),
        )
        .unwrap();
        let r = s.execute("file.import.variableDataSets", json!({"path": csv, "delimiter": delimiter})).unwrap();
        assert_eq!(r["imported"], 2);
        let rows = s.execute("variables.list", json!({})).unwrap();
        assert_eq!(rows["dataSets"][0]["name"], "Набор 日本語");
        assert_eq!(rows["dataSets"][0]["values"][1]["variable"], "заголовок café");
        assert_eq!(rows["dataSets"][1]["values"][1]["value"], text);
        s.execute("image.applyDataSet", json!({"name": "Набор 日本語"})).unwrap();
        assert_eq!(text_of(&s, title), "Café");
        s.execute("image.applyDataSet", json!({"name": "Строка \"été\""})).unwrap();
        assert_eq!(text_of(&s, title), text);
        assert!(!doc(&s).layer(badge).unwrap().visible);
        assert!(s.undo());
        assert_eq!(text_of(&s, title), "Café");
        assert!(doc(&s).layer(badge).unwrap().visible);
        assert!(s.redo());
        assert_eq!(text_of(&s, title), text);
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn csv_multiline_values_apply_persist_and_export_as_two_data_sets() {
    for (ending, suffix) in [("\n", "lf"), ("\r\n", "crlf")] {
        let (mut s, _photo, badge, title) = session();
        s.execute(
            "image.variables.define",
            json!({"defs": [
                {"name": "showBadge", "layer": badge.0, "type": "visibility"},
                {"name": "headline", "layer": title.0, "type": "textReplacement"},
            ]}),
        )
        .unwrap();
        let first = format!("Café, \"quoted\"{ending}{ending}日本語 👩‍🎨");
        let second = format!("Fin{ending}e\u{301}");
        let dir = tmp(&format!("multiline-{suffix}"));
        let csv = format!("{dir}/sets.csv");
        // Blank physical lines inside a quoted field are data; blank records between rows are
        // ignored. The last quoted field ends at EOF, without a final record terminator.
        std::fs::write(&csv, format!("DataSet,showBadge,headline{ending}one,true,\"{}\"{ending}{ending}two,false,\"{second}\"", first.replace('"', "\"\"")))
            .unwrap();
        let result = s.execute("file.import.variableDataSets", json!({"path": csv})).unwrap();
        assert_eq!(result["imported"], 2, "{suffix}");
        let rows = s.execute("variables.list", json!({})).unwrap();
        assert_eq!(rows["dataSets"][0]["values"][1]["value"], first);
        assert_eq!(rows["dataSets"][1]["values"][1]["value"], second);
        for (name, text, visible) in [("one", &first, true), ("two", &second, false)] {
            s.execute("image.applyDataSet", json!({"name": name})).unwrap();
            assert_eq!(text_of(&s, title), *text);
            assert_eq!(doc(&s).layer(badge).unwrap().visible, visible);
        }

        let native = format!("{dir}/multiline.pcraft");
        crate::file_cmds::save_doc(doc(&s), &native, None).unwrap();
        let mut reopened = Session::new();
        crate::file_cmds::open_bytes_as(&mut reopened, "multiline.pcraft", &std::fs::read(&native).unwrap(), None, Some(native)).unwrap();
        assert_eq!(doc(&reopened).variables, doc(&s).variables);
        assert_eq!(text_of(&reopened, title), second);
        reopened.execute("image.applyDataSet", json!({"name": "one"})).unwrap();
        assert_eq!(text_of(&reopened, title), first);
        let before_export = doc(&reopened).clone();
        let out = reopened.execute("file.export.dataSetsAsFiles", json!({"dir": format!("{dir}/out"), "format": "png"})).unwrap();
        assert_eq!(out["count"], 2);
        let files = out["files"].as_array().unwrap();
        assert_eq!(files.len(), 2);
        let images: Vec<_> = files
            .iter()
            .map(|path| {
                let path = path.as_str().unwrap();
                photocraft_io::import(path, &std::fs::read(path).unwrap()).unwrap().document
            })
            .collect();
        assert!(images.iter().all(|image| image.size == doc(&reopened).size));
        assert_ne!(images[0].layers[0].surface(), images[1].layers[0].surface(), "the exported rows have different content");
        assert_eq!(*doc(&reopened), before_export, "export keeps the live document unchanged");
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn csv_multiline_default_names_count_logical_nonblank_records() {
    let dir = tmp("multiline-default-names");
    let csv = format!("{dir}/sets.csv");
    for (header, first, second) in [("headline", "\"first\n\nline\"", "\"\""), ("DataSet,headline", ",\"first\n\nline\"", ",\"\"")] {
        let (mut s, _, _, title) = session();
        s.execute("image.variables.define", json!({"defs": [{"name": "headline", "layer": title.0, "type": "textReplacement"}]})).unwrap();
        std::fs::write(&csv, format!("\n \t\n{header}\n\n{first}\n \n{second}")).unwrap();
        assert_eq!(s.execute("file.import.variableDataSets", json!({"path": csv})).unwrap()["imported"], 2);
        let rows = s.execute("variables.list", json!({})).unwrap();
        assert_eq!(rows["dataSets"][0]["name"], "Data Set 1");
        assert_eq!(rows["dataSets"][1]["name"], "Data Set 2");
        assert_eq!(rows["dataSets"][0]["values"][0]["value"], "first\n\nline");
        assert_eq!(rows["dataSets"][1]["values"][0]["value"], "", "a quoted empty field is a record");
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn csv_import_preserves_unicode_pixel_paths() {
    let (mut s, photo, _badge, _title) = session();
    let dir = tmp("unicode-pixels");
    let png = format!("{dir}/изображение,日本語.png");
    let csv = format!("{dir}/sets.csv");
    let mut src = Session::new();
    src.execute("file.new", json!({"width": 8, "height": 8, "background": "#0000ff"})).unwrap();
    let bytes = photocraft_io::export(&src.active().unwrap().doc, "png", &photocraft_io::ExportOptions::default()).unwrap().bytes;
    std::fs::write(&png, bytes).unwrap();
    s.execute("image.variables.define", json!({"defs": [{"name": "фото", "layer": photo.0, "type": "pixelReplacement", "method": "conform"}]})).unwrap();
    std::fs::write(&csv, format!("DataSet,фото\nсиний,\"{png}\"\n")).unwrap();
    s.execute("file.import.variableDataSets", json!({"path": csv})).unwrap();
    let rows = s.execute("variables.list", json!({})).unwrap();
    assert_eq!(rows["dataSets"][0]["values"][0]["value"], png);
    s.execute("image.applyDataSet", json!({"name": "синий"})).unwrap();
    let pixel = doc(&s).layer(photo).unwrap().surface().unwrap().pixel(32, 24);
    assert_eq!(pixel, vec![0.0, 0.0, 1.0, 1.0]);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn csv_import_errors_leave_existing_data_sets_unchanged() {
    let (mut s, _photo, _badge, _title) = session();
    s.execute("image.variables.dataSets", json!({"dataSets": [{"name": "Сохранённый набор", "values": []}]})).unwrap();
    let dir = tmp("invalid-csv");
    let csv = format!("{dir}/sets.csv");
    for bytes in [b"".as_slice(), b"DataSet,headline\nrow,\xff".as_slice()] {
        std::fs::write(&csv, bytes).unwrap();
        let before = s.active().unwrap().doc.clone();
        assert!(s.execute("file.import.variableDataSets", json!({"path": csv})).is_err());
        assert!(Arc::ptr_eq(&before, &s.active().unwrap().doc));
        assert_eq!(doc(&s).variables.data_sets[0].name, "Сохранённый набор");
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn export_data_sets_as_files() {
    let (mut s, _p, badge, title) = session();
    s.execute(
        "image.variables.define",
        json!({"defs": [
            {"name": "showBadge", "layer": badge.0, "type": "visibility"},
            {"name": "headline", "layer": title.0, "type": "textReplacement"},
        ]}),
    )
    .unwrap();
    s.execute(
        "image.variables.dataSets",
        json!({"dataSets": [
            {"name": "one", "values": [{"variable": "headline", "kind": "text", "value": "1"}]},
            {"name": "two", "values": [{"variable": "showBadge", "kind": "visibility", "value": false}]},
        ]}),
    )
    .unwrap();
    let dir = tmp("out");
    let r = s.execute("file.export.dataSetsAsFiles", json!({"dir": dir, "format": "png"})).unwrap();
    assert_eq!(r["count"], 2);
    assert!(std::path::Path::new(&format!("{dir}/one.png")).exists());
    assert!(std::path::Path::new(&format!("{dir}/two.png")).exists());
    // Filename template with {index}.
    let d2 = tmp("tmpl");
    s.execute("file.export.dataSetsAsFiles", json!({"dir": d2, "format": "png", "naming": "row-{index}"})).unwrap();
    assert!(std::path::Path::new(&format!("{d2}/row-001.png")).exists());
    assert!(std::path::Path::new(&format!("{d2}/row-002.png")).exists());
    // Exporting doesn't mutate the live document.
    assert!(doc(&s).layer(badge).unwrap().visible);
    assert_eq!(text_of(&s, title), "Old");
}

fn visibility_export_session(names: &[&str]) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 8, "height": 8, "background": "white", "name": "card"})).unwrap();
    let layer = doc(&s).layers[0].id;
    s.execute("image.variables.define", json!({"defs": [{"name": "show", "layer": layer.0, "type": "visibility"}]})).unwrap();
    let sets: Vec<_> =
        names.iter().enumerate().map(|(i, name)| json!({"name": name, "values": [{"variable": "show", "kind": "visibility", "value": i == 0}]})).collect();
    s.execute("image.variables.dataSets", json!({"dataSets": sets})).unwrap();
    s
}

#[test]
fn export_data_sets_rejects_collisions_before_writing() {
    for (label, names, naming, collision) in [
        ("sanitized", ["unique", "a b", "a_b"], "{name}", "a_b.png"),
        ("constant", ["unique", "a b", "a_b"], "{document}", "card.png"),
        ("case", ["unique", "Card", "card"], "{name}", "card.png"),
    ] {
        let mut s = visibility_export_session(&names);
        let before = s.active().unwrap().doc.clone();
        let revision = s.active().unwrap().revision;
        let history = s.active().unwrap().history.entries();
        let past = s.active().unwrap().history.past_len();
        let redo = s.active().unwrap().history.can_redo();
        let dir = tmp(&format!("collision-{label}"));
        let first = format!("{dir}/unique.png");
        let collided = format!("{dir}/{collision}");
        std::fs::write(&first, b"keep earlier export").unwrap();
        std::fs::write(&collided, b"keep colliding export").unwrap();
        let err = s.execute("file.export.dataSetsAsFiles", json!({"dir": dir, "naming": naming})).unwrap_err();
        assert!(matches!(err, EngineError::BadParams { .. }), "{err}");
        let message = err.to_string();
        assert!(message.contains(collision) && message.contains("{index}"), "{message}");
        assert_eq!(std::fs::read(&first).unwrap(), b"keep earlier export");
        assert_eq!(std::fs::read(&collided).unwrap(), b"keep colliding export");
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 2);
        let absent = format!("{dir}/not-created");
        assert!(s.execute("file.export.dataSetsAsFiles", json!({"dir": absent, "naming": naming})).is_err());
        assert!(!std::path::Path::new(&absent).exists());
        let after = s.active().unwrap();
        assert!(Arc::ptr_eq(&before, &after.doc));
        assert_eq!(after.revision, revision);
        assert_eq!(after.history.entries(), history);
        assert_eq!(after.history.past_len(), past);
        assert_eq!(after.history.can_redo(), redo);
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn export_data_sets_unique_paths_preserve_variations_and_document() {
    for (label, names, naming, selected, expected) in [
        ("indexed", ["a b", "a_b"], "card-{index}", None, vec!["card-001.png", "card-002.png"]),
        ("distinct", ["a-b", "a_b"], "{name}", None, vec!["a-b.png", "a_b.png"]),
        ("subset", ["a b", "a_b"], "{name}", Some(vec!["a b"]), vec!["a_b.png"]),
    ] {
        let mut s = visibility_export_session(&names);
        let before = s.active().unwrap().doc.clone();
        let revision = s.active().unwrap().revision;
        let history = s.active().unwrap().history.entries();
        let past = s.active().unwrap().history.past_len();
        let redo = s.active().unwrap().history.can_redo();
        let dir = tmp(&format!("unique-{label}"));
        let result = s.execute("file.export.dataSetsAsFiles", json!({"dir": dir, "naming": naming, "dataSets": selected})).unwrap();
        let paths: Vec<_> = expected.iter().map(|name| format!("{dir}/{name}")).collect();
        assert_eq!(result, json!({"files": paths, "count": paths.len()}));
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), paths.len());
        for (i, path) in paths.iter().enumerate() {
            let exported = photocraft_io::import(path, &std::fs::read(path).unwrap()).unwrap().document;
            let pixel = photocraft_compose::render(&exported, Rect::new(0, 0, 1, 1)).px[0];
            assert_eq!(pixel, if i == 0 { [1.0; 4] } else { [0.0; 4] });
        }
        let after = s.active().unwrap();
        assert!(Arc::ptr_eq(&before, &after.doc));
        assert_eq!(after.revision, revision);
        assert_eq!(after.history.entries(), history);
        assert_eq!(after.history.past_len(), past);
        assert_eq!(after.history.can_redo(), redo);
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn pixel_replacement_changes_the_layer() {
    let (mut s, photo, _b, _t) = session();
    // Make a 8x8 solid-blue PNG to drop in.
    let dir = tmp("px");
    let png = format!("{dir}/blue.png");
    {
        let mut src = Session::new();
        src.execute("file.new", json!({"width": 8, "height": 8, "background": "#0000ff"})).unwrap();
        let bytes = photocraft_io::export(&src.active().unwrap().doc, "png", &photocraft_io::ExportOptions::default()).unwrap().bytes;
        std::fs::write(&png, bytes).unwrap();
    }
    s.execute(
        "image.variables.define",
        json!({"defs": [
            {"name": "img", "layer": photo.0, "type": "pixelReplacement", "method": "conform"},
        ]}),
    )
    .unwrap();
    s.execute(
        "image.variables.dataSets",
        json!({"dataSets": [
            {"name": "blue", "values": [{"variable": "img", "kind": "pixels", "value": png}]},
        ]}),
    )
    .unwrap();
    // Before: grey.
    let before = s.execute("document.pixel", json!({"x": 32, "y": 24})).unwrap();
    s.execute("image.applyDataSet", json!({"name": "blue"})).unwrap();
    let after = s.execute("document.pixel", json!({"x": 32, "y": 24})).unwrap();
    assert_ne!(before, after, "pixel layer content changed");
    // Blue dominates.
    let rgba = after["rgba"].as_array().cloned().unwrap_or_default();
    if rgba.len() == 4 {
        let b = rgba[2].as_f64().unwrap();
        let r = rgba[0].as_f64().unwrap();
        assert!(b > r, "replacement is blue-ish: {rgba:?}");
    }
}

#[test]
fn applied_text_is_rendered_in_the_document_and_in_exports() {
    // Applying a text value cleared the type layer's render cache and nothing rebuilt it, so the
    // text vanished from the canvas and from every exported file (#990).
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 200, "height": 60, "background": "white"})).unwrap();
    let id = s.execute("type.create", json!({"x": 4, "y": 40, "text": "HELLO", "size": 32, "color": "#000000"})).unwrap()["layer"].clone();
    let dark = |d: &Document| photocraft_compose::render(d, d.bounds()).px.iter().filter(|p| p[0] < 0.5).count();
    assert!(dark(doc(&s)) > 0);
    s.execute("image.variables.define", json!({"defs": [{"name": "headline", "layer": id, "type": "textReplacement"}]})).unwrap();
    s.execute("image.variables.dataSets", json!({"dataSets": [{"name": "row1", "values": [{"variable": "headline", "kind": "text", "value": "BYE"}]}]}))
        .unwrap();
    let dir = tmp("text-export");
    let out = s.execute("file.export.dataSetsAsFiles", json!({"dir": dir, "format": "png"})).unwrap();
    let path = out["files"][0].as_str().unwrap();
    let exported = photocraft_io::import(path, &std::fs::read(path).unwrap()).unwrap().document;
    assert!(dark(&exported) > 0, "the exported row keeps its text");
    s.execute("image.applyDataSet", json!({"name": "row1"})).unwrap();
    assert!(dark(doc(&s)) > 0, "the applied text is drawn");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A text layer created bold-italic with tracking, underline, all caps and colour, bound to a
/// `headline` text variable with one data set per value.
fn styled_text_session(values: &[&str]) -> (Session, Value) {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 400, "height": 120, "background": "white"})).unwrap();
    let id = s
        .execute(
            "type.create",
            json!({"x": 10, "y": 60, "text": "Before", "font": "Inter", "weight": 700, "italic": true, "size": 40,
                   "color": "#C0392B", "tracking": 120, "underline": true, "caps": "all"}),
        )
        .unwrap()["layer"]
        .clone();
    s.execute("image.variables.define", json!({"defs": [{"name": "headline", "layer": id, "type": "textReplacement"}]})).unwrap();
    let sets: Vec<Value> =
        values.iter().enumerate().map(|(i, v)| json!({"name": format!("s{i}"), "values": [{"variable": "headline", "kind": "text", "value": v}]})).collect();
    s.execute("image.variables.dataSets", json!({ "dataSets": sets })).unwrap();
    (s, id)
}

fn text_layer(s: &Session, id: &Value) -> TextLayer {
    match &doc(s).layer(LayerId(id.as_u64().unwrap())).unwrap().content {
        LayerContent::Text(t) => t.clone(),
        _ => panic!("not a text layer"),
    }
}

#[test]
fn applied_text_keeps_the_character_style() {
    // Applying a text value cleared the style runs, so weight, italic, tracking, underline and
    // caps fell back to the defaults (#2742). The new text takes the first character's style.
    let (mut s, id) = styled_text_session(&["After"]);
    let before = text_layer(&s, &id);
    let style = before.char_runs()[0].style.clone();
    assert_eq!((style.weight, style.italic), (700, true), "precondition: {style:?}");
    let paragraphs = before.paragraph_runs()[0].style.clone();
    s.execute("image.applyDataSet", json!({"name": "s0"})).unwrap();
    let after = text_layer(&s, &id);
    assert_eq!(after.text, "After");
    let runs = after.char_runs();
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].len, "After".len());
    assert_eq!(runs[0].style, style, "the whole character style is kept");
    assert_eq!(after.paragraph_runs()[0].style, paragraphs, "paragraph settings are kept");
    assert_eq!((after.font_family.as_str(), after.size_pt, after.color), (before.font_family.as_str(), before.size_pt, before.color));
    // type.info reports the same style as before.
    let info = s.execute("type.info", json!({"layer": id})).unwrap();
    assert_eq!(info["runs"][0]["style"]["weight"], 700);
    assert_eq!(info["runs"][0]["style"]["italic"], true);

    // Undo restores the original text and runs.
    s.execute("edit.undo", json!({})).unwrap();
    let undone = text_layer(&s, &id);
    assert_eq!(undone.text, "Before");
    assert_eq!(undone.runs, before.runs);
}

#[test]
fn applied_text_takes_the_first_runs_style_and_survives_empty_values() {
    let (mut s, id) = styled_text_session(&["One style", ""]);
    // Make the second half of the text regular, upright and blue: a two-run layer.
    s.execute("type.edit", json!({"layer": id, "runs": [{"start": 3, "end": 6, "weight": 400, "italic": false, "color": "#0000FF"}]})).unwrap();
    let before = text_layer(&s, &id);
    assert_eq!(before.char_runs().len(), 2, "precondition: two runs");
    let first = before.char_runs()[0].style.clone();
    s.execute("image.applyDataSet", json!({"name": "s0"})).unwrap();
    let after = text_layer(&s, &id);
    assert_eq!(after.text, "One style");
    let runs = after.char_runs();
    assert_eq!(runs.len(), 1, "the replacement is one run");
    assert_eq!(runs[0].style, first, "in the first run's style");

    // An empty value empties the layer without panicking and keeps the style for later text.
    s.execute("image.applyDataSet", json!({"name": "s1"})).unwrap();
    let empty = text_layer(&s, &id);
    assert_eq!(empty.text, "");
    assert_eq!(empty.char_runs()[0].style, first);
    s.execute("image.applyDataSet", json!({"name": "s0"})).unwrap();
    assert_eq!(text_layer(&s, &id).char_runs()[0].style, first);
}

#[test]
fn exported_data_sets_keep_the_character_style() {
    // file.export.dataSetsAsFiles applies each set to a copy of the document the same way.
    let (mut s, id) = styled_text_session(&["After"]);
    let style = text_layer(&s, &id).char_runs()[0].style.clone();
    let dir = tmp("styled-export");
    let out = s.execute("file.export.dataSetsAsFiles", json!({"dir": dir, "format": "psd"})).unwrap();
    let path = out["files"][0].as_str().unwrap();
    let exported = photocraft_io::import(path, &std::fs::read(path).unwrap()).unwrap().document;
    let t = exported
        .layers
        .iter()
        .find_map(|l| match &l.content {
            LayerContent::Text(t) => Some(t.clone()),
            _ => None,
        })
        .expect("the export keeps the type layer");
    assert_eq!(t.text, "After");
    let got = &t.char_runs()[0].style;
    assert_eq!((got.weight, got.italic, got.underline), (style.weight, style.italic, style.underline));
    let _ = std::fs::remove_dir_all(&dir);
}
