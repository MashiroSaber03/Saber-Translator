//! Krita documents opened through the session: read-only source, 16-bit layers, editable
//! layers with history, and saving as OpenRaster or native. The files come from
//! `tools/krita-oracles/generate.py` in https://github.com/storytold/photocraft-corpus. Run with
//! `KRITA_TEST_DIR=<dir> cargo test -p photocraft-engine --test kra -- --ignored --nocapture`.
use photocraft_engine::Session;
use serde_json::json;

fn open(path: &std::path::Path) -> Session {
    let mut session = Session::new();
    let name = path.to_string_lossy().into_owned();
    photocraft_engine::file_cmds::open_bytes_as(&mut session, &name, &std::fs::read(path).unwrap(), None, Some(name.clone())).unwrap();
    session
}

#[test]
#[ignore = "requires KRITA_TEST_DIR with Krita-authored .kra files"]
fn krita_documents_open_read_only_and_stay_editable() {
    let dir = std::path::PathBuf::from(std::env::var("KRITA_TEST_DIR").expect("set KRITA_TEST_DIR"));
    let path = dir.join("depth-rgba-u16.kra");
    let mut session = open(&path);
    let active = session.active().unwrap();
    assert!(active.source_read_only && active.path.is_none());
    assert_eq!(active.doc.depth, photocraft_color::SampleType::U16);
    assert!(session.execute("file.save", json!({})).is_err());

    // Krita's merged image is the composite (the corpus files carry no ICC profile, so both
    // sides are sRGB); crates/io/tests/krita_corpus.rs compares every file.
    let ours = photocraft_compose::flatten(&session.active().unwrap().doc);
    let zip_bytes = std::fs::read(&path).unwrap();
    let zip = photocraft_format::zip::ZipReader::new(&zip_bytes).unwrap();
    let merged = photocraft_io::import("m.png", &zip.read_by_name("mergedimage.png", 1 << 28).unwrap()).unwrap().document;
    let theirs = photocraft_compose::flatten(&merged);
    let worst = ours.px.iter().zip(&theirs.px).map(|(a, b)| (0..4).map(|c| (a[c] * a[3] - b[c] * b[3]).abs()).fold(0.0f32, f32::max)).fold(0.0, f32::max);
    assert!(worst <= 3.0 / 255.0, "{}", worst * 255.0);

    // Imported Krita layers take the ordinary commands and history.
    let mut session = open(&dir.join("layers-opacity-visibility.kra"));
    let before = session.active().unwrap().doc.clone();
    let id = before.layers.last().unwrap().id;
    session.execute("layer.setProps", json!({"layer":id.0,"name":"Edited","opacity":0.25,"blend":"Screen"})).unwrap();
    assert_eq!(session.active().unwrap().doc.layer(id).unwrap().blend, photocraft_color::BlendMode::Screen);
    session.execute("layer.select", json!({"layer":id.0})).unwrap();
    session.execute("paint.stroke", json!({"points":[[2,2],[30,30]],"size":6,"hardness":1.0,"color":"#00ff00"})).unwrap();
    assert!(session.undo() && session.undo());
    assert_eq!(session.active().unwrap().doc.layers, before.layers);
    // ...and save as OpenRaster or native, never back to .kra.
    let doc = &session.active().unwrap().doc;
    assert!(photocraft_io::export(doc, "x.kra", &Default::default()).is_err());
    let ora = photocraft_io::export(doc, "x.ora", &Default::default()).unwrap();
    assert_eq!(photocraft_io::import("x.ora", &ora.bytes).unwrap().document.layers.len(), doc.layers.len());
    let native = photocraft_io::export(doc, "x.pcraft", &Default::default()).unwrap();
    assert_eq!(photocraft_io::import("x.pcraft", &native.bytes).unwrap().document.layers, doc.layers);
}
