//! Actual imported PDN layers exercise the same commands/history as other pixel layers.
use photocraft_engine::Session;
use serde_json::json;

#[test]
#[ignore = "requires PDN_TEST_FILE pointing to a real Paint.NET document"]
fn imported_layers_remain_editable_with_history_and_native_persistence() {
    let path = std::env::var("PDN_TEST_FILE").expect("set PDN_TEST_FILE");
    let mut session = Session::new();
    photocraft_engine::file_cmds::open_bytes_as(&mut session, &path, &std::fs::read(&path).unwrap(), None, Some(path.clone())).unwrap();
    assert!(session.active().unwrap().source_read_only);
    assert!(session.active().unwrap().path.is_none());
    assert!(session.execute("file.save", json!({})).is_err());
    let before = session.active().unwrap().doc.clone();
    let id = before.layers[0].id;
    session.execute("layer.setProps", json!({"layer":id.0,"name":"Edited PDN layer","visible":true,"opacity":0.4,"blend":"Reflect"})).unwrap();
    let edited = session.active().unwrap().doc.layer(id).unwrap();
    assert_eq!(edited.name, "Edited PDN layer");
    assert_eq!(edited.blend, photocraft_color::BlendMode::Reflect);
    assert_eq!(edited.opacity, 0.4);
    assert!(session.undo());
    assert_eq!(session.active().unwrap().doc.layers, before.layers);
    assert!(session.redo());
    let copy = session.execute("layer.duplicate", json!({"layer":id.0})).unwrap()["layer"].as_u64().unwrap();
    session.execute("layer.translate", json!({"layer":copy,"dx":1,"dy":1})).unwrap();
    session.execute("layer.moveTo", json!({"layer":copy,"target":id.0,"position":"below"})).unwrap();
    session.execute("layer.select", json!({"layer":copy})).unwrap();
    let pixels_before = session.active().unwrap().doc.layer(photocraft_doc::LayerId(copy)).unwrap().surface().unwrap().clone();
    session.execute("paint.stroke", json!({"points":[[2,2],[10,10]],"size":4,"hardness":1.0,"color":"#ff0000"})).unwrap();
    assert_ne!(session.active().unwrap().doc.layer(photocraft_doc::LayerId(copy)).unwrap().surface().unwrap(), &pixels_before);
    assert_eq!(session.active().unwrap().doc.layer(id).unwrap().surface(), before.layer(id).unwrap().surface());
    assert!(session.undo());
    assert_eq!(session.active().unwrap().doc.layer(photocraft_doc::LayerId(copy)).unwrap().surface().unwrap(), &pixels_before);
    assert!(session.redo());
    session.execute("layer.layerMask.revealAll", json!({"layer":copy})).unwrap();
    assert!(session.active().unwrap().doc.layer(photocraft_doc::LayerId(copy)).unwrap().mask.is_some());
    session.execute("layer.delete", json!({"layer":copy})).unwrap();
    assert!(session.undo());
    let doc = &session.active().unwrap().doc;
    let native = photocraft_io::export(doc, "edited.pcraft", &Default::default()).unwrap();
    let reopened = photocraft_io::import("edited.pcraft", &native.bytes).unwrap().document;
    assert_eq!(doc.layers, reopened.layers);
}
