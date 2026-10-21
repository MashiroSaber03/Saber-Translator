//! `display_doc` builds a Duotone or live-adjust document once per snapshot (#4527), not on each
//! of its calls in a frame.

use super::*;
use serde_json::json;

fn app() -> PhotocraftApp {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
    app.run("file.new", json!({"width": 64, "height": 48, "mode": "gray"})).unwrap();
    app
}

#[test]
fn duotone_display_document_is_built_once_per_snapshot() {
    let mut app = app();
    app.run("image.mode.duotone", json!({"type": "duotone", "inks": [{"name": "Black"}, {"name": "Orange", "color": "#ff8000"}]})).unwrap();
    let (a, key) = display_doc(&mut app, 0);
    assert_eq!(key, 1 << 41);
    let (b, _) = display_doc(&mut app, 0);
    assert!(std::sync::Arc::ptr_eq(&a, &b), "the second call reuses the first's document");
    assert_eq!(a.layers.len(), app.session.documents()[0].doc.layers.len() + 1, "with the ink display layer");
    app.run("edit.fill", json!({"color": "#808080"})).unwrap();
    let (c, _) = display_doc(&mut app, 0);
    assert!(!std::sync::Arc::ptr_eq(&a, &c), "an edit builds it again");
    assert_eq!(c.layers.last().map(|l| l.name.as_str()), a.layers.last().map(|l| l.name.as_str()));
}

#[test]
fn live_adjust_document_is_built_once_per_setting() {
    let mut app = app();
    app.run("layer.newAdjustmentLayer.levels", json!({})).unwrap();
    let id = app.session.documents()[0].active_layer.unwrap();
    app.live_adjust = Some((id, json!({"inBlack": 20})));
    let (a, k1) = display_doc(&mut app, 0);
    let (b, k2) = display_doc(&mut app, 0);
    assert_eq!(k1, k2);
    assert!(std::sync::Arc::ptr_eq(&a, &b));
    app.live_adjust = Some((id, json!({"inBlack": 40})));
    let (c, k3) = display_doc(&mut app, 0);
    assert_ne!(k1, k3);
    assert!(!std::sync::Arc::ptr_eq(&a, &c), "a new setting builds it again");
    app.live_adjust = None;
    let (d, k4) = display_doc(&mut app, 0);
    assert_eq!(k4, 0);
    assert!(std::sync::Arc::ptr_eq(&d, &app.session.documents()[0].doc), "no preview: the document itself");
}
