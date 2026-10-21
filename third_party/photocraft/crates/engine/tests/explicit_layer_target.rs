//! A command that documents `"layer":id` checks its precondition on that layer, not on whichever
//! layer happens to be active (#466 for smart filters, generalised): a script, action or agent can
//! Replace Contents or Rasterize a smart object while another layer is selected.

use photocraft_doc::{LayerContent, LayerId};
use photocraft_engine::{EngineError, Session};
use photocraft_geom::Rect;
use serde_json::json;

/// A smart object (blue) under a plain pixel layer, which is active.
fn session() -> (Session, u64, u64) {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 8, "height": 6})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.edit("paint", |doc, active| {
        doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap().fill_rect(Rect::new(0, 0, 8, 6), &[0.0, 0.0, 1.0, 1.0]);
        Ok(())
    })
    .unwrap();
    let so = s.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap()["layer"].as_u64().unwrap();
    let pixels = s.execute("layer.new.layer", json!({})).unwrap()["layer"].as_u64().unwrap();
    (s, so, pixels)
}

fn active(s: &Session) -> u64 {
    s.active().unwrap().active_layer.unwrap().0
}

fn kind(s: &Session, id: u64) -> &'static str {
    s.active().unwrap().doc.layer(LayerId(id)).unwrap().content.kind_name()
}

#[test]
fn smart_object_commands_act_on_a_named_layer_while_another_is_active() {
    let (mut s, so, pixels) = session();
    assert_eq!(active(&s), pixels);
    // Without "layer" the active pixel layer decides, as the menus show.
    assert!(!s.is_enabled("layer.smartObjects.replaceContents"));
    // Replace Contents on the named smart object.
    let dir = std::env::temp_dir().join(format!("pcraft-explicit-layer-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let png = dir.join("red.png");
    let red = photocraft_doc::Document::with_background(
        "r",
        photocraft_geom::Size::new(8, 6),
        photocraft_color::ColorMode::Rgb,
        photocraft_color::SampleType::U8,
        photocraft_color::Color::rgba(1.0, 0.0, 0.0, 1.0),
    );
    std::fs::write(&png, photocraft_io::export(&red, "png", &Default::default()).unwrap().bytes).unwrap();
    s.execute("layer.smartObjects.replaceContents", json!({"layer": so, "path": png.to_str().unwrap()})).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    let LayerContent::Smart(sm) = &s.active().unwrap().doc.layer(LayerId(so)).unwrap().content else { panic!("smart object") };
    assert_eq!(sm.cache.as_ref().unwrap().rgba(2, 2), [1.0, 0.0, 0.0, 1.0], "the named smart object was replaced");
    assert_eq!(active(&s), pixels, "the active layer stays");
    // Rasterize it, still without selecting it.
    s.execute("layer.rasterize.smartObject", json!({"layer": so})).unwrap();
    assert_eq!(kind(&s, so), "Pixel");
    assert_eq!(active(&s), pixels);
}

#[test]
fn a_named_layer_of_the_wrong_kind_or_that_does_not_exist_is_refused() {
    let (mut s, _so, pixels) = session();
    let before = s.active().unwrap().doc.clone();
    // The named layer is checked: a pixel layer is not a smart object.
    let r = s.execute("layer.smartObjects.rasterize", json!({"layer": pixels}));
    assert!(matches!(r, Err(EngineError::Disabled(..))), "{r:?}");
    // A layer that doesn't exist falls back to the active layer's check, and still fails.
    assert!(s.execute("layer.smartObjects.rasterize", json!({"layer": u64::MAX})).is_err());
    assert!(s.execute("layer.smartObjects.rasterize", json!({"layer": "nope"})).is_err());
    assert_eq!(s.active().unwrap().doc, before);
    assert_eq!(active(&s), pixels);
}
