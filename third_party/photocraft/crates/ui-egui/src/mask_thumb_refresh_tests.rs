//! #2117: the Layers panel's mask thumbnail follows whole-mask edits such as Image › Adjustments ›
//! Invert, not only brush strokes. A fresh Reveal All mask has no tiles, so inverting it only
//! changes the surface's default pixel; the thumbnail cache key must see that.

use egui::vec2;
use egui_kittest::Harness;
use photocraft_doc::LayerId;
use photocraft_raster::Surface;
use serde_json::json;

use crate::mask_thumbs_ui::THUMB_MASK;
use crate::{PhotocraftApp, surface_fingerprint};

#[test]
fn fingerprint_sees_the_default_pixel() {
    let fmt = photocraft_color::PixelFormat::GRAY8;
    let white = Surface::with_default(fmt, &[1.0]);
    let black = Surface::with_default(fmt, &[0.0]);
    assert_eq!(white.tile_count(), 0);
    assert_ne!(surface_fingerprint(&white), surface_fingerprint(&black));
    assert_eq!(surface_fingerprint(&white), surface_fingerprint(&white.clone()));
}

#[test]
fn inverting_a_tileless_mask_refreshes_its_thumbnail() {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 200, "height": 100})).unwrap();
    let id = s.execute("layer.new.layer", json!({"name": "Masked"})).unwrap()["layer"].as_u64().unwrap();
    s.execute("layer.layerMask.revealAll", json!({"layer": id})).unwrap();
    let mut h = Harness::builder().with_size(vec2(1440.0, 1000.0)).with_max_steps(64).build_eframe(move |cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        PhotocraftApp::new(s, crate::Services::default())
    });
    let ctx = h.ctx.clone();
    let (req, _rx) = crate::control::ControlRequest::new("ui.set", json!({"dockTabs": {"layers": 0}}));
    crate::control::handle(h.state_mut(), &ctx, &req);
    h.run_steps(8);
    let key = (LayerId(id), THUMB_MASK);
    let before = h.state().thumbs.get(&key).map(|(r, _)| *r).expect("mask thumbnail drawn");

    h.state_mut().run("image.adjustments.invert", json!({"target": "mask"})).unwrap();
    h.run_steps(4);
    let mask = h.state().session.active().unwrap().doc.layer(LayerId(id)).unwrap().mask.clone().unwrap();
    assert_eq!(mask.surface.tile_count(), 0, "the inverted Reveal All mask is still tile-less");
    assert_eq!(mask.surface.default_pixel(), vec![0.0], "and now hides everything");
    let after = h.state().thumbs.get(&key).map(|(r, _)| *r).expect("mask thumbnail drawn");
    assert_ne!(before, after, "the thumbnail was rebuilt after Invert");
    assert_eq!(after, surface_fingerprint(&mask.surface) ^ 200u64 << 40, "keyed on the inverted mask");
}
