//! Layer › Rasterize acts on every selected layer, as in Photoshop ("select the layer or layers
//! you'd like to rasterize"; the Layers panel's right-click item reads "Rasterize Layers"), not
//! only on the one that was clicked. An explicit `layer` param still targets exactly that layer.

use photocraft_doc::{LayerContent, LayerId};
use photocraft_engine::Session;
use serde_json::{Value, json};

fn kind(s: &Session, id: LayerId) -> &'static str {
    match &s.active().unwrap().doc.layer(id).unwrap().content {
        LayerContent::Text(_) => "type",
        LayerContent::Shape(_) => "shape",
        LayerContent::Fill(_) => "fill",
        LayerContent::Smart(_) => "smart",
        LayerContent::Raster(_) => "raster",
        _ => "other",
    }
}

fn kinds(s: &Session, ids: &[LayerId]) -> Vec<&'static str> {
    ids.iter().map(|id| kind(s, *id)).collect()
}

fn id_of(r: Value) -> LayerId {
    LayerId(r["layer"].as_u64().unwrap())
}

struct Scene {
    s: Session,
    text: LayerId,
    shape: LayerId,
    smart: LayerId,
    fill: LayerId,
    /// A shape layer that is never selected.
    bystander: LayerId,
}

fn scene() -> Scene {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 64, "height": 64})).unwrap();
    let text = id_of(s.execute("type.create", json!({"text": "Hi", "size": 24, "x": 4, "y": 30})).unwrap());
    let shape = id_of(s.execute("shape.create", json!({"kind": "ellipse", "rect": [4, 4, 40, 30]})).unwrap());
    s.execute("layer.new.layer", json!({"name": "Source"})).unwrap();
    s.execute("paint.stroke", json!({"points": [[20, 20]], "size": 12})).unwrap();
    let smart = id_of(s.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap());
    let fill = id_of(s.execute("layer.newFillLayer.solidColor", json!({"color": "#00ff00"})).unwrap());
    let bystander = id_of(s.execute("shape.create", json!({"kind": "rect", "rect": [30, 30, 60, 60]})).unwrap());
    let sc = Scene { s, text, shape, smart, fill, bystander };
    assert_eq!(kinds(&sc.s, &[sc.text, sc.shape, sc.smart, sc.fill, sc.bystander]), ["type", "shape", "smart", "fill", "shape"]);
    sc
}

impl Scene {
    /// Select `ids` with the last one active (like ⌘-clicking rows, then right-clicking the last).
    fn select(&mut self, ids: &[LayerId]) {
        self.s.execute("layer.select", json!({"layer": ids[0].0})).unwrap();
        for id in &ids[1..] {
            self.s.execute("layer.select", json!({"layer": id.0, "mode": "add"})).unwrap();
        }
        assert_eq!(self.s.active().unwrap().selected_layers().len(), ids.len());
    }
}

#[test]
fn rasterize_layer_converts_every_selected_layer_in_one_step() {
    let mut sc = scene();
    let all = [sc.text, sc.shape, sc.smart, sc.fill];
    sc.select(&all);
    let before = sc.s.active().unwrap().selected_layers();
    let r = sc.s.execute("layer.rasterize.layer", json!({})).unwrap();
    assert_eq!(r["layers"].as_array().unwrap().len(), 4, "{r}");
    assert_eq!(kinds(&sc.s, &all), ["raster"; 4]);
    assert_eq!(kind(&sc.s, sc.bystander), "shape", "an unselected layer is left alone");
    assert_eq!(sc.s.active().unwrap().selected_layers(), before, "the selection survives");
    // One history step brings all four back.
    assert!(sc.s.undo());
    assert_eq!(kinds(&sc.s, &all), ["type", "shape", "smart", "fill"]);
    assert!(sc.s.redo());
    assert_eq!(kinds(&sc.s, &all), ["raster"; 4]);
}

#[test]
fn a_kind_command_touches_only_the_selected_layers_of_that_kind() {
    let mut sc = scene();
    sc.select(&[sc.text, sc.shape, sc.smart]);
    sc.s.execute("layer.rasterize.type", json!({})).unwrap();
    assert_eq!(kinds(&sc.s, &[sc.text, sc.shape, sc.smart]), ["raster", "shape", "smart"]);
    // Nothing selected is a type layer any more.
    assert!(sc.s.execute("layer.rasterize.type", json!({})).is_err());
    sc.s.execute("layer.rasterize.shape", json!({})).unwrap();
    assert_eq!(kinds(&sc.s, &[sc.shape, sc.smart, sc.bystander]), ["raster", "smart", "shape"]);
    sc.s.execute("layer.rasterize.smartObject", json!({})).unwrap();
    assert_eq!(kind(&sc.s, sc.smart), "raster");
    assert_eq!(kind(&sc.s, sc.bystander), "shape");
}

#[test]
fn an_explicit_layer_param_stays_a_single_layer() {
    // The "Rasterize?" prompt and scripts pass `layer`: the rest of the selection is not touched.
    let mut sc = scene();
    sc.select(&[sc.text, sc.shape, sc.smart]);
    sc.s.execute("layer.rasterize.layer", json!({"layer": sc.shape.0})).unwrap();
    assert_eq!(kinds(&sc.s, &[sc.text, sc.shape, sc.smart]), ["type", "raster", "smart"]);
    sc.s.execute("layer.rasterize.shape", json!({"layer": sc.bystander.0})).unwrap();
    assert_eq!(kind(&sc.s, sc.bystander), "raster");
    assert_eq!(kinds(&sc.s, &[sc.text, sc.smart]), ["type", "smart"]);
}

#[test]
fn a_single_selected_layer_behaves_as_before() {
    let mut sc = scene();
    sc.select(&[sc.text]);
    let r = sc.s.execute("layer.rasterize.layer", json!({})).unwrap();
    assert_eq!(r["layer"].as_u64(), Some(sc.text.0));
    assert_eq!(kinds(&sc.s, &[sc.text, sc.shape]), ["raster", "shape"]);
    // A layer with nothing to rasterize reports it, and no history step is added.
    let err = sc.s.execute("layer.rasterize.layer", json!({})).unwrap_err().to_string();
    assert!(err.contains("nothing to rasterize"), "{err}");
    let err = sc.s.execute("layer.rasterize.type", json!({})).unwrap_err().to_string();
    assert!(err.contains("not a type layer"), "{err}");
}

#[test]
fn a_selection_with_nothing_to_rasterize_is_an_error_not_a_step() {
    let mut sc = scene();
    sc.s.execute("layer.rasterize.layer", json!({"layer": sc.text.0})).unwrap();
    sc.s.execute("layer.rasterize.layer", json!({"layer": sc.shape.0})).unwrap();
    sc.select(&[sc.text, sc.shape]);
    let revision = sc.s.active().unwrap().revision;
    let err = sc.s.execute("layer.rasterize.layer", json!({})).unwrap_err().to_string();
    assert!(err.contains("none of the selected layers"), "{err}");
    assert_eq!(sc.s.active().unwrap().revision, revision, "a refused command changes nothing");
}

#[test]
fn bad_targets_fail_gracefully() {
    let mut s = Session::new();
    for id in ["layer.rasterize.layer", "layer.rasterize.type", "layer.rasterize.shape", "layer.rasterize.fillContent", "layer.rasterize.smartObject"] {
        assert!(s.execute(id, json!({})).is_err(), "{id}: no document");
    }
    let mut sc = scene();
    for id in ["layer.rasterize.layer", "layer.rasterize.shape", "layer.rasterize.smartObject"] {
        assert!(sc.s.execute(id, json!({"layer": u64::MAX})).is_err(), "{id}: no such layer");
    }
}
