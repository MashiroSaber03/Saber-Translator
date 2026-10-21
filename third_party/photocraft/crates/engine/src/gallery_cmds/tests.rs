use photocraft_geom::Rect;
use serde_json::json;

use super::*;

const W: i32 = 40;
const H: i32 = 32;

fn session(depth: u32, mode: &str) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": W, "height": H, "depth": depth, "mode": mode})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.edit("pattern", |doc, active| {
        let l = doc.layer_mut(active.unwrap()).unwrap();
        let surf = l.surface_mut().unwrap();
        let n = surf.channels();
        for y in 0..H {
            for x in 0..W {
                let v = ((x * 7 + y * 3) % 23) as f32 / 22.0;
                let mut px = vec![1.0; n];
                for (c, p) in px.iter_mut().enumerate().take(n - 1) {
                    *p = if (x / 10 + y / 8) % 2 == 0 { v } else { 1.0 - v * (c as f32 + 1.0) / 4.0 };
                }
                surf.write_pixel(x, y, &px);
            }
        }
        Ok(())
    })
    .unwrap();
    s
}

fn pixels(s: &Session) -> Vec<f32> {
    let d = s.active().unwrap();
    d.doc.layer(d.active_layer.unwrap()).unwrap().surface().unwrap().read_region(Rect::new(0, 0, W, H))
}

#[test]
fn every_gallery_filter_runs_at_every_depth_and_undoes() {
    for depth in [8, 16, 32] {
        for f in GalleryFilter::ALL {
            let mut s = session(depth, "rgb");
            let before = pixels(&s);
            let r = s.execute(f.command_id(), json!({})).unwrap_or_else(|e| panic!("{}: {e}", f.key()));
            assert!(r.get("filter").is_some());
            let after = pixels(&s);
            assert_ne!(after, before, "{} @{depth} changed nothing", f.key());
            assert!(after.iter().all(|v| v.is_finite() && (-1e-4..=1.0001).contains(v)), "{} @{depth} out of range", f.key());
            s.execute("edit.undo", json!({})).unwrap();
            assert_eq!(pixels(&s), before, "{} undo", f.key());
        }
    }
}

#[test]
fn gallery_filters_run_in_other_colour_models() {
    for mode in ["gray", "cmyk", "lab"] {
        for f in [GalleryFilter::Cutout, GalleryFilter::GraphicPen, GalleryFilter::Texturizer, GalleryFilter::StainedGlass] {
            let mut s = session(16, mode);
            let before = pixels(&s);
            s.execute(f.command_id(), json!({})).unwrap_or_else(|e| panic!("{} {mode}: {e}", f.key()));
            assert_ne!(pixels(&s), before, "{} {mode}", f.key());
        }
    }
}

#[test]
fn results_do_not_depend_on_tiling() {
    let s = session(32, "rgb");
    let d = s.active().unwrap();
    let surf = d.doc.layer(d.active_layer.unwrap()).unwrap().surface().unwrap().clone();
    let b = Rect::new(0, 0, W, H);
    for f in GalleryFilter::ALL {
        let fp = params_for(f.command_id(), &json!({})).unwrap();
        let big = photocraft_algo::apply_tiled(&surf, &fp, b, b, None, 256, Some(b));
        let small = photocraft_algo::apply_tiled(&surf, &fp, b, b, None, 16, Some(b));
        let (a, c) = (big.read_region(b), small.read_region(b));
        let worst = a.iter().zip(&c).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
        assert!(worst < 1e-3, "{} differs across tilings by {worst}", f.key());
    }
}

#[test]
fn stack_applies_in_order_and_skips_hidden_layers() {
    let run = |effects: Value| {
        let mut s = session(8, "rgb");
        s.execute("filter.filterGallery", json!({ "effects": effects })).unwrap();
        pixels(&s)
    };
    let one = run(json!([{"filter": "cutout", "params": {"numberOfLevels": 3}}]));
    let mut single = session(8, "rgb");
    single.execute("filter.gallery.cutout", json!({"numberOfLevels": 3})).unwrap();
    assert_eq!(one, pixels(&single), "one-layer gallery = the filter's own command");
    let ab = run(json!([{"filter": "cutout"}, {"filter": "glowingEdges"}]));
    let ba = run(json!([{"filter": "glowingEdges"}, {"filter": "cutout"}]));
    assert_ne!(ab, ba, "order matters");
    let hidden = run(json!([{"filter": "cutout"}, {"filter": "glowingEdges", "visible": false}]));
    assert_eq!(hidden, run(json!([{"filter": "cutout"}])));
}

#[test]
fn gallery_validates_and_lists() {
    let mut s = session(8, "rgb");
    assert!(s.execute("filter.filterGallery", json!({})).is_err());
    assert!(s.execute("filter.filterGallery", json!({"effects": [{"filter": "nope"}]})).is_err());
    let too_many: Vec<_> = (0..33).map(|_| json!({"filter": "coloredPencil"})).collect();
    assert!(s.execute("filter.filterGallery", json!({"effects": too_many})).is_err());
    let list = s.execute("filter.filterGallery", json!({"list": true})).unwrap();
    assert_eq!(list["filters"].as_array().unwrap().len(), 47);
    assert_eq!(list["categories"].as_array().unwrap().len(), 6);
    // Param values clamp to Photoshop's ranges; choices accept names.
    let e = effect_from_json(GalleryFilter::GraphicPen, &json!({"strokeLength": 99, "strokeDirection": "vertical"}), &Value::Null);
    assert_eq!(e.get("strokeLength"), 15.0);
    assert_eq!(e.choice("strokeDirection"), "vertical");
}

#[test]
fn sketch_filters_record_and_use_the_current_colours() {
    let mut s = session(8, "rgb");
    s.tools.foreground = [1.0, 0.0, 0.0, 1.0];
    s.tools.background = [0.0, 0.0, 1.0, 1.0];
    let r = s.execute("filter.gallery.stamp", json!({})).unwrap();
    assert_eq!(r["filter"]["effects"][0]["foreground"], json!([1.0, 0.0, 0.0, 1.0]));
    // Stamp is two-tone: every pixel is (close to) red or blue.
    let px = pixels(&s);
    assert!(px.chunks(4).all(|c| c[1] < 0.05), "no green in a red/blue stamp");
    assert!(px.chunks(4).any(|c| c[0] > 0.9) && px.chunks(4).any(|c| c[2] > 0.9));
}

#[test]
fn gallery_is_a_smart_filter_on_smart_objects() {
    let mut s = session(8, "rgb");
    s.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap();
    let r = s.execute("filter.filterGallery", json!({"effects": [{"filter": "texturizer", "params": {"relief": 20}}]}));
    r.unwrap();
    let d = s.active().unwrap();
    let photocraft_doc::LayerContent::Smart(sm) = &d.doc.layer(d.active_layer.unwrap()).unwrap().content else { panic!("not smart") };
    assert_eq!(sm.smart_filters.len(), 1);
    assert_eq!(sm.smart_filters[0].command, "filter.filterGallery");
    assert_eq!(sm.smart_filters[0].params["effects"][0]["filter"], json!("texturizer"));
}

#[test]
fn unknown_choice_names_are_bad_params_without_editing() {
    let mut s = session(8, "rgb");
    let before = pixels(&s);
    let history = s.active().unwrap().history.entries();
    let mut checked = 0;
    for f in GalleryFilter::ALL.iter().copied() {
        for prm in f.params() {
            let GalleryParamKind::Choice(names) = prm.kind else { continue };
            let id = f.command_id();
            for v in [json!("zzbogus"), json!(true), json!([0])] {
                let err = s.execute(id, json!({prm.key: v})).unwrap_err();
                assert!(matches!(err, EngineError::BadParams { .. }), "{id}.{}: {err}", prm.key);
                assert!(err.to_string().contains(prm.key), "{err}");
                let stack = json!({"effects": [{"filter": f.key(), "params": {prm.key: v}}]});
                assert!(matches!(s.execute("filter.filterGallery", stack), Err(EngineError::BadParams { .. })), "{id}.{}", prm.key);
            }
            assert_eq!(pixels(&s), before);
            assert_eq!(s.active().unwrap().history.entries(), history);
            // Every listed name (any case) and an index still run.
            for v in names.iter().map(|n| json!(n.to_uppercase())).chain([json!(names.len() - 1)]) {
                s.execute(id, json!({prm.key: v})).unwrap();
                s.undo();
            }
            checked += 1;
        }
    }
    assert!(checked >= 16, "{checked}");
}
