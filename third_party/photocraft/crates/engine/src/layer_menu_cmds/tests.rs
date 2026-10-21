use super::*;

const DEPTHS: [u64; 3] = [8, 16, 32];

fn session(depth: u64, mode: &str) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 40, "height": 30, "depth": depth, "mode": mode, "background": "transparent"})).unwrap();
    s
}

fn doc(s: &Session) -> &Document {
    &s.active().unwrap().doc
}

fn active(s: &Session) -> &Layer {
    let d = s.active().unwrap();
    d.doc.layer(d.active_layer.unwrap()).unwrap()
}

fn paint(s: &mut Session, f: impl Fn(i32, i32) -> [f32; 4]) {
    s.edit("setup", |doc, active| {
        let b = doc.bounds();
        let surf = doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap();
        let fmt = surf.format();
        let mut data = Vec::new();
        for y in b.y0..b.y1 {
            for x in b.x0..b.x1 {
                data.extend(photocraft_raster::from_rgba(&fmt, f(x, y)));
            }
        }
        surf.write_region(b, &data);
        surf.prune();
        Ok(())
    })
    .unwrap();
}

/// A red disc with a soft, dark-contaminated edge on transparency.
fn disc(x: i32, y: i32) -> [f32; 4] {
    let d = (((x - 20) * (x - 20) + (y - 15) * (y - 15)) as f32).sqrt();
    let a = (10.0 - d).clamp(0.0, 1.0);
    // Edge pixels were matted on black: colour scaled by alpha.
    [0.9 * if a < 1.0 { a } else { 1.0 }, 0.1 * a.min(1.0), 0.1 * a.min(1.0), a]
}

fn select_left_half(s: &mut Session) {
    s.edit("sel", |doc, _| {
        let mut sel = Surface::new(photocraft_color::PixelFormat::GRAY8);
        sel.write_region(Rect::new(0, 0, 20, 30), &vec![1.0; 600]);
        doc.selection = Some(sel);
        Ok(())
    })
    .unwrap();
}

#[test]
fn mask_apply_from_transparency_hide_selection() {
    for depth in DEPTHS {
        let mut s = session(depth, "rgb");
        paint(&mut s, |_, _| [0.2, 0.4, 0.6, 1.0]);
        assert!(!s.is_enabled("layer.layerMask.apply"));
        select_left_half(&mut s);
        s.execute("layer.layerMask.hideSelection", json!({})).unwrap();
        assert!(doc(&s).selection.is_none());
        let m = active(&s).mask.as_ref().unwrap();
        assert_eq!((m.value(5, 5), m.value(30, 5)), (0.0, 1.0));
        s.execute("layer.layerMask.apply", json!({})).unwrap();
        assert!(active(&s).mask.is_none());
        let surf = active(&s).surface().unwrap();
        assert_eq!(surf.rgba(5, 5)[3], 0.0);
        assert_eq!(surf.rgba(30, 5)[3], 1.0);
        // From Transparency: alpha moves into the mask, pixels become opaque.
        s.execute("layer.layerMask.fromTransparency", json!({})).unwrap();
        let l = active(&s);
        assert_eq!(l.mask.as_ref().unwrap().value(5, 5), 0.0);
        assert_eq!(l.mask.as_ref().unwrap().value(30, 5), 1.0);
        assert_eq!(l.surface().unwrap().rgba(30, 5)[3], 1.0);
        // The composite looks the same.
        let px = photocraft_compose::render(doc(&s), Rect::from_xywh(5, 5, 1, 1)).px[0];
        assert_eq!(px[3], 0.0);
        s.execute("edit.undo", json!({})).unwrap();
        assert!(active(&s).mask.is_none());
    }
}

#[test]
fn matting_commands_fix_dark_fringes() {
    for depth in DEPTHS {
        let mut s = session(depth, "rgb");
        paint(&mut s, disc);
        let edge = (0..40).find(|&x| {
            let a = active(&s).surface().unwrap().rgba(x, 17)[3];
            a > 0.05 && a < 0.95
        });
        let ex = edge.expect("an antialiased edge pixel");
        let before = active(&s).surface().unwrap().rgba(ex, 17);
        s.execute("layer.matting.removeBlackMatte", json!({})).unwrap();
        let after = active(&s).surface().unwrap().rgba(ex, 17);
        assert!(after[0] > before[0] + 0.05, "{depth}: {before:?} → {after:?}");
        assert!((after[3] - before[3]).abs() < 1e-2, "alpha kept");
        s.execute("edit.undo", json!({})).unwrap();
        s.execute("layer.matting.defringe", json!({"width": 2})).unwrap();
        let d = active(&s).surface().unwrap().rgba(ex, 17);
        assert!(d[0] > 0.8, "{depth}: defringed edge takes the core red: {d:?}");
        s.execute("edit.undo", json!({})).unwrap();
        s.execute("layer.matting.colorDecontaminate", json!({"amount": 100})).unwrap();
        let c = active(&s).surface().unwrap().rgba(ex, 17);
        assert!(c[0] > before[0], "{depth}: {before:?} → {c:?}");
        s.execute("edit.undo", json!({})).unwrap();
        s.execute("layer.matting.removeWhiteMatte", json!({})).unwrap();
    }
}

#[test]
fn defringe_pure_function() {
    // 5×1: transparent, fringe (blue), core red ×3.
    let mut px = vec![[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.5], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]];
    defringe(&mut px, 5, 1, 1);
    assert_eq!(px[1], [1.0, 0.0, 0.0, 0.5]);
    assert_eq!(px[0][3], 0.0);
}

#[test]
fn global_light_blending_options_scale_effects() {
    let mut s = session(8, "rgb");
    paint(&mut s, disc);
    s.execute("layer.layerStyle.globalLight", json!({"angle": 200, "altitude": 45})).unwrap();
    assert_eq!((doc(&s).global_light.angle, doc(&s).global_light.altitude), (-160.0, 45.0));
    s.execute("layer.layerStyle.blendingOptions", json!({"blend": "Multiply", "opacity": 50, "fillOpacity": 25})).unwrap();
    let l = active(&s);
    assert_eq!((l.blend, l.opacity, l.fill_opacity), (BlendMode::Multiply, 0.5, 0.25));
    assert!(s.execute("layer.layerStyle.blendingOptions", json!({"blend": "nope"})).is_err());
    assert!(s.execute("layer.layerStyle.blendingOptions", json!({"blend": "Pass Through"})).is_err());
    assert!(!s.is_enabled("layer.layerStyle.scaleEffects"));
    s.execute("layer.layerStyle.dropShadow", json!({"distance": 10, "size": 4})).unwrap();
    s.execute("layer.layerStyle.scaleEffects", json!({"scale": 200})).unwrap();
    match &active(&s).effects.items[0] {
        Effect::DropShadow(sh) => assert_eq!((sh.distance, sh.size), (20.0, 8.0)),
        e => panic!("{e:?}"),
    }
}

#[test]
fn global_light_rejects_numbers_beyond_f32() {
    let mut s = session(8, "rgb");
    // In-range values still wrap into -180..=180 and clamp to 0..=90.
    s.execute("layer.layerStyle.globalLight", json!({"angle": 450, "altitude": 120})).unwrap();
    assert_eq!((doc(&s).global_light.angle, doc(&s).global_light.altitude), (90.0, 90.0));
    s.execute("layer.layerStyle.globalLight", json!({"angle": 540, "altitude": 30})).unwrap();
    assert_eq!((doc(&s).global_light.angle.abs(), doc(&s).global_light.altitude), (180.0, 30.0));
    let before = doc(&s).global_light;
    let steps = s.active().unwrap().history.past_len();
    // 1e39 overflowed f32 to inf, which rem_euclid turned into a stored NaN (#4181).
    for p in [json!({"angle": 1e39}), json!({"angle": -1e39}), json!({"altitude": 1e39}), json!({"altitude": -1e39})] {
        assert!(s.execute("layer.layerStyle.globalLight", p.clone()).is_err(), "{p}");
        assert_eq!(doc(&s).global_light, before, "{p}");
        assert_eq!(s.active().unwrap().history.past_len(), steps, "{p}");
    }
}

#[test]
fn blending_options_blend_if() {
    for depth in DEPTHS {
        let mut s = session(depth, "rgb");
        // Black on the left, white on the right.
        paint(&mut s, |x, _| if x < 20 { [0.0, 0.0, 0.0, 1.0] } else { [1.0; 4] });
        let at = |s: &Session, x: i32| photocraft_compose::flatten(doc(s)).get(x, 5);
        // Gray › This Layer: black point 50 hides the black half.
        s.execute("layer.layerStyle.blendingOptions", json!({"blendIf": {"channel": "gray", "thisLayer": [50, 255]}})).unwrap();
        assert_eq!(at(&s, 5)[3], 0.0, "depth {depth}");
        assert_eq!(at(&s, 30), [1.0; 4], "depth {depth}");
        assert_eq!(active(&s).blend_if.get(0), [BlendRange { black: [50, 50], white: [255, 255] }, BlendRange::FULL]);
        // Setting the other slider pair keeps this one; split points are 4 values.
        s.execute(
            "layer.layerStyle.blendingOptions",
            json!({"blendIf": [{"channel": "gray", "underlying": [0, 10, 245, 255]}, {"channel": "blue", "thisLayer": [0, 200]}]}),
        )
        .unwrap();
        let bi = &active(&s).blend_if;
        assert_eq!(bi.get(0)[0].black, [50, 50]);
        assert_eq!(bi.get(0)[1], BlendRange { black: [0, 10], white: [245, 255] });
        assert_eq!(bi.get(3)[0], BlendRange { black: [0, 0], white: [200, 200] });
        let ins = crate::inspect::layer(active(&s));
        assert_eq!(ins["blendIf"][0], json!({"channel": 0, "thisLayer": [50, 50, 255, 255], "underlying": [0, 10, 245, 255]}));
        assert_eq!(ins["blendIf"][1]["channel"], 3);
        // Blending options without `blendIf` (the dialog's blend/opacity/fill) leave it alone.
        s.execute("layer.layerStyle.blendingOptions", json!({"opacity": 90})).unwrap();
        assert!(!active(&s).blend_if.is_default());
        // One undo step per call.
        s.execute("edit.undo", json!({})).unwrap();
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(active(&s).blend_if.get(0)[1], BlendRange::FULL);
        assert_eq!(active(&s).blend_if.get(0)[0].black, [50, 50]);
        // null resets everything.
        s.execute("layer.layerStyle.blendingOptions", json!({"blendIf": null})).unwrap();
        assert!(active(&s).blend_if.is_default());
        assert!(crate::inspect::layer(active(&s)).get("blendIf").is_none());
        assert_eq!(at(&s, 5), [0.0, 0.0, 0.0, 1.0]);
    }
}

#[test]
fn blend_if_rejects_bad_params() {
    let mut s = session(8, "rgb");
    paint(&mut s, disc);
    let steps = s.active().unwrap().history.past_len();
    for bad in [
        json!({"channel": "cyan", "thisLayer": [10, 255]}),
        json!({"channel": 4, "thisLayer": [10, 255]}),
        json!({"channel": "gray", "thisLayer": [200, 100]}),
        json!({"channel": "gray", "thisLayer": [0, 300]}),
        json!({"channel": "gray", "thisLayer": [0, 10, 5, 255]}),
        json!({"channel": "gray", "underlying": [1, 2, 3]}),
        json!({"channel": "gray", "underlying": "dark"}),
        json!("gray"),
    ] {
        assert!(s.execute("layer.layerStyle.blendingOptions", json!({"blendIf": bad})).is_err(), "{bad}");
    }
    assert!(active(&s).blend_if.is_default());
    assert_eq!(s.active().unwrap().history.past_len(), steps);
}

#[test]
fn blending_options_advanced_blending() {
    use photocraft_doc::{AdvancedBlending, Knockout};
    for depth in DEPTHS {
        let mut s = session(depth, "rgb");
        paint(&mut s, |x, _| if x < 20 { [0.0, 0.0, 1.0, 1.0] } else { [0.0; 4] });
        assert!(active(&s).advanced.is_default());
        s.execute(
            "layer.layerStyle.blendingOptions",
            json!({
                "knockout": "deep",
                "fillOpacity": 0,
                "blendInteriorEffectsAsGroup": true,
                "blendClippedLayersAsGroup": false,
                "transparencyShapesLayer": false,
                "layerMaskHidesEffects": true,
                "vectorMaskHidesEffects": true,
                "channels": [true, false, true],
            }),
        )
        .unwrap();
        let l = active(&s);
        assert_eq!(
            l.advanced,
            AdvancedBlending {
                knockout: Knockout::Deep,
                blend_interior: true,
                blend_clipped: false,
                transparency_shapes: false,
                layer_mask_hides_effects: true,
                vector_mask_hides_effects: true,
            }
        );
        assert_eq!(l.excluded_channels, 0b010);
        // A deep knockout at fill 0 with no Background reveals transparency where the layer has
        // pixels.
        let p = photocraft_compose::flatten(doc(&s)).get(5, 5);
        assert!(p[3] < 2.0 / 255.0, "{depth}: {p:?}");
        // Inspect reports the switches; omitted keys keep their values; one undo step each.
        let info = crate::inspect::layer(active(&s));
        assert_eq!(info["advancedBlending"]["knockout"], json!("deep"), "{info}");
        s.execute("layer.layerStyle.blendingOptions", json!({"knockout": "Shallow"})).unwrap();
        assert_eq!(active(&s).advanced.knockout, Knockout::Shallow);
        assert!(!active(&s).advanced.blend_clipped);
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(active(&s).advanced.knockout, Knockout::Deep);
        s.execute("edit.undo", json!({})).unwrap();
        assert!(active(&s).advanced.is_default());
        assert_eq!(active(&s).excluded_channels, 0);
    }
}

#[test]
fn blending_options_advanced_rejects_bad_params() {
    let mut s = session(8, "rgb");
    paint(&mut s, |_, _| [1.0; 4]);
    let steps = s.active().unwrap().history.past_len();
    for bad in [
        json!({"knockout": "medium"}),
        json!({"knockout": 2}),
        json!({"knockout": true}),
        json!({"blendInteriorEffectsAsGroup": "yes"}),
        json!({"blendClippedLayersAsGroup": 1}),
        json!({"transparencyShapesLayer": []}),
        json!({"layerMaskHidesEffects": {}}),
        json!({"vectorMaskHidesEffects": 0.5}),
        json!({"channels": [true, true, true, true, true]}),
        json!({"channels": [1, 0, 1]}),
        json!({"channels": "rgb"}),
        // A bad key fails the whole call: nothing else is applied.
        json!({"fillOpacity": 10, "knockout": "nope"}),
    ] {
        assert!(s.execute("layer.layerStyle.blendingOptions", bad.clone()).is_err(), "{bad}");
    }
    assert!(active(&s).advanced.is_default());
    assert_eq!(active(&s).fill_opacity, 1.0);
    assert_eq!(s.active().unwrap().history.past_len(), steps);
    // `null` keeps the current value.
    s.execute("layer.layerStyle.blendingOptions", json!({"knockout": null, "channels": null})).unwrap();
    assert!(active(&s).advanced.is_default());
}

#[test]
fn copy_paste_layer_style_carries_advanced_blending() {
    let mut s = session(8, "rgb");
    paint(&mut s, |_, _| [1.0; 4]);
    s.execute("layer.layerStyle.blendingOptions", json!({"knockout": "shallow", "blendClippedLayersAsGroup": false})).unwrap();
    let a = active(&s).advanced;
    s.execute("layer.layerStyle.copyLayerStyle", json!({})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    assert!(active(&s).advanced.is_default());
    s.execute("layer.layerStyle.pasteLayerStyle", json!({})).unwrap();
    assert_eq!(active(&s).advanced, a);
}

#[test]
fn blend_if_channel_names_follow_the_mode() {
    let mut s = session(8, "cmyk");
    s.execute("layer.layerStyle.blendingOptions", json!({"blendIf": {"channel": "black", "underlying": [0, 128]}})).unwrap();
    assert_eq!(active(&s).blend_if.get(4)[1].white, [128, 128]);
    assert!(s.execute("layer.layerStyle.blendingOptions", json!({"blendIf": {"channel": "red", "underlying": [0, 128]}})).is_err());
    // A grayscale document's Gray is its one channel.
    let mut s = session(16, "grayscale");
    s.execute("layer.layerStyle.blendingOptions", json!({"blendIf": {"channel": "gray", "thisLayer": [30, 255]}})).unwrap();
    assert_eq!(active(&s).blend_if.get(1)[0].black, [30, 30]);
    assert_eq!(active(&s).blend_if.get(0), [BlendRange::FULL; 2]);
}

#[test]
fn create_layer_splits_effects_and_keeps_the_look() {
    let mut s = session(8, "rgb");
    s.execute("layer.new.layer", json!({})).unwrap();
    paint(&mut s, |x, y| if (10..30).contains(&x) && (8..22).contains(&y) { [0.2, 0.5, 0.9, 1.0] } else { [0.0; 4] });
    s.execute("layer.layerStyle.dropShadow", json!({"blend": "normal", "distance": 4, "size": 2, "opacity": 60})).unwrap();
    s.execute("layer.layerStyle.colorOverlay", json!({"color": "#ff8000", "opacity": 50, "add": true})).unwrap();
    let id = active(&s).id;
    let before = photocraft_compose::flatten(doc(&s));
    let n0 = doc(&s).layers.len();
    let r = s.execute("layer.layerStyle.createLayer", json!({})).unwrap();
    assert_eq!(r["layers"], json!(2));
    let d = doc(&s);
    assert_eq!(d.layers.len(), n0 + 2);
    let i = d.layers.iter().position(|l| l.id == id).unwrap();
    assert!(d.layers[i - 1].name.ends_with("Drop Shadow") && !d.layers[i - 1].clipped);
    assert!(d.layers[i + 1].name.ends_with("Color Overlay") && d.layers[i + 1].clipped);
    assert!(d.layers[i].effects.items.is_empty());
    let after = photocraft_compose::flatten(d);
    let worst = before.px.iter().zip(&after.px).map(|(a, b)| (0..4).map(|c| (a[c] - b[c]).abs()).fold(0.0, f32::max)).fold(0.0, f32::max);
    assert!(worst < 3.0 / 255.0, "composite changed by {}", worst * 255.0);
    assert!(!s.is_enabled("layer.layerStyle.createLayer"));
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(doc(&s).layers.len(), n0);
}

#[test]
fn content_options_reports_the_editor() {
    let mut s = session(8, "rgb");
    assert!(!s.is_enabled("layer.layerContentOptions"));
    s.execute("layer.newAdjustmentLayer.levels", json!({})).unwrap();
    assert_eq!(s.execute("layer.layerContentOptions", json!({})).unwrap()["editor"], json!("adjustment:levels"));
    s.execute("layer.newFillLayer.solidColor", json!({})).unwrap();
    assert_eq!(s.execute("layer.layerContentOptions", json!({})).unwrap()["editor"], json!("fill"));
}

#[test]
fn export_layer_trims_to_its_pixels() {
    let mut s = session(16, "rgb");
    paint(&mut s, |x, y| if (5..15).contains(&x) && (4..10).contains(&y) { [1.0, 0.0, 0.0, 1.0] } else { [0.0; 4] });
    let d = layer_document(doc(&s), active(&s).id).unwrap();
    assert_eq!((d.size.width, d.size.height), (10, 6));
    assert_eq!(d.layers[0].surface().unwrap().rgba(0, 0), [1.0, 0.0, 0.0, 1.0]);
    let dir = std::env::temp_dir().join(format!("pc-layer-export-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("layer.png");
    let r = s.execute("layer.quickExportAsPng", json!({"path": path.to_string_lossy()})).unwrap();
    assert_eq!((r["width"].as_u64(), r["height"].as_u64()), (Some(10), Some(6)));
    assert!(std::fs::read(&path).unwrap().starts_with(&[0x89, b'P', b'N', b'G']));
    let jpg = dir.join("layer.jpg");
    let r = s.execute("layer.exportAs", json!({"path": jpg.to_string_lossy(), "scale": 200})).unwrap();
    assert_eq!(r["width"].as_u64(), Some(20));
    assert!(s.execute("layer.quickExportAsPng", json!({"path": jpg.to_string_lossy()})).is_err());
    assert!(s.execute("layer.exportAs", json!({})).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn stack_modes_render_from_the_nested_layers() {
    let mut s = session(8, "rgb");
    // Three frames: the same gray, one with a bright outlier pixel.
    s.edit("frames", |doc, active| {
        let fmt = doc.pixel_format();
        let mut kids = Vec::new();
        for k in 0..3 {
            let mut l = Layer::raster(format!("f{k}"), fmt);
            let v = if k == 2 { 1.0 } else { 0.4 };
            let surf = l.surface_mut().unwrap();
            surf.write_region(Rect::new(0, 0, 40, 30), &photocraft_raster::from_rgba(&fmt, [0.4, 0.4, 0.4, 1.0]).repeat(1200));
            surf.write_region(Rect::new(3, 3, 4, 4), &photocraft_raster::from_rgba(&fmt, [v, v, v, 1.0]));
            kids.push(l);
        }
        let g = Layer::new("stack", LayerContent::Group(photocraft_doc::Group { children: kids, expanded: true, artboard: None }));
        let smart = crate::smart_cmds::layer_to_smart(doc, &g)?;
        let id = smart.id;
        doc.layers = vec![smart];
        *active = Some(id);
        Ok(())
    })
    .unwrap();
    let px = |s: &Session| active(s).content.clone();
    let at = |s: &Session, x: i32, y: i32| match px(s) {
        LayerContent::Smart(sm) => sm.cache.unwrap().rgba(x, y),
        _ => panic!(),
    };
    assert!(at(&s, 3, 3)[0] > 0.9, "normal composite shows the top frame");
    s.execute("layer.smartObjects.stackMode.median", json!({})).unwrap();
    assert!((at(&s, 3, 3)[0] - 0.4).abs() < 2.0 / 255.0, "median removes the outlier");
    s.execute("layer.smartObjects.stackMode.maximum", json!({})).unwrap();
    assert!(at(&s, 3, 3)[0] > 0.99);
    s.execute("layer.smartObjects.stackMode.range", json!({})).unwrap();
    assert!(at(&s, 10, 10)[0] < 1.0 / 255.0);
    for m in StackMode::ALL {
        s.execute(&format!("layer.smartObjects.stackMode.{}", m.id()), json!({})).unwrap();
        match &active(&s).content {
            LayerContent::Smart(sm) => assert_eq!(sm.stack_mode, Some(m)),
            _ => panic!(),
        }
    }
    s.execute("layer.smartObjects.stackMode.none", json!({})).unwrap();
    assert!(at(&s, 3, 3)[0] > 0.9);
    // The stack mode survives .pcraft.
    s.execute("layer.smartObjects.stackMode.mean", json!({})).unwrap();
    let bytes = photocraft_format::save_to_bytes(doc(&s), &Default::default()).unwrap();
    let back = photocraft_format::load_from_bytes(&bytes).unwrap();
    match &back.layers[0].content {
        LayerContent::Smart(sm) => assert_eq!(sm.stack_mode, Some(StackMode::Mean)),
        _ => panic!(),
    }
    assert!(!session(8, "rgb").is_enabled("layer.smartObjects.stackMode.mean"));
}

#[test]
fn stack_mode_rejects_sources_without_visible_layers() {
    for depth in DEPTHS {
        let mut s = session(depth, "rgb");
        paint(&mut s, |_, _| [0.4, 0.4, 0.4, 1.0]);
        s.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap();
        s.execute("layer.smartObjects.editContents", json!({})).unwrap();
        let ids: Vec<u64> = doc(&s).layers.iter().map(|layer| layer.id.0).collect();
        for id in ids {
            s.execute("layer.hideLayers", json!({"layer": id})).unwrap();
        }
        s.execute("layer.smartObjects.saveContents", json!({})).unwrap();
        s.execute("file.close", json!({})).unwrap();

        let err = s.execute("layer.smartObjects.stackMode.mean", json!({})).unwrap_err();
        assert!(err.to_string().contains("no visible layers"), "{err}");
        match &active(&s).content {
            LayerContent::Smart(sm) => assert_eq!(sm.stack_mode, None, "failed edit must roll back"),
            _ => panic!(),
        }
    }
}

#[test]
fn reveal_in_finder_dry_run_and_enabled() {
    let mut s = session(8, "rgb");
    assert!(!s.is_enabled("layer.smartObjects.revealInFinder"));
    s.edit("linked", |doc, active| {
        let mut l = Layer::new(
            "linked",
            LayerContent::Smart(photocraft_doc::SmartObject::new(
                SmartSource::Linked { path: "/tmp/pics/a.png".into() },
                photocraft_geom::Affine::IDENTITY,
                None,
            )),
        );
        l.visible = true;
        *active = Some(doc.insert_above(*active, l));
        Ok(())
    })
    .unwrap();
    assert!(s.is_enabled("layer.smartObjects.revealInFinder"));
    let r = s.execute("layer.smartObjects.revealInFinder", json!({"dryRun": true})).unwrap();
    let (prog, args) = reveal_command("/tmp/pics/a.png");
    assert_eq!(r["program"], json!(prog));
    assert_eq!(r["args"], json!(args));
}

#[test]
fn mask_all_objects_masks_the_subject() {
    let mut s = session(8, "rgb");
    paint(&mut s, |x, y| {
        let inside = (((x - 20) * (x - 20) + (y - 15) * (y - 15)) as f32).sqrt() < 8.0;
        if inside { [0.9, 0.2, 0.1, 1.0] } else { [0.95, 0.95, 0.95, 1.0] }
    });
    match s.execute("layer.maskAllObjects", json!({})) {
        Ok(_) => {
            let m = active(&s).mask.as_ref().unwrap();
            assert!(m.value(20, 15) > m.value(1, 1), "subject revealed, background hidden");
        }
        // The segmentation may decline tiny images; then nothing changes.
        Err(_) => assert!(active(&s).mask.is_none()),
    }
}

#[test]
fn layer_masks_in_gray_cmyk_lab() {
    for mode in ["gray", "cmyk", "lab"] {
        let mut s = session(16, mode);
        paint(&mut s, disc);
        s.execute("layer.layerMask.fromTransparency", json!({})).unwrap();
        s.execute("layer.layerMask.apply", json!({})).unwrap();
        s.execute("layer.matting.defringe", json!({"width": 1})).unwrap();
        s.execute("layer.matting.removeBlackMatte", json!({})).unwrap();
        assert!(active(&s).surface().unwrap().rgba(20, 15)[3] > 0.99, "{mode}");
    }
}

#[test]
fn load_files_into_stack_as_smart_object_then_median() {
    let dir = std::env::temp_dir().join(format!("pc-stack-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut paths = Vec::new();
    for (k, v) in [0.2f32, 0.2, 0.9].iter().enumerate() {
        let d = Document::with_background(
            "f",
            photocraft_doc::Size::new(6, 4),
            photocraft_color::ColorMode::Rgb,
            photocraft_color::SampleType::U8,
            photocraft_doc::Color::rgba(*v, *v, *v, 1.0),
        );
        let p = dir.join(format!("f{k}.png"));
        std::fs::write(&p, photocraft_io::export(&d, "f.png", &Default::default()).unwrap().bytes).unwrap();
        paths.push(p.to_string_lossy().into_owned());
    }
    let mut s = Session::new();
    s.execute("file.scripts.loadFilesIntoStack", json!({"paths": paths, "createSmartObject": true})).unwrap();
    assert_eq!(doc(&s).layers.len(), 1);
    assert!(matches!(doc(&s).layers[0].content, LayerContent::Smart(_)));
    s.execute("layer.smartObjects.stackMode.median", json!({})).unwrap();
    let px = photocraft_compose::render(doc(&s), Rect::from_xywh(2, 2, 1, 1)).px[0];
    assert!((px[0] - 0.2).abs() < 2.0 / 255.0, "{px:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn transfer_effects_moves_or_copies_between_layers() {
    let mut s = session(8, "rgb");
    s.execute("layer.new.layer", json!({})).unwrap();
    let a = active(&s).id;
    s.execute("layer.layerStyle.dropShadow", json!({})).unwrap();
    s.execute("layer.layerStyle.colorOverlay", json!({"color": "#ff8000"})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    let b = active(&s).id;
    let layer = |s: &Session, id| s.active().unwrap().doc.layer(id).unwrap().clone();
    let labels = |l: &Layer| l.effects.items.iter().map(Effect::label).collect::<Vec<_>>();
    let both = labels(&layer(&s, a));
    assert_eq!(both.len(), 2);

    // Alt-drag of the whole fx row copies: both layers have the style.
    s.execute("layer.layerStyle.transferEffects", json!({"from": a.0, "to": b.0, "copy": true})).unwrap();
    assert_eq!(labels(&layer(&s, a)), both);
    assert_eq!(labels(&layer(&s, b)), both);

    // Dragging one effect row moves just that effect.
    s.execute("layer.new.layer", json!({})).unwrap();
    let c = active(&s).id;
    s.execute("layer.layerStyle.transferEffects", json!({"from": a.0, "to": c.0, "effect": 1})).unwrap();
    assert_eq!(labels(&layer(&s, a)), both[..1]);
    assert_eq!(labels(&layer(&s, c)), both[1..]);

    // Dragging the whole fx row moves everything and leaves the source without a style.
    s.execute("layer.layerStyle.transferEffects", json!({"from": b.0, "to": a.0})).unwrap();
    assert!(layer(&s, b).effects.items.is_empty());
    assert_eq!(labels(&layer(&s, a)), both);

    // One history step per drop.
    let steps = s.active().unwrap().history.past_len();
    s.execute("layer.layerStyle.transferEffects", json!({"from": a.0, "to": b.0})).unwrap();
    assert_eq!(s.active().unwrap().history.past_len(), steps + 1);
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(labels(&layer(&s, a)), both);
    assert!(layer(&s, b).effects.items.is_empty());

    // Bad input is an error and changes nothing.
    for bad in [
        json!({"from": a.0, "to": a.0}),
        json!({"from": a.0}),
        json!({"from": a.0, "to": u64::MAX}),
        json!({"from": a.0, "to": b.0, "effect": 7}),
        json!({"from": a.0, "to": b.0, "effect": -1}),
        json!({"from": b.0, "to": a.0}),
    ] {
        assert!(s.execute("layer.layerStyle.transferEffects", bad.clone()).is_err(), "{bad}");
    }
    assert_eq!(s.active().unwrap().history.past_len(), steps);
}

#[test]
fn applying_a_feathered_mask_preserves_its_soft_composite() {
    for depth in DEPTHS {
        let mut s = session(depth, "rgb");
        paint(&mut s, |_, _| [1.0, 0.0, 0.0, 1.0]);
        s.edit("mask fixture", |doc, active| {
            let mut mask = LayerMask::hide_all();
            mask.surface.fill_rect(Rect::new(0, 0, 20, 30), &[1.0]);
            mask.feather = 2.0;
            doc.layer_mut(active.unwrap()).unwrap().mask = Some(mask);
            Ok(())
        })
        .unwrap();
        let before = photocraft_compose::flatten(doc(&s)).px;
        assert!(before[15 * 40 + 19][3] > 0.0 && before[15 * 40 + 19][3] < 1.0, "mask must start with a soft edge");
        s.execute("layer.layerMask.apply", json!({})).unwrap();
        assert!(active(&s).mask.is_none(), "Apply removes the pixel mask");
        let after = photocraft_compose::flatten(doc(&s)).px;
        for (a, b) in before.iter().zip(&after) {
            assert!((a[3] - b[3]).abs() <= 1.0 / 255.0 + 1e-5, "depth {depth}: alpha {a:?} became {b:?}");
        }
        assert!(s.undo());
        assert!(active(&s).mask.as_ref().is_some_and(|m| m.feather == 2.0));
    }
}

#[test]
fn applying_a_disabled_feathered_mask_does_not_hide_pixels() {
    let mut s = session(8, "rgb");
    paint(&mut s, |_, _| [0.7, 0.2, 0.1, 1.0]);
    s.edit("disabled mask", |doc, active| {
        let mut mask = LayerMask::hide_all();
        mask.feather = 3.0;
        mask.enabled = false;
        doc.layer_mut(active.unwrap()).unwrap().mask = Some(mask);
        Ok(())
    })
    .unwrap();
    s.execute("layer.layerMask.apply", json!({})).unwrap();
    assert_eq!(active(&s).surface().unwrap().rgba(20, 15)[3], 1.0);
}

#[test]
fn from_transparency_preserves_mask_coverage_outside_the_canvas() {
    for depth in DEPTHS {
        let mut s = session(depth, "rgb");
        paint(&mut s, |_, _| [1.0, 0.0, 0.0, 0.5]);
        let before = photocraft_compose::flatten(doc(&s)).px;
        s.execute("layer.translate", json!({"dx": 20})).unwrap();
        s.execute("layer.layerMask.fromTransparency", json!({})).unwrap();
        let mask = active(&s).mask.as_ref().unwrap();
        assert!((mask.value(50, 10) - 0.5).abs() < 1.0 / 255.0 + 1e-6, "off-canvas mask at depth {depth}");
        s.execute("layer.translate", json!({"dx": -20})).unwrap();
        let after = photocraft_compose::flatten(doc(&s)).px;
        for (a, b) in before.iter().zip(&after) {
            assert!((a[3] - b[3]).abs() <= 1.0 / 255.0 + 1e-6, "alpha changed at depth {depth}: {a:?} -> {b:?}");
        }
    }
}

#[test]
fn applying_a_feathered_mask_keeps_off_canvas_pixels_it_reveals() {
    let mut s = session(8, "rgb");
    paint(&mut s, |_, _| [1.0, 0.0, 0.0, 1.0]);
    s.execute("layer.translate", json!({"dx": 20})).unwrap();
    s.edit("mask fixture", |doc, active| {
        // Reveal the whole layer, including the 20 px now past the right edge.
        let mut mask = LayerMask::hide_all();
        mask.surface.fill_rect(Rect::new(0, 0, 60, 30), &[1.0]);
        mask.feather = 2.0;
        doc.layer_mut(active.unwrap()).unwrap().mask = Some(mask);
        Ok(())
    })
    .unwrap();
    s.execute("layer.layerMask.apply", json!({})).unwrap();
    assert_eq!(active(&s).surface().unwrap().rgba(52, 15)[3], 1.0, "off-canvas pixel stays visible");
}
