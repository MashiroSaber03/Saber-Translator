use super::*;
use photocraft_geom::Rect;

fn session(w: u32, h: u32, depth: u64, mode: &str) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": w, "height": h, "depth": depth, "mode": mode})).unwrap();
    paint(&mut s, |x, y| [x as f32 / w as f32, y as f32 / h as f32, 0.5, 1.0]);
    s
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
        Ok(())
    })
    .unwrap();
}

fn doc(s: &Session) -> &Document {
    &s.active().unwrap().doc
}

fn colours(d: &Document) -> std::collections::HashSet<[u8; 3]> {
    let surf = d.layers[0].surface().unwrap();
    let mut set = std::collections::HashSet::new();
    for y in 0..d.size.height as i32 {
        for x in 0..d.size.width as i32 {
            let c = surf.rgba(x, y);
            if c[3] > 0.0 {
                set.insert([c[0], c[1], c[2]].map(|v| (v * 255.0).round() as u8));
            }
        }
    }
    set
}

#[test]
fn rotate_arbitrary_grows_canvas_and_keeps_background() {
    for depth in [8, 16, 32] {
        let mut s = session(40, 20, depth, "rgb");
        s.execute("layer.new.layer", json!({})).unwrap();
        paint(&mut s, |x, y| if (15..25).contains(&x) && (5..15).contains(&y) { [1.0, 0.0, 0.0, 1.0] } else { [0.0; 4] });
        s.execute("tools.setColors", json!({"background": "#00ff00"})).unwrap();
        let r = s.execute("image.rotation.arbitrary", json!({"angle": 90, "direction": "cw"})).unwrap();
        assert_eq!((r["width"].as_u64(), r["height"].as_u64()), (Some(20), Some(40)));
        let d = doc(&s);
        assert_eq!(d.layers[0].name, "Background");
        // The red square stays centred.
        assert!(d.layers[1].surface().unwrap().rgba(10, 20)[0] > 0.9, "{depth}");
        s.execute("edit.undo", json!({})).unwrap();
        s.execute("image.rotation.arbitrary", json!({"angle": 30, "direction": "ccw"})).unwrap();
        let d = doc(&s);
        assert_eq!(d.size, rotated_size(Size::new(40, 20), 30.0));
        assert!(d.size.width > 40 && d.size.height > 20);
        // Exposed corners take the background colour on the Background layer.
        let corner = d.layers[0].surface().unwrap().rgba(0, 0);
        assert!(corner[1] > 0.9 && corner[0] < 0.1, "{corner:?}");
    }
    let mut s = session(10, 10, 8, "rgb");
    assert!(s.execute("image.rotation.arbitrary", json!({"angle": 10, "direction": "up"})).is_err());
    let n = s.active().unwrap().history.past_len();
    s.execute("image.rotation.arbitrary", json!({"angle": 360})).unwrap();
    assert_eq!(s.active().unwrap().history.past_len(), n, "a full turn changes nothing");
}

#[test]
fn rotate_arbitrary_other_modes() {
    for mode in ["gray", "cmyk", "lab"] {
        let mut s = session(16, 12, 16, mode);
        s.execute("image.rotation.arbitrary", json!({"angle": 45})).unwrap();
        assert_eq!(doc(&s).size, rotated_size(Size::new(16, 12), 45.0), "{mode}");
    }
}

#[test]
fn rotate_arbitrary_rotates_grouped_locked_children_and_preserves_lock() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 16, "height": 16, "background": "white"})).unwrap();
    s.execute("layer.new.layer", json!({"name": "Paint"})).unwrap();
    s.execute("edit.fill", json!({"color": "#0000ff"})).unwrap();
    s.execute("layer.setProps", json!({"locks": {"position": true}})).unwrap();
    s.execute("layer.groupLayers", json!({})).unwrap();

    // 1. 30 degree rotation succeeds and canvas grows to 22x22.
    let r30 = s.execute("image.rotation.arbitrary", json!({"angle": 30, "interpolation": "nearest"})).unwrap();
    assert_eq!((r30["width"].as_u64(), r30["height"].as_u64()), (Some(22), Some(22)));

    // 2. Child remains position-locked after canvas rotation.
    let d = doc(&s);
    let group = &d.layers[1];
    let photocraft_doc::LayerContent::Group(g) = &group.content else { panic!("expected group") };
    assert_eq!(g.children.len(), 1);
    let child = &g.children[0];
    assert_eq!(child.name, "Paint");
    assert!(child.locks.position, "position lock must be preserved on child");

    // 3. Free Transform of the child alone still refuses with lock error.
    let child_id = child.id.0;
    s.execute("layer.select", json!({"layer": child_id})).unwrap();
    let err = s.execute("edit.transform", json!({"matrix": [1, 0, 0, 1, 0, 0]})).unwrap_err();
    assert!(err.to_string().contains("locked"), "child Free Transform must remain refused");

    // 4. Undo restores 16x16 size.
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(doc(&s).size, Size::new(16, 16));

    // 5. 90 degree rotation also succeeds and composite matches dedicated 90cw turn.
    let r90 = s.execute("image.rotation.arbitrary", json!({"angle": 90, "direction": "cw", "interpolation": "nearest"})).unwrap();
    assert_eq!((r90["width"].as_u64(), r90["height"].as_u64()), (Some(16), Some(16)));
    let d_arb = doc(&s).clone();
    s.execute("edit.undo", json!({})).unwrap();
    s.execute("image.imageRotation.90cw", json!({})).unwrap();
    let d_cw = doc(&s);
    assert_eq!(photocraft_compose::flatten(&d_arb).px, photocraft_compose::flatten(d_cw).px);
}

#[test]
fn flattened_mode_pixels_match_the_allocating_conversion_at_every_channel_count() {
    let px = [[0.0, 0.25, 0.75, 0.0], [0.9, 0.3, 0.2, 0.2], [0.1, 0.7, 0.4, 0.7], [1.0; 4]];
    for mode in [ColorMode::Indexed, ColorMode::Bitmap, ColorMode::Duotone] {
        let mut document = Document::with_background("mode", Size::new(2, 2), ColorMode::Rgb, SampleType::F32, photocraft_doc::Color::WHITE);
        let mut active = None;
        single_layer(&mut document, &mut active, mode, &px, "result", false);
        let fmt = document.pixel_format();
        let mut expected = Surface::new(fmt);
        let converted: Vec<f32> = px.iter().flat_map(|q| photocraft_raster::from_rgba(&fmt, *q)).collect();
        expected.write_region(document.bounds(), &converted);
        let actual = document.layers[0].surface().unwrap();
        assert_eq!(actual.to_interleaved(document.bounds()), expected.to_interleaved(document.bounds()), "{mode:?}");
        assert_eq!(document.depth, SampleType::U8);
        assert_eq!(active, Some(document.layers[0].id));
    }
}

#[test]
fn indexed_color_rejects_invalid_counts_without_editing_the_document() {
    let mut s = session(12, 12, 16, "rgb");
    let before = s.active().unwrap().history.past_len();
    for colors in [json!(16.5), json!(0), json!(1), json!(257), json!(512), json!(-2), json!("16"), json!(null), json!(true), json!([]), json!(1e100)] {
        assert!(s.execute("image.mode.indexedColor", json!({"colors": colors})).is_err(), "{colors}");
        assert_eq!(s.active().unwrap().history.past_len(), before);
        assert_eq!((doc(&s).mode, doc(&s).depth), (ColorMode::Rgb, SampleType::U16));
    }
    s.execute("image.mode.indexedColor", json!({"colors": 17.0})).unwrap();
    assert!(doc(&s).color_table.as_ref().unwrap().colors.len() <= 17);
}

#[test]
fn indexed_color_palettes_and_dithers() {
    for palette in ["selective", "perceptual", "adaptive", "web", "uniform", "systemMac", "systemWindows"] {
        for dither in ["none", "diffusion", "pattern", "noise"] {
            let mut s = session(24, 16, 16, "rgb");
            let r = s.execute("image.mode.indexedColor", json!({"palette": palette, "colors": 16, "dither": dither, "forced": "none"})).unwrap();
            let d = doc(&s);
            assert_eq!((d.mode, d.depth), (ColorMode::Indexed, SampleType::U8), "{palette} {dither}");
            let table = d.color_table.as_ref().unwrap();
            assert_eq!(table.colors.len() as u64, r["colors"].as_u64().unwrap());
            let used = colours(d);
            assert!(used.iter().all(|c| table.colors.contains(c)), "{palette} {dither}: pixels off the table");
            if !matches!(palette, "web" | "systemMac" | "systemWindows") {
                assert!(table.colors.len() <= 16, "{palette}: {}", table.colors.len());
            }
        }
    }
}

#[test]
fn indexed_exact_transparency_and_sources() {
    // Two colours + transparency: exact palette with a transparent entry.
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 8, "height": 8, "background": "transparent"})).unwrap();
    paint(&mut s, |x, _| {
        if x < 3 {
            [1.0, 0.0, 0.0, 1.0]
        } else if x < 6 {
            [0.0, 0.0, 1.0, 1.0]
        } else {
            [0.0; 4]
        }
    });
    s.execute("image.mode.indexedColor", json!({"palette": "exact", "forced": "none"})).unwrap();
    let d = doc(&s);
    let t = d.color_table.as_ref().unwrap();
    assert_eq!(t.colors.len(), 3);
    assert_eq!(t.transparent, Some(2));
    assert_eq!(d.layers[0].name, "Index");
    assert_eq!(d.layers[0].surface().unwrap().rgba(7, 0)[3], 0.0);
    // Exact fails on a gradient.
    let mut s = session(64, 64, 8, "rgb");
    assert!(s.execute("image.mode.indexedColor", json!({"palette": "exact"})).is_err());
    // Gray, CMYK and Lab sources flatten through the composite.
    for mode in ["gray", "cmyk", "lab"] {
        let mut s = session(12, 12, 16, mode);
        s.execute("image.mode.indexedColor", json!({"colors": 8})).unwrap();
        assert_eq!(doc(&s).mode, ColorMode::Indexed, "{mode}");
    }
    // Back to RGB drops the table.
    s.execute("image.mode.indexedColor", json!({"colors": 8})).unwrap();
    s.execute("image.mode.rgb", json!({})).unwrap();
    assert!(doc(&s).color_table.is_none());
    s.execute("edit.undo", json!({})).unwrap();
    assert!(doc(&s).color_table.is_some());
}

#[test]
fn color_table_presets_and_entries() {
    let mut s = session(16, 16, 8, "rgb");
    assert!(!s.is_enabled("image.mode.colorTable"));
    s.execute("image.mode.indexedColor", json!({"palette": "uniform", "colors": 8, "dither": "none", "forced": "none"})).unwrap();
    let before = doc(&s).color_table.clone().unwrap();
    // No changes: a report, no history step.
    let n = s.active().unwrap().history.past_len();
    let r = s.execute("image.mode.colorTable", json!({})).unwrap();
    assert_eq!(r["colors"].as_array().unwrap().len(), before.colors.len());
    assert_eq!(s.active().unwrap().history.past_len(), n);
    // Replace entry 0: pixels of index 0 change colour.
    let px0 = doc(&s).layers[0].surface().unwrap().rgba(0, 0);
    let i0 = before.nearest([px0[0], px0[1], px0[2]]);
    s.execute("image.mode.colorTable", json!({"entries": {i0.to_string(): "#ff00ff"}})).unwrap();
    assert_eq!(doc(&s).layers[0].surface().unwrap().rgba(0, 0), [1.0, 0.0, 1.0, 1.0]);
    s.execute("image.mode.colorTable", json!({"table": "blackBody"})).unwrap();
    assert_eq!(doc(&s).color_table.as_ref().unwrap().colors.len(), 256);
    s.execute("image.mode.colorTable", json!({"transparent": 0})).unwrap();
    assert!(s.execute("image.mode.colorTable", json!({"table": "nope"})).is_err());
    assert!(s.execute("image.mode.colorTable", json!({"entries": {"999": "#000000"}})).is_err());
}

#[test]
fn clearing_color_table_transparency_restores_indexed_pixels() {
    let pixel = |s: &mut Session| -> [f32; 4] { serde_json::from_value(s.execute("document.pixel", json!({"x": 5, "y": 8})).unwrap()).unwrap() };
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 16, "height": 16, "background": "#ff0000"})).unwrap();
    s.execute("image.mode.indexedColor", json!({"palette": "exact", "colors": 2, "forced": "none", "dither": "none"})).unwrap();
    assert_eq!(doc(&s).color_table.as_ref().unwrap().colors, vec![[255, 0, 0]]);
    assert_eq!(pixel(&mut s), [1.0, 0.0, 0.0, 1.0]);

    s.execute("image.mode.colorTable", json!({"transparent": 0})).unwrap();
    assert_eq!(pixel(&mut s), [0.0; 4]);
    s.execute("image.mode.colorTable", json!({"transparent": null})).unwrap();
    assert_eq!(doc(&s).layers[0].surface().unwrap().rgba(5, 8), [1.0, 0.0, 0.0, 1.0], "clearing the transparent palette entry should restore the stored pixel");
    assert_eq!(pixel(&mut s), [1.0, 0.0, 0.0, 1.0]);
    s.execute("image.mode.rgb", json!({})).unwrap();
    assert_eq!(pixel(&mut s), [1.0, 0.0, 0.0, 1.0]);

    let mut control = Session::new();
    control.execute("file.new", json!({"width": 16, "height": 16, "background": "#ff0000"})).unwrap();
    control.execute("image.mode.indexedColor", json!({"palette": "exact", "colors": 2, "forced": "none", "dither": "none"})).unwrap();
    control.execute("image.mode.colorTable", json!({"entries": {"0": "#0000ff"}})).unwrap();
    let blue: [f32; 4] = serde_json::from_value(control.execute("document.pixel", json!({"x": 5, "y": 8})).unwrap()).unwrap();
    assert_eq!(blue, [0.0, 0.0, 1.0, 1.0]);
    control.execute("image.mode.colorTable", json!({"transparent": 0})).unwrap();
    control.execute("edit.undo", json!({})).unwrap();
    let restored: [f32; 4] = serde_json::from_value(control.execute("document.pixel", json!({"x": 5, "y": 8})).unwrap()).unwrap();
    assert_eq!(restored, [0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn bitmap_methods_need_grayscale() {
    let mut s = session(32, 32, 16, "rgb");
    assert!(!s.is_enabled("image.mode.bitmap"));
    s.execute("image.mode.grayscale", json!({})).unwrap();
    for method in ["threshold", "pattern", "diffusion", "halftone"] {
        s.execute("image.mode.bitmap", json!({"method": method, "frequency": 12, "shape": "diamond"})).unwrap();
        let d = doc(&s);
        assert_eq!((d.mode, d.depth), (ColorMode::Bitmap, SampleType::U8), "{method}");
        let surf = d.layers[0].surface().unwrap();
        for y in 0..32 {
            for x in 0..32 {
                let v = surf.pixel(x, y)[0];
                assert!(v == 0.0 || v == 1.0, "{method}: {v}");
            }
        }
        s.execute("edit.undo", json!({})).unwrap();
    }
    assert!(s.execute("image.mode.bitmap", json!({"method": "nope"})).is_err());
    // An unlisted halftone shape is a bad-params error, not a silent Round (#507).
    assert!(s.execute("image.mode.bitmap", json!({"method": "halftone", "shape": "hexagon"})).is_err());
    assert_eq!(doc(&s).mode, ColorMode::Grayscale);
    for shape in ["round", "ellipse", "line", "square", "diamond", "cross"] {
        s.execute("image.mode.bitmap", json!({"method": "halftone", "shape": shape})).unwrap();
        s.execute("edit.undo", json!({})).unwrap();
    }
}

#[test]
fn bitmap_halftone_rejects_bad_angles_and_frequencies_without_editing_the_document() {
    // Photoshop's Halftone Screen takes an angle in -180..180 degrees (#4214): 1e39 used to become
    // inf, then NaN in the screen rotation, and an almost white bitmap landed in history.
    let mut s = session(16, 16, 8, "grayscale");
    let before = s.active().unwrap().history.past_len();
    let bad = [
        ("angle", json!(1e39)),
        ("angle", json!(500)),
        ("angle", json!(-181)),
        ("angle", json!(180.01)),
        ("angle", json!("x")),
        ("frequency", json!("x")),
        ("frequency", json!(true)),
    ];
    for (key, value) in bad {
        let mut p = json!({"method": "halftone"});
        p[key] = value.clone();
        assert!(s.execute("image.mode.bitmap", p).is_err(), "{key}: {value}");
        assert_eq!(s.active().unwrap().history.past_len(), before, "{key}: {value}");
        assert_eq!(doc(&s).mode, ColorMode::Grayscale, "{key}: {value}");
    }
    for angle in [-180.0, 30.0, 180.0] {
        s.execute("image.mode.bitmap", json!({"method": "halftone", "angle": angle})).unwrap();
        assert_eq!(doc(&s).mode, ColorMode::Bitmap, "{angle}");
        s.execute("edit.undo", json!({})).unwrap();
    }
    // Frequency stays clamped to the documented 1..999 lines/inch.
    s.execute("image.mode.bitmap", json!({"method": "halftone", "frequency": 1e39})).unwrap();
    assert_eq!(doc(&s).mode, ColorMode::Bitmap);
}

#[test]
fn duotone_inks_and_display() {
    let mut s = session(16, 8, 8, "gray");
    paint(&mut s, |x, _| [x as f32 / 15.0; 4].map(|v| v.min(1.0)));
    assert!(s.execute("image.mode.duotone", json!({"type": "quintone"})).is_err());
    let r = s
        .execute(
            "image.mode.duotone",
            json!({"type": "duotone", "inks": [{"name": "Black"}, {"name": "Orange", "color": "#ff8000", "curve": [[0, 0], [100, 80]]}]}),
        )
        .unwrap();
    assert_eq!(r["inks"], json!(["Black", "Orange"]));
    let d = doc(&s);
    assert_eq!(d.mode, ColorMode::Duotone);
    let inks = &d.duotone.as_ref().unwrap().inks;
    assert_eq!(inks[1].curve.last().unwrap().output, 0.8);
    // Display: warm midtones (red > blue) where the gray image is neutral.
    let shown = display_document(d).unwrap();
    let px = photocraft_compose::render(&shown, Rect::from_xywh(8, 4, 1, 1)).px[0];
    assert!(px[0] > px[2] + 0.05, "{px:?}");
    assert!(display_document(&Document::new("x", Size::new(1, 1), ColorMode::Grayscale, SampleType::U8)).is_none());
    // Tritone defaults and leaving the mode.
    s.execute("image.mode.duotone", json!({"type": "tritone"})).unwrap();
    assert_eq!(doc(&s).duotone.as_ref().unwrap().inks.len(), 3);
    s.execute("image.mode.grayscale", json!({})).unwrap();
    assert!(doc(&s).duotone.is_none());
    assert_eq!(doc(&s).mode, ColorMode::Grayscale);
    assert!(!session(4, 4, 8, "rgb").is_enabled("image.mode.duotone"));
}

#[test]
fn modes_round_trip_through_pcraft() {
    let mut s = session(12, 12, 8, "rgb");
    s.execute("image.mode.indexedColor", json!({"colors": 6})).unwrap();
    let bytes = photocraft_format::save_to_bytes(doc(&s), &Default::default()).unwrap();
    let back = photocraft_format::load_from_bytes(&bytes).unwrap();
    assert_eq!(back.color_table, doc(&s).color_table);
    assert_eq!(back.mode, ColorMode::Indexed);
}

/// Leaving Duotone for a colour mode keeps the printed look (#2383): the converted pixels are
/// the displayed ink colours, through the destination profile. Grayscale drops the inks.
#[test]
fn duotone_to_colour_modes_keeps_the_ink_colours() {
    use photocraft_cms::{Builtin, Transform};
    let configs = [
        json!({"type": "duotone"}),
        json!({"type": "monotone", "inks": [{"name": "Orange", "color": "#ff8000", "curve": [[0, 0], [100, 100]]}]}),
        json!({"type": "quadtone"}),
    ];
    for depth in [8, 16, 32] {
        for cfg in &configs {
            for mode in ["rgb", "cmyk", "lab", "grayscale"] {
                let ctx = format!("{depth}-bit {cfg} → {mode}");
                let mut s = session(16, 2, depth, "gray");
                paint(&mut s, |x, _| {
                    let v = x as f32 / 15.0;
                    [v, v, v, 1.0]
                });
                s.execute("image.mode.duotone", cfg.clone()).unwrap();
                let before = s.active().unwrap().doc.clone();
                let gray: Vec<f32> = (0..16).map(|x| before.layers[0].surface().unwrap().pixel(x, 1)[0]).collect();
                let shown = photocraft_compose::render(&display_document(&before).unwrap(), Rect::from_xywh(0, 1, 16, 1)).px;
                s.execute(&format!("image.mode.{mode}"), json!({})).unwrap();
                let d = doc(&s);
                assert!(d.duotone.is_none(), "{ctx}");
                let surf = d.layers[0].surface().unwrap();
                let target = match mode {
                    "rgb" => ColorMode::Rgb,
                    "cmyk" => ColorMode::Cmyk,
                    "lab" => ColorMode::Lab,
                    _ => ColorMode::Grayscale,
                };
                assert_eq!(d.mode, target, "{ctx}");
                let srgb = Builtin::Srgb.profile();
                let dst = crate::color_cmds::working_profile(target);
                let t = Transform::new(srgb, &dst, s.color.settings.intent(), s.color.settings.bpc).unwrap();
                for x in 0..16 {
                    let got = surf.pixel(x, 1);
                    let want: Vec<f32> = if target == ColorMode::Grayscale {
                        vec![gray[x as usize]]
                    } else {
                        let mut o = [0.0f32; 16];
                        t.eval(&shown[x as usize][..3], &mut o);
                        o[..target.color_channels()].to_vec()
                    };
                    // The display's gradient map samples the inks at 33 stops; dark CMYK
                    // separations magnify that a little.
                    let tol = if target == ColorMode::Cmyk { 0.03 } else { 0.02 };
                    for (g, w) in got.iter().zip(&want) {
                        assert!((g - w).abs() < tol, "{ctx} x={x}: got {got:?}, want {want:?}");
                    }
                }
                // Undo brings back the inks and the gray pixels.
                s.execute("edit.undo", json!({})).unwrap();
                let back = doc(&s);
                assert_eq!((back.mode, &back.duotone), (ColorMode::Duotone, &before.duotone), "{ctx}");
                assert_eq!(back.layers[0].surface().unwrap().pixel(7, 1), before.layers[0].surface().unwrap().pixel(7, 1), "{ctx}");
            }
        }
    }
}

/// Colours stored in a Duotone document (fill layers) take the ink colours too.
#[test]
fn duotone_to_rgb_inks_fill_layer_colours() {
    let mut s = session(4, 4, 8, "gray");
    s.execute("image.mode.duotone", json!({"type": "monotone", "inks": [{"name": "Orange", "color": "#ff8000"}]})).unwrap();
    s.execute("layer.newFillLayer.solidColor", json!({"color": "#808080"})).unwrap();
    s.execute("image.mode.rgb", json!({})).unwrap();
    let d = doc(&s);
    let Some(LayerContent::Fill(photocraft_doc::Fill::Solid(c))) = d.layers.last().map(|l| &l.content) else { panic!("no fill layer") };
    assert_eq!(c.mode, ColorMode::Rgb);
    // Gray 0.5 is half-density orange: (1, 0.75, 0.5).
    for (g, w) in c.c[..3].iter().zip([1.0, 0.75, 0.5]) {
        assert!((g - w).abs() < 0.02, "{c:?}");
    }
}

/// Three-entry Indexed doc (red, blue, transparent at index 2) from an RGB doc with clear pixels.
fn indexed_with_transparency() -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 8, "height": 8, "background": "transparent"})).unwrap();
    paint(&mut s, |x, _| {
        if x < 3 {
            [1.0, 0.0, 0.0, 1.0]
        } else if x < 6 {
            [0.0, 0.0, 1.0, 1.0]
        } else {
            [0.0; 4]
        }
    });
    s.execute("image.mode.indexedColor", json!({"palette": "exact", "colors": 4, "forced": "none", "dither": "none"})).unwrap();
    assert_eq!(doc(&s).color_table.as_ref().unwrap().transparent, Some(2));
    s
}

#[test]
fn shrinking_the_color_table_drops_a_transparent_index_past_its_end() {
    // #4186: the transparent index stayed at 2 after the table shrank to 2 entries, so the palette
    // PNG export refused the table.
    let mut s = indexed_with_transparency();
    s.execute("image.mode.colorTable", json!({"colors": ["#000000", "#ffffff"]})).unwrap();
    let d = doc(&s);
    let t = d.color_table.as_ref().unwrap();
    assert_eq!(t.colors.len(), 2);
    assert!(t.transparent.is_none_or(|i| usize::from(i) < t.colors.len()), "stale transparent index {:?}", t.transparent);
    // The pixels end up exactly as with an explicit `"transparent": null`.
    let mut cleared = indexed_with_transparency();
    cleared.execute("image.mode.colorTable", json!({"colors": ["#000000", "#ffffff"], "transparent": null})).unwrap();
    assert_eq!(doc(&cleared).color_table.as_ref(), Some(t));
    let (a, b) = (d.layers[0].surface().unwrap(), doc(&cleared).layers[0].surface().unwrap());
    for y in 0..8 {
        for x in 0..8 {
            assert_eq!(a.rgba(x, y), b.rgba(x, y), "({x},{y})");
        }
    }
    let png = photocraft_io::export(d, "png", &Default::default()).expect("palette PNG export");
    assert!(png.warnings.iter().any(|w| w.contains("palette PNG")), "{:?}", png.warnings);
    // Undo restores the transparent entry.
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(doc(&s).color_table.as_ref().unwrap().transparent, Some(2));
}

#[test]
fn color_table_rejects_an_invalid_transparent_index_without_editing() {
    let mut s = indexed_with_transparency();
    let before = doc(&s).color_table.clone();
    let n = s.active().unwrap().history.past_len();
    for t in [json!(999), json!(3), json!(-1), json!(1.5), json!("0"), json!(true)] {
        assert!(s.execute("image.mode.colorTable", json!({"transparent": t})).is_err(), "{t}");
        // Validated against the new colours, before any edit.
        assert!(s.execute("image.mode.colorTable", json!({"colors": ["#000000", "#ffffff"], "transparent": t})).is_err(), "{t}");
    }
    assert!(s.execute("image.mode.colorTable", json!({"colors": ["#000000", "#ffffff"], "transparent": 2})).is_err());
    assert_eq!(doc(&s).color_table, before);
    assert_eq!(s.active().unwrap().history.past_len(), n);
    // A valid index still applies.
    s.execute("image.mode.colorTable", json!({"transparent": 0})).unwrap();
    assert_eq!(doc(&s).color_table.as_ref().unwrap().transparent, Some(0));
}
