use super::*;

fn session(w: u32, h: u32) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": w, "height": h, "background": "transparent"})).unwrap();
    s
}

fn surface(s: &Session) -> Surface {
    let d = s.active().unwrap();
    d.doc.layer(d.active_layer.unwrap()).unwrap().surface().unwrap().clone()
}

fn rgba(s: &Session, x: i32, y: i32) -> [f32; 4] {
    surface(s).rgba(x, y)
}

fn same_pixels(a: &Surface, b: &Surface, r: Rect) -> bool {
    (r.y0..r.y1).all(|y| (r.x0..r.x1).all(|x| a.pixel(x, y) == b.pixel(x, y)))
}

#[test]
fn stroke_backwards_compatible() {
    let mut s = session(80, 40);
    let r = s.execute("paint.stroke", json!({"points": [[4, 20], [60, 20]], "size": 6, "color": "#0000ff"})).unwrap();
    assert_eq!(r["damage"].as_array().unwrap().len(), 4);
    assert_eq!(rgba(&s, 30, 20), [0.0, 0.0, 1.0, 1.0]);
    assert_eq!(rgba(&s, 30, 30)[3], 0.0);
    // [x, y, pressure] with the default pressure-size brush: light pressure → thinner line.
    s.execute("paint.stroke", json!({"points": [[4, 5, 0.2], [60, 5, 0.2]], "size": 10})).unwrap();
    assert!(rgba(&s, 30, 5)[3] > 0.9 && rgba(&s, 30, 8)[3] < 0.01);
}

#[test]
fn stroke_accepts_full_points_brush_and_preset() {
    let mut s = session(200, 100);
    // 7-tuples plus a brush object with tilt-driven angle on a flat tip.
    let brush = json!({"size": 30, "roundness": 0.2, "pressureSize": false, "spacing": 0.1,
        "shapeDynamics": {"enabled": true, "angle": {"control": "penTilt"}}});
    s.execute("paint.stroke", json!({"points": [[50, 50, 1, 0, -45, 0, 0], [52, 50, 1, 0, -45, 0, 16]], "brush": brush})).unwrap();
    // Tilt up = 90°: the flat tip stands vertically.
    assert!(rgba(&s, 50, 40)[3] > 0.9 && rgba(&s, 50, 60)[3] > 0.9 && rgba(&s, 60, 50)[3] < 0.01);
    // Object points.
    s.execute("paint.stroke", json!({"points": [{"x": 150, "y": 20}, {"x": 180, "y": 20, "pressure": 1.0}], "size": 4, "brush": {"pressureSize": false}}))
        .unwrap();
    assert!(rgba(&s, 165, 20)[3] > 0.9);
    // Presets by name.
    s.execute("paint.stroke", json!({"points": [[20, 80], [120, 80]], "preset": "Hard Round"})).unwrap();
    assert!(rgba(&s, 70, 80)[3] > 0.99);
    assert!(s.execute("paint.stroke", json!({"points": [[1, 1]], "preset": "Nope"})).is_err());
    assert!(s.execute("paint.stroke", json!({"points": [[1, 1]], "brush": {"size": "big"}})).is_err());
}

#[test]
fn paint_commands_bound_hostile_time_and_reject_oversized_brushes() {
    let mut s = session(32, 32);
    let timed = s.execute(
        "paint.stroke",
        json!({
            "points": [[16, 16, 1, 0, 0, 0, 0], [16, 16, 1, 0, 0, 0, 1.7976931348623157e308]],
            "brush": {"buildUp": true, "buildUpRate": 1000, "size": 4, "pressureSize": false}
        }),
    );
    assert!(timed.is_ok(), "the bounded airbrush catch-up should still commit: {timed:?}");

    for command in ["paint.stroke", "paint.pencil"] {
        let result = s.execute(command, json!({"points": [[8, 8]], "size": 1e30}));
        assert!(result.is_err(), "{command} must reject an oversized brush before rasterizing");
    }
    let result = s.execute("paint.stroke", json!({"points": [[8, 8]], "brush": {"dualBrush": {"enabled": true, "size": 1e30}}}));
    assert!(result.is_err(), "enabled dual brush dimensions are bounded too");

    s.execute("tools.setBrush", json!({"size": 5000})).expect("the shared brush limit is accepted");
    assert!(s.execute("tools.setBrush", json!({"size": 5001})).is_err(), "tools.setBrush must reject sizes above the shared limit");
    assert_eq!(s.tools.brush.size, 5000.0, "a rejected brush update must not change the session brush");
}

#[test]
fn replay_is_deterministic() {
    let params = json!({"points": [[10, 50], [60, 30, 0.7], [120, 60, 0.9], [180, 40]], "preset": "Spatter", "color": "#ff8800"});
    let run = || {
        let mut s = session(200, 100);
        s.execute("paint.stroke", params.clone()).unwrap();
        s.execute("tools.setBrush", json!({"preset": "Confetti"})).unwrap();
        s.execute("paint.stroke", json!({"points": [[10, 80], [190, 80]]})).unwrap();
        s
    };
    let (a, b) = (run(), run());
    let r = Rect::new(0, 0, 200, 100);
    assert!(same_pixels(&surface(&a), &surface(&b), r));
    // Replaying the journal into a fresh session reproduces the pixels.
    let mut c = session(200, 100);
    for (id, p) in a.journal.iter().skip(1) {
        c.execute(id, p.clone()).unwrap();
    }
    assert!(same_pixels(&surface(&a), &surface(&c), r));
    // A different explicit seed changes the scatter.
    let mut d = session(200, 100);
    let mut p2 = params.clone();
    p2["seed"] = json!(12345);
    d.execute("paint.stroke", p2).unwrap();
    let mut e = session(200, 100);
    e.execute("paint.stroke", params).unwrap();
    assert!(!same_pixels(&surface(&d), &surface(&e), r));
}

#[test]
fn pencil_is_aliased_and_auto_erases() {
    let mut s = session(60, 30);
    s.execute("paint.pencil", json!({"points": [[5, 10.3], [50, 10.3]], "size": 5, "color": "#000000"})).unwrap();
    let surf = surface(&s);
    for y in 0..30 {
        for x in 0..60 {
            let a = surf.rgba(x, y)[3];
            assert!(a == 0.0 || a == 1.0);
        }
    }
    s.execute("tools.setColors", json!({"foreground": "#000000", "background": "#ffffff"})).unwrap();
    // Starting on a foreground pixel: paints background.
    s.execute("paint.pencil", json!({"points": [[20, 10], [20, 25]], "size": 3, "autoErase": true})).unwrap();
    assert_eq!(rgba(&s, 20, 22), [1.0, 1.0, 1.0, 1.0]);
    // Starting elsewhere: paints foreground.
    s.execute("paint.pencil", json!({"points": [[40, 25], [40, 28]], "size": 3, "autoErase": true})).unwrap();
    assert_eq!(rgba(&s, 40, 26), [0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn mixer_brush_reservoir_and_cleaning() {
    let mut s = session(200, 30);
    s.execute("tools.setColors", json!({"foreground": "#ff0000"})).unwrap();
    s.execute(
        "paint.mixerBrush",
        json!({"points": [[5, 15], [195, 15]], "size": 8, "wet": 0, "load": 0, "mix": 0, "brush": {"pressureSize": false, "spacing": 0.25}}),
    )
    .unwrap();
    assert!(rgba(&s, 10, 15)[3] > 0.95 && rgba(&s, 190, 15)[3] < 0.5, "low load dries out");
    assert!(s.tools.mixer.pickup.is_none(), "cleaned after stroke");
    // Wet, not cleaned: the pickup persists into the session.
    s.execute("paint.mixerBrush", json!({"points": [[5, 5], [100, 5]], "size": 8, "wet": 100, "mix": 80, "cleanAfterStroke": false})).unwrap();
    assert!(s.tools.mixer.pickup.is_some());
    // Sample All Layers path runs.
    s.execute("paint.mixerBrush", json!({"points": [[5, 25], [50, 25]], "size": 6, "sampleAllLayers": true})).unwrap();
}

#[test]
fn color_replacement_modes() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 60, "height": 40, "background": "#33993f"})).unwrap();
    s.execute("tools.setColors", json!({"foreground": "#ff0000"})).unwrap();
    let before = rgba(&s, 30, 20);
    s.execute("paint.colorReplacement", json!({"points": [[10, 20], [50, 20]], "size": 20, "mode": "color", "tolerance": 20})).unwrap();
    let c = rgba(&s, 30, 20);
    let lum = |c: [f32; 4]| photocraft_color::blend::lum([c[0], c[1], c[2]]);
    assert!(c[0] > c[1], "{c:?}");
    assert!((lum(c) - lum(before)).abs() < 0.02);
    s.execute("paint.colorReplacement", json!({"points": [[10, 5], [50, 5]], "size": 6, "mode": "luminosity", "sampling": "once", "limits": "discontiguous"}))
        .unwrap();
    assert!((lum(rgba(&s, 30, 5)) - 0.3).abs() < 0.02);
    assert!(s.execute("paint.colorReplacement", json!({"points": [[1, 1]], "mode": "bogus"})).is_err());
}

/// The live Color Replacement stroke shows what `paint.colorReplacement` commits, at 8 and 16
/// bits: red-left/blue-right halves, a stroke from red into blue in each sampling mode. The live
/// working copy is stored at the layer's depth between dabs, so 8 bits may round a level apart.
#[test]
fn live_color_replacement_matches_the_commit() {
    for depth in [8, 16] {
        for sampling in ["continuous", "once", "backgroundSwatch"] {
            let mut s = Session::new();
            s.execute("file.new", json!({"width": 80, "height": 40, "background": "#ff0000", "depth": depth})).unwrap();
            s.execute("select.rect", json!({"x": 40, "y": 0, "width": 40, "height": 40})).unwrap();
            s.execute("edit.fill", json!({"color": "#0000ff"})).unwrap();
            s.execute("select.deselect", json!({})).unwrap();
            s.execute("tools.setColors", json!({"foreground": "#00ff00", "background": "#0000ff"})).unwrap();
            let all = [[10.0, 20.0], [30.0, 20.0], [50.0, 20.0], [70.0, 22.0]];
            let pts = |v: &[[f64; 2]]| v.iter().map(|q| StrokePoint::new(q[0], q[1], 1.0)).collect::<Vec<_>>();
            let opts = |p: &[[f64; 2]]| json!({"points": p, "size": 12, "hardness": 50, "sampling": sampling, "tolerance": 30, "target": "pixels"});
            let mut live = live_color_replacement(&s, &opts(&all[..1])).unwrap();
            live.push(&pts(&all[1..3])).unwrap();
            live.push(&pts(&all[3..])).unwrap();
            let shown = live.doc.clone();
            s.execute("paint.colorReplacement", opts(&all)).unwrap();
            let id = s.active().unwrap().active_layer.unwrap();
            let (a, b) = (shown.layer(id).unwrap().surface().unwrap(), s.active().unwrap().doc.layer(id).unwrap().surface().unwrap());
            let mut worst = 0.0f32;
            // Everywhere, the stroke's finishing dab included (the live stroke previews it).
            for y in 0..40 {
                for x in 0..80 {
                    let (got, want) = (a.rgba(x, y), b.rgba(x, y));
                    worst = (0..4).map(|c| (got[c] - want[c]).abs()).fold(worst, f32::max);
                }
            }
            assert!(b.rgba(10, 20)[1] > 0.5 || sampling == "backgroundSwatch", "{depth}-bit {sampling}: recoloured");
            assert!(worst <= 2.0 / 255.0 + 1e-6, "{depth}-bit {sampling}: live and commit differ by {worst}");
        }
    }
}

/// Color Replacement recolours layer pixels only: a mask, channel or Quick Mask target is
/// refused (it would otherwise recolour the layer behind the user's back).
#[test]
fn color_replacement_refuses_non_pixel_targets() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 20, "height": 20, "background": "#ff0000"})).unwrap();
    let steps = s.active().unwrap().history.entries().len();
    for t in [json!("mask"), json!("quickMask"), json!({"channel": 0}), json!(3)] {
        let p = json!({"points": [[10, 10]], "size": 6, "target": t});
        assert!(s.execute("paint.colorReplacement", p.clone()).is_err(), "{t}");
        assert!(live_color_replacement(&s, &p).is_err(), "{t}");
    }
    assert_eq!(s.active().unwrap().history.entries().len(), steps);
    s.execute("paint.colorReplacement", json!({"points": [[10, 10]], "size": 6, "target": "pixels"})).unwrap();
    assert_eq!(s.active().unwrap().history.entries().len(), steps + 1);
}

#[test]
fn presets_save_load_delete_round_trip() {
    let mut s = session(10, 10);
    let n0 = s.execute("brush.presets.list", json!({})).unwrap()["presets"].as_array().unwrap().len();
    assert!(n0 >= 12);
    s.execute("brush.presets.save", json!({"name": "Mine", "brush": {"size": 77, "scattering": {"enabled": true, "count": 3}}})).unwrap();
    let list = s.execute("brush.presets.list", json!({"full": true})).unwrap();
    let mine = list["presets"].as_array().unwrap().iter().find(|p| p["name"] == "Mine").unwrap().clone();
    assert_eq!(mine["brush"]["size"], 77.0);
    assert_eq!(mine["brush"]["scattering"]["count"], 3);
    // Round trip: select it, read it back.
    let b = s.execute("tools.setBrush", json!({"preset": "mine"})).unwrap();
    assert_eq!(b["size"], 77.0);
    let back: BrushSettings = serde_json::from_value(s.execute("brush.get", json!({})).unwrap()).unwrap();
    assert_eq!(back, s.tools.brush);
    // Overwrite, then delete.
    s.execute("brush.presets.save", json!({"name": "Mine"})).unwrap();
    assert_eq!(s.tools.presets.len(), n0 + 1);
    s.execute("brush.presets.delete", json!({"name": "Mine"})).unwrap();
    assert_eq!(s.tools.presets.len(), n0);
    assert!(s.execute("brush.presets.delete", json!({"name": "Mine"})).is_err());
    assert!(s.execute("brush.presets.save", json!({})).is_err());
}

/// The New Brush Preset dialog's boxes (#4490): a preset saved without Capture Brush Size or
/// Include Tool Settings leaves the tool's size, mode, opacity, flow and pressure buttons as they
/// are when picked; one saved with Include Color brings its colour back as the foreground. A preset
/// saved without the boxes picks as it always did.
#[test]
fn presets_save_captures_size_tool_settings_and_color_as_asked() {
    let mut s = session(10, 10);
    s.execute("tools.setBrush", json!({"size": 40, "opacity": 0.5, "flow": 0.25, "mode": "Multiply", "pressureSize": false})).unwrap();
    s.tools.foreground = [1.0, 0.0, 0.0, 1.0];
    s.execute("brush.presets.save", json!({"name": "Legacy"})).unwrap();
    s.execute("brush.presets.save", json!({"name": "Bare", "captureSize": false, "toolSettings": false})).unwrap();
    s.execute("brush.presets.save", json!({"name": "Red", "color": true})).unwrap();
    let caps = |n: &str| s.tools.presets.iter().find(|p| p.name == n).unwrap().captures.clone();
    assert!(caps("Legacy").is_default(), "no boxes, the captures of any other preset");
    assert_eq!(caps("Bare"), PresetCaptures { size: false, tool_settings: false, color: false });
    assert!(caps("Red").color);

    // Change the tool, then pick each.
    let other = json!({"size": 9, "opacity": 1.0, "flow": 1.0, "mode": "Normal", "pressureSize": true});
    s.execute("tools.setBrush", other.clone()).unwrap();
    s.tools.foreground = [0.0, 0.0, 1.0, 1.0];
    s.execute("tools.setBrush", json!({"preset": "Legacy"})).unwrap();
    let b = &s.tools.brush;
    assert_eq!((b.size, b.opacity, b.flow, b.mode, b.pressure_size), (40.0, 0.5, 0.25, photocraft_color::BlendMode::Multiply, false), "everything it saved");
    assert_eq!(s.tools.foreground, [0.0, 0.0, 1.0, 1.0], "a preset without Include Color leaves the foreground");

    s.execute("tools.setBrush", other.clone()).unwrap();
    s.execute("tools.setBrush", json!({"preset": "Bare"})).unwrap();
    let b = &s.tools.brush;
    assert_eq!((b.size, b.opacity, b.flow, b.mode, b.pressure_size), (9.0, 1.0, 1.0, photocraft_color::BlendMode::Normal, true), "the tool's own");
    assert_eq!(s.tools.current_preset.as_deref(), Some("Bare"));

    s.execute("tools.setBrush", json!({"preset": "Red"})).unwrap();
    assert_eq!(s.tools.foreground, [1.0, 0.0, 0.0, 1.0], "Include Color sets the foreground");
    // A stroke naming the preset takes the same brush.
    let r = resolve_brush(&s, &json!({"preset": "Bare"}), "paint.brush").unwrap();
    assert_eq!(r.size, s.tools.brush.size);
}

#[test]
fn presets_save_appends_at_the_end_and_update_overwrites_in_place() {
    let mut s = session(10, 10);
    s.execute("tools.setBrush", json!({"preset": "Chalk"})).unwrap();
    assert_eq!(s.tools.current_preset.as_deref(), Some("Chalk"));
    let n = s.tools.presets.len();
    let chalk_at = s.tools.presets.iter().position(|p| p.name == "Chalk").unwrap();
    // A new preset appends at the end, picked preset or not.
    s.execute("brush.presets.save", json!({"name": "Mine"})).unwrap();
    assert_eq!(s.tools.presets.last().unwrap().name, "Mine");
    s.execute("brush.presets.save", json!({"name": "Tail"})).unwrap();
    assert_eq!(s.tools.presets.last().unwrap().name, "Tail");
    // A reset clears the picked preset.
    s.execute("tools.setBrush", json!({"reset": true})).unwrap();
    assert_eq!(s.tools.current_preset, None);
    // Editing the brush keeps the picked preset current; update overwrites it in place.
    s.execute("tools.setBrush", json!({"preset": "Chalk"})).unwrap();
    s.execute("tools.setBrush", json!({"size": 55})).unwrap();
    assert_eq!(s.tools.current_preset.as_deref(), Some("Chalk"));
    let r = s.execute("brush.presets.update", json!({})).unwrap();
    assert_eq!(r["name"], "Chalk");
    assert_eq!(s.tools.presets.len(), n + 2, "no new preset (the two saves added two)");
    assert_eq!(s.tools.presets.iter().position(|p| p.name == "Chalk"), Some(chalk_at));
    let chalk = s.tools.presets.iter().find(|p| p.name == "Chalk").unwrap();
    assert_eq!(chalk.brush.size, 55.0);
    assert!(!chalk.builtin, "an updated built-in becomes the user's preset");
    // A named update of another preset.
    s.execute("brush.presets.update", json!({"name": "Mine", "brush": {"size": 9}})).unwrap();
    assert_eq!(s.tools.presets.iter().find(|p| p.name == "Mine").unwrap().brush.size, 9.0);
    // Errors: no such preset, and no preset to update after a reset.
    assert!(s.execute("brush.presets.update", json!({"name": "Nope"})).is_err());
    s.execute("tools.setBrush", json!({"reset": true})).unwrap();
    assert!(s.execute("brush.presets.update", json!({})).is_err());
    // Deleting the current preset clears it, so a later update fails rather than guessing.
    s.execute("tools.setBrush", json!({"preset": "Chalk"})).unwrap();
    s.execute("brush.presets.delete", json!({"name": "Chalk"})).unwrap();
    assert_eq!(s.tools.current_preset, None);
    assert!(s.execute("brush.presets.update", json!({})).is_err());
}

#[test]
fn a_failed_set_brush_leaves_no_current_preset_behind() {
    let mut s = session(10, 10);
    let before = s.tools.brush.clone();
    // A rejected brush (a wrong-typed size) must not record the preset it came with.
    assert!(s.execute("tools.setBrush", json!({"preset": "Chalk", "brush": {"size": "big"}})).is_err());
    assert_eq!(s.tools.current_preset, None);
    assert_eq!(s.tools.brush, before);
    // Once the brush is accepted, the preset is recorded.
    s.execute("tools.setBrush", json!({"preset": "Chalk"})).unwrap();
    assert_eq!(s.tools.current_preset.as_deref(), Some("Chalk"));
    // A `reset` wins over a `preset` in the same call.
    s.execute("tools.setBrush", json!({"preset": "Chalk", "reset": true})).unwrap();
    assert_eq!(s.tools.current_preset, None);
}

#[test]
fn set_brush_merges_fields() {
    let mut s = session(10, 10);
    s.execute("tools.setBrush", json!({"size": 55, "shapeDynamics": {"enabled": true, "size": {"jitter": 0.5}}})).unwrap();
    s.execute("tools.setBrush", json!({"shapeDynamics": {"angle": {"jitter": 0.25}}})).unwrap();
    let b = &s.tools.brush;
    assert_eq!(b.size, 55.0);
    assert!(b.shape_dynamics.enabled && b.shape_dynamics.size.jitter == 0.5 && b.shape_dynamics.angle.jitter == 0.25);
    s.execute("tools.setBrush", json!({"reset": true})).unwrap();
    assert_eq!(s.tools.brush, BrushSettings::default());
    // Protect texture carries the pattern across textured presets.
    s.execute("tools.setBrush", json!({"preset": "Chalk", "protectTexture": true})).unwrap();
    let chalk_pattern = s.tools.brush.texture.pattern.clone();
    s.execute("tools.setBrush", json!({"preset": "Canvas Texture"})).unwrap();
    assert_eq!(s.tools.brush.texture.pattern, chalk_pattern);
}

#[test]
fn set_brush_keeps_untouched_bitmaps_and_replaces_touched_ones() {
    let tile = |w, h, k: u32| GrayTile::from_fn(w, h, move |x, y| ((x * k + y) % 5) as f32 / 4.0);
    let base = BrushSettings {
        tip: TipShape::Sampled(tile(900, 700, 3)),
        dual_brush: photocraft_paint::DualBrush { tip: TipShape::Sampled(tile(30, 20, 2)), ..Default::default() },
        texture: photocraft_paint::Texture { pattern: photocraft_paint::Pattern::Tile(tile(64, 64, 1)), ..Default::default() },
        ..Default::default()
    };
    let cmd = "tools.setBrush";
    // Untouched bitmaps survive any patch that doesn't name them.
    for patch in [json!({"size": 12}), json!({"dualBrush": {"size": 9}}), json!({"texture": {"depth": 0.5}}), json!({})] {
        let b = merge_brush(&base, &patch, cmd).unwrap();
        assert_eq!((&b.tip, &b.dual_brush.tip, &b.texture.pattern), (&base.tip, &base.dual_brush.tip, &base.texture.pattern), "{patch}");
    }
    let b = merge_brush(&base, &json!({"size": 12, "dualBrush": {"size": 9}}), cmd).unwrap();
    assert_eq!((b.size, b.dual_brush.size), (12.0, 9.0));
    // Named ones are replaced.
    let b = merge_brush(
        &base,
        &json!({"tip": "round", "dualBrush": {"tip": "round"}, "texture": {"pattern": {"procedural": {"style": "dots", "size": 32, "seed": 2}}}}),
        cmd,
    )
    .unwrap();
    assert_eq!((&b.tip, &b.dual_brush.tip), (&TipShape::Round, &TipShape::Round));
    assert!(matches!(b.texture.pattern, photocraft_paint::Pattern::Procedural { size: 32, .. }));
    // A section replaced by a non-object is an error, not a silently kept bitmap.
    assert!(merge_brush(&base, &json!({"dualBrush": 3}), cmd).is_err());
    assert!(merge_brush(&base, &json!(null), cmd).is_err());
    // Speed: a panel edit on a big imported tip doesn't base64 the tip.
    let big = BrushSettings { tip: TipShape::Sampled(tile(2500, 2500, 7)), ..Default::default() };
    let t0 = std::time::Instant::now();
    for i in 0..10 {
        merge_brush(&big, &json!({"size": 10 + i}), cmd).unwrap();
    }
    let ms = t0.elapsed().as_secs_f64() * 100.0;
    eprintln!("merge_brush on a 2500 px tip: {ms:.2} ms per edit");
}

#[test]
fn define_brush_from_selection() {
    let mut s = session(40, 40);
    assert!(s.execute("brush.defineFromSelection", json!({"name": "Blob"})).is_err(), "needs a selection");
    // A black 6×4 block inside a larger selection.
    s.edit("setup", |doc, active| {
        doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap().fill_rect(Rect::new(10, 12, 16, 16), &[0.0, 0.0, 0.0, 1.0]);
        Ok(())
    })
    .unwrap();
    s.execute("select.rect", json!({"x": 5, "y": 5, "width": 20, "height": 20})).unwrap();
    let r = s.execute("brush.defineFromSelection", json!({"name": "Blob"})).unwrap();
    assert_eq!((r["width"].as_u64(), r["height"].as_u64()), (Some(6), Some(4)));
    assert_eq!(s.tools.brush.size, 6.0);
    assert!(matches!(&s.tools.brush.tip, TipShape::Sampled(t) if t.width == 6 && t.height == 4 && t.get(0, 0) == 1.0));
    assert!(s.tools.presets.iter().any(|p| p.name == "Blob"));
    // Paint with it.
    s.execute("select.deselect", json!({})).unwrap();
    s.execute("paint.stroke", json!({"points": [[30, 32]], "color": "#ff0000"})).unwrap();
    assert!(rgba(&s, 30, 32)[3] > 0.9);
}

#[test]
fn works_on_cmyk_and_16_bit() {
    for (mode, depth) in [("cmyk", 8), ("rgb", 16), ("grayscale", 32)] {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 60, "height": 30, "mode": mode, "depth": depth})).unwrap();
        s.execute("paint.stroke", json!({"points": [[5, 15], [55, 15]], "preset": "Chalk", "color": "#000000"})).unwrap();
        s.execute("paint.pencil", json!({"points": [[5, 5], [55, 5]], "size": 2, "color": "#000000"})).unwrap();
        assert!(rgba(&s, 30, 5)[0] < 0.2, "{mode}/{depth}: {:?}", rgba(&s, 30, 5));
    }
}

#[test]
fn brush_blend_mode_multiply() {
    // Painting with a blend mode (options-bar "Mode"): blue × yellow = black (Multiply).
    let fill = |s: &mut Session| {
        s.edit("bg", |doc, a| {
            doc.layer_mut(a.unwrap()).unwrap().surface_mut().unwrap().fill_rect(Rect::new(0, 0, 40, 40), &[1.0, 1.0, 0.0, 1.0]);
            Ok(())
        })
        .unwrap();
    };
    let mut s = session(40, 40);
    fill(&mut s);
    s.execute("paint.stroke", json!({"points": [[5, 20], [35, 20]], "size": 12, "color": "#0000ff", "mode": "multiply", "brush": {"hardness": 1.0}})).unwrap();
    let c = rgba(&s, 20, 20);
    assert!(c[0] < 0.1 && c[1] < 0.1 && c[2] < 0.1, "multiply blue×yellow ≈ black: {c:?}");
    // Normal mode paints opaque blue at the same spot.
    let mut s2 = session(40, 40);
    fill(&mut s2);
    s2.execute("paint.stroke", json!({"points": [[5, 20], [35, 20]], "size": 12, "color": "#0000ff", "brush": {"hardness": 1.0}})).unwrap();
    let n = rgba(&s2, 20, 20);
    assert!(n[2] > 0.9 && n[0] < 0.1, "normal paints blue: {n:?}");
}

#[test]
fn eraser_on_background_layer_paints_the_background_colour() {
    // Issue #15: erasing the Background layer (locked transparency) must paint the background
    // colour, not silently do nothing (opacity can't be removed there).
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 20, "height": 20, "background": "white"})).unwrap();
    s.execute("tools.setColors", json!({"background": "#ff0000"})).unwrap();
    s.execute("paint.stroke", json!({"points": [[2, 10], [18, 10]], "size": 8, "erase": true, "brush": {"hardness": 1.0}})).unwrap();
    let st = s.active().unwrap();
    let c = st.doc.layers[0].surface().unwrap().rgba(10, 10);
    assert!(c[0] > 0.9 && c[1] < 0.1 && c[2] < 0.1 && c[3] > 0.99, "erased to opaque background red: {c:?}");
    // A corner the stroke didn't touch stays white.
    let w = st.doc.layers[0].surface().unwrap().rgba(10, 2);
    assert!(w[0] > 0.9 && w[1] > 0.9 && w[2] > 0.9, "untouched stays white: {w:?}");
}

#[test]
fn live_stroke_matches_the_committed_stroke() {
    // The canvas previews strokes with `LiveStroke`; it must show what `paint.stroke` commits
    // (soft edge, scatter seed), not a stand-in.
    let mut s = session(200, 100);
    let pts = [[10.0, 50.0, 1.0], [60.0, 30.0, 0.7], [120.0, 60.0, 0.9], [180.0, 40.0, 1.0]];
    let p = json!({"points": [pts[0]], "preset": "Spatter", "brush": {"hardness": 0.0}, "smoothing": 0.0, "target": "pixels"});
    let mut live = LiveStroke::begin(&s, &p).unwrap();
    for c in pts[1..].chunks(2) {
        let sp: Vec<StrokePoint> = c.iter().map(|q| StrokePoint::new(q[0], q[1], q[2] as f32)).collect();
        assert!(!live.push(&sp).unwrap().is_empty());
    }
    let mut commit = p.clone();
    commit["points"] = json!(pts);
    commit["seed"] = json!(live.seed);
    s.execute("paint.stroke", commit).unwrap();
    let shown = live.doc.layer(s.active().unwrap().active_layer.unwrap()).unwrap().surface().unwrap().clone();
    assert!(same_pixels(&shown, &surface(&s), Rect::new(0, 0, 200, 100)));
    assert!((0..100).any(|y| (0..200).any(|x| (0.05..0.95).contains(&shown.rgba(x, y)[3]))), "soft edge previewed");
    // Bad input is an error, not a panic.
    assert!(LiveStroke::begin(&s, &json!({"points": []})).is_err());
    assert!(LiveStroke::begin(&s, &json!({"points": [[1, 1]], "target": {"channel": 9}})).is_err());
    assert!(LiveStroke::begin(&Session::new(), &json!({"points": [[1, 1]]})).is_err());
}

#[test]
fn stroke_coordinate_validation_keeps_the_existing_bounds() {
    let edges = [StrokePoint::new(-MAX_COORD, MAX_COORD, 1.0), StrokePoint::new(MAX_COORD, -MAX_COORD, 1.0)];
    assert!(check_coords(&edges, "paint.stroke").is_ok());
    assert!(check_coords(&[], "paint.stroke").is_ok(), "an empty live update is allowed");
    for (x, y) in [(MAX_COORD + 1.0, 0.0), (0.0, -MAX_COORD - 1.0), (f64::NAN, 0.0), (0.0, f64::INFINITY)] {
        assert!(check_coords(&[StrokePoint::new(x, y, 1.0)], "paint.stroke").is_err());
    }
    // The command parser still rejects empty strokes and shares the same inclusive boundary.
    assert!(parse_points(&json!({"points": []}), "paint.stroke").is_err());
    assert!(parse_points(&json!({"points": [[MAX_COORD, -MAX_COORD]]}), "paint.stroke").is_ok());
}

#[test]
fn live_brush_and_pencil_reject_invalid_batches_without_changing_the_preview() {
    for depth in [8, 16, 32] {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 32, "height": 32, "depth": depth, "background": "transparent"})).unwrap();
        let st = s.active().unwrap();
        let (doc, revision, history) = (st.doc.clone(), st.revision, st.history.past_len());
        let id = st.active_layer.unwrap();
        for (cmd, erase) in [("paint.stroke", false), ("paint.stroke", true), ("paint.pencil", false)] {
            let p = json!({"points": [[8, 16]], "size": 4, "erase": erase, "seed": 7});
            let mut live = LiveStroke::begin_with(&s, cmd, &p).unwrap();
            let mut control = LiveStroke::begin_with(&s, cmd, &p).unwrap();
            // NaN first: on an unguarded renderer this fails without starting the enormous segment.
            for (x, y) in [(f64::NAN, 16.0), (16.0, f64::NAN), (f64::INFINITY, 16.0), (16.0, f64::NEG_INFINITY), (1e300, 16.0), (16.0, -MAX_COORD - 1.0)] {
                let shown = live.doc.clone();
                let bounds = live.bounds();
                let batch = [StrokePoint::new(12.0, 18.0, 1.0), StrokePoint::new(x, y, 1.0)];
                let err = live.push(&batch).unwrap_err();
                assert!(matches!(err, EngineError::BadParams { cmd: ref actual, ref msg }
                    if actual == cmd && msg.contains("point coordinates must be finite")));
                assert!(std::sync::Arc::ptr_eq(&shown, &live.doc), "reject the whole batch before touching the preview");
                assert_eq!(live.bounds(), bounds);
            }
            // A rejected batch must not advance the renderer; resuming is identical to never
            // receiving it. Ordinary off-canvas points remain valid too.
            let next = [StrokePoint::new(-4.0, 16.0, 1.0), StrokePoint::new(24.0, 16.0, 1.0)];
            assert_eq!(live.push(&next).unwrap(), control.push(&next).unwrap());
            assert_eq!(live.bounds(), control.bounds());
            let (got, want) = (live.doc.layer(id).unwrap().surface().unwrap(), control.doc.layer(id).unwrap().surface().unwrap());
            assert!(same_pixels(got, want, live.bounds()));
            assert_eq!(got.content_bounds(), want.content_bounds());
        }
        let st = s.active().unwrap();
        assert!(std::sync::Arc::ptr_eq(&doc, &st.doc));
        assert_eq!((st.revision, st.history.past_len()), (revision, history));
    }
}

#[test]
fn live_pencil_matches_the_committed_pencil_and_auto_erases() {
    let mut s = session(80, 40);
    s.execute("tools.setColors", json!({"foreground": "#000000", "background": "#ffffff"})).unwrap();
    // A soft, low-flow session brush: the Pencil is still hard, aliased and full flow.
    s.execute("tools.setBrush", json!({"brush": {"size": 3, "hardness": 0.0, "flow": 0.3}})).unwrap();
    let pts = [[5.0, 5.0, 1.0], [30.2, 17.7, 0.6], [70.0, 33.0, 1.0]];
    for (auto, start) in [(false, pts[0]), (true, [30.4, 17.2, 1.0])] {
        let p = json!({"points": [start], "autoErase": auto, "smoothing": 0.0});
        let mut live = LiveStroke::begin_with(&s, "paint.pencil", &p).unwrap();
        let rest: Vec<StrokePoint> = pts.iter().map(|q| StrokePoint::new(q[0], q[1], q[2] as f32)).collect();
        live.push(&rest).unwrap();
        let mut commit = p.clone();
        let mut all = vec![start];
        all.extend(pts);
        commit["points"] = json!(all);
        commit["seed"] = json!(live.seed);
        s.execute("paint.pencil", commit).unwrap();
        let shown = live.doc.layer(s.active().unwrap().active_layer.unwrap()).unwrap().surface().unwrap().clone();
        assert!(same_pixels(&shown, &surface(&s), Rect::new(0, 0, 80, 40)), "auto erase {auto}");
        for y in 0..40 {
            for x in 0..80 {
                let a = shown.rgba(x, y)[3];
                assert!(a == 0.0 || a == 1.0, "partial alpha {a} at ({x},{y})");
            }
        }
    }
    // The second stroke started on the first one's black: Auto Erase painted white.
    assert_eq!(rgba(&s, 70, 33), [1.0, 1.0, 1.0, 1.0]);
    // Only the Brush and the Pencil stroke live.
    assert!(LiveStroke::begin_with(&s, "paint.mixerBrush", &json!({"points": [[1, 1]]})).is_err());
    assert!(LiveStroke::begin_with(&s, "paint.pencil", &json!({"points": "none"})).is_err());
}

/// Pencil latency on a 24 MP document: the time to start a live stroke, per pointer move, and
/// to commit. `cargo test -p photocraft-engine --release --lib pencil_latency -- --ignored --nocapture`
#[test]
#[ignore]
fn pencil_latency_24mp() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 6000, "height": 4000, "background": "white"})).unwrap();
    for size in [1.0f64, 9.0, 40.0] {
        let p = json!({"points": [[100.0, 100.0]], "size": size, "smoothing": 0.0});
        let t = std::time::Instant::now();
        let mut live = LiveStroke::begin_with(&s, "paint.pencil", &p).unwrap();
        let begin = t.elapsed().as_secs_f64() * 1000.0;
        let mut steps = Vec::new();
        let mut all = vec![[100.0, 100.0]];
        for i in 1..=300 {
            let q = [100.0 + i as f64 * 15.0, 100.0 + (i as f64 * 0.07).sin() * 900.0 + 1000.0];
            all.push(q);
            let t = std::time::Instant::now();
            live.push(&[StrokePoint::new(q[0], q[1], 1.0)]).unwrap();
            steps.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let mut commit = p.clone();
        commit["points"] = json!(all);
        commit["seed"] = json!(live.seed);
        let t = std::time::Instant::now();
        s.execute("paint.pencil", commit).unwrap();
        let done = t.elapsed().as_secs_f64() * 1000.0;
        steps.sort_by(f64::total_cmp);
        println!(
            "pencil {size:>4} px: begin {begin:.2} ms, per move p50 {:.3} ms p95 {:.3} ms max {:.3} ms, commit {done:.1} ms",
            steps[steps.len() / 2],
            steps[steps.len() * 95 / 100],
            steps[steps.len() - 1]
        );
    }
}

#[test]
fn live_stroke_equals_the_commit_at_every_zoom_with_smoothing() {
    // #189: the preview while dragging must be the committed stroke (within 1/255), with a soft
    // brush on an opaque Background with content, at any zoom (the smoothing string scales with
    // it), with smoothing on, pressure, 8 and 16 bits, and points fed one pointer move at a time.
    let pts: Vec<[f64; 3]> = (0..40)
        .map(|i| {
            let t = f64::from(i) / 39.0;
            [40.0 + 120.0 * t, 60.0 + 30.0 * (t * 7.0).sin(), 0.3 + 0.7 * t]
        })
        .collect();
    for depth in [8, 16] {
        for zoom in [0.25, 1.0, 4.0, 16.0] {
            for (amount, pulled) in [(0.1, false), (0.5, false), (0.3, true)] {
                let mut s = Session::new();
                s.execute("file.new", json!({"width": 200, "height": 120, "depth": depth, "background": "white"})).unwrap();
                s.execute("edit.fill", json!({"color": "#1a2550"})).unwrap();
                s.execute("paint.stroke", json!({"points": [[20, 20], [180, 100]], "size": 8, "hardness": 1.0, "color": "#ffffff"})).unwrap();
                s.tools.foreground = [0.0, 0.0, 0.0, 1.0];
                let brush = json!({"size": 40, "hardness": 0.0, "pressureSize": true, "smoothing": {"amount": amount, "pulledString": pulled}});
                let p = json!({"points": [pts[0]], "brush": brush, "zoom": zoom, "target": "pixels"});
                let mut live = LiveStroke::begin(&s, &p).unwrap();
                for q in &pts[1..] {
                    live.push(&[StrokePoint::new(q[0], q[1], q[2] as f32)]).unwrap();
                }
                let mut commit = p.clone();
                commit["points"] = json!(pts);
                commit["seed"] = json!(live.seed);
                s.execute("paint.stroke", commit).unwrap();
                let shown = live.doc.layer(s.active().unwrap().active_layer.unwrap()).unwrap().surface().unwrap().clone();
                let done = surface(&s);
                let worst = (0..120).flat_map(|y| (0..200).map(move |x| (x, y))).map(|(x, y)| {
                    let (a, b) = (shown.rgba(x, y), done.rgba(x, y));
                    (0..4).map(|i| (a[i] - b[i]).abs()).fold(0.0f32, f32::max)
                });
                let worst = worst.fold(0.0f32, f32::max);
                let label = format!("depth {depth}, zoom {zoom}, smoothing {amount} (pulled string {pulled})");
                assert!(worst <= 1.0 / 255.0, "{label}: preview differs from the commit by {worst}");
                // Not two blank canvases: the soft stroke is there, edge included.
                assert!(done.rgba(40, 60)[0] < 0.1 && (0.05..0.9).contains(&done.rgba(40, 60 - 14)[2]), "{label}: soft stroke");
            }
        }
    }
}

#[test]
fn new_brush_controls_round_trip_through_set_brush() {
    let mut s = session(40, 40);
    let patch = json!({
        "spacingEnabled": false,
        "shapeDynamics": {"tiltScale": 1.5, "brushProjection": true},
        "transfer": {"wetness": {"jitter": 0.3, "control": "penPressure"}, "mix": {"jitter": 0.6, "minimum": 0.2}},
        "locks": {"shapeDynamics": true, "texture": true, "noise": true},
        "mixer": {"wet": 0.8, "load": 0.25, "mix": 0.4, "flow": 0.9, "sampleAllLayers": true},
    });
    s.execute("tools.setBrush", json!({ "brush": patch })).unwrap();
    let b = &s.tools.brush;
    assert!(!b.spacing_enabled && b.shape_dynamics.brush_projection && b.shape_dynamics.tilt_scale == 1.5);
    assert_eq!(b.transfer.wetness.control, photocraft_paint::Control::PenPressure);
    assert_eq!((b.transfer.mix.jitter, b.transfer.mix.minimum), (0.6, 0.2));
    assert!(b.locks.shape_dynamics && b.locks.texture && b.locks.noise && !b.locks.scattering);
    assert_eq!((b.mixer.wet, b.mixer.load, b.mixer.mix, b.mixer.flow, b.mixer.sample_all_layers), (0.8, 0.25, 0.4, 0.9, true));
    // brush.get reports them, and a fresh session gets the same brush from the reported JSON.
    let got = s.execute("brush.get", json!({})).unwrap();
    for (path, want) in [
        ("/spacingEnabled", json!(false)),
        ("/shapeDynamics/tiltScale", json!(1.5)),
        ("/shapeDynamics/brushProjection", json!(true)),
        ("/transfer/wetness/control", json!("penPressure")),
        ("/locks/texture", json!(true)),
        ("/mixer/sampleAllLayers", json!(true)),
        ("/mixer/load", json!(0.25)),
    ] {
        assert_eq!(got.pointer(path), Some(&want), "{path}");
    }
    let mut t = session(40, 40);
    t.execute("tools.setBrush", json!({ "brush": got })).unwrap();
    assert_eq!(t.tools.brush, s.tools.brush);
    // Locked sections survive picking a preset through the command.
    s.execute("tools.setBrush", json!({"preset": "Chalk"})).unwrap();
    assert!(s.tools.brush.shape_dynamics.brush_projection && s.tools.brush.locks.texture);
    // Wrong types are errors and leave the brush alone.
    let before = s.tools.brush.clone();
    for bad in
        [json!({"spacingEnabled": 3}), json!({"locks": {"texture": "yes"}}), json!({"mixer": {"wet": "wet"}}), json!({"shapeDynamics": {"tiltScale": []}})]
    {
        assert!(s.execute("tools.setBrush", json!({ "brush": bad })).is_err());
    }
    assert_eq!(s.tools.brush, before);
}

#[test]
fn mixer_brush_uses_the_brush_mixer_settings() {
    let mut s = session(200, 30);
    s.execute("tools.setColors", json!({"foreground": "#ff0000"})).unwrap();
    // No wet/load params: the brush's Mixer settings (a dry brush with 0 % Load) apply.
    s.execute("tools.setBrush", json!({"brush": {"pressureSize": false, "spacing": 0.25, "mixer": {"wet": 0, "load": 0, "mix": 0}}})).unwrap();
    s.execute("paint.mixerBrush", json!({"points": [[5, 15], [195, 15]], "size": 8})).unwrap();
    assert!(rgba(&s, 10, 15)[3] > 0.95 && rgba(&s, 190, 15)[3] < 0.5, "the brush's 0 % Load dries out");
    // A param still overrides the brush.
    s.execute("paint.mixerBrush", json!({"points": [[5, 5], [195, 5]], "size": 8, "load": 100})).unwrap();
    assert!(rgba(&s, 190, 5)[3] > 0.95);
}

#[test]
fn coalesced_set_brush_calls_journal_once_per_gesture() {
    let mut s = session(20, 20);
    let n = s.journal.len();
    for k in 1..=10 {
        s.execute("tools.setBrush", json!({"brush": {"opacity": k as f32 / 20.0}, "coalesce": "bar:1"})).unwrap();
        s.execute("brush.get", json!({})).unwrap();
    }
    s.execute("tools.setBrush", json!({"brush": {"shapeDynamics": {"tiltScale": 0.5}}, "coalesce": "bar:1"})).unwrap();
    assert_eq!(s.journal.len(), n + 1, "one gesture, one entry");
    let (id, p) = s.journal.last().unwrap();
    assert_eq!(id, "tools.setBrush");
    assert_eq!(p["brush"], json!({"opacity": 0.5, "shapeDynamics": {"tiltScale": 0.5}}));
    // A new gesture is a new entry; so is an uncoalesced call.
    s.execute("tools.setBrush", json!({"brush": {"flow": 0.5}, "coalesce": "bar:2"})).unwrap();
    s.execute("tools.setBrush", json!({"brush": {"flow": 0.4}, "coalesce": "bar:2"})).unwrap();
    s.execute("tools.setBrush", json!({"brush": {"size": 33}})).unwrap();
    assert_eq!(s.journal.len(), n + 3);
    // Another command in between ends the gesture.
    s.execute("tools.setBrush", json!({"brush": {"size": 34}, "coalesce": "bar:3"})).unwrap();
    s.execute("paint.stroke", json!({"points": [[2, 2], [10, 10]]})).unwrap();
    s.execute("tools.setBrush", json!({"brush": {"size": 35}, "coalesce": "bar:3"})).unwrap();
    assert_eq!(s.journal.len(), n + 6);
    // Replay gives the same brush.
    let mut t = session(20, 20);
    for (id, p) in s.journal.iter().skip(1) {
        t.execute(id, p.clone()).unwrap();
    }
    assert_eq!(t.tools.brush, s.tools.brush);
}

/// A Scatter amount beyond the 1000 % maximum (the issue's 1e38, just above the maximum, or one
/// that only fits as infinity) is rejected naming the value, before the session brush changes, by
/// `tools.setBrush` and by a stroke's own brush patch; the maximum itself and an ordinary amount
/// above 100 % still paint (#977).
#[test]
fn out_of_range_scatter_is_rejected_and_the_maximum_still_paints() {
    let new = || {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 32, "height": 32})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        s
    };
    let scatter = |j: f64| json!({"scattering": {"enabled": true, "bothAxes": true, "scatter": {"jitter": j}}});
    let dual = |j: f64| json!({"dualBrush": {"enabled": true, "bothAxes": true, "size": 6, "scatter": j}});
    let stroke = json!({"points": [[10, 10]]});
    for (j, shown) in [(1e38, "1e38"), (10.01, "10.01"), (1e39, "inf"), (-1.0, "-1.0")] {
        for (patch, what) in [(scatter(j), "scatter"), (dual(j), "dual brush scatter")] {
            let mut s = new();
            let before = s.tools.brush.clone();
            let err = s.execute("tools.setBrush", json!({ "brush": patch.clone() })).unwrap_err();
            let msg = err.to_string();
            assert!(matches!(err, EngineError::BadParams { .. }) && msg.contains(what) && msg.contains(shown), "{j}: {msg}");
            assert_eq!(s.tools.brush, before, "a rejected scatter must not change the session brush");
            let (doc, revision) = (s.active().unwrap().doc.clone(), s.active().unwrap().revision);
            let err = paint_stroke(&mut s, &json!({"points": [[10, 10]], "brush": patch})).unwrap_err();
            assert!(matches!(err, EngineError::BadParams { .. }) && err.to_string().contains(shown), "{j}: {err}");
            assert!(std::sync::Arc::ptr_eq(&doc, &s.active().unwrap().doc) && s.active().unwrap().revision == revision);
        }
    }
    // The issue's repro, end to end: a stroke after the rejected update still paints normally.
    let mut s = new();
    assert!(s.execute("tools.setBrush", json!({ "brush": scatter(1e38) })).is_err());
    assert!(s.execute("paint.stroke", stroke.clone()).is_ok());
    // The maximum (1000 %) is accepted and paints a finite, well-formed dab.
    for patch in [scatter(10.0), dual(10.0)] {
        let mut s = new();
        s.execute("tools.setBrush", json!({ "brush": patch })).unwrap();
        let r = s.execute("paint.stroke", stroke.clone()).unwrap();
        let d: Vec<i64> = r["damage"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
        assert!(d[0].abs() < 1000 && d[1].abs() < 1000 && d[2] > 0 && d[3] > 0, "{r}");
    }
    // Control: 200 % scatter paints where it did before the cap (the soft default tip reaches a
    // little past its radius).
    let mut s = new();
    s.execute("tools.setBrush", json!({ "brush": scatter(2.0) })).unwrap();
    assert_eq!(s.execute("paint.stroke", stroke).unwrap(), json!({"damage": [15, 12, 27, 27]}));
}

#[test]
fn strokes_convert_through_the_documents_cmyk_profile() {
    // A synthetic uncoated-like profile: heavier dot gain than the built-in coated CMYK (whose
    // params are the defaults here), so painted ink differs from the default conversion. The
    // compositors and composite exports enter the document's space; the stroke must too (#583).
    let profile = photocraft_cms::synth::cmyk_profile(&photocraft_cms::synth::CmykParams {
        description: "Stroke Test Uncoated".into(),
        tvi: [0.26, 0.26, 0.26, 0.3],
        grid_a2b: 5,
        grid_b2a: 9,
        ..Default::default()
    })
    .to_bytes();
    let cmyk_session = |tagged: bool| {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 32, "height": 32, "mode": "cmyk", "background": "white"})).unwrap();
        if tagged {
            s.edit("icc", |doc, _| {
                doc.icc_profile = Some(profile.clone());
                Ok(())
            })
            .unwrap();
        }
        s
    };
    let params = json!({"points": [[4, 16], [28, 16]], "size": 8, "hardness": 1, "opacity": 1, "flow": 1, "color": "#cc3366", "smoothing": 0});
    let mut s = cmyk_session(true);
    // The live preview enters the same space, so it commits the pixels it shows.
    let mut live =
        LiveStroke::begin(&s, &json!({"points": [[4, 16]], "size": 8, "hardness": 1, "opacity": 1, "flow": 1, "color": "#cc3366", "smoothing": 0})).unwrap();
    live.push(&[StrokePoint::new(28.0, 16.0, 1.0)]).unwrap();
    let mut commit = params.clone();
    commit["seed"] = json!(live.seed);
    s.execute("paint.stroke", commit).unwrap();
    let shown = live.doc.layer(s.active().unwrap().active_layer.unwrap()).unwrap().surface().unwrap().clone();
    assert!(same_pixels(&shown, &surface(&s), Rect::new(0, 0, 32, 32)), "the preview commits the pixels it shows");
    // The stroke's ink differs from the same stroke without the embedded profile.
    let ink = |surf: &Surface| surf.read_region(Rect::new(15, 16, 17, 18));
    let tagged = ink(&surface(&s));
    let mut plain = cmyk_session(false);
    plain.execute("paint.stroke", params).unwrap();
    let untagged = ink(&surface(&plain));
    assert!(tagged != untagged, "the document's CMYK profile must convert the painted ink: {tagged:?} vs {untagged:?}");
}

/// A live stroke over `pts` (pushed in chunks, as the canvas does) and its commit params.
fn live_and_commit(s: &Session, cmd: &str, base: Value, pts: &[[f64; 3]]) -> (LiveStroke, Value) {
    let mut p = base;
    p["points"] = json!([pts[0]]);
    let mut live = LiveStroke::begin_with(s, cmd, &p).unwrap();
    for c in pts[1..].chunks(2) {
        let sp: Vec<StrokePoint> = c.iter().map(|q| StrokePoint::new(q[0], q[1], q[2] as f32)).collect();
        live.push(&sp).unwrap();
    }
    p["points"] = json!(pts);
    p["seed"] = json!(live.seed);
    (live, p)
}

#[test]
fn a_commit_takes_the_live_strokes_pixels_and_they_equal_a_fresh_render() {
    let pts = [[10.0, 50.0, 1.0], [60.0, 30.0, 0.7], [120.0, 60.0, 0.9], [180.0, 40.0, 0.4], [150.0, 80.0, 1.0]];
    for (cmd, base) in [
        ("paint.stroke", json!({"brush": {"size": 18, "hardness": 0.0, "flow": 0.5}, "smoothing": 0.5, "target": "pixels"})),
        ("paint.stroke", json!({"preset": "Spatter", "smoothing": 0.0, "target": "pixels"})),
        ("paint.stroke", json!({"brush": {"size": 12}, "erase": true, "target": "pixels"})),
        ("paint.pencil", json!({"smoothing": 0.0})),
    ] {
        let mut s = session(200, 100);
        let (live, p) = live_and_commit(&s, cmd, base.clone(), &pts);
        let steps = s.active().unwrap().history.past_len();
        s.offer_live_stroke(live).unwrap();
        let r = s.execute(cmd, p.clone()).unwrap();
        s.clear_live_stroke();
        assert_eq!(r["fromLive"], true, "{cmd} {base}: the commit took the live pixels");
        assert_eq!(s.active().unwrap().history.past_len(), steps + 1, "one undo step");
        // The journal's replay renders from scratch and must give the same pixels.
        let mut fresh = session(200, 100);
        let r = fresh.execute(cmd, p).unwrap();
        assert!(r.get("fromLive").is_none());
        assert!(same_pixels(&surface(&s), &surface(&fresh), Rect::new(0, 0, 200, 100)), "{cmd} {base}");
        // Undo restores the pre-stroke pixels.
        s.undo();
        assert!(same_pixels(&surface(&s), &surface(&session(200, 100)), Rect::new(0, 0, 200, 100)));
    }
}

#[test]
fn an_offer_that_does_not_match_the_commit_is_ignored() {
    let pts = [[10.0, 50.0, 1.0], [60.0, 30.0, 0.7], [120.0, 60.0, 0.9]];
    let base = json!({"brush": {"size": 10}, "smoothing": 0.0, "target": "pixels"});
    // Different points.
    let mut s = session(200, 100);
    let (live, mut p) = live_and_commit(&s, "paint.stroke", base.clone(), &pts);
    p["points"] = json!([[10.0, 50.0, 1.0], [190.0, 90.0, 1.0]]);
    s.offer_live_stroke(live).unwrap();
    let r = s.execute("paint.stroke", p.clone()).unwrap();
    s.clear_live_stroke();
    assert!(r.get("fromLive").is_none());
    let mut fresh = session(200, 100);
    fresh.execute("paint.stroke", p).unwrap();
    assert!(same_pixels(&surface(&s), &surface(&fresh), Rect::new(0, 0, 200, 100)));
    // The document changed after the stroke began.
    let mut s = session(200, 100);
    let (live, p) = live_and_commit(&s, "paint.stroke", base.clone(), &pts);
    s.execute("layer.new.layer", json!({})).unwrap();
    s.offer_live_stroke(live).unwrap();
    let r = s.execute("paint.stroke", p).unwrap();
    s.clear_live_stroke();
    assert!(r.get("fromLive").is_none(), "a stale offer is never used");
    // Another command leaves an offer untaken, and a later unrelated stroke can't use it.
    let mut s = session(200, 100);
    let (live, _) = live_and_commit(&s, "paint.stroke", base.clone(), &pts);
    s.offer_live_stroke(live).unwrap();
    let mut other = base;
    other["points"] = json!([[5.0, 5.0, 1.0], [40.0, 5.0, 1.0]]);
    let r = s.execute("paint.stroke", other).unwrap();
    assert!(r.get("fromLive").is_none());
}

#[test]
fn last_brush_is_remembered_when_the_revision_moves_only() {
    let mut s = session(8, 8);
    restore_brush(&mut s);
    // A restore is already in the preferences: nothing to write, no clone or serialise.
    s.prefs.edit(|p| p.last_brush = json!("sentinel"));
    remember_brush(&mut s);
    assert_eq!(s.prefs().last_brush, json!("sentinel"), "an unchanged brush must not serialise");
    s.execute("tools.setBrush", json!({"size": 23})).unwrap();
    assert!(s.tools.brush_rev > s.tools.brush_saved_rev);
    remember_brush(&mut s);
    assert_eq!(s.prefs().last_brush["brush"]["size"].as_f64(), Some(23.0));
    assert_eq!(s.tools.brush_saved_rev, s.tools.brush_rev, "the revision is marked saved");
    // Revision unchanged: even a hand-written value stays (the gate is a counter compare).
    s.prefs.edit(|p| p.last_brush = json!("sentinel"));
    remember_brush(&mut s);
    assert_eq!(s.prefs().last_brush, json!("sentinel"));
}

#[test]
fn rejected_brush_changes_do_not_move_the_revision() {
    let mut s = session(8, 8);
    let rev = s.tools.brush_rev;
    assert!(s.execute("tools.setBrush", json!({"size": "big"})).is_err());
    assert_eq!(s.tools.brush_rev, rev, "a rejected brush leaves no change behind");
    s.execute("tools.setBrush", json!({"size": 31})).unwrap();
    assert!(s.tools.brush_rev > rev);
}

#[test]
fn the_revision_tracks_the_picked_preset_too() {
    let mut s = session(8, 8);
    s.execute("tools.setBrush", json!({"preset": "Hard Round"})).unwrap();
    let rev = s.tools.brush_rev;
    s.execute("brush.presets.setCurrent", json!({"name": "Soft Round"})).unwrap();
    assert!(s.tools.brush_rev > rev, "the remembered preset name follows the picked preset");
    remember_brush(&mut s);
    assert_eq!(s.prefs().last_brush["preset"].as_str(), Some("Soft Round"));
}

#[test]
fn the_remembered_brush_restores_at_launch() {
    let mut s = session(8, 8);
    s.execute("tools.setBrush", json!({"preset": "Hard Round", "size": 44, "hardness": 0.3})).unwrap();
    remember_brush(&mut s);
    let stored = s.prefs().last_brush.clone();
    assert_eq!(stored["preset"].as_str(), Some("Hard Round"));
    let mut t = Session::new();
    assert_ne!(t.tools.brush.size, 44.0, "a fresh session starts on the default brush");
    t.prefs.edit(|p| p.last_brush = stored);
    restore_brush(&mut t);
    assert_eq!(t.tools.brush.size, 44.0);
    assert_eq!(t.tools.brush.hardness, 0.3);
    assert_eq!(t.tools.current_preset.as_deref(), Some("Hard Round"));
    assert_eq!(t.tools.brush_saved_rev, t.tools.brush_rev, "the restored value is already stored");
    // A malformed remembered brush keeps the default (never an error, never a panic).
    let mut bad = Session::new();
    bad.prefs.edit(|p| p.last_brush = json!({"brush": {"size": "huge"}}));
    restore_brush(&mut bad);
    assert_eq!(bad.tools.brush.size, Session::new().tools.brush.size);
}

#[test]
fn remembered_brushes_never_embed_large_bitmaps() {
    use std::sync::Arc;
    let big = GrayTile::from_fn(PREVIEW_SIDE + 1, 8, |_, _| 1.0);
    assert!(ref_tip(&TipShape::Sampled(big)).is_none(), "a big sampled tip cannot be embedded");
    let small = GrayTile::from_fn(8, 8, |_, _| 1.0);
    assert!(ref_tip(&TipShape::Sampled(small.clone())).is_some(), "small tiles embed");
    let stored =
        StoredTile { key: "k".into(), width: 300, height: 200, preview: Arc::new(small), full: Some(Arc::new(GrayTile::from_fn(300, 200, |_, _| 1.0))) };
    let t = ref_tip(&TipShape::Stored(stored)).expect("stored tips are remembered as references");
    assert!(matches!(t, TipShape::Stored(r) if r.full.is_none()), "a loaded full bitmap is dropped");
    assert!(ref_pattern(&Pattern::Tile(GrayTile::from_fn(PREVIEW_SIDE + 1, 4, |_, _| 1.0))).is_none());
}

/// Photoshop's Clear painting mode (#3439): a Brush or Pencil stroke in `"mode": "clear"` erases
/// with that brush, and History still names the tool rather than the Eraser.
#[test]
fn clear_mode_erases_with_the_brush_or_pencil() {
    for (cmd, label) in [("paint.stroke", "Brush Tool"), ("paint.pencil", "Pencil")] {
        let mut s = session(40, 20);
        s.execute("edit.fill", json!({"color": "#ff0000"})).unwrap();
        s.execute(cmd, json!({"points": [[2, 10], [38, 10]], "size": 6, "mode": "clear", "brush": {"hardness": 1.0}})).unwrap();
        assert_eq!(rgba(&s, 20, 10)[3], 0.0, "{cmd}: cleared to transparency");
        assert_eq!(rgba(&s, 20, 2), [1.0, 0.0, 0.0, 1.0], "{cmd}: untouched pixels stay");
        assert_eq!(s.active().unwrap().history.undo_label(), Some(label), "{cmd}");
    }
}

/// #4344: a foreground picked as CMYK 100/0/0/0 is painted into a CMYK document as those inks by
/// the Brush and the Pencil, not as whatever its sRGB stand-in converts back to (100/28/0/17).
#[test]
fn brush_and_pencil_paint_a_cmyk_pick_with_the_picked_inks() {
    let cyan = [1.0f32, 0.0, 0.0, 0.0];
    for cmd in ["paint.stroke", "paint.pencil"] {
        let mut s = session(16, 16);
        s.execute("image.mode.cmyk", json!({})).unwrap();
        let rgb = photocraft_color::convert::cmyk_to_rgb(cyan).map(|c| c.clamp(0.0, 1.0));
        s.execute("tools.setColors", json!({"foreground": rgb, "foregroundCmyk": cyan})).unwrap();
        s.execute(cmd, json!({"points": [[8.0, 8.0]], "size": 6.0, "hardness": 1.0})).unwrap();
        let st = s.active().unwrap();
        let surf = st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap();
        let got: Vec<f32> = surf.read_region(Rect::new(8, 8, 9, 9)).into_iter().take(4).collect();
        assert!(got.iter().zip(cyan).all(|(a, b)| (a - b).abs() < 0.02), "{cmd}: {got:?}");
    }
}
