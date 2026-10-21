use super::*;
use photocraft_color::SampleType;

fn session_depth(depth: u32) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 40, "height": 20, "depth": depth})).unwrap();
    s
}

fn session() -> Session {
    session_depth(8)
}

fn doc(s: &Session) -> &Document {
    &s.active().unwrap().doc
}

fn chan(s: &Session, i: usize, x: i32, y: i32) -> f32 {
    doc(s).channels[i].surface.sample_channel(x, y, 0)
}

fn sel_at(s: &Session, x: i32, y: i32) -> f32 {
    doc(s).selection.as_ref().map_or(0.0, |m| m.sample_channel(x, y, 0))
}

fn px(s: &Session, x: i32, y: i32) -> Vec<f32> {
    let d = s.active().unwrap();
    d.doc.layer(d.active_layer.unwrap()).unwrap().surface().unwrap().pixel(x, y)
}

fn rect(s: &mut Session, x: i32, y: i32, w: u32, h: u32, mode: &str) {
    s.execute("select.rect", json!({"x": x, "y": y, "width": w, "height": h, "mode": mode})).unwrap();
}

#[test]
fn specs_are_documented_and_registered() {
    for sp in specs() {
        assert!(crate::commands::find(sp.id).is_some(), "{} not registered", sp.id);
        assert!(sp.params.starts_with('{'), "{}", sp.id);
    }
}

#[test]
fn save_selection_new_and_operations() {
    for depth in [8, 16, 32] {
        let mut s = session_depth(depth);
        assert!(s.execute("select.saveSelection", json!({})).is_err(), "needs a selection");
        rect(&mut s, 0, 0, 10, 20, "replace");
        let r = s.execute("select.saveSelection", json!({})).unwrap();
        assert_eq!(r["channel"], 0);
        assert_eq!(r["name"], "Alpha 1");
        let d = doc(&s);
        assert_eq!(d.channels[0].surface.format(), channel_format(d));
        assert_eq!(d.channels[0].surface.format().sample, SampleType::ALL[[8, 16, 32].iter().position(|v| *v == depth).unwrap()]);
        assert_eq!(chan(&s, 0, 5, 5), 1.0);
        assert_eq!(chan(&s, 0, 15, 5), 0.0);

        // add / subtract / intersect / replace into the existing channel
        rect(&mut s, 5, 0, 10, 20, "replace");
        s.execute("select.saveSelection", json!({"channel": 0, "operation": "add"})).unwrap();
        assert_eq!((chan(&s, 0, 2, 5), chan(&s, 0, 12, 5), chan(&s, 0, 20, 5)), (1.0, 1.0, 0.0));
        rect(&mut s, 0, 0, 3, 20, "replace");
        s.execute("select.saveSelection", json!({"channel": 0, "operation": "subtract"})).unwrap();
        assert_eq!((chan(&s, 0, 1, 5), chan(&s, 0, 5, 5)), (0.0, 1.0));
        rect(&mut s, 10, 0, 30, 20, "replace");
        s.execute("select.saveSelection", json!({"channel": "Alpha 1", "operation": "intersect"})).unwrap();
        assert_eq!((chan(&s, 0, 5, 5), chan(&s, 0, 12, 5)), (0.0, 1.0));
        s.execute("select.saveSelection", json!({"channel": 0, "operation": "replace", "name": "Mine"})).unwrap();
        assert_eq!((chan(&s, 0, 5, 5), chan(&s, 0, 30, 5)), (0.0, 1.0));
        assert_eq!(doc(&s).channels[0].name, "Mine");
        assert!(s.execute("select.saveSelection", json!({"operation": "bogus"})).is_err());
        assert!(s.execute("select.saveSelection", json!({"channel": 7, "operation": "add"})).is_err());
    }
}

#[test]
fn load_selection_operations_and_invert() {
    let mut s = session();
    rect(&mut s, 0, 0, 20, 20, "replace");
    s.execute("select.saveSelection", json!({})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    s.execute("select.loadSelection", json!({"channel": 0})).unwrap();
    assert_eq!((sel_at(&s, 5, 5), sel_at(&s, 25, 5)), (1.0, 0.0));
    s.execute("select.loadSelection", json!({"channel": 0, "invert": true})).unwrap();
    assert_eq!((sel_at(&s, 5, 5), sel_at(&s, 25, 5)), (0.0, 1.0));
    rect(&mut s, 10, 0, 20, 20, "replace");
    s.execute("select.loadSelection", json!({"channel": 0, "operation": "add"})).unwrap();
    assert_eq!((sel_at(&s, 5, 5), sel_at(&s, 25, 5), sel_at(&s, 35, 5)), (1.0, 1.0, 0.0));
    rect(&mut s, 10, 0, 20, 20, "replace");
    s.execute("select.loadSelection", json!({"channel": 0, "operation": "subtract"})).unwrap();
    assert_eq!((sel_at(&s, 15, 5), sel_at(&s, 25, 5)), (0.0, 1.0));
    rect(&mut s, 10, 0, 20, 20, "replace");
    s.execute("select.loadSelection", json!({"channel": "Alpha 1", "operation": "intersect"})).unwrap();
    assert_eq!((sel_at(&s, 15, 5), sel_at(&s, 25, 5)), (1.0, 0.0));
    assert!(s.execute("select.loadSelection", json!({})).is_err());
    assert!(s.execute("select.loadSelection", json!({"channel": 0, "operation": "xor"})).is_err());
    // undo restores the previous selection
    s.undo();
    assert_eq!(sel_at(&s, 25, 5), 1.0);
}

#[test]
fn load_transparency_mask_and_composite() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 20, "height": 10, "background": "transparent"})).unwrap();
    rect(&mut s, 0, 0, 5, 10, "replace");
    s.execute("edit.fill", json!({"color": "#ffffff"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    s.execute("select.loadSelection", json!({"channel": "transparency"})).unwrap();
    assert_eq!((sel_at(&s, 2, 2), sel_at(&s, 10, 2)), (1.0, 0.0));
    // Composite luminosity: white = selected; transparent over nothing = black.
    s.execute("select.loadSelection", json!({"channel": "composite"})).unwrap();
    assert!(sel_at(&s, 2, 2) > 0.99);
    s.execute("select.loadSelection", json!({"channel": "red"})).unwrap();
    assert!(sel_at(&s, 2, 2) > 0.99);
    // Layer mask
    assert!(s.execute("select.loadSelection", json!({"channel": "mask"})).is_err());
    s.execute("layer.layerMask.hideAll", json!({})).unwrap();
    s.execute("select.loadSelection", json!({"channel": "mask", "invert": true})).unwrap();
    assert_eq!(sel_at(&s, 10, 5), 1.0);
}

#[test]
fn channel_management_and_view_state() {
    let mut s = session();
    let r = s.execute("channel.new", json!({"fill": "white", "name": "W"})).unwrap();
    assert_eq!(r["channel"], 0);
    assert_eq!(chan(&s, 0, 3, 3), 1.0);
    // A new channel is targeted and shown alone.
    let st = s.active().unwrap();
    assert_eq!(st.channel_view.target, ChannelTarget::Alpha(0));
    assert!(st.channel_view.alpha_shown(0) && st.channel_view.visible_colors(3) == 0);
    s.execute("channel.new", json!({})).unwrap();
    assert_eq!(doc(&s).channels[1].name, "Alpha 1");
    s.execute("channel.duplicate", json!({"channel": 0})).unwrap();
    assert_eq!(doc(&s).channels[2].name, "W copy");
    s.execute("channel.duplicate", json!({"channel": "red", "invert": true})).unwrap();
    assert_eq!(doc(&s).channels[3].name, "Red copy");
    assert_eq!(chan(&s, 3, 1, 1), 0.0, "white image, inverted");
    s.execute("channel.rename", json!({"channel": 1, "name": "Renamed"})).unwrap();
    assert!(s.execute("channel.rename", json!({"channel": 1, "name": " "})).is_err());
    s.execute("channel.move", json!({"channel": "Renamed", "to": 0})).unwrap();
    let names: Vec<_> = doc(&s).channels.iter().map(|c| c.name.clone()).collect();
    assert_eq!(names, ["Renamed", "W", "W copy", "Red copy"]);
    s.execute("channel.target", json!({"channel": 1})).unwrap();
    s.execute("channel.delete", json!({"channel": 0})).unwrap();
    assert_eq!(s.active().unwrap().channel_view.target, ChannelTarget::Alpha(0), "target follows its channel");
    s.execute("channel.delete", json!({})).unwrap();
    assert_eq!(s.active().unwrap().channel_view.target, ChannelTarget::Composite);
    assert_eq!(doc(&s).channels.len(), 2);
    // Visibility toggles (eye) for colour, composite, alpha.
    s.execute("channel.setVisible", json!({"channel": "blue", "visible": false})).unwrap();
    assert!(!s.active().unwrap().channel_view.color_visible(2));
    s.execute("channel.setVisible", json!({"channel": "composite"})).unwrap();
    assert_eq!(s.active().unwrap().channel_view.visible_colors(3), 3);
    s.execute("channel.setVisible", json!({"channel": 1})).unwrap();
    assert!(s.active().unwrap().channel_view.alpha_shown(1));
    assert!(!s.active().unwrap().channel_view.is_plain(doc(&s)));
    // Viewing state doesn't dirty or add history.
    let hist = s.active().unwrap().history.entries().len();
    s.execute("channel.target", json!({"channel": "green"})).unwrap();
    assert_eq!(s.active().unwrap().history.entries().len(), hist);
    let j = s.execute("channel.list", json!({})).unwrap();
    assert_eq!(j["target"]["kind"], "color");
    assert_eq!(j["alpha"].as_array().unwrap().len(), 2);
    // Undo of the delete brings the channel back; the view stays valid.
    s.undo();
    assert_eq!(doc(&s).channels.len(), 3);
}

#[test]
fn deleting_an_earlier_channel_keeps_the_last_channel_targeted() {
    for depth in [8, 16, 32] {
        let mut s = session_depth(depth);
        s.execute("channel.new", json!({"name": "Remove", "fill": "black"})).unwrap();
        s.execute("channel.new", json!({"name": "Keep", "fill": "white"})).unwrap();
        let layer_before = px(&s, 5, 5);
        s.execute("channel.delete", json!({"channel": "Remove"})).unwrap();
        assert_eq!(doc(&s).channels[0].name, "Keep");
        let view = &s.active().unwrap().channel_view;
        assert_eq!(view.target, ChannelTarget::Alpha(0), "the surviving target follows its new index");
        assert!(view.alpha_shown(0));
        assert_eq!(view.visible_colors(3), 0, "keep the channel-only view");

        s.execute("image.adjustments.invert", json!({})).unwrap();
        assert_eq!(chan(&s, 0, 5, 5), 0.0, "implicit pixel edits still target Keep");
        assert_eq!(px(&s, 5, 5), layer_before, "the layer must not be edited instead");
        assert!(s.undo());
        assert_eq!(chan(&s, 0, 5, 5), 1.0);
        assert_eq!(px(&s, 5, 5), layer_before);
        assert!(s.redo());
        assert_eq!(chan(&s, 0, 5, 5), 0.0);
        assert_eq!(px(&s, 5, 5), layer_before);
    }
}

#[test]
fn deleting_a_channel_remaps_all_surviving_eyes_and_targets() {
    for target_index in 0..3 {
        let mut s = session();
        for name in ["First", "Middle", "Last"] {
            s.execute("channel.new", json!({"name": name})).unwrap();
        }
        s.execute("channel.target", json!({"channel": target_index})).unwrap();
        for (channel, visible) in [false, false, true].into_iter().enumerate() {
            s.execute("channel.setVisible", json!({"channel": channel, "visible": visible})).unwrap();
        }
        s.execute("channel.delete", json!({"channel": "Middle"})).unwrap();
        let view = &s.active().unwrap().channel_view;
        let expected = match target_index {
            0 => ChannelTarget::Alpha(0),
            1 => ChannelTarget::Composite,
            _ => ChannelTarget::Alpha(1),
        };
        assert_eq!(view.target, expected, "old target {target_index}");
        assert!(!view.alpha_shown(0));
        assert!(view.alpha_shown(1), "Last keeps its eye when Middle is removed");
    }
}

#[test]
fn targeting_the_composite_hides_the_alpha_overlay() {
    for (cmd, params) in [("channel.target", json!({"channel": "composite"})), ("channel.target.composite", json!({}))] {
        let mut s = session();
        s.execute("channel.new", json!({})).unwrap();
        assert!(s.active().unwrap().channel_view.alpha_shown(0));
        s.execute(cmd, params).unwrap();
        let st = s.active().unwrap();
        assert_eq!(st.channel_view.visible_colors(3), 3, "{cmd}");
        assert!(!st.channel_view.alpha_shown(0), "{cmd}: the alpha eye turns off");
        assert!(st.channel_view.is_plain(&st.doc), "{cmd}: no overlay left on the image (#4538)");
    }
}

#[test]
fn every_colour_channel_can_hide_while_alpha_stays() {
    let mut s = session();
    s.execute("channel.new", json!({"fill": "white", "name": "Alpha"})).unwrap();
    // A new channel is shown alone. Bring RGB back (the overlay), then turn the RGB eye off.
    s.execute("channel.setVisible", json!({"channel": "composite", "visible": true})).unwrap();
    assert!(s.active().unwrap().channel_view.alpha_shown(0));
    assert_eq!(s.active().unwrap().channel_view.visible_colors(3), 3);
    s.execute("channel.setVisible", json!({"channel": "composite", "visible": false})).unwrap();
    let st = s.active().unwrap();
    assert_eq!(st.channel_view.visible_colors(3), 0, "the RGB eye can turn R, G and B off together");
    assert!(st.channel_view.alpha_shown(0), "Alpha stays on");
    s.execute("channel.setVisible", json!({"channel": "composite", "visible": true})).unwrap();
    for name in ["red", "green", "blue"] {
        s.execute("channel.setVisible", json!({"channel": name, "visible": false})).unwrap();
    }
    let st = s.active().unwrap();
    assert_eq!(st.channel_view.visible_colors(3), 0, "R, G and B can each be turned off");
    assert!(st.channel_view.alpha_shown(0));
}

#[test]
fn quick_mask_and_a_layer_mask_can_be_the_only_visible_channel() {
    let mut s = session();
    s.execute("edit.fill", json!({"color": "#ff0000"})).unwrap();
    s.execute("select.editInQuickMaskMode", json!({})).unwrap();
    s.execute("channel.setVisible", json!({"channel": "composite", "visible": false})).unwrap();
    assert_eq!(s.active().unwrap().channel_view.visible_colors(3), 0);
    assert!(s.active().unwrap().doc.quick_mask.is_some());
    s.execute("select.editInQuickMaskMode", json!({"on": false})).unwrap();
    // Nothing else to show: the colour channels come back.
    s.execute("channel.setVisible", json!({"channel": "composite", "visible": false})).unwrap();
    assert_eq!(s.active().unwrap().channel_view.visible_colors(3), 3);

    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("edit.fill", json!({"color": "#ff0000"})).unwrap();
    s.execute("layer.layerMask.hideAll", json!({})).unwrap();
    for name in ["red", "green", "blue"] {
        s.execute("channel.setVisible", json!({"channel": name, "visible": false})).unwrap();
    }
    assert_eq!(s.active().unwrap().channel_view.visible_colors(3), 0, "a transparency mask can be shown alone");
}

#[test]
fn shortcut_slots_target_channels() {
    let mut s = session();
    s.execute("channel.new", json!({})).unwrap();
    s.execute("channel.target.slot3", json!({})).unwrap();
    assert_eq!(s.active().unwrap().channel_view.target, ChannelTarget::Color(0));
    s.execute("channel.target.slot6", json!({})).unwrap();
    assert_eq!(s.active().unwrap().channel_view.target, ChannelTarget::Alpha(0));
    assert!(!s.is_enabled("channel.target.slot7"));
    // The reason names the shortcut in the platform's notation (⌘7 on the Mac, Ctrl+7 elsewhere).
    let err = s.execute("channel.target.slot7", json!({})).unwrap_err().to_string();
    assert!(err.contains(if cfg!(target_os = "macos") { "no channel for ⌘7" } else { "no channel for Ctrl+7" }), "{err}");
    s.execute("channel.target.composite", json!({})).unwrap();
    assert_eq!(s.active().unwrap().channel_view.target, ChannelTarget::Composite);
    // Grayscale: ⌘3 is the first alpha channel.
    let mut g = Session::new();
    g.execute("file.new", json!({"width": 4, "height": 4, "mode": "gray"})).unwrap();
    g.execute("channel.new", json!({})).unwrap();
    g.execute("channel.target.slot3", json!({})).unwrap();
    assert_eq!(g.active().unwrap().channel_view.target, ChannelTarget::Alpha(0));
}

#[test]
fn painting_filtering_and_adjusting_into_an_alpha_channel() {
    for depth in [8, 16, 32] {
        let mut s = session_depth(depth);
        s.execute("channel.new", json!({})).unwrap();
        let before = px(&s, 10, 10);
        // Targeted channel: the stroke goes into the channel, not the layer.
        s.execute("paint.stroke", json!({"points": [[10, 10]], "size": 8, "hardness": 1.0, "color": "#ffffff"})).unwrap();
        assert!(chan(&s, 0, 10, 10) > 0.99, "depth {depth}");
        assert_eq!(chan(&s, 0, 30, 10), 0.0);
        assert_eq!(px(&s, 10, 10), before);
        // Erasing paints the background colour (white) on channels; use black foreground fill.
        s.execute("edit.fill", json!({"color": "#808080"})).unwrap();
        assert!((chan(&s, 0, 30, 10) - 0.5).abs() < 0.01);
        // Adjustments apply to the channel (also where it was untouched).
        s.execute("image.adjustments.invert", json!({})).unwrap();
        assert!((chan(&s, 0, 30, 10) - 0.5).abs() < 0.01);
        assert_eq!(px(&s, 10, 10), before, "layer untouched by the adjustment");
        // Filters too, limited by the selection.
        s.execute("channel.new", json!({})).unwrap();
        s.execute("paint.stroke", json!({"points": [[20, 10]], "size": 6, "hardness": 1.0, "color": "#ffffff"})).unwrap();
        rect(&mut s, 0, 0, 20, 20, "replace");
        s.execute("filter.blur.gaussianBlur", json!({"radius": 3})).unwrap();
        assert!(chan(&s, 1, 16, 10) > 0.0, "blurred inside the selection");
        assert_eq!(chan(&s, 1, 26, 10), 0.0, "outside the selection untouched");
        // Explicit targets work without targeting the channel.
        s.execute("channel.target", json!({"channel": "composite"})).unwrap();
        s.execute("select.deselect", json!({})).unwrap();
        s.execute("paint.bucket", json!({"x": 1, "y": 1, "color": "#ffffff", "target": {"channel": 1}, "tolerance": 0})).unwrap();
        assert_eq!(chan(&s, 1, 1, 1), 1.0);
        assert!(s.execute("paint.stroke", json!({"points": [[1, 1]], "target": {"channel": 9}})).is_err());
        // Undo restores the channel.
        s.undo();
        assert_eq!(chan(&s, 1, 1, 1), 0.0);
    }
}

#[test]
fn single_colour_channel_target_limits_edits() {
    let mut s = session();
    s.execute("channel.target", json!({"channel": "red"})).unwrap();
    s.execute("edit.fill", json!({"color": "#000000"})).unwrap();
    assert_eq!(px(&s, 5, 5), vec![0.0, 1.0, 1.0, 1.0]);
    s.execute("channel.target", json!({"channel": "blue"})).unwrap();
    s.execute("paint.stroke", json!({"points": [[20, 10]], "size": 6, "hardness": 1.0, "color": "#000000"})).unwrap();
    assert_eq!(px(&s, 20, 10), vec![0.0, 1.0, 0.0, 1.0]);
    s.execute("channel.target", json!({"channel": "composite"})).unwrap();
    s.execute("edit.fill", json!({"color": "#000000"})).unwrap();
    assert_eq!(px(&s, 5, 5), vec![0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn quick_mask_enter_paint_exit() {
    let mut s = session();
    rect(&mut s, 0, 0, 20, 20, "replace");
    s.execute("select.editInQuickMaskMode", json!({})).unwrap();
    let d = doc(&s);
    let q = d.quick_mask.as_ref().unwrap();
    assert!(d.selection.is_none());
    assert_eq!((q.surface.sample_channel(5, 5, 0), q.surface.sample_channel(30, 5, 0)), (1.0, 0.0));
    assert_eq!((q.opacity, q.indicates), (0.5, ColorIndicates::MaskedAreas));
    // Painting black masks; white unmasks. The layer is untouched.
    let before = px(&s, 5, 5);
    s.execute("paint.stroke", json!({"points": [[5, 5]], "size": 4, "hardness": 1.0, "color": "#000000"})).unwrap();
    s.execute("paint.stroke", json!({"points": [[30, 10]], "size": 4, "hardness": 1.0, "color": "#ffffff"})).unwrap();
    assert_eq!(px(&s, 5, 5), before);
    s.execute("select.editInQuickMaskMode", json!({})).unwrap();
    assert!(doc(&s).quick_mask.is_none());
    assert_eq!(sel_at(&s, 5, 5), 0.0, "painted black: masked");
    assert_eq!(sel_at(&s, 12, 12), 1.0);
    assert_eq!(sel_at(&s, 30, 10), 1.0, "painted white: selected");
    assert_eq!(sel_at(&s, 35, 2), 0.0);
    // History: enter and exit are steps; undo goes back into Quick Mask with the paint.
    let labels: Vec<String> = s.active().unwrap().history.entries().iter().map(|e| e.to_string()).collect();
    assert!(labels.iter().any(|l| l.contains("Edit in Quick Mask")) && labels.iter().any(|l| l.contains("Exit Quick Mask")), "{labels:?}");
    s.undo();
    assert!(doc(&s).quick_mask.is_some());
    assert!(doc(&s).selection.is_none());
    s.redo();
    assert!(doc(&s).quick_mask.is_none());
    assert_eq!(sel_at(&s, 30, 10), 1.0);
}

#[test]
fn quick_mask_without_selection_and_options() {
    let mut s = session();
    s.execute("channel.options", json!({"channel": "quickMask", "color": "#0000ff", "opacity": 30})).unwrap();
    s.execute("select.editInQuickMaskMode", json!({"on": true})).unwrap();
    let q = doc(&s).quick_mask.clone().unwrap();
    assert_eq!(q.color, Color::rgb(0.0, 0.0, 1.0));
    assert!((q.opacity - 0.3).abs() < 1e-6);
    // Leaving an untouched Quick Mask leaves no selection.
    s.execute("select.editInQuickMaskMode", json!({"on": false})).unwrap();
    assert!(doc(&s).selection.is_none());
    // A filter in Quick Mask edits the mask.
    s.execute("select.editInQuickMaskMode", json!({})).unwrap();
    s.execute("paint.stroke", json!({"points": [[10, 10]], "size": 12, "hardness": 1.0, "color": "#000000"})).unwrap();
    s.execute("filter.blur.gaussianBlur", json!({"radius": 2})).unwrap();
    let v = doc(&s).quick_mask.as_ref().unwrap().surface.sample_channel(16, 10, 0);
    assert!(v > 0.0 && v < 1.0, "feathered edge {v}");
    s.execute("select.editInQuickMaskMode", json!({})).unwrap();
    assert!(sel_at(&s, 10, 10) < 0.05 && sel_at(&s, 30, 10) > 0.99, "{} {}", sel_at(&s, 10, 10), sel_at(&s, 30, 10));
}

/// Overlay coverage of the live Quick Mask at (x, y): how much colour the canvas shows there.
fn qm_overlay(s: &Session, x: i32, y: i32) -> f32 {
    let q = doc(s).quick_mask.as_ref().unwrap();
    q.overlay_coverage(q.surface.sample_channel(x, y, 0))
}

#[test]
fn quick_mask_without_selection_starts_empty_in_both_modes() {
    // #2743 case 1: nothing selected → nothing selected in the mask, whichever way the colour
    // reads; painting white adds to the selection that comes out.
    for depth in [8, 16, 32] {
        // Selected Areas set before entering.
        let mut s = session_depth(depth);
        s.execute("channel.options", json!({"channel": "quickMask", "indicates": "selected"})).unwrap();
        s.execute("select.editInQuickMaskMode", json!({"on": true})).unwrap();
        assert_eq!((qm_overlay(&s, 5, 5), qm_overlay(&s, 35, 15)), (0.0, 0.0), "depth {depth}: no overlay");
        s.execute("select.editInQuickMaskMode", json!({"on": false})).unwrap();
        assert!(doc(&s).selection.is_none(), "depth {depth}: untouched → no selection");
        s.execute("select.editInQuickMaskMode", json!({"on": true})).unwrap();
        s.execute("paint.stroke", json!({"points": [[5, 5]], "size": 4, "hardness": 1.0, "color": "#ffffff"})).unwrap();
        assert!(qm_overlay(&s, 5, 5) > 0.99 && qm_overlay(&s, 30, 10) == 0.0, "depth {depth}");
        s.execute("select.editInQuickMaskMode", json!({"on": false})).unwrap();
        assert!(sel_at(&s, 5, 5) > 0.99 && sel_at(&s, 30, 10) == 0.0, "depth {depth}: painted area is the selection");

        // Selected Areas switched on while in Quick Mask (the order in the report).
        let mut s = session_depth(depth);
        s.execute("select.editInQuickMaskMode", json!({"on": true})).unwrap();
        assert_eq!(qm_overlay(&s, 5, 5), 0.0, "depth {depth}: Masked Areas, nothing masked yet");
        s.execute("channel.options", json!({"channel": "quickMask", "indicates": "selected"})).unwrap();
        assert_eq!((qm_overlay(&s, 5, 5), qm_overlay(&s, 35, 15)), (0.0, 0.0), "depth {depth}: still no overlay");
        // …and back again: still the untouched, nothing-selected mask.
        s.execute("channel.options", json!({"channel": "quickMask", "indicates": "masked"})).unwrap();
        assert_eq!(qm_overlay(&s, 5, 5), 0.0, "depth {depth}");
        s.execute("select.editInQuickMaskMode", json!({"on": false})).unwrap();
        assert!(doc(&s).selection.is_none(), "depth {depth}");
    }
}

#[test]
fn quick_mask_round_trip_keeps_select_all() {
    // #2743 case 2: a full selection comes back full, in either mode.
    for depth in [8, 16] {
        for indicates in ["masked", "selected"] {
            let mut s = session_depth(depth);
            s.execute("channel.options", json!({"channel": "quickMask", "indicates": indicates})).unwrap();
            s.execute("select.all", json!({})).unwrap();
            s.execute("select.editInQuickMaskMode", json!({})).unwrap();
            s.execute("select.editInQuickMaskMode", json!({})).unwrap();
            assert!(doc(&s).selection.is_some(), "depth {depth} {indicates}: Select All survives");
            assert_eq!((sel_at(&s, 0, 0), sel_at(&s, 39, 19)), (1.0, 1.0), "depth {depth} {indicates}");
            // Switching the mode in between doesn't change what is selected.
            s.execute("select.editInQuickMaskMode", json!({})).unwrap();
            let other = if indicates == "masked" { "selected" } else { "masked" };
            s.execute("channel.options", json!({"channel": "quickMask", "indicates": other})).unwrap();
            s.execute("select.editInQuickMaskMode", json!({})).unwrap();
            assert_eq!((sel_at(&s, 0, 0), sel_at(&s, 39, 19)), (1.0, 1.0), "depth {depth} {indicates}→{other}");
        }
    }
    // A no-selection entry followed by Select All entry: the second one isn't "no selection".
    let mut s = session();
    s.execute("select.editInQuickMaskMode", json!({})).unwrap();
    s.execute("select.editInQuickMaskMode", json!({})).unwrap();
    s.execute("select.all", json!({})).unwrap();
    s.execute("select.editInQuickMaskMode", json!({})).unwrap();
    s.execute("select.editInQuickMaskMode", json!({})).unwrap();
    assert_eq!(sel_at(&s, 20, 10), 1.0);
}

#[test]
fn quick_mask_odd_states_dont_panic() {
    let mut s = Session::new();
    assert!(s.execute("select.editInQuickMaskMode", json!({})).is_err(), "no document");
    let _ = s.execute("channel.options", json!({"channel": "quickMask", "indicates": "selected"}));
    let mut s = session();
    s.execute("select.editInQuickMaskMode", json!({})).unwrap();
    s.execute("select.all", json!({})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    s.execute("channel.options", json!({"channel": "quickMask", "indicates": "selected"})).unwrap();
    s.undo();
    s.undo();
    s.execute("select.editInQuickMaskMode", json!({})).unwrap();
}

#[test]
fn channel_options_and_spot_channels() {
    let mut s = session();
    rect(&mut s, 0, 0, 10, 10, "replace");
    s.execute("channel.new", json!({"fill": "selection"})).unwrap();
    s.execute("channel.options", json!({"channel": 0, "name": "Opt", "color": "#00ff00", "opacity": 25, "indicates": "selected"})).unwrap();
    let c = &doc(&s).channels[0];
    assert_eq!((c.name.as_str(), c.color, c.indicates), ("Opt", Color::rgb(0.0, 1.0, 0.0), ColorIndicates::SelectedAreas));
    assert!((c.opacity - 0.25).abs() < 1e-6);
    assert_eq!(c.overlay_coverage(1.0), 1.0);
    s.execute("channel.options", json!({"channel": 0, "indicates": "spot", "solidity": 40})).unwrap();
    assert_eq!(doc(&s).channels[0].spot.map(|x| x.1), Some(0.4));
    let r = s.execute("channel.newSpot", json!({"color": "#ff00ff", "solidity": 100})).unwrap();
    let i = r["channel"].as_u64().unwrap() as usize;
    assert_eq!(doc(&s).channels[i].name, "Spot Color 1");
    assert_eq!(chan(&s, i, 5, 5), 1.0, "from the selection");
    // Merge spot: flattens and prints the ink (solid magenta) into the image.
    s.execute("channel.mergeSpot", json!({"channel": i})).unwrap();
    let d = doc(&s);
    assert_eq!(d.layers.len(), 1);
    assert_eq!(d.channels.len(), 1);
    let p = px(&s, 5, 5);
    assert!((p[0] - 1.0).abs() < 0.01 && p[1] < 0.01 && (p[2] - 1.0).abs() < 0.01, "{p:?}");
    assert_eq!(px(&s, 20, 15)[..3], [1.0, 1.0, 1.0]);
}

#[test]
fn apply_image_hand_computed() {
    let mut s = session();
    // Target: 50% gray. Source: a second document, red #ff0000 / 50% gray.
    s.execute("edit.fill", json!({"color": "#808080"})).unwrap();
    s.execute("file.new", json!({"width": 40, "height": 20})).unwrap();
    s.execute("edit.fill", json!({"color": "#ff4000"})).unwrap();
    s.set_active(0);
    let g = 128.0 / 255.0;
    s.execute("image.applyImage", json!({"source": {"document": 1}, "blending": "multiply"})).unwrap();
    let p = px(&s, 3, 3);
    assert!((p[0] - g).abs() < 0.003 && (p[1] - g * 64.0 / 255.0).abs() < 0.003 && p[2].abs() < 0.003, "{p:?}");
    s.undo();
    // Single source channel applied to every target channel, screen at 50% opacity, inverted.
    s.execute("image.applyImage", json!({"source": {"document": 1, "channel": "green", "invert": true}, "blending": "screen", "opacity": 50})).unwrap();
    let top = 1.0 - 64.0 / 255.0;
    let screen = 1.0 - (1.0 - g) * (1.0 - top);
    let want = g + (screen - g) * 0.5;
    let p = px(&s, 3, 3);
    assert!(p[..3].iter().all(|v| (v - want).abs() < 0.003), "{p:?} vs {want}");
    s.undo();
    // Add with scale and offset; masked by an alpha channel; limited to the selection.
    s.execute("channel.new", json!({"fill": "white"})).unwrap();
    s.execute("channel.target", json!({"channel": "composite"})).unwrap();
    rect(&mut s, 0, 0, 20, 20, "replace");
    s.execute("image.applyImage", json!({"source": {"document": 1, "channel": "red"}, "blending": "add", "scale": 2, "offset": -10, "mask": {"channel": 0}}))
        .unwrap();
    let want = ((g + 1.0) / 2.0 - 10.0 / 255.0).clamp(0.0, 1.0);
    assert!((px(&s, 3, 3)[1] - want).abs() < 0.003);
    assert!((px(&s, 30, 3)[1] - g).abs() < 0.003, "outside the selection");
    // Size mismatch is refused.
    s.execute("file.new", json!({"width": 5, "height": 5})).unwrap();
    s.set_active(0);
    assert!(s.execute("image.applyImage", json!({"source": {"document": 2}})).is_err());
    assert!(s.execute("image.applyImage", json!({"blending": "hue"})).is_err());
}

#[test]
fn apply_image_subtract_layer_from_itself_is_mid_grey() {
    // Frequency separation: Subtract, Scale 2, Offset 128, in the flat form the dialog sends.
    let mut s = session();
    s.execute("edit.fill", json!({"color": "#c08040"})).unwrap();
    s.execute("layer.new.layer", json!({"name": "High"})).unwrap();
    s.execute("edit.fill", json!({"color": "#406080"})).unwrap();
    s.execute("image.applyImage", json!({"sourceLayer": "High", "blending": "subtract", "scale": 2.0, "offset": 128})).unwrap();
    let p = px(&s, 3, 3);
    assert!(p[..3].iter().all(|v| (v - 128.0 / 255.0).abs() < 0.003), "{p:?}");
}

#[test]
fn apply_image_onto_transparent_layer() {
    for depth in [8, 16, 32] {
        let mut s = session_depth(depth);
        s.execute("edit.fill", json!({"color": "#c08040"})).unwrap();
        let (r, g, b) = (192.0 / 255.0, 128.0 / 255.0, 64.0 / 255.0);
        // An empty layer has no pixels to blend with: it takes the source, whatever the mode.
        s.execute("layer.new.layer", json!({})).unwrap();
        s.execute("image.applyImage", json!({"blending": "multiply"})).unwrap();
        let p = px(&s, 3, 3);
        assert!((p[0] - r).abs() < 0.003 && (p[1] - g).abs() < 0.003 && (p[2] - b).abs() < 0.003 && p[3] > 0.999, "depth {depth}: {p:?}");
        s.undo();
        // At 50% opacity the empty layer becomes the source at 50% alpha.
        s.execute("image.applyImage", json!({"blending": "multiply", "opacity": 50})).unwrap();
        let p = px(&s, 3, 3);
        assert!((p[0] - r).abs() < 0.003 && (p[3] - 0.5).abs() < 0.003, "depth {depth}: {p:?}");
        s.undo();
        // Preserve Transparency leaves an empty layer empty.
        s.execute("image.applyImage", json!({"blending": "multiply", "preserveTransparency": true})).unwrap();
        assert!(px(&s, 3, 3)[3] < 0.001, "depth {depth}");
    }
}

#[test]
fn apply_image_into_alpha_channel() {
    let mut s = session();
    s.execute("edit.fill", json!({"color": "#ffffff"})).unwrap();
    s.execute("channel.new", json!({"fill": "white"})).unwrap();
    // Targeted channel: Normal blend of the inverted composite → black.
    s.execute("image.applyImage", json!({"source": {"channel": "composite", "invert": true}, "blending": "normal"})).unwrap();
    assert!(chan(&s, 0, 3, 3) < 0.01);
}

#[test]
fn calculations_results() {
    let mut s = session();
    rect(&mut s, 0, 0, 20, 20, "replace");
    s.execute("select.saveSelection", json!({})).unwrap();
    rect(&mut s, 10, 0, 20, 20, "replace");
    s.execute("select.saveSelection", json!({})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    // Multiply = intersection of the two masks.
    let r = s.execute("image.calculations", json!({"source1": {"channel": 0}, "source2": {"channel": 1}, "blending": "multiply", "name": "X"})).unwrap();
    let i = r["channel"].as_u64().unwrap() as usize;
    assert_eq!(doc(&s).channels[i].name, "X");
    assert_eq!((chan(&s, i, 5, 5), chan(&s, i, 15, 5), chan(&s, i, 25, 5)), (0.0, 1.0, 0.0));
    // Screen = union, at 50% opacity; source 1 inverted.
    s.execute(
        "image.calculations",
        json!({"source1": {"channel": 0, "invert": true}, "source2": {"channel": 1}, "blending": "difference", "result": "selection"}),
    )
    .unwrap();
    // |s2 - (1 - s1)|: x=5: |0-0|=0, x=15: |1-0|=1, x=35: |0-1|=1
    assert_eq!((sel_at(&s, 5, 5), sel_at(&s, 15, 5), sel_at(&s, 35, 5)), (0.0, 1.0, 1.0));
    let r = s
        .execute(
            "image.calculations",
            json!({"source1": {"channel": "composite"}, "source2": {"channel": 1}, "blending": "normal", "opacity": 50, "result": "newDocument"}),
        )
        .unwrap();
    assert_eq!(r["document"], 1);
    let d = doc(&s);
    assert_eq!(d.mode, ColorMode::Grayscale);
    let v = d.layers[0].surface().unwrap().pixel(5, 5)[0];
    assert!((v - 0.5).abs() < 0.01, "{v}");
    s.set_active(0);
    assert!(s.execute("image.calculations", json!({"source1": {"channel": 0}, "source2": {"channel": 1}, "result": "x"})).is_err());
}

#[test]
fn split_and_merge_channels() {
    let mut s = session();
    s.execute("edit.fill", json!({"color": "#ff8000"})).unwrap();
    s.execute("channel.new", json!({"fill": "white", "name": "A"})).unwrap();
    let r = s.execute("channel.split", json!({})).unwrap();
    assert_eq!(r["documents"].as_array().unwrap().len(), 4);
    assert_eq!(s.documents().len(), 4, "original closed");
    let names: Vec<_> = s.documents().iter().map(|d| d.doc.name.clone()).collect();
    assert_eq!(names, ["Untitled_R", "Untitled_G", "Untitled_B", "Untitled_A"]);
    let v = |s: &Session, i: usize| s.documents()[i].doc.layers[0].surface().unwrap().pixel(1, 1)[0];
    assert_eq!(v(&s, 0), 1.0);
    assert!((v(&s, 1) - 128.0 / 255.0).abs() < 0.003);
    assert_eq!(v(&s, 2), 0.0);
    // Merge the three colour docs back (order: active first, then the rest).
    s.set_active(0);
    let r = s.execute("channel.merge", json!({"mode": "rgb", "documents": [0, 1, 2]})).unwrap();
    assert_eq!(s.documents().len(), 2);
    assert_eq!(r["document"], 1);
    let p = px(&s, 1, 1);
    assert!((p[0] - 1.0).abs() < 0.003 && (p[1] - 128.0 / 255.0).abs() < 0.003 && p[2].abs() < 0.003, "{p:?}");
    assert!(s.execute("channel.merge", json!({"mode": "cmyk"})).is_err());
}

#[test]
fn merge_rejects_a_repeated_source_and_keeps_unrelated_documents() {
    // #934: `documents: [0, 0, 1]` passed validation, then the close loop ran `close(1)`,
    // `close(0)`, `close(0)` and the last call removed the unrelated RGB document at index 2.
    let mut s = session();
    s.execute("channel.split", json!({})).unwrap();
    assert_eq!(s.documents().len(), 3);
    s.execute("file.new", json!({"width": 40, "height": 20})).unwrap();
    s.execute("edit.fill", json!({"color": "#00ff00"})).unwrap();
    let keep = s.documents()[3].doc.id;
    let before: Vec<_> = s.documents().iter().map(|d| d.doc.id).collect();
    let err = s.execute("channel.merge", json!({"mode": "rgb", "documents": [0, 0, 1]})).unwrap_err();
    assert!(err.to_string().contains("more than once"), "{err}");
    let after: Vec<_> = s.documents().iter().map(|d| d.doc.id).collect();
    assert_eq!(after, before, "a rejected merge closes nothing");
    assert!(s.documents().iter().any(|d| d.doc.id == keep));
    assert!(s.set_active(3) && s.undo(), "the unrelated document keeps its history");
    // Distinct indices still merge and close exactly the named sources.
    s.execute("channel.merge", json!({"mode": "rgb", "documents": [0, 1, 2]})).unwrap();
    assert_eq!(s.documents().len(), 2);
    assert!(s.documents().iter().any(|d| d.doc.id == keep));
}

#[test]
fn duplicate_to_other_document() {
    let mut s = session();
    rect(&mut s, 0, 0, 5, 5, "replace");
    s.execute("select.saveSelection", json!({"name": "S"})).unwrap();
    s.execute("file.new", json!({"width": 40, "height": 20, "depth": 16})).unwrap();
    s.set_active(0);
    s.execute("channel.duplicate", json!({"channel": "S", "document": 1, "name": "S2"})).unwrap();
    let d1 = &s.documents()[1].doc;
    assert_eq!(d1.channels[0].name, "S2");
    assert_eq!(d1.channels[0].surface.format().sample, SampleType::U16);
    assert_eq!(d1.channels[0].surface.sample_channel(2, 2, 0), 1.0);
    let r = s.execute("channel.duplicate", json!({"channel": 0, "document": "new"})).unwrap();
    assert_eq!(s.documents()[r["document"].as_u64().unwrap() as usize].doc.mode, ColorMode::Grayscale);
}

#[test]
fn image_size_and_rotate_carry_channels_and_quick_mask() {
    let mut s = session();
    rect(&mut s, 0, 0, 10, 20, "replace");
    s.execute("select.saveSelection", json!({})).unwrap();
    s.execute("select.editInQuickMaskMode", json!({})).unwrap();
    s.execute("image.imageRotation.180", json!({})).unwrap();
    s.execute("image.mode.bits16", json!({})).unwrap();
    let d = doc(&s);
    assert_eq!(d.channels[0].surface.format().sample, SampleType::U16);
    assert_eq!(d.quick_mask.as_ref().unwrap().surface.format().sample, SampleType::U16);
}

#[test]
fn apply_image_and_calculations_fail_gracefully_never_panic() {
    // Rule 9: adversarial params must return Err, not panic (guards the empty-source `src[0]` /
    // `.remove(0)` paths in apply_image/calculations).
    let mut s = session();
    for p in [
        json!({}),
        json!({"source": {"channel": 999}}),
        json!({"source": {"channel": "nope"}}),
        json!({"source": {"document": 999}}),
        json!({"blending": "bogus", "opacity": -50}),
    ] {
        // Must return a Result (Ok or Err) without panicking.
        let _ = s.execute("image.applyImage", p.clone());
        let _ = s.execute("image.calculations", p);
    }
    // A well-formed self-apply still works.
    assert!(s.execute("image.applyImage", json!({"source": {"channel": "composite"}})).is_ok());
}

#[test]
fn channel_target_layer_mask_and_vector_mask() {
    let mut s = session();
    // Initially the layer has no mask: targeting mask or vectorMask fails.
    let err_mask = s.execute("channel.target", json!({"channel": "mask"})).unwrap_err().to_string();
    assert!(err_mask.contains("no layer mask"), "{err_mask}");
    let err_vec = s.execute("channel.target", json!({"channel": "vectorMask"})).unwrap_err().to_string();
    assert!(err_vec.contains("no vector mask"), "{err_vec}");

    // Add a layer mask.
    s.execute("layer.layerMask.revealAll", json!({})).unwrap();
    // Target the mask explicitly via channel.target.
    let j = s.execute("channel.target", json!({"channel": "mask"})).unwrap();
    assert_eq!(j["target"]["kind"], "layerMask");
    assert_eq!(s.active().unwrap().channel_view.target, ChannelTarget::LayerMask);
    assert!(s.tools.mask_targeted);

    // Commands that follow target (invert, paint) edit the mask without explicit target.
    s.execute("image.adjustments.invert", json!({})).unwrap();
    let l_id = s.active().unwrap().active_layer.unwrap();
    let mask_sf = &s.active().unwrap().doc.layer(l_id).unwrap().mask.as_ref().unwrap().surface;
    assert_eq!(mask_sf.pixel(10, 10)[0], 0.0, "inverted mask is black");

    // Target composite again.
    let j2 = s.execute("channel.target", json!({"channel": "composite"})).unwrap();
    assert_eq!(j2["target"]["kind"], "composite");
    assert_eq!(s.active().unwrap().channel_view.target, ChannelTarget::Composite);
    assert!(!s.tools.mask_targeted);

    // Issue #2458 repro: new adjustment layer then channel.target {"channel": "mask"}
    s.execute("layer.newAdjustmentLayer.curves", json!({})).unwrap();
    let j_adj = s.execute("channel.target", json!({"channel": "mask"})).unwrap();
    assert_eq!(j_adj["target"]["kind"], "layerMask");
    assert_eq!(s.active().unwrap().channel_view.target, ChannelTarget::LayerMask);

    // Vector mask targeting.
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("layer.vectorMask.revealAll", json!({})).unwrap();
    let j_vec = s.execute("channel.target", json!({"channel": "vectorMask"})).unwrap();
    assert_eq!(j_vec["target"]["kind"], "vectorMask");
    assert_eq!(s.active().unwrap().channel_view.target, ChannelTarget::VectorMask);
}

#[test]
fn clear_with_a_colour_channel_targeted_clears_only_that_channel() {
    let near = |a: &[f32], b: &[f32]| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-3);
    // The Background, then a normal layer (Delete there can't make one channel transparent).
    for duplicate in [false, true] {
        let mut s = session_depth(16);
        rect(&mut s, 0, 0, 20, 20, "replace");
        s.execute("edit.fill", json!({"color": "#336699"})).unwrap();
        s.execute("select.deselect", json!({})).unwrap();
        if duplicate {
            s.execute("layer.duplicate", json!({})).unwrap();
        }
        let (left, right) = (px(&s, 5, 5), px(&s, 30, 5));
        s.tools.background = [0.8, 0.1, 0.9, 1.0];
        s.execute("channel.target", json!({"channel": "red"})).unwrap();
        rect(&mut s, 2, 2, 10, 10, "replace");
        s.execute("edit.clear", json!({})).unwrap();
        // Red takes the background colour's red; green, blue and transparency stay.
        assert!(near(&px(&s, 5, 5), &[0.8, left[1], left[2], 1.0]), "duplicate {duplicate}: {:?}", px(&s, 5, 5));
        assert!(near(&px(&s, 15, 5), &left), "outside the selection");
        assert!(near(&px(&s, 30, 5), &right));
        s.undo();
        assert!(near(&px(&s, 5, 5), &left), "one undoable step");
    }
}

#[test]
fn clear_with_an_alpha_channel_or_quick_mask_targeted_fills_it_not_the_layer() {
    let mut s = session_depth(16);
    let layer = px(&s, 5, 5);
    s.tools.background = [0.0, 0.0, 0.0, 1.0];
    // A white Alpha 1, targeted.
    s.execute("channel.new", json!({"fill": "white"})).unwrap();
    rect(&mut s, 2, 2, 10, 10, "replace");
    s.execute("edit.clear", json!({})).unwrap();
    // The selected part of the channel takes the background colour; the layer stays.
    assert_eq!(chan(&s, 0, 5, 5), 0.0);
    assert_eq!(chan(&s, 0, 15, 5), 1.0, "outside the selection");
    assert_eq!(px(&s, 5, 5), layer, "the layer is untouched");
    s.undo();
    assert_eq!(chan(&s, 0, 5, 5), 1.0, "one undoable step");
    // In Quick Mask mode Delete fills the mask (black = masked), not the layer.
    s.execute("channel.target", json!({"channel": "composite"})).unwrap();
    s.execute("select.editInQuickMaskMode", json!({"on": true})).unwrap();
    s.execute("edit.clear", json!({})).unwrap();
    let q = doc(&s).quick_mask.as_ref().unwrap().surface.sample_channel(5, 5, 0);
    assert_eq!(q, 0.0);
    assert_eq!(px(&s, 5, 5), layer, "the layer is untouched in Quick Mask mode");
}

#[test]
fn command_backslash_targets_the_layer_mask_and_command_2_the_composite() {
    let mut s = session();
    s.execute("layer.new.layer", json!({})).unwrap();
    assert!(!s.is_enabled("channel.target.layerMask"), "no mask yet");
    assert!(s.execute("channel.target.layerMask", json!({})).is_err());
    s.execute("layer.layerMask.revealAll", json!({})).unwrap();
    s.execute("channel.target.composite", json!({})).unwrap();
    assert_eq!(crate::commands::find("channel.target.layerMask").and_then(|c| c.shortcut), Some("Cmd+\\"));
    s.execute("channel.target.layerMask", json!({})).unwrap();
    assert_eq!(s.active().unwrap().channel_view.target, ChannelTarget::LayerMask);
    s.execute("channel.target.composite", json!({})).unwrap();
    assert_eq!(s.active().unwrap().channel_view.target, ChannelTarget::Composite);
}

#[test]
fn command_alt_2_loads_the_images_luminosity() {
    let mut s = session();
    rect(&mut s, 0, 0, 20, 20, "replace");
    s.execute("edit.fill", json!({"contents": "black"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    assert_eq!(crate::commands::find("select.loadLuminosity").and_then(|c| c.shortcut), Some("Cmd+Alt+2"));
    s.execute("select.loadLuminosity", json!({})).unwrap();
    assert!(sel_at(&s, 5, 5) < 0.01, "black: unselected");
    assert!(sel_at(&s, 30, 5) > 0.99, "white: selected");
}

#[test]
fn thumbnail_selection_shape_without_pixel_cache_uses_its_transparency() {
    for depth in [8, 16, 32] {
        let mut s = Session::new();
        s.execute("file.new", json!({"width":100,"height":80,"depth":depth,"background":"transparent"})).unwrap();
        let id = s.execute("shape.create", json!({"kind":"rect","rect":[20,10,40,30],"fill":"#cc3333"})).unwrap()["layer"].as_u64().unwrap();
        s.edit("Drop render cache", |doc, _| {
            let l = doc.layer_mut(photocraft_doc::LayerId(id)).unwrap();
            if let LayerContent::Shape(shape) = &mut l.content {
                shape.cache = None;
            }
            l.visible = false;
            l.opacity = 0.25;
            Ok(())
        })
        .unwrap();
        s.execute("select.loadSelection", json!({"channel":"transparency","layer":id})).unwrap();
        assert!(sel_at(&s, 30, 20) > 0.99);
        assert!(sel_at(&s, 0, 0) < 0.01, "shape transparency must not select the whole canvas at depth {depth}");
        let b = s.active().unwrap().doc.selection.as_ref().unwrap().content_bounds();
        assert_eq!((b.x0, b.y0, b.width(), b.height()), (20, 10, 40, 30));
    }
}
