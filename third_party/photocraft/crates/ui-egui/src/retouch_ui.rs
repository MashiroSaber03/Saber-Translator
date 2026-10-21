//! UI for the retouching tools (healing, patch, clone, history brush, blur/sharpen/smudge,
//! dodge/burn/sponge, Mixer Brush, Color Replacement) and the smart selection tools (Quick
//! Selection, Object Selection): gesture → engine command, options bars, and the clone-source
//! marker.

use egui::{Color32, Stroke, vec2};
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::canvas::ViewXform;
use crate::state::Tool;
use crate::theme::Tokens;

/// The tool a stroke uses with ⌥ held: Dodge and Burn swap, as do Blur and Sharpen.
pub(crate) fn alt_flipped(tool: Tool, alt: bool) -> Tool {
    match (tool, alt) {
        (Tool::Dodge, true) => Tool::Burn,
        (Tool::Burn, true) => Tool::Dodge,
        (Tool::Blur, true) => Tool::Sharpen,
        (Tool::Sharpen, true) => Tool::Blur,
        _ => tool,
    }
}

/// Clone Stamp / Healing Brush params for the options bar and the clone source (without
/// `points` and `target`): shared by the live preview and the commit. `None` when no source
/// point is set yet.
pub(crate) fn clone_params(app: &PhotocraftApp) -> Option<Value> {
    let o = &app.ui.tool_options;
    let mut p = json!({"aligned": o.clone_aligned, "sampleLayer": o.clone_sample});
    if app.ui.tool == Tool::CloneStamp {
        p["wrap"] = json!(app.ui.view.pattern_preview);
        p["opacity"] = json!(app.session.tools.brush.opacity * 100.0);
        p["flow"] = json!(app.session.tools.brush.flow * 100.0);
    }
    // The Clone Source panel's active slot (set by ⌥-click) drives the stroke: the engine keeps
    // the aligned pairing and applies the slot's scale/rotation/flip.
    let slot = app.session.presets.clone.active().source.is_some();
    match (app.ui.clone_offset.filter(|_| o.clone_aligned && !slot), app.ui.clone_source.filter(|_| !slot)) {
        _ if slot => {}
        (Some(off), _) => p["offset"] = json!(off),
        (None, Some(src)) => p["source"] = json!(src),
        (None, None) => return None,
    }
    Some(p)
}

/// The command and options-bar params (without `points` and `target`) of a sequential-dab tool:
/// Blur, Sharpen, Smudge, Dodge, Burn, Sponge. Shared by the live preview and the commit.
pub(crate) fn dab_params(app: &PhotocraftApp, tool: Tool) -> Option<(&'static str, Value)> {
    let o = &app.ui.tool_options;
    Some(match tool {
        Tool::Blur => ("paint.blur", json!({"strength": o.strength, "sampleAllLayers": o.sample_all_layers})),
        Tool::Sharpen => ("paint.sharpen", json!({"strength": o.strength, "protectDetail": o.protect_detail, "sampleAllLayers": o.sample_all_layers})),
        Tool::Smudge => ("paint.smudge", json!({"strength": o.strength, "fingerPainting": o.finger_painting, "sampleAllLayers": o.sample_all_layers})),
        Tool::Dodge | Tool::Burn => (
            if tool == Tool::Dodge { "paint.dodge" } else { "paint.burn" },
            json!({"range": o.tone_range, "exposure": o.exposure, "protectTones": o.protect_tones}),
        ),
        Tool::Sponge => ("paint.sponge", json!({"mode": o.sponge_mode, "vibrance": o.vibrance})),
        _ => return None,
    })
}

/// `paint.colorReplacement` params from the options bar (without `points`): shared by the live
/// preview and the commit. The replacement `color` is the foreground, recorded so a replayed
/// stroke paints the same colour; Background Swatch sampling matches the background colour.
pub(crate) fn color_replacement_params(app: &PhotocraftApp) -> Value {
    let o = &app.ui.tool_options;
    json!({
        "color": app.session.tools.foreground,
        "mode": o.cr_mode,
        "sampling": o.cr_sampling,
        "limits": o.cr_limits,
        "tolerance": o.cr_tolerance,
        "antiAlias": o.cr_anti_alias,
        // A mask or channel target is refused: the tool recolours layer pixels only.
        "target": crate::canvas::paint_target(app),
    })
}

/// Finish a stroke with a retouching tool. Returns false if `tool` isn't one.
pub fn finish_stroke(app: &mut PhotocraftApp, tool: Tool, points: &[[f64; 3]], mods: egui::Modifiers) -> bool {
    let o = app.ui.tool_options.clone();
    let pts = json!(points);
    let (cmd, mut p): (&str, Value) = match tool {
        Tool::SpotHealing => ("paint.spotHealing", json!({"type": o.spot_type, "sampleAllLayers": o.sample_all_layers})),
        Tool::Remove => {
            if !o.remove_after_each_stroke {
                remove_pending_add(app, points);
                return true;
            }
            ("paint.remove", remove_params(&o))
        }
        Tool::MixerBrush => ("paint.mixerBrush", json!({})),
        Tool::ColorReplacement => ("paint.colorReplacement", color_replacement_params(app)),
        Tool::PatternStamp => {
            let mut p = json!({
                "aligned": o.pattern_stamp_aligned,
                "impressionist": o.pattern_stamp_impressionist,
                "scale": o.pattern_stamp_scale,
                "angle": o.pattern_stamp_angle,
            });
            if o.pattern_stamp_aligned
                && let Some(ph) = app.ui.pattern_stamp_phase
            {
                p["phase"] = json!(ph);
            }
            if let Some(id) = photocraft_engine::presets::patterns::current(&app.session).or(app.session.patterns.items.first()).map(|x| x.id.clone()) {
                p["pattern"] = json!(id);
            }
            ("paint.patternStamp", p)
        }
        Tool::Healing | Tool::CloneStamp => {
            let Some(p) = clone_params(app) else {
                // Option-click on the Mac, Alt-click on Windows.
                app.ui.status = if cfg!(target_os = "macos") {
                    tl!("Option-click to define a source point to clone from")
                } else {
                    tl!("Alt-click to define a source point to clone from")
                }
                .into();
                app.ui.status_error = true;
                return true;
            };
            (if tool == Tool::Healing { "paint.healingBrush" } else { "paint.cloneStamp" }, p)
        }
        Tool::HistoryBrush => ("paint.historyBrush", json!({})),
        Tool::Blur | Tool::Sharpen | Tool::Smudge | Tool::Dodge | Tool::Burn | Tool::Sponge => match dab_params(app, tool) {
            Some(cp) => cp,
            None => return false,
        },
        Tool::QuickSelection => {
            let size = app.session.tools.brush.size;
            let mode = if mods.alt { "subtract" } else { "add" };
            let xy: Vec<[f64; 2]> = points.iter().map(|q| [q[0], q[1]]).collect();
            let (sample_all, enhance) = (o.sample_all_layers, o.enhance_edge);
            let r = app.run("select.quick", json!({"points": xy, "size": size, "mode": mode, "enhanceEdge": enhance, "sampleAllLayers": sample_all}));
            report_smart_result(app, r, tl!("Quick Selection found nothing there"));
            return true;
        }
        _ => return false,
    };
    p["points"] = pts;
    // The layer mask, an alpha channel or the Quick Mask when targeted, as the Brush paints.
    p["target"] = crate::canvas::paint_target(app);
    match app.run(cmd, p) {
        Ok(r) if matches!(tool, Tool::Healing | Tool::CloneStamp) => {
            // Aligned: keep the offset for later strokes; non-aligned: every stroke restarts at the source.
            app.ui.clone_offset = r.get("offset").and_then(|v| serde_json::from_value(v.clone()).ok()).filter(|_| o.clone_aligned);
        }
        Ok(r) if tool == Tool::PatternStamp => {
            app.ui.pattern_stamp_phase = r.get("phase").and_then(|v| serde_json::from_value(v.clone()).ok()).filter(|_| o.pattern_stamp_aligned);
        }
        Ok(_) => {}
        Err(e) => {
            app.ui.status = e;
            app.ui.status_error = true;
        }
    }
    true
}

/// `paint.remove` params from the options bar (without the strokes).
fn remove_params(o: &crate::state::ToolOptions) -> Value {
    json!({"sampleAllLayers": o.sample_all_layers, "newLayer": o.remove_new_layer})
}

/// Remove after each stroke off: a finished stroke waits with the others, tinted on the canvas,
/// until ↩ or the options bar's ✓ applies them all as one removal (Esc or ⊘ drops them).
pub fn remove_pending_add(app: &mut PhotocraftApp, points: &[[f64; 3]]) {
    let Some(st) = app.session.active() else { return };
    let (doc, size) = (st.doc.id, [st.doc.size.width, st.doc.size.height]);
    if app.remove_pending_doc != Some(doc) {
        app.remove_pending.clear();
        app.remove_trail = None;
        app.remove_pending_doc = Some(doc);
    }
    let diameter = app.session.tools.brush.size;
    app.remove_trail.get_or_insert_with(|| crate::stroke_trail::Trail::new(size)).feed(points, diameter);
    app.remove_pending.push(points.to_vec());
    app.ui.status = crate::i18n::fmt(tl!("Strokes to remove: {n} (Enter applies, Esc cancels)"), &[("n", &app.remove_pending.len().to_string())]);
    app.ui.status_error = false;
}

/// Apply the waiting strokes as one removal (one job, one undo step). False when none wait.
pub fn remove_pending_commit(app: &mut PhotocraftApp) -> bool {
    if app.remove_pending.is_empty() {
        return false;
    }
    let strokes = std::mem::take(&mut app.remove_pending);
    app.remove_trail = None;
    app.remove_pending_doc = None;
    let mut p = remove_params(&app.ui.tool_options);
    p["strokes"] = json!(strokes);
    p["target"] = crate::canvas::paint_target(app);
    app.ui.status.clear();
    if let Err(e) = app.run("paint.remove", p) {
        app.ui.status = e;
        app.ui.status_error = true;
    }
    true
}

/// Drop the waiting strokes. False when none wait.
pub fn remove_pending_cancel(app: &mut PhotocraftApp) -> bool {
    if app.remove_pending.is_empty() {
        return false;
    }
    app.remove_pending.clear();
    app.remove_trail = None;
    app.remove_pending_doc = None;
    app.ui.status.clear();
    app.ui.status_error = false;
    true
}

/// Whether Remove strokes are still waiting: they are dropped when the tool or the document
/// changed under them (checked every frame).
pub fn remove_pending_live(app: &mut PhotocraftApp) -> bool {
    if app.remove_pending.is_empty() {
        return false;
    }
    let doc = app.session.active().map(|st| st.doc.id);
    if app.ui.tool == Tool::Remove && doc.is_some() && app.remove_pending_doc == doc {
        return true;
    }
    remove_pending_cancel(app);
    false
}

/// The tint of the waiting Remove strokes, drawn every frame.
pub fn draw_remove_pending(app: &mut PhotocraftApp, painter: &egui::Painter, xf: &ViewXform) {
    if !remove_pending_live(app) {
        return;
    }
    let Some(bounds) = app.session.active().map(|st| st.doc.bounds()) else { return };
    let doc_rect = xf.doc_rect(bounds);
    let col = Tokens::get(painter.ctx()).danger.gamma_multiply(0.45);
    if let Some(trail) = app.remove_trail.as_mut() {
        trail.draw(painter, doc_rect, xf.flip, col);
    }
}

/// Patch Tool and Content-Aware Move Tool: a drag that starts inside the selection (without ⇧ or
/// ⌥) drags the selection; any other drag draws a lasso selection.
pub fn patch_drags_selection(app: &PhotocraftApp, at: [f64; 2], mods: egui::Modifiers) -> bool {
    if mods.shift || mods.alt {
        return false;
    }
    let Some(sel) = app.session.active().and_then(|st| st.doc.selection.as_ref()) else { return false };
    sel.sample_channel(at[0].floor() as i32, at[1].floor() as i32, 0) > 0.0
}

/// Whole-pixel offset for a drag from `start` to `end`, limited so the dragged outline stays on the
/// canvas (the engine rejects a patch or move that leaves it).
pub fn patch_offset(app: &mut PhotocraftApp, start: [f64; 2], end: [f64; 2]) -> [i32; 2] {
    let (dx, dy) = ((end[0] - start[0]).round(), (end[1] - start[1]).round());
    let (dx, dy) = if dx.is_finite() && dy.is_finite() { (dx.clamp(-1e7, 1e7) as i32, dy.clamp(-1e7, 1e7) as i32) } else { (0, 0) };
    let Some((canvas, sel)) = app.session.active().and_then(|st| Some((st.doc.bounds(), st.doc.selection.clone()?))) else { return [dx, dy] };
    let key = u64::MAX - app.session.active().map_or(0, |st| st.doc.id.0);
    let b = app.cached_bounds(key, &sel).intersect(&canvas);
    if b.is_empty() {
        return [dx, dy];
    }
    let clamp = |d: i32, lo: i32, hi: i32| if lo > hi { 0 } else { d.clamp(lo, hi) };
    [clamp(dx, canvas.x0 - b.x0, canvas.x1 - b.x1), clamp(dy, canvas.y0 - b.y0, canvas.y1 - b.y1)]
}

/// Status-bar hint for the Patch and Content-Aware Move tools: the selection is the patch and it
/// has to be dragged somewhere. Without it a lasso, or a click inside the selection, changes
/// nothing on the canvas and the tool looks dead (#1715).
pub fn patch_hint(app: &mut PhotocraftApp) {
    app.ui.status = tl!("Now drag the selection onto another area").into();
    app.ui.status_error = false;
}

/// Patch Tool: the patch was dragged from `start` to `end`.
pub fn finish_patch(app: &mut PhotocraftApp, start: [f64; 2], end: [f64; 2]) {
    let off = patch_offset(app, start, end);
    if off == [0, 0] {
        patch_hint(app);
        return;
    }
    let preview = crate::patch_preview::take(app);
    let o = &app.ui.tool_options;
    if o.patch_content_aware {
        // A background job (progress dialog, Esc cancels), like the Content-Aware Move.
        let p = json!({
            "offset": off,
            "contentAware": true,
            "structure": o.patch_structure.round().clamp(1.0, 7.0),
            "color": o.patch_color.round().clamp(0.0, 10.0),
            "sampleAllLayers": o.sample_all_layers,
            "target": crate::canvas::paint_target(app),
        });
        if let Err(e) = app.run("paint.patch", p) {
            app.ui.status = e;
            app.ui.status_error = true;
        }
        return;
    }
    let p = json!({"offset": off, "mode": app.ui.tool_options.patch_mode, "target": crate::canvas::paint_target(app)});
    match app.run("paint.patch", p) {
        Ok(_) => crate::patch_preview::committed(app, preview, off),
        Err(e) => {
            app.ui.status = e;
            app.ui.status_error = true;
        }
    }
}

/// Content-Aware Move Tool: the selection was dragged from `start` to `end`. The engine runs it as
/// a background job (progress dialog, Esc cancels); the selection follows the content. With
/// Transform On Drop a transform box comes up on the dropped patch first ([`crate::cam_transform`]).
pub fn finish_content_aware_move(app: &mut PhotocraftApp, start: [f64; 2], end: [f64; 2]) {
    let off = patch_offset(app, start, end);
    if off == [0, 0] {
        patch_hint(app);
        return;
    }
    if app.ui.tool_options.cam_transform_on_drop && crate::cam_transform::begin(app, off) {
        return;
    }
    let p = content_aware_move_params(app, [i64::from(off[0]), i64::from(off[1])]);
    if let Err(e) = app.run("paint.contentAwareMove", p) {
        app.ui.status = e;
        app.ui.status_error = true;
    }
}

/// `paint.contentAwareMove`'s params for a drag by `offset`, from the options bar.
pub fn content_aware_move_params(app: &PhotocraftApp, offset: [i64; 2]) -> Value {
    let o = &app.ui.tool_options;
    json!({
        "offset": offset,
        "mode": o.cam_mode,
        "structure": o.cam_structure.round().clamp(1.0, 7.0),
        "color": o.cam_color.round().clamp(0.0, 10.0),
        "sampleAllLayers": o.sample_all_layers,
        "target": crate::canvas::paint_target(app),
    })
}

/// Object Selection: the dragged rectangle. A click (or a sub-2 px nudge) cannot be a
/// rectangle: with Object Finder on it selects the object under the pointer, the one hovering
/// highlighted (#4087; ⇧ adds, ⌥ subtracts, ⇧⌥ intersects, otherwise replaces). With it off it
/// falls back to a Quick Selection dab at
/// the brush size instead of looking dead (⌥ subtracts, otherwise it adds). Larger drags
/// run the rectangle GrabCut (⌥ subtracts, ⇧ adds, otherwise replaces).
pub fn finish_object_selection(app: &mut PhotocraftApp, start: [f64; 2], end: [f64; 2], mods: egui::Modifiers) {
    let (x, y) = (start[0].min(end[0]), start[1].min(end[1]));
    let (w, h) = ((end[0] - start[0]).abs(), (end[1] - start[1]).abs());
    let sample_all = app.ui.tool_options.sample_all_layers;
    if (w < 2.0 || h < 2.0) && app.ui.tool_options.object_finder {
        // Object Finder (#4087): the object under the click, as hovering highlighted it.
        let mode = match (mods.shift, mods.alt) {
            (true, true) => "intersect",
            (true, false) => "add",
            (false, true) => "subtract",
            (false, false) => "replace",
        };
        let r = app.run("select.object", json!({"point": [end[0], end[1]], "mode": mode, "sampleAllLayers": sample_all}));
        crate::object_finder::clear(app);
        report_smart_result(app, r, tl!("No object found there — drag a rectangle around it"));
        return;
    }
    if w < 2.0 || h < 2.0 {
        let size = app.session.tools.brush.size;
        let mode = if mods.alt { "subtract" } else { "add" };
        let r = app.run("select.quick", json!({"points": [[end[0], end[1]]], "size": size, "mode": mode, "sampleAllLayers": sample_all}));
        report_smart_result(app, r, tl!("Quick Selection found nothing there"));
        return;
    }
    let mode = if mods.alt {
        "subtract"
    } else if mods.shift {
        "add"
    } else {
        "replace"
    };
    let r = app.run("select.object", json!({"rect": [x.round(), y.round(), w.round(), h.round()], "mode": mode, "sampleAllLayers": sample_all}));
    report_smart_result(app, r, tl!("Object Selection found nothing there — try a tighter rectangle"));
}

/// Reports a smart-selection command result in the status bar. Engine errors are already
/// reported by [`PhotocraftApp::run`], but an empty result (`changed: false`) is a silent
/// success that looks like a dead button (Select Subject finding nothing on a busy photo,
/// Object Selection missing its rectangle). `empty_msg` says what to try next.
pub fn report_smart_result(app: &mut PhotocraftApp, res: Result<Value, String>, empty_msg: &str) {
    if matches!(res, Ok(ref v) if v.get("changed").and_then(Value::as_bool) == Some(false)) {
        app.ui.status = empty_msg.into();
        app.ui.status_error = false;
    }
}

/// ⌥-click with Clone Stamp / Healing Brush sets the source.
pub fn set_source(app: &mut PhotocraftApp, x: f64, y: f64) {
    app.ui.clone_source = Some([x.round(), y.round()]);
    app.ui.clone_offset = None;
    let _ = app.run("cloneSource.set", json!({"source": [x.round(), y.round()]}));
    let template = "Clone source set at {x}, {y}";
    app.ui.status = crate::i18n::fmt(tl!(template), &[("x", &format!("{x:.0}")), ("y", &format!("{y:.0}"))]);
    app.ui.status_error = false;
}

/// Where the clone source is sampled from, shown only while a Clone Stamp or Healing Brush stroke
/// is painted, as in Photoshop: setting the source (⌥-click) leaves nothing on the canvas (#668).
pub fn source_marker_point(app: &PhotocraftApp) -> Option<[f64; 2]> {
    if !matches!(app.active_tool(), Tool::CloneStamp | Tool::Healing) {
        return None;
    }
    let at = app.drag.as_ref().filter(|d| matches!(d.tool, Tool::CloneStamp | Tool::Healing)).and_then(|d| d.points.last().map(|p| [p[0], p[1]]))?;
    match (crate::preset_panels::clone_sample_point(app, Some(at)), app.ui.clone_offset, app.ui.clone_source) {
        (Some(p), ..) => Some(p),
        (None, Some(off), _) => Some([at[0] + off[0], at[1] + off[1]]),
        (None, None, s) => s,
    }
}

/// Crosshair at [`source_marker_point`].
pub fn draw_source_marker(app: &PhotocraftApp, painter: &egui::Painter, xf: &ViewXform) {
    let Some(s) = source_marker_point(app) else { return };
    let c = xf.to_screen(s[0] as f32, s[1] as f32);
    for (w, col) in [(3.0, Color32::from_black_alpha(160)), (1.0, Color32::WHITE)] {
        painter.line_segment([c - vec2(7.0, 0.0), c + vec2(7.0, 0.0)], Stroke::new(w, col));
        painter.line_segment([c - vec2(0.0, 7.0), c + vec2(0.0, 7.0)], Stroke::new(w, col));
    }
}

fn opt(ui: &mut egui::Ui, text: &str) {
    let t = Tokens::get(ui.ctx());
    ui.label(egui::RichText::new(tl!(&text)).color(t.text_dim).size(12.0));
}

fn pct(ui: &mut egui::Ui, label: &str, v: &mut f32) {
    opt(ui, label);
    crate::widgets::value_field(ui, v, 1.0..=100.0, "%", 58.0);
}

/// Options bar for the retouching and smart-selection tools. Returns false for other tools.
pub fn options_bar(app: &mut PhotocraftApp, ui: &mut egui::Ui, tool: Tool) -> bool {
    if !tool.is_brushlike() && !matches!(tool, Tool::QuickSelection | Tool::ObjectSelection | Tool::Patch | Tool::ContentAwareMove | Tool::RedEye)
        || matches!(tool, Tool::Brush | Tool::Pencil | Tool::MixerBrush | Tool::Eraser)
    {
        return false;
    }
    let o = &mut app.ui.tool_options;
    match tool {
        Tool::SpotHealing => {
            opt(ui, tl!("Type:"));
            for (k, l) in [("contentAware", tl!("Content-Aware")), ("createTexture", tl!("Create Texture")), ("proximityMatch", tl!("Proximity Match"))] {
                let mut on = o.spot_type == k;
                if crate::widgets::checkbox(ui, &mut on, l).clicked() {
                    o.spot_type = k.into();
                }
            }
            crate::widgets::vline(ui, 22.0);
            crate::widgets::checkbox(ui, &mut o.sample_all_layers, tl!("Sample All Layers"));
        }
        Tool::Remove => {
            // Photoshop's options bar: Sample All Layers, Create a new layer, Remove after each
            // stroke; with strokes waiting, ✓ applies them and ⊘ drops them.
            crate::widgets::checkbox(ui, &mut o.sample_all_layers, tl!("Sample All Layers"));
            crate::widgets::checkbox(ui, &mut o.remove_new_layer, tl!("Create a new layer"));
            crate::widgets::checkbox(ui, &mut o.remove_after_each_stroke, tl!("Remove after each stroke"));
            if !app.remove_pending.is_empty() {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if crate::widgets::primary_button(ui, "✓", 32.0)
                        .on_hover_text(crate::i18n::fmt(tl!("Apply the removal ({key})"), &[("key", &crate::shortcuts::pretty("Enter"))]))
                        .clicked()
                    {
                        remove_pending_commit(app);
                    } else if crate::widgets::secondary_button(ui, "⊘", 32.0).on_hover_text(tl!("Cancel (Esc)")).clicked() {
                        remove_pending_cancel(app);
                    }
                });
            }
        }
        Tool::Patch => {
            // Photoshop's options bar: Patch: Normal (Source / Destination) or Content-Aware
            // (Structure, Color, Sample All Layers).
            opt(ui, tl!("Patch:"));
            let kinds = [("normal".to_string(), tl!("Normal")), ("contentAware".to_string(), tl!("Content-Aware"))];
            let mut kind = if o.patch_content_aware { "contentAware" } else { "normal" }.to_string();
            crate::widgets::dropdown(ui, "patch-kind", &mut kind, &kinds, 120.0);
            o.patch_content_aware = kind == "contentAware";
            if o.patch_content_aware {
                opt(ui, tl!("Structure:"));
                crate::widgets::value_field(ui, &mut o.patch_structure, 1.0..=7.0, "", 40.0);
                opt(ui, tl!("Color:"));
                crate::widgets::value_field(ui, &mut o.patch_color, 0.0..=10.0, "", 40.0);
                crate::widgets::checkbox(ui, &mut o.sample_all_layers, tl!("Sample All Layers"));
            } else {
                for (k, l) in [("source", tl!("Source")), ("destination", tl!("Destination"))] {
                    let mut on = o.patch_mode == k;
                    if crate::widgets::checkbox(ui, &mut on, l).clicked() {
                        o.patch_mode = k.into();
                    }
                }
            }
            crate::widgets::vline(ui, 22.0);
            opt(ui, tl!("Lasso around an area, then drag the selection"));
        }
        Tool::ContentAwareMove => {
            opt(ui, tl!("Mode:"));
            let modes = [("move".to_string(), tl!("Move")), ("extend".to_string(), tl!("Extend"))];
            crate::widgets::dropdown(ui, "cam-mode", &mut o.cam_mode, &modes, 90.0);
            opt(ui, tl!("Structure:"));
            crate::widgets::value_field(ui, &mut o.cam_structure, 1.0..=7.0, "", 40.0);
            opt(ui, tl!("Color:"));
            crate::widgets::value_field(ui, &mut o.cam_color, 0.0..=10.0, "", 40.0);
            crate::widgets::checkbox(ui, &mut o.sample_all_layers, tl!("Sample All Layers"));
            crate::widgets::checkbox(ui, &mut o.cam_transform_on_drop, tl!("Transform On Drop"));
            crate::widgets::vline(ui, 22.0);
            if crate::cam_transform::active(app) {
                // The box on the dropped patch: ✓ computes the move, ⊘ drops it (as ↩ / Esc).
                let tip = crate::i18n::fmt(tl!("Commit transform ({key})"), &[("key", &crate::shortcuts::pretty("Enter"))]);
                let commit = crate::icons::button(ui, "check", 26.0, false, &tip);
                commit.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, "Commit transform"));
                if commit.clicked() {
                    crate::cam_transform::commit(app);
                }
                let cancel = crate::icons::button(ui, "ban", 26.0, false, tl!("Cancel transform (Esc)"));
                cancel.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, "Cancel transform"));
                if cancel.clicked() {
                    crate::cam_transform::cancel(app);
                }
            } else {
                opt(ui, tl!("Lasso around an area, then drag the selection"));
            }
        }
        Tool::PatternStamp => {
            pattern_stamp_options(app, ui);
        }
        Tool::Healing | Tool::CloneStamp => {
            crate::widgets::checkbox(ui, &mut o.clone_aligned, tl!("Aligned"));
            opt(ui, tl!("Sample:"));
            let opts = [
                ("current".to_string(), tl!("Current Layer")),
                ("currentAndBelow".to_string(), tl!("Current & Below")),
                ("all".to_string(), tl!("All Layers")),
            ];
            crate::widgets::dropdown(ui, "clone-sample", &mut o.clone_sample, &opts, 130.0);
            if app.ui.clone_source.is_none() {
                crate::widgets::vline(ui, 22.0);
                opt(ui, &crate::i18n::fmt(tl!("{key}-click to set the source"), &[("key", &crate::shortcuts::pretty("Alt"))]));
            }
        }
        Tool::Dodge | Tool::Burn => {
            opt(ui, tl!("Range:"));
            let opts = [("shadows".to_string(), tl!("Shadows")), ("midtones".to_string(), tl!("Midtones")), ("highlights".to_string(), tl!("Highlights"))];
            crate::widgets::dropdown(ui, "tone-range", &mut o.tone_range, &opts, 100.0);
            pct(ui, tl!("Exposure:"), &mut o.exposure);
            crate::widgets::checkbox(ui, &mut o.protect_tones, tl!("Protect Tones"));
        }
        Tool::Sponge => {
            opt(ui, tl!("Mode:"));
            let opts = [("desaturate".to_string(), tl!("Desaturate")), ("saturate".to_string(), tl!("Saturate"))];
            crate::widgets::dropdown(ui, "sponge-mode", &mut o.sponge_mode, &opts, 110.0);
            crate::widgets::checkbox(ui, &mut o.vibrance, tl!("Vibrance"));
        }
        Tool::Blur | Tool::Sharpen | Tool::Smudge => {
            pct(ui, tl!("Strength:"), &mut o.strength);
            crate::widgets::checkbox(ui, &mut o.sample_all_layers, tl!("Sample All Layers"));
            if tool == Tool::Sharpen {
                crate::widgets::checkbox(ui, &mut o.protect_detail, tl!("Protect Detail"));
            }
            if tool == Tool::Smudge {
                crate::widgets::checkbox(ui, &mut o.finger_painting, tl!("Finger Painting"));
            }
        }
        Tool::ColorReplacement => {
            // Photoshop's options bar: Mode, Sampling (three icon toggles), Limits, Tolerance, Anti-alias.
            opt(ui, tl!("Mode:"));
            let modes = [
                ("hue".to_string(), tl!("Hue")),
                ("saturation".to_string(), tl!("Saturation")),
                ("color".to_string(), tl!("Color")),
                ("luminosity".to_string(), tl!("Luminosity")),
            ];
            crate::widgets::dropdown(ui, "color-replacement-mode", &mut o.cr_mode, &modes, 100.0);
            crate::widgets::vline(ui, 22.0);
            ui.spacing_mut().item_spacing.x = 2.0;
            for (k, icon, tip) in [
                ("continuous", "pipette", "Sampling: Continuous"),
                ("once", "circle-dot", "Sampling: Once"),
                ("backgroundSwatch", "square", "Sampling: Background Swatch"),
            ] {
                if crate::brush_picker::named(crate::icons::button(ui, icon, 24.0, o.cr_sampling == k, tl!(tip)), tip).clicked() {
                    o.cr_sampling = k.into();
                }
            }
            ui.spacing_mut().item_spacing.x = 8.0;
            crate::widgets::vline(ui, 22.0);
            opt(ui, tl!("Limits:"));
            let limits = [
                ("discontiguous".to_string(), tl!("Discontiguous")),
                ("contiguous".to_string(), tl!("Contiguous")),
                ("findEdges".to_string(), tl!("Find Edges")),
            ];
            crate::widgets::dropdown(ui, "color-replacement-limits", &mut o.cr_limits, &limits, 110.0);
            opt(ui, tl!("Tolerance:"));
            crate::widgets::value_field(ui, &mut o.cr_tolerance, 0.0..=100.0, "%", 62.0);
            crate::widgets::checkbox(ui, &mut o.cr_anti_alias, tl!("Anti-alias"));
        }
        Tool::HistoryBrush => opt(ui, "Paints from the document's opening state"),
        Tool::QuickSelection => {
            crate::widgets::checkbox(ui, &mut o.sample_all_layers, tl!("Sample All Layers"));
            crate::widgets::checkbox(ui, &mut o.enhance_edge, tl!("Enhance Edge"));
            opt(ui, &crate::i18n::fmt(tl!("{key} to subtract"), &[("key", &crate::shortcuts::pretty("Alt"))]));
            crate::widgets::vline(ui, 22.0);
            if crate::widgets::secondary_button(ui, tl!("Select Subject"), 0.0).clicked() {
                let r = app.run("select.subject", json!({}));
                report_smart_result(app, r, tl!("No subject found — try Object Selection or the Lasso"));
            }
        }
        Tool::ObjectSelection => {
            crate::widgets::checkbox(ui, &mut o.object_finder, tl!("Object Finder"));
            crate::widgets::checkbox(ui, &mut o.sample_all_layers, tl!("Sample All Layers"));
            opt(ui, tl!("Drag a rectangle around the object"));
            crate::widgets::vline(ui, 22.0);
            if crate::widgets::secondary_button(ui, tl!("Select Subject"), 0.0).clicked() {
                let r = app.run("select.subject", json!({}));
                report_smart_result(app, r, tl!("No subject found — try a rectangle around the object"));
            }
        }
        Tool::RedEye => {
            opt(ui, tl!("Pupil Size"));
            crate::widgets::value_field(ui, &mut o.red_eye_pupil_size, 1.0..=100.0, "", 50.0);
            opt(ui, tl!("Darken Amount"));
            crate::widgets::value_field(ui, &mut o.red_eye_darken, 0.0..=100.0, "", 50.0);
            crate::widgets::vline(ui, 22.0);
            opt(ui, tl!("Click a red pupil to neutralize it"));
        }
        _ => {}
    }
    true
}

fn pattern_stamp_options(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let opts: Vec<(String, String)> = app.session.patterns.items.iter().map(|p| (p.id.clone(), p.display_name().to_string())).collect();
    let mut pat = photocraft_engine::presets::patterns::current(&app.session).or(app.session.patterns.items.first()).map(|p| p.id.clone()).unwrap_or_default();
    let o = &mut app.ui.tool_options;
    crate::widgets::checkbox(ui, &mut o.pattern_stamp_aligned, tl!("Aligned"));
    crate::widgets::checkbox(ui, &mut o.pattern_stamp_impressionist, tl!("Impressionist"));
    crate::widgets::vline(ui, 22.0);
    opt(ui, tl!("Pattern"));
    let opts_ref: Vec<(String, &str)> = opts.iter().map(|(id, n)| (id.clone(), n.as_str())).collect();
    let picked = if opts_ref.is_empty() {
        ui.label(egui::RichText::new(tl!("No patterns")).color(Tokens::get(ui.ctx()).text_faint));
        false
    } else {
        crate::widgets::dropdown(ui, "pattern-stamp-pat", &mut pat, &opts_ref, 140.0)
    };
    opt(ui, tl!("Scale"));
    crate::widgets::value_field(ui, &mut o.pattern_stamp_scale, 1.0..=1000.0, "%", 58.0);
    opt(ui, tl!("Angle:"));
    crate::widgets::value_field(ui, &mut o.pattern_stamp_angle, -180.0..=180.0, "°", 58.0);
    if picked {
        let _ = app.run("pattern.presets.select", json!({"pattern": pat}));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas::{ToolEvent, tool_event};

    fn app() -> PhotocraftApp {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 100, "height": 60})).unwrap();
        app.run("tools.setBrush", json!({"brush": {"size": 16, "hardness": 1.0}})).unwrap();
        app
    }

    fn drag(app: &mut PhotocraftApp, tool: Tool) {
        app.ui.tool = tool;
        let m = egui::Modifiers::NONE;
        tool_event(app, ToolEvent::Down { x: 10.0, y: 30.0, pressure: 1.0 }, m);
        tool_event(app, ToolEvent::Move { x: 50.0, y: 30.0, pressure: 1.0 }, m);
        tool_event(app, ToolEvent::Up { x: 50.0, y: 30.0 }, m);
        assert!(!app.ui.status_error, "{tool:?}: {}", app.ui.status);
    }

    fn active(app: &PhotocraftApp) -> &photocraft_doc::Layer {
        let st = app.session.active().unwrap();
        st.active_layer.and_then(|id| st.doc.layer(id)).unwrap()
    }

    fn stripes(app: &mut PhotocraftApp, step: usize, target: &str) {
        for x in (0..100).step_by(step) {
            app.run("paint.pencil", json!({"points": [[x, 0], [x, 60]], "size": 2, "color": "#606060", "target": target})).unwrap();
        }
    }

    #[test]
    fn clone_stamp_in_pattern_preview_wraps_a_dab_across_the_right_edge() {
        for pattern_preview in [false, true] {
            let mut app = app();
            app.run("paint.pencil", json!({"points": [[10, 10]], "size": 24, "color": "#ff0000"})).unwrap();
            app.ui.tool = Tool::CloneStamp;
            app.ui.clone_source = Some([10.0, 10.0]);
            app.ui.view.pattern_preview = pattern_preview;
            let modifiers = egui::Modifiers::NONE;
            tool_event(&mut app, ToolEvent::Down { x: 98.0, y: 30.0, pressure: 1.0 }, modifiers);
            tool_event(&mut app, ToolEvent::Up { x: 98.0, y: 30.0 }, modifiers);
            assert!(!app.ui.status_error, "{}", app.ui.status);
            let surface = active(&app).surface().unwrap();
            let center = surface.rgba(98, 30);
            assert!(center[0] > 0.9 && center[1] < 0.1, "clone center: {center:?}");
            let opposite = surface.rgba(1, 30);
            if pattern_preview {
                assert!(opposite[0] > 0.9 && opposite[1] < 0.1, "Pattern Preview must wrap the clone footprint: {opposite:?}");
            } else {
                assert!(opposite[0] > 0.9 && opposite[1] > 0.9, "ordinary cloning must not wrap: {opposite:?}");
            }
        }
    }

    #[test]
    fn alt_flips_dodge_burn_and_blur_sharpen() {
        for (tool, flipped) in [(Tool::Dodge, Tool::Burn), (Tool::Burn, Tool::Dodge), (Tool::Blur, Tool::Sharpen), (Tool::Sharpen, Tool::Blur)] {
            assert_eq!(alt_flipped(tool, true), flipped);
            assert_eq!(alt_flipped(tool, false), tool);
        }
        for tool in [Tool::Sponge, Tool::Smudge, Tool::Brush] {
            assert_eq!(alt_flipped(tool, true), tool, "{tool:?} has no ⌥ counterpart");
        }
    }

    #[test]
    fn alt_stroke_runs_the_opposite_tool() {
        // ⌥ held as the stroke starts: Dodge burns, Burn dodges, Blur sharpens, Sharpen blurs.
        for (tool, cmd) in [(Tool::Dodge, "paint.burn"), (Tool::Burn, "paint.dodge"), (Tool::Blur, "paint.sharpen"), (Tool::Sharpen, "paint.blur")] {
            let mut app = app();
            app.ui.tool = tool;
            let m = egui::Modifiers::ALT;
            tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 30.0, pressure: 1.0 }, m);
            tool_event(&mut app, ToolEvent::Move { x: 50.0, y: 30.0, pressure: 1.0 }, m);
            tool_event(&mut app, ToolEvent::Up { x: 50.0, y: 30.0 }, m);
            assert_eq!(app.session.journal.last().map(|(id, _)| id.as_str()), Some(cmd), "{tool:?} with ⌥");
            assert_eq!(app.ui.tool, tool, "the selected tool stays {tool:?}");
        }
    }

    #[test]
    fn clone_stamp_brush_opacity_and_flow_reach_the_painted_pixels() {
        let mut app = app();
        app.run("paint.pencil", json!({"points": [[10, 10]], "size": 16, "color": "#ff0000"})).unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        app.ui.tool = Tool::CloneStamp;
        app.ui.clone_source = Some([10.0, 10.0]);
        app.ui.tool_options.clone_sample = "all".into();
        app.run("tools.setBrush", json!({"brush": {"opacity": 0.5, "flow": 0.5}})).unwrap();
        assert!(finish_stroke(&mut app, Tool::CloneStamp, &[[50.0, 30.0, 1.0]], egui::Modifiers::NONE));
        assert!(!app.ui.status_error, "{}", app.ui.status);
        let pixel = active(&app).surface().unwrap().rgba(50, 30);
        assert!(pixel[0] > 0.9 && pixel[3] > 0.0 && pixel[3] < 0.5, "partial-opacity, partial-flow red stamp: {pixel:?}");
        let (id, params) = app.session.journal.last().unwrap();
        assert_eq!(id, "paint.cloneStamp");
        assert_eq!(params["opacity"], 50.0);
        assert_eq!(params["flow"], 50.0);
    }

    #[test]
    fn clone_without_a_source_names_the_platform_modifier() {
        // Option-click on the Mac, Alt-click elsewhere (#251).
        for tool in [Tool::CloneStamp, Tool::Healing] {
            let mut app = app();
            app.ui.tool = tool;
            assert!(finish_stroke(&mut app, tool, &[[10.0, 30.0, 1.0], [50.0, 30.0, 1.0]], egui::Modifiers::NONE));
            assert!(app.ui.status_error, "{tool:?}");
            let want = if cfg!(target_os = "macos") { "Option-click" } else { "Alt-click" };
            assert!(app.ui.status.starts_with(want), "{tool:?}: {}", app.ui.status);
        }
    }

    #[test]
    fn retouch_strokes_paint_the_targeted_mask() {
        // #207: with the mask targeted, retouching changes the mask, never the pixels.
        for tool in [Tool::Blur, Tool::Sharpen, Tool::Smudge, Tool::Dodge, Tool::Burn] {
            let mut app = app();
            app.run("layer.new.layer", json!({})).unwrap();
            stripes(&mut app, 4, "pixels");
            app.run("layer.layerMask.revealAll", json!({})).unwrap();
            stripes(&mut app, 6, "mask");
            app.ui.mask_target = true;
            app.ui.tool_options.protect_detail = false;
            let b = app.session.active().unwrap().doc.bounds();
            let px0 = active(&app).surface().unwrap().read_region(b);
            let mask0 = active(&app).mask.as_ref().unwrap().surface.read_region(b);
            drag(&mut app, tool);
            assert_eq!(active(&app).surface().unwrap().read_region(b), px0, "{tool:?}: pixels untouched");
            assert_ne!(active(&app).mask.as_ref().unwrap().surface.read_region(b), mask0, "{tool:?}: mask changed");
        }
    }

    #[test]
    fn sample_all_layers_reaches_the_command() {
        // #207, #731: the options-bar checkbox reaches the command.
        for tool in [Tool::SpotHealing, Tool::Remove, Tool::Blur, Tool::Sharpen, Tool::Smudge] {
            for all in [false, true] {
                let mut app = app();
                stripes(&mut app, 4, "pixels");
                app.run("layer.new.layer", json!({})).unwrap();
                app.ui.tool_options.sample_all_layers = all;
                drag(&mut app, tool);
                let a = active(&app).surface().unwrap().rgba(30, 30)[3];
                assert_eq!(a > 0.0, all, "{tool:?} sampleAllLayers={all}: alpha {a}");
            }
        }
    }

    #[test]
    fn object_selection_click_falls_back_to_quick_selection() {
        // A click cannot be a rectangle: it selects at the cursor instead of looking dead.
        let mut app = app();
        app.ui.tool = Tool::ObjectSelection;
        app.ui.tool_options.object_finder = false;
        let m = egui::Modifiers::NONE;
        tool_event(&mut app, ToolEvent::Down { x: 50.0, y: 30.0, pressure: 1.0 }, m);
        tool_event(&mut app, ToolEvent::Up { x: 50.0, y: 30.0 }, m);
        assert!(!app.ui.status_error, "{}", app.ui.status);
        assert_eq!(app.session.journal.last().map(|(id, _)| id.as_str()), Some("select.quick"));
    }

    #[test]
    fn object_selection_rect_runs_object_selection_and_reports() {
        let mut app = app();
        app.run("paint.pencil", json!({"points": [[20, 10], [20, 50]], "size": 8, "color": "#ff0000"})).unwrap();
        app.ui.tool = Tool::ObjectSelection;
        let m = egui::Modifiers::NONE;
        tool_event(&mut app, ToolEvent::Down { x: 5.0, y: 5.0, pressure: 1.0 }, m);
        tool_event(&mut app, ToolEvent::Move { x: 60.0, y: 55.0, pressure: 1.0 }, m);
        tool_event(&mut app, ToolEvent::Up { x: 60.0, y: 55.0 }, m);
        assert!(!app.ui.status_error, "{}", app.ui.status);
        assert_eq!(app.session.journal.last().map(|(id, _)| id.as_str()), Some("select.object"));
    }

    #[test]
    fn empty_smart_result_is_reported_without_error() {
        // An empty smart selection (Select Subject missing on a busy photo) is an info, not
        // an error and not silence: the button must visibly answer.
        let mut app = app();
        report_smart_result(&mut app, Ok(json!({"selected": false, "changed": false})), "nothing there");
        assert_eq!(app.ui.status.as_str(), "nothing there");
        assert!(!app.ui.status_error);
        app.ui.status.clear();
        report_smart_result(&mut app, Ok(json!({"selected": true, "changed": true})), "nothing there");
        assert!(app.ui.status.is_empty());
    }

    #[test]
    fn mixer_brush_is_selectable_and_paints_only_inside_the_selection_via_control() {
        use crate::control::{ControlRequest, Outcome, handle};

        let mut app = app();
        app.run("paint.pencil", json!({"points": [[50, 30]], "size": 200, "color": "#204080"})).unwrap();
        app.run("tools.setColors", json!({"foreground": "#f02010"})).unwrap();
        app.run(
            "tools.setBrush",
            json!({
                "pressureSize": false,
                "size": 12,
                "mixer": {"wet": 0.0, "load": 1.0, "mix": 0.0, "flow": 1.0}
            }),
        )
        .unwrap();
        app.run("select.rect", json!({"x": 35, "y": 20, "width": 30, "height": 20})).unwrap();
        let before = {
            let surface = active(&app).surface().unwrap();
            [surface.rgba(20, 30), surface.rgba(50, 30), surface.rgba(80, 30)]
        };

        let ctx = egui::Context::default();
        let (req, _rx) = ControlRequest::new(
            "ui.pointer",
            json!({
                "tool": "mixerBrush",
                "events": [
                    {"kind": "down", "x": 8, "y": 30},
                    {"kind": "move", "x": 92, "y": 30},
                    {"kind": "up", "x": 92, "y": 30}
                ]
            }),
        );
        assert!(matches!(handle(&mut app, &ctx, &req), Outcome::Done(_)));
        assert_eq!(app.ui.tool, Tool::MixerBrush);
        assert_eq!(app.session.journal.last().map(|(id, _)| id.as_str()), Some("paint.mixerBrush"));

        let surface = active(&app).surface().unwrap();
        assert_ne!(surface.rgba(50, 30), before[1], "the selected pixels are mixed");
        assert_eq!(surface.rgba(20, 30), before[0], "outside the selection is unchanged");
        assert_eq!(surface.rgba(80, 30), before[2], "the far side outside the selection is unchanged");
    }

    #[test]
    fn patch_tool_lassoes_then_drags_the_patch() {
        let mut app = app();
        app.run("paint.pencil", json!({"points": [[68, 30], [74, 30]], "size": 6, "color": "#ff0000"})).unwrap();
        let red = |app: &PhotocraftApp| active(app).surface().unwrap().rgba(71, 30)[1] < 0.5;
        assert!(red(&app));
        app.ui.tool = Tool::Patch;
        let m = egui::Modifiers::NONE;
        // Outside any selection the drag is a lasso.
        tool_event(&mut app, ToolEvent::Down { x: 60.0, y: 20.0, pressure: 1.0 }, m);
        for [x, y] in [[84.0, 20.0], [84.0, 40.0], [60.0, 40.0]] {
            tool_event(&mut app, ToolEvent::Move { x, y, pressure: 1.0 }, m);
        }
        tool_event(&mut app, ToolEvent::Up { x: 60.0, y: 40.0 }, m);
        assert!(app.session.active().unwrap().doc.selection.is_some(), "lasso made a selection");
        assert!(red(&app), "the lasso alone changes no pixels");
        // ⇧-drag inside the selection adds to it rather than patching.
        assert!(!patch_drags_selection(&app, [70.0, 30.0], egui::Modifiers::SHIFT));
        // Dragging inside it patches from where it is dropped; the offset is limited to the canvas.
        assert!(patch_drags_selection(&app, [70.0, 30.0], m));
        assert_eq!(patch_offset(&mut app, [70.0, 30.0], [-200.0, 30.0]), [-60, 0]);
        tool_event(&mut app, ToolEvent::Down { x: 70.0, y: 30.0, pressure: 1.0 }, m);
        tool_event(&mut app, ToolEvent::Move { x: 50.0, y: 31.0, pressure: 1.0 }, m);
        tool_event(&mut app, ToolEvent::Up { x: 30.0, y: 30.0 }, m);
        assert!(!app.ui.status_error, "{}", app.ui.status);
        let px = active(&app).surface().unwrap().rgba(71, 30);
        assert!(px[0] > 0.95 && px[1] > 0.95 && px[2] > 0.95, "blemish patched with the white background: {px:?}");
    }

    #[test]
    fn patch_tool_content_aware_fills_from_where_it_is_dropped() {
        let mut app = app();
        app.run("paint.pencil", json!({"points": [[68, 30], [74, 30]], "size": 6, "color": "#ff0000"})).unwrap();
        app.run("select.rect", json!({"x": 60, "y": 20, "width": 24, "height": 20})).unwrap();
        app.ui.tool = Tool::Patch;
        app.ui.tool_options.patch_content_aware = true;
        app.ui.tool_options.patch_structure = 5.0;
        app.ui.tool_options.patch_color = 3.0;
        let m = egui::Modifiers::NONE;
        tool_event(&mut app, ToolEvent::Down { x: 70.0, y: 30.0, pressure: 1.0 }, m);
        tool_event(&mut app, ToolEvent::Move { x: 50.0, y: 31.0, pressure: 1.0 }, m);
        tool_event(&mut app, ToolEvent::Up { x: 30.0, y: 30.0 }, m);
        assert!(!app.ui.status_error, "{}", app.ui.status);
        let (id, p) = app.session.journal.last().cloned().unwrap();
        assert_eq!(id, "paint.patch");
        assert_eq!((p["contentAware"].as_bool(), p["structure"].as_f64(), p["color"].as_f64()), (Some(true), Some(5.0), Some(3.0)));
        let px = active(&app).surface().unwrap().rgba(71, 30);
        assert!(px[0] > 0.95 && px[1] > 0.95 && px[2] > 0.95, "blemish filled from the white background: {px:?}");
    }

    #[test]
    fn content_aware_move_tool_lassoes_then_moves_or_extends() {
        for (mode, keeps) in [("move", false), ("extend", true)] {
            let mut app = app();
            app.run("paint.pencil", json!({"points": [[20, 30], [24, 30]], "size": 6, "color": "#0000ff"})).unwrap();
            let blue = |app: &PhotocraftApp, x: i32| active(app).surface().unwrap().rgba(x, 30)[0] < 0.5;
            app.ui.tool = Tool::ContentAwareMove;
            app.ui.tool_options.cam_mode = mode.into();
            app.ui.tool_options.cam_structure = 7.0;
            let m = egui::Modifiers::NONE;
            tool_event(&mut app, ToolEvent::Down { x: 12.0, y: 20.0, pressure: 1.0 }, m);
            for [x, y] in [[32.0, 20.0], [32.0, 40.0], [12.0, 40.0]] {
                tool_event(&mut app, ToolEvent::Move { x, y, pressure: 1.0 }, m);
            }
            tool_event(&mut app, ToolEvent::Up { x: 12.0, y: 40.0 }, m);
            assert!(app.session.active().unwrap().doc.selection.is_some(), "lasso made a selection");
            assert!(blue(&app, 22), "the lasso alone changes no pixels");
            tool_event(&mut app, ToolEvent::Down { x: 22.0, y: 30.0, pressure: 1.0 }, m);
            tool_event(&mut app, ToolEvent::Move { x: 50.0, y: 31.0, pressure: 1.0 }, m);
            tool_event(&mut app, ToolEvent::Up { x: 72.0, y: 30.0 }, m);
            assert!(!app.ui.status_error, "{mode}: {}", app.ui.status);
            assert!(blue(&app, 72), "{mode}: the content lands where it was dropped");
            assert_eq!(blue(&app, 22), keeps, "{mode}: the original place");
            assert_eq!(app.session.journal.last().map(|(id, p)| (id.as_str(), p["mode"].as_str())), Some(("paint.contentAwareMove", Some(mode))));
            let sel = app.session.active().unwrap().doc.selection.as_ref().unwrap();
            assert!(sel.sample_channel(72, 30, 0) > 0.0 && sel.sample_channel(22, 30, 0) == 0.0, "{mode}: the selection follows");
        }
    }

    #[test]
    fn content_aware_move_transform_on_drop_turns_the_patch_on_enter() {
        let mut app = app();
        app.run("paint.pencil", json!({"points": [[16, 30], [28, 30]], "size": 4, "color": "#0000ff"})).unwrap();
        let blue = |app: &PhotocraftApp, x: i32, y: i32| active(app).surface().unwrap().rgba(x, y)[0] < 0.5;
        app.ui.tool = Tool::ContentAwareMove;
        app.ui.tool_options.cam_transform_on_drop = true;
        app.ui.tool_options.cam_structure = 7.0;
        let m = egui::Modifiers::NONE;
        let drag = |app: &mut PhotocraftApp, path: &[[f64; 2]]| {
            let ([x, y], rest) = (path[0], &path[1..]);
            tool_event(app, ToolEvent::Down { x, y, pressure: 1.0 }, m);
            for &[x, y] in rest {
                tool_event(app, ToolEvent::Move { x, y, pressure: 1.0 }, m);
            }
            let [x, y] = path[path.len() - 1];
            tool_event(app, ToolEvent::Up { x, y }, m);
        };
        drag(&mut app, &[[6.0, 14.0], [38.0, 14.0], [38.0, 46.0], [6.0, 46.0]]);
        let steps = app.session.active().unwrap().history.past_len();
        // The drop puts a box on the patch; nothing is computed yet.
        drag(&mut app, &[[22.0, 30.0], [50.0, 30.0], [72.0, 30.0]]);
        assert!(crate::cam_transform::active(&app), "{}", app.ui.status);
        assert!(blue(&app, 22, 30) && !blue(&app, 72, 30), "the document waits for the commit");
        assert_eq!(app.session.active().unwrap().history.past_len(), steps);
        // Dragging outside the box turns it 90° about its centre (72, 30).
        drag(&mut app, &[[98.0, 55.0], [60.0, 58.0], [47.0, 56.0]]);
        let place = crate::cam_transform::placement(app.ui.cam_transform.as_ref().unwrap()).unwrap();
        assert_eq!((place.offset, place.scale), ([50, 0], [100.0, 100.0]));
        assert!((place.rotate - 90.0).abs() < 1e-3, "{place:?}");
        crate::cam_transform::commit(&mut app);
        assert!(!crate::cam_transform::active(&app) && !app.ui.status_error, "{}", app.ui.status);
        let (id, p) = app.session.journal.last().unwrap();
        assert_eq!((id.as_str(), &p["offset"]), ("paint.contentAwareMove", &json!([50, 0])));
        assert!((p["rotate"].as_f64().unwrap() - 90.0).abs() < 1e-3, "{p}");
        // The horizontal line lands upright, in one History step.
        assert!(blue(&app, 72, 26) && blue(&app, 72, 34), "the turned line");
        assert!(!blue(&app, 66, 30) && !blue(&app, 78, 30), "no unturned line");
        assert!(!blue(&app, 22, 30), "the old place is filled");
        assert_eq!(app.session.active().unwrap().history.past_len(), steps + 1);
    }

    #[test]
    fn red_eye_click_corrects_a_pupil_and_errors_on_a_miss() {
        let mut app = app();
        app.run("paint.pencil", json!({"points": [[50, 30]], "size": 18, "hardness": 100, "color": "#f21818"})).unwrap();
        let before = active(&app).surface().unwrap().rgba(50, 30);
        assert!(before[0] > before[1] + 0.4, "{before:?}");
        app.ui.tool = Tool::RedEye;
        // Pupil Size 15 → search radius 22 px, so a click at (8,8) cannot reach the blob at (50,30).
        app.ui.tool_options.red_eye_pupil_size = 15.0;
        tool_event(&mut app, ToolEvent::Down { x: 50.0, y: 30.0, pressure: 1.0 }, egui::Modifiers::NONE);
        tool_event(&mut app, ToolEvent::Up { x: 50.0, y: 30.0 }, egui::Modifiers::NONE);
        assert!(!app.ui.status_error, "{}", app.ui.status);
        assert_eq!(app.session.journal.last().map(|(id, _)| id.as_str()), Some("paint.redEye"));
        let after = active(&app).surface().unwrap().rgba(50, 30);
        assert!(after[0] < before[0] - 0.15, "before {before:?} after {after:?}");
        tool_event(&mut app, ToolEvent::Down { x: 8.0, y: 8.0, pressure: 1.0 }, egui::Modifiers::NONE);
        tool_event(&mut app, ToolEvent::Up { x: 8.0, y: 8.0 }, egui::Modifiers::NONE);
        assert!(app.ui.status_error);
        assert!(app.ui.status.contains("no red-eye pixels"), "{}", app.ui.status);
    }

    #[test]
    fn red_eye_options_bar_shows_pupil_size_and_darken() {
        use crate::theme::ThemeKind;
        use egui::vec2;
        use egui_kittest::kittest::Queryable;
        let mut app = app();
        app.ui.tool = Tool::RedEye;
        let mut h = egui_kittest::Harness::builder().with_size(vec2(1400.0, 60.0)).build_ui_state(
            |ui, app: &mut PhotocraftApp| {
                if !ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                    return;
                }
                crate::panels::options_bar(app, ui);
            },
            app,
        );
        PhotocraftApp::setup_context(&h.ctx, ThemeKind::Studio);
        h.run_steps(4);
        h.get_by_label("Pupil Size");
        h.get_by_label("Darken Amount");
        assert_eq!(h.state().ui.tool_options.red_eye_pupil_size, 50.0);
        assert_eq!(h.state().ui.tool_options.red_eye_darken, 50.0);
    }

    #[test]
    fn patch_tool_says_to_drag_the_selection_after_the_lasso_and_after_a_click() {
        let hint = tl!("Now drag the selection onto another area");
        let m = egui::Modifiers::NONE;
        let mut app = app();
        app.ui.tool = Tool::Patch;
        app.ui.status.clear();
        // No selection yet: the drag is a lasso, which outlines the patch and says what comes next.
        tool_event(&mut app, ToolEvent::Down { x: 20.0, y: 10.0, pressure: 1.0 }, m);
        tool_event(&mut app, ToolEvent::Move { x: 40.0, y: 10.0, pressure: 1.0 }, m);
        tool_event(&mut app, ToolEvent::Move { x: 40.0, y: 30.0, pressure: 1.0 }, m);
        tool_event(&mut app, ToolEvent::Up { x: 20.0, y: 30.0 }, m);
        assert!(app.session.active().is_some_and(|st| st.doc.selection.is_some()), "the lasso made a selection");
        assert_eq!((app.ui.status.as_str(), app.ui.status_error), (hint, false));
        // A press and release inside the selection moves nothing: it says so instead of staying silent.
        app.ui.status.clear();
        tool_event(&mut app, ToolEvent::Down { x: 30.0, y: 20.0, pressure: 1.0 }, m);
        tool_event(&mut app, ToolEvent::Up { x: 30.0, y: 20.0 }, m);
        assert_eq!((app.ui.status.as_str(), app.ui.status_error), (hint, false));
        assert!(app.session.active().is_some_and(|st| st.doc.selection.is_some()), "the selection stays");
    }

    #[test]
    fn remove_tool_stroke_removes_what_it_covers() {
        let mut app = app();
        app.run("paint.pencil", json!({"points": [[0, 30], [100, 30]], "size": 200, "color": "#808080"})).unwrap();
        app.run("paint.pencil", json!({"points": [[30, 20], [30, 40]], "size": 4, "color": "#000000"})).unwrap();
        let bg = active(&app).surface().unwrap().rgba(70, 30);
        app.ui.tool = Tool::Remove;
        let m = egui::Modifiers::NONE;
        tool_event(&mut app, ToolEvent::Down { x: 30.0, y: 18.0, pressure: 1.0 }, m);
        tool_event(&mut app, ToolEvent::Move { x: 30.0, y: 42.0, pressure: 1.0 }, m);
        tool_event(&mut app, ToolEvent::Up { x: 30.0, y: 42.0 }, m);
        assert!(!app.ui.status_error, "{}", app.ui.status);
        assert_eq!(app.session.journal.last().map(|(id, _)| id.as_str()), Some("paint.remove"));
        let px = active(&app).surface().unwrap().rgba(30, 30);
        assert!(px.iter().zip(bg).all(|(a, b)| (a - b).abs() < 0.02), "{px:?} vs {bg:?}");
    }

    #[test]
    fn remove_after_each_stroke_off_collects_strokes_until_enter_or_escape() {
        let mut app = app();
        app.run("paint.pencil", json!({"points": [[0, 30], [100, 30]], "size": 200, "color": "#808080"})).unwrap();
        app.run("paint.pencil", json!({"points": [[30, 20], [30, 40]], "size": 4, "color": "#000000"})).unwrap();
        app.ui.tool = Tool::Remove;
        app.ui.tool_options.remove_after_each_stroke = false;
        let m = egui::Modifiers::NONE;
        let journal = app.session.journal.len();
        for x in [28.0, 32.0] {
            tool_event(&mut app, ToolEvent::Down { x, y: 18.0, pressure: 1.0 }, m);
            tool_event(&mut app, ToolEvent::Move { x, y: 42.0, pressure: 1.0 }, m);
            tool_event(&mut app, ToolEvent::Up { x, y: 42.0 }, m);
        }
        // Nothing ran yet: the strokes wait, tinted, and the status says so.
        assert_eq!(app.session.journal.len(), journal, "no command before Enter");
        assert_eq!(app.remove_pending.len(), 2);
        assert!(app.remove_trail.is_some());
        assert!(app.ui.status.contains("2") && !app.ui.status_error, "{}", app.ui.status);
        // Esc drops them.
        assert!(remove_pending_cancel(&mut app));
        assert!(app.remove_pending.is_empty() && app.remove_trail.is_none());
        assert!(!remove_pending_cancel(&mut app));
        // Collected again, Enter applies them as one `paint.remove` with both strokes.
        for x in [28.0, 32.0] {
            tool_event(&mut app, ToolEvent::Down { x, y: 18.0, pressure: 1.0 }, m);
            tool_event(&mut app, ToolEvent::Up { x, y: 42.0 }, m);
        }
        let bg = active(&app).surface().unwrap().rgba(70, 30);
        assert!(remove_pending_commit(&mut app));
        assert!(!app.ui.status_error, "{}", app.ui.status);
        let (id, p) = app.session.journal.last().cloned().expect("a journaled command");
        assert_eq!(id, "paint.remove");
        assert_eq!(p["strokes"].as_array().map(Vec::len), Some(2));
        assert!(p.get("points").is_none());
        assert!(app.remove_pending.is_empty());
        let px = active(&app).surface().unwrap().rgba(30, 30);
        assert!(px.iter().zip(bg).all(|(a, b)| (a - b).abs() < 0.02), "{px:?} vs {bg:?}");
        assert!(!remove_pending_commit(&mut app), "nothing left to apply");
    }

    #[test]
    fn remove_strokes_waiting_are_dropped_when_the_tool_changes() {
        let mut app = app();
        app.ui.tool = Tool::Remove;
        app.ui.tool_options.remove_after_each_stroke = false;
        remove_pending_add(&mut app, &[[10.0, 10.0, 1.0], [20.0, 10.0, 1.0]]);
        assert_eq!(app.remove_pending.len(), 1);
        assert!(remove_pending_live(&mut app));
        app.ui.tool = Tool::SpotHealing;
        assert!(!remove_pending_live(&mut app));
        assert!(app.remove_pending.is_empty(), "the strokes belong to the Remove tool");
    }

    #[test]
    fn create_a_new_layer_sends_new_layer_to_the_command() {
        let mut app = app();
        app.run("paint.pencil", json!({"points": [[0, 30], [100, 30]], "size": 200, "color": "#808080"})).unwrap();
        app.ui.tool = Tool::Remove;
        app.ui.tool_options.remove_new_layer = true;
        let layers = app.session.active().unwrap().doc.walk().len();
        let m = egui::Modifiers::NONE;
        tool_event(&mut app, ToolEvent::Down { x: 30.0, y: 18.0, pressure: 1.0 }, m);
        tool_event(&mut app, ToolEvent::Up { x: 30.0, y: 42.0 }, m);
        assert!(!app.ui.status_error, "{}", app.ui.status);
        let (id, p) = app.session.journal.last().cloned().unwrap();
        assert_eq!((id.as_str(), p["newLayer"].as_bool()), ("paint.remove", Some(true)));
        assert_eq!(app.session.active().unwrap().doc.walk().len(), layers + 1, "the fill went to a new layer");
    }

    #[test]
    fn red_eye_is_in_the_j_flyout() {
        let j = [Tool::Remove, Tool::SpotHealing, Tool::Healing, Tool::Patch, Tool::ContentAwareMove, Tool::RedEye];
        assert!(j.contains(&Tool::RedEye));
        assert!(j.iter().all(|t| t.key() == 'J'));
    }

    #[test]
    fn j_cycles_from_remove() {
        // J from another group picks the first J tool, and repeated J goes on in this order.
        let group: Vec<Tool> = Tool::ALL.iter().copied().filter(|t| t.key() == 'J').collect();
        assert_eq!(group, [Tool::Remove, Tool::SpotHealing, Tool::Healing, Tool::Patch, Tool::ContentAwareMove, Tool::RedEye]);
    }

    #[test]
    fn s_group_is_clone_stamp_then_pattern_stamp() {
        assert_eq!(Tool::CloneStamp.key(), 'S');
        assert_eq!(Tool::PatternStamp.key(), 'S');
        let group: Vec<Tool> = Tool::ALL.iter().copied().filter(|t| t.key() == 'S').collect();
        assert_eq!(group, vec![Tool::CloneStamp, Tool::PatternStamp]);
        assert_eq!(Tool::from_name("patternStamp"), Some(Tool::PatternStamp));
    }

    #[test]
    fn pattern_stamp_stroke_paints_pixels_and_keeps_phase_when_aligned() {
        let mut app = app();
        assert!(finish_stroke(&mut app, Tool::PatternStamp, &[[20.0, 30.0, 1.0], [50.0, 30.0, 1.0]], egui::Modifiers::NONE));
        assert!(!app.ui.status_error, "{}", app.ui.status);
        assert_eq!(app.session.journal.last().map(|(id, _)| id.as_str()), Some("paint.patternStamp"));
        assert!(app.ui.pattern_stamp_phase.is_some());
        let px = active(&app).surface().unwrap().rgba(30, 30);
        assert!(px[3] > 0.5, "the stamp paints through the brush: {px:?}");
        app.ui.tool_options.pattern_stamp_aligned = false;
        app.ui.pattern_stamp_phase = None;
        assert!(finish_stroke(&mut app, Tool::PatternStamp, &[[70.0, 30.0, 1.0]], egui::Modifiers::NONE));
        assert!(app.ui.pattern_stamp_phase.is_none(), "unaligned strokes do not keep phase");
    }

    #[test]
    fn pattern_stamp_does_not_require_a_clone_source() {
        let mut app = app();
        assert!(app.ui.clone_source.is_none());
        assert!(finish_stroke(&mut app, Tool::PatternStamp, &[[12.0, 30.0, 1.0]], egui::Modifiers::NONE));
        assert!(!app.ui.status_error, "{}", app.ui.status);
        assert!(app.ui.clone_source.is_none());
    }
}
