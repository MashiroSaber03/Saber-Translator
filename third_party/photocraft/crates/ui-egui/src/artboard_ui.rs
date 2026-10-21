//! Artboards in the shell: the pasteboard between boards and the board names on the canvas,
//! View › Fit Artboard on Screen, and the Properties panel section (X/Y/W/H, preset,
//! background). Everything that changes the document is `layer.artboard.set` and friends.

use egui::{Align2, Color32, Pos2, Rect, Stroke};
use photocraft_doc::{ArtboardBackground, Document, Layer};
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::canvas::ViewXform;
use crate::theme::Tokens;

/// Screen rects covering the document image minus every artboard (vertical slabs between the
/// boards' x edges, split around the boards they cross).
pub fn pasteboard_rects(xf: &ViewXform, doc: &Document) -> Vec<Rect> {
    let canvas = doc.bounds();
    let boards: Vec<photocraft_geom::Rect> = doc.artboards().iter().map(|b| b.2.rect.intersect(&canvas)).filter(|r| !r.is_empty()).collect();
    let mut xs: Vec<i32> = vec![canvas.x0, canvas.x1];
    for b in &boards {
        xs.extend([b.x0, b.x1]);
    }
    xs.sort_unstable();
    xs.dedup();
    let mut out = Vec::new();
    for w in xs.windows(2) {
        let (x0, x1) = (w[0], w[1]);
        let mut spans: Vec<(i32, i32)> = boards.iter().filter(|b| b.x0 <= x0 && b.x1 >= x1).map(|b| (b.y0, b.y1)).collect();
        spans.sort_unstable();
        let mut y = canvas.y0;
        for (a, b) in spans {
            if a > y {
                out.push(xf.doc_rect(photocraft_geom::Rect::new(x0, y, x1, a)));
            }
            y = y.max(b);
        }
        if y < canvas.y1 {
            out.push(xf.doc_rect(photocraft_geom::Rect::new(x0, y, x1, canvas.y1)));
        }
    }
    out
}

/// Board outlines and names (the active board's name is highlighted, as in Photoshop).
pub fn draw_frames(app: &PhotocraftApp, painter: &egui::Painter, xf: &ViewXform) {
    let Some(st) = app.session.active() else { return };
    let t = Tokens::get(painter.ctx());
    let active = st.active_layer.and_then(|id| st.doc.artboard_of(id));
    for (id, name, a) in st.doc.artboards() {
        let r = xf.doc_rect(a.rect);
        let on = active == Some(id);
        painter.rect_stroke(r, 0.0, Stroke::new(1.0, if on { t.accent } else { Color32::from_black_alpha(90) }), egui::StrokeKind::Outside);
        let pos = Pos2::new(r.left(), r.top() - 4.0);
        painter.text(pos, Align2::LEFT_BOTTOM, name, egui::FontId::proportional(11.0), if on { t.text } else { t.text_dim });
    }
}

/// View › Fit Artboard on Screen: zoom to the active artboard (or the first one).
pub fn fit_artboard(app: &mut PhotocraftApp) -> Result<Value, String> {
    let i = app.session.active_index().ok_or("no document")?;
    let st = app.session.active().ok_or("no document")?;
    let id = st.active_layer.and_then(|l| st.doc.artboard_of(l)).or_else(|| st.doc.artboards().last().map(|b| b.0)).ok_or("the document has no artboards")?;
    let b = st.doc.layer(id).and_then(Layer::artboard).map(|a| a.rect).ok_or("no artboard")?;
    let size = [st.doc.size.width, st.doc.size.height];
    let area = app.last_canvas_rect.size();
    let area = if area.x > 50.0 { area } else { egui::vec2(1200.0, 800.0) };
    let ppp = app.canvas_ppp();
    let v = &mut app.ui.views[i];
    // Leave room for the name above the board. The area is in egui points; the stored zoom is
    // device pixels per document pixel.
    v.zoom = ((area.x - 60.0) / b.width().max(1) as f32).min((area.y - 80.0) / b.height().max(1) as f32) * ppp;
    v.zoom = crate::zoom_levels::clamp(v.zoom, size);
    // Widen before adding: a board near ±2^30, or one whose far edge saturated at i32::MAX,
    // overflows an i32 sum (#981).
    v.center = [((f64::from(b.x0) + f64::from(b.x1)) / 2.0) as f32, ((f64::from(b.y0) + f64::from(b.y1)) / 2.0) as f32];
    v.fit_pending = false;
    Ok(json!({"zoom": v.zoom, "artboard": id.0, "bounds": [b.x0, b.y0, b.x1, b.y1]}))
}

/// Properties panel for an artboard.
pub fn properties(app: &mut PhotocraftApp, ui: &mut egui::Ui, layer: &Layer) {
    let Some(a) = layer.artboard().cloned() else { return };
    let t = Tokens::get(ui.ctx());
    let key = |k: &str| format!("artboard-{}-{k}", layer.id.0);
    let mut edit: Option<Value> = None;
    // The custom background swatch was clicked: its colour, for the Color Picker.
    let mut pick: Option<[f32; 3]> = None;
    ui.label(egui::RichText::new(tl!("Artboard")).font(crate::theme::semibold(12.0)).color(t.text));
    ui.add_space(4.0);
    let mut preset = a.preset.clone();
    let mut opts: Vec<(String, &str)> = vec![(String::new(), tl!("Custom"))];
    opts.extend(photocraft_engine::artboard_cmds::PRESETS.iter().map(|(n, _, _)| (n.to_string(), *n)));
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(tl!("Size")).color(t.text_dim).size(12.0));
        if crate::widgets::dropdown(ui, &key("preset"), &mut preset, &opts, 150.0) && !preset.is_empty() {
            edit = Some(json!({"layer": layer.id.0, "preset": preset}));
        }
    });
    let num = |ui: &mut egui::Ui, label: &str, v: &mut f32, range: std::ops::RangeInclusive<f32>| -> bool {
        ui.label(egui::RichText::new(tl!(&label)).color(t.text_dim).size(12.0));
        crate::widgets::value_field(ui, v, range, "px", 64.0).changed()
    };
    let r = a.rect;
    ui.horizontal(|ui| {
        let (mut w, mut h) = (r.width() as f32, r.height() as f32);
        let cw = num(ui, "W", &mut w, 1.0..=300_000.0);
        let ch = num(ui, "H", &mut h, 1.0..=300_000.0);
        if cw || ch {
            edit = Some(json!({"layer": layer.id.0, "width": w.round(), "height": h.round(), "coalesce": key("wh")}));
        }
    });
    ui.horizontal(|ui| {
        let (mut x, mut y) = (r.x0 as f32, r.y0 as f32);
        let cx = num(ui, "X", &mut x, -300_000.0..=300_000.0);
        let cy = num(ui, "Y", &mut y, -300_000.0..=300_000.0);
        if cx || cy {
            edit = Some(json!({"layer": layer.id.0, "x": x.round(), "y": y.round(), "coalesce": key("xy")}));
        }
    });
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(tl!("Artboard Background Color")).color(t.text_dim).size(12.0));
    });
    ui.horizontal(|ui| {
        let mut bg = a.background.name().to_string();
        let opts = [
            ("white".to_string(), tl!("White")),
            ("black".to_string(), tl!("Black")),
            ("transparent".to_string(), tl!("Transparent")),
            ("custom".to_string(), tl!("Other…")),
        ];
        if crate::widgets::dropdown(ui, &key("bg"), &mut bg, &opts, 120.0) {
            let mut params = json!({"layer": layer.id.0, "background": bg});
            if bg == "custom" {
                let hex = match a.background {
                    ArtboardBackground::Custom(c) => {
                        let [r8, g8, b8, _] = c.to_rgba8();
                        format!("#{:02x}{:02x}{:02x}", r8, g8, b8)
                    }
                    _ => "#ffffff".to_string(),
                };
                params["color"] = json!(hex);
            }
            edit = Some(params);
        }
        if let ArtboardBackground::Custom(c) = a.background {
            let [r8, g8, b8, _] = c.to_rgba8();
            // PhotoCraft's Color Picker, as everywhere else (#2144): OK sets the background.
            if crate::widgets::color_swatch_button(ui, egui::Color32::from_rgb(r8, g8, b8), tl!("Color")).clicked() {
                pick = Some(c.to_rgb());
            }
        }
    });
    if let Some(p) = edit
        && let Err(e) = app.run("layer.artboard.set", p)
    {
        app.ui.status = e;
        app.ui.status_error = true;
    }
    if let Some(rgb) = pick {
        let title = crate::color_picker_ui::title_for(tl!("Color"));
        crate::color_picker_ui::open_for_command(app, &title, rgb, "layer.artboard.set", json!({"layer": layer.id.0, "background": "custom"}));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::menus::{invoke as menu, is_live, menu_items};

    fn app() -> (PhotocraftApp, egui::Context) {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 100, "height": 50})).unwrap();
        app.sync_views();
        (app, egui::Context::default())
    }

    #[test]
    fn pasteboard_is_the_canvas_minus_the_boards() {
        let (mut app, _) = app();
        app.run("layer.new.artboard", json!({"rect": [0, 0, 40, 50]})).unwrap();
        app.run("layer.new.artboard", json!({"rect": [60, 10, 40, 30]})).unwrap();
        let doc = app.session.active().unwrap().doc.clone();
        let xf = ViewXform {
            rect: Rect::from_min_size(Pos2::ZERO, egui::vec2(100.0, 50.0)),
            zoom: 1.0,
            center: [50.0, 25.0],
            flip: false,
            rotation: 0.0,
            aspect: 1.0,
        };
        let rects = pasteboard_rects(&xf, &doc);
        let area: f32 = rects.iter().map(|r| r.area()).sum();
        // 100×50 canvas − 40×50 − 40×30 boards.
        assert!((area - (5000.0 - 2000.0 - 1200.0)).abs() < 0.5, "{rects:?}");
        assert!(rects.iter().all(|r| !r.intersects(Rect::from_min_max(Pos2::new(0.5, 0.5), Pos2::new(39.5, 49.5)))));
    }

    #[test]
    fn fit_artboard_and_menu_items() {
        let (mut app, ctx) = app();
        for id in [
            "view.fitArtboardOnScreen",
            "window.panel.layerComps",
            "layer.new.artboard",
            "file.export.artboardsToFiles",
            "file.export.artboardsToPdf",
            "file.export.layerCompsToFiles",
            "view.clearSelectedArtboardGuides",
        ] {
            assert!(is_live(id), "{id}");
        }
        let enabled = |app: &PhotocraftApp, id: &str| menu_items(app).into_iter().find(|i| i.id == id).is_some_and(|i| i.enabled);
        assert!(!enabled(&app, "view.fitArtboardOnScreen"));
        // Menu clicks without params open dialogs for the parameterised commands.
        let r = menu(&mut app, &ctx, "layer.new.artboard", json!({})).unwrap();
        assert!(r.get("dialog").is_some());
        app.run("layer.new.artboard", json!({"rect": [200, 0, 50, 20]})).unwrap();
        assert!(enabled(&app, "view.fitArtboardOnScreen"));
        let r = menu(&mut app, &ctx, "view.fitArtboardOnScreen", json!({})).unwrap();
        assert_eq!(r["bounds"], json!([200, 0, 250, 20]));
        let i = app.session.active_index().unwrap();
        assert_eq!(app.ui.views[i].center, [225.0, 10.0]);
        assert!(menu(&mut app, &ctx, "file.export.artboardsToFiles", json!({})).unwrap().get("dialog").is_some());
        // Window › Layer Comps shows the third History tab.
        menu(&mut app, &ctx, "window.panel.layerComps", json!({})).unwrap();
        assert!(app.ui.panels.history);
        assert_eq!(app.ui.dock_tabs.history, 2);
    }

    #[test]
    fn fit_artboard_centres_a_board_with_large_coordinates() {
        // #981: `(b.x0 + b.x1) as f32` overflowed i32 for boards near ±2^30 or the i32 edges,
        // panicking outside any dispatch guard.
        for (params, want) in [
            (json!({"x": 2147483000, "y": 0, "width": 600, "height": 1000}), [(2147483000.0 + 2147483600.0) / 2.0, 500.0]),
            (json!({"x": 1073741824, "width": 50, "height": 20}), [1073741849.0, 10.0]),
            (json!({"y": -1073741840, "width": 50, "height": 20}), [25.0, -1073741830.0]),
        ] {
            let (mut app, ctx) = app();
            let i = app.session.active_index().unwrap();
            app.run("layer.new.artboard", params.clone()).unwrap();
            app.sync_views();
            menu(&mut app, &ctx, "view.fitArtboardOnScreen", json!({})).unwrap();
            // The view centre is f32, so compare against the true midpoint rounded to f32.
            let c = app.ui.views[i].center;
            assert_eq!(c, [want[0] as f32, want[1] as f32], "{params}");
        }
    }

    fn drag(app: &mut PhotocraftApp, from: [f64; 2], by: [f64; 2], mods: egui::Modifiers) {
        use crate::canvas::{ToolEvent, tool_event};
        tool_event(app, ToolEvent::Down { x: from[0], y: from[1], pressure: 1.0 }, mods);
        for t in [0.5, 1.0] {
            tool_event(app, ToolEvent::Move { x: from[0] + by[0] * t, y: from[1] + by[1] * t, pressure: 1.0 }, mods);
        }
        tool_event(app, ToolEvent::Up { x: from[0] + by[0], y: from[1] + by[1] }, mods);
    }

    fn boards(app: &PhotocraftApp) -> Vec<photocraft_geom::Rect> {
        app.session.active().unwrap().doc.artboards().iter().map(|b| b.2.rect).collect()
    }

    /// #1531: the Move tool drags a board by an empty spot on it (Auto-Select picks the board, not
    /// the Background under it), and ⌥-drag drags a copy off the original.
    #[test]
    fn move_tool_drags_an_artboard_by_its_empty_area() {
        let (mut app, _) = app();
        app.run("layer.new.artboard", json!({"rect": [0, 0, 40, 40]})).unwrap();
        let bg = app.session.active().unwrap().doc.layers[0].id.0;
        app.run("layer.select", json!({"layer": bg})).unwrap();
        app.ui.tool = crate::state::Tool::Move;
        app.ui.extras.snap = false;
        app.ui.view.show.smart_guides = false;
        drag(&mut app, [20.0, 20.0], [10.0, 5.0], egui::Modifiers::NONE);
        assert_eq!(boards(&app), vec![photocraft_geom::Rect::new(10, 5, 50, 45)]);
        let st = app.session.active().unwrap();
        assert_eq!(st.active_layer, Some(st.doc.artboards()[0].0));
        // ⌥-drag: the copy follows the pointer, the original stays.
        drag(&mut app, [20.0, 20.0], [50.0, 0.0], egui::Modifiers::ALT);
        let mut got = boards(&app);
        got.sort_by_key(|r| r.x0);
        assert_eq!(got, vec![photocraft_geom::Rect::new(10, 5, 50, 45), photocraft_geom::Rect::new(60, 5, 100, 45)]);
        // One undo step takes the copy back.
        app.run("edit.undo", json!({})).unwrap();
        assert_eq!(boards(&app).len(), 1);
    }

    #[test]
    fn comp_target_follows_selection_then_last_applied() {
        let (mut app, _) = app();
        assert_eq!(crate::comps_ui::target(&app), None);
        let a = app.run("layerComp.new", json!({})).unwrap()["comp"].as_u64().unwrap() as u32;
        let b = app.run("layerComp.new", json!({})).unwrap()["comp"].as_u64().unwrap() as u32;
        assert_eq!(crate::comps_ui::target(&app), Some(b));
        app.ui.layer_comp_selected = Some(a);
        assert_eq!(crate::comps_ui::target(&app), Some(a));
        app.ui.layer_comp_selected = Some(999);
        assert_eq!(crate::comps_ui::target(&app), Some(b));
    }
}
