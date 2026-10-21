//! Window › Swatches: the swatch library (`photocraft_engine::presets::swatches`) in the shared
//! preset browser: groups with disclosure triangles, a colour grid with name tooltips, a context
//! menu (Rename, Delete, Move to), and the footer (size · New Group · New Swatch · Delete).
//!
//! As in Photoshop, clicking a swatch sets the foreground colour (and recolours selected type);
//! Alt- or Ctrl/Cmd-clicking sets the background colour; double-clicking renames it. The
//! panel menu (the group's "…" button) adds New Swatch, New Group, Import/Replace (`.aco`,
//! `.ase`), Export (`.aco`) and Export for Exchange (`.ase`) and Reset. Every action is a
//! `swatches.*` engine command; this module only draws and dispatches. Colours are shown in the
//! RGB working space through `photocraft-cms` (cached per library revision and colour settings).

use std::sync::Arc;

use egui::{Color32, CornerRadius, Rect};
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::preset_panels::{Ev, GroupView, ItemView, Place, browser, run};

/// The browser's panel key (selection, folder state and thumbnail size live under it).
pub const PANEL: &str = "swatches";
/// Default cell size: Photoshop's small swatches.
const DEFAULT_SIZE: f32 = 22.0;

fn c32(c: [f32; 3]) -> Color32 {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    Color32::from_rgb(q(c[0]), q(c[1]), q(c[2]))
}

/// Display colours of every swatch, rebuilt when the library or the working spaces change.
fn colors(app: &PhotocraftApp, ctx: &egui::Context) -> Arc<Vec<Vec<Color32>>> {
    let c = &app.session.color.settings;
    let key = egui::Id::new(("swatch-colors", app.session.presets.rev, &c.working_rgb, &c.working_cmyk, &c.working_gray));
    if let Some(v) = ctx.data(|d| d.get_temp::<Arc<Vec<Vec<Color32>>>>(key)) {
        return v;
    }
    let conv = photocraft_engine::presets::swatches::SwatchConv::new(&app.session);
    let v: Arc<Vec<Vec<Color32>>> = Arc::new(app.session.presets.swatches.iter().map(|g| g.items.iter().map(|s| c32(conv.rgb(&s.color))).collect()).collect());
    ctx.data_mut(|d| {
        // Only the newest table is kept.
        if let Some(old) = d.get_temp::<egui::Id>(egui::Id::new("swatch-colors-key")) {
            d.remove::<Arc<Vec<Vec<Color32>>>>(old);
        }
        d.insert_temp(egui::Id::new("swatch-colors-key"), key);
        d.insert_temp(key, v.clone());
    });
    v
}

/// Set the foreground (or background) colour from a swatch.
pub fn use_swatch(app: &mut PhotocraftApp, name: &str, background: bool) {
    let target = if background { "background" } else { "foreground" };
    if run(app, "swatches.use", json!({"swatch": name, "target": target})).is_some() && !background {
        // Clicking a swatch while characters are selected recolours them (#364).
        crate::type_tool::foreground_changed(app);
    }
}

/// The Swatches panel body.
pub fn panel(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let lib = &app.session.presets.swatches;
    let groups: Vec<GroupView> = lib
        .iter()
        .map(|g| GroupView { name: g.name.clone(), items: g.items.iter().map(|s| ItemView { key: s.name.clone(), name: s.name.clone() }).collect() })
        .collect();
    let table = colors(app, ui.ctx());
    let canvas = app.last_canvas_rect;
    let mut st = std::mem::take(&mut app.ui.presets_ui);
    st.sizes.entry(PANEL.to_string()).or_insert(DEFAULT_SIZE);
    let max_h = (ui.available_height() - 30.0).max(60.0);
    let card_border = crate::theme::Tokens::get(ui.ctx()).card_border;
    let events = browser(ui, &mut st, PANEL, &groups, Place { canvas, max_h, new_tip: tl!("Create new swatch") }, &mut |ui, r: Rect, gi, ii| {
        let c = table.get(gi).and_then(|g| g.get(ii)).copied().unwrap_or(Color32::GRAY);
        ui.painter().rect_filled(r, CornerRadius::same(2), c);
        ui.painter().rect_stroke(r, CornerRadius::same(2), egui::Stroke::new(1.0, card_border), egui::StrokeKind::Inside);
    });
    app.ui.presets_ui = st;
    let mods = ui.input(|i| i.modifiers);
    for e in events {
        handle(app, e, mods.alt || mods.command || mods.ctrl);
    }
}

fn handle(app: &mut PhotocraftApp, e: Ev, background: bool) {
    let sel = |app: &mut PhotocraftApp, k: &str| {
        app.ui.presets_ui.selected.insert(PANEL.into(), k.to_string());
    };
    match e {
        Ev::Select(k) => {
            sel(app, &k);
            use_swatch(app, &k, background);
        }
        Ev::Activate(k) => {
            sel(app, &k);
            app.ui.presets_ui.renaming = Some((PANEL.into(), k.clone(), k));
        }
        // Swatches don't drop onto the canvas.
        Ev::Drop(..) => {}
        Ev::Rename(k, n) => {
            if run(app, "swatches.rename", json!({"swatch": k, "name": n})).is_some() && app.ui.presets_ui.selected.get(PANEL) == Some(&k) {
                sel(app, &n);
            }
        }
        Ev::Delete(k) => {
            if run(app, "swatches.delete", json!({"swatch": k})).is_some() {
                app.ui.presets_ui.selected.remove(PANEL);
            }
        }
        Ev::Move(k, g) => drop(run(app, "swatches.move", json!({"swatch": k, "to": g}))),
        Ev::RenameGroup(g, n) => drop(run(app, "swatches.renameGroup", json!({"group": g, "name": n}))),
        Ev::DeleteGroup(g) => drop(run(app, "swatches.deleteGroup", json!({"group": g}))),
        Ev::NewGroup => new_group(app),
        Ev::New => new_swatch(app),
    }
}

/// New Swatch: the foreground colour, named and selected, then renamed in place (Photoshop asks
/// for the name).
fn new_swatch(app: &mut PhotocraftApp) {
    let group = app
        .ui
        .presets_ui
        .selected
        .get(PANEL)
        .and_then(|k| app.session.presets.swatches.iter().find(|g| g.items.iter().any(|s| &s.name == k)).map(|g| g.name.clone()));
    let p = group.map_or_else(|| json!({}), |g| json!({"group": g}));
    if let Some(r) = run(app, "swatches.add", p) {
        let name = r["name"].as_str().unwrap_or_default().to_string();
        app.ui.presets_ui.selected.insert(PANEL.into(), name.clone());
        app.ui.presets_ui.renaming = Some((PANEL.into(), name.clone(), name));
    }
}

fn new_group(app: &mut PhotocraftApp) {
    if let Some(r) = run(app, "swatches.newGroup", json!({"name": "Group"})) {
        let g = r["group"].as_str().unwrap_or_default().to_string();
        app.ui.presets_ui.renaming = Some((PANEL.into(), format!("group:{g}"), g));
    }
}

/// Command params for a picked file: a real path keeps the journal small; browser files arrive
/// as bytes.
fn source(name: &str, bytes: &[u8]) -> Value {
    #[cfg(not(target_arch = "wasm32"))]
    if std::path::Path::new(name).is_absolute() && std::path::Path::new(name).is_file() {
        return json!({"path": name});
    }
    let _ = name;
    json!({"data": photocraft_engine::paint::tile::b64_encode(bytes)})
}

/// Import an `.aco`/`.ase` file's bytes (File › Open, drag and drop, the panel menu).
pub fn import_bytes(app: &mut PhotocraftApp, name: &str, bytes: &[u8], replace: bool) -> Result<Value, String> {
    let stem = std::path::Path::new(name).file_stem().map(|s| s.to_string_lossy().to_string()).filter(|s| !s.trim().is_empty());
    let mut p = source(name, bytes);
    p["mode"] = json!(if replace { "replace" } else { "append" });
    if let Some(stem) = stem {
        p["group"] = json!(stem);
    }
    let r = app.run("swatches.import", p)?;
    let n = r["count"].as_u64().unwrap_or(0);
    let warnings: Vec<String> = r.get("warnings").and_then(|w| serde_json::from_value(w.clone()).ok()).unwrap_or_default();
    let shown = crate::file_open::display_name(name);
    app.ui.status_error = false;
    let imported = crate::i18n::fmt(
        &crate::i18n::trn(crate::i18n::current(), n, "Imported {n} swatch from {file}", "Imported {n} swatches from {file}"),
        &[("file", &shown)],
    );
    app.ui.status = match warnings.first() {
        Some(warning) => {
            let template = "{message} (notes: {n}; {warning})";
            crate::i18n::fmt(tl!(template), &[("message", &imported), ("n", &warnings.len().to_string()), ("warning", warning)])
        }
        None => imported,
    };
    app.ui.panels.color = true;
    Ok(r)
}

fn import_dialog(app: &mut PhotocraftApp, replace: bool) -> Result<Value, String> {
    app.pick_file_bytes(move |app, name, bytes| import_bytes(app, &name, &bytes, replace))
}

fn export_dialog(app: &mut PhotocraftApp, format: &str) -> Result<Value, String> {
    let r = app.run("swatches.export", json!({"format": format}))?;
    let bytes = r["data"].as_str().and_then(photocraft_engine::paint::tile::b64_decode).ok_or("the export returned no data")?;
    let n = r["count"].as_u64().unwrap_or(0);
    app.pick_save(&format!("Swatches.{format}"), move |app, path| {
        let write = app.services.write.as_mut().ok_or("no writer configured")?;
        write(&path, &bytes)?;
        app.ui.status_error = false;
        app.ui.status = crate::i18n::fmt(
            &crate::i18n::trn(crate::i18n::current(), n, "Exported {n} swatch to {file}", "Exported {n} swatches to {file}"),
            &[("file", &crate::file_open::display_name(&path))],
        );
        Ok(json!({"path": path, "count": n}))
    })
}

/// The Swatches entries of the panel group's menu ("…").
pub fn panel_menu(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let empty = app.session.presets.swatches.iter().all(|g| g.items.is_empty());
    let mut act: Option<&str> = None;
    if ui.button(tl!("New Swatch…")).clicked() {
        act = Some("new");
    }
    if ui.button(tl!("New Swatch Group…")).clicked() {
        act = Some("group");
    }
    ui.separator();
    if ui.button(tl!("Import Swatches…")).clicked() {
        act = Some("import");
    }
    if ui.button(tl!("Replace Swatches…")).clicked() {
        act = Some("replace");
    }
    if ui.add_enabled(!empty, egui::Button::new(tl!("Export Swatches…"))).clicked() {
        act = Some("aco");
    }
    if ui.add_enabled(!empty, egui::Button::new(tl!("Export Swatches for Exchange…"))).clicked() {
        act = Some("ase");
    }
    ui.separator();
    if ui.button(tl!("Reset Swatches")).clicked() {
        act = Some("reset");
    }
    let Some(a) = act else { return };
    ui.close();
    let r = match a {
        "new" => {
            new_swatch(app);
            Ok(Value::Null)
        }
        "group" => {
            new_group(app);
            Ok(Value::Null)
        }
        "import" => import_dialog(app, false),
        "replace" => import_dialog(app, true),
        "aco" | "ase" => export_dialog(app, a),
        _ => app.run("swatches.reset", json!({})),
    };
    if let Err(e) = r
        && e != "cancelled"
    {
        app.ui.status = e;
        app.ui.status_error = true;
    }
}

/// The Color Picker's Add to Swatches: the picked colour, in the model the user typed it in
/// (CMYK or Lab values kept by the picker), else RGB.
pub fn add_from_picker(app: &mut PhotocraftApp, f: &serde_json::Map<String, Value>) {
    let color = if let Some(c) = f.get("__cmyk").and_then(|v| serde_json::from_value::<[f32; 4]>(v.clone()).ok()) {
        json!({"model": "cmyk", "values": c.map(|v| v * 100.0)})
    } else if let Some(l) = f.get("__lab").and_then(|v| serde_json::from_value::<[f32; 3]>(v.clone()).ok()) {
        json!({"model": "lab", "values": l})
    } else {
        json!(f.get("color").and_then(Value::as_str).unwrap_or("#000000"))
    };
    if let Some(r) = run(app, "swatches.add", json!({"color": color})) {
        let name = r["name"].as_str().unwrap_or_default().to_string();
        app.ui.status_error = false;
        let template = "Added swatch {name}";
        app.ui.status = crate::i18n::fmt(tl!(template), &[("name", &name)]);
        app.ui.presets_ui.selected.insert(PANEL.into(), name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Tool;
    use egui::{Modifiers, vec2};
    use egui_kittest::Harness;

    fn click(h: &mut Harness<'_, PhotocraftApp>, p: egui::Pos2, modifiers: Modifiers) {
        h.hover_at(p);
        // Held modifiers, as when the user holds Alt while clicking.
        h.event(egui::Event::ModifiersChanged(modifiers));
        h.run_steps(1);
        h.event(egui::Event::PointerButton { pos: p, button: egui::PointerButton::Primary, pressed: true, modifiers });
        h.event(egui::Event::PointerButton { pos: p, button: egui::PointerButton::Primary, pressed: false, modifiers });
        h.run_steps(2);
        h.event(egui::Event::ModifiersChanged(Modifiers::NONE));
        h.run_steps(1);
    }

    /// Drag from `from` to `to` in steps, as when a user drags a group header.
    fn drag(h: &mut Harness<'_, PhotocraftApp>, from: egui::Pos2, to: egui::Pos2) {
        h.event(egui::Event::PointerMoved(from));
        h.run_steps(1);
        h.event(egui::Event::PointerButton { pos: from, button: egui::PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
        h.run_steps(1);
        for i in 1..=6 {
            h.event(egui::Event::PointerMoved(from + (to - from) * (i as f32 / 6.0)));
            h.run_steps(1);
        }
        h.event(egui::Event::PointerButton { pos: to, button: egui::PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
        h.run_steps(3);
    }

    /// The first swatch: below the first group header (22 px), after the 8 px grid inset.
    fn first_cell(h: &Harness<'_, PhotocraftApp>) -> egui::Pos2 {
        h.ctx.input(|i| i.viewport_rect()).min + vec2(8.0 + 12.0 + 8.0, 22.0 + 2.0 + 12.0 + 8.0)
    }

    /// Clicking a swatch while characters are selected recolours them, not just the foreground.
    #[test]
    fn clicking_a_swatch_recolours_selected_type() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 400, "height": 200})).unwrap();
        let id = app.run("type.create", json!({"text": "Hello world", "size": 40, "x": 20, "y": 100, "color": "#ffffff"})).unwrap()["layer"].as_u64().unwrap();
        app.ui.tool = Tool::Type;
        app.ui.text_edit = Some(crate::state::TextEdit {
            layer: id,
            caret: 0,
            anchor: 5,
            session: "s".into(),
            created: false,
            mask: false,
            dragging: false,
            resize: None,
            preedit: None,
        });
        // Recolour with a visible colour: put Red first.
        app.run("swatches.move", json!({"swatch": "Red", "to": "Grays", "position": 0})).unwrap();
        let mut h = Harness::builder().with_size(vec2(300.0, 300.0)).build_ui_state(|ui, app: &mut PhotocraftApp| panel(app, ui), app);
        h.run_steps(2);
        let p = first_cell(&h);
        click(&mut h, p, Modifiers::NONE);
        let red = [230.0 / 255.0, 40.0 / 255.0, 40.0 / 255.0, 1.0];
        assert_eq!(h.state().session.tools.foreground, red);
        let st = h.state().session.active().unwrap();
        let Some(photocraft_doc::LayerContent::Text(t)) = st.doc.layer(photocraft_doc::LayerId(id)).map(|l| &l.content) else { panic!("type layer") };
        let runs = t.char_runs();
        assert_eq!(runs[0].len, 5, "the selection is its own run");
        assert_eq!(runs[0].style.color.to_rgba8(), [230, 40, 40, 255]);
        assert_eq!(runs[1].style.color.to_rgba8(), [255, 255, 255, 255]);
    }

    /// #4790: the footer (size slider, New Group, New, Delete) stays inside the panel however short
    /// the dock makes it: the list scrolls instead of pushing the footer out of view.
    #[test]
    fn the_footer_fits_below_a_full_list() {
        use egui_kittest::kittest::Queryable;
        for h_px in [80.0, 100.0, 120.0, 160.0, 300.0] {
            let app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
            let mut h = Harness::builder().with_size(vec2(260.0, h_px)).build_ui_state(|ui, app: &mut PhotocraftApp| panel(app, ui), app);
            PhotocraftApp::setup_context(&h.ctx, Default::default());
            h.run_steps(3);
            let bottom = h.ctx.input(|i| i.viewport_rect()).max.y;
            let slider = h.get_by_role(egui::accesskit::Role::Slider).rect().max.y;
            let delete = h.get_by_label("Delete").rect().max.y;
            assert!(slider <= bottom + 0.5 && delete <= bottom + 0.5, "{h_px} px panel: slider ends at {slider}, Delete at {delete}, panel at {bottom}");
        }
    }

    #[test]
    fn alt_click_sets_the_background_and_selects() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.tools.background = [0.0, 0.0, 1.0, 1.0];
        let mut h = Harness::builder().with_size(vec2(300.0, 300.0)).build_ui_state(|ui, app: &mut PhotocraftApp| panel(app, ui), app);
        h.run_steps(2);
        let p = first_cell(&h);
        click(&mut h, p, Modifiers::ALT);
        assert_eq!(h.state().session.tools.background, [0.0, 0.0, 0.0, 1.0], "Black");
        assert_eq!(h.state().session.tools.foreground, [0.0, 0.0, 0.0, 1.0], "foreground untouched (default black)");
        assert_eq!(h.state().ui.presets_ui.selected.get(PANEL).map(String::as_str), Some("Black"));
        // The footer's trash deletes the selection.
        let n = h.state().session.presets.swatches[0].items.len();
        handle(h.state_mut(), Ev::Delete("Black".into()), false);
        assert_eq!(h.state().session.presets.swatches[0].items.len(), n - 1);
    }

    #[test]
    fn new_swatch_from_the_foreground_and_picker() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.tools.foreground = [0.5, 0.25, 1.0, 1.0];
        app.ui.presets_ui.selected.insert(PANEL.into(), "Dark Red".into());
        new_swatch(&mut app);
        let dark = app.session.presets.swatches.iter().find(|g| g.name == "Dark").unwrap();
        assert_eq!(dark.items.last().unwrap().name, "Color Swatch 1", "into the selected swatch's group");
        assert!(app.ui.presets_ui.renaming.is_some(), "renamed in place");
        let mut f = serde_json::Map::new();
        f.insert("color".into(), json!("#00ffff"));
        f.insert("__cmyk".into(), json!([1.0, 0.0, 0.0, 0.0]));
        add_from_picker(&mut app, &f);
        let s = app.session.presets.swatches[0].items.last().unwrap();
        assert_eq!(s.color, photocraft_engine::presets::swatches::SwatchColor::Cmyk([1.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn import_and_export_through_the_services() {
        use std::sync::{Arc, Mutex};
        type Written = Arc<Mutex<Vec<(String, Vec<u8>)>>>;
        let written: Written = Arc::default();
        let w = written.clone();
        let aco =
            photocraft_psd::aco::write(&[photocraft_psd::aco::AcoSwatch { name: "Teal".into(), color: photocraft_psd::aco::AcoColor::Rgb([0, 32768, 32768]) }])
                .unwrap();
        use crate::file_dialog::FileDialogAnswer;
        let answers = vec![
            Some(FileDialogAnswer::Contents("Ocean.aco".into(), aco.clone())),
            Some(FileDialogAnswer::Contents("Ocean.aco".into(), aco)),
            Some(FileDialogAnswer::SaveTo("out/Swatches.ase".into())),
        ];
        let services = crate::Services {
            file_dialog: Some(crate::file_dialog::fake(answers).0),
            write: Some(Box::new(move |p: &str, b: &[u8]| {
                w.lock().unwrap().push((p.to_string(), b.to_vec()));
                Ok(())
            })),
            ..Default::default()
        };
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), services);
        // Each dialog answers at the end of the frame.
        let frame = |app: &mut PhotocraftApp| app.poll_file_dialog(&egui::Context::default(), None);
        import_dialog(&mut app, false).unwrap();
        frame(&mut app);
        assert_eq!(app.session.presets.swatches.last().unwrap().name, "Ocean");
        import_dialog(&mut app, true).unwrap();
        frame(&mut app);
        assert_eq!(app.session.presets.swatches.len(), 1, "replace");
        export_dialog(&mut app, "ase").unwrap();
        frame(&mut app);
        let out = written.lock().unwrap();
        assert_eq!(out[0].0, "out/Swatches.ase");
        assert!(out[0].1.starts_with(b"ASEF"));
    }

    /// #3658: dragging a group header reorders the folders, and the new order sticks in the UI
    /// state (UI state is serde, so it saves and reloads with the rest of the panel state).
    #[test]
    fn dragging_a_group_header_reorders_and_persists() {
        let app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let mut h = Harness::builder().with_size(vec2(300.0, 4000.0)).build_ui_state(|ui, app: &mut PhotocraftApp| panel(app, ui), app);
        h.run_steps(2);
        let headers = crate::preset_panels::last_group_headers(&h.ctx, PANEL);
        assert!(headers.len() >= 2, "the default library has folders: {headers:?}");
        let names: Vec<String> = headers.iter().map(|(n, _)| n.clone()).collect();
        let (first, first_r) = &headers[0];
        let (last, last_r) = headers.last().unwrap();
        // Drag the last folder's header above the first; the insertion lands before it.
        drag(&mut h, last_r.center(), egui::pos2(first_r.center().x, first_r.top() + 1.0));
        let order = h.state().ui.presets_ui.display_order(PANEL, &names);
        assert_eq!(order.len(), names.len());
        assert_eq!(names.get(order[0]).map(String::as_str), Some(last.as_str()), "the dragged folder moved to the front: {order:?}");
        assert_eq!(names.get(order[1]).map(String::as_str), Some(first.as_str()));
        // The reorder is remembered by name (not by the positions it happened to have).
        assert_eq!(h.state().ui.presets_ui.group_order.get(PANEL).and_then(|o| o.first()), Some(last));
    }
}
