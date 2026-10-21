//! Window › Gradients, Patterns (tabs of the Color card, as in Photoshop's Essentials workspace),
//! Styles, Shapes, Tool Presets and Clone Source (floating panels).
//!
//! The preset panels share one browser: folders with disclosure triangles, a thumbnail grid (or a
//! list when the size slider is at its minimum), a rename field, a context menu (Rename, Delete,
//! Move to group) and the Photoshop footer (size slider · New Group · New · Delete). Clicking a
//! preset selects it, double-clicking or dragging it onto the canvas applies it. Every action is
//! an engine command (`gradient.presets.*`, `pattern.presets.*`, `style.presets.*`,
//! `shape.presets.*`, `tool.presets.*`, `cloneSource.*`); this module only draws and dispatches.

use std::collections::{BTreeMap, BTreeSet};

use egui::{Align2, Color32, ColorImage, CornerRadius, FontId, Pos2, Rect, Sense, Stroke, TextureHandle, TextureOptions, pos2, vec2};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::state::Tool;
use crate::theme::Tokens;

/// Panel state (serde, so the control channel can read and drive it).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PresetUi {
    /// Floating panels.
    pub styles: bool,
    pub shapes: bool,
    pub tool_presets: bool,
    pub clone_source: bool,
    /// Thumbnail size per panel (px; the minimum shows a list).
    pub sizes: BTreeMap<String, f32>,
    /// Folders whose open state differs from the default (only the first folder starts open).
    pub toggled: BTreeSet<String>,
    /// Selected preset per panel (by name / pattern id).
    pub selected: BTreeMap<String, String>,
    /// Tool Presets: Current Tool Only.
    pub current_tool_only: bool,
    /// Custom Shape tool: the shape it draws.
    pub custom_shape: String,
    /// Folder/group order per panel, by name (#3658), like the dock's `tabOrder` (#2272). A panel
    /// with no entry keeps the library's order; unknown names are ignored and groups missing from
    /// a list follow it in library order. Saved with the UI state.
    #[serde(rename = "groupOrder", alias = "group_order", deserialize_with = "lenient_group_order", skip_serializing_if = "BTreeMap::is_empty")]
    pub group_order: BTreeMap<String, Vec<String>>,
    /// In-place rename: (panel, preset or "group:<name>", text).
    #[serde(skip)]
    pub renaming: Option<(String, String, String)>,
}

/// Read `groupOrder` keeping what makes sense: non-object values, non-list entries and non-string
/// names are dropped instead of failing the whole saved UI state.
const MAX_GROUP_ORDER: usize = 10_000;
fn lenient_group_order<'de, D: serde::Deserializer<'de>>(d: D) -> Result<BTreeMap<String, Vec<String>>, D::Error> {
    let v = Value::deserialize(d)?;
    let mut out = BTreeMap::new();
    for (panel, names) in v.as_object().into_iter().flatten() {
        let Some(names) = names.as_array() else { continue };
        out.insert(panel.clone(), names.iter().filter_map(Value::as_str).take(MAX_GROUP_ORDER).map(str::to_owned).collect());
    }
    Ok(out)
}

impl PresetUi {
    pub fn shape(&self) -> &str {
        if self.custom_shape.is_empty() { tl!("Heart") } else { &self.custom_shape }
    }

    /// Display order of the folders for `panel`, as library indices: the stored order first
    /// (names that no longer exist are dropped), then groups missing from it in library order.
    /// Mirrors the dock's per-group tab order (#2272, #3658).
    pub(crate) fn display_order(&self, panel: &str, names: &[String]) -> Vec<usize> {
        let mut out: Vec<usize> = Vec::with_capacity(names.len());
        for name in self.group_order.get(panel).into_iter().flatten() {
            if let Some(i) = names.iter().position(|n| n == name)
                && !out.contains(&i)
            {
                out.push(i);
            }
        }
        for i in 0..names.len() {
            if !out.contains(&i) {
                out.push(i);
            }
        }
        out
    }

    /// Move folder `from` so it is drawn before `before` (`None` = last) for `panel`. The
    /// library's own order is stored as no entry, as the dock does for its tabs.
    pub(crate) fn move_group(&mut self, panel: &str, names: &[String], from: &str, before: Option<&str>) {
        if before == Some(from) {
            return;
        }
        let mut order: Vec<String> = self.display_order(panel, names).into_iter().filter_map(|i| names.get(i).cloned()).collect();
        let Some(at) = order.iter().position(|n| n == from) else { return };
        let moved = order.remove(at);
        let to = before.and_then(|b| order.iter().position(|n| n == b)).unwrap_or(order.len());
        order.insert(to, moved);
        if order.as_slice() == names {
            self.group_order.remove(panel);
        } else {
            self.group_order.insert(panel.to_string(), order);
        }
    }
}

const PANELS: [&str; 6] =
    ["window.panel.gradients", "window.panel.patterns", "window.panel.styles", "window.panel.shapes", "window.panel.toolPresets", "window.panel.cloneSource"];
const MIN_THUMB: f32 = 20.0;

/// Window-menu ids handled here.
pub fn handles(id: &str) -> bool {
    PANELS.contains(&id)
}

/// Color card tab of a Window › Gradients / Patterns id.
fn color_tab(id: &str) -> Option<usize> {
    match id {
        "window.panel.gradients" => Some(2),
        "window.panel.patterns" => Some(3),
        _ => None,
    }
}

fn float_flag<'a>(p: &'a mut PresetUi, id: &str) -> Option<&'a mut bool> {
    Some(match id {
        "window.panel.styles" => &mut p.styles,
        "window.panel.shapes" => &mut p.shapes,
        "window.panel.toolPresets" => &mut p.tool_presets,
        "window.panel.cloneSource" => &mut p.clone_source,
        _ => return None,
    })
}

/// Window › <panel>: toggles the panel (Gradients and Patterns: the Color card tab).
pub fn menu(app: &mut PhotocraftApp, id: &str, params: &Value) -> Option<Result<Value, String>> {
    if !handles(id) {
        return None;
    }
    let want = params.get("show").and_then(Value::as_bool);
    if let Some(tab) = color_tab(id) {
        // A collapsed Color group counts as not showing: the item expands it (#129).
        let showing = app.ui.panels.color && app.ui.dock_tabs.color == tab && !app.ui.dock.is_collapsed(crate::dock::Group::Color);
        let show = want.unwrap_or(!showing);
        app.ui.panels.color = show || (app.ui.panels.color && app.ui.dock_tabs.color != tab);
        if show {
            app.ui.dock_tabs.color = tab;
            crate::dock::reveal(app, crate::dock::Group::Color);
        }
        return Some(Ok(json!({"visible": show})));
    }
    let flag = float_flag(&mut app.ui.presets_ui, id)?;
    *flag = want.unwrap_or(!*flag);
    Some(Ok(json!({"visible": *flag})))
}

pub fn checked(app: &PhotocraftApp, id: &str) -> Option<bool> {
    if let Some(tab) = color_tab(id) {
        return Some(app.ui.panels.color && app.ui.dock_tabs.color == tab);
    }
    let mut p = app.ui.presets_ui.clone();
    float_flag(&mut p, id).map(|f| *f)
}

pub(crate) fn run(app: &mut PhotocraftApp, id: &str, p: Value) -> Option<Value> {
    match app.run(id, p) {
        Ok(v) => Some(v),
        Err(e) => {
            app.ui.status = e;
            app.ui.status_error = true;
            None
        }
    }
}

// ------------------------------------------------------------------ textures

fn cached_texture(ctx: &egui::Context, key: impl std::hash::Hash + std::fmt::Debug, make: impl FnOnce() -> ColorImage) -> TextureHandle {
    let id = egui::Id::new(("preset-tex", key));
    if let Some(t) = ctx.data(|d| d.get_temp::<TextureHandle>(id)) {
        return t;
    }
    let tex = ctx.load_texture(format!("preset-{id:?}"), make(), TextureOptions::LINEAR);
    ctx.data_mut(|d| d.insert_temp(id, tex.clone()));
    tex
}

fn c32(c: [f32; 4]) -> Color32 {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    Color32::from_rgba_unmultiplied(q(c[0]), q(c[1]), q(c[2]), q(c[3]))
}

/// A gradient swatch: stops as a horizontal mesh over a checkerboard where it's transparent.
pub fn paint_gradient(ui: &egui::Ui, rect: Rect, stops: &[(f32, [f32; 4])]) {
    let p = ui.painter_at(rect);
    if stops.iter().any(|s| s.1[3] < 0.999) {
        crate::widgets::checker(&p, rect, (rect.height() / 4.0).clamp(3.0, 8.0));
    }
    let mut mesh = egui::Mesh::default();
    let x = |t: f32| rect.left() + rect.width() * t.clamp(0.0, 1.0);
    let mut pts: Vec<(f32, [f32; 4])> = stops.to_vec();
    if pts.first().is_some_and(|s| s.0 > 0.0) {
        pts.insert(0, (0.0, pts[0].1));
    }
    if let Some(l) = pts.last().copied().filter(|s| s.0 < 1.0) {
        pts.push((1.0, l.1));
    }
    for w in pts.windows(2) {
        let (a, b) = (w[0], w[1]);
        let i = mesh.vertices.len() as u32;
        let (ca, cb) = (c32(a.1), c32(b.1));
        mesh.colored_vertex(pos2(x(a.0), rect.top()), ca);
        mesh.colored_vertex(pos2(x(b.0), rect.top()), cb);
        mesh.colored_vertex(pos2(x(b.0), rect.bottom()), cb);
        mesh.colored_vertex(pos2(x(a.0), rect.bottom()), ca);
        mesh.add_triangle(i, i + 1, i + 2);
        mesh.add_triangle(i, i + 2, i + 3);
    }
    p.add(mesh);
}

fn pattern_texture(ctx: &egui::Context, pat: &photocraft_doc::Pattern) -> TextureHandle {
    cached_texture(ctx, ("pattern", pat.id.clone()), || {
        let (w, h) = (pat.width.max(1), pat.height.max(1));
        let fmt = pat.surface.format();
        let px = pat.surface.read_region(photocraft_geom::Rect::new(0, 0, w as i32, h as i32));
        let n = fmt.channels();
        // Tiles smaller than the swatch repeat; larger ones are shown shrunk.
        const S: u32 = 64;
        let step = (w.max(h) as f32 / S as f32).max(1.0);
        let mut img = ColorImage::new([S as usize, S as usize], vec![Color32::TRANSPARENT; (S * S) as usize]);
        for y in 0..S {
            for x in 0..S {
                let sx = ((x as f32 * step) as u32) % w;
                let sy = ((y as f32 * step) as u32) % h;
                let i = (sy * w + sx) as usize * n;
                img.pixels[(y * S + x) as usize] = c32(photocraft_raster::to_rgba(&fmt, &px[i..i + n]));
            }
        }
        img
    })
}

/// Photoshop's pattern picker: a swatch of the library pattern `selected` (an id) that opens a grid
/// of every library pattern's swatch, each named in its tooltip. Returns the id of a newly picked
/// pattern.
pub(crate) fn pattern_picker(app: &PhotocraftApp, ui: &mut egui::Ui, selected: &str) -> Option<String> {
    const SWATCH: f32 = 40.0;
    let t = Tokens::get(ui.ctx());
    let ctx = ui.ctx().clone();
    let pats = &app.session.patterns.items;
    let current = pats.iter().find(|p| p.id == selected);
    let uv = Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0));
    let (r, resp) = ui.allocate_exact_size(vec2(SWATCH, SWATCH), Sense::click());
    if let Some(p) = current {
        ui.painter().image(pattern_texture(&ctx, p).id(), r, uv, Color32::WHITE);
    }
    ui.painter().rect_stroke(r, 0.0, Stroke::new(1.0, t.field_border), egui::StrokeKind::Outside);
    let name = current.map_or(String::new(), |p| p.display_name().to_string());
    let resp = resp.on_hover_text(&name);
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, tl!("Patterns")));
    let mut picked = None;
    egui::Popup::menu(&resp).show(|ui| {
        ui.set_max_width(6.0 * (SWATCH + 4.0));
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = vec2(4.0, 4.0);
            for p in pats {
                let (r, cell) = ui.allocate_exact_size(vec2(SWATCH, SWATCH), Sense::click());
                ui.painter().image(pattern_texture(&ctx, p).id(), r, uv, Color32::WHITE);
                let (w, c) = if p.id == selected { (2.0, t.accent) } else { (1.0, t.field_border) };
                ui.painter().rect_stroke(r, 0.0, Stroke::new(w, c), egui::StrokeKind::Outside);
                let cell = cell.on_hover_text(p.display_name());
                cell.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, p.display_name()));
                if cell.clicked() {
                    picked = Some(p.id.clone());
                    ui.close();
                }
            }
        });
    });
    picked.filter(|id| id != selected)
}

pub(crate) fn style_texture(app: &PhotocraftApp, ctx: &egui::Context, st: &photocraft_engine::presets::styles::StylePreset) -> TextureHandle {
    let key = ("style", st.name.clone(), st.effects.len(), format!("{:?}{:?}", st.blend, st.fill_opacity));
    cached_texture(ctx, key, || {
        const S: u32 = 64;
        let px = photocraft_engine::presets::styles::thumbnail(&app.session, st, S);
        ColorImage::from_rgba_unmultiplied([S as usize, S as usize], &px)
    })
}

fn shape_texture(ctx: &egui::Context, sh: &photocraft_engine::presets::shapes::ShapePreset) -> TextureHandle {
    let knots: usize = sh.path.subpaths.iter().map(|s| s.knots.len()).sum();
    cached_texture(ctx, ("shape", sh.name.clone(), knots), || {
        const S: u32 = 64;
        let cov = photocraft_engine::presets::shapes::thumbnail(&sh.path, S);
        ColorImage::new([S as usize, S as usize], cov.iter().map(|a| Color32::from_white_alpha((a.clamp(0.0, 1.0) * 255.0).round() as u8)).collect())
    })
}

// ------------------------------------------------------------------ the browser

pub(crate) struct ItemView {
    pub(crate) key: String,
    pub(crate) name: String,
}

pub(crate) struct GroupView {
    pub(crate) name: String,
    pub(crate) items: Vec<ItemView>,
}

pub(crate) enum Ev {
    Select(String),
    Activate(String),
    Drop(String, Pos2),
    Rename(String, String),
    Delete(String),
    Move(String, String),
    RenameGroup(String, String),
    DeleteGroup(String),
    NewGroup,
    New,
}

fn group_open(st: &PresetUi, panel: &str, gi: usize, name: &str) -> bool {
    // Swatch groups are small and all start open (Photoshop); other panels open the first only.
    (gi == 0 || panel == crate::swatches_ui::PANEL) != st.toggled.contains(&format!("{panel}/{name}"))
}

/// The footer's height (gap, size slider and buttons) before it has been measured.
const FOOTER_GUESS: f32 = 32.0;

/// Draws the folders and footer; `thumb` paints item `(group, item)` into a rect.
/// Where a browser sits: the canvas (drop target), its height cap and the New button's tooltip.
pub(crate) struct Place<'a> {
    pub(crate) canvas: Rect,
    pub(crate) max_h: f32,
    pub(crate) new_tip: &'a str,
}

pub(crate) fn browser(
    ui: &mut egui::Ui,
    st: &mut PresetUi,
    panel: &str,
    groups: &[GroupView],
    place: Place,
    thumb: &mut dyn FnMut(&egui::Ui, Rect, usize, usize),
) -> Vec<Ev> {
    let t = Tokens::get(ui.ctx());
    let Place { canvas, max_h, new_tip } = place;
    let mut ev = Vec::new();
    let size = *st.sizes.get(panel).unwrap_or(&40.0);
    let list = size <= MIN_THUMB + 0.5;
    let selected = st.selected.get(panel).cloned();
    // Rename field (Photoshop opens a small dialog; inline keeps the panel modeless).
    if let Some((p, key, mut text)) = st.renaming.clone().filter(|r| r.0 == panel) {
        let mut done = None;
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(tl!("Name:")).color(t.text_dim).size(11.5));
            let r = ui.add(egui::TextEdit::singleline(&mut text).desired_width(ui.available_width() - 4.0));
            if !r.has_focus() && !r.lost_focus() {
                r.request_focus();
            }
            if r.lost_focus() {
                done = Some(ui.input(|i| i.key_pressed(egui::Key::Enter)));
            }
        });
        st.renaming = Some((p, key.clone(), text.clone()));
        if let Some(ok) = done {
            st.renaming = None;
            if ok && !text.trim().is_empty() {
                match key.strip_prefix("group:") {
                    Some(g) => ev.push(Ev::RenameGroup(g.to_string(), text.trim().to_string())),
                    None => ev.push(Ev::Rename(key, text.trim().to_string())),
                }
            }
        }
    }
    let mut headers: Vec<(usize, Rect, String)> = Vec::new();
    let mut dropped: Option<(String, Pos2)> = None;
    // Folder drag (#3658): the order the folders are drawn in, and what the pointer is doing this
    // frame (the insertion line and the drop are resolved once every header is known).
    let mut drag: Option<(usize, Pos2)> = None;
    let mut released: Option<(usize, Pos2)> = None;
    let group_names: Vec<String> = groups.iter().map(|g| g.name.clone()).collect();
    let display = st.display_order(panel, &group_names);
    // The footer stays in view: the list gets the height left after it (as measured last frame)
    // and scrolls, rather than pushing the footer out of a short dock group (#4790).
    let footer_id = egui::Id::new(("preset-footer", panel));
    let footer_h = ui.data(|d| d.get_temp::<f32>(footer_id)).unwrap_or(FOOTER_GUESS);
    let max_h = max_h.min((ui.available_height() - footer_h).max(0.0));
    egui::ScrollArea::vertical().id_salt(("preset-scroll", panel)).max_height(max_h).min_scrolled_height(max_h.min(64.0)).auto_shrink([false, true]).show(
        ui,
        |ui| {
            ui.spacing_mut().item_spacing = vec2(0.0, 2.0);
            for &gi in &display {
                let Some(g) = groups.get(gi) else { continue };
                let open = group_open(st, panel, gi, &g.name);
                let (hr, hresp) = ui.allocate_exact_size(vec2(ui.available_width(), 22.0), Sense::click_and_drag());
                if hresp.hovered() || hresp.dragged() {
                    ui.painter().rect_filled(hr, 2.0, t.hover.gamma_multiply(0.5));
                }
                let tri = Rect::from_center_size(pos2(hr.left() + 9.0, hr.center().y), vec2(12.0, 12.0));
                crate::icons::paint(ui, tri, if open { "chevron-down" } else { "chevron-right" }, 10.0, t.text_dim);
                let fold = Rect::from_center_size(pos2(hr.left() + 25.0, hr.center().y), vec2(14.0, 14.0));
                crate::icons::paint(ui, fold, if open { "folder-open" } else { "folder" }, 12.0, t.icon);
                ui.painter().text(pos2(hr.left() + 37.0, hr.center().y), Align2::LEFT_CENTER, &g.name, FontId::proportional(11.5), t.text);
                if hresp.clicked() {
                    let k = format!("{panel}/{}", g.name);
                    if !st.toggled.remove(&k) {
                        st.toggled.insert(k);
                    }
                }
                hresp.context_menu(|ui| {
                    if ui.button(tl!("Rename Group…")).clicked() {
                        st.renaming = Some((panel.to_string(), format!("group:{}", g.name), g.name.clone()));
                        ui.close();
                    }
                    if ui.button(tl!("Delete Group")).clicked() {
                        ev.push(Ev::DeleteGroup(g.name.clone()));
                        ui.close();
                    }
                });
                if hresp.dragged()
                    && let Some(p) = ui.ctx().pointer_interact_pos()
                {
                    drag = Some((gi, p));
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
                }
                if hresp.drag_stopped()
                    && let Some(p) = ui.ctx().pointer_interact_pos()
                {
                    released = Some((gi, p));
                }
                headers.push((gi, hr, g.name.clone()));
                if !open {
                    continue;
                }
                let w = ui.available_width();
                let (cell, cols) = if list {
                    (vec2(w, 24.0), 1)
                } else {
                    let cols = ((w - 8.0 + 4.0) / (size + 4.0)).floor().max(1.0) as usize;
                    (vec2(size, size), cols)
                };
                let rows = g.items.len().div_ceil(cols);
                let pad = if list { 0.0 } else { 8.0 };
                let gap = if list { 0.0 } else { 4.0 };
                let (area, _) = ui.allocate_exact_size(vec2(w, rows as f32 * (cell.y + gap) + 2.0), Sense::hover());
                for (ii, it) in g.items.iter().enumerate() {
                    let (cx, cy) = ((ii % cols) as f32, (ii / cols) as f32);
                    let r = Rect::from_min_size(area.min + vec2(pad + cx * (cell.x + gap), cy * (cell.y + gap)), cell);
                    let resp = ui.interact(r, ui.id().with((panel, gi, ii)), Sense::click_and_drag());
                    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, &it.name));
                    let is_sel = selected.as_deref() == Some(&it.key);
                    if list {
                        if is_sel {
                            ui.painter().rect_filled(r, 0.0, t.row_selected);
                        } else if resp.hovered() {
                            ui.painter().rect_filled(r, 0.0, t.hover.gamma_multiply(0.5));
                        }
                        let tr = Rect::from_min_size(r.min + vec2(24.0, 2.0), vec2(20.0, 20.0));
                        thumb(ui, tr, gi, ii);
                        ui.painter().text(pos2(tr.right() + 8.0, r.center().y), Align2::LEFT_CENTER, &it.name, FontId::proportional(11.5), t.text);
                    } else {
                        thumb(ui, r, gi, ii);
                        let (stroke, kind) = if is_sel {
                            (Stroke::new(2.0, t.accent), egui::StrokeKind::Outside)
                        } else if resp.hovered() {
                            (Stroke::new(1.0, t.text_dim), egui::StrokeKind::Outside)
                        } else {
                            (Stroke::new(1.0, t.card_border), egui::StrokeKind::Inside)
                        };
                        ui.painter().rect_stroke(r, CornerRadius::same(2), stroke, kind);
                    }
                    let resp = if list { resp } else { resp.on_hover_text(&it.name) };
                    if resp.double_clicked() {
                        ev.push(Ev::Activate(it.key.clone()));
                    } else if resp.clicked() {
                        ev.push(Ev::Select(it.key.clone()));
                    }
                    if resp.dragged()
                        && let Some(pp) = ui.ctx().pointer_interact_pos()
                    {
                        // A ghost of the swatch follows the pointer.
                        let lp = ui.ctx().layer_painter(egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("preset-drag")));
                        let gr = Rect::from_center_size(pp, vec2(28.0, 28.0));
                        lp.rect_filled(gr, 3.0, t.card.gamma_multiply(0.8));
                        lp.rect_stroke(gr, 3.0, Stroke::new(1.0, t.accent), egui::StrokeKind::Outside);
                        lp.text(gr.center(), Align2::CENTER_CENTER, "+", FontId::proportional(16.0), t.accent);
                    }
                    if resp.drag_stopped()
                        && let Some(pp) = ui.ctx().pointer_interact_pos()
                        && !r.contains(pp)
                    {
                        dropped = Some((it.key.clone(), pp));
                    }
                    resp.context_menu(|ui| {
                        if ui.button(tl!("Rename…")).clicked() {
                            st.renaming = Some((panel.to_string(), it.key.clone(), it.name.clone()));
                            ui.close();
                        }
                        if ui.button(tl!("Delete")).clicked() {
                            ev.push(Ev::Delete(it.key.clone()));
                            ui.close();
                        }
                        ui.menu_button(tl!("Move to"), |ui| {
                            for gname in group_names.iter().filter(|n| **n != g.name) {
                                if ui.button(gname).clicked() {
                                    ev.push(Ev::Move(it.key.clone(), gname.clone()));
                                    ui.close();
                                }
                            }
                        });
                    });
                }
            }
            // Insertion line while a folder is dragged by its header.
            if let Some((from, p)) = drag {
                let before = group_drop_before(&headers, from, p.y);
                let anchor = match before {
                    Some(i) => headers.iter().find(|(x, _, _)| *x == i).map(|(_, r, _)| *r),
                    None => headers.last().map(|(_, r, _)| *r),
                };
                if let Some(r) = anchor {
                    let y = if before.is_some() { r.top() - 1.0 } else { r.bottom() + 1.0 };
                    ui.painter().line_segment([pos2(r.left(), y), pos2(r.right(), y)], Stroke::new(2.0, t.accent));
                }
            }
        },
    );
    // Apply the drop now every header rect is known; the next frame draws the new order.
    if let Some((from, p)) = released {
        let before = group_drop_before(&headers, from, p.y);
        let from_name = groups.get(from).map(|g| g.name.clone());
        let before_name = before.and_then(|i| groups.get(i).map(|g| g.name.clone()));
        if let Some(from_name) = from_name {
            st.move_group(panel, &group_names, &from_name, before_name.as_deref());
        }
    }
    let recorded: Vec<(String, Rect)> = headers.iter().map(|(_, r, n)| (n.clone(), *r)).collect();
    ui.ctx().data_mut(|d| d.insert_temp(egui::Id::new(("preset-group-headers", panel)), recorded));
    if let Some((key, pos)) = dropped {
        if let Some((_, _, g)) = headers.iter().find(|(_, r, _)| r.contains(pos)) {
            ev.push(Ev::Move(key, g.clone()));
        } else if canvas.contains(pos) {
            ev.push(Ev::Drop(key, pos));
        }
    }
    // Footer: thumbnail size | New Group, New, Delete.
    let footer_top = ui.min_rect().bottom();
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        let mut s = size;
        ui.add_sized(vec2(90.0, 18.0), egui::Slider::new(&mut s, MIN_THUMB..=96.0).show_value(false)).on_hover_text(tl!("Thumbnail size (smallest: list)"));
        if (s - size).abs() > 0.01 {
            st.sizes.insert(panel.to_string(), s);
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if crate::icons::button(ui, "trash", 24.0, false, tl!("Delete")).clicked()
                && let Some(k) = selected.clone()
            {
                ev.push(Ev::Delete(k));
            }
            if crate::icons::button(ui, "plus", 24.0, false, new_tip).clicked() {
                ev.push(Ev::New);
            }
            if crate::icons::button(ui, "folder-plus", 24.0, false, tl!("Create new group")).clicked() {
                ev.push(Ev::NewGroup);
            }
        });
    });
    let measured = ui.min_rect().bottom() - footer_top;
    if measured.is_finite() && (measured - footer_h).abs() > 0.5 {
        ui.data_mut(|d| d.insert_temp(footer_id, measured));
        ui.ctx().request_repaint();
    }
    ev
}

/// The folder (by library index) a header dragged to `y` lands before (`None` = last). `headers`
/// is in display order; mirrors `dock::drop_before`.
fn group_drop_before(headers: &[(usize, Rect, String)], from: usize, y: f32) -> Option<usize> {
    let hit = headers.iter().find(|(_, r, _)| y < r.center().y).map(|(i, _, _)| *i);
    match hit {
        Some(i) if i == from => Some(from),
        Some(i) => {
            let after_self = headers.iter().position(|(x, _, _)| *x == from).zip(headers.iter().position(|(x, _, _)| *x == i)).is_some_and(|(a, b)| b == a + 1);
            if after_self { Some(from) } else { Some(i) }
        }
        None => None,
    }
}

/// Folder header rects drawn last frame (name → rect), in display order; for tests and automation.
pub fn last_group_headers(ctx: &egui::Context, panel: &str) -> Vec<(String, Rect)> {
    ctx.data(|d| d.get_temp(egui::Id::new(("preset-group-headers", panel)))).unwrap_or_default()
}

/// Shared handling of the generic events through `<prefix>.edit` / `.new`.
fn generic(app: &mut PhotocraftApp, panel: &str, prefix: &str, e: &Ev, new_params: Value) -> bool {
    let edit = format!("{prefix}.edit");
    match e {
        Ev::Rename(k, n) => {
            run(app, &edit, json!({"action": "rename", "preset": k, "name": n}));
            if app.ui.presets_ui.selected.get(panel) == Some(k) {
                app.ui.presets_ui.selected.insert(panel.into(), n.clone());
            }
        }
        Ev::Delete(k) => {
            if run(app, &edit, json!({"action": "delete", "preset": k})).is_some() {
                app.ui.presets_ui.selected.remove(panel);
            }
        }
        Ev::Move(k, g) => drop(run(app, &edit, json!({"action": "move", "preset": k, "to": g}))),
        Ev::RenameGroup(g, n) => drop(run(app, &edit, json!({"action": "renameGroup", "group": g, "name": n}))),
        Ev::DeleteGroup(g) => drop(run(app, &edit, json!({"action": "deleteGroup", "group": g}))),
        Ev::NewGroup => {
            if let Some(r) = run(app, &edit, json!({"action": "newGroup", "name": "Group"})) {
                let g = r["group"].as_str().unwrap_or_default().to_string();
                app.ui.presets_ui.renaming = Some((panel.into(), format!("group:{g}"), g));
            }
        }
        Ev::New => {
            if let Some(r) = run(app, &format!("{prefix}.new"), new_params) {
                let name = r.get("name").or(r.get("pattern")).and_then(Value::as_str).unwrap_or_default().to_string();
                app.ui.presets_ui.selected.insert(panel.into(), name.clone());
                if prefix != "pattern.presets" {
                    app.ui.presets_ui.renaming = Some((panel.into(), name.clone(), name));
                }
            }
        }
        _ => return false,
    }
    true
}

fn empty(ui: &mut egui::Ui, s: &str) {
    let t = Tokens::get(ui.ctx());
    ui.label(egui::RichText::new(s).color(t.text_faint).size(11.5));
}

/// Where drops onto the canvas are accepted: the canvas proper, rulers excluded (a drop on the
/// ruler strip is not a canvas drop).
fn canvas_drop_rect(app: &PhotocraftApp) -> egui::Rect {
    crate::rulers::content_rect(app, app.last_canvas_rect)
}

// ------------------------------------------------------------------ Gradients

pub fn gradients_panel(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let (fg, bg) = (app.session.tools.foreground, app.session.tools.background);
    let groups_src = app.session.presets.gradients.clone();
    // Current gradient (what the Gradient tool paints), like the panel's top "recent" row.
    let cur = app.session.presets.gradient.clone();
    let (r, _) = ui.allocate_exact_size(vec2(ui.available_width(), 18.0), Sense::hover());
    paint_gradient(ui, r, &cur.resolve(fg, bg));
    ui.painter().rect_stroke(r, 2.0, Stroke::new(1.0, Tokens::get(ui.ctx()).card_border), egui::StrokeKind::Inside);
    ui.add_space(4.0);
    let groups: Vec<GroupView> = groups_src
        .iter()
        .map(|g| GroupView { name: g.name.clone(), items: g.items.iter().map(|i| ItemView { key: i.name.clone(), name: i.name.clone() }).collect() })
        .collect();
    let canvas = canvas_drop_rect(app);
    let mut st = std::mem::take(&mut app.ui.presets_ui);
    let max_h = (ui.available_height() - 40.0).clamp(90.0, 260.0);
    let events = browser(ui, &mut st, "gradients", &groups, Place { canvas, max_h, new_tip: tl!("Create new gradient") }, &mut |ui, r, gi, ii| {
        let stops = groups_src[gi].items[ii].resolve(fg, bg);
        paint_gradient(ui, r, &stops);
    });
    app.ui.presets_ui = st;
    for e in events {
        match &e {
            Ev::Select(k) => {
                app.ui.presets_ui.selected.insert("gradients".into(), k.clone());
                run(app, "gradient.presets.select", json!({"preset": k}));
            }
            Ev::Activate(k) | Ev::Drop(k, _) => {
                app.ui.presets_ui.selected.insert("gradients".into(), k.clone());
                run(app, "gradient.presets.apply", json!({"preset": k}));
            }
            _ => {
                generic(app, "gradients", "gradient.presets", &e, json!({"name": "Custom"}));
            }
        }
    }
}

// ------------------------------------------------------------------ Patterns

pub fn patterns_panel(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    photocraft_engine::presets::patterns::sync(&mut app.session);
    let lib = app.session.patterns.items.clone();
    let groups: Vec<GroupView> = app
        .session
        .presets
        .pattern_groups
        .iter()
        .map(|g| GroupView {
            name: g.name.clone(),
            items: g
                .items
                .iter()
                .filter_map(|id| lib.iter().find(|p| &p.id == id))
                .map(|p| ItemView { key: p.id.clone(), name: p.display_name().to_string() })
                .collect(),
        })
        .collect();
    let pats: Vec<Vec<photocraft_doc::Pattern>> =
        app.session.presets.pattern_groups.iter().map(|g| g.items.iter().filter_map(|id| lib.iter().find(|p| &p.id == id).cloned()).collect()).collect();
    let canvas = canvas_drop_rect(app);
    let mut st = std::mem::take(&mut app.ui.presets_ui);
    let max_h = (ui.available_height() - 40.0).clamp(90.0, 280.0);
    let ctx = ui.ctx().clone();
    let events =
        browser(ui, &mut st, "patterns", &groups, Place { canvas, max_h, new_tip: tl!("Create new pattern from the selection") }, &mut |ui, r, gi, ii| {
            let tex = pattern_texture(&ctx, &pats[gi][ii]);
            ui.painter().image(tex.id(), r, Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)), Color32::WHITE);
        });
    app.ui.presets_ui = st;
    for e in events {
        match &e {
            Ev::Select(k) => {
                app.ui.presets_ui.selected.insert("patterns".into(), k.clone());
                run(app, "pattern.presets.select", json!({"pattern": k}));
            }
            Ev::Activate(k) | Ev::Drop(k, _) => {
                app.ui.presets_ui.selected.insert("patterns".into(), k.clone());
                run(app, "pattern.presets.apply", json!({"pattern": k}));
            }
            _ => {
                generic(app, "patterns", "pattern.presets", &e, json!({}));
            }
        }
    }
}

// ------------------------------------------------------------------ floating panels

fn float_window(
    app: &mut PhotocraftApp,
    ctx: &egui::Context,
    key: &str,
    title: &str,
    width: f32,
    offset: f32,
    body: impl FnOnce(&mut PhotocraftApp, &mut egui::Ui),
) -> bool {
    let t = Tokens::get(ctx);
    let frame = egui::Frame::NONE
        .fill(t.card)
        .stroke(Stroke::new(1.0, t.card_border))
        .corner_radius(CornerRadius::same(t.radius_lg as u8))
        .shadow(egui::Shadow { offset: [0, 10], blur: 30, spread: 0, color: t.shadow })
        .inner_margin(egui::Margin::same(8));
    let canvas = canvas_drop_rect(app);
    let mut close = false;
    egui::Window::new(title)
        .id(egui::Id::new(("preset-window", key)))
        .title_bar(false)
        .resizable(false)
        .frame(frame)
        .default_pos(pos2((canvas.right() - width - 16.0 - offset).max(canvas.left() + 8.0), canvas.top() + 40.0))
        .show(ctx, |ui| {
            ui.set_width(width);
            // Photoshop-style tab header with a close box.
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(tl!(&title)).color(t.text).size(12.0).strong());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if crate::icons::button(ui, "x", 20.0, false, tl!("Close")).clicked() {
                        close = true;
                    }
                });
            });
            crate::widgets::hairline(ui);
            ui.add_space(4.0);
            body(app, ui);
        });
    close
}

/// Draws the open floating panels (Styles, Shapes, Tool Presets, Clone Source).
pub fn windows(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let p = app.ui.presets_ui.clone();
    if p.styles && float_window(app, ctx, "styles", tl!("Styles"), 250.0, 0.0, styles_panel) {
        app.ui.presets_ui.styles = false;
    }
    if p.shapes && float_window(app, ctx, "shapes", tl!("Shapes"), 250.0, 280.0, shapes_panel) {
        app.ui.presets_ui.shapes = false;
    }
    if p.tool_presets && float_window(app, ctx, "toolPresets", tl!("Tool Presets"), 250.0, 560.0, tool_presets_panel) {
        app.ui.presets_ui.tool_presets = false;
    }
    if p.clone_source && float_window(app, ctx, "cloneSource", tl!("Clone Source"), 270.0, 840.0, clone_source_panel) {
        app.ui.presets_ui.clone_source = false;
    }
}

// ------------------------------------------------------------------ Styles

pub fn styles_panel(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let src = app.session.presets.styles.clone();
    let groups: Vec<GroupView> = src
        .iter()
        .map(|g| GroupView { name: g.name.clone(), items: g.items.iter().map(|i| ItemView { key: i.name.clone(), name: i.name.clone() }).collect() })
        .collect();
    let canvas = canvas_drop_rect(app);
    let mut st = std::mem::take(&mut app.ui.presets_ui);
    let ctx = ui.ctx().clone();
    let t = Tokens::get(&ctx);
    let mut texes: Vec<Vec<TextureHandle>> = Vec::new();
    for g in &src {
        texes.push(g.items.iter().map(|s| style_texture(app, &ctx, s)).collect());
    }
    let events = browser(
        ui,
        &mut st,
        "styles",
        &groups,
        Place { canvas, max_h: 300.0, new_tip: tl!("Create new style from the selected layer") },
        &mut |ui, r, gi, ii| {
            ui.painter().rect_filled(r, 2.0, t.field);
            ui.painter().image(texes[gi][ii].id(), r, Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)), Color32::WHITE);
        },
    );
    app.ui.presets_ui = st;
    let shift = ui.input(|i| i.modifiers.shift);
    for e in events {
        match &e {
            Ev::Select(k) | Ev::Activate(k) => {
                app.ui.presets_ui.selected.insert("styles".into(), k.clone());
                // Clicking applies to the selected layers (⇧ adds to their effects).
                run(app, "style.presets.apply", json!({"preset": k, "add": shift}));
            }
            Ev::Drop(k, pos) => {
                // Dropping on the canvas styles the layer under the pointer's selection: the active layer.
                let _ = pos;
                run(app, "style.presets.apply", json!({"preset": k}));
            }
            _ => {
                generic(app, "styles", "style.presets", &e, json!({"name": "Style"}));
            }
        }
    }
}

// ------------------------------------------------------------------ Shapes

pub fn shapes_panel(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let src = photocraft_engine::presets::shapes::all_groups(&app.session);
    let groups: Vec<GroupView> = src
        .iter()
        .map(|g| GroupView { name: g.name.clone(), items: g.items.iter().map(|i| ItemView { key: i.name.clone(), name: i.name.clone() }).collect() })
        .collect();
    let canvas = canvas_drop_rect(app);
    let mut st = std::mem::take(&mut app.ui.presets_ui);
    let ctx = ui.ctx().clone();
    let t = Tokens::get(&ctx);
    let events = browser(
        ui,
        &mut st,
        "shapes",
        &groups,
        Place { canvas, max_h: 300.0, new_tip: tl!("Create new shape from the current path") },
        &mut |ui, r, gi, ii| {
            let tex = shape_texture(&ctx, &src[gi].items[ii]);
            ui.painter().image(tex.id(), r.shrink(2.0), Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)), t.text);
        },
    );
    app.ui.presets_ui = st;
    let fill = shape_fill(app);
    for e in events {
        match &e {
            Ev::Select(k) => {
                // Selecting a shape picks it for the Custom Shape tool (and switches to it).
                app.ui.presets_ui.selected.insert("shapes".into(), k.clone());
                app.ui.presets_ui.custom_shape = k.clone();
                app.ui.tool = Tool::CustomShape;
            }
            Ev::Activate(k) => {
                app.ui.presets_ui.custom_shape = k.clone();
                run(app, "shape.presets.place", json!({"preset": k, "fill": fill}));
            }
            Ev::Drop(k, pos) => {
                app.ui.presets_ui.custom_shape = k.clone();
                let mut p = json!({"preset": k, "fill": fill});
                if let (Some(c), Some(d)) = (crate::canvas::doc_point_at(app, *pos), app.session.active()) {
                    let side = (d.doc.size.width.min(d.doc.size.height) as f64 * 0.3).max(8.0);
                    p["rect"] = json!([c[0] - side / 2.0, c[1] - side / 2.0, side, side]);
                }
                run(app, "shape.presets.place", p);
            }
            _ => {
                generic(app, "shapes", "shape.presets", &e, json!({"name": "Shape"}));
            }
        }
    }
}

fn shape_fill(app: &PhotocraftApp) -> Value {
    let f = app.session.tools.foreground;
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    if app.ui.tool_options.shape_fill { json!(format!("#{:02x}{:02x}{:02x}", q(f[0]), q(f[1]), q(f[2]))) } else { Value::Null }
}

/// Options-bar shape picker for the Custom Shape tool.
pub fn shape_picker(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    ui.label(egui::RichText::new(tl!("Shape:")).color(t.text_dim).size(12.0));
    let groups = photocraft_engine::presets::shapes::all_groups(&app.session);
    let cur = app.ui.presets_ui.shape().to_string();
    let ctx = ui.ctx().clone();
    let (r, resp) = ui.allocate_exact_size(vec2(36.0, 22.0), Sense::click());
    ui.painter().rect_filled(r, 3.0, t.field);
    ui.painter().rect_stroke(r, 3.0, Stroke::new(1.0, t.field_border), egui::StrokeKind::Inside);
    if let Some(sh) = groups.iter().flat_map(|g| g.items.iter()).find(|s| s.name == cur) {
        let tex = shape_texture(&ctx, sh);
        ui.painter().image(
            tex.id(),
            Rect::from_center_size(r.center() - vec2(5.0, 0.0), vec2(18.0, 18.0)),
            Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)),
            t.text,
        );
    }
    crate::icons::paint(ui, Rect::from_center_size(pos2(r.right() - 8.0, r.center().y), vec2(10.0, 10.0)), "chevron-down", 9.0, t.text_dim);
    let resp = resp.on_hover_text(&cur);
    egui::Popup::menu(&resp).close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside).show(|ui| {
        ui.set_width(232.0);
        egui::ScrollArea::vertical().max_height(320.0).show(ui, |ui| {
            for g in &groups {
                ui.label(egui::RichText::new(&g.name).color(t.text_dim).size(11.0));
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = vec2(3.0, 3.0);
                    for sh in &g.items {
                        let (cr, cresp) = ui.allocate_exact_size(vec2(32.0, 32.0), Sense::click());
                        let on = sh.name == cur;
                        ui.painter().rect_filled(
                            cr,
                            2.0,
                            if on {
                                t.accent_soft
                            } else if cresp.hovered() {
                                t.hover
                            } else {
                                t.field
                            },
                        );
                        let tex = shape_texture(&ctx, sh);
                        ui.painter().image(tex.id(), cr.shrink(4.0), Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)), t.text);
                        if cresp.on_hover_text(&sh.name).clicked() {
                            app.ui.presets_ui.custom_shape = sh.name.clone();
                            ui.close();
                        }
                    }
                });
            }
        });
    });
}

/// Custom Shape tool drag: places the selected shape in the dragged rect (⇧ keeps proportions).
pub fn finish_custom_shape(app: &mut PhotocraftApp, rect: [f64; 4], keep: bool, fill: Value, stroke: Value) {
    let name = app.ui.presets_ui.shape().to_string();
    let r = run(app, "shape.presets.place", json!({"preset": name, "rect": rect, "keepAspect": keep, "fill": fill, "stroke": stroke}));
    if r.is_none() && !photocraft_engine::presets::shapes::all_groups(&app.session).iter().any(|g| g.items.iter().any(|s| s.name == name)) {
        let template = "No custom shape \"{name}\": pick one in the options bar or Window › Shapes";
        app.ui.status = crate::i18n::fmt(tl!(template), &[("name", &name)]);
    }
}

// ------------------------------------------------------------------ Tool Presets

fn tool_id(t: Tool) -> String {
    let s = format!("{t:?}");
    let mut c = s.chars();
    c.next().map(|f| f.to_ascii_lowercase().to_string() + c.as_str()).unwrap_or_default()
}

/// Apply a tool preset: the engine sets the brush/colour, the shell switches tool and options.
pub fn select_tool_preset(app: &mut PhotocraftApp, name: &str) {
    let Some(r) = run(app, "tool.presets.select", json!({"preset": name})) else { return };
    if let Some(tool) = r["tool"].as_str().and_then(Tool::from_name) {
        app.ui.tool = tool;
        // Each tool keeps its own brush (#218), so switch it in before the preset's brush lands
        // on it, and re-apply the brush the command just set for the old tool.
        crate::paint_mouse::sync_tool_brush(app);
        if let Some(b) = r["options"].get("brush").filter(|b| b.is_object()) {
            let _ = app.run("tools.setBrush", json!({"brush": b}));
        }
    }
    if let Some(o) = r["options"].get("toolOptions").and_then(Value::as_object) {
        let mut cur = serde_json::to_value(&app.ui.tool_options).unwrap_or_default();
        if let Some(c) = cur.as_object_mut() {
            for (k, v) in o {
                c.insert(k.clone(), v.clone());
            }
        }
        if let Ok(opts) = serde_json::from_value(cur) {
            app.ui.tool_options = opts;
        }
    }
}

pub fn tool_presets_panel(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    let only = app.ui.presets_ui.current_tool_only;
    let cur = app.ui.tool;
    let items: Vec<(String, String)> =
        app.session.presets.tool_presets.iter().filter(|p| !only || Tool::from_name(&p.tool) == Some(cur)).map(|p| (p.name.clone(), p.tool.clone())).collect();
    let selected = app.ui.presets_ui.selected.get("toolPresets").cloned();
    let mut action: Option<(String, Value)> = None;
    let mut pick: Option<String> = None;
    // Rename field.
    if let Some((_, key, mut text)) = app.ui.presets_ui.renaming.clone().filter(|r| r.0 == "toolPresets") {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(tl!("Name:")).color(t.text_dim).size(11.5));
            let r = ui.add(egui::TextEdit::singleline(&mut text).desired_width(ui.available_width() - 4.0));
            if !r.has_focus() && !r.lost_focus() {
                r.request_focus();
            }
            if r.lost_focus() {
                app.ui.presets_ui.renaming = None;
                if ui.input(|i| i.key_pressed(egui::Key::Enter)) && !text.trim().is_empty() {
                    action = Some(("tool.presets.edit".into(), json!({"action": "rename", "preset": key, "name": text.trim()})));
                }
            } else {
                app.ui.presets_ui.renaming = Some(("toolPresets".into(), key.clone(), text.clone()));
            }
        });
    }
    egui::ScrollArea::vertical().id_salt("tool-presets").max_height(260.0).auto_shrink([false, true]).show(ui, |ui| {
        ui.spacing_mut().item_spacing.y = 0.0;
        if items.is_empty() {
            empty(ui, if only { tl!("No presets for the current tool.") } else { tl!("No tool presets.") });
        }
        for (name, tool) in &items {
            let (r, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 24.0), Sense::click());
            if selected.as_deref() == Some(name) {
                ui.painter().rect_filled(r, 0.0, t.row_selected);
            } else if resp.hovered() {
                ui.painter().rect_filled(r, 0.0, t.hover.gamma_multiply(0.5));
            }
            let icon = Tool::from_name(tool).map(crate::icons::tool_icon).unwrap_or("settings");
            crate::icons::paint(ui, Rect::from_center_size(pos2(r.left() + 14.0, r.center().y), vec2(16.0, 16.0)), icon, 13.0, t.icon);
            ui.painter().text(pos2(r.left() + 30.0, r.center().y), Align2::LEFT_CENTER, name, FontId::proportional(11.5), t.text);
            if resp.clicked() {
                pick = Some(name.clone());
            }
            resp.context_menu(|ui| {
                if ui.button(tl!("Rename Tool Preset…")).clicked() {
                    app.ui.presets_ui.renaming = Some(("toolPresets".into(), name.clone(), name.clone()));
                    ui.close();
                }
                if ui.button(tl!("Delete Tool Preset")).clicked() {
                    action = Some(("tool.presets.edit".into(), json!({"action": "delete", "preset": name})));
                    ui.close();
                }
            });
        }
    });
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        crate::widgets::checkbox(ui, &mut app.ui.presets_ui.current_tool_only, tl!("Current Tool Only"));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if crate::icons::button(ui, "trash", 24.0, false, tl!("Delete tool preset")).clicked()
                && let Some(n) = selected.clone()
            {
                action = Some(("tool.presets.edit".into(), json!({"action": "delete", "preset": n})));
            }
            if crate::icons::button(ui, "plus", 24.0, false, tl!("Create new tool preset")).clicked() {
                let label = cur.label().to_string();
                let opts = json!({"toolOptions": serde_json::to_value(&app.ui.tool_options).unwrap_or_default()});
                action = Some(("tool.presets.new".into(), json!({"name": label, "tool": tool_id(cur), "options": opts})));
            }
        });
    });
    if let Some(n) = pick {
        app.ui.presets_ui.selected.insert("toolPresets".into(), n.clone());
        select_tool_preset(app, &n);
    }
    if let Some((cmd, p)) = action {
        let created = cmd == "tool.presets.new";
        if let Some(r) = run(app, &cmd, p)
            && created
        {
            let n = r["name"].as_str().unwrap_or_default().to_string();
            app.ui.presets_ui.selected.insert("toolPresets".into(), n.clone());
            app.ui.presets_ui.renaming = Some(("toolPresets".into(), n.clone(), n));
        }
    }
}

// ------------------------------------------------------------------ Clone Source

pub fn clone_source_panel(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    let cs = app.session.presets.clone.clone();
    let slot = cs.active().clone();
    let mut action: Option<Value> = None;
    // The five source buttons.
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        for i in 0..photocraft_engine::presets::clone_source::SLOTS {
            let set = cs.slots[i].source.is_some();
            let tip = match cs.slots[i].source {
                Some(s) => format!("Clone source {}: {:.0}, {:.0}", i + 1, s[0], s[1]),
                None => crate::i18n::fmt(
                    tl!("Clone source {n} ({key}-click with Clone Stamp to set)"),
                    &[("n", &(i + 1).to_string()), ("key", &crate::shortcuts::pretty("Alt"))],
                ),
            };
            let r = crate::icons::button(ui, "stamp", 30.0, cs.active == i, &tip);
            if set {
                let d = r.rect.right_bottom() - vec2(6.0, 6.0);
                ui.painter().circle_filled(d, 2.5, t.accent);
            }
            if r.clicked() {
                run(app, "cloneSource.select", json!({"index": i}));
            }
        }
    });
    ui.add_space(4.0);
    let src_line =
        match (slot.source, slot.layer.and_then(|l| app.session.active().and_then(|d| d.doc.layer(photocraft_doc::LayerId(l)).map(|x| x.name.clone())))) {
            (Some(_), Some(l)) => format!("Source: {} : {l}", app.session.active().map(|d| d.doc.name.clone()).unwrap_or_default()),
            (Some(_), None) => "Source: set".to_string(),
            _ => crate::i18n::fmt(tl!("Source: not set ({key}-click with Clone Stamp)"), &[("key", &crate::shortcuts::pretty("Alt"))]),
        };
    ui.label(egui::RichText::new(src_line).color(t.text_dim).size(11.0));
    ui.add_space(4.0);
    let lbl = |ui: &mut egui::Ui, s: &str| {
        ui.add_sized(vec2(18.0, 18.0), egui::Label::new(egui::RichText::new(s).color(t.text_dim).size(11.5)));
    };
    let off = slot.offset().unwrap_or([0.0, 0.0]);
    let (mut ox, mut oy) = (off[0] as f32, off[1] as f32);
    let (mut w, mut h) = (slot.scale[0] as f32, slot.scale[1] as f32);
    let mut rot = slot.rotation as f32;
    egui::Grid::new("clone-src-grid").num_columns(5).spacing(vec2(6.0, 4.0)).show(ui, |ui| {
        lbl(ui, "X:");
        if crate::widgets::value_field(ui, &mut ox, -30000.0..=30000.0, "px", 72.0).changed() && slot.source.is_some() {
            action = Some(json!({"offset": [ox, oy]}));
        }
        lbl(ui, "W:");
        if crate::widgets::value_field(ui, &mut w, -1000.0..=1000.0, "%", 64.0).changed() {
            action = Some(json!({"width": w}));
        }
        // The W/H chain: with it on, a new W or H scales the other by the same factor.
        let icon = if slot.linked { "link" } else { "unlink" };
        if crate::icons::button(ui, icon, 22.0, slot.linked, tl!("Maintain aspect ratio")).clicked() {
            action = Some(json!({"linked": !slot.linked}));
        }
        ui.end_row();
        lbl(ui, "Y:");
        if crate::widgets::value_field(ui, &mut oy, -30000.0..=30000.0, "px", 72.0).changed() && slot.source.is_some() {
            action = Some(json!({"offset": [ox, oy]}));
        }
        lbl(ui, "H:");
        if crate::widgets::value_field(ui, &mut h, -1000.0..=1000.0, "%", 64.0).changed() {
            action = Some(json!({"height": h}));
        }
        ui.end_row();
        lbl(ui, "A:");
        if crate::widgets::value_field(ui, &mut rot, -180.0..=180.0, "°", 72.0).changed() {
            action = Some(json!({"rotation": rot}));
        }
        ui.horizontal(|ui| {
            if crate::icons::button(ui, "arrow-left-right", 22.0, slot.flip_h, tl!("Flip Horizontal")).clicked() {
                action = Some(json!({"flipH": !slot.flip_h}));
            }
            if crate::icons::button(ui, "rotate-cw", 22.0, false, tl!("Reset Transform")).clicked() {
                run(app, "cloneSource.resetTransform", json!({}));
            }
        });
        ui.horizontal(|ui| {
            if crate::icons::button(ui, "scaling", 22.0, slot.flip_v, tl!("Flip Vertical")).clicked() {
                action = Some(json!({"flipV": !slot.flip_v}));
            }
        });
        ui.end_row();
    });
    if let Some(p) = action {
        run(app, "cloneSource.set", p);
    }
    ui.add_space(6.0);
    crate::widgets::hairline(ui);
    ui.add_space(4.0);
    // Overlay options.
    let mut o = cs.overlay.clone();
    let before = o.clone();
    crate::widgets::checkbox(ui, &mut o.show, tl!("Show Overlay"));
    ui.horizontal(|ui| {
        ui.add_space(18.0);
        ui.label(egui::RichText::new(tl!("Opacity:")).color(t.text_dim).size(11.5));
        crate::widgets::value_field(ui, &mut o.opacity, 0.0..=100.0, "%", 56.0);
        let opts = [
            ("normal".to_string(), tl!("Normal")),
            ("darken".to_string(), tl!("Darken")),
            ("lighten".to_string(), tl!("Lighten")),
            ("difference".to_string(), tl!("Difference")),
        ];
        crate::widgets::dropdown(ui, "clone-overlay-mode", &mut o.blend, &opts, 90.0);
    });
    ui.horizontal(|ui| {
        ui.add_space(18.0);
        crate::widgets::checkbox(ui, &mut o.clipped, tl!("Clipped"));
        crate::widgets::checkbox(ui, &mut o.auto_hide, tl!("Auto Hide"));
        crate::widgets::checkbox(ui, &mut o.invert, tl!("Invert"));
    });
    if o != before {
        run(
            app,
            "cloneSource.overlay",
            json!({"show": o.show, "opacity": o.opacity, "clipped": o.clipped, "autoHide": o.auto_hide, "invert": o.invert, "blend": o.blend}),
        );
    }
}

/// Where the active clone source samples for document point `at` (for the canvas marker).
/// The first stroke anchors at its press, so the marker follows before the stroke commits.
pub fn clone_sample_point(app: &PhotocraftApp, at: Option<[f64; 2]>) -> Option<[f64; 2]> {
    use photocraft_engine::presets::clone_source::{Mapping, transform_matrix};
    let s = app.session.presets.clone.active();
    let src = s.source?;
    let stroking = app.drag.as_ref().filter(|d| matches!(d.tool, Tool::CloneStamp | Tool::Healing)).map(|d| d.start);
    let anchor = if app.ui.tool_options.clone_aligned { s.anchor.or(stroking) } else { stroking };
    match (anchor, at) {
        (Some(a), Some(h)) => {
            let m = Mapping { source: (src[0], src[1]), anchor: (a[0], a[1]), m: transform_matrix(s.scale, s.rotation, s.flip_h, s.flip_v) };
            let (x, y) = m.map(h[0], h[1]);
            Some([x, y])
        }
        _ => Some(src),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::menus::invoke;

    fn app() -> (PhotocraftApp, egui::Context) {
        let ctx = egui::Context::default();
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 64, "height": 48})).unwrap();
        app.sync_views();
        (app, ctx)
    }

    #[test]
    fn window_menu_toggles_panels_and_color_tabs() {
        let (mut app, ctx) = app();
        for id in PANELS {
            assert!(crate::menus::is_live(id), "{id}");
        }
        invoke(&mut app, &ctx, "window.panel.gradients", Value::Null).unwrap();
        assert!(app.ui.panels.color && app.ui.dock_tabs.color == 2);
        assert_eq!(checked(&app, "window.panel.gradients"), Some(true));
        assert_eq!(checked(&app, "window.panel.patterns"), Some(false));
        invoke(&mut app, &ctx, "window.panel.patterns", Value::Null).unwrap();
        assert_eq!(app.ui.dock_tabs.color, 3);
        // Choosing it again hides the card.
        invoke(&mut app, &ctx, "window.panel.patterns", Value::Null).unwrap();
        assert!(!app.ui.panels.color);
        for (id, f) in [("window.panel.styles", 0), ("window.panel.shapes", 1), ("window.panel.toolPresets", 2), ("window.panel.cloneSource", 3)] {
            invoke(&mut app, &ctx, id, Value::Null).unwrap();
            let p = &app.ui.presets_ui;
            assert!([p.styles, p.shapes, p.tool_presets, p.clone_source][f], "{id}");
            assert_eq!(checked(&app, id), Some(true));
            invoke(&mut app, &ctx, id, json!({"show": false})).unwrap();
            assert_eq!(checked(&app, id), Some(false));
        }
    }

    #[test]
    fn tool_preset_switches_tool_and_options() {
        let (mut app, _) = app();
        app.ui.tool = Tool::Brush;
        select_tool_preset(&mut app, "Radial Gradient");
        assert_eq!(app.ui.tool, Tool::Gradient);
        assert_eq!(app.ui.tool_options.gradient_style, "radial");
        select_tool_preset(&mut app, "Soft Eraser 60 px");
        assert_eq!(app.ui.tool, Tool::Eraser);
        assert_eq!(app.session.tools.brush.size, 60.0);
        assert_eq!(tool_id(Tool::CloneStamp), "cloneStamp");
        assert_eq!(tool_id(Tool::PatternStamp), "patternStamp");
        assert_eq!(Tool::from_name(&tool_id(Tool::CustomShape)), Some(Tool::CustomShape));
    }

    #[test]
    fn custom_shape_tool_places_the_picked_shape() {
        let (mut app, _) = app();
        app.ui.presets_ui.custom_shape = "Star".into();
        finish_custom_shape(&mut app, [4.0, 4.0, 20.0, 20.0], true, json!("#ff0000"), Value::Null);
        let d = app.session.active().unwrap();
        let l = d.doc.layer(d.active_layer.unwrap()).unwrap();
        assert_eq!(l.name, "Star");
        assert!(matches!(l.content, photocraft_doc::LayerContent::Shape(_)));
    }

    /// #3658: a group/folder drag reorders by name, stores only a real reorder, and falls back to
    /// the library's order when the saved order matches it or names a group that's gone.
    #[test]
    fn moving_a_folder_reorders_like_the_dock_tabs() {
        let names: Vec<String> = ["Grays", "Pastel", "Pure", "Dark"].iter().map(|s| s.to_string()).collect();
        let mut st = PresetUi::default();
        assert_eq!(st.display_order("swatches", &names), vec![0, 1, 2, 3]);
        st.move_group("swatches", &names, "Pastel", None);
        assert_eq!(st.group_order.get("swatches").unwrap(), &["Grays", "Pure", "Dark", "Pastel"]);
        st.move_group("swatches", &names, "Pastel", Some("Grays"));
        assert_eq!(st.group_order.get("swatches").unwrap(), &["Pastel", "Grays", "Pure", "Dark"]);
        assert_eq!(st.display_order("swatches", &names), vec![1, 0, 2, 3]);
        // Back to the library's own order: no entry, as the dock stores its default tab order.
        st.move_group("swatches", &names, "Pastel", Some("Pure"));
        assert!(!st.group_order.contains_key("swatches"), "a no-op order is not stored");
        // Out-of-range names, moves onto themselves and unknown `before` names change nothing bad.
        for (from, before) in [("Nope", None), ("Grays", Some("Grays"))] {
            let mut m = PresetUi::default();
            m.move_group("swatches", &names, from, before);
            assert!(m.group_order.is_empty(), "{from} {before:?}");
            assert_eq!(m.display_order("swatches", &names), vec![0, 1, 2, 3]);
        }
        // A missing group in a saved order follows the rest in library order.
        let mut m = PresetUi::default();
        m.group_order.insert("swatches".into(), vec!["Dark".into(), "Grays".into()]);
        assert_eq!(m.display_order("swatches", &names), vec![3, 0, 1, 2]);
    }

    /// #3658: the folder order persists with the UI state and reads leniently (an old state has no
    /// field, and garbage in one never fails the whole state).
    #[test]
    fn folder_order_persists_and_reads_leniently() {
        let old: PresetUi = serde_json::from_value(json!({"sizes": {"swatches": 22.0}})).unwrap();
        assert!(old.group_order.is_empty(), "an older state without the field still loads");
        let mut st = PresetUi::default();
        st.group_order.insert("swatches".into(), vec!["Dark".into(), "Grays".into()]);
        let back: PresetUi = serde_json::from_value(serde_json::to_value(&st).unwrap()).unwrap();
        assert_eq!(back.group_order.get("swatches").unwrap(), &["Dark", "Grays"]);
        for bad in [json!("x"), json!(7), json!({"swatches": "paths"}), json!({"swatches": [1, null, {"a": 1}]})] {
            let back: PresetUi = serde_json::from_value(json!({"groupOrder": bad})).unwrap();
            assert!(back.group_order.get("swatches").is_none_or(Vec::is_empty), "{bad}");
        }
    }

    /// Regression: preset drops mapped the pointer through the raw widget rect, ignoring the
    /// rulers' 16 px inset — a dropped shape landed 8 screen px from the pointer, and a drop on
    /// the ruler strip counted as a canvas drop. Drops now map through, and target, the same
    /// content rect the canvas draws with.
    #[test]
    fn preset_drops_map_through_the_ruler_inset() {
        let (mut app, _) = app();
        app.last_canvas_rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0));
        app.ui.views[0] = crate::state::View { zoom: 2.0, center: [32.0, 24.0], fit_pending: false, fill_pending: false, doc_size: [64, 48], rotation: 0.0 };
        let p = egui::pos2(400.0, 300.0);
        let off = crate::canvas::doc_point_at(&app, p).unwrap();
        assert_eq!(off, [32.0, 24.0], "without rulers the widget rect is the whole mapping");
        // With rulers the content rect shrinks by the inset on the left and top: the same screen
        // point is 8 screen px (4 document px at zoom 2) up-left of where it was.
        app.ui.extras.rulers = true;
        let on = crate::canvas::doc_point_at(&app, p).unwrap();
        assert!((on[0] - (off[0] - 4.0)).abs() < 1e-3, "{on:?} vs {off:?}");
        assert!((on[1] - (off[1] - 4.0)).abs() < 1e-3, "{on:?} vs {off:?}");
        // The ruler strip is no longer a drop target; the canvas proper still is.
        assert!(!canvas_drop_rect(&app).contains(egui::pos2(5.0, 300.0)));
        assert!(canvas_drop_rect(&app).contains(p));
    }

    #[test]
    fn clone_marker_follows_the_engine_slot() {
        let (mut app, _) = app();
        assert_eq!(clone_sample_point(&app, Some([5.0, 5.0])), None);
        app.run("cloneSource.set", json!({"source": [10, 10]})).unwrap();
        assert_eq!(clone_sample_point(&app, Some([30.0, 30.0])), Some([10.0, 10.0]));
        app.drag = Some(crate::canvas::Drag::new(Tool::CloneStamp, [40.0, 40.0], vec![[50.0, 30.0, 1.0]], egui::Modifiers::NONE, false));
        assert_eq!(clone_sample_point(&app, Some([50.0, 30.0])), Some([20.0, 0.0]));
        app.drag = None;
        app.session.presets.clone.active_mut().anchor = Some([20.0, 20.0]);
        assert_eq!(clone_sample_point(&app, Some([30.0, 30.0])), Some([20.0, 20.0]));
    }

    #[test]
    fn clone_marker_shows_only_while_painting() {
        // #668: ⌥-click used to leave a "+" on the canvas until the next stroke.
        let (mut app, _) = app();
        app.ui.tool = Tool::CloneStamp;
        crate::retouch_ui::set_source(&mut app, 10.0, 10.0);
        app.hover_doc = Some([40.0, 40.0]);
        assert_eq!(crate::retouch_ui::source_marker_point(&app), None, "hovering after ⌥-click shows nothing");
        app.drag = Some(crate::canvas::Drag::new(Tool::CloneStamp, [40.0, 40.0], vec![[40.0, 40.0, 1.0]], egui::Modifiers::NONE, false));
        assert!(crate::retouch_ui::source_marker_point(&app).is_some(), "painting shows where it samples");
        app.drag = None;
        app.ui.tool = Tool::Brush;
        assert_eq!(crate::retouch_ui::source_marker_point(&app), None);
    }
}
