//! Type panels: Character Styles + Paragraph Styles (one floating group with two tabs, as in
//! Photoshop), Glyphs, and the Edit › Check Spelling dialog.
//!
//! Everything acts through engine commands (`type.characterStyle.*`, `type.paragraphStyle.*`,
//! `type.insertText`, `edit.checkSpelling`); the panels only hold view state
//! ([`TypePanelsUi`], part of the serialisable UI state so the control channel can drive it).

use egui::{Align2, Color32, CornerRadius, Rect, RichText, Sense, Stroke, TextureHandle, TextureOptions, pos2, vec2};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::theme::Tokens;

/// View state of the type panels.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TypePanelsUi {
    /// The Character Styles / Paragraph Styles group is open.
    pub styles: bool,
    /// 0 = Character Styles, 1 = Paragraph Styles.
    pub styles_tab: usize,
    /// Glyphs panel is open.
    pub glyphs: bool,
    /// Style Options editor open for (paragraph?, id).
    pub options: Option<(bool, u32)>,
    /// Glyphs: font family and style ("" = follow the type selection), "Show" category, cell
    /// size in points of UI, recently used glyphs (most recent first).
    pub glyph_family: String,
    pub glyph_style: String,
    pub glyph_category: String,
    pub glyph_size: f32,
    pub glyph_recent: Vec<String>,
    pub glyph_selected: Option<String>,
    /// Edit › Check Spelling dialog.
    pub spell: Option<SpellDialog>,
}

/// State of the Check Spelling dialog.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SpellDialog {
    /// Misspellings from `edit.checkSpelling` (layer, start, end, word, suggestions).
    pub items: Vec<Value>,
    pub index: usize,
    /// Ignore All words for this check.
    pub ignore: Vec<String>,
    pub change_to: String,
    pub all_layers: bool,
    pub message: String,
}

const IDS: [&str; 6] = [
    "type.panels.characterStyles",
    "window.panel.characterStyles",
    "type.panels.paragraphStyles",
    "window.panel.paragraphStyles",
    "type.panels.glyphs",
    "window.panel.glyphs",
];

/// Menu ids handled here (live menu items).
pub fn handles(id: &str) -> bool {
    IDS.contains(&id)
}

pub fn checked(app: &PhotocraftApp, id: &str) -> Option<bool> {
    let p = &app.ui.type_panels;
    Some(match id {
        "type.panels.characterStyles" | "window.panel.characterStyles" => p.styles && p.styles_tab == 0,
        "type.panels.paragraphStyles" | "window.panel.paragraphStyles" => p.styles && p.styles_tab == 1,
        "type.panels.glyphs" | "window.panel.glyphs" => p.glyphs,
        _ => return None,
    })
}

/// Window/Type › panel toggles, and Edit › Check Spelling… (opens the dialog when run without an
/// `action`). `params.show` forces a panel open or closed.
pub fn menu(app: &mut PhotocraftApp, id: &str, params: &Value) -> Option<Result<Value, String>> {
    if id == "edit.checkSpelling" && params.get("action").is_none() && params.get("ui").is_none_or(|u| u.as_bool() != Some(false)) {
        return Some(open_spelling(app, params.get("allLayers").and_then(Value::as_bool).unwrap_or(true)));
    }
    if !handles(id) {
        return None;
    }
    let want = params.get("show").and_then(Value::as_bool);
    let p = &mut app.ui.type_panels;
    let visible = if id.ends_with("glyphs") {
        p.glyphs = want.unwrap_or(!p.glyphs);
        p.glyphs
    } else {
        let tab = usize::from(id.ends_with("paragraphStyles"));
        let showing = p.styles && p.styles_tab == tab;
        // `{"options": styleId}` opens the panel with that style's Style Options.
        let options = params.get("options").and_then(Value::as_u64).map(|o| (tab == 1, o as u32));
        let show = if options.is_some() { true } else { want.unwrap_or(!showing) };
        if options.is_some() {
            p.options = options;
        }
        if show {
            p.styles = true;
            p.styles_tab = tab;
        } else if p.styles_tab == tab {
            p.styles = false;
        }
        show
    };
    Some(Ok(json!({ "visible": visible })))
}

fn run(app: &mut PhotocraftApp, id: &str, p: Value) -> Option<Value> {
    match app.run(id, p) {
        Ok(v) => Some(v),
        Err(e) => {
            app.ui.status = e;
            app.ui.status_error = true;
            None
        }
    }
}

/// The type the panels act on: the layer being edited (with its selection), else the selected
/// layers. Returns params with `layer` + `range`, or `{}` (selected layers).
fn target(app: &PhotocraftApp) -> Value {
    if let Some(ed) = &app.ui.text_edit {
        let (a, b) = (ed.caret.min(ed.anchor), ed.caret.max(ed.anchor));
        return json!({ "layer": ed.layer, "range": [a, b], "coalesce": ed.session });
    }
    json!({})
}

fn float_frame(t: &Tokens) -> egui::Frame {
    egui::Frame::NONE
        .fill(t.card)
        .stroke(Stroke::new(1.0, t.card_border))
        .corner_radius(CornerRadius::same(t.radius_lg as u8))
        .shadow(egui::Shadow { offset: [0, 10], blur: 30, spread: 0, color: t.shadow })
        .inner_margin(egui::Margin::same(8))
}

/// Draws the open type panels and the spelling dialog.
pub fn windows(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let t = Tokens::get(ctx);
    let canvas = app.last_canvas_rect;
    if app.ui.type_panels.styles {
        let mut close = false;
        egui::Window::new(tl!("Type Styles"))
            .id(egui::Id::new("type-styles-window"))
            .title_bar(false)
            .resizable(false)
            .frame(float_frame(&t))
            .default_pos(pos2((canvas.right() - 314.0).max(canvas.left() + 8.0), canvas.top() + 40.0))
            .show(ctx, |ui| {
                ui.set_width(290.0);
                ui.horizontal(|ui| {
                    let mut tab = app.ui.type_panels.styles_tab;
                    for (i, name) in [tl!("Character Styles"), tl!("Paragraph Styles")].iter().enumerate() {
                        if crate::widgets::pill_tab(ui, name, tab == i).clicked() {
                            tab = i;
                        }
                    }
                    if tab != app.ui.type_panels.styles_tab {
                        app.ui.type_panels.styles_tab = tab;
                        app.ui.type_panels.options = None;
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if crate::icons::button(ui, "x", 20.0, false, tl!("Close")).clicked() {
                            close = true;
                        }
                    });
                });
                crate::widgets::hairline(ui);
                ui.add_space(4.0);
                styles_panel(app, ui, app.ui.type_panels.styles_tab == 1);
            });
        if close {
            app.ui.type_panels.styles = false;
        }
    }
    if app.ui.type_panels.glyphs {
        let mut close = false;
        egui::Window::new(tl!("Glyphs"))
            .id(egui::Id::new("glyphs-window"))
            .title_bar(false)
            .resizable(false)
            .frame(float_frame(&t))
            .default_pos(pos2((canvas.right() - 314.0 - 372.0).max(canvas.left() + 8.0), canvas.top() + 40.0))
            .show(ctx, |ui| {
                ui.set_width(340.0);
                ui.horizontal(|ui| {
                    ui.label(RichText::new(tl!("Glyphs")).color(t.text).size(12.0).strong());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if crate::icons::button(ui, "x", 20.0, false, tl!("Close")).clicked() {
                            close = true;
                        }
                    });
                });
                crate::widgets::hairline(ui);
                ui.add_space(4.0);
                glyphs_panel(app, ui);
            });
        if close {
            app.ui.type_panels.glyphs = false;
        }
    }
    if app.ui.type_panels.spell.is_some() {
        spelling_window(app, ctx);
    }
}

pub(crate) mod color_picker;

// ------------------------------------------------------------------ styles

pub fn styles_panel(app: &mut PhotocraftApp, ui: &mut egui::Ui, paragraph: bool) {
    let t = Tokens::get(ui.ctx());
    if app.session.active().is_none() {
        ui.label(RichText::new(tl!("No document")).color(t.text_faint).size(11.5));
        return;
    }
    let prefix = if paragraph { "type.paragraphStyle" } else { "type.characterStyle" };
    let tgt = target(app);
    let Ok(list) = app.session.execute(&format!("{prefix}.list"), tgt.clone()) else { return };
    let styles = list["styles"].as_array().cloned().unwrap_or_default();
    let cur = &list["current"];
    let (cur_id, over) = if paragraph {
        (cur["paragraph"].as_u64(), cur["paragraphOverride"].as_bool().unwrap_or(false))
    } else {
        (cur["character"].as_u64(), cur["characterOverride"].as_bool().unwrap_or(false))
    };
    let has_target = !cur.is_null();
    let selected = app.ui.type_panels.options.filter(|o| o.0 == paragraph).map(|o| u64::from(o.1)).or(cur_id);
    let mut action: Option<(String, Value)> = None;
    let row_h = 22.0;
    egui::ScrollArea::vertical().id_salt(("type-styles", paragraph)).max_height(220.0).auto_shrink([false, true]).show(ui, |ui| {
        for s in &styles {
            let id = s["id"].as_u64().unwrap_or(0);
            let name = s["name"].as_str().unwrap_or("");
            let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), row_h), Sense::click());
            let is_cur = has_target && cur_id == Some(id);
            if is_cur {
                ui.painter().rect_filled(rect, 0.0, t.row_selected);
            } else if selected == Some(id) {
                ui.painter().rect_filled(rect, 0.0, t.hover);
            } else if resp.hovered() {
                ui.painter().rect_filled(rect, 0.0, t.hover.gamma_multiply(0.5));
            }
            let label = if is_cur && over { format!("{name}+") } else { name.to_string() };
            let color = if id == 0 { t.text_dim } else { t.text };
            ui.painter().text(pos2(rect.left() + 8.0, rect.center().y), Align2::LEFT_CENTER, label, egui::FontId::proportional(12.0), color);
            if resp.double_clicked() && (id != 0 || paragraph) {
                app.ui.type_panels.options = Some((paragraph, id as u32));
            } else if resp.clicked() {
                let alt = ui.input(|i| i.modifiers.alt);
                if has_target {
                    let mut p = tgt.clone();
                    p["id"] = json!(id);
                    p["clearOverrides"] = json!(alt);
                    if !paragraph && p.get("range").and_then(Value::as_array).is_some_and(|r| r[0] == r[1]) {
                        app.ui.status = tl!("Select text to apply a character style").into();
                        app.ui.status_error = false;
                    } else {
                        action = Some((format!("{prefix}.apply"), p));
                    }
                }
                if app.ui.type_panels.options.is_some_and(|o| o.0 == paragraph) {
                    app.ui.type_panels.options = Some((paragraph, id as u32));
                }
            }
            resp.on_hover_text(if id == 0 && !paragraph {
                tl!("No character style").to_string()
            } else {
                crate::i18n::fmt(
                    tl!("Click to apply, {key}-click to clear overrides, double-click for Style Options"),
                    &[("key", &crate::shortcuts::pretty("Alt"))],
                )
            });
        }
    });
    ui.add_space(4.0);
    crate::widgets::hairline(ui);
    ui.add_space(2.0);
    let sel = selected.filter(|i| *i != 0 || paragraph);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        if crate::icons::button(ui, "ban", 22.0, false, tl!("Clear Override")).clicked() && has_target && over {
            action = Some((format!("{prefix}.clearOverride"), tgt.clone()));
        }
        if crate::icons::button(ui, "check", 22.0, false, tl!("Redefine style by current selection")).clicked()
            && has_target
            && cur_id.is_some_and(|i| i != 0 || paragraph)
        {
            action = Some((format!("{prefix}.redefine"), tgt.clone()));
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if crate::icons::button(ui, "trash", 22.0, false, tl!("Delete style")).clicked()
                && let Some(id) = sel.filter(|i| *i != 0)
            {
                action = Some((format!("{prefix}.delete"), json!({ "id": id })));
                app.ui.type_panels.options = None;
            }
            if crate::icons::button(ui, "plus", 22.0, false, tl!("Create new style (from the selected type)")).clicked() {
                let mut p = tgt.clone();
                p["fromSelection"] = json!(has_target);
                action = Some((format!("{prefix}.new"), p));
            }
            if crate::icons::button(ui, "copy", 22.0, false, tl!("Duplicate style")).clicked()
                && let Some(id) = sel
            {
                action = Some((format!("{prefix}.duplicate"), json!({ "id": id })));
            }
            if crate::icons::button(ui, "settings", 22.0, false, tl!("Style Options…")).clicked()
                && let Some(id) = sel
            {
                app.ui.type_panels.options = Some((paragraph, id as u32));
            }
        });
    });
    if let Some((id, p)) = action
        && let Some(r) = run(app, &id, p)
        && id.ends_with(".new")
        && let Some(nid) = r["id"].as_u64()
    {
        app.ui.type_panels.options = Some((paragraph, nid as u32));
    }
    if let Some((para, id)) = app.ui.type_panels.options
        && para == paragraph
    {
        if let Some(s) = styles.iter().find(|s| s["id"].as_u64() == Some(u64::from(id))) {
            ui.add_space(4.0);
            crate::widgets::hairline(ui);
            options_editor(app, ui, paragraph, s);
        } else {
            app.ui.type_panels.options = None;
        }
    }
}

/// Style Options: name plus the common attributes; a checkbox marks an attribute as part of the
/// style (unchecked = inherited).
fn options_editor(app: &mut PhotocraftApp, ui: &mut egui::Ui, paragraph: bool, s: &Value) {
    let t = Tokens::get(ui.ctx());
    let prefix = if paragraph { "type.paragraphStyle" } else { "type.characterStyle" };
    let id = s["id"].as_u64().unwrap_or(0);
    let (char_attrs, para_attrs) = if paragraph { (s["characterAttrs"].clone(), s["paragraphAttrs"].clone()) } else { (s["attrs"].clone(), Value::Null) };
    let resolved_c = if paragraph { s["resolved"]["character"].clone() } else { s["resolved"].clone() };
    let resolved_p = s["resolved"]["paragraph"].clone();
    let mut set: Option<Value> = None;
    let mut clear: Vec<&str> = Vec::new();
    let mut open_color = false;
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label(RichText::new(tl!("Style Options")).color(t.text).size(11.5).strong());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if crate::icons::button(ui, "x", 18.0, false, tl!("Close Style Options")).clicked() {
                app.ui.type_panels.options = None;
            }
        });
    });
    // Name.
    let key = egui::Id::new(("style-name", paragraph, id));
    let mut name: String = ui.data(|d| d.get_temp(key)).unwrap_or_else(|| s["name"].as_str().unwrap_or("").to_string());
    ui.horizontal(|ui| {
        ui.label(RichText::new(tl!("Name")).color(t.text_dim).size(11.5));
        let r = ui.add_enabled(id != 0, egui::TextEdit::singleline(&mut name).desired_width(170.0));
        if r.lost_focus() && name.trim() != s["name"].as_str().unwrap_or("") && !name.trim().is_empty() {
            let _ = run(app, &format!("{prefix}.rename"), json!({ "id": id, "name": name.trim() }));
            ui.data_mut(|d| d.remove::<String>(key));
        } else if r.changed() {
            ui.data_mut(|d| d.insert_temp(key, name.clone()));
        }
    });
    let defined = |attrs: &Value, k: &str| attrs.get(k).is_some();
    // Preferences ▸ Type ▸ Font Preview Size sizes the family picker's samples.
    let preview = app.session.prefs().type_.font_preview;
    // Preferences ▸ Units & Rulers ▸ Type: the model stores points; the fields may show pixels
    // (by the document's resolution) or millimetres.
    let (type_unit, unit_label) = crate::type_tool::type_unit(app);
    egui::Grid::new(("style-opts", paragraph, id)).num_columns(3).spacing([6.0, 3.0]).show(ui, |ui| {
        // (label, model key, panel key, kind)
        let rows: &[(&str, &str, &str, &str)] = &[
            (tl!("Font"), "font_family", "font", "font"),
            (tl!("Size"), "size_pt", "size", "pt"),
            (tl!("Leading"), "leading_pt", "leading", "leading"),
            (tl!("Tracking"), "tracking", "tracking", "num"),
            (tl!("Baseline"), "baseline_shift_pt", "baselineShift", "pt"),
            (tl!("Color"), "color", "color", "color"),
            (tl!("Faux Bold"), "faux_bold", "fauxBold", "bool"),
            (tl!("Faux Italic"), "faux_italic", "fauxItalic", "bool"),
            (tl!("Underline"), "underline", "underline", "bool"),
            (tl!("Strikethrough"), "strikethrough", "strikethrough", "bool"),
        ];
        for &(label, key, panel, kind) in rows {
            let mut on = defined(&char_attrs, key);
            let was = on;
            ui.checkbox(&mut on, "");
            ui.label(RichText::new(tl!(&label)).color(if on { t.text } else { t.text_faint }).size(11.5));
            let v = if on { char_attrs[key].clone() } else { resolved_c[key].clone() };
            ui.add_enabled_ui(on, |ui| match kind {
                "font" => {
                    let mut f = v.as_str().unwrap_or("").to_string();
                    if family_picker(ui, &format!("style-font-{paragraph}-{id}"), &mut f, preview) {
                        set = Some(json!({ panel: f }));
                    }
                }
                "pt" | "num" | "leading" => {
                    let auto = kind == "leading" && v.is_null();
                    let mut x = v.as_f64().unwrap_or(0.0) as f32;
                    let range = if kind == "num" { -1000.0..=10000.0 } else { -1296.0..=1296.0 };
                    let unit = if kind == "num" { "" } else { " pt" };
                    if auto {
                        ui.label(RichText::new(tl!("Auto")).color(t.text_dim).size(11.5));
                    } else if crate::widgets::value_field(ui, &mut x, range, unit, 70.0).changed() {
                        set = Some(json!({ key: x }));
                    }
                }
                "color" => {
                    let c = serde_json::from_value::<photocraft_doc::Color>(v.clone()).map(|c| c.to_rgb()).unwrap_or([0.0; 3]);
                    let [r, g, b] = c.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8);
                    open_color = crate::widgets::color_swatch(ui, Color32::from_rgb(r, g, b)).on_hover_text(tl!("Color Picker (Text Color)")).clicked();
                }
                _ => {
                    let mut b = v.as_bool().unwrap_or(false);
                    if ui.checkbox(&mut b, "").changed() {
                        set = Some(json!({ key: b }));
                    }
                }
            });
            ui.end_row();
            if on != was {
                if on {
                    set = Some(json!({ key: resolved_c[key].clone() }));
                } else {
                    clear.push(key);
                }
            }
        }
        if paragraph {
            let rows: &[(&str, &str, &str)] = &[
                (tl!("Align"), "align", "align"),
                (tl!("Indent First"), "first_line_indent_pt", "pt"),
                (tl!("Indent Left"), "start_indent_pt", "pt"),
                (tl!("Indent Right"), "end_indent_pt", "pt"),
                (tl!("Space Before"), "space_before_pt", "pt"),
                (tl!("Space After"), "space_after_pt", "pt"),
                (tl!("Hyphenate"), "hyphenate", "bool"),
            ];
            for &(label, key, kind) in rows {
                let mut on = defined(&para_attrs, key);
                let was = on;
                ui.checkbox(&mut on, "");
                ui.label(RichText::new(tl!(&label)).color(if on { t.text } else { t.text_faint }).size(11.5));
                let v = if on { para_attrs[key].clone() } else { resolved_p[key].clone() };
                ui.add_enabled_ui(on, |ui| match kind {
                    "align" => {
                        let mut a = v.as_str().unwrap_or("Left").to_string();
                        // English values for the engine; labels go through `dropdown`'s `tl!`.
                        let opts: Vec<(String, &str)> = [
                            ("Left", "Left"),
                            ("Center", "Center"),
                            ("Right", "Right"),
                            ("JustifyLeft", "Justify last left"),
                            ("JustifyCenter", "Justify last centered"),
                            ("JustifyRight", "Justify last right"),
                            ("JustifyAll", "Justify all"),
                        ]
                        .into_iter()
                        .map(|(v, l)| (v.to_string(), l))
                        .collect();
                        if crate::widgets::dropdown(ui, &format!("style-align-{id}"), &mut a, &opts, 110.0) {
                            set = Some(json!({ "align": a }));
                        }
                    }
                    "pt" => {
                        let mut x = v.as_f64().unwrap_or(0.0) as f32 * type_unit;
                        if crate::widgets::value_field(ui, &mut x, -1296.0 * type_unit..=1296.0 * type_unit, &format!(" {unit_label}"), 70.0).changed() {
                            set = Some(json!({ key: x / type_unit }));
                        }
                    }
                    _ => {
                        let mut b = v.as_bool().unwrap_or(false);
                        if ui.checkbox(&mut b, "").changed() {
                            set = Some(json!({ key: b }));
                        }
                    }
                });
                ui.end_row();
                if on != was {
                    if on {
                        set = Some(json!({ key: resolved_p[key].clone() }));
                    } else {
                        clear.push(key);
                    }
                }
            }
        }
    });
    if let Some(attrs) = set {
        let _ = run(app, &format!("{prefix}.set"), json!({ "id": id, "attrs": attrs }));
    }
    if !clear.is_empty() {
        let mut c: Vec<&str> = clear.clone();
        if c.contains(&"font_family") {
            c.extend(["font_style", "postscript_name", "weight", "italic"]);
        }
        let _ = run(app, &format!("{prefix}.set"), json!({ "id": id, "clear": c }));
    }
    if open_color {
        let kind = if paragraph { color_picker::Kind::Paragraph } else { color_picker::Kind::Character };
        match u32::try_from(id).map_err(|_| "invalid text style id".to_string()).and_then(|id| color_picker::open(app, kind, id)) {
            Ok(_) => {}
            Err(error) => {
                app.ui.status = error;
                app.ui.status_error = true;
            }
        }
    }
}

/// Searchable family combo box.
fn family_picker(ui: &mut egui::Ui, salt: &str, current: &mut String, preview: photocraft_engine::prefs::FontPreview) -> bool {
    if !crate::type_tool::fonts_ready(ui, 140.0) {
        return false;
    }
    let mut shown = if current.is_empty() { photocraft_text::fonts::DEFAULT_FAMILY.to_string() } else { current.clone() };
    let changed = crate::type_tool::family_picker_in(ui, salt, &mut shown, 140.0, &crate::type_tool::families(), preview);
    if changed {
        *current = shown;
    }
    changed
}

// ------------------------------------------------------------------ glyphs

/// Font family/style the Glyphs panel shows: its own choice, else the type selection's font.
fn glyph_font(app: &PhotocraftApp) -> (String, String) {
    let p = &app.ui.type_panels;
    if !p.glyph_family.is_empty() {
        return (p.glyph_family.clone(), if p.glyph_style.is_empty() { tl!("Regular").into() } else { p.glyph_style.clone() });
    }
    let layer = app.ui.text_edit.as_ref().map(|e| photocraft_doc::LayerId(e.layer)).or_else(|| app.session.active().and_then(|s| s.active_layer));
    if let Some(st) = app.session.active()
        && let Some(id) = layer
        && let Some(photocraft_doc::LayerContent::Text(tl)) = st.doc.layer(id).map(|l| &l.content)
        && let Some(r) = tl.char_runs().first()
    {
        let fam = if r.style.font_family.is_empty() { photocraft_text::fonts::DEFAULT_FAMILY.to_string() } else { r.style.font_family.clone() };
        let style = crate::type_tool::selected_style(&r.style);
        return (fam, style);
    }
    (photocraft_text::fonts::DEFAULT_FAMILY.into(), tl!("Regular").into())
}

fn char_style_for(family: &str, style: &str) -> photocraft_doc::text::CharStyle {
    let mut st = photocraft_doc::text::CharStyle { font_family: family.into(), ..Default::default() };
    photocraft_engine::type_cmds::apply_char_props(&mut st, &json!({ "fontStyle": style }));
    st
}

fn glyph_texture(ctx: &egui::Context, family: &str, style: &str, c: char, px: u32) -> TextureHandle {
    let id = egui::Id::new(("glyph-tex", family, style, c, px));
    if let Some(t) = ctx.data(|d| d.get_temp::<TextureHandle>(id)) {
        return t;
    }
    let st = char_style_for(family, style);
    let (n, alpha) = {
        let mut eng = photocraft_text::shared().lock().unwrap_or_else(|e| e.into_inner());
        photocraft_text::glyphs::preview(&mut eng, &st, c, px)
    };
    let img = egui::ColorImage::from_rgba_unmultiplied([n as usize, n as usize], &alpha.iter().flat_map(|a| [255, 255, 255, *a]).collect::<Vec<u8>>());
    let tex = ctx.load_texture(format!("glyph-{c}-{px}"), img, TextureOptions::LINEAR);
    ctx.data_mut(|d| d.insert_temp(id, tex.clone()));
    tex
}

/// Characters of a face, cached per (family, style).
fn charmap(ctx: &egui::Context, family: &str, style: &str) -> std::sync::Arc<Vec<char>> {
    let id = egui::Id::new(("glyph-cmap", family, style));
    if let Some(v) = ctx.data(|d| d.get_temp::<std::sync::Arc<Vec<char>>>(id)) {
        return v;
    }
    let st = char_style_for(family, style);
    let v = std::sync::Arc::new(photocraft_text::shared().lock().unwrap_or_else(|e| e.into_inner()).fonts.charmap_for_style(&st));
    ctx.data_mut(|d| d.insert_temp(id, v.clone()));
    v
}

/// Inserts `g` into the type being edited (replacing the selection), else at the end of the
/// active type layer. Returns whether it was inserted.
pub fn insert_glyph(app: &mut PhotocraftApp, g: &str) -> bool {
    let p = if let Some(ed) = app.ui.text_edit.clone() {
        let (a, b) = (ed.caret.min(ed.anchor), ed.caret.max(ed.anchor));
        json!({ "layer": ed.layer, "text": g, "range": [a, b], "coalesce": ed.session })
    } else {
        json!({ "text": g })
    };
    let Some(r) = run(app, "type.insertText", p) else { return false };
    if let (Some(e), Some(c)) = (app.ui.text_edit.as_mut(), r["caret"].as_u64()) {
        e.caret = c as usize;
        e.anchor = e.caret;
    }
    let rec = &mut app.ui.type_panels.glyph_recent;
    rec.retain(|x| x != g);
    rec.insert(0, g.to_string());
    rec.truncate(12);
    true
}

pub fn glyphs_panel(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    // Every glyph cell needs the font database: wait for the font scan without blocking.
    if !crate::type_tool::fonts_ready(ui, 260.0) {
        return;
    }
    let t = Tokens::get(ui.ctx());
    let ctx = ui.ctx().clone();
    let (family, style) = glyph_font(app);
    // Font family + style.
    ui.horizontal(|ui| {
        let mut f = family.clone();
        if family_picker(ui, "glyph-family", &mut f, app.session.prefs().type_.font_preview) {
            app.ui.type_panels.glyph_family = f;
            app.ui.type_panels.glyph_style = tl!("Regular").into();
        }
        let styles = crate::type_tool::styles(&family);
        let mut s = style.clone();
        let opts: Vec<(String, &str)> = styles.iter().map(|x| (x.clone(), x.as_str())).collect();
        if crate::widgets::dropdown(ui, "glyph-style", &mut s, &opts, 110.0) {
            app.ui.type_panels.glyph_family = family.clone();
            app.ui.type_panels.glyph_style = s;
        }
    });
    // Show: category.
    let mut cat = if app.ui.type_panels.glyph_category.is_empty() { tl!("Entire Font").to_string() } else { app.ui.type_panels.glyph_category.clone() };
    ui.horizontal(|ui| {
        ui.label(RichText::new(tl!("Show:")).color(t.text_dim).size(11.5));
        let opts: Vec<(String, &str)> = photocraft_text::glyphs::CATEGORIES.iter().map(|c| (c.to_string(), *c)).collect();
        if crate::widgets::dropdown(ui, "glyph-category", &mut cat, &opts, 160.0) {
            app.ui.type_panels.glyph_category = cat.clone();
        }
    });
    let size = if app.ui.type_panels.glyph_size <= 0.0 { 32.0 } else { app.ui.type_panels.glyph_size };
    let ppp = ctx.pixels_per_point();
    let px = ((size - 6.0) * ppp).round().max(8.0) as u32;
    let mut clicked: Option<String> = None;
    let mut activated: Option<String> = None;
    // Recently used.
    let recent = app.ui.type_panels.glyph_recent.clone();
    ui.add_space(2.0);
    ui.label(RichText::new(tl!("Recently Used")).color(t.text_faint).size(10.5));
    let (rr, _) = ui.allocate_exact_size(vec2(ui.available_width(), 26.0), Sense::hover());
    ui.painter().rect_filled(rr, 2.0, t.field);
    for (i, g) in recent.iter().enumerate() {
        let cell = Rect::from_min_size(pos2(rr.left() + 2.0 + i as f32 * 26.0, rr.top() + 1.0), vec2(24.0, 24.0));
        if cell.right() > rr.right() {
            break;
        }
        let resp = ui.interact(cell, ui.id().with(("glyph-recent", i)), Sense::click());
        if resp.hovered() {
            ui.painter().rect_filled(cell, 2.0, t.hover);
        }
        if let Some(c) = g.chars().next() {
            let tex = glyph_texture(&ctx, &family, &style, c, (20.0 * ppp) as u32);
            ui.painter().image(tex.id(), cell.shrink(2.0), Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), t.text);
        }
        if resp.double_clicked() || resp.clicked() {
            activated = Some(g.clone());
        }
    }
    ui.add_space(4.0);
    // Grid.
    let cmap = charmap(&ctx, &family, &style);
    let chars: Vec<char> = cmap.iter().copied().filter(|c| photocraft_text::glyphs::in_category(*c, &cat)).collect();
    // Leave room for the scroll bar.
    let width = ui.available_width() - 12.0;
    let cols = ((width / size).floor() as usize).max(1);
    let rows = chars.len().div_ceil(cols);
    let selected = app.ui.type_panels.glyph_selected.clone();
    egui::ScrollArea::vertical().id_salt("glyph-grid").max_height(260.0).auto_shrink([false, false]).show_rows(ui, size, rows, |ui, range| {
        for row in range {
            let (rect, _) = ui.allocate_exact_size(vec2(width, size), Sense::hover());
            for col in 0..cols {
                let Some(&c) = chars.get(row * cols + col) else { break };
                let cell = Rect::from_min_size(pos2(rect.left() + col as f32 * size, rect.top()), vec2(size, size));
                let resp = ui.interact(cell, ui.id().with(("glyph", c)), Sense::click());
                let is_sel = selected.as_deref().and_then(|s| s.chars().next()) == Some(c);
                ui.painter().rect_stroke(cell, 0.0, Stroke::new(1.0, t.separator), egui::StrokeKind::Inside);
                if is_sel {
                    ui.painter().rect_filled(cell.shrink(1.0), 0.0, t.accent);
                } else if resp.hovered() {
                    ui.painter().rect_filled(cell.shrink(1.0), 0.0, t.hover);
                }
                let tex = glyph_texture(&ctx, &family, &style, c, px);
                let tint = if is_sel { Color32::WHITE } else { t.text };
                ui.painter().image(tex.id(), cell.shrink(3.0), Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), tint);
                let resp = resp.on_hover_text(format!("U+{:04X}", c as u32));
                if resp.double_clicked() {
                    activated = Some(c.to_string());
                } else if resp.clicked() {
                    clicked = Some(c.to_string());
                }
            }
        }
    });
    // Footer: family name, size slider.
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        let info = selected
            .as_deref()
            .and_then(|s| s.chars().next())
            .map(|c| format!("U+{:04X}  ·  {} glyphs", c as u32, chars.len()))
            .unwrap_or_else(|| format!("{} glyphs", chars.len()));
        ui.label(RichText::new(info).color(t.text_faint).size(10.5));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let mut s = size;
            ui.add_sized([90.0, 16.0], egui::Slider::new(&mut s, 24.0..=64.0).show_value(false));
            if (s - size).abs() > 0.1 {
                app.ui.type_panels.glyph_size = s;
            }
            let (r, _) = ui.allocate_exact_size(vec2(14.0, 14.0), Sense::hover());
            crate::icons::paint(ui, r, "zoom-in", 12.0, t.icon);
        });
    });
    if let Some(c) = clicked {
        app.ui.type_panels.glyph_selected = Some(c);
    }
    if let Some(g) = activated {
        app.ui.type_panels.glyph_selected = Some(g.clone());
        if !insert_glyph(app, &g) {
            app.ui.status = tl!("Select a type layer or click into text to insert glyphs").into();
        }
    }
}

// ------------------------------------------------------------------ spelling

fn refresh_spelling(app: &mut PhotocraftApp) {
    let Some(d) = app.ui.type_panels.spell.clone() else { return };
    let p = json!({ "action": "list", "allLayers": d.all_layers, "ignore": d.ignore, "suggestions": 8 });
    let items = run(app, "edit.checkSpelling", p).and_then(|r| r["misspellings"].as_array().cloned()).unwrap_or_default();
    if let Some(s) = app.ui.type_panels.spell.as_mut() {
        s.items = items;
        s.change_to = s.items.get(s.index).and_then(|i| i["suggestions"][0].as_str()).unwrap_or("").to_string();
    }
}

/// Opens Edit › Check Spelling (returns the misspellings).
pub fn open_spelling(app: &mut PhotocraftApp, all_layers: bool) -> Result<Value, String> {
    app.session.execute("edit.checkSpelling", json!({ "action": "list", "allLayers": all_layers })).map_err(|e| e.to_string())?;
    app.ui.type_panels.spell = Some(SpellDialog { all_layers, ..Default::default() });
    refresh_spelling(app);
    let d = app.ui.type_panels.spell.as_ref().map(|d| d.items.clone()).unwrap_or_default();
    Ok(json!({ "dialog": "checkSpelling", "misspellings": d }))
}

fn spelling_window(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let t = Tokens::get(ctx);
    let Some(d) = app.ui.type_panels.spell.clone() else { return };
    let item = d.items.get(d.index).cloned();
    let mut act: Option<&str> = None;
    let mut change_to = d.change_to.clone();
    let mut all_layers = d.all_layers;
    egui::Window::new(tl!("Check Spelling"))
        .id(egui::Id::new("check-spelling"))
        .collapsible(false)
        .resizable(false)
        .frame(float_frame(&t).inner_margin(egui::Margin::same(14)))
        .title_bar(false)
        .anchor(Align2::CENTER_CENTER, vec2(0.0, -40.0))
        .show(ctx, |ui| {
            ui.set_width(380.0);
            ui.label(RichText::new(tl!("Check Spelling")).color(t.text).size(13.0).strong());
            ui.add_space(6.0);
            ui.horizontal_top(|ui| {
                ui.vertical(|ui| {
                    ui.set_width(250.0);
                    ui.label(RichText::new(tl!("Not in Dictionary:")).color(t.text_dim).size(11.5));
                    let word = item.as_ref().and_then(|i| i["word"].as_str()).unwrap_or("");
                    let shown = if item.is_some() {
                        word.to_string()
                    } else if d.items.is_empty() || d.index >= d.items.len() {
                        "".into()
                    } else {
                        String::new()
                    };
                    ui.add_enabled(false, egui::TextEdit::singleline(&mut shown.clone()).desired_width(240.0));
                    ui.add_space(4.0);
                    ui.label(RichText::new(tl!("Change To:")).color(t.text_dim).size(11.5));
                    ui.add_enabled(item.is_some(), egui::TextEdit::singleline(&mut change_to).desired_width(240.0));
                    ui.add_space(4.0);
                    ui.label(RichText::new(tl!("Suggestions:")).color(t.text_dim).size(11.5));
                    egui::Frame::NONE.fill(t.field).corner_radius(CornerRadius::same(3)).inner_margin(egui::Margin::same(4)).show(ui, |ui| {
                        ui.set_min_size(vec2(240.0, 110.0));
                        egui::ScrollArea::vertical().id_salt("spell-sugg").max_height(110.0).show(ui, |ui| {
                            for s in item.as_ref().and_then(|i| i["suggestions"].as_array()).into_iter().flatten() {
                                let s = s.as_str().unwrap_or("");
                                let r = ui.selectable_label(s == change_to, s);
                                if r.clicked() {
                                    change_to = s.to_string();
                                }
                                if r.double_clicked() {
                                    change_to = s.to_string();
                                    act = Some("change");
                                }
                            }
                        });
                    });
                    ui.add_space(4.0);
                    if crate::widgets::checkbox(ui, &mut all_layers, tl!("Check All Layers")).changed() {
                        act = Some("relist");
                    }
                    let msg = if item.is_none() { tl!("Spell check complete.") } else { "" };
                    if !msg.is_empty() {
                        ui.label(RichText::new(msg).color(t.text).size(11.5));
                    }
                });
                ui.vertical(|ui| {
                    ui.spacing_mut().item_spacing.y = 4.0;
                    if crate::widgets::primary_button(ui, tl!("Done"), 100.0).clicked() {
                        act = Some("done");
                    }
                    ui.add_space(6.0);
                    ui.add_enabled_ui(item.is_some(), |ui| {
                        for (label, a) in [
                            (tl!("Ignore"), "ignore"),
                            (tl!("Ignore All"), "ignoreAll"),
                            (tl!("Change"), "change"),
                            (tl!("Change All"), "changeAll"),
                            (tl!("Add"), "add"),
                        ] {
                            if crate::widgets::secondary_button(ui, label, 100.0).clicked() {
                                act = Some(a);
                            }
                        }
                    });
                });
            });
        });
    if let Some(s) = app.ui.type_panels.spell.as_mut() {
        s.change_to = change_to.clone();
        s.all_layers = all_layers;
    }
    let Some(a) = act else { return };
    let word = item.as_ref().and_then(|i| i["word"].as_str()).unwrap_or("").to_string();
    match a {
        "done" => app.ui.type_panels.spell = None,
        "relist" => {
            if let Some(s) = app.ui.type_panels.spell.as_mut() {
                s.index = 0;
            }
            refresh_spelling(app);
        }
        "ignore" => {
            if let Some(s) = app.ui.type_panels.spell.as_mut() {
                s.index += 1;
                s.change_to = s.items.get(s.index).and_then(|i| i["suggestions"][0].as_str()).unwrap_or("").to_string();
            }
        }
        "ignoreAll" => {
            if let Some(s) = app.ui.type_panels.spell.as_mut() {
                s.ignore.push(word);
            }
            refresh_spelling(app);
        }
        "add" => {
            let _ = run(app, "edit.checkSpelling", json!({ "action": "addToDictionary", "word": word }));
            refresh_spelling(app);
        }
        "change" | "changeAll" => {
            if let Some(i) = &item {
                let p = if a == "change" {
                    json!({ "action": "change", "layer": i["layer"], "start": i["start"], "end": i["end"], "word": word, "replace": change_to })
                } else {
                    json!({ "action": "changeAll", "word": word, "replace": change_to, "allLayers": all_layers })
                };
                let _ = run(app, "edit.checkSpelling", p);
                refresh_spelling(app);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app_with_text(text: &str) -> PhotocraftApp {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 300, "height": 120, "background": "white"})).unwrap();
        app.run("type.create", json!({"x": 10, "y": 60, "text": text})).unwrap();
        app.sync_views();
        app
    }

    #[test]
    fn menu_toggles_panels() {
        let mut app = app_with_text("Hi");
        assert!(handles("window.panel.glyphs") && crate::menus::is_live("type.panels.paragraphStyles"));
        menu(&mut app, "type.panels.paragraphStyles", &json!({})).unwrap().unwrap();
        assert!(app.ui.type_panels.styles && app.ui.type_panels.styles_tab == 1);
        assert_eq!(checked(&app, "window.panel.paragraphStyles"), Some(true));
        assert_eq!(checked(&app, "window.panel.characterStyles"), Some(false));
        menu(&mut app, "window.panel.characterStyles", &json!({})).unwrap().unwrap();
        assert_eq!(app.ui.type_panels.styles_tab, 0);
        menu(&mut app, "window.panel.characterStyles", &json!({})).unwrap().unwrap();
        assert!(!app.ui.type_panels.styles);
        menu(&mut app, "window.panel.glyphs", &json!({"show": true})).unwrap().unwrap();
        assert!(app.ui.type_panels.glyphs);
    }

    #[test]
    fn glyph_insertion_and_recent() {
        let mut app = app_with_text("ab");
        assert!(insert_glyph(&mut app, "→"));
        assert!(insert_glyph(&mut app, "€"));
        assert!(insert_glyph(&mut app, "→"));
        assert_eq!(app.ui.type_panels.glyph_recent, vec!["→".to_string(), "€".to_string()]);
        let st = app.session.active().unwrap();
        let Some(photocraft_doc::LayerContent::Text(t)) = st.active_layer.and_then(|id| st.doc.layer(id)).map(|l| &l.content) else { panic!() };
        assert_eq!(t.text, "ab→€→");
    }

    #[test]
    fn spelling_dialog_flow() {
        let mut app = app_with_text("Teh cat");
        let r = menu(&mut app, "edit.checkSpelling", &json!({})).unwrap().unwrap();
        assert_eq!(r["misspellings"].as_array().unwrap().len(), 1);
        let d = app.ui.type_panels.spell.clone().unwrap();
        assert_eq!(d.change_to, "The");
        // Headless calls with an action go to the engine.
        assert!(menu(&mut app, "edit.checkSpelling", &json!({"action": "list"})).is_none());
    }
}
