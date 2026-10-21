//! Photoshop's Brush Preset picker: the popup that the options-bar brush chip and a right-click on
//! the canvas open (`paint_mouse`). It is the quick picker (Size, Hardness for round tips, a search
//! field, the gear menu and the preset library in its groups); the full editor is the Brush
//! Settings panel (F5), which the picker and the options bar open with one click (#258).
//!
//! The preset list is the Brushes panel's ([`brushes_tab::preset_list`]), always as cards: a
//! click picks a preset and keeps the picker open, a double-click picks it and closes the picker
//! (#1031); drag to reorder, right-click to rename or delete. Each card is a 2 × 2 grid — tip at
//! the top left, stroke preview at the top right and the name in the merged bottom row (or in the
//! stroke's cell when only the tip and the name show) — and the gear's three boxes ([`body`]'s
//! Brush Name/Brush Stroke/Brush Tip) hide its parts, at least one always on; the footer slider
//! scales the cards. Its view state (search, collapsed groups, the card's parts, the footer
//! scale, a rename in progress) is `UiState::brush_picker_list`, so the control channel can read
//! and set it. The gear's parts and the footer scale are remembered across restarts
//! ([`persist`]/[`restore`], `Preferences::brush_picker`); the search, collapsed groups and a
//! rename in progress are not.
//!
//! The picker edits a copy of the brush; the caller sends the size and hardness edits through
//! `tools.setBrush` ([`crate::brush_panel::commit_gesture`]) and applies the returned [`Pick`]s
//! with [`apply`], so every change is a journaled command (Rule 1).

use egui::{CursorIcon, RichText, Stroke, vec2};
use photocraft_engine::BrushSettings;
use photocraft_engine::paint::{BrushPreset, MAX_BRUSH_SIZE, TipShape};
use serde_json::{Value, json};

use crate::brush_panel::{BrushesPanelState, Renaming};
use crate::brushes_tab::{self, Action, ListLayout};
use crate::theme::Tokens;
use crate::{PhotocraftApp, icons, widgets};

/// Width of the picker's contents.
pub const WIDTH: f32 = 300.0;
/// The picker's content size before its grip is dragged, and the smallest it can be dragged to.
pub const DEFAULT_SIZE: [f32; 2] = [WIDTH, 420.0];
pub const MIN_SIZE: [f32; 2] = [230.0, 170.0];
/// The picker's preset list: denser than the Brushes tab's, and always the cards (its tip, stroke
/// and name cells, set by [`BrushesPanelState::show_name`] and friends).
pub(crate) const LIST: ListLayout = ListLayout { id: "brush-picker-presets", max_height: 300.0, cell: 44.0, indent: 4.0, cards: true };
/// The scale slider's height (the Size and Hardness sliders').
const SLIDER_H: f32 = 18.0;
/// The least gap between the preset list and the slider under it.
const FOOTER_GAP: f32 = 6.0;
/// The room the footer takes from the picker's bottom.
const FOOTER_H: f32 = SLIDER_H + FOOTER_GAP;
/// Room the footer leaves on its right for the resize grip ([`resize_grip`] is 16 wide): the
/// scale slider runs up to the grip's left edge, the two corner controls side by side.
const GRIP_ROOM: f32 = 20.0;
/// The block the Size and Hardness controls share, flush with the picker's left edge and at most
/// this wide: a picker dragged wider keeps the controls their size, the space past them empty.
pub(crate) const HEAD_MAX_W: f32 = 350.0;
/// The footer scale slider's range: what the cards' size scale runs between, and what a
/// remembered scale is clamped to (under the bottom the cards' padding and rows stop making
/// sense).
pub(crate) const SCALE_RANGE: std::ops::RangeInclusive<f32> = 0.15..=2.0;

/// What the picker asks for beyond the size and hardness edits.
#[derive(Clone, Debug, PartialEq)]
pub enum Pick {
    /// A preset-list action: pick, reorder, rename or delete a preset, as in the Brushes panel.
    List(Action),
    /// Open the Brush Settings panel (Window › Brush Settings, F5).
    OpenSettings,
    /// Save the current brush as a new preset, then name it in the rename bar.
    NewPreset,
    /// Import brushes (.abr) with the file dialog.
    Import,
}

impl Pick {
    /// Does this pick dismiss the picker? A double-click on a preset does (Photoshop), and so does
    /// anything that opens another panel or dialog.
    pub fn closes(&self) -> bool {
        matches!(self, Pick::List(Action::Choose(_)) | Pick::OpenSettings | Pick::Import)
    }
}

pub(crate) fn picker_id() -> egui::Id {
    egui::Id::new("canvas-brush-picker")
}

/// All opening paths protect the same frame, including the control channel and options bar.
pub(crate) fn open(ui: &mut crate::state::UiState, ctx: &egui::Context, at: [f32; 2]) {
    ui.brush_picker = Some(at);
    crate::canvas_popup::mark_open(ctx, picker_id());
}

/// The options-bar brush chip that shows and hides the picker. A press on it isn't an outside
/// press that closes the picker: the chip's click toggles it.
pub fn chip_id() -> egui::Id {
    egui::Id::new("brush-preset-chip")
}

/// Is screen point `p` on the options-bar brush chip?
pub fn on_chip(ctx: &egui::Context, p: egui::Pos2) -> bool {
    ctx.read_response(chip_id()).is_some_and(|r| r.rect.contains(p))
}

/// Close the picker. A rename left open in it goes too, like Photoshop's name dialog.
pub fn close(ui: &mut crate::state::UiState) {
    ui.brush_picker = None;
    ui.brush_picker_list.renaming = None;
}

/// The state the picker's preset list starts from: every card part on (tip, stroke and name).
pub fn list_state() -> BrushesPanelState {
    BrushesPanelState::default()
}

/// The picker's view remembered across restarts: the gear's three card parts and the footer
/// scale (`Preferences::brush_picker`).
fn view(app: &PhotocraftApp) -> Value {
    let st = &app.ui.brush_picker_list;
    json!({"showName": st.show_name, "showStroke": st.show_stroke, "showTip": st.show_tip, "scale": st.scale})
}

/// Remember the picker's view in the preferences once the user lets go of the pointer: the
/// gear's card parts and the footer scale. Cheap: a small JSON compare per frame. Run every
/// frame (not from the picker's own code) so a view set while the picker is closed — by the
/// control channel — is remembered too.
pub fn persist(app: &mut PhotocraftApp, ctx: &egui::Context) {
    if ctx.input(|i| i.pointer.any_down()) {
        return;
    }
    let now = view(app);
    if app.session.prefs().brush_picker != now {
        app.session.prefs.edit(|p| p.brush_picker = now);
    }
}

/// Restore the picker's remembered view at launch. A missing or malformed part keeps its
/// default, the scale is clamped to the slider's range, and the three parts can't all come
/// back off (an empty card would show no brush at all).
pub fn restore(app: &mut PhotocraftApp) {
    let saved = app.session.prefs().brush_picker.clone();
    let st = &mut app.ui.brush_picker_list;
    if let Some(v) = saved.get("showName").and_then(Value::as_bool) {
        st.show_name = v;
    }
    if let Some(v) = saved.get("showStroke").and_then(Value::as_bool) {
        st.show_stroke = v;
    }
    if let Some(v) = saved.get("showTip").and_then(Value::as_bool) {
        st.show_tip = v;
    }
    if let Some(v) = saved.get("scale").and_then(Value::as_f64)
        && v.is_finite()
    {
        st.scale = (v as f32).clamp(*SCALE_RANGE.start(), *SCALE_RANGE.end());
    }
    if ![st.show_name, st.show_stroke, st.show_tip].into_iter().any(|b| b) {
        (st.show_name, st.show_stroke, st.show_tip) = (true, true, true);
    }
}

/// Run what the picker asked for.
pub fn apply(app: &mut PhotocraftApp, ctx: &egui::Context, picks: Vec<Pick>) {
    for pick in picks {
        match pick {
            Pick::List(a) => brushes_tab::apply(app, vec![a]),
            Pick::OpenSettings => open_settings(app, ctx),
            // Photoshop asks for the new preset's name and what it captures first, in a dialog that
            // takes the picker's place.
            Pick::NewPreset => {
                close(&mut app.ui);
                crate::brush_preset_dialog::open(app, None);
            }
            Pick::Import => {
                let _ = app.open_dialog_file();
            }
        }
    }
}

/// Show the Brush Settings panel on its settings tab (what F5 does when it is hidden).
pub fn open_settings(app: &mut PhotocraftApp, ctx: &egui::Context) {
    if app.ui.panels.brush_settings {
        app.ui.brush_tab = 0;
        return;
    }
    if let Err(e) = crate::menus::invoke(app, ctx, "window.panel.brushSettings", json!({})) {
        app.ui.status = e;
    }
}

/// The options-bar button beside the brush chip that shows and hides the Brush Settings panel
/// (Photoshop's "Toggle the Brush Settings panel"). It runs Window › Brush Settings (F5).
pub fn settings_toggle(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let on = app.ui.panels.brush_settings;
    let tooltip = "Toggle the Brush Settings panel  (F5)";
    if named(icons::button(ui, "sliders-horizontal", 24.0, on, tl!(tooltip)), "Toggle the Brush Settings panel").clicked()
        && let Err(e) = crate::menus::invoke(app, ui.ctx(), "window.panel.brushSettings", json!({}))
    {
        app.ui.status = e;
    }
}

/// An icon button's accessible name (icon buttons have only a tooltip).
pub(crate) fn named(resp: egui::Response, label: &str) -> egui::Response {
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, tl!(label)));
    resp
}

/// The picker's contents: Size, Hardness (round tips only: a sampled tip has none), a search
/// field with the Brush Settings, New Preset and gear buttons, the preset list, and the footer
/// slider that scales the cards ([`scale_slider`]). Size and Hardness sit in a block flush with
/// the left edge, at most [`HEAD_MAX_W`] wide, so a picker dragged wider doesn't stretch them.
/// `current` is the selected preset's name (by identity, so edits don't deselect). `size` is the
/// content size the user dragged the picker to ([`DEFAULT_SIZE`] before that): the list fills
/// it, so closing every group doesn't shrink the picker and reopening one has the room to show
/// it.
pub fn body(
    ui: &mut egui::Ui,
    b: &mut BrushSettings,
    presets: &[BrushPreset],
    current: Option<&str>,
    st: &mut BrushesPanelState,
    size: egui::Vec2,
) -> Vec<Pick> {
    let t = Tokens::get(ui.ctx());
    let mut picks = Vec::new();
    // The picker is its dragged (or remembered) size: at least that tall even when every group
    // is closed, so reopening one has the room to show it.
    ui.set_width(size.x);
    ui.set_min_height(size.y);
    let top = ui.cursor().min.y;
    // The Size and Hardness controls live in one block flush with the left edge, at most
    // [`HEAD_MAX_W`] wide: a picker dragged wider leaves the space past the block empty instead of
    // stretching the rows across it.
    ui.scope(|ui| {
        ui.set_max_width(ui.available_width().min(HEAD_MAX_W));
        // Size: value field plus a logarithmic slider (small sizes get most of the travel).
        ui.horizontal(|ui| {
            ui.label(RichText::new(tl!("Size")).color(t.text_dim));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let mut size = b.size;
                if widgets::value_field(ui, &mut size, 1.0..=MAX_BRUSH_SIZE, "px", 72.0).changed() {
                    b.size = size.round().clamp(1.0, MAX_BRUSH_SIZE);
                }
            });
        });
        let mut lv = b.size.max(1.0).ln();
        let size_slider = widgets::slider(ui, &mut lv, 0.0..=MAX_BRUSH_SIZE.ln(), None);
        if size_slider.changed() {
            b.size = lv.exp().round().clamp(1.0, MAX_BRUSH_SIZE);
        }
        dismiss_popup(&size_slider);
        if b.tip == TipShape::Round {
            let mut hard = b.hardness * 100.0;
            let hardness = widgets::slider_row(ui, "Hardness", &mut hard, 0.0..=100.0, "%", None);
            if hardness.changed() {
                b.hardness = (hard / 100.0).clamp(0.0, 1.0);
            }
            dismiss_popup(&hardness);
        }
    });
    ui.add_space(6.0);
    widgets::hairline(ui);
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        icons::paint(ui, egui::Rect::from_min_size(ui.cursor().min + vec2(0.0, 3.0), vec2(16.0, 16.0)), "search", 14.0, t.text_faint);
        ui.add_space(20.0);
        widgets::text_field(ui, ui.id().with("brush-picker-search"), &mut st.filter, tl!("Search Brushes"), (size.x - 118.0).max(60.0), false);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            gear_menu(ui, current, presets, st, &mut picks);
            let new_tip = "Create new brush preset from the current settings";
            let settings_tip = "Brush Settings…  (F5): every option of this brush";
            if named(icons::button(ui, "square-plus", 24.0, false, tl!(new_tip)), "New Brush Preset").clicked() {
                picks.push(Pick::NewPreset);
            }
            if named(icons::button(ui, "sliders-horizontal", 24.0, false, tl!(settings_tip)), "Brush Settings…").clicked() {
                picks.push(Pick::OpenSettings);
            }
        });
    });
    ui.add_space(4.0);
    // The list takes the picker's height past the controls above it and the footer below.
    let mut layout = LIST;
    layout.max_height = (size.y - (ui.cursor().min.y - top) - FOOTER_H).max(80.0);
    picks.extend(brushes_tab::preset_list(ui, presets, current, st, layout).into_iter().map(Pick::List));
    // The footer lives in the picker's bottom band whatever the list left over, flush with the
    // bottom and the slider clear of the resize grip's corner.
    ui.add_space((size.y - (ui.cursor().min.y - top) - SLIDER_H).max(FOOTER_GAP));
    scale_slider(ui, st);
    picks
}

/// The footer's card-scale slider: how big the preset cards are drawn (1 the standard, 2 the
/// largest), bare like the Size and Hardness sliders — a readout would be noise for a control
/// that's set by feel. The value snaps to hundredths and runs over [`SCALE_RANGE`], from 0.15,
/// under which the cards' padding and rows stop making sense; 0.40 and below is where
/// [`brushes_tab`]'s cards go compact, the tips dropping their size numbers (the text itself
/// holds its size at every scale).
fn scale_slider(ui: &mut egui::Ui, st: &mut BrushesPanelState) {
    let w = (ui.available_width() - GRIP_ROOM).max(60.0);
    ui.allocate_ui_with_layout(vec2(w, SLIDER_H), egui::Layout::left_to_right(egui::Align::Center), |ui| {
        let resp = widgets::slider(ui, &mut st.scale, SCALE_RANGE, None);
        if resp.changed() {
            st.scale = (st.scale * 100.0).round() / 100.0;
        }
        dismiss_popup(&resp);
    });
}

/// Closes any open popup (the gear menu) as soon as `resp` is grabbed: a slider or grip drag never
/// counts as a click, so the menu's [`egui::PopupCloseBehavior::CloseOnClickOutside`] alone leaves
/// it up — and the picker's own interaction pulls its layer in front, sinking the menu behind it.
fn dismiss_popup(resp: &egui::Response) {
    if resp.is_pointer_button_down_on() {
        egui::Popup::close_all(&resp.ctx);
    }
}

/// A remembered picker size, clamped for `screen` (a corrupt or wildly stale value falls back to
/// [`DEFAULT_SIZE`]).
pub fn clamp_size(size: [f32; 2], screen: egui::Vec2) -> egui::Vec2 {
    if !size[0].is_finite() || !size[1].is_finite() {
        return egui::vec2(DEFAULT_SIZE[0], DEFAULT_SIZE[1]);
    }
    let min = egui::vec2(MIN_SIZE[0], MIN_SIZE[1]);
    let max = (screen - vec2(40.0, 40.0)).max(min);
    egui::vec2(size[0], size[1]).clamp(min, max)
}

/// The picker's resize grip, in its frame's bottom-right corner: three ticks, dragging gives the
/// new content size (clamped to [`MIN_SIZE`] and the screen). Returns the size while dragged. The
/// corner tracks the pointer's travel since the drag began, not each frame's movement: held past
/// a limit (the minimum, the screen), the overshoot isn't banked, so coming back the picker grows
/// only once the pointer reaches the corner again — it waits for the pointer.
pub fn resize_grip(ui: &mut egui::Ui, frame: egui::Rect, content: egui::Vec2) -> Option<egui::Vec2> {
    let t = Tokens::get(ui.ctx());
    let grip = egui::Rect::from_min_size(frame.right_bottom() - vec2(16.0, 16.0), vec2(16.0, 16.0));
    let resp = ui.interact(grip, ui.id().with("brush-picker-resize"), egui::Sense::drag());
    dismiss_popup(&resp);
    let corner = grip.right_bottom() - vec2(3.0, 3.0);
    for i in 0..3 {
        let o = i as f32 * 4.0;
        ui.painter().line_segment([corner - vec2(o, 0.0), corner - vec2(0.0, o)], Stroke::new(1.0, t.text_faint));
    }
    if resp.hovered() || resp.dragged() {
        ui.ctx().set_cursor_icon(CursorIcon::ResizeNwSe);
    }
    // The size this drag started from, kept while the button is held.
    let start_id = resp.id.with("start");
    if resp.drag_started() {
        ui.ctx().data_mut(|d| d.insert_temp(start_id, content));
    }
    resp.dragged().then(|| {
        let min = egui::vec2(MIN_SIZE[0], MIN_SIZE[1]);
        let max = (ui.ctx().content_rect().size() - vec2(40.0, 40.0)).max(min);
        let start: egui::Vec2 = ui.ctx().data_mut(|d| d.get_temp(start_id)).unwrap_or(content);
        (start + resp.total_drag_delta().unwrap_or_default()).clamp(min, max)
    })
}

/// The gear: Photoshop's picker menu. A new preset, rename or delete the current one, the list's
/// view, and Import Brushes.
fn gear_menu(ui: &mut egui::Ui, current: Option<&str>, presets: &[BrushPreset], st: &mut BrushesPanelState, picks: &mut Vec<Pick>) {
    let gear = named(icons::button(ui, "settings", 24.0, false, tl!("Brush Preset Options")), "Brush Preset Options");
    // The menu hangs 30 pt right of the gear, off the button's own edge; the align fallbacks still
    // keep it on screen near the window's edge.
    let anchor = match egui::PopupAnchor::from(&gear) {
        egui::PopupAnchor::ParentRect(rect) => egui::PopupAnchor::ParentRect(rect.translate(vec2(30.0, 0.0))),
        other => other,
    };
    egui::Popup::menu(&gear).anchor(anchor).close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside).show(|ui| {
        if ui.button(tl!("New Brush Preset…")).clicked() {
            picks.push(Pick::NewPreset);
            ui.close();
        }
        ui.separator();
        let current = current.filter(|n| presets.iter().any(|p| p.name.eq_ignore_ascii_case(n))).map(str::to_string);
        ui.add_enabled_ui(current.is_some(), |ui| {
            if ui.button(tl!("Rename Brush…")).clicked()
                && let Some(name) = current.clone()
            {
                st.renaming = Some(Renaming { group: false, name, folder: Vec::new(), text: String::new() });
                ui.close();
            }
            if ui.button(tl!("Delete Brush")).clicked()
                && let Some(name) = current.clone()
            {
                picks.push(Pick::List(Action::Delete(name)));
                ui.close();
            }
        });
        ui.separator();
        // What a preset's card shows: a checkbox per part. At least one stays on — clicking the
        // last checked box does nothing, since an empty card would show no brush at all.
        let on = [st.show_name, st.show_stroke, st.show_tip].into_iter().filter(|b| *b).count();
        for (flag, label) in [(&mut st.show_name, tl!("Brush Name")), (&mut st.show_stroke, tl!("Brush Stroke")), (&mut st.show_tip, tl!("Brush Tip"))] {
            let was = *flag;
            if ui.checkbox(flag, label).changed() && was && on == 1 {
                *flag = true;
            }
        }
        ui.separator();
        if ui.button(tl!("Import Brushes…")).clicked() {
            picks.push(Pick::Import);
            ui.close();
        }
    });
}
