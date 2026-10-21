//! Custom widgets for the Photocraft look: cards with pill tabs, thin sliders with round knobs,
//! monospace value fields with dimmed units, toggle switches, primary/secondary buttons.

use egui::{Align2, Color32, CornerRadius, Pos2, Rect, Response, Sense, Stroke, StrokeKind, Ui, Vec2, pos2, vec2};

use crate::theme::{self, Tokens};

mod angle_dial;
mod color_count;
pub use angle_dial::angle_dial_row;
pub use color_count::color_count_row;
mod type_ahead;
pub use type_ahead::list_open as dropdown_list_open;

/// Match the Spectrum application-menu padding and blue/white hover state in
/// secondary menus (status bar, dock hamburger, layer context menus; #2187).
/// The setting is local to this popup's Ui and leaves other themes unchanged.
pub fn style_spectrum_popup_menu(ui: &mut Ui) {
    let t = Tokens::get(ui.ctx());
    // The 6 pt gap between rows was outside both buttons, so a drag across a menu
    // dropped the highlight (#4602). Padding stays inside the row, which is the hit target.
    ui.spacing_mut().item_spacing.y = 0.0;
    if !t.pro {
        return;
    }
    ui.spacing_mut().button_padding = vec2(10.0, 4.0);
    let v = &mut ui.style_mut().visuals;
    v.widgets.hovered.weak_bg_fill = t.accent;
    v.widgets.hovered.bg_fill = t.accent;
    v.widgets.hovered.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    v.widgets.hovered.bg_stroke = Stroke::NONE;
    v.widgets.inactive.corner_radius = CornerRadius::ZERO;
    v.widgets.hovered.corner_radius = CornerRadius::ZERO;
    v.widgets.active.corner_radius = CornerRadius::ZERO;
}

/// An accent insertion line on one edge of `r` while a drag hovers it (vertical: on its left or,
/// `after`, right edge; else on its top or bottom).
pub fn drop_line(ui: &Ui, r: Rect, after: bool, vertical: bool, t: &Tokens) {
    let s = Stroke::new(2.0, t.accent);
    if vertical {
        let x = if after { r.right() } else { r.left() };
        ui.painter().line_segment([pos2(x, r.top() + 2.0), pos2(x, r.bottom() - 2.0)], s);
    } else {
        let y = if after { r.bottom() } else { r.top() };
        ui.painter().line_segment([pos2(r.left() + 4.0, y), pos2(r.right() - 4.0, y)], s);
    }
}

/// Draw a bevelled box (Classic theme) or a flat rounded box.
pub fn surface(ui: &Ui, rect: Rect, fill: Color32, raised: bool) {
    let t = Tokens::get(ui.ctx());
    let p = ui.painter();
    p.rect_filled(rect, t.radius_sm, fill);
    if t.bevel {
        let (hi, lo) = if raised { (Color32::WHITE, Color32::from_gray(64)) } else { (Color32::from_gray(64), Color32::WHITE) };
        p.line_segment([rect.left_bottom(), rect.left_top()], Stroke::new(1.0, hi));
        p.line_segment([rect.left_top(), rect.right_top()], Stroke::new(1.0, hi));
        p.line_segment([rect.right_top(), rect.right_bottom()], Stroke::new(1.0, lo));
        p.line_segment([rect.right_bottom(), rect.left_bottom()], Stroke::new(1.0, lo));
    }
}

/// A dock card: rounded container with a header of pill tabs and optional trailing actions.
/// Returns the index of the selected tab.
pub fn card(ui: &mut Ui, id: &str, tabs: &[&str], selected: &mut usize, body: impl FnOnce(&mut Ui, usize)) {
    let _ = card_ex(ui, id, tabs, selected, false, false, body);
}

/// What happened on a card's tab strip this frame (see [`card_ex`]).
pub struct CardResponse {
    /// The tab strip background: drag to move the group, double-click to collapse it.
    pub strip: Response,
    /// The panel menu button (hamburger in Pro, ellipsis in Studio).
    pub menu: Response,
    /// A tab was clicked or chosen from the overflow menu.
    pub tab_clicked: bool,
    /// A tab was double-clicked (Photoshop collapses the group).
    pub tab_double_clicked: bool,
    /// Right-click action selected for a visible tab.
    pub tab_context: Option<crate::tab_strip::TabContextAction>,
    /// Rects of the tabs on the strip, `(tab index, rect)`; tabs that don't fit are in the
    /// chevron menu instead (#151).
    pub tabs: Vec<(usize, Rect)>,
    pub responses: Vec<(usize, Response)>,
    /// The » overflow button, when some tabs didn't fit.
    pub chevron: Option<Rect>,
    /// A tab dragged along the strip was dropped: `(tab, before)` (#2272).
    pub tab_reorder: Option<(usize, Option<usize>)>,
    /// The index of a tab being dragged away from the strip.
    pub tab_dragging_out: Option<usize>,
    /// The index of a tab whose drag ended away from the strip.
    pub tab_dropped_out: Option<usize>,
}

impl CardResponse {
    fn new(strip: Response, menu: Response, tabs: crate::tab_strip::StripOut) -> Self {
        CardResponse {
            strip,
            menu,
            tab_clicked: tabs.clicked,
            tab_double_clicked: tabs.double_clicked,
            tab_context: tabs.context,
            tabs: tabs.tabs,
            responses: tabs.responses,
            chevron: tabs.chevron,
            tab_reorder: tabs.reorder,
            tab_dragging_out: tabs.dragging_out,
            tab_dropped_out: tabs.dropped_out,
        }
    }
}

/// [`card`] that can be collapsed to its tab strip and reports strip and menu interactions.
/// With `reorderable`, tabs can be dragged along the strip (reported, not applied).
#[allow(clippy::too_many_arguments)]
pub fn card_ex(
    ui: &mut Ui,
    id: &str,
    tabs: &[&str],
    selected: &mut usize,
    collapsed: bool,
    reorderable: bool,
    body: impl FnOnce(&mut Ui, usize),
) -> CardResponse {
    let t = Tokens::get(ui.ctx());
    if t.pro {
        return pro_panel(ui, id, tabs, selected, collapsed, reorderable, body);
    }
    let m = body_margin(false);
    let frame = egui::Frame::NONE
        .fill(t.card)
        .stroke(Stroke::new(1.0, t.card_border))
        .corner_radius(CornerRadius::same(t.radius as u8))
        .inner_margin(egui::Margin { bottom: if collapsed { 6 } else { m.bottom }, ..m });
    let out = frame
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            // Registered before the tabs so they keep their clicks; drags fall through to it.
            let strip_rect = Rect::from_min_size(ui.cursor().min, vec2(ui.available_width(), 22.0));
            let strip = ui.interact(strip_rect, ui.id().with((id, "strip")), Sense::click_and_drag());
            // Pill tabs left of the menu button; they elide or overflow into a chevron (#151).
            let (row, _) = ui.allocate_exact_size(vec2(ui.available_width(), 24.0), Sense::hover());
            let tip = crate::i18n::fmt(tl!("{name} options"), &[("name", tl!(tabs.get(*selected).copied().unwrap_or(id)))]);
            let menu_rect = Rect::from_min_max(pos2(row.right() - 22.0, row.top() + 1.0), pos2(row.right(), row.bottom() - 1.0));
            let menu = crate::icons::button(&mut ui.new_child(egui::UiBuilder::new().max_rect(menu_rect)), "ellipsis", 22.0, false, &tip);
            let area = Rect::from_min_max(row.min, pos2((menu_rect.left() - 4.0).max(row.left()), row.bottom()));
            let tabs_out = crate::tab_strip::pill_tabs(ui, ui.id().with((id, "tabs")), area, tabs, selected, reorderable);
            if !collapsed {
                ui.add_space(6.0);
                body(ui, *selected);
            }
            CardResponse::new(strip, menu, tabs_out)
        })
        .inner;
    ui.add_space(6.0);
    out
}

/// Inner margin of a dock panel's body: a Pro panel's, or a Studio card's below its tab strip.
fn body_margin(pro: bool) -> egui::Margin {
    if pro { egui::Margin::same(8) } else { egui::Margin { left: 8, right: 8, top: 6, bottom: 10 } }
}

/// Height of a panel footer's bar ([`panel_footer`]), under its 1 px line.
pub const FOOTER_BAR: f32 = 30.0;

/// The height a [`panel_footer`] takes from the panel body: its line, and its bar down to the
/// body's bottom margin (the bar runs on over the margin).
pub fn footer_height(ui: &Ui) -> f32 {
    1.0 + FOOTER_BAR - body_margin(Tokens::get(ui.ctx()).pro).bottom as f32
}

/// A panel footer as in Photoshop's Layers and History panels: a bar of buttons under a line that
/// runs from edge to edge of the panel, the buttons centred in the bar's height and laid out from
/// the right. Drawn last in a body that fills its group (leave it [`footer_height`]), the bar ends
/// at the panel's bottom edge.
pub fn panel_footer<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> R {
    let t = Tokens::get(ui.ctx());
    let m = body_margin(t.pro);
    let (r, _) = ui.allocate_exact_size(vec2(ui.available_width(), footer_height(ui)), Sense::hover());
    let y = r.top() + 0.5;
    ui.painter().line_segment([pos2(r.left() - m.left as f32, y), pos2(r.right() + m.right as f32, y)], Stroke::new(1.0, t.separator));
    let bar = Rect::from_min_size(pos2(r.left(), r.top() + 1.0), vec2(r.width(), FOOTER_BAR));
    let mut bar_ui = ui.new_child(egui::UiBuilder::new().max_rect(bar).layout(egui::Layout::right_to_left(egui::Align::Center)));
    bar_ui.spacing_mut().item_spacing.x = 2.0;
    add(&mut bar_ui)
}

/// Photoshop-grammar panel group: dark tab strip with flat tabs, flat body, hamburger menu.
fn pro_panel(
    ui: &mut Ui,
    id: &str,
    tabs: &[&str],
    selected: &mut usize,
    collapsed: bool,
    reorderable: bool,
    body: impl FnOnce(&mut Ui, usize),
) -> CardResponse {
    let t = Tokens::get(ui.ctx());
    let width = ui.available_width();
    // Tab strip. Its background senses drags (move the group) and double-clicks (collapse);
    // the tabs, registered after it, keep their clicks.
    let (strip, _) = ui.allocate_exact_size(vec2(width, 26.0), Sense::hover());
    let strip_resp = ui.interact(strip, ui.id().with((id, "strip")), Sense::click_and_drag());
    let rounding = if collapsed { CornerRadius::same(3) } else { CornerRadius { nw: 3, ne: 3, sw: 0, se: 0 } };
    ui.painter().rect_filled(strip, rounding, t.tab_strip);
    // Panel menu (hamburger); the tabs stay left of it, eliding or overflowing (#151).
    let menu = Rect::from_center_size(pos2(strip.right() - 14.0, strip.center().y), vec2(20.0, 18.0));
    let tabs_out = crate::tab_strip::pro_tabs(ui, ui.id().with((id, "tabs")), strip, menu.left(), tabs, selected, collapsed, reorderable);
    let mresp = ui.interact(menu, ui.id().with((id, "menu")), Sense::click());
    let c = if mresp.hovered() { t.text } else { t.text_faint };
    for k in 0..3 {
        let y = menu.center().y - 3.5 + k as f32 * 3.5;
        ui.painter().line_segment([pos2(menu.center().x - 5.0, y), pos2(menu.center().x + 5.0, y)], Stroke::new(1.0, c));
    }
    // Body.
    if !collapsed {
        egui::Frame::NONE.fill(t.card).corner_radius(CornerRadius { nw: 0, ne: 0, sw: 3, se: 3 }).inner_margin(body_margin(true)).show(ui, |ui| {
            ui.set_width(width - 16.0);
            body(ui, *selected);
        });
    }
    ui.add_space(2.0);
    CardResponse::new(strip_resp, mresp, tabs_out)
}

/// The panel menu button (hamburger in Pro, ellipsis in Studio), matching the dock and cards.
pub fn panel_menu_button(ui: &mut Ui) -> Response {
    let t = Tokens::get(ui.ctx());
    let size = vec2(20.0, 20.0);
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    let tint = crate::icons::button_chrome(ui, rect, false, resp.hovered());
    if t.pro {
        for k in 0..3 {
            let y = rect.center().y - 3.5 + k as f32 * 3.5;
            ui.painter().line_segment([pos2(rect.center().x - 5.0, y), pos2(rect.center().x + 5.0, y)], Stroke::new(1.0, tint));
        }
    } else {
        crate::icons::paint(ui, rect, "ellipsis", 14.0, tint);
    }
    let tip = tl!("Panel menu");
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, tip));
    resp.on_hover_text(tip)
}

pub fn pill_tab(ui: &mut Ui, label: &str, selected: bool) -> Response {
    let t = Tokens::get(ui.ctx());
    let font = theme::medium(12.5);
    let galley = ui.painter().layout_no_wrap(tl!(label).to_owned(), font, t.text);
    let size = vec2(galley.size().x + 20.0, 24.0);
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    if selected {
        surface(ui, rect, t.hover, true);
        if !t.bevel {
            ui.painter().rect_stroke(rect, t.radius_sm, Stroke::new(1.0, t.field_border), StrokeKind::Inside);
        }
    } else if resp.hovered() {
        ui.painter().rect_filled(rect, t.radius_sm, t.hover.gamma_multiply(0.6));
    }
    let color = if selected { t.text } else { t.text_dim };
    ui.painter().galley_with_override_text_color(rect.center() - galley.size() / 2.0, galley, color);
    resp
}

/// Monospace numeric field with a dimmed unit suffix, Photoshop style. Drag to scrub. Typed
/// arithmetic (`1920/2`) applies on Enter, Tab or click-away, and a unit after a number converts
/// into the field's unit (`1in` in a px field, at the resolution [`set_field_resolution`] gave).
pub fn value_field(ui: &mut Ui, value: &mut f32, range: std::ops::RangeInclusive<f32>, suffix: &str, width: f32) -> Response {
    value_field_in(ui, value, range, suffix, width, 0.0, FieldOpts::default()).0
}

/// A single-line label for a row that also holds fixed-width controls: laid out no wider than
/// `max_w`, cut with '…' when the text (often a longer translation) does not fit, with the full
/// string on hover. Fixed chrome must never let a label overrun its row (#3580), so an *essential*
/// label beside a fixed control should use this rather than a show/hide fit check; a *redundant*
/// label (one whose control already carries the meaning, e.g. an icon row's caption) may be dropped.
pub fn elided_label_sensed(ui: &mut Ui, text: &str, max_w: f32, sense: Sense) -> Response {
    let t = Tokens::get(ui.ctx());
    let font = egui::TextStyle::Body.resolve(ui.style());
    let galley = crate::tab_strip::elided(ui, text, font, t.text_dim, max_w);
    let elided = galley.elided;
    let (rect, resp) = ui.allocate_exact_size(galley.size(), sense);
    ui.painter().galley(rect.min, galley, t.text_dim);
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Label, ui.is_enabled(), text));
    if elided { resp.on_hover_text(text) } else { resp }
}

/// [`elided_label_sensed`] with hover only (a plain, non-interactive label).
pub fn elided_label(ui: &mut Ui, text: &str, max_w: f32) -> Response {
    elided_label_sensed(ui, text, max_w, Sense::hover())
}

/// Height of a [`value_field`]'s box.
pub const VALUE_FIELD_H: f32 = 24.0;

/// A [`value_field`] `height` points tall instead of [`VALUE_FIELD_H`], for slim bars such as
/// the status bar (#2823).
pub fn compact_value_field(ui: &mut Ui, value: &mut f32, range: std::ops::RangeInclusive<f32>, suffix: &str, width: f32, height: f32) -> Response {
    value_field_box(ui, value, range, suffix, vec2(width, height), 0.0, FieldOpts::default()).0
}

/// A compact field whose mouse scrub wraps at the native fullscreen viewport's horizontal edges.
pub(crate) fn compact_wrapping_value_field(ui: &mut Ui, value: &mut f32, range: std::ops::RangeInclusive<f32>, suffix: &str, size: Vec2) -> Response {
    value_field_box(ui, value, range, suffix, size, 0.0, FieldOpts { wrap_drag: true, ..Default::default() }).0
}

/// A size dialog's unit key (`px`, `in`, `cm`, `mm`, `pt`, `pica`, `percent`) as a unit.
pub fn unit_of_key(key: &str) -> photocraft_engine::prefs::Unit {
    use photocraft_engine::prefs::Unit;
    match key {
        "in" => Unit::Inches,
        "cm" => Unit::Centimeters,
        "mm" => Unit::Millimeters,
        "pt" => Unit::Points,
        "pica" => Unit::Picas,
        "percent" => Unit::Percent,
        _ => Unit::Pixels,
    }
}

/// Decimal places for a size dialog's unit key (`px`, `in`, `cm`, `mm`, `pt`, `pica`, `percent`).
/// The places themselves live on [`photocraft_engine::prefs::Unit`], so the dialogs, the Info
/// panel and the rulers agree (#2434).
pub fn unit_decimals(key: &str) -> u32 {
    unit_of_key(key).decimals() as u32
}

/// A [`value_field`] showing `decimals` decimal places instead of the one (or two, for a small
/// range) it picks itself: the ruler units need their own precision (#2434).
pub fn value_field_prec(ui: &mut Ui, value: &mut f32, range: std::ops::RangeInclusive<f32>, suffix: &str, width: f32, decimals: u32) -> Response {
    value_field_in(ui, value, range, suffix, width, 0.0, FieldOpts { decimals: Some(decimals), ..Default::default() }).0
}

/// A field's unit and the resolution a typed unit converts at (`1in` in a px field).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldUnits {
    pub unit: photocraft_engine::prefs::Unit,
    /// Pixels per inch.
    pub dpi: f64,
    /// Points per inch (72 PostScript, 72.27 traditional).
    pub ppi: f64,
}

/// A [`value_field_prec`] without a suffix, whose unit is chosen beside it (Image Size and New
/// Document's unit menus) and converts typed units at that dialog's own resolution.
pub fn value_field_units(ui: &mut Ui, value: &mut f32, range: std::ops::RangeInclusive<f32>, width: f32, decimals: u32, units: FieldUnits) -> Response {
    value_field_in(ui, value, range, "", width, 0.0, FieldOpts { decimals: Some(decimals), units: Some(units), ..Default::default() }).0
}

fn field_resolution_id() -> egui::Id {
    egui::Id::new("pc-field-resolution")
}

/// The resolution (pixels per inch) and point size (points per inch) a unit typed into a value
/// field converts at: the active document's. The app sets it each frame; 72 / 72 without it.
pub fn set_field_resolution(ctx: &egui::Context, dpi: f64, ppi: f64) {
    ctx.data_mut(|d| d.insert_temp(field_resolution_id(), (dpi, ppi)));
}

/// The unit a field's suffix names (`px`, `in`, `cm`, `mm`, `pt`, `pica`/`pc`, `%`), if any.
fn unit_of_suffix(suffix: &str) -> Option<photocraft_engine::prefs::Unit> {
    use photocraft_engine::prefs::Unit;
    Some(match suffix {
        "px" => Unit::Pixels,
        "in" => Unit::Inches,
        "cm" => Unit::Centimeters,
        "mm" => Unit::Millimeters,
        "pt" => Unit::Points,
        "pica" | "pc" => Unit::Picas,
        "%" => Unit::Percent,
        _ => return None,
    })
}

/// A drag on a value field's label this frame, for the value field drawn next on the label's row
/// (Photoshop's scrubby labels). Left by [`scrub_label`], taken by that field.
#[derive(Clone, Copy, Debug)]
struct LabelScrub {
    pass: u64,
    label: egui::Id,
    rect: Rect,
    /// How far right of the label the field may start (the whole row for a slider row).
    max_gap: f32,
    /// Whole steps to move the value (left negative; Shift ×10, Alt a tenth as fast).
    steps: f32,
    started: bool,
    dragging: bool,
    stopped: bool,
}

fn label_scrub_id() -> egui::Id {
    egui::Id::new("pc-label-scrub")
}

/// Makes `label` scrubby: dragging it left or right moves the value field drawn right after it on
/// the same row (one step a point, ×10 with Shift, a tenth as fast with Alt), with Photoshop's
/// scrubby cursor. A label with no value field right after it stays a plain label.
pub fn scrub_label(ui: &Ui, label: &Response) -> Response {
    scrub_label_within(ui, label, 24.0)
}

fn scrub_label_within(ui: &Ui, label: &Response, max_gap: f32) -> Response {
    // Drag only, not focusable: Tab goes to the value field, not to its label as well.
    let resp = label.interact(Sense::DRAG);
    // Whether a field took this label's drags last frame: only then is it scrubby.
    let linked = ui.data(|d| d.get_temp::<bool>(resp.id.with("pc-scrub-linked"))).unwrap_or(false);
    if linked && resp.enabled() && (resp.hovered() || resp.dragged()) {
        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
    }
    let acc_id = resp.id.with("pc-scrub-acc");
    let mut steps = 0.0;
    if resp.dragged() || resp.drag_stopped() {
        let mut acc: f32 = if resp.drag_started() { 0.0 } else { ui.data(|d| d.get_temp(acc_id)).unwrap_or(0.0) };
        let mods = ui.input(|i| i.modifiers);
        let rate = if mods.shift {
            10.0
        } else if mods.alt {
            0.1
        } else {
            1.0
        };
        // The drag starts once the pointer has moved a few points: count those too.
        let dx = match (resp.drag_started(), ui.input(|i| i.pointer.press_origin()), resp.interact_pointer_pos()) {
            (true, Some(origin), Some(pos)) => pos.x - origin.x,
            _ => resp.drag_delta().x,
        };
        acc += dx * rate;
        if !acc.is_finite() {
            acc = 0.0;
        }
        steps = acc.trunc();
        ui.data_mut(|d| d.insert_temp(acc_id, acc - steps));
    }
    let scrub = LabelScrub {
        pass: ui.ctx().cumulative_pass_nr(),
        label: resp.id,
        rect: resp.rect,
        max_gap,
        steps,
        started: resp.drag_started(),
        dragging: resp.dragged(),
        stopped: resp.drag_stopped(),
    };
    ui.data_mut(|d| {
        d.insert_temp(label_scrub_id(), scrub);
        d.insert_temp(resp.id.with("pc-scrub-linked"), false);
    });
    resp
}

/// The label drag left for the field at `rect` this frame, if its label is just left of it on
/// its row. Taken either way: a label only scrubs the next field drawn.
fn take_label_scrub(ui: &Ui, rect: Rect) -> Option<LabelScrub> {
    let s = ui.data_mut(|d| {
        let s = d.get_temp::<LabelScrub>(label_scrub_id());
        d.remove::<LabelScrub>(label_scrub_id());
        s
    })?;
    let same_row = rect.top() < s.rect.bottom() && rect.bottom() > s.rect.top();
    let gap = rect.left() - s.rect.right();
    let linked = s.pass == ui.ctx().cumulative_pass_nr() && same_row && gap >= -1.0 && gap <= s.max_gap;
    if linked {
        ui.data_mut(|d| d.insert_temp(s.label.with("pc-scrub-linked"), true));
    }
    linked.then_some(s)
}

/// How a [`value_field_in`] shows and reads its number, beyond its suffix.
#[derive(Clone, Copy, Debug, Default)]
struct FieldOpts {
    /// Decimal places shown (default: one, or two for a small range).
    decimals: Option<u32>,
    /// The unit typed units convert into (default: the suffix's, at [`set_field_resolution`]).
    units: Option<FieldUnits>,
    wrap_drag: bool,
}

/// A [`value_field`] that leaves `trailing` points free inside its box, right of the suffix, and
/// returns the box with the number's response.
fn value_field_in(
    ui: &mut Ui,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    suffix: &str,
    width: f32,
    trailing: f32,
    opts: FieldOpts,
) -> (Response, Rect) {
    value_field_box(ui, value, range, suffix, vec2(width, VALUE_FIELD_H), trailing, opts)
}

/// [`value_field_in`] with a box of any `size`.
fn value_field_box(
    ui: &mut Ui,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    suffix: &str,
    size: Vec2,
    trailing: f32,
    opts: FieldOpts,
) -> (Response, Rect) {
    let FieldOpts { decimals, units, wrap_drag } = opts;
    let t = Tokens::get(ui.ctx());
    let (rect, slot) = ui.allocate_exact_size(size, Sense::hover());
    let scrub = take_label_scrub(ui, rect);
    let (dpi, ppi) = ui.data(|d| d.get_temp::<(f64, f64)>(field_resolution_id())).unwrap_or((72.0, 72.0));
    // Typed text loses its spaces before parsing, so a " pt" suffix reads as "pt".
    let typed_suffix = suffix.trim();
    let units = match units {
        Some(u) => crate::numeric_expression::Units { unit: Some(u.unit), suffix: typed_suffix, dpi: u.dpi, ppi: u.ppi },
        None => crate::numeric_expression::Units { unit: unit_of_suffix(typed_suffix), suffix: typed_suffix, dpi, ppi },
    };
    surface(ui, rect, t.field, false);
    if !t.bevel {
        ui.painter().rect_stroke(rect, t.radius_sm, Stroke::new(1.0, t.field_border), StrokeKind::Inside);
    }
    // The number's slot ends 1 pt left of the suffix, by its measured width (the suffix is drawn
    // 6 pt from the right edge), so a one-character unit like `%` leaves the number room (#3040).
    let suffix_font = theme::mono(11.0);
    let suffix_w = if suffix.is_empty() { 0.0 } else { ui.painter().layout_no_wrap(suffix.to_owned(), suffix_font.clone(), t.text_faint).size().x + 3.0 };
    // 1 pt higher than centred: digits have no descenders, so centred text looks low. The slot
    // is at least one text row tall, so a slim box's number isn't pushed out of it (#2823).
    let font = theme::mono(12.0);
    let row_h = ui.painter().layout_no_wrap("0".into(), font.clone(), t.text).size().y;
    let slot_h = (rect.height() - 4.0).max(row_h).min(rect.height() - 2.0);
    let cy = rect.center().y - 1.0;
    let field = Rect::from_min_max(pos2(rect.left() + 4.0, cy - slot_h / 2.0), pos2(rect.right() - 4.0 - suffix_w - trailing, cy + slot_h / 2.0));
    // Small ranges (gamma 0.01–9.99, 0–1 centres) need two decimals and a finer drag, like Photoshop.
    let fine = range.end() - range.start() <= 10.0;
    let (lo, hi) = (*range.start(), *range.end());
    let (step, grid) = arrow_step(ui, slot.id, if fine { 0.01 } else { 1.0 });
    // new_child (not scope_builder): a scope would move the parent cursor back to the child rect.
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(field));
    let mut resp = {
        let ui = &mut child;
        {
            ui.style_mut().visuals.widgets.inactive.bg_fill = Color32::TRANSPARENT;
            ui.style_mut().visuals.widgets.inactive.weak_bg_fill = Color32::TRANSPARENT;
            ui.style_mut().visuals.widgets.inactive.bg_stroke = Stroke::NONE;
            ui.style_mut().visuals.widgets.hovered.bg_stroke = Stroke::NONE;
            ui.style_mut().visuals.widgets.hovered.weak_bg_fill = Color32::TRANSPARENT;
            ui.style_mut().override_font_id = Some(font);
            // The theme's interact height (24) is taller than the number's slot: the DragValue
            // would grow past the slot's bottom and sit low, or spill out of a slim box (#2823).
            // Its horizontal padding would push a long number out of the box's left edge.
            ui.spacing_mut().interact_size.y = field.height();
            ui.spacing_mut().button_padding = Vec2::ZERO;
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let layout = egui::Layout::centered_and_justified(ui.layout().main_dir());
                ui.allocate_ui_with_layout(field.size(), layout, |ui| number_edit(ui, value, range, fine, decimals, units, wrap_drag)).inner
            })
            .inner
        }
    };
    ui.data_mut(|d| d.insert_temp(slot.id, resp.id));
    if let Some(s) = scrub {
        if s.started && ui.memory(|m| m.has_focus(resp.id)) {
            ui.memory_mut(|m| m.surrender_focus(resp.id));
        }
        if s.steps != 0.0 {
            let unit = if fine { 0.01 } else { 1.0 };
            // Onto the step grid first, so a whole-number field drops its decimals (55.4 + 1 = 56).
            let v = ((*value / unit).round() + s.steps) * unit;
            let v = (v * 1e4).round() / 1e4;
            // `f32::clamp` panics on a reversed or NaN range, which `DragValue::range` accepts.
            let v = if lo <= hi { v.clamp(lo, hi) } else { v };
            if v.is_finite() && v != *value {
                *value = v;
                resp.mark_changed();
            }
        }
        // Callers coalesce a drag into one history step; the label's drag is the field's.
        use egui::response::Flags;
        resp.flags.set(Flags::DRAG_STARTED, resp.flags.contains(Flags::DRAG_STARTED) || s.started);
        resp.flags.set(Flags::DRAGGED, resp.flags.contains(Flags::DRAGGED) || s.dragging);
        resp.flags.set(Flags::DRAG_STOPPED, resp.flags.contains(Flags::DRAG_STOPPED) || s.stopped);
    }
    if !suffix.is_empty() {
        ui.painter().text(pos2(rect.right() - 6.0 - trailing, rect.center().y), Align2::RIGHT_CENTER, suffix, suffix_font, t.text_faint);
    }
    if step != 0.0 {
        // Round to the step before adding it, so whole-number fields drop decimals (55.4 + 1 = 56).
        let v = (*value / grid).round() * grid + step;
        // Round to 4 decimal places, so repeated 0.1 steps don't leave float errors.
        let v = (v * 1e4).round() / 1e4;
        // `f32::clamp` panics on a reversed or NaN range, which `DragValue::range` accepts.
        *value = if lo <= hi { v.clamp(lo, hi) } else { v };
        ui.memory_mut(|m| m.request_focus(resp.id));
        // The text stays selected, as in Photoshop: typing after a step replaces the value.
        crate::field_tab::select_all(ui.ctx(), resp.id, &if fine { fmt_num2(f64::from(*value)) } else { fmt_num(f64::from(*value)) });
        resp.mark_changed();
    }
    (resp, rect)
}

/// Width of the ▾ that opens a [`popup_value_field`]'s slider, inside the field's box.
pub const POPUP_ARROW_W: f32 = 14.0;

/// Width of a [`popup_value_field`]'s pop-up slider.
const POPUP_SLIDER_W: f32 = 160.0;

/// What a [`popup_value_field`] did this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PopupFieldResponse {
    /// The value changed (typed, scrubbed, stepped with the arrow keys, or set on the slider).
    pub changed: bool,
    /// While the value is dragged (on the slider, from the ▾, or scrubbing the number): a number
    /// for that drag, the same on every frame of it, for the caller to coalesce the drag's edits
    /// into one history step.
    pub drag: Option<u64>,
}

/// A [`value_field`] with Photoshop's pop-up slider (the Layers panel's Opacity and Fill): a click
/// on the ▾ inside the field's right edge opens a slider below it, a click outside closes it.
/// Pressing the ▾ and dragging moves the value straight away, as far as the slider would, and
/// closes the slider on release. The number can still be typed, scrubbed and stepped like any
/// value field. `name` (already translated) names the ▾ for its tooltip and for accessibility.
pub fn popup_value_field(ui: &mut Ui, name: &str, value: &mut f32, range: std::ops::RangeInclusive<f32>, suffix: &str, width: f32) -> PopupFieldResponse {
    let t = Tokens::get(ui.ctx());
    let (lo, hi) = (*range.start(), *range.end());
    let (field, rect) = value_field_in(ui, value, range.clone(), suffix, width, POPUP_ARROW_W, FieldOpts::default());
    let arrow = Rect::from_min_max(pos2(rect.right() - POPUP_ARROW_W - 2.0, rect.top() + 2.0), rect.max - vec2(2.0, 2.0));
    let resp = ui.interact(arrow, field.id.with("popup-slider"), Sense::click_and_drag());
    let name = name.trim_end_matches([':', '：']).to_string();
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), &name));
    let resp = resp.on_hover_text(&name);
    let drag_key = resp.id.with("drag");
    // Every drag gets a new number when it starts, even if its first frame changes nothing.
    let gesture = |ui: &Ui, started: bool| {
        if started {
            let n = ui.ctx().cumulative_pass_nr();
            ui.data_mut(|d| d.insert_temp(drag_key, n));
        }
        ui.data(|d| d.get_temp::<u64>(drag_key))
    };
    let mut out = PopupFieldResponse { changed: field.changed(), drag: None };
    let scrub = gesture(ui, field.drag_started());
    if field.changed() && field.dragged() {
        out.drag = scrub;
    }

    let popup_id = egui::Popup::default_response_id(&resp);
    let start_key = resp.id.with("start");
    if resp.drag_started() {
        egui::Popup::open_id(ui.ctx(), popup_id);
        ui.data_mut(|d| d.insert_temp(start_key, *value));
        gesture(ui, true);
    }
    if resp.dragged()
        && let (Some(start), Some(origin), Some(pos)) =
            (ui.data(|d| d.get_temp::<f32>(start_key)), ui.input(|i| i.pointer.press_origin()), resp.interact_pointer_pos())
    {
        // The pointer moves the value as much as it would move the slider's knob.
        let track = POPUP_SLIDER_W - 14.0;
        let v = start + (pos.x - origin.x) / track * (hi - lo);
        let v = if lo <= hi && v.is_finite() { v.clamp(lo, hi) } else { start };
        if v != *value {
            *value = v;
            out.changed = true;
        }
        out.drag = gesture(ui, false);
    }
    if resp.drag_stopped() {
        egui::Popup::close_id(ui.ctx(), popup_id);
    }

    let popup = egui::Popup::from_toggle_button_response(&resp)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .anchor(rect)
        .align(egui::RectAlign::BOTTOM_END)
        .align_alternatives(&[egui::RectAlign::TOP_END]);
    let open = popup.is_open();
    if resp.hovered() || resp.dragged() || open {
        ui.painter().rect_filled(arrow, t.radius_sm, t.field_border.gamma_multiply(0.6));
    }
    chevron_icon(ui, arrow, ui.style().interact(&resp), open);
    popup.show(|ui| {
        ui.set_width(POPUP_SLIDER_W);
        let s = slider(ui, value, range, None);
        s.widget_info(|| egui::WidgetInfo::slider(ui.is_enabled(), f64::from(*value), &name));
        let key = gesture(ui, s.drag_started());
        if s.changed() {
            out.changed = true;
            // A click on the track is one edit of its own; a drag is one edit however long.
            if s.dragged() {
                out.drag = key;
            }
        }
    });
    out
}

/// Height of a [`text_field`]'s box, matching a [`value_field`].
pub const TEXT_FIELD_H: f32 = 24.0;

/// Inner padding of a [`text_field`]'s box: 6 pt either side, 2 pt above and below.
const TEXT_FIELD_PAD: Vec2 = vec2(6.0, 2.0);

/// A single-line text field in PhotoCraft's shared field style (#3789): the same box as a
/// [`value_field`] -- the theme's field fill and `field_border`, rounded by `radius_sm` -- with a
/// consistent inner padding and no focus outline. egui's default white focus ring (its
/// `selection.stroke`) is what made the Preferences search field look different from the Color
/// Picker's hex field, so every plain, non-numeric input (search, hex, name, filter) is drawn
/// through here to look and behave the same. `id` carries the field's text and focus across
/// frames; `hint` shows while it is empty; `mono` picks the monospaced face for hex and other
/// code-like values.
pub fn text_field(ui: &mut Ui, id: egui::Id, text: &mut String, hint: &str, width: f32, mono: bool) -> Response {
    text_field_ex(ui, id, text, hint, width, mono, None)
}

/// [`text_field`] limited to `char_limit` characters (a hex field's six digits, say).
pub fn text_field_ex(ui: &mut Ui, id: egui::Id, text: &mut String, hint: &str, width: f32, mono: bool, char_limit: Option<usize>) -> Response {
    let t = Tokens::get(ui.ctx());
    let (rect, _) = ui.allocate_exact_size(vec2(width.max(16.0), TEXT_FIELD_H), Sense::hover());
    surface(ui, rect, t.field, false);
    if !t.bevel {
        ui.painter().rect_stroke(rect, t.radius_sm, Stroke::new(1.0, t.field_border), StrokeKind::Inside);
    }
    let font = if mono { theme::mono(12.0) } else { egui::FontId::proportional(12.5) };
    let mut edit = egui::TextEdit::singleline(text).id(id).hint_text(hint).frame(egui::Frame::NONE).font(font);
    if let Some(limit) = char_limit {
        edit = edit.char_limit(limit);
    }
    // A child Ui keeps the field's selection colour local: `Visuals::selection` also fills selected
    // list rows and slider trails, which must keep the theme's.
    let layout = egui::Layout::centered_and_justified(egui::Direction::TopDown);
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(rect.shrink2(TEXT_FIELD_PAD)).layout(layout));
    child.style_mut().visuals.selection.bg_fill = t.field_selection();
    child.add(edit)
}

fn first_field_id() -> egui::Id {
    egui::Id::new("pc-focus-first-field")
}

/// While `on`, the next number field drawn takes the keyboard focus with its text selected (a
/// dialog opening on its first value, as in Photoshop). Turn it off once the body is laid out.
pub fn focus_first_field(ctx: &egui::Context, on: bool) {
    ctx.data_mut(|d| {
        if on {
            d.insert_temp(first_field_id(), true);
        } else {
            d.remove::<bool>(first_field_id());
        }
    });
}

/// The number in a [`value_field`]. A typed number applies as it's typed; arithmetic waits for
/// Enter, Tab or click-away, because per keystroke `5/2` would land first and a caller that
/// rounds it would cut `5/2*2` short. `changed()` means a new value, not just a keystroke.
fn number_edit(
    ui: &mut Ui,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    fine: bool,
    decimals: Option<u32>,
    units: crate::numeric_expression::Units,
    wrap_drag: bool,
) -> Response {
    let (id, ctx) = (ui.next_auto_id(), ui.ctx().clone());
    // Focused before the DragValue is drawn, so it gains focus this frame and selects its text.
    if ui.data_mut(|d| d.remove_temp::<bool>(first_field_id())).unwrap_or(false) {
        ui.memory_mut(|m| m.request_focus(id));
    }
    // Tab in a dialog goes from one of these to the next (field_tab.rs).
    if ui.is_enabled() {
        crate::field_tab::register(&ctx, id);
    }
    let held = id.with("arithmetic");
    let math = ui.memory(|m| m.has_focus(id)) && ui.data(|d| d.get_temp(held)).unwrap_or(false);
    ui.data_mut(|d| d.insert_temp(held, math));
    let before = *value;
    let speed = if fine { 0.01 } else { 0.5 };
    let wrapping = wrap_drag && crate::scrub_wrap::active(ui);
    let wrapped_drag = wrapping && ui.input(|i| i.pointer.primary_down());
    let mut resp = ui.add(
        egui::DragValue::new(value)
            .range(range.clone())
            // A warp is an absolute-position jump. The scoped scrub below removes that jump.
            .speed(if wrapped_drag { 0.0 } else { speed })
            .custom_formatter(move |v, _| match decimals {
                Some(d) => fmt_num_prec(v, d),
                None if fine => fmt_num2(v),
                None => fmt_num(v),
            })
            .update_while_editing(!math)
            // Focus is read when parsing, not above: Tab hands it on inside `ui.add`.
            .custom_parser(move |s| {
                let v = parse_num_in(s, units);
                if ctx.memory(|m| m.has_focus(id)) && plain(s).is_none() {
                    // Still typing arithmetic: hold it, and stop applying keystrokes once it parses.
                    ctx.data_mut(|d| d.insert_temp(held, v.is_some()));
                    return None;
                }
                v
            }),
    );
    if wrap_drag {
        crate::scrub_wrap::update(ui, &resp, value, before, range, speed, wrapping);
    }
    resp.flags.set(egui::response::Flags::CHANGED, *value != before);
    resp
}

/// Increments a numerical field with the up/down arrow keys. Increments by 1 by default, 10 with shift, and 0.1 with ctrl/cmd.
/// Returns the amount to add, and what to round the value to first.
fn arrow_step(ui: &mut Ui, slot: egui::Id, step: f32) -> (f32, f32) {
    use egui::{Key, Modifiers};
    // The field's id is only known once it's drawn, so `value_field` saves it under `slot` for the next frame.
    let Some(id) = ui.data(|d| d.get_temp::<egui::Id>(slot)).filter(|id| ui.memory(|m| m.has_focus(*id))) else {
        return (0.0, step);
    };
    let (n, grid) = ui.input_mut(|i| {
        let (mut n, mut grid) = (0.0, step);
        // egui ignores an extra shift when matching, so shift is checked before the plain arrows to get its larger step.
        for (mods, k) in [(Modifiers::COMMAND, (step / 10.0).max(0.01)), (Modifiers::SHIFT, 10.0 * step), (Modifiers::NONE, step)] {
            let presses = i.count_and_consume_key(mods, Key::ArrowUp) as f32 - i.count_and_consume_key(mods, Key::ArrowDown) as f32;
            if presses != 0.0 && mods == Modifiers::COMMAND {
                grid = k;
            }
            n += k * presses;
        }
        (n, grid)
    });
    // If a sum like 5+5 has been typed, unfocus the field so it calculates it (as Enter would) before we step.
    if n != 0.0 && ui.data(|d| d.get_temp(id.with("arithmetic"))).unwrap_or(false) {
        ui.memory_mut(|m| m.surrender_focus(id));
    }
    (n, grid)
}

/// Thin-track slider with a round knob. `gradient` paints the track (e.g. hue spectrum).
pub fn slider(ui: &mut Ui, value: &mut f32, range: std::ops::RangeInclusive<f32>, gradient: Option<&[Color32]>) -> Response {
    slider_with(ui, value, range, gradient, RowGestures::default())
}

/// Photoshop's gestures on a slider row beyond click and drag, as measured on Color Balance in
/// Photoshop 25.4 (Image › Adjustments dialog and the adjustment layer's Properties). Shift, Ctrl
/// and Alt change nothing about clicks, double-clicks or drags; the wheel takes ten steps with
/// Shift. (Up/Down in the field are every value field's, see [`value_field`].)
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RowGestures {
    /// A double-click on the slider (knob or track; not the label or field) sets this value.
    pub reset: Option<f32>,
    /// Each wheel notch over the slider moves the value by this step (up = higher; ×10 with
    /// Shift). Photoshop does this in dialogs, not in the Properties panel.
    pub wheel_step: Option<f32>,
}

/// Wheel notches this frame while the pointer is over `resp` (Shift ×10). Wheel lines add up in
/// egui memory until they make whole notches (high-resolution wheels report fractions of one);
/// smooth (point) deltas count `line_scroll_speed` points per notch, like the canvas wheel.
fn wheel_notches(ui: &Ui, resp: &Response) -> f32 {
    wheel_notches_of(ui, resp.id, resp.hovered(), 10.0)
}

/// [`wheel_notches`] of widget `id` while `hovered`, a notch with Shift counting `shift_notches`.
pub(crate) fn wheel_notches_of(ui: &Ui, id: egui::Id, hovered: bool, shift_notches: f32) -> f32 {
    if !hovered {
        return 0.0;
    }
    let acc_id = id.with("wheel-acc");
    let mut acc: f32 = ui.data(|d| d.get_temp(acc_id)).unwrap_or(0.0);
    let per_line = ui.ctx().options(|o| o.input_options.line_scroll_speed);
    let per_line = if per_line.is_finite() && per_line > 0.0 { per_line } else { 40.0 };
    let mut notches = 0.0;
    let wheel: Vec<(egui::MouseWheelUnit, Vec2, bool)> = ui.input(|i| {
        i.events
            .iter()
            .filter_map(|e| match e {
                egui::Event::MouseWheel { unit, delta, modifiers, .. } => Some((*unit, *delta, modifiers.shift)),
                _ => None,
            })
            .collect()
    });
    for (unit, delta, shift) in wheel {
        // Some systems turn Shift+wheel into a horizontal scroll.
        let d = if delta.y != 0.0 { delta.y } else { delta.x };
        acc += match unit {
            egui::MouseWheelUnit::Point => d / per_line,
            _ => d,
        };
        let whole = acc.trunc();
        acc -= whole;
        notches += if shift { whole * shift_notches } else { whole };
    }
    if !acc.is_finite() {
        acc = 0.0;
    }
    ui.data_mut(|d| d.insert_temp(acc_id, acc));
    if notches.is_finite() { notches } else { 0.0 }
}

/// Windows' default double-click time (Photoshop's; egui's own default is 0.3 s).
const DOUBLE_CLICK_S: f64 = 0.5;

/// Whether this click on `resp` completes a double-click, counted the Windows way (Photoshop):
/// a click soon after a lone click at the same spot; the click after a double-click starts over.
/// (egui would call a third quick click a triple-click even when the first two weren't a double.)
fn second_click(ui: &Ui, resp: &Response) -> bool {
    if !resp.clicked() {
        return false;
    }
    let key = resp.id.with("pc-last-click");
    let (time, pos) = ui.input(|i| (i.time, i.pointer.interact_pos()));
    let dist = ui.ctx().options(|o| o.input_options.max_click_dist);
    let last: Option<(f64, Pos2)> = ui.data(|d| d.get_temp(key));
    let double = match (last, pos) {
        (Some((t, p)), Some(q)) => time - t < DOUBLE_CLICK_S && p.distance(q) < dist,
        _ => false,
    };
    ui.data_mut(|d| {
        if let (false, Some(q)) = (double, pos) {
            d.insert_temp(key, (time, q));
        } else {
            d.remove::<(f64, Pos2)>(key);
        }
    });
    double
}

fn slider_with(ui: &mut Ui, value: &mut f32, range: std::ops::RangeInclusive<f32>, gradient: Option<&[Color32]>, g: RowGestures) -> Response {
    let t = Tokens::get(ui.ctx());
    let width = ui.available_width().max(60.0);
    let (rect, mut resp) = ui.allocate_exact_size(vec2(width, 18.0), Sense::click_and_drag());
    let (lo, hi) = (*range.start(), *range.end());
    let track = Rect::from_center_size(rect.center(), vec2(rect.width() - 14.0, if gradient.is_some() { 5.0 } else { 3.0 }));
    if let Some(p) = resp.interact_pointer_pos()
        && (resp.dragged() || resp.clicked())
    {
        let f = ((p.x - track.left()) / track.width()).clamp(0.0, 1.0);
        let nv = lo + f * (hi - lo);
        if (nv - *value).abs() > f32::EPSILON {
            *value = nv;
            resp.mark_changed();
        }
    }
    // The double-click's first click has moved the knob to the pointer; the second resets.
    if let Some(reset) = g.reset
        && second_click(ui, &resp)
    {
        *value = reset;
        resp.mark_changed();
    }
    if let Some(step) = g.wheel_step {
        let n = wheel_notches(ui, &resp);
        if n != 0.0 {
            let nv = (*value + n * step).clamp(lo, hi);
            if nv != *value {
                *value = nv;
                resp.mark_changed();
            }
        }
    }
    let f = ((*value - lo) / (hi - lo)).clamp(0.0, 1.0);
    let knob_x = track.left() + f * track.width();
    let painter = ui.painter();
    match gradient {
        Some(colors) if colors.len() >= 2 => {
            let n = colors.len() - 1;
            for (i, w) in colors.windows(2).enumerate() {
                let x0 = track.left() + track.width() * i as f32 / n as f32;
                let x1 = track.left() + track.width() * (i + 1) as f32 / n as f32;
                let mut mesh = egui::Mesh::default();
                let r = Rect::from_min_max(pos2(x0, track.top()), pos2(x1, track.bottom()));
                mesh.colored_vertex(r.left_top(), w[0]);
                mesh.colored_vertex(r.right_top(), w[1]);
                mesh.colored_vertex(r.right_bottom(), w[1]);
                mesh.colored_vertex(r.left_bottom(), w[0]);
                mesh.add_triangle(0, 1, 2);
                mesh.add_triangle(0, 2, 3);
                painter.add(mesh);
            }
        }
        _ => {
            painter.rect_filled(track, 2.0, t.field_border);
            let filled = Rect::from_min_max(track.min, pos2(knob_x, track.max.y));
            painter.rect_filled(filled, 2.0, if t.bevel { t.accent } else { t.text_dim });
        }
    }
    let knob = pos2(knob_x, rect.center().y);
    if t.bevel {
        let kr = Rect::from_center_size(knob, vec2(10.0, 16.0));
        surface(ui, kr, t.card, true);
    } else {
        let kr = if t.pro { 6.0 } else { 7.0 };
        painter.circle_filled(knob + vec2(0.0, 1.0), kr + 0.5, Color32::from_black_alpha(90));
        painter.circle_filled(knob, kr, if t.pro { Color32::from_gray(236) } else { Color32::WHITE });
        if t.pro {
            painter.circle_stroke(knob, kr, Stroke::new(1.0, Color32::from_gray(40)));
        }
        if resp.hovered() || resp.dragged() {
            painter.circle_stroke(knob, 9.5, Stroke::new(2.0, t.accent_soft));
        }
    }
    resp
}

/// Labelled slider row: `Label ........ [value field]` above a full-width thin slider.
pub fn slider_row(ui: &mut Ui, label: &str, value: &mut f32, range: std::ops::RangeInclusive<f32>, suffix: &str, gradient: Option<&[Color32]>) -> Response {
    slider_row_with(ui, label, value, range, suffix, gradient, RowGestures::default())
}

/// [`slider_row`] with Photoshop's extra gestures ([`RowGestures`]).
pub fn slider_row_with(
    ui: &mut Ui,
    label: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    suffix: &str,
    gradient: Option<&[Color32]>,
    g: RowGestures,
) -> Response {
    slider_row_prec(ui, label, value, range, suffix, gradient, g, None)
}

/// [`slider_row_with`] whose field shows `decimals` places instead of the one (or two, for a
/// small range) it picks itself: Exposure's ±20 stops show two, its Offset four (#4164).
#[allow(clippy::too_many_arguments)]
pub fn slider_row_prec(
    ui: &mut Ui,
    label: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    suffix: &str,
    gradient: Option<&[Color32]>,
    g: RowGestures,
    decimals: Option<u32>,
) -> Response {
    let t = Tokens::get(ui.ctx());
    let mut changed_resp = None;
    ui.horizontal(|ui| {
        // The label scrubs the row's field, across the row (Photoshop's scrubby labels).
        let l = ui.label(egui::RichText::new(tl!(label)).color(t.text_dim));
        scrub_label_within(ui, &l, f32::INFINITY);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            changed_resp = Some(value_field_in(ui, value, range.clone(), suffix, 74.0, 0.0, FieldOpts { decimals, ..Default::default() }).0);
        });
    });
    let s = slider_with(ui, value, range, gradient, g);
    let mut r = s.clone();
    if let Some(v) = changed_resp {
        if v.changed() {
            r.mark_changed();
        }
        // A scrub of the field or its label is one drag, like the slider's, for history.
        use egui::response::Flags;
        r.flags |= v.flags & (Flags::DRAG_STARTED | Flags::DRAGGED | Flags::DRAG_STOPPED);
    }
    ui.add_space(4.0);
    r
}

/// iOS-style toggle switch.
pub fn toggle(ui: &mut Ui, on: &mut bool, label: &str) -> Response {
    let t = Tokens::get(ui.ctx());
    if t.pro {
        return checkbox(ui, on, label);
    }
    let mut resp = ui
        .horizontal(|ui| {
            let (rect, resp) = ui.allocate_exact_size(vec2(30.0, 17.0), Sense::click());
            let how_on = ui.ctx().animate_bool(resp.id, *on);
            let bg = if *on { t.accent } else { t.field_border };
            if t.bevel {
                surface(ui, rect, if *on { t.accent } else { t.field }, false);
            } else {
                ui.painter().rect_filled(rect, 8.5, bg);
            }
            let x = egui::lerp((rect.left() + 8.5)..=(rect.right() - 8.5), how_on);
            ui.painter().circle_filled(pos2(x, rect.center().y), 6.5, Color32::WHITE);
            ui.label(egui::RichText::new(tl!(label)).color(if *on { t.text } else { t.text_dim }));
            resp
        })
        .inner;
    if resp.clicked() {
        *on = !*on;
        resp.mark_changed();
    }
    resp
}

/// Big white primary button.
pub fn primary_button(ui: &mut Ui, label: &str, min_width: f32) -> Response {
    let t = Tokens::get(ui.ctx());
    button_impl(ui, label, min_width, t.primary_bg, t.primary_text, true)
}

/// The width of the widest of `labels` as a button, for a column of equal-width buttons.
pub fn button_width(ui: &Ui, labels: &[&str]) -> f32 {
    let fg = Tokens::get(ui.ctx()).text;
    labels.iter().map(|l| button_layout(ui, l, 0.0, None, fg).1.x).fold(0.0, f32::max)
}

pub fn secondary_button(ui: &mut Ui, label: &str, min_width: f32) -> Response {
    let t = Tokens::get(ui.ctx());
    button_impl(ui, label, min_width, t.field, t.text, false)
}

/// A [`secondary_button`] `height` points tall, for slim bars such as the status bar (#2823).
pub fn compact_button(ui: &mut Ui, label: &str, height: f32) -> Response {
    let t = Tokens::get(ui.ctx());
    button_impl_h(ui, label, 0.0, Some(height), t.field, t.text, false)
}

/// What a dialog button does, which decides where the platform's button order puts it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ButtonRole {
    /// The default answer (OK, Save, Yes), drawn as the primary button.
    Default,
    /// Another answer that closes the dialog (Don't Save, No).
    Alternate,
    /// Closes the dialog without acting.
    Cancel,
    /// Acts but keeps the dialog open (Apply).
    Apply,
}

impl ButtonRole {
    /// Position from the left in the platform's order. Windows and Linux put the default action
    /// first (OK Cancel Apply, Yes No Cancel); macOS puts it last, in the corner, with Cancel beside
    /// it and the other answers further left (Don't Save, Cancel, Save).
    fn slot(self, mac: bool) -> u8 {
        match (self, mac) {
            (Self::Default, false) | (Self::Alternate, true) => 0,
            (Self::Alternate, false) | (Self::Cancel, true) => 1,
            (Self::Cancel, false) | (Self::Apply, true) => 2,
            (Self::Apply, false) | (Self::Default, true) => 3,
        }
    }
}

/// One button of a [`dialog_buttons`] row.
#[derive(Clone, Copy)]
pub struct DialogButton<'a> {
    pub role: ButtonRole,
    pub label: &'a str,
    pub min_width: f32,
    pub enabled: bool,
}

impl<'a> DialogButton<'a> {
    pub fn new(role: ButtonRole, label: &'a str, min_width: f32) -> Self {
        Self { role, label, min_width, enabled: true }
    }

    pub fn enabled(self, enabled: bool) -> Self {
        Self { enabled, ..self }
    }
}

/// A dialog's button row at the cursor, in the platform's order (see [`ButtonRole`]): every modal
/// draws its buttons through this, so they all agree. Inside a right-to-left row it sits at the
/// right edge. The buttons are laid out left to right, so Tab walks them in reading order.
/// Returns the role of the button clicked this frame.
pub fn dialog_buttons(ui: &mut Ui, buttons: &[DialogButton]) -> Option<ButtonRole> {
    let mac = ui.ctx().os() == egui::os::OperatingSystem::Mac;
    let gap = ui.spacing().item_spacing.x;
    let size = buttons.iter().fold(Vec2::ZERO, |acc, b| {
        let s = button_size(ui, b.label, b.min_width);
        vec2(acc.x + s.x, acc.y.max(s.y))
    }) + vec2(gap * buttons.len().saturating_sub(1) as f32, 0.0);
    ui.allocate_ui_with_layout(size, egui::Layout::left_to_right(egui::Align::Center), |ui| {
        let mut hit = None;
        for slot in 0..4 {
            for b in buttons.iter().filter(|b| b.role.slot(mac) == slot) {
                let r = ui
                    .add_enabled_ui(b.enabled, |ui| {
                        if b.role == ButtonRole::Default { primary_button(ui, b.label, b.min_width) } else { secondary_button(ui, b.label, b.min_width) }
                    })
                    .inner;
                if r.clicked() {
                    hit = Some(b.role);
                }
            }
        }
        hit
    })
    .inner
}

/// The size of a primary or secondary button: its label plus padding, at least `min_width` wide.
fn button_size(ui: &Ui, label: &str, min_width: f32) -> Vec2 {
    button_layout(ui, label, min_width, None, Tokens::get(ui.ctx()).text).1
}

fn button_layout(ui: &Ui, label: &str, min_width: f32, height: Option<f32>, fg: Color32) -> (std::sync::Arc<egui::Galley>, Vec2) {
    let galley = ui.painter().layout_no_wrap(tl!(label).to_owned(), theme::medium(13.0), fg);
    let h = height.unwrap_or(if Tokens::get(ui.ctx()).pro { 28.0 } else { 30.0 });
    let size = vec2((galley.size().x + 28.0).max(min_width), h);
    (galley, size)
}

fn button_impl(ui: &mut Ui, label: &str, min_width: f32, bg: Color32, fg: Color32, primary: bool) -> Response {
    button_impl_h(ui, label, min_width, None, bg, fg, primary)
}

/// [`button_impl`] with an explicit `height` (`None`: the theme's button height).
fn button_impl_h(ui: &mut Ui, label: &str, min_width: f32, height: Option<f32>, bg: Color32, fg: Color32, primary: bool) -> Response {
    let t = Tokens::get(ui.ctx());
    let (galley, size) = button_layout(ui, label, min_width, height, fg);
    let h = size.y;
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    // Painted text: name the button for accessibility (and so tests and agents can find it).
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), label));
    if resp.has_focus() {
        // Keyboard focus (Tab); clicks don't focus egui buttons.
        let r = if t.pro { h / 2.0 } else { t.radius_sm } + 2.0;
        ui.painter().rect_stroke(rect.expand(2.0), r, Stroke::new(2.0, t.accent), StrokeKind::Outside);
    }
    if t.pro {
        // Spectrum buttons: fully rounded; primary = filled accent, secondary = outline.
        let down = resp.is_pointer_button_down_on();
        let r = h / 2.0;
        if primary {
            let fill = if down {
                bg.gamma_multiply(0.8)
            } else if resp.hovered() {
                bg.gamma_multiply(0.9)
            } else {
                bg
            };
            ui.painter().rect_filled(rect, r, fill);
        } else {
            if resp.hovered() || down {
                ui.painter().rect_filled(rect, r, t.hover);
            }
            ui.painter().rect_stroke(rect, r, Stroke::new(1.5, if resp.hovered() { t.text } else { t.text_dim }), StrokeKind::Inside);
        }
        ui.painter().galley(rect.center() - galley.size() / 2.0, galley, fg);
        return resp;
    }
    let fill = if resp.is_pointer_button_down_on() {
        bg.gamma_multiply(0.85)
    } else if resp.hovered() {
        if primary { bg.gamma_multiply(0.93) } else { t.hover }
    } else {
        bg
    };
    surface(ui, rect, fill, !resp.is_pointer_button_down_on());
    if !t.bevel && !primary {
        ui.painter().rect_stroke(rect, t.radius_sm, Stroke::new(1.0, t.field_border), StrokeKind::Inside);
    }
    ui.painter().galley(rect.center() - galley.size() / 2.0, galley, fg);
    resp
}

/// Cached RGB data; drawing work is bounded by the 256 display bins, never the image size.
pub fn rgb_histogram(ui: &mut Ui, histogram: &photocraft_algo::histogram::RgbHistogram, height: f32) -> egui::Response {
    let t = Tokens::get(ui.ctx());
    let (rect, response) = ui.allocate_exact_size(egui::vec2(ui.available_width().max(1.0), height.max(1.0)), egui::Sense::hover());
    ui.painter().rect_filled(rect, t.radius_sm, t.histogram_background());
    let plot = rect.shrink(4.0);
    crate::rgb_histogram::paint(ui.painter(), plot, histogram, &t);
    ui.painter().rect_stroke(rect, t.radius_sm, egui::Stroke::new(1.0, t.field_border), egui::StrokeKind::Inside);
    response
}

/// Small caps section label.
pub fn section_label(ui: &mut Ui, text: &str) {
    let t = Tokens::get(ui.ctx());
    ui.label(egui::RichText::new(tl!(text)).font(theme::medium(11.5)).color(t.text_faint));
}

/// Hairline separator.
pub fn hairline(ui: &mut Ui) {
    let t = Tokens::get(ui.ctx());
    let (r, _) = ui.allocate_exact_size(vec2(ui.available_width(), 1.0), Sense::hover());
    ui.painter().line_segment([r.left_center(), r.right_center()], Stroke::new(1.0, t.separator));
}

/// Vertical hairline for horizontal layouts.
pub fn vline(ui: &mut Ui, height: f32) {
    let t = Tokens::get(ui.ctx());
    let (r, _) = ui.allocate_exact_size(vec2(9.0, height), Sense::hover());
    ui.painter().line_segment([r.center_top(), r.center_bottom()], Stroke::new(1.0, t.separator));
}

/// Hue spectrum stops for colour sliders.
pub fn hue_stops() -> Vec<Color32> {
    (0..=12)
        .map(|i| {
            let h = i as f32 / 12.0;
            let rgb = egui::ecolor::Hsva::new(h, 0.85, 1.0, 1.0).to_srgb();
            Color32::from_rgb(rgb[0], rgb[1], rgb[2])
        })
        .collect()
}

/// A compact labelled dropdown in the studio style.
pub fn dropdown<T: PartialEq + Clone>(ui: &mut Ui, id: &str, current: &mut T, options: &[(T, &str)], width: f32) -> bool {
    dropdown_hovered(ui, id, current, options, width).0
}

/// [`dropdown`], also returning the option under the pointer in its open list (live previews).
pub fn dropdown_hovered<T: PartialEq + Clone>(ui: &mut Ui, id: &str, current: &mut T, options: &[(T, &str)], width: f32) -> (bool, Option<T>) {
    let (chosen, hovered) = dropdown_with(ui, id, current, options, width, false);
    (!chosen.is_empty(), hovered)
}

/// [`dropdown_hovered`] whose value also steps with the wheel over its button or open list: down
/// picks the next option, up the previous one, one per notch (Shift too), stopping at the ends
/// (Photoshop's Layers panel blend modes, #1747). The wheel then scrolls neither the list, which
/// follows the value instead, nor what the dropdown is in. Returns every option chosen this frame
/// in order (one per notch; the last is `current`) and the option under the pointer in the open
/// list, none while the pointer rests where the wheel last stepped.
pub fn dropdown_wheel_hovered<T: PartialEq + Clone>(ui: &mut Ui, id: &str, current: &mut T, options: &[(T, &str)], width: f32) -> (Vec<T>, Option<T>) {
    dropdown_with(ui, id, current, options, width, true)
}

fn dropdown_with<T: PartialEq + Clone>(ui: &mut Ui, id: &str, current: &mut T, options: &[(T, &str)], width: f32, wheel: bool) -> (Vec<T>, Option<T>) {
    let label = options.iter().find(|(v, _)| v == current).map(|(_, l)| tl!(l)).unwrap_or("—");
    let (mut chosen, mut hovered) = (Vec::new(), None);
    // The wheel's notches add up over the button and the open list alike.
    let wheel_id = ui.make_persistent_id(id).with("wheel");
    let wheel = wheel && ui.is_enabled();
    let (mut over_list, mut wheeled, mut keyed) = (false, false, false);
    // A wheel step scrolls the open list to the new value: at once over the list, the next frame
    // after a step over the button.
    let reveal_id = wheel_id.with("reveal");
    let mut reveal = wheel && ui.data_mut(|d| d.remove_temp::<bool>(reveal_id)).unwrap_or(false);
    // The list opens on the press: press, drag onto an option and release to choose it (#2735).
    let (press, close) = crate::press_menu::PressCombo::before(ui, id);
    let response =
        egui::ComboBox::from_id_salt(id).selected_text(label).width(width).height(420.0).icon(chevron_icon).close_behavior(close).show_ui(ui, |ui| {
            fit_widest_option(ui, options.iter().map(|(_, l)| tl!(l)));
            // A typed letter jumps to the next option starting with it, ↩ closes the list (#1481).
            let labels: Vec<&str> = options.iter().map(|(_, l)| tl!(l)).collect();
            let shown = options.iter().position(|(v, _)| v == current);
            if let Some((v, _)) = type_ahead::keys(ui, ui.make_persistent_id(id), shown, &labels).and_then(|i| options.get(i)) {
                *current = v.clone();
                chosen.push(v.clone());
                reveal = true;
                keyed = true;
            }
            // Over the open list the wheel steps the value too, instead of scrolling the list.
            over_list = wheel && ui.rect_contains_pointer(ui.clip_rect());
            if over_list {
                let stepped = wheel_steps(ui, wheel_id, true, current, options);
                wheeled = !stepped.is_empty();
                reveal |= wheeled;
                chosen.extend(stepped);
            }
            // The row nearest the pointer, by its distance to each row.
            let pointer = ui.ctx().pointer_hover_pos().filter(|_| ui.rect_contains_pointer(ui.clip_rect()));
            let mut nearest: Option<(f32, &T)> = None;
            for (v, l) in options {
                let item = ui.selectable_label(v == current, tl!(l));
                if reveal && v == current {
                    item.scroll_to_me(None);
                }
                if item.hovered() {
                    hovered = Some(v.clone());
                }
                if let Some(p) = pointer {
                    let d = (item.rect.top() - p.y).max(p.y - item.rect.bottom()).max(0.0);
                    if nearest.is_none_or(|(n, _)| d < n) {
                        nearest = Some((d, v));
                    }
                }
                if press.chosen(ui, &item) {
                    *current = v.clone();
                    chosen.push(v.clone());
                }
            }
            // Between two rows (item spacing) the pointer hovers neither; it is still over the list,
            // so keep previewing the nearest row instead of flashing the current value back (#2553).
            if hovered.is_none() {
                hovered = nearest.map(|(_, v)| v.clone());
            }
        });
    press.after(&response.response);
    if combo_box_arrow_keys(ui, &response.response, current, options) {
        chosen.push(current.clone());
    }
    if wheel && !over_list {
        let stepped = wheel_steps(ui, wheel_id, response.response.hovered(), current, options);
        wheeled = !stepped.is_empty();
        if wheeled && egui::ComboBox::is_open(ui.ctx(), response.response.id) {
            ui.data_mut(|d| d.insert_temp(reveal_id, true));
        }
        chosen.extend(stepped);
    }
    // Nor is the option under a pointer resting where the keyboard picked another.
    if wheel_rests(ui, wheel_id, wheeled || keyed) {
        hovered = None;
    }
    (chosen, hovered)
}

/// Makes an open dropdown list as wide as its widest option is when hovered, from the start. egui
/// takes a plain row's frame stroke off its padding but draws a hovered row with the stroke, so a
/// row grows by two stroke widths under the pointer, and the list never shrinks back: scrolling
/// the widest option under the pointer widened the list (#2607).
fn fit_widest_option<'a>(ui: &mut Ui, labels: impl Iterator<Item = &'a str>) {
    let extend = Some(egui::TextWrapMode::Extend);
    let widest = labels.map(|l| egui::WidgetText::from(l).into_galley(ui, extend, f32::INFINITY, egui::FontSelection::Default).size().x).fold(0.0, f32::max);
    ui.set_min_width(widest + 2.0 * ui.spacing().button_padding.x);
}

/// Whether the pointer still rests where the wheel last stepped dropdown `id` (`wheeled`: it did
/// this frame). The option under it then isn't the one to preview: the value moved on without it.
fn wheel_rests(ui: &Ui, id: egui::Id, wheeled: bool) -> bool {
    let key = id.with("at");
    let pointer = ui.input(|i| i.pointer.hover_pos());
    if wheeled && let Some(p) = pointer {
        ui.data_mut(|d| d.insert_temp(key, p));
    }
    let rests = pointer.is_some() && ui.data(|d| d.get_temp::<Pos2>(key)) == pointer;
    if !rests {
        ui.data_mut(|d| d.remove::<Pos2>(key));
    }
    rests
}

/// The options the wheel over dropdown `id` (while `hovered`) steps `current` through this frame,
/// one per notch, in order; the last is the new `current`. While the pointer is over it the wheel
/// scrolls nothing else.
fn wheel_steps<T: PartialEq + Clone>(ui: &mut Ui, id: egui::Id, hovered: bool, current: &mut T, options: &[(T, &str)]) -> Vec<T> {
    if hovered {
        ui.input_mut(|i| i.smooth_scroll_delta = Vec2::ZERO);
    }
    let notches = wheel_notches_of(ui, id, hovered, 1.0);
    let Some(mut index) = options.iter().position(|(v, _)| v == current) else { return Vec::new() };
    let mut chosen = Vec::new();
    // Down (negative) is the next option. No more steps than there are options.
    for _ in 0..(notches.abs().min(options.len() as f32) as usize) {
        let next = if notches < 0.0 { index.checked_add(1) } else { index.checked_sub(1) };
        let Some((next, (v, _))) = next.and_then(|i| options.get(i).map(|o| (i, o))) else { break };
        index = next;
        chosen.push(v.clone());
    }
    if let Some(v) = chosen.last() {
        *current = v.clone();
    }
    chosen
}

/// Give a dropdown keyboard focus when it opens, then use the arrow keys to move through its
/// choices. `egui::ComboBox` opens a popup but leaves focus on the canvas by default, which makes
/// controls such as the Layers panel's Blend Mode dropdown unreachable from the keyboard.
fn combo_box_arrow_keys<T: PartialEq + Clone>(ui: &mut Ui, response: &Response, current: &mut T, options: &[(T, &str)]) -> bool {
    if response.clicked() {
        response.request_focus();
    }
    // The popup, rather than its button, becomes the focused egui layer after it opens. While it
    // is open it owns its navigation keys, even though `response.has_focus()` is then false.
    if !egui::ComboBox::is_open(ui.ctx(), response.id) || options.is_empty() {
        return false;
    }
    let step = if ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown)) {
        1
    } else if ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp)) {
        -1
    } else {
        return false;
    };
    let index = options.iter().position(|(value, _)| value == current).unwrap_or(0);
    let next = if step > 0 { (index + 1).min(options.len() - 1) } else { index.saturating_sub(1) };
    match options.get(next) {
        Some((value, _)) if next != index => {
            *current = value.clone();
            true
        }
        _ => false,
    }
}

/// The body of a right-click menu: as tall as its items up to the part of the window that can be
/// seen (clear of a taskbar a too-tall window runs under, #315), scrolling past that, so long
/// context menus (a layer's, the canvas tools') stay reachable on small windows instead of running
/// off the bottom. egui moves a popup up to keep it in the window, so a menu opened low on the
/// screen first shifts up and only scrolls when it is taller than the window.
pub fn menu_scroll<R>(ui: &mut Ui, add_contents: impl FnOnce(&mut Ui) -> R) -> R {
    // The popup frame's margin and stroke, and a small gap to the window's edges.
    let frame = ui.spacing().menu_margin.sum().y + 2.0 + 2.0 * MENU_EDGE;
    let room = (crate::work_area::visible_rect(ui.ctx()).height() - frame).max(MENU_MIN_HEIGHT);
    // A popup's Ui is only as tall as the popup was last frame (400 pt on the first), so ask for
    // the whole room: the area still shrinks to its rows when they need less.
    egui::ScrollArea::vertical().max_height(room).min_scrolled_height(room).show(ui, add_contents).inner
}

/// Gap kept between a context menu and the window's edges, and the shortest it gets.
const MENU_EDGE: f32 = 4.0;
const MENU_MIN_HEIGHT: f32 = 120.0;

/// The colour picker popup of a colour swatch: a click on `swatch` toggles it, a click outside
/// closes it. While open, its left edge stays where it first showed (at the swatch, or further
/// left when the window edge needs it), below the swatch or above it as room allows. The picker's
/// width follows its value readouts, and placing it anew every frame moved it under the pointer,
/// flipping it from side to side near the right edge of the window (#534).
pub fn swatch_popup(swatch: &Response) -> egui::Popup<'static> {
    // Room for the readouts to widen after the picker opened.
    const SLACK: f32 = 32.0;
    let popup = egui::Popup::from_toggle_button_response(swatch).close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside);
    let ctx = &swatch.ctx;
    let key = popup.get_id().with("left");
    if !popup.is_open() {
        ctx.data_mut(|d| d.remove::<f32>(key));
        return popup;
    }
    // Its width is known from the frame after it opened (egui sizes it unseen first).
    let left = ctx.data(|d| d.get_temp::<f32>(key)).or_else(|| {
        let width = popup.get_expected_size()?.x;
        let screen = ctx.content_rect();
        let left = swatch.rect.left().min(screen.right() - width - SLACK).max(screen.left());
        ctx.data_mut(|d| d.insert_temp(key, left));
        Some(left)
    });
    let Some(left) = left else { return popup };
    popup
        .anchor(Rect::from_x_y_ranges(left..=left, swatch.rect.y_range()))
        .align(egui::RectAlign::BOTTOM_START)
        .align_alternatives(&[egui::RectAlign::TOP_START])
}

pub fn dropdown_with_tooltips<T: PartialEq + Clone>(ui: &mut Ui, id: &str, current: &mut T, options: &[(T, &str, &str)], width: f32) -> bool {
    let label = options.iter().find(|(v, _, _)| v == current).map(|(_, l, _)| tl!(l)).unwrap_or("—");
    let mut changed = false;
    let (press, close) = crate::press_menu::PressCombo::before(ui, id);
    let response =
        egui::ComboBox::from_id_salt(id).selected_text(label).width(width).height(420.0).icon(chevron_icon).close_behavior(close).show_ui(ui, |ui| {
            fit_widest_option(ui, options.iter().map(|(_, l, _)| tl!(l)));
            let labels: Vec<&str> = options.iter().map(|(_, l, _)| tl!(l)).collect();
            let shown = options.iter().position(|(v, _, _)| v == current);
            if let Some((v, _, _)) = type_ahead::keys(ui, ui.make_persistent_id(id), shown, &labels).and_then(|i| options.get(i)) {
                *current = v.clone();
                changed = true;
            }
            for (v, l, tip) in options {
                let item = ui.selectable_label(v == current, tl!(l)).on_hover_text(tl!(tip));
                if press.chosen(ui, &item) {
                    *current = v.clone();
                    changed = true;
                }
            }
        });
    press.after(&response.response);
    if let Some((_, _, tip)) = options.iter().find(|(v, _, _)| v == current) {
        let _ = response.response.on_hover_text(tl!(tip));
    }
    changed
}

/// Paint a small checkerboard (transparency) in `rect`.
pub fn checker(painter: &egui::Painter, rect: Rect, cell: f32) {
    painter.rect_filled(rect, 0.0, Color32::from_gray(250));
    let nx = (rect.width() / cell).ceil() as i32;
    let ny = (rect.height() / cell).ceil() as i32;
    for j in 0..ny {
        for i in 0..nx {
            if (i + j) % 2 == 1 {
                let r = Rect::from_min_size(Pos2::new(rect.left() + i as f32 * cell, rect.top() + j as f32 * cell), Vec2::splat(cell)).intersect(rect);
                painter.rect_filled(r, 0.0, Color32::from_gray(214));
            }
        }
    }
}

/// A transparency checkerboard of `cell`-point squares clipped to `rect` with `radius` corners,
/// for rounded swatches (#3208): [`checker`] paints square cells, which showed past a rounded
/// fill's corners. This one is a repeating two-by-two tile textured onto the rounded rect.
pub fn checker_rounded(painter: &egui::Painter, rect: Rect, cell: f32, radius: impl Into<CornerRadius>) {
    let tile = checker_tile(painter.ctx());
    painter.add(egui::epaint::RectShape::filled(rect, radius, Color32::WHITE).with_texture(tile.id(), checker_uv(rect, cell)));
}

/// The texture range that tiles `rect` with `cell`-point squares: the tile holds two cells a side.
fn checker_uv(rect: Rect, cell: f32) -> Rect {
    let cell = if cell.is_finite() && cell > 0.0 { cell } else { 4.0 };
    Rect::from_min_max(Pos2::ZERO, pos2(rect.width() / (2.0 * cell), rect.height() / (2.0 * cell)))
}

fn checker_tile_id() -> egui::Id {
    egui::Id::new("transparency-checker-tile")
}

/// The checker's two-by-two tile (light, dark / dark, light), loaded once per context and shared
/// by every rounded checker; it repeats and never blurs.
fn checker_tile(ctx: &egui::Context) -> egui::TextureHandle {
    if let Some(tile) = ctx.data(|d| d.get_temp::<egui::TextureHandle>(checker_tile_id())) {
        return tile;
    }
    let (light, dark) = (Color32::from_gray(250), Color32::from_gray(214));
    let image = egui::ColorImage::new([2, 2], vec![light, dark, dark, light]);
    let options = egui::TextureOptions {
        magnification: egui::TextureFilter::Nearest,
        minification: egui::TextureFilter::Nearest,
        wrap_mode: egui::TextureWrapMode::Repeat,
        mipmap_mode: None,
    };
    let tile = ctx.load_texture("transparency-checker-tile", image, options);
    ctx.data_mut(|d| d.insert_temp(checker_tile_id(), tile.clone()));
    tile
}

/// Spectrum-style checkbox (blue when checked).
pub fn checkbox(ui: &mut Ui, on: &mut bool, label: &str) -> Response {
    let t = Tokens::get(ui.ctx());
    let mut resp = ui
        .horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            let (rect, resp) = ui.allocate_exact_size(vec2(14.0, 14.0), Sense::click());
            let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
            let l = ui
                .add(egui::Label::new(egui::RichText::new(tl!(label)).color(t.text_dim)).selectable(false).sense(Sense::click()))
                .on_hover_cursor(egui::CursorIcon::PointingHand);
            // Pressing the label presses the box (#4476).
            let pressed = resp.is_pointer_button_down_on() || l.is_pointer_button_down_on();
            check_box(ui.painter(), rect, *on, resp.hovered() || l.hovered(), pressed, &t);
            resp.union(l)
        })
        .inner;
    if resp.clicked() {
        *on = !*on;
        resp.mark_changed();
    }
    resp
}

/// The fill of a check box: the accent when `on`, else the field colour. While it's pressed
/// (the mouse is down on it or on its label) Photoshop shows that a release will act: an
/// unchecked box darkens a little and a checked one turns a little less opaque (#4476).
pub fn check_box_fill(on: bool, pressed: bool, t: &Tokens) -> Color32 {
    match (on, pressed) {
        (true, false) => t.accent,
        (true, true) => t.accent.gamma_multiply(0.75),
        (false, false) => t.field,
        (false, true) => t.field.lerp_to_gamma(Color32::BLACK, 0.18),
    }
}

/// Paint a check box in `rect`: [`check_box_fill`] with a white tick when `on`, else an
/// outline that brightens while `hovered` or `pressed`.
pub fn check_box(p: &egui::Painter, rect: Rect, on: bool, hovered: bool, pressed: bool, t: &Tokens) {
    p.rect_filled(rect, 2.0, check_box_fill(on, pressed, t));
    if on {
        let a = rect.left_center() + vec2(3.0, 0.5);
        let b = rect.center_bottom() + vec2(-1.0, -3.5);
        let c = rect.right_top() + vec2(-3.0, 3.5);
        let tick = Stroke::new(rect.width() * 0.125, Color32::WHITE);
        p.line_segment([a, b], tick);
        p.line_segment([b, c], tick);
    } else {
        p.rect_stroke(rect, 2.0, Stroke::new(1.5, if hovered || pressed { t.text_dim } else { t.text_faint }), StrokeKind::Inside);
    }
}

/// Small chevron for dropdowns (replaces egui's filled triangle).
pub fn chevron_icon(ui: &Ui, rect: Rect, visuals: &egui::style::WidgetVisuals, _open: bool) {
    let c = rect.center();
    let s = 3.2;
    let stroke = Stroke::new(1.3, visuals.fg_stroke.color.gamma_multiply(0.8));
    ui.painter().line_segment([c + vec2(-s, -s * 0.5), c + vec2(0.0, s * 0.5)], stroke);
    ui.painter().line_segment([c + vec2(0.0, s * 0.5), c + vec2(s, -s * 0.5)], stroke);
}

/// Photoshop-style numbers: "100", "12.5" (never "100.0").
pub fn fmt_num(v: f64) -> String {
    fmt_num_prec(v, 1)
}

/// Rounds `v` to `decimals` places (half away from zero) and drops trailing zeros, so a value is
/// never shown with more precision than it has: `0.7995` at three places is `0.8`, `2.0` is `2`
/// (#2434). [`photocraft_engine::prefs::Unit::decimals`] gives each ruler unit's places.
pub fn fmt_num_prec(v: f64, decimals: u32) -> String {
    photocraft_engine::prefs::fmt_decimals(v, decimals as usize)
}

/// Two-decimal variant of [`fmt_num`] for small ranges: `1.05`, `0.78`, `2`.
pub fn fmt_num2(v: f64) -> String {
    fmt_num_prec(v, 2)
}

/// Parses a typed number or bounded arithmetic, including parentheses and powers.
/// Preserves egui-compatible whitespace and Unicode minus handling.
pub fn parse_num(text: &str) -> Option<f64> {
    if text.len() > 1024 {
        return None;
    }
    crate::numeric_expression::parse(&clean(text))
}

/// [`parse_num`] in a field with `units`: a number may carry a unit (`2in`, `5mm`, `50%`),
/// converted into the field's unit; one that can't convert makes the text invalid.
pub(crate) fn parse_num_in(text: &str, units: crate::numeric_expression::Units) -> Option<f64> {
    if text.len() > 1024 {
        return None;
    }
    crate::numeric_expression::parse_with_units(&clean(text), Some(units))
}

/// A plain typed number, no arithmetic.
fn plain(text: &str) -> Option<f64> {
    clean(text).parse().ok()
}

fn clean(text: &str) -> String {
    text.chars().filter(|c| !c.is_whitespace()).map(|c| if c == '−' { '-' } else { c }).collect()
}

#[cfg(test)]
mod tests {
    use egui::{Key, Modifiers};

    #[test]
    fn checkbox_label_is_a_button_target_not_selectable_text() {
        use egui::CursorIcon;
        use egui_kittest::{Harness, kittest::Queryable};

        let mut h = Harness::builder().with_size(egui::vec2(300.0, 80.0)).build_ui_state(
            |ui, on: &mut bool| {
                super::checkbox(ui, on, "Show Transform Controls");
            },
            false,
        );
        h.run_steps(2);
        let label = h.get_by_label("Show Transform Controls").rect();
        h.hover_at(label.center());
        h.run_steps(2);
        assert_eq!(h.output().platform_output.cursor_icon, CursorIcon::PointingHand);
        h.get_by_label("Show Transform Controls").click();
        h.run_steps(2);
        assert!(*h.state());
    }

    /// #3208: a rounded swatch's checker is one shared, repeating, unblurred two-by-two tile on
    /// the rounded rect, scaled so a cell stays `cell` points, so nothing shows past the corners.
    #[test]
    fn rounded_swatches_clip_their_checker_to_the_corners() {
        use egui::{Pos2, Rect, pos2, vec2};
        let mut h = egui_kittest::Harness::builder().with_size(vec2(200.0, 100.0)).build_ui(|ui| {
            super::color_swatch(ui, egui::Color32::RED);
            super::color_swatch_button(ui, egui::Color32::from_rgba_unmultiplied(0, 0, 255, 128), "Blue");
        });
        crate::PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::Studio);
        h.run_steps(2);
        let tile = h.ctx.data(|d| d.get_temp::<egui::TextureHandle>(super::checker_tile_id())).expect("the tile is loaded once");
        let manager = h.ctx.tex_manager();
        let manager = manager.read();
        let meta = manager.meta(tile.id()).expect("the tile is in the texture manager");
        assert_eq!(meta.size, [2, 2]);
        assert_eq!(meta.options.wrap_mode, egui::TextureWrapMode::Repeat);
        assert_eq!(meta.options.magnification, egui::TextureFilter::Nearest);
        assert_eq!(super::checker_uv(Rect::from_min_size(Pos2::ZERO, vec2(40.0, 18.0)), 4.0), Rect::from_min_max(Pos2::ZERO, pos2(5.0, 2.25)));
        assert_eq!(
            super::checker_uv(Rect::from_min_size(Pos2::ZERO, vec2(8.0, 8.0)), 0.0),
            Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)),
            "a bad cell falls back to 4 pt"
        );
    }

    /// #3580: the shared eliding label never lays out wider than the room it is given, so a longer
    /// translation shrinks with '…' instead of overrunning its row; a short label is untouched.
    #[test]
    fn elided_label_never_exceeds_its_room() {
        let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(240.0, 60.0)).build_ui(|ui| {
            let long = super::elided_label(ui, "Preenchimento:", 24.0); // pt-br Fill:, the longest catalog
            assert!(long.rect.width() <= 25.0, "elided to the room, got {}", long.rect.width());
            let short = super::elided_label(ui, "Fill", 200.0);
            assert!(short.rect.width() < 60.0, "a short label is not padded: {}", short.rect.width());
        });
        crate::PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::Studio);
        h.run_steps(2);
    }

    #[test]
    fn spectrum_popup_style_matches_main_menu_and_preserves_studio() {
        for kind in [crate::theme::ThemeKind::Pro, crate::theme::ThemeKind::ProMedium, crate::theme::ThemeKind::Studio, crate::theme::ThemeKind::StudioLight] {
            let ctx = egui::Context::default();
            crate::theme::apply(&ctx, kind);
            let tokens = crate::theme::Tokens::for_kind(kind);
            let mut out = ctx.run_ui(Default::default(), |ui| {
                let original_padding = ui.spacing().button_padding;
                let original_hover = ui.visuals().widgets.hovered.fg_stroke;
                super::style_spectrum_popup_menu(ui);
                assert_eq!(ui.spacing().item_spacing.y, 0.0, "{kind:?}: menu rows meet so the gap is selectable");
                if tokens.pro {
                    assert_eq!(ui.spacing().button_padding, egui::vec2(10.0, 4.0));
                    assert_eq!(ui.visuals().widgets.hovered.bg_fill, tokens.accent);
                    assert_eq!(ui.visuals().widgets.hovered.fg_stroke.color, egui::Color32::WHITE);
                } else {
                    assert_eq!(ui.spacing().button_padding, original_padding);
                    assert_eq!(ui.visuals().widgets.hovered.fg_stroke, original_hover);
                }
            });
            out.textures_delta.clear();
        }
    }
    use egui_kittest::{Harness, kittest::Queryable};

    /// #3789: a plain text field is one boxed, outline-free input whose text-selection colour stays
    /// local, so it isn't recoloured for the rest of the frame (selected rows, slider trails).
    #[test]
    fn text_field_is_a_boxed_input_whose_selection_colour_stays_local() {
        use egui::accesskit::Role;
        let mut h = Harness::new_ui_state(
            |ui, s: &mut (String, egui::Color32, egui::Color32)| {
                s.1 = ui.visuals().selection.bg_fill;
                super::text_field(ui, egui::Id::new("tf"), &mut s.0, "hint", 140.0, false);
                s.2 = ui.visuals().selection.bg_fill;
            },
            (String::new(), egui::Color32::WHITE, egui::Color32::BLACK),
        );
        crate::PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::Studio);
        h.run();
        assert_eq!(h.state().1, h.state().2, "the field's selection colour does not leak into its parent");
        h.get_by_role(Role::TextInput).click();
        h.run_steps(1);
        for ch in "abc".chars() {
            h.event(egui::Event::Text(ch.to_string()));
            h.run_steps(1);
        }
        assert_eq!(h.state().0, "abc", "typing reaches the bound string");
    }

    /// Sets up a test with one value field. Its state is its value and how many times it changed.
    fn field(value: f32, range: std::ops::RangeInclusive<f32>) -> Harness<'static, (f32, u32)> {
        let mut h = Harness::new_ui_state(
            move |ui, s: &mut (f32, u32)| {
                if super::value_field(ui, &mut s.0, range.clone(), "px", 80.0).changed() {
                    s.1 += 1;
                }
            },
            (value, 0),
        );
        h.run();
        h
    }

    fn press(h: &mut Harness<'static, (f32, u32)>, mods: Modifiers, key: Key) -> f32 {
        h.key_press_modifiers(mods, key);
        h.run();
        h.state().0
    }

    fn focused(value: f32, range: std::ops::RangeInclusive<f32>) -> Harness<'static, (f32, u32)> {
        let mut h = field(value, range);
        h.get_by_role(egui::accesskit::Role::SpinButton).click();
        h.run();
        h
    }

    /// Digits have no descenders, so the number sits 1 pt above the box's centre to look centred.
    #[test]
    fn value_field_digits_sit_one_point_above_centre() {
        let mut h = Harness::new_ui_state(
            |ui, s: &mut (f32, egui::Rect)| {
                s.1 = super::value_field_in(ui, &mut s.0, 0.0..=100.0, "px", 80.0, 0.0, super::FieldOpts::default()).1;
            },
            (50.0, egui::Rect::NOTHING),
        );
        h.run();
        let number = h.get_by_role(egui::accesskit::Role::SpinButton).rect();
        let field = h.state().1;
        assert!((field.center().y - number.center().y - 1.0).abs() < 0.01, "field {field:?}, number {number:?}");
    }

    /// With the app's theme (its 24 pt interact height and 4 pt button padding), in a full-size
    /// box and a slim status-bar box alike, the number stays inside the box, 1 pt above its centre,
    /// and a long number doesn't spill out of the box's left edge (#2823).
    #[test]
    fn themed_value_field_number_sits_inside_its_box_in_every_size() {
        for kind in crate::theme::ThemeKind::ALL {
            for (w, h, v) in [(80.0, super::VALUE_FIELD_H, 50.0), (64.0, 18.0, 1600.5), (78.0, 22.0, 100.6)] {
                let mut harness = Harness::new_ui_state(
                    move |ui, s: &mut (f32, egui::Rect)| {
                        let before = ui.cursor().min;
                        super::compact_value_field(ui, &mut s.0, 0.0..=3200.0, "%", w, h);
                        s.1 = egui::Rect::from_min_size(before, egui::vec2(w, h));
                    },
                    (v, egui::Rect::NOTHING),
                );
                crate::PhotocraftApp::setup_context(&harness.ctx, kind);
                harness.run();
                let number = harness.get_by_role(egui::accesskit::Role::SpinButton).rect();
                let field = harness.state().1;
                let what = format!("{kind:?} {w}x{h}: field {field:?}, number {number:?}");
                assert!(field.contains_rect(number), "{what}");
                assert!((field.center().y - number.center().y - 1.0).abs() < 0.51, "{what}");
            }
        }
    }

    /// The number's slot ends just left of its unit, by the unit's measured width: a one-character
    /// `%` (the status bar zoom) leaves the number more room than the old fixed 16 pt, and a longer
    /// unit never sits under the number (#3040).
    #[test]
    fn value_field_number_slot_stops_at_its_measured_suffix() {
        for suffix in ["%", "px", "°", "×"] {
            let mut h = Harness::new_ui_state(
                move |ui, s: &mut (f32, egui::Rect, f32)| {
                    let before = ui.cursor().min;
                    super::compact_value_field(ui, &mut s.0, 0.0..=3200.0, suffix, 64.0, 18.0);
                    s.1 = egui::Rect::from_min_size(before, egui::vec2(64.0, 18.0));
                    s.2 = ui.painter().layout_no_wrap(suffix.to_owned(), crate::theme::mono(11.0), egui::Color32::WHITE).size().x;
                },
                (33.33, egui::Rect::NOTHING, 0.0),
            );
            h.run();
            let number = h.get_by_role(egui::accesskit::Role::SpinButton).rect();
            let (_, field, suffix_w) = *h.state();
            let suffix_left = field.right() - 6.0 - suffix_w;
            assert!(number.right() <= suffix_left, "{suffix}: number {number:?} runs under the suffix at {suffix_left}");
            assert!(suffix_left - number.right() <= 1.5, "{suffix}: {:.1} pt wasted before the suffix", suffix_left - number.right());
        }
    }

    #[test]
    fn arrow_keys_step_a_focused_field_by_one_and_shift_by_ten() {
        let mut h = focused(100.0, 0.0..=1000.0);
        assert_eq!(press(&mut h, Modifiers::NONE, Key::ArrowUp), 101.0);
        assert_eq!(press(&mut h, Modifiers::SHIFT, Key::ArrowUp), 111.0);
        assert_eq!(press(&mut h, Modifiers::NONE, Key::ArrowDown), 110.0);
        assert_eq!(press(&mut h, Modifiers::SHIFT, Key::ArrowDown), 100.0);
        assert_eq!(press(&mut h, Modifiers::COMMAND, Key::ArrowUp), 100.1);
        assert_eq!(press(&mut h, Modifiers::COMMAND | Modifiers::SHIFT, Key::ArrowDown), 100.0);
        assert_eq!(h.state().1, 6, "every step reports a change, so callers apply it");
    }

    #[test]
    fn arrow_keys_on_a_field_with_a_reversed_range_do_not_panic() {
        let mut h = focused(5.0, 10.0..=0.0);
        let _ = press(&mut h, Modifiers::NONE, Key::ArrowUp);
        let mut h = focused(5.0, f32::NAN..=10.0);
        let _ = press(&mut h, Modifiers::NONE, Key::ArrowDown);
    }

    #[test]
    fn arrow_keys_round_decimals_step_fine_fields_and_clamp() {
        for (start, mods, key, want) in [
            (55.4, Modifiers::NONE, Key::ArrowUp, 56.0),
            (55.6, Modifiers::NONE, Key::ArrowUp, 57.0),
            (55.4, Modifiers::NONE, Key::ArrowDown, 54.0),
            (55.5, Modifiers::NONE, Key::ArrowUp, 57.0),
            (55.5, Modifiers::NONE, Key::ArrowDown, 55.0),
            (55.4, Modifiers::SHIFT, Key::ArrowUp, 65.0),
            (55.47, Modifiers::COMMAND, Key::ArrowUp, 55.6),
        ] {
            assert_eq!(press(&mut focused(start, 0.0..=1000.0), mods, key), want, "{start} {mods:?} {key:?}");
        }
        let mut h = focused(0.5, 0.0..=1.0);
        assert_eq!(press(&mut h, Modifiers::NONE, Key::ArrowUp), 0.51);
        assert_eq!(press(&mut h, Modifiers::SHIFT, Key::ArrowUp), 0.61);
        assert_eq!(press(&mut h, Modifiers::COMMAND, Key::ArrowUp), 0.62);
        let mut h = focused(995.0, 0.0..=1000.0);
        assert_eq!(press(&mut h, Modifiers::SHIFT, Key::ArrowUp), 1000.0);
    }

    #[test]
    fn arrow_keys_work_out_a_typed_sum_then_step_it() {
        let mut h = focused(100.0, 0.0..=1000.0);
        h.key_press_modifiers(Modifiers::COMMAND, Key::A);
        h.event(egui::Event::Text("5+5".into()));
        h.run();
        assert_eq!(press(&mut h, Modifiers::NONE, Key::ArrowUp), 11.0);
        assert_eq!(press(&mut h, Modifiers::NONE, Key::ArrowDown), 10.0, "the field keeps focus");
    }

    #[test]
    fn arrow_keys_leave_an_unfocused_field_alone() {
        let mut h = field(100.0, 0.0..=1000.0);
        assert_eq!(press(&mut h, Modifiers::NONE, Key::ArrowUp), 100.0);
        assert_eq!(h.state().1, 0);
    }

    /// Photoshop: a unit typed into a px field converts at the document resolution on Enter, and
    /// an expression that can't convert (or divides by zero) keeps the value.
    #[test]
    fn enter_commits_a_typed_unit_expression_at_the_document_resolution() {
        let mut h = field(100.0, 0.0..=10_000.0);
        super::set_field_resolution(&h.ctx, 300.0, 72.0);
        for (text, want) in [("1in+10", 310.0), ("2.54 cm * 2", 600.0), ("50%", 600.0), ("1in/0", 600.0), ("10px", 10.0)] {
            h.get_by_role(egui::accesskit::Role::SpinButton).click();
            h.run();
            h.key_press_modifiers(Modifiers::COMMAND, Key::A);
            h.event(egui::Event::Text(text.into()));
            h.run();
            h.key_press(Key::Enter);
            h.run();
            assert_eq!(h.state().0, want, "{text}");
        }
    }

    /// Photoshop's scrubby labels: dragging the label right before a field moves its value, a
    /// step a point (Shift ×10); a label with no field right after it moves nothing.
    #[test]
    fn dragging_a_label_scrubs_the_field_right_after_it() {
        let mut h = Harness::new_ui_state(
            |ui, s: &mut (f32, f32)| {
                ui.horizontal(|ui| {
                    let l = ui.label("Size:");
                    super::scrub_label(ui, &l);
                    super::value_field(ui, &mut s.0, 0.0..=1000.0, "px", 80.0);
                });
                ui.horizontal(|ui| {
                    let l = ui.label("Mode:");
                    super::scrub_label(ui, &l);
                    ui.add_space(60.0);
                    super::value_field(ui, &mut s.1, 0.0..=1000.0, "px", 80.0);
                });
            },
            (100.0, 100.0),
        );
        h.run();
        let drag = |h: &mut Harness<'static, (f32, f32)>, label: &str, dx: f32, mods: Modifiers| {
            let at = h.get_by_label(label).rect().center();
            h.hover_at(at);
            h.run();
            h.drag_at(at);
            h.run();
            for i in 1..=4 {
                h.event(egui::Event::PointerMoved(at + egui::vec2(dx * i as f32 / 4.0, 0.0)));
                h.event(egui::Event::ModifiersChanged(mods));
                h.run();
            }
            h.drop_at(at + egui::vec2(dx, 0.0));
            h.run();
        };
        drag(&mut h, "Size:", 20.0, Modifiers::NONE);
        assert_eq!(h.state().0, 120.0);
        drag(&mut h, "Size:", -3.0, Modifiers::SHIFT);
        assert_eq!(h.state().0, 90.0);
        drag(&mut h, "Mode:", 20.0, Modifiers::NONE);
        assert_eq!(h.state().1, 100.0, "the field isn't right after the label");
    }

    /// A right-aligned OK / Cancel / Apply row as `os` draws it: labels left to right, and the
    /// row's right edge with the window's.
    fn button_row(os: egui::os::OperatingSystem) -> (Vec<String>, f32, f32) {
        use super::{ButtonRole, DialogButton};
        use egui_kittest::{Harness, kittest::Queryable};
        // Drawn from the second frame, once the theme's fonts are bound.
        let mut h = Harness::builder().with_size(egui::vec2(500.0, 80.0)).build_ui_state(
            |ui, ready: &mut bool| {
                if *ready {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                        let row = [
                            DialogButton::new(ButtonRole::Default, "OK", 84.0),
                            DialogButton::new(ButtonRole::Cancel, "Cancel", 84.0),
                            DialogButton::new(ButtonRole::Apply, "Apply", 84.0),
                        ];
                        super::dialog_buttons(ui, &row);
                    });
                }
            },
            false,
        );
        crate::PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
        h.ctx.set_os(os);
        *h.state_mut() = true;
        h.run();
        let mut drawn: Vec<(f32, f32, String)> =
            ["OK", "Cancel", "Apply"].iter().map(|l| (h.get_by_label(l).rect().left(), h.get_by_label(l).rect().right(), l.to_string())).collect();
        drawn.sort_by(|a, b| a.0.total_cmp(&b.0));
        let right = drawn.last().map_or(0.0, |d| d.1);
        (drawn.into_iter().map(|d| d.2).collect(), right, h.ctx.content_rect().right())
    }

    #[test]
    fn dialog_buttons_follow_the_platform_order() {
        use egui::os::OperatingSystem as Os;
        for (os, want) in [(Os::Windows, ["OK", "Cancel", "Apply"]), (Os::Nix, ["OK", "Cancel", "Apply"]), (Os::Mac, ["Cancel", "Apply", "OK"])] {
            let (order, right, edge) = button_row(os);
            assert_eq!(order, want, "{os:?}");
            assert!(edge - right < 20.0, "{os:?}: the row hugs the right edge ({right} of {edge})");
        }
    }

    #[test]
    fn two_decimal_numbers_trim_zeros() {
        assert_eq!(super::fmt_num2(1.05), "1.05");
        assert_eq!(super::fmt_num2(0.78), "0.78");
        assert_eq!(super::fmt_num2(0.5), "0.5");
        assert_eq!(super::fmt_num2(2.0), "2");
    }

    #[test]
    fn typed_arithmetic_evaluates() {
        use super::parse_num;
        for (text, want) in [
            ("1280*2", 2560.0),
            ("658 * 1.5", 987.0),
            ("48/3", 16.0),
            ("20*2+5-2", 43.0),
            ("2+3*4", 14.0),
            ("10/4*2", 5.0),
            ("-5+3", -2.0),
            ("5--3", 8.0),
            ("2*-3+1", -5.0),
            ("−4", -4.0),
            ("1 234", 1234.0),
            ("1e3/2", 500.0),
            ("(2+3)*4", 20.0),
            ("2^3", 8.0),
            ("10%3", 1.0),
        ] {
            assert_eq!(parse_num(text), Some(want), "{text}");
        }
        for text in ["", "abc", "5+", "*2", "4/0", "0/0", "1+*2", "(2+3", "1e400"] {
            assert_eq!(parse_num(text), None, "{text}");
        }
    }

    /// Types `text` into a value field holding `start`, then presses `key`. Returns the value an OK
    /// button that also fires on Enter saw, as dialogs read it, else the field's value. `round`
    /// makes the caller round the value every frame, as pixel fields do.
    fn type_and_press(start: f32, text: &str, key: egui::Key, round: bool) -> f32 {
        use egui_kittest::{Harness, kittest::Queryable};
        let mut h = Harness::builder().with_size(egui::vec2(300.0, 100.0)).build_ui_state(
            move |ui, (v, ok): &mut (f32, Option<f32>)| {
                super::value_field(ui, v, 0.0..=300000.0, "px", 90.0);
                if round {
                    *v = v.round();
                }
                if ui.button(tl!("OK")).clicked() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    *ok = Some(*v);
                }
            },
            (start, None),
        );
        h.get_by_role(egui::accesskit::Role::SpinButton).click();
        h.run();
        for c in text.chars() {
            h.event(egui::Event::Text(c.to_string()));
            h.run();
        }
        h.key_press(key);
        h.run();
        let (v, ok) = *h.state();
        ok.unwrap_or(v)
    }

    /// Plain digits apply as they're typed; arithmetic doesn't report a change until it's committed.
    #[test]
    fn value_field_holds_arithmetic_until_committed() {
        use egui_kittest::{Harness, kittest::Queryable};
        let mut h = Harness::builder().with_size(egui::vec2(300.0, 100.0)).build_ui_state(
            |ui, (v, changes): &mut (f32, u32)| {
                if super::value_field(ui, v, 0.0..=100.0, "%", 90.0).changed() {
                    *changes += 1;
                }
            },
            (100.0, 0),
        );
        h.get_by_role(egui::accesskit::Role::SpinButton).click();
        h.run();
        for (c, want) in [('5', (5.0, 1)), ('0', (50.0, 2)), ('/', (50.0, 2)), ('2', (50.0, 2))] {
            h.event(egui::Event::Text(c.to_string()));
            h.run();
            assert_eq!(*h.state(), want, "after {c}");
        }
        h.key_press(egui::Key::Enter);
        h.run();
        assert_eq!(*h.state(), (25.0, 3));
    }

    #[test]
    fn value_field_applies_typed_arithmetic() {
        use egui::Key::{Enter, Tab};
        assert_eq!(type_and_press(500.0, "1280*2", Tab, false), 2560.0);
        assert_eq!(type_and_press(500.0, "1280*2", Enter, false), 2560.0);
        // `5/2` alone would round to 3 under the caller; the whole expression still applies.
        assert_eq!(type_and_press(1.0, "5/2*2", Tab, true), 5.0);
        assert_eq!(type_and_press(1.0, "1280/3*2", Enter, true), 853.0);
    }

    #[test]
    fn numbers_drop_trailing_zero() {
        // #2434: the unit keys map to Photoshop's places, and the shared formatter trims zeros.
        assert_eq!(super::unit_decimals("px"), 0);
        assert_eq!(super::unit_decimals("pt"), 1);
        assert_eq!(super::unit_decimals("mm"), 2);
        assert_eq!(super::unit_decimals("cm"), 2);
        assert_eq!(super::unit_decimals("pica"), 2);
        assert_eq!(super::unit_decimals("in"), 3);
        assert_eq!(super::unit_decimals("nonsense"), 0);
        assert_eq!(super::fmt_num_prec(0.7995, 3), "0.8");
        assert_eq!(super::fmt_num_prec(2.5, 0), "3");
        assert_eq!(super::fmt_num(100.0), "100");
        assert_eq!(super::fmt_num(12.46), "12.5");
        assert_eq!(super::fmt_num(-3.0), "-3");
    }

    #[test]
    fn dropdown_opens_with_focus_and_arrow_keys_change_the_value() {
        use egui::accesskit::Role;
        use egui_kittest::{Harness, kittest::Queryable};

        let mut h = Harness::builder().with_size(egui::vec2(300.0, 100.0)).build_ui_state(
            |ui, selected: &mut usize| {
                let options = [(0, "Normal"), (1, "Multiply"), (2, "Screen")];
                super::dropdown(ui, "blend-mode", selected, &options, 120.0);
            },
            0,
        );
        h.get_by_role(Role::ComboBox).click();
        h.run();
        h.key_press(egui::Key::ArrowDown);
        h.run();
        assert_eq!(*h.state(), 1);
        h.key_press(egui::Key::ArrowUp);
        h.run();
        assert_eq!(*h.state(), 0);
    }

    /// Scrolling rows under the pointer doesn't widen the list: it is as wide as the widest
    /// option hovered from the start (#2607).
    #[test]
    fn dropdown_list_keeps_its_width_while_scrolling() {
        use photocraft_color::BlendMode;
        let mut h = Harness::builder().with_size(egui::vec2(400.0, 700.0)).build_ui_state(
            |ui, mode: &mut BlendMode| {
                let opts: Vec<(BlendMode, &str)> = BlendMode::LAYER_MODES.iter().map(|m| (*m, m.label())).collect();
                super::dropdown(ui, "brush-mode", mode, &opts, 96.0);
            },
            BlendMode::Normal,
        );
        // The app's theme strokes plain rows, which is what made hovered rows wider.
        crate::theme::apply(&h.ctx, crate::theme::ThemeKind::ProMedium);
        h.run();
        h.get_by_role(egui::accesskit::Role::ComboBox).click();
        h.run();
        let list = |h: &Harness<'_, BlendMode>| h.ctx.memory(|m| m.layer_ids().filter(|l| l.order == egui::Order::Foreground).find_map(|l| m.area_rect(l.id)));
        let Some(open) = list(&h) else { panic!("the list is open") };
        for _ in 0..30 {
            h.hover_at(open.center());
            h.event(egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Line,
                delta: egui::vec2(0.0, -1.0),
                phase: egui::TouchPhase::Move,
                modifiers: Modifiers::NONE,
            });
            h.run();
            assert_eq!(list(&h).map(|r| r.width()), Some(open.width()));
        }
    }

    const MODES: [(usize, &str); 4] = [(0, "Normal"), (1, "Dissolve"), (2, "Darken"), (3, "Multiply")];

    /// A wheel dropdown (or a plain one) of [`MODES`], whose state is the chosen index.
    fn mode_dropdown(wheel: bool) -> Harness<'static, usize> {
        Harness::builder().with_size(egui::vec2(300.0, 100.0)).build_ui_state(
            move |ui, selected: &mut usize| {
                if wheel {
                    super::dropdown_wheel_hovered(ui, "blend-mode", selected, &MODES, 120.0);
                } else {
                    super::dropdown(ui, "blend-mode", selected, &MODES, 120.0);
                }
            },
            0,
        )
    }

    /// One wheel event over the dropdown: negative `delta.y` is down.
    fn wheel<T>(h: &mut Harness<'_, T>, unit: egui::MouseWheelUnit, delta: egui::Vec2, modifiers: Modifiers) {
        let at = h.get_by_role(egui::accesskit::Role::ComboBox).rect().center();
        h.event(egui::Event::ModifiersChanged(modifiers));
        h.hover_at(at);
        h.run();
        h.event(egui::Event::MouseWheel { unit, delta, phase: egui::TouchPhase::Move, modifiers });
        h.run();
        h.event(egui::Event::ModifiersChanged(Modifiers::NONE));
        h.run();
    }

    #[test]
    fn shift_wheel_steps_one_mode_per_notch() {
        let mut h = mode_dropdown(true);
        wheel(&mut h, egui::MouseWheelUnit::Line, egui::vec2(0.0, -1.0), Modifiers::SHIFT);
        assert_eq!(*h.state(), 1, "not ten like a slider");
        // Some systems turn Shift+wheel into a horizontal scroll.
        wheel(&mut h, egui::MouseWheelUnit::Line, egui::vec2(-1.0, 0.0), Modifiers::SHIFT);
        assert_eq!(*h.state(), 2);
    }

    #[test]
    fn fractional_wheel_deltas_add_up_to_whole_mode_steps() {
        let mut h = mode_dropdown(true);
        wheel(&mut h, egui::MouseWheelUnit::Line, egui::vec2(0.0, -0.5), Modifiers::NONE);
        assert_eq!(*h.state(), 0, "half a notch");
        wheel(&mut h, egui::MouseWheelUnit::Line, egui::vec2(0.0, -0.5), Modifiers::NONE);
        assert_eq!(*h.state(), 1);
        let per_line = h.ctx.options(|o| o.input_options.line_scroll_speed);
        wheel(&mut h, egui::MouseWheelUnit::Point, egui::vec2(0.0, -per_line), Modifiers::NONE);
        assert_eq!(*h.state(), 2, "a line's worth of points is a notch");
    }

    #[test]
    fn the_wheel_over_a_wheel_dropdown_does_not_scroll_its_container() {
        let mut h = Harness::builder().with_size(egui::vec2(300.0, 100.0)).build_ui_state(
            |ui, selected: &mut usize| {
                egui::ScrollArea::vertical().max_height(80.0).show(ui, |ui| {
                    super::dropdown_wheel_hovered(ui, "blend-mode", selected, &MODES, 120.0);
                    ui.add_space(400.0);
                });
            },
            0,
        );
        let top = h.get_by_role(egui::accesskit::Role::ComboBox).rect().top();
        wheel(&mut h, egui::MouseWheelUnit::Line, egui::vec2(0.0, -1.0), Modifiers::NONE);
        for _ in 0..10 {
            h.run();
        }
        assert_eq!(*h.state(), 1);
        assert_eq!(h.get_by_role(egui::accesskit::Role::ComboBox).rect().top(), top, "the container didn't scroll");
    }

    #[test]
    fn the_wheel_leaves_a_plain_dropdown_alone() {
        let mut h = mode_dropdown(false);
        wheel(&mut h, egui::MouseWheelUnit::Line, egui::vec2(0.0, -1.0), Modifiers::NONE);
        assert_eq!(*h.state(), 0);
    }
}

/// A compact colour popup with the shared desktop eyedropper. Keep egui's colour cache so hue
/// survives black/white and alpha edits, and keep linear and encoded call sites distinct.
pub fn color_edit_button_srgba(ui: &mut Ui, color: &mut Color32) -> Response {
    color_edit_button(ui, color, egui::color_picker::Alpha::BlendOrAdditive)
}
pub(crate) fn color_swatch(ui: &mut Ui, color: Color32) -> Response {
    let t = Tokens::get(ui.ctx());
    let (rect, response) = ui.allocate_exact_size(ui.spacing().interact_size, Sense::click());
    response.widget_info(|| egui::WidgetInfo::new(egui::WidgetType::ColorButton));
    checker_rounded(ui.painter(), rect, 4.0, t.radius_sm);
    ui.painter().rect_filled(rect.shrink(1.0), t.radius_sm, color);
    ui.painter().rect_stroke(rect, t.radius_sm, Stroke::new(1.0, t.field_border), StrokeKind::Inside);
    response
}
/// A colour swatch that only reports clicks (no popup): for colours edited in PhotoCraft's own
/// Color Picker dialog. `label` names it for hover text and accessibility.
pub fn color_swatch_button(ui: &mut Ui, color: Color32, label: &str) -> Response {
    let t = Tokens::get(ui.ctx());
    let (rect, response) = ui.allocate_exact_size(ui.spacing().interact_size, Sense::click());
    response.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::ColorButton, true, label));
    checker_rounded(ui.painter(), rect, 4.0, t.radius_sm);
    ui.painter().rect_filled(rect.shrink(1.0), t.radius_sm, color);
    ui.painter().rect_stroke(rect, t.radius_sm, Stroke::new(1.0, t.field_border), StrokeKind::Inside);
    response.on_hover_text(label)
}
fn color_edit_button(ui: &mut Ui, color: &mut Color32, alpha: egui::color_picker::Alpha) -> Response {
    let mut response = color_swatch(ui, *color);
    let id = response.id.with("screen-color");
    if let Some(rgb) = crate::screen_picker::take(ui.ctx(), id) {
        *color = sampled_color(rgb, *color, alpha);
        response.mark_changed();
    }
    swatch_popup(&response).show(|ui| {
        if egui::color_picker::color_picker_color32(ui, color, alpha) {
            response.mark_changed();
        }
        if let Some(rgb) = crate::screen_picker::button(ui, id) {
            *color = sampled_color(rgb, *color, alpha);
            response.mark_changed();
        }
    });
    response
}
fn sampled_color(rgb: [f32; 3], original: Color32, alpha: egui::color_picker::Alpha) -> Color32 {
    let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    if matches!(alpha, egui::color_picker::Alpha::BlendOrAdditive) && original.is_additive() {
        return Color32::from_rgb_additive(byte(rgb[0]), byte(rgb[1]), byte(rgb[2]));
    }
    Color32::from_rgba_unmultiplied(
        byte(rgb[0]),
        byte(rgb[1]),
        byte(rgb[2]),
        if matches!(alpha, egui::color_picker::Alpha::Opaque) { 255 } else { original.a() },
    )
}
pub fn color_edit_button_srgb(ui: &mut Ui, rgb: &mut [u8; 3]) -> Response {
    let mut color = Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
    let response = color_edit_button(ui, &mut color, egui::color_picker::Alpha::Opaque);
    if response.changed() {
        *rgb = [color.r(), color.g(), color.b()];
    }
    response
}
/// Like egui's float RGB widget, this entry point stores linear RGB (not encoded hex values).
pub fn color_edit_button_rgb(ui: &mut Ui, rgb: &mut [f32; 3]) -> Response {
    let mut response = color_swatch(ui, Color32::from(egui::Rgba::from_rgb(rgb[0], rgb[1], rgb[2])));
    let id = response.id.with("screen-color");
    let to_linear = |color: [f32; 3]| color.map(egui::ecolor::linear_from_gamma);
    if let Some(color) = crate::screen_picker::take(ui.ctx(), id) {
        *rgb = to_linear(color);
        response.mark_changed();
    }
    let rgba = egui::Rgba::from_rgb(rgb[0], rgb[1], rgb[2]);
    let cache = response.id.with("linear-hsva");
    let mut hsva = ui
        .ctx()
        .data(|d| d.get_temp::<(egui::Rgba, egui::ecolor::Hsva)>(cache))
        .filter(|(previous, _)| *previous == rgba)
        .map(|(_, hsva)| hsva)
        .unwrap_or_else(|| egui::ecolor::Hsva::from(rgba));
    swatch_popup(&response).show(|ui| {
        if egui::color_picker::color_picker_hsva_2d(ui, &mut hsva, egui::color_picker::Alpha::Opaque) {
            let color = egui::Rgba::from(hsva);
            *rgb = [color.r(), color.g(), color.b()];
            response.mark_changed();
        }
        if let Some(color) = crate::screen_picker::button(ui, id) {
            *rgb = to_linear(color);
            hsva = egui::ecolor::Hsva::from(egui::Rgba::from_rgb(rgb[0], rgb[1], rgb[2]));
            response.mark_changed();
        }
    });
    ui.ctx().data_mut(|d| d.insert_temp(cache, (egui::Rgba::from_rgb(rgb[0], rgb[1], rgb[2]), hsva)));
    response
}

#[cfg(test)]
mod screen_color_tests {
    use super::*;
    use crate::screen_picker::{Capture, Pending};
    use egui_kittest::{Harness, kittest::Queryable};
    #[test]
    fn float_picker_keeps_precision_and_converts_screen_srgb_to_linear() {
        let services = crate::Services {
            screen_pick: Some(Box::new(|_| {
                let (tx, receiver) = std::sync::mpsc::channel();
                tx.send(Ok(Capture::Color(Some([0.5, 0.25, 1.0])))).unwrap();
                Pending { receiver, cancelled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)) }
            })),
            ..Default::default()
        };
        let original = [0.123456, 0.234567, 0.345678];
        let app = crate::PhotocraftApp::new(photocraft_engine::Session::new(), services);
        let mut h = Harness::builder().with_size(vec2(600.0, 500.0)).build_ui_state(
            |ui, state: &mut (crate::PhotocraftApp, [f32; 3])| {
                crate::screen_picker::tick(&mut state.0, ui.ctx());
                color_edit_button_rgb(ui, &mut state.1);
            },
            (app, original),
        );
        h.run_steps(2);
        assert_eq!(h.state().1, original);
        h.get_by_role(egui::accesskit::Role::ColorWell).click();
        h.run_steps(3);
        for (actual, expected) in h.state().1.iter().zip(original) {
            assert!((actual - expected).abs() < 0.000001, "opening the float popup must not quantize to 8 bits");
        }
        h.get_by_label("Pick screen color").click();
        h.run_steps(3);
        for (actual, expected) in h.state().1.iter().zip([0.21404114, 0.05087609, 1.0]) {
            assert!((actual - expected).abs() < 0.000001);
        }
    }
    #[test]
    fn compact_picker_delivers_screen_color_after_popup_closes_and_keeps_alpha() {
        for (original, expected) in [
            (Color32::from_rgba_unmultiplied(40, 50, 60, 128), Color32::from_rgba_unmultiplied(255, 0, 128, 128)),
            (Color32::from_rgb_additive(40, 50, 60), Color32::from_rgb_additive(255, 0, 128)),
        ] {
            let (tx, rx) = std::sync::mpsc::channel();
            let mut receiver = Some(rx);
            let services = crate::Services {
                screen_pick: Some(Box::new(move |_| Pending {
                    receiver: receiver.take().unwrap(),
                    cancelled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
                })),
                ..Default::default()
            };
            let app = crate::PhotocraftApp::new(photocraft_engine::Session::new(), services);
            let mut h = Harness::builder().with_size(vec2(600.0, 500.0)).build_ui_state(
                |ui, state: &mut (crate::PhotocraftApp, Color32, bool)| {
                    crate::screen_picker::tick(&mut state.0, ui.ctx());
                    state.2 = color_edit_button_srgba(ui, &mut state.1).changed();
                },
                (app, original, false),
            );
            h.get_by_role(egui::accesskit::Role::ColorWell).click();
            h.run_steps(3);
            h.get_by_label("Pick screen color").click();
            h.run_steps(2);
            assert_eq!(h.state().1, original);
            egui::Popup::close_all(&h.ctx);
            tx.send(Ok(Capture::Color(Some([1.0, 0.0, 0.5])))).unwrap();
            h.run_steps(1);
            assert!(h.state().2);
            assert_eq!(h.state().1.a(), original.a());
            assert_eq!(h.state().1, expected);
            h.run_steps(1);
            assert!(!h.state().2, "the pick commits once");
        }
    }
}
