//! The window's own title bar on Windows and Linux, where the window has no OS decorations
//! ([`PhotocraftApp::custom_titlebar`]): the app's top bar (`panels::title_bar`) is the title bar,
//! as in Photoshop on Windows. The free gap between its menus and controls drags the window and a
//! double-click there maximizes it (`panels::title_bar`, on every platform); caption buttons
//! minimize, maximize/restore and close, and thin invisible zones along the window's edges resize
//! it.
//!
//! The caption buttons follow the desktop ([`CaptionStyle`], from `Services::caption_style`):
//! Windows' 46 pt buttons at the right end by default; on Linux libadwaita's round buttons (GNOME),
//! Breeze's (KDE) or a plain round set elsewhere, with the set of buttons and their side taken
//! from the desktop's button layout. The shapes are drawn here, not read from the icon theme.

use egui::{Color32, CursorIcon, Id, LayerId, Order, PointerButton, Pos2, Rect, ResizeDirection, Sense, Stroke, Ui, ViewportCommand, pos2, vec2};
use serde_json::json;

use crate::PhotocraftApp;
use crate::theme::Tokens;

/// One Windows caption button: as wide as the native ones.
pub const BUTTON_WIDTH: f32 = 46.0;
/// Resize zone thickness along an edge, and the side of a corner zone. 5 pt was hard to find
/// (#1581); 8 pt is about a native resize border, and only trims the toolbar buttons' outer 3 pt.
const EDGE: f32 = 8.0;
const CORNER: f32 = 12.0;
const MODAL_DRAG_AREA: &str = "titlebar-modal-drag-area";
/// Breeze's Close button on hover and press (KDE's Breeze decoration colours).
const BREEZE_CLOSE_HOVER: Color32 = Color32::from_rgb(0xff, 0x98, 0xa2);
const BREEZE_CLOSE_PRESSED: Color32 = Color32::from_rgb(0x6d, 0x22, 0x29);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Caption {
    Minimize,
    Maximize,
    Close,
}

/// Windows' order, left to right.
const CAPTIONS: [Caption; 3] = [Caption::Minimize, Caption::Maximize, Caption::Close];

/// Which desktop's caption buttons to look like.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum CaptionFamily {
    /// Windows 10/11: full-height buttons 46 pt wide; Close turns red on hover.
    #[default]
    Windows,
    /// GNOME (libadwaita): 24 pt round buttons on a faint disc that darkens on hover and press.
    Adwaita,
    /// KDE (Breeze decoration): smaller round buttons, bare at rest, inverted on hover.
    Breeze,
    /// Other Linux desktops: libadwaita's buttons without the disc at rest.
    Generic,
}

/// The title bar's unused gap, saved by `panels::title_bar` for drags that begin while a modal
/// backdrop is on top of the normal title-bar widgets.
pub(crate) fn modal_drag_area_id() -> Id {
    Id::new(MODAL_DRAG_AREA)
}

/// Restore window dragging over the title bar's empty gap after a modal has blocked the normal
/// title-bar layer. The hit target stays limited to the same gap, so menus and caption controls
/// remain owned by their widgets.
pub(crate) fn modal_drag_overlay(ctx: &egui::Context) {
    let Some(area) = ctx.data(|d| d.get_temp::<Rect>(modal_drag_area_id())) else { return };
    if !area.is_positive() || egui::Popup::is_any_open(ctx) || !crate::dialogs::titlebar_drag_area_is_clear(ctx, area) {
        return;
    }
    egui::Area::new(Id::new("titlebar-modal-drag-overlay")).order(Order::Tooltip).fixed_pos(area.min).movable(false).sense(Sense::hover()).show(ctx, |ui| {
        let response = ui.allocate_rect(area, Sense::click_and_drag());
        if response.drag_started() {
            ctx.send_viewport_cmd(ViewportCommand::StartDrag);
        }
        if response.double_clicked() {
            let max = ctx.input(|i| i.viewport().maximized.unwrap_or(false));
            ctx.send_viewport_cmd(ViewportCommand::Maximized(!max));
        }
    });
}

impl CaptionFamily {
    /// `windows`, `adwaita`, `breeze` or `generic`, in any case.
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "windows" => Some(Self::Windows),
            "adwaita" => Some(Self::Adwaita),
            "breeze" => Some(Self::Breeze),
            "generic" => Some(Self::Generic),
            _ => None,
        }
    }

    /// The button side, gap on each side of it, and gap between the outermost button and the
    /// window's edge. libadwaita: 24 pt buttons 13 pt apart, 10 pt from the edge (the gap is split
    /// between neighbours, so the hit areas touch). Breeze sizes buttons by the font (about 1.3 em,
    /// 18 pt at KDE's default 10 pt font), half a button apart, 8 pt from the edge.
    fn metrics(self) -> (f32, f32, f32) {
        match self {
            Self::Windows => (BUTTON_WIDTH, 0.0, 0.0),
            Self::Adwaita | Self::Generic => (24.0, 6.5, 10.0),
            Self::Breeze => (18.0, 4.5, 8.0),
        }
    }
}

/// The caption buttons' look and placement.
#[derive(Clone, PartialEq, Debug)]
pub struct CaptionStyle {
    pub family: CaptionFamily,
    /// The buttons at the bar's left end, left to right.
    pub left: Vec<Caption>,
    /// The buttons at the bar's right end, left to right.
    pub right: Vec<Caption>,
    /// The round buttons' glyphs from the desktop's icon theme; `None`, or an icon it lacks, draws
    /// the family's own.
    pub icons: Option<CaptionIcons>,
}

/// A caption button glyph, as the icon themes name them (`window-restore-symbolic`, …).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CaptionIcon {
    Minimize,
    Maximize,
    Restore,
    Close,
}

/// Renders an icon as a square coverage mask (`side` × `side` bytes, row by row), or `None`.
pub type RenderIconFn = dyn Fn(CaptionIcon, u32) -> Option<Vec<u8>> + Send + Sync;

/// Glyphs from an icon theme (Linux: `window-*-symbolic`), drawn in the button's ink colour as
/// symbolic icons are.
#[derive(Clone)]
pub struct CaptionIcons {
    /// The theme's name; the cached textures follow it.
    pub theme: String,
    pub render: std::sync::Arc<RenderIconFn>,
}

impl PartialEq for CaptionIcons {
    fn eq(&self, other: &Self) -> bool {
        self.theme == other.theme && std::sync::Arc::ptr_eq(&self.render, &other.render)
    }
}

impl std::fmt::Debug for CaptionIcons {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CaptionIcons").field("theme", &self.theme).finish_non_exhaustive()
    }
}

impl Default for CaptionStyle {
    fn default() -> Self {
        Self { family: CaptionFamily::Windows, left: vec![], right: CAPTIONS.to_vec(), icons: None }
    }
}

impl CaptionStyle {
    /// `family` with the buttons a GTK decoration layout names (`gtk-decoration-layout`, GNOME's
    /// `button-layout`, e.g. `appmenu:minimize,maximize,close`): those before the colon go left,
    /// those after it right; without a colon all go left, as in GTK. Names other than `minimize`,
    /// `maximize` and `close` (`appmenu`, `icon`, `spacer`, …) are skipped, and a button named
    /// twice is placed once.
    pub fn from_gtk_layout(family: CaptionFamily, layout: &str) -> Self {
        let (left, right) = layout.split_once(':').unwrap_or((layout, ""));
        let mut seen = vec![];
        let mut side = |names: &str| {
            let mut v = vec![];
            for name in names.split(',') {
                let c = match name.trim() {
                    "minimize" => Caption::Minimize,
                    "maximize" => Caption::Maximize,
                    "close" => Caption::Close,
                    _ => continue,
                };
                if !seen.contains(&c) {
                    seen.push(c);
                    v.push(c);
                }
            }
            v
        };
        let left = side(left);
        let right = side(right);
        Self { family, left, right, icons: None }
    }
}

/// The caption style the platform reports, or Windows' when it reports none.
fn style(app: &PhotocraftApp, ctx: &egui::Context) -> CaptionStyle {
    app.services.caption_style.as_ref().map(|read| read(ctx)).unwrap_or_default()
}

/// The title bar's height: `base`, or with GNOME's buttons at least libadwaita's header bar
/// (46 pt, plus the bar's 1 pt separator), so the bar is as tall as its neighbours' title bars.
pub fn bar_height(app: &PhotocraftApp, ctx: &egui::Context, base: f32) -> f32 {
    if app.custom_titlebar && style(app, ctx).family == CaptionFamily::Adwaita { base.max(47.0) } else { base }
}

/// The width the caption buttons take at the bar's (left, right) end: zero for a side without
/// buttons, and for both without the custom title bar.
pub fn reserved(app: &PhotocraftApp, ctx: &egui::Context) -> (f32, f32) {
    if !app.custom_titlebar {
        return (0.0, 0.0);
    }
    let s = style(app, ctx);
    let width = |side: &[Caption]| slots(s.family, side, 0.0, true).last().map_or(0.0, |(_, r, _)| r.1);
    (width(&s.left), width(&s.right))
}

/// Each button's horizontal extent (left, right) and face centre on a side starting at `from`, in
/// order. `outer_left` puts the edge gap on the first button's left (the left end) instead of the
/// last button's right.
fn slots(family: CaptionFamily, side: &[Caption], from: f32, outer_left: bool) -> Vec<(Caption, (f32, f32), f32)> {
    let (size, gap, end) = family.metrics();
    let mut x = from;
    let n = side.len();
    side.iter()
        .enumerate()
        .map(|(i, &c)| {
            let before = if outer_left && i == 0 { end } else { gap };
            let after = if !outer_left && i + 1 == n { end } else { gap };
            let slot = (x, x + before + size + after);
            let centre = x + before + size / 2.0;
            x = slot.1;
            (c, slot, centre)
        })
        .collect()
}

/// Every caption button's hit rect (full bar height, flush with the window's edge) and face
/// centre, for a bar `bar` between the window's `left` and `right` edges.
fn layout(s: &CaptionStyle, bar: Rect, left: f32, right: f32) -> Vec<(Caption, Rect, Pos2)> {
    let rect = |(c, (l, r), centre): (Caption, (f32, f32), f32)| {
        (c, Rect::from_min_max(pos2(l, bar.top()), pos2(r, bar.bottom() - 1.0)), pos2(centre, bar.center().y))
    };
    let lefts = slots(s.family, &s.left, left, true);
    let width = slots(s.family, &s.right, 0.0, false).last().map_or(0.0, |(_, r, _)| r.1);
    let rights = slots(s.family, &s.right, right - width, false);
    lefts.into_iter().chain(rights).map(rect).collect()
}

/// Caption button ids (stable, so tests and agents can find them).
fn caption_id(c: Caption) -> Id {
    Id::new(("titlebar-caption", c as u8))
}

fn maximized(ctx: &egui::Context) -> bool {
    ctx.input(|i| i.viewport().maximized.unwrap_or(false))
}

fn toggle_maximize(ctx: &egui::Context) {
    ctx.send_viewport_cmd(ViewportCommand::Maximized(!maximized(ctx)));
}

/// The caption buttons at the ends of `bar` (the title bar's outer rect), full bar height, flush
/// with the window's edges (even if the bar's contents run wider) so a maximized window's corner
/// hits the outermost one. Close runs File › Exit, which asks about unsaved documents first. On
/// Linux the buttons drag the window too, as GTK's and KDE's do.
pub fn caption_buttons(app: &mut PhotocraftApp, ui: &mut Ui, bar: Rect) {
    let t = Tokens::get(ui.ctx());
    let s = style(app, ui.ctx());
    let max = maximized(ui.ctx());
    let content = ui.ctx().content_rect();
    let windows = s.family == CaptionFamily::Windows;
    let mut clicked = None;
    for (c, r, centre) in layout(&s, bar, bar.left().max(content.left()), bar.right().min(content.right())) {
        let resp = ui.interact(r, caption_id(c), if windows { Sense::click() } else { Sense::click_and_drag() });
        if resp.drag_started_by(PointerButton::Primary) {
            ui.ctx().send_viewport_cmd(ViewportCommand::StartDrag);
        }
        let pressed = resp.is_pointer_button_down_on() && resp.hovered();
        if windows {
            let close = c == Caption::Close;
            let fill = match (resp.is_pointer_button_down_on(), resp.hovered()) {
                (true, _) if close => t.caption_close.gamma_multiply(0.8),
                (_, true) if close => t.caption_close,
                (true, _) => t.pressed,
                (_, true) => t.hover,
                _ => Color32::TRANSPARENT,
            };
            ui.painter().rect_filled(r, 0.0, fill);
            let ink = if close && resp.hovered() { t.caption_close_text } else { t.icon };
            paint_windows_glyph(ui, c, max, r.center(), Stroke::new(1.0, ink));
        } else {
            paint_round(ui, &t, &s, c, max, centre, resp.hovered(), pressed);
        }
        let tip = match c {
            Caption::Minimize => tl!("Minimize"),
            Caption::Maximize if max => tl!("Restore"),
            Caption::Maximize => tl!("Maximize"),
            Caption::Close => tl!("Close"),
        };
        resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, tip));
        if resp.on_hover_text(tip).clicked() {
            clicked = Some(c);
        }
    }
    match clicked {
        Some(Caption::Minimize) => ui.ctx().send_viewport_cmd(ViewportCommand::Minimized(true)),
        Some(Caption::Maximize) => toggle_maximize(ui.ctx()),
        Some(Caption::Close) => {
            let ctx = ui.ctx().clone();
            let _ = crate::menus::invoke(app, &ctx, "file.exit", json!({}));
        }
        None => {}
    }
}

/// The 10 pt line glyphs of the Windows caption buttons.
fn paint_windows_glyph(ui: &Ui, c: Caption, maximized: bool, at: egui::Pos2, s: Stroke) {
    let p = ui.painter();
    let g = Rect::from_center_size(at, vec2(10.0, 10.0));
    match c {
        Caption::Minimize => {
            p.line_segment([g.left_center(), g.right_center()], s);
        }
        // Restore: a front square with the corner of the one behind it.
        Caption::Maximize if maximized => {
            let front = Rect::from_min_max(g.min + vec2(0.0, 2.0), g.max - vec2(2.0, 0.0));
            p.rect_stroke(front, 0.0, s, egui::StrokeKind::Middle);
            p.line(vec![g.min + vec2(2.0, 2.0), g.min + vec2(2.0, 0.0), g.right_top(), g.max - vec2(0.0, 2.0), g.max - vec2(2.0, 2.0)], s);
        }
        Caption::Maximize => {
            p.rect_stroke(g, 0.0, s, egui::StrokeKind::Middle);
        }
        Caption::Close => {
            p.line_segment([g.left_top(), g.right_bottom()], s);
            p.line_segment([g.right_top(), g.left_bottom()], s);
        }
    }
}

/// A round Linux caption button centred at `centre`: its disc, then its glyph on a 16 pt grid.
/// The disc is the icon colour at a fraction of its opacity (libadwaita's `alpha(currentColor)`),
/// so it suits light and dark themes alike; Breeze instead fills it with the icon colour on hover
/// and draws the glyph in the bar's colour.
#[allow(clippy::too_many_arguments)]
fn paint_round(ui: &Ui, t: &Tokens, s: &CaptionStyle, c: Caption, maximized: bool, centre: Pos2, hovered: bool, pressed: bool) {
    let family = s.family;
    let (size, _, _) = family.metrics();
    // Whole pixels, so the glyphs' edges stay sharp.
    let ppp = ui.ctx().pixels_per_point();
    let centre = pos2((centre.x * ppp).round() / ppp, (centre.y * ppp).round() / ppp);
    let close = c == Caption::Close;
    let (fill, ink) = if family == CaptionFamily::Breeze {
        match (hovered, pressed) {
            (_, true) if close => (BREEZE_CLOSE_PRESSED, t.chrome),
            (true, _) if close => (BREEZE_CLOSE_HOVER, t.chrome),
            (_, true) => (t.icon.gamma_multiply(0.5), t.chrome),
            (true, _) => (t.icon, t.chrome),
            _ => (Color32::TRANSPARENT, t.icon),
        }
    } else {
        let rest = if family == CaptionFamily::Adwaita { 0.10 } else { 0.0 };
        let target = if pressed {
            0.30
        } else if hovered {
            0.15
        } else {
            rest
        };
        // libadwaita eases the disc over 0.2 s.
        let a = ui.ctx().animate_value_with_time(caption_id(c).with("disc"), target, 0.2);
        (t.icon.gamma_multiply(a), t.icon)
    };
    let p = ui.painter();
    if fill.a() > 0 {
        p.circle_filled(centre, size / 2.0, fill);
    }
    // The glyph grid: 16 pt, or 2 pt less than a smaller button.
    let k = (size - 2.0).min(16.0) / 16.0;
    let icon = match c {
        Caption::Minimize => CaptionIcon::Minimize,
        Caption::Maximize if maximized => CaptionIcon::Restore,
        Caption::Maximize => CaptionIcon::Maximize,
        Caption::Close => CaptionIcon::Close,
    };
    if let Some(icons) = &s.icons
        && let Some(tex) = theme_icon(ui.ctx(), icons, icon, (16.0 * k * ppp).round() as u32)
    {
        let side = tex.size_vec2() / ppp;
        let min = centre - side / 2.0;
        let rect = Rect::from_min_size(pos2((min.x * ppp).round() / ppp, (min.y * ppp).round() / ppp), side);
        p.image(tex.id(), rect, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), ink);
        return;
    }
    let at = |x: f32, y: f32| centre + vec2(x - 8.0, y - 8.0) * k;
    if family == CaptionFamily::Breeze {
        // Breeze: 1 pt chevrons (down to minimize, up to maximize), a diamond to restore, an X.
        let s = Stroke::new(1.0, ink);
        match c {
            Caption::Minimize => {
                p.line(vec![at(2.5, 5.5), at(8.0, 11.0), at(13.5, 5.5)], s);
            }
            Caption::Maximize if maximized => {
                p.add(egui::Shape::closed_line(vec![at(8.0, 3.0), at(13.0, 8.0), at(8.0, 13.0), at(3.0, 8.0)], s));
            }
            Caption::Maximize => {
                p.line(vec![at(2.5, 10.5), at(8.0, 5.0), at(13.5, 10.5)], s);
            }
            Caption::Close => {
                let s = Stroke::new(1.2, ink);
                p.line_segment([at(3.6, 3.6), at(12.4, 12.4)], s);
                p.line_segment([at(12.4, 3.6), at(3.6, 12.4)], s);
            }
        }
    } else {
        // libadwaita: a 2 pt bar low down, a 2 pt square outline (smaller to restore), an X.
        let frame = |a: f32, b: f32| Rect::from_min_max(at(a, a), at(b, b));
        match c {
            Caption::Minimize => {
                p.rect_filled(Rect::from_min_max(at(4.0, 10.0), at(12.0, 12.0)), 0.0, ink);
            }
            Caption::Maximize => {
                let square = if maximized { frame(5.0, 11.0) } else { frame(4.0, 12.0) };
                p.rect_stroke(square, 0.0, Stroke::new(2.0 * k, ink), egui::StrokeKind::Inside);
            }
            Caption::Close => {
                let s = Stroke::new(1.5 * k, ink);
                p.line_segment([at(4.5, 4.5), at(11.5, 11.5)], s);
                p.line_segment([at(11.5, 4.5), at(4.5, 11.5)], s);
            }
        }
    }
}

/// `icon` from the theme at `px` device pixels, uploaded once per theme and size; `None` (also
/// remembered) when the theme lacks it or it doesn't render.
fn theme_icon(ctx: &egui::Context, icons: &CaptionIcons, icon: CaptionIcon, px: u32) -> Option<egui::TextureHandle> {
    let id = Id::new(("caption-icon", &icons.theme, icon, px));
    if let Some(cached) = ctx.data(|d| d.get_temp::<Option<egui::TextureHandle>>(id)) {
        return cached;
    }
    let side = usize::try_from(px).ok().filter(|&s| (1..=256).contains(&s));
    let tex = side.and_then(|side| {
        let mask = (icons.render)(icon, px).filter(|m| m.len() == side * side)?;
        // White at the mask's coverage, tinted with the ink when drawn.
        let pixels = mask.into_iter().map(Color32::from_white_alpha).collect();
        let image = egui::ColorImage { size: [side, side], source_size: vec2(px as f32, px as f32), pixels };
        Some(ctx.load_texture(format!("caption-{icon:?}-{px}"), image, egui::TextureOptions::LINEAR))
    });
    ctx.data_mut(|d| d.insert_temp(id, tex.clone()));
    tex
}

/// The edge and corner zones of a window with content rect `w`: (zone, direction, cursor). Corners
/// come last so they win where they overlap the edges. Edges use the one-sided cursors
/// (`w-resize`, …), as GTK and KDE window borders do: some cursor themes ship no `ew-resize` /
/// `ns-resize`, and then X11 showed the core font's arrow and Wayland no resize cursor (#1581).
/// Corners keep the diagonal ones (`nwse-resize`), which those themes draw like the edges; their
/// one-sided corner names can be a different style (`top_left_corner`).
fn zones(w: Rect) -> [(Rect, ResizeDirection, CursorIcon); 8] {
    use ResizeDirection::*;
    let (e, c) = (EDGE, CORNER);
    [
        (Rect::from_min_max(w.left_top(), pos2(w.right(), w.top() + e)), North, CursorIcon::ResizeNorth),
        (Rect::from_min_max(pos2(w.left(), w.bottom() - e), w.right_bottom()), South, CursorIcon::ResizeSouth),
        (Rect::from_min_max(w.left_top(), pos2(w.left() + e, w.bottom())), West, CursorIcon::ResizeWest),
        (Rect::from_min_max(pos2(w.right() - e, w.top()), w.right_bottom()), East, CursorIcon::ResizeEast),
        (Rect::from_min_size(w.left_top(), vec2(c, c)), NorthWest, CursorIcon::ResizeNwSe),
        (Rect::from_min_size(w.right_top() - vec2(c, 0.0), vec2(c, c)), NorthEast, CursorIcon::ResizeNeSw),
        (Rect::from_min_size(w.left_bottom() - vec2(0.0, c), vec2(c, c)), SouthWest, CursorIcon::ResizeNeSw),
        (Rect::from_min_size(w.right_bottom() - vec2(c, c), vec2(c, c)), SouthEast, CursorIcon::ResizeNwSe),
    ]
}

fn resize_handed_off_id() -> Id {
    Id::new("titlebar-resize-handed-off")
}

/// An OS resize started at `pass`; `released` is the pass whose input ended the press
/// ([`release_after_os_resize`]).
#[derive(Clone, Copy, Debug)]
struct HandOff {
    pass: u64,
    released: Option<u64>,
}

/// The compositor takes the button release of an OS resize (KDE Wayland), so egui kept the edge
/// pressed: the zones stopped hovering until the next click, every other resize (#1581). End the
/// press ourselves on the input after the hand-off, but only while egui still has it down and this
/// input brings no release of its own: where the OS does send it (Windows, X11), nothing is added,
/// so no second release can turn into a click. Call from `raw_input_hook`.
pub fn release_after_os_resize(ctx: &egui::Context, raw: &mut egui::RawInput) {
    let Some(h) = ctx.data(|d| d.get_temp::<HandOff>(resize_handed_off_id())) else { return };
    if h.released.is_some() || h.pass >= ctx.cumulative_pass_nr() {
        return;
    }
    let os_released = raw.events.iter().any(|e| matches!(e, egui::Event::PointerButton { button: PointerButton::Primary, pressed: false, .. }));
    if let Some(pos) = ctx.input(|i| i.pointer.latest_pos()).filter(|_| !os_released && ctx.input(|i| i.pointer.primary_down())) {
        raw.events.push(egui::Event::PointerButton { pos, button: PointerButton::Primary, pressed: false, modifiers: ctx.input(|i| i.modifiers) });
    }
    let pass = ctx.cumulative_pass_nr();
    ctx.data_mut(|d| d.insert_temp(resize_handed_off_id(), HandOff { released: Some(pass), ..h }));
}

/// The compositor owns the pointer during an OS resize: it sets its own cursor, and the button
/// release goes to it, not to us. egui-winit only re-sends a cursor that changed, so afterwards the
/// window kept the compositor's arrow until the next press (KDE Wayland, #1581). On the first
/// pointer motion after the hand-off, show the default cursor for one frame so the resize cursor is
/// sent again on the next.
fn reshow_cursor_after_os_resize(ctx: &egui::Context) {
    // Only motion after the press was ended means the compositor gave the pointer back.
    let Some(HandOff { released: Some(pass), .. }) = ctx.data(|d| d.get_temp::<HandOff>(resize_handed_off_id())) else { return };
    if pass < ctx.cumulative_pass_nr() && ctx.input(|i| i.events.iter().any(|e| matches!(e, egui::Event::PointerMoved(_)))) {
        ctx.data_mut(|d| d.remove::<HandOff>(resize_handed_off_id()));
        ctx.set_cursor_icon(CursorIcon::Default);
        ctx.request_repaint();
    }
}

/// Invisible resize zones along the window's edges, above everything else; a press in one starts
/// an OS resize in that direction. None while the window is maximized or full screen.
pub fn resize_zones(ui: &mut Ui) {
    let ctx = ui.ctx().clone();
    if maximized(&ctx) || ctx.input(|i| i.viewport().fullscreen.unwrap_or(false)) {
        reshow_cursor_after_os_resize(&ctx);
        return;
    }
    let w = ctx.content_rect();
    let layer = LayerId::new(Order::Foreground, Id::new("titlebar-resize"));
    let zui = ui.new_child(egui::UiBuilder::new().layer_id(layer).max_rect(w));
    for (i, (r, dir, cursor)) in zones(w).into_iter().enumerate() {
        let resp = zui.interact(r, Id::new(("titlebar-resize", i)), Sense::drag()).on_hover_cursor(cursor);
        if resp.drag_started_by(PointerButton::Primary) {
            ctx.send_viewport_cmd(ViewportCommand::BeginResize(dir));
            let pass = ctx.cumulative_pass_nr();
            ctx.data_mut(|d| d.insert_temp(resize_handed_off_id(), HandOff { pass, released: None }));
        }
    }
    // After the zones, so its one-frame cursor wins over theirs.
    reshow_cursor_after_os_resize(&ctx);
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Event, Pos2, RawInput, ViewportId};
    use photocraft_engine::Session;

    /// A headless window `width` pt wide showing the title bar and the resize zones.
    struct Win {
        ctx: egui::Context,
        app: PhotocraftApp,
        width: f32,
        time: f64,
        maximized: bool,
    }

    impl Win {
        fn new(width: f32) -> Self {
            let ctx = egui::Context::default();
            PhotocraftApp::setup_context(&ctx, crate::theme::ThemeKind::ALL[0]);
            let mut app = PhotocraftApp::new(Session::new(), crate::Services::default());
            app.custom_titlebar = true;
            Self { ctx, app, width, time: 0.0, maximized: false }
        }

        fn frame(&mut self, events: Vec<Event>) -> egui::FullOutput {
            self.frame_with_modal(events, false)
        }

        fn frame_with_modal(&mut self, events: Vec<Event>, modal: bool) -> egui::FullOutput {
            self.time += 0.05;
            let mut input =
                RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(self.width, 600.0))), time: Some(self.time), events, ..Default::default() };
            if let Some(v) = input.viewports.get_mut(&ViewportId::ROOT) {
                v.maximized = Some(self.maximized);
            }
            // What `PhotocraftApp::raw_input_hook` does.
            release_after_os_resize(&self.ctx, &mut input);
            let app = &mut self.app;
            let mut out = self.ctx.run_ui(input, |ui| {
                crate::panels::title_bar(app, ui);
                if modal {
                    egui::Modal::new(Id::new("titlebar-drag-test-modal")).backdrop_color(Color32::TRANSPARENT).show(ui.ctx(), |ui| {
                        ui.label("Modal dialog");
                    });
                    modal_drag_overlay(ui.ctx());
                }
                resize_zones(ui);
            });
            out.textures_delta.clear();
            out
        }

        /// Frames for a pointer gesture: move to `path[0]`, press, move along `path`, release. The
        /// viewport commands the frames sent.
        fn gesture(&mut self, path: &[Pos2]) -> Vec<ViewportCommand> {
            self.gesture_with_modal(path, false)
        }

        fn gesture_with_modal(&mut self, path: &[Pos2], modal: bool) -> Vec<ViewportCommand> {
            let mut events = vec![vec![Event::PointerMoved(path[0])], vec![button(path[0], true)]];
            events.extend(path[1..].iter().map(|p| vec![Event::PointerMoved(*p)]));
            events.push(vec![button(*path.last().unwrap_or(&path[0]), false)]);
            events
                .into_iter()
                .flat_map(|e| self.frame_with_modal(e, modal).viewport_output.remove(&ViewportId::ROOT).map(|v| v.commands).unwrap_or_default())
                .collect()
        }

        fn click(&mut self, p: Pos2) -> Vec<ViewportCommand> {
            self.gesture(&[p])
        }

        fn caption(&self, c: Caption) -> Rect {
            self.ctx.read_response(caption_id(c)).map(|r| r.rect).unwrap_or(Rect::NOTHING)
        }

        /// The bar's clickable widgets other than its drag gap and the caption buttons.
        fn bar_widgets(&self) -> Vec<Rect> {
            let skip = CAPTIONS.map(caption_id);
            self.ctx.viewport(|v| {
                v.prev_pass
                    .widgets
                    .layers()
                    .filter(|(l, _)| l.order != Order::Foreground)
                    .flat_map(|(_, ws)| ws.iter())
                    .filter(|w| w.sense.senses_click() && !w.sense.senses_drag() && w.rect.top() < 30.0 && !skip.contains(&w.id))
                    .map(|w| w.rect)
                    .collect()
            })
        }
    }

    fn texts(out: &egui::FullOutput) -> Vec<(String, Rect)> {
        fn walk(s: &egui::Shape, v: &mut Vec<(String, Rect)>) {
            match s {
                egui::Shape::Text(t) => v.push((t.galley.text().to_string(), Rect::from_min_size(t.pos, t.galley.size()))),
                egui::Shape::Vec(s) => s.iter().for_each(|s| walk(s, v)),
                _ => {}
            }
        }
        let mut v = vec![];
        out.shapes.iter().for_each(|c| walk(&c.shape, &mut v));
        v
    }

    fn button(p: Pos2, pressed: bool) -> Event {
        Event::PointerButton { pos: p, button: egui::PointerButton::Primary, pressed, modifiers: Default::default() }
    }

    #[test]
    fn caption_buttons_take_the_right_end_and_nothing_overlaps_them() {
        for width in [760.0, 1100.0, 1440.0] {
            let mut w = Win::new(width);
            w.frame(vec![]);
            let out = w.frame(vec![]);
            let [min, max, close] = CAPTIONS.map(|c| w.caption(c));
            assert_eq!((close.right(), close.top()), (width, 0.0), "Close sits in the window's corner at {width}");
            // The bar's bottom point is the panel's separator line, which the buttons leave visible.
            assert!(close.height() >= 31.0 && [min, max, close].iter().all(|r| r.width() == BUTTON_WIDTH));
            assert_eq!((min.right(), max.right()), (max.left(), close.left()));
            let widgets = w.bar_widgets();
            assert!(widgets.len() >= 10, "menus, search, theme, workspace: {widgets:?}");
            for a in &widgets {
                assert!(a.right() <= min.left(), "{a:?} runs into the caption buttons at {width}");
            }
            // Wide windows show the whole group; the narrowest drops Discord (also Help › Discord).
            let has_discord = texts(&out).iter().any(|(t, _)| t == "Discord");
            assert_eq!(has_discord, width > 1000.0, "Discord at {width}");
        }
    }

    #[test]
    fn the_brand_mark_leads_the_bar() {
        let mut w = Win::new(1440.0);
        w.frame(vec![]);
        let out = w.frame(vec![]);
        let mark = crate::brand::mark_rect(&w.ctx).expect("the brand mark is painted");
        let file = texts(&out).into_iter().find(|(t, _)| t == "File").map(|(_, r)| r).expect("the File menu title");
        assert!(mark.right() < file.left(), "the mark sits left of the menus: {mark:?} {file:?}");
        assert!(mark.left() < 20.0 && mark.width() >= 16.0, "{mark:?}");
    }

    #[test]
    fn close_runs_file_exit_which_asks_about_unsaved_work() {
        let mut w = Win::new(1100.0);
        w.frame(vec![]);
        let close = w.caption(Caption::Close).center();
        let cmds = w.click(close);
        // File › Exit asks the window to close; the unsaved-changes guard (`discard_ui`) then
        // decides, so the caption button and the menu item behave the same.
        assert!(cmds.iter().any(|c| matches!(c, ViewportCommand::Close)), "{cmds:?}");
    }

    #[test]
    fn minimize_and_maximize_follow_the_window_state() {
        let mut w = Win::new(1100.0);
        w.frame(vec![]);
        let (min, max) = (w.caption(Caption::Minimize).center(), w.caption(Caption::Maximize).center());
        assert!(w.click(min).contains(&ViewportCommand::Minimized(true)));
        assert!(w.click(max).contains(&ViewportCommand::Maximized(true)));
        w.maximized = true;
        assert!(w.click(max).contains(&ViewportCommand::Maximized(false)), "Restore");
    }

    #[test]
    fn empty_bar_space_drags_and_double_click_maximizes() {
        let mut w = Win::new(1440.0);
        w.frame(vec![]);
        let empty = pos2(980.0, 16.0);
        assert!(w.bar_widgets().iter().all(|r| !r.contains(empty)), "the probe point must be empty bar space");
        let cmds = w.gesture(&[empty, empty + vec2(12.0, 4.0), empty + vec2(30.0, 8.0)]);
        assert_eq!(cmds.iter().filter(|c| matches!(c, ViewportCommand::StartDrag)).count(), 1, "{cmds:?}");
        let mut cmds = w.click(empty);
        cmds.extend(w.click(empty));
        assert!(cmds.contains(&ViewportCommand::Maximized(true)), "{cmds:?}");
        w.maximized = true;
        w.time += 1.0;
        let mut cmds = w.click(empty);
        cmds.extend(w.click(empty));
        assert!(cmds.contains(&ViewportCommand::Maximized(false)), "{cmds:?}");
        // Widgets keep their clicks: dragging from a menu title doesn't move the window.
        let file = texts(&w.frame(vec![])).into_iter().find(|(t, _)| t == "File").map(|(_, r)| r.center());
        let file = file.expect("the File menu title");
        let cmds = w.gesture(&[file, file + vec2(12.0, 4.0), file + vec2(30.0, 8.0)]);
        assert!(!cmds.iter().any(|c| matches!(c, ViewportCommand::StartDrag)), "{cmds:?}");
    }

    #[test]
    fn empty_bar_space_still_drags_with_a_modal_open() {
        let mut w = Win::new(1440.0);
        w.frame(vec![]);
        w.frame_with_modal(vec![], true);
        let empty = pos2(980.0, 16.0);
        let drag_area = w.ctx.data(|d| d.get_temp::<Rect>(modal_drag_area_id()));
        assert!(drag_area.is_some_and(|r| r.contains(empty)), "the test point must be in the current title-bar gap: {drag_area:?}");
        assert!(w.bar_widgets().iter().all(|r| !r.contains(empty)), "the probe point must be empty bar space");
        let cmds = w.gesture_with_modal(&[empty, empty + vec2(12.0, 4.0), empty + vec2(30.0, 8.0)], true);
        assert_eq!(cmds.iter().filter(|c| matches!(c, ViewportCommand::StartDrag)).count(), 1, "{cmds:?}");
    }

    #[test]
    fn modal_drag_overlay_does_not_cover_a_dialog_moved_into_the_gap() {
        let mut w = Win::new(1440.0);
        w.frame(vec![]);
        w.frame_with_modal(vec![], true);
        let area = w.ctx.data(|d| d.get_temp::<Rect>(modal_drag_area_id())).unwrap_or(Rect::NOTHING);
        assert!(area.contains(pos2(980.0, 16.0)), "the probe point must be in the gap: {area:?}");
        w.ctx.data_mut(|d| d.insert_temp(Id::new("pc-dialog-rects"), vec![area]));
        let cmds = w.gesture_with_modal(&[area.center(), area.center() + vec2(12.0, 4.0)], true);
        assert!(!cmds.iter().any(|c| matches!(c, ViewportCommand::StartDrag)), "{cmds:?}");
    }

    /// #1581: KDE Wayland keeps the button release of an OS resize. egui then kept the edge
    /// pressed (no hover until the next click) and egui-winit never re-sent the resize cursor that
    /// the compositor had replaced: the cursor came back only on every other press.
    #[test]
    fn after_an_os_resize_the_edge_hovers_and_shows_its_cursor_again() {
        let mut w = Win::new(1000.0);
        w.frame(vec![]);
        let p = pos2(3.0, 300.0);
        // Press and drag; the compositor takes over, so no release ever arrives.
        let mut cmds = vec![];
        for e in [vec![Event::PointerMoved(p)], vec![button(p, true)], vec![Event::PointerMoved(p + vec2(6.0, 0.0))]] {
            cmds.extend(w.frame(e).viewport_output.remove(&ViewportId::ROOT).map(|v| v.commands).unwrap_or_default());
        }
        assert!(cmds.contains(&ViewportCommand::BeginResize(ResizeDirection::West)), "{cmds:?}");
        w.frame(vec![]);
        assert!(!w.ctx.input(|i| i.pointer.primary_down()), "the press ends with the hand-off");
        // The pointer comes back: one frame of the default cursor, so the next one is re-sent.
        let out = w.frame(vec![Event::PointerMoved(p)]);
        assert_eq!(out.platform_output.cursor_icon, CursorIcon::Default);
        let out = w.frame(vec![Event::PointerMoved(p + vec2(1.0, 0.0))]);
        assert_eq!(out.platform_output.cursor_icon, CursorIcon::ResizeWest);
        // And the next press resizes again (it alternated before).
        assert!(w.gesture(&[p, p + vec2(6.0, 0.0)]).contains(&ViewportCommand::BeginResize(ResizeDirection::West)));
    }

    /// Where the OS does send the release of a resize (Windows, X11), no second one is added: a
    /// release right after it could count as a click on what lies under the edge.
    #[test]
    fn a_release_the_os_sends_is_not_doubled() {
        let mut w = Win::new(1000.0);
        w.frame(vec![]);
        let p = pos2(3.0, 300.0);
        w.frame(vec![Event::PointerMoved(p)]);
        w.frame(vec![button(p, true)]);
        w.frame(vec![Event::PointerMoved(p + vec2(6.0, 0.0))]);
        // The release arrives with the next input.
        let mut raw = RawInput { events: vec![button(p + vec2(6.0, 0.0), false)], ..Default::default() };
        release_after_os_resize(&w.ctx, &mut raw);
        assert_eq!(raw.events.len(), 1, "{:?}", raw.events);
        // Or it already arrived in an earlier input: egui has the button up, nothing to end.
        let mut w = Win::new(1000.0);
        w.frame(vec![]);
        w.gesture(&[p, p + vec2(6.0, 0.0)]);
        let mut raw = RawInput::default();
        release_after_os_resize(&w.ctx, &mut raw);
        assert!(raw.events.is_empty(), "{:?}", raw.events);
    }

    #[test]
    fn edge_zones_resize_unless_maximized() {
        use ResizeDirection::*;
        let mut w = Win::new(1000.0);
        w.frame(vec![]);
        for (p, dir) in [
            (pos2(1.0, 300.0), West),
            (pos2(998.0, 300.0), East),
            (pos2(500.0, 1.0), North),
            (pos2(500.0, 598.0), South),
            (pos2(2.0, 2.0), NorthWest),
            (pos2(997.0, 3.0), NorthEast),
            (pos2(3.0, 597.0), SouthWest),
            (pos2(998.0, 598.0), SouthEast),
        ] {
            let cmds = w.gesture(&[p, p + vec2(6.0, 6.0)]);
            assert_eq!(cmds, vec![ViewportCommand::BeginResize(dir)], "at {p:?}");
        }
        // The first motion after those resizes shows the default cursor for a frame (see
        // `reshow_cursor_after_os_resize`).
        w.frame(vec![Event::PointerMoved(pos2(500.0, 300.0))]);
        // One-sided cursors, which every cursor theme has (#1581), anywhere in the 8 pt edge.
        for (p, cursor) in [
            (pos2(7.0, 300.0), CursorIcon::ResizeWest),
            (pos2(993.0, 300.0), CursorIcon::ResizeEast),
            (pos2(500.0, 7.0), CursorIcon::ResizeNorth),
            (pos2(500.0, 593.0), CursorIcon::ResizeSouth),
            (pos2(2.0, 2.0), CursorIcon::ResizeNwSe),
            (pos2(997.0, 3.0), CursorIcon::ResizeNeSw),
            (pos2(3.0, 597.0), CursorIcon::ResizeNeSw),
            (pos2(998.0, 598.0), CursorIcon::ResizeNwSe),
        ] {
            let out = w.frame(vec![Event::PointerMoved(p)]);
            assert_eq!(out.platform_output.cursor_icon, cursor, "at {p:?}");
        }
        let cmds = w.gesture(&[pos2(7.0, 300.0), pos2(13.0, 306.0)]);
        assert_eq!(cmds, vec![ViewportCommand::BeginResize(West)], "the edge zone is 8 pt deep");
        // Clear of the zone and of egui's interact radius around it.
        let cmds = w.gesture(&[pos2(20.0, 300.0), pos2(26.0, 306.0)]);
        assert!(!cmds.iter().any(|c| matches!(c, ViewportCommand::BeginResize(_))), "{cmds:?}");
        // Maximized: no zones, and the window's top-right corner is Close.
        w.maximized = true;
        let cmds = w.gesture(&[pos2(1.0, 300.0), pos2(8.0, 300.0)]);
        assert!(!cmds.iter().any(|c| matches!(c, ViewportCommand::BeginResize(_))), "{cmds:?}");
        assert!(w.click(pos2(999.5, 0.5)).iter().any(|c| matches!(c, ViewportCommand::Close)));
    }

    #[test]
    fn gtk_layouts_pick_the_buttons_and_their_sides() {
        use Caption::*;
        let s = CaptionStyle::from_gtk_layout(CaptionFamily::Adwaita, "appmenu:minimize,maximize,close");
        assert_eq!((s.left, s.right), (vec![], vec![Minimize, Maximize, Close]));
        // Unknown names are skipped, spaces trimmed, a repeat placed once, a second colon is a name.
        let s = CaptionStyle::from_gtk_layout(CaptionFamily::Adwaita, "close,close , minimize:icon,maximize:close");
        assert_eq!((s.left, s.right), (vec![Close, Minimize], vec![]));
        let s = CaptionStyle::from_gtk_layout(CaptionFamily::Breeze, "minimize,close");
        assert_eq!((s.left, s.right), (vec![Minimize, Close], vec![]), "no colon: GTK puts them all at the start");
        let s = CaptionStyle::from_gtk_layout(CaptionFamily::Breeze, "");
        assert!(s.left.is_empty() && s.right.is_empty());
        assert_eq!(CaptionFamily::parse(" Breeze "), Some(CaptionFamily::Breeze));
        assert_eq!(CaptionFamily::parse("aqua"), None);
        assert_eq!(CaptionStyle::default(), CaptionStyle { family: CaptionFamily::Windows, left: vec![], right: CAPTIONS.to_vec(), icons: None });
    }

    #[test]
    fn linux_buttons_sit_where_the_layout_says_and_drag_the_window() {
        use Caption::*;
        for (family, size, gap, end) in [(CaptionFamily::Adwaita, 24.0, 6.5, 10.0), (CaptionFamily::Breeze, 18.0, 4.5, 8.0)] {
            let width = 1100.0;
            let mut w = Win::new(width);
            w.app.services.caption_style = Some(Box::new(move |_| CaptionStyle { family, left: vec![Close], right: vec![Minimize, Maximize], icons: None }));
            w.frame(vec![]);
            w.frame(vec![]);
            let [min, max, close] = CAPTIONS.map(|c| w.caption(c));
            // Flush with the window's edges, the edge gap outermost, the hit areas touching.
            assert_eq!((close.left(), close.width()), (0.0, end + size + gap), "{family:?}");
            assert_eq!((max.right(), max.width()), (width, gap + size + end), "{family:?}");
            assert_eq!((min.right(), min.width()), (max.left(), 2.0 * gap + size), "{family:?}");
            // GNOME's header bars are taller than Photoshop's bar; KDE's are not.
            if family == CaptionFamily::Adwaita {
                assert_eq!(close.height(), 46.0);
            } else {
                assert!((31.0..46.0).contains(&close.height()), "{close:?}");
            }
            let mark = crate::brand::mark_rect(&w.ctx).expect("the brand mark is painted");
            assert!(mark.left() > close.right(), "the bar starts after the left buttons: {mark:?} {close:?}");
            for a in w.bar_widgets() {
                assert!(a.left() >= close.right() && a.right() <= min.left(), "{a:?} runs into the caption buttons ({family:?})");
            }
            assert!(w.click(max.center()).contains(&ViewportCommand::Maximized(true)));
            assert!(w.click(close.center()).iter().any(|c| matches!(c, ViewportCommand::Close)));
            // A drag that starts on a button moves the window instead of pressing it.
            let at = min.center();
            let cmds = w.gesture(&[at, at + vec2(12.0, 4.0), at + vec2(30.0, 8.0)]);
            assert_eq!(cmds.iter().filter(|c| matches!(c, ViewportCommand::StartDrag)).count(), 1, "{cmds:?}");
            assert!(!cmds.contains(&ViewportCommand::Minimized(true)), "{cmds:?}");
        }
    }

    /// A side without buttons reserves nothing; without the custom title bar, neither does any.
    #[test]
    fn only_sides_with_buttons_take_room() {
        let mut w = Win::new(1000.0);
        w.app.services.caption_style = Some(Box::new(|_| CaptionStyle::from_gtk_layout(CaptionFamily::Adwaita, "appmenu:close")));
        assert_eq!(reserved(&w.app, &w.ctx), (0.0, 6.5 + 24.0 + 10.0));
        w.app.custom_titlebar = false;
        assert_eq!(reserved(&w.app, &w.ctx), (0.0, 0.0));
        w.app.custom_titlebar = true;
        w.app.services.caption_style = None;
        assert_eq!(reserved(&w.app, &w.ctx), (0.0, 3.0 * BUTTON_WIDTH));
    }

    /// Theme glyphs upload once per theme and size; an icon the theme lacks, a wrong-sized mask or
    /// an absurd size gives `None`, so the button draws its own glyph.
    #[test]
    fn theme_icons_are_cached_and_fall_back() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = std::sync::Arc::new(AtomicUsize::new(0));
        let counter = std::sync::Arc::clone(&calls);
        let render = move |icon: CaptionIcon, px: u32| {
            counter.fetch_add(1, Ordering::Relaxed);
            match icon {
                CaptionIcon::Close => Some(vec![255; (px * px) as usize]),
                CaptionIcon::Minimize => Some(vec![255; 3]),
                _ => None,
            }
        };
        let icons = CaptionIcons { theme: "Test".into(), render: std::sync::Arc::new(render) };
        let ctx = egui::Context::default();
        let mut out = ctx.run_ui(RawInput::default(), |ui| {
            let tex = theme_icon(ui.ctx(), &icons, CaptionIcon::Close, 16).expect("rendered");
            assert_eq!(tex.size(), [16, 16]);
            assert!(theme_icon(ui.ctx(), &icons, CaptionIcon::Close, 16).is_some());
            assert!(theme_icon(ui.ctx(), &icons, CaptionIcon::Restore, 16).is_none());
            assert!(theme_icon(ui.ctx(), &icons, CaptionIcon::Restore, 16).is_none());
            assert!(theme_icon(ui.ctx(), &icons, CaptionIcon::Minimize, 16).is_none(), "a mask of the wrong size");
            assert!(theme_icon(ui.ctx(), &icons, CaptionIcon::Close, 0).is_none());
            assert!(theme_icon(ui.ctx(), &icons, CaptionIcon::Close, 100_000).is_none());
        });
        out.textures_delta.clear();
        assert_eq!(calls.load(Ordering::Relaxed), 3, "each icon and size rendered once; absurd sizes never");
    }
}
