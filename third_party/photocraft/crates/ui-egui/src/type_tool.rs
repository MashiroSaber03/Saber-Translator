//! Type tool: click to add point text, drag to add paragraph text, edit inline with a caret.
//!
//! Every edit is an engine `type.*` command carrying the session's `coalesce` key, so a whole
//! typing session is one "Edit Type" history step and automation sees exactly what the user does.
//! Offsets in [`TextEdit`] are character indices (the engine's unit); the layout works in bytes.

use std::collections::HashMap;
use std::sync::{Arc, PoisonError};

use egui::{Color32, Pos2, Stroke};
use photocraft_doc::{Document, LayerContent, LayerId, TextLayer};
use photocraft_geom::{Affine, Point};
use photocraft_text::TextLayout;
use serde_json::json;

use crate::PhotocraftApp;
use crate::canvas::ViewXform;
use crate::state::TextEdit;

pub const PLACEHOLDER: &str = "Lorem Ipsum";

fn text_layer(doc: &Document, id: LayerId) -> Option<&TextLayer> {
    match &doc.layer(id)?.content {
        LayerContent::Text(t) => Some(t),
        _ => None,
    }
}

fn byte_of(text: &str, ci: usize) -> usize {
    photocraft_text::byte_index(text, ci)
}

/// Layout of a type layer (cached per document revision) and its text → document transform.
pub fn layout(app: &mut PhotocraftApp, id: LayerId) -> Option<(Arc<TextLayout>, Affine, String)> {
    let st = app.session.active()?;
    let (doc, rev) = (st.doc.clone(), st.revision);
    let t = text_layer(&doc, id)?;
    let transform = crate::type_transform::current_transform(app, id).unwrap_or(t.transform);
    let key = (doc.id.0, rev, id.0);
    if let Some((k, l)) = &app.type_layout
        && *k == key
    {
        return Some((l.clone(), transform, t.text.clone()));
    }
    let l = Arc::new(photocraft_text::shared().lock().ok()?.layout(t, doc.resolution_dpi));
    app.type_layout = Some((key, l.clone()));
    Some((l, transform, t.text.clone()))
}

fn to_text(aff: &Affine, x: f64, y: f64) -> (f32, f32) {
    let p = aff.inverse().unwrap_or(Affine::IDENTITY).apply(Point::new(x, y));
    (p.x as f32, p.y as f32)
}

/// Topmost shown type layer (it and every group around it visible, #4320) whose laid-out text
/// contains the document point.
fn hit_layer(app: &mut PhotocraftApp, x: f64, y: f64) -> Option<LayerId> {
    let doc = app.session.active()?.doc.clone();
    let slop = 6.0 / app.point_zoom().max(0.01);
    // A Type Mask session's text is a hidden layer; it is still the text being edited.
    let editing = app.ui.text_edit.as_ref().filter(|ed| ed.mask).map(|ed| LayerId(ed.layer));
    let mut ids: Vec<LayerId> = doc
        .walk()
        .into_iter()
        .filter(|(path, _, l)| matches!(l.content, LayerContent::Text(_)) && (editing == Some(l.id) || doc.shown_at(path)))
        .map(|(_, _, l)| l.id)
        .collect();
    ids.reverse(); // walk() is bottom-up; hit the topmost first
    ids.into_iter().find(|id| {
        // The pixels shown count too: a PSD's type keeps Photoshop's rendering until edited,
        // which can sit off our layout when fonts are substituted (#123).
        let shown = text_layer(&doc, *id).and_then(|t| t.cache.as_ref()).map(|c| c.content_bounds()).is_some_and(|r| {
            let s = f64::from(slop);
            !r.is_empty() && x >= f64::from(r.x0) - s && x <= f64::from(r.x1) + s && y >= f64::from(r.y0) - s && y <= f64::from(r.y1) + s
        });
        let Some((l, aff, _)) = layout(app, *id) else { return shown };
        let (tx, ty) = to_text(&aff, x, y);
        shown || photocraft_text::text_point_inside(&l, tx, ty, slop)
    })
}

/// True when the layer's pixels are what our engine draws for it. A PSD's type layer keeps
/// Photoshop's pixels until it is edited, and with substituted fonts or a different line layout
/// they sit somewhere else than the glyphs the caret and selection are placed on.
pub(crate) fn shows_own_layout(doc: &Document, t: &TextLayer) -> bool {
    let Some(cache) = &t.cache else { return false };
    let mut eng = photocraft_text::shared().lock().unwrap_or_else(PoisonError::into_inner);
    let ours = eng.render(t, doc.resolution_dpi, doc.pixel_format()).1.surface.content_bounds();
    let have = cache.content_bounds();
    [(ours.x0, have.x0), (ours.y0, have.y0), (ours.x1, have.x1), (ours.y1, have.y1)].iter().all(|(a, b)| (a - b).abs() <= 1)
}

/// Start editing an existing type layer. Like Photoshop, editing shows the text as the type
/// engine lays it out, so a PSD layer is re-rendered first (inside the edit session's history
/// step, so Cancel brings Photoshop's pixels back).
fn begin_edit(app: &mut PhotocraftApp, id: LayerId, key: &str) -> Result<(), String> {
    let st = app.session.active().ok_or("no document open")?;
    let doc = st.doc.clone();
    if let Some(t) = text_layer(&doc, id)
        && !shows_own_layout(&doc, t)
    {
        app.run("type.edit", json!({"layer": id.0, "coalesce": key}))?;
    }
    Ok(())
}

/// Edit the active type layer from its thumbnail, selecting all text like Photoshop. Reuse the
/// current session when already editing it, so Cancel and undo still cover the whole edit.
pub fn edit_active(app: &mut PhotocraftApp) -> Result<(), String> {
    let st = app.session.active().ok_or("no document open")?;
    let id = st.active_layer.ok_or("no active layer")?;
    let n = text_layer(&st.doc, id).ok_or("active layer is not a type layer")?.text.chars().count();
    if app.ui.text_edit.as_ref().is_none_or(|ed| ed.layer != id.0) {
        commit(app);
        let key = session_key(app);
        begin_edit(app, id, &key)?;
        app.ui.text_edit =
            Some(TextEdit { layer: id.0, caret: n, anchor: 0, session: key, created: false, mask: false, dragging: false, resize: None, preedit: None });
    } else if let Some(ed) = app.ui.text_edit.as_mut() {
        ed.anchor = 0;
        ed.caret = n;
    }
    let vertical = app.session.active().and_then(|st| text_layer(&st.doc, id)).is_some_and(|t| t.orientation == photocraft_doc::text::Orientation::Vertical);
    app.ui.tool = if vertical { crate::state::Tool::VerticalType } else { crate::state::Tool::Type };
    app.ui.mask_target = false;
    app.ui.vector_mask_target = false;
    Ok(())
}

/// Double-click with the Move tool (Photoshop CC 2019+): edit the type layer under document
/// point (`x`, `y`) with the caret on the glyph there, switching to the Type tool. False when no
/// type is there.
pub fn edit_at(app: &mut PhotocraftApp, x: f64, y: f64) -> bool {
    let Some(id) = hit_layer(app, x, y) else { return false };
    commit(app);
    if app.session.select_layer(id).is_err() {
        return false;
    }
    let key = session_key(app);
    if begin_edit(app, id, &key).is_err() {
        return false;
    }
    let off = hit_offset(app, id, x, y);
    app.ui.text_edit =
        Some(TextEdit { layer: id.0, caret: off, anchor: off, session: key, created: false, mask: false, dragging: false, resize: None, preedit: None });
    let vertical = app.session.active().and_then(|st| text_layer(&st.doc, id)).is_some_and(|t| t.orientation == photocraft_doc::text::Orientation::Vertical);
    app.ui.tool = if vertical { crate::state::Tool::VerticalType } else { crate::state::Tool::Type };
    app.ui.mask_target = false;
    app.ui.vector_mask_target = false;
    true
}

fn hit_offset(app: &mut PhotocraftApp, id: LayerId, x: f64, y: f64) -> usize {
    offset_at(app, id, x, y)
}

/// The character offset of the caret position nearest document point (`x`, `y`) in layer `id`.
pub(crate) fn offset_at(app: &mut PhotocraftApp, id: LayerId, x: f64, y: f64) -> usize {
    let Some((l, aff, text)) = layout(app, id) else { return 0 };
    let (tx, ty) = to_text(&aff, x, y);
    photocraft_text::hit_char(&l, &text, tx, ty).0
}

fn hex(c: [f32; 4]) -> String {
    let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", b(c[0]), b(c[1]), b(c[2]))
}

/// Smallest paragraph box a handle drag leaves, in text-space px.
const MIN_BOX: f32 = 4.0;

/// Which box edges handle `i` moves: (left, right, top, bottom). The order matches the overlay.
fn handle_sides(i: u8) -> (bool, bool, bool, bool) {
    match i {
        0 => (true, false, true, false),
        1 => (false, true, true, false),
        2 => (false, true, false, true),
        3 => (true, false, false, true),
        4 => (false, false, true, false),
        5 => (false, true, false, false),
        6 => (false, false, false, true),
        _ => (true, false, false, false),
    }
}

pub(crate) fn box_shape(app: &PhotocraftApp, id: LayerId) -> Option<(f32, f32, f32, f32)> {
    match app.session.active().and_then(|s| text_layer(&s.doc, id).map(|t| t.shape.clone()))? {
        photocraft_doc::text::TextShape::Box { x, y, width, height } => Some((x, y, width, height)),
        _ => None,
    }
}

/// The paragraph-box handle under the document point, if the layer is paragraph text.
pub(crate) fn box_handle_at(app: &mut PhotocraftApp, id: LayerId, x: f64, y: f64) -> Option<u8> {
    let (bx, by, w, h) = box_shape(app, id)?;
    let (_, aff, _) = layout(app, id)?;
    let (r, b) = (bx + w, by + h);
    let (mx, my) = (bx + w / 2.0, by + h / 2.0);
    let spots = [(bx, by), (r, by), (r, b), (bx, b), (mx, by), (r, my), (mx, b), (bx, my)];
    let tol = f64::from(6.0 / app.point_zoom().max(0.01));
    spots
        .iter()
        .position(|&(sx, sy)| {
            let p = aff.apply(Point::new(f64::from(sx), f64::from(sy)));
            (p.x - x).abs() <= tol && (p.y - y).abs() <= tol
        })
        .map(|i| i as u8)
}

/// Drag handle `i` to the document point: the dragged edges follow it, the others stay put.
fn resize_box(app: &mut PhotocraftApp, ed: &TextEdit, i: u8, x: f64, y: f64) {
    let id = LayerId(ed.layer);
    let Some((bx, by, w, h)) = box_shape(app, id) else { return };
    let Some((_, aff, _)) = layout(app, id) else { return };
    let (px, py) = to_text(&aff, x, y);
    let (l, r, t, b) = handle_sides(i);
    let (mut x0, mut y0, mut x1, mut y1) = (bx, by, bx + w, by + h);
    if l {
        x0 = px.min(x1 - MIN_BOX);
    }
    if r {
        x1 = px.max(x0 + MIN_BOX);
    }
    if t {
        y0 = py.min(y1 - MIN_BOX);
    }
    if b {
        y1 = py.max(y0 + MIN_BOX);
    }
    // type.edit places the box's top-left at the given document point.
    let o = aff.apply(Point::new(f64::from(x0), f64::from(y0)));
    let _ = app.run("type.edit", json!({"layer": ed.layer, "box": [o.x, o.y, x1 - x0, y1 - y0], "coalesce": ed.session}));
}

/// Pointer down with the Type tool. Returns true when the press was consumed (no box drag).
pub fn pointer_down(app: &mut PhotocraftApp, x: f64, y: f64, mods: egui::Modifiers) -> bool {
    if let Some(ed) = app.ui.text_edit.clone() {
        let id = LayerId(ed.layer);
        if let Some(i) = box_handle_at(app, id, x, y) {
            if let Some(e) = app.ui.text_edit.as_mut() {
                e.resize = Some(i);
            }
            return true;
        }
        if hit_layer(app, x, y) == Some(id) {
            let off = hit_offset(app, id, x, y);
            if let Some(e) = app.ui.text_edit.as_mut() {
                e.caret = off;
                if !mods.shift {
                    e.anchor = off;
                }
                e.dragging = true;
            }
            return true;
        }
        // ⇧/⌥ on that click combine a Type Mask with the selection.
        commit_with(app, mods);
        return true; // Photoshop: a click outside commits without starting new text
    }
    // The Type Mask tools always start new text: existing type layers are just pixels to them;
    // so does a ⇧-click with the Type tool, as in Photoshop, to start new text over old.
    if app.ui.tool.is_type_mask() || mods.shift {
        return false;
    }
    if let Some(id) = hit_layer(app, x, y) {
        let _ = app.session.select_layer(id);
        let key = session_key(app);
        if begin_edit(app, id, &key).is_err() {
            return true;
        }
        let off = hit_offset(app, id, x, y);
        app.ui.text_edit =
            Some(TextEdit { layer: id.0, caret: off, anchor: off, session: key, created: false, mask: false, dragging: true, resize: None, preedit: None });
        return true;
    }
    false
}

pub fn pointer_move(app: &mut PhotocraftApp, x: f64, y: f64) {
    let Some(ed) = app.ui.text_edit.clone() else { return };
    if let Some(i) = ed.resize {
        resize_box(app, &ed, i, x, y);
        return;
    }
    if ed.dragging {
        let off = hit_offset(app, LayerId(ed.layer), x, y);
        if let Some(e) = app.ui.text_edit.as_mut() {
            e.caret = off;
        }
    }
}

/// Pointer up. `rect` is the dragged box (document px) when no edit session consumed the press.
pub fn pointer_up(app: &mut PhotocraftApp, start: [f64; 2], end: [f64; 2]) {
    if let Some(e) = app.ui.text_edit.as_mut() {
        e.dragging = false;
        e.resize = None;
        return;
    }
    let (w, h) = ((end[0] - start[0]).abs(), (end[1] - start[1]).abs());
    let min = 4.0 / app.point_zoom().max(0.01) as f64;
    let o = app.ui.tool_options.clone();
    let mask = app.ui.tool.is_type_mask();
    // Preferences ▸ Type: "Fill new type layers with placeholder text" (on by default). The Type
    // Mask tools start empty, as Photoshop's do.
    let text = if !mask && app.session.prefs().type_.fill_new_type_layers_with_placeholder { PLACEHOLDER } else { "" };
    let mut p = json!({
        "text": text,
        "orientation": if app.ui.tool.is_vertical_type() { "vertical" } else { "horizontal" },
        "font": o.type_font,
        "fontStyle": o.type_style,
        "size": o.type_size,
        "align": o.type_align,
        "color": hex(app.session.tools.foreground),
    });
    if w >= min && h >= min {
        p["box"] = json!([start[0].min(end[0]).round(), start[1].min(end[1]).round(), w.round(), h.round()]);
    } else {
        p["x"] = json!(start[0].round());
        p["y"] = json!(start[1].round());
    }
    let key = session_key(app);
    p["coalesce"] = json!(key);
    if let Ok(v) = app.run("type.create", p)
        && let Some(id) = v.get("layer").and_then(serde_json::Value::as_u64)
    {
        if o.type_aa != "sharp" {
            let _ = app.run("type.edit", json!({"layer": id, "antialias": o.type_aa, "coalesce": key}));
        }
        // A Type Mask's text is a hidden working layer: the canvas shows the mask overlay
        // instead, and committing turns it into the selection.
        if mask {
            let _ = app.run("layer.setProps", json!({"layer": id, "visible": false, "coalesce": key}));
        }
        // Like Photoshop: the placeholder is selected, so typing replaces it.
        let n = text.chars().count();
        app.ui.text_edit = Some(TextEdit { layer: id, caret: n, anchor: 0, session: key, created: true, mask, dragging: false, resize: None, preedit: None });
    }
}

fn session_key(app: &mut PhotocraftApp) -> String {
    format!("type-{}", app.ui.alloc_id())
}

fn current_text(app: &PhotocraftApp, id: LayerId) -> Option<String> {
    Some(text_layer(&app.session.active()?.doc, id)?.text.clone())
}

/// The selected characters, clamped after undo/redo may have shortened the text.
pub(crate) fn selected_range(app: &PhotocraftApp) -> Option<[usize; 2]> {
    let ed = app.ui.text_edit.as_ref()?;
    let n = text_layer(&app.session.active()?.doc, LayerId(ed.layer))?.text.chars().count();
    let (a, b) = (ed.caret.min(ed.anchor).min(n), ed.caret.max(ed.anchor).min(n));
    (a < b).then_some([a, b])
}

/// Copy the selected characters to the system clipboard. Returns the copied range. Shared by the
/// clipboard Copy event and Edit > Copy during a type edit.
pub(crate) fn copy_selection(app: &PhotocraftApp, ctx: &egui::Context) -> Option<[usize; 2]> {
    let [a, b] = selected_range(app)?;
    let ed = app.ui.text_edit.as_ref()?;
    let text = text_layer(&app.session.active()?.doc, LayerId(ed.layer))?;
    ctx.copy_text(text.text.chars().skip(a).take(b - a).collect());
    Some([a, b])
}

/// Edit > Paste during a type edit: ask the window for the clipboard text, which arrives as a
/// Paste event next frame and replaces the selection there (like ⌘V).
pub(crate) fn request_paste(ctx: &egui::Context) {
    ctx.send_viewport_cmd(egui::ViewportCommand::RequestPaste);
}

/// Select > All during a type edit selects every character (like ⌘A).
pub(crate) fn select_all(app: &mut PhotocraftApp) {
    let Some(n) = app.ui.text_edit.as_ref().and_then(|ed| current_text(app, LayerId(ed.layer))).map(|t| t.chars().count()) else { return };
    if let Some(e) = app.ui.text_edit.as_mut() {
        (e.anchor, e.caret) = (0, n);
    }
}

/// Shared by the clipboard Cut event and Edit > Cut during a type edit.
pub(crate) fn cut_selection(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let Some([a, b]) = copy_selection(app, ctx) else { return };
    if let Some(ed) = app.ui.text_edit.as_mut() {
        ed.anchor = a;
        ed.caret = b;
    }
    insert(app, "");
}

/// Replace the selection with `s`.
fn insert(app: &mut PhotocraftApp, s: &str) {
    let Some(ed) = app.ui.text_edit.clone() else { return };
    let (a, b) = (ed.caret.min(ed.anchor), ed.caret.max(ed.anchor));
    let s = s.replace("\r\n", "\n").replace('\r', "\n");
    if a == b && s.is_empty() {
        return;
    }
    crate::type_transform::finish(app);
    if app.run("type.edit", json!({"layer": ed.layer, "replace": {"start": a, "end": b, "text": s}, "coalesce": ed.session})).is_ok()
        && let Some(e) = app.ui.text_edit.as_mut()
    {
        e.caret = a + s.chars().count();
        e.anchor = e.caret;
    }
}

/// Typed text (not a paste): with Preferences ▸ Type ▸ Use Smart Quotes, straight quotes become
/// typographic (Photoshop does not rewrite pasted text).
fn insert_typed(app: &mut PhotocraftApp, s: &str) {
    if !app.session.prefs().type_.smart_quotes {
        return insert(app, s);
    }
    let prev = app.ui.text_edit.clone().and_then(|ed| {
        let at = ed.caret.min(ed.anchor);
        let text = current_text(app, LayerId(ed.layer))?;
        (at > 0).then(|| text.chars().nth(at - 1)).flatten()
    });
    insert(app, &smart_quotes(s, prev));
}

/// Straight quotes become “ ” and ‘ ’: opening after whitespace, a bracket or a dash (or at the
/// start of the text), closing next to a letter or another quote. `prev` is the character before
/// the caret.
fn smart_quotes(s: &str, mut prev: Option<char>) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        let c = match c {
            '"' => {
                if opens_quote(prev) {
                    '\u{201c}'
                } else {
                    '\u{201d}'
                }
            }
            '\'' => {
                if opens_quote(prev) {
                    '\u{2018}'
                } else {
                    '\u{2019}'
                }
            }
            _ => c,
        };
        out.push(c);
        prev = Some(c);
    }
    out
}

fn opens_quote(prev: Option<char>) -> bool {
    prev.is_none_or(|p| p.is_whitespace() || matches!(p, '(' | '[' | '{' | '-' | '\u{2013}' | '\u{2014}'))
}

/// Alt+←/→: kern the pair before the caret by `by` (1/1000 em). No pair (caret at a text or line
/// edge): nothing happens, like Photoshop. Not coalesced: one history step per press.
fn kern_pair(app: &mut PhotocraftApp, id: LayerId, caret: usize, by: f32) {
    let Some(text) = current_text(app, id) else { return };
    let before = caret.checked_sub(1).and_then(|i| text.chars().nth(i));
    let after = text.chars().nth(caret);
    if before.is_none_or(|c| c == '\n') || after.is_none_or(|c| c == '\n') {
        return;
    }
    crate::type_transform::finish(app);
    let _ = app.run("type.edit", json!({"layer": id.0, "kernPair": {"at": caret, "by": by}}));
}

/// A character style the type editor's ⇧⌘ keys toggle (Photoshop: ⇧⌘K All Caps, ⇧⌘U
/// Underline, ⇧⌘B Faux Bold, ⇧⌘I Faux Italic, ⇧⌘/ Strikethrough).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CharToggle {
    AllCaps,
    Underline,
    FauxBold,
    FauxItalic,
    Strikethrough,
}

impl CharToggle {
    fn is_on(self, st: &photocraft_doc::text::CharStyle) -> bool {
        match self {
            Self::AllCaps => st.caps == photocraft_doc::text::Caps::AllCaps,
            Self::Underline => st.underline,
            Self::FauxBold => st.faux_bold,
            Self::FauxItalic => st.faux_italic,
            Self::Strikethrough => st.strikethrough,
        }
    }

    fn param(self, on: bool) -> (&'static str, serde_json::Value) {
        match self {
            Self::AllCaps => ("caps", json!(if on { "all" } else { "normal" })),
            Self::Underline => ("underline", json!(on)),
            Self::FauxBold => ("fauxBold", json!(on)),
            Self::FauxItalic => ("fauxItalic", json!(on)),
            Self::Strikethrough => ("strikethrough", json!(on)),
        }
    }
}

/// The byte range of characters `a..b` of `text`.
fn byte_range(text: &str, a: usize, b: usize) -> (usize, usize) {
    let at = |c: usize| text.char_indices().nth(c).map_or(text.len(), |(i, _)| i);
    (at(a.min(b)), at(a.max(b)))
}

/// The character styles of the selection, or of the typing style when the text is empty. Empty
/// when nothing is selected in non-empty text.
fn selected_styles(t: &TextLayer, a: usize, b: usize) -> Vec<photocraft_doc::text::CharStyle> {
    let runs = t.char_runs();
    if t.text.is_empty() {
        return runs.into_iter().take(1).map(|r| r.style).collect();
    }
    let (ba, bb) = byte_range(&t.text, a, b);
    let mut at = 0usize;
    let mut out = Vec::new();
    for r in runs {
        let (rs, re) = (at, at.saturating_add(r.len));
        at = re;
        if rs < bb && re > ba {
            out.push(r.style);
        }
    }
    out
}

/// The character range the type editor's formatting keys act on: the selection, or the empty
/// text's typing style. `None` for a collapsed caret in non-empty text: Photoshop would style the
/// next characters typed, which the document model can't hold apart from the text, so the keys
/// do nothing there (rather than restyle a word the user didn't select).
fn format_range(app: &PhotocraftApp, ed: &TextEdit) -> Option<(usize, usize)> {
    let t = text_layer(&app.session.active()?.doc, LayerId(ed.layer))?;
    let n = t.text.chars().count();
    let (a, b) = (ed.caret.min(ed.anchor).min(n), ed.caret.max(ed.anchor).min(n));
    (a < b || n == 0).then_some((a, b))
}

/// One `type.setStyle` step on the edited layer's `range`, kept apart from the typing session's
/// coalesced edit so Undo takes back one key press at a time (as with kerning).
fn set_style(app: &mut PhotocraftApp, ed: &TextEdit, range: (usize, usize), mut p: serde_json::Value) {
    crate::type_transform::finish(app);
    p["layer"] = json!(ed.layer);
    p["range"] = json!([range.0, range.1]);
    if let Err(e) = app.run("type.setStyle", p) {
        app.ui.status = e;
        app.ui.status_error = true;
    }
}

/// ⇧⌘K/U/B/I/: flips a character style over the selection: off when every selected character
/// already has it, else on.
pub(crate) fn toggle_char_style(app: &mut PhotocraftApp, which: CharToggle) {
    let Some(ed) = app.ui.text_edit.clone() else { return };
    let Some(range) = format_range(app, &ed) else { return };
    let Some(styles) = app.session.active().and_then(|s| text_layer(&s.doc, LayerId(ed.layer))).map(|t| selected_styles(t, range.0, range.1)) else {
        return;
    };
    let on = !styles.iter().all(|st| which.is_on(st));
    let (key, value) = which.param(on);
    set_style(app, &ed, range, json!({ key: value }));
}

/// ⌥↑/↓ leading, ⇧⌥↑/↓ baseline shift and ⌥←/→ tracking: steps every selected character from its
/// own value (`key` is `leadingBy`, `baselineShiftBy` or `trackingBy`).
pub(crate) fn step_char_metric(app: &mut PhotocraftApp, key: &str, by: f32) {
    let Some(ed) = app.ui.text_edit.clone() else { return };
    let Some(range) = format_range(app, &ed) else { return };
    // Leading steps in the points the Character panel shows for a scaled layer.
    set_style(app, &ed, range, json!({ key: by, "metricsAsShown": key == "leadingBy" }));
}

/// ⌥⇧⌘H: flips Hyphenate for the paragraphs the selection (or the caret) is in: off when all of
/// them hyphenate, else on.
pub(crate) fn toggle_hyphenate(app: &mut PhotocraftApp) {
    let Some(ed) = app.ui.text_edit.clone() else { return };
    let Some(t) = app.session.active().and_then(|s| text_layer(&s.doc, LayerId(ed.layer))) else { return };
    let n = t.text.chars().count();
    let (a, b) = (ed.caret.min(ed.anchor).min(n), ed.caret.max(ed.anchor).min(n));
    let (ba, bb) = byte_range(&t.text, a, b);
    let len = t.text.len();
    let mut at = 0usize;
    let mut all = true;
    let mut any = false;
    for r in t.paragraph_runs() {
        let (ps, pe) = (at, at.saturating_add(r.len));
        at = pe;
        let touches = if ba == bb { ps <= ba && (ba < pe || pe >= len) } else { ps < bb && pe > ba };
        if touches {
            any = true;
            all &= r.style.hyphenate;
        }
    }
    let on = !(any && all);
    set_style(app, &ed, (a, b), json!({ "hyphenate": on }));
}

/// IME composition. The preedit text is written into the layer (so it lays out and reflows like
/// typed text) and replaced by every update; `commit` makes the result final.
fn ime_update(app: &mut PhotocraftApp, s: &str, commit: bool) {
    let Some(ed) = app.ui.text_edit.clone() else { return };
    let Some(text) = current_text(app, LayerId(ed.layer)) else { return };
    let n = text.chars().count();
    let s = s.replace("\r\n", "\n").replace('\r', "\n");
    // An empty preedit or commit while nothing is being composed changes nothing. Wayland input
    // methods send one on every state change (the caret rectangle moving as a selection grows),
    // and it used to replace the selection, deleting the text as it was selected (#1381).
    if s.is_empty() && ed.preedit.is_none() {
        return;
    }
    let (start, end) = match ed.preedit {
        Some((p, l)) if p.saturating_add(l) <= n => (p, p + l),
        _ => (ed.caret.min(ed.anchor).min(n), ed.caret.max(ed.anchor).min(n)),
    };
    if start == end && s.is_empty() {
        if let Some(e) = app.ui.text_edit.as_mut() {
            e.preedit = None;
        }
        return;
    }
    let len = s.chars().count();
    crate::type_transform::finish(app);
    if app.run("type.edit", json!({"layer": ed.layer, "replace": {"start": start, "end": end, "text": s}, "coalesce": ed.session})).is_ok()
        && let Some(e) = app.ui.text_edit.as_mut()
    {
        e.caret = start + len;
        e.anchor = e.caret;
        e.preedit = if commit || len == 0 { None } else { Some((start, len)) };
    }
}

/// Character index of the previous / next word boundary (shared with `type.navigate`).
fn word_boundary(text: &str, from: usize, forward: bool) -> usize {
    photocraft_text::word_boundary(text, from, forward)
}

/// Most clicks in a run that select more: 2 a word, 3 a line, 4 a paragraph, 5 all text. The
/// next quick click starts over with a caret.
const MAX_CLICKS: u32 = 5;

/// Count of this click in a run of clicks at one spot. `egui_count` is egui's own count for the
/// click (1 single, 2 double, 3 triple; it never goes past 3), which already applies the
/// double-click delay and distance; `prev` is our count for the previous click (0: none). A slow
/// or distant click (egui says 1) starts a new run; after five clicks the run starts over.
pub(crate) fn next_click_count(prev: u32, egui_count: u32) -> u32 {
    match (egui_count, prev) {
        (0 | 1, _) => 1,
        (n, 0) => n.min(MAX_CLICKS),
        (_, p) => p % MAX_CLICKS + 1,
    }
}

/// The selection (anchor, caret) a run of `count` clicks makes at character `caret`: 1 a caret,
/// 2 the word under it, 3 its visual line as laid out (a column in vertical type), 4 its
/// paragraph up to the hard return, 5 all text. `line` is the laid-out line the click hit; it
/// decides at a soft wrap, where one caret offset ends a line and starts the next. Without a
/// layout a line is its paragraph. Offsets past the text are clamped, so an empty layer or a
/// stale caret never panics.
pub(crate) fn click_selection(layout: Option<&TextLayout>, text: &str, caret: usize, line: Option<usize>, count: u32) -> (usize, usize) {
    let n = text.chars().count();
    let c = caret.min(n);
    match count {
        2 => {
            let chars: Vec<char> = text.chars().collect();
            let word = |i: usize| chars.get(i).is_some_and(|ch| ch.is_alphanumeric());
            let (mut a, mut b) = (c, c);
            while a > 0 && word(a - 1) {
                a -= 1;
            }
            while word(b) {
                b += 1;
            }
            (a, b)
        }
        3 => {
            let Some(l) = layout else { return paragraph_of(text, c) };
            let ln = line.and_then(|i| l.lines.get(i)).or_else(|| l.lines.get(photocraft_text::line_index(l, byte_of(text, c))));
            ln.map_or((c, c), |ln| (photocraft_text::char_index(text, ln.range.start), photocraft_text::char_index(text, ln.range.end)))
        }
        4 => paragraph_of(text, c),
        5 => (0, n),
        _ => (c, c),
    }
}

/// Characters of the paragraph holding character `c`, without its hard return.
fn paragraph_of(text: &str, c: usize) -> (usize, usize) {
    let b = byte_of(text, c);
    let paras = photocraft_text::layout::split_paragraphs(text);
    let Some(r) = paras.iter().find(|r| b >= r.start && b < r.end).or_else(|| paras.last()) else { return (c, c) };
    let body = text.get(r.clone()).unwrap_or("").trim_end_matches(['\r', '\n']);
    let start = photocraft_text::char_index(text, r.start);
    (start, start + body.chars().count())
}

/// A click while editing type at document point (`x`, `y`), after it placed the caret there.
/// `egui_count` is egui's count for the click (see [`next_click_count`]). Like Photoshop, two
/// clicks select a word, three the line, four the paragraph and five all text.
pub fn multi_click(app: &mut PhotocraftApp, egui_count: u32, x: f64, y: f64) {
    let Some(ed) = app.ui.text_edit.clone() else {
        app.type_clicks = None;
        return;
    };
    let prev = app.type_clicks.filter(|(layer, _)| *layer == ed.layer).map_or(0, |(_, n)| n);
    let count = next_click_count(prev, egui_count);
    app.type_clicks = Some((ed.layer, count));
    if count < 2 {
        return;
    }
    let id = LayerId(ed.layer);
    let Some(text) = current_text(app, id) else { return };
    let laid = layout(app, id);
    let (caret, line) = match &laid {
        Some((l, aff, t)) if *t == text && x.is_finite() && y.is_finite() => {
            let (tx, ty) = to_text(aff, x, y);
            let (ci, line) = photocraft_text::hit_char(l, t, tx, ty);
            (ci, Some(line))
        }
        _ => (ed.caret, None),
    };
    let (a, b) = click_selection(laid.as_ref().map(|(l, _, _)| l.as_ref()), &text, caret, line, count);
    if let Some(e) = app.ui.text_edit.as_mut() {
        (e.anchor, e.caret) = (a, b);
    }
}

/// Caret on the neighbouring line (±1; a column in vertical type), keeping the position
/// along the line. Works in line space, so it serves both orientations.
fn line_step(app: &mut PhotocraftApp, id: LayerId, caret: usize, dir: i32) -> usize {
    let Some((l, _, text)) = layout(app, id) else { return caret };
    let x = l.caret(byte_of(&text, caret)).0;
    photocraft_text::line_step(&l, &text, caret, x, dir)
}

fn is_vertical(app: &PhotocraftApp, id: LayerId) -> bool {
    app.session.active().and_then(|s| text_layer(&s.doc, id)).is_some_and(|t| t.orientation == photocraft_doc::text::Orientation::Vertical)
}

/// Arrow keys follow the text flow: in vertical type ↑/↓ move along the column (previous/next
/// character) and ←/→ move to the next/previous column (columns advance right to left). Returns
/// the horizontal-type key with the same meaning.
pub(crate) fn flow_key(key: egui::Key, vertical: bool) -> egui::Key {
    use egui::Key;
    if !vertical {
        return key;
    }
    match key {
        Key::ArrowUp => Key::ArrowLeft,
        Key::ArrowDown => Key::ArrowRight,
        Key::ArrowLeft => Key::ArrowDown,
        Key::ArrowRight => Key::ArrowUp,
        k => k,
    }
}

/// Line start / end for the caret's line.
fn line_edge(app: &mut PhotocraftApp, id: LayerId, caret: usize, end: bool) -> usize {
    let Some((l, _, text)) = layout(app, id) else { return caret };
    photocraft_text::line_edge(&l, &text, caret, end)
}

/// Keyboard input while editing. Returns true when a type edit session is active (single-key
/// tool shortcuts must then be skipped). Handled events are removed from the frame's input.
pub fn handle_keys(app: &mut PhotocraftApp, ctx: &egui::Context) -> bool {
    crate::type_transform::cancel_stale(app);
    if !ctx.input(|i| i.focused) {
        crate::type_transform::cancel_drag(app);
    }
    let Some(ed) = app.ui.text_edit.clone() else { return false };
    // The font picker (and other focused fields) owns keyboard input while it is open.
    if ctx.text_edit_focused() || ctx.egui_wants_keyboard_input() {
        return true;
    }
    let id = LayerId(ed.layer);
    let Some(text) = current_text(app, id) else {
        app.ui.text_edit = None;
        return false;
    };
    // Undo/redo can shorten the text under us.
    let n = text.chars().count();
    if let Some(e) = app.ui.text_edit.as_mut() {
        e.caret = e.caret.min(n);
        e.anchor = e.anchor.min(n);
    }
    ctx.request_repaint_after(std::time::Duration::from_millis(530)); // caret blink
    let events = ctx.input(|i| i.events.clone());
    let vertical_flow = is_vertical(app, id);
    let mut handled = vec![false; events.len()];
    for (k, ev) in events.iter().enumerate() {
        let Some(ed) = app.ui.text_edit.clone() else { break };
        let (a, b) = (ed.caret.min(ed.anchor), ed.caret.max(ed.anchor));
        let text = current_text(app, id).unwrap_or_default();
        let n = text.chars().count();
        let set = |app: &mut PhotocraftApp, caret: usize, extend: bool| {
            if let Some(e) = app.ui.text_edit.as_mut() {
                e.caret = caret.min(n);
                if !extend {
                    e.anchor = e.caret;
                }
            }
        };
        handled[k] = true;
        match ev {
            egui::Event::Text(s) => insert_typed(app, s),
            egui::Event::Paste(s) => insert(app, s),
            // A bare line break from the IME is the Enter key, which the key arm handles.
            egui::Event::Ime(egui::ImeEvent::Preedit { text: s, .. } | egui::ImeEvent::Commit(s)) if s == "\n" || s == "\r" => handled[k] = false,
            egui::Event::Ime(egui::ImeEvent::Preedit { text, .. }) => ime_update(app, text, false),
            egui::Event::Ime(egui::ImeEvent::Commit(s)) => ime_update(app, s, true),
            egui::Event::Ime(_) => {}
            egui::Event::Copy => {
                copy_selection(app, ctx);
            }
            egui::Event::Cut => cut_selection(app, ctx),
            egui::Event::Key { key, pressed: true, modifiers: m, .. } => {
                use egui::Key;
                let key = &flow_key(*key, vertical_flow);
                match key {
                    // Photoshop's type formatting keys act on the selected characters and win
                    // over the global ⇧⌘K/U/B/I (Color Settings, Desaturate, Auto Color, Inverse)
                    // while typing. Each press is one history step.
                    Key::K | Key::U | Key::B | Key::I | Key::Slash | Key::Questionmark if m.command && m.shift && !m.alt => {
                        let which = match key {
                            Key::K => CharToggle::AllCaps,
                            Key::U => CharToggle::Underline,
                            Key::B => CharToggle::FauxBold,
                            Key::I => CharToggle::FauxItalic,
                            _ => CharToggle::Strikethrough,
                        };
                        toggle_char_style(app, which);
                    }
                    Key::H if m.command && m.shift && m.alt => toggle_hyphenate(app),
                    // ⌥↑/↓ leading -/+2 pt, ⇧⌥↑/↓ baseline shift +/-2 pt (10 with ⌘/Ctrl).
                    Key::ArrowUp | Key::ArrowDown if m.alt => {
                        let step = if m.command { 10.0 } else { 2.0 };
                        let up = *key == Key::ArrowUp;
                        if m.shift {
                            step_char_metric(app, "baselineShiftBy", if up { step } else { -step });
                        } else {
                            step_char_metric(app, "leadingBy", if up { -step } else { step });
                        }
                    }
                    // ⌥←/→ over a selection tracks it by -/+20 (100 with ⌘/Ctrl).
                    Key::ArrowLeft | Key::ArrowRight if m.alt && !m.shift && a != b => {
                        let step = if m.command { 100.0 } else { 20.0 };
                        step_char_metric(app, "trackingBy", if *key == Key::ArrowRight { step } else { -step });
                    }
                    Key::Backspace | Key::Delete => {
                        if a == b {
                            let fwd = *key == Key::Delete;
                            let to = match (fwd, m.alt, m.command) {
                                (false, _, true) => line_edge(app, id, a, false),
                                (false, true, _) => word_boundary(&text, a, false),
                                (false, _, _) => a.saturating_sub(1),
                                (true, true, _) => word_boundary(&text, a, true),
                                (true, _, _) => (a + 1).min(n),
                            };
                            if to != a {
                                if let Some(e) = app.ui.text_edit.as_mut() {
                                    e.anchor = to;
                                }
                                insert(app, "");
                            }
                        } else {
                            insert(app, "");
                        }
                    }
                    // Photoshop: Alt/Option+←/→ at a collapsed caret kerns the pair before it by
                    // 20/1000 em (100 with ⌘/Ctrl), one history step per press.
                    Key::ArrowLeft | Key::ArrowRight if m.alt && !m.shift && a == b => {
                        let step = if m.command { 100.0 } else { 20.0 };
                        kern_pair(app, id, ed.caret, if *key == Key::ArrowRight { step } else { -step });
                    }
                    Key::ArrowLeft | Key::ArrowRight => {
                        let fwd = *key == Key::ArrowRight;
                        // Word movement: ⌘/Ctrl (Photoshop's; Alt is kerning), and Alt+Shift
                        // extends the selection by words; Home/End go to the line edges.
                        let to = if m.command || m.alt {
                            word_boundary(&text, ed.caret, fwd)
                        } else if a != b && !m.shift {
                            if fwd { b } else { a }
                        } else if fwd {
                            ed.caret + 1
                        } else {
                            ed.caret.saturating_sub(1)
                        };
                        set(app, to, m.shift);
                    }
                    Key::ArrowUp | Key::ArrowDown => {
                        let to = if m.command {
                            if *key == Key::ArrowUp { 0 } else { n }
                        } else {
                            line_step(app, id, ed.caret, if *key == Key::ArrowUp { -1 } else { 1 })
                        };
                        set(app, to, m.shift);
                    }
                    Key::Home | Key::End => {
                        let to = if m.command {
                            if *key == Key::End { n } else { 0 }
                        } else {
                            line_edge(app, id, ed.caret, *key == Key::End)
                        };
                        set(app, to, m.shift);
                    }
                    Key::Enter if m.command => commit_with(app, *m),
                    Key::Enter => insert(app, "\n"),
                    // Photoshop types a tab character; layout advances it to the next tab stop.
                    Key::Tab if !m.command && !m.alt && !m.shift => insert(app, "\t"),
                    Key::Escape if crate::type_transform::active(app) => crate::type_transform::cancel_drag(app),
                    // The Type Mask tools: Escape always cancels, leaving the selection as it was.
                    Key::Escape if ed.mask => cancel(app),
                    // Preferences ▸ Type: with "Use Escape to Commit" off, Escape cancels the
                    // session (undoes it, removing a just-created layer) like the Cancel button.
                    Key::Escape if app.session.prefs().type_.use_esc_to_commit => commit(app),
                    Key::Escape => cancel(app),
                    Key::A if m.command => {
                        if let Some(e) = app.ui.text_edit.as_mut() {
                            e.anchor = 0;
                            e.caret = n;
                        }
                    }
                    // Photoshop: while editing type, ⌘/Ctrl+T shows or hides the Character panel
                    // (Free Transform of the layer being typed into is not what it means here).
                    Key::T if m.command && !m.shift && !m.alt => {
                        let _ = crate::menus::invoke(app, ctx, "window.panel.character", json!({}));
                    }
                    // Other command shortcuts (⌘Z, ⌘S, …) pass through to the menus.
                    _ if m.command => {
                        if *key == Key::Z {
                            crate::type_transform::cancel_drag(app);
                        }
                        handled[k] = false;
                    }
                    _ => {}
                }
            }
            egui::Event::Key { pressed: false, modifiers: m, .. } => handled[k] = !m.command,
            _ => handled[k] = false,
        }
    }
    if handled.iter().any(|h| *h) {
        let mut k = 0;
        ctx.input_mut(|i| {
            i.events.retain(|_| {
                let keep = !handled.get(k).copied().unwrap_or(false);
                k += 1;
                keep
            })
        });
    }
    app.ui.text_edit.is_some()
}

/// End the editing session. A new layer left empty is deleted; a new layer is named after its text.
pub fn commit(app: &mut PhotocraftApp) {
    commit_with(app, egui::Modifiers::NONE);
}

/// [`commit`] with the modifiers held while committing: a Type Mask adds its glyphs to the
/// selection with ⇧, subtracts them with ⌥ and intersects with ⇧⌥ (Photoshop); with none it
/// replaces the selection.
pub fn commit_with(app: &mut PhotocraftApp, mods: egui::Modifiers) {
    if app.ui.tool.is_type() {
        crate::type_transform::finish(app);
    }
    crate::type_transform::reset(app);
    let Some(ed) = app.ui.text_edit.take() else { return };
    let Some(text) = current_text(app, LayerId(ed.layer)) else { return };
    if ed.mask {
        // Type Mask: the text's glyphs become (or combine with) the selection and the working
        // layer goes, all in the session's one history step. No text: nothing changes.
        let operation = crate::tool_feedback::selection_mode(crate::state::Tool::HorizontalTypeMask, 0, mods);
        if !text.trim().is_empty()
            && let Err(e) = app.run("type.createSelection", json!({"layer": ed.layer, "removeLayer": true, "operation": operation, "coalesce": ed.session}))
        {
            app.ui.status = e;
            app.ui.status_error = true;
        }
        if current_text(app, LayerId(ed.layer)).is_some() {
            undo_session(app, &ed);
        }
        return;
    }
    if text.trim().is_empty() && ed.created {
        let _ = app.run("layer.delete", json!({"layer": ed.layer}));
    } else if ed.created {
        let name = photocraft_engine::type_cmds::layer_name(&text);
        let _ = app.run("type.edit", json!({"layer": ed.layer, "name": name, "coalesce": ed.session}));
    }
}

/// Selection highlight, caret and text frame over the canvas.
pub fn draw_overlay(app: &mut PhotocraftApp, painter: &egui::Painter, xf: &ViewXform) {
    let Some(ed) = app.ui.text_edit.clone() else { return };
    let id = LayerId(ed.layer);
    let Some((l, aff, text)) = layout(app, id) else { return };
    if ed.mask {
        draw_mask_overlay(app, painter, xf, id);
    }
    let t = crate::theme::Tokens::get(painter.ctx());
    // Text space → screen.
    let scr_t = |x: f32, y: f32| -> Pos2 {
        let p = aff.apply(Point::new(x as f64, y as f64));
        xf.to_screen(p.x as f32, p.y as f32)
    };
    // Line space (lines, clusters, carets) → screen: vertical type turns it 90° clockwise.
    let scr = |x: f32, y: f32| -> Pos2 {
        let (x, y) = l.to_text(x, y);
        scr_t(x, y)
    };
    // Tell the OS where the caret is: this is what enables the IME and places its candidate window.
    {
        let (x, top, bot) = l.caret(byte_of(&text, ed.caret));
        let (x, top, bot) = if l.lines.is_empty() { (0.0, -(12.0 * l.px_per_pt.max(1.0)), 3.0) } else { (x, top, bot) };
        let r = egui::Rect::from_two_pos(scr(x, top), scr(x, bot)).expand(1.0);
        painter.ctx().output_mut(|o| {
            o.ime = Some(egui::output::IMEOutput { purpose: egui::IMEPurpose::Normal, rect: r, cursor_rect: r, should_interrupt_composition: false });
        });
    }
    // Frame: paragraph text shows its box with handles; point text an underline per line.
    let mods = crate::workspace_ui::sticky_mods(app, painter.ctx().input(|i| i.modifiers));
    if crate::type_transform::draw(app, painter, xf, mods) {
        return;
    }
    let shape = app.session.active().and_then(|s| text_layer(&s.doc, id).map(|t| t.shape.clone()));
    let frame = Stroke::new(1.0, t.accent);
    match shape {
        Some(photocraft_doc::text::TextShape::Box { x, y, width, height }) => {
            let c = [scr_t(x, y), scr_t(x + width, y), scr_t(x + width, y + height), scr_t(x, y + height)];
            painter.add(egui::Shape::closed_line(c.to_vec(), frame));
            let mids = [c[0].lerp(c[1], 0.5), c[1].lerp(c[2], 0.5), c[2].lerp(c[3], 0.5), c[3].lerp(c[0], 0.5)];
            for p in c.iter().chain(mids.iter()) {
                let r = egui::Rect::from_center_size(*p, egui::vec2(7.0, 7.0));
                painter.rect_filled(r, 0.0, Color32::WHITE);
                painter.rect_stroke(r, 0.0, frame, egui::StrokeKind::Inside);
            }
        }
        _ => {
            for ln in &l.lines {
                // Horizontal: just under the baseline; vertical: the column's centre line.
                let y = if l.vertical { ln.baseline } else { ln.baseline + ln.descent * 0.25 };
                painter.line_segment([scr(ln.x0.min(0.0), y), scr(ln.x1.max(ln.x0 + 1.0), y)], Stroke::new(1.0, t.accent.gamma_multiply(0.8)));
            }
        }
    }
    // Uncommitted IME text is underlined.
    if let Some((ps, pl)) = ed.preedit {
        let (a, b) = (byte_of(&text, ps), byte_of(&text, ps + pl));
        for (li, ln) in l.lines.iter().enumerate() {
            let xs: Vec<(f32, f32)> =
                l.clusters.iter().filter(|c| c.line == li && c.range.start >= a && c.range.end <= b).map(|c| (c.x, c.x + c.advance)).collect();
            let (x0, x1) = xs.iter().fold((f32::MAX, f32::MIN), |(lo, hi), (p, q)| (lo.min(*p), hi.max(*q)));
            if x0 < x1 {
                let y = ln.baseline + ln.descent * 0.5;
                painter.line_segment([scr(x0, y), scr(x1, y)], Stroke::new(1.5, t.accent));
            }
        }
    }
    // Selection: per line, the clusters inside [a, b).
    let (a, b) = (byte_of(&text, ed.caret.min(ed.anchor)), byte_of(&text, ed.caret.max(ed.anchor)));
    if a < b {
        let fill = Color32::from_rgba_unmultiplied(t.accent.r(), t.accent.g(), t.accent.b(), 110);
        for (li, ln) in l.lines.iter().enumerate() {
            let xs: Vec<(f32, f32)> =
                l.clusters.iter().filter(|c| c.line == li && c.range.start >= a && c.range.end <= b).map(|c| (c.x, c.x + c.advance)).collect();
            let (mut x0, mut x1) = xs.iter().fold((f32::MAX, f32::MIN), |(lo, hi), (p, q)| (lo.min(*p), hi.max(*q)));
            // A selected line break shows as a small sliver past the line end.
            if b > ln.range.end && a <= ln.range.end {
                x1 = x1.max(ln.x1 + (ln.ascent + ln.descent) * 0.25);
                x0 = x0.min(ln.x1);
            }
            if x0 < x1 {
                let (top, bot) = (ln.baseline - ln.ascent, ln.baseline + ln.descent);
                painter.add(egui::Shape::convex_polygon(vec![scr(x0, top), scr(x1, top), scr(x1, bot), scr(x0, bot)], fill, Stroke::NONE));
            }
        }
    } else {
        // Blinking caret (Photoshop's ~0.53 s rhythm), solid while dragging.
        let time = painter.ctx().input(|i| i.time);
        if ed.dragging || (time * 1000.0 / 530.0) as i64 % 2 == 0 {
            let (x, top, bot) = l.caret(a);
            let (x, top, bot) = if l.lines.is_empty() { (0.0, -(12.0 * l.px_per_pt.max(1.0)), 3.0) } else { (x, top, bot) };
            let c = if crate::theme::Tokens::get(painter.ctx()).pro { Color32::WHITE } else { Color32::BLACK };
            painter.line_segment([scr(x, top), scr(x, bot)], Stroke::new(1.5, c));
            painter.line_segment([scr(x, top), scr(x, bot)], Stroke::new(0.75, Color32::BLACK));
        }
    }
}

/// The Type Mask tools while typing: Quick Mask's colour over the canvas with the glyphs clear,
/// as Photoshop shows the selection-to-be. The glyphs come from the hidden working layer's pixels.
fn draw_mask_overlay(app: &PhotocraftApp, painter: &egui::Painter, xf: &ViewXform, id: LayerId) {
    let Some(st) = app.session.active() else { return };
    let canvas = st.doc.bounds();
    let opts = app.session.quick_mask_options;
    let rgb = opts.color.to_rgb();
    let tint = |a: f32| {
        let a = (a * opts.opacity).clamp(0.0, 1.0);
        let q = |v: f32| (v.clamp(0.0, 1.0) * a * 255.0 + 0.5) as u8;
        Color32::from_rgba_premultiplied(q(rgb[0]), q(rgb[1]), q(rgb[2]), (a * 255.0 + 0.5) as u8)
    };
    let corners = |r: photocraft_geom::Rect| -> Vec<Pos2> {
        let (x0, y0, x1, y1) = (r.x0 as f32, r.y0 as f32, r.x1 as f32, r.y1 as f32);
        vec![xf.to_screen(x0, y0), xf.to_screen(x1, y0), xf.to_screen(x1, y1), xf.to_screen(x0, y1)]
    };
    let surface = text_layer(&st.doc, id).and_then(|t| t.cache.as_ref());
    let area = surface.map(|s| s.content_bounds().intersect(&canvas)).unwrap_or(photocraft_geom::Rect::EMPTY);
    // One unfeathered mesh, so the bands and the text's box meet without seams.
    let mut bands = egui::Mesh::default();
    let mut band = |x0: i32, y0: i32, x1: i32, y1: i32| {
        let r = photocraft_geom::Rect { x0, y0, x1, y1 };
        if !r.is_empty() {
            let base = bands.vertices.len() as u32;
            for p in corners(r) {
                bands.colored_vertex(p, tint(1.0));
            }
            bands.indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    };
    let Some(surface) = surface.filter(|_| !area.is_empty()) else {
        band(canvas.x0, canvas.y0, canvas.x1, canvas.y1);
        painter.add(egui::Shape::mesh(bands));
        return;
    };
    band(canvas.x0, canvas.y0, canvas.x1, area.y0);
    band(canvas.x0, area.y1, canvas.x1, canvas.y1);
    band(canvas.x0, area.y0, area.x0, area.y1);
    band(area.x1, area.y0, canvas.x1, area.y1);
    painter.add(egui::Shape::mesh(bands));
    // The text's box: a texture of the tint where there is no glyph, cached per revision.
    let ctx = painter.ctx();
    let key = (st.doc.id.0, st.revision, id.0);
    let cache = egui::Id::new("pc-type-mask-overlay");
    let tex = ctx.data(|d| d.get_temp::<((u64, u64, u64), egui::TextureHandle)>(cache)).filter(|(k, _)| *k == key).map(|(_, t)| t);
    let tex = tex.unwrap_or_else(|| {
        let max = (ctx.input(|i| i.max_texture_side) as u32).clamp(256, 4096);
        let factor = area.width().max(area.height()).div_ceil(max).max(1);
        let (w, h) = (area.width().div_ceil(factor) as usize, area.height().div_ceil(factor) as usize);
        let mut row = vec![[0.0f32; 4]; area.width() as usize];
        let mut px = Vec::with_capacity(w * h);
        for y in (area.y0..area.y1).step_by(factor as usize) {
            surface.read_rgba_into(photocraft_geom::Rect { x0: area.x0, y0: y, x1: area.x1, y1: y + 1 }, &mut row);
            px.extend(row.iter().step_by(factor as usize).take(w).map(|p| tint(1.0 - p[3])));
        }
        px.resize(w * h, tint(1.0));
        let img = egui::ColorImage::new([w, h], px);
        let t = ctx.load_texture("type-mask-overlay", img, egui::TextureOptions::LINEAR);
        ctx.data_mut(|d| d.insert_temp(cache, (key, t.clone())));
        t
    });
    let c = corners(area);
    let mut mesh = egui::Mesh::with_texture(tex.id());
    for (p, uv) in c.iter().zip([egui::pos2(0.0, 0.0), egui::pos2(1.0, 0.0), egui::pos2(1.0, 1.0), egui::pos2(0.0, 1.0)]) {
        mesh.vertices.push(egui::epaint::Vertex { pos: *p, uv, color: Color32::WHITE });
    }
    mesh.indices.extend([0, 1, 2, 0, 2, 3]);
    painter.add(egui::Shape::mesh(mesh));
}

/// Font families: bundled, system, and those the host serves (`photocraft_text::served`, fetched
/// when picked). Cached until the shared engine registers fonts or the served catalog changes.
pub fn families() -> Arc<Vec<String>> {
    type Cached = Option<((u64, u64), Arc<Vec<String>>)>;
    static FAMILIES: std::sync::Mutex<Cached> = std::sync::Mutex::new(None);
    let key = (photocraft_text::fonts::shared_generation(), photocraft_text::served::generation());
    let mut cached = FAMILIES.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((k, v)) = cached.as_ref()
        && *k == key
    {
        return v.clone();
    }
    let mut v = photocraft_text::shared().lock().map(|mut e| e.fonts.families()).unwrap_or_default();
    for f in photocraft_text::served::families() {
        if !v.contains(&f) {
            v.push(f);
        }
    }
    v.sort_by_key(|s| s.to_lowercase());
    let v = Arc::new(v);
    *cached = Some((key, v.clone()));
    v
}

/// Where [`sync_font_names`] leaves the font menus' localized labels for this frame.
fn font_labels_id() -> egui::Id {
    egui::Id::new("photocraft-font-menu-localized-names")
}

/// Preferences ▸ Type ▸ Show Font Names in English: when off, font menus list a family that has a
/// localized (native-language) name by that name, as Photoshop does ("MS Gothic" reads
/// "ＭＳ ゴシック"). Called once a frame; the menus read the labels from the context.
pub fn sync_font_names(ctx: &egui::Context, show_in_english: bool) {
    let labels = (!show_in_english).then(localized_families);
    ctx.data_mut(|d| {
        if let Some(labels) = labels {
            d.insert_temp(font_labels_id(), labels);
        } else {
            d.remove::<Arc<HashMap<String, String>>>(font_labels_id());
        }
    });
}

/// The fonts' localized family names, keyed by family. Cached until the shared engine registers fonts.
fn localized_families() -> Arc<HashMap<String, String>> {
    type Cached = Option<(u64, Arc<HashMap<String, String>>)>;
    static LOCALIZED: std::sync::Mutex<Cached> = std::sync::Mutex::new(None);
    let key = photocraft_text::fonts::shared_generation();
    let mut cached = LOCALIZED.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((k, v)) = cached.as_ref()
        && *k == key
    {
        return v.clone();
    }
    let v = Arc::new(photocraft_text::shared().lock().map(|mut e| e.fonts.localized_family_names()).unwrap_or_default());
    *cached = Some((key, v.clone()));
    v
}

/// The font menu's rows: each family with the label it shows (its localized name when `labels`
/// has one, else the family), matching `query` (lowercase) by either name. With localized labels
/// the rows sort by label, so the menu stays alphabetical as shown.
fn font_menu_rows<'a>(families: &'a [String], labels: Option<&'a HashMap<String, String>>, query: &str) -> Vec<(&'a String, &'a str)> {
    let label = |f: &'a String| labels.and_then(|l| l.get(f)).map_or(f.as_str(), String::as_str);
    let mut rows: Vec<(&String, &str)> = families
        .iter()
        .map(|f| (f, label(f)))
        .filter(|(f, l)| query.is_empty() || f.to_lowercase().contains(query) || l.to_lowercase().contains(query))
        .collect();
    if labels.is_some_and(|l| !l.is_empty()) {
        rows.sort_by_cached_key(|(_, l)| l.to_lowercase());
    }
    rows
}

fn weight_name(w: f32) -> &'static str {
    match w.round() as i32 {
        ..=150 => "Thin",
        151..=250 => "ExtraLight",
        251..=350 => "Light",
        351..=450 => "Regular",
        451..=550 => "Medium",
        551..=650 => "SemiBold",
        651..=750 => "Bold",
        751..=850 => "ExtraBold",
        _ => "Black",
    }
}

/// Style names available for a family: each face's actual OpenType subfamily (including numeric
/// and nonstandard ones), plus the standard weights a variable face's `wght` axis covers.
pub fn styles(family: &str) -> Vec<String> {
    let faces = photocraft_text::shared().lock().unwrap_or_else(|e| e.into_inner()).fonts.faces(family);
    let v = style_names(&faces);
    if v.is_empty() { vec!["Regular".into()] } else { v }
}

/// [`styles`] of these faces, lightest first. A variable face reports only its default instance
/// (Montserrat's is Thin), yet the layout draws any weight its `wght` axis covers, so the standard
/// weights in that range are listed too.
fn style_names(faces: &[photocraft_text::FaceInfo]) -> Vec<String> {
    let label = |weight: f32, italic: bool| {
        let name = match (weight_name(weight), italic) {
            ("Regular", true) => "Italic".to_string(),
            (w, true) => format!("{w} Italic"),
            (w, false) => w.to_string(),
        };
        (weight.round() as i32, italic, name)
    };
    let mut v: Vec<(i32, bool, String)> = faces.iter().map(|f| (f.weight.round() as i32, f.italic, f.style.clone())).collect();
    for f in faces {
        if let Some((_, min, _, max)) = f.axes.iter().find(|a| a.0 == "wght") {
            v.extend((100..=900).step_by(100).map(|w| w as f32).filter(|w| (*min..=*max).contains(w)).map(|w| label(w, f.italic)));
        }
    }
    v.sort();
    let mut names: Vec<String> = Vec::new();
    for (_, _, name) in v {
        if !names.contains(&name) {
            names.push(name);
        }
    }
    names
}

/// A variable-font axis: (tag, min, default, max), as [`photocraft_text::FaceInfo::axes`].
pub(crate) type Axis = (String, f32, f32, f32);

/// The axes the Character section shows as sliders (#2985): those of the faces matching the
/// style's italic, else of any face. Optical size has none: it follows the type size.
pub(crate) fn slider_axes(faces: &[photocraft_text::FaceInfo], italic: bool) -> Vec<Axis> {
    let pick = |want: Option<bool>| {
        let mut v: Vec<Axis> = Vec::new();
        for f in faces.iter().filter(|f| want.is_none_or(|w| f.italic == w)) {
            for a in &f.axes {
                let usable = a.0 != "opsz" && a.1.is_finite() && a.3.is_finite() && a.1 < a.3;
                if usable && !v.iter().any(|b| b.0 == a.0) {
                    v.push(a.clone());
                }
            }
        }
        v
    };
    let v = pick(Some(italic));
    if v.is_empty() { pick(None) } else { v }
}

/// The value an axis slider shows: the style's own value for the axis, else what the layout
/// draws (the style's weight on `wght`, the axis default otherwise), within the axis range.
pub(crate) fn axis_value(c: &photocraft_doc::text::CharStyle, (tag, min, default, max): &Axis) -> f32 {
    let v = c.variations.iter().find(|v| &v.axis == tag).map(|v| v.value).unwrap_or_else(|| match tag.as_str() {
        "wght" => f32::from(c.weight),
        "ital" if c.italic => *max,
        _ => *default,
    });
    let d = default.clamp(*min, *max);
    if v.is_finite() { v.clamp(*min, *max) } else { d }
}

/// Slider label for an axis: the registered axes by name, others by their tag (as the font
/// names them in its `fvar` table, e.g. `GRAD`).
fn axis_label(tag: &str) -> String {
    match tag {
        "wght" => tl!("Weight").to_string(),
        "wdth" => tl!("Width").to_string(),
        "slnt" => tl!("Slant").to_string(),
        "ital" => tl!("Italic").to_string(),
        _ => tag.to_string(),
    }
}

/// One slider row per variable-font axis of the style's font, below the font style (#2985).
fn axis_sliders(app: &mut PhotocraftApp, ui: &mut egui::Ui, c: &photocraft_doc::text::CharStyle) {
    // Drawn every frame: no sliders until the background font scan is done (#3021).
    let Some(engine) = photocraft_text::try_shared() else { return };
    let axes = {
        let faces = engine.lock().unwrap_or_else(PoisonError::into_inner).fonts.faces(&c.font_family);
        slider_axes(&faces, c.italic)
    };
    let t = crate::theme::Tokens::get(ui.ctx());
    for axis in &axes {
        let (min, max) = (axis.1, axis.3);
        let mut v = axis_value(c, axis);
        let mut changed = false;
        ui.push_id(("type-axis", &axis.0), |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(axis_label(&axis.0)).color(t.text_dim));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    changed |= crate::widgets::value_field(ui, &mut v, min..=max, "", 74.0).changed();
                });
            });
            changed |= crate::widgets::slider(ui, &mut v, min..=max, None).changed();
        });
        ui.add_space(crate::theme::ROW_GAP);
        if changed {
            // Whole units on wide axes (weight 100–900), hundredths on narrow ones (ital 0–1).
            let v = if max - min >= 10.0 { v.round() } else { (v * 100.0).round() / 100.0 };
            let mut m = serde_json::Map::new();
            m.insert(axis.0.clone(), json!(v));
            apply(app, ui.ctx(), json!({ "variations": m }));
        }
    }
}

/// PSD text can identify a face only by its PostScript name. Show its real subfamily too.
pub(crate) fn selected_style(style: &photocraft_doc::text::CharStyle) -> String {
    // Drawn every frame: while the font scan runs, the stored style name stands in.
    if let Some(ps) = &style.postscript_name
        && let Some(engine) = photocraft_text::try_shared()
    {
        let faces = engine.lock().unwrap_or_else(|e| e.into_inner()).fonts.faces(&style.font_family);
        if let Some(face) = faces.into_iter().find(|f| f.postscript_name.as_ref() == Some(ps)) {
            return face.style;
        }
    }
    if style.font_style.is_empty() { if style.italic { "Italic".into() } else { "Regular".into() } } else { style.font_style.clone() }
}

/// Localized style label for the dropdown. The raw string stays the engine's `fontStyle` key;
/// the words (weights and "Italic") are translated in the `fontWeight` context, where "Light"
/// means a weight — unlike the Camera Raw "Light" section.
pub fn style_label(style: &str) -> String {
    style.split(' ').map(|w| crate::i18n::tr_ctx(crate::i18n::current(), "fontWeight", w)).collect::<Vec<_>>().join(" ")
}

/// Starts scanning the installed fonts on a background thread (desktop start-up), so choosing
/// the Type tool doesn't stall the UI while they are read (#3021). Does nothing on the web.
pub fn warm_fonts() {
    photocraft_text::warm_shared();
}

/// Whether the font database is ready. The installed fonts are scanned on a background thread
/// from launch (`photocraft_text::warm_shared`); until that finishes, font menus draw this
/// disabled "Loading fonts…" box `width` wide instead of blocking the UI, and a repaint picks the
/// menus up once the scan is done.
pub(crate) fn fonts_ready(ui: &mut egui::Ui, width: f32) -> bool {
    if photocraft_text::try_shared().is_some() {
        return true;
    }
    let label = tl!("Loading fonts…").to_string();
    ui.add_enabled_ui(false, |ui| crate::widgets::dropdown(ui, "fonts-loading", &mut label.clone(), &[(label.clone(), label.as_str())], width));
    ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
    false
}

/// Searchable font-family combo box.
fn font_picker(ui: &mut egui::Ui, current: &mut String, width: f32, preview: photocraft_engine::prefs::FontPreview) -> bool {
    family_picker_in(ui, "type-font", current, width, &families(), preview)
}

/// Maximum height of the font menu.
const FONT_MENU_HEIGHT: f32 = 460.0;

/// [`font_picker`] over a given family list (tests pass their own).
#[cfg(test)]
fn font_picker_in(ui: &mut egui::Ui, current: &mut String, width: f32, families: &[String], preview: photocraft_engine::prefs::FontPreview) -> bool {
    family_picker_in(ui, "type-font", current, width, families, preview)
}

/// Shared by the Type options, Character, style and Glyphs panels. Selection changes still
/// flow through each caller's existing engine command.
pub(crate) fn family_picker_in(
    ui: &mut egui::Ui,
    salt: &str,
    current: &mut String,
    width: f32,
    families: &[String],
    preview: photocraft_engine::prefs::FontPreview,
) -> bool {
    let mut changed = false;
    let search_id = ui.id().with(("font-search", salt));
    // The list opens on the press, so a press-drag-release picks a font too (#2735).
    let (press, close) = crate::press_menu::PressCombo::before(ui, salt);
    let labels: Option<Arc<HashMap<String, String>>> = ui.ctx().data(|d| d.get_temp(font_labels_id()));
    let labels = labels.as_deref();
    let shown = labels.and_then(|l| l.get(current.as_str())).unwrap_or(current).clone();
    let combo = egui::ComboBox::from_id_salt(salt)
        .selected_text(shown)
        .width(width)
        .height(FONT_MENU_HEIGHT)
        .icon(crate::widgets::chevron_icon)
        // Clicks inside the menu (the search field, the scroll bar) must not close it (#1369);
        // picking a font closes it explicitly below.
        .close_behavior(if matches!(close, egui::PopupCloseBehavior::IgnoreClicks) { close } else { egui::PopupCloseBehavior::CloseOnClickOutside });

    let response = combo.show_ui(ui, |ui| {
        let (pass_id, focus_id, height_id) = (search_id.with("pass"), search_id.with("focus"), search_id.with("height"));
        let pass = ui.ctx().cumulative_pass_nr();
        let last_pass: Option<u64> = ui.data(|d| d.get_temp(pass_id));
        // Freshly opened (not drawn last pass): focus the search field whatever the query, and keep
        // asking until it has focus (the first pass is egui's invisible sizing pass) (#1369).
        let opened = last_pass.is_none_or(|p| p.saturating_add(1) < pass);
        let mut focus = opened || ui.data(|d| d.get_temp(focus_id)).unwrap_or(false);
        let mut q: String = ui.data(|d| d.get_temp(search_id)).unwrap_or_default();
        // Consume these before TextEdit and the canvas can interpret them as caret movement.
        let down = ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown));
        let up = ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp));
        let enter = ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Enter));
        let r = ui.add(egui::TextEdit::singleline(&mut q).hint_text(tl!("Search fonts")).desired_width(200.0));
        if r.has_focus() {
            focus = false;
        } else if focus {
            r.request_focus();
        }
        ui.data_mut(|d| {
            d.insert_temp(pass_id, pass);
            d.insert_temp(focus_id, focus);
            d.insert_temp(search_id, q.clone());
        });
        let ql = q.to_lowercase();
        let rows = font_menu_rows(families, labels, &ql);
        if down || up {
            let index = rows.iter().position(|(f, _)| *f == current);
            let next = match index {
                Some(i) if down => i.saturating_add(1).min(rows.len().saturating_sub(1)),
                Some(i) => i.saturating_sub(1),
                None if down => 0,
                None => rows.len().saturating_sub(1),
            };
            if let Some((f, _)) = rows.get(next) {
                changed = **f != *current;
                current.clone_from(f);
                // A served family not fetched yet: start now, as a click does.
                photocraft_text::served::request(f);
            }
        }
        let accept = enter && rows.iter().any(|(f, _)| *f == current);
        if accept {
            ui.data_mut(|d| d.remove::<String>(search_id));
            ui.close();
        }
        let preview = crate::font_preview::sample(preview);
        let row_h = preview.map_or(28.0, |(_, [_, h])| (h + 4.0).max(28.0));

        let mut hovered: Option<String> = None;
        for (f, label) in rows {
            let mut button = egui::Button::selectable(f == current, label).truncate();
            if let Some((size, [_, h])) = preview {
                button =
                    button.right_text(egui::Atom::custom(ui.id().with(("font-sample", f)), egui::vec2(crate::font_preview::slot_width(ui.ctx(), f, size), h)));
            }
            let response = ui.add_sized([330.0, row_h], button);
            if (down || up) && f == current {
                response.scroll_to_me(Some(egui::Align::Center));
            }
            if let Some((size, _)) = preview
                && ui.is_rect_visible(response.rect)
            {
                crate::font_preview::paint(ui, f, response.rect, size);
            }
            if response.hovered() {
                hovered = Some(f.clone());
            }
            if press.chosen(ui, &response) {
                *current = f.clone();
                // A served family not fetched yet: start now, before any text needs it.
                photocraft_text::served::request(f);
                changed = true;
                ui.data_mut(|d| d.remove::<String>(search_id));
                hovered = None;
                ui.close();
            }
        }
        // egui measures a popup only when it opens and never lets it grow back, so a filtered
        // list would leave the menu short after the query is cleared (#1369). Keep the unfiltered
        // height while searching.
        if ql.is_empty() {
            let h = ui.min_rect().height();
            ui.data_mut(|d| d.insert_temp(height_id, h));
        } else if let Some(h) = ui.data(|d| d.get_temp::<f32>(height_id)) {
            ui.set_min_height(h.min(FONT_MENU_HEIGHT));
        }
        if accept { None } else { hovered }
    });
    press.after(&response.response);
    let hovered_font = response.inner.unwrap_or_default();

    // The wheel over the closed button steps through the families, a notch each, like Photoshop's
    // options-bar font menu: no need to open the list (#3350).
    if !egui::ComboBox::is_open(ui.ctx(), response.response.id) && ui.is_enabled() {
        let hovered = response.response.hovered();
        if hovered {
            ui.input_mut(|i| i.smooth_scroll_delta = egui::Vec2::ZERO);
        }
        let notches = crate::widgets::wheel_notches_of(ui, response.response.id.with("family-wheel"), hovered, 1.0);
        let rows = font_menu_rows(families, labels, "");
        if let Some(i) = rows.iter().position(|(f, _)| *f == current)
            && let Some(n) = wheel_family_index(i, rows.len(), notches)
            && let Some((f, _)) = rows.get(n)
            && **f != *current
        {
            current.clone_from(f);
            photocraft_text::served::request(f);
            changed = true;
        }
    }

    if salt == "type-font" {
        let hover_id = egui::Id::new("type-font-hover");
        if let Some(f) = hovered_font {
            ui.ctx().data_mut(|d| d.insert_temp(hover_id, f));
        } else {
            ui.ctx().data_mut(|d| d.remove::<String>(hover_id));
        }
    }

    changed
}

/// The row `notches` wheel notches from row `index` of `len` (down, negative, is the next one),
/// stopping at the ends; none when the wheel did not move.
fn wheel_family_index(index: usize, len: usize, notches: f32) -> Option<usize> {
    let steps = notches.abs().min(len as f32) as usize;
    if steps == 0 || len == 0 {
        return None;
    }
    let last = len - 1;
    Some(if notches < 0.0 { index.saturating_add(steps).min(last) } else { index.saturating_sub(steps) })
}

/// Representative for the displayed properties: the edit layer, else a selected type layer.
pub(crate) fn target(app: &PhotocraftApp) -> Option<(u64, Option<[usize; 2]>)> {
    if let Some(ed) = &app.ui.text_edit {
        let (a, b) = (ed.caret.min(ed.anchor), ed.caret.max(ed.anchor));
        return Some((ed.layer, (a < b).then_some([a, b])));
    }
    let st = app.session.active()?;
    let id = st
        .active_layer
        .filter(|id| text_layer(&st.doc, *id).is_some())
        .or_else(|| st.selected_layers().into_iter().find(|id| text_layer(&st.doc, *id).is_some()))?;
    Some((id.0, None))
}

/// The type layer whose properties the Type controls and dialogs show (see [`target`]).
pub(crate) fn target_text(app: &PhotocraftApp) -> Option<&TextLayer> {
    let (id, _) = target(app)?;
    text_layer(&app.session.active()?.doc, LayerId(id))
}

/// Snapshot the mutation scope separately from the representative used to display properties.
/// An inline edit remains on its layer/range; layer selection formats every selected type layer.
pub(crate) fn formatting_params(app: &PhotocraftApp) -> Option<serde_json::Value> {
    if let Some(ed) = &app.ui.text_edit {
        let mut params = json!({"layer": ed.layer, "coalesce": ed.session});
        let (a, b) = (ed.caret.min(ed.anchor), ed.caret.max(ed.anchor));
        if a < b {
            params["range"] = json!([a, b]);
        }
        return Some(params);
    }
    let st = app.session.active()?;
    let layers: Vec<u64> = st.selected_layers().into_iter().filter(|id| text_layer(&st.doc, *id).is_some()).map(|id| id.0).collect();
    (!layers.is_empty()).then(|| json!({"layers": layers}))
}

/// Scale of the target layer's transform. Like Photoshop, sizes show and edit as the layer
/// appears: a 12 pt layer scaled 200% (common in PSDs) reads 24 pt.
fn shown_scale(app: &PhotocraftApp) -> f32 {
    let Some((id, _)) = target(app) else { return 1.0 };
    let Some(t) = app.session.active().and_then(|st| text_layer(&st.doc, LayerId(id))) else { return 1.0 };
    let m = t.transform.m;
    let k = (m[0] * m[3] - m[1] * m[2]).abs().sqrt() as f32;
    if k.is_finite() && k > 1e-3 { k } else { 1.0 }
}

/// The display unit for type metrics (Preferences ▸ Units & Rulers ▸ Type): the multiplier from
/// the model's points to the unit, and its label. Pixels follow the document's resolution.
pub(crate) fn type_unit(app: &PhotocraftApp) -> (f32, &'static str) {
    let dpi = app.session.active().map_or(72.0, |d| d.doc.resolution_dpi.max(1.0));
    match app.session.prefs().units_and_rulers.type_units {
        photocraft_engine::prefs::TypeUnit::Points => (1.0, "pt"),
        photocraft_engine::prefs::TypeUnit::Pixels => (dpi / 72.0, "px"),
        photocraft_engine::prefs::TypeUnit::Millimeters => (25.4 / 72.0, "mm"),
    }
}

/// Coalesce key while a field is being scrubbed: every step of one drag shares one history step
/// (Photoshop's single step per scrub, #124). Unique per press, so two drags are two steps.
fn drag_key(ctx: &egui::Context) -> Option<String> {
    let id = ctx.dragged_id()?;
    let t = ctx.input(|i| i.pointer.press_start_time())?;
    Some(format!("type-drag-{}-{}", id.value(), t.to_bits()))
}

/// Apply only the supplied properties to the text-edit scope or selected type layers.
fn apply(app: &mut PhotocraftApp, ctx: &egui::Context, props: serde_json::Value) {
    let Some(mut p) = formatting_params(app) else { return };
    if let Some(props) = props.as_object() {
        for (key, value) in props {
            p[key] = value.clone();
        }
    }
    if let Some(key) = app.ui.text_edit.as_ref().map(|ed| ed.session.clone()).or_else(|| drag_key(ctx)) {
        p["coalesce"] = json!(key);
    }
    let _ = app.run("type.setStyle", p);
}

/// `props` aimed at the type target: its layer, the selected characters while typing, and the
/// typing session's undo step. `None` without a type target.
pub(crate) fn with_target(app: &PhotocraftApp, mut props: serde_json::Value) -> Option<serde_json::Value> {
    let (layer, range) = target(app)?;
    props["layer"] = json!(layer);
    if let Some(r) = range {
        props["range"] = json!(r);
    }
    if let Some(key) = app.ui.text_edit.as_ref().map(|ed| ed.session.clone()) {
        props["coalesce"] = json!(key);
    }
    Some(props)
}

/// Open the shared Color Picker with a snapshot of the text target. Sampling or cancelling
/// the dialog never changes the text selection, tool colours, or document history.
pub fn open_color_picker(app: &mut PhotocraftApp, rgb: [f32; 3]) -> u64 {
    if let Some(params) = formatting_params(app) {
        crate::color_picker_ui::open_for_command(app, "Color Picker (Text Color)", rgb, "type.setStyle", params)
    } else {
        crate::color_picker_ui::open(app, "foreground")
    }
}

/// While characters are selected in a type layer, a new foreground colour (Color and Swatches
/// panels, the Color Picker) recolours them, in the editing session's history step.
pub fn foreground_changed(app: &mut PhotocraftApp) {
    let Some((layer, Some(range))) = target(app) else { return };
    let Some(ed) = app.ui.text_edit.as_ref() else { return };
    let p = json!({"layer": layer, "range": range, "color": hex(app.session.tools.foreground), "coalesce": ed.session});
    let _ = app.run("type.setStyle", p);
}

/// Photoshop's Type options bar.
pub fn options_bar(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let t = crate::theme::Tokens::get(ui.ctx());
    // Show the target layer's (first-run) style, else the tool defaults.
    let shown = target(app).and_then(|(id, _)| {
        let st = app.session.active()?;
        let tl = text_layer(&st.doc, LayerId(id))?;
        let run = tl.char_runs().into_iter().next().map(|r| r.style);
        // The size at the selection (else the first run), as the layer shows it (#124).
        let size = styles_at(app).map_or(tl.size_pt, |(c, _)| c.size_pt) * shown_scale(app);
        Some((tl.font_family.clone(), run.as_ref().map(selected_style).unwrap_or_default(), size, styles_at(app).map_or(tl.color, |(c, _)| c.color)))
    });
    let o = app.ui.tool_options.clone();
    let (mut fam, mut style, size) = match &shown {
        Some((f, s, z, _)) => (f.clone(), if s.is_empty() { "Regular".into() } else { s.clone() }, *z),
        None => (o.type_font.clone(), o.type_style.clone(), o.type_size),
    };
    if crate::icons::button(ui, "text-cursor", 24.0, false, tl!("Toggle text orientation")).clicked()
        && let Some((layer, _)) = target(app)
        && let Some(params) = formatting_params(app)
    {
        let to = if is_vertical(app, LayerId(layer)) { "horizontal" } else { "vertical" };
        if let Err(e) = app.run(&format!("type.orientation.{to}"), params) {
            app.ui.status = e;
            app.ui.status_error = true;
        }
    }
    // Font family and style menus (one placeholder in their place while the fonts load).
    if fonts_ready(ui, 170.0 + ui.spacing().item_spacing.x + 110.0) {
        if font_picker(ui, &mut fam, 170.0, app.session.prefs().type_.font_preview) {
            app.ui.tool_options.type_font = fam.clone();
            let st = styles(&fam);
            style = if st.contains(&style) { style } else { st.first().cloned().unwrap_or_else(|| "Regular".into()) };
            app.ui.tool_options.type_style = style.clone();
            apply(app, ui.ctx(), json!({"font": fam}));
        }
        let opts: Vec<(String, String)> = styles(&fam).into_iter().map(|s| (s.clone(), style_label(&s))).collect();
        let opts_ref: Vec<(String, &str)> = opts.iter().map(|(a, b)| (a.clone(), b.as_str())).collect();
        if crate::widgets::dropdown(ui, "type-style", &mut style, &opts_ref, 110.0) {
            app.ui.tool_options.type_style = style.clone();
            apply(app, ui.ctx(), json!({"fontStyle": style}));
        }
    }
    let (r, _) = ui.allocate_exact_size(egui::vec2(18.0, 22.0), egui::Sense::hover());
    crate::icons::paint(ui, r, "type", 13.0, t.icon);
    let (unit, label) = type_unit(app);
    let mut display = size * unit;
    if crate::widgets::value_field(ui, &mut display, unit..=1296.0 * unit, label, 66.0).changed() {
        let size = display / unit;
        app.ui.tool_options.type_size = size;
        apply(app, ui.ctx(), json!({"size": size, "metricsAsShown": true}));
    }
    let mut aa = target(app)
        .and_then(|(id, _)| app.session.active().and_then(|st| text_layer(&st.doc, LayerId(id))))
        .map(|t| photocraft_engine::type_extra_cmds::aa_name(t.antialias).to_string())
        .unwrap_or_else(|| o.type_aa.clone());
    let aa_opts = [
        ("none".to_string(), tl!("None")),
        ("sharp".to_string(), tl!("Sharp")),
        ("crisp".to_string(), tl!("Crisp")),
        ("strong".to_string(), tl!("Strong")),
        ("smooth".to_string(), tl!("Smooth")),
    ];
    if crate::widgets::dropdown(ui, "type-aa", &mut aa, &aa_opts, 80.0) {
        app.ui.tool_options.type_aa = aa.clone();
        if let Some(params) = formatting_params(app) {
            let _ = app.run(&format!("type.antiAlias.{aa}"), params);
        }
    }
    crate::widgets::vline(ui, 22.0);
    ui.spacing_mut().item_spacing.x = 2.0;
    for (align, icon, tip) in
        [("left", "align-left", tl!("Left align text")), ("center", "align-center", tl!("Center text")), ("right", "align-right", tl!("Right align text"))]
    {
        if crate::icons::button(ui, icon, 24.0, o.type_align == align, tip).clicked() {
            app.ui.tool_options.type_align = align.into();
            apply(app, ui.ctx(), json!({"align": align}));
        }
    }
    ui.spacing_mut().item_spacing.x = 8.0;
    crate::widgets::vline(ui, 22.0);
    // Text colour swatch with a picker popup.
    let c = shown
        .as_ref()
        .map(|s| s.3)
        .map(|c| {
            let v = c.to_rgb();
            [v[0], v[1], v[2], 1.0]
        })
        .unwrap_or(app.session.tools.foreground);
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(28.0, 18.0), egui::Sense::click());
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    ui.painter().rect_filled(rect, 2.0, Color32::from_rgb(q(c[0]), q(c[1]), q(c[2])));
    ui.painter().rect_stroke(rect, 2.0, Stroke::new(1.0, t.field_border), egui::StrokeKind::Outside);
    let resp = resp.on_hover_text(tl!("Set the text color"));
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, tl!("Set the text color")));
    if resp.clicked() {
        open_color_picker(app, [c[0], c[1], c[2]]);
    }
    if app.ui.text_edit.is_some() {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_space(8.0);
            if crate::icons::button(
                ui,
                "check",
                24.0,
                false,
                &crate::i18n::fmt(tl!("Commit any current edits ({key})"), &[("key", &crate::shortcuts::pretty("Cmd+Enter"))]),
            )
            .clicked()
            {
                commit_with(app, ui.input(|i| i.modifiers));
            }
            if crate::icons::button(ui, "ban", 24.0, false, tl!("Cancel any current edits (Esc)")).clicked() {
                cancel(app);
            }
        });
    }
}

/// Character and paragraph style at the target (selection start, else the layer's first run).
pub(crate) fn styles_at(app: &PhotocraftApp) -> Option<(photocraft_doc::text::CharStyle, photocraft_doc::text::ParagraphStyle)> {
    let (layer, range) = target(app)?;
    let st = app.session.active()?;
    let t = text_layer(&st.doc, LayerId(layer))?;
    let at = range.map_or(0, |r| byte_of(&t.text, r[0]));
    let pick = |lens: Vec<usize>| -> usize {
        let mut acc = 0;
        for (i, len) in lens.iter().enumerate() {
            acc += len;
            if at < acc {
                return i;
            }
        }
        lens.len().saturating_sub(1)
    };
    let runs = t.char_runs();
    let paras = t.paragraph_runs();
    let c = runs.get(pick(runs.iter().map(|r| r.len).collect())).map(|r| r.style.clone())?;
    let p = paras.get(pick(paras.iter().map(|r| r.len).collect())).map(|r| r.style.clone()).unwrap_or_default();
    Some((c, p))
}

/// Kerning shown for the target: at a collapsed caret, the pair before it (Photoshop's
/// Character panel); else the style at the selection.
fn kerning_at(app: &PhotocraftApp) -> Option<(photocraft_doc::text::Kerning, f32)> {
    let ed = app.ui.text_edit.as_ref().filter(|e| e.caret == e.anchor && e.caret > 0)?;
    let st = app.session.active()?;
    let t = text_layer(&st.doc, LayerId(ed.layer))?;
    let at = byte_of(&t.text, ed.caret - 1);
    let mut end = 0;
    t.char_runs()
        .into_iter()
        .find(|r| {
            end += r.len;
            at < end
        })
        .map(|r| (r.style.kerning, r.style.kern))
}

/// The Character panel's kerning text: "Metrics", "Optical" or the manual value.
pub(crate) fn kerning_label((mode, kern): (photocraft_doc::text::Kerning, f32)) -> String {
    use photocraft_doc::text::Kerning;
    match mode {
        _ if kern != 0.0 && kern.is_finite() => format!("{}", kern.round() as i64),
        Kerning::Metrics => "Metrics".into(),
        Kerning::Optical => "Optical".into(),
        Kerning::Off => "0".into(),
    }
}

/// Parses a typed kerning value: Metrics / Optical (any case) or a number in 1/1000 em
/// (Photoshop's range, -1000..10000). `None` for anything else.
pub(crate) fn parse_kerning(s: &str) -> Option<serde_json::Value> {
    let s = s.trim();
    match s.to_ascii_lowercase().as_str() {
        "metrics" => Some(json!("metrics")),
        "optical" => Some(json!("optical")),
        _ => crate::numeric_expression::parse(s).filter(|v| v.is_finite() && (-1000.0..=10000.0).contains(v)).map(|v| json!(v.round())),
    }
}

/// Photoshop's editable kerning combo: type a value or Metrics/Optical, or pick a preset.
fn kerning_field(ui: &mut egui::Ui, shown: &str, width: f32) -> Option<serde_json::Value> {
    let id = ui.make_persistent_id("props-kern-text");
    let mut buf: String = ui.data_mut(|d| d.get_temp::<String>(id)).unwrap_or_else(|| shown.to_string());
    let mut out = None;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        let resp = ui.add(egui::TextEdit::singleline(&mut buf).desired_width((width - 40.0).max(24.0)).id(id.with("edit")));
        if resp.has_focus() {
            ui.data_mut(|d| d.insert_temp(id, buf.clone()));
        } else {
            ui.data_mut(|d| d.remove::<String>(id));
        }
        if resp.lost_focus() && buf.trim() != shown {
            out = parse_kerning(&buf);
        }
        let presets = ["Metrics", "Optical", "0", "-100", "-75", "-50", "-25", "-10", "-5", "5", "10", "25", "50", "75", "100", "200"];
        let (press, close) = crate::press_menu::PressCombo::before(ui, "props-kern-presets");
        let combo = egui::ComboBox::from_id_salt("props-kern-presets")
            .selected_text("")
            .width(16.0)
            .height(420.0)
            .icon(crate::widgets::chevron_icon)
            .close_behavior(close)
            .show_ui(ui, |ui| {
                for p in presets {
                    let item = ui.selectable_label(p == shown, p);
                    if press.chosen(ui, &item) {
                        out = parse_kerning(p);
                    }
                }
            });
        press.after(&combo.response);
    });
    out
}

/// Applies a kerning value: at a collapsed caret to the pair before it, else to the selection
/// (or the whole layer).
fn apply_kerning(app: &mut PhotocraftApp, ctx: &egui::Context, v: serde_json::Value) {
    match app.ui.text_edit.clone().filter(|e| e.caret == e.anchor) {
        Some(ed) if ed.caret > 0 => {
            let _ = app.run("type.edit", json!({"layer": ed.layer, "range": [ed.caret - 1, ed.caret], "kerning": v}));
        }
        Some(_) => {}
        None => apply(app, ctx, json!({ "kerning": v })),
    }
}

fn icon_label(ui: &mut egui::Ui, icon: &str, tip: &str) {
    let t = crate::theme::Tokens::get(ui.ctx());
    let (r, resp) = ui.allocate_exact_size(egui::vec2(crate::props_layout::LABEL_W, 22.0), egui::Sense::hover());
    crate::icons::paint(ui, r, icon, 12.0, t.text_dim);
    // Photoshop's Character panel icons scrub the field beside them.
    crate::widgets::scrub_label(ui, &resp.on_hover_text(tip));
}

/// Labelled numeric field (Photoshop's icon + value pairs). Returns the new value when edited.
fn num_field(ui: &mut egui::Ui, label: &str, tip: &str, v: f32, range: std::ops::RangeInclusive<f32>, unit: &str, width: f32) -> Option<f32> {
    let t = crate::theme::Tokens::get(ui.ctx());
    let mut v = v;
    let mut out = None;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        let (r, resp) = ui.allocate_exact_size(egui::vec2(22.0, 22.0), egui::Sense::hover());
        ui.painter().text(r.center(), egui::Align2::CENTER_CENTER, label, crate::theme::semibold(10.0), t.text_dim);
        crate::widgets::scrub_label(ui, &resp.on_hover_text(tl!(tip)));
        let response = crate::widgets::value_field(ui, &mut v, range, unit, width);
        response.widget_info(|| {
            let mut info = egui::WidgetInfo::drag_value(ui.is_enabled(), f64::from(v));
            info.label = Some(tl!(tip).to_owned());
            info
        });
        if response.changed() {
            out = Some(v);
        }
    });
    out
}

/// Photoshop's paragraph alignment glyphs: four text lines whose widths/offsets show the mode
/// (justify variants run full width, with the last line placed left/centre/right).
fn align_glyph(p: &egui::Painter, r: egui::Rect, a: photocraft_doc::text::TextAlign, c: Color32) {
    use photocraft_doc::text::TextAlign as A;
    let widths = [1.0, 0.62, 1.0, 0.62];
    for (i, w) in widths.iter().enumerate() {
        let y = r.top() + i as f32 * r.height() / 3.0;
        let full = r.width();
        let (len, x0) = match a {
            A::Left => (full * w, r.left()),
            A::Center => (full * w, r.center().x - full * w / 2.0),
            A::Right => (full * w, r.right() - full * w),
            A::JustifyAll => (full, r.left()),
            _ if i < 3 => (full, r.left()),
            A::JustifyLeft => (full * 0.55, r.left()),
            A::JustifyCenter => (full * 0.55, r.center().x - full * 0.275),
            _ => (full * 0.55, r.right() - full * 0.55),
        };
        p.line_segment([egui::pos2(x0, y), egui::pos2(x0 + len, y)], Stroke::new(1.5, c));
    }
}

/// A toggle button (faux styles, paragraph alignment) of width `w`, painted by `paint`.
fn toggle_cell(ui: &mut egui::Ui, w: f32, on: bool, tip: &str, paint: impl FnOnce(&egui::Ui, egui::Rect, Color32)) -> bool {
    let t = crate::theme::Tokens::get(ui.ctx());
    let (r, resp) = ui.allocate_exact_size(egui::vec2(w, 24.0), egui::Sense::click());
    let bg = if on {
        t.accent_soft
    } else if resp.hovered() {
        t.hover
    } else {
        Color32::TRANSPARENT
    };
    ui.painter().rect_filled(r, t.radius_sm, bg);
    if on {
        ui.painter().rect_stroke(r, t.radius_sm, Stroke::new(1.0, t.accent_border), egui::StrokeKind::Inside);
    }
    paint(ui, r, if on { t.text } else { t.text_dim });
    let tip = tip.to_string();
    resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::Button, true, on, &tip));
    resp.on_hover_text(tl!(&tip)).clicked()
}

/// The Character panel's OpenType buttons, in Photoshop's order: (feature tag, command id,
/// tooltip). The tooltips are the Type › OpenType menu labels.
pub(crate) const OPENTYPE_BUTTONS: [(&str, &str, &str); 8] = [
    ("liga", "type.openType.standardLigatures", "Standard Ligatures"),
    ("calt", "type.openType.contextualAlternates", "Contextual Alternates"),
    ("dlig", "type.openType.discretionaryLigatures", "Discretionary Ligatures"),
    ("swsh", "type.openType.swash", "Swash"),
    ("salt", "type.openType.stylisticAlternates", "Stylistic Alternates"),
    ("titl", "type.openType.titlingAlternates", "Titling Alternates"),
    ("ordn", "type.openType.ordinals", "Ordinals"),
    ("frac", "type.openType.fractions", "Fractions"),
];

/// Photoshop's OpenType button row: each shows pressed while its feature is on at the type
/// target (the selection start, else the layer) and toggles it through the Type › OpenType
/// command, on the same scope as the other Character controls.
fn opentype_row(app: &mut PhotocraftApp, ui: &mut egui::Ui, c: &photocraft_doc::text::CharStyle, full: f32) {
    let cw = cell_width(full, OPENTYPE_BUTTONS.len());
    let mut clicked = None;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        for (tag, id, tip) in OPENTYPE_BUTTONS {
            let on = photocraft_engine::type_extra_cmds::feature_on(c, tag);
            if toggle_cell(ui, cw, on, tip, |ui, r, col| opentype_glyph(ui.painter(), r, tag, col)) {
                clicked = Some((id, on));
            }
        }
    });
    if let Some((id, on)) = clicked {
        toggle_opentype(app, id, on);
    }
}

/// Runs an OpenType toggle command on the Character panel's scope: the edited layer's selected
/// characters (or its whole text) while typing, else every selected type layer. The new state
/// is explicit, so it follows the button the user saw rather than the layer's first run.
pub(crate) fn toggle_opentype(app: &mut PhotocraftApp, id: &str, shown_on: bool) {
    let Some(mut p) = formatting_params(app) else { return };
    p["on"] = json!(!shown_on);
    let _ = app.run(id, p);
}

/// Small glyph labels for the OpenType buttons (drawn here, no Adobe artwork): fi, st, ct with
/// a ligature arc, a swash A, aa, T, 1st and ½.
fn opentype_glyph(p: &egui::Painter, r: egui::Rect, tag: &str, col: Color32) {
    let font = egui::FontId::proportional(13.0);
    let centred = |s: &str, font: egui::FontId| {
        let g = p.layout_no_wrap(s.to_string(), font, col);
        let pos = r.center() - g.size() / 2.0;
        let gr = egui::Rect::from_min_size(pos, g.size());
        p.galley(pos, g, col);
        gr
    };
    match tag {
        "liga" => {
            centred("fi", font);
        }
        "calt" => {
            centred("st", font);
        }
        "dlig" => {
            let gr = centred("ct", font);
            // The ligature's connecting arc from the c over to the t.
            let a = egui::pos2(gr.left() + gr.width() * 0.3, gr.top() + gr.height() * 0.42);
            let b = egui::pos2(gr.right() - gr.width() * 0.2, gr.top() + 4.0);
            let mid = egui::pos2((a.x + b.x) / 2.0, gr.top() + 1.5);
            p.line_segment([a, mid], Stroke::new(1.0, col));
            p.line_segment([mid, b], Stroke::new(1.0, col));
        }
        "swsh" => {
            let gr = centred("A", crate::theme::semibold(13.0));
            // A swash tail sweeping out under the letter.
            let y = gr.bottom() - 3.0;
            p.line_segment([egui::pos2(gr.left() - 3.0, y + 1.5), egui::pos2(gr.right() + 2.0, y - 0.5)], Stroke::new(1.0, col));
        }
        "salt" => {
            centred("aa", font);
        }
        "titl" => {
            centred("T", crate::theme::semibold(13.0));
        }
        "ordn" => {
            let one = p.layout_no_wrap("1".into(), font, col);
            let st = p.layout_no_wrap("st".into(), egui::FontId::proportional(9.0), col);
            let left = r.center().x - (one.size().x + st.size().x) / 2.0;
            let top = r.center().y - one.size().y / 2.0;
            let st_x = left + one.size().x;
            p.galley(egui::pos2(left, top), one, col);
            p.galley(egui::pos2(st_x, top - 1.0), st, col);
        }
        _ => {
            centred("½", font);
        }
    }
}

/// Width of each of `n` equal cells filling `avail` with 2 pt gaps.
fn cell_width(avail: f32, n: usize) -> f32 {
    let n = n.max(1) as f32;
    ((avail - (n - 1.0) * 2.0) / n).floor().max(20.0)
}

/// Properties panel sections for a type layer: Character, Paragraph and Type Options (#155).
pub fn type_properties(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    type_sections(app, ui, true, true);
    type_options(app, ui);
}

/// Window › Character / Paragraph (#150): the same controls as Properties, in their own dock
/// group. Without a type layer (or a type edit) there is nothing to show.
pub fn character_panel(app: &mut PhotocraftApp, ui: &mut egui::Ui, paragraph: bool) {
    if styles_at(app).is_none() {
        let t = crate::theme::Tokens::get(ui.ctx());
        ui.add_space(8.0);
        ui.label(egui::RichText::new(tl!("Select a type layer to edit its character and paragraph settings.")).color(t.text_dim));
        return;
    }
    type_sections(app, ui, !paragraph, paragraph);
}

/// The Character and/or Paragraph sections, with the shared collapsible headers and fields that
/// fill the panel width (#155).
fn type_sections(app: &mut PhotocraftApp, ui: &mut egui::Ui, character: bool, paragraph: bool) {
    use crate::props_layout::{COL_GAP, LABEL_GAP, LABEL_W, field_width, section};
    use crate::theme::ROW_GAP;
    let Some((c, para)) = styles_at(app) else { return };
    let t = crate::theme::Tokens::get(ui.ctx());
    // Row helper: `ui.horizontal` with the column gap, then the shared row gap.
    let row = |ui: &mut egui::Ui, add: &mut dyn FnMut(&mut egui::Ui)| {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = COL_GAP;
            add(ui)
        });
        ui.add_space(ROW_GAP);
    };
    if character && section(ui, "character", tl!("Character")) {
        let full = ui.available_width();
        let w = field_width(full, 2, LABEL_W);
        let mut fam = c.font_family.clone();
        let mut ready = true;
        row(ui, &mut |ui| {
            ready = fonts_ready(ui, full);
            if ready && font_picker(ui, &mut fam, full, app.session.prefs().type_.font_preview) {
                app.ui.tool_options.type_font = fam.clone();
                apply(app, ui.ctx(), json!({"font": fam}));
            }
        });
        row(ui, &mut |ui| {
            if !ready {
                return;
            }
            let mut style = selected_style(&c);
            let opts: Vec<(String, String)> = styles(&fam).into_iter().map(|s| (s.clone(), style_label(&s))).collect();
            let opts_ref: Vec<(String, &str)> = opts.iter().map(|(a, b)| (a.clone(), b.as_str())).collect();
            if crate::widgets::dropdown(ui, "props-type-style", &mut style, &opts_ref, full) {
                apply(app, ui.ctx(), json!({"fontStyle": style}));
            }
        });
        axis_sliders(app, ui, &c);
        row(ui, &mut |ui| {
            let k = shown_scale(app);
            // Hover tips stay English until wired through `tl!` in a multi-locale change
            // (`every_tl_literal_is_translated` requires every complete catalog).
            if let Some(v) = num_field(ui, "tT", "Font size", c.size_pt * k, 0.1..=1296.0, "pt", w) {
                app.ui.tool_options.type_size = v;
                apply(app, ui.ctx(), json!({"size": v, "metricsAsShown": true}));
            }
            let lead = c.leading_pt.unwrap_or(c.size_pt * para.auto_leading.max(0.01)) * k;
            if let Some(v) = num_field(ui, "A↕", "Leading (set to the font size × auto-leading when Auto)", lead, 0.1..=5000.0, "pt", w) {
                apply(app, ui.ctx(), json!({"leading": v, "metricsAsShown": true}));
            }
        });
        row(ui, &mut |ui| {
            let shown = kerning_label(kerning_at(app).unwrap_or((c.kerning, c.kern)));
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = LABEL_GAP;
                icon_label(ui, "text-cursor", "Kerning: Metrics, Optical or a value in 1/1000 em (Alt+←/→ at the caret)");
                if let Some(v) = kerning_field(ui, &shown, w) {
                    apply_kerning(app, ui.ctx(), v);
                }
            });
            if let Some(v) = num_field(ui, "VA", "Tracking (1/1000 em)", c.tracking, -1000.0..=10000.0, "", w) {
                apply(app, ui.ctx(), json!({"tracking": v}));
            }
        });
        row(ui, &mut |ui| {
            if let Some(v) = num_field(ui, "↕T", "Vertical scale", c.vertical_scale * 100.0, 0.0..=1000.0, "%", w) {
                apply(app, ui.ctx(), json!({"verticalScale": v}));
            }
            if let Some(v) = num_field(ui, "↔T", "Horizontal scale", c.horizontal_scale * 100.0, 0.0..=1000.0, "%", w) {
                apply(app, ui.ctx(), json!({"horizontalScale": v}));
            }
        });
        row(ui, &mut |ui| {
            if let Some(v) = num_field(ui, "Aª", "Baseline shift", c.baseline_shift_pt, -1296.0..=1296.0, "pt", w) {
                apply(app, ui.ctx(), json!({"baselineShift": v}));
            }
            // Colour: label + a swatch filling the rest of the second column.
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = LABEL_GAP;
                let g = ui.painter().layout_no_wrap(tl!("Color").into(), egui::FontId::proportional(12.0), t.text_dim);
                let lw = g.size().x.max(LABEL_W);
                let (lr, _) = ui.allocate_exact_size(egui::vec2(lw, 22.0), egui::Sense::hover());
                ui.painter().galley(egui::pos2(lr.left(), lr.center().y - g.size().y / 2.0), g, t.text_dim);
                let rgb = c.color.to_rgb();
                let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                let sw = (LABEL_W + LABEL_GAP + w - lw - LABEL_GAP).max(24.0);
                let (rect, resp) = ui.allocate_exact_size(egui::vec2(sw, 22.0), egui::Sense::click());
                ui.painter().rect_filled(rect, t.radius_sm, Color32::from_rgb(q(rgb[0]), q(rgb[1]), q(rgb[2])));
                ui.painter().rect_stroke(rect, t.radius_sm, Stroke::new(1.0, t.field_border), egui::StrokeKind::Inside);
                let resp = resp.on_hover_text(tl!("Text color"));
                resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, tl!("Text color")));
                if resp.clicked() {
                    open_color_picker(app, rgb);
                }
            });
        });
        // Faux styles: T (bold)  T (italic)  TT  Tᴛ  T̲  T̶, sharing the row evenly.
        let caps = c.caps;
        // Tip strings are English keys; `toggle_cell` runs them through `tl!`.
        let toggles: [(&str, &str, bool, serde_json::Value); 6] = [
            ("T", "Faux Bold", c.faux_bold, json!({"fauxBold": !c.faux_bold})),
            ("T", "Faux Italic", c.faux_italic, json!({"fauxItalic": !c.faux_italic})),
            (
                "TT",
                "All Caps",
                caps == photocraft_doc::text::Caps::AllCaps,
                json!({"caps": if caps == photocraft_doc::text::Caps::AllCaps { "normal" } else { "allCaps" }}),
            ),
            (
                "Tᴛ",
                "Small Caps",
                caps == photocraft_doc::text::Caps::SmallCaps,
                json!({"caps": if caps == photocraft_doc::text::Caps::SmallCaps { "normal" } else { "smallCaps" }}),
            ),
            ("T", "Underline", c.underline, json!({"underline": !c.underline})),
            ("T", "Strikethrough", c.strikethrough, json!({"strikethrough": !c.strikethrough})),
        ];
        let cw = cell_width(full, toggles.len());
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            for (i, (glyph, tip, on, props)) in toggles.into_iter().enumerate() {
                let clicked = toggle_cell(ui, cw, on, tip, |ui, r, col| {
                    let font = if i == 0 { crate::theme::semibold(13.0) } else { egui::FontId::proportional(13.0) };
                    let g = ui.painter().layout_no_wrap(glyph.to_string(), font, col);
                    let pos = r.center() - g.size() / 2.0;
                    let gr = egui::Rect::from_min_size(pos, g.size());
                    if i == 1 {
                        // Faux italic: a slanted T drawn as strokes (no italic face is bundled).
                        let (top, bot, cx) = (gr.top() + 3.0, gr.bottom() - 3.0, gr.center().x);
                        let slant = (bot - top) * 0.25;
                        ui.painter().line_segment([egui::pos2(cx - 4.0 + slant / 2.0, top), egui::pos2(cx + 4.0 + slant / 2.0, top)], Stroke::new(1.3, col));
                        ui.painter().line_segment([egui::pos2(cx + slant / 2.0, top), egui::pos2(cx - slant / 2.0, bot)], Stroke::new(1.3, col));
                    } else {
                        ui.painter().galley(pos, g, col);
                    }
                    if i == 4 {
                        ui.painter().line_segment([egui::pos2(gr.left(), gr.bottom() - 2.0), egui::pos2(gr.right(), gr.bottom() - 2.0)], Stroke::new(1.0, col));
                    }
                    if i == 5 {
                        ui.painter()
                            .line_segment([egui::pos2(gr.left() - 1.0, gr.center().y), egui::pos2(gr.right() + 1.0, gr.center().y)], Stroke::new(1.0, col));
                    }
                });
                if clicked {
                    apply(app, ui.ctx(), props);
                }
            }
        });
        ui.add_space(ROW_GAP);
        opentype_row(app, ui, &c, full);
        ui.add_space(ROW_GAP);
    }
    if paragraph && section(ui, "paragraph", tl!("Paragraph")) {
        use photocraft_doc::text::TextAlign as A;
        let full = ui.available_width();
        let w = field_width(full, 2, LABEL_W);
        let items = [
            ("left", tl!("Left align text"), A::Left),
            ("center", tl!("Center text"), A::Center),
            ("right", tl!("Right align text"), A::Right),
            ("justifyLeft", tl!("Justify last left"), A::JustifyLeft),
            ("justifyCenter", tl!("Justify last centered"), A::JustifyCenter),
            ("justifyRight", tl!("Justify last right"), A::JustifyRight),
            ("justifyAll", tl!("Justify all"), A::JustifyAll),
        ];
        let cw = cell_width(full, items.len());
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            for (key, tip, a) in items {
                let clicked = toggle_cell(ui, cw, para.align == a, tip, |ui, r, col| {
                    // The glyph keeps its size; the cell grows around it.
                    let g = egui::Rect::from_center_size(r.center(), egui::vec2(12.0, 10.0));
                    align_glyph(ui.painter(), g, a, if para.align == a { col } else { t.icon });
                });
                if clicked {
                    apply(app, ui.ctx(), json!({"align": key}));
                }
            }
        });
        ui.add_space(ROW_GAP);
        row(ui, &mut |ui| {
            if let Some(v) = num_field(ui, "→|", "Indent left margin", para.start_indent_pt, -1296.0..=1296.0, "pt", w) {
                apply(app, ui.ctx(), json!({"startIndent": v}));
            }
            if let Some(v) = num_field(ui, "|←", "Indent right margin", para.end_indent_pt, -1296.0..=1296.0, "pt", w) {
                apply(app, ui.ctx(), json!({"endIndent": v}));
            }
        });
        row(ui, &mut |ui| {
            if let Some(v) = num_field(ui, "¶→", "Indent first line", para.first_line_indent_pt, -1296.0..=1296.0, "pt", w) {
                apply(app, ui.ctx(), json!({"firstLineIndent": v}));
            }
        });
        row(ui, &mut |ui| {
            if let Some(v) = num_field(ui, "↑¶", "Add space before paragraph", para.space_before_pt, 0.0..=1296.0, "pt", w) {
                apply(app, ui.ctx(), json!({"spaceBefore": v}));
            }
            if let Some(v) = num_field(ui, "¶↓", "Add space after paragraph", para.space_after_pt, 0.0..=1296.0, "pt", w) {
                apply(app, ui.ctx(), json!({"spaceAfter": v}));
            }
        });
        let mut hy = para.hyphenate;
        if crate::widgets::checkbox(ui, &mut hy, "Hyphenate").changed() {
            apply(app, ui.ctx(), json!({"hyphenate": hy}));
        }
        ui.add_space(ROW_GAP);
    }
}

/// Type Options: anti-aliasing and orientation of the whole layer (the Type menu's commands).
fn type_options(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    use crate::props_layout::{COL_GAP, LABEL_GAP};
    let Some((layer, _)) = target(app) else { return };
    let Some((aa, vertical)) = app
        .session
        .active()
        .and_then(|st| text_layer(&st.doc, LayerId(layer)))
        .map(|t| (photocraft_engine::type_extra_cmds::aa_name(t.antialias).to_string(), t.orientation == photocraft_doc::text::Orientation::Vertical))
    else {
        return;
    };
    if !crate::props_layout::section(ui, "typeOptions", "Type Options") {
        return;
    }
    let full = ui.available_width();
    let mut run: Option<String> = None;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = COL_GAP;
        // Anti-aliasing dropdown (aₐ glyph label), then the two orientation toggles.
        let orient_w = 2.0 * 28.0 + 2.0;
        let dd_w = (full - crate::props_layout::LABEL_W - LABEL_GAP - COL_GAP - orient_w).max(60.0);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = LABEL_GAP;
            let t = crate::theme::Tokens::get(ui.ctx());
            let (r, resp) = ui.allocate_exact_size(egui::vec2(crate::props_layout::LABEL_W, 22.0), egui::Sense::hover());
            ui.painter().text(r.center(), egui::Align2::CENTER_CENTER, "aa", crate::theme::semibold(11.0), t.text_dim);
            resp.on_hover_text(tl!("Anti-aliasing method"));
            let mut cur = aa.clone();
            let opts: Vec<(String, &str)> = ["none", "sharp", "crisp", "strong", "smooth", "windowsLcd", "windows"]
                .into_iter()
                .filter_map(|k| photocraft_engine::commands::find(&format!("type.antiAlias.{k}")).map(|c| (k.to_string(), c.label)))
                .collect();
            if crate::widgets::dropdown(ui, "props-type-aa", &mut cur, &opts, dd_w) {
                run = Some(format!("type.antiAlias.{cur}"));
            }
        });
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            for (key, tip, on) in [("horizontal", "Horizontal type", !vertical), ("vertical", "Vertical type", vertical)] {
                let clicked = toggle_cell(ui, 28.0, on, tip, |ui, r, col| {
                    let g = egui::Rect::from_center_size(r.center(), egui::vec2(12.0, 12.0));
                    let s = Stroke::new(1.3, col);
                    if key == "horizontal" {
                        ui.painter().line_segment([g.left_top(), g.right_top()], s);
                        ui.painter().line_segment([g.center_top(), g.center_bottom()], s);
                    } else {
                        ui.painter().line_segment([g.left_top(), g.left_bottom()], s);
                        ui.painter().line_segment([g.left_center(), g.right_center()], s);
                    }
                });
                if clicked && !on {
                    run = Some(format!("type.orientation.{key}"));
                }
            }
        });
    });
    ui.add_space(crate::theme::ROW_GAP);
    if let Some(id) = run
        && let Some(params) = formatting_params(app)
        && let Err(e) = app.run(&id, params)
    {
        app.ui.status = e;
        app.ui.status_error = true;
    }
}

/// Cancel the editing session: undo it back to where it started (removes a new layer).
pub fn cancel(app: &mut PhotocraftApp) {
    crate::type_transform::reset(app);
    let Some(ed) = app.ui.text_edit.take() else { return };
    undo_session(app, &ed);
}

/// Undo the session's history step, if it is still the latest.
fn undo_session(app: &mut PhotocraftApp, ed: &TextEdit) {
    let coalesced = app.session.active().is_some_and(|s| s.coalesce.as_deref() == Some(ed.session.as_str()));
    if coalesced {
        let _ = app.run("edit.undo", json!({}));
    }
}

#[cfg(test)]
#[path = "type_tool_tests.rs"]
mod canvas_tests;

#[cfg(test)]
#[path = "type_font_picker_tests.rs"]
mod font_picker_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn app() -> PhotocraftApp {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.execute("file.new", json!({"width": 400, "height": 200})).unwrap();
        app.sync_views();
        app.ui.tool = crate::state::Tool::Type;
        app
    }

    fn layer_text(app: &PhotocraftApp) -> String {
        let ed = app.ui.text_edit.as_ref().unwrap();
        current_text(app, LayerId(ed.layer)).unwrap()
    }

    /// A committed type layer reading `s` at the same spot as the others (stacked variants, #4320).
    fn stacked_variant(app: &mut PhotocraftApp, s: &str) -> LayerId {
        app.ui.text_edit = None;
        pointer_up(app, [50.0, 100.0], [50.0, 100.0]);
        insert(app, s);
        commit(app);
        app.session.active().unwrap().active_layer.unwrap()
    }

    fn hide(app: &mut PhotocraftApp, id: LayerId) {
        app.run("layer.setProps", json!({"layer": id.0, "visible": false})).unwrap();
    }

    /// Wraps `id` in a new group and returns the group.
    fn group_of(app: &mut PhotocraftApp, id: LayerId) -> LayerId {
        app.run("layer.select", json!({"layer": id.0, "mode": "replace"})).unwrap();
        app.run("layer.groupLayers", json!({})).unwrap();
        let doc = &app.session.active().unwrap().doc;
        let path = doc.path_of(id).unwrap();
        doc.layer_at(&path[..path.len() - 1]).unwrap().id
    }

    /// Three type variants at one spot, bottom to top: A, B, C.
    fn variants() -> (PhotocraftApp, [LayerId; 3]) {
        let mut app = app();
        let ids = ["Variant A", "Variant B", "Variant C"].map(|s| stacked_variant(&mut app, s));
        (app, ids)
    }

    const ON_TEXT: (f64, f64) = (60.0, 95.0);

    /// #4320: variants hidden through their groups above the shown one are not picked.
    #[test]
    fn type_click_skips_text_in_hidden_groups_above_the_shown_variant() {
        let (mut app, [a, b, c]) = variants();
        for id in [b, c] {
            let g = group_of(&mut app, id);
            hide(&mut app, g);
        }
        assert_eq!(hit_layer(&mut app, ON_TEXT.0, ON_TEXT.1), Some(a));
        assert!(pointer_down(&mut app, ON_TEXT.0, ON_TEXT.1, egui::Modifiers::NONE));
        assert_eq!(app.ui.text_edit.as_ref().map(|e| e.layer), Some(a.0), "the click edits the shown variant");
    }

    #[test]
    fn type_click_skips_text_in_a_nested_hidden_group() {
        let (mut app, [a, b, c]) = variants();
        hide(&mut app, b);
        let inner = group_of(&mut app, c);
        let outer = group_of(&mut app, inner);
        hide(&mut app, outer);
        assert!(app.session.active().unwrap().doc.layer(c).unwrap().visible, "only the outer group is hidden");
        assert_eq!(hit_layer(&mut app, ON_TEXT.0, ON_TEXT.1), Some(a));
    }

    #[test]
    fn type_click_skips_own_hidden_variants_above_and_below() {
        let (mut app, [a, b, c]) = variants();
        hide(&mut app, b);
        hide(&mut app, c);
        assert_eq!(hit_layer(&mut app, ON_TEXT.0, ON_TEXT.1), Some(a), "hidden variants above");
        let (mut app, [a, b, c]) = variants();
        hide(&mut app, a);
        hide(&mut app, b);
        assert_eq!(hit_layer(&mut app, ON_TEXT.0, ON_TEXT.1), Some(c), "hidden variants below");
    }

    /// The Move tool's double-click on type (#1132) shares the pick.
    #[test]
    fn move_double_click_edits_the_shown_variant_not_one_in_a_hidden_group() {
        let (mut app, [a, b, c]) = variants();
        for id in [b, c] {
            let g = group_of(&mut app, id);
            hide(&mut app, g);
        }
        app.ui.tool = crate::state::Tool::Move;
        assert!(edit_at(&mut app, ON_TEXT.0, ON_TEXT.1));
        assert_eq!(app.ui.text_edit.as_ref().map(|e| e.layer), Some(a.0));
    }

    /// Photoshop: a click on existing type edits it; a ⇧-click there starts a new type layer.
    #[test]
    fn shift_click_on_existing_type_starts_new_text() {
        let mut app = app();
        pointer_up(&mut app, [50.0, 100.0], [50.0, 100.0]);
        insert(&mut app, "Existing words");
        commit(&mut app);
        let existing = app.session.active().unwrap().doc.layers.last().unwrap().id;
        let (x, y) = (60.0, 95.0);
        assert_eq!(hit_layer(&mut app, x, y), Some(existing), "the click lands on the existing text");
        assert!(!pointer_down(&mut app, x, y, egui::Modifiers::SHIFT), "⇧: not consumed, so a new type layer starts");
        assert!(app.ui.text_edit.is_none());
        assert!(pointer_down(&mut app, x, y, egui::Modifiers::NONE), "a plain click edits the existing text");
        assert_eq!(app.ui.text_edit.as_ref().map(|e| e.layer), Some(existing.0));
    }

    #[test]
    fn click_creates_placeholder_selected_and_typing_replaces_it() {
        let mut app = app();
        assert!(!pointer_down(&mut app, 50.0, 100.0, egui::Modifiers::NONE));
        pointer_up(&mut app, [50.0, 100.0], [50.0, 100.0]);
        let ed = app.ui.text_edit.clone().unwrap();
        assert_eq!((ed.anchor, ed.caret), (0, PLACEHOLDER.chars().count()));
        insert(&mut app, "Héllo");
        insert(&mut app, " world");
        assert_eq!(layer_text(&app), "Héllo world");
        // Creating and typing share the session key: one step after the initial snapshot.
        let steps = app.session.active().unwrap().history.entries();
        assert_eq!(steps, ["Open", "New Type Layer"]);
        commit(&mut app);
        let doc = &app.session.active().unwrap().doc;
        assert_eq!(doc.layers.last().unwrap().name, "Héllo world");
        assert!(app.ui.text_edit.is_none());
    }

    /// Preferences ▸ Units & Rulers ▸ Type switches the display unit of type metrics: points by
    /// default, pixels by the document's resolution, or millimetres.
    #[test]
    fn type_unit_follows_the_units_and_rulers_preference() {
        let mut app = app();
        assert_eq!(type_unit(&app), (1.0, "pt"));
        app.run("prefs.set", json!({"path": "unitsAndRulers.typeUnits", "value": "pixels"})).unwrap();
        assert_eq!(type_unit(&app), (1.0, "px"), "a 72 ppi document: 1 pt is 1 px");
        app.run("image.imageSize", json!({"resolution": 144, "resample": "none"})).unwrap();
        assert_eq!(type_unit(&app), (2.0, "px"), "144 ppi: a point is two pixels");
        app.run("prefs.set", json!({"path": "unitsAndRulers.typeUnits", "value": "mm"})).unwrap();
        assert!((type_unit(&app).0 - 25.4 / 72.0).abs() < 1e-6, "millimetres ignore the resolution");
    }

    /// Preferences ▸ Type ▸ Fill new type layers with placeholder text: off, a click creates an
    /// empty layer (nothing selected) and committing without typing removes it again.
    #[test]
    fn placeholder_preference_controls_new_type_layer_text() {
        let mut app = app();
        app.run("prefs.set", json!({"path": "type.fillNewTypeLayersWithPlaceholder", "value": false})).unwrap();
        pointer_up(&mut app, [50.0, 100.0], [50.0, 100.0]);
        let ed = app.ui.text_edit.clone().unwrap();
        assert_eq!((ed.anchor, ed.caret), (0, 0), "nothing to select");
        assert_eq!(layer_text(&app), "");
        assert_eq!(app.session.active().unwrap().doc.layers.len(), 2, "the layer was created");
        commit(&mut app);
        assert_eq!(app.session.active().unwrap().doc.layers.len(), 1, "an empty type layer is removed on commit");
        assert!(app.ui.text_edit.is_none());
    }

    /// Pure smart-quote mapping: opening after whitespace, a bracket or a dash, closing next to a
    /// letter or another quote.
    #[test]
    fn smart_quotes_are_typographic() {
        assert_eq!(smart_quotes("\"Hello\"", None), "\u{201c}Hello\u{201d}");
        assert_eq!(smart_quotes("don't", None), "don\u{2019}t");
        assert_eq!(smart_quotes("(\"x\")", None), "(\u{201c}x\u{201d})");
        assert_eq!(smart_quotes("--'x'", None), "--\u{2018}x\u{2019}", "a dash stays a dash, the quote opens after it");
        assert_eq!(smart_quotes("\"\"", None), "\u{201c}\u{201d}");
    }

    /// Preferences ▸ Type ▸ Use Smart Quotes switches typed quotes (typed text only, not paste).
    #[test]
    fn smart_quotes_preference_controls_typed_quotes() {
        for (smart, expected) in [(true, "hi \u{201c}you\u{201d}"), (false, "hi \"you\"")] {
            let mut app = app();
            app.run("prefs.set", json!({"path": "type.smartQuotes", "value": smart})).unwrap();
            pointer_up(&mut app, [50.0, 100.0], [50.0, 100.0]);
            insert_typed(&mut app, "hi \"you\"");
            assert_eq!(layer_text(&app), expected, "smartQuotes={smart}");
        }
    }

    #[test]
    fn a_name_from_text_after_blank_lines_keeps_following_the_text() {
        let mut app = app();
        pointer_up(&mut app, [50.0, 100.0], [50.0, 100.0]);
        insert(&mut app, "\n\nTitle");
        let id = app.ui.text_edit.as_ref().unwrap().layer;
        commit(&mut app);
        let name = |app: &PhotocraftApp| app.session.active().unwrap().doc.layers.last().unwrap().name.clone();
        assert_eq!(name(&app), "Title");
        // Edited later, outside the session that created it (#483).
        app.run("type.edit", json!({"layer": id, "text": "\nSubtitle"})).unwrap();
        assert_eq!(name(&app), "Subtitle");
    }

    #[test]
    fn dragging_a_box_handle_resizes_the_paragraph_box() {
        let mut app = app();
        pointer_up(&mut app, [10.0, 10.0], [110.0, 60.0]);
        let id = LayerId(app.ui.text_edit.as_ref().unwrap().layer);
        let steps = app.session.active().unwrap().history.entries().len();
        // Top-left corner: the box keeps its bottom-right corner.
        assert!(pointer_down(&mut app, 10.0, 10.0, egui::Modifiers::NONE));
        assert_eq!(app.ui.text_edit.as_ref().unwrap().resize, Some(0));
        pointer_move(&mut app, 20.0, 25.0);
        pointer_move(&mut app, 30.0, 30.0);
        pointer_up(&mut app, [10.0, 10.0], [30.0, 30.0]);
        assert_eq!(box_shape(&app, id), Some((0.0, 0.0, 80.0, 30.0)));
        let aff = layout(&mut app, id).unwrap().1;
        assert_eq!((aff.m[4], aff.m[5]), (30.0, 30.0));
        // The whole drag is one history step, and the edit session survives it.
        assert_eq!(app.session.active().unwrap().history.entries().len(), steps);
        assert!(app.ui.text_edit.is_some());
        // Right edge: only the width changes, and never below the minimum.
        assert!(pointer_down(&mut app, 110.0, 45.0, egui::Modifiers::NONE));
        pointer_move(&mut app, 0.0, 99.0);
        pointer_up(&mut app, [110.0, 45.0], [0.0, 99.0]);
        assert_eq!(box_shape(&app, id), Some((0.0, 0.0, MIN_BOX, 30.0)));
    }

    #[test]
    fn ime_preedit_is_replaced_by_each_update_and_commit_finalises() {
        let mut app = app();
        pointer_up(&mut app, [50.0, 100.0], [50.0, 100.0]);
        // Each preedit update replaces the previous one (and the placeholder selection first).
        ime_update(&mut app, "に", false);
        assert_eq!(layer_text(&app), "に");
        ime_update(&mut app, "にほん", false);
        ime_update(&mut app, "にほんご", false);
        assert_eq!(layer_text(&app), "にほんご");
        assert_eq!(app.ui.text_edit.as_ref().unwrap().preedit, Some((0, 4)));
        ime_update(&mut app, "日本語", true);
        assert_eq!(layer_text(&app), "日本語");
        let ed = app.ui.text_edit.clone().unwrap();
        assert_eq!((ed.caret, ed.anchor, ed.preedit), (3, 3, None));
        // The next composition starts after the committed text; cancelling removes it again.
        ime_update(&mut app, "あ", false);
        assert_eq!(layer_text(&app), "日本語あ");
        ime_update(&mut app, "", false);
        assert_eq!(layer_text(&app), "日本語");
        assert_eq!(app.ui.text_edit.as_ref().unwrap().preedit, None);
        // A stale preedit range (text shortened by undo) must not panic.
        app.ui.text_edit.as_mut().unwrap().preedit = Some((10, 5));
        ime_update(&mut app, "x", true);
        assert_eq!(layer_text(&app), "日本語x");
    }

    #[test]
    fn empty_new_layer_is_deleted_on_commit() {
        let mut app = app();
        pointer_up(&mut app, [10.0, 50.0], [10.0, 50.0]);
        let n = app.session.active().unwrap().doc.layers.len();
        insert(&mut app, "");
        if let Some(e) = app.ui.text_edit.as_mut() {
            e.anchor = 0;
        }
        insert(&mut app, "");
        commit(&mut app);
        assert_eq!(app.session.active().unwrap().doc.layers.len(), n - 1);
    }

    #[test]
    fn word_boundaries() {
        assert_eq!(word_boundary("hello big world", 0, true), 5);
        assert_eq!(word_boundary("hello big world", 7, false), 6);
        assert_eq!(word_boundary("hello big world", 15, false), 10);
    }
}
