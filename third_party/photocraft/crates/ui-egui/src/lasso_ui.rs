//! One lasso outline can alternate between freehand strokes and Alt-held straight segments:
//! with Alt held, each click adds a straight segment and a drag draws freehand. Releasing Alt
//! while the button is up closes the outline and makes the selection.
//! Selection intent is latched at the first press: Alt held then subtracts from an existing
//! selection; with no selection there is nothing to subtract, so it starts straight segments.
//! Alt pressed later changes drawing mode.
//!
//! The Freeform Pen draws the same way (Photoshop): a drag is freehand, Alt-clicks add straight
//! segments, releasing Alt with the button down goes on freehand and with it up finishes the
//! path (`vector_ui::freeform_finish`, which keeps the straight segments straight).

use crate::{
    PhotocraftApp,
    canvas::{Drag, ToolEvent, ViewXform},
    state::Tool,
};
use egui::{Event, Modifiers, PointerButton};

#[derive(Clone, Debug)]
pub struct Lasso {
    pub cursor: [f64; 2],
    down: bool,
    polygonal: bool,
    document: Option<photocraft_doc::DocId>,
    /// Indices of the points an Alt-click added: the segment into each one is straight.
    straight: Vec<usize>,
}

impl Lasso {
    /// Indices into the drag's points whose incoming segment is straight (an Alt-click).
    pub fn straight(&self) -> &[usize] {
        &self.straight
    }
}

/// The tools this module draws with: the Lasso and the Freeform Pen.
pub fn owns(tool: Tool) -> bool {
    matches!(tool, Tool::Lasso | Tool::FreeformPen)
}

pub fn active(app: &PhotocraftApp) -> bool {
    app.drag.as_ref().is_some_and(|d| owns(d.tool) && d.lasso.is_some())
}

/// Between polygonal clicks, Backspace/right-click retracts the last fixed vertex.
/// Do not affect a freehand stroke whose primary button is still held.
pub fn waiting_for_vertex(app: &PhotocraftApp) -> bool {
    owns(app.ui.tool)
        && app
            .drag
            .as_ref()
            .is_some_and(|d| d.tool == app.ui.tool && d.lasso.as_ref().is_some_and(|l| !l.down && l.document == app.session.active().map(|st| st.doc.id)))
}

pub fn undo_last_vertex(app: &mut PhotocraftApp) -> bool {
    cancel_stale(app);
    if !waiting_for_vertex(app) {
        return false;
    }
    if app.drag.as_ref().is_some_and(|d| d.points.len() <= 1) {
        app.drag = None;
    } else if let Some(d) = app.drag.as_mut() {
        d.points.pop();
        let n = d.points.len();
        if let Some(lasso) = &mut d.lasso {
            lasso.straight.retain(|&i| i < n);
        }
    }
    true
}

/// A lasso drag that moves the selection (started inside it, `canvas::selection_drag_kind`):
/// the canvas's own selection-move path handles it, this module only feeds it the events.
fn moving_selection(app: &PhotocraftApp) -> bool {
    app.drag.as_ref().is_some_and(|d| d.tool == Tool::Lasso && d.sel_move.is_some())
}

/// A drag the canvas handles rather than this module, which only feeds it the events: moving the
/// selection, a guide that ⌘ took (`canvas::command_guide_at`, #2690), the layer that ⌘ moves
/// (`canvas::command_moves_layer`, refused on a locked layer: `move_blocked`, #2725), or a path
/// that the Freeform Pen's ⌘ (Direct Selection) edits.
fn handed_over(app: &PhotocraftApp) -> bool {
    moving_selection(app)
        || app.guide_drag.is_some_and(|d| d.index.is_some())
        || app.drag.as_ref().is_some_and(|d| d.tool == Tool::Move)
        || app.move_blocked
        || app.ui.direct_selection.drag.is_some()
}

pub fn cancel_stale(app: &mut PhotocraftApp) {
    let document = app.session.active().map(|st| st.doc.id);
    if app.drag.as_ref().is_some_and(|d| d.lasso.as_ref().is_some_and(|lasso| app.ui.tool != d.tool || lasso.document != document)) {
        app.drag = None;
    }
}

fn append(d: &mut Drag, p: [f64; 2]) {
    if d.points.last().is_none_or(|last| last[0] != p[0] || last[1] != p[1]) {
        d.points.push([p[0], p[1], 1.0]);
    }
}

fn modifiers(app: &mut PhotocraftApp, mods: Modifiers) {
    let Some(d) = app.drag.as_mut().filter(|d| owns(d.tool)) else { return };
    d.track(mods);
    // Alt held before the initial press still means subtract, until released and pressed again.
    let polygonal = mods.alt && (!d.modifiers.alt || d.released.alt);
    // Alt released between straight segments (button up): the outline is done.
    let close = d.lasso.as_ref().is_some_and(|l| l.polygonal && !polygonal && !l.down);
    let anchor = d.lasso.as_ref().and_then(|l| (l.polygonal && !polygonal && l.down).then_some(l.cursor));
    if let Some(p) = anchor {
        append(d, p);
    }
    if let Some(lasso) = &mut d.lasso {
        lasso.polygonal = polygonal;
    }
    if close {
        commit(app);
    }
}

pub fn commit(app: &mut PhotocraftApp) {
    cancel_stale(app);
    if active(app)
        && let Some(d) = app.drag.take()
    {
        crate::canvas::finish_gesture(app, d);
    }
}

pub fn pointer(app: &mut PhotocraftApp, ev: ToolEvent, mods: Modifiers) -> bool {
    cancel_stale(app);
    let tool = app.ui.tool;
    if !owns(tool) || handed_over(app) {
        return false;
    }
    let p = match ev {
        ToolEvent::Down { x, y, .. } | ToolEvent::Move { x, y, .. } | ToolEvent::Up { x, y } => [x, y],
    };
    // A press inside the selection (no outline in progress) drags the selection instead; a ⌘
    // press outside it moves the layer, as the Move tool (#2725).
    if matches!(ev, ToolEvent::Down { .. })
        && tool == Tool::Lasso
        && !active(app)
        && p.iter().all(|n| n.is_finite())
        && (crate::canvas::selection_drag_kind(app, Tool::Lasso, p, mods).is_some() || crate::canvas::command_moves_layer(app, Tool::Lasso, p, mods))
    {
        return false;
    }
    if !p.iter().all(|n| n.is_finite()) {
        return true;
    }
    modifiers(app, mods);
    match ev {
        ToolEvent::Down { .. } => {
            if active(app) {
                // A click on the first point, or on the last one again, closes the outline.
                let reach = 8.0 / app.point_zoom().max(0.01) as f64;
                let near = |q: [f64; 2]| (q[0] - p[0]).hypot(q[1] - p[1]) < reach;
                let close = app.drag.as_ref().is_some_and(|d| d.points.len() >= 3 && (near(d.start) || d.points.last().is_some_and(|l| near([l[0], l[1]]))));
                if close {
                    // A path closes on its start point (a selection outline always closes).
                    if let Some(d) = app.drag.as_mut().filter(|d| d.tool == Tool::FreeformPen && near(d.start)) {
                        let (start, n) = (d.start, d.points.len());
                        append(d, start);
                        if let Some(lasso) = d.lasso.as_mut().filter(|_| d.points.len() > n) {
                            lasso.straight.push(n);
                        }
                    }
                    commit(app);
                    return true;
                }
                if let Some(d) = &mut app.drag {
                    let n = d.points.len();
                    append(d, p);
                    let added = d.points.len() > n;
                    if let Some(lasso) = &mut d.lasso {
                        lasso.cursor = p;
                        lasso.down = true;
                        if added {
                            lasso.straight.push(n);
                        }
                    }
                }
            } else {
                // Alt with no selection has nothing to subtract from: the outline starts with
                // straight segments, as a new selection.
                // A path has nothing to subtract from either.
                let nothing_selected = tool == Tool::FreeformPen || app.session.active().is_none_or(|st| st.doc.selection.is_none());
                let polygonal = mods.alt && nothing_selected;
                let intent = if polygonal { Modifiers { alt: false, ..mods } } else { mods };
                let mut d = Drag::new(tool, p, vec![[p[0], p[1], 1.0]], intent, false);
                let document = app.session.active().map(|st| st.doc.id);
                d.lasso = Some(Lasso { cursor: p, down: true, polygonal, document, straight: Vec::new() });
                app.drag = Some(d);
            }
        }
        ToolEvent::Move { .. } | ToolEvent::Up { .. } => {
            let up = matches!(ev, ToolEvent::Up { .. });
            let Some(d) = app.drag.as_mut().filter(|d| d.tool == tool) else { return true };
            let Some(lasso) = &d.lasso else { return true };
            let (down, polygonal, previous) = (lasso.down, lasso.polygonal, lasso.cursor);
            if d.reposition {
                // Move relative to the rubber-band endpoint, not the last fixed vertex.
                let last = d.points.last().map_or(d.start, |p| [p[0], p[1]]);
                d.shift_to([last[0] + p[0] - previous[0], last[1] + p[1] - previous[1]]);
            } else if down {
                // While the button is down the outline follows the pointer (freehand), Alt or
                // not; between Alt clicks only the rubber band moves.
                append(d, p);
            }
            if let Some(lasso) = &mut d.lasso {
                lasso.cursor = p;
                if up {
                    lasso.down = false;
                }
            }
            if up && down && !polygonal {
                commit(app);
            }
        }
    }
    true
}

/// Read actual presses/releases, including clicks below egui's drag threshold. Hover moves
/// update only the rubber band; they must never become freehand vertices between Alt clicks.
pub fn canvas_input(app: &mut PhotocraftApp, ctx: &egui::Context, xf: &ViewXform, response: &egui::Response) {
    let events = ctx.input(|i| i.events.clone());
    let mut mods = app.drag.as_ref().map_or(ctx.input(|i| i.modifiers), |d| d.live);
    for ev in events {
        let tool_ev = match ev {
            Event::ModifiersChanged(m) => {
                mods = m;
                modifiers(app, crate::workspace_ui::sticky_mods(app, m));
                None
            }
            Event::PointerButton { pos, button: PointerButton::Primary, pressed, modifiers: m } => {
                mods = m;
                let p = xf.to_doc(pos);
                if pressed && response.contains_pointer() && response.rect.contains(pos) {
                    Some(ToolEvent::Down { x: p[0], y: p[1], pressure: 1.0 })
                } else if !pressed && (active(app) || handed_over(app)) {
                    Some(ToolEvent::Up { x: p[0], y: p[1] })
                } else {
                    None
                }
            }
            Event::PointerMoved(pos) if active(app) || handed_over(app) => {
                let p = xf.to_doc(pos);
                Some(ToolEvent::Move { x: p[0], y: p[1], pressure: 1.0 })
            }
            Event::WindowFocused(false) => {
                if active(app) || moving_selection(app) {
                    app.drag = None;
                }
                None
            }
            _ => None,
        };
        if let Some(ev) = tool_ev {
            crate::canvas::tool_event(app, ev, mods);
        }
    }
    if response.double_clicked() {
        commit(app);
    }
}

#[cfg(test)]
mod tests;
