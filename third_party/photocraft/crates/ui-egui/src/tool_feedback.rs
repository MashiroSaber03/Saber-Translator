//! On-canvas tool feedback: the selection tools' intent badge (#170) and outlines that stay
//! visible on any image (#172).
//!
//! Photoshop marks the selection-tool cursor with a small `+` (add), `−` (subtract) or `×`
//! (intersect) reflecting the effective mode: the options bar's mode, overridden by the held
//! modifiers. Selection outlines and drag previews are drawn as black-and-white dashes (marching
//! ants) so they read on light and dark pixels alike.

use egui::{Color32, Pos2, Shape, Stroke, pos2, vec2};

use crate::PhotocraftApp;
use crate::state::Tool;

/// The intent a selection tool's next gesture has, shown as a cursor badge (none for New).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Badge {
    Add,
    Subtract,
    Intersect,
}

impl Badge {
    /// The badge for an engine selection mode name (`replace` has none).
    pub fn from_mode(mode: &str) -> Option<Badge> {
        match mode {
            "add" => Some(Badge::Add),
            "subtract" => Some(Badge::Subtract),
            "intersect" => Some(Badge::Intersect),
            _ => None,
        }
    }
}

/// Tools whose cursor carries the selection-mode badge.
pub fn is_selection_tool(tool: Tool) -> bool {
    matches!(
        tool,
        Tool::RectMarquee
            | Tool::EllipseMarquee
            | Tool::SingleRowMarquee
            | Tool::SingleColumnMarquee
            | Tool::Lasso
            | Tool::PolygonLasso
            | Tool::MagneticLasso
            | Tool::MagicWand
            | Tool::QuickSelection
            | Tool::ObjectSelection
    )
}

/// The effective selection mode of `tool` for the options-bar mode `bar` (0 New, 1 Add,
/// 2 Subtract, 3 Intersect) and the held modifiers: exactly what the tool's gesture will use.
pub fn selection_mode(tool: Tool, bar: u8, m: egui::Modifiers) -> &'static str {
    match tool {
        // Quick Selection always adds (⌥ subtracts), like Photoshop's after the first stroke.
        Tool::QuickSelection => {
            if m.alt {
                "subtract"
            } else {
                "add"
            }
        }
        // Object Selection: ⌥ subtracts, ⇧ adds (see `retouch_ui::finish_object_selection`).
        Tool::ObjectSelection => {
            if m.alt {
                "subtract"
            } else if m.shift {
                "add"
            } else {
                "replace"
            }
        }
        _ => {
            if m.shift && m.alt {
                "intersect"
            } else if m.shift {
                "add"
            } else if m.alt {
                "subtract"
            } else {
                ["replace", "add", "subtract", "intersect"][bar.min(3) as usize]
            }
        }
    }
}

/// [`selection_mode`] for the active document: with nothing selected, ⌥ (or ⇧⌥) has nothing to
/// subtract from or intersect with, so it draws a new selection, as in Photoshop (where ⌥ then
/// only draws a marquee from its centre). An options-bar Subtract or Intersect is kept (#1106).
pub fn document_selection_mode(app: &PhotocraftApp, tool: Tool, m: egui::Modifiers) -> &'static str {
    let mode = selection_mode(tool, app.ui.selection_mode, m);
    let selected = app.session.active().and_then(|st| st.doc.selection.as_ref()).is_some_and(|s| !s.content_bounds().is_empty());
    if m.alt && !selected && matches!(mode, "subtract" | "intersect") && selection_mode(tool, app.ui.selection_mode, egui::Modifiers::NONE) == "replace" {
        "replace"
    } else {
        mode
    }
}

/// The badge the cursor of `tool` shows with modifiers `m` (None for non-selection tools or New;
/// the anchor point tools show theirs).
pub fn badge(app: &PhotocraftApp, tool: Tool, m: egui::Modifiers) -> Option<Badge> {
    // Add / Delete Anchor Point: + or − beside the crosshair, as on Photoshop's pen cursors (⌥
    // swaps them; ⌘ is Direct Selection, which shows none).
    if let Some(t) = crate::anchor_tools::effective(tool, m) {
        return (!m.command).then_some(if t == Tool::AddAnchorPoint { Badge::Add } else { Badge::Subtract });
    }
    // The Pen over the active path with Auto Add/Delete on: what a click would do (#4144).
    if let Some(p) = app.hover_doc.filter(|_| tool == Tool::Pen) {
        return crate::anchor_tools::pen_auto(app, tool, m, p).map(|(t, ..)| if t == Tool::AddAnchorPoint { Badge::Add } else { Badge::Subtract });
    }
    if !is_selection_tool(tool) {
        return None;
    }
    // ⌘ (⌘⌥) drags pixels or the layer instead of combining selections.
    if m.command && app.drag.is_none() {
        return None;
    }
    let mut m = app.drag.as_ref().filter(|d| d.tool == Tool::Lasso && tool == Tool::Lasso).map_or(m, |d| d.modifiers);
    // The Lasso's Alt with nothing selected draws straight segments; it subtracts nothing.
    if matches!(tool, Tool::Lasso | Tool::PolygonLasso) && app.drag.is_none() && app.session.active().is_none_or(|st| st.doc.selection.is_none()) {
        m.alt = false;
    }
    Badge::from_mode(selection_mode(tool, app.ui.selection_mode, m))
}

/// The badge's glyph as line segments around `c` (half-size `h`).
pub fn badge_glyph(b: Badge, c: Pos2, h: f32) -> Vec<[Pos2; 2]> {
    match b {
        Badge::Add => vec![[c - vec2(h, 0.0), c + vec2(h, 0.0)], [c - vec2(0.0, h), c + vec2(0.0, h)]],
        Badge::Subtract => vec![[c - vec2(h, 0.0), c + vec2(h, 0.0)]],
        Badge::Intersect => {
            let d = h * 0.8;
            vec![[c + vec2(-d, -d), c + vec2(d, d)], [c + vec2(-d, d), c + vec2(d, -d)]]
        }
    }
}

/// Draw the badge next to the cursor hotspot `p` (Photoshop puts it below-right of the
/// crosshair; Quick Selection shows it inside the brush circle, at the centre).
pub fn draw_badge(painter: &egui::Painter, p: Pos2, b: Badge, centred: bool) {
    // Right of the system arrow cursor: at 1× the arrow covers (11, 11), so the badge sat under it.
    let c = if centred { p } else { p + vec2(30.0, 11.0) };
    // White glyph with a dark halo: readable on any pixels.
    for (w, col) in [(3.0, Color32::from_black_alpha(200)), (1.2, Color32::WHITE)] {
        for seg in badge_glyph(b, c, 3.5) {
            painter.line_segment(seg, Stroke::new(w, col));
        }
    }
}

/// Dash length of the marching ants, in screen points.
pub const DASH: f32 = 4.0;

/// Marching ants along a polyline: a solid white line under black dashes, the dashes shifted by
/// `phase` (animate it with time). Visible on any image, like Photoshop's selection edges.
pub fn ants(points: &[Pos2], closed: bool, phase: f32) -> Vec<Shape> {
    let Some(&first) = points.first().filter(|_| points.len() >= 2) else { return Vec::new() };
    let mut pts = points.to_vec();
    if closed {
        pts.push(first);
    }
    let mut out = vec![Shape::line(pts.clone(), Stroke::new(1.0, Color32::WHITE))];
    let black = Stroke::new(1.0, Color32::BLACK);
    // Arc length so dashes run continuously across vertices.
    let mut along = -phase.rem_euclid(DASH * 2.0);
    for w in pts.windows(2) {
        let (a, b) = (w[0], w[1]);
        let len = a.distance(b);
        if len < 1e-3 {
            continue;
        }
        let dir = (b - a) / len;
        // Position of this segment's start within the dash period.
        let mut t = -(along.rem_euclid(DASH * 2.0));
        while t < len {
            let (s0, s1) = (t.max(0.0), (t + DASH).min(len));
            if s1 > s0 {
                out.push(Shape::line_segment([a + dir * s0, a + dir * s1], black));
            }
            t += DASH * 2.0;
        }
        along += len;
    }
    out
}

/// Marching-ants phase for the frame time `time` (s), and keep the animation running.
pub fn ants_phase(ctx: &egui::Context) -> f32 {
    let time = ctx.input(|i| i.time);
    ctx.request_repaint_after(std::time::Duration::from_millis(100));
    ((time * 10.0) % (DASH as f64 * 2.0)) as f32
}

/// Draw animated marching ants along `points`.
pub fn draw_ants(painter: &egui::Painter, points: &[Pos2], closed: bool) {
    let phase = ants_phase(painter.ctx());
    painter.extend(ants(points, closed, phase));
}

/// Polygon approximating the ellipse inscribed in `r` (enough vertices to look smooth).
pub fn ellipse_points(r: egui::Rect) -> Vec<Pos2> {
    let (c, a, b) = (r.center(), r.width() / 2.0, r.height() / 2.0);
    let n = ((a.abs() + b.abs()) * 0.5).clamp(16.0, 256.0) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / n as f32 * std::f32::consts::TAU;
            pos2(c.x + a * t.cos(), c.y + b * t.sin())
        })
        .collect()
}

/// A path preview in `color` with a contrasting dark halo, so a blue shape outline still shows
/// on blue (or any) pixels: the shape tools' drag preview.
pub fn contrast_path(points: Vec<Pos2>, closed: bool, color: Color32) -> Vec<Shape> {
    let make = |pts: Vec<Pos2>, s: Stroke| if closed { Shape::closed_line(pts, s) } else { Shape::line(pts, s) };
    vec![make(points.clone(), Stroke::new(3.0, Color32::from_black_alpha(150))), make(points, Stroke::new(1.0, color))]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strokes(shapes: &[Shape]) -> Vec<Color32> {
        shapes
            .iter()
            .filter_map(|s| match s {
                Shape::LineSegment { stroke, .. } => Some(stroke.color),
                Shape::Path(p) => match p.stroke.color {
                    egui::epaint::ColorMode::Solid(c) => Some(c),
                    _ => None,
                },
                _ => None,
            })
            .collect()
    }

    #[test]
    fn badge_follows_modifiers_and_the_options_bar() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        let none = egui::Modifiers::NONE;
        let (shift, alt) = (egui::Modifiers::SHIFT, egui::Modifiers::ALT);
        let both = egui::Modifiers { shift: true, alt: true, ..Default::default() };
        app.run("file.new", serde_json::json!({"width": 40, "height": 30})).unwrap();
        app.run("select.rect", serde_json::json!({"x": 5, "y": 5, "width": 10, "height": 10})).unwrap();
        for t in [Tool::RectMarquee, Tool::EllipseMarquee, Tool::Lasso, Tool::PolygonLasso, Tool::MagicWand] {
            assert_eq!(badge(&app, t, none), None, "{t:?}: New selection has no badge");
            assert_eq!(badge(&app, t, shift), Some(Badge::Add));
            assert_eq!(badge(&app, t, alt), Some(Badge::Subtract));
            assert_eq!(badge(&app, t, both), Some(Badge::Intersect));
        }
        // With nothing selected the Lasso's Alt draws straight segments: no − badge.
        app.run("select.deselect", serde_json::json!({})).unwrap();
        assert_eq!(badge(&app, Tool::Lasso, alt), None);
        assert_eq!(badge(&app, Tool::RectMarquee, alt), Some(Badge::Subtract));
        // The options bar's mode shows without any key held; keys still override it.
        app.ui.selection_mode = 2;
        assert_eq!(badge(&app, Tool::Lasso, none), Some(Badge::Subtract));
        assert_eq!(badge(&app, Tool::Lasso, shift), Some(Badge::Add));
        app.ui.selection_mode = 3;
        assert_eq!(badge(&app, Tool::RectMarquee, none), Some(Badge::Intersect));
        // Smart selection tools.
        assert_eq!(badge(&app, Tool::QuickSelection, none), Some(Badge::Add));
        assert_eq!(badge(&app, Tool::QuickSelection, alt), Some(Badge::Subtract));
        assert_eq!(badge(&app, Tool::ObjectSelection, none), None);
        assert_eq!(badge(&app, Tool::ObjectSelection, shift), Some(Badge::Add));
        // Other tools never show one.
        for t in [Tool::Brush, Tool::Move, Tool::Zoom, Tool::Crop] {
            assert_eq!(badge(&app, t, shift), None);
        }
        // Distinct glyphs: + has two strokes, − one, × two diagonals.
        let c = pos2(10.0, 10.0);
        assert_eq!(badge_glyph(Badge::Add, c, 3.0).len(), 2);
        assert_eq!(badge_glyph(Badge::Subtract, c, 3.0).len(), 1);
        assert!(badge_glyph(Badge::Intersect, c, 3.0).iter().all(|[a, b]| a.x != b.x && a.y != b.y));
    }

    #[test]
    fn ants_use_two_contrasting_colours_and_march() {
        let pts = [pos2(0.0, 0.0), pos2(40.0, 0.0), pos2(40.0, 30.0)];
        let a = ants(&pts, true, 0.0);
        let cols = strokes(&a);
        assert!(cols.contains(&Color32::WHITE) && cols.contains(&Color32::BLACK), "{cols:?}");
        // Roughly half the length is black dashes: (40 + 30 + 50) / 8 ≈ 15 dashes.
        let dashes = cols.iter().filter(|c| **c == Color32::BLACK).count();
        assert!((13..=18).contains(&dashes), "{dashes}");
        // The phase moves the dashes (the ants march).
        let first = |s: &[Shape]| {
            s.iter().find_map(|s| match s {
                Shape::LineSegment { points, stroke } if stroke.color == Color32::BLACK => Some(points[0]),
                _ => None,
            })
        };
        assert_ne!(first(&a), first(&ants(&pts, true, 2.0)));
        assert!(ants(&pts[..1], false, 0.0).is_empty());
        // Shape previews: the accent path over a dark halo.
        let s = contrast_path(pts.to_vec(), true, Color32::from_rgb(0, 120, 255));
        let cols = strokes(&s);
        assert_eq!(cols.len(), 2);
        assert!(cols[0].r() < 10 && cols[0].a() > 100 && cols[1] == Color32::from_rgb(0, 120, 255));
        // Ellipses close smoothly.
        let e = ellipse_points(egui::Rect::from_min_max(pos2(0.0, 0.0), pos2(100.0, 50.0)));
        assert!(e.len() >= 16 && e.iter().all(|p| p.x >= -0.01 && p.x <= 100.01 && p.y >= -0.01 && p.y <= 50.01));
    }
}
