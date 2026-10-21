//! Edit › Perspective Warp on the canvas.
//!
//! **Layout** mode: drag on the image to draw a plane, drag its corners onto the image's
//! perspective lines; a corner dropped near another plane's corner snaps to it, which links the
//! planes along that edge. **Warp** mode: drag corners (linked corners move together) and the
//! image follows; the options bar straightens near-vertical / near-horizontal edges. ↩ / Commit
//! runs `edit.perspectiveWarp` with the planes. The preview is the layer texture on a 40×40 mesh
//! pushed through the same piecewise-homography map the engine uses.

use std::sync::Arc;

use egui::{Color32, Pos2, Stroke, TextureHandle, pos2};
use photocraft_algo::perspective::{PerspectiveMap, Plane, Straighten, linked_corners, plane_point, straighten};
use photocraft_doc::{Document, LayerId};
use photocraft_geom::Rect;
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::canvas::{ToolEvent, ViewXform};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub enum PerspMode {
    Layout,
    Warp,
}

enum Drag {
    NewPlane {
        start: [f64; 2],
        cur: [f64; 2],
    },
    /// Linked corners being moved, and the pointer offset from the corner.
    Corner {
        group: Vec<(usize, usize)>,
        offset: [f64; 2],
    },
}

pub struct PerspSession {
    pub key: u64,
    pub layer: LayerId,
    command: String,
    bounds: Rect,
    pub planes: Vec<Plane>,
    pub mode: PerspMode,
    drag: Option<Drag>,
    texture: TextureHandle,
    opacity: f32,
    pub preview_doc: Arc<Document>,
    /// Preview mesh: (screen-independent destination points) for a 40×40 grid over `bounds`.
    grid: Vec<[f64; 2]>,
}

const GRID: usize = 40;

impl PerspSession {
    pub fn describe(&self) -> Value {
        json!({"layer": self.layer.0, "mode": self.mode, "planes": self.planes})
    }

    fn update_grid(&mut self) {
        let b = self.bounds;
        let map = if self.mode == PerspMode::Warp { PerspectiveMap::new(&self.planes) } else { None };
        self.grid.clear();
        for j in 0..=GRID {
            for i in 0..=GRID {
                let x = f64::from(b.x0) + f64::from(b.width()) * i as f64 / GRID as f64;
                let y = f64::from(b.y0) + f64::from(b.height()) * j as f64 / GRID as f64;
                let (u, v) = map.as_ref().map_or((x, y), |m| m.map(x, y));
                self.grid.push([u, v]);
            }
        }
    }

    fn corners(&self, p: usize) -> &[[f64; 2]; 4] {
        if self.mode == PerspMode::Layout { &self.planes[p].src } else { &self.planes[p].dst }
    }

    fn corner_at(&self, p: [f64; 2], tol: f64) -> Option<(usize, usize)> {
        let mut best = None;
        let mut bd = tol;
        for pi in 0..self.planes.len() {
            for (ci, c) in self.corners(pi).iter().enumerate() {
                let d = (c[0] - p[0]).hypot(c[1] - p[1]);
                if d <= bd {
                    bd = d;
                    best = Some((pi, ci));
                }
            }
        }
        best
    }

    fn snap(&self, p: [f64; 2], tol: f64, skip: &[(usize, usize)]) -> [f64; 2] {
        for (pi, pl) in self.planes.iter().enumerate() {
            for (ci, c) in pl.src.iter().enumerate() {
                if !skip.contains(&(pi, ci)) && (c[0] - p[0]).hypot(c[1] - p[1]) <= tol {
                    return *c;
                }
            }
        }
        p
    }

    fn set_corner(&mut self, group: &[(usize, usize)], p: [f64; 2]) {
        let layout = self.mode == PerspMode::Layout;
        for &(pi, ci) in group {
            let Some(pl) = self.planes.get_mut(pi) else { continue };
            if layout && let Some(c) = pl.src.get_mut(ci) {
                *c = p;
            }
            if let Some(c) = pl.dst.get_mut(ci) {
                *c = p;
            }
        }
    }
}

pub fn begin(app: &mut PhotocraftApp, ctx: &egui::Context, command: &str) -> Result<(), String> {
    if app.ui.transform.is_some() {
        return Err("finish Free Transform first".into());
    }
    let (layer, surface, _) = crate::distort_ui::active_pixels(app)?;
    let st = app.session.active().ok_or("no document")?;
    let l = st.doc.layer(layer).ok_or("no layer")?;
    let bounds = surface.content_bounds();
    if bounds.is_empty() {
        return Err("Could not use Perspective Warp: the layer is empty".into());
    }
    let opacity = l.opacity * l.fill_opacity;
    let preview_doc = crate::distort_ui::without_layer(&st.doc, layer);
    let key = app.ui.alloc_id();
    let texture = ctx.load_texture(format!("persp-{key}"), crate::distort_ui::surface_image(&surface, bounds, 2048), egui::TextureOptions::LINEAR);
    let mut s = PerspSession {
        key,
        layer,
        command: command.to_string(),
        bounds,
        planes: Vec::new(),
        mode: PerspMode::Layout,
        drag: None,
        texture,
        opacity,
        preview_doc,
        grid: Vec::new(),
    };
    s.update_grid();
    app.distort.perspective = Some(s);
    app.ui.status = tl!("Perspective Warp: draw planes along the image's perspective, then switch to Warp").into();
    Ok(())
}

pub fn commit(app: &mut PhotocraftApp) {
    let Some(s) = app.distort.perspective.take() else { return };
    if s.planes.is_empty() || PerspectiveMap::is_identity(&s.planes) {
        return;
    }
    let _ = app.run(&s.command, json!({"planes": s.planes, "layer": s.layer.0}));
}

fn set_mode(s: &mut PerspSession, m: PerspMode) {
    s.mode = m;
    if m == PerspMode::Warp {
        // Linked corners share one position from here on.
        photocraft_algo::perspective::unify(&mut s.planes);
    }
    s.update_grid();
}

/// Control channel: `edit.perspectiveWarp {"ui": {...}}` while the mode is active.
pub fn control(app: &mut PhotocraftApp, ui: &Value) -> Result<Value, String> {
    if ui.get("commit").and_then(Value::as_bool) == Some(true) {
        commit(app);
        return Ok(json!({"committed": true}));
    }
    if ui.get("cancel").and_then(Value::as_bool) == Some(true) {
        app.distort.perspective = None;
        return Ok(json!({"cancelled": true}));
    }
    let s = app.distort.perspective.as_mut().ok_or(tl!("Perspective Warp is not active"))?;
    if let Some(m) = ui.get("mode").and_then(Value::as_str) {
        set_mode(s, if m == "warp" { PerspMode::Warp } else { PerspMode::Layout });
    }
    if let Some(p) = ui.get("planes") {
        s.planes = serde_json::from_value(p.clone()).map_err(|e| format!("bad planes: {e}"))?;
        // A corner drag holds (plane, corner) indices into the old list (#952).
        s.drag = None;
        s.update_grid();
    }
    if let Some(st) = ui.get("straighten").and_then(Value::as_str) {
        let m = Straighten::parse(st).ok_or("straighten: horizontal|vertical|auto")?;
        straighten(&mut s.planes, m);
        s.update_grid();
    }
    Ok(s.describe())
}

pub fn pointer(app: &mut PhotocraftApp, ev: ToolEvent, _mods: egui::Modifiers) {
    let tol = crate::distort_ui::tolerance(app);
    let Some(s) = app.distort.perspective.as_mut() else { return };
    match ev {
        ToolEvent::Down { x, y, .. } => {
            let p = [x, y];
            if let Some((pi, ci)) = s.corner_at(p, tol) {
                let c = s.corners(pi)[ci];
                // Move every corner linked to this one.
                let group = linked_corners(&s.planes).into_iter().find(|g| g.contains(&(pi, ci))).unwrap_or_else(|| vec![(pi, ci)]);
                s.drag = Some(Drag::Corner { group, offset: [c[0] - x, c[1] - y] });
            } else if s.mode == PerspMode::Layout {
                let start = s.snap(p, tol, &[]);
                s.drag = Some(Drag::NewPlane { start, cur: p });
            }
        }
        ToolEvent::Move { x, y, .. } => match &mut s.drag {
            Some(Drag::NewPlane { cur, .. }) => *cur = [x, y],
            Some(Drag::Corner { group, offset }) => {
                let group = group.clone();
                let mut q = [x + offset[0], y + offset[1]];
                if s.mode == PerspMode::Layout {
                    q = s.snap(q, tol, &group);
                }
                s.set_corner(&group, q);
                if s.mode == PerspMode::Warp {
                    s.update_grid();
                }
            }
            None => {}
        },
        ToolEvent::Up { x, y } => {
            match s.drag.take() {
                Some(Drag::NewPlane { start, .. }) => {
                    let end = s.snap([x, y], tol, &[]);
                    let (x0, y0, x1, y1) = (start[0].min(end[0]), start[1].min(end[1]), start[0].max(end[0]), start[1].max(end[1]));
                    if x1 - x0 > 4.0 && y1 - y0 > 4.0 {
                        // Keep the snapped corners exact.
                        let mut q = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
                        for c in q.iter_mut() {
                            *c = s.snap(*c, tol, &[]);
                        }
                        s.planes.push(Plane::identity(q));
                    }
                }
                Some(Drag::Corner { group, offset }) => {
                    let mut q = [x + offset[0], y + offset[1]];
                    if s.mode == PerspMode::Layout {
                        q = s.snap(q, tol, &group);
                    }
                    s.set_corner(&group, q);
                }
                None => {}
            }
            s.update_grid();
        }
    }
}

pub fn draw(s: &PerspSession, painter: &egui::Painter, xf: &ViewXform) {
    let scr = |p: [f64; 2]| xf.to_screen(p[0] as f32, p[1] as f32);
    let mut mesh = egui::Mesh::with_texture(s.texture.id());
    let tint = Color32::from_white_alpha((s.opacity.clamp(0.0, 1.0) * 255.0) as u8);
    for j in 0..=GRID {
        for i in 0..=GRID {
            let p = s.grid[j * (GRID + 1) + i];
            mesh.vertices.push(egui::epaint::Vertex { pos: scr(p), uv: pos2(i as f32 / GRID as f32, j as f32 / GRID as f32), color: tint });
        }
    }
    for j in 0..GRID {
        for i in 0..GRID {
            let a = (j * (GRID + 1) + i) as u32;
            mesh.add_triangle(a, a + 1, a + GRID as u32 + 2);
            mesh.add_triangle(a, a + GRID as u32 + 2, a + GRID as u32 + 1);
        }
    }
    painter.add(mesh);
    let accent = crate::theme::Tokens::get(painter.ctx()).accent;
    let warp = s.mode == PerspMode::Warp;
    for p in &s.planes {
        // Plane grid at thirds (like Photoshop's), then the outline.
        let thin = Stroke::new(0.75, Color32::from_rgba_unmultiplied(255, 255, 255, 150));
        for k in 1..3 {
            let t = k as f64 / 3.0;
            painter.line_segment([scr(plane_point(p, t, 0.0, warp)), scr(plane_point(p, t, 1.0, warp))], thin);
            painter.line_segment([scr(plane_point(p, 0.0, t, warp)), scr(plane_point(p, 1.0, t, warp))], thin);
        }
        let q = if warp { &p.dst } else { &p.src };
        let pts: Vec<Pos2> = q.iter().map(|c| scr(*c)).collect();
        painter.add(egui::Shape::closed_line(pts.clone(), Stroke::new(1.5, accent)));
        for c in pts {
            painter.circle_filled(c, 4.5, Color32::WHITE);
            painter.circle_stroke(c, 4.5, Stroke::new(1.25, accent));
        }
    }
    if let Some(Drag::NewPlane { start, cur }) = &s.drag {
        let r = egui::Rect::from_two_pos(scr(*start), scr(*cur));
        painter.rect_stroke(r, 0.0, Stroke::new(1.0, accent), egui::StrokeKind::Inside);
    }
}

pub fn options_bar(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let Some(s) = app.distort.perspective.as_mut() else { return };
    if crate::widgets::pill_tab(ui, "Layout", s.mode == PerspMode::Layout).clicked() {
        set_mode(s, PerspMode::Layout);
    }
    if crate::widgets::pill_tab(ui, "Warp", s.mode == PerspMode::Warp).clicked() && !s.planes.is_empty() {
        set_mode(s, PerspMode::Warp);
    }
    crate::widgets::vline(ui, 22.0);
    if s.mode == PerspMode::Warp {
        for (label, tip, m) in [
            (tl!("Straighten"), tl!("Automatically straighten near-vertical line segments"), Straighten::Vertical),
            (tl!("Level"), tl!("Automatically level near-horizontal line segments"), Straighten::Horizontal),
            (tl!("Both"), tl!("Automatically straighten and level"), Straighten::Auto),
        ] {
            if ui.button(tl!(&label)).on_hover_text(tip).clicked() {
                straighten(&mut s.planes, m);
                s.update_grid();
            }
        }
    } else {
        ui.label(crate::i18n::trn(crate::i18n::current(), s.planes.len() as u64, "{n} plane", "{n} planes"));
        if ui.button(tl!("Remove Planes")).clicked() {
            s.planes.clear();
            s.update_grid();
        }
    }
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        if crate::widgets::primary_button(ui, "✓", 32.0)
            .on_hover_text(crate::i18n::fmt(tl!("Commit Perspective Warp ({key})"), &[("key", &crate::shortcuts::pretty("Enter"))]))
            .clicked()
        {
            commit(app);
        } else if crate::widgets::secondary_button(ui, "⊘", 32.0).on_hover_text(tl!("Cancel (Esc)")).clicked() {
            app.distort.perspective = None;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_snaps_planes_then_warp_and_commit() {
        let ctx = egui::Context::default();
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.session.execute("file.new", json!({"width": 200, "height": 120, "depth": 8})).unwrap();
        app.session.execute("layer.new.layer", json!({})).unwrap();
        app.session
            .edit("paint", |doc, active| {
                doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap().fill_rect(Rect::new(10, 10, 190, 110), &[0.2, 0.6, 0.9, 1.0]);
                Ok(())
            })
            .unwrap();
        app.sync_views();
        crate::distort_ui::menu(&mut app, &ctx, "edit.perspectiveWarp", &json!({})).unwrap().unwrap();
        let ev = |app: &mut PhotocraftApp, e| crate::distort_ui::pointer(app, e, egui::Modifiers::NONE);
        // Two planes sharing an edge (the second starts at the first one's corner and snaps).
        ev(&mut app, ToolEvent::Down { x: 20.0, y: 20.0, pressure: 1.0 });
        ev(&mut app, ToolEvent::Move { x: 100.0, y: 100.0, pressure: 1.0 });
        ev(&mut app, ToolEvent::Up { x: 100.0, y: 100.0 });
        ev(&mut app, ToolEvent::Down { x: 120.0, y: 30.0, pressure: 1.0 });
        ev(&mut app, ToolEvent::Up { x: 180.0, y: 101.0 });
        // Drop its left corners next to the first plane's right corners: they snap.
        ev(&mut app, ToolEvent::Down { x: 120.0, y: 30.0, pressure: 1.0 });
        ev(&mut app, ToolEvent::Up { x: 102.0, y: 21.0 });
        ev(&mut app, ToolEvent::Down { x: 120.0, y: 101.0, pressure: 1.0 });
        ev(&mut app, ToolEvent::Up { x: 101.0, y: 99.0 });
        let s = app.distort.perspective.as_ref().unwrap();
        assert_eq!(s.planes.len(), 2);
        assert_eq!(s.planes[1].src[0], [100.0, 20.0], "snapped to the first plane's corner");
        assert_eq!(s.planes[1].src[3], [100.0, 100.0]);
        control(&mut app, &json!({"mode": "warp"})).unwrap();
        // Dragging the shared top corner moves both planes.
        ev(&mut app, ToolEvent::Down { x: 100.0, y: 20.0, pressure: 1.0 });
        ev(&mut app, ToolEvent::Up { x: 100.0, y: 5.0 });
        let s = app.distort.perspective.as_ref().unwrap();
        assert_eq!(s.planes[0].dst[1], [100.0, 5.0]);
        assert_eq!(s.planes[1].dst[0], [100.0, 5.0]);
        assert_eq!(s.planes[0].src[1], [100.0, 20.0], "warp mode keeps the layout");
        let before = app.session.active().unwrap().history.past_len();
        commit(&mut app);
        assert_eq!(app.session.active().unwrap().history.past_len(), before + 1);
    }

    #[test]
    fn replacing_the_planes_mid_corner_drag_ends_the_drag() {
        // #952: the drag kept (plane, corner) indices into the old list, so the next move or
        // release indexed the new, shorter one and panicked.
        for release in [false, true] {
            let ctx = egui::Context::default();
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
            app.session.execute("file.new", json!({"width": 100, "height": 80, "depth": 8})).unwrap();
            app.session.execute("layer.new.layer", json!({})).unwrap();
            app.session
                .edit("paint", |doc, active| {
                    doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap().fill_rect(Rect::new(20, 30, 80, 50), &[1.0, 0.0, 0.0, 1.0]);
                    Ok(())
                })
                .unwrap();
            app.sync_views();
            crate::distort_ui::menu(&mut app, &ctx, "edit.perspectiveWarp", &json!({})).unwrap().unwrap();
            let ev = |app: &mut PhotocraftApp, e| crate::distort_ui::pointer(app, e, egui::Modifiers::NONE);
            ev(&mut app, ToolEvent::Down { x: 20.0, y: 20.0, pressure: 1.0 });
            ev(&mut app, ToolEvent::Move { x: 90.0, y: 70.0, pressure: 1.0 });
            ev(&mut app, ToolEvent::Up { x: 90.0, y: 70.0 });
            // Press a corner, then replace the planes while it's held.
            ev(&mut app, ToolEvent::Down { x: 20.0, y: 20.0, pressure: 1.0 });
            assert!(matches!(app.distort.perspective.as_ref().unwrap().drag, Some(Drag::Corner { .. })));
            control(&mut app, &json!({"planes": []})).unwrap();
            if release {
                ev(&mut app, ToolEvent::Up { x: 30.0, y: 30.0 });
            } else {
                ev(&mut app, ToolEvent::Move { x: 30.0, y: 30.0, pressure: 1.0 });
            }
            let s = app.distort.perspective.as_ref().unwrap();
            assert!(s.planes.is_empty() && s.drag.is_none(), "release {release}");
        }
    }
}
