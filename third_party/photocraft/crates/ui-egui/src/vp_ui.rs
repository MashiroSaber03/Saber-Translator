//! Filter › Vanishing Point on the canvas (dialog mode).
//!
//! **Create Plane** (C): four clicks set a plane's corners, in any order. **Edit Plane** (V):
//! drag a corner node, or an edge node to slide that edge along the plane's own perspective;
//! click a plane to select it, Delete removes it. The plane's grid is drawn at the Grid Size and,
//! like Photoshop's, in the accent colour when the plane is valid, yellow when its perspective is
//! doubtful and red when it is invalid ([`VpPlane::check`]). With an image on the clipboard the
//! selected plane shows it pasted in perspective; drag it to move it on the plane. ↩ / OK runs
//! `filter.vanishingPoint` with the planes and the paste, so it is one history step and an
//! action or agent can replay it exactly.

use egui::{Color32, Pos2, Stroke, TextureHandle, pos2};
use photocraft_algo::transform::Homography;
use photocraft_algo::vanishing::{PlaneCheck, Scene, VpPlane};
use photocraft_doc::{LayerContent, LayerId};
use photocraft_engine::vp_cmds::VP;
use photocraft_geom::Rect;
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::canvas::{ToolEvent, ViewXform};
use crate::edit_history::{EditHistory, Step};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub enum VpTool {
    EditPlane,
    CreatePlane,
}

enum Drag {
    /// A corner node, and the pointer offset from it.
    Corner { plane: usize, corner: usize, offset: [f64; 2] },
    /// An edge node (0 top, 1 right, 2 bottom, 3 left): the edge slides along the plane's
    /// perspective as it was at the press.
    Edge { plane: usize, edge: usize, h: Homography, inv: Homography },
    /// The pasted image, and the pointer offset from its top-left in plane coordinates.
    Paste { offset: [f64; 2] },
}

/// The clipboard image shown pasted onto the plane.
struct Clip {
    texture: TextureHandle,
    width: f64,
    height: f64,
}

pub struct VpSession {
    pub layer: LayerId,
    canvas: Rect,
    pub planes: Vec<VpPlane>,
    pub tool: VpTool,
    /// Corners clicked so far with Create Plane.
    pending: Vec<[f64; 2]>,
    pub selected: Option<usize>,
    /// Grid cell size in px at the plane's nearest edge.
    pub grid_size: f64,
    /// Paste the clipboard onto the selected plane on OK.
    pub paste: bool,
    /// Top-left of the pasted image in plane coordinates (0..1).
    pub paste_at: [f64; 2],
    /// Width of the pasted image as a fraction of the plane's width.
    pub paste_width: f64,
    pub new_layer: bool,
    clip: Option<Clip>,
    drag: Option<Drag>,
    history: EditHistory<VpEdit>,
}

/// What ⌘Z steps through: the planes and where the paste sits on them.
type VpEdit = (Vec<VpPlane>, [f64; 2], f64);

/// Most planes a session holds (the control channel can't make it allocate without bound).
const MAX_PLANES: usize = 64;
/// Most grid lines drawn per direction.
const MAX_LINES: usize = 200;

impl VpSession {
    pub fn describe(&self) -> Value {
        let center = self.center();
        let planes: Vec<Value> = self.planes.iter().map(|p| json!({"corners": p.corners, "check": p.check(center)})).collect();
        json!({
            "layer": self.layer.0,
            "tool": self.tool,
            "planes": planes,
            "pending": self.pending,
            "selected": self.selected,
            "gridSize": self.grid_size,
            "paste": self.paste && self.clip.is_some(),
            "pasteAt": self.paste_at,
            "pasteWidth": self.paste_width,
            "newLayer": self.new_layer,
        })
    }

    fn center(&self) -> [f64; 2] {
        let c = self.canvas;
        [f64::from(c.x0 + c.x1) / 2.0, f64::from(c.y0 + c.y1) / 2.0]
    }

    /// The planes with the shared camera the engine derives from the first one.
    fn scene(&self) -> Option<Scene> {
        let first = self.planes.first()?.clone();
        let fallback = f64::from(self.canvas.width().max(self.canvas.height()));
        let mut sc = Scene::new(first, self.center(), None, fallback);
        sc.planes.extend(self.planes.iter().skip(1).cloned());
        Some(sc)
    }

    /// Plane `i`'s unit square → image map and metric side lengths.
    fn frame(&self, i: usize) -> Option<(Homography, f64, f64)> {
        if i >= self.planes.len() {
            return None;
        }
        let (h, lu, lv) = self.scene()?.frame(i)?;
        (lu.is_finite() && lv.is_finite() && lu > 1e-12 && lv > 1e-12).then_some((h, lu, lv))
    }

    /// The plane the paste lands on.
    fn paste_plane(&self) -> Option<usize> {
        if !self.paste || self.clip.is_none() {
            return None;
        }
        self.selected.filter(|i| *i < self.planes.len()).or_else(|| (!self.planes.is_empty()).then_some(0))
    }

    /// Size of the pasted image in plane coordinates (the engine keeps its metric aspect).
    fn paste_extent(&self, lu: f64, lv: f64) -> Option<[f64; 2]> {
        let c = self.clip.as_ref()?;
        Some([self.paste_width, self.paste_width * lu / lv * c.height / c.width.max(1.0)])
    }

    /// The command's params: the planes, and the paste when there is one.
    pub fn params(&self) -> Value {
        let planes: Vec<Value> = self.planes.iter().map(|p| json!({"corners": p.corners})).collect();
        let mut p = json!({"planes": planes});
        if let Some(i) = self.paste_plane() {
            p["paste"] = json!([{"plane": i, "at": self.paste_at, "width": self.paste_width}]);
        }
        if self.new_layer {
            p["newLayer"] = json!(true);
        }
        p
    }

    fn add_corner(&mut self, p: [f64; 2]) {
        if self.planes.len() >= MAX_PLANES {
            return;
        }
        self.pending.push(p);
        if let Ok(corners) = <[[f64; 2]; 4]>::try_from(self.pending.as_slice()) {
            self.pending.clear();
            self.planes.push(VpPlane { corners }.normalized());
            self.selected = Some(self.planes.len() - 1);
            // Photoshop hands over to Edit Plane once a plane is closed.
            self.tool = VpTool::EditPlane;
        }
    }

    fn edit(&self) -> VpEdit {
        (self.planes.clone(), self.paste_at, self.paste_width)
    }

    fn settle(&mut self, busy: bool) {
        let now = self.edit();
        self.history.settle(&now, busy || self.drag.is_some());
    }

    pub(crate) fn step(&mut self, step: Step) {
        let mut now = self.edit();
        if self.history.step(&mut now, step) {
            (self.planes, self.paste_at, self.paste_width) = now;
            self.drag = None;
            self.selected = self.selected.filter(|i| *i < self.planes.len());
        }
    }

    fn delete_selected(&mut self) {
        if let Some(i) = self.selected.take().filter(|i| *i < self.planes.len()) {
            self.planes.remove(i);
            self.drag = None;
        }
    }

    /// Image positions of the edge nodes of plane `i`.
    fn edge_nodes(&self, i: usize) -> Option<[[f64; 2]; 4]> {
        let h = self.planes.get(i)?.homography()?;
        let at = |u: f64, v: f64| {
            let (x, y) = h.apply(u, v);
            [x, y]
        };
        Some([at(0.5, 0.0), at(1.0, 0.5), at(0.5, 1.0), at(0.0, 0.5)])
    }

    fn press(&mut self, p: [f64; 2], tol: f64) {
        if self.tool == VpTool::CreatePlane {
            self.add_corner(p);
            return;
        }
        let near = |c: [f64; 2]| (c[0] - p[0]).hypot(c[1] - p[1]) <= tol;
        // The selected plane's nodes win over the others'.
        let order: Vec<usize> = self.selected.into_iter().chain((0..self.planes.len()).rev()).filter(|i| *i < self.planes.len()).collect();
        for &i in &order {
            if let Some(ci) = self.planes[i].corners.iter().position(|c| near(*c)) {
                let c = self.planes[i].corners[ci];
                self.selected = Some(i);
                self.drag = Some(Drag::Corner { plane: i, corner: ci, offset: [c[0] - p[0], c[1] - p[1]] });
                return;
            }
            if let Some(e) = self.edge_nodes(i).and_then(|n| n.iter().position(|c| near(*c)))
                && let Some(h) = self.planes[i].homography()
                && let Some(inv) = h.inverse()
            {
                self.selected = Some(i);
                self.drag = Some(Drag::Edge { plane: i, edge: e, h, inv });
                return;
            }
        }
        if let Some(i) = self.paste_plane()
            && let Some((h, lu, lv)) = self.frame(i)
            && let Some(ext) = self.paste_extent(lu, lv)
            && let Some(inv) = h.inverse()
        {
            let (u, v) = inv.apply(p[0], p[1]);
            let (a, b) = (self.paste_at, [self.paste_at[0] + ext[0], self.paste_at[1] + ext[1]]);
            if (a[0]..=b[0]).contains(&u) && (a[1]..=b[1]).contains(&v) {
                self.drag = Some(Drag::Paste { offset: [a[0] - u, a[1] - v] });
                return;
            }
        }
        self.selected = self.scene().and_then(|sc| sc.plane_at(p));
    }

    fn drag_to(&mut self, p: [f64; 2]) {
        match &self.drag {
            Some(Drag::Corner { plane, corner, offset }) => {
                if let Some(c) = self.planes.get_mut(*plane).and_then(|pl| pl.corners.get_mut(*corner)) {
                    *c = [p[0] + offset[0], p[1] + offset[1]];
                }
            }
            Some(Drag::Edge { plane, edge, h, inv }) => {
                let (u, v) = inv.apply(p[0], p[1]);
                // The plane's (u, v) rectangle after the move.
                let (u0, v0, u1, v1) = match edge {
                    0 => (0.0, v.min(0.99), 1.0, 1.0),
                    1 => (0.0, 0.0, u.max(0.01), 1.0),
                    2 => (0.0, 0.0, 1.0, v.max(0.01)),
                    _ => (u.min(0.99), 0.0, 1.0, 1.0),
                };
                if let Some(q) = slid_quad(h, u0, v0, u1, v1)
                    && let Some(pl) = self.planes.get_mut(*plane)
                {
                    pl.corners = q;
                }
            }
            Some(Drag::Paste { offset }) => {
                let offset = *offset;
                if let Some(i) = self.paste_plane()
                    && let Some(inv) = self.planes.get(i).and_then(VpPlane::homography).and_then(|h| h.inverse())
                {
                    let (u, v) = inv.apply(p[0], p[1]);
                    let at = [(u + offset[0]).clamp(-1.0, 2.0), (v + offset[1]).clamp(-1.0, 2.0)];
                    if at.iter().all(|x| x.is_finite()) {
                        self.paste_at = at;
                    }
                }
            }
            None => {}
        }
    }
}

/// Corners of the image of `[u0, u1] × [v0, v1]` under `h`, or `None` when part of it would lie
/// on or beyond the plane's horizon (where the edge can't follow the perspective).
fn slid_quad(h: &Homography, u0: f64, v0: f64, u1: f64, v1: f64) -> Option<[[f64; 2]; 4]> {
    let m = &h.0;
    let w = |u: f64, v: f64| m[6] * u + m[7] * v + m[8];
    let side = w(0.5, 0.5).signum();
    let uv = [[u0, v0], [u1, v0], [u1, v1], [u0, v1]];
    if uv.iter().any(|c| w(c[0], c[1]) * side <= 1e-9 * w(0.5, 0.5).abs()) {
        return None;
    }
    let q = uv.map(|c| {
        let (x, y) = h.apply(c[0], c[1]);
        [x, y]
    });
    (VpPlane { corners: q }.check([0.0, 0.0]) != PlaneCheck::Invalid).then_some(q)
}

/// Starts Vanishing Point on the active pixel layer.
pub fn begin(app: &mut PhotocraftApp, ctx: &egui::Context) -> Result<(), String> {
    if app.ui.transform.is_some() {
        return Err("finish Free Transform first".into());
    }
    let st = app.session.active().ok_or("no document")?;
    let layer = st.active_layer.ok_or("no active layer")?;
    match st.doc.layer(layer).map(|l| &l.content) {
        Some(LayerContent::Raster(_)) => {}
        Some(other) => return Err(format!("Vanishing Point needs a pixel layer (active layer is {} {} layer)", other.article(), other.kind_name())),
        None => return Err("no active layer".into()),
    }
    let canvas = st.doc.bounds();
    let key = app.ui.alloc_id();
    let clip = app.session.clipboard.as_ref().and_then(|c| {
        let r = c.surface.content_bounds();
        (!r.is_empty()).then(|| Clip {
            texture: ctx.load_texture(format!("vp-paste-{key}"), crate::distort_ui::surface_image(&c.surface, r, 1024), egui::TextureOptions::LINEAR),
            width: f64::from(r.width()),
            height: f64::from(r.height()),
        })
    });
    app.distort.vanishing = Some(VpSession {
        layer,
        canvas,
        planes: Vec::new(),
        tool: VpTool::CreatePlane,
        pending: Vec::new(),
        selected: None,
        grid_size: 100.0,
        paste: clip.is_some(),
        paste_at: [0.25, 0.25],
        paste_width: 0.5,
        new_layer: false,
        clip,
        drag: None,
        history: EditHistory::new(&(Vec::new(), [0.25, 0.25], 0.5)),
    });
    app.ui.status = tl!("Vanishing Point: click four corners to create a plane").into();
    Ok(())
}

/// OK: runs `filter.vanishingPoint` with the planes (and the paste). An invalid plane keeps the
/// mode open; with nothing to paste the mode just closes.
pub fn commit(app: &mut PhotocraftApp) -> Result<Value, String> {
    let Some(s) = app.distort.vanishing.as_ref() else { return Ok(Value::Null) };
    let center = s.center();
    if s.planes.iter().any(|p| p.check(center) == PlaneCheck::Invalid) {
        let msg = tl!("Fix the red plane first: its corners must form a convex quad");
        app.ui.status = msg.into();
        return Err(msg.into());
    }
    let params = s.params();
    app.distort.vanishing = None;
    if params.get("paste").is_none() {
        return Ok(json!({"committed": false}));
    }
    app.run(VP, params).map(|r| json!({"committed": true, "result": r}))
}

/// Control channel: `filter.vanishingPoint {"ui": {...}}` while the mode is active.
pub fn control(app: &mut PhotocraftApp, ui: &Value) -> Result<Value, String> {
    if ui.get("commit").and_then(Value::as_bool) == Some(true) {
        return commit(app);
    }
    if ui.get("cancel").and_then(Value::as_bool) == Some(true) {
        app.distort.vanishing = None;
        return Ok(json!({"cancelled": true}));
    }
    let s = app.distort.vanishing.as_mut().ok_or(tl!("Vanishing Point is not active"))?;
    if let Some(t) = ui.get("tool").and_then(Value::as_str) {
        s.tool = match t {
            "edit" | "editPlane" => VpTool::EditPlane,
            "create" | "createPlane" => VpTool::CreatePlane,
            other => return Err(format!("tool: edit|create, not `{other}`")),
        };
        s.pending.clear();
    }
    if let Some(p) = ui.get("planes") {
        let planes: Vec<VpPlane> = serde_json::from_value(p.clone()).map_err(|e| format!("bad planes: {e}"))?;
        if planes.len() > MAX_PLANES {
            return Err(format!("at most {MAX_PLANES} planes"));
        }
        s.planes = planes;
        // A drag holds indices into the old list.
        s.drag = None;
        s.selected = (!s.planes.is_empty()).then(|| s.planes.len() - 1);
    }
    if let Some(i) = ui.get("select").and_then(Value::as_u64) {
        s.selected = usize::try_from(i).ok().filter(|i| *i < s.planes.len());
    }
    if let Some(g) = ui.get("gridSize").and_then(Value::as_f64).filter(|g| g.is_finite()) {
        s.grid_size = g.clamp(2.0, 2000.0);
    }
    if let Some(b) = ui.get("paste").and_then(Value::as_bool) {
        s.paste = b;
    }
    if let Some(at) = ui.get("pasteAt").and_then(|v| Some([v.get(0)?.as_f64()?, v.get(1)?.as_f64()?])) {
        s.paste_at = at.map(|x| if x.is_finite() { x.clamp(-1.0, 2.0) } else { 0.25 });
    }
    if let Some(w) = ui.get("pasteWidth").and_then(Value::as_f64).filter(|w| w.is_finite()) {
        s.paste_width = w.clamp(0.01, 10.0);
    }
    if let Some(b) = ui.get("newLayer").and_then(Value::as_bool) {
        s.new_layer = b;
    }
    s.settle(false);
    Ok(s.describe())
}

pub fn pointer(app: &mut PhotocraftApp, ev: ToolEvent, _mods: egui::Modifiers) {
    let tol = crate::distort_ui::tolerance(app);
    let Some(s) = app.distort.vanishing.as_mut() else { return };
    match ev {
        ToolEvent::Down { x, y, .. } => s.press([x, y], tol),
        ToolEvent::Move { x, y, .. } => s.drag_to([x, y]),
        ToolEvent::Up { x, y } => {
            s.drag_to([x, y]);
            s.drag = None;
        }
    }
    s.settle(false);
}

/// Keys while the mode is active (↩ and Esc are handled by `distort_ui::keys`).
pub fn keys(app: &mut PhotocraftApp, ctx: &egui::Context) {
    use egui::{Key, Modifiers};
    if ctx.memory(|m| m.focused().is_some()) {
        return;
    }
    let Some(s) = app.distort.vanishing.as_mut() else { return };
    let key = |k: Key| ctx.input_mut(|i| i.consume_key(Modifiers::NONE, k));
    if key(Key::V) {
        s.tool = VpTool::EditPlane;
        s.pending.clear();
    } else if key(Key::C) {
        s.tool = VpTool::CreatePlane;
    } else if key(Key::Delete) || key(Key::Backspace) {
        s.delete_selected();
        s.settle(false);
    }
}

/// Esc: drops a half-clicked plane first, then leaves the mode.
pub fn escape(app: &mut PhotocraftApp) {
    match app.distort.vanishing.as_mut() {
        Some(s) if !s.pending.is_empty() => s.pending.clear(),
        _ => app.distort.vanishing = None,
    }
}

pub fn draw(s: &VpSession, painter: &egui::Painter, xf: &ViewXform) {
    let scr = |p: [f64; 2]| xf.to_screen(p[0] as f32, p[1] as f32);
    let tokens = crate::theme::Tokens::get(painter.ctx());
    let center = s.center();
    // The paste first, under the grids.
    if let Some(i) = s.paste_plane()
        && s.planes.get(i).is_some_and(|p| p.check(center) != PlaneCheck::Invalid)
        && let (Some((h, lu, lv)), Some(c)) = (s.frame(i), s.clip.as_ref())
        && let Some(ext) = s.paste_extent(lu, lv)
    {
        draw_paste(painter, &scr, &h, s.paste_at, ext, &c.texture);
    }
    for (i, p) in s.planes.iter().enumerate() {
        let check = p.check(center);
        let colour = match check {
            PlaneCheck::Valid => tokens.accent,
            PlaneCheck::Questionable => tokens.warning,
            PlaneCheck::Invalid => tokens.danger,
        };
        let selected = s.selected == Some(i);
        // An invalid plane has no perspective to grid: just its red outline, as in Photoshop.
        if check != PlaneCheck::Invalid
            && let Some((h, lu, lv)) = s.frame(i)
        {
            let thin = Stroke::new(0.75, colour.gamma_multiply(0.6));
            let at = |u: f64, v: f64| {
                let (x, y) = h.apply(u, v);
                scr([x, y])
            };
            for u in grid_steps(p, lu, lv, s.grid_size, true) {
                painter.line_segment([at(u, 0.0), at(u, 1.0)], thin);
            }
            for v in grid_steps(p, lu, lv, s.grid_size, false) {
                painter.line_segment([at(0.0, v), at(1.0, v)], thin);
            }
        }
        let pts: Vec<Pos2> = p.corners.iter().map(|c| scr(*c)).collect();
        painter.add(egui::Shape::closed_line(pts.clone(), Stroke::new(if selected { 2.0 } else { 1.25 }, colour)));
        if selected {
            let nodes = pts.into_iter().chain(s.edge_nodes(i).into_iter().flatten().map(scr));
            for c in nodes {
                let r = egui::Rect::from_center_size(c, egui::vec2(7.0, 7.0));
                painter.rect_filled(r, 0.0, Color32::WHITE);
                painter.rect_stroke(r, 0.0, Stroke::new(1.25, colour), egui::StrokeKind::Inside);
            }
        }
    }
    // Create Plane: the corners so far and a rubber band to the pointer.
    if !s.pending.is_empty() {
        let mut pts: Vec<Pos2> = s.pending.iter().map(|c| scr(*c)).collect();
        if let Some(hover) = painter.ctx().pointer_hover_pos() {
            pts.push(hover);
        }
        painter.add(egui::Shape::line(pts.clone(), Stroke::new(1.25, tokens.accent)));
        for c in pts.iter().take(s.pending.len()) {
            painter.rect_filled(egui::Rect::from_center_size(*c, egui::vec2(6.0, 6.0)), 0.0, tokens.accent);
        }
    }
}

/// Plane coordinates (0..1, exclusive) of the grid lines across `u` (or `v`) for cells of
/// `size` px at the plane's nearest edge, square in the plane's metric.
fn grid_steps(p: &VpPlane, lu: f64, lv: f64, size: f64, along_u: bool) -> Vec<f64> {
    let q = &p.corners;
    let d = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).hypot(a[1] - b[1]);
    // Image px per metric unit at the nearest (longest-looking) edge.
    let k = [d(q[0], q[1]) / lu, d(q[3], q[2]) / lu, d(q[0], q[3]) / lv, d(q[1], q[2]) / lv].into_iter().fold(0.0, f64::max);
    let cell = size.max(1.0) / k;
    let len = if along_u { lu } else { lv };
    if !cell.is_finite() || cell <= 0.0 || !len.is_finite() {
        return Vec::new();
    }
    let step = cell / len;
    (1..MAX_LINES).map(|j| j as f64 * step).take_while(|t| *t < 1.0 - 1e-9).collect()
}

/// The clipboard image on the plane: a mesh through the plane's homography, cells outside the
/// plane left out (the engine clips the paste to the plane).
fn draw_paste(painter: &egui::Painter, scr: &dyn Fn([f64; 2]) -> Pos2, h: &Homography, at: [f64; 2], ext: [f64; 2], tex: &TextureHandle) {
    const N: usize = 24;
    let mut mesh = egui::Mesh::with_texture(tex.id());
    let tint = Color32::from_white_alpha(230);
    for j in 0..=N {
        for i in 0..=N {
            let (s, t) = (i as f64 / N as f64, j as f64 / N as f64);
            let (x, y) = h.apply(at[0] + s * ext[0], at[1] + t * ext[1]);
            mesh.vertices.push(egui::epaint::Vertex { pos: scr([x, y]), uv: pos2(s as f32, t as f32), color: tint });
        }
    }
    for j in 0..N {
        for i in 0..N {
            let (u, v) = (at[0] + (i as f64 + 0.5) / N as f64 * ext[0], at[1] + (j as f64 + 0.5) / N as f64 * ext[1]);
            if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
                continue;
            }
            let a = (j * (N + 1) + i) as u32;
            mesh.add_triangle(a, a + 1, a + N as u32 + 2);
            mesh.add_triangle(a, a + N as u32 + 2, a + N as u32 + 1);
        }
    }
    painter.add(mesh);
}

pub fn options_bar(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let Some(s) = app.distort.vanishing.as_mut() else { return };
    if crate::widgets::pill_tab(ui, tl!("Edit Plane"), s.tool == VpTool::EditPlane).on_hover_text("V").clicked() {
        s.tool = VpTool::EditPlane;
        s.pending.clear();
    }
    if crate::widgets::pill_tab(ui, tl!("Create Plane"), s.tool == VpTool::CreatePlane).on_hover_text("C").clicked() {
        s.tool = VpTool::CreatePlane;
    }
    crate::widgets::vline(ui, 22.0);
    ui.label(tl!("Grid Size:"));
    let mut g = s.grid_size as f32;
    if crate::widgets::value_field(ui, &mut g, 2.0..=2000.0, "px", 56.0).changed() {
        s.grid_size = f64::from(g);
    }
    if s.clip.is_some() {
        crate::widgets::vline(ui, 22.0);
        crate::widgets::checkbox(ui, &mut s.paste, tl!("Paste"));
        if s.paste {
            ui.label(tl!("Width:"));
            let mut w = (s.paste_width * 100.0) as f32;
            if crate::widgets::value_field(ui, &mut w, 1.0..=1000.0, "%", 56.0).changed() {
                s.paste_width = f64::from(w) / 100.0;
            }
        }
        crate::widgets::checkbox(ui, &mut s.new_layer, tl!("New Layer"));
    }
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        if crate::widgets::primary_button(ui, "✓", 32.0)
            .on_hover_text(crate::i18n::fmt(tl!("Apply Vanishing Point ({key})"), &[("key", &crate::shortcuts::pretty("Enter"))]))
            .clicked()
        {
            let _ = commit(app);
        } else if crate::widgets::secondary_button(ui, "⊘", 32.0).on_hover_text(tl!("Cancel (Esc)")).clicked() {
            app.distort.vanishing = None;
        }
    });
    if let Some(s) = app.distort.vanishing.as_mut() {
        s.settle(ui.input(|i| i.pointer.any_down()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app_with_layer() -> (egui::Context, PhotocraftApp) {
        let ctx = egui::Context::default();
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.session.execute("file.new", json!({"width": 240, "height": 160, "depth": 16})).unwrap();
        app.session.execute("layer.new.layer", json!({})).unwrap();
        app.session.execute("edit.fill", json!({"color": "#4080c0"})).unwrap();
        app.sync_views();
        (ctx, app)
    }

    #[test]
    fn four_clicks_make_a_plane_and_ok_pastes_in_one_step() {
        let (ctx, mut app) = app_with_layer();
        // Something to paste: a red square on the clipboard.
        app.session.execute("select.rect", json!({"x": 0, "y": 0, "width": 20, "height": 20})).unwrap();
        app.session.execute("edit.fill", json!({"color": "#ff0000"})).unwrap();
        app.session.execute("edit.copy", json!({})).unwrap();
        app.session.execute("select.deselect", json!({})).unwrap();
        crate::distort_ui::menu(&mut app, &ctx, VP, &json!({})).unwrap().unwrap();
        let click = |app: &mut PhotocraftApp, x: f64, y: f64| {
            crate::distort_ui::pointer(app, ToolEvent::Down { x, y, pressure: 1.0 }, egui::Modifiers::NONE);
            crate::distort_ui::pointer(app, ToolEvent::Up { x, y }, egui::Modifiers::NONE);
        };
        // A floor receding to the top, clicked counter-clockwise from the bottom-left.
        for (x, y) in [(20.0, 150.0), (220.0, 150.0), (170.0, 60.0), (70.0, 60.0)] {
            click(&mut app, x, y);
        }
        let s = app.distort.vanishing.as_ref().unwrap();
        assert_eq!(s.planes.len(), 1);
        assert_eq!(s.planes[0].corners, [[70.0, 60.0], [170.0, 60.0], [220.0, 150.0], [20.0, 150.0]], "clockwise from the top-left");
        assert_eq!(s.tool, VpTool::EditPlane);
        assert_eq!(s.describe()["planes"][0]["check"], "valid");
        // Edit Plane: drag the top edge node up; the edge slides along the perspective.
        let top = s.edge_nodes(0).unwrap()[0];
        crate::distort_ui::pointer(&mut app, ToolEvent::Down { x: top[0], y: top[1], pressure: 1.0 }, egui::Modifiers::NONE);
        crate::distort_ui::pointer(&mut app, ToolEvent::Up { x: top[0], y: top[1] - 15.0 }, egui::Modifiers::NONE);
        let c = app.distort.vanishing.as_ref().unwrap().planes[0].corners;
        assert!(c[0][1] < 60.0 && c[1][1] < 60.0 && c[1][0] - c[0][0] < 100.0, "taller and narrower at the far edge: {c:?}");
        assert_eq!(c[2], [220.0, 150.0]);
        let before = app.session.active().unwrap().history.past_len();
        commit(&mut app).unwrap();
        assert!(app.distort.vanishing.is_none());
        assert_eq!(app.session.active().unwrap().history.past_len(), before + 1);
        // The paste landed on the plane: red at its middle, blue outside it.
        let st = app.session.active().unwrap();
        let l = st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap();
        let h = VpPlane { corners: c }.homography().unwrap();
        let (x, y) = h.apply(0.5, 0.35);
        assert!(l.rgba(x as i32, y as i32)[0] > 0.9, "{:?}", l.rgba(x as i32, y as i32));
        assert!(l.rgba(230, 20)[0] < 0.5, "outside the plane");
    }
}
