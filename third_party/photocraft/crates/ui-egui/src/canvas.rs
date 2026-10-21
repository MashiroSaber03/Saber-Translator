//! Document canvas: display texture cache, view transform, tool input, extra document windows.
//!
//! Rendering is CPU (`photocraft-compose`) for now, uploaded into an egui texture. Brush strokes
//! update only their damage rectangle (`set_partial`). The wgpu compositor (M5) will replace this
//! with direct GPU rendering behind the same `CanvasCache` interface.

use egui::{Color32, Mesh, PointerButton, Pos2, Rect, Sense, Stroke, TextureOptions, Vec2, pos2, vec2};
use photocraft_doc::{Document, LayerContent};
use photocraft_geom::Rect as DRect;
use serde_json::json;

use crate::PhotocraftApp;
use crate::state::{Tool, View};

/// Largest texture side we upload; bigger documents display downsampled until the GPU path lands.
pub const MAX_TEXTURE: u32 = 4096;

pub(crate) struct ClonePreviewCache {
    key: String,
    texture: egui::TextureHandle,
}

/// A transient source image beneath the brush outline; never part of the document render.
fn draw_clone_preview(app: &mut PhotocraftApp, painter: &egui::Painter, xf: &ViewXform, at: [f64; 2], output: Option<u32>) {
    use photocraft_engine::presets::clone_source::{Mapping, overlay_blend, transform_matrix};
    let overlay = app.session.presets.clone.overlay.clone();
    if app.ui.tool != Tool::CloneStamp || !overlay.show || (overlay.auto_hide && app.drag.is_some()) {
        return;
    }
    let slot = app.session.presets.clone.active().clone();
    let Some(source) = slot.source else { return };
    let radius = f64::from(app.session.tools.brush.size) / 2.0;
    if !radius.is_finite() || radius <= 0.0 || radius > 500.0 || !at.iter().all(|v| v.is_finite() && v.abs() < 1_000_000.0) {
        return;
    }
    let stroke_start = app.drag.as_ref().map(|d| d.start);
    let anchor = if app.ui.tool_options.clone_aligned { slot.anchor.or(stroke_start) } else { stroke_start }.unwrap_or(at);
    let map =
        Mapping { source: (source[0], source[1]), anchor: (anchor[0], anchor[1]), m: transform_matrix(slot.scale, slot.rotation, slot.flip_h, slot.flip_v) };
    let rect = DRect::new((at[0] - radius).floor() as i32, (at[1] - radius).floor() as i32, (at[0] + radius).ceil() as i32, (at[1] + radius).ceil() as i32);
    let Some(st) = app.session.active() else { return };
    let (display, display_key) = canvas_display(app, &st.doc, output);
    // A shaped tip (sampled bitmap, or a rotated / flattened round one) clips the overlay to its
    // outline instead of a circle, as Photoshop does with Clipped on.
    let tip_rings = if overlay.clipped { crate::brush_cursor::rings(painter.ctx(), &app.session.tools.brush, true) } else { None };
    let tip_key = tip_rings.as_ref().map(|r| {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        r.iter().flatten().for_each(|p| (p[0].to_bits(), p[1].to_bits()).hash(&mut h));
        (h.finish(), radius.to_bits(), at.map(f64::to_bits))
    });
    let key = format!(
        "{:?}:{}:{:?}:{rect:?}:{map:?}:{}:{}:{}:{}:{}:{tip_key:?}",
        st.doc.id, st.revision, st.active_layer, app.ui.tool_options.clone_sample, display_key, overlay.invert, overlay.blend, app.ui.view.pattern_preview
    );
    if app.clone_preview.as_ref().is_none_or(|c| c.key != key) {
        let Ok(mut buf) =
            photocraft_engine::retouch_cmds::clone_preview_periodic(&app.session, rect, &map, &app.ui.tool_options.clone_sample, app.ui.view.pattern_preview)
        else {
            return;
        };
        if overlay.invert {
            for px in &mut buf.px {
                for c in px.iter_mut().take(3) {
                    *c = 1.0 - *c;
                }
            }
        }
        // Darken, Lighten and Difference mix the source with the image under the brush (the
        // composite, whatever Sample reads), so Difference turns black where they line up.
        if overlay.blend != "normal" {
            let here = Mapping { source: (0.0, 0.0), anchor: (0.0, 0.0), m: [1.0, 0.0, 0.0, 1.0] };
            let Ok(under) = photocraft_engine::retouch_cmds::clone_preview_periodic(&app.session, rect, &here, "all", app.ui.view.pattern_preview) else {
                return;
            };
            for (px, d) in buf.px.iter_mut().zip(&under.px) {
                *px = overlay_blend(&overlay.blend, *px, *d);
            }
        }
        if let Some(rings) = &tip_rings {
            let mask = crate::brush_cursor::tip_mask(rings, at, radius, [rect.x0, rect.y0, rect.x1, rect.y1]);
            for (px, inside) in buf.px.iter_mut().zip(mask) {
                if !inside {
                    *px = [0.0; 4];
                }
            }
        }
        let image = display_image(display.as_deref(), &buf);
        match app.clone_preview.as_mut() {
            Some(cache) => {
                cache.texture.set(image, TextureOptions::LINEAR);
                cache.key = key;
            }
            None => app.clone_preview = Some(ClonePreviewCache { key, texture: painter.ctx().load_texture("clone-preview", image, TextureOptions::LINEAR) }),
        }
    }
    let Some(cache) = &app.clone_preview else { return };
    let tint = Color32::from_white_alpha((overlay.opacity.clamp(0.0, 100.0) * 2.55).round() as u8);
    // A textured triangle fan provides a circular clip, including mirrored canvas views.
    let mut mesh = egui::Mesh::with_texture(cache.texture.id());
    let vertex = |x: f64, y: f64| egui::epaint::Vertex {
        pos: xf.to_screen(x as f32, y as f32),
        uv: pos2((x - f64::from(rect.x0)) as f32 / rect.width() as f32, (y - f64::from(rect.y0)) as f32 / rect.height() as f32),
        color: tint,
    };
    if !overlay.clipped || tip_rings.is_some() {
        for (x, y) in [(rect.x0, rect.y0), (rect.x1, rect.y0), (rect.x1, rect.y1), (rect.x0, rect.y1)] {
            mesh.vertices.push(vertex(f64::from(x), f64::from(y)));
        }
        mesh.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
        painter.add(egui::Shape::mesh(mesh));
        return;
    }
    mesh.vertices.push(vertex(at[0], at[1]));
    for i in 0..=64 {
        let angle = f64::from(i) * std::f64::consts::TAU / 64.0;
        mesh.vertices.push(vertex(at[0] + radius * angle.cos(), at[1] + radius * angle.sin()));
    }
    for i in 1..=64 {
        mesh.indices.extend_from_slice(&[0, i, i + 1]);
    }
    painter.add(egui::Shape::mesh(mesh));
}

pub struct CanvasCache {
    pub revision: u64,
    pub texture: Option<egui::TextureHandle>,
    /// Display pixels per document pixel of the cached image (≤ 1).
    pub scale: f32,
    /// Hash of the live-adjust preview used for this render (0 = none).
    pub preview_key: u64,
    /// The current image lives in the GPU canvas (`gpu_canvas`) rather than `texture`.
    pub on_gpu: bool,
    /// Revision and preview key `texture` was rendered at. Tracked apart from the GPU canvas's
    /// `revision`, so the Navigator never shows a stale CPU image after GPU-path edits.
    pub tex_revision: u64,
    pub tex_preview_key: u64,
    /// The CPU path's pixel grid for the part of the document in view (pixel_grid.rs).
    pub grid: Option<crate::pixel_grid::GridLines>,
}

/// An in-progress pointer gesture on the canvas.
#[derive(Clone, Debug)]
pub struct Drag {
    pub tool: Tool,
    pub start: [f64; 2],
    pub points: Vec<[f64; 3]>,
    pub modifiers: egui::Modifiers,
    /// A Brush/Eraser stroke that erases: the Eraser, or a right-button Brush drag with
    /// Preferences › Tools › Right-click with painting tools set to Erase.
    pub erase: bool,
    /// A Brush or Pencil stroke in Photoshop's Clear mode (`paint_mouse::clears`, #3439): it
    /// erases, sent as `"mode": "clear"` so History names the tool rather than the Eraser.
    pub clear: bool,
    /// ⇧ constraint of a painting stroke (stroke_constraint.rs).
    pub constrain: Option<crate::stroke_constraint::Axis>,
    /// Modifiers held now (updated on every pointer event of the gesture).
    pub live: egui::Modifiers,
    /// Which of the modifiers held at the press have been released since.
    pub released: egui::Modifiers,
    /// The reposition key (Space) is held: pointer moves move the whole gesture instead of
    /// sizing it (marquees, lasso, shapes; `hold_keys`).
    pub reposition: bool,
    /// A marquee or lasso drag that started inside the selection moves it instead of drawing:
    /// `Some(false)` moves the outline, `Some(true)` moves the floating piece (`select.float`), as
    /// every Move-tool drag with a selection does (`move_ui::moves_selected_pixels_with`).
    pub sel_move: Option<bool>,
    pub lasso: Option<crate::lasso_ui::Lasso>,
}

impl Drag {
    pub fn new(tool: Tool, start: [f64; 2], points: Vec<[f64; 3]>, modifiers: egui::Modifiers, erase: bool) -> Self {
        Self {
            tool,
            start,
            points,
            modifiers,
            erase,
            clear: false,
            constrain: None,
            live: modifiers,
            released: egui::Modifiers::NONE,
            reposition: false,
            sel_move: None,
            lasso: None,
        }
    }

    /// Reposition: move the start and every point so the last one lands on `to` (same size).
    pub fn shift_to(&mut self, to: [f64; 2]) {
        let last = self.points.last().map_or(self.start, |p| [p[0], p[1]]);
        let (dx, dy) = (to[0] - last[0], to[1] - last[1]);
        if !(dx.is_finite() && dy.is_finite()) {
            return;
        }
        self.start = [self.start[0] + dx, self.start[1] + dy];
        for p in &mut self.points {
            p[0] += dx;
            p[1] += dy;
        }
        if self.points.is_empty() {
            self.points.push([to[0], to[1], 1.0]);
        }
    }

    /// Record the modifiers of a pointer event.
    pub(crate) fn track(&mut self, mods: egui::Modifiers) {
        self.live = mods;
        self.released.shift |= !mods.shift;
        self.released.alt |= !mods.alt;
    }

    /// ⇧ (square / circle) and ⌥ (from the centre) for a marquee drag. A modifier held at the press
    /// picks the selection mode instead (add / subtract, #188) until it is released and pressed
    /// again, as in Photoshop.
    pub fn marquee_mods(&self) -> (bool, bool) {
        let fresh = |now: bool, at_press: bool, released: bool| now && (!at_press || released);
        (fresh(self.live.shift, self.modifiers.shift, self.released.shift), fresh(self.live.alt, self.modifiers.alt, self.released.alt))
    }
}

/// The marquee dragged from `d.start` to `end` as two document-space corners: the options-bar
/// style (fixed ratio / fixed size), ⇧ for a square or circle, ⌥ to draw from the centre.
pub fn marquee_corners(o: &crate::state::ToolOptions, d: &Drag, end: [f64; 2]) -> ([f64; 2], [f64; 2]) {
    let (shift, alt) = d.marquee_mods();
    let e = crate::chrome_ui::marquee_end(&o.marquee_style, o.marquee_width as f64, o.marquee_height as f64, shift, d.start, end);
    if !alt {
        return (d.start, e);
    }
    let (dx, dy) = (e[0] - d.start[0], e[1] - d.start[1]);
    // A fixed size is centred on the press point; otherwise the drag is the half extent.
    let (hx, hy) = if o.marquee_style == "fixedSize" { (dx / 2.0, dy / 2.0) } else { (dx, dy) };
    ([d.start[0] - hx, d.start[1] - hy], [d.start[0] + hx, d.start[1] + hy])
}

/// The pixel rectangle a Rectangular/Elliptical Marquee drag selects so far: shown live (ants
/// and readout) exactly as the release commits it, on whole pixel edges as in Photoshop. `None`
/// while the gesture is still a click (it never left the press point by more than a few screen
/// pixels at `zoom`, points per document pixel) or the rectangle has no area: a release then
/// deselects. A one-pixel row or column is a selection (#2391).
pub(crate) fn marquee_preview_px(o: &crate::state::ToolOptions, d: &Drag, zoom: f32) -> Option<[f64; 4]> {
    if !matches!(d.tool, Tool::RectMarquee | Tool::EllipseMarquee) || crate::shape_dialog::is_click(d.start, &d.points, zoom) {
        return None;
    }
    let last = d.points.last().map_or(d.start, |p| [p[0], p[1]]);
    let (a, b) = marquee_corners(o, d, last);
    let r = marquee_px(a, b);
    (r[2] > r[0] && r[3] > r[1]).then_some(r)
}

/// The pixel rectangle `[x0, y0, x1, y1]` a marquee between two corners selects.
pub fn marquee_px(a: [f64; 2], b: [f64; 2]) -> [f64; 4] {
    [a[0].min(b[0]).floor(), a[1].min(b[1]).floor(), a[0].max(b[0]).ceil(), a[1].max(b[1]).ceil()]
}

/// The one-pixel line `[x0, y0, x1, y1]` a Single Row (Single Column) Marquee at `p` selects: the
/// pixel row (column) under `p`, kept on the canvas, across the whole `size`. `None` for other
/// tools, an empty canvas or a point that is not a number.
pub(crate) fn single_line_px(tool: Tool, size: [u32; 2], p: [f64; 2]) -> Option<[f64; 4]> {
    let [w, h] = size.map(f64::from);
    if w < 1.0 || h < 1.0 || !p[0].is_finite() || !p[1].is_finite() {
        return None;
    }
    match tool {
        Tool::SingleRowMarquee => {
            let y = p[1].floor().clamp(0.0, h - 1.0);
            Some([0.0, y, w, y + 1.0])
        }
        Tool::SingleColumnMarquee => {
            let x = p[0].floor().clamp(0.0, w - 1.0);
            Some([x, 0.0, x + 1.0, h])
        }
        _ => None,
    }
}

/// The line a Single Row / Single Column Marquee gesture `d` selects so far: it follows the
/// pointer until the release.
fn single_line_drag_px(app: &PhotocraftApp, d: &Drag) -> Option<[f64; 4]> {
    let st = app.session.active()?;
    let last = d.points.last().map_or(d.start, |p| [p[0], p[1]]);
    single_line_px(d.tool, [st.doc.size.width, st.doc.size.height], last)
}

/// The marching ants of a marquee drag: its rectangle, or the ellipse inscribed in it, built in
/// document space and mapped point by point through the view, so the hinting lands where the
/// release commits the selection.
///
/// A rotated view projects the rectangle onto a rotated one, and an axis-aligned screen box
/// through two of its corners is a different shape in a different place: the commit was always
/// right, only the hinting was drawn elsewhere (#2512). Every point is snapped to the half-pixel
/// grid like every other ants outline (`marching_ants_segments`), so the lines stay crisp at any
/// zoom (#172).
pub(crate) fn marquee_ants(xf: &ViewXform, tool: Tool, a: [f64; 2], b: [f64; 2]) -> Vec<Pos2> {
    let (x0, x1) = (a[0].min(b[0]), a[0].max(b[0]));
    let (y0, y1) = (a[1].min(b[1]), a[1].max(b[1]));
    let at = |x: f64, y: f64| {
        let p = xf.to_screen(x as f32, y as f32);
        pos2(p.x.round() + 0.5, p.y.round() + 0.5)
    };
    if tool != Tool::EllipseMarquee {
        return vec![at(x0, y0), at(x1, y0), at(x1, y1), at(x0, y1)];
    }
    // An axis-aligned view (rotation a multiple of 90 degrees) maps the ellipse onto the one
    // inscribed in the snapped screen box of two opposite corners. Draw that, unsnapped: snapping
    // each vertex on its own turns the nearly flat arcs at the edge midpoints into 1 px stairs,
    // whose antialiased steps wash the ants out to grey on white and black.
    let quarter = xf.rotation.rem_euclid(90.0);
    if quarter < 1e-4 || 90.0 - quarter < 1e-4 {
        let r = Rect::from_two_pos(at(x0, y0), at(x1, y1));
        return crate::tool_feedback::ellipse_points(r);
    }
    // As many vertices as the same ellipse would get drawn on screen (`ellipse_points`).
    let scr = Rect::from_points(&[xf.to_screen(x0 as f32, y0 as f32), xf.to_screen(x1 as f32, y1 as f32)]);
    let n = (((scr.width() + scr.height()) * 0.25) as usize).clamp(16, 256);
    let (cx, cy, rx, ry) = ((x0 + x1) * 0.5, (y0 + y1) * 0.5, (x1 - x0) * 0.5, (y1 - y0) * 0.5);
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64 * std::f64::consts::TAU;
            at(cx + rx * t.cos(), cy + ry * t.sin())
        })
        .collect()
}

/// The ants of the marquee being drawn: `marquee` when the drag has one (the options-bar style
/// and the modifiers applied), the raw press and pointer corners otherwise. This is the whole of
/// what the hinting draws for the selection tools, so the test in `marquee_tests` checks it
/// against the selection the release commits (#2512).
pub(crate) fn marquee_drag_ants(d: &Drag, xf: &ViewXform, marquee: Option<[f64; 4]>) -> Vec<Pos2> {
    let last = d.points.last().map_or(d.start, |p| [p[0], p[1]]);
    let (a, b) = marquee.map_or((d.start, last), |[x0, y0, x1, y1]| ([x0, y0], [x1, y1]));
    marquee_ants(xf, d.tool, a, b)
}

/// The size readout shown beside the cursor while dragging a marquee: the width and height values.
pub fn marquee_readout(r: [f64; 4]) -> [String; 2] {
    [format!("{} px", r[2] - r[0]), format!("{} px", r[3] - r[1])]
}

/// [`marquee_readout`] in the Units & Rulers unit, as Photoshop shows it (cm when the rulers are in
/// centimeters), from the active document's resolution; pixels without a document.
pub fn size_readout(app: &PhotocraftApp, r: [f64; 4]) -> [String; 2] {
    let Some(d) = app.session.active() else { return marquee_readout(r) };
    let (dpi, size) = (f64::from(d.doc.resolution_dpi), d.doc.size);
    readout_in(&app.session.prefs().units_and_rulers, dpi, [size.width as f64, size.height as f64], r)
}

/// `r`'s width and height formatted in `ur`'s ruler unit at `dpi` (percent of `extent`).
pub fn readout_in(ur: &photocraft_engine::prefs::UnitsAndRulers, dpi: f64, extent: [f64; 2], r: [f64; 4]) -> [String; 2] {
    let s = ur.rulers.suffix();
    [format!("{} {s}", ur.format(r[2] - r[0], dpi, extent[0])), format!("{} {s}", ur.format(r[3] - r[1], dpi, extent[1]))]
}

/// Draw the marquee size readout below-right of the cursor (kept on screen), like Photoshop's:
/// two rows, `W:` / `H:` labels on the left and the values right-aligned.
fn draw_marquee_readout(app: &PhotocraftApp, ctx: &egui::Context, cursor: Pos2, values: [String; 2]) {
    draw_value_readout(app, ctx, "marquee-readout", cursor, ["W:", "H:"], values);
}

/// A transformation-values readout beside the pointer (a marquee's or the crop frame's size, an
/// angle), unless Preferences › Tools › Show Transformation Values is off, as in Photoshop (#204).
/// The brush HUD isn't one: Show Transformation Values doesn't hide it, so it calls [`draw_readout`].
pub(crate) fn draw_value_readout<const N: usize>(app: &PhotocraftApp, ctx: &egui::Context, id: &str, cursor: Pos2, labels: [&str; N], values: [String; N]) {
    if app.session.prefs().tools.show_transformation_values {
        draw_readout(ctx, id, cursor, labels, values);
    }
}

/// A readout of `N` rows beside the pointer (labels left, values right-aligned), above everything.
pub(crate) fn draw_readout<const N: usize>(ctx: &egui::Context, id: &str, cursor: Pos2, labels: [&str; N], values: [String; N]) {
    let t = crate::theme::Tokens::get(ctx);
    let font = egui::FontId::proportional(11.5);
    let width = |text: &str| ctx.fonts_mut(|f| f.layout_no_wrap(text.to_owned(), font.clone(), t.text).size().x);
    let labels = labels.map(|label| tl!(label));
    let (lw, vw) = (labels.map(width), values.each_ref().map(|v| width(v)));
    let (label_col, value_col) = (lw.iter().fold(0.0f32, |a, &b| a.max(b)), vw.iter().fold(0.0f32, |a, &b| a.max(b)));
    egui::Area::new(egui::Id::new(id)).order(egui::Order::Tooltip).fixed_pos(cursor + vec2(16.0, 18.0)).interactable(false).constrain(true).show(ctx, |ui| {
        egui::Frame::new().fill(t.card).stroke(Stroke::new(1.0, t.card_border)).corner_radius(t.radius_sm).inner_margin(egui::Margin::symmetric(7, 4)).show(
            ui,
            |ui| {
                ui.spacing_mut().item_spacing = vec2(0.0, 1.0);
                for (i, value) in values.into_iter().enumerate() {
                    // Labels left, values right-aligned on one edge, sharing a baseline.
                    ui.horizontal(|ui| {
                        ui.add(egui::Label::new(egui::RichText::new(labels[i]).font(font.clone()).color(t.text_dim)).extend());
                        ui.add_space(label_col - lw[i] + 12.0 + value_col - vw[i]);
                        ui.add(egui::Label::new(egui::RichText::new(value).font(font.clone()).color(t.text)).extend());
                    });
                }
            },
        );
    });
}

/// The engine's live stroke of a tool: the Brush, Pencil and Eraser paint dabs; the Clone Stamp,
/// Healing Brush and History Brush composite their source.
enum EngineStroke {
    Brush(Box<photocraft_engine::brush_cmds::LiveStroke>),
    Retouch(photocraft_engine::retouch_cmds::LiveRetouch),
    Dab(photocraft_engine::retouch_cmds::LiveDab),
}

impl EngineStroke {
    /// The brush engine's strokes can change while the pointer is held still (see
    /// `LiveStroke::wants_time`); the retouch strokes don't.
    fn wants_time(&self) -> bool {
        match self {
            EngineStroke::Brush(l) => l.wants_time(),
            EngineStroke::Retouch(_) | EngineStroke::Dab(_) => false,
        }
    }

    fn doc(&self) -> &std::sync::Arc<photocraft_doc::Document> {
        match self {
            Self::Brush(s) => &s.doc,
            Self::Retouch(s) => &s.doc,
            Self::Dab(s) => &s.doc,
        }
    }
    fn bounds(&self) -> DRect {
        match self {
            Self::Brush(s) => s.bounds(),
            Self::Retouch(s) => s.bounds(),
            Self::Dab(s) => s.bounds(),
        }
    }
    fn push(&mut self, pts: &[photocraft_engine::paint::StrokePoint]) -> photocraft_engine::Result<DRect> {
        match self {
            Self::Brush(s) => s.push(pts),
            Self::Retouch(s) => s.push(pts),
            Self::Dab(s) => s.push(pts),
        }
    }
    /// The jitter seed the commit must reuse (the Clone Stamp's comes from the session brush).
    fn seed(&self) -> Option<u64> {
        match self {
            Self::Brush(s) => Some(s.seed),
            Self::Retouch(_) => None,
            Self::Dab(_) => None,
        }
    }
}

/// A stroke shown while it is drawn: the engine renders the real result onto a copy of the
/// document, and the canvas redraws only what each step changed.
pub(crate) struct LiveStroke {
    stroke: EngineStroke,
    doc: photocraft_doc::DocId,
    revision: u64,
    /// Preview key of the stroke; step `n` displays as `key + n`.
    key: u64,
    /// Damage of each step (step 0 = the document before the stroke).
    damage: Vec<DRect>,
    /// Drag points rendered so far.
    fed: usize,
}

impl LiveStroke {
    fn display_key(&self) -> u64 {
        self.key + self.damage.len() as u64
    }

    /// What changed since the canvas showed preview `key` (0 = the document before the stroke).
    fn since(&self, key: u64) -> Option<DRect> {
        let seen = if key == 0 { 0 } else { usize::try_from(key.checked_sub(self.key)?).ok()? };
        Some(self.damage.get(seen..)?.iter().fold(DRect::EMPTY, |a, r| a.union(r)))
    }
}

/// A live stroke whose pixels its commit took over (`"fromLive"`): document `doc` at `revision`
/// holds exactly the live stroke's last step, so a cache that showed step `n` of it only misses
/// the steps after `n` (step 0 = the document before the stroke: everything).
pub(crate) struct CommittedStroke {
    pub(crate) doc: photocraft_doc::DocId,
    pub(crate) revision: u64,
    key: u64,
    damage: Vec<DRect>,
}

impl CommittedStroke {
    #[cfg(test)]
    pub(crate) fn for_test(key: u64, damage: Vec<DRect>) -> Self {
        Self { doc: photocraft_doc::DocId(0), revision: 0, key, damage }
    }

    /// What a cache that showed raw preview `key` of this stroke at the revision before misses.
    pub(crate) fn since(&self, key: u64) -> Option<DRect> {
        let seen = if key == 0 { 0 } else { usize::try_from(key.checked_sub(self.key)?).ok()? };
        Some(self.damage.get(seen..)?.iter().fold(DRect::EMPTY, |a, r| a.union(r)))
    }
}

/// The live stroke on document `idx`, while it is current.
fn live_stroke(app: &PhotocraftApp, idx: usize) -> Option<&LiveStroke> {
    let st = app.session.documents().get(idx)?;
    app.live_stroke.as_ref().filter(|l| app.drag.is_some() && l.doc == st.doc.id && l.revision == st.revision)
}

/// `paint.stroke` params for a Brush/Eraser drag (shared by the live preview and the commit). The
/// stroke smoothing is the session brush's (the options bar's Smoothing %).
fn stroke_params(app: &PhotocraftApp, d: &Drag, points: &[Vec<f64>]) -> serde_json::Value {
    let tool = d.tool;
    let mut p = json!({ "points": points, "erase": d.erase, "zoom": app.point_zoom(), "target": paint_target(app) });
    if d.clear {
        p["mode"] = json!("clear");
    }
    if tool == Tool::Pencil {
        p["autoErase"] = json!(app.ui.tool_options.pencil_auto_erase);
    }
    // Block mode: the engine sizes the square from the stroke's zoom (16 / zoom px, #2770).
    if crate::eraser_ui::block_mode(app, tool) {
        p["block"] = json!(true);
    }
    p
}

/// Tools whose strokes the engine renders while they are drawn (`LiveStroke`).
pub(crate) fn strokes_live(tool: Tool) -> bool {
    matches!(
        tool,
        Tool::Brush
            | Tool::Pencil
            | Tool::Eraser
            | Tool::ColorReplacement
            | Tool::Blur
            | Tool::Sharpen
            | Tool::Smudge
            | Tool::Dodge
            | Tool::Burn
            | Tool::Sponge
    ) || live_retouch_command(tool).is_some()
}

/// The retouching tools drawn live (`LiveRetouch`), by the command their live stroke previews.
/// The Healing Brush shows its texture while drawing; the blend comes on release.
fn live_retouch_command(tool: Tool) -> Option<&'static str> {
    match tool {
        Tool::CloneStamp => Some("paint.cloneStamp"),
        Tool::Healing => Some("paint.healingBrush"),
        Tool::HistoryBrush => Some("paint.historyBrush"),
        _ => None,
    }
}

/// The command a live-stroking tool commits: the Pencil's (and the Pencil- or Block-mode
/// Eraser's) `paint.pencil`, else `paint.stroke`.
pub(crate) fn stroke_command(app: &PhotocraftApp, tool: Tool) -> &'static str {
    if tool == Tool::Pencil || crate::eraser_ui::pencil_mode(app, tool) || crate::eraser_ui::block_mode(app, tool) { "paint.pencil" } else { "paint.stroke" }
}

/// Windows' crosshair cursor inverts the pixels under it, so over mid-grey (the pasteboard, many
/// photos) it vanishes (#737), and the Linux themes' and macOS's differ. Use the same black-and-white
/// bitmap on every desktop platform, carried by the OS so its movement does not wait for the canvas
/// to render (#1343).
fn visible_crosshair(icon: egui::CursorIcon, painter: &egui::Painter, p: Pos2, draw: bool) -> egui::CursorIcon {
    if !draw || icon != egui::CursorIcon::Crosshair {
        return icon;
    }
    crate::tool_cursor::crosshair(painter, p, 8.0, 2.0)
}

/// The Pencil's cursor at `doc` (document pixels): the whole-pixel square its dab fills
/// (`paint::grid_square`), in screen points with its edges on physical pixels (`ppp` = pixels
/// per point), so it lines up with the pixel grid at any zoom.
pub(crate) fn pencil_cursor_rect(xf: &ViewXform, doc: [f64; 2], size: f32, ppp: f32) -> Rect {
    let [x0, y0, x1, y1] = photocraft_engine::paint::grid_square(doc[0], doc[1], size);
    let r = Rect::from_two_pos(xf.to_screen(x0 as f32, y0 as f32), xf.to_screen(x1 as f32, y1 as f32));
    let ppp = if ppp.is_finite() && ppp > 0.0 { ppp } else { 1.0 };
    let snap = |v: f32| (v * ppp).round() / ppp;
    Rect::from_min_max(pos2(snap(r.min.x), snap(r.min.y)), pos2(snap(r.max.x), snap(r.max.y)))
}

/// Whether an aliased (Pencil) dab of `size` pixels is round rather than its whole
/// [`pencil_cursor_rect`] square. The dab fills the pixels whose centres lie within its radius of
/// the grid centre (`paint::grid_center`), so up to 3 px the corner pixels are in and the dab is
/// a square; from 4 px on they fall outside and it is a disc (#2662).
pub(crate) fn pencil_tip_is_round(size: f32) -> bool {
    let n = if size.is_finite() { size.round().clamp(1.0, 100_000.0) } else { 1.0 };
    let h = n / 2.0;
    (h - 0.5) * std::f32::consts::SQRT_2 > h
}

/// Whether a brush-tip circle draws its centre mark.
/// Clone Stamp and Healing Brush stay an empty circle until Option is held, the source-point
/// cursor. Other brushes follow Preferences › Cursors › Show Crosshair in Brush Tip, as in
/// Photoshop; the Background Eraser always shows its sampling hotspot.
fn brush_tip_centre(tool: Tool, option: bool, show_crosshair: bool) -> bool {
    if tool == Tool::QuickSelection {
        return false;
    }
    if matches!(tool, Tool::CloneStamp | Tool::Healing) {
        return option || show_crosshair;
    }
    show_crosshair || tool == Tool::BackgroundEraser
}

/// The display colour of an RGB triple (0..1 per channel) as an egui colour.
fn display_color(c: [f32; 3]) -> Color32 {
    let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    Color32::from_rgb(b(c[0]), b(c[1]), b(c[2]))
}

/// The Eyedropper ring's two colours at document point (x, y): the colour under the pointer
/// (what a click would pick) and the current foreground, or `None` where there is no colour
/// to sample (#213).
pub(crate) fn eyedropper_ring_colors(app: &mut PhotocraftApp, x: f64, y: f64) -> Option<(Color32, Color32)> {
    let new = eyedropper_color(app, x, y)?;
    let fg = app.session.tools.foreground;
    Some((display_color(new), display_color([fg[0], fg[1], fg[2]])))
}

/// Draw the Eyedropper's comparison ring at `p`: the sampled colour on the upper arc, the
/// current one below, both over a thin dark outline so they read on any image (#213). Like
/// Photoshop it shows only while the mouse button is held (`held`): hovering shows the plain
/// crosshair and samples nothing. `None` when not held, there is nothing to sample, or the
/// Precise-cursor preference wants the plain crosshair.
fn eyedropper_ring(app: &mut PhotocraftApp, painter: &egui::Painter, xf: &ViewXform, p: Pos2, held: bool) -> Option<egui::CursorIcon> {
    // The options bar's Show Sampling Ring turns it off (#1649).
    if !held || !app.ui.tool_options.eyedropper_ring || app.session.prefs().cursors.other == photocraft_engine::prefs::OtherCursor::Precise {
        return None;
    }
    let [x, y] = xf.to_doc(p);
    let (new, current) = eyedropper_ring_colors(app, x, y)?;
    // Two arcs with a small gap at 3 and 9 o'clock. Screen y grows down, so PI..TAU is the
    // upper half (the new colour) and 0..PI the lower (the current one).
    use std::f32::consts::{PI, TAU};
    let (r, w, gap) = (11.0_f32, 4.0_f32, 0.16_f32);
    let arc = |a0: f32, a1: f32| -> Vec<Pos2> {
        (0..=24)
            .map(|i| {
                let a = a0 + (a1 - a0) * i as f32 / 24.0;
                p + vec2(a.cos(), a.sin()) * r
            })
            .collect()
    };
    let outline = crate::theme::Tokens::get(painter.ctx()).shadow;
    for (a0, a1, colour) in [(PI + gap, TAU - gap, new), (gap, PI - gap, current)] {
        let points = arc(a0, a1);
        painter.add(egui::Shape::line(points.clone(), Stroke::new(w + 1.5, outline)));
        painter.add(egui::Shape::line(points, Stroke::new(w, colour)));
    }
    Some(egui::CursorIcon::None)
}

fn begin_live_stroke(app: &PhotocraftApp) -> Option<LiveStroke> {
    static STROKES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let st = app.session.active()?;
    let d = app.drag.as_ref()?;
    let stroke = if d.tool == Tool::ColorReplacement {
        // The params `retouch_ui::finish_stroke` commits.
        let mut p = crate::retouch_ui::color_replacement_params(app);
        p["points"] = json!(d.points);
        EngineStroke::Dab(photocraft_engine::brush_cmds::live_color_replacement(&app.session, &p).ok()?)
    } else if let Some((cmd, mut p)) = crate::retouch_ui::dab_params(app, d.tool) {
        // The params `retouch_ui::finish_stroke` commits; Sample All Layers isn't previewed.
        p["points"] = json!(d.points);
        p["target"] = paint_target(app);
        EngineStroke::Dab(photocraft_engine::retouch_cmds::LiveDab::begin(&app.session, cmd, &p).ok()?)
    } else if let Some(cmd) = live_retouch_command(d.tool) {
        // The params `retouch_ui::finish_stroke` commits.
        let mut p = if d.tool == Tool::HistoryBrush { json!({}) } else { crate::retouch_ui::clone_params(app)? };
        p["points"] = json!(d.points);
        p["target"] = paint_target(app);
        EngineStroke::Retouch(photocraft_engine::retouch_cmds::LiveRetouch::begin(&app.session, cmd, &p).ok()?)
    } else {
        let p = stroke_params(app, d, &app.stylus.stroke_points(&d.points));
        EngineStroke::Brush(Box::new(photocraft_engine::brush_cmds::LiveStroke::begin_with(&app.session, stroke_command(app, d.tool), &p).ok()?))
    };
    let n = STROKES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) & 0xff_ffff;
    let damage = vec![stroke.bounds()];
    Some(LiveStroke { stroke, doc: st.doc.id, revision: st.revision, key: (1 << 44) | (n << 20), damage, fed: d.points.len() })
}

/// Render the drag points the live stroke hasn't seen yet, with the pen pressure, tilt and
/// rotation the commit's `paint.stroke` gets for them.
fn feed_live_stroke(app: &mut PhotocraftApp) {
    // A batched move replays several samples in one frame and calls this once at the end, so the
    // per-sample calls from `tool_event` are skipped (see `canvas_view`).
    if app.defer_live_stroke {
        return;
    }
    let (Some(l), Some(d)) = (app.live_stroke.as_mut(), app.drag.as_ref()) else { return };
    let pose = &app.stylus.stroke;
    // The same times `stroke_points` gives the commit (from the drag's first point).
    let t0 = app.stylus.point_ms(0);
    let pts: Vec<_> = (l.fed..d.points.len())
        .filter_map(|i| {
            let p = d.points.get(i)?;
            let t = pose.get(i).or(pose.last()).copied().unwrap_or_default();
            let mut sp = photocraft_engine::paint::StrokePoint::new(p[0], p[1], p[2] as f32);
            (sp.tilt_x, sp.tilt_y, sp.rotation) = (t[0], t[1], t[2]);
            sp.time = app.stylus.point_ms(i) - t0;
            Some(sp)
        })
        .collect();
    if pts.is_empty() {
        return;
    }
    l.fed = d.points.len();
    let started = crate::gpu_canvas::now_ms();
    match l.stroke.push(&pts) {
        Ok(r) => l.damage.push(r),
        Err(_) => app.live_stroke = None,
    }
    app.stroke_trace.stroke(crate::gpu_canvas::now_ms() - started);
}

/// The points the live retouching stroke has rendered.
#[cfg(test)]
pub(crate) fn live_retouch_points(app: &PhotocraftApp) -> Option<Vec<photocraft_engine::paint::StrokePoint>> {
    match &app.live_stroke.as_ref()?.stroke {
        EngineStroke::Retouch(l) => Some(l.points().to_vec()),
        _ => None,
    }
}

/// Offer the Clone Stamp's or History Brush's live stroke, with the drag points it hasn't rendered
/// yet, to the command `retouch_ui::finish_stroke` commits, which then takes its pixels.
fn offer_live_retouch(app: &mut PhotocraftApp, d: &Drag) {
    if !matches!(d.tool, Tool::CloneStamp | Tool::HistoryBrush) {
        return;
    }
    let Some(LiveStroke { stroke: EngineStroke::Retouch(mut live), fed, .. }) = app.live_stroke.take() else { return };
    let rest: Vec<_> = d.points.get(fed..).unwrap_or_default().iter().map(|q| photocraft_engine::paint::StrokePoint::new(q[0], q[1], q[2] as f32)).collect();
    if rest.is_empty() || live.push(&rest).is_ok() {
        let _ = app.session.offer_live_retouch(live);
    }
}

/// A pen tap (a drag still at its one press point when released) paints its dab at the highest
/// pressure the pen reached during the contact, not just the first sample's (#1798). A mouse is
/// unchanged at full pressure.
fn tap_at_peak_pressure(app: &mut PhotocraftApp) {
    if app.stylus.sample().is_none() {
        return;
    }
    let peak = f64::from(app.stylus.tap_pressure());
    if let Some(p) = app.drag.as_mut().filter(|d| d.points.len() == 1).and_then(|d| d.points.first_mut()) {
        p[2] = p[2].max(peak);
    }
}

/// While a stroke is held still, its point repeats this often (in ms) so time-based brush
/// features (airbrush build-up, smoothing catch-up) keep going, as when the pen moves.
const HOLD_REPEAT_MS: f64 = 16.0;

/// Is the live stroke one that changes while the pointer is held still (airbrush Build-up, or
/// smoothing catching up)? Only then does a held pointer repeat its point and keep repainting.
fn stroke_wants_time(app: &PhotocraftApp) -> bool {
    app.live_stroke.as_ref().is_some_and(|l| l.stroke.wants_time())
}

/// Time of sample `k` of the `n` samples a frame delivered, spread evenly after `base` (the
/// previous point) up to `now` (this frame): egui's pointer events carry no timestamps.
pub(crate) fn spread_ms(base: f64, now: f64, k: usize, n: usize) -> f64 {
    // Not after `base` (or not comparable, NaN): every sample takes the frame time.
    if n == 0 || now.partial_cmp(&base) != Some(std::cmp::Ordering::Greater) {
        return now;
    }
    base + (now - base) * (k + 1) as f64 / n as f64
}

/// Tools whose gesture follows a freehand path (a polyline of the input points), so every pointer
/// sample the OS delivered improves the result. Other tools are driven by the pointer's latest
/// position (endpoints, anchors, guides), so feeding them a whole frame's moves only repeats work.
pub(crate) fn freehand_tool(tool: Tool) -> bool {
    matches!(
        tool,
        Tool::Brush
            | Tool::Pencil
            | Tool::MixerBrush
            | Tool::ColorReplacement
            | Tool::Eraser
            | Tool::BackgroundEraser
            | Tool::HistoryBrush
            | Tool::Remove
            | Tool::SpotHealing
            | Tool::Healing
            | Tool::CloneStamp
            | Tool::PatternStamp
            | Tool::Blur
            | Tool::Sharpen
            | Tool::Smudge
            | Tool::Dodge
            | Tool::Burn
            | Tool::Sponge
            | Tool::Lasso
            | Tool::MagneticLasso
            | Tool::Patch
            | Tool::ContentAwareMove
            | Tool::QuickSelection
            | Tool::FreeformPen
    )
}

/// The pointer moves this frame delivered while `button` was held, in order. `down_at_start` is
/// whether the button was already down when the frame began. A move before a press or after a
/// release in the same frame is dropped, so a gesture never picks up input from outside its own
/// press..release interval.
fn pointer_moves(events: &[egui::Event], button: egui::PointerButton, down_at_start: bool) -> Vec<Pos2> {
    let mut down = down_at_start;
    let mut out = Vec::new();
    for e in events {
        match e {
            egui::Event::PointerButton { button: b, pressed, .. } if *b == button => down = *pressed,
            egui::Event::PointerMoved(p) if down => out.push(*p),
            _ => {}
        }
    }
    out
}

/// Modifiers attached to the pointer event itself. Some input sources provide these without a
/// separate `ModifiersChanged` event, so use them for click gestures before falling back to the
/// frame-wide modifier state.
fn pointer_button_modifiers(events: &[egui::Event], button: egui::PointerButton) -> Option<egui::Modifiers> {
    events.iter().rev().find_map(|e| match e {
        egui::Event::PointerButton { button: b, modifiers, .. } if *b == button => Some(*modifiers),
        _ => None,
    })
}

/// Abstract tool event, produced by the mouse or by automation (`ui.pointer`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ToolEvent {
    Down { x: f64, y: f64, pressure: f32 },
    Move { x: f64, y: f64, pressure: f32 },
    Up { x: f64, y: f64 },
}

/// View › Show › Layer Edges: the colour of the active layer's outline.
pub(crate) const LAYER_EDGES: Color32 = Color32::from_rgb(0x2d, 0x8c, 0xeb);

/// Document ↔ screen mapping for a canvas rect and a view.
#[derive(Clone, Copy, Debug)]
pub struct ViewXform {
    pub rect: Rect,
    pub zoom: f32,
    pub center: [f32; 2],
    /// View › Flip Horizontal: the view is mirrored left-right (the document is not).
    pub flip: bool,
    /// Camera rotation in degrees around [`Self::center`]. Clockwise (Y-down) is positive; applied
    /// before flip.
    pub rotation: f32,
    /// View › Pixel Aspect Ratio Correction: a document pixel is drawn this many times wider than
    /// tall (1 = square). Applied in document space, before rotation, so a rotated view still
    /// shows the stretched image turned as a whole.
    pub aspect: f32,
}

impl ViewXform {
    /// The main canvas's mapping for the active document as last laid out (inside the rulers).
    pub fn active(app: &PhotocraftApp) -> Option<Self> {
        let v = app.ui.views.get(app.session.active_index()?)?;
        Some(Self {
            rect: crate::rulers::content_rect(app, app.last_canvas_rect),
            zoom: v.zoom / app.canvas_ppp(),
            center: v.center,
            flip: app.ui.view.flip_horizontal,
            rotation: v.rotation,
            aspect: app.ui.view.display_aspect(),
        })
    }

    fn sincos(self) -> (f32, f32) {
        if self.rotation.abs() < 1e-8 {
            (1.0, 0.0)
        } else {
            let r = self.rotation.to_radians();
            (r.cos(), r.sin())
        }
    }

    /// Document pixel width over height on screen: [`Self::aspect`], or 1 when it is unusable.
    fn sx(self) -> f32 {
        if self.aspect.is_finite() && self.aspect > 0.0 { self.aspect } else { 1.0 }
    }

    /// Screen points per document pixel along the document's x axis.
    pub fn zoom_x(&self) -> f32 {
        self.zoom * self.sx()
    }

    /// Unflip, unrotate and unstretch a screen-space vector (no zoom). Hand pan and zoom-about
    /// use this.
    pub fn unmap_vec(self, d: Vec2) -> Vec2 {
        let d = if self.flip { vec2(-d.x, d.y) } else { d };
        let (c, s) = self.sincos();
        vec2((d.x * c + d.y * s) / self.sx(), -d.x * s + d.y * c)
    }

    /// A document-space vector on screen: stretched, rotated, flipped and zoomed as the view
    /// draws it (the inverse of [`Self::unmap_vec`], with the zoom).
    pub fn map_vec(self, d: Vec2) -> Vec2 {
        let (c, s) = self.sincos();
        let dx = d.x * self.sx();
        let v = vec2(dx * c - d.y * s, dx * s + d.y * c) * self.zoom;
        if self.flip { vec2(-v.x, v.y) } else { v }
    }

    pub fn to_screen(&self, x: f32, y: f32) -> Pos2 {
        let (c, s) = self.sincos();
        let dx = (x - self.center[0]) * self.sx();
        let dy = y - self.center[1];
        let rx = dx * c - dy * s;
        let ry = dx * s + dy * c;
        let sx = if self.flip { -1.0 } else { 1.0 };
        self.rect.center() + vec2(rx * self.zoom * sx, ry * self.zoom)
    }
    pub fn to_doc(&self, p: Pos2) -> [f64; 2] {
        let d = (p - self.rect.center()) / self.zoom;
        let u = self.unmap_vec(d);
        [(u.x + self.center[0]) as f64, (u.y + self.center[1]) as f64]
    }
    /// On-screen bounds of the image, including camera rotation and pixel aspect correction.
    pub fn image_bounds(self, size: [u32; 2]) -> Rect {
        let (c, s) = self.sincos();
        let w = size[0] as f32 * self.sx() * self.zoom;
        let h = size[1] as f32 * self.zoom;
        let center = self.to_screen(size[0] as f32 / 2.0, size[1] as f32 / 2.0);
        Rect::from_center_size(center, vec2(c.abs() * w + s.abs() * h, s.abs() * w + c.abs() * h))
    }
    pub fn doc_rect(&self, r: DRect) -> Rect {
        let pts = [
            self.to_screen(r.x0 as f32, r.y0 as f32),
            self.to_screen(r.x1 as f32, r.y0 as f32),
            self.to_screen(r.x1 as f32, r.y1 as f32),
            self.to_screen(r.x0 as f32, r.y1 as f32),
        ];
        let mut min = pts[0];
        let mut max = pts[0];
        for p in pts {
            min = min.min(p);
            max = max.max(p);
        }
        Rect::from_min_max(min, max)
    }
}

/// Document point under a screen position on the main canvas, or `None` when the position is
/// outside it: the rulers are not part of the canvas, so drops on the ruler strip are not
/// canvas drops. Preset-panel drops and drag-and-drop placement share this so both land exactly
/// where the canvas draws.
pub fn doc_point_at(app: &PhotocraftApp, pos: Pos2) -> Option<[f64; 2]> {
    let xf = ViewXform::active(app)?;
    xf.rect.contains(pos).then(|| xf.to_doc(pos))
}

/// Fit the document into `area` (egui points) for a display with `ppp` physical pixels per
/// point; `View::zoom` is stored in physical pixels per document pixel. `aspect` is the
/// displayed pixel aspect ratio ([`ViewXform::aspect`]): the image is that much wider on screen.
pub fn fit_view(view: &mut View, doc: &Document, area: Vec2, ppp: f32, aspect: f32) {
    let (w, h) = (doc.size.width as f32, doc.size.height as f32);
    let shown_w = w * if aspect.is_finite() && aspect > 0.0 { aspect } else { 1.0 };
    // Fit never magnifies past 100%: one image pixel per *physical* pixel, so the cap applies
    // after the points → device pixels conversion (a 150% display would otherwise open at 150%).
    let fit = ((area.x - 40.0) / shown_w).min((area.y - 40.0) / h) * ppp;
    let zoom = crate::zoom_levels::clamp(fit.min(1.0), [doc.size.width, doc.size.height]);
    view.zoom = zoom;
    view.center = [w / 2.0, h / 2.0];
    view.fit_pending = false;
    view.fill_pending = false;
}

/// Centre the document and zoom until it fills the canvas. One axis can extend beyond the
/// viewport, matching the Hand tool's Fill Screen action.
pub fn fill_view(view: &mut View, doc: &Document, area: Vec2, ppp: f32, aspect: f32) {
    let (w, h) = (doc.size.width as f32, doc.size.height as f32);
    let shown_w = w * if aspect.is_finite() && aspect > 0.0 { aspect } else { 1.0 };
    let points = (area.x / shown_w).max(area.y / h);
    let zoom = crate::zoom_levels::clamp(points * ppp, [doc.size.width, doc.size.height]);
    view.zoom = zoom;
    view.center = [w / 2.0, h / 2.0];
    view.fit_pending = false;
    view.fill_pending = false;
}

fn checker(app: &mut PhotocraftApp, ctx: &egui::Context) -> egui::TextureId {
    // Preferences › Transparency & Gamut colours (the texture is rebuilt when they change).
    let [ca, cb] = app.session.prefs().transparency_and_gamut.colors();
    if app.prefs_rt.checker_key != Some([ca, cb]) {
        app.prefs_rt.checker_key = Some([ca, cb]);
        app.checker = None;
    }
    app.checker
        .get_or_insert_with(|| {
            let (a, b) = (Color32::from_rgb(ca[0], ca[1], ca[2]), Color32::from_rgb(cb[0], cb[1], cb[2]));
            let img = egui::ColorImage::new([2, 2], vec![a, b, b, a]);
            let opts = TextureOptions {
                magnification: egui::TextureFilter::Nearest,
                minification: egui::TextureFilter::Nearest,
                wrap_mode: egui::TextureWrapMode::Repeat,
                mipmap_mode: None,
            };
            ctx.load_texture("checker", img, opts)
        })
        .id()
}

fn buffer_to_image(buf: &photocraft_compose::Buffer) -> egui::ColorImage {
    // `buf.to_rgba8()` then `color_image`, fused: one parallel pass, no intermediate image.
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    let mut pixels = vec![egui::Color32::TRANSPARENT; buf.px.len()];
    let convert = |out: &mut [egui::Color32], src: &[[f32; 4]]| {
        for (o, p) in out.iter_mut().zip(src) {
            *o = egui::Color32::from_rgba_unmultiplied(q(p[0]), q(p[1]), q(p[2]), q(p[3]));
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        pixels.par_chunks_mut(16 * 1024).zip(buf.px.par_chunks(16 * 1024)).for_each(|(o, s)| convert(o, s));
    }
    #[cfg(target_arch = "wasm32")]
    convert(&mut pixels, &buf.px);
    egui::ColorImage::new([buf.rect.width() as usize, buf.rect.height() as usize], pixels)
}

/// `img` as an egui image: `ColorImage::from_rgba_unmultiplied`, in parallel off wasm. egui's own
/// conversion runs on one thread and took 37 ms of a 24 MP canvas or preview update, three times
/// the quantize before it.
fn color_image(img: &photocraft_raster::Rgba8Image) -> egui::ColorImage {
    let size = [img.width as usize, img.height as usize];
    let src = img.pixels.as_chunks::<4>().0;
    let mut pixels = vec![egui::Color32::TRANSPARENT; src.len()];
    let convert = |out: &mut [egui::Color32], src: &[[u8; 4]]| {
        for (o, &[r, g, b, a]) in out.iter_mut().zip(src) {
            *o = egui::Color32::from_rgba_unmultiplied(r, g, b, a);
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        pixels.par_chunks_mut(16 * 1024).zip(src.par_chunks(16 * 1024)).for_each(|(o, s)| convert(o, s));
    }
    #[cfg(target_arch = "wasm32")]
    convert(&mut pixels, src);
    egui::ColorImage::new(size, pixels)
}

/// A composite as monitor pixels: document profile → monitor profile (nothing to do in the
/// common sRGB-on-sRGB case).
pub(crate) fn display_image(display: Option<&photocraft_engine::display_color::CanvasDisplay>, buf: &photocraft_compose::Buffer) -> egui::ColorImage {
    match display {
        Some(d) if !d.is_identity() => color_image(&d.to_rgba8(buf)),
        _ => buffer_to_image(buf),
    }
}

/// The canvas display of `doc` on `display` (`None`: the main window's) and a key that changes
/// with it (folded into the canvas caches' preview keys, so a monitor or profile change
/// re-renders).
pub(crate) fn canvas_display(
    app: &PhotocraftApp,
    doc: &Document,
    display: Option<u32>,
) -> (Option<std::sync::Arc<photocraft_engine::display_color::CanvasDisplay>>, u64) {
    match app.session.color.canvas_display_for(doc, display.or(app.session.color.main_display)) {
        Ok(d) => {
            let k = d.key;
            (Some(d), k)
        }
        Err(_) => (None, 0),
    }
}

/// [`canvas_display`]'s key for the GPU canvas texture: what the texture stores, the same on
/// every display (the display LUT holds the monitor profile).
fn texture_key(display: Option<&photocraft_engine::display_color::CanvasDisplay>) -> u64 {
    display.map_or(0, |d| d.texture_key)
}

/// `app.canvases` key: CPU canvas textures are per document and display (they hold monitor
/// values); the GPU canvas state uses [`GPU_OUTPUT`].
pub(crate) fn cache_key(doc: photocraft_doc::DocId, display: Option<u32>) -> (photocraft_doc::DocId, u32) {
    (doc, display.unwrap_or(0))
}

/// `app.canvases` output of the GPU canvas state (one texture per document for all displays).
pub(crate) const GPU_OUTPUT: u32 = u32::MAX;

/// The document to render: the committed one, or a clone with the live adjustment preview applied.
pub(crate) fn display_doc(app: &mut PhotocraftApp, idx: usize) -> (std::sync::Arc<Document>, u64) {
    if let Some(shown) = crate::type_panels_ui::color_picker::display_doc(app, idx) {
        return shown;
    }
    if let Some(shown) = crate::solid_fill_ui::display_doc(app, idx) {
        return shown;
    }
    if let Some(shown) = crate::pattern_fill_ui::display_doc(app, idx) {
        return shown;
    }
    if let Some(shown) = crate::shape_fill_ui::display_doc(app, idx) {
        return shown;
    }
    if let Some(shown) = crate::type_transform::display_doc(app, idx) {
        return shown;
    }
    if let Some(shown) = crate::type_font_preview::display_doc(app, idx) {
        return shown;
    }
    // Puppet / Perspective Warp previews hide the layer they draw on a mesh.
    if let Some(shown) = crate::distort_ui::display_doc(app, idx) {
        return shown;
    }
    // Image › Adjustments dialog: a temporary adjustment layer clipped to the target.
    if let Some(shown) = crate::adjust_preview::display_doc(app, idx) {
        return shown;
    }
    // Gradient tool (live) drag, or a stop dragged in the Properties panel.
    if let Some(shown) = crate::gradient_ui::display_doc(app, idx) {
        return shown;
    }
    // Move tool drag: the moving layers at the pointer.
    if let Some(shown) = crate::move_ui::display_doc(app, idx) {
        return shown;
    }
    // Patch Tool drag: the selection healed from where the pointer is.
    if let Some(shown) = crate::patch_preview::display_doc(app, idx) {
        return shown;
    }
    // Content-Aware Move drag or box: the result, once computed in the background.
    if let Some(shown) = crate::cam_preview::display_doc(app, idx) {
        return shown;
    }
    // A blend mode hovered in the Layers panel.
    if let Some(shown) = crate::blend_preview::display_doc(app, idx) {
        return shown;
    }
    crate::transform_tool::follow_edits(app);
    let st = &app.session.documents()[idx];
    if let Some(l) = live_stroke(app, idx) {
        return (l.stroke.doc().clone(), l.display_key());
    }
    if let (Some(t), Some(pv)) = (&app.ui.transform, &app.transform_preview)
        && app.session.active_index() == Some(idx)
        && st.doc.layer(photocraft_doc::LayerId(t.layer)).is_some()
    {
        return (pv.doc.clone(), (1 << 40) + pv.session);
    }
    // Layer Style dialog: show its effects live (Cancel just drops the preview).
    let style = app.ui.dialogs.iter().find(|d| d.kind == crate::state::DialogKind::LayerStyle);
    // The Preview checkbox off shows the document as it was when the dialog opened;
    // edits still land in the dialog state and OK applies them whatever it says.
    let preview_on = style.and_then(|d| d.fields.get("preview")).and_then(serde_json::Value::as_bool).unwrap_or(true);
    if style.is_none() || !preview_on {
        app.style_preview = None;
    }
    if let Some(d) = style.filter(|_| preview_on)
        && app.session.active_index() == Some(idx)
    {
        let fields = crate::layer_style::color_picker::preview_fields(app, d.id, &d.fields);
        let key = (crate::layer_style::preview_hash(&fields) ^ st.revision.wrapping_mul(0x9e37_79b9_7f4a_7c15)) | 1 << 63;
        if app.style_preview.as_ref().map(|p| p.0) != Some(key) {
            let shown = crate::layer_style::preview_document(&st.doc, &app.session.patterns, &fields).map(std::sync::Arc::new);
            if let Err(error) = &shown {
                log::warn!("Layer Style preview: {error}");
            }
            app.style_preview = Some((key, shown));
        }
        if let Some((_, Ok(doc))) = &app.style_preview {
            return (doc.clone(), key);
        }
    }
    let snapshot = st.doc.clone();
    if let Some((layer, params)) = app.live_adjust.clone()
        && let Some(l) = snapshot.layer(layer)
        && let LayerContent::Adjustment(a) = &l.content
    {
        let key = 1 + params.to_string().bytes().fold(0u64, |h, b| h.wrapping_mul(31).wrapping_add(b as u64));
        let shown = memo_display(app, &snapshot, key, || {
            let kind = photocraft_engine::commands::adjustment_kind(a);
            let preview = photocraft_engine::adjust_params::from_params(kind, &params, Some(a), snapshot.mode).unwrap_or_else(|_| a.clone());
            let mut doc = (*snapshot).clone();
            if let Some(lm) = doc.layer_mut(layer) {
                lm.content = LayerContent::Adjustment(preview);
            }
            Some(doc)
        });
        if let Some(doc) = shown {
            return (doc, key);
        }
    }
    // Duotone documents display through their inks.
    if let Some(shown) = memo_display(app, &snapshot, 1 << 41, || photocraft_engine::mode_cmds::display_document(&snapshot)) {
        return (shown, 1 << 41);
    }
    (snapshot, 0)
}

/// `display_doc`'s live-adjust and Duotone documents, built once per document snapshot and key
/// rather than on each of its 3–4 calls a frame (#4527): each build clones the whole document.
fn memo_display(
    app: &mut PhotocraftApp,
    snapshot: &std::sync::Arc<Document>,
    key: u64,
    build: impl FnOnce() -> Option<Document>,
) -> Option<std::sync::Arc<Document>> {
    let id = snapshot.id;
    if let Some((doc, k, shown)) = app.display_memo.get(&id)
        && *k == key
        && doc.ptr_eq(&std::sync::Arc::downgrade(snapshot))
    {
        return Some(shown.clone());
    }
    let Some(shown) = build().map(std::sync::Arc::new) else {
        app.display_memo.remove(&id);
        return None;
    };
    // Entries of closed documents go when another is stored.
    app.display_memo.retain(|_, (doc, _, _)| doc.strong_count() > 0);
    app.display_memo.insert(id, (std::sync::Arc::downgrade(snapshot), key, shown.clone()));
    Some(shown)
}

/// Longest side of the Navigator panel's image (about twice the panel's width, for HiDPI).
pub const NAVIGATOR_SIDE: u32 = 512;
/// The navigator image of a document: its snapshot, preview key and revision, and the texture.
pub(crate) type NavigatorCache = (std::sync::Weak<Document>, u64, u64, egui::TextureHandle);
/// How long adjustment-dialog settings must stay unchanged before the navigator shows them.
const NAVIGATOR_SETTLE_MS: f64 = 200.0;

/// While a Move or Patch drag or a live paint stroke is under way, the navigator keeps its image
/// of document `idx` and catches up on release: each step of the drag is a new preview, and a
/// thumbnail composite of the whole document per step makes painting lag on large images.
fn navigator_waits_for_drag(app: &PhotocraftApp, idx: usize) -> bool {
    crate::move_ui::showing(app) || crate::patch_preview::showing(app) || crate::cam_preview::showing(app) || live_stroke(app, idx).is_some()
}

/// The Navigator panel's image of document `idx`. With the CPU canvas that is the canvas texture
/// itself; with the GPU canvas a thumbnail cached per revision, so an edit to a huge document
/// doesn't also pay a full-resolution CPU composite for the navigator.
/// The Navigator's preview from the GPU canvas (#4518): the main canvas's texture of the
/// document, drawn at `rect` with its own mip chain, so an edit costs no CPU composite. `None`
/// while the canvas isn't showing the plain document on the GPU (the CPU thumbnail then).
pub fn navigator_gpu_view(app: &PhotocraftApp, idx: usize, rect: Rect, ctx: &egui::Context) -> Option<crate::gpu_canvas::ViewParams> {
    app.gpu.as_ref()?;
    let st = app.session.documents().get(idx)?;
    let params = *app.navigator_gpu.get(&st.doc.id)?;
    let [w, h] = params.doc_size;
    if w == 0 || h == 0 || !rect.is_positive() || [st.doc.size.width, st.doc.size.height] != params.doc_size {
        return None;
    }
    Some(crate::gpu_canvas::ViewParams {
        zoom: rect.width() / w as f32,
        center: [w as f32 / 2.0, h as f32 / 2.0],
        rotation: 0.0,
        shadow: false,
        pixel_grid: false,
        view_key: egui::Id::new(("pc-navigator", ctx.viewport_id(), idx)).value(),
        offset: [0.0, 0.0],
        ..params
    })
}

/// Frames the Navigator waits for the GPU canvas to draw a document it hasn't drawn yet, rather
/// than build a CPU thumbnail it would only use once (a full composite on a large document).
const NAVIGATOR_WAIT_FRAMES: u32 = 3;

/// Whether the Navigator should wait a frame for the GPU canvas (#4518): the GPU canvas is on but
/// hasn't drawn document `idx` yet (the Navigator can draw before the canvas in a frame). After
/// [`NAVIGATOR_WAIT_FRAMES`] it stops waiting, so it never stays empty.
pub fn navigator_waits_for_gpu(app: &PhotocraftApp, idx: usize, ctx: &egui::Context) -> bool {
    if app.gpu.is_none() {
        return false;
    }
    let Some(id) = app.session.documents().get(idx).map(|st| st.doc.id) else { return false };
    if app.canvases.keys().any(|(d, _)| *d == id) {
        return false;
    }
    let key = egui::Id::new(("navigator-waits", id.0));
    let waited = ctx.data(|d| d.get_temp::<u32>(key)).unwrap_or(0);
    if waited >= NAVIGATOR_WAIT_FRAMES {
        return false;
    }
    ctx.data_mut(|d| d.insert_temp(key, waited + 1));
    ctx.request_repaint();
    true
}

pub fn navigator_texture(app: &mut PhotocraftApp, ctx: &egui::Context, idx: usize) -> Option<egui::TextureId> {
    if app.gpu.is_none() {
        return ensure_texture(app, ctx, idx, app.session.color.main_display).map(|(t, _)| t);
    }
    // Keyed by the document snapshot, not its revision: selecting a layer changes no pixels.
    let (snapshot, id, revision, last_damage) =
        app.session.documents().get(idx).map(|st| (std::sync::Arc::downgrade(&st.doc), st.doc.id, st.revision, st.last_damage))?;
    let (doc, preview_key) = display_doc(app, idx);
    let (display, display_key) = canvas_display(app, &doc, None);
    let preview_key = preview_key ^ display_key;
    let cached = app.navigator_textures.get(&id).cloned();
    if let Some((r, p, _, t)) = &cached
        && r.ptr_eq(&snapshot)
        && *p == preview_key
    {
        return Some(t.id());
    }
    if let Some((_, _, _, t)) = &cached
        && navigator_waits_for_drag(app, idx)
    {
        return Some(t.id());
    }
    // While an adjustment dialog's settings are changing, the navigator keeps its image and
    // catches up once they settle (a thumbnail composite per change would cost more than the canvas).
    if let Some((r, _, _, t)) = &cached
        && r.ptr_eq(&snapshot)
        && crate::adjust_preview::settling(app, NAVIGATOR_SETTLE_MS)
    {
        ctx.request_repaint_after(std::time::Duration::from_millis(NAVIGATOR_SETTLE_MS as u64));
        return Some(t.id());
    }
    let t0 = crate::gpu_canvas::now_ms();
    // One edit since the cached image, with the area it changed: update just that part (nothing
    // for a rename, a selection or a new empty layer). A thumbnail composite of the whole
    // document on every edit cost a full-resolution CPU composite on the UI thread.
    let size = photocraft_compose::thumbnail_size(&doc, NAVIGATOR_SIDE);
    if let Some((_, p, rev, t)) = &cached
        && *p == preview_key
        && rev.checked_add(1) == Some(revision)
        && let Some(d) = last_damage
        && t.size() == [size.0 as usize, size.1 as usize]
    {
        let mut t = t.clone();
        if !d.is_empty() {
            let buf = photocraft_compose::render_reduced_damage(&doc, size.0, size.1, d.inflate(effect_reach(&doc.layers)));
            let r = buf.rect;
            if !r.is_empty() {
                t.set_partial([r.x0 as usize, r.y0 as usize], display_image(display.as_deref(), &buf), TextureOptions::LINEAR);
            }
        }
        app.perf.span("navigator", crate::gpu_canvas::now_ms() - t0);
        app.navigator_textures.insert(id, (snapshot, preview_key, revision, t.clone()));
        return Some(t.id());
    }
    let image = display_image(display.as_deref(), &photocraft_compose::thumbnail_buffer(&doc, NAVIGATOR_SIDE));
    let tex = match cached {
        Some((_, _, _, mut t)) => {
            t.set(image, TextureOptions::LINEAR);
            t
        }
        None => ctx.load_texture(format!("navigator-{}", id.0), image, TextureOptions::LINEAR),
    };
    app.perf.span("navigator", crate::gpu_canvas::now_ms() - t0);
    app.navigator_textures.insert(id, (snapshot, preview_key, revision, tex.clone()));
    Some(tex.id())
}

/// The CPU canvas texture's sampling (#4810). Zoomed in, a full-resolution texture shows each
/// document pixel as a crisp square, as the GPU canvas's sharp nearest filter and Photoshop do;
/// zoomed out it is still filtered. A reduced texture (`factor` > 1) stays bilinear: its texels
/// are averages of several pixels, and squares of them would be blocks the image doesn't have.
fn canvas_texture_options(factor: u32) -> TextureOptions {
    if factor > 1 { TextureOptions::LINEAR } else { TextureOptions { magnification: egui::TextureFilter::Nearest, ..TextureOptions::LINEAR } }
}

/// Make sure the canvas texture for document `idx` on `display` is current; returns (texture
/// id, scale).
pub fn ensure_texture(app: &mut PhotocraftApp, ctx: &egui::Context, idx: usize, display: Option<u32>) -> Option<(egui::TextureId, f32)> {
    let output = display;
    let (revision, last_damage, id) = {
        let st = app.session.documents().get(idx)?;
        (st.revision, st.last_damage.map(|r| if r.is_empty() { r } else { r.inflate(effect_reach(&st.doc.layers)) }), st.doc.id)
    };
    let (mut doc, mut preview_key) = display_doc(app, idx);
    // A flipped view draws a GPU machine's canvas through here: an adjustment dialog's preview of
    // a large document (taken by the GPU path) would cost a full-size CPU composite per change.
    if crate::adjust_preview::shown_key(app) == Some(preview_key)
        && let Some(st) = app.session.documents().get(idx)
        && crate::proxy::preview_factor(&st.doc, crate::proxy::reduced_previews(app)) > 1
    {
        (doc, preview_key) = (st.doc.clone(), 0);
    }
    let (display, display_key) = canvas_display(app, &doc, output);
    let seen = app.canvases.get(&cache_key(id, output)).map(|c| (c.tex_revision, c.tex_preview_key));
    let damage = seen.and_then(|seen| damage_since(app, idx, seen, (revision, preview_key), display_key, last_damage));
    let preview_key = preview_key ^ display_key;
    let cache = app.canvases.entry(cache_key(id, output)).or_insert(CanvasCache {
        revision: 0,
        texture: None,
        scale: 1.0,
        preview_key: 0,
        on_gpu: false,
        tex_revision: 0,
        tex_preview_key: 0,
        grid: None,
    });
    if cache.tex_revision != revision || cache.texture.is_none() || cache.tex_preview_key != preview_key {
        let t0 = crate::gpu_canvas::now_ms();
        let longest = doc.size.width.max(doc.size.height);
        let factor = longest.div_ceil(MAX_TEXTURE).max(1);
        let (w, h) = ((doc.size.width / factor).max(1), (doc.size.height / factor).max(1));
        let current = cache.texture.as_mut().filter(|t| t.size() == [w as usize, h as usize]);
        if let (Some(d), Some(t)) = (damage, current) {
            // Only what the edit or stroke touched: the reduced texture's pixels over it, with the
            // values the whole reduction gives them (factor 1 is the plain composite of `d`).
            let buf = photocraft_compose::render_reduced_damage(&doc, w, h, d);
            let r = buf.rect;
            if !r.is_empty() {
                let t1 = crate::gpu_canvas::now_ms();
                t.set_partial([r.x0 as usize, r.y0 as usize], display_image(display.as_deref(), &buf), canvas_texture_options(factor));
                let px = (r.width() as u64 * r.height() as u64).saturating_mul(u64::from(factor).pow(2));
                app.perf.record("rect", px, t1 - t0, crate::gpu_canvas::now_ms() - t1);
            }
        } else {
            // Reduced in bands straight from the compositor: no full-size composite in memory.
            let full = photocraft_compose::render_reduced(&doc, w, h);
            let t1 = crate::gpu_canvas::now_ms();
            let (img, scale) = (display_image(display.as_deref(), &full), 1.0 / factor as f32);
            match cache.texture.as_mut() {
                Some(t) if t.size() == img.size => t.set(img, canvas_texture_options(factor)),
                _ => cache.texture = Some(ctx.load_texture(format!("canvas-{}-{}", id.0, cache_key(id, output).1), img, canvas_texture_options(factor))),
            }
            cache.scale = scale;
            app.perf.record("full", doc.size.width as u64 * doc.size.height as u64, t1 - t0, crate::gpu_canvas::now_ms() - t1);
        }
        cache.tex_revision = revision;
        cache.tex_preview_key = preview_key;
    }
    Some((cache.texture.as_ref()?.id(), cache.scale))
}

/// What changed since a canvas cache showed (`revision`, `preview key`) `seen`, when only a
/// rectangle did: the last edit's damage, or the live stroke's dabs since then. Cached keys have
/// the colour display's key folded in (`^ display_key`); `now`'s is still raw.
pub(crate) fn damage_since(app: &PhotocraftApp, idx: usize, seen: (u64, u64), now: (u64, u64), display_key: u64, last_damage: Option<DRect>) -> Option<DRect> {
    // A brush stroke committed with the pixels its live preview already showed: only the steps
    // this cache hadn't shown yet (usually nothing), not the stroke's whole bounding box.
    if seen.0 + 1 == now.0
        && let Some(st) = app.session.documents().get(idx)
        && let Some(c) = app.committed_stroke.as_ref().filter(|c| c.doc == st.doc.id && c.revision == now.0)
        && let Some(r) = c.since(seen.1 ^ display_key)
    {
        return Some(if r.is_empty() { r } else { r.inflate(effect_reach(&st.doc.layers)) });
    }
    if seen.1 == now.1 ^ display_key && seen.0 + 1 == now.0 {
        return last_damage;
    }
    if seen.0 == now.0
        && let Some(st) = app.session.documents().get(idx)
        && let Some(r) = crate::type_transform::damage(app, st.doc.id, now.0, seen.1 ^ display_key, now.1)
    {
        return Some(if r.is_empty() { r } else { r.inflate(effect_reach(&st.doc.layers)) });
    }
    // Between an adjustment dialog's previews (and the document): the target's area.
    if seen.0 == now.0
        && let Some(st) = app.session.documents().get(idx)
        && let Some(r) = crate::adjust_preview::switch_region(app, st.doc.id, now.0, seen.1 ^ display_key, now.1)
    {
        return Some(r);
    }
    // Between a Move drag's offsets: where the moving layers were and are (Auto-Select's pick
    // at the press changes no pixels).
    if (seen.0 == now.0 || (seen.0 + 1 == now.0 && last_damage.is_some_and(|r| r.is_empty())))
        && let Some(st) = app.session.documents().get(idx)
        && let Some(r) = crate::move_ui::damage(app, st.doc.id, now.0, seen.1 ^ display_key, now.1)
    {
        return Some(if r.is_empty() { r } else { r.inflate(effect_reach(&st.doc.layers)) });
    }
    // Between a Patch drag's previews: the areas they healed.
    if seen.0 == now.0
        && let Some(st) = app.session.documents().get(idx)
        && let Some(r) = crate::patch_preview::damage(app, st.doc.id, now.0, seen.1 ^ display_key, now.1)
    {
        return Some(if r.is_empty() { r } else { r.inflate(effect_reach(&st.doc.layers)) });
    }
    // Between hovered blend modes (and the document): where the layer's blending shows.
    if seen.0 == now.0
        && let Some(st) = app.session.documents().get(idx)
        && let Some(r) = crate::blend_preview::damage(app, st.doc.id, now.0, seen.1 ^ display_key, now.1)
    {
        return Some(if r.is_empty() { r } else { r.inflate(effect_reach(&st.doc.layers)) });
    }
    // Between hovered fonts (and the document): the text layer's old/new ink bounds.
    if seen.0 == now.0
        && let Some(st) = app.session.documents().get(idx)
        && let Some(r) = crate::type_font_preview::damage(app, st.doc.id, now.0, seen.1 ^ display_key, now.1)
    {
        return Some(if r.is_empty() { r } else { r.inflate(effect_reach(&st.doc.layers)) });
    }
    let l = live_stroke(app, idx).filter(|l| seen.0 == now.0 && l.display_key() == now.1)?;
    let r = l.since(seen.1 ^ display_key)?;
    Some(if r.is_empty() { r } else { r.inflate(effect_reach(&l.stroke.doc().layers)) })
}

/// Document `doc`'s canvas caches showed a preview that the edit just committed reproduces
/// (`was_preview` tells its keys): count it as the document itself, so the commit's damage rect
/// refreshes only that area instead of everything.
pub(crate) fn shown_as_document(app: &mut PhotocraftApp, doc: photocraft_doc::DocId, was_preview: impl Fn(u64) -> bool) {
    let Some(d) = app.session.documents().iter().find(|st| st.doc.id == doc).map(|st| st.doc.clone()) else { return };
    if let Some(c) = app.channel_views.get_mut(&doc.0)
        && was_preview(c.preview_key)
    {
        c.preview_key = 0;
    }
    // Every display's cache of the document: the GPU state folds in the texture key, CPU
    // textures their display's key.
    let outputs: Vec<u32> = app.canvases.keys().filter(|k| k.0 == doc).map(|k| k.1).collect();
    for out in outputs {
        let (display, key) = canvas_display(app, &d, (out != 0 && out != GPU_OUTPUT).then_some(out));
        let gpu_key = texture_key(display.as_deref());
        let Some(c) = app.canvases.get_mut(&(doc, out)) else { continue };
        if was_preview(c.preview_key ^ gpu_key) {
            c.preview_key = gpu_key;
        }
        if was_preview(c.tex_preview_key ^ key) {
            c.tex_preview_key = key;
        }
    }
}

/// How far beyond an edit's damage rect the composite can change: layer effects (shadows, glows,
/// strokes, …) on the edited layer and on the groups around it reach that far.
pub(crate) fn effect_reach(layers: &[photocraft_doc::Layer]) -> i32 {
    layers
        .iter()
        .filter(|l| l.visible)
        .map(|l| {
            let own = if photocraft_compose::effects::has_effects(l) { photocraft_compose::effects::margin(l) } else { 0 };
            let inner = match &l.content {
                LayerContent::Group(g) => effect_reach(&g.children),
                _ => 0,
            };
            own + inner
        })
        .max()
        .unwrap_or(0)
}

/// GPU path: make sure document `idx` is current in the GPU canvas. Edits draw their damage;
/// unchanged lower-stack composites can be reused by the compositor inside that region.
/// Returns false if there is no GPU canvas.
fn ensure_gpu(app: &mut PhotocraftApp, ctx: &egui::Context, idx: usize, visible: DRect) -> bool {
    let Some(gpu) = app.gpu.clone() else { return false };
    let Some((revision, last_damage, id)) = app
        .session
        .documents()
        .get(idx)
        .map(|st| (st.revision, st.last_damage.map(|r| if r.is_empty() { r } else { r.inflate(effect_reach(&st.doc.layers)) }), st.doc.id))
    else {
        return false;
    };
    let (doc, raw_key) = display_doc(app, idx);
    let (display, _) = canvas_display(app, &doc, None);
    // The texture is shared by every display: only what it stores counts, not the monitor.
    let display_key = texture_key(display.as_deref());
    let seen = app.canvases.get(&(id, GPU_OUTPUT)).map(|c| (c.revision, c.preview_key));
    let mut damage = seen.and_then(|seen| damage_since(app, idx, seen, (revision, raw_key), display_key, last_damage));
    // To an adjustment dialog's preview: only what the view shows now (the rest as it moves there).
    if damage.is_some()
        && let Some(r) = seen.filter(|s| s.0 == revision).and_then(|s| crate::adjust_preview::switch_region(app, id, revision, s.1 ^ display_key, raw_key))
    {
        damage = Some(crate::adjust_preview::switch_damage(app, raw_key, r, visible));
    }
    let preview_key = raw_key ^ display_key;
    // The Navigator may draw this texture only while it holds the plain document (no live
    // preview, which can be composited only where the view is).
    if raw_key == 0 {
        app.navigator_plain.insert(id);
    } else {
        app.navigator_plain.remove(&id);
        app.navigator_gpu.remove(&id);
    }
    let size = [doc.size.width, doc.size.height];
    // A full CPU redraw running in the background: show the blocks it has finished.
    crate::background_redraw::poll(app, &gpu, id);
    let cache = app.canvases.entry((id, GPU_OUTPUT)).or_insert(CanvasCache {
        revision: 0,
        texture: None,
        scale: 1.0,
        preview_key: 0,
        on_gpu: false,
        tex_revision: 0,
        tex_preview_key: 0,
        grid: None,
    });
    let present = cache.on_gpu && gpu.has(id.0, size);
    if present && cache.revision == revision && cache.preview_key == preview_key {
        // An adjustment preview composited where the view was: catch up where it moved to.
        if let Some(r) = crate::adjust_preview::uncovered(app, id, raw_key, visible) {
            let r = gpu.refresh(id.0, &doc, Some(r), display.as_deref());
            app.perf.record(r.kind, r.px, r.composite_ms, r.upload_ms);
        }
        return true;
    }
    // Display textures for a canvas this large would not fit the GPU: the reduced CPU canvas
    // draws it (checked before the memory budget, which walks the history, every frame).
    if !gpu.fits_canvas(&doc) {
        crate::background_redraw::cancel(app, id);
        return false;
    }
    let partial = present && damage.is_some();
    gpu_budget(app, &gpu, idx, partial, visible);
    let defer = crate::background_redraw::enabled(app);
    let r = gpu.refresh_with(id.0, &doc, if partial { damage } else { None }, display.as_deref(), defer);
    if r.kind == "lost" || r.kind == crate::gpu_canvas::REFRESH_OVER_BUDGET {
        // The device is gone (`gpu_status` then drops the GPU canvas) or the canvas is over the
        // texture budget: draw this frame on the CPU path.
        crate::background_redraw::cancel(app, id);
        return false;
    }
    if r.kind == crate::gpu_canvas::REFRESH_DEFERRED {
        // Composited off the UI thread, the visible part first (#4024).
        let region = if partial { damage.unwrap_or(doc.bounds()) } else { doc.bounds() };
        crate::background_redraw::start(app, ctx, &gpu, doc.clone(), display.clone(), visible, region, r.fallback.clone());
    } else if partial {
        // The edit drew its damage; blocks of a running redraw still to come follow the edit.
        crate::background_redraw::restart(app, ctx, &gpu, doc.clone(), display.clone(), visible);
    } else {
        crate::background_redraw::cancel(app, id);
    }
    if let Some(e) = &r.fallback
        && app.perf.gpu_fallback.as_deref() != Some(e.as_str())
    {
        log::info!("{e}; using the CPU compositor");
    }
    if r.kind != crate::gpu_canvas::REFRESH_DEFERRED {
        app.perf.record(r.kind, r.px, r.composite_ms, r.upload_ms);
    }
    if r.kind == "full" {
        crate::notices::slow_cpu_refresh(app, id.0, r.composite_ms + r.upload_ms, r.fallback.as_deref());
    }
    if r.kind.starts_with("gpu") {
        app.perf.gpu_uploads = r.uploads;
    } else {
        app.perf.gpu_fallback = r.fallback;
    }
    let Some(cache) = app.canvases.get_mut(&(id, GPU_OUTPUT)) else { return true };
    cache.revision = revision;
    cache.preview_key = preview_key;
    cache.on_gpu = true;
    cache.texture = None;
    true
}

/// Point the wgpu compositor at what the view shows, and size its memory budget from Memory
/// Usage, the document's pixels and History, and physical memory: on full refreshes (pixels
/// counted once per structural change, not per brush dab) and whenever the preference changes.
fn gpu_budget(app: &mut PhotocraftApp, gpu: &crate::gpu_canvas::GpuCanvas, idx: usize, partial: bool, visible: DRect) {
    gpu.set_focus(Some(visible));
    let allowance = u64::from(app.session.prefs().performance.memory_usage_mb).saturating_mul(1 << 20);
    if partial && app.perf.gpu_budget_allowance == allowance && gpu.memory_budget().is_some() {
        return;
    }
    let Some(st) = app.session.documents().get(idx) else { return };
    let pixels = st.history.pixel_bytes(&st.doc) as u64;
    let budget = crate::gpu_canvas::memory_budget(allowance, pixels, crate::gpu_canvas::physical_memory());
    gpu.set_memory_budget(budget);
    app.perf.gpu_budget_mb = budget >> 20;
    app.perf.gpu_budget_allowance = allowance;
}

/// Live preview for an open filter dialog: run the filter on the proxy and upload it. With
/// background jobs (the desktop app) it computes on one background worker
/// (`filter_preview_worker`); otherwise (web, tests, `PHOTOCRAFT_INLINE_JOBS`) it keeps the
/// deterministic inline path (#1676), which also lets headless/CPU tests inspect previews
/// without a GPU texture.
pub(crate) fn ensure_filter_preview(app: &mut PhotocraftApp, idx: usize, ctx: &egui::Context) -> Option<(u32, u64, [u32; 2])> {
    if let Some(held) = committed_filter_preview(app, idx) {
        return held;
    }
    // Command dialogs always edit the active document. Never show their preview in another tab.
    if app.session.active_index() != Some(idx) {
        return None;
    }
    if crate::adjust_preview::on_layer(app, idx) {
        return None;
    }
    let d = app.ui.dialogs.iter().find(|d| d.fields.contains_key("__filter"))?;
    if d.fields.get("__preview").and_then(serde_json::Value::as_bool) != Some(true) {
        return None;
    }
    let cmd = d.fields.get("__command")?.as_str()?.to_string();
    // Previews edit what the command will: a targeted layer mask included (#780).
    let mut params = app.with_mask_target(&cmd, crate::filter_dialog::params_of(&d.fields));
    // Select and Mask previews in its view mode; holding X shows the image without one.
    if crate::select_mask_view::owns(&d.fields) {
        if crate::select_mask_view::hides_view(&d.fields) {
            return None;
        }
        params = crate::select_mask_view::request_params(&d.fields, params);
    }
    let st = app.session.documents().get(idx)?;
    // Filters whose features don't scale with a proxy preview at full resolution (#2063).
    let k = crate::filter_dialog::preview_factor(&cmd, &params, crate::proxy::preview_factor(&st.doc, crate::proxy::reduced_previews(app)));
    let request = crate::filter_dialog::FilterPreviewKey {
        doc: st.doc.id,
        revision: st.revision,
        dialog: d.id,
        active: st.active_layer,
        target: st.channel_view.target,
        command: cmd,
        params,
        k,
    };
    let key = request.doc.0 ^ (1u64 << 61);
    // Every filter's preview computes off the UI thread in the desktop app: Gaussian Blur at
    // 1000 px takes long enough to freeze the window. Once the slider settles, a preview still
    // running for an older value is cancelled.
    #[cfg(not(target_arch = "wasm32"))]
    if app.background_jobs {
        app.filter_preview_worker.supersede(&request, crate::gpu_canvas::now_ms());
        if let Some((finished, computed)) = app.filter_preview_worker.poll()
            && finished == request
        {
            match computed {
                Ok(computed) => publish_filter_preview(app, idx, finished, computed.result, computed.ms),
                Err(error) => {
                    app.filter_preview = Some(crate::filter_dialog::FilterPreview { key: finished, result: None, committing: false });
                    crate::notices::error(app, error);
                }
            }
        }
        let fresh = app.filter_preview.as_ref().is_some_and(|p| p.key == request);
        if !fresh && !app.filter_preview_worker.busy() {
            let doc = app.session.documents().get(idx)?.doc.clone();
            let documents: Vec<_> = app.session.documents().iter().map(|d| d.doc.clone()).collect();
            let task = request.clone();
            if let Err(error) = app.filter_preview_worker.start(request.clone(), ctx.clone(), move |cancel| {
                let t0 = crate::gpu_canvas::now_ms();
                let result = crate::filter_dialog::preview_document_for_target(
                    &doc,
                    task.active,
                    &task.command,
                    &task.params,
                    task.k,
                    Some(cancel),
                    task.target,
                    &documents,
                )
                .map(|doc| {
                    let buffer = photocraft_compose::flatten(&doc);
                    (std::sync::Arc::new(doc), buffer)
                });
                crate::filter_preview_worker::Computed { result, ms: crate::gpu_canvas::now_ms() - t0 }
            }) {
                app.filter_preview = Some(crate::filter_dialog::FilterPreview { key: request.clone(), result: None, committing: false });
                crate::notices::error(app, error);
            }
        }
        if app.filter_preview_worker.busy() {
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        }
        let preview = app.filter_preview.as_ref()?;
        // Keep the last preview during a drag, but never one from another source or dialog.
        // Completed worker results above must match every parameter before being uploaded.
        if preview.key.doc != request.doc
            || preview.key.revision != request.revision
            || preview.key.dialog != request.dialog
            || preview.key.active != request.active
            || preview.key.target != request.target
            || preview.key.k != request.k
            || preview.key.command != request.command
        {
            return None;
        }
        let result = preview.result.as_ref()?;
        return Some((preview.key.k, key, [result.size.width, result.size.height]));
    }
    // Web and explicitly inline sessions keep the deterministic synchronous path (#1676):
    // computed inline, uploaded when a GPU texture exists, still inspectable without one.
    let _ = ctx;
    if app.filter_preview.as_ref().is_none_or(|p| p.key != request) {
        let doc = app.session.documents().get(idx)?.doc.clone();
        let documents: Vec<_> = app.session.documents().iter().map(|d| d.doc.clone()).collect();
        let t0 = crate::gpu_canvas::now_ms();
        let result = crate::filter_dialog::preview_document_for_target(
            &doc,
            request.active,
            &request.command,
            &request.params,
            request.k,
            None,
            request.target,
            &documents,
        )
        .map(std::sync::Arc::new);
        if let Some(r) = &result {
            let buf = photocraft_compose::flatten(r);
            let t1 = crate::gpu_canvas::now_ms();
            let (display, _) = canvas_display(app, &doc, None);
            if let Some(gpu) = app.gpu.as_ref() {
                gpu.upload_buffer_full(key, &texture_buffer(display.as_deref(), &buf), doc.depth);
            }
            app.perf.record("filter-preview", r.size.area(), t1 - t0, crate::gpu_canvas::now_ms() - t1);
        }
        let buffer = result.as_ref().map(|r| photocraft_compose::flatten(r));
        publish_filter_preview(app, idx, request, result.zip(buffer), crate::gpu_canvas::now_ms() - t0);
    }
    let preview = app.filter_preview.as_ref()?;
    let result = preview.result.as_ref()?;
    Some((preview.key.k, key, [result.size.width, result.size.height]))
}

/// Channel overlays must use the filter's current proxy too, including on the CPU canvas.
/// The cache key changes with the completed request; unchanged frames reuse the overlay texture.
pub(crate) fn filter_channel_preview(app: &mut PhotocraftApp, idx: usize, ctx: &egui::Context) -> Option<(std::sync::Arc<Document>, u64)> {
    use std::hash::{Hash, Hasher};
    ensure_filter_preview(app, idx, ctx)?;
    let preview = app.filter_preview.as_ref()?;
    let doc = preview.result.as_ref()?.clone();
    let mut key = std::collections::hash_map::DefaultHasher::new();
    format!("{:?}", preview.key).hash(&mut key);
    Some((doc, key.finish() | (1 << 59)))
}

fn publish_filter_preview(
    app: &mut PhotocraftApp,
    idx: usize,
    request: crate::filter_dialog::FilterPreviewKey,
    computed: Option<(std::sync::Arc<photocraft_doc::Document>, photocraft_compose::Buffer)>,
    ms: f64,
) {
    let result = computed.and_then(|(result, buffer)| {
        let t0 = crate::gpu_canvas::now_ms();
        let doc = app.session.documents().get(idx)?.doc.clone();
        let (display, _) = canvas_display(app, &doc, None);
        // A GPU texture is an optimisation: headless/CPU tests must still see the preview.
        if let Some(gpu) = app.gpu.as_ref() {
            gpu.upload_buffer_full(request.doc.0 ^ (1u64 << 61), &texture_buffer(display.as_deref(), &buffer), doc.depth);
        }
        app.perf.record("filter-preview", result.size.area(), ms, crate::gpu_canvas::now_ms() - t0);
        Some(result)
    });
    app.filter_preview = Some(crate::filter_dialog::FilterPreview { key: request, result, committing: false });
}

/// After a filter dialog's OK: `Some(shown)` while its preview stands in for the background job
/// that commits it (the document unchanged and the job still running, for this document's views),
/// `None` otherwise (other documents' views included). A job that lands, fails or is cancelled
/// drops it.
pub(crate) fn committed_filter_preview(app: &mut PhotocraftApp, idx: usize) -> Option<Option<(u32, u64, [u32; 2])>> {
    let p = app.filter_preview.as_ref().filter(|p| p.committing)?;
    let st = app.session.documents().get(idx)?;
    if st.doc.id != p.key.doc {
        return None;
    }
    let size = p.result.as_ref().map(|r| [r.size.width, r.size.height]);
    if let Some(size) = size
        && st.revision == p.key.revision
        && app.session.job_on(p.key.doc).is_some()
    {
        return Some(Some((p.key.k, p.key.doc.0 ^ (1u64 << 61), size)));
    }
    app.filter_preview = None;
    Some(None)
}

/// The document pixels a view shows, with a margin for filtering. Uses all four canvas corners
/// so a rotated camera still covers the visible tiles.
fn visible_doc_rect(xf: &ViewXform) -> DRect {
    let r = xf.rect;
    let docs = [xf.to_doc(r.min), xf.to_doc(pos2(r.max.x, r.min.y)), xf.to_doc(r.max), xf.to_doc(pos2(r.min.x, r.max.y))];
    let mut x0 = docs[0][0];
    let mut y0 = docs[0][1];
    let mut x1 = x0;
    let mut y1 = y0;
    for d in docs {
        x0 = x0.min(d[0]);
        y0 = y0.min(d[1]);
        x1 = x1.max(d[0]);
        y1 = y1.max(d[1]);
    }
    let c = |v: f64| v.clamp(-1e9, 1e9) as i32;
    DRect::new(c(x0.floor()) - 2, c(y0.floor()) - 2, c(x1.ceil()) + 2, c(y1.ceil()) + 2)
}

/// Zoomed-out Image › Adjustments preview on a large document: the wgpu compositor renders the
/// reduced preview document (`adjust_preview::gpu_proxy`) into its own texture, over the target's
/// area after the first frame. Returns (factor, gpu key, dimensions) to draw.
fn ensure_adjust_proxy(app: &mut PhotocraftApp, idx: usize, zoom: f32) -> Option<(u32, u64, [u32; 2])> {
    let frame = crate::adjust_preview::gpu_proxy(app, idx, zoom)?;
    let key = frame.doc.id.0;
    let size = [frame.doc.size.width, frame.doc.size.height];
    let gpu = app.gpu.clone()?;
    if frame.stale || !gpu.has(key, size) {
        let doc = app.session.documents().get(idx)?.doc.clone();
        let (display, _) = canvas_display(app, &doc, None);
        let r = gpu.refresh(key, &frame.doc, frame.damage, display.as_deref());
        app.perf.record(if r.kind.starts_with("gpu") { "gpu-adjust-proxy" } else { "adjust-proxy" }, r.px, r.composite_ms, r.upload_ms);
        crate::adjust_preview::proxy_drawn(app, &frame);
    }
    Some((frame.k, key, size))
}

/// If a live adjustment preview is active on a large document, composite it on the proxy and upload
/// it under its own GPU key. Returns (factor, gpu key, dimensions) when the proxy should be drawn.
fn ensure_proxy_preview(app: &mut PhotocraftApp, idx: usize) -> Option<(u32, u64, [u32; 2])> {
    let (layer, params) = app.live_adjust.clone()?;
    let (doc_id, revision, doc) = {
        let st = app.session.documents().get(idx)?;
        (st.doc.id, st.revision, st.doc.clone())
    };
    // Low-resolution previews off: k is 1 and the canvas shows `display_doc` (full size).
    let k = crate::proxy::preview_factor(&doc, crate::proxy::reduced_previews(app));
    if k <= 1 {
        return None;
    }
    let fresh = matches!(&app.proxy, Some((d, r, kk, _)) if *d == doc_id && *r == revision && *kk == k);
    if !fresh {
        app.proxy = Some((doc_id, revision, k, std::sync::Arc::new(crate::proxy::proxy_document(&doc, k))));
    }
    let proxy = app.proxy.as_ref()?.3.clone();
    let key = doc_id.0 ^ (1u64 << 62);
    let hash = params.to_string().bytes().fold(k as u64, |h, b| h.wrapping_mul(31).wrapping_add(b as u64));
    if app.proxy_uploaded != Some((doc_id, hash)) {
        let l = proxy.layer(layer)?;
        let LayerContent::Adjustment(a) = &l.content else { return None };
        let kind = photocraft_engine::commands::adjustment_kind(a);
        let preview = photocraft_engine::adjust_params::from_params(kind, &params, Some(a), proxy.mode).unwrap_or_else(|_| a.clone());
        let mut p = (*proxy).clone();
        if let Some(lm) = p.layer_mut(layer) {
            lm.content = LayerContent::Adjustment(preview);
        }
        let t0 = crate::gpu_canvas::now_ms();
        let buf = photocraft_compose::flatten(&p);
        let t1 = crate::gpu_canvas::now_ms();
        let (display, _) = canvas_display(app, &doc, None);
        app.gpu.as_ref()?.upload_buffer_full(key, &texture_buffer(display.as_deref(), &buf), doc.depth);
        app.perf.record("proxy", p.size.area(), t1 - t0, crate::gpu_canvas::now_ms() - t1);
        app.proxy_uploaded = Some((doc_id, hash));
    }
    Some((k, key, [proxy.size.width, proxy.size.height]))
}

/// A CPU composite as the GPU canvas texture stores it (sRGB-encoded for linear documents).
fn texture_buffer<'a>(
    display: Option<&photocraft_engine::display_color::CanvasDisplay>,
    buf: &'a photocraft_compose::Buffer,
) -> std::borrow::Cow<'a, photocraft_compose::Buffer> {
    match display {
        Some(d) => d.texture_buffer(buf),
        None => std::borrow::Cow::Borrowed(buf),
    }
}

pub(crate) fn retain_gpu_documents(app: &mut PhotocraftApp) {
    // Remove a closed adjustment owner's proxy before collecting its live GPU keys.
    crate::adjust_preview::retain_documents(app);
    if let Some(gpu) = &app.gpu {
        // Keep each document's texture plus its preview textures (filter preview, adjustment proxy);
        // retaining only document ids freed the previews every frame (blank canvas while previewing).
        let mut live: Vec<u64> = app.session.documents().iter().flat_map(|st| [st.doc.id.0, st.doc.id.0 ^ (1u64 << 61), st.doc.id.0 ^ (1u64 << 62)]).collect();
        live.extend(crate::adjust_preview::gpu_keys(app));
        gpu.retain(&live);
    }
    crate::background_redraw::retain(app);
    // Upload markers must not outlive the resources they describe: native reopen can reuse
    // both the document ID and revision before the next frame while a preview dialog stays open.
    let documents = app.session.documents();
    if app.filter_preview.as_ref().is_some_and(|preview| !documents.iter().any(|st| st.doc.id == preview.key.doc)) {
        app.filter_preview = None;
    }
    if app.proxy_uploaded.is_some_and(|(doc, _)| !documents.iter().any(|st| st.doc.id == doc)) {
        app.proxy_uploaded = None;
        app.proxy = None;
    }
}

/// Tabs + canvas for the active document, or the start screen; then drops layers dragged onto
/// another document (`layer_transfer`).
pub fn document_area(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    documents(app, ui);
    crate::layer_transfer::finish(app, ui.ctx());
    // Once a frame, not per document view: the right-click or options-bar Brush Preset picker.
    crate::paint_mouse::show_picker(app, ui.ctx());
}

/// Window › Arrange tiles: each document shown with its tile in `rect`, or `None` when the
/// active document fills the area.
pub(crate) fn arranged_cells(app: &PhotocraftApp, rect: Rect) -> Option<Vec<(usize, Rect)>> {
    let idx = app.session.active_index()?;
    let n = app.session.documents().len();
    let cells = crate::view_cmds::cells(&app.ui.view.arrange, rect, n)?;
    let mut shown: Vec<(usize, Rect)> = (0..n).map(|k| (idx + k) % n).zip(cells).collect();
    shown.sort_by_key(|(d, _)| *d);
    Some(shown)
}

fn documents(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    app.drop_canvas_rect = None;
    app.tab_strip = None;
    retain_gpu_documents(app);
    crate::transform_tool::track_steps(app, ui.ctx());
    crate::cas_transform::tick(app, ui.ctx());
    crate::symmetry_ui::track(app, ui.ctx());
    let n = app.session.documents().len();
    // Files opening in the background (#210) have tabs before they have documents.
    let opening = !app.jobs.opens.is_empty();
    if !opening && app.ui.chrome.shows_home(n, app.session.prefs().general.auto_show_home_screen) {
        start_screen(app, ui);
        return;
    }
    if n == 0 && !opening {
        // Auto show the Home Screen is off: an empty workspace, like Photoshop.
        paint_dots(ui, ui.available_rect_before_wrap(), app.session.prefs().interface.show_canvas_dots);
        return;
    }
    if (!app.ui.view.hides_tabs() || opening) && !(app.services.host_save.is_some() && n == 1 && !opening) {
        // Flat Spectrum tabs sit directly against the canvas. The parent's default vertical
        // item spacing would otherwise leave a visible gap below the tab strip (#2188).
        let spacing = ui.spacing().item_spacing.y;
        if crate::theme::Tokens::get(ui.ctx()).pro {
            ui.spacing_mut().item_spacing.y = 0.0;
        }
        app.tab_strip = Some(tabs(app, ui));
        ui.spacing_mut().item_spacing.y = spacing;
        drop_slot_line(app, ui);
    }
    if let Some(job) = app.jobs.focus.or_else(|| (n == 0).then(|| app.jobs.opens.last().map(|o| o.job)).flatten()) {
        crate::jobs_ui::open_card(app, ui, job);
        return;
    }
    let Some(idx) = app.session.active_index() else { return };
    let rect = ui.available_rect_before_wrap();
    app.last_canvas_rect = rect;
    app.drop_canvas_rect = Some(rect);
    // Window › Arrange: tiled / n-up layouts show several documents side by side; the active one
    // takes input, a click elsewhere activates that document.
    if let Some(shown) = arranged_cells(app, rect) {
        let t = crate::theme::Tokens::get(ui.ctx());
        for (d, cell) in shown {
            let cell = cell.shrink(1.0);
            let view = app.ui.views[d].clone();
            let v = canvas_view(app, ui, d, cell, view, d == idx);
            if d != idx {
                app.ui.views[d] = v;
                if ui.input(|i| i.pointer.primary_pressed() && i.pointer.interact_pos().is_some_and(|p| cell.contains(p))) {
                    app.session.set_active(d);
                }
            }
            let stroke = if d == idx { Stroke::new(1.5, t.accent) } else { Stroke::new(1.0, Color32::from_black_alpha(160)) };
            ui.painter().rect_stroke(cell, 0, stroke, egui::StrokeKind::Inside);
        }
        return;
    }
    let view = app.ui.views[idx].clone();
    canvas_view(app, ui, idx, rect, view, true);
}

/// The tab strip; returns where its document tabs are.
fn tabs(app: &mut PhotocraftApp, ui: &mut egui::Ui) -> TabStrip {
    let t = crate::theme::Tokens::get(ui.ctx());
    if t.pro {
        return pro_tabs(app, ui);
    }
    let active = app.session.active_index();
    let tab_count = app.session.documents().len();
    let mut doc_tabs = Vec::with_capacity(tab_count);
    let mut activate = None;
    let mut close = None;
    let mut tab_action = None;
    let mut tab_move = None;
    let mut tear_off = None;
    let (mut focus_open, mut cancel_open) = (None, None);
    let focused_open = app.jobs.focus.is_some();
    // Layers dragged over a tab show its document (`layer_transfer`).
    let dragging = crate::layer_transfer::pointer_if_armed(app, ui.ctx());
    let mut drag_over = None;
    let large_tabs = app.session.prefs().workspace.large_tabs;
    let tab_h = if large_tabs { 34.0 } else { 26.0 };
    let font = crate::theme::medium(if large_tabs { 14.0 } else { 12.5 });
    let meta_font = egui::FontId::proportional(if large_tabs { 12.0 } else { 10.5 });
    // Files opening in the background (#210) are tabs too; they share the fit's index space,
    // after the documents.
    let opening = crate::jobs_ui::open_tabs(app);
    // Every tab's natural width, whether or not it stays on the strip.
    let natural: Vec<f32> = app
        .session
        .documents()
        .iter()
        .map(|st| {
            let name = studio_tab_name(app, st);
            ui.painter().layout_no_wrap(name, font.clone(), t.text).size().x
                + ui.painter().layout_no_wrap(format!("{}/{}", mode_label(&st.doc), st.doc.depth.bits()), meta_font.clone(), t.text_faint).size().x
                + STUDIO_TAB_PAD
        })
        .chain(opening.iter().map(|(_, name, frac)| {
            ui.painter().layout_no_wrap(name.clone(), font.clone(), t.text).size().x
                + ui.painter().layout_no_wrap(format!("{:.0}%", frac * 100.0), meta_font.clone(), t.text_faint).size().x
                + STUDIO_TAB_PAD
        }))
        .collect();
    let selected = app
        .jobs
        .focus
        .and_then(|job| opening.iter().position(|(open, _, _)| *open == job))
        .map_or_else(|| app.session.active_index().unwrap_or(0), |p| tab_count + p);
    let frame = egui::Frame::NONE.fill(t.canvas).inner_margin(egui::Margin { left: 8, right: 8, top: 6, bottom: 4 }).show(ui, |ui| {
        let (row, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), tab_h), Sense::hover());
        let f = crate::tab_strip::fit(&natural, selected, row.width(), DOC_TAB_MIN_W, crate::tab_strip::CHEVRON_W);
        let mut x = row.left();
        let mut placed: Vec<(usize, Rect)> = Vec::with_capacity(f.shown.len());
        for &(i, w) in &f.shown {
            // The old horizontal layout put 4 pt between tabs; `w` carries it, the tab doesn't.
            let r = Rect::from_min_size(egui::pos2(x, row.top()), egui::vec2((w - STUDIO_TAB_GAP).max(8.0), row.height()));
            placed.push((i, r));
            x = r.right() + STUDIO_TAB_GAP;
        }
        for &(i, r) in placed.iter().filter(|(i, _)| *i < tab_count) {
            let Some(st) = app.session.documents().get(i) else { continue };
            let sel = Some(i) == active && !focused_open;
            let name = studio_tab_name(app, st);
            let meta = format!("{}/{}", mode_label(&st.doc), st.doc.depth.bits());
            let natural_w = natural.get(i).copied().unwrap_or(0.0);
            let meta_g = ui.painter().layout_no_wrap(meta, meta_font.clone(), t.text_faint);
            let name_max = (r.width() - STUDIO_TAB_PAD + STUDIO_TAB_GAP - meta_g.size().x).max(1.0);
            let name_g = crate::tab_strip::elided(ui, &name, font.clone(), t.text, name_max);
            // The title was cut: the tooltip has the rest of it.
            let cut = name_g.size().x + 0.5 < natural_w - STUDIO_TAB_PAD + STUDIO_TAB_GAP - meta_g.size().x;
            let resp = ui.interact(r, ui.id().with(("dtab", i)), Sense::click_and_drag());
            let saving = crate::jobs_ui::save_progress(app, st.doc.id);
            resp.widget_info(|| {
                let label = saving.as_ref().map_or_else(|| st.doc.name.clone(), |s| format!("{} - {s}", st.doc.name));
                egui::WidgetInfo::selected(egui::WidgetType::SelectableLabel, true, sel, label)
            });
            doc_tabs.push((i, r));
            if sel {
                ui.painter().rect_filled(r, t.radius_sm, t.card);
                ui.painter().rect_stroke(r, t.radius_sm, Stroke::new(1.0, t.card_border), egui::StrokeKind::Inside);
            } else if resp.hovered() {
                ui.painter().rect_filled(r, t.radius_sm, t.hover.gamma_multiply(0.5));
            }
            if dragging.is_some_and(|p| r.contains(p)) {
                drag_over = Some(i);
                ui.painter().rect_stroke(r, t.radius_sm, Stroke::new(1.5, t.accent), egui::StrokeKind::Inside);
            }
            let color = if sel { t.text } else { t.text_dim };
            let ny = r.center().y - name_g.size().y / 2.0;
            ui.painter().galley_with_override_text_color(egui::pos2(r.left() + 10.0, ny), name_g.clone(), color);
            ui.painter().galley(egui::pos2(r.left() + 16.0 + name_g.size().x, r.center().y - meta_g.size().y / 2.0), meta_g, t.text_faint);
            let xr = Rect::from_center_size(egui::pos2(r.right() - 12.0, r.center().y), egui::vec2(16.0, 16.0));
            let xresp = ui.interact(xr, ui.id().with(("dtabx", i)), Sense::click());
            if xresp.hovered() {
                ui.painter().rect_filled(xr, 4.0, t.hover);
            }
            crate::icons::paint(ui, xr, "x", 11.0, if xresp.hovered() { t.text } else { t.text_faint });
            let resp = if cut { resp.on_hover_text(name) } else { resp };
            if tab_close_clicked(&resp, &xresp) {
                close = Some(i);
            } else if resp.clicked() {
                activate = Some(i);
            }
            match tab_drag(ui, &resp, i, r, row, &placed, &t) {
                Some(TabDrop::Move(to)) => tab_move = Some((i, to)),
                Some(TabDrop::Float(at)) => tear_off = Some((i, at)),
                None => {}
            }
            let has_path = st.path.is_some();
            resp.context_menu(|ui| {
                tab_action = tab_context_menu(ui, i, tab_count, has_path);
            });
        }
        // Files opening in the background: a tab with a progress underline; × cancels.
        for &(i, r) in placed.iter().filter(|(i, _)| *i >= tab_count) {
            let Some((job, name, frac)) = opening.get(i - tab_count) else { continue };
            let sel = app.jobs.focus == Some(*job);
            let meta_g = ui.painter().layout_no_wrap(format!("{:.0}%", frac * 100.0), meta_font.clone(), t.text_faint);
            let name_g = crate::tab_strip::elided(ui, name, font.clone(), t.text, (r.width() - STUDIO_TAB_PAD + STUDIO_TAB_GAP - meta_g.size().x).max(1.0));
            let resp = ui.interact(r, ui.id().with(("dtabjob", job.0)), Sense::click());
            if sel {
                ui.painter().rect_filled(r, t.radius_sm, t.card);
                ui.painter().rect_stroke(r, t.radius_sm, Stroke::new(1.0, t.card_border), egui::StrokeKind::Inside);
            } else if resp.hovered() {
                ui.painter().rect_filled(r, t.radius_sm, t.hover.gamma_multiply(0.5));
            }
            let ny = r.center().y - name_g.size().y / 2.0;
            ui.painter().galley_with_override_text_color(egui::pos2(r.left() + 10.0, ny), name_g.clone(), if sel { t.text } else { t.text_dim });
            ui.painter().galley(egui::pos2(r.left() + 16.0 + name_g.size().x, r.center().y - meta_g.size().y / 2.0), meta_g, t.text_faint);
            crate::jobs_ui::tab_underline(ui, r, *frac, &t);
            let xr = Rect::from_center_size(egui::pos2(r.right() - 12.0, r.center().y), egui::vec2(16.0, 16.0));
            let xresp = ui.interact(xr, ui.id().with(("dtabjobx", job.0)), Sense::click());
            if xresp.hovered() {
                ui.painter().rect_filled(xr, 4.0, t.hover);
            }
            crate::icons::paint(ui, xr, "x", 11.0, if xresp.hovered() { t.text } else { t.text_faint });
            if xresp.on_hover_text(tl!("Cancel opening")).clicked() {
                cancel_open = Some(*job);
            } else if resp.clicked() {
                focus_open = Some(*job);
            }
        }
        if !f.overflow.is_empty() {
            let left = x.min((row.right() - crate::tab_strip::CHEVRON_W).max(row.left()));
            let r = Rect::from_min_size(egui::pos2(left, row.top()), egui::vec2(crate::tab_strip::CHEVRON_W, row.height()));
            let labels: Vec<&str> =
                app.session.documents().iter().map(|st| st.doc.name.as_str()).chain(opening.iter().map(|(_, name, _)| name.as_str())).collect();
            let mut picked = None;
            crate::tab_strip::overflow_button(ui, ui.id().with("dtab-overflow"), r, tl!("More documents"), &labels, &f.overflow, &mut picked);
            match picked {
                Some(i) if i < tab_count => activate = Some(i),
                Some(i) => focus_open = opening.get(i - tab_count).map(|(job, _, _)| *job).or(focus_open),
                None => {}
            }
        }
    });
    open_tab_clicks(app, activate, focus_open, cancel_open);
    if let Some(i) = drag_over {
        crate::layer_transfer::over_tab(app, ui.ctx(), i);
    }
    if let Some(i) = close {
        let _ = crate::menus::invoke(app, ui.ctx(), "file.close", json!({"document": i}));
    }
    if let Some((from, to)) = tab_move
        && let Err(e) = crate::menus::invoke(app, ui.ctx(), "document.move", json!({"document": from, "to": to}))
    {
        app.ui.status = e;
        app.ui.status_error = true;
    }
    float_torn_off(app, ui.ctx(), tear_off);
    if let Some((id, params)) = tab_action
        && let Err(e) = crate::menus::invoke(app, ui.ctx(), id, params)
    {
        app.ui.status = e;
        app.ui.status_error = true;
    }
    TabStrip { rect: frame.response.rect, tabs: doc_tabs }
}

/// Whether a document tab was asked to close: its × was clicked (#619), or the tab was clicked
/// with the middle button (#2570), as in a browser. A middle click over the × counts too, so the
/// whole tab behaves the same way. Both strips share this rule; the caller routes it through the
/// guarded `file.close`, so a document with unsaved changes still gets the save prompt.
fn tab_close_clicked(resp: &egui::Response, xresp: &egui::Response) -> bool {
    xresp.clicked() || xresp.middle_clicked() || resp.middle_clicked()
}

/// All tab actions go through the same guarded File commands as the menu bar, including the
/// unsaved-changes prompt. The clicked tab is explicit even when another document is active.
fn tab_context_items(index: usize, count: usize, has_path: bool) -> [(&'static str, &'static str, serde_json::Value, bool); 4] {
    [
        ("Close", "file.close", json!({"document": index}), true),
        ("Close Others", "file.closeOthers", json!({"document": index}), count > 1),
        ("Close All", "file.closeAll", json!({}), true),
        (crate::menus::reveal_label(), "file.revealInFinder", json!({"document": index}), has_path),
    ]
}

fn tab_context_menu(ui: &mut egui::Ui, index: usize, count: usize, has_path: bool) -> Option<(&'static str, serde_json::Value)> {
    crate::widgets::menu_scroll(ui, |ui| {
        ui.set_min_width(170.0);
        for (label, id, params, enabled) in tab_context_items(index, count, has_path) {
            if ui.add_enabled(enabled, egui::Button::new(tl!(label))).clicked() {
                ui.close();
                return Some((id, params));
            }
        }
        None
    })
}

/// Where a tab dragged to `x` would land: before the first shown tab whose middle is right of
/// `x`, else after the last shown tab (its document index + 1), like [`TabStrip::slot`].
/// How far (points) above or below the tab strip a dragged tab must be to tear off (#3733).
const TEAR_OFF: f32 = 24.0;

/// Whether a tab dragged to `p` has left the strip and floats into its own window, as in
/// Photoshop; sideways drags along the strip still reorder (#217).
fn torn_off(strip: Rect, p: egui::Pos2) -> bool {
    p.y > strip.bottom() + TEAR_OFF || p.y < strip.top() - TEAR_OFF
}

/// What a finished document-tab drag asks for.
#[derive(Clone, Copy, Debug, PartialEq)]
enum TabDrop {
    /// Reorder the tab to this index.
    Move(usize),
    /// Float the document in its own OS window at this screen position (points), when known.
    Float(Option<[f32; 2]>),
}

/// Drag a document tab: along the strip it reorders (a line shows where it will land); off the
/// strip it tears off into a floating window that can go anywhere, other screens included
/// (an outline under the pointer shows it). Returns the drop once the drag ends.
fn tab_drag(ui: &egui::Ui, resp: &egui::Response, i: usize, r: Rect, strip: Rect, placed: &[(usize, Rect)], t: &crate::theme::Tokens) -> Option<TabDrop> {
    if resp.dragged()
        && let Some(p) = resp.interact_pointer_pos()
    {
        if torn_off(strip, p) {
            let ghost = Rect::from_min_size(p - egui::vec2(40.0, 12.0), egui::vec2(240.0, 160.0));
            let painter = ui.ctx().layer_painter(egui::LayerId::new(egui::Order::Tooltip, ui.id().with("tab-tear-off")));
            painter.rect_filled(ghost, t.radius_sm, t.accent.gamma_multiply(0.12));
            painter.rect_stroke(ghost, t.radius_sm, Stroke::new(1.5, t.accent), egui::StrokeKind::Inside);
        } else {
            tab_drop_line(ui, placed, p.x, t);
        }
    }
    if !resp.drag_stopped() {
        return None;
    }
    let p = resp.interact_pointer_pos().or_else(|| ui.input(|i| i.pointer.interact_pos()));
    if let Some(p) = p.filter(|&p| torn_off(strip, p)) {
        // The pointer is in this window's points; the OS wants screen coordinates. Keep the new
        // window's title bar under the pointer, as the ghost showed.
        let at = ui.ctx().input(|i| i.viewport().inner_rect).map(|inner| {
            let s = inner.min + p.to_vec2() - egui::vec2(40.0, 12.0);
            [s.x, s.y]
        });
        return Some(TabDrop::Float(at));
    }
    let to = tab_move_target(i, tab_drop_index(placed, p.map_or(r.center().x, |p| p.x)));
    (to != i).then_some(TabDrop::Move(to))
}

/// Float a torn-off tab's document in its own window (`window.arrange.floatInWindow`).
fn float_torn_off(app: &mut PhotocraftApp, ctx: &egui::Context, tear_off: Option<(usize, Option<[f32; 2]>)>) {
    let Some((doc, at)) = tear_off else { return };
    let params = match at {
        Some(at) => json!({"document": doc, "position": at}),
        None => json!({"document": doc}),
    };
    if let Err(e) = crate::menus::invoke(app, ctx, "window.arrange.floatInWindow", params) {
        app.ui.status = e;
        app.ui.status_error = true;
    }
}

fn tab_drop_index(placed: &[(usize, Rect)], x: f32) -> usize {
    match placed.iter().find(|(_, r)| r.center().x >= x) {
        Some(&(i, _)) => i,
        None => placed.last().map_or(0, |&(i, _)| i.saturating_add(1)),
    }
}

/// The `document.move` target for dropping the tab at `from` at insertion index `k`: the slot
/// shifts down when the dragged tab itself is removed first; next to itself it stays put.
fn tab_move_target(from: usize, k: usize) -> usize {
    if k > from { k - 1 } else { k }
}

/// The insertion line shown while a tab is dragged to reorder it.
fn tab_drop_line(ui: &egui::Ui, placed: &[(usize, Rect)], x: f32, t: &crate::theme::Tokens) {
    match placed.iter().find(|(_, r)| r.center().x >= x) {
        Some(&(_, r)) => crate::widgets::drop_line(ui, r, false, true, t),
        None => {
            if let Some(&(_, r)) = placed.last() {
                crate::widgets::drop_line(ui, r, true, true, t);
            }
        }
    }
}

/// Apply tab-strip clicks: a document tab shows that document, an opening tab its progress, and
/// an opening tab's × cancels the open (closing the half-open tab).
fn open_tab_clicks(
    app: &mut PhotocraftApp,
    activate: Option<usize>,
    focus_open: Option<photocraft_engine::jobs::JobId>,
    cancel_open: Option<photocraft_engine::jobs::JobId>,
) {
    if let Some(i) = activate {
        app.session.set_active(i);
        app.jobs.focus = None;
    }
    if let Some(job) = focus_open {
        app.jobs.focus = Some(job);
    }
    if let Some(job) = cancel_open {
        crate::jobs_ui::cancel(app, job);
    }
}

/// Document tabs: the chrome around a title (the × and its gaps), and the width a tab shrinks to
/// before it moves into the » overflow menu.
const DOC_TAB_PAD: f32 = 42.0;
const DOC_TAB_MIN_W: f32 = 72.0;
/// The Studio strip's tabs carried 4 pt of spacing between them and 44 pt of chrome; `fit` sees
/// one number per tab, so the gap rides in the natural width and comes back out when painted.
const STUDIO_TAB_PAD: f32 = 48.0;
const STUDIO_TAB_GAP: f32 = 4.0;

/// Photoshop's tab title: "name @ 50% (Layer 1, RGB/8)", "(Layer 1, Layer Mask/8)" when the mask
/// is targeted; the Background layer's name is omitted.
fn pro_tab_title(app: &PhotocraftApp, i: usize) -> String {
    let Some(st) = app.session.documents().get(i) else { return String::new() };
    let zoom = app.ui.views.get(i).map_or(100.0, |v| v.zoom * 100.0);
    let active_layer = st.active_layer.and_then(|id| st.doc.layer(id)).filter(|l| !st.doc.is_background(l.id));
    let mask = app.session.active_index() == Some(i) && app.ui.mask_target && active_layer.is_some_and(|l| l.mask.is_some());
    let model = if mask { tl!("Layer Mask").to_string() } else { mode_label(&st.doc).to_string() };
    let layer = active_layer.map(|l| format!("{}, ", l.name)).unwrap_or_default();
    // A save in progress: "name @ 100% (Layer 1, RGB/16) - Saving 45%", as Photoshop does (#3389).
    let saving = crate::jobs_ui::save_progress(app, st.doc.id).map(|s| format!(" - {s}")).unwrap_or_default();
    format!("{} @ {}% ({layer}{model}/{}){}{saving}", st.doc.name, fmt_zoom(zoom), st.doc.depth.bits(), if st.is_dirty() { "*" } else { "" })
}

/// The Studio strip's tab name: "name *" when unsaved, "name * - Saving 45%" while it saves (#3389).
fn studio_tab_name(app: &PhotocraftApp, st: &photocraft_engine::DocState) -> String {
    let saving = crate::jobs_ui::save_progress(app, st.doc.id).map(|s| format!(" - {s}")).unwrap_or_default();
    format!("{}{}{saving}", st.doc.name, if st.is_dirty() { " *" } else { "" })
}

/// Photoshop document tabs: "name @ 33.3% (RGB/8)" on a dark strip; active tab matches panels.
/// Tabs that don't fit shrink and elide, and the ones that still don't fit move into a » menu at
/// the end of the strip; the active document's tab always stays on the strip (#1276).
fn pro_tabs(app: &mut PhotocraftApp, ui: &mut egui::Ui) -> TabStrip {
    let t = crate::theme::Tokens::get(ui.ctx());
    let active = app.session.active_index().filter(|_| app.jobs.focus.is_none());
    let (mut activate, mut close) = (None, None);
    let mut tab_action = None;
    let mut tab_move = None;
    let mut tear_off = None;
    let tab_count = app.session.documents().len();
    let dragging = crate::layer_transfer::pointer_if_armed(app, ui.ctx());
    let mut drag_over = None;
    let mac = ui.ctx().os() == egui::os::OperatingSystem::Mac;
    let large_tabs = app.session.prefs().workspace.large_tabs;
    let font = egui::FontId::proportional(if large_tabs { 13.0 } else { 11.5 });
    let tab_h = if large_tabs { 34.0 } else { 26.0 };
    let (strip, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), tab_h), Sense::hover());
    ui.painter().rect_filled(strip, 0.0, t.tab_strip);
    // Files opening in the background (#210) are tabs too ("name (Opening… 45%)" with a progress
    // underline); they share the fit's index space, after the documents.
    let opening = crate::jobs_ui::open_tabs(app);
    let mut titles: Vec<String> = (0..tab_count).map(|i| pro_tab_title(app, i)).collect();
    titles.extend(opening.iter().map(|(_, name, frac)| format!("{name} ({} {:.0}%)", tl!("Opening…"), frac * 100.0)));
    // Every tab's natural width, whether or not it stays on the strip.
    let natural: Vec<f32> = titles.iter().map(|title| ui.painter().layout_no_wrap(title.clone(), font.clone(), t.text).size().x + DOC_TAB_PAD).collect();
    let selected = app
        .jobs
        .focus
        .and_then(|job| opening.iter().position(|(open, _, _)| *open == job))
        .map_or_else(|| app.session.active_index().unwrap_or(0), |p| tab_count + p);
    let f = crate::tab_strip::fit(&natural, selected, strip.width(), DOC_TAB_MIN_W, crate::tab_strip::CHEVRON_W);
    let mut x = strip.left();
    let mut placed: Vec<(usize, Rect)> = Vec::with_capacity(f.shown.len());
    for &(i, w) in &f.shown {
        let r = Rect::from_min_size(egui::pos2(x, strip.top()), egui::vec2(w, strip.height()));
        placed.push((i, r));
        x = r.right();
    }
    let mut doc_tabs = Vec::with_capacity(tab_count);
    let (mut focus_open, mut cancel_open) = (None, None);
    for &(i, r) in placed.iter().filter(|(i, _)| *i < tab_count) {
        let Some(title) = titles.get(i) else { continue };
        let g = crate::tab_strip::elided(ui, title, font.clone(), t.text, (r.width() - DOC_TAB_PAD).max(1.0));
        let resp = ui.interact(r, ui.id().with(("ptab", i)), Sense::click_and_drag());
        let sel = Some(i) == active;
        if sel {
            ui.painter().rect_filled(r, 0.0, t.chrome);
        } else if resp.hovered() {
            ui.painter().rect_filled(r, 0.0, t.hover.gamma_multiply(0.35));
        }
        if dragging.is_some_and(|p| r.contains(p)) {
            drag_over = Some(i);
            ui.painter().rect_stroke(r, 0.0, Stroke::new(1.5, t.accent), egui::StrokeKind::Inside);
        }
        ui.painter().line_segment([r.right_top(), r.right_bottom()], Stroke::new(1.0, t.separator));
        let (xr, at) = pro_tab_layout(r, g.size().y, mac);
        let xresp = ui.interact(xr, ui.id().with(("ptabx", i)), Sense::click());
        // Painted: name the tab and its × for accessibility (and so tests and agents can find them).
        resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::SelectableLabel, true, sel, g.text()));
        xresp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, tl!("Close")));
        crate::icons::paint(ui, xr, "x", 10.0, if xresp.hovered() { t.text } else { t.text_faint });
        ui.painter().galley_with_override_text_color(at, g, if sel { t.text } else { t.text_faint });
        let cut = natural.get(i).copied().unwrap_or(0.0) > r.width() + 0.5;
        let resp = if cut { resp.on_hover_text(title.as_str()) } else { resp };
        if tab_close_clicked(&resp, &xresp) {
            close = Some(i);
        } else if resp.clicked() {
            activate = Some(i);
        }
        match tab_drag(ui, &resp, i, r, strip, &placed, &t) {
            Some(TabDrop::Move(to)) => tab_move = Some((i, to)),
            Some(TabDrop::Float(at)) => tear_off = Some((i, at)),
            None => {}
        }
        let has_path = app.session.documents().get(i).is_some_and(|d| d.path.is_some());
        resp.context_menu(|ui| {
            tab_action = tab_context_menu(ui, i, tab_count, has_path);
        });
        doc_tabs.push((i, r));
    }
    for &(i, r) in placed.iter().filter(|(i, _)| *i >= tab_count) {
        let Some((job, _, frac)) = opening.get(i - tab_count) else { continue };
        let Some(title) = titles.get(i) else { continue };
        let g = crate::tab_strip::elided(ui, title, font.clone(), t.text, (r.width() - DOC_TAB_PAD).max(1.0));
        let resp = ui.interact(r, ui.id().with(("ptabjob", job.0)), Sense::click());
        let sel = app.jobs.focus == Some(*job);
        if sel {
            ui.painter().rect_filled(r, 0.0, t.chrome);
        } else if resp.hovered() {
            ui.painter().rect_filled(r, 0.0, t.hover.gamma_multiply(0.35));
        }
        ui.painter().line_segment([r.right_top(), r.right_bottom()], Stroke::new(1.0, t.separator));
        crate::jobs_ui::tab_underline(ui, r, *frac, &t);
        let (xr, at) = pro_tab_layout(r, g.size().y, mac);
        let xresp = ui.interact(xr, ui.id().with(("ptabjobx", job.0)), Sense::click());
        xresp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, tl!("Close")));
        crate::icons::paint(ui, xr, "x", 10.0, if xresp.hovered() { t.text } else { t.text_faint });
        ui.painter().galley_with_override_text_color(at, g, if sel { t.text } else { t.text_faint });
        if xresp.on_hover_text(tl!("Cancel opening")).clicked() {
            cancel_open = Some(*job);
        } else if resp.clicked() {
            focus_open = Some(*job);
        }
    }
    if !f.overflow.is_empty() {
        let left = x.min((strip.right() - crate::tab_strip::CHEVRON_W).max(strip.left()));
        let r = Rect::from_min_size(egui::pos2(left, strip.top()), egui::vec2(crate::tab_strip::CHEVRON_W, strip.height()));
        let labels: Vec<&str> = titles.iter().map(String::as_str).collect();
        let mut picked = None;
        crate::tab_strip::overflow_button(ui, ui.id().with("ptab-overflow"), r, tl!("More documents"), &labels, &f.overflow, &mut picked);
        match picked {
            Some(i) if i < tab_count => activate = Some(i),
            Some(i) => focus_open = opening.get(i - tab_count).map(|(job, _, _)| *job).or(focus_open),
            None => {}
        }
    }
    open_tab_clicks(app, activate, focus_open, cancel_open);
    if let Some(i) = drag_over {
        crate::layer_transfer::over_tab(app, ui.ctx(), i);
    }
    if let Some(i) = close {
        let _ = crate::menus::invoke(app, ui.ctx(), "file.close", json!({"document": i}));
    }
    if let Some((from, to)) = tab_move
        && let Err(e) = crate::menus::invoke(app, ui.ctx(), "document.move", json!({"document": from, "to": to}))
    {
        app.ui.status = e;
        app.ui.status_error = true;
    }
    float_torn_off(app, ui.ctx(), tear_off);
    if let Some((id, params)) = tab_action
        && let Err(e) = crate::menus::invoke(app, ui.ctx(), id, params)
    {
        app.ui.status = e;
        app.ui.status_error = true;
    }
    TabStrip { rect: strip, tabs: doc_tabs }
}

/// Where the document tabs were drawn (opening files' tabs aren't slots): a file dropped on the
/// strip opens at the slot under the pointer.
#[derive(Clone, Debug, PartialEq)]
pub struct TabStrip {
    pub rect: Rect,
    /// The document tabs shown, left to right, with their document index. When the tabs overflow
    /// into the » menu, some documents have no tab here.
    pub tabs: Vec<(usize, Rect)>,
}

impl TabStrip {
    /// The document position a drop at `x` opens at: before the first shown tab whose middle is
    /// right of it, else after the last shown tab.
    pub fn slot(&self, x: f32) -> usize {
        match self.tabs.iter().find(|(_, r)| r.center().x >= x) {
            Some(&(i, _)) => i,
            None => self.tabs.last().map_or(0, |&(i, _)| i.saturating_add(1)),
        }
    }
}

/// While files are dragged over the tab strip, an insertion line where they would open.
fn drop_slot_line(app: &mut PhotocraftApp, ui: &egui::Ui) {
    if ui.input(|i| i.raw.hovered_files.is_empty()) {
        return;
    }
    let at = app.services.cursor_pos.as_mut().and_then(|f| f(ui.ctx()));
    let crate::file_open::DropTarget::Tabs(slot) = app.drop_target(ui.ctx(), at) else { return };
    let Some(tabs) = app.tab_strip.as_ref().map(|s| &s.tabs) else { return };
    let Some((r, after)) = tabs.iter().find(|(i, _)| *i == slot).map(|&(_, r)| (r, false)).or_else(|| tabs.last().map(|&(_, r)| (r, true))) else {
        return;
    };
    crate::widgets::drop_line(ui, r, after, true, &crate::theme::Tokens::get(ui.ctx()));
}

/// A Photoshop-style tab's close button and where its title starts. Photoshop puts the × after the
/// title on Windows (and so Linux) and before it on macOS (#619).
fn pro_tab_layout(tab: Rect, title_height: f32, mac: bool) -> (Rect, egui::Pos2) {
    let (x, title) = if mac { (tab.left() + 13.0, tab.left() + 26.0) } else { (tab.right() - 13.0, tab.left() + 16.0) };
    (Rect::from_center_size(egui::pos2(x, tab.center().y), egui::vec2(14.0, 14.0)), egui::pos2(title, tab.center().y - title_height / 2.0))
}

/// Photoshop-style zoom label: "33.3", "100", "12.5".
pub fn fmt_zoom(pct: f32) -> String {
    if (pct - pct.round()).abs() < 0.05 { format!("{}", pct.round() as i64) } else { format!("{pct:.1}") }
}

/// Repeating dot-grid texture for the canvas surround.
fn dots(ctx: &egui::Context, t: &crate::theme::Tokens) -> Option<egui::TextureId> {
    if t.bevel {
        return None;
    }
    let key = egui::Id::new(("canvas-dots", format!("{:?}", t.kind)));
    if let Some(tex) = ctx.data(|d| d.get_temp::<egui::TextureHandle>(key)) {
        return Some(tex.id());
    }
    let n = 22usize;
    let mut px = vec![Color32::TRANSPARENT; n * n];
    for (dx, dy, a) in [(0usize, 0usize, 255u8), (1, 0, 110), (0, 1, 110), (1, 1, 60)] {
        let c = t.canvas_dot;
        px[(n / 2 + dy) * n + n / 2 + dx] = Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), a);
    }
    let opts = TextureOptions {
        magnification: egui::TextureFilter::Linear,
        minification: egui::TextureFilter::Linear,
        wrap_mode: egui::TextureWrapMode::Repeat,
        mipmap_mode: None,
    };
    let tex = ctx.load_texture("canvas-dots", egui::ColorImage::new([n, n], px), opts);
    let id = tex.id();
    ctx.data_mut(|d| d.insert_temp(key, tex));
    Some(id)
}

fn paint_dots(ui: &egui::Ui, rect: Rect, enabled: bool) {
    if !enabled {
        return;
    }
    let t = crate::theme::Tokens::get(ui.ctx());
    if let Some(id) = dots(ui.ctx(), &t) {
        let uv = Rect::from_min_max(Pos2::ZERO, pos2(rect.width() / 22.0, rect.height() / 22.0));
        ui.painter_at(rect).image(id, rect, uv, Color32::WHITE);
    }
}

pub fn mode_label(doc: &Document) -> &'static str {
    match doc.mode {
        photocraft_doc::ColorMode::Rgb => "RGB",
        photocraft_doc::ColorMode::Grayscale => "Gray",
        photocraft_doc::ColorMode::Cmyk => "CMYK",
        photocraft_doc::ColorMode::Lab => "Lab",
        photocraft_doc::ColorMode::Indexed => "Indexed",
        photocraft_doc::ColorMode::Bitmap => "Bitmap",
        photocraft_doc::ColorMode::Duotone => "Duotone",
        photocraft_doc::ColorMode::Multichannel => "Multichannel",
    }
}

fn start_screen(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let t = crate::theme::Tokens::get(ui.ctx());
    let area = ui.available_rect_before_wrap();
    paint_dots(ui, area, app.session.prefs().interface.show_canvas_dots);
    // File › Open Recent, newest first (on the web there are no paths to reopen).
    let recent: Vec<String> = if cfg!(target_arch = "wasm32") { Vec::new() } else { app.ui.recent_files.iter().take(HOME_RECENT).cloned().collect() };
    let recent_h = if recent.is_empty() { 0.0 } else { 34.0 + recent.len() as f32 * HOME_RECENT_ROW };
    let card = Rect::from_center_size(area.center(), egui::vec2(460.0, 330.0 + recent_h));
    let new_label = crate::shortcuts::command_label(app, "New document…", "file.new");
    let open_label = crate::shortcuts::command_label(app, "Open…", "file.open");
    ui.scope_builder(egui::UiBuilder::new().max_rect(card), |ui| {
        ui.vertical_centered(|ui| {
            ui.horizontal(|ui| {
                let title = ui.painter().layout_no_wrap(tl!("PhotoCraft").into(), crate::theme::semibold(38.0), t.text);
                let by = ui.painter().layout_no_wrap(tl!("open source").into(), egui::FontId::proportional(13.0), t.text_faint);
                let total = title.size().x + by.size().x + 10.0;
                ui.add_space(((card.width() - total) / 2.0).max(0.0));
                let (r, _) = ui.allocate_exact_size(title.size(), Sense::hover());
                ui.painter().galley(r.min, title, t.text);
                let (r2, _) = ui.allocate_exact_size(egui::vec2(by.size().x + 10.0, r.height()), Sense::hover());
                ui.painter().galley(egui::pos2(r2.left() + 10.0, r.bottom() - by.size().y - 8.0), by, t.text_faint);
            });
            ui.add_space(6.0);
            ui.label(egui::RichText::new(tl!("Create a new document or open an existing file.")).color(t.text_dim).size(14.0));
            ui.add_space(22.0);
            ui.horizontal(|ui| {
                ui.add_space(((card.width() - 2.0 * 190.0 - 12.0) / 2.0).max(0.0));
                ui.spacing_mut().item_spacing.x = 12.0;
                if crate::widgets::primary_button(ui, &new_label, 190.0).clicked() {
                    let fields = app.new_document_fields();
                    app.ui.open_dialog(crate::state::DialogKind::NewDocument, fields);
                }
                if crate::widgets::secondary_button(ui, &open_label, 190.0).clicked() {
                    let _ = app.open_dialog_file();
                }
            });
            ui.add_space(22.0);
            ui.horizontal(|ui| {
                let msg = start_screen_drop_hint(app);
                let g = ui.painter().layout_no_wrap(msg.to_string(), egui::FontId::proportional(12.5), t.text_faint);
                ui.add_space(((card.width() - g.size().x - 24.0) / 2.0).max(0.0));
                let (r, _) = ui.allocate_exact_size(egui::vec2(18.0, 18.0), Sense::hover());
                crate::icons::paint(ui, r, "image", 15.0, t.text_faint);
                ui.label(egui::RichText::new(msg).color(t.text_faint));
            });
            if !recent.is_empty() {
                ui.add_space(22.0);
                home_recent(app, ui, &recent);
            }
            ui.add_space(26.0);
            crate::links::discord_button(app, ui, 190.0);
            ui.add_space(10.0);
            crate::links::link_row(app, ui);
        });
    });
}

/// Wayland has no native file drops (winit 0.30, #386), so the hint offers File › Open and paste.
fn start_screen_drop_hint(app: &PhotocraftApp) -> std::borrow::Cow<'static, str> {
    if app.services.is_wayland {
        crate::i18n::fmt(tl!("Use File › Open, or paste a copied image with {paste}."), &[("paste", &crate::notices::paste_hint(app))]).into()
    } else {
        tl!("Drop an image or PSD anywhere to open it.").into()
    }
}

/// Recent files listed on the Home screen.
const HOME_RECENT: usize = 6;
/// Height of one Home-screen recent-file row.
const HOME_RECENT_ROW: f32 = 30.0;

/// The Home screen's "Recent" list: file name, its folder on the right, click to open. The
/// hovered row shows a × that takes it off the list, and right-click offers Remove from Recent
/// and Clear Recent Files (#2691).
fn home_recent(app: &mut PhotocraftApp, ui: &mut egui::Ui, recent: &[String]) {
    let t = crate::theme::Tokens::get(ui.ctx());
    let width = 380.0;
    ui.horizontal(|ui| {
        // Line the heading up with the file icons.
        ui.add_space(((ui.available_width() - width) / 2.0).max(0.0) + 8.0);
        ui.label(egui::RichText::new(tl!("Recent")).font(crate::theme::semibold(12.5)).color(t.text_dim));
    });
    ui.add_space(4.0);
    let mut open = None;
    let mut remove = None;
    let mut clear = false;
    for path in recent {
        let name = crate::file_open::display_name(path);
        let folder = std::path::Path::new(path).parent().map(|d| d.to_string_lossy().into_owned()).unwrap_or_default();
        let (row, resp) = ui.allocate_exact_size(egui::vec2(width, HOME_RECENT_ROW), Sense::click());
        // The × sits on top of the row, so the row counts as hovered while the pointer is on it.
        let hovered = ui.rect_contains_pointer(row);
        if hovered {
            ui.painter().rect_filled(row, t.radius_sm, t.hover);
        }
        let close = Rect::from_center_size(egui::pos2(row.right() - 16.0, row.center().y), egui::vec2(22.0, 22.0));
        let right = if hovered { close.left() - 4.0 } else { row.right() - 10.0 };
        let icon = Rect::from_center_size(egui::pos2(row.left() + 16.0, row.center().y), egui::vec2(16.0, 16.0));
        crate::icons::paint(ui, icon, "file", 14.0, t.text_dim);
        let x = row.left() + 32.0;
        let name_g = crate::tab_strip::elided(ui, &name, egui::FontId::proportional(12.5), t.text, 170.0);
        let name_w = name_g.size().x;
        ui.painter().galley(egui::pos2(x, row.center().y - name_g.size().y / 2.0), name_g, t.text);
        let folder_max = (right - (x + name_w + 14.0)).max(0.0);
        if folder_max > 20.0 && !folder.is_empty() {
            let fg = crate::tab_strip::elided(ui, &folder, egui::FontId::proportional(11.5), t.text_faint, folder_max);
            ui.painter().galley(egui::pos2(right - fg.size().x, row.center().y - fg.size().y / 2.0), fg, t.text_faint);
        }
        resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, &name));
        let resp = resp.on_hover_text(path).on_hover_cursor(egui::CursorIcon::PointingHand);
        if hovered {
            let x = ui.interact(close, resp.id.with("remove"), Sense::click());
            let label = crate::i18n::fmt(tl!("Remove {name} from Recent"), &[("name", &name)]);
            x.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, &label));
            let color = crate::icons::button_chrome(ui, close, false, x.hovered());
            crate::icons::paint(ui, close, "x", 12.0, color);
            if x.on_hover_text(tl!("Remove from Recent")).clicked() {
                remove = Some(path.clone());
            }
        }
        if resp.clicked() && remove.is_none() {
            open = Some(path.clone());
        }
        resp.context_menu(|ui| {
            if ui.button(tl!("Remove from Recent")).clicked() {
                remove = Some(path.clone());
                ui.close();
            }
            if ui.button(tl!("Clear Recent Files")).clicked() {
                clear = true;
                ui.close();
            }
        });
    }
    if clear {
        app.clear_recent();
    } else if let Some(path) = remove {
        app.remove_recent(&path);
    }
    if let Some(path) = open
        && let Err(e) = app.open_path(&path)
    {
        app.open_failed(&crate::file_open::display_name(&path), &e);
    }
}

/// Lattice size of the canvas display LUT (colour management, Proof Colors, Gamut Warning).
const DISPLAY_LUT: usize = 33;

/// Keep the GPU display LUT under `key` (the document's texture or a preview of it) in step with
/// `doc`'s colour management: document → monitor profile and View › Proof Colors / Gamut
/// Warning (the 32-bit preview is applied by the canvas shader, see [`hdr_preview`]). Returns the canvas `display` mode (0 none — the identity, e.g. sRGB on
/// an sRGB monitor —, 1 LUT, 2 LUT + gamut warning).
fn sync_display_lut(app: &mut PhotocraftApp, doc: &photocraft_doc::Document, key: u64, display: Option<u32>) -> u8 {
    let Some(gpu) = app.gpu.clone() else { return 0 };
    let output = display.unwrap_or(0);
    // Rebuild only when anything feeding the LUT changes.
    let sig = app.session.color.display_signature_for(doc, display);
    if let Some((s, mode)) = gpu.display_lut_signature(key, output)
        && s == sig
    {
        return mode;
    }
    let gamut = app.session.color.proof(doc.id).gamut_warning;
    let mode = match app.session.color.gpu_canvas_lut_for(doc, DISPLAY_LUT, display) {
        Ok(Some(bytes)) => {
            gpu.set_display_lut(key, output, DISPLAY_LUT as u32, Some(&bytes));
            if gamut { 2 } else { 1 }
        }
        Ok(None) => {
            gpu.set_display_lut(key, output, DISPLAY_LUT as u32, None);
            0
        }
        Err(e) => {
            let template = "Color management: {error}";
            app.ui.status = crate::i18n::fmt(tl!(template), &[("error", &e.to_string())]);
            gpu.set_display_lut(key, output, DISPLAY_LUT as u32, None);
            0
        }
    };
    gpu.cache_display_lut_signature(key, output, sig, mode);
    mode
}

/// View › 32-bit Preview Options for the GPU canvas shader: (exposure, gamma) when active.
fn hdr_preview(app: &PhotocraftApp, doc: &photocraft_doc::Document) -> Option<[f32; 2]> {
    app.session.color.hdr_preview(doc).map(|h| [h.exposure, h.gamma])
}

/// Draw one canvas view and handle its input. `primary` = main window (tools active).
pub fn canvas_view(app: &mut PhotocraftApp, ui: &mut egui::Ui, idx: usize, rect: Rect, mut view: View, primary: bool) -> View {
    let ctx = ui.ctx().clone();
    // The camera as the app holds it before this frame's input. The canvas works on its own copy
    // (`view`) and writes it back at the end; a tool that moves the camera itself meanwhile (the
    // Crop tool's default mode pans and turns the view with the image, `crop_mode`) must not be
    // undone by that write-back.
    let camera = if primary { app.ui.views.get(idx).map(|v| (v.center, v.rotation)) } else { None };
    // `View::zoom` is device pixels per document pixel; this canvas's geometry is in its own
    // viewport's egui points. A document window can sit on another display with a different
    // scale, so only the primary canvas records `app.ppp` for the frame's shared helpers
    // (`point_zoom`, the navigator, control-channel coordinate mapping). `pixels_per_point`
    // folds in the display scale and Interface › UI Scale alike, so dividing by it keeps 100%
    // one document pixel per physical display pixel at any UI scale (#2121).
    let ppp = ctx.pixels_per_point();
    if primary {
        app.ppp = ppp;
    }
    // The display this window is on: its monitor profile (#569). A document window on a display
    // the last reading didn't know asks for a new one.
    let output = crate::monitor_status::view_display(app, &ctx);
    if !primary {
        crate::monitor_status::check_window(app, &ctx);
    }
    let full = rect;
    let rect = if primary { crate::rulers::content_rect(app, rect) } else { rect };
    let doc = app.session.documents()[idx].doc.clone();
    let size = [doc.size.width, doc.size.height];
    if view.doc_size != size {
        // Like Photoshop: keep the zoom level, re-centre the resized document.
        if view.doc_size != [0, 0] {
            view.center = [size[0] as f32 / 2.0, size[1] as f32 / 2.0];
        }
        view.doc_size = size;
    }
    // View › Pixel Aspect Ratio Correction: document pixels are shown this much wider.
    let aspect = app.ui.view.display_aspect();
    if view.fit_pending && rect.width() > 50.0 {
        fit_view(&mut view, &doc, rect.size(), ppp, aspect);
    }
    if view.fill_pending && rect.width() > 50.0 {
        fill_view(&mut view, &doc, rect.size(), ppp, aspect);
    }
    // Preferences › Tools › Overscroll off: clamp before anything is drawn (scrollbars.rs).
    if !app.session.prefs().tools.overscroll && crate::scrollbars::clamp_view_with_aspect(&mut view, rect.size(), ppp, aspect) {
        ctx.request_repaint();
    }
    let flip = app.ui.view.flip_horizontal;
    // Canvas geometry (ViewXform, ViewParams, painter rects) is in egui points.
    let point_zoom = view.zoom / ppp;
    let xf = ViewXform { rect, zoom: point_zoom, center: view.center, flip, rotation: view.rotation, aspect };
    let pixel_grid = app.ui.view.shows(app.ui.view.show.pixel_grid);
    let response = ui.allocate_rect(rect, Sense::click_and_drag());
    let stroke_dismissal = primary && crate::shape_stroke_ui::dismiss_from_canvas(app, &response);
    let painter = ui.painter_at(rect);

    match crate::prefs_ui::pasteboard_color(app) {
        Some(c) => {
            ui.painter_at(rect).rect_filled(rect, 0.0, c);
        }
        None => paint_dots(ui, rect, app.session.prefs().interface.show_canvas_dots),
    }
    let border = app.session.prefs().interface.canvas_border;
    // Default-mode Crop changes its camera during input. Defer its image until the new camera
    // is known; other tools keep their existing paint order and texture/outline snapshots.
    let crop_camera = primary && crate::crop_mode::moves_image(app);
    let paint_canvas = |app: &mut PhotocraftApp, ui: &egui::Ui, view: &View, xf: ViewXform| {
        let point_zoom = view.zoom / ppp;
        let drop_shadow = border == photocraft_engine::prefs::CanvasBorder::DropShadow;
        // Drop shadow, checkerboard, document image.
        let img_rect = xf.doc_rect(doc.bounds());
        // Crop tool: the layers' pixels past the canvas while a frame is edited, under the canvas
        // (whose edge blends into them) and without its shadow.
        let beyond = draw_beyond_canvas(app, &painter, &xf, idx, output);
        // View › Pattern Preview (#1067): copies of the document around it, from the same texture.
        let pattern = if app.ui.view.pattern_preview {
            let v = visible_doc_rect(&xf);
            crate::pattern_preview::copies([v.x0, v.y0, v.x1, v.y1].map(f64::from), size)
        } else {
            Vec::new()
        };
        let drop_shadow = drop_shadow && !beyond && !app.ui.view.pattern_preview;
        // Live adjustment previews on big documents use a downsampled proxy (see proxy.rs), unless
        // Preferences › Performance › Low Resolution Previews is off.
        let mut on_gpu = false;
        // A flipped or pixel-aspect-corrected view draws through the CPU path (the GPU canvas shader
        // has no mirroring or non-square pixels). Rotation stays on the GPU: it is a view uniform.
        let gpu_ok = crate::gpu_canvas::gpu_view_ok(flip, view.rotation, aspect);
        if app.gpu.is_some()
            && gpu_ok
            && let Some((k, key, preview_size)) =
                ensure_adjust_proxy(app, idx, view.zoom).or_else(|| ensure_filter_preview(app, idx, &ctx)).or_else(|| ensure_proxy_preview(app, idx))
        {
            on_gpu = true;
            app.navigator_gpu.remove(&doc.id);
            let original_size = [doc.size.width.div_ceil(k), doc.size.height.div_ceil(k)];
            let params = crate::gpu_canvas::ViewParams {
                doc: key,
                doc_size: preview_size,
                zoom: view.zoom * k as f32 / ppp,
                // A size-changing preview grows around the old image center; keep the view's pan.
                center: [
                    view.center[0] / k as f32 + (preview_size[0] as f32 - original_size[0] as f32) / 2.0,
                    view.center[1] / k as f32 + (preview_size[1] as f32 - original_size[1] as f32) / 2.0,
                ],
                rotation: if view.rotation.is_finite() { view.rotation.to_radians() } else { 0.0 },
                shadow: {
                    let t = crate::theme::Tokens::get(&ctx);
                    !t.bevel && !t.pro && drop_shadow
                },
                pixel_grid: false,
                view_key: egui::Id::new(("pc-canvas-proxy", ctx.viewport_id(), idx)).value(),
                display: sync_display_lut(app, &doc, key, output),
                output: output.unwrap_or(0),
                hdr: hdr_preview(app, &doc),
                offset: [0.0, 0.0],
            };
            paint_gpu_view(&painter, rect, params, &pattern);
        } else if gpu_ok && ensure_gpu(app, &ctx, idx, visible_doc_rect(&xf)) {
            on_gpu = true;
            app.perf.gpu = true;
            // Shadow, checkerboard, document and pixel grid in one custom shader (gpu_canvas.rs).
            let params = crate::gpu_canvas::ViewParams {
                doc: doc.id.0,
                doc_size: [doc.size.width, doc.size.height],
                zoom: point_zoom,
                center: view.center,
                rotation: if view.rotation.is_finite() { view.rotation.to_radians() } else { 0.0 },
                shadow: {
                    let t = crate::theme::Tokens::get(&ctx);
                    !t.bevel && !t.pro && drop_shadow
                },
                pixel_grid,
                view_key: egui::Id::new(("pc-canvas", ctx.viewport_id(), idx)).value(),
                display: sync_display_lut(app, &doc, doc.id.0, output),
                output: output.unwrap_or(0),
                hdr: hdr_preview(app, &doc),
                offset: [0.0, 0.0],
            };
            if app.navigator_plain.contains(&doc.id) {
                app.navigator_gpu.insert(doc.id, params);
            } else {
                app.navigator_gpu.remove(&doc.id);
            }
            paint_gpu_view(&painter, rect, params, &pattern);
        } else {
            app.navigator_gpu.remove(&doc.id);
            if !crate::theme::Tokens::get(&ctx).bevel && drop_shadow {
                painter.add(egui::Shadow { offset: [0, 8], blur: 28, spread: 0, color: Color32::from_black_alpha(150) }.as_shape(img_rect, 0));
            }
            let rotated = view.rotation.abs() > 0.05;
            let checkerboard = app.session.prefs().transparency_and_gamut.square().map(|square| (checker(app, &ctx), square));
            let tex = ensure_texture(app, &ctx, idx, output).map(|(tex, _scale)| tex);
            paint_cpu_document(&painter, &xf, doc.bounds(), rotated, checkerboard, tex);
            for &c in &pattern {
                paint_cpu_document(&painter, &pattern_xf(&xf, c, size), doc.bounds(), rotated, checkerboard, tex);
            }
        }
        // Channels panel: alpha / Quick Mask overlays and single-channel views (channel_view.rs).
        if let Some(tex) = crate::channel_view::ensure(app, &ctx, idx) {
            if view.rotation.abs() > 0.05 {
                paint_mapped_image(
                    &painter,
                    &xf,
                    tex,
                    Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)),
                    [doc.size.width as f32, doc.size.height as f32],
                    Color32::WHITE,
                );
            } else {
                let uv = if flip { Rect::from_min_max(pos2(1.0, 0.0), pos2(0.0, 1.0)) } else { Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)) };
                painter.image(tex, img_rect, uv, Color32::WHITE);
            }
        }
        crate::color_range_ui::paint_selection_preview(app, &ctx, &painter, doc.id, img_rect, flip);
        // Artboards: pasteboard between the boards, outlines and names (artboard_ui.rs).
        if doc.has_artboards() {
            let t = crate::theme::Tokens::get(&ctx);
            let pasteboard = crate::prefs_ui::pasteboard_color(app);
            let dot_tex = dots(&ctx, &t);
            for r in crate::artboard_ui::pasteboard_rects(&xf, &doc) {
                let r = r.intersect(rect);
                if !r.is_positive() {
                    continue;
                }
                match pasteboard {
                    Some(c) => {
                        painter.rect_filled(r, 0.0, c);
                    }
                    None => {
                        painter.rect_filled(r, 0.0, t.canvas);
                        if let Some(id) = dot_tex {
                            // Same phase as the dots around the document.
                            let uv = Rect::from_min_max(((r.min - rect.min) / 22.0).to_pos2(), ((r.max - rect.min) / 22.0).to_pos2());
                            painter.image(id, r, uv, Color32::WHITE);
                        }
                    }
                }
            }
            if primary {
                crate::artboard_ui::draw_frames(app, &painter, &xf);
            }
        }

        // Pixel grid at high zoom, over pixels with content only (the GPU path draws its own).
        if !on_gpu && pixel_grid && crate::pixel_grid::shows_at(view.zoom) {
            crate::pixel_grid::paint(app, &painter, &xf, idx, output, visible_doc_rect(&xf));
        }

        // Developer tile grid (`ui.set {"tileGrid": true}`): the active layer's storage tiles.
        // Not tied to View › Extras (⌘H), which hides only the View › Show overlays.
        if app.ui.view.tile_grid {
            crate::tile_overlay::paint(app, &ctx, &painter, &xf, idx, visible_doc_rect(&xf));
        }

        // View › Show › Layer Edges: drawn at this depth, but only once this frame's tool events are
        // in (`layer_edges_rect`), so a press that Auto-Selects a layer outlines that layer.
        let layer_edges = painter.add(egui::Shape::Noop);

        // Select and Mask outlines its refined selection (on its preview proxy, reduced by `k`) in
        // the Marching Ants view and hides the ants in the others.
        let refined = crate::select_mask_view::ants(app, idx);
        let (outlined, k) = match &refined {
            None => (doc.selection.as_ref(), 1),
            Some(None) => (None, 1),
            Some(Some((proxy, k))) => (proxy.selection.as_ref(), (*k).max(1)),
        };
        // Selection outline: true boundary, animated marching ants (cached per revision).
        if let Some(sel) = outlined.filter(|_| app.ui.view.shows(app.ui.view.show.selection_edges) && !polygon_replaces_selection(app)) {
            // Trace at display resolution over the visible part only; key by the mask's tile identity
            // (not the document revision) so unrelated edits don't re-trace it. The step is in
            // device pixels per document pixel (`View::zoom`, not the point zoom), so a 1 px
            // selection still traces at 100% on a scaled display.
            let step = (1.0 / (view.zoom * k as f32).max(1e-3)).log2().floor().exp2().clamp(1.0, 64.0) as u32;
            let corners = [rect.min, pos2(rect.max.x, rect.min.y), rect.max, pos2(rect.min.x, rect.max.y)];
            let docs = [xf.to_doc(corners[0]), xf.to_doc(corners[1]), xf.to_doc(corners[2]), xf.to_doc(corners[3])];
            let mut x0 = docs[0][0];
            let mut y0 = docs[0][1];
            let mut x1 = x0;
            let mut y1 = y0;
            for d in docs {
                x0 = x0.min(d[0]);
                y0 = y0.min(d[1]);
                x1 = x1.max(d[0]);
                y1 = y1.max(d[1]);
            }
            let (x0, y0, x1, y1) = (x0 / f64::from(k), y0 / f64::from(k), x1 / f64::from(k), y1 / f64::from(k));
            let q = 256 * step as i32; // quantise the region so small pans reuse the cache
            // The view centre can be any finite number (`ui.set`, #980). Past ±2³⁰ there is no document
            // to outline, and the padding below (at most 2 · 256 · 64) can't overflow i32 from there.
            let px = |v: f64| v.clamp(-f64::from(1 << 30), f64::from(1 << 30)) as i32;
            let vis = photocraft_geom::Rect::new(
                px(x0.floor()).div_euclid(q) * q - q,
                px(y0.floor()).div_euclid(q) * q - q,
                (px(x1.ceil()).div_euclid(q) + 2) * q,
                (px(y1.ceil()).div_euclid(q) + 2) * q,
            );
            let key = crate::surface_fingerprint(sel)
                ^ (step as u64) << 56
                ^ doc.id.0.rotate_left(17)
                ^ (vis.x0 as u64) << 8
                ^ (vis.y0 as u64) << 24
                ^ (vis.x1 as u64) << 36
                ^ (vis.y1 as u64) << 48
                ^ u64::from(k).rotate_left(5);
            let fresh = matches!(&app.outline_cache, Some((d, k, _)) if *d == doc.id && *k == key);
            if !fresh {
                let t0 = crate::gpu_canvas::now_ms();
                let bounds_key = if k > 1 { u64::MAX / 2 - doc.id.0 } else { u64::MAX - doc.id.0 };
                let b = app.cached_bounds(bounds_key, sel).intersect(&vis);
                // Filters can occupy every worker in Rayon's global pool. A parallel scan here would
                // block the UI thread waiting for those workers, preventing job cancellation (#1554).
                let mut segs =
                    if app.session.has_jobs() { crate::outline::outline_scaled_serial(sel, b, step) } else { crate::outline::outline_scaled(sel, b, step) };
                if k > 1 {
                    let s = i32::try_from(k).unwrap_or(1);
                    let up = |p: [i32; 2]| [p[0].saturating_mul(s), p[1].saturating_mul(s)];
                    segs.iter_mut().for_each(|(a, b)| (*a, *b) = (up(*a), up(*b)));
                }
                app.perf.span("outline", crate::gpu_canvas::now_ms() - t0);
                app.outline_cache = Some((doc.id, key, std::sync::Arc::new(segs)));
            }
            // Under Content-Aware Move's Transform On Drop box the ants are drawn where it puts them.
            if let Some((_, _, segs)) = app.outline_cache.as_ref().filter(|_| !crate::cam_transform::active(app)) {
                let time = ui.input(|i| i.time);
                // A floating piece (and a selection being dragged) draws its ants where it is shown.
                let mut shown = xf;
                if let Some((dx, dy)) = selection_shown_offset(app) {
                    shown.center = [xf.center[0] - dx as f32, xf.center[1] - dy as f32];
                }
                marching_ants_segments(&painter, &shown, segs, time);
                ctx.request_repaint_after(std::time::Duration::from_millis(100));
            }
        }

        (img_rect, beyond, layer_edges)
    };
    let painted = if crop_camera { None } else { Some(paint_canvas(app, ui, &view, xf)) };
    let img_rect = xf.doc_rect(doc.bounds());

    // Under an open dialog the canvas widget is inert, but the image still pans and zooms.
    let under_dialog = !app.ui.dialogs.is_empty();
    let free_hover = under_dialog && crate::dialogs::free_pointer_over(&ctx, rect).is_some();
    // A press on the canvas while a tool letter is held makes that tool temporary (#2672).
    if !under_dialog && ctx.input(|i| i.pointer.any_pressed() && i.pointer.interact_pos().is_some_and(|p| rect.contains(p))) {
        crate::spring_tools::note_canvas_press(app);
    }
    // Navigation (wheel_nav.rs): scroll pans (⇧ or ⌘/Ctrl sideways); pinch and ⌥-scroll zoom around the pointer.
    let wheel =
        crate::wheel_nav::read(&ctx, app.session.prefs().general.zoom_with_scroll_wheel, app.session.prefs().enhanced_controls.zoom_with_trackpad_pinch);
    // The wheel also scrolls over the scrollbars drawn on top of the canvas (last frame's hover).
    let bars_id = egui::Id::new(("pc-canvas-bars-hover", idx));
    let over_bars = ctx.data(|d| d.get_temp::<bool>(bars_id)).unwrap_or(false);
    if response.hovered() || free_hover || over_bars {
        let pointer = ui.input(|i| i.pointer.hover_pos());
        match (wheel, pointer) {
            (Some(crate::wheel_nav::Wheel::Zoom(f)), Some(p)) => {
                // At a limit the view stays put, as in Photoshop: a notch past 12800 % neither
                // zooms nor slides the image towards the pointer.
                let nz = crate::zoom_levels::clamp(view.zoom * f, view.doc_size);
                if nz != view.zoom {
                    // Over the pasteboard the anchor is the image's nearest point (#3720).
                    let at = wheel_anchor(&xf, p, view.doc_size);
                    zoom_about(&mut view, &xf, at, nz, false, ppp);
                }
            }
            (Some(crate::wheel_nav::Wheel::Pan(scroll)), _) => {
                let d = xf.unmap_vec(scroll) / (view.zoom / ppp);
                view.center[0] -= d.x;
                view.center[1] -= d.y;
            }
            _ => {}
        }
        let rotate_trackpad = app.session.prefs().enhanced_controls.rotate_view_with_trackpad;
        if let Some(delta) = crate::rotate_view::trackpad_delta(&ctx, rotate_trackpad) {
            let next = view.rotation + crate::rotate_view::screen_delta(delta, xf.flip);
            if next.is_finite() {
                view.rotation = crate::rotate_view::wrap_deg(next);
            }
        }
    }

    // Pen pressure/tilt for this frame's tool events (mouse = 1.0), unless Preferences › Tools ›
    // Use Tablet Pressure is off; the pen's eraser end selects the Eraser.
    app.stylus.use_pressure = app.session.prefs().tools.use_tablet_pressure;
    let curve = app.session.prefs().tools.pressure_curve.clone();
    app.stylus.set_pressure_curve(&curve);
    app.stylus.update_for_frame(ui.ctx().cumulative_frame_nr(), &ui.input(|i| i.events.clone()));
    crate::stylus::Stylus::sync_eraser_tool(app);
    // A Magnetic Lasso border left behind by a tool or document switch is dropped.
    crate::magnetic_lasso_ui::frame(app);
    // A Properties Levels/Curves eyedropper left behind by a tool or layer switch is dropped.
    crate::layer_picker::prune(app);
    // Held keys (hold_keys.rs): Space repositions a crop frame, marquee, lasso or shape being
    // drawn; otherwise Space is the Hand and ⌘Space / ⌥Space the Zoom tool while held.
    let reposition = crate::hold_keys::reposition_held(app, &ctx);
    crate::crop_ui::set_space(app, reposition);
    // ⌘ (Ctrl) held: a temporary Straighten with the Crop tool.
    crate::crop_straighten::set_held(app, !ctx.text_edit_focused() && ui.input(|i| i.modifiers.command));
    // A tab switched this frame drops the other document's pending crop (#1918).
    crate::crop_ui::cancel_stale(app);
    crate::crop_ui::ensure_frame(app);
    let mut drawing = crate::crop_ui::active(app);
    if let Some(d) = app.drag.as_mut().filter(|d| crate::hold_keys::repositions(d.tool)) {
        d.reposition = reposition;
        drawing = true;
    }
    let temporary = crate::hold_keys::for_frame(app, &ctx, drawing || crate::type_transform::active(app));
    let space_pan = temporary == Some(crate::hold_keys::Temporary::Hand);
    let middle = ui.input(|i| i.pointer.middle_down());
    let tool = match temporary {
        Some(t) => t.tool(),
        None if middle => Tool::Hand,
        None => app.ui.tool,
    };
    // ⌘ held: this frame's gestures go to the Move tool (`PhotocraftApp::active_tool`). The Hand
    // and Zoom never reach `tool_event`, so only the Move needs the state machine to know.
    app.tool_override = (temporary == Some(crate::hold_keys::Temporary::Move)).then_some(Tool::Move);
    // `~` held: a Brush or Pencil stroke that starts now erases (`tool_event`'s `Down`).
    app.held_erase = crate::hold_keys::erase_held(app, &ctx);
    // Zoom direction: the temporary zoom key decides, else ⌥ (Zoom tool), held or latched in
    // Window › Modifier Keys (the way around a desktop that takes Alt+click, `alt_grab`).
    let sticky_alt = app.ui.shell.sticky_alt;
    let zoom_out = |alt: bool| match temporary {
        Some(crate::hold_keys::Temporary::ZoomOut) => true,
        Some(crate::hold_keys::Temporary::ZoomIn) => false,
        _ => alt || sticky_alt,
    };

    if under_dialog {
        // With a colour dialog picker armed the image is its eyedropper, whatever the tool; Space
        // and the middle button still pan. Curves and Replace Color use the same merged-composite
        // sampler as the Color Picker, never the reduced adjustment preview.
        let curves_picking = crate::adjust_dialog::picker_armed(app);
        let replace_picking = crate::filter_dialog::color_eyedropper_armed(app);
        let picking = primary && (crate::color_picker_ui::top(app).is_some() || curves_picking || replace_picking);
        let range_picking =
            primary && app.ui.dialogs.last().is_some_and(|d| crate::color_range_ui::owns(&d.fields) && crate::color_range_ui::controls(&d.fields).sampling);
        // Select and Mask's Refine Edge Brush paints on the image.
        let refine_brushing =
            primary && app.ui.dialogs.last().is_some_and(|d| crate::select_mask_view::owns(&d.fields) && crate::select_mask_view::brush_on(&d.fields));
        // Field Blur's pins are placed and dragged on the image (`blur_pins`).
        let pinning = primary && app.ui.dialogs.last().is_some_and(|d| crate::blur_pins::owns(&d.fields));
        let hand = app.ui.tool == Tool::Hand && !picking && !range_picking && !refine_brushing && !pinning;
        if let Some(d) = crate::dialogs::pan_delta(&ctx, rect, hand) {
            let d = xf.unmap_vec(d) / (view.zoom / ppp);
            view.center[0] -= d.x;
            view.center[1] -= d.y;
            ctx.set_cursor_icon(crate::tool_cursor::hand(&ctx, true));
        } else if free_hover && (space_pan || hand) {
            ctx.set_cursor_icon(crate::tool_cursor::hand(&ctx, false));
        } else if picking && let Some(p) = crate::dialogs::free_pointer_over(&ctx, rect) {
            if app.session.prefs().cursors.other == photocraft_engine::prefs::OtherCursor::Precise {
                ctx.set_cursor_icon(egui::CursorIcon::Crosshair);
            } else {
                ctx.set_cursor_icon(pipette_cursor(&ctx, p));
            }
            if let Some(p) = crate::dialogs::free_press(&ctx, rect) {
                let d = xf.to_doc(p);
                if curves_picking {
                    crate::adjust_dialog::sample_at(app, d[0], d[1]);
                } else if replace_picking {
                    crate::filter_dialog::color_eyedropper_sample(app, d[0], d[1]);
                } else {
                    crate::color_picker_ui::sample_at(app, d[0], d[1]);
                }
            }
        } else if pinning
            && !middle
            && let Some(p) = crate::dialogs::free_pointer_over(&ctx, rect)
        {
            let press = crate::dialogs::free_press(&ctx, rect);
            crate::blur_pins::canvas(app, &ctx, &xf, p, press);
        } else if refine_brushing
            && !middle
            && let Some(p) = crate::dialogs::free_pointer_over(&ctx, rect)
        {
            let press = crate::dialogs::free_press(&ctx, rect).map(|q| xf.to_doc(q));
            crate::select_mask_view::canvas_brush(app, &painter, p, press, xf.zoom_x());
        } else if primary
            && !middle
            && let Some(p) = crate::dialogs::free_pointer_over(&ctx, rect).filter(|p| img_rect.contains(*p))
        {
            // Color Range samples colours with its eyedropper on the image itself.
            let press = crate::dialogs::free_press(&ctx, rect)
                .filter(|p| img_rect.contains(*p) && ctx.input(|i| i.pointer.primary_pressed() || i.pointer.delta() != egui::Vec2::ZERO))
                .map(|q| xf.to_doc(q));
            crate::color_range_ui::canvas_eyedropper(app, &ctx, p, press);
        }
    }
    // The selected Hand yields to an open Free Transform box in this (primary) window: the box
    // owns the pointer, so the tool behind it cannot steal the resize drag (#2302). The held Hand
    // (Space) and a middle-button drag still pan, and a secondary window's Hand never stops, as the
    // box is only drawn and driven on the primary canvas.
    let transform_owns_pointer = primary && app.ui.transform.is_some() && temporary.is_none() && !middle;
    let hand_pan = tool == Tool::Hand && !transform_owns_pointer;
    // Preferences › Tools › Enable Flick Panning: a Hand drag let go while moving glides on.
    let flick_id = egui::Id::new(("pc-flick-pan", idx, primary, ctx.viewport_id()));
    let flick_on = app.session.prefs().tools.enable_flick_panning;
    if let Some(d) = crate::flick_pan::frame(&ctx, flick_id, hand_pan && response.dragged() && !stroke_dismissal, response.drag_delta(), flick_on) {
        let d = xf.unmap_vec(d) / (view.zoom / ppp);
        if d.is_finite() {
            view.center[0] -= d.x;
            view.center[1] -= d.y;
        }
        // Overscroll off: the glide stops at the pan limit, as the scrollbars do.
        if !app.session.prefs().tools.overscroll && crate::scrollbars::clamp_view_with_aspect(&mut view, rect.size(), ppp, aspect) {
            ctx.data_mut(|m| m.get_temp_mut_or_default::<crate::flick_pan::Flick>(flick_id).stop());
        }
    }
    if hand_pan && response.dragged() && !stroke_dismissal {
        let d = xf.unmap_vec(response.drag_delta()) / (view.zoom / ppp);
        view.center[0] -= d.x;
        view.center[1] -= d.y;
        // The tool cursors below are skipped while panning: the fist is set here (#2196).
        ctx.set_cursor_icon(crate::tool_cursor::hand(&ctx, true));
    } else if primary && !stroke_dismissal && temporary.is_none() && !middle && crate::layer_picker::armed(app).is_some() {
        // A Properties Levels/Curves eyedropper (layer_picker.rs) owns the image until Esc or another
        // tool; Space and the middle button still pan. It samples on the press.
        if response.hovered()
            && let Some(p) = ui.input(|i| i.pointer.hover_pos())
        {
            if app.session.prefs().cursors.other == photocraft_engine::prefs::OtherCursor::Precise {
                ctx.set_cursor_icon(egui::CursorIcon::Crosshair);
            } else {
                ctx.set_cursor_icon(pipette_cursor(&ctx, p));
            }
        }
        if response.is_pointer_button_down_on()
            && ui.input(|i| i.pointer.primary_pressed())
            && let Some(p) = ui.input(|i| i.pointer.interact_pos())
        {
            let d = xf.to_doc(p);
            crate::layer_picker::sample_at(app, d[0], d[1]);
        }
    } else if primary && !stroke_dismissal {
        let mods = ui.input(|i| i.modifiers);
        // An ⌥-click the window manager took never arrives; say so (alt_grab.rs).
        crate::alt_grab::watch(app, &ctx, egui::Id::new(("pc-alt-grab", idx)), rect, tool, !under_dialog);
        // Tools follow the left button; the right one opens the Brush Preset picker or erases
        // (Preferences › Tools, `paint_mouse`).
        crate::paint_mouse::sync_tool_brush(app);
        let mut buttons = crate::paint_mouse::canvas_buttons(app, &response, tool);
        // The Crop tool's frame is edited from the press (before egui's drag threshold), so the
        // pixels past the canvas show at once, as in Photoshop.
        if tool == Tool::Crop && response.is_pointer_button_down_on() && ui.input(|i| i.pointer.primary_down()) && crate::crop_ui::press(app) {
            ctx.request_repaint();
        }
        // Capture intent at the press, before egui's drag threshold: Type transforms keep
        // Command, and marquee add/subtract must not become a live Shift/Alt constraint.
        let capture_press = tool.is_type() || matches!(tool, Tool::RectMarquee | Tool::EllipseMarquee | Tool::SingleRowMarquee | Tool::SingleColumnMarquee);
        let tool_press = egui::Id::new(("tool-pointer-press-modifiers", idx));
        if capture_press
            && let Some((p, press_mods)) = ui.input(|i| {
                i.events.iter().find_map(|e| match e {
                    egui::Event::PointerButton { pos, button: PointerButton::Primary, pressed: true, modifiers } if rect.contains(*pos) => {
                        Some((*pos, *modifiers))
                    }
                    _ => None,
                })
            })
        {
            let press_mods = crate::workspace_ui::sticky_mods(app, press_mods);
            ctx.data_mut(|d| d.insert_temp(tool_press, press_mods));
            if tool.is_type() && press_mods.command && app.ui.text_edit.is_some() {
                let d = xf.to_doc(p);
                tool_event(app, ToolEvent::Down { x: d[0], y: d[1], pressure: app.stylus.pressure() }, press_mods);
            }
        }
        // Right-click while transforming: switch the box's mode (Free Transform, Scale, Rotate,
        // Skew, Distort, Perspective, Warp), or rotate or flip the box.
        let transforming = app.ui.transform.as_ref().is_some_and(|t| t.warp.is_none());
        if response.secondary_clicked()
            && transforming
            && let Some(p) = response.interact_pointer_pos()
        {
            crate::canvas_tool_menu::open_transform(app, &ctx, [p.x, p.y]);
        }
        let lasso_retracted = response.secondary_clicked()
            && match tool {
                Tool::Lasso | Tool::FreeformPen => crate::lasso_ui::undo_last_vertex(app),
                Tool::PolygonLasso => polygon_retract(app),
                _ => false,
            };
        // The Lasso (and the Freeform Pen, which draws the same way) reads its own presses and
        // drops the moves of a drag it didn't start, so an open Free Transform box takes the
        // usual drag path instead (#2153).
        if crate::lasso_ui::owns(tool) && !transform_owns_pointer {
            crate::lasso_ui::canvas_input(app, &ctx, &xf, &response);
            (buttons.started, buttons.dragged, buttons.stopped, buttons.clicked) = (false, false, false, false);
        }
        // Right-click with the Move tool, or ⌘/Ctrl+right-click: the layers under the pointer.
        if response.secondary_clicked()
            && !lasso_retracted
            && !transforming
            && crate::layer_pick_ui::is_gesture(tool, mods)
            && let Some(p) = response.interact_pointer_pos()
        {
            let d = xf.to_doc(p);
            app.ui.canvas_tool_menu = None;
            // No layers there (the pasteboard, with the Move tool): its colour menu instead.
            if !crate::layer_pick_ui::open(app, &ctx, [p.x, p.y], d[0], d[1]) && !mods.command && crate::canvas_tool_menu::pasteboard_at(app, tool, d) {
                crate::canvas_tool_menu::open_pasteboard(app, &ctx, tool, [p.x, p.y]);
            }
        }
        if response.secondary_clicked()
            && !lasso_retracted
            && !transforming
            && !crate::layer_pick_ui::is_gesture(tool, mods)
            && let Some(p) = response.interact_pointer_pos()
        {
            // Outside the image: the pasteboard colour menu (#2462), else the tool's own. The view
            // menu belongs to the selected Zoom or Hand, not to Space or a held zoom key.
            if crate::canvas_tool_menu::pasteboard_at(app, tool, xf.to_doc(p)) {
                crate::canvas_tool_menu::open_pasteboard(app, &ctx, tool, [p.x, p.y]);
            } else if crate::canvas_tool_menu::is_view_tool(tool) {
                if tool == app.ui.tool {
                    crate::canvas_tool_menu::open_view(app, &ctx, tool, [p.x, p.y], xf.to_doc(p));
                }
            } else {
                crate::canvas_tool_menu::open_at(app, &ctx, tool, [p.x, p.y], Some(xf.to_doc(p)));
            }
        }
        // The (temporary) Hand pans above; its gestures never reach the tool underneath. While a
        // Free Transform box is open the Hand does not pan, so the box gets these events (#2302).
        if hand_pan {
            (buttons.started, buttons.dragged, buttons.stopped) = (false, false, false);
        }
        // Rotate View: drag around the view centre. Compass clicks are not a rotate drag.
        let over_compass = crate::rotate_view::compass_visible(view.rotation)
            && response.interact_pointer_pos().is_some_and(|p| crate::rotate_view::compass_rect(xf.rect).contains(p));
        if tool == Tool::RotateView && !over_compass && crate::rotate_view::drag(&ctx, &mut view, &xf, &buttons, response.interact_pointer_pos()) {
            (buttons.started, buttons.dragged, buttons.stopped, buttons.clicked) = (false, false, false, false);
        }
        // Zoom tool drags: scrubby zoom or a zoom rectangle (zoom_tool.rs); clicks step below.
        // Like the Hand, the selected Zoom yields to an open Free Transform box (#4799).
        if tool == Tool::Zoom && !transform_owns_pointer && crate::zoom_tool::drag(app, &ctx, &mut view, &xf, &buttons, response.interact_pointer_pos()) {
            (buttons.started, buttons.dragged, buttons.stopped) = (false, false, false);
        }
        // A drag is only recognised once the pointer has moved past egui's click distance: the
        // gesture starts where the button went down, not where it is now (#123).
        let gesture_active_before = app.drag.is_some();
        // Stroke timing: egui's events carry no timestamps, so a press takes the frame time and
        // the moves below are spread between the previous point and this frame (`spread_ms`).
        let now_ms = ui.input(|i| i.time) * 1000.0;
        app.stylus.clock_ms = now_ms;
        // Live painting tools start on the press, not once the pointer passes egui's click
        // distance: the first dab shows at once. The drag recognised later continues that stroke,
        // and a click (or a long press that never moved) just ends it.
        if strokes_live(tool)
            && app.drag.is_none()
            && response.is_pointer_button_down_on()
            && ui.input(|i| i.pointer.primary_pressed())
            && let Some(p) = ui.input(|i| i.pointer.press_origin()).filter(|p| rect.contains(*p))
        {
            let d = xf.to_doc(p);
            // The press's own modifiers: a ⇧ that arrives with the click still connects the line.
            let press_mods = ui.input(|i| pointer_button_modifiers(&i.events, PointerButton::Primary)).unwrap_or(mods);
            tool_event(app, ToolEvent::Down { x: d[0], y: d[1], pressure: app.stylus.pressure() }, press_mods);
            app.press_stroke = app.drag.is_some();
        }
        let press_stroke = app.press_stroke;
        if press_stroke {
            buttons.started = false;
        }
        if buttons.started
            && let Some(p) = ui.input(|i| i.pointer.press_origin()).filter(|p| rect.contains(*p)).or(response.interact_pointer_pos())
        {
            if tool == Tool::Move && app.ui.transform.is_none() {
                begin_transform_controls_at(app, &ctx, &xf, p);
            }
            let d = xf.to_doc(p);
            let press_mods = if capture_press { ctx.data(|d| d.get_temp(tool_press)).unwrap_or(mods) } else { mods };
            tool_event(app, ToolEvent::Down { x: d[0], y: d[1], pressure: app.stylus.pressure() }, press_mods);
        }
        // A stroke started on the press takes its moves from the press on, not only once egui
        // recognises a drag (past its click distance): otherwise the first few points are lost
        // and the stroke starts with a straight segment and no pressure ramp.
        let press_stroke_moving = press_stroke && app.drag.is_some();
        if buttons.dragged || buttons.stopped || press_stroke_moving {
            // Feed every pointer move the OS delivered this frame, not just the latest position, so
            // a fast stroke is sampled densely and renders as a smooth curve instead of a coarse
            // polyline. egui-winit pushes one `PointerMoved` per `CursorMoved`, and they accumulate
            // while a frame is slow, so reading them all recovers the moves a per-frame
            // `interact_pointer_pos()` would drop. Only freehand tools take the whole batch; the
            // rest follow the pointer's latest position. The moves are bounded to the gesture's own
            // press..release interval (`pointer_moves`), so the start and stop frames keep their
            // valid samples without leaking a move from outside the gesture.
            let events = ui.input(|i| i.events.clone());
            let button = if response.dragged_by(PointerButton::Secondary)
                || response.drag_started_by(PointerButton::Secondary)
                || response.drag_stopped_by(PointerButton::Secondary)
            {
                PointerButton::Secondary
            } else {
                PointerButton::Primary
            };
            let press_this_frame = events.iter().any(|e| matches!(e, egui::Event::PointerButton { button: b, pressed: true, .. } if *b == button));
            let down_at_start = !press_this_frame && (gesture_active_before || buttons.started);
            let mut positions = if freehand_tool(tool) {
                pointer_moves(&events, button, down_at_start)
            } else if buttons.dragged {
                response.interact_pointer_pos().into_iter().collect()
            } else {
                Vec::new()
            };
            // A frame with no raw move still tracks a held gesture (a still pointer, a modifier
            // change): fall back to the latest position, as the old code always did.
            if positions.is_empty()
                && (buttons.dragged || press_stroke_moving)
                && let Some(p) = response.interact_pointer_pos().or(ui.input(|i| i.pointer.latest_pos()))
            {
                positions.push(p);
            }
            app.defer_live_stroke = true;
            let base = app.stylus.last_point_ms().map_or(now_ms, |t| t.min(now_ms));
            let n = positions.len();
            for (k, p) in positions.into_iter().enumerate() {
                app.stylus.clock_ms = spread_ms(base, now_ms, k, n);
                // This move's share of the pen samples the frame brought (pressure, tilt, rotation).
                app.stylus.select(k, n);
                let d = xf.to_doc(p);
                tool_event(app, ToolEvent::Move { x: d[0], y: d[1], pressure: app.stylus.pressure() }, mods);
            }
            app.stylus.clear_selection();
            app.stylus.clock_ms = now_ms;
            // A held pen keeps the clock running for airbrush build-up and smoothing catch-up.
            if app.drag.is_some() && stroke_wants_time(app) {
                ctx.request_repaint_after(std::time::Duration::from_millis(HOLD_REPEAT_MS as u64));
            }
            app.defer_live_stroke = false;
            // One live-stroke update for the whole frame, not one per recovered sample.
            feed_live_stroke(app);
            crate::magnetic_lasso_ui::flush(app);
        }
        if buttons.stopped {
            // The live painting press already created the stroke and this release commits it.
            // egui may also report clicked on the same release; replaying that click paints a
            // second full-pressure dab over the real (possibly light-pressure) pen dab.
            if press_stroke {
                buttons.clicked = false;
            }
            let p = response.interact_pointer_pos().map(|p| xf.to_doc(p)).or_else(|| app.drag.as_ref().and_then(|d| d.points.last().map(|q| [q[0], q[1]])));
            if let Some(d) = p {
                tap_at_peak_pressure(app);
                tool_event(app, ToolEvent::Up { x: d[0], y: d[1] }, mods);
            }
        }
        // A stroke started on the press ends with the release, whether egui saw a drag, a click,
        // or neither (a long press that never moved).
        if press_stroke && !buttons.stopped && ui.input(|i| i.pointer.primary_released()) {
            buttons.clicked = false;
            let p = response.interact_pointer_pos().map(|p| xf.to_doc(p)).or_else(|| app.drag.as_ref().and_then(|d| d.points.last().map(|q| [q[0], q[1]])));
            if let Some(d) = p
                && app.drag.is_some()
            {
                tap_at_peak_pressure(app);
                tool_event(app, ToolEvent::Up { x: d[0], y: d[1] }, mods);
            }
        }
        if press_stroke && (app.drag.is_none() || !ui.input(|i| i.pointer.primary_down())) {
            app.press_stroke = false;
        }
        if buttons.clicked
            && let Some(p) = response.interact_pointer_pos()
        {
            let click_button = if app.secondary_erase { PointerButton::Secondary } else { PointerButton::Primary };
            let click_mods = ui.input(|i| pointer_button_modifiers(&i.events, click_button)).unwrap_or(mods);
            let d = xf.to_doc(p);
            match tool {
                Tool::Zoom if !transform_owns_pointer => {
                    let nz = crate::zoom_levels::step(view.zoom, if zoom_out(click_mods.alt) { -1 } else { 1 }, view.doc_size);
                    let center = app.session.prefs().tools.zoom_clicked_point_to_center;
                    zoom_about(&mut view, &xf, p, nz, center, ppp);
                }
                // A click with the (temporary) Hand does nothing, never the tool underneath.
                Tool::Hand | Tool::RotateView => {}
                // Double-clicking closes the polygonal lasso: the first click placed the last
                // vertex (or already closed it), so the second never starts a new polygon. egui
                // reports it as a triple click when a vertex went down shortly before.
                Tool::PolygonLasso if response.double_clicked() || response.triple_clicked() => commit_polygon(app),
                // The same for the Magnetic Lasso: along the edges, or straight with ⌥.
                Tool::MagneticLasso if response.double_clicked() || response.triple_clicked() => crate::magnetic_lasso_ui::close(app, mods.alt),
                // The first click placed (or grabbed) the point; the second toggles it.
                Tool::CurvaturePen if response.double_clicked() => crate::vector_ui::curvature_toggle(app, d[0], d[1]),
                _ => {
                    if tool == Tool::Move && app.ui.transform.is_none() {
                        begin_transform_controls_at(app, &ctx, &xf, p);
                    }
                    // If this tool was not handled on pointer-down, fall back to a
                    // click-sized gesture while preserving pen pressure (mouse stays at 1).
                    tool_event(app, ToolEvent::Down { x: d[0], y: d[1], pressure: app.stylus.pressure() }, click_mods);
                    tap_at_peak_pressure(app);
                    tool_event(app, ToolEvent::Up { x: d[0], y: d[1] }, click_mods);
                }
            }
        }
        // Magnetic Lasso: between clicks the border follows the pointer with the button up.
        if tool == Tool::MagneticLasso && app.ui.magnetic.active() && !(buttons.started || buttons.dragged || buttons.stopped) && response.hovered() {
            let moves: Vec<Pos2> = ui.input(|i| i.events.iter().filter_map(|e| if let egui::Event::PointerMoved(p) = e { Some(*p) } else { None }).collect());
            app.defer_live_stroke = true;
            for p in moves.into_iter().filter(|p| rect.contains(*p)) {
                let d = xf.to_doc(p);
                tool_event(app, ToolEvent::Move { x: d[0], y: d[1], pressure: app.stylus.pressure() }, mods);
            }
            app.defer_live_stroke = false;
            crate::magnetic_lasso_ui::flush(app);
        }
        // Move tool: a double-click on type edits it, as in Photoshop (#1132). Only the selected
        // Move tool, not ⌘ held over another tool; an open transform box keeps its double-click
        // (commit, below).
        if tool == Tool::Move
            && app.ui.tool == Tool::Move
            && app.ui.transform.is_none()
            && response.double_clicked()
            && let Some(p) = response.interact_pointer_pos()
        {
            let d = xf.to_doc(p);
            crate::type_tool::edit_at(app, d[0], d[1]);
        }
        if app.ui.transform.is_some() && response.double_clicked() {
            crate::transform_tool::commit(app);
        }
        if crate::cam_transform::active(app) && response.double_clicked() {
            crate::cam_transform::commit(app);
        }
        // Crop: a double-click inside the frame commits it, as ↵ does (#1792).
        if tool == Tool::Crop && response.double_clicked() && response.interact_pointer_pos().is_some_and(|p| crate::crop_ui::commits_at(app, xf.to_doc(p))) {
            commit_crop(app);
        }
        // Type: 2 clicks select a word, 3 the line, 4 the paragraph, 5 all text (Photoshop).
        if tool.is_type()
            && response.clicked()
            && let Some(p) = response.interact_pointer_pos()
        {
            if crate::type_transform::visible(app, crate::workspace_ui::sticky_mods(app, mods)) {
                app.type_clicks = None;
            } else {
                let n = if response.triple_clicked() {
                    3
                } else if response.double_clicked() {
                    2
                } else {
                    1
                };
                let d = xf.to_doc(p);
                crate::type_tool::multi_click(app, n, d[0], d[1]);
            }
        }
    }
    if crop_camera && let (Some((center, rotation)), Some(now)) = (camera, app.ui.views.get(idx)) {
        if now.center != center {
            view.center = now.center;
            ctx.request_repaint();
        }
        if now.rotation != rotation {
            view.rotation = now.rotation;
            ctx.request_repaint();
        }
    }
    let xf = if crop_camera { ViewXform { center: view.center, rotation: view.rotation, ..xf } } else { xf };
    let (img_rect, beyond, layer_edges) = painted.unwrap_or_else(|| paint_canvas(app, ui, &view, xf));
    if primary && !(hand_pan && response.dragged()) {
        if app.ui.extras.grid && app.ui.view.extras {
            crate::rulers::draw_grid(app, &painter, &xf, &doc);
        }
        if app.ui.view.shows(app.ui.view.show.canvas_guides) || crate::guide_layout_ui::preview(app, &doc).is_some() {
            crate::rulers::draw_guides(app, &painter, &xf, &doc);
        }
        // The pending crop frame shows on its own document only, not on the other tiles (#1918).
        let foreign_crop = if app.session.active_index() == Some(idx) { None } else { app.ui.crop_rect.take() };
        draw_drag_preview(app, &painter, &xf);
        if app.session.active_index() == Some(idx) {
            crate::move_box::draw(app, &painter, &xf);
        }
        if foreign_crop.is_some() {
            app.ui.crop_rect = foreign_crop;
        }
        crate::zoom_tool::draw(&ctx, &painter);
        let resizing = crate::brush_resize::draw(app, &painter, &xf);
        draw_transform_controls(app, &painter, &xf);
        crate::move_hover::draw(app, &painter, &xf, idx, response.hover_pos());
        crate::object_finder::draw(app, &painter, &xf, idx, response.hover_pos());
        crate::layer_pick_ui::show(app, &ctx);
        let shown = app.ui.views.get(idx).cloned();
        crate::canvas_tool_menu::show(app, &ctx);
        // A view menu row (Zoom In, Fit on Screen…) changed the camera: it wins over this
        // frame's copy, which is written back at the end.
        if let Some(now) = app.ui.views.get(idx)
            && Some(now) != shown.as_ref()
        {
            view = now.clone();
            ctx.request_repaint();
        }
        crate::snap_ui::draw(app, &painter, &xf);
        // The canvas edge isn't outlined while the pixels past it show (Photoshop's crop preview),
        // nor while the Crop tool hides everything outside its box (Show Cropped Area off).
        if border == photocraft_engine::prefs::CanvasBorder::Line && !beyond && !crate::crop_shield::hides_outside(app) {
            painter.rect_stroke(img_rect, 0.0, Stroke::new(1.0, Color32::from_gray(20)), egui::StrokeKind::Outside);
        }
        crate::type_tool::draw_overlay(app, &painter, &xf);
        crate::transform_tool::draw_overlay(app, &painter, &xf);
        crate::cam_transform::draw(app, &painter, &xf);
        crate::cam_preview::wake(app, painter.ctx());
        crate::distort_ui::draw_overlay(app, &painter, &xf);
        crate::perspective_crop_ui::draw_overlay(app, &painter, &xf);
        crate::retouch_ui::draw_source_marker(app, &painter, &xf);
        crate::vector_ui::draw_overlay(app, &painter, &xf, &doc);
        crate::symmetry_ui::draw(app, &painter, &xf);
        crate::analysis_ui::draw_overlay(app, &painter, &xf);
        crate::gradient_ui::draw_overlay(app, &painter, &xf);
        crate::slice_ui::draw_overlay(app, &painter, &xf);
        crate::frame_tool::draw_overlay(app, &painter, &xf);
        crate::artboard_tool::draw_overlay(app, &painter, &xf);
        // Tool cursors (Photoshop-style).
        let guide_hover = response.hover_pos().and_then(|p| {
            let d = xf.to_doc(p);
            if tool == Tool::Move {
                return crate::rulers::guide_at(app, d[0], d[1]);
            }
            command_guide_at(app, tool, d, crate::workspace_ui::sticky_mods(app, ui.input(|i| i.modifiers)))
        });
        if let Some((vertical, _)) = guide_hover {
            ui.ctx().set_cursor_icon(if vertical { egui::CursorIcon::ResizeHorizontal } else { egui::CursorIcon::ResizeVertical });
        } else if let Some(p) = response.hover_pos()
            && let Some(c) = crate::transform_tool::cursor(app, xf.to_doc(p), ui.input(|i| i.modifiers.alt || i.modifiers.command))
        {
            ui.ctx().set_cursor_icon(c);
            // CursorIcon::None hides the pointer; the curved arrow has to be drawn or the
            // rotate zone looks like a plain cursor (#4050).
            let d = xf.to_doc(p);
            if let Some(k) = crate::transform_tool::rotate_cursor_dir(app, d) {
                let toward = xf.to_screen((d[0] + k[0] * 16.0) as f32, (d[1] + k[1] * 16.0) as f32) - p;
                crate::crop_ui::draw_turn_cursor(&ctx, p, toward);
            }
        } else if let Some(c) = response.hover_pos().and_then(|p| crate::symmetry_ui::cursor(app, &xf, xf.to_doc(p))) {
            ui.ctx().set_cursor_icon(c);
        } else if let Some(c) = response.hover_pos().and_then(|p| crate::cam_transform::cursor(app, &ctx, &xf, p)) {
            ui.ctx().set_cursor_icon(c);
        } else if let Some((r, p, c)) =
            response.hover_pos().and_then(|p| transform_controls_rect(app, &xf).and_then(|r| transform_controls_cursor(r, p).map(|c| (r, p, c))))
        {
            // The Move tool draws handles without opening a Free Transform session.
            // Give them their resize/rotate cursors before the first pointer press (#2871).
            ui.ctx().set_cursor_icon(c);
            if let Some(toward) = transform_controls_turn(r, p) {
                crate::crop_ui::draw_turn_cursor(&ctx, p, toward);
            }
        } else if let Some(c) = response.hover_pos().filter(|_| tool.is_type()).and_then(|p| {
            let mods = crate::workspace_ui::sticky_mods(app, ui.input(|i| i.modifiers));
            crate::type_transform::cursor(app, &ctx, &xf, p, mods)
        }) {
            ui.ctx().set_cursor_icon(c);
        } else if let Some(c) = response.hover_pos().filter(|_| tool == Tool::Artboard).and_then(|p| crate::artboard_tool::cursor(app, xf.to_doc(p))) {
            ui.ctx().set_cursor_icon(c);
        } else if let Some((p, c)) = response.hover_pos().filter(|_| tool == Tool::Crop).and_then(|p| Some((p, crate::crop_ui::cursor(app, xf.to_doc(p))?))) {
            ui.ctx().set_cursor_icon(c);
            let d = xf.to_doc(p);
            if let Some(k) = crate::crop_ui::turn_cursor_dir(app, d) {
                let toward = xf.to_screen((d[0] + k[0] * 16.0) as f32, (d[1] + k[1] * 16.0) as f32) - p;
                crate::crop_ui::draw_turn_cursor(&ctx, p, toward);
            }
        } else if let Some(p) = response.hover_pos()
            && let Some(c) = selection_cursor(app, tool, xf.to_doc(p), crate::workspace_ui::sticky_mods(app, ui.input(|i| i.modifiers)))
        {
            // Over a selection (Photoshop's cursors): the move cursor where a press drags the
            // outline (#1428), the arrow with a badge for what a press does to the selected
            // pixels, a hollow arrowhead while they are dragged, a plain arrow over a floating
            // piece.
            let icon = match c {
                SelCursor::Outline => egui::CursorIcon::Move,
                SelCursor::Dragging => {
                    // The tip of the arrowhead is at (4, 4) of its 24-unit box.
                    crate::icons::cursor(ui.ctx(), "mouse-pointer-2", p, vec2(4.0, 4.0) / 24.0, 18.0);
                    egui::CursorIcon::None
                }
                SelCursor::Piece => egui::CursorIcon::Default,
                SelCursor::Cut => {
                    crate::icons::cursor_badge(ui.ctx(), "scissors", p);
                    egui::CursorIcon::Default
                }
                SelCursor::Copy => {
                    crate::icons::cursor_copy_badge(ui.ctx(), p);
                    egui::CursorIcon::Default
                }
            };
            ui.ctx().set_cursor_icon(icon);
        } else if let Some(p) = response.hover_pos() {
            let alt = ui.input(|i| i.modifiers.alt);
            if !resizing && !alt && !app.ui.shell.sticky_alt {
                draw_clone_preview(app, &painter, &xf, xf.to_doc(p), output);
            }
            let icon = match tool {
                // Resizing the brush: the circle stays where the drag began (`brush_resize`).
                t if resizing && crate::brush_resize::applies(t) => egui::CursorIcon::None,
                // ⌥ turns a painting tool into the Eyedropper (`alt_eyedropper`): its cursor too,
                // unless Preferences › Cursors › Other Cursors asks for the precise crosshair.
                t if app.alt_sampling || (app.drag.is_none() && alt_samples(t, crate::workspace_ui::sticky_mods(app, ui.input(|i| i.modifiers)))) => {
                    // While it samples (button held), the comparison ring replaces the cursor (#213).
                    if let Some(icon) = eyedropper_ring(app, &painter, &xf, p, response.is_pointer_button_down_on()) {
                        icon
                    } else if app.session.prefs().cursors.other == photocraft_engine::prefs::OtherCursor::Precise {
                        egui::CursorIcon::Crosshair
                    } else {
                        pipette_cursor(ui.ctx(), p)
                    }
                }
                t if t.is_brushlike() || t == Tool::QuickSelection => {
                    // Preferences › Cursors: brush tip outline (normal = the 50% contour, or
                    // full size), precise crosshair, or the standard pointer.
                    use photocraft_engine::prefs::PaintingCursor;
                    let cur = app.session.prefs().cursors.clone();
                    // ⇪ forces the precise crosshair, whatever the cursor preference (#1758).
                    if app.caps_lock {
                        crate::tool_cursor::crosshair(&painter, p, 6.0, 0.0)
                    } else {
                        let painting = app.drag.is_some();
                        let brush = &app.session.tools.brush;
                        let full = (brush.size / 2.0 * xf.zoom).max(1.0);
                        match cur.painting {
                            PaintingCursor::Standard => egui::CursorIcon::Default,
                            PaintingCursor::Precise => crate::tool_cursor::crosshair(&painter, p, 6.0, 0.0),
                            _ if painting && cur.show_only_crosshair_while_painting => crate::tool_cursor::crosshair(&painter, p, 5.0, 0.0),
                            // The Pencil (and the Eraser in Pencil mode): the outline of the
                            // whole pixels its dab fills, on the pixel grid. A round tip is a
                            // disc, a tiny one the whole square (#2662). The Block-mode Eraser:
                            // its square, 16 screen pixels at any zoom (#2770).
                            _ if tool == Tool::Pencil || crate::eraser_ui::pencil_mode(app, tool) || crate::eraser_ui::block_mode(app, tool) => {
                                let block = crate::eraser_ui::block_mode(app, tool);
                                let size = if block { crate::eraser_ui::block_size(app) } else { brush.size };
                                let ppp = painter.ctx().pixels_per_point();
                                let sq = pencil_cursor_rect(&xf, xf.to_doc(p), size, ppp);
                                let px = 1.0 / ppp;
                                let (dark, light) = (Stroke::new(px, Color32::from_black_alpha(160)), Stroke::new(px, Color32::from_white_alpha(230)));
                                if !block && pencil_tip_is_round(size) {
                                    let r = sq.width() / 2.0;
                                    painter.circle_stroke(sq.center(), r + px / 2.0, dark);
                                    painter.circle_stroke(sq.center(), (r - px / 2.0).max(0.0), light);
                                } else {
                                    painter.rect_stroke(sq, 0.0, dark, egui::StrokeKind::Outside);
                                    painter.rect_stroke(sq, 0.0, light, egui::StrokeKind::Inside);
                                }
                                // Too small to see where it is: the hotspot as well.
                                if cur.show_crosshair_in_brush_tip || sq.width() < 6.0 {
                                    crate::tool_cursor::crosshair(&painter, p, 4.0, 0.0);
                                }
                                egui::CursorIcon::None
                            }
                            // A shaped tip (sampled bitmap, or a rotated/flattened round tip)
                            // traces its outline, turned and flipped with the view; a plain round
                            // tip keeps the circle.
                            _ => {
                                let centre = brush_tip_centre(tool, alt || app.ui.shell.sticky_alt, cur.show_crosshair_in_brush_tip);
                                let axes = crate::brush_cursor::tip_axes(&xf, brush.size);
                                crate::brush_cursor::cursor(&painter, p, brush, full, axes, cur.painting == PaintingCursor::NormalTip, centre)
                            }
                        }
                    }
                }
                // The Eyedropper: the comparison ring follows the pointer while it samples (#213);
                // otherwise its pipette, or the crosshair for precise Other Cursors.
                Tool::Eyedropper => {
                    if let Some(icon) = eyedropper_ring(app, &painter, &xf, p, response.is_pointer_button_down_on()) {
                        icon
                    } else if precise_other_cursor(app) {
                        egui::CursorIcon::Crosshair
                    } else {
                        pipette_cursor(ui.ctx(), p)
                    }
                }
                // The Color Sampler: a pipette, or the crosshair for precise Other Cursors.
                Tool::ColorSampler => {
                    if precise_other_cursor(app) {
                        egui::CursorIcon::Crosshair
                    } else {
                        pipette_cursor(ui.ctx(), p)
                    }
                }
                // The Paint Bucket: a bucket, or the crosshair for precise Other Cursors (#3968).
                Tool::PaintBucket => {
                    if precise_other_cursor(app) {
                        egui::CursorIcon::Crosshair
                    } else {
                        paint_bucket_cursor(ui.ctx(), p)
                    }
                }
                // Over the first anchor, the close-path pointer (#4596), before the pens' own cursors.
                Tool::Pen | Tool::CurvaturePen
                    if {
                        let [x, y] = xf.to_doc(p);
                        crate::vector_ui::closes_path_at(app, tool, x, y)
                    } =>
                {
                    crate::tool_cursor::close_path(&painter, p)
                }
                // The pen tools: their own icon with its nib on the point, or the crosshair for
                // precise Other Cursors (#4639).
                Tool::Pen | Tool::FreeformPen | Tool::CurvaturePen | Tool::AddAnchorPoint | Tool::DeleteAnchorPoint | Tool::ConvertPoint => {
                    if precise_other_cursor(app) {
                        egui::CursorIcon::Crosshair
                    } else {
                        pen_cursor(ui.ctx(), tool, p)
                    }
                }
                // Preferences › Cursors › Other Cursors: Precise shows a crosshair for every tool.
                Tool::Move | Tool::Type | Tool::VerticalType | Tool::HorizontalTypeMask | Tool::VerticalTypeMask
                    if app.session.prefs().cursors.other == photocraft_engine::prefs::OtherCursor::Precise =>
                {
                    egui::CursorIcon::Crosshair
                }
                Tool::Move => egui::CursorIcon::Move,
                // Dragging is handled with the pan itself, above.
                Tool::Hand => crate::tool_cursor::hand(ui.ctx(), false),
                Tool::RotateView => crate::rotate_view::cursor(response.dragged()),
                Tool::Zoom => crate::tool_cursor::zoom(ui.ctx(), zoom_out(alt)),
                Tool::Type | Tool::VerticalType | Tool::HorizontalTypeMask | Tool::VerticalTypeMask => egui::CursorIcon::Text,
                Tool::MagneticLasso => crate::magnetic_lasso_ui::cursor(app, &painter, p, xf.zoom),
                Tool::RedEye => {
                    let pupil = app.ui.tool_options.red_eye_pupil_size.clamp(1.0, 100.0);
                    let r_doc = photocraft_algo::redeye::search_radius(pupil);
                    let r = (r_doc * xf.zoom).max(2.0);
                    painter.circle_stroke(p, r + 0.5, Stroke::new(1.0, Color32::from_black_alpha(140)));
                    painter.circle_stroke(p, r, Stroke::new(1.0, Color32::from_white_alpha(220)));
                    for (w, c) in [(2.5, Color32::from_black_alpha(140)), (1.0, Color32::from_white_alpha(220))] {
                        painter.line_segment([p - vec2(3.0, 0.0), p + vec2(3.0, 0.0)], Stroke::new(w, c));
                        painter.line_segment([p - vec2(0.0, 3.0), p + vec2(0.0, 3.0)], Stroke::new(w, c));
                    }
                    egui::CursorIcon::None
                }
                _ => egui::CursorIcon::Crosshair,
            };
            let icon = visible_crosshair(icon, &painter, p, cfg!(not(target_arch = "wasm32")));
            ui.ctx().set_cursor_icon(icon);
            // Selection tools: + / − / × badge for the effective mode (#170). A gesture keeps the
            // mode it started with (⇧ then constrains the marquee instead of adding).
            let held = crate::workspace_ui::sticky_mods(app, ui.input(|i| i.modifiers));
            let mods = app.drag.as_ref().map_or(held, |d| d.modifiers);
            if let Some(b) = crate::tool_feedback::badge(app, tool, mods) {
                crate::tool_feedback::draw_badge(&painter, p, b, tool == Tool::QuickSelection);
            }
            // ⇧ after a stroke: the straight line a click would paint (#257).
            crate::stroke_constraint::draw_line_preview(app, &painter, &xf, p, tool, held.shift);
        }
    }
    if primary && hand_pan && response.dragged() {
        crate::transform_tool::draw_overlay(app, &painter, &xf);
    }
    app.tool_override = None;
    if let Some(b) = layer_edges_rect(app, idx) {
        painter.set(layer_edges, egui::Shape::rect_stroke(xf.doc_rect(b), 0, Stroke::new(1.0, LAYER_EDGES), egui::StrokeKind::Outside));
    }
    // A tool drag held past the canvas edge scrolls the view toward the pointer (#4183); the
    // drag's next move then extends into what scrolled in.
    if primary
        && app.drag.is_some()
        && tool != Tool::Hand
        && !middle
        && let Some(p) = ui.input(|i| i.pointer.primary_down().then(|| i.pointer.latest_pos()).flatten())
    {
        let step = crate::canvas_autoscroll::edge_step(p, rect, ui.input(|i| i.stable_dt));
        if crate::canvas_autoscroll::scroll(&mut view, step, xf) {
            ctx.request_repaint();
        }
    }
    // Scrollbars (scrollbars.rs): drawn over the canvas edges, they take the pointer there.
    let t0 = crate::gpu_canvas::now_ms();
    let before = view.center;
    let over_bars = app.ui.view.shows_scrollbars()
        && crate::scrollbars::show(ui, rect, &mut view, flip, aspect, app.session.prefs().tools.overscroll, egui::Id::new(("pc-canvas", idx)));
    ctx.data_mut(|d| d.insert_temp(bars_id, over_bars));
    if view.center != before {
        ctx.request_repaint();
    }
    if primary {
        app.perf.span("scrollbars", crate::gpu_canvas::now_ms() - t0);
        app.hover_doc = response.hover_pos().map(|p| xf.to_doc(p));
    }
    if primary && app.ui.extras.rulers {
        crate::rulers::draw_rulers(app, ui, full, &xf);
    }
    crate::rotate_view::overlay(app, &ctx, &painter, &xf, &mut view, &response);
    if primary {
        // Tool input can change the stored camera while navigation edits the local view.
        // Merge changed camera values so write-back preserves tool-driven pan and rotation.
        if let (Some((center, rotation)), Some(now)) = (camera, app.ui.views.get(idx)) {
            if now.center != center {
                view.center = now.center;
                ctx.request_repaint();
            }
            if now.rotation != rotation {
                view.rotation = now.rotation;
                ctx.request_repaint();
            }
        }
        app.ui.views[idx] = view.clone();
    }
    view
}

/// Zoom to `new_zoom` about the pointer `p`. By default the document point under it stays under
/// it; with Preferences › Tools › Zoom Clicked Point to Center (`center_on_point`) it moves to the
/// centre of the view instead (#204).
/// Where a wheel or pinch zoom anchors for a pointer at `p`: the pointer itself over the image,
/// otherwise the nearest point of the image (#3720). Anchoring on a pointer out on the gray
/// pasteboard slid the image away from it, out of the window after a few notches; Photoshop keeps
/// the image in view. Clamped in document space, so a rotated or flipped view behaves the same.
fn wheel_anchor(xf: &ViewXform, p: Pos2, doc_size: [u32; 2]) -> Pos2 {
    let [w, h] = doc_size;
    let d = xf.to_doc(p);
    if w == 0 || h == 0 || !(d[0].is_finite() && d[1].is_finite()) {
        return p;
    }
    let c = [d[0].clamp(0.0, f64::from(w)), d[1].clamp(0.0, f64::from(h))];
    if c == d { p } else { xf.to_screen(c[0] as f32, c[1] as f32) }
}

fn zoom_about(view: &mut View, xf: &ViewXform, p: Pos2, new_zoom: f32, center_on_point: bool, ppp: f32) {
    let before = xf.to_doc(p);
    view.zoom = new_zoom;
    if center_on_point {
        view.center = [before[0] as f32, before[1] as f32];
        return;
    }
    // `new_zoom` is device px per document pixel; the pointer vector is in egui points.
    let ppp = if ppp.is_finite() && ppp > 0.0 { ppp } else { 1.0 };
    let u = xf.unmap_vec((p - xf.rect.center()) / (new_zoom / ppp));
    view.center = [before[0] as f32 - u.x, before[1] as f32 - u.y];
}

/// Textured document quad whose corners follow [`ViewXform`] (rotation + flip).
fn paint_mapped_image(painter: &egui::Painter, xf: &ViewXform, tex: egui::TextureId, uv: Rect, size: [f32; 2], tint: Color32) {
    let [w, h] = size;
    let pts = [xf.to_screen(0.0, 0.0), xf.to_screen(w, 0.0), xf.to_screen(w, h), xf.to_screen(0.0, h)];
    let uvs = [uv.min, pos2(uv.max.x, uv.min.y), uv.max, pos2(uv.min.x, uv.max.y)];
    let mut mesh = Mesh::with_texture(tex);
    for i in 0..4 {
        mesh.vertices.push(egui::epaint::Vertex { pos: pts[i], uv: uvs[i], color: tint });
    }
    mesh.add_triangle(0, 1, 2);
    mesh.add_triangle(0, 2, 3);
    painter.add(egui::Shape::mesh(mesh));
}

/// The GPU canvas view, then its View › Pattern Preview copies (#1067): the same texture drawn
/// again whole document sizes away, each with its own view uniform and no shadow.
fn paint_gpu_view(painter: &egui::Painter, rect: Rect, params: crate::gpu_canvas::ViewParams, copies: &[[i32; 2]]) {
    crate::gpu_canvas::GpuCanvas::paint(painter, rect, params);
    for (k, &c) in copies.iter().enumerate() {
        let copy = crate::gpu_canvas::ViewParams {
            offset: crate::pattern_preview::offset(c, params.doc_size),
            shadow: false,
            // Keyed by slot, not by tile index, so panning reuses the same few uniform buffers.
            view_key: egui::Id::new((params.view_key, "pattern", k)).value(),
            ..params
        };
        crate::gpu_canvas::GpuCanvas::paint(painter, rect, copy);
    }
}

/// The mapping that draws Pattern Preview copy `c` of a `size` document where it sits next to
/// the document: the same view, looking the matching number of document sizes back.
fn pattern_xf(xf: &ViewXform, c: [i32; 2], size: [u32; 2]) -> ViewXform {
    let [ox, oy] = crate::pattern_preview::offset(c, size);
    ViewXform { center: [xf.center[0] - ox, xf.center[1] - oy], ..*xf }
}

/// CPU canvas: the transparency checkerboard (or white) and the display texture, for the
/// document or one of its Pattern Preview copies (`xf` from [`pattern_xf`]).
fn paint_cpu_document(
    painter: &egui::Painter,
    xf: &ViewXform,
    bounds: DRect,
    rotated: bool,
    checker: Option<(egui::TextureId, f32)>,
    tex: Option<egui::TextureId>,
) {
    let size = [bounds.width() as f32, bounds.height() as f32];
    let img_rect = xf.doc_rect(bounds);
    match checker {
        Some((checker_id, square)) => {
            if rotated {
                let tiles = vec2(size[0], size[1]) / (2.0 * square);
                paint_mapped_image(painter, xf, checker_id, Rect::from_min_max(Pos2::ZERO, pos2(tiles.x, tiles.y)), size, Color32::WHITE);
            } else {
                let tiles = img_rect.size() / (2.0 * square);
                painter.image(checker_id, img_rect, Rect::from_min_max(Pos2::ZERO, pos2(tiles.x, tiles.y)), Color32::WHITE);
            }
        }
        None => {
            if rotated {
                paint_mapped_quad(painter, xf, size, Color32::WHITE);
            } else {
                painter.rect_filled(img_rect, 0.0, Color32::WHITE);
            }
        }
    }
    if let Some(tex) = tex {
        if rotated {
            paint_mapped_image(painter, xf, tex, Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)), size, Color32::WHITE);
        } else {
            let uv = if xf.flip { Rect::from_min_max(pos2(1.0, 0.0), pos2(0.0, 1.0)) } else { Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)) };
            painter.image(tex, img_rect, uv, Color32::WHITE);
        }
    }
}

fn paint_mapped_quad(painter: &egui::Painter, xf: &ViewXform, size: [f32; 2], color: Color32) {
    let [w, h] = size;
    let pts = vec![xf.to_screen(0.0, 0.0), xf.to_screen(w, 0.0), xf.to_screen(w, h), xf.to_screen(0.0, h)];
    painter.add(egui::Shape::convex_polygon(pts, color, Stroke::NONE));
}

/// Draw boundary segments as marching ants: white base, black dashes phased along x + y.
fn marching_ants_segments(painter: &egui::Painter, xf: &ViewXform, segs: &[crate::outline::Segment], time: f64) {
    marching_ants_mapped(painter, segs, time, |p| xf.to_screen(p[0] as f32, p[1] as f32));
}

/// [`marching_ants_segments`] with each outline point placed on screen by `to_screen` (a
/// transformed outline: Content-Aware Move's Transform On Drop).
pub(crate) fn marching_ants_mapped(painter: &egui::Painter, segs: &[crate::outline::Segment], time: f64, to_screen: impl Fn([i32; 2]) -> Pos2) {
    let dash = 4.0f32;
    let phase = ((time * 10.0) % (dash as f64 * 2.0)) as f32;
    let clip = painter.clip_rect();
    // Horizontal and vertical segments go in one mesh of 1 px quads, which egui draws as it is: a
    // line shape per dash took egui ~35 ms a frame to tessellate for a speckled selection seen
    // whole (about 10⁵ segments), ten frames a second while the selection is shown. A slanted
    // segment (a transformed outline) stays a line shape, which egui antialiases.
    let mut mesh = egui::Mesh::default();
    // A 1 px wide quad from `a` to `b` (`side` is half a pixel across the line).
    let mut quad = |a: egui::Pos2, b: egui::Pos2, side: egui::Vec2, color: Color32| {
        let i = mesh.vertices.len() as u32;
        for p in [a + side, a - side, b - side, b + side] {
            mesh.colored_vertex(p, color);
        }
        mesh.add_triangle(i, i + 1, i + 2);
        mesh.add_triangle(i, i + 2, i + 3);
    };
    for (a, b) in segs {
        let pa = to_screen(*a);
        let pb = to_screen(*b);
        if !clip.intersects(Rect::from_two_pos(pa, pb).expand(1.0)) {
            continue;
        }
        let pa = pos2(pa.x.round() + 0.5, pa.y.round() + 0.5);
        let pb = pos2(pb.x.round() + 0.5, pb.y.round() + 0.5);
        let len = pa.distance(pb);
        if len < 0.5 {
            continue;
        }
        let dir = (pb - pa) / len;
        let side = dir.rot90() * 0.5;
        let straight = pa.x == pb.x || pa.y == pb.y;
        if straight {
            // The white base runs half a pixel past both ends, so joined segments meet at corners.
            quad(pa - dir * 0.5, pb + dir * 0.5, side, Color32::WHITE);
        } else {
            painter.line_segment([pa, pb], Stroke::new(1.0, Color32::WHITE));
        }
        // Phase by screen position so dashes line up across joined segments.
        let start = (pa.x + pa.y + phase).rem_euclid(dash * 2.0);
        let mut t = -start;
        while t < len {
            let s0 = t.max(0.0);
            let s1 = (t + dash).min(len);
            if s1 > s0 {
                if straight {
                    quad(pa + dir * s0, pa + dir * s1, side, Color32::BLACK);
                } else {
                    painter.line_segment([pa + dir * s0, pa + dir * s1], Stroke::new(1.0, Color32::BLACK));
                }
            }
            t += dash * 2.0;
        }
    }
    if !mesh.is_empty() {
        painter.add(egui::Shape::mesh(mesh));
    }
}

#[allow(dead_code)]
fn marching_ants(painter: &egui::Painter, r: Rect, time: f64) {
    let dash = 4.0;
    let offset = ((time * 8.0) % (dash as f64 * 2.0)) as f32;
    painter.rect_stroke(r, 0.0, Stroke::new(1.0, Color32::WHITE), egui::StrokeKind::Middle);
    let pts = [r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom(), r.left_top()];
    for w in pts.windows(2) {
        let (a, b) = (w[0], w[1]);
        let len = a.distance(b);
        let dir = (b - a) / len.max(1e-3);
        let mut t = -offset;
        while t < len {
            let s = t.max(0.0);
            let e = (t + dash).min(len);
            if e > s {
                painter.line_segment([a + dir * s, a + dir * e], Stroke::new(1.0, Color32::BLACK));
            }
            t += dash * 2.0;
        }
    }
}

/// Photoshop crop overlay: the `shield` outside (`crop_shield::fill`; none when disabled), bright
/// frame, the composition `guides` (unit-square polylines, `crop_overlay::guides`), corner and
/// edge handles.
fn crop_overlay(painter: &egui::Painter, r: Rect, shield: Option<Color32>, guides: &[crate::crop_overlay::Polyline]) {
    let clip = painter.clip_rect();
    if let Some(dim) = shield {
        for band in [
            Rect::from_min_max(clip.min, egui::pos2(clip.max.x, r.min.y)),
            Rect::from_min_max(egui::pos2(clip.min.x, r.max.y), clip.max),
            Rect::from_min_max(egui::pos2(clip.min.x, r.min.y), egui::pos2(r.min.x, r.max.y)),
            Rect::from_min_max(egui::pos2(r.max.x, r.min.y), egui::pos2(clip.max.x, r.max.y)),
        ] {
            painter.rect_filled(band, 0.0, dim);
        }
    }
    painter.rect_stroke(r, 0.0, Stroke::new(1.0, Color32::WHITE), egui::StrokeKind::Middle);
    draw_crop_guides(painter, guides, |a, b| r.min + r.size() * vec2(a, b));
    let h = Stroke::new(3.0, Color32::WHITE);
    let l = 14.0f32.min(r.width() / 3.0).min(r.height() / 3.0);
    for (c, dx, dy) in [(r.left_top(), 1.0, 1.0), (r.right_top(), -1.0, 1.0), (r.left_bottom(), 1.0, -1.0), (r.right_bottom(), -1.0, -1.0)] {
        painter.line_segment([c, c + vec2(l * dx, 0.0)], h);
        painter.line_segment([c, c + vec2(0.0, l * dy)], h);
    }
    // Edge handles: short bars at the middle of each side.
    let (lx, ly) = (l.min(r.width() / 4.0) / 2.0, l.min(r.height() / 4.0) / 2.0);
    for (c, horizontal) in [(r.center_top(), true), (r.center_bottom(), true), (r.left_center(), false), (r.right_center(), false)] {
        let e = if horizontal { vec2(lx, 0.0) } else { vec2(0.0, ly) };
        painter.line_segment([c - e, c + e], h);
    }
}

/// [`crop_overlay`] for a turned frame (or a rotated view): `q` is the frame's top-left, top-right,
/// bottom-right and bottom-left corner on screen, a parallelogram.
fn crop_overlay_turned(painter: &egui::Painter, q: [Pos2; 4], shield: Option<Color32>, guides: &[crate::crop_overlay::Polyline]) {
    let (u, v) = (q[1] - q[0], q[3] - q[0]);
    let (lu, lv) = (u.length(), v.length());
    if !(lu.is_finite() && lv.is_finite()) || lu < 1e-3 || lv < 1e-3 {
        return;
    }
    let at = |a: f32, b: f32| q[0] + u * a + v * b;
    // The shield: four trapezoids from the frame out to the frame grown past the whole view.
    let clip = painter.clip_rect();
    let reach = clip.size().length() + (clip.center() - at(0.5, 0.5)).length();
    let (eu, ev) = (reach / lu, reach / lv);
    let outer = [at(-eu, -ev), at(1.0 + eu, -ev), at(1.0 + eu, 1.0 + ev), at(-eu, 1.0 + ev)];
    if let Some(dim) = shield {
        let mut mesh = egui::Mesh::default();
        for p in q.iter().chain(outer.iter()) {
            mesh.colored_vertex(*p, dim);
        }
        for i in 0..4u32 {
            let j = (i + 1) % 4;
            mesh.add_triangle(i, j, 4 + j);
            mesh.add_triangle(i, 4 + j, 4 + i);
        }
        painter.add(egui::Shape::mesh(mesh));
    }
    painter.add(egui::Shape::closed_line(q.to_vec(), Stroke::new(1.0, Color32::WHITE)));
    draw_crop_guides(painter, guides, at);
    let h = Stroke::new(3.0, Color32::WHITE);
    let l = 14.0f32.min(lu / 3.0).min(lv / 3.0);
    let (du, dv) = (u / lu, v / lv);
    for (c, su, sv) in [(q[0], 1.0, 1.0), (q[1], -1.0, 1.0), (q[3], 1.0, -1.0), (q[2], -1.0, -1.0)] {
        painter.line_segment([c, c + du * (l * su)], h);
        painter.line_segment([c, c + dv * (l * sv)], h);
    }
    // Edge handles: short bars at the middle of each side, along it.
    let (lx, ly) = (l.min(lu / 4.0) / 2.0, l.min(lv / 4.0) / 2.0);
    for (c, e) in [(at(0.5, 0.0), du * lx), (at(0.5, 1.0), du * lx), (at(0.0, 0.5), dv * ly), (at(1.0, 0.5), dv * ly)] {
        painter.line_segment([c - e, c + e], h);
    }
}

/// The crop overlay's guides, each unit-square point placed on screen by `at`.
fn draw_crop_guides(painter: &egui::Painter, guides: &[crate::crop_overlay::Polyline], at: impl Fn(f32, f32) -> Pos2) {
    let thin = Stroke::new(1.0, Color32::from_white_alpha(90));
    for l in guides {
        painter.add(egui::Shape::line(l.iter().map(|p| at(p[0], p[1])).collect(), thin));
    }
}

/// What the Crop tool shows past the canvas, cached per document state ([`draw_beyond_canvas`]).
#[derive(Clone)]
struct BeyondCanvas {
    snapshot: std::sync::Weak<Document>,
    key: u64,
    /// Every layer's pixels and the canvas (`extra_cmds::reveal_all_bounds`).
    extent: DRect,
    /// The composite around the canvas: each band, the area its texture covers, the texture.
    bands: Vec<(DRect, DRect, egui::TextureHandle)>,
}

/// Band `r` around the canvas `c` (on screen) pushed a point under the canvas's edge, so that the
/// edge's anti-aliased pixels blend with the band rather than with the pasteboard.
fn under_canvas(mut r: Rect, c: Rect) -> Rect {
    let near = |a: f32, b: f32| (a - b).abs() < 0.01;
    if near(r.max.y, c.min.y) {
        r.max.y += 1.0;
    }
    if near(r.min.y, c.max.y) {
        r.min.y -= 1.0;
    }
    if near(r.max.x, c.min.x) {
        r.max.x += 1.0;
    }
    if near(r.min.x, c.max.x) {
        r.min.x -= 1.0;
    }
    r
}

/// `outer` minus `inner`: the bands above, below, left and right of it (empty ones dropped).
fn ring(outer: DRect, inner: DRect) -> Vec<DRect> {
    let i = inner.intersect(&outer);
    if i.is_empty() {
        return if outer.is_empty() { Vec::new() } else { vec![outer] };
    }
    [
        DRect::new(outer.x0, outer.y0, outer.x1, i.y0),
        DRect::new(outer.x0, i.y1, outer.x1, outer.y1),
        DRect::new(outer.x0, i.y0, i.x0, i.y1),
        DRect::new(i.x1, i.y0, outer.x1, i.y1),
    ]
    .into_iter()
    .filter(|r| !r.is_empty())
    .collect()
}

/// Crop tool, while a frame is edited (`crop_ui::shows_beyond_canvas`): like Photoshop's crop
/// preview, the canvas grows to every layer's pixels (a crop with Delete Cropped Pixels off keeps
/// them; a moved layer reaches past the edges) and to the frame, transparency drawn as the
/// checkerboard, all under the shield `crop_overlay` adds. Only the ring around the canvas is
/// composited, on the CPU, once per document state: at most `MAX_TEXTURE` texels per side, from a
/// downsampled proxy past that (so a far-off layer costs no more than a 16 MP composite).
/// Drawn before the canvas, each band reaching a point under its edge. Returns whether anything
/// past the canvas was drawn.
fn draw_beyond_canvas(app: &mut PhotocraftApp, painter: &egui::Painter, xf: &ViewXform, idx: usize, output: Option<u32>) -> bool {
    let ctx = painter.ctx().clone();
    let Some((snapshot, id)) = app.session.documents().get(idx).map(|st| (std::sync::Arc::downgrade(&st.doc), st.doc.id)) else { return false };
    let key = egui::Id::new(("crop-beyond-canvas", id.0, output));
    let frame =
        app.ui.crop_rect.filter(|f| f.iter().all(|v| v.is_finite()) && app.session.active_index() == Some(idx) && crate::crop_ui::shows_beyond_canvas(app));
    let Some(f) = frame else {
        ctx.data_mut(|d| d.remove::<BeyondCanvas>(key));
        return false;
    };
    let (doc, preview_key) = display_doc(app, idx);
    if doc.has_artboards() {
        return false;
    }
    let (display, display_key) = canvas_display(app, &doc, output);
    let doc_key = preview_key ^ display_key;
    let cached: Option<BeyondCanvas> = ctx.data(|d| d.get_temp(key));
    let c = match cached.filter(|c| c.snapshot.ptr_eq(&snapshot) && c.key == doc_key) {
        Some(c) => c,
        None => {
            let t0 = crate::gpu_canvas::now_ms();
            let extent = photocraft_engine::extra_cmds::reveal_all_bounds(&doc);
            // One texel per `k` document pixels keeps every texture within MAX_TEXTURE.
            let k = extent.width().max(extent.height()).div_ceil(MAX_TEXTURE).max(1);
            let proxy = (k > 1).then(|| photocraft_compose::proxy::proxy_document(&doc, k));
            let src = proxy.as_ref().unwrap_or(&*doc);
            let k = k as i32;
            let opts = TextureOptions { magnification: egui::TextureFilter::Nearest, ..TextureOptions::LINEAR };
            let bands = ring(extent, doc.bounds())
                .into_iter()
                .map(|b| {
                    // The proxy pixels over `b`: document pixels / k, rounded outwards.
                    let up = |v: i32| v.div_euclid(k) + i32::from(v.rem_euclid(k) != 0);
                    let p = DRect::new(b.x0.div_euclid(k), b.y0.div_euclid(k), up(b.x1), up(b.y1));
                    let buf = photocraft_compose::render_reduced_rect(src, p, p.width(), p.height());
                    let tex = ctx.load_texture("crop-beyond-canvas", display_image(display.as_deref(), &buf), opts);
                    (b, DRect::new(p.x0.saturating_mul(k), p.y0.saturating_mul(k), p.x1.saturating_mul(k), p.y1.saturating_mul(k)), tex)
                })
                .collect();
            app.perf.span("crop-beyond-canvas", crate::gpu_canvas::now_ms() - t0);
            let c = BeyondCanvas { snapshot, key: doc_key, extent, bands };
            ctx.data_mut(|d| d.insert_temp(key, c.clone()));
            c
        }
    };
    // Transparency out to the frame, in phase with the canvas's own checkerboard.
    let canvas = xf.doc_rect(doc.bounds());
    // A turned frame reaches as far as its corners.
    let f = crate::crop_ui::turned_bounds(f, crate::crop_ui::angle(app));
    let fr = DRect::new(f[0].floor() as i32, f[1].floor() as i32, f[2].ceil() as i32, f[3].ceil() as i32);
    let square = app.session.prefs().transparency_and_gamut.square();
    let checker_id = square.map(|_| checker(app, &ctx));
    let around = ring(c.extent.union(&fr), doc.bounds());
    // A turned view (Rotate View, or the Crop tool's default mode turning the image behind its
    // box) draws each band as its turned quad: its screen bounding box would stretch the band's
    // texture and spread the checkerboard past the band.
    let turned = xf.rotation != 0.0;
    let checker_uv = |p: Pos2, sq: f32| ((p - canvas.min) / (2.0 * sq)).to_pos2();
    for &b in &around {
        if turned {
            let q = doc_quad(xf, b);
            match square.zip(checker_id) {
                Some((sq, id)) => textured_quad(painter, id, q, q.map(|p| checker_uv(p, sq))),
                None => {
                    painter.add(egui::Shape::convex_polygon(q.to_vec(), Color32::WHITE, Stroke::NONE));
                }
            }
            continue;
        }
        let r = under_canvas(xf.doc_rect(b), canvas);
        match square.zip(checker_id) {
            Some((sq, id)) => {
                let uv = Rect::from_min_max(checker_uv(r.min, sq), checker_uv(r.max, sq));
                painter.image(id, r, uv, Color32::WHITE);
            }
            None => {
                painter.rect_filled(r, 0.0, Color32::WHITE);
            }
        }
    }
    for (b, t, tex) in &c.bands {
        let part = |a: i32, o: i32, len: u32| (i64::from(a) - i64::from(o)) as f32 / len.max(1) as f32;
        let (mut u0, mut u1) = (part(b.x0, t.x0, t.width()), part(b.x1, t.x0, t.width()));
        if turned {
            // The quad's corners are the band's own (flip included), so no swap.
            let (v0, v1) = (part(b.y0, t.y0, t.height()), part(b.y1, t.y0, t.height()));
            textured_quad(painter, tex.id(), doc_quad(xf, *b), [pos2(u0, v0), pos2(u1, v0), pos2(u1, v1), pos2(u0, v1)]);
            continue;
        }
        if xf.flip {
            (u0, u1) = (u1, u0);
        }
        let uv = Rect::from_min_max(pos2(u0, part(b.y0, t.y0, t.height())), pos2(u1, part(b.y1, t.y0, t.height())));
        // The texture's edge texels stretch over the point under the canvas.
        let (r, e) = (xf.doc_rect(*b), under_canvas(xf.doc_rect(*b), canvas));
        let at = |p: Pos2| uv.min + (p - r.min) / r.size() * uv.size();
        painter.image(tex.id(), e, Rect::from_min_max(at(e.min), at(e.max)), Color32::WHITE);
    }
    !around.is_empty()
}

/// Document rect `b`'s corners on screen, clockwise from its top-left (a parallelogram in a turned
/// view).
fn doc_quad(xf: &ViewXform, b: DRect) -> [Pos2; 4] {
    [(b.x0, b.y0), (b.x1, b.y0), (b.x1, b.y1), (b.x0, b.y1)].map(|(x, y)| xf.to_screen(x as f32, y as f32))
}

/// Texture `tex` drawn over the quad `q`, `uv` at its corners.
fn textured_quad(painter: &egui::Painter, tex: egui::TextureId, q: [Pos2; 4], uv: [Pos2; 4]) {
    if !q.iter().chain(uv.iter()).all(|p| p.x.is_finite() && p.y.is_finite()) {
        return;
    }
    let mut mesh = egui::Mesh::with_texture(tex);
    for (pos, uv) in q.into_iter().zip(uv) {
        mesh.vertices.push(egui::epaint::Vertex { pos, uv, color: Color32::WHITE });
    }
    mesh.add_triangle(0, 1, 2);
    mesh.add_triangle(0, 2, 3);
    painter.add(egui::Shape::mesh(mesh));
}

/// Overlays that persist between gestures: polygonal or magnetic lasso in progress, pending crop box.
fn draw_tool_state(app: &PhotocraftApp, painter: &egui::Painter, xf: &ViewXform, hover: Option<Pos2>) {
    crate::magnetic_lasso_ui::draw(app, painter, xf, hover);
    if !app.ui.polygon.is_empty() {
        let mut pts: Vec<Pos2> = app.ui.polygon.iter().map(|p| xf.to_screen(p[0] as f32, p[1] as f32)).collect();
        if let Some(h) = hover {
            // The rubber-band preview must use the same constrained point as the click,
            // including when the canvas is zoomed, rotated or flipped.
            let mods = painter.ctx().input(|i| i.modifiers);
            if app.ui.tool == Tool::PolygonLasso && mods.shift && !mods.alt {
                let raw = xf.to_doc(h);
                let last = app.ui.polygon.last().copied().unwrap_or(raw);
                let snapped = constrained_polygon_point(last, raw);
                pts.push(xf.to_screen(snapped[0] as f32, snapped[1] as f32));
            } else {
                pts.push(h);
            }
        }
        // Just the outline and its rubber band: the vertices aren't handles to grab.
        crate::tool_feedback::draw_ants(painter, &pts, false);
    }
    crate::crop_straighten::draw(app, painter, |p| xf.to_screen(p[0] as f32, p[1] as f32), hover);
    crate::blur_pins::draw(app, painter, xf);
    if let Some(c) = app.ui.crop_rect {
        let deg = crate::crop_ui::angle(app);
        let shield = crate::crop_shield::current(app, painter.ctx());
        if deg == 0.0 && xf.rotation == 0.0 {
            let r = Rect::from_two_pos(xf.to_screen(c[0] as f32, c[1] as f32), xf.to_screen(c[2] as f32, c[3] as f32));
            crop_overlay(painter, r, shield, &crate::crop_overlay::current_guides(app, r.width(), r.height()));
        } else {
            let q = crate::crop_ui::corners(c, deg).map(|p| xf.to_screen(p[0] as f32, p[1] as f32));
            crop_overlay_turned(painter, q, shield, &crate::crop_overlay::current_guides(app, q[0].distance(q[1]), q[0].distance(q[3])));
        }
        // The frame's angle beside the pointer while it turns (#1792).
        if let (Some(a), Some(h)) = (crate::crop_ui::turning(app), hover) {
            draw_value_readout(app, painter.ctx(), "crop-angle-readout", h, ["Angle:"], [format!("{a:.1}°")]);
        } else if let (Some([w, h]), Some(at)) = (crate::crop_ui::sizing(app), hover) {
            // Its W × H while it is drawn or resized (#1919), like the marquee's.
            draw_marquee_readout(app, painter.ctx(), at, size_readout(app, [0.0, 0.0, w, h]));
        }
    }
}

/// View › Show › Layer Edges: the active layer's content bounds (cached per tile) where the canvas
/// shows them. During a Move drag that is at the offset the image was drawn with this frame, so
/// the outline moves in lockstep with the layer's pixels (#2413).
pub(crate) fn layer_edges_rect(app: &PhotocraftApp, idx: usize) -> Option<photocraft_geom::Rect> {
    if !app.ui.view.shows(app.ui.view.show.layer_edges) {
        return None;
    }
    let st = app.session.documents().get(idx)?;
    let id = st.active_layer?;
    let b = st.doc.layer(id)?.surface().map(photocraft_compose::bounds::content_bounds)?;
    if b.is_empty() {
        return None;
    }
    let (dx, dy) = crate::move_ui::shown_offset(app, st.doc.id, st.revision, id).unwrap_or((0, 0));
    Some(b.translate(dx, dy))
}

/// Move tool › Show Transform Controls: the active layer's transform bounds.
fn transform_controls_rect(app: &PhotocraftApp, xf: &ViewXform) -> Option<Rect> {
    if app.ui.tool != Tool::Move || !app.ui.tool_options.move_show_transform || app.ui.transform.is_some() || app.drag.is_some() {
        return None;
    }
    let st = app.session.active()?;
    let l = st.active_layer.and_then(|id| st.doc.layer(id))?;
    if crate::doc_props_ui::is_background(&st.doc, l) {
        return None;
    }
    let b = photocraft_engine::transform_cmds::transform_bounds(&st.doc, l);
    let (dx, dy) = crate::move_ui::floating_offset(app).unwrap_or((0, 0));
    let b = b.translate(dx, dy);
    if b.is_empty() {
        return None;
    }
    Some(Rect::from_two_pos(xf.to_screen(b.x0 as f32, b.y0 as f32), xf.to_screen(b.x1 as f32, b.y1 as f32)))
}

/// The Move tool shows transform handles even before a Free Transform session exists.
/// Use the same hit areas for pointer-down and hover, so a control never behaves like
/// resize/rotate while showing the ordinary four-arrow Move cursor (#2871).
fn transform_controls_cursor(r: Rect, p: Pos2) -> Option<egui::CursorIcon> {
    let handles = [r.left_top(), r.center_top(), r.right_top(), r.right_center(), r.right_bottom(), r.center_bottom(), r.left_bottom(), r.left_center()];
    let grab = crate::transform_tool::HANDLE_PX as f32;
    for (i, handle) in handles.into_iter().enumerate() {
        if handle.distance(p) <= grab {
            return Some(match i {
                0 | 4 => egui::CursorIcon::ResizeNwSe,
                2 | 6 => egui::CursorIcon::ResizeNeSw,
                1 | 5 => egui::CursorIcon::ResizeVertical,
                _ => egui::CursorIcon::ResizeHorizontal,
            });
        }
    }
    // The rotation band: the pointer gives way to the curved rotate arrow (`transform_controls_turn`).
    in_rotation_band(r, p).then_some(egui::CursorIcon::None)
}

fn in_rotation_band(r: Rect, p: Pos2) -> bool {
    r.expand(18.0).contains(p) && !r.expand(5.0).contains(p)
}

/// In the rotation band just outside the Move tool's transform controls, the screen direction the
/// curved rotate arrow bends round: into the box along the nearest corner's bisector, as Free
/// Transform draws it (#4658). Before, the band showed a plain alias pointer, so nothing said a
/// drag there rotates.
fn transform_controls_turn(r: Rect, p: Pos2) -> Option<Vec2> {
    if !in_rotation_band(r, p)
        || [r.left_top(), r.center_top(), r.right_top(), r.right_center(), r.right_bottom(), r.center_bottom(), r.left_bottom(), r.left_center()]
            .into_iter()
            .any(|h| h.distance(p) <= crate::transform_tool::HANDLE_PX as f32)
    {
        return None;
    }
    let corner = pos2(if p.x < r.center().x { r.left() } else { r.right() }, if p.y < r.center().y { r.top() } else { r.bottom() });
    Some(vec2((r.center().x - corner.x).signum(), (r.center().y - corner.y).signum()).normalized() * 16.0)
}

/// A visible handle starts scaling; the narrow band just outside the box starts rotation.
/// The interior stays the normal Move-tool drag target.
fn transform_controls_hit(r: Rect, p: Pos2) -> bool {
    transform_controls_cursor(r, p).is_some()
}

/// Enter the existing Free Transform session when a Move-tool transform control is pressed.
fn begin_transform_controls_at(app: &mut PhotocraftApp, ctx: &egui::Context, xf: &ViewXform, p: Pos2) -> bool {
    let Some(r) = transform_controls_rect(app, xf) else { return false };
    if !transform_controls_hit(r, p) {
        return false;
    }
    match crate::transform_tool::begin(app, ctx) {
        Ok(()) => true,
        Err(e) => {
            app.ui.status = e;
            false
        }
    }
}

/// Move tool › Show Transform Controls: the active layer's bounding box with its eight handles.
fn draw_transform_controls(app: &mut PhotocraftApp, painter: &egui::Painter, xf: &ViewXform) {
    let Some(r) = transform_controls_rect(app, xf) else { return };
    let accent = crate::theme::Tokens::get(painter.ctx()).accent;
    painter.rect_stroke(r, 0.0, Stroke::new(1.0, accent), egui::StrokeKind::Middle);
    for p in [r.left_top(), r.center_top(), r.right_top(), r.right_center(), r.right_bottom(), r.center_bottom(), r.left_bottom(), r.left_center()] {
        let h = Rect::from_center_size(p, vec2(7.0, 7.0));
        painter.rect_filled(h, 0.0, Color32::WHITE);
        painter.rect_stroke(h, 0.0, Stroke::new(1.0, accent), egui::StrokeKind::Inside);
    }
}

/// The line from the brush to the pointer while stroke smoothing lags (Pulled String Mode and
/// Stroke Catch-up, #3479): a dark underlay and a light line stay visible on any pixels, and a dot
/// marks the brush end.
fn draw_smoothing_string(painter: &egui::Painter, xf: &ViewXform, (brush, pointer): ([f64; 2], [f64; 2])) {
    let (a, b) = (xf.to_screen(brush[0] as f32, brush[1] as f32), xf.to_screen(pointer[0] as f32, pointer[1] as f32));
    if a.distance(b) < 2.0 {
        return;
    }
    painter.line_segment([a, b], Stroke::new(3.0, Color32::from_black_alpha(110)));
    painter.line_segment([a, b], Stroke::new(1.0, Color32::WHITE));
    painter.circle_filled(a, 2.5, Color32::WHITE);
}

fn draw_drag_preview(app: &mut PhotocraftApp, painter: &egui::Painter, xf: &ViewXform) {
    draw_tool_state(app, painter, xf, painter.ctx().input(|i| i.pointer.hover_pos()));
    // Remove strokes waiting for ↩ stay tinted between drags.
    crate::retouch_ui::draw_remove_pending(app, painter, xf);
    let Some(d) = &app.drag else {
        app.trail = None;
        return;
    };
    // Dragging the selection: its own ants follow the pointer (`selection_shown_offset`).
    if d.sel_move.is_some() {
        return;
    }
    let last = d.points.last().map(|p| [p[0], p[1]]).unwrap_or(d.start);
    let marquee = marquee_preview_px(&app.ui.tool_options, d, app.point_zoom());
    if let Some(r) = marquee {
        draw_marquee_readout(app, painter.ctx(), xf.to_screen(last[0] as f32, last[1] as f32), size_readout(app, r));
    }
    match d.tool {
        // The canvas shows the live stroke itself (`LiveStroke`); a stroke that couldn't start one
        // (Sample All Layers, say) keeps the trail.
        t if strokes_live(t) && app.live_stroke.is_some() => {
            // Smoothing holds the brush back from the pointer: draw the string between them.
            if let Some(EngineStroke::Brush(l)) = app.live_stroke.as_ref().map(|l| &l.stroke)
                && let Some(s) = l.smoothing_string()
            {
                draw_smoothing_string(painter, xf, s);
            }
        }
        t if t.is_brushlike() || t == Tool::QuickSelection => {
            // Retouching strokes preview as a translucent trail of the brush footprint: a mask,
            // not a brush-wide egui polyline (which zoomed in tessellates into wedges, #189).
            let col = match t {
                Tool::QuickSelection => Color32::from_white_alpha(40),
                // The Remove Tool tints what it will remove, like Photoshop's.
                Tool::Remove => crate::theme::Tokens::get(painter.ctx()).danger.gamma_multiply(0.45),
                _ => Color32::from_white_alpha(60),
            };
            let Some(st) = app.session.active() else { return };
            let size = [st.doc.size.width, st.doc.size.height];
            let doc_rect = xf.doc_rect(st.doc.bounds());
            let trail = app.trail.get_or_insert_with(|| crate::stroke_trail::Trail::new(size));
            if t == Tool::SpotHealing {
                // What the heal covers, dab by dab: soft edges, pressure and shape dynamics (#2364).
                trail.feed_dabs(&d.points, || photocraft_engine::retouch_cmds::stroke_brush(&app.session.tools.brush, &json!({})));
            } else {
                trail.feed(&d.points, app.session.tools.brush.size);
            }
            trail.draw(painter, doc_rect, xf.flip, col);
        }
        t if crate::vector_ui::is_shape_tool(t) => crate::vector_ui::draw_shape_preview(app, painter, xf, t, d.start, last, d.live),
        Tool::SingleRowMarquee | Tool::SingleColumnMarquee => {
            // Mapped corner by corner like the other marquees, so a rotated view shows the line
            // where the release commits it (#2512).
            if let Some([x0, y0, x1, y1]) = single_line_drag_px(app, d) {
                crate::tool_feedback::draw_ants(painter, &marquee_ants(xf, d.tool, [x0, y0], [x1, y1]), true);
            }
        }
        // Nothing selected yet: no outline, as the release would deselect.
        Tool::RectMarquee | Tool::EllipseMarquee if marquee.is_none() => {}
        Tool::RectMarquee | Tool::EllipseMarquee | Tool::ObjectSelection => {
            // Marching ants, visible on any pixels (#172): built in document space and mapped
            // corner by corner, so a rotated view shows them where the release commits (#2512).
            let pts = marquee_drag_ants(d, xf, marquee);
            crate::tool_feedback::draw_ants(painter, &pts, true);
        }
        // Patch / Content-Aware Move dragging the selection: its outline follows the pointer.
        Tool::Patch | Tool::ContentAwareMove if crate::retouch_ui::patch_drags_selection(app, d.start, d.modifiers) => {
            let start = d.start;
            let [dx, dy] = crate::retouch_ui::patch_offset(app, start, last);
            if let Some((_, _, segs)) = &app.outline_cache {
                let moved: Vec<crate::outline::Segment> = segs.iter().map(|(a, b)| ([a[0] + dx, a[1] + dy], [b[0] + dx, b[1] + dy])).collect();
                marching_ants_segments(painter, xf, &moved, painter.ctx().input(|i| i.time));
            }
        }
        Tool::FreeformPen => {
            let mut pts: Vec<Pos2> = d.points.iter().map(|p| xf.to_screen(p[0] as f32, p[1] as f32)).collect();
            // Between ⌥-clicks: the straight segment the next click makes.
            if let Some(lasso) = &d.lasso {
                pts.push(xf.to_screen(lasso.cursor[0] as f32, lasso.cursor[1] as f32));
            }
            painter.add(egui::Shape::line(pts, Stroke::new(1.5, crate::theme::Tokens::get(painter.ctx()).accent)));
        }
        Tool::Lasso | Tool::Patch | Tool::ContentAwareMove => {
            let mut pts: Vec<Pos2> = d.points.iter().map(|p| xf.to_screen(p[0] as f32, p[1] as f32)).collect();
            if let Some(lasso) = &d.lasso {
                pts.push(xf.to_screen(lasso.cursor[0] as f32, lasso.cursor[1] as f32));
            }
            // On pixel centres like the marquee's ants, so a 1 pt line stays crisp wherever the
            // canvas sits (a pixel edge would blur it to grey).
            let pts: Vec<Pos2> = pts.into_iter().map(|p| p.round() + vec2(0.5, 0.5)).collect();
            crate::tool_feedback::draw_ants(painter, &pts, false);
        }
        Tool::Gradient => {
            let a = xf.to_screen(d.start[0] as f32, d.start[1] as f32);
            let b = xf.to_screen(last[0] as f32, last[1] as f32);
            painter.line_segment([a, b], Stroke::new(3.0, Color32::from_black_alpha(140)));
            painter.line_segment([a, b], Stroke::new(1.0, Color32::WHITE));
            painter.circle_filled(a, 3.0, Color32::WHITE);
            painter.circle_filled(b, 3.0, Color32::WHITE);
        }
        Tool::Type | Tool::VerticalType | Tool::HorizontalTypeMask | Tool::VerticalTypeMask => {
            let r = Rect::from_two_pos(xf.to_screen(d.start[0] as f32, d.start[1] as f32), xf.to_screen(last[0] as f32, last[1] as f32));
            let pts = [r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom(), r.left_top()];
            painter.add(egui::Shape::line(pts.to_vec(), Stroke::new(1.0, Color32::WHITE)));
            painter.add(egui::Shape::dashed_line(&pts, Stroke::new(1.0, Color32::BLACK), 3.0, 3.0));
        }
        // The layers themselves follow the pointer (`move_ui`); the arrow only when they can't.
        Tool::Move if !crate::move_ui::showing(app) => {
            let from = xf.to_screen(d.start[0] as f32, d.start[1] as f32);
            let off = xf.to_screen(last[0] as f32, last[1] as f32) - from;
            painter.arrow(from, off, Stroke::new(2.0, crate::theme::Tokens::get(painter.ctx()).accent));
        }
        _ => {}
    }
}

/// Tools on which holding ⌥ (Alt) switches to the Eyedropper: a click or drag sets the
/// foreground colour, as with the Eyedropper itself (#417).
pub(crate) fn alt_samples(tool: Tool, mods: egui::Modifiers) -> bool {
    // Control+Alt is the brush-resize drag (`brush_resize`), not sampling.
    mods.alt && !mods.ctrl && matches!(tool, Tool::Brush | Tool::Pencil | Tool::Gradient | Tool::PaintBucket)
}

/// Preferences › Cursors › Other Cursors asks for the precise crosshair, or Caps Lock swaps the
/// standard cursor for it (and back), as in Photoshop (#4639).
fn precise_other_cursor(app: &PhotocraftApp) -> bool {
    (app.session.prefs().cursors.other == photocraft_engine::prefs::OtherCursor::Precise) != app.caps_lock
}

/// A pen tool's standard cursor: its toolbar icon with the nib (the Convert Point tool: the
/// anchor) on `p`. Draws it at `p` and returns the OS cursor to set (hidden).
pub(crate) fn pen_cursor(ctx: &egui::Context, tool: Tool, p: Pos2) -> egui::CursorIcon {
    // Where each icon's point is in its 24-unit box: the full-size pen's nib at (2.3, 2.3); the
    // Freeform and Curvature pens draw it at 0.68 × from (7, 7); Convert Point's apex anchor.
    let hot = match tool {
        Tool::FreeformPen | Tool::CurvaturePen => vec2(8.56, 8.56),
        Tool::ConvertPoint => vec2(12.0, 7.0),
        _ => vec2(2.3, 2.3),
    };
    crate::icons::cursor(ctx, crate::icons::tool_icon(tool), p, hot / 24.0, 22.0);
    egui::CursorIcon::None
}

/// The sampling cursor: a pipette whose tip is the sampled pixel. Draws it at `p` and returns the
/// OS cursor to set (hidden).
pub(crate) fn pipette_cursor(ctx: &egui::Context, p: Pos2) -> egui::CursorIcon {
    // The tip of the icon's pipette is at (2, 22) of its 24-unit box.
    crate::icons::cursor(ctx, "pipette", p, vec2(2.0, 22.0) / 24.0, 20.0);
    egui::CursorIcon::None
}

/// The Paint Bucket's cursor: a bucket whose pouring corner is the filled pixel. Draws it at `p`
/// and returns the OS cursor to set (hidden).
pub(crate) fn paint_bucket_cursor(ctx: &egui::Context, p: Pos2) -> egui::CursorIcon {
    // The left corner of the icon's bucket is at (2, 12) of its 24-unit box.
    crate::icons::cursor(ctx, "paint-bucket", p, vec2(2.0, 12.0) / 24.0, 22.0);
    egui::CursorIcon::None
}

/// Decided when the press starts, so ⌥ pressed or released mid-stroke never switches between
/// painting and sampling.
fn alt_eyedropper(app: &mut PhotocraftApp, ev: ToolEvent, mods: egui::Modifiers) -> bool {
    if matches!(ev, ToolEvent::Down { .. }) {
        app.alt_sampling = alt_samples(app.active_tool(), mods);
    }
    if !app.alt_sampling {
        return false;
    }
    match ev {
        // Without ⌥: the sample sets the foreground colour, whatever the Eyedropper would do.
        ToolEvent::Down { x, y, .. } | ToolEvent::Move { x, y, .. } => sample_eyedropper(app, x, y, egui::Modifiers::NONE),
        ToolEvent::Up { .. } => app.alt_sampling = false,
    }
    true
}

fn sample_eyedropper(app: &mut PhotocraftApp, x: f64, y: f64, mods: egui::Modifiers) {
    if let Some([r, g, b]) = eyedropper_color(app, x, y) {
        let key = if mods.alt { "background" } else { "foreground" };
        let _ = app.run("tools.setColors", json!({ key: [r, g, b, 1.0] }));
    }
}

/// The colour at document point (x, y) with the Eyedropper's Sample Size (the composite pixel,
/// or the average of the square around it) from the layers `sample_layer` names (`document.sampleColor`).
/// `None` off the image or over transparency.
fn sample_color(app: &mut PhotocraftApp, x: f64, y: f64, sample_layer: &str) -> Option<[f32; 3]> {
    if !(x.is_finite() && y.is_finite()) {
        return None;
    }
    let size = app.ui.tool_options.eyedropper_size;
    let v = app.run("document.sampleColor", json!({"x": x, "y": y, "size": size, "sampleLayer": sample_layer})).ok()?;
    match serde_json::from_value::<Vec<f32>>(v).ok()?[..] {
        [r, g, b, a] if a > 0.0 => Some([r, g, b]),
        _ => None,
    }
}

/// What the Eyedropper tool (and a painting tool's ⌥-click) picks at document point (x, y): its
/// options bar's Sample Size and Sample (#1649).
pub(crate) fn eyedropper_color(app: &mut PhotocraftApp, x: f64, y: f64) -> Option<[f32; 3]> {
    let layers = app.ui.tool_options.eyedropper_sample.clone();
    sample_color(app, x, y, &layers)
}

/// The active document's composite colour at document point (x, y), averaged over the
/// Eyedropper's Sample Size as Photoshop's dialog eyedroppers (Curves, Color Picker) do.
/// `None` off the image or over transparency.
pub(crate) fn composite_color(app: &mut PhotocraftApp, x: f64, y: f64) -> Option<[f32; 3]> {
    sample_color(app, x, y, "all")
}

/// The body of a `Move` for every tool: tracked position, ⇧ constraint, the moving layer, and so
/// on. It does not feed the live stroke, so the canvas can push a whole frame's recovered samples
/// and update the live stroke once (see `canvas_view`).
fn tool_move(app: &mut PhotocraftApp, x: f64, y: f64, pressure: f32, mods: egui::Modifiers) {
    let tool = app.active_tool();
    if tool.is_type() && app.drag.is_none() {
        crate::type_tool::pointer_move(app, x, y);
    }
    if tool == Tool::Pen && mods.alt {
        crate::vector_ui::pen_break_handle(app);
    }
    if tool == Tool::Pen {
        crate::vector_ui::pen_move(app, x, y);
    } else if tool == Tool::CurvaturePen {
        crate::vector_ui::curvature_move(app, x, y);
    }
    let zoom = app.point_zoom();
    if let Some(d) = app.drag.as_mut().filter(|d| d.reposition) {
        d.track(mods);
        d.shift_to([x, y]);
    } else if let Some(d) = &mut app.drag {
        d.track(mods);
        // ⇧: straight 0/45/90° strokes, 45° gradient angles (stroke_constraint.rs).
        let last = d.points.last().map_or(d.start, |p| [p[0], p[1]]);
        let [x, y] = crate::stroke_constraint::constrain(d.tool, &mut d.constrain, d.start, last, [x, y], mods.shift, zoom);
        let moved = d.points.last().is_none_or(|p| (p[0] - x).abs() + (p[1] - y).abs() > 0.25);
        // A held pen repeats its point now and then, so the stroke's time keeps advancing.
        let held = app.stylus.last_point_ms().is_some_and(|t| app.stylus.clock_ms - t >= HOLD_REPEAT_MS)
            && app.live_stroke.as_ref().is_some_and(|l| l.stroke.wants_time());
        if moved || held {
            d.points.push([x, y, pressure as f64]);
            app.stylus.record_point();
        }
    }
}

/// Tool state machine. Shared by mouse input and automation.
pub fn tool_event(app: &mut PhotocraftApp, ev: ToolEvent, mods: egui::Modifiers) {
    if crate::symmetry_ui::pointer(app, ev, crate::workspace_ui::sticky_mods(app, mods)) {
        return;
    }
    // An Alt+right-drag armed for this press (`paint_mouse`): taken before anything else can
    // consume the event, so it never outlives the press it was armed for (#297).
    let armed = std::mem::take(&mut app.brush_resize_armed);
    crate::transform_tool::end_if_left(app);
    let mods = crate::workspace_ui::sticky_mods(app, mods);
    if app.ui.transform.is_none() && crate::type_transform::pointer(app, ev, mods) {
        return;
    }
    // Content-Aware Move's Transform On Drop box takes every press until ↩ or Esc.
    if crate::cam_transform::pointer(app, ev, mods) {
        return;
    }
    // View › Snap / Snap To and smart guides (snap_ui.rs).
    let raw = ev;
    // The Move tool's marquee selecting layers takes its moves and release unsnapped (move_box.rs).
    if crate::move_box::pointer(app, raw) {
        return;
    }
    // A press anywhere but on the floating piece (or with ⇧ / ⌥, to draw) drops it first; the
    // Move tool (⌘ with a painting tool too, `event_tool`) drags it from anywhere.
    if let ToolEvent::Down { x, y, .. } = raw
        && app.session.active().is_some_and(|st| photocraft_engine::float_cmds::floating(st).is_some())
        && !crate::move_ui::moves_selected_pixels_with(app, event_tool(app, mods))
        && selection_drag_kind(app, app.ui.tool, [x, y], mods) != Some(true)
    {
        let _ = app.run("select.drop", json!({}));
    }
    let ev = crate::snap_ui::filter_event(app, ev, mods);
    // Free Transform picks the handle under the press where the user clicked: the snapped press
    // can land outside the corner's grab area and turn a corner drag into a move of the whole box.
    let transform_ev = if matches!(raw, ToolEvent::Down { .. }) { raw } else { ev };
    if crate::transform_tool::pointer(app, transform_ev, mods) {
        return;
    }
    if crate::distort_ui::pointer(app, ev, mods) {
        return;
    }
    if crate::perspective_crop_ui::pointer(app, ev) {
        return;
    }
    // Window › Modifier Keys: sticky Shift/⌘/⌥ act as held keys.
    let mods = crate::workspace_ui::sticky_mods(app, mods);
    // ⌘⌥⌃-click with any tool selects the topmost layer with pixels there (quick_pick.rs); it
    // must run before the ⌃⌥ brush resize below, which the same keys would trigger.
    if crate::quick_pick::pointer(app, ev, mods) {
        return;
    }
    // Control+Alt-drag or Alt+right-drag with a painting tool resizes the brush instead of
    // painting (#231, #297).
    if crate::brush_resize::pointer(app, ev, mods, armed) {
        return;
    }
    // ⌥ (Alt) with a painting tool is the Eyedropper for that press.
    if alt_eyedropper(app, ev, mods) {
        return;
    }
    // Move tool: ⇧ locks the axis, ⌥ duplicates (move_mods.rs).
    let ev = crate::move_mods::filter_event(app, ev, mods);
    // Direct Selection, and the Pen's ⌘ (Direct Selection) and ⌥ (Convert Point) modes (#790).
    if crate::direct_select::pointer(app, ev, mods) {
        return;
    }
    // Add / Delete Anchor Point tools (#1044).
    if crate::anchor_tools::pointer(app, ev, mods) {
        return;
    }
    // Ruler, Count and Note tools.
    if crate::analysis_ui::pointer(app, ev, mods) {
        return;
    }
    // Slice and Slice Select tools.
    if crate::slice_ui::pointer(app, ev, mods) {
        return;
    }
    // Frame tool: draw rectangular and elliptical frames.
    if crate::frame_tool::pointer(app, ev, mods) {
        return;
    }
    // Artboard tool: draw, select, move and resize boards.
    if crate::artboard_tool::pointer(app, ev, mods) {
        return;
    }
    // Crop tool: draw, move and resize the frame.
    if crate::crop_ui::pointer(app, ev, mods) {
        return;
    }
    // Gradient tool, live mode: draw and edit Gradient Fill layers.
    if crate::gradient_ui::pointer(app, ev, mods) {
        return;
    }
    // ⌘ with a selection tool over a guide takes the guide (#2690); the guide drag below moves it.
    if let ToolEvent::Down { x, y, .. } = raw
        && let Some((vertical, i)) = command_guide_at(app, app.active_tool(), [x, y], mods)
    {
        app.guide_drag = Some(crate::rulers::GuideDrag { vertical, index: Some(i), pos: if vertical { x } else { y } });
        crate::snap_ui::begin_guide(app, [x, y]);
        return;
    }
    if crate::lasso_ui::pointer(app, ev, mods) {
        return;
    }
    // Magnetic Lasso: fastening points and the border following the pointer.
    if crate::magnetic_lasso_ui::pointer(app, ev, mods) {
        return;
    }
    let tool = event_tool(app, mods);
    if tool == Tool::Eyedropper {
        match ev {
            ToolEvent::Down { x, y, .. } | ToolEvent::Move { x, y, .. } => {
                sample_eyedropper(app, x, y, mods);
                return;
            }
            ToolEvent::Up { .. } => return,
        }
    }
    // A locked layer (the Background without a selection, say): no drag, and Photoshop's message
    // once the pointer moves (a click says nothing). ⌘ with a selection tool here (#896); the Move
    // tool below, once Auto-Select has picked the layer.
    match ev {
        ToolEvent::Down { x, y, .. } if tool != Tool::Move && app.ui.transform.is_none() && crate::move_lock::blocked(app, tool, [x, y], mods) => {
            app.move_blocked = true;
            return;
        }
        ToolEvent::Move { .. } if app.move_blocked => {
            app.move_blocked = false;
            crate::move_lock::prompt(app);
            return;
        }
        ToolEvent::Up { .. } if app.move_blocked => {
            app.move_blocked = false;
            return;
        }
        _ => {}
    }
    // Move tool over a guide drags the guide (off the canvas deletes it).
    match ev {
        ToolEvent::Down { x, y, pressure } if tool == Tool::Move => {
            if let Some((vertical, i)) = crate::rulers::guide_at(app, x, y) {
                app.guide_drag = Some(crate::rulers::GuideDrag { vertical, index: Some(i), pos: if vertical { x } else { y } });
                return;
            }
            // Auto-Select (or ⌘-click while it is off) picks the layer under the pointer first
            // (not when the selected pixels move: those are the active layer's).
            if !crate::move_ui::moves_selected_pixels_with(app, tool) && app.ui.tool_options.move_auto_select != mods.command {
                let target = app.ui.tool_options.move_target.clone();
                let mode = if mods.shift { "add" } else { "replace" };
                let before = app.session.active().map(|st| st.selected_layers());
                // A locked layer under the pointer (the Background) can't move. Photoshop draws
                // the layer marquee instead of reporting the lock (#4058). Probe without
                // selecting so a locked hit doesn't become the selection first.
                let probe = app.run("layer.pickAt", json!({"x": x, "y": y, "target": target, "select": false}));
                let locked_hit = probe.as_ref().ok().and_then(|v| v["layer"].as_u64()).is_some_and(|id| {
                    app.session.active().is_some_and(|st| {
                        let id = photocraft_doc::LayerId(id);
                        let locks = st.doc.effective_locks(id);
                        locks.position || locks.all || st.doc.is_background(id)
                    })
                });
                if locked_hit || (probe.as_ref().is_ok_and(|v| v["layer"].is_null()) && crate::move_box::empty_at(app, [x, y])) {
                    if locked_hit {
                        // A click still selects the locked layer, as in Photoshop; a drag's
                        // marquee then replaces that selection on release.
                        let _ = app.run("layer.pickAt", json!({"x": x, "y": y, "target": target, "mode": mode}));
                    }
                    crate::move_box::begin(app, [x, y], mods.shift);
                    return;
                }
                let picked = app.run("layer.pickAt", json!({"x": x, "y": y, "target": target, "mode": mode}));
                // No pixels there at all (an empty spot, or off the canvas): drag a marquee that
                // selects the layers it touches instead (move_box.rs).
                if picked.is_ok_and(|v| v["layer"].is_null()) && crate::move_box::empty_at(app, [x, y]) {
                    crate::move_box::begin(app, [x, y], mods.shift);
                    return;
                }
                // Snapping (and the drag's box) started before the pick: point it at what moves.
                if app.session.active().map(|st| st.selected_layers()) != before {
                    crate::snap_ui::retarget_move(app, [x, y], mods);
                }
            }
            // A locked layer: no drag, and Photoshop's message once the pointer moves (`move_lock`).
            if crate::move_lock::blocked(app, tool, [x, y], mods) {
                app.move_blocked = true;
                return;
            }
            // With a selection: cut the selected pixels (⌥ copies them) and drag them as a floating
            // piece, from anywhere, as a marquee ⌘-drag does.
            if crate::move_ui::moves_selected_pixels_with(app, tool) {
                if crate::move_ui::float_selected(app, mods.alt, 0.0, 0.0) {
                    let mut d = Drag::new(tool, [x, y], vec![[x, y, pressure as f64]], mods, false);
                    d.sel_move = Some(true);
                    app.drag = Some(d);
                }
                return;
            }
        }
        ToolEvent::Move { x, y, .. } => {
            if let Some(d) = app.guide_drag.as_mut().filter(|d| d.index.is_some()) {
                d.pos = if d.vertical { x } else { y };
                return;
            }
        }
        ToolEvent::Up { x, y } => {
            if let Some(mut d) = app.guide_drag.filter(|d| d.index.is_some()) {
                app.guide_drag = None;
                d.pos = if d.vertical { x } else { y };
                crate::rulers::finish_drag(app, d);
                return;
            }
        }
        _ => {}
    }
    match ev {
        ToolEvent::Down { x, y, pressure } => {
            // Painting a type, shape, Smart Object or fill layer asks to rasterize it first
            // (⌥-click with the Clone Stamp or Healing Brush only sets the source).
            let sets_source = matches!(tool, Tool::CloneStamp | Tool::Healing) && mods.alt;
            // Painting on or moving a hidden layer is refused at the press, as in Photoshop (#571).
            if !sets_source && let Some(why) = hidden_target(app, tool) {
                app.ui.status = why.into();
                app.ui.status_error = true;
                return;
            }
            if !sets_source && crate::rasterize_prompt::intercept(app, tool, x, y, pressure) {
                return;
            }
            // ⌘ with a selection tool is the Move tool for the drag: outside the selection (or
            // without one) it moves the whole layer, ⌘⌥ a duplicate of it. It never combines
            // selections.
            if command_moves_layer(app, tool, [x, y], mods) {
                if mods.alt
                    && let Err(e) = app.run("layer.duplicate", json!({"inPlace": true}))
                {
                    app.ui.status = e;
                    app.ui.status_error = true;
                    return;
                }
                app.drag = Some(Drag::new(Tool::Move, [x, y], vec![[x, y, pressure as f64]], mods, false));
                return;
            }
            // Marquee / lasso inside the selection: drag the outline, or ⌘-drag to cut the selected
            // pixels into a floating piece (`select.float`) and drag that.
            if let Some(cut) = selection_drag_kind(app, tool, [x, y], mods) {
                // ⌘⌥ on a floating piece leaves it where it is and drags a copy of it.
                if cut
                    && mods.alt
                    && mods.command
                    && app.session.active().is_some_and(|st| photocraft_engine::float_cmds::floating(st).is_some())
                    && let Err(e) = app.run("select.drop", json!({}))
                {
                    app.ui.status = e;
                    app.ui.status_error = true;
                    return;
                }
                if cut
                    && app.session.active().is_some_and(|st| photocraft_engine::float_cmds::floating(st).is_none())
                    && let Err(e) = app.run("select.float", json!({"dx": 0, "dy": 0, "copy": mods.alt}))
                {
                    app.ui.status = e;
                    app.ui.status_error = true;
                    return;
                }
                let mut d = Drag::new(tool, [x, y], vec![[x, y, pressure as f64]], mods, false);
                d.sel_move = Some(cut);
                app.drag = Some(d);
                return;
            }
            match tool {
                Tool::PathSelection => {
                    let tol = 6.0 / f64::from(app.point_zoom().max(0.01));
                    if let Some(st) = app.session.active()
                        && let Some(id) = crate::vector_ui::pick_shape_at(&st.doc, [x, y], tol)
                    {
                        let _ = app.run("layer.select", json!({"layer": id.0}));
                    }
                }
                Tool::Pen => {
                    if !(mods.alt && crate::vector_ui::pen_grab_handle(app, x, y)) {
                        crate::vector_ui::pen_down(app, x, y);
                    }
                    return;
                }
                Tool::CurvaturePen => {
                    crate::vector_ui::curvature_down(app, x, y);
                    return;
                }
                Tool::CloneStamp | Tool::Healing if mods.alt => {
                    crate::retouch_ui::set_source(app, x, y);
                    return;
                }
                Tool::MagicWand => {
                    let o = app.ui.tool_options.clone();
                    let mode = selection_mode(app, mods);
                    let _ = app.run("select.magicWand", json!({"x": x.floor(), "y": y.floor(), "tolerance": o.tolerance, "contiguous": o.contiguous, "antiAlias": o.anti_alias, "sampleAllLayers": o.sample_all_layers, "mode": mode}));
                    return;
                }
                Tool::MagicEraser => {
                    crate::eraser_ui::click(app, tool, x, y);
                    return;
                }
                Tool::PaintBucket => {
                    let o = app.ui.tool_options.clone();
                    let contents = if o.bucket_fill_pattern { "pattern" } else { "foreground" };
                    let _ = app.run("paint.bucket", json!({"x": x.floor(), "y": y.floor(), "tolerance": o.tolerance, "contiguous": o.contiguous, "antiAlias": o.anti_alias, "sampleAllLayers": o.sample_all_layers, "opacity": o.fill_opacity, "contents": contents, "target": paint_target(app)}));
                    return;
                }
                Tool::RedEye => {
                    let o = app.ui.tool_options.clone();
                    let _ = app.run(
                        "paint.redEye",
                        json!({
                            "x": x.floor(),
                            "y": y.floor(),
                            "pupilSize": o.red_eye_pupil_size.clamp(1.0, 100.0),
                            "darken": o.red_eye_darken.clamp(0.0, 100.0),
                            "target": paint_target(app),
                        }),
                    );
                    return;
                }
                Tool::PolygonLasso => {
                    polygon_click(app, x, y, mods);
                    return;
                }
                Tool::Type | Tool::VerticalType | Tool::HorizontalTypeMask | Tool::VerticalTypeMask if crate::type_tool::pointer_down(app, x, y, mods) => {
                    return;
                }
                _ => {}
            }
            crate::paint_mouse::sync_tool_brush(app);
            // Clear mode (`~` held, or chosen in the options bar) erases with the tool's own brush.
            let clear = crate::paint_mouse::clears(app, tool);
            let erase = std::mem::take(&mut app.secondary_erase) || tool == Tool::Eraser || clear;
            // A 1 px pencil line depends on the pixels, not where inside the pixel the click
            // landed (#4071). Snap both ends of the stroke to the aliased tip's grid.
            let (x, y) = snap_aliased_tip(app, tool, x, y);
            // ⇧-click connects to the last endpoint, including its pen pressure (#2827), so
            // the renderer interpolates size and opacity across the segment.
            let active = app.session.active().map(|st| st.doc.id);
            let from = app.last_stroke_end.filter(|(doc, _)| mods.shift && crate::stroke_constraint::connects(tool) && Some(*doc) == active).map(|(_, p)| {
                let (sx, sy) = snap_aliased_tip(app, tool, p[0], p[1]);
                [sx, sy, p[2]]
            });
            let mut points = vec![[x, y, pressure as f64]];
            if let Some(p) = from {
                points.insert(0, p);
            }
            // ⌥ flips Dodge/Burn and Blur/Sharpen for this stroke.
            let tool = crate::retouch_ui::alt_flipped(tool, mods.alt);
            app.drag = Some(Drag { clear, ..Drag::new(tool, from.map_or([x, y], |p| [p[0], p[1]]), points, mods, erase) });
            app.trail = None;
            app.stylus.begin_stroke();
            if from.is_some() {
                app.stylus.record_point();
            }
            app.live_stroke = if strokes_live(tool) { begin_live_stroke(app) } else { None };
        }
        ToolEvent::Move { x, y, pressure } => {
            // Polygonal Lasso: ⌥ held while the button is down draws freehand into the polygon;
            // releasing ⌥ goes back to straight segments (the polygon stays open).
            if tool == Tool::PolygonLasso && mods.alt && app.drag.is_none() && !app.ui.polygon.is_empty() {
                if app.ui.polygon.last().is_none_or(|l| (l[0] - x).hypot(l[1] - y) >= 1.0) {
                    app.ui.polygon.push([x, y]);
                }
                return;
            }
            tool_move(app, x, y, pressure, mods);
            feed_live_stroke(app);
        }
        ToolEvent::Up { x, y } => {
            if tool.is_type()
                && let Some(e) = app.ui.text_edit.as_mut()
            {
                e.dragging = false;
                e.resize = None;
            }
            if tool == Tool::Pen {
                crate::vector_ui::pen_up(app);
            } else if tool == Tool::CurvaturePen {
                crate::vector_ui::curvature_up(app);
            }
            let zoom = app.point_zoom();
            let Some(mut d) = app.drag.take() else { return };
            d.track(mods);
            if d.reposition {
                d.shift_to([x, y]);
            }
            let last = d.points.last().map_or(d.start, |p| [p[0], p[1]]);
            let [x, y] = crate::stroke_constraint::constrain(d.tool, &mut d.constrain, d.start, last, [x, y], mods.shift, zoom);
            // A Move-tool click (released where it was pressed) selects, it never moves: snapping
            // the release point would otherwise nudge the layer onto a nearby edge.
            if d.tool == Tool::Move && d.points.len() < 2 && matches!(raw, ToolEvent::Up { x, y } if [x, y] == d.start) {
                app.move_preview = None;
                crate::move_mods::finish(app);
                if d.sel_move.is_some() {
                    finish_selection_drag(app, true, d.start, d.start);
                }
                return;
            }
            if d.points.last().is_none_or(|p| p[0] != x || p[1] != y) {
                d.points.push([x, y, d.points.last().map_or(1.0, |p| p[2])]);
                app.stylus.record_point();
            }
            finish_gesture(app, d);
            crate::move_mods::finish(app);
        }
    }
}

/// Why a press with `tool` must not start: it would paint on or move a hidden target layer (the
/// engine refuses the command; refusing the press keeps the live stroke from drawing first).
fn hidden_target(app: &PhotocraftApp, tool: Tool) -> Option<&'static str> {
    let cmd = match tool {
        Tool::Move => "layer.translate",
        // The live Gradient makes a new Gradient Fill layer above the target, which Photoshop
        // allows over a hidden layer; only the classic one paints the target itself.
        Tool::Gradient if !app.ui.tool_options.gradient_classic => return None,
        // Every painting command follows the same target (layer, mask or channel) as a stroke.
        t if t.is_brushlike() || matches!(t, Tool::Gradient | Tool::PaintBucket | Tool::MagicEraser) => "paint.stroke",
        _ => return None,
    };
    photocraft_engine::hidden_target::refusal(&app.session, cmd, &json!({}))
}

/// Is document point `p` inside the active document's selection?
fn inside_selection(app: &PhotocraftApp, p: [f64; 2]) -> bool {
    let Some(sel) = app.session.active().and_then(|st| st.doc.selection.as_ref()) else { return false };
    let mut v = [0.0f32];
    sel.read_pixel(p[0].floor() as i32, p[1].floor() as i32, &mut v);
    v[0] > 0.0
}

/// Does a press with `tool` at `p` drag the selection rather than draw a new one? `Some(true)`
/// moves the floating piece (⌘ cuts one first, ⌘⌥ copies; a floating piece moves with a plain
/// drag; the Polygonal Lasso and Magic Wand need ⌘, as a plain press there places or samples),
/// `Some(false)` just the outline (no ⇧ / ⌥ and the options bar on New Selection, so a combining
/// drag still draws).
pub fn selection_drag_kind(app: &PhotocraftApp, tool: Tool, p: [f64; 2], mods: egui::Modifiers) -> Option<bool> {
    // Click-driven selection tools (a press places a point or samples): only ⌘ drags the
    // selection, and not while a polygon is being drawn.
    let clicky = matches!(tool, Tool::PolygonLasso | Tool::MagicWand);
    if !(clicky || matches!(tool, Tool::RectMarquee | Tool::EllipseMarquee | Tool::Lasso)) || (clicky && !app.ui.polygon.is_empty()) {
        return None;
    }
    // ⌘ cuts, ⌘⌥ copies (`select.float`'s `copy`).
    let cut = mods.command && !mods.shift;
    if let Some(f) = app.session.active().and_then(photocraft_engine::float_cmds::floating) {
        let on = inside_selection(app, [p[0] - f64::from(f.offset.0), p[1] - f64::from(f.offset.1)]);
        return (on && !mods.shift && (!mods.alt || mods.command) && (cut || !clicky)).then_some(true);
    }
    if !inside_selection(app, p) {
        return None;
    }
    if cut {
        return Some(true);
    }
    (!clicky && !mods.command && selection_mode(app, mods) == "replace").then_some(false)
}

/// The tool a pointer event with `mods` goes to: ⌘ held is the Move tool (`hold_keys::cmd_moves`).
/// The canvas resolves the held key before the event (`tool_override`); automation and tests send
/// the modifier with the event.
pub(crate) fn event_tool(app: &PhotocraftApp, mods: egui::Modifiers) -> Tool {
    match app.active_tool() {
        t if app.tool_override.is_none() && mods.command && crate::hold_keys::cmd_moves(t) && app.ui.transform.is_none() && app.ui.text_edit.is_none() => {
            Tool::Move
        }
        t => t,
    }
}

/// Does a ⌘ (⌘⌥) press with selection tool `tool` at `p` move the whole layer (a duplicate with
/// ⌥), as the Move tool would? Outside the selection or without one; inside it, ⌘ drags the
/// selected pixels instead (`selection_drag_kind`).
pub(crate) fn command_moves_layer(app: &PhotocraftApp, tool: Tool, p: [f64; 2], mods: egui::Modifiers) -> bool {
    let selection_tool = matches!(tool, Tool::RectMarquee | Tool::EllipseMarquee | Tool::Lasso | Tool::PolygonLasso | Tool::MagicWand);
    let floating = app.session.active().is_some_and(|st| photocraft_engine::float_cmds::floating(st).is_some());
    selection_tool && mods.command && !mods.shift && !floating && app.ui.polygon.is_empty() && !inside_selection(app, p)
}

/// The guide a ⌘ press with selection tool `tool` at `p` drags (vertical?, index), as the Move
/// tool's press would (#2690): ⌘ is the Move tool, and a guide under the pointer comes before the
/// layer or the selected pixels. Not while a polygon or lasso outline is being drawn.
pub(crate) fn command_guide_at(app: &PhotocraftApp, tool: Tool, p: [f64; 2], mods: egui::Modifiers) -> Option<(bool, usize)> {
    let selection_tool = matches!(tool, Tool::RectMarquee | Tool::EllipseMarquee | Tool::Lasso | Tool::PolygonLasso | Tool::MagicWand);
    let idle = app.ui.polygon.is_empty() && !crate::lasso_ui::active(app) && app.ui.transform.is_none();
    (selection_tool && mods.command && idle).then(|| crate::rulers::guide_at(app, p[0], p[1])).flatten()
}

/// Photoshop's cursor over a selection: what a press (or the drag under way) would do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelCursor {
    /// A marquee or lasso inside the selection moves the outline (the move cursor, #1428), also
    /// while it is being dragged.
    Outline,
    /// Arrow with scissors: the Move tool (⌘ with a marquee) cuts and moves the selected pixels.
    Cut,
    /// Arrow with a second arrow: ⌥ lifts a copy of them instead.
    Copy,
    /// Plain arrow: a floating piece, moved again by a plain drag until it is dropped.
    Piece,
    /// Hollow arrowhead: the selected pixels are being dragged.
    Dragging,
}

/// The cursor at document point `p` with `tool` in effect and `mods` held (`None`: not over a
/// selection the press would move). The Move tool moves the selected pixels from anywhere
/// (`move_ui::moves_selected_pixels_with`); a selection tool from inside the selection, or on the
/// floating piece (`selection_drag_kind`).
pub fn selection_cursor(app: &PhotocraftApp, tool: Tool, p: [f64; 2], mods: egui::Modifiers) -> Option<SelCursor> {
    if let Some(d) = &app.drag {
        return match d.sel_move {
            Some(true) => Some(SelCursor::Dragging),
            Some(false) => Some(SelCursor::Outline),
            None => None,
        };
    }
    let floating = app.session.active().and_then(photocraft_engine::float_cmds::floating).is_some();
    let pixels = if tool == Tool::Move {
        crate::move_ui::moves_selected_pixels_with(app, tool)
    } else {
        match selection_drag_kind(app, tool, p, mods)? {
            false => return Some(SelCursor::Outline),
            true => true,
        }
    };
    if !pixels {
        return None;
    }
    // ⌥ (⌘⌥ with a selection tool) floats a copy, of the layer's pixels or of the piece.
    let copy = mods.alt && (tool == Tool::Move || mods.command);
    Some(match (floating, copy) {
        (_, true) => SelCursor::Copy,
        (true, false) => SelCursor::Piece,
        (false, false) => SelCursor::Cut,
    })
}

/// Whole-pixel offset of a selection drag in progress (`Drag::sel_move`).
pub(crate) fn selection_drag_delta(app: &PhotocraftApp) -> Option<(i32, i32)> {
    let d = app.drag.as_ref().filter(|d| d.sel_move.is_some())?;
    let end = d.points.last().map_or(d.start, |p| [p[0], p[1]]);
    Some(((end[0] - d.start[0]).round().clamp(-1e7, 1e7) as i32, (end[1] - d.start[1]).round().clamp(-1e7, 1e7) as i32))
}

/// Where the selection is drawn relative to where it is: a floating piece's offset plus a drag
/// in progress.
fn selection_shown_offset(app: &PhotocraftApp) -> Option<(i32, i32)> {
    crate::move_ui::floating_offset(app).or_else(|| selection_drag_delta(app))
}

/// End of a selection drag: move the outline, or the floating piece. A click without moving
/// deselects, like a marquee click (a click on a floating piece leaves it floating, unless it never
/// moved: then it is put back, so Undo isn't spent on it).
fn finish_selection_drag(app: &mut PhotocraftApp, floating: bool, start: [f64; 2], end: [f64; 2]) {
    let (dx, dy) = ((end[0] - start[0]).round(), (end[1] - start[1]).round());
    let unmoved = app.session.active().and_then(photocraft_engine::float_cmds::floating).is_some_and(|f| f.offset == (0, 0));
    let r = match (floating, dx == 0.0 && dy == 0.0) {
        (true, true) if unmoved => app.run("select.drop", json!({})),
        (true, true) => return,
        (false, true) if app.session.is_enabled("select.deselect") => app.run("select.deselect", json!({})),
        (false, true) => return,
        (true, false) => app.run("select.float", json!({"dx": dx, "dy": dy})),
        (false, false) => app.run("select.transformSelection", json!({"dx": dx, "dy": dy})),
    };
    if let Err(e) = r {
        app.ui.status = e;
        app.ui.status_error = true;
    }
}

/// Commit a Brush, Pencil or Eraser drag. The live stroke already rendered it: its pixels are
/// offered to the commit (`Session::offer_live_stroke`), so release costs no second render, and
/// the canvas then refreshes only what it hadn't shown yet (`CommittedStroke`).
fn commit_brush_stroke(app: &mut PhotocraftApp, d: &Drag) {
    let live = app.live_stroke.take();
    let mut p = stroke_params(app, d, &app.stylus.stroke_points(&d.points));
    if let Some(seed) = live.as_ref().and_then(|l| l.stroke.seed()) {
        p["seed"] = json!(seed);
    }
    let Some(mut l) = live else {
        let _ = app.run(stroke_command(app, d.tool), p);
        return;
    };
    // Points the live stroke hasn't rendered yet (the release frame's last moves).
    let t0 = app.stylus.point_ms(0);
    let rest: Vec<_> = (l.fed..d.points.len())
        .filter_map(|i| {
            let q = d.points.get(i)?;
            let t = app.stylus.stroke.get(i).or(app.stylus.stroke.last()).copied().unwrap_or_default();
            let mut sp = photocraft_engine::paint::StrokePoint::new(q[0], q[1], q[2] as f32);
            (sp.tilt_x, sp.tilt_y, sp.rotation) = (t[0], t[1], t[2]);
            sp.time = app.stylus.point_ms(i) - t0;
            Some(sp)
        })
        .collect();
    if !rest.is_empty() {
        match l.stroke.push(&rest) {
            Ok(r) => l.damage.push(r),
            // The live stroke can't show these points: the commit renders the stroke itself.
            Err(_) => {
                let _ = app.run(stroke_command(app, d.tool), p);
                return;
            }
        }
    }
    let LiveStroke { stroke, doc, key, damage, .. } = l;
    if let EngineStroke::Brush(stroke) = stroke
        && app.session.offer_live_stroke(*stroke).is_err()
    {
        app.session.clear_live_stroke();
    }
    let started = crate::gpu_canvas::now_ms();
    let result = app.run(stroke_command(app, d.tool), p);
    app.stroke_trace.stroke(crate::gpu_canvas::now_ms() - started);
    app.session.clear_live_stroke();
    let Ok(r) = result else { return };
    let revision = app.session.documents().iter().find(|st| st.doc.id == doc).map(|st| st.revision);
    if r.get("fromLive").and_then(serde_json::Value::as_bool) == Some(true)
        && let Some(revision) = revision
    {
        app.committed_stroke = Some(CommittedStroke { doc, revision, key, damage });
    } else {
        // Rendered again from scratch: the commit's damage rect refreshes the stroke.
        let shown = CommittedStroke { doc, revision: 0, key, damage };
        // Raw preview key 0 = the document itself (its colour display folded in).
        shown_as_document(app, doc, |k| shown.since(k).is_some());
    }
}

/// Pencil (and a Pencil-mode Eraser) dabs sit on the pixel grid. Clicks anywhere inside one
/// pixel must produce the same point, or a ⇧-click line changes with the sub-pixel position.
fn snap_aliased_tip(app: &PhotocraftApp, tool: Tool, x: f64, y: f64) -> (f64, f64) {
    let pencil = tool == Tool::Pencil || crate::eraser_ui::pencil_mode(app, tool);
    if !pencil {
        return (x, y);
    }
    let (cx, cy) = photocraft_engine::paint::grid_center(x, y, app.session.tools.brush.size);
    (f64::from(cx), f64::from(cy))
}

pub(crate) fn finish_gesture(app: &mut PhotocraftApp, d: Drag) {
    let mut end = d.points.last().copied().unwrap_or([d.start[0], d.start[1], 1.0]);
    let (sx, sy) = snap_aliased_tip(app, d.tool, end[0], end[1]);
    end[0] = sx;
    end[1] = sy;
    if let Some(floating) = d.sel_move {
        finish_selection_drag(app, floating, d.start, [end[0], end[1]]);
        return;
    }
    // Where the next ⇧-click line starts.
    if crate::stroke_constraint::connects(d.tool)
        && let Some(st) = app.session.active()
    {
        app.last_stroke_end = Some((st.doc.id, end));
    }
    // A live Clone Stamp preview ends here; the commit below replaces it (the Clone Stamp's and
    // History Brush's with the pixels it already drew).
    offer_live_retouch(app, &d);
    if live_retouch_command(d.tool).is_some() || crate::retouch_ui::dab_params(app, d.tool).is_some() || d.tool == Tool::ColorReplacement {
        app.live_stroke = None;
    }
    let finished = crate::eraser_ui::finish_stroke(app, d.tool, &d.points) || crate::retouch_ui::finish_stroke(app, d.tool, &d.points, d.modifiers);
    app.session.clear_live_stroke();
    if finished {
        return;
    }
    match d.tool {
        Tool::ObjectSelection => crate::retouch_ui::finish_object_selection(app, d.start, [end[0], end[1]], d.modifiers),
        Tool::FreeformPen => crate::vector_ui::freeform_finish(app, &d.points, d.lasso.as_ref().map_or(&[], |l| l.straight())),
        // Photoshop: a click without dragging opens Create Rectangle / Ellipse / … instead.
        t if crate::vector_ui::is_shape_tool(t)
            && crate::shape_dialog::is_click(d.start, &d.points, app.point_zoom())
            && crate::shape_dialog::open(app, t, d.start).is_some() => {}
        t if crate::vector_ui::is_shape_tool(t) => crate::vector_ui::finish_shape(app, t, d.start, [end[0], end[1]], d.live),
        Tool::PathSelection => crate::vector_ui::path_selection_finish(app, d.start, [end[0], end[1]]),
        Tool::Type | Tool::VerticalType | Tool::HorizontalTypeMask | Tool::VerticalTypeMask => crate::type_tool::pointer_up(app, d.start, [end[0], end[1]]),
        Tool::Brush | Tool::Pencil | Tool::Eraser => commit_brush_stroke(app, &d),
        Tool::RectMarquee | Tool::EllipseMarquee => {
            // Exactly what the drag showed (`marquee_preview_px`); a click deselects.
            let Some([x0, y0, x1, y1]) = marquee_preview_px(&app.ui.tool_options, &d, app.point_zoom()) else {
                if app.session.is_enabled("select.deselect") {
                    let _ = app.run("select.deselect", json!({}));
                }
                return;
            };
            let mode = selection_mode(app, d.modifiers);
            let (aa, feather) = (app.ui.tool_options.anti_alias, app.ui.tool_options.feather);
            let _ = app.run("select.rect", json!({"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0, "mode": mode, "ellipse": d.tool == Tool::EllipseMarquee, "antiAlias": aa, "feather": feather}));
        }
        // A click (or the end of a drag) selects the line under the pointer, never deselects.
        Tool::SingleRowMarquee | Tool::SingleColumnMarquee => {
            let Some([x0, y0, x1, y1]) = single_line_drag_px(app, &d) else { return };
            let mode = selection_mode(app, d.modifiers);
            let feather = app.ui.tool_options.feather;
            // `line` names the History step after the tool (Single Row / Single Column Marquee).
            let line = if d.tool == Tool::SingleRowMarquee { "row" } else { "column" };
            let _ = app.run(
                "select.rect",
                json!({"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0, "mode": mode, "line": line, "antiAlias": false, "feather": feather}),
            );
        }
        Tool::Patch if crate::retouch_ui::patch_drags_selection(app, d.start, d.modifiers) => crate::retouch_ui::finish_patch(app, d.start, [end[0], end[1]]),
        Tool::ContentAwareMove if crate::retouch_ui::patch_drags_selection(app, d.start, d.modifiers) => {
            crate::retouch_ui::finish_content_aware_move(app, d.start, [end[0], end[1]])
        }
        Tool::Lasso | Tool::Patch | Tool::ContentAwareMove => {
            let pts: Vec<[f64; 2]> = d.points.iter().map(|p| [p[0], p[1]]).collect();
            if pts.len() >= 3 {
                let mode = selection_mode(app, d.modifiers);
                let o = &app.ui.tool_options;
                // The Patch and Content-Aware Move tools have no Feather in their options bar.
                let feather = if d.tool == Tool::Lasso { o.feather } else { 0.0 };
                let made = app.run("select.lasso", json!({"points": pts, "mode": mode, "antiAlias": o.anti_alias, "feather": feather}));
                // The lasso only outlines the patch; say that it is dragged next (#1715).
                if made.is_ok() && d.tool != Tool::Lasso {
                    crate::retouch_ui::patch_hint(app);
                }
            } else if app.session.is_enabled("select.deselect") {
                let _ = app.run("select.deselect", json!({}));
            }
        }
        Tool::Gradient if (end[0] - d.start[0]).abs() + (end[1] - d.start[1]).abs() >= 2.0 => {
            let o = app.ui.tool_options.clone();
            let _ = app.run(
                "paint.gradient",
                json!({"from": [d.start[0], d.start[1]], "to": [end[0], end[1]], "style": o.gradient_style, "reverse": o.gradient_reverse, "dither": o.gradient_dither, "opacity": o.fill_opacity, "mode": o.gradient_blend_mode.label(), "target": paint_target(app)}),
            );
        }
        Tool::Move => {
            let (dx, dy) = ((end[0] - d.start[0]).round(), (end[1] - d.start[1]).round());
            crate::move_ui::finish(app, dx, dy);
        }
        _ => {}
    }
}

/// Extra OS windows showing documents (multi-window / multi-monitor).
pub fn extra_windows(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let wins = app.ui.windows.clone();
    for w in wins.into_iter().filter(|w| w.open) {
        let Some(st) = app.session.documents().get(w.document) else { continue };
        let title = format!("{} — window {}", st.doc.name, w.id);
        let vid = egui::ViewportId::from_hash_of(("docwin", w.id));
        let mut builder = egui::ViewportBuilder::default().with_title(title).with_inner_size([800.0, 600.0]);
        if let Some([x, y]) = w.position {
            builder = builder.with_position([x, y]);
        }
        let mut view = w.view.clone();
        let mut close = false;
        ctx.show_viewport_immediate(vid, builder, |ui, _class| {
            if ui.ctx().input(|i| i.viewport().close_requested()) {
                close = true;
            }
            egui::CentralPanel::default().frame(egui::Frame::NONE.fill(crate::theme::Tokens::get(ui.ctx()).canvas)).show(ui, |ui| {
                let rect = ui.available_rect_before_wrap();
                view = canvas_view(app, ui, w.document, rect, view.clone(), false);
            });
        });
        if let Some(win) = app.ui.windows.iter_mut().find(|x| x.id == w.id) {
            win.view = view;
            if close {
                win.open = false;
            }
        }
    }
    app.ui.windows.retain(|w| w.open);
}

/// Selection mode from the options bar, overridden by modifier keys (⇧ add, ⌥ subtract, ⇧⌥ intersect).
/// The cursor badge announces the same mode (`tool_feedback`).
pub(crate) fn selection_mode(app: &PhotocraftApp, m: egui::Modifiers) -> &'static str {
    crate::tool_feedback::document_selection_mode(app, Tool::Lasso, m)
}

/// Shift constrains a Polygonal Lasso segment to Photoshop's 45° directions.
/// Work in document coordinates so zoom, view rotation and mirroring cannot skew it.
pub(crate) fn constrained_polygon_point(from: [f64; 2], point: [f64; 2]) -> [f64; 2] {
    let dx = point[0] - from[0];
    let dy = point[1] - from[1];
    if !dx.is_finite() || !dy.is_finite() {
        return point;
    }
    let length = dx.hypot(dy);
    if length == 0.0 {
        return point;
    }
    let angle = (dy.atan2(dx) / std::f64::consts::FRAC_PI_4).round() * std::f64::consts::FRAC_PI_4;
    [from[0] + length * angle.cos(), from[1] + length * angle.sin()]
}

/// A polygonal lasso click adds a vertex; clicking near the first vertex closes the polygon. The
/// first click fixes the selection mode; a new-selection polygon hides the old outline while it is
/// drawn, and replaces it in one history step when it closes.
fn polygon_click(app: &mut PhotocraftApp, x: f64, y: f64, mods: egui::Modifiers) {
    let close = app
        .ui
        .polygon
        .first()
        .is_some_and(|p0| app.ui.polygon.len() >= 3 && ((p0[0] - x).powi(2) + (p0[1] - y).powi(2)).sqrt() < 8.0 / app.point_zoom().max(0.01) as f64);
    if close {
        commit_polygon(app);
        return;
    }
    if app.ui.polygon.is_empty() {
        app.ui.polygon_doc = app.session.active().map(|st| st.doc.id);
        // ⌥ with nothing selected draws freehand; there is nothing to subtract from.
        let nothing_selected = app.session.active().is_none_or(|st| st.doc.selection.is_none());
        let intent = if nothing_selected { egui::Modifiers { alt: false, ..mods } } else { mods };
        app.ui.polygon_mode = selection_mode(app, intent).into();
    }
    // Only constrain segments after the first vertex; the first click still sets selection intent.
    // Alt-drag is freehand and must not snap to an angle.
    let point = if mods.shift && !mods.alt { app.ui.polygon.last().copied().map_or([x, y], |last| constrained_polygon_point(last, [x, y])) } else { [x, y] };
    app.ui.polygon.push(point);
}

/// Remove the Polygonal Lasso's last vertex (⌫, Delete or a right-click while drawing, #1229);
/// removing the only one cancels the polygon. False when no polygon is being drawn.
pub fn polygon_retract(app: &mut PhotocraftApp) -> bool {
    if app.ui.polygon.pop().is_none() {
        return false;
    }
    if app.ui.polygon.is_empty() {
        app.ui.polygon_mode.clear();
    }
    true
}

/// The polygon in progress belongs to the document it was started on (`UiState::polygon_doc`):
/// once another document becomes active it is dropped, so its outline doesn't show on (or get
/// committed to) the other document, and the selection it would have replaced is shown again.
/// Switching tools is handled in `PhotocraftApp::sync_views`. Like the Lasso's
/// ([`crate::lasso_ui::cancel_stale`]) and the Magnetic Lasso's ([`crate::magnetic_lasso_ui::frame`]).
pub fn polygon_cancel_stale(app: &mut PhotocraftApp) {
    let active = app.session.active().map(|st| st.doc.id);
    if app.ui.polygon.is_empty() {
        app.ui.polygon_doc = None;
    } else if app.ui.polygon_doc.is_some_and(|d| Some(d) != active) {
        app.ui.polygon.clear();
        app.ui.polygon_mode.clear();
        app.ui.polygon_doc = None;
    }
}

/// A new-selection polygonal (or magnetic) lasso is being drawn, so the selection it will replace
/// is hidden.
pub fn polygon_replaces_selection(app: &PhotocraftApp) -> bool {
    (!app.ui.polygon.is_empty() && app.ui.polygon_mode == "replace") || crate::magnetic_lasso_ui::replaces_selection(app)
}

/// Close the polygonal lasso and make the selection in the mode it started in.
pub fn commit_polygon(app: &mut PhotocraftApp) {
    let pts = std::mem::take(&mut app.ui.polygon);
    let mode = std::mem::take(&mut app.ui.polygon_mode);
    if pts.len() >= 3 {
        let mode = if mode.is_empty() { selection_mode(app, egui::Modifiers::NONE).to_owned() } else { mode };
        let o = &app.ui.tool_options;
        let _ = app.run("select.lasso", json!({"points": pts, "mode": mode, "antiAlias": o.anti_alias, "feather": o.feather}));
    }
}

/// Apply the crop tool's rectangle.
pub fn commit_crop(app: &mut PhotocraftApp) {
    // Never another document's frame (#1918).
    crate::crop_ui::cancel_stale(app);
    // Committing ends the box's own Undo/Redo: from here Edit › Undo reverts the document.
    crate::crop_ui::clear_history(app);
    let Some(r) = app.ui.crop_rect.take() else { return };
    // The crop's default mode leaves the view upright (`crop_mode`).
    crate::crop_mode::upright(app);
    // The untouched default frame around the whole canvas crops nothing (Photoshop's ↵ on it does
    // nothing); one framing the selection's bounds crops to them (#1789).
    if std::mem::take(&mut app.crop.default_frame) {
        let whole = app.session.active().map(|st| st.doc.bounds()).map(|b| [f64::from(b.x0), f64::from(b.y0), f64::from(b.x1), f64::from(b.y1)]);
        if whole.is_none_or(|b| b == r) {
            return;
        }
    }
    let (x, y) = (r[0].round(), r[1].round());
    let (w, h) = ((r[2] - r[0]).round().max(1.0), (r[3] - r[1]).round().max(1.0));
    let delete = app.ui.tool_options.crop_delete;
    let mut p = json!({"x": x, "y": y, "width": w, "height": h, "deleteCroppedPixels": delete});
    // A turned frame (#1792): the document turns with it, then is cropped to it. (The untouched
    // default frame above is never turned: turning it makes it a real frame.)
    let angle = crate::crop_ui::angle(app);
    app.ui.crop_angle = 0.0;
    if angle != 0.0 {
        p["angle"] = json!(angle);
    }
    // W x H x Resolution (#2443): resample to that size and set the resolution.
    crate::crop_size::commit_params(app, &mut p);
    if app.run("image.crop", p).is_ok()
        && let Some(i) = app.session.active_index()
    {
        app.ui.views[i].fit_pending = true;
    }
}

/// "mask" when the Layers panel targets (or the canvas shows) the active layer's mask, or the layer
/// is an adjustment layer with one (it has no pixels: #4298), else "pixels".
pub fn paint_target(app: &PhotocraftApp) -> serde_json::Value {
    use photocraft_engine::channel_cmds::ChannelTarget;
    let Some(st) = app.session.active() else { return json!("pixels") };
    // A targeted alpha channel (Channels panel) or Quick Mask mode wins over the layer.
    match st.channel_view.target {
        ChannelTarget::Alpha(i) if i < st.doc.channels.len() => return json!({ "channel": i }),
        ChannelTarget::Composite if st.doc.quick_mask.is_some() => return json!("quickMask"),
        ChannelTarget::LayerMask => return json!("mask"),
        _ => {}
    }
    let has_mask = st.active_layer.and_then(|id| st.doc.layer(id)).is_some_and(|l| l.mask.is_some());
    // Viewing the mask (⌥-click its thumbnail, #196) paints the mask.
    let viewing = photocraft_engine::mask_view_cmds::current(st).is_some();
    let mask = app.ui.mask_target || viewing || crate::adjustment_layer_active(st);
    json!(if mask && has_mask { "mask" } else { "pixels" })
}

#[cfg(test)]
mod tap_pressure_tests {
    #[test]
    fn fallback_click_pressure_matches_pen_or_mouse() {
        let mut s = crate::stylus::Stylus::default();
        assert_eq!(s.pressure(), 1.0, "mouse clicks paint at full pressure");
        s.feed.set(Some(crate::stylus::PenSample { pressure: 0.18, ..Default::default() }));
        assert_eq!(s.pressure(), 0.18, "a tap without a canvas drag still has pen pressure");
        s.use_pressure = false;
        assert_eq!(s.pressure(), 1.0, "Use Tablet Pressure off stays full pressure");
    }
}

#[cfg(test)]
mod tabs_tests;

#[cfg(test)]
mod tests {
    #[test]
    fn display_image_matches_egui_unmultiplied_conversion() {
        // Several parallel pieces and a ragged one; translucent, out-of-range and NaN values.
        let n: usize = 3 * 16 * 1024 + 77;
        let px: Vec<[f32; 4]> = (0..n)
            .map(|i| {
                let v = match i % 9 {
                    0 => f32::NAN,
                    1 => -0.5,
                    2 => 3.25,
                    _ => (i % 1001) as f32 / 1000.0,
                };
                [v, (i % 257) as f32 / 256.0, 1.0 - (i % 113) as f32 / 112.0, (i % 6) as f32 / 5.0]
            })
            .collect();
        let buf = photocraft_compose::Buffer { rect: photocraft_geom::Rect::new(0, 0, n as i32, 1), px };
        let reference = |img: photocraft_raster::Rgba8Image| egui::ColorImage::from_rgba_unmultiplied([img.width as usize, img.height as usize], &img.pixels);
        assert_eq!(super::display_image(None, &buf).pixels, reference(buf.to_rgba8()).pixels);
        let mut doc = photocraft_doc::Document::new("t", photocraft_geom::Size::new(1, 1), photocraft_doc::ColorMode::Rgb, photocraft_doc::SampleType::U8);
        doc.icc_profile = Some(photocraft_cms::Builtin::AdobeRgbCompat.profile().to_bytes());
        let mut color = photocraft_engine::color_cmds::ColorState::default();
        color.settings.monitor_profile = "srgb".into();
        let d = color.canvas_display(&doc).unwrap();
        assert!(!d.is_identity());
        assert_eq!(super::display_image(Some(&d), &buf).pixels, reference(d.to_rgba8(&buf)).pixels);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn every_filter_previews_off_the_ui_thread_and_a_newer_value_cancels_the_old() {
        // Dragging Gaussian Blur's radius to 1000 px froze the window: the preview ran inline in
        // the frame, and every slider step waited for the previous blur to finish.
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 64, "height": 48})).unwrap();
        app.background_jobs = true;
        crate::filter_dialog::open(&mut app, "filter.blur.gaussianBlur").unwrap();
        let ctx = egui::Context::default();
        let wait_for = |app: &mut PhotocraftApp, radius: f64| {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while app.filter_preview.as_ref().is_none_or(|p| p.key.params["radius"].as_f64() != Some(radius)) {
                assert!(std::time::Instant::now() < deadline, "no preview for radius {radius}");
                ensure_filter_preview(app, 0, &ctx);
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        };
        // The first frame hands the blur to the worker and returns.
        assert!(ensure_filter_preview(&mut app, 0, &ctx).is_none());
        assert!(app.filter_preview_worker.busy(), "the preview computes on the worker");
        let first = app.ui.dialogs[0].fields["radius"].as_f64().unwrap();
        wait_for(&mut app, first);
        assert!(app.filter_preview.as_ref().unwrap().result.is_some());

        // A preview for old values occupies the worker until it is cancelled.
        let st = app.session.active().unwrap();
        let d = &app.ui.dialogs[0];
        let stale = crate::filter_dialog::FilterPreviewKey {
            doc: st.doc.id,
            revision: st.revision,
            dialog: d.id,
            active: st.active_layer,
            target: st.channel_view.target,
            command: "filter.blur.gaussianBlur".into(),
            params: json!({"radius": 999.0}),
            k: 1,
        };
        let (told, cancelled) = std::sync::mpsc::channel();
        app.filter_preview_worker
            .start(stale, ctx.clone(), move |job| {
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                while !job.cancelled() && std::time::Instant::now() < deadline {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                let _ = told.send(job.cancelled());
                crate::filter_preview_worker::Computed { result: None, ms: 0.0 }
            })
            .unwrap();
        // The slider moves on and stops: once the value settles, the stale preview is cancelled
        // instead of being waited out.
        app.ui.dialogs[0].fields.insert("radius".into(), json!(12.0));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let was_cancelled = loop {
            ensure_filter_preview(&mut app, 0, &ctx);
            if let Ok(c) = cancelled.try_recv() {
                break c;
            }
            assert!(std::time::Instant::now() < deadline, "the stale preview was never cancelled");
            std::thread::sleep(std::time::Duration::from_millis(5));
        };
        assert!(was_cancelled, "the stale preview stopped because it was cancelled, not by timing out");
        wait_for(&mut app, 12.0);
        assert!(app.filter_preview.as_ref().unwrap().result.is_some());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn indexed_preview_discards_a_finished_job_after_slider_change_or_reopening() {
        for reopen in [false, true] {
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
            app.run("file.new", json!({"width": 8, "height": 8})).unwrap();
            app.background_jobs = true;
            crate::filter_dialog::open(&mut app, "image.mode.indexedColor").unwrap();
            let d = app.ui.dialogs.first().unwrap();
            let st = app.session.active().unwrap();
            let old = crate::filter_dialog::FilterPreviewKey {
                doc: st.doc.id,
                revision: st.revision,
                dialog: d.id,
                active: st.active_layer,
                target: st.channel_view.target,
                command: "image.mode.indexedColor".into(),
                params: app.with_mask_target("image.mode.indexedColor", crate::filter_dialog::params_of(&d.fields)),
                k: 1,
            };
            let (release, wait) = std::sync::mpsc::channel();
            let ctx = egui::Context::default();
            app.filter_preview_worker
                .start(old.clone(), ctx.clone(), move |_| {
                    wait.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
                    crate::filter_preview_worker::Computed { result: None, ms: 0.0 }
                })
                .unwrap();
            if reopen {
                app.ui.dialogs.clear();
                crate::filter_preview_worker::discard_closed(&mut app);
                crate::filter_dialog::open(&mut app, "image.mode.indexedColor").unwrap();
            } else {
                app.ui.dialogs.first_mut().unwrap().fields.insert("colors".into(), json!(16));
            }
            assert!(ensure_filter_preview(&mut app, 0, &ctx).is_none());
            assert!(app.filter_preview.is_none());
            release.send(()).unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while app.filter_preview.is_none() {
                assert!(std::time::Instant::now() < deadline);
                ensure_filter_preview(&mut app, 0, &ctx);
                if let Some(preview) = &app.filter_preview {
                    assert_ne!(preview.key, old, "an outdated preview was published");
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            assert_eq!(app.session.active().unwrap().doc.mode, photocraft_color::ColorMode::Rgb);
        }
    }

    #[test]
    fn windows_uses_a_visible_crosshair() {
        // #737: Windows' inverting crosshair vanishes over mid-grey; use a black/white glyph.
        let ctx = egui::Context::default();
        let painter = egui::Painter::new(ctx, egui::LayerId::background(), egui::Rect::EVERYTHING);
        let p = egui::pos2(10.0, 10.0);
        assert_eq!(super::visible_crosshair(egui::CursorIcon::Crosshair, &painter, p, true), egui::CursorIcon::None);
        assert_eq!(super::visible_crosshair(egui::CursorIcon::Crosshair, &painter, p, false), egui::CursorIcon::Crosshair);
        assert_eq!(super::visible_crosshair(egui::CursorIcon::Move, &painter, p, true), egui::CursorIcon::Move);
    }

    use super::*;

    /// #569: displays 1 (sRGB) and 4 (Display P3) side by side, and a document filled with an
    /// sRGB colour.
    fn app_on_two_displays() -> PhotocraftApp {
        use photocraft_engine::color_cmds::resolve_profile;
        use photocraft_engine::display_color::Display;
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 64, "height": 64})).unwrap();
        app.run("edit.fill", json!({"color": "#cc8040"})).unwrap();
        let icc = |id: &str| Some(resolve_profile(id, None, None).unwrap().to_bytes());
        app.session.color.set_displays(Ok(vec![
            Display { id: 1, name: "A".into(), frame: [0.0, 0.0, 100.0, 100.0], profile_name: None, icc: icc("srgb") },
            Display { id: 4, name: "B".into(), frame: [100.0, 0.0, 100.0, 100.0], profile_name: None, icc: icc("display-p3") },
        ]));
        app
    }

    #[test]
    #[ignore = "manual texture-retention measurement on six 6 MP documents"]
    fn closed_canvas_memory_measurement() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        let ctx = egui::Context::default();
        ctx.input_mut(|i| i.max_texture_side = 8192);
        for cycle in 1..=6 {
            app.run("file.new", json!({"width":3000,"height":2000})).unwrap();
            ensure_texture(&mut app, &ctx, 0, None).unwrap();
            // Drain uploaded image deltas just as a renderer does; count live texture owners.
            ctx.tex_manager().write().take_delta().clear();
            app.run("file.close", json!({})).unwrap();
            let bytes: usize = ctx.tex_manager().read().allocated().map(|(_, m)| m.bytes_used()).sum();
            println!("closed cycle={cycle} texture_bytes={bytes} rss={:?}", photocraft_testkit::perf::current_rss_bytes());
        }
    }

    #[test]
    fn closed_documents_release_navigator_but_keep_open_document_textures() {
        let mut app = app_on_two_displays();
        let ctx = egui::Context::default();
        let first = app.session.active().unwrap().doc.clone();
        let a = ensure_texture(&mut app, &ctx, 0, Some(1)).unwrap().0;
        // The same handle ownership as navigator_texture, without requiring a GPU adapter.
        let tex = ctx.load_texture("navigator-test", egui::ColorImage::filled([64, 64], egui::Color32::WHITE), TextureOptions::LINEAR);
        let nav = tex.id();
        app.navigator_textures.insert(first.id, (std::sync::Arc::downgrade(&first), 0, 0, tex));
        app.run("file.new", json!({"width":32,"height":32})).unwrap();
        let other = app.session.active().unwrap().doc.id;
        let b = ensure_texture(&mut app, &ctx, 1, Some(1)).unwrap().0;
        drop(app.session.close(0));
        app.sync_views();
        assert!(ctx.tex_manager().read().meta(a).is_none());
        assert!(ctx.tex_manager().read().meta(nav).is_none());
        assert!(ctx.tex_manager().read().meta(b).is_some());
        assert!(app.canvases.keys().all(|(id, _)| *id == other));
        assert!(app.navigator_textures.is_empty());
        app.session.add_document((*first).clone(), None);
        app.sync_views();
        assert_ne!(ensure_texture(&mut app, &ctx, 1, Some(1)).unwrap().0, a, "reopening a preserved ID creates a fresh texture");
    }

    #[test]
    fn closed_documents_release_cpu_canvas_textures() {
        let mut app = app_on_two_displays();
        let ctx = egui::Context::default();
        let original = app.session.active().unwrap().doc.clone();
        for cycle in 0..12 {
            if cycle > 0 {
                app.session.add_document((*original).clone(), None);
                app.sync_views();
            }
            let a = ensure_texture(&mut app, &ctx, 0, Some(1)).unwrap().0;
            let b = ensure_texture(&mut app, &ctx, 0, Some(4)).unwrap().0;
            assert_ne!(a, b);
            app.run("file.close", json!({})).unwrap();
            assert!(app.canvases.is_empty(), "closed canvas metadata retained at cycle {cycle}");
            assert!(ctx.tex_manager().read().meta(a).is_none(), "first display texture retained");
            assert!(ctx.tex_manager().read().meta(b).is_none(), "second display texture retained");
        }
    }

    #[test]
    fn cpu_canvas_textures_are_per_display() {
        // Two windows of one document on different displays: each has its own texture of
        // monitor values, and drawing them in turn re-renders neither.
        let mut app = app_on_two_displays();
        let ctx = egui::Context::default();
        let id = app.session.documents()[0].doc.id;
        let a = ensure_texture(&mut app, &ctx, 0, Some(1)).unwrap().0;
        let b = ensure_texture(&mut app, &ctx, 0, Some(4)).unwrap().0;
        assert_ne!(a, b);
        app.perf.last_refresh = "";
        for _ in 0..3 {
            assert_eq!(ensure_texture(&mut app, &ctx, 0, Some(1)).unwrap().0, a);
            assert_eq!(ensure_texture(&mut app, &ctx, 0, Some(4)).unwrap().0, b);
        }
        assert_eq!(app.perf.last_refresh, "", "no re-render");
        let doc = app.session.documents()[0].doc.clone();
        assert!(canvas_display(&app, &doc, Some(1)).0.unwrap().is_identity());
        assert!(!canvas_display(&app, &doc, Some(4)).0.unwrap().is_identity());
        assert_ne!(app.canvases[&(id, 1)].tex_preview_key, app.canvases[&(id, 4)].tex_preview_key);
        // An edit updates both.
        app.run("edit.fill", json!({"color": "#2060c0"})).unwrap();
        ensure_texture(&mut app, &ctx, 0, Some(1));
        ensure_texture(&mut app, &ctx, 0, Some(4));
        let rev = app.session.documents()[0].revision;
        assert_eq!((app.canvases[&(id, 1)].tex_revision, app.canvases[&(id, 4)].tex_revision), (rev, rev));
    }

    #[test]
    fn cpu_canvas_shows_crisp_pixels_when_zoomed_in() {
        // #4810: above 100% the CPU canvas shows each pixel as a square, like the GPU canvas's
        // sharp nearest filter; below 100% it still filters. Both the first upload and a
        // damage-rect update carry the sampling, since the renderer takes it from each delta.
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 2, "height": 2})).unwrap();
        let ctx = egui::Context::default();
        let tex = ensure_texture(&mut app, &ctx, 0, None).unwrap().0;
        let uploads = |ctx: &egui::Context| {
            let mut delta = ctx.tex_manager().write().take_delta();
            let seen = delta.set.get(&tex).into_iter().flatten().map(|d| (d.pos.is_some(), d.options)).collect::<Vec<_>>();
            delta.clear();
            seen
        };
        let crisp = TextureOptions { magnification: egui::TextureFilter::Nearest, ..TextureOptions::LINEAR };
        assert_eq!(uploads(&ctx), vec![(false, crisp)]);
        app.run("select.rect", json!({"x": 0, "y": 0, "width": 1, "height": 1})).unwrap();
        ensure_texture(&mut app, &ctx, 0, None).unwrap();
        uploads(&ctx);
        app.run("edit.fill", json!({"color": "#ff0000"})).unwrap();
        assert_eq!(ensure_texture(&mut app, &ctx, 0, None).unwrap().0, tex);
        assert_eq!(uploads(&ctx), vec![(true, crisp)], "a partial update keeps the crisp sampling");
    }

    #[test]
    fn cpu_canvas_of_a_reduced_document_stays_bilinear() {
        // Over MAX_TEXTURE the texture holds one texel per `factor` pixels: a texel is not a
        // document pixel, so showing it as a square would draw blocks that aren't in the image.
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": MAX_TEXTURE + 1, "height": 2})).unwrap();
        let ctx = egui::Context::default();
        let (tex, scale) = ensure_texture(&mut app, &ctx, 0, None).unwrap();
        assert!(scale < 1.0);
        let mut delta = ctx.tex_manager().write().take_delta();
        let uploads: Vec<_> = delta.set.get(&tex).into_iter().flatten().map(|d| d.options).collect();
        delta.clear();
        assert_eq!(uploads, vec![TextureOptions::LINEAR]);
    }

    #[test]
    fn gpu_display_luts_are_per_display() {
        let Ok(rs) = std::panic::catch_unwind(|| egui_kittest::wgpu::create_render_state(crate::gpu_canvas::wgpu_setup(), Default::default())) else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let mut app = app_on_two_displays();
        app.set_wgpu(rs);
        let doc = app.session.documents()[0].doc.clone();
        let key = doc.id.0;
        // sRGB on the sRGB display needs no LUT; on the P3 display it does.
        assert_eq!(sync_display_lut(&mut app, &doc, key, Some(1)), 0);
        assert_eq!(sync_display_lut(&mut app, &doc, key, Some(4)), 1);
        let gpu = app.gpu.clone().unwrap();
        let (s1, s4) = (gpu.display_lut_signature(key, 1).unwrap(), gpu.display_lut_signature(key, 4).unwrap());
        assert_ne!(s1.0, s4.0);
        assert!(gpu.has_display_lut(key, 4) && !gpu.has_display_lut(key, 1));
        // Drawing one window leaves the other's LUT alone.
        for _ in 0..3 {
            assert_eq!(sync_display_lut(&mut app, &doc, key, Some(1)), 0);
            assert_eq!(sync_display_lut(&mut app, &doc, key, Some(4)), 1);
        }
        assert_eq!((gpu.display_lut_signature(key, 1), gpu.display_lut_signature(key, 4)), (Some(s1), Some(s4)));
        assert!(gpu.has_display_lut(key, 4));
        // The shared texture's key is the same for both displays.
        assert_eq!(texture_key(canvas_display(&app, &doc, Some(1)).0.as_deref()), texture_key(canvas_display(&app, &doc, Some(4)).0.as_deref()));
    }

    /// The navigator's update after an edit (its image with the edit's damage, grown by the
    /// effect reach, rendered over it) equals a new thumbnail of the edited document.
    #[test]
    fn navigator_damage_updates_match_a_full_thumbnail() {
        use photocraft_compose::{render_reduced_damage, thumbnail_buffer, thumbnail_size};
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 1500, "height": 900, "background": "white"})).unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        app.run("select.rect", json!({"x": 200, "y": 150, "width": 300, "height": 200})).unwrap();
        app.run("edit.fill", json!({"color": "#3070a0"})).unwrap();
        app.run("select.deselect", json!({})).unwrap();
        app.run("layer.layerStyle.dropShadow", json!({"distance": 30, "size": 20})).unwrap();
        let steps = [
            ("layer.translate", json!({"dx": 37, "dy": -11})),
            ("layer.setProps", json!({"opacity": 0.4})),
            ("paint.stroke", json!({"points": [[700, 500], [900, 520]], "size": 25})),
            ("layer.setProps", json!({"blend": "Multiply"})),
            ("layer.new.layer", json!({})),
        ];
        for (id, p) in steps {
            let before = app.session.active().unwrap().doc.clone();
            app.run(id, p).unwrap();
            let st = app.session.active().unwrap();
            let d = st.last_damage.unwrap_or_else(|| panic!("{id}: a bounded damage"));
            let (w, h) = thumbnail_size(&st.doc, NAVIGATOR_SIDE);
            let mut shown = thumbnail_buffer(&before, NAVIGATOR_SIDE);
            assert_eq!((shown.rect.width(), shown.rect.height()), (w, h));
            if !d.is_empty() {
                let patch = render_reduced_damage(&st.doc, w, h, d.inflate(effect_reach(&st.doc.layers)));
                let r = patch.rect;
                for y in 0..r.height() as usize {
                    let at = (r.y0 as usize + y) * w as usize + r.x0 as usize;
                    shown.px[at..at + r.width() as usize].copy_from_slice(&patch.px[y * r.width() as usize..][..r.width() as usize]);
                }
            }
            assert!(shown.px == thumbnail_buffer(&st.doc, NAVIGATOR_SIDE).px, "{id}: the updated navigator differs from a new thumbnail");
        }
    }

    #[test]
    fn tab_context_uses_clicked_document_and_existing_close_commands() {
        let items = tab_context_items(2, 3, true);
        assert_eq!(items.iter().map(|(_, id, _, _)| *id).collect::<Vec<_>>(), ["file.close", "file.closeOthers", "file.closeAll", "file.revealInFinder"]);
        assert_eq!(items[0].2, json!({"document": 2}));
        assert_eq!(items[1].2, json!({"document": 2}));
        assert_eq!(items[2].2, json!({}));
        assert_eq!(items[3].2, json!({"document": 2}));
        assert!(items[3].3, "a saved document can be revealed in the file manager");
        assert!(!tab_context_items(2, 3, false)[3].3, "an unsaved document cannot");
        assert!(!tab_context_items(0, 1, false)[1].3);
        assert!(items.iter().all(|(_, id, _, _)| photocraft_engine::commands::find(id).is_some()));
    }

    #[test]
    fn tab_context_reveal_names_the_platform_file_manager() {
        let (label, id, _, _) = &tab_context_items(0, 1, true)[3];
        assert_eq!((*label, *id), (crate::menus::reveal_label(), "file.revealInFinder"));
    }

    /// Photoshop's document tab × is after the title on Windows and Linux, before it on macOS (#619).
    #[test]
    fn document_tab_close_button_sits_on_the_platform_side() {
        use egui::os::OperatingSystem as Os;
        use egui_kittest::kittest::Queryable;
        for (os, after_title) in [(Os::Windows, true), (Os::Nix, true), (Os::Mac, false)] {
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
            app.run("file.new", json!({"width": 8, "height": 8})).unwrap();
            app.sync_views();
            // Drawn from the second frame, once the Pro theme is set.
            let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(600.0, 40.0)).build_ui_state(
                |ui, (app, ready): &mut (PhotocraftApp, bool)| {
                    if *ready {
                        tabs(app, ui);
                    }
                },
                (app, false),
            );
            PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::Pro);
            h.ctx.set_os(os);
            h.state_mut().1 = true;
            h.run_steps(2);
            let tab = h.get_by_label_contains("Untitled @").rect();
            let x = h.get_by_label("Close").rect();
            assert!(tab.contains_rect(x), "{os:?}: the × is inside its tab");
            assert_eq!(x.center().x > tab.center().x, after_title, "{os:?}: × at {x:?} in tab {tab:?}");
            h.get_by_label("Close").click();
            h.run_steps(2);
            assert!(h.state().0.session.documents().is_empty(), "{os:?}: the × closes the document");
        }
    }

    /// Middle-clicking a document tab closes it (#2570), in both tab strips, and the tab under the
    /// pointer is the one that goes even when another document is active.
    #[test]
    fn middle_clicking_a_tab_closes_that_document() {
        use egui_kittest::kittest::Queryable;
        for kind in [crate::theme::ThemeKind::Studio, crate::theme::ThemeKind::Pro] {
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
            for name in ["keep", "doomed", "active"] {
                app.run("file.new", json!({"width": 8, "height": 8, "name": name})).unwrap();
            }
            app.sync_views();
            assert_eq!(app.session.active_index(), Some(2));
            let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(700.0, 40.0)).build_ui_state(
                |ui, (app, ready): &mut (PhotocraftApp, bool)| {
                    if *ready {
                        tabs(app, ui);
                    }
                },
                (app, false),
            );
            PhotocraftApp::setup_context(&h.ctx, kind);
            h.state_mut().1 = true;
            h.run_steps(2);
            h.get_by_label_contains("doomed").click_button(egui::PointerButton::Middle);
            h.run_steps(2);
            let names: Vec<String> = h.state().0.session.documents().iter().map(|d| d.doc.name.clone()).collect();
            assert_eq!(names, ["keep", "active"], "{kind:?}: the middle-clicked tab closed");
            // A clean document closes straight away; a dirty one would park in `discard`.
            assert!(h.state().0.discard.is_none(), "{kind:?}: nothing to prompt about");
        }
    }

    /// A middle click goes through the same guarded `file.close` as the ×, so unsaved changes
    /// still get the Save / Don’t Save / Cancel prompt instead of being thrown away (#2570).
    #[test]
    fn middle_clicking_an_edited_tab_prompts_instead_of_discarding() {
        use egui_kittest::kittest::Queryable;
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 8, "height": 8, "name": "edited"})).unwrap();
        app.run("edit.fill", json!({"color": "#ff0000"})).unwrap();
        app.sync_views();
        let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(700.0, 40.0)).build_ui_state(
            |ui, (app, ready): &mut (PhotocraftApp, bool)| {
                if *ready {
                    tabs(app, ui);
                }
            },
            (app, false),
        );
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::Pro);
        h.state_mut().1 = true;
        h.run_steps(2);
        h.get_by_label_contains("edited").click_button(egui::PointerButton::Middle);
        h.run_steps(2);
        assert!(h.state().0.discard.is_some(), "the unsaved-changes prompt must come up");
        assert_eq!(h.state().0.session.documents().len(), 1, "and the document is still open");
    }

    #[test]
    fn spectrum_document_tabs_meet_canvas_without_a_gap() {
        for kind in [crate::theme::ThemeKind::Pro, crate::theme::ThemeKind::ProMedium] {
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
            app.run("file.new", json!({"width": 80, "height": 60})).unwrap();
            app.sync_views();
            let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(680.0, 480.0)).build_ui_state(
                |ui, (app, ready): &mut (PhotocraftApp, bool)| {
                    if *ready {
                        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| document_area(app, ui));
                    }
                },
                (app, false),
            );
            PhotocraftApp::setup_context(&h.ctx, kind);
            h.state_mut().1 = true;
            h.run_steps(3);
            let app = &h.state().0;
            let strip = app.tab_strip.as_ref().expect("a document opens with tabs").rect;
            let gap = app.last_canvas_rect.top() - strip.bottom();
            assert!(gap.abs() < 0.1, "{kind:?}: document strip should touch canvas, got {gap} pt");
        }
    }

    #[test]
    fn large_tabs_preference_increases_tab_strip_height() {
        use egui_kittest::kittest::Queryable;
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 8, "height": 8})).unwrap();
        app.sync_views();
        let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(600.0, 60.0)).build_ui_state(
            |ui, (app, ready): &mut (PhotocraftApp, bool)| {
                if *ready {
                    tabs(app, ui);
                }
            },
            (app, false),
        );
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::Pro);
        h.state_mut().1 = true;
        h.run_steps(2);
        let normal_h = h.get_by_label_contains("Untitled @").rect().height();
        assert_eq!(normal_h, 26.0);

        h.state_mut().0.session.execute("prefs.set", json!({"values": {"workspace.largeTabs": true}})).unwrap();
        h.run_steps(2);
        let large_h = h.get_by_label_contains("Untitled @").rect().height();
        assert_eq!(large_h, 34.0);
    }

    #[test]
    fn tab_close_others_prompts_for_unsaved_nonactive_document() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 8, "height": 8, "name": "Keep"})).unwrap();
        app.run("file.new", json!({"width": 8, "height": 8, "name": "Edited"})).unwrap();
        app.run("edit.fill", json!({"color": "#ff0000"})).unwrap();
        assert_eq!(app.session.active_index(), Some(1));
        let (_, id, params, _) = tab_context_items(0, 2, false)[1].clone();
        crate::menus::invoke(&mut app, &egui::Context::default(), id, params).unwrap();
        assert!(app.discard.is_some(), "close others must ask before discarding the edited tab");
        assert_eq!(app.session.documents().len(), 2);
    }

    /// The pure half of tab drag-to-reorder (UI-217-6): where a drop lands and which
    /// `document.move` target that is.
    #[test]
    fn tab_drop_index_and_move_target_cover_reorder_drops() {
        let r = |x: f32| Rect::from_min_size(egui::pos2(x, 0.0), egui::vec2(100.0, 26.0));
        let placed = vec![(0, r(0.0)), (1, r(100.0)), (2, r(200.0))];
        assert_eq!(tab_drop_index(&placed, 10.0), 0);
        assert_eq!(tab_drop_index(&placed, 160.0), 2);
        assert_eq!(tab_drop_index(&placed, 350.0), 3, "past the end lands after the last shown tab");
        assert_eq!(tab_move_target(0, 3), 2);
        assert_eq!(tab_move_target(2, 0), 0);
        assert_eq!(tab_move_target(1, 1), 1, "before itself: no move");
        assert_eq!(tab_move_target(1, 2), 1, "just after itself: no move");
        assert_eq!(tab_move_target(0, 1), 0);
    }

    /// Dragging a tab onto another's slot reorders the open documents (UI-217-6), in both the
    /// studio and the Photoshop-style strips.
    #[test]
    fn dragging_a_tab_reorders_the_documents() {
        use egui_kittest::kittest::Queryable;
        for kind in [crate::theme::ThemeKind::Studio, crate::theme::ThemeKind::Pro] {
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
            for name in ["a", "b", "c"] {
                app.run("file.new", json!({"width": 8, "height": 8, "name": name})).unwrap();
            }
            app.sync_views();
            let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(700.0, 40.0)).build_ui_state(
                |ui, (app, ready): &mut (PhotocraftApp, bool)| {
                    if *ready {
                        tabs(app, ui);
                    }
                },
                (app, false),
            );
            PhotocraftApp::setup_context(&h.ctx, kind);
            h.state_mut().1 = true;
            h.run_steps(2);
            let b = h.get_by_label_contains("b").rect();
            let a = h.get_by_label_contains("a").rect();
            let target = egui::pos2(a.left() + 2.0, a.center().y);
            h.drag_at(b.center());
            h.run_steps(1);
            h.hover_at(target);
            h.run_steps(1);
            h.drop_at(target);
            h.run_steps(2);
            let names: Vec<String> = h.state().0.session.documents().iter().map(|d| d.doc.name.clone()).collect();
            assert_eq!(names, ["b", "a", "c"], "{kind:?}: the dragged tab took the dropped slot");
        }
    }

    /// Dragging a tab off the strip floats its document in its own OS window instead of
    /// reordering (#3733), in both strips; a short wobble below the strip still reorders.
    #[test]
    fn dragging_a_tab_off_the_strip_floats_it_in_a_window() {
        use egui_kittest::kittest::Queryable;
        let strip = Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(700.0, 26.0));
        assert!(!torn_off(strip, egui::pos2(50.0, 13.0)));
        assert!(!torn_off(strip, egui::pos2(50.0, 26.0 + TEAR_OFF - 1.0)), "a wobble below the strip still reorders");
        assert!(torn_off(strip, egui::pos2(50.0, 26.0 + TEAR_OFF + 1.0)));
        assert!(torn_off(strip, egui::pos2(50.0, -TEAR_OFF - 1.0)), "dragged up, off the window's top");
        for kind in [crate::theme::ThemeKind::Studio, crate::theme::ThemeKind::Pro] {
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
            for name in ["a", "b", "c"] {
                app.run("file.new", json!({"width": 8, "height": 8, "name": name})).unwrap();
            }
            app.sync_views();
            let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(700.0, 300.0)).build_ui_state(
                |ui, (app, ready): &mut (PhotocraftApp, bool)| {
                    if *ready {
                        tabs(app, ui);
                    }
                },
                (app, false),
            );
            PhotocraftApp::setup_context(&h.ctx, kind);
            h.state_mut().1 = true;
            h.run_steps(2);
            let b = h.get_by_label_contains("b").rect();
            let target = egui::pos2(b.center().x, b.bottom() + 150.0);
            h.drag_at(b.center());
            h.run_steps(1);
            h.hover_at(target);
            h.run_steps(1);
            h.drop_at(target);
            h.run_steps(2);
            let app = &h.state().0;
            let names: Vec<&str> = app.session.documents().iter().map(|d| d.doc.name.as_str()).collect();
            assert_eq!(names, ["a", "b", "c"], "{kind:?}: tearing off doesn't reorder");
            assert_eq!(app.ui.windows.iter().map(|w| w.document).collect::<Vec<_>>(), [1], "{kind:?}: b floats in its own window");
            assert_eq!(app.session.active_index(), Some(1), "{kind:?}: the floated document is active");
        }
    }

    #[test]
    fn clone_overlay_is_circular_and_respects_visibility() {
        let ctx = egui::Context::default();
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 64, "height": 64})).unwrap();
        app.run("cloneSource.set", json!({"source": [16,16]})).unwrap();
        app.ui.tool = Tool::CloneStamp;
        app.session.tools.brush.size = 20.0;
        let xf =
            ViewXform { rect: Rect::from_min_size(Pos2::ZERO, vec2(100.0, 100.0)), zoom: 1.0, center: [32.0, 32.0], flip: false, rotation: 0.0, aspect: 1.0 };
        let revision = app.session.active().unwrap().revision;
        let mut output = ctx.run_ui(Default::default(), |ui| {
            let ctx = ui.ctx();
            let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("clone-test")));
            draw_clone_preview(&mut app, &painter, &xf, [32.0, 32.0], None);
        });
        output.textures_delta.clear();
        let mesh = output
            .shapes
            .iter()
            .find_map(|s| match &s.shape {
                egui::Shape::Mesh(m) => Some(m),
                _ => None,
            })
            .unwrap();
        assert_eq!(mesh.indices.len(), 64 * 3);
        let center = xf.to_screen(32.0, 32.0);
        assert!(mesh.vertices.iter().all(|v| v.pos.distance(center) <= 10.01));
        assert_eq!(app.session.active().unwrap().revision, revision);
        assert!(app.session.presets.clone.active().anchor.is_none());
        let ordinary_key = app.clone_preview.as_ref().unwrap().key.clone();
        app.ui.view.pattern_preview = true;
        let mut periodic_output = ctx.run_ui(Default::default(), |ui| {
            let painter = ui.ctx().layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("clone-test")));
            draw_clone_preview(&mut app, &painter, &xf, [32.0, 32.0], None);
        });
        assert_ne!(app.clone_preview.as_ref().unwrap().key, ordinary_key, "toggling Pattern Preview invalidates source pixels");
        assert!(!periodic_output.textures_delta.set.is_empty(), "the overlay texture is refreshed without a document edit");
        periodic_output.textures_delta.clear();
        assert_eq!(app.session.active().unwrap().revision, revision);
        app.session.presets.clone.overlay.show = false;
        let mut output = ctx.run_ui(Default::default(), |ui| {
            let ctx = ui.ctx();
            let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("clone-test")));
            draw_clone_preview(&mut app, &painter, &xf, [32.0, 32.0], None);
        });
        output.textures_delta.clear();
        assert!(output.shapes.iter().all(|s| !matches!(s.shape, egui::Shape::Mesh(_))));
    }

    #[test]
    fn clone_overlay_blend_mode_mixes_with_the_image_under_the_brush() {
        let ctx = egui::Context::default();
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 64, "height": 64})).unwrap();
        app.run("cloneSource.set", json!({"source": [16,16]})).unwrap();
        app.ui.tool = Tool::CloneStamp;
        app.session.tools.brush.size = 20.0;
        let xf =
            ViewXform { rect: Rect::from_min_size(Pos2::ZERO, vec2(100.0, 100.0)), zoom: 1.0, center: [32.0, 32.0], flip: false, rotation: 0.0, aspect: 1.0 };
        // The centre texel of the overlay texture uploaded for `mode`.
        let mut centre = |mode: &str| {
            app.run("cloneSource.overlay", json!({"blend": mode})).unwrap();
            let mut output = ctx.run_ui(Default::default(), |ui| {
                let painter = ui.ctx().layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("clone-test")));
                draw_clone_preview(&mut app, &painter, &xf, [32.0, 32.0], None);
            });
            // The brush's 20 × 20 px square (the font atlas may upload too).
            let image = output.textures_delta.set.values().flatten().find_map(|d| match &d.image {
                egui::ImageData::Color(c) if c.size == [20, 20] => Some(c.clone()),
                _ => None,
            });
            output.textures_delta.clear();
            let image = image.expect("the overlay uploads a texture");
            image.pixels[image.width() * (image.height() / 2) + image.width() / 2]
        };
        // A white source over the white image: Normal shows white, Difference black.
        assert_eq!(centre("normal"), Color32::WHITE);
        assert_eq!(centre("difference"), Color32::BLACK, "Difference turns black where source and image match");
        assert_eq!(centre("lighten"), Color32::WHITE);
    }

    /// A shaped tip clips the Clone Source overlay to its outline, not to the brush circle.
    #[test]
    fn clone_overlay_is_clipped_to_a_shaped_tip() {
        let ctx = egui::Context::default();
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 64, "height": 64})).unwrap();
        app.run("cloneSource.set", json!({"source": [16,16]})).unwrap();
        app.ui.tool = Tool::CloneStamp;
        app.session.tools.brush.size = 20.0;
        app.session.tools.brush.roundness = 0.25;
        let xf =
            ViewXform { rect: Rect::from_min_size(Pos2::ZERO, vec2(100.0, 100.0)), zoom: 1.0, center: [32.0, 32.0], flip: false, rotation: 0.0, aspect: 1.0 };
        let mut output = ctx.run_ui(Default::default(), |ui| {
            let painter = ui.ctx().layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("clone-test")));
            draw_clone_preview(&mut app, &painter, &xf, [32.0, 32.0], None);
        });
        let image = output
            .textures_delta
            .set
            .values()
            .flatten()
            .find_map(|d| match &d.image {
                egui::ImageData::Color(c) if c.size == [20, 20] => Some(c.clone()),
                _ => None,
            })
            .expect("the overlay uploads a texture");
        output.textures_delta.clear();
        let alpha = |x: usize, y: usize| image.pixels[y * image.width() + x].a();
        assert_eq!(alpha(10, 10), 255, "the tip's centre shows the source");
        assert_eq!(alpha(10, 2), 0, "inside the circle but outside the flat tip: clipped");
        assert_eq!(alpha(1, 10), 255, "the tip's long axis stays");
    }

    #[test]
    fn wayland_start_screen_hint_does_not_claim_file_drop_works() {
        let x11 = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        let wayland = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services { is_wayland: true, ..Default::default() });
        assert!(start_screen_drop_hint(&x11).starts_with("Drop"));
        // Pasting a copied image file works on Wayland (#338): the hint says how.
        let hint = start_screen_drop_hint(&wayland);
        assert!(!hint.contains("Drop"), "{hint}");
        assert!(hint.ends_with(&format!("paste a copied image with {}.", crate::notices::paste_hint(&wayland))), "{hint}");
    }

    /// The Home screen's × takes a file off Recent without opening it, and its right-click menu
    /// removes one entry or clears the list; the preferences keep the change (#2691).
    #[test]
    fn home_recent_entries_can_be_removed() {
        use egui_kittest::kittest::Queryable;
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        for name in ["c.tif", "b.psd", "a.png"] {
            app.push_recent(&format!("/tmp/photocraft-2691/{name}"));
        }
        let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(900.0, 760.0)).build_ui_state(
            |ui, (app, ready): &mut (PhotocraftApp, bool)| {
                if *ready {
                    egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| document_area(app, ui));
                }
            },
            (app, false),
        );
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::default());
        h.state_mut().1 = true;
        h.run_steps(3);
        let listed = |h: &egui_kittest::Harness<'_, (PhotocraftApp, bool)>| {
            let app = &h.state().0;
            assert_eq!(app.session.prefs().file_handling.recent_files, app.ui.recent_files, "stored in the preferences");
            app.ui.recent_files.iter().map(|p| crate::file_open::display_name(p)).collect::<Vec<_>>()
        };
        assert_eq!(listed(&h), ["a.png", "b.psd", "c.tif"]);
        assert!(h.query_by_label("Remove b.psd from Recent").is_none(), "the × shows on the hovered row only");

        h.get_by_label("b.psd").hover();
        h.run_steps(2);
        h.get_by_label("Remove b.psd from Recent").click();
        h.run_steps(2);
        assert_eq!(listed(&h), ["a.png", "c.tif"]);
        assert!(h.state().0.session.documents().is_empty(), "the file was not opened");

        h.get_by_label("c.tif").click_secondary();
        h.run_steps(2);
        h.get_by_label("Remove from Recent").click();
        h.run_steps(2);
        assert_eq!(listed(&h), ["a.png"]);

        h.get_by_label("a.png").click_secondary();
        h.run_steps(2);
        h.get_by_label("Clear Recent Files").click();
        h.run_steps(2);
        assert!(listed(&h).is_empty());
        assert!(h.state().0.session.documents().is_empty());
    }

    #[test]
    fn pointer_moves_are_bounded_by_the_press_and_release() {
        use egui::{Event, PointerButton, pos2};
        let button = |pos, pressed| Event::PointerButton { pos, button: PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE };
        let events = [
            Event::PointerMoved(pos2(1.0, 0.0)),
            button(pos2(2.0, 0.0), true),
            Event::PointerMoved(pos2(3.0, 0.0)),
            Event::PointerMoved(pos2(4.0, 0.0)),
            button(pos2(5.0, 0.0), false),
            Event::PointerMoved(pos2(6.0, 0.0)),
        ];
        // The press is in this frame: only the moves between press and release count.
        assert_eq!(pointer_moves(&events, PointerButton::Primary, false), vec![pos2(3.0, 0.0), pos2(4.0, 0.0)]);
        // The button was already down when the frame began: the move before the release counts,
        // the one after it does not.
        assert_eq!(pointer_moves(&events, PointerButton::Primary, true), vec![pos2(1.0, 0.0), pos2(3.0, 0.0), pos2(4.0, 0.0)]);
    }

    #[test]
    fn freehand_tools_are_the_ones_that_follow_a_path() {
        for t in [
            Tool::Brush,
            Tool::Pencil,
            Tool::MixerBrush,
            Tool::ColorReplacement,
            Tool::Eraser,
            Tool::BackgroundEraser,
            Tool::CloneStamp,
            Tool::PatternStamp,
            Tool::Smudge,
            Tool::Dodge,
            Tool::Lasso,
            Tool::QuickSelection,
        ] {
            assert!(freehand_tool(t), "{t:?} paints or retouches along a path");
        }
        for t in [Tool::Move, Tool::Eyedropper, Tool::Gradient, Tool::Crop, Tool::RectMarquee, Tool::Type, Tool::Hand] {
            assert!(!freehand_tool(t), "{t:?} is driven by the pointer's latest position");
        }
    }

    #[test]
    fn clone_stamp_centre_appears_only_with_option() {
        assert!(!brush_tip_centre(Tool::CloneStamp, false, false));
        assert!(brush_tip_centre(Tool::CloneStamp, true, false));
        assert!(brush_tip_centre(Tool::Healing, true, false));
        assert!(!brush_tip_centre(Tool::Healing, false, false));
        assert!(brush_tip_centre(Tool::CloneStamp, false, true));
        assert!(!brush_tip_centre(Tool::PatternStamp, true, false), "Pattern Stamp keeps the brush cursor; Option does not switch it to a clone-source mark");
        assert!(!brush_tip_centre(Tool::QuickSelection, false, true));
        assert!(brush_tip_centre(Tool::BackgroundEraser, false, false));
    }

    #[test]
    fn brush_tip_centre_follows_the_crosshair_preference_at_any_size() {
        // A large tip used to show the centre mark whatever Show Crosshair in Brush Tip said.
        for tool in [Tool::Brush, Tool::Eraser, Tool::PatternStamp, Tool::Dodge] {
            assert!(!brush_tip_centre(tool, false, false), "{tool:?}: off");
            assert!(brush_tip_centre(tool, false, true), "{tool:?}: on");
        }
    }

    #[test]
    fn view_transform_roundtrip() {
        for flip in [false, true] {
            let xf = ViewXform {
                rect: Rect::from_min_size(pos2(100.0, 50.0), vec2(800.0, 600.0)),
                zoom: 2.5,
                center: [320.0, 240.0],
                flip,
                rotation: 0.0,
                aspect: 1.0,
            };
            let s = xf.to_screen(10.0, 20.0);
            let d = xf.to_doc(s);
            assert!((d[0] - 10.0).abs() < 1e-3 && (d[1] - 20.0).abs() < 1e-3);
            assert_eq!(xf.to_screen(320.0, 240.0), xf.rect.center());
            // Flipped, document x grows to the left.
            assert_eq!(xf.to_screen(330.0, 240.0).x > xf.rect.center().x, !flip);
            let r = xf.doc_rect(DRect::new(0, 0, 10, 10));
            assert!(r.width() > 0.0 && r.height() > 0.0);
        }
    }

    #[test]
    fn eyedropper_ring_colors_pair_the_sampled_and_current_colours() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 40, "height": 20, "background": "transparent"})).unwrap();
        app.run("shape.create", json!({"kind": "rect", "rect": [0, 0, 20, 20], "fill": "#ff0000"})).unwrap();
        app.run("tools.setColors", json!({"foreground": [0.0, 0.2, 1.0, 1.0]})).unwrap();
        let (new, current) = super::eyedropper_ring_colors(&mut app, 10.0, 10.0).expect("over the red square");
        let near = |c: egui::Color32, r: u8, g: u8, b: u8| {
            let d = |a: u8, b: u8| i32::from(a).abs_diff(i32::from(b));
            d(c.r(), r) <= 1 && d(c.g(), g) <= 1 && d(c.b(), b) <= 1
        };
        assert!(near(new, 255, 0, 0), "{new:?}");
        assert!(near(current, 0, 51, 255), "{current:?}");
        // Transparency and points outside the canvas have nothing to compare.
        assert!(super::eyedropper_ring_colors(&mut app, 30.0, 10.0).is_none());
        assert!(super::eyedropper_ring_colors(&mut app, -1.0, 10.0).is_none());
    }

    #[test]
    fn eyedropper_drag_updates_the_sampled_colour() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 40, "height": 20, "background": "transparent"})).unwrap();
        app.run("shape.create", json!({"kind": "rect", "rect": [0, 0, 20, 20], "fill": "#ff0000"})).unwrap();
        app.run("shape.create", json!({"kind": "rect", "rect": [20, 0, 20, 20], "fill": "#00ff00"})).unwrap();
        app.ui.tool = Tool::Eyedropper;

        tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 10.0, pressure: 1.0 }, egui::Modifiers::NONE);
        assert!(app.session.tools.foreground[0] > 0.99 && app.session.tools.foreground[1] < 0.01);

        tool_event(&mut app, ToolEvent::Move { x: 30.0, y: 10.0, pressure: 1.0 }, egui::Modifiers::NONE);
        assert!(app.session.tools.foreground[1] > 0.99 && app.session.tools.foreground[0] < 0.01);
    }

    /// #4298: an adjustment layer has no pixels, so after its icon thumbnail is clicked (the
    /// Layers panel no longer targets the mask) Fill, the fill keys and the Paint Bucket still
    /// edit its mask, as in Photoshop, instead of failing with "not a pixel layer".
    #[test]
    fn fills_on_an_adjustment_layer_edit_its_mask_when_the_icon_is_targeted() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 40, "height": 20, "background": "white"})).unwrap();
        app.run("layer.newAdjustmentLayer.curves", json!({})).unwrap();
        // Click the Curves icon thumbnail: the panel targets the layer, not its mask.
        app.ui.mask_target = false;
        app.sync_mask_targets();
        assert!(!app.ui.mask_target, "the panel's target is the icon");
        let mask_at = |app: &PhotocraftApp| {
            let st = app.session.active().unwrap();
            st.doc.layer(st.active_layer.unwrap()).unwrap().mask.as_ref().unwrap().surface.sample_channel(10, 10, 0)
        };
        assert_eq!(paint_target(&app), json!("mask"));
        assert!(crate::menus::is_enabled(&app, "edit.fill"), "Edit > Fill (⇧F5) is enabled");
        assert!(crate::menus::is_enabled(&app, "edit.fillForeground"));
        app.ui.tool = Tool::PaintBucket;
        app.run("tools.setColors", json!({"foreground": [0.0, 0.0, 0.0, 1.0], "background": [1.0, 1.0, 1.0, 1.0]})).unwrap();
        tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 10.0, pressure: 1.0 }, egui::Modifiers::NONE);
        tool_event(&mut app, ToolEvent::Up { x: 10.0, y: 10.0 }, egui::Modifiers::NONE);
        assert_eq!(mask_at(&app), 0.0, "the Paint Bucket filled the mask black");
        app.run("edit.fill", json!({"contents": "white"})).unwrap();
        assert_eq!(mask_at(&app), 1.0, "Edit > Fill filled the mask white");
        assert!(!app.ui.mask_target, "filling doesn't move the panel's target");
        // A pixel layer with an untargeted mask still paints its pixels.
        app.run("layer.new.layer", json!({})).unwrap();
        app.run("layer.layerMask.revealAll", json!({})).unwrap();
        app.ui.mask_target = false;
        assert_eq!(paint_target(&app), json!("pixels"));
        assert!(!app.layer_mask_targeted());
    }

    #[test]
    fn alt_with_a_painting_tool_samples_the_foreground_instead_of_painting() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 40, "height": 20, "background": "transparent"})).unwrap();
        app.run("shape.create", json!({"kind": "rect", "rect": [0, 0, 20, 20], "fill": "#ff0000"})).unwrap();
        app.run("shape.create", json!({"kind": "rect", "rect": [20, 0, 20, 20], "fill": "#00ff00"})).unwrap();
        let alt = egui::Modifiers::ALT;
        for tool in [Tool::Brush, Tool::Pencil, Tool::Gradient, Tool::PaintBucket] {
            app.ui.tool = tool;
            app.run("tools.setColors", json!({"foreground": [0.0, 0.0, 1.0, 1.0], "background": [1.0, 1.0, 1.0, 1.0]})).unwrap();
            let rev = app.session.active().unwrap().revision;
            tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 10.0, pressure: 1.0 }, alt);
            assert_eq!(app.session.tools.foreground, [1.0, 0.0, 0.0, 1.0], "{tool:?}: ⌥-click sets the foreground");
            // A drag keeps sampling, even after ⌥ is let go; the background is untouched.
            tool_event(&mut app, ToolEvent::Move { x: 30.0, y: 10.0, pressure: 1.0 }, egui::Modifiers::NONE);
            assert_eq!(app.session.tools.foreground, [0.0, 1.0, 0.0, 1.0], "{tool:?}");
            tool_event(&mut app, ToolEvent::Up { x: 30.0, y: 10.0 }, egui::Modifiers::NONE);
            assert_eq!(app.session.tools.background, [1.0, 1.0, 1.0, 1.0]);
            assert_eq!(app.session.active().unwrap().revision, rev, "{tool:?}: sampling must not edit the document");
            assert!(app.drag.is_none() && !app.alt_sampling);
        }
        // Agents get the same through `ui.pointer` (MCP `ui_pointer`).
        app.ui.tool = Tool::Brush;
        let ctx = egui::Context::default();
        let events = json!([{"kind": "down", "x": 10, "y": 10}, {"kind": "up", "x": 10, "y": 10}]);
        let (req, _rx) = crate::control::ControlRequest::new("ui.pointer", json!({"modifiers": {"alt": true}, "events": events}));
        let _ = crate::control::handle(&mut app, &ctx, &req);
        assert_eq!(app.session.tools.foreground, [1.0, 0.0, 0.0, 1.0]);
        app.run("tools.setColors", json!({"foreground": [0.0, 1.0, 0.0, 1.0]})).unwrap();
        // Without ⌥ the Brush paints again, and ⌥ pressed mid-stroke doesn't switch to sampling.
        app.run("layer.new.layer", json!({})).unwrap();
        let rev = app.session.active().unwrap().revision;
        tool_event(&mut app, ToolEvent::Down { x: 5.0, y: 5.0, pressure: 1.0 }, egui::Modifiers::NONE);
        tool_event(&mut app, ToolEvent::Move { x: 30.0, y: 5.0, pressure: 1.0 }, alt);
        tool_event(&mut app, ToolEvent::Up { x: 30.0, y: 5.0 }, alt);
        assert_eq!(app.session.tools.foreground, [0.0, 1.0, 0.0, 1.0]);
        assert!(app.session.active().unwrap().revision > rev, "the stroke was painted");
        // Control+Alt stays the brush-resize gesture, never a sample.
        assert!(!alt_samples(Tool::Brush, egui::Modifiers { alt: true, ctrl: true, ..Default::default() }));
        assert!(!alt_samples(Tool::Eraser, alt), "⌥ with the Eraser is not the Eyedropper");
    }

    /// #2078: the opaque mask-view texture must show the live stroke, not cover it with
    /// the committed mask until the mouse is released.
    #[test]
    fn mask_view_uploads_each_live_stroke_before_release() {
        for depth in [8, 16, 32] {
            for mode in ["gray", "overlay"] {
                for (width, factor) in [(120u32, 1u32), (513, 3)] {
                    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
                    app.run("file.new", json!({"width": width, "height": 60, "depth": depth})).unwrap();
                    app.run("layer.layerMask.revealAll", json!({})).unwrap();
                    app.run("view.layerMask", json!({"mode": mode})).unwrap();
                    app.run("tools.setColors", json!({"foreground": [0.0, 0.0, 0.0, 1.0]})).unwrap();
                    app.run("tools.setBrush", json!({"brush": {"size": 12, "hardness": 1.0, "smoothing": {"amount": 0.0}}})).unwrap();
                    app.ui.tool = Tool::Brush;
                    let ctx = egui::Context::default();
                    ctx.input_mut(|i| i.max_texture_side = 256);
                    let tex = crate::channel_view::ensure(&mut app, &ctx, 0).unwrap();
                    ctx.tex_manager().write().take_delta().clear();
                    let revision = app.session.documents()[0].revision;
                    let steps = app.session.documents()[0].history.past_len();
                    let layer = app.session.documents()[0].active_layer.unwrap();
                    let original = app.session.documents()[0].doc.layer(layer).unwrap().clone();
                    tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 30.0, pressure: 1.0 }, egui::Modifiers::NONE);
                    for x in [30.0, 50.0, 70.0] {
                        tool_event(&mut app, ToolEvent::Move { x, y: 30.0, pressure: 1.0 }, egui::Modifiers::NONE);
                        let shown = display_doc(&mut app, 0).0;
                        assert!(shown.layer(layer).unwrap().mask.as_ref().unwrap().value(x as i32, 30) < 0.1);
                        assert_eq!(crate::channel_view::ensure(&mut app, &ctx, 0), Some(tex));
                        let mut delta = ctx.tex_manager().write().take_delta();
                        let uploads = delta.set.get(&tex).unwrap_or_else(|| panic!("{depth}-bit {mode}: live mask texture must upload before release at {x}"));
                        assert_eq!(uploads.len(), 1);
                        let upload = &uploads[0];
                        let pos = upload.pos.expect("only the damaged region is uploaded");
                        let egui::ImageData::Color(image) = &upload.image;
                        assert!(image.pixels.len() < (width.div_ceil(factor) * 60u32.div_ceil(factor)) as usize, "the whole mask is not uploaded");
                        let f = factor as i32;
                        let r = DRect::new(pos[0] as i32 * f, pos[1] as i32 * f, (pos[0] + image.size[0]) as i32 * f, (pos[1] + image.size[1]) as i32 * f)
                            .intersect(&shown.bounds());
                        let expected = crate::channel_view::render(
                            &shown,
                            &app.session.documents()[0].channel_view,
                            r,
                            factor,
                            false,
                            app.session.documents()[0].active_layer,
                        )
                        .unwrap();
                        assert_eq!(image.pixels, expected, "uploaded mask values match the live stroke");
                        delta.clear();
                        assert_eq!(app.session.documents()[0].revision, revision);
                        assert_eq!(app.session.documents()[0].history.past_len(), steps);
                        assert_eq!(app.session.documents()[0].doc.layer(layer).unwrap(), &original);
                        crate::channel_view::ensure(&mut app, &ctx, 0);
                        assert!(ctx.tex_manager().write().take_delta().set.is_empty(), "idle frame does not upload");
                    }
                    tool_event(&mut app, ToolEvent::Up { x: 70.0, y: 30.0 }, egui::Modifiers::NONE);
                    let st = &app.session.documents()[0];
                    assert_eq!(st.history.past_len(), steps + 1);
                    assert!(st.doc.layer(layer).unwrap().mask.as_ref().unwrap().value(50, 30) < 0.1);
                    assert_eq!(st.doc.layer(layer).unwrap().content, original.content);
                    crate::channel_view::ensure(&mut app, &ctx, 0);
                    let mut delta = ctx.tex_manager().write().take_delta();
                    // The commit takes the live stroke's pixels, which the texture already shows: at
                    // most a partial upload, never the whole mask again.
                    assert!(delta.set.get(&tex).is_none_or(|u| u.iter().all(|u| u.pos.is_some())), "commit keeps its partial upload");
                    delta.clear();
                    app.session.undo();
                    assert_eq!(app.session.documents()[0].doc.layer(layer).unwrap(), &original);
                    crate::channel_view::ensure(&mut app, &ctx, 0);
                    let mut delta = ctx.tex_manager().write().take_delta();
                    let egui::ImageData::Color(image) = &delta.set.get(&tex).unwrap()[0].image;
                    let st = &app.session.documents()[0];
                    assert_eq!(
                        image.pixels,
                        crate::channel_view::render(&st.doc, &st.channel_view, st.doc.bounds(), factor, false, st.active_layer).unwrap(),
                        "undo restores the shown mask"
                    );
                    delta.clear();
                }
            }
        }
    }

    #[test]
    fn channel_views_show_live_targets_and_discarded_strokes() {
        for target in ["quickMask", "alpha", "red"] {
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
            app.run("file.new", json!({"width": 120, "height": 60})).unwrap();
            match target {
                "quickMask" => {
                    app.run("select.rect", json!({"x": 0, "y": 0, "width": 120, "height": 60})).unwrap();
                    app.run("select.editInQuickMaskMode", json!({})).unwrap();
                }
                "alpha" => {
                    app.run("channel.new", json!({"fill": "white"})).unwrap();
                }
                _ => {
                    app.run("channel.target", json!({"channel": "red"})).unwrap();
                }
            }
            app.run("tools.setColors", json!({"foreground": [0.0, 0.0, 0.0, 1.0]})).unwrap();
            app.run("tools.setBrush", json!({"brush": {"size": 12, "hardness": 1.0, "smoothing": {"amount": 0.0}}})).unwrap();
            app.ui.tool = Tool::Brush;
            let ctx = egui::Context::default();
            let tex = crate::channel_view::ensure(&mut app, &ctx, 0).unwrap();
            ctx.tex_manager().write().take_delta().clear();
            let original = app.session.documents()[0].doc.clone();
            let steps = app.session.documents()[0].history.past_len();
            tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 30.0, pressure: 1.0 }, egui::Modifiers::NONE);
            tool_event(&mut app, ToolEvent::Move { x: 50.0, y: 30.0, pressure: 1.0 }, egui::Modifiers::NONE);
            let shown = display_doc(&mut app, 0).0;
            crate::channel_view::ensure(&mut app, &ctx, 0);
            let mut delta = ctx.tex_manager().write().take_delta();
            let upload = &delta.set.get(&tex).expect("live channel upload")[0];
            let pos = upload.pos.expect("partial channel upload");
            let egui::ImageData::Color(image) = &upload.image;
            let r = DRect::new(pos[0] as i32, pos[1] as i32, (pos[0] + image.size[0]) as i32, (pos[1] + image.size[1]) as i32);
            assert_eq!(
                image.pixels,
                crate::channel_view::render(&shown, &app.session.documents()[0].channel_view, r, 1, false, app.session.documents()[0].active_layer).unwrap(),
                "{target}"
            );
            assert_ne!(
                image.pixels,
                crate::channel_view::render(&original, &app.session.documents()[0].channel_view, r, 1, false, app.session.documents()[0].active_layer).unwrap(),
                "target was painted"
            );
            delta.clear();
            assert_eq!(app.session.documents()[0].history.past_len(), steps);
            assert!(std::sync::Arc::ptr_eq(&original, &app.session.documents()[0].doc));
            // Dropping a preview without committing must restore its texture, even though the
            // document revision did not change. There is no committed damage rectangle to use.
            app.drag = None;
            app.live_stroke = None;
            crate::channel_view::ensure(&mut app, &ctx, 0);
            let mut delta = ctx.tex_manager().write().take_delta();
            let upload = &delta.set.get(&tex).unwrap()[0];
            assert!(upload.pos.is_none(), "discarding a preview refreshes the view");
            let egui::ImageData::Color(image) = &upload.image;
            assert_eq!(
                image.pixels,
                crate::channel_view::render(
                    &original,
                    &app.session.documents()[0].channel_view,
                    original.bounds(),
                    1,
                    false,
                    app.session.documents()[0].active_layer
                )
                .unwrap()
            );
            delta.clear();
        }
    }

    #[test]
    fn brush_drag_shows_the_real_stroke_and_commits_it() {
        // The drag used to draw a hard, flat stand-in and only showed the soft brush on release.
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 120, "height": 60, "background": "transparent"})).unwrap();
        app.run("tools.setBrush", json!({"brush": {"size": 20, "hardness": 0.0}})).unwrap();
        app.ui.tool = Tool::Brush;
        let m = egui::Modifiers::NONE;
        let alpha = |d: &Document, x, y| d.layers[0].surface().unwrap().rgba(x, y)[3];
        let rev = app.session.documents()[0].revision;
        tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 30.0, pressure: 1.0 }, m);
        for x in [40.0, 70.0, 100.0] {
            tool_event(&mut app, ToolEvent::Move { x, y: 30.0, pressure: 1.0 }, m);
        }
        let (shown, key) = display_doc(&mut app, 0);
        assert_ne!(key, 0);
        // A partial value two thirds of the way out: the soft tip, not a hard flat stand-in.
        assert!(alpha(&shown, 40, 30) > 0.5 && (0.01..0.95).contains(&alpha(&shown, 40, 36)), "soft stroke while drawing");
        assert_eq!(alpha(&app.session.documents()[0].doc, 40, 30), 0.0, "not committed yet");
        // The canvas redraws only what the stroke touched.
        let dk = canvas_display(&app, &app.session.documents()[0].doc, None).1;
        let d = damage_since(&app, 0, (rev, dk), (rev, key), dk, None).unwrap();
        assert!(d.contains(40, 30) && !d.contains(40, 2) && d.height() < 60, "{d:?}");
        tool_event(&mut app, ToolEvent::Up { x: 100.0, y: 30.0 }, m);
        let doc = app.session.documents()[0].doc.clone();
        assert!(app.live_stroke.is_none() && display_doc(&mut app, 0).1 == 0);
        assert_eq!((alpha(&doc, 40, 30), alpha(&doc, 40, 38)), (alpha(&shown, 40, 30), alpha(&shown, 40, 38)), "commit matches the preview");
    }

    #[test]
    fn live_stroke_uses_pen_tilt_and_updates_a_reduced_texture_partially() {
        // Pen tilt drives the size; the preview must use the tilt the commit gets. The document
        // is wider than MAX_TEXTURE, so the CPU texture is reduced (factor 2) and each step
        // must update only the reduced pixels it touched.
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        let ctx = egui::Context::default();
        app.run("file.new", json!({"width": 4097, "height": 90, "background": "transparent"})).unwrap();
        let tilt = json!({"size": 30, "hardness": 1.0, "spacing": 0.05, "shapeDynamics": {"enabled": true, "size": {"control": "penTilt"}}});
        app.run("tools.setBrush", json!({ "brush": tilt })).unwrap();
        app.ui.tool = Tool::Brush;
        assert_eq!(ensure_texture(&mut app, &ctx, 0, None).map(|t| t.1), Some(0.5));
        let m = egui::Modifiers::NONE;
        app.stylus.feed.set(Some(crate::stylus::PenSample { pressure: 1.0, tilt_x: 60.0, tilt_y: 0.0, rotation: 0.0, eraser: false }));
        tool_event(&mut app, ToolEvent::Down { x: 100.0, y: 45.0, pressure: 1.0 }, m);
        for x in [300.0, 600.0, 900.0] {
            tool_event(&mut app, ToolEvent::Move { x, y: 45.0, pressure: 1.0 }, m);
            ensure_texture(&mut app, &ctx, 0, None);
            assert_eq!(app.perf.last_refresh, "rect");
            assert!(app.perf.last_refresh_px < 4097 * 90 / 4, "{}", app.perf.last_refresh_px);
        }
        let shown = display_doc(&mut app, 0).0;
        tool_event(&mut app, ToolEvent::Up { x: 900.0, y: 45.0 }, m);
        ensure_texture(&mut app, &ctx, 0, None);
        assert_eq!(app.perf.last_refresh, "rect", "the commit refreshes only the stroke");
        let doc = app.session.documents()[0].doc.clone();
        let (a, b) = (doc.layers[0].surface().unwrap(), shown.layers[0].surface().unwrap());
        // 60° tilt shrinks the dab well below 30 px: the preview shows that size, as committed.
        assert!(a.rgba(500, 45)[3] > 0.5 && a.rgba(500, 45 + 12)[3] == 0.0);
        // (The smoothed tail catches up to the end point only when the stroke finishes.)
        assert!((0..90).all(|y| (0..700).all(|x| a.rgba(x, y) == b.rgba(x, y))), "commit matches the preview");
    }

    /// Brush drag along y = 40 with one canvas frame per pointer move; returns the document the
    /// canvas showed at the last move, the committed one, and whether the release refreshed only
    /// the stroke's rectangle.
    fn drag_frames(smoothing: f32, xs: &[f64]) -> (std::sync::Arc<Document>, std::sync::Arc<Document>, bool, PhotocraftApp) {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        let ctx = egui::Context::default();
        app.run("file.new", json!({"width": 200, "height": 80, "background": "transparent"})).unwrap();
        app.run("tools.setBrush", json!({"brush": {"size": 16, "hardness": 0.5}})).unwrap();
        // What the options bar's Smoothing field writes.
        app.session.tools.brush.smoothing.amount = smoothing;
        app.ui.tool = Tool::Brush;
        ensure_texture(&mut app, &ctx, 0, None);
        let m = egui::Modifiers::NONE;
        tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 40.0, pressure: 1.0 }, m);
        for &x in xs {
            tool_event(&mut app, ToolEvent::Move { x, y: 40.0 + (x / 7.0).sin() * 8.0, pressure: 1.0 }, m);
            ensure_texture(&mut app, &ctx, 0, None);
        }
        let live = display_doc(&mut app, 0).0;
        let last = *xs.last().unwrap();
        tool_event(&mut app, ToolEvent::Up { x: last, y: 40.0 + (last / 7.0).sin() * 8.0 }, m);
        let (next, key) = display_doc(&mut app, 0);
        assert_eq!(key, 0, "the frame after release shows the committed document");
        ensure_texture(&mut app, &ctx, 0, None);
        let partial = app.perf.last_refresh == "rect";
        (live, next, partial, app)
    }

    fn same_pixels(a: &Document, b: &Document) -> bool {
        let (a, b) = (a.layers[0].surface().unwrap(), b.layers[0].surface().unwrap());
        (0..80).all(|y| (0..200).all(|x| a.rgba(x, y) == b.rgba(x, y)))
    }

    #[test]
    fn release_shows_nothing_new_without_smoothing() {
        // #73: at 0 % the stroke follows the pointer; the last frame drawn while dragging is
        // exactly the committed stroke, so release changes nothing on screen.
        let (live, done, partial, app) = drag_frames(0.0, &[40.0, 70.0, 100.0, 130.0, 160.0]);
        assert!(same_pixels(&live, &done), "preview at the last move = committed stroke");
        assert!(partial, "the handover refreshes only the stroke");
        let p = app.session.journal.iter().rev().find(|(id, _)| id == "paint.stroke").map(|(_, p)| p.clone()).unwrap();
        assert!(p.get("smoothing").is_none(), "the commit uses the session brush's smoothing, not a hard-coded one");
    }

    #[test]
    fn navigator_keeps_its_image_during_a_live_brush_stroke() {
        // #1631: every step of a live stroke is a new preview key, so a navigator keyed by it
        // re-composited the whole document per frame while painting. It waits for the release.
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 200, "height": 80, "background": "transparent"})).unwrap();
        app.ui.tool = Tool::Brush;
        let m = egui::Modifiers::NONE;
        assert!(!navigator_waits_for_drag(&app, 0));
        tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 40.0, pressure: 1.0 }, m);
        let mut keys = Vec::new();
        for x in [40.0, 70.0, 100.0] {
            tool_event(&mut app, ToolEvent::Move { x, y: 40.0, pressure: 1.0 }, m);
            keys.push(display_doc(&mut app, 0).1);
            assert!(navigator_waits_for_drag(&app, 0), "painting at x = {x}");
        }
        assert!(keys.windows(2).all(|k| k[0] != k[1]), "each step is a new preview: {keys:?}");
        tool_event(&mut app, ToolEvent::Up { x: 100.0, y: 40.0 }, m);
        assert!(!navigator_waits_for_drag(&app, 0), "the navigator catches up on release");
    }

    #[test]
    fn smoothed_stroke_shows_its_catch_up_tail_while_drawing() {
        // #73: with smoothing the brush lags behind the pointer and catches up at the end; the
        // preview draws that tail live, so no frame after release is missing the end.
        let xs = [30.0, 50.0, 70.0, 90.0, 110.0, 130.0, 150.0, 170.0];
        let (live, done, partial, _) = drag_frames(0.5, &xs);
        assert!(same_pixels(&live, &done), "preview at the last move = committed stroke, tail included");
        assert!(partial);
        let end = |d: &Document| (0..200).rev().find(|&x| (0..80).any(|y| d.layers[0].surface().unwrap().rgba(x, y)[3] > 0.0)).unwrap();
        assert!(end(&live) >= 170, "the end reaches the pointer while drawing ({})", end(&live));
        // The options-bar value reaches the stroke: 50 % smooths the wiggle, 0 % doesn't.
        let (_, rough, _, _) = drag_frames(0.0, &xs);
        assert!(!same_pixels(&rough, &done), "smoothing changes the stroke");
        // And the commit is what `paint.stroke` gives with that smoothing.
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 200, "height": 80, "background": "transparent"})).unwrap();
        s.execute("tools.setBrush", json!({"brush": {"size": 16, "hardness": 0.5, "smoothing": {"amount": 0.5}}})).unwrap();
        let mut pts = vec![json!([10.0, 40.0, 1.0])];
        pts.extend(xs.iter().map(|&x| json!([x, 40.0 + (x / 7.0).sin() * 8.0, 1.0])));
        // (Round tips without dynamics draw the same for any seed.)
        s.execute("paint.stroke", json!({"points": pts, "seed": 0})).unwrap();
        assert!(same_pixels(&s.active().unwrap().doc, &done));
    }

    #[test]
    fn smoothing_preview_redraws_the_tail_each_step() {
        // The tail drawn at one step must not linger once the brush moves on: a sharp turn would
        // leave a stale tail behind if it weren't restored.
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 200, "height": 120, "background": "transparent"})).unwrap();
        app.run("tools.setBrush", json!({"brush": {"size": 10, "hardness": 1.0, "smoothing": {"amount": 0.6}}})).unwrap();
        app.ui.tool = Tool::Brush;
        let m = egui::Modifiers::NONE;
        tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 20.0, pressure: 1.0 }, m);
        for (x, y) in [(60.0, 20.0), (110.0, 20.0), (110.0, 70.0), (110.0, 110.0), (60.0, 110.0)] {
            tool_event(&mut app, ToolEvent::Move { x, y, pressure: 1.0 }, m);
        }
        let live = display_doc(&mut app, 0).0;
        tool_event(&mut app, ToolEvent::Up { x: 60.0, y: 110.0 }, m);
        let done = app.session.documents()[0].doc.clone();
        let (a, b) = (live.layers[0].surface().unwrap(), done.layers[0].surface().unwrap());
        assert!((0..120).all(|y| (0..200).all(|x| a.rgba(x, y) == b.rgba(x, y))), "no stale tails in the preview");
    }

    /// Shapes the drag preview paints for `app` this frame.
    fn drag_preview_shapes(app: &mut PhotocraftApp, ctx: &egui::Context, zoom: f32) -> Vec<egui::epaint::ClippedShape> {
        let out = ctx.run_ui(egui::RawInput::default(), |ui| {
            let ctx = ui.ctx();
            let rect = Rect::from_min_size(Pos2::ZERO, vec2(800.0, 600.0));
            let painter = ctx.layer_painter(egui::LayerId::background()).with_clip_rect(rect);
            let xf = ViewXform { rect, zoom, center: [60.0, 30.0], flip: false, rotation: 0.0, aspect: 1.0 };
            draw_drag_preview(app, &painter, &xf);
        });
        let egui::FullOutput { mut textures_delta, shapes, .. } = out;
        textures_delta.clear();
        shapes
    }

    #[test]
    fn a_huge_view_centre_draws_the_selection_outline_without_overflow() {
        // #980: `ui.set` takes any finite centre, and the outline's visible region was quantised
        // with unchecked i32 arithmetic: the frame panicked ("attempt to multiply/subtract with
        // overflow") and release builds wrapped to an inverted region.
        let ctx = egui::Context::default();
        // Zoomed far out the quantisation step, and so the padding, is largest (64 · 256 px).
        let cases = [([1e30, 1e30], false), ([-1e30, -1e30], false), ([3e9, -3e9], false), ([512.0, 512.0], true)];
        for ((centre, traced), zoom) in cases.into_iter().flat_map(|c| [(c, 1.0), (c, 1.0 / 64.0)]) {
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
            app.run("file.new", json!({"width": 1024, "height": 1024})).unwrap();
            app.run("select.rect", json!({"x": 0, "y": 0, "width": 512, "height": 512})).unwrap();
            app.sync_views();
            let (req, _rx) = crate::control::ControlRequest::new("ui.set", json!({ "center": centre, "zoom": zoom }));
            let _ = crate::control::handle(&mut app, &ctx, &req);
            let view = app.ui.views[0].clone();
            assert_eq!(view.zoom, zoom, "ui.set applied the zoom");
            let mut out = ctx.run_ui(egui::RawInput::default(), |ui| {
                canvas_view(&mut app, ui, 0, Rect::from_min_size(Pos2::ZERO, vec2(200.0, 200.0)), view.clone(), false);
            });
            out.textures_delta.clear();
            // Control: a centre on the selection's corner still traces the outline (the selection
            // spans several 64 px trace cells even zoomed out).
            let segs = app.outline_cache.as_ref().map_or(0, |(_, _, s)| s.len());
            assert_eq!(segs > 0, traced, "{centre:?} at zoom {zoom}");
        }
    }

    #[test]
    fn a_one_pixel_selection_traces_at_hundred_percent_on_a_scaled_display() {
        // #1943 review: the outline step is display resolution in device pixels, not points, so a
        // 1 px selection still traces at 100 % on a 2x display (a points-based step sampled every
        // second pixel and lost it: zero segments).
        let ctx = egui::Context::default();
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 200, "height": 200})).unwrap();
        app.run("select.rect", json!({"x": 50, "y": 50, "width": 1, "height": 100})).unwrap();
        app.run("view.actualPixels", json!({})).unwrap();
        app.sync_views();
        let view = app.ui.views[0].clone();
        let mut input = egui::RawInput::default();
        input.viewports.entry(egui::ViewportId::ROOT).or_default().native_pixels_per_point = Some(2.0);
        let mut out = ctx.run_ui(input, |ui| {
            canvas_view(&mut app, ui, 0, Rect::from_min_size(Pos2::ZERO, vec2(200.0, 200.0)), view.clone(), true);
        });
        out.textures_delta.clear();
        assert_eq!(app.ppp, 2.0, "the primary canvas records the display scale");
        let segs = app.outline_cache.as_ref().map_or(0, |(_, _, s)| s.len());
        assert!(segs > 0, "a 1 px selection traces at 100 % on a 2x display");
    }

    #[test]
    fn a_secondary_canvas_does_not_overwrite_the_main_canvas_scale() {
        // #1943 review: a document window can sit on a display with another scale; only the
        // primary canvas records `app.ppp`, so the main mapping doesn't change under it.
        let ctx = egui::Context::default();
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 400, "height": 300})).unwrap();
        app.run("view.actualPixels", json!({})).unwrap();
        app.sync_views();
        let view = app.ui.views[0].clone();
        let frame = |app: &mut PhotocraftApp, ppp: f32, primary: bool| {
            let mut input = egui::RawInput::default();
            input.viewports.entry(egui::ViewportId::ROOT).or_default().native_pixels_per_point = Some(ppp);
            let mut out = ctx.run_ui(input, |ui| {
                canvas_view(app, ui, 0, Rect::from_min_size(Pos2::ZERO, vec2(400.0, 300.0)), view.clone(), primary);
            });
            out.textures_delta.clear();
        };
        frame(&mut app, 2.0, true);
        assert_eq!(app.ppp, 2.0);
        frame(&mut app, 1.0, false);
        assert_eq!(app.ppp, 2.0, "a secondary window's scale doesn't replace the main canvas's");
        // The main canvas mapping still uses its own scale: 100 % is 1 doc px per physical px.
        let xf = ViewXform::active(&app).unwrap();
        let step = xf.to_screen(1.0, 0.0) - xf.to_screen(0.0, 0.0);
        assert!((step.x - 0.5).abs() < 1e-3, "{step:?}");
    }

    #[test]
    fn a_flicked_hand_drag_glides_on_only_with_flick_panning() {
        // #204: Preferences › Tools › Enable Flick Panning. A fast Hand drag let go while moving
        // keeps the view gliding for a while; off, the view stops where the drag ended.
        for flick in [true, false] {
            let ctx = egui::Context::default();
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
            app.run("file.new", json!({"width": 2000, "height": 1500})).unwrap();
            app.run("prefs.set", json!({"path": "tools.enableFlickPanning", "value": flick})).unwrap();
            app.ui.tool = Tool::Hand;
            app.sync_views();
            let mut view = app.ui.views[0].clone();
            view.zoom = 1.0;
            view.center = [1000.0, 750.0];
            view.fit_pending = false;
            view.fill_pending = false;
            let rect = Rect::from_min_size(Pos2::ZERO, vec2(400.0, 300.0));
            let mut frame = |app: &mut PhotocraftApp, events: Vec<egui::Event>| -> View {
                let input = egui::RawInput { screen_rect: Some(rect), events, ..Default::default() };
                let mut out = ctx.run_ui(input, |ui| {
                    view = canvas_view(app, ui, 0, rect, view.clone(), true);
                });
                out.textures_delta.clear();
                view.clone()
            };
            let start = Pos2::new(200.0, 150.0);
            let press =
                |pos: Pos2, pressed: bool| egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE };
            frame(&mut app, vec![egui::Event::PointerMoved(start)]);
            frame(&mut app, vec![press(start, true)]);
            let mut end = start;
            for i in 1..=8 {
                end = start + vec2(-20.0 * i as f32, 0.0);
                frame(&mut app, vec![egui::Event::PointerMoved(end)]);
            }
            let released = frame(&mut app, vec![press(end, false)]).center;
            let mut centers = vec![released];
            for _ in 0..5 {
                centers.push(frame(&mut app, vec![]).center);
            }
            if flick {
                // Dragged left: the image follows the pointer, the view keeps moving right.
                for w in centers.windows(2) {
                    assert!(w[1][0] > w[0][0], "glides on: {centers:?}");
                }
                // ...and comes to rest.
                let mut last = *centers.last().unwrap();
                for _ in 0..120 {
                    last = frame(&mut app, vec![]).center;
                }
                assert_eq!(frame(&mut app, vec![]).center, last, "the glide stops");
            } else {
                assert!(centers.iter().all(|c| *c == released), "no glide when off: {centers:?}");
            }
        }
    }

    #[test]
    fn a_secondary_canvas_hand_pans_while_a_transform_is_open() {
        // #2302 review: `app.ui.transform` is global, but only the primary canvas draws and drives
        // the box. A secondary window's Hand must still pan while the main window transforms.
        let ctx = egui::Context::default();
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 400, "height": 300})).unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        app.run("select.rect", json!({"x": 100, "y": 100, "width": 100, "height": 60})).unwrap();
        app.run("edit.fill", json!({"color": "#ff0000"})).unwrap();
        app.run("select.deselect", json!({})).unwrap();
        app.run("view.actualPixels", json!({})).unwrap();
        app.ui.tool = Tool::Hand;
        app.sync_views();
        crate::transform_tool::begin(&mut app, &ctx).expect("a transform starts");
        let mut view0 = app.ui.views[0].clone();
        view0.zoom = 1.0;
        view0.center = [200.0, 150.0];
        view0.fit_pending = false;
        view0.fill_pending = false;
        let center0 = view0.center;
        let rect = Rect::from_min_size(Pos2::ZERO, vec2(400.0, 300.0));
        let frame = |app: &mut PhotocraftApp, events: Vec<egui::Event>| -> View {
            let input = egui::RawInput { screen_rect: Some(rect), events, ..Default::default() };
            let mut seen = view0.clone();
            let mut out = ctx.run_ui(input, |ui| {
                seen = canvas_view(app, ui, 0, rect, view0.clone(), false);
            });
            out.textures_delta.clear();
            seen
        };
        let start = Pos2::new(200.0, 150.0);
        let press =
            |pos: Pos2, pressed: bool| egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE };
        frame(&mut app, vec![egui::Event::PointerMoved(start)]);
        frame(&mut app, vec![press(start, true)]);
        for dx in [10.0, 20.0, 40.0] {
            frame(&mut app, vec![egui::Event::PointerMoved(start + vec2(dx, 0.0))]);
        }
        let dragged = frame(&mut app, vec![egui::Event::PointerMoved(start + vec2(60.0, 0.0))]);
        frame(&mut app, vec![press(start + vec2(60.0, 0.0), false)]);
        assert_ne!(dragged.center, center0, "a secondary window's Hand pans while transforming");
    }

    #[test]
    fn brush_drags_paint_no_stand_in_shape_over_the_canvas() {
        // #189: v0.1 drew the stroke being dragged as a foreground-coloured egui polyline as wide
        // as the brush; zoomed in, egui tessellated it into hard black wedges fanning out from the
        // start. The canvas shows the real stroke (`LiveStroke`) and nothing is drawn over it.
        let ctx = egui::Context::default();
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 120, "height": 60})).unwrap();
        app.run("tools.setBrush", json!({"brush": {"size": 30, "hardness": 0.0}})).unwrap();
        // A smoothed stroke also draws its string line (see
        // `smoothed_brush_drag_draws_the_string_from_brush_to_pointer`): that is no stand-in shape.
        let overlay = |app: &mut PhotocraftApp, ctx: &egui::Context| {
            drag_preview_shapes(app, ctx, 12.0).iter().filter(|s| !matches!(s.shape, egui::Shape::LineSegment { .. } | egui::Shape::Circle(_))).count()
        };
        for tool in [Tool::Brush, Tool::Eraser] {
            app.ui.tool = tool;
            let idle = overlay(&mut app, &ctx);
            tool_event(&mut app, ToolEvent::Down { x: 20.0, y: 30.0, pressure: 1.0 }, egui::Modifiers::NONE);
            for i in 1..40 {
                let t = f64::from(i);
                tool_event(&mut app, ToolEvent::Move { x: 20.0 + t, y: 30.0 + (t / 3.0).sin() * 4.0, pressure: 1.0 }, egui::Modifiers::NONE);
            }
            assert!(app.live_stroke.is_some(), "{tool:?}: the canvas shows the live stroke");
            assert_eq!(overlay(&mut app, &ctx), idle, "{tool:?}: no overlay while dragging");
            tool_event(&mut app, ToolEvent::Up { x: 59.0, y: 30.0 }, egui::Modifiers::NONE);
        }
    }

    #[test]
    fn smoothed_brush_drag_draws_the_string_from_brush_to_pointer() {
        // #3479: with smoothing the brush lags behind the pointer; a line joins them while drawing.
        let ctx = egui::Context::default();
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 120, "height": 60})).unwrap();
        app.ui.tool = Tool::Brush;
        let lines = |app: &mut PhotocraftApp| drag_preview_shapes(app, &ctx, 4.0).iter().filter(|s| matches!(s.shape, egui::Shape::LineSegment { .. })).count();
        for (amount, string) in [(0.0, false), (0.8, true)] {
            app.run("tools.setBrush", json!({"brush": {"size": 10, "smoothing": {"amount": amount}}})).unwrap();
            tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 30.0, pressure: 1.0 }, egui::Modifiers::NONE);
            for x in [30.0, 60.0, 90.0] {
                tool_event(&mut app, ToolEvent::Move { x, y: 30.0, pressure: 1.0 }, egui::Modifiers::NONE);
            }
            assert_eq!(lines(&mut app) > 0, string, "smoothing {amount}");
            tool_event(&mut app, ToolEvent::Up { x: 90.0, y: 30.0 }, egui::Modifiers::NONE);
        }
    }

    #[test]
    fn retouch_drags_show_the_footprint_without_wedges() {
        // The retouching tools' trail was the same brush-wide polyline (#189): now a mask of the
        // footprint, with nothing outside the brush radius of the path, however small the steps.
        let ctx = egui::Context::default();
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 120, "height": 60})).unwrap();
        app.run("tools.setBrush", json!({"brush": {"size": 30}})).unwrap();
        // Remove: the dab tools draw their stroke live instead of a trail, and Spot Healing's
        // trail is its dabs' coverage.
        app.ui.tool = Tool::Remove;
        tool_event(&mut app, ToolEvent::Down { x: 20.0, y: 30.0, pressure: 1.0 }, egui::Modifiers::NONE);
        let mut pts = vec![[20.0, 30.0]];
        for i in 1..80 {
            let t = f64::from(i) * 0.5;
            let p = [20.0 + t, 30.0 + (t / 2.0).sin() * 3.0];
            tool_event(&mut app, ToolEvent::Move { x: p[0], y: p[1], pressure: 1.0 }, egui::Modifiers::NONE);
            pts.push(p);
            if i % 20 == 0 {
                // One mesh for the trail, however long the drag.
                let shapes = drag_preview_shapes(&mut app, &ctx, 12.0);
                assert!(!shapes.iter().any(|s| matches!(s.shape, egui::Shape::Path(_))), "no polyline");
            }
        }
        drag_preview_shapes(&mut app, &ctx, 12.0);
        let trail = app.trail.as_ref().unwrap();
        assert_eq!(trail.scale(), 1.0);
        for y in 0..60 {
            for x in 0..120 {
                let d = pts.iter().map(|p| (x as f64 + 0.5 - p[0]).hypot(y as f64 + 0.5 - p[1])).fold(f64::MAX, f64::min);
                if d < 14.0 {
                    assert_eq!(trail.coverage(x, y), 255, "({x}, {y}) inside the footprint");
                } else if d > 16.0 {
                    assert_eq!(trail.coverage(x, y), 0, "({x}, {y}) outside the footprint");
                }
            }
        }
        tool_event(&mut app, ToolEvent::Up { x: 59.5, y: 30.0 }, egui::Modifiers::NONE);
        drag_preview_shapes(&mut app, &ctx, 12.0);
        assert!(app.trail.is_none(), "the trail ends with the drag");
    }

    #[test]
    fn spot_healing_trail_is_the_coverage_the_heal_uses() {
        // #2364: a soft brush with pen-pressure size showed as a hard footprint at full size.
        let ctx = egui::Context::default();
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 120, "height": 60})).unwrap();
        let brush = json!({"size": 30, "hardness": 0.0, "shapeDynamics": {"enabled": true, "size": {"control": "penPressure"}}});
        app.run("tools.setBrush", json!({ "brush": brush })).unwrap();
        app.ui.tool = Tool::SpotHealing;
        tool_event(&mut app, ToolEvent::Down { x: 20.0, y: 30.0, pressure: 0.5 }, egui::Modifiers::NONE);
        for x in [40.0, 60.0, 80.0] {
            tool_event(&mut app, ToolEvent::Move { x, y: 30.0, pressure: 0.5 }, egui::Modifiers::NONE);
            drag_preview_shapes(&mut app, &ctx, 12.0);
        }
        // The stroke the heal gets on release, and its coverage.
        let points = app.drag.as_ref().unwrap().points.iter().map(|p| photocraft_paint::StrokePoint::new(p[0], p[1], p[2] as f32)).collect();
        let brush = photocraft_engine::retouch_cmds::stroke_brush(&app.session.tools.brush, &json!({}));
        let (b, cov) = photocraft_paint::retouch::stroke_coverage(&photocraft_paint::Stroke { brush, points });
        let trail = app.trail.as_ref().unwrap();
        let mut soft = 0;
        for y in 0..60 {
            for x in 0..120 {
                let want = if b.contains(x, y) { cov[(y - b.y0) as usize * b.width() as usize + (x - b.x0) as usize] } else { 0.0 };
                let got = trail.coverage(x as usize, y as usize);
                assert!((f32::from(got) - want * 255.0).abs() <= 1.0, "({x}, {y}): trail {got}, heal {want}");
                soft += usize::from(got > 0 && got < 200);
            }
        }
        assert!(soft > 50, "a soft brush shows a soft edge");
        // Half pressure halves the 30 px tip: nothing past its (soft) reach.
        let reach = photocraft_paint::tip_reach(7.5, app.session.tools.brush.hardness).ceil() as usize + 2;
        assert_eq!(trail.coverage(50, 30 + reach), 0, "half pressure halves the 30 px tip");
    }

    #[test]
    fn layer_style_dialog_previews_live_and_cancel_restores() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 16, "height": 16})).unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        let fx = |d: &Document| d.layers.iter().map(|l| l.effects.items.len()).sum::<usize>();
        let id = crate::layer_style::open(&mut app, Some("colorOverlay")).unwrap();
        let (shown, key) = display_doc(&mut app, 0);
        assert_eq!((fx(&shown), fx(&app.session.documents()[0].doc)), (1, 0), "previewed, not committed");
        // Adding a stroke through the dialog's instance list re-renders the canvas.
        if let Some(d) = app.ui.dialog_mut(id) {
            d.fields["effects"].as_array_mut().unwrap().push(json!({"id": "fx9", "kind": "stroke", "on": true, "params": {"size": 3}}));
        }
        let (shown, key2) = display_doc(&mut app, 0);
        assert_eq!(fx(&shown), 2);
        assert_ne!(key, key2, "an edit re-renders the canvas");
        app.ui.close_dialog(id);
        let (shown, key) = display_doc(&mut app, 0);
        assert_eq!((fx(&shown), key), (0, 0));
        assert!(app.style_preview.is_none());
    }

    /// Layer Style opened while Free Transform is open applies the box first, as Photoshop does,
    /// so the dialog previews on the transformed layer (#4821): a dropped file (placed in the box,
    /// untouched) and a box opened by hand and moved.
    #[test]
    fn layer_style_opened_during_free_transform_applies_the_box_and_previews() {
        for placed in [true, false] {
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
            app.run("file.new", json!({"width": 64, "height": 64})).unwrap();
            app.run("layer.new.layer", json!({})).unwrap();
            app.run("select.rect", json!({"x": 8, "y": 8, "width": 16, "height": 16})).unwrap();
            app.run("edit.fill", json!({"color": "#0000ff"})).unwrap();
            app.run("select.deselect", json!({})).unwrap();
            app.sync_views();
            let ctx = egui::Context::default();
            let dx = if placed {
                crate::transform_tool::begin_placed(&mut app, &ctx).unwrap();
                0
            } else {
                crate::transform_tool::begin(&mut app, &ctx).unwrap();
                app.ui.transform.as_mut().unwrap().quad = [[28.0, 8.0], [44.0, 8.0], [44.0, 24.0], [28.0, 24.0]];
                20
            };
            crate::layer_style::open(&mut app, Some("dropShadow")).unwrap();
            assert!(app.ui.transform.is_none() && app.transform_preview.is_none(), "placed {placed}: the box is applied");
            let (shown, _) = display_doc(&mut app, 0);
            let rect = photocraft_geom::Rect::new(0, 0, 64, 64);
            let with = photocraft_compose::render(&shown, rect);
            let without = photocraft_compose::render(&app.session.documents()[0].doc, rect);
            assert!(
                with.get(16 + dx, 16)[0] < 0.1 && with.get(16 + dx, 16)[2] > 0.9,
                "placed {placed}: the blue block where the box left it, not the white under it"
            );
            // The shadow darkens the white around the block, which the document doesn't have yet.
            let block = |x: i32, y: i32| (8 + dx..24 + dx).contains(&x) && (8..24).contains(&y);
            let shadow =
                (0..64).flat_map(|y| (0..64).map(move |x| (x, y))).filter(|&(x, y)| !block(x, y) && without.get(x, y)[0] - with.get(x, y)[0] > 0.05).count();
            assert!(shadow > 20, "placed {placed}: the Drop Shadow is previewed ({shadow} px)");
        }
    }

    #[test]
    fn fill_view_covers_the_canvas_and_centres_the_document() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 400, "height": 200})).unwrap();
        let doc = app.session.active().unwrap().doc.clone();
        let mut view = View { fill_pending: true, ..View::default() };

        fill_view(&mut view, &doc, vec2(600.0, 600.0), 1.0, 1.0);

        assert_eq!(view.zoom, 3.0, "the short edge fills the available height");
        assert_eq!(view.center, [200.0, 100.0]);
        assert!(!view.fill_pending);

        // A scaled display stores the physical factor: the same point zoom is 2× the device px.
        let mut view = View { fill_pending: true, ..View::default() };
        fill_view(&mut view, &doc, vec2(600.0, 600.0), 2.0, 1.0);
        assert_eq!(view.zoom, 6.0);

        // Fit never magnifies past 100% = one image pixel per physical pixel, at any display
        // scale (it used to cap at one per *point*, so a 150% display opened images at 150%).
        for ppp in [1.0f32, 1.5, 2.0] {
            let mut view = View { fit_pending: true, ..View::default() };
            fit_view(&mut view, &doc, vec2(600.0, 600.0), ppp, 1.0);
            assert_eq!(view.zoom, 1.0, "ppp {ppp}: a small image opens at 100%");
        }
        // A large image still shrinks to fit: 1000×500 into 440 points is 0.44 pt/px, so 0.66
        // physical at ppp 1.5.
        let big = {
            app.run("file.new", json!({"width": 1000, "height": 500})).unwrap();
            app.session.active().unwrap().doc.clone()
        };
        let mut view = View { fit_pending: true, ..View::default() };
        fit_view(&mut view, &big, vec2(480.0, 480.0), 1.5, 1.0);
        assert!((view.zoom - 0.66).abs() < 1e-5, "{}", view.zoom);
    }

    #[test]
    fn a_hundred_percent_is_one_physical_pixel_per_document_pixel() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 400, "height": 200})).unwrap();
        app.sync_views();
        for ppp in [1.0f32, 1.25, 2.0] {
            app.ppp = ppp;
            let xf = ViewXform::active(&app).unwrap();
            // 100% zoom: one document pixel is one physical pixel, i.e. 1/ppp egui points.
            let step = xf.to_screen(1.0, 0.0) - xf.to_screen(0.0, 0.0);
            // `to_screen` differences around the view centre cancel in f32; 1e-3 is still exact
            // enough to see a wrong ppp factor (which would be a 20–100% error).
            assert!((step.x - 1.0 / ppp).abs() < 1e-3, "ppp {ppp}: {step:?} want {:?}", 1.0 / ppp);
            assert!((app.point_zoom() - 1.0 / ppp).abs() < 1e-5, "ppp {ppp}: {}", app.point_zoom());
            // Screen ↔ document stays an exact round trip.
            let back = xf.to_doc(xf.to_screen(123.0, 45.0));
            assert!((back[0] - 123.0).abs() < 1e-3 && (back[1] - 45.0).abs() < 1e-3, "ppp {ppp}: {back:?}");
        }
    }

    #[test]
    fn wheel_anchor_is_the_pointer_over_the_image_and_the_nearest_image_point_off_it() {
        let rect = Rect::from_min_size(Pos2::ZERO, vec2(800.0, 600.0));
        for (rotation, flip) in [(0.0, false), (30.0, false), (90.0, true)] {
            let xf = ViewXform { rect, zoom: 0.5, center: [400.0, 300.0], flip, rotation, aspect: 1.0 };
            let inside = xf.to_screen(100.0, 200.0);
            assert_eq!(wheel_anchor(&xf, inside, [800, 600]), inside);
            for (p, want) in [((-300.0, -200.0), [0.0, 0.0]), ((1200.0, 300.0), [800.0, 300.0]), ((400.0, 900.0), [400.0, 600.0])] {
                let a = xf.to_doc(wheel_anchor(&xf, xf.to_screen(p.0, p.1), [800, 600]));
                assert!((a[0] - want[0]).abs() < 0.01 && (a[1] - want[1]).abs() < 0.01, "rot {rotation}: {p:?} -> {a:?}");
            }
        }
        let p = pos2(5.0, 5.0);
        let xf = ViewXform { rect, zoom: 1.0, center: [400.0, 300.0], flip: false, rotation: 0.0, aspect: 1.0 };
        assert_eq!(wheel_anchor(&xf, p, [0, 0]), p, "no document: the pointer");
    }

    #[test]
    fn zoom_about_centres_the_clicked_point_with_the_preference() {
        let rect = Rect::from_min_size(Pos2::ZERO, vec2(800.0, 600.0));
        let p = pos2(600.0, 200.0);
        for ppp in [1.0f32, 2.0] {
            // `View::zoom` is physical; the transform works in points.
            let xf = ViewXform { rect, zoom: 1.0 / ppp, center: [400.0, 300.0], flip: false, rotation: 0.0, aspect: 1.0 };
            let doc = xf.to_doc(p);
            // Off: the document point under the pointer stays under it.
            let mut view = View { zoom: 1.0, center: [400.0, 300.0], ..Default::default() };
            zoom_about(&mut view, &xf, p, 2.0, false, ppp);
            let after = ViewXform { rect, zoom: 2.0 / ppp, center: view.center, flip: false, rotation: 0.0, aspect: 1.0 };
            let s = after.to_screen(doc[0] as f32, doc[1] as f32);
            assert!((s - p).length() < 0.5, "ppp {ppp}: {s:?} vs {p:?}");
            // On: the clicked point is the view centre.
            let mut view = View { zoom: 1.0, center: [400.0, 300.0], ..Default::default() };
            zoom_about(&mut view, &xf, p, 2.0, true, ppp);
            assert!((view.center[0] - doc[0] as f32).abs() < 0.5, "ppp {ppp}: {:?}", view.center);
            assert!((view.center[1] - doc[1] as f32).abs() < 0.5, "ppp {ppp}: {:?}", view.center);
        }
    }

    #[test]
    fn damage_grows_by_nested_effect_reach() {
        use photocraft_doc::{Effect, Layer};
        let fmt = photocraft_color::PixelFormat::RGBA8;
        assert_eq!(effect_reach(&[Layer::raster("plain", fmt)]), 0);
        let mut inner = Layer::raster("inner", fmt);
        inner.effects.items.push(Effect::default_drop_shadow());
        let m = photocraft_compose::effects::margin(&inner);
        let mut group = Layer::group("g", vec![inner.clone()]);
        group.effects.items.push(Effect::default_drop_shadow());
        assert_eq!(effect_reach(std::slice::from_ref(&inner)), m);
        // A child's edit moves the group's shape, whose effects reach further.
        assert_eq!(effect_reach(&[group.clone()]), 2 * m);
        group.visible = false;
        assert_eq!(effect_reach(&[group]), 0);
    }
}

#[cfg(test)]
mod transform_controls_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn move_transform_controls_start_free_transform_for_vector_shapes() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.execute("file.new", json!({"width": 200, "height": 200})).unwrap();
        app.session.execute("shape.create", json!({"kind": "rect", "rect": [20, 30, 80, 40], "fill": "#ff0000"})).unwrap();
        app.sync_views();
        app.ui.tool = Tool::Move;
        app.ui.tool_options.move_show_transform = true;

        let xf =
            ViewXform { rect: Rect::from_min_size(Pos2::ZERO, vec2(200.0, 200.0)), zoom: 1.0, center: [100.0, 100.0], flip: false, rotation: 0.0, aspect: 1.0 };
        let r = transform_controls_rect(&app, &xf).expect("shape layers have transform bounds");
        assert!(transform_controls_hit(r, r.right_bottom()));

        let ctx = egui::Context::default();
        assert!(begin_transform_controls_at(&mut app, &ctx, &xf, r.right_bottom()));
        let t = app.ui.transform.as_ref().expect("control press starts Free Transform");
        assert_eq!(t.rect, [20.0, 30.0, 100.0, 70.0]);
    }

    #[test]
    fn move_transform_controls_follow_floating_selected_pixels() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 400, "height": 300})).unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        app.run("select.rect", json!({"x": 100, "y": 100, "width": 100, "height": 60})).unwrap();
        app.run("edit.fill", json!({"color": "#ff0000"})).unwrap();
        app.ui.tool = Tool::Move;
        app.ui.tool_options.move_show_transform = true;
        assert!(crate::move_ui::float_selected(&mut app, false, 100.0, 50.0));
        let xf =
            ViewXform { rect: Rect::from_min_size(Pos2::ZERO, vec2(400.0, 300.0)), zoom: 1.0, center: [200.0, 150.0], flip: false, rotation: 0.0, aspect: 1.0 };
        let r = transform_controls_rect(&app, &xf).unwrap();
        assert_eq!(r, Rect::from_min_max(pos2(200.0, 150.0), pos2(300.0, 210.0)));
        let ctx = egui::Context::default();
        assert!(begin_transform_controls_at(&mut app, &ctx, &xf, r.right_bottom()));
        assert_eq!(app.ui.transform.as_ref().unwrap().rect, [200.0, 150.0, 300.0, 210.0]);
    }

    /// #2585: Free Transform on a position-locked layer is refused before the box opens, with
    /// Photoshop's alert, from ⌘T and from the Move tool's transform controls alike. It used to
    /// open, preview every drag and fail only at commit.
    #[test]
    fn free_transform_on_a_locked_layer_is_refused_up_front() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.execute("file.new", json!({"width": 200, "height": 200})).unwrap();
        app.session.execute("shape.create", json!({"kind": "rect", "rect": [20, 30, 80, 40], "fill": "#ff0000"})).unwrap();
        app.session
            .edit("lock", |doc, active| {
                doc.layer_mut(active.unwrap()).unwrap().locks.position = true;
                Ok(())
            })
            .unwrap();
        app.sync_views();
        let errors = |app: &PhotocraftApp| -> Vec<String> {
            app.ui
                .dialogs
                .iter()
                .filter(|d| d.kind == crate::state::DialogKind::Error)
                .filter_map(|d| d.fields.get("message")?.as_str().map(String::from))
                .collect()
        };
        let ctx = egui::Context::default();
        // ⌘T.
        let r = crate::menus::invoke(&mut app, &ctx, "edit.freeTransform", json!({}));
        assert_eq!(r, Err(crate::transform_tool::LOCKED.to_string()));
        assert!(app.ui.transform.is_none() && app.transform_preview.is_none(), "no box opened");
        assert_eq!(errors(&app), [crate::transform_tool::LOCKED]);
        app.ui.dialogs.clear();
        // The Move tool's transform controls.
        app.ui.tool = Tool::Move;
        app.ui.tool_options.move_show_transform = true;
        let xf =
            ViewXform { rect: Rect::from_min_size(Pos2::ZERO, vec2(200.0, 200.0)), zoom: 1.0, center: [100.0, 100.0], flip: false, rotation: 0.0, aspect: 1.0 };
        let r = transform_controls_rect(&app, &xf).unwrap();
        assert!(!begin_transform_controls_at(&mut app, &ctx, &xf, r.right_bottom()));
        assert!(app.ui.transform.is_none());
        assert_eq!(errors(&app), [crate::transform_tool::LOCKED]);
    }

    #[test]
    fn move_transform_controls_show_resize_and_rotate_cursors_before_first_interaction() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.execute("file.new", json!({"width": 200, "height": 200})).unwrap();
        app.session.execute("shape.create", json!({"kind": "rect", "rect": [20, 30, 80, 40], "fill": "#ff0000"})).unwrap();
        app.sync_views();
        app.ui.tool = Tool::Move;
        app.ui.tool_options.move_show_transform = true;
        let xf =
            ViewXform { rect: Rect::from_min_size(Pos2::ZERO, vec2(200.0, 200.0)), zoom: 1.0, center: [100.0, 100.0], flip: false, rotation: 0.0, aspect: 1.0 };

        let r = transform_controls_rect(&app, &xf).expect("visible shape controls");
        assert!(app.ui.transform.is_none(), "no transform session is needed for hover");
        for (p, expected) in [
            (r.left_top(), egui::CursorIcon::ResizeNwSe),
            (r.right_bottom(), egui::CursorIcon::ResizeNwSe),
            (r.right_top(), egui::CursorIcon::ResizeNeSw),
            (r.left_bottom(), egui::CursorIcon::ResizeNeSw),
            (r.center_top(), egui::CursorIcon::ResizeVertical),
            (r.center_bottom(), egui::CursorIcon::ResizeVertical),
            (r.right_center(), egui::CursorIcon::ResizeHorizontal),
            (r.left_center(), egui::CursorIcon::ResizeHorizontal),
            // Between handles: the rotation band (a point above the top-centre handle is that handle),
            // where the curved rotate arrow replaces the pointer (#4658).
            (pos2(r.left() + r.width() * 0.25, r.top() - 12.0), egui::CursorIcon::None),
        ] {
            assert_eq!(transform_controls_cursor(r, p), Some(expected), "hover at {p:?}");
            assert!(transform_controls_hit(r, p), "same region must start a transform");
        }
        assert_eq!(transform_controls_cursor(r, r.center()), None, "interior stays Move");
        assert_eq!(transform_controls_cursor(r, pos2(r.center().x, r.top() - 30.0)), None);
        // The rotate arrow bends round the nearest corner, pointing into the box; handles and the
        // interior have none.
        let into = |p: Pos2| transform_controls_turn(r, p).map(|v| (v.x.signum(), v.y.signum()));
        assert_eq!(into(pos2(r.left() + r.width() * 0.25, r.top() - 12.0)), Some((1.0, 1.0)), "top left");
        assert_eq!(into(pos2(r.right() + 12.0, r.bottom() - r.height() * 0.25)), Some((-1.0, -1.0)), "bottom right");
        assert_eq!(into(pos2(r.left() - 12.0, r.bottom() + 12.0)), Some((1.0, -1.0)), "beyond the bottom-left corner");
        assert_eq!(into(r.left_top()), None);
        assert_eq!(into(r.center()), None);

        // Changing the active shape changes the hover bounds without requiring a drag first.
        app.session.execute("shape.create", json!({"kind": "rect", "rect": [120, 100, 40, 50], "fill": "#00ff00"})).unwrap();
        let next = transform_controls_rect(&app, &xf).expect("new active shape controls");
        assert_ne!(r, next);
        assert_eq!(transform_controls_cursor(next, next.right_bottom()), Some(egui::CursorIcon::ResizeNwSe));
        assert_eq!(transform_controls_cursor(next, next.center()), None);
    }

    #[test]
    fn transform_control_hit_keeps_move_interior_free_and_has_rotation_band() {
        let r = Rect::from_min_max(pos2(20.0, 30.0), pos2(100.0, 70.0));
        assert!(transform_controls_hit(r, r.left_top()));
        assert!(transform_controls_hit(r, pos2(60.0, 18.0)));
        assert!(!transform_controls_hit(r, r.center()));
        assert!(!transform_controls_hit(r, pos2(60.0, 4.0)));
    }
}

#[cfg(test)]
mod layer_edges_tests;

#[cfg(test)]
mod rotation_preview_isolation_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn channel_overlay_shows_filter_preview_and_restores_on_cancel() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.background_jobs = false;
        app.run("file.new", json!({"width":16,"height":16})).unwrap();
        app.run("channel.new", json!({"name":"Alpha","fill":"black"})).unwrap();
        app.session
            .edit("Fixture", |doc, _| {
                for y in 0..16 {
                    for x in 0..8 {
                        doc.channels[0].surface.write_pixel(x, y, &[1.0]);
                    }
                }
                Ok(())
            })
            .unwrap();
        let before = app.session.active().unwrap().doc.clone();
        let ctx = egui::Context::default();
        let texture = crate::channel_view::ensure(&mut app, &ctx, 0).unwrap();
        ctx.tex_manager().write().take_delta().clear();
        let dialog = crate::filter_dialog::open(&mut app, "filter.blur.gaussianBlur").unwrap();
        app.ui.dialogs.last_mut().unwrap().fields.insert("radius".into(), json!(2.0));
        ensure_filter_preview(&mut app, 0, &ctx).unwrap();
        crate::channel_view::ensure(&mut app, &ctx, 0);
        let mut delta = ctx.tex_manager().write().take_delta();
        let egui::ImageData::Color(image) = &delta.set.get(&texture).expect("filter preview must update channel overlay")[0].image;
        let st = app.session.active().unwrap();
        let original = crate::channel_view::render(&before, &st.channel_view, before.bounds(), 1, false, st.active_layer).unwrap();
        assert_ne!(image.pixels, original, "a nontrivial filter must be visible before OK");
        delta.clear();
        assert!(std::sync::Arc::ptr_eq(&before, &app.session.active().unwrap().doc));
        app.ui.dialogs.last_mut().unwrap().fields.insert("__preview".into(), json!(false));
        crate::channel_view::ensure(&mut app, &ctx, 0);
        let mut delta = ctx.tex_manager().write().take_delta();
        let egui::ImageData::Color(image) = &delta.set.get(&texture).expect("Preview off restores channel")[0].image;
        assert_eq!(image.pixels, original);
        delta.clear();
        app.ui.dialogs.last_mut().unwrap().fields.insert("__preview".into(), json!(true));
        crate::channel_view::ensure(&mut app, &ctx, 0);
        ctx.tex_manager().write().take_delta().clear();
        crate::dialogs::cancel(&mut app, dialog).unwrap();
        crate::channel_view::ensure(&mut app, &ctx, 0);
        let mut delta = ctx.tex_manager().write().take_delta();
        let egui::ImageData::Color(image) = &delta.set.get(&texture).expect("Cancel restores channel")[0].image;
        assert_eq!(image.pixels, original);
        delta.clear();
        assert!(std::sync::Arc::ptr_eq(&before, &app.session.active().unwrap().doc));
    }

    fn assert_filter_preview_target(target: photocraft_engine::channel_cmds::ChannelTarget) {
        for depth in [8, 16, 32] {
            let mut session = photocraft_engine::Session::new();
            session.execute("file.new", json!({"width":16,"height":16,"depth":depth})).unwrap();
            session.execute("channel.new", json!({"name":"Alpha","fill":"black"})).unwrap();
            session
                .edit("Fixture", |doc, active| {
                    let layer = doc.layer_mut(active.unwrap()).unwrap();
                    layer.mask = Some(photocraft_doc::LayerMask::hide_all());
                    for y in 0..16 {
                        for x in 0..8 {
                            layer.surface_mut().unwrap().write_pixel(x, y, &[0.0, 0.25, 0.5, 1.0]);
                            layer.mask.as_mut().unwrap().surface.write_pixel(x, y, &[1.0]);
                        }
                    }
                    for y in 0..16 {
                        for x in 0..8 {
                            doc.channels[0].surface.write_pixel(x, y, &[1.0]);
                        }
                    }
                    Ok(())
                })
                .unwrap();
            session.active_mut().unwrap().channel_view.target = target;
            let mut app = PhotocraftApp::new(session, crate::Services::default());
            app.background_jobs = false;
            let dialog = crate::filter_dialog::open(&mut app, "filter.blur.gaussianBlur").unwrap();
            app.ui.dialogs.last_mut().unwrap().fields.insert("radius".into(), json!(2.0));
            ensure_filter_preview(&mut app, 0, &egui::Context::default()).unwrap();
            let preview = app.filter_preview.as_ref().unwrap().result.as_ref().unwrap().clone();
            crate::dialogs::confirm(&mut app, dialog).unwrap();
            let final_doc = &app.session.active().unwrap().doc;
            let bounds = final_doc.bounds();
            assert_eq!(
                preview.layers[0].surface().unwrap().read_region(bounds),
                final_doc.layers[0].surface().unwrap().read_region(bounds),
                "RGB: {depth} {target:?}"
            );
            assert_eq!(preview.channels[0].surface.read_region(bounds), final_doc.channels[0].surface.read_region(bounds), "alpha: {depth} {target:?}");
            assert_eq!(
                preview.layers[0].mask.as_ref().unwrap().surface.read_region(bounds),
                final_doc.layers[0].mask.as_ref().unwrap().surface.read_region(bounds),
                "mask: {depth} {target:?}"
            );
        }
    }

    #[test]
    fn filter_preview_matches_applied_alpha_target() {
        assert_filter_preview_target(photocraft_engine::channel_cmds::ChannelTarget::Alpha(0));
    }

    #[test]
    fn filter_preview_matches_applied_color_target() {
        assert_filter_preview_target(photocraft_engine::channel_cmds::ChannelTarget::Color(0));
    }

    #[test]
    fn filter_preview_matches_applied_layer_mask_target() {
        assert_filter_preview_target(photocraft_engine::channel_cmds::ChannelTarget::LayerMask);
    }

    #[test]
    fn filter_preview_refreshes_when_channel_target_changes() {
        use photocraft_engine::channel_cmds::ChannelTarget;
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.background_jobs = false;
        app.run("file.new", json!({"width":16,"height":16})).unwrap();
        for name in ["First", "Second"] {
            app.run("channel.new", json!({"name":name,"fill":"black"})).unwrap();
        }
        app.session
            .edit("Fixture", |doc, _| {
                for y in 0..16 {
                    for x in 0..8 {
                        doc.channels[1].surface.write_pixel(x, y, &[1.0]);
                    }
                }
                Ok(())
            })
            .unwrap();
        app.session.active_mut().unwrap().channel_view.target = ChannelTarget::Alpha(0);
        let revision = app.session.active().unwrap().revision;
        let dialog = crate::filter_dialog::open(&mut app, "filter.blur.gaussianBlur").unwrap();
        app.ui.dialogs.last_mut().unwrap().fields.insert("radius".into(), json!(2.0));
        let ctx = egui::Context::default();
        let (first, first_key) = filter_channel_preview(&mut app, 0, &ctx).unwrap();
        app.session.active_mut().unwrap().channel_view.target = ChannelTarget::Alpha(1);
        let (second, second_key) = filter_channel_preview(&mut app, 0, &ctx).unwrap();
        assert_eq!(app.session.active().unwrap().revision, revision);
        assert_ne!(first_key, second_key);
        assert_ne!(first.channels[1].surface.read_region(first.bounds()), second.channels[1].surface.read_region(second.bounds()));
        let (unchanged, unchanged_key) = filter_channel_preview(&mut app, 0, &ctx).unwrap();
        assert!(std::sync::Arc::ptr_eq(&second, &unchanged));
        assert_eq!(second_key, unchanged_key);
        crate::dialogs::confirm(&mut app, dialog).unwrap();
        let applied = &app.session.active().unwrap().doc;
        assert_eq!(second.channels[1].surface.read_region(second.bounds()), applied.channels[1].surface.read_region(applied.bounds()));
    }

    #[test]
    fn filter_preview_overlay_matches_quick_mask_and_mixed_color_views() {
        for quick_mask in [false, true] {
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
            app.background_jobs = false;
            app.run("file.new", json!({"width":16,"height":16})).unwrap();
            app.session
                .edit("Fixture", |doc, active| {
                    if quick_mask {
                        doc.quick_mask =
                            Some(photocraft_doc::AlphaChannel::new("Quick Mask", photocraft_raster::Surface::new(photocraft_color::PixelFormat::GRAY8)));
                    }
                    for y in 0..16 {
                        for x in 0..8 {
                            if let Some(mask) = &mut doc.quick_mask {
                                mask.surface.write_pixel(x, y, &[1.0]);
                            } else {
                                doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap().write_pixel(x, y, &[0.0, 0.25, 0.5, 1.0]);
                            }
                        }
                    }
                    Ok(())
                })
                .unwrap();
            if !quick_mask {
                let view = &mut app.session.active_mut().unwrap().channel_view;
                view.target = photocraft_engine::channel_cmds::ChannelTarget::Color(0);
                view.set_color_visible(2, false);
            }
            let ctx = egui::Context::default();
            let texture = crate::channel_view::ensure(&mut app, &ctx, 0).unwrap();
            ctx.tex_manager().write().take_delta().clear();
            let dialog = crate::filter_dialog::open(&mut app, "filter.blur.gaussianBlur").unwrap();
            app.ui.dialogs.last_mut().unwrap().fields.insert("radius".into(), json!(2.0));
            crate::channel_view::ensure(&mut app, &ctx, 0).unwrap();
            let mut delta = ctx.tex_manager().write().take_delta();
            let egui::ImageData::Color(image) = &delta.set.get(&texture).expect("preview must refresh overlay")[0].image;
            crate::dialogs::confirm(&mut app, dialog).unwrap();
            let st = app.session.active().unwrap();
            let expected = crate::channel_view::render(&st.doc, &st.channel_view, st.doc.bounds(), 1, false, st.active_layer).unwrap();
            assert_eq!(image.pixels, expected, "quick_mask={quick_mask}");
            delta.clear();
        }
    }

    /// A CPU-only app must be able to generate the actual rotation preview without a
    /// GPU adapter; checking the result also verifies Preview off and document integrity.
    #[test]
    fn rotation_preview_is_computed_headlessly_without_mutating_document() {
        let mut session = photocraft_engine::Session::new();
        session.execute("file.new", json!({"width": 96, "height": 48})).unwrap();
        let mut app = PhotocraftApp::new(session, crate::Services::default());
        assert!(app.gpu.is_none());
        crate::filter_dialog::open(&mut app, "image.rotation.arbitrary").unwrap();
        let dialog = app.ui.dialogs.last_mut().unwrap();
        dialog.fields.insert("angle".into(), json!(90.0));
        dialog.fields.insert("direction".into(), json!("cw"));

        let (factor, _, size) = ensure_filter_preview(&mut app, 0, &egui::Context::default()).expect("CPU preview computed");
        assert_eq!(factor, 1);
        assert_eq!(size, [48, 96]);
        let original = &app.session.active().unwrap().doc;
        assert_eq!([original.size.width, original.size.height], [96, 48]);

        app.ui.dialogs.last_mut().unwrap().fields.insert("__preview".into(), json!(false));
        assert!(ensure_filter_preview(&mut app, 0, &egui::Context::default()).is_none(), "disabling Preview must hide the preview");
        assert_eq!([app.session.active().unwrap().doc.size.width, app.session.active().unwrap().doc.size.height], [96, 48]);
    }

    /// A dialog operates on the active document and cannot render a preview into a
    /// background tab, even if that tab has a valid canvas index.
    #[test]
    fn filter_preview_is_not_returned_for_inactive_document() {
        let mut session = photocraft_engine::Session::new();
        session.execute("file.new", json!({"width": 96, "height": 48})).unwrap();
        let mut app = PhotocraftApp::new(session, crate::Services::default());
        crate::filter_dialog::open(&mut app, "image.rotation.arbitrary").unwrap();
        app.ui.dialogs.last_mut().unwrap().fields.insert("angle".into(), json!(90.0));
        assert!(ensure_filter_preview(&mut app, 0, &egui::Context::default()).is_some());

        app.session.execute("file.new", json!({"width": 40, "height": 20})).unwrap();
        assert_eq!(app.session.active_index(), Some(1));
        assert!(ensure_filter_preview(&mut app, 0, &egui::Context::default()).is_none(), "an inactive tab must never inherit the command preview");
    }
}

#[cfg(test)]
#[path = "canvas_display_memo_tests.rs"]
mod display_memo_tests;
