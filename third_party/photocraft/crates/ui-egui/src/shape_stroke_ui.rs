//! Shape stroke geometry controls shared by the tool defaults and Properties.
use egui::vec2;
use photocraft_doc::{LineCap, LineJoin, ShapeStroke, StrokeAlign};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// Colour and width stay in their existing controls. These options never replace a shape's paint.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StrokeOptions {
    pub align: StrokeAlign,
    pub cap: LineCap,
    pub join: LineJoin,
    pub miter_limit: f32,
    pub dashes: Vec<f32>,
    pub dash_offset: f32,
    pub opacity: f32,
}

impl Default for StrokeOptions {
    fn default() -> Self {
        Self::from(&ShapeStroke::default())
    }
}

impl From<&ShapeStroke> for StrokeOptions {
    fn from(s: &ShapeStroke) -> Self {
        Self { align: s.align, cap: s.cap, join: s.join, miter_limit: s.miter_limit, dashes: s.dashes.clone(), dash_offset: s.dash_offset, opacity: s.opacity }
    }
}

impl StrokeOptions {
    pub fn params(&self) -> Value {
        json!({"align": format!("{:?}", self.align).to_ascii_lowercase(),
            "cap": format!("{:?}", self.cap).to_ascii_lowercase(), "join": format!("{:?}", self.join).to_ascii_lowercase(),
            "miterLimit": self.miter_limit, "dashes": self.dashes, "dashOffset": self.dash_offset, "opacity": self.opacity * 100.0})
    }

    pub fn stroke(&self, width: f32, paint: photocraft_doc::Fill) -> ShapeStroke {
        ShapeStroke {
            width,
            paint,
            align: self.align,
            cap: self.cap,
            join: self.join,
            miter_limit: self.miter_limit,
            dashes: self.dashes.clone(),
            dash_offset: self.dash_offset,
            opacity: self.opacity,
        }
    }

    fn label(&self) -> &'static str {
        if self.dashes.is_empty() {
            "Solid"
        } else if self.dashes == [4.0, 2.0] {
            "Dashed"
        } else if self.dashes == [0.0, 2.0] && self.cap == LineCap::Round {
            "Dotted"
        } else {
            "Custom"
        }
    }

    fn preset(&mut self, name: &str) {
        self.dash_offset = 0.0;
        self.dashes = match name {
            "Dashed" => vec![4.0, 2.0],
            "Dotted" => {
                self.cap = LineCap::Round;
                vec![0.0, 2.0]
            }
            _ => Vec::new(),
        };
    }

    fn valid(&self) -> bool {
        self.stroke(3.0, photocraft_doc::Fill::Solid(photocraft_doc::Color::BLACK)).validate().is_ok()
    }
}

#[derive(Clone)]
struct DragCache {
    shape: photocraft_doc::ShapeLayer,
    clip: egui::Rect,
    image_rect: egui::Rect,
    texture: egui::TextureHandle,
}

/// The drag uses the committed vector rasterizer. Cache by geometry/style/visible region, and
/// cap proxy resolution so oversized windows cannot allocate a document-sized preview.
pub fn paint_shape_preview(painter: &egui::Painter, shape: &photocraft_doc::ShapeLayer) {
    let clip = painter.clip_rect();
    if !clip.is_finite() || clip.is_negative() {
        return;
    }
    let id = egui::Id::new("shape-drag-render");
    let cached = painter.ctx().data(|d| d.get_temp::<DragCache>(id));
    let cache = cached.filter(|c| c.shape == *shape && c.clip == clip).or_else(|| {
        let scale = (2048.0 / clip.width().max(clip.height()).max(1.0)).min(1.0);
        let mut proxy = shape.clone();
        let tr = photocraft_geom::Affine { m: [scale as f64, 0.0, 0.0, scale as f64, -(clip.left() * scale) as f64, -(clip.top() * scale) as f64] };
        proxy.path = proxy.path.transform(&tr);
        if let Some(stroke) = proxy.stroke.as_mut() {
            stroke.width *= scale;
        }
        let canvas = photocraft_geom::Rect::from_size(photocraft_geom::Size::new((clip.width() * scale).ceil() as u32, (clip.height() * scale).ceil() as u32));
        let compiled = photocraft_vector::CompiledShape::new(&proxy, photocraft_vector::DEFAULT_TOLERANCE, canvas);
        let bounds = compiled.bounds()?.intersect(&canvas);
        if bounds.is_empty() {
            return None;
        }
        let rgba = compiled.render_rgba(bounds);
        let bytes: Vec<u8> = rgba.iter().flat_map(|p| p.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)).collect();
        let texture = painter.ctx().load_texture(
            "shape-drag",
            egui::ColorImage::from_rgba_unmultiplied([bounds.width() as usize, bounds.height() as usize], &bytes),
            egui::TextureOptions::LINEAR,
        );
        let screen = |x: i32, y: i32| clip.min + vec2(x as f32, y as f32) / scale;
        Some(DragCache {
            shape: shape.clone(),
            clip,
            image_rect: egui::Rect::from_min_max(screen(bounds.x0, bounds.y0), screen(bounds.x1, bounds.y1)),
            texture,
        })
    });
    if let Some(cache) = cache {
        painter.image(cache.texture.id(), cache.image_rect, egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
        painter.ctx().data_mut(|d| d.insert_temp(id, cache));
    }
}

mod editor;

pub use editor::{StrokeEditor, button};

fn open_frame_id() -> egui::Id {
    egui::Id::new("shape-stroke-open-frame")
}

pub(super) fn mark_rendered(ctx: &egui::Context, key: &str) {
    let pass = ctx.cumulative_pass_nr();
    ctx.data_mut(|d| d.insert_temp(egui::Id::new(("shape-stroke-rendered", key)), pass));
}

/// A hidden options row or collapsed Appearance section no longer owns input.
/// Called after all options bars and Properties windows have had a chance to draw.
pub(crate) fn finish_frame(app: &mut crate::PhotocraftApp, ctx: &egui::Context) {
    let Some(e) = &app.ui.stroke_editor else { return };
    let shown = ctx.data(|d| d.get_temp::<u64>(egui::Id::new(("shape-stroke-rendered", e.key.as_str()))));
    if shown != Some(ctx.cumulative_pass_nr()) {
        editor::close_children(ctx, &e.key);
        app.ui.stroke_editor = None;
    }
}

#[derive(Clone, Default)]
struct CanvasDismissal {
    held: bool,
    consumed_frame: Option<u64>,
}

/// Pressing the canvas dismisses this popup and consumes that entire gesture. In particular,
/// its release must not open a shape-size dialog or commit a shape dragged behind the popup.
pub(crate) fn dismiss_from_canvas(app: &mut crate::PhotocraftApp, response: &egui::Response) -> bool {
    let ctx = &response.ctx;
    let frame = ctx.cumulative_frame_nr();
    let was_open = app.ui.stroke_editor.is_some() || ctx.data(|d| d.get_temp::<u64>(open_frame_id())) == Some(frame);
    let (pressed, down) = ctx.input(|i| (i.pointer.any_pressed(), i.pointer.any_down()));
    let dismiss = was_open && pressed && response.contains_pointer();
    if dismiss && let Some(e) = app.ui.stroke_editor.take() {
        editor::close_children(ctx, &e.key);
        ctx.request_repaint();
    }
    ctx.data_mut(|d| {
        let state = d.get_temp_mut_or_default::<CanvasDismissal>(egui::Id::new("shape-stroke-canvas-dismissal"));
        state.held |= dismiss;
        let consumed = state.held || state.consumed_frame == Some(frame);
        if consumed {
            state.held = down;
            state.consumed_frame = Some(frame);
        }
        consumed
    })
}

/// Drop a popup whose source disappeared or changed outside this editor (including Undo).
pub(crate) fn sync_editor(app: &mut crate::PhotocraftApp, ctx: &egui::Context) {
    let Some(e) = &app.ui.stroke_editor else { return };
    let valid = if let Some((doc, layer)) = e.target {
        app.session.active().is_some_and(|st| {
            let locks = st.doc.effective_locks(layer);
            st.doc.id == doc
                && st.active_layer == Some(layer)
                && !locks.all
                && !locks.pixels
                && st.doc.layer(layer).is_some_and(|l| match &l.content {
                    photocraft_doc::LayerContent::Shape(shape) => shape.stroke.as_ref().map(StrokeOptions::from).unwrap_or_default() == e.original,
                    _ => false,
                })
        })
    } else {
        let tool = app.ui.tool;
        ((e.key == "tool-stroke" && crate::vector_ui::is_shape_tool(tool))
            || (e.key == "pen-stroke" && tool == crate::state::Tool::Pen && app.ui.tool_options.vector_mode == "shape"))
            && app.ui.tool_options.shape_stroke == e.original
    };
    if !valid {
        editor::close_children(ctx, &e.key);
        app.ui.stroke_editor = None;
    } else {
        // The options row can process a batched press+release before the canvas is drawn.
        // Keep the opening state for this frame so that click is still dismissal-only.
        let frame = ctx.cumulative_frame_nr();
        ctx.data_mut(|d| d.insert_temp(open_frame_id(), frame));
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod shortcut_tests;
