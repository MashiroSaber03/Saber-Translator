//! Layers panel row: the mask thumbnails after the layer thumbnail (#153).
//!
//! Like Photoshop, each mask (the pixel mask, then the vector mask) gets a link chain and a
//! thumbnail: the chain shows whether the mask moves with the layer and toggles it on click
//! (`layer.layerMask.linked` / `layer.vectorMask.linked`); a disabled mask is crossed out with
//! a red X and ⇧-clicking a mask thumbnail toggles it (`…enabled`). Vector masks render as a
//! white (revealed) / grey (hidden) thumbnail, cached like the other row thumbnails.
//!
//! #196: ⌥-click a layer-mask thumbnail to view the mask (⇧⌥: as an overlay), ⌘-click either
//! mask to load it as a selection, and click the vector mask to target it (brackets).

use egui::{Color32, Painter, Pos2, Rect, Sense, Stroke, StrokeKind, pos2, vec2};
use photocraft_doc::{Document, Layer, LayerId, VectorMask};
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::theme::Tokens;

/// Width of the link-chain slot in front of a mask thumbnail.
pub const CHAIN_W: f32 = 12.0;
/// Thumbnail cache kinds (the second half of `PhotocraftApp::thumbs` keys).
pub const THUMB_LAYER: u8 = 0;
pub const THUMB_MASK: u8 = 1;
pub const THUMB_VECTOR: u8 = 2;

/// Which of a layer's masks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskKind {
    Pixel,
    Vector,
}

impl MaskKind {
    /// The engine command prefix for this mask.
    fn command(self) -> &'static str {
        match self {
            MaskKind::Pixel => "layer.layerMask",
            MaskKind::Vector => "layer.vectorMask",
        }
    }
    fn label(self) -> &'static str {
        match self {
            MaskKind::Pixel => "layer mask",
            MaskKind::Vector => "vector mask",
        }
    }
}

/// `(kind, enabled, linked)` for each of `l`'s masks, in row order.
pub fn masks(l: &Layer) -> Vec<(MaskKind, bool, bool)> {
    let mut v = Vec::with_capacity(2);
    if let Some(m) = &l.mask {
        v.push((MaskKind::Pixel, m.enabled, m.linked));
    }
    if let Some(m) = &l.vector_mask {
        v.push((MaskKind::Vector, m.enabled, m.linked));
    }
    v
}

/// Width `l`'s masks take in a row with `ts`-point thumbnails (chain + thumbnail + gap each).
pub fn width(l: &Layer, ts: f32) -> f32 {
    masks(l).len() as f32 * (CHAIN_W + ts + 6.0)
}

/// Thumbnail size for `l`'s row: `ts`, shrunk (down to 14 pt) when the layer and mask
/// thumbnails would leave the name less than its minimum of the `room` they share.
pub fn thumb_size(l: &Layer, ts: f32, room: f32) -> f32 {
    let n = masks(l).len() as f32;
    let need = ts + 6.0 + width(l, ts) + crate::layer_row_ui::MIN_NAME_W;
    let short = need - room;
    if short <= 0.0 { ts } else { (ts - short / (1.0 + n)).clamp(14.0_f32.min(ts), ts) }
}

/// Where a row's mask thumbnails and chains went.
#[derive(Clone, Debug, Default)]
pub struct MaskRects {
    pub thumbs: Vec<(MaskKind, Rect)>,
    pub chains: Vec<(MaskKind, Rect)>,
    /// A chain was clicked this frame (the row shouldn't treat the click as a selection).
    pub clicked: bool,
}

impl MaskRects {
    pub fn thumb(&self, kind: MaskKind) -> Option<Rect> {
        self.thumbs.iter().find(|(k, _)| *k == kind).map(|(_, r)| *r)
    }
    /// The mask thumbnail under `p`, if any.
    pub fn hit(&self, p: Pos2) -> Option<MaskKind> {
        self.thumbs.iter().find(|(_, r)| r.expand(2.0).contains(p)).map(|(k, _)| *k)
    }
}

/// Paint `l`'s chains and mask thumbnails from `*x` (just right of the layer thumbnail plus its
/// gap), advancing `*x` past them, and push chain clicks as commands.
#[allow(clippy::too_many_arguments)]
pub fn paint(
    app: &mut PhotocraftApp,
    ctx: &egui::Context,
    ui: &egui::Ui,
    painter: &Painter,
    doc: &Document,
    l: &Layer,
    x: &mut f32,
    cy: f32,
    ts: f32,
    actions: &mut Vec<(String, Value)>,
) -> MaskRects {
    let t = Tokens::get(ctx);
    let mut out = MaskRects::default();
    for (kind, enabled, linked) in masks(l) {
        // The chain sits in the gap before the thumbnail, centred between the two.
        let chain = Rect::from_min_max(pos2(*x - 4.0, cy - ts / 2.0), pos2(*x - 4.0 + CHAIN_W, cy + ts / 2.0));
        let resp = ui.interact(chain, ui.id().with(("mask-chain", l.id.0, kind == MaskKind::Vector)), Sense::click());
        if linked {
            paint_chain(painter, chain.center(), if resp.hovered() { t.text } else { t.icon });
        } else if resp.hovered() {
            paint_chain(painter, chain.center(), t.text_faint);
        }
        let (verb, what, name) = (if linked { "Unlink" } else { "Link" }, kind.label(), l.name.clone());
        let resp = resp.on_hover_text(if linked { tl!("Unlink the mask from the layer") } else { tl!("Link the mask to the layer") });
        let template = "{verb} {mask} {layer}";
        resp.widget_info(|| {
            egui::WidgetInfo::labeled(
                egui::WidgetType::Button,
                true,
                crate::i18n::fmt(tl!(template), &[("verb", tl!(verb)), ("mask", tl!(what)), ("layer", &name)]),
            )
        });
        if resp.clicked() {
            out.clicked = true;
            actions.push((format!("{}.linked", kind.command()), json!({"layer": l.id.0, "linked": !linked})));
        }
        out.chains.push((kind, chain));
        *x += CHAIN_W - 2.0;
        let r = Rect::from_min_size(pos2(*x, cy - ts / 2.0), vec2(ts, ts));
        if ui.is_rect_visible(r) {
            let tex = match kind {
                MaskKind::Pixel => l.mask.as_ref().map(|m| app.mask_thumb(ctx, doc, l.id, m)),
                MaskKind::Vector => l.vector_mask.as_ref().map(|m| app.vector_mask_thumb(ctx, doc, l.id, m)),
            };
            if let Some(tex) = tex {
                painter.image(tex, r, Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)), Color32::WHITE);
            }
            if !enabled {
                // Photoshop's red X over a disabled mask.
                let st = Stroke::new(1.5, t.danger);
                let k = r.shrink(1.0);
                painter.line_segment([k.left_top(), k.right_bottom()], st);
                painter.line_segment([k.right_top(), k.left_bottom()], st);
            }
        }
        painter.rect_stroke(
            r,
            if t.pro { 0.0 } else { 4.0 },
            Stroke::new(1.0, if t.pro { Color32::from_gray(20) } else { t.field_border }),
            StrokeKind::Outside,
        );
        out.thumbs.push((kind, r));
        *x += ts + 6.0 + 2.0;
    }
    let rects = (out.thumbs.clone(), out.chains.clone());
    ctx.data_mut(|d| d.get_temp_mut_or_default::<std::collections::HashMap<u64, Rects>>(rects_id()).insert(l.id.0, rects));
    out
}

type Rects = (Vec<(MaskKind, Rect)>, Vec<(MaskKind, Rect)>);

fn rects_id() -> egui::Id {
    egui::Id::new("mask-thumb-rects")
}

/// The mask thumbnails and chains last drawn for layer `id` (for tests and automation).
pub fn recorded(ctx: &egui::Context, id: u64) -> Option<Rects> {
    ctx.data(|d| d.get_temp::<std::collections::HashMap<u64, Rects>>(rects_id())).and_then(|m| m.get(&id).cloned())
}

/// A small vertical chain: two interlocking links.
fn paint_chain(painter: &Painter, c: Pos2, color: Color32) {
    let st = Stroke::new(1.2, color);
    let link = vec2(5.0, 7.0);
    painter.rect_stroke(Rect::from_center_size(c - vec2(0.0, 2.5), link), 2.0, st, StrokeKind::Middle);
    painter.rect_stroke(Rect::from_center_size(c + vec2(0.0, 2.5), link), 2.0, st, StrokeKind::Middle);
}

/// The command a click on mask thumbnail `kind` issues with modifiers `m`, if it is one that
/// replaces the normal click: ⇧ toggles the mask on/off; ⌥ views a layer mask alone in
/// grayscale and ⇧⌥ as a rubylith overlay (again: back to the composite), like Photoshop.
pub fn click_command(l: &Layer, kind: MaskKind, m: egui::Modifiers) -> Option<(String, Value)> {
    if m.command {
        return None;
    }
    if m.alt {
        let mode = if m.shift { "toggleOverlay" } else { "toggleGray" };
        return (kind == MaskKind::Pixel && l.mask.is_some()).then(|| (photocraft_engine::mask_view_cmds::ID.into(), json!({"layer": l.id.0, "mode": mode})));
    }
    if !m.shift {
        return None;
    }
    let enabled = masks(l).into_iter().find(|(k, _, _)| *k == kind)?.1;
    Some((format!("{}.enabled", kind.command()), json!({"layer": l.id.0, "enabled": !enabled})))
}

/// The `select.loadSelection` params for a ⌘-click on mask thumbnail `kind` (⌘⇧ add, ⌘⌥
/// subtract, ⌘⇧⌥ intersect).
pub fn load_params(l: &Layer, kind: MaskKind, m: egui::Modifiers) -> Value {
    let channel = match kind {
        MaskKind::Pixel => "mask",
        MaskKind::Vector => "vectorMask",
    };
    json!({"channel": channel, "layer": l.id.0, "operation": crate::channels_panel::load_operation(m)})
}

/// What double-clicking a mask thumbnail does: pixel mask launches Select and Mask if preference is enabled.
pub fn double_click_command(app: &PhotocraftApp, kind: MaskKind) -> Option<(&'static str, Value)> {
    if kind == MaskKind::Pixel
        && app.session.prefs().tools.double_click_layer_mask_launches_select_and_mask
        && crate::menus::is_enabled(app, "select.selectAndMask")
    {
        Some(("select.selectAndMask", Value::Null))
    } else {
        None
    }
}

/// The active layer of `st` has a vector mask (shape layers' paths are content, not masks).
pub fn has_vector_mask(st: &photocraft_engine::DocState) -> bool {
    st.active_layer.and_then(|id| st.doc.layer(id)).is_some_and(|l| l.vector_mask.is_some() && !matches!(l.content, photocraft_doc::LayerContent::Shape(_)))
}

/// Photoshop's target brackets: corner marks just outside thumbnail `r`.
pub fn paint_brackets(painter: &Painter, r: Rect, color: Color32) {
    let r = r.expand(3.0);
    let k = 6.0;
    let st = Stroke::new(1.5, color);
    for (c, dx, dy) in [(r.left_top(), 1.0, 1.0), (r.right_top(), -1.0, 1.0), (r.right_bottom(), -1.0, -1.0), (r.left_bottom(), 1.0, -1.0)] {
        painter.line_segment([c, c + vec2(k * dx, 0.0)], st);
        painter.line_segment([c, c + vec2(0.0, k * dy)], st);
    }
    painter.ctx().data_mut(|d| d.insert_temp(bracket_id(), r));
}

fn bracket_id() -> egui::Id {
    egui::Id::new("layer-target-brackets")
}

/// Where the target brackets were last drawn (for tests and automation).
pub fn brackets(ctx: &egui::Context) -> Option<Rect> {
    ctx.data(|d| d.get_temp(bracket_id()))
}

/// Cheap identity of a vector mask's geometry and state.
pub fn vector_fingerprint(m: &VectorMask) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |v: u64| h = (h ^ v).wrapping_mul(0x100_0000_01b3);
    mix(u64::from(m.path.inverted) | u64::from(m.path.fill_rule == photocraft_doc::FillRule::EvenOdd) << 1);
    mix(u64::from(m.density.to_bits()));
    for s in &m.path.subpaths {
        mix(u64::from(s.closed) | (s.op as u64) << 1 | (s.knots.len() as u64) << 8);
        for k in &s.knots {
            for p in [k.anchor, k.in_ctrl, k.out_ctrl] {
                mix(p.x.to_bits());
                mix(p.y.to_bits());
            }
        }
    }
    h
}

/// A `side`-pixel square thumbnail of vector mask `m`: white where it reveals, grey where it
/// hides, letterboxed to the document's aspect like the other row thumbnails. Disabled masks
/// still show their shape (the row crosses them out).
pub fn vector_thumb_image(doc: &Document, m: &VectorMask, side: usize) -> egui::ColorImage {
    let (w, h) = (f64::from(doc.size.width.max(1)), f64::from(doc.size.height.max(1)));
    let s = side as f64 / w.max(h);
    let (ox, oy) = ((side as f64 - w * s) / 2.0, (side as f64 - h * s) / 2.0);
    let shown = VectorMask { path: m.path.transform(&photocraft_geom::Affine { m: [s, 0.0, 0.0, s, ox, oy] }), enabled: true, ..m.clone() };
    let n = side as i32;
    let values = photocraft_vector::vector_mask_values(&shown, photocraft_geom::Rect::new(0, 0, n, n));
    let (x0, y0, x1, y1) = (ox.floor() as usize, oy.floor() as usize, (ox + w * s).ceil() as usize, (oy + h * s).ceil() as usize);
    let mut px = vec![Color32::TRANSPARENT; side * side];
    for (i, p) in px.iter_mut().enumerate() {
        let (tx, ty) = (i % side, i / side);
        if tx < x0 || ty < y0 || tx >= x1 || ty >= y1 {
            continue;
        }
        let v = values.get(i).copied().unwrap_or(1.0).clamp(0.0, 1.0);
        *p = Color32::from_gray((128.0 + 127.0 * v + 0.5) as u8);
    }
    egui::ColorImage::new([side, side], px)
}

impl PhotocraftApp {
    /// Cached thumbnail of `l`'s vector mask (see [`vector_thumb_image`]).
    pub fn vector_mask_thumb(&mut self, ctx: &egui::Context, doc: &Document, id: LayerId, m: &VectorMask) -> egui::TextureId {
        let rev = vector_fingerprint(m) ^ (u64::from(doc.size.width) << 40) ^ (u64::from(doc.size.height) << 20);
        let key = (id, THUMB_VECTOR);
        if let Some((r, tex)) = self.thumbs.get(&key)
            && *r == rev
        {
            return tex.id();
        }
        self.store_thumb(ctx, key, rev, vector_thumb_image(doc, m, 64))
    }
}

#[cfg(test)]
mod tests;
