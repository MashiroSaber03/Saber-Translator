//! Edit › Free Transform / Transform: projective transforms of layers (and the selection).

use photocraft_algo::transform::{Homography, Interp, warp_output_bounds, warp_surface, warp_surface_with};
use photocraft_color::PixelFormat;
use photocraft_doc::{Document, Layer, LayerContent, LayerId, Locks};
use photocraft_geom::{Affine, Rect};
use photocraft_raster::{Interrupt, Surface};
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

fn has_layer(s: &Session) -> std::result::Result<(), String> {
    s.active().and_then(|d| d.active_layer).map(|_| ()).ok_or_else(|| "no active layer".into())
}

fn bad(msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: "edit.transform".into(), msg: msg.into() }
}

/// Layers a transform moves as a whole, even with a selection: groups, and the kinds that keep the
/// transform instead of pixels (type, shapes, smart objects). Only pixel layers move just the
/// selected pixels.
pub fn moves_whole(layer: &Layer) -> bool {
    matches!(layer.content, LayerContent::Group(_) | LayerContent::Text(_) | LayerContent::Shape(_) | LayerContent::Smart(_))
}

/// Document-space bounds a transform of `layer` starts from (what Free Transform frames): the
/// content within the selection, or all of it for a layer that [`moves_whole`] (#2630).
/// Content scans are cached per tile: snapping asks for every layer's bounds per Move drag.
pub fn transform_bounds(doc: &Document, layer: &Layer) -> Rect {
    let content = whole_layer_bounds(layer);
    match &doc.selection {
        Some(sel) if !moves_whole(layer) => content.intersect(&sel.content_bounds()),
        _ => content,
    }
}

/// A group moves all its descendants, so its frame must not clip raster children to the selection.
fn whole_layer_bounds(layer: &Layer) -> Rect {
    let content = match &layer.content {
        LayerContent::Group(g) => g.children.iter().map(whole_layer_bounds).fold(Rect::EMPTY, |a, b| a.union(&b)),
        _ => layer.surface().map_or(Rect::EMPTY, photocraft_compose::bounds::content_bounds),
    };
    if content.is_empty() { layer.mask.as_ref().map_or(Rect::EMPTY, |m| photocraft_compose::bounds::content_bounds(&m.surface)) } else { content }
}

/// Warp a surface whose pixels outside the content read as `default` (masks, selections):
/// warp the content with alpha, then flatten back onto the default value.
pub(crate) fn warp_gray(s: &Surface, h: &Homography, interp: Interp) -> Result<Surface> {
    let default = s.default_pixel().first().copied().unwrap_or(0.0);
    let fmt = s.format();
    let src = s.content_bounds();
    let mut out = Surface::with_default(fmt, &[default]);
    if src.is_empty() {
        return Ok(out);
    }
    // Content-only copy (default 0 + alpha) so the "outside" is transparent during the warp.
    let with_alpha = PixelFormat::new(fmt.mode, fmt.sample, true);
    let mut tmp = Surface::new(with_alpha);
    let v = crate::allocation::read_region(s, src, "transforming a mask or channel")?;
    let mut lifted =
        crate::allocation::filled(v.len().checked_mul(2).ok_or_else(|| bad("the mask is too large to transform"))?, 0.0, "transforming a mask or channel")?;
    let (lifted_pairs, _) = lifted.as_chunks_mut::<2>();
    for (g, px) in v.iter().zip(lifted_pairs) {
        px.copy_from_slice(&[*g, 1.0]);
    }
    tmp.try_write_region(src, &lifted)?;
    let w = warp_surface(&tmp, src, h, interp)?;
    // Clear the old content region, then composite the warped content over it. These are written as
    // two sparse regions rather than one dense `src ∪ warped` block: a transform that moves the
    // content far away (e.g. a huge translation) would otherwise allocate a buffer spanning both and
    // hang/OOM. `Surface` is tile-sparse, so distant regions cost only their own tiles.
    let old_len = (src.width() as usize).checked_mul(src.height() as usize).ok_or_else(|| bad("the mask is too large to transform"))?;
    let old = crate::allocation::filled(old_len, default, "transforming a mask or channel")?;
    out.try_write_region(src, &old)?;
    let b = w.content_bounds();
    if !b.is_empty() {
        let px = crate::allocation::read_region(&w, b, "flattening a transformed mask or channel")?;
        let mut flat = crate::allocation::filled(px.len() / 2, 0.0, "flattening a transformed mask or channel")?;
        for (p, f) in px.as_chunks::<2>().0.iter().zip(&mut flat) {
            *f = p[0] * p[1] + default * (1.0 - p[1]);
        }
        out.try_write_region(b, &flat)?;
    }
    out.prune();
    Ok(out)
}

/// Split a single-channel surface whose outside reads as its default (a mask, a channel) by a
/// selection: (the selected values as grey + alpha = selection, the surface with the selected
/// values replaced by the default). `None` for surfaces that aren't a single grey channel.
pub fn split_gray_selected(s: &Surface, sel: &Surface) -> Result<Option<(Surface, Surface)>> {
    let fmt = s.format();
    if fmt.alpha || fmt.channels() != 1 {
        return Ok(None);
    }
    let default = s.default_pixel().first().copied().unwrap_or(0.0);
    let mut rest = s.clone();
    let mut lifted = Surface::new(PixelFormat::new(fmt.mode, fmt.sample, true));
    // Everything selected moves, including default-valued parts outside the painted content, so
    // the moved region stays aligned with the moved pixels.
    let area = sel.content_bounds();
    if area.is_empty() {
        return Ok(Some((lifted, rest)));
    }
    let v = crate::allocation::read_region(s, area, "transforming a selected mask or channel")?;
    let w = area.width() as usize;
    let mut lp = crate::allocation::capacity(
        v.len().checked_mul(2).ok_or_else(|| bad("the selected mask is too large to transform"))?,
        "transforming a selected mask or channel",
    )?;
    let mut rp = crate::allocation::capacity(v.len(), "transforming a selected mask or channel")?;
    for (i, g) in v.iter().enumerate() {
        let (x, y) = (area.x0 + (i % w) as i32, area.y0 + (i / w) as i32);
        let k = sel.sample_channel(x, y, 0);
        lp.extend([*g, k]);
        rp.push(g * (1.0 - k) + default * k);
    }
    lifted.try_write_region(area, &lp)?;
    lifted.prune();
    rest.try_write_region(area, &rp)?;
    rest.prune();
    Ok(Some((lifted, rest)))
}

/// [`warp_gray`] limited to a selection: only the selected values move (the vacated area reads as
/// the default), as the selected pixels of the layer do.
pub(crate) fn warp_gray_selected(s: &Surface, sel: &Surface, h: &Homography, interp: Interp) -> Result<Surface> {
    move_gray_selected(s, sel, &|lifted, src| Ok(warp_surface(lifted, src, h, interp)?), &|s| warp_gray(s, h, interp))
}

/// [`warp_gray_selected`] with any geometric warp: `warp` moves the lifted grey + alpha values
/// (given their bounds), `whole` moves `s` when it isn't a single grey channel.
pub(crate) fn move_gray_selected(
    s: &Surface,
    sel: &Surface,
    warp: &dyn Fn(&Surface, Rect) -> Result<Surface>,
    whole: &dyn Fn(&Surface) -> Result<Surface>,
) -> Result<Surface> {
    let Some((lifted, mut out)) = split_gray_selected(s, sel)? else { return whole(s) };
    let src = lifted.content_bounds();
    if src.is_empty() {
        return Ok(out);
    }
    let w = warp(&lifted, src)?;
    let b = w.content_bounds();
    if !b.is_empty() {
        let moved = crate::allocation::read_region(&w, b, "compositing transformed mask pixels")?;
        let under = crate::allocation::read_region(&out, b, "compositing transformed mask pixels")?;
        let mut flat = crate::allocation::filled(under.len(), 0.0, "compositing transformed mask pixels")?;
        for (i, (p, u)) in moved.as_chunks::<2>().0.iter().zip(&under).enumerate() {
            flat[i] = p[0] * p[1] + u * (1.0 - p[1]);
        }
        out.try_write_region(b, &flat)?;
    }
    out.prune();
    Ok(out)
}

/// The surface Free Transform moves by itself when the params target one (`"target"`): an
/// unlinked layer mask, an alpha channel or the Quick Mask. `None` means the layer, with its
/// linked masks (a targeted *linked* mask moves together with its layer, as in Photoshop).
pub fn lone_target<'a>(doc: &'a Document, layer: Option<LayerId>, p: &Value) -> Result<Option<&'a Surface>> {
    use crate::channel_cmds::Target;
    match crate::channel_cmds::target_of(p) {
        Target::Pixels => Ok(None),
        Target::Mask => {
            let id = layer.ok_or_else(|| EngineError::Other("no active layer".into()))?;
            let l = doc.layer(id).ok_or(EngineError::NoLayer(id))?;
            let m = l.mask.as_ref().ok_or_else(|| EngineError::Other("layer has no mask".into()))?;
            Ok((!m.linked).then_some(&m.surface))
        }
        Target::Alpha(i) => doc
            .channels
            .get(i)
            .map(|c| Some(&c.surface))
            .ok_or_else(|| EngineError::Other(format!("no alpha channel {i} (document has {})", doc.channels.len()))),
        Target::QuickMask => doc.quick_mask.as_ref().map(|c| Some(&c.surface)).ok_or_else(|| EngineError::Other("not in Quick Mask mode".into())),
    }
}

/// [`lone_target`], mutably.
pub fn lone_target_mut<'a>(doc: &'a mut Document, layer: Option<LayerId>, p: &Value) -> Result<Option<&'a mut Surface>> {
    if lone_target(doc, layer, p)?.is_none() {
        return Ok(None);
    }
    crate::channel_cmds::target_surface(doc, layer, p).map(|(s, _)| Some(s))
}

/// The frame Free Transform starts from on a lone target: its painted content (the whole canvas
/// when nothing is painted), within the selection.
pub fn target_bounds(doc: &Document, surf: &Surface) -> Rect {
    let content = surf.content_bounds();
    let content = if content.is_empty() { doc.bounds() } else { content };
    match &doc.selection {
        Some(sel) => content.intersect(&sel.content_bounds()),
        None => content,
    }
}

/// The locks `id` inherits from the groups around it.
pub(crate) fn group_locks(doc: &Document, id: LayerId) -> Locks {
    doc.path_of(id).and_then(|p| Some(doc.locks_at(p.split_last()?.1))).unwrap_or_default()
}

/// The locks a transform of `l` meets: its own and `group`'s (those it inherits). The Background's
/// position lock doesn't count: Photoshop turns it into a normal layer first.
fn transform_locks(l: &Layer, group: Locks, background: bool) -> Locks {
    let mut own = l.locks;
    if background {
        own.position = false;
        own.transparency = false;
    }
    own.union(group)
}

/// Would a transform of the layer `id` be refused because it, a group around it, or (for a group)
/// a layer inside it is locked in place? The same rule [`transform_layer`] applies at commit, so
/// Free Transform can refuse before its box opens, as Photoshop does.
pub fn locked_for_transform(doc: &Document, id: LayerId) -> bool {
    let Some(l) = doc.layer(id) else { return false };
    // An explicit stack rather than recursion: documents can nest deeply.
    let mut stack = vec![(l, group_locks(doc, id), doc.is_background(id))];
    while let Some((l, group, background)) = stack.pop() {
        let locks = transform_locks(l, group, background);
        if locks.position || locks.all {
            return true;
        }
        if let LayerContent::Group(g) = &l.content {
            stack.extend(g.children.iter().map(|c| (c, locks, false)));
        }
    }
    false
}

/// The largest result (px) a single Free Transform warp may build: 2³⁰, the bound Puppet Warp uses
/// too ([`photocraft_algo::puppet::MAX_WARP_PIXELS`]). Only the size of the result counts, not
/// where it lands, so moving a layer far off the canvas stays allowed. A scale past it is refused
/// rather than clipped (#3773): the warp would otherwise fill every tile of a result up to 2²¹ px
/// on a side before anything could fail.
pub const MAX_TRANSFORM_PIXELS: u64 = 1 << 30;

/// Tests use a far lower budget, so proving a refusal never comes near allocating the real one.
#[cfg(not(test))]
const BUDGET: u64 = MAX_TRANSFORM_PIXELS;
#[cfg(test)]
const BUDGET: u64 = 1 << 20;

/// Refuse a warp of the content inside `src` by `h` whose result would exceed the pixel budget,
/// before anything is built. `what` names the surface in the message.
pub(crate) fn check_budget(src: Rect, h: &Homography, what: &str) -> Result<()> {
    let Some(b) = warp_output_bounds(src, h) else { return Ok(()) };
    let (w, ht) = (u64::from(b.width()), u64::from(b.height()));
    if w.saturating_mul(ht) > BUDGET {
        let detail = format!("the transformed {what} would be {w} x {ht} px, more than the {BUDGET} px limit");
        return Err(crate::allocation::error("transforming pixels", &detail));
    }
    Ok(())
}

/// [`check_budget`] for every warp [`transform_layer`] would make of `l` (for a group, of every
/// layer inside it), so a transform that doesn't fit is refused before any layer is built and a
/// group that partly fits isn't half-transformed.
fn check_layer_budget(doc_sel: Option<&Surface>, l: &Layer, h: &Homography) -> Result<()> {
    if h.is_identity() {
        return Ok(());
    }
    // An explicit stack rather than recursion: documents can nest deeply.
    let mut stack = vec![(l, doc_sel)];
    while let Some((l, sel)) = stack.pop() {
        match &l.content {
            LayerContent::Group(g) => stack.extend(g.children.iter().map(|c| (c, None))),
            LayerContent::Text(_) | LayerContent::Shape(_) => {}
            LayerContent::Smart(sm) => {
                if let Some(c) = &sm.cache {
                    check_budget(c.content_bounds(), h, "smart object")?;
                }
                if let Some(m) = &sm.filter_mask {
                    check_budget(m.surface.content_bounds(), h, "smart filter mask")?;
                }
            }
            _ => {
                if let Some(surf) = l.surface() {
                    check_budget(surf.content_bounds(), h, &format!("layer \"{}\"", l.name))?;
                }
            }
        }
        if let Some(m) = &l.mask
            && m.linked
        {
            // With a selection only its region of the mask moves (`warp_gray_selected`).
            let src = match sel.filter(|_| !moves_whole(l)) {
                Some(sel) => sel.content_bounds(),
                None => m.surface.content_bounds(),
            };
            check_budget(src, h, "layer mask")?;
        }
    }
    Ok(())
}

/// `group` holds the locks `l` inherits from the groups around it ([`group_locks`]). A Background
/// is turned into a normal layer by the caller first ([`crate::extra_cmds::background_to_layer`]).
/// Refused, with `l` untouched, when a result would exceed [`MAX_TRANSFORM_PIXELS`].
pub(crate) fn transform_layer(doc_sel: Option<&Surface>, group: Locks, l: &mut Layer, h: &Homography, affine: Option<Affine>, interp: Interp) -> Result<()> {
    transform_layer_with(doc_sel, group, l, h, affine, interp, &Interrupt::NONE)
}

/// [`transform_layer`] with progress and cancellation for the pixel warp (the slow part on a
/// large layer, #1015).
fn transform_layer_with(
    doc_sel: Option<&Surface>,
    group: Locks,
    l: &mut Layer,
    h: &Homography,
    affine: Option<Affine>,
    interp: Interp,
    ctl: &Interrupt,
) -> Result<()> {
    check_layer_budget(doc_sel, l, h)?;
    apply_transform(doc_sel, group, l, h, affine, interp, ctl)
}

fn apply_transform(
    doc_sel: Option<&Surface>,
    group: Locks,
    l: &mut Layer,
    h: &Homography,
    affine: Option<Affine>,
    interp: Interp,
    ctl: &Interrupt,
) -> Result<()> {
    crate::allocation::checkpoint("transforming pixels")?;
    let locks = transform_locks(l, group, false);
    if locks.position || locks.all {
        return Err(EngineError::Other(format!("layer \"{}\" is locked", l.name)));
    }
    if h.is_identity() {
        if let LayerContent::Group(g) = &mut l.content {
            for c in g.children.iter_mut() {
                apply_transform(None, locks, c, h, affine, interp, ctl)?;
            }
        }
        return Ok(());
    }
    // With a selection only the selected pixels move, and so only the same region of a linked
    // mask (#205). Groups, type, shapes and smart objects move whole.
    let mask_sel = doc_sel.filter(|_| !moves_whole(l));
    match &mut l.content {
        LayerContent::Group(g) => {
            for c in g.children.iter_mut() {
                apply_transform(None, locks, c, h, affine, interp, ctl)?;
            }
        }
        LayerContent::Text(t) => {
            let Some(a) = affine else {
                return Err(EngineError::Other("Distort and Perspective need rasterized type (Type › Rasterize Type Layer)".into()));
            };
            t.transform = a.mul(&t.transform);
        }
        LayerContent::Shape(sh) => match affine {
            Some(a) => crate::vector_cmds::transform_shape(sh, &a),
            // Distort and Perspective map the path and keep the shape (#3009).
            None => crate::vector_cmds::map_shape(sh, Some(h), None)?,
        },
        LayerContent::Smart(sm) => {
            // Smart objects keep the transform and re-render from their source afterwards
            // (`refresh_text`), so repeated transforms don't degrade the pixels. Distort and
            // Perspective keep the full projective map, so the fourth corner survives re-renders.
            match affine {
                Some(a) => crate::smart_cmds::transform_placement(sm, &a),
                None => crate::smart_cmds::set_placement(sm, h.mul(&crate::smart_cmds::placement(sm))),
            }
            // Fallback appearance for sources that can't be re-rendered.
            if let Some(c) = &mut sm.cache {
                let src = c.content_bounds();
                *c = warp_surface(c, src, h, interp)?;
            }
            if let Some(m) = &mut sm.filter_mask {
                m.surface = warp_gray(&m.surface, h, interp)?;
            }
        }
        _ => {
            if let Some(surf) = l.surface_mut() {
                let src = surf.content_bounds();
                *surf = match doc_sel {
                    // Only the selected pixels move: lift them, clear them, warp and paste back.
                    Some(sel) => {
                        let (lifted, mut rest) = split_selected(surf, sel)?;
                        let moved = warp_surface_with(&lifted, src, h, interp, ctl)?.ok_or(EngineError::Cancelled)?;
                        composite_over(&mut rest, &moved);
                        rest.prune();
                        rest
                    }
                    None => warp_surface_with(surf, src, h, interp, ctl)?.ok_or(EngineError::Cancelled)?,
                };
            }
        }
    }
    if let Some(m) = l.mask.as_mut()
        && m.linked
    {
        m.surface = match mask_sel {
            Some(sel) => warp_gray_selected(&m.surface, sel, h, interp)?,
            None => warp_gray(&m.surface, h, interp)?,
        };
    }
    if let Some(vm) = l.vector_mask.as_mut()
        && vm.linked
    {
        vm.path = match affine {
            Some(a) => vm.path.transform(&a),
            None => crate::vector_cmds::map_path(&vm.path, Some(h), None)?,
        };
    }
    Ok(())
}

/// Split a layer surface by a selection: (selected pixels, everything else), both with alpha.
pub fn split_selected(surf: &Surface, sel: &Surface) -> Result<(Surface, Surface)> {
    let fmt = surf.format();
    let with_alpha = PixelFormat::new(fmt.mode, fmt.sample, true);
    let mut lifted = Surface::new(with_alpha);
    let mut rest = surf.try_convert(with_alpha)?;
    let src = surf.content_bounds();
    if src.is_empty() {
        return Ok((lifted, rest));
    }
    let n = with_alpha.channels();
    let px = crate::allocation::read_region(&rest, src, "transforming selected pixels")?;
    let mut lp = crate::allocation::filled(px.len(), 0.0, "transforming selected pixels")?;
    lp.copy_from_slice(&px);
    let mut rp = px;
    let w = src.width() as usize;
    // A pixel the selection fully lifts out is reset to the surface's (transparent) default;
    // keeping its colour would make `content_bounds` still count it, so the layer's bounds would
    // include the area the pixels were moved from.
    let mut rest_clear = rest.default_pixel();
    if rest_clear.last().is_some_and(|a| *a > 0.0) {
        rest_clear.fill(0.0);
    }
    for (i, (l, r)) in lp.chunks_exact_mut(n).zip(rp.chunks_exact_mut(n)).enumerate() {
        let (x, y) = (src.x0 + (i % w) as i32, src.y0 + (i / w) as i32);
        let k = sel.sample_channel(x, y, 0);
        l[n - 1] *= k;
        r[n - 1] *= 1.0 - k;
        if r[n - 1] <= 0.0 {
            r.iter_mut().zip(&rest_clear).for_each(|(d, c)| *d = *c);
        }
    }
    lifted.try_write_region(src, &lp)?;
    lifted.prune();
    rest.try_write_region(src, &rp)?;
    Ok((lifted, rest))
}

pub(crate) fn refresh_text(doc: &Document, l: &mut Layer) {
    match &mut l.content {
        LayerContent::Text(t) => crate::type_cmds::refresh(doc, t),
        LayerContent::Shape(sh) => crate::vector_cmds::refresh_shape(doc, sh),
        LayerContent::Smart(_) => {
            // Unavailable sources keep the warped cache.
            let _ = crate::smart_cmds::refresh_layer(doc, l);
        }
        LayerContent::Group(g) => g.children.iter_mut().for_each(|c| refresh_text(doc, c)),
        _ => {}
    }
}

/// Normal-blend `top` over `dst` (both straight alpha, same format).
pub(crate) fn composite_over(dst: &mut Surface, top: &Surface) {
    let b = top.content_bounds();
    if b.is_empty() {
        return;
    }
    let n = dst.format().channels();
    let a = n - 1;
    let t = top.read_region(b);
    let mut d = dst.read_region(b);
    for (dp, tp) in d.chunks_exact_mut(n).zip(t.chunks_exact(n)) {
        let (ta, da) = (tp[a], dp[a]);
        let oa = ta + da * (1.0 - ta);
        if oa > 0.0 {
            for c in 0..a {
                dp[c] = (tp[c] * ta + dp[c] * da * (1.0 - ta)) / oa;
            }
        }
        dp[a] = oa;
    }
    dst.write_region(b, &d);
}

/// The relative form (`scaleX`, `scaleY` in percent, `rotate` in degrees clockwise, `dx`, `dy`
/// in pixels), as Photoshop records a Free Transform: scale and rotate about the frame's
/// centre, then move. `None` when none of them is given.
fn relative_matrix(p: &Value, rect: [f64; 4]) -> Result<Option<Homography>> {
    let num = |k: &str| -> Result<Option<f64>> {
        match p.get(k) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => v.as_f64().filter(|f| f.is_finite()).map(Some).ok_or_else(|| bad(format!("`{k}` must be a finite number"))),
        }
    };
    let (sx, sy, rot, dx, dy) = (num("scaleX")?, num("scaleY")?, num("rotate")?, num("dx")?, num("dy")?);
    if sx.is_none() && sy.is_none() && rot.is_none() && dx.is_none() && dy.is_none() {
        return Ok(None);
    }
    let (sx, sy) = (sx.unwrap_or(100.0) / 100.0, sy.unwrap_or(100.0) / 100.0);
    if sx.abs() < 1e-6 || sy.abs() < 1e-6 {
        return Err(bad("`scaleX` and `scaleY` must not be 0"));
    }
    let (cx, cy) = ((rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0);
    let (sin, cos) = rot.unwrap_or(0.0).to_radians().sin_cos();
    // (x, y) → translate(-c) → scale → rotate → translate(c + d).
    let (a, b, c, d) = (sx * cos, sx * sin, -sy * sin, sy * cos);
    let e = cx + dx.unwrap_or(0.0) - a * cx - c * cy;
    let f = cy + dy.unwrap_or(0.0) - b * cx - d * cy;
    Ok(Some(Homography([a, c, e, b, d, f, 0.0, 0.0, 1.0])))
}

fn quad_param(p: &Value) -> Option<[[f64; 2]; 4]> {
    let a = p.get("quad")?.as_array()?;
    if a.len() != 4 {
        return None;
    }
    let mut q = [[0.0; 2]; 4];
    for (i, v) in a.iter().enumerate() {
        let c = v.as_array()?;
        q[i] = [c.first()?.as_f64()?, c.get(1)?.as_f64()?];
    }
    Some(q)
}

/// `p`, with the Channels panel's targeted alpha channel or Quick Mask as its `"target"` when it
/// names neither a target nor a layer (#2835).
fn with_panel_target(s: &Session, p: &Value) -> Value {
    let mut p = if p.is_object() { p.clone() } else { json!({}) };
    if p.get("target").is_none()
        && p.get("layer").is_none()
        && let (Some(m), Some(Value::Object(t))) = (p.as_object_mut(), crate::channel_cmds::targeted_channel(s))
    {
        m.extend(t);
    }
    p
}

fn transform(s: &mut Session, p: &Value) -> Result<Value> {
    let p = &with_panel_target(s, p);
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let id = crate::commands::explicit_layer(s, p)?.or(st.active_layer);
    let lone = lone_target(&st.doc, id, p)?;
    let id = match (id, lone.is_some() && crate::channel_cmds::is_channel_target(p)) {
        (_, true) => None,
        (Some(id), false) => Some(id),
        (None, false) => return Err(EngineError::Other("no active layer".into())),
    };
    let rect = match p.get("rect").and_then(Value::as_array) {
        Some(r) if r.len() == 4 => {
            let v: Vec<f64> = r.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect();
            [v[0], v[1], v[2], v[3]]
        }
        _ => {
            let b = match (lone, id) {
                (Some(surf), _) => target_bounds(&st.doc, surf),
                (None, Some(id)) => transform_bounds(&st.doc, st.doc.layer(id).ok_or(EngineError::NoLayer(id))?),
                (None, None) => Rect::EMPTY,
            };
            if b.is_empty() {
                return Err(EngineError::Other("nothing to transform".into()));
            }
            [b.x0 as f64, b.y0 as f64, b.x1 as f64, b.y1 as f64]
        }
    };
    let h = if let Some(q) = quad_param(p) {
        Homography::rect_to_quad(rect, q).ok_or_else(|| bad("degenerate quad"))?
    } else if let Some(m) = p.get("matrix").and_then(Value::as_array) {
        let v: Vec<f64> = m.iter().filter_map(Value::as_f64).collect();
        if v.len() != 6 {
            return Err(bad("matrix must be [a, b, c, d, e, f]"));
        }
        // Affine [a b c d e f] maps (x, y) → (a·x + c·y + e, b·x + d·y + f).
        Homography([v[0], v[2], v[4], v[1], v[3], v[5], 0.0, 0.0, 1.0])
    } else if let Some(m) = relative_matrix(p, rect)? {
        m
    } else {
        return Err(bad("pass `quad` (where the rect's corners go), `matrix`, or `scaleX`/`scaleY`/`rotate`/`dx`/`dy`"));
    };
    if h.inverse().is_none() {
        return Err(bad("the transform collapses the layer"));
    }
    let m = h.0;
    let affine =
        (m[6].abs() < 1e-12 && m[7].abs() < 1e-12).then(|| Affine { m: [m[0] / m[8], m[3] / m[8], m[1] / m[8], m[4] / m[8], m[2] / m[8], m[5] / m[8]] });
    let interp = Interp::parse(p.get("interpolation").and_then(Value::as_str).unwrap_or("bicubic"));
    let lone = lone.is_some();
    let p = p.clone();
    // A background job when started with `Session::start` (#1015): the warp of a large layer runs
    // on a worker with progress, and can be cancelled.
    crate::jobs::edit_job(
        s,
        "Free Transform",
        move |doc, _, ctx| ctx.stage(0.0, 1.0, "Transforming", |ctl| transform_doc(doc, id, &p, lone, &h, affine, interp, ctl)),
        move |()| json!({"rect": rect}),
    )
}

/// The edit of [`transform`]: the layer (or a lone mask or channel) and the selection outline.
#[allow(clippy::too_many_arguments)]
fn transform_doc(
    doc: &mut Document,
    id: Option<LayerId>,
    p: &Value,
    lone: bool,
    h: &Homography,
    affine: Option<Affine>,
    interp: Interp,
    ctl: &Interrupt,
) -> Result<()> {
    let sel = doc.selection.clone();
    // The selection outline moves too: price it before anything is built.
    if let Some(sel) = sel.as_ref().filter(|_| !h.is_identity()) {
        check_budget(sel.content_bounds(), h, "selection")?;
    }
    if lone {
        // A targeted unlinked mask, alpha channel or Quick Mask transforms by itself.
        if !h.is_identity() {
            let (surf, _) = crate::channel_cmds::target_surface(doc, id, p)?;
            let src = sel.as_ref().map_or_else(|| surf.content_bounds(), Surface::content_bounds);
            check_budget(src, h, "mask or channel")?;
            *surf = match &sel {
                Some(sel) => warp_gray_selected(surf, sel, h, interp)?,
                None => warp_gray(surf, h, interp)?,
            };
            if let Some(sel) = &doc.selection {
                doc.selection = Some(warp_gray(sel, h, Interp::Bilinear)?).filter(|s| !s.content_bounds().is_empty());
            }
        }
        return Ok(());
    }
    let id = id.ok_or_else(|| EngineError::Other("no active layer".into()))?;
    let is_group = doc.layer(id).is_some_and(Layer::is_group);
    let group = group_locks(doc, id);
    // Photoshop turns the Background into a normal layer before transforming it.
    crate::extra_cmds::background_to_layer(doc, id);
    let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
    transform_layer_with(if is_group { None } else { sel.as_ref() }, group, l, h, affine, interp, ctl)?;
    if !h.is_identity() {
        // Type layers re-render from their new transform.
        let snapshot = doc.clone();
        if let Some(l) = doc.layer_mut(id) {
            refresh_text(&snapshot, l);
        }
        // The selection outline moves with the pixels.
        if let Some(sel) = &doc.selection {
            doc.selection = Some(warp_gray(sel, h, Interp::Bilinear)?).filter(|s| !s.content_bounds().is_empty());
        }
    }
    Ok(())
}

pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: "edit.transform",
        label: "Free Transform",
        menu: &[],
        shortcut: None,
        params: r##"{"layer":id?,"rect":[x0,y0,x1,y1]? (source frame; default = layer content ∩ selection),"quad":[[x,y]×4]? (where the frame's corners go, clockwise from top-left),"matrix":[a,b,c,d,e,f]? (affine alternative),"scaleX":%?,"scaleY":%?,"rotate":deg?,"dx":px?,"dy":px? (relative alternative: scale and rotate about the frame's centre, then move, as a Photoshop action records a Free Transform),"interpolation":"bicubic|bilinear|nearest"="bicubic","target":"pixels"|"mask"|"quickMask"|{"channel":i}=Channels panel target (an unlinked mask, an alpha channel or the Quick Mask transforms alone; a linked mask moves with its layer)}"##,
        enabled: has_layer,
        journal: true,
        run: transform,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> Session {
        session_depth(8)
    }

    fn session_depth(depth: u32) -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 100, "height": 100, "depth": depth})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        s.edit("paint", |doc, active| {
            let l = doc.layer_mut(active.unwrap()).unwrap();
            l.surface_mut().unwrap().fill_rect(Rect::new(10, 10, 30, 20), &[1.0, 0.0, 0.0, 1.0]);
            Ok(())
        })
        .unwrap();
        s
    }

    fn active_bounds(s: &Session) -> Rect {
        let st = s.active().unwrap();
        st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().content_bounds()
    }

    // ---- Pixel budget (#3773). Tests run with a 2²⁰ px budget, so nothing here comes near the real one. ----

    /// A 16 x 16 document whose active layer is a 1 x 8 column at x = 10.
    fn column() -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 16, "height": 16})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        s.edit("paint", |doc, active| {
            doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap().fill_rect(Rect::new(10, 2, 11, 10), &[1.0, 0.0, 0.0, 1.0]);
            Ok(())
        })
        .unwrap();
        s
    }

    /// Stretch x by `k` about x = 10 (the column's left edge).
    fn stretch(k: f64) -> Value {
        json!({"matrix": [k, 0, 0, 1, -10.0 * k, 0], "interpolation": "nearest"})
    }

    #[test]
    fn a_scale_past_the_pixel_budget_is_refused_and_changes_nothing() {
        let mut s = column();
        let st = s.active().unwrap();
        let (before, revision, steps) = (st.doc.clone(), st.revision, st.history.past_len());
        // 200000 x 10 px of output: twice the test budget.
        let err = s.execute("edit.transform", stretch(2e5)).unwrap_err().to_string();
        assert!(err.starts_with("not enough memory for transforming pixels"), "{err}");
        assert!(err.contains("would be 200002 x 10 px") && err.ends_with("the document was not changed"), "{err}");
        let st = s.active().unwrap();
        assert!(st.doc == before, "the document is untouched");
        assert_eq!((st.revision, st.history.past_len()), (revision, steps));
        // Within the budget the same stretch still works, nothing clipped.
        s.execute("edit.transform", stretch(1e4)).unwrap();
        assert_eq!(active_bounds(&s), Rect::new(0, 2, 10_000, 10));
    }

    #[test]
    fn a_group_with_one_layer_past_the_budget_is_not_half_transformed() {
        let mut s = column();
        let wide = s.active().unwrap().active_layer.unwrap();
        // Over the column: a 1 px dot, which fits the budget stretched by the same amount.
        s.execute("layer.new.layer", json!({})).unwrap();
        let dot = s.active().unwrap().active_layer.unwrap();
        s.edit("paint", |doc, _| {
            doc.layer_mut(dot).unwrap().surface_mut().unwrap().fill_rect(Rect::new(10, 12, 11, 13), &[0.0, 1.0, 0.0, 1.0]);
            // Widen the column to 8 px (x 2..10), so it alone needs 400002 x 10 px.
            doc.layer_mut(wide).unwrap().surface_mut().unwrap().fill_rect(Rect::new(2, 2, 10, 10), &[1.0, 0.0, 0.0, 1.0]);
            Ok(())
        })
        .unwrap();
        s.execute("layer.select", json!({"layer": wide.0})).unwrap();
        s.execute("layer.select", json!({"layer": dot.0, "mode": "add"})).unwrap();
        let group = LayerId(s.execute("layer.groupLayers", json!({})).unwrap()["layer"].as_u64().unwrap());
        let st = s.active().unwrap();
        assert!(matches!(&st.doc.layer(group).unwrap().content, LayerContent::Group(g) if g.children.len() == 2));
        let (before, revision) = (st.doc.clone(), st.revision);
        let k = 5e4;
        assert!(s.execute("edit.transform", json!({"layer": group.0, "matrix": [k, 0, 0, 1, -10.0 * k, 0]})).is_err());
        let st = s.active().unwrap();
        assert!(st.doc == before && st.revision == revision, "nothing changed");
        // The layer-level check runs before any child is warped, so even the layer itself
        // (not just the edit's copy of the document) is left as it was.
        let mut l = st.doc.layer(group).unwrap().clone();
        let h = Homography([k, 0.0, -10.0 * k, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert!(transform_layer(None, Locks::default(), &mut l, &h, None, Interp::Nearest).is_err());
        assert!(&l == st.doc.layer(group).unwrap(), "no child was transformed");
    }

    #[test]
    fn moving_a_small_layer_far_off_the_canvas_is_within_budget() {
        let mut s = session();
        s.execute("edit.transform", json!({"matrix": [1, 0, 0, 1, 900_000, -900_000]})).unwrap();
        assert_eq!(active_bounds(&s), Rect::new(900_010, -899_990, 900_030, -899_980));
        s.execute("edit.transform", json!({"matrix": [1, 0, 0, 1, -900_000, 900_000]})).unwrap();
        assert_eq!(active_bounds(&s), Rect::new(10, 10, 30, 20), "and back, every pixel kept");
    }

    #[test]
    fn selections_masks_and_channels_have_the_same_budget() {
        // The selection outline of a layer transform.
        let mut s = column();
        s.execute("select.rect", json!({"x": 10, "y": 2, "width": 1, "height": 8})).unwrap();
        let before = s.active().unwrap().doc.clone();
        let err = s.execute("edit.transform", stretch(2e5)).unwrap_err().to_string();
        assert!(err.contains("transformed selection would be"), "{err}");
        // Transform Selection.
        let err = s.execute("select.transformSelection", stretch(2e5)).unwrap_err().to_string();
        assert!(err.contains("transformed selection would be"), "{err}");
        assert!(s.active().unwrap().doc == before);
        // A targeted alpha channel.
        let mut s = column();
        s.execute("channel.new", json!({})).unwrap();
        s.edit("paint channel", |doc, _| {
            doc.channels[0].surface.fill_rect(Rect::new(10, 2, 11, 10), &[1.0]);
            Ok(())
        })
        .unwrap();
        let before = s.active().unwrap().doc.clone();
        let mut p = stretch(2e5);
        p["target"] = json!({"channel": 0});
        let err = s.execute("edit.transform", p).unwrap_err().to_string();
        assert!(err.contains("transformed mask or channel would be"), "{err}");
        assert!(s.active().unwrap().doc == before);
    }

    #[test]
    fn scale_via_quad_and_undo() {
        let mut s = session();
        s.execute("edit.transform", json!({"quad": [[10, 10], [50, 10], [50, 30], [10, 30]]})).unwrap();
        let b = active_bounds(&s);
        assert!(b.width().abs_diff(40) <= 2 && b.height().abs_diff(20) <= 2, "{b:?}");
        s.undo();
        assert_eq!(active_bounds(&s), Rect::new(10, 10, 30, 20));
    }

    #[test]
    fn affine_matrix_and_perspective() {
        let mut s = session();
        // Translate by (40, 50).
        s.execute("edit.transform", json!({"matrix": [1, 0, 0, 1, 40, 50]})).unwrap();
        assert_eq!(active_bounds(&s), Rect::new(50, 60, 70, 70));
        // Perspective: top edge narrower than the bottom.
        s.execute("edit.transform", json!({"quad": [[55, 60], [65, 60], [75, 70], [45, 70]]})).unwrap();
        let b = active_bounds(&s);
        assert!(b.x0 <= 46 && b.x1 >= 74, "{b:?}");
        assert!(s.execute("edit.transform", json!({"quad": [[0, 0], [0, 0], [0, 0], [0, 0]]})).is_err());
    }

    #[test]
    fn background_becomes_layer_0() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 40, "height": 40})).unwrap();
        s.execute("edit.transform", json!({"quad": [[0, 0], [20, 0], [20, 20], [0, 20]]})).unwrap();
        let st = s.active().unwrap();
        let l = &st.doc.layers[0];
        assert_eq!(l.name, "Layer 0");
        assert!(l.surface().unwrap().format().alpha);
        assert_eq!(l.surface().unwrap().pixel(30, 30)[3], 0.0, "revealed area is transparent");
    }

    /// `locked_for_transform` (what Free Transform asks before its box opens, #2585) agrees with
    /// what `edit.transform` refuses at commit: a position lock, its own or a group's; never the
    /// Background's (it becomes Layer 0).
    #[test]
    fn locked_for_transform_matches_the_commit() {
        let quad = json!({"quad": [[10, 10], [50, 10], [50, 30], [10, 30]]});
        let lock = |s: &mut Session, id: LayerId, on: bool| {
            s.edit("lock", |doc, _| {
                doc.layer_mut(id).unwrap().locks.position = on;
                Ok(())
            })
            .unwrap();
        };
        let mut s = session();
        let layer = s.active().unwrap().active_layer.unwrap();
        lock(&mut s, layer, true);
        assert!(locked_for_transform(&s.active().unwrap().doc, layer));
        assert!(s.execute("edit.transform", quad.clone()).is_err());
        lock(&mut s, layer, false);
        let group = LayerId(s.execute("layer.groupLayers", json!({"layer": layer.0})).unwrap()["layer"].as_u64().unwrap());
        lock(&mut s, group, true);
        s.execute("layer.select", json!({"layer": layer.0})).unwrap();
        assert!(locked_for_transform(&s.active().unwrap().doc, layer), "the group's lock reaches the layer");
        assert!(s.execute("edit.transform", quad.clone()).is_err());
        let bg = s.active().unwrap().doc.layers[0].id;
        assert!(!locked_for_transform(&s.active().unwrap().doc, bg), "the Background transforms");
    }

    #[test]
    fn with_selection_only_selected_pixels_move() {
        let mut s = session();
        s.execute("select.rect", json!({"x": 10, "y": 10, "width": 10, "height": 10})).unwrap();
        s.execute("edit.transform", json!({"matrix": [1, 0, 0, 1, 0, 50]})).unwrap();
        let st = s.active().unwrap();
        let surf = st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap();
        assert_eq!(surf.pixel(15, 15)[3], 0.0, "lifted area is cleared");
        assert_eq!(surf.pixel(25, 15)[3], 1.0, "unselected part stays");
        assert_eq!(surf.pixel(15, 65)[3], 1.0, "moved pixels land");
        let sel = st.doc.selection.as_ref().unwrap().content_bounds();
        assert_eq!((sel.y0, sel.y1), (60, 70), "selection moves too");
    }

    #[test]
    fn with_selection_bounds_drop_the_vacated_area() {
        // #1097: moving every pixel through a selection left transparent but coloured pixels
        // behind, so the layer bounds were the union of the old and new positions.
        let mut s = session();
        s.execute("select.rect", json!({"x": 10, "y": 10, "width": 20, "height": 10})).unwrap();
        s.execute("edit.transform", json!({"matrix": [1, 0, 0, 1, 50, 0]})).unwrap();
        assert_eq!(active_bounds(&s), Rect::new(60, 10, 80, 20));
    }

    #[test]
    fn type_layers_transform_as_vectors() {
        let mut s = session();
        let r = s.execute("type.create", json!({"x": 10, "y": 50, "text": "Hi", "size": 20})).unwrap();
        let id = r["layer"].as_u64().unwrap();
        let w0 = active_bounds(&s).width();
        s.execute("edit.transform", json!({"layer": id, "matrix": [2, 0, 0, 2, -10, -50]})).unwrap();
        let w1 = active_bounds(&s).width();
        assert!(w1 as f32 > w0 as f32 * 1.8, "{w0} → {w1}");
        let st = s.active().unwrap();
        let LayerContent::Text(t) = &st.doc.layer(LayerId(id)).unwrap().content else { panic!() };
        assert!((t.transform.m[0] - 2.0).abs() < 1e-9);
        // Perspective on type is refused, like Photoshop.
        assert!(s.execute("edit.transform", json!({"layer": id, "quad": [[0, 0], [10, 0], [12, 10], [-2, 10]]})).is_err());
    }

    #[test]
    fn shapes_move_and_transform_as_vectors() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 200, "height": 200})).unwrap();
        s.execute("shape.create", json!({"kind": "rect", "rect": [10, 10, 40, 20], "fill": "#ff0000"})).unwrap();
        s.execute("layer.translate", json!({"dx": 30, "dy": 5})).unwrap();
        assert_eq!(active_bounds(&s), Rect::new(40, 15, 80, 35));
        s.execute("edit.transform", json!({"matrix": [2, 0, 0, 2, -40, -15]})).unwrap();
        assert_eq!(active_bounds(&s), Rect::new(40, 15, 120, 55));
        // Distort maps the path and keeps a live shape (#3009): the rectangle's corners go to the quad.
        let quad = [[40.0, 15.0], [120.0, 15.0], [130.0, 55.0], [30.0, 55.0]];
        s.execute("edit.transform", json!({ "quad": quad })).unwrap();
        let st = s.active().unwrap();
        let LayerContent::Shape(sh) = &st.doc.layer(st.active_layer.unwrap()).unwrap().content else { panic!("still a shape layer") };
        let mut anchors: Vec<[f64; 2]> = sh.path.subpaths[0].knots.iter().map(|k| [k.anchor.x, k.anchor.y]).collect();
        let mut want = quad.to_vec();
        anchors.sort_by(|a, b| a.partial_cmp(b).unwrap());
        want.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(anchors.iter().zip(&want).all(|(a, b)| (a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6), "{anchors:?}");
        assert!(sh.live.is_none(), "the live rectangle is now a path");
        assert_eq!(active_bounds(&s), Rect::new(30, 15, 130, 55));
        // Warp bends the path too: the arc adds anchors along the curved edges.
        s.execute("edit.transform.warp", json!({"style": "arc", "bend": 50})).unwrap();
        let st = s.active().unwrap();
        let LayerContent::Shape(sh) = &st.doc.layer(st.active_layer.unwrap()).unwrap().content else { panic!("still a shape layer") };
        assert!(sh.path.subpaths[0].knots.len() > 4, "{}", sh.path.subpaths[0].knots.len());
        assert!(!active_bounds(&s).is_empty());
    }

    #[test]
    fn type_moves_with_the_move_command() {
        let mut s = session();
        s.execute("type.create", json!({"x": 10, "y": 50, "text": "Hi", "size": 20})).unwrap();
        let b0 = active_bounds(&s);
        s.execute("layer.translate", json!({"dx": 20, "dy": 10})).unwrap();
        let b1 = active_bounds(&s);
        assert_eq!((b1.x0 - b0.x0, b1.y0 - b0.y0), (20, 10));
    }

    // ---- Layer masks and targets (#205) ----

    /// `session()` plus a reveal-all mask with `hide` painted black (linked unless `linked` is false).
    fn masked(hide: &[Rect], linked: bool) -> Session {
        masked_depth(hide, linked, 8)
    }

    fn masked_depth(hide: &[Rect], linked: bool, depth: u32) -> Session {
        let mut s = session_depth(depth);
        s.execute("layer.layerMask.revealAll", json!({})).unwrap();
        s.edit("mask", |doc, active| {
            let m = doc.layer_mut(active.unwrap()).unwrap().mask.as_mut().unwrap();
            for r in hide {
                m.surface.fill_rect(*r, &[0.0]);
            }
            m.linked = linked;
            Ok(())
        })
        .unwrap();
        s
    }

    fn mask_at(s: &Session, x: i32, y: i32) -> f32 {
        let st = s.active().unwrap();
        st.doc.layer(st.active_layer.unwrap()).unwrap().mask.as_ref().unwrap().surface.sample_channel(x, y, 0)
    }

    fn alpha_at(s: &Session, x: i32, y: i32) -> f32 {
        let st = s.active().unwrap();
        st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().pixel(x, y)[3]
    }

    #[test]
    fn linked_raster_mask_moves_with_the_pixels() {
        let mut s = masked(&[Rect::new(10, 10, 20, 20)], true);
        s.execute("edit.transform", json!({"matrix": [1, 0, 0, 1, 40, 50]})).unwrap();
        assert_eq!(active_bounds(&s), Rect::new(50, 60, 70, 70));
        assert!(mask_at(&s, 55, 65) < 0.01, "the hidden half moved with the pixels");
        assert!(mask_at(&s, 65, 65) > 0.99);
        assert!(mask_at(&s, 15, 15) > 0.99, "nothing hidden left behind");
    }

    #[test]
    fn linked_mask_with_a_selection_moves_only_the_selected_region() {
        // The selection covers the left half of the red rect; the mask hides its left quarter
        // (selected) and its right quarter (not selected).
        let mut s = masked(&[Rect::new(10, 10, 15, 20), Rect::new(25, 10, 30, 20)], true);
        s.execute("select.rect", json!({"x": 10, "y": 10, "width": 10, "height": 10})).unwrap();
        s.execute("edit.transform", json!({"matrix": [1, 0, 0, 1, 0, 50]})).unwrap();
        // Moved pixels and their mask line up.
        assert_eq!(alpha_at(&s, 12, 65), 1.0);
        assert!(mask_at(&s, 12, 65) < 0.01, "selected hidden quarter moved");
        assert!(mask_at(&s, 17, 65) > 0.99, "selected visible quarter moved");
        // The vacated area reads as revealed (default), with transparent pixels.
        assert_eq!(alpha_at(&s, 12, 15), 0.0);
        assert!(mask_at(&s, 12, 15) > 0.99);
        // The unselected part of the mask stays, aligned with the unselected pixels.
        assert_eq!(alpha_at(&s, 27, 15), 1.0);
        assert!(mask_at(&s, 27, 15) < 0.01, "unselected mask stays put");
        assert!(mask_at(&s, 27, 65) > 0.99, "and isn't copied along");
        // Every pixel's mask value is where it was relative to the pixel.
        for (x, y) in [(11, 61), (14, 69), (15, 61), (19, 69)] {
            let want = if x < 15 { 0.0 } else { 1.0 };
            assert!((mask_at(&s, x, y) - want).abs() < 0.01, "({x},{y})");
        }
    }

    #[test]
    fn unlinked_mask_stays_put() {
        let mut s = masked(&[Rect::new(10, 10, 20, 20)], false);
        s.execute("edit.transform", json!({"matrix": [1, 0, 0, 1, 40, 50]})).unwrap();
        assert_eq!(active_bounds(&s), Rect::new(50, 60, 70, 70));
        assert!(mask_at(&s, 15, 15) < 0.01);
        assert!(mask_at(&s, 55, 65) > 0.99);
        // With a selection too.
        let mut s = masked(&[Rect::new(10, 10, 20, 20)], false);
        s.execute("select.rect", json!({"x": 10, "y": 10, "width": 10, "height": 10})).unwrap();
        s.execute("edit.transform", json!({"matrix": [1, 0, 0, 1, 0, 50]})).unwrap();
        assert!(mask_at(&s, 15, 15) < 0.01 && mask_at(&s, 15, 65) > 0.99);
    }

    #[test]
    fn targeted_unlinked_mask_transforms_alone() {
        let mut s = masked(&[Rect::new(10, 10, 20, 20)], false);
        s.execute("edit.transform", json!({"matrix": [1, 0, 0, 1, 40, 50], "target": "mask"})).unwrap();
        assert_eq!(active_bounds(&s), Rect::new(10, 10, 30, 20), "pixels untouched");
        assert!(mask_at(&s, 55, 65) < 0.01, "mask moved");
        assert!(mask_at(&s, 15, 15) > 0.99, "vacated area reads as the default");
        s.undo();
        assert!(mask_at(&s, 15, 15) < 0.01, "one undo step");
        // With a selection, only its part of the mask.
        s.execute("select.rect", json!({"x": 10, "y": 10, "width": 5, "height": 10})).unwrap();
        s.execute("edit.transform", json!({"matrix": [1, 0, 0, 1, 0, 50], "target": "mask"})).unwrap();
        assert!(mask_at(&s, 12, 65) < 0.01 && mask_at(&s, 12, 15) > 0.99);
        assert!(mask_at(&s, 17, 15) < 0.01, "unselected mask stays");
        assert_eq!(active_bounds(&s), Rect::new(10, 10, 30, 20), "pixels untouched");
    }

    #[test]
    fn targeted_linked_mask_moves_with_its_layer() {
        // Photoshop: a linked mask can't move without its layer.
        let mut s = masked(&[Rect::new(10, 10, 20, 20)], true);
        s.execute("edit.transform", json!({"matrix": [1, 0, 0, 1, 40, 50], "target": "mask"})).unwrap();
        assert_eq!(active_bounds(&s), Rect::new(50, 60, 70, 70));
        assert!(mask_at(&s, 55, 65) < 0.01);
    }

    #[test]
    fn targeted_alpha_channel_transforms_alone() {
        let mut s = session();
        s.execute("channel.new", json!({})).unwrap();
        s.edit("paint channel", |doc, _| {
            doc.channels[0].surface.fill_rect(Rect::new(10, 10, 20, 20), &[1.0]);
            Ok(())
        })
        .unwrap();
        s.execute("edit.transform", json!({"matrix": [1, 0, 0, 1, 40, 50], "target": {"channel": 0}})).unwrap();
        let st = s.active().unwrap();
        let ch = &st.doc.channels[0].surface;
        assert!(ch.sample_channel(55, 65, 0) > 0.99 && ch.sample_channel(15, 15, 0) < 0.01);
        assert_eq!(active_bounds(&s), Rect::new(10, 10, 30, 20), "layer untouched");
    }

    /// The Channels panel targets an alpha channel: Edit › Transform › Flip Horizontal and a
    /// 50% `edit.transform` (no `target` param) work on the channel, not the layer (BUG-207-4).
    #[test]
    fn transforms_follow_the_targeted_alpha_channel() {
        let mut s = session();
        s.execute("channel.new", json!({})).unwrap();
        s.edit("paint channel", |doc, _| {
            let ch = &mut doc.channels[0].surface;
            ch.fill_rect(Rect::new(10, 10, 20, 20), &[1.0]);
            ch.fill_rect(Rect::new(10, 10, 12, 20), &[0.5]);
            Ok(())
        })
        .unwrap();
        s.execute("channel.target", json!({"channel": 0})).unwrap();
        let layer = |s: &Session| crate::active_layer_of(s).unwrap().surface().unwrap().read_region(Rect::new(0, 0, 100, 100));
        let (pixels, past) = (layer(&s), s.active().unwrap().history.past_len());
        s.execute("edit.transform.flipHorizontal", json!({})).unwrap();
        // 8-bit channel: 0.5 is stored as 128/255.
        let ch = |s: &Session, x, y| (s.active().unwrap().doc.channels[0].surface.sample_channel(x, y, 0) * 10.0).round() / 10.0;
        assert_eq!((ch(&s, 19, 15), ch(&s, 10, 15)), (0.5, 1.0), "the channel flips within its frame");
        assert!(layer(&s) == pixels, "the layer is untouched");
        assert_eq!(s.active().unwrap().history.past_len(), past + 1, "one undo step");
        s.execute("edit.transform", json!({"matrix": [0.5, 0, 0, 0.5, 0, 0], "interpolation": "nearest"})).unwrap();
        assert_eq!((ch(&s, 7, 7), ch(&s, 15, 15)), (1.0, 0.0), "the channel scales to 50%");
        assert!(layer(&s) == pixels, "the layer is untouched");
        s.undo();
        s.undo();
        assert_eq!((ch(&s, 10, 15), ch(&s, 19, 15)), (0.5, 1.0));
    }

    #[test]
    fn transform_targets_fail_gracefully() {
        let mut s = session();
        let m = json!([1, 0, 0, 1, 5, 5]);
        assert!(s.execute("edit.transform", json!({"matrix": m, "target": "mask"})).is_err(), "no mask");
        assert!(s.execute("edit.transform", json!({"matrix": m, "target": {"channel": 9}})).is_err());
        assert!(s.execute("edit.transform", json!({"matrix": m, "target": "quickMask"})).is_err());
        // Unknown target shapes fall back to the pixels.
        s.execute("edit.transform", json!({"matrix": m, "target": 7})).unwrap();
        s.execute("edit.transform", json!({"matrix": m, "target": {"channel": "x"}})).unwrap();
        // Selections that miss the mask entirely transform nothing, without panicking.
        let mut s = masked(&[Rect::new(10, 10, 20, 20)], false);
        s.execute("select.rect", json!({"x": 80, "y": 80, "width": 5, "height": 5})).unwrap();
        let _ = s.execute("edit.transform", json!({"matrix": [1e9, 0, 0, 1e9, 0, 0], "target": "mask"}));
        let _ = s.execute("edit.transform", json!({"matrix": [1, 0, 0, 1, 1e12, 0], "target": "mask"}));
    }

    #[test]
    fn split_gray_selected_rejects_non_gray() {
        let s = Surface::new(PixelFormat::RGBA8);
        let mut sel = Surface::new(PixelFormat::GRAY8);
        sel.fill_rect(Rect::new(0, 0, 4, 4), &[1.0]);
        assert!(matches!(split_gray_selected(&s, &sel), Ok(None)));
    }

    /// #2630: type can't be partly transformed, so a selection doesn't narrow the frame: Free
    /// Transform (and the frame-based presets) act on the whole type layer.
    #[test]
    fn a_selection_does_not_narrow_a_type_layers_frame() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 200, "height": 120})).unwrap();
        s.execute("type.create", json!({"x": 20, "y": 60, "text": "Hello", "size": 36})).unwrap();
        let whole = active_bounds(&s);
        s.execute("select.rect", json!({"x": whole.x0, "y": whole.y0, "width": whole.width() / 2, "height": whole.height()})).unwrap();
        let st = s.active().unwrap();
        let l = st.doc.layer(st.active_layer.unwrap()).unwrap();
        assert_eq!(transform_bounds(&st.doc, l), whole, "the frame is the whole type layer");
        // Corners of the frame moved 10 px right: a pure translation of the whole text.
        let q = [[whole.x0 + 10, whole.y0], [whole.x1 + 10, whole.y0], [whole.x1 + 10, whole.y1], [whole.x0 + 10, whole.y1]];
        s.execute("edit.transform", json!({"quad": q})).unwrap();
        let moved = active_bounds(&s);
        assert!(moved.x0.abs_diff(whole.x0 + 10) <= 1 && moved.width().abs_diff(whole.width()) <= 1, "{whole:?} -> {moved:?}");
        // Rotate 180° turns it about its own centre, not the selection's.
        s.execute("edit.transform.rotate180", json!({})).unwrap();
        let turned = active_bounds(&s);
        assert!(turned.x0.abs_diff(moved.x0) <= 2 && turned.x1.abs_diff(moved.x1) <= 2, "{moved:?} -> {turned:?}");
        // A pixel layer still frames only the selected pixels.
        let mut s = session();
        s.execute("select.rect", json!({"x": 10, "y": 10, "width": 5, "height": 10})).unwrap();
        let st = s.active().unwrap();
        assert_eq!(transform_bounds(&st.doc, st.doc.layer(st.active_layer.unwrap()).unwrap()), Rect::new(10, 10, 15, 20));
    }

    #[test]
    fn group_transform_ignores_selections_inside_its_children() {
        group_transform_with_selection(Rect::new(10, 10, 15, 15));
    }

    #[test]
    fn group_transform_ignores_selections_outside_its_children() {
        group_transform_with_selection(Rect::new(70, 70, 80, 80));
    }

    #[test]
    fn group_transform_presets_rotate_about_the_whole_group() {
        for depth in [8, 16, 32] {
            for x in [10, 70] {
                let mut s = masked_depth(&[Rect::new(10, 10, 15, 20)], true, depth);
                let child = s.active().unwrap().active_layer.unwrap();
                s.execute("layer.groupLayers", json!({})).unwrap();
                s.execute("select.rect", json!({"x": x, "y": 10, "width": 5, "height": 5})).unwrap();
                s.execute("edit.transform.rotate180", json!({})).unwrap();
                let layer = s.active().unwrap().doc.layer(child).unwrap();
                assert_eq!(layer.surface().unwrap().content_bounds(), Rect::new(10, 10, 30, 20), "rotate about the group's centre, not the selection's");
                let mask = &layer.mask.as_ref().unwrap().surface;
                assert_eq!(mask.sample_channel(27, 15, 0), 0.0);
                assert_eq!(mask.sample_channel(12, 15, 0), 1.0);
            }
        }
    }

    fn group_transform_with_selection(selection: Rect) {
        for depth in [8, 16, 32] {
            for nested in [false, true] {
                let mut s = masked_depth(&[Rect::new(10, 10, 15, 20)], true, depth);
                let child = s.active().unwrap().active_layer.unwrap();
                s.execute("layer.groupLayers", json!({})).unwrap();
                if nested {
                    s.execute("layer.groupLayers", json!({})).unwrap();
                }
                s.execute("select.rect", json!({"x": selection.x0, "y": selection.y0, "width": selection.width(), "height": selection.height()})).unwrap();
                let steps = s.active().unwrap().history.entries().len();
                let result = s.execute("edit.transform", json!({"quad": [[10, 10], [50, 10], [50, 30], [10, 30]], "interpolation": "nearest"})).unwrap();
                assert_eq!(result["rect"], json!([10.0, 10.0, 30.0, 20.0]), "frame must include the whole group (nested={nested})");
                let st = s.active().unwrap();
                let layer = st.doc.layer(child).unwrap();
                assert_eq!(layer.surface().unwrap().content_bounds(), Rect::new(10, 10, 50, 30));
                let mask = &layer.mask.as_ref().unwrap().surface;
                assert_eq!(mask.sample_channel(19, 20, 0), 0.0, "the linked mask scales with its complete layer");
                assert_eq!(mask.sample_channel(21, 20, 0), 1.0);
                assert_eq!(st.history.entries().len(), steps + 1);
                assert!(s.undo());
                let st = s.active().unwrap();
                let layer = st.doc.layer(child).unwrap();
                assert_eq!(layer.surface().unwrap().content_bounds(), Rect::new(10, 10, 30, 20));
                assert_eq!(layer.mask.as_ref().unwrap().surface.sample_channel(19, 15, 0), 1.0);
                assert_eq!(st.doc.selection.as_ref().unwrap().content_bounds(), selection);
                assert!(s.redo());
                assert_eq!(s.active().unwrap().doc.layer(child).unwrap().surface().unwrap().content_bounds(), Rect::new(10, 10, 50, 30));
            }
        }
    }

    #[test]
    fn feathered_selection_identity_transform_preserves_pixels() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 16, "height": 16, "background": "transparent"})).unwrap();
        s.execute("edit.fill", json!({"color": "#ff0000"})).unwrap();
        s.execute("select.rect", json!({"x": 4, "y": 4, "width": 8, "height": 8})).unwrap();
        s.execute("select.modify.feather", json!({"radius": 1})).unwrap();

        let mut before_px = Vec::new();
        for y in 0..16 {
            for x in 0..16 {
                before_px.push(s.execute("document.pixel", json!({"x": x, "y": y})).unwrap());
            }
        }

        s.execute("edit.transform", json!({"matrix": [1, 0, 0, 1, 0, 0], "interpolation": "nearest"})).unwrap();

        let mut after_px = Vec::new();
        for y in 0..16 {
            for x in 0..16 {
                after_px.push(s.execute("document.pixel", json!({"x": x, "y": y})).unwrap());
            }
        }

        assert_eq!(before_px, after_px, "feathered selection identity transform must preserve all pixels");
        let p4_8 = s.execute("document.pixel", json!({"x": 4, "y": 8})).unwrap();
        assert_eq!(p4_8, json!([1.0, 0.0, 0.0, 1.0]), "(4,8) must remain opaque red");

        // Undo restores original document.
        s.execute("edit.undo", json!({})).unwrap();
        let undo_p4_8 = s.execute("document.pixel", json!({"x": 4, "y": 8})).unwrap();
        assert_eq!(undo_p4_8, json!([1.0, 0.0, 0.0, 1.0]));

        // Locked layer still refuses transform even if identity.
        s.execute("layer.setProps", json!({"locks": {"position": true}})).unwrap();
        assert!(s.execute("edit.transform", json!({"matrix": [1, 0, 0, 1, 0, 0]})).is_err(), "locked layer refuses identity transform");
    }

    #[test]
    fn relative_params_scale_and_rotate_about_the_frame_centre() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 100, "height": 100})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        s.execute("select.rect", json!({"x": 20, "y": 30, "width": 40, "height": 20})).unwrap();
        s.execute("edit.fill", json!({"contents": "black"})).unwrap();
        s.execute("select.deselect", json!({})).unwrap();
        // 50% about the centre (40, 40), then 10 px right: the content ends at x 40..60, y 35..45.
        let r = s.execute("edit.transform", json!({"scaleX": 50, "scaleY": 50, "dx": 10})).unwrap();
        let rect: Vec<f64> = serde_json::from_value(r["rect"].clone()).unwrap();
        assert_eq!(rect, [20.0, 30.0, 60.0, 50.0], "the reported frame is the source frame");
        let bounds = |s: &Session| {
            let d = s.active().unwrap();
            let b = transform_bounds(&d.doc, d.doc.layer(d.active_layer.unwrap()).unwrap());
            (b.x0, b.y0, b.x1, b.y1)
        };
        assert_eq!(bounds(&s), (40, 35, 60, 45));
        // A quarter turn about the centre (50, 40) swaps the frame's sides.
        s.execute("edit.transform", json!({"rotate": 90})).unwrap();
        assert_eq!(bounds(&s), (45, 30, 55, 50));
        for bad in [json!({"scaleX": 0}), json!({"rotate": "x"}), json!({"dx": f64::NAN})] {
            assert!(s.execute("edit.transform", bad.clone()).is_err(), "{bad}");
        }
    }
}
