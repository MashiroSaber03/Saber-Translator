//! Layer menu remnants: Layer Mask › Apply / From Transparency / Hide Selection, Mask All
//! Objects, Matting, Smart Objects › Stack Mode and Reveal in Finder, Layer Style › Blending
//! Options / Global Light / Create Layer / Scale Effects, Layer Content Options, and exporting
//! just the active layer (Quick Export as PNG, Export As).

use photocraft_algo::selection::Region;
use photocraft_color::{BlendMode, ColorMode};
use photocraft_doc::{BlendIf, BlendRange, Document, Effect, Effects, Knockout, Layer, LayerContent, LayerId, LayerMask, SmartSource, StackMode};
use photocraft_geom::Rect;
use photocraft_raster::{Surface, from_rgba_into, to_rgba};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, layer_param};
use crate::{EngineError, Result, Session};

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

fn other(msg: impl Into<String>) -> EngineError {
    EngineError::Other(msg.into())
}

fn num(p: &Value, key: &str, default: f32) -> f32 {
    p.get(key).and_then(Value::as_f64).map_or(default, |v| v as f32)
}

type Enabled = std::result::Result<(), String>;

fn active_layer(s: &Session) -> std::result::Result<&Layer, String> {
    let d = s.active().ok_or("no document open")?;
    d.active_layer.and_then(|id| d.doc.layer(id)).ok_or_else(|| "no active layer".into())
}

fn has_layer(s: &Session) -> Enabled {
    active_layer(s).map(|_| ())
}

fn has_raster(s: &Session) -> Enabled {
    let l = active_layer(s)?;
    if matches!(l.content, LayerContent::Raster(_)) {
        Ok(())
    } else {
        Err(format!("active layer is {} {} layer, not a pixel layer", l.content.article(), l.content.kind_name()))
    }
}

fn has_raster_with_mask(s: &Session) -> Enabled {
    has_raster(s)?;
    if active_layer(s)?.mask.is_some() { Ok(()) } else { Err("the layer has no layer mask".into()) }
}

fn has_selection_layer(s: &Session) -> Enabled {
    has_layer(s)?;
    if s.active().is_some_and(|d| d.doc.selection.is_some()) { Ok(()) } else { Err("no selection".into()) }
}

fn has_doc(s: &Session) -> Enabled {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

fn has_smart(s: &Session) -> Enabled {
    if matches!(active_layer(s)?.content, LayerContent::Smart(_)) { Ok(()) } else { Err("the active layer is not a smart object".into()) }
}

fn has_linked_smart(s: &Session) -> Enabled {
    match &active_layer(s)?.content {
        LayerContent::Smart(sm) if matches!(sm.source, SmartSource::Linked { .. }) => Ok(()),
        _ => Err("the active layer is not a linked smart object".into()),
    }
}

fn has_effects(s: &Session) -> Enabled {
    let l = active_layer(s)?;
    if l.effects.items.iter().any(Effect::enabled) { Ok(()) } else { Err("the layer has no layer effects".into()) }
}

fn has_content_options(s: &Session) -> Enabled {
    let l = active_layer(s)?;
    match l.content {
        LayerContent::Adjustment(_) | LayerContent::Fill(_) | LayerContent::Text(_) | LayerContent::Smart(_) | LayerContent::Shape(_) => Ok(()),
        _ => Err(format!("{} layers have no content options", l.content.kind_name())),
    }
}

// ---------- RGBA helpers ----------

/// Straight RGBA of `surf` over `r` (row-major).
fn read_rgba(surf: &Surface, r: Rect) -> Vec<[f32; 4]> {
    let fmt = surf.format();
    let n = fmt.channels();
    surf.read_region(r).chunks_exact(n).map(|q| to_rgba(&fmt, q)).collect()
}

fn write_rgba(surf: &mut Surface, r: Rect, px: &[[f32; 4]]) {
    let fmt = surf.format();
    let mut out = Vec::with_capacity(px.len() * fmt.channels());
    let mut enc = [0.0f32; 8];
    for q in px {
        let m = from_rgba_into(&fmt, *q, &mut enc);
        out.extend_from_slice(&enc[..m]);
    }
    surf.write_region(r, &out);
}

/// One history step editing the active raster layer's RGBA (content bounds grown by `grow`).
fn edit_rgba(s: &mut Session, label: &str, grow: i32, f: impl FnOnce(&mut Vec<[f32; 4]>, Rect)) -> Result<Value> {
    let id = layer_param(s, &Value::Null)?;
    s.edit(label, |doc, _| {
        let bounds = doc.bounds();
        let surf = crate::commands::paint_surface(doc, id, &Value::Null)?;
        let r = surf.content_bounds().inflate(grow).intersect(&bounds.inflate(grow));
        if r.is_empty() {
            return Ok(Value::Null);
        }
        let mut px = read_rgba(surf, r);
        f(&mut px, r);
        write_rgba(surf, r, &px);
        surf.prune();
        Ok(Value::Null)
    })
}

// ---------- layer masks ----------

/// Layer › Layer Mask › Apply: multiply the pixels' alpha by the mask, then delete the mask.
fn apply_mask(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, p)?;
    s.edit("Apply Layer Mask", |doc, _| {
        // Applying a feathered mask must bake the same softened coverage the compositor
        // displays. LayerMask::values_into accounts for density but not feather (#3044).
        let feathered = {
            let layer = doc.layer(id).ok_or(EngineError::NoLayer(id))?;
            if layer.mask.as_ref().is_some_and(|m| m.enabled && photocraft_compose::masks::feather_sigma(m.feather) > 0.0) {
                // Do not bake an independent vector mask that remains attached after Apply.
                let mut mask_only = layer.clone();
                mask_only.vector_mask = None;
                // Cover off-canvas pixels too: outside its area the combined mask is only the
                // mask's default value, which would hide moved-off content (#3043).
                let area = layer.surface().map_or(doc.bounds(), |surf| doc.bounds().union(&surf.content_bounds()));
                photocraft_compose::masks::combined_mask(&mask_only, area)
            } else {
                None
            }
        };
        let mask = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?.mask.take().ok_or_else(|| other("the layer has no layer mask"))?;
        let surf = crate::commands::paint_surface(doc, id, &Value::Null)?;
        let r = surf.content_bounds();
        if !r.is_empty() {
            let mut px = read_rgba(surf, r);
            let mut m = Vec::new();
            if let Some(ref softened) = feathered {
                softened.read_region_into(r, &mut m);
            } else {
                mask.values_into(r, &mut m);
            }
            for (q, k) in px.iter_mut().zip(&m) {
                q[3] *= k;
            }
            write_rgba(surf, r, &px);
            surf.prune();
        }
        Ok(Value::Null)
    })
}

/// Layer › Layer Mask › From Transparency: the alpha becomes a layer mask and the pixels opaque.
fn mask_from_transparency(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, p)?;
    s.edit("From Transparency", |doc, _| {
        let surf = crate::commands::paint_surface(doc, id, &Value::Null)?;
        // Masks are attached to layer coordinates, including pixels temporarily off-canvas.
        // Clipping to document bounds loses their alpha permanently when moved back (#3043).
        let r = surf.content_bounds();
        let mut mask = LayerMask::hide_all();
        if !r.is_empty() {
            let mut px = read_rgba(surf, r);
            let alpha: Vec<f32> = px.iter().map(|q| q[3]).collect();
            mask.surface.write_region(r, &alpha);
            mask.surface.prune();
            for q in &mut px {
                q[3] = 1.0;
            }
            write_rgba(surf, r, &px);
        }
        doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?.mask = Some(mask);
        Ok(Value::Null)
    })
}

/// Layer › Layer Mask › Hide Selection: a mask that hides the selected area.
fn hide_selection(s: &mut Session, p: &Value) -> Result<Value> {
    let ids = crate::layer_multi_cmds::mask_targets(s, p)?;
    s.edit("Add Layer Mask", |doc, _| {
        let sel = doc.selection.clone().ok_or_else(|| other("no selection"))?;
        let bounds = doc.bounds();
        let mut v = sel.read_region(bounds);
        for x in &mut v {
            *x = 1.0 - *x;
        }
        let mut mask = LayerMask::reveal_all();
        mask.surface.write_region(bounds, &v);
        mask.surface.prune();
        for id in ids {
            crate::extra_cmds::background_to_layer_for_mask(doc, id);
            doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?.mask = Some(mask.clone());
        }
        doc.selection = None;
        Ok(Value::Null)
    })
}

/// Layer › Mask All Objects: detects the main objects (the Select › Subject segmentation over the
/// layer, or the composite for non-pixel layers) and masks the layer to them. Photoshop makes one
/// group per object; here the objects share one mask.
fn mask_all_objects(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, p)?;
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let doc = d.doc.clone();
    let layer = doc.layer(id).ok_or(EngineError::NoLayer(id))?;
    let region: Option<Region> = match layer.surface() {
        Some(surf) if matches!(layer.content, LayerContent::Raster(_)) => {
            photocraft_algo::segment::subject::select_subject(&photocraft_algo::segment::SurfaceSampler(surf), doc.bounds())
        }
        _ => {
            struct Composite<'a>(&'a Document);
            impl photocraft_algo::segment::Sampler for Composite<'_> {
                fn rgba(&self, r: Rect) -> Vec<[f32; 4]> {
                    photocraft_compose::render(self.0, r).px
                }
            }
            photocraft_algo::segment::subject::select_subject(&Composite(&doc), doc.bounds())
        }
    };
    let region = region.ok_or_else(|| other("no objects were found"))?;
    let bbox = region.bbox;
    s.edit("Mask All Objects", |doc, _| {
        let mut mask = LayerMask::hide_all();
        let v: Vec<f32> = region.mask.iter().map(|m| f32::from(*m) / 255.0).collect();
        mask.surface.write_region(bbox, &v);
        mask.surface.prune();
        doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?.mask = Some(mask);
        Ok(())
    })?;
    Ok(json!({"bounds": [bbox.x0, bbox.y0, bbox.width(), bbox.height()]}))
}

// ---------- matting ----------

/// Layer › Matting › Defringe: edge pixels (within `width` of transparency) take the colour of
/// the nearest pixels further inside, grown outward one pixel at a time. Alpha is unchanged.
pub fn defringe(px: &mut [[f32; 4]], w: usize, h: usize, width: usize) {
    let outside: Vec<bool> = px.iter().map(|q| q[3] <= 0.01).collect();
    if !outside.iter().any(|o| *o) {
        return;
    }
    let dist = photocraft_algo::selection::edt(&outside, w, h);
    let mut known: Vec<bool> = (0..px.len()).map(|i| !outside[i] && dist[i] > width as f32).collect();
    let mut todo: Vec<usize> = (0..px.len()).filter(|&i| !outside[i] && !known[i]).collect();
    while !todo.is_empty() {
        let mut next = Vec::new();
        let mut updates = Vec::new();
        for &i in &todo {
            let (x, y) = ((i % w) as i32, (i / w) as i32);
            let mut acc = [0.0f32; 3];
            let mut n = 0.0;
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)] {
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                    continue;
                }
                let j = ny as usize * w + nx as usize;
                if known[j] {
                    for c in 0..3 {
                        acc[c] += px[j][c];
                    }
                    n += 1.0;
                }
            }
            if n > 0.0 {
                updates.push((i, acc.map(|v| v / n)));
            } else {
                next.push(i);
            }
        }
        if updates.is_empty() {
            break; // isolated fringe with no core colour nearby
        }
        for (i, c) in updates {
            px[i][..3].copy_from_slice(&c);
            known[i] = true;
        }
        todo = next;
    }
}

fn matting_defringe(s: &mut Session, p: &Value) -> Result<Value> {
    let width = num(p, "width", 1.0).clamp(1.0, 200.0).round() as usize;
    edit_rgba(s, "Defringe", 1, |px, r| defringe(px, r.width() as usize, r.height() as usize, width))
}

/// Remove Black / White Matte: un-premultiply colours that were composited over black or white.
fn remove_matte(s: &mut Session, white: bool) -> Result<Value> {
    edit_rgba(s, if white { "Remove White Matte" } else { "Remove Black Matte" }, 0, |px, _| {
        for q in px.iter_mut().filter(|q| q[3] > 0.0 && q[3] < 1.0) {
            let a = q[3];
            for v in &mut q[..3] {
                *v = if white { (*v - (1.0 - a)) / a } else { *v / a }.clamp(0.0, 1.0);
            }
        }
    })
}

/// Layer › Matting › Color Decontaminate: fringe colours move toward nearby opaque colours
/// (the Select and Mask decontamination, driven by the layer's own alpha).
fn color_decontaminate(s: &mut Session, p: &Value) -> Result<Value> {
    let amount = num(p, "amount", 100.0).clamp(0.0, 100.0);
    let radius = num(p, "radius", 4.0).clamp(1.0, 100.0);
    let id = layer_param(s, &Value::Null)?;
    s.edit("Color Decontaminate", |doc, _| {
        let surf = crate::commands::paint_surface(doc, id, &Value::Null)?;
        let r = surf.content_bounds();
        if r.is_empty() || !surf.format().alpha {
            return Ok(());
        }
        let alpha: Vec<u8> = read_rgba(surf, r).iter().map(|q| (q[3].clamp(0.0, 1.0) * 255.0).round() as u8).collect();
        let region = Region { bbox: r, mask: alpha };
        *surf = photocraft_algo::matting::decontaminate(surf, &region, radius, amount);
        Ok(())
    })?;
    Ok(Value::Null)
}

// ---------- smart object stack modes ----------

fn stack_mode(s: &mut Session, p: &Value, mode: Option<StackMode>) -> Result<Value> {
    let id = layer_param(s, p)?;
    let label = mode.map_or("None", StackMode::label);
    s.edit(&format!("Stack Mode: {label}"), |doc, _| {
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        let LayerContent::Smart(sm) = &mut l.content else {
            return Err(other("the active layer is not a smart object"));
        };
        sm.stack_mode = mode;
        crate::smart_cmds::refresh(doc, id)?;
        Ok(())
    })?;
    Ok(json!({"stackMode": mode.map(StackMode::id)}))
}

macro_rules! stack_spec {
    ($id:literal, $label:literal, $mode:expr) => {
        CommandSpec {
            id: concat!("layer.smartObjects.stackMode.", $id),
            label: $label,
            menu: &["Layer", "Smart Objects", "Stack Mode"],
            shortcut: None,
            params: r##"{"layer":id?}"##,
            enabled: has_smart,
            run: |s, p| stack_mode(s, p, $mode),
            journal: true,
        }
    };
}

/// Native-only: show a linked smart object's file in the platform file manager.
fn reveal_in_finder(s: &mut Session, p: &Value) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let l = d.active_layer.and_then(|id| d.doc.layer(id)).ok_or_else(|| other("no active layer"))?;
    let LayerContent::Smart(sm) = &l.content else {
        return Err(other("the active layer is not a smart object"));
    };
    let SmartSource::Linked { path } = &sm.source else {
        return Err(other("embedded smart objects have no file to reveal"));
    };
    reveal(path, p.get("dryRun").and_then(Value::as_bool).unwrap_or(false))
}

/// Show `path` in the platform file manager ([`reveal_command`] + [`spawn`]); `dry_run` answers
/// the program and arguments instead (tests). Shared by the smart-object command and the document
/// tab's Reveal in Finder (UI-217-6).
pub(crate) fn reveal(path: &str, dry_run: bool) -> Result<Value> {
    let (program, args) = reveal_command(path);
    if dry_run {
        return Ok(json!({"program": program, "args": args}));
    }
    spawn(program, &args)?;
    Ok(json!({"path": path}))
}

/// The file-manager invocation that selects `path` on this platform.
pub fn reveal_command(path: &str) -> (&'static str, Vec<String>) {
    if cfg!(target_os = "macos") {
        ("open", vec!["-R".into(), path.into()])
    } else if cfg!(target_os = "windows") {
        ("explorer", vec![format!("/select,{path}")])
    } else {
        let parent = std::path::Path::new(path).parent().map_or_else(|| ".".to_string(), |p| p.to_string_lossy().into_owned());
        ("xdg-open", vec![parent])
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn spawn(program: &str, args: &[String]) -> Result<()> {
    std::process::Command::new(program).args(args).spawn().map(|_| ()).map_err(|e| other(format!("{program}: {e}")))
}

#[cfg(target_arch = "wasm32")]
fn spawn(_: &str, _: &[String]) -> Result<()> {
    Err(other("Reveal in Finder needs the desktop app"))
}

fn native(_: &Session) -> Enabled {
    if cfg!(target_arch = "wasm32") { Err("not available in the browser".into()) } else { Ok(()) }
}

// ---------- layer style ----------

fn blending_options(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "layer.layerStyle.blendingOptions";
    let id = layer_param(s, p)?;
    let blend = match p.get("blend").and_then(Value::as_str) {
        Some(b) => Some(crate::commands::blend_from_str(b).ok_or_else(|| bad(CMD, format!("unknown blend mode `{b}`")))?),
        None => None,
    };
    let opacity = p.get("opacity").and_then(Value::as_f64).map(|v| (v as f32 / 100.0).clamp(0.0, 1.0));
    let fill = p.get("fillOpacity").and_then(Value::as_f64).map(|v| (v as f32 / 100.0).clamp(0.0, 1.0));
    let mode = s.active().ok_or(EngineError::NoDocument)?.doc.mode;
    let blend_if = match p.get("blendIf") {
        None => None,
        Some(v) => Some(blend_if_entries(v, mode)?),
    };
    let adv = advanced_params(p, mode)?;
    s.edit("Blending Options", |doc, _| {
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        if let Some(b) = blend {
            if b == BlendMode::PassThrough && !l.is_group() {
                return Err(bad(CMD, "Pass Through is for groups"));
            }
            l.blend = b;
        }
        if let Some(o) = opacity {
            l.opacity = o;
        }
        if let Some(f) = fill {
            l.fill_opacity = f;
        }
        match &blend_if {
            // `null`: Blend If back to the defaults.
            Some(None) => l.blend_if = BlendIf::default(),
            Some(Some(entries)) => {
                for (i, this, under) in entries {
                    let [cur_this, cur_under] = l.blend_if.get(*i);
                    l.blend_if.set(*i, [this.unwrap_or(cur_this), under.unwrap_or(cur_under)]);
                }
            }
            None => {}
        }
        adv.apply(l);
        Ok(())
    })?;
    Ok(Value::Null)
}

/// The Advanced Blending parameters of `layer.layerStyle.blendingOptions`; `None` = keep.
#[derive(Default)]
struct AdvancedParams {
    knockout: Option<Knockout>,
    blend_interior: Option<bool>,
    blend_clipped: Option<bool>,
    transparency_shapes: Option<bool>,
    layer_mask_hides: Option<bool>,
    vector_mask_hides: Option<bool>,
    /// Excluded-channel bit mask (from `channels`).
    excluded: Option<u32>,
}

impl AdvancedParams {
    fn apply(&self, l: &mut Layer) {
        let a = &mut l.advanced;
        if let Some(k) = self.knockout {
            a.knockout = k;
        }
        let set = |dst: &mut bool, v: Option<bool>| {
            if let Some(v) = v {
                *dst = v;
            }
        };
        set(&mut a.blend_interior, self.blend_interior);
        set(&mut a.blend_clipped, self.blend_clipped);
        set(&mut a.transparency_shapes, self.transparency_shapes);
        set(&mut a.layer_mask_hides_effects, self.layer_mask_hides);
        set(&mut a.vector_mask_hides_effects, self.vector_mask_hides);
        if let Some(x) = self.excluded {
            l.excluded_channels = x;
        }
    }
}

/// Parse and validate the Advanced Blending keys (wrong types and unknown values are errors).
fn advanced_params(p: &Value, mode: ColorMode) -> Result<AdvancedParams> {
    const CMD: &str = "layer.layerStyle.blendingOptions";
    let flag = |key: &str| -> Result<Option<bool>> {
        match p.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Bool(b)) => Ok(Some(*b)),
            Some(v) => Err(bad(CMD, format!("{key}: expected true or false (got {v})"))),
        }
    };
    let knockout = match p.get("knockout") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(Knockout::from_name(s).ok_or_else(|| bad(CMD, format!("knockout: expected none, shallow or deep (got `{s}`)")))?),
        Some(v) => return Err(bad(CMD, format!("knockout: expected \"none\", \"shallow\" or \"deep\" (got {v})"))),
    };
    let excluded = match p.get("channels") {
        None | Some(Value::Null) => None,
        Some(Value::Array(a)) => {
            let n = mode.color_channels().max(1);
            if a.len() > n {
                return Err(bad(CMD, format!("channels: a {mode:?} document has {n} colour channel(s), got {}", a.len())));
            }
            let mut mask = 0u32;
            for (i, v) in a.iter().enumerate() {
                match v {
                    Value::Bool(true) => {}
                    Value::Bool(false) => mask |= 1 << i,
                    other => return Err(bad(CMD, format!("channels[{i}]: expected true or false (got {other})"))),
                }
            }
            Some(mask)
        }
        Some(v) => return Err(bad(CMD, format!("channels: expected a list of true/false per colour channel (got {v})"))),
    };
    Ok(AdvancedParams {
        knockout,
        blend_interior: flag("blendInteriorEffectsAsGroup")?,
        blend_clipped: flag("blendClippedLayersAsGroup")?,
        transparency_shapes: flag("transparencyShapesLayer")?,
        layer_mask_hides: flag("layerMaskHidesEffects")?,
        vector_mask_hides: flag("vectorMaskHidesEffects")?,
        excluded,
    })
}

/// One `blendIf` entry: (range index, This Layer, Underlying Layer); `None` = keep.
type BlendIfEntry = (usize, Option<BlendRange>, Option<BlendRange>);

/// Parse `blendIf`: `null` (reset) or one or more `{"channel", "thisLayer"?, "underlying"?}`.
fn blend_if_entries(v: &Value, mode: ColorMode) -> Result<Option<Vec<BlendIfEntry>>> {
    const CMD: &str = "layer.layerStyle.blendingOptions";
    let items: Vec<&Value> = match v {
        Value::Null => return Ok(None),
        Value::Array(a) => a.iter().collect(),
        Value::Object(_) => vec![v],
        _ => {
            return Err(bad(CMD, "blendIf: expected an object, a list of objects or null"));
        }
    };
    let mut out = Vec::with_capacity(items.len());
    for it in items {
        let ch = it.get("channel").unwrap_or(&Value::Null);
        let i = blend_if_channel(ch, mode).ok_or_else(|| bad(CMD, format!("blendIf: unknown channel {ch} for a {mode:?} document")))?;
        let range = |key: &str| -> Result<Option<BlendRange>> {
            match it.get(key) {
                None => Ok(None),
                Some(r) => blend_range(r).map(Some).ok_or_else(|| {
                    bad(
                        CMD,
                        format!("blendIf.{key}: expected [black, white] or [blackLow, blackHigh, whiteLow, whiteHigh], 0..255, in increasing order (got {r})"),
                    )
                }),
            }
        };
        out.push((i, range("thisLayer")?, range("underlying")?));
    }
    Ok(Some(out))
}

/// Blend If channel → range index: 0 = Gray, then the mode's colour channels in order. A
/// grayscale document's Gray is its one channel (the PSD spec marks the composite entry
/// irrelevant there).
fn blend_if_channel(v: &Value, mode: ColorMode) -> Option<usize> {
    let n = mode.color_channels();
    if let Some(i) = v.as_u64() {
        return usize::try_from(i).ok().filter(|i| *i <= n);
    }
    let name = v.as_str().unwrap_or("gray").to_ascii_lowercase();
    let names: &[&str] = match mode {
        ColorMode::Rgb | ColorMode::Indexed => &["red", "green", "blue"],
        ColorMode::Cmyk => &["cyan", "magenta", "yellow", "black"],
        ColorMode::Lab => &["lightness", "a", "b"],
        _ => &[],
    };
    if name == "gray" || name == "grey" {
        return Some(usize::from(n == 1));
    }
    names.iter().position(|c| *c == name).map(|i| i + 1)
}

/// `[black, white]` (unsplit) or `[blackLow, blackHigh, whiteLow, whiteHigh]`, each 0..=255 and
/// in increasing order (the dialog's sliders can't cross).
fn blend_range(v: &Value) -> Option<BlendRange> {
    let a = v.as_array()?;
    let n: Vec<u8> = a.iter().map(|x| x.as_f64().filter(|f| (0.0..=255.0).contains(f)).map(|f| f.round() as u8)).collect::<Option<_>>()?;
    let q = match n.as_slice() {
        [b, w] => [*b, *b, *w, *w],
        [a, b, c, d] => [*a, *b, *c, *d],
        _ => return None,
    };
    q.windows(2).all(|w| w[0] <= w[1]).then(|| BlendRange::from_bytes(q))
}

fn global_light(s: &mut Session, p: &Value) -> Result<Value> {
    let cur = s.active().ok_or(EngineError::NoDocument)?.doc.global_light;
    // Read as f64 and reject what f32 can't hold: 1e39 would become inf, and rem_euclid NaN.
    let f32_param = |key: &str, default: f32| match p.get(key).and_then(Value::as_f64) {
        None => Ok(default),
        Some(v) if v.is_finite() && v.abs() <= f64::from(f32::MAX) => Ok(v as f32),
        Some(_) => Err(bad("layer.layerStyle.globalLight", format!("{key} must be a finite number"))),
    };
    let angle = f32_param("angle", cur.angle)?;
    let altitude = f32_param("altitude", cur.altitude)?.clamp(0.0, 90.0);
    let angle = (angle + 180.0).rem_euclid(360.0) - 180.0;
    s.edit("Global Light", |doc, _| {
        doc.global_light.angle = angle;
        doc.global_light.altitude = altitude;
        Ok(())
    })?;
    Ok(json!({"angle": angle, "altitude": altitude}))
}

fn scale_effects(s: &mut Session, p: &Value) -> Result<Value> {
    let k = num(p, "scale", 100.0).clamp(1.0, 1000.0) / 100.0;
    let id = layer_param(s, p)?;
    s.edit("Scale Effects", |doc, _| {
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        crate::image_cmds::scale_effects(&mut l.effects, k);
        l.effects.psd_raw = None;
        Ok(())
    })?;
    Ok(Value::Null)
}

/// Dragging a layer's fx onto another layer in the Layers panel: all its effects (or just the one
/// at `effect`) move there, or with `copy` are copied. Moving all effects replaces the target's
/// style, as pasting a layer style does; a single effect replaces the target's effect of the same
/// kind, else is added.
fn transfer_effects(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "layer.layerStyle.transferEffects";
    let id_of = |key: &str| p.get(key).and_then(Value::as_u64).map(LayerId).ok_or_else(|| bad(CMD, format!("`{key}` must be a layer id")));
    let from = id_of("from")?;
    let to = id_of("to")?;
    if from == to {
        return Err(bad(CMD, "the effects are already on that layer"));
    }
    let copy = p.get("copy").and_then(Value::as_bool).unwrap_or(false);
    let index = match p.get("effect") {
        None | Some(Value::Null) => None,
        Some(v) => Some(v.as_u64().and_then(|i| usize::try_from(i).ok()).ok_or_else(|| bad(CMD, "`effect` must be an effect index"))?),
    };
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let src = d.doc.layer(from).ok_or(EngineError::NoLayer(from))?;
    d.doc.layer(to).ok_or(EngineError::NoLayer(to))?;
    let payload = match index {
        Some(i) => {
            let e = src.effects.items.get(i).cloned().ok_or_else(|| bad(CMD, format!("the layer has no effect {i}")))?;
            Effects { enabled: true, items: vec![e], psd_raw: None, reference: None }
        }
        None => Effects { psd_raw: None, ..src.effects.clone() },
    };
    if payload.items.is_empty() {
        return Err(bad(CMD, "the layer has no layer effects"));
    }
    let label = if copy { "Copy Layer Effects" } else { "Move Layer Effects" };
    s.edit(label, |doc, _| {
        let dst = doc.layer_mut(to).ok_or(EngineError::NoLayer(to))?;
        dst.effects.psd_raw = None;
        if index.is_some() {
            dst.effects.enabled = true;
            for e in payload.items {
                let same = dst.effects.items.iter().position(|x| std::mem::discriminant(x) == std::mem::discriminant(&e));
                match same.and_then(|i| dst.effects.items.get_mut(i)) {
                    Some(slot) => *slot = e,
                    None => dst.effects.items.push(e),
                }
            }
        } else {
            dst.effects = payload;
        }
        if !copy {
            let src = doc.layer_mut(from).ok_or(EngineError::NoLayer(from))?;
            match index {
                Some(i) if i < src.effects.items.len() => {
                    src.effects.items.remove(i);
                    src.effects.psd_raw = None;
                }
                Some(_) => {}
                None => src.effects = Effects::default(),
            }
        }
        Ok(())
    })?;
    Ok(json!({"from": from.0, "to": to.0}))
}

/// Effects drawn beneath the layer (they become separate layers below it, unclipped).
fn is_below(e: &Effect) -> bool {
    match e {
        Effect::DropShadow(_) | Effect::OuterGlow(_) => true,
        Effect::Stroke(st) => st.position == photocraft_doc::StrokePosition::Outside,
        Effect::BevelEmboss(b) => {
            matches!(b.style, photocraft_doc::BevelStyle::OuterBevel | photocraft_doc::BevelStyle::Emboss | photocraft_doc::BevelStyle::PillowEmboss)
        }
        _ => false,
    }
}

fn effect_common_mut(e: &mut Effect) -> Option<&mut photocraft_doc::FxCommon> {
    match e {
        Effect::DropShadow(s) | Effect::InnerShadow(s) => Some(&mut s.common),
        Effect::OuterGlow(g) | Effect::InnerGlow(g) => Some(&mut g.common),
        Effect::Stroke(s) => Some(&mut s.common),
        Effect::ColorOverlay { common, .. } | Effect::GradientOverlay { common, .. } | Effect::PatternOverlay { common, .. } => Some(common),
        Effect::Satin(s) => Some(&mut s.common),
        Effect::BevelEmboss(_) => None,
    }
}

/// Layer › Layer Style › Create Layer: each enabled effect is rendered on its own (normal blend,
/// full opacity) into a pixel layer named like Photoshop's ("<layer>'s Drop Shadow"), carrying the
/// effect's blend mode and opacity. Shadows/glows/outer strokes go below the layer; inner effects
/// and overlays go above it, clipped to it. The layer keeps its pixels and loses its effects.
fn create_layer(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, p)?;
    let made = s.edit("Create Layers", |doc, active| {
        let canvas = doc.bounds();
        let fmt = doc.pixel_format();
        let l = doc.layer(id).ok_or(EngineError::NoLayer(id))?.clone();
        let effects: Vec<Effect> = l.effects.items.iter().filter(|e| e.enabled()).cloned().collect();
        if effects.is_empty() || !l.effects.enabled {
            return Err(other("the layer has no layer effects"));
        }
        let area = l.surface().map_or(canvas, |s| s.content_bounds().union(&canvas)).inflate(256);
        let light = doc.global_light;
        let mut below = Vec::new();
        let mut above = Vec::new();
        for e in effects {
            let mut alone = l.clone();
            alone.visible = true;
            alone.opacity = 1.0;
            alone.blend = BlendMode::Normal;
            alone.clipped = false;
            alone.mask = None;
            // Fill opacity 0 hides the layer's own pixels; effects still draw.
            alone.fill_opacity = 0.0;
            let mut e1 = e.clone();
            let (blend, opacity) = match effect_common_mut(&mut e1) {
                Some(c) => {
                    let r = (c.blend, c.opacity);
                    c.blend = BlendMode::Normal;
                    c.opacity = 1.0;
                    r
                }
                None => (BlendMode::Normal, 1.0),
            };
            alone.effects.items = vec![e1];
            alone.effects.psd_raw = None;
            let mut tmp = Document::new("fx", doc.size, doc.mode, doc.depth);
            tmp.global_light = light;
            tmp.patterns = doc.patterns.clone();
            tmp.layers = vec![alone];
            let buf = photocraft_compose::render(&tmp, area);
            let mut surf = Surface::new(fmt);
            let data: Vec<f32> = buf.px.iter().flat_map(|q| photocraft_raster::from_rgba(&fmt, *q)).collect();
            surf.write_region(area, &data);
            surf.prune();
            let mut nl = Layer::new(format!("{}'s {}", l.name, e.label()), LayerContent::Raster(surf));
            nl.blend = blend;
            nl.opacity = opacity;
            if is_below(&e) {
                below.push(nl);
            } else {
                nl.clipped = true;
                above.push(nl);
            }
        }
        let n = below.len() + above.len();
        // Insert in order: below layers under the source, clipped layers above it.
        let path = doc.path_of(id).ok_or(EngineError::NoLayer(id))?;
        let (&at, parent) = path.split_last().ok_or_else(|| other("bad layer path"))?;
        let sib =
            if parent.is_empty() { &mut doc.layers } else { doc.layer_at_mut(parent).and_then(|t| t.children_mut()).ok_or_else(|| other("bad layer path"))? };
        let src = &mut sib[at];
        src.effects.items.clear();
        src.effects.psd_raw = None;
        for (k, nl) in above.into_iter().enumerate() {
            sib.insert(at + 1 + k, nl);
        }
        for nl in below.into_iter().rev() {
            sib.insert(at, nl);
        }
        *active = Some(id);
        Ok(n)
    })?;
    Ok(json!({"layers": made}))
}

/// Layer › Layer Content Options…: which editor the active layer opens (UIs show it).
fn content_options(s: &mut Session, _: &Value) -> Result<Value> {
    let l = active_layer(s).map_err(other)?;
    let editor = match &l.content {
        LayerContent::Adjustment(a) => {
            format!("adjustment:{}", crate::commands::adjustment_kind(a))
        }
        LayerContent::Fill(_) => "fill".into(),
        LayerContent::Text(_) => "text".into(),
        LayerContent::Smart(_) => "smartObject".into(),
        LayerContent::Shape(_) => "shape".into(),
        other => {
            return Err(EngineError::Other(format!("{} layers have no content options", other.kind_name())));
        }
    };
    Ok(json!({"layer": l.id.0, "editor": editor}))
}

// ---------- export just the active layer ----------

/// The layer rendered alone (visible, its blend mode ignored) and trimmed to its pixels, as a
/// one-layer document — what Export As does for a layer.
pub fn layer_document(doc: &Document, id: LayerId) -> Result<Document> {
    let l = doc.layer(id).ok_or(EngineError::NoLayer(id))?;
    let mut alone = l.clone();
    alone.visible = true;
    alone.blend = if alone.is_group() { BlendMode::PassThrough } else { BlendMode::Normal };
    alone.clipped = false;
    let mut tmp = doc.clone();
    tmp.layers = vec![alone];
    tmp.selection = None;
    let area = doc.bounds();
    let buf = photocraft_compose::render(&tmp, area);
    let w = area.width() as usize;
    let mut b = Rect::EMPTY;
    for (i, q) in buf.px.iter().enumerate() {
        if q[3] > 0.0 {
            let (x, y) = (area.x0 + (i % w) as i32, area.y0 + (i / w) as i32);
            b = b.union(&Rect::new(x, y, x + 1, y + 1));
        }
    }
    if b.is_empty() {
        return Err(other("the layer is empty"));
    }
    let fmt = doc.pixel_format();
    let mut out = Document::new(l.name.clone(), photocraft_doc::Size::new(b.width(), b.height()), doc.mode, doc.depth);
    out.resolution_dpi = doc.resolution_dpi;
    out.icc_profile = doc.icc_profile.clone();
    let mut surf = Surface::new(fmt);
    let rows: Vec<f32> = (b.y0..b.y1)
        .flat_map(|y| {
            let row = (y - area.y0) as usize * w;
            let px = &buf.px;
            (b.x0..b.x1).flat_map(move |x| photocraft_raster::from_rgba(&fmt, px[row + (x - area.x0) as usize]))
        })
        .collect();
    surf.write_region(Rect::from_xywh(0, 0, b.width(), b.height()), &rows);
    out.layers.push(Layer::new(l.name.clone(), LayerContent::Raster(surf)));
    Ok(out)
}

fn export_layer(s: &mut Session, p: &Value, cmd: &str, png_only: bool) -> Result<Value> {
    let id = layer_param(s, p)?;
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let mut ldoc = layer_document(&d.doc, id)?;
    let path = p.get("path").and_then(Value::as_str).filter(|s| !s.is_empty()).ok_or_else(|| bad(cmd, "missing `path` (UIs ask for one)"))?;
    let lower = path.to_ascii_lowercase();
    if png_only && !lower.ends_with(".png") {
        return Err(bad(cmd, "Quick Export as PNG writes .png files"));
    }
    let scale = num(p, "scale", 100.0).clamp(1.0, 1000.0);
    if (scale - 100.0).abs() > 1e-3 {
        let mut tmp = Session::new();
        tmp.add_document(ldoc, None);
        let w = (tmp.active().map_or(1, |d| d.doc.size.width) as f32 * scale / 100.0).round().max(1.0);
        tmp.execute("image.imageSize", json!({"width": w}))?;
        ldoc = (*tmp.active().ok_or(EngineError::NoDocument)?.doc).clone();
    }
    let out = photocraft_io::export(&ldoc, path, &Default::default()).map_err(|e| other(e.to_string()))?;
    write_file(path, &out.bytes)?;
    Ok(json!({"path": path, "bytes": out.bytes.len(), "width": ldoc.size.width, "height": ldoc.size.height, "warnings": out.warnings}))
}

#[cfg(not(target_arch = "wasm32"))]
fn write_file(path: &str, bytes: &[u8]) -> Result<()> {
    crate::file_cmds::write_file(path, bytes)
}

#[cfg(target_arch = "wasm32")]
fn write_file(_: &str, _: &[u8]) -> Result<()> {
    Err(other("writing files needs the desktop app"))
}

macro_rules! spec {
    ($id:literal, $label:literal, [$($m:literal),*], $params:literal, $en:expr, $run:expr) => {
        CommandSpec { id: $id, label: $label, menu: &[$($m),*], shortcut: None, params: $params, enabled: $en, run: $run, journal: true }
    };
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        spec!("layer.layerMask.apply", "Apply", ["Layer", "Layer Mask"], r##"{"layer":id?}"##, has_raster_with_mask, apply_mask),
        spec!("layer.layerMask.fromTransparency", "From Transparency", ["Layer", "Layer Mask"], r##"{"layer":id?}"##, has_raster, mask_from_transparency),
        spec!(
            "layer.layerMask.hideSelection",
            "Hide Selection",
            ["Layer", "Layer Mask"],
            r##"{"layer":id?} (no layer: every selected layer that can take a mask)"##,
            has_selection_layer,
            hide_selection
        ),
        spec!("layer.maskAllObjects", "Mask All Objects", ["Layer"], r##"{"layer":id?}"##, has_layer, mask_all_objects),
        spec!("layer.matting.defringe", "Defringe…", ["Layer", "Matting"], r##"{"width":1..200=1}"##, has_raster, matting_defringe),
        spec!("layer.matting.removeBlackMatte", "Remove Black Matte", ["Layer", "Matting"], "{}", has_raster, |s, _| remove_matte(s, false)),
        spec!("layer.matting.removeWhiteMatte", "Remove White Matte", ["Layer", "Matting"], "{}", has_raster, |s, _| remove_matte(s, true)),
        spec!(
            "layer.matting.colorDecontaminate",
            "Color Decontaminate…",
            ["Layer", "Matting"],
            r##"{"amount":0..100=100,"radius":1..100=4}"##,
            has_raster,
            color_decontaminate
        ),
        stack_spec!("entropy", "Entropy", Some(StackMode::Entropy)),
        stack_spec!("kurtosis", "Kurtosis", Some(StackMode::Kurtosis)),
        stack_spec!("maximum", "Maximum", Some(StackMode::Maximum)),
        stack_spec!("mean", "Mean", Some(StackMode::Mean)),
        stack_spec!("median", "Median", Some(StackMode::Median)),
        stack_spec!("minimum", "Minimum", Some(StackMode::Minimum)),
        stack_spec!("range", "Range", Some(StackMode::Range)),
        stack_spec!("skewness", "Skewness", Some(StackMode::Skewness)),
        stack_spec!("standardDeviation", "Standard Deviation", Some(StackMode::StandardDeviation)),
        stack_spec!("summation", "Summation", Some(StackMode::Summation)),
        stack_spec!("variance", "Variance", Some(StackMode::Variance)),
        stack_spec!("none", "None", None),
        spec!(
            "layer.smartObjects.revealInFinder",
            "Reveal in Finder",
            ["Layer", "Smart Objects"],
            r##"{"dryRun":bool=false}"##,
            |s| native(s).and_then(|_| has_linked_smart(s)),
            reveal_in_finder
        ),
        spec!(
            "layer.layerStyle.blendingOptions",
            "Blending Options…",
            ["Layer", "Layer Style"],
            r##"{"layer":id?,"blend":"normal|multiply|…"?,"opacity":0..100?,"fillOpacity":0..100?,"blendIf":{"channel":"gray|red|green|blue|cyan|…"|index="gray","thisLayer":[black,white]|[blackLo,blackHi,whiteLo,whiteHi]?,"underlying":[…]?}|[{…},…]|null?,"channels":[bool,…]?,"knockout":"none|shallow|deep"?,"blendInteriorEffectsAsGroup":bool?,"blendClippedLayersAsGroup":bool?,"transparencyShapesLayer":bool?,"layerMaskHidesEffects":bool?,"vectorMaskHidesEffects":bool?} (Blend If values 0..255; split points fade; null resets. Advanced Blending: channels = which colour channels blend (R G B / C M Y K / L a b); PhotoCraft's defaults are knockout none, interior effects off, clipped layers as group on, transparency shapes on, masks hide effects off)"##,
            has_layer,
            blending_options
        ),
        spec!(
            "layer.layerStyle.globalLight",
            "Global Light…",
            ["Layer", "Layer Style"],
            r##"{"angle":-180..180=120,"altitude":0..90=30}"##,
            has_doc,
            global_light
        ),
        spec!("layer.layerStyle.createLayer", "Create Layer", ["Layer", "Layer Style"], r##"{"layer":id?}"##, has_effects, create_layer),
        spec!("layer.layerStyle.scaleEffects", "Scale Effects…", ["Layer", "Layer Style"], r##"{"scale":1..1000=100}"##, has_effects, scale_effects),
        spec!(
            "layer.layerStyle.transferEffects",
            "Move Layer Effects",
            [],
            r##"{"from":id,"to":id,"effect":index? (default: all effects),"copy":bool=false}"##,
            has_doc,
            transfer_effects
        ),
        CommandSpec {
            id: "layer.layerContentOptions",
            label: "Layer Content Options…",
            menu: &["Layer"],
            shortcut: None,
            params: "{}",
            enabled: has_content_options,
            run: content_options,
            journal: false,
        },
        spec!("layer.quickExportAsPng", "Quick Export as PNG", ["Layer"], r##"{"layer":id?,"path":text}"##, has_layer, |s, p| export_layer(
            s,
            p,
            "layer.quickExportAsPng",
            true
        )),
        spec!(
            "layer.exportAs",
            "Export As…",
            ["Layer"],
            r##"{"layer":id?,"path":text,"scale":1..1000=100} (format from the path's extension)"##,
            has_layer,
            |s, p| export_layer(s, p, "layer.exportAs", false)
        ),
    ]
}

#[cfg(test)]
mod tests;
