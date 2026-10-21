//! Planner: walks the layer tree into a linear list of GPU passes over abstract buffer slots.
//!
//! The structure mirrors `photocraft_compose::composite_stack` / `composite_layer` /
//! `composite_atop` one to one, so both backends share Photoshop semantics (pass-through vs
//! isolated groups, clipping, masks, opacity × fill, adjustments, layer effects). Every pass
//! reads slots and writes a fresh slot; slots are recycled as soon as nothing refers to them.
//!
//! Layer effects follow `photocraft_compose::effects::composite_with_effects`: the layer's
//! effect maps (shadow / glow / satin / bevel coverage, stroke bands) are built once per layer
//! state by [`crate::fx`] and sampled here; exterior effects paint into a copy of the backdrop,
//! the layer at fill opacity takes the interior effects, and the two merge with the layer's mode
//! (or the layer merges first and the effects paint over it: `effects::premerges_layer`) and
//! opacity. Effect passes are clipped to the layer's effect region and the result is copied
//! back into the backdrop in place, so a small text layer costs only its own pixels.

use photocraft_color::BlendMode;
use photocraft_compose::adjust::{self, Transfer};
use photocraft_compose::effects::has_effects;
use photocraft_doc::adjust::{ToneSpace, lut3d_len};
use photocraft_doc::{Adjustment, Document, Effect, Fill, FxPaint, GlobalLight, Glow, Gradient, Layer, LayerContent, LayerId, Pattern, StrokePosition};
use photocraft_geom::Rect;
use photocraft_raster::Surface;

use crate::bounds;

/// Index of a chunk-sized accumulator texture.
pub type Slot = u32;

/// Pipeline (fragment entry point) of a pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kernel {
    /// Transparent fill (no draw).
    Clear,
    Content,
    Mask,
    Blend,
    Atop,
    Adjust,
    AdjMix,
    Lerp,
    /// Effect chain start: the layer at fill opacity, or an opaque copy of a clipping base.
    FxInit,
    /// Paint an effect through a coverage map (or the layer's alpha) into A.
    FxPaint,
    /// Layer + interior effects over the exterior result, then layer opacity against the backdrop.
    FxMerge,
    /// Outside strokes: accumulate one stroke's share (`kind` 0, premultiplied colour) or its
    /// coverage (`kind` 1).
    FxStroke,
    /// Outside strokes: resolve the accumulated strokes over the exterior result.
    FxStrokeEnd,
    /// Copy slot A into `dst` (an existing slot) over the pass clip (no draw).
    CopyRect,
    /// Copy all of slot A into the fresh slot `dst` (no draw).
    CopyFull,
    // Effect-map kernels (single-channel targets over an effect region, see `fx`).
    MShift,
    MDilate,
    MBlur,
    MGlow,
    MFinish,
    MBevelH,
    MBevelShade,
    MStroke,
    MBevelTex,
}

impl Kernel {
    /// Fragment entry point in `compose.wgsl` (`None` for copies and clears).
    pub fn entry(self) -> Option<&'static str> {
        Some(match self {
            Kernel::Clear | Kernel::CopyRect | Kernel::CopyFull => return None,
            Kernel::Content => "fs_content",
            Kernel::Mask => "fs_mask",
            Kernel::Blend => "fs_blend",
            Kernel::Atop => "fs_atop",
            Kernel::Adjust => "fs_adjust",
            Kernel::AdjMix => "fs_adjmix",
            Kernel::Lerp => "fs_lerp",
            Kernel::FxInit => "fs_fxinit",
            Kernel::FxPaint => "fs_fxpaint",
            Kernel::FxMerge => "fs_fxmerge",
            Kernel::FxStroke => "fs_fxstroke",
            Kernel::FxStrokeEnd => "fs_fxstrokeend",
            Kernel::MShift => "fs_mshift",
            Kernel::MDilate => "fs_mdilate",
            Kernel::MBlur => "fs_mblur",
            Kernel::MGlow => "fs_mglow",
            Kernel::MFinish => "fs_mfinish",
            Kernel::MBevelH => "fs_mbevelh",
            Kernel::MBevelShade => "fs_mbevelshade",
            Kernel::MStroke => "fs_mstroke",
            Kernel::MBevelTex => "fs_mbeveltex",
        })
    }

    /// Kernels that write effect maps (R32F / R16F targets) rather than RGBA accumulators.
    pub fn is_map(self) -> bool {
        matches!(
            self,
            Kernel::MShift
                | Kernel::MDilate
                | Kernel::MBlur
                | Kernel::MGlow
                | Kernel::MFinish
                | Kernel::MBevelH
                | Kernel::MBevelShade
                | Kernel::MStroke
                | Kernel::MBevelTex
        )
    }

    /// Every kernel with a pipeline.
    pub const DRAWN: [Kernel; 21] = [
        Kernel::Content,
        Kernel::Mask,
        Kernel::Blend,
        Kernel::Atop,
        Kernel::Adjust,
        Kernel::AdjMix,
        Kernel::Lerp,
        Kernel::FxInit,
        Kernel::FxPaint,
        Kernel::FxMerge,
        Kernel::FxStroke,
        Kernel::FxStrokeEnd,
        Kernel::MShift,
        Kernel::MDilate,
        Kernel::MBlur,
        Kernel::MGlow,
        Kernel::MFinish,
        Kernel::MBevelH,
        Kernel::MBevelShade,
        Kernel::MStroke,
        Kernel::MBevelTex,
    ];
}

/// Which resident texture a pass samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role {
    Content,
    Mask,
    /// A stroked shape's stroke alone (its fill uses `Content`).
    Stroke,
}

/// A surface the pass needs on the GPU.
#[derive(Clone, Debug)]
pub struct TexUse<'a> {
    pub layer: LayerId,
    pub role: Role,
    pub surface: SurfaceRef<'a>,
}

/// An effect map sampled by a pass: map `map` of enabled effect `item` of `plan.fx[fx]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MapRef {
    pub fx: usize,
    pub item: usize,
    pub map: usize,
}

#[derive(Clone, Debug)]
pub struct Pass<'a> {
    pub kernel: Kernel,
    pub dst: Slot,
    pub a: Option<Slot>,
    pub b: Option<Slot>,
    pub c: Option<Slot>,
    /// A fourth slot, bound in place of `tex` (passes without layer pixels).
    pub d: Option<Slot>,
    pub mode: BlendMode,
    pub opacity: f32,
    /// The layer's Fill where it isn't folded into `opacity`: the final blend of a plain layer
    /// (`Kernel::Blend`), so the special eight can apply it inside the blend. 1 elsewhere.
    pub fill: f32,
    /// Layer pixels (raster / text / shape / smart cache / fill cache).
    pub tex: Option<TexUse<'a>>,
    /// Colour outside `tex` (the surface's default pixel), the solid fill colour, or the effect
    /// colour.
    pub color: [f32; 4],
    pub mask: Option<MaskUse<'a>>,
    /// Adjustment kind (see `adjust` in compose.wgsl) and parameters; the coverage kind of an
    /// effect paint.
    pub adjust_kind: i32,
    pub params: [[f32; 4]; 4],
    pub extra: [f32; 4],
    /// Extra shader flags (`F_KNOCKOUT`, …).
    pub flags: u32,
    /// 4096-entry LUT rows (Levels / Curves / Gradient map / gradient stops).
    pub lut: Option<Vec<[f32; 4096]>>,
    pub gradient: bool,
    /// Effect map sampled by `FxPaint`.
    pub map: Option<MapRef>,
    /// Pattern painted by `FxPaint`.
    pub pattern: Option<&'a Pattern>,
    /// Only pixels inside this rect (document coordinates) are computed; the pass is skipped for
    /// chunks it misses. Consumers of the slot must be clipped alike.
    pub clip: Option<Rect>,
}

/// Pixels a pass samples: a surface of the document, or one derived from it on the CPU (a
/// combined pixel × vector mask, a shape's fill or stroke alone).
#[derive(Clone, Debug)]
pub enum SurfaceRef<'a> {
    Doc(&'a Surface),
    Derived(std::sync::Arc<Surface>),
}

impl SurfaceRef<'_> {
    pub fn get(&self) -> &Surface {
        match self {
            SurfaceRef::Doc(s) => s,
            SurfaceRef::Derived(s) => s,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MaskUse<'a> {
    pub layer: LayerId,
    pub surface: SurfaceRef<'a>,
    pub density: f32,
    pub default: f32,
}

impl<'a> Pass<'a> {
    pub(crate) fn new(kernel: Kernel, dst: Slot) -> Self {
        Pass {
            kernel,
            dst,
            a: None,
            b: None,
            c: None,
            d: None,
            mode: BlendMode::Normal,
            opacity: 1.0,
            fill: 1.0,
            tex: None,
            color: [0.0; 4],
            mask: None,
            adjust_kind: 0,
            params: [[0.0; 4]; 4],
            extra: [0.0; 4],
            flags: 0,
            lut: None,
            gradient: false,
            map: None,
            pattern: None,
            clip: None,
        }
    }
}

/// Why a document can't be composited on the GPU (the caller falls back to the CPU compositor).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unsupported(pub String);

impl std::fmt::Display for Unsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GPU compositor: {}", self.0)
    }
}

impl std::error::Error for Unsupported {}

/// A layer whose effects the plan renders: its maps are built (or reused) before the chunks run.
#[derive(Clone, Copy, Debug)]
pub struct FxLayer<'a> {
    pub layer: &'a Layer,
    /// Region the maps cover (`compose::effect_maps`).
    pub region: Rect,
    /// Layer bounds (gradients and linked patterns are laid out in them).
    pub bounds: Rect,
}

#[derive(Debug)]
pub struct Plan<'a> {
    pub passes: Vec<Pass<'a>>,
    pub slots: u32,
    pub root: Slot,
    /// Effect layers in dependency order (a group's inner effect layers come first).
    pub fx: Vec<FxLayer<'a>>,
    /// Complete root stack units, where only the accumulated backdrop remains live.
    pub checkpoints: Vec<Checkpoint>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Checkpoint {
    /// First root layer included after occlusion pruning.
    pub layer_start: usize,
    /// Root layers below this boundary (including the complete clipping unit).
    pub layer_end: usize,
    /// Resume at this pass with `slot` restored.
    pub pass_end: usize,
    pub slot: Slot,
}

/// Document-level inputs of a plan.
#[derive(Clone, Copy)]
pub struct DocCtx<'a> {
    pub canvas: Rect,
    pub transfer: Transfer,
    pub light: GlobalLight,
    pub patterns: &'a [Pattern],
    /// Colour mode (channel restrictions name its channels).
    pub mode: photocraft_color::ColorMode,
    /// Sample depth (adjustment results are rounded to it).
    pub depth: photocraft_color::SampleType,
    /// The document (effect geometry the CPU computes, e.g. gradient stroke frames).
    pub doc: &'a Document,
}

impl<'a> DocCtx<'a> {
    pub fn of(doc: &'a Document) -> Self {
        DocCtx {
            canvas: doc.bounds(),
            transfer: Transfer::for_document(doc.mode, doc.depth),
            light: doc.global_light,
            patterns: &doc.patterns,
            mode: doc.mode,
            depth: doc.depth,
            doc,
        }
    }
}

/// Build the pass list for `doc`.
pub fn plan(doc: &Document) -> Result<Plan<'_>, Unsupported> {
    if doc.mode == photocraft_color::ColorMode::Multichannel {
        return Err(Unsupported("Multichannel inks (printed on the CPU)".into()));
    }
    let mut p = Planner::new(DocCtx::of(doc));
    let root = p.clear();
    // Start at the topmost layer that hides everything beneath it (an opaque fill layer): the
    // layers below can't change the result, so they cost no passes or uploads.
    let start = doc.layers.iter().rposition(|l| photocraft_compose::occludes_below(l, doc.mode)).unwrap_or(0);
    let root = p.stack_inner(doc.layers.get(start..).unwrap_or(&doc.layers), root, Some(start))?;
    Ok(p.finish(root))
}

struct Planner<'a> {
    passes: Vec<Pass<'a>>,
    free: Vec<Slot>,
    refs: Vec<u32>,
    cx: DocCtx<'a>,
    fx: Vec<FxLayer<'a>>,
    checkpoints: Vec<Checkpoint>,
}

/// Coverage source of an effect paint (`kind` in `fs_fxpaint`); flags refine it.
#[derive(Clone, Copy)]
enum Cov {
    /// An effect map; `outside` beyond its region.
    Map(MapRef, f32),
    /// Full coverage.
    One,
}

/// Shader flags for effect passes (keep in sync with compose.wgsl). Coverage `m` of a paint, with
/// `a` the layer's alpha and `inside` = `a > 0.5/255`:
/// knockout `m × (1 − a × k)` (k in `p4.w`); gate `inside ? m : 0`; rel `inside ? min(m / a, 1) : 0`;
/// stroke-out `inside ? (vector ? 0 : (1 − a) / (1 − layer alpha), 1 at an opaque layer) : m`.
pub const F_KNOCKOUT: u32 = 16;
/// Effect merge: A already holds the layer; only mix with the backdrop by the layer's opacity.
pub const F_NO_LAYER: u32 = 32;
/// Outside strokes along a filled shape's outline (`effects::outline_share`; the coverage slot
/// carries the layer's alpha in `.g`).
pub const F_OUTLINE: u32 = 131072;
/// Shape layer (outside strokes never show inside it).
pub const F_VECTOR: u32 = 64;
/// Merge onto an opaque clipping base, keeping its alpha.
pub const F_ATOP: u32 = 128;
pub const F_GATE: u32 = 256;
pub const F_REL: u32 = 512;
pub const F_STROKE_OUT: u32 = 1024;
/// First outside stroke (no accumulated share yet).
pub const F_FIRST: u32 = 2048;
/// `Lerp` per channel: weights in `p0` (channel restrictions), no opacity or mask.
pub const F_CHANNELS: u32 = 4096;
/// Lab document: Normal blending mixes in CIELAB (`psblend::LAB_MIX`).
pub const F_LAB: u32 = 65536;
/// 32-bit float document: Add, Divide and Exposure retain values above 1; the non-separable
/// blend modes clip only below 0 (`psblend::HDR`).
pub const F_HDR: u32 = 262144;
/// `Lerp`: A rounded to `p0.x` steps per unit (adjustment results on integer documents).
pub const F_QUANT: u32 = 32768;
/// `Lerp` as A + (B − C) premultiplied (layers clipped to pass-through groups).
pub const F_ADD_DIFF: u32 = 16384;
/// `Lerp`: B mixed back towards A (the backdrop) by the layer's Blend If weight
/// (`compose::apply_blend_if`), from C ("This Layer", or B itself for an adjustment layer, `p4.y`)
/// and A ("Underlying Layer"). The ranges are packed in `p0..p3` (`blend_if_params`); `p4.x` marks
/// a one-channel document.
pub const F_BLEND_IF: u32 = 524288;
/// Blend / Atop / FxMerge of a type layer: coverage mixed at `psblend::TEXT_GAMMA`.
pub const F_TEXT_GAMMA: u32 = 8192;

/// A layer's Blend If ranges for `F_BLEND_IF`: Gray, then the colour channels, each as
/// [this layer, underlying layer]; every range as two floats, `black low × 256 + black high` and
/// `white low × 256 + white high` (exact in f32), so entry `e`, side `s` starts at float
/// `(e × 2 + s) × 2`. Entries past the fourth never apply (RGB has four, one-channel modes two).
fn blend_if_params(bi: &photocraft_doc::BlendIf) -> [[f32; 4]; 4] {
    let mut f = [0.0f32; 16];
    for e in 0..4 {
        for (s, r) in bi.get(e).iter().enumerate() {
            let i = (e * 2 + s) * 2;
            f[i] = f32::from(r.black[0]) * 256.0 + f32::from(r.black[1]);
            f[i + 1] = f32::from(r.white[0]) * 256.0 + f32::from(r.white[1]);
        }
    }
    [[f[0], f[1], f[2], f[3]], [f[4], f[5], f[6], f[7]], [f[8], f[9], f[10], f[11]], [f[12], f[13], f[14], f[15]]]
}

/// `F_TEXT_GAMMA` for type layers while text gamma blending is on; the gamma goes in `p4.w`.
fn gamma_flag(layer: &Layer) -> u32 {
    if photocraft_compose::text_gamma(layer) != 1.0 { F_TEXT_GAMMA } else { 0 }
}

/// `FxInit` kinds: A at `opacity` × alpha; an opaque copy of A; A's colour with alpha `opacity`
/// inside B's shape; A with alpha × B's alpha.
pub const INIT_SCALE: i32 = 0;
pub const INIT_OPAQUE: i32 = 1;
pub const INIT_INSIDE: i32 = 2;
pub const INIT_ALPHA: i32 = 3;
/// The vector stroke C over A relative to the shape B, at `opacity`.
pub const INIT_VSTROKE: i32 = 4;
/// A's colour with alpha `opacity` × A's alpha relative to B's.
pub const INIT_RELATIVE: i32 = 5;
/// Outside-stroke coverage seed: (0, A's alpha).
pub const INIT_COVER: i32 = 6;
/// A with its alpha joined with the effect shape (the map).
pub const INIT_OUTLINE: i32 = 7;

/// `MapRef::item` of the layer's effect shape (`compose::layer_shape`) instead of an effect map.
pub const SHAPE_MAP: usize = usize::MAX;

impl<'a> Planner<'a> {
    fn new(cx: DocCtx<'a>) -> Self {
        Planner { passes: Vec::new(), free: Vec::new(), refs: Vec::new(), cx, fx: Vec::new(), checkpoints: Vec::new() }
    }

    fn finish(self, root: Slot) -> Plan<'a> {
        Plan { passes: self.passes, slots: self.refs.len() as u32, root, fx: self.fx, checkpoints: self.checkpoints }
    }

    fn alloc(&mut self) -> Slot {
        if let Some(s) = self.free.pop() {
            self.refs[s as usize] = 1;
            return s;
        }
        self.refs.push(1);
        (self.refs.len() - 1) as Slot
    }
    fn retain(&mut self, s: Slot) -> Slot {
        self.refs[s as usize] += 1;
        s
    }
    fn release(&mut self, s: Slot) {
        let r = &mut self.refs[s as usize];
        *r -= 1;
        if *r == 0 {
            self.free.push(s);
        }
    }
    /// Emit `pass` into a fresh slot, releasing its inputs.
    fn emit(&mut self, mut pass: Pass<'a>) -> Slot {
        let dst = self.alloc();
        pass.dst = dst;
        if self.cx.mode == photocraft_color::ColorMode::Lab {
            pass.flags |= F_LAB;
        }
        if self.cx.depth == photocraft_color::SampleType::F32 {
            pass.flags |= F_HDR;
        }
        let (a, b, c, d) = (pass.a, pass.b, pass.c, pass.d);
        self.passes.push(pass);
        for s in [a, b, c, d].into_iter().flatten() {
            self.release(s);
        }
        dst
    }
    fn clear(&mut self) -> Slot {
        self.emit(Pass::new(Kernel::Clear, 0))
    }

    /// composite_stack: returns the new backdrop (consumes `backdrop`).
    fn stack(&mut self, layers: &'a [Layer], backdrop: Slot) -> Result<Slot, Unsupported> {
        self.stack_inner(layers, backdrop, None)
    }

    fn stack_inner(&mut self, layers: &'a [Layer], mut backdrop: Slot, root_start: Option<usize>) -> Result<Slot, Unsupported> {
        let mut i = 0;
        while i < layers.len() {
            let base = &layers[i];
            let mut j = i + 1;
            while j < layers.len() && layers[j].clipped && !base.clipped {
                j += 1;
            }
            let clipped = &layers[i + 1..j];
            if base.visible {
                backdrop = self.layer(base, clipped, backdrop)?;
                if let (LayerContent::Adjustment(_), Some(q)) = (&base.content, photocraft_compose::adjustment_quantum(self.cx.depth)) {
                    // compose: adjustment results are rounded to the document's depth.
                    let mut p = Pass::new(Kernel::Lerp, 0);
                    p.a = Some(backdrop);
                    p.flags = F_QUANT;
                    p.params[0] = [q, 0.0, 0.0, 0.0];
                    backdrop = self.emit(p);
                }
            }
            i = j.max(i + 1);
            if let Some(start) = root_start
                && self.refs.iter().enumerate().all(|(slot, &refs)| refs == u32::from(slot == backdrop as usize))
            {
                self.checkpoints.push(Checkpoint { layer_start: start, layer_end: start.saturating_add(i), pass_end: self.passes.len(), slot: backdrop });
            }
        }
        Ok(backdrop)
    }

    fn mask_use(&self, layer: &'a Layer) -> Option<MaskUse<'a>> {
        if let Some(c) = photocraft_compose::masks::combined_mask(layer, self.cx.canvas) {
            let default = c.default_pixel().first().copied().unwrap_or(1.0);
            return Some(MaskUse { layer: layer.id, surface: SurfaceRef::Derived(std::sync::Arc::new(c)), density: 1.0, default });
        }
        let m = layer.mask.as_ref()?;
        if !m.enabled {
            return None;
        }
        Some(MaskUse {
            layer: layer.id,
            surface: SurfaceRef::Doc(&m.surface),
            density: m.density,
            default: m.surface.default_pixel().first().copied().unwrap_or(1.0),
        })
    }

    /// `compose::apply_blend_if`: `after` (the backdrop with `layer` drawn) mixed back towards
    /// `before` (the backdrop without it) where the layer's Blend If ranges hide it. "This Layer"
    /// is the layer's own content with its masks; an adjustment layer is judged by its result.
    /// Consumes both slots.
    fn blend_if(&mut self, layer: &'a Layer, before: Slot, after: Slot) -> Result<Slot, Unsupported> {
        let own = match layer.content {
            LayerContent::Adjustment(_) => None,
            _ => Some(self.content(layer)?),
        };
        let mut p = Pass::new(Kernel::Lerp, 0);
        p.a = Some(before);
        p.b = Some(after);
        p.c = own;
        p.flags = F_BLEND_IF;
        p.params = blend_if_params(&layer.blend_if);
        let one_channel = self.cx.mode != photocraft_color::ColorMode::Rgb;
        p.extra = [f32::from(u8::from(one_channel)), f32::from(u8::from(own.is_none())), 0.0, 0.0];
        Ok(self.emit(p))
    }

    /// Channel restrictions and Blend If around `draw`, which composites `layer` onto `backdrop`
    /// (`compose::composite_layer` / `composite_atop`: both are relative to the same backdrop).
    fn restricted(&mut self, layer: &'a Layer, backdrop: Slot, draw: impl FnOnce(&mut Self, Slot) -> Result<Slot, Unsupported>) -> Result<Slot, Unsupported> {
        let w = photocraft_compose::channel_weights(layer, self.cx.mode);
        let blend_if = photocraft_compose::blend_if_active(layer, self.cx.mode);
        if w.is_none() && !blend_if {
            return draw(self, backdrop);
        }
        let before = self.retain(backdrop);
        let mut after = draw(self, backdrop)?;
        if let Some(w) = w {
            let b = if blend_if { self.retain(before) } else { before };
            after = self.restore_channels(b, after, w);
        }
        if blend_if {
            after = self.blend_if(layer, before, after)?;
        }
        Ok(after)
    }

    /// Blending Options › Advanced Blending (knockout, clipped layers blended individually, blend
    /// interior effects as group, transparency shapes off, masks hiding effects) has no GPU pass:
    /// documents using it are composited on the CPU (`photocraft_compose::advanced`). Layers at
    /// Photoshop's defaults pass this check for free.
    fn check_advanced(&self, layer: &Layer, clipped: &[Layer]) -> Result<(), Unsupported> {
        if photocraft_compose::advanced_active(layer, clipped) {
            return Err(Unsupported(format!("Advanced Blending on `{}` (composited on the CPU)", layer.name)));
        }
        Ok(())
    }

    /// composite_layer, honouring the layer's channel restrictions.
    fn layer(&mut self, layer: &'a Layer, clipped: &'a [Layer], backdrop: Slot) -> Result<Slot, Unsupported> {
        self.check_advanced(layer, clipped)?;
        self.restricted(layer, backdrop, |s, b| s.layer_any(layer, clipped, b))
    }

    /// `compose::restore_channels`: `after` with the channels a layer leaves out taken from
    /// `before` (consumes both).
    fn restore_channels(&mut self, before: Slot, after: Slot, w: [f32; 3]) -> Slot {
        let mut p = Pass::new(Kernel::Lerp, 0);
        p.a = Some(before);
        p.b = Some(after);
        p.flags = F_CHANNELS;
        p.params[0] = [w[0], w[1], w[2], 1.0];
        self.emit(p)
    }

    fn layer_any(&mut self, layer: &'a Layer, clipped: &'a [Layer], backdrop: Slot) -> Result<Slot, Unsupported> {
        match layer.artboard() {
            Some(ab) => self.artboard(layer, ab, clipped, backdrop),
            None => self.layer_plain(layer, clipped, backdrop),
        }
    }

    /// compose::composite_artboard: the board's background and the group composited over the
    /// backdrop, kept only inside the board (contents and effects outside it are clipped away).
    fn artboard(&mut self, layer: &'a Layer, ab: &photocraft_doc::Artboard, clipped: &'a [Layer], backdrop: Slot) -> Result<Slot, Unsupported> {
        let board = ab.rect.intersect(&self.cx.canvas);
        if board.is_empty() {
            return Ok(backdrop);
        }
        let before = self.retain(backdrop);
        let mut sub = backdrop;
        if let Some(bg) = ab.background.rgba() {
            let mut c = Pass::new(Kernel::Content, 0);
            c.color = bg;
            let c = self.emit(c);
            let mut p = Pass::new(Kernel::Blend, 0);
            p.a = Some(sub);
            p.b = Some(c);
            sub = self.emit(p);
        }
        let sub = self.layer_plain(layer, clipped, sub)?;
        let dst = if self.refs[before as usize] == 1 {
            before
        } else {
            let n = self.alloc();
            let mut p = Pass::new(Kernel::CopyFull, n);
            p.a = Some(before);
            self.passes.push(p);
            self.release(before);
            n
        };
        let mut p = Pass::new(Kernel::CopyRect, dst);
        p.a = Some(sub);
        p.clip = Some(board);
        self.passes.push(p);
        self.release(sub);
        Ok(dst)
    }

    fn layer_plain(&mut self, layer: &'a Layer, clipped: &'a [Layer], backdrop: Slot) -> Result<Slot, Unsupported> {
        let opacity = layer.opacity * layer.fill_opacity;
        let visible_clipped: Vec<&'a Layer> = clipped.iter().filter(|c| c.visible).collect();

        // Below 100% fill a pass-through group renders isolated (as the CPU compositor does).
        if let LayerContent::Group(g) = &layer.content
            && layer.blend == BlendMode::PassThrough
            && layer.fill_opacity >= 1.0
            && !has_effects(layer)
        {
            let before = self.retain(backdrop);
            let mut after = self.stack(&g.children, backdrop)?;
            if opacity < 1.0 || layer.mask.is_some() || layer.vector_mask.is_some() {
                let mut p = Pass::new(Kernel::Lerp, 0);
                p.a = Some(self.retain(before));
                p.b = Some(after);
                p.opacity = opacity;
                p.mask = self.mask_use(layer);
                after = self.emit(p);
            }
            if visible_clipped.is_empty() {
                self.release(before);
                return Ok(after);
            }
            // Layers clipped to a pass-through group (compose: their effect on the group's
            // isolated rendering, each placed over the original backdrop, is added to the
            // pass-through result, premultiplied).
            let iso = self.content(layer)?;
            let iso_keep = self.retain(iso);
            let mut clipped_iso = iso;
            for c in visible_clipped {
                clipped_iso = self.atop(c, clipped_iso)?;
            }
            let over = |s: &mut Self, src: Slot, before: Slot| {
                let mut p = Pass::new(Kernel::Blend, 0);
                p.a = Some(before);
                p.b = Some(src);
                p.opacity = opacity;
                s.emit(p)
            };
            let b2 = self.retain(before);
            let without = over(self, iso_keep, b2);
            let with = over(self, clipped_iso, before);
            let mut p = Pass::new(Kernel::Lerp, 0);
            p.a = Some(after);
            p.b = Some(with);
            p.c = Some(without);
            p.flags = F_ADD_DIFF;
            return Ok(self.emit(p));
        }

        if let LayerContent::Adjustment(adj) = &layer.content {
            let before = self.retain(backdrop);
            let mut adjusted = self.adjust(adj, backdrop)?;
            for c in visible_clipped {
                adjusted = self.atop(c, adjusted)?;
            }
            let mut p = Pass::new(Kernel::AdjMix, 0);
            p.a = Some(before);
            p.b = Some(adjusted);
            p.mode = layer.blend;
            p.opacity = opacity;
            p.mask = self.mask_use(layer);
            return Ok(self.emit(p));
        }

        if has_effects(layer) {
            return self.effects(layer, &visible_clipped, backdrop, false);
        }

        if let LayerContent::Shape(sh) = &layer.content
            && !visible_clipped.is_empty()
            && let Some((fill, stroke)) = photocraft_compose::shape_split::split(sh, self.cx.canvas, self.cx.depth)
        {
            // The vector stroke goes above the clipped layers: the fill is the clipping base,
            // the stroke is laid over the clipped result, then the masks apply to both
            // (compose::shape_parts).
            let mut content = self.shape_part(layer, Role::Content, fill);
            for c in visible_clipped {
                content = self.atop(c, content)?;
            }
            let stroke = self.shape_part(layer, Role::Stroke, stroke);
            let mut p = Pass::new(Kernel::Blend, 0);
            p.a = Some(content);
            p.b = Some(stroke);
            let mut content = self.emit(p);
            if let Some(m) = self.mask_use(layer) {
                let mut p = Pass::new(Kernel::Mask, 0);
                p.a = Some(content);
                p.mask = Some(m);
                content = self.emit(p);
            }
            let mut p = Pass::new(Kernel::Blend, 0);
            p.a = Some(backdrop);
            p.b = Some(content);
            p.mode = layer.blend;
            p.opacity = layer.opacity;
            p.fill = layer.fill_opacity;
            return Ok(self.emit(p));
        }

        // A layer that covers little of the canvas composites only over its bounds and is
        // copied back into the backdrop (#125): a layout with hundreds of small text and shape
        // layers otherwise shades every layer over every chunk.
        let clip = self.local_clip(layer);
        if clip.is_some_and(|c| c.is_empty()) {
            // Nothing to draw (an empty layer, or a group of them).
            for c in &visible_clipped {
                self.check_advanced(c, &[])?;
            }
            return Ok(backdrop);
        }
        let start = self.passes.len();
        let mut content = self.content(layer)?;
        for c in visible_clipped {
            content = self.atop(c, content)?;
        }
        if let Some(clip) = clip {
            // Everything that built the layer's content only matters inside its bounds.
            for p in &mut self.passes[start..] {
                if p.clip.is_none() && !matches!(p.kernel, Kernel::Clear | Kernel::CopyFull) {
                    p.clip = Some(clip);
                }
            }
        }
        let mut p = Pass::new(Kernel::Blend, 0);
        p.a = Some(if clip.is_some() { self.retain(backdrop) } else { backdrop });
        p.b = Some(content);
        p.mode = layer.blend;
        p.opacity = layer.opacity;
        p.fill = layer.fill_opacity;
        p.flags = gamma_flag(layer);
        p.extra[3] = photocraft_compose::text_gamma(layer);
        p.clip = clip;
        let merged = self.emit(p);
        match clip {
            Some(clip) => Ok(self.write_back(backdrop, merged, clip)),
            None => Ok(merged),
        }
    }

    /// The rect a plain layer composites over: its composite bounds when they cover at most half
    /// the canvas (beyond that, compositing everywhere costs less than the copy back).
    fn local_clip(&self, layer: &Layer) -> Option<Rect> {
        let canvas = self.cx.canvas;
        let b = bounds::composite_bounds(layer, canvas)?;
        (b.width() as u64 * b.height() as u64 * 2 <= canvas.width() as u64 * canvas.height() as u64).then_some(b)
    }

    /// Copy `merged` into `backdrop` over `clip` (in place when nothing else holds the backdrop);
    /// returns the new backdrop. Consumes both.
    fn write_back(&mut self, backdrop: Slot, merged: Slot, clip: Rect) -> Slot {
        let dst = if self.refs[backdrop as usize] == 1 {
            backdrop
        } else {
            let n = self.alloc();
            let mut p = Pass::new(Kernel::CopyFull, n);
            p.a = Some(backdrop);
            self.passes.push(p);
            self.release(backdrop);
            n
        };
        let mut p = Pass::new(Kernel::CopyRect, dst);
        p.a = Some(merged);
        p.clip = Some(clip);
        self.passes.push(p);
        self.release(merged);
        dst
    }

    /// A stroked shape's fill or vector stroke alone (`compose::shape_split`), unmasked.
    fn shape_part(&mut self, layer: &'a Layer, role: Role, surface: Surface) -> Slot {
        let mut p = Pass::new(Kernel::Content, 0);
        p.color = photocraft_raster::to_rgba(&surface.format(), &surface.default_pixel());
        if surface.tile_count() > 0 {
            p.tex = Some(TexUse { layer: layer.id, role, surface: SurfaceRef::Derived(std::sync::Arc::new(surface)) });
        }
        self.emit(p)
    }

    /// render_content for non-adjustment layers.
    fn content(&mut self, layer: &'a Layer) -> Result<Slot, Unsupported> {
        match &layer.content {
            LayerContent::Group(g) => {
                let empty = self.clear();
                let s = self.stack(&g.children, empty)?;
                if layer.mask.is_some() || layer.vector_mask.is_some() {
                    let mut p = Pass::new(Kernel::Mask, 0);
                    p.a = Some(s);
                    p.mask = self.mask_use(layer);
                    return Ok(self.emit(p));
                }
                Ok(s)
            }
            // Adjustments are handled by the caller and never get here.
            LayerContent::Adjustment(_) => Err(Unsupported("adjustment layer reached the content planner".into())),
            LayerContent::Fill(f @ Fill::Pattern { name, scale, id, angle, link, phase }) if layer.fill_cache.as_ref().is_none_or(|c| c.fill != *f) => {
                // compose::render_fill: the pattern tiled from the layer's frame (transparent
                // when missing), then the layer's masks.
                let empty = self.clear();
                let Some(pat) = photocraft_doc::pattern::find(self.cx.patterns, id, name).filter(|p| p.is_paintable()) else { return Ok(empty) };
                let frame = photocraft_compose::fill_frame(layer, self.cx.canvas);
                let anchor = if frame.is_empty() { (0.0, 0.0) } else { (f64::from(frame.x0), f64::from(frame.y0)) };
                let paint = Paint::Pattern(pat, placement(anchor, *link, *phase, *scale, *angle));
                let canvas = self.cx.canvas;
                let s = self.paint(empty, empty, Cov::One, &paint, BlendMode::Normal, 1.0, 0, canvas, canvas);
                if layer.mask.is_some() || layer.vector_mask.is_some() {
                    let mut p = Pass::new(Kernel::Mask, 0);
                    p.a = Some(s);
                    p.mask = self.mask_use(layer);
                    return Ok(self.emit(p));
                }
                Ok(s)
            }
            _ => {
                let p = self.content_pass(layer)?;
                Ok(self.emit(p))
            }
        }
    }

    /// The Content pass of a raster / text / shape / smart / fill layer.
    fn content_pass(&self, layer: &'a Layer) -> Result<Pass<'a>, Unsupported> {
        let mut p = Pass::new(Kernel::Content, 0);
        p.mask = self.mask_use(layer);
        match &layer.content {
            LayerContent::Fill(f) => match &layer.fill_cache {
                Some(c) if c.fill == *f => self.surface_tex(&mut p, layer.id, &c.surface),
                _ => self.fill(&mut p, f, photocraft_compose::fill_frame(layer, self.cx.canvas))?,
            },
            _ => {
                if let Some(s) = layer.surface() {
                    self.surface_tex(&mut p, layer.id, s);
                }
            }
        }
        Ok(p)
    }

    fn surface_tex(&self, p: &mut Pass<'a>, id: LayerId, s: &'a Surface) {
        let dp = s.default_pixel();
        p.color = photocraft_raster::to_rgba(&s.format(), &dp);
        if s.tile_count() > 0 {
            p.tex = Some(TexUse { layer: id, role: Role::Content, surface: SurfaceRef::Doc(s) });
        }
    }

    fn fill(&self, p: &mut Pass<'a>, f: &Fill, frame: photocraft_geom::Rect) -> Result<(), Unsupported> {
        // The RGB shader cannot interpolate or dither native ink channels.
        if self.cx.mode == photocraft_color::ColorMode::Cmyk && !matches!(f, Fill::Pattern { .. }) {
            return Err(Unsupported("native CMYK fill (composited on the CPU)".into()));
        }
        match f {
            Fill::Solid(c) => {
                let rgb = c.to_rgb();
                p.color = [rgb[0], rgb[1], rgb[2], c.alpha];
            }
            Fill::Gradient { stops, opacity_stops, angle, scale, style, reverse, offset, dither, .. } => {
                check_gradient_stops(stops)?;
                check_gradient_stops(opacity_stops)?;
                p.gradient = true;
                // compose::render_fill: whole-pixel end points (fill_layout).
                let (angle, scale, offset) = photocraft_compose::fill_layout::gradient_layout(*style, *angle, *scale, *offset, frame);
                p.params[0] = [angle, scale, if *reverse { 1.0 } else { 0.0 }, style_index(*style)];
                let c = frame;
                p.params[1] = [c.x0 as f32, c.y0 as f32, c.width() as f32, c.height() as f32];
                // p2.xy: centre offset; p2.z: the depth's levels a dithered pixel is rounded to
                // (0: float, compose::gradient_fill::render_quantized); p2.w: dither (the shared
                // position hash, see the shader).
                let quantum = photocraft_compose::adjustment_quantum(self.cx.depth).unwrap_or(0.0);
                p.params[2] = [offset.0, offset.1, quantum, if *dither { 1.0 } else { 0.0 }];
                let ramp = photocraft_compose::gradient_fill::Ramp::new(f);
                let mut rows = vec![[0.0f32; 4096]; 4];
                for k in 0..4096 {
                    let v = ramp.as_ref().map_or([0.0; 4], |r| r.sample(k as f32 / 4095.0));
                    for (ch, row) in rows.iter_mut().enumerate() {
                        row[k] = v[ch];
                    }
                }
                p.lut = Some(rows);
            }
            // Pattern fills fall back to the CPU compositor (see `check`).
            Fill::Pattern { .. } => {}
        }
        Ok(())
    }

    /// composite_atop: `layer` onto `base`, restricted to the base's alpha, honouring the
    /// layer's channel restrictions.
    fn atop(&mut self, layer: &'a Layer, base: Slot) -> Result<Slot, Unsupported> {
        self.check_advanced(layer, &[])?;
        self.restricted(layer, base, |s, b| s.atop_any(layer, b))
    }

    fn atop_any(&mut self, layer: &'a Layer, base: Slot) -> Result<Slot, Unsupported> {
        let opacity = layer.opacity * layer.fill_opacity;
        if let LayerContent::Adjustment(adj) = &layer.content {
            let before = self.retain(base);
            let adjusted = self.adjust(adj, base)?;
            let mut p = Pass::new(Kernel::AdjMix, 0);
            p.a = Some(before);
            p.b = Some(adjusted);
            p.mode = layer.blend;
            p.opacity = opacity;
            p.mask = self.mask_use(layer);
            return Ok(self.emit(p));
        }
        if has_effects(layer) {
            return self.effects(layer, &[], base, true);
        }
        let content = self.content(layer)?;
        let mut p = Pass::new(Kernel::Atop, 0);
        p.a = Some(base);
        p.b = Some(content);
        p.mode = layer.blend;
        p.opacity = opacity;
        p.flags = gamma_flag(layer);
        p.extra[3] = photocraft_compose::text_gamma(layer);
        Ok(self.emit(p))
    }

    /// adjust::apply_with on a slot (consumes it).
    fn adjust(&mut self, adj: &Adjustment, src: Slot) -> Result<Slot, Unsupported> {
        if !adjustment_on_gpu(adj, self.cx.mode) {
            return Err(Unsupported(format!("{} on CMYK/Lab channels (evaluated on the CPU)", adj.label())));
        }
        if let Adjustment::GradientMap { stops, .. } = adj {
            check_gradient_stops(stops)?;
        }
        let mut p = Pass::new(Kernel::Adjust, 0);
        p.a = Some(src);
        let (kind, params, lut) = adjustment_program(adj, self.cx.transfer, self.cx.depth);
        p.adjust_kind = kind;
        p.params = params;
        p.lut = lut;
        Ok(self.emit(p))
    }

    /// `composite_with_effects` (atop = false) or the clipped-layer variant of `composite_atop`
    /// (atop = true: effects over the base treated as opaque, keeping the base's alpha).
    /// Consumes `backdrop`.
    fn effects(&mut self, layer: &'a Layer, clipped: &[&'a Layer], backdrop: Slot, atop: bool) -> Result<Slot, Unsupported> {
        let canvas = self.cx.canvas;
        let region = bounds::effect_region(layer, canvas);
        let sb = photocraft_compose::paint_bounds(layer).unwrap_or_else(|| bounds::layer_bounds(layer, canvas));
        let clip = if bounds::transparent_outside(layer) { region } else { canvas };
        // A stroked shape's vector stroke goes above its clipped layers and interior effects
        // (compose::split_parts): the fill and the stroke unmasked, the masks applied after.
        let split = match &layer.content {
            LayerContent::Shape(sh) if sh.stroke.is_some() => photocraft_compose::shape_split::split(sh, canvas, self.cx.depth),
            _ => None,
        };
        let (mut content, vstroke) = match split {
            Some((fill, stroke)) => {
                let f = self.shape_part(layer, Role::Content, fill);
                let s = self.shape_part(layer, Role::Stroke, stroke);
                (f, Some(s))
            }
            None => (self.content(layer)?, None),
        };
        for p in self.passes.iter_mut().rev().take(if vstroke.is_some() { 2 } else { 1 }) {
            if !matches!(layer.content, LayerContent::Group(_)) {
                p.clip = Some(clip);
            }
        }
        for c in clipped {
            content = self.atop(c, content)?;
        }
        let fx = self.fx.len();
        self.fx.push(FxLayer { layer, region, bounds: sb });
        let outline = photocraft_compose::effect_outline(layer).is_some();
        let relative = outline || vstroke.is_some();
        // `content` becomes the effect shape (B of every effect pass); `lay_src` the layer's colour.
        let (lay_src, vstroke) = match vstroke {
            Some(s) => {
                let mut p = Pass::new(Kernel::Blend, 0);
                p.a = Some(self.retain(content));
                p.b = Some(self.retain(s));
                p.clip = Some(clip);
                let union = self.emit(p);
                let m = self.mask_use(layer);
                let masked = |me: &mut Self, slot: Slot| match &m {
                    Some(m) => {
                        let mut p = Pass::new(Kernel::Mask, 0);
                        p.a = Some(slot);
                        p.mask = Some(m.clone());
                        p.clip = Some(clip);
                        me.emit(p)
                    }
                    None => slot,
                };
                let fill = masked(self, content);
                content = masked(self, union);
                (fill, Some(masked(self, s)))
            }
            None => (self.retain(content), None),
        };
        if outline {
            let mut p = Pass::new(Kernel::FxInit, 0);
            p.a = Some(content);
            p.adjust_kind = INIT_OUTLINE;
            p.map = Some(MapRef { fx, item: SHAPE_MAP, map: 0 });
            p.clip = Some(clip);
            content = self.emit(p);
        }

        let items: Vec<&'a Effect> = layer.effects.items.iter().filter(|e| e.enabled()).collect();
        for e in &items {
            let gradient = match e {
                Effect::GradientOverlay { gradient, .. } => Some(gradient),
                Effect::Stroke(s) => match &s.paint {
                    FxPaint::Gradient(g) => Some(g),
                    _ => None,
                },
                Effect::OuterGlow(g) | Effect::InnerGlow(g) => match &g.paint {
                    FxPaint::Gradient(g) => Some(g),
                    _ => None,
                },
                _ => None,
            };
            if let Some(g) = gradient {
                check_gradient_stops(&g.stops)?;
                check_gradient_stops(&g.opacity_stops)?;
            }
        }
        // Linked patterns tile from the effects reference point (else the layer's top-left).
        let anchor = layer.effects.reference.unwrap_or((f64::from(sb.x0), f64::from(sb.y0)));
        let vector_shape = matches!(layer.content, LayerContent::Shape(_)) && !outline;
        let map = |item: usize, map: usize| MapRef { fx, item, map };
        let rev: Vec<(usize, &'a Effect)> = items.iter().copied().enumerate().rev().collect();
        let init = |s: &mut Self, src: Slot, b: Option<Slot>, kind: i32, opacity: f32| {
            let mut p = Pass::new(Kernel::FxInit, 0);
            p.a = Some(src);
            p.b = b;
            p.adjust_kind = kind;
            p.opacity = opacity;
            p.clip = Some(clip);
            s.emit(p)
        };

        // Exterior effects over the backdrop (an opaque copy of it for clipped layers).
        let mut w = if atop {
            let b = self.retain(backdrop);
            init(self, b, None, INIT_OPAQUE, 1.0)
        } else {
            self.retain(backdrop)
        };
        for &(i, e) in &rev {
            if let Effect::DropShadow(s) = e {
                // The layer hides the shadow beneath it where its fill is see-through, and wholly
                // in a mode that blends with it (as the CPU renderer: `shadow_see_through`).
                let flags = if s.knocks_out { F_KNOCKOUT } else { 0 };
                let see_through = photocraft_compose::effects::shadow_see_through(layer);
                w = self.paint_k(
                    w,
                    content,
                    Cov::Map(map(i, 0), 0.0),
                    &Paint::Color(s.color.to_rgb()),
                    s.common.blend,
                    s.common.opacity,
                    flags,
                    clip,
                    sb,
                    see_through,
                );
            }
        }
        for &(i, e) in &rev {
            if let Effect::OuterGlow(g) = e {
                let paint = self.glow_paint(g, anchor);
                // Match CPU exterior knockout: a see-through fill must not reveal the
                // glow through its own shape. The shader applies m × (1 − alpha × k).
                let see_through = 1.0 - layer.fill_opacity.clamp(0.0, 1.0);
                w = self.paint_k(w, content, Cov::Map(map(i, 0), 0.0), &paint, g.common.blend, g.common.opacity, F_KNOCKOUT, clip, sb, see_through);
            }
        }

        // The layer: its colour at fill opacity inside its shape; interior effects are painted
        // relative to the shape (coverage within it), then the shape's alpha applies.
        let mut l = if relative {
            let c = self.retain(content);
            init(self, lay_src, Some(c), INIT_RELATIVE, layer.fill_opacity)
        } else {
            init(self, lay_src, None, INIT_INSIDE, layer.fill_opacity)
        };
        // Blend Interior Effects as Group off: the layer merges in its mode first and the
        // interior effects paint over the result (`effects::premerges_layer`).
        let premerge = photocraft_compose::effects::premerges_layer(layer);
        if premerge {
            let mut p = Pass::new(Kernel::Blend, 0);
            p.a = Some(self.retain(w));
            p.b = Some(l);
            p.mode = layer.blend;
            p.flags = gamma_flag(layer);
            p.extra[3] = photocraft_compose::text_gamma(layer);
            p.clip = Some(clip);
            l = self.emit(p);
        }
        for &(_, e) in &rev {
            if let Effect::PatternOverlay { common, name, id, scale, angle, link, phase } = e
                && let Some(pat) = photocraft_doc::pattern::find(self.cx.patterns, id, name).filter(|p| p.is_paintable())
            {
                let paint = Paint::Pattern(pat, placement(anchor, *link, *phase, *scale, *angle));
                l = self.paint(l, content, Cov::One, &paint, common.blend, common.opacity, F_GATE, clip, sb);
            }
        }
        for &(_, e) in &rev {
            if let Effect::GradientOverlay { common, gradient, .. } = e {
                l = self.paint(l, content, Cov::One, &Paint::Gradient(gradient), common.blend, common.opacity, F_GATE, clip, sb);
            }
        }
        for &(_, e) in &rev {
            if let Effect::ColorOverlay { common, color } = e {
                l = self.paint(l, content, Cov::One, &Paint::Color(color.to_rgb()), common.blend, common.opacity, F_GATE, clip, sb);
            }
        }
        for &(i, e) in &rev {
            if let Effect::Satin(s) = e {
                l = self.paint(l, content, Cov::Map(map(i, 0), 0.0), &Paint::Color(s.color.to_rgb()), s.common.blend, s.common.opacity, F_REL, clip, sb);
            }
        }
        for &(i, e) in &rev {
            if let Effect::InnerGlow(g) = e {
                let paint = self.glow_paint(g, anchor);
                l = self.paint(l, content, Cov::Map(map(i, 0), 0.0), &paint, g.common.blend, g.common.opacity, F_REL, clip, sb);
            }
        }
        for &(i, e) in &rev {
            if let Effect::InnerShadow(s) = e {
                l = self.paint(l, content, Cov::Map(map(i, 0), 0.0), &Paint::Color(s.color.to_rgb()), s.common.blend, s.common.opacity, F_REL, clip, sb);
            }
        }
        if let Some(s) = vstroke {
            let mut p = Pass::new(Kernel::FxInit, 0);
            p.a = Some(l);
            p.b = Some(self.retain(content));
            p.c = Some(s);
            p.adjust_kind = INIT_VSTROKE;
            p.opacity = layer.fill_opacity;
            p.clip = Some(clip);
            l = self.emit(p);
        }
        // Inside stroke parts are painted over the layer (bottom instance first).
        for &(i, e) in &rev {
            if let Effect::Stroke(st) = e {
                let (in_w, _) = stroke_widths(st);
                if in_w > 0.0 {
                    let paint = self.fx_paint(&st.paint, anchor);
                    l = self.paint(
                        l,
                        content,
                        Cov::Map(map(i, 1), (in_w + 0.5).clamp(0.0, 1.0)),
                        &paint,
                        st.common.blend,
                        st.common.opacity,
                        F_GATE,
                        clip,
                        photocraft_compose::stroke_frame(self.cx.doc, layer, st).unwrap_or(sb),
                    );
                }
            }
        }
        let bevel_paint = |b: &photocraft_doc::Bevel| photocraft_compose::effects::bevel_geom(b).paint;
        for &(i, e) in &rev {
            if let Effect::BevelEmboss(b) = e
                && bevel_paint(b) == photocraft_compose::effects::BevelPaint::Inner
            {
                l = self.paint(
                    l,
                    content,
                    Cov::Map(map(i, 0), 0.0),
                    &Paint::Color(b.highlight_color.to_rgb()),
                    b.highlight.blend,
                    b.highlight.opacity,
                    F_REL,
                    clip,
                    sb,
                );
                l = self.paint(l, content, Cov::Map(map(i, 1), 0.0), &Paint::Color(b.shadow_color.to_rgb()), b.shadow.blend, b.shadow.opacity, F_REL, clip, sb);
            }
        }
        // The shape's own alpha.
        let c = self.retain(content);
        l = init(self, l, Some(c), INIT_ALPHA, 1.0);
        // Outside stroke parts lie beneath the layer and blend onto the exterior result with their
        // own mode; a higher stroke covers the ones below it (each takes its coverage × opacity
        // not yet taken above it, blended over the result as it was before the strokes).
        let outs: Vec<(usize, &photocraft_doc::StrokeFx)> = items
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(i, e)| if let Effect::Stroke(st) = e { Some((i, st)) } else { None })
            .filter(|(_, st)| stroke_widths(st).1 > 0.0)
            .collect();
        if !outs.is_empty() {
            let (mut acc, mut cover): (Option<Slot>, Option<Slot>) = (None, None);
            if outline || !vector_shape {
                // The layer's alpha rides along the coverage: `effects::outline_share`, and for
                // pixel layers the stroke's share beneath a partly covering layer (fill < 100 %).
                let ll = self.retain(l);
                cover = Some(init(self, ll, None, INIT_COVER, 1.0));
            }
            for (n, &(i, st)) in outs.iter().enumerate() {
                let paint = self.fx_paint(&st.paint, anchor);
                let flags = F_STROKE_OUT | if vector_shape { F_VECTOR } else { 0 } | if outline { F_OUTLINE } else { 0 } | if n == 0 { F_FIRST } else { 0 };
                // Share × blended colour, then coverage (both read the coverage so far).
                let mut p = self.fx_pass_kernel(
                    Kernel::FxStroke,
                    w,
                    content,
                    &paint,
                    st.common.blend,
                    st.common.opacity,
                    flags,
                    clip,
                    photocraft_compose::stroke_frame(self.cx.doc, layer, st).unwrap_or(sb),
                );
                if matches!(paint, Paint::None) {
                    // A missing pattern paints nothing: the stroke's share keeps the result as is.
                    p.params[2][2] = 3.0;
                }
                p.adjust_kind = 0;
                p.map = Some(map(i, 0));
                p.c = cover.map(|c| self.retain(c));
                p.d = acc;
                let a = self.emit(p);
                let mut p = self.fx_pass_kernel(Kernel::FxStroke, w, content, &Paint::Color([0.0; 3]), BlendMode::Normal, st.common.opacity, flags, clip, sb);
                p.adjust_kind = 1;
                p.map = Some(map(i, 0));
                p.c = cover;
                cover = Some(self.emit(p));
                acc = Some(a);
            }
            let mut p = Pass::new(Kernel::FxStrokeEnd, 0);
            p.a = Some(w);
            p.b = acc;
            p.c = cover;
            p.clip = Some(clip);
            w = self.emit(p);
        }
        for &(i, e) in &rev {
            if let Effect::BevelEmboss(b) = e
                && bevel_paint(b) == photocraft_compose::effects::BevelPaint::Outer
            {
                let k = 0;
                w = self.paint(
                    w,
                    content,
                    Cov::Map(map(i, k), 0.0),
                    &Paint::Color(b.highlight_color.to_rgb()),
                    b.highlight.blend,
                    b.highlight.opacity,
                    0,
                    clip,
                    sb,
                );
                w = self.paint(w, content, Cov::Map(map(i, k + 1), 0.0), &Paint::Color(b.shadow_color.to_rgb()), b.shadow.blend, b.shadow.opacity, 0, clip, sb);
            }
        }

        let mode = if premerge || layer.blend == BlendMode::PassThrough { BlendMode::Normal } else { layer.blend };
        let late: Vec<(usize, &'a photocraft_doc::Bevel)> = rev
            .iter()
            .filter_map(|&(i, e)| if let Effect::BevelEmboss(b) = e { Some((i, b)) } else { None })
            .filter(|(_, b)| bevel_paint(b) == photocraft_compose::effects::BevelPaint::Both)
            .collect();
        let merged = if late.is_empty() {
            let mut p = Pass::new(Kernel::FxMerge, 0);
            p.a = Some(w);
            p.b = Some(l);
            p.c = Some(self.retain(backdrop));
            p.mode = mode;
            p.opacity = layer.opacity;
            p.flags = if atop { F_ATOP } else { 0 } | gamma_flag(layer);
            p.extra[3] = photocraft_compose::text_gamma(layer);
            p.clip = Some(clip);
            self.emit(p)
        } else {
            // Emboss styles shade the composited layer (`composite_with_effects`): merge at full
            // opacity, paint their maps (inside and outside halves), then mix.
            let mut p = Pass::new(Kernel::FxMerge, 0);
            p.a = Some(w);
            p.b = Some(l);
            p.c = Some(self.retain(w));
            p.mode = mode;
            p.opacity = 1.0;
            p.flags = gamma_flag(layer);
            p.extra[3] = photocraft_compose::text_gamma(layer);
            p.clip = Some(clip);
            let mut m = self.emit(p);
            for (i, b) in late {
                for (k, color, fxc) in [(0, &b.highlight_color, &b.highlight), (1, &b.shadow_color, &b.shadow)] {
                    m = self.paint(m, content, Cov::Map(map(i, k), 0.0), &Paint::Color(color.to_rgb()), fxc.blend, fxc.opacity, 0, clip, sb);
                }
            }
            let mut p = Pass::new(Kernel::FxMerge, 0);
            p.b = Some(self.retain(m));
            p.a = Some(m);
            p.c = Some(self.retain(backdrop));
            p.mode = mode;
            p.opacity = layer.opacity;
            p.flags = if atop { F_ATOP } else { 0 } | F_NO_LAYER;
            p.clip = Some(clip);
            self.emit(p)
        };
        self.release(content);

        // Write the clipped result back into the backdrop (in place when nothing else holds it).
        Ok(self.write_back(backdrop, merged, clip))
    }

    fn fx_paint(&self, p: &'a FxPaint, anchor: (f64, f64)) -> Paint<'a> {
        match p {
            FxPaint::Color(c) => Paint::Color(c.to_rgb()),
            FxPaint::Gradient(g) => Paint::Gradient(g),
            FxPaint::Pattern { name, id, scale } => match photocraft_doc::pattern::find(self.cx.patterns, id, name).filter(|p| p.is_paintable()) {
                Some(pat) => Paint::Pattern(pat, placement(anchor, true, (0.0, 0.0), *scale, 0.0)),
                None => Paint::None,
            },
        }
    }

    fn glow_paint(&self, g: &'a Glow, anchor: (f64, f64)) -> Paint<'a> {
        match &g.paint {
            FxPaint::Gradient(gradient) => Paint::GlowGradient(gradient, photocraft_compose::effects::glow_gradient_gain(g.range)),
            p => self.fx_paint(p, anchor),
        }
    }

    /// An effect pass reading `dst` (A, retained) and the layer `content` (B, retained) with its
    /// paint set up.
    #[allow(clippy::too_many_arguments)]
    fn fx_pass(&mut self, dst: Slot, content: Slot, paint: &Paint<'a>, blend: BlendMode, opacity: f32, flags: u32, clip: Rect, sb: Rect) -> Pass<'a> {
        self.fx_pass_kernel(Kernel::FxPaint, dst, content, paint, blend, opacity, flags, clip, sb)
    }

    #[allow(clippy::too_many_arguments)]
    fn fx_pass_kernel(
        &mut self,
        kernel: Kernel,
        dst: Slot,
        content: Slot,
        paint: &Paint<'a>,
        blend: BlendMode,
        opacity: f32,
        flags: u32,
        clip: Rect,
        sb: Rect,
    ) -> Pass<'a> {
        let mut p = Pass::new(kernel, 0);
        p.a = Some(self.retain(dst));
        p.b = Some(self.retain(content));
        p.mode = blend;
        p.opacity = opacity;
        p.flags = flags;
        p.clip = Some(clip);
        match paint {
            Paint::None => {}
            Paint::Color(c) => p.color = [c[0], c[1], c[2], 1.0],
            Paint::Gradient(g) => {
                // effects::paint_fx: whole-pixel end points (fill_layout::gradient_layout).
                let (angle, scale, offset) = photocraft_compose::fill_layout::gradient_layout(g.style, g.angle, g.scale, g.offset, sb);
                p.params[0] = [angle, scale, if g.reverse { 1.0 } else { 0.0 }, style_index(g.style)];
                p.params[1] = [sb.x0 as f32, sb.y0 as f32, sb.width() as f32, sb.height() as f32];
                p.params[2][0] = offset.0;
                p.params[2][1] = offset.1;
                p.params[2][2] = 1.0;
                p.lut = Some(gradient_rows(g));
            }
            Paint::GlowGradient(g, gain) => {
                // effects::paint_glow: the gradient at 1 - strength, opaque from 1 / gain.
                p.params[0][0] = *gain;
                p.params[2][2] = 4.0;
                p.lut = Some(gradient_rows(g));
            }
            Paint::Pattern(pat, pl) => {
                p.params[2][2] = 2.0;
                p.params[3] = [pl.origin.0 as f32, pl.origin.1 as f32, pl.cs.0 as f32, pl.cs.1 as f32];
                p.extra = [pl.inv_scale as f32, pat.width as f32, pat.height as f32, 0.0];
                p.pattern = Some(pat);
            }
        }
        p
    }

    /// One effect paint into `dst` (consumed); reads the layer's alpha from `content` (kept).
    #[allow(clippy::too_many_arguments)]
    fn paint(&mut self, dst: Slot, content: Slot, cov: Cov, paint: &Paint<'a>, blend: BlendMode, opacity: f32, flags: u32, clip: Rect, sb: Rect) -> Slot {
        self.paint_k(dst, content, cov, paint, blend, opacity, flags, clip, sb, 1.0)
    }

    /// [`Self::paint`] with the knockout strength (`F_KNOCKOUT`: coverage × (1 − alpha × k)).
    #[allow(clippy::too_many_arguments)]
    fn paint_k(
        &mut self,
        dst: Slot,
        content: Slot,
        cov: Cov,
        paint: &Paint<'a>,
        blend: BlendMode,
        opacity: f32,
        flags: u32,
        clip: Rect,
        sb: Rect,
        knockout: f32,
    ) -> Slot {
        if matches!(paint, Paint::None) {
            // Missing pattern: compose paints nothing.
            return dst;
        }
        let mut p = self.fx_pass(dst, content, paint, blend, opacity, flags, clip, sb);
        self.release(dst);
        match cov {
            Cov::Map(m, outside) => {
                p.adjust_kind = 0;
                p.map = Some(m);
                p.params[2][3] = outside;
            }
            Cov::One => p.adjust_kind = 1,
        }
        p.extra[3] = knockout;
        self.emit(p)
    }
}

/// Paint source of an effect.
enum Paint<'a> {
    None,
    Color([f32; 3]),
    Gradient(&'a Gradient),
    /// A glow's gradient with its opacity gain (`effects::paint_glow`).
    GlowGradient(&'a Gradient, f32),
    Pattern(&'a Pattern, Placement),
}

/// `compose::pattern::Placement` (origin, inverse rotation, inverse scale).
#[derive(Clone, Copy, Debug)]
struct Placement {
    origin: (f64, f64),
    cs: (f64, f64),
    inv_scale: f64,
}

/// `Placement::anchored`: linked patterns tile from `anchor` (the effects reference point).
fn placement(anchor: (f64, f64), link: bool, phase: (f32, f32), scale: f32, angle: f32) -> Placement {
    let base = if link { anchor } else { (0.0, 0.0) };
    let origin = (base.0 + f64::from(phase.0), base.1 + f64::from(phase.1));
    let s = f64::from(scale);
    let inv_scale = if s.is_finite() && s > 1e-3 { 1.0 / s } else { 1.0 };
    let a = f64::from(angle).to_radians();
    Placement { origin, cs: (a.cos(), a.sin()), inv_scale }
}

/// (inside width, outside width) of a stroke.
pub fn stroke_widths(st: &photocraft_doc::StrokeFx) -> (f32, f32) {
    match st.position {
        StrokePosition::Outside => (0.0, st.size),
        StrokePosition::Inside => (st.size, 0.0),
        StrokePosition::Center => (st.size / 2.0, st.size / 2.0),
    }
}

fn style_index(s: photocraft_doc::GradientStyle) -> f32 {
    match s {
        photocraft_doc::GradientStyle::Linear => 0.0,
        photocraft_doc::GradientStyle::Radial => 1.0,
        photocraft_doc::GradientStyle::Angle => 2.0,
        photocraft_doc::GradientStyle::Reflected => 3.0,
        photocraft_doc::GradientStyle::Diamond => 4.0,
    }
}

fn sample_stops4(stops: &[(f32, [f32; 4])], t: f32) -> [f32; 4] {
    match stops {
        [] => [0.0; 4],
        [only] => only.1,
        _ => {
            if t <= stops[0].0 {
                return stops[0].1;
            }
            for w in stops.windows(2) {
                let (a, b) = (&w[0], &w[1]);
                if t <= b.0 {
                    let k = if b.0 > a.0 { (t - a.0) / (b.0 - a.0) } else { 0.0 };
                    return std::array::from_fn(|i| a.1[i] + (b.1[i] - a.1[i]) * k);
                }
            }
            stops[stops.len() - 1].1
        }
    }
}

fn lut_rows(f: impl Fn(usize, f32) -> f32, rows: usize) -> Vec<[f32; 4096]> {
    (0..rows)
        .map(|r| {
            let mut row = [0.0f32; 4096];
            for (k, v) in row.iter_mut().enumerate() {
                *v = f(r, k as f32 / 4095.0);
            }
            row
        })
        .collect()
}

type Program = (i32, [[f32; 4]; 4], Option<Vec<[f32; 4096]>>);

/// A CPU LUT (4096 entries) as one texture row.
fn to_row(t: &[f32]) -> [f32; 4096] {
    let mut row = [0.0f32; 4096];
    for (o, v) in row.iter_mut().zip(t) {
        *o = *v;
    }
    row
}

/// Whether the adjustment kernel can evaluate `adj` in a document of colour `mode`: Levels and
/// Curves on CMYK ink or Lab channels convert through ICC profiles per pixel, and so do the
/// adjustments a Lab document evaluates on L, a and b (`photocraft_compose::adjust::lab_channel_adjustment`)
/// and a CMYK document on its inks (`cmyk_channel_adjustment`), which only the CPU does.
pub fn adjustment_on_gpu(adj: &Adjustment, mode: photocraft_color::ColorMode) -> bool {
    use photocraft_compose::adjust::{cmyk_channel_adjustment, lab_channel_adjustment};
    let tone_in_space =
        matches!(adj, Adjustment::Levels { space: ToneSpace::Cmyk | ToneSpace::Lab, .. } | Adjustment::Curves { space: ToneSpace::Cmyk | ToneSpace::Lab, .. });
    let own_channels = match mode {
        photocraft_color::ColorMode::Lab => lab_channel_adjustment(adj),
        photocraft_color::ColorMode::Cmyk => cmyk_channel_adjustment(adj),
        _ => false,
    };
    !tone_in_space && !own_channels
}

/// A tone transfer as the shader's `t_decode`/`t_encode` exponent (0: the sRGB curve).
fn transfer_exponent(t: Transfer) -> f32 {
    match t {
        Transfer::Srgb => 0.0,
        Transfer::Gamma(g) => g,
    }
}

/// Adjustment → (kernel kind, parameters, LUT rows). Kinds are the `switch` in `adjust()`.
pub fn adjustment_program(adj: &Adjustment, transfer: Transfer, depth: photocraft_color::SampleType) -> Program {
    let mut p = [[0.0f32; 4]; 4];
    match adj {
        Adjustment::Invert => (1, p, None),
        Adjustment::Threshold { level } => {
            p[0][0] = (level * 255.0).round();
            (2, p, None)
        }
        Adjustment::Posterize { levels } => {
            p[0][0] = (*levels).clamp(2, 255) as f32;
            (3, p, None)
        }
        Adjustment::BrightnessContrast { brightness, contrast, legacy: true } => {
            let c = contrast.clamp(-100.0, 99.0);
            let k = if c >= 0.0 { 1.0 / (1.0 - c / 100.0) } else { 1.0 + c / 100.0 };
            // Brightness before contrast when it is raised, after it when lowered (compose::adjust).
            let (pre, post) = if c >= 0.0 { (brightness / 255.0, 0.0) } else { (0.0, brightness / 255.0) };
            p[0] = [pre, k, photocraft_compose::adjust::LEGACY_PIVOT, post];
            (4, p, None)
        }
        Adjustment::BrightnessContrast { brightness, contrast, .. } => {
            // Modern B/C: the shader rebuilds the curves from raw slider values (see compose.wgsl
            // `mbright`/`mcontrast`, mirroring compose::adjust::modern_brightness/modern_contrast).
            p[0] = [*brightness, *contrast, 0.0, 0.0];
            (5, p, None)
        }
        Adjustment::Exposure { exposure, offset, gamma } => {
            p[0] = [2f32.powf(*exposure), *offset, gamma.max(0.01), transfer_exponent(transfer.for_exposure())];
            (6, p, None)
        }
        Adjustment::Levels { master, per_channel, space: ToneSpace::Rgb, .. } if depth == photocraft_color::SampleType::F32 => {
            // Twenty coefficients exceed the sixteen uniform parameters. Store four records
            // in LUT row 0, read texel-exact by the analytic float Levels kernel.
            let mut row = [0.0f32; 4096];
            for (i, ch) in std::iter::once(master).chain(per_channel).enumerate() {
                let coefficients = [ch.in_black, ch.in_white, ch.gamma.max(0.01), ch.out_black, ch.out_white];
                row[i * 5..i * 5 + 5].copy_from_slice(&coefficients);
            }
            (17, p, Some(vec![row]))
        }
        // RGB space only (see `adjustment_on_gpu`); the rows are the CPU's channel∘master LUTs.
        Adjustment::Levels { .. } | Adjustment::Curves { .. } => {
            for (v, identity) in p[0].iter_mut().zip(adjust::float_tone_identity(adj, Some(depth))) {
                *v = if identity { 1.0 } else { 0.0 };
            }
            (7, p, Some(adjust::tone_luts_depth(adj, Some(depth)).iter().take(3).map(|t| to_row(t)).collect()))
        }
        Adjustment::HueSaturation { hue, saturation, lightness, colorize, ranges } => {
            p[0] = [*hue, saturation / 100.0, lightness / 100.0, if *colorize { 1.0 } else { 0.0 }];
            if !*colorize && ranges.iter().any(|r| !r.is_neutral()) {
                // Range edits: hue-indexed rows (shift, saturation, lightness), as on the CPU.
                p[1][0] = 1.0;
                (8, p, Some(adjust::hue_range_tables(ranges).iter().map(|t| to_row(t)).collect()))
            } else {
                (8, p, None)
            }
        }
        Adjustment::Vibrance { vibrance, saturation } => {
            // compose::adjust::vibrance_px; 32-bit samples aren't clipped at 1.
            let hi = if depth == photocraft_color::SampleType::F32 { f32::MAX } else { 1.0 };
            p[0] = [vibrance / 100.0, saturation / 100.0, transfer_exponent(transfer), hi];
            let b = adjust::VIBRANCE_BOOST;
            p[1] = [b[0], b[1], b[2], b[3]];
            p[2] = [b[4], b[5], b[6], 0.0];
            (9, p, None)
        }
        Adjustment::ChannelMixer { matrix, monochrome } => {
            p[0] = matrix[0];
            p[1] = matrix[1];
            p[2] = matrix[2];
            p[3][0] = if *monochrome { 1.0 } else { 0.0 };
            (10, p, None)
        }
        Adjustment::PhotoFilter { color, density, preserve_luminosity } => {
            // compose::adjust: the linear-light matrix, then SetLum on the encoded values (8/16-bit).
            let linear_doc = depth == photocraft_color::SampleType::F32;
            let m = adjust::photo_filter_matrix(*color, *density, *preserve_luminosity && linear_doc);
            for (row, m) in p.iter_mut().zip(m) {
                *row = [m[0], m[1], m[2], 0.0];
            }
            p[3] = [if *preserve_luminosity && !linear_doc { 1.0 } else { 0.0 }, transfer_exponent(transfer), 0.0, 0.0];
            (11, p, None)
        }
        Adjustment::BlackWhite { weights, tint } => {
            p[0] = [weights[0], weights[1], weights[2], weights[3]];
            p[1] = [weights[4], weights[5], if tint.is_some() { 1.0 } else { 0.0 }, 0.0];
            if let Some(t) = tint {
                p[2] = [t[0], t[1], t[2], 0.0];
            }
            (12, p, None)
        }
        Adjustment::GradientMap { stops, reverse, dither } => {
            p[0][0] = if *reverse { 1.0 } else { 0.0 };
            p[0][1] = if *dither { 1.0 } else { 0.0 };
            let s4: Vec<(f32, [f32; 4])> = stops.iter().map(|(t, c)| (*t, [c[0], c[1], c[2], 1.0])).collect();
            let rows = lut_rows(|r, t| if s4.is_empty() { t } else { sample_stops4(&s4, t)[r] }, 3);
            (13, p, Some(rows))
        }
        Adjustment::ColorBalance { shadows, midtones, highlights, preserve_luminosity } => {
            // Per-channel curves (compose::adjust::color_balance_channel), through the Levels kernel.
            let tones = [*shadows, *midtones, *highlights];
            (7, p, Some(lut_rows(|ch, v| adjust::color_balance_channel(tones, *preserve_luminosity, ch, v), 3)))
        }
        Adjustment::SelectiveColor { relative, adjustments } => {
            // 9 ranges × CMYK don't fit the 16 params: they go in LUT row 0 (read texel-exact).
            p[0][0] = if *relative { 1.0 } else { 0.0 };
            let mut row = [0.0f32; 4096];
            for (i, v) in adjustments.iter().flatten().enumerate() {
                row[i] = *v;
            }
            (15, p, Some(vec![row]))
        }
        Adjustment::ColorLookup { lut: Some(table), size, tetrahedral, dither, .. } if lut3d_len(*size).is_some_and(|len| table.len() >= len) => {
            // The flattened table (n³ RGB triplets) spans as many 4096-wide rows as it needs.
            let n = *size as usize;
            let rows = lut3d_len(*size)
                .and_then(|len| table.get(..len))
                .unwrap_or_default()
                .chunks(4096)
                .map(|c| {
                    let mut row = [0.0f32; 4096];
                    row[..c.len()].copy_from_slice(c);
                    row
                })
                .collect();
            p[0] = [n as f32, if *tetrahedral { 1.0 } else { 0.0 }, if *dither { 1.0 } else { 0.0 }, 0.0];
            (16, p, Some(rows))
        }
        // Identity on the CPU too (not evaluated there).
        Adjustment::ColorLookup { .. } | Adjustment::Unsupported { .. } => (0, p, None),
    }
}

/// A gradient's colour and alpha as four 4096-entry LUT rows (`lut_tex`).
fn gradient_rows(g: &Gradient) -> Vec<[f32; 4096]> {
    let mut rows = vec![[0.0f32; 4096]; 4];
    for k in 0..4096 {
        let v = photocraft_compose::effects::sample_gradient(g, k as f32 / 4095.0);
        for (ch, row) in rows.iter_mut().enumerate() {
            row[k] = v[ch];
        }
    }
    rows
}

/// Linear interpolation between LUT entries cannot preserve a discontinuity. Let the canvas
/// use the CPU oracle for coincident stops, including opacity stops and unsorted fill ramps
/// (which `gradient_fill::Ramp` sorts). Check only ramps we will sample: cached fills and
/// disabled effects can still use the GPU.
fn check_gradient_stops<T>(stops: &[(f32, T)]) -> Result<(), Unsupported> {
    let mut positions: Vec<f32> = stops.iter().map(|s| s.0).collect();
    positions.sort_by(f32::total_cmp);
    // A 4096-entry LUT cannot keep two stops that fall in the same entry apart (#3053).
    if positions.windows(2).any(|w| (w[1] - w[0]).abs() < 1.0 / 4095.0) {
        return Err(Unsupported("coincident gradient stops (composited on the CPU)".into()));
    }
    Ok(())
}

/// Mode index used by the shader (declaration order of [`BlendMode`]).
pub fn mode_index(m: BlendMode) -> i32 {
    match m {
        BlendMode::PassThrough => 0,
        BlendMode::Normal => 1,
        BlendMode::Dissolve => 2,
        BlendMode::Darken => 3,
        BlendMode::Multiply => 4,
        BlendMode::ColorBurn => 5,
        BlendMode::LinearBurn => 6,
        BlendMode::DarkerColor => 7,
        BlendMode::Lighten => 8,
        BlendMode::Screen => 9,
        BlendMode::ColorDodge => 10,
        BlendMode::LinearDodge => 11,
        BlendMode::LighterColor => 12,
        BlendMode::Overlay => 13,
        BlendMode::SoftLight => 14,
        BlendMode::HardLight => 15,
        BlendMode::VividLight => 16,
        BlendMode::LinearLight => 17,
        BlendMode::PinLight => 18,
        BlendMode::HardMix => 19,
        BlendMode::Difference => 20,
        BlendMode::Exclusion => 21,
        BlendMode::Subtract => 22,
        BlendMode::Divide => 23,
        BlendMode::Hue => 24,
        BlendMode::Saturation => 25,
        BlendMode::Color => 26,
        BlendMode::Luminosity => 27,
        BlendMode::Reflect => 28,
        BlendMode::Glow => 29,
        BlendMode::Negation => 30,
        BlendMode::Xor => 31,
        BlendMode::PaintNetColorBurn => 32,
        BlendMode::PaintNetColorDodge => 33,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_color::{Color, ColorMode, SampleType};
    use photocraft_geom::Size;

    #[test]
    fn stops_closer_than_one_lut_step_stay_on_the_cpu() {
        assert!(check_gradient_stops(&[(0.4999_f32, ()), (0.5, ())]).is_err());
        assert!(check_gradient_stops(&[(0.0_f32, ()), (1.0, ())]).is_ok());
    }

    #[test]
    fn lab_channel_adjustments_defer_to_the_cpu_in_lab_documents() {
        let lab_rules = [
            Adjustment::Invert,
            Adjustment::Posterize { levels: 4 },
            Adjustment::Threshold { level: 0.5 },
            Adjustment::BrightnessContrast { brightness: 10.0, contrast: 5.0, legacy: false },
            Adjustment::GradientMap { stops: vec![(0.0, [0.0; 3]), (1.0, [1.0; 3])], reverse: false, dither: false },
        ];
        for adj in &lab_rules {
            assert!(!adjustment_on_gpu(adj, ColorMode::Lab), "{adj:?}");
            for mode in [ColorMode::Rgb, ColorMode::Grayscale] {
                assert!(adjustment_on_gpu(adj, mode), "{adj:?} in {mode:?}");
            }
            // CMYK documents invert, posterize and threshold their inks, on the CPU.
            let inks = matches!(adj, Adjustment::Invert | Adjustment::Posterize { .. } | Adjustment::Threshold { .. });
            assert_eq!(adjustment_on_gpu(adj, ColorMode::Cmyk), !inks, "{adj:?} in CMYK");
        }
        // Screen-colour adjustments stay on the GPU in Lab documents.
        assert!(adjustment_on_gpu(&Adjustment::Exposure { exposure: 1.0, offset: 0.0, gamma: 1.0 }, ColorMode::Lab));
    }

    #[test]
    fn native_cmyk_fills_fall_back_only_without_a_valid_cache() {
        use photocraft_doc::{FillCache, GradientStyle};
        for depth in photocraft_color::SampleType::ALL {
            for fill in [Fill::Solid(Color::WHITE), Fill::gradient(vec![(0.0, Color::BLACK), (1.0, Color::WHITE)], 0.0, 1.0, GradientStyle::Linear, false)] {
                let mut doc = Document::new("CMYK", photocraft_geom::Size::new(3, 2), photocraft_color::ColorMode::Cmyk, depth);
                doc.layers.push(Layer::new("Fill", LayerContent::Fill(fill.clone())));
                assert!(plan(&doc).unwrap_err().0.contains("native CMYK fill"));
                doc.layers[0].visible = false;
                assert!(plan(&doc).is_ok());
                doc.layers[0].visible = true;
                doc.layers[0].fill_cache = Some(FillCache { fill, surface: photocraft_raster::Surface::new(doc.pixel_format()) });
                assert!(plan(&doc).is_ok());
                doc.layers[0].fill_cache.as_mut().unwrap().fill = Fill::Solid(Color::TRANSPARENT);
                assert!(plan(&doc).is_err());
                doc.mode = photocraft_color::ColorMode::Rgb;
                assert!(plan(&doc).is_ok());
            }
        }
    }
    #[test]
    fn outer_glow_plan_knocks_out_see_through_fill() {
        use photocraft_doc::{Contour, FxCommon, GlowSource, GlowTechnique};

        for fill in [0.0, 0.4, 1.0] {
            let mut d = Document::with_background("glow", Size::new(64, 64), ColorMode::Rgb, SampleType::U8, Color::WHITE);
            let mut layer = Layer::raster("frame", d.pixel_format());
            layer.surface_mut().unwrap().fill_rect(Rect::new(16, 16, 48, 48), &[1.0, 0.0, 0.0, 1.0]);
            layer.fill_opacity = fill;
            layer.effects.items.push(Effect::OuterGlow(Glow {
                common: FxCommon::new(BlendMode::Normal, 1.0),
                paint: FxPaint::Color(Color::rgb(1.0, 1.0, 0.0)),
                technique: GlowTechnique::Softer,
                spread: 0.3,
                size: 8.0,
                contour: Contour::Linear,
                anti_alias: false,
                range: 0.5,
                jitter: 0.0,
                noise: 0.0,
                source: GlowSource::Edge,
            }));
            d.layers.push(layer);
            let planned = plan(&d).unwrap();
            let glow = planned
                .passes
                .iter()
                .find(|pass| {
                    pass.kernel == Kernel::FxPaint && pass.flags & F_KNOCKOUT != 0 && pass.color[0] > 0.9 && pass.color[1] > 0.9 && pass.color[2] < 0.1
                })
                .expect("outer glow paint pass");
            assert!((glow.extra[3] - (1.0 - fill)).abs() < 1e-6, "fill {fill}");
        }
    }

    #[test]
    fn color_lookup_with_an_overflowing_stored_size_uploads_no_table() {
        for size in [1 << 22, u32::MAX] {
            let a = Adjustment::ColorLookup { name: "x".into(), lut: Some(std::sync::Arc::new(vec![0.5; 24])), size, tetrahedral: false, dither: false };
            let (kind, _, rows) = adjustment_program(&a, Transfer::Srgb, SampleType::U8);
            assert_eq!((kind, rows.is_none()), (0, true), "size {size}");
        }
    }

    #[test]
    fn root_checkpoints_have_only_one_live_input_and_keep_clips_whole() {
        for mode in [BlendMode::Normal, BlendMode::PassThrough] {
            let mut d = Document::with_background("t", Size::new(32, 24), ColorMode::Rgb, SampleType::U8, Color::WHITE);
            let mut clip = Layer::new("clip", LayerContent::Fill(Fill::Solid(Color::rgba(0.2, 0.3, 0.7, 0.4))));
            clip.clipped = true;
            let mut group = Layer::group(
                "group",
                vec![
                    Layer::new("fill", LayerContent::Fill(Fill::Solid(Color::rgba(0.9, 0.1, 0.3, 0.6)))),
                    Layer::new("invert", LayerContent::Adjustment(Adjustment::Invert)),
                ],
            );
            group.blend = mode;
            d.layers.extend([group, clip, Layer::new("top", LayerContent::Adjustment(Adjustment::Invert))]);
            let plan = plan(&d).unwrap();
            assert_eq!(plan.checkpoints.iter().map(|c| c.layer_end).collect::<Vec<_>>(), vec![1, 3, 4]);
            for checkpoint in &plan.checkpoints {
                let mut written = std::collections::HashSet::from([checkpoint.slot]);
                for pass in &plan.passes[checkpoint.pass_end..] {
                    for input in [pass.a, pass.b, pass.c, pass.d].into_iter().flatten() {
                        assert!(written.contains(&input), "a suffix must not need another saved slot: {checkpoint:?}");
                    }
                    written.insert(pass.dst);
                }
                assert!(written.contains(&plan.root));
            }
        }
    }

    #[test]
    fn slots_are_recycled() {
        let mut d = Document::with_background("t", Size::new(8, 8), ColorMode::Rgb, SampleType::U8, Color::WHITE);
        for i in 0..10 {
            d.layers.push(Layer::raster(format!("l{i}"), d.pixel_format()));
        }
        let p = plan(&d).unwrap();
        // Empty layers draw nothing.
        assert_eq!(p.passes.len(), 1 + 2);
        for (i, l) in d.layers.iter_mut().skip(1).enumerate() {
            l.surface_mut().unwrap().fill_rect(Rect::new(i as i32 % 4, 0, 4, 8), &[0.0, 0.0, 1.0, 1.0]);
        }
        let p = plan(&d).unwrap();
        assert!(p.slots <= 3, "{} slots", p.slots);
        // The background, then each small layer over its bounds and copied back.
        assert_eq!(p.passes.len(), 1 + 2 + 10 * 3);
    }

    #[test]
    fn small_layers_composite_only_over_their_bounds() {
        let mut d = Document::with_background("t", Size::new(64, 64), ColorMode::Rgb, SampleType::U8, Color::WHITE);
        let mut l = Layer::raster("small", d.pixel_format());
        l.surface_mut().unwrap().fill_rect(Rect::new(10, 12, 20, 30), &[1.0, 0.0, 0.0, 1.0]);
        let mut group = Layer::group("g", vec![l]);
        group.blend = BlendMode::Normal;
        d.layers.push(group);
        let p = plan(&d).unwrap();
        let r = Rect::new(10, 12, 20, 30);
        // The group's content, its blend and the copy back are all limited to the layer.
        assert!(p.passes.iter().filter(|p| matches!(p.kernel, Kernel::Content | Kernel::Blend)).skip(2).all(|p| p.clip == Some(r)), "{:?}", p.passes);
        let last = p.passes.last().unwrap();
        assert_eq!((last.kernel, last.dst, last.clip), (Kernel::CopyRect, p.root, Some(r)));
        // A layer that covers most of the canvas composites everywhere.
        d.layers[1].children_mut().unwrap()[0].surface_mut().unwrap().fill_rect(Rect::new(0, 0, 60, 60), &[1.0, 0.0, 0.0, 1.0]);
        let p = plan(&d).unwrap();
        assert!(p.passes.iter().all(|p| p.clip.is_none()));
    }

    #[test]
    fn effects_are_planned_in_place() {
        let mut d = Document::with_background("t", Size::new(8, 8), ColorMode::Rgb, SampleType::U8, Color::WHITE);
        let mut l = Layer::raster("fx", d.pixel_format());
        l.surface_mut().unwrap().fill_rect(Rect::new(2, 2, 5, 5), &[1.0, 0.0, 0.0, 1.0]);
        l.effects.items.push(photocraft_doc::Effect::ColorOverlay { common: photocraft_doc::FxCommon::new(BlendMode::Normal, 1.0), color: Color::WHITE });
        l.effects.items.push(photocraft_doc::Effect::default_drop_shadow());
        d.layers.push(l);
        let p = plan(&d).unwrap();
        assert_eq!(p.fx.len(), 1);
        let m = photocraft_compose::effects::margin(&d.layers[1]);
        assert_eq!(p.fx[0].region, Rect::new(2, 2, 5, 5).inflate(m).intersect(&d.bounds().inflate(m)));
        // The effect result lands back in the backdrop slot.
        let last = p.passes.last().unwrap();
        assert_eq!(last.kernel, Kernel::CopyRect);
        assert_eq!(last.dst, p.root);
        assert!(p.slots <= 6, "{} slots", p.slots);
    }

    #[test]
    fn opaque_fill_layers_skip_what_they_cover() {
        let opaque = || Fill::gradient(vec![(0.0, Color::BLACK), (1.0, Color::WHITE)], 30.0, 1.0, photocraft_doc::GradientStyle::Linear, false);
        let mut d = Document::with_background("t", Size::new(8, 8), ColorMode::Rgb, SampleType::U8, Color::WHITE);
        for i in 0..4 {
            // Small painted layers (empty ones plan no passes at all).
            let mut l = Layer::raster(format!("l{i}"), d.pixel_format());
            l.surface_mut().unwrap().fill_rect(Rect::new(i, i, i + 2, i + 2), &[0.0, 0.0, 1.0, 1.0]);
            d.layers.push(l);
        }
        d.layers.push(Layer::new("g", LayerContent::Fill(opaque())));
        let mut top = Layer::raster("top", d.pixel_format());
        top.clipped = true;
        d.layers.push(top);
        // Clear, then the fill and its clipped layer: the background and four rasters are skipped.
        let skipped = plan(&d).unwrap().passes.len();
        let mut alone = d.clone();
        alone.layers.drain(..5);
        assert_eq!(skipped, plan(&alone).unwrap().passes.len());
        // Anything that lets the layers below show through keeps them.
        let see_through: [fn(&mut Layer); 6] = [
            |l| l.opacity = 0.99,
            |l| l.fill_opacity = 0.5,
            |l| l.blend = BlendMode::Multiply,
            |l| l.visible = false,
            |l| l.excluded_channels = 1,
            |l| {
                if let LayerContent::Fill(Fill::Gradient { opacity_stops, .. }) = &mut l.content {
                    opacity_stops.push((1.0, 0.5));
                }
            },
        ];
        for (k, f) in see_through.iter().enumerate() {
            let mut d2 = d.clone();
            f(&mut d2.layers[5]);
            assert!(plan(&d2).unwrap().passes.len() > skipped, "case {k}: the layers below are planned");
            assert!(!photocraft_compose::occludes_below(&d2.layers[5], ColorMode::Rgb), "case {k}");
        }
        let mut translucent = d.clone();
        translucent.layers[5].content = LayerContent::Fill(Fill::Solid(Color::rgba(1.0, 0.0, 0.0, 0.5)));
        assert!(!photocraft_compose::occludes_below(&translucent.layers[5], ColorMode::Rgb));
        assert!(photocraft_compose::occludes_below(&d.layers[5], ColorMode::Rgb));
    }

    #[test]
    fn blend_if_plans_on_the_gpu() {
        let mut d = Document::with_background("t", Size::new(8, 8), ColorMode::Rgb, SampleType::U8, Color::WHITE);
        let mut l = Layer::raster("bi", d.pixel_format());
        l.surface_mut().unwrap().fill_rect(photocraft_geom::Rect::new(0, 0, 4, 4), &[1.0, 0.0, 0.0, 1.0]);
        l.blend_if.set(0, [photocraft_doc::BlendRange { black: [40, 40], white: [255, 255] }, photocraft_doc::BlendRange::FULL]);
        d.layers.push(l.clone());
        let blend_if = |d: &Document| plan(d).unwrap().passes.iter().filter(|p| p.flags & F_BLEND_IF != 0).count();
        assert_eq!(blend_if(&d), 1);
        // Clipped layers too.
        l.clipped = true;
        d.layers[1].blend_if = Default::default();
        d.layers.push(l);
        assert_eq!(blend_if(&d), 1);
        // Outside RGB, Grayscale and Duotone the setting round-trips without applying.
        d.mode = ColorMode::Lab;
        assert_eq!(blend_if(&d), 0);
    }

    #[test]
    fn blend_if_params_pack_every_range_exactly() {
        let mut bi = photocraft_doc::BlendIf::default();
        bi.set(3, [photocraft_doc::BlendRange { black: [255, 255], white: [0, 1] }, photocraft_doc::BlendRange { black: [12, 200], white: [201, 254] }]);
        let f: Vec<f32> = blend_if_params(&bi).iter().flatten().copied().collect();
        // Entry 3 (B), this layer then underlying; unset entries are full ranges.
        assert_eq!(&f[12..16], &[255.0 * 256.0 + 255.0, 1.0, 12.0 * 256.0 + 200.0, 201.0 * 256.0 + 254.0]);
        assert_eq!(&f[0..2], &[0.0, 255.0 * 256.0 + 255.0]);
    }

    #[test]
    fn advanced_blending_falls_back_to_the_cpu() {
        use photocraft_doc::Knockout;
        let mut d = Document::with_background("t", Size::new(8, 8), ColorMode::Rgb, SampleType::U8, Color::WHITE);
        let mut l = Layer::raster("ko", d.pixel_format());
        l.surface_mut().unwrap().fill_rect(photocraft_geom::Rect::new(0, 0, 4, 4), &[1.0, 0.0, 0.0, 1.0]);
        d.layers.push(l.clone());
        assert!(plan(&d).is_ok(), "defaults stay on the GPU");
        // Switches with nothing to act on (no effects, no clipped layers) stay on the GPU too.
        d.layers[1].advanced.blend_interior = true;
        d.layers[1].advanced.blend_clipped = false;
        assert!(plan(&d).is_ok());
        d.layers[1].advanced.knockout = Knockout::Deep;
        assert!(plan(&d).unwrap_err().0.contains("Advanced Blending"));
        // Inside groups and clipping groups.
        d.layers[1].advanced.knockout = Knockout::None;
        let mut c = l.clone();
        c.clipped = true;
        d.layers.push(c.clone());
        assert!(plan(&d).is_err(), "clipped layers blended individually");
        d.layers[1].advanced.blend_clipped = true;
        assert!(plan(&d).is_ok());
        d.layers[2].advanced.knockout = Knockout::Shallow;
        assert!(plan(&d).is_err(), "clipped knockout");
        d.layers.truncate(1);
        let mut inner = l;
        inner.advanced.knockout = Knockout::Shallow;
        d.layers.push(Layer::group("g", vec![inner]));
        assert!(plan(&d).is_err(), "knockout inside a group");
    }
}
