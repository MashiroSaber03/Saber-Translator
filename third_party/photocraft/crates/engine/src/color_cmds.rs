//! Colour management commands and helpers: document profiles (Edit › Assign / Convert to
//! Profile), CMS-based Image › Mode conversions, soft proofing (View › Proof Setup / Proof
//! Colors / Gamut Warning) and the display transform the canvas applies.
//!
//! Document space: `Document::icc_profile` holds the document's ICC bytes (kept byte-exact
//! from PSD resource 1039 / PNG iCCP / JPEG APP2 / TIFF). `None` means untagged: the working
//! profile of the mode is assumed (sRGB, sGray, the built-in coated CMYK, Lab D50).
//!
//! Compositing stays in document space for RGB and gray documents. CMYK and Lab documents are
//! composited in sRGB after a per-layer conversion with the built-in profiles (see
//! `photocraft_color::convert`), so their [`composite_profile`] is sRGB; documents tagged with
//! a different CMYK profile therefore display approximately until compositing is mode-native.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use photocraft_cms::{Builtin, ColorSpace, GamutCheck, Intent, Lut3d, Profile, SampleKind, Transform};
use photocraft_color::{Color, ColorMode, PixelFormat, SampleType};
use photocraft_doc::{DocId, Document, Duotone, Effect, Fill, FxPaint, Layer, LayerContent};
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

// ------------------------------------------------------------------ state

/// View › Proof Setup settings.
#[derive(Clone, Debug, PartialEq)]
pub struct ProofSetup {
    /// What the user asked for (built-in id or path), for display.
    pub name: String,
    pub profile: Arc<Profile>,
    pub intent: Intent,
    pub bpc: bool,
    /// Simulate paper colour (absolute colorimetric proof → display, black ink simulated).
    pub simulate_paper: bool,
    /// What is simulated: the profile itself, one or more of its plates, an RGB display or a
    /// colour vision deficiency (see `proof_sim`).
    pub kind: crate::proof_sim::ProofKind,
}

impl Default for ProofSetup {
    fn default() -> Self {
        ProofSetup {
            name: "working-cmyk".into(),
            profile: Arc::new(Builtin::CoatedCmyk.profile().clone()),
            intent: Intent::RelativeColorimetric,
            bpc: true,
            simulate_paper: false,
            kind: crate::proof_sim::ProofKind::Profile,
        }
    }
}

/// Per-document proofing state the UI reads (see [`ColorState::proof`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProofView {
    pub setup: ProofSetup,
    /// View › Proof Colors.
    pub enabled: bool,
    /// View › Gamut Warning.
    pub gamut_warning: bool,
    pub gamut_threshold: f32,
}

type DisplayKey = (u64, u64, Option<(u64, Intent, bool, bool)>);

/// Colour management policy for documents whose embedded profile differs from the working
/// space (Edit › Color Settings).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Policy {
    /// Keep the embedded profile.
    #[default]
    Preserve,
    /// Convert the pixels to the working space.
    Convert,
    /// Discard the embedded profile (the document is assumed to be in the working space).
    Off,
}

impl Policy {
    pub fn id(self) -> &'static str {
        match self {
            Policy::Preserve => "preserve",
            Policy::Convert => "convert",
            Policy::Off => "off",
        }
    }
    pub fn parse(s: &str) -> Option<Policy> {
        match s {
            "preserve" | "preserveEmbedded" => Some(Policy::Preserve),
            "convert" | "convertToWorking" => Some(Policy::Convert),
            "off" => Some(Policy::Off),
            _ => None,
        }
    }
}

/// Edit › Color Settings: working spaces, colour management policies and conversion options.
/// Honoured when files are opened ([`Session::open_document`]), by Image › Mode conversions
/// and by "working" profile specs (Convert to Profile, Proof Setup).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ColorSettings {
    /// Working spaces: built-in profile ids (see `edit.profileInfo`) or `.icc` paths.
    pub working_rgb: String,
    pub working_cmyk: String,
    pub working_gray: String,
    pub policy_rgb: Policy,
    pub policy_cmyk: Policy,
    pub policy_gray: Policy,
    /// Ask what to do when an opened file's embedded profile differs from the working space.
    pub ask_on_mismatch: bool,
    pub ask_on_paste: bool,
    /// Ask when an opened file has no profile.
    pub ask_on_missing: bool,
    /// Conversion options: rendering intent and black point compensation.
    pub intent: String,
    pub bpc: bool,
    pub dither: bool,
    /// Advanced › "Blend Text Colors Using Gamma" (1 = off; Photoshop's default 1.45): type
    /// layers mix anti-aliased edges in this gamma (`photocraft_compose::psblend::set_text_gamma`).
    pub blend_text_gamma: f32,
    /// Monitor profile the canvas is displayed in: `auto` (the main display's profile when the
    /// platform supplies it, else sRGB), a built-in RGB profile id or an `.icc` path.
    pub monitor_profile: String,
}

impl Default for ColorSettings {
    fn default() -> Self {
        Self {
            working_rgb: Builtin::Srgb.id().into(),
            working_cmyk: Builtin::CoatedCmyk.id().into(),
            // Photoshop's default working gray is Dot Gain 20 %.
            working_gray: Builtin::DotGain20Compat.id().into(),
            policy_rgb: Policy::Preserve,
            policy_cmyk: Policy::Preserve,
            policy_gray: Policy::Preserve,
            ask_on_mismatch: true,
            ask_on_paste: true,
            ask_on_missing: false,
            intent: Intent::RelativeColorimetric.id().into(),
            bpc: true,
            dither: true,
            blend_text_gamma: photocraft_compose::psblend::TEXT_GAMMA,
            monitor_profile: "auto".into(),
        }
    }
}

impl ColorSettings {
    pub fn intent(&self) -> Intent {
        Intent::parse(&self.intent).unwrap_or(Intent::RelativeColorimetric)
    }
    /// Policy for documents in `mode`'s colour space (Lab documents are never converted).
    pub fn policy(&self, mode: ColorMode) -> Policy {
        match mode_space(mode) {
            ColorSpace::Cmyk => self.policy_cmyk,
            ColorSpace::Gray => self.policy_gray,
            ColorSpace::Rgb => self.policy_rgb,
            _ => Policy::Preserve,
        }
    }
    fn working_spec(&self, space: ColorSpace) -> Option<&str> {
        Some(match space {
            ColorSpace::Rgb => &self.working_rgb,
            ColorSpace::Cmyk => &self.working_cmyk,
            ColorSpace::Gray => &self.working_gray,
            _ => return None,
        })
    }
}

/// Check that each working space resolves to the right colour space and RGB can be both a
/// source and editing destination for the configured rendering intent.
pub fn validate_settings(c: &ColorSettings) -> std::result::Result<(), String> {
    for (space, spec) in [(ColorSpace::Rgb, &c.working_rgb), (ColorSpace::Cmyk, &c.working_cmyk), (ColorSpace::Gray, &c.working_gray)] {
        let p = resolve_profile(spec, None, space_mode(space)).map_err(|e| e.to_string())?;
        if p.color_space != space {
            return Err(format!("working space `{spec}` is {:?}, not {space:?}", p.color_space));
        }
        if space == ColorSpace::Rgb {
            p.pcs_to_device(c.intent()).map_err(|e| format!("working RGB space `{spec}` cannot be a destination for {}: {e}", c.intent().id()))?;
            p.device_to_pcs(c.intent()).map_err(|e| format!("working RGB space `{spec}` cannot be a source for {}: {e}", c.intent().id()))?;
        }
    }
    let m = c.monitor_profile.as_str();
    if !(m.is_empty() || m == "auto") {
        let p = resolve_profile(m, None, Some(ColorMode::Rgb)).map_err(|e| e.to_string())?;
        if p.color_space != ColorSpace::Rgb {
            return Err(format!("monitor profile `{m}` is {:?}, not RGB", p.color_space));
        }
    }
    if Intent::parse(&c.intent).is_none() {
        return Err(format!("unknown intent `{}` (perceptual|relative|saturation|absolute)", c.intent));
    }
    Ok(())
}

/// Session-wide colour settings and caches (`Session::color`).
#[derive(Default)]
pub struct ColorState {
    /// Edit › Color Settings (persisted with the preferences).
    pub settings: ColorSettings,
    proofs: HashMap<DocId, ProofView>,
    /// The displays and their ICC profiles as read by the platform, primary (menu-bar) display
    /// first (used when Color Settings › Monitor Profile is `auto`; none = sRGB display).
    pub displays: Vec<crate::display_color::Display>,
    /// Whether `displays` was read, or why not (see [`ColorState::set_displays`]).
    pub monitor_detection: crate::display_color::MonitorDetection,
    /// The display showing the main window (set by the UI each frame; `None` = unknown, the
    /// primary display). Other windows pass their own display to the `*_for` methods.
    pub main_display: Option<u32>,
    display_cache: Mutex<HashMap<DisplayKey, Arc<Transform>>>,
    pub(crate) display: crate::display_color::DisplayCaches,
    /// View › 32-bit Preview Options per document.
    pub hdr: HashMap<DocId, crate::proof_sim::HdrPreview>,
}

impl ColorState {
    /// The working profile of a mode from Color Settings (the built-in default when the
    /// setting does not resolve or cannot be both an RGB source and editing destination).
    pub fn working(&self, mode: ColorMode) -> Arc<Profile> {
        let space = mode_space(mode);
        self.settings
            .working_spec(space)
            .and_then(|spec| resolve_profile(spec, None, Some(mode)).ok())
            .filter(|p| {
                p.color_space == space
                    && (space != ColorSpace::Rgb || (p.pcs_to_device(self.settings.intent()).is_ok() && p.device_to_pcs(self.settings.intent()).is_ok()))
            })
            .unwrap_or_else(|| working_profile(mode))
    }

    /// Resolve a profile spec, reading "working…" specs from Color Settings.
    pub fn resolve(&self, spec: &str, doc: Option<&Document>, mode: Option<ColorMode>) -> Result<Arc<Profile>> {
        match spec {
            "working" | "default" => Ok(self.working(mode.or(doc.map(|d| d.mode)).unwrap_or(ColorMode::Rgb))),
            "working-cmyk" | "workingCmyk" => Ok(self.working(ColorMode::Cmyk)),
            "working-rgb" | "workingRgb" => Ok(self.working(ColorMode::Rgb)),
            "working-gray" | "workingGray" => Ok(self.working(ColorMode::Grayscale)),
            _ => resolve_profile(spec, doc, mode),
        }
    }

    /// Apply the colour management policy to a document being opened. Returns what happened
    /// (`action`: kept|converted|discarded|assigned|untagged) and whether to ask the user.
    ///
    /// An RGB file with no embedded profile whose EXIF ColorSpace tag says sRGB (as cameras write)
    /// is opened as sRGB, as in Photoshop; `ignore_exif` (Preferences › File Handling › Ignore
    /// EXIF Profile Tag) treats it as untagged instead. The report then has `exif: true`.
    pub fn open_policy(&self, doc: &mut Document, ignore_exif: bool) -> Value {
        let space = mode_space(doc.mode);
        if !matches!(space, ColorSpace::Rgb | ColorSpace::Cmyk | ColorSpace::Gray) {
            return json!({"action": "kept"});
        }
        let exif_srgb = space == ColorSpace::Rgb
            && !ignore_exif
            && doc.icc_profile.is_none()
            && doc.metadata.exif.as_deref().and_then(|e| photocraft_codecs::exif_color_space(e)) == Some(photocraft_codecs::EXIF_COLOR_SPACE_SRGB);
        if exif_srgb {
            doc.icc_profile = Some(Builtin::Srgb.profile().to_bytes());
        }
        let report = self.apply_open_policy(doc, space);
        if exif_srgb { merge(report, json!({"exif": true})) } else { report }
    }

    /// [`ColorState::open_policy`] once the document's profile is known.
    fn apply_open_policy(&self, doc: &mut Document, space: ColorSpace) -> Value {
        let working = self.working(doc.mode);
        let policy = self.settings.policy(doc.mode);
        let embedded = doc.icc_profile.as_ref().and_then(|b| profile_from_bytes(b).ok()).filter(|p| p.color_space == space);
        match embedded {
            Some(emb) => {
                // Compared by colour, not bytes: Photoshop's sRGB IEC61966-2.1 is our working sRGB.
                let mismatch = !emb.same_colors(&working);
                let base = json!({"embedded": emb.description, "working": working.description, "mismatch": mismatch, "policy": policy.id()});
                // 32-bit linear images (EXR/HDR are tagged linear sRGB or linear gray on import)
                // stay linear, as Photoshop keeps 32-bit documents in a linear version of the
                // working space.
                let linear_hdr =
                    doc.depth == SampleType::F32 && (emb.same_colors(Builtin::LinearSrgb.profile()) || emb.same_colors(photocraft_cms::builtin::linear_gray()));
                if !mismatch || linear_hdr {
                    return merge(base, json!({"action": "kept"}));
                }
                // An input-only profile can't be the document's space: nothing could be
                // composited into it (#3876). Preserve converts it to the working space instead.
                let input_only = !emb.can_be_destination();
                let action = match policy {
                    Policy::Preserve if !input_only => "kept",
                    Policy::Off => {
                        doc.icc_profile = None;
                        "discarded"
                    }
                    Policy::Preserve | Policy::Convert => match convert_document(doc, &working, self.settings.intent(), self.settings.bpc) {
                        Ok(()) => "converted",
                        Err(_) => "kept",
                    },
                };
                let report = merge(base, json!({"action": action, "ask": self.settings.ask_on_mismatch}));
                if input_only { merge(report, json!({"inputOnly": true})) } else { report }
            }
            None => {
                // Untagged files are assumed to be in the working space; tag them when it is not
                // the built-in default so they display (and save) as intended.
                let default = working_profile(doc.mode);
                let assign = policy != Policy::Off && working.content_hash() != default.content_hash();
                if assign {
                    doc.icc_profile = Some(working.to_bytes());
                }
                json!({"action": if assign { "assigned" } else { "untagged" }, "working": working.description, "policy": policy.id(), "ask": self.settings.ask_on_missing, "missing": true})
            }
        }
    }

    /// Whether opening `doc` asks about its profile first (Ask When Opening, for a mismatched or
    /// missing profile), and if so what the prompt shows: the report [`Self::open_policy`]
    /// would give, with `ask: true` and `action` the answer the policy suggests. Nothing is
    /// changed: the answer is applied with [`Self::resolve_open`] before the document opens.
    pub fn open_question(&self, doc: &Document) -> Option<Value> {
        let space = mode_space(doc.mode);
        if !matches!(space, ColorSpace::Rgb | ColorSpace::Cmyk | ColorSpace::Gray) {
            return None;
        }
        let working = self.working(doc.mode);
        let policy = self.settings.policy(doc.mode);
        let embedded = doc.icc_profile.as_ref().and_then(|b| profile_from_bytes(b).ok()).filter(|p| p.color_space == space);
        match embedded {
            Some(emb) => {
                let linear_hdr =
                    doc.depth == SampleType::F32 && (emb.same_colors(Builtin::LinearSrgb.profile()) || emb.same_colors(photocraft_cms::builtin::linear_gray()));
                if !self.settings.ask_on_mismatch || emb.same_colors(&working) || linear_hdr {
                    return None;
                }
                let action = match policy {
                    Policy::Preserve => "preserve",
                    Policy::Off => "discard",
                    Policy::Convert => "convert",
                };
                Some(
                    json!({"embedded": emb.description, "working": working.description, "mismatch": true, "policy": policy.id(), "action": action, "ask": true}),
                )
            }
            None if self.settings.ask_on_missing => {
                Some(json!({"working": working.description, "policy": policy.id(), "missing": true, "action": "preserve", "ask": true}))
            }
            None => None,
        }
    }

    /// Applies the answer to an [`Self::open_question`] to a document not opened yet:
    /// `preserve` (keep the embedded profile, or leave an untagged file as is), `convert` to the
    /// working space, `discard` the embedded profile, or `assignWorking` (untagged files).
    pub fn resolve_open(&self, doc: &mut Document, action: &str) -> Result<()> {
        match action {
            "preserve" | "useEmbedded" => Ok(()),
            "convert" | "convertToWorking" => convert_document(doc, &self.working(doc.mode), self.settings.intent(), self.settings.bpc),
            "discard" | "off" => {
                doc.icc_profile = None;
                Ok(())
            }
            "assignWorking" => {
                doc.icc_profile = Some(self.working(doc.mode).to_bytes());
                Ok(())
            }
            other => Err(EngineError::Other(format!("unknown answer `{other}` (preserve|convert|discard|assignWorking)"))),
        }
    }

    /// Proofing state of a document (defaults: coated CMYK, relative colorimetric + BPC, off).
    pub fn proof(&self, doc: DocId) -> ProofView {
        self.proofs.get(&doc).cloned().unwrap_or_else(|| ProofView { gamut_threshold: photocraft_cms::gamut::DEFAULT_THRESHOLD, ..Default::default() })
    }

    /// Borrow the document's proofing state without cloning its profile or allocating defaults.
    /// `None` means the unchanged defaults documented by [`Self::proof`].
    pub fn proof_ref(&self, doc: DocId) -> Option<&ProofView> {
        self.proofs.get(&doc)
    }

    pub(crate) fn proof_mut(&mut self, doc: DocId) -> &mut ProofView {
        self.proofs.entry(doc).or_insert_with(|| ProofView { gamut_threshold: photocraft_cms::gamut::DEFAULT_THRESHOLD, ..Default::default() })
    }

    /// Transform from the canvas texture values (the composite, `CanvasDisplay::source`) to
    /// the monitor, including the soft proof when View › Proof Colors is on. Cached per
    /// profile pair and proof settings.
    pub fn display_transform(&self, doc: &Document) -> Result<Arc<Transform>> {
        self.display_transform_for(doc, self.main_display)
    }

    /// [`ColorState::display_transform`] to `display`'s monitor profile.
    pub fn display_transform_for(&self, doc: &Document, display: Option<u32>) -> Result<Arc<Transform>> {
        let src = self.canvas_display_for(doc, display)?.source.clone();
        let dst = self.monitor_for(display);
        let pv = self.proof(doc.id);
        let proof = pv.enabled.then(|| (pv.setup.profile.content_hash(), pv.setup.intent, pv.setup.bpc, pv.setup.simulate_paper));
        let key = (src.content_hash(), dst.content_hash(), proof);
        if let Some(t) = self.display_cache.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
            return Ok(t.clone());
        }
        let t = if pv.enabled {
            Transform::proof(&src, &pv.setup.profile, &dst, pv.setup.intent, pv.setup.bpc, pv.setup.simulate_paper)
        } else {
            Transform::new(&src, &dst, crate::display_color::DISPLAY_INTENT, crate::display_color::DISPLAY_BPC)
        }
        .map_err(cms_err)?;
        let t = Arc::new(t);
        let mut c = self.display_cache.lock().unwrap_or_else(|e| e.into_inner());
        if c.len() > 32 {
            c.clear();
        }
        c.insert(key, t.clone());
        Ok(t)
    }

    /// The display transform as an `size³` RGBA 3D LUT (upload with
    /// [`Lut3d::to_rgba16f_bytes`] and apply in the canvas shader).
    pub fn display_lut(&self, doc: &Document, size: usize) -> Result<Lut3d> {
        self.display_lut_with(doc, size, true, self.main_display)
    }

    /// [`ColorState::display_lut`], with or without the 32-bit preview (exposure/gamma), for
    /// `display`.
    fn display_lut_with(&self, doc: &Document, size: usize, hdr: bool, display: Option<u32>) -> Result<Lut3d> {
        if let Some(lut) = crate::proof_sim::display_lut_with(self, doc, size, hdr, display)? {
            return Ok(lut);
        }
        let t = self.display_transform_for(doc, display)?;
        Ok(Lut3d::from_transform(&t, size))
    }

    /// View › 32-bit Preview Options of `doc` when they change the display (a 32-bit document
    /// with a non-default exposure or gamma).
    pub fn hdr_preview(&self, doc: &Document) -> Option<crate::proof_sim::HdrPreview> {
        if crate::proof_sim::hdr_active(self, doc) { self.hdr.get(&doc.id).copied() } else { None }
    }

    /// What the canvas should do for `doc`: `None` when the canvas values go to the screen
    /// unchanged (the document's display profile matches the monitor and neither Proof Colors,
    /// Gamut Warning nor a 32-bit preview is on), else an RGBA8 display LUT (`size`³, red fastest)
    /// mapping canvas texture values to the monitor, whose alpha is 255 where the colour is out
    /// of the proof gamut (only with Gamut Warning on).
    pub fn canvas_lut(&self, doc: &Document, size: usize) -> Result<Option<Vec<u8>>> {
        self.canvas_lut_with(doc, size, true, self.main_display)
    }

    /// [`ColorState::canvas_lut`] without the 32-bit preview, which the GPU canvas shader applies
    /// itself (see [`ColorState::hdr_preview`]) so that 32-bit values above 1.0, kept by its float
    /// texture, are exposed into range rather than clipped by the LUT's 0..1 domain.
    pub fn gpu_canvas_lut(&self, doc: &Document, size: usize) -> Result<Option<Vec<u8>>> {
        self.gpu_canvas_lut_for(doc, size, self.main_display)
    }

    /// [`ColorState::gpu_canvas_lut`] for a window on `display`.
    pub fn gpu_canvas_lut_for(&self, doc: &Document, size: usize, display: Option<u32>) -> Result<Option<Vec<u8>>> {
        self.canvas_lut_with(doc, size, false, display)
    }

    fn canvas_lut_with(&self, doc: &Document, size: usize, hdr: bool, on: Option<u32>) -> Result<Option<Vec<u8>>> {
        let size = size.max(2);
        let pv = self.proof(doc.id);
        let display = self.canvas_display_for(doc, on)?;
        if !(pv.enabled || pv.gamut_warning || hdr && crate::proof_sim::hdr_active(self, doc)) {
            return Ok(display.transform.as_ref().map(|t| Lut3d::from_transform(t, size).to_rgba8()));
        }
        let lut = self.display_lut_with(doc, size, hdr, on)?;
        let mut bytes = lut.to_rgba8();
        let check = if pv.gamut_warning { Some(GamutCheck::new(&display.source, &pv.setup.profile, pv.gamut_threshold).map_err(cms_err)?) } else { None };
        let s = (size - 1) as f32;
        for (i, px) in bytes.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let rgb = [(i % size) as f32 / s, ((i / size) % size) as f32 / s, (i / (size * size)) as f32 / s];
            px[3] = if check.as_ref().is_some_and(|c| c.out_of_gamut(&rgb)) { 255 } else { 0 };
        }
        Ok(Some(bytes))
    }
}

fn merge(mut a: Value, b: Value) -> Value {
    if let (Value::Object(a), Value::Object(b)) = (&mut a, b) {
        a.extend(b);
    }
    a
}

fn cms_err(e: photocraft_cms::CmsError) -> EngineError {
    EngineError::Other(format!("colour management: {e}"))
}

// ------------------------------------------------------------------ profiles

/// ICC colour space of a document mode.
pub fn mode_space(mode: ColorMode) -> ColorSpace {
    match mode {
        ColorMode::Grayscale | ColorMode::Bitmap | ColorMode::Duotone => ColorSpace::Gray,
        ColorMode::Cmyk => ColorSpace::Cmyk,
        ColorMode::Lab => ColorSpace::Lab,
        _ => ColorSpace::Rgb,
    }
}

fn space_mode(cs: ColorSpace) -> Option<ColorMode> {
    Some(match cs {
        ColorSpace::Rgb => ColorMode::Rgb,
        ColorSpace::Gray => ColorMode::Grayscale,
        ColorSpace::Cmyk => ColorMode::Cmyk,
        ColorSpace::Lab => ColorMode::Lab,
        _ => return None,
    })
}

/// Working (default) profile of a mode.
pub fn working_profile(mode: ColorMode) -> Arc<Profile> {
    Arc::new(photocraft_cms::builtin::default_for(mode_space(mode)).unwrap_or(Builtin::Srgb.profile()).clone())
}

/// Parses ICC bytes, caching by allocation so repeated lookups for the same document are free.
pub fn profile_from_bytes(bytes: &Arc<Vec<u8>>) -> std::result::Result<Arc<Profile>, photocraft_cms::CmsError> {
    type Cache = Mutex<Vec<(Weak<Vec<u8>>, Arc<Profile>)>>;
    static CACHE: std::sync::OnceLock<Cache> = std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    {
        let mut c = cache.lock().unwrap_or_else(|e| e.into_inner());
        c.retain(|(w, _)| w.strong_count() > 0);
        if let Some((_, p)) = c.iter().find(|(w, _)| w.as_ptr() == Arc::as_ptr(bytes)) {
            return Ok(p.clone());
        }
    }
    let p = Arc::new(Profile::parse(bytes)?);
    cache.lock().unwrap_or_else(|e| e.into_inner()).push((Arc::downgrade(bytes), p.clone()));
    Ok(p)
}

/// The document's profile: its embedded profile when it parses and matches the mode,
/// otherwise the mode's working profile.
pub fn document_profile(doc: &Document) -> Arc<Profile> {
    doc.icc_profile
        .as_ref()
        .and_then(|b| profile_from_bytes(b).ok())
        .filter(|p| p.color_space == mode_space(doc.mode))
        .unwrap_or_else(|| working_profile(doc.mode))
}

/// Profile describing the compositor's RGB output for this document (see the module docs).
pub fn composite_profile(doc: &Document) -> Arc<Profile> {
    match mode_space(doc.mode) {
        ColorSpace::Rgb => document_profile(doc),
        ColorSpace::Gray => {
            let p = document_profile(doc);
            // sGray is a TRC profile, so its RGB view always exists; sRGB is the last resort.
            p.gray_as_rgb().or_else(|| Builtin::SGray.profile().gray_as_rgb()).map(Arc::new).unwrap_or_else(|| Arc::new(Builtin::Srgb.profile().clone()))
        }
        _ => Arc::new(Builtin::Srgb.profile().clone()),
    }
}

/// Resolves a profile parameter: a built-in id/alias/description, `"working"` (with `mode`),
/// `"document"`, or a path to an `.icc`/`.icm` file.
pub fn resolve_profile(spec: &str, doc: Option<&Document>, mode: Option<ColorMode>) -> Result<Arc<Profile>> {
    match spec {
        "working" | "default" => {
            let m = mode.or(doc.map(|d| d.mode)).unwrap_or(ColorMode::Rgb);
            return Ok(working_profile(m));
        }
        "working-cmyk" | "workingCmyk" => return Ok(working_profile(ColorMode::Cmyk)),
        "working-rgb" | "workingRgb" => return Ok(working_profile(ColorMode::Rgb)),
        "working-gray" | "workingGray" => return Ok(working_profile(ColorMode::Grayscale)),
        "document" => return doc.map(document_profile).ok_or(EngineError::NoDocument),
        _ => {}
    }
    if let Some(b) = Builtin::from_id(spec) {
        return Ok(Arc::new(b.profile().clone()));
    }
    let lower = spec.to_ascii_lowercase();
    if lower.ends_with(".icc") || lower.ends_with(".icm") || spec.contains('/') || spec.contains('\\') {
        let bytes = std::fs::read(spec).map_err(|e| EngineError::Other(format!("cannot read profile `{spec}`: {e}")))?;
        return Profile::parse(&bytes).map(Arc::new).map_err(cms_err);
    }
    let ids: Vec<&str> = Builtin::ALL.iter().map(|b| b.id()).collect();
    Err(EngineError::Other(format!("unknown profile `{spec}` (built-ins: {}, or a path to an .icc file)", ids.join(", "))))
}

fn intent_param(p: &Value) -> Result<Intent> {
    intent_or(p, Intent::RelativeColorimetric)
}

fn intent_or(p: &Value, default: Intent) -> Result<Intent> {
    match p.get("intent").and_then(Value::as_str) {
        None => Ok(default),
        Some(s) => Intent::parse(s).ok_or_else(|| EngineError::Other(format!("unknown intent `{s}` (perceptual|relative|saturation|absolute)"))),
    }
}

fn bool_param(p: &Value, k: &str, d: bool) -> bool {
    p.get(k).and_then(Value::as_bool).unwrap_or(d)
}

fn profile_json(p: &Profile) -> Value {
    json!({
        "description": p.description,
        "colorSpace": format!("{:?}", p.color_space),
        "class": format!("{:?}", p.class),
        "version": format!("{}.{}", p.version.0, p.version.1 >> 4),
        "bytes": p.to_bytes().len(),
        "lut": p.a2b.iter().any(Option::is_some),
        "intents": Intent::ALL.iter().filter(|i| p.supports_intent(**i, false)).map(|i| i.id()).collect::<Vec<_>>(),
    })
}

// ------------------------------------------------------------------ conversion

/// Converts one surface's colour channels with `t` into `to` (alpha copied). Surfaces whose
/// model is not `from` fall back to the generic per-pixel conversion.
pub fn convert_surface(s: &Surface, from: ColorMode, to: PixelFormat, t: &Transform) -> Surface {
    let sf = s.format();
    if sf.mode != from || t.inputs() != sf.mode.color_channels() || t.outputs() != to.mode.color_channels() || sf.alpha != to.alpha {
        return s.convert(to);
    }
    let (ss, ds) = (sf.channels(), to.channels());
    let to = PixelFormat { sample: sf.sample, ..to };
    // Default (untouched) pixel.
    let dp = s.default_pixel();
    let mut dpo = vec![0.0f32; ds];
    t.convert_f32(&dp, ss, &mut dpo, ds, true);
    let mut out = Surface::with_default(to, &dpo);
    // Convert every distinct tile once, in one parallel pass into a staging buffer, then store
    // them. Coordinates that share a tile (a filled background) share the converted one (#4531).
    let mut distinct: Vec<std::sync::Arc<photocraft_raster::Tile>> = Vec::new();
    let mut slot: std::collections::HashMap<*const photocraft_raster::Tile, usize> = std::collections::HashMap::new();
    let tiles: Vec<(photocraft_geom::TileCoord, usize)> = s
        .tiles()
        .map(|(c, tile)| {
            let i = *slot.entry(std::sync::Arc::as_ptr(tile)).or_insert_with(|| {
                distinct.push(tile.clone());
                distinct.len() - 1
            });
            (*c, i)
        })
        .collect();
    let tile_px = (photocraft_geom::TILE_SIZE * photocraft_geom::TILE_SIZE) as usize;
    let dst_len = tile_px * to.bytes_per_pixel();
    let mut staging = vec![0u8; dst_len * distinct.len()];
    let jobs: Vec<(&[u8], &mut [u8])> = distinct.iter().map(|t| t.bytes()).zip(staging.chunks_mut(dst_len.max(1))).collect();
    let kind = match sf.sample {
        SampleType::U8 => SampleKind::U8,
        SampleType::U16 => SampleKind::U16,
        SampleType::F32 => SampleKind::F32,
    };
    t.convert_bytes_many(kind, jobs, ss, ds, true);
    let mut converted: Vec<Option<std::sync::Arc<photocraft_raster::Tile>>> = vec![None; distinct.len()];
    for (c, i) in tiles {
        if let Some(Some(tile)) = converted.get(i) {
            out.put_tiles([(c, tile.clone())]);
            continue;
        }
        let Some(bytes) = staging.chunks(dst_len.max(1)).nth(i) else { continue };
        out.tile_mut(c).bytes_mut().copy_from_slice(bytes);
        if let Some(slot) = converted.get_mut(i) {
            *slot = out.tile(c).cloned();
        }
    }
    out
}

/// Converts a colour value that is in `from` (the document's old mode) to the new mode.
fn convert_color(c: &mut Color, from: ColorMode, to: ColorMode, t: &Transform) {
    if c.mode != from {
        return;
    }
    let n = from.color_channels();
    let mut o = [0.0f32; 16];
    t.eval(&c.c[..n], &mut o);
    let m = to.color_channels();
    let mut nc = [0.0f32; 4];
    for (d, v) in nc.iter_mut().zip(&o[..m.min(4)]) {
        *d = v.clamp(0.0, 1.0);
    }
    c.mode = to;
    c.c = nc;
}

fn convert_fill(f: &mut Fill, cc: &mut dyn FnMut(&mut Color)) {
    match f {
        Fill::Solid(c) => cc(c),
        Fill::Gradient { stops, .. } => stops.iter_mut().for_each(|s| cc(&mut s.1)),
        Fill::Pattern { .. } => {}
    }
}

/// Applies `cc` to every colour value of `l` (fill, text, shape and effect colours), recursing
/// into groups.
fn convert_layer_colors(l: &mut Layer, cc: &mut dyn FnMut(&mut Color)) {
    let mut touched_fx = false;
    for e in &mut l.effects.items {
        touched_fx = true;
        match e {
            Effect::DropShadow(s) | Effect::InnerShadow(s) => cc(&mut s.color),
            Effect::OuterGlow(g) | Effect::InnerGlow(g) => paint_colors(&mut g.paint, cc),
            Effect::Stroke(s) => paint_colors(&mut s.paint, cc),
            Effect::ColorOverlay { color, .. } => cc(color),
            Effect::GradientOverlay { gradient, .. } => gradient.stops.iter_mut().for_each(|s| cc(&mut s.1)),
            Effect::Satin(s) => cc(&mut s.color),
            Effect::BevelEmboss(b) => {
                cc(&mut b.highlight_color);
                cc(&mut b.shadow_color);
            }
            Effect::PatternOverlay { .. } => {}
        }
    }
    let _ = touched_fx;
    match &mut l.content {
        LayerContent::Fill(f) => convert_fill(f, cc),
        LayerContent::Text(tx) => {
            cc(&mut tx.color);
            for r in &mut tx.runs {
                cc(&mut r.style.color);
            }
        }
        LayerContent::Shape(sh) => {
            if let Some(f) = &mut sh.fill {
                convert_fill(f, cc);
            }
            if let Some(s) = &mut sh.stroke {
                convert_fill(&mut s.paint, cc);
            }
        }
        LayerContent::Group(g) => {
            for c in &mut g.children {
                convert_layer_colors(c, cc);
            }
        }
        _ => {}
    }
}

fn paint_colors(p: &mut FxPaint, cc: &mut dyn FnMut(&mut Color)) {
    match p {
        FxPaint::Color(c) => cc(c),
        FxPaint::Gradient(g) => g.stops.iter_mut().for_each(|s| cc(&mut s.1)),
        FxPaint::Pattern { .. } => {}
    }
}

/// Converts every pixel layer (and cached text/shape/smart pixels and fill caches) of `doc`
/// from its profile to `dst`, switching the mode to `dst`'s colour space. Masks, alpha
/// channels and the selection are untouched; fill, text, shape and effect colours are
/// converted. A Duotone document going to a colour space starts from its ink colours
/// ([`ink_duotone`]). The document is tagged with `dst`.
pub fn convert_document(doc: &mut Document, dst: &Profile, intent: Intent, bpc: bool) -> Result<()> {
    let to_mode = space_mode(dst.color_space).ok_or_else(|| EngineError::Other(format!("cannot convert to a {:?} profile", dst.color_space)))?;
    if to_mode != ColorMode::Grayscale {
        ink_duotone(doc);
    }
    let src = document_profile(doc);
    let from_mode = doc.pixel_format().mode;
    let t = Transform::new(&src, dst, intent, bpc).map_err(cms_err)?;
    let fmt = PixelFormat::new(to_mode, doc.depth, true);
    crate::image_cmds::for_each_surface(&mut doc.layers, false, &mut |surf, _| {
        let f = PixelFormat { alpha: surf.format().alpha, ..fmt };
        *surf = convert_surface(surf, from_mode, f, &t);
    });
    for l in &mut doc.layers {
        convert_layer_colors(l, &mut |c| convert_color(c, from_mode, to_mode, &t));
        // Preserved PSD blocks describing colours in the old mode no longer apply.
    }
    doc.mode = to_mode;
    doc.icc_profile = Some(dst.to_bytes());
    // Leaving Indexed / Duotone: the palette and inks no longer apply.
    doc.color_table = None;
    doc.duotone = None;
    Ok(())
}

/// Turns a Duotone document into the sRGB document it displays as: every gray pixel and colour
/// value becomes its printed ink colour ([`Duotone::render`], sRGB like the inks), so converting
/// to a colour mode keeps the inks' look, as Photoshop converts them to their equivalents.
/// Other documents (and Duotone without inks) are left alone.
fn ink_duotone(doc: &mut Document) {
    if doc.mode != ColorMode::Duotone {
        return;
    }
    let Some(d) = doc.duotone.take_if(|d| !d.inks.is_empty()) else { return };
    crate::image_cmds::for_each_surface(&mut doc.layers, false, &mut |surf, _| *surf = ink_surface(surf, &d));
    for l in &mut doc.layers {
        convert_layer_colors(l, &mut |c| {
            if matches!(c.mode, ColorMode::Grayscale | ColorMode::Duotone) {
                let [r, g, b] = d.render(c.c[0]);
                *c = Color::rgba(r, g, b, c.alpha);
            }
        });
    }
    doc.mode = ColorMode::Rgb;
    doc.icc_profile = Some(Builtin::Srgb.profile().to_bytes());
}

/// The RGB surface of a gray surface's ink colours (same depth and alpha), one parallel pass
/// over the tiles. Non-gray surfaces are returned unchanged.
fn ink_surface(s: &Surface, d: &Duotone) -> Surface {
    use rayon::prelude::*;
    let sf = s.format();
    if sf.mode != ColorMode::Grayscale {
        return s.clone();
    }
    let to = PixelFormat { mode: ColorMode::Rgb, ..sf };
    let (ss, ds) = (sf.channels(), to.channels());
    let ink = |src: &[f32], out: &mut [f32]| {
        let [r, g, b] = d.render(src.first().copied().unwrap_or(0.0));
        for (o, v) in out.iter_mut().zip([r, g, b].into_iter().chain(src.get(1).copied())) {
            *o = v;
        }
    };
    let mut dp = vec![0.0f32; ds];
    ink(&s.default_pixel(), &mut dp);
    let mut out = Surface::with_default(to, &dp);
    let tiles: Vec<_> = s.tiles().map(|(c, t)| (*c, t.clone())).collect();
    let inked: Vec<Vec<u8>> = tiles
        .par_iter()
        .map(|(_, t)| {
            let src = t.bytes();
            let n = src.len() / sf.bytes_per_pixel().max(1);
            let mut dst = vec![0u8; n * to.bytes_per_pixel()];
            let (mut px, mut o) = ([0.0f32; 2], [0.0f32; 4]);
            for i in 0..n {
                for (k, v) in px.iter_mut().enumerate().take(ss) {
                    *v = photocraft_color::read_sample(src, sf.sample, i * ss + k);
                }
                ink(&px[..ss], &mut o[..ds]);
                for (k, v) in o.iter().enumerate().take(ds) {
                    photocraft_color::write_sample(&mut dst, to.sample, i * ds + k, *v);
                }
            }
            dst
        })
        .collect();
    for ((c, _), bytes) in tiles.iter().zip(inked) {
        let tile = out.tile_mut(*c).bytes_mut();
        if tile.len() == bytes.len() {
            tile.copy_from_slice(&bytes);
        }
    }
    out
}

/// Image › Mode › RGB/Grayscale/CMYK/Lab through the CMS. Params: `profile` (destination,
/// default the mode's working profile), `intent` (default relative), `bpc` (default true).
pub fn convert_mode(s: &mut Session, mode: ColorMode, p: &Value) -> Result<Value> {
    let doc = &s.active().ok_or(EngineError::NoDocument)?.doc;
    if doc.mode == ColorMode::Multichannel && mode != ColorMode::Multichannel {
        return crate::multichannel_cmds::convert_from(s, mode, p);
    }
    if doc.mode == mode && p.get("profile").is_none() {
        return Ok(Value::Null);
    }
    let dst = match p.get("profile").and_then(Value::as_str) {
        Some(spec) => s.color.resolve(spec, Some(doc), Some(mode))?,
        None => s.color.working(mode),
    };
    if space_mode(dst.color_space)
        != Some(match mode {
            ColorMode::Bitmap | ColorMode::Duotone => ColorMode::Grayscale,
            ColorMode::Indexed | ColorMode::Multichannel => ColorMode::Rgb,
            m => m,
        })
    {
        return Err(EngineError::Other(format!("profile `{}` is {:?}, not {mode:?}", dst.description, dst.color_space)));
    }
    let intent = intent_or(p, s.color.settings.intent())?;
    let bpc = bool_param(p, "bpc", s.color.settings.bpc);
    s.edit("Mode Change", |doc, _| convert_document(doc, &dst, intent, bpc))?;
    Ok(json!({ "profile": dst.description, "intent": intent.id(), "bpc": bpc }))
}

// ------------------------------------------------------------------ gamut

/// Out-of-gamut mask of the document composite against `setup`'s profile: one byte per
/// canvas pixel (255 = out of gamut, transparent pixels never are), plus the count.
pub fn gamut_mask(doc: &Document, setup: &ProofSetup, threshold: f32) -> Result<(Vec<u8>, usize)> {
    let check = GamutCheck::new(&composite_profile(doc), &setup.profile, threshold).map_err(cms_err)?;
    let buf = photocraft_compose::flatten(doc);
    let mut count = 0;
    let mask = buf
        .px
        .iter()
        .map(|p| {
            if p[3] > 0.0 && check.out_of_gamut(&p[..3]) {
                count += 1;
                255
            } else {
                0
            }
        })
        .collect();
    Ok((mask, count))
}

// ------------------------------------------------------------------ commands

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

fn always(_: &Session) -> std::result::Result<(), String> {
    Ok(())
}

fn assign_profile(s: &mut Session, p: &Value) -> Result<Value> {
    let spec =
        p.get("profile").and_then(Value::as_str).ok_or_else(|| EngineError::BadParams { cmd: "edit.assignProfile".into(), msg: "missing `profile`".into() })?;
    let doc = &s.active().ok_or(EngineError::NoDocument)?.doc;
    let bytes = if spec == "none" {
        None
    } else {
        let prof = s.color.resolve(spec, Some(doc), None)?;
        if prof.color_space != mode_space(doc.mode) {
            return Err(EngineError::Other(format!("profile `{}` is {:?}; the document is {:?}", prof.description, prof.color_space, doc.mode)));
        }
        Some(prof.to_bytes())
    };
    let desc = match &bytes {
        Some(b) => profile_from_bytes(b).map(|p| p.description.clone()).unwrap_or_default(),
        None => "none".into(),
    };
    s.edit("Assign Profile", |doc, _| {
        doc.icc_profile = bytes;
        Ok(())
    })?;
    Ok(json!({ "profile": desc }))
}

fn convert_to_profile(s: &mut Session, p: &Value) -> Result<Value> {
    let spec = p
        .get("profile")
        .and_then(Value::as_str)
        .ok_or_else(|| EngineError::BadParams { cmd: "edit.convertToProfile".into(), msg: "missing `profile`".into() })?;
    let doc = &s.active().ok_or(EngineError::NoDocument)?.doc;
    let dst = s.color.resolve(spec, Some(doc), None)?;
    let intent = intent_or(p, s.color.settings.intent())?;
    let bpc = bool_param(p, "bpc", s.color.settings.bpc);
    s.edit("Convert to Profile", |doc, _| convert_document(doc, &dst, intent, bpc))?;
    let d = &s.active().ok_or(EngineError::NoDocument)?.doc;
    Ok(json!({ "profile": dst.description, "mode": format!("{:?}", d.mode), "intent": intent.id(), "bpc": bpc }))
}

fn profile_info(s: &mut Session, p: &Value) -> Result<Value> {
    let builtins: Vec<Value> =
        Builtin::ALL.iter().map(|b| json!({ "id": b.id(), "description": b.description(), "colorSpace": format!("{:?}", b.profile().color_space) })).collect();
    let doc = s.active().map(|d| d.doc.clone());
    let info = match p.get("profile").and_then(Value::as_str) {
        Some(spec) => {
            let prof = resolve_profile(spec, doc.as_deref(), None)?;
            Some(profile_json(&prof))
        }
        None => doc.as_deref().map(|d| {
            let mut j = profile_json(&document_profile(d));
            j["embedded"] = json!(d.icc_profile.is_some());
            j
        }),
    };
    Ok(json!({ "profile": info, "builtins": builtins }))
}

fn proof_setup(s: &mut Session, p: &Value) -> Result<Value> {
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let (id, doc) = (st.doc.id, st.doc.clone());
    let name = p.get("profile").and_then(Value::as_str).unwrap_or("working-cmyk").to_string();
    let profile = s.color.resolve(&name, Some(&doc), Some(ColorMode::Cmyk))?;
    let intent = intent_param(p)?;
    let setup = ProofSetup {
        name,
        profile,
        intent,
        bpc: bool_param(p, "bpc", true),
        simulate_paper: bool_param(p, "simulatePaper", false),
        kind: crate::proof_sim::ProofKind::Profile,
    };
    let pv = s.color.proof_mut(id);
    pv.setup = setup.clone();
    Ok(
        json!({ "profile": setup.profile.description, "intent": setup.intent.id(), "bpc": setup.bpc, "simulatePaper": setup.simulate_paper, "proofColors": pv.enabled }),
    )
}

fn proof_colors(s: &mut Session, p: &Value) -> Result<Value> {
    let id = s.active().ok_or(EngineError::NoDocument)?.doc.id;
    let pv = s.color.proof_mut(id);
    pv.enabled = p.get("on").and_then(Value::as_bool).unwrap_or(!pv.enabled);
    Ok(json!({ "on": pv.enabled, "profile": pv.setup.profile.description }))
}

fn gamut_warning(s: &mut Session, p: &Value) -> Result<Value> {
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let doc = st.doc.clone();
    let pv = s.color.proof_mut(doc.id);
    pv.gamut_warning = p.get("on").and_then(Value::as_bool).unwrap_or(!pv.gamut_warning);
    if let Some(t) = p.get("threshold").and_then(Value::as_f64) {
        pv.gamut_threshold = t.max(0.0) as f32;
    }
    let mut setup = pv.setup.clone();
    let (on, threshold) = (pv.gamut_warning, pv.gamut_threshold);
    if let Some(spec) = p.get("profile").and_then(Value::as_str) {
        setup.profile = resolve_profile(spec, Some(&doc), Some(ColorMode::Cmyk))?;
    }
    let (_, count) = gamut_mask(&doc, &setup, threshold)?;
    let total = doc.size.width as usize * doc.size.height as usize;
    Ok(json!({ "on": on, "profile": setup.profile.description, "threshold": threshold, "outOfGamut": count, "fraction": count as f64 / total.max(1) as f64 }))
}

/// Edit › Color Settings: read or change working spaces, policies and conversion options.
fn color_settings(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "edit.colorSettings";
    let mut next = s.color.settings.clone();
    if p.get("reset").and_then(Value::as_bool) == Some(true) {
        next = ColorSettings::default();
    }
    let str_of = |k: &str| p.get(k).and_then(Value::as_str).map(str::to_string);
    if let Some(v) = str_of("workingRgb") {
        next.working_rgb = v;
    }
    if let Some(v) = str_of("workingCmyk") {
        next.working_cmyk = v;
    }
    if let Some(v) = str_of("workingGray") {
        next.working_gray = v;
    }
    if let Some(v) = str_of("monitorProfile") {
        next.monitor_profile = v;
    }
    for (k, slot) in [("policyRgb", &mut next.policy_rgb), ("policyCmyk", &mut next.policy_cmyk), ("policyGray", &mut next.policy_gray)] {
        if let Some(v) = p.get(k).and_then(Value::as_str) {
            *slot = Policy::parse(v).ok_or_else(|| EngineError::BadParams { cmd: cmd.into(), msg: format!("`{k}` must be preserve|convert|off") })?;
        }
    }
    for (k, slot) in [
        ("askOnMismatch", &mut next.ask_on_mismatch),
        ("askOnPaste", &mut next.ask_on_paste),
        ("askOnMissing", &mut next.ask_on_missing),
        ("bpc", &mut next.bpc),
        ("dither", &mut next.dither),
    ] {
        if let Some(v) = p.get(k).and_then(Value::as_bool) {
            *slot = v;
        }
    }
    if let Some(v) = str_of("intent") {
        next.intent = Intent::parse(&v).map(|i| i.id().to_string()).unwrap_or(v);
    }
    match p.get("blendTextGamma") {
        Some(Value::Bool(false)) => next.blend_text_gamma = 1.0,
        Some(Value::Bool(true)) => next.blend_text_gamma = photocraft_compose::psblend::TEXT_GAMMA,
        Some(v) => {
            let g = v
                .as_f64()
                .filter(|g| (1.0..=2.2).contains(g))
                .ok_or_else(|| EngineError::BadParams { cmd: cmd.into(), msg: "`blendTextGamma` must be 1.0..2.2 or a bool".into() })?;
            next.blend_text_gamma = g as f32;
        }
        None => {}
    }
    validate_settings(&next).map_err(|msg| EngineError::BadParams { cmd: cmd.into(), msg })?;
    if next != s.color.settings {
        photocraft_compose::psblend::set_text_gamma(next.blend_text_gamma);
        s.color.settings = next;
        // Persisted with the preferences.
        s.prefs.edit(|_| ());
    }
    let c = &s.color.settings;
    let desc = |m: ColorMode| s.color.working(m).description.clone();
    Ok(json!({
        "settings": c,
        "working": {"rgb": desc(ColorMode::Rgb), "cmyk": desc(ColorMode::Cmyk), "gray": desc(ColorMode::Grayscale)},
        "monitor": s.color.monitor().description,
        "monitorDetected": !s.color.displays.is_empty(),
        "monitorStatus": s.color.monitor_status(),
        "displays": s.color.display_statuses(),
    }))
}

/// Resolve an embedded-profile mismatch reported when opening (the "Embedded Profile
/// Mismatch" dialog): keep the embedded profile, convert to the working space, or discard it.
fn profile_mismatch(s: &mut Session, p: &Value) -> Result<Value> {
    let action = p.get("action").and_then(Value::as_str).unwrap_or("preserve");
    let mode = s.active().ok_or(EngineError::NoDocument)?.doc.mode;
    match action {
        "preserve" | "useEmbedded" => Ok(json!({"action": "kept"})),
        "convert" | "convertToWorking" => {
            let dst = s.color.working(mode);
            let (intent, bpc) = (s.color.settings.intent(), s.color.settings.bpc);
            s.edit("Convert to Working RGB", |doc, _| convert_document(doc, &dst, intent, bpc))?;
            Ok(json!({"action": "converted", "profile": dst.description}))
        }
        "discard" | "off" => {
            s.edit("Discard Profile", |doc, _| {
                doc.icc_profile = None;
                Ok(())
            })?;
            Ok(json!({"action": "discarded"}))
        }
        "assignWorking" => {
            let w = s.color.working(mode);
            s.edit("Assign Profile", |doc, _| {
                doc.icc_profile = Some(w.to_bytes());
                Ok(())
            })?;
            Ok(json!({"action": "assigned", "profile": w.description}))
        }
        other => Err(EngineError::BadParams {
            cmd: "color.profileMismatch".into(),
            msg: format!("unknown action `{other}` (preserve|convert|discard|assignWorking)"),
        }),
    }
}

impl Session {
    /// Add a document read from a file, applying the Color Settings policy first (preserve,
    /// convert to the working space, or discard the embedded profile). Returns the document
    /// index and a report (`action`, `mismatch`, `ask`) the UI can turn into a prompt.
    pub fn open_document(&mut self, mut doc: Document, path: Option<String>) -> (usize, Value) {
        let report = self.color.open_policy(&mut doc, self.prefs().file_handling.ignore_exif_profile_tag);
        (self.add_document(doc, path), report)
    }
}

macro_rules! spec {
    ($id:literal, $label:literal, [$($m:literal),*], $params:literal, $en:expr, $run:expr, $journal:expr) => {
        CommandSpec { id: $id, label: $label, menu: &[$($m),*], shortcut: None, params: $params, enabled: $en, run: $run, journal: $journal }
    };
    ($id:literal, $label:literal, [$($m:literal),*], $params:literal, $en:expr, $run:expr, $journal:expr, $sc:expr) => {
        CommandSpec { id: $id, label: $label, menu: &[$($m),*], shortcut: Some($sc), params: $params, enabled: $en, run: $run, journal: $journal }
    };
}

/// Colour management command specs.
pub fn specs() -> Vec<CommandSpec> {
    vec![
        spec!(
            "edit.assignProfile",
            "Assign Profile…",
            ["Edit"],
            r##"{"profile":"working|srgb|display-p3|adobe-rgb-compat|prophoto-compat|linear-srgb|rec2020|gray-gamma-2.2|sgray|dot-gain-20-compat|lab-d50|coated-cmyk|none" (or a path to an .icc file)}"##,
            has_doc,
            assign_profile,
            true
        ),
        spec!(
            "edit.convertToProfile",
            "Convert to Profile…",
            ["Edit"],
            r##"{"profile":"srgb|display-p3|adobe-rgb-compat|prophoto-compat|linear-srgb|rec2020|gray-gamma-2.2|sgray|lab-d50|coated-cmyk|working" (or a path to an .icc file),"intent":"perceptual|relative|saturation|absolute"="relative","bpc":bool=true}"##,
            has_doc,
            convert_to_profile,
            true
        ),
        spec!(
            "edit.colorSettings",
            "Color Settings…",
            ["Edit"],
            r##"{"workingRgb":"srgb|display-p3|adobe-rgb-compat|prophoto-compat|linear-srgb|rec2020","workingCmyk":"coated-cmyk","workingGray":"dot-gain-20-compat|sgray|gray-gamma-2.2","policyRgb":"preserve|convert|off","policyCmyk":"preserve|convert|off","policyGray":"preserve|convert|off","askOnMismatch":bool=true,"askOnPaste":bool=true,"askOnMissing":bool=false,"intent":"relative|perceptual|saturation|absolute","blendTextGamma":1.0..2.2|bool=1.45,"bpc":bool=true,"dither":bool=true,"monitorProfile":"auto|srgb|display-p3|adobe-rgb-compat|prophoto-compat|rec2020","reset":bool=false} (working spaces and the monitor profile also accept .icc paths; monitor `auto` = the main display's profile when the platform provides it, else sRGB; the reply's `monitorStatus` says which profile is in use and why: source auto|manual|fallback, reason)"##,
            always,
            color_settings,
            true,
            "Cmd+Shift+K"
        ),
        spec!(
            "color.profileMismatch",
            "Embedded Profile Mismatch",
            [],
            r##"{"action":"preserve|convert|discard|assignWorking"}"##,
            has_doc,
            profile_mismatch,
            true
        ),
        spec!("edit.profileInfo", "Profile Info", [], r##"{"profile":"<builtin id>|document|/path/to/profile.icc"=document}"##, always, profile_info, false),
        spec!(
            "view.proofSetup",
            "Proof Setup…",
            [],
            r##"{"profile":"working-cmyk|srgb|display-p3|adobe-rgb-compat|prophoto-compat|linear-srgb|rec2020|gray-gamma-2.2|sgray|lab-d50|coated-cmyk" (or a path to an .icc file)="working-cmyk","intent":"perceptual|relative|saturation|absolute"="relative","bpc":bool=true,"simulatePaper":bool=false}"##,
            has_doc,
            proof_setup,
            false
        ),
        spec!("view.proofColors", "Proof Colors", ["View"], r##"{"on":bool=toggle}"##, has_doc, proof_colors, false, "Cmd+Y"),
        spec!(
            "view.gamutWarning",
            "Gamut Warning",
            ["View"],
            r##"{"on":bool=toggle,"threshold":deltaE=4,"profile":"<proof profile override>"}"##,
            has_doc,
            gamut_warning,
            false,
            "Cmd+Shift+Y"
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_geom::Rect;

    /// Image › Mode to CMYK keeps a filled background's shared tile shared (#4531): one
    /// converted tile for the whole grid, the same colour everywhere.
    #[test]
    fn mode_conversion_keeps_shared_tiles_shared() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 2048, "height": 1024, "background": "white"})).unwrap();
        let distinct = |s: &Session| {
            let doc = &s.active().unwrap().doc;
            let surf = doc.layers[0].surface().unwrap();
            let mut seen: Vec<_> = surf.tiles().map(|(_, t)| std::sync::Arc::as_ptr(t)).collect();
            seen.sort();
            seen.dedup();
            (seen.len(), surf.tile_count())
        };
        let (before, count) = distinct(&s);
        s.execute("image.mode.cmyk", json!({})).unwrap();
        let (after, count_after) = distinct(&s);
        assert_eq!((after, count_after), (before, count), "{before} distinct of {count} before");
        let doc = &s.active().unwrap().doc;
        let surf = doc.layers[0].surface().unwrap();
        assert_eq!(surf.pixel(5, 5), surf.pixel(2000, 1000), "the same converted white everywhere");
    }

    fn session(mode: &str, depth: u32) -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 64, "height": 32, "mode": mode, "depth": depth})).unwrap();
        s
    }

    fn paint_rgb(s: &mut Session) {
        s.execute("layer.new.layer", json!({})).unwrap();
        s.edit("paint", |doc, active| {
            let l = doc.layer_mut(active.unwrap()).unwrap();
            let surf = l.surface_mut().unwrap();
            surf.fill_rect(Rect::new(0, 0, 16, 16), &[1.0, 0.0, 0.0, 1.0]);
            surf.fill_rect(Rect::new(16, 0, 32, 16), &[0.0, 0.0, 1.0, 1.0]);
            surf.fill_rect(Rect::new(32, 0, 48, 16), &[0.5, 0.5, 0.5, 0.5]);
            l.mask = Some(photocraft_doc::LayerMask::reveal_all());
            Ok(())
        })
        .unwrap();
    }

    fn doc(s: &Session) -> &Document {
        &s.active().unwrap().doc
    }

    #[test]
    fn mode_conversions_all_depths_with_undo() {
        for depth in [8, 16, 32] {
            let mut s = session("rgb", depth);
            paint_rgb(&mut s);
            let before = doc(&s).layers[1].surface().unwrap().pixel(40, 8);
            s.execute("image.mode.cmyk", json!({})).unwrap();
            let d = doc(&s);
            assert_eq!(d.mode, ColorMode::Cmyk);
            assert_eq!(d.icc_profile.as_deref(), Some(&*Builtin::CoatedCmyk.profile().to_bytes()), "tagged with the destination");
            let l = &d.layers[1];
            assert_eq!(l.surface().unwrap().format().mode, ColorMode::Cmyk);
            assert_eq!(l.mask.as_ref().unwrap().surface.format().mode, ColorMode::Grayscale, "masks untouched");
            let gray = l.surface().unwrap().pixel(40, 8);
            assert!((gray[4] - 0.5).abs() < 1e-2, "alpha kept: {gray:?}");
            assert!(gray[..4].iter().all(|v| (0.0..=1.0).contains(v)));
            // Red goes to M+Y heavy separation.
            let red = l.surface().unwrap().pixel(8, 8);
            assert!(red[1] > 0.7 && red[2] > 0.7 && red[0] < 0.2, "{depth}: {red:?}");
            // Background white → no ink.
            let bg = d.layers[0].surface().unwrap().pixel(60, 30);
            assert!(bg[..4].iter().all(|v| *v < 0.01), "{bg:?}");
            s.execute("image.mode.lab", json!({})).unwrap();
            let lab = doc(&s).layers[1].surface().unwrap().pixel(40, 8);
            assert!((lab[1] - 128.0 / 255.0).abs() < 0.03 && (lab[2] - 128.0 / 255.0).abs() < 0.03, "{lab:?}");
            s.execute("image.mode.grayscale", json!({"intent": "perceptual"})).unwrap();
            s.execute("image.mode.rgb", json!({})).unwrap();
            let after = doc(&s).layers[1].surface().unwrap().pixel(40, 8);
            assert!((after[0] - before[0]).abs() < 0.04 && (after[0] - after[2]).abs() < 0.01, "{depth}: {before:?} -> {after:?}");
            for _ in 0..4 {
                s.execute("edit.undo", json!({})).unwrap();
            }
            assert_eq!(doc(&s).mode, ColorMode::Rgb);
            assert_eq!(doc(&s).layers[1].surface().unwrap().pixel(40, 8), before);
        }
    }

    #[test]
    fn rgb_lab_rgb_roundtrip_16bit() {
        let mut s = session("rgb", 16);
        paint_rgb(&mut s);
        let before = doc(&s).layers[1].surface().unwrap().pixel(8, 8);
        s.execute("image.mode.lab", json!({})).unwrap();
        s.execute("image.mode.rgb", json!({})).unwrap();
        let after = doc(&s).layers[1].surface().unwrap().pixel(8, 8);
        for k in 0..3 {
            assert!((before[k] - after[k]).abs() < 1.0 / 255.0, "{before:?} -> {after:?}");
        }
    }

    #[test]
    fn assign_and_convert_profile() {
        let mut s = session("rgb", 8);
        paint_rgb(&mut s);
        s.execute("edit.assignProfile", json!({"profile": "display-p3"})).unwrap();
        assert_eq!(document_profile(doc(&s)).description, "Display P3");
        let px = doc(&s).layers[1].surface().unwrap().pixel(8, 8);
        assert_eq!(px, vec![1.0, 0.0, 0.0, 1.0], "assign does not change pixels");
        assert!(s.execute("edit.assignProfile", json!({"profile": "coated-cmyk"})).is_err(), "wrong colour space");
        s.execute("edit.convertToProfile", json!({"profile": "srgb", "intent": "relative"})).unwrap();
        let px = doc(&s).layers[1].surface().unwrap().pixel(8, 8);
        assert!(px[0] > 0.99 && px[1] < 0.01, "P3 red clips to sRGB red: {px:?}");
        s.execute("edit.convertToProfile", json!({"profile": "coated-cmyk", "intent": "perceptual"})).unwrap();
        assert_eq!(doc(&s).mode, ColorMode::Cmyk);
        s.execute("edit.assignProfile", json!({"profile": "none"})).unwrap();
        assert!(doc(&s).icc_profile.is_none());
        let info = s.execute("edit.profileInfo", json!({})).unwrap();
        assert_eq!(info["profile"]["colorSpace"], "Cmyk");
        assert!(info["builtins"].as_array().unwrap().len() >= 10);
        assert!(s.execute("edit.convertToProfile", json!({"profile": "nope"})).is_err());
    }

    #[test]
    fn fill_and_effect_colours_follow_the_mode() {
        let mut s = session("rgb", 8);
        s.edit("fill layer", |doc, _| {
            let mut l = Layer::new("Fill", LayerContent::Fill(Fill::Solid(Color::rgb(0.0, 0.0, 1.0))));
            l.effects.items.push(Effect::default_drop_shadow());
            doc.layers.push(l);
            Ok(())
        })
        .unwrap();
        s.execute("image.mode.cmyk", json!({})).unwrap();
        let l = &doc(&s).layers[1];
        let LayerContent::Fill(Fill::Solid(c)) = &l.content else { panic!() };
        assert_eq!(c.mode, ColorMode::Cmyk);
        assert!(c.c[0] > 0.6, "{c:?}");
        let Effect::DropShadow(sh) = &l.effects.items[0] else { panic!() };
        assert_eq!(sh.color.mode, ColorMode::Cmyk);
    }

    #[test]
    fn proofing_state_and_gamut_warning() {
        let mut s = session("rgb", 8);
        paint_rgb(&mut s);
        let id = doc(&s).id;
        assert!(!s.color.proof(id).enabled);
        s.execute("view.proofSetup", json!({"profile": "coated-cmyk", "intent": "perceptual", "simulatePaper": true})).unwrap();
        let pv = s.color.proof(id);
        assert_eq!((pv.setup.intent, pv.setup.simulate_paper), (Intent::Perceptual, true));
        s.execute("view.proofColors", json!({})).unwrap();
        assert!(s.color.proof(id).enabled);
        let d = doc(&s).clone();
        let t = s.color.display_transform(&d).unwrap();
        let mut o = [0.0f32; 3];
        t.eval(&[1.0, 1.0, 1.0], &mut o);
        assert!(o[0] < 0.97, "paper simulation: {o:?}");
        let lut = s.color.display_lut(&d, 17).unwrap();
        assert_eq!(lut.data.len(), 17 * 17 * 17);
        s.execute("view.proofColors", json!({"on": false})).unwrap();
        let t2 = s.color.display_transform(&d).unwrap();
        t2.eval(&[1.0, 1.0, 1.0], &mut o);
        assert!(o.iter().all(|v| (v - 1.0).abs() < 1e-3), "no proof: identity-ish {o:?}");
        // Saturated sRGB blue is outside coated CMYK; mid gray and white are inside.
        let r = s.execute("view.gamutWarning", json!({"on": true})).unwrap();
        assert!(r["on"].as_bool().unwrap());
        let (mask, count) = gamut_mask(&d, &s.color.proof(id).setup, 4.0).unwrap();
        assert_eq!(r["outOfGamut"].as_u64().unwrap() as usize, count);
        let w = d.size.width as usize;
        assert_eq!(mask[8 * w + 20], 255, "blue out of gamut");
        assert_eq!(mask[8 * w + 40], 0, "gray in gamut");
        assert_eq!(mask[30 * w + 60], 0, "white in gamut");
    }

    #[test]
    fn gray_and_cmyk_composite_profiles() {
        let s = session("gray", 8);
        let p = composite_profile(doc(&s));
        assert_eq!(p.color_space, ColorSpace::Rgb);
        let t = s.color.display_transform(doc(&s)).unwrap();
        let mut o = [0.0f32; 3];
        t.eval(&[0.5, 0.5, 0.5], &mut o);
        // The default working gray (Dot Gain 20 %-compatible, gamma 1.732) shows gray 0.5 as the
        // sRGB value of linear 0.5^1.732, as Photoshop does: lighter than sGray's 0.5.
        let want = photocraft_color::convert::linear_to_srgb(0.5f32.powf(1.732));
        assert!(o.iter().all(|v| (v - want).abs() < 2e-3), "Dot Gain 20 % gray displays as {want}: {o:?}");
        let s = session("cmyk", 8);
        assert_eq!(composite_profile(doc(&s)).description, "sRGB IEC61966-2.1");
    }

    /// `cargo test -p photocraft-engine --release -- --ignored bench_cmyk --nocapture`
    #[test]
    #[ignore]
    fn bench_cmyk_conversion_6016() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 6016, "height": 6016, "mode": "rgb", "depth": 8})).unwrap();
        s.edit("noise", |doc, _| {
            let r = doc.bounds();
            let surf = doc.layers[0].surface_mut().unwrap();
            let mut bytes = vec![0u8; r.width() as usize * r.height() as usize * 4];
            let mut x = 1u32;
            for px in bytes.as_chunks_mut::<4>().0 {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                px.copy_from_slice(&[x as u8, (x >> 8) as u8, (x >> 16) as u8, 255]);
            }
            surf.write_interleaved(r, &bytes);
            Ok(())
        })
        .unwrap();
        let mut d = (*s.active().unwrap().doc).clone();
        let t = std::time::Instant::now();
        convert_document(&mut d, Builtin::CoatedCmyk.profile(), Intent::RelativeColorimetric, true).unwrap();
        eprintln!("convert_document alone: {} ms", t.elapsed().as_millis());
        let t = std::time::Instant::now();
        s.execute("image.mode.cmyk", json!({})).unwrap();
        eprintln!("6016×6016 RGB8 → CMYK: {} ms", t.elapsed().as_millis());
    }
}

#[cfg(test)]
mod settings_tests {
    use super::*;

    fn tagged(spec: &str) -> Document {
        let mut d = Document::with_background("t", photocraft_doc::Size::new(8, 8), ColorMode::Rgb, SampleType::U8, Color::rgba(0.2, 0.6, 0.9, 1.0));
        d.icc_profile = Some(Builtin::from_id(spec).unwrap().profile().to_bytes());
        d
    }

    fn desc(d: &Document) -> Option<String> {
        d.icc_profile.as_ref().map(|b| profile_from_bytes(b).unwrap().description.clone())
    }

    #[test]
    fn color_settings_validate_and_report() {
        let mut s = Session::new();
        let r = s.execute("edit.colorSettings", json!({})).unwrap();
        assert_eq!(r["settings"]["workingRgb"], "srgb");
        assert!(s.execute("edit.colorSettings", json!({"workingRgb": "coated-cmyk"})).is_err());
        assert!(s.execute("edit.colorSettings", json!({"policyRgb": "sometimes"})).is_err());
        assert!(s.execute("edit.colorSettings", json!({"intent": "vivid"})).is_err());
        let r = s.execute("edit.colorSettings", json!({"workingRgb": "adobe-rgb-compat", "intent": "perceptual", "bpc": false})).unwrap();
        assert!(r["working"]["rgb"].as_str().unwrap().starts_with("Adobe RGB"));
        assert_eq!(s.color.settings.intent(), Intent::Perceptual);
        // Settings travel with the preferences file.
        let mut t = Session::new();
        t.load_prefs_json(&s.prefs_to_json()).unwrap();
        assert_eq!(t.color.settings, s.color.settings);
    }

    // The Blend Text Colors Using Gamma test is `tests/text_gamma.rs`: it changes the
    // process-wide text gamma, which would race every unit test here that composites type.

    #[test]
    fn open_question_asks_only_when_set_and_resolve_open_applies_each_answer() {
        let mut s = Session::new();
        // Ask When Opening off: nothing to ask, mismatched or missing.
        s.execute("edit.colorSettings", json!({"askOnMismatch": false, "askOnMissing": false})).unwrap();
        assert!(s.color.open_question(&tagged("adobe-rgb-compat")).is_none());
        let mut untagged = tagged("srgb");
        untagged.icc_profile = None;
        assert!(s.color.open_question(&untagged).is_none());
        // On: a mismatch asks, suggesting the policy's answer; a matching profile never asks.
        s.execute("edit.colorSettings", json!({"askOnMismatch": true, "askOnMissing": true, "policyRgb": "convert"})).unwrap();
        let q = s.color.open_question(&tagged("adobe-rgb-compat")).unwrap();
        assert_eq!((q["mismatch"].as_bool(), q["action"].as_str()), (Some(true), Some("convert")));
        assert!(q["embedded"].as_str().unwrap().starts_with("Adobe RGB"));
        assert!(s.color.open_question(&tagged("srgb")).is_none());
        assert_eq!(s.color.open_question(&untagged).unwrap()["missing"], true);
        // Asking changes nothing; the answer is applied to the document.
        let mut d = tagged("adobe-rgb-compat");
        s.color.resolve_open(&mut d, "preserve").unwrap();
        assert!(desc(&d).unwrap().starts_with("Adobe RGB"));
        s.color.resolve_open(&mut d, "convert").unwrap();
        assert_eq!(desc(&d).as_deref(), Some("sRGB IEC61966-2.1"));
        s.color.resolve_open(&mut d, "discard").unwrap();
        assert_eq!(desc(&d), None);
        s.color.resolve_open(&mut d, "assignWorking").unwrap();
        assert_eq!(desc(&d).as_deref(), Some("sRGB IEC61966-2.1"));
        assert!(s.color.resolve_open(&mut d, "explode").is_err());
    }

    #[test]
    fn open_policies() {
        let mut s = Session::new();
        // Preserve (default): an Adobe RGB file keeps its profile and asks.
        let (_, r) = s.open_document(tagged("adobe-rgb-compat"), None);
        assert_eq!((r["action"].as_str(), r["mismatch"].as_bool(), r["ask"].as_bool()), (Some("kept"), Some(true), Some(true)));
        assert!(desc(&s.active().unwrap().doc).unwrap().starts_with("Adobe RGB"));
        // Matching profiles never ask.
        let (_, r) = s.open_document(tagged("srgb"), None);
        assert_eq!(r["mismatch"], false);
        // Nor does the same space in other bytes: sRGB as Photoshop embeds it (v2, 1024-entry
        // tables) is the working sRGB, even under Convert, and the file keeps its own profile.
        s.execute("edit.colorSettings", json!({"policyRgb": "convert"})).unwrap();
        let mut v2 = Builtin::Srgb.profile().clone();
        let trc = v2.trc.clone().unwrap();
        let table = |c: &photocraft_cms::Curve| photocraft_cms::Curve::Table((0..1024).map(|i| c.eval64(f64::from(i) / 1023.0) as f32).collect());
        v2.trc = Some([table(&trc[0]), table(&trc[1]), table(&trc[2])]);
        v2.version = (2, 0x10);
        let bytes = v2.with_encoded_bytes().to_bytes();
        assert_ne!(bytes, Builtin::Srgb.profile().to_bytes());
        let mut d = tagged("srgb");
        d.icc_profile = Some(bytes.clone());
        let (_, r) = s.open_document(d, None);
        assert_eq!((r["action"].as_str(), r["mismatch"].as_bool(), r.get("ask")), (Some("kept"), Some(false), None));
        assert_eq!(s.active().unwrap().doc.icc_profile.as_ref(), Some(&bytes));
        // A 32-bit document in Photoshop's linear sRGB (a v2 profile recording the D65 display
        // white) stays linear without asking, like one tagged with our own linear sRGB.
        let mut lin = Builtin::LinearSrgb.profile().clone();
        lin.version = (2, 0x10);
        lin.white_point = [0.95047, 1.0, 1.08905];
        let bytes = lin.with_encoded_bytes().to_bytes();
        assert_ne!(bytes, Builtin::LinearSrgb.profile().to_bytes());
        let mut d = Document::with_background("t", photocraft_doc::Size::new(8, 8), ColorMode::Rgb, SampleType::F32, Color::rgba(0.2, 0.6, 0.9, 1.0));
        d.icc_profile = Some(bytes.clone());
        let (_, r) = s.open_document(d, None);
        assert_eq!((r["action"].as_str(), r.get("ask")), (Some("kept"), None));
        assert_eq!(s.active().unwrap().doc.icc_profile.as_ref(), Some(&bytes));
        s.execute("edit.colorSettings", json!({"policyRgb": "preserve"})).unwrap();
        // Convert to working: pixels converted and the document tagged with sRGB.
        s.execute("edit.colorSettings", json!({"policyRgb": "convert"})).unwrap();
        let (_, r) = s.open_document(tagged("adobe-rgb-compat"), None);
        assert_eq!(r["action"], "converted");
        let d = &s.active().unwrap().doc;
        assert!(desc(d).unwrap().starts_with("sRGB"));
        assert!((d.layers[0].surface().unwrap().pixel(1, 1)[0] - 0.2).abs() > 0.01, "values changed by the conversion");
        // Off: the embedded profile is discarded.
        s.execute("edit.colorSettings", json!({"policyRgb": "off"})).unwrap();
        let (_, r) = s.open_document(tagged("display-p3"), None);
        assert_eq!(r["action"], "discarded");
        assert!(s.active().unwrap().doc.icc_profile.is_none());
        // Untagged files are assumed to be in a non-default working space and tagged with it.
        s.execute("edit.colorSettings", json!({"policyRgb": "preserve", "workingRgb": "display-p3"})).unwrap();
        let mut d = tagged("srgb");
        d.icc_profile = None;
        let (_, r) = s.open_document(d, None);
        assert_eq!(r["action"], "assigned");
        assert_eq!(desc(&s.active().unwrap().doc).as_deref(), Some("Display P3"));
    }

    /// Display P3 as an input profile with only an AToB table, like Apple's Wide Color Sharing
    /// Profile embedded in iPhone photos.
    fn input_only_p3() -> Profile {
        let p3 = Builtin::DisplayP3.profile();
        let t = Transform::new(p3, Builtin::LabD50.profile(), Intent::RelativeColorimetric, false).unwrap();
        let clut = photocraft_cms::Clut::sample(vec![9; 3], 3, |inp, out| t.eval(inp, out));
        let ident = || photocraft_cms::pipeline::Stage::Curves(vec![photocraft_cms::Curve::Identity; 3]);
        let lut = photocraft_cms::profile::Lut {
            kind: photocraft_cms::profile::LutKind::Ab,
            inputs: 3,
            outputs: 3,
            stages: vec![ident(), photocraft_cms::pipeline::Stage::Clut(clut), ident()],
        };
        let mut p = p3.clone();
        p.class = photocraft_cms::ProfileClass::Input;
        p.pcs = photocraft_cms::Pcs::Lab;
        p.description = "Input-only P3".into();
        (p.matrix, p.trc) = (None, None);
        p.a2b = [Some(lut), None, None];
        p.b2a = [None, None, None];
        p.with_encoded_bytes()
    }

    #[test]
    fn input_only_profiles_are_converted_on_open() {
        // #3876: an iPhone photo's input-only profile was kept as the document's space, so
        // compositing anything into it failed with "RGB profile without matrix/TRC or BToA tags".
        let input = input_only_p3();
        assert!(Builtin::DisplayP3.profile().can_be_destination());
        assert!(!input.can_be_destination());
        let mut s = Session::new();
        let mut d = tagged("srgb");
        d.icc_profile = Some(input.to_bytes());
        let (photo, r) = s.open_document(d, None);
        assert_eq!((r["action"].as_str(), r["inputOnly"].as_bool(), r["ask"].as_bool()), (Some("converted"), Some(true), Some(true)));
        let doc = &s.active().unwrap().doc;
        assert!(desc(doc).unwrap().starts_with("sRGB"));
        // Converted from P3, not reinterpreted as sRGB.
        assert!((doc.layers[0].surface().unwrap().pixel(1, 1)[0] - 0.2).abs() > 0.01);
        let (other, _) = s.open_document(tagged("srgb"), None);
        s.execute("layer.copyToDocument", json!({"document": photo, "source": other})).unwrap();
        // Off still discards the profile, and profiles that can be a destination are still kept.
        s.execute("edit.colorSettings", json!({"policyRgb": "off"})).unwrap();
        let mut d = tagged("srgb");
        d.icc_profile = Some(input.to_bytes());
        let (_, r) = s.open_document(d, None);
        assert_eq!((r["action"].as_str(), r["inputOnly"].as_bool()), (Some("discarded"), Some(true)));
        s.execute("edit.colorSettings", json!({"policyRgb": "preserve"})).unwrap();
        let (_, r) = s.open_document(tagged("display-p3"), None);
        assert_eq!((r["action"].as_str(), r.get("inputOnly")), (Some("kept"), None));
    }

    /// A little-endian EXIF block whose Exif IFD holds ColorSpace = `space`.
    fn exif_color_space(space: u16) -> Vec<u8> {
        let mut b = b"Exif\0\0II\x2a\0\x08\0\0\0".to_vec();
        // IFD0: the Exif IFD pointer (LONG) to offset 26, then no next IFD.
        b.extend([1, 0, 0x69, 0x87, 4, 0, 1, 0, 0, 0, 26, 0, 0, 0, 0, 0, 0, 0]);
        // Exif IFD: ColorSpace (SHORT).
        b.extend([1, 0, 0x01, 0xA0, 3, 0, 1, 0, 0, 0]);
        b.extend(space.to_le_bytes());
        b.extend([0, 0, 0, 0, 0, 0]);
        b
    }

    fn untagged_camera_file(space: Option<u16>) -> Document {
        let mut d = tagged("srgb");
        d.icc_profile = None;
        d.metadata.exif = space.map(|v| Arc::new(exif_color_space(v)));
        d
    }

    /// The tag survives a real JPEG: written without an ICC profile, read back, opened as sRGB.
    #[test]
    fn exif_srgb_jpeg_round_trip() {
        let mut s = Session::new();
        s.execute("edit.colorSettings", json!({"workingRgb": "display-p3"})).unwrap();
        let r = photocraft_io::export(&untagged_camera_file(Some(1)), "x.jpg", &photocraft_io::ExportOptions::default()).unwrap();
        let back = photocraft_io::import("x.jpg", &r.bytes).unwrap().document;
        assert!(back.icc_profile.is_none(), "no ICC profile in the file");
        let (_, r) = s.open_document(back, None);
        assert_eq!((r["action"].as_str(), r["exif"].as_bool()), (Some("kept"), Some(true)));
        assert!(desc(&s.active().unwrap().doc).unwrap().starts_with("sRGB"));
    }

    /// #204: a camera file with no ICC profile but EXIF ColorSpace = sRGB opens as sRGB, as in
    /// Photoshop; Ignore EXIF Profile Tag opens it untagged (the working space).
    #[test]
    fn exif_srgb_tag_unless_ignored() {
        let mut s = Session::new();
        // Working sRGB: tagged sRGB, nothing to ask.
        let (_, r) = s.open_document(untagged_camera_file(Some(1)), None);
        assert_eq!((r["action"].as_str(), r["exif"].as_bool(), r.get("ask")), (Some("kept"), Some(true), None));
        assert!(desc(&s.active().unwrap().doc).unwrap().starts_with("sRGB"));
        // Working Display P3, Preserve: it stays sRGB instead of being read as P3.
        s.execute("edit.colorSettings", json!({"policyRgb": "preserve", "workingRgb": "display-p3"})).unwrap();
        let (_, r) = s.open_document(untagged_camera_file(Some(1)), None);
        assert_eq!((r["action"].as_str(), r["mismatch"].as_bool(), r["exif"].as_bool()), (Some("kept"), Some(true), Some(true)));
        assert!(desc(&s.active().unwrap().doc).unwrap().starts_with("sRGB"));
        // Convert: converted from sRGB to P3, so the pixel values change.
        s.execute("edit.colorSettings", json!({"policyRgb": "convert"})).unwrap();
        let (_, r) = s.open_document(untagged_camera_file(Some(1)), None);
        assert_eq!(r["action"], "converted");
        let d = &s.active().unwrap().doc;
        assert_eq!(desc(d).as_deref(), Some("Display P3"));
        assert!((d.layers[0].surface().unwrap().pixel(1, 1)[0] - 0.2).abs() > 0.01);
        // Uncalibrated, no EXIF, or Ignore EXIF Profile Tag on: untagged, so the working space.
        s.execute("edit.colorSettings", json!({"policyRgb": "preserve"})).unwrap();
        for (space, ignore) in [(Some(0xFFFF), false), (None, false), (Some(1), true)] {
            s.execute("prefs.set", json!({"path": "fileHandling.ignoreExifProfileTag", "value": ignore})).unwrap();
            let (_, r) = s.open_document(untagged_camera_file(space), None);
            assert_eq!((r["action"].as_str(), r.get("exif")), (Some("assigned"), None), "{space:?}, ignore {ignore}");
            assert_eq!(desc(&s.active().unwrap().doc).as_deref(), Some("Display P3"));
        }
        // An embedded profile always wins over the EXIF tag.
        s.execute("prefs.set", json!({"path": "fileHandling.ignoreExifProfileTag", "value": false})).unwrap();
        let mut d = tagged("adobe-rgb-compat");
        d.metadata.exif = Some(Arc::new(exif_color_space(1)));
        let (_, r) = s.open_document(d, None);
        assert_eq!((r["action"].as_str(), r.get("exif")), (Some("kept"), None));
        assert_eq!(desc(&s.active().unwrap().doc), Some(Builtin::from_id("adobe-rgb-compat").unwrap().profile().description.clone()));
    }

    #[test]
    fn mode_conversion_uses_working_spaces() {
        let mut s = Session::new();
        s.execute("edit.colorSettings", json!({"workingRgb": "rec2020"})).unwrap();
        for depth in [8, 16, 32] {
            s.execute("file.new", json!({"width": 8, "height": 8, "mode": "gray", "depth": depth})).unwrap();
            s.execute("image.mode.rgb", json!({})).unwrap();
            let d = &s.active().unwrap().doc;
            assert_eq!(d.mode, ColorMode::Rgb);
            assert!(desc(d).unwrap().contains("2020"), "depth {depth}: {:?}", desc(d));
        }
        // "working" specs follow the settings too.
        let p = s.color.resolve("working-rgb", None, None).unwrap();
        assert!(p.description.contains("2020"));
    }

    #[test]
    fn mismatch_resolution_command() {
        let mut s = Session::new();
        s.open_document(tagged("adobe-rgb-compat"), None);
        s.execute("color.profileMismatch", json!({"action": "convert"})).unwrap();
        assert!(desc(&s.active().unwrap().doc).unwrap().starts_with("sRGB"));
        s.undo();
        s.execute("color.profileMismatch", json!({"action": "discard"})).unwrap();
        assert!(s.active().unwrap().doc.icc_profile.is_none());
        assert!(s.execute("color.profileMismatch", json!({"action": "explode"})).is_err());
    }
}
