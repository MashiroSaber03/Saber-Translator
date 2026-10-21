//! The Photocraft engine façade: open documents, history, and the command registry.
//!
//! Every user-visible action is a command with a stable id (Photoshop-style, such as
//! `layer.newAdjustmentLayer.invert`) and JSON parameters. Every frontend goes through the same
//! [`Session::execute`] entry point: the egui UI, the CLI, the remote-control channel and the MCP
//! server. This is what makes the UI swappable and the app fully scriptable.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod actions_cmds;
pub mod actions_edit_cmds;
pub mod adjust_cmds;
pub mod adjust_params;
pub mod align_cmds;
mod allocation;
pub mod analysis_cmds;
pub mod artboard_cmds;
pub mod atn_cmds;
pub mod automate_cmds;
pub mod background_cmds;
pub mod brush_cmds;
pub mod brush_key_cmds;
pub mod brush_preset_cmds;
pub mod build_info;
mod canvas_geom;
mod channel_clip;
pub mod channel_cmds;
mod clip_group;
pub mod color_cmds;
pub mod color_to_alpha_cmds;
pub mod commands;
pub mod comps_cmds;
pub mod cutout_cmds;
pub mod display_color;
pub mod distort_cmds;
pub mod document_preset_cmds;
pub mod edit_cmds;
pub mod edit_menu_cmds;
pub mod eraser_cmds;
pub mod exr_cmds;
pub mod extra_cmds;
pub mod file_cmds;
pub mod fill_cmds;
pub mod fill_key_cmds;
pub mod filters;
pub mod filters_ext;
pub mod float_cmds;
mod frame_cmds;
pub mod fx_view_cmds;
pub mod fx_visibility_cmds;
pub mod gallery_cmds;
pub mod gradient_fill_cmds;
pub mod group_view_cmds;
pub mod guide_layout;
pub mod hidden_target;
pub mod history_cmds;
pub mod history_select_cmds;
pub mod image_cmds;
pub mod inspect;
pub mod installed_profiles;
pub mod jobs;
pub mod kys;
pub mod layer_copy_cmds;
pub mod layer_label_cmds;
pub mod layer_mask_props_cmds;
pub mod layer_menu_cmds;
pub mod layer_multi_cmds;
pub mod layer_nav_cmds;
pub mod layer_style;
pub mod lens_cmds;
pub mod link_watch;
pub mod magnetic_cmds;
pub mod mask_view_cmds;
mod migrate_cmds;
pub mod mode_cmds;
pub mod multichannel_cmds;
pub mod new_layer;
pub mod notes_cmds;
pub mod paint_cmds;
mod param_check;
mod path_edit_cmds;
pub mod pattern_cmds;
pub mod pattern_fill_cmds;
pub mod perspective_crop_cmds;
pub mod photo_cmds;
pub mod pick_cmds;
mod pixels;
pub mod plugin_cmds;
pub mod prefs;
pub mod preset_import_cmds;
pub mod preset_store;
pub mod presets;
pub mod print_cmds;
pub mod proof_sim;
pub mod rasterize_style_cmds;
pub mod redeye_cmds;
pub mod render_cmds;
pub mod retouch_cmds;
pub mod sample_cmds;
pub mod sampler_cmds;
pub mod select_extra_cmds;
pub mod selection_cmds;
pub mod slice_cmds;
pub mod smart_cmds;
pub mod smartselect_cmds;
pub mod snap;
pub mod solid_fill_cmds;
pub mod stamp_cmds;
pub mod swatch_cmds;
pub mod symmetry_cmds;
mod timeline_cmds;
pub mod transform_cmds;
mod trap_cmds;
pub mod type_caret_cmds;
pub mod type_cmds;
pub mod type_extra_cmds;
pub mod type_mask_cmds;
pub mod type_spell_cmds;
pub mod type_styles_cmds;
mod variables_cmds;
pub use variables_cmds::{export_data_sets_with, import_data_sets_text};
pub mod vector_cmds;
mod video_cmds;
pub use video_cmds::render_video_with;
pub mod vp_cmds;
pub mod warp_cmds;
pub mod web_cmds;
mod wia_cmds;

use std::sync::Arc;

use photocraft_doc::{Document, LayerId};
use photocraft_ops::{History, LayerTarget};
use serde_json::Value;

pub use commands::{CommandSpec, command_specs};
pub use photocraft_doc as doc;
pub use photocraft_paint as paint;
pub use photocraft_paint::BrushSettings;

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("unknown command `{0}`")]
    UnknownCommand(String),
    #[error("command `{0}` is not available right now: {1}")]
    Disabled(String, String),
    #[error("invalid parameters for `{cmd}`: {msg}")]
    BadParams { cmd: String, msg: String },
    #[error("no active document")]
    NoDocument,
    #[error("no such layer {0:?}")]
    NoLayer(LayerId),
    #[error("{0}")]
    Other(String),
    /// A background job was cancelled (see [`jobs`]); nothing changed.
    #[error("cancelled")]
    Cancelled,
}

pub type Result<T> = std::result::Result<T, EngineError>;

impl From<photocraft_raster::AllocationError> for EngineError {
    fn from(error: photocraft_raster::AllocationError) -> Self {
        Self::Other(format!("not enough memory for this operation ({error}); the document was not changed"))
    }
}

/// The pixels of a raster layer (typically one a command just created), as an error instead
/// of a panic if the layer has none.
pub(crate) fn pixels_mut(l: &mut photocraft_doc::Layer) -> Result<&mut photocraft_raster::Surface> {
    let id = l.id;
    l.surface_mut().ok_or_else(|| EngineError::Other(format!("layer {id:?} has no pixels")))
}

/// The active document's active layer, for `enabled` predicates.
pub(crate) fn active_layer_of(s: &Session) -> std::result::Result<&photocraft_doc::Layer, String> {
    let d = s.active().ok_or("no document open")?;
    d.active_layer.and_then(|id| d.doc.layer(id)).ok_or_else(|| "no active layer".into())
}

/// Per-document editing state.
#[derive(Clone, Debug)]
pub struct DocState {
    pub doc: Arc<Document>,
    pub history: History,
    /// The primary ("key") layer: what single-layer commands act on. Always a member of
    /// `selected_layers` when set.
    pub active_layer: Option<LayerId>,
    /// Every selected layer in the Layers panel (⌘/⇧-click), in selection order. Like
    /// `active_layer`, selecting is not a history step, but undo and redo restore the layers each
    /// state targeted when the next step was taken; see [`DocState::selected_layers`].
    pub selected_layers: Vec<LayerId>,
    /// Anchor of ⇧-click range selection (the last plainly or ⌘-clicked layer).
    pub layer_anchor: Option<LayerId>,
    pub path: Option<String>,
    /// An Affinity document (or its preview) must not acquire the native source as its Save path.
    pub source_read_only: bool,
    /// Increments on every change; UIs re-render when it moves.
    pub revision: u64,
    pub saved_revision: u64,
    /// How much of `revision` came from view-only changes (channel eyes, targeting): pixel caches can ignore those.
    pub view_revision: u64,
    /// Area changed by the latest revision (None = assume everything changed).
    pub last_damage: Option<photocraft_geom::Rect>,
    /// Coalescing key of the latest history step (see [`Session::execute`]'s `coalesce` param).
    pub coalesce: Option<String>,
    /// Channels panel: targeted channel and eye toggles (view state, not history).
    pub channel_view: channel_cmds::ChannelView,
    /// Select › Isolate Layers: the Layers panel lists only these layers (empty = off; view state).
    pub isolated_layers: Vec<LayerId>,
    /// Painting symmetry preset or sampled path (tool state, not document pixels).
    pub symmetry: Option<symmetry_cmds::PaintingSymmetry>,
    /// Layers panel: layers whose effects list is collapsed under their row (the fx triangle;
    /// view state, not history). Effects lists start open.
    pub fx_collapsed: Vec<LayerId>,
    /// ⌥-click on a layer's eye (`layer.showOnly`): the layer shown alone and every layer's
    /// visibility before, so the next ⌥-click restores it (view state, not history).
    pub show_only: Option<(LayerId, Vec<(LayerId, bool)>)>,
    /// A floating selection (`select.float`): the cut piece and where it floats, until dropped
    /// (view state: the document is unchanged until `select.drop`).
    pub floating: Option<float_cmds::Floating>,
    /// Color Sampler points (`view.colorSamplers.*`, #1046): numbered sample positions in document
    /// pixel coordinates. Per-document view state: discarded when the document closes, never saved
    /// to `.pcraft` or PSD, and not a history step.
    pub color_samplers: Vec<[f64; 2]>,
    /// History panel snapshots, oldest first (`history.newSnapshot`, #2966). Kept for the
    /// session only, as in Photoshop: not history states, and not saved with the document.
    pub snapshots: Vec<history_cmds::Snapshot>,
    /// The id the next snapshot gets; ids are never reused, so deleting one keeps the others.
    next_snapshot: u64,
}

impl DocState {
    pub fn new(doc: Document, path: Option<String>) -> Self {
        let active_layer = doc.top_layer();
        let mut history = History::default();
        history.set_current_layers(LayerTarget { active: active_layer, selected: active_layer.into_iter().collect() });
        Self {
            doc: Arc::new(doc),
            history,
            active_layer,
            selected_layers: active_layer.into_iter().collect(),
            layer_anchor: active_layer,
            path,
            source_read_only: false,
            revision: 1,
            saved_revision: 1,
            view_revision: 0,
            last_damage: None,
            coalesce: None,
            channel_view: Default::default(),
            isolated_layers: Vec::new(),
            symmetry: None,
            fx_collapsed: Vec::new(),
            show_only: None,
            floating: None,
            color_samplers: Vec::new(),
            snapshots: Vec::new(),
            next_snapshot: 1,
        }
    }
    /// The selected layers in bottom-to-top document order, always including the active layer.
    /// Robust against stale state: ids no longer in the document are skipped, and if the active
    /// layer was changed without updating the set, the selection is just the active layer.
    pub fn selected_layers(&self) -> Vec<LayerId> {
        let Some(active) = self.active_layer else { return Vec::new() };
        let set: &[LayerId] = if self.selected_layers.contains(&active) { &self.selected_layers } else { std::slice::from_ref(&active) };
        self.doc.walk().into_iter().map(|(_, _, l)| l.id).filter(|id| set.contains(id)).collect()
    }
    pub fn is_layer_selected(&self, id: LayerId) -> bool {
        self.active_layer == Some(id) || (self.active_layer.is_some_and(|a| self.selected_layers.contains(&a)) && self.selected_layers.contains(&id))
    }
    pub fn is_dirty(&self) -> bool {
        self.revision != self.saved_revision
    }
    /// The targeted layers, as history stores them with each state.
    fn layer_target(&self) -> LayerTarget {
        LayerTarget { active: self.active_layer, selected: self.selected_layers.clone() }
    }
}

/// App-wide tool state that commands read (foreground colour, brush…).
#[derive(Clone, Debug)]
pub struct ToolState {
    pub foreground: [f32; 4],
    pub background: [f32; 4],
    pub brush: photocraft_paint::BrushSettings,
    /// Brush presets (built-ins plus user presets; see `brush.presets.*`).
    pub presets: Vec<photocraft_paint::BrushPreset>,
    /// Bumped whenever `presets` changes ([`Session::brush_presets_changed`]); the preset store
    /// syncs when it moves.
    pub presets_rev: u64,
    /// Mixer Brush paint carried between strokes.
    pub mixer: photocraft_paint::mixer::MixerState,
    /// The coalescing key of the running `tools.setBrush` gesture and the brush before it, so the
    /// gesture journals as one call ([`brush_cmds::coalesce_journal`]).
    pub brush_gesture: Option<(String, photocraft_paint::BrushSettings)>,
    /// Name of the preset the current brush was last picked from. Kept across later edits (the
    /// preset stays "the current brush" so `brush.presets.update` can overwrite it); cleared when
    /// the brush is reset or the preset is gone.
    pub current_preset: Option<String>,
    /// Bumped whenever `brush` or `current_preset` changes ([`ToolState::brush_changed`]). The
    /// last-brush persistence watches it instead of comparing the brush itself, so an unchanged
    /// brush costs a counter compare per frame, not a clone and serialise.
    pub brush_rev: u64,
    /// The revision whose value is in `Preferences::last_brush` (a restored one counts as saved):
    /// [`brush_rev`](Self::brush_rev) == `brush_saved_rev` means there is nothing new to write.
    pub brush_saved_rev: u64,
    /// A layer mask is the edit target ([`ToolState::target_mask`]).
    pub mask_targeted: bool,
    /// The foreground/background pair of the edit target that isn't current: the layer pixels'
    /// colours while a mask is targeted, the mask's while the pixels are (#2166).
    pub other_colors: [[f32; 4]; 2],
    /// Colours picked as CMYK, newest last: the sRGB colour that stands for them and the ink
    /// values chosen. The tool colours are stored as sRGB, and an sRGB round trip does not give
    /// back what was typed (100/0/0/0 cyan is outside sRGB), so a CMYK document is filled with
    /// the inks the person picked (#4196). Matched by colour, so it follows a swap, a reset or a
    /// mask pair without bookkeeping.
    pub cmyk_picks: Vec<([f32; 3], [f32; 4])>,
}

impl ToolState {
    /// How many CMYK picks are remembered.
    const CMYK_PICKS: usize = 8;

    /// Remember that sRGB colour `rgb` was picked as the inks `cmyk` (each 0..1).
    pub fn remember_cmyk(&mut self, rgb: [f32; 3], cmyk: [f32; 4]) {
        self.cmyk_picks.retain(|(c, _)| *c != rgb);
        self.cmyk_picks.push((rgb, cmyk));
        let extra = self.cmyk_picks.len().saturating_sub(Self::CMYK_PICKS);
        self.cmyk_picks.drain(..extra);
    }

    /// The inks `rgb` was picked as, if it was picked as CMYK.
    pub fn picked_cmyk(&self, rgb: [f32; 3]) -> Option<[f32; 4]> {
        self.cmyk_picks.iter().rev().find(|(c, _)| *c == rgb).map(|(_, k)| *k)
    }

    /// Photoshop's colour pair for a layer mask that has none yet: white foreground, black
    /// background (reveal-paint first).
    pub const MASK_COLORS: [[f32; 4]; 2] = [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]];

    /// Note that `brush` or `current_preset` changed. Every setter of either (commands and the
    /// shell's tool-switch paths) calls this; the last-brush persistence only serialises when the
    /// revision has moved since the stored value ([`Self::brush_saved_rev`]).
    pub fn brush_changed(&mut self) {
        self.brush_rev = self.brush_rev.wrapping_add(1);
    }

    /// Switch the edit target between layer pixels and a layer mask. Like Photoshop, the mask and
    /// the pixels keep separate app-wide foreground/background pairs: changing the target puts
    /// the current pair aside and brings back the other one, so colours picked while a mask is
    /// targeted are remembered for masks (#2166). The mask pair starts at [`Self::MASK_COLORS`].
    /// Returns whether the target changed.
    pub fn target_mask(&mut self, mask: bool) -> bool {
        if self.mask_targeted == mask {
            return false;
        }
        self.mask_targeted = mask;
        let [fg, bg] = std::mem::replace(&mut self.other_colors, [self.foreground, self.background]);
        self.foreground = fg;
        self.background = bg;
        true
    }
}

impl Default for ToolState {
    fn default() -> Self {
        Self {
            foreground: [0.0, 0.0, 0.0, 1.0],
            background: [1.0, 1.0, 1.0, 1.0],
            // Photoshop's Brush tool starts at 10 % Smoothing.
            brush: photocraft_paint::BrushSettings {
                smoothing: photocraft_paint::brush::Smoothing { amount: 0.1, ..Default::default() },
                ..Default::default()
            },
            presets: photocraft_paint::presets::builtin(),
            presets_rev: 0,
            mixer: Default::default(),
            brush_gesture: None,
            current_preset: None,
            brush_rev: 0,
            brush_saved_rev: 0,
            mask_targeted: false,
            other_colors: Self::MASK_COLORS,
            cmyk_picks: Vec::new(),
        }
    }
}

#[derive(Default)]
pub struct Session {
    docs: Vec<DocState>,
    active: Option<usize>,
    pub tools: ToolState,
    /// Log of executed commands (for action recording and debugging).
    pub journal: Vec<(String, Value)>,
    /// Coalescing key of the command being executed.
    coalesce_request: Option<String>,
    /// Pixels copied with Edit › Copy / Cut (shared by all documents, like Photoshop).
    pub clipboard: Option<edit_cmds::Clip>,
    /// Layer › Layer Style › Copy Layer Style: effects, blend mode, fill opacity and the Advanced
    /// Blending switches.
    pub style_clipboard: Option<(photocraft_doc::Effects, photocraft_color::BlendMode, f32, photocraft_doc::AdvancedBlending)>,
    /// Shape path context menu: copied vector fill and stroke styles.
    pub path_fill_clipboard: Option<photocraft_doc::Fill>,
    pub path_stroke_clipboard: Option<photocraft_doc::ShapeStroke>,
    /// Colour management: proofing state, monitor profile, display transforms.
    pub color: color_cmds::ColorState,
    /// Open Edit Contents documents and the smart objects they update.
    pub smart_links: Vec<smart_cmds::SmartLink>,
    /// Type › Save Default Type Styles: character and paragraph style new type layers start from.
    pub type_defaults: Option<(photocraft_doc::text::CharStyle, photocraft_doc::text::ParagraphStyle)>,
    /// Initial overflow policy for new type layers. Existing layers own their saved policy.
    pub type_overflow_visible: bool,
    /// Quick Mask Options (colour, opacity, colour indicates) used when entering Quick Mask.
    pub quick_mask_options: channel_cmds::QuickMaskOptions,
    /// Colour channel the running command may change (a single colour channel is targeted).
    color_restrict: Option<usize>,
    /// Edit › Preferences, keyboard shortcuts, menu and toolbar customisation (see `prefs`).
    pub prefs: prefs::PrefsStore,
    /// Edit menu state: Fade source, custom shape library (see `edit_menu_cmds`).
    pub edit_state: edit_menu_cmds::EditState,
    /// The pattern library (Window › Patterns; see `pattern.*`).
    pub patterns: pattern_cmds::PatternLibrary,
    /// Image › Analysis: Measurement Log and Select Data Points (see `analysis_cmds`).
    pub analysis: analysis_cmds::AnalysisState,
    /// Window › Gradients, Patterns (groups), Styles, Shapes, Tool Presets and Clone Source.
    pub presets: presets::PresetState,
    /// File menu state: Lock Slices, Image Assets, last Print / Save for Web settings, script
    /// event log (see `automate_cmds`).
    pub file_menu: automate_cmds::FileMenuState,
    /// Persistent brush preset store (desktop only; `None` keeps presets session-only, as in
    /// headless and test sessions). See `preset_store`. The same store holds the Actions list.
    pub preset_store: Option<preset_store::PresetStore>,
    /// Window › Actions. The list persists with the preset store when one is attached.
    pub actions: actions_cmds::ActionState,
    /// Gate for every command the session runs, including the ones a command runs on its own
    /// behalf and `actions.play` steps. Untrusted sessions (MCP, the control channel) install
    /// the same check a top-level request sees. `None` runs everything, which is what a local
    /// UI and `photocraft-cli run` do.
    pub authorize: Option<fn(&str, &serde_json::Value) -> Result<()>>,
    /// Background jobs (see [`jobs`]).
    jobs: jobs::Jobs,
    /// Color Sampler points dropped by the last canvas-size change (crop, Canvas Size, Image Size,
    /// rotation), for the status message. Read with [`Session::take_samplers_removed`].
    samplers_removed: usize,
    /// The canvas-geometry map the next [`Session::edit`] applies to the Color Sampler points (crop,
    /// Canvas Size, Image Size, rotation set it just before their edit). Consumed by the edit.
    sampler_map: Option<photocraft_geom::Affine>,
    /// Each open document's linked smart objects' files: size and modification time when last
    /// seen ([`link_watch`]).
    pub link_stamps: std::collections::HashMap<(u64, String), link_watch::Stamp>,
    /// The stamps being taken on a worker thread ([`Session::start_link_scan`]).
    pub link_scan: Option<link_watch::LinkScan>,
    /// Select Subject's last result, reused while the pixels are unchanged (#2865).
    subject_cache: smartselect_cmds::SubjectCache,
    /// Object Selection's Object Finder results (#4087), shared by the hover preview and the
    /// click that commits it; valid while the pixels are unchanged.
    pub object_finder: smartselect_cmds::ObjectFinderCache,
    /// A finished live stroke offered to the `paint.stroke` / `paint.pencil` that commits it
    /// ([`Session::offer_live_stroke`]).
    pub(crate) live_commit: Option<brush_cmds::LiveCommit>,
    /// Documents a background open decoded but held back until the user answers the profile
    /// prompt ([`ColorState::open_question`](color_cmds::ColorState::open_question)): `(id,
    /// document, source read-only)`. The shell takes them with [`Session::take_held_open`].
    held_opens: Vec<(u64, Document, bool)>,
    next_held_open: u64,
}

/// Move item `i` of `v` to position `to`, clamped to the end. Returns where it went; `None` when
/// `i` is out of range.
pub fn move_item<T>(v: &mut Vec<T>, i: usize, to: usize) -> Option<usize> {
    if i >= v.len() {
        return None;
    }
    let x = v.remove(i);
    let to = to.min(v.len());
    v.insert(to, x);
    Some(to)
}

impl Session {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn documents(&self) -> &[DocState] {
        &self.docs
    }
    pub fn active_index(&self) -> Option<usize> {
        self.active
    }
    pub fn active(&self) -> Option<&DocState> {
        self.active.and_then(|i| self.docs.get(i))
    }
    pub fn active_mut(&mut self) -> Option<&mut DocState> {
        self.active.and_then(|i| self.docs.get_mut(i))
    }
    pub fn set_active(&mut self, index: usize) -> bool {
        if index < self.docs.len() {
            self.active = Some(index);
            self.sync_tools_mask_target();
            true
        } else {
            false
        }
    }

    /// Color Sampler points dropped by the last canvas-size change (crop, Canvas Size, Image Size,
    /// rotation) since this was last called, for the status message. See [`edit`](Self::edit).
    pub fn take_samplers_removed(&mut self) -> usize {
        std::mem::take(&mut self.samplers_removed)
    }

    /// Move the Color Sampler points through `a` in the next [`Session::edit`] (crop, Canvas Size,
    /// Image Size, rotation), dropping the ones that land outside the new canvas. View state: the
    /// points stay pinned to their pixels, no history step. Cleared by the edit it is set for.
    pub(crate) fn set_sampler_map(&mut self, a: photocraft_geom::Affine) {
        self.sampler_map = Some(a);
    }

    pub fn sync_tools_mask_target(&mut self) {
        let mask = self.active().is_some_and(|st| st.channel_view.target == crate::channel_cmds::ChannelTarget::LayerMask);
        self.tools.target_mask(mask);
    }

    /// Move the document at `from` to tab position `to` (clamped to the last), keeping the active
    /// document active. Returns its new index; `None` when `from` is out of range.
    pub fn move_document(&mut self, from: usize, to: usize) -> Option<usize> {
        let active = self.active().map(|d| d.doc.id);
        let to = move_item(&mut self.docs, from, to)?;
        self.active = active.and_then(|id| self.docs.iter().position(|d| d.doc.id == id));
        Some(to)
    }

    /// Add a document (from File → New, an import, etc.) and make it active.
    /// Preserve its identity unless another open document already owns it.
    /// Identity-dependent callers must read the admitted document at the returned index.
    pub fn add_document(&mut self, mut doc: Document, path: Option<String>) -> usize {
        // Persisted IDs can overlap the allocator, so check every replacement too.
        while self.docs.iter().any(|st| st.doc.id == doc.id) {
            doc.id = photocraft_doc::DocId::fresh();
        }
        let mut st = DocState::new(doc, path);
        st.history.max_states = self.prefs.get().performance.history_states.max(1) as usize;
        st.history.max_bytes = self.prefs.get().performance.history_budget_bytes();
        self.docs.push(st);
        let i = self.docs.len() - 1;
        self.active = Some(i);
        i
    }

    /// Holds a decoded document back from opening (see `held_opens`); returns its id.
    pub(crate) fn hold_open(&mut self, doc: Document, source_read_only: bool) -> u64 {
        self.next_held_open = self.next_held_open.wrapping_add(1);
        self.held_opens.push((self.next_held_open, doc, source_read_only));
        self.next_held_open
    }

    /// Takes a document a background open held back for the profile prompt: the document and
    /// whether its source is read-only. `None` when there is no such document (already taken).
    pub fn take_held_open(&mut self, id: u64) -> Option<(Document, bool)> {
        let i = self.held_opens.iter().position(|(h, _, _)| *h == id)?;
        let (_, doc, read_only) = self.held_opens.remove(i);
        Some((doc, read_only))
    }

    pub fn close(&mut self, index: usize) -> Option<DocState> {
        if index >= self.docs.len() {
            return None;
        }
        smart_cmds::on_close(self, index);
        if let Some(id) = self.docs.get(index).map(|d| d.doc.id) {
            self.cancel_jobs_on(id);
            self.subject_cache.forget_doc(id);
            self.object_finder.forget_doc(id);
        }
        let d = self.docs.remove(index);
        self.active = if self.docs.is_empty() {
            None
        } else {
            Some(match self.active {
                Some(active) if active > index => active - 1,
                Some(active) if active < index => active,
                _ => index.min(self.docs.len() - 1),
            })
        };
        Some(d)
    }

    /// Run a command by id with JSON params. Returns a JSON result.
    ///
    /// Any command accepts an optional `"coalesce": "<key>"` param: consecutive edits with the same
    /// key (and no other edit, undo or redo in between) share one history step, like Photoshop's
    /// single "Edit Type Layer" step for a whole typing session or one step per slider drag.
    ///
    /// Runs synchronously, including job-capable commands (see [`jobs`]; [`Session::start`] runs
    /// those in the background).
    pub fn execute(&mut self, id: &str, params: Value) -> Result<Value> {
        match self.dispatch(id, params, false)? {
            jobs::Started::Done(v) => Ok(v),
            // Not reached: inline dispatch never starts a job. Waiting is still correct.
            jobs::Started::Job(j) => self.wait_job(j),
        }
    }

    /// Offer a finished live stroke to the next command: a `paint.stroke` / `paint.pencil` whose
    /// params match it exactly takes its pixels instead of rendering the stroke again (its result
    /// then says `"fromLive": true`); any other command drops the offer. The command still runs
    /// through the normal dispatch, so authorization, the journal and history are unchanged.
    pub fn offer_live_stroke(&mut self, live: brush_cmds::LiveStroke) -> Result<()> {
        self.live_commit = Some(live.into_commit()?);
        Ok(())
    }

    /// [`Self::offer_live_stroke`] for a live Clone Stamp or History Brush stroke, taken by the
    /// `paint.cloneStamp` / `paint.historyBrush` it was drawn for. Refused for the Healing Brush
    /// (it blends on release) and when the brush, layer or clone source changed during the stroke.
    pub fn offer_live_retouch(&mut self, live: retouch_cmds::LiveRetouch) -> Result<()> {
        let commit = live.into_commit(self)?;
        self.live_commit = Some(commit);
        Ok(())
    }

    /// Withdraw an offer the command didn't take.
    pub fn clear_live_stroke(&mut self) {
        self.live_commit = None;
    }

    /// Is the command currently runnable? (drives menu enablement)
    pub fn is_enabled(&self, id: &str) -> bool {
        self.is_enabled_with(id, &Value::Null)
    }

    /// [`Self::is_enabled`] for a call with `params`: their `"target"` can enable a command, as a
    /// targeted layer mask enables Image › Adjustments › Invert on an adjustment layer (#780).
    pub fn is_enabled_with(&self, id: &str, params: &Value) -> bool {
        self.disabled_reason_with(id, params).is_none()
    }

    /// Why the command can't run now (None = it can): its own precondition, or a background job
    /// running on the active document.
    pub fn disabled_reason(&self, id: &str) -> Option<String> {
        self.disabled_reason_with(id, &Value::Null)
    }

    /// [`Self::disabled_reason`] for a call with `params`.
    pub fn disabled_reason_with(&self, id: &str, params: &Value) -> Option<String> {
        let Some(s) = commands::find(id) else { return Some(format!("unknown command `{id}`")) };
        self.precondition(s, &channel_cmds::inject_target(self, id, params.clone())).err().or_else(|| self.job_conflict(id, s.journal))
    }

    /// The command's own precondition for a call with `params` (their target filled in).
    fn precondition(&self, spec: &commands::CommandSpec, params: &Value) -> std::result::Result<(), String> {
        if self.names_existing_layer(spec.id, params) {
            return Ok(());
        }
        // `ifPresent`: with no Background there is nothing to do, which isn't an error.
        if spec.id == "layer.new.layerFromBackground" && params.get("ifPresent").and_then(Value::as_bool) == Some(true) {
            return self.active().map(|_| ()).ok_or_else(|| "no document open".into());
        }
        channel_cmds::mask_target_enabled(self, spec.id, params).unwrap_or_else(|| (spec.enabled)(self))
    }

    /// Whether `id` is a layer command whose `"layer"` param names an existing layer, so it needs no
    /// active layer (#2457). Matches how `layer_param` resolves the layer when the command runs.
    fn names_existing_layer(&self, id: &str, params: &Value) -> bool {
        matches!(id, "layer.hideLayers" | "layer.showLayers" | "layer.renameLayer")
            && params.get("layer").and_then(Value::as_u64).is_some_and(|l| self.active().is_some_and(|st| st.doc.layer(LayerId(l)).is_some()))
    }

    /// The precondition of a command whose params document `"layer":id` (it acts on that layer),
    /// when `params` name an existing layer of the active document: checked on that layer rather
    /// than on the active one, which stays active (#466). A script, action or agent can then
    /// Replace Contents, Rasterize or Merge Down a layer without selecting it first. `None` for
    /// other commands, or without such a layer.
    pub(crate) fn explicit_layer_enabled(&mut self, spec: &commands::CommandSpec, params: &Value) -> Option<std::result::Result<(), String>> {
        if !spec.params.contains("\"layer\":id") {
            return None;
        }
        let target = LayerId(params.get("layer")?.as_u64()?);
        let d = self.active_mut()?;
        d.doc.layer(target)?;
        let active = d.active_layer.replace(target);
        let r = (spec.enabled)(self);
        if let Some(d) = self.active_mut() {
            d.active_layer = active;
        }
        Some(r)
    }

    /// Apply an undoable edit to the active document.
    pub fn edit<R>(&mut self, label: &str, f: impl FnOnce(&mut Document, &mut Option<LayerId>) -> Result<R>) -> Result<R> {
        let restrict = self.color_restrict;
        let sampler_map = self.sampler_map.take();
        let st = self.active_mut().ok_or(EngineError::NoDocument)?;
        let before = st.doc.clone();
        // Selecting layers is not a step, so the state this edit leaves behind targets what was
        // selected just before it (#1356): undoing a stroke keeps the painted layer active.
        let prior = st.layer_target();
        let mut doc = (*before).clone();
        let mut active = st.active_layer;
        let r = f(&mut doc, &mut active)?;
        if let (Some(k), Some(id)) = (restrict, active) {
            channel_cmds::restrict_to_color(&before, &mut doc, id, k);
        }
        st.doc = Arc::new(doc);
        st.active_layer = active;
        fix_selection(st);
        channel_cmds::fix_view(st);
        // A canvas-size change (crop, Canvas Size, Image Size, rotation) moves the Color Sampler
        // points with their pixels and drops the ones that land outside the new canvas. View
        // state: not a history step. A mapped edit (the four above) gives the exact map; any other
        // size change falls back to dropping whatever is now outside.
        let mut samplers_removed = 0;
        if let Some(a) = sampler_map {
            let bounds = st.doc.bounds();
            let n = st.color_samplers.len();
            st.color_samplers.retain_mut(|p| {
                let q = a.apply(photocraft_geom::Point::new(p[0], p[1]));
                *p = [q.x, q.y];
                bounds.contains(q.x.floor() as i32, q.y.floor() as i32)
            });
            samplers_removed = n - st.color_samplers.len();
        } else if st.doc.size != before.size {
            let bounds = st.doc.bounds();
            let n = st.color_samplers.len();
            st.color_samplers.retain(|p| bounds.contains(p[0].floor() as i32, p[1].floor() as i32));
            samplers_removed = n - st.color_samplers.len();
        }
        // The pixels the edit can have changed, as its undo would report them: nothing for a
        // selection, guides or a layer's name (a new lasso selection used to recomposite a large
        // document, seconds on the CPU path, #1773), a new or deleted layer's bounds (a
        // duplicate or a paste), and everything when that can't be bounded. A command may still
        // report its own, tighter damage after this.
        let damage = layer_multi_cmds::step_damage(&before, &st.doc);
        let key = self.coalesce_request.clone();
        let st = self.active_mut().ok_or(EngineError::NoDocument)?;
        let layers = st.layer_target();
        if key.is_none() || st.coalesce != key || !st.history.can_undo() {
            st.history.set_current_layers(prior);
            st.history.record(label, before, layers);
            st.history.trim(&st.doc);
        } else {
            st.history.set_current_layers(layers);
        }
        st.coalesce = key;
        st.revision += 1;
        st.last_damage = damage;
        self.samplers_removed += samplers_removed;
        if damage.is_none_or(|d| !d.is_empty()) {
            // Free the mask early; a stale entry would only miss (it is keyed by pixels).
            self.subject_cache = Default::default();
            self.object_finder = Default::default();
        }
        Ok(r)
    }

    /// Replace the active document's active layer without creating a history step.
    pub fn select_layer(&mut self, id: LayerId) -> Result<()> {
        let st = self.active_mut().ok_or(EngineError::NoDocument)?;
        if st.doc.layer(id).is_none() {
            return Err(EngineError::NoLayer(id));
        }
        st.active_layer = Some(id);
        st.selected_layers = vec![id];
        st.layer_anchor = Some(id);
        channel_cmds::fix_view(st);
        // Selecting a layer is not an edit: keep a clean document clean.
        let clean = st.saved_revision == st.revision;
        st.revision += 1;
        st.view_revision += 1;
        if clean {
            st.saved_revision = st.revision;
        }
        st.last_damage = Some(photocraft_geom::Rect::EMPTY);
        Ok(())
    }

    /// Re-renders the pixels of type layers whose font arrived after they were drawn (the web
    /// build fetches served fonts on demand, `photocraft_text::served`). The layers' model is
    /// unchanged, so this is no history step and leaves a clean document clean. Unknown documents
    /// and layers are skipped. A document a background job is computing from must not move under
    /// it: its layers are returned, to try again later.
    pub fn refresh_type_layers(&mut self, layers: &[(photocraft_doc::DocId, LayerId)]) -> Vec<(photocraft_doc::DocId, LayerId)> {
        let mut busy = Vec::new();
        let mut docs: Vec<photocraft_doc::DocId> = Vec::new();
        for (d, _) in layers {
            if !docs.contains(d) {
                docs.push(*d);
            }
        }
        for doc_id in docs {
            let mine = layers.iter().filter(|(d, _)| *d == doc_id);
            if self.job_on(doc_id).is_some() {
                busy.extend(mine);
                continue;
            }
            let Some(st) = self.docs.iter_mut().find(|st| st.doc.id == doc_id) else { continue };
            let snapshot = st.doc.clone();
            let mut doc = (*snapshot).clone();
            let mut changed = false;
            for (_, id) in mine {
                if let Some(photocraft_doc::Layer { content: photocraft_doc::LayerContent::Text(t), .. }) = doc.layer_mut(*id) {
                    type_cmds::refresh(&snapshot, t);
                    changed = true;
                }
            }
            if changed {
                let clean = st.saved_revision == st.revision;
                st.doc = Arc::new(doc);
                st.revision += 1;
                if clean {
                    st.saved_revision = st.revision;
                }
                st.last_damage = None;
            }
        }
        busy
    }

    pub fn undo(&mut self) -> bool {
        // A background job is computing from the current state: it must not move under it.
        if self.active_job().is_some() {
            return false;
        }
        let Some(st) = self.active_mut() else { return false };
        st.coalesce = None;
        match st.history.undo(st.doc.clone()) {
            Some((mut d, layers)) => {
                // Save As changes file identity outside history, just like the saved path.
                if d.name != st.doc.name {
                    Arc::make_mut(&mut d).name.clone_from(&st.doc.name);
                }
                // Pixels this step can have touched, so the canvas recomposites only that
                // (it recomposited everything before).
                let damage = layer_multi_cmds::step_damage(&d, &st.doc);
                st.doc = d;
                restore_target(st, layers);
                st.revision += 1;
                st.last_damage = damage;
                true
            }
            None => false,
        }
    }

    pub fn redo(&mut self) -> bool {
        if self.active_job().is_some() {
            return false;
        }
        let Some(st) = self.active_mut() else { return false };
        st.coalesce = None;
        match st.history.redo(st.doc.clone()) {
            Some((mut d, layers)) => {
                if d.name != st.doc.name {
                    Arc::make_mut(&mut d).name.clone_from(&st.doc.name);
                }
                let damage = layer_multi_cmds::step_damage(&st.doc, &d);
                st.doc = d;
                restore_target(st, layers);
                st.revision += 1;
                st.last_damage = damage;
                true
            }
            None => false,
        }
    }
}

/// Undo / redo: target the layers the restored state targeted, as far as they still exist.
fn restore_target(st: &mut DocState, layers: LayerTarget) {
    // Deselect Layers is a valid history target, not a missing layer to replace with the top
    // layer. An edit after deselection must undo back to no active or selected layers (#3042).
    if layers.active.is_none() && layers.selected.is_empty() {
        st.active_layer = None;
        st.selected_layers.clear();
        st.layer_anchor = None;
        channel_cmds::fix_view(st);
        return;
    }
    if layers.active.is_some_and(|id| st.doc.layer(id).is_some()) {
        st.active_layer = layers.active;
        st.selected_layers = layers.selected;
    }
    fix_active(st);
}

fn fix_active(st: &mut DocState) {
    if st.active_layer.is_none_or(|id| st.doc.layer(id).is_none()) {
        st.active_layer = st.doc.top_layer();
    }
    fix_selection(st);
    channel_cmds::fix_view(st);
}

/// Keep the layer selection valid after the document or the active layer changed: drop deleted
/// layers, and collapse to the active layer when a command made a non-member active.
pub(crate) fn fix_selection(st: &mut DocState) {
    let Some(active) = st.active_layer else {
        st.selected_layers.clear();
        return;
    };
    if !st.selected_layers.contains(&active) {
        st.selected_layers = vec![active];
    } else {
        let live: std::collections::HashSet<LayerId> = st.doc.walk().into_iter().map(|(_, _, l)| l.id).collect();
        st.selected_layers.retain(|id| live.contains(id));
        let mut seen = std::collections::HashSet::new();
        st.selected_layers.retain(|id| seen.insert(*id));
    }
    if st.layer_anchor.is_some_and(|a| !st.selected_layers.contains(&a)) {
        st.layer_anchor = Some(active);
    }
}

#[cfg(test)]
mod fill_layer_mode_tests;
#[cfg(test)]
mod pattern_mode_tests;
#[cfg(test)]
mod tests;

#[cfg(test)]
mod fx_mode_tests;
