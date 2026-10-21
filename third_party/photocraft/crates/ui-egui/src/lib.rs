//! Photocraft's first UI shell, built on egui/eframe.
//!
//! This crate is deliberately thin. Every action goes through
//! [`photocraft_engine::Session::execute`], and all UI state lives in [`state::UiState`] (plain
//! data). The [`control`] module exposes both to automation, so agents can drive and inspect every
//! part of the interface.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

/// Translate a string literal into the current UI language: `tl!("Image Size…")`.
macro_rules! tl {
    ($s:expr) => {
        $crate::i18n::t($s)
    };
}

pub mod actions;
pub mod adjust_dialog;
pub mod adjust_editors;
pub mod adjust_preview;
pub mod adjust_ui;
mod alt_grab;
pub mod analysis_ui;
pub mod anchor_tools;
pub mod artboard_tool;
pub mod artboard_ui;
pub(crate) mod background_redraw;
mod blend_keys;
pub(crate) mod blend_preview;
pub mod blur_pins;
mod brand;
pub(crate) mod brush_cursor;
pub mod brush_panel;
pub mod brush_picker;
pub mod brush_preset_dialog;
pub mod brush_preview;
pub mod brush_resize;
pub mod brush_sections;
pub mod brushes_tab;
pub(crate) mod cam_preview;
mod cam_transform;
mod camera_raw_detail_ui;
mod camera_raw_preview_ui;
mod camera_raw_scope_ui;
pub mod camera_raw_ui;
pub mod canvas;
pub mod canvas_autoscroll;
mod canvas_popup;
pub mod canvas_tool_menu;
pub mod cas_transform;
pub mod channel_options_ui;
pub mod channel_view;
pub mod channels_panel;
pub mod chrome_ui;
pub mod cjk_fonts;
pub mod clip_line_ui;
pub mod color_picker_ui;
pub mod color_range_ui;
pub mod comps_ui;
pub mod control;
pub mod credits;
pub mod crop_mode;
pub mod crop_overlay;
pub mod crop_presets;
pub mod crop_shield;
pub mod crop_size;
pub mod crop_straighten;
pub mod crop_ui;
pub mod delete_layer_prompt;
pub mod dialog_alt_reset;
pub mod dialog_blend_ui;
pub mod dialogs;
pub mod direct_select;
pub mod direction;
pub mod discard_ui;
pub mod distort_ui;
pub mod doc_props_ui;
pub mod dock;
pub mod duplicate_layer_ui;
pub mod edit_history;
pub mod egui_fonts;
pub mod enable_rules;
pub mod eraser_ui;
pub mod export_dialog;
pub mod eyedropper_ui;
pub mod field_tab;
pub mod file_dialog;
pub mod file_open;
pub mod file_ui;
pub mod fill_ui;
pub mod filter_dialog;
mod filter_preview_ui;
#[cfg(not(target_arch = "wasm32"))]
mod filter_preview_worker;
pub mod flick_pan;
mod font_preview;
pub mod frame_tool;
pub mod gallery_ui;
pub mod gpu_canvas;
pub mod gpu_status;
pub mod gradient_ui;
pub mod guide_layout_ui;
pub mod hold_keys;
pub mod i18n;
mod icon_data;
pub mod icons;
pub mod jobs_ui;
pub mod kys_import;
pub mod lasso_ui;
pub(crate) mod last_brush;
pub mod layer_menu_ui;
pub mod layer_pick_ui;
pub mod layer_picker;
pub mod layer_props_ui;
mod layer_reveal;
pub mod layer_row_ui;
pub mod layer_style;
mod layer_transfer;
pub mod layer_tree_ui;
mod layout_keys;
pub mod links;
pub mod liquify_ui;
pub mod magnetic_lasso_ui;
mod mask_props_ui;
pub mod mask_thumbs_ui;
pub mod mask_trash_prompt;
pub mod menu_catalog;
pub mod menu_nav;
pub mod menus;
pub mod monitor_status;
pub mod move_box;
pub mod move_hover;
pub mod move_lock;
pub mod move_mods;
pub mod move_ui;
pub mod native_menu;
pub mod new_doc_ui;
pub mod new_guide_ui;
pub mod new_layer_ui;
pub mod notices;
mod numeric_expression;
pub mod object_finder;
mod opacity_keys;
pub mod open_prompt;
pub mod outline;
pub mod paint_mouse;
pub mod palette;
mod panel_docking;
pub mod panels;
pub mod parity;
pub mod patch_preview;
pub(crate) mod pattern_fill_ui;
pub mod pattern_preview;
pub mod perspective_crop_ui;
pub mod perspective_ui;
pub mod pixel_grid;
pub mod plugin_ui;
pub mod point_curve;
pub mod prefs_ui;
pub mod preset_files_ui;
pub mod preset_panels;
pub mod press_menu;
mod pressure_curve_ui;
pub mod props_layout;
pub mod proxy;
pub mod puppet_ui;
pub mod quick_pick;
pub mod rasterize_prompt;
pub mod retouch_ui;
mod rgb_histogram;
pub mod rotate_view;
pub mod rulers;
pub mod screen_picker;
pub mod scrollbars;
mod scrub_wrap;
pub mod select_mask_view;
pub mod served_fonts;
pub mod shape_dialog;
pub(crate) mod shape_fill_ui;
pub mod shape_stroke_ui;
pub mod shortcut_dispatch;
pub mod shortcuts;
mod single_line_drag;
mod sizing;
pub mod slice_ui;
pub mod smart_ui;
pub mod snap_ui;
pub(crate) mod solid_fill_ui;
pub mod spring_tools;
pub mod stack_dialog;
pub mod state;
pub mod stroke_constraint;
pub mod stroke_trace;
pub mod stroke_trail;
pub mod stroke_ui;
pub mod stylus;
pub mod swap_arrow;
pub mod swatches_ui;
pub mod symmetry_ui;
mod tab_strip;
pub mod targa_options_ui;
pub mod text_field_menu;
pub mod theme;
pub mod tiff_options_ui;
mod tile_overlay;
mod timeline_ui;
mod titlebar;
pub use titlebar::{Caption, CaptionFamily, CaptionIcon, CaptionIcons, CaptionStyle, RenderIconFn};
pub mod tone;
mod tool_cursor;
pub mod tool_feedback;
pub(crate) mod tool_memory;
mod tool_reset;
mod tool_shortcuts;
pub mod transform_fx;
pub mod transform_tex;
pub mod transform_tool;
mod type_font_preview;
pub mod type_panels_ui;
pub mod type_tool;
mod type_transform;
mod variables_ui;
pub mod vector_ui;
pub mod view_cmds;
pub mod vp_ui;
pub mod warp_preview;
pub mod wheel_nav;
pub mod wide_angle_ui;
pub mod widgets;
pub mod work_area;
pub mod workspace_ui;
pub mod zoom_levels;
pub mod zoom_tool;

use std::collections::HashMap;
use std::sync::mpsc::{Receiver, Sender};

use photocraft_doc::{DocId, Document};
use photocraft_engine::Session;
use serde_json::{Value, json};

pub use control::{ControlRequest, ControlResponse};
pub use file_dialog::{FileDialogAnswer, FileDialogFn, FileDialogReply, FileDialogRequest};
pub use file_open::OsEvent;
pub use state::{Tool, UiState};

/// Decode a file: (document, warnings about anything approximated or dropped).
pub type ImportFn = Box<dyn Fn(&str, &[u8], usize) -> Result<(Document, Vec<String>), String>>;
/// Encoder settings chosen in Export As (the file format comes from the name's extension).
#[derive(Clone, Debug, PartialEq)]
pub struct ExportSettings {
    /// JPEG quality 1–100 (None = codec default).
    pub jpeg_quality: Option<u8>,
    /// WebP: lossless (VP8L) rather than lossy (VP8 at [`Self::webp_quality`]).
    pub webp_lossless: bool,
    /// Lossy WebP quality 1–100 (None = codec default).
    pub webp_quality: Option<u8>,
    /// TIFF: keep the layers (Photoshop layer data); `false` is "Discard Layers and Save a Copy".
    pub tiff_layers: bool,
    /// Embed the document's whole XMP packet. `true` by default (Save As keeps the metadata);
    /// Export As starts at Metadata: None, since the packet can carry the text of every type
    /// layer and one id per placed document (#647).
    pub xmp_all: bool,
    /// Targa: bits per pixel from Targa Options or Export As (`None`: 32 with an alpha channel or
    /// transparency, else 24).
    pub tga_bits: Option<photocraft_io::TgaBits>,
    /// PSD/PSB: embed a thumbnail (Preferences › File Handling › Image Previews; see
    /// [`PhotocraftApp::save_settings`]).
    pub image_previews: bool,
    /// PSD/PSB: write the real merged image (Preferences › File Handling › Maximize PSD and PSB
    /// File Compatibility; see [`PhotocraftApp::save_settings`]).
    pub psd_maximize_compatibility: bool,
}

impl Default for ExportSettings {
    fn default() -> Self {
        ExportSettings {
            jpeg_quality: None,
            webp_lossless: true,
            webp_quality: None,
            tiff_layers: true,
            xmp_all: true,
            tga_bits: None,
            image_previews: true,
            psd_maximize_compatibility: true,
        }
    }
}

/// The document's lone layer when it is a plain raster a flat file keeps: visible, at full
/// opacity, in Normal mode, with no masks, effects or blending restrictions (#2234).
pub(crate) fn plain_raster(doc: &Document) -> Option<&photocraft_doc::Layer> {
    match doc.layers.as_slice() {
        [layer]
            if matches!(layer.content, photocraft_doc::LayerContent::Raster(_))
                && layer.visible
                && layer.opacity >= 1.0
                && layer.fill_opacity >= 1.0
                && layer.mask.is_none()
                && layer.vector_mask.is_none()
                && layer.effects.items.is_empty()
                && layer.effects.psd_raw.is_none()
                && matches!(layer.blend, photocraft_color::BlendMode::Normal | photocraft_color::BlendMode::PassThrough)
                && photocraft_compose::channel_weights(layer, doc.mode).is_none()
                && !photocraft_compose::blend_if_active(layer, doc.mode) =>
        {
            Some(layer)
        }
        _ => None,
    }
}

/// Encode a document: (file bytes, warnings about anything approximated or dropped).
pub type ExportFn = Box<dyn Fn(&Document, &str, &ExportSettings) -> Result<(Vec<u8>, Vec<String>), String>>;
pub type WriteFn = Box<dyn FnMut(&str, &[u8]) -> Result<(), String>>;
/// Encode a document and write it to a path, reporting the stage to the job; returns the export
/// warnings. Runs on a worker thread (see [`Services::save_file`]). It must call
/// [`jobs_ui::SaveCtl::commit`] right before replacing the file, and write nothing if that fails.
pub type SaveFileFn = std::sync::Arc<dyn Fn(&Document, &str, &ExportSettings, &jobs_ui::SaveCtl) -> Result<Vec<String>, String> + Send + Sync>;

/// Who a finished save serves.
#[derive(Clone, Copy)]
pub(crate) enum SaveKind {
    /// Interactive: arm the document's file and fire user-configured hooks.
    User,
    /// Automation: flat formats stay copies (#2579) and hooks never fire.
    Automation,
}
/// Read bytes through the desktop control session's authorized read root.
pub type AutomationReadFn = Box<dyn FnMut(&str) -> Result<(String, Vec<u8>), String>>;
/// Write bytes through the desktop control session's authorized write root.
pub type AutomationWriteFn = Box<dyn FnMut(&str, &[u8]) -> Result<(), String>>;
/// Load a directory-form `.pcraft` through the authorized read root, returning its display
/// name (the file name) and the loaded document. `Ok(None)` marks a regular file: the
/// caller falls back to the bytes path for the zip form.
pub type AutomationOpenPcraftFn = Box<dyn FnMut(&str) -> Result<Option<(String, Document)>, String>>;
/// Reject engine commands which still perform ambient filesystem I/O.
pub type AutomationCommandFn = Box<dyn Fn(&str, &Value) -> Result<(), String>>;
/// Open a URL in the system browser (native) — reliable cross-platform, unlike `ctx.open_url`.
pub type OpenUrlFn = Box<dyn Fn(&str) -> Result<(), String>>;
pub type EncodePngFn = Box<dyn Fn(u32, u32, &[u8]) -> Result<Vec<u8>, String>>;
/// Put an RGBA8 image (width, height, pixels) on the OS clipboard.
pub type ClipboardSetFn = Box<dyn FnMut(u32, u32, &[u8]) -> Result<(), String>>;
/// Read an RGBA8 image from the OS clipboard.
pub type ClipboardGetFn = Box<dyn FnMut() -> Option<(u32, u32, Vec<u8>)>>;
/// Shared queue of (file name, bytes) delivered asynchronously.
pub type Inbox = std::sync::Arc<std::sync::Mutex<Vec<(String, Vec<u8>)>>>;
/// Read the saved preferences text (`None` when there is none yet).
pub type LoadTextFn = Box<dyn FnMut() -> Option<String>>;
/// Persist the preferences text.
pub type SaveTextFn = Box<dyn FnMut(&str) -> Result<(), String>>;
/// Platform appearance when egui cannot detect it (for example, Wayland without a theme event).
pub type SystemThemeFn = Box<dyn Fn(&egui::Context) -> Option<egui::Theme>>;
/// Autosave a document snapshot for crash recovery: (snapshot, revision, original path).
pub type AutosaveFn = Box<dyn FnMut(&std::sync::Arc<Document>, u64, Option<&str>) -> Result<(), String>>;
/// Poll successful or failed background writes: (document id, revision, result).
pub type AutosaveResultsFn = Box<dyn FnMut() -> Vec<(u64, u64, Result<(), String>)>>;
/// Drop the recovery data of a document (by `DocId` value) once it is saved or closed.
pub type DiscardAutosaveFn = Box<dyn FnMut(u64)>;
/// List recoverable documents without decoding them. Their data stays until the documents are
/// saved or closed; the shell runs each entry's loader on a background worker.
pub type RecoverFn = Box<dyn FnMut() -> Vec<Recoverable>>;
/// Photoshop's own keyboard shortcut set on this machine, as (source path, `.kys` XML text):
/// the newest install's live `Keyboard Shortcuts.psp` on the desktop, `None` without one.
pub type PhotoshopShortcutsFn = Box<dyn FnMut() -> Option<(String, String)>>;
/// A recovered document (by `DocId` value, once open) takes over its recovery entry (by key):
/// its autosaves replace the entry, and saving or closing it drops the entry.
pub type AdoptAutosaveFn = Box<dyn FnMut(u64, &str)>;

/// A recovery entry [`RecoverFn`] found; its loader owns only the data it needs to read.
pub struct Recoverable {
    /// The recovery entry to adopt once loading succeeds (see [`AdoptAutosaveFn`]).
    pub key: String,
    pub name: String,
    /// Where the user last saved it, if anywhere.
    pub path: Option<String>,
    pub load: Box<dyn FnOnce() -> Result<Document, String> + Send + 'static>,
}
/// Append text to a file (History Log).
pub type AppendTextFn = Box<dyn FnMut(&str, &str) -> Result<(), String>>;
/// Requests from the operating system since the last call (see [`OsEvent`]).
pub type OsEventsFn = Box<dyn FnMut() -> Vec<OsEvent>>;
/// Ends the app the platform's own way (see [`Services::quit`]).
pub type QuitFn = Box<dyn FnMut()>;
/// Where the OS pointer is now, in egui points within the window; `None` when unknown.
pub type CursorPosFn = Box<dyn FnMut(&egui::Context) -> Option<egui::Pos2>>;
/// Whether Caps Lock is toggled on, read from the OS. `None` where the platform cannot
/// report it (native Wayland): the cursor then follows the cursor preference (#1758).
pub type CapsLockFn = Box<dyn FnMut() -> bool>;
/// Keeps an undecorated window's OS frame from offsetting its content and pointer (#2246).
/// Called once per frame; `None` when the window is decorated or the platform needs no fixing.
pub type WindowFrameFn = Box<dyn FnMut()>;
/// The caption buttons' current style ([`Services::caption_style`]).
pub type CaptionStyleFn = Box<dyn Fn(&egui::Context) -> CaptionStyle>;

/// Platform services injected by the app binary (file dialogs, codecs), keeping this crate free of
/// I/O dependencies.
#[derive(Default)]
pub struct Services {
    /// Embedded editors commit through the host's page save queue.
    pub host_save: Option<Box<dyn FnMut()>>,
    /// User-initiated desktop/browser pixel sampling.
    pub screen_pick: Option<screen_picker::Service>,
    /// Decode a file's bytes into a document (PSD, PNG, JPEG, …).
    pub import: Option<ImportFn>,
    /// Encode a document for a file name (format chosen by extension).
    pub export: Option<ExportFn>,
    /// Background save over the automation write authority (`app.save`, #2017): same worker
    /// contract as `save_file`, but the bytes go through the capability-scoped workspace.
    pub automation_save: Option<SaveFileFn>,
    /// Show an Open or Save dialog without waiting for it (see `file_dialog`).
    pub file_dialog: Option<FileDialogFn>,
    /// Write bytes to a path (native) or trigger a download (web).
    pub write: Option<WriteFn>,
    /// Encode and write in one step on a worker thread, so a large save doesn't freeze the
    /// window (#2017). With background jobs on, saves use it instead of `export` and `write`.
    pub save_file: Option<SaveFileFn>,
    /// File access used only by control/MCP requests. Interactive dialogs keep
    /// using `file_dialog` and `write` with the user's authority.
    pub automation_read: Option<AutomationReadFn>,
    pub automation_write: Option<AutomationWriteFn>,
    pub automation_open_pcraft: Option<AutomationOpenPcraftFn>,
    pub automation_command: Option<AutomationCommandFn>,
    /// Same policy as [`Self::automation_command`], as a function pointer the engine calls for
    /// each step of `actions.play`. Installed on the session only while a control request or
    /// an automation-driven [`PhotocraftApp::run`] runs, so a local play of a recorded `file.*` step still works.
    pub automation_authorize: Option<fn(&str, &serde_json::Value) -> photocraft_engine::Result<()>>,
    /// Encode an RGBA8 image as PNG (used for screenshots and `ui.render`).
    pub encode_png: Option<EncodePngFn>,
    /// Open a URL in the system browser (native). Falls back to `ctx.open_url` (web) when unset.
    pub open_url: Option<OpenUrlFn>,
    /// Preferences ▸ General ▸ Beep When Done: the platform's default beep, played when a
    /// background job finishes successfully. `None` in the web build (and in tests).
    pub beep: Option<fn()>,
    /// Files delivered asynchronously (web drag-and-drop): drained every frame.
    pub inbox: Option<Inbox>,
    /// OS clipboard images: copies go out, screenshots and images from other apps come in.
    pub clipboard_set_image: Option<ClipboardSetFn>,
    pub clipboard_get_image: Option<ClipboardGetFn>,
    /// Preferences persistence: the platform config directory on native, browser storage on
    /// the web (see `prefs_ui`).
    pub load_prefs: Option<LoadTextFn>,
    pub save_prefs: Option<SaveTextFn>,
    /// Photoshop's live keyboard shortcut set, imported once at first launch (`kys_import`).
    pub photoshop_shortcuts: Option<PhotoshopShortcutsFn>,
    pub system_theme: Option<SystemThemeFn>,
    /// The native window is connected directly to a Wayland compositor.
    pub is_wayland: bool,
    /// On Wayland, the shell command that starts this install under XWayland, where native file
    /// drops work (#386); `None` when there is no X server to run it on.
    pub xwayland_command: Option<String>,
    /// Crash-recovery autosave (Preferences › File Handling) and recovery at launch.
    pub autosave: Option<AutosaveFn>,
    /// None for services that report synchronous success through `autosave`.
    pub autosave_results: Option<AutosaveResultsFn>,
    pub discard_autosave: Option<DiscardAutosaveFn>,
    pub recover: Option<RecoverFn>,
    pub adopt_autosave: Option<AdoptAutosaveFn>,
    /// History Log text file output.
    pub append_text: Option<AppendTextFn>,
    /// OS requests (macOS open-documents / quit Apple events), polled every frame.
    pub os_events: Option<OsEventsFn>,
    /// The pointer position read from the OS (desktop): winit 0.30's file drops carry none, and
    /// the window gets no pointer events during an OS drag (see `file_open::DropTarget`).
    pub cursor_pos: Option<CursorPosFn>,
    /// Caps Lock toggled on (desktop; `None` on Wayland and the web). Read once per frame, so
    /// the canvas can show the precise crosshair for painting tools, whatever the preference.
    pub caps_lock: Option<CapsLockFn>,
    /// Keeps the undecorated window borderless, so the content and the pointer stay aligned
    /// (Windows custom title bar, #2246). Called once per frame; `None` when the window is
    /// decorated or the platform needs no fixing.
    pub window_frame: Option<WindowFrameFn>,
    /// How the custom title bar's caption buttons look and where they sit (Linux: the desktop's
    /// style and button layout). Read every frame, with the context to wake when it changes;
    /// `None` keeps Windows' buttons at the right end.
    pub caption_style: Option<CaptionStyleFn>,
    /// The persistent brush preset store, loading in the background (desktop; see
    /// `photocraft_engine::preset_store`). Attached to the session once it arrives; without
    /// one, brush presets are session-only unless the shell attached a store before startup.
    pub preset_store: Option<std::sync::mpsc::Receiver<photocraft_engine::preset_store::Opened>>,
    /// Reads the displays and their ICC profiles in the background (desktop macOS; see
    /// `monitor_status`). Without one, the canvas uses the profile chosen in Color Settings, or sRGB.
    pub read_displays: Option<monitor_status::ReadDisplaysFn>,
    /// The macOS menu bar, when the desktop app installed one; the in-window menus are hidden then.
    pub native_menu: Option<native_menu::NativeMenu>,
    /// Ends the app once leaving is settled (nothing unsaved, or the prompt answered), instead of
    /// letting eframe close the window. macOS: `-[NSApplication terminate:]`. Closing the window
    /// while AppKit's run loop is still going crashes or hangs Touch Bar Macs (#1575, #1458).
    /// Without one, eframe closes the window and the app ends with it.
    pub quit: Option<QuitFn>,
}

/// A document histogram being computed off the UI thread: (document, revision, receiver of
/// (document, compute ms, histograms)).
pub(crate) type HistJob = (DocId, u64, std::sync::mpsc::Receiver<(DocId, f64, std::sync::Arc<tone::Histograms>)>);

/// The Info panel's cached sample: pixel x, y, document revision and Eyedropper Sample Size.
type InfoSampleKey = (i32, i32, u64, u32);

pub struct PhotocraftApp {
    pub session: Session,
    pub ui: UiState,
    pub services: Services,
    /// Canvas caches per (document, display): CPU textures hold monitor values; the GPU
    /// canvas state is shared (`canvas::GPU_OUTPUT`).
    canvases: HashMap<(DocId, u32), canvas::CanvasCache>,
    /// Navigator textures belong to open documents, not the lifetime of the egui context.
    navigator_textures: HashMap<DocId, canvas::NavigatorCache>,
    /// `canvas::display_doc`'s live-adjust and Duotone documents per document: (snapshot,
    /// preview key, the document shown).
    display_memo: HashMap<DocId, (std::sync::Weak<Document>, u64, std::sync::Arc<Document>)>,
    /// The main canvas's last GPU view of each document while it shows the plain document on
    /// the GPU canvas: the Navigator draws the same texture at its own size (#4518).
    navigator_gpu: HashMap<DocId, crate::gpu_canvas::ViewParams>,
    /// Documents whose GPU canvas texture holds the plain document (`canvas::ensure_gpu`).
    navigator_plain: std::collections::HashSet<DocId>,
    /// Display profile readings (#569).
    monitors: monitor_status::State,
    checker: Option<egui::TextureHandle>,
    drag: Option<canvas::Drag>,
    /// A Move-tool press landed on a locked layer: the first pointer move shows Photoshop's
    /// message (`move_lock`), a plain click shows nothing.
    pub(crate) move_blocked: bool,
    /// The tool pointer events go to this frame when it isn't the selected one: the Move tool
    /// while ⌘ is held (`hold_keys::cmd_moves`). Set by the canvas for its gestures, never saved.
    pub(crate) tool_override: Option<state::Tool>,
    /// Brush/Eraser stroke being drawn, rendered by the engine (see `canvas::LiveStroke`).
    live_stroke: Option<canvas::LiveStroke>,
    /// A live stroke just committed with its own pixels: the canvas caches that showed it
    /// refresh only what it drew after the step they showed (`canvas::damage_since`).
    committed_stroke: Option<canvas::CommittedStroke>,
    /// Footprint trail of a retouching drag (see `stroke_trail`).
    trail: Option<stroke_trail::Trail>,
    /// Remove tool strokes waiting to be applied together (Remove after each stroke off), the
    /// document they belong to, and their tint on the canvas (`retouch_ui`).
    pub(crate) remove_pending: Vec<Vec<[f64; 3]>>,
    pub(crate) remove_pending_doc: Option<photocraft_doc::DocId>,
    pub(crate) remove_trail: Option<stroke_trail::Trail>,
    /// The open menus' items, reused between frames (`menus::ItemCache`).
    pub(crate) menu_cache: Option<menus::ItemCache>,
    /// Move tool drag shown live (`move_ui`).
    pub(crate) move_preview: Option<move_ui::MovePreview>,
    /// Move tool › Show Highlight on Rollover: the last hover answer (`move_hover`).
    pub(crate) move_hover: Option<move_hover::Hover>,
    /// Object Selection › Object Finder: the hover preview (`object_finder`).
    pub(crate) object_finder: Option<object_finder::Finder>,
    /// A blend mode hovered in the Layers panel, shown live (`blend_preview`).
    pub(crate) blend_preview: Option<blend_preview::BlendPreview>,
    /// Patch Tool drag: the healed document at the pointer (`patch_preview`).
    pub(crate) patch_preview: Option<patch_preview::PatchPreview>,
    /// Content-Aware Move drag or box shown with its result (`cam_preview`).
    pub(crate) cam_preview: Option<cam_preview::CamPreview>,
    /// The pixels a Magnetic Lasso border follows (`magnetic_lasso_ui`).
    pub(crate) magnetic: magnetic_lasso_ui::Runtime,
    /// The next tool `Down` is a right-button drag that erases (see `paint_mouse`).
    secondary_erase: bool,
    /// The Brush or Pencil erase key is held this frame (`hold_keys::erase_held`, #3439).
    held_erase: bool,
    /// While a batch of recovered pointer samples is replayed, defer the live-stroke update to one
    /// call for the whole frame (see `canvas::canvas_view`).
    defer_live_stroke: bool,
    /// A live painting stroke started on the press (`canvas_view`): the drag egui recognises later,
    /// or the click, continues or ends it rather than starting another.
    press_stroke: bool,
    /// Position and pen pressure at the last stroke's endpoint: ⇧-click interpolates from them (#178, #2827).
    last_stroke_end: Option<(DocId, [f64; 3])>,
    /// Control+Alt-drag brush resize in progress (`brush_resize`, #231).
    pub(crate) brush_resize: Option<brush_resize::Resize>,
    /// A ⌘⌥⌃-click layer pick is in progress; its drag and release are swallowed (`quick_pick`).
    pub(crate) quick_pick: bool,
    /// The next tool `Down` is an Alt+right-drag that resizes the brush (#297). `tool_event`
    /// takes it on every event, so a press another handler consumes can't leave it set.
    pub(crate) brush_resize_armed: bool,
    /// This press began with ⌥ (Alt) held on a painting tool, so it samples colours instead of
    /// painting until it is released (`canvas::alt_eyedropper`, #417).
    pub(crate) alt_sampling: bool,
    /// Caps Lock toggled on, read from the OS each frame (`services.caps_lock`): painting tools
    /// show the precise crosshair whatever the cursor preference (#1758).
    pub caps_lock: bool,
    /// The first digit of a two-digit opacity typed on the number keys (`opacity_keys`, #352).
    pub(crate) opacity_keys: opacity_keys::Pending,
    control_rx: Option<Receiver<ControlRequest>>,
    pending_screenshots: Vec<(u64, Option<String>, Sender<ControlResponse>)>,
    /// Screenshots not yet requested from the viewport: (token, earliest time in ms, frames seen).
    queued_screenshots: Vec<(u64, f64, u32)>,
    /// Control replies waiting for queued synthetic input to be processed.
    input_waiters: Vec<Sender<ControlResponse>>,
    /// Live (uncommitted) adjustment edit shown on canvas while a slider is dragged.
    pub live_adjust: Option<(photocraft_doc::LayerId, Value)>,
    /// Frames rendered (for tests and the status bar).
    pub frame: u64,
    /// The pointer rested on the notice stack last frame. Used to give a fresh auto-hide delay in
    /// the frame the pointer leaves, so a long stationary hover never counts as elapsed time
    /// (#2022); set by `notices::show`.
    pub(crate) notices_hovered: bool,
    /// Apply theme on first frame.
    styled: bool,
    /// Whether the window uses an integrated (transparent) macOS title bar.
    pub integrated_titlebar: bool,
    /// Windows and Linux: the window has no OS decorations and the app's top bar is the title bar
    /// (caption buttons, window dragging and edge resizing, `titlebar`).
    pub custom_titlebar: bool,
    /// Last window title sent to the OS (`ViewportCommand::Title`, see `panels::sync_window_title`):
    /// sent again only when it changes, so idle frames don't spam the backend.
    last_window_title: String,
    fonts_ready: bool,
    /// Screen rect of the main canvas last frame (for overlays and the navigator).
    pub last_canvas_rect: egui::Rect,
    /// Physical pixels per egui point of the canvas last frame (`ctx.pixels_per_point`, which
    /// folds in both the display scale and Interface › UI Scale). The canvas maps document pixels
    /// to physical pixels, so point-space geometry divides the view zoom by this (see
    /// [`Self::point_zoom`]).
    pub ppp: f32,
    /// The document area showing the active document's canvas last frame (not the tabs, the
    /// start screen or an opening file's card): files dropped here are placed as layers.
    pub(crate) drop_canvas_rect: Option<egui::Rect>,
    /// The document tab strip last frame: a drop there opens the file at the slot under it.
    pub(crate) tab_strip: Option<canvas::TabStrip>,
    /// Files dropped on the canvas still to place, one Free Transform at a time.
    pub(crate) drop_places: std::collections::VecDeque<egui::DroppedFileHandle>,
    /// The document each of `ui.views` belongs to, as of the last [`Self::sync_views`].
    view_docs: Vec<DocId>,
    pub fps: f32,
    last_frame_time: f64,
    thumbs: HashMap<(photocraft_doc::LayerId, u8), (u64, egui::TextureHandle)>,
    /// History snapshot thumbnails by (document, snapshot id): a snapshot never changes, so each
    /// is made once (`snapshot_thumb`, #2966).
    snapshot_thumbs: HashMap<(DocId, u64), egui::TextureHandle>,
    /// Snapshots whose live layer/mask keys were last used to prune thumbnail handles.
    thumb_documents: Vec<(DocId, std::sync::Weak<Document>)>,
    /// Content bounds cached per (key, revision): scanning a 36 MP layer every frame cost ~77 ms.
    bounds_cache: HashMap<u64, (u64, photocraft_geom::Rect)>,
    /// Downsampled proxy of the active document for live previews: (doc, revision, k, proxy).
    pub(crate) proxy: Option<(DocId, u64, u32, std::sync::Arc<Document>)>,
    /// Key of the preview currently uploaded to the GPU (doc, params hash).
    pub(crate) proxy_uploaded: Option<(DocId, u64)>,
    /// Selection outline cache: (doc, revision, segments).
    /// Live filter preview (proxy document with the filter applied).
    pub(crate) filter_preview: Option<filter_dialog::FilterPreview>,
    #[cfg(not(target_arch = "wasm32"))]
    filter_preview_worker: filter_preview_worker::Worker,
    /// Full CPU canvas redraws running in the background, by document (#4024).
    pub(crate) redraws: std::collections::HashMap<photocraft_doc::DocId, background_redraw::Redraw>,
    /// Select › Color Range dialog preview (proxy document + mask / image textures).
    pub(crate) color_range: Option<color_range_ui::Preview>,
    /// Image › Adjustments dialog preview through a temporary clipped adjustment layer.
    pub(crate) adjust_preview: Option<adjust_preview::AdjustPreview>,
    /// Synthetic input events queued by automation (`ui.click`, `ui.key`, …), injected next frame.
    pub(crate) synthetic: Vec<egui::Event>,
    /// True only while a frame is processing synthetic automation input. It
    /// makes shortcut- and dialog-triggered commands pass the same policy as
    /// direct control calls.
    pub(crate) automation_input: bool,
    /// Levels/Curves histogram cache: (document, adjustment layer, revision it is valid for).
    pub(crate) tone_hist: Option<(DocId, photocraft_doc::LayerId, u64, std::sync::Arc<tone::Histograms>)>,
    /// Histogram panel cache: (document, revision, computed at ms, histograms).
    pub(crate) doc_hist: Option<(DocId, u64, f64, std::sync::Arc<tone::Histograms>)>,
    /// The document histogram being computed on a worker thread: (document, revision, receiver).
    pub(crate) hist_job: Option<HistJob>,
    /// Free Transform preview (document without the moving pixels + their texture).
    pub(crate) transform_preview: Option<transform_tool::TransformPreview>,
    /// Move-tool ⇧/⌥ drag state (move_mods).
    pub(crate) move_mods: move_mods::MoveDrag,
    /// Cached document for a modal text Style Options color preview.
    pub(crate) text_style_preview: Option<type_panels_ui::color_picker::Preview>,
    /// Live Layer Style dialog preview: (key over revision + style fields, preview or validation error).
    pub(crate) style_preview: Option<(u64, Result<std::sync::Arc<Document>, String>)>,
    pub(crate) solid_fill_preview: Option<solid_fill_ui::Preview>,
    /// Live Pattern Fill dialog preview (pattern_fill_ui).
    pub(crate) pattern_fill_preview: Option<pattern_fill_ui::Preview>,
    /// Shape fill Color Picker preview (shape_fill_ui).
    pub(crate) shape_fill_preview: Option<shape_fill_ui::Preview>,
    /// Liquify dialog, Puppet Warp and Perspective Warp sessions (distort_ui).
    pub(crate) distort: distort_ui::Distort,
    /// Gradient tool live-mode drags and previews (gradient_ui).
    pub(crate) gradient: gradient_ui::LiveGradient,
    /// Filter › Camera Raw Filter dialog (camera_raw_ui).
    pub(crate) camera_raw: Option<camera_raw_ui::CameraRawDialog>,
    /// A raw file just opened interactively, waiting for the open-time Camera Raw dialog (shown
    /// on the next frame, which has the egui context).
    pub(crate) pending_raw_open: Option<camera_raw_ui::RawOpen>,
    /// The open-time re-develop (or its Camera Raw step) running in the background.
    pub(crate) raw_redevelop: Option<camera_raw_ui::Redevelop>,
    /// Filter › Adaptive Wide Angle dialog (wide_angle_ui).
    pub(crate) wide_angle: Option<wide_angle_ui::WideAngleDialog>,
    /// ⌘Z history of each open parameter dialog (edit_history).
    pub(crate) dialog_edits: edit_history::DialogEdits,
    /// The filter a "Rasterize?" prompt answered for, run after the dialog closes, where the
    /// egui context is available again (rasterize_prompt).
    pub(crate) pending_after_prompt: Option<(String, serde_json::Value)>,
    /// Signature of the image we last put on the OS clipboard (to tell ours from other apps').
    os_clip_sig: Option<u64>,
    /// The session clipboard currently holds an image imported from the OS clipboard: it has no
    /// original position, so Paste centres it (see `menus` "edit.paste").
    pub(crate) clip_external: bool,
    /// The OS clipboard was already read for the paste in flight (Edit › Paste positions it first),
    /// so `run` doesn't read it a second time.
    pub(crate) clip_read_for_paste: bool,
    /// Pointer position over the canvas (document px), for the Info panel and status bar.
    pub(crate) hover_doc: Option<[f64; 2]>,
    /// Layer hovered in the Layers panel, outlined on the canvas.
    pub(crate) hover_layer: Option<photocraft_doc::LayerId>,
    /// Layer hovered on the canvas, highlighted in the Layers panel.
    pub(crate) canvas_hover_layer: Option<photocraft_doc::LayerId>,
    pub(crate) clone_preview: Option<crate::canvas::ClonePreviewCache>,
    /// Info panel sample cache: ((x, y, revision, Sample Size), composite RGBA).
    info_sample: Option<(InfoSampleKey, [f32; 4])>,
    /// Guide being dragged (from a ruler or with the Move tool).
    pub(crate) guide_drag: Option<rulers::GuideDrag>,
    /// Move tool marquee selecting layers (`move_box`).
    pub(crate) layer_box: Option<move_box::LayerBox>,
    /// Crop tool gesture in progress (see `crop_ui`).
    pub(crate) crop: crop_ui::CropState,
    /// Perspective Crop tool gesture and clicked corners (see `perspective_crop_ui`).
    pub(crate) pcrop: perspective_crop_ui::PerspectiveCrop,
    /// Type tool layout cache: ((doc, revision, layer), layout).
    pub(crate) type_layout: Option<((u64, u64, u64), std::sync::Arc<photocraft_text::TextLayout>)>,
    /// Type tool: (layer, clicks in the current run) for word / line / paragraph / all selection.
    pub(crate) type_clicks: Option<(u64, u32)>,
    pub(crate) type_transform_preview: Option<type_transform::Preview>,
    /// A font hovered in the Type tool's font picker, shown live (`type_font_preview`).
    pub(crate) type_font_preview: Option<type_font_preview::TypeFontPreview>,
    /// Channel thumbnails for one document snapshot; view-only revisions reuse their pixels.
    channel_thumbs: Option<(DocId, std::sync::Weak<Document>, bool, Vec<egui::TextureHandle>)>,
    /// Channels panel overlays / channel views drawn over the canvas, per document id.
    pub(crate) channel_views: HashMap<u64, channel_view::Cache>,
    /// Selection outline keyed by (document, mask identity × step × visible region).
    pub(crate) outline_cache: Option<(DocId, u64, std::sync::Arc<Vec<outline::Segment>>)>,
    /// GPU canvas renderer, when running on the wgpu backend (see [`Self::set_wgpu`]). Dropped
    /// for the rest of the session when the device is lost (see `gpu_status`).
    gpu: Option<gpu_canvas::GpuCanvas>,
    /// Run once the first frames have rendered (see [`Self::on_started`]).
    started: Option<gpu_status::StartedHook>,
    /// Frame and canvas-upload timings (exposed via `ui.inspect`).
    pub perf: gpu_canvas::Perf,
    /// Per-frame stroke timings for latency reports (`PHOTOCRAFT_STROKE_TRACE`, see `stroke_trace`).
    pub stroke_trace: stroke_trace::StrokeTrace,
    /// Preferences, autosave and snapping runtime state (see `prefs_ui`, `snap_ui`).
    pub(crate) prefs_rt: prefs_ui::Runtime,
    /// Close, Revert or Exit parked behind the unsaved-changes prompt (see `discard_ui`).
    pub(crate) discard: Option<discard_ui::Prompt>,
    /// The Open or Save dialog in progress, and the action waiting on it (see `file_dialog`).
    pub(crate) file_dialog: Option<file_dialog::Pending>,
    /// A Save As to a layered TIFF parked behind the TIFF Options prompt (see `tiff_options_ui`).
    pub(crate) tiff_options: Option<tiff_options_ui::Prompt>,
    /// Where each document was last saved or exported to through a dialog (#1826): its next Save
    /// As or export dialog starts there rather than beside the document. Session-only.
    pub(crate) save_dirs: HashMap<photocraft_doc::DocId, std::path::PathBuf>,
    /// A save to an RGB Targa parked behind the Targa Options prompt (see `targa_options_ui`).
    pub(crate) targa_options: Option<targa_options_ui::Prompt>,
    /// Files read but not opened yet: the first waits on its profile prompt, the rest behind it
    /// (see `open_prompt`).
    pub(crate) held_opens: std::collections::VecDeque<open_prompt::Held>,
    /// Set once the user has agreed to quit, so the resulting close request goes through.
    pub(crate) allow_close: bool,
    /// Pen pressure/tilt from the platform (see `stylus`).
    pub stylus: stylus::Stylus,
    /// Run long commands and file opens as background jobs with progress and Cancel (#210; see
    /// `jobs_ui`). The desktop app turns it on; off (the default), everything runs inline as
    /// before, which tests and scripts rely on.
    pub background_jobs: bool,
    /// Background job bookkeeping: opening tabs, control replies waiting on a job.
    pub jobs: jobs_ui::JobsUi,
    #[cfg(all(debug_assertions, not(target_arch = "wasm32")))]
    live_tokens: theme::live::LiveTokens,
}

impl PhotocraftApp {
    pub fn new(session: Session, services: Services) -> Self {
        let mut app = Self {
            session,
            ui: UiState::default(),
            services,
            canvases: HashMap::new(),
            navigator_textures: HashMap::new(),
            display_memo: HashMap::new(),
            navigator_gpu: HashMap::new(),
            navigator_plain: std::collections::HashSet::new(),
            monitors: Default::default(),
            checker: None,
            drag: None,
            move_blocked: false,
            tool_override: None,
            live_stroke: None,
            committed_stroke: None,
            trail: None,
            remove_pending: Vec::new(),
            remove_pending_doc: None,
            remove_trail: None,
            menu_cache: None,
            move_preview: None,
            move_hover: None,
            object_finder: None,
            blend_preview: None,
            patch_preview: None,
            cam_preview: None,
            magnetic: Default::default(),
            secondary_erase: false,
            held_erase: false,
            defer_live_stroke: false,
            press_stroke: false,
            last_stroke_end: None,
            brush_resize: None,
            quick_pick: false,
            brush_resize_armed: false,
            alt_sampling: false,
            caps_lock: false,
            opacity_keys: None,
            control_rx: None,
            pending_screenshots: Vec::new(),
            queued_screenshots: Vec::new(),
            input_waiters: Vec::new(),
            live_adjust: None,
            frame: 0,
            notices_hovered: false,
            styled: false,
            integrated_titlebar: false,
            custom_titlebar: false,
            last_window_title: String::new(),
            fonts_ready: false,
            last_canvas_rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)),
            ppp: 1.0,
            drop_canvas_rect: None,
            tab_strip: None,
            drop_places: Default::default(),
            view_docs: Vec::new(),
            fps: 0.0,
            last_frame_time: 0.0,
            thumbs: HashMap::new(),
            snapshot_thumbs: HashMap::new(),
            thumb_documents: Vec::new(),
            bounds_cache: HashMap::new(),
            proxy: None,
            proxy_uploaded: None,
            outline_cache: None,
            filter_preview: None,
            #[cfg(not(target_arch = "wasm32"))]
            filter_preview_worker: Default::default(),
            redraws: Default::default(),
            color_range: None,
            adjust_preview: None,
            synthetic: Vec::new(),
            automation_input: false,
            channel_thumbs: None,
            channel_views: HashMap::new(),
            type_layout: None,
            type_clicks: None,
            type_transform_preview: None,
            type_font_preview: None,
            guide_drag: None,
            layer_box: None,
            crop: Default::default(),
            pcrop: Default::default(),
            hover_doc: None,
            hover_layer: None,
            canvas_hover_layer: None,
            clone_preview: None,
            info_sample: None,
            os_clip_sig: None,
            clip_external: false,
            clip_read_for_paste: false,
            transform_preview: None,
            move_mods: Default::default(),
            style_preview: None,
            text_style_preview: None,
            solid_fill_preview: None,
            pattern_fill_preview: None,
            shape_fill_preview: None,
            distort: Default::default(),
            gradient: Default::default(),
            camera_raw: None,
            pending_raw_open: None,
            raw_redevelop: None,
            wide_angle: None,
            dialog_edits: Default::default(),
            pending_after_prompt: None,
            tone_hist: None,
            doc_hist: None,
            hist_job: None,
            gpu: None,
            started: None,
            perf: Default::default(),
            stroke_trace: stroke_trace::StrokeTrace::from_env(),
            prefs_rt: Default::default(),
            discard: None,
            file_dialog: None,
            tiff_options: None,
            save_dirs: HashMap::new(),
            targa_options: None,
            held_opens: Default::default(),
            allow_close: false,
            stylus: Default::default(),
            background_jobs: false,
            jobs: Default::default(),
            #[cfg(all(debug_assertions, not(target_arch = "wasm32")))]
            live_tokens: theme::live::LiveTokens::from_env(),
        };
        // Saved preferences are in place before the first frame; recovery starts in upkeep.
        prefs_ui::load(&mut app);
        // After the saved preferences, which say whether the set was already offered.
        kys_import::offer_import(&mut app);
        notices::wayland_file_drop_guidance(&mut app);
        // File › Scripts › Script Events Manager: "Start Application".
        photocraft_engine::automate_cmds::fire_event(&mut app.session, "startApplication");
        app
    }

    /// Draw the document canvas on the GPU (custom WGSL shader) instead of via egui textures.
    /// Call from the app creator with `cc.wgpu_render_state`; without it the CPU path is used.
    pub fn set_wgpu(&mut self, rs: eframe::egui_wgpu::RenderState) {
        // Preferences › Performance › cache tile size (PHOTOCRAFT_GPU_TILE still overrides).
        let tile = self.session.prefs().performance.cache_tile_size;
        // Escaped driver/setup panics must leave the session and CPU canvas alive.
        self.perf.gpu_info.set_adapter(&rs.adapter.get_info());
        let gpu = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| gpu_canvas::GpuCanvas::with_tile(&rs, Some(tile)))) {
            Ok(gpu) => gpu,
            Err(payload) => {
                let detail = payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                    .unwrap_or_else(|| "GPU canvas initialization failed".into());
                self.perf.gpu_info.canvas = "cpu".into();
                self.perf.gpu_info.fallback = Some(detail.clone());
                gpu_status::queue_fallback_notice(self, detail);
                return;
            }
        };
        self.perf.gpu_info.canvas = "gpu".into();
        self.gpu = Some(gpu);
        self.prefs_rt.gpu_style = None;
    }

    /// Whether the document canvas currently draws on the GPU (false on the CPU path, and after
    /// the device was lost).
    pub fn gpu_active(&self) -> bool {
        self.gpu.is_some()
    }

    /// The GPU canvas's device health flag (tests inject faults through it).
    pub fn gpu_health(&self) -> Option<photocraft_gpu::DeviceHealth> {
        self.gpu.as_ref().map(|g| g.health().clone())
    }

    /// Run `hook` once the app has rendered its first frames (and a document opened at launch
    /// has drawn): the desktop app clears its crash-safe GPU startup marker there.
    pub fn on_started(&mut self, hook: impl FnOnce(&mut PhotocraftApp) + 'static) {
        self.started = Some(Box::new(hook));
    }

    /// Per-frame GPU health check (called from `logic`; tests call it directly): falls back to
    /// the CPU canvas after a device loss and runs the started hook.
    pub fn check_gpu(&mut self, ctx: &egui::Context) {
        gpu_status::check(self, ctx);
    }

    /// Attach a control channel (requests arrive from a transport thread: TCP, stdin, tests).
    pub fn with_control(mut self, rx: Receiver<ControlRequest>) -> Self {
        self.control_rx = Some(rx);
        self
    }

    /// Run an engine command, reporting errors in the status bar.
    pub fn run(&mut self, id: &str, params: Value) -> Result<Value, String> {
        if self.services.host_save.is_some() && (menus::host_manages(id) || id == "layer.duplicate" && params.get("document").is_some()) {
            return Err("页面由 Saber 管理；调整内容请使用图层变换".into());
        }
        if let Some(result) = delete_layer_prompt::intercept(self, id, &params) {
            return result;
        }
        if let Some(result) = mask_trash_prompt::intercept(self, id, &params) {
            return result;
        }
        // Automation input also gates every step a command runs on its behalf (`actions.play`).
        let gate = if self.automation_input && self.session.authorize.is_none() { self.services.automation_authorize } else { None };
        if gate.is_some() {
            self.session.authorize = gate;
        }
        let result = self.run_command(id, params);
        if gate.is_some() {
            self.session.authorize = None;
        }
        // An explicit channel target (⌘2, ⌘\, ⌘3…) decides what the Layers panel targets too:
        // ⌘2 after the mask was targeted goes back to the layer's pixels.
        if result.is_ok() && id.starts_with("channel.target") {
            if let Some(t) = self.session.active().map(|st| st.channel_view.target) {
                use photocraft_engine::channel_cmds::ChannelTarget;
                self.ui.mask_target = t == ChannelTarget::LayerMask;
                self.ui.vector_mask_target = t == ChannelTarget::VectorMask;
            }
            self.sync_mask_targets();
        }
        result
    }

    fn run_command(&mut self, id: &str, params: Value) -> Result<Value, String> {
        if matches!(id, "paint.setSymmetry" | "paint.symmetryFromPath" | "paint.symmetryDisable") {
            self.ui.symmetry_transform = None;
        }
        let clip_read = std::mem::take(&mut self.clip_read_for_paste);
        if self.automation_input
            && let Some(authorize) = self.services.automation_command.as_ref()
        {
            authorize(id, &params)?;
        }
        if let Some(result) = panel_docking::command(self, id, &params) {
            return result;
        }
        if let Some(r) = transform_tool::intercept(self, id) {
            return r;
        }
        if let Some(r) = crate::crop_ui::intercept(self, id) {
            return r;
        }
        let suppress_events = self.automation_input && self.session.prefs().script_events.enabled;
        if suppress_events {
            self.session.edit_prefs(|prefs| prefs.script_events.enabled = false);
        }
        let t0 = gpu_canvas::now_ms();
        // The OS clipboard is read only on an explicit paste, never in the background (privacy, CPU).
        if matches!(id, "edit.paste" | "edit.pasteSpecial.pasteInPlace" | "file.newFromClipboard") {
            if !clip_read {
                self.import_os_clipboard();
            }
            if self.session.clipboard.is_none() && self.services.clipboard_get_image.is_some() {
                // Enabled on the strength of the OS clipboard, which held no image: a quiet no-op.
                let message = "Nothing to paste: the clipboard holds no image";
                self.ui.status = tl!(message).into();
                self.ui.status_error = false;
                return Ok(serde_json::json!({"pasted": false}));
            }
        }
        // Long commands become background jobs when enabled (`jobs_ui`); the rest run inline.
        let params = self.with_mask_target(id, params);
        let params = vector_ui::with_active_path(self, id, params);
        let path_mask = vector_ui::takes_path_mask(id) && params.get("path").is_some_and(|v| !v.is_null());
        let on_active_layer = self
            .session
            .active()
            .is_some_and(|st| st.active_layer.is_some_and(|layer| params.get("layer").and_then(Value::as_u64).is_none_or(|target| target == layer.0)));
        let creates_active_mask =
            matches!(id, "layer.layerMask.revealAll" | "layer.layerMask.hideAll" | "layer.layerMask.revealSelection" | "layer.layerMask.hideSelection")
                && on_active_layer;
        let creates_active_vector_mask = matches!(
            id,
            "layer.vectorMask.add" | "layer.vectorMask.fromPath" | "layer.vectorMask.revealAll" | "layer.vectorMask.hideAll" | "layer.vectorMask.currentPath"
        ) && on_active_layer;
        let r = if id == "actions.play" {
            actions::play(self, &params)
        } else if photocraft_engine::actions_cmds::shell_view_command(id) {
            actions::run_view(self, id, params)
        } else {
            jobs_ui::run(self, id, params)
        };
        let creates_adjustment_or_fill = id.starts_with("layer.newAdjustmentLayer.") || id.starts_with("layer.newFillLayer.");
        if r.is_ok() && (ADDS_LAYER_MASK.contains(&id) || creates_adjustment_or_fill) {
            // Adding a layer mask targets it, as in Photoshop (#2166).
            self.ui.mask_target = true;
            self.ui.vector_mask_target = false;
            // Properties still shows the new adjustment or fill layer's settings, not its mask,
            // as in Photoshop (#3369): clicking the mask thumbnail switches it to the mask.
            self.ui.properties_layer = if creates_adjustment_or_fill { self.session.active().and_then(|st| st.active_layer).map(|l| l.0) } else { None };
        }
        if r.is_ok() && creates_adjustment_or_fill {
            // Properties comes to the front, shown and expanded, so the new layer's settings can
            // be edited at once, as in Photoshop (#4028). The layer is made whatever happens here.
            if self.ui.docking.is_some() {
                let _ = panel_docking::command(self, "window.panel.activate", &json!({"panel": "properties"}));
            } else {
                self.ui.dock_tabs.properties = 0;
                dock::reveal(self, dock::Group::Properties);
            }
        }
        if r.is_ok() && id == "select.toWorkPath" {
            // Make Work Path selects the new work path in the Paths panel, as in Photoshop.
            self.ui.selected_path = Some("work".into());
        }
        if r.is_ok() && path_mask {
            // The new layer's vector mask becomes the active path, as in Photoshop: the path it
            // was made from is no longer selected, so the next fill layer isn't masked by it too.
            self.ui.selected_path = Some("layer".into());
        }
        if r.is_ok() && creates_active_mask {
            // Adding a mask selects its thumbnail: the next brush or footer Delete targets it.
            self.ui.mask_target = true;
            self.ui.vector_mask_target = false;
        }
        if r.is_ok() && creates_active_vector_mask {
            // Likewise a new vector mask (#4728): Properties shows its Density and Feather, and a
            // mask made from a path becomes the active path.
            self.ui.vector_mask_target = true;
            self.ui.mask_target = false;
            if matches!(id, "layer.vectorMask.currentPath" | "layer.vectorMask.fromPath") {
                self.ui.selected_path = Some("layer".into());
            }
        }
        if r.is_ok() && matches!(id, "edit.copy" | "edit.cut" | "edit.copyMerged") {
            self.clip_external = false;
            self.export_os_clipboard();
        }
        self.perf.last_command = id.to_string();
        self.perf.command_ms = gpu_canvas::now_ms() - t0;
        match &r {
            Ok(_) => {
                self.sync_views();
                let removed = self.session.take_samplers_removed();
                if removed > 0 {
                    // A canvas-size change (crop, Canvas Size, Image Size, rotation) dropped Color
                    // Sampler points that fell outside the new canvas (#1046).
                    self.ui.status = crate::i18n::trn(crate::i18n::current(), removed as u64, "{n} sample point removed", "{n} sample points removed");
                    self.ui.status_error = false;
                }
                if self.ui.status_error {
                    self.ui.status.clear();
                    self.ui.status_error = false;
                }
            }
            Err(e) => {
                self.ui.status = e.clone();
                self.ui.status_error = true;
            }
        }
        if suppress_events {
            self.session.edit_prefs(|prefs| prefs.script_events.enabled = true);
        }
        r
    }

    /// Keep one view per document, in tab order: a view and its windows stay with their document
    /// when tabs move (`document.move`) or close.
    /// The tool pointer events go to: a held temporary tool when there is one (⌘ is the Move
    /// tool, `hold_keys::cmd_moves`), else the selected tool.
    pub fn active_tool(&self) -> state::Tool {
        self.tool_override.unwrap_or(self.ui.tool)
    }

    pub fn sync_views(&mut self) {
        type_transform::cancel_stale(self);
        crate::cam_transform::cancel_stale(self);
        crate::lasso_ui::cancel_stale(self);
        // A polygon under construction belongs to the Polygonal Lasso tool, not the next tool.
        if self.ui.tool != Tool::PolygonLasso {
            self.ui.polygon.clear();
            self.ui.polygon_mode.clear();
        }
        // ... and to the document it was started on.
        crate::canvas::polygon_cancel_stale(self);
        crate::crop_ui::cancel_stale(self);
        let ids: Vec<DocId> = self.session.documents().iter().map(|d| d.doc.id).collect();
        // Where the document of view `i` is now. Views not tracked yet keep their index.
        let now = |i: usize| match self.view_docs.get(i) {
            Some(id) => ids.iter().position(|d| d == id),
            None => (i < ids.len()).then_some(i),
        };
        let mut views: Vec<Option<state::View>> = ids.iter().map(|_| None).collect();
        for (i, v) in std::mem::take(&mut self.ui.views).into_iter().enumerate() {
            if let Some(slot) = now(i).and_then(|to| views.get_mut(to)) {
                *slot = Some(v);
            }
        }
        self.ui.views = views.into_iter().map(Option::unwrap_or_default).collect();
        self.ui.windows.retain_mut(|w| now(w.document).map(|d| w.document = d).is_some());
        if !self.session.prefs().workspace.open_documents_as_tabs {
            let new_docs: Vec<usize> = ids.iter().enumerate().filter(|(_, id)| !self.view_docs.contains(id)).map(|(i, _)| i).collect();
            for &idx in &new_docs {
                if !self.ui.windows.iter().any(|w| w.document == idx) {
                    crate::view_cmds::float_window(self, idx, idx);
                }
            }
        }
        self.canvases.retain(|(doc, _), _| ids.contains(doc));
        self.navigator_textures.retain(|doc, _| ids.contains(doc));
        self.view_docs = ids;
        self.prune_thumbs();
        self.sync_mask_targets();
        // Channel-view textures outlive a hidden view (cheap re-show), not their document.
        if !self.channel_views.is_empty() {
            let docs = self.session.documents();
            self.channel_views.retain(|id, _| docs.iter().any(|d| d.doc.id.0 == *id));
        }
        // Commands may close and reopen a preserved-ID document before the next repaint.
        canvas::retain_gpu_documents(self);
    }

    /// Record `path` as the most-recently-opened file (File › Open Recent): de-duplicated, newest
    /// first, capped at Preferences › File Handling › Recent File List Contains. The list lives in
    /// the preferences (`fileHandling.recentFiles`), so it survives a restart; `ui.recent_files`
    /// mirrors it for the menus and the Home screen.
    pub fn push_recent(&mut self, path: &str) {
        let cap = self.recent_cap();
        let mut r = self.session.prefs().file_handling.recent_files.clone();
        r.retain(|p| p != path);
        r.insert(0, path.to_string());
        r.truncate(cap);
        self.ui.recent_files = r.clone();
        self.session.prefs.edit(|p| p.file_handling.recent_files = r);
    }

    /// File › Open Recent › Clear Recent File List.
    pub fn clear_recent(&mut self) {
        self.ui.recent_files.clear();
        self.session.prefs.edit(|p| p.file_handling.recent_files.clear());
    }

    /// Take one file off the recent list (the Home screen's × or Remove from Recent, #2691). The
    /// file stays on disk, and opening it again lists it again.
    pub fn remove_recent(&mut self, path: &str) {
        self.ui.recent_files.retain(|p| p != path);
        self.session.prefs.edit(|p| p.file_handling.recent_files.retain(|r| r != path));
    }

    /// How many recent files to remember ("Recent File List Contains", 0–100).
    pub fn recent_cap(&self) -> usize {
        self.session.prefs().file_handling.recent_file_count.min(100) as usize
    }

    /// Mirror the stored recent-file list into the UI state (after loading the preferences, or
    /// when the Preferences dialog or an agent changes the list or its length). Cheap enough to
    /// run every frame: a slice comparison of at most 100 paths, no allocation unless it changed.
    pub fn sync_recent(&mut self) {
        let cap = self.recent_cap();
        let stored = &self.session.prefs().file_handling.recent_files;
        let want = stored.get(..cap.min(stored.len())).unwrap_or_default();
        if self.ui.recent_files.as_slice() != want {
            self.ui.recent_files = want.to_vec();
        }
    }

    /// Open a file's bytes as a new document named `name`; returns the import warnings (also
    /// shown to the user). Files from disk go through [`open_file`](Self::open_file), which also
    /// remembers the path. Brush and gradient files go to the preset libraries instead.
    pub fn open_bytes(&mut self, name: &str, bytes: &[u8]) -> Result<Vec<String>, String> {
        self.open_bytes_from(name, bytes, None)
    }

    /// [`Self::open_bytes`] for a file read from `path` (File › Save writes back there; it goes
    /// to Open Recent), when known.
    pub(crate) fn open_bytes_from(&mut self, name: &str, bytes: &[u8], path: Option<&str>) -> Result<Vec<String>, String> {
        if let Some(r) = preset_files_ui::open(self, name, bytes) {
            return r.map(|()| Vec::new());
        }
        if self.services.host_save.is_some() {
            return self.place_bytes(name, bytes.to_vec(), None).map(|_| Vec::new());
        }
        let name = &self.open_name(name);
        // Decoded on a worker: a tab with progress appears now, the document when it's ready
        // (warnings are shown then).
        if self.background_jobs {
            jobs_ui::start_open(self, name, path.map(str::to_string), jobs_ui::bytes(bytes))?;
            return Ok(Vec::new());
        }
        let import = self.services.import.as_ref().ok_or("no importer configured")?;
        let max_svg_group_depth = self.session.prefs().file_handling.rasterize_svg_groups_deeper_than as usize;
        let (doc, warnings) = import(name, bytes, max_svg_group_depth)?;
        // Edit › Color Settings policies apply on open; Ask When Opening holds the file back
        // until its prompt is answered, and later files wait behind it (`open_prompt`).
        // Missing fonts are read from the document itself: it may not be open yet (#4559).
        let warnings = file_open::with_missing_fonts_of(&doc, warnings);
        // A developed raw has no embedded profile to ask about.
        let developed = warnings.iter().any(|w| w.contains(photocraft_io::raw::DEVELOPED_NOTE));
        let question = self.session.color.open_question(&doc).filter(|_| !developed);
        let wants_raw = camera_raw_ui::wants_open_dialog(self, &warnings);
        open_prompt::arrive(
            self,
            open_prompt::Held {
                doc,
                read_only: photocraft_io::affinity::is_affinity(bytes),
                name: name.to_string(),
                path: path.map(str::to_string),
                warnings: warnings.clone(),
                bytes: (wants_raw && path.is_none()).then(|| std::sync::Arc::new(bytes.to_vec())),
                slot: None,
                question,
                policy_applied: false,
            },
        );
        Ok(warnings)
    }

    /// Open bytes supplied by an authenticated automation client without
    /// invoking user-configured script-event paths outside the granted root.
    /// Returns the import warnings (also shown to the user). Brush and gradient
    /// files go to the preset libraries, as for interactive opens.
    pub fn open_automation_bytes(&mut self, name: &str, bytes: &[u8]) -> Result<Vec<String>, String> {
        let events_enabled = self.session.prefs().script_events.enabled;
        if events_enabled {
            self.session.edit_prefs(|prefs| prefs.script_events.enabled = false);
        }
        let result = match preset_files_ui::open(self, name, bytes) {
            Some(r) => r.map(|()| Vec::new()),
            None => self.import_automation_document(name, bytes),
        };
        if events_enabled {
            self.session.edit_prefs(|prefs| prefs.script_events.enabled = true);
        }
        result
    }

    /// The document half of [`open_automation_bytes`](Self::open_automation_bytes): no script
    /// events and no Color Settings policy (which may read user-configured profile paths).
    fn import_automation_document(&mut self, name: &str, bytes: &[u8]) -> Result<Vec<String>, String> {
        let import = self.services.import.as_ref().ok_or("no importer configured")?;
        let max_svg_group_depth = self.session.prefs().file_handling.rasterize_svg_groups_deeper_than as usize;
        let (doc, warnings) = import(name, bytes, max_svg_group_depth)?;
        // The caller records the path it read from.
        self.session.add_document(doc, None);
        if let Some(st) = self.session.active_mut() {
            st.source_read_only = photocraft_io::affinity::is_affinity(bytes);
        }
        self.sync_views();
        self.ui.status = crate::i18n::fmt(tl!("Opened {name}"), &[("name", name)]);
        self.ui.status_error = false;
        notices::io_warnings(self, &crate::i18n::fmt(tl!("Opened {name}"), &[("name", name)]), &warnings);
        Ok(warnings)
    }

    /// Save the active document to `path`, or to a path chosen in the save dialog (which
    /// continues on a later frame, see `file_dialog`). Returns `{"path", "warnings"}` (the export
    /// warnings are also shown to the user).
    pub fn save_as(&mut self, path: Option<String>) -> Result<Value, String> {
        self.save_as_with(path, None)
    }

    /// [`Self::save_as`], writing a Targa with `tga_bits` per pixel when given (no Targa Options
    /// prompt): `file.save` and `file.saveAs` with `tgaBits`.
    pub fn save_as_with(&mut self, path: Option<String>, tga_bits: Option<photocraft_io::TgaBits>) -> Result<Value, String> {
        if self.save_options_open() {
            return Err("Answer TIFF or Targa Options before starting another save".into());
        }
        // Edit Contents documents save back into their smart object.
        if path.is_none() && self.session.is_enabled("layer.smartObjects.saveContents") {
            self.run("layer.smartObjects.saveContents", serde_json::json!({}))?;
            return Ok(serde_json::json!({"path": "smart object", "warnings": []}));
        }
        if let Some(path) = path {
            return self.save_to(path, tga_bits);
        }
        let st = self.session.active().ok_or("no document")?;
        // PDN imports default to our native format, which preserves Paint.NET's blend modes;
        // Krita imports to OpenRaster, which Krita opens with its layers.
        // Other files keep their format only when it can retain their layers. A plain raster
        // (including an unlocked transparent PNG) can keep its flat format; editable contents,
        // masks and layer appearance settings need a layered format even with just one layer.
        let flat = plain_raster(&st.doc).is_some();
        let ext = st.path.as_deref().and_then(|p| std::path::Path::new(p).extension()).map(|e| e.to_string_lossy().to_ascii_lowercase());
        // An OpenRaster file only when it holds the whole document (#3806): else PSD is offered.
        let holds_ora = st.path.as_deref().is_some_and(|p| photocraft_engine::file_cmds::holds_at(&st.doc, p));
        let writable = ext.is_some_and(|e| {
            matches!(e.as_str(), photocraft_format::EXTENSION | "psd" | "psb" | "tif" | "tiff")
                || (e == "ora" && holds_ora)
                || (flat && photocraft_codecs::from_extension(&e).is_some_and(|f| f.caps().write))
        });
        let suggested = match &st.path {
            Some(p) if writable => p.clone(),
            p => {
                let source = p.as_deref().unwrap_or(&st.doc.name);
                let ext = match photocraft_engine::file_cmds::extension(source).as_deref() {
                    Some("pdn") => "pcraft",
                    Some("kra" | "krz") => "ora",
                    _ => "psd",
                };
                std::path::Path::new(source).with_extension(ext).to_string_lossy().into_owned()
            }
        };
        let doc = st.doc.id;
        self.pick_save(&suggested, move |app, path| app.with_document(doc, |app| app.save_to(path, tga_bits)))
    }

    /// [`Self::save_as_with`] once the path is known.
    fn save_to(&mut self, path: String, tga_bits: Option<photocraft_io::TgaBits>) -> Result<Value, String> {
        // An RGB Targa asks for its bits per pixel, unless the caller chose them; the save
        // continues from the prompt (still a copy when it would be one without the prompt),
        // which records the step once answered.
        if tga_bits.is_none() && targa_options_ui::wants_prompt(self, &path) {
            let copy = self.saves_copy(&path, None)?;
            targa_options_ui::park(self, path.clone(), copy, "file.saveAs")?;
            return Ok(serde_json::json!({"path": path, "warnings": []}));
        }
        // A layered TIFF asks about its layers first (Preferences › File Handling); the save
        // continues from the prompt, which records the step once answered.
        if tiff_options_ui::wants_prompt(self, &path) {
            tiff_options_ui::park(self, path.clone())?;
            return Ok(serde_json::json!({"path": path, "warnings": []}));
        }
        self.save_as_step(path, None, tga_bits)
    }

    /// File › Save back to the document's own layered file, recorded as a `file.save` step.
    pub(crate) fn save_in_place(&mut self, path: String) -> Result<Value, String> {
        if self.save_options_open() {
            return Err("Answer TIFF or Targa Options before starting another save".into());
        }
        let r = self.write_save(path, None, None)?;
        self.session.journal.push(("file.save".into(), Value::Object(Default::default())));
        Ok(r)
    }

    /// Writes the Save As and journals it as a `file.saveAs` step with its path (and TIFF or
    /// Targa answer), so an action being recorded keeps it (#2032).
    fn save_as_step(&mut self, path: String, tiff_layers: Option<bool>, tga_bits: Option<photocraft_io::TgaBits>) -> Result<Value, String> {
        let r = self.write_save(path.clone(), tiff_layers, tga_bits)?;
        let mut step = serde_json::json!({"path": path});
        if let Some(layers) = tiff_layers {
            step["tiffLayers"] = layers.into();
        }
        if let Some(bits) = tga_bits {
            step["tgaBits"] = bits.bits().into();
        }
        self.session.journal.push(("file.saveAs".into(), step));
        Ok(r)
    }

    /// Replays a recorded `file.save` / `file.saveAs` / `file.saveACopy` step: no dialog or prompt, and the
    /// recorded TIFF or Targa answer (if any) stands in for the TIFF or Targa Options prompt.
    pub(crate) fn replay_save(&mut self, id: &str, params: &Value) -> Result<Value, String> {
        if self.save_options_open() {
            return Err("Answer TIFF or Targa Options before starting another save".into());
        }
        if id == "file.save" {
            let st = self.session.active().ok_or("no document")?;
            let path =
                st.path.clone().filter(|p| photocraft_engine::file_cmds::holds_at(&st.doc, p)).ok_or(
                    "Save writes back only to the document's own PSD, PSB, .pcraft or OpenRaster file, and to OpenRaster only when it holds everything",
                )?;
            return self.save_in_place(path);
        }
        let path = params.get("path").and_then(Value::as_str).filter(|p| !p.is_empty()).ok_or("Save As needs a `path`")?;
        let tga_bits = targa_options_ui::bits_param(params, id)?;
        if id == "file.saveACopy" {
            // A copy: the document keeps its own file and unsaved state, as when recorded (#4373).
            let doc = self.session.active().ok_or("no document")?.doc.clone();
            return crate::view_cmds::write_copy(self, &doc, path, tga_bits);
        }
        self.save_as_step(path.to_string(), params.get("tiffLayers").and_then(Value::as_bool), tga_bits)
    }

    /// What a save writes by default under the preferences: File Handling › Image Previews
    /// decides whether a PSD/PSB gets its thumbnail, and Maximize PSD and PSB File Compatibility
    /// whether it gets its real merged image. Ask When Saving has no dialog here yet, so it saves
    /// both, as Photoshop's checkboxes start on. A JPEG or WebP is written the way the active
    /// document's file was (its JPEG quality, a WebP lossy at its quality or lossless), as
    /// Photoshop's Save and Save As options start there.
    pub(crate) fn save_settings(&self) -> ExportSettings {
        let file_handling = &self.session.prefs().file_handling;
        let previews = file_handling.image_previews != photocraft_engine::prefs::Ask::Never;
        let maximize = file_handling.maximize_psd_compatibility != photocraft_engine::prefs::Ask::Never;
        let mut settings = ExportSettings { image_previews: previews, psd_maximize_compatibility: maximize, ..ExportSettings::default() };
        if let Some(st) = self.session.active() {
            let e = &st.doc.file_encoding;
            settings.jpeg_quality = e.jpeg_quality.map(|q| q.clamp(1, 100));
            settings.webp_lossless = e.webp_lossless.unwrap_or(settings.webp_lossless);
            settings.webp_quality = e.webp_quality.map(|q| q.min(100));
        }
        settings
    }

    /// Whether saving the active document to `path` writes a copy. `tiff_layers` is the TIFF
    /// Options answer: `Some(false)` discards the layers and saves a copy.
    fn saves_copy(&self, path: &str, tiff_layers: Option<bool>) -> Result<bool, String> {
        // A flat file that can't hold the document (its layers, or the layered file it lives in)
        // is written as a copy, as in Photoshop: the document keeps its file, so Save still
        // writes the layered original and the edits stay unsaved (#2550).
        // An OpenRaster file that can't hold everything (adjustment layers, styles…) is a copy too,
        // so the editable document isn't marked saved after losing them (#3806).
        let st = self.session.active().ok_or("no document")?;
        let tiff = |p: &str| matches!(photocraft_engine::file_cmds::extension(p).as_deref(), Some("tif" | "tiff"));
        let holds = photocraft_engine::file_cmds::holds_at(&st.doc, path) || tiff(path);
        let lives_layered = st.path.as_deref().is_some_and(|p| photocraft_engine::file_cmds::saves_in_place(p) || tiff(p));
        Ok(tiff_layers == Some(false) || (!holds && (plain_raster(&st.doc).is_none() || lives_layered)))
    }

    /// Writes the active document to a known path. `tiff_layers` is the TIFF Options answer:
    /// `Some(false)` discards the layers and saves a copy. `tga_bits` is the Targa bits per
    /// pixel (`None`: 32 with an alpha channel or transparency, else 24).
    fn write_save(&mut self, path: String, tiff_layers: Option<bool>, tga_bits: Option<photocraft_io::TgaBits>) -> Result<Value, String> {
        let copy = self.saves_copy(&path, tiff_layers)?;
        let settings = ExportSettings { tiff_layers: tiff_layers.unwrap_or(ExportSettings::default().tiff_layers), tga_bits, ..self.save_settings() };
        match self.write_document(path.clone(), &settings, copy)? {
            Some((path, warnings)) => Ok(serde_json::json!({"path": path, "warnings": warnings})),
            None => Ok(serde_json::json!({"path": path, "warnings": [], "pending": true, "job": self.jobs.last_started.map(|j| j.0)})),
        }
    }

    /// Encodes the active document with `settings` and writes it to `path`, which becomes the
    /// document's path unless saving a copy. A copy leaves the original's path and unsaved
    /// changes intact. Returns the path and the export warnings (also shown to the user), or
    /// `None` when the save went to a background job (#2017), which finishes it in
    /// [`jobs_ui::tick`].
    pub(crate) fn write_document(&mut self, path: String, settings: &ExportSettings, copy: bool) -> Result<Option<(String, Vec<String>)>, String> {
        if self.background_jobs
            && let Some(save) = self.services.save_file.clone()
        {
            return jobs_ui::start_save(self, path, settings.clone(), copy, save, false);
        }
        let st = self.session.active().ok_or("no document")?;
        let (doc, revision) = (st.doc.id, st.revision);
        let export = self.services.export.as_ref().ok_or("no exporter configured")?;
        let (bytes, warnings) = export(&st.doc, &path, settings)?;
        let write = self.services.write.as_mut().ok_or("no writer configured")?;
        write(&path, &bytes)?;
        self.saved(doc, revision, &path, &warnings, copy);
        Ok(Some((path, warnings)))
    }

    /// Record a written save of document `doc` as it was at `revision`: its name, path and saved
    /// state (unless a copy), the status, script events and the warnings.
    pub(crate) fn saved(&mut self, doc: DocId, revision: u64, path: &str, warnings: &[String], copy: bool) {
        self.saved_inner(doc, revision, path, warnings, copy, SaveKind::User);
    }

    /// [`Self::saved`] for automation saves: no user-configured hooks (docs/control-protocol.md).
    pub(crate) fn saved_automation(&mut self, doc: DocId, revision: u64, path: &str, warnings: &[String], copy: bool) {
        self.saved_inner(doc, revision, path, warnings, copy, SaveKind::Automation);
    }

    fn saved_inner(&mut self, doc: DocId, revision: u64, path: &str, warnings: &[String], copy: bool, kind: SaveKind) {
        // Automation: only a layered save becomes the document's file; a flat one is a
        // copy, as in the headless server (#2579).
        let (hooks, arm) = match kind {
            SaveKind::User => (true, true),
            SaveKind::Automation => {
                let holds = self.session.documents().iter().find(|st| st.doc.id == doc).is_some_and(|st| photocraft_engine::file_cmds::holds_at(&st.doc, path));
                (false, holds)
            }
        };
        self.ui.status = crate::i18n::fmt(tl!("Saved {name}"), &[("name", path)]);
        if !copy {
            // A background save may finish while another document is active (or after its
            // document was closed, when there is nothing left to record).
            let _ = self.with_document(doc, |app| {
                if let Some(st) = app.session.active_mut()
                    && arm
                {
                    st.saved_to(path.to_string());
                    // The job locked the document, but record the revision that was written.
                    st.saved_revision = revision;
                }
                // "Save Document" script events and File › Generate › Image Assets.
                if hooks
                    && let Some(i) = app.session.active_index()
                    && let Some(r) = photocraft_engine::automate_cmds::document_saved(&mut app.session, i)
                {
                    let template = "Saved {path}; image assets: {n} in {dir}";
                    app.ui.status = crate::i18n::fmt(
                        tl!(template),
                        &[("path", path), ("n", &r["files"].as_array().map_or(0, Vec::len).to_string()), ("dir", r["dir"].as_str().unwrap_or(""))],
                    );
                }
                Ok(())
            });
            // The file the document now lives in goes to the top of Open Recent and the Home
            // screen, as opening it does (#4400). Automation saves don't touch the user's list.
            if hooks {
                self.push_recent(path);
            }
        }
        self.ui.status_error = false;
        let what = crate::i18n::fmt(tl!("Saved {name}"), &[("name", file_open::display_name(path).as_str())]);
        notices::io_warnings(self, &what, warnings);
        self.sync_views();
    }

    /// TIFF Options or Targa Options is waiting on an answer before a save is written.
    pub(crate) fn save_options_open(&self) -> bool {
        self.tiff_options.is_some() || self.targa_options.is_some()
    }

    /// A print worker may still be handing a job to the system queue.
    pub(crate) fn printing(&self) -> bool {
        self.session.jobs().iter().any(|j| matches!(j.command.as_str(), "file.print" | "file.printOneCopy"))
    }

    /// A save is running in the background.
    pub(crate) fn saving(&self) -> bool {
        !self.jobs.saves.is_empty()
    }

    /// Save through the control session's capability-scoped writer. No file
    /// picker or ambient writer is reachable from this path. Returns the path
    /// written and the export warnings (also shown to the user).
    pub fn save_automation(&mut self, path: Option<String>) -> Result<Value, String> {
        self.save_automation_with(path, None)
    }

    /// [`Self::save_automation`] writing a Targa with `tga_bits` per pixel (`None`: 32 with an
    /// alpha channel or transparency, else 24). Automation saves never prompt.
    pub fn save_automation_with(&mut self, path: Option<String>, tga_bits: Option<photocraft_io::TgaBits>) -> Result<Value, String> {
        let target = {
            let state = self.session.active().ok_or("no document")?;
            // As File › Save: without `path` only a layered file is written back (#416).
            path.or_else(|| state.path.clone().filter(|p| photocraft_engine::file_cmds::holds_at(&state.doc, p))).ok_or(
                "pass `path`: a save without one writes back only to the document's own PSD, PSB, .pcraft or OpenRaster file, and to OpenRaster only when it holds everything (adjustment layers, styles and masks don't fit)",
            )?
        };
        let reply = |path: &str, warnings: &[String]| json!({"path": path, "warnings": warnings});
        // #2017: the File › Save worker path, so the encode doesn't block the frame loop.
        if self.background_jobs
            && let Some(save) = self.services.automation_save.clone()
        {
            return match jobs_ui::start_save(self, target.clone(), ExportSettings { tga_bits, ..self.save_settings() }, false, save, true)? {
                Some((path, warnings)) => Ok(reply(&path, &warnings)),
                None => {
                    let job = self.jobs.last_started;
                    Ok(json!({"path": target, "job": job, "pending": true, "warnings": []}))
                }
            };
        }
        let state = self.session.active().ok_or("no document")?;
        let (doc, revision) = (state.doc.id, state.revision);
        let settings = ExportSettings { tga_bits, ..self.save_settings() };
        let export = self.services.export.as_ref().ok_or("no exporter configured")?;
        let (bytes, warnings) = export(&state.doc, &target, &settings)?;
        let write = self.services.automation_write.as_mut().ok_or("automation write authority is not configured")?;
        write(&target, &bytes)?;
        self.saved_automation(doc, revision, &target, &warnings, false);
        Ok(reply(&target, &warnings))
    }

    fn drain_control(&mut self, ctx: &egui::Context) {
        let Some(rx) = self.control_rx.take() else { return };
        while let Ok(req) = rx.try_recv() {
            let reply = req.reply.clone();
            if req.deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline) || !req.dispatch.start() {
                let _ = reply.send(serde_json::json!({"ok": false, "error": control::expired_error()}));
                continue;
            }
            match control::handle(self, ctx, &req) {
                control::Outcome::Done(v) => {
                    let _ = reply.send(v);
                }
                control::Outcome::AfterInput => self.input_waiters.push(reply),
                control::Outcome::AfterJob(job) => self.jobs.waiters.push((job, reply)),
                control::Outcome::Screenshot { token, path } => {
                    // Wait out egui's fade animations (~83 ms) and a few rendered frames first.
                    let settle = ctx.global_style().animation_time as f64 * 2000.0 + 60.0;
                    self.queued_screenshots.push((token, gpu_canvas::now_ms() + settle, 0));
                    self.pending_screenshots.push((token, path, reply));
                }
            }
            // `ui.set` may have changed the edit target: the next request sees its colours.
            self.sync_mask_targets();
        }
        self.control_rx = Some(rx);
    }

    fn issue_screenshots(&mut self, ctx: &egui::Context) {
        let now = gpu_canvas::now_ms();
        self.queued_screenshots.retain_mut(|(token, at, frames)| {
            *frames += 1;
            if now >= *at && *frames >= 3 {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::new(*token)));
                false
            } else {
                true
            }
        });
    }

    fn collect_screenshots(&mut self, ctx: &egui::Context) {
        if self.pending_screenshots.is_empty() {
            return;
        }
        let events: Vec<_> = ctx.input(|i| {
            i.raw
                .events
                .iter()
                .filter_map(|e| match e {
                    egui::Event::Screenshot { user_data, image, .. } => {
                        let token = user_data.data.as_ref().and_then(|d| d.downcast_ref::<u64>()).copied()?;
                        Some((token, image.clone()))
                    }
                    _ => None,
                })
                .collect()
        });
        for (token, image) in events {
            if let Some(i) = self.pending_screenshots.iter().position(|(t, _, _)| *t == token) {
                let (_, path, reply) = self.pending_screenshots.remove(i);
                let r = control::save_screenshot(self, &image, path.as_deref());
                let _ = reply.send(r);
            }
        }
    }
}

impl eframe::App for PhotocraftApp {
    fn logic(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        i18n::set_current(i18n::Lang::from_pref(&self.session.prefs().interface.language));
        // Caps Lock state read from the OS once per frame (see `Services::caps_lock`): the canvas
        // cursor block needs it before the frame renders. `None` (Wayland/web) keeps `false`.
        self.caps_lock = self.services.caps_lock.as_mut().is_some_and(|f| f());
        if !self.styled {
            Self::setup_context(ctx, self.ui.theme);
            self.styled = true;
        } else {
            self.fonts_ready = true;
        }
        self.frame += 1;
        let now = ctx.input(|i| i.time);
        let dt = (now - self.last_frame_time) as f32;
        if dt > 0.0 {
            self.fps = self.fps * 0.9 + (1.0 / dt).min(240.0) * 0.1;
        }
        self.last_frame_time = now;
        self.sync_views();
        crate::type_font_preview::follow_hover(self, ctx);
        self.check_gpu(ctx);
        #[cfg(all(debug_assertions, not(target_arch = "wasm32")))]
        if self.live_tokens.poll(ctx, self.ui.theme) {
            self.checker = None;
        }
        self.drain_control(ctx);
        native_menu::run(self, ctx);
        // Switching tools commits, also between the Type and Type Mask tools.
        if self.ui.text_edit.as_ref().is_some_and(|ed| !self.ui.tool.is_type() || ed.mask != self.ui.tool.is_type_mask()) {
            type_tool::commit(self);
        }
        if self.ui.pen.as_ref().is_some_and(|p| !p.drawn_with(self.ui.tool)) {
            vector_ui::pen_commit(self, false);
        }
        transform_tool::end_if_left(self);
        shape_stroke_ui::sync_editor(self, ctx);
        self.collect_screenshots(ctx);
        self.issue_screenshots(ctx);
        prefs_ui::tick(self, ctx);
        monitor_status::poll_frame(self, ctx, frame);
        // Control requests and persisted preferences can change the language in this frame.
        i18n::sync_context(ctx, &self.session.prefs().interface.language);
        type_tool::sync_font_names(ctx, self.session.prefs().type_.show_font_names_in_english);
        // A window bigger than its display (1440 × 900 on 1366 × 768) runs under the taskbar:
        // maximize it into the work area once (#315).
        work_area::fit_window(ctx);
        discard_ui::guard_window_close(self, ctx);
        // Background jobs: apply finished ones, keep frames coming, Esc cancels (before the
        // shortcuts see Esc).
        if !screen_picker::tick(self, ctx) {
            jobs_ui::tick(self, ctx);
            #[cfg(not(target_arch = "wasm32"))]
            filter_preview_worker::discard_closed(self);
            shortcuts::handle(self, ctx);
        }
        let arrived: Vec<(String, Vec<u8>)> =
            self.services.inbox.as_ref().map(|q| std::mem::take(&mut *q.lock().unwrap_or_else(|e| e.into_inner()))).unwrap_or_default();
        for (name, bytes) in arrived {
            if let Err(e) = self.open_bytes(&name, &bytes) {
                self.open_failed(&name, &e);
            }
        }
        // Finder double-click / Open With / Dock drops (macOS open-documents events).
        self.drain_os_events(ctx);
        // While files are dragged over the window it gets no pointer events: keep frames coming so
        // the tab strip can follow the pointer.
        if ctx.input(|i| !i.raw.hovered_files.is_empty()) {
            ctx.request_repaint();
        }
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        if !dropped.is_empty() {
            let at = self.services.cursor_pos.as_mut().and_then(|f| f(ctx));
            self.open_dropped(ctx, dropped, at);
        }
        self.place_next_dropped(ctx);
        // The control transport wakes the UI on arrival (ctx.request_repaint); only poll while a
        // screenshot is pending. (Polling every 50 ms here made idle apps render at 20 fps.)
        if !self.pending_screenshots.is_empty() {
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        }
        self.poll_file_dialog(ctx, Some(frame));
    }

    fn raw_input_hook(&mut self, ctx: &egui::Context, raw_input: &mut egui::RawInput) {
        let frame_start = gpu_canvas::now_ms();
        // An undecorated window gets its caption back when the toolkit changes its flags
        // (maximize, restore, …): take it off again before the frame is laid out (#2246).
        if let Some(window_frame) = self.services.window_frame.as_mut() {
            window_frame();
        }
        // Native menu key equivalents become the key presses they were (see `native_menu`).
        if let Some(menu) = self.services.native_menu.as_mut() {
            menu.raw_input(raw_input);
        }
        shortcuts::clipboard_keys(ctx, ctx.text_edit_focused() || self.ui.text_edit.is_some(), raw_input);
        // Windows precision touchpads and some Linux Wayland compositors send pinches
        // as fractional Ctrl + wheel rather than a native Zoom event (#4566).
        // Leave Ctrl+whole-notch mouse scrolling alone, and honour the pinch preference.
        if cfg!(any(target_os = "windows", target_os = "linux")) && self.session.prefs().enhanced_controls.zoom_with_trackpad_pinch {
            wheel_nav::fold_legacy_pinch(ctx, raw_input);
        }
        raw_input.events.extend(self.take_synthetic_step());
        // A pen held still is no touch screen's press-and-hold right click (stylus.rs).
        let pen_down = self.stylus.pen_contacts.observe(&raw_input.events);
        stylus::pen_hold_is_no_right_click(ctx, pen_down);
        if self.custom_titlebar {
            titlebar::release_after_os_resize(ctx, raw_input);
        }
        if self.stroke_trace.enabled() {
            // The OS's moves and the control channel's (injected just above).
            let moves = raw_input.events.iter().filter(|e| matches!(e, egui::Event::PointerMoved(_))).count();
            self.stroke_trace.begin_frame(frame_start, moves);
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.hover_layer = None;
        i18n::set_current(i18n::Lang::from_pref(&self.session.prefs().interface.language));
        // Fonts registered via set_fonts only take effect next frame; named families would panic now.
        if !self.fonts_ready {
            ctx.request_repaint();
            self.automation_input = false;
            return;
        }
        let t0 = gpu_canvas::now_ms();
        self.perf.frame_refresh_ms = 0.0;
        self.stroke_trace.cpu_prev(frame.info().cpu_usage);
        // A text field's right-click menu, before any field is drawn (text_field_menu.rs).
        text_field_menu::show(&ctx);
        // Each painting tool keeps its own brush (#218), so switch the active tool's brush in
        // before anything this frame reads it (the cursor, the options bar, a stroke).
        paint_mouse::sync_tool_brush(self);
        // A unit typed into a value field (`1in` in a px field) converts at the active document's
        // resolution, as in Photoshop.
        let ppi = self.session.prefs().units_and_rulers.point_size.per_inch();
        let dpi = self.session.active().map_or(72.0, |d| f64::from(d.doc.resolution_dpi.max(1.0)));
        crate::widgets::set_field_resolution(&ctx, dpi, ppi);
        // View › Screen Mode › Full Screen Mode: only the image, on black (F or Esc returns).
        screen_picker::show(&ctx);
        if screen_picker::busy(&ctx) && screen_picker::showing(&ctx) {
            egui::Modal::new(egui::Id::new("screen-color-wait")).show(&ctx, |ui| {
                ui.spinner();
                ui.label(tl!("Pick a screen pixel or press Esc to cancel"));
                if ui.button(tl!("Cancel")).clicked() {
                    screen_picker::cancel(&ctx);
                }
            });
            self.automation_input = false;
            return;
        }
        if screen_picker::busy(&ctx) {
            ui.disable();
            ui.set_opacity(1.0);
        }
        // The OS title bar (and the taskbar / Alt-Tab entry) follows the active file; with the
        // system title bar this is where the document name lives, as the in-app title is hidden.
        panels::sync_window_title(self, &ctx);
        let chrome = !self.ui.view.hides_chrome();
        if chrome && self.ui.panels.menu_bar {
            panels::title_bar(self, ui);
        }
        if chrome && self.ui.panels.options_bar {
            panels::options_bar(self, ui);
        }
        if chrome && self.ui.panels.status_bar {
            panels::status_bar(self, ui);
        }
        if chrome && self.ui.panels.toolbar {
            panels::toolbar(self, ui);
        }
        if chrome && self.ui.panels.dock {
            panels::right_dock(self, ui);
        }
        let t = theme::Tokens::get(&ctx);
        let backdrop = if chrome { prefs_ui::pasteboard_color(self).unwrap_or(t.canvas) } else { egui::Color32::BLACK };
        egui::CentralPanel::default().frame(egui::Frame::NONE.fill(backdrop)).show(ui, |ui| {
            canvas::document_area(self, ui);
        });
        panels::properties_window(self, &ctx);
        brush_panel::window(self, &ctx);
        preset_panels::windows(self, &ctx);
        gradient_ui::editor_window(self, &ctx);
        type_panels_ui::windows(self, &ctx);
        analysis_ui::windows(self, &ctx);
        timeline_ui::windows(self, &ctx);
        workspace_ui::windows(self, &ctx);
        palette::show(self, &ctx);
        dialogs::show(self, &ctx);
        rasterize_prompt::run_pending(self, &ctx);
        jobs_ui::dialog(self, &ctx);
        discard_ui::show(self, &ctx);
        tiff_options_ui::show(self, &ctx);
        targa_options_ui::show(self, &ctx);
        distort_ui::show(self, &ctx);
        camera_raw_ui::show(self, &ctx);
        wide_angle_ui::show(self, &ctx);
        canvas::extra_windows(self, &ctx);
        notices::show(self, &ctx);
        gpu_status::show_fallback(self, &ctx);
        kys_import::show_offer(self, &ctx);
        if chrome && self.ui.panels.menu_bar && self.custom_titlebar && !self.ui.dialogs.is_empty() {
            titlebar::modal_drag_overlay(&ctx);
        }
        if self.custom_titlebar {
            titlebar::resize_zones(ui);
        }
        // A device lost while drawing this frame: switch to the CPU canvas before the next one.
        gpu_status::check(self, &ctx);
        // The Brush Preset picker's view (its gear's card parts and the footer scale) is
        // remembered once the pointer is up. Here, after every panel, rather than in the
        // picker's own code: the control channel can set it while the picker is closed.
        brush_picker::persist(self, &ctx);
        // The tool's brush (settings and tip references) is remembered the same way. Like the
        // picker's view, in the frame loop, so the control channel can set it with the panels
        // closed; the engine's revision counter keeps an unchanged brush free.
        last_brush::persist(self, &ctx);
        // Tool options are remembered between launches the same way (#2814).
        tool_memory::persist(self, &ctx);
        // So are View › Extras, Show and Snap To (#3762).
        view_cmds::persist(self, &ctx);
        self.automation_input = false;
        shape_stroke_ui::finish_frame(self, &ctx);
        native_menu::sync(self, &ctx);
        if screen_picker::busy(&ctx) {
            // Block input without a backdrop: the worker may still be capturing the screen.
            egui::Modal::new(egui::Id::new("screen-color-wait")).frame(egui::Frame::NONE).backdrop_color(egui::Color32::TRANSPARENT).show(&ctx, |_| {});
            ctx.set_cursor_icon(egui::CursorIcon::Wait);
        }
        let ui_ms = gpu_canvas::now_ms() - t0;
        self.perf.frame(ui_ms);
        if self.stroke_trace.enabled() {
            self.stroke_trace.canvas(self.perf.frame_refresh_ms);
            let info = &self.perf.gpu_info;
            self.stroke_trace.end_frame(ui_ms, self.live_stroke.is_some(), || format!("{} {}", info.backend, info.adapter).trim().to_string());
        }
        // Synthetic input is injected one press/release step per frame: keep frames coming until
        // the queue is empty, then release control replies waiting on it.
        if !self.synthetic.is_empty() {
            ctx.request_repaint();
        } else if !self.input_waiters.is_empty() {
            for w in self.input_waiters.drain(..) {
                let _ = w.send(serde_json::json!({"ok": true, "result": null}));
            }
        }
        // A menu or dialog can start a filter after `jobs_ui::tick` ran in this frame. Keep
        // polling it even when the platform would otherwise put the event loop to sleep.
        jobs_ui::keep_polling(self, &ctx);
        // Menus and buttons in this frame may have asked for a file dialog.
        self.poll_file_dialog(&ctx, Some(frame));
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn read_dropped(f: &(dyn egui::DroppedFile + Send + Sync)) -> Result<Vec<u8>, String> {
    f.bytes()
}

/// The active layer is an adjustment layer. It has no pixels, so painting and filling it edit its
/// mask even when its icon thumbnail, not the mask, is the Layers panel's target, as in
/// Photoshop (#4298).
pub(crate) fn adjustment_layer_active(st: &photocraft_engine::DocState) -> bool {
    st.active_layer.and_then(|id| st.doc.layer(id)).is_some_and(|l| matches!(l.content, photocraft_doc::LayerContent::Adjustment(_)))
}

/// On the web, dropped-file bytes arrive asynchronously; the web shell handles drops itself.
#[cfg(target_arch = "wasm32")]
fn read_dropped(_f: &dyn egui::DroppedFile) -> Result<Vec<u8>, String> {
    Err("drag-and-drop on the web is handled by the page; use File → Open".into())
}

impl PhotocraftApp {
    /// `params` aimed at the active layer's mask (`"target":"mask"`) when the Layers panel targets
    /// it and command `id` edits the target (adjustments, filters, fills) without naming one:
    /// ⌘I then inverts the mask, as in Photoshop (#780). A targeted alpha channel or Quick Mask
    /// mode wins, as the engine routes those itself.
    pub fn with_mask_target(&self, id: &str, params: Value) -> Value {
        if !photocraft_engine::channel_cmds::follows_target(id) || params.get("target").is_some() || !self.layer_mask_targeted() {
            return params;
        }
        match params {
            Value::Object(mut m) => {
                m.insert("target".into(), Value::from("mask"));
                Value::Object(m)
            }
            _ => serde_json::json!({ "target": "mask" }),
        }
    }

    /// The active layer's mask is what pixel commands edit: the Layers panel targets it, the layer
    /// is an adjustment layer (no pixels of its own), or every colour channel is off and that mask
    /// is the picture. An alpha channel or Quick Mask wins, as the engine routes those itself.
    pub fn layer_mask_targeted(&self) -> bool {
        let Some(st) = self.session.active() else { return false };
        let layer_mask = st.channel_view.target == photocraft_engine::channel_cmds::ChannelTarget::LayerMask;
        let composite = (st.channel_view.target == photocraft_engine::channel_cmds::ChannelTarget::Composite || layer_mask) && st.doc.quick_mask.is_none();
        let has_mask = st.active_layer.and_then(|id| st.doc.layer(id)).is_some_and(|l| l.mask.is_some());
        if !composite || !has_mask {
            return false;
        }
        if self.ui.mask_target || layer_mask || adjustment_layer_active(st) {
            return true;
        }
        let colors_off = st.channel_view.visible_colors(photocraft_engine::channel_cmds::color_count(&st.doc)) == 0;
        let alpha_on = (0..st.doc.channels.len()).any(|i| st.channel_view.alpha_shown(i));
        colors_off && !alpha_on
    }

    /// Viewing a layer mask (#196) targets it; a vector-mask target needs a vector mask on the
    /// active layer (a shape layer's path is its content, not a mask). Targeting a mask or the
    /// pixels brings back that target's foreground/background pair, as in Photoshop (#2166).
    pub(crate) fn sync_mask_targets(&mut self) {
        if let Some(st) = self.session.active_mut() {
            if photocraft_engine::mask_view_cmds::current(st).is_some() || st.channel_view.target == photocraft_engine::channel_cmds::ChannelTarget::LayerMask {
                self.ui.mask_target = true;
                self.ui.vector_mask_target = false;
            } else if st.channel_view.target == photocraft_engine::channel_cmds::ChannelTarget::VectorMask {
                self.ui.vector_mask_target = true;
                self.ui.mask_target = false;
            }
            if self.ui.vector_mask_target && !mask_thumbs_ui::has_vector_mask(st) {
                self.ui.vector_mask_target = false;
            }
            if self.ui.mask_target && st.active_layer.and_then(|id| st.doc.layer(id)).is_none_or(|l| l.mask.is_none()) {
                self.ui.mask_target = false;
            }
            if !self.ui.mask_target && st.channel_view.target == photocraft_engine::channel_cmds::ChannelTarget::LayerMask {
                st.channel_view.target = photocraft_engine::channel_cmds::ChannelTarget::Composite;
            }
            if !self.ui.mask_target || self.ui.properties_layer != st.active_layer.map(|l| l.0) {
                self.ui.properties_layer = None;
            }
        }
        let mask = self.layer_mask_targeted();
        self.session.tools.target_mask(mask);
    }

    fn prune_thumbs(&mut self) {
        let documents = self.session.documents();
        if self.thumb_documents.len() == documents.len()
            && self.thumb_documents.iter().zip(documents).all(|((id, snapshot), st)| *id == st.doc.id && snapshot.ptr_eq(&std::sync::Arc::downgrade(&st.doc)))
        {
            return;
        }
        // Check snapshots every sync, but walk the layer tree only after it changes. A Weak
        // tracks replacement (including undo and preserved-ID reopen) without retaining pixels.
        if !self.thumbs.is_empty() {
            let mut live = std::collections::HashSet::new();
            let mut pending: Vec<_> = documents.iter().flat_map(|st| &st.doc.layers).collect();
            while let Some(layer) = pending.pop() {
                live.insert((layer.id, mask_thumbs_ui::THUMB_LAYER));
                if layer.mask.is_some() {
                    live.insert((layer.id, mask_thumbs_ui::THUMB_MASK));
                }
                if layer.vector_mask.is_some() {
                    live.insert((layer.id, mask_thumbs_ui::THUMB_VECTOR));
                }
                if let Some(children) = layer.children() {
                    pending.extend(children);
                }
            }
            // Native files can restore IDs: retain the union across all open documents, rather
            // than deleting an ID just because one document containing it was closed.
            self.thumbs.retain(|key, _| live.contains(key));
        }
        self.thumb_documents = documents.iter().map(|st| (st.doc.id, std::sync::Arc::downgrade(&st.doc))).collect();
    }

    pub fn set_theme(&mut self, ctx: &egui::Context, kind: theme::ThemeKind) {
        if let Err(e) = self.run("prefs.set", serde_json::json!({"path": "interface.theme", "value": kind.id()})) {
            self.ui.status = e;
        } else {
            self.apply_theme(ctx, kind);
        }
    }

    pub(crate) fn apply_theme(&mut self, ctx: &egui::Context, kind: theme::ThemeKind) {
        // Color and Swatches trade indices between Pro and Studio layouts.
        if let Some(selected) = dock::Group::Color.tabs(theme::Tokens::for_kind(self.ui.theme).pro).get(self.ui.dock_tabs.color)
            && let Some(tab) = dock::Group::Color.tabs(theme::Tokens::for_kind(kind).pro).iter().position(|name| name == selected)
        {
            self.ui.dock_tabs.color = tab;
        }
        self.ui.theme = kind;
        panel_docking::sync_visibility(self);
        theme::apply(ctx, kind);
        self.checker = None;
    }

    /// Cached 64px thumbnail of a pixel-ish layer, laid out in document space.
    pub fn layer_thumb(&mut self, ctx: &egui::Context, doc: &Document, layer: &photocraft_doc::Layer) -> egui::TextureId {
        // Key by content, not document revision: COW tiles change pointer only when their pixels
        // change, so unrelated edits (e.g. painting another layer) don't rebuild this thumbnail.
        let rev = layer.surface().map_or(0, surface_fingerprint) ^ (doc.size.width as u64) << 40;
        let key = (layer.id, mask_thumbs_ui::THUMB_LAYER);
        if let Some((r, tex)) = self.thumbs.get(&key)
            && *r == rev
        {
            return tex.id();
        }
        let mut px = [0.0f32; 8];
        let img = thumb_image(doc, 64, |x, y| {
            let Some(s) = layer.surface() else { return [0.0; 4] };
            let n = s.channels();
            s.read_pixel(x, y, &mut px[..n]);
            photocraft_raster::to_rgba(&s.format(), &px[..n])
        });
        self.store_thumb(ctx, key, rev, img)
    }

    /// The History panel thumbnail of snapshot `snap` of document `doc` (#2966): its composite,
    /// centred in a square. Thumbnails of `doc`'s snapshots not in `live` are dropped.
    pub fn snapshot_thumb(&mut self, ctx: &egui::Context, doc: DocId, snap: &photocraft_engine::history_cmds::Snapshot, live: &[u64]) -> egui::TextureId {
        self.snapshot_thumbs.retain(|(d, id), _| *d != doc || live.contains(id));
        if let Some(tex) = self.snapshot_thumbs.get(&(doc, snap.id)) {
            return tex.id();
        }
        let comp = photocraft_compose::thumbnail(&snap.doc, 56);
        let (w, h) = (comp.width as usize, comp.height as usize);
        let side = w.max(h);
        let (ox, oy) = ((side - w) / 2, (side - h) / 2);
        let mut px = vec![egui::Color32::TRANSPARENT; side * side];
        for (i, &[r, g, b, a]) in comp.pixels.as_chunks::<4>().0.iter().enumerate() {
            if let Some(out) = px.get_mut((i / w + oy) * side + i % w + ox) {
                *out = egui::Color32::from_rgba_unmultiplied(r, g, b, a);
            }
        }
        let tex = ctx.load_texture(format!("snapshot-{}", snap.id), egui::ColorImage::new([side, side], px), egui::TextureOptions::LINEAR);
        let id = tex.id();
        self.snapshot_thumbs.insert((doc, snap.id), tex);
        id
    }

    pub fn mask_thumb(&mut self, ctx: &egui::Context, doc: &Document, id: photocraft_doc::LayerId, mask: &photocraft_doc::LayerMask) -> egui::TextureId {
        let rev = surface_fingerprint(&mask.surface) ^ (doc.size.width as u64) << 40;
        let key = (id, mask_thumbs_ui::THUMB_MASK);
        if let Some((r, tex)) = self.thumbs.get(&key)
            && *r == rev
        {
            return tex.id();
        }
        let mut v = [0.0f32; 1];
        let img = thumb_image(doc, 64, |x, y| {
            mask.surface.read_pixel(x, y, &mut v);
            [v[0], v[0], v[0], 1.0]
        });
        self.store_thumb(ctx, key, rev, img)
    }

    fn store_thumb(&mut self, ctx: &egui::Context, key: (photocraft_doc::LayerId, u8), rev: u64, img: egui::ColorImage) -> egui::TextureId {
        match self.thumbs.get_mut(&key) {
            Some((r, tex)) => {
                tex.set(img, egui::TextureOptions::LINEAR);
                *r = rev;
                tex.id()
            }
            None => {
                let tex = ctx.load_texture(format!("thumb-{}-{}", key.0.0, key.1), img, egui::TextureOptions::LINEAR);
                let id = tex.id();
                self.thumbs.insert(key, (rev, tex));
                id
            }
        }
    }
}

/// Cheap identity of a surface's pixels: its default (untouched) pixel, tile coordinates and
/// `Arc` pointers. The default pixel matters: inverting or filling a tile-less mask only changes
/// it (#2117).
pub fn surface_fingerprint(s: &photocraft_raster::Surface) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ s.tile_count() as u64;
    for &b in s.default_bytes() {
        h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
    }
    for (c, t) in s.tiles() {
        let p = std::sync::Arc::as_ptr(t) as usize as u64;
        h = (h ^ p ^ ((c.tx as u64) << 32 | c.ty as u32 as u64)).wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// Square thumbnail of the canvas area, letterboxed, sampling `f(x, y)` in document space.
fn thumb_image(doc: &Document, side: usize, mut f: impl FnMut(i32, i32) -> [f32; 4]) -> egui::ColorImage {
    let (w, h) = (doc.size.width.max(1) as f32, doc.size.height.max(1) as f32);
    let scale = w.max(h) / side as f32;
    let (ox, oy) = ((side as f32 - w / scale) / 2.0, (side as f32 - h / scale) / 2.0);
    let mut px = vec![egui::Color32::TRANSPARENT; side * side];
    for ty in 0..side {
        for tx in 0..side {
            let dx = (tx as f32 - ox + 0.5) * scale;
            let dy = (ty as f32 - oy + 0.5) * scale;
            if dx < 0.0 || dy < 0.0 || dx >= w || dy >= h {
                continue;
            }
            let c = f(dx as i32, dy as i32);
            let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            px[ty * side + tx] = egui::Color32::from_rgba_unmultiplied(q(c[0]), q(c[1]), q(c[2]), q(c[3]));
        }
    }
    egui::ColorImage::new([side, side], px)
}

impl PhotocraftApp {
    /// Next step of queued synthetic input (automation): events up to and including the first
    /// release, so egui sees press and release in separate frames. Hosts that don't call
    /// `raw_input_hook` (offscreen harnesses) feed these to their input themselves.
    pub fn take_synthetic_step(&mut self) -> Vec<egui::Event> {
        let n = self
            .synthetic
            .iter()
            .position(|e| matches!(e, egui::Event::PointerButton { pressed: false, .. } | egui::Event::Key { pressed: false, .. }))
            .map_or(self.synthetic.len(), |i| i + 1);
        let events: Vec<_> = self.synthetic.drain(..n).collect();
        if !events.is_empty() {
            self.automation_input = true;
        }
        events
    }

    /// Install fonts, image loaders and the theme. Call from the app creator when possible so the
    /// very first frame renders; otherwise `logic` does it and the first frame is skipped.
    pub fn setup_context(ctx: &egui::Context, kind: theme::ThemeKind) {
        ctx.add_plugin(tool_cursor::CursorLifecycle);
        ctx.add_plugin(single_line_drag::SingleLineDrag::default());
        theme::install_fonts(ctx);
        egui_extras::install_image_loaders(ctx);
        theme::apply(ctx, kind);
        // Right-click in any text field: its Cut / Copy / Paste menu (text_field_menu.rs).
        ctx.add_plugin(text_field_menu::TextFieldMenu);
        // egui's own ⌘+ / ⌘- / ⌘0 scale the whole interface; PhotoCraft zooms the canvas instead
        // (shortcuts.rs), like Photoshop.
        ctx.options_mut(|o| o.zoom_with_keyboard = false);
        // ⌘/Ctrl + wheel scrolls sideways, as in Photoshop; only a pinch zooms (wheel_nav.rs).
        wheel_nav::configure(ctx);
    }
}

impl PhotocraftApp {
    /// Cached `Surface::content_bounds` keyed by an id and the surface's tile identity.
    pub fn cached_bounds(&mut self, key: u64, surface: &photocraft_raster::Surface) -> photocraft_geom::Rect {
        let rev = surface_fingerprint(surface);
        if let Some((r, b)) = self.bounds_cache.get(&key)
            && *r == rev
        {
            return *b;
        }
        let b = surface.content_bounds();
        if self.bounds_cache.len() > 256 {
            self.bounds_cache.clear();
        }
        self.bounds_cache.insert(key, (rev, b));
        b
    }
}

impl PhotocraftApp {
    /// Zoom of the active document's main view: screen (device) pixels per document pixel, the
    /// user-facing factor (`100%` is `1.0`).
    pub fn current_zoom(&self) -> f32 {
        self.session.active_index().and_then(|i| self.ui.views.get(i)).map_or(1.0, |v| v.zoom)
    }

    /// Screen (device) pixels per egui point of the canvas (`ctx.pixels_per_point`), as of the
    /// last canvas frame. `1.0` before the first frame.
    pub fn canvas_ppp(&self) -> f32 {
        if self.ppp.is_finite() && self.ppp > 0.0 { self.ppp } else { 1.0 }
    }

    /// The active view's zoom in egui points per document pixel: the unit every screen-space
    /// distance, tolerance and texture-level choice on the canvas is measured in.
    pub fn point_zoom(&self) -> f32 {
        self.current_zoom() / self.canvas_ppp()
    }
}

impl PhotocraftApp {
    /// Channels panel thumbnails of the active document snapshot: the composite,
    /// each colour channel (when there is more than one), each alpha channel, then the Quick Mask.
    pub fn channel_thumbs(&mut self, ctx: &egui::Context) -> Vec<egui::TextureId> {
        let Some(st) = self.session.active() else {
            self.channel_thumbs = None;
            return Vec::new();
        };
        let (id, doc) = (st.doc.id, st.doc.clone());
        // View-only commands bump revision without changing pixels. Keeping a Weak pins allocation
        // identity against address reuse without retaining the document's pixel data.
        let snapshot = std::sync::Arc::downgrade(&doc);
        let show_color = self.session.prefs().interface.show_channels_in_color;
        // A step that changes only the selection (a marquee, Deselect) makes a new snapshot with
        // the same pixels: keep the thumbnails, a full composite at 56 px costs ~0.7 s on a
        // layered 20 MP document (#4518).
        let current = match &mut self.channel_thumbs {
            Some((d, old, in_color, _)) if *d == id && *in_color == show_color => {
                let same = old.ptr_eq(&snapshot) || old.upgrade().is_some_and(|prev| same_but_selection(&prev, &doc));
                if same {
                    *old = snapshot.clone();
                }
                same
            }
            _ => false,
        };
        if !current {
            let comp = photocraft_compose::thumbnail(&doc, 56);
            let (w, h) = (comp.width as usize, comp.height as usize);
            let side = w.max(h);
            let make = |f: &dyn Fn(&[u8]) -> egui::Color32, name: &str| {
                let mut px = vec![egui::Color32::TRANSPARENT; side * side];
                let (ox, oy) = ((side - w) / 2, (side - h) / 2);
                for y in 0..h {
                    for x in 0..w {
                        let o = (y * w + x) * 4;
                        px[(y + oy) * side + x + ox] = f(&comp.pixels[o..o + 4]);
                    }
                }
                (format!("chan-{name}"), egui::ColorImage::new([side, side], px))
            };
            let fmt = doc.pixel_format();
            let colors = fmt.mode.color_channels();
            let mut texs = vec![make(&|p| egui::Color32::from_rgb(p[0], p[1], p[2]), "composite")];
            if colors > 1 {
                for k in 0..colors {
                    let cmyk = fmt.mode == photocraft_doc::ColorMode::Cmyk;
                    texs.push(make(
                        &|p| {
                            let rgba = [p[0], p[1], p[2], 255].map(|v| f32::from(v) / 255.0);
                            let x = photocraft_raster::from_rgba(&fmt, rgba)[k];
                            let g = if cmyk { 1.0 - x } else { x };
                            let byte = (g.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                            channel_tint(fmt.mode, k, byte, show_color)
                        },
                        &format!("c{k}"),
                    ));
                }
            }
            // Alpha channels and the Quick Mask, sampled at thumbnail resolution.
            let (dw, dh) = (doc.size.width.max(1) as f32, doc.size.height.max(1) as f32);
            for (i, ch) in doc.channels.iter().chain(doc.quick_mask.as_ref()).enumerate() {
                let mut px = vec![egui::Color32::TRANSPARENT; side * side];
                let (ox, oy) = ((side - w) / 2, (side - h) / 2);
                for y in 0..h {
                    for x in 0..w {
                        let sx = ((x as f32 + 0.5) * dw / w as f32) as i32;
                        let sy = ((y as f32 + 0.5) * dh / h as f32) as i32;
                        let v = ch.surface.sample_channel(sx, sy, 0);
                        px[(y + oy) * side + x + ox] = egui::Color32::from_gray((v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
                    }
                }
                texs.push((format!("chan-a{i}"), egui::ColorImage::new([side, side], px)));
            }
            // Update the textures already uploaded rather than allocating new ones.
            let mut handles = match self.channel_thumbs.take() {
                Some((d, _, _, h)) if d == id => h,
                _ => Vec::new(),
            };
            handles.truncate(texs.len());
            for (i, (name, image)) in texs.into_iter().enumerate() {
                match handles.get_mut(i) {
                    Some(h) => h.set(image, egui::TextureOptions::LINEAR),
                    None => handles.push(ctx.load_texture(name, image, egui::TextureOptions::LINEAR)),
                }
            }
            self.channel_thumbs = Some((id, snapshot, show_color, handles));
        }
        self.channel_thumbs.as_ref().map(|(_, _, _, t)| t.iter().map(|t| t.id()).collect()).unwrap_or_default()
    }
}

/// Whether `new` differs from `old` at most in its selection, so every composite and channel
/// thumbnail of it is unchanged. Tiles compare by pointer first, so this costs little.
fn same_but_selection(old: &Document, new: &Document) -> bool {
    if old.selection == new.selection {
        return old == new;
    }
    let mut o = old.clone();
    o.selection = new.selection.clone();
    o == *new
}

/// Layer › Layer Mask commands that add a layer mask: Photoshop targets the new mask (#2166).
const ADDS_LAYER_MASK: [&str; 5] = [
    "layer.layerMask.revealAll",
    "layer.layerMask.hideAll",
    "layer.layerMask.revealSelection",
    "layer.layerMask.hideSelection",
    "layer.layerMask.fromTransparency",
];

/// The New Document dialog's key in the preferences' `dialogs` map.
const NEW_DOCUMENT: &str = "file.new";

/// New Document fields its OK remembers (#1810): the size, resolution, mode, depth, background and
/// display units. Not the name, the preset or the clipboard size.
const NEW_DOCUMENT_REMEMBERED: [&str; 8] = ["width", "height", "resolution", "mode", "depth", "background", "__unit", "__resUnit"];

/// A remembered value the dialog can show (a corrupt preference is ignored).
fn remembered_new_document_value(key: &str, v: &serde_json::Value) -> bool {
    match key {
        "width" | "height" => v.as_u64().is_some_and(|n| (1..=300_000).contains(&n)),
        "resolution" => v.as_f64().is_some_and(|r| r.is_finite() && r > 0.0 && r <= 30_000.0),
        "depth" => v.as_u64().is_some_and(|d| matches!(d, 1 | 8 | 16 | 32)),
        _ => v.as_str().is_some_and(|s| !s.is_empty() && s.len() <= 64),
    }
}

/// Remember the New Document fields `f` its OK used, for the next New Document.
pub(crate) fn remember_new_document(app: &mut PhotocraftApp, f: &serde_json::Map<String, serde_json::Value>) {
    let kept: serde_json::Map<String, serde_json::Value> =
        NEW_DOCUMENT_REMEMBERED.iter().filter_map(|k| f.get(*k).filter(|v| remembered_new_document_value(k, v)).map(|v| (k.to_string(), v.clone()))).collect();
    app.session.prefs.edit(|p| p.dialogs.insert(NEW_DOCUMENT.into(), serde_json::Value::Object(kept)));
}

fn clip_signature(w: u32, h: u32, px: &[u8]) -> u64 {
    let mut sig = (w as u64) << 32 | h as u64;
    for b in px.iter().step_by(997) {
        sig = sig.rotate_left(5) ^ *b as u64;
    }
    sig
}

/// A channel thumbnail pixel whose lightness is `byte` (255 = white: full light, no ink).
/// Preferences ▸ Interface ▸ Show Channels in Color tints RGB channels from black to their
/// primary and CMYK channels from white to their ink, as Photoshop does; otherwise grey.
fn channel_tint(mode: photocraft_doc::ColorMode, k: usize, byte: u8, in_color: bool) -> egui::Color32 {
    use photocraft_doc::ColorMode::{Cmyk, Rgb};
    if !in_color {
        return egui::Color32::from_gray(byte);
    }
    match (mode, k) {
        (Rgb, 0) => egui::Color32::from_rgb(byte, 0, 0),
        (Rgb, 1) => egui::Color32::from_rgb(0, byte, 0),
        (Rgb, 2) => egui::Color32::from_rgb(0, 0, byte),
        (Cmyk, 0) => egui::Color32::from_rgb(byte, 255, 255),
        (Cmyk, 1) => egui::Color32::from_rgb(255, byte, 255),
        (Cmyk, 2) => egui::Color32::from_rgb(255, 255, byte),
        _ => egui::Color32::from_gray(byte),
    }
}

impl PhotocraftApp {
    /// Mirror the session clipboard onto the OS clipboard (RGBA8).
    fn export_os_clipboard(&mut self) {
        if !self.session.prefs().general.export_clipboard {
            return;
        }
        let (Some(set), Some(clip)) = (self.services.clipboard_set_image.as_mut(), self.session.clipboard.as_ref()) else { return };
        let b = clip.bounds;
        if b.is_empty() {
            return;
        }
        let mut px = vec![[0u8; 4]; b.width() as usize * b.height() as usize];
        clip.surface.read_rgba8_into(b, &mut px);
        let bytes: Vec<u8> = px.into_iter().flatten().collect();
        if set(b.width(), b.height(), &bytes).is_ok() {
            self.os_clip_sig = Some(clip_signature(b.width(), b.height(), &bytes));
        }
    }

    /// File › New's fields: the defaults, then the settings of the last document made with the
    /// dialog (Photoshop starts from them, #1810), then the Clipboard preset (the clipboard image's
    /// size, selected) when the clipboard holds an image. Opening the dialog is an explicit
    /// request, so the OS clipboard is read here, as for a paste.
    pub(crate) fn new_document_fields(&mut self) -> serde_json::Map<String, serde_json::Value> {
        let mut f = crate::state::UiState::new_document_fields();
        if let Some(serde_json::Value::Object(saved)) = self.session.prefs().dialogs.get(NEW_DOCUMENT) {
            for k in NEW_DOCUMENT_REMEMBERED {
                if let Some(v) = saved.get(k).filter(|v| remembered_new_document_value(k, v)) {
                    f.insert(k.into(), v.clone());
                }
            }
        }
        self.import_os_clipboard();
        if let Some(c) = self.session.clipboard.as_ref().filter(|c| !c.bounds.is_empty()) {
            crate::new_doc_ui::set_clipboard(&mut f, c.bounds.width(), c.bounds.height());
        }
        f
    }

    /// If the OS clipboard holds an image that isn't the one we put there, make it the session
    /// clipboard (so ⌘V pastes screenshots and images copied in other apps, like Photoshop).
    /// Returns true when a new external image was imported. Once the OS clipboard no longer holds
    /// the image we mirrored or imported (text or a file was copied since), that image is stale:
    /// it is dropped rather than pasted.
    pub(crate) fn import_os_clipboard(&mut self) -> bool {
        let Some(get) = self.services.clipboard_get_image.as_mut() else { return false };
        let image = get().filter(|(w, h, bytes)| *w > 0 && *h > 0 && bytes.len() == *w as usize * *h as usize * 4);
        let Some((w, h, bytes)) = image else {
            if self.os_clip_sig.take().is_some() {
                self.session.clipboard = None;
                self.clip_external = false;
            }
            return false;
        };
        let sig = clip_signature(w, h, &bytes);
        if self.os_clip_sig == Some(sig) && self.session.clipboard.is_some() {
            return false;
        }
        let r = photocraft_geom::Rect::new(0, 0, w as i32, h as i32);
        let mut surface = photocraft_raster::Surface::from_interleaved(photocraft_color::PixelFormat::RGBA8, r, &bytes);
        surface.prune();
        self.session.clipboard = Some(photocraft_engine::edit_cmds::Clip { surface, bounds: r, layers: None, resolution: None });
        self.os_clip_sig = Some(sig);
        self.clip_external = true;
        true
    }
}

#[cfg(test)]
mod navigator_gpu_tests;

#[cfg(test)]
mod color_swatch_tests;

#[cfg(test)]
mod profile_open_tests;

#[cfg(test)]
mod profile_prompt_tests;

#[cfg(test)]
mod input_tests;

#[cfg(test)]
mod pen_variants_tests;

#[cfg(test)]
mod color_replacement_tests;
#[cfg(test)]
mod layer_shortcut_tests;
#[cfg(test)]
mod pencil_tests;
#[cfg(test)]
mod tool_shortcut_tests;

#[cfg(test)]
mod pixel_aspect_tests;

#[cfg(test)]
mod transform_fx_tests;

#[cfg(test)]
mod transform_undo_tests;

#[cfg(test)]
mod transform_type_tests;

#[cfg(test)]
mod save_identity_tests;

#[cfg(test)]
mod move_auto_select_tests;

#[cfg(test)]
mod new_group_button_tests;

#[cfg(test)]
mod color_tab_theme_tests;

#[cfg(test)]
mod new_layer_button_tests;

#[cfg(test)]
mod smart_filter_rows_tests;

#[cfg(test)]
mod blur_pins_tests;

#[cfg(test)]
mod move_outline_tests;

#[cfg(test)]
mod mask_thumb_refresh_tests;

#[cfg(test)]
mod new_doc_remember_tests;

#[cfg(test)]
mod hidden_layer_tests;

#[cfg(test)]
mod blend_dropdown_keys_tests;

#[cfg(test)]
mod warp_text_dialog_tests;

#[cfg(test)]
mod blend_dropdown_wheel_tests;

#[cfg(test)]
mod options_bar_mode_wheel_tests;

#[cfg(test)]
mod dropdown_press_tests;

#[cfg(test)]
mod marquee_tests;

#[cfg(test)]
mod canvas_menu_tests;

#[cfg(test)]
mod caps_lock_tests;

#[cfg(test)]
mod view_sync_tests;

#[cfg(test)]
mod stamp_tests;

#[cfg(test)]
mod alt_click_tests;
#[cfg(test)]
mod status_bar_tests;
#[cfg(test)]
mod stroke_timing_tests;

#[cfg(test)]
mod polygon_lasso_tests;

#[cfg(test)]
mod window_frame_tests;

#[cfg(test)]
mod clipboard_tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn os_clipboard_bridge() {
        let os: OsClip = Arc::default();
        let (a, b) = (os.clone(), os.clone());
        let services = Services {
            clipboard_set_image: Some(Box::new(move |w: u32, h: u32, px: &[u8]| {
                *a.lock().unwrap() = Some((w, h, px.to_vec()));
                Ok(())
            })),
            clipboard_get_image: Some(Box::new(move || b.lock().unwrap().clone())),
            ..Default::default()
        };
        let mut app = PhotocraftApp::new(Session::new(), services);
        app.session.execute("file.new", serde_json::json!({"width": 64, "height": 64})).unwrap();
        app.sync_views();
        app.run("select.rect", serde_json::json!({"x": 0, "y": 0, "width": 8, "height": 4})).unwrap();
        app.run("edit.copy", serde_json::json!({})).unwrap();
        let (w, h, px) = os.lock().unwrap().clone().unwrap();
        assert_eq!((w, h, px.len()), (8, 4, 8 * 4 * 4));
        // Our own image comes back unchanged (no re-import, keeps the original position).
        assert!(!app.import_os_clipboard());
        // Text copied elsewhere replaces our image: nothing to paste, not the stale image.
        let ours = os.lock().unwrap().take();
        assert!(!app.import_os_clipboard());
        assert!(app.session.clipboard.is_none());
        *os.lock().unwrap() = ours;
        app.run("edit.copy", serde_json::json!({})).unwrap();
        // Another app puts a 3×2 red image on the clipboard: ⌘V pastes it.
        *os.lock().unwrap() = Some((3, 2, [255u8, 0, 0, 255].repeat(6)));
        app.run("edit.paste", serde_json::json!({})).unwrap();
        let st = app.session.active().unwrap();
        let surf = st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap();
        assert_eq!(surf.content_bounds().width(), 3);
    }

    #[test]
    fn os_clipboard_honours_export_clipboard_preference() {
        let os: OsClip = Arc::default();
        let a = os.clone();
        let services = Services {
            clipboard_set_image: Some(Box::new(move |w: u32, h: u32, px: &[u8]| {
                *a.lock().unwrap() = Some((w, h, px.to_vec()));
                Ok(())
            })),
            ..Default::default()
        };
        let mut app = PhotocraftApp::new(Session::new(), services);
        app.session.execute("prefs.set", serde_json::json!({"values": {"general.exportClipboard": false}})).unwrap();
        app.session.execute("file.new", serde_json::json!({"width": 32, "height": 32})).unwrap();
        app.sync_views();
        app.run("select.rect", serde_json::json!({"x": 0, "y": 0, "width": 8, "height": 4})).unwrap();
        app.run("edit.copy", serde_json::json!({})).unwrap();
        assert!(os.lock().unwrap().is_none(), "copy does not mirror to OS clipboard when export_clipboard is off");
    }

    type OsClip = Arc<Mutex<Option<(u32, u32, Vec<u8>)>>>;

    /// An app whose OS clipboard is `os`, counting every read in `reads`.
    fn app_with_os_clipboard(os: &OsClip, reads: &Arc<std::sync::atomic::AtomicUsize>) -> PhotocraftApp {
        let (b, n) = (os.clone(), reads.clone());
        let get = move || {
            n.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            b.lock().unwrap().clone()
        };
        let mut app = PhotocraftApp::new(Session::new(), Services { clipboard_get_image: Some(Box::new(get)), ..Default::default() });
        app.session.execute("file.new", serde_json::json!({"width": 64, "height": 64})).unwrap();
        app.sync_views();
        app
    }

    #[test]
    fn external_clipboard_enables_and_centres_paste() {
        // Paste is enabled from the clipboard service alone (no read); the explicit paste imports
        // the OS image, once, and centres it like Photoshop does for foreign clipboard content.
        let (os, reads) = (OsClip::default(), Arc::default());
        let mut app = app_with_os_clipboard(&os, &reads);
        *os.lock().unwrap() = Some((4, 4, [255u8, 0, 0, 255].repeat(16)));
        assert!(app.session.clipboard.is_none());
        assert!(crate::menus::is_enabled(&app, "edit.paste"));
        assert!(crate::menus::is_enabled(&app, "edit.pasteSpecial.pasteInPlace"));
        assert_eq!(reads.load(std::sync::atomic::Ordering::SeqCst), 0, "enablement never reads the clipboard");
        let r = crate::menus::invoke(&mut app, &egui::Context::default(), "edit.paste", serde_json::json!({})).unwrap();
        assert_ne!(r["offset"], serde_json::json!([0, 0]), "external image is centred: {r}");
        assert_eq!(reads.load(std::sync::atomic::Ordering::SeqCst), 1, "one read per explicit paste");
        // Without a clipboard service Paste greys until something is copied in the app.
        let mut plain = PhotocraftApp::new(Session::new(), Services::default());
        plain.session.execute("file.new", serde_json::json!({"width": 8, "height": 8})).unwrap();
        assert!(!crate::menus::is_enabled(&plain, "edit.paste"));
    }

    /// #368: an image copied in another app opens as a document of its own, from File › New
    /// from Clipboard or from Paste with nothing open.
    #[test]
    fn os_clipboard_image_becomes_a_new_document() {
        let (os, reads) = (OsClip::default(), Arc::new(std::sync::atomic::AtomicUsize::new(0)));
        let (b, n) = (os.clone(), Arc::clone(&reads));
        let get = move || {
            n.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            b.lock().unwrap().clone()
        };
        let mut app = PhotocraftApp::new(Session::new(), Services { clipboard_get_image: Some(Box::new(get)), ..Default::default() });
        let ctx = egui::Context::default();
        // Listed right after File › New…, enabled without reading the clipboard.
        let items = crate::menus::menu_items(&app);
        let at = items.iter().position(|i| i.id == "file.new").unwrap();
        assert_eq!(items[at + 1].id, "file.newFromClipboard");
        assert!(items[at + 1].enabled && crate::menus::is_enabled(&app, "edit.paste"));
        assert!(!crate::menus::is_enabled(&app, "edit.pasteSpecial.pasteInPlace"), "Paste in Place needs a document");
        assert_eq!(reads.load(std::sync::atomic::Ordering::SeqCst), 0);
        // An empty clipboard is a quiet no-op.
        let r = crate::menus::invoke(&mut app, &ctx, "file.newFromClipboard", serde_json::json!({})).unwrap();
        assert_eq!(r["pasted"], serde_json::json!(false));
        assert!(app.session.documents().is_empty() && !app.ui.status_error);
        // Paste with nothing open makes the document.
        *os.lock().unwrap() = Some((5, 3, [0u8, 0, 255, 255].repeat(15)));
        crate::menus::invoke(&mut app, &ctx, "edit.paste", serde_json::json!({})).unwrap();
        let d = &app.session.active().unwrap().doc;
        assert_eq!((d.size.width, d.size.height, d.layers.len()), (5, 3, 1));
        assert_eq!(app.ui.views.len(), 1, "the new document has a view");
        assert!(app.has_unsaved_work(), "closing after Paste must ask to save");
        // New from Clipboard with a document open adds another.
        crate::menus::invoke(&mut app, &ctx, "file.newFromClipboard", serde_json::json!({})).unwrap();
        assert_eq!(app.session.documents().len(), 2);
        assert!(app.session.documents().iter().all(|d| d.is_dirty()));
        assert!(crate::discard_ui::intercept(&mut app, "file.close", &serde_json::json!({})), "Close must ask before discarding clipboard pixels");
    }

    #[test]
    fn pasting_an_empty_or_non_image_os_clipboard_is_a_quiet_no_op() {
        let (os, reads) = (OsClip::default(), Arc::default());
        let mut app = app_with_os_clipboard(&os, &reads);
        let layers = app.session.active().unwrap().doc.layers.len();
        let r = crate::menus::invoke(&mut app, &egui::Context::default(), "edit.paste", serde_json::json!({})).unwrap();
        assert_eq!(r["pasted"], serde_json::json!(false));
        assert_eq!(app.run("edit.pasteSpecial.pasteInPlace", serde_json::json!({})).unwrap()["pasted"], serde_json::json!(false));
        assert_eq!(app.session.active().unwrap().doc.layers.len(), layers);
        assert!(!app.ui.status_error && !app.ui.status.is_empty());
    }

    #[test]
    fn idle_frames_never_read_the_os_clipboard() {
        let (os, reads) = (OsClip::default(), Arc::<std::sync::atomic::AtomicUsize>::default());
        *os.lock().unwrap() = Some((4, 4, [255u8, 0, 0, 255].repeat(16)));
        let (b, n) = (os.clone(), reads.clone());
        let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(1200.0, 800.0)).with_max_steps(64).build_eframe(move |cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            let get = move || {
                n.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                b.lock().unwrap().clone()
            };
            let mut app = PhotocraftApp::new(Session::new(), Services { clipboard_get_image: Some(Box::new(get)), ..Default::default() });
            app.session.execute("file.new", serde_json::json!({"width": 64, "height": 64})).unwrap();
            app
        });
        for _ in 0..20 {
            h.step();
        }
        assert_eq!(reads.load(std::sync::atomic::Ordering::SeqCst), 0, "the OS clipboard is read only on an explicit paste");
        assert!(h.state().session.clipboard.is_none());
    }

    /// #2928: with the layer mask targeted, ⌘C copies the mask's values as grayscale, not the
    /// layer's pixels — so ⌘V (targeting another layer's mask, #1035) moves the mask itself.
    #[test]
    fn copy_with_the_mask_targeted_copies_the_mask_not_the_pixels() {
        let (os, reads) = (OsClip::default(), Arc::default());
        let mut app = app_with_os_clipboard(&os, &reads);
        app.run("layer.new.layer", serde_json::json!({})).unwrap();
        app.run("layer.layerMask.hideAll", serde_json::json!({})).unwrap();
        // Black pixels, a white square in the mask: the copy must carry the mask.
        app.run("select.all", serde_json::json!({})).unwrap();
        app.run("edit.fill", serde_json::json!({"color": "#000000"})).unwrap();
        app.run("select.rect", serde_json::json!({"x": 4, "y": 4, "width": 8, "height": 8})).unwrap();
        app.run("edit.fill", serde_json::json!({"color": "#ffffff", "target": "mask"})).unwrap();
        app.run("select.deselect", serde_json::json!({})).unwrap();
        app.ui.mask_target = true;
        crate::menus::invoke(&mut app, &egui::Context::default(), "edit.copy", serde_json::json!({})).unwrap();
        let clip = app.session.clipboard.clone().expect("the mask was copied");
        let st = app.session.active().unwrap();
        let (w, h) = (st.doc.size.width as i32, st.doc.size.height as i32);
        assert_eq!(
            [clip.bounds.x0, clip.bounds.y0, clip.bounds.width() as i32, clip.bounds.height() as i32],
            [0, 0, w, h],
            "the mask, whole — not the black pixels"
        );
        let px = clip.surface.rgba(6, 6);
        assert!(px[0] > 0.99 && px[3] > 0.99, "white mask values: {px:?}");
        assert!(clip.surface.rgba(20, 20)[0] < 0.01, "the hidden area copies as black");
    }

    /// #1035: with the layer mask targeted (its thumbnail clicked), ⌘V pastes the clipboard's
    /// luminosity into the mask instead of making a layer.
    #[test]
    fn paste_with_the_mask_targeted_goes_into_the_mask() {
        let (os, reads) = (OsClip::default(), Arc::default());
        let mut app = app_with_os_clipboard(&os, &reads);
        app.run("layer.new.layer", serde_json::json!({})).unwrap();
        app.run("layer.layerMask.hideAll", serde_json::json!({})).unwrap();
        app.ui.mask_target = true;
        app.ui.views[0].center = [32.0, 32.0];
        *os.lock().unwrap() = Some((4, 4, [128u8, 128, 128, 255].repeat(16)));
        let layers = app.session.active().unwrap().doc.layer_count();
        crate::menus::invoke(&mut app, &egui::Context::default(), "edit.paste", serde_json::json!({})).unwrap();
        let st = app.session.active().unwrap();
        assert_eq!(st.doc.layer_count(), layers, "no new layer");
        let mask = st.doc.layer(st.active_layer.unwrap()).unwrap().mask.as_ref().unwrap();
        assert!((mask.value(32, 32) - 128.0 / 255.0).abs() < 2.0 / 255.0, "the grey, centred in the view: {}", mask.value(32, 32));
        assert_eq!(mask.value(0, 0), 0.0, "the rest of the mask is unchanged");
    }

    /// #2166, measured in Photoshop 25.1: adding a layer mask targets it with white/black, and
    /// the pixels and the mask each keep their own foreground/background pair.
    #[test]
    fn targeting_a_mask_swaps_to_its_own_colour_pair() {
        use serde_json::json;
        const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
        const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
        const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
        const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
        const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
        let mut app = PhotocraftApp::new(Session::new(), Services::default());
        app.run("file.new", json!({"width": 32, "height": 32})).unwrap();
        let masked = app.run("layer.new.layer", json!({})).unwrap()["layer"].clone();
        app.run("tools.setColors", json!({"foreground": "#ff0000", "background": "#0000ff"})).unwrap();
        let pair = |app: &PhotocraftApp| (app.session.tools.foreground, app.session.tools.background);
        let target = |app: &mut PhotocraftApp, mask: bool| {
            app.ui.mask_target = mask;
            app.sync_views();
        };
        app.run("layer.layerMask.revealAll", json!({})).unwrap();
        assert!(app.ui.mask_target, "the new mask is targeted");
        assert_eq!(pair(&app), (WHITE, BLACK));
        target(&mut app, false);
        assert_eq!(pair(&app), (RED, BLUE), "the pixels get their colours back");
        target(&mut app, true);
        assert_eq!(pair(&app), (WHITE, BLACK));
        target(&mut app, false);
        app.run("tools.setColors", json!({"foreground": "#00ff00"})).unwrap();
        target(&mut app, true);
        assert_eq!(pair(&app), (WHITE, BLACK));
        target(&mut app, false);
        assert_eq!(pair(&app), (GREEN, BLUE));
        // A layer without a mask edits its pixels: their pair.
        target(&mut app, true);
        app.run("layer.new.layer", json!({})).unwrap();
        assert_eq!(pair(&app), (GREEN, BLUE));
        app.run("layer.select", json!({"layer": masked})).unwrap();
        target(&mut app, true);
        assert_eq!(pair(&app), (WHITE, BLACK));
        app.run("layer.layerMask.delete", json!({})).unwrap();
        assert_eq!(pair(&app), (GREEN, BLUE), "deleting the targeted mask goes back to the pixels");
    }
}

#[cfg(test)]
mod channel_tint_tests {
    use super::channel_tint;
    use egui::Color32;
    use photocraft_doc::ColorMode::{Cmyk, Grayscale, Rgb};

    #[test]
    fn channels_in_color_tint_rgb_from_black_and_cmyk_inks_from_white() {
        assert_eq!(channel_tint(Rgb, 0, 255, true), Color32::from_rgb(255, 0, 0));
        assert_eq!(channel_tint(Rgb, 2, 0, true), Color32::BLACK);
        // CMYK: no ink is white, full ink is the ink's colour.
        assert_eq!(channel_tint(Cmyk, 0, 255, true), Color32::WHITE);
        assert_eq!(channel_tint(Cmyk, 0, 0, true), Color32::from_rgb(0, 255, 255));
        assert_eq!(channel_tint(Cmyk, 1, 0, true), Color32::from_rgb(255, 0, 255));
        assert_eq!(channel_tint(Cmyk, 2, 0, true), Color32::from_rgb(255, 255, 0));
        assert_eq!(channel_tint(Cmyk, 3, 0, true), Color32::BLACK);
        assert_eq!(channel_tint(Grayscale, 0, 77, true), Color32::from_gray(77));
        assert_eq!(channel_tint(Rgb, 0, 77, false), Color32::from_gray(77));
    }
}
