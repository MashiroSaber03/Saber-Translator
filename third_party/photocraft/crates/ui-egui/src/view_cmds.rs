//! Shell-side View, Window and Type menu items: screen modes, Extras and the Show / Snap To
//! toggles, zoom presets, view flipping, document arrangement (tiles, n-up, floating windows,
//! Match Zoom/Location), panel shortcuts that pick the right dock tab, type UI preferences, and
//! the picker/dialog front ends of the engine's File commands.
//!
//! Everything here is view or window state ([`ViewOptions`], serde, part of `UiState`, so the
//! control channel reads and drives it). Anything that changes a document is an engine command;
//! this module only opens its dialog or file picker first.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::PhotocraftApp;
use crate::state::DialogKind;

/// View › Show flags (Photoshop defaults).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Show {
    pub layer_edges: bool,
    pub selection_edges: bool,
    pub target_path: bool,
    pub pixel_grid: bool,
    pub smart_guides: bool,
    pub brush_preview: bool,
    pub slices: bool,
    pub notes: bool,
    pub mesh: bool,
    pub edit_pins: bool,
    pub count: bool,
    pub canvas_guides: bool,
    pub artboard_guides: bool,
}

impl Default for Show {
    fn default() -> Self {
        Self {
            layer_edges: false,
            selection_edges: true,
            target_path: true,
            pixel_grid: true,
            smart_guides: true,
            brush_preview: true,
            slices: true,
            notes: true,
            mesh: false,
            edit_pins: true,
            count: true,
            canvas_guides: true,
            artboard_guides: true,
        }
    }
}

/// View › Snap To targets (snapping itself is the `view.snap` master switch).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SnapTo {
    pub guides: bool,
    pub grid: bool,
    pub layers: bool,
    pub slices: bool,
    pub document_bounds: bool,
}

impl Default for SnapTo {
    fn default() -> Self {
        Self { guides: true, grid: true, layers: true, slices: true, document_bounds: true }
    }
}

/// View-, window- and type-preference state of the shell.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ViewOptions {
    /// "standard" | "fullScreenWithMenuBar" | "fullScreen" (View › Screen Mode, F cycles).
    pub screen_mode: String,
    /// View › Extras (⌘H): master switch for guides, grid, edges and other overlays.
    pub extras: bool,
    pub show: Show,
    pub snap_to: SnapTo,
    /// View › Flip Horizontal: mirror the view only (the document is untouched).
    pub flip_horizontal: bool,
    /// View › Pixel Aspect Ratio preset id ("square", "d1DvNtsc", …) and its correction toggle.
    pub pixel_aspect: String,
    pub pixel_aspect_correction: bool,
    pub pattern_preview: bool,
    pub pixel_art_preview: bool,
    /// Window › Arrange layout of the document area: "tabs" or a tile/n-up layout id.
    pub arrange: String,
    /// Type › Font Preview Size: "small" | "medium" | "large" | "extraLarge" | "huge".
    pub font_preview_size: String,
    /// Type › Language Options: "defaultFeatures" | "eastAsianFeatures" | "middleEasternFeatures",
    /// and the Middle Eastern & South Asian composer.
    pub language_features: String,
    pub middle_eastern_composer: bool,
    /// Parameters from the last successfully applied New Guide Layout dialog.
    pub guide_layout: Value,
    /// Developer overlay of the active layer's storage tiles (tile_overlay.rs), set with the
    /// control channel's `ui.set {"tileGrid": true}`.
    pub tile_grid: bool,
}

/// Factory New Guide Layout values (Alt+Reset and a fresh ViewOptions).
pub(crate) fn default_guide_layout() -> Value {
    json!({"columns": 8, "gutter": 20, "rows": 0, "rowGutter": 0, "margin": 0, "centerColumns": false, "clearExisting": false})
}

impl Default for ViewOptions {
    fn default() -> Self {
        Self {
            screen_mode: "standard".into(),
            extras: true,
            show: Show::default(),
            snap_to: SnapTo::default(),
            flip_horizontal: false,
            pixel_aspect: "square".into(),
            // Photoshop shows non-square pixels corrected until the user turns it off.
            pixel_aspect_correction: true,
            pattern_preview: false,
            pixel_art_preview: false,
            arrange: "tabs".into(),
            font_preview_size: "medium".into(),
            language_features: "defaultFeatures".into(),
            middle_eastern_composer: false,
            guide_layout: default_guide_layout(),
            tile_grid: false,
        }
    }
}

impl ViewOptions {
    /// Is an overlay with this Show flag visible (Extras on and the flag on)?
    pub fn shows(&self, flag: bool) -> bool {
        self.extras && flag
    }
    pub fn hides_chrome(&self) -> bool {
        self.screen_mode == "fullScreen"
    }
    pub fn hides_tabs(&self) -> bool {
        self.screen_mode != "standard"
    }
    /// Horizontal display scale of a document pixel: the pixel aspect ratio while View › Pixel
    /// Aspect Ratio Correction is on, else 1 (square). The document's pixels never change.
    pub fn display_aspect(&self) -> f32 {
        let r = pixel_aspect_ratio(&self.pixel_aspect);
        // `ui.set` can store any string; only the dialog's 0.1–10 range is drawn.
        if self.pixel_aspect_correction && r.is_finite() { r.clamp(0.1, 10.0) } else { 1.0 }
    }
    /// Photoshop's full screen modes show no scroll bars.
    pub fn shows_scrollbars(&self) -> bool {
        self.screen_mode == "standard"
    }
}

const SCREEN_MODES: [&str; 3] = ["standard", "fullScreenWithMenuBar", "fullScreen"];
const PIXEL_ASPECTS: [&str; 8] = ["square", "d1DvNtsc", "d1DvPal", "d1DvNtscWidescreen", "hdv1080", "d1DvPalWidescreen", "dvcproHd1080", "anamorphic2To1"];
const FONT_PREVIEW: [&str; 5] = ["small", "medium", "large", "extraLarge", "huge"];
const LANGUAGE: [&str; 3] = ["defaultFeatures", "eastAsianFeatures", "middleEasternFeatures"];
/// Window › Arrange layouts drawn by [`cells`].
const LAYOUTS: [&str; 10] = [
    "tileAllVertically",
    "tileAllHorizontally",
    "twoUpVertical",
    "twoUpHorizontal",
    "threeUpVertical",
    "threeUpHorizontal",
    "threeUpStacked",
    "fourUp",
    "sixUp",
    "tile",
];

/// Whether the window was maximized when a screen mode took it full screen (egui temp data).
fn was_maximized_id() -> egui::Id {
    egui::Id::new("pc-full-screen-was-maximized")
}

/// The window commands for switching to screen mode `next`, and whether the window should be
/// maximized again when it leaves full screen. With `keep_apart` (Windows), full screen and
/// maximized never overlap: a maximized window is un-maximized before it goes full screen and
/// maximized again after it leaves. Why: winit on Windows keeps the maximized style on a window
/// that goes full screen from maximized, and the shell then draws the taskbar black instead of
/// hiding it; on the way out the window stays maximized at the monitor's full size, over the
/// taskbar (#3550, #4018, #4230). [OBSERVED 2026-10-12, Windows 11, eframe 0.36.2, winit 0.30.13,
/// a minimal eframe window; un-maximizing first gave a clean full screen and a work-area maximize.]
///
/// WORKAROUND(winit#4756): remove `keep_apart` once the winit in Cargo.lock (0.30.13 today) has
/// the fix; check first with https://gist.github.com/aweussom/3f80ffb82df7f0102341858afbbffd9b
/// (`--max`, no `--fix`: full screen must have no WS_MAXIMIZE and hide the taskbar).
fn screen_mode_commands(keep_apart: bool, full_screen: bool, maximized: bool, was_maximized: bool, next: &str) -> (Vec<egui::ViewportCommand>, bool) {
    let want = next != "standard";
    let mut commands = Vec::new();
    let mut remember = was_maximized;
    if keep_apart && want && !full_screen && maximized {
        commands.push(egui::ViewportCommand::Maximized(false));
        remember = true;
    }
    commands.push(egui::ViewportCommand::Fullscreen(want));
    if !want {
        if keep_apart && full_screen && remember {
            commands.push(egui::ViewportCommand::Maximized(true));
        }
        remember = false;
    }
    (commands, remember)
}

/// Pixel aspect ratio (width / height of a pixel) of a preset id.
pub fn pixel_aspect_ratio(id: &str) -> f32 {
    // View › Pixel Aspect Ratio › Custom: `custom:<ratio>:<name>` (see `workspace_ui`).
    if let Some(r) = id.strip_prefix("custom:").and_then(|r| r.split(':').next()?.parse::<f32>().ok()) {
        return r;
    }
    match id {
        "d1DvNtsc" => 0.91,
        "d1DvPal" => 1.09,
        "d1DvNtscWidescreen" => 1.21,
        "hdv1080" => 1.33,
        "d1DvPalWidescreen" => 1.46,
        "dvcproHd1080" => 1.5,
        "anamorphic2To1" => 2.0,
        _ => 1.0,
    }
}

fn show_slot<'a>(s: &'a mut Show, key: &str) -> Option<&'a mut bool> {
    Some(match key {
        "layerEdges" => &mut s.layer_edges,
        "selectionEdges" => &mut s.selection_edges,
        "targetPath" => &mut s.target_path,
        "pixelGrid" => &mut s.pixel_grid,
        "smartGuides" => &mut s.smart_guides,
        "brushPreview" => &mut s.brush_preview,
        "slices" => &mut s.slices,
        "notes" => &mut s.notes,
        "mesh" => &mut s.mesh,
        "editPins" => &mut s.edit_pins,
        "count" => &mut s.count,
        "canvasGuides" => &mut s.canvas_guides,
        "artboardGuides" => &mut s.artboard_guides,
        _ => return None,
    })
}

fn snap_slot<'a>(s: &'a mut SnapTo, key: &str) -> Option<&'a mut bool> {
    Some(match key {
        "guides" => &mut s.guides,
        "grid" => &mut s.grid,
        "layers" => &mut s.layers,
        "slices" => &mut s.slices,
        "documentBounds" => &mut s.document_bounds,
        _ => return None,
    })
}

/// Every [`show_slot`] key.
const SHOW_KEYS: [&str; 13] = [
    "layerEdges",
    "selectionEdges",
    "targetPath",
    "pixelGrid",
    "smartGuides",
    "brushPreview",
    "slices",
    "notes",
    "mesh",
    "editPins",
    "count",
    "canvasGuides",
    "artboardGuides",
];
/// Every [`snap_slot`] key.
const SNAP_KEYS: [&str; 5] = ["guides", "grid", "layers", "slices", "documentBounds"];

/// The flags of `now` that differ from `def`, by key (`None` when none do).
fn changed_flags<T: Copy>(now: T, def: T, keys: &[&str], slot: impl for<'a> Fn(&'a mut T, &str) -> Option<&'a mut bool>) -> Option<Value> {
    let (mut now, mut def) = (now, def);
    let diff: Map<String, Value> = keys
        .iter()
        .filter_map(|k| {
            let v = *slot(&mut now, k)?;
            (Some(v) != slot(&mut def, k).copied()).then(|| ((*k).to_string(), Value::Bool(v)))
        })
        .collect();
    (!diff.is_empty()).then_some(Value::Object(diff))
}

/// View › Extras, Show and Snap To as remembered across restarts (`Preferences::view_extras`):
/// only what differs from the defaults, so a default changed in a later release still reaches
/// users who never touched that item. `Null` when nothing differs.
fn remembered(app: &PhotocraftApp) -> Value {
    let (v, e, def) = (&app.ui.view, &app.ui.extras, crate::state::Extras::default());
    let mut m = Map::new();
    if v.extras != ViewOptions::default().extras {
        m.insert("extras".into(), Value::Bool(v.extras));
    }
    // View › Show › Grid and Guides live in `UiState::extras` beside the rulers and snapping.
    if e.grid != def.grid {
        m.insert("grid".into(), Value::Bool(e.grid));
    }
    if e.guides != def.guides {
        m.insert("guides".into(), Value::Bool(e.guides));
    }
    if let Some(show) = changed_flags(v.show, Show::default(), &SHOW_KEYS, show_slot) {
        m.insert("show".into(), show);
    }
    if let Some(snap) = changed_flags(v.snap_to, SnapTo::default(), &SNAP_KEYS, snap_slot) {
        m.insert("snapTo".into(), snap);
    }
    if m.is_empty() { Value::Null } else { Value::Object(m) }
}

/// Remember View › Extras, Show and Snap To in the preferences once the user lets go of the
/// pointer (#3762). Cheap: a small JSON compare per frame. Run every frame, so items set by the
/// control channel or the Show Extras Options dialog are remembered too.
pub fn persist(app: &mut PhotocraftApp, ctx: &egui::Context) {
    if ctx.input(|i| i.pointer.any_down()) {
        return;
    }
    let now = remembered(app);
    if app.session.prefs().view_extras != now {
        app.session.prefs.edit(|p| p.view_extras = now);
    }
}

/// Restore the remembered View › Extras, Show and Snap To items at launch. A hand-edited file
/// can't break them: an unknown key or a value that isn't a boolean keeps its default.
pub fn restore(app: &mut PhotocraftApp) {
    let saved = app.session.prefs().view_extras.clone();
    if !saved.is_null() && !saved.is_object() {
        log::warn!("Preferences: the remembered View › Show items are not an object; using the defaults");
        return;
    }
    let flag = |k: &str| saved.get(k).and_then(Value::as_bool);
    if let Some(on) = flag("extras") {
        app.ui.view.extras = on;
    }
    if let Some(on) = flag("grid") {
        app.ui.extras.grid = on;
    }
    if let Some(on) = flag("guides") {
        app.ui.extras.guides = on;
    }
    for (k, v) in saved.get("show").and_then(Value::as_object).into_iter().flatten() {
        if let (Some(slot), Some(on)) = (show_slot(&mut app.ui.view.show, k), v.as_bool()) {
            *slot = on;
        }
    }
    for (k, v) in saved.get("snapTo").and_then(Value::as_object).into_iter().flatten() {
        if let (Some(slot), Some(on)) = (snap_slot(&mut app.ui.view.snap_to, k), v.as_bool()) {
            *slot = on;
        }
    }
}

/// Window › <panel>: (panel, dock tab) for panels that live as tabs of a dock card.
fn panel_tab(app: &PhotocraftApp, id: &str) -> Option<(&'static str, usize)> {
    let pro = app.ui.theme.is_pro();
    Some(match id {
        "window.panel.info" => ("navigator", 2),
        "window.panel.histogram" => ("navigator", 1),
        "window.panel.navigator" => ("navigator", 0),
        "window.panel.actions" => ("history", 1),
        "window.panel.layerComps" => ("history", 2),
        "window.panel.history" => ("history", 0),
        "window.panel.channels" => ("layers", 1),
        "window.panel.paths" => ("layers", 2),
        "window.panel.layers" => ("layers", 0),
        "window.panel.adjustments" => ("properties", 1),
        "window.panel.properties" => ("properties", 0),
        // Their own Character | Paragraph group, so Properties stays open (#150).
        "window.panel.character" | "type.panels.character" => ("character", 0),
        "window.panel.paragraph" | "type.panels.paragraph" => ("character", 1),
        "window.panel.swatches" => ("color", usize::from(pro)),
        "window.panel.color" => ("color", usize::from(!pro)),
        _ => return None,
    })
}

fn panel_state<'a>(app: &'a mut PhotocraftApp, panel: &str) -> (&'a mut bool, &'a mut usize) {
    let (p, t) = (&mut app.ui.panels, &mut app.ui.dock_tabs);
    match panel {
        "navigator" => (&mut p.navigator, &mut t.navigator),
        "history" => (&mut p.history, &mut t.history),
        "layers" => (&mut p.layers, &mut t.layers),
        "color" => (&mut p.color, &mut t.color),
        "character" => (&mut p.character, &mut t.character),
        _ => (&mut p.properties, &mut t.properties),
    }
}

/// Ids handled here (live menu items).
pub fn handles(id: &str) -> bool {
    if let Some(k) = id.strip_prefix("view.show.") {
        return show_slot(&mut Show::default(), k).is_some() || k == "all";
    }
    if let Some(k) = id.strip_prefix("view.snapTo.") {
        return snap_slot(&mut SnapTo::default(), k).is_some() || k == "all" || k == "none";
    }
    if let Some(k) = id.strip_prefix("view.screenMode.") {
        return SCREEN_MODES.contains(&k) || k == "cycle";
    }
    if let Some(k) = id.strip_prefix("view.pixelAspectRatio.") {
        return PIXEL_ASPECTS.contains(&k);
    }
    if let Some(k) = id.strip_prefix("type.fontPreviewSize.") {
        return FONT_PREVIEW.contains(&k);
    }
    if let Some(k) = id.strip_prefix("type.languageOptions.") {
        return LANGUAGE.contains(&k) || k == "middleEasternAndSouthAsianComposer";
    }
    if let Some(k) = id.strip_prefix("window.arrange.") {
        return LAYOUTS.contains(&k)
            || matches!(
                k,
                "consolidateAllToTabs"
                    | "cascade"
                    | "floatInWindow"
                    | "floatAllInWindows"
                    | "newWindowForDocument"
                    | "matchZoom"
                    | "matchLocation"
                    | "matchRotation"
                    | "matchAll"
            );
    }
    matches!(
        id,
        "view.extras"
            | "view.twoHundredPercent"
            | "view.printSize"
            | "view.fitLayersOnScreen"
            | "view.fitArtboardOnScreen"
            | "window.panel.layerComps"
            | "view.flipHorizontal"
            | "view.pixelAspectRatioCorrection"
            | "view.patternPreview"
            | "view.pixelArtPreview"
            | "view.rotateView"
            | "view.resetView"
            | "window.panel.layers"
            | "window.panel.history"
            | "window.panel.navigator"
            | "window.panel.properties"
            | "window.panel.color"
            | "window.panel.actions"
            | "window.panel.channels"
            | "window.panel.paths"
            | "window.panel.character"
            | "window.panel.paragraph"
            | "window.panel.info"
            | "window.panel.histogram"
            | "window.panel.swatches"
            | "window.panel.adjustments"
            | "type.panels.character"
            | "type.panels.paragraph"
    )
}

pub fn is_enabled(app: &PhotocraftApp, id: &str) -> Option<bool> {
    if !handles(id) && !wraps(id) {
        return None;
    }
    let doc = app.session.active().is_some();
    Some(match id {
        "view.screenMode.cycle" => app.ui.text_edit.is_none(),
        "view.twoHundredPercent" | "view.printSize" | "view.fitLayersOnScreen" | "view.rotateView" | "view.resetView" => doc,
        "view.fitArtboardOnScreen" => app.session.active().is_some_and(|d| d.doc.has_artboards()),
        i if i.starts_with("window.arrange.") => match &i["window.arrange.".len()..] {
            "consolidateAllToTabs" => true,
            "floatInWindow" | "newWindowForDocument" => doc,
            "matchZoom" | "matchLocation" | "matchRotation" | "matchAll" => app.session.documents().len() > 1 || !app.ui.windows.is_empty(),
            _ => {
                app.session.documents().len() > 1
                    || (doc && matches!(&i["window.arrange.".len()..], "floatAllInWindows" | "cascade" | "tile" | "tileAllVertically" | "tileAllHorizontally"))
            }
        },
        i if wraps(i) => app.session.is_enabled(i),
        _ => true,
    })
}

pub fn checked(app: &PhotocraftApp, id: &str) -> Option<bool> {
    let o = &app.ui.view;
    if let Some(k) = id.strip_prefix("view.show.") {
        let mut s = o.show;
        return show_slot(&mut s, k).map(|b| *b);
    }
    if let Some(k) = id.strip_prefix("view.snapTo.") {
        let mut s = o.snap_to;
        return snap_slot(&mut s, k).map(|b| *b);
    }
    if let Some(k) = id.strip_prefix("view.screenMode.") {
        return SCREEN_MODES.contains(&k).then(|| o.screen_mode == k);
    }
    if let Some(k) = id.strip_prefix("view.pixelAspectRatio.") {
        return PIXEL_ASPECTS.contains(&k).then(|| o.pixel_aspect == k);
    }
    if let Some(k) = id.strip_prefix("type.fontPreviewSize.") {
        return FONT_PREVIEW.contains(&k).then(|| o.font_preview_size == k);
    }
    if let Some(k) = id.strip_prefix("type.languageOptions.") {
        return if k == "middleEasternAndSouthAsianComposer" {
            Some(o.middle_eastern_composer)
        } else {
            LANGUAGE.contains(&k).then(|| o.language_features == k)
        };
    }
    if let Some(k) = id.strip_prefix("window.arrange.")
        && LAYOUTS.contains(&k)
    {
        return Some(o.arrange == k);
    }
    if let Some((panel, tab)) = panel_tab(app, id) {
        let (p, t) = (&app.ui.panels, &app.ui.dock_tabs);
        let (vis, cur) = match panel {
            "navigator" => (p.navigator, t.navigator),
            "history" => (p.history, t.history),
            "layers" => (p.layers, t.layers),
            "color" => (p.color, t.color),
            "character" => (p.character, t.character),
            _ => (p.properties, t.properties),
        };
        return Some(vis && cur == tab);
    }
    if let Some(c) = type_checked(app, id) {
        return Some(c);
    }
    Some(match id {
        "view.extras" => o.extras,
        "view.flipHorizontal" => o.flip_horizontal,
        "view.pixelAspectRatioCorrection" => o.pixel_aspect_correction,
        "view.patternPreview" => o.pattern_preview,
        "view.pixelArtPreview" => o.pixel_art_preview,
        _ => return None,
    })
}

/// Checkmarks of the Type menu's anti-aliasing, orientation and OpenType items, for the layer the
/// options bar and Properties show: the one being edited, else the active or a selected type layer.
/// Anti-aliasing and orientation stay check items (`Some(false)`) without a type layer: a native
/// menu can't change an item's kind in place.
fn type_checked(app: &PhotocraftApp, id: &str) -> Option<bool> {
    let t = crate::type_tool::target_text(app);
    if let Some(k) = id.strip_prefix("type.antiAlias.") {
        return Some(t.is_some_and(|t| photocraft_engine::type_extra_cmds::aa_name(t.antialias) == k));
    }
    if let Some(k) = id.strip_prefix("type.orientation.") {
        return Some(t.is_some_and(|t| (t.orientation == photocraft_doc::text::Orientation::Vertical) == (k == "vertical")));
    }
    let t = t?;
    let tag = match id.strip_prefix("type.openType.")? {
        "standardLigatures" => "liga",
        "contextualAlternates" => "calt",
        "discretionaryLigatures" => "dlig",
        "swash" => "swsh",
        "oldstyle" => "onum",
        "stylisticAlternates" => "salt",
        "titlingAlternates" => "titl",
        "ornaments" => "ornm",
        "ordinals" => "ordn",
        "fractions" => "frac",
        _ => return None,
    };
    Some(t.char_runs().first().is_some_and(|r| photocraft_engine::type_extra_cmds::feature_on(&r.style, tag)))
}

/// Engine commands that get a dialog or file picker here when invoked without parameters.
fn wraps(id: &str) -> bool {
    matches!(
        id,
        "file.openAs"
            | "file.saveACopy"
            | "file.placeEmbedded"
            | "file.placeLinked"
            | "layer.smartObjects.replaceContents"
            | "layer.smartObjects.exportContents"
            | "layer.smartObjects.convertToLinked"
            | "file.closeAll"
            | "file.closeOthers"
            | "file.fileInfo"
            | "file.automate.fitImage"
            | "file.automate.conditionalModeChange"
            | "file.automate.batch"
            | "file.scripts.imageProcessor"
            | "file.scripts.loadFilesIntoStack"
            | "file.automate.photomerge"
            | "file.automate.mergeToHdrPro"
            | "file.automate.lensCorrection"
            | "file.export.layersToFiles"
            | "file.export.layerCompsToFiles"
            | "file.export.artboardsToFiles"
            | "file.export.artboardsToPdf"
            | "layer.new.artboard"
            | "file.export.colorLookupTables"
            | "view.newGuideLayout"
            | "type.warpText"
            | "type.pasteLoremIpsum"
            | "image.applyImage"
            | "image.calculations"
            | "select.saveSelection"
            | "select.loadSelection"
    )
}

fn no_params(p: &Value) -> bool {
    p.as_object().is_none_or(|o| o.is_empty())
}

/// Run a View/Window/Type shell command, or front an engine command with its dialog. `None` when
/// `id` isn't ours.
pub fn invoke(app: &mut PhotocraftApp, ctx: &egui::Context, id: &str, params: &Value) -> Option<Result<Value, String>> {
    if wraps(id) {
        return front(app, id, params);
    }
    if !handles(id) {
        return None;
    }
    Some(run(app, ctx, id, params))
}

fn flag_param(p: &Value, cur: bool) -> bool {
    p.get("on").and_then(Value::as_bool).unwrap_or(!cur)
}

fn run(app: &mut PhotocraftApp, ctx: &egui::Context, id: &str, p: &Value) -> Result<Value, String> {
    if let Some((panel, tab)) = panel_tab(app, id) {
        let group = crate::dock::Group::from_key(panel);
        let collapsed = group.is_some_and(|g| app.ui.dock.is_collapsed(g));
        let (vis, cur) = panel_state(app, panel);
        let requested = match p.get("show").or_else(|| p.get("on")) {
            None => None,
            Some(value) => Some(value.as_bool().ok_or("show must be a boolean")?),
        };
        if let Some(show) = requested {
            *vis = show;
            if show {
                *cur = tab;
            }
        } else {
            // Like Photoshop: choosing a visible panel's menu item again hides it. A collapsed
            // group is expanded instead, so the menu item always brings the panel back (#129).
            if *vis && *cur == tab && !collapsed && !id.starts_with("type.panels.") {
                *vis = false;
            } else {
                *vis = true;
                *cur = tab;
            }
        }
        let visible = *vis;
        if let Some(g) = group.filter(|_| visible) {
            crate::dock::reveal(app, g);
        }
        return Ok(json!({"panel": panel, "tab": tab, "visible": visible}));
    }
    if id == "view.rotateView" {
        return crate::rotate_view::command(app, p);
    }
    if id == "view.resetView" {
        return crate::rotate_view::reset(app);
    }
    let o = &mut app.ui.view;
    if let Some(k) = id.strip_prefix("view.show.") {
        if k == "all" {
            o.show = Show { layer_edges: true, mesh: true, ..Show::default() };
            o.extras = true;
            app.ui.extras.grid = true;
            app.ui.extras.guides = true;
            return Ok(json!(true));
        }
        let slot = show_slot(&mut o.show, k).ok_or("unknown Show item")?;
        *slot = flag_param(p, *slot);
        let v = *slot;
        if v {
            // Showing an item turns Extras back on, as in Photoshop.
            o.extras = true;
        }
        return Ok(json!(v));
    }
    if let Some(k) = id.strip_prefix("view.snapTo.") {
        if k == "all" || k == "none" {
            let on = k == "all";
            o.snap_to = SnapTo { guides: on, grid: on, layers: on, slices: on, document_bounds: on };
            return Ok(json!(on));
        }
        let slot = snap_slot(&mut o.snap_to, k).ok_or("unknown Snap To item")?;
        *slot = flag_param(p, *slot);
        return Ok(json!(*slot));
    }
    if let Some(k) = id.strip_prefix("view.screenMode.") {
        let next = if k == "cycle" {
            let i = SCREEN_MODES.iter().position(|m| *m == o.screen_mode).unwrap_or(0);
            SCREEN_MODES[(i + 1) % SCREEN_MODES.len()]
        } else {
            k
        };
        o.screen_mode = next.to_string();
        // Both full-screen modes take the whole display; the menu bar stays in the first.
        let (full_screen, maximized) = ctx.input(|i| (i.viewport().fullscreen.unwrap_or(false), i.viewport().maximized.unwrap_or(false)));
        let was_maximized = ctx.data(|d| d.get_temp::<bool>(was_maximized_id())).unwrap_or(false);
        let (commands, was_maximized) = screen_mode_commands(cfg!(target_os = "windows"), full_screen, maximized, was_maximized, next);
        ctx.data_mut(|d| d.insert_temp(was_maximized_id(), was_maximized));
        for command in commands {
            ctx.send_viewport_cmd(command);
        }
        return Ok(json!({"screenMode": next}));
    }
    if let Some(k) = id.strip_prefix("view.pixelAspectRatio.") {
        o.pixel_aspect = k.to_string();
        return Ok(json!({"pixelAspectRatio": pixel_aspect_ratio(k)}));
    }
    if let Some(k) = id.strip_prefix("type.fontPreviewSize.") {
        o.font_preview_size = k.to_string();
        return Ok(json!(k));
    }
    if let Some(k) = id.strip_prefix("type.languageOptions.") {
        if k == "middleEasternAndSouthAsianComposer" {
            o.middle_eastern_composer = flag_param(p, o.middle_eastern_composer);
            return Ok(json!(o.middle_eastern_composer));
        }
        o.language_features = k.to_string();
        return Ok(json!(k));
    }
    if let Some(k) = id.strip_prefix("window.arrange.") {
        return arrange(app, k, p);
    }
    match id {
        "view.extras" => {
            o.extras = flag_param(p, o.extras);
            Ok(json!(o.extras))
        }
        "view.flipHorizontal" => {
            o.flip_horizontal = flag_param(p, o.flip_horizontal);
            Ok(json!(o.flip_horizontal))
        }
        "view.pixelAspectRatioCorrection" => {
            o.pixel_aspect_correction = flag_param(p, o.pixel_aspect_correction);
            Ok(json!(o.pixel_aspect_correction))
        }
        "view.patternPreview" => {
            o.pattern_preview = flag_param(p, o.pattern_preview);
            Ok(json!(o.pattern_preview))
        }
        "view.pixelArtPreview" => {
            o.pixel_art_preview = flag_param(p, o.pixel_art_preview);
            Ok(json!(o.pixel_art_preview))
        }
        "view.twoHundredPercent" | "view.printSize" => {
            let i = app.session.active_index().ok_or("no document")?;
            // Print Size shows the document at its print size for the screen resolution
            // (Preferences ▸ Units & Rulers: Screen Resolution, 72 ppi by default).
            let dpi = app.session.active().map_or(72.0, |d| d.doc.resolution_dpi.max(1.0));
            let size = app.session.active().map_or([0, 0], |d| [d.doc.size.width, d.doc.size.height]);
            let screen = app.session.prefs().units_and_rulers.screen_resolution.max(1.0) as f32;
            let z = if id == "view.twoHundredPercent" { 2.0 } else { screen / dpi };
            app.ui.views[i].zoom = crate::zoom_levels::clamp(z, size);
            Ok(json!({"zoom": app.ui.views[i].zoom}))
        }
        "view.fitLayersOnScreen" => fit_layers(app),
        "view.fitArtboardOnScreen" => crate::artboard_ui::fit_artboard(app),
        _ => Err(format!("unhandled {id}")),
    }
}

/// View › Fit Layer(s) on Screen: zoom and centre on the selected layers' bounds.
fn fit_layers(app: &mut PhotocraftApp) -> Result<Value, String> {
    let i = app.session.active_index().ok_or("no document")?;
    let st = app.session.active().ok_or("no document")?;
    let mut b = photocraft_geom::Rect::EMPTY;
    for id in st.selected_layers() {
        if let Some(r) = st.doc.layer(id).and_then(|l| l.surface()).map(|s| s.content_bounds()) {
            b = if b.is_empty() { r } else { b.union(&r) };
        }
    }
    if b.is_empty() {
        b = st.doc.bounds();
    }
    let size = [st.doc.size.width, st.doc.size.height];
    let area = app.last_canvas_rect.size();
    let area = if area.x > 50.0 { area } else { egui::vec2(1200.0, 800.0) };
    // The area is in egui points; the stored zoom is device pixels per document pixel.
    let ppp = app.canvas_ppp();
    let points = ((area.x - 40.0) / b.width().max(1) as f32).min((area.y - 40.0) / b.height().max(1) as f32);
    let v = &mut app.ui.views[i];
    v.zoom = crate::zoom_levels::clamp(points * ppp, size);
    v.center = [(b.x0 + b.x1) as f32 / 2.0, (b.y0 + b.y1) as f32 / 2.0];
    v.fit_pending = false;
    Ok(json!({"zoom": v.zoom, "bounds": [b.x0, b.y0, b.x1, b.y1]}))
}

/// A finite `[x, y]` screen position within a sane range (window systems reject wild values).
fn screen_position(v: &Value) -> Option<[f32; 2]> {
    const LIMIT: f64 = 1.0e6;
    let a = v.as_array().filter(|a| a.len() == 2)?;
    let n = |i: usize| a.get(i).and_then(Value::as_f64).filter(|x| x.is_finite() && x.abs() <= LIMIT).map(|x| x as f32);
    Some([n(0)?, n(1)?])
}

pub(crate) fn float_window(app: &mut PhotocraftApp, doc: usize, offset: usize) -> u64 {
    let wid = app.ui.alloc_id();
    let mut view = app.ui.views.get(doc).cloned().unwrap_or_default();
    view.fit_pending = offset > 0 || view.fit_pending;
    app.ui.windows.push(crate::state::DocWindow { id: wid, document: doc, view, open: true, position: None });
    wid
}

fn arrange(app: &mut PhotocraftApp, k: &str, p: &Value) -> Result<Value, String> {
    if LAYOUTS.contains(&k) {
        app.ui.view.arrange = k.to_string();
        // Each document fits its new cell.
        app.ui.views.iter_mut().for_each(|v| v.fit_pending = true);
        return Ok(json!({"arrange": k}));
    }
    match k {
        "consolidateAllToTabs" => {
            app.ui.view.arrange = "tabs".into();
            app.ui.windows.clear();
            Ok(json!({"arrange": "tabs"}))
        }
        "newWindowForDocument" => {
            let doc = app.session.active_index().ok_or("no document")?;
            Ok(json!({"window": float_window(app, doc, 0)}))
        }
        // Optional `document` (index; default the active one) and `position` (`[x, y]`, the
        // window's screen position in points): dragging a tab off the tab strip uses both (#3733).
        "floatInWindow" => {
            let doc = match p.get("document") {
                None | Some(Value::Null) => app.session.active_index().ok_or("no document")?,
                Some(v) => {
                    let i = v.as_u64().ok_or("document must be a document index")?;
                    usize::try_from(i).ok().filter(|&i| i < app.session.documents().len()).ok_or(format!("no document {i}"))?
                }
            };
            let position = match p.get("position") {
                None | Some(Value::Null) => None,
                Some(v) => Some(screen_position(v).ok_or("position must be [x, y] in points")?),
            };
            let wid = float_window(app, doc, 0);
            if let Some(w) = app.ui.windows.iter_mut().find(|w| w.id == wid) {
                w.position = position;
            }
            app.session.set_active(doc);
            Ok(json!({"window": wid}))
        }
        "floatAllInWindows" | "cascade" => {
            let n = app.session.documents().len();
            let ids: Vec<u64> = (0..n).map(|d| float_window(app, d, d)).collect();
            Ok(json!({"windows": ids}))
        }
        "matchZoom" | "matchLocation" | "matchRotation" | "matchAll" => {
            let i = app.session.active_index().ok_or("no document")?;
            let src = app.ui.views.get(i).cloned().ok_or("no document")?;
            let (zoom, loc, rot) =
                (matches!(k, "matchZoom" | "matchAll"), matches!(k, "matchLocation" | "matchAll"), matches!(k, "matchRotation" | "matchAll"));
            let apply = |v: &mut crate::state::View| {
                if zoom {
                    v.zoom = src.zoom;
                }
                if loc {
                    v.center = src.center;
                }
                if rot {
                    v.rotation = src.rotation;
                }
                v.fit_pending = false;
            };
            app.ui.views.iter_mut().for_each(apply);
            app.ui.windows.iter_mut().for_each(|w| apply(&mut w.view));
            Ok(json!({"zoom": src.zoom, "center": src.center, "rotation": src.rotation}))
        }
        _ => Err(format!("unknown arrangement {k}")),
    }
}

/// Screen rectangles for `n` documents in an arrangement (`None` = tabs). The active document
/// always gets a cell; n-up layouts show that many documents starting from the active one.
pub fn cells(layout: &str, rect: egui::Rect, n: usize) -> Option<Vec<egui::Rect>> {
    use craft_ui::layout::{SplitAxis, SplitSize, split_rect};

    if n < 2 || !LAYOUTS.contains(&layout) {
        return None;
    }
    let grid = |cols: usize, rows: usize, count: usize| -> Vec<egui::Rect> {
        let (w, h) = (rect.width() / cols as f32, rect.height() / rows as f32);
        (0..count.min(cols * rows))
            .map(|i| egui::Rect::from_min_size(rect.min + egui::vec2((i % cols) as f32 * w, (i / cols) as f32 * h), egui::vec2(w, h)))
            .collect()
    };
    Some(match layout {
        "tileAllVertically" => grid(n, 1, n),
        "tileAllHorizontally" => grid(1, n, n),
        "twoUpVertical" => {
            let panes = split_rect(rect, SplitAxis::Horizontal, SplitSize::Ratio(0.5), 0.0, 0.0)?;
            vec![panes.first, panes.second]
        }
        "twoUpHorizontal" => {
            let panes = split_rect(rect, SplitAxis::Vertical, SplitSize::Ratio(0.5), 0.0, 0.0)?;
            vec![panes.first, panes.second]
        }
        "threeUpVertical" => grid(3, 1, 3),
        "threeUpHorizontal" => grid(1, 3, 3),
        "threeUpStacked" => {
            let columns = split_rect(rect, SplitAxis::Horizontal, SplitSize::Ratio(0.5), 0.0, 0.0)?;
            let rows = split_rect(columns.second, SplitAxis::Vertical, SplitSize::Ratio(0.5), 0.0, 0.0)?;
            vec![columns.first, rows.first, rows.second]
        }
        "fourUp" => grid(2, 2, 4),
        "sixUp" => grid(3, 2, 6),
        _ => {
            let cols = (n as f32).sqrt().ceil() as usize;
            grid(cols, n.div_ceil(cols), n)
        }
    })
}

// ---------- dialogs and pickers in front of engine commands ----------

/// A generic form dialog that runs `command` with its fields on OK (rendered by [`form_body`]).
/// File › Open EXR Parts…: pick an EXR, then choose the parts and channel groups that become
/// the layers of one document (a Maya/Arnold render writes one part per AOV). The dialog is
/// a plain command form, so the control channel can drive the same selection headlessly.
fn open_exr_parts_dialog(app: &mut PhotocraftApp) -> Option<Result<Value, String>> {
    Some(app.pick_file_bytes_filtered(&["exr"], |app, name, _bytes| {
        let info = app.run("file.exrParts", json!({"path": name})).map_err(|e| e.to_string())?;
        let entries = info.as_array().cloned().unwrap_or_default();
        let mut fields = Map::new();
        fields.insert("path".into(), json!(name));
        fields.insert("view".into(), json!("both"));
        fields.insert("precomp".into(), json!(false));
        let mut labels = Map::new();
        // The path field shows the picked file's name (data, verbatim); the rest translate.
        if let Some(file) = name.rsplit(['/', '\\']).next() {
            labels.insert("path".into(), json!(file));
        }
        labels.insert("view".into(), json!(crate::i18n::t("View")));
        labels.insert("both".into(), json!(crate::i18n::t("Both")));
        labels.insert("left".into(), json!(crate::i18n::t("Left")));
        labels.insert("right".into(), json!(crate::i18n::t("Right")));
        labels.insert("precomp".into(), json!(crate::i18n::t("Add beauty precomp from the AOVs")));
        let mut any = false;
        for e in &entries {
            let kind = e["kind"].as_str().unwrap_or_default().to_string();
            let ename = e["name"].as_str().unwrap_or_default().to_string();
            let channels = e["channels"].as_u64().unwrap_or(0);
            let key = format!("{kind}/{ename}");
            match kind.as_str() {
                "part" => {
                    labels.insert(key.clone(), json!(format!("{ename} ({}×{}, {channels} ch)", e["width"], e["height"])));
                    let on = !any && channels >= 3;
                    any |= on;
                    fields.insert(key, json!(on));
                }
                "group" => {
                    labels.insert(key.clone(), json!(format!("{ename} ({channels} ch)")));
                    fields.insert(key, json!(false));
                }
                _ => {}
            }
        }
        // Cryptomatte layers are information here (the picker comes separately); the array is
        // not rendered but rides along for the control channel.
        let crypto: Vec<Value> = entries.iter().filter(|e| e["kind"] == "cryptomatte").cloned().collect();
        if !crypto.is_empty() {
            fields.insert("cryptomatte".into(), Value::Array(crypto));
        }
        fields.insert("__labels".into(), Value::Object(labels));
        let choices = json!({"view": ["both", "left", "right"]});
        Ok(json!({"dialog": form(app, "file.openExrParts", "Open EXR Parts…", Value::Object(fields), choices)}))
    }))
}

fn form(app: &mut PhotocraftApp, command: &str, label: &str, fields: Value, choices: Value) -> u64 {
    let mut f = Map::new();
    f.insert("__command".into(), json!(command));
    f.insert("__label".into(), json!(label));
    f.insert("__form".into(), json!(true));
    f.insert("__choices".into(), choices);
    if let Value::Object(m) = fields {
        f.extend(m);
    }
    // Snapshot user-facing params at open so Alt+Reset can restore them.
    let orig: Map<String, Value> = f.iter().filter(|(k, _)| !k.starts_with("__")).map(|(k, v)| (k.clone(), v.clone())).collect();
    f.insert("__formOrig".into(), Value::Object(orig));
    app.ui.open_dialog(DialogKind::Command, f)
}

/// Alt+Cancel (Reset): restore the `__form` params captured at open.
pub fn reset_form(f: &mut Map<String, Value>) {
    if !f.contains_key("__form") {
        return;
    }
    let Some(Value::Object(orig)) = f.get("__formOrig").cloned() else { return };
    f.retain(|k, _| k.starts_with("__"));
    f.extend(orig);
}

fn label_of(key: &str) -> String {
    // A name that already reads as text ("iPhone 14 Pro Max", "Web 1920") is not camelCase.
    if key.contains(' ') {
        return tl!(key).to_owned();
    }
    let mut out = String::new();
    for (i, c) in key.chars().enumerate() {
        if i == 0 {
            out.extend(c.to_uppercase());
        } else if c.is_uppercase() {
            out.push(' ');
            out.extend(c.to_lowercase());
        } else {
            out.push(c);
        }
    }
    let translated = tl!(&out);
    if translated != out {
        return translated.to_owned();
    }
    // Fall back to the Title Case filter label's translation, but keep the sentence-case
    // English when that is untranslated too (English and partial catalogs read as before).
    let title = crate::filter_dialog::label(key);
    if title == crate::filter_dialog::source_label(key) { out } else { title }
}

#[cfg(test)]
mod label_tests {
    use super::label_of;

    #[test]
    fn untranslated_form_labels_stay_sentence_case() {
        crate::i18n::with_language(crate::i18n::Lang::EN, || {
            assert_eq!(label_of("useAntialias"), "Use antialias");
            assert_eq!(label_of("radius"), "Radius");
            assert_eq!(label_of("iPhone 14 Pro Max"), "iPhone 14 Pro Max");
        });
    }
}

#[cfg(test)]
mod form_reset_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn alt_reset_restores_form_orig() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let id = form(&mut app, "test.form", "Test", json!({"radius": 5.0, "on": true}), json!({}));
        let f = &mut app.ui.dialog_mut(id).unwrap().fields;
        f.insert("radius".into(), json!(99.0));
        f.insert("on".into(), json!(false));
        reset_form(f);
        assert_eq!(f["radius"], json!(5.0));
        assert_eq!(f["on"], json!(true));
        assert!(f.contains_key("__formOrig"));
    }
}

/// Body of a `__form` dialog: text fields, number fields, checkboxes and `__choices` dropdowns.
pub fn form_body(ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    let choices = f.get("__choices").cloned().unwrap_or(Value::Null);
    // Data-driven labels (a per-part checkbox shows the part's name, not a catalogue entry)
    // come from `__labels` verbatim; everything else goes through the catalogue.
    let labels = f.get("__labels").cloned().unwrap_or(Value::Null);
    let label = |k: &str| labels.get(k).and_then(Value::as_str).map_or_else(|| label_of(k), str::to_string);
    let keys: Vec<String> = f.keys().filter(|k| !k.starts_with("__")).cloned().collect();
    egui::Grid::new("form-dialog").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
        for k in keys {
            let v = f.get(&k).cloned().unwrap_or(Value::Null);
            if k == "naming" && f.get("__command").and_then(Value::as_str) == Some("file.automate.batch") {
                let label = ui.label(tl!("File Naming"));
                let mut naming = v.as_str().unwrap_or_default().to_owned();
                ui.add(egui::TextEdit::singleline(&mut naming).desired_width(260.0).hint_text("{name}"))
                    .labelled_by(label.id)
                    .on_hover_text("{name} · {serial} · {date} (UTC)");
                f.insert(k, json!(naming));
                ui.end_row();
                continue;
            }
            if k == "quality" && f.get("__command").and_then(Value::as_str) == Some("file.automate.batch") {
                ui.label(label(&k));
                let mut custom = v.as_f64().is_some();
                let mut quality = v.as_f64().or_else(|| f.get("__batchQuality").and_then(Value::as_f64)).unwrap_or(8.0).clamp(0.0, 12.0) as f32;
                ui.horizontal(|ui| {
                    crate::widgets::checkbox(ui, &mut custom, tl!("Custom"));
                    ui.add_enabled_ui(custom, |ui| crate::widgets::value_field(ui, &mut quality, 0.0..=12.0, "", 74.0));
                });
                // None preserves the command's existing JPEG and lossless WebP defaults.
                // The inactive custom value is shell state, never a command parameter.
                f.insert("__batchQuality".into(), json!(quality));
                f.insert(k, if custom { json!(quality) } else { Value::Null });
                ui.end_row();
                continue;
            }
            let new = match &v {
                Value::Array(_) | Value::Object(_) | Value::Null => continue,
                Value::Bool(b) => {
                    ui.label("");
                    let mut b = *b;
                    ui.checkbox(&mut b, label(&k));
                    json!(b)
                }
                Value::String(s) => {
                    ui.label(label(&k));
                    let mut s = s.clone();
                    match choices.get(&k).and_then(Value::as_array) {
                        Some(opts) => {
                            let opts: Vec<(String, String)> = opts.iter().filter_map(Value::as_str).map(|o| (o.to_string(), label(o))).collect();
                            let refs: Vec<(String, &str)> = opts.iter().map(|(a, b)| (a.clone(), b.as_str())).collect();
                            crate::widgets::dropdown(ui, &format!("form-{k}"), &mut s, &refs, 200.0);
                        }
                        None => {
                            ui.add(egui::TextEdit::singleline(&mut s).desired_width(260.0));
                        }
                    }
                    json!(s)
                }
                Value::Number(n) => {
                    ui.label(label(&k));
                    if let Some(mut i) = n.as_i64() {
                        ui.add(egui::DragValue::new(&mut i).custom_parser(crate::widgets::parse_num));
                        json!(i)
                    } else {
                        let mut x = n.as_f64().unwrap_or(0.0);
                        ui.add(egui::DragValue::new(&mut x).speed(0.5).custom_parser(crate::widgets::parse_num));
                        json!(x)
                    }
                }
            };
            f.insert(k, new);
            ui.end_row();
        }
    });
    camera_data(ui, f);
}

/// File Info's read-only Camera Data section (`__camera`, from the document's EXIF/XMP).
fn camera_data(ui: &mut egui::Ui, f: &Map<String, Value>) {
    let Some(camera) = f.get("__camera").and_then(Value::as_object).filter(|c| !c.is_empty()) else { return };
    ui.add_space(10.0);
    ui.label(egui::RichText::new(tl!("Camera Data")).font(crate::theme::semibold(13.0)));
    ui.add_space(4.0);
    egui::Grid::new("form-camera-data").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
        for (key, label) in photocraft_engine::file_cmds::CAMERA_FIELDS {
            if let Some(v) = camera.get(key).and_then(Value::as_str) {
                ui.label(crate::i18n::tr_ctx(crate::i18n::current(), "cameraData", label));
                ui.add(egui::Label::new(v).selectable(true));
                ui.end_row();
            }
        }
    });
}

/// Folder the batch dialogs start from: next to the active document, else the working directory.
fn default_dir(app: &PhotocraftApp) -> String {
    app.session.active().and_then(|d| d.path.as_deref()).and_then(|p| p.rfind(['/', '\\']).map(|i| p[..i].to_string())).unwrap_or_else(|| ".".into())
}

fn front(app: &mut PhotocraftApp, id: &str, params: &Value) -> Option<Result<Value, String>> {
    if id == "file.closeAll" || id == "file.closeOthers" {
        // Keep the remaining document's view (views are index-aligned with documents).
        let keep = app.session.active_index().and_then(|i| app.ui.views.get(i).cloned());
        let r = app.run(id, params.clone());
        if r.is_ok() {
            app.ui.views = if id == "file.closeAll" { Vec::new() } else { keep.into_iter().collect() };
            app.ui.windows.clear();
            app.sync_views();
        }
        return Some(r);
    }
    if id == "type.pasteLoremIpsum" && no_params(params) {
        // While typing, paste at the caret.
        if let Some(te) = app.ui.text_edit.clone() {
            let r = app.run(id, json!({"layer": te.layer, "at": te.caret, "coalesce": te.session}));
            if let (Ok(v), Some(te)) = (&r, app.ui.text_edit.as_mut())
                && let Some(c) = v.get("caret").and_then(Value::as_u64)
            {
                te.caret = c as usize;
                te.anchor = c as usize;
            }
            return Some(r);
        }
        return None;
    }
    if !no_params(params) {
        return None;
    }
    let doc = app.session.active().map(|d| (d.doc.size.width, d.doc.size.height, d.doc.name.clone()));
    let dir = default_dir(app);
    let label = photocraft_engine::commands::find(id).map_or(id, |c| c.label);
    let dialog = |app: &mut PhotocraftApp, fields: Value, choices: Value| Some(Ok(json!({"dialog": form(app, id, label, fields, choices)})));
    match id {
        "file.openAs" => Some(app.open_dialog_file()),
        "file.openExrParts" => open_exr_parts_dialog(app),
        "file.saveACopy" => Some(save_a_copy(app)),
        "file.placeEmbedded" | "file.placeLinked" => {
            let doc = match app.active_doc_id() {
                Ok(doc) => doc,
                Err(e) => return Some(Err(e)),
            };
            let linked = id == "file.placeLinked";
            Some(app.pick_file_bytes(move |app, name, bytes| {
                app.refocus(doc)?;
                let linked = linked.then(|| name.clone());
                app.place_bytes(&name, bytes, linked)
            }))
        }
        "layer.smartObjects.replaceContents" => {
            let st = app.session.active()?;
            let (doc, layer) = (st.doc.id, st.active_layer?);
            Some(app.pick_file_bytes(move |app, name, bytes| {
                app.with_document(doc, |app| {
                    let r = photocraft_engine::smart_cmds::replace_contents_bytes(&mut app.session, layer, &name, bytes).map_err(|e| e.to_string());
                    app.sync_views();
                    r
                })
            }))
        }
        "layer.smartObjects.exportContents" => Some(export_contents(app)),
        "layer.smartObjects.convertToLinked" => {
            let st = app.session.active()?;
            let (doc, layer) = (st.doc.id, st.active_layer?);
            let (name, _) = match photocraft_engine::smart_cmds::contents_of(&st.doc, layer) {
                Ok(c) => c,
                Err(e) => return Some(Err(e.to_string())),
            };
            Some(app.pick_save(&name, move |app, path| {
                app.with_document(doc, |app| app.run("layer.smartObjects.convertToLinked", json!({"layer": layer.0, "path": path})))
            }))
        }
        "file.fileInfo" => {
            let info = app.session.execute("file.fileInfo", json!({})).ok()?;
            let kw = info["keywords"].as_array().map(|a| a.iter().filter_map(Value::as_str).collect::<Vec<_>>().join("; ")).unwrap_or_default();
            dialog(
                app,
                json!({"title": info["title"], "author": info["author"], "authorTitle": info["authorTitle"], "description": info["description"], "keywords": kw, "copyright": info["copyright"], "copyrightStatus": info["copyrightStatus"], "copyrightUrl": info["copyrightUrl"], "__camera": info["camera"]}),
                json!({"copyrightStatus": ["unknown", "copyrighted", "publicDomain"]}),
            )
        }
        "file.automate.fitImage" => {
            let (w, h, _) = doc?;
            dialog(app, json!({"width": w, "height": h, "dontEnlarge": false, "__compact": true}), json!({}))
        }
        "file.automate.conditionalModeChange" => dialog(
            app,
            json!({"from": "any", "to": "rgb"}),
            json!({"from": ["any", "rgb", "grayscale", "cmyk", "lab", "indexed", "bitmap", "duotone", "multichannel"], "to": ["rgb", "grayscale", "cmyk", "lab"]}),
        ),
        "view.newGuideLayout" => Some(Ok(json!({"dialog": crate::guide_layout_ui::open(app)}))),
        "type.warpText" => Some(Ok(json!({"dialog": crate::dialogs::open_command_dialog(app, id, label)}))),
        "file.export.layersToFiles" => {
            let (_, _, name) = doc?;
            dialog(
                app,
                json!({"dir": dir, "prefix": name.rsplit_once('.').map_or(name.as_str(), |(a, _)| a), "format": "png", "visibleOnly": true}),
                json!({"format": ["png", "jpg", "psd", "tiff", "webp", "bmp"]}),
            )
        }
        "file.export.layerCompsToFiles" | "file.export.artboardsToFiles" => {
            let (_, _, name) = doc?;
            let prefix = name.rsplit_once('.').map_or(name.as_str(), |(a, _)| a).to_string();
            let fields = if id == "file.export.layerCompsToFiles" {
                json!({"dir": dir, "prefix": prefix, "format": "png", "selectedOnly": false})
            } else {
                json!({"dir": dir, "prefix": prefix, "format": "png"})
            };
            dialog(app, fields, json!({"format": ["png", "jpg", "psd", "tiff", "webp", "bmp"]}))
        }
        "file.export.artboardsToPdf" => {
            let (_, _, name) = doc?;
            let stem = name.rsplit_once('.').map_or(name.as_str(), |(a, _)| a).to_string();
            dialog(app, json!({"path": format!("{dir}/{stem}.pdf"), "quality": 10}), json!({}))
        }
        "layer.new.artboard" => {
            let (w, h, _) = doc?;
            let presets: Vec<&str> = std::iter::once("").chain(photocraft_engine::artboard_cmds::PRESETS.iter().map(|p| p.0)).collect();
            dialog(
                app,
                json!({"name": "", "preset": "", "width": w, "height": h, "background": "white"}),
                json!({"preset": presets, "background": ["white", "black", "transparent"]}),
            )
        }
        "file.export.colorLookupTables" => {
            let (_, _, name) = doc?;
            let stem = name.rsplit_once('.').map_or(name.as_str(), |(a, _)| a).to_string();
            // A selected adjustment is a common use case, but existing no-selection exports
            // continue to bake the entire visible stack. The scope remains explicit in the UI.
            let selected_adjustments = app.session.active().is_some_and(|st| {
                let chosen = st.selected_layers();
                !chosen.is_empty()
                    && chosen.iter().all(|id| {
                        st.doc.layers.iter().any(|root| root.id == *id)
                            && st.doc.layer(*id).is_some_and(|layer| layer.visible && matches!(layer.content, photocraft_doc::LayerContent::Adjustment(_)))
                    })
            });
            let scope = if selected_adjustments { "selected" } else { "all" };
            dialog(app, json!({"path": format!("{dir}/{stem}.cube"), "size": 33, "title": stem, "scope": scope}), json!({"scope": ["all", "selected"]}))
        }
        "file.scripts.loadFilesIntoStack" => Some(Ok(json!({"dialog": crate::stack_dialog::open(app)}))),
        // Photography automation (photo_cmds / lens_cmds): a folder (or the open documents).
        "file.automate.photomerge" => dialog(
            app,
            json!({"paths": dir, "useOpenDocuments": false, "layout": "auto", "blend": true, "vignetteRemoval": false, "geometricCorrection": false, "contentAwareFill": false}),
            json!({"layout": ["auto", "perspective", "cylindrical", "spherical", "collage", "reposition"]}),
        ),
        "file.automate.mergeToHdrPro" => dialog(
            app,
            json!({"paths": dir, "useOpenDocuments": false, "align": true, "removeGhosts": false, "mode": "32", "method": "localAdaptation", "radius": 7.0, "strength": 0.52, "gamma": 1.0, "exposure": 0.0, "detail": 30.0, "saturation": 20.0}),
            json!({"mode": ["32", "16", "8"], "method": ["localAdaptation", "exposureGamma", "highlightCompression", "equalizeHistogram"]}),
        ),
        "file.automate.lensCorrection" => dialog(
            app,
            json!({"input": dir, "output": format!("{dir}/corrected"), "format": "same", "profile": "auto", "correctDistortion": true, "correctVignette": true, "correctCA": true, "autoScale": true, "edge": "transparency"}),
            json!({"format": ["same", "png", "jpg", "psd", "tiff"], "profile": ["auto", "generic", "none"], "edge": ["transparency", "edgeExtension", "black", "white"]}),
        ),
        "file.scripts.imageProcessor" => dialog(
            app,
            json!({"input": dir, "output": format!("{dir}/processed"), "format": "jpg", "quality": 8, "width": 0, "height": 0, "convertToSrgb": true}),
            json!({"format": ["jpg", "png", "psd", "tiff", "webp"]}),
        ),
        "file.automate.batch" => {
            let Some(action) = crate::actions::selected_action(app) else {
                return Some(Err("record an action in the Actions panel first".into()));
            };
            let steps = crate::actions::action_steps(action);
            let name = action.name.clone();
            dialog(
                app,
                json!({"action": name, "steps": steps, "input": dir, "output": format!("{dir}/batch"), "format": "same", "quality": null, "__batchQuality": 8, "naming": "", "startingSerial": 1}),
                json!({"format": ["same", "png", "jpg", "psd", "tiff", "webp"]}),
            )
        }
        // Channel workflow dialogs (see channels_panel.rs for the channel references).
        "select.saveSelection" | "select.loadSelection" | "image.applyImage" | "image.calculations" => {
            let st = app.session.active()?;
            let d = &st.doc;
            let mode = d.pixel_format().mode;
            let mut chans: Vec<String> = vec![photocraft_engine::channel_cmds::composite_name(mode).to_string()];
            if mode.color_channels() > 1 {
                chans.extend(photocraft_engine::channel_cmds::color_names(mode).iter().map(|n| n.to_string()));
            }
            chans.extend(d.channels.iter().map(|c| c.name.clone()));
            let alphas: Vec<String> = std::iter::once("new".to_string()).chain(d.channels.iter().map(|c| c.name.clone())).collect();
            let blends = [
                "multiply",
                "screen",
                "normal",
                "overlay",
                "softLight",
                "hardLight",
                "darken",
                "lighten",
                "colorDodge",
                "colorBurn",
                "linearBurn",
                "linearDodge",
                "difference",
                "exclusion",
                "add",
                "subtract",
            ];
            // The active layer's transparency, and its mask only when it has one.
            let layer = st.active_layer.and_then(|l| d.layer(l));
            let mut load = chans.clone();
            if layer.is_some() {
                load.push("transparency".into());
            }
            if layer.is_some_and(|l| l.mask.is_some()) {
                load.push("mask".into());
            }
            load.push("selection".into());
            match id {
                "select.saveSelection" => dialog(
                    app,
                    json!({"channel": "new", "name": "", "operation": "new"}),
                    json!({"channel": alphas, "operation": ["new", "replace", "add", "subtract", "intersect"]}),
                ),
                "select.loadSelection" => {
                    let first = d.channels.first().map_or_else(|| chans[0].clone(), |c| c.name.clone());
                    dialog(
                        app,
                        json!({"channel": first, "invert": false, "operation": "new"}),
                        json!({"channel": load, "operation": ["new", "add", "subtract", "intersect"]}),
                    )
                }
                "image.applyImage" => {
                    let mut masks = vec!["none".to_string()];
                    masks.extend(load.iter().cloned());
                    // Source layer: Merged, then the pixel layers top to bottom (the engine takes
                    // the topmost layer of a name), shown by their own names.
                    let mut layers = vec!["merged".to_string()];
                    for (_, _, l) in d.walk().into_iter().rev() {
                        if l.surface().is_some() && !layers.contains(&l.name) {
                            layers.push(l.name.clone());
                        }
                    }
                    let names: Map<String, Value> = layers.iter().skip(1).map(|n| (n.clone(), json!(n))).collect();
                    // Scale and Offset only affect Add and Subtract.
                    dialog(
                        app,
                        json!({"sourceLayer": "merged", "sourceChannel": chans[0], "sourceInvert": false, "blending": "multiply", "opacity": 100.0, "scale": 1.0, "offset": 0, "preserveTransparency": false, "maskChannel": "none", "__labels": names}),
                        json!({"sourceLayer": layers, "sourceChannel": load, "blending": blends, "maskChannel": masks}),
                    )
                }
                _ => dialog(
                    app,
                    json!({"source1Channel": chans[0], "source1Invert": false, "source2Channel": chans[0], "source2Invert": false, "blending": "multiply", "opacity": 100.0, "result": "newChannel"}),
                    json!({"source1Channel": load.clone(), "source2Channel": load, "blending": blends, "result": ["newChannel", "newDocument", "selection"]}),
                ),
            }
        }
        _ => None,
    }
}

/// File › Save a Copy: pick a name, encode with the export service, write; the document's path
/// and saved state are untouched. An action being recorded keeps it as a `file.saveACopy` step
/// with its path (#4373).
fn save_a_copy(app: &mut PhotocraftApp) -> Result<Value, String> {
    let doc = app.session.active().ok_or("no document")?.doc.clone();
    let stem = doc.name.rsplit_once('.').map_or(doc.name.as_str(), |(a, _)| a);
    let suggested = format!("{stem} copy.psd");
    // The copy is of the document as it was when asked.
    app.pick_save(&suggested, move |app, path| {
        // An RGB Targa asks for its bits per pixel first; the copy is written from the prompt.
        let parked = app.with_document(doc.id, |app| {
            let ask = crate::targa_options_ui::wants_prompt(app, &path);
            if ask {
                crate::targa_options_ui::park(app, path.clone(), true, "file.saveACopy")?;
            }
            Ok(ask)
        })?;
        if parked {
            return Ok(json!({"path": path, "warnings": []}));
        }
        let r = write_copy(app, &doc, &path, None)?;
        app.session.journal.push(("file.saveACopy".into(), json!({"path": path})));
        Ok(r)
    })
}

/// Writes `doc` to `path` as a copy (Save a Copy, and its replayed action step): the document's
/// path and saved state are untouched. `tga_bits` is the Targa answer, when there is one.
pub(crate) fn write_copy(
    app: &mut PhotocraftApp,
    doc: &photocraft_doc::Document,
    path: &str,
    tga_bits: Option<photocraft_io::TgaBits>,
) -> Result<Value, String> {
    let settings = crate::ExportSettings { tga_bits, ..app.save_settings() };
    let export = app.services.export.as_ref().ok_or("no exporter configured")?;
    let (bytes, warnings) = export(doc, path, &settings)?;
    let write = app.services.write.as_mut().ok_or("no writer configured")?;
    write(path, &bytes)?;
    let template = "Saved a copy as {path}";
    app.ui.status = crate::i18n::fmt(tl!(template), &[("path", path)]);
    app.ui.status_error = false;
    let what = crate::i18n::fmt(tl!(template), &[("path", &crate::file_open::display_name(path))]);
    crate::notices::io_warnings(app, &what, &warnings);
    Ok(json!({"path": path, "warnings": warnings}))
}

/// Layer › Smart Objects › Export Contents: saves the contents file under its own name by default.
fn export_contents(app: &mut PhotocraftApp) -> Result<Value, String> {
    let st = app.session.active().ok_or("no document")?;
    let layer = st.active_layer.ok_or("no layer selected")?;
    // The contents as they were when asked, like Save a Copy.
    let (name, bytes) = photocraft_engine::smart_cmds::contents_of(&st.doc, layer).map_err(|e| e.to_string())?;
    app.pick_save(&name.clone(), move |app, path| {
        let write = app.services.write.as_mut().ok_or("no writer configured")?;
        write(&path, &bytes)?;
        let template = "Exported contents to {path}";
        app.ui.status = crate::i18n::fmt(tl!(template), &[("path", &path)]);
        app.ui.status_error = false;
        Ok(json!({"path": path, "fileName": name, "bytes": bytes.len()}))
    })
}

#[cfg(test)]
#[path = "view_cmds/tests.rs"]
mod tests;
