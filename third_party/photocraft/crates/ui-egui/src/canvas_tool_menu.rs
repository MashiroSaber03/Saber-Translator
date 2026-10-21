//! Canvas context menus for the selection tools, the Pen, the Zoom and Hand tools, an active Free
//! Transform box, text being edited with a Type tool and the pasteboard (outside the image).

use egui::Context;
use photocraft_doc::LayerContent;
use photocraft_engine::prefs::{self, CanvasColor};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{PhotocraftApp, state::Tool};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CanvasToolMenu {
    pub pos: [f32; 2],
    pub tool: Tool,
    pub has_selection: bool,
    #[serde(default)]
    pub has_path: bool,
    #[serde(default)]
    pub path_name: Option<String>,
    /// Opened over a Free Transform box: its modes instead of the tool's actions.
    #[serde(default)]
    pub transform: bool,
    /// Opened outside the image: the pasteboard colour instead of the tool's actions.
    #[serde(default)]
    pub pasteboard: bool,
    /// The Zoom and Hand tools' view menu: the document point right-clicked, which Zoom In and
    /// Zoom Out keep under the pointer.
    #[serde(default)]
    pub anchor: Option<[f64; 2]>,
    /// Opened while a Type tool edits text: the text and type rows instead of the tool's actions.
    #[serde(default)]
    pub text: bool,
    /// The misspelled word under the pointer when the type edit menu opened: its suggestions head
    /// the menu.
    #[serde(default)]
    pub spelling: Option<Spelling>,
}

/// A misspelled word in the type layer being edited (`edit.checkSpelling`'s `list` entry).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Spelling {
    pub layer: u64,
    /// The word's characters `start..end` in the layer's text.
    pub start: usize,
    pub end: usize,
    pub word: String,
    pub suggestions: Vec<String>,
}

/// Most suggestions the type edit menu lists.
const MAX_SUGGESTIONS: usize = 3;
/// The spelling rows' ids: `spelling.replace.<n>` picks the n-th suggestion.
const SPELLING_REPLACE: &str = "spelling.replace.";
const SPELLING_NONE: &str = "spelling.none";
const SPELLING_ADD: &str = "spelling.add";

/// One menu row: label and command id. `None` is a separator.
pub type Row = Option<(&'static str, &'static str)>;

/// Right-click while transforming, in Photoshop's order: what the box's handles do, then Rotate
/// and Flip, which turn the open box without committing it (`transform_tool::intercept`).
/// Content-Aware Scale and Puppet Warp are greyed while a box is open.
pub const TRANSFORM_MENU: &[Row] = &[
    Some(("Free Transform", "edit.freeTransform")),
    Some(("Scale", "edit.transform.scale")),
    Some(("Rotate", "edit.transform.rotate")),
    Some(("Skew", "edit.transform.skew")),
    Some(("Distort", "edit.transform.distort")),
    Some(("Perspective", "edit.transform.perspective")),
    Some(("Warp", "edit.transform.warp")),
    None,
    Some(("Content-Aware Scale", "edit.contentAwareScale")),
    Some(("Puppet Warp", "edit.puppetWarp")),
    None,
    Some(("Rotate 180°", "edit.transform.rotate180")),
    Some(("Rotate 90° Clockwise", "edit.transform.rotate90Cw")),
    Some(("Rotate 90° Counter Clockwise", "edit.transform.rotate90Ccw")),
    None,
    Some(("Flip Horizontal", "edit.transform.flipHorizontal")),
    Some(("Flip Vertical", "edit.transform.flipVertical")),
];

/// Photoshop's pasteboard menu (right-click outside the image): the pasteboard colour, the choice
/// Preferences › Interface also offers. The ids are this menu's own: [`choose`] turns them into
/// `prefs.set` (`pasteboard.<CanvasColor name>`) or opens the Color Picker.
pub const PASTEBOARD_MENU: &[Row] = &[
    Some(("Default", "pasteboard.default")),
    Some(("Black", "pasteboard.black")),
    Some(("Dark Gray", "pasteboard.darkGray")),
    Some(("Medium Gray", "pasteboard.mediumGray")),
    Some(("Light Gray", "pasteboard.lightGray")),
    Some(("Custom", "pasteboard.custom")),
    None,
    Some(("Select Custom Color…", PASTEBOARD_PICK)),
];

/// Photoshop's right-click menu inside text being edited: the edit rows act on the text (its
/// selection, the OS clipboard), the type rows on the layer, as the Type menu's do. Faux Bold and
/// Faux Italic are the Character panel's toggles (this menu's own ids, run as `type.setStyle`).
/// Convert to Paragraph Text and Convert to Point Text are both listed; the one that doesn't apply
/// greys out. The exact Photoshop placement is inferred (docs/context-menu-parity.md).
pub const TYPE_EDIT_MENU: &[Row] = &[
    Some(("Undo", "edit.undo")),
    None,
    Some(("Cut", "edit.cut")),
    Some(("Copy", "edit.copy")),
    Some(("Paste", "edit.paste")),
    Some(("Select All", "select.all")),
    None,
    Some(("Check Spelling…", "edit.checkSpelling")),
    Some(("Find and Replace Text…", "edit.findAndReplaceText")),
    None,
    Some(("Faux Bold", FAUX_BOLD)),
    Some(("Faux Italic", FAUX_ITALIC)),
    None,
    Some(("Rasterize Type", "type.rasterizeTypeLayer")),
    Some(("Create Work Path", "type.createWorkPath")),
    Some(("Convert to Shape", "type.convertToShape")),
    None,
    Some(("Horizontal", "type.orientation.horizontal")),
    Some(("Vertical", "type.orientation.vertical")),
    None,
    Some(("None", "type.antiAlias.none")),
    Some(("Sharp", "type.antiAlias.sharp")),
    Some(("Crisp", "type.antiAlias.crisp")),
    Some(("Strong", "type.antiAlias.strong")),
    Some(("Smooth", "type.antiAlias.smooth")),
    Some(("Windows LCD", "type.antiAlias.windowsLcd")),
    Some(("Windows", "type.antiAlias.windows")),
    None,
    Some(("Convert to Paragraph Text", "type.convertToParagraphText")),
    Some(("Convert to Point Text", "type.convertToPointText")),
    Some(("Warp Text…", "type.warpText")),
];

/// The type edit menu's Faux Bold and Faux Italic rows.
pub const FAUX_BOLD: &str = "type.fauxBold";
pub const FAUX_ITALIC: &str = "type.fauxItalic";

/// Type edit rows that act on the whole layer (it may stop being type): the edit ends first.
const ENDS_TYPE_EDIT: [&str; 5] =
    ["type.rasterizeTypeLayer", "type.createWorkPath", "type.convertToShape", "type.convertToParagraphText", "type.convertToPointText"];

/// The pasteboard menu's Select Custom Color… row.
pub const PASTEBOARD_PICK: &str = "pasteboard.selectCustomColor";

/// The pasteboard colours in the menu's order, which Space+F steps through.
const CANVAS_COLORS: [CanvasColor; 6] =
    [CanvasColor::Default, CanvasColor::Black, CanvasColor::DarkGray, CanvasColor::MediumGray, CanvasColor::LightGray, CanvasColor::Custom];

/// Photoshop's Space+F (⇧Space+F backwards): the next pasteboard colour, round and round. Saved
/// like a choice in the pasteboard menu (`interface.canvasColor`).
pub fn cycle_canvas_color(app: &mut PhotocraftApp, backward: bool) -> Result<(), String> {
    let n = CANVAS_COLORS.len();
    let at = CANVAS_COLORS.iter().position(|c| *c == app.session.prefs().interface.canvas_color).unwrap_or(0);
    let next = if backward { (at + n - 1) % n } else { (at + 1) % n };
    let name = CANVAS_COLORS.get(next).map_or("default", |c| c.name());
    app.run("prefs.set", json!({"values": {"interface.canvasColor": name}})).map(|_| ())
}

/// The Zoom and Hand tools' context menu: the zoom steps and fixed zooms of the View menu, plus
/// the Hand tool's Fill Screen.
pub const VIEW_MENU: &[Row] = &[
    Some(("Zoom In", "view.zoomIn")),
    Some(("Zoom Out", "view.zoomOut")),
    None,
    Some(("Fit on Screen", "view.fitOnScreen")),
    Some(("Fill Screen", FILL_SCREEN)),
    None,
    Some(("100%", "view.actualPixels")),
    Some(("200%", "view.twoHundredPercent")),
    Some(("Print Size", "view.printSize")),
];

/// The view menu's Fill Screen row. Photoshop has no View menu command for it, so the menu sets
/// the view itself.
pub const FILL_SCREEN: &str = "view.fillScreen";

/// Photoshop's selection-tool context menu with an active selection, in its order. Its Generative
/// Fill row has no PhotoCraft command and is left out. Rows stay visible and grey out exactly like
/// their menu-bar twins.
pub const SELECTION_MENU: &[Row] = &[
    Some(("Deselect", "select.deselect")),
    Some(("Select Inverse", "select.inverse")),
    Some(("Feather…", "select.modify.feather")),
    Some(("Select and Mask…", "select.selectAndMask")),
    None,
    Some(("Save Selection…", "select.saveSelection")),
    Some(("Make Work Path…", "select.toWorkPath")),
    None,
    Some(("Layer via Copy", "layer.new.layerViaCopy")),
    Some(("Layer via Cut", "layer.new.layerViaCut")),
    Some(("New Layer…", "layer.new.layer")),
    None,
    Some(("Free Transform", "edit.freeTransform")),
    Some(("Transform Selection", "select.transformSelection")),
    Some(("Distort", "edit.transform.distort")),
    Some(("Perspective", "edit.transform.perspective")),
    None,
    Some(("Delete and Fill Selection", "edit.deleteAndFillSelection")),
    Some(("Fill…", "edit.fill")),
    Some(("Stroke…", "edit.stroke")),
    Some(("Content-Aware Fill…", "edit.contentAwareFill")),
    None,
    Some(("Last Filter", "filter.lastFilter")),
    None,
    Some(("Fade…", "edit.fade")),
];

/// The selection tools' context menu with nothing selected: making a selection.
pub const NO_SELECTION_MENU: &[Row] = &[
    Some(("Select All", "select.all")),
    Some(("Reselect", "select.reselect")),
    Some(("Color Range…", "select.colorRange")),
    None,
    Some(("Load Selection…", "select.loadSelection")),
];

/// Photoshop's Pen context menu order. Rows whose operation is not applicable to the current
/// path or layer stay visible and disabled.
pub const PEN_MENU: &[Row] = &[
    Some(("Create Vector Mask", "layer.vectorMask.fromPath")),
    Some(("Delete Path", "path.delete")),
    None,
    Some(("Define Custom Shape…", "edit.defineCustomShape")),
    None,
    Some(("Make Selection…", "path.toSelection")),
    Some(("New Guides From Shape", "view.newGuidesFromShape")),
    Some(("Fill Path…", "path.fill")),
    Some(("Stroke Path…", "path.stroke")),
    None,
    Some(("Clipping Path…", "path.clippingPath.set")),
    None,
    Some(("Free Transform Path", "path.transform")),
    None,
    Some(("Unite Shapes", "layer.combineShapes.unite")),
    Some(("Subtract Front Shape", "layer.combineShapes.subtractFrontShape")),
    Some(("Unite Shapes at Overlap", "layer.combineShapes.intersectShapeAreas")),
    Some(("Subtract Shapes at Overlap", "layer.combineShapes.excludeOverlappingShapes")),
    None,
    Some(("Copy Fill", "path.style.copyFill")),
    Some(("Copy Complete Stroke", "path.style.copyStroke")),
    None,
    Some(("Paste Fill", "path.style.pasteFill")),
    Some(("Paste Complete Stroke", "path.style.pasteStroke")),
    None,
    Some(("Isolate Layers", "select.isolateLayers")),
    None,
    Some(("Make Symmetry Path", "paint.symmetryFromPath")),
    Some(("Disable Symmetry Path", "paint.symmetryDisable")),
];

/// Tools whose plain canvas right-click offers selection actions.
pub fn applies(tool: Tool) -> bool {
    matches!(
        tool,
        Tool::RectMarquee
            | Tool::EllipseMarquee
            | Tool::SingleRowMarquee
            | Tool::SingleColumnMarquee
            | Tool::Lasso
            | Tool::PolygonLasso
            | Tool::MagneticLasso
            | Tool::MagicWand
            | Tool::ObjectSelection
            | Tool::Pen
    )
}

/// Tools whose plain canvas right-click offers the view menu.
pub fn is_view_tool(tool: Tool) -> bool {
    matches!(tool, Tool::Zoom | Tool::Hand)
}

/// The selection tools' rows, with or without an active selection.
pub fn selection_rows(has_selection: bool) -> &'static [Row] {
    if has_selection { SELECTION_MENU } else { NO_SELECTION_MENU }
}

/// The rows of an open menu.
pub fn rows(menu: &CanvasToolMenu) -> &'static [Row] {
    if menu.pasteboard {
        PASTEBOARD_MENU
    } else if menu.transform {
        TRANSFORM_MENU
    } else if menu.anchor.is_some() {
        VIEW_MENU
    } else if menu.text {
        TYPE_EDIT_MENU
    } else if menu.tool == Tool::Pen {
        PEN_MENU
    } else {
        selection_rows(menu.has_selection)
    }
}

/// Is `command` one of the open menu's rows, and enabled?
pub fn available(app: &PhotocraftApp, menu: &CanvasToolMenu, command: &str) -> bool {
    (rows(menu).iter().flatten().any(|(_, id)| *id == command) || is_spelling(menu, command)) && entry_enabled(app, menu, command)
}

/// Is `command` the transform box's current mode (checked in the transform menu)?
/// Scale stays Free Transform's mode (only Free Transform is checked). Rotate is its own mode.
/// Warp is not a box mode (it is its own session), and `for_command` would read it as Free.
pub(crate) fn mode_checked(app: &PhotocraftApp, menu: &CanvasToolMenu, command: &str) -> bool {
    if menu.pasteboard {
        return pasteboard_choice(command) == Some(app.session.prefs().interface.canvas_color);
    }
    if menu.text {
        return type_checked(app, command);
    }
    menu.transform
        && matches!(command, "edit.freeTransform" | "edit.transform.rotate" | "edit.transform.skew" | "edit.transform.distort" | "edit.transform.perspective")
        && app.ui.transform.as_ref().is_some_and(|t| t.mode == crate::state::TransformMode::for_command(command))
}

/// The type edit menu's checkmarks: the faux styles at the selection, the layer's orientation and
/// anti-aliasing.
fn type_checked(app: &PhotocraftApp, command: &str) -> bool {
    use photocraft_doc::text::Orientation;
    let Some(t) = crate::type_tool::target_text(app) else { return false };
    match command {
        FAUX_BOLD | FAUX_ITALIC => crate::type_tool::styles_at(app).is_some_and(|(c, _)| if command == FAUX_BOLD { c.faux_bold } else { c.faux_italic }),
        "type.orientation.horizontal" => t.orientation == Orientation::Horizontal,
        "type.orientation.vertical" => t.orientation == Orientation::Vertical,
        c => c.strip_prefix("type.antiAlias.") == Some(photocraft_engine::type_extra_cmds::aa_name(t.antialias)),
    }
}

/// Is the Type tool editing text, so that a right-click opens the type edit menu?
pub fn editing_text(app: &PhotocraftApp, tool: Tool) -> bool {
    tool.is_type() && app.ui.text_edit.is_some()
}

/// The misspelled word of the edited text under document point `doc`, with its suggestions.
fn spelling_at(app: &mut PhotocraftApp, doc: [f64; 2]) -> Option<Spelling> {
    let layer = app.ui.text_edit.as_ref()?.layer;
    if !doc.iter().all(|v| v.is_finite()) {
        return None;
    }
    let at = crate::type_tool::offset_at(app, photocraft_doc::LayerId(layer), doc[0], doc[1]);
    let found = app.session.execute("edit.checkSpelling", json!({"action": "list", "layer": layer, "suggestions": MAX_SUGGESTIONS})).ok()?;
    let hit = found["misspellings"].as_array()?.iter().find(|m| {
        let (a, b) = (m["start"].as_u64().unwrap_or(u64::MAX) as usize, m["end"].as_u64().unwrap_or(0) as usize);
        a <= at && at <= b
    })?;
    Some(Spelling {
        layer,
        start: hit["start"].as_u64()? as usize,
        end: hit["end"].as_u64()? as usize,
        word: hit["word"].as_str()?.to_string(),
        suggestions: hit["suggestions"].as_array()?.iter().filter_map(|s| s.as_str().map(str::to_string)).take(MAX_SUGGESTIONS).collect(),
    })
}

/// The spelling rows heading the type edit menu (label, id): the suggestions, or a greyed note
/// when there are none, then Add to Dictionary. Empty away from a misspelled word.
pub fn spelling_rows(menu: &CanvasToolMenu) -> Vec<(String, String)> {
    let Some(sp) = menu.spelling.as_ref().filter(|_| menu.text) else { return Vec::new() };
    let mut rows: Vec<(String, String)> = sp.suggestions.iter().enumerate().map(|(i, s)| (s.clone(), format!("{SPELLING_REPLACE}{i}"))).collect();
    if rows.is_empty() {
        rows.push(("No Spelling Suggestions".into(), SPELLING_NONE.into()));
    }
    rows.push(("Add to Dictionary".into(), SPELLING_ADD.into()));
    rows
}

/// Is `command` one of the menu's spelling rows?
fn is_spelling(menu: &CanvasToolMenu, command: &str) -> bool {
    spelling_rows(menu).iter().any(|(_, id)| id == command)
}

/// Does a spelling row's label come from the UI (translated) rather than from the dictionary?
pub fn spelling_label_is_ui(id: &str) -> bool {
    matches!(id, SPELLING_NONE | SPELLING_ADD)
}

/// Right-click over a Free Transform box: its modes.
pub fn open_transform(app: &mut PhotocraftApp, ctx: &Context, pos: [f32; 2]) -> bool {
    if !pos.iter().all(|v| v.is_finite()) {
        return false;
    }
    app.ui.brush_picker = None;
    app.ui.layer_menu = None;
    app.ui.canvas_tool_menu = Some(CanvasToolMenu {
        pos,
        tool: app.ui.tool,
        has_selection: false,
        has_path: false,
        path_name: None,
        transform: true,
        pasteboard: false,
        anchor: None,
        text: false,
        spelling: None,
    });
    crate::canvas_popup::mark_open(ctx, menu_id());
    true
}

/// The pasteboard colour a pasteboard menu row picks (`None` for Select Custom Color… or a row
/// of another menu).
fn pasteboard_choice(command: &str) -> Option<CanvasColor> {
    command.strip_prefix("pasteboard.").and_then(CanvasColor::parse)
}

/// The engine command (and params) a pasteboard row runs, for the automation policy: every row
/// changes a preference.
pub fn engine_call(menu: &CanvasToolMenu, command: &str) -> (String, Value) {
    match pasteboard_choice(command) {
        Some(c) if menu.pasteboard => ("prefs.set".into(), json!({"values": {"interface.canvasColor": c.name()}})),
        _ if menu.pasteboard => ("prefs.set".into(), json!({})),
        _ if menu.text && matches!(command, FAUX_BOLD | FAUX_ITALIC) => ("type.setStyle".into(), json!({})),
        _ if is_spelling(menu, command) => spelling_call(menu, command),
        _ => (command.into(), json!({})),
    }
}

/// The `edit.checkSpelling` call a spelling row makes: replace the word with a suggestion, or add
/// it to the user dictionary.
fn spelling_call(menu: &CanvasToolMenu, command: &str) -> (String, Value) {
    let Some(sp) = menu.spelling.as_ref() else { return ("edit.checkSpelling".into(), json!({})) };
    let replace = command.strip_prefix(SPELLING_REPLACE).and_then(|n| n.parse::<usize>().ok()).and_then(|n| sp.suggestions.get(n));
    match replace {
        Some(to) => {
            ("edit.checkSpelling".into(), json!({"action": "change", "layer": sp.layer, "start": sp.start, "end": sp.end, "word": sp.word, "replace": to}))
        }
        None => ("edit.checkSpelling".into(), json!({"action": "addToDictionary", "word": sp.word})),
    }
}

/// Does a right-click with `tool` at document point `doc` open the pasteboard menu? Outside the
/// image, for the tools whose right-click has no job there: not the painting tools (their Brush
/// Preset picker or right-button erase), the Pen (its path menu) or text being edited.
pub fn pasteboard_at(app: &PhotocraftApp, tool: Tool, doc: [f64; 2]) -> bool {
    let Some(st) = app.session.active() else { return false };
    let (w, h) = (f64::from(st.doc.size.width), f64::from(st.doc.size.height));
    let outside = doc.iter().all(|v| v.is_finite()) && (doc[0] < 0.0 || doc[1] < 0.0 || doc[0] >= w || doc[1] >= h);
    outside && tool != Tool::Pen && !crate::paint_mouse::has_brush_picker(tool) && !crate::paint_mouse::right_erases(app, tool) && app.ui.text_edit.is_none()
}

/// Right-click on the pasteboard: its colour menu at screen point `pos`.
pub fn open_pasteboard(app: &mut PhotocraftApp, ctx: &Context, tool: Tool, pos: [f32; 2]) -> bool {
    if !pos.iter().all(|v| v.is_finite()) {
        return false;
    }
    app.ui.brush_picker = None;
    app.ui.layer_menu = None;
    app.ui.canvas_tool_menu = Some(CanvasToolMenu {
        pos,
        tool,
        has_selection: false,
        has_path: false,
        path_name: None,
        transform: false,
        pasteboard: true,
        anchor: None,
        text: false,
        spelling: None,
    });
    crate::canvas_popup::mark_open(ctx, menu_id());
    true
}

/// A pasteboard row: set the colour, or open the Color Picker on the custom one.
fn choose_pasteboard(app: &mut PhotocraftApp, menu: &CanvasToolMenu, command: &str) -> Result<(), String> {
    if command == PASTEBOARD_PICK {
        let rgb = prefs::parse_hex(&app.session.prefs().interface.canvas_custom_color).map_or([0.5; 3], |c| c.map(|v| f32::from(v) / 255.0));
        crate::color_picker_ui::open_for_target(app, "pasteboard", "Color Picker (Custom Canvas Color)", rgb);
        return Ok(());
    }
    let (command, params) = engine_call(menu, command);
    app.run(&command, params).map(|_| ())
}

pub fn entry_enabled(app: &PhotocraftApp, menu: &CanvasToolMenu, command: &str) -> bool {
    if menu.pasteboard {
        return PASTEBOARD_MENU.iter().flatten().any(|(_, id)| *id == command);
    }
    // Warp needs a layer, selection or path box: a lone mask or channel keeps the plain box.
    if menu.transform && command == "edit.transform.warp" {
        return app.ui.transform.as_ref().is_some_and(|t| t.warp.is_none() && t.target.is_none());
    }
    if menu.anchor.is_some() && command == FILL_SCREEN {
        return app.session.active().is_some();
    }
    if menu.text {
        return match command {
            SPELLING_NONE => false,
            c if is_spelling(menu, c) => app.ui.text_edit.is_some(),
            FAUX_BOLD | FAUX_ITALIC => app.ui.text_edit.is_some(),
            c => TYPE_EDIT_MENU.iter().flatten().any(|(_, id)| *id == c) && crate::menus::is_enabled(app, c),
        };
    }
    if menu.transform || menu.tool != Tool::Pen {
        return crate::menus::is_enabled(app, command);
    }
    let Some(st) = app.session.active() else { return false };
    let layer = st.active_layer.and_then(|id| st.doc.layer(id));
    let locked = layer.is_some_and(|l| st.doc.effective_locks(l.id).all);
    let can_paint = layer.is_some_and(|l| matches!(l.content, LayerContent::Raster(_))) && !locked;
    let shape_layer = layer.and_then(|l| match &l.content {
        LayerContent::Shape(sh) => Some(sh),
        _ => None,
    });
    let shape = shape_layer.is_some();
    let can_mask = layer.is_some_and(|l| !matches!(l.content, LayerContent::Shape(_))) && !locked;
    let saved = menu.path_name.as_deref().is_some_and(|n| n != "work" && n != "layer");
    let has_path = menu.has_path;
    let pending = app.ui.pen.as_ref().is_some_and(|p| p.knots.len() >= 2);
    let pending_layer = pending && (app.ui.tool_options.vector_mode == "shape" || app.ui.vector_mask_target);
    match command {
        "layer.vectorMask.fromPath" => has_path && can_mask && !pending_layer,
        "path.delete" => has_path && !pending_layer && (pending || menu.path_name.as_deref() != Some("layer")),
        "edit.defineCustomShape" | "path.toSelection" | "path.transform" | "paint.symmetryFromPath" => has_path,
        "view.newGuidesFromShape" => shape && !pending,
        "path.fill" | "path.stroke" => has_path && can_paint && !pending_layer,
        "path.clippingPath.set" => saved && !pending,
        "layer.combineShapes.unite"
        | "layer.combineShapes.subtractFrontShape"
        | "layer.combineShapes.intersectShapeAreas"
        | "layer.combineShapes.excludeOverlappingShapes" => !pending && shape_layer.is_some_and(|sh| sh.path.subpaths.len() > 1),
        "path.style.copyFill" => !pending && shape_layer.is_some_and(|sh| sh.fill.is_some()),
        "path.style.copyStroke" => !pending && shape_layer.is_some_and(|sh| sh.stroke.is_some()),
        "path.style.pasteFill" => !pending && shape && app.session.path_fill_clipboard.is_some(),
        "path.style.pasteStroke" => !pending && shape && app.session.path_stroke_clipboard.is_some(),
        "select.isolateLayers" => layer.is_some(),
        "paint.symmetryDisable" => st.symmetry.is_some(),
        _ => false,
    }
}

pub fn open(app: &mut PhotocraftApp, ctx: &Context, tool: Tool, pos: [f32; 2]) -> bool {
    open_at(app, ctx, tool, pos, None)
}

/// [`open`] for a right-click at document point `doc`: over a misspelled word of the text being
/// edited, the menu starts with its spelling suggestions.
pub fn open_at(app: &mut PhotocraftApp, ctx: &Context, tool: Tool, pos: [f32; 2], doc: Option<[f64; 2]>) -> bool {
    let text = editing_text(app, tool);
    if !(applies(tool) || text) || !pos.iter().all(|v| v.is_finite()) {
        return false;
    }
    app.ui.brush_picker = None;
    app.ui.layer_menu = None;
    let has_selection = app.session.active().is_some_and(|s| s.doc.selection.is_some());
    let has_path = crate::vector_ui::active_path_name(app).is_some() || app.ui.pen.as_ref().is_some_and(|p| p.knots.len() >= 2);
    let path_name = crate::vector_ui::active_path_name(app);
    let spelling = if text { doc.and_then(|d| spelling_at(app, d)) } else { None };
    app.ui.canvas_tool_menu =
        Some(CanvasToolMenu { pos, tool, has_selection, has_path, path_name, transform: false, pasteboard: false, anchor: None, text, spelling });
    crate::canvas_popup::mark_open(ctx, menu_id());
    true
}

/// Right-click with the Zoom or Hand tool at screen point `pos`, over document point `doc`: the
/// view menu.
pub fn open_view(app: &mut PhotocraftApp, ctx: &Context, tool: Tool, pos: [f32; 2], doc: [f64; 2]) -> bool {
    if !is_view_tool(tool) || !pos.iter().all(|v| v.is_finite()) || !doc.iter().all(|v| v.is_finite()) {
        return false;
    }
    app.ui.brush_picker = None;
    app.ui.layer_menu = None;
    app.ui.canvas_tool_menu = Some(CanvasToolMenu {
        pos,
        tool,
        has_selection: false,
        has_path: false,
        path_name: None,
        transform: false,
        pasteboard: false,
        anchor: Some(doc),
        text: false,
        spelling: None,
    });
    crate::canvas_popup::mark_open(ctx, menu_id());
    true
}

/// A view menu row. Zoom In and Zoom Out step about the right-clicked point, as a Zoom tool click
/// does; the other rows run their View menu command.
fn choose_view(app: &mut PhotocraftApp, ctx: &Context, menu: &CanvasToolMenu, command: &str) -> Result<(), String> {
    let i = app.session.active_index().ok_or("no document")?;
    if command == FILL_SCREEN {
        let v = app.ui.views.get_mut(i).ok_or("no document view")?;
        v.fit_pending = false;
        v.fill_pending = true;
        return Ok(());
    }
    let before = app.ui.views.get(i).map(|v| (v.zoom, v.center));
    crate::menus::invoke(app, ctx, command, json!({}))?;
    if !matches!(command, "view.zoomIn" | "view.zoomOut") {
        return Ok(());
    }
    let (Some((zoom0, c)), Some(d)) = (before, menu.anchor) else { return Ok(()) };
    let center_on_point = app.session.prefs().tools.zoom_clicked_point_to_center;
    let Some(v) = app.ui.views.get_mut(i) else { return Ok(()) };
    let d = [d[0] as f32, d[1] as f32];
    // The view scales about the centre, so the point keeps its screen position when its offset
    // from the centre shrinks (or grows) by the zoom ratio, whatever the rotation or flip.
    let ratio = zoom0 / v.zoom;
    v.center = if center_on_point || !ratio.is_finite() { d } else { [d[0] - (d[0] - c[0]) * ratio, d[1] - (d[1] - c[1]) * ratio] };
    Ok(())
}

pub fn choose(app: &mut PhotocraftApp, ctx: &Context, command: &str) {
    let Some(menu) = app.ui.canvas_tool_menu.take() else { return };
    if !available(app, &menu, command) {
        return;
    }
    let result = if menu.pasteboard {
        choose_pasteboard(app, &menu, command)
    } else if menu.anchor.is_some() {
        choose_view(app, ctx, &menu, command)
    } else if menu.text && is_spelling(&menu, command) {
        choose_spelling(app, &menu, command)
    } else if menu.text {
        choose_type(app, ctx, command)
    } else if menu.tool == Tool::Pen && !menu.transform {
        choose_pen(app, ctx, &menu, command)
    } else if command == "select.toWorkPath" {
        // Photoshop's Make Work Path… asks for the tolerance first.
        spec_dialog(app, command, "Make Work Path", r##"{"tolerance":0.5..10=2}"##, serde_json::Map::new());
        Ok(())
    } else {
        crate::menus::invoke(app, ctx, command, json!({})).map(|_| ())
    };
    if let Err(e) = result {
        app.ui.status = e;
        app.ui.status_error = true;
    }
}

/// A spelling row: replace the word under the pointer (the caret lands after the replacement), or
/// add it to the user dictionary.
fn choose_spelling(app: &mut PhotocraftApp, menu: &CanvasToolMenu, command: &str) -> Result<(), String> {
    let (id, params) = engine_call(menu, command);
    let result = app.run(&id, params)?;
    if let Some(end) = result["end"].as_u64().map(|e| e as usize)
        && let Some(e) = app.ui.text_edit.as_mut()
    {
        (e.caret, e.anchor) = (end, end);
    }
    Ok(())
}

/// A type edit row. Undo, Cut, Copy, Paste and Select All run as their Edit and Select menu items,
/// which act on the text while it is being edited (Paste asks the window for the clipboard text).
fn choose_type(app: &mut PhotocraftApp, ctx: &Context, command: &str) -> Result<(), String> {
    match command {
        FAUX_BOLD | FAUX_ITALIC => {
            let bold = command == FAUX_BOLD;
            let on = !type_checked(app, command);
            let mut p = crate::type_tool::formatting_params(app).ok_or("No type is being edited")?;
            p[if bold { "fauxBold" } else { "fauxItalic" }] = json!(on);
            app.run("type.setStyle", p)?;
        }
        _ => {
            if ENDS_TYPE_EDIT.contains(&command) {
                crate::type_tool::commit(app);
            }
            crate::menus::invoke(app, ctx, command, json!({}))?;
        }
    }
    Ok(())
}

/// A parameter dialog for `command` following `spec` (registry notation), prefilled with `fields`.
fn spec_dialog(app: &mut PhotocraftApp, command: &str, label: &str, spec: &str, mut fields: serde_json::Map<String, serde_json::Value>) {
    fields.insert("__command".into(), json!(command));
    fields.insert("__label".into(), json!(label));
    fields.insert("__filter".into(), json!(true));
    fields.insert("__spec".into(), json!(spec));
    app.ui.open_dialog(crate::state::DialogKind::Command, fields);
}

/// Photoshop's Stroke Path dialog: the brush-based tool to stroke with (painting with that tool's
/// current options) and Simulate Pressure (#2978).
const STROKE_PATH_SPEC: &str = r##"{"tool":"brush|pencil|eraser|backgroundEraser|cloneStamp|patternStamp|healingBrush|spotHealingBrush|historyBrush|blur|sharpen|smudge|dodge|burn|sponge|colorReplacement|mixerBrush"="brush","simulatePressure":bool=false}"##;

/// The Stroke Path dialog for the path `name` (Paths panel menu, Alt-click on its Stroke button).
pub(crate) fn stroke_path_dialog(app: &mut PhotocraftApp, name: &str) {
    pen_dialog(app, "path.stroke", "Stroke Path…", name);
}

fn pen_dialog(app: &mut PhotocraftApp, command: &str, label: &str, name: &str) {
    let spec = match command {
        "edit.defineCustomShape" => r##"{"name":text}"##,
        "path.toSelection" => r##"{"feather":0..250=0,"antiAlias":bool=true,"mode":"replace|add|subtract|intersect"="replace"}"##,
        "path.fill" => r##"{"color":text,"opacity":0..100=100,"feather":0..250=0,"antiAlias":bool=true}"##,
        "path.stroke" => STROKE_PATH_SPEC,
        "path.clippingPath.set" => r##"{"flatness":0..100=0}"##,
        "path.transform" => r##"{"translateX":-30000..30000=0,"translateY":-30000..30000=0,"scaleX":0.01..100=1,"scaleY":0.01..100=1,"angle":-360..360=0}"##,
        _ => "{}",
    };
    let mut fields = serde_json::Map::new();
    if command == "edit.defineCustomShape" {
        fields.insert("path".into(), json!(name));
        fields.insert("name".into(), json!(""));
    } else {
        fields.insert("name".into(), json!(name));
    }
    if command == "path.fill" {
        let fg = app.session.tools.foreground;
        let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        fields.insert("color".into(), json!(format!("#{:02x}{:02x}{:02x}", byte(fg[0]), byte(fg[1]), byte(fg[2]))));
    }
    if command == "path.transform" {
        for (key, value) in [("translateX", 0.0), ("translateY", 0.0), ("scaleX", 1.0), ("scaleY", 1.0), ("angle", 0.0)] {
            fields.insert(key.into(), json!(value));
        }
    }
    spec_dialog(app, command, label, spec, fields);
}

fn choose_pen(app: &mut PhotocraftApp, ctx: &Context, menu: &CanvasToolMenu, command: &str) -> Result<(), String> {
    if command == "select.isolateLayers" {
        crate::menus::invoke(app, ctx, command, json!({}))?;
        return Ok(());
    }
    if command == "paint.symmetryDisable" {
        app.run(command, json!({}))?;
        return Ok(());
    }
    let pending = app.ui.pen.as_ref().is_some_and(|p| p.knots.len() >= 2);
    let pending_layer = pending && (app.ui.tool_options.vector_mode == "shape" || app.ui.vector_mask_target);
    if pending {
        crate::vector_ui::pen_commit(app, false);
    }
    let name = if pending { Some(if pending_layer { "layer" } else { "work" }.to_string()) } else { menu.path_name.clone() }
        .ok_or_else(|| "No path is available for this action".to_string())?;
    match command {
        "edit.defineCustomShape" | "path.toSelection" | "path.fill" | "path.stroke" | "path.clippingPath.set" | "path.transform" => {
            let label = PEN_MENU.iter().flatten().find(|(_, id)| *id == command).map_or(command, |(label, _)| *label);
            pen_dialog(app, command, label, &name);
        }
        _ => {
            app.run(command, json!({"name": name}))?;
        }
    }
    Ok(())
}

fn menu_id() -> egui::Id {
    egui::Id::new("canvas-selection-menu")
}

pub fn show(app: &mut PhotocraftApp, ctx: &Context) {
    let Some(menu) = app.ui.canvas_tool_menu.clone() else { return };
    if ctx.input(|i| i.key_pressed(egui::Key::Escape))
        || app.ui.tool != menu.tool
        || (menu.transform && app.ui.transform.is_none())
        || (menu.text && app.ui.text_edit.is_none())
    {
        app.ui.canvas_tool_menu = None;
        return;
    }
    let id = menu_id();
    let screen = ctx.content_rect();
    let size = ctx.memory(|m| m.area_rect(id)).map_or(egui::vec2(210.0, 140.0), |r| r.size());
    let pos = egui::pos2(menu.pos[0].min(screen.right() - size.x).max(screen.left()), menu.pos[1].min(screen.bottom() - size.y).max(screen.top()));
    let mut selected = None;
    let area = egui::Area::new(id).order(egui::Order::Foreground).fixed_pos(pos).show(ctx, |ui| {
        egui::Frame::menu(ui.style()).show(ui, |ui| {
            let t = crate::theme::Tokens::get(ui.ctx());
            let v = &mut ui.style_mut().visuals;
            v.widgets.inactive.weak_bg_fill = egui::Color32::TRANSPARENT;
            v.widgets.inactive.bg_stroke = egui::Stroke::NONE;
            v.widgets.hovered.weak_bg_fill = t.accent;
            v.widgets.hovered.bg_fill = t.accent;
            v.widgets.hovered.bg_stroke = egui::Stroke::NONE;
            v.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);
            v.widgets.hovered.corner_radius = egui::CornerRadius::same(3);
            ui.spacing_mut().item_spacing.y = 0.0;
            ui.set_width(200.0);
            crate::widgets::menu_scroll(ui, |ui| {
                let spelling = spelling_rows(&menu);
                for (label, command) in &spelling {
                    let text = if spelling_label_is_ui(command) { tl!(label).to_string() } else { label.clone() };
                    let item = egui::Button::new(text).min_size(egui::vec2(200.0, 20.0));
                    if ui.add_enabled(entry_enabled(app, &menu, command), item).clicked() {
                        selected = Some(command.clone());
                    }
                }
                if !spelling.is_empty() {
                    ui.separator();
                }
                for row in rows(&menu) {
                    let Some((label, command)) = *row else {
                        ui.separator();
                        continue;
                    };
                    let item = egui::Button::selectable(mode_checked(app, &menu, command), tl!(label)).min_size(egui::vec2(200.0, 20.0));
                    if ui.add_enabled(entry_enabled(app, &menu, command), item).clicked() {
                        selected = Some((*command).to_string());
                    }
                }
            });
        });
    });
    if let Some(command) = selected {
        choose(app, ctx, &command);
    } else if crate::canvas_popup::outside_pressed(ctx, id, area.response.rect) {
        app.ui.canvas_tool_menu = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Modifiers, PointerButton, Pos2, vec2};
    use egui_kittest::{Harness, kittest::Queryable};

    fn app() -> PhotocraftApp {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 32, "height": 32})).unwrap();
        app
    }

    fn work_path(app: &mut PhotocraftApp) {
        app.run(
            "path.set",
            json!({"name": "work", "path": {"subpaths": [{"closed": true, "knots": [
                {"anchor": [2, 2], "in": [2, 2], "out": [2, 2]},
                {"anchor": [20, 2], "in": [20, 2], "out": [20, 2]},
                {"anchor": [20, 20], "in": [20, 20], "out": [20, 20]}
            ]}]}}),
        )
        .unwrap();
    }

    fn choose_open(app: &mut PhotocraftApp, command: &str) {
        assert!(open(app, &Context::default(), Tool::Pen, [10.0, 10.0]));
        choose(app, &Context::default(), command);
    }

    fn harness(mut app: PhotocraftApp) -> Harness<'static, PhotocraftApp> {
        app.sync_views();
        let mut h = Harness::builder().with_size(vec2(1000.0, 700.0)).build_ui_state(
            |ui, app: &mut PhotocraftApp| {
                let ctx = ui.ctx().clone();
                if !ctx.fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                    return;
                }
                egui::CentralPanel::default().show(ui, |ui| crate::canvas::document_area(app, ui));
            },
            app,
        );
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::default());
        h.run_steps(4);
        h
    }

    fn right_click(h: &mut Harness<'static, PhotocraftApp>, p: Pos2, modifiers: Modifiers) {
        h.event(egui::Event::ModifiersChanged(modifiers));
        h.event(egui::Event::PointerMoved(p));
        h.run_steps(1);
        h.event(egui::Event::PointerButton { pos: p, button: PointerButton::Secondary, pressed: true, modifiers });
        h.run_steps(1);
        h.event(egui::Event::PointerButton { pos: p, button: PointerButton::Secondary, pressed: false, modifiers });
        h.event(egui::Event::ModifiersChanged(Modifiers::NONE));
        h.run_steps(2);
    }

    #[test]
    fn canvas_gesture_preserves_selection_and_command_override() {
        let mut app = app();
        app.ui.tool = Tool::Lasso;
        app.run("select.rect", json!({"x": 1, "y": 1, "width": 8, "height": 8})).unwrap();
        let mut h = harness(app);
        let center = h.state().last_canvas_rect.center();
        let revision = h.state().session.active().unwrap().revision;
        right_click(&mut h, center, Modifiers::NONE);
        assert!(h.state().ui.canvas_tool_menu.is_some());
        assert_eq!(h.state().session.active().unwrap().revision, revision, "right-click cannot alter selection");
        h.key_press(egui::Key::Escape);
        h.run_steps(2);
        assert!(h.state().ui.canvas_tool_menu.is_none());
        right_click(&mut h, center, Modifiers::COMMAND);
        assert!(h.state().ui.canvas_tool_menu.is_none(), "command right-click belongs to layer picker");
    }

    /// A type layer reading "Hello", being edited with every character selected.
    fn editing_type(app: &mut PhotocraftApp) -> u64 {
        app.ui.tool = Tool::Type;
        let id = app.run("type.create", json!({"text": "Hello", "size": 12, "x": 4, "y": 20})).unwrap()["layer"].as_u64().unwrap();
        crate::menus::invoke(app, &Context::default(), "type.editText", json!({})).unwrap();
        id
    }

    fn text_of(app: &PhotocraftApp, id: u64) -> Option<photocraft_doc::TextLayer> {
        match &app.session.active().unwrap().doc.layer(photocraft_doc::LayerId(id)).unwrap().content {
            LayerContent::Text(t) => Some(t.clone()),
            _ => None,
        }
    }

    #[test]
    fn right_click_while_typing_opens_the_type_edit_menu_and_esc_keeps_the_edit() {
        let mut app = app();
        editing_type(&mut app);
        app.sync_views();
        let mut h = Harness::builder().with_size(vec2(1000.0, 700.0)).build_ui_state(
            |ui, app: &mut PhotocraftApp| {
                let ctx = ui.ctx().clone();
                if !ctx.fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                    return;
                }
                // The app's order: shortcuts (inline type's keys among them), then the canvas.
                crate::shortcuts::handle(app, &ctx);
                egui::CentralPanel::default().show(ui, |ui| crate::canvas::document_area(app, ui));
            },
            app,
        );
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::default());
        h.run_steps(4);
        let center = h.state().last_canvas_rect.center();
        right_click(&mut h, center, Modifiers::NONE);
        let menu = h.state().ui.canvas_tool_menu.clone().expect("right-click in a type edit opens a menu");
        assert!(menu.text && !menu.pasteboard && !menu.transform);
        assert_eq!(rows(&menu), TYPE_EDIT_MENU);
        let edit = h.state().ui.text_edit.clone().expect("the edit goes on under the menu");
        assert_eq!((edit.anchor, edit.caret), (0, 5), "the right-click keeps the selection for Cut and Copy");
        h.key_press(egui::Key::Escape);
        h.run_steps(2);
        assert!(h.state().ui.canvas_tool_menu.is_none(), "Esc closes the menu");
        assert!(h.state().ui.text_edit.is_some(), "and only the menu: the edit isn't committed");
    }

    #[test]
    fn type_edit_menu_rows_act_on_the_text_and_the_layer() {
        let mut app = app();
        let id = editing_type(&mut app);
        let ctx = Context::default();
        let reopen = |app: &mut PhotocraftApp| assert!(open(app, &Context::default(), Tool::Type, [10.0, 10.0]));
        // Faux Bold toggles the style of the selected characters, and shows checked.
        reopen(&mut app);
        let menu = app.ui.canvas_tool_menu.clone().unwrap();
        assert!(!mode_checked(&app, &menu, FAUX_BOLD) && mode_checked(&app, &menu, "type.antiAlias.smooth"));
        choose(&mut app, &ctx, FAUX_BOLD);
        assert!(text_of(&app, id).unwrap().char_runs().iter().all(|r| r.style.faux_bold));
        assert!(mode_checked(&app, &menu, FAUX_BOLD));
        // Cut takes the selection out of the text; with nothing selected Copy and Cut grey out.
        reopen(&mut app);
        choose(&mut app, &ctx, "edit.cut");
        assert_eq!(text_of(&app, id).unwrap().text, "");
        reopen(&mut app);
        let menu = app.ui.canvas_tool_menu.clone().unwrap();
        assert!(!entry_enabled(&app, &menu, "edit.copy") && !entry_enabled(&app, &menu, "edit.cut"));
        app.ui.canvas_tool_menu = None;
        crate::menus::invoke(&mut app, &ctx, "edit.undo", json!({})).unwrap();
        assert_eq!(text_of(&app, id).unwrap().text, "Hello");
        // Select All selects every character, not the canvas.
        reopen(&mut app);
        choose(&mut app, &ctx, "select.all");
        let edit = app.ui.text_edit.clone().unwrap();
        assert_eq!((edit.anchor, edit.caret), (0, 5));
        assert!(app.session.active().unwrap().doc.selection.is_none());
        // Warp Text… opens its dialog; Rasterize Type ends the edit and rasterizes the layer.
        reopen(&mut app);
        choose(&mut app, &ctx, "type.warpText");
        assert!(!app.ui.dialogs.is_empty(), "Warp Text… asks for its settings");
        app.ui.dialogs.clear();
        reopen(&mut app);
        choose(&mut app, &ctx, "type.rasterizeTypeLayer");
        assert!(app.ui.text_edit.is_none() && text_of(&app, id).is_none(), "the type layer is pixels now");
    }

    #[test]
    fn type_edit_menu_lists_spelling_suggestions_for_the_word_under_the_pointer() {
        let mut app = app();
        app.ui.tool = Tool::Type;
        let id = app.run("type.create", json!({"text": "Teh cat", "size": 12, "x": 4, "y": 20})).unwrap()["layer"].as_u64().unwrap();
        let ctx = Context::default();
        crate::menus::invoke(&mut app, &ctx, "type.editText", json!({})).unwrap();
        // Away from the misspelling (past the end of the line): no spelling rows.
        assert!(open_at(&mut app, &ctx, Tool::Type, [10.0, 10.0], Some([600.0, 15.0])));
        assert!(spelling_rows(app.ui.canvas_tool_menu.as_ref().unwrap()).is_empty());
        // Over "Teh": its suggestions head the menu, then Add to Dictionary.
        assert!(open_at(&mut app, &ctx, Tool::Type, [10.0, 10.0], Some([5.0, 15.0])));
        let menu = app.ui.canvas_tool_menu.clone().unwrap();
        let rows = spelling_rows(&menu);
        assert!(rows.iter().any(|(l, _)| l == "The"), "{rows:?}");
        assert_eq!(rows.last().map(|(_, id)| id.as_str()), Some(SPELLING_ADD));
        assert!(rows.len() <= MAX_SUGGESTIONS + 1, "at most {MAX_SUGGESTIONS} suggestions and Add to Dictionary: {rows:?}");
        assert!(available(&app, &menu, &rows[0].1));
        let pick = rows.iter().find(|(l, _)| l == "The").unwrap().1.clone();
        choose(&mut app, &ctx, &pick);
        assert_eq!(text_of(&app, id).unwrap().text, "The cat");
        assert_eq!(app.ui.text_edit.as_ref().map(|e| (e.caret, e.anchor)), Some((3, 3)), "the caret lands after the replacement");
        // Add to Dictionary stops flagging a word.
        app.run("type.insertText", json!({"layer": id, "text": " Zqx"})).unwrap();
        assert!(open_at(&mut app, &ctx, Tool::Type, [10.0, 10.0], Some([600.0, 15.0])));
        let menu = app.ui.canvas_tool_menu.clone().unwrap();
        assert!(menu.spelling.as_ref().is_some_and(|s| s.word == "Zqx"));
        choose(&mut app, &ctx, SPELLING_ADD);
        assert!(app.session.prefs().user_dictionary.iter().any(|w| w == "Zqx"));
    }

    #[test]
    fn right_click_outside_the_image_picks_the_pasteboard_color() {
        let mut app = app();
        app.ui.tool = Tool::RectMarquee;
        let mut h = harness(app);
        let rect = h.state().last_canvas_rect;
        let outside = rect.min + vec2(5.0, 5.0);
        right_click(&mut h, rect.center(), Modifiers::NONE);
        assert!(h.state().ui.canvas_tool_menu.as_ref().is_some_and(|m| !m.pasteboard), "inside: the tool's own menu");
        h.state_mut().ui.canvas_tool_menu = None;
        right_click(&mut h, outside, Modifiers::NONE);
        let menu = h.state().ui.canvas_tool_menu.clone().unwrap();
        assert!(menu.pasteboard && rows(&menu) == PASTEBOARD_MENU, "outside: the pasteboard menu");
        let ctx = h.ctx.clone();
        choose(h.state_mut(), &ctx, "pasteboard.black");
        assert_eq!(crate::prefs_ui::pasteboard_color(h.state()), Some(egui::Color32::BLACK));
        // Select Custom Color… opens the Color Picker; OK makes its colour the custom pasteboard.
        right_click(&mut h, outside, Modifiers::NONE);
        choose(h.state_mut(), &ctx, PASTEBOARD_PICK);
        let mut f = h.state().ui.dialogs.last().unwrap().fields.clone();
        f.insert("color".into(), json!("#123456"));
        crate::color_picker_ui::confirm(h.state_mut(), &f).unwrap();
        assert_eq!(crate::prefs_ui::pasteboard_color(h.state()), Some(egui::Color32::from_rgb(0x12, 0x34, 0x56)));
        // A painting tool keeps its Brush Preset picker there.
        h.state_mut().ui.tool = Tool::Brush;
        h.run_steps(2);
        right_click(&mut h, outside, Modifiers::NONE);
        assert!(h.state().ui.canvas_tool_menu.is_none() && h.state().ui.brush_picker.is_some());
    }

    /// #2832: right-click over an open Free Transform box: Photoshop's transform menu. Skew
    /// switches the box's mode; Rotate and Flip turn the box about its reference point without
    /// committing it; Content-Aware Scale and Puppet Warp are greyed.
    #[test]
    fn right_click_during_free_transform_switches_modes_rotates_and_flips_the_box() {
        let mut app = app();
        app.ui.tool = Tool::RectMarquee;
        app.run("layer.new.layer", json!({})).unwrap();
        app.run("select.rect", json!({"x": 4, "y": 6, "width": 10, "height": 8})).unwrap();
        app.run("edit.fill", json!({"color": "#ff0000"})).unwrap();
        app.run("select.deselect", json!({})).unwrap();
        let mut h = harness(app);
        let ctx = h.ctx.clone();
        crate::menus::invoke(h.state_mut(), &ctx, "edit.freeTransform", json!({})).unwrap();
        h.run_steps(2);
        let center = h.state().last_canvas_rect.center();
        let (revision, journal) = (h.state().session.active().unwrap().revision, h.state().session.journal.len());
        right_click(&mut h, center, Modifiers::NONE);
        let menu = h.state().ui.canvas_tool_menu.clone().expect("the transform menu");
        assert!(menu.transform && rows(&menu) == TRANSFORM_MENU);
        for (id, on) in
            [("edit.contentAwareScale", false), ("edit.puppetWarp", false), ("edit.transform.rotate90Cw", true), ("edit.transform.flipVertical", true)]
        {
            assert_eq!(entry_enabled(h.state(), &menu, id), on, "{id}");
        }
        choose(h.state_mut(), &ctx, "edit.transform.skew");
        assert_eq!(h.state().ui.transform.as_ref().unwrap().mode, crate::state::TransformMode::Skew);
        let t = h.state().ui.transform.clone().unwrap();
        let (q0, c) = (t.quad, t.pivot);
        right_click(&mut h, center, Modifiers::NONE);
        choose(h.state_mut(), &ctx, "edit.transform.flipHorizontal");
        let q = h.state().ui.transform.as_ref().expect("the box stays open").quad;
        for (a, b) in q.iter().zip(q0) {
            assert!((a[0] - (2.0 * c[0] - b[0])).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9, "mirrored about the reference point: {q0:?} -> {q:?}");
        }
        right_click(&mut h, center, Modifiers::NONE);
        choose(h.state_mut(), &ctx, "edit.transform.rotate180");
        let q = h.state().ui.transform.as_ref().unwrap().quad;
        for (a, b) in q.iter().zip(q0) {
            assert!((a[1] - (2.0 * c[1] - b[1])).abs() < 1e-9 && (a[0] - b[0]).abs() < 1e-9, "flipped then turned 180°: {q0:?} -> {q:?}");
        }
        let st = h.state().session.active().unwrap();
        assert_eq!((st.revision, h.state().session.journal.len()), (revision, journal), "nothing is committed");
    }

    #[test]
    fn transform_menu_follows_photoshop_order() {
        assert_eq!(
            labels(TRANSFORM_MENU),
            vec![
                Some("Free Transform"),
                Some("Scale"),
                Some("Rotate"),
                Some("Skew"),
                Some("Distort"),
                Some("Perspective"),
                Some("Warp"),
                None,
                Some("Content-Aware Scale"),
                Some("Puppet Warp"),
                None,
                Some("Rotate 180°"),
                Some("Rotate 90° Clockwise"),
                Some("Rotate 90° Counter Clockwise"),
                None,
                Some("Flip Horizontal"),
                Some("Flip Vertical"),
            ]
        );
        // Over a box on editable type, Distort and Perspective are greyed (#2630).
        let mut app = app();
        app.run("type.create", json!({"x": 2, "y": 20, "text": "Hi", "size": 16})).unwrap();
        crate::transform_tool::begin(&mut app, &Context::default()).unwrap();
        assert!(open_transform(&mut app, &Context::default(), [10.0, 10.0]));
        let menu = app.ui.canvas_tool_menu.clone().unwrap();
        for (id, on) in [("edit.transform.skew", true), ("edit.transform.distort", false), ("edit.transform.perspective", false)] {
            assert_eq!(entry_enabled(&app, &menu, id), on, "{id}");
        }
    }

    fn labels(rows: &[Row]) -> Vec<Option<&str>> {
        rows.iter().map(|row| row.map(|(label, _)| label)).collect()
    }

    #[test]
    fn selection_menus_follow_photoshop_order() {
        assert_eq!(
            labels(SELECTION_MENU),
            vec![
                Some("Deselect"),
                Some("Select Inverse"),
                Some("Feather…"),
                Some("Select and Mask…"),
                None,
                Some("Save Selection…"),
                Some("Make Work Path…"),
                None,
                Some("Layer via Copy"),
                Some("Layer via Cut"),
                Some("New Layer…"),
                None,
                Some("Free Transform"),
                Some("Transform Selection"),
                Some("Distort"),
                Some("Perspective"),
                None,
                Some("Delete and Fill Selection"),
                Some("Fill…"),
                Some("Stroke…"),
                Some("Content-Aware Fill…"),
                None,
                Some("Last Filter"),
                None,
                Some("Fade…"),
            ]
        );
        assert_eq!(labels(NO_SELECTION_MENU), vec![Some("Select All"), Some("Reselect"), Some("Color Range…"), None, Some("Load Selection…")]);
        // Every row runs a real command: one the menu bar has, or an engine command.
        let app = app();
        let items = crate::menus::menu_items(&app);
        for (label, id) in [SELECTION_MENU, NO_SELECTION_MENU, PEN_MENU, TRANSFORM_MENU].into_iter().flatten().flatten() {
            assert!(items.iter().any(|i| i.id == *id) || photocraft_engine::commands::find(id).is_some(), "{label}: unknown command {id}");
        }
    }

    #[test]
    fn selection_menu_routes_actions_through_commands() {
        let mut app = app();
        app.ui.tool = Tool::RectMarquee;
        app.run("select.rect", json!({"x": 1, "y": 1, "width": 8, "height": 8})).unwrap();
        assert!(open(&mut app, &Context::default(), Tool::RectMarquee, [10.0, 10.0]));
        assert_eq!(rows(app.ui.canvas_tool_menu.as_ref().unwrap()), SELECTION_MENU);
        let before = app.session.journal.len();
        choose(&mut app, &Context::default(), "select.inverse");
        assert!(app.ui.canvas_tool_menu.is_none());
        assert!(app.session.journal[before..].iter().any(|(id, _)| id == "select.inverse"));
        let before = app.session.journal.len();
        assert!(open(&mut app, &Context::default(), Tool::RectMarquee, [10.0, 10.0]));
        choose(&mut app, &Context::default(), "select.deselect");
        assert!(app.session.active().unwrap().doc.selection.is_none());
        assert!(app.session.journal[before..].iter().any(|(id, _)| id == "select.deselect"));
        // Nothing selected: the menu that makes a selection.
        assert!(open(&mut app, &Context::default(), Tool::RectMarquee, [10.0, 10.0]));
        assert_eq!(rows(app.ui.canvas_tool_menu.as_ref().unwrap()), NO_SELECTION_MENU);
        choose(&mut app, &Context::default(), "select.reselect");
        assert!(app.session.active().unwrap().doc.selection.is_some());
        assert!(app.session.journal.iter().any(|(id, _)| id == "select.reselect"));
        app.run("select.deselect", json!({})).unwrap();
        assert!(open(&mut app, &Context::default(), Tool::Lasso, [10.0, 10.0]));
        choose(&mut app, &Context::default(), "select.all");
        assert!(app.session.active().unwrap().doc.selection.is_some());
    }

    #[test]
    fn selection_menu_layer_and_path_actions() {
        let mut app = app();
        app.ui.tool = Tool::EllipseMarquee;
        app.run("layer.new.layer", json!({})).unwrap();
        app.run("edit.fill", json!({"color": "#ff0000"})).unwrap();
        app.run("select.rect", json!({"x": 4, "y": 4, "width": 10, "height": 10})).unwrap();
        let layers = app.session.active().unwrap().doc.layers.len();
        assert!(open(&mut app, &Context::default(), Tool::EllipseMarquee, [10.0, 10.0]));
        choose(&mut app, &Context::default(), "layer.new.layerViaCopy");
        assert_eq!(app.session.active().unwrap().doc.layers.len(), layers + 1, "Layer via Copy adds a layer");
        assert_eq!(app.session.journal.last().map(|(id, _)| id.as_str()), Some("layer.new.layerViaCopy"));
        // Make Work Path… asks for the tolerance, then traces the selection.
        app.run("select.rect", json!({"x": 4, "y": 4, "width": 10, "height": 10})).unwrap();
        assert!(open(&mut app, &Context::default(), Tool::EllipseMarquee, [10.0, 10.0]));
        choose(&mut app, &Context::default(), "select.toWorkPath");
        let dialog = app.ui.dialogs.last().map(|d| d.id).expect("tolerance dialog");
        assert!(app.session.active().unwrap().doc.work_path.is_none(), "nothing happens before OK");
        crate::dialogs::confirm(&mut app, dialog).unwrap();
        assert!(app.session.active().unwrap().doc.work_path.is_some());
        assert_eq!(app.session.journal.last().map(|(id, _)| id.as_str()), Some("select.toWorkPath"));
    }

    #[test]
    fn disabled_selection_rows_do_nothing() {
        let mut app = app();
        app.ui.tool = Tool::MagicWand;
        app.run("select.rect", json!({"x": 1, "y": 1, "width": 8, "height": 8})).unwrap();
        assert!(open(&mut app, &Context::default(), Tool::MagicWand, [10.0, 10.0]));
        let menu = app.ui.canvas_tool_menu.clone().unwrap();
        // No filter has run and nothing can be faded yet.
        for id in ["filter.lastFilter", "edit.fade"] {
            assert!(!entry_enabled(&app, &menu, id), "{id} is greyed out");
        }
        let before = app.session.journal.len();
        choose(&mut app, &Context::default(), "edit.fade");
        assert_eq!(app.session.journal.len(), before);
        assert!(app.ui.dialogs.is_empty());
        // Rows of the other variant are not part of this menu.
        assert!(open(&mut app, &Context::default(), Tool::MagicWand, [10.0, 10.0]));
        choose(&mut app, &Context::default(), "select.all");
        assert_eq!(app.session.journal.len(), before);
    }

    #[test]
    fn view_menu_follows_photoshop_order() {
        assert_eq!(
            labels(VIEW_MENU),
            vec![Some("Zoom In"), Some("Zoom Out"), None, Some("Fit on Screen"), Some("Fill Screen"), None, Some("100%"), Some("200%"), Some("Print Size")]
        );
        // Every row but Fill Screen (the Hand tool's options-bar button) is a View menu command.
        let app = app();
        let items = crate::menus::menu_items(&app);
        for (label, id) in VIEW_MENU.iter().flatten().filter(|(_, id)| *id != FILL_SCREEN) {
            assert!(items.iter().any(|i| i.id == *id), "{label}: unknown command {id}");
        }
    }

    fn zoom_app(w: u32, h: u32) -> PhotocraftApp {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": w, "height": h})).unwrap();
        app
    }

    fn screen_of(app: &PhotocraftApp, doc: [f32; 2]) -> Pos2 {
        crate::canvas::ViewXform::active(app).unwrap().to_screen(doc[0], doc[1])
    }

    #[test]
    fn zoom_and_hand_right_click_opens_the_view_menu() {
        for tool in [Tool::Zoom, Tool::Hand] {
            let mut app = zoom_app(400, 300);
            app.ui.tool = tool;
            let mut h = harness(app);
            let zoom = h.state().ui.views[0].zoom;
            let inside = screen_of(h.state(), [300.0, 100.0]);
            right_click(&mut h, inside, Modifiers::NONE);
            let menu = h.state().ui.canvas_tool_menu.clone().unwrap_or_else(|| panic!("{tool:?}: right-click opens no menu"));
            assert!(rows(&menu) == VIEW_MENU, "{tool:?}: the view menu");
            assert_eq!(h.state().ui.views[0].zoom, zoom, "{tool:?}: the right-click alone does not zoom");
            assert!(entry_enabled(h.state(), &menu, FILL_SCREEN) && entry_enabled(h.state(), &menu, "view.printSize"));
            // Outside the image the pasteboard colour menu wins, as for the other tools.
            h.state_mut().ui.canvas_tool_menu = None;
            let outside = h.state().last_canvas_rect.min + vec2(5.0, 5.0);
            right_click(&mut h, outside, Modifiers::NONE);
            assert!(h.state().ui.canvas_tool_menu.as_ref().is_some_and(|m| m.pasteboard), "{tool:?}: pasteboard menu outside");
        }
    }

    #[test]
    fn view_menu_zoom_in_keeps_the_clicked_point_under_the_pointer() {
        let mut app = zoom_app(400, 300);
        app.ui.tool = Tool::Zoom;
        let mut h = harness(app);
        let zoom = h.state().ui.views[0].zoom;
        let doc = [300.0, 100.0];
        let p = screen_of(h.state(), doc);
        right_click(&mut h, p, Modifiers::NONE);
        // Chosen in the menu itself: the canvas frame that draws the menu must keep the new view.
        h.get_by_label("Zoom In").click();
        h.run_steps(3);
        assert!(h.state().ui.canvas_tool_menu.is_none());
        let after = h.state().ui.views[0].zoom;
        assert_eq!(after, crate::zoom_levels::step(zoom, 1, [400, 300]), "one Zoom In step");
        let s = screen_of(h.state(), doc);
        assert!((s - p).length() < 1.0, "the right-clicked point stays put: {s:?} vs {p:?}");
        assert_eq!(h.state().session.journal.last().map(|(id, _)| id.as_str()), Some("view.zoomIn"));
        // Zoom Out returns to where it was.
        right_click(&mut h, p, Modifiers::NONE);
        h.get_by_label("Zoom Out").click();
        h.run_steps(3);
        assert_eq!(h.state().ui.views[0].zoom, zoom);
        assert!((screen_of(h.state(), doc) - p).length() < 1.0);
    }

    #[test]
    fn view_menu_zoom_in_centres_the_clicked_point_with_the_preference() {
        let mut app = zoom_app(400, 300);
        app.ui.tool = Tool::Hand;
        app.run("prefs.set", json!({"path": "tools.zoomClickedPointToCenter", "value": true})).unwrap();
        let mut h = harness(app);
        let doc = [300.0, 100.0];
        let p = screen_of(h.state(), doc);
        right_click(&mut h, p, Modifiers::NONE);
        h.get_by_label("Zoom In").click();
        h.run_steps(3);
        assert_eq!(h.state().ui.views[0].center, doc);
    }

    #[test]
    fn view_menu_fits_fills_and_sets_fixed_zooms() {
        let mut app = zoom_app(400, 300);
        app.ui.tool = Tool::Zoom;
        let mut h = harness(app);
        let p = screen_of(h.state(), [200.0, 150.0]);
        let area = crate::rulers::content_rect(h.state(), h.state().last_canvas_rect).size();
        let ppp = h.state().canvas_ppp();
        let pick = |h: &mut Harness<'static, PhotocraftApp>, label: &str| {
            right_click(h, p, Modifiers::NONE);
            h.get_by_label(label).click();
            h.run_steps(3);
            h.state().ui.views[0].zoom
        };
        assert_eq!(pick(&mut h, "200%"), 2.0);
        // Fill Screen: the image covers the whole canvas (one axis can overflow).
        let fill = pick(&mut h, "Fill Screen") / ppp;
        assert!(400.0 * fill >= area.x - 1.0 && 300.0 * fill >= area.y - 1.0, "fill {fill} for {area:?}");
        assert!(400.0 * fill <= area.x + 1.0 || 300.0 * fill <= area.y + 1.0, "fill {fill} for {area:?}");
        assert_eq!(h.state().ui.views[0].center, [200.0, 150.0]);
        // Fit on Screen: back to the whole image (capped at 100%, as on open).
        let fit = pick(&mut h, "Fit on Screen") / ppp;
        assert!(400.0 * fit <= area.x && 300.0 * fit <= area.y && fit < fill, "fit {fit} fill {fill}");
    }

    #[test]
    fn view_menu_needs_finite_points_and_a_view_tool() {
        let mut app = app();
        assert!(!open_view(&mut app, &Context::default(), Tool::Brush, [1.0, 1.0], [1.0, 1.0]));
        assert!(!open_view(&mut app, &Context::default(), Tool::Zoom, [f32::NAN, 1.0], [1.0, 1.0]));
        assert!(!open_view(&mut app, &Context::default(), Tool::Zoom, [1.0, 1.0], [f64::INFINITY, 1.0]));
        assert!(app.ui.canvas_tool_menu.is_none());
        assert!(open_view(&mut app, &Context::default(), Tool::Zoom, [1.0, 1.0], [1.0, 1.0]));
        let before = app.session.journal.len();
        choose(&mut app, &Context::default(), "select.all");
        assert_eq!(app.session.journal.len(), before, "the view menu cannot run another menu's row");
        // With no document every row is greyed out and choosing one does nothing.
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        assert!(open_view(&mut app, &Context::default(), Tool::Hand, [1.0, 1.0], [1.0, 1.0]));
        let menu = app.ui.canvas_tool_menu.clone().unwrap();
        for (_, id) in VIEW_MENU.iter().flatten() {
            assert!(!entry_enabled(&app, &menu, id), "{id} is greyed out with no document");
        }
        choose(&mut app, &Context::default(), FILL_SCREEN);
        choose(&mut app, &Context::default(), "view.zoomIn");
    }

    #[test]
    fn unsupported_tools_and_nonfinite_positions_do_not_open_menu() {
        let mut app = app();
        assert!(!open(&mut app, &Context::default(), Tool::Brush, [1.0, 1.0]));
        assert!(!open(&mut app, &Context::default(), Tool::Lasso, [f32::NAN, 1.0]));
        assert!(app.ui.canvas_tool_menu.is_none());
        assert!(open(&mut app, &Context::default(), Tool::MagicWand, [1.0, 1.0]));
        let before = app.session.journal.len();
        choose(&mut app, &Context::default(), "file.new");
        assert_eq!(app.session.journal.len(), before, "context menu cannot invoke an unrelated command");
    }

    #[test]
    fn completed_pen_path_right_click_makes_selection() {
        let mut app = app();
        app.ui.tool = Tool::Pen;
        for (x, y) in [(2.0, 2.0), (20.0, 2.0), (20.0, 20.0)] {
            crate::vector_ui::pen_down(&mut app, x, y);
            crate::vector_ui::pen_up(&mut app);
        }
        crate::vector_ui::pen_commit(&mut app, true);
        assert!(app.session.active().unwrap().doc.work_path.is_some());
        let mut h = harness(app);
        let center = h.state().last_canvas_rect.center();
        right_click(&mut h, center, Modifiers::NONE);
        assert!(h.state().ui.canvas_tool_menu.as_ref().is_some_and(|m| m.tool == Tool::Pen && m.has_path));
        let ctx = h.ctx.clone();
        choose(h.state_mut(), &ctx, "path.toSelection");
        let dialog = h.state().ui.dialogs.last().map(|d| d.id).unwrap();
        crate::dialogs::confirm(h.state_mut(), dialog).unwrap();
        assert!(h.state().session.active().unwrap().doc.selection.is_some());
        assert_eq!(h.state().session.journal.last().map(|(id, _)| id.as_str()), Some("path.toSelection"));
    }

    #[test]
    fn pen_menu_finishes_in_progress_path_before_selection() {
        let mut app = app();
        app.ui.tool = Tool::Pen;
        for (x, y) in [(2.0, 2.0), (20.0, 2.0), (20.0, 20.0)] {
            crate::vector_ui::pen_down(&mut app, x, y);
            crate::vector_ui::pen_up(&mut app);
        }
        assert!(app.session.active().unwrap().doc.work_path.is_none());
        assert!(open(&mut app, &Context::default(), Tool::Pen, [10.0, 10.0]));
        let menu = app.ui.canvas_tool_menu.as_ref().unwrap();
        assert!(entry_enabled(&app, menu, "path.toSelection"));
        choose(&mut app, &Context::default(), "path.toSelection");
        let dialog = app.ui.dialogs.last().map(|d| d.id).unwrap();
        crate::dialogs::confirm(&mut app, dialog).unwrap();
        assert!(app.session.active().unwrap().doc.work_path.is_some());
        assert!(app.session.active().unwrap().doc.selection.is_some());
    }

    #[test]
    fn pen_menu_has_all_reference_rows_in_order_and_guards_unavailable_actions() {
        let labels: Vec<_> = PEN_MENU.iter().map(|row| row.map(|(label, _)| label)).collect();
        assert_eq!(
            labels,
            vec![
                Some("Create Vector Mask"),
                Some("Delete Path"),
                None,
                Some("Define Custom Shape…"),
                None,
                Some("Make Selection…"),
                Some("New Guides From Shape"),
                Some("Fill Path…"),
                Some("Stroke Path…"),
                None,
                Some("Clipping Path…"),
                None,
                Some("Free Transform Path"),
                None,
                Some("Unite Shapes"),
                Some("Subtract Front Shape"),
                Some("Unite Shapes at Overlap"),
                Some("Subtract Shapes at Overlap"),
                None,
                Some("Copy Fill"),
                Some("Copy Complete Stroke"),
                None,
                Some("Paste Fill"),
                Some("Paste Complete Stroke"),
                None,
                Some("Isolate Layers"),
                None,
                Some("Make Symmetry Path"),
                Some("Disable Symmetry Path"),
            ]
        );
        let mut app = app();
        app.ui.tool = Tool::Pen;
        assert!(open(&mut app, &Context::default(), Tool::Pen, [10.0, 10.0]));
        let menu = app.ui.canvas_tool_menu.as_ref().unwrap();
        assert!(!entry_enabled(&app, menu, "path.toSelection"));
        let before = app.session.journal.len();
        choose(&mut app, &Context::default(), "path.toSelection");
        assert_eq!(app.session.journal.len(), before);
        assert!(app.ui.dialogs.is_empty());
    }

    #[test]
    fn pen_menu_creates_mask_deletes_path_and_opens_named_path_dialogs() {
        let mut app = app();
        app.ui.tool = Tool::Pen;
        work_path(&mut app);
        choose_open(&mut app, "layer.vectorMask.fromPath");
        let st = app.session.active().unwrap();
        assert!(st.active_layer.and_then(|id| st.doc.layer(id)).is_some_and(|l| l.vector_mask.is_some()));
        // The new vector mask's path becomes the active one (#4728); pick the Work Path again.
        assert_eq!(app.ui.selected_path.as_deref(), Some("layer"));
        app.ui.selected_path = Some("work".into());
        choose_open(&mut app, "edit.defineCustomShape");
        let id = app.ui.dialogs.last().map(|d| d.id).unwrap();
        let fields = &app.ui.dialogs.last().unwrap().fields;
        assert_eq!(fields.get("path"), Some(&json!("work")));
        app.ui.dialog_mut(id).unwrap().fields.insert("name".into(), json!("Menu Shape"));
        crate::dialogs::confirm(&mut app, id).unwrap();
        assert!(app.session.edit_state.custom_shapes.iter().any(|s| s.name == "Menu Shape"));
        choose_open(&mut app, "path.fill");
        let id = app.ui.dialogs.last().map(|d| d.id).unwrap();
        assert_eq!(app.ui.dialogs.last().unwrap().fields.get("name"), Some(&json!("work")));
        crate::dialogs::confirm(&mut app, id).unwrap();
        assert_eq!(app.session.journal.last().map(|(id, _)| id.as_str()), Some("path.fill"));
        choose_open(&mut app, "path.delete");
        assert!(app.session.active().unwrap().doc.work_path.is_none());
    }

    #[test]
    fn pen_menu_saved_path_clipping_and_numeric_transform() {
        let mut app = app();
        app.ui.tool = Tool::Pen;
        work_path(&mut app);
        app.run("path.rename", json!({"name": "work", "to": "Cutout"})).unwrap();
        app.ui.selected_path = Some("Cutout".into());
        assert!(open(&mut app, &Context::default(), Tool::Pen, [10.0, 10.0]));
        assert!(entry_enabled(&app, app.ui.canvas_tool_menu.as_ref().unwrap(), "path.clippingPath.set"));
        choose(&mut app, &Context::default(), "path.clippingPath.set");
        let id = app.ui.dialogs.last().map(|d| d.id).unwrap();
        app.ui.dialog_mut(id).unwrap().fields.insert("flatness".into(), json!(2.0));
        crate::dialogs::confirm(&mut app, id).unwrap();
        assert_eq!(app.session.active().unwrap().doc.clipping_path.as_ref().map(|p| p.name.as_str()), Some("Cutout"));
        choose_open(&mut app, "path.transform");
        let id = app.ui.dialogs.last().map(|d| d.id).unwrap();
        app.ui.dialog_mut(id).unwrap().fields.insert("translateX".into(), json!(5.0));
        crate::dialogs::confirm(&mut app, id).unwrap();
        assert_eq!(app.session.journal.last().map(|(id, _)| id.as_str()), Some("path.transform"));
    }

    #[test]
    fn pen_shape_menu_enables_shape_ops_and_style_copy_only_when_applicable() {
        let mut app = app();
        app.ui.tool = Tool::Pen;
        app.run(
            "shape.create",
            json!({"kind":"path", "path":{"subpaths":[
            {"closed":true,"knots":[[2,2],[20,2],[20,20]]},
            {"closed":true,"knots":[[5,5],[10,5],[10,10]]}
        ]}, "fill":"#ff0000", "stroke":{"width":2,"color":"#000000"}}),
        )
        .unwrap();
        assert!(open(&mut app, &Context::default(), Tool::Pen, [10.0, 10.0]));
        let menu = app.ui.canvas_tool_menu.as_ref().unwrap();
        assert!(entry_enabled(&app, menu, "layer.combineShapes.unite"));
        assert!(entry_enabled(&app, menu, "path.style.copyFill"));
        assert!(!entry_enabled(&app, menu, "path.style.pasteFill"));
        assert!(!entry_enabled(&app, menu, "path.fill"));
        choose(&mut app, &Context::default(), "path.style.copyFill");
        assert!(app.session.path_fill_clipboard.is_some());
        assert!(open(&mut app, &Context::default(), Tool::Pen, [10.0, 10.0]));
        assert!(entry_enabled(&app, app.ui.canvas_tool_menu.as_ref().unwrap(), "path.style.pasteFill"));
    }

    #[test]
    fn unfinished_shape_path_disables_actions_that_would_target_the_previous_layer() {
        let mut app = app();
        app.ui.tool = Tool::Pen;
        app.ui.tool_options.vector_mode = "shape".into();
        for (x, y) in [(2.0, 2.0), (20.0, 2.0), (20.0, 20.0)] {
            crate::vector_ui::pen_down(&mut app, x, y);
            crate::vector_ui::pen_up(&mut app);
        }
        assert!(open(&mut app, &Context::default(), Tool::Pen, [10.0, 10.0]));
        let menu = app.ui.canvas_tool_menu.as_ref().unwrap();
        for id in ["layer.vectorMask.fromPath", "path.delete", "path.fill", "path.stroke", "path.clippingPath.set", "view.newGuidesFromShape"] {
            assert!(!entry_enabled(&app, menu, id), "{id} must not act on the layer replaced by the pending shape");
        }
        assert!(entry_enabled(&app, menu, "path.toSelection"));
    }
}
