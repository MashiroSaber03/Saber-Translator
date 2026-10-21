//! UI state as plain data, so the control channel (and later MCP) can read and drive every
//! aspect of the interface: tools, zoom, panels, dialogs and windows.

use serde::{Deserialize, Serialize};

/// Shared selection and capture for a point curve; document parameters remain with the host.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct PointCurveState {
    pub selected: Option<usize>,
    /// Captured point (None while removed outside), grab offset, and re-entry eligibility.
    pub drag: Option<PointCurveDrag>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct PointCurveDrag {
    pub index: Option<usize>,
    pub offset: [f32; 2],
    pub removed: bool,
}

/// Curves editor memory (egui temp data): the edited channel and the point gesture.
#[derive(Clone, Copy, Debug, Default)]
pub struct CurvesEditorState {
    pub channel: usize,
    pub gesture: PointCurveState,
}

/// Camera Raw scope preferences are view state, never filter parameters or document history.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct CameraRawScopeState {
    pub shadows: bool,
    pub highlights: bool,
    pub lab: bool,
    pub sampler_tool: bool,
    pub samplers: Vec<[f32; 2]>,
    pub vectorscope: bool,
    pub selected_region: bool,
    pub red_right: bool,
    pub hide_skin_line: bool,
    pub floating: bool,
    pub floating_rect: Option<[f32; 4]>,
}

/// Camera Raw navigation, separate from filter settings. Reset for each opened image.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct CameraRawPreviewState {
    /// None fits the image; Some is physical display pixels per source pixel.
    pub zoom: Option<f32>,
    pub center: [f32; 2],
    pub hand: bool,
}

impl Default for CameraRawPreviewState {
    fn default() -> Self {
        Self { zoom: None, center: [0.5, 0.5], hand: false }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Tool {
    Move,
    Artboard,
    RectMarquee,
    EllipseMarquee,
    /// Selects one pixel row across the whole canvas at the clicked y (Photoshop's flyout only:
    /// no key, not in the ⇧M cycle).
    SingleRowMarquee,
    /// Selects one pixel column down the whole canvas at the clicked x.
    SingleColumnMarquee,
    Lasso,
    PolygonLasso,
    MagneticLasso,
    MagicWand,
    Crop,
    PerspectiveCrop,
    Eyedropper,
    ColorSampler,
    Ruler,
    Note,
    Count,
    Brush,
    Pencil,
    MixerBrush,
    ColorReplacement,
    Eraser,
    BackgroundEraser,
    MagicEraser,
    Gradient,
    PaintBucket,
    Type,
    VerticalType,
    /// Horizontal Type Mask: types a selection in the shape of the glyphs, no type layer.
    HorizontalTypeMask,
    VerticalTypeMask,
    Hand,
    RotateView,
    Zoom,
    Remove,
    SpotHealing,
    Healing,
    Patch,
    ContentAwareMove,
    RedEye,
    CloneStamp,
    PatternStamp,
    HistoryBrush,
    Blur,
    Sharpen,
    Smudge,
    Dodge,
    Burn,
    Sponge,
    QuickSelection,
    ObjectSelection,
    Pen,
    FreeformPen,
    CurvaturePen,
    AddAnchorPoint,
    DeleteAnchorPoint,
    /// Converts an anchor between smooth and corner, and pulls handles out of a corner
    /// (Photoshop's Convert Point tool; also the Pen's ⌥ mode, `direct_select.rs`).
    ConvertPoint,
    PathSelection,
    DirectSelection,
    Rectangle,
    EllipseShape,
    Triangle,
    Polygon,
    Line,
    CustomShape,
    Slice,
    SliceSelect,
    Frame,
}

impl Tool {
    pub const ALL: [Tool; 67] = [
        Tool::Move,
        Tool::Artboard,
        Tool::RectMarquee,
        Tool::EllipseMarquee,
        Tool::SingleRowMarquee,
        Tool::SingleColumnMarquee,
        Tool::Lasso,
        Tool::PolygonLasso,
        Tool::MagneticLasso,
        Tool::MagicWand,
        Tool::Crop,
        Tool::PerspectiveCrop,
        Tool::Eyedropper,
        Tool::ColorSampler,
        Tool::Ruler,
        Tool::Note,
        Tool::Count,
        Tool::Brush,
        Tool::Pencil,
        Tool::MixerBrush,
        Tool::ColorReplacement,
        Tool::Eraser,
        Tool::BackgroundEraser,
        Tool::MagicEraser,
        Tool::Gradient,
        Tool::PaintBucket,
        Tool::Type,
        Tool::VerticalType,
        Tool::HorizontalTypeMask,
        Tool::VerticalTypeMask,
        Tool::Hand,
        Tool::RotateView,
        Tool::Zoom,
        Tool::Remove,
        Tool::SpotHealing,
        Tool::Healing,
        Tool::Patch,
        Tool::ContentAwareMove,
        Tool::RedEye,
        Tool::CloneStamp,
        Tool::PatternStamp,
        Tool::HistoryBrush,
        Tool::Blur,
        Tool::Sharpen,
        Tool::Smudge,
        Tool::Dodge,
        Tool::Burn,
        Tool::Sponge,
        Tool::QuickSelection,
        Tool::ObjectSelection,
        Tool::Pen,
        Tool::FreeformPen,
        Tool::CurvaturePen,
        Tool::AddAnchorPoint,
        Tool::DeleteAnchorPoint,
        Tool::ConvertPoint,
        Tool::PathSelection,
        Tool::DirectSelection,
        Tool::Rectangle,
        Tool::EllipseShape,
        Tool::Triangle,
        Tool::Polygon,
        Tool::Line,
        Tool::CustomShape,
        Tool::Slice,
        Tool::SliceSelect,
        Tool::Frame,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Tool::Move => "Move Tool",
            Tool::Artboard => "Artboard Tool",
            Tool::RectMarquee => "Rectangular Marquee Tool",
            Tool::EllipseMarquee => "Elliptical Marquee Tool",
            Tool::SingleRowMarquee => "Single Row Marquee Tool",
            Tool::SingleColumnMarquee => "Single Column Marquee Tool",
            Tool::Brush => "Brush Tool",
            Tool::Pencil => "Pencil Tool",
            Tool::MixerBrush => "Mixer Brush Tool",
            Tool::ColorReplacement => "Color Replacement Tool",
            Tool::Eraser => "Eraser Tool",
            Tool::BackgroundEraser => "Background Eraser Tool",
            Tool::MagicEraser => "Magic Eraser Tool",
            Tool::Eyedropper => "Eyedropper Tool",
            Tool::ColorSampler => "Color Sampler Tool",
            Tool::Ruler => "Ruler Tool",
            Tool::Note => "Note Tool",
            Tool::Count => "Count Tool",
            Tool::Lasso => "Lasso Tool",
            Tool::PolygonLasso => "Polygonal Lasso Tool",
            Tool::MagneticLasso => "Magnetic Lasso Tool",
            Tool::MagicWand => "Magic Wand Tool",
            Tool::Crop => "Crop Tool",
            Tool::PerspectiveCrop => "Perspective Crop Tool",
            Tool::Slice => "Slice Tool",
            Tool::SliceSelect => "Slice Select Tool",
            Tool::Frame => "Frame Tool",
            Tool::Gradient => "Gradient Tool",
            Tool::PaintBucket => "Paint Bucket Tool",
            Tool::Type => "Horizontal Type Tool",
            Tool::VerticalType => "Vertical Type Tool",
            Tool::HorizontalTypeMask => "Horizontal Type Mask Tool",
            Tool::VerticalTypeMask => "Vertical Type Mask Tool",
            Tool::Hand => "Hand Tool",
            Tool::RotateView => "Rotate View Tool",
            Tool::Zoom => "Zoom Tool",
            Tool::Remove => "Remove Tool",
            Tool::SpotHealing => "Spot Healing Brush Tool",
            Tool::Healing => "Healing Brush Tool",
            Tool::Patch => "Patch Tool",
            Tool::ContentAwareMove => "Content-Aware Move Tool",
            Tool::RedEye => "Red Eye Tool",
            Tool::CloneStamp => "Clone Stamp Tool",
            Tool::PatternStamp => "Pattern Stamp Tool",
            Tool::HistoryBrush => "History Brush Tool",
            Tool::Blur => "Blur Tool",
            Tool::Sharpen => "Sharpen Tool",
            Tool::Smudge => "Smudge Tool",
            Tool::Dodge => "Dodge Tool",
            Tool::Burn => "Burn Tool",
            Tool::Sponge => "Sponge Tool",
            Tool::QuickSelection => "Quick Selection Tool",
            Tool::ObjectSelection => "Object Selection Tool",
            Tool::Pen => "Pen Tool",
            Tool::FreeformPen => "Freeform Pen Tool",
            Tool::CurvaturePen => "Curvature Pen Tool",
            Tool::AddAnchorPoint => "Add Anchor Point Tool",
            Tool::DeleteAnchorPoint => "Delete Anchor Point Tool",
            Tool::ConvertPoint => "Convert Point Tool",
            Tool::PathSelection => "Path Selection Tool",
            Tool::DirectSelection => "Direct Selection Tool",
            Tool::Rectangle => "Rectangle Tool",
            Tool::EllipseShape => "Ellipse Tool",
            Tool::Triangle => "Triangle Tool",
            Tool::Polygon => "Polygon Tool",
            Tool::Line => "Line Tool",
            Tool::CustomShape => "Custom Shape Tool",
        }
    }
    pub fn is_type(self) -> bool {
        matches!(self, Self::Type | Self::VerticalType | Self::HorizontalTypeMask | Self::VerticalTypeMask)
    }
    /// The Type Mask tools: their text becomes a selection on commit.
    pub fn is_type_mask(self) -> bool {
        matches!(self, Self::HorizontalTypeMask | Self::VerticalTypeMask)
    }
    /// Type tools that lay text out vertically.
    pub fn is_vertical_type(self) -> bool {
        matches!(self, Self::VerticalType | Self::VerticalTypeMask)
    }

    /// Retouching and painting tools that stroke with the brush (share the brush cursor and chip).
    pub fn is_brushlike(self) -> bool {
        matches!(
            self,
            Tool::Brush
                | Tool::Pencil
                | Tool::MixerBrush
                | Tool::ColorReplacement
                | Tool::Eraser
                | Tool::BackgroundEraser
                | Tool::Remove
                | Tool::SpotHealing
                | Tool::Healing
                | Tool::CloneStamp
                | Tool::PatternStamp
                | Tool::HistoryBrush
                | Tool::Blur
                | Tool::Sharpen
                | Tool::Smudge
                | Tool::Dodge
                | Tool::Burn
                | Tool::Sponge
        )
    }
    /// Photoshop default single-key shortcut (`'\0'` = none, e.g. the Blur group).
    pub fn key(self) -> char {
        match self {
            Tool::Move | Tool::Artboard => 'V',
            Tool::RectMarquee | Tool::EllipseMarquee => 'M',
            Tool::Brush | Tool::Pencil | Tool::MixerBrush | Tool::ColorReplacement => 'B',
            Tool::Eraser | Tool::BackgroundEraser | Tool::MagicEraser => 'E',
            Tool::Eyedropper | Tool::ColorSampler | Tool::Ruler | Tool::Note | Tool::Count => 'I',
            Tool::Lasso | Tool::PolygonLasso | Tool::MagneticLasso => 'L',
            Tool::MagicWand => 'W',
            Tool::Crop | Tool::PerspectiveCrop | Tool::Slice | Tool::SliceSelect => 'C',
            Tool::Frame => 'K',
            Tool::Gradient | Tool::PaintBucket => 'G',
            Tool::Type | Tool::VerticalType | Tool::HorizontalTypeMask | Tool::VerticalTypeMask => 'T',
            Tool::Hand => 'H',
            Tool::RotateView => 'R',
            Tool::Zoom => 'Z',
            Tool::Remove | Tool::SpotHealing | Tool::Healing | Tool::Patch | Tool::ContentAwareMove | Tool::RedEye => 'J',
            Tool::CloneStamp | Tool::PatternStamp => 'S',
            Tool::HistoryBrush => 'Y',
            Tool::Blur | Tool::Sharpen | Tool::Smudge => '\0',
            // In the Marquee flyout, but M cycles only the Rectangular and Elliptical Marquee.
            Tool::SingleRowMarquee | Tool::SingleColumnMarquee => '\0',
            Tool::Dodge | Tool::Burn | Tool::Sponge => 'O',
            Tool::QuickSelection | Tool::ObjectSelection => 'W',
            Tool::Pen | Tool::FreeformPen | Tool::CurvaturePen | Tool::AddAnchorPoint | Tool::DeleteAnchorPoint | Tool::ConvertPoint => 'P',
            Tool::PathSelection | Tool::DirectSelection => 'A',
            Tool::Rectangle | Tool::EllipseShape | Tool::Triangle | Tool::Polygon | Tool::Line | Tool::CustomShape => 'U',
        }
    }
    /// Glyph drawn in the toolbar (vector icons come later).
    pub fn glyph(self) -> &'static str {
        match self {
            Tool::Move => "✥",
            Tool::RectMarquee => "⬚",
            Tool::EllipseMarquee => "◌",
            Tool::Brush => "🖌",
            Tool::MixerBrush => "🖌",
            Tool::Eraser => "⌫",
            Tool::Eyedropper => "💧",
            Tool::ColorSampler => "◎",
            Tool::Lasso | Tool::PolygonLasso | Tool::MagneticLasso => "L",
            Tool::MagicWand => "W",
            Tool::Crop => "C",
            Tool::Gradient | Tool::PaintBucket => "G",
            Tool::Type | Tool::VerticalType | Tool::HorizontalTypeMask | Tool::VerticalTypeMask => "T",
            Tool::Hand => "✋",
            Tool::RotateView => "🧭",
            Tool::Zoom => "🔍",
            _ => "•",
        }
    }
    pub fn from_name(s: &str) -> Option<Tool> {
        let n = s.to_ascii_lowercase().replace([' ', '_', '-'], "");
        Tool::ALL.into_iter().find(|t| {
            let a = format!("{t:?}").to_ascii_lowercase();
            n == a || n == a.replace("marquee", "") || n == t.label().to_ascii_lowercase().replace(' ', "")
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Panels {
    pub layers: bool,
    pub history: bool,
    pub properties: bool,
    pub color: bool,
    pub navigator: bool,
    pub toolbar: bool,
    pub options_bar: bool,
    pub status_bar: bool,
    /// Window › Brush Settings (F5): floating, like Photoshop's.
    #[serde(default)]
    pub brush_settings: bool,
    /// Window › Character / Paragraph: the Character | Paragraph dock group (#150).
    #[serde(default)]
    pub character: bool,
    /// The toolbar's header chevron: two columns even when one fits (#1197).
    #[serde(default)]
    pub toolbar_double: bool,
    /// The right-hand panel dock. ⇧Tab hides and shows it, as in Photoshop (#1313).
    #[serde(default = "yes")]
    pub dock: bool,
    /// What Tab hid (toolbar, options bar, dock), so a second Tab brings back just those.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hidden_by_tab: Option<[bool; 3]>,
    /// The right dock's icon rail. An app that embeds PhotoCraft's UI can hide it for a simpler view.
    #[serde(default = "yes")]
    pub rail: bool,
    /// The title bar with the in-window menus. An app that embeds PhotoCraft's UI and draws its own
    /// bar can hide it; the caption buttons of a custom title bar go with it.
    #[serde(default = "yes")]
    pub menu_bar: bool,
    /// The Gradient tool's options-bar swatch opened the Gradient Editor window (`gradient_ui`).
    /// Photoshop opens its Gradient Editor from that swatch; this is the same idea, drawn as a
    /// floating window rather than a modal so the canvas stays usable while a gradient is edited.
    #[serde(default)]
    pub gradient_editor: bool,
}

impl Default for Panels {
    fn default() -> Self {
        Self {
            layers: true,
            history: false,
            properties: true,
            color: true,
            navigator: false,
            toolbar: true,
            options_bar: true,
            status_bar: true,
            brush_settings: false,
            character: false,
            toolbar_double: false,
            dock: true,
            hidden_by_tab: None,
            rail: true,
            menu_bar: true,
            gradient_editor: false,
        }
    }
}

/// A modal or modeless dialog, identified by `id`, with editable fields.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dialog {
    pub id: u64,
    pub kind: DialogKind,
    /// Field values by name. Dialog widgets read and write these; so can automation.
    pub fields: serde_json::Map<String, serde_json::Value>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DialogKind {
    NewDocument,
    About,
    /// Parameter dialog for a command (generated from its params).
    Command,
    Error,
    /// Photoshop Layer Style dialog (see `layer_style.rs`).
    LayerStyle,
}

/// Per-document view (camera) state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct View {
    /// Screen (device) pixels per document pixel: the user-facing zoom (`100%` is `1.0`).
    /// Canvas geometry works in egui points and divides by `ctx.pixels_per_point`
    /// (`PhotocraftApp::point_zoom`), so neither the display scale nor Interface › UI Scale
    /// magnifies the image (#1943, #2121).
    pub zoom: f32,
    /// Document-space point shown at the canvas centre.
    pub center: [f32; 2],
    /// Recompute fit-to-screen on next frame.
    pub fit_pending: bool,
    /// Recompute fill-screen on next frame. Unlike fit, this may crop an edge of the document.
    #[serde(default)]
    pub fill_pending: bool,
    /// Document size this view last showed; a change (Image/Canvas Size, crop) re-centres it.
    #[serde(default)]
    pub doc_size: [u32; 2],
    /// Canvas camera rotation in degrees, wrapped to (−180, 180]. Turns the view, not the pixels.
    #[serde(default)]
    pub rotation: f32,
}

impl Default for View {
    fn default() -> Self {
        Self { zoom: 1.0, center: [0.0, 0.0], fit_pending: true, fill_pending: false, doc_size: [0, 0], rotation: 0.0 }
    }
}

/// An extra OS window showing a document ("Window → Arrange → New Window for Document").
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DocWindow {
    pub id: u64,
    pub document: usize,
    pub view: View,
    pub open: bool,
    /// Where the window opens on screen (outer top-left, in points); `None` lets the OS place it.
    /// A tab dragged off the tab strip opens its window where it was dropped (#3733).
    #[serde(default)]
    pub position: Option<[f32; 2]>,
}

/// Options-bar state for tools (Photoshop keeps these per tool).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolOptions {
    pub tolerance: f32,
    pub contiguous: bool,
    pub anti_alias: bool,
    pub sample_all_layers: bool,
    /// Object Selection › Object Finder: hovering highlights the object a click selects (#4087).
    pub object_finder: bool,
    /// Remove tool: the fill goes to a new layer above the target (Photoshop's Create a new layer).
    pub remove_new_layer: bool,
    /// Remove tool: each stroke is filled when released (Photoshop's default). Off, the strokes
    /// collect as a tint on the canvas and ↩ applies them together (Esc drops them).
    pub remove_after_each_stroke: bool,
    pub feather: f32,
    /// linear | radial | angle | reflected | diamond
    pub gradient_style: String,
    pub gradient_reverse: bool,
    /// Gradient tool: dither to reduce 8-bit banding (Photoshop default on).
    pub gradient_dither: bool,
    /// Gradient tool mode: false = "Gradient" (live: a Gradient Fill layer, editable on canvas),
    /// true = "Classic gradient" (paints the pixels).
    pub gradient_classic: bool,
    /// Blend mode used by both live and classic gradient drags.
    pub gradient_blend_mode: photocraft_color::BlendMode,
    pub fill_opacity: f32,
    /// Paint Bucket fill source: false = Foreground colour, true = Pattern (Patterns panel selection).
    pub bucket_fill_pattern: bool,
    /// Type tool: family, style name, size (pt), anti-aliasing and paragraph alignment.
    pub type_font: String,
    pub type_style: String,
    pub type_size: f32,
    pub type_aa: String,
    pub type_align: String,
    /// Clone Stamp / Healing Brush.
    pub clone_aligned: bool,
    /// Pattern Stamp: lock the tile origin in document space across strokes.
    pub pattern_stamp_aligned: bool,
    /// Pattern Stamp: jitter each dab's phase (seeded, replayable).
    pub pattern_stamp_impressionist: bool,
    /// Pattern Stamp scale %, 1..1000 (100 = the tile's native size).
    pub pattern_stamp_scale: f32,
    /// Pattern Stamp rotation in degrees (counter-clockwise).
    pub pattern_stamp_angle: f32,
    /// current | currentAndBelow | all
    pub clone_sample: String,
    /// Spot Healing: contentAware | createTexture | proximityMatch
    pub spot_type: String,
    /// Patch: source (repair the selection) | destination (repair where it is dragged).
    pub patch_mode: String,
    /// Patch: Content-Aware instead of Normal, with its Structure 1..7 and Color 0..10.
    pub patch_content_aware: bool,
    pub patch_structure: f32,
    pub patch_color: f32,
    /// Content-Aware Move: move | extend, Structure 1..7, Color 0..10.
    pub cam_mode: String,
    pub cam_structure: f32,
    pub cam_color: f32,
    /// Content-Aware Move: Transform On Drop (a transform box on the dropped patch).
    pub cam_transform_on_drop: bool,
    /// Dodge/Burn: shadows | midtones | highlights, exposure %, protect tones.
    pub tone_range: String,
    pub exposure: f32,
    pub protect_tones: bool,
    /// Sponge: desaturate | saturate, vibrance.
    pub sponge_mode: String,
    pub vibrance: bool,
    /// Blur/Sharpen/Smudge strength %, Sharpen protect detail, Smudge finger painting.
    pub strength: f32,
    pub protect_detail: bool,
    pub finger_painting: bool,
    /// Quick Selection enhance edge.
    pub enhance_edge: bool,
    /// Pen: "path" (work path) or "shape" (shape layer).
    pub vector_mode: String,
    /// Shape tools: fill with the foreground colour, stroke width (0 = none), rectangle corner
    /// radius, polygon sides, line weight.
    pub shape_fill: bool,
    pub stroke_width: f32,
    #[serde(default)]
    pub shape_stroke: crate::shape_stroke_ui::StrokeOptions,
    pub corner_radius: f32,
    pub polygon_sides: u32,
    pub line_weight: f32,
    /// Freeform Pen › Curve Fit: how far (px) the fitted path may stray from the pointer trace.
    #[serde(default = "default_curve_fit")]
    pub curve_fit: f32,
    /// Pen › Auto Add/Delete: a Pen click on a segment of the active path adds an anchor there,
    /// on one of its anchors deletes it (#4144).
    #[serde(default = "yes")]
    pub pen_auto_add_delete: bool,
    /// Marquee options bar: "normal" | "fixedRatio" | "fixedSize", with the ratio or size.
    #[serde(default = "default_marquee_style")]
    pub marquee_style: String,
    #[serde(default = "one")]
    pub marquee_width: f32,
    #[serde(default = "one")]
    pub marquee_height: f32,
    /// Move tool: Auto-Select (with "layer" or "group" target) and Show Transform Controls.
    /// Auto-Select is on with the "group" target by default, as in current Photoshop;
    /// ⌘/Ctrl-click inverts it.
    #[serde(default = "yes")]
    pub move_auto_select: bool,
    #[serde(default = "default_move_target")]
    pub move_target: String,
    #[serde(default)]
    pub move_show_transform: bool,
    /// Crop options bar: aspect ratio ("" = Ratio/unconstrained, "w:h", or "original") and
    /// Delete Cropped Pixels.
    #[serde(default)]
    pub crop_ratio: String,
    /// In-progress custom crop ratio fields. Kept separately because entering one side before the
    /// other must survive a frame while the ratio itself remains unconstrained.
    #[serde(default)]
    pub crop_custom_ratio_width: f32,
    #[serde(default)]
    pub crop_custom_ratio_height: f32,
    #[serde(default = "yes")]
    pub crop_delete: bool,
    /// Crop W x H x Resolution (`crop_ratio` = "whr", #2443): width and height as typed lengths
    /// with their unit ("4 in", "1024 px"; empty = unset), the resolution ("" = the document's)
    /// and its unit ("px/in" or "px/cm"). See `crop_size`.
    #[serde(default)]
    pub crop_width: String,
    #[serde(default)]
    pub crop_height: String,
    #[serde(default)]
    pub crop_resolution: String,
    #[serde(default = "default_crop_resolution_unit")]
    pub crop_resolution_unit: String,
    /// Perspective Crop's Show Grid (`perspective_crop_ui`); it shares W, H and the resolution above.
    #[serde(default = "yes")]
    pub perspective_crop_grid: bool,
    /// Crop overlay (#1919): the guide in the crop box, when it shows and its orientation
    /// (Photoshop's defaults: Rule of Thirds, Auto Show Overlay). See `crop_overlay`.
    #[serde(default)]
    pub crop_overlay: crate::crop_overlay::CropOverlay,
    #[serde(default)]
    pub crop_overlay_show: crate::crop_overlay::OverlayShow,
    #[serde(default)]
    pub crop_overlay_orientation: u8,
    /// Crop gear menu (#1919): Show Cropped Area and the crop shield. See `crop_shield`.
    #[serde(default)]
    pub crop_shield: crate::crop_shield::CropShield,
    /// Magic Eraser opacity % (tolerance, anti-alias, contiguous and sample-all are shared with the
    /// Magic Wand and Paint Bucket).
    pub magic_eraser_opacity: f32,
    /// Background Eraser: sampling (continuous | once | backgroundSwatch), limits (discontiguous |
    /// contiguous | findEdges), tolerance % and Protect Foreground Color.
    pub bg_sampling: String,
    pub bg_limits: String,
    pub bg_tolerance: f32,
    pub bg_protect_fg: bool,
    /// Color Replacement: mode (hue | saturation | color | luminosity), sampling (continuous |
    /// once | backgroundSwatch), limits (discontiguous | contiguous | findEdges), tolerance % and
    /// anti-alias, as `paint.colorReplacement` takes them.
    pub cr_mode: String,
    pub cr_sampling: String,
    pub cr_limits: String,
    pub cr_tolerance: f32,
    pub cr_anti_alias: bool,
    /// Zoom tool › Scrubby Zoom: dragging left/right zooms continuously (else a zoom rectangle).
    #[serde(default = "yes")]
    pub zoom_scrubby: bool,
    /// Pencil › Auto Erase: a stroke that starts on the foreground colour paints the background colour.
    #[serde(default)]
    pub pencil_auto_erase: bool,
    /// Brush › Mode: Clear, which erases with the Brush (Photoshop's painting-only mode, #3439).
    /// Not a blend mode, so it is kept beside the brush's `mode` rather than in it.
    #[serde(default)]
    pub brush_clear: bool,
    /// Pencil › Mode: Clear (see `brush_clear`).
    #[serde(default)]
    pub pencil_clear: bool,
    /// Eraser › Mode: `brush` (soft, anti-aliased) or `pencil` (hard, aliased pixels, through
    /// `paint.pencil`), as in Photoshop's options bar (#2662).
    #[serde(default = "default_eraser_mode")]
    pub eraser_mode: String,
    /// Magnetic Lasso: detection width (px, 1..256), edge contrast (%, 1..100), how often it
    /// fastens points by itself (0..100), and whether pen pressure narrows the width.
    pub magnetic_width: f32,
    pub magnetic_contrast: f32,
    pub magnetic_frequency: f32,
    pub magnetic_pressure: bool,
    /// Red Eye: pupil search size (1–100) and how far corrected pixels darken (0–100).
    #[serde(default = "fifty")]
    pub red_eye_pupil_size: f32,
    #[serde(default = "fifty")]
    pub red_eye_darken: f32,
    /// Eyedropper › Sample Size: 1 = Point Sample, else the side of the averaged square (3, 5,
    /// 11, 31, 51, 101; `photocraft_engine::sample_cmds::SAMPLE_SIZES`). Shared, as in Photoshop,
    /// with the ⌥-click eyedropper of the painting tools, the dialog eyedroppers and the Info panel.
    #[serde(default = "one_u32")]
    pub eyedropper_size: u32,
    /// Eyedropper › Sample: current | currentAndBelow | all | allNoAdjustments |
    /// currentAndBelowNoAdjustments (`document.sampleColor`'s `sampleLayer`).
    #[serde(default = "default_eyedropper_sample")]
    pub eyedropper_sample: String,
    /// Eyedropper › Show Sampling Ring.
    #[serde(default = "yes")]
    pub eyedropper_ring: bool,
}

fn yes() -> bool {
    true
}

fn default_crop_resolution_unit() -> String {
    crate::crop_size::PX_PER_IN.into()
}

fn default_move_target() -> String {
    "group".into()
}

fn default_curve_fit() -> f32 {
    2.0
}

fn default_eraser_mode() -> String {
    "brush".into()
}

fn default_marquee_style() -> String {
    "normal".into()
}

fn one() -> f32 {
    1.0
}

fn fifty() -> f32 {
    50.0
}

/// The session brush starts out as the Brush's (the tool the app opens with).
fn default_brush_tool() -> Tool {
    Tool::Brush
}

fn one_u32() -> u32 {
    1
}

fn default_eyedropper_sample() -> String {
    "all".into()
}

impl Default for ToolOptions {
    fn default() -> Self {
        Self {
            tolerance: 32.0,
            contiguous: true,
            anti_alias: true,
            sample_all_layers: false,
            object_finder: true,
            remove_new_layer: false,
            remove_after_each_stroke: true,
            feather: 0.0,
            gradient_style: "linear".into(),
            gradient_reverse: false,
            gradient_dither: true,
            gradient_classic: false,
            gradient_blend_mode: photocraft_color::BlendMode::Normal,
            fill_opacity: 100.0,
            bucket_fill_pattern: false,
            type_font: "Inter".into(),
            type_style: "Regular".into(),
            type_size: 48.0,
            type_aa: "sharp".into(),
            type_align: "left".into(),
            clone_aligned: true,
            pattern_stamp_aligned: true,
            pattern_stamp_impressionist: false,
            pattern_stamp_scale: 100.0,
            pattern_stamp_angle: 0.0,
            clone_sample: "current".into(),
            spot_type: "contentAware".into(),
            patch_mode: "source".into(),
            patch_content_aware: false,
            patch_structure: 4.0,
            patch_color: 0.0,
            cam_mode: "move".into(),
            cam_structure: 4.0,
            cam_color: 0.0,
            cam_transform_on_drop: false,
            tone_range: "midtones".into(),
            exposure: 50.0,
            protect_tones: true,
            sponge_mode: "desaturate".into(),
            vibrance: true,
            strength: 50.0,
            protect_detail: true,
            finger_painting: false,
            enhance_edge: false,
            vector_mode: "path".into(),
            shape_fill: true,
            stroke_width: 0.0,
            shape_stroke: Default::default(),
            corner_radius: 0.0,
            polygon_sides: 5,
            line_weight: 3.0,
            curve_fit: default_curve_fit(),
            pen_auto_add_delete: true,
            marquee_style: default_marquee_style(),
            marquee_width: 1.0,
            marquee_height: 1.0,
            move_auto_select: true,
            move_target: default_move_target(),
            move_show_transform: false,
            crop_ratio: String::new(),
            crop_custom_ratio_width: 0.0,
            crop_custom_ratio_height: 0.0,
            crop_delete: true,
            crop_width: String::new(),
            crop_height: String::new(),
            crop_resolution: String::new(),
            crop_resolution_unit: default_crop_resolution_unit(),
            perspective_crop_grid: true,
            crop_overlay: Default::default(),
            crop_overlay_show: Default::default(),
            crop_overlay_orientation: 0,
            crop_shield: Default::default(),
            magic_eraser_opacity: 100.0,
            bg_sampling: "continuous".into(),
            bg_limits: "contiguous".into(),
            bg_tolerance: 50.0,
            bg_protect_fg: false,
            cr_mode: "color".into(),
            cr_sampling: "continuous".into(),
            cr_limits: "contiguous".into(),
            cr_tolerance: 30.0,
            cr_anti_alias: true,
            zoom_scrubby: true,
            pencil_auto_erase: false,
            brush_clear: false,
            pencil_clear: false,
            eraser_mode: default_eraser_mode(),
            magnetic_width: 10.0,
            magnetic_contrast: 10.0,
            magnetic_frequency: 57.0,
            magnetic_pressure: false,
            red_eye_pupil_size: 50.0,
            red_eye_darken: 50.0,
            eyedropper_size: 1,
            eyedropper_sample: default_eyedropper_sample(),
            eyedropper_ring: true,
        }
    }
}

/// Edit › Transform's mode: what a handle drag does with no modifier keys held.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TransformMode {
    /// Free Transform and Scale: handles scale, drag outside rotates.
    #[default]
    Free,
    /// Edit › Transform › Rotate: handle drags rotate about the reference point (#3941).
    Rotate,
    /// Edge and corner handles skew: each moves along one axis.
    Skew,
    /// Corner handles move freely, one at a time; nothing else moves them (no rotating, no
    /// edges) and nothing snaps.
    Distort,
    /// Corner handles move in pairs, mirrored (one-point perspective).
    Perspective,
}

impl TransformMode {
    /// The mode an `edit.transform.*` / `edit.freeTransform` menu id starts.
    pub fn for_command(id: &str) -> Self {
        match id {
            "edit.transform.rotate" => Self::Rotate,
            "edit.transform.skew" => Self::Skew,
            "edit.transform.distort" => Self::Distort,
            "edit.transform.perspective" => Self::Perspective,
            _ => Self::Free,
        }
    }
}

/// Free Transform in progress: the source frame `rect` and where its corners currently are.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransformSession {
    pub session: u64,
    pub layer: u64,
    pub rect: [f64; 4],
    /// Corners clockwise from top-left (document px).
    pub quad: [[f64; 2]; 4],
    /// Reference point (rotation / ⌥-scale centre).
    pub pivot: [f64; 2],
    pub interpolation: String,
    /// Warp mode (Edit › Transform › Warp): the warp being edited over `rect`, in document px.
    #[serde(default)]
    pub warp: Option<photocraft_geom::warp::Warp>,
    /// Select › Transform Selection: the box transforms the selection outline, not pixels.
    #[serde(default)]
    pub selection: bool,
    /// `edit.transform`'s `"target"` when the box moves an unlinked layer mask, an alpha channel or
    /// the Quick Mask by itself (`None`: the layer, with its linked masks).
    #[serde(default)]
    pub target: Option<serde_json::Value>,
    /// Free Transform Path: `path.transform`'s `name` (and `layer`) when the box moves a
    /// path's anchors and handles instead of pixels.
    #[serde(default)]
    pub path: Option<serde_json::Value>,
    /// The layer was made for this session (⌥⌘T's copy, #352; a file dropped on the canvas), so
    /// Cancel takes it back and OK folds it into one history step with the transform.
    #[serde(default)]
    pub made: Option<MadeLayer>,
    /// Edit › Transform › Rotate / Skew / Distort / Perspective (`Free` for Free Transform and Scale).
    #[serde(default)]
    pub mode: TransformMode,
    /// Edit › Content-Aware Scale (#3096): the box only scales and moves, and OK runs
    /// `edit.contentAwareScale` with these options instead of `edit.transform`.
    #[serde(default)]
    pub content_aware: Option<ContentAwareOptions>,
    /// The options bar's V (vertical skew, degrees) last typed. Any box can be read as scale,
    /// skew and rotation with any vertical skew, so the bar keeps showing the typed one.
    #[serde(default)]
    pub skew_v: f64,
}

/// Content-Aware Scale's options bar: Amount, Protect and Protect Skin Tones.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContentAwareOptions {
    /// How much is content-aware, 0..100 % (the rest is a plain resample).
    pub amount: f32,
    /// The alpha channel whose white areas keep their shape (`None`: no protection).
    #[serde(default)]
    pub protect: Option<String>,
    #[serde(default)]
    pub protect_skin_tones: bool,
}

impl Default for ContentAwareOptions {
    fn default() -> Self {
        Self { amount: 100.0, protect: None, protect_skin_tones: false }
    }
}

/// Why a Free Transform session's layer was made for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MadeLayer {
    /// Free Transform on a copy (⌥⌘T): OK makes the copy and the transform one Free Transform step.
    Copy,
    /// A file dropped on the canvas: OK makes the place and the transform one Place Embedded step.
    Place,
}

/// In-progress inline type editing (Type tool). Offsets are character indices.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextEdit {
    pub layer: u64,
    pub caret: usize,
    pub anchor: usize,
    /// History coalescing key: the whole editing session is one "Edit Type Layer" step.
    pub session: String,
    /// The layer was created by this session (its name follows the text; empty on commit = delete).
    pub created: bool,
    /// A Type Mask tool session: the layer is temporary and commits as a selection.
    #[serde(default)]
    pub mask: bool,
    #[serde(skip)]
    pub dragging: bool,
    /// Paragraph-box handle being dragged (0-3 corners from top-left clockwise, 4-7 top/right/bottom/left edges).
    #[serde(skip)]
    pub resize: Option<u8>,
    /// IME composition in progress: (start, length) in characters. The preedit text lives in the
    /// layer so it lays out like typed text; each IME update replaces it.
    #[serde(skip)]
    pub preedit: Option<(usize, usize)>,
}

/// Temporary Type-tool drag. The document is unchanged until release; this frame is view state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TypeTransform {
    pub document: u64,
    pub revision: u64,
    pub original: photocraft_geom::Affine,
    pub frame: TransformSession,
    pub(crate) gesture: crate::transform_tool::Gesture,
}

/// Content-Aware Move with Transform On Drop: the box on the dropped patch. The document is
/// unchanged until it is committed; this is view state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CamTransform {
    pub document: u64,
    pub revision: u64,
    /// How far the selection was dragged; the box's `rect` is the selection's bounds there.
    pub offset: [i32; 2],
    pub frame: TransformSession,
    #[serde(skip)]
    pub(crate) gesture: Option<crate::transform_tool::Gesture>,
}

/// View-menu overlays.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Extras {
    pub rulers: bool,
    pub grid: bool,
    pub guides: bool,
    pub snap: bool,
    pub lock_guides: bool,
}

impl Default for Extras {
    fn default() -> Self {
        Self { rulers: false, grid: false, guides: true, snap: true, lock_guides: false }
    }
}

/// Selected tab per dock card.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct DockTabs {
    pub properties: usize,
    pub color: usize,
    pub layers: usize,
    /// Navigator | Histogram | Info.
    pub navigator: usize,
    /// History | Actions.
    pub history: usize,
    /// Character | Paragraph.
    pub character: usize,
}

/// Color panel state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ColorPanelState {
    /// The panel edits the background colour (its chip was clicked), not the foreground.
    pub background: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UiState {
    pub tool: Tool,
    /// The last tool used in each shortcut group (the tools sharing a key, [`Tool::key`]), at most
    /// one per group: the group's key brings it back, as in Photoshop (#2608).
    #[serde(default)]
    pub group_tools: Vec<Tool>,
    /// A tool letter still held (spring-loaded tool, #2672): the tool to give back on release.
    /// Exposed to automation, never restored from settings.
    #[serde(default, skip_deserializing)]
    pub spring: Option<crate::spring_tools::Spring>,
    /// Recently opened file paths, most-recent first (File › Open Recent). Capped; de-duplicated.
    #[serde(default)]
    pub recent_files: Vec<String>,
    /// Inline type editing session, if any.
    #[serde(default)]
    pub text_edit: Option<TextEdit>,
    /// Ctrl/Cmd Type-tool gesture, exposed to automation but never restored as a held pointer.
    #[serde(default, skip_deserializing)]
    pub type_transform: Option<TypeTransform>,
    /// Reference point for the current type editing session (document coordinates).
    #[serde(skip)]
    pub type_transform_pivot: Option<[f64; 2]>,
    /// Content-Aware Move's Transform On Drop box, waiting for ↩ / Esc.
    #[serde(default, skip_deserializing)]
    pub cam_transform: Option<CamTransform>,
    /// Free Transform session, if any.
    #[serde(default)]
    pub transform: Option<TransformSession>,
    /// The options bar's Maintain Aspect Ratio link is off: W/H change on their own, and handle
    /// drags stretch freely (⇧ keeps the proportions), as in Photoshop (#4174).
    #[serde(default)]
    pub transform_unlinked: bool,
    /// Temporary symmetry-axis editing, exposed to automation but never restored from settings.
    #[serde(default, skip_deserializing)]
    pub symmetry_transform: Option<crate::symmetry_ui::Transform>,
    /// The Properties panel's armed Levels/Curves eyedropper on an adjustment layer (`layer_picker`),
    /// exposed to automation but never restored from settings.
    #[serde(default, skip_deserializing)]
    pub layer_picker: Option<crate::layer_picker::LayerPicker>,
    /// Clone Stamp / Healing source point (⌥-click) and the aligned offset once a stroke started.
    #[serde(default)]
    pub clone_source: Option<[f64; 2]>,
    #[serde(default)]
    pub clone_offset: Option<[f64; 2]>,
    /// Pattern Stamp aligned origin in document pixels, kept across strokes.
    #[serde(default)]
    pub pattern_stamp_phase: Option<[f32; 2]>,
    /// Painting targets the active layer's mask instead of its pixels.
    #[serde(default)]
    pub mask_target: bool,
    /// The active layer's vector mask is targeted (its Layers thumbnail is bracketed): the path
    /// tools edit it (#196). Never set together with `mask_target`.
    #[serde(default)]
    pub vector_mask_target: bool,
    /// The adjustment or fill layer (id) whose own settings Properties shows although its mask
    /// is targeted: set when that layer is created, as in Photoshop (#3369). Targeting the mask
    /// thumbnail, the layer's pixels or another layer clears it.
    #[serde(default)]
    pub properties_layer: Option<u64>,
    /// Brush Preset picker opened by a right-click on the canvas or the options-bar brush chip:
    /// its screen position (points).
    #[serde(default)]
    pub brush_picker: Option<[f32; 2]>,
    /// The Brush Preset picker's preset list: search, collapsed groups, view, a rename in progress.
    #[serde(default = "crate::brush_picker::list_state")]
    pub brush_picker_list: crate::brush_panel::BrushesPanelState,
    /// The Brush Preset picker's content size, once its corner grip was dragged
    /// ([`crate::brush_picker::DEFAULT_SIZE`] before that).
    #[serde(default)]
    pub brush_picker_size: Option<[f32; 2]>,
    /// Layers under the pointer, listed by a right-click on the canvas with the Move tool or
    /// ⌘/Ctrl+right-click with any tool (`layer_pick_ui`, #307).
    #[serde(default)]
    pub layer_menu: Option<crate::layer_pick_ui::LayerMenu>,
    /// Selection-tool context menu opened by a plain canvas right-click.
    #[serde(default)]
    pub canvas_tool_menu: Option<crate::canvas_tool_menu::CanvasToolMenu>,
    /// The brush is a per-tool option (the Brush, the Eraser and every retouching tool each keep
    /// their own size, hardness, tip and dynamics, as in Photoshop): the tool whose brush the
    /// session brush holds, and the other tools' saved brushes.
    #[serde(default = "default_brush_tool")]
    pub brush_tool: Tool,
    #[serde(default)]
    pub tool_brushes: Vec<(Tool, photocraft_engine::paint::BrushSettings)>,
    /// Pen path under construction.
    #[serde(default)]
    pub pen: Option<crate::vector_ui::PenPath>,
    #[serde(default)]
    pub stroke_editor: Option<crate::shape_stroke_ui::StrokeEditor>,
    /// Direct Selection tool: selected anchors and the drag in progress (#790).
    #[serde(default)]
    pub direct_selection: crate::direct_select::DirectSelection,
    /// Selected row in the Paths panel ("work" or a saved path name).
    #[serde(default)]
    pub selected_path: Option<String>,
    /// Layers panel kind filter ("pixel", "adjustment", "type", "shape", "smart"); empty = all.
    #[serde(default)]
    pub layer_filter: Vec<String>,
    /// Actions panel: which row is selected and which are expanded. The list lives on the session.
    #[serde(default)]
    pub actions: crate::actions::ActionsUi,
    /// Layer Comps panel: the selected comp (by comp id).
    #[serde(default)]
    pub layer_comp_selected: Option<u32>,
    /// Preset panels (Gradients, Patterns, Styles, Shapes, Tool Presets, Clone Source).
    #[serde(default)]
    pub presets_ui: crate::preset_panels::PresetUi,
    /// Character/Paragraph Styles, Glyphs and Check Spelling (see `type_panels_ui`).
    #[serde(default)]
    pub type_panels: crate::type_panels_ui::TypePanelsUi,
    /// Ruler/Count/Note tools, Measurement Log and Notes panels (see `analysis_ui`).
    #[serde(default)]
    pub analysis: crate::analysis_ui::AnalysisUi,
    pub timeline: crate::timeline_ui::TimelineUi,
    /// Slice and Slice Select tools (see `slice_ui`).
    #[serde(default)]
    pub slices: crate::slice_ui::SliceUi,
    /// Frame tool (see `frame_tool`).
    #[serde(default)]
    pub frame_tool: crate::frame_tool::FrameToolUi,
    /// Artboard tool (see `artboard_tool`).
    pub artboard_tool: crate::artboard_tool::ArtboardToolUi,
    /// Modifier Keys panel, custom pixel aspect ratios, workspace dialogs (see `workspace_ui`).
    #[serde(default)]
    pub shell: crate::workspace_ui::ShellUi,
    /// View extras: rulers (⌘R), grid (⌘'), guides (⌘;), snapping (⇧⌘;), locked guides (⌥⌘;).
    #[serde(default)]
    pub extras: Extras,
    /// View / Window / Type preferences: screen mode, Extras, Show and Snap To, flip, arrangement.
    #[serde(default)]
    pub view: crate::view_cmds::ViewOptions,
    pub panels: Panels,
    /// Views per open document (index-aligned with the session's documents).
    pub views: Vec<View>,
    pub dialogs: Vec<Dialog>,
    pub windows: Vec<DocWindow>,
    pub theme: crate::theme::ThemeKind,
    pub workspace: String,
    pub palette_open: bool,
    pub dock_tabs: DockTabs,
    /// Right-dock group order, heights and collapsed groups (see `dock`).
    #[serde(default)]
    pub dock: crate::dock::DockLayout,
    /// Authoritative custom panel membership and placement; legacy fields are compatibility projections.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub docking: Option<craft_ui::docking::Layout<String>>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub docking_hidden: std::collections::BTreeMap<String, craft_ui::docking::Location<String>>,
    /// Invalidates in-flight gestures when a workspace is replaced.
    #[serde(skip)]
    pub docking_generation: u64,
    /// Panels belonging to explicitly collapsed custom tab groups.
    #[serde(default)]
    pub docking_collapsed: Vec<String>,
    /// One-frame focus/raise request from an explicit panel reveal.
    #[serde(skip)]
    pub docking_reveal: Option<String>,
    /// Which chip the Color panel edits.
    #[serde(default)]
    pub color_panel: ColorPanelState,
    /// Brush Settings: selected section (0 = Brush Tip Shape) and tab (0 settings, 1 Brushes).
    #[serde(default)]
    pub brush_section: usize,
    #[serde(default)]
    pub brush_tab: usize,
    /// Brushes panel: collapsed groups and the search filter.
    #[serde(default)]
    pub brushes_panel: crate::brush_panel::BrushesPanelState,
    /// Marquee options-bar mode: 0 new, 1 add, 2 subtract, 3 intersect (modifier keys override).
    #[serde(default)]
    pub selection_mode: u8,
    #[serde(default)]
    pub tool_options: ToolOptions,
    /// In-progress polygonal lasso vertices (document coordinates).
    #[serde(default)]
    pub polygon: Vec<[f64; 2]>,
    /// The selection mode the polygonal lasso started in ("replace", "add", ...), set by the
    /// modifiers held at its first click.
    #[serde(default)]
    pub polygon_mode: String,
    /// The document the polygon is drawn on (set at its first click): another document becoming
    /// active drops it (`canvas::polygon_cancel_stale`).
    #[serde(default)]
    pub polygon_doc: Option<photocraft_doc::DocId>,
    /// Magnetic Lasso border in progress (`magnetic_lasso_ui`).
    #[serde(default)]
    pub magnetic: crate::magnetic_lasso_ui::MagneticLasso,
    /// Crop tool rectangle being edited [x0, y0, x1, y1] (document coordinates).
    #[serde(default)]
    pub crop_rect: Option<[f64; 4]>,
    /// The crop frame's turn in degrees, clockwise on screen about its centre (`crop_rect` is the
    /// frame before the turn); 0 when upright. Committing passes it as `image.crop`'s `angle`.
    #[serde(default)]
    pub crop_angle: f64,
    /// Perspective Crop quad being edited: TL, TR, BR, BL (document coordinates).
    #[serde(default)]
    pub perspective_crop: Option<[[f64; 2]; 4]>,
    pub next_id: u64,
    /// Last status message (errors from commands, hints).
    pub status: String,
    /// The status message is an error (shown in the warning colour).
    #[serde(default)]
    pub status_error: bool,
    /// Non-blocking notices (import/export warnings, files that couldn't open), newest last.
    #[serde(default)]
    pub notices: Vec<crate::notices::Notice>,
    /// Pending GPU fallback warning, visible to automation.
    #[serde(default)]
    pub gpu_fallback_notice: Option<String>,
    /// A keyboard shortcut set found at first launch, waiting for Import / Don't Import.
    #[serde(default)]
    pub kys_offer: Option<crate::kys_import::Offer>,
    /// Documents (ids) whose slow full refresh on the CPU compositor has had its notice.
    #[serde(default)]
    pub slow_refresh_noticed: Vec<u64>,
    /// Status bar info field, Home screen (see `chrome_ui`).
    #[serde(default)]
    pub chrome: crate::chrome_ui::ChromeState,
    #[serde(default)]
    pub camera_raw_scope: CameraRawScopeState,
    #[serde(default)]
    pub camera_raw_preview: CameraRawPreviewState,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            tool: Tool::Brush,
            group_tools: Vec::new(),
            spring: None,
            recent_files: Vec::new(),
            text_edit: None,
            type_transform: None,
            type_transform_pivot: None,
            cam_transform: None,
            transform: None,
            transform_unlinked: false,
            symmetry_transform: None,
            layer_picker: None,
            mask_target: false,
            vector_mask_target: false,
            properties_layer: None,
            brush_picker: None,
            brush_picker_list: crate::brush_picker::list_state(),
            brush_picker_size: None,
            layer_menu: None,
            canvas_tool_menu: None,
            brush_tool: Tool::Brush,
            tool_brushes: Vec::new(),
            clone_source: None,
            clone_offset: None,
            pattern_stamp_phase: None,
            extras: Extras::default(),
            view: Default::default(),
            actions: Default::default(),
            layer_comp_selected: None,
            presets_ui: Default::default(),
            type_panels: Default::default(),
            analysis: Default::default(),
            timeline: Default::default(),
            slices: Default::default(),
            frame_tool: Default::default(),
            artboard_tool: Default::default(),
            shell: Default::default(),
            layer_filter: Vec::new(),
            pen: None,
            stroke_editor: None,
            direct_selection: Default::default(),
            selected_path: None,
            panels: Panels::default(),
            views: Vec::new(),
            dialogs: Vec::new(),
            windows: Vec::new(),
            theme: crate::theme::ThemeKind::ProMedium,
            workspace: "Essentials".into(),
            palette_open: false,
            dock_tabs: DockTabs::default(),
            dock: Default::default(),
            docking: None,
            docking_hidden: Default::default(),
            docking_generation: 0,
            docking_collapsed: Vec::new(),
            docking_reveal: None,
            color_panel: Default::default(),
            brush_section: 0,
            brush_tab: 0,
            brushes_panel: Default::default(),
            selection_mode: 0,
            tool_options: ToolOptions::default(),
            polygon: Vec::new(),
            polygon_mode: String::new(),
            polygon_doc: None,
            magnetic: Default::default(),
            crop_rect: None,
            crop_angle: 0.0,
            perspective_crop: None,
            next_id: 1,
            status: String::new(),
            status_error: false,
            notices: Vec::new(),
            gpu_fallback_notice: None,
            kys_offer: None,
            slow_refresh_noticed: Vec::new(),
            chrome: Default::default(),
            camera_raw_scope: Default::default(),
            camera_raw_preview: Default::default(),
        }
    }
}

impl UiState {
    /// Records the current tool as the last one used in its shortcut group.
    pub fn remember_group_tool(&mut self) {
        let tool = self.tool;
        if !self.group_tools.contains(&tool) {
            self.group_tools.retain(|t| t.key() != tool.key());
            self.group_tools.push(tool);
        }
    }

    /// The last tool used in the shortcut group of `key`, if any.
    pub fn group_tool(&self, key: char) -> Option<Tool> {
        self.group_tools.iter().copied().find(|t| t.key() == key)
    }

    pub fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn open_dialog(&mut self, kind: DialogKind, fields: serde_json::Map<String, serde_json::Value>) -> u64 {
        let id = self.alloc_id();
        self.dialogs.push(Dialog { id, kind, fields });
        id
    }

    pub fn dialog_mut(&mut self, id: u64) -> Option<&mut Dialog> {
        self.dialogs.iter_mut().find(|d| d.id == id)
    }

    pub fn close_dialog(&mut self, id: u64) -> Option<Dialog> {
        let i = self.dialogs.iter().position(|d| d.id == id)?;
        Some(self.dialogs.remove(i))
    }

    pub fn new_document_fields() -> serde_json::Map<String, serde_json::Value> {
        let v = serde_json::json!({"name": "Untitled-1", "width": 1920, "height": 1080, "mode": "rgb", "depth": 8, "background": "white"});
        v.as_object().cloned().unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_names_parse() {
        assert_eq!(Tool::from_name("brush"), Some(Tool::Brush));
        assert_eq!(Tool::from_name("Rect"), Some(Tool::RectMarquee));
        assert_eq!(Tool::from_name("RectMarquee"), Some(Tool::RectMarquee));
        assert_eq!(Tool::from_name("Eraser Tool"), Some(Tool::Eraser));
        assert_eq!(Tool::from_name("mixerBrush"), Some(Tool::MixerBrush));
        assert_eq!(Tool::from_name("Mixer Brush Tool"), Some(Tool::MixerBrush));
        assert_eq!(Tool::from_name("colorReplacement"), Some(Tool::ColorReplacement));
        assert_eq!(Tool::from_name("Color Replacement Tool"), Some(Tool::ColorReplacement));
        assert_eq!(Tool::from_name("redEye"), Some(Tool::RedEye));
        assert_eq!(Tool::from_name("Red Eye Tool"), Some(Tool::RedEye));
        assert_eq!(Tool::RedEye.key(), 'J');
        assert!(!Tool::RedEye.is_brushlike());
        assert_eq!(Tool::from_name("patternStamp"), Some(Tool::PatternStamp));
        assert_eq!(Tool::from_name("Pattern Stamp Tool"), Some(Tool::PatternStamp));
        assert_eq!(Tool::ALL.len(), 67);
        assert_eq!(Tool::from_name("Vertical Type Mask Tool"), Some(Tool::VerticalTypeMask));
        assert_eq!(Tool::from_name("addAnchorPoint"), Some(Tool::AddAnchorPoint));
        assert_eq!(Tool::from_name("Delete Anchor Point Tool"), Some(Tool::DeleteAnchorPoint));
        assert_eq!(Tool::DeleteAnchorPoint.key(), 'P');
        assert_eq!(Tool::from_name("convertPoint"), Some(Tool::ConvertPoint));
        assert_eq!(Tool::from_name("Convert Point Tool"), Some(Tool::ConvertPoint));
        assert_eq!(Tool::ConvertPoint.key(), 'P', "it shares the Pen's key");
        assert!(!Tool::ConvertPoint.is_brushlike());
        assert_eq!(Tool::from_name("Remove Tool"), Some(Tool::Remove));
        assert_eq!(Tool::Remove.key(), 'J');
        assert!(Tool::Remove.is_brushlike());
        assert_eq!(Tool::from_name("RotateView"), Some(Tool::RotateView));
        assert_eq!(Tool::from_name("Rotate View Tool"), Some(Tool::RotateView));
        assert_eq!(Tool::RotateView.key(), 'R');
        assert_eq!(Tool::from_name("nope"), None);
    }

    /// The README's "Everything in the box" tool heading and list name exactly the tools a user can
    /// pick, `Tool::ALL` (#996).
    #[test]
    fn readme_tool_list_matches_tool_all() {
        let readme = include_str!("../../../README.md");
        let (_, rest) = readme.split_once("<h4>🧰 ").unwrap();
        let (count, rest) = rest.split_once(" tools</h4>").unwrap();
        assert_eq!(count.parse::<usize>().unwrap(), Tool::ALL.len(), "README.md tool heading");
        let list = rest.trim_start().lines().next().unwrap();
        // "Rectangular and Elliptical Marquee" names two tools: "Rectangular Marquee", "Elliptical Marquee".
        let mut named: Vec<String> = Vec::new();
        for entry in list.split(" · ") {
            match entry.split_once(" and ").and_then(|(first, rest)| Some((first, rest.split_once(' ')?))) {
                Some((first, (second, noun))) => named.extend([format!("{first} {noun}"), format!("{second} {noun}")]),
                None => named.push(entry.to_owned()),
            }
        }
        named.sort();
        let mut labels: Vec<String> = Tool::ALL.iter().map(|tool| tool.label().trim_end_matches(" Tool").to_owned()).collect();
        labels.sort();
        assert_eq!(named, labels, "README.md tool list");
    }

    #[test]
    fn dialogs_open_and_close() {
        let mut s = UiState::default();
        let a = s.open_dialog(DialogKind::About, Default::default());
        let b = s.open_dialog(DialogKind::NewDocument, UiState::new_document_fields());
        assert_ne!(a, b);
        assert_eq!(s.dialog_mut(b).unwrap().fields["width"], 1920);
        assert!(s.close_dialog(a).is_some());
        assert_eq!(s.dialogs.len(), 1);
    }

    #[test]
    fn ui_state_serializes() {
        let s = UiState::default();
        let v = serde_json::to_value(&s).unwrap();
        assert_eq!(v["tool"], "Brush");
        let back: UiState = serde_json::from_value(v).unwrap();
        assert_eq!(back, s);
    }
}
