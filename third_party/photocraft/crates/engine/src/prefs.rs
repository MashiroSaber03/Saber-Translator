//! Persistent preferences (Edit › Preferences), keyboard-shortcut, menu and toolbar
//! customisation, grouped like Photoshop's Preferences dialog sections.
//!
//! The whole set is plain serde data owned by the [`Session`] ([`Session::prefs`]), so every
//! frontend reads the same values and agents change them with `prefs.get` / `prefs.set` /
//! `prefs.reset`. Sections that only the GUI shell honours (cursors, checkerboard, guide colours,
//! interface theme…) live here too: keeping one document means the headless MCP server sees and
//! edits exactly what the desktop app persists, and the shell has no second store to keep in sync.
//!
//! The engine never touches the filesystem for preferences (it must build for wasm). Frontends
//! persist [`Session::prefs_to_json`] wherever the platform keeps settings (the desktop app: the
//! OS config directory; the web shell: browser storage) and restore it with
//! [`Session::load_prefs_json`].
//!
//! Engine-side values are applied by [`Session::apply_prefs`]: the history state limit (every
//! open document), the effect-cache memory budget, and units for commands that report lengths.
//! Colour settings (`colorSettings`) are stored with the colour state
//! ([`crate::color_cmds::ColorSettings`]) and persisted in the same document.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

/// A string-valued choice enum with its JSON names (used to validate `prefs.set` and to fill
/// dropdowns in the Preferences dialog).
macro_rules! choice {
    ($(#[$m:meta])* $name:ident { $($var:ident = $s:literal),+ $(,)? } default $def:ident) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub enum $name { $(#[serde(rename = $s)] $var),+ }
        impl Default for $name {
            fn default() -> Self { Self::$def }
        }
        impl $name {
            pub const NAMES: &'static [&'static str] = &[$($s),+];
            pub fn name(self) -> &'static str {
                match self { $(Self::$var => $s),+ }
            }
            pub fn parse(s: &str) -> Option<Self> {
                match s { $($s => Some(Self::$var),)+ _ => None }
            }
        }
    };
}

choice!(
    /// Length units for rulers, the Info panel and size dialogs.
    Unit { Pixels = "pixels", Inches = "inches", Centimeters = "cm", Millimeters = "mm", Points = "points", Picas = "picas", Percent = "percent" } default Pixels
);
choice!(TypeUnit { Points = "points", Pixels = "pixels", Millimeters = "mm" } default Points);
choice!(PointSize { PostScript = "postScript", Traditional = "traditional" } default PostScript);
choice!(Interpolation { BicubicAutomatic = "bicubicAutomatic", Nearest = "nearestNeighbor", Bilinear = "bilinear", Bicubic = "bicubic", BicubicSmoother = "bicubicSmoother", BicubicSharper = "bicubicSharper", PreserveDetails = "preserveDetails" } default BicubicAutomatic);
choice!(ColorPicker { Adobe = "adobe", System = "system" } default Adobe);
choice!(Theme { Pro = "pro", ProMedium = "proMedium", ProLightGray = "proLightGray", ProLight = "proLight", Studio = "studio", StudioLight = "studioLight", Classic = "classic", SolarizedDark = "solarizedDark", Adwaita = "adwaita", AdwaitaDark = "adwaitaDark", BreezeLight = "breezeLight", BreezeDark = "breezeDark", SoftDark = "softDark", SoftLight = "softLight" } default ProMedium);
choice!(AppearanceMode { Auto = "auto", Dark = "dark", Light = "light" } default Dark);
choice!(DarkTheme { Pro = "pro", ProMedium = "proMedium", Studio = "studio", SolarizedDark = "solarizedDark", AdwaitaDark = "adwaitaDark", BreezeDark = "breezeDark", SoftDark = "softDark" } default ProMedium);
choice!(LightTheme { ProLight = "proLight", ProLightGray = "proLightGray", StudioLight = "studioLight", Classic = "classic", Adwaita = "adwaita", BreezeLight = "breezeLight", SoftLight = "softLight" } default ProLight);
choice!(CanvasColor { Default = "default", Black = "black", DarkGray = "darkGray", MediumGray = "mediumGray", LightGray = "lightGray", Custom = "custom" } default Default);
choice!(CanvasBorder { DropShadow = "dropShadow", Line = "line", None = "none" } default DropShadow);
choice!(UiScale { Auto = "auto", P75 = "75", P80 = "80", P85 = "85", P90 = "90", P95 = "95", P100 = "100", P125 = "125", P150 = "150", P175 = "175", P200 = "200", P250 = "250", P300 = "300" } default Auto);
choice!(
    /// Graphics backend of the desktop app's window and GPU canvas (applies at next launch).
    /// `auto` lets PhotoCraft pick (DX12 for Intel adapters on Windows); `cpu` composites on the
    /// CPU and draws the window with a software adapter where the platform has one. A start that
    /// crashes inside the graphics driver moves this to the next safer choice.
    GpuBackend { Auto = "auto", Vulkan = "vulkan", Dx12 = "dx12", Metal = "metal", Gl = "gl", Cpu = "cpu" } default Auto
);
choice!(
    /// Rendering policy, independent of the advanced graphics backend selection.
    /// CPU disables image acceleration; the native window may still need hardware graphics.
    RenderingMode { Auto = "auto", Gpu = "gpu", Cpu = "cpu" } default Auto
);
choice!(
    /// Linux display server of the desktop app's window (applies at next launch). `x11` runs
    /// PhotoCraft through XWayland on a Wayland session, where native file drag and drop works
    /// (winit 0.30 has none on Wayland, issue #386); fractional scaling may then look softer.
    /// It needs an X server (`DISPLAY`); without one PhotoCraft starts as `auto`. Other
    /// platforms ignore it.
    LinuxDisplayServer { Auto = "auto", X11 = "x11" } default Auto
);
choice!(UiFontSize { Tiny = "tiny", Small = "small", Medium = "medium", Large = "large" } default Small);
choice!(LogDestination { Metadata = "metadata", TextFile = "textFile", Both = "both" } default Metadata);
choice!(LogDetail { SessionsOnly = "sessionsOnly", Concise = "concise", Detailed = "detailed" } default Concise);
choice!(Ask { Always = "always", Never = "never", Ask = "ask" } default Always);
choice!(QuickExportFormat { Png = "png", Jpg = "jpg", Gif = "gif", Webp = "webp" } default Png);
choice!(ExportLocation { Ask = "ask", SameFolder = "sameFolder" } default Ask);
choice!(ExportMetadata { None = "none", Copyright = "copyright", All = "all" } default Copyright);
choice!(
    /// Painting cursors: Standard (tool icon), Precise (crosshair), Normal Brush Tip (the 50%
    /// opacity contour), Full Size Brush Tip (the whole diameter).
    PaintingCursor { Standard = "standard", Precise = "precise", NormalTip = "normalTip", FullSizeTip = "fullSizeTip" } default NormalTip
);
choice!(
    /// Right mouse button on the canvas with a painting tool: open the Brush Preset picker at the
    /// pointer (Photoshop), or erase with the current brush while dragging (Krita, Paint).
    RightClickPaint { BrushPicker = "brushPicker", Erase = "erase" } default BrushPicker
);
choice!(OtherCursor { Standard = "standard", Precise = "precise" } default Standard);
choice!(CheckerSize { None = "none", Small = "small", Medium = "medium", Large = "large" } default Medium);
choice!(CheckerColors { Light = "light", Medium = "medium", Dark = "dark", Red = "red", Orange = "orange", Green = "green", Blue = "blue", Purple = "purple", Custom = "custom" } default Light);
choice!(LineStyle { Lines = "lines", Dashed = "dashedLines", Dots = "dots" } default Lines);
choice!(TextEngine { Modern = "modern", EastAsian = "eastAsian", WorldReady = "worldReady" } default WorldReady);
choice!(FontPreview { Off = "off", Small = "small", Medium = "medium", Large = "large", ExtraLarge = "extraLarge", Huge = "huge" } default Medium);
choice!(RawColorSpace { Srgb = "srgb", AdobeRgb = "adobeRgb", ProPhoto = "proPhoto", DisplayP3 = "displayP3" } default AdobeRgb);
choice!(RawDepth { Eight = "8", Sixteen = "16" } default Sixteen);
choice!(RawSharpen { None = "none", Screen = "screen", Glossy = "glossy", Matte = "matte" } default None);

impl Unit {
    /// Document pixels → this unit. `dpi` is the document resolution, `extent` the 100% length
    /// (for percent), `ppi` points per inch (72 PostScript, 72.27 traditional).
    pub fn from_px(self, px: f64, dpi: f64, extent: f64, ppi: f64) -> f64 {
        let dpi = dpi.max(1e-6);
        match self {
            Unit::Pixels => px,
            Unit::Inches => px / dpi,
            Unit::Centimeters => px / dpi * 2.54,
            Unit::Millimeters => px / dpi * 25.4,
            Unit::Points => px / dpi * ppi,
            Unit::Picas => px / dpi * ppi / 12.0,
            Unit::Percent => px / extent.max(1e-6) * 100.0,
        }
    }
    /// This unit → document pixels (inverse of [`Unit::from_px`]).
    pub fn to_px(self, v: f64, dpi: f64, extent: f64, ppi: f64) -> f64 {
        let dpi = dpi.max(1e-6);
        match self {
            Unit::Pixels => v,
            Unit::Inches => v * dpi,
            Unit::Centimeters => v / 2.54 * dpi,
            Unit::Millimeters => v / 25.4 * dpi,
            Unit::Points => v / ppi.max(1e-6) * dpi,
            Unit::Picas => v * 12.0 / ppi.max(1e-6) * dpi,
            Unit::Percent => v * extent / 100.0,
        }
    }
    /// Short suffix for readouts ("px", "in", "cm", …).
    pub fn suffix(self) -> &'static str {
        match self {
            Unit::Pixels => "px",
            Unit::Inches => "in",
            Unit::Centimeters => "cm",
            Unit::Millimeters => "mm",
            Unit::Points => "pt",
            Unit::Picas => "pica",
            Unit::Percent => "%",
        }
    }
    /// Decimal places a readout in this unit needs, following Photoshop: whole pixels, one place
    /// for points and percent, two for millimetres, centimetres and picas, three for inches
    /// (#2434). [`fmt_decimals`] rounds to this many places and drops trailing zeros.
    pub fn decimals(self) -> usize {
        match self {
            Unit::Pixels => 0,
            Unit::Points | Unit::Percent => 1,
            Unit::Millimeters | Unit::Centimeters | Unit::Picas => 2,
            Unit::Inches => 3,
        }
    }
    /// A length as typed in Photoshop's numeric fields: a number with an optional unit suffix
    /// ("50%", "2.5 cm", "12pt", "3pc", "1,5 in", "100"). Returns the number and its unit (`None`
    /// when there is no suffix, so the caller picks the default), or `None` if the text is not a
    /// finite number followed by a known unit.
    pub fn parse_length(text: &str) -> Option<(f64, Option<Unit>)> {
        let t = text.trim();
        let end = t.find(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | ',' | '-' | '+'))).unwrap_or(t.len());
        let (number, suffix) = t.split_at(end);
        let v: f64 = number.trim().replace(',', ".").parse().ok()?;
        if !v.is_finite() {
            return None;
        }
        let suffix = suffix.trim();
        let unit = if suffix.is_empty() { None } else { Some(Unit::from_suffix(suffix)?) };
        Some((v, unit))
    }
    /// The unit a typed suffix names, in any case: `px`, `in` or `"`, `cm`, `mm`, `pt`, `pc` or
    /// `pica`, `%`, and their long forms (`pixels`, `inches`, `centimetres`…). The one table every
    /// length field reads.
    pub fn from_suffix(suffix: &str) -> Option<Unit> {
        Some(match suffix.to_ascii_lowercase().as_str() {
            "px" | "pixel" | "pixels" => Unit::Pixels,
            "in" | "inch" | "inches" | "\"" => Unit::Inches,
            "cm" | "centimeter" | "centimeters" | "centimetre" | "centimetres" => Unit::Centimeters,
            "mm" | "millimeter" | "millimeters" | "millimetre" | "millimetres" => Unit::Millimeters,
            "pt" | "point" | "points" => Unit::Points,
            "pc" | "pica" | "picas" => Unit::Picas,
            "%" | "percent" => Unit::Percent,
            _ => return None,
        })
    }
}

impl PointSize {
    /// Points per inch.
    pub fn per_inch(self) -> f64 {
        match self {
            PointSize::PostScript => 72.0,
            PointSize::Traditional => 72.27,
        }
    }
}

// ------------------------------------------------------------------ sections

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct General {
    pub color_picker: ColorPicker,
    /// Default resampling for Image Size and transforms.
    pub image_interpolation: Interpolation,
    pub beep_when_done: bool,
    pub export_clipboard: bool,
    pub resize_image_during_place: bool,
    pub always_create_smart_objects_when_placing: bool,
    /// Scroll wheel zooms instead of panning.
    pub zoom_with_scroll_wheel: bool,
    pub animated_zoom: bool,
    pub zoom_resizes_windows: bool,
    pub use_legacy_free_transform: bool,
    pub auto_show_home_screen: bool,
}

impl Default for General {
    fn default() -> Self {
        Self {
            color_picker: ColorPicker::Adobe,
            image_interpolation: Interpolation::BicubicAutomatic,
            beep_when_done: false,
            export_clipboard: true,
            resize_image_during_place: true,
            always_create_smart_objects_when_placing: true,
            zoom_with_scroll_wheel: false,
            animated_zoom: true,
            zoom_resizes_windows: false,
            use_legacy_free_transform: false,
            auto_show_home_screen: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Interface {
    /// Legacy single-theme preference, retained for old automation and saved settings.
    pub theme: Theme,
    pub appearance_mode: AppearanceMode,
    pub dark_theme: DarkTheme,
    pub light_theme: LightTheme,
    /// Theme shortcuts shown in the appearance button's context menu.
    pub favorite_themes: Vec<Theme>,
    /// Pasteboard colour in standard screen mode (`canvasCustomColor` when "custom").
    pub canvas_color: CanvasColor,
    pub canvas_custom_color: String,
    /// Show the repeating dot pattern on the canvas surround (including the empty workspace).
    /// Disable for a uniform background, for example on OLED displays (#2264).
    pub show_canvas_dots: bool,
    /// Document border in standard screen mode.
    pub canvas_border: CanvasBorder,
    pub ui_scale: UiScale,
    /// UI language: `auto` (follow the system) or a language code such as `en`, `ja`. The list of
    /// languages belongs to the shell (`ui-egui` i18n); an unknown code falls back to `auto`.
    /// Command ids and the control protocol stay English.
    pub language: String,
    pub ui_font_size: UiFontSize,
    pub show_channels_in_color: bool,
    pub dynamic_color_sliders: bool,
    /// Draw menu item colours set with Edit › Menus.
    pub show_menu_colors: bool,
    pub show_tooltips: bool,
    /// Menus, popups and collapsing sections fade and slide, and scroll areas glide to a new
    /// position. Off, everything changes at once (a reduced-motion mode; #1218, #3273).
    pub enable_animations: bool,
    /// Move tool drags show only the layer's outline and an arrow, leaving its pixels in place
    /// until release. Off (the default), the pixels follow the pointer live inside the outline.
    pub show_bounding_box_when_dragging_layer: bool,
    /// Windows and Linux: use the system's title bar and window buttons instead of PhotoCraft's
    /// own one-row title bar (tiling window managers, desktops that draw their own decorations;
    /// #1271, #1316). Read when the app starts. macOS always uses the system's.
    pub system_title_bar: bool,
    /// Notices (the lower-right cards) hide themselves after
    /// [`Interface::notification_duration_seconds`] unless the pointer rests on them (#2022).
    pub notification_auto_hide: bool,
    /// Seconds a notice stays on screen before it hides itself (Auto Hide Notifications).
    pub notification_duration_seconds: u32,
}

impl Default for Interface {
    fn default() -> Self {
        Self {
            theme: Theme::ProMedium,
            appearance_mode: AppearanceMode::Dark,
            dark_theme: DarkTheme::ProMedium,
            light_theme: LightTheme::ProLight,
            favorite_themes: Vec::new(),
            canvas_color: CanvasColor::Default,
            canvas_custom_color: "#282828".into(),
            show_canvas_dots: true,
            canvas_border: CanvasBorder::DropShadow,
            ui_scale: UiScale::Auto,
            language: "auto".into(),
            ui_font_size: UiFontSize::Small,
            show_channels_in_color: false,
            dynamic_color_sliders: true,
            show_menu_colors: true,
            show_tooltips: true,
            enable_animations: true,
            show_bounding_box_when_dragging_layer: false,
            system_title_bar: false,
            notification_auto_hide: true,
            notification_duration_seconds: 6,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Workspace {
    pub auto_collapse_icon_panels: bool,
    pub auto_show_hidden_panels: bool,
    pub open_documents_as_tabs: bool,
    pub enable_floating_document_window_docking: bool,
    pub large_tabs: bool,
    pub enable_narrow_options_bar: bool,
    pub remember_workspace_changes: bool,
}

impl Default for Workspace {
    fn default() -> Self {
        Self {
            auto_collapse_icon_panels: false,
            auto_show_hidden_panels: true,
            open_documents_as_tabs: true,
            enable_floating_document_window_docking: true,
            large_tabs: false,
            enable_narrow_options_bar: false,
            remember_workspace_changes: true,
        }
    }
}

impl Workspace {
    /// Preferences ▸ Workspace ▸ Enable Narrow Options Bar: the options bar drops to its compact
    /// height, as in Photoshop, so the document gets more room (see `panels::options_bar`).
    pub fn narrow_options_bar(&self) -> bool {
        self.enable_narrow_options_bar
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Tools {
    pub show_tooltips: bool,
    /// Tool keys cycle a tool group only with ⇧ held (Photoshop's "Use Shift Key for Tool Switch", on by default there too).
    pub use_shift_key_for_tool_switch: bool,
    pub zoom_clicked_point_to_center: bool,
    pub enable_flick_panning: bool,
    pub vary_round_brush_hardness_on_hud: bool,
    /// Round snapped vector and transform coordinates to whole pixels.
    pub snap_vector_tools_and_transforms_to_pixel_grid: bool,
    /// The size and angle readouts beside the pointer while a marquee or crop frame is drawn,
    /// resized or turned (off: none).
    pub show_transformation_values: bool,
    /// Move tool with Auto-Select: outline the layer a click would pick under the pointer.
    pub show_highlight_on_rollover: bool,
    pub overscroll: bool,
    pub double_click_layer_mask_launches_select_and_mask: bool,
    /// What the right mouse button does on the canvas with the Brush and other painting tools.
    pub right_click_with_painting_tools: RightClickPaint,
    /// Pen tablets: pressure, tilt and rotation reach the brush (off: a pen paints like a mouse).
    pub use_tablet_pressure: bool,
    /// Pen pressure response: `[input, output]` control points in 0..1 that the pen's pressure
    /// passes through before it reaches any brush ([`PressureCurve`]). Linear by default.
    pub pressure_curve: Vec<[f32; 2]>,
}

impl Default for Tools {
    fn default() -> Self {
        Self {
            show_tooltips: true,
            use_shift_key_for_tool_switch: true,
            zoom_clicked_point_to_center: false,
            enable_flick_panning: true,
            vary_round_brush_hardness_on_hud: true,
            snap_vector_tools_and_transforms_to_pixel_grid: true,
            show_transformation_values: true,
            show_highlight_on_rollover: true,
            overscroll: true,
            double_click_layer_mask_launches_select_and_mask: true,
            right_click_with_painting_tools: RightClickPaint::BrushPicker,
            use_tablet_pressure: true,
            pressure_curve: vec![[0.0, 0.0], [1.0, 1.0]],
        }
    }
}

/// Preferences › Tools › Pressure Curve, ready to evaluate: a monotone cubic (Fritsch–Carlson)
/// through the control points, so a firmer press never gives less pressure. The points are
/// sanitised here, not trusted: a whole-section update or a hand-edited preferences file can store
/// anything. Non-finite points are dropped, the rest clamped to 0..1, sorted by input (one point
/// per input), capped at [`Self::MAX_POINTS`], and their outputs made non-decreasing; before the
/// first point and after the last the curve stays flat. Fewer than two usable points is linear.
#[derive(Clone, Debug, PartialEq)]
pub struct PressureCurve {
    xs: Vec<f32>,
    ys: Vec<f32>,
    /// Tangents at the points.
    ms: Vec<f32>,
}

impl PressureCurve {
    pub const MAX_POINTS: usize = 16;

    pub fn new(points: &[[f32; 2]]) -> Self {
        let mut pts: Vec<[f32; 2]> =
            points.iter().filter(|p| p[0].is_finite() && p[1].is_finite()).map(|p| [p[0].clamp(0.0, 1.0), p[1].clamp(0.0, 1.0)]).collect();
        pts.sort_by(|a, b| a[0].total_cmp(&b[0]));
        pts.dedup_by(|b, a| {
            // Same input: keep the later point's output.
            let same = (a[0] - b[0]).abs() < 1e-6;
            if same {
                a[1] = b[1];
            }
            same
        });
        pts.truncate(Self::MAX_POINTS);
        if pts.len() < 2 {
            pts = vec![[0.0, 0.0], [1.0, 1.0]];
        }
        let xs: Vec<f32> = pts.iter().map(|p| p[0]).collect();
        let mut ys: Vec<f32> = pts.iter().map(|p| p[1]).collect();
        for i in 1..ys.len() {
            ys[i] = ys[i].max(ys[i - 1]);
        }
        // Fritsch–Carlson tangents: secant averages, zero at flat steps, limited so the cubic
        // never overshoots (which keeps it monotone).
        let n = xs.len();
        let d: Vec<f32> = (0..n - 1).map(|k| (ys[k + 1] - ys[k]) / (xs[k + 1] - xs[k])).collect();
        let mut ms = vec![0.0f32; n];
        ms[0] = d[0];
        ms[n - 1] = d[n - 2];
        for k in 1..n - 1 {
            ms[k] = if d[k - 1] * d[k] <= 0.0 { 0.0 } else { (d[k - 1] + d[k]) / 2.0 };
        }
        for k in 0..n - 1 {
            if d[k] == 0.0 {
                ms[k] = 0.0;
                ms[k + 1] = 0.0;
                continue;
            }
            let (a, b) = (ms[k] / d[k], ms[k + 1] / d[k]);
            let s = a * a + b * b;
            if s > 9.0 {
                let t = 3.0 / s.sqrt();
                ms[k] = t * a * d[k];
                ms[k + 1] = t * b * d[k];
            }
        }
        Self { xs, ys, ms }
    }

    /// The curve's output for pen pressure `x` (0..1; anything else is clamped, NaN reads as 0).
    pub fn eval(&self, x: f32) -> f32 {
        let x = if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) };
        let n = self.xs.len();
        let (Some(&x0), Some(&xn)) = (self.xs.first(), self.xs.last()) else { return x };
        if x <= x0 {
            return self.ys.first().copied().unwrap_or(x);
        }
        if x >= xn {
            return self.ys.last().copied().unwrap_or(x);
        }
        let k = self.xs.partition_point(|&v| v <= x).saturating_sub(1).min(n.saturating_sub(2));
        let (Some(&xa), Some(&xb), Some(&ya), Some(&yb), Some(&ma), Some(&mb)) =
            (self.xs.get(k), self.xs.get(k + 1), self.ys.get(k), self.ys.get(k + 1), self.ms.get(k), self.ms.get(k + 1))
        else {
            return x;
        };
        let h = xb - xa;
        let t = (x - xa) / h;
        let (t2, t3) = (t * t, t * t * t);
        let y = (2.0 * t3 - 3.0 * t2 + 1.0) * ya + (t3 - 2.0 * t2 + t) * h * ma + (-2.0 * t3 + 3.0 * t2) * yb + (t3 - t2) * h * mb;
        y.clamp(ya.min(yb), ya.max(yb))
    }

    /// The identity (the default curve): pressure passes through unchanged.
    pub fn is_linear(&self) -> bool {
        self.xs == [0.0, 1.0] && self.ys == [0.0, 1.0]
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct HistoryLog {
    pub enabled: bool,
    pub destination: LogDestination,
    /// Text file the log is appended to (destination textFile/both).
    pub file_path: String,
    pub detail: LogDetail,
}

impl Default for HistoryLog {
    fn default() -> Self {
        Self { enabled: false, destination: LogDestination::Metadata, file_path: String::new(), detail: LogDetail::Concise }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct FileHandling {
    pub image_previews: Ask,
    pub lowercase_extension: bool,
    pub save_in_background: bool,
    /// Save recovery information (crash recovery autosave).
    pub autosave: bool,
    /// Minutes between autosaves of documents with unsaved changes.
    pub autosave_minutes: u32,
    /// Offer to reopen recovered documents at launch.
    pub recover_on_launch: bool,
    /// Open a file with no ICC profile as untagged even when its EXIF ColorSpace tag says sRGB.
    pub ignore_exif_profile_tag: bool,
    pub ask_before_saving_layered_tiff: bool,
    pub maximize_psd_compatibility: Ask,
    /// SVG groups deeper than this are rasterised on import; parser and document safety caps remain fixed.
    pub rasterize_svg_groups_deeper_than: u32,
    pub recent_file_count: u32,
    /// Most recently opened files, newest first (File › Open Recent).
    pub recent_files: Vec<String>,
}

impl Default for FileHandling {
    fn default() -> Self {
        Self {
            image_previews: Ask::Always,
            lowercase_extension: true,
            save_in_background: true,
            autosave: true,
            autosave_minutes: 10,
            recover_on_launch: true,
            ignore_exif_profile_tag: false,
            ask_before_saving_layered_tiff: true,
            maximize_psd_compatibility: Ask::Always,
            rasterize_svg_groups_deeper_than: photocraft_doc::MAX_GROUP_DEPTH as u32,
            recent_file_count: 20,
            recent_files: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Export {
    pub quick_export_format: QuickExportFormat,
    pub quick_export_location: ExportLocation,
    pub jpeg_quality: u32,
    /// Keep the existing lossless Quick Export default until the user opts into lossy WebP.
    pub webp_lossless: bool,
    pub webp_quality: u32,
    pub metadata: ExportMetadata,
    pub convert_to_srgb: bool,
}

impl Default for Export {
    fn default() -> Self {
        Self {
            quick_export_format: QuickExportFormat::Png,
            quick_export_location: ExportLocation::Ask,
            jpeg_quality: 85,
            webp_lossless: true,
            webp_quality: 85,
            metadata: ExportMetadata::Copyright,
            convert_to_srgb: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Performance {
    /// Memory PhotoCraft may use, in MB: bounds each document's pixels plus its History (the
    /// oldest states are dropped beyond it).
    pub memory_usage_mb: u32,
    /// Undo steps kept per document (History panel states).
    pub history_states: u32,
    /// Display cache levels (mip levels kept for zoomed-out views).
    pub cache_levels: u32,
    /// Side of the GPU canvas texture tiles, in pixels (applies at next launch).
    pub cache_tile_size: u32,
    /// Draw the canvas with the GPU (applies at next launch).
    pub use_gpu: bool,
    /// Explicit rendering policy. None preserves older useGpu/gpuBackend preferences.
    pub rendering_mode: Option<RenderingMode>,
    /// Graphics backend (applies at next launch; see [`GpuBackend`]).
    pub gpu_backend: GpuBackend,
    /// Live previews of large documents (adjustment and filter dialogs, an adjustment layer's
    /// sliders while they drag) render on a reduced copy: fast, but blocky when zoomed in. Off:
    /// they render at full resolution.
    pub low_resolution_previews: bool,
    /// Linux display server (applies at next launch; see [`LinuxDisplayServer`]).
    pub linux_display_server: LinuxDisplayServer,
    /// Memory budget of the layer-effect cache, in MB.
    pub effect_cache_mb: u32,
    pub legacy_compositing: bool,
}

impl Performance {
    /// Resolve old preferences without allowing legacy flags to override an explicit mode.
    pub fn effective_rendering_mode(&self) -> RenderingMode {
        self.rendering_mode.unwrap_or_else(|| if !self.use_gpu || self.gpu_backend == GpuBackend::Cpu { RenderingMode::Cpu } else { RenderingMode::Auto })
    }

    /// Pixel memory a document and its History may hold (Memory Usage), in bytes: beyond it
    /// the oldest history states are dropped.
    pub fn history_budget_bytes(&self) -> usize {
        (self.memory_usage_mb as usize).saturating_mul(1 << 20)
    }
}

impl Default for Performance {
    fn default() -> Self {
        Self {
            memory_usage_mb: 8192,
            history_states: 50,
            cache_levels: 4,
            cache_tile_size: 8192,
            use_gpu: true,
            rendering_mode: None,
            gpu_backend: GpuBackend::Auto,
            low_resolution_previews: true,
            linux_display_server: LinuxDisplayServer::Auto,
            effect_cache_mb: 768,
            legacy_compositing: false,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ScratchDisk {
    pub path: String,
    pub enabled: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ScratchDisks {
    /// Disks for spilling tiles once the memory budget is reached (in priority order).
    pub disks: Vec<ScratchDisk>,
}

impl Default for ScratchDisks {
    fn default() -> Self {
        Self { disks: vec![ScratchDisk { path: "(system temp)".into(), enabled: true }] }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Cursors {
    pub painting: PaintingCursor,
    pub show_crosshair_in_brush_tip: bool,
    pub show_only_crosshair_while_painting: bool,
    pub other: OtherCursor,
    /// Brush preview (HUD) colour.
    pub brush_preview_color: String,
}

impl Default for Cursors {
    fn default() -> Self {
        Self {
            painting: PaintingCursor::NormalTip,
            show_crosshair_in_brush_tip: false,
            show_only_crosshair_while_painting: false,
            other: OtherCursor::Standard,
            brush_preview_color: "#ff0000".into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct TransparencyAndGamut {
    pub grid_size: CheckerSize,
    pub grid_colors: CheckerColors,
    /// Checkerboard colours used when `gridColors` is "custom".
    pub custom_light: String,
    pub custom_dark: String,
    pub gamut_warning_color: String,
    /// Gamut warning opacity, percent.
    pub gamut_warning_opacity: u32,
}

impl Default for TransparencyAndGamut {
    fn default() -> Self {
        Self {
            grid_size: CheckerSize::Medium,
            grid_colors: CheckerColors::Light,
            custom_light: "#ffffff".into(),
            custom_dark: "#cccccc".into(),
            gamut_warning_color: "#808080".into(),
            gamut_warning_opacity: 100,
        }
    }
}

impl TransparencyAndGamut {
    /// Checker square side in screen points (`None`: no checkerboard, transparency shows white).
    pub fn square(&self) -> Option<f32> {
        match self.grid_size {
            CheckerSize::None => None,
            CheckerSize::Small => Some(4.0),
            CheckerSize::Medium => Some(8.0),
            CheckerSize::Large => Some(16.0),
        }
    }
    /// The two checkerboard colours (light, dark) as sRGB bytes.
    pub fn colors(&self) -> [[u8; 3]; 2] {
        match self.grid_colors {
            CheckerColors::Light => [[255, 255, 255], [204, 204, 204]],
            CheckerColors::Medium => [[153, 153, 153], [102, 102, 102]],
            CheckerColors::Dark => [[102, 102, 102], [51, 51, 51]],
            CheckerColors::Red => [[255, 255, 255], [255, 204, 204]],
            CheckerColors::Orange => [[255, 255, 255], [255, 229, 204]],
            CheckerColors::Green => [[255, 255, 255], [204, 255, 204]],
            CheckerColors::Blue => [[255, 255, 255], [204, 229, 255]],
            CheckerColors::Purple => [[255, 255, 255], [229, 204, 255]],
            CheckerColors::Custom => [parse_hex(&self.custom_light).unwrap_or([255; 3]), parse_hex(&self.custom_dark).unwrap_or([204; 3])],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct UnitsAndRulers {
    pub rulers: Unit,
    pub type_units: TypeUnit,
    pub column_width: f64,
    pub gutter: f64,
    pub print_resolution: f64,
    pub screen_resolution: f64,
    pub point_size: PointSize,
    /// View › Rulers as last set: written when the user toggles the rulers and applied at
    /// launch, so new and opened documents show them as Photoshop does. Set from the View
    /// menu (⌘R), not from the Preferences dialog.
    pub show_rulers: bool,
}

impl Default for UnitsAndRulers {
    fn default() -> Self {
        Self {
            rulers: Unit::Pixels,
            type_units: TypeUnit::Points,
            column_width: 180.0,
            gutter: 12.0,
            print_resolution: 300.0,
            screen_resolution: 72.0,
            point_size: PointSize::PostScript,
            show_rulers: false,
        }
    }
}

/// Rounds `v` to `decimals` places (half away from zero) and drops trailing zeros, so a readout
/// never shows more precision than it has: 0.7995 at three places is `0.8`, never `0.799` or
/// `0.800`, and a whole number keeps no `.0` (#2434).
pub fn fmt_decimals(v: f64, decimals: usize) -> String {
    let p = 10f64.powi(decimals as i32);
    let r = (v * p).round() / p;
    // -0.0 would format as "-0".
    let r = if r == 0.0 { 0.0 } else { r };
    let s = format!("{r:.*}", decimals);
    // Only a fractional part may lose zeros: "550" must not become "55".
    if !s.contains('.') {
        return s;
    }
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

impl UnitsAndRulers {
    /// Format a length in document pixels in the ruler unit, e.g. `"2.5 in"`.
    pub fn format(&self, px: f64, dpi: f64, extent: f64) -> String {
        let v = self.rulers.from_px(px, dpi, extent, self.point_size.per_inch());
        fmt_decimals(v, self.rulers.decimals())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GuidesGridAndSlices {
    pub guide_color: String,
    pub guide_style: LineStyle,
    pub smart_guide_color: String,
    pub grid_color: String,
    pub grid_style: LineStyle,
    /// Major gridline spacing in `gridUnit`.
    pub gridline_every: f64,
    pub grid_unit: Unit,
    pub subdivisions: u32,
    pub slice_color: String,
    pub show_slice_numbers: bool,
}

impl Default for GuidesGridAndSlices {
    fn default() -> Self {
        Self {
            guide_color: "#4affff".into(),
            guide_style: LineStyle::Lines,
            smart_guide_color: "#ff00ff".into(),
            grid_color: "#8c8c8c".into(),
            grid_style: LineStyle::Lines,
            gridline_every: 1.0,
            grid_unit: Unit::Inches,
            subdivisions: 4,
            slice_color: "#38b5ff".into(),
            show_slice_numbers: true,
        }
    }
}

impl GuidesGridAndSlices {
    /// Major gridline spacing in document pixels.
    pub fn major_px(&self, dpi: f64, extent: f64, ppi: f64) -> f64 {
        self.grid_unit.to_px(self.gridline_every.max(1e-3), dpi, extent, ppi).max(1e-3)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct PlugIns {
    /// Load the WebAssembly plug-ins (`*.wasm`) in `additional_plugins_folder` (native only).
    pub use_additional_plugins_folder: bool,
    pub additional_plugins_folder: String,
    pub show_extension_panels: bool,
    pub allow_scripts_to_connect: bool,
    pub generator_enabled: bool,
}

impl Default for PlugIns {
    fn default() -> Self {
        Self {
            use_additional_plugins_folder: false,
            additional_plugins_folder: String::new(),
            show_extension_panels: true,
            allow_scripts_to_connect: false,
            generator_enabled: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct TypePrefs {
    pub smart_quotes: bool,
    pub missing_glyph_protection: bool,
    pub show_font_names_in_english: bool,
    pub use_esc_to_commit: bool,
    pub text_engine: TextEngine,
    pub font_preview: FontPreview,
    pub fill_new_type_layers_with_placeholder: bool,
    pub recent_fonts: u32,
}

impl Default for TypePrefs {
    fn default() -> Self {
        Self {
            smart_quotes: true,
            missing_glyph_protection: true,
            show_font_names_in_english: true,
            use_esc_to_commit: true,
            text_engine: TextEngine::WorldReady,
            font_preview: FontPreview::Medium,
            fill_new_type_layers_with_placeholder: true,
            recent_fonts: 10,
        }
    }
}

impl TypePrefs {
    /// Preferences ▸ Type ▸ Missing Glyph Protection: with it on (the default, as in Photoshop),
    /// glyphs the chosen face lacks fall back through the coverage fonts (#1909). The text engine
    /// applies this in `Session::apply_prefs`.
    pub fn missing_glyph_protection(&self) -> bool {
        self.missing_glyph_protection
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct EnhancedControls {
    pub scrubby_slider_acceleration: bool,
    pub touch_gestures: bool,
    pub zoom_with_trackpad_pinch: bool,
    pub rotate_view_with_trackpad: bool,
}

impl Default for EnhancedControls {
    fn default() -> Self {
        Self { scrubby_slider_acceleration: true, touch_gestures: true, zoom_with_trackpad_pinch: true, rotate_view_with_trackpad: false }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct RawDefaults {
    pub color_space: RawColorSpace,
    pub bit_depth: RawDepth,
    pub resolution: f64,
    pub sharpen_for: RawSharpen,
    pub open_as_smart_object: bool,
    pub apply_auto_tone: bool,
    /// Opening a raw file interactively shows the Camera Raw dialog first (Open / Cancel), as
    /// Photoshop does; off develops it with the defaults straight away. Automation opens never
    /// show the dialog.
    pub open_in_camera_raw: bool,
}

impl Default for RawDefaults {
    fn default() -> Self {
        Self {
            color_space: RawColorSpace::AdobeRgb,
            bit_depth: RawDepth::Sixteen,
            resolution: 300.0,
            sharpen_for: RawSharpen::None,
            open_as_smart_object: false,
            apply_auto_tone: false,
            open_in_camera_raw: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Integrations {
    /// Allow agents to drive the app over the loopback control channel / MCP bridge.
    pub allow_agent_control: bool,
    /// Default control port (0 = only when `--control` is given).
    pub control_port: u32,
}

impl Default for Integrations {
    fn default() -> Self {
        Self { allow_agent_control: true, control_port: 0 }
    }
}

/// Edit › Menus: hidden items and item colours, by command id.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct MenuCustomization {
    pub hidden: Vec<String>,
    /// Command id → colour name (red, orange, yellow, green, blue, violet, gray).
    pub colors: BTreeMap<String, String>,
}

/// Edit › Toolbar: hidden tools and a custom order (tool names as in `ui.set {tool}`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ToolbarCustomization {
    pub hidden: Vec<String>,
    pub order: Vec<String>,
}

/// All preferences. JSON keys are the `edit.preferences.<section>` ids.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Preferences {
    pub general: General,
    pub interface: Interface,
    pub workspace: Workspace,
    pub tools: Tools,
    pub history_log: HistoryLog,
    pub file_handling: FileHandling,
    pub export: Export,
    pub performance: Performance,
    pub scratch_disks: ScratchDisks,
    pub cursors: Cursors,
    pub transparency_and_gamut: TransparencyAndGamut,
    pub units_and_rulers: UnitsAndRulers,
    pub guides_grid_and_slices: GuidesGridAndSlices,
    pub plug_ins: PlugIns,
    #[serde(rename = "type")]
    pub type_: TypePrefs,
    pub enhanced_controls: EnhancedControls,
    pub raw_defaults: RawDefaults,
    pub integrations: Integrations,
    /// Edit › Keyboard Shortcuts: command id → shortcut (`Cmd+Shift+N` notation); an empty
    /// string removes the default shortcut.
    pub shortcuts: BTreeMap<String, String>,
    pub menus: MenuCustomization,
    pub toolbar: ToolbarCustomization,
    /// Edit › Check Spelling: words added to the dictionary ("Add").
    pub user_dictionary: Vec<String>,
    /// Window › Workspace › New Workspace…: saved layouts by name (JSON owned by the shell).
    pub workspaces: BTreeMap<String, Value>,
    /// Window › Workspace › Lock Workspace: panels can't be moved.
    pub workspace_locked: bool,
    /// The live panel layout (dock group heights, order, collapsed groups, open panels), saved
    /// as the user changes it and restored at launch when Workspace › Remember Workspace
    /// Changes is on. JSON owned by the shell.
    pub panel_layout: Value,
    /// The last choices of dialogs that remember them across restarts, by command id (Edit ›
    /// Fill…: `"edit.fill"` → its params). JSON owned by the shell.
    pub dialogs: BTreeMap<String, Value>,
    /// The Brush Preset picker's remembered view: which card parts show (name, stroke, tip) and
    /// the footer slider's card scale, saved as the user changes them and restored at launch.
    /// JSON owned by the shell.
    pub brush_picker: Value,
    /// The brush the tool was last using (settings and tip references, never the full bitmaps;
    /// shaped by `brush_cmds::remembered_brush`), restored at launch so the tool starts where it
    /// was left. Written only when `ToolState::brush_rev` moves.
    pub last_brush: Value,
    /// The tools' options-bar settings (Move › Auto-Select, marquee Feather, tolerance, …) that
    /// differ from their defaults, saved as the user changes them and restored at launch, as
    /// Photoshop remembers each tool's options between sessions. JSON owned by the shell.
    pub tool_options: Value,
    /// View › Extras, the View › Show items (Pixel Grid, Grid, Guides, …) and the View › Snap To
    /// targets that differ from their defaults, saved as the user toggles them and restored at
    /// launch, as Photoshop keeps them between sessions (#3762). JSON owned by the shell.
    pub view_extras: Value,
    /// File › Scripts › Script Events Manager: event → script bindings.
    pub script_events: crate::automate_cmds::ScriptEvents,
}

/// Preferences dialog sections in Photoshop's order: (id, title).
pub const SECTIONS: [(&str, &str); 18] = [
    ("general", "General"),
    ("interface", "Interface"),
    ("workspace", "Workspace"),
    ("tools", "Tools"),
    ("historyLog", "History Log"),
    ("fileHandling", "File Handling"),
    ("export", "Export"),
    ("performance", "Performance"),
    ("scratchDisks", "Scratch Disks"),
    ("cursors", "Cursors"),
    ("transparencyAndGamut", "Transparency & Gamut"),
    ("unitsAndRulers", "Units & Rulers"),
    ("guidesGridAndSlices", "Guides, Grid & Slices"),
    ("plugIns", "Plug-ins"),
    ("type", "Type"),
    ("enhancedControls", "Enhanced Controls"),
    ("rawDefaults", "Camera Raw Defaults"),
    ("integrations", "Integrations"),
];

/// Preferences that nothing reads yet, so Edit › Preferences doesn't show them: a control that
/// does nothing is worse than a missing one (https://github.com/storytold/photocraft/issues/204).
/// They still load, save and round-trip through `prefs.get` / `prefs.set` unchanged.
///
/// When you implement one, remove it here; the `prefs_usage` test fails while a listed
/// preference is read anywhere outside `prefs.rs`, and while an unlisted one is read nowhere.
pub const HIDDEN_UNTIL_IMPLEMENTED: &[&str] = &[
    "general.colorPicker",
    "general.animatedZoom",
    "general.zoomResizesWindows",
    "workspace.autoShowHiddenPanels",
    "workspace.enableFloatingDocumentWindowDocking",
    "fileHandling.saveInBackground",
    "performance.cacheLevels",
    "performance.effectCacheMb",
    "performance.legacyCompositing",
    "scratchDisks.disks",
    "unitsAndRulers.columnWidth",
    "unitsAndRulers.gutter",
    "plugIns.showExtensionPanels",
    "plugIns.allowScriptsToConnect",
    "plugIns.generatorEnabled",
    "type.textEngine",
    "type.recentFonts",
    "enhancedControls.scrubbySliderAcceleration",
    "enhancedControls.touchGestures",
    "rawDefaults.colorSpace",
    "rawDefaults.bitDepth",
    "rawDefaults.resolution",
    "rawDefaults.sharpenFor",
    "rawDefaults.openAsSmartObject",
    "rawDefaults.applyAutoTone",
    // Agent access is governed by the launch flags (`--control`, the automation roots), not
    // by these yet.
    "integrations.allowAgentControl",
    "integrations.controlPort",
];

/// Is the preference at `path` (`"section.key"`) hidden from the Preferences dialog?
pub fn is_hidden(path: &str) -> bool {
    HIDDEN_UNTIL_IMPLEMENTED.contains(&path) || (!cfg!(target_os = "linux") && LINUX_ONLY.contains(&path))
}

/// Preferences that only do something on Linux; the dialog doesn't show them elsewhere. They
/// still load, save and round-trip on every platform.
pub const LINUX_ONLY: &[&str] = &["performance.linuxDisplayServer"];

/// Choices of an enumerated preference (dotted path, e.g. `"cursors.painting"`).
pub fn choices(path: &str) -> Option<&'static [&'static str]> {
    Some(match path {
        "general.colorPicker" => ColorPicker::NAMES,
        "general.imageInterpolation" => Interpolation::NAMES,
        "interface.theme" => Theme::NAMES,
        "interface.appearanceMode" => AppearanceMode::NAMES,
        "interface.darkTheme" => DarkTheme::NAMES,
        "interface.lightTheme" => LightTheme::NAMES,
        "interface.canvasColor" => CanvasColor::NAMES,
        "interface.canvasBorder" => CanvasBorder::NAMES,
        "interface.uiScale" => UiScale::NAMES,
        "interface.uiFontSize" => UiFontSize::NAMES,
        "historyLog.destination" => LogDestination::NAMES,
        "historyLog.detail" => LogDetail::NAMES,
        "fileHandling.imagePreviews" | "fileHandling.maximizePsdCompatibility" => Ask::NAMES,
        "export.quickExportFormat" => QuickExportFormat::NAMES,
        "export.quickExportLocation" => ExportLocation::NAMES,
        "export.metadata" => ExportMetadata::NAMES,
        "cursors.painting" => PaintingCursor::NAMES,
        "cursors.other" => OtherCursor::NAMES,
        "tools.rightClickWithPaintingTools" => RightClickPaint::NAMES,
        "transparencyAndGamut.gridSize" => CheckerSize::NAMES,
        "transparencyAndGamut.gridColors" => CheckerColors::NAMES,
        "unitsAndRulers.rulers" | "guidesGridAndSlices.gridUnit" => Unit::NAMES,
        "unitsAndRulers.typeUnits" => TypeUnit::NAMES,
        "unitsAndRulers.pointSize" => PointSize::NAMES,
        "guidesGridAndSlices.guideStyle" | "guidesGridAndSlices.gridStyle" => LineStyle::NAMES,
        "type.textEngine" => TextEngine::NAMES,
        "type.fontPreview" => FontPreview::NAMES,
        "rawDefaults.colorSpace" => RawColorSpace::NAMES,
        "rawDefaults.bitDepth" => RawDepth::NAMES,
        "rawDefaults.sharpenFor" => RawSharpen::NAMES,
        "performance.gpuBackend" => GpuBackend::NAMES,
        "performance.linuxDisplayServer" => LinuxDisplayServer::NAMES,
        "performance.renderingMode" => RenderingMode::NAMES,
        _ => return None,
    })
}

/// Valid range of a numeric preference.
pub fn range(path: &str) -> Option<(f64, f64)> {
    Some(match path {
        "fileHandling.autosaveMinutes" => (1.0, 240.0),
        "fileHandling.recentFileCount" => (0.0, 100.0),
        "fileHandling.rasterizeSvgGroupsDeeperThan" => (0.0, photocraft_doc::MAX_GROUP_DEPTH as f64),
        "interface.notificationDurationSeconds" => (1.0, 120.0),
        "export.jpegQuality" | "export.webpQuality" => (1.0, 100.0),
        "performance.memoryUsageMb" => (256.0, 1_048_576.0),
        "performance.historyStates" => (1.0, 1000.0),
        "performance.cacheLevels" => (1.0, 8.0),
        "performance.cacheTileSize" => (256.0, 16384.0),
        "performance.effectCacheMb" => (16.0, 65536.0),
        "transparencyAndGamut.gamutWarningOpacity" => (1.0, 100.0),
        "unitsAndRulers.columnWidth" | "unitsAndRulers.gutter" => (0.0, 10_000.0),
        "unitsAndRulers.printResolution" | "unitsAndRulers.screenResolution" | "rawDefaults.resolution" => (1.0, 10_000.0),
        "guidesGridAndSlices.gridlineEvery" => (0.001, 10_000.0),
        "guidesGridAndSlices.subdivisions" => (1.0, 100.0),
        "type.recentFonts" => (0.0, 50.0),
        "integrations.controlPort" => (0.0, 65535.0),
        _ => return None,
    })
}

/// Is this preference a `#rrggbb` colour?
pub fn is_color(path: &str) -> bool {
    if choices(path).is_some() || path.starts_with("shortcuts.") || path.starts_with("menus.") {
        return false;
    }
    let leaf = path.rsplit('.').next().unwrap_or(path);
    matches!(
        leaf,
        "guideColor"
            | "smartGuideColor"
            | "gridColor"
            | "sliceColor"
            | "canvasCustomColor"
            | "brushPreviewColor"
            | "gamutWarningColor"
            | "customLight"
            | "customDark"
    )
}

pub fn parse_hex(s: &str) -> Option<[u8; 3]> {
    let s = s.trim().trim_start_matches('#');
    if s.len() != 6 {
        return None;
    }
    let b = |i: usize| s.get(i..i + 2).and_then(|s| u8::from_str_radix(s, 16).ok());
    Some([b(0)?, b(2)?, b(4)?])
}

// ------------------------------------------------------------------ JSON paths

/// Map keys that are command ids (which contain dots): `shortcuts.<id>`, `menus.colors.<id>`.
fn keyed(path: &str) -> Option<(&str, &str)> {
    for prefix in ["shortcuts.", "menus.colors."] {
        if let Some(id) = path.strip_prefix(prefix) {
            return Some((&prefix[..prefix.len() - 1], id));
        }
    }
    None
}

fn get_path<'a>(root: &'a Value, path: &str) -> Option<&'a Value> {
    if path.is_empty() {
        return Some(root);
    }
    if let Some((map, id)) = keyed(path) {
        return get_path(root, map)?.get(id);
    }
    path.split('.').try_fold(root, |v, k| v.get(k))
}

/// Set `value` at `path` (creating keys only inside free-form maps such as `shortcuts`).
fn set_path(root: &mut Value, path: &str, value: Value) -> std::result::Result<(), String> {
    if let Some((map, id)) = keyed(path) {
        let mut cur = &mut *root;
        for k in map.split('.') {
            cur = cur.get_mut(k).ok_or_else(|| format!("unknown preference `{path}`"))?;
        }
        cur.as_object_mut().ok_or_else(|| format!("`{map}` is not a map"))?.insert(id.to_string(), value);
        return Ok(());
    }
    let keys: Vec<&str> = path.split('.').filter(|k| !k.is_empty()).collect();
    let Some((last, parents)) = keys.split_last() else { return Err("empty path".into()) };
    let mut cur = root;
    for k in parents {
        cur = cur.get_mut(*k).ok_or_else(|| format!("unknown preference `{path}`"))?;
    }
    let free_map = matches!(parents.first(), Some(&"shortcuts")) || (parents.len() == 2 && parents[0] == "menus" && parents[1] == "colors");
    let obj = cur.as_object_mut().ok_or_else(|| format!("`{path}` is not inside a section"))?;
    if !obj.contains_key(*last) && !free_map {
        return Err(format!("unknown preference `{path}`"));
    }
    obj.insert(last.to_string(), value);
    Ok(())
}

/// Validate one value for `path` before it is stored (choices, ranges, colours).
fn check_value(path: &str, v: &Value) -> std::result::Result<(), String> {
    if path == "interface.favoriteThemes" {
        let items = v.as_array().ok_or("`interface.favoriteThemes` must be a list of theme ids")?;
        if items.len() > Theme::NAMES.len() {
            return Err("`interface.favoriteThemes` contains too many themes".into());
        }
        let mut seen = Vec::with_capacity(items.len());
        for item in items {
            let theme = item.as_str().and_then(Theme::parse).ok_or_else(|| format!("unknown favorite theme `{item}`"))?;
            if seen.contains(&theme) {
                return Err(format!("duplicate favorite theme `{}`", theme.name()));
            }
            seen.push(theme);
        }
    }
    if path == "performance.renderingMode" && v.is_null() {
        return Ok(()); // Legacy policy, resolved from useGpu and gpuBackend.
    }
    if let Some(c) = choices(path) {
        let s = v.as_str().ok_or_else(|| format!("`{path}` must be one of {}", c.join("|")))?;
        if !c.contains(&s) {
            return Err(format!("`{path}` must be one of {} (got `{s}`)", c.join("|")));
        }
    }
    if path == "interface.language" {
        let ok = v.as_str().is_some_and(|s| !s.is_empty() && s.len() <= 16 && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'));
        if !ok {
            return Err("`interface.language` must be `auto` or a language code such as `en` or `ja`".into());
        }
    }
    if let Some((lo, hi)) = range(path) {
        let x = v.as_f64().ok_or_else(|| format!("`{path}` must be a number"))?;
        if !(lo..=hi).contains(&x) {
            return Err(format!("`{path}` must be within {lo}..{hi} (got {x})"));
        }
    }
    if is_color(path) && v.as_str().and_then(parse_hex).is_none() {
        return Err(format!("`{path}` must be a #rrggbb colour"));
    }
    if path == "tools.pressureCurve" {
        let pts = v.as_array().ok_or("`tools.pressureCurve` must be a list of [input, output] points")?;
        if !(2..=PressureCurve::MAX_POINTS).contains(&pts.len()) {
            return Err(format!("`tools.pressureCurve` needs 2..={} points (got {})", PressureCurve::MAX_POINTS, pts.len()));
        }
        for p in pts {
            let ok = p.as_array().is_some_and(|a| a.len() == 2 && a.iter().all(|x| x.as_f64().is_some_and(|x| (0.0..=1.0).contains(&x))));
            if !ok {
                return Err(format!("`tools.pressureCurve` points are [input, output] pairs within 0..1 (got {p})"));
            }
        }
    }
    if let Some(sc) = path.strip_prefix("shortcuts.") {
        let s = v.as_str().ok_or_else(|| format!("shortcut for `{sc}` must be a string"))?;
        if !s.is_empty() && normalize_shortcut(s).is_none() {
            return Err(format!("`{s}` is not a valid shortcut (e.g. Cmd+Shift+N, F7)"));
        }
    }
    Ok(())
}

impl Preferences {
    /// The preferences as JSON (camelCase keys).
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }

    /// Read a value by dotted path (`""` = everything).
    pub fn get(&self, path: &str) -> Option<Value> {
        get_path(&self.to_json(), path).cloned()
    }

    /// Set a value by dotted path, validating it. Unknown paths and wrong types are errors.
    pub fn set(&mut self, path: &str, value: Value) -> std::result::Result<(), String> {
        let value = match (path.strip_prefix("shortcuts."), &value) {
            // Normalise shortcut spelling so conflicts compare equal.
            (Some(_), Value::String(s)) if !s.is_empty() => Value::String(normalize_shortcut(s).ok_or_else(|| format!("`{s}` is not a valid shortcut"))?),
            (Some(_), Value::Null) => {
                // null = back to the default.
                let id = &path["shortcuts.".len()..];
                self.shortcuts.remove(id);
                return Ok(());
            }
            _ => value,
        };
        check_value(path, &value)?;
        let mut root = self.to_json();
        set_path(&mut root, path, value)?;
        *self = serde_json::from_value(root).map_err(|e| format!("invalid value for `{path}`: {e}"))?;
        Ok(())
    }

    /// Reset one section (or one dotted path), or everything with `None`.
    pub fn reset(&mut self, path: Option<&str>) -> std::result::Result<(), String> {
        let Some(path) = path.filter(|p| !p.is_empty()) else {
            *self = Preferences::default();
            return Ok(());
        };
        // These maps have no stored defaults; removing an override restores the fallback.
        match keyed(path) {
            Some(("shortcuts", id)) => {
                self.shortcuts.remove(id);
                return Ok(());
            }
            Some(("menus.colors", id)) => {
                self.menus.colors.remove(id);
                return Ok(());
            }
            _ => {}
        }
        let def = Preferences::default().get(path).ok_or_else(|| format!("unknown preference `{path}`"))?;
        if path == "shortcuts" {
            self.shortcuts.clear();
            return Ok(());
        }
        let mut root = self.to_json();
        set_path(&mut root, path, def)?;
        *self = serde_json::from_value(root).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// The shortcut in effect for a command: the user's override, else its default.
    pub fn shortcut<'a>(&'a self, id: &str, default: Option<&'a str>) -> Option<&'a str> {
        match self.shortcuts.get(id) {
            Some(s) if s.is_empty() => None,
            Some(s) => Some(s.as_str()),
            None => default,
        }
    }
}

// ------------------------------------------------------------------ shortcuts

/// Canonical spelling of a shortcut: modifiers in `Cmd+Ctrl+Alt+Shift` order, key last
/// (single characters upper-cased). `None` if it has no key or an unknown modifier.
pub fn normalize_shortcut(s: &str) -> Option<String> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let (mut cmd, mut ctrl, mut alt, mut shift) = (false, false, false, false);
    let mut key: Option<String> = None;
    // A trailing "+" key is spelled "Cmd++".
    let (body, plus_key) = if s.ends_with("++") || s == "+" { (s.trim_end_matches('+'), true) } else { (s, false) };
    for part in body.split('+').filter(|p| !p.is_empty()) {
        match part.to_ascii_lowercase().as_str() {
            "cmd" | "command" | "⌘" | "meta" | "super" => cmd = true,
            "ctrl" | "control" | "⌃" => ctrl = true,
            "alt" | "option" | "opt" | "⌥" => alt = true,
            "shift" | "⇧" => shift = true,
            _ => {
                if key.is_some() {
                    return None;
                }
                let k = if part.chars().count() == 1 { part.to_uppercase() } else { capitalize(part) };
                key = Some(k);
            }
        }
    }
    if plus_key {
        if key.is_some() {
            return None;
        }
        key = Some("+".into());
    }
    let key = key?;
    let mut out = Vec::new();
    if cmd {
        out.push("Cmd");
    }
    if ctrl {
        out.push("Ctrl");
    }
    if alt {
        out.push("Alt");
    }
    if shift {
        out.push("Shift");
    }
    let mut s = out.join("+");
    if !s.is_empty() {
        s.push('+');
    }
    s.push_str(&key);
    Some(s)
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) if f.is_ascii_alphabetic() => {
            let rest: String = c.collect();
            // Function keys stay upper-case (F1…F24); named keys are Title case.
            if f.eq_ignore_ascii_case(&'f') && rest.chars().all(|c| c.is_ascii_digit()) && !rest.is_empty() {
                format!("F{rest}")
            } else {
                format!("{}{}", f.to_ascii_uppercase(), rest.to_ascii_lowercase())
            }
        }
        _ => s.to_string(),
    }
}

/// Hold-and-release gestures (#249): temporary tools that last while their key is held and give
/// the previous tool back on release. They are bound like shortcuts (Edit › Keyboard Shortcuts ›
/// Tools › Temporary, `edit.keyboardShortcuts`) but held rather than pressed, so they are not
/// commands; the UI reads their keys through [`Preferences::shortcut`]. While a selection or
/// shape is being dragged, the Hand key repositions it instead of panning (Photoshop's Space).
/// The erase key keeps the Brush or Pencil and makes its strokes erase (Photoshop's `~`).
pub const TEMPORARY_TOOLS: &[(&str, &str, &str)] = &[
    ("tools.temporary.hand", "Hand Tool (hold)", "Space"),
    ("tools.temporary.zoomIn", "Zoom In (hold, drag to scrub)", "Cmd+Space"),
    ("tools.temporary.zoomOut", "Zoom Out (hold)", "Alt+Space"),
    ("tools.temporary.erase", "Erase with Brush or Pencil (hold)", "`"),
];

/// Every bindable id with its default: commands, then the temporary tools.
fn bindable() -> impl Iterator<Item = (&'static str, Option<&'static str>)> {
    crate::command_specs().iter().map(|c| (c.id, c.shortcut)).chain(TEMPORARY_TOOLS.iter().map(|t| (t.0, Some(t.2))))
}

/// Shortcuts bound to more than one command: (shortcut, ids). `bindings` are (id, shortcut).
pub fn conflicts<'a>(bindings: impl IntoIterator<Item = (&'a str, &'a str)>) -> Vec<(String, Vec<String>)> {
    let mut by: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (id, sc) in bindings {
        if let Some(n) = normalize_shortcut(sc) {
            by.entry(n).or_default().push(id.to_string());
        }
    }
    by.into_iter().filter(|(_, ids)| ids.len() > 1).collect()
}

// ------------------------------------------------------------------ store

/// The session's preferences plus a revision counter that moves on every change (frontends
/// persist when it moves).
#[derive(Clone, Debug, Default)]
pub struct PrefsStore {
    values: Preferences,
    rev: u64,
}

impl PrefsStore {
    pub fn get(&self) -> &Preferences {
        &self.values
    }
    /// Change preferences; bumps the revision.
    pub fn edit<R>(&mut self, f: impl FnOnce(&mut Preferences) -> R) -> R {
        self.rev += 1;
        f(&mut self.values)
    }
    pub fn rev(&self) -> u64 {
        self.rev
    }
}

impl Session {
    /// The current preferences.
    pub fn prefs(&self) -> &Preferences {
        self.prefs.get()
    }

    /// Change preferences and apply the engine-side ones.
    pub fn edit_prefs<R>(&mut self, f: impl FnOnce(&mut Preferences) -> R) -> R {
        let r = self.prefs.edit(f);
        self.apply_prefs();
        r
    }

    /// Push engine-relevant preferences into live state: every document's history limit, the
    /// layer-effect cache budget and the text engine's missing-glyph protection.
    pub fn apply_prefs(&mut self) {
        let n = self.prefs().performance.history_states.max(1) as usize;
        let bytes = self.prefs().performance.history_budget_bytes();
        let budget = self.prefs().performance.effect_cache_mb as usize;
        for st in &mut self.docs {
            st.history.max_states = n;
            st.history.max_bytes = bytes;
            st.history.trim(&st.doc);
        }
        photocraft_compose::set_effect_cache_budget(budget << 20);
        // Without waiting for the system font scan: this runs on the UI thread at launch.
        photocraft_text::set_shared_missing_glyph_protection(self.prefs().type_.missing_glyph_protection());
        crate::plugin_cmds::sync_prefs(self);
    }

    /// Everything persisted as one JSON document: the preferences plus `colorSettings`.
    pub fn prefs_to_json(&self) -> String {
        serde_json::to_string_pretty(&self.prefs_value()).unwrap_or_default()
    }

    /// [`Session::prefs_to_json`] as a JSON tree (frontends merge it with what storage holds).
    pub fn prefs_value(&self) -> Value {
        let mut v = self.prefs().to_json();
        if let Value::Object(m) = &mut v {
            m.insert("colorSettings".into(), serde_json::to_value(&self.color.settings).unwrap_or(Value::Null));
            m.insert("version".into(), json!(1));
            m.insert("presets".into(), self.presets.to_json(self));
        }
        v
    }

    /// Restore preferences saved by [`Session::prefs_to_json`]. Missing keys keep their
    /// defaults and unknown keys are ignored, so files from older and newer versions load.
    pub fn load_prefs_json(&mut self, s: &str) -> std::result::Result<(), String> {
        let mut v: Value = serde_json::from_str(s).map_err(|e| format!("preferences: {e}"))?;
        // Saved preferences before appearance modes had one concrete theme. Preserve its
        // appearance instead of silently switching established users to Auto.
        if let Some(interface) = v.get_mut("interface").and_then(Value::as_object_mut)
            && !interface.contains_key("appearanceMode")
            && let Some(theme) = interface.get("theme").and_then(Value::as_str).and_then(Theme::parse)
        {
            let (mode, slot) = match theme {
                Theme::Pro | Theme::ProMedium | Theme::Studio | Theme::SolarizedDark | Theme::AdwaitaDark | Theme::BreezeDark | Theme::SoftDark => {
                    ("dark", "darkTheme")
                }
                Theme::StudioLight | Theme::ProLightGray | Theme::ProLight | Theme::Classic | Theme::Adwaita | Theme::BreezeLight | Theme::SoftLight => {
                    ("light", "lightTheme")
                }
            };
            interface.insert("appearanceMode".into(), json!(mode));
            // A partially migrated file may already contain a newer theme selection (#5139).
            interface.entry(slot).or_insert_with(|| json!(theme.name()));
        }
        let color = v.as_object_mut().and_then(|m| m.remove("colorSettings"));
        let presets = v.as_object_mut().and_then(|m| m.remove("presets"));
        let prefs: Preferences = serde_json::from_value(v).map_err(|e| format!("preferences: {e}"))?;
        if let Some(c) = color {
            self.color.settings = serde_json::from_value(c).unwrap_or_default();
            photocraft_compose::psblend::set_text_gamma(self.color.settings.blend_text_gamma);
        }
        self.prefs.edit(|p| *p = prefs);
        if let Some(v) = presets {
            self.load_presets_json(v);
        }
        self.apply_prefs();
        Ok(())
    }

    /// Full JSON view used by `prefs.get`: preferences plus `colorSettings`.
    fn prefs_view(&self) -> Value {
        let mut v = self.prefs().to_json();
        if let Value::Object(m) = &mut v {
            m.insert("colorSettings".into(), serde_json::to_value(&self.color.settings).unwrap_or(Value::Null));
        }
        v
    }

    /// Set one dotted path, routing `colorSettings.*` to the colour state.
    /// Batches defer colour validation until their complete candidate is staged. Returns
    /// whether colour settings were touched, including whole-section/root-object updates.
    fn set_pref(&mut self, path: &str, value: Value, validate_color: bool) -> std::result::Result<bool, String> {
        if let Some(rest) = path.strip_prefix("colorSettings") {
            let mut cur = serde_json::to_value(&self.color.settings).map_err(|e| e.to_string())?;
            let rest = rest.trim_start_matches('.');
            if rest.is_empty() {
                cur = value;
            } else {
                set_path(&mut cur, rest, value)?;
            }
            let next = serde_json::from_value(cur).map_err(|e| format!("invalid value for `{path}`: {e}"))?;
            if validate_color {
                crate::color_cmds::validate_settings(&next)?;
            }
            self.color.settings = next;
            self.prefs.edit(|_| ());
            return Ok(true);
        }
        let mut next = self.prefs().clone();
        let mut color_changed = false;
        if path.is_empty() || SECTIONS.iter().any(|(id, _)| *id == path) || matches!(path, "shortcuts" | "menus" | "toolbar") {
            // Whole section (or everything): merge the object key by key so each value is checked.
            let obj = value.as_object().ok_or_else(|| format!("`{path}` needs an object"))?;
            for (k, v) in obj {
                let p = if path.is_empty() { k.clone() } else { format!("{path}.{k}") };
                if path.is_empty() && k == "colorSettings" {
                    color_changed |= self.set_pref("colorSettings", v.clone(), validate_color)?;
                    continue;
                }
                if v.is_object() && !p.starts_with("shortcuts") && !p.ends_with(".colors") && next.get(&p).is_some_and(|c| c.is_object()) {
                    for (k2, v2) in v.as_object().into_iter().flatten() {
                        next.set(&format!("{p}.{k2}"), v2.clone())?;
                    }
                } else {
                    next.set(&p, v.clone())?;
                }
            }
        } else {
            next.set(path, value)?;
        }
        // Old `prefs.set interface.theme` clients (by path or inside a section object) still
        // select a visible theme.
        if path == "interface.theme" || next.interface.theme != self.prefs().interface.theme {
            match next.interface.theme {
                Theme::Pro => {
                    next.interface.dark_theme = DarkTheme::Pro;
                    next.interface.appearance_mode = AppearanceMode::Dark;
                }
                Theme::ProMedium => {
                    next.interface.dark_theme = DarkTheme::ProMedium;
                    next.interface.appearance_mode = AppearanceMode::Dark;
                }
                Theme::Studio => {
                    next.interface.dark_theme = DarkTheme::Studio;
                    next.interface.appearance_mode = AppearanceMode::Dark;
                }
                Theme::StudioLight => {
                    next.interface.light_theme = LightTheme::StudioLight;
                    next.interface.appearance_mode = AppearanceMode::Light;
                }
                Theme::ProLightGray => {
                    next.interface.light_theme = LightTheme::ProLightGray;
                    next.interface.appearance_mode = AppearanceMode::Light;
                }
                Theme::ProLight => {
                    next.interface.light_theme = LightTheme::ProLight;
                    next.interface.appearance_mode = AppearanceMode::Light;
                }
                Theme::Classic => {
                    next.interface.light_theme = LightTheme::Classic;
                    next.interface.appearance_mode = AppearanceMode::Light;
                }
                Theme::Adwaita => {
                    next.interface.light_theme = LightTheme::Adwaita;
                    next.interface.appearance_mode = AppearanceMode::Light;
                }
                Theme::SolarizedDark => {
                    next.interface.dark_theme = DarkTheme::SolarizedDark;
                    next.interface.appearance_mode = AppearanceMode::Dark;
                }
                Theme::AdwaitaDark => {
                    next.interface.dark_theme = DarkTheme::AdwaitaDark;
                    next.interface.appearance_mode = AppearanceMode::Dark;
                }
                Theme::BreezeDark => {
                    next.interface.dark_theme = DarkTheme::BreezeDark;
                    next.interface.appearance_mode = AppearanceMode::Dark;
                }
                Theme::BreezeLight => {
                    next.interface.light_theme = LightTheme::BreezeLight;
                    next.interface.appearance_mode = AppearanceMode::Light;
                }
                Theme::SoftDark => {
                    next.interface.dark_theme = DarkTheme::SoftDark;
                    next.interface.appearance_mode = AppearanceMode::Dark;
                }
                Theme::SoftLight => {
                    next.interface.light_theme = LightTheme::SoftLight;
                    next.interface.appearance_mode = AppearanceMode::Light;
                }
            }
        }
        self.prefs.edit(|p| *p = next);
        Ok(color_changed)
    }
}

// ------------------------------------------------------------------ commands

fn always(_: &Session) -> std::result::Result<(), String> {
    Ok(())
}

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

fn prefs_get(s: &mut Session, p: &Value) -> Result<Value> {
    let path = p.get("path").and_then(Value::as_str).unwrap_or("");
    let v = s.prefs_view();
    get_path(&v, path).cloned().ok_or_else(|| bad("prefs.get", format!("unknown preference `{path}`")))
}

fn prefs_set(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "prefs.set";
    let mut changes: Vec<(String, Value)> = Vec::new();
    if let Some(path) = p.get("path").and_then(Value::as_str) {
        changes.push((path.to_string(), p.get("value").cloned().ok_or_else(|| bad(cmd, "missing `value`"))?));
    }
    if let Some(m) = p.get("values").and_then(Value::as_object) {
        changes.extend(m.iter().map(|(k, v)| (k.clone(), v.clone())));
    }
    if changes.is_empty() {
        return Err(bad(cmd, "give `path` and `value`, or `values: {path: value, …}`"));
    }
    // All or nothing: validate against a scratch copy first.
    let (saved_prefs, saved_color) = (s.prefs().clone(), s.color.settings.clone());
    let mut color_changed = false;
    for (path, v) in &changes {
        match s.set_pref(path, v.clone(), false) {
            Ok(touched) => color_changed |= touched,
            Err(e) => {
                s.prefs.edit(|p| *p = saved_prefs);
                s.color.settings = saved_color;
                return Err(bad(cmd, e));
            }
        }
    }
    if color_changed && let Err(e) = crate::color_cmds::validate_settings(&s.color.settings) {
        s.prefs.edit(|p| *p = saved_prefs);
        s.color.settings = saved_color;
        return Err(bad(cmd, e));
    }
    // `colorSettings.blendTextGamma` lives in the compositor too, as `edit.colorSettings` sets it.
    photocraft_compose::psblend::set_text_gamma(s.color.settings.blend_text_gamma);
    s.apply_prefs();
    let view = s.prefs_view();
    let out: Map<String, Value> = changes.iter().map(|(k, _)| (k.clone(), get_path(&view, k).cloned().unwrap_or(Value::Null))).collect();
    Ok(Value::Object(out))
}

fn prefs_reset(s: &mut Session, p: &Value) -> Result<Value> {
    let path = p.get("path").or_else(|| p.get("section")).and_then(Value::as_str);
    match path {
        Some("colorSettings") => s.color.settings = Default::default(),
        // One colour setting: `Preferences` has no `colorSettings`, so take the default
        // from `ColorSettings` and route it like `prefs.set` does.
        Some(path) if path.starts_with("colorSettings.") => {
            let defaults = serde_json::to_value(crate::color_cmds::ColorSettings::default()).map_err(|e| bad("prefs.reset", e.to_string()))?;
            let key = path.strip_prefix("colorSettings.").unwrap_or(path);
            let def = get_path(&defaults, key).cloned().ok_or_else(|| bad("prefs.reset", format!("unknown preference `{path}`")))?;
            s.set_pref(path, def, true).map_err(|e| bad("prefs.reset", e))?;
        }
        None => {
            s.color.settings = Default::default();
            s.edit_prefs(|p| p.reset(None)).map_err(|e| bad("prefs.reset", e))?;
        }
        Some(path) => {
            let mut next = s.prefs().clone();
            next.reset(Some(path)).map_err(|e| bad("prefs.reset", e))?;
            s.edit_prefs(|p| *p = next);
        }
    }
    s.prefs.edit(|_| ());
    photocraft_compose::psblend::set_text_gamma(s.color.settings.blend_text_gamma);
    s.apply_prefs();
    if path.is_some_and(|path| keyed(path).is_some()) {
        // The removed override is absent, so reading its old path would report an error.
        Ok(Value::Null)
    } else {
        prefs_get(s, &json!({"path": path.unwrap_or("")}))
    }
}

/// `edit.preferences.<section>`: the section's values (the GUI opens the dialog on it instead).
fn preferences_section(s: &mut Session, p: &Value) -> Result<Value> {
    let section = p.get("__section").and_then(Value::as_str).unwrap_or("general").to_string();
    let title = SECTIONS.iter().find(|(id, _)| *id == section).map_or("General", |(_, t)| *t);
    let values = s.prefs().get(&section).unwrap_or(Value::Null);
    Ok(json!({"section": section, "title": title, "values": values}))
}

/// Edit › Keyboard Shortcuts (engine part): list, change and reset application shortcuts.
fn keyboard_shortcuts(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "edit.keyboardShortcuts";
    if let Some(r) = p.get("reset") {
        match r {
            Value::Bool(true) => s.edit_prefs(|p| p.shortcuts.clear()),
            Value::Array(ids) => s.edit_prefs(|p| {
                for id in ids.iter().filter_map(Value::as_str) {
                    p.shortcuts.remove(id);
                }
            }),
            _ => {}
        }
    }
    // A Photoshop `.kys` set (`importKys`, its XML text) becomes `set` entries, matched by label
    // (`crate::kys`); a key equal to the command's default restores the default, and keys given
    // in `set` as well win over the file's.
    let imported;
    let mut import = None;
    let p = match p.get("importKys") {
        Some(xml) => {
            let xml = xml.as_str().ok_or_else(|| bad(cmd, "`importKys` is the text of a .kys file"))?;
            let set = crate::kys::parse(xml).map_err(|e| bad(cmd, e))?;
            let plan = crate::kys::plan(crate::kys::command_candidates(), &set);
            let mut merged = serde_json::Map::new();
            for (id, sc) in &plan.set {
                let default = crate::command_specs().iter().find(|c| c.id == id).and_then(|c| c.shortcut).and_then(normalize_shortcut);
                merged.insert(id.clone(), if default.as_deref() == Some(sc.as_str()) { Value::Null } else { json!(sc) });
            }
            merged.extend(p.get("set").and_then(Value::as_object).cloned().unwrap_or_default());
            import = Some(json!({
                "name": set.name,
                "imported": plan.set.len(),
                "set": plan.set,
                "unknown": plan.unknown,
                "unreadable": plan.unreadable,
                "alternates": plan.alternates,
                "toolKeys": set.tool_keys,
            }));
            let mut next = p.clone();
            next["set"] = Value::Object(merged);
            imported = next;
            &imported
        }
        None => p,
    };
    if let Some(m) = p.get("set").and_then(Value::as_object) {
        let mut next = s.prefs().clone();
        for (id, v) in m {
            if !bindable().any(|(b, _)| b == id) && !p.get("allowUnknown").and_then(Value::as_bool).unwrap_or(false) {
                return Err(bad(cmd, format!("unknown command `{id}`")));
            }
            next.set(&format!("shortcuts.{id}"), v.clone()).map_err(|e| bad(cmd, e))?;
        }
        // Like Photoshop, a shortcut moved to a new command is taken away from the old one.
        if p.get("removeConflicts").and_then(Value::as_bool).unwrap_or(true) {
            for (id, v) in m {
                let Some(sc) = v.as_str().and_then(normalize_shortcut) else { continue };
                for (c, def) in bindable() {
                    // Never strip a command that this same call is assigning:
                    // two entries for one key are a clash inside the call, which
                    // the returned conflicts list reports (#719). Stripping
                    // each other here unbound both silently.
                    if c == id || m.contains_key(c) {
                        continue;
                    }
                    if next.shortcut(c, def).and_then(normalize_shortcut).as_deref() == Some(sc.as_str()) {
                        next.shortcuts.insert(c.to_string(), String::new());
                    }
                }
            }
        }
        s.edit_prefs(|p| *p = next);
    }
    let prefs = s.prefs();
    let filter = p.get("filter").and_then(Value::as_str).map(str::to_ascii_lowercase);
    let temporary = TEMPORARY_TOOLS.iter().map(|t| (t.0, t.1, &["Tools", "Temporary"][..], Some(t.2), true));
    let list: Vec<Value> = crate::command_specs()
        .iter()
        .map(|c| (c.id, c.label, c.menu, c.shortcut, false))
        .chain(temporary)
        .filter(|c| filter.as_ref().is_none_or(|f| c.0.to_ascii_lowercase().contains(f) || c.1.to_ascii_lowercase().contains(f)))
        .filter(|c| p.get("list").and_then(Value::as_bool).unwrap_or(false) || filter.is_some() || prefs.shortcuts.contains_key(c.0))
        .map(|(id, label, menu, def, hold)| {
            let mut v = json!({"id": id, "label": label, "menu": menu, "default": def, "shortcut": prefs.shortcut(id, def)});
            if hold {
                v["hold"] = json!(true);
            }
            v
        })
        .collect();
    let bindings: Vec<(&str, &str)> = bindable().filter_map(|(id, def)| Some((id, prefs.shortcut(id, def)?))).collect();
    let conflicts: Vec<Value> = conflicts(bindings).into_iter().map(|(sc, ids)| json!({"shortcut": sc, "commands": ids})).collect();
    let mut out = json!({"overrides": prefs.shortcuts, "commands": list, "conflicts": conflicts});
    if let Some(import) = import {
        out["import"] = import;
    }
    Ok(out)
}

/// Edit › Menus: hide/show items and give them colours.
fn menus(s: &mut Session, p: &Value) -> Result<Value> {
    let ids = |k: &str| -> Vec<String> {
        p.get(k).and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default()
    };
    const COLORS: [&str; 8] = ["none", "red", "orange", "yellow", "green", "blue", "violet", "gray"];
    if let Some(m) = p.get("color").and_then(Value::as_object) {
        for v in m.values() {
            if let Some(c) = v.as_str().filter(|c| !COLORS.contains(c)) {
                return Err(bad("edit.menus", format!("unknown colour `{c}` ({})", COLORS.join("|"))));
            }
        }
    }
    let (hide, show) = (ids("hide"), ids("show"));
    s.edit_prefs(|pr| {
        if p.get("reset").and_then(Value::as_bool) == Some(true) {
            pr.menus = MenuCustomization::default();
        }
        for id in hide {
            if !pr.menus.hidden.contains(&id) {
                pr.menus.hidden.push(id);
            }
        }
        pr.menus.hidden.retain(|h| !show.contains(h));
        if let Some(m) = p.get("color").and_then(Value::as_object) {
            for (id, v) in m {
                match v.as_str() {
                    Some(c) if c != "none" => {
                        pr.menus.colors.insert(id.clone(), c.to_string());
                    }
                    _ => {
                        pr.menus.colors.remove(id);
                    }
                }
            }
        }
    });
    Ok(serde_json::to_value(&s.prefs().menus).unwrap_or_default())
}

/// Edit › Toolbar: hide tools and reorder the toolbar.
fn toolbar(s: &mut Session, p: &Value) -> Result<Value> {
    let ids = |k: &str| -> Option<Vec<String>> { p.get(k).and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()) };
    s.edit_prefs(|pr| {
        if p.get("reset").and_then(Value::as_bool) == Some(true) {
            pr.toolbar = ToolbarCustomization::default();
        }
        if let Some(h) = ids("hidden") {
            pr.toolbar.hidden = h;
        }
        if let Some(o) = ids("order") {
            pr.toolbar.order = o;
        }
    });
    Ok(serde_json::to_value(&s.prefs().toolbar).unwrap_or_default())
}

macro_rules! spec {
    ($id:literal, $label:literal, [$($m:literal),*], $sc:expr, $params:literal, $run:expr, $journal:expr) => {
        CommandSpec { id: $id, label: $label, menu: &[$($m),*], shortcut: $sc, params: $params, enabled: always, run: $run, journal: $journal }
    };
}

macro_rules! section {
    ($id:literal, $label:literal) => {
        section!($id, $label, None)
    };
    ($id:literal, $label:literal, $shortcut:expr) => {
        CommandSpec { id: $id, label: $label, menu: &["Edit", "Preferences"], shortcut: $shortcut, params: r##"{}"##, enabled: always, run: |s, _| preferences_section(s, &json!({"__section": section_of($id)})), journal: false }
    };
}

fn section_of(id: &str) -> &str {
    id.rsplit('.').next().unwrap_or("general")
}

/// Preference command specs.
pub fn specs() -> Vec<CommandSpec> {
    vec![
        spec!(
            "prefs.get",
            "Get Preferences",
            [],
            None,
            r##"{"path":"section.key"?=everything (e.g. "performance.historyStates", "colorSettings.workingRgb")}"##,
            prefs_get,
            false
        ),
        spec!(
            "prefs.set",
            "Set Preferences",
            [],
            None,
            r##"{"path":"section.key","value":json} or {"values":{"section.key":json,…}} (validated; all or nothing)"##,
            prefs_set,
            false
        ),
        spec!("prefs.reset", "Reset Preferences", [], None, r##"{"path":"section|section.key"?=everything}"##, prefs_reset, false),
        // Photoshop: ⌘K opens Preferences › General (⌘, is Layer › Hide Layers).
        section!("edit.preferences.general", "General…", Some("Cmd+K")),
        section!("edit.preferences.interface", "Interface…"),
        section!("edit.preferences.workspace", "Workspace…"),
        section!("edit.preferences.tools", "Tools…"),
        section!("edit.preferences.historyLog", "History Log…"),
        section!("edit.preferences.fileHandling", "File Handling…"),
        section!("edit.preferences.export", "Export…"),
        section!("edit.preferences.performance", "Performance…"),
        section!("edit.preferences.scratchDisks", "Scratch Disks…"),
        section!("edit.preferences.cursors", "Cursors…"),
        section!("edit.preferences.transparencyAndGamut", "Transparency & Gamut…"),
        section!("edit.preferences.unitsAndRulers", "Units & Rulers…"),
        section!("edit.preferences.guidesGridAndSlices", "Guides, Grid & Slices…"),
        section!("edit.preferences.plugIns", "Plug-ins…"),
        section!("edit.preferences.type", "Type…"),
        section!("edit.preferences.enhancedControls", "Enhanced Controls…"),
        section!("edit.preferences.rawDefaults", "Camera Raw…"),
        section!("edit.preferences.integrations", "Integrations…"),
        spec!(
            "edit.keyboardShortcuts",
            "Keyboard Shortcuts…",
            ["Edit"],
            Some("Cmd+Alt+Shift+K"),
            r##"{"set":{"<command id>|tools.temporary.hand|zoomIn|zoomOut":"Cmd+Shift+X"|""(remove)|null(default)}?,"reset":true|["<id>",…]?,"removeConflicts":bool=true,"filter":str?,"list":bool=false,"importKys":"<.kys XML>"?}"##,
            keyboard_shortcuts,
            true
        ),
        spec!(
            "edit.menus",
            "Menus…",
            ["Edit"],
            Some("Cmd+Alt+Shift+M"),
            r##"{"hide":["<id>",…]?,"show":["<id>",…]?,"color":{"<id>":"red|orange|yellow|green|blue|violet|gray|none"}?,"reset":bool=false}"##,
            menus,
            true
        ),
        spec!("edit.toolbar", "Toolbar…", ["Edit"], None, r##"{"hidden":["<tool>",…]?,"order":["<tool>",…]?,"reset":bool=false}"##, toolbar, true),
    ]
}

#[cfg(test)]
#[path = "prefs/tests.rs"]
mod tests;
