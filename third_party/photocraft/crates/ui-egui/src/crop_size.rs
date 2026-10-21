//! The Crop tool's W x H x Resolution mode (#2443), as in Photoshop's options bar.
//!
//! The preset dropdown's "W x H x Resolution" (`crop_ratio` = [`WHR`]) shows a width and a height
//! field, each a length with its unit ("4 in", "10 cm", "25 mm", "1024 px"; a bare number keeps the
//! field's unit, pixels for an empty field), and a resolution with its unit (px/in or px/cm).
//!
//! - While the frame is drawn or resized it holds W : H ([`ratio`]).
//! - Committing (`canvas::commit_crop`) passes `targetWidth`/`targetHeight` (px) and `resolution`
//!   (ppi) to `image.crop` ([`commit_params`]): the cropped area is resampled to exactly W x H px and
//!   the document takes the resolution, in one undo step. Physical lengths convert at the typed
//!   resolution, or at the document's own when the resolution field is empty (which then leaves
//!   the resolution alone). With W or H empty the crop is not resampled (a free crop); a
//!   resolution alone still sets the document's resolution.
//! - The dropdown's built-in presets (4 x 5 in 300 ppi, …) fill the fields; Front Image (or `I`
//!   with the Crop tool, [`keys`]) fills them with the active document's size in px and its
//!   resolution; Clear empties them; ⇄ and X swap W and H.

use egui::{Modifiers, Stroke, StrokeKind, vec2};
use photocraft_engine::prefs::Unit;
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::state::{Tool, ToolOptions};
use crate::theme::Tokens;

/// `crop_ratio` key of the W x H x Resolution mode.
pub const WHR: &str = "whr";
/// Dropdown key of Front Image (applied at once, never kept in `crop_ratio`).
pub const FRONT: &str = "front";
/// Resolution units of the resolution field.
pub const PX_PER_IN: &str = "px/in";
pub const PX_PER_CM: &str = "px/cm";
/// Points per inch for pt / pica lengths (PostScript, Photoshop's default).
const POINTS_PER_INCH: f64 = 72.0;
/// The largest number a length or resolution field keeps (well past any canvas or ppi limit,
/// which `image.crop` enforces with its own message).
const MAX_FIELD: f64 = 1e9;

/// Photoshop's built-in crop presets: (dropdown key, label, width, height, resolution in px/in).
pub const PRESETS: &[(&str, &str, &str, &str, &str)] = &[
    ("preset:4x5in", "4 x 5 in 300 ppi", "4 in", "5 in", "300"),
    ("preset:8.5x11in", "8.5 x 11 in 300 ppi", "8.5 in", "11 in", "300"),
    ("preset:1024x768px", "1024 x 768 px 92 ppi", "1024 px", "768 px", "92"),
    ("preset:1280x800px", "1280 x 800 px 113 ppi", "1280 px", "800 px", "113"),
    ("preset:1366x768px", "1366 x 768 px 135 ppi", "1366 px", "768 px", "135"),
    ("preset:4x6in", "4 x 6 in 300 ppi", "4 in", "6 in", "300"),
    ("preset:5x7in", "5 x 7 in 300 ppi", "5 in", "7 in", "300"),
    ("preset:8x10in", "8 x 10 in 300 ppi", "8 in", "10 in", "300"),
];

/// The preset dropdown: Ratio, W x H x Resolution, the ratio presets, the size presets, Front Image.
pub fn dropdown_options() -> Vec<(String, &'static str)> {
    let ratios = crate::chrome_ui::CROP_RATIOS;
    let mut v: Vec<(String, &'static str)> = Vec::with_capacity(ratios.len() + PRESETS.len() + 2);
    let mut rest = ratios.iter();
    if let Some((k, l)) = rest.next() {
        v.push((k.to_string(), l));
    }
    v.push((WHR.to_string(), tl!("W x H x Resolution")));
    v.extend(rest.map(|(k, l)| (k.to_string(), *l)));
    v.extend(PRESETS.iter().map(|(k, l, ..)| (k.to_string(), *l)));
    v.push((FRONT.to_string(), tl!("Front Image")));
    v
}

/// The unit a suffix names ([`Unit::from_suffix`]); a crop size has no percent.
fn unit_of(suffix: &str) -> Option<Unit> {
    Unit::from_suffix(suffix).filter(|u| *u != Unit::Percent)
}

/// A typed length: a positive number and an optional unit ("4 in", "4in", "10,5 cm", "1024").
/// A bare number takes `fallback`. `None` for anything else (empty, zero, negative, unknown unit).
pub fn parse_length(text: &str, fallback: Unit) -> Option<(f64, Unit)> {
    let text = text.trim();
    if text.is_empty() || text.len() > 64 {
        return None;
    }
    let split = text.find(|c: char| c.is_alphabetic() || c == '"').unwrap_or(text.len());
    let (num, suffix) = (text.get(..split)?.trim(), text.get(split..)?.trim());
    let v = crate::widgets::parse_num(&num.replace(',', "."))?;
    let unit = if suffix.is_empty() { fallback } else { unit_of(suffix)? };
    (v.is_finite() && v > 0.0 && v <= MAX_FIELD).then_some((v, unit))
}

/// "4 in", "1024 px", "8.5 in", "2.54 cm": the number (up to its unit's decimals) and the unit.
pub fn format_length(v: f64, unit: Unit) -> String {
    let n = format!("{v:.*}", unit.decimals().max(2));
    let n = if n.contains('.') { n.trim_end_matches('0').trim_end_matches('.') } else { &n };
    format!("{n} {}", unit.suffix())
}

/// The unit a stored field value is in (pixels when it has none).
fn field_unit(stored: &str) -> Unit {
    parse_length(stored, Unit::Pixels).map_or(Unit::Pixels, |(_, u)| u)
}

/// What a length field keeps once typed: the normalised length, empty for an empty field, and the
/// previous value for something that isn't a length.
pub fn normalize_length(typed: &str, previous: &str) -> String {
    if typed.trim().is_empty() {
        return String::new();
    }
    match parse_length(typed, field_unit(previous)) {
        Some((v, u)) => format_length(v, u),
        None => previous.to_string(),
    }
}

/// What the resolution field keeps once typed: a positive number, empty, or the previous value.
pub fn normalize_resolution(typed: &str, previous: &str) -> String {
    if typed.trim().is_empty() {
        return String::new();
    }
    match crate::widgets::parse_num(&typed.trim().replace(',', ".")).filter(|v| v.is_finite() && *v > 0.0 && *v <= MAX_FIELD) {
        Some(v) => crate::widgets::fmt_num2(v),
        None => previous.to_string(),
    }
}

/// The resolution field in pixels per inch, `None` when empty or not a positive number.
pub fn resolution_ppi(o: &ToolOptions) -> Option<f64> {
    let v = crate::widgets::parse_num(&o.crop_resolution.trim().replace(',', ".")).filter(|v| v.is_finite() && *v > 0.0)?;
    Some(if o.crop_resolution_unit == PX_PER_CM { v * 2.54 } else { v })
}

/// W x H x Resolution's target: the size in pixels the crop is resampled to (unrounded) and the
/// resolution it sets, when the mode is on and both W and H hold lengths. Physical lengths convert
/// at the typed resolution, or at `doc_dpi` when there is none.
pub fn target(o: &ToolOptions, doc_dpi: f64) -> Option<(f64, f64, Option<f64>)> {
    if o.crop_ratio != WHR {
        return None;
    }
    let res = resolution_ppi(o);
    let dpi = res.unwrap_or(doc_dpi);
    let px = |text: &str| {
        let (v, u) = parse_length(text, Unit::Pixels)?;
        let px = u.to_px(v, dpi, 0.0, POINTS_PER_INCH);
        (px.is_finite() && px > 0.0).then_some(px)
    };
    Some((px(&o.crop_width)?, px(&o.crop_height)?, res))
}

/// The ratio W x H x Resolution holds while the frame is drawn or resized (width / height).
pub fn ratio(o: &ToolOptions, doc_dpi: f64) -> Option<f64> {
    let (w, h, _) = target(o, doc_dpi)?;
    Some(w / h).filter(|k| k.is_finite() && *k > 0.0)
}

/// Editable W and H shown for a custom aspect ratio. An existing valid ratio is authoritative;
/// when either side is zero, use the in-progress values retained in ToolOptions.
pub fn custom_ratio_fields(o: &ToolOptions, doc_size: (f64, f64)) -> (f32, f32) {
    crate::chrome_ui::crop_ratio(&o.crop_ratio, doc_size.0, doc_size.1)
        .filter(|(w, h)| w.is_finite() && h.is_finite() && *w > 0.0 && *h > 0.0 && *w <= 99_999.0 && *h <= 99_999.0)
        .map(|(w, h)| (w as f32, h as f32))
        .unwrap_or((o.crop_custom_ratio_width, o.crop_custom_ratio_height))
}

/// Apply the selected non-WHR ratio to its editable W/H fields. Free Ratio starts empty.
pub fn select_custom_ratio(o: &mut ToolOptions) {
    let sides = crate::chrome_ui::crop_ratio(&o.crop_ratio, 0.0, 0.0)
        .filter(|(w, h)| w.is_finite() && h.is_finite() && *w > 0.0 && *h > 0.0 && *w <= 99_999.0 && *h <= 99_999.0)
        .map(|(w, h)| (w as f32, h as f32))
        .unwrap_or((0.0, 0.0));
    (o.crop_custom_ratio_width, o.crop_custom_ratio_height) = sides;
}

/// Keep custom ratio fields across frames. An incomplete pair means free Ratio until both sides
/// are positive.
pub fn set_custom_ratio_fields(o: &mut ToolOptions, width: f32, height: f32) {
    o.crop_custom_ratio_width = width;
    o.crop_custom_ratio_height = height;
    if width.is_finite() && height.is_finite() && width > 0.0 && height > 0.0 {
        o.crop_ratio = format!("{}:{}", crate::widgets::fmt_num(f64::from(width)), crate::widgets::fmt_num(f64::from(height)));
    } else {
        o.crop_ratio.clear();
    }
}

/// `image.crop`'s target params for the pending crop: `targetWidth`/`targetHeight` (whole px) when
/// W and H are set, and `resolution` (ppi) when the resolution is; the Image Interpolation
/// preference resamples.
pub fn commit_params(app: &PhotocraftApp, p: &mut Value) {
    let o = &app.ui.tool_options;
    if o.crop_ratio != WHR {
        return;
    }
    let doc_dpi = app.session.active().map_or(72.0, |st| f64::from(st.doc.resolution_dpi));
    let px = |v: f64| v.round().clamp(1.0, MAX_FIELD) as u64;
    if let Some((w, h, _)) = target(o, doc_dpi) {
        p["targetWidth"] = json!(px(w));
        p["targetHeight"] = json!(px(h));
        p["resample"] = json!(crate::sizing::pref_resample(app.session.prefs().general.image_interpolation));
    }
    if let Some(r) = resolution_ppi(o) {
        p["resolution"] = json!(r);
    }
}

/// Fills W, H (px) and the resolution from the active document, in W x H x Resolution mode.
pub fn front_image(app: &mut PhotocraftApp) {
    let Some(d) = app.session.active().map(|st| (st.doc.size, f64::from(st.doc.resolution_dpi))) else { return };
    let (size, dpi) = d;
    let o = &mut app.ui.tool_options;
    o.crop_ratio = WHR.into();
    o.crop_width = format_length(f64::from(size.width), Unit::Pixels);
    o.crop_height = format_length(f64::from(size.height), Unit::Pixels);
    let res = if o.crop_resolution_unit == PX_PER_CM { dpi / 2.54 } else { dpi };
    o.crop_resolution = crate::widgets::fmt_num2(res);
}

/// Clear in W x H x Resolution mode: empties W, H and the resolution (the mode stays).
pub fn clear(o: &mut ToolOptions) {
    o.crop_width.clear();
    o.crop_height.clear();
    o.crop_resolution.clear();
}

/// Swaps W and H (the ⇄ button, X). False outside W x H x Resolution mode.
pub fn swap(o: &mut ToolOptions) -> bool {
    if o.crop_ratio != WHR {
        return false;
    }
    std::mem::swap(&mut o.crop_width, &mut o.crop_height);
    true
}

/// After the preset dropdown chose `crop_ratio`: a size preset fills the fields and Front Image
/// reads the document, both leaving W x H x Resolution mode on.
pub fn chosen(app: &mut PhotocraftApp) {
    let key = app.ui.tool_options.crop_ratio.clone();
    if key == FRONT {
        app.ui.tool_options.crop_ratio = WHR.into();
        front_image(app);
    } else if let Some((_, _, w, h, r)) = PRESETS.iter().find(|p| p.0 == key) {
        let o = &mut app.ui.tool_options;
        o.crop_ratio = WHR.into();
        o.crop_width = (*w).into();
        o.crop_height = (*h).into();
        o.crop_resolution = (*r).into();
        o.crop_resolution_unit = PX_PER_IN.into();
    }
    crate::crop_ui::apply_ratio(app);
}

/// I with the Crop tool (no text field focused, which the caller checks): Front Image. Returns
/// true when the key was used; otherwise I stays the Eyedropper tool key.
pub fn keys(app: &mut PhotocraftApp, ctx: &egui::Context) -> bool {
    if app.ui.tool != Tool::Crop || app.crop.drag.is_some() || app.session.active().is_none() {
        return false;
    }
    if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, egui::Key::I)) {
        front_image(app);
        crate::crop_ui::apply_ratio(app);
        return true;
    }
    false
}

/// A text field for a length or the resolution: what is typed is kept while it has focus and
/// normalised into `value` (by `normalize(typed, previous)`) when it loses it. Returns true then.
fn text_field(ui: &mut egui::Ui, id: &str, value: &mut String, hint: &str, width: f32, normalize: fn(&str, &str) -> String) -> bool {
    let t = Tokens::get(ui.ctx());
    let id = ui.id().with(id);
    let mut buf = ui.data(|d| d.get_temp::<String>(id)).unwrap_or_else(|| value.clone());
    let (rect, _) = ui.allocate_exact_size(vec2(width, 24.0), egui::Sense::hover());
    crate::widgets::surface(ui, rect, t.field, false);
    if !t.bevel {
        ui.painter().rect_stroke(rect, t.radius_sm, Stroke::new(1.0, t.field_border), StrokeKind::Inside);
    }
    let inner = rect.shrink2(vec2(4.0, 2.0));
    let r = ui.put(inner, egui::TextEdit::singleline(&mut buf).id(id).char_limit(32).hint_text(hint).frame(egui::Frame::NONE).font(crate::theme::mono(12.0)));
    if r.has_focus() {
        ui.data_mut(|d| d.insert_temp(id, buf));
        return false;
    }
    ui.data_mut(|d| d.remove::<String>(id));
    if r.lost_focus() {
        let new = normalize(&buf, value);
        let changed = new != *value;
        *value = new;
        return changed;
    }
    false
}

/// The W x H x Resolution fields of the options bar: W, ⇄, H, the resolution and its unit.
pub fn fields(o: &mut ToolOptions, ui: &mut egui::Ui) {
    text_field(ui, "crop-width", &mut o.crop_width, tl!("Width"), 64.0, normalize_length);
    if crate::icons::button(ui, "arrow-left-right", 22.0, false, tl!("Swaps height and width")).clicked() {
        swap(o);
    }
    text_field(ui, "crop-height", &mut o.crop_height, tl!("Height"), 64.0, normalize_length);
    text_field(ui, "crop-resolution", &mut o.crop_resolution, tl!("Resolution"), 64.0, normalize_resolution);
    let units = [(PX_PER_IN.to_string(), "px/in"), (PX_PER_CM.to_string(), "px/cm")];
    let before = o.crop_resolution_unit.clone();
    if crate::widgets::dropdown(ui, "crop-resolution-unit", &mut o.crop_resolution_unit, &units, 64.0) && before != o.crop_resolution_unit {
        // The same resolution in the other unit, as in Photoshop.
        if let Some(v) = crate::widgets::parse_num(&o.crop_resolution.trim().replace(',', ".")).filter(|v| v.is_finite() && *v > 0.0) {
            let v = if o.crop_resolution_unit == PX_PER_CM { v / 2.54 } else { v * 2.54 };
            o.crop_resolution = crate::widgets::fmt_num2(v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_doc::{Color, ColorMode, Document, SampleType, Size};

    fn app(depth: SampleType) -> PhotocraftApp {
        let mut doc = Document::with_background("crop", Size::new(400, 300), ColorMode::Rgb, depth, Color::WHITE);
        doc.resolution_dpi = 72.0;
        let mut s = photocraft_engine::Session::new();
        s.add_document(doc, None);
        let mut app = PhotocraftApp::new(s, crate::Services::default());
        app.ui.tool = Tool::Crop;
        app.ui.extras.snap = false;
        app.ui.view.show.smart_guides = false;
        app
    }

    fn whr(w: &str, h: &str, res: &str, unit: &str) -> ToolOptions {
        ToolOptions {
            crop_ratio: WHR.into(),
            crop_width: w.into(),
            crop_height: h.into(),
            crop_resolution: res.into(),
            crop_resolution_unit: unit.into(),
            ..ToolOptions::default()
        }
    }

    #[test]
    fn custom_ratio_keeps_the_first_side_until_the_second_is_entered() {
        let mut options = ToolOptions::default();
        select_custom_ratio(&mut options);
        set_custom_ratio_fields(&mut options, 4.0, 0.0);

        assert_eq!(options.crop_ratio, "", "an incomplete ratio stays unconstrained");
        assert_eq!(custom_ratio_fields(&options, (0.0, 0.0)), (4.0, 0.0), "first entry survives the next frame");

        let (width, _) = custom_ratio_fields(&options, (0.0, 0.0));
        set_custom_ratio_fields(&mut options, width, 3.0);
        assert_eq!(options.crop_ratio, "4:3");
        assert_eq!(custom_ratio_fields(&options, (0.0, 0.0)), (4.0, 3.0));
        let (width, height) = crate::chrome_ui::crop_ratio(&options.crop_ratio, 0.0, 0.0).unwrap_or_default();
        assert_eq!(width / height, 4.0 / 3.0);
    }

    #[test]
    fn selecting_free_ratio_clears_previous_custom_dimensions() {
        let mut options = ToolOptions { crop_ratio: "16:9".into(), ..ToolOptions::default() };
        select_custom_ratio(&mut options);
        assert_eq!(custom_ratio_fields(&options, (0.0, 0.0)), (16.0, 9.0));

        options.crop_ratio.clear();
        select_custom_ratio(&mut options);
        assert_eq!(custom_ratio_fields(&options, (0.0, 0.0)), (0.0, 0.0));
    }

    #[test]
    fn lengths_parse_with_their_units() {
        assert_eq!(parse_length("4 in", Unit::Pixels), Some((4.0, Unit::Inches)));
        assert_eq!(parse_length("4in", Unit::Pixels), Some((4.0, Unit::Inches)));
        assert_eq!(parse_length(" 10 cm ", Unit::Pixels), Some((10.0, Unit::Centimeters)));
        assert_eq!(parse_length("25 MM", Unit::Pixels), Some((25.0, Unit::Millimeters)));
        assert_eq!(parse_length("1024 px", Unit::Inches), Some((1024.0, Unit::Pixels)));
        assert_eq!(parse_length("8,5 in", Unit::Pixels), Some((8.5, Unit::Inches)));
        assert_eq!(parse_length("12", Unit::Centimeters), Some((12.0, Unit::Centimeters)), "a bare number keeps the field's unit");
        assert_eq!(parse_length("3 pc", Unit::Pixels), Some((3.0, Unit::Picas)), "the shared unit table");
        for bad in ["", "in", "0 in", "-4 in", "4 furlongs", "NaN", "1e400 px", "4 in 5", "50 %", "50 percent"] {
            assert_eq!(parse_length(bad, Unit::Pixels), None, "{bad}");
        }
        assert_eq!(normalize_length("4in", ""), "4 in");
        assert_eq!(normalize_length("8.50 in", ""), "8.5 in");
        assert_eq!(normalize_length("7", "5 cm"), "7 cm", "bare numbers keep the unit typed before");
        assert_eq!(normalize_length("7", ""), "7 px");
        assert_eq!(normalize_length("nonsense", "5 cm"), "5 cm", "something else keeps the previous value");
        assert_eq!(normalize_length("  ", "5 cm"), "");
        assert_eq!(normalize_resolution("300", ""), "300");
        assert_eq!(normalize_resolution("-3", "72"), "72");
        assert_eq!(normalize_resolution("", "72"), "");
    }

    #[test]
    fn target_converts_physical_units_at_the_resolution_or_the_documents() {
        let close = |a: f64, b: f64| (a - b).abs() < 1e-6;
        let (w, h, r) = target(&whr("4 in", "5 in", "300", PX_PER_IN), 72.0).unwrap();
        assert!(close(w, 1200.0) && close(h, 1500.0) && r == Some(300.0));
        let (w, h, r) = target(&whr("10 cm", "25 mm", "100", PX_PER_CM), 72.0).unwrap();
        assert!(close(w, 1000.0) && close(h, 250.0), "{w} {h}");
        assert!(close(r.unwrap(), 254.0), "100 px/cm is 254 ppi");
        // No resolution: inches convert at the document's, and none is set.
        let (w, h, r) = target(&whr("2 in", "100 px", "", PX_PER_IN), 72.0).unwrap();
        assert!(close(w, 144.0) && close(h, 100.0) && r.is_none());
        // W or H missing, or another mode: no target (a free crop).
        assert_eq!(target(&whr("2 in", "", "300", PX_PER_IN), 72.0), None);
        assert_eq!(target(&ToolOptions { crop_ratio: "1:1".into(), ..whr("2 in", "3 in", "", PX_PER_IN) }, 72.0), None);
    }

    #[test]
    fn the_frame_holds_w_by_h() {
        let mut app = app(SampleType::U8);
        app.ui.tool_options = whr("4 in", "5 in", "300", PX_PER_IN);
        assert!((ratio(&app.ui.tool_options, 72.0).unwrap() - 0.8).abs() < 1e-9);
        let mods = Modifiers::NONE;
        use crate::canvas::{ToolEvent, tool_event};
        tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 10.0, pressure: 1.0 }, mods);
        tool_event(&mut app, ToolEvent::Move { x: 110.0, y: 300.0, pressure: 1.0 }, mods);
        tool_event(&mut app, ToolEvent::Up { x: 110.0, y: 300.0 }, mods);
        let r = app.ui.crop_rect.unwrap();
        assert!(((r[2] - r[0]) / (r[3] - r[1]) - 0.8).abs() < 1e-9, "{r:?}");
    }

    #[test]
    fn presets_front_image_clear_and_swap_fill_the_fields() {
        let mut app = app(SampleType::U8);
        app.ui.tool_options.crop_ratio = "preset:8.5x11in".into();
        chosen(&mut app);
        let o = &app.ui.tool_options;
        assert_eq!((o.crop_ratio.as_str(), o.crop_width.as_str(), o.crop_height.as_str(), o.crop_resolution.as_str()), (WHR, "8.5 in", "11 in", "300"));
        assert!(swap(&mut app.ui.tool_options));
        assert_eq!((app.ui.tool_options.crop_width.as_str(), app.ui.tool_options.crop_height.as_str()), ("11 in", "8.5 in"));
        app.ui.tool_options.crop_ratio = FRONT.into();
        chosen(&mut app);
        let o = &app.ui.tool_options;
        assert_eq!((o.crop_ratio.as_str(), o.crop_width.as_str(), o.crop_height.as_str(), o.crop_resolution.as_str()), (WHR, "400 px", "300 px", "72"));
        clear(&mut app.ui.tool_options);
        let o = &app.ui.tool_options;
        assert_eq!((o.crop_ratio.as_str(), o.crop_width.as_str(), o.crop_height.as_str(), o.crop_resolution.as_str()), (WHR, "", "", ""));
        // px/cm shows the document's resolution per centimetre.
        app.ui.tool_options.crop_resolution_unit = PX_PER_CM.into();
        front_image(&mut app);
        assert_eq!(app.ui.tool_options.crop_resolution, "28.35");
        // Outside the mode ⇄ is the ratio's.
        app.ui.tool_options.crop_ratio = "16:9".into();
        assert!(!swap(&mut app.ui.tool_options));
        assert!(dropdown_options().iter().any(|(k, _)| k == WHR) && dropdown_options().iter().any(|(k, _)| k == FRONT));
    }

    #[test]
    fn x_swaps_w_and_h_with_the_frame() {
        let mut app = app(SampleType::U8);
        app.ui.tool_options = whr("4 in", "6 in", "300", PX_PER_IN);
        app.ui.crop_rect = Some([0.0, 0.0, 40.0, 60.0]);
        assert!(crate::crop_ui::swap_orientation(&mut app));
        let o = &app.ui.tool_options;
        assert_eq!((o.crop_ratio.as_str(), o.crop_width.as_str(), o.crop_height.as_str()), (WHR, "6 in", "4 in"));
    }

    #[test]
    fn i_with_the_crop_tool_is_front_image() {
        let mut app = app(SampleType::U8);
        let ctx = egui::Context::default();
        ctx.input_mut(|i| i.events.push(egui::Event::Key { key: egui::Key::I, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE }));
        assert!(keys(&mut app, &ctx));
        assert_eq!((app.ui.tool_options.crop_ratio.as_str(), app.ui.tool_options.crop_width.as_str()), (WHR, "400 px"));
        // Another tool keeps I.
        let mut app2 = self::app(SampleType::U8);
        app2.ui.tool = Tool::Brush;
        ctx.input_mut(|i| i.events.push(egui::Event::Key { key: egui::Key::I, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE }));
        assert!(!keys(&mut app2, &ctx));
    }

    /// Committing a W x H x Resolution crop resamples the frame's area to exactly W x H px and sets
    /// the resolution, in one undo step, at 8 and 16 bits.
    #[test]
    fn choosing_a_size_with_the_canvas_ratio_commits_without_a_resize() {
        let mut app = app(SampleType::U8);
        crate::crop_ui::ensure_frame(&mut app);
        app.ui.tool_options = whr("800 px", "600 px", "92", PX_PER_IN);
        chosen(&mut app);
        assert_eq!(app.ui.crop_rect, Some([0.0, 0.0, 400.0, 300.0]));
        assert!(!app.crop.default_frame, "an output size is a pending crop even when the frame fits");
        crate::canvas::commit_crop(&mut app);
        let doc = &app.session.active().unwrap().doc;
        assert_eq!((doc.size, doc.resolution_dpi), (Size::new(800, 600), 92.0));
    }

    #[test]
    fn commit_resamples_to_the_target_size_and_resolution() {
        for depth in [SampleType::U8, SampleType::U16] {
            let mut app = app(depth);
            app.ui.tool_options = whr("4 in", "5 in", "300", PX_PER_IN);
            app.ui.crop_rect = Some([10.0, 10.0, 110.0, 135.0]);
            let mut p = json!({});
            commit_params(&app, &mut p);
            assert_eq!((p["targetWidth"].as_u64(), p["targetHeight"].as_u64(), p["resolution"].as_f64()), (Some(1200), Some(1500), Some(300.0)), "{p}");
            let past = app.session.active().unwrap().history.past_len();
            crate::canvas::commit_crop(&mut app);
            let st = app.session.active().unwrap();
            assert_eq!((st.doc.size, st.doc.resolution_dpi), (Size::new(1200, 1500), 300.0), "{depth:?}");
            assert_eq!(st.doc.pixel_format().sample, depth);
            assert_eq!(st.history.past_len(), past + 1, "{depth:?}: one undo step");
        }
        // Empty resolution: inches at the document's 72 ppi, which stays.
        let mut app = app(SampleType::U8);
        app.ui.tool_options = whr("2 in", "1 in", "", PX_PER_IN);
        app.ui.crop_rect = Some([0.0, 0.0, 100.0, 50.0]);
        crate::canvas::commit_crop(&mut app);
        let d = &app.session.active().unwrap().doc;
        assert_eq!((d.size, d.resolution_dpi), (Size::new(144, 72), 72.0));
        // W empty: a plain crop.
        let mut app = self::app(SampleType::U8);
        app.ui.tool_options = whr("", "1 in", "", PX_PER_IN);
        app.ui.crop_rect = Some([0.0, 0.0, 100.0, 50.0]);
        crate::canvas::commit_crop(&mut app);
        assert_eq!(app.session.active().unwrap().doc.size, Size::new(100, 50));
    }

    #[test]
    fn older_tool_options_load_without_the_new_fields() {
        let o: ToolOptions = serde_json::from_str(r#"{"crop_ratio": "1:1"}"#).unwrap();
        assert_eq!((o.crop_width.as_str(), o.crop_resolution_unit.as_str()), ("", PX_PER_IN));
    }
}
