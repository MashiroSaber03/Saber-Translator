//! Photoshop's Color Picker (Foreground / Background Color) dialog: a 2D colour field and a slider
//! for the selected component (H, S, B, R, G, B, L, a or b radio), new/current swatches with the
//! out-of-gamut and web-colour warnings, and HSB, RGB, Lab, CMYK and hex fields. The colour lives in the dialog fields (`color` as `#rrggbb`, plus the HSB
//! floats so hue survives greys), so `ui.dialog.set` drives it; OK runs `tools.setColors`.
//! While it is the top dialog the image is its eyedropper, as in Photoshop: the pointer over the
//! canvas is a pipette, and a click or drag there samples into the new colour ([`sample_at`]).

use egui::{Color32, Mesh, Rect, Sense, Stroke, StrokeKind, pos2, vec2};
use serde_json::{Map, Value, json};

use crate::PhotocraftApp;
use crate::state::DialogKind;
use crate::theme::Tokens;
use crate::widgets;

/// Radio components, Photoshop order: H, S, B, R, G, B.
pub const MODES: &[(&str, &str)] = &[("h", "H:"), ("s", "S:"), ("v", "B:"), ("r", "R:"), ("g", "G:"), ("b", "B:")];
/// Lab radio components (Photoshop's right-hand column): L, a, b.
pub const LAB_MODES: &[(&str, &str)] = &[("lab_l", "L:"), ("lab_a", "a:"), ("lab_b", "b:")];

/// The Lab colour at field position (x, y) in 0..1 (y down) with the slider at `z`, for a Lab
/// `mode`, as Photoshop 25.4 lays them out: L shows a across and b upwards; a shows b across and L
/// upwards; b shows a across and L upwards (a and b run -128..127, L 0..100).
fn field_lab(mode: &str, z: f32, x: f32, y: f32) -> Option<[f32; 3]> {
    let ab = |t: f32| t * 255.0 - 128.0;
    match mode {
        "lab_l" => Some([z * 100.0, ab(x), ab(1.0 - y)]),
        "lab_a" => Some([(1.0 - y) * 100.0, ab(z), ab(x)]),
        "lab_b" => Some([(1.0 - y) * 100.0, ab(x), ab(z)]),
        _ => None,
    }
}

/// Lab → sRGB, clamped to what the field can show.
fn lab_rgb(lab: [f32; 3]) -> [f32; 3] {
    photocraft_color::convert::lab_to_srgb(lab).map(|c| c.clamp(0.0, 1.0))
}

/// Colour differences the CMYK round trip leaves inside the printable gamut (ΔE ≤ 0.4, black ≈ 1.4
/// with black-point compensation) stay below this; out-of-gamut colours move far more.
const GAMUT_DE: f32 = 3.0;

/// The nearest printable colour, when `rgb` is outside the CMYK gamut (Photoshop's ⚠ warning).
pub fn out_of_gamut(rgb: [f32; 3]) -> Option<[f32; 3]> {
    use photocraft_color::convert::{cmyk_to_rgb, rgb_to_cmyk, srgb_to_lab};
    let back = cmyk_to_rgb(rgb_to_cmyk(rgb)).map(|c| c.clamp(0.0, 1.0));
    let (a, b) = (srgb_to_lab(rgb), srgb_to_lab(back));
    let de = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    (de > GAMUT_DE).then_some(back)
}

/// The nearest web colour (channels in steps of 51), when `rgb` isn't one.
pub fn not_web_safe(rgb: [f32; 3]) -> Option<[f32; 3]> {
    let web = rgb.map(|v| (v.clamp(0.0, 1.0) * 5.0).round() / 5.0);
    (hex(web) != hex(rgb)).then_some(web)
}

pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h = (h.rem_euclid(360.0)) / 60.0;
    let c = v * s;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [r + m, g + m, b + m]
}

pub fn rgb_to_hsv(c: [f32; 3]) -> [f32; 3] {
    let (mx, mn) = (c[0].max(c[1]).max(c[2]), c[0].min(c[1]).min(c[2]));
    let d = mx - mn;
    let h = if d <= 0.0 {
        0.0
    } else if mx == c[0] {
        60.0 * ((c[1] - c[2]) / d).rem_euclid(6.0)
    } else if mx == c[1] {
        60.0 * ((c[2] - c[0]) / d + 2.0)
    } else {
        60.0 * ((c[0] - c[1]) / d + 4.0)
    };
    [h, if mx <= 0.0 { 0.0 } else { d / mx }, mx]
}

pub fn hex(c: [f32; 3]) -> String {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", q(c[0]), q(c[1]), q(c[2]))
}

pub fn parse_hex(s: &str) -> Option<[f32; 3]> {
    let h = s.trim().trim_start_matches('#');
    if h.len() != 6 {
        return None;
    }
    // `get`, not `[..]`: typed text may be multi-byte, and a byte range can split a character.
    let c = |i: usize| u8::from_str_radix(h.get(i..i + 2)?, 16).ok().map(|v| f32::from(v) / 255.0);
    Some([c(0)?, c(2)?, c(4)?])
}

/// The colour at field position (x, y) in 0..1 (y down) for `mode` with the slider at `z` (0..1),
/// given the current HSV/RGB (used for the components the field doesn't vary).
pub fn field_color(mode: &str, z: f32, x: f32, y: f32) -> [f32; 3] {
    if let Some(lab) = field_lab(mode, z, x, y) {
        return lab_rgb(lab);
    }
    match mode {
        "s" => hsv_to_rgb(x * 360.0, z, 1.0 - y),
        "v" => hsv_to_rgb(x * 360.0, 1.0 - y, z),
        "r" => [z, 1.0 - y, x],
        "g" => [1.0 - y, z, x],
        "b" => [x, 1.0 - y, z],
        _ => hsv_to_rgb(z * 360.0, x, 1.0 - y),
    }
}

/// Field position and slider value of a colour in `mode` (inverse of [`field_color`]).
pub fn locate(mode: &str, hsv: [f32; 3], rgb: [f32; 3], lab: [f32; 3]) -> (f32, f32, f32) {
    let h = hsv[0] / 360.0;
    let ab = |v: f32| ((v + 128.0) / 255.0).clamp(0.0, 1.0);
    let l = (lab[0] / 100.0).clamp(0.0, 1.0);
    match mode {
        "lab_l" => (ab(lab[1]), 1.0 - ab(lab[2]), l),
        "lab_a" => (ab(lab[2]), 1.0 - l, ab(lab[1])),
        "lab_b" => (ab(lab[1]), 1.0 - l, ab(lab[2])),
        "s" => (h, 1.0 - hsv[2], hsv[1]),
        "v" => (h, 1.0 - hsv[1], hsv[2]),
        "r" => (rgb[2], 1.0 - rgb[1], rgb[0]),
        "g" => (rgb[2], 1.0 - rgb[0], rgb[1]),
        "b" => (rgb[0], 1.0 - rgb[1], rgb[2]),
        _ => (hsv[1], 1.0 - hsv[2], h),
    }
}

/// Colour for slider position `z` (0..1) in `mode`, other components from the current colour.
fn slider_color(mode: &str, z: f32, hsv: [f32; 3], rgb: [f32; 3], lab: [f32; 3]) -> [f32; 3] {
    if let Some(l) = slider_lab(mode, z, lab) {
        return lab_rgb(l);
    }
    match mode {
        "s" => hsv_to_rgb(hsv[0], z, hsv[2]),
        "v" => hsv_to_rgb(hsv[0], hsv[1], z),
        "r" => [z, rgb[1], rgb[2]],
        "g" => [rgb[0], z, rgb[2]],
        "b" => [rgb[0], rgb[1], z],
        _ => hsv_to_rgb(z * 360.0, 1.0, 1.0),
    }
}

/// The Lab colour with the slider of a Lab `mode` at `z`, the other components kept.
fn slider_lab(mode: &str, z: f32, lab: [f32; 3]) -> Option<[f32; 3]> {
    let ab = z * 255.0 - 128.0;
    match mode {
        "lab_l" => Some([z * 100.0, lab[1], lab[2]]),
        "lab_a" => Some([lab[0], ab, lab[2]]),
        "lab_b" => Some([lab[0], lab[1], ab]),
        _ => None,
    }
}

fn c32(c: [f32; 3]) -> Color32 {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    Color32::from_rgb(q(c[0]), q(c[1]), q(c[2]))
}

/// Open the picker for the foreground or background colour.
pub fn open(app: &mut PhotocraftApp, target: &str) -> u64 {
    let c = if target == "background" { app.session.tools.background } else { app.session.tools.foreground };
    let rgb = [c[0], c[1], c[2]];
    let label = if target == "background" { "Color Picker (Background Color)" } else { "Color Picker (Foreground Color)" };
    let mut fields = initial_fields(target, label, rgb);
    // A colour that was picked as CMYK shows the inks it was picked as, not what its sRGB stand-in converts to (#4196).
    if let Some(cmyk) = app.session.tools.picked_cmyk(rgb) {
        fields.insert("__cmyk".into(), json!(cmyk));
    }
    app.ui.open_dialog(DialogKind::Command, fields)
}

/// Shared initial state for every target; the picker layout is independent of its destination.
pub(crate) fn initial_fields(target: &str, label: &str, rgb: [f32; 3]) -> Map<String, Value> {
    let mut f = Map::new();
    f.insert("__colorPicker".into(), json!(target));
    f.insert("__label".into(), json!(label));
    f.insert("color".into(), json!(hex(rgb)));
    f.insert("__orig".into(), json!(hex(rgb)));
    f.insert("__hsv".into(), json!(rgb_to_hsv(rgb)));
    f.insert("__mode".into(), json!("h"));
    f.insert("__webOnly".into(), json!(false));
    f
}

/// Open the picker on `rgb` for something other than the tool colours: OK runs `command` with
/// `params` plus `"color": "#rrggbb"` (e.g. a gradient stop's colour).
pub fn open_for_command(app: &mut PhotocraftApp, label: &str, rgb: [f32; 3], command: &str, params: Value) -> u64 {
    let mut f = initial_fields("command", label, rgb);
    f.insert("__command".into(), json!(command));
    f.insert("__params".into(), params);
    app.ui.open_dialog(DialogKind::Command, f)
}

/// Open the picker on `rgb` for a colour field of another dialog (e.g. Edit › Fill's Color…): OK
/// writes `"#rrggbb"` into `field` of dialog `dialog`, which stays open; Cancel leaves it alone.
pub fn open_for_field(app: &mut PhotocraftApp, label: &str, rgb: [f32; 3], dialog: u64, field: &str) -> u64 {
    let hsv = rgb_to_hsv(rgb);
    let mut f = Map::new();
    f.insert("__colorPicker".into(), json!("field"));
    f.insert("__label".into(), json!(label));
    f.insert("__dialog".into(), json!(dialog));
    f.insert("__field".into(), json!(field));
    f.insert("color".into(), json!(hex(rgb)));
    f.insert("__orig".into(), json!(hex(rgb)));
    f.insert("__hsv".into(), json!(hsv));
    f.insert("__mode".into(), json!("h"));
    f.insert("__webOnly".into(), json!(false));
    app.ui.open_dialog(DialogKind::Command, f)
}

/// Open the picker on `rgb` for a colour the shell keeps itself (e.g. the Crop shield's custom
/// colour): OK hands `"#rrggbb"` to `target`'s setter in [`confirm`]; Cancel leaves it alone.
pub fn open_for_target(app: &mut PhotocraftApp, target: &str, label: &str, rgb: [f32; 3]) -> u64 {
    let hsv = rgb_to_hsv(rgb);
    let mut f = Map::new();
    f.insert("__colorPicker".into(), json!(target));
    f.insert("__label".into(), json!(label));
    f.insert("color".into(), json!(hex(rgb)));
    f.insert("__orig".into(), json!(hex(rgb)));
    f.insert("__hsv".into(), json!(hsv));
    f.insert("__mode".into(), json!("h"));
    f.insert("__webOnly".into(), json!(false));
    app.ui.open_dialog(DialogKind::Command, f)
}

/// Set by a dialog body (to the picker's title) to open the Color Picker on its `color` field.
const REQUEST: &str = "__pickColor";

/// A dialog body asks for the Color Picker on its `color` field (Edit › Fill's Color… and colour
/// swatch, Edit › Stroke's swatch). The dialog loop opens it ([`open_requested`]), since that
/// needs the dialog's id.
pub fn request(f: &mut Map<String, Value>, title: &str) {
    request_field(f, "color", title);
}

/// [`request`] for another `"#rrggbb"` field of the dialog: a top-level key (`vineColor`), or a
/// JSON pointer into the fields (`/values/guidesGridAndSlices/guideColor`).
pub fn request_field(f: &mut Map<String, Value>, field: &str, title: &str) {
    f.insert(REQUEST.into(), json!({ "title": title, "field": field }));
}

/// The picker's title for the colour called `name`: "Color Picker (Vine Color)".
pub fn title_for(name: &str) -> String {
    crate::i18n::fmt(tl!("Color Picker ({name})"), &[("name", name)])
}

/// The `"#rrggbb"` at `field` of `f` (a key or a JSON pointer).
fn field_value<'a>(f: &'a Map<String, Value>, field: &str) -> Option<&'a str> {
    match field.strip_prefix('/') {
        Some(rest) => {
            let (first, tail) = rest.split_once('/').unwrap_or((rest, ""));
            let v = f.get(first)?;
            if tail.is_empty() { v.as_str() } else { v.pointer(&format!("/{tail}"))?.as_str() }
        }
        None => f.get(field)?.as_str(),
    }
}

/// Sets `field` of `f` (a key or a JSON pointer to an existing value) to `value`.
fn set_field(f: &mut Map<String, Value>, field: &str, value: Value) -> Result<(), String> {
    match field.strip_prefix('/') {
        Some(rest) => {
            let (first, tail) = rest.split_once('/').unwrap_or((rest, ""));
            let v = f.get_mut(first).ok_or("the dialog has no such colour")?;
            let slot = if tail.is_empty() { v } else { v.pointer_mut(&format!("/{tail}")).ok_or("the dialog has no such colour")? };
            *slot = value;
        }
        None => {
            f.insert(field.into(), value);
        }
    }
    Ok(())
}

/// Opens the Color Picker dialog `id`'s body asked for ([`request`], [`request_field`]), on its
/// colour: OK sets that field of the dialog (not the foreground), Cancel leaves it as it was.
pub fn open_requested(app: &mut PhotocraftApp, id: u64) {
    let Some(d) = app.ui.dialog_mut(id) else { return };
    let (title, field) = match d.fields.remove(REQUEST) {
        Some(Value::String(title)) => (title, "color".to_string()),
        Some(Value::Object(o)) => {
            let get = |k: &str| o.get(k).and_then(Value::as_str).map(str::to_string);
            let (Some(title), Some(field)) = (get("title"), get("field")) else { return };
            (title, field)
        }
        _ => return,
    };
    let rgb = field_value(&d.fields, &field).and_then(parse_hex).unwrap_or([0.0; 3]);
    open_for_field(app, &title, rgb, id, &field);
}

pub fn owns(f: &Map<String, Value>) -> bool {
    f.contains_key("__colorPicker")
}

/// Alt+Cancel (Reset): restore the colour from when the picker opened (same as clicking Current).
pub fn reset(f: &mut Map<String, Value>) {
    if !owns(f) {
        return;
    }
    let orig = f.get("__orig").and_then(Value::as_str).and_then(parse_hex).unwrap_or([0.0; 3]);
    set_rgb(f, orig, Keep::Nothing);
}

/// The Color Picker, when it is the top dialog (another dialog opened over it takes the input).
pub fn top(app: &PhotocraftApp) -> Option<u64> {
    app.ui.dialogs.last().filter(|d| owns(&d.fields)).map(|d| d.id)
}

/// Eyedropper: the top Color Picker's new colour becomes the image's composite colour at document
/// point (x, y). Off the image or over transparency nothing changes.
pub fn sample_at(app: &mut PhotocraftApp, x: f64, y: f64) {
    let Some(id) = top(app) else { return };
    let Some(rgb) = crate::canvas::composite_color(app, x, y) else { return };
    if let Some(d) = app.ui.dialog_mut(id) {
        set_rgb(&mut d.fields, rgb, Keep::Nothing);
    }
}

/// The colour in every model the dialog shows.
pub struct Components {
    pub rgb: [f32; 3],
    pub hsv: [f32; 3],
    pub lab: [f32; 3],
    pub cmyk: [f32; 4],
}

/// Component values an edit stores next to the new `color`. sRGB alone loses them: greys have no
/// hue, and Lab or CMYK values outside sRGB are clamped. Typing "60" into L passes through L = 6
/// first; without the stored Lab the clamped colour would change `a` and `b` under the user.
enum Keep {
    Nothing,
    Hsv([f32; 3]),
    Lab([f32; 3]),
    Cmyk([f32; 4]),
}

/// The stored `key` components, if they still produce `rgb`.
fn kept<const N: usize>(f: &Map<String, Value>, key: &str, rgb: [f32; 3], to_rgb: impl Fn([f32; N]) -> [f32; 3]) -> Option<[f32; N]>
where
    [f32; N]: serde::de::DeserializeOwned,
{
    let v: [f32; N] = f.get(key).and_then(|v| serde_json::from_value(v.clone()).ok())?;
    (hex(to_rgb(v)) == hex(rgb)).then_some(v)
}

/// Current colour, keeping the stored HSB, Lab or CMYK values while they still match `color`.
pub fn current(f: &Map<String, Value>) -> Components {
    let rgb = f.get("color").and_then(Value::as_str).and_then(parse_hex).unwrap_or([0.0; 3]);
    Components {
        rgb,
        hsv: kept(f, "__hsv", rgb, |h| hsv_to_rgb(h[0], h[1], h[2])).unwrap_or_else(|| rgb_to_hsv(rgb)),
        lab: kept(f, "__lab", rgb, photocraft_color::convert::lab_to_srgb).unwrap_or_else(|| photocraft_color::convert::srgb_to_lab(rgb)),
        cmyk: kept(f, "__cmyk", rgb, photocraft_color::convert::cmyk_to_rgb).unwrap_or_else(|| photocraft_color::convert::rgb_to_cmyk(rgb)),
    }
}

fn set_rgb(f: &mut Map<String, Value>, rgb: [f32; 3], keep: Keep) {
    let web = f.get("__webOnly").and_then(Value::as_bool).unwrap_or(false);
    let (rgb, keep) = if web { (rgb.map(|v| (v * 5.0).round() / 5.0), Keep::Nothing) } else { (rgb, keep) };
    f.insert("color".into(), json!(hex(rgb)));
    for key in ["__hsv", "__lab", "__cmyk"] {
        f.remove(key);
    }
    match keep {
        Keep::Nothing => {}
        Keep::Hsv(h) => {
            f.insert("__hsv".into(), json!(h));
        }
        Keep::Lab(l) => {
            f.insert("__lab".into(), json!(l));
        }
        Keep::Cmyk(k) => {
            f.insert("__cmyk".into(), json!(k));
        }
    }
}

fn grid_mesh(rect: Rect, n: usize, color: impl Fn(f32, f32) -> [f32; 3]) -> Mesh {
    let mut m = Mesh::default();
    for j in 0..=n {
        for i in 0..=n {
            let (x, y) = (i as f32 / n as f32, j as f32 / n as f32);
            m.colored_vertex(pos2(rect.left() + x * rect.width(), rect.top() + y * rect.height()), c32(color(x, y)));
        }
    }
    let w = (n + 1) as u32;
    for j in 0..n as u32 {
        for i in 0..n as u32 {
            let a = j * w + i;
            m.add_triangle(a, a + 1, a + w + 1);
            m.add_triangle(a, a + w + 1, a + w);
        }
    }
    m
}

/// Cells per side of the banded Only Web Colors field: fine enough that the band edges follow the
/// colour model, coarse enough to stay cheap to draw each frame.
const WEB_CELLS: usize = 64;

/// Snap each channel to the web-safe ramp (00, 33, 66, 99, CC, FF).
fn web_safe(c: [f32; 3]) -> [f32; 3] {
    c.map(|v| (v.clamp(0.0, 1.0) * 5.0).round() / 5.0)
}

/// Like [`grid_mesh`], but every cell is one flat colour sampled at its centre, so a quantized
/// field shows hard bands instead of blending between the snapped corners.
fn flat_grid_mesh(rect: Rect, n: usize, color: impl Fn(f32, f32) -> [f32; 3]) -> Mesh {
    let mut m = Mesh::default();
    let n = n.max(1);
    let (cw, ch) = (rect.width() / n as f32, rect.height() / n as f32);
    for j in 0..n {
        for i in 0..n {
            let c = c32(color((i as f32 + 0.5) / n as f32, (j as f32 + 0.5) / n as f32));
            let (x0, y0) = (rect.left() + i as f32 * cw, rect.top() + j as f32 * ch);
            let a = m.vertices.len() as u32;
            for (dx, dy) in [(0.0, 0.0), (cw, 0.0), (cw, ch), (0.0, ch)] {
                m.colored_vertex(pos2(x0 + dx, y0 + dy), c);
            }
            m.add_triangle(a, a + 1, a + 2);
            m.add_triangle(a, a + 2, a + 3);
        }
    }
    m
}

/// Room for both component columns, including units, without overflowing the title's width; with
/// the frame and title the window fits Photoshop's (about 534 x 376 points).
pub(crate) const CONTENT_WIDTH: f32 = 488.0;
/// Side of the colour field (Photoshop's is about 256 points).
const FIELD: f32 = 200.0;
/// Width of the component fields.
const VALUE_W: f32 = 42.0;

/// Draw the picker and its top-right actions. `Some(true)` confirms, `Some(false)` cancels.
pub fn body(ui: &mut egui::Ui, f: &mut Map<String, Value>) -> Option<bool> {
    ui.set_width(CONTENT_WIDTH);
    ui.spacing_mut().item_spacing.y = 4.0;
    let mut outcome = None;
    ui.horizontal_top(|ui| {
        ui.spacing_mut().item_spacing.x = 12.0;
        ui.vertical(|ui| {
            ui.horizontal_top(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                color_field(ui, f);
            });
            ui.add_space(8.0);
            let mut web = f.get("__webOnly").and_then(Value::as_bool).unwrap_or(false);
            if widgets::checkbox(ui, &mut web, tl!("Only Web Colors")).changed() {
                f.insert("__webOnly".into(), json!(web));
                set_rgb(f, current(f).rgb, Keep::Nothing);
            }
            // Under the colour field, not in the action column: the dialog must still fit a
            // 760 x 480 window.
            ui.add_space(4.0);
            if let Some(rgb) = crate::screen_picker::button(ui, ui.id().with("screen-color")) {
                set_rgb(f, rgb, Keep::Nothing);
            }
        });
        ui.vertical(|ui| {
            ui.horizontal_top(|ui| {
                swatches(ui, f);
                // As wide as the longest label, at the right edge like Photoshop's.
                let button_width = widgets::button_width(ui, &[tl!("OK"), tl!("Cancel"), tl!("Add to Swatches")]).max(96.0);
                ui.with_layout(egui::Layout::top_down(egui::Align::Max), |ui| {
                    if widgets::primary_button(ui, tl!("OK"), button_width).clicked() {
                        outcome = Some(true);
                    }
                    ui.add_space(4.0);
                    if widgets::secondary_button(ui, tl!("Cancel"), button_width).clicked() {
                        outcome = Some(false);
                    }
                    ui.add_space(4.0);
                    // Handled by the dialog host, which has the app (`take_add_swatch`).
                    if widgets::secondary_button(ui, tl!("Add to Swatches"), button_width).clicked() {
                        f.insert("__addSwatch".into(), json!(true));
                    }
                });
            });
            ui.add_space(4.0);
            let mode = f.get("__mode").and_then(Value::as_str).unwrap_or("h").to_string();
            fields(ui, f, &mode, &current(f));
        });
    });
    outcome
}

fn color_field(ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    let t = Tokens::get(ui.ctx());
    let mode = f.get("__mode").and_then(Value::as_str).unwrap_or("h").to_string();
    let now = current(f);
    let (rgb, hsv, lab) = (now.rgb, now.hsv, now.lab);
    let (fx, fy, fz) = locate(&mode, hsv, rgb, lab);
    // Colour field.
    let (field, resp) = ui.allocate_exact_size(vec2(FIELD, FIELD), Sense::click_and_drag());
    let web_only = f.get("__webOnly").and_then(Value::as_bool).unwrap_or(false);
    // Only Web Colors bands the field to the 216 web-safe colours, as in Photoshop (#4107).
    let mesh = |rect: Rect, color: &dyn Fn(f32, f32) -> [f32; 3]| {
        if web_only { flat_grid_mesh(rect, WEB_CELLS, |x, y| web_safe(color(x, y))) } else { grid_mesh(rect, 32, color) }
    };
    ui.painter().add(mesh(field, &|x, y| field_color(&mode, fz, x, y)));
    ui.painter().rect_stroke(field, 0.0, Stroke::new(1.0, t.field_border), StrokeKind::Outside);
    let marker = pos2(field.left() + fx * field.width(), field.top() + fy * field.height());
    ui.painter().circle_stroke(marker, 5.0, Stroke::new(1.5, if hsv[2] > 0.6 && hsv[1] < 0.4 { Color32::BLACK } else { Color32::WHITE }));
    if let Some(p) = resp.interact_pointer_pos().filter(|_| resp.dragged() || resp.clicked()) {
        let (x, y) = (((p.x - field.left()) / field.width()).clamp(0.0, 1.0), ((p.y - field.top()) / field.height()).clamp(0.0, 1.0));
        let c = field_color(&mode, fz, x, y);
        let keep = match mode.as_str() {
            "h" => Keep::Hsv([fz * 360.0, x, 1.0 - y]),
            "s" => Keep::Hsv([x * 360.0, fz, 1.0 - y]),
            "v" => Keep::Hsv([x * 360.0, 1.0 - y, fz]),
            m => field_lab(m, fz, x, y).map_or(Keep::Nothing, Keep::Lab),
        };
        set_rgb(f, c, keep);
    }
    // Component slider (hue runs 360° at the top to 0° at the bottom, like Photoshop).
    let (strip, sresp) = ui.allocate_exact_size(vec2(18.0, FIELD), Sense::click_and_drag());
    ui.painter().add(mesh(strip, &|_, y| slider_color(&mode, 1.0 - y, hsv, rgb, lab)));
    let sy = strip.top() + (1.0 - fz) * strip.height();
    for (x, dir) in [(strip.left() - 1.0, 1.0f32), (strip.right() + 1.0, -1.0)] {
        let tri = vec![pos2(x, sy), pos2(x - 6.0 * dir, sy - 4.0), pos2(x - 6.0 * dir, sy + 4.0)];
        ui.painter().add(egui::Shape::convex_polygon(tri, t.text, Stroke::NONE));
    }
    if let Some(p) = sresp.interact_pointer_pos().filter(|_| sresp.dragged() || sresp.clicked()) {
        let z = 1.0 - ((p.y - strip.top()) / strip.height()).clamp(0.0, 1.0);
        let h = match mode.as_str() {
            "h" => Some([z * 360.0, hsv[1], hsv[2]]),
            "s" => Some([hsv[0], z, hsv[2]]),
            "v" => Some([hsv[0], hsv[1], z]),
            _ => None,
        };
        match (h, slider_lab(&mode, z, lab)) {
            (Some(h), _) => set_rgb(f, hsv_to_rgb(h[0], h[1], h[2]), Keep::Hsv(h)),
            (None, Some(l)) => set_rgb(f, lab_rgb(l), Keep::Lab(l)),
            (None, None) => set_rgb(f, slider_color(&mode, z, hsv, rgb, lab), Keep::Nothing),
        }
    }
}

fn swatches(ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    let t = Tokens::get(ui.ctx());
    let rgb = current(f).rgb;
    let orig = f.get("__orig").and_then(Value::as_str).and_then(parse_hex).unwrap_or(rgb);
    let (area, _) = ui.allocate_exact_size(vec2(56.0 + 24.0, 92.0), Sense::hover());
    // new / current swatches, labelled above and below.
    let sw = Rect::from_min_size(area.min + vec2(0.0, 16.0), vec2(56.0, 64.0));
    let p = ui.painter();
    let label = |text: &str, at: egui::Pos2, align: egui::Align2| {
        p.text(at, align, text, egui::FontId::proportional(11.0), t.text_dim);
    };
    label(tl!("new"), pos2(sw.center().x, sw.top() - 3.0), egui::Align2::CENTER_BOTTOM);
    label(tl!("current"), pos2(sw.center().x, sw.bottom() + 3.0), egui::Align2::CENTER_TOP);
    p.rect_filled(Rect::from_min_size(sw.min, vec2(56.0, 32.0)), 0.0, c32(rgb));
    let cur = Rect::from_min_size(sw.min + vec2(0.0, 32.0), vec2(56.0, 32.0));
    p.rect_filled(cur, 0.0, c32(orig));
    p.rect_stroke(sw, 0.0, Stroke::new(1.0, t.field_border), StrokeKind::Outside);
    let click_cur = ui.interact(cur, ui.id().with("cp-current"), Sense::click());
    click_cur.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), tl!("Click to restore the current colour")));
    if click_cur.on_hover_text(tl!("Click to restore the current colour")).clicked() {
        set_rgb(f, orig, Keep::Nothing);
    }
    // Photoshop's warnings beside the swatch, each over a chip of the nearest colour that passes
    // (a click takes it): out of gamut for printing on top, not a web colour below.
    let x = sw.right() + 8.0;
    let gamut = tl!("Out of gamut for printing: click for the nearest printable colour");
    let web = tl!("Not a web colour: click for the nearest web colour");
    for (slot, warning, tip) in [(0.0, out_of_gamut(rgb), gamut), (34.0, not_web_safe(rgb), web)] {
        let Some(near) = warning else { continue };
        let icon = Rect::from_min_size(pos2(x, sw.top() + 2.0 + slot), vec2(13.0, 13.0));
        let chip = Rect::from_min_size(pos2(x + 1.5, icon.bottom() + 3.0), vec2(10.0, 10.0));
        let p = ui.painter();
        if slot == 0.0 {
            warning_triangle(p, icon, t.text, t.chrome);
        } else {
            web_cube(p, icon, t.text);
        }
        p.rect_filled(chip, 0.0, c32(near));
        p.rect_stroke(chip, 0.0, Stroke::new(1.0, t.text_dim), StrokeKind::Outside);
        let resp = ui.interact(icon.union(chip), ui.id().with(("cp-warning", slot as i32)), Sense::click());
        resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), tip));
        if resp.on_hover_text(tip).clicked() {
            set_rgb(f, near, Keep::Nothing);
        }
    }
}

/// ⚠: a filled triangle with an exclamation mark cut out of it.
fn warning_triangle(p: &egui::Painter, r: Rect, ink: Color32, cut: Color32) {
    let tri = vec![pos2(r.center().x, r.top()), pos2(r.right(), r.bottom()), pos2(r.left(), r.bottom())];
    p.add(egui::Shape::convex_polygon(tri, ink, Stroke::NONE));
    let x = r.center().x;
    p.line_segment([pos2(x, r.top() + 4.5), pos2(x, r.bottom() - 4.0)], Stroke::new(1.5, cut));
    p.circle_filled(pos2(x, r.bottom() - 2.0), 0.9, cut);
}

/// A cube seen corner-on: a hexagon with three edges meeting in the middle.
fn web_cube(p: &egui::Painter, r: Rect, ink: Color32) {
    let (c, rad) = (r.center(), r.width() / 2.0);
    let corner = |k: f32| {
        let a = std::f32::consts::FRAC_PI_3 * k - std::f32::consts::FRAC_PI_2;
        c + vec2(rad * a.cos(), rad * a.sin())
    };
    let hex: Vec<_> = (0..6).map(|k| corner(k as f32)).collect();
    let stroke = Stroke::new(1.2, ink);
    p.add(egui::Shape::closed_line(hex, stroke));
    for k in [1.0, 3.0, 5.0] {
        p.line_segment([c, corner(k)], stroke);
    }
}

/// A component radio: true when clicked.
fn radio(ui: &mut egui::Ui, on: bool, label: &str) -> bool {
    let t = Tokens::get(ui.ctx());
    let (r, resp) = ui.allocate_exact_size(vec2(14.0, 14.0), Sense::click());
    ui.painter().circle_stroke(r.center(), 5.5, Stroke::new(1.2, if on { t.accent } else { t.text_faint }));
    if on {
        ui.painter().circle_filled(r.center(), 3.0, t.accent);
    }
    resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::RadioButton, ui.is_enabled(), on, label));
    resp.clicked()
}

/// Component radios and numeric fields: HSB, RGB, Lab, CMYK and hex.
fn fields(ui: &mut egui::Ui, f: &mut Map<String, Value>, mode: &str, now: &Components) {
    let t = Tokens::get(ui.ctx());
    let &Components { rgb, hsv, lab, cmyk } = now;
    let mut edit: Option<([f32; 3], Keep)> = None;
    let mut new_mode: Option<&str> = None;
    ui.horizontal_top(|ui| {
        ui.spacing_mut().item_spacing.x = 10.0;
        ui.vertical(|ui| {
            egui::Grid::new("cp-hsb-rgb").min_row_height(0.0).num_columns(4).spacing([4.0, 2.0]).show(ui, |ui| {
                let vals = [hsv[0], hsv[1] * 100.0, hsv[2] * 100.0, rgb[0] * 255.0, rgb[1] * 255.0, rgb[2] * 255.0];
                let units = ["°", "%", "%", "", "", ""];
                let ranges = [0.0..=360.0, 0.0..=100.0, 0.0..=100.0, 0.0..=255.0, 0.0..=255.0, 0.0..=255.0];
                for i in 0..6 {
                    let (key, label) = MODES[i];
                    let on = mode == key;
                    if radio(ui, on, label) {
                        new_mode = Some(key);
                    }
                    ui.label(egui::RichText::new(tl!(&label)).color(t.text_dim));
                    let mut v = vals[i].round();
                    if widgets::value_field(ui, &mut v, ranges[i].clone(), "", VALUE_W).changed() {
                        let (mut h, mut c) = (hsv, rgb);
                        if i < 3 {
                            h[i] = if i == 0 { v } else { v / 100.0 };
                            c = hsv_to_rgb(h[0], h[1], h[2]);
                            edit = Some((c, Keep::Hsv(h)));
                        } else {
                            c[i - 3] = v / 255.0;
                            edit = Some((c, Keep::Nothing));
                        }
                    }
                    ui.label(egui::RichText::new(units[i]).color(t.text_faint));
                    ui.end_row();
                    if i == 2 {
                        ui.allocate_exact_size(vec2(0.0, 3.0), Sense::hover());
                        ui.end_row();
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                ui.label(egui::RichText::new("#").color(t.text_dim));
                // While the field has focus it shows what is being typed, not the colour: rebuilding it from
                // the colour every frame would throw away every keystroke until six valid digits were in.
                let id = ui.id().with("cp-hex");
                crate::field_tab::register(ui.ctx(), id);
                if !f.contains_key("__focused") {
                    f.insert("__focused".into(), json!(true));
                    crate::field_tab::focus(ui.ctx(), id);
                    let current_hex = hex(rgb);
                    crate::field_tab::select_all(ui.ctx(), id, current_hex.trim_start_matches('#'));
                    ui.memory_mut(|m| m.request_focus(id));
                }
                let typing = if ui.memory(|m| m.has_focus(id)) { ui.data(|d| d.get_temp::<String>(id)) } else { None };
                let mut h = typing.unwrap_or_else(|| hex(rgb).trim_start_matches('#').to_ascii_uppercase());
                let r = crate::widgets::text_field_ex(ui, id, &mut h, "", 64.0, true, Some(6));
                if r.gained_focus() {
                    crate::field_tab::select_all(ui.ctx(), id, &h);
                }
                if r.has_focus() {
                    ui.data_mut(|d| d.insert_temp(id, h.clone()));
                } else {
                    ui.data_mut(|d| d.remove::<String>(id));
                }
                if r.changed()
                    && let Some(c) = parse_hex(&h)
                {
                    edit = Some((c, Keep::Nothing));
                }
            });
        });
        egui::Grid::new("cp-lab-cmyk").min_row_height(0.0).num_columns(4).spacing([4.0, 2.0]).show(ui, |ui| {
            for (i, (v, range)) in [(lab[0], 0.0..=100.0), (lab[1], -128.0..=127.0), (lab[2], -128.0..=127.0)].into_iter().enumerate() {
                let (key, label) = LAB_MODES[i];
                if radio(ui, mode == key, label) {
                    new_mode = Some(key);
                }
                ui.label(egui::RichText::new(tl!(&label)).color(t.text_dim));
                let mut x = v.round();
                if widgets::value_field(ui, &mut x, range, "", VALUE_W).changed() {
                    let mut l = lab;
                    l[i] = x;
                    edit = Some((photocraft_color::convert::lab_to_srgb(l).map(|c| c.clamp(0.0, 1.0)), Keep::Lab(l)));
                }
                ui.label("");
                ui.end_row();
            }
            ui.allocate_exact_size(vec2(0.0, 3.0), Sense::hover());
            ui.end_row();
            for (i, label) in ["C:", "M:", "Y:", "K:"].into_iter().enumerate() {
                ui.allocate_exact_size(vec2(14.0, 14.0), Sense::hover());
                ui.label(egui::RichText::new(tl!(&label)).color(t.text_dim));
                let mut x = (cmyk[i] * 100.0).round();
                if widgets::value_field(ui, &mut x, 0.0..=100.0, "", VALUE_W).changed() {
                    let mut k = cmyk;
                    k[i] = x / 100.0;
                    edit = Some((photocraft_color::convert::cmyk_to_rgb(k).map(|c| c.clamp(0.0, 1.0)), Keep::Cmyk(k)));
                }
                ui.label(egui::RichText::new("%").color(t.text_faint));
                ui.end_row();
            }
        });
    });
    if let Some(m) = new_mode {
        f.insert("__mode".into(), json!(m));
    }
    if let Some((c, keep)) = edit {
        set_rgb(f, c, keep);
    }
}

/// Add to Swatches was clicked: add the picked colour to the Swatches panel.
pub fn take_add_swatch(app: &mut PhotocraftApp, f: &mut Map<String, Value>) {
    if f.remove("__addSwatch").and_then(|v| v.as_bool()) == Some(true) {
        crate::swatches_ui::add_from_picker(app, f);
    }
}

/// OK: set the foreground or background colour.
pub fn confirm(app: &mut PhotocraftApp, f: &Map<String, Value>) -> Result<Value, String> {
    if crate::type_panels_ui::color_picker::owns(f) {
        return crate::type_panels_ui::color_picker::confirm(app, f);
    }
    if crate::layer_style::color_picker::owns(f) {
        return crate::layer_style::color_picker::confirm(app, f);
    }
    if crate::solid_fill_ui::owns(f) {
        return crate::solid_fill_ui::confirm(app, f);
    }
    if crate::shape_fill_ui::owns(f) {
        return crate::shape_fill_ui::confirm(app, f);
    }
    let target = f.get("__colorPicker").and_then(Value::as_str).unwrap_or("foreground");
    let color = f.get("color").and_then(Value::as_str).unwrap_or("#000000");
    if let Some(cmd) = f.get("__command").and_then(Value::as_str) {
        let mut p = f.get("__params").cloned().filter(Value::is_object).unwrap_or_else(|| json!({}));
        p["color"] = json!(color);
        return app.run(cmd, p);
    }
    if target == "field" {
        let (Some(dialog), Some(field)) = (f.get("__dialog").and_then(Value::as_u64), f.get("__field").and_then(Value::as_str)) else {
            return Err("the Color Picker has no field to set".into());
        };
        let d = app.ui.dialog_mut(dialog).ok_or("the dialog this colour was for is closed")?;
        set_field(&mut d.fields, field, json!(color))?;
        return Ok(json!({ "color": color }));
    }
    if target == "cropShield" {
        crate::crop_shield::set_custom_color(app, color)?;
        return Ok(json!({ "color": color }));
    }
    // The Note tool's colour for new notes (#2144).
    if target == "noteColor" {
        app.ui.analysis.note_color = parse_hex(color).ok_or("not a colour")?;
        return Ok(json!({ "color": color }));
    }
    if target == "pasteboard" {
        // The pasteboard menu's Select Custom Color…: the custom colour, and use it.
        return app.run("prefs.set", json!({"values": {"interface.canvasColor": "custom", "interface.canvasCustomColor": color}}));
    }
    let mut params = json!({ target: color });
    // The CMYK values the person entered stay the colour's inks: a CMYK document is filled with them (#4196).
    if let Some(cmyk) = parse_hex(color).and_then(|rgb| kept::<4>(f, "__cmyk", rgb, photocraft_color::convert::cmyk_to_rgb)) {
        params[format!("{target}Cmyk")] = json!(cmyk);
    }
    let r = app.run("tools.setColors", params)?;
    if target == "foreground" {
        crate::type_tool::foreground_changed(app);
    }
    Ok(r)
}

#[cfg(test)]
mod tests {
    use egui::accesskit::Role;
    use egui_kittest::Harness;
    use egui_kittest::kittest::Queryable;

    use super::*;

    /// #4107: with Only Web Colors on, the field is drawn in flat web-safe bands.
    #[test]
    fn web_only_field_is_flat_web_safe_bands() {
        let rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(200.0, 200.0));
        let m = flat_grid_mesh(rect, WEB_CELLS, |x, y| web_safe(field_color("h", 0.3, x, y)));
        let ramp = [0u8, 51, 102, 153, 204, 255];
        for v in &m.vertices {
            assert!([v.color.r(), v.color.g(), v.color.b()].iter().all(|c| ramp.contains(c)), "{:?}", v.color);
        }
        // Each cell's four corners share one colour (no blending between bands).
        for q in m.vertices.chunks(4) {
            assert!(q.iter().all(|v| v.color == q[0].color));
        }
        let distinct: std::collections::HashSet<_> = m.vertices.iter().map(|v| v.color).collect();
        assert!(distinct.len() > 1 && distinct.len() <= 216);
    }

    fn picker(color: &str) -> Harness<'static, Map<String, Value>> {
        let mut f = Map::new();
        f.insert("__colorPicker".into(), json!("foreground"));
        f.insert("color".into(), json!(color));
        f.insert("__orig".into(), json!(color));
        let mut h = Harness::builder().with_size(vec2(700.0, 500.0)).build_ui_state(
            |ui, f: &mut Map<String, Value>| {
                if ui.ctx().fonts(|fonts| fonts.families().contains(&egui::FontFamily::Name("medium".into()))) {
                    body(ui, f);
                }
            },
            f,
        );
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
        h.run_steps(2);
        h
    }

    /// Type `text` into the focused field one key per frame, like a person typing.
    fn type_keys(h: &mut Harness<'static, Map<String, Value>>, text: &str) {
        for ch in text.chars() {
            h.event(egui::Event::Text(ch.to_string()));
            h.run_steps(1);
        }
    }

    fn color(h: &Harness<'static, Map<String, Value>>) -> String {
        h.state().get("color").and_then(Value::as_str).unwrap().to_string()
    }

    #[test]
    fn hex_field_shows_uppercase_without_rewriting_the_selected_color() {
        let h = picker("#12abef");
        assert_eq!(h.get_by_role(Role::TextInput).value().as_deref(), Some("12ABEF"));
        assert_eq!(color(&h), "#12abef", "casing is presentation only");
    }

    #[test]
    fn hex_field_takes_typed_digits() {
        let mut h = picker("#000000");
        h.get_by_role(Role::TextInput).click();
        h.run_steps(1);
        h.key_press_modifiers(egui::Modifiers::COMMAND, egui::Key::A);
        h.run_steps(1);
        type_keys(&mut h, "ff88");
        // Partial input stays in the field instead of snapping back to the colour.
        assert_eq!(h.get_by_role(Role::TextInput).value().as_deref(), Some("ff88"));
        assert_eq!(color(&h), "#000000");
        type_keys(&mut h, "00");
        assert_eq!(color(&h), "#ff8800");
        h.key_press(egui::Key::Tab);
        h.run_steps(2);
        assert_eq!(color(&h), "#ff8800");
    }

    #[test]
    fn color_picker_opens_with_hex_focused_and_selected() {
        let mut h = picker("#336699");
        let hex_field = h.get_by_role(Role::TextInput);
        assert!(hex_field.is_focused(), "the hex field automatically has focus on open");
        // Typing directly replaces the selected hex digits without manual clicking or Command+A.
        type_keys(&mut h, "ff8800");
        assert_eq!(color(&h), "#ff8800", "typing replaces the selected hex code");
    }

    #[test]
    fn color_picker_dialog_hex_field_is_in_tab_order_and_confirms_typed_color() {
        let app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let mut h = Harness::builder().with_size(vec2(760.0, 480.0)).build_ui_state(|ui, app| crate::dialogs::show(app, ui.ctx()), app);
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
        open(h.state_mut(), "foreground");
        h.run_steps(3);
        assert!(h.get_by_role(Role::TextInput).is_focused(), "hex field has focus in dialog frame");
        for ch in "ff8800".chars() {
            h.event(egui::Event::Text(ch.to_string()));
            h.run_steps(1);
        }
        h.key_press(egui::Key::Enter);
        h.run_steps(3);
        assert_eq!(hex([h.state().session.tools.foreground[0], h.state().session.tools.foreground[1], h.state().session.tools.foreground[2]]), "#ff8800");
    }

    #[test]
    fn alt_reset_restores_orig_color() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.tools.foreground = [0.2, 0.4, 0.6, 1.0];
        let id = open(&mut app, "foreground");
        let f = &mut app.ui.dialog_mut(id).unwrap().fields;
        f.insert("color".into(), json!("#ff0000"));
        reset(f);
        assert_eq!(f["color"], json!(hex([0.2, 0.4, 0.6])));
        assert_eq!(f["__orig"], json!(hex([0.2, 0.4, 0.6])));
    }

    #[test]
    fn lab_and_cmyk_fields_take_multi_digit_values() {
        let base = parse_hex("#336699").unwrap();
        let lab = photocraft_color::convert::srgb_to_lab(base);
        let cmyk = photocraft_color::convert::rgb_to_cmyk(base);
        // Fields in order: H S B R G B, L a b, C M Y K.
        for (field, typed, want) in [
            (6, "60", photocraft_color::convert::lab_to_srgb([60.0, lab[1], lab[2]])),
            (7, "30", photocraft_color::convert::lab_to_srgb([lab[0], 30.0, lab[2]])),
            (9, "40", photocraft_color::convert::cmyk_to_rgb([0.4, cmyk[1], cmyk[2], cmyk[3]])),
        ] {
            let mut h = picker("#336699");
            h.query_all_by_role(Role::SpinButton).nth(field).unwrap().click();
            h.run_steps(1);
            type_keys(&mut h, typed);
            h.key_press(egui::Key::Tab);
            h.run_steps(2);
            assert_eq!(color(&h), hex(want), "field {field} <- {typed}");
        }
    }

    #[test]
    fn picker_places_actions_above_two_columns_and_web_colors_under_the_field() {
        let h = picker("#336699");
        let inputs: Vec<_> = h.query_all_by_role(Role::SpinButton).map(|n| n.rect()).collect();
        assert_eq!(inputs.len(), 13);
        let (hue, red, lightness, cyan, black) = (inputs[0], inputs[3], inputs[6], inputs[9], inputs[12]);
        assert!(lightness.left() > hue.right(), "Lab is beside HSB, not below RGB");
        assert!((lightness.top() - hue.top()).abs() < 1.0);
        assert!((cyan.top() - red.top()).abs() < 1.0, "CMYK and RGB start at the same height");
        let ok = h.get_by_label("OK").rect();
        let cancel = h.get_by_label("Cancel").rect();
        assert!(ok.left() > hue.left() && ok.bottom() < hue.top(), "actions are above the values and beside the preview: {ok:?}, {hue:?}");
        assert!((ok.left() - cancel.left()).abs() < 1.0 && cancel.top() > ok.bottom());
        assert_eq!(h.query_all_by_label("OK").count(), 1);
        let web = h.get_by_label("Only Web Colors").rect();
        assert!(web.left() < hue.left() && web.top() > hue.bottom());
        // The full dialog should fit even in PhotoCraft's minimum 760 x 480 window.
        assert!(black.bottom() - ok.top() < 360.0);
        assert!(black.right() - web.left() < 600.0);
        let hex = h.get_by_role(Role::TextInput).rect();
        assert!(hex.left() < lightness.left() && hex.top() > inputs[5].bottom());
    }

    /// The whole Color Picker window (title, frame and all) is no bigger than Photoshop's: about
    /// 534 x 376 points (Photoshop 27.11 on macOS, measured from a screenshot at the same scale as
    /// PhotoCraft's, which was 682 x 438).
    #[test]
    fn the_picker_window_is_no_bigger_than_photoshops() {
        let mut h = Harness::builder().with_size(vec2(1400.0, 900.0)).with_max_steps(64).build_eframe(|cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            PhotocraftApp::new(photocraft_engine::Session::new(), Default::default())
        });
        h.run_steps(4);
        let id = open(h.state_mut(), "foreground");
        h.run_steps(6);
        let rect = h.ctx.memory(|m| m.area_rect(egui::Id::new(("dialog", id)))).expect("the picker window");
        assert!(rect.width() <= 534.0 && rect.height() <= 376.0, "Color Picker window {} x {} points, Photoshop's 534 x 376", rect.width(), rect.height());
        // Everything still shows: the three actions and all 13 component fields.
        for label in ["OK", "Cancel", "Add to Swatches", "Only Web Colors"] {
            assert!(rect.contains_rect(h.get_by_label(label).rect()), "{label} inside the window");
        }
        assert_eq!(h.query_all_by_role(Role::SpinButton).filter(|n| rect.contains_rect(n.rect())).count(), 13);
        // The actions are as wide as the longest label ("Add to Swatches" is about 133 points in
        // the button font), not stretched across the column (160). Photoshop's are about 116.
        let widths: Vec<f32> = ["OK", "Cancel", "Add to Swatches"].iter().map(|l| h.get_by_label(l).rect().width()).collect();
        assert!(widths.iter().all(|w| (w - widths[0]).abs() < 0.5), "one width for all three actions: {widths:?}");
        assert!(widths[0] <= 136.0, "actions {} points wide, Photoshop's about 116", widths[0]);
    }

    #[test]
    fn web_colors_and_current_swatch_still_update_the_picker() {
        let mut h = picker("#123456");
        h.get_by_label("Only Web Colors").click();
        h.run_steps(2);
        assert_eq!(color(&h), "#003366");
        h.get_by_label("Only Web Colors").click();
        h.run_steps(2);
        h.get_by_label("Click to restore the current colour").click();
        h.run_steps(2);
        assert_eq!(color(&h), "#123456");
    }

    #[test]
    fn side_buttons_and_keyboard_confirm_or_cancel_without_changing_the_other_color() {
        for target in ["foreground", "background"] {
            for action in ["OK", "Cancel", "Enter", "Escape"] {
                let app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
                let mut h = Harness::builder().with_size(vec2(760.0, 480.0)).build_ui_state(|ui, app| crate::dialogs::show(app, ui.ctx()), app);
                PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
                let original_fg = h.state().session.tools.foreground;
                let original_bg = h.state().session.tools.background;
                let id = open(h.state_mut(), target);
                h.state_mut().ui.dialog_mut(id).unwrap().fields.insert("color".into(), json!("#336699"));
                h.run_steps(4);
                let screen = Rect::from_min_size(pos2(0.0, 0.0), vec2(760.0, 480.0));
                for node in h.query_all_by_role(Role::SpinButton) {
                    assert!(screen.contains_rect(node.rect()), "numeric inputs fit the minimum window");
                }
                for label in ["OK", "Cancel", "Only Web Colors"] {
                    assert!(screen.contains_rect(h.get_by_label(label).rect()), "{label} fits the minimum window");
                }
                match action {
                    "Enter" => h.key_press(egui::Key::Enter),
                    "Escape" => h.key_press(egui::Key::Escape),
                    _ => h.get_by_label(action).click(),
                }
                h.run_steps(3);
                assert!(h.state().ui.dialogs.is_empty(), "{target}: {action} closes the picker");
                let confirm = matches!(action, "OK" | "Enter");
                let tools = &h.state().session.tools;
                if confirm && target == "foreground" {
                    assert_eq!(hex([tools.foreground[0], tools.foreground[1], tools.foreground[2]]), "#336699");
                } else {
                    assert_eq!(tools.foreground, original_fg);
                }
                if confirm && target == "background" {
                    assert_eq!(hex([tools.background[0], tools.background[1], tools.background[2]]), "#336699");
                } else {
                    assert_eq!(tools.background, original_bg);
                }
            }
        }
    }

    #[test]
    fn parse_hex_rejects_multibyte_text() {
        // Six bytes, three characters: the byte range 0..2 splits "é", which used to panic.
        assert_eq!(parse_hex("aé€"), None);
        assert_eq!(parse_hex("#ff880"), None);
    }

    #[test]
    fn hsv_round_trips() {
        for c in [[1.0, 0.0, 0.0], [0.2, 0.6, 0.4], [0.5, 0.5, 0.5], [0.0, 0.0, 1.0], [0.9, 0.8, 0.1]] {
            let h = rgb_to_hsv(c);
            let back = hsv_to_rgb(h[0], h[1], h[2]);
            for i in 0..3 {
                assert!((back[i] - c[i]).abs() < 1e-5, "{c:?} -> {h:?} -> {back:?}");
            }
        }
        assert_eq!(hex([1.0, 0.5, 0.0]), "#ff8000");
        assert_eq!(parse_hex("#FF8000").map(hex).as_deref(), Some("#ff8000"));
    }

    #[test]
    fn field_and_locate_are_inverse_in_every_mode() {
        let rgb = [0.8, 0.3, 0.2];
        let (hsv, lab) = (rgb_to_hsv(rgb), photocraft_color::convert::srgb_to_lab(rgb));
        for (m, _) in MODES.iter().chain(LAB_MODES) {
            let (x, y, z) = locate(m, hsv, rgb, lab);
            let c = field_color(m, z, x, y);
            assert_eq!(hex(c), hex(rgb), "mode {m}");
        }
    }

    #[test]
    fn lab_fields_are_laid_out_like_photoshops() {
        // Photoshop 25.4. L: a across (green to red), b upwards (blue to yellow).
        let l = |x, y| photocraft_color::convert::srgb_to_lab(field_color("lab_l", 0.6, x, y));
        assert!(l(0.0, 0.5)[1] < -40.0 && l(1.0, 0.5)[1] > 40.0, "{:?} {:?}", l(0.0, 0.5), l(1.0, 0.5));
        assert!(l(0.5, 1.0)[2] < -40.0 && l(0.5, 0.0)[2] > 40.0, "{:?} {:?}", l(0.5, 1.0), l(0.5, 0.0));
        // a: b across, L upwards (white at the top, black at the bottom).
        assert!(field_color("lab_a", 0.5, 0.5, 0.0).iter().all(|&c| c > 0.99), "white at the top");
        assert!(field_color("lab_a", 0.5, 0.5, 1.0).iter().all(|&c| c < 0.01), "black at the bottom");
        let a = |x| photocraft_color::convert::srgb_to_lab(field_color("lab_a", 0.5, x, 0.5));
        assert!(a(0.0)[2] < -40.0 && a(1.0)[2] > 40.0);
        // b: a across, L upwards; its slider runs from yellow (top) to blue.
        let b = |x| photocraft_color::convert::srgb_to_lab(field_color("lab_b", 0.5, x, 0.5));
        assert!(b(0.0)[1] < -40.0 && b(1.0)[1] > 40.0);
        let lab = [60.0, 0.0, 0.0];
        let top = photocraft_color::convert::srgb_to_lab(slider_color("lab_b", 1.0, [0.0; 3], [0.0; 3], lab));
        let bottom = photocraft_color::convert::srgb_to_lab(slider_color("lab_b", 0.0, [0.0; 3], [0.0; 3], lab));
        assert!(top[2] > 40.0 && bottom[2] < -40.0, "{top:?} {bottom:?}");
    }

    #[test]
    fn gamut_and_web_warnings() {
        // Photoshop 25.4: #714343 is printable but not a web colour; #fd0303 is neither.
        let rgb = |h| parse_hex(h).unwrap();
        assert_eq!(out_of_gamut(rgb("#714343")), None);
        assert!(out_of_gamut(rgb("#fd0303")).is_some());
        assert_eq!(out_of_gamut(rgb("#000000")), None, "black prints");
        assert_eq!(out_of_gamut(rgb("#ffffff")), None);
        assert_eq!(not_web_safe(rgb("#714343")).map(hex).as_deref(), Some("#663333"));
        assert_eq!(not_web_safe(rgb("#cc3366")), None);
    }

    #[test]
    fn hue_survives_grey_and_ok_sets_the_colour() {
        let mut f = Map::new();
        f.insert("color".into(), json!("#808080"));
        f.insert("__hsv".into(), json!([200.0, 0.0, 128.0 / 255.0]));
        assert_eq!(current(&f).hsv[0], 200.0);
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        let id = open(&mut app, "background");
        let d = app.ui.dialog_mut(id).unwrap();
        d.fields.insert("color".into(), json!("#3366cc"));
        let fields = d.fields.clone();
        confirm(&mut app, &fields).unwrap();
        assert_eq!(hex([app.session.tools.background[0], app.session.tools.background[1], app.session.tools.background[2]]), "#3366cc");
    }

    /// #4196: CMYK values typed into the picker survive OK, so the colour fills a CMYK document with those inks and
    /// reopening the picker shows them again.
    #[test]
    fn cmyk_values_survive_ok_and_a_reopen() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        let cyan = [1.0, 0.0, 0.0, 0.0];
        let id = open(&mut app, "foreground");
        let d = app.ui.dialog_mut(id).unwrap();
        set_rgb(&mut d.fields, photocraft_color::convert::cmyk_to_rgb(cyan).map(|c| c.clamp(0.0, 1.0)), Keep::Cmyk(cyan));
        let fields = d.fields.clone();
        confirm(&mut app, &fields).unwrap();
        let fg = app.session.tools.foreground;
        assert_eq!(app.session.tools.picked_cmyk([fg[0], fg[1], fg[2]]), Some(cyan));
        let id = open(&mut app, "foreground");
        let again = current(&app.ui.dialogs.iter().find(|d| d.id == id).unwrap().fields);
        assert_eq!(again.cmyk, cyan);
    }

    fn dialog_color(app: &PhotocraftApp, id: u64) -> &str {
        app.ui.dialogs.iter().find(|d| d.id == id).and_then(|d| d.fields.get("color")).and_then(Value::as_str).unwrap()
    }

    #[test]
    fn the_top_picker_samples_the_image() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 40, "height": 20, "background": "transparent"})).unwrap();
        app.run("shape.create", json!({"kind": "rect", "rect": [0, 0, 20, 20], "fill": "#ff0000"})).unwrap();
        app.run("shape.create", json!({"kind": "rect", "rect": [20, 0, 10, 20], "fill": "#00ff00"})).unwrap();
        let id = open(&mut app, "foreground");
        assert_eq!(top(&app), Some(id));
        sample_at(&mut app, 25.5, 5.5);
        assert_eq!(dialog_color(&app, id), "#00ff00");
        assert_eq!(app.ui.dialog_mut(id).unwrap().fields["__orig"], json!("#000000"), "the current colour stays");
        // Off the image or over transparency nothing changes.
        for (x, y) in [(35.0, 5.0), (-3.0, 5.0), (5.0, 20.0), (1e12, -1e12), (f64::NAN, f64::INFINITY)] {
            sample_at(&mut app, x, y);
            assert_eq!(dialog_color(&app, id), "#00ff00", "({x}, {y})");
        }

        // Agents: `ui.pointer` samples into the picker instead of driving the tool.
        app.ui.tool = crate::state::Tool::Brush;
        let rev = app.session.active().unwrap().revision;
        let ctx = egui::Context::default();
        let events = json!([{"kind": "down", "x": 5, "y": 5}, {"kind": "up", "x": 5, "y": 5}]);
        let (req, _rx) = crate::control::ControlRequest::new("ui.pointer", json!({"events": events}));
        let _ = crate::control::handle(&mut app, &ctx, &req);
        assert_eq!(dialog_color(&app, id), "#ff0000");
        assert_eq!(app.session.active().unwrap().revision, rev, "the Brush must not paint");
        assert_eq!(app.session.tools.foreground, [0.0, 0.0, 0.0, 1.0], "only OK sets the foreground");

        // A dialog opened over the picker takes the input.
        let about = app.ui.open_dialog(DialogKind::About, Map::new());
        assert_eq!(top(&app), None);
        sample_at(&mut app, 25.0, 5.0);
        assert_eq!(dialog_color(&app, id), "#ff0000");
        app.ui.close_dialog(about);
        sample_at(&mut app, 25.0, 5.0);
        assert_eq!(dialog_color(&app, id), "#00ff00");
    }
}

#[cfg(test)]
mod screen_integration_tests {
    use super::*;
    use crate::screen_picker::{Capture, Pending};
    use egui_kittest::{Harness, kittest::Queryable};
    #[test]
    fn text_swatch_opens_full_dialog_and_screen_pick_waits_for_ok() {
        let mut session = photocraft_engine::Session::new();
        session.execute("file.new", json!({"width":160,"height":120})).unwrap();
        let id = session.execute("type.create", json!({"text":"Hello world","color":"#000000","size":24,"x":10,"y":60})).unwrap()["layer"].as_u64().unwrap();
        let services = crate::Services {
            screen_pick: Some(Box::new(|_| {
                let (tx, receiver) = std::sync::mpsc::channel();
                tx.send(Ok(Capture::Color(Some([1.0, 136.0 / 255.0, 0.0])))).unwrap();
                Pending { receiver, cancelled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)) }
            })),
            ..Default::default()
        };
        let mut h = Harness::builder().with_size(vec2(1440.0, 900.0)).with_max_steps(64).build_eframe(move |cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            let mut app = PhotocraftApp::new(session, services);
            app.ui.tool = crate::state::Tool::Type;
            app
        });
        h.run_steps(8);
        let before = h.state().session.active().unwrap().history.entries().len();
        let fg = h.state().session.tools.foreground;
        h.get_by_label("Set the text color").click();
        h.run_steps(6);
        assert_eq!(h.state().ui.dialogs.last().unwrap().fields["__label"], "Color Picker (Text Color)");
        h.get_by_label("Pick screen color").click();
        h.run_steps(6);
        assert_eq!(h.state().ui.dialogs.last().unwrap().fields["color"], "#ff8800");
        assert_eq!(h.state().session.active().unwrap().history.entries().len(), before, "sampling is only a dialog field change");
        assert_eq!(h.state().session.tools.foreground, fg);
        h.get_by_label("OK").click();
        h.run_steps(4);
        let doc = &h.state().session.active().unwrap().doc;
        let layer = doc.layer(photocraft_doc::LayerId(id)).unwrap();
        let photocraft_doc::LayerContent::Text(text) = &layer.content else { panic!("not text") };
        assert_eq!(text.color.to_rgba8(), [255, 136, 0, 255]);
        assert_eq!(h.state().session.active().unwrap().history.entries().len(), before + 1);
    }
}
