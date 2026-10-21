//! The Crop tool's shield options (Photoshop's options-bar gear menu, "Set additional Crop
//! options", #1919): whether the area outside the crop box shows at all (Show Cropped Area, H),
//! and the shield dimming it (Enable Crop Shield: Match Canvas or a custom colour, an opacity, and
//! Auto Adjust Opacity, which lightens it while the box is dragged). Photoshop's defaults: all on,
//! Match Canvas, 75 %. The menu also holds the crop mode: Use Classic Mode (P; off by default) and
//! Auto Center Preview (on by default, default mode only), see `crop_mode`.
//!
//! The state lives in `ToolOptions::crop_shield` ([`CropShield`]), which `ui.inspect` reports and
//! `ui.set {cropShield: {...}}` patches. [`fill`] resolves it to the colour the canvas paints over
//! the area outside the box.

use egui::{Color32, Key, Modifiers};
use serde::{Deserialize, Serialize};

use crate::PhotocraftApp;
use crate::state::Tool;

/// Photoshop's default shield opacity, percent.
pub const DEFAULT_OPACITY: f32 = 75.0;
/// Auto Adjust Opacity: the share of the opacity the shield keeps while the box is dragged.
pub const EDITING_FACTOR: f32 = 0.5;

/// The shield's colour.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ShieldColor {
    /// The pasteboard's colour (Preferences › Interface, or the theme's canvas).
    #[default]
    MatchCanvas,
    /// [`CropShield::custom_color`].
    Custom,
}

/// The gear menu's options. Every field defaults on its own, so older settings load.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CropShield {
    /// Use Classic Mode (P): the box turns and moves over the image instead of the image turning
    /// and moving behind an upright box (`crop_mode`). Off by default, as in Photoshop.
    pub classic_mode: bool,
    /// Auto Center Preview: in default mode, drawing or resizing the box re-pans the view so the
    /// box sits in the middle of the canvas.
    pub auto_center_preview: bool,
    /// Show Cropped Area (H): off hides everything outside the box behind the canvas colour.
    pub show_cropped_area: bool,
    /// Enable Crop Shield: off shows the area outside the box undimmed.
    pub enabled: bool,
    pub color: ShieldColor,
    /// sRGB, used when `color` is `Custom`.
    pub custom_color: [u8; 3],
    /// Percent, 0..=100 ([`CropShield::opacity`] clamps a stray value).
    pub opacity: f32,
    /// Auto Adjust Opacity: lighter while the box is dragged.
    pub auto_adjust: bool,
}

impl Default for CropShield {
    fn default() -> Self {
        Self {
            classic_mode: false,
            auto_center_preview: true,
            show_cropped_area: true,
            enabled: true,
            color: ShieldColor::MatchCanvas,
            custom_color: [0, 0, 0],
            opacity: DEFAULT_OPACITY,
            auto_adjust: true,
        }
    }
}

impl CropShield {
    /// Show Cropped Area as it acts: always on in Classic Mode, where Photoshop greys the option
    /// out (the stored choice is kept for the default mode).
    pub fn shows_cropped_area(&self) -> bool {
        self.classic_mode || self.show_cropped_area
    }

    /// The opacity as a fraction in 0..=1: a non-finite value (a hand-edited settings file) is
    /// the default, an out-of-range one is clamped.
    pub fn opacity(&self) -> f32 {
        let o = if self.opacity.is_finite() { self.opacity } else { DEFAULT_OPACITY };
        o.clamp(0.0, 100.0) / 100.0
    }
}

/// What to paint over the area outside the crop box, `canvas` being the pasteboard colour and
/// `editing` a crop drag in progress: the opaque canvas colour when Show Cropped Area is off, no
/// shield when it is disabled (or fully transparent), otherwise its colour at its opacity.
pub fn fill(s: &CropShield, canvas: Color32, editing: bool) -> Option<Color32> {
    let [r, g, b, _] = canvas.to_srgba_unmultiplied();
    if !s.shows_cropped_area() {
        return Some(Color32::from_rgb(r, g, b));
    }
    if !s.enabled {
        return None;
    }
    let [r, g, b] = match s.color {
        ShieldColor::MatchCanvas => [r, g, b],
        ShieldColor::Custom => s.custom_color,
    };
    let a = s.opacity() * if s.auto_adjust && editing { EDITING_FACTOR } else { 1.0 };
    let a = (a * 255.0).round().clamp(0.0, 255.0) as u8;
    (a > 0).then(|| Color32::from_rgba_unmultiplied(r, g, b, a))
}

/// The pasteboard colour the shield matches: Preferences › Interface's, else the theme's canvas.
pub fn canvas_color(app: &PhotocraftApp, ctx: &egui::Context) -> Color32 {
    crate::prefs_ui::pasteboard_color(app).unwrap_or_else(|| crate::theme::Tokens::get(ctx).canvas)
}

/// The shield over the pending crop box now ([`fill`]).
pub fn current(app: &PhotocraftApp, ctx: &egui::Context) -> Option<Color32> {
    fill(&app.ui.tool_options.crop_shield, canvas_color(app, ctx), app.crop.drag.is_some())
}

/// Show Cropped Area is off with the Crop tool showing a box: nothing outside it shows.
pub fn hides_outside(app: &PhotocraftApp) -> bool {
    app.ui.tool == Tool::Crop && app.ui.crop_rect.is_some() && !app.ui.tool_options.crop_shield.shows_cropped_area()
}

/// H with the Crop tool showing a box and no drag in progress toggles Show Cropped Area (in
/// Classic Mode, where the option is greyed out, H is taken and does nothing), and P Use Classic
/// Mode. Returns true when a key was used; otherwise H stays the Hand tool key and P the Pen's.
pub fn keys(app: &mut PhotocraftApp, ctx: &egui::Context) -> bool {
    if app.ui.tool != Tool::Crop || app.ui.crop_rect.is_none() || app.crop.drag.is_some() {
        return false;
    }
    if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::H)) {
        let s = &mut app.ui.tool_options.crop_shield;
        if !s.classic_mode {
            s.show_cropped_area = !s.show_cropped_area;
        }
        return true;
    }
    if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::P)) {
        let s = &mut app.ui.tool_options.crop_shield;
        s.classic_mode = !s.classic_mode;
        // The view follows at once, about its own centre: default mode turns it with the frame,
        // Classic Mode puts the camera upright (never back to a Rotate View turn).
        crate::crop_mode::sync(app, None);
        return true;
    }
    false
}

/// The options bar's gear button and its menu. Returns true when Custom… asks for the Color
/// Picker ([`pick_custom_color`], which needs the app).
pub fn options_button(s: &mut CropShield, ui: &mut egui::Ui) -> bool {
    let mut pick = false;
    let r = crate::icons::button(ui, "settings", 24.0, false, tl!("Set additional Crop options"));
    egui::Popup::menu(&r).close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside).show(|ui| {
        ui.set_min_width(240.0);
        ui.set_max_width(300.0);
        // Photoshop 25.1's order; Classic Mode greys out the other two (Show Cropped Area then
        // shows as on, which is how it acts there).
        ui.horizontal(|ui| {
            crate::widgets::checkbox(ui, &mut s.classic_mode, tl!("Use Classic Mode"));
            ui.weak(crate::shortcuts::pretty("P"));
        });
        let classic = s.classic_mode;
        ui.add_enabled_ui(!classic, |ui| {
            ui.horizontal(|ui| {
                if classic {
                    crate::widgets::checkbox(ui, &mut true, tl!("Show Cropped Area"));
                } else {
                    crate::widgets::checkbox(ui, &mut s.show_cropped_area, tl!("Show Cropped Area"));
                }
                ui.weak(crate::shortcuts::pretty("H"));
            });
            crate::widgets::checkbox(ui, &mut s.auto_center_preview, tl!("Auto Center Preview"));
        });
        ui.separator();
        crate::widgets::checkbox(ui, &mut s.enabled, tl!("Enable Crop Shield"));
        ui.add_enabled_ui(s.enabled, |ui| {
            ui.indent("crop-shield", |ui| {
                ui.horizontal(|ui| {
                    ui.label(tl!("Color:"));
                    ui.radio_value(&mut s.color, ShieldColor::MatchCanvas, tl!("Match Canvas"));
                    if ui.radio(s.color == ShieldColor::Custom, tl!("Custom…")).clicked() {
                        s.color = ShieldColor::Custom;
                        pick = true;
                    }
                    if s.color == ShieldColor::Custom {
                        let [r, g, b] = s.custom_color;
                        if crate::widgets::color_swatch_button(ui, Color32::from_rgb(r, g, b), tl!("Crop Shield color")).clicked() {
                            pick = true;
                        }
                    }
                });
                if !s.opacity.is_finite() {
                    s.opacity = DEFAULT_OPACITY;
                }
                crate::widgets::slider_row(ui, tl!("Opacity:"), &mut s.opacity, 0.0..=100.0, "%", None);
                crate::widgets::checkbox(ui, &mut s.auto_adjust, tl!("Auto Adjust Opacity"));
            });
        });
    });
    pick
}

/// Custom…: PhotoCraft's Color Picker on the custom shield colour; OK sets it
/// (`color_picker_ui::confirm`, target `cropShield`).
pub fn pick_custom_color(app: &mut PhotocraftApp) -> u64 {
    let rgb = app.ui.tool_options.crop_shield.custom_color.map(|v| v as f32 / 255.0);
    crate::color_picker_ui::open_for_target(app, "cropShield", "Color Picker (Crop Shield Color)", rgb)
}

/// The Color Picker's OK for the custom shield colour (`"#rrggbb"`).
pub fn set_custom_color(app: &mut PhotocraftApp, hex: &str) -> Result<(), String> {
    let rgb = crate::color_picker_ui::parse_hex(hex).ok_or_else(|| format!("not a colour: {hex}"))?;
    let s = &mut app.ui.tool_options.crop_shield;
    s.custom_color = rgb.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8);
    s.color = ShieldColor::Custom;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const CANVAS: Color32 = Color32::from_rgb(40, 40, 40);

    #[test]
    fn defaults_match_photoshop() {
        let s = CropShield::default();
        assert!(s.show_cropped_area && s.enabled && s.auto_adjust);
        assert_eq!((s.color, s.opacity), (ShieldColor::MatchCanvas, 75.0));
        assert_eq!(fill(&s, CANVAS, false), Some(Color32::from_rgba_unmultiplied(40, 40, 40, 191)));
    }

    #[test]
    fn auto_adjust_lightens_the_shield_while_dragging() {
        let mut s = CropShield::default();
        let still = fill(&s, CANVAS, false).unwrap();
        let dragging = fill(&s, CANVAS, true).unwrap();
        assert!(dragging.a() < still.a(), "{dragging:?} {still:?}");
        assert_eq!(dragging.to_srgba_unmultiplied()[3], 96);
        s.auto_adjust = false;
        assert_eq!(fill(&s, CANVAS, true), Some(still), "without Auto Adjust the opacity holds");
    }

    #[test]
    fn a_disabled_shield_leaves_the_outside_undimmed() {
        let s = CropShield { enabled: false, ..Default::default() };
        assert_eq!(fill(&s, CANVAS, false), None);
        assert_eq!(fill(&s, CANVAS, true), None);
        let s = CropShield { opacity: 0.0, ..Default::default() };
        assert_eq!(fill(&s, CANVAS, false), None, "0 % draws nothing");
    }

    #[test]
    fn custom_colour_and_match_canvas() {
        let s = CropShield { color: ShieldColor::Custom, custom_color: [200, 10, 30], opacity: 100.0, ..Default::default() };
        assert_eq!(fill(&s, CANVAS, false), Some(Color32::from_rgb(200, 10, 30)));
        let s = CropShield { opacity: 100.0, ..Default::default() };
        assert_eq!(fill(&s, Color32::from_rgb(184, 184, 184), false), Some(Color32::from_rgb(184, 184, 184)));
    }

    #[test]
    fn hidden_cropped_area_is_opaque_canvas_whatever_the_shield() {
        for enabled in [true, false] {
            let s =
                CropShield { show_cropped_area: false, enabled, color: ShieldColor::Custom, custom_color: [255, 0, 0], opacity: 10.0, ..Default::default() };
            assert_eq!(fill(&s, CANVAS, false), Some(CANVAS));
            assert_eq!(fill(&s, CANVAS, true), Some(CANVAS));
        }
    }

    #[test]
    fn stray_opacities_are_clamped() {
        for (o, want) in [(f32::NAN, 0.75), (f32::INFINITY, 0.75), (f32::NEG_INFINITY, 0.75), (-20.0, 0.0), (250.0, 1.0), (40.0, 0.4)] {
            let s = CropShield { opacity: o, ..Default::default() };
            assert!((s.opacity() - want).abs() < 1e-6, "{o}");
            assert_eq!(fill(&s, CANVAS, false).map_or(0, |c| c.to_srgba_unmultiplied()[3]), (want * 255.0f32).round() as u8, "{o}");
        }
        // From JSON: out-of-range numbers load and are clamped when drawn.
        let s: CropShield = serde_json::from_str(r#"{"opacity": 1e9}"#).unwrap();
        assert_eq!(s.opacity(), 1.0);
        assert!(serde_json::from_str::<CropShield>(r#"{"opacity": "high"}"#).is_err());
    }

    #[test]
    fn older_settings_without_the_shield_get_the_defaults() {
        let o: crate::state::ToolOptions = serde_json::from_str(r#"{"crop_delete": false}"#).unwrap();
        assert_eq!(o.crop_shield, CropShield::default());
        assert!(!o.crop_delete);
        let s: CropShield = serde_json::from_str(r#"{"enabled": false}"#).unwrap();
        assert_eq!(s, CropShield { enabled: false, ..Default::default() });
        let back: CropShield = serde_json::from_value(serde_json::to_value(&s).unwrap()).unwrap();
        assert_eq!(back, s);
        assert_eq!(serde_json::to_value(ShieldColor::MatchCanvas).unwrap(), "matchCanvas");
    }

    fn press_h(app: &mut PhotocraftApp) {
        let ctx = egui::Context::default();
        ctx.input_mut(|i| i.events.push(egui::Event::Key { key: Key::H, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE }));
        crate::shortcuts::handle(app, &ctx);
    }

    fn crop_app() -> PhotocraftApp {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", serde_json::json!({"width": 200, "height": 120})).unwrap();
        app.ui.tool = Tool::Crop;
        crate::crop_ui::ensure_frame(&mut app);
        assert!(app.ui.crop_rect.is_some());
        app
    }

    #[test]
    fn h_toggles_show_cropped_area_while_the_crop_tool_shows_a_box() {
        let mut app = crop_app();
        assert!(!hides_outside(&app));
        press_h(&mut app);
        assert_eq!(app.ui.tool, Tool::Crop, "H doesn't switch to the Hand tool");
        assert!(!app.ui.tool_options.crop_shield.show_cropped_area);
        assert!(hides_outside(&app));
        press_h(&mut app);
        assert!(app.ui.tool_options.crop_shield.show_cropped_area);
    }

    #[test]
    fn h_stays_the_hand_key_without_a_crop_box() {
        // No document: no box, so H picks the Hand tool.
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.ui.tool = Tool::Crop;
        press_h(&mut app);
        assert_eq!(app.ui.tool, Tool::Hand);
        assert!(app.ui.tool_options.crop_shield.show_cropped_area);
        // Another tool.
        let mut app = crop_app();
        app.ui.tool = Tool::Brush;
        press_h(&mut app);
        assert_eq!(app.ui.tool, Tool::Hand);
        assert!(app.ui.tool_options.crop_shield.show_cropped_area);
        // A crop drag in progress.
        let mut app = crop_app();
        app.crop.drag = Some(crate::crop_ui::CropDrag::Move { start: [0.0, 0.0], rect: [0.0, 0.0, 10.0, 10.0] });
        let ctx = egui::Context::default();
        ctx.input_mut(|i| i.events.push(egui::Event::Key { key: Key::H, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE }));
        assert!(!keys(&mut app, &ctx));
        assert!(app.ui.tool_options.crop_shield.show_cropped_area);
    }

    #[test]
    fn the_color_picker_sets_the_custom_colour() {
        let mut app = crop_app();
        let id = pick_custom_color(&mut app);
        let f = app.ui.dialog_mut(id).map(|d| d.fields.clone()).unwrap();
        let mut f = f;
        f.insert("color".into(), serde_json::json!("#ff8000"));
        crate::color_picker_ui::confirm(&mut app, &f).unwrap();
        let s = &app.ui.tool_options.crop_shield;
        assert_eq!((s.color, s.custom_color), (ShieldColor::Custom, [255, 128, 0]));
        assert!(set_custom_color(&mut app, "nope").is_err());
        assert_eq!(app.ui.tool_options.crop_shield.custom_color, [255, 128, 0]);
    }
}
