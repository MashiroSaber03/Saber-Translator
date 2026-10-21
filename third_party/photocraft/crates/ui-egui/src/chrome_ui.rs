//! Photoshop 2026 window chrome details: the status bar's info field and its "Show" menu, and the
//! Home button at the start of the options bar.

use egui::{RichText, Sense, Stroke, vec2};
use photocraft_doc::{Document, LayerContent};
use serde::{Deserialize, Serialize};

use crate::theme::Tokens;
use crate::{PhotocraftApp, icons, widgets};

/// Shell-only chrome state (serialised with the UI state, so `ui.inspect` reports it).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ChromeState {
    /// What the status bar shows next to the zoom field (one of [`STATUS_INFO`] keys).
    pub status_info: String,
    /// Home screen shown over the open documents (Options bar › Home); holds the document count
    /// when it was opened, so opening or creating a document leaves it.
    pub home: Option<usize>,
}

impl ChromeState {
    /// Is the Home screen up, given the current number of open documents? With no documents it
    /// shows by itself when `auto_show` (Preferences › General › Auto show the Home Screen).
    ///
    /// `home` stores the document count when Home was opened so File › New / Open can leave it.
    /// Closing a tab that returns to that count must not bring Home back (#3981): once the
    /// document set has changed, the overlay is gone until the user opens Home again.
    pub fn shows_home(&mut self, documents: usize, auto_show: bool) -> bool {
        if self.home.is_some_and(|n| n != documents) {
            self.home = None;
        }
        (documents == 0 && auto_show) || self.home == Some(documents)
    }
}

impl Default for ChromeState {
    fn default() -> Self {
        Self { status_info: "dimensions".into(), home: None }
    }
}

/// The status bar "Show" menu, in Photoshop's order: (key, menu label).
pub const STATUS_INFO: &[(&str, &str)] = &[
    ("sizes", "Document Sizes"),
    ("profile", "Document Profile"),
    ("dimensions", "Document Dimensions"),
    ("measurementScale", "Measurement Scale"),
    ("scratch", "Scratch Sizes"),
    ("efficiency", "Efficiency"),
    ("tool", "Current Tool"),
    ("layers", "Layer Count"),
];

/// Photoshop's compact byte format: "10.3M", "412.5K".
pub fn fmt_bytes(b: u64) -> String {
    let b = b as f64;
    if b >= 1024.0 * 1024.0 * 1024.0 {
        format!("{:.1}G", b / (1024.0 * 1024.0 * 1024.0))
    } else if b >= 1024.0 * 1024.0 {
        format!("{:.1}M", b / (1024.0 * 1024.0))
    } else {
        format!("{:.1}K", b / 1024.0)
    }
}

/// Flattened size and layered size in bytes ("Doc: flat/layered").
pub fn document_sizes(doc: &Document) -> (u64, u64) {
    let bpc = (doc.depth.bits() / 8).max(1) as u64;
    let channels = doc.mode.color_channels().max(1) as u64;
    let flat = doc.size.width as u64 * doc.size.height as u64 * channels * bpc;
    let mut layered = 0u64;
    for (_, _, l) in doc.walk() {
        if let LayerContent::Raster(s) = &l.content {
            // Layers carry transparency on top of the colour channels; clip to the canvas.
            let r = s.tile_bounds().intersect(&doc.bounds());
            layered += r.width() as u64 * r.height() as u64 * (channels + 1) * bpc;
        }
    }
    (flat, layered.max(flat))
}

/// The status bar text for `key` (Photoshop wording).
pub fn status_info_text(doc: &Document, key: &str, tool: &str, profile: &str) -> String {
    status_info_text_with_units(doc, key, tool, profile, &Default::default())
}

fn status_info_text_with_units(doc: &Document, key: &str, tool: &str, profile: &str, units: &photocraft_engine::prefs::UnitsAndRulers) -> String {
    let bits = doc.depth.bits();
    match key {
        "sizes" => {
            let (flat, layered) = document_sizes(doc);
            crate::i18n::fmt(tl!("Doc: {flat}/{layered}"), &[("flat", &fmt_bytes(flat)), ("layered", &fmt_bytes(layered))])
        }
        "profile" => format!("{profile} ({bits}bpc)"),
        "measurementScale" => "1 pixel = 1.0000 pixels".into(),
        "scratch" => {
            let (_, layered) = document_sizes(doc);
            crate::i18n::fmt(tl!("Scratch: {size}"), &[("size", &fmt_bytes(layered))])
        }
        "efficiency" => tl!("Efficiency: 100%").into(),
        "tool" => tool.to_string(),
        "layers" => {
            let n = doc.layer_count();
            crate::i18n::trn(crate::i18n::current(), n as u64, "{n} Layer", "{n} Layers")
        }
        _ if units.rulers == photocraft_engine::prefs::Unit::Pixels => crate::i18n::fmt(
            tl!("{w} px x {h} px ({ppi} ppi)"),
            &[("w", &doc.size.width.to_string()), ("h", &doc.size.height.to_string()), ("ppi", &widgets::fmt_num(doc.resolution_dpi as f64))],
        ),
        _ => {
            let dpi = doc.resolution_dpi as f64;
            let (w, h) = (doc.size.width as f64, doc.size.height as f64);
            crate::i18n::fmt(
                tl!("{w} {unit} x {h} {unit} ({ppi} ppi)"),
                &[("w", &units.format(w, dpi, w)), ("h", &units.format(h, dpi, h)), ("unit", units.rulers.suffix()), ("ppi", &widgets::fmt_num(dpi))],
            )
        }
    }
}

fn profile_name(doc: &Document) -> String {
    let mode = crate::canvas::mode_label(doc);
    let Some(bytes) = doc.icc_profile.as_ref() else {
        return crate::i18n::fmt(tl!("Untagged {mode}"), &[("mode", tl!(mode))]);
    };
    let Ok(profile) = photocraft_engine::color_cmds::profile_from_bytes(bytes) else {
        return crate::i18n::fmt(tl!("Invalid {mode} profile"), &[("mode", tl!(mode))]);
    };
    if profile.color_space != photocraft_engine::color_cmds::mode_space(doc.mode) {
        return crate::i18n::fmt(tl!("Invalid {mode} profile"), &[("mode", tl!(mode))]);
    }
    let name: String = profile.description.chars().filter(|c| !c.is_control()).take(128).collect();
    let name = name.trim();
    if name.is_empty() {
        return crate::i18n::fmt(tl!("Unnamed {mode} profile"), &[("mode", tl!(mode))]);
    }
    name.to_owned()
}

/// Status bar body (Pro): zoom %, the chosen info field and its ">" menu, then status messages.
pub fn status_bar_pro(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    let (Some(st), Some(i)) = (app.session.active(), app.session.active_index()) else {
        ui.label(RichText::new(tl!("No document")).color(t.text_dim));
        // A command run with no document open still reports here (#1567: File › Automate › Batch).
        status_message(app, ui, &t);
        return;
    };
    let text = status_info_text_with_units(
        &st.doc,
        &app.ui.chrome.status_info,
        tl!(app.ui.tool.label()),
        &profile_name(&st.doc),
        &app.session.prefs().units_and_rulers,
    );
    let mut pct = app.ui.views[i].zoom * 100.0;
    let control_h = crate::panels::status_bar_heights(&t).1;
    if widgets::compact_wrapping_value_field(ui, &mut pct, crate::zoom_levels::percent_range(&app.ui.views[i]), "%", vec2(64.0, control_h)).changed() {
        app.ui.views[i].zoom = crate::zoom_levels::clamp(pct / 100.0, app.ui.views[i].doc_size);
        app.ui.views[i].fit_pending = false;
    }
    ui.add_space(12.0);
    ui.label(RichText::new(tl!(&text)).color(t.text_dim).size(12.0));
    let (r, resp) = ui.allocate_exact_size(vec2(18.0, 18.0), Sense::click());
    if resp.hovered() {
        ui.painter().rect_filled(r, t.radius_sm, t.hover);
    }
    // Points along the bar, away from the zoom field.
    icons::paint(ui, r, if crate::direction::is_rtl() { "chevron-left" } else { "chevron-right" }, 11.0, t.text_dim);
    let resp = resp.on_hover_text(tl!("Show"));
    egui::Popup::menu(&resp).show(|ui| {
        crate::widgets::style_spectrum_popup_menu(ui);
        ui.set_min_width(200.0);
        for (key, label) in STATUS_INFO {
            let on = app.ui.chrome.status_info == *key;
            if ui.add(egui::Button::selectable(on, *label)).clicked() {
                app.ui.chrome.status_info = (*key).to_string();
                ui.close();
            }
        }
    });
    status_message(app, ui, &t);
}

/// The latest status message after a separator, in the warning colour when it is an error.
fn status_message(app: &PhotocraftApp, ui: &mut egui::Ui, t: &Tokens) {
    if app.ui.status.is_empty() {
        return;
    }
    let (r, _) = ui.allocate_exact_size(vec2(17.0, 16.0), Sense::hover());
    ui.painter().line_segment([r.center_top(), r.center_bottom()], Stroke::new(1.0, t.separator));
    let is_err = app.ui.status_error || app.ui.status.starts_with("Couldn");
    ui.label(RichText::new(&app.ui.status).color(if is_err { t.warning } else { t.text_faint }));
}

/// Home button at the very start of Photoshop 2026's options bar: toggles the Home (start)
/// screen over the open documents, which stay open.
pub fn home_button(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let n = app.session.documents().len();
    let auto = app.session.prefs().general.auto_show_home_screen;
    let on = app.ui.chrome.shows_home(n, auto);
    // With no documents and auto-show on, Home can't be dismissed (there's nothing behind it).
    let can_toggle = n > 0 || !auto;
    if icons::button(ui, "house", 26.0, on && can_toggle, tl!("Home")).clicked() && can_toggle {
        app.ui.chrome.home = if on { None } else { Some(n) };
    }
}

/// Marquee drag end point under the options-bar Style (Normal, Fixed Ratio, Fixed Size) and Shift
/// (square/circle while drawing with Normal style), Photoshop behaviour.
pub fn marquee_end(style: &str, w: f64, h: f64, shift: bool, start: [f64; 2], end: [f64; 2]) -> [f64; 2] {
    let (dx, dy) = (end[0] - start[0], end[1] - start[1]);
    let (sx, sy) = (if dx < 0.0 { -1.0 } else { 1.0 }, if dy < 0.0 { -1.0 } else { 1.0 });
    match style {
        "fixedSize" => [start[0] + sx * w.max(1.0), start[1] + sy * h.max(1.0)],
        "fixedRatio" if w > 0.0 && h > 0.0 => {
            // The larger drag extent wins; the other follows the ratio.
            let k = (dx.abs() / w).max(dy.abs() / h);
            [start[0] + sx * k * w, start[1] + sy * k * h]
        }
        _ if shift => {
            let m = dx.abs().max(dy.abs());
            [start[0] + sx * m, start[1] + sy * m]
        }
        _ => end,
    }
}

/// Crop options bar ratio presets: (key, label).
pub const CROP_RATIOS: &[(&str, &str)] = &[
    ("", "Ratio"),
    ("original", "Original Ratio"),
    ("1:1", "1 : 1 (Square)"),
    ("4:5", "4 : 5 (8 : 10)"),
    ("5:7", "5 : 7"),
    ("2:3", "2 : 3 (4 : 6)"),
    ("16:9", "16 : 9"),
];

/// Width/height of a crop ratio key (`original` uses the document size).
pub fn crop_ratio(key: &str, doc_w: f64, doc_h: f64) -> Option<(f64, f64)> {
    if key == "original" {
        return Some((doc_w, doc_h));
    }
    let (a, b) = key.split_once(':')?;
    Some((crate::numeric_expression::parse(a)?, crate::numeric_expression::parse(b)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn doc() -> Document {
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 2400, "height": 1500, "resolution": 72, "background": "white"})).unwrap();
        (*s.active().unwrap().doc).clone()
    }

    #[test]
    fn default_status_shows_dimensions_like_photoshop() {
        let d = doc();
        assert_eq!(ChromeState::default().status_info, "dimensions");
        assert_eq!(status_info_text(&d, "dimensions", "", ""), "2400 px x 1500 px (72 ppi)");
        assert_eq!(status_info_text(&d, "layers", "", ""), "1 Layer");
        assert_eq!(status_info_text(&d, "tool", "Brush Tool", ""), "Brush Tool");
        assert!(status_info_text(&d, "profile", "", "sRGB IEC61966-2.1").ends_with("(8bpc)"));
    }

    #[test]
    fn status_dimensions_follow_ruler_units_and_document_resolution() {
        use photocraft_engine::prefs::{PointSize, Unit, UnitsAndRulers};
        let mut d = doc();
        d.resolution_dpi = 300.0;
        for (unit, expected) in [
            (Unit::Pixels, "2400 px x 1500 px (300 ppi)"),
            (Unit::Inches, "8 in x 5 in (300 ppi)"),
            (Unit::Centimeters, "20.32 cm x 12.7 cm (300 ppi)"),
            (Unit::Millimeters, "203.2 mm x 127 mm (300 ppi)"),
            (Unit::Points, "576 pt x 360 pt (300 ppi)"),
            (Unit::Picas, "48 pica x 30 pica (300 ppi)"),
            (Unit::Percent, "100 % x 100 % (300 ppi)"),
        ] {
            let units = UnitsAndRulers { rulers: unit, ..Default::default() };
            assert_eq!(status_info_text_with_units(&d, "dimensions", "", "", &units), expected);
            for (key, _) in STATUS_INFO.iter().filter(|(key, _)| *key != "dimensions") {
                assert_eq!(status_info_text_with_units(&d, key, "Brush Tool", "sRGB", &units), status_info_text(&d, key, "Brush Tool", "sRGB"));
            }
        }
        d.resolution_dpi = 240.0;
        let units = UnitsAndRulers { rulers: Unit::Points, point_size: PointSize::Traditional, ..Default::default() };
        assert_eq!(status_info_text_with_units(&d, "dimensions", "", "", &units), "722.7 pt x 451.7 pt (240 ppi)");
    }

    #[test]
    fn status_bar_updates_dimensions_after_a_ruler_preference_change() {
        use egui_kittest::{Harness, kittest::Queryable};
        let mut failures = Vec::new();
        for theme in [crate::theme::ThemeKind::Pro, crate::theme::ThemeKind::ProMedium] {
            for width in [800.0, 1200.0] {
                let builder = Harness::builder().with_size(vec2(width, 600.0)).with_max_steps(64);
                let mut h = builder.build_eframe(move |cc| {
                    PhotocraftApp::setup_context(&cc.egui_ctx, theme);
                    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
                    app.run("prefs.set", json!({"path": "interface.theme", "value": theme.id()})).unwrap();
                    app.ui.theme = theme;
                    app
                });
                h.state_mut().run("file.new", json!({"width": 1920, "height": 1080, "resolution": 72})).unwrap();
                h.state_mut().sync_views();
                h.run_steps(4);
                assert_eq!(h.state().ui.theme, theme, "the fixture renders the requested theme");
                assert!(h.query_by_label("1920 px x 1080 px (72 ppi)").is_some());
                h.state_mut().run("prefs.set", json!({"path": "unitsAndRulers.rulers", "value": "inches"})).unwrap();
                h.run_steps(4);
                if h.query_by_label("26.667 in x 15 in (72 ppi)").is_none() || h.query_by_label("1920 px x 1080 px (72 ppi)").is_some() {
                    failures.push(format!("{theme:?}, width={width}: inches"));
                }
                h.state_mut().run("prefs.set", json!({"path": "unitsAndRulers.rulers", "value": "cm"})).unwrap();
                h.run_steps(4);
                if h.query_by_label("67.73 cm x 38.1 cm (72 ppi)").is_none() {
                    failures.push(format!("{theme:?}, width={width}: centimeters"));
                }
            }
        }
        assert!(failures.is_empty(), "the status bar must follow the preference immediately: {failures:?}");
    }

    #[test]
    fn status_profile_describes_rgb_gray_and_custom_profiles_without_mutating_the_document() {
        let mut rgb = doc();
        rgb.icc_profile = Some(photocraft_engine::color_cmds::working_profile(rgb.mode).to_bytes());
        let before = rgb.icc_profile.clone();
        assert_eq!(profile_name(&rgb), "sRGB IEC61966-2.1");
        assert_eq!(status_info_text(&rgb, "profile", "", &profile_name(&rgb)), "sRGB IEC61966-2.1 (8bpc)");
        assert_eq!(rgb.icc_profile, before);

        let mut gray = doc();
        gray.mode = photocraft_doc::ColorMode::Grayscale;
        gray.icc_profile = Some(photocraft_engine::color_cmds::working_profile(gray.mode).to_bytes());
        assert_eq!(profile_name(&gray), "Dot Gain 20%-compatible (gamma 1.732, Photocraft)");

        let mut custom = (*photocraft_engine::color_cmds::working_profile(photocraft_doc::ColorMode::Cmyk)).clone();
        custom.description = "PhotoCraft Studio CMYK".into();
        let mut custom_doc = doc();
        custom_doc.mode = photocraft_doc::ColorMode::Cmyk;
        custom_doc.icc_profile = Some(custom.with_encoded_bytes().to_bytes());
        assert_eq!(profile_name(&custom_doc), "PhotoCraft Studio CMYK");
    }

    #[test]
    fn status_profile_falls_back_for_malformed_unnamed_and_mismatched_profiles() {
        let mut malformed = doc();
        malformed.icc_profile = Some(std::sync::Arc::new(vec![1, 2, 3]));
        assert_eq!(profile_name(&malformed), "Invalid RGB profile");

        let mut unnamed_profile = (*photocraft_engine::color_cmds::working_profile(photocraft_doc::ColorMode::Cmyk)).clone();
        unnamed_profile.description.clear();
        let mut unnamed = doc();
        unnamed.mode = photocraft_doc::ColorMode::Cmyk;
        unnamed.icc_profile = Some(unnamed_profile.with_encoded_bytes().to_bytes());
        assert_eq!(profile_name(&unnamed), "Unnamed CMYK profile");

        let mut mismatched = doc();
        mismatched.mode = photocraft_doc::ColorMode::Grayscale;
        mismatched.icc_profile = Some(photocraft_engine::color_cmds::working_profile(photocraft_doc::ColorMode::Rgb).to_bytes());
        assert_eq!(profile_name(&mismatched), "Invalid Gray profile");
    }

    #[test]
    fn document_sizes_match_photoshop_rounding() {
        let d = doc();
        // 2400 x 1500 x 3 bytes = 10.3M flattened, as Photoshop shows.
        assert_eq!(status_info_text(&d, "sizes", "", ""), "Doc: 10.3M/13.7M");
        assert_eq!(fmt_bytes(512 * 1024), "512.0K");
    }

    #[test]
    fn home_screen_closes_when_a_document_opens() {
        let mut c = ChromeState::default();
        assert!(c.shows_home(0, true));
        assert!(!c.shows_home(2, true));
        c.home = Some(2);
        assert!(c.shows_home(2, true));
        assert!(!c.shows_home(3, true));
        assert!(c.home.is_none(), "leaving Home by opening a document dismisses it");
        c.home = Some(1);
        assert!(!c.shows_home(2, true));
        assert!(!c.shows_home(1, true), "closing back to the stored count must not re-open Home");
        assert!(c.home.is_none());
    }

    #[test]
    fn closing_a_tab_does_not_return_to_home() {
        use crate::PhotocraftApp;
        use photocraft_engine::Session;
        use serde_json::json;

        let mut app = PhotocraftApp::new(Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 8, "height": 8, "name": "Keep"})).unwrap();
        let n = app.session.documents().len();
        app.ui.chrome.home = Some(n);
        assert!(app.ui.chrome.shows_home(n, true));
        app.run("file.new", json!({"width": 8, "height": 8, "name": "Other"})).unwrap();
        assert!(!app.ui.chrome.shows_home(app.session.documents().len(), true));
        app.run("file.close", json!({"document": 1, "discard": true})).unwrap();
        assert_eq!(app.session.documents().len(), 1);
        assert!(!app.ui.chrome.shows_home(1, true), "Home must stay dismissed after closing the extra tab");
        assert_eq!(app.session.documents()[0].doc.name, "Keep");
    }

    #[test]
    fn auto_show_home_screen_off_leaves_an_empty_workspace() {
        let mut c = ChromeState::default();
        assert!(!c.shows_home(0, false), "no Home by itself");
        c.home = Some(0);
        assert!(c.shows_home(0, false), "the Home button still opens it");
        assert!(!c.shows_home(1, false), "opening a document leaves it");
    }

    #[test]
    fn marquee_styles_constrain_the_drag() {
        assert_eq!(marquee_end("normal", 1.0, 1.0, false, [10.0, 10.0], [40.0, 20.0]), [40.0, 20.0]);
        assert_eq!(marquee_end("normal", 1.0, 1.0, true, [10.0, 10.0], [40.0, 20.0]), [40.0, 40.0]);
        assert_eq!(marquee_end("fixedSize", 64.0, 32.0, false, [10.0, 10.0], [5.0, 50.0]), [-54.0, 42.0]);
        assert_eq!(marquee_end("fixedRatio", 2.0, 1.0, false, [0.0, 0.0], [10.0, 30.0]), [60.0, 30.0]);
    }

    #[test]
    fn crop_ratios_parse() {
        assert_eq!(crop_ratio("16:9", 1.0, 1.0), Some((16.0, 9.0)));
        assert_eq!(crop_ratio("original", 2400.0, 1500.0), Some((2400.0, 1500.0)));
        assert_eq!(crop_ratio("", 1.0, 1.0), None);
        let (w, h) = crop_ratio("1:1", 0.0, 0.0).unwrap();
        assert_eq!(marquee_end("fixedRatio", w, h, false, [0.0, 0.0], [50.0, 20.0]), [50.0, 50.0]);
    }

    #[test]
    fn status_message_shows_with_no_document_open() {
        // #1567: File › Automate › Batch with no recorded action and no document answered nothing.
        use egui_kittest::{Harness, kittest::Queryable};
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.actions.list.clear();
        let e = crate::menus::invoke(&mut app, &egui::Context::default(), "file.automate.batch", json!({})).unwrap_err();
        app.ui.status = e.clone();
        app.ui.status_error = true;
        let mut h = Harness::builder().with_size(vec2(900.0, 40.0)).build_ui_state(|ui, app| status_bar_pro(app, ui), app);
        h.run_steps(2);
        assert!(h.query_by_label("No document").is_some());
        assert!(h.query_by_label(&e).is_some(), "status bar shows {e:?}");
    }

    #[test]
    fn every_status_key_has_text() {
        let d = doc();
        for (k, _) in STATUS_INFO {
            assert!(!status_info_text(&d, k, "x", "p").is_empty());
        }
    }
}
