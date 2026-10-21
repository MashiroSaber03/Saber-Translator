//! The Eyedropper tool's options bar (#1649): Sample Size (Point Sample or an N by N Average),
//! Sample (which layers) and Show Sampling Ring, as in Photoshop. The values live in
//! `state::ToolOptions` (`eyedropper_size`, `eyedropper_sample`, `eyedropper_ring`), so the
//! control channel can read and set them; sampling itself is `document.sampleColor`
//! (`canvas::eyedropper_color`). The Sample Size is shared with the painting tools' ⌥-click
//! eyedropper, the dialog eyedroppers and the Info panel, as in Photoshop.

use crate::PhotocraftApp;
use crate::theme::Tokens;
use crate::widgets;

/// The Sample Size menu: each size with its label, in Photoshop's order.
pub fn size_options() -> [(u32, &'static str); 7] {
    [
        (1, tl!("Point Sample")),
        (3, tl!("3 by 3 Average")),
        (5, tl!("5 by 5 Average")),
        (11, tl!("11 by 11 Average")),
        (31, tl!("31 by 31 Average")),
        (51, tl!("51 by 51 Average")),
        (101, tl!("101 by 101 Average")),
    ]
}

/// The Sample menu: `document.sampleColor`'s `sampleLayer` ids with their labels.
pub fn layer_options() -> [(String, &'static str); 5] {
    [
        ("current".into(), tl!("Current Layer")),
        ("currentAndBelow".into(), tl!("Current & Below")),
        ("all".into(), tl!("All Layers")),
        ("allNoAdjustments".into(), tl!("All Layers No Adjustments")),
        ("currentAndBelowNoAdjustments".into(), tl!("Current & Below No Adjustments")),
    ]
}

fn opt(ui: &mut egui::Ui, text: &str) {
    let t = Tokens::get(ui.ctx());
    ui.label(egui::RichText::new(text).color(t.text_dim).size(12.0));
}

/// The options bar for the Eyedropper tool.
pub fn options(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let o = &mut app.ui.tool_options;
    opt(ui, tl!("Sample Size:"));
    widgets::dropdown(ui, "eyedropper-size", &mut o.eyedropper_size, &size_options(), 130.0);
    opt(ui, tl!("Sample:"));
    widgets::dropdown(ui, "eyedropper-sample", &mut o.eyedropper_sample, &layer_options(), 190.0);
    widgets::checkbox(ui, &mut o.eyedropper_ring, tl!("Show Sampling Ring"));
    widgets::vline(ui, 22.0);
    let t = Tokens::get(ui.ctx());
    ui.label(
        egui::RichText::new(crate::i18n::fmt(
            tl!("Click to sample the foreground colour  ·  {key}-click for background"),
            &[("key", &crate::shortcuts::pretty("Alt"))],
        ))
        .color(t.text_faint),
    );
}

#[cfg(test)]
mod tests {
    use egui::vec2;
    use egui_kittest::Harness;
    use egui_kittest::kittest::Queryable;
    use serde_json::json;

    use crate::PhotocraftApp;
    use crate::canvas::{ToolEvent, tool_event};
    use crate::state::Tool;

    /// An app with a 40×40 white document whose top-left 3×3 pixels are black.
    fn app() -> PhotocraftApp {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 40, "height": 40, "background": "white"})).unwrap();
        app.run("tools.setColors", json!({"foreground": [0.0, 0.0, 0.0, 1.0]})).unwrap();
        app.session
            .edit("paint", |doc, _| {
                let bg = doc.layers[0].surface_mut().unwrap();
                let black = photocraft_raster::from_rgba(&bg.format(), [0.0, 0.0, 0.0, 1.0]);
                bg.fill_rect(photocraft_geom::Rect::new(0, 0, 3, 3), &black);
                Ok(())
            })
            .unwrap();
        app.ui.tool = Tool::Eyedropper;
        app
    }

    fn fg(app: &PhotocraftApp) -> [f32; 4] {
        app.session.tools.foreground
    }

    fn click(app: &mut PhotocraftApp, x: f64, y: f64, mods: egui::Modifiers) {
        tool_event(app, ToolEvent::Down { x, y, pressure: 1.0 }, mods);
        tool_event(app, ToolEvent::Up { x, y }, mods);
    }

    #[test]
    fn a_click_samples_the_average_of_the_sample_size() {
        let mut app = app();
        // Point Sample at (3, 3): white.
        click(&mut app, 3.5, 3.5, egui::Modifiers::NONE);
        assert_eq!(fg(&app), [1.0; 4]);
        // 3 by 3 Average: one black pixel of nine, rounded to 8 bit.
        app.ui.tool_options.eyedropper_size = 3;
        click(&mut app, 3.5, 3.5, egui::Modifiers::NONE);
        let want = (255.0f32 * 8.0 / 9.0).round() / 255.0;
        assert_eq!(fg(&app), [want, want, want, 1.0]);
        // 5 by 5 at the corner: clipped to the black 3×3.
        app.ui.tool_options.eyedropper_size = 5;
        click(&mut app, 0.5, 0.5, egui::Modifiers::NONE);
        assert_eq!(fg(&app), [0.0, 0.0, 0.0, 1.0]);
        // ⌥-click sets the background colour, with the same size.
        app.ui.tool_options.eyedropper_size = 3;
        click(&mut app, 3.5, 3.5, egui::Modifiers::ALT);
        assert_eq!(app.session.tools.background, [want, want, want, 1.0]);
        // The painting tools' ⌥-click eyedropper uses it too.
        app.ui.tool = Tool::Brush;
        app.ui.tool_options.eyedropper_size = 101;
        click(&mut app, 20.0, 20.0, egui::Modifiers::ALT);
        let want = (255.0f32 * (1600.0 - 9.0) / 1600.0).round() / 255.0;
        assert_eq!(fg(&app), [want, want, want, 1.0]);
        // The dialog eyedroppers (and the Color Picker's) average the same square.
        app.ui.tool_options.eyedropper_size = 5;
        let c = crate::canvas::composite_color(&mut app, 1.0, 1.0).unwrap();
        let want = (255.0f32 * 7.0 / 16.0).round() / 255.0;
        assert_eq!(c, [want; 3]);
    }

    #[test]
    fn the_sample_menu_picks_the_layers() {
        let mut app = app();
        // A new empty layer on top: Current Layer finds nothing; All Layers the background.
        app.run("layer.new.layer", json!({})).unwrap();
        app.ui.tool_options.eyedropper_size = 3;
        app.ui.tool_options.eyedropper_sample = "current".into();
        app.run("tools.setColors", json!({"foreground": [0.2, 0.4, 0.6, 1.0]})).unwrap();
        click(&mut app, 10.0, 10.0, egui::Modifiers::NONE);
        assert_eq!(fg(&app), [0.2, 0.4, 0.6, 1.0], "nothing to sample on an empty layer");
        app.ui.tool_options.eyedropper_sample = "all".into();
        click(&mut app, 10.0, 10.0, egui::Modifiers::NONE);
        assert_eq!(fg(&app), [1.0; 4]);
    }

    #[test]
    fn tool_options_round_trip_through_serde_with_defaults() {
        let o = crate::state::ToolOptions::default();
        assert_eq!((o.eyedropper_size, o.eyedropper_sample.as_str(), o.eyedropper_ring), (1, "all", true));
        // Older saved state without the fields loads with Photoshop's defaults.
        let old: crate::state::ToolOptions = serde_json::from_value(json!({"tolerance": 12.0})).unwrap();
        assert_eq!((old.eyedropper_size, old.eyedropper_sample.as_str(), old.eyedropper_ring), (1, "all", true));
        let mut o = o;
        o.eyedropper_size = 31;
        o.eyedropper_sample = "currentAndBelow".into();
        o.eyedropper_ring = false;
        let back: crate::state::ToolOptions = serde_json::from_value(serde_json::to_value(&o).unwrap()).unwrap();
        assert_eq!(back, o);
        // Every menu entry is a size and a layer choice the engine accepts.
        for (size, _) in super::size_options() {
            assert!(photocraft_engine::sample_cmds::SAMPLE_SIZES.contains(&size));
        }
        for (id, _) in super::layer_options() {
            assert!(photocraft_engine::sample_cmds::SampleLayers::parse(&id).is_some(), "{id}");
        }
    }

    /// The options bar: the Sample Size dropdown sets the size, and the next click on the canvas
    /// samples that average.
    #[test]
    fn the_options_bar_dropdown_sets_the_sample_size() {
        let mut h = Harness::builder().with_size(vec2(1400.0, 600.0)).build_ui_state(
            |ui, app: &mut PhotocraftApp| {
                if !ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                    return;
                }
                crate::panels::options_bar(app, ui);
            },
            app(),
        );
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::Studio);
        h.run_steps(4);
        for label in ["Sample Size:", "Sample:", "Show Sampling Ring"] {
            assert!(h.get_by_label(label).rect().is_positive(), "missing Eyedropper option {label}");
        }
        let combo = |h: &Harness<'static, PhotocraftApp>, i: usize| h.get_all_by_role(egui::accesskit::Role::ComboBox).nth(i).unwrap().click();
        combo(&h, 0);
        h.run_steps(3);
        h.get_by_label("5 by 5 Average").click();
        h.run_steps(3);
        assert_eq!(h.state().ui.tool_options.eyedropper_size, 5);
        combo(&h, 1);
        h.run_steps(3);
        h.get_by_label("Current Layer").click();
        h.run_steps(3);
        assert_eq!(h.state().ui.tool_options.eyedropper_sample, "current");
        h.get_by_label("Show Sampling Ring").click();
        h.run_steps(3);
        assert!(!h.state().ui.tool_options.eyedropper_ring);
        // A click at (1, 3) now averages the 5×5 around it: rows 1..6 × cols 0..4 hold 6 black.
        click(h.state_mut(), 1.5, 3.5, egui::Modifiers::NONE);
        let want = (255.0f32 * 14.0 / 20.0).round() / 255.0;
        assert_eq!(fg(h.state()), [want, want, want, 1.0]);
    }
}
