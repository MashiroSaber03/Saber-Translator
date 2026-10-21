//! The Color panel tab draws Photoshop's spectrum field in every theme (#4367): the light themes
//! used to show HSB sliders only. Those themes keep the sliders, under the field.

use egui::{Rect, pos2};
use egui_kittest::kittest::Queryable;

use crate::PhotocraftApp;
use crate::dock::Group;
use crate::theme::{ThemeKind, Tokens};

fn left_top() -> egui::Id {
    egui::Id::new("color-tab-test-origin")
}

/// The Color tab of `kind`, alone in a 300 pt wide panel, starting from a red foreground.
fn color_tab(kind: ThemeKind) -> egui_kittest::Harness<'static, PhotocraftApp> {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.session.tools.foreground = [1.0, 0.0, 0.0, 1.0];
    let tab = Group::Color.tabs(Tokens::for_kind(kind).pro).iter().position(|name| *name == "Color").unwrap();
    let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(300.0, 400.0)).build_ui_state(
        move |ui, app: &mut PhotocraftApp| {
            if ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                let origin = ui.cursor().min;
                ui.data_mut(|d| d.insert_temp(left_top(), Rect::from_min_size(origin, egui::Vec2::ZERO)));
                crate::panels::dock_body(app, ui, Group::Color, tab);
            }
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, kind);
    h.run_steps(3);
    h
}

#[test]
fn every_theme_shows_the_spectrum_field() {
    for kind in ThemeKind::ALL {
        let mut h = color_tab(kind);
        let origin = h.ctx.data(|d| d.get_temp::<Rect>(left_top())).unwrap().min;
        // Left edge of the saturation/brightness field (after the 38 pt chips and the 8 pt gap),
        // halfway down its 120 pt: no saturation, half brightness.
        let p = pos2(origin.x + 38.0 + 8.0 + 1.0, origin.y + 60.0);
        h.hover_at(p);
        h.run_steps(1);
        h.drag_at(p);
        h.run_steps(1);
        h.drop_at(p);
        h.run_steps(2);
        let [r, g, b, _] = h.state().session.tools.foreground;
        assert!((r - g).abs() < 0.02 && (g - b).abs() < 0.02 && (0.3..0.7).contains(&r), "{kind:?}: the field picks a mid grey, got {r} {g} {b}");
        if !Tokens::for_kind(kind).pro {
            assert!(h.query_by_label("Saturation").is_some(), "{kind:?}: the sliders stay under the field");
        }
    }
}
