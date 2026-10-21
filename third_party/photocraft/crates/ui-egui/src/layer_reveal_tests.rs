//! #152: a new active layer opens its parent groups and is scrolled into view.

use egui::{Rect, vec2};
use egui_kittest::{Harness, kittest::Queryable};
use photocraft_doc::LayerId;
use serde_json::json;

use crate::PhotocraftApp;
use crate::dock::{Group, last_rects};
use crate::theme::ThemeKind;

/// 30 layers above a closed group holding `inner` (near the bottom of the panel).
fn tall_doc() -> (PhotocraftApp, LayerId, LayerId, LayerId, LayerId) {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 200, "height": 150})).unwrap();
    app.run("layer.new.layer", json!({"name": "Deep inside"})).unwrap();
    let active = |app: &PhotocraftApp| app.session.active().unwrap().active_layer.unwrap();
    let inner = active(&app);
    app.run("layer.groupLayers", json!({})).unwrap();
    let group = app.session.active().unwrap().doc.walk().into_iter().find(|(_, _, l)| l.is_group()).map(|(_, _, l)| l.id).unwrap();
    app.run("layer.setExpanded", json!({"layer": group.0, "expanded": false})).unwrap();
    app.run("layer.select", json!({"layer": group.0})).unwrap();
    let mut stack = Vec::new();
    for i in 0..30 {
        app.run("layer.new.layer", json!({"name": format!("Stack {i:02}")})).unwrap();
        stack.push(active(&app));
    }
    app.sync_views();
    (app, inner, group, stack[29], stack[28])
}

fn harness(app: PhotocraftApp, theme: ThemeKind) -> Harness<'static, PhotocraftApp> {
    let mut h = Harness::builder().with_size(vec2(1440.0, 900.0)).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            if !ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            crate::panels::right_dock(app, ui);
            egui::CentralPanel::default().show(ui, |_| {});
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, theme);
    h.state_mut().ui.theme = theme;
    h.run_steps(4);
    h
}

/// The Layers panel row of `name`, if drawn.
fn row(h: &Harness<'static, PhotocraftApp>, name: &str) -> Option<Rect> {
    let layers = last_rects(&h.ctx).into_iter().find(|(g, _)| *g == Group::Layers)?.1;
    h.query_all_by_label(name).map(|n| n.rect()).find(|r| layers.contains(r.center()) || (r.left() >= layers.left() && r.right() <= layers.right()))
}

/// Fully inside the rows viewport (below the strip and the blend/lock rows, above the footer).
fn in_view(h: &Harness<'static, PhotocraftApp>, r: Rect) -> bool {
    let layers = last_rects(&h.ctx).into_iter().find(|(g, _)| *g == Group::Layers).unwrap().1;
    r.top() >= layers.top() + 26.0 && r.bottom() <= layers.bottom() - (crate::widgets::FOOTER_BAR + 1.0)
}

fn expanded(app: &PhotocraftApp, group: LayerId) -> bool {
    matches!(&app.session.active().unwrap().doc.layer(group).unwrap().content, photocraft_doc::LayerContent::Group(g) if g.expanded)
}

#[test]
fn a_new_active_layer_opens_its_groups_and_scrolls_into_view() {
    for theme in [ThemeKind::ProMedium, ThemeKind::Studio] {
        let (app, inner, group, top, below_top) = tall_doc();
        let mut h = harness(app, theme);
        let top_row = row(&h, "Stack 29").expect("top row drawn");
        assert!(in_view(&h, top_row), "{theme:?}: the panel starts at the top");
        assert!(row(&h, "Deep inside").is_none(), "{theme:?}: hidden in its closed group");

        // Select › layer, Auto-Select, automation: all end in a new active layer.
        h.state_mut().run("layer.select", json!({"layer": inner.0})).unwrap();
        h.run_steps(4);
        assert!(expanded(h.state(), group), "{theme:?}: the parent group opened");
        let r = row(&h, "Deep inside").expect("row listed once its group is open");
        assert!(in_view(&h, r), "{theme:?}: scrolled into view: {r:?}");
        assert!(!in_view(&h, row(&h, "Stack 29").unwrap()), "{theme:?}: the top scrolled away");

        // Back to the top layer: scrolls back up.
        h.state_mut().run("layer.select", json!({"layer": top.0})).unwrap();
        h.run_steps(4);
        assert!(in_view(&h, row(&h, "Stack 29").unwrap()), "{theme:?}: scrolled back to the top");

        // A layer already in view doesn't move the list.
        let before = row(&h, "Stack 29").unwrap();
        h.state_mut().run("layer.select", json!({"layer": below_top.0})).unwrap();
        h.run_steps(4);
        assert_eq!(row(&h, "Stack 29").unwrap(), before, "{theme:?}: no scroll for a visible row");

        // Closing the group by hand is respected until the active layer changes again.
        h.state_mut().run("layer.select", json!({"layer": inner.0})).unwrap();
        h.run_steps(3);
        h.state_mut().run("layer.setExpanded", json!({"layer": group.0, "expanded": false})).unwrap();
        h.run_steps(4);
        assert!(!expanded(h.state(), group), "{theme:?}: stays closed");
        // Undo and other commands that change the active layer reveal it too: undoing a new
        // layer made deep in the list makes the layer active before it active again (#1356).
        h.state_mut().run("layer.select", json!({"layer": top.0})).unwrap();
        h.run_steps(3);
        h.state_mut().run("layer.select", json!({"layer": inner.0})).unwrap();
        h.run_steps(3);
        assert!(expanded(h.state(), group));
        h.state_mut().run("layer.new.layer", json!({"name": "Scratch"})).unwrap();
        h.run_steps(4);
        h.state_mut().run("layer.select", json!({"layer": top.0})).unwrap();
        h.run_steps(4);
        assert!(!in_view(&h, row(&h, "Deep inside").unwrap()));
        h.state_mut().run("edit.undo", json!({})).unwrap();
        h.run_steps(4);
        assert_eq!(h.state().session.active().unwrap().active_layer, Some(inner), "{theme:?}: undo made the inner layer active");
        assert!(in_view(&h, row(&h, "Deep inside").unwrap()), "{theme:?}: undo revealed it");
    }
}

#[test]
fn reveal_survives_odd_documents() {
    // No document, a document without layers in groups, a layer deleted between frames.
    let app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    let mut h = harness(app, ThemeKind::ProMedium);
    h.run_steps(2);
    h.state_mut().run("file.new", json!({"width": 20, "height": 20})).unwrap();
    h.state_mut().sync_views();
    h.run_steps(3);
    h.state_mut().run("layer.new.layer", json!({})).unwrap();
    h.run_steps(3);
    h.state_mut().run("layer.delete", json!({})).unwrap();
    h.run_steps(3);
    h.state_mut().run("file.close", json!({})).ok();
    h.state_mut().sync_views();
    h.run_steps(3);
}
