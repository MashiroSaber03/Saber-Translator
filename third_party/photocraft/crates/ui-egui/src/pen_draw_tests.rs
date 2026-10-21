//! The Pen's modifiers while drawing, through the real canvas (#3306): ⌥/Alt breaks the handle
//! being pulled out and grabs the last anchor's outgoing handle alone, and dragging on the first
//! anchor closes the path with that anchor's handles pulled out.

use egui::{Key, Modifiers, PointerButton, Pos2, vec2};
use egui_kittest::Harness;
use serde_json::json;

use crate::PhotocraftApp;
use crate::canvas::ViewXform;
use crate::state::Tool;

fn harness() -> Harness<'static, PhotocraftApp> {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 200, "height": 150})).unwrap();
    app.sync_views();
    app.ui.tool = Tool::Pen;
    let mut h = Harness::builder().with_size(vec2(900.0, 600.0)).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            let ctx = ui.ctx().clone();
            if !ctx.fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            crate::shortcuts::handle(app, &ctx);
            egui::CentralPanel::default().show(ui, |ui| crate::canvas::document_area(app, ui));
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
    h.run_steps(4);
    let v = &mut h.state_mut().ui.views[0];
    v.zoom = 1.0;
    v.center = [100.0, 75.0];
    v.fit_pending = false;
    h.run_steps(2);
    h
}

fn screen(h: &Harness<'static, PhotocraftApp>, p: [f32; 2]) -> Pos2 {
    let app = h.state();
    let v = &app.ui.views[0];
    let xf = ViewXform {
        rect: crate::rulers::content_rect(app, app.last_canvas_rect),
        zoom: v.zoom,
        center: v.center,
        flip: app.ui.view.flip_horizontal,
        rotation: v.rotation,
        aspect: app.ui.view.display_aspect(),
    };
    xf.to_screen(p[0], p[1])
}

fn mods(h: &mut Harness<'static, PhotocraftApp>, m: Modifiers) {
    h.event(egui::Event::ModifiersChanged(m));
    h.run_steps(1);
}

fn button(h: &mut Harness<'static, PhotocraftApp>, p: Pos2, down: bool, m: Modifiers) {
    h.event(egui::Event::PointerButton { pos: p, button: PointerButton::Primary, pressed: down, modifiers: m });
    h.run_steps(1);
}

fn move_to(h: &mut Harness<'static, PhotocraftApp>, from: [f32; 2], to: [f32; 2]) {
    for i in 1..=6 {
        let t = i as f32 / 6.0;
        let q = screen(h, [from[0] + (to[0] - from[0]) * t, from[1] + (to[1] - from[1]) * t]);
        h.event(egui::Event::PointerMoved(q));
        h.run_steps(1);
    }
}

/// A drag from `pts[0]` through the other points; `alt_from` is the index of the point from
/// which ⌥/Alt is held (`Some(0)`: before the press).
fn drag(h: &mut Harness<'static, PhotocraftApp>, pts: &[[f32; 2]], alt_from: Option<usize>) {
    let held = |i: usize| if alt_from.is_some_and(|a| i >= a) { Modifiers::ALT } else { Modifiers::NONE };
    let p = screen(h, pts[0]);
    mods(h, held(0));
    h.event(egui::Event::PointerMoved(p));
    h.run_steps(1);
    button(h, p, true, held(0));
    for (i, w) in pts.windows(2).enumerate() {
        mods(h, held(i + 1));
        move_to(h, w[0], w[1]);
    }
    let end = *pts.last().unwrap();
    button(h, screen(h, end), false, held(pts.len() - 1));
    mods(h, Modifiers::NONE);
    h.run_steps(1);
}

fn click(h: &mut Harness<'static, PhotocraftApp>, p: [f32; 2]) {
    drag(h, &[p], None);
}

fn near(a: [f64; 2], b: [f64; 2]) -> bool {
    (a[0] - b[0]).abs() < 0.05 && (a[1] - b[1]).abs() < 0.05
}

#[test]
fn alt_breaks_the_handle_being_dragged_and_grabs_the_last_outgoing_handle() {
    let mut h = harness();
    click(&mut h, [20.0, 20.0]);
    // Drag out a smooth anchor, then hold ⌥ halfway: the incoming handle stays where it was.
    drag(&mut h, &[[100.0, 20.0], [120.0, 40.0], [100.0, 60.0]], Some(2));
    let pen = h.state().ui.pen.clone().unwrap();
    let k = pen.knots[1];
    assert!(near(k[1], [80.0, 0.0]), "incoming handle stayed put: {k:?}");
    assert!(near(k[2], [100.0, 60.0]), "outgoing handle followed the pointer: {k:?}");

    // ⌥-drag the outgoing handle's end: it turns alone and no anchor is added.
    drag(&mut h, &[[100.0, 60.0], [140.0, 20.0]], Some(0));
    let pen = h.state().ui.pen.clone().unwrap();
    assert_eq!(pen.knots.len(), 2, "no new anchor on the handle");
    assert!(near(pen.knots[1][1], [80.0, 0.0]) && near(pen.knots[1][2], [140.0, 20.0]), "{:?}", pen.knots[1]);

    click(&mut h, [150.0, 120.0]);
    h.key_press(Key::Enter);
    h.run_steps(2);
    let doc = &h.state().session.active().unwrap().doc;
    let k = &doc.work_path.as_ref().unwrap().subpaths[0].knots[1];
    assert!(!k.smooth, "the broken handle is a cusp in the work path");
    assert!(near([k.out_ctrl.x, k.out_ctrl.y], [140.0, 20.0]));
}

#[test]
fn dragging_on_the_first_anchor_closes_with_its_handles_in_one_step() {
    let mut h = harness();
    let before = h.state().session.active().unwrap().history.past_len();
    for p in [[20.0, 20.0], [120.0, 20.0], [120.0, 100.0]] {
        click(&mut h, p);
    }
    drag(&mut h, &[[20.0, 20.0], [40.0, 0.0]], None);
    assert!(h.state().ui.pen.is_none(), "the path is finished");
    let st = h.state().session.active().unwrap();
    assert_eq!(st.history.past_len(), before + 1, "one History step");
    let sp = &st.doc.work_path.as_ref().unwrap().subpaths[0];
    assert!(sp.closed && sp.knots.len() == 3, "closed, without an extra anchor: {sp:?}");
    let k = &sp.knots[0];
    assert!(near([k.in_ctrl.x, k.in_ctrl.y], [0.0, 40.0]) && near([k.out_ctrl.x, k.out_ctrl.y], [40.0, 0.0]), "{k:?}");
    assert!(k.smooth);

    // A plain click on a smooth first anchor closes without touching its handles. This path is
    // drawn over the anchors of the one just finished, which Auto Add/Delete would delete (#4144).
    h.state_mut().ui.tool_options.pen_auto_add_delete = false;
    drag(&mut h, &[[20.0, 20.0], [40.0, 20.0]], None);
    click(&mut h, [120.0, 20.0]);
    click(&mut h, [120.0, 100.0]);
    click(&mut h, [20.0, 20.0]);
    // The second path joins the Work Path as its last subpath (#4721).
    let subs = h.state().session.active().unwrap().doc.work_path.clone().unwrap().subpaths;
    assert_eq!(subs.len(), 2, "the first path stays");
    let sp = subs.last().unwrap();
    let k = &sp.knots[0];
    assert!(sp.closed && sp.knots.len() == 3);
    assert!(near([k.in_ctrl.x, k.in_ctrl.y], [0.0, 20.0]) && near([k.out_ctrl.x, k.out_ctrl.y], [40.0, 20.0]), "{k:?}");
}
