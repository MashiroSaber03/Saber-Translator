//! #2966: a click on a History panel state returns the document there with one command
//! (`history.selectState`), not one Undo / Redo per state, so the canvas refreshes once and only
//! where the jump changed pixels.

use egui::vec2;
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use serde_json::json;

use super::*;

/// Open, then New Layer, Fill, New Layer, Fill.
fn harness() -> Harness<'static, PhotocraftApp> {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
    app.run("file.new", json!({"width": 64, "height": 48})).unwrap();
    for (cmd, p) in
        [("layer.new.layer", json!({})), ("edit.fill", json!({"color": "#ff0000"})), ("layer.new.layer", json!({})), ("edit.fill", json!({"color": "#00ff00"}))]
    {
        app.run(cmd, p).unwrap();
    }
    let mut ready = false;
    let mut h = Harness::builder().with_size(vec2(300.0, 400.0)).build_ui_state(
        move |ui, app: &mut PhotocraftApp| {
            if !ready {
                PhotocraftApp::setup_context(ui.ctx(), theme::ThemeKind::ProMedium);
                ready = true;
                return;
            }
            history(app, ui);
        },
        app,
    );
    h.run_steps(3);
    h
}

fn state(h: &Harness<'_, PhotocraftApp>) -> (usize, usize, u64) {
    let st = h.state().session.active().unwrap();
    (st.history.past_len(), st.history.redo_labels().count(), st.revision)
}

#[test]
fn clicking_a_state_jumps_there_with_one_command() {
    let mut h = harness();
    let (past, redo, rev) = state(&h);
    assert_eq!((past, redo), (4, 0));
    // Back three states: one command, one revision (the canvas sees one change).
    h.get_all_by_label("New Layer").next().expect("the first New Layer row").click();
    h.run_steps(2);
    assert_eq!(state(&h), (1, 3, rev + 1));
    assert_eq!(h.state().perf.last_command, "history.selectState");
    // Forward to the last (dimmed) state, again in one step.
    h.get_all_by_label("Fill").last().expect("the last Fill row").click();
    h.run_steps(2);
    assert_eq!(state(&h), (4, 0, rev + 2));
    // The current state: nothing to do.
    h.get_all_by_label("Fill").last().expect("the last Fill row").click();
    h.run_steps(2);
    assert_eq!(state(&h), (4, 0, rev + 2));
}
