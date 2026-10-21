//! Undo inside a dialog (#4697): ⌘Z / ⇧⌘Z step through the open dialog's own settings, as in
//! Photoshop, instead of reaching the document's History beneath it (which the dialog previews
//! and is about to change), or doing nothing.
//!
//! A dialog keeps an [`EditHistory`] of its settings and calls [`EditHistory::settle`] every frame
//! and after a control-channel edit. A change becomes one step once no edit is in progress, so a
//! slider drag is one step rather than one per frame. Liquify, Puppet Warp and Camera Raw keep
//! their own histories.

use std::collections::HashMap;

use serde_json::{Map, Value};

use crate::PhotocraftApp;
use crate::state::DialogKind;

/// Most steps a dialog keeps.
const LIMIT: usize = 100;

pub struct EditHistory<T> {
    undo: Vec<T>,
    redo: Vec<T>,
    /// The settings as of the latest step.
    settled: T,
}

impl<T: Clone + PartialEq> EditHistory<T> {
    pub fn new(now: &T) -> Self {
        Self { undo: Vec::new(), redo: Vec::new(), settled: now.clone() }
    }

    /// Records a change since the latest step as a new step, unless `busy` (a drag in progress).
    pub fn settle(&mut self, now: &T, busy: bool) {
        if busy || *now == self.settled {
            return;
        }
        if self.undo.len() >= LIMIT {
            self.undo.remove(0);
        }
        self.undo.push(std::mem::replace(&mut self.settled, now.clone()));
        self.redo.clear();
    }

    /// Steps `now` back (`Undo`) or forward (`Redo`); an unfinished edit is the latest step.
    /// False when there is nowhere to go.
    pub fn step(&mut self, now: &mut T, step: Step) -> bool {
        self.settle(now, false);
        let (from, to) = match step {
            Step::Undo => (&mut self.undo, &mut self.redo),
            Step::Redo => (&mut self.redo, &mut self.undo),
        };
        let Some(next) = from.pop() else { return false };
        to.push(std::mem::replace(now, next.clone()));
        self.settled = next;
        true
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Undo,
    Redo,
}

/// ⌘Z or ⇧⌘Z pressed this frame, consumed.
fn take_step(ctx: &egui::Context) -> Option<Step> {
    let cmd_shift = egui::Modifiers::COMMAND | egui::Modifiers::SHIFT;
    if ctx.input_mut(|i| i.consume_key(cmd_shift, egui::Key::Z)) {
        Some(Step::Redo)
    } else if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::Z)) {
        Some(Step::Undo)
    } else {
        None
    }
}

/// A parameter dialog's settings: its fields without the private `__` keys (preview toggle,
/// command, label…).
fn settings(fields: &Map<String, Value>) -> Map<String, Value> {
    fields.iter().filter(|(k, _)| !k.starts_with("__")).map(|(k, v)| (k.clone(), v.clone())).collect()
}

/// The parameter dialogs' histories, by dialog id.
pub type DialogEdits = HashMap<u64, EditHistory<Map<String, Value>>>;

/// Once a frame: records each open parameter dialog's settings and forgets closed dialogs.
pub fn track_dialogs(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let busy = ctx.input(|i| i.pointer.any_down());
    let dialogs = &app.ui.dialogs;
    app.dialog_edits.retain(|id, _| dialogs.iter().any(|d| d.id == *id));
    for d in dialogs.iter().filter(|d| d.kind == DialogKind::Command) {
        let now = settings(&d.fields);
        app.dialog_edits.entry(d.id).or_insert_with(|| EditHistory::new(&now)).settle(&now, busy);
    }
}

/// ⌘Z / ⇧⌘Z while a dialog with its own history is in front: steps that dialog, and the keys
/// never reach the document's History beneath it. A focused text field keeps them for its typing.
pub fn keys(app: &mut PhotocraftApp, ctx: &egui::Context) {
    if ctx.text_edit_focused() {
        return;
    }
    if let Some(top) = app.ui.dialogs.last() {
        if top.kind != DialogKind::Command {
            return;
        }
        let id = top.id;
        let mut now = settings(&top.fields);
        if let Some(step) = take_step(ctx)
            && app.dialog_edits.entry(id).or_insert_with(|| EditHistory::new(&now)).step(&mut now, step)
            && let Some(d) = app.ui.dialog_mut(id)
        {
            d.fields.retain(|k, _| k.starts_with("__"));
            d.fields.extend(now);
        }
    } else if let Some(d) = app.distort.gallery.as_mut() {
        if let Some(step) = take_step(ctx) {
            d.step(step);
        }
    } else if let Some(d) = app.wide_angle.as_mut() {
        if let Some(step) = take_step(ctx) {
            d.step(step);
        }
    } else if let Some(s) = app.distort.vanishing.as_mut()
        && let Some(step) = take_step(ctx)
    {
        s.step(step);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use serde_json::json;

    use super::*;

    /// One frame of the shell's key handling with ⌘Z (plus `modifiers`) pressed.
    pub(crate) fn press_z(app: &mut PhotocraftApp, ctx: &egui::Context, modifiers: egui::Modifiers) {
        let raw = egui::RawInput {
            events: vec![egui::Event::Key { key: egui::Key::Z, physical_key: None, pressed: true, repeat: false, modifiers }],
            ..Default::default()
        };
        let mut out = ctx.run_ui(raw, |ui| crate::shortcuts::handle(app, ui.ctx()));
        out.textures_delta.clear();
    }

    fn app() -> PhotocraftApp {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 120, "height": 80})).unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        app.run("edit.fill", json!({"color": "#808080"})).unwrap();
        app
    }

    fn past(app: &PhotocraftApp) -> usize {
        app.session.active().unwrap().history.past_len()
    }

    #[test]
    fn a_drag_is_one_step_and_a_new_edit_drops_redo() {
        let mut h = EditHistory::new(&0);
        for v in 1..=5 {
            h.settle(&v, true);
        }
        h.settle(&5, false);
        h.settle(&7, false);
        let mut now = 7;
        assert!(h.step(&mut now, Step::Undo));
        assert_eq!(now, 5);
        assert!(h.step(&mut now, Step::Undo));
        assert_eq!(now, 0, "the whole drag was one step");
        assert!(!h.step(&mut now, Step::Undo));
        assert!(h.step(&mut now, Step::Redo));
        assert_eq!(now, 5);
        now = 9;
        assert!(!h.step(&mut now, Step::Redo), "a new edit drops what was undone");
        assert!(h.step(&mut now, Step::Undo));
        assert_eq!(now, 5, "an unsettled edit is undone first");
    }

    #[test]
    fn command_z_steps_a_parameter_dialog_and_leaves_its_private_keys() {
        let ctx = egui::Context::default();
        let mut app = app();
        let steps = past(&app);
        let id = crate::dialogs::open_command_dialog(&mut app, "filter.lensCorrection", "Lens Correction");
        track_dialogs(&mut app, &ctx);
        let field = |app: &PhotocraftApp, k: &str| app.ui.dialogs.iter().find(|d| d.id == id).unwrap().fields.get(k).cloned();
        let vertical = field(&app, "vertical");
        let private: Vec<String> = app.ui.dialog_mut(id).unwrap().fields.keys().filter(|k| k.starts_with("__")).cloned().collect();
        assert!(!private.is_empty());
        app.ui.dialog_mut(id).unwrap().fields.insert("vertical".into(), json!(-50.0));
        track_dialogs(&mut app, &ctx);
        app.ui.dialog_mut(id).unwrap().fields.insert("horizontal".into(), json!(30.0));
        track_dialogs(&mut app, &ctx);
        press_z(&mut app, &ctx, egui::Modifiers::COMMAND);
        assert_ne!(field(&app, "horizontal"), Some(json!(30.0)));
        assert_eq!(field(&app, "vertical"), Some(json!(-50.0)), "one step at a time");
        press_z(&mut app, &ctx, egui::Modifiers::COMMAND);
        assert_eq!(field(&app, "vertical"), vertical);
        assert_eq!(past(&app), steps, "the document beneath is untouched");
        for k in &private {
            assert!(field(&app, k).is_some(), "{k} survives");
        }
        press_z(&mut app, &ctx, egui::Modifiers::COMMAND | egui::Modifiers::SHIFT);
        assert_eq!(field(&app, "vertical"), Some(json!(-50.0)));
    }

    #[test]
    fn command_z_steps_the_filter_gallery_stack() {
        let ctx = egui::Context::default();
        let mut app = app();
        let steps = past(&app);
        crate::distort_ui::menu(&mut app, &ctx, "filter.filterGallery", &json!({})).unwrap().unwrap();
        let before = app.distort.gallery.as_ref().unwrap().effects.clone();
        crate::distort_ui::menu(&mut app, &ctx, "filter.filterGallery", &json!({"ui": {"add": true}})).unwrap().unwrap();
        assert_eq!(app.distort.gallery.as_ref().unwrap().effects.len(), before.len() + 1);
        press_z(&mut app, &ctx, egui::Modifiers::COMMAND);
        let d = app.distort.gallery.as_ref().unwrap();
        assert_eq!(d.effects, before);
        assert!(d.selected < d.effects.len());
        assert_eq!(past(&app), steps);
    }

    #[test]
    fn command_z_steps_vanishing_point_planes() {
        let ctx = egui::Context::default();
        let mut app = app();
        let steps = past(&app);
        crate::distort_ui::menu(&mut app, &ctx, "filter.vanishingPoint", &json!({})).unwrap().unwrap();
        let plane = json!([{"corners": [[30, 20], [90, 20], [110, 70], [10, 70]]}]);
        crate::distort_ui::menu(&mut app, &ctx, "filter.vanishingPoint", &json!({"ui": {"planes": plane}})).unwrap().unwrap();
        press_z(&mut app, &ctx, egui::Modifiers::COMMAND);
        let s = app.distort.vanishing.as_ref().unwrap();
        assert!(s.planes.is_empty());
        assert_eq!(s.selected, None, "no selection left on a plane that is gone");
        assert_eq!(past(&app), steps);
        press_z(&mut app, &ctx, egui::Modifiers::COMMAND | egui::Modifiers::SHIFT);
        assert_eq!(app.distort.vanishing.as_ref().unwrap().planes.len(), 1);
    }
}
