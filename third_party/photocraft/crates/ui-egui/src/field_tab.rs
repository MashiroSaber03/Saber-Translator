//! Photoshop-style Tab in dialogs: Tab and ⇧Tab cycle a dialog's text and number fields only
//! (Name → Width → Height → Resolution in New Document, Hue → Saturation → Lightness in
//! Hue/Saturation), never its sliders, dropdowns, checkboxes or buttons, and a field reached this
//! way has its text selected so typing replaces it.
//!
//! egui's own Tab walks every focusable widget in creation order (the title bar, the preset
//! cards, each slider…). So the dialog frame (`dialogs::show`) takes the key before egui's focus
//! machinery acts on it ([`take_step`]), the fields register themselves as the body is laid out
//! ([`register`], from `widgets::number_edit` and the dialogs' text fields), and the frame then
//! hands focus to the next one ([`end`]).

use egui::{Context, Id, Key, Modifiers};

fn list_id() -> Id {
    Id::new("pc-field-tab-order")
}

fn collecting_id() -> Id {
    Id::new("pc-field-tab-collecting")
}

fn pending_id() -> Id {
    Id::new("pc-field-tab-pending")
}

/// Give `id` the focus from the next frame on. Focusing a widget after it was laid out would
/// never read as "gained focus" to it (egui compares with the focus at the start of the frame),
/// and that is what selects a field's text; so the move waits for [`take_step`] next frame.
pub fn focus(ctx: &Context, id: Id) {
    ctx.data_mut(|d| d.insert_temp(pending_id(), Some(id)));
}

/// Tab (+1) or ⇧Tab (−1) pressed this frame, taken from the input so nothing else acts on it;
/// 0 when neither was. Also applies a pending [`focus`]. Call before the dialog is laid out.
pub fn take_step(ctx: &Context) -> i32 {
    if let Some(id) = ctx.data_mut(|d| d.remove_temp::<Option<Id>>(pending_id())).flatten() {
        ctx.memory_mut(|m| m.request_focus(id));
    }
    let step = ctx.input_mut(|i| {
        if i.consume_key(Modifiers::SHIFT, Key::Tab) {
            -1
        } else if i.consume_key(Modifiers::NONE, Key::Tab) {
            1
        } else {
            0
        }
    });
    if step != 0 {
        // egui read the key before this frame's widgets were laid out: cancel its own focus move.
        ctx.memory_mut(|m| m.move_focus(egui::FocusDirection::None));
    }
    step
}

/// Start collecting the fields of a dialog body.
pub fn begin(ctx: &Context) {
    ctx.data_mut(|d| {
        d.insert_temp(collecting_id(), true);
        d.insert_temp(list_id(), Vec::<Id>::new());
    });
}

/// A text or number field, in layout order. Outside a dialog body this does nothing.
pub fn register(ctx: &Context, id: Id) {
    ctx.data_mut(|d| {
        if d.get_temp::<bool>(collecting_id()).unwrap_or(false) {
            d.get_temp_mut_or_default::<Vec<Id>>(list_id()).push(id);
        }
    });
}

/// End of the body: move focus by `step` through the registered fields, wrapping around (from
/// nothing focused, Tab starts at the first and ⇧Tab at the last). Returns the field focused.
pub fn end(ctx: &Context, step: i32) -> Option<Id> {
    let fields: Vec<Id> = ctx.data_mut(|d| {
        d.insert_temp(collecting_id(), false);
        d.get_temp(list_id()).unwrap_or_default()
    });
    if step == 0 || fields.is_empty() {
        return None;
    }
    let n = fields.len() as i32;
    let at = ctx.memory(|m| m.focused()).and_then(|f| fields.iter().position(|&id| id == f)).map(|i| i as i32);
    let next = match at {
        Some(i) => (i + step).rem_euclid(n),
        None if step > 0 => 0,
        None => n - 1,
    };
    let id = fields.get(next as usize).copied()?;
    focus(ctx, id);
    Some(id)
}

/// Select all of a text field's text (so typing replaces it), as Photoshop does when a field is
/// reached with Tab or stepped with the arrow keys.
pub fn select_all(ctx: &Context, id: Id, text: &str) {
    let mut state = egui::TextEdit::load_state(ctx, id).unwrap_or_default();
    state.cursor.set_char_range(Some(egui::text::CCursorRange::two(egui::text::CCursor::default(), egui::text::CCursor::new(text.chars().count()))));
    state.store(ctx, id);
}

#[cfg(test)]
mod tests {
    use egui::accesskit::Role;
    use egui_kittest::{Harness, kittest::Queryable};
    use serde_json::{Value, json};

    use crate::PhotocraftApp;
    use crate::state::{DialogKind, UiState};

    fn harness(app: PhotocraftApp) -> Harness<'static, PhotocraftApp> {
        let h = Harness::builder().with_size(egui::vec2(1400.0, 900.0)).build_ui_state(|ui, app| crate::dialogs::show(app, ui.ctx()), app);
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
        h
    }

    fn fields(h: &Harness<'static, PhotocraftApp>) -> serde_json::Map<String, Value> {
        h.state().ui.dialogs.first().map(|d| d.fields.clone()).expect("the dialog is open")
    }

    fn num(h: &Harness<'static, PhotocraftApp>, key: &str) -> f64 {
        fields(h).get(key).and_then(Value::as_f64).unwrap_or(f64::NAN)
    }

    fn tab(h: &mut Harness<'static, PhotocraftApp>) {
        h.key_press(egui::Key::Tab);
        h.run_steps(2);
    }

    fn arrow(h: &mut Harness<'static, PhotocraftApp>, key: egui::Key, m: egui::Modifiers) {
        h.key_press_modifiers(m, key);
        h.run_steps(2);
    }

    fn type_text(h: &mut Harness<'static, PhotocraftApp>, text: &str) {
        for c in text.chars() {
            h.event(egui::Event::Text(c.to_string()));
            h.run_steps(1);
        }
    }

    /// Hue/Saturation: Tab goes Hue → Saturation → Lightness (never the dropdown or the
    /// sliders), ⇧Tab back, and ↑ / ↓ step the focused field by 1 (⇧: 10).
    #[test]
    fn hue_saturation_tabs_through_its_fields_and_arrows_step_them() {
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
        let mut h = harness(PhotocraftApp::new(s, crate::Services::default()));
        crate::adjust_dialog::open(h.state_mut(), "image.adjustments.hueSaturation").unwrap();
        h.run_steps(3);
        assert_eq!(h.ctx.memory(|m| m.focused()), None, "nothing is focused to begin with");
        tab(&mut h);
        assert!(h.query_all_by_role(Role::SpinButton).next().is_some_and(|n| n.is_focused()), "the first Tab lands on Hue");
        arrow(&mut h, egui::Key::ArrowUp, egui::Modifiers::NONE);
        assert_eq!(num(&h, "hue"), 1.0);
        arrow(&mut h, egui::Key::ArrowUp, egui::Modifiers::SHIFT);
        assert_eq!(num(&h, "hue"), 11.0);
        arrow(&mut h, egui::Key::ArrowDown, egui::Modifiers::NONE);
        assert_eq!(num(&h, "hue"), 10.0);
        // The text stays selected: typing replaces it.
        type_text(&mut h, "-5");
        assert_eq!(num(&h, "hue"), -5.0);
        tab(&mut h);
        arrow(&mut h, egui::Key::ArrowUp, egui::Modifiers::NONE);
        assert_eq!((num(&h, "hue"), num(&h, "saturation")), (-5.0, 1.0), "the second Tab is Saturation");
        tab(&mut h);
        arrow(&mut h, egui::Key::ArrowDown, egui::Modifiers::SHIFT);
        assert_eq!(num(&h, "lightness"), -10.0, "the third is Lightness");
        tab(&mut h);
        arrow(&mut h, egui::Key::ArrowUp, egui::Modifiers::NONE);
        assert_eq!(num(&h, "hue"), -4.0, "Tab wraps around to Hue");
        h.key_press_modifiers(egui::Modifiers::SHIFT, egui::Key::Tab);
        h.run_steps(2);
        arrow(&mut h, egui::Key::ArrowUp, egui::Modifiers::NONE);
        assert_eq!(num(&h, "lightness"), -9.0, "⇧Tab goes back to Lightness");
        // The range clamps.
        for _ in 0..12 {
            arrow(&mut h, egui::Key::ArrowDown, egui::Modifiers::SHIFT);
        }
        assert_eq!(num(&h, "lightness"), -100.0);
        assert!(h.state().ui.dialogs.len() == 1, "Tab never closed the dialog");
    }

    /// New Document opens with the name selected; Tab goes Width → Height → Resolution and wraps.
    #[test]
    fn new_document_opens_on_the_name_and_tabs_to_width_then_height() {
        let mut h = harness(PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default()));
        h.state_mut().ui.open_dialog(DialogKind::NewDocument, UiState::new_document_fields());
        h.run_steps(3);
        assert!(h.get_by_role(Role::TextInput).is_focused(), "the name field has focus");
        type_text(&mut h, "Poster");
        assert_eq!(fields(&h)["name"], "Poster", "typing replaced the selected name");
        tab(&mut h);
        type_text(&mut h, "512");
        tab(&mut h);
        type_text(&mut h, "300");
        tab(&mut h);
        type_text(&mut h, "150");
        let f = fields(&h);
        assert_eq!((f["width"].as_u64(), f["height"].as_u64(), f["resolution"].as_f64()), (Some(512), Some(300), Some(150.0)), "{f:?}");
        // Arrows step a size field too.
        h.key_press_modifiers(egui::Modifiers::SHIFT, egui::Key::Tab);
        h.run_steps(2);
        arrow(&mut h, egui::Key::ArrowUp, egui::Modifiers::SHIFT);
        assert_eq!(fields(&h)["height"].as_u64(), Some(310));
        // Tab from the last field wraps to the name.
        tab(&mut h);
        tab(&mut h);
        assert!(h.get_by_role(Role::TextInput).is_focused(), "Tab wrapped to the name");
        type_text(&mut h, "Flyer");
        assert_eq!(fields(&h)["name"], "Flyer");
        h.key_press(egui::Key::Enter);
        h.run_steps(3);
        let d = &h.state().session.active().expect("a new document").doc;
        assert_eq!((d.name.as_str(), d.size.width, d.size.height), ("Flyer", 512, 310));
    }

    /// Tab is the dialog's: the modal underneath never sees it, and a dialog with no fields swallows it.
    #[test]
    fn tab_in_a_dialog_without_fields_does_nothing() {
        let mut h = harness(PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default()));
        h.state_mut().ui.open_dialog(DialogKind::About, Default::default());
        h.run_steps(3);
        tab(&mut h);
        assert_eq!(h.ctx.memory(|m| m.focused()), None);
        assert_eq!(h.state().ui.dialogs.len(), 1);
    }
}
