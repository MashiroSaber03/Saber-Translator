//! Alt+Cancel → Reset for command dialogs that opt in.
//!
//! Holding Alt in the dialog chrome relabels Cancel as Reset and, on click, restores the dialog's
//! fields without closing it (Photoshop). Each marker implements its own `reset`; this module only
//! dispatches. Esc stays Cancel (`Modifiers::NONE` in the frame). Opt out with `"__altReset": false`.

use serde_json::{Map, Value};

use crate::PhotocraftApp;

/// Whether holding Alt turns Cancel into Reset for this dialog's fields.
pub fn supported(fields: &Map<String, Value>) -> bool {
    if fields.get("__altReset") == Some(&Value::Bool(false)) {
        return false;
    }
    match_arm(fields).is_some()
}

/// Apply Reset in place. No-op when the dialog has not opted in.
pub fn apply(app: &mut PhotocraftApp, ctx: &egui::Context, fields: &mut Map<String, Value>) {
    if fields.get("__altReset") == Some(&Value::Bool(false)) {
        return;
    }
    // Dispatch order matches `dialogs::show` so `__adjust` wins over `__filter`.
    if crate::guide_layout_ui::owns(fields) {
        crate::guide_layout_ui::reset(app, fields);
    } else if crate::fill_ui::owns(fields) {
        crate::fill_ui::reset(app, fields);
    } else if crate::variables_ui::owns(fields) {
        crate::variables_ui::reset(fields);
    } else if crate::color_picker_ui::owns(fields) {
        crate::color_picker_ui::reset(fields);
    } else if fields.contains_key("__sizing") {
        crate::sizing::reset(fields);
    } else if crate::adjust_dialog::owns(fields) {
        crate::adjust_dialog::reset(app, ctx, fields);
    } else if fields.contains_key("__filter") {
        crate::filter_dialog::reset_fields(app, fields);
    } else if fields.contains_key("__form") {
        crate::view_cmds::reset_form(fields);
    }
}

/// First matching opt-in arm, if any (same order as [`apply`]).
fn match_arm(fields: &Map<String, Value>) -> Option<&'static str> {
    if crate::guide_layout_ui::owns(fields) {
        Some("guideLayout")
    } else if crate::fill_ui::owns(fields) {
        Some("fill")
    } else if crate::variables_ui::owns(fields) && fields.contains_key("__variablesOrig") {
        Some("variables")
    } else if crate::color_picker_ui::owns(fields) {
        Some("colorPicker")
    } else if fields.contains_key("__sizing") {
        Some("sizing")
    } else if crate::adjust_dialog::owns(fields) {
        Some("adjust")
    } else if fields.contains_key("__filter") {
        Some("filter")
    } else if fields.contains_key("__form") && fields.contains_key("__formOrig") {
        Some("form")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn alt_reset_false_opts_out() {
        let mut f = Map::new();
        f.insert("__adjust".into(), json!("levels"));
        f.insert("__altReset".into(), json!(false));
        assert!(!supported(&f));
    }

    #[test]
    fn adjust_marker_opts_in() {
        let mut f = Map::new();
        f.insert("__adjust".into(), json!("levels"));
        assert!(supported(&f));
    }

    #[test]
    fn bare_confirm_does_not_opt_in() {
        let mut f = Map::new();
        f.insert("__command".into(), json!("edit.purge"));
        assert!(!supported(&f));
    }
}
