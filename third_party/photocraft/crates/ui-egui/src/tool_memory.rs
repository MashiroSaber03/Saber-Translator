//! Tool options remembered between launches (#2814), as Photoshop keeps each tool's options-bar
//! settings across sessions.
//!
//! `Preferences::tool_options` holds the [`ToolOptions`] fields that differ from their defaults,
//! so a default changed in a later release still reaches users who never touched that option.
//! Saved once the pointer is up (not on every slider step), restored at launch. A hand-edited or
//! old file can't break the options: an unknown key is ignored and a field that doesn't read back
//! (wrong type, a number too large for its field) keeps its default.

use serde_json::{Map, Value};

use crate::PhotocraftApp;
use crate::state::ToolOptions;

/// The options that differ from the defaults, as JSON (`Null` when none do).
fn changed(o: &ToolOptions) -> Value {
    let (Ok(Value::Object(now)), Ok(Value::Object(def))) = (serde_json::to_value(o), serde_json::to_value(ToolOptions::default())) else {
        return Value::Null;
    };
    let diff: Map<String, Value> = now.into_iter().filter(|(k, v)| def.get(k) != Some(v)).collect();
    if diff.is_empty() { Value::Null } else { Value::Object(diff) }
}

/// Remember the tool options in the preferences once the user lets go of the pointer. Cheap
/// when nothing changed: one struct compare per frame. Run every frame, so options set by the
/// control channel or a tool preset are remembered too.
pub fn persist(app: &mut PhotocraftApp, ctx: &egui::Context) {
    if ctx.input(|i| i.pointer.any_down()) || app.prefs_rt.tool_options.as_ref() == Some(&app.ui.tool_options) {
        return;
    }
    app.prefs_rt.tool_options = Some(app.ui.tool_options.clone());
    let now = changed(&app.ui.tool_options);
    if app.session.prefs().tool_options != now {
        app.session.prefs.edit(|p| p.tool_options = now);
    }
}

/// Restore the remembered tool options at launch, field by field over the current ones.
pub fn restore(app: &mut PhotocraftApp) {
    let saved = app.session.prefs().tool_options.clone();
    if let Some(saved) = saved.as_object()
        && let Ok(Value::Object(mut cur)) = serde_json::to_value(&app.ui.tool_options)
    {
        for (k, v) in saved {
            let Some(old) = cur.get(k).cloned() else { continue };
            cur.insert(k.clone(), v.clone());
            if !reads_back(&cur, k) {
                log::warn!("Preferences: ignoring the remembered tool option `{k}`: {v}");
                cur.insert(k.clone(), old);
            }
        }
        if let Ok(o) = serde_json::from_value(Value::Object(cur)) {
            app.ui.tool_options = o;
        }
    } else if !saved.is_null() {
        log::warn!("Preferences: the remembered tool options are not an object; using the defaults");
    }
    app.prefs_rt.tool_options = Some(app.ui.tool_options.clone());
}

/// Does `opts` deserialize, with field `k` coming back as a value? A number out of an `f32`'s
/// range reads as infinity, which writes back as `null`.
fn reads_back(opts: &Map<String, Value>, k: &str) -> bool {
    let Ok(o) = serde_json::from_value::<ToolOptions>(Value::Object(opts.clone())) else { return false };
    let Ok(Value::Object(back)) = serde_json::to_value(&o) else { return false };
    back.get(k).is_some_and(|v| nulls(v) <= opts.get(k).map_or(0, nulls))
}

fn nulls(v: &Value) -> usize {
    match v {
        Value::Null => 1,
        Value::Array(a) => a.iter().map(nulls).sum(),
        Value::Object(o) => o.values().map(nulls).sum(),
        _ => 0,
    }
}
