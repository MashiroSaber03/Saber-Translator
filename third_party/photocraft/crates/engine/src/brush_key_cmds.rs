//! Photoshop's brush keys (#626): `[` / `]` step the brush size and ⇧[ / ⇧] the hardness. Photoshop
//! has no menu item for them but lists them in Edit › Keyboard Shortcuts › Tools ("Decrease Brush
//! Size"…); here they are ordinary commands with default shortcuts, so they can be rebound,
//! recorded and driven by agents. Each one is a `tools.setBrush` on the session brush.
//!
//! `tools.rotateBrush` turns the tip: Photoshop's ← / → while painting (1°, ⇧ 15°). Arrows mean
//! other things with other tools, so the shell gives them to it only for the painting tools
//! (`move_mods::arrow_keys`).

use serde_json::{Value, json};

use crate::commands::{CommandSpec, always};
use crate::{EngineError, Result, Session};

/// Photoshop's largest brush diameter, in pixels.
pub const MAX_SIZE: f32 = 5000.0;

/// `[` / `]`: 25 % smaller or larger, and at least a pixel either way so small brushes still grow
/// and shrink (2 / 1.25 rounds back to 2, which kept `[` from reaching 1 px, #3229).
fn size_step(size: f32, up: bool) -> f32 {
    let next = if up { (size * 1.25).round().max(size + 1.0) } else { (size / 1.25).round().min(size - 1.0) };
    next.clamp(1.0, MAX_SIZE)
}

/// ⇧[ / ⇧]: to the previous or next quarter (0, 25, 50, 75, 100 %).
fn hardness_step(hardness: f32, up: bool) -> f32 {
    let quarter = (hardness * 4.0).round();
    let next = if up { quarter + 1.0 } else { quarter - 1.0 };
    next.clamp(0.0, 4.0) / 4.0
}

fn size(s: &mut Session, up: bool) -> Result<Value> {
    let size = size_step(s.tools.brush.size, up);
    crate::brush_cmds::set_brush(s, &json!({ "brush": { "size": size } }))
}

fn hardness(s: &mut Session, up: bool) -> Result<Value> {
    let hardness = hardness_step(s.tools.brush.hardness, up);
    crate::brush_cmds::set_brush(s, &json!({ "brush": { "hardness": hardness } }))
}

/// The brush angle turned by `degrees` (positive is anticlockwise, as the angle field counts),
/// wrapped into Photoshop's -180..=180.
fn turned(angle: f32, degrees: f32) -> f32 {
    let a = (angle + degrees).rem_euclid(360.0);
    if a > 180.0 { a - 360.0 } else { a }
}

fn rotate(s: &mut Session, p: &Value) -> Result<Value> {
    let by = p
        .get("degrees")
        .and_then(Value::as_f64)
        .filter(|d| d.is_finite() && d.abs() <= 360.0)
        .ok_or_else(|| EngineError::BadParams { cmd: "tools.rotateBrush".into(), msg: "`degrees` must be a number from -360 to 360".into() })?;
    let angle = turned(s.tools.brush.angle, by as f32);
    crate::brush_cmds::set_brush(s, &json!({ "brush": { "angle": angle } }))
}

macro_rules! key_cmd {
    ($id:literal, $label:literal, $sc:literal, $run:expr) => {
        CommandSpec { id: $id, label: $label, menu: &[], shortcut: Some($sc), params: "{}", enabled: always, run: $run, journal: true }
    };
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        key_cmd!("tools.decreaseBrushSize", "Decrease Brush Size", "[", |s, _| size(s, false)),
        key_cmd!("tools.increaseBrushSize", "Increase Brush Size", "]", |s, _| size(s, true)),
        key_cmd!("tools.decreaseBrushHardness", "Decrease Brush Hardness", "Shift+[", |s, _| hardness(s, false)),
        key_cmd!("tools.increaseBrushHardness", "Increase Brush Hardness", "Shift+]", |s, _| hardness(s, true)),
        CommandSpec {
            id: "tools.rotateBrush",
            label: "Rotate Brush Tip",
            menu: &[],
            shortcut: None,
            params: r##"{"degrees":-360..360 (positive anticlockwise; ← / → while painting send 1, ⇧ 15)}"##,
            enabled: always,
            run: rotate,
            journal: true,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_like_photoshop() {
        assert_eq!(size_step(1.0, false), 1.0, "never below 1 px");
        assert_eq!(size_step(1.0, true), 2.0, "small brushes still grow");
        assert_eq!(size_step(2.0, false), 1.0, "small brushes still shrink (#3229)");
        assert_eq!(size_step(3.0, false), 2.0);
        assert_eq!(size_step(4.0, false), 3.0);
        for size in [2.0, 3.0, 5.0, 8.0, 13.0, 100.0] {
            assert!(size_step(size, false) < size, "`[` always shrinks {size}");
            assert!(size_step(size, true) > size, "`]` always grows {size}");
        }
        assert_eq!(size_step(100.0, true), 125.0);
        assert_eq!(size_step(125.0, false), 100.0);
        assert_eq!(size_step(MAX_SIZE, true), MAX_SIZE, "never past the largest brush");
        assert_eq!(hardness_step(0.5, true), 0.75);
        assert_eq!(hardness_step(0.6, false), 0.25, "snaps to the quarter below the nearest");
        assert_eq!(hardness_step(1.0, true), 1.0);
        assert_eq!(hardness_step(0.0, false), 0.0);
    }

    #[test]
    fn rotate_brush_turns_the_tip_and_wraps_at_180() {
        assert_eq!(turned(0.0, 1.0), 1.0);
        assert_eq!(turned(179.0, 15.0), -166.0);
        assert_eq!(turned(-175.0, -15.0), 170.0);
        assert_eq!(turned(180.0, 0.0), 180.0);
        let mut s = Session::new();
        s.execute("tools.rotateBrush", json!({"degrees": -15})).unwrap();
        assert_eq!(s.tools.brush.angle, -15.0);
        for bad in [json!({}), json!({"degrees": "x"}), json!({"degrees": 1e9}), json!(null)] {
            assert!(s.execute("tools.rotateBrush", bad.clone()).is_err(), "{bad}");
        }
        assert_eq!(s.tools.brush.angle, -15.0);
    }

    #[test]
    fn commands_step_the_session_brush_and_journal() {
        let mut s = Session::new();
        s.execute("tools.setBrush", json!({"brush": {"size": 40, "hardness": 0.5}})).unwrap();
        s.execute("tools.increaseBrushSize", json!({})).unwrap();
        assert_eq!(s.tools.brush.size, 50.0);
        s.execute("tools.decreaseBrushSize", Value::Null).unwrap();
        assert_eq!(s.tools.brush.size, 40.0);
        // Pixel artists: `[` held down reaches 1 px (#3229).
        s.execute("tools.setBrush", json!({"brush": {"size": 3}})).unwrap();
        for expected in [2.0, 1.0, 1.0] {
            s.execute("tools.decreaseBrushSize", Value::Null).unwrap();
            assert_eq!(s.tools.brush.size, expected);
        }
        s.execute("tools.increaseBrushHardness", json!({})).unwrap();
        assert_eq!(s.tools.brush.hardness, 0.75);
        s.execute("tools.decreaseBrushHardness", json!({"junk": [1, 2]})).unwrap();
        assert_eq!(s.tools.brush.hardness, 0.5);
        assert_eq!(s.journal.last().map(|(id, _)| id.as_str()), Some("tools.decreaseBrushHardness"));
        // They need no document and are bindable under Tools.
        let sc = |id| crate::commands::find(id).and_then(|c| c.shortcut);
        assert_eq!(sc("tools.decreaseBrushSize"), Some("["));
        assert_eq!(sc("tools.increaseBrushHardness"), Some("Shift+]"));
    }
}
