//! Filter › Other › Color to Alpha (`filter.other.colorToAlpha`, #1576): turns a chosen colour
//! into transparency, unmixing it from partly covered pixels so antialiased edges keep their
//! shape over any new background (the algorithm is `photocraft_algo::color_to_alpha`).
//!
//! Photoshop has no such filter (GIMP and Krita do), so the id is our own; the menu item sits in
//! Filter › Other after Offset. It runs through [`crate::filters::run_filter`] like every other
//! filter (selection, smart filters, Last Filter, one history step), which turns a Background
//! layer into a normal layer ("Layer 0") first, as the erasers do, since the Background can't
//! hold transparency. A layer whose transparency is locked is refused rather than silently
//! getting its old alpha back.

use photocraft_algo::FilterParams;
use photocraft_color::ColorMode;
use serde_json::Value;

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

/// The command id.
pub const ID: &str = "filter.other.colorToAlpha";

const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

fn bad(msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: ID.into(), msg: msg.into() }
}

/// A colour channel value: finite and clamped to 0–1 (a fallback for already validated or
/// recorded params, e.g. a smart filter read back from a file).
fn unit(v: f32, default: f32) -> f32 {
    if v.is_finite() { v.clamp(0.0, 1.0) } else { default }
}

/// Algorithm parameters (lenient: out-of-range values are clamped, unreadable ones fall back to
/// the defaults). [`run`] validates strictly before it gets here.
pub(crate) fn params(p: &Value) -> FilterParams {
    let c = crate::commands::color_param(p, "color", WHITE);
    let num = |k: &str, d: f32| p.get(k).and_then(Value::as_f64).map_or(d, |v| unit(v as f32, d));
    FilterParams::ColorToAlpha {
        color: [unit(c[0], 1.0), unit(c[1], 1.0), unit(c[2], 1.0), 1.0],
        transparency_threshold: num("transparencyThreshold", 0.0),
        opacity_threshold: num("opacityThreshold", 1.0),
    }
}

fn threshold(p: &Value, key: &str) -> Result<()> {
    match p.get(key) {
        None => Ok(()),
        Some(v) => match v.as_f64() {
            Some(x) if x.is_finite() && (0.0..=1.0).contains(&x) => Ok(()),
            Some(_) => Err(bad(format!("`{key}` must be in 0..1"))),
            None => Err(bad(format!("`{key}` must be a number in 0..1"))),
        },
    }
}

/// Rejects params the lenient [`params`] would otherwise quietly replace.
fn validate(p: &Value) -> Result<()> {
    if !p.is_object() {
        return Err(bad("params must be a JSON object"));
    }
    match p.get("color") {
        None => {}
        Some(Value::String(s)) => {
            crate::commands::parse_hex(s).ok_or_else(|| bad(format!("`color` must be \"#rrggbb\", not {s:?}")))?;
        }
        Some(Value::Array(a)) if (3..=4).contains(&a.len())
            && a.iter().all(|v| v.as_f64().is_some_and(|x| x.is_finite() && (0.0..=1.0).contains(&x))) => {}
        Some(Value::Array(a)) if (3..=4).contains(&a.len()) =>
            return Err(bad("`color` as an array must hold 3 or 4 numbers in 0..1")),
        Some(_) => return Err(bad("`color` must be \"#rrggbb\" or [r, g, b] in 0..1")),
    }
    threshold(p, "transparencyThreshold")?;
    threshold(p, "opacityThreshold")
}

/// Menu state: a pixel layer (or smart object) in an RGB or Grayscale document whose pixels and
/// transparency may change. The Background counts: it becomes a normal layer when run.
fn enabled(s: &Session) -> std::result::Result<(), String> {
    if crate::channel_cmds::edits_channel(s) {
        return Err("Color to Alpha works on layer pixels, not on a channel or Quick Mask".into());
    }
    crate::filters::has_filterable_layer(s)?;
    let d = s.active().ok_or("no document open")?;
    match d.doc.mode {
        ColorMode::Rgb | ColorMode::Grayscale => {}
        m => return Err(format!("Color to Alpha needs an RGB or Grayscale document (this one is {m:?})")),
    }
    let l = crate::active_layer_of(s)?;
    let locks = d.doc.effective_locks(l.id);
    if locks.pixels || locks.all {
        return Err(format!("the layer \"{}\" is locked", l.name));
    }
    if locks.transparency && !d.doc.is_background(l.id) {
        return Err(format!("the transparency of layer \"{}\" is locked", l.name));
    }
    Ok(())
}

fn run(s: &mut Session, p: &Value) -> Result<Value> {
    validate(p)?;
    crate::filters::run_filter(s, ID, p)
}

/// The command spec.
pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: ID,
        label: "Color to Alpha…",
        menu: &["Filter", "Other"],
        shortcut: None,
        params: r##"{"color":color=#ffffff,"transparencyThreshold":0..1=0,"opacityThreshold":0..1=1}"##,
        enabled,
        run,
        journal: true,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_geom::Rect;
    use serde_json::json;

    const W: i32 = 24;
    const H: i32 = 16;

    /// A document with a white Background and, when `layer`, a pixel layer on top; both filled
    /// with a black-to-white ramp (column 0 black, last column white).
    fn session(depth: u32, mode: &str, layer: bool) -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": W, "height": H, "depth": depth, "mode": mode})).unwrap();
        if layer {
            s.execute("layer.new.layer", json!({})).unwrap();
        }
        s.edit("ramp", |doc, active| {
            let surf = doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap();
            for x in 0..W {
                let v = x as f32 / (W - 1) as f32;
                surf.fill_rect(Rect::new(x, 0, x + 1, H), &photocraft_raster::from_rgba(&surf.format(), [v, v, v, 1.0]));
            }
            Ok(())
        })
        .unwrap();
        s
    }

    fn rgba(s: &Session, x: i32, y: i32) -> [f32; 4] {
        let d = s.active().unwrap();
        d.doc.layer(d.active_layer.unwrap()).unwrap().surface().unwrap().rgba(x, y)
    }

    fn pixels(s: &Session) -> Vec<f32> {
        let d = s.active().unwrap();
        d.doc.layer(d.active_layer.unwrap()).unwrap().surface().unwrap().read_region(Rect::new(0, 0, W, H))
    }

    #[test]
    fn removes_white_at_every_depth_and_undoes_in_one_step() {
        for (depth, mode) in [(8, "rgb"), (16, "rgb"), (32, "rgb"), (8, "gray"), (16, "gray"), (32, "gray")] {
            let mut s = session(depth, mode, true);
            let before = pixels(&s);
            let steps = s.active().unwrap().history.past_len();
            let r = s.execute(ID, json!({})).unwrap_or_else(|e| panic!("{depth} {mode}: {e}"));
            assert_eq!(r["filter"]["filter"], "colorToAlpha", "{r}");
            assert_eq!(s.active().unwrap().history.past_len(), steps + 1);
            assert_eq!(s.active().unwrap().history.undo_label(), Some("Color to Alpha"));
            let tol = if depth == 8 { 3.0 / 255.0 } else { 2e-3 };
            assert!(rgba(&s, W - 1, 3)[3] <= tol, "{depth} {mode}: white is transparent");
            assert_eq!(rgba(&s, 0, 3), [0.0, 0.0, 0.0, 1.0], "{depth} {mode}: black is kept");
            // A grey becomes black with the grey's darkness as opacity.
            let x = W / 2;
            let v = x as f32 / (W - 1) as f32;
            let p = rgba(&s, x, 3);
            assert!((p[3] - (1.0 - v)).abs() <= tol && p[0] <= tol, "{depth} {mode}: {p:?}");
            s.execute("edit.undo", json!({})).unwrap();
            assert_eq!(pixels(&s), before, "{depth} {mode}: undo");
        }
    }

    #[test]
    fn thresholds_and_colour_params_apply() {
        let mut s = session(16, "rgb", true);
        s.execute(ID, json!({"color": "#000000", "transparencyThreshold": 0.25, "opacityThreshold": 0.75})).unwrap();
        // Black key: dark greys (opacity ≤ 0.25) vanish, light ones (≥ 0.75) stay as they were.
        assert_eq!(rgba(&s, 2, 0)[3], 0.0);
        assert_eq!(rgba(&s, W - 2, 0)[3], 1.0);
        let mid = rgba(&s, W / 2, 0);
        assert!(mid[3] > 0.0 && mid[3] < 1.0, "{mid:?}");
        // The array form of the colour works too.
        let mut t = session(8, "rgb", true);
        t.execute(ID, json!({"color": [1.0, 1.0, 1.0]})).unwrap();
        assert!(rgba(&t, W - 1, 0)[3] < 0.02);
    }

    #[test]
    fn respects_the_selection() {
        let mut s = session(8, "rgb", true);
        s.execute("select.rect", json!({"x": W / 2, "y": 0, "width": W / 2, "height": H / 2})).unwrap();
        s.execute(ID, json!({})).unwrap();
        assert!(rgba(&s, W - 1, 0)[3] < 0.02, "inside the selection");
        assert_eq!(rgba(&s, W - 1, H - 1)[3], 1.0, "outside the selection");
    }

    #[test]
    fn the_background_becomes_a_normal_layer_and_undo_restores_it() {
        let mut s = session(8, "rgb", false);
        assert!(s.is_enabled(ID));
        s.execute(ID, json!({})).unwrap();
        let name = |s: &Session| {
            let d = s.active().unwrap();
            d.doc.layer(d.active_layer.unwrap()).unwrap().name.clone()
        };
        assert_eq!(name(&s), "Layer 0");
        assert!(rgba(&s, W - 1, 0)[3] < 0.02, "the Background really became transparent");
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(name(&s), "Background");
        let d = s.active().unwrap();
        assert!(d.doc.is_background(d.active_layer.unwrap()));
    }

    #[test]
    fn locked_layers_and_unsupported_documents_are_refused() {
        let mut s = session(8, "rgb", true);
        s.execute("layer.lockLayers", json!({"transparency": true})).unwrap();
        assert!(!s.is_enabled(ID));
        let before = pixels(&s);
        assert!(s.execute(ID, json!({})).is_err());
        assert_eq!(pixels(&s), before);
        // Even when the menu check is bypassed (Last Filter), the filter refuses the lock.
        assert!(crate::filters::run_filter(&mut s, ID, &json!({})).is_err());
        assert_eq!(pixels(&s), before);

        let mut s = session(8, "rgb", true);
        s.execute("layer.lockLayers", json!({"pixels": true})).unwrap();
        assert!(!s.is_enabled(ID));

        for mode in ["cmyk", "lab"] {
            let mut s = Session::new();
            s.execute("file.new", json!({"width": 8, "height": 8, "mode": mode})).unwrap();
            s.execute("layer.new.layer", json!({})).unwrap();
            assert!(!s.is_enabled(ID), "{mode}");
        }
        let mut s = Session::new();
        assert!(!s.is_enabled(ID));
        s.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
        s.execute("layer.newAdjustmentLayer.invert", json!({})).unwrap();
        assert!(!s.is_enabled(ID));
    }

    #[test]
    fn bad_params_are_rejected_without_changes() {
        let mut s = session(8, "rgb", true);
        let before = pixels(&s);
        for p in [
            json!(null),
            json!(7),
            json!("white"),
            json!({"color": "white"}),
            json!({"color": "#12"}),
            json!({"color": "#zzzzzz"}),
            json!({"color": [1.0, 1.0]}),
            json!({"color": [1.0, 2.0, 0.0]}),
            json!({"color": [1.0, "a", 0.0]}),
            json!({"color": 16777215}),
            json!({"transparencyThreshold": -0.1}),
            json!({"opacityThreshold": 1.5}),
            json!({"opacityThreshold": "1"}),
            json!({"transparencyThreshold": 1e308}),
        ] {
            assert!(matches!(s.execute(ID, p.clone()), Err(EngineError::BadParams { .. })), "{p}");
        }
        assert_eq!(pixels(&s), before);
    }

    #[test]
    fn smart_objects_get_a_smart_filter_and_last_filter_repeats() {
        let mut s = session(8, "rgb", true);
        s.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap();
        s.execute(ID, json!({"color": "#ffffff"})).unwrap();
        let d = s.active().unwrap();
        let photocraft_doc::LayerContent::Smart(sm) = &d.doc.layer(d.active_layer.unwrap()).unwrap().content else { panic!("not a smart object") };
        assert_eq!(sm.smart_filters.len(), 1);
        assert_eq!(sm.smart_filters[0].command, ID);
        assert!(sm.cache.as_ref().unwrap().rgba(W - 1, 0)[3] < 0.02, "the cache was re-rendered");

        let mut t = session(8, "rgb", true);
        t.execute(ID, json!({"opacityThreshold": 0.5})).unwrap();
        let once = pixels(&t);
        t.execute("filter.lastFilter", json!({})).unwrap();
        assert_ne!(pixels(&t), once, "Last Filter ran Color to Alpha again");
    }

    #[test]
    fn lenient_mapping_clamps_recorded_params() {
        assert_eq!(
            params(&json!({"color": [2.0, -1.0, 0.5], "transparencyThreshold": -3, "opacityThreshold": 9})),
            FilterParams::ColorToAlpha { color: [1.0, 0.0, 0.5, 1.0], transparency_threshold: 0.0, opacity_threshold: 1.0 }
        );
        assert_eq!(params(&Value::Null), FilterParams::ColorToAlpha { color: WHITE, transparency_threshold: 0.0, opacity_threshold: 1.0 });
    }
}
