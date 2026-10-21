//! Editing a Solid Color Fill is an undoable content edit, independent of its masks.

use photocraft_color::{Color, ColorMode};
use photocraft_doc::{Fill, LayerContent};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, check_pixels_unlocked, layer_param};
use crate::{EngineError, Result, Session};

pub const SET: &str = "layer.fill.solidColor.set";

fn bad(msg: &str) -> EngineError {
    EngineError::BadParams { cmd: SET.into(), msg: msg.into() }
}

/// Parse an RGB picker value or a full colour in its stored model. A three-component RGB
/// value preserves alpha; opening the picker must not make a translucent fill opaque.
pub fn color_param(value: &Value, original: Color, mode: ColorMode) -> Result<Color> {
    let color = match value {
        Value::String(s) => {
            let h = s.strip_prefix('#').unwrap_or(s);
            let channel = |i: usize| h.get(i..i + 2).and_then(|v| u8::from_str_radix(v, 16).ok()).map(|v| f32::from(v) / 255.0);
            if h.len() != 6 && h.len() != 8 {
                return Err(bad("color must be #rrggbb or #rrggbbaa"));
            }
            let (Some(r), Some(g), Some(b)) = (channel(0), channel(2), channel(4)) else { return Err(bad("invalid hexadecimal color")) };
            let alpha = if h.len() == 8 { channel(6).ok_or_else(|| bad("invalid alpha"))? } else { original.alpha };
            Color::rgba(r, g, b, alpha).in_mode(mode)
        }
        Value::Array(a) if (3..=4).contains(&a.len()) => {
            let channel = |i: usize| a.get(i).and_then(Value::as_f64).filter(|v| v.is_finite() && (0.0..=1.0).contains(v)).map(|v| v as f32);
            let (Some(r), Some(g), Some(b)) = (channel(0), channel(1), channel(2)) else {
                return Err(bad("RGB components must be finite numbers within 0..1"));
            };
            let alpha = if a.len() == 4 { channel(3).ok_or_else(|| bad("alpha must be a finite number within 0..1"))? } else { original.alpha };
            Color::rgba(r, g, b, alpha).in_mode(mode)
        }
        Value::Object(_) => serde_json::from_value::<Color>(value.clone()).map_err(|_| bad("invalid stored Color"))?,
        _ => return Err(bad("color must be hexadecimal RGB, [r,g,b,a?], or a stored Color")),
    };
    if !color.c.iter().all(|v| v.is_finite()) || !color.alpha.is_finite() || !(0.0..=1.0).contains(&color.alpha) {
        return Err(bad("color must be finite and alpha within 0..1"));
    }
    Ok(color)
}

fn set(s: &mut Session, p: &Value) -> Result<Value> {
    if !p.is_object() || p.get("layer").is_some_and(|v| v.as_u64().is_none()) {
        return Err(bad("params must be an object and layer an unsigned id"));
    }
    let st = s.active().ok_or(EngineError::NoDocument)?;
    if let Some(document) = p.get("document")
        && document.as_u64() != Some(st.doc.id.0)
    {
        return Err(bad("the target document is no longer active"));
    }
    let id = layer_param(s, p)?;
    let layer = st.doc.layer(id).ok_or(EngineError::NoLayer(id))?;
    let LayerContent::Fill(Fill::Solid(original)) = layer.content else { return Err(bad("the target must be a Solid Color Fill layer")) };
    let color = color_param(p.get("color").ok_or_else(|| bad("color is required"))?, original, st.doc.mode)?;
    if color != original {
        check_pixels_unlocked(&st.doc, id)?;
        if st.doc.effective_locks(id).transparency && color.alpha != original.alpha {
            return Err(bad("the layer's transparency is locked"));
        }
        s.edit("Solid Color", |doc, _| {
            let layer = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
            layer.content = LayerContent::Fill(Fill::Solid(color));
            Ok(())
        })?;
    }
    Ok(json!({"layer": id.0, "color": color}))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: SET,
        label: "Solid Color…",
        menu: &[],
        shortcut: None,
        params: r##"{"layer":id? (default active),"document":id? (reject a stale dialog),"color":"#rrggbb"|"#rrggbbaa"|[r,g,b,a?]|{mode,c:[f32;4],alpha:f32}}; RGB values without alpha preserve it; stored Color preserves its model and precision"##,
        enabled: |s| s.active().map(|_| ()).ok_or_else(|| "no document open".into()),
        run: set,
        journal: true,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_preserves_layer_and_masks_at_all_depths_and_undo_redo() {
        for depth in [8, 16, 32] {
            let mut s = Session::new();
            s.execute("file.new", json!({"width": 32, "height": 24, "depth": depth})).unwrap();
            let id = s.execute("layer.newFillLayer.solidColor", json!({"color": "#12345680"})).unwrap()["layer"].as_u64().unwrap();
            s.execute("layer.layerMask.revealAll", json!({})).unwrap();
            s.execute("layer.vectorMask.revealAll", json!({})).unwrap();
            let before = s.active().unwrap().doc.clone();
            let count = s.active().unwrap().history.entries().len();
            let color = Color { mode: ColorMode::Lab, c: [0.512345, 0.48, 0.6, 0.0], alpha: 0.345678 };
            s.execute(SET, json!({"layer": id, "color": color})).unwrap();
            let mut expected = (*before).clone();
            expected.layer_mut(photocraft_doc::LayerId(id)).unwrap().content = LayerContent::Fill(Fill::Solid(color));
            assert_eq!(*s.active().unwrap().doc, expected);
            assert_eq!(s.active().unwrap().history.entries().len(), count + 1);
            s.execute("edit.undo", json!({})).unwrap();
            assert_eq!(*s.active().unwrap().doc, *before);
            s.execute("edit.redo", json!({})).unwrap();
            assert_eq!(s.active().unwrap().doc.layer(photocraft_doc::LayerId(id)).unwrap().content, LayerContent::Fill(Fill::Solid(color)));
            let count = s.active().unwrap().history.entries().len();
            s.execute(SET, json!({"color": color})).unwrap();
            assert_eq!(s.active().unwrap().history.entries().len(), count, "unchanged colour is not a history step");
            s.execute(SET, json!({"color": "#abcdef"})).unwrap();
            let LayerContent::Fill(Fill::Solid(changed)) = s.active().unwrap().doc.layer(photocraft_doc::LayerId(id)).unwrap().content else { panic!() };
            assert_eq!(changed.alpha, color.alpha);
        }
    }

    #[test]
    fn bad_params_wrong_layer_and_stale_document_leave_history_untouched() {
        let mut s = Session::new();
        assert!(s.execute(SET, json!({"color": "#ffffff"})).is_err());
        s.execute("file.new", json!({"width": 2, "height": 2})).unwrap();
        let pixels = s.execute("layer.new.layer", json!({})).unwrap()["layer"].as_u64().unwrap();
        s.execute("layer.newFillLayer.solidColor", json!({})).unwrap();
        let before = s.active().unwrap().doc.clone();
        let count = s.active().unwrap().history.entries().len();
        for p in [
            json!({}),
            json!(null),
            json!({"color": true}),
            json!({"color": [1, "bad", 0]}),
            json!({"color": [1, 0, 0, 2]}),
            json!({"color": "#💥💥"}),
            json!({"layer": "bad", "color": "#ffffff"}),
            json!({"layer": pixels, "color": "#ffffff"}),
            json!({"layer": u64::MAX, "color": "#ffffff"}),
            json!({"document": u64::MAX, "color": "#ffffff"}),
        ] {
            assert!(s.execute(SET, p.clone()).is_err(), "{p}");
            assert!(std::sync::Arc::ptr_eq(&before, &s.active().unwrap().doc));
            assert_eq!(s.active().unwrap().history.entries().len(), count);
        }
    }

    #[test]
    fn edits_respect_layer_and_transparency_locks() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 2, "height": 2})).unwrap();
        s.execute("layer.newFillLayer.solidColor", json!({"color": "#12345680"})).unwrap();
        for locks in [json!({"all": true}), json!({"all": false, "pixels": true}), json!({"pixels": false, "transparency": true})] {
            s.execute("layer.setProps", json!({"locks": locks})).unwrap();
            let before = s.active().unwrap().doc.clone();
            assert!(s.execute(SET, json!({"color": "#abcdef00"})).is_err());
            assert!(std::sync::Arc::ptr_eq(&before, &s.active().unwrap().doc));
        }
        s.execute(SET, json!({"color": "#abcdef"})).unwrap();
    }

    #[test]
    fn rgb_picker_inputs_follow_document_mode_and_preserve_alpha() {
        for (mode_command, mode) in [(None, ColorMode::Rgb), (Some("image.mode.cmyk"), ColorMode::Cmyk), (Some("image.mode.lab"), ColorMode::Lab)] {
            let mut s = Session::new();
            s.execute("file.new", json!({"width": 8, "height": 8, "depth": 16})).unwrap();
            if let Some(cmd) = mode_command {
                s.execute(cmd, json!({})).unwrap();
            }
            let id = s.execute("layer.newFillLayer.solidColor", json!({"color": "#12345680"})).unwrap()["layer"].as_u64().unwrap();
            let original = match s.active().unwrap().doc.layer(photocraft_doc::LayerId(id)).unwrap().content {
                LayerContent::Fill(Fill::Solid(c)) => c,
                _ => panic!(),
            };
            assert_eq!(s.active().unwrap().doc.mode, mode);
            let count = s.active().unwrap().history.entries().len();
            for (value, rgb) in [
                (json!("#336699"), Color::rgba(51.0 / 255.0, 102.0 / 255.0, 153.0 / 255.0, original.alpha)),
                (json!([0.25, 0.5, 0.75]), Color::rgba(0.25, 0.5, 0.75, original.alpha)),
            ] {
                let expected = rgb.in_mode(mode);
                let result = s.execute(SET, json!({"layer": id, "color": value})).unwrap();
                assert_eq!(serde_json::from_value::<Color>(result["color"].clone()).unwrap(), expected);
                let LayerContent::Fill(Fill::Solid(stored)) = s.active().unwrap().doc.layer(photocraft_doc::LayerId(id)).unwrap().content else { panic!() };
                assert_eq!(stored, expected);
                assert_eq!(stored.alpha, original.alpha);
            }
            assert_eq!(s.active().unwrap().history.entries().len(), count + 2);
            // Reapplying the same RGB array is a no-op, including after mode conversion.
            let before_noop = s.active().unwrap().doc.clone();
            s.execute(SET, json!({"layer": id, "color": [0.25, 0.5, 0.75]})).unwrap();
            assert!(std::sync::Arc::ptr_eq(&before_noop, &s.active().unwrap().doc));
            assert_eq!(s.active().unwrap().history.entries().len(), count + 2);
            let expected = Color::rgba(0.25, 0.5, 0.75, 0.25).in_mode(mode);
            s.execute(SET, json!({"layer": id, "color": [0.25, 0.5, 0.75, 0.25]})).unwrap();
            let LayerContent::Fill(Fill::Solid(stored)) = s.active().unwrap().doc.layer(photocraft_doc::LayerId(id)).unwrap().content else { panic!() };
            assert_eq!(stored, expected);
            s.execute("edit.undo", json!({})).unwrap();
            let LayerContent::Fill(Fill::Solid(stored)) = s.active().unwrap().doc.layer(photocraft_doc::LayerId(id)).unwrap().content else { panic!() };
            assert_eq!(stored, Color::rgba(0.25, 0.5, 0.75, original.alpha).in_mode(mode));
            s.execute("edit.redo", json!({})).unwrap();
            let LayerContent::Fill(Fill::Solid(stored)) = s.active().unwrap().doc.layer(photocraft_doc::LayerId(id)).unwrap().content else { panic!() };
            assert_eq!(stored, expected);
        }
    }

    #[test]
    fn stored_colour_remains_exact_without_rgb_conversion_or_extra_history() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 4, "height": 4})).unwrap();
        s.execute("image.mode.cmyk", json!({})).unwrap();
        s.execute("layer.newFillLayer.solidColor", json!({})).unwrap();
        let imported = Color { mode: ColorMode::Lab, c: [0.5123457, 0.4987654, 0.5678912, 0.0], alpha: 0.3456789 };
        s.execute(SET, json!({"color": imported})).unwrap();
        let before = s.active().unwrap().doc.clone();
        let count = s.active().unwrap().history.entries().len();
        let result = s.execute(SET, json!({"color": imported})).unwrap();
        assert_eq!(serde_json::from_value::<Color>(result["color"].clone()).unwrap(), imported);
        assert!(std::sync::Arc::ptr_eq(&before, &s.active().unwrap().doc));
        assert_eq!(s.active().unwrap().history.entries().len(), count);
    }

    #[test]
    fn invalid_rgb_components_leave_document_and_history_unchanged() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 4, "height": 4})).unwrap();
        s.execute("layer.newFillLayer.solidColor", json!({})).unwrap();
        let before = s.active().unwrap().doc.clone();
        let count = s.active().unwrap().history.entries().len();
        for color in [
            json!([5.0, 0.0, 0.0]),
            json!([-0.01, 0.0, 0.0]),
            json!([1.00000000000001_f64, 0.0, 0.0]),
            json!([0.0, 0.0, 0.0, -0.1]),
            json!([0.0, 0.0, 0.0, 1.1]),
            json!([1e300, 0.0, 0.0]),
            json!(["NaN", 0.0, 0.0]),
            json!(["Infinity", 0.0, 0.0]),
            json!([null, 0.0, 0.0]),
            json!([0.0, 0.0]),
            json!([0.0, 0.0, 0.0, 0.0, 0.0]),
        ] {
            let err = s.execute(SET, json!({"color": color})).unwrap_err();
            assert!(matches!(err, EngineError::BadParams { .. }), "{err}");
            assert!(std::sync::Arc::ptr_eq(&before, &s.active().unwrap().doc));
            assert_eq!(s.active().unwrap().history.entries().len(), count);
        }
        // Stored non-RGB colours are not subject to the RGB picker component limits.
        let wide = Color { mode: ColorMode::Lab, c: [1.1, -0.1, 0.6, 0.0], alpha: 0.75 };
        s.execute(SET, json!({"color": wide})).unwrap();
        let active = s.active().unwrap().active_layer.unwrap();
        let LayerContent::Fill(Fill::Solid(stored)) = s.active().unwrap().doc.layer(active).unwrap().content else { panic!() };
        assert_eq!(stored, wide);
    }
}
