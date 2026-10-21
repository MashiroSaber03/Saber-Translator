//! Validate new core-filter invocations, without changing how old smart-filter records decode.

use serde_json::Value;

use crate::{EngineError, Result};

#[derive(Clone, Copy)]
enum Rule {
    Number(f64, f64),
    Unsigned(u64),
    Choice(&'static str),
    Bool,
}

impl Rule {
    fn accepts(self, value: &Value) -> bool {
        match self {
            Self::Number(min, max) => value.as_f64().is_some_and(|n| n.is_finite() && (min..=max).contains(&n)),
            Self::Unsigned(max) => value.as_u64().is_some_and(|n| n <= max),
            Self::Choice(choices) => value.as_str().is_some_and(|s| choices.split('|').any(|choice| choice == s)),
            Self::Bool => value.is_boolean(),
        }
    }

    fn expected(self) -> String {
        match self {
            Self::Number(min, max) if min == f64::MIN && max == f64::MAX => "a finite number".into(),
            Self::Number(min, max) => format!("a number in {min}..{max}"),
            Self::Unsigned(max) => format!("an integer in 0..{max}"),
            Self::Choice(choices) => format!("one of {choices}"),
            Self::Bool => "a boolean".into(),
        }
    }
}

// The bounds and choices are the core filters' public dialog-unit contract. Defaults remain in
// params_for; gallery filters are checked in gallery_cmds.
fn fields(id: &str) -> Option<&'static [(&'static str, Rule)]> {
    use Rule::{Bool, Choice, Number, Unsigned};
    Some(match id {
        "filter.blur.gaussianBlur" | "filter.other.highPass" => &[("radius", Number(0.1, 1000.0))],
        "filter.blur.boxBlur" => &[("radius", Number(1.0, 2000.0))],
        "filter.blur.motionBlur" => &[("angle", Number(-360.0, 360.0)), ("distance", Number(1.0, 2000.0))],
        "filter.blur.radialBlur" => &[
            ("amount", Number(1.0, 100.0)),
            ("method", Choice("spin|zoom")),
            ("quality", Choice("draft|good|best")),
            ("centerX", Number(0.0, 1.0)),
            ("centerY", Number(0.0, 1.0)),
        ],
        "filter.blur.surfaceBlur" => &[("radius", Number(1.0, 100.0)), ("threshold", Number(2.0, 255.0))],
        "filter.sharpen.unsharpMask" => &[("amount", Number(1.0, 500.0)), ("radius", Number(0.1, 1000.0)), ("threshold", Number(0.0, 255.0))],
        "filter.sharpen.smartSharpen" => &[("amount", Number(1.0, 500.0)), ("radius", Number(0.1, 64.0)), ("reduceNoise", Number(0.0, 100.0))],
        "filter.noise.addNoise" => {
            &[("amount", Number(0.1, 400.0)), ("distribution", Choice("uniform|gaussian")), ("monochromatic", Bool), ("seed", Unsigned(u32::MAX as u64))]
        }
        "filter.noise.median" => &[("radius", Number(1.0, 500.0))],
        "filter.noise.dustAndScratches" => &[("radius", Number(1.0, 500.0)), ("threshold", Number(0.0, 255.0))],
        "filter.pixelate.mosaic" => &[("cellSize", Number(2.0, 200.0))],
        "filter.stylize.emboss" => &[("angle", Number(-180.0, 180.0)), ("height", Number(1.0, 100.0)), ("amount", Number(1.0, 500.0))],
        "filter.distort.twirl" => &[("angle", Number(-999.0, 999.0))],
        "filter.distort.pinch" => &[("amount", Number(-100.0, 100.0))],
        "filter.distort.spherize" => &[("amount", Number(-100.0, 100.0)), ("mode", Choice("normal|horizontalOnly|verticalOnly"))],
        "filter.distort.wave" => &[
            ("generators", Number(1.0, 999.0)),
            ("wavelengthMin", Number(1.0, 998.0)),
            ("wavelengthMax", Number(2.0, 999.0)),
            ("amplitudeMin", Number(1.0, 998.0)),
            ("amplitudeMax", Number(1.0, 999.0)),
            ("type", Choice("sine|triangle|square")),
            ("undefinedAreas", Choice("wrap|repeat")),
            ("seed", Unsigned(u32::MAX as u64)),
        ],
        "filter.distort.ripple" => &[("amount", Number(-999.0, 999.0)), ("size", Choice("small|medium|large"))],
        "filter.distort.polarCoordinates" => &[("mode", Choice("rectangularToPolar|polarToRectangular"))],
        "filter.other.minimum" | "filter.other.maximum" => &[("radius", Number(0.2, 500.0)), ("preserve", Choice("squareness|roundness"))],
        "filter.other.offset" => &[
            // The hint specifies pixels, with no range: keep the existing rounding/saturation.
            ("horizontal", Number(f64::MIN, f64::MAX)),
            ("vertical", Number(f64::MIN, f64::MAX)),
            ("undefinedAreas", Choice("wrap|repeat|transparent")),
        ],
        "filter.blur.blur"
        | "filter.blur.blurMore"
        | "filter.sharpen.sharpen"
        | "filter.sharpen.sharpenMore"
        | "filter.sharpen.sharpenEdges"
        | "filter.noise.despeckle"
        | "filter.stylize.findEdges"
        | "filter.stylize.solarize" => &[],
        _ => return None,
    })
}

// Extended filters: only their documented choices are checked for now, since `prepare` injects
// extra keys (colours, map documents) and their numeric ranges clamp in params_for.
fn choices(id: &str) -> Option<&'static [(&'static str, &'static str)]> {
    Some(match id {
        "filter.pixelate.mezzotint" => {
            &[("type", "fineDots|mediumDots|grainyDots|coarseDots|shortLines|mediumLines|longLines|shortStrokes|mediumStrokes|longStrokes")]
        }
        "filter.stylize.diffuse" => &[("mode", "normal|darkenOnly|lightenOnly|anisotropic")],
        "filter.stylize.extrude" => &[("type", "blocks|pyramids"), ("depthMode", "random|levelBased")],
        "filter.stylize.tiles" => &[("fill", "background|foreground|inverse|unaltered")],
        "filter.stylize.traceContour" => &[("edge", "lower|upper")],
        "filter.stylize.wind" => &[("method", "wind|blast|stagger"), ("direction", "fromRight|fromLeft")],
        "filter.distort.displace" => &[("fit", "stretch|tile"), ("undefinedAreas", "repeat|wrap")],
        "filter.distort.shear" => &[("undefinedAreas", "wrap|repeat")],
        "filter.distort.zigZag" => &[("style", "pondRipples|outFromCenter|aroundCenter")],
        "filter.render.lensFlare" => &[("lens", "zoom|prime35|prime105|moviePrime")],
        "filter.render.lightingEffects" => &[("lightType", "spot|point|infinite"), ("texture", "none|red|green|blue|alpha|luminance")],
        "filter.blur.smartBlur" => &[("quality", "high|medium|low"), ("mode", "normal|edgeOnly|overlayEdge")],
        "filter.blur.lensBlur" => {
            &[("shape", "hexagon|triangle|square|pentagon|heptagon|octagon"), ("depthMap", "none|transparency|layerMask"), ("distribution", "uniform|gaussian")]
        }
        "filter.blur.shapeBlur" => &[("shape", "circle|ring|square|diamond|triangle|hexagon|star|heart|cross")],
        "filter.other.hsbHsl" => &[("inputMode", "rgb|hsb|hsl"), ("rowOrder", "hsb|hsl|rgb")],
        "filter.video.deInterlace" => &[("eliminate", "oddFields|evenFields"), ("createBy", "interpolation|duplication")],
        _ => return None,
    })
}

/// Runs both at dispatch (before committing a floating selection) and at the shared runner
/// (direct specs/Last Filter). Filters outside the core and extended families are unaffected.
pub(crate) fn validate_params(id: &str, params: &Value) -> Result<()> {
    let bad = |msg| EngineError::BadParams { cmd: id.into(), msg };
    if let Some(choices) = choices(id) {
        for &(key, allowed) in choices {
            if let Some(value) = params.get(key)
                && !Rule::Choice(allowed).accepts(value)
            {
                return Err(bad(format!("`{key}` must be {}", Rule::Choice(allowed).expected())));
            }
        }
        return Ok(());
    }
    let Some(fields) = fields(id) else { return crate::gallery_cmds::validate_params(id, params) };
    if params.is_null() {
        return Ok(());
    }
    let params = params.as_object().ok_or_else(|| bad("params must be a JSON object".into()))?;
    for (key, value) in params {
        let (valid, expected) = if let Some((_, rule)) = fields.iter().find(|(name, _)| *name == key) {
            (rule.accepts(value), rule.expected())
        } else {
            match key.as_str() {
                "layer" => (value.as_u64().is_some(), "an unsigned layer id".into()),
                "coalesce" => (value.is_string(), "a string".into()),
                "target" => (
                    value.as_str().is_some_and(|s| matches!(s, "pixels" | "mask" | "quickMask"))
                        || value.as_object().is_some_and(|o| o.len() == 1 && o.get("channel").and_then(Value::as_u64).is_some()),
                    "pixels, mask, quickMask or {\"channel\":index}".into(),
                ),
                _ => return Err(bad(format!("unknown parameter `{key}`"))),
            }
        };
        if !valid {
            return Err(bad(format!("`{key}` must be {expected}")));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use photocraft_geom::Rect;
    use serde_json::json;

    use super::*;
    use crate::Session;

    fn session(depth: u32) -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width":16,"height":12,"depth":depth})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        s.edit("pattern", |doc, active| {
            let surface = doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap();
            surface.fill_rect(Rect::new(0, 0, 8, 12), &[0.2, 0.6, 0.8, 1.0]);
            Ok(())
        })
        .unwrap();
        s
    }

    fn rejected(s: &mut Session, id: &str, key: &str, value: Value) {
        let doc = s.active().unwrap().doc.clone();
        let history = s.active().unwrap().history.entries();
        let journal = s.journal.clone();
        let active = s.active().unwrap().active_layer;
        let error = s.execute(id, json!({key:value})).unwrap_err();
        assert!(matches!(error, EngineError::BadParams { .. }), "{id}: {error}");
        let error = error.to_string();
        assert!(error.contains(id) && error.contains(key), "{error}");
        assert!(Arc::ptr_eq(&s.active().unwrap().doc, &doc), "{id}: {key}");
        assert_eq!(s.active().unwrap().history.entries(), history);
        assert_eq!(s.active().unwrap().active_layer, active);
        assert_eq!(s.journal, journal);
        assert!(s.jobs().is_empty());
    }

    #[test]
    fn every_core_filter_rejects_unknown_keys_without_editing() {
        for depth in [8, 16, 32] {
            let mut s = session(depth);
            for spec in crate::filters::specs().into_iter().filter(|s| s.id != "filter.lastFilter") {
                rejected(&mut s, spec.id, "radus", json!(3));
            }
        }
    }

    #[test]
    fn core_filter_numbers_reject_wrong_types_and_documented_range_violations() {
        // Each numeric field in the core registry, independently of the validation table.
        let cases = [
            ("blur.gaussianBlur", "radius", 0.1, 1000.0),
            ("blur.boxBlur", "radius", 1.0, 2000.0),
            ("blur.motionBlur", "angle", -360.0, 360.0),
            ("blur.motionBlur", "distance", 1.0, 2000.0),
            ("blur.radialBlur", "amount", 1.0, 100.0),
            ("blur.radialBlur", "centerX", 0.0, 1.0),
            ("blur.radialBlur", "centerY", 0.0, 1.0),
            ("blur.surfaceBlur", "radius", 1.0, 100.0),
            ("blur.surfaceBlur", "threshold", 2.0, 255.0),
            ("sharpen.unsharpMask", "amount", 1.0, 500.0),
            ("sharpen.unsharpMask", "radius", 0.1, 1000.0),
            ("sharpen.unsharpMask", "threshold", 0.0, 255.0),
            ("sharpen.smartSharpen", "amount", 1.0, 500.0),
            ("sharpen.smartSharpen", "radius", 0.1, 64.0),
            ("sharpen.smartSharpen", "reduceNoise", 0.0, 100.0),
            ("noise.addNoise", "amount", 0.1, 400.0),
            ("noise.median", "radius", 1.0, 500.0),
            ("noise.dustAndScratches", "radius", 1.0, 500.0),
            ("noise.dustAndScratches", "threshold", 0.0, 255.0),
            ("pixelate.mosaic", "cellSize", 2.0, 200.0),
            ("stylize.emboss", "angle", -180.0, 180.0),
            ("stylize.emboss", "height", 1.0, 100.0),
            ("stylize.emboss", "amount", 1.0, 500.0),
            ("distort.twirl", "angle", -999.0, 999.0),
            ("distort.pinch", "amount", -100.0, 100.0),
            ("distort.spherize", "amount", -100.0, 100.0),
            ("distort.wave", "generators", 1.0, 999.0),
            ("distort.wave", "wavelengthMin", 1.0, 998.0),
            ("distort.wave", "wavelengthMax", 2.0, 999.0),
            ("distort.wave", "amplitudeMin", 1.0, 998.0),
            ("distort.wave", "amplitudeMax", 1.0, 999.0),
            ("distort.ripple", "amount", -999.0, 999.0),
            ("other.highPass", "radius", 0.1, 1000.0),
            ("other.minimum", "radius", 0.2, 500.0),
            ("other.maximum", "radius", 0.2, 500.0),
        ];
        for depth in [8, 16, 32] {
            let mut s = session(depth);
            for (suffix, key, min, max) in cases {
                let id = format!("filter.{suffix}");
                for value in [json!("big"), json!(true), Value::Null, json!([]), json!({}), json!(min - 0.01), json!(max + 0.01)] {
                    rejected(&mut s, &id, key, value);
                }
                // Large limits are validated without allocating a large blur kernel.
                for value in [min, max] {
                    validate_params(&id, &json!({key:value})).unwrap();
                }
            }
            for key in ["horizontal", "vertical"] {
                rejected(&mut s, "filter.other.offset", key, json!("4"));
            }
        }
    }

    #[test]
    fn core_filter_choices_booleans_seeds_and_shared_fields_are_validated() {
        let mut s = session(8);
        for (suffix, key, choices) in [
            ("blur.radialBlur", "method", "spin|zoom"),
            ("blur.radialBlur", "quality", "draft|good|best"),
            ("noise.addNoise", "distribution", "uniform|gaussian"),
            ("distort.spherize", "mode", "normal|horizontalOnly|verticalOnly"),
            ("distort.wave", "type", "sine|triangle|square"),
            ("distort.wave", "undefinedAreas", "wrap|repeat"),
            ("distort.ripple", "size", "small|medium|large"),
            ("distort.polarCoordinates", "mode", "rectangularToPolar|polarToRectangular"),
            ("other.minimum", "preserve", "squareness|roundness"),
            ("other.maximum", "preserve", "squareness|roundness"),
            ("other.offset", "undefinedAreas", "wrap|repeat|transparent"),
        ] {
            let id = format!("filter.{suffix}");
            for value in [json!("typo"), json!(false), json!(1), Value::Null] {
                rejected(&mut s, &id, key, value);
            }
            for value in choices.split('|') {
                validate_params(&id, &json!({key:value})).unwrap();
            }
        }
        for value in [json!("true"), json!(0), Value::Null] {
            rejected(&mut s, "filter.noise.addNoise", "monochromatic", value);
        }
        for id in ["filter.noise.addNoise", "filter.distort.wave"] {
            for value in [json!(-1), json!(4_294_967_296_u64), json!(0.5), json!("12")] {
                rejected(&mut s, id, "seed", value);
            }
            // A valid u32 seed must not be saturated through an i32 helper.
            let result = s.execute(id, json!({"seed":u32::MAX})).unwrap();
            assert_eq!(result["filter"]["seed"], json!(u32::MAX));
        }
        for (key, value) in [("layer", json!("2")), ("target", json!("typo")), ("target", json!({"channel":-1})), ("coalesce", json!(false))] {
            rejected(&mut s, "filter.blur.gaussianBlur", key, value);
        }
    }

    #[test]
    fn extended_filter_choices_match_their_specs_and_reject_unlisted_values() {
        let mut s = session(8);
        let specs = crate::filters_ext::specs();
        let ids = [
            "filter.pixelate.mezzotint",
            "filter.stylize.diffuse",
            "filter.stylize.extrude",
            "filter.stylize.tiles",
            "filter.stylize.traceContour",
            "filter.stylize.wind",
            "filter.distort.displace",
            "filter.distort.shear",
            "filter.distort.zigZag",
            "filter.render.lensFlare",
            "filter.render.lightingEffects",
            "filter.blur.smartBlur",
            "filter.blur.lensBlur",
            "filter.blur.shapeBlur",
            "filter.other.hsbHsl",
            "filter.video.deInterlace",
        ];
        for id in ids {
            let doc = specs.iter().find(|spec| spec.id == id).unwrap().params;
            for &(key, allowed) in choices(id).unwrap() {
                // The table repeats the registry's params doc, so the two cannot drift apart.
                assert!(doc.contains(&format!("\"{key}\":\"{allowed}\"")), "{id}.{key}");
                for value in [json!("zzbogus"), json!(false), json!(1), Value::Null] {
                    rejected(&mut s, id, key, value);
                }
                for value in allowed.split('|') {
                    validate_params(id, &json!({key:value})).unwrap();
                }
            }
        }
        // Keys `prepare` injects, and undocumented ones, stay accepted as before.
        s.execute("filter.stylize.tiles", json!({"count":4,"foreground":[1,0,0,1]})).unwrap();
        s.execute("filter.stylize.wind", json!({"direction":"fromLeft"})).unwrap();
    }

    #[test]
    fn invalid_filters_leave_floating_selection_and_background_jobs_alone() {
        let mut s = session(16);
        s.execute("select.rect", json!({"x":0,"y":0,"width":4,"height":4})).unwrap();
        s.execute("select.float", json!({"dx":8})).unwrap();
        let doc = s.active().unwrap().doc.clone();
        let history = s.active().unwrap().history.entries();
        for params in [json!({"radius":"big"}), json!({"radus":3}), json!({"radius":5000}), json!({"radius":-4})] {
            assert!(s.start("filter.blur.gaussianBlur", params.clone()).is_err());
            let spec = crate::commands::find("filter.blur.gaussianBlur").unwrap();
            assert!((spec.run)(&mut s, &params).is_err());
            assert!(crate::float_cmds::floating(s.active().unwrap()).is_some());
            assert!(Arc::ptr_eq(&s.active().unwrap().doc, &doc));
            assert_eq!(s.active().unwrap().history.entries(), history);
            assert!(s.jobs().is_empty());
        }
    }

    #[test]
    fn valid_filters_keep_defaults_explicit_targets_and_undo_redo() {
        for depth in [8, 16, 32] {
            let mut s = session(depth);
            let layer = s.active().unwrap().active_layer.unwrap();
            let pixels = |s: &Session| s.active().unwrap().doc.layer(layer).unwrap().surface().unwrap().read_region(Rect::new(0, 0, 16, 12));
            let before = pixels(&s);
            for params in [Value::Null, json!({})] {
                let result = s.execute("filter.blur.gaussianBlur", params).unwrap();
                assert_eq!(result["filter"]["radius"], json!(1.0));
                assert_ne!(pixels(&s), before);
                s.undo();
                assert_eq!(pixels(&s), before);
            }
            s.execute("layer.new.layer", json!({})).unwrap();
            let other = s.active().unwrap().active_layer.unwrap();
            let result = s.execute("filter.blur.gaussianBlur", json!({"radius":3,"layer":layer.0,"target":"pixels","coalesce":"blur"})).unwrap();
            assert_eq!(result["filter"]["radius"], json!(3.0));
            let after = pixels(&s);
            assert_ne!(after, before);
            assert_eq!(s.active().unwrap().doc.layer(other).unwrap().surface().unwrap().tile_count(), 0);
            s.undo();
            assert_eq!(pixels(&s), before);
            s.redo();
            assert_eq!(pixels(&s), after);
        }
    }

    #[test]
    fn injected_channel_targets_and_extended_filter_colours_keep_working() {
        let mut s = session(8);
        s.execute("channel.new", json!({"fill":"white"})).unwrap();
        let pixels = s.active().unwrap().doc.layers.clone();
        s.execute("filter.noise.addNoise", json!({"amount":40,"seed":2})).unwrap();
        assert_eq!(s.active().unwrap().doc.layers, pixels);
        assert!(s.active().unwrap().doc.channels[0].surface.read_region(Rect::new(0, 0, 16, 12)).iter().any(|&v| v < 0.99));
        s.execute("channel.target", json!({"channel":"composite"})).unwrap();
        s.tools.foreground = [0.0, 0.0, 1.0, 1.0];
        s.tools.background = [1.0, 0.0, 0.0, 1.0];
        let params = crate::filters_ext::prepare(&s, "filter.render.fibers", &json!({}));
        assert_eq!(params["foreground"], json!(s.tools.foreground));
        assert_eq!(params["background"], json!(s.tools.background));
        s.execute("filter.render.fibers", params).unwrap();
    }
}
