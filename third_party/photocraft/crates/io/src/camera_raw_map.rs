//! Camera Raw Filter descriptors observed in real PSD files (filter version 18.4, process 6).
//! Verified develop controls are editable. Other processing versions
//! or settings retain the entire filter through the existing opaque-filter path.
use photocraft_algo::camera_raw::CameraRaw;
use photocraft_doc::SmartFilter;
use photocraft_psd::descriptor::{Descriptor, UnicodeString, Value};
use serde_json::{Value as J, json};

use super::{blend_options, from_hex, get_desc, int, rgbc, set, to_hex};

pub(super) const CLASS: &str = "Adobe Camera Raw Filter";
pub(super) const COMMAND: &str = "filter.cameraRaw";
const FILTER_ID: i32 = 2783;
const TEMPLATE_KEY: &str = "__cameraRawPsd";
const MAX_TEMPLATE_BYTES: usize = 1024 * 1024;

// Exposure and sharpening radius are doubles; other observed controls are integer levels.
const SCALARS: &[(&str, &str)] = &[
    ("temperature", "Temp"),
    ("tint", "Tint"),
    ("exposure", "Ex12"),
    ("contrast", "Cr12"),
    ("highlights", "Hi12"),
    ("shadows", "Sh12"),
    ("whites", "Wh12"),
    ("blacks", "Bk12"),
    ("texture", "CrTx"),
    ("clarity", "Cl12"),
    ("dehaze", "Dhze"),
    ("vibrance", "Vibr"),
    ("saturation", "Strt"),
    ("curveShadows", "PC_S"),
    ("curveDarks", "PC_D"),
    ("curveLights", "PC_L"),
    ("curveHighlights", "PC_H"),
    ("sharpenAmount", "Shrp"),
    ("sharpenRadius", "ShpR"),
    ("sharpenDetail", "ShpD"),
    ("sharpenMasking", "ShpM"),
    ("noiseLuminance", "LNR "),
    ("noiseLuminanceDetail", "LNRD"),
    ("noiseColor", "CNR "),
    ("noiseColorDetail", "CNRD"),
    ("gradeShadows.hue", "STSH"),
    ("gradeShadows.sat", "STSS"),
    ("gradeShadows.lum", "CgSL"),
    ("gradeMidtones.hue", "CgMH"),
    ("gradeMidtones.sat", "CgMS"),
    ("gradeMidtones.lum", "CgML"),
    ("gradeHighlights.hue", "STHH"),
    ("gradeHighlights.sat", "STHS"),
    ("gradeHighlights.lum", "CgHL"),
    ("gradeGlobal.hue", "CgGH"),
    ("gradeGlobal.sat", "CgGS"),
    ("gradeGlobal.lum", "CgGL"),
    ("gradeBlending", "CgBl"),
    ("gradeBalance", "STB "),
    ("grainAmount", "GRNA"),
    ("grainSize", "GRNS"),
    ("grainRoughness", "GRNF"),
    ("vignetteAmount", "PCVA"),
    ("vignetteMidpoint", "PCVM"),
    ("vignetteFeather", "PCVF"),
    ("vignetteRoundness", "PCVR"),
];
const ARRAYS: &[(&str, &[&str])] = &[
    ("curveSplits", &["PC_1", "PC_2", "PC_3"]),
    ("hslHue", &["HA_R", "HA_O", "HA_Y", "HA_G", "HA_A", "HA_B", "HA_P", "HA_M"]),
    ("hslSat", &["SA_R", "SA_O", "SA_Y", "SA_G", "SA_A", "SA_B", "SA_P", "SA_M"]),
    ("hslLum", &["LA_R", "LA_O", "LA_Y", "LA_G", "LA_A", "LA_B", "LA_P", "LA_M"]),
];
const CURVES: &[(&str, &str)] = &[("pointCurve", "Crv "), ("pointCurveRed", "CrvR"), ("pointCurveGreen", "CrvG"), ("pointCurveBlue", "CrvB")];

// These controls have no PhotoCraft counterpart yet. Their observed neutral values can be
// retained in the template; active values must keep the entire filter opaque.
const NEUTRAL_ONLY: &[(&str, i32)] = &[("LNRC", 0), ("CNRS", 50), ("MDis", 0), ("VigA", 0), ("VigM", 50), ("crfs", 100), ("TMMs", 0), ("PGTM", 0), ("RGBt", 0)];

fn default_params() -> Option<J> {
    serde_json::to_value(CameraRaw::default()).ok()
}

fn known_key(key: &str) -> bool {
    matches!(key, "CrVe" | "PrVN" | "PrVe" | "WBal")
        || SCALARS.iter().any(|(_, k)| *k == key)
        || ARRAYS.iter().any(|(_, keys)| keys.contains(&key))
        || CURVES.iter().any(|(_, k)| *k == key)
        || NEUTRAL_ONLY.iter().any(|(k, _)| *k == key)
}

fn get_param<'a>(params: &'a J, name: &str) -> Option<&'a J> {
    if let Some((wheel, field)) = name.split_once('.') { params.get(wheel)?.get(field) } else { params.get(name) }
}

fn get_param_mut<'a>(params: &'a mut J, name: &str) -> Option<&'a mut J> {
    if let Some((wheel, field)) = name.split_once('.') { params.get_mut(wheel)?.get_mut(field) } else { params.get_mut(name) }
}

fn number_range(key: &str) -> std::ops::RangeInclusive<f64> {
    match key {
        "Ex12" => -5.0..=5.0,
        "ShpR" => 0.5..=3.0,
        "Shrp" => 0.0..=150.0,
        "STSH" | "STHH" | "CgMH" | "CgGH" => 0.0..=360.0,
        "PC_1" | "PC_2" | "PC_3" | "ShpD" | "ShpM" | "LNR " | "LNRD" | "CNR " | "CNRD" | "STSS" | "STHS" | "CgMS" | "CgGS" | "CgBl" | "GRNA" | "GRNS"
        | "GRNF" | "PCVM" | "PCVF" => 0.0..=100.0,
        _ => -100.0..=100.0,
    }
}

fn read_number(d: &Descriptor, key: &str) -> Option<f64> {
    let value = match (key, d.get(key)?) {
        ("Ex12" | "ShpR", Value::Double(value)) => *value,
        (_, Value::Integer(value)) if !matches!(key, "Ex12" | "ShpR") => f64::from(*value),
        _ => return None,
    };
    (value.is_finite() && number_range(key).contains(&value)).then_some(value)
}

fn read_curve(value: &Value) -> Option<J> {
    let Value::List(values) = value else { return None };
    if values.len() < 4 || !values.len().is_multiple_of(2) || values.len() > photocraft_algo::camera_raw::MAX_CURVE_POINTS * 2 {
        return None;
    }
    let points: Option<Vec<[f32; 2]>> = values
        .as_chunks::<2>()
        .0
        .iter()
        .map(|[x, y]| {
            let (Value::Integer(x), Value::Integer(y)) = (x, y) else { return None };
            Some([*x as f32, *y as f32])
        })
        .collect();
    let points = points?;
    photocraft_algo::camera_raw::validate_curve(&points).ok()?;
    if points == [[0.0, 0.0], [255.0, 255.0]] {
        return Some(json!([]));
    }
    serde_json::to_value(points).ok()
}

fn descriptor_params(d: &Descriptor) -> Option<J> {
    if !d.class_id.is(CLASS)
        || !matches!((int(d, "PrVN"), int(d, "PrVe")), (Some(5), Some(184549376)) | (Some(6), Some(251920384)))
        || !matches!(d.get("CrVe"), Some(Value::Text(_)))
    {
        return None;
    }
    // A partial decode would omit active processing (masks, profiles, HDR or other settings)
    // when the layer is re-rendered. Keep such a filter opaque instead.
    if d.items.iter().any(|(key, _)| !known_key(&String::from_utf8_lossy(key.as_bytes()))) {
        return None;
    }
    if d.get("WBal").is_some_and(|v| !matches!(v, Value::Enumerated { type_id, value } if type_id.is("WBal") && value.is("Cst ")))
        || NEUTRAL_ONLY.iter().any(|(key, neutral)| d.get(key).is_some() && int(d, key) != Some(*neutral))
    {
        return None;
    }
    let mut params = default_params()?;
    for (param, key) in SCALARS {
        if d.get(key).is_some() {
            *get_param_mut(&mut params, param)? = json!(read_number(d, key)?);
        }
    }
    for (param, key) in CURVES {
        if let Some(value) = d.get(key) {
            *params.get_mut(*param)? = read_curve(value)?;
        }
    }
    for (param, keys) in ARRAYS {
        for (index, key) in keys.iter().enumerate() {
            if d.get(key).is_some() {
                *params.get_mut(*param)?.get_mut(index)? = json!(read_number(d, key)?);
            }
        }
    }
    validate_splits(&params).ok()?;
    // Use the same f32 representation as the editor so opening and confirming without changes
    // does not rewrite a stored double such as 1.15 to 1.149999976158142.
    let typed: CameraRaw = serde_json::from_value(params).ok()?;
    serde_json::to_value(typed).ok()
}

pub(super) fn import_params(item: &Descriptor) -> Option<J> {
    if int(item, "filterID") != Some(FILTER_ID) {
        return None;
    }
    let mut params = descriptor_params(get_desc(item, "Fltr")?)?;
    let mut template = item.clone();
    let neutral = SmartFilter { command: String::new(), params: J::Null, blend: photocraft_color::BlendMode::Normal, opacity: 1.0, visible: true };
    set(&mut template, "blendOptions", blend_options(&neutral));
    set(&mut template, "enab", Value::Boolean(true));
    let bytes = template.to_bytes();
    if bytes.len() > MAX_TEMPLATE_BYTES {
        return None;
    }
    params.as_object_mut()?.insert(TEMPLATE_KEY.into(), json!(to_hex(&bytes)));
    Some(params)
}

fn validate_splits(params: &J) -> Result<(), String> {
    let values = params.get("curveSplits").and_then(J::as_array).ok_or("curveSplits must contain three numbers")?;
    if values.len() != 3 {
        return Err("curveSplits must contain three numbers".into());
    }
    let mut previous = -1.0;
    for value in values {
        let value = value.as_f64().filter(|v| v.is_finite() && (0.0..=100.0).contains(v)).ok_or("curve splits must be finite percentages")?;
        if value <= previous {
            return Err("curve splits must be strictly increasing".into());
        }
        previous = value;
    }
    Ok(())
}

fn supported_param(key: &str) -> bool {
    SCALARS.iter().any(|(p, _)| p.split('.').next() == Some(key)) || ARRAYS.iter().any(|(p, _)| *p == key) || CURVES.iter().any(|(p, _)| *p == key)
}

fn write_number(d: &mut Descriptor, key: &str, value: &J) -> Result<(), String> {
    let number = value.as_f64().filter(|v| v.is_finite()).ok_or_else(|| format!("Camera Raw {key} must be finite"))?;
    if !number_range(key).contains(&number) {
        return Err(format!("Camera Raw {key} is outside the supported range"));
    }
    // PSD stores these as integer slider levels; PhotoCraft's sliders are continuous. Rounding
    // (at most half a level) keeps the filter instead of leaving it out of the PSD.
    // Hue 360 is the same angle as 0.
    let value = match key {
        "Ex12" | "ShpR" => Value::Double(number),
        "STSH" | "STHH" | "CgMH" | "CgGH" => Value::Integer(number.round() as i32 % 360),
        _ => Value::Integer(number.round() as i32),
    };
    set(d, key, value);
    Ok(())
}

pub(super) fn export_item(params: &J) -> Result<Descriptor, String> {
    let object = params.as_object().ok_or("Camera Raw parameters must be an object")?;
    let defaults = default_params().ok_or("cannot encode Camera Raw defaults")?;
    for key in object.keys() {
        if key != "layer" && !key.starts_with("__") && defaults.get(key).is_none() {
            return Err(format!("Camera Raw setting `{key}` has no verified PSD mapping"));
        }
    }
    for wheel in ["gradeShadows", "gradeMidtones", "gradeHighlights", "gradeGlobal"] {
        if params.get(wheel).and_then(J::as_object).is_some_and(|v| v.keys().any(|k| !matches!(k.as_str(), "hue" | "sat" | "lum"))) {
            return Err(format!("Camera Raw `{wheel}` contains an unknown wheel setting"));
        }
    }
    // Bound variable-sized curves before deserialization allocates their point arrays.
    for (param, _) in CURVES {
        if params.get(*param).and_then(J::as_array).is_some_and(|v| v.len() > photocraft_algo::camera_raw::MAX_CURVE_POINTS) {
            return Err(format!("Camera Raw `{param}` has too many points"));
        }
    }
    let mut item = if let Some(raw) = object.get(TEMPLATE_KEY) {
        let raw = raw.as_str().ok_or("Camera Raw PSD template must be hex text")?;
        if raw.len() > MAX_TEMPLATE_BYTES * 2 {
            return Err("Camera Raw PSD template is too large".into());
        }
        let bytes = from_hex(raw).ok_or("Camera Raw PSD template is not valid hex")?;
        let item = Descriptor::from_bytes(&bytes).map_err(|e| format!("bad Camera Raw PSD template: {e}"))?;
        if !item.class_id.is("filterFX") || int(&item, "filterID") != Some(FILTER_ID) || get_desc(&item, "Fltr").and_then(descriptor_params).is_none() {
            return Err("Camera Raw PSD template is not a supported Camera Raw filter".into());
        }
        item
    } else {
        Descriptor::new("filterFX")
            .with("Nm  ", Value::Text(UnicodeString::new_nul("Camera Raw Filter")))
            .with("hasoptions", Value::Boolean(true))
            .with("FrgC", rgbc(0.0))
            .with("BckC", rgbc(255.0))
            .with(
                "Fltr",
                Value::Descriptor(
                    Descriptor::new(CLASS)
                        .with("CrVe", Value::Text(UnicodeString::new_nul("18.4")))
                        .with("PrVN", Value::Integer(6))
                        .with("PrVe", Value::Integer(251920384)),
                ),
            )
            .with("filterID", Value::Integer(FILTER_ID))
    };
    let typed: CameraRaw = serde_json::from_value(params.clone()).map_err(|e| format!("bad Camera Raw parameters: {e}"))?;
    typed.validate()?;
    let normalized = serde_json::to_value(typed).map_err(|e| e.to_string())?;
    for (key, value) in normalized.as_object().ok_or("cannot encode Camera Raw parameters")? {
        if !supported_param(key) && defaults.get(key) != Some(value) {
            return Err(format!("Camera Raw setting `{key}` has no verified PSD mapping"));
        }
    }
    validate_splits(&normalized)?;
    let mut d = get_desc(&item, "Fltr").cloned().ok_or("Camera Raw PSD template has no filter settings")?;
    let original = descriptor_params(&d).ok_or("unsupported Camera Raw processing version or settings")?;
    for (param, key) in SCALARS {
        let value = get_param(&normalized, param).ok_or("missing Camera Raw parameter")?;
        if get_param(&original, param) != Some(value) {
            write_number(&mut d, key, value)?;
            if matches!(*key, "Temp" | "Tint") {
                set(
                    &mut d,
                    "WBal",
                    Value::Enumerated { type_id: photocraft_psd::descriptor::Id::new("WBal"), value: photocraft_psd::descriptor::Id::new("Cst ") },
                );
            }
        }
    }
    for (param, key) in CURVES {
        let value = normalized.get(*param).ok_or("missing Camera Raw curve")?;
        if original.get(*param) != Some(value) {
            let points = value.as_array().ok_or("Camera Raw curve must be an array")?;
            let mut levels: Vec<[f32; 2]> = Vec::new();
            for point in points {
                let mut level = [0.0; 2];
                let coordinates = point.as_array().filter(|c| c.len() == 2).ok_or("Camera Raw curve point must contain two coordinates")?;
                for (out, coordinate) in level.iter_mut().zip(coordinates) {
                    let number = coordinate
                        .as_f64()
                        .filter(|v| v.is_finite() && (0.0..=255.0).contains(v))
                        .ok_or("Camera Raw curve coordinates must be levels 0–255")?;
                    *out = number.round() as f32;
                }
                levels.push(level);
            }
            if levels.is_empty() {
                levels = vec![[0.0, 0.0], [255.0, 255.0]];
            }
            photocraft_algo::camera_raw::validate_curve(&levels).map_err(|e| format!("Camera Raw `{param}`: {e}"))?;
            let values = levels.iter().flatten().map(|v| Value::Integer(*v as i32)).collect();
            set(&mut d, key, Value::List(values));
        }
    }
    for (param, keys) in ARRAYS {
        for (index, key) in keys.iter().enumerate() {
            let value = normalized.get(*param).and_then(|v| v.get(index)).ok_or("missing Camera Raw array parameter")?;
            if original.get(*param).and_then(|v| v.get(index)) != Some(value) {
                write_number(&mut d, key, value)?;
            }
        }
    }
    set(&mut item, "Fltr", Value::Descriptor(d));
    Ok(item)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smart_map::{UNSUPPORTED_FILTER, filter_from_item, item_for_filter};
    use photocraft_color::BlendMode;

    // Parameter values observed in the supplied 18.4 PSD. No source pixels or private
    // document metadata are included in this synthetic regression fixture.
    fn source_item() -> Descriptor {
        let mut d = Descriptor::new(CLASS)
            .with("CrVe", Value::Text(UnicodeString::new_nul("18.4")))
            .with("PrVN", Value::Integer(6))
            .with("PrVe", Value::Integer(251920384))
            .with("Ex12", Value::Double(1.15));
        for (key, value) in [
            ("Cr12", -38),
            ("Hi12", 35),
            ("Sh12", -50),
            ("Wh12", 49),
            ("Bk12", -34),
            ("PC_S", 0),
            ("PC_D", -21),
            ("PC_L", 38),
            ("PC_H", 0),
            ("PC_1", 25),
            ("PC_2", 50),
            ("PC_3", 75),
        ] {
            d = d.with(key, Value::Integer(value));
        }
        for (_, keys) in ARRAYS.iter().skip(1) {
            for key in *keys {
                let value = match *key {
                    "HA_R" => -50,
                    "HA_Y" => 29,
                    "HA_G" => 1,
                    _ => 0,
                };
                d = d.with(key, Value::Integer(value));
            }
        }
        let f = SmartFilter { command: COMMAND.into(), params: J::Null, blend: BlendMode::Normal, opacity: 1.0, visible: true };
        Descriptor::new("filterFX")
            .with("Nm  ", Value::Text(UnicodeString::new_nul("Camera Raw Filter")))
            .with("blendOptions", blend_options(&f))
            .with("enab", Value::Boolean(true))
            .with("hasoptions", Value::Boolean(true))
            .with("FrgC", rgbc(0.0))
            .with("BckC", rgbc(255.0))
            .with("Fltr", Value::Descriptor(d))
            .with("filterID", Value::Integer(FILTER_ID))
    }

    fn all_controls_item() -> Descriptor {
        let mut item = source_item();
        let mut d = get_desc(&item, "Fltr").unwrap().clone();
        set(&mut d, "WBal", Value::Enumerated { type_id: photocraft_psd::descriptor::Id::new("WBal"), value: photocraft_psd::descriptor::Id::new("Cst ") });
        set(&mut d, "ShpR", Value::Double(1.2));
        for (key, value) in [
            ("Temp", -26),
            ("Tint", -25),
            ("CrTx", -31),
            ("Cl12", 30),
            ("Dhze", -34),
            ("Vibr", 33),
            ("Strt", 33),
            ("Shrp", 43),
            ("ShpD", 36),
            ("ShpM", 30),
            ("LNR ", 47),
            ("LNRD", 50),
            ("CNR ", 25),
            ("CNRD", 50),
            ("STSH", 117),
            ("STSS", 52),
            ("STHH", 316),
            ("STHS", 70),
            ("STB ", 35),
            ("CgMH", 51),
            ("CgMS", 51),
            ("CgSL", -24),
            ("CgML", 41),
            ("CgHL", 28),
            ("CgBl", 36),
            ("CgGH", 20),
            ("CgGS", 61),
            ("CgGL", 45),
            ("GRNA", 45),
            ("GRNS", 25),
            ("GRNF", 50),
            ("PCVA", 31),
            ("PCVM", 50),
            ("PCVF", 50),
            ("PCVR", 0),
        ] {
            set(&mut d, key, Value::Integer(value));
        }
        for (key, neutral) in NEUTRAL_ONLY {
            set(&mut d, key, Value::Integer(*neutral));
        }
        for (key, x, y) in [("Crv ", 146, 102), ("CrvR", 90, 174), ("CrvG", 162, 122), ("CrvB", 129, 202)] {
            set(&mut d, key, Value::List([0, 0, x, y, 255, 255].into_iter().map(Value::Integer).collect()));
        }
        set(&mut item, "Fltr", Value::Descriptor(d));
        item
    }

    #[test]
    fn all_observed_develop_controls_roundtrip_and_patch_only_the_edited_field() {
        let source = all_controls_item();
        let f = filter_from_item(&source);
        assert_eq!(f.command, COMMAND);
        assert_eq!(f.params["temperature"], -26.0);
        assert_eq!(f.params["texture"], -31.0);
        assert!((f.params["sharpenRadius"].as_f64().unwrap() - 1.2).abs() < 1e-6);
        assert_eq!(f.params["gradeHighlights"], json!({"hue": 316.0, "sat": 70.0, "lum": 28.0}));
        assert_eq!(f.params["pointCurveBlue"], json!([[0.0, 0.0], [129.0, 202.0], [255.0, 255.0]]));
        assert_eq!(item_for_filter(&f).unwrap(), source);
        let defaults = default_params().unwrap();
        for (param, key) in SCALARS {
            let mut edited = f.clone();
            let mut number = get_param(&defaults, param).unwrap().as_f64().unwrap();
            if get_param(&f.params, param).unwrap().as_f64() == Some(number) {
                number += 1.0;
            }
            *get_param_mut(&mut edited.params, param).unwrap() = json!(number);
            let mut expected = get_desc(&source, "Fltr").unwrap().clone();
            set(&mut expected, key, if matches!(*key, "Ex12" | "ShpR") { Value::Double(number) } else { Value::Integer(number as i32) });
            assert_eq!(get_desc(&item_for_filter(&edited).unwrap(), "Fltr"), Some(&expected), "{param}");
        }
        for (param, key) in CURVES {
            let mut edited = f.clone();
            edited.params[*param] = json!([]);
            let item = item_for_filter(&edited).unwrap();
            let mut expected = get_desc(&source, "Fltr").unwrap().clone();
            set(&mut expected, key, Value::List([0, 0, 255, 255].into_iter().map(Value::Integer).collect()));
            assert_eq!(get_desc(&item, "Fltr"), Some(&expected), "{param}");
            assert_eq!(filter_from_item(&item).params[*param], json!([]), "{param}");
        }
    }

    #[test]
    fn active_unimplemented_controls_and_malformed_curves_stay_opaque() {
        for (key, neutral) in NEUTRAL_ONLY {
            let mut item = all_controls_item();
            let mut d = get_desc(&item, "Fltr").unwrap().clone();
            set(&mut d, key, Value::Integer(neutral + 1));
            set(&mut item, "Fltr", Value::Descriptor(d));
            let f = filter_from_item(&item);
            assert_eq!(f.command, UNSUPPORTED_FILTER, "{key}");
            assert_eq!(item_for_filter(&f).unwrap(), item, "{key}");
        }
        for value in [
            Value::List(vec![Value::Integer(0)]),
            Value::List([0, 0, 0, 255].into_iter().map(Value::Integer).collect()),
            Value::List([0, 0, 255, 256].into_iter().map(Value::Integer).collect()),
            Value::List(vec![Value::Double(0.0); 4]),
            Value::List(vec![Value::Integer(0); 34]),
        ] {
            let mut item = all_controls_item();
            let mut d = get_desc(&item, "Fltr").unwrap().clone();
            set(&mut d, "Crv ", value);
            set(&mut item, "Fltr", Value::Descriptor(d));
            let f = filter_from_item(&item);
            assert_eq!(f.command, UNSUPPORTED_FILTER);
            assert_eq!(item_for_filter(&f).unwrap(), item);
        }
    }

    #[test]
    #[ignore = "requires a user-supplied Camera Raw PSD"]
    fn all_sliders_fixture() {
        let path = std::env::var("PHOTOCRAFT_CAMERA_RAW_PSD").expect("set PHOTOCRAFT_CAMERA_RAW_PSD");
        let doc = crate::import("fixture.psd", &std::fs::read(path).unwrap()).unwrap().document;
        let original = doc
            .walk()
            .into_iter()
            .find_map(|(_, _, layer)| {
                let photocraft_doc::LayerContent::Smart(sm) = &layer.content else { return None };
                sm.smart_filters.iter().find_map(|f| {
                    let item = item_for_filter(f).ok()?;
                    get_desc(&item, "Fltr").is_some_and(|d| d.class_id.is(CLASS)).then_some(item)
                })
            })
            .expect("a Camera Raw descriptor");
        assert_eq!(item_for_filter(&filter_from_item(&original)).unwrap(), original);
        let mut projected = original.clone();
        let mut d = get_desc(&original, "Fltr").unwrap().clone();
        // This is a codec projection, not an editable replacement for the source filter:
        // active Optics settings have no rendering counterpart, and PCVS is unverified.
        d.items.retain(|(key, _)| !["MDis", "VigA", "PCVS"].iter().any(|k| key.is(k)));
        set(&mut projected, "Fltr", Value::Descriptor(d));
        let mut f = filter_from_item(&projected);
        assert_eq!(f.command, COMMAND);
        assert_eq!(f.params["temperature"], -26.0);
        assert_eq!(f.params["sharpenAmount"], 43.0);
        assert_eq!(f.params["gradeGlobal"], json!({"hue": 20.0, "sat": 61.0, "lum": 45.0}));
        assert_eq!(f.params["pointCurveRed"], json!([[0.0, 0.0], [90.0, 174.0], [255.0, 255.0]]));
        assert_eq!(item_for_filter(&f).unwrap(), projected);
        f.params["exposure"] = json!(0.5);
        let mut expected = get_desc(&projected, "Fltr").unwrap().clone();
        set(&mut expected, "Ex12", Value::Double(0.5));
        assert_eq!(get_desc(&item_for_filter(&f).unwrap(), "Fltr"), Some(&expected));
    }

    #[test]
    fn light_curve_and_hsl_are_editable_without_rewriting_unchanged_settings() {
        let source = source_item().with("futureOuterData", Value::RawData(vec![1, 2, 3, 4]));
        let mut f = filter_from_item(&source);
        assert_eq!(f.command, COMMAND);
        assert!((f.params["exposure"].as_f64().unwrap() - 1.15).abs() < 1e-6);
        assert_eq!(f.params["contrast"], -38.0);
        assert_eq!(f.params["curveDarks"], -21.0);
        assert_eq!(f.params["curveLights"], 38.0);
        assert_eq!(f.params["curveSplits"], json!([25.0, 50.0, 75.0]));
        assert_eq!(f.params["hslHue"], json!([-50.0, 0.0, 29.0, 1.0, 0.0, 0.0, 0.0, 0.0]));
        assert_eq!(item_for_filter(&f).unwrap(), source);
        f.params["exposure"] = json!(0.0);
        f.params["hslHue"][2] = json!(-20.0);
        f.visible = false;
        f.opacity = 0.4;
        f.blend = BlendMode::Multiply;
        let edited = item_for_filter(&f).unwrap();
        let mut expected = get_desc(&source, "Fltr").unwrap().clone();
        set(&mut expected, "Ex12", Value::Double(0.0));
        set(&mut expected, "HA_Y", Value::Integer(-20));
        assert_eq!(get_desc(&edited, "Fltr"), Some(&expected));
        assert_eq!(edited.get("futureOuterData"), source.get("futureOuterData"));
        let back = filter_from_item(&edited);
        assert_eq!((back.visible, back.blend, back.opacity), (false, BlendMode::Multiply, 0.4));
        assert_eq!(back.params["exposure"], 0.0);
        assert_eq!(back.params["hslHue"][2], -20.0);
    }

    #[test]
    fn new_filters_write_every_verified_control_and_accept_explicit_defaults() {
        let mut params = default_params().unwrap();
        params["seed"] = json!(0);
        for (param, key) in SCALARS {
            let number = match *key {
                "Ex12" => 0.75,
                "ShpR" => 1.2,
                _ => 12.0,
            };
            *get_param_mut(&mut params, param).unwrap() = json!(number);
        }
        for (param, keys) in ARRAYS {
            params[*param] = if *param == "curveSplits" { json!([20, 45, 80]) } else { json!((0..keys.len()).map(|i| i as f64 + 1.0).collect::<Vec<_>>()) };
        }
        let f = SmartFilter { command: COMMAND.into(), params: params.clone(), blend: BlendMode::Screen, opacity: 0.5, visible: false };
        let item = item_for_filter(&f).unwrap();
        let back = filter_from_item(&Descriptor::from_bytes(&item.to_bytes()).unwrap());
        assert_eq!((&back.command, back.visible, back.blend, back.opacity), (&f.command, false, BlendMode::Screen, 0.5));
        // Compare slider numbers after both sides have been normalized through CameraRaw.
        let typed: CameraRaw = serde_json::from_value(params).unwrap();
        let expected = serde_json::to_value(typed).unwrap();
        for param in SCALARS.iter().map(|(p, _)| *p).chain(ARRAYS.iter().map(|(p, _)| *p)) {
            assert_eq!(get_param(&back.params, param), get_param(&expected, param));
        }
    }

    #[test]
    fn unsupported_processing_settings_remain_opaque_and_lossless() {
        for (key, value) in [
            ("LCs ", Value::Text(UnicodeString::new("unknown masks"))),
            ("PrVN", Value::Integer(7)),
            ("PrVe", Value::Integer(123)),
            ("Ex12", Value::Double(f64::INFINITY)),
            ("Cr12", Value::Text(UnicodeString::new("bad type"))),
        ] {
            let mut item = source_item();
            let mut d = get_desc(&item, "Fltr").unwrap().clone();
            set(&mut d, key, value);
            set(&mut item, "Fltr", Value::Descriptor(d));
            let f = filter_from_item(&item);
            assert_eq!(f.command, UNSUPPORTED_FILTER, "{key}");
            assert_eq!(item_for_filter(&f).unwrap(), item, "{key}");
        }
    }

    #[test]
    fn fractional_slider_levels_round_instead_of_dropping_the_filter() {
        let mut f = filter_from_item(&source_item());
        f.params["contrast"] = json!(-12.4);
        f.params["hslHue"][0] = json!(10.5);
        f.params["gradeShadows"]["hue"] = json!(359.7);
        f.params["pointCurve"] = json!([[0, 0], [100.2, 120.6], [200.6, 180.4], [255, 255]]);
        let item = item_for_filter(&f).unwrap();
        let d = get_desc(&item, "Fltr").unwrap();
        assert_eq!(d.get("Cr12"), Some(&Value::Integer(-12)));
        assert_eq!(d.get("HA_R"), Some(&Value::Integer(11)));
        assert_eq!(d.get("STSH"), Some(&Value::Integer(0)));
        assert_eq!(d.get("Crv "), Some(&Value::List([0, 0, 100, 121, 201, 180, 255, 255].into_iter().map(Value::Integer).collect())));
        let back = filter_from_item(&item);
        assert_eq!(back.command, COMMAND);
        assert_eq!(back.params["contrast"], -12.0);
    }

    #[test]
    fn unsupported_or_malformed_exports_report_errors() {
        for params in [
            J::Null,
            json!({"exposure": "bright"}),
            json!({"exposure": 1e40}),
            json!({"exposure": 6}),
            json!({"contrast": 101}),
            json!({"hslHue": [1, 2]}),
            json!({"curveSplits": [80, 50, 20]}),
            json!({"curveSplits": [25, 25, 75]}),
            json!({"pointCurve": [[0, 0], [0, 255]]}),
            json!({"sharpenRadius": 4}),
            json!({"gradeShadows": {"hue": 361}}),
            json!({"gradeShadows": {"unknown": 1}}),
            json!({"seed": 1}),
            json!({"vignetteHighlights": 10}),
            json!({"futureSetting": 1}),
            json!({TEMPLATE_KEY: "zz"}),
            json!({TEMPLATE_KEY: "00"}),
            json!({TEMPLATE_KEY: 12}),
            json!({TEMPLATE_KEY: "0".repeat(MAX_TEMPLATE_BYTES * 2 + 1)}),
        ] {
            assert!(export_item(&params).is_err());
        }
        let wrong = Descriptor::new("filterFX").with("filterID", Value::Integer(1));
        assert!(export_item(&json!({TEMPLATE_KEY: to_hex(&wrong.to_bytes())})).is_err());
    }
}
