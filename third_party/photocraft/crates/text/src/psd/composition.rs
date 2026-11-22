//! PSD paragraph composition also lives in the document's numbered `Txt2` model.
//! Project the final type layers into that model without storing a second layout cache.

use std::collections::BTreeMap;

use photocraft_doc::TextLayer;

use super::{D, E, HARD_KINSOKU, HARD_KINSOKU_RULES, MAX_TEXT_INDEX, TATE_CHU_YOKO, ed, engine_data, parse_tysh, real, txt2_mode_runs, with_modes};

fn dict<const N: usize>(entries: [(&str, E); N]) -> E {
    E::Dict(entries.into_iter().map(|(key, value)| (key.into(), value)).collect())
}

fn list(key: &str, values: Vec<E>) -> E {
    dict([(key, E::Array(values))])
}

fn properties(source: &E, keys: &[(&str, &str)]) -> E {
    E::Dict(keys.iter().filter_map(|&(name, id)| Some((id.into(), source.get(name)?.clone()))).collect())
}

fn indexed_resources(values: Vec<E>) -> E {
    let indices = (0..values.len()).map(|i| dict([("0", E::Int(i as i64))])).collect();
    dict([("0", E::Array(values)), ("1", E::Array(indices))])
}

fn kinsoku_resources() -> E {
    let entry = |name: &str, strict: bool| {
        let mut rules = E::Dict(HARD_KINSOKU_RULES.into_iter().map(|(_, key, text)| {
            (key.into(), E::String(if strict { text.into() } else { String::new() }))
        }).collect());
        rules.set("4", E::Int(i64::from(strict)));
        dict([("0", dict([("0", E::String(name.into())), ("5", rules)]))])
    };
    indexed_resources(vec![entry("None", false), entry("Hard", true)])
}

fn mojikumi_resources() -> E {
    // Photoshop's built-in presets refer to an engine spacing table, not to a duplicated table.
    indexed_resources(
        [(4, 2), (3, 4), (2, 3), (1, 1)]
            .into_iter()
            .map(|(name, preset)| {
                dict([("0", dict([("0", E::String(format!("Photoshop6MojiKumiSet{name}"))), ("5", dict([("0", E::Int(0)), ("3", E::Int(preset))]))]))])
            })
            .collect(),
    )
}

fn font_entry(name: String) -> E {
    // Same CoolType font reference used by PhotoCraft's native Txt2 writer upstream (#10313).
    dict([("0", dict([("99", E::Name("CoolTypeFont".into())), ("0", dict([("0", E::String(name)), ("2", E::Int(0))]))]))])
}

fn paint(color: &E) -> Option<E> {
    Some(dict([("99", E::Name("SimplePaint".into())), ("0", dict([("0", color.get("Type")?.clone()), ("1", color.get("Values")?.clone())]))]))
}

fn tagged_run(field: &str, default: &str, value: E, length: &E) -> E {
    let sheet = dict([("0", E::String(String::new())), (field, value), (default, E::Int(0))]);
    dict([("0", dict([("0", sheet)])), ("1", length.clone())])
}

fn default_sheet(field: &str) -> E {
    // Runs store the layer's actual styles. The base sheet only resolves their parent reference.
    indexed_resources(vec![dict([("0", dict([("0", E::String("Normal".into())), (field, E::dict())]))])])
}

fn document_preferences() -> E {
    // Neutral text document settings. Photoshop requires them before it reads the text objects.
    dict([
        ("1", E::Int(0)),
        ("2", E::Int(0)),
        ("3", real(0.0)),
        ("4", real(0.0)),
        ("5", real(0.0)),
        ("6", real(0.0)),
        ("7", real(0.0)),
        ("8", E::Bool(true)),
        ("16", E::Bool(true)),
        ("17", E::Int(0)),
    ])
}

fn frame(child: &E) -> Option<E> {
    let photoshop = child.path(&["Cookie", "Photoshop"])?;
    let vertical = child.get("Procession").and_then(E::as_i64) == Some(1);
    let mut info = dict([("6", E::Array(vec![real(-1.0), real(-1.0)])), ("11", dict([("4", E::Int(-1)), ("7", E::Bool(false))]))]);
    let mut entry = E::dict();
    match photoshop.get("ShapeType").and_then(E::as_i64)? {
        0 => {
            info.set("0", E::Int(0));
            entry.set("0", photoshop.get("PointBase")?.clone());
        }
        1 => {
            let [left, top, right, bottom] = photoshop.get("BoxBounds")?.as_array()? else { return None };
            let corners = [(left, top), (right, top), (right, bottom), (left, bottom)];
            let path = corners
                .iter()
                .zip(corners.iter().cycle().skip(1))
                .flat_map(|(&(x0, y0), &(x1, y1))| [x0, y0, x0, y0, x1, y1, x1, y1].map(Clone::clone))
                .collect();
            info.set("0", E::Int(1));
            info.set("6", E::Array(vec![real(-2.0), real(-2.0)]));
            info.set("10", dict([("0", E::Int(2)), ("1", real(0.0))]));
            info.set("11", dict([("4", E::Int(-2)), ("18", real(-2.0)), ("22", real(0.025))]));
            entry.set("1", list("0", path));
        }
        _ => return None,
    }
    if vertical {
        info.set("1", E::Int(2));
    }
    entry.set("2", info);
    Some(dict([("0", entry)]))
}

fn style_runs(engine: &E, fonts: &mut Vec<String>, layer: &TextLayer, needs_model: &mut bool) -> Option<Vec<E>> {
    let local_fonts = engine.path(&["ResourceDict", "FontSet"])?.as_array()?;
    let keys = [
        ("FontSize", "1"),
        ("FauxBold", "2"),
        ("FauxItalic", "3"),
        ("AutoLeading", "4"),
        ("Leading", "5"),
        ("HorizontalScale", "6"),
        ("VerticalScale", "7"),
        ("Tracking", "8"),
        ("BaselineShift", "9"),
        ("FontCaps", "12"),
        ("FontBaseline", "13"),
        ("Ligatures", "18"),
        ("DLigatures", "19"),
        ("BaselineDirection", "35"),
        ("Tsume", "36"),
        ("StyleRunAlignment", "37"),
        ("Language", "38"),
        ("NoBreak", "52"),
        ("FillFlag", "56"),
    ];
    let runs = engine.path(&["EngineDict", "StyleRun"])?;
    let runs = runs
        .get("RunArray")?
        .as_array()?
        .iter()
        .zip(runs.get("RunLengthArray")?.as_array()?)
        .map(|(row, length)| {
            let source = row.path(&["StyleSheet", "StyleSheetData"])?;
            *needs_model |= source.get("BaselineDirection").and_then(E::as_i64) == Some(TATE_CHU_YOKO)
                || source.get("NoBreak").and_then(E::as_bool) == Some(true);
            let local = usize::try_from(source.get("Font")?.as_i64()?).ok()?;
            let name = local_fonts.get(local)?.get("Name")?.as_str()?;
            let id = fonts.iter().position(|known| known == name).unwrap_or_else(|| {
                fonts.push(name.into());
                fonts.len() - 1
            });
            let mut style = properties(source, &keys);
            style.set("0", E::Int(id as i64));
            for (key, id) in [("Strikethrough", "15"), ("Underline", "16")] {
                if let Some(on) = source.get(key).and_then(E::as_bool) {
                    style.set(id, E::Int(i64::from(on)));
                }
            }
            style.set("53", paint(source.get("FillColor")?)?);
            if let Some(stroke) = source.get("StrokeColor").and_then(paint) {
                style.set("54", stroke);
            }
            Some(tagged_run("6", "5", style, length))
        })
        .collect::<Option<Vec<_>>>()?;
    with_modes(&runs, &txt2_mode_runs(layer))
}

fn paragraph_runs(engine: &E, has_strict: &mut bool) -> Option<Vec<E>> {
    let rules = engine.path(&["ResourceDict", "KinsokuSet"]).and_then(E::as_array).unwrap_or(&[]);
    let keys = [
        ("Justification", "0"),
        ("FirstLineIndent", "1"),
        ("StartIndent", "2"),
        ("EndIndent", "3"),
        ("SpaceBefore", "4"),
        ("SpaceAfter", "5"),
        ("AutoLeading", "7"),
        ("LeadingType", "8"),
        ("AutoHyphenate", "9"),
        ("HyphenatedWordSize", "10"),
        ("PreHyphen", "11"),
        ("PostHyphen", "12"),
        ("Zone", "14"),
        ("WordSpacing", "17"),
        ("LetterSpacing", "18"),
        ("GlyphSpacing", "19"),
        ("Hanging", "21"),
        ("KinsokuOrder", "25"),
        ("EveryLineComposer", "29"),
    ];
    let runs = engine.path(&["EngineDict", "ParagraphRun"])?;
    runs.get("RunArray")?
        .as_array()?
        .iter()
        .zip(runs.get("RunLengthArray")?.as_array()?)
        .map(|(row, length)| {
            let source = row.path(&["ParagraphSheet", "Properties"])?;
            let mut style = properties(source, &keys);
            let strict = source
                .get("Kinsoku")
                .and_then(E::as_i64)
                .and_then(|i| usize::try_from(i).ok())
                .and_then(|i| rules.get(i))
                .and_then(|r| r.get("Name"))
                .and_then(E::as_str)
                == Some(HARD_KINSOKU);
            *has_strict |= strict;
            style.set("27", E::Int(i64::from(strict)));
            if let Some(on) = source.get("Burasagari").and_then(E::as_bool) {
                style.set("24", E::Int(i64::from(on)));
            }
            // The East Asian engine (35) and its single/every-line option (29) are independent.
            if strict || source.get("MojiKumi").is_some() {
                style.set("35", E::Int(1));
            }
            if let Some(spacing @ 0..=3) = source.get("MojiKumi").and_then(E::as_i64) {
                style.set("28", E::Int(3 - spacing));
            }
            Some(tagged_run("5", "6", style, length))
        })
        .collect()
}

/// New PSDs with strict composition or vertical comic styles need a real text model.
/// Use the final `TySh` payloads so run lengths, styles, geometry and TextIndex stay in agreement.
pub fn build_composition_txt2<'a>(tysh: impl Iterator<Item = &'a [u8]>, layers: &[(i32, &TextLayer)]) -> Option<Vec<u8>> {
    let mut objects = BTreeMap::new();
    let by_index: BTreeMap<_, _> = layers.iter().copied().collect();
    let mut frames = Vec::new();
    let mut fonts = vec!["AdobeInvisFont".into()];
    let mut needs_model = false;
    for bytes in tysh {
        let ty = parse_tysh(bytes)?;
        let Some(D::Integer(index)) = ty.text.get("TextIndex") else { return None };
        if !(0..=MAX_TEXT_INDEX).contains(index) {
            return None;
        }
        let layer = by_index.get(index)?;
        let engine = engine_data(&ty.text)?;
        let text = engine.path(&["EngineDict", "Editor", "Text"])?.clone();
        let p = paragraph_runs(&engine, &mut needs_model)?;
        let s = style_runs(&engine, &mut fonts, layer, &mut needs_model)?;
        let child = engine.path(&["EngineDict", "Rendered", "Shapes", "Children"])?.as_array()?.first()?;
        let view = list("0", vec![dict([("0", E::Int(frames.len() as i64))])]);
        frames.push(frame(child)?);
        let model = dict([("0", text), ("5", list("0", p)), ("6", list("0", s))]);
        if objects.insert(*index, dict([("0", model), ("1", view)])).is_some() {
            return None;
        }
    }
    if !needs_model || objects.len() != by_index.len() {
        return None;
    }
    let last = *objects.last_key_value()?.0;
    let entries = (0..=last).map(|index| objects.remove(&index).unwrap_or_else(E::dict)).collect();
    // Photoshop reads resources in sequence. Styles must be available before the text frames.
    let resources = dict([
        ("1", list("0", fonts.into_iter().map(font_entry).collect())),
        ("2", indexed_resources(vec![dict([("0", dict([("0", E::String(String::new()))]))])])),
        ("3", mojikumi_resources()),
        ("4", kinsoku_resources()),
        ("5", default_sheet("6")),
        ("6", default_sheet("5")),
        ("8", list("0", frames)),
    ]);
    let document = dict([("0", document_preferences()), ("1", E::Array(entries))]);
    Some(ed::write_bare(&[("98".into(), dict([("0", E::Int(13))])), ("0".into(), resources), ("1".into(), document)]))
}
