//! Filter › Filter Gallery… (`filter.filterGallery`) and one command per gallery filter
//! (`filter.gallery.<key>`, not in menus).
//!
//! `filter.filterGallery {effects:[{filter, params, visible}]}` applies a stack of effect layers
//! in order (the gallery dialog's effect list, bottom to top). Like every filter it goes through
//! [`crate::filters::run_filter`]: selection, smart objects (recorded as a smart filter),
//! history, channels and `filter.lastFilter` all work. The foreground/background colours the
//! Sketch filters read are recorded as explicit params, so smart filters re-render identically.

use photocraft_algo::{FilterParams, GalleryEffect, GalleryFilter, GalleryParamKind};
use serde_json::{Map, Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

/// Straight RGBA colour from `[r, g, b(, a)]` (0–1) or `#rrggbb`.
fn colour(v: Option<&Value>) -> Option<[f32; 4]> {
    match v? {
        Value::Array(a) if a.len() >= 3 => {
            let c = |j: usize, d: f32| a.get(j).and_then(Value::as_f64).map_or(d, |v| v as f32);
            Some([c(0, 0.0), c(1, 0.0), c(2, 0.0), c(3, 1.0)])
        }
        Value::String(h) if h.len() == 7 && h.starts_with('#') => {
            let ch = |j: usize| h.get(j..j + 2).and_then(|s| u8::from_str_radix(s, 16).ok()).map_or(0.0, |v| v as f32 / 255.0);
            Some([ch(1), ch(3), ch(5), 1.0])
        }
        _ => None,
    }
}

/// One effect from its params object (`fallback` supplies colours not given per effect).
pub fn effect_from_json(filter: GalleryFilter, p: &Value, fallback: &Value) -> GalleryEffect {
    let mut e = GalleryEffect::new(filter);
    for prm in filter.params() {
        let Some(v) = p.get(prm.key) else { continue };
        match (&prm.kind, v) {
            (GalleryParamKind::Choice(_), Value::String(name)) => {
                e.set_choice(prm.key, name);
            }
            (_, Value::Bool(b)) => e.set(prm.key, *b as u8 as f32),
            (_, v) => {
                if let Some(n) = v.as_f64() {
                    e.set(prm.key, n as f32);
                }
            }
        }
    }
    let pick = |k: &str| colour(p.get(k)).or_else(|| colour(fallback.get(k)));
    if let Some(c) = pick("foreground") {
        e.foreground = c;
    }
    if let Some(c) = pick("background") {
        e.background = c;
    }
    if let Some(c) = pick("glowColor") {
        e.color = c;
    }
    e
}

/// The effect stack of a `filter.filterGallery` params object. Entries are
/// `{"filter": "<key>", "params": {...}, "visible": bool}` (params may also sit beside
/// `filter`); hidden entries are skipped.
pub fn effects_from_json(p: &Value) -> std::result::Result<Vec<GalleryEffect>, String> {
    let Some(list) = p.get("effects").and_then(Value::as_array) else { return Ok(Vec::new()) };
    if list.len() > 32 {
        return Err("too many effects in filterGallery stack (max 32)".into());
    }
    let mut out = Vec::new();
    for (k, item) in list.iter().enumerate() {
        if item.get("visible").and_then(Value::as_bool) == Some(false) {
            continue;
        }
        let key = item.get("filter").and_then(Value::as_str).ok_or_else(|| format!("effect {k}: missing `filter`"))?;
        let f = GalleryFilter::from_key(key).ok_or_else(|| format!("effect {k}: unknown gallery filter `{key}`"))?;
        let params = item.get("params").filter(|v| v.is_object()).unwrap_or(item);
        out.push(effect_from_json(f, params, p));
    }
    Ok(out)
}

/// Algorithm parameters for the ids this module owns.
pub fn params_for(id: &str, p: &Value) -> Option<FilterParams> {
    if id == "filter.filterGallery" {
        return Some(FilterParams::FilterGallery { effects: effects_from_json(p).unwrap_or_default() });
    }
    let f = id.strip_prefix("filter.gallery.").and_then(GalleryFilter::from_key)?;
    Some(FilterParams::FilterGallery { effects: vec![effect_from_json(f, p, &Value::Null)] })
}

/// An unknown choice name or a wrong-typed choice is a bad-params error, for one gallery filter
/// or each effect of a `filter.filterGallery` stack. Checked only for new invocations (via
/// `filters::validate_params`); stored smart filters keep decoding tolerantly.
pub(crate) fn validate_params(id: &str, p: &Value) -> Result<()> {
    let bad = |msg: String| EngineError::BadParams { cmd: id.into(), msg };
    if id == "filter.filterGallery" {
        for (k, item) in p.get("effects").and_then(Value::as_array).into_iter().flatten().enumerate() {
            let Some(f) = item.get("filter").and_then(Value::as_str).and_then(GalleryFilter::from_key) else { continue };
            let params = item.get("params").filter(|v| v.is_object()).unwrap_or(item);
            check_choices(f, params).map_err(|m| bad(format!("effect {k}: {m}")))?;
        }
        return Ok(());
    }
    match id.strip_prefix("filter.gallery.").and_then(GalleryFilter::from_key) {
        Some(f) => check_choices(f, p).map_err(bad),
        None => Ok(()),
    }
}

fn check_choices(f: GalleryFilter, p: &Value) -> std::result::Result<(), String> {
    for prm in f.params() {
        let (Some(v), GalleryParamKind::Choice(names)) = (p.get(prm.key), &prm.kind) else { continue };
        let ok = match v {
            Value::String(name) => names.iter().any(|n| n.eq_ignore_ascii_case(name)),
            // A choice index, clamped as the dialog stores it.
            Value::Number(_) => true,
            _ => false,
        };
        if !ok {
            return Err(format!("`{}` must be one of {}", prm.key, names.join("|")));
        }
    }
    Ok(())
}

/// Whether the command reads the foreground/background colours.
pub(crate) fn uses_colours(id: &str) -> bool {
    id == "filter.filterGallery" || id.strip_prefix("filter.gallery.").and_then(GalleryFilter::from_key).is_some_and(GalleryFilter::uses_colours)
}

/// The gallery catalogue for dialogs and agents: categories, filters and parameter notation.
pub fn catalogue() -> Value {
    let filters: Vec<Value> = GalleryFilter::ALL
        .iter()
        .map(|f| json!({"key": f.key(), "name": f.name(), "category": f.category(), "command": f.command_id(), "params": f.params_doc()}))
        .collect();
    json!({"categories": photocraft_algo::GALLERY_CATEGORIES, "filters": filters})
}

fn run_gallery(s: &mut Session, p: &Value) -> Result<Value> {
    if p.get("list").and_then(Value::as_bool) == Some(true) {
        return Ok(catalogue());
    }
    let effects = effects_from_json(p).map_err(EngineError::Other)?;
    if effects.is_empty() {
        return Err(EngineError::Other(
            "filter.filterGallery needs `effects`: [{\"filter\": \"<key>\", \"params\": {...}}] (pass {\"list\": true} for the catalogue)".into(),
        ));
    }
    // Normalize the recorded params (journal / smart filter) to the canonical form.
    let mut q: Map<String, Value> = p.as_object().cloned().unwrap_or_default();
    q.remove("list");
    crate::filters::run_filter(s, "filter.filterGallery", &Value::Object(q))
}

macro_rules! gallery_specs {
    ($($v:ident),* $(,)?) => {
        vec![$(
            CommandSpec {
                id: GalleryFilter::$v.command_id(),
                label: GalleryFilter::$v.name(),
                menu: &[],
                shortcut: None,
                params: GalleryFilter::$v.params_doc(),
                enabled: crate::filters::has_filterable_layer,
                run: |s, p| crate::filters::run_filter(s, GalleryFilter::$v.command_id(), p),
                journal: true,
            }
        ),*]
    };
}

/// The command specs.
pub fn specs() -> Vec<CommandSpec> {
    let mut v = vec![CommandSpec {
        id: "filter.filterGallery",
        label: "Filter Gallery…",
        menu: &["Filter"],
        shortcut: None,
        params: r##"{"effects":json,"foreground":json,"background":json,"list":bool}"##,
        enabled: crate::filters::has_filterable_layer,
        run: run_gallery,
        journal: true,
    }];
    v.extend(gallery_specs![
        ColoredPencil,
        Cutout,
        DryBrush,
        FilmGrain,
        Fresco,
        NeonGlow,
        PaintDaubs,
        PaletteKnife,
        PlasticWrap,
        PosterEdges,
        RoughPastels,
        SmudgeStick,
        Sponge,
        Underpainting,
        Watercolor,
        AccentedEdges,
        AngledStrokes,
        Crosshatch,
        DarkStrokes,
        InkOutlines,
        Spatter,
        SprayedStrokes,
        SumiE,
        DiffuseGlow,
        Glass,
        OceanRipple,
        BasRelief,
        ChalkCharcoal,
        Charcoal,
        Chrome,
        ConteCrayon,
        GraphicPen,
        HalftonePattern,
        NotePaper,
        Photocopy,
        Plaster,
        Reticulation,
        Stamp,
        TornEdges,
        WaterPaper,
        GlowingEdges,
        Craquelure,
        Grain,
        MosaicTiles,
        Patchwork,
        StainedGlass,
        Texturizer,
    ]);
    v
}

#[cfg(test)]
mod tests;
