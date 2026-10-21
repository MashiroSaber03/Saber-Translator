//! Window › Styles: layer style presets. A style is a list of layer effects plus, optionally,
//! the blending options it was saved with (blend mode, fill opacity). Applying one replaces the
//! selected layers' effects (Photoshop; ⇧-click adds instead: `"add": true`).
//!
//! The built-in groups are our own designs, built from the same effect params as
//! `layer.layerStyle.*`.

use photocraft_color::{BlendMode, ColorMode, SampleType};
use photocraft_doc::{Document, Effect, Layer, Size};
use photocraft_geom::Rect;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{Group, Named, always, bad, edit_groups, find, group_index, has_layer, req_str, str_param, unique_name};
use crate::commands::CommandSpec;
use crate::layer_style::effect_from_params;
use crate::{EngineError, Result, Session};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StylePreset {
    pub name: String,
    pub effects: Vec<Effect>,
    /// Blending options saved with the style ("Include Layer Blending Options").
    #[serde(default)]
    pub blend: Option<BlendMode>,
    #[serde(default)]
    pub fill_opacity: Option<f32>,
}

impl Named for StylePreset {
    fn name(&self) -> &str {
        &self.name
    }
    fn set_name(&mut self, n: String) {
        self.name = n;
    }
}

impl StylePreset {
    /// From `[[kind, params], …]` in `layer.layerStyle.<kind>` form.
    fn from_spec(name: &str, fx: Value, fill: Option<f32>) -> Self {
        let effects = fx
            .as_array()
            .map(|a| a.iter().filter_map(|e| effect_from_params(e.get(0)?.as_str()?, e.get(1).unwrap_or(&Value::Null))).collect())
            .unwrap_or_default();
        StylePreset { name: name.into(), effects, blend: None, fill_opacity: fill }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "effects": self.effects.iter().map(|e| e.label()).collect::<Vec<_>>(),
            "blend": self.blend.map(|b| b.label()),
            "fillOpacity": self.fill_opacity.map(|f| (f * 100.0).round()),
        })
    }
}

/// Built-in style groups (our own designs).
pub fn builtin() -> Vec<Group<StylePreset>> {
    let s = StylePreset::from_spec;
    vec![
        Group::new(
            "Basics",
            vec![
                s("Default Style (None)", json!([]), None),
                s("Drop Shadow", json!([["dropShadow", {"distance": 6, "size": 8, "opacity": 60}]]), None),
                s("Soft Shadow", json!([["dropShadow", {"distance": 2, "size": 16, "opacity": 45}]]), None),
                s("Black Stroke", json!([["stroke", {"size": 3, "color": "#000000", "position": "outside"}]]), None),
                s("White Stroke", json!([["stroke", {"size": 3, "color": "#ffffff", "position": "outside"}]]), None),
                s("Inner Shadow", json!([["innerShadow", {"distance": 4, "size": 6, "opacity": 55}]]), None),
                s("Outer Glow", json!([["outerGlow", {"size": 14, "color": "#ffe9a8", "opacity": 80}]]), None),
                s("Emboss", json!([["bevelEmboss", {"style": "emboss", "size": 4, "depth": 120}]]), None),
            ],
        ),
        Group::new(
            "Text Effects",
            vec![
                s(
                    "Neon Blue",
                    json!([["colorOverlay", {"color": "#d8f4ff"}], ["innerGlow", {"color": "#3fb6ff", "size": 6, "opacity": 90, "blend": "normal"}], ["outerGlow", {"color": "#1e90ff", "size": 22, "opacity": 90}]]),
                    None,
                ),
                s(
                    "Neon Pink",
                    json!([["colorOverlay", {"color": "#ffe1f3"}], ["innerGlow", {"color": "#ff4fb4", "size": 6, "opacity": 90, "blend": "normal"}], ["outerGlow", {"color": "#ff2d95", "size": 22, "opacity": 90}]]),
                    None,
                ),
                s(
                    "Chrome",
                    json!([["gradientOverlay", {"from": "#3a3f47", "to": "#f4f6f8", "angle": 90}], ["bevelEmboss", {"style": "inner", "size": 6, "depth": 250, "soften": 1}], ["stroke", {"size": 1, "color": "#2a2d33", "position": "outside"}]]),
                    None,
                ),
                s(
                    "Gold",
                    json!([["gradientOverlay", {"from": "#8a5a12", "to": "#ffe08a", "angle": 90}], ["bevelEmboss", {"style": "inner", "size": 5, "depth": 180}], ["dropShadow", {"distance": 3, "size": 5, "opacity": 50}]]),
                    None,
                ),
                s(
                    "Comic Outline",
                    json!([["colorOverlay", {"color": "#ffd23f"}], ["stroke", {"size": 4, "color": "#111111", "position": "outside"}], ["dropShadow", {"distance": 6, "size": 0, "opacity": 100, "angle": 135}]]),
                    None,
                ),
                s(
                    "Letterpress",
                    json!([["colorOverlay", {"color": "#5b6168"}], ["innerShadow", {"distance": 2, "size": 2, "opacity": 70}], ["dropShadow", {"distance": 1, "size": 0, "color": "#ffffff", "opacity": 60, "blend": "screen", "angle": 90}]]),
                    None,
                ),
                s("Hollow", json!([["stroke", {"size": 2, "color": "#222222", "position": "inside"}]]), Some(0.0)),
            ],
        ),
        Group::new(
            "Buttons",
            vec![
                s(
                    "Glass",
                    json!([["gradientOverlay", {"from": "#2b6cb0", "to": "#90cdf4", "angle": 90}], ["innerGlow", {"color": "#ffffff", "size": 6, "opacity": 45, "blend": "screen"}], ["bevelEmboss", {"style": "inner", "size": 8, "depth": 100, "soften": 4}], ["dropShadow", {"distance": 3, "size": 8, "opacity": 40}]]),
                    None,
                ),
                s(
                    "Gel Green",
                    json!([["gradientOverlay", {"from": "#1f7a3a", "to": "#7be495", "angle": 90}], ["innerShadow", {"distance": 2, "size": 5, "opacity": 35}], ["bevelEmboss", {"style": "inner", "size": 10, "depth": 150, "soften": 6}]]),
                    None,
                ),
                s(
                    "Pressed",
                    json!([["colorOverlay", {"color": "#d7dbe0"}], ["innerShadow", {"distance": 3, "size": 6, "opacity": 60}], ["stroke", {"size": 1, "color": "#9aa1a9", "position": "inside"}]]),
                    None,
                ),
                s(
                    "Pill Red",
                    json!([["gradientOverlay", {"from": "#9b1c1c", "to": "#f87171", "angle": 90}], ["bevelEmboss", {"style": "pillow", "size": 6, "depth": 120}], ["dropShadow", {"distance": 2, "size": 4, "opacity": 45}]]),
                    None,
                ),
                s("Flat Shadow", json!([["colorOverlay", {"color": "#3b82f6"}], ["dropShadow", {"distance": 5, "size": 0, "opacity": 35, "angle": 90}]]), None),
                s(
                    "Satin Plum",
                    json!([["colorOverlay", {"color": "#7e3a8c"}], ["satin", {"color": "#1b0420", "opacity": 55, "size": 18, "distance": 14}], ["bevelEmboss", {"style": "inner", "size": 4, "depth": 80}]]),
                    None,
                ),
            ],
        ),
    ]
}

pub fn find_style<'a>(s: &'a Session, p: &Value, cmd: &str) -> Result<&'a StylePreset> {
    let name = req_str(p, "preset", cmd)?;
    let (gi, ii) = find(&s.presets.styles, name, str_param(p, "group")).ok_or_else(|| bad(cmd, format!("no style \"{name}\" (see style.presets.list)")))?;
    Ok(&s.presets.styles[gi].items[ii])
}

fn list(s: &mut Session, _: &Value) -> Result<Value> {
    let groups: Vec<Value> =
        s.presets.styles.iter().map(|g| json!({"name": g.name, "presets": g.items.iter().map(StylePreset::to_json).collect::<Vec<_>>()})).collect();
    Ok(json!({"groups": groups}))
}

/// Target layers: `"layers": [ids]`, `"layer": id`, else every selected layer.
fn targets(s: &Session, p: &Value) -> Vec<photocraft_doc::LayerId> {
    if let Some(a) = p.get("layers").and_then(Value::as_array) {
        return a.iter().filter_map(Value::as_u64).map(photocraft_doc::LayerId).collect();
    }
    if let Some(id) = p.get("layer").and_then(Value::as_u64) {
        return vec![photocraft_doc::LayerId(id)];
    }
    crate::layer_multi_cmds::selected(s)
}

fn apply(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "style.presets.apply";
    let style = find_style(s, p, CMD)?.clone();
    let add = p.get("add").and_then(Value::as_bool).unwrap_or(false);
    let ids = targets(s, p);
    if ids.is_empty() {
        return Err(bad(CMD, "no target layers"));
    }
    // Pattern overlays name their pattern; resolve to library patterns copied into the document.
    let mut effects = Vec::new();
    let mut pats = Vec::new();
    for fx in style.effects {
        let (fx, pat) = crate::pattern_cmds::resolve_effect(s, fx)?;
        effects.push(fx);
        pats.extend(pat);
    }
    let label = format!("Apply Style: {}", style.name);
    s.edit(&label, |doc, _| {
        for pat in &pats {
            crate::pattern_cmds::ensure_in_doc(doc, pat);
        }
        // A preset's colours are RGB; the layer's are in the document's mode.
        let mode = doc.mode;
        let effects: Vec<_> = effects.iter().cloned().map(|e| e.in_mode(mode)).collect();
        for id in &ids {
            // Styles can't sit on the Background: Photoshop turns it into a normal layer first.
            crate::extra_cmds::background_to_layer(doc, *id);
            let l = doc.layer_mut(*id).ok_or(EngineError::NoLayer(*id))?;
            if add {
                l.effects.items.extend(effects.iter().cloned());
            } else {
                l.effects.items = effects.clone();
                l.effects.psd_raw = None;
            }
            l.effects.psd_raw = None;
            l.effects.enabled = true;
            if !add || style.blend.is_some() {
                if let Some(b) = style.blend {
                    l.blend = b;
                }
                if let Some(f) = style.fill_opacity {
                    l.fill_opacity = f;
                } else if !add {
                    l.fill_opacity = 1.0;
                }
            }
        }
        Ok(())
    })?;
    Ok(json!({"layers": ids.iter().map(|i| i.0).collect::<Vec<_>>(), "style": style.name}))
}

fn new_preset(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "style.presets.new";
    let id = crate::commands::layer_param(s, p)?;
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let l = d.doc.layer(id).ok_or(EngineError::NoLayer(id))?;
    let with_fx = p.get("includeEffects").and_then(Value::as_bool).unwrap_or(true);
    let with_blend = p.get("includeBlending").and_then(Value::as_bool).unwrap_or(true);
    if !with_fx && !with_blend {
        return Err(bad(CMD, "include effects and/or blending options"));
    }
    // An explicit effect list (the Layer Style dialog's pending state, in
    // `layer.layerStyle.<kind>` param form) replaces the layer's own effects.
    let listed = p
        .get("effects")
        .and_then(Value::as_array)
        .map(|list| -> Result<Vec<Effect>> {
            list.iter()
                .map(|e| {
                    let kind = e.get(0).and_then(Value::as_str).ok_or_else(|| bad(CMD, "each effect is [kind, params]"))?;
                    effect_from_params(kind, e.get(1).unwrap_or(&Value::Null)).ok_or_else(|| bad(CMD, format!("unknown effect {kind}")))
                })
                .collect()
        })
        .transpose()?;
    let style = StylePreset {
        name: unique_name(&s.presets.styles, str_param(p, "name").unwrap_or("Style")),
        effects: match listed {
            Some(fx) => fx,
            None => {
                if with_fx {
                    l.effects.items.clone()
                } else {
                    Vec::new()
                }
            }
        },
        blend: p.get("blend").and_then(Value::as_str).and_then(crate::commands::blend_from_str).or_else(|| with_blend.then_some(l.blend)),
        fill_opacity: p.get("fillOpacity").and_then(Value::as_f64).map(|v| (v / 100.0).clamp(0.0, 1.0) as f32).or_else(|| with_blend.then_some(l.fill_opacity)),
    };
    let gi = group_index(&mut s.presets.styles, str_param(p, "group"));
    let out = json!({"name": style.name, "group": s.presets.styles[gi].name, "effects": style.effects.len()});
    s.presets.styles[gi].items.push(style);
    s.presets_changed();
    Ok(out)
}

fn edit(s: &mut Session, p: &Value) -> Result<Value> {
    let action = req_str(p, "action", "style.presets.edit")?.to_string();
    let r = edit_groups(&mut s.presets.styles, &action, p, "style.presets.edit")?;
    s.presets_changed();
    Ok(r)
}

fn reset(s: &mut Session, p: &Value) -> Result<Value> {
    let mut b = builtin();
    if p.get("append").and_then(Value::as_bool).unwrap_or(false) {
        for g in &mut b {
            g.name = super::gradients::unique_group(&s.presets.styles, &g.name);
        }
        s.presets.styles.extend(b);
    } else {
        s.presets.styles = b;
    }
    s.presets_changed();
    Ok(json!({"groups": s.presets.styles.len()}))
}

/// Swatch for the Styles panel: the style on a rounded square, `size`² RGBA8 (straight alpha,
/// transparent around the square).
pub fn thumbnail(s: &Session, style: &StylePreset, size: u32) -> Vec<u8> {
    let size = size.clamp(8, 256);
    let mut doc = Document::new("style", Size::new(size, size), ColorMode::Rgb, SampleType::U8);
    let fmt = doc.pixel_format();
    let mut l = Layer::raster("swatch", fmt);
    let m = (size as f32 * 0.2).round();
    let (x0, x1) = (m, size as f32 - m);
    let r = size as f32 * 0.12;
    if let Some(surf) = l.surface_mut() {
        let mut px = Vec::with_capacity((size * size * 4) as usize);
        for y in 0..size {
            for x in 0..size {
                // Coverage of a rounded square (1 px anti-aliasing).
                let (cx, cy) = (x as f32 + 0.5, y as f32 + 0.5);
                let qx = (cx - (x0 + x1) / 2.0).abs() - ((x1 - x0) / 2.0 - r);
                let qy = (cy - (x0 + x1) / 2.0).abs() - ((x1 - x0) / 2.0 - r);
                let d = (qx.max(0.0).hypot(qy.max(0.0)) + qx.max(qy).min(0.0)) - r;
                let a = (0.5 - d).clamp(0.0, 1.0);
                px.extend_from_slice(&[0.86, 0.86, 0.86, a]);
            }
        }
        surf.write_region(Rect::new(0, 0, size as i32, size as i32), &px);
    }
    for fx in &style.effects {
        if let Ok((fx, pat)) = crate::pattern_cmds::resolve_effect(s, fx.clone()) {
            if let Some(p) = pat {
                crate::pattern_cmds::ensure_in_doc(&mut doc, &p);
            }
            l.effects.items.push(fx);
        }
    }
    l.effects.enabled = true;
    if let Some(b) = style.blend {
        l.blend = b;
    }
    if let Some(f) = style.fill_opacity {
        l.fill_opacity = f;
    }
    doc.layers.push(l);
    photocraft_compose::render(&doc, doc.bounds()).to_rgba8().pixels
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "style.presets.list",
            label: "Style Presets",
            menu: &[],
            shortcut: None,
            params: "{} → {groups:[{name,presets:[{name,effects,blend,fillOpacity}]}]}",
            enabled: always,
            run: list,
            journal: false,
        },
        CommandSpec {
            id: "style.presets.apply",
            label: "Apply Style",
            menu: &[],
            shortcut: None,
            params: r##"{"preset":name,"group":name?,"layers":[ids]?|"layer":id? (default: the selected layers),"add":bool=false (⇧-click: add to the existing effects)} → {layers,style}"##,
            enabled: has_layer,
            run: apply,
            journal: true,
        },
        CommandSpec {
            id: "style.presets.new",
            label: "New Style…",
            menu: &[],
            shortcut: None,
            params: r##"{"name":str="Style","group":name?,"layer":id?,"includeEffects":bool=true,"includeBlending":bool=true,"effects":[[kind, params]]? (explicit list, e.g. the Layer Style dialog's pending state; overrides includeEffects),"blend":str?,"fillOpacity":0..100? (override the layer's blending options)}"##,
            enabled: has_layer,
            run: new_preset,
            journal: true,
        },
        CommandSpec {
            id: "style.presets.edit",
            label: "Edit Style Presets",
            menu: &[],
            shortcut: None,
            params: super::GROUP_EDIT_PARAMS,
            enabled: always,
            run: edit,
            journal: true,
        },
        CommandSpec {
            id: "style.presets.reset",
            label: "Restore Default Styles",
            menu: &[],
            shortcut: None,
            params: r##"{"append":bool=false}"##,
            enabled: always,
            run: reset,
            journal: true,
        },
    ]
}
