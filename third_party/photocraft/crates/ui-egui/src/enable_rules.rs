//! Photoshop menu enablement the engine's per-command checks don't express: the Background layer
//! (no styles, masks, clipping, arranging, transforms without a selection), layer-kind specific
//! items (Rasterize ›, Combine Shapes ›), and single-layer documents (Merge Visible, Flatten Image).
//! Applied on top of `menus::is_enabled`, so menus, shortcuts and the Layers context menu grey the
//! same items Photoshop greys.

use photocraft_doc::{Document, Layer, LayerContent, LayerId};

/// Commands Photoshop greys while the Background layer is the active layer.
fn background_blocks(id: &str, has_selection: bool) -> bool {
    let prefixes = ["layer.layerMask.", "layer.vectorMask.", "layer.arrange.", "layer.matting."];
    if prefixes.iter().any(|p| id.starts_with(p)) {
        return true;
    }
    if id.starts_with("layer.layerStyle.") {
        // Global Light and Hide All Effects are document-wide.
        return !matches!(id, "layer.layerStyle.globalLight" | "layer.layerStyle.hideAllEffects" | "layer.layerStyle.copyLayerStyle");
    }
    // Transforms need a selection on the Background (they then move the selected pixels).
    if !has_selection && (id == "edit.freeTransform" || (id.starts_with("edit.transform.") && id != "edit.transform.again")) {
        return true;
    }
    matches!(
        id,
        "layer.createClippingMask"
            | "layer.releaseClippingMask"
            | "layer.mergeDown"
            | "layer.lockLayers"
            | "layer.maskAllObjects"
            | "edit.contentAwareScale"
            | "edit.puppetWarp"
            | "edit.perspectiveWarp"
    )
}

/// Commands that only apply to some layer kinds.
fn kind_blocks(id: &str, l: &Layer) -> bool {
    let c = &l.content;
    match id {
        "layer.rasterize.type" | "layer.rasterize.rasterizeType" => !matches!(c, LayerContent::Text(_)),
        "layer.rasterize.shape" | "layer.rasterize.rasterizeShape" => !matches!(c, LayerContent::Shape(_)),
        "layer.rasterize.fillContent" => !matches!(c, LayerContent::Fill(_)),
        "layer.rasterize.smartObject" => !matches!(c, LayerContent::Smart(_)),
        "layer.rasterize.vectorMask" => l.vector_mask.is_none(),
        // What the command converts: type, shape, fill and Smart Object layers, and vector masks
        // ("all vector data"). Effects stay live: Rasterize › Layer Style bakes them.
        "layer.rasterize.layer" => {
            l.vector_mask.is_none() && !matches!(c, LayerContent::Text(_) | LayerContent::Shape(_) | LayerContent::Fill(_) | LayerContent::Smart(_))
        }
        "layer.rasterize.layerStyle" => !photocraft_engine::rasterize_style_cmds::can_bake(l),
        "layer.layerStyle.copyLayerStyle" | "layer.layerStyle.clear" => l.effects.items.is_empty(),
        _ => id.starts_with("layer.combineShapes.") && !matches!(c, LayerContent::Shape(_)),
    }
}

/// Is `id` greyed in Photoshop for this document state, regardless of the engine's own check?
pub fn disabled_for(doc: &Document, active: Option<&Layer>, id: &str) -> bool {
    let only_background = doc.layers.len() == 1 && doc.layers.first().is_some_and(|b| crate::doc_props_ui::is_background(doc, b));
    if only_background && matches!(id, "layer.mergeVisible" | "layer.flattenImage" | "layer.mergeLayers" | "layer.rasterize.allLayers") {
        return true;
    }
    if id == "layer.delete.hiddenLayers" && doc.walk().iter().all(|(_, _, l)| l.visible) {
        return true;
    }
    let Some(l) = active else { return false };
    if crate::doc_props_ui::is_background(doc, l) && background_blocks(id, doc.selection.is_some()) {
        return true;
    }
    kind_blocks(id, l)
}

/// How a command that acts on the whole layer selection decides it is available.
#[derive(Clone, Copy)]
enum Targets {
    /// Some selected layer can take it.
    Any,
    /// Some selected layer other than the lowest can (Create Clipping Mask clips every selected layer
    /// but the lowest, which becomes the base).
    AllButLowest,
    /// Some selected layer without a layer mask can take one.
    WithoutMask,
}

/// The commands that run on every selected layer, and so are judged by the selection and not by
/// the active layer (the same selection used to grey or enable them depending on which of its
/// layers happened to be active).
fn selection_targets(id: &str) -> Option<Targets> {
    Some(match id {
        "layer.rasterize.layer"
        | "layer.rasterize.layerStyle"
        | "layer.rasterize.type"
        | "layer.rasterize.shape"
        | "layer.rasterize.fillContent"
        | "layer.rasterize.smartObject" => Targets::Any,
        "layer.releaseClippingMask" | "layer.layerStyle.pasteLayerStyle" | "layer.layerStyle.clear" => Targets::Any,
        "layer.createClippingMask" => Targets::AllButLowest,
        "layer.layerMask.revealAll" | "layer.layerMask.hideAll" | "layer.layerMask.revealSelection" | "layer.layerMask.hideSelection" => Targets::WithoutMask,
        _ => return None,
    })
}

/// [`disabled_for`] with the layer selection in mind: with several layers selected, a command that
/// runs on all of them is greyed only when none of them can take it (the active layer alone must
/// not decide). Everything else, and a single selected layer, is judged by the active layer.
pub fn disabled_for_selection(doc: &Document, selected: &[LayerId], active: Option<&Layer>, id: &str) -> bool {
    let Some(rule) = selection_targets(id).filter(|_| selected.len() > 1) else { return disabled_for(doc, active, id) };
    let layers = selected.iter().filter_map(|l| doc.layer(*l));
    let can = |l: &Layer| !disabled_for(doc, Some(l), id) && (!matches!(rule, Targets::WithoutMask) || l.mask.is_none());
    let any = match rule {
        Targets::AllButLowest => layers.skip(1).any(&can),
        Targets::Any | Targets::WithoutMask => layers.into_iter().any(can),
    };
    !any
}

pub fn disabled(app: &crate::PhotocraftApp, id: &str) -> bool {
    let Some(st) = app.session.active() else { return false };
    let active = st.active_layer.and_then(|a| st.doc.layer(a));
    // Free Transform Path works on the Background too: it moves the path, not the pixels.
    let transform = id == "edit.freeTransform" || (id.starts_with("edit.transform.") && id != "edit.transform.again");
    if transform && crate::vector_ui::free_transform_path(app).is_some() {
        return false;
    }
    // So does a targeted alpha channel or the Quick Mask: it transforms instead of the layer (#2835).
    let target = crate::canvas::paint_target(app);
    if transform && (target.is_object() || target == "quickMask") {
        return false;
    }
    // Editable type can be scaled, rotated and skewed but not distorted (#2630): Photoshop greys
    // Distort and Perspective until the type is rasterized, and the split warps (Warp on type
    // offers only the warp styles, #3004). A targeted unlinked mask or channel of the type layer
    // is pixels, and can be.
    if matches!(
        id,
        "edit.transform.distort"
            | "edit.transform.perspective"
            | "edit.transform.splitWarpCrosswise"
            | "edit.transform.splitWarpHorizontally"
            | "edit.transform.splitWarpVertically"
            | "edit.transform.removeWarpSplit"
    ) && active.is_some_and(|l| matches!(l.content, LayerContent::Text(_)))
        && photocraft_engine::transform_cmds::lone_target(&st.doc, st.active_layer, &serde_json::json!({ "target": crate::canvas::paint_target(app) }))
            .is_ok_and(|t| t.is_none())
    {
        return true;
    }
    // Only these commands depend on the selection; skip walking the layers for the rest.
    if selection_targets(id).is_some() {
        return disabled_for_selection(&st.doc, &st.selected_layers(), active, id);
    }
    disabled_for(&st.doc, active, id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn session() -> photocraft_engine::Session {
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 20, "height": 20})).unwrap();
        s
    }

    fn off(s: &photocraft_engine::Session, id: &str) -> bool {
        let st = s.active().unwrap();
        disabled_for(&st.doc, st.active_layer.and_then(|a| st.doc.layer(a)), id)
    }

    #[test]
    fn background_greys_styles_masks_and_transforms() {
        let mut s = session();
        for id in [
            "layer.layerStyle.dropShadow",
            "layer.layerMask.revealAll",
            "layer.arrange.bringForward",
            "layer.mergeDown",
            "edit.freeTransform",
            "edit.transform.scale",
            "layer.flattenImage",
            "layer.mergeVisible",
            "layer.createClippingMask",
            "layer.rasterize.layer",
            "layer.combineShapes.unite",
            "layer.delete.hiddenLayers",
        ] {
            assert!(off(&s, id), "{id} should be greyed on a lone Background");
        }
        assert!(!off(&s, "layer.layerStyle.globalLight"));
        // A selection makes Free Transform available on the Background.
        s.execute("select.all", json!({})).unwrap();
        assert!(!off(&s, "edit.freeTransform"));
        // So does a targeted alpha channel, which transforms instead of the Background (#2835).
        s.execute("select.deselect", json!({})).unwrap();
        s.execute("channel.new", json!({})).unwrap();
        s.execute("channel.target", json!({"channel": 0})).unwrap();
        let app = crate::PhotocraftApp::new(s, crate::Services::default());
        assert!(!disabled(&app, "edit.freeTransform") && !disabled(&app, "edit.transform.flipHorizontal"));
    }

    /// With several layers selected, Rasterize is judged by the whole selection, not by the layer
    /// that happens to be active (or was right-clicked).
    #[test]
    fn rasterize_follows_the_whole_selection() {
        let mut s = session();
        s.execute("layer.new.layer", json!({})).unwrap();
        let raster = s.active().unwrap().active_layer.unwrap();
        let text = photocraft_doc::LayerId(s.execute("type.create", json!({"text": "Hi", "size": 12, "x": 2, "y": 12})).unwrap()["layer"].as_u64().unwrap());
        let sel = |s: &photocraft_engine::Session, ids: &[photocraft_doc::LayerId], id: &str| {
            let st = s.active().unwrap();
            disabled_for_selection(&st.doc, ids, ids.last().and_then(|a| st.doc.layer(*a)), id)
        };
        // The raster layer is active, but the type layer in the selection can be rasterized.
        let ids = [text, raster];
        assert!(!sel(&s, &ids, "layer.rasterize.layer"));
        assert!(!sel(&s, &ids, "layer.rasterize.type"));
        assert!(sel(&s, &ids, "layer.rasterize.shape"), "no shape layer is selected");
        assert!(sel(&s, &ids, "layer.rasterize.smartObject"));
        // Nothing selected can be rasterized: greyed, as for a single raster layer.
        let bg = s.active().unwrap().doc.layers[0].id;
        assert!(sel(&s, &[bg, raster], "layer.rasterize.layer"));
        // One selected layer is judged by itself, and other commands are not affected.
        assert!(sel(&s, &[raster], "layer.rasterize.layer") && !sel(&s, &[text], "layer.rasterize.layer"));
        let st = s.active().unwrap();
        let active = ids.last().and_then(|a| st.doc.layer(*a));
        for id in ["layer.layerStyle.dropShadow", "layer.mergeDown", "layer.rasterize.vectorMask"] {
            assert_eq!(sel(&s, &ids, id), disabled_for(&st.doc, active, id), "{id}: decided by the active layer, as before");
        }
    }

    /// A document with a Background, plain layers, one with a mask, one with a vector mask, a type layer, a shape, a Smart
    /// Object, an adjustment layer and one with a drop shadow; `select` picks a selection (the last
    /// layer is active) and `copied` has a layer style on the clipboard.
    fn world(select: &[&str], copied: bool) -> photocraft_engine::Session {
        use photocraft_doc::LayerId;
        let mut s = session();
        let id = |r: serde_json::Value| LayerId(r["layer"].as_u64().unwrap());
        let mut ids = std::collections::HashMap::new();
        ids.insert("BG", s.active().unwrap().doc.layers[0].id);
        ids.insert("R1", id(s.execute("layer.new.layer", json!({"name": "R1"})).unwrap()));
        ids.insert("R2", id(s.execute("layer.new.layer", json!({"name": "R2"})).unwrap()));
        s.execute("layer.layerMask.revealAll", json!({})).unwrap();
        ids.insert("V", id(s.execute("layer.new.layer", json!({"name": "V"})).unwrap()));
        s.execute("layer.vectorMask.revealAll", json!({})).unwrap();
        ids.insert("T", id(s.execute("type.create", json!({"text": "T", "size": 12, "x": 2, "y": 12})).unwrap()));
        ids.insert("S", id(s.execute("shape.create", json!({"kind": "rect", "rect": [4, 4, 6, 6]})).unwrap()));
        s.execute("layer.new.layer", json!({"name": "M"})).unwrap();
        s.execute("paint.stroke", json!({"points": [[10, 10]], "size": 4})).unwrap();
        ids.insert("M", id(s.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap()));
        ids.insert("A", id(s.execute("layer.newAdjustmentLayer.hueSaturation", json!({"hue": 10})).unwrap()));
        ids.insert("FX", id(s.execute("layer.new.layer", json!({"name": "FX"})).unwrap()));
        s.execute("layer.layerStyle.dropShadow", json!({})).unwrap();
        if copied {
            s.execute("layer.layerStyle.copyLayerStyle", json!({"layer": ids["FX"].0})).unwrap();
        }
        s.execute("layer.select", json!({"layer": ids[select[0]].0})).unwrap();
        for name in &select[1..] {
            s.execute("layer.select", json!({"layer": ids[name].0, "mode": "add"})).unwrap();
        }
        s
    }

    /// What a layer command can change: kind, clipping, masks and style of every layer.
    fn fingerprint(s: &photocraft_engine::Session) -> Vec<String> {
        s.active()
            .unwrap()
            .doc
            .walk()
            .iter()
            .map(|(_, _, l)| {
                format!("{:?}|{}|{}|{}|{}", std::mem::discriminant(&l.content), l.clipped, l.mask.is_some(), l.vector_mask.is_some(), l.effects.items.len())
            })
            .collect()
    }

    /// The menu item and the command agree for every selection, whichever of its layers is active:
    /// an enabled item runs, a greyed one would have changed nothing. (The same selection used to be
    /// greyed or not depending on which layer was active, and Rasterize Layer was enabled for layers
    /// it then refused.)
    #[test]
    fn enabled_commands_run_and_greyed_ones_would_do_nothing() {
        // (command, an enabled item must really change something)
        let commands = [
            ("layer.rasterize.layer", true),
            ("layer.rasterize.layerStyle", true),
            ("layer.createClippingMask", true),
            ("layer.layerMask.revealAll", true),
            ("layer.layerMask.hideAll", true),
            ("layer.layerStyle.pasteLayerStyle", true),
            ("layer.layerStyle.clear", true),
            ("layer.releaseClippingMask", false),
        ];
        let selections: &[&[&str]] = &[
            &["R1", "R2"],
            &["R2", "R1"],
            &["BG", "R1"],
            &["R1", "BG"],
            &["T", "S"],
            &["S", "T"],
            &["M", "R1"],
            &["R1", "M"],
            &["BG", "R1", "M", "A"],
            &["R1", "A"],
            &["A", "R1"],
            &["FX", "R1"],
            &["R1", "FX"],
            &["BG", "FX", "T", "S", "M", "A"],
            &["R2", "A"],
            &["BG", "R2"],
            &["T"],
            &["R1"],
            &["BG"],
            &["FX"],
            &["V", "R1"],
            &["R1", "V"],
            &["V"],
            &["V", "FX", "T"],
        ];
        let mut checked = 0;
        for sel in selections {
            for copied in [false, true] {
                for (cmd, must_change) in commands {
                    let app = crate::PhotocraftApp::new(world(sel, copied), crate::Services::default());
                    let enabled = crate::menus::is_enabled(&app, cmd);
                    let mut s = app.session;
                    let before = fingerprint(&s);
                    let r = s.execute(cmd, json!({}));
                    let changed = fingerprint(&s) != before;
                    let what = format!(
                        "{cmd} for {sel:?} (copied style: {copied}), enabled={enabled}, result={:?}",
                        r.as_ref().map(|_| ()).map_err(|e| e.to_string())
                    );
                    if enabled {
                        assert!(r.is_ok(), "an enabled item must run: {what}");
                        assert!(changed || !must_change || sel.len() < 2, "an enabled item must do something: {what}");
                    } else if sel.len() > 1 {
                        // For one layer the greying is the menu's own policy (the engine, for instance,
                        // lets a script clip the Background); several layers must be exact.
                        assert!(r.is_err() || !changed, "a greyed item would have changed something: {what}");
                    }
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, selections.len() * 2 * commands.len());
    }

    #[test]
    fn ordinary_layers_keep_their_items() {
        let mut s = session();
        s.execute("layer.new.layer", json!({})).unwrap();
        for id in ["layer.layerStyle.dropShadow", "layer.layerMask.revealAll", "layer.mergeDown", "edit.freeTransform", "layer.flattenImage"] {
            assert!(!off(&s, id), "{id}");
        }
        assert!(off(&s, "layer.rasterize.type"));
        assert!(off(&s, "layer.layerStyle.copyLayerStyle"));
    }
}
