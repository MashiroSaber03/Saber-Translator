//! JSON views of engine state for automation, tests and debugging.

use photocraft_doc::{Layer, LayerContent, SmartSource};
use serde_json::{Value, json};

use crate::{DocState, Session};

pub fn session(s: &Session) -> Value {
    json!({
        "active": s.active_index(),
        "documents": s.documents().iter().enumerate().map(|(i, d)| json!({
            "index": i,
            "name": d.doc.name,
            "path": d.path,
            "width": d.doc.size.width,
            "height": d.doc.size.height,
            "dirty": d.is_dirty(),
            "revision": d.revision,
        })).collect::<Vec<_>>(),
        "foreground": s.tools.foreground,
        "background": s.tools.background,
    })
}

pub fn document(d: &DocState) -> Value {
    let doc = &d.doc;
    let selected = d.selected_layers();
    json!({
        "name": doc.name,
        "path": &d.path,
        "width": doc.size.width,
        "height": doc.size.height,
        "mode": format!("{:?}", doc.mode),
        "depth": doc.depth.bits(),
        "resolution": doc.resolution_dpi,
        "activeLayer": d.active_layer.map(|l| l.0),
        "selectedLayers": selected.iter().map(|l| l.0).collect::<Vec<_>>(),
        "isolatedLayers": d.isolated_layers.iter().map(|l| l.0).collect::<Vec<_>>(),
        "hasSelection": doc.selection.is_some(),
        "selectionBounds": doc.selection.as_ref().map(|s| { let r = s.content_bounds(); [r.x0, r.y0, r.width() as i32, r.height() as i32] }),
        "layers": doc.layers.iter().rev().map(|l| layer_sel(l, &selected, Some(&doc.metadata))).collect::<Vec<_>>(),
        "history": d.history.entries(),
        "snapshots": d.snapshots.iter().map(|snap| json!({"id": snap.id, "name": snap.name})).collect::<Vec<_>>(),
        "canUndo": d.history.can_undo(),
        "canRedo": d.history.can_redo(),
        "revision": d.revision,
        "channels": crate::channel_cmds::channels_json(d),
        "quickMask": doc.quick_mask.is_some(),
        "symmetry": d.symmetry.as_ref().map(|s| s.inspect()),
        "layerMaskView": crate::mask_view_cmds::view_json(d),
        "layerComps": doc.layer_comps.iter().map(|c| json!({"id": c.id, "name": c.name})).collect::<Vec<_>>(),
        "lastAppliedComp": doc.last_applied_comp,
        "measurement": {"scale": doc.measurement.scale.describe(), "ruler": doc.measurement.ruler.map(|r| json!({"start": r.start, "end": r.end, "protractor": r.protractor})), "count": doc.measurement.count_total(), "countGroups": doc.measurement.count_groups.len()},
        "notes": doc.notes.iter().map(|n| json!({"author": n.author, "text": n.text, "position": n.position})).collect::<Vec<_>>(),
        "timeline": doc.timeline.as_ref().map(|t| json!({"fps": t.fps, "duration": t.duration, "current": t.current})),
        "variables": (!doc.variables.is_empty()).then(|| json!({
            "defs": doc.variables.defs.iter().map(|vd| json!({"name": vd.name, "layer": vd.layer.0})).collect::<Vec<_>>(),
            "dataSets": doc.variables.data_sets.iter().map(|s| s.name.clone()).collect::<Vec<_>>(),
            "active": doc.variables.active,
        })),
    })
}

/// Layer tree top-to-bottom (display order).
pub fn layer(l: &Layer) -> Value {
    layer_sel(l, &[], None)
}

/// [`layer`] with a `selected` flag on every node (Layers panel multi-selection); with the
/// document's `meta`, a smart object whose file is embedded in the PSD says so.
fn layer_sel(l: &Layer, selected: &[photocraft_doc::LayerId], meta: Option<&photocraft_doc::Metadata>) -> Value {
    let bounds = l.surface().map(|s| {
        let r = s.content_bounds();
        [r.x0, r.y0, r.width() as i32, r.height() as i32]
    });
    let mut v = json!({
        "id": l.id.0,
        "name": l.name,
        "kind": l.content.kind_name(),
        "visible": l.visible,
        "labelColor": l.label.id(),
        "opacity": l.opacity,
        "fill": l.fill_opacity,
        "blend": l.blend.label(),
        "clipped": l.clipped,
        "hasMask": l.mask.is_some(),
        "hasVectorMask": l.vector_mask.is_some(),
        "vectorMaskEnabled": l.vector_mask.as_ref().map(|mask| mask.enabled),
        "bounds": bounds,
        "selected": selected.contains(&l.id),
        "linkGroup": l.link_group,
    });
    match &l.content {
        LayerContent::Group(g) => {
            v["children"] = Value::Array(g.children.iter().rev().map(|c| layer_sel(c, selected, meta)).collect());
            v["expanded"] = json!(g.expanded);
            if let Some(a) = &g.artboard {
                v["artboard"] = json!({"rect": [a.rect.x0, a.rect.y0, a.rect.width(), a.rect.height()], "background": a.background.name(), "preset": a.preset});
            }
        }
        LayerContent::Adjustment(a) => {
            // The same kind and parameters the adjustment commands take (`inBlack` 0..255, …),
            // not the internal model.
            v["adjustment"] = json!({"kind": crate::commands::adjustment_kind(a), "params": crate::adjust_params::to_params(a)});
        }
        LayerContent::Smart(sm) => {
            v["smartFilters"] = Value::Array(
                sm.smart_filters
                    .iter()
                    .map(|f| json!({"command": f.command, "params": f.params, "visible": f.visible, "opacity": f.opacity, "blend": f.blend.label()}))
                    .collect(),
            );
            v["smartFiltersEnabled"] = json!(sm.filters_enabled);
            // Where the contents come from. A PSD placed layer (embedded or linked in Photoshop) is
            // `linked` to the `Idnt` uuid of its file, which its duplicates share; when the file
            // is embedded in the PSD (a `liFD` item) it is reported as embedded, with that uuid.
            // New Smart Object via Copy embeds its own copy of the file instead.
            v["smartSource"] = match &sm.source {
                SmartSource::Embedded { file_name, .. } => json!({"kind": "embedded", "fileName": file_name}),
                SmartSource::Linked { path } => match meta.and_then(|m| photocraft_io::linked::embedded_file_name(m, path)) {
                    Some(name) => json!({"kind": "embedded", "fileName": name, "uuid": path}),
                    None => json!({"kind": "linked", "path": path}),
                },
            };
        }
        LayerContent::Text(t) => {
            v["text"] = json!({"text": t.text, "font": t.font_family, "sizePt": t.size_pt});
        }
        _ => {}
    }
    // Advanced Blending › Channels (only when some channel is left out).
    if l.excluded_channels != 0 {
        v["channels"] = json!((0..4).map(|i| l.excluded_channels & (1 << i) == 0).collect::<Vec<_>>());
    }
    // Blending Options › Advanced Blending switches (only when one differs from Photoshop's default).
    if !l.advanced.is_default() {
        v["advancedBlending"] = serde_json::to_value(l.advanced).unwrap_or(Value::Null);
    }
    // Blending Options › Blend If (only when set): range index (0 = Gray, then the mode's
    // channels) with This Layer / Underlying Layer as [blackLo, blackHi, whiteLo, whiteHi].
    if !l.blend_if.is_default() {
        v["blendIf"] = l
            .blend_if
            .ranges
            .iter()
            .enumerate()
            .filter(|(_, r)| !r.iter().all(photocraft_doc::BlendRange::is_full))
            .map(|(i, [this, under])| json!({"channel": i, "thisLayer": this.to_bytes(), "underlying": under.to_bytes()}))
            .collect();
    }
    // Layer styles, so agents can verify what they applied (full settings via the style commands).
    if !l.effects.items.is_empty() {
        v["effects"] = json!({
            "enabled": l.effects.enabled,
            "items": l.effects.items.iter().map(|e| json!({"kind": e.label(), "enabled": e.enabled()})).collect::<Vec<_>>(),
        });
    }
    // Video layer (Layer › Video Layers): the frame stack behind the displayed content.
    if let Some(vid) = &l.video {
        let source = match &vid.source {
            photocraft_doc::VideoSource::Blank => Value::String("blank".into()),
            photocraft_doc::VideoSource::File { path } => json!({ "file": path }),
        };
        v["video"] = json!({"frames": vid.frames.len(), "fps": vid.fps, "showAltered": vid.show_altered, "source": source});
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_color::PixelFormat;
    use photocraft_doc::{LayerMask, Path, VectorMask};

    #[test]
    fn an_adjustment_layer_is_reported_with_the_params_its_command_takes() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 4, "height": 4})).unwrap();
        s.execute("layer.newAdjustmentLayer.levels", json!({"inBlack": 10, "inWhite": 240, "gamma": 1.2})).unwrap();
        let info = s.execute("document.inspect", json!({})).unwrap();
        let adj = &info["layers"][0]["adjustment"];
        assert_eq!(adj["kind"], "levels");
        assert_eq!((adj["params"]["inBlack"].as_f64(), adj["params"]["inWhite"].as_f64()), (Some(10.0), Some(240.0)), "{adj}");
        // Fed back to the command, the params give the same adjustment; no internal field names.
        let model = match &s.active().unwrap().doc.layers[1].content {
            LayerContent::Adjustment(a) => a.clone(),
            c => panic!("{c:?}"),
        };
        let back = crate::adjust_params::from_params("levels", &adj["params"], None, photocraft_color::ColorMode::Rgb).unwrap();
        assert_eq!(crate::adjust_params::to_params(&back), crate::adjust_params::to_params(&model));
        assert!(!adj.to_string().contains('_'), "{adj}");
    }

    #[test]
    fn document_inspect_reports_raster_and_vector_masks_when_nested() {
        let cases = [(false, None), (true, None), (false, Some(true)), (false, Some(false)), (true, Some(true))];
        let mut session = Session::new();
        session.execute("file.new", json!({"width": 1, "height": 1})).unwrap();
        let layers = cases
            .iter()
            .map(|&(raster, enabled)| {
                let mut layer = Layer::raster("Mask case", PixelFormat::RGBA8);
                layer.mask = raster.then(LayerMask::reveal_all);
                layer.vector_mask = enabled.map(|enabled| VectorMask { enabled, ..VectorMask::new(Path::default()) });
                layer
            })
            .collect::<Vec<_>>();
        let mut group = Layer::group("Inner", layers.clone());
        group.vector_mask = Some(VectorMask { enabled: false, ..VectorMask::new(Path::default()) });
        let mut roots = layers;
        roots.push(Layer::group("Outer", vec![group]));
        session
            .edit("Mask fixtures", |doc, _| {
                doc.layers = roots;
                Ok(())
            })
            .unwrap();

        let inspected = session.execute("document.inspect", json!({})).unwrap();

        let roots = inspected["layers"].as_array().unwrap();
        let outer = &roots[0];
        let inner = &outer["children"][0];
        assert_eq!(outer.get("hasMask"), Some(&json!(false)));
        assert_eq!(outer.get("hasVectorMask"), Some(&json!(false)));
        assert_eq!(outer.get("vectorMaskEnabled"), Some(&Value::Null));
        assert_eq!(inner.get("hasMask"), Some(&json!(false)));
        assert_eq!(inner.get("hasVectorMask"), Some(&json!(true)));
        assert_eq!(inner.get("vectorMaskEnabled"), Some(&json!(false)));
        for nodes in [&roots[1..], inner["children"].as_array().unwrap().as_slice()] {
            assert_eq!(nodes.len(), cases.len());
            for (node, &(raster, enabled)) in nodes.iter().zip(cases.iter().rev()) {
                assert_eq!(node.get("hasMask"), Some(&json!(raster)));
                assert_eq!(node.get("hasVectorMask"), Some(&json!(enabled.is_some())));
                assert_eq!(node.get("vectorMaskEnabled"), Some(&json!(enabled)));
            }
        }
    }
}
