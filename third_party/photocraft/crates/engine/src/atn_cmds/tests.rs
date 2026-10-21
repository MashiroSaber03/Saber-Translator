use super::*;
use photocraft_psd::atn::{AtnAction, AtnEvent, AtnFile, AtnItem};
use photocraft_psd::descriptor::{Descriptor, Id, ReferenceItem, UnicodeString, Value as D};
use serde_json::json;

fn item(event: &str, label: &str, d: Option<Descriptor>) -> AtnItem {
    let event = match event.as_bytes() {
        [a, b, c, dd] => AtnEvent::Code([*a, *b, *c, *dd]),
        _ => AtnEvent::Name(event.into()),
    };
    AtnItem { event, dictionary: label.into(), descriptor: d, ..Default::default() }
}

fn b64(bytes: &[u8]) -> String {
    photocraft_paint::tile::b64_encode(bytes)
}

/// A set with a mapped filter, an adjustment layer, selection steps, a layer by name, a Stop,
/// a Play, a tool selection (no document effect) and an event PhotoCraft has no command for.
fn sample() -> AtnFile {
    let levels = desc("Lvls").with(
        "Adjs",
        D::List(vec![D::Descriptor(
            desc("LvlA")
                .with("Chnl", channel_ref(None))
                .with("Inpt", D::List(vec![long(10.0), long(240.0)]))
                .with("Gmm ", D::Double(1.2))
                .with("Otpt", D::List(vec![long(0.0), long(250.0)])),
        )]),
    );
    let make_levels = desc("null").with("null", class_ref("AdjL")).with("Usng", D::Descriptor(desc("AdjL").with("Type", D::Descriptor(levels))));
    let select_all =
        desc("null").with("null", D::Reference(vec![ReferenceItem::Property { class: class("Chnl"), key: Id::new("fsel") }])).with("T   ", enm("Ordn", "Al  "));
    let select_layer = desc("null").with("null", name_ref("Lyr ", "Shadow")).with("MkVs", D::Boolean(false));
    let tool = desc("null").with("null", class_ref("MvTl"));
    let play = desc("null").with("null", D::Reference(vec![ReferenceItem::Name { class: class("Actn"), name: UnicodeString::new_nul("Finish") }]));
    let stop = desc("null").with("Msge", text("Check the result")).with("Cntn", D::Boolean(true));
    let fancy = desc("null").with("prompt", text("a cat")).with("DocI", long(5.0));
    let mut blur = item("GsnB", "Gaussian Blur", Some(desc("null").with("Rds ", px(2.5))));
    blur.with_dialog = true;
    let mut off = item("Fltn", "Flatten Image", None);
    off.enabled = false;
    AtnFile {
        name: UnicodeString::new_nul("Test Set"),
        expanded: true,
        actions: vec![
            AtnAction {
                name: UnicodeString::new_nul("Soften"),
                function_key: 3,
                items: vec![
                    blur,
                    item("make", "Make", Some(make_levels)),
                    item("setd", "Set", Some(select_all)),
                    item("select", "Select", Some(select_layer)),
                    item("slct", "Select", Some(tool)),
                    item("stop", "Stop", Some(stop)),
                    item("syntheticFill", "Generative Fill", Some(fancy)),
                    item("play", "Play", Some(play)),
                    off,
                ],
                ..Default::default()
            },
            AtnAction { name: UnicodeString::new_nul("Finish"), items: vec![item("Invs", "Inverse", None)], ..Default::default() },
        ],
        warnings: Vec::new(),
    }
}

#[test]
fn load_maps_events_keeps_flags_and_plays() {
    let mut s = Session::new();
    let bytes = atn::write(&sample()).unwrap();
    let r = s.execute("actions.load", json!({"data": b64(&bytes)})).unwrap();
    assert_eq!(r["set"], "Test Set");
    assert_eq!(r["actions"], json!(["Soften", "Finish"]));
    assert_eq!(r["steps"], 10);
    assert_eq!(r["functionKeys"], json!([{"action": "Soften", "key": "F3"}]));
    assert_eq!(s.prefs().shortcuts.get("actions.play:Soften").map(String::as_str), Some("F3"));
    let unsupported: Vec<(String, String)> =
        r["unsupported"].as_array().unwrap().iter().map(|u| (u["event"].as_str().unwrap().into(), u["name"].as_str().unwrap().into())).collect();
    assert_eq!(unsupported, [("slct".to_string(), "Select".to_string()), ("syntheticFill".to_string(), "Generative Fill".to_string())]);

    let a = &s.actions.list[0];
    assert_eq!(a.set.as_deref(), Some("Test Set"));
    let ids: Vec<&str> = a.steps.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(
        ids,
        [
            "filter.blur.gaussianBlur",
            "layer.newAdjustmentLayer.levels",
            "select.all",
            "layer.select",
            PHOTOSHOP_STEP,
            STOP_MESSAGE,
            PHOTOSHOP_STEP,
            "actions.play",
            "layer.flattenImage"
        ]
    );
    assert_eq!(a.steps[0].1, json!({"radius": 2.5}));
    assert_eq!(a.steps[1].1, json!({"inBlack": 10, "inWhite": 240, "gamma": 1.2, "outBlack": 0, "outWhite": 250}));
    assert_eq!(a.steps[3].1, json!({"layer": "Shadow"}));
    assert_eq!(a.steps[5].1, json!({"message": "Check the result", "allowContinue": true}));
    assert_eq!(a.steps[6].1["event"], "syntheticFill");
    assert_eq!(a.steps[7].1, json!({"action": "Finish"}));
    assert_eq!(a.dialogs.iter().copied().collect::<Vec<_>>(), [0], "the dialog toggle is kept");
    assert_eq!(a.disabled.iter().copied().collect::<Vec<_>>(), [4, 8], "tool selections are switched off, as is the step the file had off");
    assert_eq!(s.actions.list[1].steps, [("select.inverse".to_string(), json!({}))]);

    // Playback: the mapped steps run, the layer is found by name, and the kept Photoshop step
    // stops the action with its name.
    s.execute("file.new", json!({"width": 16, "height": 16})).unwrap();
    s.execute("layer.new.layer", json!({"name": "Shadow"})).unwrap();
    s.execute("layer.new.layer", json!({"name": "Top"})).unwrap();
    let played = s.execute("actions.play", json!({"action": "Soften"})).unwrap();
    assert_eq!(played["ran"], 5, "{played}");
    assert_eq!(played["failed"]["step"], 6);
    assert!(played["failed"]["error"].as_str().unwrap().contains("Generative Fill"), "{played}");
    let d = s.active().unwrap();
    assert_eq!(d.doc.layer(d.active_layer.unwrap()).unwrap().name, "Shadow");
    assert!(d.doc.walk().iter().any(|(_, _, l)| l.name.starts_with("Levels")));

    // Loading the same set again replaces it; a different set with the same action names
    // gets unique names and its Play steps follow.
    s.execute("actions.load", json!({"data": b64(&bytes)})).unwrap();
    assert_eq!(s.actions.list.len(), 2);
    let r = s.execute("actions.load", json!({"data": b64(&bytes), "set": "Other"})).unwrap();
    assert_eq!(r["actions"], json!(["Soften 2", "Finish 2"]));
    assert_eq!(s.actions.list[2].steps[7].1, json!({"action": "Finish 2"}));
    assert_eq!(s.actions.list[2].set.as_deref(), Some("Other"));
}

#[test]
fn save_round_trips_through_photoshop_events() {
    let mut s = Session::new();
    let bytes = atn::write(&sample()).unwrap();
    s.execute("actions.load", json!({"data": b64(&bytes)})).unwrap();
    let first = s.actions.list.clone();
    let r = s.execute("actions.save", json!({"set": "Test Set"})).unwrap();
    assert_eq!(r["actions"], json!(["Soften", "Finish"]));
    assert_eq!(r["skipped"], json!([]));
    let saved = photocraft_paint::tile::b64_decode(r["data"].as_str().unwrap()).unwrap();
    let file = atn::parse(&saved).unwrap();
    assert_eq!(file.name.to_string_lossy(), "Test Set");
    assert_eq!(file.actions[0].function_key, 3);
    let events: Vec<String> = file.actions[0].items.iter().map(|i| i.event.id()).collect();
    assert_eq!(events, ["GsnB", "Mk  ", "setd", "slct", "slct", "Stop", "syntheticFill", "Ply ", "Fltn"]);
    assert!(file.actions[0].items[0].with_dialog);
    assert!(!file.actions[0].items[8].enabled);
    // The kept Photoshop step is written back byte for byte.
    assert_eq!(file.actions[0].items[6].descriptor, sample().actions[0].items[6].descriptor);
    // Play names the set too, as Photoshop does.
    assert_eq!(get_ref(file.actions[0].items[7].descriptor.as_ref().unwrap(), "null").unwrap().len(), 2);

    // Loading what was saved gives the same actions.
    let mut t = Session::new();
    t.execute("actions.load", json!({"data": b64(&saved)})).unwrap();
    assert_eq!(t.actions.list, first);

    // A step with no Photoshop equivalent fails unless skipped.
    s.actions.list.push(Action::new("Ours", vec![("filter.render.clouds".into(), json!({}))]));
    let e = s.execute("actions.save", json!({"action": "Ours"})).unwrap_err().to_string();
    assert!(e.contains("filter.render.clouds"), "{e}");
    let r = s.execute("actions.save", json!({"action": "Ours", "skipUnsupported": true})).unwrap();
    assert_eq!(r["skipped"], json!([{"action": "Ours", "step": 0, "id": "filter.render.clouds"}]));
    assert_eq!(r["set"], "Ours");
}

#[test]
fn every_mapped_command_exports_and_imports_back() {
    let cases: Vec<(&str, Value)> = vec![
        ("filter.blur.gaussianBlur", json!({"radius": 3.5})),
        ("filter.sharpen.unsharpMask", json!({"amount": 120, "radius": 1.5, "threshold": 3})),
        ("filter.noise.addNoise", json!({"amount": 12.5, "distribution": "gaussian", "monochromatic": true, "seed": 7})),
        ("filter.noise.median", json!({"radius": 2})),
        ("filter.blur.motionBlur", json!({"angle": 45, "distance": 20})),
        ("filter.other.offset", json!({"horizontal": 10, "vertical": -5, "undefinedAreas": "wrap"})),
        ("filter.blur.blur", json!({})),
        (
            "image.adjustments.levels",
            json!({"inBlack": 5, "inWhite": 250, "gamma": 0.9, "outBlack": 0, "outWhite": 255, "red": {"inBlack": 2, "inWhite": 255, "gamma": 1, "outBlack": 0, "outWhite": 255}}),
        ),
        ("image.adjustments.curves", json!({"points": [[0, 0], [128, 140], [255, 255]], "blue": [[0, 10], [255, 245]]})),
        (
            "image.adjustments.hueSaturation",
            json!({"colorize": false, "hue": 10, "saturation": -20, "lightness": 0, "reds": {"hue": 5, "saturation": 0, "lightness": 0, "range": [315, 345, 15, 45]}}),
        ),
        ("image.adjustments.brightnessContrast", json!({"brightness": 10, "contrast": -5, "legacy": false})),
        ("image.adjustments.threshold", json!({"level": 100})),
        ("image.adjustments.posterize", json!({"levels": 4})),
        ("image.adjustments.invert", json!({})),
        ("image.adjustments.colorBalance", json!({"shadows": [1, 2, 3], "midtones": [0, 0, 0], "highlights": [-5, 0, 5], "preserveLuminosity": true})),
        ("image.adjustments.vibrance", json!({"vibrance": 30, "saturation": 5})),
        ("image.adjustments.exposure", json!({"exposure": 0.5, "offset": 0.01, "gamma": 1.1})),
        ("image.adjustments.photoFilter", json!({"color": "#ec8a00", "density": 25, "preserveLuminosity": true})),
        (
            "image.adjustments.blackWhite",
            json!({"reds": 40, "yellows": 60, "greens": 40, "cyans": 60, "blues": 20, "magentas": 80, "tint": true, "tintColor": "#e4d5b0"}),
        ),
        ("image.adjustments.channelMixer", json!({"red": [100, 0, 0, 0], "green": [0, 90, 10, 0], "blue": [0, 0, 100, 5], "monochrome": false})),
        ("image.adjustments.selectiveColor", json!({"method": "absolute", "reds": [10, 0, -5, 0], "neutrals": [0, 0, 0, 3]})),
        ("image.adjustments.desaturate", json!({})),
        ("image.autoTone", json!({})),
        ("image.imageSize", json!({"width": 800, "height": 600, "resolution": 72, "resample": "bicubicSharper"})),
        ("image.canvasSize", json!({"width": 100, "height": 50, "relative": true, "anchor": "topLeft", "extensionColor": "white"})),
        ("image.imageRotation.90cw", json!({})),
        ("image.rotation.arbitrary", json!({"angle": 12.5, "direction": "ccw"})),
        ("edit.transform.flipHorizontal", json!({})),
        ("image.imageRotation.flipCanvasVertical", json!({})),
        ("image.crop", json!({"x": 10, "y": 20, "width": 100, "height": 50, "deleteCroppedPixels": true})),
        ("image.trim", json!({"basedOn": "transparent", "top": true, "bottom": true, "left": false, "right": true})),
        ("image.mode.grayscale", json!({})),
        ("image.mode.bits16", json!({})),
        ("layer.duplicate", json!({"name": "copy"})),
        ("image.duplicate", json!({"mergedOnly": true})),
        ("layer.new.layer", json!({"name": "Ink", "opacity": 0.5, "mode": "Multiply"})),
        ("layer.new.group", json!({"name": "G"})),
        ("layer.groupLayers", json!({})),
        ("layer.new.layerViaCopy", json!({})),
        ("layer.newFillLayer.solidColor", json!({"color": "#336699"})),
        ("layer.newAdjustmentLayer.hueSaturation", json!({})),
        ("layer.newAdjustmentLayer.threshold", json!({"level": 77})),
        ("file.new", json!({"width": 640, "height": 480, "resolution": 72, "mode": "rgb", "depth": 8, "background": "transparent", "name": "Doc"})),
        ("select.all", json!({})),
        ("select.deselect", json!({})),
        ("select.loadSelection", json!({"channel": "transparency"})),
        ("select.loadSelection", json!({"channel": "Alpha 1"})),
        ("select.rect", json!({"x": 1, "y": 2, "width": 30, "height": 40})),
        ("select.rect", json!({"x": 1, "y": 2, "width": 30, "height": 40, "ellipse": true})),
        ("layer.setProps", json!({"name": "Renamed", "opacity": 0.4, "blend": "Linear Light"})),
        ("layer.setProps", json!({"layer": "Shadow", "visible": false})),
        ("layer.setLabelColor", json!({"color": "red"})),
        ("layer.select", json!({"layer": "Shadow", "mode": "add"})),
        ("layer.selectTop", json!({})),
        ("layer.selectBelow", json!({})),
        ("layer.addBelowToSelection", json!({})),
        ("layer.addAboveToSelection", json!({})),
        ("layer.delete", json!({})),
        ("edit.clear", json!({})),
        ("layer.hideLayers", json!({"layer": "Background"})),
        ("layer.showLayers", json!({})),
        ("layer.translate", json!({"dx": 3, "dy": -4})),
        ("layer.arrange.sendToBack", json!({})),
        ("layer.flattenImage", json!({})),
        ("layer.stampVisible", json!({})),
        ("layer.mergeVisible", json!({})),
        ("layer.matting.defringe", json!({"width": 2})),
        ("layer.align.leftEdges", json!({"to": "canvas"})),
        ("select.inverse", json!({})),
        ("select.modify.feather", json!({"radius": 4})),
        ("select.modify.contract", json!({"radius": 1})),
        ("edit.fill", json!({"contents": "color", "color": "#ff0080", "opacity": 50, "mode": "Multiply", "preserveTransparency": true})),
        ("edit.fill", json!({"contents": "foreground", "opacity": 100, "mode": "Normal"})),
        ("edit.stroke", json!({"width": 3, "location": "outside", "opacity": 100, "mode": "Normal", "color": "#000000", "preserveTransparency": false})),
        ("edit.copyMerged", json!({})),
        ("edit.pasteSpecial.pasteInPlace", json!({})),
        ("edit.paste", json!({})),
        ("edit.purge.histories", json!({})),
        ("tools.defaultColors", json!({})),
        ("edit.convertToProfile", json!({"profile": "display-p3", "intent": "perceptual", "bpc": false})),
        ("edit.assignProfile", json!({"profile": "srgb"})),
        ("file.close", json!({})),
        ("file.save", json!({})),
        ("layer.layerMask.delete", json!({})),
        ("select.lasso", json!({"points": [[1, 2], [30, 2], [15, 40]]})),
        ("select.editInQuickMaskMode", json!({"on": true})),
        ("select.editInQuickMaskMode", json!({"on": false})),
        ("edit.transform", json!({"dx": 10, "dy": -5, "scaleX": 50, "scaleY": 75, "rotate": 12.5, "interpolation": "bicubic"})),
        ("edit.transform", json!({"layer": "Shadow", "dx": 0, "dy": 0, "scaleX": 200})),
        ("select.transformSelection", json!({"dx": 3, "dy": 4, "scaleX": 110, "scaleY": 110, "rotate": -30})),
        (
            "image.applyImage",
            json!({"source": {"channel": "composite", "layer": "High-Frequency"}, "blending": "subtract", "scale": 2, "offset": 128, "preserveTransparency": true}),
        ),
        ("image.applyImage", json!({"source": {"channel": "red", "layer": "merged", "invert": true}, "blending": "multiply", "opacity": 50})),
        (
            "file.export.saveForWebLegacy",
            json!({"path": "/tmp/out/banner.png", "format": "png24", "interlaced": false, "transparency": true, "embedIcc": false, "progressive": false, "optimized": true, "matte": "#ffffff"}),
        ),
        (
            "file.export.saveForWebLegacy",
            json!({"path": "C:\\shots\\a.jpg", "format": "jpeg", "interlaced": false, "transparency": false, "embedIcc": true, "progressive": true, "optimized": true, "quality": 80, "matte": "none"}),
        ),
        ("image.adjustments.gradientMap", json!({"stops": [[0, "#000000"], [0.5, "#ff8000"], [1, "#ffffff"]], "reverse": true, "dither": false})),
        ("layer.newAdjustmentLayer.gradientMap", json!({"stops": [[0, "#102030"], [1, "#f0e0d0"]], "reverse": false, "dither": true})),
        ("layer.smartObjects.convertToSmartObject", json!({})),
        ("layer.smartObjects.editContents", json!({})),
        ("layer.smartObjects.updateModifiedContent", json!({})),
        ("paint.bucket", json!({"x": 10, "y": 20, "tolerance": 32, "antiAlias": true, "contiguous": false, "opacity": 100})),
        ("image.imageSize", json!({"percent": 200, "resample": "bicubicAutomatic"})),
        ("file.placeEmbedded", json!({"path": "/tmp/logo.png", "scale": 50})),
        ("file.placeLinked", json!({"path": "/tmp/logo.png"})),
        ("file.openAs", json!({"path": "/tmp/a.psd"})),
        ("actions.conditional", json!({"if": "hasLayers", "then": "FS8-Layers", "else": "FS8-No-Layers", "set": "S"})),
        ("actions.conditional", json!({"if": "landscape", "then": "Wide", "else": null, "set": "S"})),
        ("actions.play", json!({"action": "Other", "set": "S"})),
        (STOP_MESSAGE, json!({"message": "Hi", "allowContinue": false})),
        // The TrueGrit Texture Supply sets (Debaser, Distress Press, Halftone Zine Machine).
        ("filter.distort.ripple", json!({"amount": 50, "size": "medium"})),
        ("filter.pixelate.crystallize", json!({"cellSize": 5})),
        ("filter.gallery.spatter", json!({"sprayRadius": 7, "smoothness": 15})),
        (
            "select.refineEdge",
            json!({"radius": 0, "smartRadius": false, "smooth": 2, "feather": 0.5, "contrast": 15, "shiftEdge": -100, "decontaminate": false, "output": "layerMask"}),
        ),
        ("channel.target", json!({"channel": "mask"})),
        ("channel.target", json!({"channel": "composite"})),
        ("edit.undo", json!({})),
        ("edit.redo", json!({})),
        ("layer.select", json!({"layer": "YELLOW", "mode": "remove"})),
        ("layer.translate", json!({"dx": 0.36, "dy": -1.08, "units": "pt"})),
        ("select.loadSelection", json!({"channel": "targeted"})),
        ("layer.newFillLayer.pattern", json!({"pattern": "Debaser-Distressed-Dark", "scale": 50, "angle": 0, "link": true})),
        ("layer.fill.pattern.set", json!({"pattern": "1 -R2.jpg"})),
        ("image.mode.bitmap", json!({"method": "threshold"})),
        ("image.imageSize", json!({"resolution": 1200, "resample": "preserveDetails", "keepPrintSize": true})),
        ("layer.new.layerFromBackground", json!({"ifPresent": true, "select": false})),
    ];
    for (id, params) in &cases {
        let item = export_item(id, params, "S").unwrap_or_else(|| panic!("{id} does not export"));
        if *id == "file.close" {
            // Close exports "don't save" and imports as a plain close; the Save answer is tested below.
            assert_eq!(item.event.id(), "Cls ");
            continue;
        }
        let e = event_for(&item.event).unwrap_or_else(|| panic!("{id} exported as an unknown event {}", item.event.id()));
        match (e.from)(item.descriptor.as_ref().unwrap()) {
            Mapped::Steps(steps) => {
                assert_eq!(steps.len(), 1, "{id}: {steps:?}");
                assert_eq!(steps[0].0, *id, "{id} came back as another command");
                let mut back = steps[0].1.clone();
                normalise(&mut back);
                if *id == "actions.play" || *id == "actions.conditional" {
                    back["set"] = json!("S");
                }
                assert_eq!(back, *params, "{id}");
            }
            Mapped::Unsupported { .. } => panic!("{id} does not import back"),
        }
    }
    // Every command the table claims to export has a case above.
    let covered: std::collections::BTreeSet<&str> = cases.iter().map(|(id, _)| *id).collect();
    let mut missing: Vec<&str> = EVENTS.iter().flat_map(|e| e.ids.iter().copied()).filter(|id| !covered.contains(id)).collect();
    missing.sort_unstable();
    let allowed = [
        "edit.transform.flipVertical",
        "edit.transform.rotate180",
        "edit.transform.rotate90Ccw",
        "edit.transform.rotate90Cw",
        "image.autoColor",
        "image.autoContrast",
        "image.imageRotation.180",
        "image.imageRotation.90ccw",
        "image.imageRotation.flipCanvasHorizontal",
        "image.mode.bits32",
        "image.mode.bits8",
        "image.mode.cmyk",
        "image.mode.lab",
        "image.mode.multichannel",
        "image.mode.rgb",
        "layer.align.bottomEdges",
        "layer.align.horizontalCenters",
        "layer.align.rightEdges",
        "layer.align.topEdges",
        "layer.align.verticalCenters",
        "layer.arrange.bringForward",
        "layer.arrange.bringToFront",
        "layer.arrange.sendBackward",
        "layer.mergeLayers",
        "layer.new.layerViaCut",
        "layer.newAdjustmentLayer.blackWhite",
        "layer.newAdjustmentLayer.brightnessContrast",
        "layer.newAdjustmentLayer.channelMixer",
        "layer.newAdjustmentLayer.colorBalance",
        "layer.newAdjustmentLayer.curves",
        "layer.newAdjustmentLayer.exposure",
        "layer.newAdjustmentLayer.invert",
        "layer.newAdjustmentLayer.levels",
        "layer.newAdjustmentLayer.photoFilter",
        "layer.newAdjustmentLayer.posterize",
        "layer.newAdjustmentLayer.selectiveColor",
        "layer.newAdjustmentLayer.vibrance",
        "layer.rasterize.layer",
        "layer.renameLayer",
        "layer.selectAbove",
        "layer.selectBottom",
        "layer.setAdjustment",
        "select.allLayers",
        "select.deselectLayers",
        "select.grow",
        "select.modify.border",
        "select.modify.expand",
        "select.modify.smooth",
        "select.reselect",
        "select.similar",
        "tools.swapColors",
        "edit.cut",
        "edit.copy",
        "edit.purge.all",
        "edit.purge.clipboard",
        "edit.purge.undo",
        "file.newFromClipboard",
        "file.revert",
        "image.adjustments.equalize",
        "image.revealAll",
        "filter.blur.blurMore",
        "filter.blur.boxBlur",
        "filter.blur.radialBlur",
        "filter.blur.smartBlur",
        "filter.blur.surfaceBlur",
        "filter.noise.despeckle",
        "filter.noise.dustAndScratches",
        "filter.other.highPass",
        "filter.other.maximum",
        "filter.other.minimum",
        "filter.sharpen.sharpen",
        "filter.sharpen.sharpenEdges",
        "filter.sharpen.sharpenMore",
    ];
    let unexpected: Vec<&str> = missing.iter().copied().filter(|id| !allowed.contains(id)).collect();
    assert!(unexpected.is_empty(), "add round-trip cases for {unexpected:?}");
    // Every command the table names exists.
    // (Save is the desktop shell's: recorded by it, replayed by it, written by Batch.)
    for id in EVENTS.iter().flat_map(|e| e.ids.iter()) {
        assert!(crate::commands::find(id).is_some() || crate::actions_cmds::shell_save_command(id), "the .atn table names an unknown command `{id}`");
    }
}

#[test]
fn unsupported_shapes_become_photoshop_steps_not_errors() {
    // Image size in two different percentages, a Paint Bucket click in percent of the canvas, a
    // move in percent, an unknown adjustment class.
    let items = vec![
        item("ImgS", "Image Size", Some(desc("null").with("Wdth", pct(200.0)).with("Hght", pct(150.0)))),
        item(
            "Fl  ",
            "Fill",
            Some(desc("null").with("From", D::Descriptor(desc("Pnt ").with("Hrzn", pct(0.0)).with("Vrtc", pct(0.0)))).with("Usng", enm("FlCn", "FrgC"))),
        ),
        item(
            "move",
            "Move",
            Some(desc("null").with("null", target_ref("Lyr ")).with("T   ", D::Descriptor(desc("Ofst").with("Hrzn", pct(3.0)).with("Vrtc", pct(1.0))))),
        ),
        item(
            "make",
            "Make",
            Some(desc("null").with("null", class_ref("AdjL")).with("Usng", D::Descriptor(desc("AdjL").with("Type", D::Descriptor(desc("GdMp")))))),
        ),
        item("GsnB", "Gaussian Blur", None),
    ];
    let file = AtnFile {
        name: UnicodeString::new_nul("Odd"),
        expanded: false,
        actions: vec![AtnAction { name: UnicodeString::new_nul("A"), items, ..Default::default() }],
        warnings: vec![],
    };
    let imp = import(&atn::write(&file).unwrap()).unwrap();
    let ids: Vec<&str> = imp.actions[0].steps.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(ids, [PHOTOSHOP_STEP, PHOTOSHOP_STEP, PHOTOSHOP_STEP, PHOTOSHOP_STEP, "filter.blur.gaussianBlur"]);
    assert_eq!(imp.unsupported.len(), 4);
    assert!(imp.actions[0].disabled.is_empty(), "steps with a document effect stay on so playback reports them");
}

#[test]
fn step_commands_and_bad_params_fail_gracefully() {
    let mut s = Session::new();
    for p in [
        json!({}),
        json!({"path": ""}),
        json!({"path": "/nonexistent/set.atn"}),
        json!({"data": 42}),
        json!({"data": "%%%"}),
        json!({"data": b64(b"")}),
        json!({"data": b64(&[0, 0, 0, 12, 0, 0, 0, 0])}),
        json!({"data": b64(&atn::write(&sample()).unwrap()[..40])}),
        json!({"data": b64(&atn::write(&sample()).unwrap()), "set": ""}),
        json!({"data": b64(&atn::write(&sample()).unwrap()), "set": 3}),
        json!({"data": b64(&atn::write(&sample()).unwrap()), "replace": "yes"}),
    ] {
        assert!(s.execute("actions.load", p.clone()).is_err(), "{p}");
    }
    assert!(s.actions.list.is_empty());
    for p in [json!({}), json!({"set": "Nope"}), json!({"action": 3}), json!({"path": 7}), json!({"skipUnsupported": "no"}), json!({"set": ""})] {
        assert!(s.execute("actions.save", p.clone()).is_err(), "{p}");
    }
    s.actions.list.push(Action::new("One", vec![("select.all".into(), json!({}))]));
    assert!(s.execute("actions.save", json!({"path": ""})).is_err());
    assert!(s.execute("actions.save", json!({})).unwrap()["data"].is_string());

    assert!(s.execute(STOP_MESSAGE, json!({"message": "halt"})).unwrap_err().to_string().contains("halt"));
    assert_eq!(s.execute(STOP_MESSAGE, json!({"message": "go on", "allowContinue": true})).unwrap()["continued"], true);
    assert!(s.execute(STOP_MESSAGE, json!({"message": 5, "allowContinue": "x"})).is_err());
    for p in [json!({}), json!({"event": 1}), json!({"event": "GsnB", "name": "Gaussian Blur", "descriptor": "not base64"})] {
        assert!(s.execute(PHOTOSHOP_STEP, p.clone()).is_err(), "{p}");
    }
    // A Photoshop step with a damaged descriptor is left out of a save rather than crashing it.
    s.actions.list.push(Action::new("Broken", vec![(PHOTOSHOP_STEP.into(), json!({"event": "GsnB", "name": "Blur", "descriptor": b64(b"\x00\x00\x00")}))]));
    assert!(s.execute("actions.save", json!({"action": "Broken"})).is_err());
    assert_eq!(s.execute("actions.save", json!({"action": "Broken", "skipUnsupported": true})).unwrap()["skipped"].as_array().unwrap().len(), 1);
}

#[test]
fn save_writes_the_file_and_the_set_of_the_selected_action() {
    let mut s = Session::new();
    s.execute("actions.load", json!({"data": b64(&atn::write(&sample()).unwrap())})).unwrap();
    s.actions.list.push(Action::new("Loose", vec![("select.all".into(), json!({}))]));
    let dir = std::env::temp_dir().join(format!("pc-atn-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("Saved.atn");
    let r = s.execute("actions.save", json!({"action": "Finish", "path": path.to_string_lossy()})).unwrap();
    assert_eq!((r["set"].as_str(), r["actions"].as_array().map(Vec::len)), (Some("Test Set"), Some(1)));
    let file = atn::parse(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(file.actions.len(), 1);
    assert_eq!(file.actions[0].name.to_string_lossy(), "Finish");
    // Without a set, everything goes out under the default name.
    let r = s.execute("actions.save", json!({})).unwrap();
    assert_eq!((r["set"].as_str(), r["actions"].as_array().map(Vec::len)), (Some(DEFAULT_SET), Some(3)));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn layer_param_accepts_a_name() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
    s.execute("layer.new.layer", json!({"name": "A"})).unwrap();
    s.execute("layer.new.layer", json!({"name": "B"})).unwrap();
    s.execute("layer.select", json!({"layer": "A"})).unwrap();
    let d = s.active().unwrap();
    assert_eq!(d.doc.layer(d.active_layer.unwrap()).unwrap().name, "A");
    assert!(s.execute("layer.select", json!({"layer": "Nope"})).is_err());
    assert!(s.execute("layer.select", json!({"layer": true})).is_err());
    assert!(s.execute("layer.select", json!({"layer": -1})).is_err());
}

#[test]
fn colours_convert_both_ways() {
    assert_eq!(color_hex(&desc("RGBC").with("Rd  ", D::Double(255.0)).with("Grn ", D::Double(127.6)).with("Bl  ", D::Double(0.0))).as_deref(), Some("#ff8000"));
    assert_eq!(color_hex(&desc("HSBC").with("H   ", ang(0.0)).with("Strt", D::Double(0.0)).with("Brgh", D::Double(100.0))).as_deref(), Some("#ffffff"));
    assert_eq!(color_hex(&desc("HSBC").with("H   ", ang(120.0)).with("Strt", D::Double(100.0)).with("Brgh", D::Double(100.0))).as_deref(), Some("#00ff00"));
    assert_eq!(color_hex(&desc("Grsc").with("Gry ", D::Double(100.0))).as_deref(), Some("#000000"));
    assert_eq!(color_hex(&desc("CMYC")), None);
    assert_eq!(color_desc("#12ab9F").map(|d| color_hex(&d)), Some(Some("#12ab9f".into())));
    assert_eq!(color_desc("12ab9f"), None);
    assert_eq!(color_desc("#12ab9"), None);
    assert_eq!(color_desc("#ééééé"), None);
}

#[test]
fn remaining_events_import_in_their_photoshop_shapes() {
    // Close with the Save answer becomes Save then Close.
    let close_yes = item("Cls ", "Close", Some(desc("null").with("Svng", enm("YsN ", "Ys  "))));
    let close_no = item("close", "Close", Some(desc("null").with("Svng", enm("YsN ", "N   "))));
    // A layer only moved is a translation; scaled, it is the relative Free Transform; the
    // selection's frame goes to Transform Selection; points (not pixels) stay unsupported.
    let moved = desc("null")
        .with("null", target_ref("Lyr "))
        .with("FTcs", enm("QCSt", "Qcsa"))
        .with("Ofst", D::Descriptor(desc("Ofst").with("Hrzn", px(-57.4)).with("Vrtc", px(12.0))))
        .with("Intr", enm("Intp", "Bcbc"));
    let scaled = desc("null")
        .with("null", target_ref("Lyr "))
        .with("FTcs", enm("QCSt", "Qcsa"))
        .with("Ofst", D::Descriptor(desc("Ofst").with("Hrzn", px(10.0)).with("Vrtc", px(0.0))))
        .with("Wdth", pct(50.0))
        .with("Hght", pct(50.0))
        .with("Lnkd", D::Boolean(true))
        .with("Intr", enm("Intp", "Bcbc"));
    let sel = desc("null")
        .with("null", D::Reference(vec![ReferenceItem::Property { class: class("Chnl"), key: Id::new("fsel") }]))
        .with("FTcs", enm("QCSt", "Qcs0"))
        .with("Ofst", D::Descriptor(desc("Ofst").with("Hrzn", px(1014.0)).with("Vrtc", px(0.0))));
    let points = desc("null").with("null", target_ref("Lyr ")).with(
        "Ofst",
        D::Descriptor(desc("Ofst").with("Hrzn", D::UnitFloat { unit: *b"#Rlt", value: 3.0 }).with("Vrtc", D::UnitFloat { unit: *b"#Rlt", value: 1.0 })),
    );
    // Apply Image as recorded by the frequency-separation action.
    let apply = desc("null").with(
        "With",
        D::Descriptor(
            desc("null")
                .with(
                    "T   ",
                    D::Reference(vec![
                        ReferenceItem::Enumerated { class: class("Chnl"), type_id: Id::new("Chnl"), value: Id::new("RGB ") },
                        ReferenceItem::Name { class: class("Lyr "), name: UnicodeString::new_nul("High-Frequency") },
                    ]),
                )
                .with("Clcl", enm("Clcn", "Sbtr"))
                .with("Scl ", D::Double(2.0))
                .with("Ofst", D::Double(128.0))
                .with("PrsT", D::Boolean(true)),
        ),
    );
    // Save for Web with and without a destination.
    let web = desc("null")
        .with(
            "Usng",
            D::Descriptor(
                desc("SaveForWeb")
                    .with("Fmt ", enm("IRFm", "PN24"))
                    .with("Intr", D::Boolean(false))
                    .with("Trns", D::Boolean(true))
                    .with("Mtt ", D::Boolean(true))
                    .with("MttR", long(255.0))
                    .with("MttG", long(255.0))
                    .with("MttB", long(255.0))
                    .with("EICC", D::Boolean(false)),
            ),
        )
        .with("In  ", path_value("/Users/me/Desktop/out.png"));
    let web_nowhere = desc("null").with("Usng", D::Descriptor(desc("SaveForWeb").with("Fmt ", enm("IRFm", "PN24"))));
    // Quick Mask on/off, a polygon selection, the mask of the target layer deleted.
    let qm_on = desc("null").with("null", quick_mask_ref());
    let polygon = desc("null").with("null", D::Reference(vec![ReferenceItem::Property { class: class("Chnl"), key: Id::new("fsel") }])).with(
        "T   ",
        D::Descriptor(desc("Plgn").with(
            "Pts ",
            D::List(vec![
                D::Descriptor(desc("Pnt ").with("Hrzn", px(1.0)).with("Vrtc", px(2.0))),
                D::Descriptor(desc("Pnt ").with("Hrzn", px(30.0)).with("Vrtc", px(2.0))),
                D::Descriptor(desc("Pnt ").with("Hrzn", px(15.0)).with("Vrtc", px(40.0))),
            ]),
        )),
    );
    let mask_delete = desc("null").with("null", target_ref("Chnl"));
    let items = vec![
        item("GsnB", "Gaussian Blur", Some(desc("null").with("Rds ", px(1.0)))),
        close_yes,
        close_no,
        item("transform", "Transform", Some(moved)),
        item("Trnf", "Transform", Some(scaled)),
        item("transform", "Transform", Some(sel)),
        item("transform", "Transform", Some(points)),
        item("applyImageEvent", "Apply Image", Some(apply)),
        item("export", "Export", Some(web)),
        item("export", "Export", Some(web_nowhere)),
        item("setd", "Set", Some(qm_on.clone())),
        item("clearEvent", "Clear", Some(qm_on.clone())),
        item("set", "Set", Some(polygon)),
        item("delete", "Delete", Some(mask_delete)),
        item("newPlacedLayer", "Convert to Smart Object", None),
        item("placedLayerEditContents", "Edit Contents", None),
        item("updatePlacedLayer", "Update Smart Objects", None),
    ];
    let file = AtnFile {
        name: UnicodeString::new_nul("More"),
        expanded: false,
        actions: vec![AtnAction { name: UnicodeString::new_nul("A"), items, ..Default::default() }],
        warnings: vec![],
    };
    let imp = import(&atn::write(&file).unwrap()).unwrap();
    let steps = &imp.actions[0].steps;
    let ids: Vec<&str> = steps.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(
        ids,
        [
            "filter.blur.gaussianBlur",
            "file.save",
            "file.close",
            "file.close",
            "layer.translate",
            "edit.transform",
            "select.transformSelection",
            PHOTOSHOP_STEP,
            "image.applyImage",
            "file.export.saveForWebLegacy",
            PHOTOSHOP_STEP,
            "select.editInQuickMaskMode",
            "select.editInQuickMaskMode",
            "select.lasso",
            "layer.layerMask.delete",
            "layer.smartObjects.convertToSmartObject",
            "layer.smartObjects.editContents",
            "layer.smartObjects.updateModifiedContent",
        ]
    );
    assert_eq!(steps[4].1, json!({"dx": -57, "dy": 12}));
    assert_eq!(steps[5].1, json!({"dx": 10, "dy": 0, "scaleX": 50, "scaleY": 50, "interpolation": "bicubic"}));
    assert_eq!(steps[6].1, json!({"dx": 1014, "dy": 0}));
    assert_eq!(
        steps[8].1,
        json!({"source": {"channel": "composite", "layer": "High-Frequency"}, "blending": "subtract", "scale": 2, "offset": 128, "preserveTransparency": true})
    );
    assert_eq!(
        steps[9].1,
        json!({"path": "/Users/me/Desktop/out.png", "format": "png24", "interlaced": false, "transparency": true, "embedIcc": false, "matte": "#ffffff"})
    );
    assert_eq!(steps[11].1, json!({"on": true}));
    assert_eq!(steps[12].1, json!({"on": false}));
    assert_eq!(steps[13].1, json!({"points": [[1, 2], [30, 2], [15, 40]]}));
    assert_eq!(imp.unsupported.len(), 2);

    // The frequency-separation set's own conditional, a percent Image Size, a Paint Bucket
    // click in pixels (percent stays unsupported) and a Place.
    let cond = desc("null")
        .with("null", enm("Cndt", "Lyrs"))
        .with(
            "then",
            D::Reference(vec![
                ReferenceItem::Name { class: class("Actn"), name: UnicodeString::new_nul("FS8-Layers") },
                ReferenceItem::Name { class: class("ASet"), name: UnicodeString::new_nul("FS") },
            ]),
        )
        .with("else", D::Reference(vec![ReferenceItem::Name { class: class("Actn"), name: UnicodeString::new_nul("FS8-No-Layers") }]));
    let size = desc("null")
        .with("Wdth", pct(200.0))
        .with("scaleStyles", D::Boolean(true))
        .with("CnsP", D::Boolean(true))
        .with("Intr", enm("Intp", "bicubicAutomatic"));
    let bucket_px = desc("null")
        .with("From", D::Descriptor(desc("Pnt ").with("Hrzn", px(0.0)).with("Vrtc", px(0.0))))
        .with("Tlrn", long(32.0))
        .with("Usng", enm("FlCn", "FrgC"))
        .with("Cntg", D::Boolean(false));
    let bucket_pct = desc("null")
        .with("From", D::Descriptor(desc("Pnt ").with("Hrzn", pct(0.0)).with("Vrtc", pct(0.0))))
        .with("Tlrn", long(32.0))
        .with("Usng", enm("FlCn", "FrgC"));
    let place = desc("null")
        .with("null", path_value("/Users/me/logo.png"))
        .with("FTcs", enm("QCSt", "Qcsa"))
        .with("Wdth", pct(50.0))
        .with("Hght", pct(50.0))
        .with("Lnkd", D::Boolean(false));
    let items = vec![
        item("conditional", "If", Some(cond)),
        item("imageSize", "Image Size", Some(size)),
        item("fill", "Fill", Some(bucket_px)),
        item("fill", "Fill", Some(bucket_pct)),
        item("placeEvent", "Place", Some(place)),
        item("Opn ", "Open", Some(desc("null").with("null", path_value("/Users/me/in.psd")))),
    ];
    let file = AtnFile {
        name: UnicodeString::new_nul("Rest"),
        expanded: false,
        actions: vec![AtnAction { name: UnicodeString::new_nul("A"), items, ..Default::default() }],
        warnings: vec![],
    };
    let rest = import(&atn::write(&file).unwrap()).unwrap();
    let rest_steps = &rest.actions[0].steps;
    let rest_ids: Vec<&str> = rest_steps.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(rest_ids, ["actions.conditional", "image.imageSize", "paint.bucket", PHOTOSHOP_STEP, "file.placeEmbedded", "file.openAs"]);
    assert_eq!(rest_steps[0].1, json!({"if": "hasLayers", "then": "FS8-Layers", "else": "FS8-No-Layers"}));
    assert_eq!(rest_steps[1].1, json!({"percent": 200, "resample": "bicubicAutomatic"}));
    assert_eq!(rest_steps[2].1, json!({"x": 0, "y": 0, "tolerance": 32, "contiguous": false}));
    assert_eq!(rest_steps[4].1, json!({"path": "/Users/me/logo.png", "scale": 50}));
    assert_eq!(rest_steps[5].1, json!({"path": "/Users/me/in.psd"}));

    // The Photoshop path record round-trips and rejects other shapes.
    assert_eq!(path_text(&path_value("/a/b c.png")).as_deref(), Some("/a/b c.png"));
    assert_eq!(path_text(&D::Path(b"txtu".to_vec())), None);
    assert_eq!(path_text(&D::Path(vec![])), None);
    assert_eq!(path_text(&D::Text(UnicodeString::new("x"))), None);
    assert_eq!(path_text(&path_value("")), None);

    // Apply Image by layer name runs on a document; a missing name is a clear error.
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
    s.execute("layer.new.layer", json!({"name": "High-Frequency"})).unwrap();
    s.execute("layer.new.layer", json!({"name": "Top"})).unwrap();
    s.execute("image.applyImage", steps[8].1.clone()).unwrap();
    let e = s.execute("image.applyImage", json!({"source": {"layer": "Nope"}})).unwrap_err().to_string();
    assert!(e.contains("Nope"), "{e}");
}

fn channel_ord(value: &str) -> D {
    D::Reference(vec![ReferenceItem::Enumerated { class: class("Chnl"), type_id: Id::new("Chnl"), value: code(value) }])
}

fn fsel(t: D) -> Descriptor {
    desc("null").with("null", D::Reference(vec![ReferenceItem::Property { class: class("Chnl"), key: Id::new("fsel") }])).with("T   ", t)
}

fn set_of(name: &str, actions: Vec<(&str, Vec<AtnItem>)>) -> String {
    let file = AtnFile {
        name: UnicodeString::new_nul(name),
        expanded: false,
        actions: actions.into_iter().map(|(n, items)| AtnAction { name: UnicodeString::new_nul(n), items, ..Default::default() }).collect(),
        warnings: vec![],
    };
    b64(&atn::write(&file).unwrap())
}

/// The step shapes of the TrueGrit Texture Supply sets (Distress Press, Debaser, Transparency,
/// Halftone Zine Machine), rebuilt here as Photoshop records them, import and play: the
/// distressing filters land on the targeted layer mask, Remove White leaves the white
/// transparent, nudges in points follow the resolution, and Step Backward undoes a step.
#[test]
fn distressing_and_remove_white_shapes_import_and_play() {
    let ord = |cls: &str, ty: &str, v: &str| D::Reference(vec![ReferenceItem::Enumerated { class: class(cls), type_id: Id::new(ty), value: code(v) }]);
    let black = D::Descriptor(desc("RGBC").with("Rd  ", D::Double(0.0)).with("Grn ", D::Double(0.0)).with("Bl  ", D::Double(0.0)));
    let fill = desc("null")
        .with("null", class_ref("contentLayer"))
        .with("Usng", D::Descriptor(desc("contentLayer").with("Type", D::Descriptor(desc("solidColorLayer").with("Clr ", black)))));
    let refine = desc("null")
        .with("refineEdgeBorderRadius", px(0.0))
        .with("refineEdgeBorderContrast", pct(100.0))
        .with("refineEdgeSmooth", long(2.0))
        .with("refineEdgeFeatherRadius", px(0.0))
        .with("refineEdgeChoke", pct(0.0))
        .with("refineEdgeAutoRadius", D::Boolean(false))
        .with("refineEdgeDecontaminate", D::Boolean(false))
        .with("refineEdgeOutput", enm("refineEdgeOutput", "refineEdgeOutputUserMask"));
    let distress = vec![
        item("setd", "Set", Some(fsel(channel_ord("Trsp")))),
        item("Hd  ", "Hide", Some(desc("null").with("null", D::List(vec![target_ref("Lyr ")])))),
        item("Mk  ", "Make", Some(fill)),
        item("slct", "Select", Some(desc("null").with("null", channel_ord("Msk")).with("MkVs", D::Boolean(false)))),
        item("refineSelectionEdge", "Select and Mask", Some(refine)),
        item("Rple", "Ripple", Some(desc("null").with("Amnt", long(50.0)).with("RplS", enm("RplS", "Sml")))),
        item("Crst", "Crystallize", Some(desc("null").with("ClSz", long(3.0)))),
        item("GEfc", "Filter Gallery", Some(desc("null").with("GEfk", enm("GEft", "Spt")).with("SprR", long(7.0)).with("Smth", long(15.0)))),
        item("slct", "Select", Some(desc("null").with("null", ord("Chnl", "Ordn", "Trgt")).with("MkVs", D::Boolean(false)))),
        item("setd", "Set", Some(fsel(ord("Chnl", "Ordn", "Trgt")))),
    ];
    let bckg = D::Reference(vec![ReferenceItem::Property { class: class("Lyr "), key: Id::new("Bckg") }]);
    let remove_white = vec![
        item("CnvM", "Convert Mode", Some(desc("null").with("T   ", D::Class(class("Grys"))))),
        item(
            "setd",
            "Set",
            Some(desc("null").with("null", bckg).with("T   ", D::Descriptor(desc("Lyr ").with("Opct", pct(100.0)).with("Md  ", enm("BlnM", "Nrml"))))),
        ),
        item("setd", "Set", Some(fsel(ord("Chnl", "Ordn", "Trgt")))),
        item("Dlt ", "Delete", None),
        item("setd", "Set", Some(fsel(enm("Ordn", "None")))),
    ];
    let pt = |v: f64| D::UnitFloat { unit: PT, value: v };
    let nudge = vec![item(
        "move",
        "Move",
        Some(desc("null").with("null", target_ref("Lyr ")).with("T   ", D::Descriptor(desc("Ofst").with("Hrzn", pt(0.36)).with("Vrtc", pt(-0.36))))),
    )];
    let undo = vec![item("Invs", "Inverse", None), item("slct", "Select", Some(desc("null").with("null", ord("HstS", "Ordn", "Prvs"))))];
    let data = set_of("TrueGrit", vec![("Distress", distress), ("Remove White", remove_white), ("Nudge", nudge), ("Undo", undo)]);

    let mut s = Session::new();
    s.execute("file.new", json!({"width": 120, "height": 80, "resolution": 200})).unwrap();
    s.execute("layer.new.layer", json!({"name": "Ink"})).unwrap();
    s.execute("select.rect", json!({"x": 20, "y": 20, "width": 80, "height": 40})).unwrap();
    s.execute("edit.fill", json!({"contents": "color", "color": "#000000"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    let r = s.execute("actions.load", json!({"data": data})).unwrap();
    // Only the re-selection of the targeted channel stays a Photoshop step, switched off.
    let unsupported: Vec<&str> = r["unsupported"].as_array().unwrap().iter().map(|u| u["action"].as_str().unwrap()).collect();
    assert_eq!(unsupported, ["Distress"], "{r}");

    let r = s.execute("actions.play", json!({"action": "Distress"})).unwrap();
    assert!(r.get("failed").is_none(), "{r}");
    let st = s.active().unwrap();
    let fill = st.doc.layer(st.active_layer.unwrap()).unwrap();
    let mask = &fill.mask.as_ref().expect("the fill layer is masked by the ink").surface;
    // The rectangle's straight top edge is roughened: pixels along it differ.
    let edge: Vec<f32> = (20..100).map(|x| mask.sample_channel(x, 20, 0)).collect();
    assert!(edge.iter().any(|v| *v < 0.5) && edge.iter().any(|v| *v > 0.5), "{edge:?}");
    assert!(mask.sample_channel(60, 40, 0) > 0.9 && mask.sample_channel(5, 5, 0) < 0.1);
    assert!(st.doc.selection.is_some(), "the targeted mask loads as the selection");
    assert_eq!(st.channel_view.target, crate::channel_cmds::ChannelTarget::LayerMask);

    let mut s = Session::new();
    s.execute("file.new", json!({"width": 40, "height": 30})).unwrap();
    s.execute("select.rect", json!({"x": 10, "y": 10, "width": 20, "height": 10})).unwrap();
    s.execute("edit.fill", json!({"contents": "color", "color": "#202020"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    s.execute("actions.load", json!({"data": data})).unwrap();
    let r = s.execute("actions.play", json!({"action": "Remove White"})).unwrap();
    assert!(r.get("failed").is_none(), "{r}");
    let d = &s.active().unwrap().doc;
    assert_eq!(d.mode, photocraft_color::ColorMode::Grayscale);
    let l = d.layers.last().unwrap();
    assert!(!d.is_background(l.id), "Layer From Background");
    let surf = l.surface().unwrap();
    assert!(surf.sample_channel(2, 2, 1) < 0.01, "white is cleared");
    // #202020 is about 22 in the default (Dot Gain 20 %-compatible) working gray.
    let ink = surf.sample_channel(20, 15, 1);
    assert!((ink - (1.0 - 22.0 / 255.0)).abs() < 0.02, "the ink keeps its darkness as opacity: {ink}");
    // An image that opened as a normal layer has no Background: the step leaves it as it is.
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 40, "height": 30, "background": "transparent"})).unwrap();
    s.execute("edit.fill", json!({"contents": "white"})).unwrap();
    s.execute("actions.load", json!({"data": data})).unwrap();
    let r = s.execute("actions.play", json!({"action": "Remove White"})).unwrap();
    assert!(r.get("failed").is_none(), "{r}");
    let surf = s.active().unwrap().doc.layers.last().unwrap().surface().unwrap().clone();
    assert!(surf.sample_channel(2, 2, 1) < 0.01, "white is cleared without a Background");

    let mut s = Session::new();
    s.execute("file.new", json!({"width": 40, "height": 30, "resolution": 200})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("select.rect", json!({"x": 10, "y": 10, "width": 5, "height": 5})).unwrap();
    s.execute("edit.fill", json!({"contents": "color", "color": "#ff0000"})).unwrap();
    s.execute("actions.load", json!({"data": data})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    assert!(s.execute("actions.play", json!({"action": "Nudge"})).unwrap().get("failed").is_none());
    let st = s.active().unwrap();
    let b = st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().content_bounds();
    assert_eq!((b.x0, b.y0), (11, 9), "0.36 pt is 1 px at 200 ppi");
    s.execute("select.rect", json!({"x": 0, "y": 0, "width": 5, "height": 5})).unwrap();
    let before = s.active().unwrap().doc.selection.clone();
    assert!(s.execute("actions.play", json!({"action": "Undo"})).unwrap().get("failed").is_none());
    assert_eq!(s.active().unwrap().doc.selection, before, "Step Backward undid the Inverse");
}

/// Remove White on art in its own layer above a white Background: the Background becomes a
/// normal layer, but the art layer stays targeted, so the white is cleared from the art.
#[test]
fn remove_white_clears_the_targeted_layer_not_the_converted_background() {
    let ord = |cls: &str, ty: &str, v: &str| D::Reference(vec![ReferenceItem::Enumerated { class: class(cls), type_id: Id::new(ty), value: code(v) }]);
    let bckg = D::Reference(vec![ReferenceItem::Property { class: class("Lyr "), key: Id::new("Bckg") }]);
    let steps = vec![
        item("CnvM", "Convert Mode", Some(desc("null").with("T   ", D::Class(class("Grys"))))),
        item(
            "setd",
            "Set",
            Some(desc("null").with("null", bckg).with("T   ", D::Descriptor(desc("Lyr ").with("Opct", pct(100.0)).with("Md  ", enm("BlnM", "Nrml"))))),
        ),
        item("setd", "Set", Some(fsel(ord("Chnl", "Ordn", "Trgt")))),
        item("Dlt ", "Delete", None),
        item("setd", "Set", Some(fsel(enm("Ordn", "None")))),
    ];
    let data = set_of("TrueGrit", vec![("Remove White", steps)]);
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 40, "height": 30})).unwrap();
    let art = s.execute("layer.new.layer", json!({"name": "Art"})).unwrap()["layer"].as_u64().unwrap();
    s.execute("edit.fill", json!({"contents": "white"})).unwrap();
    s.execute("select.rect", json!({"x": 10, "y": 10, "width": 20, "height": 10})).unwrap();
    s.execute("edit.fill", json!({"contents": "color", "color": "#000000"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    s.execute("actions.load", json!({"data": data})).unwrap();
    let r = s.execute("actions.play", json!({"action": "Remove White"})).unwrap();
    assert!(r.get("failed").is_none(), "{r}");
    let st = s.active().unwrap();
    assert_eq!(st.active_layer.map(|l| l.0), Some(art), "the art layer stays targeted");
    let art = st.doc.layer(photocraft_doc::LayerId(art)).unwrap().surface().unwrap();
    assert!(art.sample_channel(2, 2, 1) < 0.01, "the art's white is cleared");
    assert!(art.sample_channel(20, 15, 1) > 0.99, "the black stays");
    let bottom = st.doc.layers.first().unwrap();
    assert!(!st.doc.is_background(bottom.id), "the Background became a normal layer");
    assert!(bottom.surface().unwrap().sample_channel(2, 2, 1) > 0.99, "and kept its pixels");
}
