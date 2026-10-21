use super::*;

fn tmp(name: &str) -> String {
    let d = std::env::temp_dir().join(format!("photocraft-automate-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d.to_string_lossy().into_owned()
}

/// Writes `n` small PNGs (w×h, grey level i·40) into `dir`.
fn images(dir: &str, n: usize, w: u32, h: u32) -> Vec<String> {
    (0..n)
        .map(|i| {
            let mut s = Session::new();
            let v = (i as f32 * 40.0 + 20.0) / 255.0;
            s.execute(
                "file.new",
                json!({"width": w, "height": h, "background": format!("#{:02x}{:02x}{:02x}", (v * 255.0) as u8, (v * 255.0) as u8, (v * 255.0) as u8)}),
            )
            .unwrap();
            let path = format!("{dir}/img{i}.png");
            crate::file_cmds::save_doc(&s.active().unwrap().doc, &path, None).unwrap();
            path
        })
        .collect()
}

#[test]
fn scripts_parse_json_and_lines() {
    let a = parse_script(r#"[["file.new", {"width": 10}], {"command": "layer.flattenImage"}, "edit.undo"]"#).unwrap();
    assert_eq!(a.len(), 3);
    assert_eq!(a[0].1["width"], 10);
    let b = parse_script("# make a doc\nfile.new {\"width\": 12, \"height\": 8}\n\nimage.imageRotation.90cw\n").unwrap();
    assert_eq!(b.len(), 2);
    assert_eq!(b[1].0, "image.imageRotation.90cw");
    let c = parse_script(r#"{"photocraftDroplet":1,"action":{"steps":[["file.new",{}]]}}"#).unwrap();
    assert_eq!(c.len(), 1);
    assert!(parse_script("file.new {bad json").is_err());
}

#[test]
fn browse_runs_a_script_file() {
    let dir = tmp("browse");
    let path = format!("{dir}/make.txt");
    std::fs::write(&path, "file.new {\"width\": 30, \"height\": 20}\nimage.imageRotation.90cw\n").unwrap();
    let mut s = Session::new();
    let r = s.execute("file.scripts.browse", json!({"path": path})).unwrap();
    assert_eq!(r["ok"], true);
    let d = &s.active().unwrap().doc;
    assert_eq!((d.size.width, d.size.height), (20, 30));
    assert!(s.execute("file.scripts.browse", json!({"script": "no.such.command"})).is_err());
    // A failing step stops the script and is reported.
    let r = s.execute("file.scripts.browse", json!({"steps": [["layer.delete", {"layer": u64::MAX}], ["image.imageRotation.90cw", {}]]})).unwrap();
    assert_eq!(r["results"].as_array().unwrap().len(), 1);
}

#[test]
fn recording_browse_script_captures_each_effect_once() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 12, "height": 8})).unwrap();
    s.execute("actions.record", json!({"name": "Rotate script"})).unwrap();
    let run = s.execute("file.scripts.browse", json!({"steps": [["image.imageRotation.90cw", {}]]})).unwrap();
    assert_eq!(run["ok"], true);
    assert_eq!((s.active().unwrap().doc.size.width, s.active().unwrap().doc.size.height), (8, 12));
    let done = s.execute("actions.stop", json!({})).unwrap();
    assert_eq!(done["steps"], 1, "Browse must not be recorded a second time");
    let recorded = s.execute("actions.get", json!({"action": "Rotate script"})).unwrap();
    assert_eq!(recorded["steps"], json!([["image.imageRotation.90cw", {}]]));

    s.execute("image.imageRotation.90ccw", json!({})).unwrap();
    assert_eq!((s.active().unwrap().doc.size.width, s.active().unwrap().doc.size.height), (12, 8));
    let replay = s.execute("actions.play", json!({"action": "Rotate script"})).unwrap();
    assert_eq!(replay["ran"], 1);
    assert_eq!((s.active().unwrap().doc.size.width, s.active().unwrap().doc.size.height), (8, 12));
}

#[test]
fn standalone_browse_script_still_runs_and_journals_concrete_commands() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 12, "height": 8})).unwrap();
    let start = s.journal.len();
    let result = s.execute("file.scripts.browse", json!({"steps": [["image.imageRotation.90cw", {}]]})).unwrap();
    assert_eq!(result["ok"], true);
    assert_eq!(s.journal.len(), start + 2, "the child and Browse retain their diagnostic journal entries");
    assert_eq!(s.journal[start].0, "image.imageRotation.90cw");
    assert_eq!((s.active().unwrap().doc.size.width, s.active().unwrap().doc.size.height), (8, 12));
}

#[test]
fn script_events_fire_on_new_and_close() {
    let mut s = Session::new();
    let r = s
        .execute("file.scripts.scriptEventsManager", json!({"add": {"event": "newDocument", "steps": [["image.imageRotation.90cw", {}]], "name": "Rotate"}}))
        .unwrap();
    assert_eq!(r["enabled"], true);
    assert_eq!(r["bindings"].as_array().unwrap().len(), 1);
    assert!(s.execute("file.scripts.scriptEventsManager", json!({"add": {"event": "nope", "steps": []}})).is_err());
    assert!(s.execute("file.scripts.scriptEventsManager", json!({"add": {"event": "print", "steps": [["no.such", {}]]}})).is_err());
    s.execute("file.new", json!({"width": 40, "height": 10})).unwrap();
    let d = &s.active().unwrap().doc;
    assert_eq!((d.size.width, d.size.height), (10, 40), "the New Document action rotated it");
    assert_eq!(s.file_menu.event_log.len(), 1);
    // Events don't re-enter: the script's own commands fire nothing.
    s.execute("file.scripts.scriptEventsManager", json!({"add": {"event": "closeDocument", "steps": [["file.new", {"width": 5, "height": 5}]]}})).unwrap();
    s.execute("file.close", json!({})).unwrap();
    assert_eq!(s.documents().len(), 1, "closing ran the script, which made a new document");
    assert_eq!(s.documents()[0].doc.size.width, 5, "…without the New Document rotation");
    // Disabled: nothing fires.
    s.execute("file.scripts.scriptEventsManager", json!({"enabled": false})).unwrap();
    s.execute("file.new", json!({"width": 40, "height": 10})).unwrap();
    assert_eq!(s.active().unwrap().doc.size.width, 40);
    let r = s.execute("file.scripts.scriptEventsManager", json!({"removeAll": true})).unwrap();
    assert!(r["bindings"].as_array().unwrap().is_empty());
    assert_eq!(r["events"].as_array().unwrap().len(), EVENTS.len());
}

#[test]
fn droplets_are_written_and_run() {
    let dir = tmp("droplet");
    let inputs = images(&dir, 2, 40, 20);
    let mut s = Session::new();
    let path = format!("{dir}/Rotate");
    let r = s
        .execute(
            "file.automate.createDroplet",
            json!({"path": path, "steps": [["image.imageRotation.90cw", {}]], "format": "png", "output": format!("{dir}/out")}),
        )
        .unwrap();
    let dp = r["path"].as_str().unwrap().to_string();
    assert!(dp.ends_with("Rotate.pcdroplet"));
    let v: Value = serde_json::from_slice(&std::fs::read(&dp).unwrap()).unwrap();
    assert_eq!(v["photocraftDroplet"], 1);
    #[cfg(unix)]
    {
        let shim = r["shim"].as_str().unwrap();
        let text = std::fs::read_to_string(shim).unwrap();
        assert!(text.starts_with("#!/bin/sh") && text.contains("droplet"));
        use std::os::unix::fs::PermissionsExt;
        assert_ne!(std::fs::metadata(shim).unwrap().permissions().mode() & 0o111, 0, "executable");
    }
    let r = s.execute("file.automate.runDroplet", json!({"droplet": dp, "input": [dir.clone()]})).unwrap();
    assert_eq!(r["files"].as_array().unwrap().len(), inputs.len(), "{r}");
    let out = photocraft_codecs::decode(&std::fs::read(format!("{dir}/out/img0.png")).unwrap()).unwrap();
    assert_eq!(out.dimensions(), (20, 40));
    assert!(s.execute("file.automate.createDroplet", json!({"path": path, "steps": [["bogus", {}]]})).is_err());
}

#[cfg(unix)]
#[test]
fn droplet_shim_passes_shell_metacharacters_as_a_literal_path() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tmp("droplet-shell-quote");
    let path = format!("{dir}/Droplet's \"quote\" $HOME `printf expanded` $(touch shim-injected) \\\nline");
    let mut session = Session::new();
    let result = session.execute("file.automate.createDroplet", json!({"path": path, "steps": [["file.new", {}]]})).unwrap();
    let droplet = result["path"].as_str().unwrap();
    let shim = result["shim"].as_str().unwrap();

    let cli = format!("{dir}/cli stub");
    let capture = format!("{dir}/captured path");
    let marker = format!("{dir}/shim-injected");
    let input = format!("{dir}/input file");
    std::fs::write(
        &cli,
        "#!/bin/sh\nif [ \"$#\" -ne 3 ] || [ \"$1\" != droplet ] || [ \"$3\" != \"$PHOTOCRAFT_EXPECTED_INPUT\" ]; then exit 90; fi\nprintf '%s' \"$2\" > \"$PHOTOCRAFT_CAPTURE\"\n",
    )
    .unwrap();
    std::fs::set_permissions(&cli, std::fs::Permissions::from_mode(0o755)).unwrap();
    let output = std::process::Command::new(shim)
        .arg(&input)
        .current_dir(&dir)
        .env("PHOTOCRAFT_CLI", &cli)
        .env("PHOTOCRAFT_CAPTURE", &capture)
        .env("PHOTOCRAFT_EXPECTED_INPUT", &input)
        .output()
        .unwrap();

    assert!(output.status.success(), "shim exited {:?}: {}", output.status, String::from_utf8_lossy(&output.stderr));
    assert!(!std::path::Path::new(&marker).exists(), "filename command substitution ran");
    let expected = std::fs::canonicalize(droplet).unwrap();
    assert_eq!(std::fs::read(&capture).unwrap(), expected.to_string_lossy().as_bytes());
}

#[test]
fn droplets_carry_the_actions_their_play_steps_call() {
    use crate::actions_cmds::Action;
    let dir = tmp("droplet-called");
    let inputs = images(&dir, 2, 40, 20);
    let mut s = Session::new();
    let act = |name: &str, steps: Value| -> Action { serde_json::from_value(json!({"name": name, "steps": steps})).unwrap() };
    s.actions.list = vec![
        act("Unrelated", json!([["image.imageRotation.180", {}]])),
        act("Rotate", json!([["image.imageRotation.90cw", {}]])),
        // Calls Rotate by index (as the panel records it) and itself (caught at run time).
        act("Outer", json!([["actions.play", {"action": 1}]])),
        act("Loop", json!([["actions.play", {"action": "Loop"}]])),
    ];
    let make = |s: &mut Session, name: &str, steps: Value| {
        let path = format!("{dir}/{name}");
        let out = format!("{dir}/out-{name}");
        s.execute("file.automate.createDroplet", json!({"path": path, "steps": steps, "format": "png", "output": out, "shim": false})).unwrap()["path"]
            .as_str()
            .unwrap()
            .to_string()
    };
    let dp = make(&mut s, "Outer", json!([["actions.play", {"action": "Outer"}]]));
    let looping = make(&mut s, "Loop", json!([["actions.play", {"action": "Loop"}]]));
    let v: Value = serde_json::from_slice(&std::fs::read(&dp).unwrap()).unwrap();
    assert!(v.get("actions").is_none(), "the CLI reads `actions` as an action set");
    let names: Vec<&str> = v["calledActions"].as_array().unwrap().iter().map(|a| a["name"].as_str().unwrap()).collect();
    assert_eq!(names, ["Rotate", "Outer"], "only the called actions, transitively");

    // A fresh session, as `photocraft-cli droplet` and the droplet shim use.
    let mut fresh = Session::new();
    let r = fresh.execute("file.automate.runDroplet", json!({"droplet": dp, "input": [dir.clone()]})).unwrap();
    assert_eq!(r["files"].as_array().unwrap().len(), inputs.len(), "{r}");
    let out = photocraft_codecs::decode(&std::fs::read(format!("{dir}/out-Outer/img0.png")).unwrap()).unwrap();
    assert_eq!(out.dimensions(), (20, 40));
    let r = fresh.execute("file.automate.runDroplet", json!({"droplet": looping, "input": [dir.clone()]})).unwrap();
    assert_eq!(r["errors"].as_array().unwrap().len(), inputs.len(), "a self-call fails instead of looping: {r}");

    // Droplets written before `calledActions` still run.
    let old = format!("{dir}/old.pcdroplet");
    std::fs::write(
        &old,
        json!({"photocraftDroplet": 1, "action": {"steps": [["image.imageRotation.90cw", {}]]}, "options": {"output": format!("{dir}/out-old")}}).to_string(),
    )
    .unwrap();
    let r = fresh.execute("file.automate.runDroplet", json!({"droplet": old, "input": [dir.clone()]})).unwrap();
    assert_eq!(r["files"].as_array().unwrap().len(), inputs.len(), "{r}");
}

#[test]
fn batch_and_droplets_skip_recorded_view_steps() {
    let dir = tmp("view-steps");
    images(&dir, 1, 40, 20);
    let steps = json!([["view.zoomIn", {}], ["image.imageRotation.90cw", {}], ["view.fitOnScreen", {}]]);
    let mut s = Session::new();
    let r = s.execute("file.automate.batch", json!({"steps": steps, "input": dir.clone(), "output": format!("{dir}/batch"), "format": "png"})).unwrap();
    assert!(r["errors"].as_array().unwrap().is_empty(), "{r}");
    let out = photocraft_codecs::decode(&std::fs::read(format!("{dir}/batch/img0.png")).unwrap()).unwrap();
    assert_eq!(out.dimensions(), (20, 40));
    let r = s.execute("file.automate.createDroplet", json!({"path": format!("{dir}/Rotate"), "steps": steps})).unwrap();
    let v: Value = serde_json::from_slice(&std::fs::read(r["path"].as_str().unwrap()).unwrap()).unwrap();
    assert_eq!(v["action"]["steps"], json!([["image.imageRotation.90cw", {}]]));
}

#[test]
fn statistics_makes_a_stack_mode_smart_object() {
    let dir = tmp("stats");
    let files = images(&dir, 3, 16, 12);
    let mut s = Session::new();
    let r = s.execute("file.scripts.statistics", json!({"mode": "mean", "input": files})).unwrap();
    assert_eq!(r["images"], 3);
    let d = &s.active().unwrap().doc;
    assert_eq!(d.layers.len(), 1);
    let LayerContent::Smart(so) = &d.layers[0].content else { panic!("not a smart object") };
    assert_eq!(so.stack_mode, Some(photocraft_doc::StackMode::Mean));
    // Mean of 20, 60, 100 = 60.
    let px = photocraft_compose::flatten(d).px[0];
    assert!((px[0] * 255.0 - 60.0).abs() < 1.5, "{px:?}");
    assert!(s.execute("file.scripts.statistics", json!({"mode": "average", "input": dir})).is_err());
}

#[test]
fn stacked_smart_object_holds_the_loaded_layers_at_its_top_level() {
    // Like Photoshop, Load Files into Stack › Create Smart Object (Statistics without alignment) and
    // Statistics with alignment embed one layer per file, bottom to top, with no wrapper group.
    let dir = tmp("stack-top");
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 96, "height": 72, "background": "#606060"})).unwrap();
    s.edit("texture", |doc, active| {
        let surf = doc.layer_mut(active.ok_or(EngineError::NoDocument)?).and_then(Layer::surface_mut).ok_or(EngineError::NoDocument)?;
        for k in 0..40 {
            let (x, y, v) = ((k * 37) % 88, (k * 23) % 66, (k % 7) as f32 / 6.0);
            surf.fill_rect(photocraft_geom::Rect::new(x, y, x + 3 + k % 5, y + 2 + k % 4), &[v, 1.0 - v, 0.5, 1.0]);
        }
        Ok(())
    })
    .unwrap();
    let source = photocraft_compose::flatten(&s.active().unwrap().doc);
    let files: Vec<String> = (0..3)
        .map(|i| {
            let path = format!("{dir}/img{i}.png");
            crate::file_cmds::save_doc(&s.active().unwrap().doc, &path, None).unwrap();
            path
        })
        .collect();
    for align in [false, true] {
        let mut s = Session::new();
        s.execute("file.scripts.statistics", json!({"mode": "median", "input": files, "align": align})).unwrap();
        let d = &s.active().unwrap().doc;
        let LayerContent::Smart(so) = &d.layers[0].content else { panic!("not a smart object") };
        let photocraft_doc::SmartSource::Embedded { file_name, bytes } = &so.source else { panic!("not embedded") };
        let inner = crate::smart_cmds::decode_source(file_name, bytes).unwrap();
        let names: Vec<&str> = inner.layers.iter().map(|l| l.name.as_str()).collect();
        assert_eq!(names, ["img0.png", "img1.png", "img2.png"], "align {align}");
        assert_eq!(photocraft_compose::flatten(d).px, source.px, "align {align}");
    }
}

#[test]
fn contact_sheet_places_thumbnails_with_captions() {
    let dir = tmp("contact");
    images(&dir, 5, 60, 30);
    for depth in [8, 16] {
        let mut s = Session::new();
        let r = s
            .execute("file.automate.contactSheetII", json!({"input": dir, "units": "pixels", "width": 400, "height": 300, "resolution": 72, "columns": 2, "rows": 2, "depth": depth, "fontSize": 10, "rotateForBestFit": true}))
            .unwrap();
        assert_eq!(r["pages"], 2, "5 images on 2×2 sheets");
        assert_eq!(r["images"], 5);
        let first = r["documents"][0].as_u64().unwrap() as usize;
        let d = &s.documents()[first].doc;
        assert_eq!((d.size.width, d.size.height), (400, 300));
        let names: Vec<&str> = d.layers.iter().map(|l| l.name.as_str()).collect();
        assert!(names.contains(&"img0") && names.contains(&"img3"), "{names:?}");
        let texts = d.layers.iter().filter(|l| matches!(l.content, LayerContent::Text(_))).count();
        assert_eq!(texts, 4, "a caption per image");
        let thumb = d.layers.iter().find(|l| l.name == "img0" && matches!(l.content, LayerContent::Raster(_))).unwrap();
        let b = thumb.surface().unwrap().content_bounds();
        assert!(b.width() <= 200 && b.height() < 150, "fits its cell: {b:?}");
        // Flattened variant.
        let r = s
            .execute(
                "file.automate.contactSheetII",
                json!({"input": dir, "units": "pixels", "width": 200, "height": 200, "columns": 3, "rows": 2, "caption": false, "flatten": true}),
            )
            .unwrap();
        let d = &s.documents()[r["documents"][0].as_u64().unwrap() as usize].doc;
        assert_eq!(d.layers.len(), 1);
    }
}

#[test]
fn a_failed_step_inside_a_called_action_fails_scripts_and_batch() {
    use crate::actions_cmds::Action;
    let dir = tmp("nested-failure");
    images(&dir, 1, 8, 4);
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 8, "height": 6})).unwrap();
    s.actions.list.push(Action {
        name: "Rotate".into(),
        steps: vec![("image.imageRotation.90cw".into(), json!({}))],
        disabled: Default::default(),
        ..Default::default()
    });
    s.actions.list.push(Action {
        name: "Bad".into(),
        steps: vec![("layer.delete".into(), json!({"layer": u64::MAX}))],
        disabled: Default::default(),
        ..Default::default()
    });
    // A script stops at the failed call and reports it.
    let r = s.execute("file.scripts.browse", json!({"steps": [["actions.play", {"action": "Bad"}], ["image.imageRotation.90cw", {}]]})).unwrap();
    assert_eq!(r["ok"], false, "{r}");
    assert_eq!(r["results"].as_array().unwrap().len(), 1);
    // Batch plays called actions, and a failure inside one is a file error, not a saved result.
    let batch = |s: &mut Session, action: &str| {
        let p = json!({"steps": [["actions.play", {"action": action}]], "input": dir.clone(), "output": format!("{dir}/{action}"), "format": "png"});
        s.execute("file.automate.batch", p).unwrap()
    };
    let r = batch(&mut s, "Rotate");
    assert!(r["errors"].as_array().unwrap().is_empty(), "{r}");
    let r = batch(&mut s, "Bad");
    assert_eq!(r["files"], json!([]), "{r}");
    assert!(r["errors"][0]["error"].as_str().unwrap().contains("no such layer"), "{r}");
    // An action that batches itself stops as a recursive call instead of overflowing the stack.
    let inner = json!({"steps": [["actions.play", {"action": "Loop"}]], "input": dir.clone(), "output": format!("{dir}/inner"), "format": "png"});
    s.actions.list.push(Action { name: "Loop".into(), steps: vec![("file.automate.batch".into(), inner)], disabled: Default::default(), ..Default::default() });
    batch(&mut s, "Loop");
}

/// #4671: Load Files into Stack's Attempt to Automatically Align Source Images runs Auto-Align
/// Layers on the loaded layers, then (with Create Smart Object) puts them in one smart object;
/// without the smart object they stay loose layers, selected.
#[test]
fn load_files_into_stack_aligns_before_the_smart_object() {
    let dir = tmp("stack-align");
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 96, "height": 72, "background": "#606060"})).unwrap();
    s.edit("texture", |doc, active| {
        let surf = doc.layer_mut(active.ok_or(EngineError::NoDocument)?).and_then(Layer::surface_mut).ok_or(EngineError::NoDocument)?;
        for k in 0..40 {
            let (x, y, v) = ((k * 37) % 88, (k * 23) % 66, (k % 7) as f32 / 6.0);
            surf.fill_rect(photocraft_geom::Rect::new(x, y, x + 3 + k % 5, y + 2 + k % 4), &[v, 1.0 - v, 0.5, 1.0]);
        }
        Ok(())
    })
    .unwrap();
    let files: Vec<String> = (0..2)
        .map(|i| {
            let path = format!("{dir}/img{i}.png");
            crate::file_cmds::save_doc(&s.active().unwrap().doc, &path, None).unwrap();
            path
        })
        .collect();
    let mut s = Session::new();
    let r = s.execute("file.scripts.loadFilesIntoStack", json!({"paths": files, "align": true})).unwrap();
    assert_eq!(r["layers"], 2);
    let st = s.active().unwrap();
    assert_eq!(st.doc.layers.len(), 2, "loose layers");
    assert!(st.doc.layers.iter().all(|l| matches!(l.content, LayerContent::Raster(_))));
    assert_eq!(st.selected_layers().len(), 2, "Auto-Align ran on them all");
    let mut s = Session::new();
    s.execute("file.scripts.loadFilesIntoStack", json!({"paths": files, "align": true, "createSmartObject": true})).unwrap();
    let d = &s.active().unwrap().doc;
    assert_eq!(d.layers.len(), 1);
    let LayerContent::Smart(_) = &d.layers[0].content else { panic!("not a smart object") };
    // A single file has nothing to align with: it just loads.
    let mut s = Session::new();
    s.execute("file.scripts.loadFilesIntoStack", json!({"paths": [files[0].clone()], "align": true})).unwrap();
    assert_eq!(s.active().unwrap().doc.layers.len(), 1);
}
