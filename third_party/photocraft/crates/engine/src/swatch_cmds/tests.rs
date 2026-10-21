use super::*;
use photocraft_paint::tile::{b64_decode, b64_encode};

fn names(s: &Session) -> Vec<String> {
    s.presets.swatches.iter().flat_map(|g| g.items.iter().map(|i| i.name.clone())).collect()
}

fn export_bytes(s: &mut Session, p: Value) -> Vec<u8> {
    let r = s.execute("swatches.export", p).unwrap();
    b64_decode(r["data"].as_str().unwrap()).unwrap()
}

#[test]
fn defaults_and_listing() {
    let mut s = Session::new();
    let r = s.execute("swatches.list", json!({})).unwrap();
    assert_eq!(r["count"], 46);
    assert_eq!(r["groups"][0]["name"], "Grays");
    assert_eq!(r["groups"][0]["swatches"][0], json!({"name": "Black", "model": "rgb", "values": [0.0, 0.0, 0.0], "hex": "#000000"}));
    assert_eq!(r["groups"][1]["swatches"][2]["hex"], "#00ff00", "RGB Green is exact (#4302)");
    assert_eq!(r["groups"][3]["swatches"][0]["hex"], "#e62828");
}

#[test]
fn add_from_foreground_and_explicit_models() {
    let mut s = Session::new();
    s.tools.foreground = [1.0, 0.5, 0.0, 1.0];
    let r = s.execute("swatches.add", json!({})).unwrap();
    assert_eq!((r["name"].as_str(), r["group"].as_str()), (Some("Color Swatch 1"), Some("Grays")));
    assert_eq!(s.presets.swatches[0].items.last().unwrap().color, SwatchColor::Rgb([1.0, 0.5, 0.0]));
    let r = s.execute("swatches.add", json!({})).unwrap();
    assert_eq!(r["name"], "Color Swatch 2");
    // CMYK into a new group at the front, spot.
    let r = s
        .execute("swatches.add", json!({"name": "Ink", "group": "Print", "index": 0, "kind": "spot", "color": {"model": "cmyk", "values": [100, 0, 0, 0]}}))
        .unwrap();
    assert_eq!(r["group"], "Print");
    let g = s.presets.swatches.last().unwrap();
    assert_eq!(g.items[0], Swatch { name: "Ink".into(), color: SwatchColor::Cmyk([1.0, 0.0, 0.0, 0.0]), kind: SwatchKind::Spot });
    // Names stay unique.
    let r = s.execute("swatches.add", json!({"name": "Ink"})).unwrap();
    assert_eq!(r["name"], "Ink 2");
}

#[test]
fn use_sets_foreground_and_background_through_colour_management() {
    let mut s = Session::new();
    s.execute("swatches.use", json!({"swatch": "Red"})).unwrap();
    assert_eq!(s.tools.foreground, [230.0 / 255.0, 40.0 / 255.0, 40.0 / 255.0, 1.0]);
    s.execute("swatches.use", json!({"index": 9, "target": "background"})).unwrap();
    assert_eq!(s.tools.background, [1.0, 1.0, 1.0, 1.0], "index 9 is White");
    s.execute("swatches.add", json!({"name": "Lab white", "color": {"model": "lab", "values": [100, 0, 0]}})).unwrap();
    let r = s.execute("swatches.use", json!({"swatch": "Lab white"})).unwrap();
    assert_eq!(r["hex"], "#ffffff");
    s.execute("swatches.add", json!({"name": "Paper", "color": {"model": "gray", "values": [0]}})).unwrap();
    s.execute("swatches.use", json!({"swatch": "Paper", "target": "background"})).unwrap();
    assert!(s.tools.background.iter().all(|v| (v - 1.0).abs() < 0.01), "{:?}", s.tools.background);
}

#[test]
fn rename_delete_move_and_groups() {
    let mut s = Session::new();
    s.execute("swatches.rename", json!({"swatch": "Black", "name": "Ink Black"})).unwrap();
    assert!(names(&s).contains(&"Ink Black".to_string()));
    assert!(s.execute("swatches.rename", json!({"swatch": "Ink Black", "name": "White"})).is_err(), "duplicate");
    // Reorder within the group, then into another group.
    s.execute("swatches.move", json!({"swatch": "White", "position": 0})).unwrap();
    assert_eq!(s.presets.swatches[0].items[0].name, "White");
    let r = s.execute("swatches.move", json!({"swatch": "White", "to": "Dark", "position": 1})).unwrap();
    assert_eq!(r, json!({"group": "Dark", "position": 1}));
    assert_eq!(s.presets.swatches[4].items[1].name, "White");
    // Out-of-range positions clamp.
    let r = s.execute("swatches.move", json!({"swatch": "White", "position": 999_999})).unwrap();
    assert_eq!(r["position"], 10);
    // New group from swatches, rename it, delete it keeping the swatches.
    let r = s.execute("swatches.newGroup", json!({"name": "Mine", "swatches": ["Red", "Blue"]})).unwrap();
    assert_eq!(r, json!({"group": "Mine", "moved": 2}));
    let r = s.execute("swatches.newGroup", json!({"name": "Mine"})).unwrap();
    assert_eq!(r["group"], "Mine 1");
    s.execute("swatches.renameGroup", json!({"group": "Mine", "name": "Favourites"})).unwrap();
    assert!(s.execute("swatches.renameGroup", json!({"group": "Favourites", "name": "Dark"})).is_err());
    let r = s.execute("swatches.deleteGroup", json!({"group": "Favourites", "keepSwatches": true})).unwrap();
    assert_eq!(r["kept"], 2);
    assert!(s.presets.swatches[4].items.iter().any(|i| i.name == "Red"), "kept in the group above");
    s.execute("swatches.deleteGroup", json!({"group": "Mine 1"})).unwrap();
    // Delete several by name, and one by index.
    let r = s.execute("swatches.delete", json!({"swatch": ["Red", "Blue", "Red"]})).unwrap();
    assert_eq!(r["deleted"].as_array().unwrap().len(), 2);
    let r = s.execute("swatches.delete", json!({"index": 0})).unwrap();
    assert_eq!(r["deleted"], json!(["Ink Black"]));
    assert_eq!(names(&s).len(), 43);
    // Reset restores the defaults; append adds them again with unique names.
    s.execute("swatches.reset", json!({})).unwrap();
    assert_eq!(names(&s).len(), 46);
    s.execute("swatches.reset", json!({"append": true})).unwrap();
    assert_eq!(names(&s).len(), 92);
    assert!(names(&s).contains(&"Black 2".to_string()));
    assert_eq!(s.presets.swatches[5].name, "Grays 1");
}

#[test]
fn changes_persist_with_the_preferences() {
    let mut s = Session::new();
    s.execute("swatches.add", json!({"name": "Keep me", "color": {"model": "lab", "values": [40, 12.5, -7.25]}})).unwrap();
    let saved = s.presets.to_json(&s);
    let mut t = Session::new();
    t.load_presets_json(saved);
    assert_eq!(t.presets.swatches, s.presets.swatches);
    // A prefs file from before swatches existed keeps the defaults.
    let mut u = Session::new();
    u.load_presets_json(json!({"gradients": null}));
    assert_eq!(u.presets.swatches, builtin());
}

#[test]
fn aco_round_trip_keeps_16_bit_values_and_books() {
    let swatches = vec![
        AcoSwatch { name: "R".into(), color: AcoColor::Rgb([65535, 12345, 1]) },
        AcoSwatch { name: "H".into(), color: AcoColor::Hsb([40000, 30000, 65535]) },
        AcoSwatch { name: "C".into(), color: AcoColor::Cmyk([65535, 1, 32768, 7]) },
        AcoSwatch { name: "L".into(), color: AcoColor::Lab { l: 5321, a: -12800, b: 12700 } },
        AcoSwatch { name: "G".into(), color: AcoColor::Gray(1234) },
        AcoSwatch { name: "Book".into(), color: AcoColor::Other { space: 3, values: [9, 8, 7, 6] } },
    ];
    let file = photocraft_psd::aco::write(&swatches).unwrap();
    let mut s = Session::new();
    let r = s.execute("swatches.import", json!({"data": b64_encode(&file), "group": "Pro", "mode": "replace"})).unwrap();
    assert_eq!((r["format"].as_str(), r["count"].as_u64()), (Some("aco"), Some(6)));
    assert_eq!(r["warnings"].as_array().unwrap().len(), 1, "the colour book is reported");
    assert_eq!(s.presets.swatches.len(), 1);
    assert_eq!(s.presets.swatches[0].items[3].color, SwatchColor::Lab([53.21, -128.0, 127.0]));
    assert_eq!(s.presets.swatches[0].items[4].color.model(), "gray");
    assert_eq!(export_bytes(&mut s, json!({"format": "aco"})), file, "byte-identical .aco");
    // On disk too, with the group taken from the file name.
    let dir = std::env::temp_dir().join(format!("pc-swatch-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("Brand Colors.aco");
    std::fs::write(&path, &file).unwrap();
    let r = s.execute("swatches.import", json!({"path": path.to_string_lossy()})).unwrap();
    assert_eq!(r["groups"], json!(["Brand Colors"]));
    assert_eq!(names(&s).iter().filter(|n| n.starts_with("R")).count(), 2, "imported names are made unique");
    let out = dir.join("out.ase");
    let r = s.execute("swatches.export", json!({"path": out.to_string_lossy(), "group": "Pro"})).unwrap();
    assert_eq!((r["format"].as_str(), r["count"].as_u64()), (Some("ase"), Some(5)), "the book can't go to .ase");
    assert!(std::fs::read(&out).unwrap().starts_with(b"ASEF"));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn ase_round_trip_keeps_groups_floats_and_kinds() {
    let src = AseFile {
        entries: vec![
            AseEntry::Swatch(AseSwatch { name: "Loose".into(), color: AseColor::Rgb([0.1, 0.2, 0.3]), kind: AseKind::Process }),
            AseEntry::Group {
                name: "Brand".into(),
                swatches: vec![
                    AseSwatch { name: "Spot ink".into(), color: AseColor::Cmyk([0.0, 0.85, 1.0, 0.05]), kind: AseKind::Spot },
                    AseSwatch { name: "Global lab".into(), color: AseColor::Lab([0.5, 20.0, -30.0]), kind: AseKind::Global },
                    AseSwatch { name: "Gray".into(), color: AseColor::Gray(0.25), kind: AseKind::Process },
                    AseSwatch { name: "Weird".into(), color: AseColor::Other { model: *b"XYZ ", data: vec![0; 12] }, kind: AseKind::Process },
                ],
            },
        ],
    };
    let bytes = photocraft_psd::ase::write(&src).unwrap();
    let mut s = Session::new();
    let r = s.execute("swatches.import", json!({"data": b64_encode(&bytes), "group": "Exchange"})).unwrap();
    assert_eq!(r["groups"], json!(["Exchange", "Brand"]));
    assert_eq!(r["count"], 4);
    assert_eq!(r["warnings"].as_array().unwrap().len(), 1, "the unknown model is reported");
    let brand = s.presets.swatches.iter().find(|g| g.name == "Brand").unwrap();
    assert_eq!(brand.items[0].kind, SwatchKind::Spot);
    assert_eq!(brand.items[1].color, SwatchColor::Lab([50.0, 20.0, -30.0]));
    assert_eq!(brand.items[2].color, SwatchColor::Gray(0.75));
    // Export both groups and read them back.
    let out = export_bytes(&mut s, json!({"format": "ase", "swatches": ["Loose", "Spot ink", "Global lab", "Gray"]}));
    let back = photocraft_psd::ase::parse(&out).unwrap();
    let flat: Vec<&AseSwatch> = back
        .entries
        .iter()
        .flat_map(|e| match e {
            AseEntry::Group { swatches, .. } => swatches.iter().collect::<Vec<_>>(),
            AseEntry::Swatch(s) => vec![s],
        })
        .collect();
    assert_eq!(flat.len(), 4);
    assert_eq!(flat[0].color, AseColor::Rgb([0.1, 0.2, 0.3]));
    assert_eq!(flat[1].color, AseColor::Cmyk([0.0, 0.85, 1.0, 0.05]));
    assert_eq!(flat[1].kind, AseKind::Spot);
    assert_eq!(flat[2].kind, AseKind::Global);
    let (AseColor::Lab(l), AseColor::Gray(g)) = (&flat[2].color, &flat[3].color) else { panic!("models") };
    assert!((l[0] - 0.5).abs() < 1e-6 && l[1] == 20.0 && l[2] == -30.0);
    assert!((g - 0.25).abs() < 1e-6);
    // HSB swatches become RGB in .ase.
    s.execute("swatches.add", json!({"name": "Hue", "color": {"model": "hsb", "values": [120, 100, 100]}})).unwrap();
    let out = export_bytes(&mut s, json!({"format": "ase", "swatches": ["Hue"]}));
    let back = photocraft_psd::ase::parse(&out).unwrap();
    assert_eq!(
        back.entries,
        vec![AseEntry::Group {
            name: "Grays".into(),
            swatches: vec![AseSwatch { name: "Hue".into(), color: AseColor::Rgb([0.0, 1.0, 0.0]), kind: AseKind::Process }]
        }]
    );
}

#[test]
fn bad_params_fail_gracefully() {
    let mut s = Session::new();
    let before = s.presets.swatches.clone();
    let cases: &[(&str, Value)] = &[
        ("swatches.add", json!({"color": "#12"})),
        ("swatches.add", json!({"color": {"model": "lab"}})),
        ("swatches.add", json!({"color": 5})),
        ("swatches.add", json!({"name": 7})),
        ("swatches.add", json!({"name": "   "})),
        ("swatches.add", json!({"group": []})),
        ("swatches.add", json!({"index": -1})),
        ("swatches.add", json!({"kind": "metallic"})),
        ("swatches.delete", json!({})),
        ("swatches.delete", json!({"swatch": "Nope"})),
        ("swatches.delete", json!({"swatch": ["Red", 5]})),
        ("swatches.delete", json!({"swatch": ["Red", "Nope"]})),
        ("swatches.delete", json!({"index": 46})),
        ("swatches.delete", json!({"index": "0"})),
        ("swatches.delete", json!({"swatch": "Red", "group": "Grays"})),
        ("swatches.rename", json!({"swatch": "Red"})),
        ("swatches.rename", json!({"swatch": "Red", "name": ""})),
        ("swatches.rename", json!({"name": "x"})),
        ("swatches.move", json!({"swatch": "Red", "to": "Nowhere"})),
        ("swatches.move", json!({"swatch": "Red", "position": -3})),
        ("swatches.move", json!({"index": 999_999})),
        ("swatches.newGroup", json!({"swatches": "Red"})),
        ("swatches.newGroup", json!({"swatches": ["Nope"]})),
        ("swatches.newGroup", json!({"name": 3})),
        ("swatches.renameGroup", json!({"group": "Nope", "name": "x"})),
        ("swatches.renameGroup", json!({"group": "Grays"})),
        ("swatches.deleteGroup", json!({})),
        ("swatches.deleteGroup", json!({"group": "Grays", "keepSwatches": "yes"})),
        ("swatches.use", json!({"swatch": "Red", "target": "middle"})),
        ("swatches.use", json!({})),
        ("swatches.reset", json!({"append": 1})),
        ("swatches.import", json!({})),
        ("swatches.import", json!({"path": "/nonexistent/x.aco"})),
        ("swatches.import", json!({"data": "%%%"})),
        ("swatches.import", json!({"data": b64_encode(b"junk junk")})),
        ("swatches.import", json!({"data": b64_encode(b"ASEF\0\x01\0\0\xff\xff\xff\xff")})),
        ("swatches.import", json!({"data": b64_encode(&[0, 1, 0xff, 0xff])})),
        ("swatches.import", json!({"data": b64_encode(&[0, 1, 0, 0])})),
        ("swatches.import", json!({"data": b64_encode(&[0, 1, 0, 0]), "format": "gpl"})),
        ("swatches.import", json!({"data": b64_encode(&[0, 1, 0, 0]), "mode": "merge"})),
        ("swatches.export", json!({"format": "pdf"})),
        ("swatches.export", json!({"group": "Nope"})),
        ("swatches.export", json!({"swatches": ["Nope"]})),
        ("swatches.export", json!({"swatches": "Red"})),
    ];
    for (id, p) in cases {
        assert!(s.execute(id, p.clone()).is_err(), "{id} {p}");
    }
    // An unwritable path (its "folder" is a file) is an error.
    let file = std::env::temp_dir().join(format!("pc-swatch-file-{}", std::process::id()));
    std::fs::write(&file, b"x").unwrap();
    assert!(s.execute("swatches.export", json!({"path": file.join("x.aco").to_string_lossy()})).is_err());
    std::fs::remove_file(&file).ok();
    assert_eq!(s.presets.swatches, before, "failed commands change nothing");
    // Empty library: the swatch commands are disabled, not crashing.
    s.execute("swatches.delete", json!({"swatch": names(&s)})).unwrap();
    for g in ["Grays", "RGB", "Pastel", "Pure", "Dark"] {
        s.execute("swatches.deleteGroup", json!({"group": g})).unwrap();
    }
    for id in ["swatches.delete", "swatches.rename", "swatches.move", "swatches.use", "swatches.export", "swatches.renameGroup", "swatches.deleteGroup"] {
        assert!(s.execute(id, json!({"swatch": "Red", "group": "Grays", "name": "x"})).is_err(), "{id}");
    }
    // Adding to an empty library creates a group.
    let r = s.execute("swatches.add", json!({})).unwrap();
    assert_eq!(r["group"], "Swatches");
}

#[test]
fn library_limits() {
    let mut s = Session::new();
    let one = vec![AcoSwatch { name: "x".into(), color: AcoColor::Gray(0) }; 60_000];
    let file = b64_encode(&photocraft_psd::aco::write(&one).unwrap());
    s.execute("swatches.import", json!({"data": file, "mode": "replace"})).unwrap();
    assert!(s.execute("swatches.import", json!({"data": file})).is_err(), "over the library limit");
    assert_eq!(names(&s).len(), 60_000);
    // A hand-edited preferences file is trimmed, not trusted.
    let mut g = vec![Group::new(&"n".repeat(1000), vec![Swatch::new("a", SwatchColor::Gray(0.0)); MAX_SWATCHES + 5])];
    cap_library(&mut g);
    assert_eq!(total(&g), MAX_SWATCHES);
    assert_eq!(g[0].name.chars().count(), MAX_NAME);
}

/// Groups without a name take the import's group name. The import used to number each repeated
/// group name by searching the library once for every number it tried, so importing about 10,000
/// unnamed groups took minutes. The test fails when the import takes more than 20 seconds.
#[test]
fn unnamed_groups_are_numbered_in_linear_time() {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut s = Session::new();
        s.execute("swatches.newGroup", json!({"name": "Ink 2"})).unwrap();
        let n = MAX_GROUPS - s.presets.swatches.len();
        let paper = AseSwatch { name: "Paper".into(), color: AseColor::Gray(1.0), kind: AseKind::Process };
        let mut entries = vec![AseEntry::Group { name: String::new(), swatches: Vec::new() }; n];
        entries[0] = AseEntry::Group { name: String::new(), swatches: vec![paper] };
        let file = b64_encode(&photocraft_psd::ase::write(&AseFile { entries }).unwrap());
        let r = s.execute("swatches.import", json!({"data": file, "group": "Ink"})).unwrap();
        assert_eq!(s.presets.swatches.len(), MAX_GROUPS);
        let names: Vec<String> = r["groups"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
        let _ = tx.send(names);
    });
    let names = match rx.recv_timeout(std::time::Duration::from_secs(20)) {
        Ok(names) => names,
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => panic!("importing the groups did not finish"),
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => panic!("the import thread panicked"),
    };
    // Numbering starts at 1 and skips the number the library already uses.
    assert_eq!(names[..4], ["Ink", "Ink 1", "Ink 3", "Ink 4"]);
    assert_eq!(names.last(), Some(&format!("Ink {}", names.len())));
    assert_eq!(names.iter().collect::<std::collections::HashSet<_>>().len(), names.len());
}

#[test]
fn repeated_group_names_take_the_first_free_number() {
    let mut s = Session::new();
    for name in ["Brand", "Brand 2"] {
        s.execute("swatches.newGroup", json!({"name": name})).unwrap();
    }
    let paper = AseSwatch { name: "Paper".into(), color: AseColor::Gray(1.0), kind: AseKind::Process };
    let entries = ["Brand", "Brand", "Brand", "Brand 1"].into_iter().map(|name| AseEntry::Group { name: name.into(), swatches: vec![paper.clone()] }).collect();
    let file = b64_encode(&photocraft_psd::ase::write(&AseFile { entries }).unwrap());
    let r = s.execute("swatches.import", json!({"data": file})).unwrap();
    // The file's "Brand 1" comes after the first "Brand" took that name.
    assert_eq!(r["groups"], json!(["Brand 1", "Brand 3", "Brand 4", "Brand 1 1"]));
}

/// A value beyond f32 range used to pass the finite check, become inf as f32 and wrap to a NaN
/// hue, which the prefs file stores as `null`; the next launch then dropped every saved preset.
#[test]
fn values_beyond_f32_range_are_rejected_so_presets_survive_a_restart() {
    let mut s = Session::new();
    s.execute("swatches.add", json!({"name": "Keep me", "color": {"model": "lab", "values": [40, 12.5, -7.25]}})).unwrap();
    let before = s.presets.swatches.clone();
    for (model, values) in [
        ("hsb", json!([1e39, 100, 100])),
        ("hsb", json!([-1e39, 100, 100])),
        ("rgb", json!([1e39, 0, 0])),
        ("cmyk", json!([0, 0, 0, -1e300])),
        ("lab", json!([50, 1e39, 0])),
        ("gray", json!([1e39])),
    ] {
        let r = s.execute("swatches.add", json!({"name": "Bad", "color": {"model": model, "values": values}}));
        assert!(matches!(r, Err(crate::EngineError::BadParams { .. })), "{model} {values}: {r:?}");
    }
    assert_eq!(s.presets.swatches, before, "a rejected colour leaves the library unchanged");
    // The largest f32 hue is still accepted and stays finite.
    s.execute("swatches.add", json!({"name": "Big hue", "color": {"model": "hsb", "values": [f64::from(f32::MAX), 100, 100]}})).unwrap();
    let mut t = Session::new();
    t.load_prefs_json(&s.prefs_to_json()).unwrap();
    assert_eq!(t.presets.swatches, s.presets.swatches, "user swatches survive a save and reload");
    assert!(names(&t).contains(&"Keep me".to_string()));
}
