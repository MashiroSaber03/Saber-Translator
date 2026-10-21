use super::*;

fn session() -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 32, "height": 32})).unwrap();
    s
}

#[test]
fn defaults_match_photoshop() {
    let p = Preferences::default();
    assert_eq!(p.performance.history_states, 50);
    assert_eq!(p.guides_grid_and_slices.subdivisions, 4);
    assert_eq!(p.guides_grid_and_slices.guide_color, "#4affff");
    assert_eq!(p.transparency_and_gamut.colors(), [[255, 255, 255], [204, 204, 204]]);
    assert_eq!(p.transparency_and_gamut.square(), Some(8.0));
    assert_eq!(p.cursors.painting, PaintingCursor::NormalTip);
    // Every dialog section is a key of the JSON form.
    let v = p.to_json();
    for (id, _) in SECTIONS {
        assert!(v.get(id).is_some_and(Value::is_object), "{id}");
    }
}

/// #2264: the Studio workspace dot pattern can be hidden without changing themes.
#[test]
fn canvas_dot_grid_can_be_disabled_persisted_and_reset() {
    let mut s = Session::new();
    assert!(s.prefs().interface.show_canvas_dots, "existing Studio appearance stays unchanged by default");
    assert_eq!(s.execute("prefs.get", json!({"path": "interface.showCanvasDots"})).unwrap(), json!(true));

    s.execute("prefs.set", json!({"path": "interface.showCanvasDots", "value": false})).unwrap();
    assert!(!s.prefs().interface.show_canvas_dots);
    assert_eq!(s.execute("prefs.get", json!({"path": "interface.showCanvasDots"})).unwrap(), json!(false));

    let mut restored = Session::new();
    restored.load_prefs_json(&s.prefs_to_json()).unwrap();
    assert!(!restored.prefs().interface.show_canvas_dots, "disabling the pattern must survive restart");

    // Older preference files without the new key retain the original dotted workspace.
    let mut legacy = Session::new();
    legacy.load_prefs_json(r#"{"interface": {"theme": "studio"}}"#).unwrap();
    assert!(legacy.prefs().interface.show_canvas_dots);

    assert!(s.execute("prefs.set", json!({"path": "interface.showCanvasDots", "value": "no"})).is_err());
    assert!(!s.prefs().interface.show_canvas_dots, "an invalid type must not change the preference");

    s.execute("prefs.reset", json!({"path": "interface.showCanvasDots"})).unwrap();
    assert!(s.prefs().interface.show_canvas_dots);
}

#[test]
fn appearance_defaults_and_legacy_theme_migrate() {
    let p = Preferences::default();
    // New users keep Photoshop's default; following the system is opt-in.
    assert_eq!(p.interface.appearance_mode, AppearanceMode::Dark);
    assert_eq!(p.interface.dark_theme, DarkTheme::ProMedium);
    assert_eq!(p.interface.light_theme, LightTheme::ProLight);
    for (theme, mode, dark, light) in
        [("pro", AppearanceMode::Dark, DarkTheme::Pro, LightTheme::ProLight), ("classic", AppearanceMode::Light, DarkTheme::ProMedium, LightTheme::Classic)]
    {
        let mut s = Session::new();
        s.load_prefs_json(&json!({"interface": {"theme": theme}}).to_string()).unwrap();
        assert_eq!(s.prefs().interface.appearance_mode, mode);
        assert_eq!(s.prefs().interface.dark_theme, dark);
        assert_eq!(s.prefs().interface.light_theme, light);
    }
}

#[test]
fn legacy_theme_migration_preserves_saved_theme_slots() {
    for (legacy, slot, saved) in [("proMedium", "darkTheme", "solarizedDark"), ("studioLight", "lightTheme", "adwaita")] {
        let mut interface = json!({"theme": legacy});
        interface[slot] = json!(saved);
        let mut session = Session::new();
        session.load_prefs_json(&json!({"interface": interface}).to_string()).unwrap();
        assert_eq!(session.prefs_value()["interface"][slot], saved, "explicit {slot} survives migration");
    }
}

#[test]
fn appearance_choices_validate_and_legacy_theme_still_selects() {
    let mut s = Session::new();
    s.execute("prefs.set", json!({"values": {"interface.appearanceMode": "auto", "interface.darkTheme": "studio", "interface.lightTheme": "classic"}}))
        .unwrap();
    assert_eq!(s.prefs().interface.appearance_mode, AppearanceMode::Auto);
    assert_eq!(s.prefs().interface.dark_theme, DarkTheme::Studio);
    assert_eq!(s.prefs().interface.light_theme, LightTheme::Classic);
    let mut restarted = Session::new();
    restarted.load_prefs_json(&s.prefs_to_json()).unwrap();
    assert_eq!(restarted.prefs().interface, s.prefs().interface);
    assert!(s.execute("prefs.set", json!({"path": "interface.darkTheme", "value": "classic"})).is_err());
    s.execute("prefs.set", json!({"path": "interface.theme", "value": "pro"})).unwrap();
    assert_eq!(s.prefs().interface.appearance_mode, AppearanceMode::Dark);
    assert_eq!(s.prefs().interface.dark_theme, DarkTheme::Pro);
    assert_eq!(s.prefs().interface.light_theme, LightTheme::Classic);
    s.execute("prefs.set", json!({"path": "interface", "value": {"theme": "studioLight"}})).unwrap();
    assert_eq!(s.prefs().interface.appearance_mode, AppearanceMode::Light);
    assert_eq!(s.prefs().interface.light_theme, LightTheme::StudioLight);
    assert_eq!(s.prefs().interface.dark_theme, DarkTheme::Pro);
    // #2661: the light Pro brightness levels are light themes.
    for (theme, light) in [("proLightGray", LightTheme::ProLightGray), ("proLight", LightTheme::ProLight)] {
        s.execute("prefs.set", json!({"path": "interface.theme", "value": theme})).unwrap();
        assert_eq!(s.prefs().interface.appearance_mode, AppearanceMode::Light);
        assert_eq!(s.prefs().interface.light_theme, light);
        s.execute("prefs.set", json!({"path": "interface.lightTheme", "value": theme})).unwrap();
        let mut migrated = Session::new();
        migrated.load_prefs_json(&json!({"interface": {"theme": theme}}).to_string()).unwrap();
        assert_eq!(migrated.prefs().interface.appearance_mode, AppearanceMode::Light);
        assert_eq!(migrated.prefs().interface.light_theme, light);
    }
}

#[test]
fn favorite_themes_persist_and_reject_invalid_lists() {
    let mut s = Session::new();
    assert!(s.prefs().interface.favorite_themes.is_empty());
    s.execute("prefs.set", json!({"path": "interface.favoriteThemes", "value": ["classic", "studio"]})).unwrap();
    assert_eq!(s.prefs().interface.favorite_themes, vec![Theme::Classic, Theme::Studio]);
    s.execute("prefs.set", json!({"path": "interface.theme", "value": "pro"})).unwrap();
    let mut restored = Session::new();
    restored.load_prefs_json(&s.prefs_to_json()).unwrap();
    assert_eq!(restored.prefs().interface.favorite_themes, vec![Theme::Classic, Theme::Studio]);
    for invalid in [json!("classic"), json!(["unknown"]), json!(["classic", "classic"]), json!([23])] {
        assert!(s.execute("prefs.set", json!({"path": "interface.favoriteThemes", "value": invalid})).is_err());
        assert_eq!(s.prefs().interface.favorite_themes, vec![Theme::Classic, Theme::Studio]);
    }
}

#[test]
fn get_set_reset_by_path() {
    let mut s = session();
    assert_eq!(s.execute("prefs.get", json!({"path": "performance.historyStates"})).unwrap(), json!(50));
    s.execute("prefs.set", json!({"path": "performance.historyStates", "value": 3})).unwrap();
    assert_eq!(s.prefs().performance.history_states, 3);
    // Applied to open documents: only three undo steps remain.
    for i in 0..6 {
        s.execute("layer.new.layer", json!({"name": format!("L{i}")})).unwrap();
    }
    let mut n = 0;
    while s.undo() {
        n += 1;
    }
    assert_eq!(n, 3);
    // New documents too.
    s.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
    assert_eq!(s.active().unwrap().history.max_states, 3);
    s.execute("prefs.reset", json!({"path": "performance"})).unwrap();
    assert_eq!(s.prefs().performance.history_states, 50);
    assert_eq!(s.active().unwrap().history.max_states, 50);
}

#[test]
fn reset_one_colour_setting() {
    let mut s = session();
    s.execute("prefs.set", json!({"values": {"colorSettings.bpc": false, "colorSettings.workingRgb": "display-p3"}})).unwrap();
    assert_eq!(s.execute("prefs.reset", json!({"path": "colorSettings.bpc"})).unwrap(), json!(true));
    assert!(s.color.settings.bpc);
    assert_eq!(s.color.settings.working_rgb, "display-p3", "other colour settings are untouched");
    s.execute("prefs.reset", json!({"path": "colorSettings.workingRgb"})).unwrap();
    assert_eq!(s.color.settings, crate::color_cmds::ColorSettings::default());
    assert!(s.execute("prefs.reset", json!({"path": "colorSettings.notAField"})).is_err());
}

#[test]
fn reset_one_keyed_override() {
    let mut s = session();
    s.execute(
        "prefs.set",
        json!({"values": {
            "shortcuts.edit.undo": "Ctrl+Alt+Z", "shortcuts.edit.redo": "Ctrl+Alt+Y",
            "menus.colors.edit.fill": "red", "menus.colors.filter.blur.gaussianBlur": "blue",
            "interface.theme": "studioLight"
        }}),
    )
    .unwrap();
    let undo = crate::commands::find("edit.undo").unwrap().shortcut;
    assert_eq!(s.prefs().shortcut("edit.undo", undo), Some("Ctrl+Alt+Z"));
    for path in ["shortcuts.edit.undo", "menus.colors.filter.blur.gaussianBlur"] {
        let revision = s.prefs.rev();
        assert_eq!(s.execute("prefs.reset", json!({"path": path})).unwrap(), Value::Null);
        assert!(s.prefs.rev() > revision);
        assert!(s.execute("prefs.get", json!({"path": path})).is_err(), "removed override: {path}");
        let mut restored = Session::new();
        restored.load_prefs_json(&s.prefs_to_json()).unwrap();
        assert_eq!(restored.prefs(), s.prefs(), "removal survives saving: {path}");
    }
    assert_eq!(s.execute("prefs.get", json!({"path": "shortcuts"})).unwrap(), json!({"edit.redo": "Ctrl+Alt+Y"}));
    assert_eq!(s.execute("prefs.get", json!({"path": "menus.colors"})).unwrap(), json!({"edit.fill": "red"}));
    assert_eq!(s.execute("prefs.get", json!({"path": "interface.theme"})).unwrap(), json!("studioLight"));
    assert_eq!(s.prefs().shortcut("edit.undo", undo), undo);
}

#[test]
fn reset_keyed_overrides_is_idempotent_and_restores_disabled_shortcuts() {
    let mut s = session();
    s.execute("prefs.set", json!({"path": "shortcuts.edit.undo", "value": ""})).unwrap();
    let undo = crate::commands::find("edit.undo").unwrap().shortcut;
    assert_eq!(s.prefs().shortcut("edit.undo", undo), None);
    for path in ["shortcuts.edit.undo", "menus.colors.edit.fill"] {
        for _ in 0..2 {
            assert_eq!(s.execute("prefs.reset", json!({"section": path})).unwrap(), Value::Null);
        }
    }
    assert_eq!(s.prefs().shortcut("edit.undo", undo), undo);
    assert_eq!(s.prefs(), &Preferences::default());
}

#[test]
fn preferences_reset_removes_keyed_overrides() {
    let mut p = Preferences::default();
    for (path, value) in [("shortcuts.edit.undo", json!("Ctrl+Z")), ("menus.colors.edit.fill", json!("red"))] {
        p.set(path, value).unwrap();
        assert!(p.get(path).is_some());
        p.reset(Some(path)).unwrap();
        assert!(p.get(path).is_none());
        p.reset(Some(path)).unwrap();
    }
    assert_eq!(p, Preferences::default());
}

#[test]
fn keyed_reset_keeps_section_resets_and_unknown_path_errors() {
    let mut s = session();
    s.execute("prefs.set", json!({"values": {"shortcuts.edit.undo": "Ctrl+Z", "menus.colors.edit.fill": "red"}})).unwrap();
    let saved = s.prefs_value();
    let revision = s.prefs.rev();
    for path in ["shortcut.edit.undo", "menus.color.edit.fill", "menus.notAField", "general.notAField"] {
        assert!(matches!(s.execute("prefs.reset", json!({"path": path})), Err(EngineError::BadParams { .. })));
        assert_eq!(s.prefs_value(), saved, "rejected reset changes nothing: {path}");
        assert_eq!(s.prefs.rev(), revision);
    }
    assert_eq!(s.execute("prefs.reset", json!({"path": "shortcuts"})).unwrap(), json!({}));
    assert_eq!(s.execute("prefs.get", json!({"path": "menus.colors"})).unwrap(), json!({"edit.fill": "red"}));
    assert_eq!(s.execute("prefs.reset", json!({"path": "menus.colors"})).unwrap(), json!({}));
    s.execute("prefs.set", json!({"path": "shortcuts.edit.undo", "value": "Ctrl+Z"})).unwrap();
    let all = s.execute("prefs.reset", json!({})).unwrap();
    assert_eq!(all["shortcuts"], json!({}));
    assert_eq!(all["menus"]["colors"], json!({}));
    assert_eq!(s.prefs(), &Preferences::default());
}

#[test]
fn set_validates_and_is_all_or_nothing() {
    let mut s = session();
    for (path, value) in [
        ("cursors.painting", json!("huge")),
        ("performance.historyStates", json!(0)),
        ("performance.historyStates", json!("ten")),
        ("guidesGridAndSlices.guideColor", json!("cyan")),
        ("nope.nothing", json!(1)),
        ("general.notAField", json!(true)),
        ("shortcuts.edit.undo", json!("Cmd+Banana+Q")),
    ] {
        assert!(s.execute("prefs.set", json!({"path": path, "value": value})).is_err(), "{path}");
    }
    let r = s.execute("prefs.set", json!({"values": {"cursors.painting": "precise", "performance.historyStates": -4}}));
    assert!(r.is_err());
    assert_eq!(s.prefs().cursors.painting, PaintingCursor::NormalTip, "nothing applied");
    let r = s.execute("prefs.set", json!({"values": {"cursors.painting": "precise", "unitsAndRulers.rulers": "cm"}})).unwrap();
    assert_eq!(r["cursors.painting"], "precise");
    assert_eq!(s.prefs().units_and_rulers.rulers, Unit::Centimeters);
}

#[test]
fn whole_section_set_merges_keys() {
    let mut s = session();
    s.execute("prefs.set", json!({"path": "transparencyAndGamut", "value": {"gridSize": "large", "gridColors": "custom", "customDark": "#336699"}})).unwrap();
    let t = &s.prefs().transparency_and_gamut;
    assert_eq!(t.square(), Some(16.0));
    assert_eq!(t.colors()[1], [0x33, 0x66, 0x99]);
    assert_eq!(t.gamut_warning_color, "#808080", "untouched keys keep their values");
}

#[test]
fn json_round_trip_tolerates_unknown_and_missing_keys() {
    let mut s = session();
    s.execute("prefs.set", json!({"values": {"interface.theme": "studioLight", "fileHandling.autosaveMinutes": 5, "shortcuts.edit.fill": "Cmd+Shift+F"}}))
        .unwrap();
    s.execute("edit.colorSettings", json!({"workingRgb": "display-p3", "policyRgb": "convert"})).unwrap();
    let text = s.prefs_to_json();
    let mut t = Session::new();
    t.load_prefs_json(&text).unwrap();
    assert_eq!(t.prefs(), s.prefs());
    assert_eq!(t.color.settings.working_rgb, "display-p3");
    assert_eq!(t.color.settings.policy_rgb, crate::color_cmds::Policy::Convert);
    // Older/newer files: unknown keys ignored, missing ones default.
    let mut u = Session::new();
    u.load_prefs_json(r#"{"general": {"beepWhenDone": true, "futureThing": 1}, "someNewSection": {}}"#).unwrap();
    assert!(u.prefs().general.beep_when_done);
    assert_eq!(u.prefs().performance.history_states, 50);
    assert!(u.load_prefs_json("not json").is_err());
}

#[test]
fn low_resolution_previews_default_on_and_switch_off() {
    // A preferences file from before the setting keeps the fast previews.
    let mut s = Session::new();
    s.load_prefs_json(r#"{"performance": {"historyStates": 20}}"#).unwrap();
    assert!(s.prefs().performance.low_resolution_previews);
    s.execute("prefs.set", json!({"path": "performance.lowResolutionPreviews", "value": false})).unwrap();
    assert!(!s.prefs().performance.low_resolution_previews);
    let mut t = Session::new();
    t.load_prefs_json(&s.prefs_to_json()).unwrap();
    assert!(!t.prefs().performance.low_resolution_previews);
}

#[test]
fn units_convert_both_ways() {
    for u in [Unit::Pixels, Unit::Inches, Unit::Centimeters, Unit::Millimeters, Unit::Points, Unit::Picas, Unit::Percent] {
        let v = u.from_px(450.0, 300.0, 900.0, 72.0);
        assert!((u.to_px(v, 300.0, 900.0, 72.0) - 450.0).abs() < 1e-9, "{u:?}");
    }
    assert_eq!(Unit::Inches.from_px(600.0, 300.0, 0.0, 72.0), 2.0);
    assert_eq!(Unit::Points.from_px(300.0, 300.0, 0.0, 72.0), 72.0);
    assert_eq!(Unit::Percent.from_px(450.0, 300.0, 900.0, 72.0), 50.0);
    let r = UnitsAndRulers { rulers: Unit::Centimeters, ..Default::default() };
    assert_eq!(r.format(300.0, 300.0, 0.0), "2.54");
}

#[test]
fn shortcuts_normalise_and_detect_conflicts() {
    assert_eq!(normalize_shortcut("shift+cmd+n").as_deref(), Some("Cmd+Shift+N"));
    assert_eq!(normalize_shortcut("Alt+Cmd+f7").as_deref(), Some("Cmd+Alt+F7"));
    assert_eq!(normalize_shortcut("Cmd++").as_deref(), Some("Cmd++"));
    assert_eq!(normalize_shortcut("Cmd+Shift"), None);
    assert_eq!(normalize_shortcut("Cmd+A+B"), None);
    let c = conflicts([("a", "Cmd+J"), ("b", "cmd+j"), ("c", "Cmd+K")]);
    assert_eq!(c, vec![("Cmd+J".to_string(), vec!["a".to_string(), "b".to_string()])]);
}

#[test]
fn keyboard_shortcuts_reassign_and_reset() {
    let mut s = session();
    let undo = crate::commands::find("edit.undo").unwrap().shortcut;
    assert_eq!(s.prefs().shortcut("edit.undo", undo), Some("Cmd+Z"));
    // Give ⌘Z to Fill: Undo loses it (Photoshop moves a shortcut rather than duplicating it).
    let r = s.execute("edit.keyboardShortcuts", json!({"set": {"edit.fill": "Cmd+Z"}})).unwrap();
    assert_eq!(s.prefs().shortcut("edit.fill", None), Some("Cmd+Z"));
    assert_eq!(s.prefs().shortcut("edit.undo", undo), None);
    assert!(r["conflicts"].as_array().unwrap().is_empty());
    // Keeping the duplicate reports a conflict.
    let r = s.execute("edit.keyboardShortcuts", json!({"set": {"edit.undo": "Cmd+Z"}, "removeConflicts": false})).unwrap();
    assert_eq!(r["conflicts"][0]["shortcut"], "Cmd+Z");
    assert!(s.execute("edit.keyboardShortcuts", json!({"set": {"no.such.command": "F9"}})).is_err());
    s.execute("edit.keyboardShortcuts", json!({"reset": true})).unwrap();
    assert!(s.prefs().shortcuts.is_empty());
    assert_eq!(s.prefs().shortcut("edit.undo", undo), Some("Cmd+Z"));
    // Listing with a filter.
    let r = s.execute("edit.keyboardShortcuts", json!({"filter": "gaussian"})).unwrap();
    assert!(r["commands"].as_array().unwrap().iter().any(|c| c["id"] == "filter.blur.gaussianBlur"));
}

/// #249: temporary tools (held keys) and the fill / colour keys are bindable like commands.
#[test]
fn temporary_tools_and_fill_keys_are_rebindable() {
    let mut s = session();
    let r = s.execute("edit.keyboardShortcuts", json!({"filter": "tools.temporary"})).unwrap();
    let ids: Vec<&str> = r["commands"].as_array().unwrap().iter().filter_map(|c| c["id"].as_str()).collect();
    assert!(
        ids.contains(&"tools.temporary.hand")
            && ids.contains(&"tools.temporary.zoomIn")
            && ids.contains(&"tools.temporary.zoomOut")
            && ids.contains(&"tools.temporary.erase"),
        "{ids:?}"
    );
    assert!(r["commands"].as_array().unwrap().iter().all(|c| c["hold"] == true));
    // Rebind the temporary zoom; an unknown id still fails.
    s.execute("edit.keyboardShortcuts", json!({"set": {"tools.temporary.zoomIn": "Ctrl+Space"}})).unwrap();
    assert_eq!(s.prefs().shortcut("tools.temporary.zoomIn", Some("Cmd+Space")), Some("Ctrl+Space"));
    assert!(s.execute("edit.keyboardShortcuts", json!({"set": {"tools.temporary.unknown": "Space"}})).is_err());
    // Taking a temporary tool's key for a command removes it from the temporary tool.
    s.execute("edit.keyboardShortcuts", json!({"set": {"edit.fillForeground": "Space"}})).unwrap();
    assert_eq!(s.prefs().shortcut("tools.temporary.hand", Some("Space")), None);
    assert_eq!(s.prefs().shortcut("edit.fillForeground", Some("Alt+Backspace")), Some("Space"));
    // D / X and the fill keys are listed with their Photoshop defaults.
    let r = s.execute("edit.keyboardShortcuts", json!({"list": true})).unwrap();
    let def = |id: &str| r["commands"].as_array().unwrap().iter().find(|c| c["id"] == id).map(|c| c["default"].clone());
    assert_eq!(def("tools.swapColors"), Some(json!("X")));
    assert_eq!(def("tools.defaultColors"), Some(json!("D")));
    assert_eq!(def("edit.fillBackground"), Some(json!("Cmd+Backspace")));
    // Bad values fail.
    assert!(s.execute("edit.keyboardShortcuts", json!({"set": {"tools.temporary.hand": 7}})).is_err());
}

#[test]
fn menus_and_toolbar_customisation_persist() {
    let mut s = session();
    s.execute("edit.menus", json!({"hide": ["edit.fade"], "color": {"edit.fill": "red"}})).unwrap();
    assert_eq!(s.prefs().menus.hidden, vec!["edit.fade".to_string()]);
    assert_eq!(s.prefs().menus.colors["edit.fill"], "red");
    assert!(s.execute("edit.menus", json!({"color": {"edit.fill": "plaid"}})).is_err());
    s.execute("edit.menus", json!({"show": ["edit.fade"], "color": {"edit.fill": "none"}})).unwrap();
    assert!(s.prefs().menus.hidden.is_empty() && s.prefs().menus.colors.is_empty());
    s.execute("edit.toolbar", json!({"hidden": ["Sponge"]})).unwrap();
    let back: Preferences = serde_json::from_str(&s.prefs_to_json()).unwrap();
    assert_eq!(back.toolbar.hidden, vec!["Sponge".to_string()]);
}

#[test]
fn preference_sections_are_commands() {
    let mut s = Session::new();
    for (id, title) in SECTIONS {
        let r = s.execute(&format!("edit.preferences.{id}"), json!({})).unwrap();
        assert_eq!(r["section"], id);
        assert_eq!(r["title"], title);
        assert!(r["values"].is_object(), "{id}");
    }
}

#[test]
fn choice_and_range_tables_cover_enum_fields() {
    let p = Preferences::default().to_json();
    for (id, _) in SECTIONS {
        for (k, v) in p[id].as_object().unwrap() {
            let path = format!("{id}.{k}");
            if let Some(c) = choices(&path) {
                if path == "performance.renderingMode" && v.is_null() {
                    assert_eq!(p["performance"]["renderingMode"], json!(null), "legacy mode is inferred until explicitly selected");
                } else {
                    assert!(c.contains(&v.as_str().unwrap()), "{path}");
                }
            }
            if let Some((lo, hi)) = range(&path) {
                let x = v.as_f64().unwrap();
                assert!(x >= lo && x <= hi, "{path} default {x} outside {lo}..{hi}");
            }
            if is_color(&path) {
                assert!(parse_hex(v.as_str().unwrap()).is_some(), "{path}");
            }
        }
    }
}

#[test]
fn svg_group_rasterization_threshold_is_bounded_by_document_depth() {
    let mut s = session();
    let pref = "fileHandling.rasterizeSvgGroupsDeeperThan";
    assert_eq!(s.execute("prefs.get", json!({"path": pref})).unwrap(), json!(photocraft_doc::MAX_GROUP_DEPTH));
    s.execute("prefs.set", json!({"path": pref, "value": 4})).unwrap();
    assert_eq!(s.prefs().file_handling.rasterize_svg_groups_deeper_than, 4);
    s.execute("prefs.set", json!({"path": pref, "value": 0})).unwrap();
    assert_eq!(s.prefs().file_handling.rasterize_svg_groups_deeper_than, 0, "zero disables the preference threshold");
    assert!(s.execute("prefs.set", json!({"path": pref, "value": photocraft_doc::MAX_GROUP_DEPTH + 1})).is_err());
    assert_eq!(s.prefs().file_handling.rasterize_svg_groups_deeper_than, 0, "out-of-range preferences leave the valid value unchanged");
}

#[test]
fn right_click_with_painting_tools_pref() {
    let mut s = session();
    assert_eq!(s.prefs().tools.right_click_with_painting_tools, RightClickPaint::BrushPicker, "Photoshop: the Brush Preset picker");
    assert_eq!(s.execute("prefs.get", json!({"path": "tools.rightClickWithPaintingTools"})).unwrap(), json!("brushPicker"));
    assert_eq!(choices("tools.rightClickWithPaintingTools"), Some(RightClickPaint::NAMES));
    s.execute("prefs.set", json!({"path": "tools.rightClickWithPaintingTools", "value": "erase"})).unwrap();
    assert_eq!(s.prefs().tools.right_click_with_painting_tools, RightClickPaint::Erase);
    for bad in [json!("smudge"), json!(1), json!(null), json!(["erase"])] {
        assert!(s.execute("prefs.set", json!({"path": "tools.rightClickWithPaintingTools", "value": bad})).is_err());
    }
    assert_eq!(s.prefs().tools.right_click_with_painting_tools, RightClickPaint::Erase, "rejected values change nothing");
    // Preferences saved before the option existed load with the default.
    let old: Tools = serde_json::from_value(json!({"showTooltips": false})).unwrap();
    assert_eq!(old.right_click_with_painting_tools, RightClickPaint::BrushPicker);
}

#[test]
fn gpu_backend_round_trips_and_validates() {
    let mut s = session();
    assert_eq!(s.prefs().performance.gpu_backend, GpuBackend::Auto);
    assert_eq!(s.execute("prefs.get", json!({"path": "performance.gpuBackend"})).unwrap(), json!("auto"));
    s.execute("prefs.set", json!({"path": "performance.gpuBackend", "value": "dx12"})).unwrap();
    assert_eq!(s.prefs().performance.gpu_backend, GpuBackend::Dx12);
    // Unknown names and wrong types are errors, and leave the value alone.
    assert!(s.execute("prefs.set", json!({"path": "performance.gpuBackend", "value": "directx"})).is_err());
    assert!(s.execute("prefs.set", json!({"path": "performance.gpuBackend", "value": 3})).is_err());
    assert_eq!(s.prefs().performance.gpu_backend, GpuBackend::Dx12);
    // Persisted and restored with the rest of the preferences.
    let text = s.prefs_to_json();
    let mut t = Session::new();
    t.load_prefs_json(&text).unwrap();
    assert_eq!(t.prefs().performance.gpu_backend, GpuBackend::Dx12);
    // Files from before the setting load as `auto`.
    let mut u = Session::new();
    u.load_prefs_json(r#"{"performance": {"useGpu": true}}"#).unwrap();
    assert_eq!(u.prefs().performance.gpu_backend, GpuBackend::Auto);
    assert_eq!(choices("performance.gpuBackend"), Some(GpuBackend::NAMES));
    for n in GpuBackend::NAMES {
        assert_eq!(GpuBackend::parse(n).map(GpuBackend::name), Some(*n));
    }
}

/// #2022: notices auto-hide after a user-set delay by default; both settings round-trip.
#[test]
fn notification_autohide_preferences() {
    let p = Preferences::default();
    assert!(p.interface.notification_auto_hide);
    assert_eq!(p.interface.notification_duration_seconds, 6);
    assert_eq!(range("interface.notificationDurationSeconds"), Some((1.0, 120.0)));

    let mut s = Session::new();
    s.execute("prefs.set", json!({"path": "interface.notificationAutoHide", "value": false})).unwrap();
    s.execute("prefs.set", json!({"path": "interface.notificationDurationSeconds", "value": 30})).unwrap();
    assert!(!s.prefs().interface.notification_auto_hide);
    assert_eq!(s.prefs().interface.notification_duration_seconds, 30);
    // Out-of-range and wrong-typed values are rejected and change nothing.
    for bad in [json!(0), json!(121), json!("soon")] {
        assert!(s.execute("prefs.set", json!({"path": "interface.notificationDurationSeconds", "value": bad})).is_err());
    }
    assert_eq!(s.prefs().interface.notification_duration_seconds, 30);

    let mut t = Session::new();
    t.load_prefs_json(&s.prefs_to_json()).unwrap();
    assert!(!t.prefs().interface.notification_auto_hide);
    assert_eq!(t.prefs().interface.notification_duration_seconds, 30);
    // Older files without the settings keep the defaults.
    let mut u = Session::new();
    u.load_prefs_json(r#"{"interface": {"theme": "studio"}}"#).unwrap();
    assert!(u.prefs().interface.notification_auto_hide);
    assert_eq!(u.prefs().interface.notification_duration_seconds, 6);
}

#[test]
fn linux_only_preferences_show_only_on_linux() {
    assert_eq!(is_hidden("performance.linuxDisplayServer"), !cfg!(target_os = "linux"));
    assert!(!is_hidden("performance.gpuBackend"));
    assert!(LINUX_ONLY.iter().all(|p| choices(p).is_some()), "every Linux-only preference is a real one");
}

#[test]
fn the_default_pressure_curve_is_linear() {
    let c = PressureCurve::new(&Preferences::default().tools.pressure_curve);
    assert!(c.is_linear());
    for x in [0.0, 0.1, 0.37, 0.5, 0.99, 1.0] {
        assert!((c.eval(x) - x).abs() < 1e-6, "{x}");
    }
}

#[test]
fn a_pressure_curve_is_monotone_and_passes_through_its_points() {
    // A soft curve: light pressure already gives a lot.
    let pts = [[0.0, 0.0], [0.25, 0.5], [0.6, 0.8], [1.0, 1.0]];
    let c = PressureCurve::new(&pts);
    for p in pts {
        assert!((c.eval(p[0]) - p[1]).abs() < 1e-5, "{p:?}");
    }
    let ys: Vec<f32> = (0..=200).map(|i| c.eval(i as f32 / 200.0)).collect();
    assert!(ys.windows(2).all(|w| w[1] >= w[0] - 1e-6), "firmer never gives less: {ys:?}");
    assert!(ys.iter().all(|y| (0.0..=1.0).contains(y)));
    // Flat outside the points: a floor and a ceiling.
    let c = PressureCurve::new(&[[0.2, 0.3], [0.8, 0.9]]);
    assert_eq!((c.eval(0.0), c.eval(0.1), c.eval(0.95), c.eval(1.0)), (0.3, 0.3, 0.9, 0.9));
}

#[test]
fn a_hostile_pressure_curve_still_evaluates_monotone_within_range() {
    let many: Vec<[f32; 2]> = (0..100).map(|i| [((i * 37) % 100) as f32 / 99.0, ((i * 53) % 100) as f32 / 99.0]).collect();
    for pts in [
        vec![],
        vec![[0.5, 0.5]],
        vec![[f32::NAN, 0.5], [f32::INFINITY, 1.0], [0.3, f32::NEG_INFINITY]],
        vec![[2.0, -1.0], [-3.0, 5.0], [0.5, 0.2]],
        vec![[0.9, 0.1], [0.1, 0.9]],
        vec![[0.5, 0.2], [0.5, 0.7], [1.0, 1.0], [0.0, 0.0]],
        many,
    ] {
        let c = PressureCurve::new(&pts);
        let ys: Vec<f32> = (0..=100).map(|i| c.eval(i as f32 / 100.0)).collect();
        assert!(ys.iter().all(|y| y.is_finite() && (0.0..=1.0).contains(y)), "{pts:?}");
        assert!(ys.windows(2).all(|w| w[1] >= w[0] - 1e-6), "{pts:?}: {ys:?}");
        for x in [f32::NAN, -1.0, 2.0, f32::INFINITY] {
            assert!((0.0..=1.0).contains(&c.eval(x)), "{pts:?} at {x}");
        }
    }
    // Two points at the same input: the later one's output wins.
    let c = PressureCurve::new(&[[0.0, 0.0], [0.5, 0.2], [0.5, 0.7], [1.0, 1.0]]);
    assert!((c.eval(0.5) - 0.7).abs() < 1e-6);
}

#[test]
fn prefs_set_validates_the_pressure_curve() {
    let mut p = Preferences::default();
    p.set("tools.pressureCurve", serde_json::json!([[0.0, 0.1], [0.5, 0.7], [1.0, 1.0]])).unwrap();
    assert_eq!(p.tools.pressure_curve, vec![[0.0, 0.1], [0.5, 0.7], [1.0, 1.0]]);
    for bad in [
        serde_json::json!("linear"),
        serde_json::json!([[0.0, 0.0]]),
        serde_json::json!([[0.0, 0.0], [1.5, 1.0]]),
        serde_json::json!([[0.0, 0.0], [1.0]]),
        serde_json::json!([[0.0, 0.0], ["a", 1.0]]),
        serde_json::json!(vec![[0.5, 0.5]; 17]),
    ] {
        assert!(p.set("tools.pressureCurve", bad.clone()).is_err(), "{bad}");
    }
    assert_eq!(p.tools.pressure_curve, vec![[0.0, 0.1], [0.5, 0.7], [1.0, 1.0]], "a refused value changes nothing");
}

/// #2434: each unit's readout keeps Photoshop's precision, rounded half away from zero and
/// without trailing zeros, so 0.7995 in reads 0.8 rather than 0.799 or 0.800.
#[test]
fn ruler_units_round_to_their_own_decimal_places() {
    use super::{Unit, fmt_decimals};
    assert_eq!(Unit::Pixels.decimals(), 0);
    assert_eq!(Unit::Points.decimals(), 1);
    for u in [Unit::Millimeters, Unit::Centimeters, Unit::Picas] {
        assert_eq!(u.decimals(), 2, "{u:?}");
    }
    assert_eq!(Unit::Inches.decimals(), 3);
    // Rounded to the unit's places, then trailing zeros dropped.
    assert_eq!(fmt_decimals(0.7995, 3), "0.8");
    assert_eq!(fmt_decimals(0.799, 3), "0.799");
    assert_eq!(fmt_decimals(0.799, 2), "0.8");
    assert_eq!(fmt_decimals(2.0, 3), "2");
    assert_eq!(fmt_decimals(1234.5678, 2), "1234.57");
    // Half away from zero, both signs, and no "-0".
    assert_eq!(fmt_decimals(0.125, 2), "0.13");
    assert_eq!(fmt_decimals(-0.125, 2), "-0.13");
    assert_eq!(fmt_decimals(-0.004, 2), "0");
    // A whole number keeps its zeros: 550 px is not 55.
    assert_eq!(fmt_decimals(550.0, 0), "550");
    // A 1500 px side at 300 dpi is 5 in / 12.7 cm / 127 mm / 360 pt / 30 pica.
    let r = super::UnitsAndRulers { rulers: Unit::Inches, ..Default::default() };
    assert_eq!(r.format(1500.0, 300.0, 1500.0), "5");
    let r = super::UnitsAndRulers { rulers: Unit::Centimeters, ..Default::default() };
    assert_eq!(r.format(1500.0, 300.0, 1500.0), "12.7");
    let r = super::UnitsAndRulers { rulers: Unit::Inches, ..Default::default() };
    assert_eq!(r.format(100.0, 300.0, 1500.0), "0.333");
}

/// #3296: like Photoshop, "Use Shift Key for Tool Switch" is on by default; a stored choice wins.
#[test]
fn shift_key_tool_switch_is_on_by_default_and_a_stored_false_is_kept() {
    assert!(Preferences::default().tools.use_shift_key_for_tool_switch);
    let missing: Tools = serde_json::from_value(json!({"showTooltips": false})).unwrap();
    assert!(missing.use_shift_key_for_tool_switch, "a settings file without the key gets the new default");
    let stored: Tools = serde_json::from_value(json!({"useShiftKeyForToolSwitch": false})).unwrap();
    assert!(!stored.use_shift_key_for_tool_switch);
}
