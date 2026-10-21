use super::*;
use crate::{canvas, dialogs, theme::ThemeKind};
use egui::{Key, vec2};
use egui_kittest::{Harness, kittest::Queryable};

fn fixture(kind: Kind, depth: u32) -> (PhotocraftApp, u32) {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("prefs.set", json!({"path":"interface.language","value":"en"})).unwrap();
    app.run("file.new", json!({"width":320,"height":160,"depth":depth,"background":"transparent"})).unwrap();
    let layer = app.run("type.create", json!({"x":10,"y":55,"text":"Shared color","size":30})).unwrap()["layer"].as_u64().unwrap();
    let prefix = if kind.paragraph() { "type.paragraphStyle" } else { "type.characterStyle" };
    let id = app.run(&format!("{prefix}.new"), json!({"fromSelection":false,"attrs":{"color":"#ff0000"}})).unwrap()["id"].as_u64().unwrap() as u32;
    app.run(&format!("{prefix}.apply"), json!({"id":id,"layer":layer})).unwrap();
    (app, id)
}
fn fields(app: &PhotocraftApp, id: u64) -> Map<String, Value> {
    app.ui.dialogs.iter().find(|d| d.id == id).unwrap().fields.clone()
}
fn set(app: &mut PhotocraftApp, id: u64, color: &str) {
    app.ui.dialog_mut(id).unwrap().fields.insert("color".into(), json!(color));
}

#[test]
fn preview_is_cached_cancel_is_clean_and_ok_is_one_undo_step_at_all_depths() {
    for kind in [Kind::Character, Kind::Paragraph] {
        for depth in [8, 16, 32] {
            let (mut app, id) = fixture(kind, depth);
            let st = app.session.active().unwrap();
            let (doc, rev, dirty, history) = (st.doc.clone(), st.revision, st.is_dirty(), st.history.past_len());
            let original_pixels = photocraft_compose::flatten(&doc);
            let (x, y) =
                (0..160).flat_map(|y| (0..320).map(move |x| (x, y))).find(|&(x, y)| original_pixels.get(x, y)[3] > 0.1).expect("fixture renders visible text");
            let picker = open(&mut app, kind, id).unwrap();
            set(&mut app, picker, "#00ff00");
            let (shown, key) = canvas::display_doc(&mut app, 0);
            let again = canvas::display_doc(&mut app, 0);
            assert!(Arc::ptr_eq(&shown, &again.0));
            assert_eq!(key, again.1);
            assert_ne!(photocraft_compose::flatten(&shown).get(x, y), photocraft_compose::flatten(&doc).get(x, y));
            let st = app.session.active().unwrap();
            assert!(Arc::ptr_eq(&st.doc, &doc));
            assert_eq!((st.revision, st.is_dirty(), st.history.past_len()), (rev, dirty, history));
            set(&mut app, picker, "#0000ff");
            assert_ne!(canvas::display_doc(&mut app, 0).1, key);
            dialogs::cancel(&mut app, picker).unwrap();
            assert!(app.text_style_preview.is_none());
            assert!(Arc::ptr_eq(&canvas::display_doc(&mut app, 0).0, &doc));
            let picker = open(&mut app, kind, id).unwrap();
            set(&mut app, picker, "#00ff00");
            let shown = canvas::display_doc(&mut app, 0).0;
            dialogs::confirm(&mut app, picker).unwrap();
            let st = app.session.active().unwrap();
            assert_eq!(st.history.past_len(), history + 1);
            assert_eq!(serde_json::to_value(&st.doc.text_styles).unwrap(), serde_json::to_value(&shown.text_styles).unwrap());
            assert_eq!(photocraft_compose::flatten(&st.doc).get(x, y), photocraft_compose::flatten(&shown).get(x, y));
            assert!(app.text_style_preview.is_none());
            app.run("edit.undo", json!({})).unwrap();
            assert_eq!(serde_json::to_value(&app.session.active().unwrap().doc.text_styles).unwrap(), serde_json::to_value(&doc.text_styles).unwrap());
            app.run("edit.redo", json!({})).unwrap();
            assert_eq!(color(&app.session.active().unwrap().doc, kind, id).unwrap().to_rgb(), [0.0, 1.0, 0.0]);
        }
    }
}

#[test]
fn no_op_keeps_precise_imported_color_and_returning_to_original_removes_preview() {
    let (mut app, id) = fixture(Kind::Character, 32);
    app.run("type.characterStyle.set", json!({"id":id,"attrs":{"color":photocraft_doc::Color::rgba(0.12345,0.67891,0.33333,0.45678)}})).unwrap();
    let before = app.session.active().unwrap().doc.clone();
    let history = app.session.active().unwrap().history.past_len();
    let picker = open(&mut app, Kind::Character, id).unwrap();
    let original = fields(&app, picker)["color"].as_str().unwrap().to_string();
    set(&mut app, picker, "#ffffff");
    canvas::display_doc(&mut app, 0);
    set(&mut app, picker, &original);
    assert!(Arc::ptr_eq(&canvas::display_doc(&mut app, 0).0, &before));
    assert!(app.text_style_preview.is_none());
    dialogs::confirm(&mut app, picker).unwrap();
    assert!(Arc::ptr_eq(&app.session.active().unwrap().doc, &before));
    assert_eq!(app.session.active().unwrap().history.past_len(), history);
}

#[test]
fn invalidated_targets_never_redirect_and_bad_color_never_commits() {
    for change in 0..6 {
        let (mut app, id) = fixture(Kind::Character, 8);
        let picker = open(&mut app, Kind::Character, id).unwrap();
        set(&mut app, picker, "#00ff00");
        canvas::display_doc(&mut app, 0);
        let pending = fields(&app, picker);
        match change {
            0 => {
                app.run("file.new", json!({"width":10,"height":10})).unwrap();
            }
            1 => {
                app.run("type.characterStyle.delete", json!({"id":id})).unwrap();
            }
            2 => {
                app.run("type.characterStyle.set", json!({"id":id,"attrs":{"size":42}})).unwrap();
            }
            3 => app.ui.type_panels.options = None,
            4 => app.ui.type_panels.styles = false,
            _ => app.ui.type_panels.styles_tab = 1,
        }
        let doc = app.session.active().unwrap().doc.clone();
        assert!(confirm(&mut app, &pending).is_err());
        prune(&mut app);
        assert!(app.ui.dialogs.is_empty());
        assert!(app.text_style_preview.is_none());
        assert!(Arc::ptr_eq(&app.session.active().unwrap().doc, &doc));
    }
    let (mut app, id) = fixture(Kind::Character, 8);
    let picker = open(&mut app, Kind::Character, id).unwrap();
    set(&mut app, picker, "invalid");
    let doc = app.session.active().unwrap().doc.clone();
    assert!(dialogs::confirm(&mut app, picker).is_err());
    assert!(Arc::ptr_eq(&app.session.active().unwrap().doc, &doc));
}

#[test]
fn inherited_colors_and_invalid_open_requests_are_atomic() {
    let (mut app, _) = fixture(Kind::Character, 8);
    let inherited = app.run("type.characterStyle.new", json!({"fromSelection":false})).unwrap()["id"].as_u64().unwrap() as u32;
    let panels = app.ui.type_panels.clone();
    for (kind, id) in [(Kind::Character, 0), (Kind::Character, u32::MAX), (Kind::Character, inherited), (Kind::Paragraph, 0)] {
        assert!(open(&mut app, kind, id).is_err());
        assert_eq!(app.ui.type_panels, panels);
        assert!(app.ui.dialogs.is_empty());
    }
    app.run("type.paragraphStyle.set", json!({"id":0,"attrs":{"color":"#112233"}})).unwrap();
    let picker = open(&mut app, Kind::Paragraph, 0).unwrap();
    set(&mut app, picker, "#abcdef");
    dialogs::confirm(&mut app, picker).unwrap();
    assert_eq!(color_picker_ui::hex(color(&app.session.active().unwrap().doc, Kind::Paragraph, 0).unwrap().to_rgb()), "#abcdef");
}

fn harness(kind: Kind) -> Harness<'static, PhotocraftApp> {
    let (mut app, id) = fixture(kind, 8);
    app.ui.type_panels.styles = true;
    app.ui.type_panels.styles_tab = usize::from(kind.paragraph());
    app.ui.type_panels.options = Some((kind.paragraph(), id));
    let mut h = Harness::builder().with_size(vec2(1500.0, 1000.0)).build_ui_state(
        move |ui, app| {
            if ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                super::super::styles_panel(app, ui, kind.paragraph());
                dialogs::show(app, ui.ctx());
            }
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, ThemeKind::ProMedium);
    h.run_steps(4);
    h
}

#[test]
fn actual_style_swatch_opens_shared_picker_enter_commits_escape_cancels() {
    for kind in [Kind::Character, Kind::Paragraph] {
        for key in [Key::Enter, Key::Escape] {
            let mut h = harness(kind);
            let doc = h.state().session.active().unwrap().doc.clone();
            h.get_by_role(egui::accesskit::Role::ColorWell).click();
            h.run();
            assert_eq!(h.state().ui.dialogs.len(), 1);
            let picker = h.state().ui.dialogs[0].id;
            assert_eq!(fields(h.state(), picker)["__label"], "Color Picker (Text Color)");
            set(h.state_mut(), picker, "#00ff00");
            h.key_press(key);
            h.run();
            assert!(h.state().ui.dialogs.is_empty());
            if key == Key::Escape {
                assert!(Arc::ptr_eq(&h.state().session.active().unwrap().doc, &doc));
            } else {
                assert!(!Arc::ptr_eq(&h.state().session.active().unwrap().doc, &doc));
            }
            assert!(h.state().ui.type_panels.options.is_some());
        }
    }
}

#[test]
#[ignore = "manual preview performance comparison"]
fn preview_timings() {
    let (mut app, id) = fixture(Kind::Character, 8);
    let t = std::time::Instant::now();
    for _ in 0..100 {
        canvas::display_doc(&mut app, 0);
    }
    let plain = t.elapsed();
    let picker = open(&mut app, Kind::Character, id).unwrap();
    let t = std::time::Instant::now();
    for i in 0..20 {
        set(&mut app, picker, &format!("#{:06x}", 0x008000 + i));
        canvas::display_doc(&mut app, 0);
    }
    let changes = t.elapsed();
    let t = std::time::Instant::now();
    for _ in 0..100 {
        canvas::display_doc(&mut app, 0);
    }
    println!("plain 100 frames: {plain:?}; 20 color changes: {changes:?}; cached 100 frames: {:?}", t.elapsed());
}
