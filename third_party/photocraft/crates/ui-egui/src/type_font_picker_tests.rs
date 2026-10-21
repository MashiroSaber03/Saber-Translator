//! The Type tool's searchable font menu (#1369): it keeps its height and focuses its search field.

use super::font_picker_in;
use egui::accesskit::Role;
use egui::{Key, Modifiers, Rect, vec2};
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use photocraft_engine::prefs::FontPreview;

struct State {
    current: String,
    families: Vec<String>,
    popup: egui::Id,
    preview: FontPreview,
}

fn harness() -> Harness<'static, State> {
    harness_with(FontPreview::Medium)
}

fn harness_with(preview: FontPreview) -> Harness<'static, State> {
    // Enough families that the unfiltered menu hits its maximum height and scrolls.
    let mut families: Vec<String> = (0..60).map(|i| format!("Family {i:02}")).collect();
    families.push("Zeta Sans".into());
    let state = State { current: "Family 00".into(), families, popup: egui::Id::NULL, preview };
    let mut h = Harness::builder().with_size(vec2(600.0, 900.0)).build_ui_state(
        |ui, s: &mut State| {
            s.popup = ui.make_persistent_id(egui::IdSalt::new("type-font")).with("popup");
            let fams = s.families.clone();
            font_picker_in(ui, &mut s.current, 170.0, &fams, s.preview);
        },
        state,
    );
    h.run();
    h
}

fn popup_rect(h: &Harness<'static, State>) -> Option<Rect> {
    let id = h.state().popup;
    let open = egui::Popup::is_id_open(&h.ctx, id);
    if open { h.ctx.memory(|m| m.area_rect(id)) } else { None }
}

fn open(h: &mut Harness<'static, State>) {
    h.query_all_by_role(Role::ComboBox).next().unwrap().click();
    h.run();
}

#[test]
fn hovering_a_font_row_sets_preview_and_leaving_clears_it() {
    let mut h = harness();
    open(&mut h);
    let row = h.get_by_label("Family 02").rect().center();
    let popup = popup_rect(&h).expect("font menu open");
    assert!(popup.contains(row), "row {row:?} is outside popup {popup:?}");
    h.hover_at(row);
    h.run_steps(2);
    let hover_id = egui::Id::new("type-font-hover");
    assert_eq!(h.ctx.input(|i| i.pointer.interact_pos()), Some(row));
    assert!(popup_rect(&h).is_some(), "popup closes while moving pointer to the row");
    assert_eq!(
        h.ctx.data(|d| d.get_temp::<String>(hover_id)).as_deref(),
        Some("Family 02"),
        "row {row:?}, popup {popup:?}, top layer {:?}",
        h.ctx.layer_id_at(row)
    );

    let search = h.get_by_role(Role::TextInput).rect().center();
    h.hover_at(search);
    h.run();
    assert!(h.ctx.data(|d| d.get_temp::<String>(hover_id)).is_none());
}

#[test]
fn clearing_the_query_restores_the_menu_height() {
    let mut h = harness();
    open(&mut h);
    let full = popup_rect(&h).expect("menu open").height();
    h.get_by_role(Role::TextInput).type_text("zeta");
    h.run();
    assert!(h.query_by_label("Family 01").is_none(), "the query filters the list");
    let searching = popup_rect(&h).expect("still open").height();
    for _ in 0.."zeta".len() {
        h.key_press(Key::Backspace);
    }
    h.run();
    assert!(h.query_by_label("Family 01").is_some(), "the cleared query lists every family");
    let cleared = popup_rect(&h).expect("still open").height();
    assert!((cleared - full).abs() < 0.5, "height after clearing {cleared} != original {full}");
    assert!((searching - full).abs() < 0.5, "the menu keeps its height while searching: {searching} vs {full}");
}

#[test]
fn reopening_focuses_the_search_field_and_clicking_it_keeps_the_menu_open() {
    let mut h = harness();
    open(&mut h);
    assert!(h.get_by_role(Role::TextInput).is_focused(), "search is focused on first open");
    h.get_by_role(Role::TextInput).type_text("zeta");
    h.run();
    // Close without choosing, then reopen: the query stays and the field is focused again.
    h.key_press(Key::Escape);
    h.run();
    assert!(popup_rect(&h).is_none(), "Escape closes the menu");
    open(&mut h);
    let search = h.get_by_role(Role::TextInput);
    assert_eq!(search.value().as_deref(), Some("zeta"));
    assert!(search.is_focused(), "search is focused on reopen with a query");
    // Clicking the search field (to edit the query) must not close the menu.
    h.get_by_role(Role::TextInput).click();
    h.run();
    assert!(popup_rect(&h).is_some(), "clicking the search field keeps the menu open");
    h.key_press_modifiers(Modifiers::COMMAND, Key::A);
    h.get_by_role(Role::TextInput).type_text("Zeta S");
    h.run();
    assert!(popup_rect(&h).is_some());
    // Picking a font still closes the menu.
    h.get_by_label("Zeta Sans").click();
    h.run();
    assert_eq!(h.state().current, "Zeta Sans");
    assert!(popup_rect(&h).is_none(), "choosing a font closes the menu");
}

#[test]
fn arrows_apply_fonts_without_closing_and_enter_accepts() {
    let mut h = harness();
    open(&mut h);
    h.key_press(Key::ArrowDown);
    h.run();
    assert_eq!(h.state().current, "Family 01");
    assert!(popup_rect(&h).is_some());
    h.key_press(Key::ArrowDown);
    h.key_press(Key::ArrowUp);
    h.run();
    assert_eq!(h.state().current, "Family 01");
    h.key_press(Key::Enter);
    h.run();
    assert!(popup_rect(&h).is_none());
}

#[test]
fn arrows_follow_the_filtered_list_and_handle_no_matches() {
    let mut h = harness();
    open(&mut h);
    h.get_by_role(Role::TextInput).type_text("Family 5");
    h.run();
    h.key_press(Key::ArrowDown);
    h.run();
    assert_eq!(h.state().current, "Family 50");
    h.key_press(Key::ArrowUp);
    h.run();
    assert_eq!(h.state().current, "Family 50", "clamp at the first match");
    h.get_by_role(Role::TextInput).type_text("missing");
    h.run();
    h.key_press(Key::ArrowDown);
    h.key_press(Key::ArrowUp);
    h.key_press(Key::Enter);
    h.run();
    assert_eq!(h.state().current, "Family 50");
    assert!(popup_rect(&h).is_some(), "no match cannot accept an unrelated font");
}

#[test]
fn arrows_scroll_the_selected_font_into_view() {
    let mut h = harness();
    open(&mut h);
    for _ in 0..59 {
        h.key_press(Key::ArrowDown);
        h.run();
    }
    assert_eq!(h.state().current, "Family 59");
    let row = h.get_by_label("Family 59");
    assert!(popup_rect(&h).unwrap().contains(row.rect().center()));
    h.key_press(Key::ArrowDown);
    h.key_press(Key::ArrowDown);
    h.run();
    assert_eq!(h.state().current, "Zeta Sans", "clamp at the last font");
}

#[test]
fn empty_font_list_ignores_navigation() {
    let mut h = harness();
    h.state_mut().families.clear();
    open(&mut h);
    h.key_press(Key::ArrowDown);
    h.key_press(Key::ArrowUp);
    h.key_press(Key::Enter);
    h.run();
    assert_eq!(h.state().current, "Family 00");
    assert!(popup_rect(&h).is_some());
}

#[test]
fn font_cycle_in_options_bar_edits_the_layer_and_can_be_undone() {
    use crate::{PhotocraftApp, Services};
    use serde_json::json;
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Services::default());
    app.run("file.new", json!({"width": 200, "height": 100})).unwrap();
    app.run("type.create", json!({"text": "Sample", "font": "Inter", "size": 18})).unwrap();
    let mut h = Harness::builder().with_size(vec2(900.0, 700.0)).build_ui_state(
        |ui, app| {
            if ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                ui.horizontal(|ui| crate::type_tool::options_bar(app, ui));
            }
        },
        app,
    );
    crate::PhotocraftApp::setup_context(&h.ctx, Default::default());
    h.run_steps(4);
    h.query_all_by_role(Role::ComboBox).next().unwrap().click();
    h.run();
    h.get_by_role(Role::TextInput).type_text("JetBrains");
    h.key_press(Key::ArrowDown);
    h.run();
    let font = |app: &PhotocraftApp| {
        let doc = &app.session.active().unwrap().doc;
        match &doc.layer(app.session.active().unwrap().active_layer.unwrap()).unwrap().content {
            photocraft_doc::LayerContent::Text(t) => t.font_family.clone(),
            _ => panic!("type layer"),
        }
    };
    assert_eq!(font(h.state()), "JetBrains Mono");
    h.key_press(Key::Enter);
    h.run();
    h.state_mut().run("edit.undo", json!({})).unwrap();
    assert_eq!(font(h.state()), "Inter");
    h.state_mut().run("edit.redo", json!({})).unwrap();
    assert_eq!(font(h.state()), "JetBrains Mono");
}

#[test]
fn hovering_a_font_does_not_pollute_history_and_clears_on_close() {
    use crate::{PhotocraftApp, Services};
    use serde_json::json;

    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Services::default());
    app.run("file.new", json!({"width": 200, "height": 100})).unwrap();
    app.run("type.create", json!({"text": "Sample", "font": "Inter", "size": 18})).unwrap();

    let layer_id = app.session.active().unwrap().active_layer.unwrap();
    let history_before = app.session.active().unwrap().history.entries().len();

    // Simulate hovering a font directly via the preview module
    let ctx = egui::Context::default();
    ctx.data_mut(|d| d.insert_temp(egui::Id::new("type-font-hover"), "JetBrains Mono".to_string()));
    crate::type_font_preview::follow_hover(&mut app, &ctx);

    // Verify a preview document is generated
    let (_, preview_key) = crate::type_font_preview::display_doc(&mut app, 0).expect("preview document");

    // Verify the original document's history is NOT polluted
    let history_after = app.session.active().unwrap().history.entries().len();
    assert_eq!(history_before, history_after, "Hovering a font does not create an undo step");
    let doc = app.session.active().unwrap().doc.id;
    let revision = app.session.active().unwrap().revision;
    assert!(crate::type_font_preview::damage(&app, doc, revision, 0, preview_key).is_some(), "preview bounds survive dismissal for canvas invalidation");

    // Verify the actual document layer still has the original font
    let original_font = match &app.session.active().unwrap().doc.layer(layer_id).unwrap().content {
        photocraft_doc::LayerContent::Text(t) => t.font_family.clone(),
        _ => panic!("type layer"),
    };
    assert_eq!(original_font, "Inter", "Hovering does not commit the font to the document");

    // Simulate the font menu closing (hover cleared)
    crate::type_font_preview::follow_hover(&mut app, &ctx);
    let no_preview = crate::type_font_preview::display_doc(&mut app, 0);
    assert!(no_preview.is_none(), "No preview document is returned after clearing");

    crate::type_font_preview::set(&mut app, "Unavailable Font".into());
    assert!(crate::type_font_preview::display_doc(&mut app, 0).is_none(), "unavailable fonts do not render a misleading fallback preview");
}

#[test]
fn options_bar_hover_previews_on_canvas_without_committing() {
    use crate::{PhotocraftApp, Services};
    use photocraft_doc::LayerContent;
    use serde_json::json;

    struct PreviewApp {
        app: PhotocraftApp,
    }

    impl eframe::App for PreviewApp {
        fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
            crate::type_font_preview::follow_hover(&mut self.app, ctx);
        }

        fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
            ui.horizontal(|ui| crate::type_tool::options_bar(&mut self.app, ui));
        }
    }

    let mut h = Harness::builder().with_size(vec2(900.0, 700.0)).build_eframe(|cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, crate::theme::ThemeKind::ALL[0]);
        PreviewApp { app: PhotocraftApp::new(photocraft_engine::Session::new(), Services::default()) }
    });
    h.state_mut().app.run("file.new", json!({"width": 200, "height": 100})).unwrap();
    h.state_mut().app.run("type.create", json!({"text": "Sample", "font": "Inter", "size": 18})).unwrap();
    let layer = h.state().app.session.active().unwrap().active_layer.unwrap();
    h.state_mut().app.ui.tool = crate::state::Tool::Type;
    h.run_steps(3);
    h.query_all_by_role(Role::ComboBox).next().unwrap().click();
    h.run();
    h.get_by_role(Role::TextInput).type_text("JetBrains Mono");
    h.run();
    h.get_by_label("JetBrains Mono").hover();
    h.run();

    let (preview, _) = crate::canvas::display_doc(&mut h.state_mut().app, 0);
    let LayerContent::Text(preview_text) = &preview.layer(layer).unwrap().content else {
        panic!("type layer");
    };
    assert_eq!(preview_text.char_runs()[0].style.font_family, "JetBrains Mono");
    let LayerContent::Text(original_text) = &h.state().app.session.active().unwrap().doc.layer(layer).unwrap().content else {
        panic!("type layer");
    };
    assert_eq!(original_text.char_runs()[0].style.font_family, "Inter");
    assert_eq!(h.state().app.session.active().unwrap().history.entries().len(), 2, "hover adds no history entry");

    h.get_by_role(Role::TextInput).hover();
    h.run();
    let (restored, key) = crate::canvas::display_doc(&mut h.state_mut().app, 0);
    let LayerContent::Text(restored_text) = &restored.layer(layer).unwrap().content else {
        panic!("type layer");
    };
    assert_eq!(restored_text.char_runs()[0].style.font_family, "Inter");
    assert_eq!(key, 0, "leaving the font row restores committed canvas content");

    h.get_by_label("JetBrains Mono").hover();
    h.run();
    h.get_by_label("JetBrains Mono").click();
    h.run();
    let (committed, key) = crate::canvas::display_doc(&mut h.state_mut().app, 0);
    let LayerContent::Text(committed_text) = &committed.layer(layer).unwrap().content else {
        panic!("type layer");
    };
    assert_eq!(committed_text.char_runs()[0].style.font_family, "JetBrains Mono");
    assert_eq!(key, 0, "selection replaces the preview with committed content");
    assert_eq!(h.state().app.session.active().unwrap().history.entries().len(), 3, "selection adds one history step");
    let committed_font = |app: &PhotocraftApp| {
        let LayerContent::Text(text) = &app.session.active().unwrap().doc.layer(layer).unwrap().content else {
            panic!("type layer");
        };
        text.char_runs()[0].style.font_family.clone()
    };
    h.state_mut().app.run("edit.undo", json!({})).unwrap();
    assert_eq!(committed_font(&h.state().app), "Inter");
    h.state_mut().app.run("edit.redo", json!({})).unwrap();
    assert_eq!(committed_font(&h.state().app), "JetBrains Mono");
}

#[test]
fn hovering_a_font_previews_only_the_selected_characters() {
    use crate::{PhotocraftApp, Services};
    use serde_json::json;

    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Services::default());
    app.run("file.new", json!({"width": 200, "height": 100})).unwrap();
    app.run("type.create", json!({"text": "abcdef", "font": "Inter", "size": 18})).unwrap();
    let layer = app.session.active().unwrap().active_layer.unwrap();
    app.run("type.setStyle", json!({"layer": layer.0, "range": [0, 2], "font": "JetBrains Mono"})).unwrap();
    app.ui.text_edit = Some(crate::state::TextEdit {
        layer: layer.0,
        caret: 5,
        anchor: 3,
        session: "font-preview-test".into(),
        created: false,
        mask: false,
        dragging: false,
        resize: None,
        preedit: None,
    });

    crate::type_font_preview::set(&mut app, "Inter".into());
    let (preview, _) = crate::type_font_preview::display_doc(&mut app, 0).expect("preview document");
    let fonts = |doc: &photocraft_doc::Document| {
        let photocraft_doc::LayerContent::Text(text) = &doc.layer(layer).unwrap().content else {
            panic!("type layer");
        };
        text.char_runs().into_iter().flat_map(|run| std::iter::repeat_n(run.style.font_family, run.len)).collect::<Vec<_>>()
    };

    assert_eq!(fonts(&preview), ["JetBrains Mono", "JetBrains Mono", "Inter", "Inter", "Inter", "Inter"]);
    assert_eq!(fonts(&app.session.active().unwrap().doc), ["JetBrains Mono", "JetBrains Mono", "Inter", "Inter", "Inter", "Inter"]);
}

/// Preferences ▸ Type ▸ Font Preview Size: Extra Large grows the sample rows, Off drops the
/// samples (rows shrink to the plain height).
#[test]
fn the_preview_size_preference_resizes_or_removes_the_samples() {
    let mut h = harness_with(FontPreview::Medium);
    open(&mut h);
    let medium = h.get_by_label("Family 00").rect().height();
    assert!((medium - 28.0).abs() < 0.5, "medium rows hold the 24 px sample: {medium}");
    let mut h = harness_with(FontPreview::ExtraLarge);
    open(&mut h);
    let extra = h.get_by_label("Family 00").rect().height();
    assert!(extra > medium, "extra large rows are taller: {extra} vs {medium}");
    let mut h = harness_with(FontPreview::Off);
    open(&mut h);
    let off = h.get_by_label("Family 00").rect().height();
    assert!((off - 28.0).abs() < 0.5, "off keeps the plain row height: {off}");
}

/// Preferences ▸ Type ▸ Show Font Names in English off: the menu lists a family by its localized
/// name, finds it by either name, and picking it still applies the English family.
#[test]
fn localized_font_names_label_the_menu_but_pick_the_family() {
    use std::collections::HashMap;
    use std::sync::Arc;
    let state = State {
        current: "Arial".into(),
        families: vec!["Arial".into(), "MS Gothic".into(), "Zeta Sans".into()],
        popup: egui::Id::NULL,
        preview: FontPreview::Medium,
    };
    let mut h = Harness::builder().with_size(vec2(600.0, 900.0)).build_ui_state(
        |ui, s: &mut State| {
            s.popup = ui.make_persistent_id(egui::IdSalt::new("type-font")).with("popup");
            let fams = s.families.clone();
            font_picker_in(ui, &mut s.current, 170.0, &fams, s.preview);
        },
        state,
    );
    // Show Font Names in English on: no localized labels.
    crate::type_tool::sync_font_names(&h.ctx, true);
    let labels: HashMap<String, String> = [("MS Gothic".to_string(), "ＭＳ ゴシック".to_string())].into();
    h.ctx.data_mut(|d| d.insert_temp(super::font_labels_id(), Arc::new(labels.clone())));
    h.run();
    open(&mut h);
    assert!(h.query_by_label("ＭＳ ゴシック").is_some(), "the localized name labels the family");
    assert!(h.query_by_label("MS Gothic").is_none());
    assert!(h.query_by_label("Arial").is_some(), "fonts without a localized name keep theirs");
    h.get_by_role(Role::TextInput).type_text("gothic");
    h.run();
    assert!(h.query_by_label("ＭＳ ゴシック").is_some(), "the English name still finds it");
    assert!(h.query_by_label("Arial").is_none());
    h.get_by_label("ＭＳ ゴシック").click();
    h.run();
    assert_eq!(h.state().current, "MS Gothic", "the document keeps the English family");
    // Turning the preference back on drops the labels.
    crate::type_tool::sync_font_names(&h.ctx, true);
    h.run();
    open(&mut h);
    assert!(h.query_by_label("MS Gothic").is_some());
    assert!(h.query_by_label("ＭＳ ゴシック").is_none());
}

#[test]
fn font_menu_rows_search_both_names_and_sort_by_label() {
    use std::collections::HashMap;
    let families: Vec<String> = ["Arial", "Microsoft YaHei", "MS Gothic"].iter().map(|s| s.to_string()).collect();
    let labels: HashMap<String, String> =
        [("MS Gothic", "ＭＳ ゴシック"), ("Microsoft YaHei", "微软雅黑")].iter().map(|(a, b)| (a.to_string(), b.to_string())).collect();
    let rows = |labels: Option<&HashMap<String, String>>, q: &str| -> Vec<(String, String)> {
        super::font_menu_rows(&families, labels, q).into_iter().map(|(f, l)| (f.clone(), l.to_string())).collect()
    };
    let english = rows(None, "");
    assert!(english.iter().all(|(f, l)| f == l));
    let local = rows(Some(&labels), "");
    assert_eq!(local.iter().map(|(_, l)| l.as_str()).collect::<Vec<_>>(), ["Arial", "微软雅黑", "ＭＳ ゴシック"]);
    assert_eq!(rows(Some(&labels), "yahei"), [("Microsoft YaHei".to_string(), "微软雅黑".to_string())]);
    assert_eq!(rows(Some(&labels), "雅黑"), [("Microsoft YaHei".to_string(), "微软雅黑".to_string())]);
    assert!(rows(Some(&labels), "nothing").is_empty());
}

#[test]
fn wheel_over_the_closed_font_button_steps_through_the_families() {
    let mut h = harness();
    let button = h.query_all_by_role(Role::ComboBox).next().unwrap().rect().center();
    h.hover_at(button);
    h.run();
    let wheel = |h: &mut Harness<'static, State>, y: f32| {
        h.event(egui::Event::MouseWheel { unit: egui::MouseWheelUnit::Line, delta: vec2(0.0, y), phase: egui::TouchPhase::Move, modifiers: Modifiers::NONE });
        h.run();
    };
    wheel(&mut h, -2.0);
    assert_eq!(h.state().current, "Family 02", "two notches down pick the second next family");
    assert!(popup_rect(&h).is_none(), "the list stays closed");
    wheel(&mut h, 1.0);
    assert_eq!(h.state().current, "Family 01");
    wheel(&mut h, 5.0);
    assert_eq!(h.state().current, "Family 00", "stops at the first family");
}

#[test]
fn wheel_family_index_clamps_to_the_ends() {
    assert_eq!(super::wheel_family_index(0, 3, -5.0), Some(2));
    assert_eq!(super::wheel_family_index(1, 3, 1.0), Some(0));
    assert_eq!(super::wheel_family_index(1, 3, 0.0), None);
    assert_eq!(super::wheel_family_index(0, 0, -1.0), None);
}
