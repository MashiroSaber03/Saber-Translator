use super::*;
use egui::accesskit::Role;
use egui::{Key, Modifiers, PointerButton, Pos2, Rect, Vec2, pos2};
use egui_kittest::{
    Harness,
    kittest::{NodeT, Queryable},
};

struct State {
    current: StrokeOptions,
    editor: Option<StrokeEditor>,
    changes: Vec<String>,
    stroke_enabled: bool,
    closed: bool,
    corners: bool,
    popup: Option<Rect>,
}

impl Default for State {
    fn default() -> Self {
        Self { current: StrokeOptions::default(), editor: None, changes: Vec::new(), stroke_enabled: true, closed: true, corners: true, popup: None }
    }
}

fn sized_harness(size: Vec2, scale: f32, theme: crate::theme::ThemeKind) -> Harness<'static, State> {
    let mut h = Harness::builder().with_size(size).with_pixels_per_point(scale).build_ui_state(
        |ui, s: &mut State| {
            let _language = crate::i18n::language_scope(crate::i18n::Lang::EN);
            let popup_id = ui.next_auto_id().with("popup");
            if let Some(change) = button(ui, "test", &s.current, &mut s.editor, s.closed, s.corners, None, s.stroke_enabled) {
                s.current = change.options;
                s.stroke_enabled = true;
                s.changes.push(change.coalesce);
            }
            s.popup = s.editor.as_ref().and_then(|_| ui.ctx().read_response(popup_id).map(|r| r.rect));
        },
        State::default(),
    );
    crate::PhotocraftApp::setup_context(&h.ctx, theme);
    h.run_steps(3);
    h
}

fn harness() -> Harness<'static, State> {
    sized_harness(vec2(640.0, 800.0), 1.0, crate::theme::ThemeKind::ProMedium)
}

fn click_at<S>(h: &mut Harness<'_, S>, pos: Pos2) {
    h.hover_at(pos);
    h.run_steps(1);
    for pressed in [true, false] {
        h.event(egui::Event::PointerButton { pos, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE });
        h.run_steps(1);
    }
    h.run_steps(3);
}

fn open(h: &mut Harness<'_, State>) {
    let rect = h.get_by_role_and_label(Role::Button, "Stroke Options").rect();
    click_at(h, rect.center());
    assert!(h.state().editor.is_some());
    assert!(h.query_by_role_and_label(Role::RadioButton, "Dotted").is_some());
}

fn combo(h: &mut Harness<'_, State>, label: &str) {
    let rect = h.get_by_role_and_label(Role::ComboBox, label).rect();
    click_at(h, rect.center());
}

#[test]
fn line_samples_and_row_edges_apply_immediately_and_survive_dismissal() {
    let mut h = harness();
    open(&mut h);
    assert!(h.query_by_label("Apply").is_none());
    assert!(h.query_by_label("Cancel").is_none());
    // The right half is painted line sample, well away from the text.
    let dotted = h.get_by_role_and_label(Role::RadioButton, "Dotted").rect();
    click_at(&mut h, pos2(dotted.right() - 16.0, dotted.center().y));
    assert_eq!(h.state().current.dashes, [0.0, 2.0]);
    assert_eq!(h.state().current.cap, LineCap::Round);
    assert_eq!(h.state().changes.len(), 1);
    assert!(h.state().editor.is_some(), "a direct edit keeps the editor open");
    assert_eq!(h.get_by_role_and_label(Role::RadioButton, "Dotted").accesskit_node().toggled(), Some(egui::accesskit::Toggled::True));
    // The empty padding at the row's left edge is part of the same target.
    let dashed = h.get_by_role_and_label(Role::RadioButton, "Dashed").rect();
    click_at(&mut h, pos2(dashed.left() + 3.0, dashed.center().y));
    assert_eq!(h.state().current.dashes, [4.0, 2.0]);
    assert_eq!(h.state().changes.len(), 2);
    assert_ne!(h.state().changes[0], h.state().changes[1], "separate clicks are separate undo gestures");
    click_at(&mut h, pos2(620.0, 780.0));
    assert!(h.state().editor.is_none());
    assert_eq!(h.state().current.dashes, [4.0, 2.0], "dismissal must retain the live change");
    assert_eq!(h.state().changes.len(), 2);
}

#[test]
fn child_dropdown_open_selection_and_escape_leave_the_parent_alive() {
    let mut h = harness();
    open(&mut h);
    combo(&mut h, "Alignment");
    assert!(h.state().editor.is_some(), "opening a ComboBox must not replace the parent popup");
    assert!(egui::Popup::is_any_open(&h.ctx));
    let inside = h.get_by_label("Inside").rect();
    click_at(&mut h, inside.center());
    assert_eq!(h.state().current.align, StrokeAlign::Inside);
    assert_eq!(h.state().changes.len(), 1);
    assert!(h.state().editor.is_some(), "selecting a child item must not dismiss its parent");
    assert!(!egui::Popup::is_any_open(&h.ctx));
    combo(&mut h, "Alignment");
    h.key_press(Key::Escape);
    h.run_steps(3);
    assert!(!egui::Popup::is_any_open(&h.ctx));
    assert!(h.state().editor.is_some(), "the first Escape belongs to the child");
    h.key_press(Key::Escape);
    h.run_steps(3);
    assert!(h.state().editor.is_none(), "the second Escape dismisses the parent");
    assert_eq!(h.state().current.align, StrokeAlign::Inside);
    assert_eq!(h.state().changes.len(), 1);
    assert!(h.get_by_role_and_label(Role::Button, "Stroke Options").is_focused());
}

#[test]
fn reopened_child_stays_above_a_parent_with_restored_area_order() {
    let mut h = harness();
    open(&mut h);
    let alignment = h.get_by_role_and_label(Role::ComboBox, "Alignment").rect().center();
    let parent_layer = h.ctx.layer_id_at(alignment).unwrap();
    combo(&mut h, "Alignment");
    let outside = h.get_by_label("Outside").rect().center();
    let child_layer = h.ctx.layer_id_at(outside).unwrap();
    assert_ne!(parent_layer, child_layer);
    h.key_press(Key::Escape);
    h.run_steps(3);
    // Reproduce persisted Area order where the parent was brought above its closed child.
    h.ctx.move_to_top(parent_layer);
    h.run_steps(1);
    h.key_press(Key::Escape);
    h.run_steps(3);
    open(&mut h);
    combo(&mut h, "Alignment");
    let outside = h.get_by_label("Outside").rect().center();
    assert_eq!(h.ctx.layer_id_at(outside), Some(child_layer), "the open child must receive input above its parent");
    click_at(&mut h, outside);
    assert_eq!(h.state().current.align, StrokeAlign::Outside);
    assert!(h.state().editor.is_some());
    assert!(!egui::Popup::is_any_open(&h.ctx));
}

#[test]
fn click_outside_parent_and_child_closes_both_without_changing_the_style() {
    let mut h = harness();
    open(&mut h);
    combo(&mut h, "Corners");
    assert!(egui::Popup::is_any_open(&h.ctx));
    click_at(&mut h, pos2(620.0, 780.0));
    assert!(!egui::Popup::is_any_open(&h.ctx));
    assert!(h.state().editor.is_none());
    assert!(h.state().changes.is_empty());
    open(&mut h);
    assert!(!egui::Popup::is_any_open(&h.ctx), "reopening does not restore an abandoned child");
}

#[test]
fn closing_the_opener_cleans_up_its_child_in_the_same_frame() {
    let mut h = harness();
    open(&mut h);
    combo(&mut h, "Alignment");
    assert!(egui::Popup::is_any_open(&h.ctx));
    let pos = h.get_by_role_and_label(Role::Button, "Stroke Options").rect().center();
    h.hover_at(pos);
    h.run_steps(1);
    for pressed in [true, false] {
        h.event(egui::Event::PointerButton { pos, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE });
        h.run_steps(1);
    }
    assert!(h.state().editor.is_none());
    assert!(!egui::Popup::is_any_open(&h.ctx), "a hidden parent must not leave a child owning popup state");
    assert!(h.state().changes.is_empty());
}

#[test]
fn opening_and_closing_by_button_escape_or_outside_is_a_no_op() {
    let mut h = harness();
    h.state_mut().current.opacity = 0.613;
    h.state_mut().current.dashes = vec![1.25, 2.75, 5.5, 3.0];
    h.state_mut().current.dash_offset = -0.625;
    let before = h.state().current.clone();
    for mode in 0..3 {
        open(&mut h);
        h.run_steps(8);
        assert_eq!(h.state().current, before);
        match mode {
            0 => {
                h.get_by_role_and_label(Role::Button, "Stroke Options").click();
                h.run_steps(3);
            }
            1 => {
                h.key_press(Key::Escape);
                h.run_steps(3);
            }
            _ => click_at(&mut h, pos2(620.0, 780.0)),
        }
        assert!(h.state().editor.is_none());
        assert_eq!(h.state().current, before);
        assert!(h.state().changes.is_empty(), "merely viewing custom settings emits no edit");
    }
}

#[test]
fn stroke_options_remain_available_without_stroke_and_a_preset_enables_it() {
    let mut h = harness();
    h.state_mut().stroke_enabled = false;
    h.run_steps(2);
    open(&mut h);
    assert!(!h.state().stroke_enabled, "opening the editor must preserve No Color");
    h.key_press(Key::Escape);
    h.run_steps(3);
    assert!(!h.state().stroke_enabled);
    assert!(h.state().changes.is_empty());
    open(&mut h);
    let solid = h.get_by_role_and_label(Role::RadioButton, "Solid").rect();
    click_at(&mut h, pos2(solid.right() - 12.0, solid.center().y));
    assert!(h.state().stroke_enabled, "a deliberate preset choice can enable a zero-width stroke");
    assert_eq!(h.state().changes.len(), 1);
    assert!(h.state().current.dashes.is_empty());
}

#[test]
fn tab_and_space_operate_the_opener_and_the_complete_style_target() {
    let mut h = harness();
    h.key_press(Key::Tab);
    h.run_steps(3);
    assert!(h.get_by_role_and_label(Role::Button, "Stroke Options").is_focused());
    h.key_press(Key::Space);
    h.run_steps(3);
    assert!(h.state().editor.is_some());
    // Tab navigation reaches the radio targets without a nested sample button.
    let mut reached = false;
    for _ in 0..12 {
        h.key_press(Key::Tab);
        h.run_steps(2);
        if h.get_by_role_and_label(Role::RadioButton, "Dotted").is_focused() {
            reached = true;
            break;
        }
    }
    assert!(reached, "the complete Dotted target participates in keyboard navigation");
    h.key_press(Key::Space);
    h.run_steps(3);
    assert_eq!(h.state().current.dashes, [0.0, 2.0]);
    assert_eq!(h.state().changes.len(), 1);
    assert!(h.state().editor.is_some());
}

#[test]
fn popup_is_content_sized_and_keeps_presets_inside_small_windows_in_all_themes() {
    for theme in crate::theme::ThemeKind::ALL {
        for scale in [1.0, 2.0] {
            let mut h = sized_harness(vec2(720.0, 900.0), scale, theme);
            open(&mut h);
            let large = h.state().popup.expect("the parent popup response");
            assert!(large.height() < 500.0, "{theme:?} {scale}x: empty space must not fill the 900pt window: {large:?}");
            h.set_size(vec2(720.0, 1200.0));
            h.run_steps(4);
            let taller = h.state().popup.unwrap();
            assert!((taller.height() - large.height()).abs() <= 1.0, "{theme:?} {scale}x: unchanged content has a stable natural height");
            h.set_size(vec2(340.0, 380.0));
            h.run_steps(5);
            let small = h.state().popup.unwrap();
            let viewport = Rect::from_min_size(Pos2::ZERO, vec2(340.0, 380.0)).expand(1.0);
            assert!(viewport.contains_rect(small), "{theme:?} {scale}x: the popup fits the smaller viewport: {small:?}");
            for name in ["Solid", "Dashed", "Dotted"] {
                let row = h.get_by_role_and_label(Role::RadioButton, name).rect();
                assert!(viewport.contains_rect(row), "{theme:?} {scale}x: {name} is clipped: {row:?}");
                assert!(row.width() >= 230.0, "the sample and label retain a useful combined target");
            }
            let dotted = h.get_by_role_and_label(Role::RadioButton, "Dotted").rect();
            click_at(&mut h, pos2(dotted.right() - 16.0, dotted.center().y));
            assert_eq!(h.state().current.dashes, [0.0, 2.0]);
            assert!(viewport.contains_rect(h.state().popup.unwrap()));
        }
    }
}

fn assert_bottom_sample_visible(h: &Harness<'_, State>) {
    let popup = h.state().popup.unwrap();
    let (sample, clip) = h
        .output()
        .shapes
        .iter()
        .find_map(|shape| {
            let egui::Shape::Mesh(mesh) = &shape.shape else {
                return None;
            };
            let rect = mesh.calc_bounds();
            (rect.width() > 200.0 && rect.height() > 50.0).then_some((rect, shape.clip_rect))
        })
        .expect("the large rendered stroke/corner sample at the bottom of the popup");
    assert!(popup.expand(0.5).contains_rect(sample), "bottom sample {sample:?} falls outside popup {popup:?}");
    assert!(clip.expand(0.5).contains_rect(sample), "bottom sample {sample:?} is clipped by the previous popup size {clip:?}");
}

#[test]
fn adding_dash_controls_grows_the_popup_and_solid_shrinks_it_again() {
    for theme in crate::theme::ThemeKind::ALL {
        let mut h = sized_harness(vec2(720.0, 1000.0), 2.0, theme);
        open(&mut h);
        let solid_height = h.state().popup.unwrap().height();
        assert_bottom_sample_visible(&h);
        let dotted = h.get_by_role_and_label(Role::RadioButton, "Dotted").rect();
        click_at(&mut h, pos2(dotted.right() - 16.0, dotted.center().y));
        let dashed_popup = h.state().popup.unwrap();
        assert!(dashed_popup.height() > solid_height + 60.0, "{theme:?}: dash controls must grow the popup beyond its old Solid size");
        for (role, label) in [(Role::Button, "Add pair"), (Role::Button, "Remove pair"), (Role::Slider, "Dash offset")] {
            let control = h.get_by_role_and_label(role, label).rect();
            assert!(dashed_popup.contains_rect(control), "{theme:?}: {label} is outside the grown popup: {control:?} in {dashed_popup:?}");
        }
        assert_bottom_sample_visible(&h);
        let add = h.get_by_role_and_label(Role::Button, "Add pair").rect();
        click_at(&mut h, add.center());
        assert_eq!(h.state().current.dashes.len(), 4);
        assert!(h.state().popup.unwrap().height() > dashed_popup.height() + 20.0, "{theme:?}: an additional dash pair also grows the content");
        assert_bottom_sample_visible(&h);
        let remove = h.get_by_role_and_label(Role::Button, "Remove pair").rect();
        click_at(&mut h, remove.center());
        assert_eq!(h.state().current.dashes.len(), 2);
        assert!((h.state().popup.unwrap().height() - dashed_popup.height()).abs() <= 1.0, "{theme:?}: removing a pair shrinks the popup");
        let solid = h.get_by_role_and_label(Role::RadioButton, "Solid").rect();
        click_at(&mut h, pos2(solid.right() - 16.0, solid.center().y));
        assert!(h.state().current.dashes.is_empty());
        assert!(h.query_by_label("Add pair").is_none());
        assert!((h.state().popup.unwrap().height() - solid_height).abs() <= 1.0, "{theme:?}: Solid restores the natural height without empty space");
        assert_bottom_sample_visible(&h);
    }
}

#[test]
fn options_serde_defaults_keep_old_tool_settings_and_full_stroke() {
    assert_eq!(serde_json::from_value::<StrokeOptions>(json!({})).unwrap(), StrokeOptions::default());
    let mut style = StrokeOptions::default();
    style.preset("Dotted");
    style.dashes = vec![0.0, 2.0, 4.0, 1.0];
    style.dash_offset = -0.5;
    let stroke = style.stroke(5.0, photocraft_doc::Fill::Solid(photocraft_doc::Color::BLACK));
    assert_eq!(StrokeOptions::from(&stroke), style);
    assert_eq!(serde_json::from_value::<StrokeOptions>(serde_json::to_value(&style).unwrap()).unwrap(), style);
}

#[test]
fn moving_the_editor_between_ui_roots_keeps_history_gestures_distinct() {
    let mut h = Harness::builder().with_size(vec2(640.0, 800.0)).build_ui_state(
        |ui, (root, state): &mut (u8, State)| {
            let _language = crate::i18n::language_scope(crate::i18n::Lang::EN);
            ui.push_id(*root, |ui| {
                if let Some(change) = button(ui, "same-layer", &state.current, &mut state.editor, true, true, None, true) {
                    state.current = change.options;
                    state.changes.push(change.coalesce);
                }
            });
        },
        (0, State::default()),
    );
    crate::PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ProMedium);
    h.run_steps(3);
    for (root, preset) in [(0, "Dashed"), (1, "Dotted")] {
        h.state_mut().0 = root;
        h.run_steps(3);
        h.get_by_role_and_label(Role::Button, "Stroke Options").click();
        h.run_steps(3);
        let row = h.get_by_role_and_label(Role::RadioButton, preset).rect();
        click_at(&mut h, pos2(row.right() - 16.0, row.center().y));
        h.key_press(Key::Escape);
        h.run_steps(3);
    }
    let changes = &h.state().1.changes;
    assert_eq!(changes.len(), 2);
    assert_ne!(changes[0], changes[1], "independent dock/floating counters must not merge edits of the same layer");
}

fn properties_harness(mut params: Value, depth: u32) -> Harness<'static, crate::PhotocraftApp> {
    let mut app = crate::PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
    app.run("file.new", json!({"width": 80, "height": 60, "depth": depth})).unwrap();
    params["fill"] = json!("#d92070");
    params["stroke"] = json!({"width": 5, "color": "#21bb66"});
    app.run("shape.create", params).unwrap();
    let mut h = Harness::builder().with_size(vec2(640.0, 800.0)).with_step_dt(1.0 / 60.0).build_ui_state(
        |ui, app: &mut crate::PhotocraftApp| {
            let _language = crate::i18n::language_scope(crate::i18n::Lang::EN);
            if !ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("semibold".into()))) {
                return;
            }
            if let Some(id) = app.session.active().and_then(|st| st.active_layer) {
                crate::vector_ui::shape_properties(app, ui, id);
            }
        },
        app,
    );
    crate::PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ProMedium);
    h.run_steps(4);
    h
}

fn selected_shape(h: &Harness<'_, crate::PhotocraftApp>) -> photocraft_doc::ShapeLayer {
    let st = h.state().session.active().unwrap();
    let layer = st.doc.layer(st.active_layer.unwrap()).unwrap();
    match &layer.content {
        photocraft_doc::LayerContent::Shape(shape) => shape.clone(),
        _ => panic!("expected the selected vector shape"),
    }
}

#[test]
fn selected_shape_presets_edit_live_preserve_paint_and_undo_at_all_depths() {
    for depth in [8, 16, 32] {
        for params in [
            json!({"kind": "rect", "rect": [12, 12, 45, 30]}),
            json!({"kind": "polygon", "sides": 3, "rect": [12, 12, 45, 30]}),
            json!({"kind": "ellipse", "rect": [12, 12, 45, 30]}),
            json!({"kind": "path", "path": {"subpaths": [{"closed": false, "knots": [[12, 12], [35, 42], [57, 12]]}]}}),
        ] {
            let mut h = properties_harness(params, depth);
            let original = selected_shape(&h);
            let before = h.state().session.active().unwrap().history.past_len();
            let opener = h.get_by_role_and_label(Role::Button, "Stroke Options").rect();
            click_at(&mut h, opener.center());
            assert_eq!(h.state().session.active().unwrap().history.past_len(), before);
            let dotted = h.get_by_role_and_label(Role::RadioButton, "Dotted").rect();
            click_at(&mut h, pos2(dotted.right() - 14.0, dotted.center().y));
            let edited = selected_shape(&h);
            assert_eq!(edited.path, original.path);
            assert_eq!(edited.live, original.live);
            assert_eq!(edited.fill, original.fill);
            let stroke = edited.stroke.as_ref().unwrap();
            assert_eq!(stroke.paint, original.stroke.as_ref().unwrap().paint);
            assert_eq!(stroke.width, original.stroke.as_ref().unwrap().width);
            assert_eq!(stroke.dashes, [0.0, 2.0]);
            assert_eq!(stroke.cap, LineCap::Round);
            assert_eq!(h.state().session.active().unwrap().history.past_len(), before + 1);
            h.key_press(Key::Escape);
            h.run_steps(3);
            assert!(h.state().ui.stroke_editor.is_none());
            assert_eq!(selected_shape(&h), edited, "Escape retains the directly applied style");
            assert_eq!(h.state().session.active().unwrap().history.past_len(), before + 1);
            h.state_mut().run("edit.undo", json!({})).unwrap();
            assert_eq!(selected_shape(&h), original);
            h.state_mut().run("edit.redo", json!({})).unwrap();
            assert_eq!(selected_shape(&h), edited);
        }
    }
}

fn opacity_drag(h: &mut Harness<'_, crate::PhotocraftApp>, offsets: &[f32], expected_history: usize) -> f32 {
    let start = h.get_by_role_and_label(Role::Slider, "Opacity").rect().center();
    h.hover_at(start);
    h.run_steps(1);
    h.event(egui::Event::PointerButton { pos: start, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    h.run_steps(1);
    let initial = selected_shape(h).stroke.unwrap().opacity;
    let mut end = start;
    for &offset in offsets {
        end = start + vec2(offset, 0.0);
        h.hover_at(end);
        h.run_steps(2);
    }
    let live = selected_shape(h).stroke.unwrap().opacity;
    assert!(live < initial, "the shape is updated while the pointer is still held: {initial} -> {live}");
    assert_eq!(h.state().session.active().unwrap().history.past_len(), expected_history, "all updates in one drag coalesce");
    h.event(egui::Event::PointerButton { pos: end, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
    h.run_steps(3);
    assert_eq!(selected_shape(h).stroke.unwrap().opacity, live);
    assert_eq!(h.state().session.active().unwrap().history.past_len(), expected_history, "release does not add a second edit");
    live
}

#[test]
fn selected_shape_numeric_drags_are_live_and_undo_as_separate_gestures() {
    let mut h = properties_harness(json!({"kind": "rect", "rect": [12, 12, 45, 30]}), 16);
    let before = h.state().session.active().unwrap().history.past_len();
    let original = selected_shape(&h);
    let opener = h.get_by_role_and_label(Role::Button, "Stroke Options").rect();
    click_at(&mut h, opener.center());
    let first = opacity_drag(&mut h, &[-8.0, -16.0, -28.0], before + 1);
    let second = opacity_drag(&mut h, &[-8.0, -18.0, -32.0], before + 2);
    assert!(second < first);
    h.key_press(Key::Escape);
    h.run_steps(3);
    assert_eq!(selected_shape(&h).stroke.unwrap().opacity, second);
    h.state_mut().run("edit.undo", json!({})).unwrap();
    assert_eq!(selected_shape(&h).stroke.unwrap().opacity, first);
    h.state_mut().run("edit.undo", json!({})).unwrap();
    assert_eq!(selected_shape(&h), original);
    h.state_mut().run("edit.redo", json!({})).unwrap();
    assert_eq!(selected_shape(&h).stroke.unwrap().opacity, first);
    h.state_mut().run("edit.redo", json!({})).unwrap();
    assert_eq!(selected_shape(&h).stroke.unwrap().opacity, second);
}

struct CanvasScene {
    app: crate::PhotocraftApp,
    properties: bool,
    show_controls: bool,
}

fn canvas_harness(properties: bool) -> Harness<'static, CanvasScene> {
    let mut app = crate::PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
    app.run("file.new", json!({"width": 800, "height": 600})).unwrap();
    app.ui.tool = crate::state::Tool::Rectangle;
    app.ui.tool_options.vector_mode = "shape".into();
    app.ui.tool_options.stroke_width = 3.0;
    if properties {
        app.run("shape.create", json!({"rect": [20, 20, 80, 60], "stroke": {"width": 3, "color": "#1188dd"}})).unwrap();
    }
    app.sync_views();
    let mut h = Harness::builder().with_size(vec2(1000.0, 750.0)).with_max_steps(64).build_ui_state(
        |ui, scene: &mut CanvasScene| {
            let _language = crate::i18n::language_scope(crate::i18n::Lang::EN);
            if !ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            let app = &mut scene.app;
            // Use the native ordering: logic, controls, live canvas input, then ownership cleanup.
            sync_editor(app, ui.ctx());
            crate::shortcuts::handle(app, ui.ctx());
            if scene.show_controls {
                if scene.properties {
                    ui.scope(|ui| {
                        ui.set_max_width(320.0);
                        let layer = app.session.active().unwrap().active_layer.unwrap();
                        crate::vector_ui::shape_properties(app, ui, layer);
                    });
                } else {
                    let tool = app.ui.tool;
                    ui.horizontal(|ui| crate::vector_ui::options_bar(app, ui, tool));
                }
            }
            let rect = Rect::from_min_max(pos2(350.0, 110.0), pos2(990.0, 740.0));
            app.last_canvas_rect = rect;
            let view = app.ui.views[0].clone();
            crate::canvas::canvas_view(app, ui, 0, rect, view, true);
            finish_frame(app, ui.ctx());
        },
        CanvasScene { app, properties, show_controls: true },
    );
    crate::PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ProMedium);
    h.run_steps(4);
    h
}

fn scene_open(h: &mut Harness<'_, CanvasScene>) {
    let opener = h.get_by_role_and_label(Role::Button, "Stroke Options").rect();
    click_at(h, opener.center());
    assert!(h.state().app.ui.stroke_editor.is_some());
}

fn canvas_press(h: &mut Harness<'_, CanvasScene>, pos: Pos2, pressed: bool) {
    h.event(egui::Event::PointerButton { pos, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE });
    h.run_steps(1);
}

#[test]
fn canvas_dismissal_gestures_neither_draw_shapes_nor_open_creation_dialogs() {
    for properties in [false, true] {
        let mut h = canvas_harness(properties);
        let original = h.state().app.session.active().unwrap().doc.clone();
        let history = h.state().app.session.active().unwrap().history.past_len();
        let start = pos2(820.0, 550.0);
        let end = pos2(880.0, 610.0);
        // A press and release delivered in one frame must remain dismissal-only even though
        // the options row has already dismissed its popup before the canvas handles the click.
        scene_open(&mut h);
        h.input_mut().events.extend([
            egui::Event::PointerMoved(start),
            egui::Event::PointerButton { pos: start, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE },
            egui::Event::PointerButton { pos: start, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE },
        ]);
        h.run_steps(3);
        assert!(h.state().app.ui.stroke_editor.is_none());
        assert!(h.state().app.ui.dialogs.is_empty(), "the dismissal click must not open Create Rectangle");
        assert_eq!(h.state().app.session.active().unwrap().doc, original);
        scene_open(&mut h);
        h.hover_at(start);
        h.run_steps(1);
        canvas_press(&mut h, start, true);
        assert!(h.state().app.ui.stroke_editor.is_none(), "the canvas press dismisses the popup immediately");
        for pos in [start + vec2(12.0, 12.0), end] {
            h.hover_at(pos);
            h.run_steps(2);
            assert!(h.state().app.drag.is_none(), "no shape preview starts behind the dismissed popup");
        }
        canvas_press(&mut h, end, false);
        h.run_steps(3);
        assert!(h.state().app.ui.dialogs.is_empty());
        assert_eq!(h.state().app.session.active().unwrap().doc, original);
        assert_eq!(h.state().app.session.active().unwrap().history.past_len(), history);
        // A new, deliberate gesture works normally once the dismissal pointer is released.
        h.hover_at(start);
        h.run_steps(1);
        canvas_press(&mut h, start, true);
        h.hover_at(end);
        h.run_steps(2);
        canvas_press(&mut h, end, false);
        h.run_steps(3);
        assert_eq!(h.state().app.session.active().unwrap().history.past_len(), history + 1);
        assert!(h.state().app.drag.is_none());
    }
}

#[test]
fn hidden_controls_and_collapsed_appearance_release_popup_and_keyboard_ownership() {
    for collapsed in [false, true] {
        let mut h = canvas_harness(true);
        let original = h.state().app.session.active().unwrap().doc.clone();
        scene_open(&mut h);
        h.get_by_role_and_label(Role::ComboBox, "Alignment").click();
        h.run_steps(3);
        assert!(egui::Popup::is_any_open(&h.ctx));
        if collapsed {
            h.ctx.data_mut(|d| d.insert_temp(egui::Id::new(("props-section", "appearance")), false));
        } else {
            h.state_mut().show_controls = false;
        }
        h.run_steps(1);
        assert!(h.state().app.ui.stroke_editor.is_none(), "controls that were not drawn no longer own the keyboard");
        assert!(!egui::Popup::is_any_open(&h.ctx), "their child popup must be cleaned up too");
        assert_eq!(h.state().app.session.active().unwrap().doc, original);
        h.key_press(Key::B);
        h.run_steps(3);
        assert_eq!(h.state().app.ui.tool, crate::state::Tool::Brush, "tool shortcuts resume after the hidden popup closes");
    }
}
