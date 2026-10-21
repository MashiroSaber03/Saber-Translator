//! Colour swatches open PhotoCraft's Color Picker, not egui's compact popup (#2144: "use the same
//! color picker throughout the app"): filter dialog colours, Preferences colours, the Note and
//! Count tools' colours and an artboard's custom background. OK sets that colour (in sRGB, as
//! everywhere else); the foreground is untouched.

use egui::accesskit::Role;
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::state::Tool;

const PICKED: &str = "#336699";
const PICKED_RGB: [f32; 3] = [0x33 as f32 / 255.0, 0x66 as f32 / 255.0, 0x99 as f32 / 255.0];

fn harness(setup: impl FnOnce(&mut PhotocraftApp) + 'static) -> Harness<'static, PhotocraftApp> {
    let mut h = Harness::builder().with_size(egui::vec2(1400.0, 1800.0)).with_max_steps(64).build_eframe(|cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 200, "height": 120})).unwrap();
        setup(&mut app);
        app
    });
    h.run_steps(6);
    h
}

/// Clicks the swatch called `label` (a colour button), answers the Color Picker it opens with
/// [`PICKED`], and returns the picker's title. Panics when no Color Picker opens.
fn pick(h: &mut Harness<'_, PhotocraftApp>, label: &str) -> String {
    // Panels scroll: bring the swatch into view first.
    h.get_by_role_and_label(Role::ColorWell, label).scroll_to_me();
    h.run_steps(2);
    h.get_by_role_and_label(Role::ColorWell, label).click();
    answer(h, label)
}

/// Answers the Color Picker a swatch click just opened with [`PICKED`]; returns its title.
fn answer(h: &mut Harness<'_, PhotocraftApp>, label: &str) -> String {
    let fg = h.state().session.tools.foreground;
    h.run_steps(4);
    let id = crate::color_picker_ui::top(h.state()).unwrap_or_else(|| panic!("the {label} swatch opens the Color Picker"));
    assert!(!egui::Popup::is_any_open(&h.ctx), "{label}: no egui colour popup");
    let title = h.state().ui.dialogs.iter().find(|d| d.id == id).and_then(|d| d.fields.get("__label")).and_then(Value::as_str).unwrap_or("").to_string();
    h.state_mut().ui.dialog_mut(id).unwrap().fields.insert("color".into(), json!(PICKED));
    crate::dialogs::confirm(h.state_mut(), id).unwrap();
    h.run_steps(3);
    assert!(crate::color_picker_ui::top(h.state()).is_none(), "{label}: OK closes the picker");
    assert_eq!(h.state().session.tools.foreground, fg, "{label}: the foreground is untouched");
    title
}

fn close(a: [f32; 3], b: [f32; 3]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1.0 / 255.0)
}

#[test]
fn filter_dialog_colours_open_the_color_picker() {
    for (cmd, key, label) in [("filter.other.colorToAlpha", "color", "Color"), ("filter.render.pictureFrame", "vineColor", "Vine Color")] {
        let mut h = harness(|_| {});
        let ctx = h.ctx.clone();
        let id = crate::menus::invoke(h.state_mut(), &ctx, cmd, json!({})).unwrap()["dialog"].as_u64().unwrap();
        h.run_steps(4);
        let title = pick(&mut h, label);
        assert_eq!(title, format!("Color Picker ({label})"), "{cmd}");
        let d = h.state().ui.dialogs.iter().find(|d| d.id == id).expect("the filter dialog stays open");
        assert_eq!(d.fields.get(key), Some(&json!(PICKED)), "{cmd}: {key}");
    }
}

#[test]
fn preferences_colours_open_the_color_picker() {
    let mut h = harness(|_| {});
    let id = crate::prefs_ui::open_preferences(h.state_mut(), "guidesGridAndSlices");
    h.run_steps(4);
    let values = |h: &Harness<'_, PhotocraftApp>| h.state().ui.dialogs.iter().find(|d| d.id == id).unwrap().fields["values"]["guidesGridAndSlices"].clone();
    let before = values(&h);
    h.get_all_by_role(Role::ColorWell).next().expect("a colour swatch").click();
    answer(&mut h, "the first Preferences colour");
    let after = values(&h);
    let changed: Vec<&str> = after.as_object().unwrap().iter().filter(|(k, v)| before.get(k.as_str()) != Some(v)).map(|(k, _)| k.as_str()).collect();
    assert_eq!(changed.len(), 1, "one colour changed: {changed:?}");
    assert_eq!(after[changed[0]], json!(PICKED));
}

#[test]
fn note_and_count_colours_open_the_color_picker() {
    let mut h = harness(|app| app.ui.tool = Tool::Note);
    pick(&mut h, "Color");
    assert!(close(h.state().ui.analysis.note_color, PICKED_RGB), "note colour {:?}", h.state().ui.analysis.note_color);

    let mut h = harness(|app| {
        app.ui.tool = Tool::Count;
        app.run("count.newGroup", json!({})).unwrap();
    });
    pick(&mut h, "Color");
    let st = h.state().session.active().unwrap();
    let m = &st.doc.measurement;
    let g = &m.count_groups[m.active_count_group];
    assert!(close(g.color.to_rgb(), PICKED_RGB), "count group colour {:?}", g.color.to_rgb());
}

#[test]
fn an_artboard_background_opens_the_color_picker() {
    let mut h = harness(|app| {
        app.run("layer.new.artboard", json!({"x": 10, "y": 10, "width": 100, "height": 80})).unwrap();
        app.run("layer.artboard.set", json!({"background": "custom", "color": "#ff0000"})).unwrap();
    });
    pick(&mut h, "Color");
    let st = h.state().session.active().unwrap();
    let l = st.doc.layer(st.active_layer.unwrap()).unwrap();
    let photocraft_doc::LayerContent::Group(g) = &l.content else { panic!("an artboard") };
    let bg = g.artboard.as_ref().expect("an artboard").background;
    let photocraft_doc::ArtboardBackground::Custom(c) = bg else { panic!("custom background: {bg:?}") };
    assert!(close(c.to_rgb(), PICKED_RGB), "artboard background {:?}", c.to_rgb());
}
