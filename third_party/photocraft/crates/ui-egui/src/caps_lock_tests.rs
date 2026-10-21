//! Caps Lock swaps the painting tools' cursor to the precise crosshair, exactly like Photoshop,
//! whatever the Cursors › Painting preference is (#1758). The override changes only painting
//! tools (Brush, Pencil, Clone Stamp, Healing, Eraser, …): Move, Marquee, Crop and Zoom keep
//! their own cursors. Tests drive the real canvas (`document_area`) and compare the painted
//! cursor that caps lock produces with the one the Precise preference draws — they must match,
//! and the Standard preference (off) must not. The tools under Cursors › Other Cursors with a
//! cursor of their own (the pens, Eyedropper, Color Sampler, Paint Bucket) swap it for the
//! crosshair and back with caps lock (#4639).

use egui::CursorIcon;
use egui_kittest::Harness;
use serde_json::json;

use crate::PhotocraftApp;
use crate::state::Tool;

fn capslock_harness() -> Harness<'static, PhotocraftApp> {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 400, "height": 300})).unwrap();
    app.run("tools.setBrush", json!({"brush": {"size": 40, "hardness": 1.0}})).unwrap();
    app.sync_views();
    app.ui.extras.rulers = false;
    app.ui.tool = Tool::Brush;
    let mut h = Harness::builder().with_size(egui::vec2(900.0, 600.0)).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            let ctx = ui.ctx().clone();
            if !ctx.fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            egui::CentralPanel::default().show(ui, |ui| crate::canvas::document_area(app, ui));
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
    h.run_steps(4);
    // 100 %: one document pixel per point; the pointer hovers the canvas centre.
    let v = &mut h.state_mut().ui.views[0];
    v.zoom = 1.0;
    v.center = [200.0, 150.0];
    v.fit_pending = false;
    h.run_steps(2);
    let p = h.state().last_canvas_rect.center();
    h.event(egui::Event::PointerMoved(p));
    h.run_steps(3);
    h
}

/// Everything the cursor path can change in one frame: the cursor icon, the OS cursor bitmap
/// (Windows) and the shapes painted on the canvas (the tool-cursor path paints the crosshair
/// and the brush tip outline). Comparison across two preference/caps-lock states of the same
/// harness is platform-independent.
#[derive(Debug, PartialEq)]
struct CursorFp {
    icon: CursorIcon,
    image: Option<[u16; 2]>,
    shapes: Vec<egui::epaint::ClippedShape>,
}

fn fp(h: &Harness<'_, PhotocraftApp>) -> CursorFp {
    let out = h.output();
    CursorFp { icon: out.platform_output.cursor_icon, image: out.platform_output.cursor_image.as_ref().map(|i| i.size), shapes: out.shapes.clone() }
}

fn painting(h: &mut Harness<'_, PhotocraftApp>, value: &str) {
    h.state_mut().run("prefs.set", json!({"path": "cursors.painting", "value": value})).unwrap();
    h.run_steps(2);
}

/// Caps Lock is read from the OS by the app shell; tests toggle the per-frame flag directly.
fn caps(h: &mut Harness<'_, PhotocraftApp>, on: bool) {
    h.state_mut().caps_lock = on;
    h.run_steps(2);
}

#[test]
fn caps_lock_forces_the_precise_crosshair_for_painting_tools_whatever_the_preference() {
    for tool in [
        Tool::Brush,
        Tool::Pencil,
        Tool::CloneStamp,
        Tool::Healing,
        Tool::Eraser,
        // The block that honours the painting preference also covers the Quick Selection.
        Tool::QuickSelection,
    ] {
        let mut h = capslock_harness();
        // Each tool keeps its own brush (#218), so switch the tool before painting.
        h.state_mut().ui.tool = tool;
        h.run_steps(2);
        painting(&mut h, "standard");
        let off = fp(&h);
        painting(&mut h, "precise");
        let precise = fp(&h);
        painting(&mut h, "standard");
        caps(&mut h, true);
        let on = fp(&h);
        assert_ne!(off, precise, "{tool:?}: the Standard and Precise cursors must differ");
        assert_eq!(on, precise, "{tool:?}: caps lock must draw the same crosshair as the Precise preference");
        assert_ne!(on, off, "{tool:?}: caps lock must not keep the Standard cursor");
        caps(&mut h, false);
        assert_eq!(fp(&h), off, "{tool:?}: releasing caps lock restores the preference cursor");
    }
}

#[test]
fn caps_lock_leaves_non_painting_tools_cursors_alone() {
    for tool in [Tool::Move, Tool::RectMarquee, Tool::EllipseMarquee, Tool::Crop, Tool::Zoom, Tool::Hand] {
        let mut h = capslock_harness();
        h.state_mut().ui.tool = tool;
        h.run_steps(2);
        let off = fp(&h);
        caps(&mut h, true);
        let on = fp(&h);
        assert_eq!(on, off, "{tool:?}: caps lock must not change a non-painting cursor");
        caps(&mut h, false);
    }
}

fn other(h: &mut Harness<'_, PhotocraftApp>, value: &str) {
    h.state_mut().run("prefs.set", json!({"path": "cursors.other", "value": value})).unwrap();
    h.run_steps(2);
}

/// #4639: the pen tools show their own icon under Standard Other Cursors and the crosshair
/// under Precise, and caps lock swaps one for the other either way, as in Photoshop. The
/// Eyedropper, Color Sampler and Paint Bucket, which already followed the preference, swap too.
#[test]
fn caps_lock_swaps_the_other_cursors_between_their_icon_and_the_crosshair() {
    for tool in [
        Tool::Pen,
        Tool::FreeformPen,
        Tool::CurvaturePen,
        Tool::AddAnchorPoint,
        Tool::DeleteAnchorPoint,
        Tool::ConvertPoint,
        Tool::Eyedropper,
        Tool::ColorSampler,
        Tool::PaintBucket,
    ] {
        let mut h = capslock_harness();
        h.state_mut().ui.tool = tool;
        h.run_steps(2);
        other(&mut h, "standard");
        let standard = fp(&h);
        assert_eq!(standard.icon, CursorIcon::None, "{tool:?}: the standard cursor is the tool's own icon, drawn");
        other(&mut h, "precise");
        let precise = fp(&h);
        assert_ne!(standard, precise, "{tool:?}: Standard and Precise differ");
        caps(&mut h, true);
        assert_eq!(fp(&h), standard, "{tool:?}: caps lock turns the precise crosshair back into the icon");
        other(&mut h, "standard");
        assert_eq!(fp(&h), precise, "{tool:?}: caps lock turns the icon into the precise crosshair");
        caps(&mut h, false);
        assert_eq!(fp(&h), standard, "{tool:?}: releasing caps lock restores the icon");
    }
}
