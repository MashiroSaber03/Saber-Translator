//! Clipping marker orientation and the Alt-click gesture that reveals it.

use egui::{Event, Modifiers, PointerButton, pos2, vec2};
use egui_kittest::Harness;
use serde_json::json;

use crate::PhotocraftApp;

#[test]
fn clipping_indicator_points_down_inside_the_existing_row_slot() {
    let ctx = egui::Context::default();
    let at = pos2(100.0, 100.0);
    let mut output = ctx.run_ui(Default::default(), |ui| super::paint_clipping_indicator(ui.painter(), at, egui::Color32::LIGHT_GRAY));
    output.textures_delta.clear();
    let paths: Vec<_> = output.shapes.iter().filter_map(|shape| if let egui::Shape::Path(path) = &shape.shape { Some(path) } else { None }).collect();
    assert_eq!(paths.len(), 2, "the marker has a bent shaft and an arrowhead, independent of font glyphs");
    let head = paths.iter().find(|path| path.points.len() == 3 && path.points[0].y == path.points[2].y).unwrap();
    assert!(head.points[1].y > head.points[0].y && head.points[1].y > head.points[2].y, "the arrow points down toward the clipping base");
    let slot = egui::Rect::from_min_max(at + vec2(0.0, -5.0), at + vec2(12.0, 5.0));
    assert!(paths.iter().flat_map(|path| &path.points).all(|point| slot.contains(*point)), "the marker does not shift or overlap the thumbnail");
}

fn harness(theme: crate::theme::ThemeKind, width: f32, scale: f32, gpu: bool) -> Harness<'static, PhotocraftApp> {
    let mut session = photocraft_engine::Session::new();
    session.execute("prefs.set", json!({"path": "interface.language", "value": "en"})).unwrap();
    session.execute("file.new", json!({"width": 64, "height": 64, "background": "white"})).unwrap();
    let upper = session.execute("layer.newFillLayer.gradient", json!({"from": "#235b49", "to": "#a7d4ab", "angle": 90})).unwrap()["layer"].as_u64().unwrap();
    session.execute("layer.setProps", json!({"layer": upper, "name": "Gradient Fill 1"})).unwrap();
    let builder = Harness::builder().with_size(vec2(width, 250.0)).with_pixels_per_point(scale);
    let builder = if gpu { builder.wgpu() } else { builder };
    let mut h = builder.build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            if !ui.ctx().fonts(|fonts| fonts.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            crate::i18n::with_language(crate::i18n::Lang::EN, || {
                app.ppp = ui.ctx().pixels_per_point();
                ui.heading("Layers");
                super::layers(app, ui);
            });
        },
        PhotocraftApp::new(session, Default::default()),
    );
    PhotocraftApp::setup_context(&h.ctx, theme);
    h.run_steps(4);
    h
}

fn alt_click_line(h: &mut Harness<'_, PhotocraftApp>) {
    let rows = crate::layer_row_ui::recorded(&h.ctx);
    assert_eq!(rows.len(), 2);
    let p = pos2(rows[1].row.left() + 100.0, rows[1].row.top());
    h.event(Event::PointerMoved(p));
    h.event(Event::ModifiersChanged(Modifiers::ALT));
    h.step();
    for pressed in [true, false] {
        h.event(Event::PointerButton { pos: p, button: PointerButton::Primary, pressed, modifiers: Modifiers::ALT });
        h.step();
    }
    h.event(Event::ModifiersChanged(Modifiers::NONE));
    h.event(Event::PointerMoved(pos2(0.0, 0.0)));
    h.run_steps(2);
}

#[test]
fn alt_click_clipping_still_creates_releases_and_undoes_without_moving_rows() {
    let mut h = harness(crate::theme::ThemeKind::default(), 420.0, 1.0, false);
    let active = h.state().session.active().unwrap().active_layer.unwrap();
    let before = crate::layer_row_ui::recorded(&h.ctx);
    alt_click_line(&mut h);
    assert!(h.state().session.active().unwrap().doc.layer(active).unwrap().clipped);
    assert_eq!(h.state().session.active().unwrap().active_layer, Some(active));
    let after = crate::layer_row_ui::recorded(&h.ctx);
    assert!(before.iter().zip(&after).all(|(before, after)| before.row == after.row));
    h.state_mut().run("edit.undo", json!({})).unwrap();
    assert!(!h.state().session.active().unwrap().doc.layer(active).unwrap().clipped);
    h.state_mut().run("edit.redo", json!({})).unwrap();
    assert!(h.state().session.active().unwrap().doc.layer(active).unwrap().clipped);
    alt_click_line(&mut h);
    assert!(!h.state().session.active().unwrap().doc.layer(active).unwrap().clipped);
}

#[test]
#[ignore = "writes synthetic English Layers-panel screenshots for before/after inspection"]
fn write_clipping_indicator_preview() {
    let output = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../plan/evidence/clipping");
    std::fs::create_dir_all(&output).unwrap();
    for theme in crate::theme::ThemeKind::ALL {
        for width in [310.0, 420.0] {
            for scale in [1.0, 2.0] {
                let mut h = harness(theme, width, scale, true);
                alt_click_line(&mut h);
                h.ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
                h.step();
                h.step();
                let image = h
                    .ctx
                    .input(|input| input.events.iter().find_map(|event| if let Event::Screenshot { image, .. } = event { Some(image.clone()) } else { None }))
                    .unwrap();
                let rgba = image.pixels.iter().flat_map(|pixel| pixel.to_array()).collect();
                let image =
                    photocraft_codecs::Image::from_u8(image.size[0] as u32, image.size[1] as u32, photocraft_codecs::ChannelLayout::Rgba, rgba).unwrap();
                let png = photocraft_codecs::encode(&image, photocraft_codecs::Format::Png, &Default::default()).unwrap();
                std::fs::write(output.join(format!("{}-{width:.0}-{scale:.0}x.png", theme.id())), png).unwrap();
            }
        }
    }
}
