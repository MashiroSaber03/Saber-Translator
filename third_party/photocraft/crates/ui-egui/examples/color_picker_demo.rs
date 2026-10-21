//! Render synthetic color-picker evidence without capturing the user's desktop.
//! cargo run -p photocraft-ui-egui --example color_picker_demo -- /tmp/color-picker
use egui_kittest::{Harness, kittest::Queryable};
use photocraft_ui_egui::{
    PhotocraftApp, Services,
    screen_picker::{self, Capture, Pending, ScreenImage},
};
use serde_json::json;

fn main() {
    let output = std::env::args().nth(1).unwrap_or_else(|| "/tmp/photocraft-color-picker".into());
    std::fs::create_dir_all(&output).expect("output directory");
    let mut session = photocraft_engine::Session::new();
    session.execute("file.new", json!({"width":800,"height":500,"background":"white"})).unwrap();
    session.execute("type.create", json!({"text":"Color from anywhere","color":"#2070c0","size":48,"x":90,"y":200})).unwrap();
    let mut text = Harness::builder().with_size(egui::vec2(1200.0, 800.0)).with_pixels_per_point(1.0).with_max_steps(64).wgpu().build_eframe(move |cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        let services = Services {
            screen_pick: Some(Box::new(|_| {
                let (tx, receiver) = std::sync::mpsc::channel();
                tx.send(Ok(Capture::Color(Some([1.0, 136.0 / 255.0, 0.0])))).unwrap();
                Pending { receiver, cancelled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)) }
            })),
            ..Default::default()
        };
        let mut app = PhotocraftApp::new(session, services);
        app.ui.tool = photocraft_ui_egui::state::Tool::Type;
        app
    });
    text.run_steps(8);
    text.get_by_label("Set the text color").click();
    text.run_steps(6);
    text.render().unwrap().save(format!("{output}/text-color-picker.png")).unwrap();
    text.get_by_label("Pick screen color").click();
    text.run_steps(6);
    text.render().unwrap().save(format!("{output}/text-color-picked.png")).unwrap();

    let services = Services {
        screen_pick: Some(Box::new(|_| {
            let mut rgba = Vec::with_capacity(800 * 600 * 4);
            for y in 0..600 {
                for x in 0..800 {
                    rgba.extend_from_slice(&[(x * 255 / 799) as u8, (y * 255 / 599) as u8, 128, 255]);
                }
            }
            let (tx, receiver) = std::sync::mpsc::channel();
            tx.send(Ok(Capture::Images(vec![ScreenImage {
                monitor: None,
                position: egui::Pos2::ZERO,
                size: egui::vec2(800.0, 600.0),
                width: 800,
                height: 600,
                rgba,
                profile: None,
            }])))
            .unwrap();
            Pending { receiver, cancelled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)) }
        })),
        ..Default::default()
    };
    let mut loupe = Harness::builder().with_size(egui::vec2(800.0, 600.0)).with_pixels_per_point(1.0).wgpu().build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            screen_picker::tick(app, ui.ctx());
            if screen_picker::showing(ui.ctx()) {
                screen_picker::show(ui.ctx());
            } else {
                screen_picker::button(ui, egui::Id::new("demo"));
            }
        },
        PhotocraftApp::new(photocraft_engine::Session::new(), services),
    );
    loupe.run_steps(2);
    loupe.get_by_label("Pick screen color").click();
    loupe.run_steps(3);
    loupe.input_mut().events.push(egui::Event::PointerMoved(egui::pos2(360.0, 220.0)));
    loupe.run_steps(2);
    loupe.render().unwrap().save(format!("{output}/screen-color-loupe.png")).unwrap();
    screen_picker::cancel(&loupe.ctx);
    loupe.run_steps(1);
}
