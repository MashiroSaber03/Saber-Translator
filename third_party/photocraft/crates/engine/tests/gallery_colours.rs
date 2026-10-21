//! Regression tests for issue #709: the single-filter gallery commands must
//! record the session foreground/background colours for every filter that
//! reads them. Colored Pencil (background) and Neon Glow (both) read the
//! colours in their pixel code but omit them from their params notation, so
//! the injection was skipped and they always ran with the defaults.
use photocraft_engine::Session;
use serde_json::json;

fn px(s: &mut Session, x: i32, y: i32) -> Vec<f32> {
    serde_json::from_value(s.execute("document.pixel", json!({"x": x, "y": y})).unwrap()).unwrap()
}

fn image(s: &mut Session) -> Vec<f32> {
    (0..16).flat_map(|y| (0..16).map(move |x| (x, y))).flat_map(|(x, y)| px(s, x, y)).collect()
}

fn one(filter: &str, bg: &str) -> Vec<f32> {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 16, "height": 16, "background": "#808080"})).unwrap();
    s.execute("tools.setColors", json!({"background": bg})).unwrap();
    s.execute(&format!("filter.gallery.{filter}"), json!({})).unwrap();
    image(&mut s)
}

#[test]
fn colored_pencil_honours_session_background() {
    assert_ne!(one("coloredPencil", "#ffffff"), one("coloredPencil", "#0000ff"), "filter.gallery.coloredPencil must honour the session background");
}

#[test]
fn neon_glow_honours_session_background() {
    assert_ne!(one("neonGlow", "#ffffff"), one("neonGlow", "#0000ff"), "filter.gallery.neonGlow must honour the session background");
}
