//! Colour strings supplied through the same commands used by CLI/MCP clients.
use photocraft_engine::{Session, command_specs};
use serde_json::{Value, json};

fn run(id: &str, params: Value) {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
    let spec = command_specs().iter().find(|c| c.id == id).unwrap();
    // Exercise the command directly, without the last-resort shell guard.
    (spec.run)(&mut s, &params).unwrap();
}

#[test]
fn unicode_pixelate_colour_uses_the_existing_fallback() {
    run("filter.pixelate.pointillize", json!({"background": "#€€"}));
}

#[test]
fn unicode_gallery_colour_uses_the_existing_fallback() {
    run("filter.gallery.photocopy", json!({"foreground": "#€€"}));
}

#[test]
fn unicode_render_colour_uses_the_existing_fallback() {
    run("filter.render.tree", json!({"leavesColor": "#€€"}));
}

#[test]
fn unicode_preference_colour_is_invalid() {
    assert_eq!(photocraft_engine::prefs::parse_hex("#€€"), None);
    assert_eq!(photocraft_engine::prefs::parse_hex(""), None);
    assert_eq!(photocraft_engine::prefs::parse_hex("#12aBcD"), Some([0x12, 0xab, 0xcd]));
}
