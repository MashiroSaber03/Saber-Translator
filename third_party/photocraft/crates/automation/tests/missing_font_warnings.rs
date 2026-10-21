//! #3391: `doc.save`, `doc.open`, and type commands emit warnings when text uses a missing font.

use photocraft_automation::Headless;
use serde_json::json;

fn tmp_dir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "pc-font-warn-{}-{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn missing_font_warnings_emitted_in_headless_and_type_commands() {
    let dir = tmp_dir();
    let psd_path = dir.join("card.psd");
    let png_path = dir.join("card.png");

    let mut h = Headless::trusted_local();
    h.handle("doc.new", json!({"width": 600, "height": 300, "background": "white"})).unwrap();

    // 1. type.create with missing font
    let created = h
        .handle("engine.execute", json!({"command": "type.create", "params": {"x": 30, "y": 100, "text": "Headline", "font": "NoSuchFamily123", "size": 40}}))
        .unwrap();
    assert_eq!(created["warnings"], json!(["font 'NoSuchFamily123' is not installed; text may render using a fallback face"]));

    // 2. doc.save card.psd
    let saved_psd = h.handle("doc.save", json!({"path": psd_path.to_str().unwrap()})).unwrap();
    let warnings = saved_psd["warnings"].as_array().expect("warnings array in doc.save");
    assert!(
        warnings.iter().any(|w| w == "font 'NoSuchFamily123' is not installed; text may render using a fallback face"),
        "doc.save PSD warning: {saved_psd}"
    );

    // 3. doc.save card.png
    let saved_png = h.handle("doc.save", json!({"path": png_path.to_str().unwrap()})).unwrap();
    let warnings = saved_png["warnings"].as_array().expect("warnings array in doc.save");
    assert!(
        warnings.iter().any(|w| w == "font 'NoSuchFamily123' is not installed; text may render using a fallback face"),
        "doc.save PNG warning: {saved_png}"
    );

    // 4. doc.close then doc.open
    h.handle("doc.close", json!({})).unwrap();
    let opened = h.handle("doc.open", json!({"path": psd_path.to_str().unwrap()})).unwrap();
    let warnings = opened["warnings"].as_array().expect("warnings array in doc.open");
    assert!(
        warnings.iter().any(|w| w.as_str().unwrap_or("").contains("is not installed; text may render using a fallback face")),
        "doc.open warning: {opened}"
    );

    let _ = std::fs::remove_dir_all(dir);
}
