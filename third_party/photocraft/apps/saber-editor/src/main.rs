use serde_json::{Value, json};
use std::io::{Read, Write};

fn process(request: Value) -> Result<Value, String> {
    let mut session = photocraft_engine::Session::new();
    let page = request.get("pageId").and_then(Value::as_str).ok_or("pageId required")?;
    let mut fonts = serde_json::Map::new();
    for font in request.get("fonts").and_then(Value::as_array).into_iter().flatten() {
        let path = font.get("path").and_then(Value::as_str).ok_or("font path required")?;
        let face = photocraft_saber::register_font(std::fs::read(path).map_err(|e| format!("字体读取失败：{e}"))?)?;
        fonts.insert(font.get("id").and_then(Value::as_str).ok_or("font id required")?.into(), face);
    }
    let read = |key: &str| -> Result<Vec<u8>, String> {
        std::fs::read(request.get(key).and_then(Value::as_str).ok_or_else(|| format!("{key} required"))?).map_err(|e| e.to_string())
    };
    if let Some(default) = request.get("defaultFontId").and_then(Value::as_str).and_then(|id| fonts.get(id)).cloned() {
        fonts.insert("__default".into(), default);
        fonts.insert("__defaultId".into(), request["defaultFontId"].clone());
    }
    if request.get("project").and_then(Value::as_str).is_some() {
        photocraft_saber::open(&mut session, &read("project")?, page)?;
    } else {
        let clean = if request.get("base").and_then(Value::as_str).is_some() { Some(read("base")?) } else { None };
        photocraft_saber::create(&mut session, &read("source")?, clean.as_deref(), page)?;
    }
    let fonts = Value::Object(fonts);
    photocraft_saber::set_fonts(&mut session, &fonts)?;
    if let Some(bubbles) = request.get("bubbles").and_then(Value::as_array) {
        photocraft_saber::set_bubbles(&mut session, bubbles, &fonts)?;
    }
    if request.get("userMask").is_some() {
        let mask = if request.get("userMask").is_some_and(Value::is_null) { None } else { Some(read("userMask")?) };
        photocraft_saber::user_mask(&mut session, mask.as_deref())?;
    }
    photocraft_saber::refresh(&mut session);
    if request.get("baseUpdate").is_some() {
        photocraft_saber::repair(&mut session, &read("baseUpdate")?)?;
    }
    if request.get("repair").is_some() {
        photocraft_saber::repair(&mut session, &read("repair")?)?;
    }
    if let Some(commands) = request.get("commands").and_then(Value::as_array) {
        for command in commands {
            session
                .execute(command.get("id").and_then(Value::as_str).ok_or("command id required")?, command.get("params").cloned().unwrap_or(json!({})))
                .map_err(|e| e.to_string())?;
        }
    }
    for (key, bytes) in [
        ("output", request.get("output").is_some().then(|| photocraft_saber::snapshot(&session))),
        ("png", request.get("png").is_some().then(|| photocraft_saber::png(&session, false))),
        ("clean", request.get("clean").is_some().then(|| photocraft_saber::png(&session, true))),
        ("text", request.get("text").is_some().then(|| photocraft_saber::text_png(&session))),
    ] {
        if let Some(bytes) = bytes {
            let bytes = bytes?;
            let path = request.get(key).and_then(Value::as_str).ok_or("output path required")?;
            std::fs::write(path, bytes).map_err(|e| e.to_string())?;
        }
    }
    photocraft_saber::project(&session)
}

fn main() {
    let result = (|| {
        let mut input = String::new();
        std::io::stdin().take(32 * 1024 * 1024).read_to_string(&mut input).map_err(|e| e.to_string())?;
        let request = serde_json::from_str(&input).map_err(|e| e.to_string())?;
        process(request)
    })();
    let (code, value) = match result {
        Ok(value) => (0, json!({"ok":true,"document":value})),
        Err(error) => (1, json!({"ok":false,"error":error})),
    };
    if std::io::stdout().write_all(value.to_string().as_bytes()).is_err() {
        std::process::exit(2);
    }
    std::process::exit(code);
}
