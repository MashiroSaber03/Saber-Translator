use std::collections::HashMap;
use std::sync::OnceLock;

use photocraft_engine::Session;
use photocraft_engine::presets::{Group, styles::StylePreset};
use serde_json::{Value, json};

use crate::{Result, project};

pub fn manga_styles() -> Result<&'static [Group<StylePreset>]> {
    static STYLES: OnceLock<std::result::Result<Vec<Group<StylePreset>>, serde_json::Error>> = OnceLock::new();
    STYLES.get_or_init(|| serde_json::from_str(include_str!("manga_styles.json")))
        .as_deref().map_err(|e| format!("无法加载漫画图层样式：{e}"))
}

pub fn apply_bubble_style(session: &mut Session, command: &Value) -> Result<()> {
    let projection = project(session)?;
    let by_bubble: HashMap<_, _> = projection["bubbles"].as_array().into_iter().flatten().filter_map(|bubble| {
        Some((bubble["bubbleId"].as_str()?, bubble["layer"].as_u64()?))
    }).collect();
    let layers: Vec<_> = command["bubbleIds"].as_array().ok_or("缺少文字对象 ID")?.iter().map(|id| {
        id.as_str().and_then(|id| by_bubble.get(id).copied()).ok_or("找不到文字层")
    }).collect::<std::result::Result<_, _>>()?;
    let params = match (command.get("presetName"), command.get("recipe")) {
        (None, Some(recipe)) => json!({"layers": layers, "recipe": recipe}),
        (Some(Value::Null), None) => json!({"layers": layers, "preset": "Default Style (None)", "group": "Basics", "bundled": false}),
        (Some(Value::String(name)), None) => {
            session.presets.bundled_styles = manga_styles()?;
            json!({"layers": layers, "preset": name, "bundled": true})
        }
        _ => return Err("无效图层样式名称".into()),
    };
    session.execute("style.presets.apply", params).map_err(|e| e.to_string())?;
    Ok(())
}
