//! Photoshop refuses to paint on or move a hidden layer (#571): the painting and retouching
//! tools, the Paint Bucket, the Gradient and the Move tool stop with "the target layer is hidden"
//! instead of changing pixels nobody can see. A layer inside a hidden group is hidden too. An
//! alpha channel or Quick Mask target has no layer, so it is never refused.

use photocraft_doc::{Document, LayerId};
use serde_json::Value;

use crate::Session;

/// Photoshop's alert text.
pub const HIDDEN: &str = "Could not complete your request because the target layer is hidden.";

/// Commands that paint on, or move, the target layer.
const GATED: &[&str] = &[
    "paint.stroke",
    "paint.pencil",
    "paint.mixerBrush",
    "paint.colorReplacement",
    "paint.bucket",
    "paint.gradient",
    "paint.magicEraser",
    "paint.backgroundEraser",
    "paint.cloneStamp",
    "paint.healingBrush",
    "paint.spotHealing",
    "paint.patch",
    "paint.dodge",
    "paint.burn",
    "paint.sponge",
    "paint.blur",
    "paint.sharpen",
    "paint.smudge",
    "paint.historyBrush",
    "layer.translate",
];

/// Is the layer shown: it and every group around it visible.
fn shown(doc: &Document, id: LayerId) -> bool {
    doc.path_of(id).is_some_and(|path| (1..=path.len()).all(|n| path.get(..n).and_then(|p| doc.layer_at(p)).is_some_and(|l| l.visible)))
}

/// Why command `id` with `params` must not run: its target layer is hidden. `None` for other
/// commands, channel targets, and visible layers.
pub fn refusal(s: &Session, id: &str, params: &Value) -> Option<&'static str> {
    if !GATED.contains(&id) {
        return None;
    }
    let params = crate::channel_cmds::inject_target(s, id, params.clone());
    if crate::channel_cmds::is_channel_target(&params) {
        return None;
    }
    let st = s.active()?;
    let layer = params.get("layer").and_then(Value::as_u64).map(LayerId).or(st.active_layer)?;
    (st.doc.layer(layer).is_some() && !shown(&st.doc, layer)).then_some(HIDDEN)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::HIDDEN;
    use crate::Session;

    fn session() -> (Session, u64) {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 64, "height": 64, "background": "white"})).unwrap();
        let id = s.execute("layer.new.layer", json!({})).unwrap()["layer"].as_u64().unwrap();
        (s, id)
    }

    fn pixel(s: &mut Session) -> serde_json::Value {
        s.execute("document.pixel", json!({"x": 10, "y": 10})).unwrap()
    }

    #[test]
    fn painting_or_moving_a_hidden_layer_is_refused() {
        let (mut s, id) = session();
        s.execute("layer.setProps", json!({"layer": id, "visible": false})).unwrap();
        let steps = s.active().unwrap().history.entries().len();
        for (cmd, p) in [
            ("paint.stroke", json!({"points": [[10, 10], [20, 20]], "color": "#ff0000"})),
            ("paint.bucket", json!({"x": 10, "y": 10})),
            ("paint.gradient", json!({"from": [0, 0], "to": [60, 60]})),
            ("layer.translate", json!({"dx": 5, "dy": 0})),
        ] {
            let e = s.execute(cmd, p).unwrap_err().to_string();
            assert!(e.contains(HIDDEN), "{cmd}: {e}");
        }
        assert_eq!(s.active().unwrap().history.entries().len(), steps, "nothing changed");
        // Shown again, it paints.
        s.execute("layer.setProps", json!({"layer": id, "visible": true})).unwrap();
        s.execute("paint.stroke", json!({"points": [[10, 10], [12, 12]], "color": "#ff0000", "size": 9})).unwrap();
        assert_ne!(pixel(&mut s), json!([1.0, 1.0, 1.0, 1.0]));
    }

    #[test]
    fn a_layer_in_a_hidden_group_is_hidden() {
        let (mut s, id) = session();
        s.execute("layer.groupLayers", json!({})).unwrap();
        let g = s.active().unwrap().active_layer.unwrap().0;
        assert_ne!(g, id, "the new group is active");
        s.execute("layer.setProps", json!({"layer": g, "visible": false})).unwrap();
        s.execute("layer.select", json!({"layer": id})).unwrap();
        assert!(s.execute("paint.stroke", json!({"points": [[10, 10]]})).unwrap_err().to_string().contains(HIDDEN));
    }

    #[test]
    fn alpha_channels_of_a_hidden_layer_still_paint() {
        let (mut s, id) = session();
        s.execute("layer.setProps", json!({"layer": id, "visible": false})).unwrap();
        s.execute("channel.new", json!({})).unwrap();
        s.execute("paint.stroke", json!({"points": [[10, 10], [20, 20]], "target": {"channel": 0}})).unwrap();
    }
}
