//! Keyboard layer navigation, the shortcuts that have no menu item (#352): ⌥] / ⌥[ select the
//! layer above / below, ⌥. / ⌥, the top / bottom layer, and ⇧⌥] / ⇧⌥[ add the layer above /
//! below to the selection. They follow the Layers panel's rows, top to bottom: a group's row comes
//! before its children, and the children of a collapsed group are skipped.

use photocraft_doc::{Document, Layer, LayerContent, LayerId};
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::layer_multi_cmds::set_selection;
use crate::{EngineError, Result, Session};

fn has_layers(s: &Session) -> std::result::Result<(), String> {
    let st = s.active().ok_or("no document open")?;
    if st.doc.layers.is_empty() { Err("the document has no layers".into()) } else { Ok(()) }
}

/// The Layers panel's rows, top to bottom: each group before its children, collapsed groups
/// without them.
pub fn panel_rows(doc: &Document) -> Vec<LayerId> {
    fn rec(layers: &[Layer], out: &mut Vec<LayerId>) {
        for l in layers.iter().rev() {
            out.push(l.id);
            if let LayerContent::Group(g) = &l.content
                && g.expanded
            {
                rec(&g.children, out);
            }
        }
    }
    let mut out = Vec::new();
    rec(&doc.layers, &mut out);
    out
}

/// Where `id` is in `rows`; a layer inside a collapsed group counts as that group's row (the
/// nearest enclosing one that has a row).
fn row_of(doc: &Document, rows: &[LayerId], id: LayerId) -> Option<usize> {
    let path = doc.path_of(id)?;
    (1..=path.len()).rev().find_map(|n| doc.layer_at(&path[..n]).and_then(|l| rows.iter().position(|r| *r == l.id)))
}

#[derive(Clone, Copy)]
enum Step {
    Above,
    Below,
    Top,
    Bottom,
}

/// Selects (or with `add`, adds to the selection) the row `step` away from the active layer.
/// At the top or bottom row the selection stays as it is.
fn navigate(s: &mut Session, step: Step, add: bool) -> Result<Value> {
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let rows = panel_rows(&st.doc);
    let last = rows.len().checked_sub(1).ok_or_else(|| EngineError::Other("the document has no layers".into()))?;
    let current = st.active_layer.and_then(|a| row_of(&st.doc, &rows, a));
    let target = match (step, current) {
        (Step::Top, _) | (Step::Above, None) => 0,
        (Step::Bottom, _) | (Step::Below, None) => last,
        (Step::Above, Some(i)) => i.saturating_sub(1),
        (Step::Below, Some(i)) => (i + 1).min(last),
    };
    let id = rows.get(target).copied().ok_or_else(|| EngineError::Other("no layer there".into()))?;
    if add {
        let mut sel = st.selected_layers();
        if !sel.contains(&id) {
            sel.push(id);
        }
        let anchor = st.layer_anchor;
        set_selection(s, sel, Some(id), anchor)?;
    } else {
        s.select_layer(id)?;
    }
    let st = s.active().ok_or(EngineError::NoDocument)?;
    Ok(json!({"layer": id.0, "selected": st.selected_layers().iter().map(|l| l.0).collect::<Vec<_>>()}))
}

pub fn specs() -> Vec<CommandSpec> {
    macro_rules! spec {
        ($id:expr, $label:expr, $sc:expr, $step:expr, $add:expr) => {
            CommandSpec {
                id: $id,
                label: $label,
                menu: &[],
                shortcut: Some($sc),
                params: "{} (follows the Layers panel's rows; collapsed groups are one row)",
                enabled: has_layers,
                journal: true,
                run: |s, _| navigate(s, $step, $add),
            }
        };
    }
    vec![
        spec!("layer.selectAbove", "Select Layer Above", "Alt+]", Step::Above, false),
        spec!("layer.selectBelow", "Select Layer Below", "Alt+[", Step::Below, false),
        spec!("layer.selectTop", "Select Top Layer", "Alt+.", Step::Top, false),
        spec!("layer.selectBottom", "Select Bottom Layer", "Alt+,", Step::Bottom, false),
        spec!("layer.addAboveToSelection", "Add Layer Above to Selection", "Alt+Shift+]", Step::Above, true),
        spec!("layer.addBelowToSelection", "Add Layer Below to Selection", "Alt+Shift+[", Step::Below, true),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bottom to top: Background, A, Group { B, C }, D. Panel rows: D, Group, C, B, A, Background.
    fn session() -> (Session, [LayerId; 6]) {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 16, "height": 16})).unwrap();
        let bg = s.active().unwrap().doc.layers[0].id;
        let id = |s: &mut Session, cmd: &str, p: Value| LayerId(s.execute(cmd, p).unwrap()["layer"].as_u64().unwrap());
        let a = id(&mut s, "layer.new.layer", json!({"name": "A"}));
        let b = id(&mut s, "layer.new.layer", json!({"name": "B"}));
        let c = id(&mut s, "layer.new.layer", json!({"name": "C"}));
        s.execute("layer.select", json!({"layer": b.0})).unwrap();
        s.execute("layer.select", json!({"layer": c.0, "mode": "add"})).unwrap();
        s.execute("layer.groupLayers", json!({})).unwrap();
        let group = s.active().unwrap().doc.layers.iter().find(|l| l.is_group()).unwrap().id;
        let d = id(&mut s, "layer.new.layer", json!({"name": "D"}));
        (s, [bg, a, group, b, c, d])
    }

    fn active(s: &Session) -> LayerId {
        s.active().unwrap().active_layer.unwrap()
    }

    #[test]
    fn steps_follow_the_panel_rows_and_stop_at_the_ends() {
        let (mut s, [bg, a, group, b, c, d]) = session();
        assert_eq!(panel_rows(&s.active().unwrap().doc), vec![d, group, c, b, a, bg]);
        s.execute("layer.select", json!({"layer": d.0})).unwrap();
        for want in [group, c, b, a, bg, bg] {
            s.execute("layer.selectBelow", json!({})).unwrap();
            assert_eq!(active(&s), want);
        }
        for want in [a, b, c, group, d, d] {
            s.execute("layer.selectAbove", json!({})).unwrap();
            assert_eq!(active(&s), want);
        }
        s.execute("layer.selectBottom", json!({})).unwrap();
        assert_eq!((active(&s), s.active().unwrap().selected_layers()), (bg, vec![bg]));
        s.execute("layer.selectTop", json!({})).unwrap();
        assert_eq!(active(&s), d);
    }

    #[test]
    fn collapsed_groups_are_one_row_and_adding_extends_the_selection() {
        let (mut s, [bg, a, group, _b, c, d]) = session();
        s.edit("collapse", |doc, _| {
            if let Some(LayerContent::Group(g)) = doc.layer_mut(group).map(|l| &mut l.content) {
                g.expanded = false;
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(panel_rows(&s.active().unwrap().doc), vec![d, group, a, bg]);
        // A layer inside the collapsed group steps from the group's row.
        s.execute("layer.select", json!({"layer": c.0})).unwrap();
        s.execute("layer.selectBelow", json!({})).unwrap();
        assert_eq!(active(&s), a);
        s.execute("layer.addAboveToSelection", json!({})).unwrap();
        s.execute("layer.addAboveToSelection", json!({})).unwrap();
        let mut sel = s.active().unwrap().selected_layers();
        sel.sort();
        let mut want = vec![a, group, d];
        want.sort();
        assert_eq!((active(&s), sel), (d, want));
        s.execute("layer.addBelowToSelection", json!({})).unwrap();
        assert_eq!(active(&s), group, "the active layer moves; the selection keeps every layer");
        assert_eq!(s.active().unwrap().selected_layers().len(), 3);
    }

    #[test]
    fn selecting_never_dirties_and_bad_states_are_errors() {
        let (mut s, _) = session();
        let st = s.active_mut().unwrap();
        st.saved_revision = st.revision;
        s.execute("layer.selectTop", json!({"junk": [1, 2]})).unwrap();
        assert!(!s.active().unwrap().is_dirty(), "selecting a layer is not an edit");
        let mut empty = Session::new();
        for c in specs() {
            assert!(!empty.is_enabled(c.id));
            assert!(empty.execute(c.id, json!({})).is_err());
        }
        empty.execute("file.new", json!({"width": 4, "height": 4})).unwrap();
        empty
            .edit("clear", |doc, active| {
                doc.layers.clear();
                *active = None;
                Ok(())
            })
            .unwrap();
        for c in specs() {
            assert!(empty.execute(c.id, json!({})).is_err(), "{}: no layers", c.id);
        }
    }
}
