//! Clip groups follow layers as they are added, moved and deleted, as in Photoshop (#2203).
//!
//! A clipped layer clips to the nearest unclipped layer below it (its base); a base and the run of
//! clipped layers directly above it form a clip group. Photoshop keeps those groups whole while
//! the stack changes, by these rules (measured in Photoshop 27.11, the table on #2203):
//!
//! 1. A layer added or moved joins the clip group it lands in: it is clipped when the layer
//!    directly above its new slot is clipped (unless that layer is one of its own group travelling
//!    with it).
//! 2. A clipped layer moved stays clipped while it still sits on a clip group: the layer below its
//!    new slot is clipped, or was in its group (`B, C1*, C2*` + Bring Forward on C1 → `B, C2*, C1*`).
//!    Otherwise it is released (`B, C*, U` + Bring Forward on C → `B, U, C`).
//! 3. When a base is moved away or deleted, the layers that were clipped to it are released. Rule
//!    1 reads the stack before this release, so a base moved up into its own group joins it while
//!    the layers it left are released (`B, C1*, C2*` + Bring Forward on B → `C1, B*, C2`).
//! 4. A clipped layer left with nothing below it to clip to is released.

use photocraft_doc::{Document, Layer, LayerId};

/// The sibling list holding the layer at `path`, and the layer's index in it.
fn siblings<'a>(doc: &'a Document, path: &[usize]) -> Option<(&'a [Layer], usize)> {
    let (&idx, parent) = path.split_last()?;
    let list = if parent.is_empty() { &doc.layers[..] } else { doc.layer_at(parent)?.children()? };
    Some((list, idx))
}

/// The layer `delta` slots above (+) or below (−) `id` among its siblings.
fn neighbour(doc: &Document, id: LayerId, delta: isize) -> Option<&Layer> {
    let path = doc.path_of(id)?;
    let (list, idx) = siblings(doc, &path)?;
    list.get(idx.checked_add_signed(delta)?)
}

/// The clip group `id` belongs to, base first, when it has one (a clipped layer, or a base with
/// layers clipped to it).
fn group_of(doc: &Document, id: LayerId) -> Vec<LayerId> {
    let Some(path) = doc.path_of(id) else { return Vec::new() };
    let Some((list, idx)) = siblings(doc, &path) else { return Vec::new() };
    let base = list.iter().take(idx.saturating_add(1)).rposition(|l| !l.clipped);
    let Some(base) = base else { return Vec::new() };
    let group: Vec<LayerId> = list.iter().skip(base).enumerate().take_while(|(i, l)| *i == 0 || l.clipped).map(|(_, l)| l.id).collect();
    if group.len() > 1 { group } else { Vec::new() }
}

/// The layers clipped to `id`: the clipped run directly above it. Empty when `id` is itself
/// clipped (it is not a base) or nothing is clipped to it.
fn clipped_to(doc: &Document, id: LayerId) -> Vec<LayerId> {
    match group_of(doc, id).split_first() {
        Some((&base, rest)) if base == id => rest.to_vec(),
        _ => Vec::new(),
    }
}

/// The clip groups of layers about to move or be deleted, read before they leave their slots.
#[derive(Default)]
pub(crate) struct Before {
    moving: Vec<LayerId>,
    /// Each moving layer that was clipped.
    clipped: Vec<LayerId>,
    /// Each moving layer's clip group (base first), when it had one.
    groups: Vec<(LayerId, Vec<LayerId>)>,
    /// The layers that lose their base: those clipped to a moving base, except the moving ones.
    left: Vec<LayerId>,
}

impl Before {
    /// Read the clip groups of `moving` (moved or deleted together).
    pub(crate) fn read(doc: &Document, moving: &[LayerId]) -> Self {
        let clipped = moving.iter().copied().filter(|&id| doc.layer(id).is_some_and(|l| l.clipped)).collect();
        let groups = moving.iter().map(|&id| (id, group_of(doc, id))).filter(|(_, g)| !g.is_empty()).collect();
        let mut left: Vec<LayerId> = moving.iter().flat_map(|&id| clipped_to(doc, id)).filter(|id| !moving.contains(id)).collect();
        left.dedup();
        Self { moving: moving.to_vec(), clipped, groups, left }
    }

    fn group(&self, id: LayerId) -> &[LayerId] {
        self.groups.iter().find(|(m, _)| *m == id).map_or(&[], |(_, g)| g)
    }
}

fn set_clipped(doc: &mut Document, id: LayerId, clipped: bool) {
    if let Some(l) = doc.layer_mut(id) {
        l.clipped = clipped;
    }
}

/// After `placed` were added or moved (`before` read before they moved; default for new layers):
/// each placed layer joins or leaves clip groups by rules 1–2, topmost first so a run of placed
/// layers joins together, or is clipped anyway when `force` (New Layer's Use Previous Layer to
/// Create Clipping Mask). Then the layers that lost their base are released (rule 3), and any
/// clipped layer with nothing below it (rule 4).
pub(crate) fn settle(doc: &mut Document, placed: &[LayerId], before: &Before, force: bool) {
    let mut order: Vec<(Vec<usize>, LayerId)> = placed.iter().filter_map(|&id| Some((doc.path_of(id)?, id))).collect();
    // Paths sort bottom to top within a sibling list: walk them from the top.
    order.sort();
    for (_, id) in order.into_iter().rev() {
        let group = before.group(id);
        let travelling = |n: LayerId| before.moving.contains(&n) && group.contains(&n);
        let joins = neighbour(doc, id, 1).is_some_and(|a| a.clipped && !travelling(a.id));
        let stays = before.clipped.contains(&id) && neighbour(doc, id, -1).is_some_and(|b| b.clipped || group.contains(&b.id));
        set_clipped(doc, id, force || joins || stays);
    }
    for &id in &before.left {
        if !placed.contains(&id) {
            set_clipped(doc, id, false);
        }
    }
    release_orphans(doc);
}

/// Rule 4: release the clipped layers at the bottom of a sibling list (nothing below to clip to).
/// The compositor already draws such a run unclipped, so this changes no pixels. Every list is
/// visited once from an explicit stack, so a deep document can't recurse without bound.
pub(crate) fn release_orphans(doc: &mut Document) {
    let mut stack: Vec<&mut Vec<Layer>> = vec![&mut doc.layers];
    while let Some(list) = stack.pop() {
        for l in list.iter_mut().take_while(|l| l.clipped) {
            l.clipped = false;
        }
        for l in list {
            if let Some(ch) = l.children_mut() {
                stack.push(ch);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use crate::Session;

    /// A document holding `spec` (bottom to top, `*` = clipped) above its Background.
    fn stack(spec: &str) -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 16, "height": 16})).unwrap();
        for name in spec.split_whitespace() {
            let id = s.execute("layer.new.layer", json!({"name": name.trim_end_matches('*')})).unwrap()["layer"].clone();
            if name.ends_with('*') {
                s.execute("layer.setProps", json!({"layer": id, "clipped": true})).unwrap();
            }
        }
        assert_eq!(shown(&s), spec);
        s
    }

    /// The stack above the Background, as [`stack`] writes it.
    fn shown(s: &Session) -> String {
        let doc = &s.active().unwrap().doc;
        let names = doc.layers.iter().skip(1).map(|l| format!("{}{}", l.name, if l.clipped { "*" } else { "" }));
        names.collect::<Vec<_>>().join(" ")
    }

    fn id(s: &Session, name: &str) -> Value {
        let doc = &s.active().unwrap().doc;
        json!(doc.walk().into_iter().find(|(_, _, l)| l.name == name).unwrap().2.id.0)
    }

    fn run(s: &mut Session, cmd: &str, p: Value) {
        s.execute(cmd, p).unwrap_or_else(|e| panic!("{cmd}: {e}"));
    }

    fn select(s: &mut Session, name: &str) {
        let l = id(s, name);
        run(s, "layer.select", json!({"layer": l}));
    }

    /// New Layer joins the clip group it lands in (#2203 rows 1–6).
    #[test]
    fn new_layer_joins_the_group_it_lands_in() {
        for (spec, active, want) in [("B C*", "B", "B N* C*"), ("B C*", "C", "B C* N"), ("B C1* C2*", "C1", "B C1* N* C2*"), ("B U", "B", "B N U")] {
            let mut s = stack(spec);
            select(&mut s, active);
            run(&mut s, "layer.new.layer", json!({"name": "N"}));
            assert_eq!(shown(&s), want, "{spec}, {active} active");
        }
        // Use Previous Layer to Create Clipping Mask.
        let mut s = stack("B");
        run(&mut s, "layer.new.layer", json!({"name": "N", "clipToPrevious": true}));
        assert_eq!(shown(&s), "B N*");
        // The report: three new layers with the base active all join its group.
        let mut s = stack("tpl Cont*");
        for n in ["L1", "L2", "L3"] {
            select(&mut s, "tpl");
            run(&mut s, "layer.new.layer", json!({"name": n}));
        }
        assert_eq!(shown(&s), "tpl L3* L2* L1* Cont*");
    }

    /// Bring Forward / Send Backward (rows 7–12).
    #[test]
    fn arrange_joins_and_leaves_groups() {
        for (spec, layer, cmd, times, want) in [
            ("B C1* C2* U", "U", "sendBackward", 1, "B C1* U* C2*"),
            ("B C1* C2* U", "U", "sendBackward", 2, "B U* C1* C2*"),
            ("B C* U", "C", "bringForward", 1, "B U C"),
            ("B C1* C2*", "C1", "bringForward", 1, "B C2* C1*"),
            // Photoshop also moves C below X here (`C X B`); the release is the same.
            ("X B C*", "C", "sendBackward", 1, "X C B"),
            // The base moves up into its own group: it joins it, and the layers it left go.
            ("B C1* C2*", "B", "bringForward", 1, "C1 B* C2"),
            ("B C1* C2* U", "U", "sendToBack", 1, "U B C1* C2*"),
            ("B C1* C2* U", "C1", "bringToFront", 1, "B C2* U C1"),
        ] {
            let mut s = stack(spec);
            for _ in 0..times {
                let l = id(&s, layer);
                run(&mut s, &format!("layer.arrange.{cmd}"), json!({"layer": l}));
            }
            assert_eq!(shown(&s), want, "{spec}: {cmd} {layer} ×{times}");
        }
    }

    /// Layers panel drags (rows 13–16), several layers at once, and ⌥-drag copies.
    #[test]
    fn reorder_joins_and_leaves_groups() {
        for (spec, layers, target, pos, want) in [
            ("B C1* C2* U", vec!["U"], "C1", "above", "B C1* U* C2*"),
            ("U B C1* C2*", vec!["U"], "C1", "above", "B C1* U* C2*"),
            ("B C1* C2* U", vec!["C1"], "U", "above", "B C2* U C1"),
            ("X B C1* C2*", vec!["C2"], "B", "below", "X C2 B C1*"),
            ("B C1* C2* U V", vec!["U", "V"], "C1", "above", "B C1* U* V* C2*"),
            ("X B C1* C2* U", vec!["B", "C1"], "U", "above", "X C2 U B C1*"),
        ] {
            let mut s = stack(spec);
            let ids: Vec<Value> = layers.iter().map(|n| id(&s, n)).collect();
            let t = id(&s, target);
            run(&mut s, "layer.moveTo", json!({"layers": ids, "target": t, "position": pos}));
            assert_eq!(shown(&s), want, "{spec}: {layers:?} {pos} {target}");
        }
        let mut s = stack("B C1* C2* U");
        let (u, c1) = (id(&s, "U"), id(&s, "C1"));
        run(&mut s, "layer.moveTo", json!({"layer": u, "target": c1, "position": "above", "copy": true}));
        assert_eq!(shown(&s), "B C1* U copy* C2* U");
    }

    /// Deleting a base releases its group (row 17); deleting a clipped layer keeps the rest.
    #[test]
    fn delete_releases_the_group_of_a_deleted_base() {
        let mut s = stack("X B C1* C2*");
        select(&mut s, "B");
        run(&mut s, "layer.delete", json!({}));
        assert_eq!(shown(&s), "X C1 C2");
        let mut s = stack("X B C1* C2*");
        let c1 = id(&s, "C1");
        run(&mut s, "layer.delete", json!({"layer": c1}));
        assert_eq!(shown(&s), "X B C2*");
        // Several selected: the base and one of its layers.
        let mut s = stack("X B C1* C2*");
        select(&mut s, "B");
        let c1 = id(&s, "C1");
        run(&mut s, "layer.select", json!({"layer": c1, "mode": "toggle"}));
        run(&mut s, "layer.delete", json!({}));
        assert_eq!(shown(&s), "X C2");
        // Each change is one history step.
        run(&mut s, "edit.undo", json!({}));
        assert_eq!(shown(&s), "X B C1* C2*");
    }

    /// A clipped run left at the bottom of a group has nothing to clip to: it is released (the
    /// compositor already draws it unclipped), wherever the edit happened.
    #[test]
    fn orphaned_clipped_layers_are_released() {
        let mut s = stack("A B U");
        let (a, b, u) = (id(&s, "A"), id(&s, "B"), id(&s, "U"));
        run(&mut s, "layer.select", json!({"layer": a}));
        run(&mut s, "layer.select", json!({"layer": b, "mode": "toggle"}));
        run(&mut s, "layer.groupLayers", json!({"name": "G"}));
        s.edit("orphans", |doc, _| {
            for l in doc.layers.iter_mut().flat_map(|g| g.children_mut().into_iter().flatten()) {
                l.clipped = true;
            }
            Ok(())
        })
        .unwrap();
        run(&mut s, "layer.arrange.sendBackward", json!({"layer": u}));
        let doc = &s.active().unwrap().doc;
        let g = doc.walk().into_iter().find(|(_, _, l)| l.name == "G").unwrap().2;
        assert!(g.children().unwrap().iter().all(|l| !l.clipped));
    }
}
