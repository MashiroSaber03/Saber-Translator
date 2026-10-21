//! Artboards recognized from current .af properties move with their children and undo together.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[path = "../../io/tests/support/current_affinity.rs"]
mod fixture;

use fixture::{Marker, document};
use photocraft_affinity::synth::Method;
use photocraft_doc::{LayerContent, LayerId};
use photocraft_engine::{Session, file_cmds::open_bytes_as};
use photocraft_geom::{Point, Rect};
use serde_json::json;

fn state(s: &Session, id: LayerId) -> (Rect, Point) {
    let layer = s.active().unwrap().doc.layer(id).unwrap();
    let LayerContent::Group(group) = &layer.content else { panic!("not an artboard group") };
    let LayerContent::Shape(child) = &group.children[0].content else { panic!("not an editable child") };
    (group.artboard.as_ref().unwrap().rect, child.path.subpaths[0].knots[0].anchor)
}

#[test]
fn imported_current_boards_move_with_their_art_and_undo_without_moving_the_other_board() {
    for marker in [Marker::Legacy, Marker::Current, Marker::OtherProperties] {
        let mut s = Session::new();
        let bytes = document(marker, false, Method::Zlib);
        open_bytes_as(&mut s, "synthetic.af", &bytes, None, None).unwrap();
        let ids: Vec<_> = s.active().unwrap().doc.artboards().into_iter().map(|(id, _, _)| id).collect();
        if matches!(marker, Marker::OtherProperties) {
            assert!(ids.is_empty());
            assert!(!s.is_enabled("file.export.artboardsToFiles"));
            continue;
        }
        assert_eq!(ids.len(), 2);
        let before = state(&s, ids[0]);
        let other = state(&s, ids[1]);
        s.execute("layer.artboard.set", json!({"layer": ids[0].0, "x": 3, "y": 5})).unwrap();
        let (board, child) = state(&s, ids[0]);
        assert_eq!(board, Rect::new(3, 5, 19, 21));
        assert_eq!(child, Point::new(before.1.x + 3.0, before.1.y + 5.0));
        assert_eq!(state(&s, ids[1]), other);
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(state(&s, ids[0]), before);
        assert_eq!(state(&s, ids[1]), other);
        s.execute("edit.redo", json!({})).unwrap();
        assert_eq!(state(&s, ids[0]), (board, child));
    }
}
