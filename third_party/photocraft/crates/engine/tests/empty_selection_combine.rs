use photocraft_engine::Session;
use serde_json::json;

fn session() -> Session {
    let mut session = Session::new();
    session.execute("file.new", json!({"width": 32, "height": 24})).unwrap();
    session
}

#[test]
fn subtract_and_intersect_never_create_a_selection_from_none() {
    for mode in ["subtract", "intersect"] {
        for feather in [0.0, 2.0] {
            let mut session = session();
            session
                .execute(
                    "select.rect",
                    json!({
                        "x": 4, "y": 5, "width": 12, "height": 10,
                        "mode": mode, "feather": feather
                    }),
                )
                .unwrap();
            assert!(session.active().unwrap().doc.selection.is_none(), "mode={mode}, feather={feather}");
        }
    }
}

#[test]
fn replace_and_add_still_create_selections_from_none() {
    for mode in ["replace", "add"] {
        let mut session = session();
        session
            .execute(
                "select.rect",
                json!({
                    "x": 4, "y": 5, "width": 12, "height": 10, "mode": mode
                }),
            )
            .unwrap();
        let selection = session.active().unwrap().doc.selection.as_ref().unwrap();
        assert!(!selection.content_bounds().is_empty(), "mode={mode}");
    }
}

#[test]
fn intersect_with_an_existing_selection_keeps_the_overlap() {
    let mut session = session();
    session.execute("select.rect", json!({"x": 4, "y": 5, "width": 12, "height": 10})).unwrap();
    session
        .execute(
            "select.rect",
            json!({
                "x": 8, "y": 9, "width": 12, "height": 10, "mode": "intersect"
            }),
        )
        .unwrap();
    let selection = session.active().unwrap().doc.selection.as_ref().unwrap();
    let mut inside = [0.0];
    let mut outside = [0.0];
    selection.read_pixel(8, 9, &mut inside);
    selection.read_pixel(4, 5, &mut outside);
    assert!(inside[0] > 0.0);
    assert_eq!(outside[0], 0.0);
}

#[test]
fn subtracting_from_an_existing_selection_preserves_other_pixels_and_history() {
    for feather in [0.0, 2.0] {
        let mut session = session();
        session.execute("select.rect", json!({"x": 4, "y": 5, "width": 12, "height": 10})).unwrap();
        session
            .execute(
                "select.rect",
                json!({
                    "x": 8, "y": 9, "width": 12, "height": 10, "mode": "subtract", "feather": feather
                }),
            )
            .unwrap();
        let coverage = |session: &Session, x, y| {
            let mut pixel = [0.0];
            session.active().unwrap().doc.selection.as_ref().unwrap().read_pixel(x, y, &mut pixel);
            pixel[0]
        };
        let remaining = coverage(&session, 4, 5);
        let removed = coverage(&session, 12, 12);
        assert!(remaining > removed, "feather={feather}");
        assert!(session.undo());
        assert_eq!(coverage(&session, 12, 12), 1.0);
        assert!(session.redo());
        assert_eq!(coverage(&session, 12, 12), removed);
    }
}
