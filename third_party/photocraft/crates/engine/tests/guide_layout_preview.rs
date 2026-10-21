use photocraft_engine::{Session, guide_layout};
use serde_json::json;

fn session() -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width":100,"height":60})).unwrap();
    s
}

#[test]
fn pure_preview_matches_commit_and_one_undo_restores_existing_guides() {
    let mut s = session();
    s.execute("view.newGuideLayout", json!({"columns":2})).unwrap();
    let before = s.active().unwrap().doc.guides.clone();
    let revision = s.active().unwrap().revision;
    let p = json!({"columns":3,"rows":2,"gutter":4,"rowGutter":6,"margin":[3,7,5,11],"clearExisting":true});
    let preview = guide_layout::calculate(&s, &p).unwrap();
    assert_eq!(s.active().unwrap().revision, revision);
    assert_eq!(s.active().unwrap().doc.guides, before);
    s.execute("view.newGuideLayout", p).unwrap();
    assert_eq!(s.active().unwrap().doc.guides, preview.guides);
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(s.active().unwrap().doc.guides, before);
    s.execute("edit.redo", json!({})).unwrap();
    assert_eq!(s.active().unwrap().doc.guides, preview.guides);
}

#[test]
fn fixed_sizes_center_and_keep_existing_lines() {
    let mut s = session();
    s.execute("view.newGuideLayout", json!({"margin":5})).unwrap();
    let before = s.active().unwrap().doc.guides.clone();
    let p = json!({"columns":2,"width":20,"gutter":10,"rows":2,"height":10,"rowGutter":4,"centerColumns":true});
    let result = guide_layout::calculate(&s, &p).unwrap();
    assert_eq!(result.added.vertical, vec![25.0, 45.0, 55.0, 75.0]);
    assert_eq!(result.added.horizontal, vec![18.0, 28.0, 32.0, 42.0]);
    assert!(before.vertical.iter().all(|g| result.guides.vertical.contains(g)));
    assert!(before.horizontal.iter().all(|g| result.guides.horizontal.contains(g)));
}

#[test]
fn invalid_geometry_is_atomic() {
    let mut s = session();
    let revision = s.active().unwrap().revision;
    for params in [
        json!({"columns":-1}),
        json!({"rows":1.5}),
        json!({"columns":2,"width":60}),
        json!({"columns":3,"gutter":50}),
        json!({"margin":[1,2,3]}),
        json!({"margin":[0,100,0,0]}),
        json!({"columns":2,"gutter":"bad"}),
        json!({"columns":2,"clearExisting":1}),
        json!({"columns":1,"target":"selectedArtboards"}),
    ] {
        assert!(s.execute("view.newGuideLayout", params.clone()).is_err(), "accepted {params}");
        assert_eq!(s.active().unwrap().revision, revision);
    }
}

#[test]
fn selected_artboards_use_offsets_and_clear_only_the_target_ranges() {
    let mut s = session();
    s.execute("view.newGuideLayout", json!({"margin":5})).unwrap();
    s.execute("layer.new.artboard", json!({"rect":[200,100,100,60]})).unwrap();
    let p = json!({"columns":2,"margin":[5,10,5,10],"target":"selectedArtboards","clearExisting":true});
    let preview = guide_layout::calculate(&s, &p).unwrap();
    assert_eq!(preview.added.vertical, vec![210.0, 290.0, 250.0]);
    assert_eq!(preview.added.horizontal, vec![105.0, 155.0]);
    assert!(preview.guides.vertical.contains(&5.0));
    s.execute("view.newGuideLayout", p).unwrap();
    assert_eq!(s.active().unwrap().doc.guides, preview.guides);
}
