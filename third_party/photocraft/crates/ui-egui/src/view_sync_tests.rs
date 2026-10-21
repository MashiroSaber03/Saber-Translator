//! Views follow their document, not their slot: closing a tab must not hand its zoom or centre
//! to the document that shifts into its place.

use serde_json::json;

use crate::PhotocraftApp;

fn app_with_two_documents() -> PhotocraftApp {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 64, "height": 48})).unwrap();
    app.run("file.new", json!({"width": 32, "height": 32})).unwrap();
    assert_eq!(app.ui.views.len(), 2);
    // The first document is zoomed in; the second keeps its own default view.
    app.ui.views[0].zoom = 3.2;
    app.ui.views[0].fit_pending = false;
    app
}

/// Regression: views lived next to the documents by index alone, so closing a non-last tab
/// shifted the surviving document into the closed one's slot and it inherited that view — open a
/// 3200 % document, close it, and the next one came up at 3200 %. Views are matched by document
/// id now, so the survivor keeps its own (default) view.
#[test]
fn closing_a_tab_keeps_the_next_documents_own_view() {
    let mut app = app_with_two_documents();
    app.run("file.close", json!({"document": 0})).unwrap();
    assert_eq!(app.session.documents().len(), 1);
    assert_eq!(app.ui.views.len(), 1);
    assert_eq!(app.ui.views[0].zoom, 1.0, "the survivor must not inherit the closed document's zoom");
}

/// Opening and closing documents in any order keeps one view per document, and re-showing a
/// document (undo of a close is out of scope, but reopening by path is not) starts fresh rather
/// than claiming another document's view.
#[test]
fn every_document_keeps_its_own_view_through_syncs() {
    let mut app = app_with_two_documents();
    app.ui.views[1].zoom = 0.5;
    // A third document joins at the end, then the middle one closes: the third keeps its zoom.
    app.run("file.new", json!({"width": 16, "height": 16})).unwrap();
    app.ui.views[2].zoom = 4.0;
    app.run("file.close", json!({"document": 1})).unwrap();
    assert_eq!(app.ui.views.len(), 2);
    assert_eq!(app.ui.views[0].zoom, 3.2, "the first document's view is untouched");
    assert_eq!(app.ui.views[1].zoom, 4.0, "the last document's view follows it to its new index");
}
