//! Closing a background tab must preserve the active document, not its old index.
use photocraft_doc::DocId;
use photocraft_engine::Session;
use serde_json::json;

fn session_with_documents(count: usize) -> (Session, Vec<DocId>) {
    let mut session = Session::new();
    let mut ids = Vec::new();
    for i in 0..count {
        session.execute("file.new", json!({"name": format!("Document {i}"), "width": 8, "height": 8})).unwrap();
        ids.push(session.active().unwrap().doc.id);
    }
    (session, ids)
}

fn document_ids(session: &Session) -> Vec<DocId> {
    session.documents().iter().map(|state| state.doc.id).collect()
}

#[test]
fn closing_a_background_document_keeps_the_active_document() {
    for (active, closing) in [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)] {
        let (mut session, mut ids) = session_with_documents(3);
        assert!(session.set_active(active));
        let active_id = ids[active];
        ids.remove(closing);

        session.execute("file.close", json!({"document": closing})).unwrap();

        assert_eq!(session.active().unwrap().doc.id, active_id, "active {active}, closing {closing}");
        assert_eq!(session.active_index(), ids.iter().position(|id| *id == active_id));
        assert_eq!(document_ids(&session), ids);
    }
}

#[test]
fn closing_the_active_document_selects_the_next_or_previous_document() {
    for (closing, next) in [(0, 1), (1, 2), (2, 1)] {
        let (mut session, mut ids) = session_with_documents(3);
        assert!(session.set_active(closing));
        let next_id = ids[next];
        ids.remove(closing);

        session.execute("file.close", json!({})).unwrap();

        assert_eq!(session.active().unwrap().doc.id, next_id);
        assert_eq!(session.active_index(), ids.iter().position(|id| *id == next_id));
        assert_eq!(document_ids(&session), ids);
    }
}

#[test]
fn closing_the_last_document_clears_the_active_document() {
    let (mut session, _) = session_with_documents(1);
    session.execute("file.close", json!({})).unwrap();
    assert!(session.documents().is_empty());
    assert_eq!(session.active_index(), None);
    assert!(session.active().is_none());
    assert!(session.close(0).is_none());
    assert!(session.execute("file.close", json!({})).is_err());
}

#[test]
fn closing_an_invalid_document_preserves_the_session() {
    let (mut session, ids) = session_with_documents(3);
    assert!(session.set_active(1));
    for index in [3, usize::MAX] {
        assert!(session.close(index).is_none());
        assert!(session.execute("file.close", json!({"document": index})).is_err());
        assert_eq!(session.active_index(), Some(1));
        assert_eq!(session.active().unwrap().doc.id, ids[1]);
        assert_eq!(document_ids(&session), ids);
    }
}

#[test]
fn closing_other_documents_keeps_the_requested_document() {
    let (mut session, ids) = session_with_documents(3);
    session.execute("file.closeOthers", json!({"document": 1})).unwrap();
    assert_eq!(document_ids(&session), vec![ids[1]]);
    assert_eq!(session.active_index(), Some(0));
    session.execute("file.closeAll", json!({})).unwrap();
    assert!(session.documents().is_empty());
    assert_eq!(session.active_index(), None);
}
