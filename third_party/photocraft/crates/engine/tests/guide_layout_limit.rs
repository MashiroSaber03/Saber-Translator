//! Regression tests for issue #704: `view.newGuideLayout` must reject an absurd
//! column/row count instead of running an unbounded O(n^2) loop. Photoshop's
//! New Guide Layout caps at 32; we reject counts above a generous 1000
//! (still O(10^6) worst case) with an actionable BadParams error.
use photocraft_engine::Session;
use serde_json::json;

fn session() -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 3000, "height": 2000})).unwrap();
    s
}

#[test]
fn huge_column_count_is_rejected_fast() {
    let mut s = session();
    // The issue's repro: 200 million columns never returned on main.
    let r = s.execute("view.newGuideLayout", json!({"columns": 200000000, "gutter": 0}));
    let e = r.expect_err("absurd column count must be rejected");
    let msg = format!("{e:?}");
    assert!(msg.contains("column") || msg.contains("Column"), "error should mention columns: {msg}");
    assert!(!msg.contains("internal error"), "must be an actionable error, not the panic guard: {msg}");
}

#[test]
fn huge_row_count_is_rejected_fast() {
    let mut s = session();
    let e = s.execute("view.newGuideLayout", json!({"rows": 200000000})).expect_err("absurd row count must be rejected");
    let msg = format!("{e:?}");
    assert!(msg.contains("row") || msg.contains("Row"), "error should mention rows: {msg}");
}

#[test]
fn bounded_layouts_still_work() {
    let mut s = session();
    // A grid at the cap must still succeed quickly.
    let v = s.execute("view.newGuideLayout", json!({"columns": 1000, "gutter": 0})).unwrap();
    let n = v["vertical"].as_u64().unwrap();
    assert!((1000..=1001).contains(&n), "1000-column grid should produce ~1000 guides, got {n}");
    let v = s.execute("view.newGuideLayout", json!({"rows": 3, "columns": 3, "clearExisting": true})).unwrap();
    for k in ["vertical", "horizontal"] {
        let n = v[k].as_u64().unwrap();
        assert!((3..=4).contains(&n), "3x3 grid {k} count should be 3-4, got {n}");
    }
}
