use photocraft_engine::Session;
use serde_json::json;

fn select(s: &mut Session, id: u64, mode: &str) {
    s.execute("layer.select", json!({"layer": id, "mode": mode})).unwrap();
}

fn selected(s: &Session) -> Vec<u64> {
    let mut ids: Vec<_> = s.active().unwrap().selected_layers.iter().map(|id| id.0).collect();
    ids.sort_unstable();
    ids
}

#[test]
fn shift_range_follows_visible_rows_on_both_sides_of_an_expanded_group() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
    let b = s.execute("layer.new.layer", json!({"name": "B"})).unwrap()["layer"].as_u64().unwrap();
    let c = s.execute("layer.new.layer", json!({"name": "C"})).unwrap()["layer"].as_u64().unwrap();
    select(&mut s, b, "replace");
    select(&mut s, c, "add");
    let group = s.execute("layer.groupLayers", json!({})).unwrap()["layer"].as_u64().unwrap();
    let d = s.execute("layer.new.layer", json!({"name": "D"})).unwrap()["layer"].as_u64().unwrap();

    for (anchor, target, wanted) in
        [(d, group, vec![d, group]), (group, d, vec![d, group]), (group, b, vec![group, c, b]), (b, group, vec![group, c, b]), (c, b, vec![c, b])]
    {
        select(&mut s, anchor, "replace");
        select(&mut s, target, "range");
        let mut expected = wanted;
        expected.sort_unstable();
        assert_eq!(selected(&s), expected, "range from {anchor} to {target}");
    }
}
