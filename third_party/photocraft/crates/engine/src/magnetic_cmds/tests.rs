use photocraft_geom::Rect;
use serde_json::json;

use super::*;

/// A black 40×40 square at (40, 30) on a white 120×100 background.
fn session() -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 120, "height": 100})).unwrap();
    s.edit("paint", |doc, _| {
        let bg = doc.layers[0].surface_mut().unwrap();
        bg.fill_rect(Rect::new(0, 0, 120, 100), &[1.0, 1.0, 1.0, 1.0]);
        bg.fill_rect(Rect::new(40, 30, 80, 70), &[0.0, 0.0, 0.0, 1.0]);
        Ok(())
    })
    .unwrap();
    s
}

fn coverage(s: &Session, x: i32, y: i32) -> f32 {
    s.active().unwrap().doc.selection.as_ref().map_or(0.0, |m| m.sample_channel(x, y, 0))
}

fn selected_area(s: &Session) -> f32 {
    (0..100).flat_map(|y| (0..120).map(move |x| (x, y))).map(|(x, y)| coverage(s, x, y)).sum()
}

/// Rough points near the square's corners, some inside and some outside it.
const ROUGH: [[f64; 2]; 4] = [[43.0, 33.0], [77.0, 27.0], [83.0, 72.0], [37.0, 66.0]];

#[test]
fn rough_points_select_the_shape_along_its_edges() {
    let mut s = session();
    let r = s.execute("select.magneticLasso", json!({"points": ROUGH})).unwrap();
    assert_eq!(r["selected"], json!(true));
    assert!(r["points"].as_u64().unwrap() >= 4, "{r}");
    let area = selected_area(&s);
    assert!((area - 1600.0).abs() < 40.0, "the square's area, not the rough quadrilateral's: {area}");
    for (x, y) in [(60, 50), (41, 31), (78, 68), (41, 68), (78, 31)] {
        assert!(coverage(&s, x, y) > 0.99, "({x}, {y}) inside");
    }
    for (x, y) in [(38, 50), (81, 50), (60, 28), (60, 71), (36, 66), (82, 71)] {
        assert!(coverage(&s, x, y) < 0.01, "({x}, {y}) outside");
    }
    let st = s.active().unwrap();
    assert_eq!(st.history.entries().last().map(String::as_str), Some("Magnetic Lasso"));
    // One step: Undo takes it all back.
    s.execute("edit.undo", json!({})).unwrap();
    assert!(s.active().unwrap().doc.selection.is_none());
}

#[test]
fn modes_and_a_given_outline() {
    let mut s = session();
    // `trace: false` selects the outline as given, like the tool's own commit.
    s.execute("select.magneticLasso", json!({"points": [[0, 0], [20, 0], [0, 20]], "trace": false})).unwrap();
    assert_eq!((coverage(&s, 2, 2), coverage(&s, 18, 18)), (1.0, 0.0));
    s.execute("select.magneticLasso", json!({"points": ROUGH, "mode": "add"})).unwrap();
    assert_eq!((coverage(&s, 2, 2), coverage(&s, 60, 50)), (1.0, 1.0));
    s.execute("select.magneticLasso", json!({"points": ROUGH, "mode": "subtract", "close": "straight", "width": 20, "contrast": 50})).unwrap();
    assert_eq!((coverage(&s, 2, 2), coverage(&s, 60, 50)), (1.0, 0.0));
    s.execute("select.magneticLasso", json!({"points": ROUGH, "feather": 4})).unwrap();
    let edge = coverage(&s, 40, 50);
    assert!(edge > 0.2 && edge < 0.8, "feathered: {edge}");
}

#[test]
fn edges_come_from_the_active_pixel_layer() {
    let mut s = session();
    s.execute("layer.new.layer", json!({})).unwrap();
    // An empty layer has no edges: the point stays where it is.
    let settings = Settings::default();
    let mut src = EdgeSource::new(&s).unwrap();
    assert_eq!(src.snap([44.0, 50.0], settings), Some([44.0, 50.0]));
    let bg = s.active().unwrap().doc.layers[0].id;
    s.execute("layer.select", json!({"layer": bg.0})).unwrap();
    assert!(!src.is_current(&s), "another layer");
    let mut src = EdgeSource::new(&s).unwrap();
    assert!(src.is_current(&s));
    let p = src.snap([44.0, 50.0], settings).unwrap();
    assert!((p[0] - 40.0).abs() < 0.01 && (p[1] - 50.0).abs() <= 1.0, "on the square's edge: {p:?}");
    // A layer without pixels of its own (an Invert adjustment): the composite, its square white
    // on black but its edge where it was.
    s.execute("layer.newAdjustmentLayer.invert", json!({})).unwrap();
    let st = s.active().unwrap();
    assert!(st.active_layer.and_then(|id| st.doc.layer(id)).is_some_and(|l| l.surface().is_none()));
    let mut src = EdgeSource::new(&s).unwrap();
    let p = src.snap([44.0, 50.0], settings).unwrap();
    assert!((p[0] - 40.0).abs() < 0.01, "{p:?}");
}

#[test]
fn reads_only_around_the_border_of_a_large_image() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 6000, "height": 4000, "background": "transparent"})).unwrap();
    s.edit("paint", |doc, _| {
        doc.layers[0].surface_mut().unwrap().fill_rect(Rect::new(2000, 1500, 2200, 1700), &[1.0, 0.0, 0.0, 1.0]);
        Ok(())
    })
    .unwrap();
    let mut src = EdgeSource::new(&s).unwrap();
    let border = trace_border(&mut src, &[[2003.0, 1497.0], [2198.0, 1504.0], [2203.0, 1702.0], [1998.0, 1695.0]], Settings::default(), false).unwrap();
    assert!(border.iter().all(|p| (1999.0..=2201.0).contains(&p[0]) && (1499.0..=1701.0).contains(&p[1])), "{border:?}");
    let read = src.tracer.cached_pixels();
    assert!(read < 40 * 128 * 128, "read {read} of 24M px");
}

#[test]
fn bad_params_are_refused() {
    let mut s = session();
    for p in [
        json!({}),
        json!({"points": "here"}),
        json!({"points": [[0, 0], [10, 0]]}),
        json!({"points": [[0, 0], [10, 0], ["a", 3]]}),
        json!({"points": [[0, 0], [10, 0], [3]]}),
        json!({"points": [[0, 0], [10, 0], [3, 4, 5]]}),
        json!({"points": ROUGH, "width": 0}),
        json!({"points": ROUGH, "width": 300}),
        json!({"points": ROUGH, "width": "wide"}),
        json!({"points": ROUGH, "contrast": 0}),
        json!({"points": ROUGH, "contrast": 101}),
        json!({"points": ROUGH, "close": "zigzag"}),
        json!({"points": ROUGH, "close": 1}),
        json!({"points": ROUGH, "trace": "yes"}),
        json!({"points": ROUGH, "feather": -1}),
        json!({"points": ROUGH, "feather": 5000}),
        json!({"points": vec![[1, 1]; MAX_POINTS + 1]}),
    ] {
        assert!(s.execute("select.magneticLasso", p.clone()).is_err(), "{p}");
    }
    assert!(s.active().unwrap().doc.selection.is_none());
    // Far outside the canvas is kept on it; a border far too long is refused.
    s.execute("select.magneticLasso", json!({"points": [[-1e9, -1e9], [1e9, -1e9], [1e9, 1e9]]})).unwrap();
    let mut big = Session::new();
    big.execute("file.new", json!({"width": 30000, "height": 30000, "background": "transparent"})).unwrap();
    let zigzag: Vec<[f64; 2]> = (0..20).map(|i| [f64::from(i % 2) * 30000.0, f64::from(i) * 1500.0]).collect();
    assert!(big.execute("select.magneticLasso", json!({"points": zigzag})).is_err());
    let mut none = Session::new();
    assert!(none.execute("select.magneticLasso", json!({"points": ROUGH})).is_err());
}
