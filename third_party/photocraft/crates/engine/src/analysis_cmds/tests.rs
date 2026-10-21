use super::*;
use photocraft_doc::ColorMode;

fn session(w: u32, h: u32, depth: u32) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": w, "height": h, "depth": depth, "background": "white"})).unwrap();
    s
}

fn d(s: &Session) -> Arc<Document> {
    s.active().unwrap().doc.clone()
}

#[test]
fn record_measurements_refuses_an_area_too_large_to_measure() {
    // #933: with no selection the whole document is measured, and `width * height` overflowed
    // u32 on a legal 65536×65536 document, panicking inside `run`.
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 65536, "height": 65536, "background": "transparent"})).unwrap();
    let err = s.execute("image.analysis.recordMeasurements", json!({"source": "selection"})).unwrap_err();
    assert!(err.to_string().contains("too large to measure"), "{err}");
    assert!(s.analysis.log.is_empty());
    // A failed measurement does not use up a label: the next one is still "Measurement 1".
    s.execute("file.new", json!({"width": 20, "height": 20, "background": "white"})).unwrap();
    let r = s.execute("image.analysis.recordMeasurements", json!({"source": "selection"})).unwrap();
    assert_eq!(r["rows"][0]["values"]["label"], "Measurement 1");
    assert_eq!(r["rows"][0]["values"]["area"], 400.0);
}

#[test]
fn measurement_scale_set_query_default_and_undo() {
    let mut s = session(50, 40, 8);
    let r = s.execute("image.analysis.setMeasurementScale", json!({})).unwrap();
    assert_eq!(r["default"], true);
    let r = s.execute("image.analysis.setMeasurementScale", json!({"pixelLength": 100, "logicalLength": 2.5, "units": "cm"})).unwrap();
    assert_eq!(r["factor"], 0.025);
    assert_eq!(r["text"], "100 pixels = 2.5000 cm");
    assert_eq!(d(&s).measurement.scale.units, "cm");
    assert!(s.execute("image.analysis.setMeasurementScale", json!({"pixelLength": 0})).is_err());
    assert!(s.execute("image.analysis.setMeasurementScale", json!({"units": " "})).is_err());
    assert!(s.execute("image.analysis.setMeasurementScale", json!({"preset": "bogus"})).is_err());
    s.execute("image.analysis.setMeasurementScale", json!({"preset": "default"})).unwrap();
    assert!(d(&s).measurement.scale.is_default());
    assert!(s.undo());
    assert_eq!(d(&s).measurement.scale.units, "cm");
}

#[test]
fn features_area_perimeter_circularity() {
    // A 10×10 square and a 1-pixel diagonal line of 5 pixels (one 8-connected feature).
    let (w, h) = (30usize, 20usize);
    let mut mask = vec![false; w * h];
    for y in 2..12 {
        for x in 2..12 {
            mask[y * w + x] = true;
        }
    }
    for i in 0..5 {
        mask[(5 + i) * w + 20 + i] = true;
    }
    let gray: Vec<f32> = (0..w * h).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
    let f = measure_features(&mask, &gray, w, h, (100, 200), 255.0);
    assert_eq!(f.len(), 2);
    assert_eq!(f[0].area, 100.0);
    assert_eq!(f[0].bounds, Rect::new(102, 202, 112, 212));
    // Straight edges exact, corners cut by √½: 4·9 + 4·√½.
    assert!((f[0].perimeter - (36.0 + 4.0 * std::f64::consts::FRAC_1_SQRT_2)).abs() < 1e-9, "{}", f[0].perimeter);
    assert!(f[0].circularity() > 0.75 && f[0].circularity() < 0.9);
    assert_eq!(f[0].gray_min, 0.0);
    assert_eq!(f[0].gray_max, 255.0);
    assert!((f[0].gray_mean - 127.5).abs() < 3.0);
    assert_eq!(f[0].histogram.iter().sum::<u64>(), 100);
    assert_eq!(f[1].area, 5.0);
    // A disc's circularity approaches 1.
    let (w, h) = (101usize, 101usize);
    let mask: Vec<bool> = (0..w * h).map(|i| ((i % w) as f64 - 50.0).hypot((i / w) as f64 - 50.0) <= 40.0).collect();
    let f = measure_features(&mask, &vec![0.5; w * h], w, h, (0, 0), 1.0);
    assert_eq!(f.len(), 1);
    // Digital discs read ≈0.89 (the 8-direction contour is ~5 % long), as in ImageJ.
    assert!(f[0].circularity() > 0.85, "{}", f[0].circularity());
    assert!((f[0].perimeter / (2.0 * std::f64::consts::PI * 40.0) - 1.0).abs() < 0.08, "{}", f[0].perimeter);
    assert!((f[0].gray_median - 0.5).abs() < 1e-4);
}

#[test]
fn record_selection_rows_respect_scale_and_data_points() {
    for depth in [8, 16, 32] {
        let mut s = session(64, 48, depth);
        s.execute("select.all", json!({})).unwrap_or_default();
        s.execute("select.deselect", json!({})).unwrap_or_default();
        // Two separate rectangles.
        s.execute("select.rect", json!({"x": 4, "y": 4, "width": 10, "height": 10})).unwrap();
        s.execute("select.rect", json!({"x": 30, "y": 20, "width": 20, "height": 5, "mode": "add"})).unwrap();
        s.execute("image.analysis.setMeasurementScale", json!({"pixelLength": 2, "logicalLength": 1, "units": "mm"})).unwrap();
        let r = s.execute("image.analysis.recordMeasurements", json!({"dateTime": "2026-10-01T00:00:00Z"})).unwrap();
        assert_eq!(r["source"], "selection");
        let rows = r["rows"].as_array().unwrap();
        assert_eq!(rows.len(), 3, "summary + 2 features");
        let sum = &rows[0]["values"];
        assert_eq!(sum["count"], 2);
        assert_eq!(sum["area"], (100.0 + 100.0) * 0.25);
        assert_eq!(sum["scaleUnits"], "mm");
        assert_eq!(sum["source"], "Selection");
        // White background: gray max reads as white in the depth's range.
        let white = match depth {
            8 => 255.0,
            16 => 32768.0,
            _ => 1.0,
        };
        assert!((sum["grayMean"].as_f64().unwrap() - white).abs() < white * 0.01, "{depth}: {sum}");
        assert_eq!(rows[1]["values"]["label"], "Feature 1");
        assert_eq!(rows[2]["values"]["width"], 10.0);
        assert_eq!(rows[2]["values"]["height"], 2.5);
    }
    // Data points filter the columns.
    let mut s = session(20, 20, 8);
    s.execute("image.analysis.selectDataPoints", json!({"selection": ["label", "area"]})).unwrap();
    let r = s.execute("image.analysis.recordMeasurements", json!({"source": "selection"})).unwrap();
    let v = r["rows"][0]["values"].as_object().unwrap();
    assert_eq!(v.keys().collect::<Vec<_>>(), ["area", "label"]);
    assert_eq!(v["area"], 400.0, "no selection: the whole image");
    let r = s.execute("image.analysis.selectDataPoints", json!({"selection": {"perimeter": true, "label": false}})).unwrap();
    assert_eq!(r["dataPoints"]["selection"], json!(["area", "perimeter"]));
    assert!(s.execute("image.analysis.selectDataPoints", json!({"ruler": ["area"]})).is_err());
    s.execute("image.analysis.selectDataPoints", json!({"reset": true})).unwrap();
    assert_eq!(s.analysis.data_points, DataPoints::default());
}

#[test]
fn ruler_protractor_and_record() {
    let mut s = session(100, 100, 8);
    let r = s.execute("image.analysis.rulerTool", json!({"start": [10, 50], "end": [40, 10]})).unwrap();
    let i = &r["info"];
    assert_eq!(i["l1"], 50.0);
    assert_eq!(i["w"], 30.0);
    assert_eq!(i["h"], -40.0);
    assert!((i["angle"].as_f64().unwrap() - 53.1301).abs() < 1e-3);
    // The ruler doesn't add history and doesn't dirty the document.
    assert_eq!(s.active().unwrap().history.past_len(), 0);
    assert!(!s.active().unwrap().is_dirty());
    // Protractor: a second arm straight down from the vertex.
    let r = s.execute("image.analysis.rulerTool", json!({"protractor": [10, 90]})).unwrap();
    assert!((r["info"]["angle"].as_f64().unwrap() - (180.0 - 36.8699)).abs() < 1e-3, "{r}");
    assert_eq!(r["info"]["l2"], 40.0);
    s.execute("image.analysis.setMeasurementScale", json!({"pixelLength": 10, "logicalLength": 1, "units": "cm"})).unwrap();
    let r = s.execute("image.analysis.recordMeasurements", json!({})).unwrap();
    assert_eq!(r["source"], "ruler");
    assert_eq!(r["rows"][0]["values"]["length"], 5.0);
    assert!(s.execute("image.analysis.rulerTool", json!({"start": [1, 1]})).is_err());
    s.execute("image.analysis.rulerTool", json!({"clear": true})).unwrap();
    assert!(d(&s).measurement.ruler.is_none());
    assert!(!s.is_enabled("image.analysis.straightenLayer"));
}

#[test]
fn straighten_layer_and_image() {
    // A layer: rotated about the canvas centre, canvas unchanged.
    let mut s = session(80, 60, 8);
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("select.rect", json!({"x": 10, "y": 28, "width": 60, "height": 4})).unwrap();
    s.execute("edit.fill", json!({"color": "#000000"})).unwrap_or_default();
    s.execute("select.deselect", json!({})).unwrap();
    s.execute("image.analysis.rulerTool", json!({"start": [10, 30], "end": [70, 25]})).unwrap();
    let before = s.active().unwrap().history.past_len();
    let r = s.execute("image.analysis.straightenLayer", json!({})).unwrap();
    assert_eq!(r["cropped"], false);
    assert!((r["angle"].as_f64().unwrap() - 4.7636).abs() < 1e-3, "{r}");
    assert_eq!(s.active().unwrap().history.past_len(), before + 1, "one history step");
    assert_eq!(d(&s).size.width, 80);
    assert!(d(&s).measurement.ruler.is_none());
    // The Background: rotate the canvas and crop to the image.
    let mut s = session(80, 60, 16);
    s.execute("image.analysis.rulerTool", json!({"start": [0, 0], "end": [40, 40]})).unwrap();
    let r = s.execute("image.analysis.straightenLayer", json!({})).unwrap();
    assert_eq!(r["cropped"], true);
    assert_eq!(r["angle"].as_f64().unwrap().abs(), 45.0);
    let nd = d(&s);
    assert!(nd.size.width < 80 && nd.size.height < 60);
    assert!(s.undo());
    assert_eq!(d(&s).size.width, 80);
}

#[test]
fn inscribed_rect() {
    let (w, h) = inscribed(100.0, 50.0, 0.0);
    assert!((w - 100.0).abs() < 1e-9 && (h - 50.0).abs() < 1e-9);
    let (w, h) = inscribed(100.0, 100.0, std::f64::consts::FRAC_PI_4);
    assert!((w - 70.710678).abs() < 1e-4 && (h - 70.710678).abs() < 1e-4);
}

#[test]
fn count_groups_markers_and_undo() {
    let mut s = session(100, 100, 8);
    let r = s.execute("image.analysis.countTool", json!({})).unwrap();
    assert_eq!(r["total"], 0);
    let r = s.execute("count.add", json!({"x": 10, "y": 10})).unwrap();
    assert_eq!(r["number"], 1);
    assert_eq!(r["groups"][0]["name"], "Count Group 1");
    s.execute("count.add", json!({"x": 20, "y": 20})).unwrap();
    s.execute("count.newGroup", json!({"name": "Cells", "color": "#00ff00"})).unwrap();
    let r = s.execute("count.add", json!({"x": 50, "y": 50})).unwrap();
    assert_eq!(r["activeGroup"], 1);
    assert_eq!(r["groups"][1]["count"], 1);
    assert_eq!(r["total"], 3);
    // Remove by position, move by index.
    s.execute("count.remove", json!({"x": 21, "y": 19})).unwrap();
    assert_eq!(d(&s).measurement.count_groups[0].points, vec![[10.0, 10.0]]);
    s.execute("count.move", json!({"group": 0, "index": 0, "to": [12, 13]})).unwrap();
    assert_eq!(d(&s).measurement.count_groups[0].points[0], [12.0, 13.0]);
    assert!(s.execute("count.remove", json!({"x": 90, "y": 90})).is_err());
    s.execute("count.setGroup", json!({"group": 0, "markerSize": 50, "labelSize": 20, "visible": false, "name": "Spots"})).unwrap();
    let g = &d(&s).measurement.count_groups[0];
    assert_eq!((g.marker_size, g.label_size, g.visible, g.name.as_str()), (10, 20, false, "Spots"));
    // Record counts.
    let r = s.execute("image.analysis.recordMeasurements", json!({"source": "count"})).unwrap();
    assert_eq!(r["rows"][0]["values"]["count"], 2);
    s.execute("count.clear", json!({})).unwrap();
    assert_eq!(d(&s).measurement.count_total(), 1, "clears the active group only");
    s.execute("count.clear", json!({"group": "all"})).unwrap();
    assert_eq!(d(&s).measurement.count_total(), 0);
    assert!(s.undo());
    assert_eq!(d(&s).measurement.count_total(), 1);
    s.execute("count.deleteGroup", json!({"group": 1})).unwrap();
    assert_eq!(d(&s).measurement.count_groups.len(), 1);
    assert_eq!(d(&s).measurement.active_count_group, 0);
    s.execute("count.setGroup", json!({"group": 0})).unwrap();
    assert!(s.execute("count.setGroup", json!({"group": 7})).is_err());
    assert!(s.execute("count.add", json!({"x": 1})).is_err());
}

#[test]
fn log_delete_and_csv_export() {
    let mut s = session(10, 10, 8);
    assert!(!s.is_enabled("measurementLog.delete"));
    s.execute("image.analysis.recordMeasurements", json!({"dateTime": "t"})).unwrap();
    s.execute("image.analysis.rulerTool", json!({"start": [0, 0], "end": [3, 4]})).unwrap();
    s.execute("image.analysis.recordMeasurements", json!({"source": "ruler", "dateTime": "t"})).unwrap();
    let r = s.execute("measurementLog.list", json!({})).unwrap();
    assert_eq!(r["rows"].as_array().unwrap().len(), 2);
    let r = s.execute("measurementLog.export", json!({})).unwrap();
    let csv = r["csv"].as_str().unwrap();
    let mut lines = csv.lines();
    let header = lines.next().unwrap();
    assert!(header.starts_with("Label,Date and Time,Document,Source,Scale"));
    assert!(header.contains("Length") && header.contains("Gray Value (Mean)"));
    assert!(lines.next().unwrap().starts_with("Measurement 1,t,"));
    assert!(lines.next().unwrap().contains(",5.0,"), "{csv}");
    #[cfg(not(target_arch = "wasm32"))]
    {
        let path = std::env::temp_dir().join(format!("pc-log-{}.csv", std::process::id()));
        let r = s.execute("measurementLog.export", json!({"path": path.to_str().unwrap(), "rows": [2]})).unwrap();
        assert_eq!(r["rows"], 1);
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.lines().count(), 2);
        let _ = std::fs::remove_file(path);
    }
    s.execute("measurementLog.delete", json!({"rows": [1]})).unwrap();
    assert_eq!(s.analysis.log.len(), 1);
    assert!(s.execute("measurementLog.delete", json!({})).is_err());
    s.execute("measurementLog.delete", json!({"all": true})).unwrap();
    assert!(s.analysis.log.is_empty());
    assert_eq!(csv_field(&json!("a,\"b\"")), "\"a,\"\"b\"\"\"");
}

#[test]
fn place_scale_marker_builds_one_group_in_one_step() {
    let mut s = session(400, 300, 8);
    s.execute("image.analysis.setMeasurementScale", json!({"pixelLength": 50, "logicalLength": 1, "units": "cm"})).unwrap();
    let before = s.active().unwrap().history.past_len();
    let r = s.execute("image.analysis.placeScaleMarker", json!({"color": "black"})).unwrap();
    assert_eq!(s.active().unwrap().history.past_len(), before + 1);
    // Nice length near width/5 = 80 px = 1.6 cm → 1 cm = 50 px.
    assert_eq!(r["label"], "1 cm");
    assert_eq!(r["pixels"], 50.0);
    let doc = d(&s);
    let g = doc.layers.last().unwrap();
    assert_eq!(g.name, "Measurement Scale Marker");
    let kids = g.children().unwrap();
    assert_eq!(kids.len(), 2, "bar + text");
    let bar = kids.iter().find(|l| l.name == "Scale Bar").unwrap();
    let b = bar.surface().unwrap().content_bounds();
    assert_eq!(b.width(), 50);
    assert!(b.y1 < 300);
    // Without text, white, on top.
    let r = s.execute("image.analysis.placeScaleMarker", json!({"length": 2, "displayText": false, "color": "white", "textPosition": "top"})).unwrap();
    assert_eq!(r["pixels"], 100.0);
    assert!(r["text"].is_null());
    assert!(s.execute("image.analysis.placeScaleMarker", json!({"length": 100})).is_err(), "wider than the image");
    assert!(s.execute("image.analysis.placeScaleMarker", json!({"color": "red"})).is_err());
    assert!(s.undo());
    assert!(s.undo());
    assert!(d(&s).layers.iter().all(|l| l.name != "Measurement Scale Marker"));
}

#[test]
fn place_scale_marker_on_a_short_document_errors_without_panicking() {
    // A document under 8 px tall inverted the bar-height clamp and panicked (#932). Call `run`
    // directly too: dispatch would catch the panic and hide it behind a generic error.
    for (w, h, p) in [
        (100, 4, json!({})),
        (100, 1, json!({})),
        (100, 7, json!({"displayText": false, "textPosition": "top"})),
        (300_000, 3, json!({"fontSize": 1000})),
        (2, 7, json!({"length": 1})),
        (7, 7, json!({"length": 7, "fontSize": 1})),
    ] {
        for h in [h, 1, 2, 3, 5, 6] {
            let mut s = session(w, h, 8);
            let before = s.active().unwrap().history.past_len();
            let e = place_scale_marker(&mut s, &p).unwrap_err().to_string();
            assert!(e.contains("tall"), "{w}x{h}: {e}");
            let e = s.execute("image.analysis.placeScaleMarker", p.clone()).unwrap_err().to_string();
            assert!(!e.contains("internal error"), "{w}x{h}: {e}");
            assert_eq!(s.active().unwrap().history.past_len(), before, "{w}x{h}: nothing added");
            assert!(d(&s).layers.iter().all(|l| l.name != "Measurement Scale Marker"));
        }
    }
    // Control: 8 px is the shortest that fits, and tall, narrow or roomy documents still get a
    // marker that lies on the canvas.
    for (w, h, p) in [(100, 8, json!({})), (100, 100, json!({})), (20, 4000, json!({})), (2, 8, json!({"length": 2}))] {
        let mut s = session(w, h, 8);
        let r = place_scale_marker(&mut s, &p).unwrap();
        let b: Vec<i64> = r["barRect"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
        assert!(0 <= b[0] && b[0] < b[2] && b[2] <= i64::from(w), "{w}x{h}: bar on the canvas: {r}");
        assert!(0 <= b[1] && b[1] < b[3] && b[3] <= i64::from(h), "{w}x{h}: bar on the canvas: {r}");
        assert_eq!(s.active().unwrap().history.past_len(), 1);
        let doc = d(&s);
        let bar = doc.layers.last().unwrap().children().unwrap().iter().find(|l| l.name == "Scale Bar").unwrap();
        assert_eq!(f64::from(bar.surface().unwrap().content_bounds().width()), r["pixels"].as_f64().unwrap(), "{w}x{h}");
    }
}

#[test]
fn analysis_info_and_cmyk_document() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 30, "height": 20, "mode": "cmyk", "background": "white"})).unwrap();
    assert_eq!(d(&s).mode, ColorMode::Cmyk);
    s.execute("count.add", json!({"x": 3, "y": 4})).unwrap();
    let r = s.execute("image.analysis.recordMeasurements", json!({"source": "selection"})).unwrap();
    assert_eq!(r["rows"][0]["values"]["area"], 600.0);
    let r = s.execute("image.analysis.info", json!({})).unwrap();
    assert_eq!(r["count"]["total"], 1);
    assert_eq!(r["logRows"], 1);
}

#[test]
fn now_iso_shape() {
    let t = now_iso();
    assert_eq!(t.len(), 20, "{t}");
    assert!(t.ends_with('Z') && &t[4..5] == "-");
}

#[test]
fn count_set_group_takes_sizes_sent_as_whole_floats() {
    // The options bar sends `{"markerSize": 8.0, "labelSize": 20.0}`; `as_u64` read neither, so
    // the fields did nothing.
    let mut s = session(100, 100, 8);
    s.execute("count.add", json!({"x": 10, "y": 10})).unwrap();
    s.execute("count.setGroup", json!({"markerSize": 8.0, "labelSize": 20.0})).unwrap();
    let g = &d(&s).measurement.count_groups[0];
    assert_eq!((g.marker_size, g.label_size), (8, 20));
    // Out-of-range whole numbers clamp, also huge ones that `as u32` would have wrapped.
    s.execute("count.setGroup", json!({"markerSize": 4_294_967_297u64, "labelSize": 1e300})).unwrap();
    let g = &d(&s).measurement.count_groups[0];
    assert_eq!((g.marker_size, g.label_size), (10, 72));
}

#[test]
fn count_set_group_rejects_sizes_that_are_not_whole_numbers() {
    let mut s = session(100, 100, 8);
    s.execute("count.add", json!({"x": 10, "y": 10})).unwrap();
    s.execute("count.setGroup", json!({"markerSize": 3, "labelSize": 12})).unwrap();
    let past = s.active().unwrap().history.past_len();
    for p in [
        json!({"markerSize": 8.5}),
        json!({"labelSize": 12.25}),
        json!({"markerSize": -2}),
        json!({"labelSize": -9.0}),
        json!({"markerSize": "8"}),
        json!({"labelSize": true}),
    ] {
        let err = s.execute("count.setGroup", p.clone()).unwrap_err();
        assert!(matches!(err, EngineError::BadParams { .. }), "{p}: {err}");
    }
    let g = &d(&s).measurement.count_groups[0];
    assert_eq!((g.marker_size, g.label_size), (3, 12), "a rejected call changes nothing");
    assert_eq!(s.active().unwrap().history.past_len(), past);
}
