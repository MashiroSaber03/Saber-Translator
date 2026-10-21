//! M11 acceptance: an agent completes 10 scripted edit tasks over MCP (roadmap M11 DoD).
//!
//! Each task uses only MCP tools (`doc_*`, `command_run`) the way an agent would, and checks the
//! outcome through `doc_inspect`, `document.pixel` or the exported files, never engine internals.

use std::io::Write;

use photocraft_automation::{AuthorizedWorkspace, PhotocraftMcp};
use rmcp::model::{CallToolRequestParams, CallToolResult, ClientConfig};
use rmcp::service::RunningService;
use rmcp::{ClientHandler, RoleClient, ServiceExt};
use serde_json::{Value, json};

#[derive(Clone, Default)]
struct Client;
impl ClientHandler for Client {
    fn get_info(&self) -> ClientConfig {
        ClientConfig::default()
    }
}

type Conn = RunningService<RoleClient, Client>;

/// The server side of a test connection. It owns the headless session and its workspace,
/// which holds open directory handles on the test folder.
type Server = tokio::task::JoinHandle<()>;

async fn connect(root: &std::path::Path) -> (Conn, Server) {
    let (s, c) = tokio::io::duplex(1 << 20);
    let workspace = AuthorizedWorkspace::new(Some(root), Some(root)).expect("test workspace");
    let server = tokio::spawn(async move {
        if let Ok(running) = PhotocraftMcp::headless_with_workspace(workspace).serve(s).await {
            let _ = running.waiting().await;
        }
    });
    (Client.serve(c).await.expect("client init"), server)
}

/// Closes the client and waits for the server task to end, so the session and its workspace
/// (with their open directory handles) are dropped before the test removes its folder. Windows
/// refuses to delete a directory that still has open handles (os error 32).
async fn shutdown(c: Conn, server: Server) {
    c.cancel().await.unwrap();
    server.await.unwrap();
}

fn text(r: &CallToolResult) -> String {
    r.content.iter().filter_map(|c| c.as_text()).map(|t| t.text.clone()).collect::<Vec<_>>().join("\n")
}

/// Call a tool and parse its JSON result, failing the test on a tool error.
async fn tool(c: &Conn, name: &str, args: Value) -> Value {
    let mut p = CallToolRequestParams::new(name.to_owned());
    if let Value::Object(m) = args {
        p = p.with_arguments(m);
    }
    let r = c.call_tool(p).await.expect("call_tool transport");
    assert_ne!(r.is_error, Some(true), "{name} failed: {}", text(&r));
    serde_json::from_str(&text(&r)).unwrap_or(Value::String(text(&r)))
}

async fn run(c: &Conn, id: &str, params: Value) -> Value {
    tool(c, "command_run", json!({"id": id, "params": params})).await
}

async fn inspect(c: &Conn) -> Value {
    tool(c, "doc_inspect", json!({})).await
}

/// Composite RGBA at (x, y), whatever envelope `command_run` wraps the result in.
async fn pixel(c: &Conn, x: i32, y: i32) -> Vec<f64> {
    let v = run(c, "document.pixel", json!({"x": x, "y": y})).await;
    let arr = if v.is_array() { v } else { v.get("result").cloned().unwrap_or(v) };
    arr.as_array().unwrap_or_else(|| panic!("pixel: {arr}")).iter().map(|x| x.as_f64().unwrap()).collect()
}

fn top_layer(doc: &Value) -> &Value {
    &doc["layers"][0]
}

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("pc-agent-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A 64×48 RGB gradient PNG (red ramps across, green down).
fn gradient_png(dir: &std::path::Path) -> &'static str {
    let (w, h) = (64u32, 48u32);
    let mut px = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            px.extend_from_slice(&[(x * 4) as u8, (y * 5) as u8, 128]);
        }
    }
    let img = photocraft_codecs::Image::from_u8(w, h, photocraft_codecs::ChannelLayout::Rgb, px).unwrap();
    let path = dir.join("gradient.png");
    std::fs::File::create(&path).unwrap().write_all(&photocraft_codecs::encode(&img, photocraft_codecs::Format::Png, &Default::default()).unwrap()).unwrap();
    "gradient.png"
}

fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

#[tokio::test(flavor = "multi_thread")]
async fn agent_completes_ten_scripted_tasks() {
    let dir = tmp("tasks");
    let png = gradient_png(&dir);
    let (c, server) = connect(&dir).await;

    // 1. Make a title card: type layer with a drop shadow, exported as PNG.
    tool(&c, "doc_new", json!({"width": 200, "height": 120, "background": "white"})).await;
    run(&c, "type.create", json!({"x": 20, "y": 70, "text": "Hello", "size": 36})).await;
    run(&c, "layer.layerStyle.dropShadow", json!({"distance": 4, "size": 3})).await;
    let doc = inspect(&c).await;
    let t = top_layer(&doc);
    assert_eq!(t["kind"], "Type", "task 1: {t}");
    assert_eq!(t["text"]["text"], "Hello", "task 1: {t}");
    assert_eq!(t["effects"]["items"][0]["kind"], "Drop Shadow", "task 1: {t}");
    let out = dir.join("card.png");
    tool(&c, "doc_export", json!({"path": "card.png"})).await;
    let card = photocraft_codecs::decode(&std::fs::read(&out).unwrap()).unwrap();
    assert_eq!(card.dimensions(), (200, 120), "task 1");
    tool(&c, "doc_close", json!({})).await;

    // 2. Colour grade a photo non-destructively: Curves brightens the midtones, Vibrance on top.
    tool(&c, "doc_open", json!({"path": png})).await;
    let before = pixel(&c, 16, 24).await;
    run(&c, "layer.newAdjustmentLayer.curves", json!({"points": [[0, 0], [128, 200], [255, 255]]})).await;
    run(&c, "layer.newAdjustmentLayer.vibrance", json!({"vibrance": 40, "saturation": 0})).await;
    let after = pixel(&c, 16, 24).await;
    assert!(after[1] > before[1] + 0.05, "task 2: curves should brighten green {before:?} -> {after:?}");
    let doc = inspect(&c).await;
    let kinds: Vec<&str> = doc["layers"].as_array().unwrap().iter().filter_map(|l| l["kind"].as_str()).collect();
    assert_eq!(kinds, ["Adjustment", "Adjustment", "Pixel"], "task 2");

    // 3. Undo/redo: remove both adjustments, restore one.
    run(&c, "edit.undo", json!({})).await;
    run(&c, "edit.undo", json!({})).await;
    assert!(close(&pixel(&c, 16, 24).await, &before, 1e-6), "task 3: undo should restore the original");
    run(&c, "edit.redo", json!({})).await;
    assert_eq!(inspect(&c).await["layers"].as_array().unwrap().len(), 2, "task 3: redo restores Curves");
    run(&c, "edit.undo", json!({})).await;

    // 4. Editable blur: smart object + Gaussian Blur, retune the radius, then hide the filter.
    run(&c, "layer.smartObjects.convertToSmartObject", json!({})).await;
    run(&c, "filter.blur.gaussianBlur", json!({"radius": 2})).await;
    run(&c, "layer.smartFilter.setParams", json!({"index": 0, "params": {"radius": 6}})).await;
    let t = inspect(&c).await;
    let sf = &top_layer(&t)["smartFilters"][0];
    assert_eq!(sf["command"], "filter.blur.gaussianBlur", "task 4: {t}");
    assert_eq!(sf["params"]["radius"].as_f64(), Some(6.0), "task 4: {sf}");
    run(&c, "layer.smartFilter.setVisible", json!({"index": 0, "visible": false})).await;
    assert!(close(&pixel(&c, 16, 24).await, &before, 1.0 / 255.0), "task 4: hiding the filter restores the pixels");

    // 5. Selection → layer mask: an elliptical reveal on a red layer.
    run(&c, "layer.new.layer", json!({"name": "Red"})).await;
    run(&c, "select.all", json!({})).await;
    run(&c, "edit.fill", json!({"color": "#ff0000"})).await;
    run(&c, "select.rect", json!({"x": 16, "y": 12, "width": 32, "height": 24, "ellipse": true})).await;
    run(&c, "layer.layerMask.revealSelection", json!({})).await;
    assert_eq!(top_layer(&inspect(&c).await)["hasMask"], true, "task 5");
    let inside = pixel(&c, 32, 24).await;
    let outside = pixel(&c, 1, 1).await;
    assert!(inside[0] > 0.99 && inside[1] < 0.01, "task 5: centre is red {inside:?}");
    assert!(outside[0] < 0.1, "task 5: corner shows the photo {outside:?}");
    assert_eq!(inspect(&c).await["hasSelection"], false, "task 5: the selection became the mask");

    // 6. Save a selection to a channel and load it back.
    run(&c, "select.rect", json!({"x": 4, "y": 6, "width": 20, "height": 10})).await;
    run(&c, "select.saveSelection", json!({"name": "Keep"})).await;
    run(&c, "select.deselect", json!({})).await;
    assert_eq!(inspect(&c).await["hasSelection"], false, "task 6");
    run(&c, "select.loadSelection", json!({"channel": "Keep"})).await;
    let doc = inspect(&c).await;
    assert_eq!(doc["selectionBounds"], json!([4, 6, 20, 10]), "task 6: {}", doc["selectionBounds"]);
    assert_eq!(doc["channels"]["alpha"][0]["name"], "Keep", "task 6");
    run(&c, "select.deselect", json!({})).await;
    tool(&c, "doc_close", json!({})).await;

    // 7. Lay out three swatches and align their left edges.
    tool(&c, "doc_new", json!({"width": 120, "height": 90, "background": "white"})).await;
    for (i, x) in [10, 40, 70].into_iter().enumerate() {
        run(&c, "layer.new.layer", json!({"name": format!("S{i}")})).await;
        run(&c, "select.rect", json!({"x": x, "y": 10 + 25 * i as i32, "width": 20, "height": 15})).await;
        run(&c, "edit.fill", json!({"color": "#3366cc"})).await;
    }
    run(&c, "select.deselect", json!({})).await;
    run(&c, "select.allLayers", json!({})).await;
    run(&c, "layer.align.leftEdges", json!({})).await;
    let doc = inspect(&c).await;
    let xs: Vec<i64> = doc["layers"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|l| l["name"].as_str().is_some_and(|n| n.starts_with('S')))
        .map(|l| l["bounds"][0].as_i64().unwrap())
        .collect();
    assert_eq!(xs, [10, 10, 10], "task 7: {}", doc["layers"]);

    // 8. Export through the capability-scoped document API.
    let layers_dir = dir.join("layers");
    std::fs::create_dir_all(&layers_dir).unwrap();
    tool(&c, "doc_export", json!({"path": "layers/swatch.png"})).await;
    let n = std::fs::read_dir(&layers_dir).unwrap().filter(|e| e.as_ref().is_ok_and(|e| e.path().extension().is_some_and(|x| x == "png"))).count();
    assert_eq!(n, 1, "task 8: capability-scoped export");

    // 9. Resize, then crop to a square.
    run(&c, "image.imageSize", json!({"width": 60, "height": 45})).await;
    run(&c, "image.crop", json!({"x": 5, "y": 0, "width": 45, "height": 45})).await;
    let doc = inspect(&c).await;
    assert_eq!((doc["width"].as_u64(), doc["height"].as_u64()), (Some(45), Some(45)), "task 9");

    // 10. Prepare for print: convert to CMYK and save natively, then reopen.
    run(&c, "image.mode.cmyk", json!({"intent": "perceptual"})).await;
    assert_eq!(inspect(&c).await["mode"], "Cmyk", "task 10");
    tool(&c, "doc_save", json!({"path": "print.pcraft"})).await;
    tool(&c, "doc_close", json!({})).await;
    tool(&c, "doc_open", json!({"path": "print.pcraft"})).await;
    let doc = inspect(&c).await;
    assert_eq!((doc["mode"].as_str(), doc["width"].as_u64()), (Some("Cmyk"), Some(45)), "task 10");
    assert_eq!(doc["layers"].as_array().unwrap().len(), 4, "task 10");

    shutdown(c, server).await;
    std::fs::remove_dir_all(dir).unwrap();
}

/// Click just after "Big", insert a word, then bold through the end of the line. Geometry comes
/// from `type.hitTest` / `type.caret` / `type.navigate`; the edit is the existing `type.edit`.
#[tokio::test(flavor = "multi_thread")]
async fn agent_edits_text_by_position() {
    let dir = tmp("type-caret");
    let (c, server) = connect(&dir).await;
    tool(&c, "doc_new", json!({"width": 480, "height": 200, "background": "white"})).await;
    let listed = tool(&c, "command_list", json!({"filter": "type."})).await;
    let ids: Vec<&str> = listed.as_array().unwrap().iter().filter_map(|c| c["id"].as_str()).collect();
    for id in ["type.hitTest", "type.caret", "type.navigate"] {
        assert!(ids.contains(&id), "command_list missing {id}: {ids:?}");
    }

    run(&c, "type.create", json!({"x": 40, "y": 90, "text": "Big Sale", "size": 48})).await;
    let before = inspect(&c).await["history"].as_array().unwrap().len();
    // The caret after "Big" and the caret after the following space. A quarter of the way is
    // the left half of that space, which hit-tests as the caret after "Big".
    let after_big = run(&c, "type.caret", json!({"index": 3})).await;
    let after_space = run(&c, "type.caret", json!({"index": 4})).await;
    let seg = after_big["segment"].as_array().unwrap();
    let (x3, y0, y1) = (seg[0][0].as_f64().unwrap(), seg[0][1].as_f64().unwrap(), seg[1][1].as_f64().unwrap());
    let x4 = after_space["segment"][0][0].as_f64().unwrap();
    assert!(x4 > x3 + 0.5, "space should have width, caret x {x3} -> {x4}");
    let hit = run(&c, "type.hitTest", json!({"x": x3 + 0.25 * (x4 - x3), "y": (y0 + y1) / 2.0})).await;
    assert_eq!(hit["index"].as_u64(), Some(3), "{hit}");
    assert_eq!(hit["inside"], true, "{hit}");
    let quiet = inspect(&c).await["history"].as_array().unwrap().len();
    assert_eq!(quiet, before, "caret queries are not history steps");

    run(&c, "type.edit", json!({"replace": {"start": hit["index"], "end": hit["index"], "text": " Summer"}})).await;
    let info = run(&c, "type.info", json!({})).await;
    assert_eq!(info["text"], "Big Summer Sale", "{info}");

    let end = run(&c, "type.navigate", json!({"index": 0, "move": "lineEnd"})).await;
    let n = info["text"].as_str().unwrap().chars().count() as u64;
    assert_eq!(end["index"].as_u64(), Some(n), "{end}");
    // "Summer" starts one character after the insertion caret (the space in " Summer").
    let summer = hit["index"].as_u64().unwrap() + 1;
    run(&c, "type.setStyle", json!({"range": [summer, end["index"]], "fauxBold": true})).await;
    let info = run(&c, "type.info", json!({})).await;
    assert_eq!(info["text"], "Big Summer Sale");
    let bold = info["runs"].as_array().unwrap().iter().find(|r| r["style"]["faux_bold"] == true);
    let bold = bold.unwrap_or_else(|| panic!("no bold run in {info}"));
    assert_eq!(bold["start"].as_u64(), Some(summer), "{bold}");
    assert_eq!(bold["end"].as_u64(), end["index"].as_u64(), "{bold}");

    tool(&c, "doc_close", json!({})).await;
    shutdown(c, server).await;
    std::fs::remove_dir_all(dir).unwrap();
}

async fn connect_headless(headless: photocraft_automation::Headless) -> (Conn, Server) {
    let (s, c) = tokio::io::duplex(1 << 20);
    let server = tokio::spawn(async move {
        let backend = photocraft_automation::Backend::Headless(std::sync::Arc::new(std::sync::Mutex::new(headless)));
        if let Ok(running) = PhotocraftMcp::with_backend(backend).serve(s).await {
            let _ = running.waiting().await;
        }
    });
    (Client.serve(c).await.expect("client init"), server)
}

#[tokio::test(flavor = "multi_thread")]
async fn agent_records_and_replays_an_action() {
    let dir = tmp("actions");
    let (c, server) = connect(&dir).await;
    let listed = tool(&c, "command_list", json!({"filter": "actions."})).await;
    let ids: Vec<&str> = listed.as_array().unwrap().iter().filter_map(|item| item.get("id").and_then(Value::as_str)).collect();
    for id in ["actions.list", "actions.get", "actions.record", "actions.stop", "actions.play", "actions.delete"] {
        assert!(ids.contains(&id), "{id} missing from {ids:?}");
    }

    run(&c, "file.new", json!({"width": 32, "height": 32, "background": "white"})).await;
    run(&c, "actions.record", json!({"name": "Red"})).await;
    run(&c, "layer.new.layer", json!({})).await;
    run(&c, "select.rect", json!({"x": 0, "y": 0, "width": 8, "height": 8})).await;
    run(&c, "edit.fill", json!({"color": "#ff0000"})).await;
    let stopped = run(&c, "actions.stop", json!({})).await;
    assert_eq!(stopped["steps"], 3, "{stopped}");
    let got = run(&c, "actions.get", json!({"action": "Red"})).await;
    assert_eq!(got["steps"][0][0], "layer.new.layer");
    assert_eq!(got["steps"][1][0], "select.rect");
    assert_eq!(got["steps"][2][0], "edit.fill");

    run(&c, "file.new", json!({"width": 32, "height": 32, "background": "white"})).await;
    let played = run(&c, "actions.play", json!({"action": "Red"})).await;
    assert_eq!(played["ran"], 3, "{played}");
    assert!(played.get("failed").is_none(), "{played}");
    let px = pixel(&c, 2, 2).await;
    assert!(close(&px, &[1.0, 0.0, 0.0, 1.0], 1e-4), "{px:?}");

    shutdown(c, server).await;
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn agent_play_refuses_a_recorded_file_open() {
    let dir = tmp("actions-auth");
    let workspace = AuthorizedWorkspace::new(Some(&dir), Some(&dir)).expect("test workspace");
    let mut headless = photocraft_automation::Headless::with_workspace(workspace);
    headless.session.actions.list.push(photocraft_engine::actions_cmds::Action {
        name: "Open".into(),
        steps: vec![("file.open".into(), json!({"path": "/etc/passwd"})), ("layer.new.layer".into(), json!({}))],
        disabled: Default::default(),
        ..Default::default()
    });
    let (c, server) = connect_headless(headless).await;
    let played = run(&c, "actions.play", json!({"action": "Open"})).await;
    assert_eq!(played["ran"], 0, "{played}");
    assert_eq!(played["failed"]["step"], 0, "{played}");
    assert_eq!(played["failed"]["id"], "file.open", "{played}");
    assert!(played["failed"]["error"].as_str().unwrap_or("").contains("disabled"), "{played}");
    let session = tool(&c, "session_list", json!({})).await;
    assert!(session["documents"].as_array().is_some_and(|d| d.is_empty()), "{session}");
    shutdown(c, server).await;
    std::fs::remove_dir_all(dir).unwrap();
}

/// The next ten acceptance tasks (11–20): strokes, inverse fills, duplication, merges,
/// flattening, adjustments, feathering, expand, renaming/deleting and an export/reopen round
/// trip — all through MCP tools and checked from outside the engine.
#[tokio::test(flavor = "multi_thread")]
async fn agent_completes_ten_more_scripted_tasks() {
    let dir = tmp("tasks2");
    let (c, server) = connect(&dir).await;

    // 11. Stroke a selection: a red square with a black outside stroke.
    tool(&c, "doc_new", json!({"width": 80, "height": 80, "background": "white"})).await;
    run(&c, "select.rect", json!({"x": 20, "y": 20, "width": 40, "height": 40})).await;
    run(&c, "edit.fill", json!({"color": "#ff0000"})).await;
    run(&c, "edit.stroke", json!({"width": 4, "color": "#000000", "location": "outside"})).await;
    let inside = pixel(&c, 40, 40).await;
    let edge = pixel(&c, 18, 40).await;
    let away = pixel(&c, 5, 5).await;
    assert!(inside[0] > 0.99 && inside[1] < 0.01, "task 11: the centre is red {inside:?}");
    assert!(edge[0] < 0.1 && edge[1] < 0.1, "task 11: the stroke is black {edge:?}");
    assert!(away[0] > 0.99 && away[1] > 0.99, "task 11: outside stays white {away:?}");
    run(&c, "select.deselect", json!({})).await;

    // 12. Inverse the selection: a blue frame around the red square.
    run(&c, "select.rect", json!({"x": 20, "y": 20, "width": 40, "height": 40})).await;
    run(&c, "select.inverse", json!({})).await;
    run(&c, "edit.fill", json!({"color": "#0000ff"})).await;
    let centre = pixel(&c, 40, 40).await;
    let frame = pixel(&c, 4, 4).await;
    assert!(centre[0] > 0.99 && centre[2] < 0.01, "task 12: the centre keeps red {centre:?}");
    assert!(frame[2] > 0.99 && frame[0] < 0.01, "task 12: the frame is blue {frame:?}");
    run(&c, "select.deselect", json!({})).await;

    // 13. Paint a green patch on a new layer and duplicate it (offset, as Photoshop's default).
    run(&c, "layer.new.layer", json!({"name": "Patch"})).await;
    run(&c, "select.rect", json!({"x": 10, "y": 10, "width": 20, "height": 20})).await;
    run(&c, "edit.fill", json!({"color": "#00ff00"})).await;
    run(&c, "select.deselect", json!({})).await;
    run(&c, "layer.duplicate", json!({})).await;
    let doc = inspect(&c).await;
    assert_eq!(doc["layers"].as_array().unwrap().len(), 3, "task 13: {doc}");
    let patches = doc["layers"].as_array().unwrap().iter().filter(|l| l["name"].as_str().is_some_and(|n| n.starts_with("Patch"))).count();
    assert_eq!(patches, 2, "task 13: {doc}");

    // 14. Merge the copy down; the two patches become one layer.
    run(&c, "layer.mergeDown", json!({})).await;
    assert_eq!(inspect(&c).await["layers"].as_array().unwrap().len(), 2, "task 14");

    // 15. Flatten the image; the patch survives.
    run(&c, "layer.flattenImage", json!({})).await;
    let doc = inspect(&c).await;
    assert_eq!(doc["layers"].as_array().unwrap().len(), 1, "task 15: {doc}");
    let green = pixel(&c, 20, 20).await;
    assert!(green[1] > 0.99 && green[0] < 0.01, "task 15: the patch survives {green:?}");

    // 16. A Hue/Saturation adjustment layer turns the red square green without touching pixels.
    run(&c, "layer.newAdjustmentLayer.hueSaturation", json!({"hue": 120})).await;
    assert_eq!(top_layer(&inspect(&c).await)["kind"], "Adjustment", "task 16");
    let shifted = pixel(&c, 40, 40).await;
    assert!(shifted[1] > 0.5 && shifted[0] < 0.5, "task 16: red turns green {shifted:?}");
    run(&c, "layer.delete", json!({})).await;

    // 17. Feather the selection and reveal a mask through it: the edge blends.
    run(&c, "select.rect", json!({"x": 20, "y": 20, "width": 40, "height": 40})).await;
    run(&c, "select.modify.feather", json!({"radius": 4})).await;
    run(&c, "layer.layerMask.revealSelection", json!({})).await;
    let doc = inspect(&c).await;
    assert_eq!(top_layer(&doc)["hasMask"], true, "task 17");
    let boundary = pixel(&c, 20, 40).await;
    assert!(boundary[3] > 0.05 && boundary[3] < 0.95, "task 17: the feathered edge keeps partial alpha {boundary:?}");
    assert_eq!(doc["hasSelection"], false, "task 17: the selection became the mask");

    // 18. Expand a selection and read its bounds back.
    run(&c, "select.rect", json!({"x": 20, "y": 20, "width": 20, "height": 20})).await;
    run(&c, "select.modify.expand", json!({"radius": 5})).await;
    let bounds = inspect(&c).await["selectionBounds"].clone();
    let b = bounds.as_array().unwrap_or_else(|| panic!("task 18: {bounds}"));
    let (x, w) = (b[0].as_i64().unwrap(), b[2].as_i64().unwrap());
    assert!((13..=15).contains(&x) && (30..=32).contains(&w), "task 18: expanded outward {bounds}");
    run(&c, "select.deselect", json!({})).await;

    // 19. Rename a layer and delete it again.
    run(&c, "layer.new.layer", json!({"name": "Scratch"})).await;
    run(&c, "layer.renameLayer", json!({"name": "Delivery"})).await;
    assert_eq!(top_layer(&inspect(&c).await)["name"], "Delivery", "task 19");
    run(&c, "layer.delete", json!({})).await;
    assert_eq!(inspect(&c).await["layers"].as_array().unwrap().len(), 1, "task 19: deleted");

    // 20. Export the final image and reopen the PNG as a flat document.
    tool(&c, "doc_export", json!({"path": "final.png"})).await;
    tool(&c, "doc_close", json!({})).await;
    tool(&c, "doc_open", json!({"path": "final.png"})).await;
    let doc = inspect(&c).await;
    assert_eq!((doc["width"].as_u64(), doc["height"].as_u64()), (Some(80), Some(80)), "task 20");
    assert_eq!(doc["layers"].as_array().unwrap().len(), 1, "task 20: a flat reopen");
    tool(&c, "doc_close", json!({})).await;

    shutdown(c, server).await;
    std::fs::remove_dir_all(dir).unwrap();
}
