//! M11 acceptance, part three: six rounding-out tasks (31–36) over MCP (roadmap M11 DoD).
//!
//! Same contract as the other acceptance files: only MCP tools (`doc_*`, `command_run`), checked
//! through `doc_inspect`, `document.pixel` and exported/saved files.

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

async fn shutdown(c: Conn, server: Server) {
    c.cancel().await.unwrap();
    server.await.unwrap();
}

fn text(r: &CallToolResult) -> String {
    r.content.iter().filter_map(|c| c.as_text()).map(|t| t.text.clone()).collect::<Vec<_>>().join("\n")
}

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

/// Six more acceptance tasks (31–36): an undo/redo sweep, a painted barcode, a grayscale
/// conversion with an export check, a native save round trip that preserves type, a layer-style
/// stroke and inverse fills.
#[tokio::test(flavor = "multi_thread")]
async fn agent_completes_six_rounding_out_tasks() {
    let dir = tmp("tasks4");
    let (c, server) = connect(&dir).await;

    // 31. Undo/redo sweep: four fills go away and come back.
    tool(&c, "doc_new", json!({"width": 40, "height": 40, "background": "white"})).await;
    let h0 = inspect(&c).await["history"].as_array().unwrap().len();
    let colors = ["#ff0000", "#00ff00", "#0000ff", "#000000"];
    for (i, color) in colors.iter().enumerate() {
        let x = 10 * i as i32;
        run(&c, "select.rect", json!({"x": x, "y": 0, "width": 10, "height": 10})).await;
        run(&c, "edit.fill", json!({"color": color})).await;
    }
    run(&c, "select.deselect", json!({})).await;
    let steps = inspect(&c).await["history"].as_array().unwrap().len() - h0;
    assert!(steps >= 4, "at least the four fills are history steps: {steps}");
    for _ in 0..steps {
        run(&c, "edit.undo", json!({})).await;
    }
    for i in 0..4 {
        let x = 10 * i + 5;
        let p = pixel(&c, x, 5).await;
        assert!(p[0] > 0.99 && p[1] > 0.99 && p[2] > 0.99, "task 31: fill {i} undone {p:?}");
    }
    for _ in 0..steps {
        run(&c, "edit.redo", json!({})).await;
    }
    let red = pixel(&c, 5, 5).await;
    let black = pixel(&c, 35, 5).await;
    assert!(red[0] > 0.99 && red[1] < 0.01, "task 31: the first fill redid {red:?}");
    assert!(black[0] < 0.01 && black[1] < 0.01, "task 31: the last fill redid {black:?}");

    // 32. Paint a barcode: five black bars on white.
    tool(&c, "doc_new", json!({"width": 48, "height": 24, "background": "white"})).await;
    for i in 0..5 {
        let x = 4 + 8 * i;
        run(&c, "select.rect", json!({"x": x, "y": 4, "width": 3, "height": 16})).await;
        run(&c, "edit.fill", json!({"color": "#000000"})).await;
    }
    run(&c, "select.deselect", json!({})).await;
    for i in 0..5 {
        let bar = pixel(&c, 5 + 8 * i, 12).await;
        let gap = pixel(&c, 9 + 8 * i, 12).await;
        assert!(bar[0] < 0.01, "task 32: bar {i} is black {bar:?}");
        assert!(gap[0] > 0.99, "task 32: gap {i} is white {gap:?}");
    }

    // 33. Convert to grayscale: the channels flatten and the export keeps them equal.
    run(&c, "image.mode.grayscale", json!({})).await;
    assert_eq!(inspect(&c).await["mode"], "Grayscale", "task 33");
    let g = pixel(&c, 5, 12).await;
    assert!((g[0] - g[1]).abs() < 1e-4 && (g[1] - g[2]).abs() < 1e-4, "task 33: the channels agree {g:?}");
    tool(&c, "doc_export", json!({"path": "bars.png"})).await;
    let img = photocraft_codecs::decode(&std::fs::read(dir.join("bars.png")).unwrap()).unwrap();
    let px = img.to_rgba8();
    let at = |x: usize, y: usize| px[(y * img.width() as usize + x) * 4];
    assert!(at(5, 12) < 8 && at(9, 12) > 247, "task 33: the grayscale export matches the canvas");
    tool(&c, "doc_close", json!({})).await;

    // 34. A native save round trip preserves a styled type layer.
    tool(&c, "doc_new", json!({"width": 160, "height": 80, "background": "white"})).await;
    run(&c, "type.create", json!({"x": 16, "y": 48, "text": "Round", "size": 24, "color": "#3366cc"})).await;
    tool(&c, "doc_save", json!({"path": "type.pcraft"})).await;
    tool(&c, "doc_close", json!({})).await;
    tool(&c, "doc_open", json!({"path": "type.pcraft"})).await;
    let doc = inspect(&c).await;
    assert_eq!(top_layer(&doc)["kind"], "Type", "task 34: {doc}");
    assert_eq!(top_layer(&doc)["text"]["text"], "Round", "task 34: {doc}");
    tool(&c, "doc_close", json!({})).await;

    // 35. A layer-style stroke frames a swatch.
    tool(&c, "doc_new", json!({"width": 48, "height": 48, "background": "white"})).await;
    run(&c, "layer.new.layer", json!({"name": "Swatch"})).await;
    run(&c, "select.rect", json!({"x": 12, "y": 12, "width": 24, "height": 24})).await;
    run(&c, "edit.fill", json!({"color": "#ff0000"})).await;
    run(&c, "select.deselect", json!({})).await;
    run(&c, "layer.layerStyle.stroke", json!({"size": 3, "position": "outside", "color": "#0000ff"})).await;
    let doc = inspect(&c).await;
    assert_eq!(top_layer(&doc)["effects"]["items"][0]["kind"], "Stroke", "task 35: {doc}");
    let border = pixel(&c, 11, 24).await;
    let centre = pixel(&c, 24, 24).await;
    assert!(border[2] > 0.5 && border[0] < 0.5, "task 35: the stroke is blue {border:?}");
    assert!(centre[0] > 0.99 && centre[2] < 0.01, "task 35: the centre stays red {centre:?}");

    // 36. Inverse fill: gray around the swatch, then Select All paints everything white.
    run(&c, "select.rect", json!({"x": 12, "y": 12, "width": 24, "height": 24})).await;
    run(&c, "select.inverse", json!({})).await;
    run(&c, "edit.fill", json!({"color": "#808080"})).await;
    run(&c, "select.deselect", json!({})).await;
    let corner = pixel(&c, 2, 2).await;
    let centre = pixel(&c, 24, 24).await;
    assert!(corner[0] > 0.4 && corner[0] < 0.6 && (corner[0] - corner[1]).abs() < 0.01, "task 36: the inverse fill is gray {corner:?}");
    assert!(centre[0] > 0.99 && centre[2] < 0.01, "task 36: the swatch stays red {centre:?}");
    run(&c, "select.all", json!({})).await;
    run(&c, "edit.fill", json!({"color": "#ffffff"})).await;
    run(&c, "select.deselect", json!({})).await;
    let white = pixel(&c, 2, 2).await;
    assert!(white[0] > 0.99 && white[1] > 0.99 && white[2] > 0.99, "task 36: everything is white {white:?}");
    tool(&c, "doc_close", json!({})).await;

    shutdown(c, server).await;
    std::fs::remove_dir_all(dir).unwrap();
}
