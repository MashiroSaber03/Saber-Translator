//! Spin the MCP server in-process (tokio duplex), call tools as a client.

use std::io::Write;

use base64::Engine as _;
use photocraft_automation::{AuthorizedWorkspace, PhotocraftMcp};
use rmcp::model::{CallToolRequestParams, CallToolResult, ClientConfig};
use rmcp::service::RunningService;
use rmcp::{ClientHandler, RoleClient, ServiceExt};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

const CONTROL_TOKEN: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

#[derive(Clone, Default)]
struct Client;
impl ClientHandler for Client {
    fn get_info(&self) -> ClientConfig {
        ClientConfig::default()
    }
}

async fn connect(server: PhotocraftMcp) -> RunningService<RoleClient, Client> {
    let (s, c) = tokio::io::duplex(1 << 20);
    tokio::spawn(async move {
        if let Ok(running) = server.serve(s).await {
            let _ = running.waiting().await;
        }
    });
    Client.serve(c).await.expect("client init")
}

async fn call(client: &RunningService<RoleClient, Client>, name: &str, args: Value) -> CallToolResult {
    let mut p = CallToolRequestParams::new(name.to_owned());
    if let Value::Object(m) = args {
        p = p.with_arguments(m);
    }
    client.call_tool(p).await.expect("call_tool transport")
}

fn text(r: &CallToolResult) -> String {
    r.content.iter().filter_map(|c| c.as_text()).map(|t| t.text.clone()).collect::<Vec<_>>().join("\n")
}

fn json_of(r: &CallToolResult) -> Value {
    assert_ne!(r.is_error, Some(true), "tool error: {}", text(r));
    serde_json::from_str(&text(r)).unwrap_or_else(|e| panic!("not JSON ({e}): {}", text(r)))
}

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("pc-mcp-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn cleanup(path: &std::path::Path) {
    for _ in 0..20 {
        match std::fs::remove_dir_all(path) {
            Ok(()) => return,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return,
            Err(_) => std::thread::sleep(std::time::Duration::from_millis(10)),
        }
    }
    std::fs::remove_dir_all(path).expect("remove test workspace after server shutdown");
}

fn headless_in(root: &std::path::Path) -> PhotocraftMcp {
    let workspace = AuthorizedWorkspace::new(Some(root), Some(root)).expect("test workspace");
    PhotocraftMcp::headless_with_workspace(workspace)
}

#[tokio::test(flavor = "multi_thread")]
async fn lists_expected_tools() {
    let client = connect(PhotocraftMcp::headless()).await;
    let tools = client.list_all_tools().await.unwrap();
    let names: Vec<String> = tools.iter().map(|t| t.name.to_string()).collect();
    for n in [
        "session_list",
        "doc_open",
        "doc_new",
        "doc_place",
        "doc_export_layers",
        "doc_export_data_sets",
        "doc_import_data_sets",
        "doc_save",
        "doc_export",
        "doc_inspect",
        "doc_render_preview",
        "doc_select",
        "doc_close",
        "command_list",
        "command_run",
        "command_batch",
        "ui_inspect",
        "ui_screenshot",
        "ui_pointer",
        "ui_menu_invoke",
        "ui_set",
        "control_call",
    ] {
        assert!(names.contains(&n.to_string()), "missing tool {n}: {names:?}");
    }
    for t in &tools {
        assert!(t.name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'), "tool name `{}`", t.name);
        assert!(t.description.as_ref().is_some_and(|d| !d.is_empty()));
    }
    let info = client.peer_info().expect("server info");
    assert_eq!(info.server_info.as_ref().unwrap().name, "photocraft");
    assert!(info.instructions.as_ref().is_some_and(|i| i.contains("command_list")));
    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn headless_edit_render_save_roundtrip() {
    let dir = tmp("edit");
    let client = connect(headless_in(&dir)).await;

    let r = call(&client, "doc_new", json!({"width": 64, "height": 48, "background": "white", "name": "Agent"})).await;
    assert_ne!(r.is_error, Some(true), "{}", text(&r));
    let r = call(&client, "command_run", json!({"id": "layer.new.layer", "params": {"name": "Ink"}})).await;
    assert_ne!(r.is_error, Some(true), "{}", text(&r));
    let r =
        call(&client, "command_run", json!({"id": "paint.stroke", "params": {"points": [[5, 5, 1.0], [50, 40, 1.0]], "size": 6, "color": "#ff0000"}})).await;
    assert_ne!(r.is_error, Some(true), "{}", text(&r));

    let doc = json_of(&call(&client, "doc_inspect", json!({})).await);
    assert_eq!(doc["width"], 64);
    let names: Vec<&str> = doc["layers"].as_array().unwrap().iter().filter_map(|l| l["name"].as_str()).collect();
    assert_eq!(names, ["Ink", "Background"]);

    let r = call(&client, "doc_render_preview", json!({"max_side": 32})).await;
    let img = r.content.iter().find_map(|c| c.as_image()).expect("image content");
    assert_eq!(img.mime_type, "image/png");
    let png = base64::engine::general_purpose::STANDARD.decode(&img.data).unwrap();
    let decoded = photocraft_codecs::decode(&png).unwrap();
    assert_eq!(decoded.dimensions(), (32, 24));

    let r = json_of(&call(&client, "doc_save", json!({"path": "agent.pcraft"})).await);
    assert_eq!(r["path"], "agent.pcraft");
    let png_path = dir.join("agent.png");
    json_of(&call(&client, "doc_export", json!({"path": "agent.png"})).await);
    let jpg_path = dir.join("agent.jpg");
    json_of(&call(&client, "doc_export", json!({"path": "agent.jpg", "quality": 70})).await);
    assert!(photocraft_codecs::decode(&std::fs::read(&png_path).unwrap()).is_ok());
    assert!(std::fs::metadata(&jpg_path).unwrap().len() > 100);

    // Re-open the native file: identical layer tree.
    let o = json_of(&call(&client, "doc_open", json!({"path": "agent.pcraft"})).await);
    assert_eq!(o["index"], 1);
    let doc2 = json_of(&call(&client, "doc_inspect", json!({"index": 1})).await);
    assert_eq!(doc2["layers"].as_array().unwrap().len(), 2);
    let sess = json_of(&call(&client, "session_list", json!({})).await);
    assert_eq!(sess["documents"].as_array().unwrap().len(), 2);
    json_of(&call(&client, "doc_close", json!({"index": 1})).await);
    json_of(&call(&client, "doc_select", json!({"index": 0})).await);
    client.cancel().await.unwrap();
    cleanup(&dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn command_list_filters() {
    let client = connect(PhotocraftMcp::headless()).await;
    let all = json_of(&call(&client, "command_list", json!({})).await);
    let n_all = all.as_array().unwrap().len();
    assert!(n_all > 20);
    let blur = json_of(&call(&client, "command_list", json!({"filter": "blur"})).await);
    assert!(!blur.as_array().unwrap().is_empty() && blur.as_array().unwrap().len() < n_all);
    assert!(blur.as_array().unwrap().iter().all(|c| c["params"].is_string()));
    let enabled = json_of(&call(&client, "command_list", json!({"enabled_only": true})).await);
    assert!(enabled.as_array().unwrap().iter().any(|c| c["id"] == "file.new"));
    assert!(!enabled.as_array().unwrap().iter().any(|c| c["id"] == "layer.new.layer"), "needs a document");
    // The full list says why a document command is off with no document open (#3664).
    let layer = all.as_array().unwrap().iter().find(|c| c["id"] == "layer.new.layer").unwrap();
    assert_eq!((layer["enabled"].clone(), layer["needsDocument"].clone()), (json!(false), json!(true)), "{layer}");
    assert!(layer["disabledReason"].as_str().is_some_and(|r| r.starts_with("no document")), "{layer}");
    let new = all.as_array().unwrap().iter().find(|c| c["id"] == "file.new").unwrap();
    assert_eq!((new["needsDocument"].clone(), new["disabledReason"].clone()), (json!(false), Value::Null), "{new}");
    client.cancel().await.unwrap();
}

/// Constructs strict tool-schema validators reject: a `$ref` / `$defs`, or a boolean in a schema
/// position (`items: true`), anywhere in `schema`.
fn strict_schema_problems(path: &str, schema: &Value, out: &mut Vec<String>) {
    let Value::Object(m) = schema else {
        if schema.is_boolean() {
            out.push(format!("{path}: boolean schema {schema}"));
        }
        return;
    };
    for key in ["$ref", "$defs", "definitions"] {
        if m.contains_key(key) {
            out.push(format!("{path}: {key}"));
        }
    }
    if let Some(Value::Object(props)) = m.get("properties") {
        for (k, v) in props {
            strict_schema_problems(&format!("{path}.properties.{k}"), v, out);
        }
    }
    for key in ["items", "not"] {
        if let Some(v) = m.get(key) {
            strict_schema_problems(&format!("{path}.{key}"), v, out);
        }
    }
    for key in ["anyOf", "oneOf", "allOf", "prefixItems"] {
        for (i, v) in m.get(key).and_then(Value::as_array).into_iter().flatten().enumerate() {
            strict_schema_problems(&format!("{path}.{key}[{i}]"), v, out);
        }
    }
    if let Some(v) = m.get("additionalProperties").filter(|v| v.is_object()) {
        strict_schema_problems(&format!("{path}.additionalProperties"), v, out);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn tool_schemas_pass_strict_validators() {
    // #1782: `ui_pointer`'s `items: true` and `command_batch`'s `$ref`/`$defs` made strict
    // providers (e.g. Moonshot via Volcano Ark) reject the whole request.
    let client = connect(PhotocraftMcp::headless()).await;
    let tools = client.list_all_tools().await.unwrap();
    let mut problems = Vec::new();
    for t in &tools {
        strict_schema_problems(&t.name, &Value::Object((*t.input_schema).clone()), &mut problems);
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
    let schema = |name: &str| Value::Object((*tools.iter().find(|t| t.name == name).unwrap().input_schema).clone());
    // The inlined shapes still describe the arguments.
    let event = &schema("ui_pointer")["properties"]["events"]["items"];
    assert_eq!(event["type"], "object", "{event}");
    for k in ["kind", "x", "y", "pressure"] {
        assert!(event["properties"][k].is_object(), "ui_pointer event field {k}: {event}");
    }
    let step = &schema("command_batch")["properties"]["steps"]["items"];
    assert!(step["properties"]["id"]["description"].as_str().is_some_and(|d| d.contains("layer.new.layer")), "{step}");
    client.cancel().await.unwrap();
}

/// The backticked examples in every `id` property description of a schema.
fn id_examples(schema: &Value, out: &mut Vec<String>) {
    match schema {
        Value::Object(m) => {
            if let Some(d) = m.get("properties").and_then(|p| p["id"]["description"].as_str()) {
                out.extend(d.split('`').skip(1).step_by(2).map(str::to_owned));
            }
            m.values().for_each(|v| id_examples(v, out));
        }
        Value::Array(a) => a.iter().for_each(|v| id_examples(v, out)),
        _ => {}
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn command_ids_in_tool_schemas_exist() {
    // Agents copy these examples verbatim (#378).
    let client = connect(PhotocraftMcp::headless()).await;
    let all = json_of(&call(&client, "command_list", json!({})).await);
    let ids: Vec<&str> = all.as_array().unwrap().iter().filter_map(|c| c["id"].as_str()).collect();
    let tools = client.list_all_tools().await.unwrap();
    for name in ["command_run", "command_batch"] {
        let tool = tools.iter().find(|t| t.name == name).unwrap();
        let mut examples = Vec::new();
        id_examples(&Value::Object((*tool.input_schema).clone()), &mut examples);
        assert!(!examples.is_empty(), "{name}: no example ids");
        for id in examples {
            assert!(ids.contains(&id.as_str()), "{name}: `{id}` is not a registered command");
        }
    }
    client.cancel().await.unwrap();
}

/// `UI_SET_FIELDS` in `crates/ui-egui/src/control.rs`, read from source. The desktop shell owns the
/// list; this crate describes it over MCP without depending on the UI crate (both are L6).
/// Anchored on the declaration, not on the doc comment that mentions it first, and checked against
/// the declared array length so a mis-parse fails loudly instead of testing the wrong names.
fn ui_set_fields() -> Vec<String> {
    let src = std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../ui-egui/src/control.rs")).expect("ui-egui control.rs");
    let (_, rest) = src.split_once("UI_SET_FIELDS: [&str; ").expect("UI_SET_FIELDS is declared");
    let declared: usize = rest.split_once(']').expect("declared length").0.parse().expect("declared length");
    let body = rest.split_once('[').expect("field array").1.split_once("];").expect("terminated array").0;
    let fields: Vec<String> = body.split('"').skip(1).step_by(2).map(str::to_string).collect();
    assert_eq!(fields.len(), declared, "parsed {fields:?} from UI_SET_FIELDS; its source shape changed");
    fields
}

/// #1548: `ui_set` must name every field `ui.set` accepts. An agent copies this list out of
/// `tools/list`, and it ends "Other fields are an error", so an omission reads as a rejection of
/// a field that in fact works. This guards the whole list against future drift, not only the
/// three fields missing when the issue was filed.
#[tokio::test(flavor = "multi_thread")]
async fn ui_set_description_names_every_accepted_field() {
    let client = connect(PhotocraftMcp::headless()).await;
    let tools = client.list_all_tools().await.unwrap();
    let tool = tools.iter().find(|t| t.name == "ui_set").expect("ui_set tool");
    let schema = Value::Object((*tool.input_schema).clone());
    let description = schema["properties"]["fields"]["description"].as_str().expect("ui_set fields description").to_string();
    for field in ui_set_fields() {
        assert!(description.contains(&field), "ui_set's description omits `{field}`, which ui.set accepts:\n{description}");
    }
    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn errors_are_tool_errors_not_crashes() {
    let client = connect(PhotocraftMcp::headless()).await;
    let r = call(&client, "command_run", json!({"id": "no.such.command"})).await;
    assert_eq!(r.is_error, Some(true));
    assert!(text(&r).contains("unknown command"));
    let r = call(&client, "doc_inspect", json!({"index": 0})).await;
    assert_eq!(r.is_error, Some(true));
    let r = call(&client, "doc_open", json!({"path": "/definitely/missing.png"})).await;
    assert_eq!(r.is_error, Some(true));
    let r = call(&client, "ui_inspect", json!({})).await;
    assert_eq!(r.is_error, Some(true));
    assert!(text(&r).contains("bridge"));
    let r = call(&client, "doc_export", json!({})).await;
    assert_eq!(r.is_error, Some(true));
    // Server still alive.
    json_of(&call(&client, "session_list", json!({})).await);
    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn preview_budget_failure_preserves_the_mcp_session() {
    let client = connect(PhotocraftMcp::headless()).await;
    json_of(&call(&client, "doc_new", json!({"width": 2049, "height": 1})).await);
    for side in [0, 2049] {
        let reply = call(&client, "doc_render_preview", json!({"max_side": side})).await;
        assert_eq!(reply.is_error, Some(true));
        assert!(text(&reply).contains("preview side exceeds"));
    }
    let preview = call(&client, "doc_render_preview", json!({"max_side": 32})).await;
    assert!(preview.content.iter().any(|content| content.as_image().is_some()));
    let inspected = json_of(&call(&client, "doc_inspect", json!({})).await);
    assert_eq!(inspected["width"], 2049);
    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn bridge_doc_save_rejects_headless_only_options_instead_of_saving_with_defaults() {
    // Bridge mode forwards only the path to the running app: quality/format/tiffLayers/index
    // must be refused up front (no bridge connection needed for that) rather than silently
    // saving with the app's current settings.
    let server = PhotocraftMcp::bridge("127.0.0.1:1", &"a".repeat(64)).unwrap();
    let client = connect(server).await;
    for params in [
        json!({"path": "out.jpg", "quality": 50}),
        json!({"path": "out.png", "format": "png"}),
        json!({"path": "out.tif", "tiffLayers": true}),
        json!({"path": "out.pcraft", "index": 0}),
    ] {
        let r = call(&client, "doc_save", params.clone()).await;
        assert_eq!(r.is_error, Some(true), "{params}: {:?}", text(&r));
        assert!(text(&r).contains("headless"), "{params}: {}", text(&r));
    }
}

/// #946: like headless mode, a bridged `doc_save` without `path` reaches the app's `app.save`,
/// which writes back to the document's own PSD, PSB or .pcraft file or refuses with its own error.
#[tokio::test(flavor = "multi_thread")]
async fn bridge_doc_save_without_path_forwards_app_save_for_write_back() {
    let (addr, app) = fake_app().await;
    let client = connect(PhotocraftMcp::bridge(&addr, CONTROL_TOKEN).unwrap()).await;
    let saved = json_of(&call(&client, "doc_save", json!({})).await);
    assert_eq!(saved, json!({"saved": {}}), "no `path` is sent, not even null");
    let saved = json_of(&call(&client, "doc_save", json!({"path": "out.psd"})).await);
    assert_eq!(saved, json!({"saved": {"path": "out.psd"}}));
    client.cancel().await.unwrap();
    app.abort();
}

/// `tgaBits` reaches the app's `app.save` through the bridge; a value other than 24 or 32 is
/// refused before anything is sent.
#[tokio::test(flavor = "multi_thread")]
async fn bridge_doc_save_forwards_targa_bits() {
    let (addr, app) = fake_app().await;
    let client = connect(PhotocraftMcp::bridge(&addr, CONTROL_TOKEN).unwrap()).await;
    let saved = json_of(&call(&client, "doc_save", json!({"path": "out.tga", "tgaBits": 24})).await);
    assert_eq!(saved, json!({"saved": {"path": "out.tga", "tgaBits": 24}}));
    let r = call(&client, "doc_save", json!({"path": "out.tga", "tgaBits": 16})).await;
    assert_eq!(r.is_error, Some(true), "{}", text(&r));
    assert!(text(&r).contains("24 or 32"), "{}", text(&r));
    client.cancel().await.unwrap();
    app.abort();
}

/// Headless `doc_save`/`doc_export` write the TGA bits per pixel asked for (byte 16 of the
/// header); unset, a document with an alpha channel is written as 32-bit.
#[tokio::test(flavor = "multi_thread")]
async fn headless_doc_save_writes_targa_bits() {
    let dir = tmp("tga-bits");
    let client = connect(headless_in(&dir)).await;
    call(&client, "doc_new", json!({"width": 16, "height": 8, "background": "white"})).await;
    for id in ["select.all", "select.saveSelection"] {
        let r = call(&client, "command_run", json!({"id": id, "params": {}})).await;
        assert_ne!(r.is_error, Some(true), "{}", text(&r));
    }
    for (tool, params, bpp) in [
        ("doc_export", json!({"path": "auto.tga"}), 32),
        ("doc_export", json!({"path": "flat.tga", "tgaBits": 24}), 24),
        ("doc_save", json!({"path": "alpha.tga", "tgaBits": 32}), 32),
    ] {
        let r = call(&client, tool, params.clone()).await;
        assert_ne!(r.is_error, Some(true), "{params}: {}", text(&r));
        let bytes = std::fs::read(dir.join(params["path"].as_str().unwrap())).unwrap();
        assert_eq!(bytes[16], bpp, "{params}");
    }
    let r = call(&client, "doc_export", json!({"path": "bad.tga", "tgaBits": 8})).await;
    assert_eq!(r.is_error, Some(true), "{}", text(&r));
    assert!(!dir.join("bad.tga").exists());
    client.cancel().await.unwrap();
    cleanup(&dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn bridge_response_budget_drops_connection_without_retrying_the_operation() {
    use photocraft_automation::{BridgeClient, budgets::MAX_RESPONSE_BYTES};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let app = tokio::spawn(async move {
        let (sock, _) = listener.accept().await.unwrap();
        let (read, mut write) = sock.into_split();
        let mut reader = BufReader::new(read);
        let mut line = String::new();
        reader.read_line(&mut line).await.unwrap();
        let auth: Value = serde_json::from_str(&line).unwrap();
        let reply = json!({"id": auth["id"], "ok": true, "result": {"authenticated": true}});
        write.write_all(format!("{reply}\n").as_bytes()).await.unwrap();
        line.clear();
        reader.read_line(&mut line).await.unwrap();
        let request: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(request["method"], "engine.execute");
        let mut oversized = vec![b'x'; MAX_RESPONSE_BYTES];
        // The read limit cuts this multi-byte character in half. Classification must
        // still be a non-retryable budget error, not a UTF-8 transport failure.
        oversized.extend_from_slice("é".as_bytes());
        write.write_all(&oversized).await.unwrap();
        // A retry would connect to this still-live listener. The budget error must return
        // directly instead, without waiting for another authentication exchange.
        let next = tokio::time::timeout(std::time::Duration::from_millis(250), listener.accept()).await;
        assert!(next.is_err(), "oversized reply must not retry the operation");
    });
    let bridge = BridgeClient::new(&addr, CONTROL_TOKEN).unwrap();
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), bridge.call("engine.execute", json!({"command": "command.list"}))).await.unwrap();
    assert!(result.unwrap_err().to_string().contains("bridge response exceeds"));
    app.await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn open_png_and_inspect() {
    let dir = tmp("open");
    let img = photocraft_codecs::Image::from_u8(8, 4, photocraft_codecs::ChannelLayout::Rgb, vec![200; 96]).unwrap();
    let path = dir.join("in.png");
    std::fs::File::create(&path).unwrap().write_all(&photocraft_codecs::encode(&img, photocraft_codecs::Format::Png, &Default::default()).unwrap()).unwrap();
    let client = connect(headless_in(&dir)).await;
    let o = json_of(&call(&client, "doc_open", json!({"path": "in.png"})).await);
    assert_eq!((o["width"].as_u64(), o["height"].as_u64()), (Some(8), Some(4)));
    let px = json_of(&call(&client, "command_run", json!({"id": "document.pixel", "params": {"x": 1, "y": 1}})).await);
    assert!(px.to_string().contains("0.78"), "{px}");
    client.cancel().await.unwrap();
    cleanup(&dir);
}

/// #518, #523: `doc_open` returns why an opened image is incomplete.
#[tokio::test(flavor = "multi_thread")]
async fn open_reports_decode_warnings() {
    let dir = tmp("open-warnings");
    let jpeg = write_image(&dir, "whole.jpg", photocraft_codecs::Format::Jpeg);
    std::fs::write(dir.join("cut.jpg"), &jpeg[..jpeg.len() - 4]).unwrap();
    // Two 1x1 frames.
    let mut gif = b"GIF89a\x01\0\x01\0\x80\0\0\0\0\0\xFF\xFF\xFF".to_vec();
    for _ in 0..2 {
        gif.extend_from_slice(b"\x2C\0\0\0\0\x01\0\x01\0\0\x02\x02\x44\x01\0");
    }
    gif.push(0x3B);
    std::fs::write(dir.join("anim.gif"), gif).unwrap();
    let client = connect(headless_in(&dir)).await;
    let warnings = async |path: &str| json_of(&call(&client, "doc_open", json!({"path": path})).await)["warnings"].clone();
    assert_eq!(warnings("whole.jpg").await, json!([]));
    assert_eq!(warnings("cut.jpg").await, json!(["JPEG data ends early (the file is truncated or damaged); part of the image is missing"]));
    assert_eq!(warnings("anim.gif").await, json!(["only the first of 2 frames was imported"]));
    client.cancel().await.unwrap();
    cleanup(&dir);
}

fn write_image(dir: &std::path::Path, name: &str, format: photocraft_codecs::Format) -> Vec<u8> {
    let img = photocraft_codecs::Image::from_u8(16, 8, photocraft_codecs::ChannelLayout::Rgb, (0..384).map(|i| (i * 7 % 251) as u8).collect()).unwrap();
    let bytes = photocraft_codecs::encode(&img, format, &Default::default()).unwrap();
    std::fs::write(dir.join(name), &bytes).unwrap();
    bytes
}

#[tokio::test(flavor = "multi_thread")]
async fn affinity_preview_warns_and_requires_a_new_save_path() {
    let dir = tmp("affinity-preview");
    let png = write_image(&dir, "preview.png", photocraft_codecs::Format::Png);
    // Synthetic v12 envelope; the native document graph is intentionally absent.
    let mut bytes = vec![0; 72];
    bytes[..4].copy_from_slice(b"\x00\xffKA");
    bytes[4..6].copy_from_slice(&12u16.to_le_bytes());
    bytes[8..12].copy_from_slice(b"nsrP");
    bytes[12..16].copy_from_slice(b"#Inf");
    bytes[24..32].copy_from_slice(&72u64.to_le_bytes());
    bytes[64..68].copy_from_slice(b"Prot");
    bytes.extend(b"\xff\xff\xff\xffThmb");
    bytes.extend(1u32.to_le_bytes());
    bytes.extend((png.len() as u32 + 13).to_le_bytes());
    bytes.extend(29u32.to_le_bytes());
    bytes.extend(0u32.to_le_bytes());
    bytes.extend((png.len() as u32).to_le_bytes());
    bytes.push(1);
    bytes.extend(png);
    std::fs::write(dir.join("source.af"), &bytes).unwrap();
    // A renamed layered extension must not bypass source protection.
    std::fs::write(dir.join("renamed.psd"), &bytes).unwrap();
    let client = connect(headless_in(&dir)).await;
    for name in ["source.af", "renamed.psd"] {
        let opened = json_of(&call(&client, "doc_open", json!({"path": name})).await);
        assert!(opened["warnings"].to_string().contains("only its embedded 16×8 PNG preview"));
        assert_eq!(opened["width"], 16);
        let refused = call(&client, "doc_save", json!({})).await;
        assert_eq!(refused.is_error, Some(true), "{}", text(&refused));
        assert!(text(&refused).contains("pass `path`"), "{}", text(&refused));
        let native = call(&client, "doc_save", json!({"path": "source.af"})).await;
        assert_eq!(native.is_error, Some(true));
        assert!(text(&native).contains("Affinity export"));
        assert_eq!(std::fs::read(dir.join(name)).unwrap(), bytes);
    }
    json_of(&call(&client, "doc_save", json!({"path": "copy.pcraft"})).await);
    assert_eq!(json_of(&call(&client, "doc_save", json!({})).await)["path"], "copy.pcraft");
    assert_eq!(std::fs::read(dir.join("source.af")).unwrap(), bytes);
    client.cancel().await.unwrap();
    cleanup(&dir);
}

/// A save without `path` writes back only to a layered file in its own format (#416).
#[tokio::test(flavor = "multi_thread")]
async fn save_without_path_never_flattens_over_the_opened_file() {
    let dir = tmp("save-in-place");
    let png = write_image(&dir, "seed.png", photocraft_codecs::Format::Png);
    let jpg = write_image(&dir, "seed.jpg", photocraft_codecs::Format::Jpeg);
    let client = connect(headless_in(&dir)).await;
    let refused = |r: &CallToolResult| r.is_error == Some(true) && text(r).contains("pass `path`");

    // A flat file, edited or not, is left unchanged.
    json_of(&call(&client, "doc_open", json!({"path": "seed.png"})).await);
    json_of(&call(&client, "command_run", json!({"id": "layer.newAdjustmentLayer.curves", "params": {"points": [[0, 0], [128, 170], [255, 255]]}})).await);
    let r = call(&client, "doc_save", json!({})).await;
    assert!(refused(&r), "{}", text(&r));
    assert_eq!(std::fs::read(dir.join("seed.png")).unwrap(), png);
    json_of(&call(&client, "doc_open", json!({"path": "seed.jpg"})).await);
    let r = call(&client, "doc_save", json!({})).await;
    assert!(refused(&r), "{}", text(&r));
    assert_eq!(std::fs::read(dir.join("seed.jpg")).unwrap(), jpg);

    // An explicit path, even the opened file's own, still writes and reports what was lost.
    let r = json_of(&call(&client, "doc_save", json!({"path": "seed.jpg"})).await);
    assert!(r["warnings"].to_string().contains("lossy"), "{r}");

    // A layered file saves in place in its own format, but not converted over itself.
    json_of(&call(&client, "doc_select", json!({"index": 0})).await);
    json_of(&call(&client, "doc_save", json!({"path": "layered.psd"})).await);
    json_of(&call(&client, "doc_open", json!({"path": "layered.psd"})).await);
    let psd = std::fs::read(dir.join("layered.psd")).unwrap();
    let r = call(&client, "doc_save", json!({"format": "png"})).await;
    assert!(refused(&r), "{}", text(&r));
    assert_eq!(std::fs::read(dir.join("layered.psd")).unwrap(), psd);
    assert_eq!(json_of(&call(&client, "doc_save", json!({})).await)["path"], "layered.psd");
    assert!(photocraft_io::is_psd(&std::fs::read(dir.join("layered.psd")).unwrap()));

    // Once saved as .pcraft, a flat-born document saves in place there.
    json_of(&call(&client, "doc_select", json!({"index": 0})).await);
    json_of(&call(&client, "doc_save", json!({"path": "work.pcraft"})).await);
    assert_eq!(json_of(&call(&client, "doc_save", json!({})).await)["path"], "work.pcraft");
    assert_eq!(std::fs::read(dir.join("seed.png")).unwrap(), png);

    // A template opens untitled, without its path.
    std::fs::write(dir.join("card.psdt"), &psd).unwrap();
    assert_eq!(json_of(&call(&client, "doc_open", json!({"path": "card.psdt"})).await)["name"], "Untitled-1");
    let r = call(&client, "doc_save", json!({})).await;
    assert!(refused(&r), "{}", text(&r));
    assert_eq!(std::fs::read(dir.join("card.psdt")).unwrap(), psd);
    client.cancel().await.unwrap();
    cleanup(&dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn filesystem_policy_rejects_absolute_and_escaping_paths_before_effects() {
    let base = tmp("filesystem-policy");
    let root = base.join("workspace");
    let outside = base.join("outside");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::create_dir_all(&outside).unwrap();

    let img = photocraft_codecs::Image::from_u8(4, 3, photocraft_codecs::ChannelLayout::Rgb, vec![42; 36]).unwrap();
    let bytes = photocraft_codecs::encode(&img, photocraft_codecs::Format::Png, &Default::default()).unwrap();
    std::fs::write(root.join("inside.png"), bytes).unwrap();
    let client = connect(headless_in(&root)).await;

    assert_ne!(call(&client, "doc_open", json!({"path": "inside.png"})).await.is_error, Some(true));
    // An absolute path beneath the root is the same file (#2176).
    let inside = call(&client, "doc_open", json!({"path": root.join("inside.png").to_string_lossy()})).await;
    assert_ne!(inside.is_error, Some(true), "{}", text(&inside));
    let absolute = outside.join("absolute.png");
    let response = call(&client, "doc_save", json!({"path": absolute.to_string_lossy()})).await;
    assert_eq!(response.is_error, Some(true));
    assert!(text(&response).contains("automation path rejected"), "{}", text(&response));
    assert!(!absolute.exists());

    for path in ["../outside/traversal.png", "..\\outside\\mixed.png", "C:/outside/prefix.png", ""] {
        let response = call(&client, "doc_save", json!({"path": path})).await;
        assert_eq!(response.is_error, Some(true), "path {path:?}: {}", text(&response));
    }
    assert!(!outside.join("traversal.png").exists());
    assert!(!outside.join("mixed.png").exists());

    let response = call(&client, "command_run", json!({"id": "layer.smartObjects.exportContents", "params": {"path": "outside.bin"}})).await;
    assert_eq!(response.is_error, Some(true));
    assert!(text(&response).contains("ambient filesystem paths"), "{}", text(&response));
    client.cancel().await.unwrap();
    cleanup(&base);
}

/// #2810: a file from the read root places as a Smart Object layer; `file.placeEmbedded` itself
/// stays refused, so `doc_place` is the only way in and the root is still the scope.
#[tokio::test(flavor = "multi_thread")]
async fn doc_place_is_capability_scoped() {
    let base = tmp("place");
    let root = base.join("workspace");
    let outside = base.join("outside");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::create_dir_all(&outside).unwrap();
    let img = photocraft_codecs::Image::from_u8(8, 6, photocraft_codecs::ChannelLayout::Rgb, vec![42; 144]).unwrap();
    let bytes = photocraft_codecs::encode(&img, photocraft_codecs::Format::Png, &Default::default()).unwrap();
    std::fs::write(root.join("inside.png"), &bytes).unwrap();
    std::fs::write(outside.join("outside.png"), bytes).unwrap();
    let client = connect(headless_in(&root)).await;

    json_of(&call(&client, "doc_new", json!({"width": 40, "height": 30})).await);
    let placed = call(&client, "doc_place", json!({"path": "inside.png", "center": [10, 10]})).await;
    assert_ne!(placed.is_error, Some(true), "{}", text(&placed));
    let layers = json_of(&call(&client, "doc_inspect", json!({})).await)["layers"].as_array().unwrap().len();
    assert_eq!(layers, 2, "the placed layer is in the document");
    // An absolute path beneath the root is the same file (#2176), so it places too; outside it,
    // nothing.
    let inside = call(&client, "doc_place", json!({"path": root.join("inside.png").to_string_lossy()})).await;
    assert_ne!(inside.is_error, Some(true), "{}", text(&inside));
    let placed = json_of(&call(&client, "doc_inspect", json!({})).await)["layers"].as_array().unwrap().len();
    assert_eq!(placed, 3, "the absolute path beneath the root places as well");
    let refused = call(&client, "doc_place", json!({"path": outside.join("outside.png").to_string_lossy()})).await;
    assert_eq!(refused.is_error, Some(true));
    assert!(text(&refused).contains("must be inside the automation root"), "{}", text(&refused));
    let after = json_of(&call(&client, "doc_inspect", json!({})).await)["layers"].as_array().unwrap().len();
    assert_eq!(after, placed, "nothing outside the root was placed");
    // The ambient command keeps its refusal: the capability-scoped method is the only way in.
    let ambient = call(&client, "command_run", json!({"id": "file.placeEmbedded", "params": {"path": "inside.png"}})).await;
    assert_eq!(ambient.is_error, Some(true));
    assert!(text(&ambient).contains("ambient filesystem paths"), "{}", text(&ambient));
    client.cancel().await.unwrap();
    cleanup(&base);
}

/// #3216: the export half of #2810. The write root scopes both exports, the read root the CSV,
/// and the three ambient commands keep their refusal.
#[tokio::test(flavor = "multi_thread")]
async fn doc_exports_and_data_set_import_are_capability_scoped() {
    let base = tmp("export");
    let root = base.join("workspace");
    std::fs::create_dir_all(&root).unwrap();
    let client = connect(headless_in(&root)).await;

    json_of(&call(&client, "doc_new", json!({"width": 10, "height": 6, "background": "white", "name": "card"})).await);
    let inspected = json_of(&call(&client, "doc_inspect", json!({})).await);
    let layer = inspected["layers"][0]["id"].clone();
    json_of(
        &call(&client, "command_run", json!({"id": "image.variables.define", "params": {"defs": [{"name": "show", "layer": layer, "type": "visibility"}]}}))
            .await,
    );
    json_of(
        &call(
            &client,
            "command_run",
            json!({"id": "image.variables.dataSets", "params": {"dataSets": [
                {"name": "one", "values": [{"variable": "show", "kind": "visibility", "value": true}]},
                {"name": "two", "values": [{"variable": "show", "kind": "visibility", "value": false}]},
            ]}}),
        )
        .await,
    );

    let layers = call(&client, "doc_export_layers", json!({"dir": "layers", "prefix": "card"})).await;
    assert_ne!(layers.is_error, Some(true), "{}", text(&layers));
    assert_eq!(json_of(&layers)["count"], 1, "{}", text(&layers));
    assert_eq!(std::fs::read_dir(root.join("layers")).unwrap().count(), 1);
    let sets = call(&client, "doc_export_data_sets", json!({"dir": "sets", "format": "png"})).await;
    assert_ne!(sets.is_error, Some(true), "{}", text(&sets));
    assert_eq!(json_of(&sets)["count"], 2, "{}", text(&sets));
    assert_eq!(std::fs::read_dir(root.join("sets")).unwrap().count(), 2);
    // A colliding template is refused before any file of the batch is written.
    let clash = call(&client, "doc_export_data_sets", json!({"dir": "clash", "naming": "{document}"})).await;
    assert_eq!(clash.is_error, Some(true), "{}", text(&clash));
    assert_eq!(std::fs::read_dir(root.join("clash")).unwrap().count(), 0);
    // A CSV from the read root.
    std::fs::write(root.join("sets.csv"), "Data Set,show\none,true\n").unwrap();
    let imported = call(&client, "doc_import_data_sets", json!({"path": "sets.csv"})).await;
    assert_ne!(imported.is_error, Some(true), "{}", text(&imported));
    assert_eq!(json_of(&imported)["imported"], 1, "{}", text(&imported));

    // Outside the roots nothing happens at all.
    let escaped = call(&client, "doc_export_layers", json!({"dir": base.join("escaped").to_string_lossy()})).await;
    assert_eq!(escaped.is_error, Some(true), "{}", text(&escaped));
    assert!(!base.join("escaped").exists());
    std::fs::write(base.join("outside.csv"), "Data Set,show\none,true\n").unwrap();
    let outside = call(&client, "doc_import_data_sets", json!({"path": base.join("outside.csv").to_string_lossy()})).await;
    assert_eq!(outside.is_error, Some(true), "{}", text(&outside));
    for command in ["file.export.layersToFiles", "file.export.dataSetsAsFiles", "file.import.variableDataSets"] {
        let refused = call(&client, "command_run", json!({"id": command, "params": {}})).await;
        assert_eq!(refused.is_error, Some(true), "{command}");
        assert!(text(&refused).contains("ambient filesystem paths"), "{}", text(&refused));
    }
    client.cancel().await.unwrap();
    cleanup(&base);
}

/// The template workflow stays in memory, including saving nested contents back to the parent.
#[tokio::test(flavor = "multi_thread")]
async fn embedded_smart_objects_can_be_edited_saved_and_unpacked_without_file_authority() {
    let client = connect(PhotocraftMcp::headless()).await;
    let run = async |id: &str, params: Value| json_of(&call(&client, "command_run", json!({"id": id, "params": params})).await);
    json_of(&call(&client, "doc_new", json!({"width": 16, "height": 12, "background": "white"})).await);
    run("layer.smartObjects.convertToSmartObject", json!({})).await;
    let outer = run("layer.smartObjects.convertToSmartObject", json!({})).await["layer"].clone();
    // An explicit target works even with a regular pixel layer selected.
    run("layer.new.layer", json!({"name": "Other layer"})).await;
    assert_eq!(run("layer.smartObjects.editContents", json!({"layer": outer})).await["document"], 1);
    assert_eq!(run("layer.smartObjects.editContents", json!({})).await["document"], 2);
    run("layer.setProps", json!({"name": "Edited photo", "opacity": 0.5})).await;
    assert_eq!(run("layer.smartObjects.saveContents", json!({})).await["updated"], true);
    json_of(&call(&client, "doc_close", json!({})).await);
    run("layer.smartObjects.saveContents", json!({})).await;
    json_of(&call(&client, "doc_close", json!({})).await);
    assert_eq!(json_of(&call(&client, "session_list", json!({})).await)["documents"].as_array().unwrap().len(), 1);

    // Reopening the saved nested source proves both in-memory parent updates survived closing.
    run("layer.smartObjects.editContents", json!({"layer": outer})).await;
    run("layer.smartObjects.editContents", json!({})).await;
    let doc = json_of(&call(&client, "doc_inspect", json!({})).await);
    assert_eq!(doc["layers"][0]["name"], "Edited photo");
    assert_eq!(doc["layers"][0]["opacity"], 0.5);
    json_of(&call(&client, "doc_close", json!({})).await);
    json_of(&call(&client, "doc_close", json!({})).await);
    let unpacked = run("layer.smartObjects.convertToLayers", json!({"layer": outer})).await["layer"].clone();
    run("layer.smartObjects.convertToLayers", json!({"layer": unpacked})).await;
    let doc = json_of(&call(&client, "doc_inspect", json!({})).await);
    assert!(doc["layers"].as_array().unwrap().iter().all(|l| l["kind"] == "Pixel"));
    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn smart_object_commands_refuse_real_linked_files_without_changing_the_session() {
    use photocraft_engine::doc::{LayerContent, SmartSource};

    let dir = tmp("linked-smart-object");
    let original = write_image(&dir, "external.png", photocraft_codecs::Format::Png);
    let mut session = photocraft_engine::Session::new();
    session.execute("file.new", json!({"width": 16, "height": 8, "background": "white"})).unwrap();
    let layer = session.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap()["layer"].clone();
    let state = session.active_mut().unwrap();
    let LayerContent::Smart(smart) = &mut std::sync::Arc::make_mut(&mut state.doc).layers[0].content else { panic!("smart object") };
    smart.source = SmartSource::Linked { path: dir.join("external.png").to_string_lossy().into_owned() };
    std::fs::write(dir.join("parent.pcraft"), photocraft_engine::smart_cmds::encode_source(&state.doc).unwrap()).unwrap();
    let client = connect(headless_in(&dir)).await;
    json_of(&call(&client, "doc_open", json!({"path": "parent.pcraft"})).await);
    let before = json_of(&call(&client, "doc_inspect", json!({})).await);
    for id in ["layer.smartObjects.editContents", "layer.smartObjects.convertToLayers"] {
        let response = call(&client, "command_run", json!({"id": id, "params": {"layer": layer}})).await;
        assert_eq!(response.is_error, Some(true));
        assert!(text(&response).contains("ambient filesystem paths"), "{}", text(&response));
        assert_eq!(json_of(&call(&client, "doc_inspect", json!({})).await), before);
        assert_eq!(json_of(&call(&client, "session_list", json!({})).await)["documents"].as_array().unwrap().len(), 1);
    }
    assert_eq!(std::fs::read(dir.join("external.png")).unwrap(), original);
    client.cancel().await.unwrap();
    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// Bridge mode against a fake control server
// ---------------------------------------------------------------------------

async fn fake_app() -> (String, tokio::task::JoinHandle<Vec<Value>>) {
    fake_app_with_screenshot(None).await
}

async fn fake_app_with_screenshot(screenshot_png: Option<Vec<u8>>) -> (String, tokio::task::JoinHandle<Vec<Value>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let h = tokio::spawn(async move {
        let mut seen = Vec::new();
        let mut authenticated = false;
        let (sock, _) = listener.accept().await.unwrap();
        let (r, mut w) = sock.into_split();
        let mut lines = BufReader::new(r).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let req: Value = serde_json::from_str(&line).unwrap();
            let id = req["id"].clone();
            let method = req["method"].as_str().unwrap_or("").to_owned();
            let reply = match method.as_str() {
                "auth" if req["params"]["token"] == CONTROL_TOKEN => {
                    authenticated = true;
                    json!({"id": id, "ok": true, "result": {"authenticated": true}})
                }
                _ if !authenticated => {
                    json!({"id": id, "ok": false, "error": "authentication required"})
                }
                "ui.inspect" => {
                    json!({"id": id, "ok": true, "result": {"tool": "brush", "panels": ["layers"]}})
                }
                "engine.execute" => {
                    json!({"id": id, "ok": true, "result": {"ran": req["params"]["command"], "params": req["params"]["params"], "wait": req["params"]["wait"]}})
                }
                "app.save" => json!({"id": id, "ok": true, "result": {"saved": req["params"]}}),
                "engine.commands" => {
                    json!({"id": id, "ok": true, "result": [{"id": "file.new", "label": "New…", "enabled": true}]})
                }
                "ui.pointer" => json!({"id": id, "ok": true, "result": req["params"]}),
                "ui.screenshot" => {
                    let bytes = screenshot_png.clone().unwrap_or_else(|| {
                        let img = photocraft_codecs::Image::from_u8(40, 20, photocraft_codecs::ChannelLayout::Rgba, vec![9; 3200]).unwrap();
                        photocraft_codecs::encode(&img, photocraft_codecs::Format::Png, &Default::default()).unwrap()
                    });
                    let png = base64::engine::general_purpose::STANDARD.encode(bytes);
                    json!({"id": id, "ok": true, "result": {"mimeType": "image/png", "base64": png}})
                }
                _ => json!({"id": id, "ok": false, "error": format!("unknown tool `{method}`")}),
            };
            // A stale line first, to check id matching.
            w.write_all(b"{\"id\":999999,\"ok\":true,\"result\":null}\n").await.unwrap();
            w.write_all(format!("{reply}\n").as_bytes()).await.unwrap();
            seen.push(req);
        }
        seen
    });
    (addr, h)
}

#[tokio::test(flavor = "multi_thread")]
async fn bridge_forwards_to_control_protocol() {
    let (addr, app) = fake_app().await;
    let client = connect(PhotocraftMcp::bridge(&addr, CONTROL_TOKEN).unwrap()).await;

    let ui = json_of(&call(&client, "ui_inspect", json!({})).await);
    assert_eq!(ui["tool"], "brush");
    let r = json_of(&call(&client, "command_run", json!({"id": "layer.new.layer", "params": {"name": "X"}})).await);
    assert_eq!(r["ran"], "layer.new.layer");
    let inspected = json_of(&call(&client, "doc_inspect", json!({"index": 1})).await);
    assert_eq!(inspected["ran"], "document.inspect");
    assert_eq!(inspected["params"], json!({"document": 1}));
    let l = json_of(&call(&client, "command_list", json!({})).await);
    assert_eq!(l[0]["id"], "file.new");
    let shot = call(&client, "ui_screenshot", json!({"max_side": 20})).await;
    let img = shot.content.iter().find_map(|c| c.as_image()).expect("image");
    let png = base64::engine::general_purpose::STANDARD.decode(&img.data).unwrap();
    assert_eq!(photocraft_codecs::decode(&png).unwrap().dimensions(), (20, 10));
    let e = call(&client, "control_call", json!({"method": "bogus.method"})).await;
    assert_eq!(e.is_error, Some(true));
    assert!(text(&e).contains("unknown tool"));
    let r = call(&client, "doc_select", json!({"index": 0})).await;
    assert_eq!(r.is_error, Some(true));

    client.cancel().await.unwrap();
    app.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn bridge_lost_reply_does_not_replay_an_edit_and_next_call_reconnects() {
    use photocraft_automation::{BridgeClient, Headless};
    use std::time::Duration;

    tokio::time::timeout(Duration::from_secs(5), async {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let app = tokio::spawn(async move {
            let mut backend = Headless::new();
            backend.handle("doc.new", json!({"width": 8, "height": 8})).unwrap();
            let initial_layers = backend.session.active().unwrap().doc.layer_count();
            let initial_history = backend.session.active().unwrap().history.past_len();
            let mut seen = Vec::new();
            for connection in 0..2 {
                let (socket, _) = listener.accept().await.unwrap();
                let (read, mut write) = socket.into_split();
                let mut reader = BufReader::new(read);
                let mut line = String::new();
                reader.read_line(&mut line).await.unwrap();
                let (auth, authenticated) = photocraft_automation::security::authentication_reply(&line, CONTROL_TOKEN);
                assert!(authenticated, "each connection must authenticate");
                write.write_all(format!("{auth}\n").as_bytes()).await.unwrap();
                line.clear();
                reader.read_line(&mut line).await.unwrap();
                let request: Value = serde_json::from_str(&line).unwrap();
                seen.push(request["params"]["command"].as_str().unwrap().to_owned());
                let result = backend.handle(request["method"].as_str().unwrap(), request["params"].clone()).unwrap();
                if connection == 0 {
                    // The edit has completed, but both socket halves close before its reply.
                    continue;
                }
                let reply = json!({"id": request["id"], "ok": true, "result": result});
                write.write_all(format!("{reply}\n").as_bytes()).await.unwrap();
            }
            let document = backend.session.active().unwrap();
            assert_eq!(document.doc.layer_count(), initial_layers + 1, "the edit was applied once");
            assert_eq!(document.history.past_len(), initial_history + 1, "one undo step");
            seen
        });
        let bridge = BridgeClient::new(&addr, CONTROL_TOKEN).unwrap();
        let result = bridge.call("engine.execute", json!({"command": "layer.new.layer", "params": {"name": "Once"}})).await;
        let error = result.unwrap_err().to_string();
        assert!(error.contains("operation may have completed"), "{error}");
        assert!(error.contains("inspect"), "{error}");
        let document = bridge.call("engine.execute", json!({"command": "document.inspect"})).await.unwrap();
        assert_eq!(document["layers"].as_array().unwrap().len(), 2);
        assert_eq!(app.await.unwrap(), ["layer.new.layer", "document.inspect"]);
    })
    .await
    .expect("bridge lost-reply regression must finish");
}

#[tokio::test(flavor = "multi_thread")]
async fn bridge_previews_downscale_before_enforcing_the_png_budget() {
    use photocraft_automation::budgets::MAX_PNG_BYTES;

    let (width, height) = (1400, 1000);
    let mut pixels = vec![0; width * height * 4];
    let mut state = 0x6d2b_79f5_u32;
    for pixel in &mut pixels {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        *pixel = state as u8;
    }
    let image = photocraft_codecs::Image::from_u8(width as u32, height as u32, photocraft_codecs::ChannelLayout::Rgba, pixels).unwrap();
    let source_png = photocraft_codecs::encode(&image, photocraft_codecs::Format::Png, &Default::default()).unwrap();
    assert!(source_png.len() > MAX_PNG_BYTES, "fixture must exceed the PNG budget");

    let (addr, app) = fake_app_with_screenshot(Some(source_png)).await;
    let client = connect(PhotocraftMcp::bridge(&addr, CONTROL_TOKEN).unwrap()).await;

    for (tool, args) in [("ui_screenshot", json!({"max_side": 32})), ("doc_render_preview", json!({"max_side": 32}))] {
        let reply = call(&client, tool, args).await;
        assert_ne!(reply.is_error, Some(true), "{tool}: {}", text(&reply));
        let img = reply.content.iter().find_map(|content| content.as_image()).expect("preview image");
        let png = base64::engine::general_purpose::STANDARD.decode(&img.data).unwrap();
        assert!(png.len() <= MAX_PNG_BYTES, "{tool} returned {} bytes", png.len());
        let decoded = photocraft_codecs::decode(&png).unwrap();
        assert_eq!(decoded.dimensions(), (32, 22), "{tool}");
    }

    let full_size = call(&client, "ui_screenshot", json!({})).await;
    assert_eq!(full_size.is_error, Some(true));
    assert!(text(&full_size).contains("automation PNG exceeds"), "{}", text(&full_size));

    client.cancel().await.unwrap();
    app.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn bridge_previews_reject_document_indices() {
    let (addr, app) = fake_app().await;
    let client = connect(PhotocraftMcp::bridge(&addr, CONTROL_TOKEN).unwrap()).await;
    let mut errors = Vec::new();
    for tool in ["ui_screenshot", "doc_render_preview"] {
        let args = json!({"index": 0, "max_side": 20}).as_object().unwrap().clone();
        let result = client.call_tool(CallToolRequestParams::new(tool).with_arguments(args)).await;
        let error = match result {
            Err(error) => Some(error.to_string()),
            Ok(reply) if reply.is_error == Some(true) => Some(text(&reply)),
            Ok(_) => None,
        };
        errors.push((tool, error));

        let reply = call(&client, tool, json!({"max_side": 20})).await;
        assert_ne!(reply.is_error, Some(true), "{tool}: {}", text(&reply));
        let image = reply.content.iter().find_map(|content| content.as_image()).expect("screenshot");
        let png = base64::engine::general_purpose::STANDARD.decode(&image.data).unwrap();
        assert_eq!(photocraft_codecs::decode(&png).unwrap().dimensions(), (20, 10), "{tool}");
    }
    let tools = client.list_all_tools().await.unwrap();
    let schema = &tools.iter().find(|tool| tool.name == "ui_screenshot").unwrap().input_schema;
    client.cancel().await.unwrap();
    let seen = tokio::time::timeout(std::time::Duration::from_secs(5), app).await.unwrap().unwrap();

    assert!(errors.iter().all(|(_, error)| error.as_ref().is_some_and(|error| error.contains("index"))), "{errors:?}");
    assert_eq!(seen.iter().filter(|request| request["method"] == "ui.screenshot").count(), 2, "rejected indices must not reach the app");
    assert!(schema["properties"].get("index").is_none(), "{schema:?}");
    assert!(schema["properties"].get("max_side").is_some(), "{schema:?}");
    assert_eq!(schema.get("additionalProperties"), Some(&json!(false)));
}

#[tokio::test(flavor = "multi_thread")]
async fn headless_previews_preserve_document_indices() {
    let client = connect(PhotocraftMcp::headless()).await;
    json_of(&call(&client, "doc_new", json!({"width": 12, "height": 6, "background": "white"})).await);
    json_of(&call(&client, "doc_new", json!({"width": 6, "height": 12, "background": "black"})).await);
    for (args, dimensions, pixel) in [(json!({"index": 0, "max_side": 6}), (6, 3), [255, 255, 255, 255]), (json!({"max_side": 6}), (3, 6), [0, 0, 0, 255])] {
        let reply = call(&client, "doc_render_preview", args).await;
        assert_ne!(reply.is_error, Some(true), "{}", text(&reply));
        let image = reply.content.iter().find_map(|content| content.as_image()).expect("preview");
        let png = base64::engine::general_purpose::STANDARD.decode(&image.data).unwrap();
        let image = photocraft_codecs::decode(&png).unwrap();
        assert_eq!(image.dimensions(), dimensions);
        assert!(image.to_rgba8().as_chunks::<4>().0.iter().all(|actual| actual[..] == pixel[..]));
    }
    assert_eq!(json_of(&call(&client, "session_list", json!({})).await)["active"], 1);
    let tools = client.list_all_tools().await.unwrap();
    let schema = &tools.iter().find(|tool| tool.name == "doc_render_preview").unwrap().input_schema;
    assert!(schema["properties"].get("index").is_some(), "{schema:?}");
    client.cancel().await.unwrap();
}

#[test]
fn bridge_rejects_non_loopback() {
    assert!(PhotocraftMcp::bridge("10.0.0.5:7878", CONTROL_TOKEN).is_err());
    assert!(PhotocraftMcp::bridge("127.0.0.1:7878", CONTROL_TOKEN).is_ok());
    assert!(PhotocraftMcp::bridge("localhost:1", CONTROL_TOKEN).is_ok());
    assert!(PhotocraftMcp::bridge("localhost:1", "short").is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn bridge_rejects_wrong_token_before_control_methods() {
    let (addr, app) = fake_app().await;
    let wrong = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let client = connect(PhotocraftMcp::bridge(&addr, wrong).unwrap()).await;
    let r = call(&client, "ui_inspect", json!({})).await;
    assert_eq!(r.is_error, Some(true));
    assert!(text(&r).contains("authentication required"), "{}", text(&r));
    client.cancel().await.unwrap();
    app.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn bridge_reports_unreachable_app() {
    let client = connect(PhotocraftMcp::bridge("127.0.0.1:1", CONTROL_TOKEN).unwrap()).await;
    let r = call(&client, "ui_inspect", json!({})).await;
    assert_eq!(r.is_error, Some(true));
    assert!(text(&r).contains("--control"), "{}", text(&r));
    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn command_batch_description_exposes_non_atomic_semantics_when_tools_are_listed() {
    let client = connect(PhotocraftMcp::headless()).await;
    let tools = client.list_all_tools().await.unwrap();
    let tool = tools.iter().find(|tool| tool.name == "command_batch").expect("command_batch tool");
    let description = tool.description.as_deref().expect("command_batch description");
    for contract in ["in order", "Not an atomic transaction", "first error", "stop_on_error is false", "no rollback"] {
        assert!(description.contains(contract), "command_batch description omits `{contract}`: {description}");
    }
    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn command_batch_runs_steps_in_order() {
    let client = connect(PhotocraftMcp::headless()).await;
    json_of(&call(&client, "doc_new", json!({"width": 32, "height": 32})).await);
    let r = json_of(
        &call(
            &client,
            "command_batch",
            json!({"steps": [
                {"id": "layer.new.layer", "params": {"name": "One"}},
                {"id": "layer.new.layer", "params": {"name": "Two"}},
                {"id": "no.such.command"},
                {"id": "layer.new.layer", "params": {"name": "Three"}}
            ]}),
        )
        .await,
    );
    assert_eq!(r["completed"], 2);
    assert_eq!(r["failed"], 1);
    let doc = json_of(&call(&client, "doc_inspect", json!({})).await).to_string();
    assert!(doc.contains("\"Two\"") && !doc.contains("\"Three\""));
    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn command_batch_step_without_waiting_starts_a_background_job() {
    let client = connect(PhotocraftMcp::headless()).await;
    json_of(&call(&client, "doc_new", json!({"width": 600, "height": 400})).await);
    let blur = json!({"id": "filter.blur.gaussianBlur", "params": {"radius": 40}, "wait": false});
    let r = json_of(
        &call(&client, "command_batch", json!({"steps": [{"id": "layer.new.layer"}, {"id": "edit.fill", "params": {"color": "#808080"}}, blur]})).await,
    );
    assert_eq!(r["completed"], 3, "{r}");
    let started = &r["results"][2]["result"];
    assert_eq!(started["pending"], true, "the step returned without waiting: {r}");
    let job = started["job"].as_u64().expect("a job id");
    // The job is listed and finishes like one started by command_run.
    let t = std::time::Instant::now();
    loop {
        let l = json_of(&call(&client, "jobs_list", json!({})).await);
        let state = l["jobs"].as_array().unwrap().iter().find(|j| j["id"] == job).map(|j| j["state"].clone());
        assert!(state.is_some(), "job {job} not listed: {l}");
        if state == Some(json!("done")) {
            break;
        }
        assert!(t.elapsed().as_secs() < 60, "{l}");
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    // Without `wait` a step still returns the command's own result.
    let r = json_of(&call(&client, "command_batch", json!({"steps": [{"id": "filter.blur.gaussianBlur", "params": {"radius": 2}}]})).await);
    assert!(r["results"][0]["result"]["filter"].is_object(), "{r}");
    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn bridge_command_batch_forwards_each_steps_wait() {
    let (addr, app) = fake_app().await;
    let client = connect(PhotocraftMcp::bridge(&addr, CONTROL_TOKEN).unwrap()).await;
    let r = json_of(&call(&client, "command_batch", json!({"steps": [{"id": "filter.blur.gaussianBlur", "wait": false}, {"id": "layer.new.layer"}]})).await);
    assert_eq!(r["completed"], 2, "{r}");
    assert_eq!(r["results"][0]["result"]["wait"], false, "{r}");
    assert_eq!(r["results"][1]["result"]["wait"], true, "{r}");
    client.cancel().await.unwrap();
    app.abort();
}

/// #514: `ui_pointer` forwards `button` (a right-click opens the layer menu or the Brush Preset
/// picker) and rejects arguments it doesn't forward instead of dropping them.
#[tokio::test(flavor = "multi_thread")]
async fn bridge_ui_pointer_forwards_the_button_and_rejects_unknown_arguments() {
    let (addr, app) = fake_app().await;
    let client = connect(PhotocraftMcp::bridge(&addr, CONTROL_TOKEN).unwrap()).await;
    let events = json!([{"kind": "down", "x": 5, "y": 6}, {"kind": "up", "x": 5, "y": 6}]);
    let sent = json_of(&call(&client, "ui_pointer", json!({"events": events, "button": "right", "modifiers": {"command": true}})).await);
    assert_eq!(sent, json!({"events": events, "button": "right", "modifiers": {"command": true}}));
    let sent = json_of(&call(&client, "ui_pointer", json!({"events": events})).await);
    assert_eq!(sent, json!({"events": events}), "unset fields are not sent");
    for bad in [json!({"events": events, "buton": "right"}), json!({"events": events, "space": true})] {
        let Value::Object(args) = bad.clone() else { unreachable!() };
        let r = client.call_tool(CallToolRequestParams::new("ui_pointer").with_arguments(args)).await;
        assert!(!r.as_ref().is_ok_and(|r| r.is_error != Some(true)), "{bad} accepted: {r:?}");
    }
    let tools = client.list_all_tools().await.unwrap();
    let schema = &tools.iter().find(|t| t.name == "ui_pointer").unwrap().input_schema;
    assert_eq!(schema.get("additionalProperties"), Some(&json!(false)), "{schema:?}");
    assert!(schema.get("properties").and_then(|p| p.get("button")).is_some(), "{schema:?}");
    client.cancel().await.unwrap();
    app.abort();
}

/// #368: agents can open the clipboard as a document of its own.
#[tokio::test(flavor = "multi_thread")]
async fn new_from_clipboard_opens_the_copy_as_a_document() {
    let client = connect(PhotocraftMcp::headless()).await;
    json_of(&call(&client, "doc_new", json!({"width": 40, "height": 30})).await);
    let r = json_of(
        &call(
            &client,
            "command_batch",
            json!({"steps": [
                {"id": "select.rect", "params": {"x": 5, "y": 5, "width": 12, "height": 7}},
                {"id": "edit.copy"},
                {"id": "file.newFromClipboard"}
            ]}),
        )
        .await,
    );
    assert_eq!(r["completed"], 3, "{r}");
    let doc = json_of(&call(&client, "doc_inspect", json!({})).await);
    assert_eq!((doc["width"].as_u64(), doc["height"].as_u64()), (Some(12), Some(7)), "{doc}");
    let sess = json_of(&call(&client, "session_list", json!({})).await);
    assert_eq!(sess["documents"].as_array().unwrap().len(), 2);
    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn command_batch_rejects_too_many_steps() {
    let client = connect(PhotocraftMcp::headless()).await;
    let steps: Vec<Value> = (0..=photocraft_automation::security::MAX_BATCH_STEPS).map(|_| json!({"id": "command.list"})).collect();
    let r = call(&client, "command_batch", json!({"steps": steps})).await;
    assert_eq!(r.is_error, Some(true));
    assert!(text(&r).contains("maximum is 256"), "{}", text(&r));
    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn convention_tools_have_titles_hints_and_strict_schemas() {
    let client = connect(PhotocraftMcp::headless()).await;
    let tools = client.list_all_tools().await.unwrap();
    assert!(tools.iter().any(|t| t.name == "render_preview"));
    // The existing documented spelling remains listed.
    assert!(tools.iter().any(|t| t.name == "doc_render_preview"));
    for tool in tools {
        let value = serde_json::to_value(&tool).unwrap();
        assert!(value["title"].as_str().is_some_and(|s| !s.is_empty()), "{value}");
        for hint in ["readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"] {
            assert!(value["annotations"][hint].is_boolean(), "{value}");
        }
        assert_eq!(value["inputSchema"]["additionalProperties"], false, "{value}");
    }
    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn convention_unknown_arguments_name_the_key_without_side_effects() {
    let client = connect(PhotocraftMcp::headless()).await;
    for (name, args) in [
        ("doc_new", json!({"widht": 8})),
        ("session_list", json!({"surprise": true})),
        ("command_run", json!({"id":"file.new", "surprise":true})),
        ("command_batch", json!({"steps":[{"id":"file.new", "surprise":true}]})),
    ] {
        let key = if name == "doc_new" { "widht" } else { "surprise" };
        let result = client.call_tool(CallToolRequestParams::new(name).with_arguments(args.as_object().unwrap().clone())).await;
        let error = result.expect_err("unknown key must fail before execution").to_string();
        assert!(error.contains("-32602") && error.contains(key), "{error}");
    }
    assert_eq!(json_of(&call(&client, "session_list", json!({})).await)["documents"], json!([]));
    client.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn convention_resources_are_live_json() {
    let client = connect(PhotocraftMcp::headless()).await;
    let resources = client.list_all_resources().await.unwrap();
    for uri in ["photocraft://document", "photocraft://commands"] {
        assert!(resources.iter().any(|r| r.uri == uri));
    }
    let read = async |uri: &str| {
        let r = client.read_resource(rmcp::model::ReadResourceRequestParams::new(uri)).await.unwrap();
        let v = serde_json::to_value(r).unwrap();
        serde_json::from_str::<Value>(v["contents"][0]["text"].as_str().unwrap()).unwrap()
    };
    assert!(read("photocraft://document").await["document"].is_null());
    json_of(&call(&client, "doc_new", json!({"width":8,"height":6})).await);
    assert_eq!(read("photocraft://document").await["width"], 8);
    // Edit › Purge › All is enabled while the process-wide effect, mask and stack caches hold
    // anything, and the other tests in this binary fill and purge those caches concurrently, so
    // its `enabled` flag and paired `disabledReason` can flip between the two reads. Validate
    // each availability pair; every stable field and every other command must match exactly.
    let stable = |mut v: Value| {
        for c in v.as_array_mut().into_iter().flatten() {
            if c["id"] == "edit.purge.all" {
                let reason = if c["enabled"].as_bool().unwrap() { Value::Null } else { json!("nothing to purge") };
                assert_eq!(c["disabledReason"], reason, "purge availability stays internally consistent");
                let command = c.as_object_mut().unwrap();
                command.remove("enabled");
                command.remove("disabledReason");
            }
        }
        v
    };
    let commands = stable(read("photocraft://commands").await);
    assert!(commands.as_array().is_some_and(|a| a.len() > 100), "{commands}");
    assert_eq!(commands, stable(json_of(&call(&client, "command_list", json!({})).await)));
    assert!(client.read_resource(rmcp::model::ReadResourceRequestParams::new("photocraft://missing")).await.is_err());
    let preview = call(&client, "render_preview", json!({"max_side":4})).await;
    assert!(preview.content.iter().any(|c| c.as_image().is_some()));
    client.cancel().await.unwrap();
}
