//! Exercise the actual line transport, including recovery and modern result fields.
use photocraft_automation::PhotocraftMcp;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

#[tokio::test(flavor = "multi_thread")]
async fn malformed_line_recovers_and_modern_lists_are_complete() {
    let (mut input, server_in) = tokio::io::duplex(1 << 20);
    let (server_out, output) = tokio::io::duplex(1 << 20);
    let server = tokio::spawn(PhotocraftMcp::headless().serve_io(server_in, server_out));
    let mut lines = BufReader::new(output).lines();
    let init = json!({"jsonrpc":"2.0", "id":1, "method":"initialize", "params":{"protocolVersion":"2025-06-18", "capabilities":{}, "clientInfo":{"name":"test", "version":"1"}}});
    input.write_all(format!("{init}\n").as_bytes()).await.unwrap();
    lines.next_line().await.unwrap().unwrap();
    input.write_all(b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\nBROKEN\n").await.unwrap();
    let error: Value = serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
    assert_eq!(error["id"], Value::Null);
    assert_eq!(error["error"]["code"], -32700);
    let meta = json!({"io.modelcontextprotocol/protocolVersion":"2026-07-28", "io.modelcontextprotocol/clientInfo":{"name":"test", "version":"1"}, "io.modelcontextprotocol/clientCapabilities":{}});
    for (id, method, extra) in [(2, "tools/list", json!({})), (3, "resources/list", json!({})), (4, "resources/read", json!({"uri":"photocraft://document"}))] {
        let mut params = extra;
        params["_meta"] = meta.clone();
        input.write_all(format!("{}\n", json!({"jsonrpc":"2.0", "id":id, "method":method, "params":params})).as_bytes()).await.unwrap();
        let reply: Value = serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
        assert_eq!(reply["id"], id);
        assert_eq!(reply["result"]["resultType"], "complete", "{reply}");
        assert!(reply["result"]["ttlMs"].is_number());
        assert_eq!(reply["result"]["cacheScope"], "private");
    }
    drop(input);
    tokio::time::timeout(std::time::Duration::from_secs(5), server).await.unwrap().unwrap().unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn video_export_progress_ping_cancel_and_scoped_cleanup() {
    use photocraft_automation::AuthorizedWorkspace;
    use std::time::Duration;
    let root = std::env::temp_dir().join(format!("pc-conventions-video-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let workspace = AuthorizedWorkspace::new(None, Some(&root)).unwrap();
    let (mut input, server_in) = tokio::io::duplex(1 << 20);
    let (server_out, output) = tokio::io::duplex(1 << 20);
    let server = tokio::spawn(PhotocraftMcp::headless_with_workspace(workspace).serve_io(server_in, server_out));
    let mut lines = BufReader::new(output).lines();
    macro_rules! send {
        ($value:expr) => {
            input.write_all(format!("{}\n", $value).as_bytes()).await.unwrap()
        };
    }
    macro_rules! next {
        () => {
            serde_json::from_str::<Value>(&tokio::time::timeout(Duration::from_secs(30), lines.next_line()).await.unwrap().unwrap().unwrap()).unwrap()
        };
    }
    let call = |id: u64, name: &str, args: Value| json!({"jsonrpc":"2.0", "id":id, "method":"tools/call", "params":{"name":name, "arguments":args}});
    send!(
        json!({"jsonrpc":"2.0", "id":1, "method":"initialize", "params":{"protocolVersion":"2025-06-18", "capabilities":{}, "clientInfo":{"name":"test", "version":"1"}}})
    );
    next!();
    send!(json!({"jsonrpc":"2.0", "method":"notifications/initialized"}));
    send!(call(2, "doc_new", json!({"width":64,"height":48,"name":"clip"})));
    assert_eq!(next!()["result"]["isError"], false);
    send!(call(3, "command_run", json!({"id":"timeline.create", "params":{"duration":6,"fps":6}})));
    assert_eq!(next!()["result"]["isError"], false);
    let render = |id, dir, format, token: Option<Value>| {
        let mut value = call(id, "command_run", json!({"id":"file.export.renderVideo", "params":{"dir":dir,"format":format}}));
        if let Some(token) = token {
            value["params"]["_meta"] = json!({"progressToken":token});
        }
        value
    };
    for (id, dir, format, token) in [(4, "complete", "png", Some(json!("frames"))), (5, "quiet", "png", None), (6, "gif", "gif", Some(json!(77)))] {
        send!(render(id, dir, format, token.clone()));
        let mut progress = Vec::new();
        loop {
            let reply = next!();
            if reply["id"] == id {
                assert_eq!(reply["result"]["isError"], false, "{reply}");
                break;
            }
            assert!(token.is_some(), "no token: {reply}");
            assert_eq!(reply["params"]["progressToken"], token.clone().unwrap());
            assert_eq!(reply["params"]["total"].as_f64(), Some(6.0));
            progress.push(reply["params"]["progress"].as_f64().unwrap());
        }
        if token.is_some() {
            assert_eq!(progress.first(), Some(&0.0));
            assert_eq!(progress.last(), Some(&6.0));
            assert!(progress.windows(2).all(|w| w[0] < w[1]));
        }
    }
    assert_eq!(std::fs::read_dir(root.join("complete")).unwrap().count(), 6);
    assert!(photocraft_codecs::decode(&std::fs::read(root.join("gif/clip.gif")).unwrap()).is_ok());
    // A collision must preserve the earlier file and remove only frames from this failed run.
    std::fs::create_dir(root.join("collision")).unwrap();
    std::fs::write(root.join("collision/clip_0001.png"), b"earlier export").unwrap();
    send!(render(7, "collision", "png", None));
    assert_eq!(next!()["result"]["isError"], true);
    assert_eq!(std::fs::read_dir(root.join("collision")).unwrap().count(), 1);
    assert_eq!(std::fs::read(root.join("collision/clip_0001.png")).unwrap(), b"earlier export");
    send!(render(8, "../outside", "png", None));
    assert_eq!(next!()["result"]["isError"], true);
    send!(call(9, "doc_new", json!({"width":1600,"height":1200,"name":"clip"})));
    assert_eq!(next!()["result"]["isError"], false);
    send!(call(10, "command_run", json!({"id":"timeline.create", "params":{"duration":100,"fps":30}})));
    assert_eq!(next!()["result"]["isError"], false);
    std::fs::create_dir(root.join("cancel")).unwrap();
    std::fs::write(root.join("cancel/clip_9999.png"), b"unrelated").unwrap();
    send!(render(11, "cancel", "png", Some(json!(11))));
    let mut pinged = false;
    loop {
        let note = next!();
        assert!(note["id"].is_null(), "export finished early: {note}");
        if !pinged {
            send!(json!({"jsonrpc":"2.0", "id":12, "method":"ping"}));
            loop {
                let reply = next!();
                assert_ne!(reply["id"], 11);
                if reply["id"] == 12 {
                    break;
                }
            }
            pinged = true;
        }
        if note["params"]["progress"].as_f64().unwrap_or(0.0) >= 1.0 {
            break;
        }
    }
    assert!(std::fs::read_dir(root.join("cancel")).unwrap().count() > 1);
    send!(json!({"jsonrpc":"2.0", "method":"notifications/cancelled", "params":{"requestId":11}}));
    // Inspect waits for the same session lock, so cleanup must have finished before its reply.
    send!(call(13, "doc_inspect", json!({})));
    loop {
        let reply = next!();
        assert_ne!(reply["id"], 11, "cancelled requests have no response: {reply}");
        if reply["id"] == 13 {
            assert_eq!(reply["result"]["isError"], false);
            break;
        }
    }
    assert_eq!(std::fs::read_dir(root.join("cancel")).unwrap().count(), 1);
    assert_eq!(std::fs::read(root.join("cancel/clip_9999.png")).unwrap(), b"unrelated");
    drop(input);
    tokio::time::timeout(Duration::from_secs(5), server).await.unwrap().unwrap().unwrap();
    std::fs::remove_dir_all(root).unwrap();
}
