//! Bridge client behavior against a scripted fake control server: reconnect before
//! idle sockets die, retry only caller-vouched reads, never resend edits (#1007),
//! never retry timeouts.

use std::time::Duration;

use photocraft_automation::security::authentication_reply;
use photocraft_automation::{AutomationError, BridgeClient};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};

const CONTROL_TOKEN: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

/// What the fake server does after serving one request on a connection.
#[derive(Clone, Copy, PartialEq)]
enum After {
    KeepOpen,
    Close,
}

/// Serve connections until `max_connections` were seen. Each connection authenticates,
/// then serves requests until it closes; every served request is recorded, and `accepted`
/// counts connections as they arrive.
async fn fake_server(
    max_connections: usize,
    after: After,
    seen: tokio::sync::mpsc::UnboundedSender<Vec<Value>>,
    accepted: std::sync::Arc<std::sync::atomic::AtomicUsize>,
) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    tokio::spawn(async move {
        let mut connections = 0;
        while connections < max_connections {
            let Ok((socket, _)) = listener.accept().await else { break };
            connections += 1;
            accepted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let seen = seen.clone();
            let after = after;
            tokio::spawn(async move {
                let mut served = Vec::new();
                let mut socket = socket;
                authenticate(&mut socket).await;
                let (read, mut write) = socket.into_split();
                let mut reader = BufReader::new(read);
                let mut line = String::new();
                loop {
                    line.clear();
                    match reader.read_line(&mut line).await {
                        Ok(0) | Err(_) => break,
                        Ok(_) => {}
                    }
                    let Ok(request) = serde_json::from_str::<Value>(&line) else { continue };
                    served.push(request.clone());
                    let reply = json!({"id": request["id"], "ok": true, "result": {"served": served.len()}});
                    if write.write_all(format!("{reply}\n").as_bytes()).await.is_err() {
                        break;
                    }
                    if after == After::Close {
                        break; // the reply is flushed, then both halves drop: a dead connection
                    }
                }
                let _ = seen.send(served);
            });
        }
    });
    addr
}

/// Authenticate one connection: read the auth line, reply to it.
async fn authenticate(socket: &mut TcpStream) {
    let (read, mut write) = socket.split();
    let mut reader = BufReader::new(read);
    let mut line = String::new();
    reader.read_line(&mut line).await.unwrap();
    let (auth, ok) = authentication_reply(&line, CONTROL_TOKEN);
    assert!(ok, "each connection must authenticate");
    write.write_all(format!("{auth}\n").as_bytes()).await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn read_retries_on_a_dropped_connection_and_edits_do_not() {
    // Connection 1 serves one read then dies; the next vouched read retries on
    // connection 2. The edit, sent after connection 2 also died, must fail with the
    // #1007 wording and never open connection 3.
    let (seen_tx, mut seen_rx) = tokio::sync::mpsc::unbounded_channel();
    let accepted = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let addr = fake_server(2, After::Close, seen_tx, accepted).await;
    let bridge = BridgeClient::new(&addr, CONTROL_TOKEN).unwrap();

    let first = bridge.call_read("document.inspect", json!({})).await.expect("read served before the connection dies");
    assert_eq!(first["served"], 1);

    let retried = bridge.call_read("document.inspect", json!({})).await.expect("a vouched read retried on a fresh connection");
    assert_eq!(retried["served"], 1, "connection 2 sees it as its own first request");

    let edit = bridge.call("engine.execute", json!({"command": "layer.new.layer", "params": {}})).await;
    let error = edit.unwrap_err().to_string();
    assert!(error.contains("operation may have completed"), "{error}");
    assert!(error.contains("inspect state before retrying"), "{error}");
    assert!(!error.contains("retried"), "{error}");

    let mut all = Vec::new();
    while let Ok(served) = seen_rx.try_recv() {
        all.extend(served);
    }
    assert_eq!(all.len(), 2, "one request per connection, no edit resend: {all:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_idle_connection_is_replaced_before_the_next_call() {
    // The production idle threshold sits under the app's 30s idle cutoff; the test
    // shrinks it so the reconnect path runs quickly. The next call after idling must
    // authenticate a new connection before sending.
    let (seen_tx, mut seen_rx) = tokio::sync::mpsc::unbounded_channel();
    let accepted = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let addr = fake_server(2, After::KeepOpen, seen_tx, std::sync::Arc::clone(&accepted)).await;
    let bridge = BridgeClient::new(&addr, CONTROL_TOKEN).unwrap().with_idle_reconnect(Duration::from_millis(100));

    bridge.call_read("document.inspect", json!({})).await.expect("first call connects");
    tokio::time::sleep(Duration::from_millis(250)).await;
    bridge.call_read("document.inspect", json!({})).await.expect("second call reconnects after idling");

    assert_eq!(accepted.load(std::sync::atomic::Ordering::SeqCst), 2, "the idle connection was replaced, not reused");
    let first = seen_rx.try_recv().expect("the replaced connection closed, reporting what it served");
    assert_eq!(first.len(), 1, "each connection served only the request that followed its auth: {first:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_timed_out_call_is_not_retried() {
    // The server authenticates but never answers requests, keeping the connection open:
    // the call hits the client's own timeout, which must name the likely causes and must
    // not double the wait by retrying (the fake server counts every request it sees).
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let requests = std::sync::Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
    let seen = std::sync::Arc::clone(&requests);
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        authenticate(&mut socket).await;
        let (read, _write) = socket.into_split();
        let mut reader = BufReader::new(read);
        let mut line = String::new();
        loop {
            line.clear();
            if reader.read_line(&mut line).await.unwrap_or(0) == 0 {
                return;
            }
            if let Ok(request) = serde_json::from_str::<Value>(&line) {
                seen.lock().unwrap().push(request);
            }
        }
    });

    let bridge = BridgeClient::new(&addr, CONTROL_TOKEN).unwrap().with_timeout(Duration::from_millis(300));
    let error = match bridge.call_read("document.inspect", json!({})).await {
        Err(e @ AutomationError::Bridge(_)) => e.to_string(),
        other => panic!("expected the bridge timeout, got {other:?}"),
    };
    assert!(error.contains("timed out after"), "{error}");
    assert!(error.contains("not drawing frames"), "{error}");
    assert!(error.contains("a read may be retried safely"), "{error}");
    assert!(!error.contains("may have completed"), "a read timeout never claims an applied edit: {error}");
    assert_eq!(requests.lock().unwrap().len(), 1, "a timed-out call is not resent");
}
