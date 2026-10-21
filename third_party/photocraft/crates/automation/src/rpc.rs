//! Headless JSON-lines server: the control protocol's envelope
//! (`{"id","method","params"}` → `{"id","ok","result"|"error"}`) over a
//! [`Headless`] session, so scripts and agents can keep one editing session
//! open without MCP or the GUI. `photocraft-cli serve` runs it on stdio or a
//! loopback TCP port.
//!
//! Methods (camelCase params):
//! - `engine.execute {command, params?, wait?}` / `engine.commands {filter?}`
//! - `jobs.list` / `jobs.cancel {job?}`: background jobs (#210)
//! - `session.list`, `doc.open {path}`, `doc.new {…file.new params}`,
//!   `doc.place {path, scale?, fit?, center?, index?}` (#2810: the read root is the scope,
//!   the counterpart of the refused ambient `file.placeEmbedded`),
//!   `doc.exportLayers {dir, format?, prefix?, visibleOnly?, quality?, index?}`,
//!   `doc.exportDataSets {dir, format?, dataSets?, naming?, index?}`,
//!   `doc.importDataSets {path, delimiter?, index?}` (#3216: the same treatment for the export
//!   and CSV half of #2810), `doc.save {path?, format?, quality?, index?}`, `doc.inspect {index?}`,
//!   `doc.render {index?, maxSide?, path?}` (writes a PNG to `path`, else
//!   returns it base64-encoded), `doc.select {index}`, `doc.close {index?}`
//! - `batch {steps: [{command, params?, wait?} | {method, params?}], stopOnError?}`
//! - `methods`: this list.

use std::io::{BufRead, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};

use base64::Engine as _;
use serde_json::{Value, json};

use crate::budgets::{BatchReplyBudget, write_reply};
use crate::security::{
    ConnectionLimiter, LineRead, MAX_BATCH_STEPS, MAX_CONNECTIONS, MAX_REQUEST_BYTES, authentication_reply, configure_stream, discard_rest_of_line,
    read_bounded_line, repeat_auth_reply,
};
use crate::{AutomationError, Headless};

/// Method names served by [`Headless::handle`].
pub const METHODS: &[&str] = &[
    "engine.execute",
    "jobs.list",
    "jobs.cancel",
    "engine.commands",
    "session.list",
    "doc.open",
    "doc.place",
    "doc.new",
    "doc.save",
    "doc.exportLayers",
    "doc.exportDataSets",
    "doc.importDataSets",
    "doc.inspect",
    "doc.render",
    "doc.select",
    "doc.close",
    "batch",
    "methods",
];

fn bad(msg: impl Into<String>) -> AutomationError {
    AutomationError::BadRequest(msg.into())
}

/// The optional `index` param: missing or null means the active document; anything else must be a
/// non-negative integer (never silently fall back to the active document, which `doc.close` would close).
fn index_of(p: &Value) -> Result<Option<usize>, AutomationError> {
    match p.get("index") {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v.as_u64().and_then(|i| usize::try_from(i).ok()).map(Some).ok_or_else(|| bad("`index` must be a non-negative integer")),
    }
}

fn str_of<'a>(p: &'a Value, k: &str) -> Option<&'a str> {
    p.get(k).and_then(Value::as_str)
}

impl Headless {
    /// Dispatch one request. Unknown methods and bad params are errors, never panics.
    pub fn handle(&mut self, method: &str, params: Value) -> Result<Value, AutomationError> {
        self.sync_jobs();
        let p = if params.is_null() { json!({}) } else { params };
        match method {
            "engine.execute" => {
                let id = str_of(&p, "command").ok_or_else(|| bad("engine.execute needs `command`"))?;
                let wait = p.get("wait").and_then(Value::as_bool).unwrap_or(true);
                self.command_start(id, p.get("params").cloned().unwrap_or(Value::Null), wait)
            }
            // Background jobs (#210): list (applying any that finished) and cancel.
            "jobs.list" => self.command_start("jobs.list", json!({}), true),
            "jobs.cancel" => self.command_start("jobs.cancel", p, true),
            "engine.commands" => {
                let all = self.command_list();
                Ok(match str_of(&p, "filter").map(str::to_lowercase) {
                    None => all,
                    Some(n) => Value::Array(
                        all.as_array()
                            .into_iter()
                            .flatten()
                            .filter(|c| {
                                let hay = format!("{} {}", c["id"].as_str().unwrap_or(""), c["label"].as_str().unwrap_or(""));
                                hay.to_lowercase().contains(&n)
                            })
                            .cloned()
                            .collect(),
                    ),
                })
            }
            "session.list" => Ok(self.session_list()),
            "doc.open" => {
                let path = str_of(&p, "path").ok_or_else(|| bad("doc.open needs `path`"))?;
                self.open(&PathBuf::from(path))
            }
            "doc.place" => {
                let path = str_of(&p, "path").ok_or_else(|| bad("doc.place needs `path`"))?;
                let mut opts = p.as_object().cloned().unwrap_or_default();
                opts.remove("path");
                opts.remove("index");
                self.place(index_of(&p)?, path, &Value::Object(opts))
            }
            "doc.new" => self.command_run("file.new", p),
            "doc.save" => {
                let tga_bits = photocraft_engine::file_cmds::tga_bits_param(&p, "doc.save").map_err(|e| bad(e.to_string()))?;
                let mut opts =
                    photocraft_io::ExportOptions { tiff_layers: p.get("tiffLayers").and_then(Value::as_bool).unwrap_or(false), tga_bits, ..Default::default() };
                opts.encode.dpx_bits = photocraft_engine::file_cmds::dpx_bits_param(&p, "doc.save").map_err(|e| bad(e.to_string()))?;
                if let Some(q) = p.get("quality").and_then(Value::as_u64) {
                    opts.encode.jpeg_quality = q.clamp(1, 100) as u8;
                    opts.encode.webp_quality = q.clamp(1, 100) as u8;
                    opts.encode.webp_lossless = false;
                }
                let path = str_of(&p, "path").map(PathBuf::from);
                self.save(index_of(&p)?, path.as_deref(), str_of(&p, "format"), &opts)
            }
            "doc.exportLayers" | "doc.exportDataSets" => {
                let dir = str_of(&p, "dir").ok_or_else(|| bad(format!("{method} needs `dir`")))?;
                let mut opts = p.as_object().cloned().unwrap_or_default();
                opts.remove("dir");
                opts.remove("index");
                let opts = Value::Object(opts);
                if method == "doc.exportLayers" { self.export_layers(index_of(&p)?, dir, &opts) } else { self.export_data_sets(index_of(&p)?, dir, &opts) }
            }
            "doc.importDataSets" => {
                let path = str_of(&p, "path").ok_or_else(|| bad("doc.importDataSets needs `path`"))?;
                let mut opts = p.as_object().cloned().unwrap_or_default();
                opts.remove("path");
                opts.remove("index");
                self.import_data_sets(index_of(&p)?, path, &Value::Object(opts))
            }
            "doc.inspect" => self.inspect(index_of(&p)?),
            "doc.render" => {
                let max = match p.get("maxSide") {
                    None => 1024,
                    Some(value) => {
                        value.as_u64().and_then(|value| u32::try_from(value).ok()).ok_or_else(|| bad("maxSide must be an unsigned 32-bit integer"))?
                    }
                };
                let png = self.render_png(index_of(&p)?, max)?;
                match str_of(&p, "path") {
                    Some(path) => {
                        self.write_render(&PathBuf::from(path), &png)?;
                        Ok(json!({"path": path, "bytes": png.len()}))
                    }
                    None => Ok(json!({
                        "mime": "image/png",
                        "base64": base64::engine::general_purpose::STANDARD.encode(&png),
                    })),
                }
            }
            "doc.select" => {
                let i = index_of(&p)?.ok_or_else(|| bad("doc.select needs `index`"))?;
                self.select(i)
            }
            "doc.close" => self.close(index_of(&p)?),
            "batch" => self.batch(&p),
            "methods" => Ok(json!(METHODS)),
            other => Err(bad(format!("unknown method `{other}` (try `methods`)"))),
        }
    }

    /// Run `steps` in order. Each step is `{command, params?, wait?}` (an engine command) or
    /// `{method, params?}` (any [`METHODS`] entry). Stops at the first error unless
    /// `stopOnError` is false; the reply lists every step's result.
    pub fn batch(&mut self, p: &Value) -> Result<Value, AutomationError> {
        self.batch_with_budget(p, BatchReplyBudget::default())
    }

    /// [`Headless::batch`] with a caller's reply budget (MCP charges escaped sizes, see
    /// [`BatchReplyBudget::escaped`]). Steps stop once the budget runs out.
    pub fn batch_with_budget(&mut self, p: &Value, mut reply_budget: BatchReplyBudget) -> Result<Value, AutomationError> {
        let steps = p.get("steps").and_then(Value::as_array).ok_or_else(|| bad("batch needs `steps`"))?;
        if steps.len() > MAX_BATCH_STEPS {
            return Err(bad(format!("batch contains {} steps; maximum is {MAX_BATCH_STEPS}", steps.len())));
        }
        let stop = p.get("stopOnError").and_then(Value::as_bool).unwrap_or(true);
        let mut results = Vec::with_capacity(steps.len());
        let mut failed = 0usize;
        for (i, s) in steps.iter().enumerate() {
            let params = s.get("params").cloned().unwrap_or(Value::Null);
            let r = if let Some(c) = str_of(s, "command") {
                // Like `engine.execute`: `wait: false` starts a long command as a background job.
                let wait = s.get("wait").and_then(Value::as_bool).unwrap_or(true);
                self.command_start(c, params, wait)
            } else if let Some(m) = str_of(s, "method") {
                if m == "batch" { Err(bad("nested batch")) } else { self.handle(m, params) }
            } else {
                Err(bad(format!("step {i} needs `command` or `method`")))
            };
            let was_error = r.is_err();
            let result = match r {
                Ok(v) => json!({"ok": true, "result": v}),
                Err(e) => json!({"ok": false, "error": e.to_string()}),
            };
            if let Err(error) = reply_budget.charge(&result) {
                failed += 1;
                results.push(json!({"ok": false, "error": format!("batch response budget exceeded; current step may have completed: {error}")}));
                break;
            }
            results.push(result);
            if was_error {
                failed += 1;
                if stop {
                    break;
                }
            }
        }
        Ok(json!({"completed": results.len() - failed, "failed": failed, "results": results}))
    }
}

/// Answer one request line. Malformed JSON gets an error reply with `id: null`.
pub fn respond(h: &Mutex<Headless>, line: &str) -> Value {
    let req: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => return json!({"id": null, "ok": false, "error": format!("bad JSON: {e}")}),
    };
    let id = req.get("id").cloned().unwrap_or(Value::Null);
    let Some(method) = req.get("method").and_then(Value::as_str) else {
        return json!({"id": id, "ok": false, "error": "missing `method`"});
    };
    let params = req.get("params").cloned().unwrap_or(Value::Null);
    // Last-resort guard (AGENTS.md, Never crash): a command that panics answers this request
    // with an error instead of taking down the connection (or the whole stdio server). Edits
    // run on a copy of the document (`Session::edit`), so the session stays consistent and a
    // poisoned lock is safe to keep using, the same as the MCP server does.
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| h.lock().unwrap_or_else(PoisonError::into_inner).handle(method, params)))
        .unwrap_or_else(|_| Err(AutomationError::Other(format!("internal error: `{method}` panicked"))));
    match r {
        Ok(v) => json!({"id": id, "ok": true, "result": v}),
        Err(e) => json!({"id": id, "ok": false, "error": e.to_string()}),
    }
}

/// Serve JSON lines from `r` to `w` until EOF. Blank lines are ignored. An over-long or non-UTF-8
/// line gets an error reply and is skipped; the session (and its open documents) keeps serving.
pub fn serve_lines(h: &Mutex<Headless>, mut r: impl BufRead, mut w: impl Write) -> std::io::Result<()> {
    let mut line = String::new();
    loop {
        let (reply, unread_rest) = match read_bounded_line(&mut r, &mut line) {
            Ok(LineRead::Eof) => return Ok(()),
            Ok(LineRead::TooLong) => (json!({"id": null, "ok": false, "error": format!("request exceeds {MAX_REQUEST_BYTES} bytes")}), !line.ends_with('\n')),
            Err(e) if e.kind() == std::io::ErrorKind::InvalidData => (json!({"id": null, "ok": false, "error": "request is not valid UTF-8"}), false),
            Err(e) => return Err(e),
            Ok(LineRead::Line) if line.trim().is_empty() => continue,
            Ok(LineRead::Line) => (respond(h, &line), false),
        };
        write_reply(&mut w, &reply)?;
        w.flush()?;
        if unread_rest {
            discard_rest_of_line(&mut r)?;
        }
    }
}

fn serve_tcp_connection(h: &Mutex<Headless>, stream: std::net::TcpStream, token: &str) -> std::io::Result<()> {
    configure_stream(&stream)?;
    let read = stream.try_clone()?;
    let mut reader = std::io::BufReader::new(read);
    let mut out = stream;
    let mut line = String::new();
    let mut authenticated = false;
    loop {
        match read_bounded_line(&mut reader, &mut line)? {
            LineRead::Eof => return Ok(()),
            LineRead::TooLong => {
                let reply = json!({
                    "id": null,
                    "ok": false,
                    "error": format!("request exceeds {MAX_REQUEST_BYTES} bytes"),
                });
                write_reply(&mut out, &reply)?;
                out.flush()?;
                return Ok(());
            }
            LineRead::Line if line.trim().is_empty() => continue,
            LineRead::Line => {}
        }
        let reply = if authenticated {
            repeat_auth_reply(&line).unwrap_or_else(|| respond(h, &line))
        } else {
            let (reply, ok) = authentication_reply(&line, token);
            authenticated = ok;
            reply
        };
        write_reply(&mut out, &reply)?;
        out.flush()?;
        if !authenticated {
            return Ok(());
        }
    }
}

/// Serve authenticated JSON lines on a loopback TCP address with one shared session.
/// Refuses non-loopback addresses and caps active connections and request bytes.
pub fn serve_tcp(addr: &str, h: Arc<Mutex<Headless>>, token: String, ready: impl FnOnce(std::net::SocketAddr)) -> Result<(), AutomationError> {
    let listener = TcpListener::bind(addr).map_err(|e| AutomationError::Io(format!("bind {addr}: {e}")))?;
    let local = listener.local_addr().map_err(|e| AutomationError::Io(e.to_string()))?;
    if !local.ip().is_loopback() {
        return Err(bad(format!("{addr} is not a loopback address")));
    }
    ready(local);
    let limiter = ConnectionLimiter::new(MAX_CONNECTIONS);
    let token = Arc::new(token);
    for stream in listener.incoming() {
        let Ok(mut stream) = stream else { continue };
        let Some(permit) = limiter.try_acquire() else {
            let _ = configure_stream(&stream);
            let _ = writeln!(stream, "{}", json!({"id": null, "ok": false, "error": "connection limit reached"}));
            continue;
        };
        let h = h.clone();
        let token = Arc::clone(&token);
        std::thread::spawn(move || {
            let _permit = permit;
            let _ = serve_tcp_connection(&h, stream, &token);
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> Mutex<Headless> {
        Mutex::new(Headless::trusted_local())
    }

    #[test]
    fn execute_reports_invalid_filter_parameters_without_changing_history() {
        let h = session();
        assert_eq!(respond(&h, r#"{"method":"doc.new","params":{"width":8,"height":8}}"#)["ok"], true);
        let before = h.lock().unwrap().inspect(None).unwrap();
        for (params, key) in
            [(json!({"radius":"big"}), "radius"), (json!({"radus":3}), "radus"), (json!({"radius":5000}), "radius"), (json!({"radius":-4}), "radius")]
        {
            for wait in [true, false] {
                let request = json!({"method":"engine.execute","params":{"command":"filter.blur.gaussianBlur","params":params,"wait":wait}});
                let reply = respond(&h, &request.to_string());
                assert_eq!(reply["ok"], false);
                let error = reply["error"].as_str().unwrap();
                assert!(error.contains("filter.blur.gaussianBlur") && error.contains(key), "{error}");
                assert_eq!(h.lock().unwrap().inspect(None).unwrap(), before);
            }
        }
    }

    /// Pins the reply shape documented in docs/control-protocol.md › Command list replies
    /// (#3684): a bare JSON array (not `{"commands": [...]}`), one object per command with
    /// exactly these keys, from `engine.commands` (filtered or not) and `command.list`.
    #[test]
    fn command_list_replies_are_documented_array_of_objects() {
        let h = session();
        let mut g = h.lock().unwrap();
        let all = g.handle("engine.commands", json!({})).unwrap();
        let filtered = g.handle("engine.commands", json!({"filter": "BLUR"})).unwrap();
        let engine = g.handle("engine.execute", json!({"command": "command.list"})).unwrap();
        for (name, reply) in [("engine.commands", &all), ("engine.commands {filter}", &filtered), ("command.list", &engine)] {
            let items = reply.as_array().unwrap_or_else(|| panic!("{name} must reply with a bare array, got {reply}"));
            assert!(!items.is_empty(), "{name}");
            for c in items {
                let mut keys: Vec<&str> = c.as_object().unwrap().keys().map(String::as_str).collect();
                keys.sort_unstable();
                assert_eq!(keys, ["disabledReason", "enabled", "id", "label", "menu", "needsDocument", "params", "shortcut"], "{name}: {c}");
                assert!(c["id"].is_string() && c["label"].is_string() && c["params"].is_string() && c["enabled"].is_boolean(), "{c}");
                assert!(c["needsDocument"].is_boolean(), "{c}");
                assert_eq!(c["disabledReason"].is_null(), c["enabled"] == true, "disabledReason is null exactly when enabled: {c}");
                assert!(c["menu"].as_array().unwrap().iter().all(Value::is_string), "{c}");
                assert!(c["shortcut"].is_string() || c["shortcut"].is_null(), "{c}");
            }
        }
        let n_all = all.as_array().unwrap().len();
        assert_eq!(engine.as_array().unwrap().len(), n_all);
        let filtered = filtered.as_array().unwrap();
        assert!(filtered.len() < n_all);
        assert!(filtered.iter().all(|c| format!("{} {}", c["id"], c["label"]).to_lowercase().contains("blur")));
        let new = all.as_array().unwrap().iter().find(|c| c["id"] == "file.new").unwrap();
        assert_eq!((&new["menu"], &new["shortcut"], &new["enabled"]), (&json!(["File"]), &json!("Cmd+N"), &json!(true)), "{new}");
        assert_eq!((&new["disabledReason"], &new["needsDocument"]), (&Value::Null, &json!(false)), "{new}");
    }

    #[test]
    fn lines_round_trip_and_errors() {
        let h = session();
        let input = concat!(
            r#"{"id":1,"method":"doc.new","params":{"width":64,"height":32,"background":"white"}}"#,
            "\n",
            "\n",
            r#"{"id":2,"method":"engine.execute","params":{"command":"layer.new.layer","params":{"name":"Ink"}}}"#,
            "\n",
            r#"{"id":3,"method":"doc.inspect"}"#,
            "\n",
            r#"{"id":4,"method":"nope"}"#,
            "\n",
            "not json\n",
        );
        let mut out = Vec::new();
        serve_lines(&h, input.as_bytes(), &mut out).unwrap();
        let replies: Vec<Value> = String::from_utf8(out).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(replies.len(), 5);
        assert!(replies[0]["ok"].as_bool().unwrap());
        assert_eq!(replies[1]["id"], 2);
        assert!(replies[2]["result"].to_string().contains("Ink"));
        assert!(!replies[3]["ok"].as_bool().unwrap());
        assert!(replies[3]["error"].as_str().unwrap().contains("unknown method"));
        assert_eq!(replies[4]["id"], Value::Null);
    }

    /// Regression: after a panic while the session lock was held, every later request
    /// failed with "session lock poisoned"; the server now keeps serving.
    #[test]
    fn poisoned_session_keeps_serving() {
        let h = session();
        let _ = std::thread::scope(|sc| {
            sc.spawn(|| {
                let _g = h.lock().unwrap();
                panic!("poison the session lock");
            })
            .join()
        });
        assert!(h.is_poisoned());
        let r = respond(&h, r#"{"id":1,"method":"doc.new","params":{"width":8,"height":8}}"#);
        assert_eq!(r["ok"], true, "{r}");
    }

    #[test]
    fn batch_stops_on_error_unless_told_not_to() {
        let h = session();
        let mut g = h.lock().unwrap();
        g.handle("doc.new", json!({"width": 16, "height": 16})).unwrap();
        let steps = json!([
            {"command": "layer.new.layer", "params": {"name": "A"}},
            {"command": "no.such.command"},
            {"command": "layer.new.layer", "params": {"name": "B"}},
            {"method": "doc.inspect"}
        ]);
        let r = g.handle("batch", json!({"steps": steps})).unwrap();
        assert_eq!(r["completed"], 1);
        assert_eq!(r["failed"], 1);
        assert_eq!(r["results"].as_array().unwrap().len(), 2);
        let r = g.handle("batch", json!({"steps": steps, "stopOnError": false})).unwrap();
        assert_eq!(r["completed"], 3);
        let insp = &r["results"][3]["result"];
        assert!(insp.to_string().contains("\"B\""));
    }

    #[test]
    fn batch_rejects_too_many_steps() {
        let h = session();
        let mut g = h.lock().unwrap();
        let steps = vec![json!({"command": "command.list"}); MAX_BATCH_STEPS + 1];
        let error = g.handle("batch", json!({"steps": steps})).unwrap_err();
        assert!(error.to_string().contains("maximum is 256"));
    }

    #[test]
    fn render_to_file_and_base64() {
        let h = session();
        let mut g = h.lock().unwrap();
        g.handle("doc.new", json!({"width": 40, "height": 20, "background": "black"})).unwrap();
        let b = g.handle("doc.render", json!({"maxSide": 10})).unwrap();
        let png = base64::engine::general_purpose::STANDARD.decode(b["base64"].as_str().unwrap()).unwrap();
        let img = photocraft_codecs::decode(&png).unwrap();
        assert_eq!(img.dimensions(), (10, 5));
        let path = std::env::temp_dir().join(format!("pc-rpc-{}.png", std::process::id()));
        let r = g.handle("doc.render", json!({"path": path.to_string_lossy(), "maxSide": 0})).unwrap();
        assert!(r["bytes"].as_u64().unwrap() > 0);
        assert_eq!(photocraft_codecs::decode(&std::fs::read(&path).unwrap()).unwrap().dimensions(), (40, 20));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn background_jobs_list_wait_and_cancel() {
        let mut h = Headless::new();
        h.handle("doc.new", json!({"width": 600, "height": 400})).unwrap();
        h.handle("engine.execute", json!({"command": "layer.new.layer"})).unwrap();
        h.handle("engine.execute", json!({"command": "edit.fill", "params": {"color": "#808080"}})).unwrap();
        // Default: waits, the result is the command's.
        let r = h.handle("engine.execute", json!({"command": "filter.noise.addNoise", "params": {"amount": 20}})).unwrap();
        assert!(r["filter"].is_object(), "{r}");
        // wait: false returns a job id at once.
        let r = h.handle("engine.execute", json!({"command": "filter.blur.gaussianBlur", "params": {"radius": 40}, "wait": false})).unwrap();
        let job = r["job"].as_u64().unwrap();
        let listed = h.handle("jobs.list", json!({})).unwrap();
        assert!(listed["jobs"].as_array().unwrap().iter().any(|j| j["id"] == job), "{listed}");
        // Wait for it via jobs.list (which applies finished jobs).
        let t = std::time::Instant::now();
        loop {
            let l = h.handle("jobs.list", json!({})).unwrap();
            let state = l["jobs"].as_array().unwrap().iter().find(|j| j["id"] == job).map(|j| j["state"].clone());
            if state == Some(json!("done")) {
                break;
            }
            assert!(t.elapsed().as_secs() < 60, "{l}");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        // Cancel: a second job, cancelled at once, leaves the document unchanged.
        let before = h.session.active().unwrap().doc.clone();
        let r = h.handle("engine.execute", json!({"command": "filter.blur.gaussianBlur", "params": {"radius": 40}, "wait": false})).unwrap();
        let job = r["job"].as_u64().unwrap();
        h.handle("jobs.cancel", json!({"job": job})).unwrap();
        assert!(h.handle("jobs.cancel", json!({"job": "nope"})).is_err());
        let l = h.handle("jobs.list", json!({})).unwrap();
        assert_eq!(l["jobs"].as_array().unwrap().iter().find(|j| j["id"] == job).unwrap()["state"], "cancelled");
        assert!(std::sync::Arc::ptr_eq(&h.session.active().unwrap().doc, &before));
    }

    #[test]
    fn batch_steps_can_start_background_jobs() {
        let mut h = Headless::new();
        h.handle("doc.new", json!({"width": 600, "height": 400})).unwrap();
        let steps = json!([
            {"command": "layer.new.layer"},
            {"command": "edit.fill", "params": {"color": "#808080"}},
            {"command": "filter.blur.gaussianBlur", "params": {"radius": 40}, "wait": false}
        ]);
        let r = h.handle("batch", json!({"steps": steps})).unwrap();
        assert_eq!(r["completed"], 3, "{r}");
        let job = r["results"][2]["result"]["job"].as_u64().unwrap_or_else(|| panic!("no job id: {r}"));
        assert_eq!(r["results"][2]["result"]["pending"], true);
        let listed = h.handle("jobs.list", json!({})).unwrap();
        assert!(listed["jobs"].as_array().unwrap().iter().any(|j| j["id"] == job), "{listed}");
        let t = std::time::Instant::now();
        while h.session.jobs().iter().any(|j| j.id.0 == job) {
            assert!(t.elapsed().as_secs() < 60);
            std::thread::sleep(std::time::Duration::from_millis(5));
            h.handle("jobs.list", json!({})).unwrap();
        }
    }

    /// Regression (#503): a finished `wait: false` job was only applied by a later command or
    /// `jobs.list`, so save, inspect, render and `session.list` still saw the document without it.
    #[test]
    fn every_request_sees_a_finished_background_job() {
        let dir = std::env::temp_dir().join(format!("pc-rpc-job-sync-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (before, after) = (dir.join("before.png"), dir.join("after.png"));
        let mut h = Headless::trusted_local();
        h.handle("doc.new", json!({"width": 64, "height": 48})).unwrap();
        h.handle("engine.execute", json!({"command": "filter.noise.addNoise", "params": {"amount": 50}})).unwrap();
        h.handle("doc.save", json!({"path": before.to_string_lossy()})).unwrap();
        let render = |h: &mut Headless| h.handle("doc.render", json!({"maxSide": 0})).unwrap()["base64"].clone();
        let unblurred = render(&mut h);
        let revision = |h: &mut Headless| h.handle("session.list", json!({})).unwrap()["documents"][0]["revision"].clone();
        let start = revision(&mut h);
        h.handle("engine.execute", json!({"command": "filter.blur.gaussianBlur", "params": {"radius": 4}, "wait": false})).unwrap();
        // Only reading requests from here on: no command and no `jobs.list`.
        let t = std::time::Instant::now();
        while revision(&mut h) == start {
            assert!(t.elapsed().as_secs() < 60, "session.list never saw the finished job");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let history = h.handle("doc.inspect", json!({})).unwrap()["history"].clone();
        assert_eq!(history.as_array().unwrap().last().unwrap(), "Gaussian Blur", "{history}");
        h.handle("doc.save", json!({"path": after.to_string_lossy()})).unwrap();
        let decode = |p: &std::path::Path| photocraft_codecs::decode(&std::fs::read(p).unwrap()).unwrap().to_rgba8();
        assert_ne!(decode(&before), decode(&after), "the export holds the blur");
        assert_ne!(render(&mut h), unblurred, "the preview holds the blur");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn automation_preview_rejects_oversized_full_size_and_numeric_wraparound() {
        let mut h = Headless::new();
        h.handle("doc.new", json!({"width": 2049, "height": 1})).unwrap();
        for max in [json!(0), json!(2049), json!(u64::from(u32::MAX) + 1), json!(-1), json!(1.5)] {
            assert!(h.handle("doc.render", json!({"maxSide": max})).is_err());
        }
        // A smaller preview still works, and rejection leaves the source document intact.
        let rendered = h.handle("doc.render", json!({"maxSide": 32})).unwrap();
        let png = base64::engine::general_purpose::STANDARD.decode(rendered["base64"].as_str().unwrap()).unwrap();
        assert_eq!(photocraft_codecs::decode(&png).unwrap().dimensions(), (32, 1));
        assert_eq!(h.session.active().unwrap().doc.size.width, 2049);
    }

    #[test]
    fn stdio_rejects_oversized_frames_before_dispatch() {
        let h = Mutex::new(Headless::new());
        let input = " ".repeat(MAX_REQUEST_BYTES + 1);
        let mut out = Vec::new();
        serve_lines(&h, input.as_bytes(), &mut out).unwrap();
        let reply: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(reply["ok"], false);
        assert!(reply["error"].as_str().unwrap().contains("request exceeds"));
        assert!(h.lock().unwrap().session.documents().is_empty());
    }

    /// Regression (#505): an over-long or non-UTF-8 line ended the stdio session, losing every
    /// open document. Now it gets one error reply, the rest of the line is skipped (never
    /// dispatched), and the next request is served.
    #[test]
    fn stdio_skips_a_rejected_line_and_keeps_the_session() {
        let h = Mutex::new(Headless::new());
        let mut input = b"{\"id\":1,\"method\":\"doc.new\",\"params\":{\"width\":8,\"height\":8}}\n".to_vec();
        input.extend(" ".repeat(MAX_REQUEST_BYTES + 1).into_bytes());
        input.extend(b"{\"id\":9,\"method\":\"doc.close\"}\n");
        input.extend(b"\xff\xfe{\"id\":8,\"method\":\"doc.close\"}\n");
        input.extend(b"{\"id\":2,\"method\":\"session.list\"}\n");
        let mut out = Vec::new();
        serve_lines(&h, input.as_slice(), &mut out).unwrap();
        let replies: Vec<Value> = String::from_utf8(out).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(replies.len(), 4, "{replies:?}");
        assert_eq!(replies[0]["ok"], true);
        assert!(replies[1]["error"].as_str().unwrap().contains("request exceeds"), "{}", replies[1]);
        assert!(replies[2]["error"].as_str().unwrap().contains("not valid UTF-8"), "{}", replies[2]);
        assert_eq!(replies[3]["id"], 2);
        assert_eq!(replies[3]["result"]["documents"].as_array().unwrap().len(), 1, "the document stays open");
    }

    #[test]
    fn preview_rejection_precedes_output_file_creation() {
        let root = std::env::temp_dir().join(format!("pc-preview-budget-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let workspace = crate::AuthorizedWorkspace::new(None, Some(&root)).unwrap();
        let mut h = Headless::with_workspace(workspace);
        h.handle("doc.new", json!({"width": 16, "height": 8})).unwrap();
        let result = h.handle("doc.render", json!({"maxSide": 2049, "path": "rejected.png"}));
        assert!(result.unwrap_err().to_string().contains("preview side exceeds"));
        assert!(!root.join("rejected.png").exists());
        assert!(h.handle("doc.render", json!({"maxSide": 16, "path": "accepted.png"})).is_ok());
        assert!(root.join("accepted.png").exists());
        // Release directory capabilities before cleanup on Windows.
        drop(h);
        std::fs::remove_file(root.join("accepted.png")).unwrap();
        std::fs::remove_dir(root).unwrap();
    }

    #[test]
    fn doc_place_reads_through_the_read_root_and_never_outside_it() {
        let base = std::env::temp_dir().join(format!("pc-doc-place-{}", std::process::id()));
        let root = base.join("workspace");
        let outside = base.join("outside");
        for dir in [&root, &outside] {
            std::fs::create_dir_all(dir).unwrap();
        }
        let img = photocraft_codecs::Image::from_u8(8, 6, photocraft_codecs::ChannelLayout::Rgb, vec![42; 8 * 6 * 3]).unwrap();
        let png = photocraft_codecs::encode(&img, photocraft_codecs::Format::Png, &Default::default()).unwrap();
        std::fs::write(root.join("logo.png"), &png).unwrap();
        std::fs::write(outside.join("logo.png"), &png).unwrap();
        let absolute = outside.join("logo.png").to_string_lossy().into_owned();

        let mut h = Headless::with_workspace(crate::AuthorizedWorkspace::new(Some(&root), Some(&root)).unwrap());
        h.handle("doc.new", json!({"width": 40, "height": 30})).unwrap();
        let placed = h.handle("doc.place", json!({"path": "logo.png", "scale": 50})).unwrap();
        assert!(placed["layer"].as_u64().is_some(), "{placed}");
        assert_eq!(placed["scale"].as_f64(), Some(50.0), "the options reach the command: {placed}");
        let inspected = h.handle("doc.inspect", json!({})).unwrap();
        assert_eq!(inspected["layers"].as_array().map(Vec::len), Some(2), "the placed layer is in the document: {inspected}");
        // A path outside the read root is refused the way the ambient command is (#2810).
        for (path, reason) in [("../outside/logo.png", "parent traversal"), (absolute.as_str(), "must be inside the automation root")] {
            let refused = h.handle("doc.place", json!({"path": path})).unwrap_err().to_string();
            assert!(refused.contains(reason), "{refused}");
            let inspected = h.handle("doc.inspect", json!({})).unwrap();
            assert_eq!(inspected["layers"].as_array().map(Vec::len), Some(2), "nothing was placed: {refused}");
        }
        // Trusted local keeps the ambient command: that caller may read the path itself.
        let mut t = Headless::trusted_local();
        t.handle("doc.new", json!({"width": 40, "height": 30})).unwrap();
        let trusted = t.handle("doc.place", json!({"path": root.join("logo.png").to_string_lossy()})).unwrap();
        assert!(trusted["layer"].as_u64().is_some(), "{trusted}");
        // Release directory capabilities before cleanup on Windows.
        drop(t);
        drop(h);
        std::fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn doc_export_layers_writes_only_beneath_the_write_root() {
        let base = std::env::temp_dir().join(format!("pc-doc-export-layers-{}", std::process::id()));
        let root = base.join("workspace");
        std::fs::create_dir_all(&root).unwrap();
        let mut h = Headless::with_workspace(crate::AuthorizedWorkspace::new(Some(&root), Some(&root)).unwrap());
        h.handle("doc.new", json!({"width": 12, "height": 8, "background": "white", "name": "card"})).unwrap();
        h.handle("engine.execute", json!({"command": "layer.new.layer", "params": {"name": "Badge"}})).unwrap();

        let out = root.join("out");
        let exported = h.handle("doc.exportLayers", json!({"dir": out.to_string_lossy(), "format": "png", "prefix": "card"})).unwrap();
        assert_eq!(exported["count"].as_u64(), Some(2), "{exported}");
        let mut names: Vec<String> = std::fs::read_dir(&out).unwrap().map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned()).collect();
        names.sort();
        assert_eq!(names.len(), 2, "{exported}");
        for name in names {
            assert!(name.starts_with("card_") && name.ends_with(".png"), "{name}");
            let bytes = std::fs::read(out.join(&name)).unwrap();
            assert_eq!(photocraft_codecs::decode(&bytes).unwrap().dimensions(), (12, 8), "{name}");
        }
        // Outside the write root nothing is created, not even the directory.
        let escaped = base.join("escaped");
        let refused = h.handle("doc.exportLayers", json!({"dir": escaped.to_string_lossy()})).unwrap_err().to_string();
        assert!(refused.contains("must be inside the automation root"), "{refused}");
        assert!(!escaped.exists());
        // The ambient command keeps its refusal for the very same directory.
        let ambient =
            h.handle("engine.execute", json!({"command": "file.export.layersToFiles", "params": {"dir": out.to_string_lossy()}})).unwrap_err().to_string();
        assert!(ambient.contains("ambient filesystem paths"), "{ambient}");
        // Release directory capabilities before cleanup on Windows.
        drop(h);
        std::fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn doc_export_data_sets_refuse_collisions_and_import_reads_the_root() {
        let base = std::env::temp_dir().join(format!("pc-doc-data-sets-{}", std::process::id()));
        let root = base.join("workspace");
        std::fs::create_dir_all(&root).unwrap();
        let mut h = Headless::with_workspace(crate::AuthorizedWorkspace::new(Some(&root), Some(&root)).unwrap());
        h.handle("doc.new", json!({"width": 8, "height": 8, "background": "white", "name": "card"})).unwrap();
        let inspected = h.handle("doc.inspect", json!({})).unwrap();
        let layer = inspected["layers"][0]["id"].as_u64().expect("the layer id");
        h.handle("engine.execute", json!({"command": "image.variables.define", "params": {"defs": [{"name": "show", "layer": layer, "type": "visibility"}]}}))
            .unwrap();
        h.handle(
            "engine.execute",
            json!({"command": "image.variables.dataSets", "params": {"dataSets": [
                {"name": "one", "values": [{"variable": "show", "kind": "visibility", "value": true}]},
                {"name": "two", "values": [{"variable": "show", "kind": "visibility", "value": false}]},
            ]}}),
        )
        .unwrap();

        let sets = root.join("sets");
        let exported = h.handle("doc.exportDataSets", json!({"dir": sets.to_string_lossy(), "format": "png"})).unwrap();
        assert_eq!(exported["count"].as_u64(), Some(2), "{exported}");
        assert_eq!(std::fs::read_dir(&sets).unwrap().count(), 2, "{exported}");
        // Colliding names are refused as in the command, before any file is written.
        let clash = root.join("clash");
        let refused = h.handle("doc.exportDataSets", json!({"dir": clash.to_string_lossy(), "naming": "{document}"})).unwrap_err().to_string();
        assert!(refused.contains("{index}"), "{refused}");
        assert_eq!(std::fs::read_dir(&clash).unwrap().count(), 0, "no file of the refused batch");
        // A CSV from the read root, and one outside it.
        std::fs::write(root.join("sets.csv"), "Data Set,show\none,true\ntwo,false\n").unwrap();
        let imported = h.handle("doc.importDataSets", json!({"path": "sets.csv"})).unwrap();
        assert_eq!(imported["imported"].as_u64(), Some(2), "{imported}");
        std::fs::write(base.join("outside.csv"), "Data Set,show\none,true\n").unwrap();
        let outside = h.handle("doc.importDataSets", json!({"path": base.join("outside.csv").to_string_lossy()})).unwrap_err().to_string();
        assert!(outside.contains("must be inside the automation root"), "{outside}");
        // Release directory capabilities before cleanup on Windows.
        drop(h);
        std::fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn trusted_local_render_keeps_full_size_behavior() {
        let mut h = Headless::trusted_local();
        h.handle("doc.new", json!({"width": 2049, "height": 1})).unwrap();
        let rendered = h.handle("doc.render", json!({"maxSide": 0})).unwrap();
        let png = base64::engine::general_purpose::STANDARD.decode(rendered["base64"].as_str().unwrap()).unwrap();
        assert_eq!(photocraft_codecs::decode(&png).unwrap().dimensions(), (2049, 1));
    }

    #[test]
    fn methods_lists_the_job_methods_and_every_listed_name_is_served() {
        let mut h = Headless::new();
        let listed = h.handle("methods", Value::Null).unwrap();
        let listed: Vec<&str> = listed.as_array().unwrap().iter().map(|m| m.as_str().unwrap()).collect();
        assert_eq!(listed, METHODS);
        // The job methods `engine.execute {"wait": false}` depends on are discoverable (#414).
        assert!(listed.contains(&"jobs.list") && listed.contains(&"jobs.cancel"));
        for m in METHODS {
            if let Err(e) = h.handle(m, Value::Null) {
                assert!(!e.to_string().contains("unknown method"), "{m} is listed but not served: {e}");
            }
        }
        assert!(h.handle("jobs.nope", Value::Null).unwrap_err().to_string().contains("unknown method"));
    }

    #[test]
    fn batch_response_budget_stops_later_steps_even_when_stop_on_error_is_false() {
        let mut h = Headless::new();
        let mut steps = vec![json!({"command": "command.list"}); MAX_BATCH_STEPS - 1];
        steps.push(json!({"command": "file.new", "params": {"width": 8, "height": 8}}));
        let result = h.batch(&json!({"steps": steps, "stopOnError": false})).unwrap();
        assert_eq!(result["failed"], 1);
        assert!(result["results"].as_array().unwrap().last().unwrap()["error"].as_str().unwrap().contains("batch response budget exceeded"));
        assert!(h.session.documents().is_empty(), "later edit must not execute after the quota is exhausted");
        assert!(crate::budgets::json_bytes(&result).is_ok());
        assert!(h.handle("methods", Value::Null).is_ok());
    }

    #[test]
    fn tcp_serves_loopback() {
        use std::io::{BufReader, Write as _};
        const TOKEN: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let h = Arc::new(Mutex::new(Headless::new()));
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = serve_tcp("127.0.0.1:0", h, TOKEN.into(), move |a| tx.send(a).unwrap());
        });
        let addr = rx.recv().unwrap();
        let mut s = std::net::TcpStream::connect(addr).unwrap();
        writeln!(s, r#"{{"id":"auth","method":"auth","params":{{"token":"{TOKEN}"}}}}"#).unwrap();
        let mut reader = BufReader::new(s.try_clone().unwrap());
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        let auth: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(auth["result"]["authenticated"], true);
        writeln!(s, r#"{{"id":"a","method":"methods"}}"#).unwrap();
        line.clear();
        reader.read_line(&mut line).unwrap();
        let v: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["id"], "a");
        assert!(v["result"].as_array().unwrap().iter().any(|m| m == "batch"));
    }

    #[test]
    fn tcp_repeat_auth_reports_already_authenticated_and_keeps_serving() {
        use std::io::{BufReader, Write as _};
        const TOKEN: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let h = Arc::new(Mutex::new(Headless::new()));
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = serve_tcp("127.0.0.1:0", h, TOKEN.into(), move |a| tx.send(a).unwrap());
        });
        let addr = rx.recv().unwrap();
        let mut s = std::net::TcpStream::connect(addr).unwrap();
        let mut reader = BufReader::new(s.try_clone().unwrap());
        let mut line = String::new();
        for id in ["auth", "again"] {
            writeln!(s, r#"{{"id":"{id}","method":"auth","params":{{"token":"{TOKEN}"}}}}"#).unwrap();
            line.clear();
            reader.read_line(&mut line).unwrap();
        }
        let again: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(again["id"], "again");
        assert_eq!(again["ok"], false);
        assert_eq!(again["error"], "already authenticated");
        writeln!(s, r#"{{"id":"m","method":"methods"}}"#).unwrap();
        line.clear();
        reader.read_line(&mut line).unwrap();
        let v: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["id"], "m");
        assert_eq!(v["ok"], true);
    }

    #[test]
    fn tcp_rejects_requests_before_authentication() {
        use std::io::{BufReader, Write as _};
        const TOKEN: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let h = Arc::new(Mutex::new(Headless::new()));
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = serve_tcp("127.0.0.1:0", h, TOKEN.into(), move |a| tx.send(a).unwrap());
        });
        let addr = rx.recv().unwrap();
        let mut s = std::net::TcpStream::connect(addr).unwrap();
        writeln!(s, r#"{{"id":1,"method":"methods"}}"#).unwrap();
        let mut line = String::new();
        BufReader::new(s).read_line(&mut line).unwrap();
        let v: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["ok"], false);
        assert_eq!(v["error"], "authentication required");
        assert!(v.get("result").is_none());
    }

    #[test]
    fn invalid_document_index_is_rejected_not_the_active_document() {
        let h = session();
        for _ in 0..2 {
            assert_eq!(respond(&h, r#"{"method":"doc.new","params":{"width":8,"height":8}}"#)["ok"], true);
        }
        let count = |h: &Mutex<Headless>| h.lock().unwrap().session_list()["documents"].as_array().unwrap().len();
        for index in [json!("0"), json!(-1), json!(0.5), json!(true)] {
            for method in ["doc.close", "doc.inspect", "doc.select"] {
                let reply = respond(&h, &json!({"method": method, "params": {"index": index}}).to_string());
                assert_eq!(reply["ok"], false, "{method} {index}: {reply}");
                assert!(reply["error"].as_str().unwrap().contains("`index` must be a non-negative integer"), "{reply}");
            }
            assert_eq!(count(&h), 2, "index {index} closed a document");
        }
        // Null means "the active document", like a missing `index`.
        assert_eq!(respond(&h, r#"{"method":"doc.inspect","params":{"index":null}}"#)["ok"], true);
        // A valid index still closes that document.
        let reply = respond(&h, r#"{"method":"doc.close","params":{"index":0}}"#);
        assert_eq!(reply["ok"], true, "{reply}");
        assert_eq!(count(&h), 1);
    }
}
