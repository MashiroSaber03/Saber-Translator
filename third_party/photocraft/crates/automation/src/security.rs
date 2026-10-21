//! Shared security primitives for the loopback JSON control transports.

use std::fs::OpenOptions;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use serde_json::{Value, json};

use crate::AutomationError;

/// The first request on every TCP connection must use this method.
pub const AUTH_METHOD: &str = "auth";
/// Maximum encoded JSON request line, including its newline.
pub const MAX_REQUEST_BYTES: usize = 1 << 20;
/// Maximum simultaneously serviced TCP connections per listener.
pub const MAX_CONNECTIONS: usize = 16;
/// Maximum commands or method calls in one batch.
pub const MAX_BATCH_STEPS: usize = 256;
/// Idle/read and write timeout for loopback TCP connections.
pub const IO_TIMEOUT: Duration = Duration::from_secs(30);

const TOKEN_BYTES: usize = 32;
const TOKEN_HEX_LEN: usize = TOKEN_BYTES * 2;

/// Result of reading one bounded JSON-lines frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineRead {
    Eof,
    Line,
    TooLong,
}

/// Read one line without ever buffering more than [`MAX_REQUEST_BYTES`] plus one byte.
///
/// `TooLong` leaves the bytes read so far in `line` (lossily decoded); the rest of that line is
/// still unread unless `line` ends with a newline (see [`discard_rest_of_line`]). A line within
/// the limit that is not UTF-8 is an `InvalidData` error, and the whole line has been consumed.
pub fn read_bounded_line(reader: &mut impl BufRead, line: &mut String) -> std::io::Result<LineRead> {
    line.clear();
    let mut bytes = Vec::new();
    let n = std::io::Read::take(reader, (MAX_REQUEST_BYTES + 1) as u64).read_until(b'\n', &mut bytes)?;
    if n == 0 {
        Ok(LineRead::Eof)
    } else if n > MAX_REQUEST_BYTES {
        // A cut can split a multi-byte character; the length, not the text, decides.
        line.push_str(&String::from_utf8_lossy(&bytes));
        Ok(LineRead::TooLong)
    } else {
        *line = String::from_utf8(bytes).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(LineRead::Line)
    }
}

/// Skip the rest of the current line (through its newline, or to EOF) without buffering it, so
/// a stream transport can answer an over-long request and keep reading the next one.
pub fn discard_rest_of_line(reader: &mut impl BufRead) -> std::io::Result<()> {
    loop {
        let buf = reader.fill_buf()?;
        if buf.is_empty() {
            return Ok(());
        }
        if let Some(end) = buf.iter().position(|&b| b == b'\n') {
            reader.consume(end + 1);
            return Ok(());
        }
        let n = buf.len();
        reader.consume(n);
    }
}

/// Generate a 256-bit bearer token with the operating system CSPRNG.
pub fn generate_token() -> Result<String, AutomationError> {
    let mut bytes = [0u8; TOKEN_BYTES];
    getrandom::fill(&mut bytes).map_err(|e| AutomationError::Other(format!("cannot generate control token: {e}")))?;
    let mut token = String::with_capacity(TOKEN_HEX_LEN);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in bytes {
        token.push(HEX[(byte >> 4) as usize] as char);
        token.push(HEX[(byte & 0x0f) as usize] as char);
    }
    Ok(token)
}

/// Accept only the fixed-width hexadecimal representation emitted by [`generate_token`].
pub fn validate_token(token: &str) -> Result<(), AutomationError> {
    if token.len() != TOKEN_HEX_LEN || !token.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(AutomationError::BadRequest("control token must contain exactly 64 hexadecimal characters".into()));
    }
    Ok(())
}

/// Compare fixed-width tokens without an early exit on a mismatching byte.
pub fn token_matches(expected: &str, supplied: &str) -> bool {
    if expected.len() != TOKEN_HEX_LEN || supplied.len() != TOKEN_HEX_LEN {
        return false;
    }
    expected.bytes().zip(supplied.bytes()).fold(0u8, |different, (a, b)| different | (a ^ b)) == 0
}

/// Validate the first TCP frame without exposing any control method before authentication.
pub fn authentication_reply(line: &str, expected_token: &str) -> (Value, bool) {
    let req: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(_) => {
            return (json!({"id": null, "ok": false, "error": "authentication required"}), false);
        }
    };
    let id = req.get("id").cloned().unwrap_or(Value::Null);
    let supplied = req.get("params").and_then(|p| p.get("token")).and_then(Value::as_str).unwrap_or("");
    let ok = req.get("method").and_then(Value::as_str) == Some(AUTH_METHOD) && token_matches(expected_token, supplied);
    if ok {
        (json!({"id": id, "ok": true, "result": {"authenticated": true}}), true)
    } else {
        (json!({"id": id, "ok": false, "error": "authentication required"}), false)
    }
}

/// Answer a repeat `auth` frame on an already authenticated connection. Returns `None` for any
/// other frame, which the caller dispatches as usual.
/// Without this, a second `auth` fell through to the method dispatcher as an unknown method.
pub fn repeat_auth_reply(line: &str) -> Option<Value> {
    let req: Value = serde_json::from_str(line).ok()?;
    if req.get("method").and_then(Value::as_str) != Some(AUTH_METHOD) {
        return None;
    }
    let id = req.get("id").cloned().unwrap_or(Value::Null);
    Some(json!({"id": id, "ok": false, "error": "already authenticated"}))
}

fn read_token_file(path: &Path) -> Result<String, AutomationError> {
    let token = std::fs::read_to_string(path).map_err(|e| AutomationError::Io(format!("{}: {e}", path.display())))?;
    let token = token.trim().to_owned();
    validate_token(&token)?;
    Ok(token)
}

fn create_token_file(path: &Path, token: &str) -> Result<(), AutomationError> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| AutomationError::Io(format!("{}: {e}", parent.display())))?;
    }
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path).map_err(|e| AutomationError::Io(format!("{}: {e}", path.display())))?;
    writeln!(file, "{token}").map_err(|e| AutomationError::Io(format!("{}: {e}", path.display())))
}

/// Resolve a server token. With no supplied token or file, a fresh token is returned.
/// A missing token file is created atomically; an existing one is read and validated.
pub fn server_token(supplied: Option<&str>, token_file: Option<&Path>) -> Result<String, AutomationError> {
    if supplied.is_some() && token_file.is_some() {
        return Err(AutomationError::BadRequest("use either a control token or a control token file, not both".into()));
    }
    if let Some(token) = supplied {
        validate_token(token)?;
        return Ok(token.to_ascii_lowercase());
    }
    let token = generate_token()?;
    let Some(path) = token_file else {
        return Ok(token);
    };
    match read_token_file(path) {
        Ok(existing) => Ok(existing.to_ascii_lowercase()),
        Err(AutomationError::Io(_)) if !path.exists() => {
            match create_token_file(path, &token) {
                Ok(()) => Ok(token),
                Err(AutomationError::Io(_)) if path.exists() => {
                    // Another process won the create-new race.
                    read_token_file(path).map(|v| v.to_ascii_lowercase())
                }
                Err(e) => Err(e),
            }
        }
        Err(e) => Err(e),
    }
}

/// Resolve the token used by a client. Clients never silently generate credentials.
pub fn client_token(supplied: Option<&str>, token_file: Option<&Path>) -> Result<String, AutomationError> {
    if supplied.is_some() && token_file.is_some() {
        return Err(AutomationError::BadRequest("use either a control token or a control token file, not both".into()));
    }
    if let Some(token) = supplied {
        validate_token(token)?;
        return Ok(token.to_ascii_lowercase());
    }
    token_file.map_or_else(
        || {
            Err(AutomationError::BadRequest(
                "bridge mode needs --control-token, --control-token-file, PHOTOCRAFT_CONTROL_TOKEN, or PHOTOCRAFT_CONTROL_TOKEN_FILE".into(),
            ))
        },
        |path| read_token_file(path).map(|v| v.to_ascii_lowercase()),
    )
}

/// Counts active connections and returns a permit only while below the configured maximum.
pub struct ConnectionLimiter {
    active: AtomicUsize,
    max: usize,
}

impl ConnectionLimiter {
    pub fn new(max: usize) -> Arc<Self> {
        Arc::new(Self { active: AtomicUsize::new(0), max })
    }

    pub fn try_acquire(self: &Arc<Self>) -> Option<ConnectionPermit> {
        let mut current = self.active.load(Ordering::Acquire);
        loop {
            if current >= self.max {
                return None;
            }
            match self.active.compare_exchange_weak(current, current + 1, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return Some(ConnectionPermit { limiter: Arc::clone(self) }),
                Err(actual) => current = actual,
            }
        }
    }
}

pub struct ConnectionPermit {
    limiter: Arc<ConnectionLimiter>,
}

impl Drop for ConnectionPermit {
    fn drop(&mut self) {
        self.limiter.active.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Apply idle and write timeouts before handing a socket to a connection worker.
pub fn configure_stream(stream: &std::net::TcpStream) -> std::io::Result<()> {
    stream.set_read_timeout(Some(IO_TIMEOUT))?;
    stream.set_write_timeout(Some(IO_TIMEOUT))
}

/// Environment-aware token inputs shared by the desktop app and CLI.
pub fn token_inputs(supplied: Option<String>, token_file: Option<PathBuf>) -> (Option<String>, Option<PathBuf>) {
    let supplied = supplied.or_else(|| std::env::var("PHOTOCRAFT_CONTROL_TOKEN").ok());
    let token_file = token_file.or_else(|| std::env::var_os("PHOTOCRAFT_CONTROL_TOKEN_FILE").map(PathBuf::from));
    (supplied, token_file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_tokens_are_valid_and_distinct() {
        let a = generate_token().unwrap();
        let b = generate_token().unwrap();
        validate_token(&a).unwrap();
        assert_ne!(a, b);
        assert!(token_matches(&a, &a));
        assert!(!token_matches(&a, &b));
        assert!(!token_matches(&a, "short"));
    }

    #[test]
    fn bounded_reader_rejects_an_oversized_line() {
        let input = format!("{}\n", "x".repeat(MAX_REQUEST_BYTES + 1));
        let mut reader = std::io::Cursor::new(input);
        let mut line = String::new();
        assert_eq!(read_bounded_line(&mut reader, &mut line).unwrap(), LineRead::TooLong);
        assert_eq!(line.len(), MAX_REQUEST_BYTES + 1);
        discard_rest_of_line(&mut reader).unwrap();
        assert_eq!(read_bounded_line(&mut reader, &mut line).unwrap(), LineRead::Eof);
    }

    #[test]
    fn bounded_reader_reports_a_cut_multibyte_line_as_too_long_and_bad_utf8_as_invalid_data() {
        // The cut at the limit falls inside a two-byte character.
        let input = format!("xx{}\nnext\n", "é".repeat(MAX_REQUEST_BYTES / 2));
        let mut reader = std::io::Cursor::new(input);
        let mut line = String::new();
        assert_eq!(read_bounded_line(&mut reader, &mut line).unwrap(), LineRead::TooLong);
        discard_rest_of_line(&mut reader).unwrap();
        assert_eq!(read_bounded_line(&mut reader, &mut line).unwrap(), LineRead::Line);
        assert_eq!(line, "next\n");
        // Invalid UTF-8 consumes its whole line, so the next read starts on the next line.
        let mut reader = std::io::Cursor::new(b"\xff\xfe\nnext\n".to_vec());
        assert_eq!(read_bounded_line(&mut reader, &mut line).unwrap_err().kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(read_bounded_line(&mut reader, &mut line).unwrap(), LineRead::Line);
        assert_eq!(line, "next\n");
    }

    #[test]
    fn authentication_does_not_dispatch_without_the_token() {
        let token = generate_token().unwrap();
        let (reply, authenticated) = authentication_reply(r#"{"id":1,"method":"methods","params":{}}"#, &token);
        assert!(!authenticated);
        assert_eq!(reply["error"], "authentication required");
        assert!(reply.get("result").is_none());

        let line = json!({"id": 2, "method": AUTH_METHOD, "params": {"token": token}}).to_string();
        let (reply, authenticated) = authentication_reply(&line, &token);
        assert!(authenticated);
        assert_eq!(reply["result"]["authenticated"], true);
    }

    #[test]
    fn repeat_auth_on_an_authenticated_connection_says_so() {
        let line = json!({"id": 3, "method": AUTH_METHOD, "params": {"token": "anything"}}).to_string();
        let reply = repeat_auth_reply(&line).unwrap();
        assert_eq!(reply["id"], 3);
        assert_eq!(reply["ok"], false);
        assert_eq!(reply["error"], "already authenticated");
        assert!(repeat_auth_reply(r#"{"id":4,"method":"methods"}"#).is_none());
        assert!(repeat_auth_reply("{not json").is_none());
    }

    #[test]
    fn token_file_round_trips_between_server_and_client() {
        let path = std::env::temp_dir().join(format!("photocraft-control-token-{}-{}.txt", std::process::id(), generate_token().unwrap()));
        let server = server_token(None, Some(&path)).unwrap();
        let client = client_token(None, Some(&path)).unwrap();
        assert_eq!(server, client);
        assert!(token_matches(&server, &client));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn connection_limiter_releases_capacity() {
        let limiter = ConnectionLimiter::new(1);
        let permit = limiter.try_acquire().unwrap();
        assert!(limiter.try_acquire().is_none());
        drop(permit);
        assert!(limiter.try_acquire().is_some());
    }
}
