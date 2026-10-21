//! Client for the desktop app's JSON-lines control protocol
//! (`docs/control-protocol.md`): one JSON request per line, replies matched
//! by `id`. Keeps one connection, reconnecting after a failure — or before
//! a call, once the connection has been idle long enough that the app
//! (which closes idle control connections after
//! [`IO_TIMEOUT`](crate::security::IO_TIMEOUT)) may already have closed it.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::sync::Mutex;

use crate::AutomationError;
use crate::budgets::MAX_RESPONSE_BYTES;
use crate::security::{AUTH_METHOD, IO_TIMEOUT, MAX_REQUEST_BYTES, validate_token};

type Conn = (BufReader<tokio::net::tcp::OwnedReadHalf>, tokio::net::tcp::OwnedWriteHalf);

/// Reconnect before sending when the cached connection has been idle this long: the app
/// drops idle control connections after [`IO_TIMEOUT`], and reconnecting before the send
/// keeps the request out of the "was it applied?" state. The margin covers a request
/// still in flight when the idle clock runs out.
const IDLE_RECONNECT: Duration = IO_TIMEOUT.saturating_sub(Duration::from_secs(5));

/// The default call timeout sits strictly below the app's 60s reply deadline, so this
/// client's error — not the app's deadline reply, not the caller's own cutoff — is the
/// one the caller sees.
pub const DEFAULT_CALL_TIMEOUT: Duration = Duration::from_secs(55);

const NO_REPLY_CAUSES: &str =
    "the command may still be running, or the app was not drawing frames (display asleep, window fully occluded, or a long operation blocked the frame loop)";

pub struct BridgeClient {
    addr: String,
    token: String,
    conn: Mutex<Option<(Conn, Instant)>>,
    next_id: AtomicU64,
    timeout: Duration,
    idle_reconnect: Duration,
}

impl BridgeClient {
    /// `addr` such as `127.0.0.1:7878`. Only loopback addresses are accepted,
    /// matching the server, which binds to loopback only.
    pub fn new(addr: impl Into<String>, token: impl Into<String>) -> Result<Self, AutomationError> {
        let addr = addr.into();
        let token = token.into();
        let host = addr.rsplit_once(':').map(|(h, _)| h).unwrap_or(&addr);
        if !matches!(host, "127.0.0.1" | "localhost" | "[::1]" | "::1") {
            return Err(AutomationError::BadRequest(format!("bridge address must be loopback, got `{addr}`")));
        }
        validate_token(&token)?;
        Ok(BridgeClient {
            addr,
            token: token.to_ascii_lowercase(),
            conn: Mutex::new(None),
            next_id: AtomicU64::new(1),
            timeout: DEFAULT_CALL_TIMEOUT,
            idle_reconnect: IDLE_RECONNECT,
        })
    }

    pub fn with_timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }

    /// Shrink the idle-reconnect threshold; tests use it to run the reconnect path quickly.
    pub fn with_idle_reconnect(mut self, t: Duration) -> Self {
        self.idle_reconnect = t;
        self
    }

    pub fn addr(&self) -> &str {
        &self.addr
    }

    /// Call a control method; returns its `result` or the app's error. Never resends after
    /// a transport failure: the request may have been applied (#1007).
    pub async fn call(&self, method: &str, params: Value) -> Result<Value, AutomationError> {
        self.dispatch(method, &params, false).await
    }

    /// [`Self::call`] for a method the caller knows only reads state. A transport failure
    /// applied nothing, so one fresh attempt is made on a new connection.
    pub async fn call_read(&self, method: &str, params: Value) -> Result<Value, AutomationError> {
        self.dispatch(method, &params, true).await
    }

    async fn dispatch(&self, method: &str, params: &Value, read: bool) -> Result<Value, AutomationError> {
        match self.attempt(method, params).await {
            Ok(v) => Ok(v),
            // A dropped socket or an unsent request applied nothing: safe to retry once,
            // and only for a call the caller vouched is a read.
            Err(Failure::NotSent(_)) | Err(Failure::Transport(_)) if read => match self.attempt(method, params).await {
                Ok(v) => Ok(v),
                Err(retry) => Err(self.failure(method, retry, read)),
            },
            Err(f) => Err(self.failure(method, f, read)),
        }
    }

    /// The message for a failed attempt, worded by what the failure proves.
    fn failure(&self, method: &str, f: Failure, read: bool) -> AutomationError {
        match f {
            Failure::NotSent(e) => AutomationError::Bridge(format!("`{method}` failed: {e}; the request was not sent")),
            Failure::Transport(e) => AutomationError::Bridge(format!("`{method}` failed: {e}; operation may have completed; inspect state before retrying")),
            Failure::TimedOut(_) => {
                let what = if read { "a read may be retried safely" } else { "operation may have completed; inspect state before retrying" };
                AutomationError::Bridge(format!("`{method}` timed out after {:?}; {NO_REPLY_CAUSES}; {what}", self.timeout))
            }
            Failure::Definite(e) => e,
        }
    }

    /// One exchange over one connection, reconnecting first when the cached one may be dead.
    async fn attempt(&self, method: &str, params: &Value) -> Result<Value, Failure> {
        let mut guard = self.conn.lock().await;
        if let Some((_, last_used)) = guard.as_ref()
            && last_used.elapsed() >= self.idle_reconnect
        {
            // The request below goes to a fresh socket, not one the app may have dropped.
            *guard = None;
        }
        if guard.is_none() {
            let s = tokio::time::timeout(Duration::from_secs(5), TcpStream::connect(&self.addr))
                .await
                .map_err(|_| Failure::NotSent(format!("timed out connecting to {}", self.addr)))?
                .map_err(|e| {
                    Failure::NotSent(format!(
                        "cannot connect to {} ({e}); start the app with `photocraft --control <port>` and matching control credentials",
                        self.addr
                    ))
                })?;
            let (r, w) = s.into_split();
            let mut conn = (BufReader::new(r), w);
            let auth_id = self.next_id.fetch_add(1, Ordering::Relaxed);
            let auth = tokio::time::timeout(Duration::from_secs(5), exchange(&mut conn, auth_id, AUTH_METHOD, &json!({"token": self.token})))
                .await
                .map_err(|_| Failure::NotSent("control authentication timed out".into()))?
                .map_err(|e| Failure::NotSent(e.to_string()))?;
            if let Err(e) = auth {
                return Err(Failure::Definite(e));
            }
            *guard = Some((conn, Instant::now()));
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let Some((conn, _)) = guard.as_mut() else {
            return Err(Failure::NotSent(format!("not connected to {}", self.addr)));
        };
        let outcome = tokio::time::timeout(self.timeout, exchange(conn, id, method, params)).await;
        match outcome {
            Ok(Ok(Err(error @ AutomationError::BadRequest(_)))) => {
                // An oversized frame leaves unread bytes. Drop this connection and
                // report the budget failure without retrying a possibly completed edit.
                *guard = None;
                Err(Failure::Definite(error))
            }
            Ok(Ok(Ok(v))) => {
                if let Some((_, last_used)) = guard.as_mut() {
                    *last_used = Instant::now();
                }
                Ok(v)
            }
            Ok(Ok(Err(e))) => {
                // The app answered: the operation did not apply — except the app's own
                // deadline reply, which fires while the command may still be running.
                if let Some((_, last_used)) = guard.as_mut() {
                    *last_used = Instant::now();
                }
                Err(Failure::Definite(e))
            }
            // A failed write or lost reply does not prove the edit was not applied.
            // Reconnect on the next call; only a vouched read may resend (#1007).
            Ok(Err(e)) => {
                *guard = None;
                Err(Failure::Transport(e.to_string()))
            }
            Err(_) => {
                *guard = None;
                Err(Failure::TimedOut(()))
            }
        }
    }
}

/// How a single attempt failed, deciding the retry and the wording above.
enum Failure {
    /// The connection could not be established or authenticated: nothing was sent.
    NotSent(String),
    /// The request was sent and the reply was lost: the outcome is uncertain.
    Transport(String),
    /// The call's own timeout elapsed. Retrying would only double the wait.
    TimedOut(()),
    /// The app answered (ok or error) or the request was refused before it was sent.
    Definite(AutomationError),
}

/// Outer `Err` = transport failure with an uncertain outcome; inner = app-level result.
async fn exchange(conn: &mut Conn, id: u64, method: &str, params: &Value) -> Result<Result<Value, AutomationError>, AutomationError> {
    let mut line = serde_json::to_string(&json!({"id": id, "method": method, "params": params})).map_err(|e| AutomationError::Other(e.to_string()))?;
    line.push('\n');
    if line.len() > MAX_REQUEST_BYTES {
        return Ok(Err(AutomationError::BadRequest(format!("request is {} bytes; maximum is {MAX_REQUEST_BYTES}", line.len()))));
    }
    let io = |e: std::io::Error| AutomationError::Bridge(e.to_string());
    conn.1.write_all(line.as_bytes()).await.map_err(io)?;
    conn.1.flush().await.map_err(io)?;
    let mut buf = Vec::new();
    loop {
        buf.clear();
        // Read raw bytes so truncation inside a UTF-8 character still reports the
        // budget error instead of a decoding/transport error.
        let n = (&mut conn.0).take((MAX_RESPONSE_BYTES + 1) as u64).read_until(b'\n', &mut buf).await.map_err(io)?;
        if n > MAX_RESPONSE_BYTES {
            return Ok(Err(AutomationError::BadRequest(format!("bridge response exceeds {MAX_RESPONSE_BYTES} bytes; operation may have completed"))));
        }
        if n == 0 {
            return Err(AutomationError::Bridge("connection closed by the app".into()));
        }
        let Ok(v) = serde_json::from_slice::<Value>(&buf) else {
            continue;
        };
        if v.get("id").and_then(Value::as_u64) != Some(id) {
            continue; // stale reply from an earlier, timed-out request
        }
        return Ok(if v.get("ok").and_then(Value::as_bool) == Some(true) {
            Ok(v.get("result").cloned().unwrap_or(Value::Null))
        } else {
            Err(AutomationError::App(v.get("error").and_then(Value::as_str).unwrap_or("unknown error").to_owned()))
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_call_timeout_stays_below_the_app_reply_deadline() {
        // At or above the app's 60s deadline, callers would see their own cutoff first.
        assert!(DEFAULT_CALL_TIMEOUT < Duration::from_secs(60));
    }

    #[test]
    fn idle_reconnect_stays_below_the_app_idle_cutoff() {
        // Reconnecting later than IO_TIMEOUT would send into a socket the app dropped.
        assert!(IDLE_RECONNECT < IO_TIMEOUT);
    }
}
