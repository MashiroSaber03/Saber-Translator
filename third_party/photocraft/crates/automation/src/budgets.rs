//! Fixed ceilings for automation previews and outbound JSON. These bound individual
//! outputs, not total document memory or the duration of an engine command.

use std::io::{self, Write};

use serde::Serialize;
use serde_json::{Value, json};

use crate::AutomationError;
use crate::security::MAX_REQUEST_BYTES;

/// Maximum encoded JSON reply, including its newline for JSON-lines transports.
pub const MAX_RESPONSE_BYTES: usize = 8 << 20;
/// Maximum PNG bytes before base64 expansion (leaves room for JSON/MCP metadata).
pub const MAX_PNG_BYTES: usize = 5 << 20;
/// Maximum requested preview edge. Zero still requests full size, within this ceiling.
pub const MAX_PREVIEW_SIDE: u32 = 2048;
/// Maximum source pixels processed by a headless automation preview.
pub const MAX_RENDER_SOURCE_PIXELS: u64 = 64 << 20;

fn bad(message: impl Into<String>) -> AutomationError {
    AutomationError::BadRequest(message.into())
}

/// Check before the compositor runs. Oversized requests fail rather than silently downscale.
pub fn check_preview(width: u32, height: u32, max_side: u32) -> Result<(), AutomationError> {
    if width == 0 || height == 0 {
        return Err(bad("automation preview requires nonzero document dimensions"));
    }
    if u64::from(width) * u64::from(height) > MAX_RENDER_SOURCE_PIXELS {
        return Err(bad(format!("automation preview source exceeds {MAX_RENDER_SOURCE_PIXELS} pixels")));
    }
    let side = if max_side == 0 { width.max(height) } else { max_side };
    if side > MAX_PREVIEW_SIDE {
        return Err(bad(format!("automation preview side exceeds {MAX_PREVIEW_SIDE} pixels")));
    }
    Ok(())
}

/// Check before base64 encoding or writing a rendered PNG.
pub fn check_png(bytes: usize) -> Result<(), AutomationError> {
    if bytes > MAX_PNG_BYTES {
        return Err(bad(format!("automation PNG exceeds {MAX_PNG_BYTES} bytes")));
    }
    Ok(())
}

struct LimitedWriter {
    bytes: Vec<u8>,
    maximum: usize,
}

impl Write for LimitedWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if buf.len() > self.maximum.saturating_sub(self.bytes.len()) {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("response exceeds {} bytes", self.maximum)));
        }
        self.bytes.try_reserve(buf.len()).map_err(|error| io::Error::other(format!("response allocation failed: {error}")))?;
        self.bytes.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Serialize into a bounded buffer. No partial response is sent to the transport.
pub fn json_bytes(value: &impl Serialize) -> Result<Vec<u8>, AutomationError> {
    encode_with_limit(value, MAX_RESPONSE_BYTES - 1)
}

fn encode_with_limit(value: &impl Serialize, maximum: usize) -> Result<Vec<u8>, AutomationError> {
    let mut writer = LimitedWriter { bytes: Vec::new(), maximum };
    serde_json::to_writer(&mut writer, value).map_err(|error| bad(format!("response encoding failed: {error}")))?;
    Ok(writer.bytes)
}

/// Encode the entire envelope before writing. A rejected reply is replaced by a small
/// error with the original ID; the operation may already have completed.
pub fn write_reply(out: &mut impl Write, reply: &Value) -> io::Result<()> {
    let encoded = match json_bytes(reply) {
        Ok(bytes) => bytes,
        Err(_) => {
            let error = json!({
                "id": reply.get("id").cloned().unwrap_or(Value::Null),
                "ok": false,
                "error": format!("response exceeds {MAX_RESPONSE_BYTES} bytes; operation may have completed"),
            });
            // Request IDs arriving over the wire fit in MAX_REQUEST_BYTES. Bound even
            // callers that construct a reply directly instead of using a transport.
            match json_bytes(&error) {
                Ok(bytes) => bytes,
                Err(_) => b"{\"id\":null,\"ok\":false,\"error\":\"response budget exceeded\"}".to_vec(),
            }
        }
    };
    out.write_all(&encoded)?;
    out.write_all(b"\n")
}

/// Reserve envelope/ID space and bound retained batch results incrementally, so
/// many individually valid replies cannot accumulate into an enormous batch.
pub struct BatchReplyBudget {
    remaining: usize,
    /// Charge each result as it costs inside a JSON string (MCP text content).
    escaped: bool,
}

impl Default for BatchReplyBudget {
    fn default() -> Self {
        Self { remaining: MAX_RESPONSE_BYTES - MAX_REQUEST_BYTES - 4096, escaped: false }
    }
}

impl BatchReplyBudget {
    /// For a batch reply sent as MCP text content: the encoded reply is embedded in a JSON
    /// string, which escapes every quote and backslash again, so a batch the plain budget accepts
    /// could exceed the tool-result ceiling and lose every step result.
    pub fn escaped() -> Self {
        Self { escaped: true, ..Self::default() }
    }

    /// Must be called before retaining the result or dispatching the next step.
    pub fn charge(&mut self, result: &Value) -> Result<(), AutomationError> {
        let limit = self.remaining.saturating_sub(1);
        let bytes = encode_with_limit(result, limit)?;
        // Compact JSON has no raw control characters, so only `"` and `\` grow when escaped.
        let size = if self.escaped { bytes.len() + bytes.iter().filter(|&&b| b == b'"' || b == b'\\').count() } else { bytes.len() };
        if size > limit {
            return Err(bad(format!("response exceeds {limit} bytes once escaped")));
        }
        self.remaining = self.remaining.saturating_sub(size + 1);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_checks_full_size_requested_edge_and_source_before_rendering() {
        assert!(check_preview(64, 32, 0).is_ok());
        assert!(check_preview(MAX_PREVIEW_SIDE, MAX_PREVIEW_SIDE, 0).is_ok());
        assert!(check_preview(MAX_PREVIEW_SIDE + 1, 1, 0).is_err());
        assert!(check_preview(64, 32, MAX_PREVIEW_SIDE + 1).is_err());
        assert!(check_preview(8192, 8192, 1024).is_ok());
        assert!(check_preview(8193, 8192, 1024).is_err());
        assert!(check_preview(u32::MAX, u32::MAX, 1).is_err());
        assert!(check_preview(0, 32, 1).is_err());
    }

    #[test]
    fn encoded_bytes_include_json_escaping_and_accept_exact_boundary() {
        assert_eq!(encode_with_limit(&json!("abc"), 5).unwrap(), b"\"abc\"");
        assert!(encode_with_limit(&json!("abc"), 4).is_err());
        assert!(encode_with_limit(&json!("\n\n"), 5).is_err());
        assert!(encode_with_limit(&Value::Null, 0).is_err());
        assert!(check_png(MAX_PNG_BYTES).is_ok());
        assert!(check_png(MAX_PNG_BYTES + 1).is_err());
    }

    #[test]
    fn oversized_reply_is_one_complete_error_with_matching_id() {
        let reply = json!({"id": 7, "ok": true, "result": "x".repeat(MAX_RESPONSE_BYTES)});
        let mut out = Vec::new();
        write_reply(&mut out, &reply).unwrap();
        assert!(out.len() < 1024);
        assert_eq!(out.iter().filter(|&&byte| byte == b'\n').count(), 1);
        let error: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(error["id"], 7);
        assert_eq!(error["ok"], false);
        assert!(error["error"].as_str().unwrap().contains("operation may have completed"));
    }

    #[test]
    fn batch_cannot_retain_more_than_the_aggregate_reply_budget() {
        let mut budget = BatchReplyBudget { remaining: 10, escaped: false };
        budget.charge(&json!("abc")).unwrap();
        assert!(budget.charge(&json!("abc")).is_err());
        assert_eq!(budget.remaining, 4);
    }

    #[test]
    fn escaped_budget_charges_the_size_inside_a_json_string() {
        // `"a\"b"` is 6 bytes plain and 10 once embedded in a string (`\"a\\\"b\"`).
        let value = json!("a\"b");
        let escaped_len = serde_json::to_string(&Value::String(value.to_string())).unwrap().len() - 2;
        assert_eq!(escaped_len, 10);
        let mut budget = BatchReplyBudget { remaining: 11, escaped: true };
        budget.charge(&value).unwrap();
        assert_eq!(budget.remaining, 0);
        let mut budget = BatchReplyBudget { remaining: 10, escaped: true };
        assert!(budget.charge(&value).is_err());
        assert!(BatchReplyBudget { remaining: 10, escaped: false }.charge(&value).is_ok());
    }
}
