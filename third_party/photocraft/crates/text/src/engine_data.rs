//! Photoshop's `EngineData`: the PostScript-like text engine document stored in `TySh`.
//!
//! Syntax (from observation of real files and public notes): dictionaries `<< /Key value … >>`,
//! arrays `[ … ]`, strings `( … )` holding UTF-16BE with a `FE FF` byte-order mark and `\`
//! escapes, numbers (`12`, `-1.5`, `.5`), `true`/`false`, and `/Name` values.

use std::fmt::Write as _;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Dict(Vec<(String, Value)>),
    Array(Vec<Value>),
    String(String),
    Int(i64),
    Real(f64),
    Bool(bool),
    Name(String),
}

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Dict(items) => items.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    /// Follows a path of dictionary keys.
    pub fn path(&self, keys: &[&str]) -> Option<&Value> {
        keys.iter().try_fold(self, |v, k| v.get(k))
    }
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        match self {
            Value::Dict(items) => items.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    /// Sets (or appends) a dictionary entry.
    pub fn set(&mut self, key: &str, v: Value) {
        if let Value::Dict(items) = self {
            match items.iter_mut().find(|(k, _)| k == key) {
                Some(e) => e.1 = v,
                None => items.push((key.to_string(), v)),
            }
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::Real(r) => Some(*r),
            _ => None,
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            Value::Real(r) => Some(*r as i64),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            Value::Int(i) => Some(*i != 0),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) | Value::Name(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
    pub fn dict() -> Value {
        Value::Dict(Vec::new())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError(pub String);

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EngineData: {}", self.0)
    }
}
impl std::error::Error for ParseError {}

const MAX_DEPTH: usize = 64;

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\r' | b'\n' | 0) {
            self.i += 1;
        }
    }
    fn err<T>(&self, m: &str) -> Result<T, ParseError> {
        Err(ParseError(format!("{m} at byte {}", self.i)))
    }
    fn value(&mut self, depth: usize) -> Result<Value, ParseError> {
        if depth > MAX_DEPTH {
            return self.err("nesting too deep");
        }
        self.ws();
        let b = self.b;
        match b.get(self.i) {
            None => self.err("unexpected end"),
            Some(b'<') if b.get(self.i + 1) == Some(&b'<') => {
                self.i += 2;
                let mut items = Vec::new();
                loop {
                    self.ws();
                    match b.get(self.i) {
                        Some(b'>') if b.get(self.i + 1) == Some(&b'>') => {
                            self.i += 2;
                            return Ok(Value::Dict(items));
                        }
                        Some(b'/') => {
                            let k = self.name();
                            let v = self.value(depth + 1)?;
                            items.push((k, v));
                        }
                        None => return self.err("unterminated dictionary"),
                        _ => return self.err("expected key"),
                    }
                }
            }
            Some(b'[') => {
                self.i += 1;
                let mut items = Vec::new();
                loop {
                    self.ws();
                    match b.get(self.i) {
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Value::Array(items));
                        }
                        None => return self.err("unterminated array"),
                        _ => items.push(self.value(depth + 1)?),
                    }
                }
            }
            Some(b'(') => {
                self.i += 1;
                let mut raw = Vec::new();
                loop {
                    match b.get(self.i) {
                        None => return self.err("unterminated string"),
                        Some(b'\\') => {
                            if let Some(&c) = b.get(self.i + 1) {
                                raw.push(c);
                            }
                            self.i += 2;
                        }
                        Some(b')') => {
                            self.i += 1;
                            break;
                        }
                        Some(&c) => {
                            raw.push(c);
                            self.i += 1;
                        }
                    }
                }
                Ok(Value::String(decode_string(&raw)))
            }
            Some(b'/') => Ok(Value::Name(self.name())),
            Some(_) => {
                let start = self.i;
                while self.i < b.len() && !matches!(b[self.i], b' ' | b'\t' | b'\r' | b'\n' | b'[' | b']' | b'<' | b'>' | b'(' | b'/') {
                    self.i += 1;
                }
                let tok = std::str::from_utf8(&b[start..self.i]).unwrap_or("");
                match tok {
                    "true" => Ok(Value::Bool(true)),
                    "false" => Ok(Value::Bool(false)),
                    "" => {
                        self.i += 1;
                        self.err("unexpected byte")
                    }
                    t if t.contains('.') || t.contains('e') || t.contains('E') => t.parse().map(Value::Real).or_else(|_| self.err("bad number")),
                    t => t.parse().map(Value::Int).or_else(|_| t.parse().map(Value::Real).or_else(|_| self.err("bad token"))),
                }
            }
        }
    }
    fn name(&mut self) -> String {
        self.i += 1; // '/'
        let start = self.i;
        while self.i < self.b.len() && !matches!(self.b[self.i], b' ' | b'\t' | b'\r' | b'\n' | b'[' | b']' | b'<' | b'>' | b'(' | b'/') {
            self.i += 1;
        }
        String::from_utf8_lossy(&self.b[start..self.i]).into_owned()
    }
}

fn decode_string(raw: &[u8]) -> String {
    if raw.len() >= 2 && raw[0] == 0xFE && raw[1] == 0xFF {
        let u: Vec<u16> = raw[2..].as_chunks::<2>().0.iter().map(|c| u16::from_be_bytes([c[0], c[1]])).collect();
        String::from_utf16_lossy(&u)
    } else {
        raw.iter().map(|&c| c as char).collect()
    }
}

/// Parses an EngineData document (the top-level dictionary).
pub fn parse(data: &[u8]) -> Result<Value, ParseError> {
    let mut p = Parser { b: data, i: 0 };
    p.value(0)
}

/// Serializes in Photoshop's layout (tab-indented, one key per line).
pub fn write(v: &Value) -> Vec<u8> {
    let mut out = b"\n\n".to_vec();
    write_value(v, 0, &mut out);
    out.push(b'\n');
    out
}

/// Serializes `/key value` pairs without an enclosing dictionary: the layout of the document's
/// `Txt2` block (see [`crate::psd::build_txt2`]), which `parse_txt2` wraps in `<< >>` to parse.
pub fn write_bare(items: &[(String, Value)]) -> Vec<u8> {
    let mut out = b"\n\n".to_vec();
    for (k, v) in items {
        out.push(b'/');
        out.extend_from_slice(k.as_bytes());
        if is_container(v) {
            out.push(b'\n');
            indent(&mut out, 1);
        } else {
            out.push(b' ');
        }
        write_value(v, 1, &mut out);
        out.push(b'\n');
    }
    out
}

fn indent(out: &mut Vec<u8>, n: usize) {
    out.extend(std::iter::repeat_n(b'\t', n));
}

fn is_container(v: &Value) -> bool {
    matches!(v, Value::Dict(_)) || matches!(v, Value::Array(a) if a.iter().any(|x| matches!(x, Value::Dict(_) | Value::Array(_))))
}

fn write_value(v: &Value, depth: usize, out: &mut Vec<u8>) {
    match v {
        Value::Dict(items) => {
            out.extend_from_slice(b"<<\n");
            for (k, v) in items {
                indent(out, depth + 1);
                out.push(b'/');
                out.extend_from_slice(k.as_bytes());
                if is_container(v) {
                    out.push(b'\n');
                    indent(out, depth + 1);
                } else {
                    out.push(b' ');
                }
                write_value(v, depth + 1, out);
                out.push(b'\n');
            }
            indent(out, depth);
            out.extend_from_slice(b">>");
        }
        Value::Array(items) if is_container(v) => {
            out.extend_from_slice(b"[\n");
            for x in items {
                indent(out, depth);
                write_value(x, depth, out);
                out.push(b'\n');
            }
            indent(out, depth);
            out.push(b']');
        }
        Value::Array(items) => {
            out.push(b'[');
            for x in items {
                out.push(b' ');
                write_value(x, depth, out);
            }
            out.extend_from_slice(b" ]");
        }
        Value::String(s) => {
            out.push(b'(');
            let mut raw = vec![0xFE, 0xFF];
            for u in s.encode_utf16() {
                raw.extend_from_slice(&u.to_be_bytes());
            }
            for c in raw {
                if matches!(c, b'(' | b')' | b'\\') {
                    out.push(b'\\');
                }
                out.push(c);
            }
            out.push(b')');
        }
        Value::Int(i) => {
            let _ = write!(Wrap(out), "{i}");
        }
        Value::Real(r) => out.extend_from_slice(format_real(*r).as_bytes()),
        Value::Bool(b) => out.extend_from_slice(if *b { b"true" } else { b"false" }),
        Value::Name(n) => {
            out.push(b'/');
            out.extend_from_slice(n.as_bytes());
        }
    }
}

struct Wrap<'a>(&'a mut Vec<u8>);
impl std::fmt::Write for Wrap<'_> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0.extend_from_slice(s.as_bytes());
        Ok(())
    }
}

/// Photoshop style: `1.0`, `.5`, `-.25`.
fn format_real(r: f64) -> String {
    if !r.is_finite() {
        return "0.0".into();
    }
    if r.fract() == 0.0 && r.abs() < 1e15 {
        return format!("{r:.1}");
    }
    let mut s = format!("{r:.5}");
    while s.ends_with('0') {
        s.pop();
    }
    if matches!(s.as_str(), "0." | "-0.") {
        return "0.0".into();
    }
    if let Some(rest) = s.strip_prefix("0.") {
        s = format!(".{rest}");
    } else if let Some(rest) = s.strip_prefix("-0.") {
        s = format!("-.{rest}");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &[u8] = b"\n\n<<\n\t/EngineDict\n\t<<\n\t\t/Editor\n\t\t<<\n\t\t\t/Text (\xFE\xFF\x00H\x00i\x00\\)\x00\r)\n\t\t>>\n\t\t/Nums [ 1.0 .5 -2 0 ]\n\t\t/Flag true\n\t\t/Kind /Roman\n\t\t/Runs [\n\t\t<<\n\t\t\t/A 1\n\t\t>>\n\t\t]\n\t>>\n>>";

    #[test]
    fn parses_sample() {
        let v = parse(SAMPLE).unwrap();
        assert_eq!(v.path(&["EngineDict", "Editor", "Text"]).and_then(Value::as_str), Some("Hi)\r"));
        let nums: Vec<f64> = v.path(&["EngineDict", "Nums"]).unwrap().as_array().unwrap().iter().filter_map(Value::as_f64).collect();
        assert_eq!(nums, [1.0, 0.5, -2.0, 0.0]);
        assert_eq!(v.path(&["EngineDict", "Flag"]).and_then(Value::as_bool), Some(true));
        assert_eq!(v.path(&["EngineDict", "Kind"]), Some(&Value::Name("Roman".into())));
        assert_eq!(v.path(&["EngineDict", "Runs"]).unwrap().as_array().unwrap()[0].get("A").and_then(Value::as_i64), Some(1));
    }

    #[test]
    fn write_parse_roundtrip() {
        let v = parse(SAMPLE).unwrap();
        let bytes = write(&v);
        assert_eq!(parse(&bytes).unwrap(), v);
        // Stable: writing twice gives identical bytes.
        assert_eq!(write(&parse(&bytes).unwrap()), bytes);
    }

    #[test]
    fn reals_format_like_photoshop() {
        assert_eq!(format_real(1.0), "1.0");
        assert_eq!(format_real(0.5), ".5");
        assert_eq!(format_real(-0.25), "-.25");
        assert_eq!(format_real(12.125), "12.125");
    }

    #[test]
    fn tiny_reals_round_trip_as_numbers() {
        for value in [0.000004, -0.000004] {
            assert_eq!(format_real(value), "0.0");
            let source = Value::Dict(vec![(
                "EngineDict".into(),
                Value::Dict(vec![("StyleRun".into(), Value::Array(vec![Value::Dict(vec![("BaselineShift".into(), Value::Real(value))])]))]),
            )]);
            let encoded = write(&source);
            let parsed = parse(&encoded).expect("formatted tiny real should parse");
            let baseline_shift = parsed
                .path(&["EngineDict", "StyleRun"])
                .and_then(Value::as_array)
                .and_then(|runs| runs.first())
                .and_then(|run| run.get("BaselineShift"))
                .and_then(Value::as_f64);
            assert_eq!(baseline_shift, Some(0.0));
        }
    }

    #[test]
    fn malformed_input_never_panics() {
        for cut in 0..SAMPLE.len() {
            let _ = parse(&SAMPLE[..cut]);
        }
        assert!(parse(b"<< /A").is_err());
        assert!(parse(&[b'['; 200]).is_err());
    }
}
