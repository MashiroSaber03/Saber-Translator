//! Photoshop Actions files (`.atn`): one action set, its actions and their steps.
//!
//! Implemented clean-room from Adobe's "Photoshop File Formats" specification (the *Actions*
//! and *Descriptor structure* sections) and observation of files Photoshop writes. No code was
//! copied. All numbers are big-endian; strings are the Unicode strings of descriptors (u32
//! code-unit count + UTF-16BE, usually NUL-terminated) unless noted.
//!
//! ```text
//! file   := version u32 (16), set name UnicodeString, expanded u8, count u32, count × action
//! action := function key u16 (0 = none, n = Fn), shift u8, command u8, colour u16,
//!           name UnicodeString, expanded u8, count u32, count × item
//! item   := expanded u8, enabled u8, with dialog u8, dialog options u8,
//!           event kind [4]: "TEXT" (u32 length + ASCII event name) | "long" (4-char event code),
//!           dictionary name (u32 length + ASCII: the name the Actions panel shows),
//!           has descriptor i32 (-1 = a descriptor follows, 0 = none), [descriptor]
//! ```
//!
//! The descriptor is a plain ActionDescriptor body ([`Descriptor`], no version prefix): the
//! event's parameters, exactly as Photoshop's own "Make", "Set", "Gaussian Blur"… events
//! carry them. Photoshop 5-era files (version 12) use a different, undocumented layout and are
//! rejected as unsupported.
//!
//! [`parse`] never panics: every count is bounds-checked against the bytes left, descriptor
//! nesting is bounded, and allocations are capped. [`write`] produces the same layout, so
//! `parse(write(f)) == f`, and `write(parse(b)) == b` for files Photoshop wrote.

use crate::descriptor::{Descriptor, UnicodeString};
use crate::error::{PsdError, Result};
use crate::io::{Reader, WriteExt};

/// The only version this module reads and writes (Photoshop 6 and every later release).
pub const VERSION: u32 = 16;
/// Most actions read from one set (Photoshop's own sets hold a few dozen).
pub const MAX_ACTIONS: usize = 10_000;
/// Most steps read from one action.
pub const MAX_ITEMS: usize = 100_000;

/// A parsed action set.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AtnFile {
    /// The set's name (the Actions panel folder).
    pub name: UnicodeString,
    /// Whether the set is shown expanded in the Actions panel.
    pub expanded: bool,
    /// The actions, in panel order.
    pub actions: Vec<AtnAction>,
    /// Problems that did not stop the parse (trailing bytes…).
    #[cfg_attr(feature = "serde", serde(default))]
    pub warnings: Vec<String>,
}

/// One action of a set.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AtnAction {
    /// Function key that plays the action: 0 = none, `n` = F`n`.
    pub function_key: u16,
    /// Shift modifier of the function key.
    pub shift: bool,
    /// Command (Control on Windows) modifier of the function key.
    pub command: bool,
    /// Button Mode colour index (0 = none, then red, orange, yellow, green, blue, violet, gray).
    pub color: u16,
    /// The action's name.
    pub name: UnicodeString,
    /// Whether the action is shown expanded in the Actions panel.
    pub expanded: bool,
    /// The steps, in playback order.
    pub items: Vec<AtnItem>,
}

/// One step of an action: a Photoshop event with its parameters.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AtnItem {
    /// Whether the step's parameters are shown expanded in the Actions panel.
    pub expanded: bool,
    /// The panel's check mark: a switched-off step is skipped by playback.
    pub enabled: bool,
    /// The panel's dialog toggle: playback shows the step's dialog.
    pub with_dialog: bool,
    /// Dialog options (0 = none, 2 = show the dialog, 3 = don't show), kept verbatim.
    pub dialog_options: u8,
    /// The event.
    pub event: AtnEvent,
    /// The name the Actions panel shows for the step ("Gaussian Blur", "Make", …).
    pub dictionary: String,
    /// The event's parameters; `None` for events without any.
    pub descriptor: Option<Descriptor>,
}

impl Default for AtnItem {
    fn default() -> Self {
        AtnItem {
            expanded: false,
            enabled: true,
            with_dialog: false,
            dialog_options: 0,
            event: AtnEvent::Code(*b"    "),
            dictionary: String::new(),
            descriptor: None,
        }
    }
}

/// A Photoshop event id: a four-character code (`GsnB`) or a string name (`trim`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AtnEvent {
    /// Stored as `long` + 4 bytes.
    Code([u8; 4]),
    /// Stored as `TEXT` + length-prefixed ASCII.
    Name(String),
}

impl AtnEvent {
    /// The event id as text: the code's four characters, or the name.
    pub fn id(&self) -> String {
        match self {
            AtnEvent::Code(c) => String::from_utf8_lossy(c).into_owned(),
            AtnEvent::Name(n) => n.clone(),
        }
    }
    /// `true` if the event is the code or name `s`.
    pub fn is(&self, s: &str) -> bool {
        match self {
            AtnEvent::Code(c) => c == s.as_bytes(),
            AtnEvent::Name(n) => n == s,
        }
    }
}

/// Reads a u32 length + ASCII bytes (event and dictionary names).
fn read_ascii(r: &mut Reader<'_>) -> Result<String> {
    let n = r.u32()?;
    Ok(String::from_utf8_lossy(r.bytes_u64(u64::from(n))?).into_owned())
}

fn write_ascii(out: &mut Vec<u8>, s: &str) -> Result<()> {
    let n = u32::try_from(s.len()).map_err(|_| PsdError::LimitExceeded("action name too long"))?;
    out.put_u32(n);
    out.put(s.as_bytes());
    Ok(())
}

fn read_item(r: &mut Reader<'_>) -> Result<AtnItem> {
    let expanded = r.u8()? != 0;
    let enabled = r.u8()? != 0;
    let with_dialog = r.u8()? != 0;
    let dialog_options = r.u8()?;
    let event = match &r.array::<4>()? {
        b"TEXT" => AtnEvent::Name(read_ascii(r)?),
        b"long" => AtnEvent::Code(r.array()?),
        other => return Err(PsdError::invalid(format!("unknown event kind {:?}", String::from_utf8_lossy(other)))),
    };
    let dictionary = read_ascii(r)?;
    let descriptor = match r.i32()? {
        0 => None,
        -1 => Some(Descriptor::read(r)?),
        n => return Err(PsdError::invalid(format!("bad descriptor flag {n}"))),
    };
    Ok(AtnItem { expanded, enabled, with_dialog, dialog_options, event, dictionary, descriptor })
}

fn read_action(r: &mut Reader<'_>) -> Result<AtnAction> {
    let function_key = r.u16()?;
    let shift = r.u8()? != 0;
    let command = r.u8()? != 0;
    let color = r.u16()?;
    let name = UnicodeString::read(r)?;
    let expanded = r.u8()? != 0;
    let n = r.u32()?;
    // Each item needs at least its 4 flags, the event kind, two empty names and the descriptor flag.
    r.check_count(u64::from(n), 20)?;
    if n as usize > MAX_ITEMS {
        return Err(PsdError::LimitExceeded("too many steps in one action"));
    }
    let mut items = Vec::with_capacity((n as usize).min(1024));
    for _ in 0..n {
        items.push(read_item(r)?);
    }
    Ok(AtnAction { function_key, shift, command, color, name, expanded, items })
}

/// Parses an `.atn` file.
pub fn parse(data: &[u8]) -> Result<AtnFile> {
    let mut r = Reader::new(data);
    let version = r.u32()?;
    if version != VERSION {
        return Err(PsdError::Unsupported(format!("actions file version {version} (only version 16, Photoshop 6 and later, is supported)")));
    }
    let name = UnicodeString::read(&mut r)?;
    let expanded = r.u8()? != 0;
    let n = r.u32()?;
    // Each action needs at least its key, modifiers, colour, an empty name, the flag and a count.
    r.check_count(u64::from(n), 15)?;
    if n as usize > MAX_ACTIONS {
        return Err(PsdError::LimitExceeded("too many actions in one set"));
    }
    let mut actions = Vec::with_capacity((n as usize).min(256));
    for _ in 0..n {
        actions.push(read_action(&mut r)?);
    }
    let mut warnings = Vec::new();
    if !r.is_empty() {
        warnings.push(format!("{} trailing bytes ignored", r.remaining()));
    }
    Ok(AtnFile { name, expanded, actions, warnings })
}

fn write_item(out: &mut Vec<u8>, item: &AtnItem) -> Result<()> {
    out.put_u8(u8::from(item.expanded));
    out.put_u8(u8::from(item.enabled));
    out.put_u8(u8::from(item.with_dialog));
    out.put_u8(item.dialog_options);
    match &item.event {
        AtnEvent::Name(n) => {
            out.put(b"TEXT");
            write_ascii(out, n)?;
        }
        AtnEvent::Code(c) => {
            out.put(b"long");
            out.put(c);
        }
    }
    write_ascii(out, &item.dictionary)?;
    match &item.descriptor {
        None => out.put_i32(0),
        Some(d) => {
            out.put_i32(-1);
            d.write(out);
        }
    }
    Ok(())
}

fn count(n: usize, what: &'static str) -> Result<u32> {
    u32::try_from(n).map_err(|_| PsdError::LimitExceeded(what))
}

/// Writes an `.atn` file (version 16).
pub fn write(f: &AtnFile) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.put_u32(VERSION);
    f.name.write(&mut out);
    out.put_u8(u8::from(f.expanded));
    out.put_u32(count(f.actions.len(), "too many actions")?);
    for a in &f.actions {
        out.put_u16(a.function_key);
        out.put_u8(u8::from(a.shift));
        out.put_u8(u8::from(a.command));
        out.put_u16(a.color);
        a.name.write(&mut out);
        out.put_u8(u8::from(a.expanded));
        out.put_u32(count(a.items.len(), "too many steps")?);
        for item in &a.items {
            write_item(&mut out, item)?;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::Value;

    fn px(v: f64) -> Value {
        Value::UnitFloat { unit: *b"#Pxl", value: v }
    }

    fn set() -> AtnFile {
        let blur = AtnItem {
            expanded: true,
            enabled: true,
            with_dialog: true,
            dialog_options: 2,
            event: AtnEvent::Code(*b"GsnB"),
            dictionary: "Gaussian Blur".into(),
            descriptor: Some(Descriptor::new("null").with("Rds ", px(2.5))),
        };
        let trim = AtnItem {
            event: AtnEvent::Name("trim".into()),
            dictionary: "Trim".into(),
            descriptor: Some(
                Descriptor { name: UnicodeString::new_nul("Trim"), class_id: crate::descriptor::Id::Str(b"trim".to_vec()), items: Vec::new() }
                    .with(
                        "trimBasedOn",
                        Value::Enumerated { type_id: crate::descriptor::Id::new("trimBasedOn"), value: crate::descriptor::Id::new("topLeftPixelColor") },
                    )
                    .with("Top ", Value::Boolean(true)),
            ),
            ..Default::default()
        };
        let flatten = AtnItem { enabled: false, event: AtnEvent::Code(*b"Fltn"), dictionary: "Flatten Image".into(), ..Default::default() };
        AtnFile {
            name: UnicodeString::new_nul("Set ✓"),
            expanded: true,
            actions: vec![
                AtnAction {
                    function_key: 2,
                    shift: true,
                    command: false,
                    color: 3,
                    name: UnicodeString::new_nul("Soften"),
                    expanded: true,
                    items: vec![blur, flatten],
                },
                AtnAction { name: UnicodeString::new_nul("Tidy"), items: vec![trim], ..Default::default() },
                AtnAction { name: UnicodeString::new_nul("Empty"), ..Default::default() },
            ],
            warnings: Vec::new(),
        }
    }

    #[test]
    fn round_trip() {
        let f = set();
        let bytes = write(&f).unwrap();
        let parsed = parse(&bytes).unwrap();
        assert_eq!(parsed, f);
        assert_eq!(write(&parsed).unwrap(), bytes, "writing is stable");
        assert_eq!(parsed.actions[0].items[0].event.id(), "GsnB");
        assert!(parsed.actions[1].items[0].event.is("trim"));
        assert_eq!(parsed.actions[1].items[0].event.id(), "trim");
    }

    /// The layout Photoshop writes for a one-step Trim action, assembled by hand from the
    /// format description: field by field, so a mistake names the field.
    #[test]
    fn layout_matches_the_specification() {
        let f = AtnFile {
            name: UnicodeString::new_nul("Trim"),
            expanded: false,
            actions: vec![AtnAction {
                name: UnicodeString::new_nul("Action 1"),
                expanded: true,
                items: vec![AtnItem {
                    expanded: true,
                    event: AtnEvent::Name("trim".into()),
                    dictionary: "Trim".into(),
                    descriptor: Some(Descriptor {
                        name: UnicodeString::new_nul("Trim"),
                        class_id: crate::descriptor::Id::Str(b"trim".to_vec()),
                        items: Vec::new(),
                    }),
                    ..Default::default()
                }],
                ..Default::default()
            }],
            warnings: Vec::new(),
        };
        let mut expect = Vec::new();
        expect.extend_from_slice(&[0, 0, 0, 16]); // version
        expect.extend_from_slice(&[0, 0, 0, 5, 0, b'T', 0, b'r', 0, b'i', 0, b'm', 0, 0]); // set name
        expect.push(0); // set collapsed
        expect.extend_from_slice(&[0, 0, 0, 1]); // one action
        expect.extend_from_slice(&[0, 0, 0, 0, 0, 0]); // no function key, no modifiers, no colour
        expect.extend_from_slice(&[0, 0, 0, 9]);
        for c in "Action 1\0".encode_utf16() {
            expect.extend_from_slice(&c.to_be_bytes());
        }
        expect.push(1); // action expanded
        expect.extend_from_slice(&[0, 0, 0, 1]); // one step
        expect.extend_from_slice(&[1, 1, 0, 0]); // expanded, enabled, no dialog, options 0
        expect.extend_from_slice(b"TEXT\0\0\0\x04trim");
        expect.extend_from_slice(b"\0\0\0\x04Trim");
        expect.extend_from_slice(&[0xff; 4]); // a descriptor follows
        expect.extend_from_slice(&[0, 0, 0, 5, 0, b'T', 0, b'r', 0, b'i', 0, b'm', 0, 0]); // class name
        expect.extend_from_slice(b"\0\0\0\x04trim"); // class id, length-4 string form
        expect.extend_from_slice(&[0, 0, 0, 0]); // no items
        assert_eq!(write(&f).unwrap(), expect);
        assert_eq!(parse(&expect).unwrap(), f);
    }

    #[test]
    fn truncation_never_parses_and_never_panics() {
        let bytes = write(&set()).unwrap();
        for cut in 0..bytes.len() {
            assert!(parse(&bytes[..cut]).is_err(), "cut at {cut}/{}", bytes.len());
        }
        let mut padded = bytes.clone();
        padded.extend_from_slice(b"junk");
        let f = parse(&padded).unwrap();
        assert_eq!(f.warnings, ["4 trailing bytes ignored"]);
        assert_eq!(f.actions, set().actions);
    }

    #[test]
    fn rejects_other_versions_and_bad_structure() {
        let mut bytes = write(&set()).unwrap();
        bytes[3] = 12;
        assert!(matches!(parse(&bytes), Err(PsdError::Unsupported(_))));
        // A count far beyond the data must fail before allocating.
        let mut huge = vec![0, 0, 0, 16, 0, 0, 0, 0, 0];
        huge.extend_from_slice(&u32::MAX.to_be_bytes());
        assert!(matches!(parse(&huge), Err(PsdError::UnexpectedEof { .. })));
        // Unknown event kinds and descriptor flags are reported, not guessed.
        let mut bad_kind = write(&set()).unwrap();
        let at = bad_kind.windows(4).position(|w| w == b"long").unwrap();
        bad_kind[at..at + 4].copy_from_slice(b"what");
        assert!(matches!(parse(&bad_kind), Err(PsdError::Invalid(_))));
        let mut bad_flag = write(&set()).unwrap();
        let at = bad_flag.windows(4).position(|w| w == [0xff; 4]).unwrap();
        bad_flag[at..at + 4].copy_from_slice(&[0, 0, 0, 7]);
        assert!(matches!(parse(&bad_flag), Err(PsdError::Invalid(_))));
    }

    #[test]
    fn garbage_never_panics() {
        let seed = write(&set()).unwrap();
        let mut x: u32 = 0x9e37_79b9;
        for i in 0..2000 {
            let mut bytes = seed.clone();
            for _ in 0..(i % 7 + 1) {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                let at = (x as usize) % bytes.len();
                bytes[at] = (x >> 8) as u8;
            }
            let _ = parse(&bytes);
        }
    }
}
