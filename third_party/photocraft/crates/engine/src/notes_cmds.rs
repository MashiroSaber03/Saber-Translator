//! Note tool annotations (Window › Notes, File › Import › Notes…). Notes are document data
//! ([`photocraft_doc::Note`]), saved in `.pcraft` and as the PSD `Anno` block; each change is
//! one history step.

use photocraft_color::Color;
use photocraft_doc::{Document, Note};
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

fn has_notes(s: &Session) -> std::result::Result<(), String> {
    has_doc(s)?;
    if s.active().is_some_and(|d| !d.doc.notes.is_empty()) { Ok(()) } else { Err("the document has no notes".into()) }
}

/// PDF-style date (`D:YYYYMMDDhhmmssZ`) as Photoshop stores in notes; empty without a clock.
fn pdf_date() -> String {
    let iso = crate::analysis_cmds::now_iso();
    if iso.len() < 19 {
        return String::new();
    }
    let digits: String = iso[..19].chars().filter(char::is_ascii_digit).collect();
    format!("D:{digits}Z")
}

pub fn note_json(i: usize, n: &Note) -> Value {
    json!({"index": i, "author": n.author, "text": n.text, "color": n.color.to_rgb(), "position": n.position, "open": n.open, "modified": n.modified})
}

fn list_json(d: &Document) -> Value {
    json!({"notes": d.notes.iter().enumerate().map(|(i, n)| note_json(i, n)).collect::<Vec<_>>()})
}

fn color_of(p: &Value) -> Option<Color> {
    p.get("color")?;
    let c = crate::commands::color_param(p, "color", [1.0, 1.0, 0.51, 1.0]);
    Some(Color::rgb(c[0], c[1], c[2]))
}

fn index_of(s: &Session, p: &Value, cmd: &str) -> Result<usize> {
    let n = s.active().ok_or(EngineError::NoDocument)?.doc.notes.len();
    let i = p.get("index").and_then(Value::as_u64).ok_or_else(|| bad(cmd, "pass `index`"))? as usize;
    if i < n { Ok(i) } else { Err(bad(cmd, format!("no note {i} (the document has {n})"))) }
}

fn add(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "notes.add";
    let x = p.get("x").and_then(Value::as_f64).ok_or_else(|| bad(CMD, "pass `x` and `y`"))?;
    let y = p.get("y").and_then(Value::as_f64).ok_or_else(|| bad(CMD, "pass `x` and `y`"))?;
    if !x.is_finite() || !y.is_finite() {
        return Err(bad(CMD, "coordinates must be finite"));
    }
    let mut n = Note {
        author: p.get("author").and_then(Value::as_str).unwrap_or("").to_string(),
        text: p.get("text").and_then(Value::as_str).unwrap_or("").to_string(),
        position: [x, y],
        popup: [x + 20.0, y + 20.0, x + 260.0, y + 160.0],
        open: p.get("open").and_then(Value::as_bool).unwrap_or(true),
        modified: pdf_date(),
        ..Default::default()
    };
    if let Some(c) = color_of(p) {
        n.color = c;
    }
    let i = s.edit("New Note", |d, _| {
        d.notes.push(n);
        Ok(d.notes.len() - 1)
    })?;
    let d = s.active().ok_or(EngineError::NoDocument)?.doc.clone();
    Ok(json!({"index": i, "note": note_json(i, &d.notes[i])}))
}

fn set(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "notes.set";
    let i = index_of(s, p, CMD)?;
    let d = s.active().ok_or(EngineError::NoDocument)?.doc.clone();
    let mut n = d.notes[i].clone();
    if let Some(t) = p.get("text").and_then(Value::as_str) {
        n.text = t.to_string();
    }
    if let Some(a) = p.get("author").and_then(Value::as_str) {
        n.author = a.to_string();
    }
    if let Some(c) = color_of(p) {
        n.color = c;
    }
    if let Some(o) = p.get("open").and_then(Value::as_bool) {
        n.open = o;
    }
    let (x, y) = (p.get("x").and_then(Value::as_f64), p.get("y").and_then(Value::as_f64));
    if x.is_some() || y.is_some() {
        let (nx, ny) = (x.unwrap_or(n.position[0]), y.unwrap_or(n.position[1]));
        if !nx.is_finite() || !ny.is_finite() {
            return Err(bad(CMD, "coordinates must be finite"));
        }
        let (dx, dy) = (nx - n.position[0], ny - n.position[1]);
        n.position = [nx, ny];
        // The popup travels with its icon.
        n.popup = [n.popup[0] + dx, n.popup[1] + dy, n.popup[2] + dx, n.popup[3] + dy];
    }
    if n == d.notes[i] {
        return Ok(json!({"index": i, "note": note_json(i, &n), "changed": false}));
    }
    // Content edits update the modification date (opening/closing the popup doesn't).
    if n.text != d.notes[i].text || n.author != d.notes[i].author {
        n.modified = pdf_date();
    }
    let label = if n.position != d.notes[i].position { "Move Note" } else { "Edit Note" };
    let out = note_json(i, &n);
    s.edit(label, |d, _| {
        d.notes[i] = n;
        Ok(())
    })?;
    Ok(json!({"index": i, "note": out, "changed": true}))
}

fn delete(s: &mut Session, p: &Value) -> Result<Value> {
    if p.get("all").and_then(Value::as_bool) == Some(true) {
        let n = s.active().map_or(0, |d| d.doc.notes.len());
        s.edit("Delete All Notes", |d, _| {
            d.notes.clear();
            Ok(())
        })?;
        return Ok(json!({"deleted": n, "remaining": 0}));
    }
    let i = index_of(s, p, "notes.delete")?;
    s.edit("Delete Note", |d, _| {
        d.notes.remove(i);
        Ok(())
    })?;
    Ok(json!({"deleted": 1, "remaining": s.active().map_or(0, |d| d.doc.notes.len())}))
}

fn list(s: &mut Session, _p: &Value) -> Result<Value> {
    Ok(list_json(&s.active().ok_or(EngineError::NoDocument)?.doc))
}

#[cfg(not(target_arch = "wasm32"))]
fn read(path: &str) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|e| EngineError::Other(format!("{path}: {e}")))
}
#[cfg(target_arch = "wasm32")]
fn read(path: &str) -> Result<Vec<u8>> {
    Err(EngineError::Other(format!("cannot read {path}: no file system on the web")))
}

/// File › Import › Notes…: append the notes of another PSD or `.pcraft` file. Notes outside the
/// canvas are pulled inside, as Photoshop does when sizes differ.
fn import(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "file.import.notes";
    let path = p.get("path").and_then(Value::as_str).ok_or_else(|| bad(CMD, "pass `path` (.psd, .psb or .pcraft)"))?;
    let bytes = read(path)?;
    let name = std::path::Path::new(path).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| path.to_string());
    import_notes_from(s, &name, &bytes)
}

/// File › Import › Notes… from file bytes (what a file picker returns).
pub fn import_notes_from(s: &mut Session, name: &str, bytes: &[u8]) -> Result<Value> {
    let src = crate::file_cmds::import(name, bytes)?;
    if src.notes.is_empty() {
        return Err(EngineError::Other(format!("{name} has no notes")));
    }
    let d = s.active().ok_or(EngineError::NoDocument)?.doc.clone();
    let (w, h) = (f64::from(d.size.width), f64::from(d.size.height));
    let mut notes = src.notes;
    for n in &mut notes {
        let (nx, ny) = (n.position[0].clamp(0.0, (w - 16.0).max(0.0)), n.position[1].clamp(0.0, (h - 20.0).max(0.0)));
        let (dx, dy) = (nx - n.position[0], ny - n.position[1]);
        n.position = [nx, ny];
        n.popup = [n.popup[0] + dx, n.popup[1] + dy, n.popup[2] + dx, n.popup[3] + dy];
    }
    let count = notes.len();
    s.edit("Import Notes", |d, _| {
        d.notes.extend(notes);
        Ok(())
    })?;
    let mut r = list_json(&s.active().ok_or(EngineError::NoDocument)?.doc);
    r["imported"] = json!(count);
    Ok(r)
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "notes.add",
            label: "New Note",
            menu: &[],
            shortcut: None,
            params: r##"{"x":px,"y":px,"text":str="","author":str="","color":"#rrggbb"|[r,g,b]=pale yellow,"open":bool=true}"##,
            enabled: has_doc,
            run: add,
            journal: true,
        },
        CommandSpec {
            id: "notes.set",
            label: "Edit Note",
            menu: &[],
            shortcut: None,
            params: r##"{"index":n,"text":str?,"author":str?,"color":"#rrggbb"|[r,g,b]?,"x":px?,"y":px?,"open":bool?}"##,
            enabled: has_notes,
            run: set,
            journal: true,
        },
        CommandSpec {
            id: "notes.delete",
            label: "Delete Note",
            menu: &[],
            shortcut: None,
            params: r##"{"index":n}|{"all":true}"##,
            enabled: has_notes,
            run: delete,
            journal: true,
        },
        CommandSpec {
            id: "notes.list",
            label: "Notes",
            menu: &[],
            shortcut: None,
            params: r##"{} → notes (index, author, text, colour, position, open, modified)"##,
            enabled: has_doc,
            run: list,
            journal: false,
        },
        CommandSpec {
            id: "file.import.notes",
            label: "Notes…",
            menu: &["File", "Import"],
            shortcut: None,
            params: r##"{"path":".psd|.psb|.pcraft with notes"}"##,
            enabled: has_doc,
            run: import,
            journal: true,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 100, "height": 80})).unwrap();
        s
    }

    #[test]
    fn add_edit_move_delete_with_undo() {
        let mut s = session();
        assert!(!s.is_enabled("notes.delete"));
        let r = s.execute("notes.add", json!({"x": 10, "y": 12, "text": "fix this", "author": "QA", "color": "#ff0000"})).unwrap();
        assert_eq!(r["index"], 0);
        assert_eq!(r["note"]["color"], json!([1.0, 0.0, 0.0]));
        s.execute("notes.add", json!({"x": 50, "y": 40})).unwrap();
        s.execute("notes.set", json!({"index": 0, "text": "fixed", "x": 20})).unwrap();
        let d = s.active().unwrap().doc.clone();
        assert_eq!(d.notes[0].text, "fixed");
        assert_eq!(d.notes[0].position, [20.0, 12.0]);
        assert_eq!(d.notes[0].popup[0], 40.0);
        // Unchanged edits don't add history.
        let before = s.active().unwrap().history.past_len();
        assert_eq!(s.execute("notes.set", json!({"index": 0, "text": "fixed"})).unwrap()["changed"], false);
        assert_eq!(s.active().unwrap().history.past_len(), before);
        s.execute("notes.delete", json!({"index": 1})).unwrap();
        assert_eq!(s.execute("notes.list", json!({})).unwrap()["notes"].as_array().unwrap().len(), 1);
        assert!(s.undo());
        assert_eq!(s.active().unwrap().doc.notes.len(), 2);
        s.execute("notes.delete", json!({"all": true})).unwrap();
        assert!(s.active().unwrap().doc.notes.is_empty());
        assert!(s.execute("notes.set", json!({"index": 0})).is_err());
    }

    #[test]
    fn bad_params() {
        let mut s = session();
        assert!(s.execute("notes.add", json!({"x": 1})).is_err());
        s.execute("notes.add", json!({"x": 1, "y": 1})).unwrap();
        assert!(s.execute("notes.set", json!({"index": 5, "text": "x"})).is_err());
        assert!(s.execute("notes.set", json!({})).is_err());
        assert!(s.execute("file.import.notes", json!({})).is_err());
        assert!(s.execute("file.import.notes", json!({"path": "/nonexistent/x.psd"})).is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn import_notes_from_pcraft_and_psd() {
        let dir = std::env::temp_dir().join(format!("pc-notes-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut src = Document::new("src", photocraft_doc::Size::new(400, 400), photocraft_color::ColorMode::Rgb, photocraft_color::SampleType::U8);
        src.notes.push(Note { text: "from pcraft".into(), position: [300.0, 300.0], ..Default::default() });
        let pc = dir.join("a.pcraft");
        std::fs::write(&pc, photocraft_format::save_to_bytes(&src, &Default::default()).unwrap()).unwrap();
        let mut s = session();
        let r = s.execute("file.import.notes", json!({"path": pc.to_str().unwrap()})).unwrap();
        assert_eq!(r["imported"], 1);
        // Pulled inside the 100×80 canvas.
        let n = &s.active().unwrap().doc.notes[0];
        assert_eq!(n.text, "from pcraft");
        assert!(n.position[0] <= 84.0 && n.position[1] <= 60.0);
        // A PSD with notes (corpus sample, feature `corpus`) appends two more.
        #[cfg(feature = "corpus")]
        {
            let psd = concat!(env!("CARGO_MANIFEST_DIR"), "/../../corpus/psd/ag-psd/read-write/annotations/src.psd");
            assert!(std::path::Path::new(psd).exists(), "{psd} is missing: run `cargo xtask corpus --all`");
            s.execute("file.import.notes", json!({"path": psd})).unwrap();
            assert_eq!(s.active().unwrap().doc.notes.len(), 3);
        }
        // A file without notes is an error.
        let empty = dir.join("b.pcraft");
        src.notes.clear();
        std::fs::write(&empty, photocraft_format::save_to_bytes(&src, &Default::default()).unwrap()).unwrap();
        assert!(s.execute("file.import.notes", json!({"path": empty.to_str().unwrap()})).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
