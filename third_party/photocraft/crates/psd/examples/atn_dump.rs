//! Print the actions and steps of `.atn` files, one event per line, with the top-level keys of
//! each step's descriptor: `cargo run -p photocraft-psd --example atn_dump -- set.atn…`
//!
//! Handy when adding a Photoshop event to the engine's `.atn` import table: it shows the event
//! id, the panel name and the parameter keys Photoshop recorded.

use photocraft_psd::atn;
use photocraft_psd::descriptor::{Descriptor, ReferenceItem, Value};

fn summary(v: &Value) -> String {
    match v {
        Value::Integer(i) => i.to_string(),
        Value::Double(d) => format!("{d}"),
        Value::UnitFloat { unit, value } => format!("{value}{}", String::from_utf8_lossy(unit)),
        Value::Boolean(b) => b.to_string(),
        Value::Text(t) => format!("{:?}", t.to_string_lossy()),
        Value::Enumerated { type_id, value } => format!("{}.{}", String::from_utf8_lossy(type_id.as_bytes()), String::from_utf8_lossy(value.as_bytes())),
        Value::Descriptor(d) | Value::GlobalObject(d) => format!("{{{}}}", keys(d)),
        Value::List(l) => format!("[{}]", l.iter().map(summary).collect::<Vec<_>>().join(", ")),
        Value::Reference(r) => format!("ref[{}]", r.iter().map(reference).collect::<Vec<_>>().join(" ")),
        other => format!("{other:?}").chars().take(40).collect(),
    }
}

fn id(i: &photocraft_psd::descriptor::Id) -> String {
    String::from_utf8_lossy(i.as_bytes()).trim().to_string()
}

fn reference(item: &ReferenceItem) -> String {
    match item {
        ReferenceItem::Property { class, key } => format!("prop({},{})", id(&class.class_id), id(key)),
        ReferenceItem::Class(c) => format!("Clss({})", id(&c.class_id)),
        ReferenceItem::Enumerated { class, type_id, value } => format!("Enmr({},{}.{})", id(&class.class_id), id(type_id), id(value)),
        ReferenceItem::Offset { class, offset } => format!("rele({},{offset})", id(&class.class_id)),
        ReferenceItem::Identifier { class, id: v } => format!("Idnt({},{v})", id(&class.class_id)),
        ReferenceItem::Index { class, index } => format!("indx({},{index})", id(&class.class_id)),
        ReferenceItem::Name { class, name } => format!("name({},{:?})", id(&class.class_id), name.to_string_lossy()),
    }
}

fn keys(d: &Descriptor) -> String {
    d.items.iter().map(|(k, v)| format!("{}={}", String::from_utf8_lossy(k.as_bytes()), summary(v))).collect::<Vec<_>>().join(" ")
}

fn main() {
    let mut failed = false;
    for path in std::env::args().skip(1) {
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("{path}: {e}");
                failed = true;
                continue;
            }
        };
        match atn::parse(&bytes) {
            Ok(f) => {
                println!(
                    "== {path}: set {:?}, {} actions{}",
                    f.name.to_string_lossy(),
                    f.actions.len(),
                    if f.warnings.is_empty() { String::new() } else { format!(" ({})", f.warnings.join("; ")) }
                );
                for a in &f.actions {
                    println!("  action {:?} ({} steps)", a.name.to_string_lossy(), a.items.len());
                    for item in &a.items {
                        let flags = format!("{}{}", if item.enabled { "" } else { " off" }, if item.with_dialog { " dialog" } else { "" });
                        match &item.descriptor {
                            Some(d) => println!("    {:<22} {:<28}{flags}  {}", item.event.id(), item.dictionary, keys(d)),
                            None => println!("    {:<22} {:<28}{flags}", item.event.id(), item.dictionary),
                        }
                    }
                }
            }
            Err(e) => {
                println!("== {path}: {e}");
                failed = true;
            }
        }
    }
    if failed {
        std::process::exit(1);
    }
}
