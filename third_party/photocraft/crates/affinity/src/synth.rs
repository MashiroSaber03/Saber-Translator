//! Synthetic containers for tests and fuzz seeds (cargo feature `synth`).
//!
//! These builders write the layout the reader accepts, so tests can describe documents in a few
//! lines. They are **not** an Affinity exporter: nothing written here has been opened in Affinity.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::stream::Tag;

/// A field value to encode.
#[derive(Debug, Clone)]
pub enum F {
    U8(u8),
    U32(u32),
    I32(i32),
    F32(f32),
    F64(f64),
    Bool(bool),
    Enum(u16, u16),
    Str(String),
    /// f64 vector of 2..6 values (points, rectangles, transforms).
    F64s(Vec<f64>),
    /// Fixed-size struct (colour components), 1..64 bytes.
    Struct(Vec<u8>),
    /// Array of fixed-size records, all `size` bytes.
    Records(u16, Vec<Vec<u8>>),
    U8s(Vec<u8>),
    Entry(String),
    /// Shared object: `Def` defines id with class chain and fields, `Ref` links back.
    Def(u32, Vec<Tag>, Vec<(Tag, F)>),
    Ref(u32),
    Null,
    /// Inline object.
    Obj(Tag, Vec<(Tag, F)>),
    /// Positional (untagged) object.
    Pos(Vec<F>),
    /// Array of shared objects / links.
    Shared(Vec<F>),
    /// Array of inline objects of one class.
    Objs(Tag, Vec<Vec<(Tag, F)>>),
    /// Raw bytes appended verbatim (malformed-input tests).
    Raw(Vec<u8>),
}

pub fn tag(s: &[u8; 4]) -> Tag {
    Tag::of(s)
}

fn put_tag(out: &mut Vec<u8>, t: Tag) {
    out.extend(t.0.to_le_bytes());
}

fn code(f: &F) -> u8 {
    match f {
        F::U8(_) => 0x01,
        F::U32(_) => 0x03,
        F::I32(_) => 0x07,
        F::F32(_) => 0x09,
        F::F64(_) => 0x0a,
        F::Bool(_) => 0x29,
        F::Enum(..) => 0x2a,
        F::Str(_) => 0x2b,
        F::F64s(v) => 0x22 + v.len() as u8,
        F::Struct(b) => 0x34 + b.len() as u8,
        F::Records(..) => 0x80 | 0x2c,
        F::U8s(_) => 0x80 | 0x01,
        F::Entry(_) => 0x33,
        F::Def(..) | F::Ref(_) => 0x31,
        F::Null => 0x31,
        F::Obj(..) => 0x32,
        F::Pos(_) => 0x30,
        F::Shared(_) => 0x80 | 0x31,
        F::Objs(..) => 0x80 | 0x32,
        F::Raw(_) => 0,
    }
}

fn payload(out: &mut Vec<u8>, f: &F) {
    match f {
        F::U8(v) => out.push(*v),
        F::U32(v) => out.extend(v.to_le_bytes()),
        F::I32(v) => out.extend(v.to_le_bytes()),
        F::F32(v) => out.extend(v.to_le_bytes()),
        F::F64(v) => out.extend(v.to_le_bytes()),
        F::Bool(v) => out.push(u8::from(*v)),
        F::Enum(id, ver) => {
            out.extend(id.to_le_bytes());
            out.extend(ver.to_le_bytes());
        }
        F::Str(s) => {
            out.extend((s.len() as u32).to_le_bytes());
            out.extend(s.as_bytes());
        }
        F::F64s(v) => v.iter().for_each(|x| out.extend(x.to_le_bytes())),
        F::Struct(b) => out.extend(b),
        F::Records(size, recs) => {
            out.extend((recs.len() as u32).to_le_bytes());
            out.extend(size.to_le_bytes());
            recs.iter().for_each(|r| out.extend(r));
        }
        F::U8s(v) => {
            out.extend((v.len() as u32).to_le_bytes());
            out.extend(v);
        }
        F::Entry(name) => {
            put_tag(out, tag(b"Data"));
            out.extend((name.len() as u32).to_le_bytes());
            out.extend(name.as_bytes());
        }
        F::Def(id, chain, fields) => {
            out.push(1);
            out.extend(id.to_le_bytes());
            // Class chain: each level but the last carries no fields here; the last ends the chain.
            let (last, rest) = chain.split_last().expect("class chain");
            for t in rest {
                out.push(0);
                put_tag(out, *t);
                out.extend(0u16.to_le_bytes());
                out.push(0);
            }
            out.push(1);
            put_tag(out, *last);
            fields_into(out, fields, true);
        }
        F::Ref(id) => {
            out.push(2);
            out.extend(id.to_le_bytes());
        }
        F::Null => out.push(0),
        F::Obj(class, fields) => {
            out.push(1);
            put_tag(out, *class);
            out.extend(0u16.to_le_bytes());
            fields_into(out, fields, true);
        }
        F::Pos(items) => {
            for f in items {
                out.push(code(f));
                payload(out, f);
            }
            out.push(0);
        }
        F::Shared(items) => {
            out.extend((items.len() as u32).to_le_bytes());
            items.iter().for_each(|f| payload(out, f));
        }
        F::Objs(class, items) => {
            out.extend((items.len() as u32).to_le_bytes());
            put_tag(out, *class);
            out.extend(0u16.to_le_bytes());
            for fields in items {
                out.push(1);
                fields_into(out, fields, true);
            }
        }
        F::Raw(b) => out.extend(b),
    }
}

fn fields_into(out: &mut Vec<u8>, fields: &[(Tag, F)], tagged: bool) {
    for (t, f) in fields {
        if let F::Raw(b) = f {
            out.extend(b);
            continue;
        }
        out.push(code(f));
        if tagged {
            put_tag(out, *t);
        }
        payload(out, f);
    }
    out.push(0);
}

/// A `doc.dat` stream with root class `Pers`.
pub fn stream(root: &[(Tag, F)]) -> Vec<u8> {
    let mut out = b"\x00\xffKS".to_vec();
    out.extend(2u16.to_le_bytes());
    put_tag(&mut out, tag(b"Pers"));
    out.extend(0u16.to_le_bytes());
    out.extend(30u32.to_le_bytes());
    fields_into(&mut out, root, true);
    out
}

/// How an entry is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Stored,
    Zlib,
    Zstd,
}

/// A version-12 container with one `#FT4` table holding `entries`, then an optional thumbnail record.
pub fn container(entries: &[(&str, &[u8], Method)], thumbnail_png: Option<&[u8]>) -> Vec<u8> {
    let mut out = vec![0u8; 72];
    out[..4].copy_from_slice(crate::container::MAGIC);
    out[4..6].copy_from_slice(&12u16.to_le_bytes());
    out[6..8].copy_from_slice(&4u16.to_le_bytes());
    out[8..12].copy_from_slice(b"nsrP");
    out[12..16].copy_from_slice(b"#Inf");
    out[64..68].copy_from_slice(b"Prot");
    let mut records = Vec::new();
    for (i, (name, data, method)) in entries.iter().enumerate() {
        let packed = match method {
            Method::Stored => data.to_vec(),
            Method::Zlib => {
                use std::io::Write;
                let mut e = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
                e.write_all(data).unwrap();
                e.finish().unwrap()
            }
            Method::Zstd => ruzstd::encoding::compress_to_vec(*data, ruzstd::encoding::CompressionLevel::Fastest),
        };
        let offset = out.len() as u64;
        out.extend(b"#Fil");
        out.extend(&packed);
        let mut r = Vec::new();
        r.extend((i as u32 + 1).to_le_bytes());
        r.push(0);
        r.extend(offset.to_le_bytes());
        r.extend((data.len() as u64).to_le_bytes());
        r.extend((packed.len() as u64).to_le_bytes());
        r.extend(crc32fast::hash(data).to_le_bytes());
        r.push(match method {
            Method::Stored => 0,
            Method::Zlib => 1,
            Method::Zstd => 2,
        });
        r.extend(0u32.to_le_bytes());
        r.extend(0u32.to_le_bytes());
        r.extend((name.len() as u16).to_le_bytes());
        r.extend(name.as_bytes());
        records.push(r);
    }
    let fat = out.len() as u64;
    out.extend(b"#FT4");
    out.extend(0u64.to_le_bytes());
    out.extend(1_700_000_000u64.to_le_bytes());
    out.extend([0u8; 24]);
    out.extend((records.len() as u32).to_le_bytes());
    out.extend([0u8; 8]);
    out.extend(0u16.to_le_bytes());
    out.push(0);
    records.iter().for_each(|r| out.extend(r));
    let thumb = out.len() as u64;
    out[16..24].copy_from_slice(&fat.to_le_bytes());
    if let Some(png) = thumbnail_png {
        out[24..32].copy_from_slice(&thumb.to_le_bytes());
        out.extend(b"\xff\xff\xff\xffThmb");
        out.extend(1u32.to_le_bytes());
        out.extend((png.len() as u32 + 13).to_le_bytes());
        out.extend(29u32.to_le_bytes());
        out.extend(0u32.to_le_bytes());
        out.extend((png.len() as u32).to_le_bytes());
        out.push(1);
        out.extend(png);
    }
    out
}
