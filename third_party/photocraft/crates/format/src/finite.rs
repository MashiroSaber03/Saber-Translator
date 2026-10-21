//! Save-time check that the manifest holds only finite floats (issue #1101).
//!
//! JSON cannot spell NaN or the infinities: `serde_json` writes them as `null`, which the
//! manifest's plain `f32`/`f64` fields refuse on load. Such a document would save (and autosave)
//! with a success report and then never open again. So the save is refused instead, naming the
//! value, by walking the manifest with a serializer that produces nothing and fails on the first
//! non-finite float.

use std::fmt;

use serde::ser::{self, Serialize};

/// Path of the first non-finite float, e.g. `document.layers[2].content.adjustment.exposure`
/// (`Other` is reserved for a `Serialize` impl failing on its own, which the derived manifest
/// never does).
#[derive(Debug)]
pub enum Problem {
    NonFinite { path: String },
    Other(String),
}

impl fmt::Display for Problem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Problem::NonFinite { path } => write!(f, "`{path}` is NaN or infinite"),
            Problem::Other(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for Problem {}

impl ser::Error for Problem {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        Problem::Other(msg.to_string())
    }
}

/// `Ok(())` when every float in `value`'s serialized form is finite; otherwise the path of the
/// first one that is not.
pub fn check<T: Serialize + ?Sized>(value: &T) -> Result<(), Problem> {
    value.serialize(Check { path: String::new() })
}

struct Check {
    path: String,
}

impl Check {
    fn at(&self, seg: fmt::Arguments<'_>) -> Check {
        Check { path: format!("{}{}", self.path, seg) }
    }
    fn float(self, finite: bool) -> Result<(), Problem> {
        if finite { Ok(()) } else { Err(Problem::NonFinite { path: self.path.trim_start_matches('.').to_owned() }) }
    }
}

type R = Result<(), Problem>;

impl ser::Serializer for Check {
    type Ok = ();
    type Error = Problem;
    type SerializeSeq = Seq;
    type SerializeTuple = Seq;
    type SerializeTupleStruct = Seq;
    type SerializeTupleVariant = Seq;
    type SerializeMap = Map;
    type SerializeStruct = Struct;
    type SerializeStructVariant = Struct;

    fn serialize_bool(self, _: bool) -> R {
        Ok(())
    }
    fn serialize_i8(self, _: i8) -> R {
        Ok(())
    }
    fn serialize_i16(self, _: i16) -> R {
        Ok(())
    }
    fn serialize_i32(self, _: i32) -> R {
        Ok(())
    }
    fn serialize_i64(self, _: i64) -> R {
        Ok(())
    }
    fn serialize_i128(self, _: i128) -> R {
        Ok(())
    }
    fn serialize_u8(self, _: u8) -> R {
        Ok(())
    }
    fn serialize_u16(self, _: u16) -> R {
        Ok(())
    }
    fn serialize_u32(self, _: u32) -> R {
        Ok(())
    }
    fn serialize_u64(self, _: u64) -> R {
        Ok(())
    }
    fn serialize_u128(self, _: u128) -> R {
        Ok(())
    }
    fn serialize_f32(self, v: f32) -> R {
        self.float(v.is_finite())
    }
    fn serialize_f64(self, v: f64) -> R {
        self.float(v.is_finite())
    }
    fn serialize_char(self, _: char) -> R {
        Ok(())
    }
    fn serialize_str(self, _: &str) -> R {
        Ok(())
    }
    fn serialize_bytes(self, _: &[u8]) -> R {
        Ok(())
    }
    fn serialize_none(self) -> R {
        Ok(())
    }
    fn serialize_some<T: Serialize + ?Sized>(self, v: &T) -> R {
        v.serialize(self)
    }
    fn serialize_unit(self) -> R {
        Ok(())
    }
    fn serialize_unit_struct(self, _: &'static str) -> R {
        Ok(())
    }
    fn serialize_unit_variant(self, _: &'static str, _: u32, _: &'static str) -> R {
        Ok(())
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(self, _: &'static str, v: &T) -> R {
        v.serialize(self)
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(self, _: &'static str, _: u32, variant: &'static str, v: &T) -> R {
        v.serialize(self.at(format_args!(".{variant}")))
    }
    fn serialize_seq(self, _: Option<usize>) -> Result<Seq, Problem> {
        Ok(Seq { path: self.path, i: 0 })
    }
    fn serialize_tuple(self, _: usize) -> Result<Seq, Problem> {
        Ok(Seq { path: self.path, i: 0 })
    }
    fn serialize_tuple_struct(self, _: &'static str, _: usize) -> Result<Seq, Problem> {
        Ok(Seq { path: self.path, i: 0 })
    }
    fn serialize_tuple_variant(self, _: &'static str, _: u32, variant: &'static str, _: usize) -> Result<Seq, Problem> {
        Ok(Seq { path: format!("{}.{variant}", self.path), i: 0 })
    }
    fn serialize_map(self, _: Option<usize>) -> Result<Map, Problem> {
        Ok(Map { path: self.path, key: String::new() })
    }
    fn serialize_struct(self, _: &'static str, _: usize) -> Result<Struct, Problem> {
        Ok(Struct { path: self.path })
    }
    fn serialize_struct_variant(self, _: &'static str, _: u32, variant: &'static str, _: usize) -> Result<Struct, Problem> {
        Ok(Struct { path: format!("{}.{variant}", self.path) })
    }
}

struct Seq {
    path: String,
    i: usize,
}

impl Seq {
    fn next<T: Serialize + ?Sized>(&mut self, v: &T) -> R {
        let c = Check { path: format!("{}[{}]", self.path, self.i) };
        self.i += 1;
        v.serialize(c)
    }
}

impl ser::SerializeSeq for Seq {
    type Ok = ();
    type Error = Problem;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, v: &T) -> R {
        self.next(v)
    }
    fn end(self) -> R {
        Ok(())
    }
}

impl ser::SerializeTuple for Seq {
    type Ok = ();
    type Error = Problem;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, v: &T) -> R {
        self.next(v)
    }
    fn end(self) -> R {
        Ok(())
    }
}

impl ser::SerializeTupleStruct for Seq {
    type Ok = ();
    type Error = Problem;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, v: &T) -> R {
        self.next(v)
    }
    fn end(self) -> R {
        Ok(())
    }
}

impl ser::SerializeTupleVariant for Seq {
    type Ok = ();
    type Error = Problem;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, v: &T) -> R {
        self.next(v)
    }
    fn end(self) -> R {
        Ok(())
    }
}

struct Map {
    path: String,
    key: String,
}

impl ser::SerializeMap for Map {
    type Ok = ();
    type Error = Problem;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> R {
        // Keys are checked like any value; their text (when they have one) names the entry.
        key.serialize(Check { path: format!("{}.<key>", self.path) })?;
        self.key = serde_json::to_string(key).map_or_else(|_| "?".to_owned(), |k| k.trim_matches('"').to_owned());
        Ok(())
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, v: &T) -> R {
        v.serialize(Check { path: format!("{}.{}", self.path, self.key) })
    }
    fn end(self) -> R {
        Ok(())
    }
}

struct Struct {
    path: String,
}

impl Struct {
    fn field<T: Serialize + ?Sized>(&mut self, key: &'static str, v: &T) -> R {
        v.serialize(Check { path: format!("{}.{key}", self.path) })
    }
}

impl ser::SerializeStruct for Struct {
    type Ok = ();
    type Error = Problem;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, key: &'static str, v: &T) -> R {
        self.field(key, v)
    }
    fn end(self) -> R {
        Ok(())
    }
}

impl ser::SerializeStructVariant for Struct {
    type Ok = ();
    type Error = Problem;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, key: &'static str, v: &T) -> R {
        self.field(key, v)
    }
    fn end(self) -> R {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;

    #[derive(Serialize)]
    struct Inner {
        gamma: f32,
        offsets: Vec<f64>,
    }
    #[derive(Serialize)]
    enum Content {
        Exposure { exposure: f32 },
        Levels(f64, f64),
    }
    #[derive(Serialize)]
    struct Outer {
        dpi: f32,
        inner: Option<Inner>,
        layers: Vec<Content>,
        named: std::collections::BTreeMap<String, f32>,
    }

    fn outer() -> Outer {
        Outer {
            dpi: 72.0,
            inner: Some(Inner { gamma: 1.0, offsets: vec![0.0, 1.5] }),
            layers: vec![Content::Levels(0.0, 1.0), Content::Exposure { exposure: 0.5 }],
            named: [("blur".to_owned(), 2.0)].into_iter().collect(),
        }
    }

    fn path(v: &impl Serialize) -> String {
        match check(v) {
            Err(Problem::NonFinite { path }) => path,
            other => panic!("expected a non-finite report, got {other:?}"),
        }
    }

    #[test]
    fn finite_values_pass_and_each_position_is_named() {
        assert!(check(&outer()).is_ok());
        let mut o = outer();
        o.dpi = f32::NAN;
        assert_eq!(path(&o), "dpi");
        let mut o = outer();
        if let Some(i) = o.inner.as_mut() {
            i.offsets[1] = f64::INFINITY;
        }
        assert_eq!(path(&o), "inner.offsets[1]");
        let mut o = outer();
        o.layers[1] = Content::Exposure { exposure: f32::NEG_INFINITY };
        assert_eq!(path(&o), "layers[1].Exposure.exposure");
        let mut o = outer();
        o.layers[0] = Content::Levels(0.0, f64::NAN);
        assert_eq!(path(&o), "layers[0].Levels[1]");
        let mut o = outer();
        o.named.insert("blur".to_owned(), f32::NAN);
        assert_eq!(path(&o), "named.blur");
        assert_eq!(path(&f32::NAN), "");
    }
}
