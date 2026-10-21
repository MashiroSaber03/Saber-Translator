//! Typed reads of [`Stream`] objects. Every accessor returns `None` when a field is missing or has
//! another type, so the document mapping treats odd data as absent rather than failing.

use crate::stream::{ObjId, Object, Stream, Tag, Value};

impl Stream {
    pub fn object(&self, id: ObjId) -> Option<&Object> {
        self.objects.get(id)
    }

    pub fn field(&self, id: ObjId, tag: &[u8; 4]) -> Option<&Value> {
        self.object(id)?.get(Tag::of(tag))
    }

    pub fn class(&self, id: ObjId) -> Option<Tag> {
        self.object(id).map(|o| o.class)
    }

    pub fn is(&self, id: ObjId, class: &[u8; 4]) -> bool {
        self.class(id) == Some(Tag::of(class))
    }

    /// A child object (shared, linked or inline); `None` for null or missing.
    pub fn obj(&self, id: ObjId, tag: &[u8; 4]) -> Option<ObjId> {
        match self.field(id, tag)? {
            Value::Obj(o) => *o,
            _ => None,
        }
    }

    /// An array of objects (nulls skipped).
    pub fn objs(&self, id: ObjId, tag: &[u8; 4]) -> Vec<ObjId> {
        match self.field(id, tag) {
            Some(Value::Array(a)) => a.iter().filter_map(|v| if let Value::Obj(Some(o)) = v { Some(*o) } else { None }).collect(),
            _ => Vec::new(),
        }
    }

    pub fn f64(&self, id: ObjId, tag: &[u8; 4]) -> Option<f64> {
        match self.field(id, tag)? {
            Value::Float(f) if f.is_finite() => Some(*f),
            _ => None,
        }
    }

    pub fn int(&self, id: ObjId, tag: &[u8; 4]) -> Option<i64> {
        match self.field(id, tag)? {
            Value::Int(i) => Some(*i),
            Value::UInt(u) => i64::try_from(*u).ok(),
            _ => None,
        }
    }

    pub fn bool(&self, id: ObjId, tag: &[u8; 4]) -> Option<bool> {
        match self.field(id, tag)? {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn str(&self, id: ObjId, tag: &[u8; 4]) -> Option<&str> {
        match self.field(id, tag)? {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn enumeration(&self, id: ObjId, tag: &[u8; 4]) -> Option<(u16, u16)> {
        match self.field(id, tag)? {
            Value::Enum { id, version } => Some((*id, *version)),
            _ => None,
        }
    }

    /// A fixed vector of `N` finite floats (points, rectangles, transforms).
    pub fn floats<const N: usize>(&self, id: ObjId, tag: &[u8; 4]) -> Option<[f64; N]> {
        match self.field(id, tag)? {
            Value::Floats(v) if v.len() == N && v.iter().all(|f| f.is_finite()) => {
                let mut a = [0.0; N];
                a.copy_from_slice(v);
                Some(a)
            }
            _ => None,
        }
    }

    pub fn bytes(&self, id: ObjId, tag: &[u8; 4]) -> Option<&[u8]> {
        match self.field(id, tag)? {
            Value::Bytes(b) | Value::Blob(b) => Some(b),
            _ => None,
        }
    }

    pub fn entry(&self, id: ObjId, tag: &[u8; 4]) -> Option<&str> {
        match self.field(id, tag)? {
            Value::Entry(s) => Some(s),
            _ => None,
        }
    }

    /// Untagged fields of a positional object, in order.
    pub fn positional(&self, id: ObjId) -> impl Iterator<Item = &Value> {
        self.object(id).into_iter().flat_map(|o| o.fields.iter().map(|(_, v)| v))
    }
}
