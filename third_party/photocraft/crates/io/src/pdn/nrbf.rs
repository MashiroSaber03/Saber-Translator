//! Data-only subset of MS-NRBF used by PDN3. References stay as ids; no .NET types are executed.
//! https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-nrbf/

use std::collections::HashMap;
use std::sync::Arc;

use photocraft_raster::Interrupt;

use super::{Result, invalid};

const MAX_ITEMS: usize = 1_000_000;
const MAX_STRINGS: usize = 64 << 20;
const MAX_DEPTH: usize = 64;

pub(super) struct Reader<'a> {
    pub bytes: &'a [u8],
    pub pos: usize,
}

impl<'a> Reader<'a> {
    pub fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(len).ok_or_else(|| invalid("length overflow"))?;
        let b = self.bytes.get(self.pos..end).ok_or_else(|| invalid("truncated document"))?;
        self.pos = end;
        Ok(b)
    }

    pub fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?.first().copied().unwrap_or_default())
    }

    pub fn le_i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().map_err(|_| invalid("invalid integer"))?))
    }

    pub fn be_u32(&mut self) -> Result<usize> {
        usize::try_from(u32::from_be_bytes(self.take(4)?.try_into().map_err(|_| invalid("invalid integer"))?)).map_err(|_| invalid("length overflow"))
    }

    fn count(&mut self) -> Result<usize> {
        let n = usize::try_from(self.le_i32()?).map_err(|_| invalid("negative count"))?;
        if n > MAX_ITEMS {
            return Err(invalid("serialized array or class is too large"));
        }
        Ok(n)
    }
}

#[derive(Debug)]
pub(super) struct Class {
    name: String,
    names: Vec<String>,
    primitives: Vec<Option<u8>>,
}

#[derive(Debug)]
pub(super) enum Value {
    Null,
    Bool(bool),
    Int(i64),
    String(String),
    Ref(i32),
    Array(Vec<Value>),
    Object { class: Arc<Class>, members: Vec<Value> },
}

pub(super) struct Graph {
    pub root: Value,
    objects: HashMap<i32, Value>,
    order: Vec<i32>,
}

impl Graph {
    pub fn objects_of_class<'a>(&'a self, name: &'a str) -> impl Iterator<Item = (i32, &'a Value)> {
        self.order
            .iter()
            .filter_map(|id| self.objects.get(id).map(|v| (*id, v)))
            .filter(move |(_, v)| matches!(v, Value::Object { class, .. } if class.name == name))
    }
    pub fn resolve<'a>(&'a self, mut value: &'a Value) -> Result<&'a Value> {
        for _ in 0..MAX_DEPTH {
            match value {
                Value::Ref(id) => value = self.objects.get(id).ok_or_else(|| invalid("missing object reference"))?,
                _ => return Ok(value),
            }
        }
        Err(invalid("cyclic or excessively deep object reference"))
    }

    pub fn class_name<'a>(&'a self, value: &'a Value) -> Result<&'a str> {
        match self.resolve(value)? {
            Value::Object { class, .. } => Ok(&class.name),
            _ => Err(invalid("expected a serialized object")),
        }
    }

    pub fn field<'a>(&'a self, value: &'a Value, name: &str) -> Result<&'a Value> {
        self.optional_field(value, name)?.ok_or_else(|| invalid(format!("missing field {name}")))
    }

    pub fn optional_field<'a>(&'a self, value: &'a Value, name: &str) -> Result<Option<&'a Value>> {
        match self.resolve(value)? {
            Value::Object { class, members } => Ok(class.names.iter().position(|n| n == name).and_then(|i| members.get(i))),
            _ => Err(invalid("expected a serialized object")),
        }
    }

    pub fn integer(&self, value: &Value) -> Result<i64> {
        match self.resolve(value)? {
            Value::Int(n) => Ok(*n),
            _ => Err(invalid("expected an integer")),
        }
    }

    pub fn boolean(&self, value: &Value) -> Result<bool> {
        match self.resolve(value)? {
            Value::Bool(b) => Ok(*b),
            _ => Err(invalid("expected a boolean")),
        }
    }

    pub fn string<'a>(&'a self, value: &'a Value) -> Result<&'a str> {
        match self.resolve(value)? {
            Value::String(s) => Ok(s),
            _ => Err(invalid("expected a string")),
        }
    }

    pub fn array<'a>(&'a self, value: &'a Value) -> Result<&'a [Value]> {
        match self.resolve(value)? {
            Value::Array(a) => Ok(a),
            _ => Err(invalid("expected an array")),
        }
    }
}

struct Parser<'a, 'b> {
    reader: &'b mut Reader<'a>,
    ctl: &'b Interrupt<'b>,
    graph: Graph,
    classes: HashMap<i32, Arc<Class>>,
    items: usize,
    strings: usize,
}

pub(super) fn parse(reader: &mut Reader<'_>, ctl: &Interrupt<'_>) -> Result<Graph> {
    if reader.byte()? != 0 {
        return Err(invalid("missing NRBF stream header"));
    }
    let root = reader.le_i32()?;
    let _header = reader.le_i32()?;
    if root == 0 || reader.le_i32()? != 1 || reader.le_i32()? != 0 {
        return Err(invalid("unsupported NRBF stream version"));
    }
    let mut parser = Parser {
        reader,
        ctl,
        graph: Graph { root: Value::Ref(root), objects: HashMap::new(), order: Vec::new() },
        classes: HashMap::new(),
        items: 0,
        strings: 0,
    };
    loop {
        let tag = parser.reader.byte()?;
        if tag == 11 {
            break;
        }
        parser.record(tag, 0)?;
    }
    parser.graph.resolve(&parser.graph.root)?;
    Ok(parser.graph)
}

impl Parser<'_, '_> {
    fn budget(&mut self, n: usize) -> Result<()> {
        self.items = self.items.checked_add(n).ok_or_else(|| invalid("object count overflow"))?;
        if self.items > MAX_ITEMS {
            return Err(invalid("too many serialized objects or members"));
        }
        Ok(())
    }

    fn string(&mut self) -> Result<String> {
        let mut len = 0usize;
        for shift in (0..35).step_by(7) {
            let b = self.reader.byte()?;
            if shift == 28 && b > 7 {
                return Err(invalid("invalid string length"));
            }
            len |= usize::from(b & 127) << shift;
            if b & 128 == 0 {
                self.strings = self.strings.checked_add(len).ok_or_else(|| invalid("string length overflow"))?;
                if self.strings > MAX_STRINGS {
                    return Err(invalid("serialized strings are too large"));
                }
                return String::from_utf8(self.reader.take(len)?.to_vec()).map_err(|_| invalid("invalid UTF-8 string"));
            }
        }
        Err(invalid("invalid string length"))
    }

    fn primitive(&mut self, tag: u8) -> Result<Value> {
        macro_rules! number {
            ($ty:ty, $variant:ident) => {
                Value::$variant(<$ty>::from_le_bytes(self.reader.take(size_of::<$ty>())?.try_into().map_err(|_| invalid("invalid primitive"))?).into())
            };
        }
        Ok(match tag {
            1 => match self.reader.byte()? {
                0 => Value::Bool(false),
                1 => Value::Bool(true),
                _ => return Err(invalid("invalid boolean")),
            },
            2 => Value::Int(i64::from(self.reader.byte()?)),
            3 => {
                let first = self.reader.byte()?;
                let len = match first {
                    0..=0x7f => 1,
                    0xc2..=0xdf => 2,
                    0xe0..=0xef => 3,
                    _ => return Err(invalid("invalid character")),
                };
                let mut b = vec![first];
                b.extend_from_slice(self.reader.take(len - 1)?);
                Value::String(String::from_utf8(b).map_err(|_| invalid("invalid character"))?)
            }
            5 | 18 => Value::String(self.string()?),
            6 => {
                self.reader.take(8)?;
                Value::Null
            }
            7 => number!(i16, Int),
            8 => number!(i32, Int),
            9 | 12 => number!(i64, Int),
            10 => Value::Int(i64::from(self.reader.byte()? as i8)),
            11 => {
                self.reader.take(4)?;
                Value::Null
            }
            13 | 16 => {
                let n = u64::from_le_bytes(self.reader.take(8)?.try_into().map_err(|_| invalid("invalid integer"))?);
                // Metadata dates may use the high bits; their bit pattern stays intact.
                Value::Int(n as i64)
            }
            14 => number!(u16, Int),
            15 => number!(u32, Int),
            17 => Value::Null,
            _ => return Err(invalid(format!("unsupported primitive type {tag}"))),
        })
    }

    fn type_info(&mut self, tag: u8) -> Result<Option<u8>> {
        match tag {
            0 => Ok(Some(self.reader.byte()?)),
            1 | 2 | 5 | 6 => Ok(None),
            3 => {
                self.string()?;
                Ok(None)
            }
            4 => {
                self.string()?;
                self.reader.le_i32()?;
                Ok(None)
            }
            7 => {
                self.reader.byte()?;
                Ok(None)
            }
            _ => Err(invalid(format!("unsupported binary type {tag}"))),
        }
    }

    fn reserve(&mut self, id: i32) -> Result<()> {
        if id == 0 || self.graph.objects.insert(id, Value::Null).is_some() {
            return Err(invalid("duplicate or zero object id"));
        }
        self.graph.order.push(id);
        Ok(())
    }

    fn value(&mut self, depth: usize) -> Result<Value> {
        loop {
            let tag = self.reader.byte()?;
            if tag == 12 {
                self.record(tag, depth)?;
            } else {
                return self.record(tag, depth);
            }
        }
    }

    fn object(&mut self, id: i32, class: Arc<Class>, depth: usize) -> Result<Value> {
        self.reserve(id)?;
        self.budget(class.names.len())?;
        let mut members = Vec::with_capacity(class.names.len());
        for primitive in &class.primitives {
            members.push(match primitive {
                Some(p) => self.primitive(*p)?,
                None => self.value(depth + 1)?,
            });
        }
        self.graph.objects.insert(id, Value::Object { class, members });
        Ok(Value::Ref(id))
    }

    fn array(&mut self, id: i32, len: usize, primitive: Option<u8>, depth: usize) -> Result<Value> {
        self.reserve(id)?;
        self.budget(len)?;
        if primitive == Some(2) {
            self.reader.take(len)?;
            return Ok(Value::Ref(id));
        }
        let mut values = Vec::with_capacity(len);
        while values.len() < len {
            self.ctl.check().map_err(|_| crate::IoError::Cancelled)?;
            if let Some(p) = primitive {
                values.push(self.primitive(p)?);
                continue;
            }
            let tag = self.reader.byte()?;
            if matches!(tag, 13 | 14) {
                let count = if tag == 13 { usize::from(self.reader.byte()?) } else { self.reader.count()? };
                if count == 0 || count > len - values.len() {
                    return Err(invalid("invalid array null run"));
                }
                values.resize_with(values.len() + count, || Value::Null);
            } else if tag == 12 {
                self.record(tag, depth + 1)?;
            } else {
                values.push(self.record(tag, depth + 1)?);
            }
        }
        self.graph.objects.insert(id, Value::Array(values));
        Ok(Value::Ref(id))
    }

    fn record(&mut self, tag: u8, depth: usize) -> Result<Value> {
        self.ctl.check().map_err(|_| crate::IoError::Cancelled)?;
        self.budget(1)?;
        if depth > MAX_DEPTH {
            return Err(invalid("serialized objects nested too deeply"));
        }
        match tag {
            1 => {
                let id = self.reader.le_i32()?;
                let metadata = self.reader.le_i32()?;
                let class = self.classes.get(&metadata).cloned().ok_or_else(|| invalid("missing class metadata"))?;
                self.object(id, class, depth)
            }
            2..=5 => {
                let id = self.reader.le_i32()?;
                let name = self.string()?;
                let count = self.reader.count()?;
                self.budget(count)?;
                let names = (0..count).map(|_| self.string()).collect::<Result<Vec<_>>>()?;
                let primitives = if tag >= 4 {
                    let types = self.reader.take(count)?.to_vec();
                    types.into_iter().map(|t| self.type_info(t)).collect::<Result<Vec<_>>>()?
                } else {
                    vec![None; count]
                };
                if matches!(tag, 3 | 5) {
                    self.reader.le_i32()?;
                }
                let class = Arc::new(Class { name, names, primitives });
                if self.classes.insert(id, class.clone()).is_some() {
                    return Err(invalid("duplicate class metadata"));
                }
                self.object(id, class, depth)
            }
            6 => {
                let id = self.reader.le_i32()?;
                self.reserve(id)?;
                let value = Value::String(self.string()?);
                self.graph.objects.insert(id, value);
                Ok(Value::Ref(id))
            }
            7 => {
                let id = self.reader.le_i32()?;
                let kind = self.reader.byte()?;
                let rank = self.reader.count()?;
                if kind > 5 || rank == 0 || rank > 32 {
                    return Err(invalid("invalid array dimensions"));
                }
                let mut len = 1usize;
                for _ in 0..rank {
                    len = len.checked_mul(self.reader.count()?).ok_or_else(|| invalid("array size overflow"))?;
                }
                if kind >= 3 {
                    for _ in 0..rank {
                        if self.reader.le_i32()? != 0 {
                            return Err(invalid("nonzero array lower bounds are unsupported"));
                        }
                    }
                }
                let tag = self.reader.byte()?;
                let primitive = self.type_info(tag)?;
                self.array(id, len, primitive, depth)
            }
            8 => {
                let primitive = self.reader.byte()?;
                self.primitive(primitive)
            }
            9 => Ok(Value::Ref(self.reader.le_i32()?)),
            10 => Ok(Value::Null),
            12 => {
                self.reader.le_i32()?;
                self.string()?;
                Ok(Value::Null)
            }
            15..=17 => {
                let id = self.reader.le_i32()?;
                let len = self.reader.count()?;
                let primitive = if tag == 15 { Some(self.reader.byte()?) } else { None };
                self.array(id, len, primitive, depth)
            }
            _ => Err(invalid(format!("unsupported NRBF record {tag}"))),
        }
    }
}
