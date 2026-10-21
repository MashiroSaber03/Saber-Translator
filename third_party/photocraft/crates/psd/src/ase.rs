//! Adobe Swatch Exchange files (`.ase`), shared by Photoshop, Illustrator and InDesign.
//!
//! Layout from public descriptions of the format (big-endian throughout): the signature
//! `ASEF`, version `u16` major (1) and minor (0), a `u32` block count, then blocks of a `u16`
//! type, a `u32` body length and the body:
//!
//! - `0xC001` group start: a name;
//! - `0xC002` group end: empty;
//! - `0x0001` colour entry: a name, a four-character colour model (`RGB `, `CMYK`, `LAB `,
//!   `Gray`), its components as `f32` (3, 4, 3 and 1 of them) and a `u16` colour type
//!   (0 global, 1 spot, 2 normal/process).
//!
//! A name is a `u16` UTF-16 code-unit count that includes the trailing null, then the units.
//! Components are kept exactly as stored: RGB, CMYK and gray in 0..1 (gray 1 = white), Lab as
//! L*/100 followed by a* and b*. Unknown colour models and block types are kept verbatim
//! (colour models) or skipped (block types). Converting colours is the caller's job.

use crate::error::{PsdError, Result};
use crate::io::{Reader, WriteExt};

/// File signature.
pub const SIGNATURE: &[u8; 4] = b"ASEF";
/// Most colour entries read from one file.
pub const MAX_SWATCHES: usize = 100_000;
/// Longest name read or written, in UTF-16 code units.
pub const MAX_NAME_UNITS: usize = 1024;
// `put_name` writes the name length as `units.len() as u16 + 1`, so the cap must leave room in a
// u16; this guards that invariant if the constant is ever raised.
const _: () = assert!(MAX_NAME_UNITS < u16::MAX as usize);

const GROUP_START: u16 = 0xC001;
const GROUP_END: u16 = 0xC002;
const COLOR: u16 = 0x0001;

/// A colour as stored.
#[derive(Debug, Clone, PartialEq)]
pub enum AseColor {
    /// Red, green, blue, 0..1.
    Rgb([f32; 3]),
    /// Cyan, magenta, yellow, black ink, 0..1.
    Cmyk([f32; 4]),
    /// L*/100 (0..1), a*, b*.
    Lab([f32; 3]),
    /// Gray level 0..1 (1 = white).
    Gray(f32),
    /// Any other colour model, kept verbatim: the model tag and the bytes after it, up to the
    /// colour type.
    Other {
        /// Four-character model tag.
        model: [u8; 4],
        /// Component bytes as stored.
        data: Vec<u8>,
    },
}

/// The colour type of an entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AseKind {
    /// A global colour (edits update every use).
    Global,
    /// A spot colour (its own ink).
    Spot,
    /// A normal (process) colour.
    #[default]
    Process,
}

impl AseKind {
    fn from_u16(v: u16) -> AseKind {
        match v {
            0 => AseKind::Global,
            1 => AseKind::Spot,
            _ => AseKind::Process,
        }
    }
    fn to_u16(self) -> u16 {
        match self {
            AseKind::Global => 0,
            AseKind::Spot => 1,
            AseKind::Process => 2,
        }
    }
}

/// One colour entry.
#[derive(Debug, Clone, PartialEq)]
pub struct AseSwatch {
    /// Name.
    pub name: String,
    /// Colour.
    pub color: AseColor,
    /// Colour type.
    pub kind: AseKind,
}

/// A top-level entry: a colour, or a group of colours (the format has one level of groups).
#[derive(Debug, Clone, PartialEq)]
pub enum AseEntry {
    /// A colour outside any group.
    Swatch(AseSwatch),
    /// A named group.
    Group {
        /// Group name.
        name: String,
        /// Its colours.
        swatches: Vec<AseSwatch>,
    },
}

/// A parsed `.ase` file.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AseFile {
    /// Entries in file order.
    pub entries: Vec<AseEntry>,
}

impl AseFile {
    /// Number of colour entries (in and outside groups).
    pub fn swatch_count(&self) -> usize {
        self.entries
            .iter()
            .map(|e| match e {
                AseEntry::Swatch(_) => 1,
                AseEntry::Group { swatches, .. } => swatches.len(),
            })
            .sum()
    }
}

fn f32_be(r: &mut Reader<'_>) -> Result<f32> {
    let v = f32::from_bits(r.u32()?);
    // Non-finite components would poison every conversion downstream.
    Ok(if v.is_finite() { v } else { 0.0 })
}

fn read_name(r: &mut Reader<'_>) -> Result<String> {
    if r.is_empty() {
        return Ok(String::new());
    }
    let n = r.u16()? as usize;
    r.check_count(n as u64, 2)?;
    let mut units = Vec::with_capacity(n.min(MAX_NAME_UNITS));
    for _ in 0..n {
        let u = r.u16()?;
        if units.len() < MAX_NAME_UNITS {
            units.push(u);
        }
    }
    while units.last() == Some(&0) {
        units.pop();
    }
    Ok(String::from_utf16_lossy(&units))
}

fn read_color(body: &mut Reader<'_>) -> Result<AseSwatch> {
    let name = read_name(body)?;
    let model: [u8; 4] = body.array()?;
    let color = match &model {
        b"RGB " => AseColor::Rgb([f32_be(body)?, f32_be(body)?, f32_be(body)?]),
        b"CMYK" => AseColor::Cmyk([f32_be(body)?, f32_be(body)?, f32_be(body)?, f32_be(body)?]),
        b"LAB " => AseColor::Lab([f32_be(body)?, f32_be(body)?, f32_be(body)?]),
        b"Gray" => AseColor::Gray(f32_be(body)?),
        _ => {
            // Unknown model: everything but the trailing colour type is its data.
            let n = body.remaining().saturating_sub(2);
            AseColor::Other { model, data: body.bytes(n)?.to_vec() }
        }
    };
    // Some writers omit the colour type; it defaults to process.
    let kind = if body.remaining() >= 2 { AseKind::from_u16(body.u16()?) } else { AseKind::Process };
    Ok(AseSwatch { name, color, kind })
}

/// Parse an `.ase` file. A wrong signature or version, a truncated block, a block count beyond
/// the data or more than [`MAX_SWATCHES`] colours is an error; never panics. Unknown block
/// types are skipped, a group end without a start is ignored, and a group left open at the end
/// (or a group started inside another) is closed.
pub fn parse(data: &[u8]) -> Result<AseFile> {
    let mut r = Reader::new(data);
    let sig: [u8; 4] = r.array()?;
    if &sig != SIGNATURE {
        return Err(PsdError::InvalidSignature { expected: "ASEF", found: sig });
    }
    let major = r.u16()?;
    let _minor = r.u16()?;
    if major != 1 {
        return Err(PsdError::UnsupportedVersion(major));
    }
    let count = u64::from(r.u32()?);
    // Every block has at least a type and a length.
    r.check_count(count, 6)?;
    let mut entries = Vec::new();
    let mut open: Option<(String, Vec<AseSwatch>)> = None;
    let mut total = 0usize;
    for _ in 0..count {
        let kind = r.u16()?;
        let len = u64::from(r.u32()?);
        let mut body = r.sub(len)?;
        match kind {
            GROUP_START => {
                if let Some((name, swatches)) = open.take() {
                    entries.push(AseEntry::Group { name, swatches });
                }
                open = Some((read_name(&mut body)?, Vec::new()));
            }
            GROUP_END => {
                if let Some((name, swatches)) = open.take() {
                    entries.push(AseEntry::Group { name, swatches });
                }
            }
            COLOR => {
                total += 1;
                if total > MAX_SWATCHES {
                    return Err(PsdError::LimitExceeded("too many swatches in an .ase file"));
                }
                let s = read_color(&mut body)?;
                match open.as_mut() {
                    Some((_, v)) => v.push(s),
                    None => entries.push(AseEntry::Swatch(s)),
                }
            }
            _ => {}
        }
    }
    if let Some((name, swatches)) = open.take() {
        entries.push(AseEntry::Group { name, swatches });
    }
    Ok(AseFile { entries })
}

fn put_name(out: &mut Vec<u8>, name: &str) {
    let units = crate::io::utf16_units(name, MAX_NAME_UNITS);
    out.put_u16(units.len() as u16 + 1);
    for u in units {
        out.put_u16(u);
    }
    out.put_u16(0);
}

fn put_block(out: &mut Vec<u8>, kind: u16, body: &[u8]) -> Result<()> {
    out.put_u16(kind);
    out.put_u32(u32::try_from(body.len()).map_err(|_| PsdError::LimitExceeded("swatch block too large"))?);
    out.put(body);
    Ok(())
}

fn put_color(out: &mut Vec<u8>, s: &AseSwatch) -> Result<()> {
    let mut b = Vec::with_capacity(64);
    put_name(&mut b, &s.name);
    let floats = |b: &mut Vec<u8>, v: &[f32]| {
        for x in v {
            b.put_u32(x.to_bits());
        }
    };
    match &s.color {
        AseColor::Rgb(c) => {
            b.put(b"RGB ");
            floats(&mut b, c);
        }
        AseColor::Cmyk(c) => {
            b.put(b"CMYK");
            floats(&mut b, c);
        }
        AseColor::Lab(c) => {
            b.put(b"LAB ");
            floats(&mut b, c);
        }
        AseColor::Gray(g) => {
            b.put(b"Gray");
            floats(&mut b, &[*g]);
        }
        AseColor::Other { model, data } => {
            b.put(model);
            b.put(data);
        }
    }
    b.put_u16(s.kind.to_u16());
    put_block(out, COLOR, &b)
}

/// Write an `.ase` file (version 1.0). Fails only when a block or the block count overflows the
/// format's 32-bit fields or there are more than [`MAX_SWATCHES`] colours.
pub fn write(file: &AseFile) -> Result<Vec<u8>> {
    if file.swatch_count() > MAX_SWATCHES {
        return Err(PsdError::LimitExceeded("too many swatches for an .ase file"));
    }
    let mut body = Vec::with_capacity(64 * file.swatch_count().max(1));
    let mut blocks = 0u64;
    for e in &file.entries {
        match e {
            AseEntry::Swatch(s) => {
                put_color(&mut body, s)?;
                blocks += 1;
            }
            AseEntry::Group { name, swatches } => {
                let mut b = Vec::with_capacity(name.len() * 2 + 4);
                put_name(&mut b, name);
                put_block(&mut body, GROUP_START, &b)?;
                for s in swatches {
                    put_color(&mut body, s)?;
                }
                put_block(&mut body, GROUP_END, &[])?;
                blocks += 2 + swatches.len() as u64;
            }
        }
    }
    let mut out = Vec::with_capacity(12 + body.len());
    out.put(SIGNATURE);
    out.put_u16(1);
    out.put_u16(0);
    out.put_u32(u32::try_from(blocks).map_err(|_| PsdError::LimitExceeded("too many blocks"))?);
    out.put(&body);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sw(name: &str, color: AseColor, kind: AseKind) -> AseSwatch {
        AseSwatch { name: name.into(), color, kind }
    }

    fn sample() -> AseFile {
        AseFile {
            entries: vec![
                AseEntry::Swatch(sw("Loose red", AseColor::Rgb([1.0, 0.0, 0.0]), AseKind::Process)),
                AseEntry::Group {
                    name: "Brand ✓".into(),
                    swatches: vec![
                        sw("Ink", AseColor::Cmyk([0.1, 0.2, 0.3, 0.4]), AseKind::Spot),
                        sw("Lab", AseColor::Lab([0.5321, -12.25, 40.5]), AseKind::Global),
                        sw("Mid gray", AseColor::Gray(0.5), AseKind::Process),
                        sw("Odd", AseColor::Other { model: *b"HSV ", data: vec![1, 2, 3, 4, 5, 6] }, AseKind::Process),
                    ],
                },
                AseEntry::Group { name: String::new(), swatches: vec![] },
            ],
        }
    }

    #[test]
    fn round_trips_groups_models_and_types() {
        let f = sample();
        let bytes = write(&f).unwrap();
        assert_eq!(&bytes[..4], b"ASEF");
        // 1 loose + (start + 4 + end) + (start + end).
        assert_eq!(u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]), 9);
        let back = parse(&bytes).unwrap();
        assert_eq!(back, f);
        assert_eq!(write(&back).unwrap(), bytes, "byte-stable");
        assert_eq!(back.swatch_count(), 5);
    }

    #[test]
    fn lenient_structure() {
        // A colour without a type, an unknown block, an end without start, an unclosed group.
        let mut body = Vec::new();
        let mut c = Vec::new();
        put_name(&mut c, "A");
        c.put(b"Gray");
        c.put_u32(0.25f32.to_bits());
        put_block(&mut body, GROUP_END, &[]).unwrap();
        put_block(&mut body, 0x7777, &[9, 9, 9]).unwrap();
        let mut g = Vec::new();
        put_name(&mut g, "G");
        put_block(&mut body, GROUP_START, &g).unwrap();
        put_block(&mut body, COLOR, &c).unwrap();
        let mut f = b"ASEF\0\x01\0\0\0\0\0\x04".to_vec();
        f.extend_from_slice(&body);
        let p = parse(&f).unwrap();
        assert_eq!(p.entries, vec![AseEntry::Group { name: "G".into(), swatches: vec![sw("A", AseColor::Gray(0.25), AseKind::Process)] }]);
    }

    #[test]
    fn non_finite_components_become_zero() {
        let f = AseFile { entries: vec![AseEntry::Swatch(sw("n", AseColor::Rgb([f32::NAN, f32::INFINITY, 0.5]), AseKind::Process))] };
        let p = parse(&write(&f).unwrap()).unwrap();
        assert_eq!(p.entries, vec![AseEntry::Swatch(sw("n", AseColor::Rgb([0.0, 0.0, 0.5]), AseKind::Process))]);
    }

    #[test]
    fn hostile_input_is_an_error_not_a_panic() {
        let good = write(&sample()).unwrap();
        for n in 0..good.len() {
            assert!(parse(&good[..n]).is_err(), "truncated at {n}");
        }
        assert!(parse(b"ASEX\0\x01\0\0\0\0\0\0").is_err(), "signature");
        assert!(parse(b"ASEF\0\x02\0\0\0\0\0\0").is_err(), "version");
        assert!(parse(b"ASEF\0\x01\0\0\xff\xff\xff\xff").is_err(), "huge block count");
        // A block length beyond the data.
        assert!(parse(b"ASEF\0\x01\0\0\0\0\0\x01\0\x01\xff\xff\xff\xff").is_err());
        // A name length beyond the block.
        assert!(parse(b"ASEF\0\x01\0\0\0\0\0\x01\0\x01\0\0\0\x04\xff\xff\0A").is_err());
        // A colour block too short for its components.
        assert!(parse(b"ASEF\0\x01\0\0\0\0\0\x01\0\x01\0\0\0\x0a\0\x01\0\0RGB \0\0").is_err());
        let mut seed = 0xDEAD_BEEFu32;
        for len in 0..400 {
            let junk: Vec<u8> = (0..len)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (seed >> 24) as u8
                })
                .collect();
            let mut v = b"ASEF\0\x01\0\0\0\0\0\x10".to_vec();
            v.extend_from_slice(&junk);
            let _ = parse(&v);
            let _ = parse(&junk);
        }
    }

    #[test]
    fn too_many_swatches() {
        let s = sw("", AseColor::Gray(0.0), AseKind::Process);
        let f = AseFile { entries: vec![AseEntry::Group { name: "g".into(), swatches: vec![s; MAX_SWATCHES + 1] }] };
        assert!(write(&f).is_err());
    }
}
