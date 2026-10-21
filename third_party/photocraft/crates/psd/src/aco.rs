//! Photoshop Color Swatches files (`.aco`), versions 1 and 2.
//!
//! Layout from Adobe's public "Photoshop File Formats Specification" (Color Swatches): a
//! version 1 section (`u16` version = 1, `u16` count, then `count` colour records of a `u16`
//! colour space id and four `u16` values), usually followed by a version 2 section with the
//! same colours plus a name each (a `u32` UTF-16 code-unit count that includes the trailing
//! null, then the big-endian units). Files that start with the version 2 section are read too.
//!
//! Colour encodings (spec, "Color structure"): RGB and HSB are full 16-bit values; CMYK stores
//! 0 for 100 % ink; Lab stores L* × 100 (0..=10000) and signed a*, b* × 100; grayscale stores a
//! gray level 0..=10000 where 0 is black (public notes on the format convert it with
//! `level / 39.0625` to 0..255). Colours are kept at their 16-bit precision in the space they
//! were stored in; any other colour space (colour books, wide CMYK, …) is kept verbatim so a
//! read → write round trip never loses it. Converting colours is the caller's job.

use crate::error::{PsdError, Result};
use crate::io::{Reader, WriteExt};

/// Most swatches in one file (the counts are 16-bit).
pub const MAX_SWATCHES: usize = u16::MAX as usize;
/// Longest name read or written, in UTF-16 code units (Photoshop's own limit is far lower).
pub const MAX_NAME_UNITS: usize = 1024;

/// Colour space ids of the `.aco` colour record.
pub mod space {
    /// RGB.
    pub const RGB: u16 = 0;
    /// HSB.
    pub const HSB: u16 = 1;
    /// CMYK.
    pub const CMYK: u16 = 2;
    /// CIE L*a*b*.
    pub const LAB: u16 = 7;
    /// Grayscale.
    pub const GRAY: u16 = 8;
}

/// A swatch colour as stored, at full 16-bit precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AcoColor {
    /// Red, green, blue, 0..=65535.
    Rgb([u16; 3]),
    /// Hue (0..=65535 for 0..360°), saturation and brightness (0..=65535).
    Hsb([u16; 3]),
    /// Cyan, magenta, yellow, black **ink** amounts, 0..=65535 (65535 = 100 % ink; the file
    /// stores the complement).
    Cmyk([u16; 4]),
    /// L* × 100 (0..=10000), a* × 100 and b* × 100 (−12800..=12700).
    Lab {
        /// Lightness × 100.
        l: u16,
        /// a* × 100.
        a: i16,
        /// b* × 100.
        b: i16,
    },
    /// Gray level 0..=10000 (0 = black, 10000 = white).
    Gray(u16),
    /// Any other colour space, kept verbatim (the four values as stored).
    Other {
        /// Colour space id.
        space: u16,
        /// The four stored values.
        values: [u16; 4],
    },
}

impl AcoColor {
    /// Decode a colour record (space id + four stored values).
    pub fn from_record(space: u16, v: [u16; 4]) -> AcoColor {
        match space {
            space::RGB => AcoColor::Rgb([v[0], v[1], v[2]]),
            space::HSB => AcoColor::Hsb([v[0], v[1], v[2]]),
            space::CMYK => AcoColor::Cmyk(v.map(|x| u16::MAX - x)),
            space::LAB => AcoColor::Lab { l: v[0], a: v[1] as i16, b: v[2] as i16 },
            space::GRAY => AcoColor::Gray(v[0]),
            _ => AcoColor::Other { space, values: v },
        }
    }

    /// The colour record (space id + four stored values) for this colour.
    pub fn to_record(self) -> (u16, [u16; 4]) {
        match self {
            AcoColor::Rgb([r, g, b]) => (space::RGB, [r, g, b, 0]),
            AcoColor::Hsb([h, s, b]) => (space::HSB, [h, s, b, 0]),
            AcoColor::Cmyk(c) => (space::CMYK, c.map(|x| u16::MAX - x)),
            AcoColor::Lab { l, a, b } => (space::LAB, [l, a as u16, b as u16, 0]),
            AcoColor::Gray(g) => (space::GRAY, [g, 0, 0, 0]),
            AcoColor::Other { space, values } => (space, values),
        }
    }
}

/// One swatch: a colour and its name (empty in version 1 only files).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcoSwatch {
    /// Name (from the version 2 section).
    pub name: String,
    /// Colour.
    pub color: AcoColor,
}

/// A parsed `.aco` file.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AcoFile {
    /// Swatches in file order.
    pub swatches: Vec<AcoSwatch>,
    /// Whether the file had a version 2 section (names).
    pub has_names: bool,
}

fn read_record(r: &mut Reader<'_>) -> Result<AcoColor> {
    let space = r.u16()?;
    let v = [r.u16()?, r.u16()?, r.u16()?, r.u16()?];
    Ok(AcoColor::from_record(space, v))
}

fn read_name(r: &mut Reader<'_>) -> Result<String> {
    let n = u64::from(r.u32()?);
    r.check_count(n, 2)?;
    let n = usize::try_from(n).map_err(|_| PsdError::LimitExceeded("swatch name length"))?;
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

fn read_section(r: &mut Reader<'_>, named: bool) -> Result<Vec<AcoSwatch>> {
    let count = r.u16()? as usize;
    // Each record is at least 10 bytes (plus a 4-byte name length in version 2).
    r.check_count(count as u64, if named { 14 } else { 10 })?;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let color = read_record(r)?;
        let name = if named { read_name(r)? } else { String::new() };
        out.push(AcoSwatch { name, color });
    }
    Ok(out)
}

/// Parse an `.aco` file. Truncated sections, unknown versions and impossible counts are errors;
/// never panics.
pub fn parse(data: &[u8]) -> Result<AcoFile> {
    let mut r = Reader::new(data);
    match r.u16()? {
        1 => {
            let v1 = read_section(&mut r, false)?;
            // A version 2 section follows in files written by Photoshop 6 and later; anything
            // else after version 1 (padding some writers add) is ignored.
            if r.remaining() >= 4 && r.peek_rest().get(..2) == Some(&[0, 2]) {
                r.u16()?;
                let v2 = read_section(&mut r, true)?;
                return Ok(AcoFile { swatches: v2, has_names: true });
            }
            Ok(AcoFile { swatches: v1, has_names: false })
        }
        2 => Ok(AcoFile { swatches: read_section(&mut r, true)?, has_names: true }),
        v => Err(PsdError::UnsupportedVersion(v)),
    }
}

/// Write an `.aco` file: a version 1 section followed by a version 2 section with the names,
/// as Photoshop writes them. Fails when there are more than [`MAX_SWATCHES`] swatches.
pub fn write(swatches: &[AcoSwatch]) -> Result<Vec<u8>> {
    let count = u16::try_from(swatches.len()).map_err(|_| PsdError::LimitExceeded("an .aco file holds at most 65535 swatches"))?;
    let mut out = Vec::with_capacity(8 + swatches.len() * 40);
    for version in [1u16, 2] {
        out.put_u16(version);
        out.put_u16(count);
        for s in swatches {
            let (space, v) = s.color.to_record();
            out.put_u16(space);
            for x in v {
                out.put_u16(x);
            }
            if version == 2 {
                let units = crate::io::utf16_units(&s.name, MAX_NAME_UNITS);
                // The length counts the terminating null.
                out.put_u32(units.len() as u32 + 1);
                for u in units {
                    out.put_u16(u);
                }
                out.put_u16(0);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<AcoSwatch> {
        vec![
            AcoSwatch { name: "Pure Red".into(), color: AcoColor::Rgb([65535, 0, 0]) },
            AcoSwatch { name: "Hue".into(), color: AcoColor::Hsb([12000, 65535, 40000]) },
            AcoSwatch { name: "Cyan ink".into(), color: AcoColor::Cmyk([65535, 0, 0, 1234]) },
            AcoSwatch { name: "Lab −".into(), color: AcoColor::Lab { l: 5321, a: -12800, b: 12700 } },
            AcoSwatch { name: "Gray".into(), color: AcoColor::Gray(5000) },
            AcoSwatch { name: "Book 日本".into(), color: AcoColor::Other { space: 3, values: [1, 2, 3, 4] } },
            AcoSwatch { name: String::new(), color: AcoColor::Rgb([1, 2, 3]) },
        ]
    }

    #[test]
    fn round_trips_every_space_at_16_bits() {
        let s = sample();
        let bytes = write(&s).unwrap();
        let f = parse(&bytes).unwrap();
        assert!(f.has_names);
        assert_eq!(f.swatches, s);
        assert_eq!(write(&f.swatches).unwrap(), bytes, "byte-stable");
    }

    #[test]
    fn decodes_the_spec_encodings() {
        // Pure cyan as the spec gives it: 0, 65535, 65535, 65535 (0 = 100 % ink).
        assert_eq!(AcoColor::from_record(2, [0, 65535, 65535, 65535]), AcoColor::Cmyk([65535, 0, 0, 0]));
        // White Lab = 10000, 0, 0; negative chrominance is two's complement.
        assert_eq!(AcoColor::from_record(7, [10000, 0xFF38, 200, 0]), AcoColor::Lab { l: 10000, a: -200, b: 200 });
        assert_eq!(AcoColor::from_record(8, [10000, 0, 0, 0]), AcoColor::Gray(10000));
        assert_eq!(AcoColor::from_record(9, [7, 8, 9, 10]), AcoColor::Other { space: 9, values: [7, 8, 9, 10] });
    }

    #[test]
    fn reads_version_1_only_and_version_2_only() {
        // Version 1: two RGB colours, no names.
        let v1 = [0u8, 1, 0, 2, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 8, 0x13, 0x88, 0, 0, 0, 0, 0, 0];
        let f = parse(&v1).unwrap();
        assert!(!f.has_names);
        assert_eq!(f.swatches.iter().map(|s| s.color).collect::<Vec<_>>(), vec![AcoColor::Rgb([65535, 0, 0]), AcoColor::Gray(5000)]);
        // Trailing padding after a version 1 section is ignored.
        let mut padded = v1.to_vec();
        padded.extend_from_slice(&[0, 0, 0, 0]);
        assert_eq!(parse(&padded).unwrap().swatches.len(), 2);
        // Version 2 only (the v1 section is skipped by some writers); name without its null.
        let mut v2 = vec![0u8, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 3, 0, 0];
        v2.extend_from_slice(&[0, 0, 0, 2, 0, b'O', 0, b'K']);
        let f = parse(&v2).unwrap();
        assert_eq!(f.swatches, vec![AcoSwatch { name: "OK".into(), color: AcoColor::Rgb([1, 2, 3]) }]);
    }

    #[test]
    fn hostile_input_is_an_error_not_a_panic() {
        let good = write(&sample()).unwrap();
        for n in 0..good.len() {
            // Every truncation inside a section fails (a cut exactly after v1 reads v1).
            let r = parse(&good[..n]);
            let v1_end = 4 + sample().len() * 10;
            if n == v1_end || (n > v1_end && n < v1_end + 4) {
                assert!(r.is_ok(), "{n}");
            } else {
                assert!(r.is_err(), "{n}");
            }
        }
        assert!(parse(&[]).is_err());
        assert!(parse(&[0, 3, 0, 0]).is_err(), "unknown version");
        assert!(parse(&[0, 1, 0xFF, 0xFF]).is_err(), "huge count");
        // A name length far beyond the data.
        let mut bad = vec![0u8, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 3, 0, 0];
        bad.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF, 0, 65]);
        assert!(parse(&bad).is_err());
        // Garbage of every length never panics.
        let mut seed = 0x1234_5678u32;
        for len in 0..300 {
            let junk: Vec<u8> = (0..len)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (seed >> 24) as u8
                })
                .collect();
            let _ = parse(&junk);
            let mut v = vec![0, 1];
            v.extend_from_slice(&junk);
            let _ = parse(&v);
        }
    }

    #[test]
    fn too_many_swatches_is_an_error() {
        let many = vec![AcoSwatch { name: String::new(), color: AcoColor::Gray(0) }; MAX_SWATCHES + 1];
        assert!(write(&many).is_err());
    }

    #[test]
    fn long_names_are_capped() {
        let long = AcoSwatch { name: "x".repeat(5000), color: AcoColor::Gray(0) };
        let f = parse(&write(&[long]).unwrap()).unwrap();
        assert_eq!(f.swatches[0].name.len(), MAX_NAME_UNITS);
    }
}
