//! Photoshop contour preset files (`.shc`), the Layer Style dialog's Contour Editor Load/Save.
//!
//! Layout from Adobe's public "Photoshop File Formats Specification" (Additional File Formats ›
//! Contours): a 4-byte type `8BFS`, a `u16` version (1) and a `u32` count of contours. Each
//! contour is a `u32` version (1 or 2), a Unicode name, a `u16` point count and the points as two
//! `u16`s each (vertical first, then horizontal). Version 2 then has one boolean byte per point
//! ("whether the point is continuous"); both versions end with a 4-byte minimum and maximum
//! input range. All values are big-endian.
//!
//! The spec leaves two details open, handled defensively:
//! - The name is read like the spec's other Unicode strings (a `u32` UTF-16 code-unit count that
//!   includes a trailing null, as in `.aco`); a name whose count leaves out the null and is
//!   followed by one is read too. Names are written with the null counted.
//! - Coordinates are read as 0..=255 levels, the scale of the layer-style contour descriptors
//!   (`ShpC` points); the input range is written as 0..=255.

use crate::error::{PsdError, Result};
use crate::io::{Reader, WriteExt};

/// Most contours read from one file.
pub const MAX_CONTOURS: usize = 4096;
/// Most points in one contour (the counts are 16-bit; Photoshop allows far fewer).
pub const MAX_POINTS: usize = 256;
/// Longest name read or written, in UTF-16 code units.
pub const MAX_NAME_UNITS: usize = 1024;

/// One contour point: input (horizontal) and output (vertical), `0..=255`, and whether the curve
/// passes smoothly through it (`false` = a corner).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShcPoint {
    /// Input level, `0..=255`.
    pub input: f32,
    /// Output level, `0..=255`.
    pub output: f32,
    /// Smooth (`true`) or corner (`false`).
    pub continuous: bool,
}

/// One named contour.
#[derive(Debug, Clone, PartialEq)]
pub struct ShcContour {
    /// Preset name.
    pub name: String,
    /// Points, in file order.
    pub points: Vec<ShcPoint>,
}

fn read_name(r: &mut Reader<'_>) -> Result<String> {
    let n = u64::from(r.u32()?);
    r.check_count(n, 2)?;
    let n = usize::try_from(n).map_err(|_| PsdError::LimitExceeded("contour name length"))?;
    let mut units = Vec::with_capacity(n.min(MAX_NAME_UNITS));
    // The last unit read, not the last one kept: past the cap the counted null is read but not kept.
    let mut last = None;
    for _ in 0..n {
        let u = r.u16()?;
        last = Some(u);
        if units.len() < MAX_NAME_UNITS {
            units.push(u);
        }
    }
    // A count that leaves out the terminator: the null follows.
    if last != Some(0) && r.peek_rest().get(..2) == Some(&[0, 0]) {
        r.u16()?;
    }
    while units.last() == Some(&0) {
        units.pop();
    }
    Ok(String::from_utf16_lossy(&units))
}

/// Parse an `.shc` file. Bad signatures, unknown versions, truncation and impossible counts are
/// errors; never panics.
pub fn parse(data: &[u8]) -> Result<Vec<ShcContour>> {
    let mut r = Reader::new(data);
    let sig = r.array::<4>()?;
    if &sig != b"8BFS" {
        return Err(PsdError::InvalidSignature { expected: "8BFS", found: sig });
    }
    let version = r.u16()?;
    if version != 1 {
        return Err(PsdError::UnsupportedVersion(version));
    }
    let count = u64::from(r.u32()?);
    // A contour takes at least 4 (version) + 4 (name) + 2 (points) + 8 (range) bytes.
    r.check_count(count, 18)?;
    let count = usize::try_from(count).map_err(|_| PsdError::LimitExceeded("contour count"))?;
    if count > MAX_CONTOURS {
        return Err(PsdError::LimitExceeded("too many contours in an .shc file"));
    }
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let v = r.u32()?;
        if v != 1 && v != 2 {
            return Err(PsdError::invalid(format!("unknown contour version {v}")));
        }
        let name = read_name(&mut r)?;
        let n = usize::from(r.u16()?);
        r.check_count(n as u64, if v == 2 { 5 } else { 4 })?;
        if n > MAX_POINTS {
            return Err(PsdError::LimitExceeded("too many points in a contour"));
        }
        let mut points = Vec::with_capacity(n);
        for _ in 0..n {
            let output = f32::from(r.u16()?).min(255.0);
            let input = f32::from(r.u16()?).min(255.0);
            points.push(ShcPoint { input, output, continuous: true });
        }
        if v == 2 {
            for p in &mut points {
                p.continuous = r.u8()? != 0;
            }
        }
        // Minimum and maximum input range: not needed to draw the curve.
        r.skip(8)?;
        out.push(ShcContour { name, points });
    }
    Ok(out)
}

/// Write an `.shc` file (every contour as version 2, with its corner flags). Fails when a contour
/// has more than [`MAX_POINTS`] points or there are more than [`MAX_CONTOURS`].
pub fn write(contours: &[ShcContour]) -> Result<Vec<u8>> {
    if contours.len() > MAX_CONTOURS {
        return Err(PsdError::LimitExceeded("too many contours for an .shc file"));
    }
    let mut out = Vec::with_capacity(10 + contours.len() * 64);
    out.put(b"8BFS");
    out.put_u16(1);
    out.put_u32(contours.len() as u32);
    for c in contours {
        let n = u16::try_from(c.points.len()).ok().filter(|n| usize::from(*n) <= MAX_POINTS).ok_or(PsdError::LimitExceeded("too many points in a contour"))?;
        out.put_u32(2);
        // Capped on a character boundary, so a surrogate pair is never split.
        let units = crate::io::utf16_units(&c.name, MAX_NAME_UNITS);
        out.put_u32(units.len() as u32 + 1);
        for u in units {
            out.put_u16(u);
        }
        out.put_u16(0);
        out.put_u16(n);
        let level = |v: f32| if v.is_finite() { v.round().clamp(0.0, 255.0) as u16 } else { 0 };
        for p in &c.points {
            out.put_u16(level(p.output));
            out.put_u16(level(p.input));
        }
        for p in &c.points {
            out.put_u8(u8::from(p.continuous));
        }
        out.put_u32(0);
        out.put_u32(255);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shc_round_trips_and_rejects_hostile_input() {
        let cone = ShcContour {
            name: "Cone".into(),
            points: vec![
                ShcPoint { input: 0.0, output: 0.0, continuous: true },
                ShcPoint { input: 128.0, output: 255.0, continuous: false },
                ShcPoint { input: 255.0, output: 0.0, continuous: true },
            ],
        };
        let bytes = write(std::slice::from_ref(&cone)).unwrap();
        assert_eq!(&bytes[..4], b"8BFS");
        assert_eq!(parse(&bytes).unwrap(), vec![cone.clone()]);

        // Version 1 without corner flags, with a name count that leaves out the null.
        let mut v1 = b"8BFS".to_vec();
        v1.put_u16(1);
        v1.put_u32(1);
        v1.put_u32(1);
        v1.put_u32(2);
        v1.put_u16(u16::from(b'O'));
        v1.put_u16(u16::from(b'K'));
        v1.put_u16(0);
        v1.put_u16(2);
        for v in [0u16, 0, 255, 255] {
            v1.put_u16(v);
        }
        v1.put_u32(0);
        v1.put_u32(255);
        let c = parse(&v1).unwrap();
        assert_eq!(c[0].name, "OK");
        assert!(c[0].points.iter().all(|p| p.continuous) && c[0].points[1].input == 255.0);

        // Truncations, a wrong signature and absurd counts are errors, never panics.
        for cut in 0..bytes.len() {
            assert!(parse(&bytes[..cut]).is_err(), "cut at {cut}");
        }
        assert!(parse(b"8BFX\0\x01\0\0\0\0").is_err());
        assert!(parse(b"8BFS\0\x01\xff\xff\xff\xff").is_err());
        let many = ShcContour { name: String::new(), points: vec![ShcPoint { input: 0.0, output: 0.0, continuous: true }; MAX_POINTS + 1] };
        assert!(write(&[many]).is_err());
    }

    fn pts(n: usize) -> Vec<ShcPoint> {
        (0..n).map(|i| ShcPoint { input: (i * 10) as f32, output: (i * 20) as f32, continuous: i % 2 == 0 }).collect()
    }

    #[test]
    fn shc_name_cap_never_splits_a_surrogate_pair() {
        let name = format!("{}\u{1F600}", "a".repeat(MAX_NAME_UNITS - 1));
        let c = ShcContour { name, points: pts(2) };
        let back = parse(&write(&[c]).unwrap()).unwrap();
        assert_eq!(back[0].name, "a".repeat(MAX_NAME_UNITS - 1));
    }

    #[test]
    fn shc_names_at_and_over_the_cap_keep_the_stream_in_sync() {
        for len in [MAX_NAME_UNITS, MAX_NAME_UNITS + 1, MAX_NAME_UNITS * 3] {
            for first_points in [0, 3] {
                let long = ShcContour { name: "b".repeat(len), points: pts(first_points) };
                let next = ShcContour { name: "Next".into(), points: pts(4) };
                let back = parse(&write(&[long, next.clone()]).unwrap()).unwrap_or_else(|e| panic!("len {len}, points {first_points}: {e}"));
                assert_eq!(back.len(), 2);
                assert_eq!(back[0].name, "b".repeat(len.min(MAX_NAME_UNITS)));
                assert_eq!(back[0].points.len(), first_points);
                assert_eq!(back[1], next, "len {len}, points {first_points}");
            }
        }
    }

    #[test]
    fn shc_name_count_without_null_at_the_cap_still_reads() {
        // A hostile/foreign file: 1024 units counted, no null counted, null follows.
        let mut v = b"8BFS".to_vec();
        v.put_u16(1);
        v.put_u32(1);
        v.put_u32(1);
        v.put_u32(MAX_NAME_UNITS as u32);
        for _ in 0..MAX_NAME_UNITS {
            v.put_u16(u16::from(b'c'));
        }
        v.put_u16(0);
        v.put_u16(1);
        v.put_u16(7);
        v.put_u16(9);
        v.put_u32(0);
        v.put_u32(255);
        let c = parse(&v).unwrap();
        assert_eq!(c[0].name, "c".repeat(MAX_NAME_UNITS));
        assert_eq!((c[0].points[0].input, c[0].points[0].output), (9.0, 7.0));
    }
}
