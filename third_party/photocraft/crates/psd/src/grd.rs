//! Photoshop gradient files (`.grd`, version 5: Photoshop 6 and later).
//!
//! Layout from public descriptions of the format: `8BGR`, version u16 (5), then a versioned
//! ActionDescriptor whose `GrdL` list holds `Grdn` objects, each with a `Grad` object:
//! `Nm  ` name, `GrdF` form (`CstS` custom stops or `ClNs` noise), `Clrs` colour stops
//! (`Clr ` colour, `Type` user/foreground/background, `Lctn` 0..4096, `Mdpn` midpoint %) and
//! `Trns` transparency stops (`Opct` %, `Lctn`, `Mdpn`). Older (version 3) files are rejected.
//! Colours are returned in the model they were stored in; converting them is the caller's job.

use crate::descriptor::{Descriptor, Value, VersionedDescriptor};
use crate::error::{PsdError, Result};

/// Most gradients read from one file.
pub const MAX_GRADIENTS: usize = 10_000;
/// Most stops read per gradient.
pub const MAX_STOPS: usize = 1_000;

/// A stop colour as stored.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GrdColor {
    /// 0..1 RGB (from 0..255 components).
    Rgb([f32; 3]),
    /// Hue degrees, saturation and brightness 0..1.
    Hsb([f32; 3]),
    /// 0..1 C, M, Y, K ink.
    Cmyk([f32; 4]),
    /// L 0..100, a, b.
    Lab([f32; 3]),
    /// 0..1 ink (0 = white).
    Gray(f32),
    /// The live foreground colour.
    Foreground,
    /// The live background colour.
    Background,
}

/// One colour stop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrdStop {
    /// 0..1.
    pub location: f32,
    /// 0..1 (0.5 = centred).
    pub midpoint: f32,
    /// Colour.
    pub color: GrdColor,
}

/// One gradient.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GrdGradient {
    /// Name.
    pub name: String,
    /// Noise gradient (`ClNs`): stops are absent; callers usually skip or approximate it.
    pub noise: bool,
    /// Colour stops, in file order.
    pub stops: Vec<GrdStop>,
    /// Transparency stops: (location 0..1, opacity 0..1, midpoint 0..1).
    pub opacity: Vec<(f32, f32, f32)>,
}

/// The raw number behind a key. It is deliberately *not* filtered: every caller passes the result
/// through `finite`, so filtering here as well would give two guards with two different answers to
/// the same input, and mutation testing showed the second one was unreachable.
fn num(d: &Descriptor, k: &str) -> Option<f64> {
    match d.get(k)? {
        Value::Double(v) => Some(*v),
        Value::UnitFloat { value, .. } => Some(*value),
        Value::Integer(v) => Some(f64::from(*v)),
        Value::LargeInteger(v) => Some(*v as f64),
        _ => None,
    }
}
/// The single guard every number passes through on its way to an `f32`. Only a NaN is truly
/// without a value, so only a NaN returns `None`; the caller then falls back to the default its
/// own field documents (opacity opaque, a midpoint centred, a channel zero).
///
/// Everything else — including `±inf` and a value finite as `f64` but larger than `f32::MAX`
/// (`1e300`, which would otherwise overflow the cast to an infinity) — is *saturated* to the
/// representable range before the cast. That keeps the mapping monotonic: a bigger input never
/// yields a smaller output, so "too large" lands at the top of the range rather than snapping to a
/// field default and inverting the result.
fn finite(v: f64) -> Option<f32> {
    if v.is_nan() {
        return None;
    }
    Some(v.clamp(f64::from(f32::MIN), f64::from(f32::MAX)) as f32)
}
fn f(d: &Descriptor, k: &str) -> f32 {
    finite(num(d, k).unwrap_or(0.0)).unwrap_or(0.0)
}
fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}
/// A hue in degrees. An in-range value (0..=360, including exactly 360) is kept verbatim; only an
/// out-of-range value wraps with `rem_euclid`, so 370 -> 10 and -10 -> 350. Wrapping every value
/// would alter a legal 360 to 0 (and f32 round-off on a tiny negative can itself land on 360),
/// changing the colour at the boundary. The input is already finite, so the result stays finite.
fn hue(h: f32) -> f32 {
    if (0.0..=360.0).contains(&h) { h } else { h.rem_euclid(360.0) }
}

fn obj<'a>(d: &'a Descriptor, k: &str) -> Option<&'a Descriptor> {
    match d.get(k)? {
        Value::Descriptor(o) | Value::GlobalObject(o) => Some(o),
        _ => None,
    }
}
fn list<'a>(d: &'a Descriptor, k: &str) -> &'a [Value] {
    match d.get(k) {
        Some(Value::List(v)) => v,
        _ => &[],
    }
}
fn enum_is(d: &Descriptor, k: &str, v: &str) -> bool {
    matches!(d.get(k), Some(Value::Enumerated { value, .. }) if value.is(v))
}

fn color(c: &Descriptor) -> Option<GrdColor> {
    let id = c.class_id.as_bytes();
    Some(match id {
        // Every component documented as 0..=1 is clamped here, like every other range in this
        // module: a hostile file can store any integer at all, and dividing by 100 or 255 does not
        // bring it back into range on its own.
        b"RGBC" => GrdColor::Rgb([f(c, "Rd  ") / 255.0, f(c, "Grn ") / 255.0, f(c, "Bl  ") / 255.0].map(clamp01)),
        // Hue is a circular degree value in 0..=360; see `hue`. An in-range value (including
        // exactly 360) is kept, only an out-of-range one wraps, so a hostile hue cannot become a
        // different, wrong colour at the boundary.
        b"HSBC" => GrdColor::Hsb([hue(f(c, "H   ")), (f(c, "Strt") / 100.0).clamp(0.0, 1.0), (f(c, "Brgh") / 100.0).clamp(0.0, 1.0)]),
        b"CMYC" => GrdColor::Cmyk([f(c, "Cyn ") / 100.0, f(c, "Mgnt") / 100.0, f(c, "Ylw ") / 100.0, f(c, "Blck") / 100.0].map(clamp01)),
        // L* is 0..=100; a*/b* are signed and bounded to the CIELAB gamut (-128..=127) so a hostile
        // value cannot hand a downstream colour converter an out-of-gamut number to overflow on.
        b"LbCl" => GrdColor::Lab([f(c, "Lmnc").clamp(0.0, 100.0), f(c, "A   ").clamp(-128.0, 127.0), f(c, "B   ").clamp(-128.0, 127.0)]),
        b"Grsc" => GrdColor::Gray(clamp01(f(c, "Gry ") / 100.0)),
        _ => return None,
    })
}

fn location(d: &Descriptor) -> f32 {
    (f(d, "Lctn") / 4096.0).clamp(0.0, 1.0)
}
fn midpoint(d: &Descriptor) -> f32 {
    finite(num(d, "Mdpn").unwrap_or(50.0) / 100.0).unwrap_or(0.5).clamp(0.0, 1.0)
}

fn gradient(g: &Descriptor) -> GrdGradient {
    let name = match g.get("Nm  ") {
        Some(Value::Text(t)) => t.to_string_lossy(),
        _ => String::new(),
    };
    let mut out = GrdGradient { name, noise: enum_is(g, "GrdF", "ClNs"), ..Default::default() };
    for v in list(g, "Clrs").iter().take(MAX_STOPS) {
        let Value::Descriptor(s) = v else { continue };
        let c = if enum_is(s, "Type", "FrgC") {
            Some(GrdColor::Foreground)
        } else if enum_is(s, "Type", "BckC") {
            Some(GrdColor::Background)
        } else {
            obj(s, "Clr ").and_then(color)
        };
        if let Some(color) = c {
            out.stops.push(GrdStop { location: location(s), midpoint: midpoint(s), color });
        }
    }
    for v in list(g, "Trns").iter().take(MAX_STOPS) {
        let Value::Descriptor(s) = v else { continue };
        out.opacity.push((location(s), finite(num(s, "Opct").unwrap_or(100.0) / 100.0).unwrap_or(1.0).clamp(0.0, 1.0), midpoint(s)));
    }
    out
}

/// Parses a `.grd` file.
pub fn parse(data: &[u8]) -> Result<Vec<GrdGradient>> {
    let sig = data.get(0..4).ok_or(PsdError::UnexpectedEof { offset: 0, needed: 4 })?;
    if sig != b"8BGR" {
        let mut found = [0u8; 4];
        found.copy_from_slice(sig);
        return Err(PsdError::InvalidSignature { expected: "8BGR", found });
    }
    let version = data.get(4..6).map(|b| u16::from_be_bytes([b[0], b[1]])).ok_or(PsdError::UnexpectedEof { offset: 4, needed: 2 })?;
    if version != 5 {
        return Err(PsdError::Unsupported(format!("gradient file version {version} (only version 5, Photoshop 6 and later)")));
    }
    let (vd, _) = VersionedDescriptor::parse_prefix(data.get(6..).unwrap_or_default())?;
    let out: Vec<GrdGradient> = list(&vd.descriptor, "GrdL")
        .iter()
        .take(MAX_GRADIENTS)
        .filter_map(|v| match v {
            Value::Descriptor(d) => obj(d, "Grad").map(gradient),
            _ => None,
        })
        .collect();
    if out.is_empty() {
        return Err(PsdError::invalid("the gradient file holds no gradients"));
    }
    Ok(out)
}

/// Writes a version 5 `.grd` file (RGB / foreground / background stops; used by tests).
pub fn write(gradients: &[GrdGradient]) -> Vec<u8> {
    use crate::descriptor::{Id, UnicodeString};
    let en = |t: &str, v: &str| Value::Enumerated { type_id: Id::new(t), value: Id::new(v) };
    let items = gradients
        .iter()
        .map(|g| {
            let stops = g
                .stops
                .iter()
                .map(|s| {
                    let (ty, c) = match s.color {
                        GrdColor::Foreground => ("FrgC", None),
                        GrdColor::Background => ("BckC", None),
                        GrdColor::Rgb([r, gg, b]) => ("UsrS", Some((r, gg, b))),
                        _ => ("UsrS", Some((0.0, 0.0, 0.0))),
                    };
                    let mut d = Descriptor::new("Clrt");
                    if let Some((r, gg, b)) = c {
                        let rgb = Descriptor::new("RGBC")
                            .with("Rd  ", Value::Double(f64::from(r) * 255.0))
                            .with("Grn ", Value::Double(f64::from(gg) * 255.0))
                            .with("Bl  ", Value::Double(f64::from(b) * 255.0));
                        d = d.with("Clr ", Value::Descriptor(rgb));
                    }
                    Value::Descriptor(
                        d.with("Type", en("Clry", ty))
                            .with("Lctn", Value::Integer((s.location * 4096.0).round() as i32))
                            .with("Mdpn", Value::Integer((s.midpoint * 100.0).round() as i32)),
                    )
                })
                .collect();
            let trns = g
                .opacity
                .iter()
                .map(|&(l, o, m)| {
                    Value::Descriptor(
                        Descriptor::new("TrnS")
                            .with("Opct", Value::UnitFloat { unit: *b"#Prc", value: f64::from(o) * 100.0 })
                            .with("Lctn", Value::Integer((l * 4096.0).round() as i32))
                            .with("Mdpn", Value::Integer((m * 100.0).round() as i32)),
                    )
                })
                .collect();
            let grad = Descriptor::new("Grdn")
                .with("Nm  ", Value::Text(UnicodeString::new_nul(&g.name)))
                .with("GrdF", en("GrdF", if g.noise { "ClNs" } else { "CstS" }))
                .with("Intr", Value::Double(4096.0))
                .with("Clrs", Value::List(stops))
                .with("Trns", Value::List(trns));
            Value::Descriptor(Descriptor::new("Grdn").with("Grad", Value::Descriptor(grad)))
        })
        .collect();
    let mut out = b"8BGR".to_vec();
    out.extend_from_slice(&5u16.to_be_bytes());
    out.extend_from_slice(&VersionedDescriptor::new(Descriptor::new("null").with("GrdL", Value::List(items))).to_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<GrdGradient> {
        vec![
            GrdGradient {
                name: "Sunset ✓".into(),
                noise: false,
                stops: vec![
                    GrdStop { location: 0.0, midpoint: 0.5, color: GrdColor::Foreground },
                    GrdStop { location: 0.25, midpoint: 0.3, color: GrdColor::Rgb([1.0, 0.5, 0.0]) },
                    GrdStop { location: 1.0, midpoint: 0.5, color: GrdColor::Background },
                ],
                opacity: vec![(0.0, 1.0, 0.5), (1.0, 0.25, 0.5)],
            },
            GrdGradient { name: "Noise".into(), noise: true, ..Default::default() },
        ]
    }

    #[test]
    fn round_trip() {
        let b = write(&sample());
        assert_eq!(parse(&b).unwrap(), sample());
    }

    #[test]
    fn hostile_input_errors() {
        assert!(parse(b"").is_err());
        assert!(parse(b"8BGR\0\x03").is_err());
        assert!(parse(b"8BPS\0\x05").is_err());
        let b = write(&sample());
        for cut in 0..b.len() {
            let _ = parse(&b[..cut]);
        }
        for i in 0..b.len() {
            let mut c = b.clone();
            c[i] ^= 0x5a;
            let _ = parse(&c);
        }
        assert!(parse(&write(&[])).is_err());
    }
}
