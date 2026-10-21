//! Fills, strokes and colours. Two schemas exist: Affinity 1 (`BFil` fill, `PFil` stroke paint,
//! `LSty` stroke style) and Affinity 2/3 (`BFFl`, `LIFl` and `LILn` lists with a current index).

use crate::model::{Affine, Reader};
use crate::stream::{ObjId, Value};

/// A colour as stored: RGB and grey values are encoded in the document's RGB space (sRGB unless
/// the document embeds another profile), CMYK values go through its CMYK profile. Alpha is straight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Color {
    Rgb {
        r: f64,
        g: f64,
        b: f64,
        a: f64,
    },
    Cmyk {
        c: f64,
        m: f64,
        y: f64,
        k: f64,
        a: f64,
    },
    Gray {
        v: f64,
        a: f64,
    },
    /// CIE L*a*b* (D50).
    Lab {
        l: f64,
        a: f64,
        b: f64,
        alpha: f64,
    },
}

impl Color {
    pub fn alpha(&self) -> f64 {
        match self {
            Color::Rgb { a, .. } | Color::Cmyk { a, .. } | Color::Gray { a, .. } => *a,
            Color::Lab { alpha, .. } => *alpha,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientKind {
    Linear,
    /// Elliptical and radial: distance from the centre in unit space.
    Radial,
    /// Angle around the centre.
    Conical,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stop {
    pub offset: f64,
    pub color: Color,
    /// Affinity's midpoint bias for the segment from this stop to the next (0.5 = even blend).
    pub bias: f64,
}

impl Stop {
    /// Where, between this stop (0) and the next (1), the blend reaches half way: the bias curve
    /// fitted to Affinity's renders is `t^k` (bias ≥ 0.5) or `1-(1-t)^k`, with `k = 1 + 8·|bias-0.5|`.
    pub fn half_point(&self) -> f64 {
        let k = 1.0 + 8.0 * (self.bias - 0.5).abs();
        let h = 0.5f64.powf(1.0 / k);
        if self.bias >= 0.5 { h } else { 1.0 - h }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gradient {
    pub kind: GradientKind,
    pub stops: Vec<Stop>,
    /// Unit space to document pixels: linear runs from unit (0,0) to (1,0); radial is the unit
    /// circle; conical starts along unit +x and turns towards unit +y.
    pub transform: Affine,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum Paint {
    #[default]
    None,
    Solid(Color),
    Gradient(Gradient),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cap {
    Butt,
    Round,
    Square,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Join {
    Miter,
    Round,
    Bevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Align {
    Center,
    Inside,
    Outside,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stroke {
    pub paint: Paint,
    /// Width in document pixels.
    pub width: f64,
    pub cap: Cap,
    pub join: Join,
    pub miter_limit: f64,
    pub align: Align,
    /// Dash and gap lengths in document pixels, and the phase.
    pub dash: Option<(Vec<f64>, f64)>,
    /// Painted below the fill instead of above it.
    pub behind: bool,
}

/// Gradient stops per gradient.
const MAX_STOPS: usize = 1024;

fn floats(v: &Value) -> Option<Vec<f64>> {
    match v {
        Value::Floats(f) => Some(f.clone()),
        Value::Bytes(b) if b.len() % 4 == 0 => Some(b.as_chunks::<4>().0.iter().map(|c| f64::from(f32::from_le_bytes(*c))).collect()),
        _ => None,
    }
}

/// A colour object (`RGBA`, `HSLA`, `CMYK`, `GRAY`, `LABA`, `Pant`).
pub(crate) fn color(r: &mut Reader, id: ObjId) -> Option<Color> {
    let s = r.s;
    let class = s.class(id)?;
    let col = || -> Option<Vec<f64>> {
        let v = floats(s.field(id, b"_col")?)?;
        v.iter().all(|f| f.is_finite()).then_some(v)
    };
    Some(match &class.0.to_be_bytes() {
        b"RGBA" => {
            let v = col()?;
            Color::Rgb { r: *v.first()?, g: *v.get(1)?, b: *v.get(2)?, a: v.get(3)?.clamp(0.0, 1.0) }
        }
        b"HSLA" => {
            let v = col()?;
            let (r2, g, b) = hsl(*v.first()?, *v.get(1)?, *v.get(2)?);
            Color::Rgb { r: r2, g, b, a: v.get(3)?.clamp(0.0, 1.0) }
        }
        b"CMYK" => {
            let v = col()?;
            Color::Cmyk { c: *v.first()?, m: *v.get(1)?, y: *v.get(2)?, k: *v.get(3)?, a: v.get(4)?.clamp(0.0, 1.0) }
        }
        b"GRAY" => {
            let v = col()?;
            Color::Gray { v: *v.first()?, a: v.get(1)?.clamp(0.0, 1.0) }
        }
        b"LABA" => {
            let b = s.bytes(id, b"_col")?;
            let u = |i: usize| -> Option<f64> { Some(f64::from(u16::from_le_bytes([*b.get(2 * i)?, *b.get(2 * i + 1)?])) / 65535.0) };
            Color::Lab { l: 100.0 * u(0)?, a: 255.0 * u(1)? - 128.0, b: 255.0 * u(2)? - 128.0, alpha: u(3)? }
        }
        b"Pant" => {
            r.warn("spot colours (imported as their RGB equivalent)");
            return s.obj(id, b"srgb").and_then(|c| color(r, c)).or_else(|| s.obj(id, b"base").and_then(|c| color(r, c)));
        }
        b"RegC" => Color::Cmyk { c: 1.0, m: 1.0, y: 1.0, k: 1.0, a: 1.0 },
        _ => {
            r.warn("colours in an unknown model");
            return None;
        }
    })
}

fn hsl(h: f64, s: f64, l: f64) -> (f64, f64, f64) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let hp = (h.rem_euclid(1.0)) * 6.0;
    let x = c * (1.0 - (hp % 2.0 - 1.0).abs());
    let (r, g, b) = match hp as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = l - c / 2.0;
    (r + m, g + m, b + m)
}

/// A fill descriptor (`FDsc`) or, in Affinity 1 files, a bare `FilS`; `local` maps the node's
/// space to document pixels.
pub(crate) fn descriptor(r: &mut Reader, id: ObjId, local: Affine) -> Option<Paint> {
    let s = r.s;
    let (fill, unit) = if s.is(id, b"FDsc") {
        let unit = s.floats::<6>(id, b"FDeX").map(Affine::from_xfrm).unwrap_or(Affine::IDENTITY);
        (s.obj(id, b"FDeF")?, unit)
    } else {
        (id, Affine::IDENTITY)
    };
    match &s.class(fill)?.0.to_be_bytes() {
        b"FilN" => Some(Paint::None),
        b"FilS" => Some(Paint::Solid(color(r, s.obj(fill, b"Colr")?)?)),
        b"FilG" => {
            let kind = match s.enumeration(fill, b"Type").map(|(i, _)| i) {
                Some(0) | None => GradientKind::Linear,
                Some(1 | 2) => GradientKind::Radial,
                Some(3) => GradientKind::Conical,
                Some(_) => {
                    r.warn("gradients of an unknown kind (imported as linear)");
                    GradientKind::Linear
                }
            };
            let grad = s.obj(fill, b"Grad")?;
            let cols = s.objs(grad, b"Cols");
            let posn: Vec<Vec<f64>> = match s.field(grad, b"Posn") {
                Some(Value::Array(a)) => a.iter().filter_map(floats).collect(),
                _ => Vec::new(),
            };
            let mut stops = Vec::new();
            for (c, p) in cols.iter().zip(&posn).take(MAX_STOPS) {
                let color = color(r, *c)?;
                let offset = p.first().copied().filter(|v| v.is_finite()).unwrap_or(0.0).clamp(0.0, 1.0);
                let bias = p.get(1).copied().filter(|v| v.is_finite()).unwrap_or(0.5).clamp(0.0, 1.0);
                stops.push(Stop { offset, color, bias });
            }
            if stops.is_empty() {
                return Some(Paint::None);
            }
            Some(Paint::Gradient(Gradient { kind, stops, transform: unit.then(local) }))
        }
        b"FilB" => {
            r.warn("bitmap and pattern fills (left empty)");
            Some(Paint::None)
        }
        _ => {
            r.warn("fills of an unknown kind (left empty)");
            Some(Paint::None)
        }
    }
}

fn current(r: &Reader, id: ObjId, list: &[u8; 4], index: &[u8; 4]) -> Option<ObjId> {
    let items = r.s.objs(id, list);
    let i = r.s.int(id, index).and_then(|i| usize::try_from(i).ok()).unwrap_or(0);
    items.get(i).or_else(|| items.first()).copied()
}

/// The fills of `list` (artboard backgrounds use `BFFl`).
pub(crate) fn fills(r: &mut Reader, id: ObjId, local: Affine, list: &[u8; 4]) -> Vec<crate::paint::Paint> {
    let d = current(r, id, list, b"BFCr").or_else(|| r.s.obj(id, b"BFil"));
    d.and_then(|d| descriptor(r, d, local)).filter(|p| *p != Paint::None).into_iter().collect()
}

/// A node's fill and stroke. `local` maps the node's space to document pixels.
pub(crate) fn node_paint(r: &mut Reader, id: ObjId, local: Affine) -> (Vec<Paint>, Vec<Stroke>) {
    let s = r.s;
    let fills = fills(r, id, local, b"BFFl");
    if r.s.objs(id, b"BFFl").len() > 1 || r.s.objs(id, b"LIFl").len() > 1 {
        r.warn("several fills or strokes on one object (only the active one is imported)");
    }
    if s.f64(id, b"FOpc").is_some_and(|o| o < 0.999) {
        r.warn("fill opacity (imported at full fill opacity)");
    }
    if s.obj(id, b"Trns").is_some() {
        r.warn("transparency gradients (imported without them)");
    }
    let (pen, style) = if s.field(id, b"LIFl").is_some() {
        (current(r, id, b"LIFl", b"LICr"), current(r, id, b"LILn", b"LICr"))
    } else {
        (s.obj(id, b"PFil"), s.obj(id, b"LSty"))
    };
    let behind = matches!(s.field(id, b"DrwO"), Some(Value::Array(a)) if a.first() == Some(&Value::Enum { id: 1, version: 0 }));
    let stroke = match (pen, style) {
        (Some(pen), Some(style)) => stroke(r, pen, style, local, behind),
        _ => None,
    };
    (fills, stroke.into_iter().collect())
}

fn stroke(r: &mut Reader, pen: ObjId, ldsc: ObjId, local: Affine, behind: bool) -> Option<Stroke> {
    let s = r.s;
    let paint = descriptor(r, pen, local)?;
    if paint == Paint::None {
        return None;
    }
    let lsty = if s.is(ldsc, b"LDsc") { s.obj(ldsc, b"LDeL")? } else { ldsc };
    let weight = s.f64(lsty, b"Wght").filter(|w| *w > 0.0 && *w < 1e7)?;
    // "Scale with object": the weight follows the object's transform; otherwise it is absolute.
    let scales = s.bool(ldsc, b"LDSc").unwrap_or(true);
    let width = if scales { weight * local.scale() } else { weight };
    let data = s.bytes(lsty, b"Data").unwrap_or_default();
    let miter_limit = data.get(0..8).and_then(|b| b.try_into().ok()).map(f64::from_le_bytes).filter(|m| m.is_finite() && *m >= 1.0).unwrap_or(4.0);
    // Cap and join bytes are only partly understood: (0, 0) is butt/miter in every clean sample;
    // (2, 2) ends plain lines square and brush lines round; joins import round, which never
    // looks wrong.
    let (cap, join) = match (data.get(8).copied(), data.get(9).copied(), data.get(10).copied()) {
        (Some(0), Some(0), _) => (Cap::Butt, Join::Miter),
        (Some(0), Some(_), _) => (Cap::Butt, Join::Round),
        (Some(2), Some(2), Some(3)) => (Cap::Round, Join::Round),
        (Some(2), Some(2), _) => (Cap::Square, Join::Round),
        _ => (Cap::Round, Join::Round),
    };
    let align = match s.int(ldsc, b"LDSa") {
        Some(1) => Align::Inside,
        Some(2) => Align::Outside,
        _ => Align::Center,
    };
    let dash = match s.field(lsty, b"Patn") {
        Some(Value::Array(a)) if data.get(10) == Some(&2) || !a.is_empty() => {
            let mut v: Vec<f64> = a.iter().filter_map(|x| if let Value::Float(f) = x { Some(*f) } else { None }).collect();
            while v.len() >= 2 && v.iter().rev().take(2).all(|x| *x == 0.0) {
                v.truncate(v.len() - 2);
            }
            let valid = !v.is_empty() && v.iter().all(|x| x.is_finite() && *x >= 0.0) && v.iter().any(|x| *x > 0.0);
            valid.then(|| (v.iter().map(|x| x * width).collect(), s.f64(lsty, b"Phse").unwrap_or(0.0) * width))
        }
        _ => None,
    };
    if s.obj(lsty, b"Brus").is_some() || s.obj(ldsc, b"LDeP").is_some() {
        r.warn("brush and pressure strokes (imported as plain strokes)");
    }
    Some(Stroke { paint, width, cap, join, miter_limit, align, dash, behind })
}
