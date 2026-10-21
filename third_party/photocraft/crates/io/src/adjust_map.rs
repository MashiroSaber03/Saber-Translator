//! Mapping between PSD adjustment-layer blocks and [`Adjustment`].
//!
//! Binary layouts follow the Adobe spec ("Adjustment layer" section). The
//! mapped subset keeps the master/composite parameters; other channels of
//! Levels/Curves/Hue-Sat are written back with defaults.
//!
//! Every [`Adjustment`] kind is written (`vibA`, `blnc`, `blwh`, `phfl` and `mixr` included);
//! a Color Lookup without a table is written as an identity cube and a Gradient Map with fewer
//! than two stops as the equivalent two-stop gradient, so nothing is dropped on save. Blocks
//! that cannot be read (noise gradients, profile lookups, CMYK channel mixers, unknown versions)
//! stay [`Adjustment::Unsupported`] and are written back verbatim.

use photocraft_doc::Adjustment;
use photocraft_doc::adjust::{CURVE_MAP_LEN, CurvePoint, HueRange, LevelsChannel, ToneSpace, curve_map, curve_map_outputs};
use photocraft_psd::descriptor::{Descriptor, Value, VersionedDescriptor};

/// All PSD adjustment keys recognized as adjustment layers.
pub const ADJUSTMENT_KEYS: [&[u8; 4]; 16] =
    [b"levl", b"curv", b"hue2", b"brit", b"nvrt", b"thrs", b"post", b"expA", b"vibA", b"blnc", b"mixr", b"grdm", b"phfl", b"selc", b"blwh", b"clrL"];

fn be16(d: &[u8], at: usize) -> Option<u16> {
    d.get(at..at + 2).map(|b| u16::from_be_bytes([b[0], b[1]]))
}
fn bei16(d: &[u8], at: usize) -> Option<i16> {
    be16(d, at).map(|v| v as i16)
}
fn bef32(d: &[u8], at: usize) -> Option<f32> {
    d.get(at..at + 4).map(|b| f32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

fn unsupported(key: &[u8; 4], data: &[u8]) -> Adjustment {
    Adjustment::Unsupported { psd_key: String::from_utf8_lossy(key).into_owned(), raw: data.to_vec() }
}

fn levels_rec(d: &[u8], at: usize) -> Option<LevelsChannel> {
    Some(LevelsChannel {
        in_black: f32::from(be16(d, at)?) / 255.0,
        in_white: f32::from(be16(d, at + 2)?) / 255.0,
        out_black: f32::from(be16(d, at + 4)?) / 255.0,
        out_white: f32::from(be16(d, at + 6)?) / 255.0,
        gamma: f32::from(be16(d, at + 8)?) / 100.0,
    })
}

fn parse_curves(d: &[u8]) -> Option<Adjustment> {
    // map flag(1) version(2) bitmap(4). A map (Photoshop's pencil-drawn curves) stores 256 output
    // bytes per channel instead of a point list (psd-tools reads the flag as `is_map`).
    let pencil = *d.first()? != 0;
    let version = be16(d, 1)?;
    if version != 1 && version != 4 {
        return None;
    }
    let bits = u32::from_be_bytes(d.get(3..7)?.try_into().ok()?);
    let mut at = 7;
    let line = || vec![CurvePoint { input: 0.0, output: 0.0 }, CurvePoint { input: 1.0, output: 1.0 }];
    let mut curves: Vec<Vec<CurvePoint>> = vec![line(), line(), line(), line(), line()];
    for bit in 0..32usize {
        if bits & (1 << bit) == 0 {
            continue;
        }
        if pencil {
            let map = d.get(at..at + CURVE_MAP_LEN)?;
            at += CURVE_MAP_LEN;
            if let Some(slot) = curves.get_mut(bit) {
                *slot = curve_map(&map.iter().map(|&b| f32::from(b) / 255.0).collect::<Vec<_>>());
            }
            continue;
        }
        let n = usize::from(be16(d, at)?);
        at += 2;
        let mut pts = Vec::with_capacity(n.min(64));
        for _ in 0..n {
            let out = be16(d, at)?;
            let inp = be16(d, at + 2)?;
            at += 4;
            pts.push(CurvePoint { input: f32::from(inp) / 255.0, output: f32::from(out) / 255.0 });
        }
        if let Some(slot) = curves.get_mut(bit) {
            *slot = pts;
        }
    }
    let mut it = curves.into_iter();
    let master = it.next()?;
    let (r, g, b, k) = (it.next()?, it.next()?, it.next()?, it.next()?);
    Some(Adjustment::Curves { master, per_channel: [r, g, b], space: ToneSpace::Rgb, black: k, pencil })
}

fn desc_num(d: &Descriptor, key: &str) -> Option<f32> {
    match d.get(key)? {
        Value::Integer(v) => Some(*v as f32),
        Value::Double(v) => Some(*v as f32),
        Value::UnitFloat { value, .. } => Some(*value as f32),
        _ => None,
    }
}

fn desc_bool(d: &Descriptor, key: &str) -> Option<bool> {
    match d.get(key)? {
        Value::Boolean(b) => Some(*b),
        _ => None,
    }
}

/// Channel interpretation of per-channel Levels/Curves records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channels {
    /// Records 1..=3 are R, G, B.
    Rgb,
    /// Record 1 is the gray channel (applied to all three display channels).
    Gray,
    /// Records 1..=4 are cyan, magenta, yellow, black ([`ToneSpace::Cmyk`]).
    Cmyk,
    /// Records 1..=3 are lightness, a, b ([`ToneSpace::Lab`]).
    Lab,
    /// Channel records do not map to display channels (multichannel, indexed…); ignored.
    Other,
}

/// The menu name of the adjustment a PSD key holds (for messages).
pub fn label(key: &[u8; 4]) -> &'static str {
    match key {
        b"levl" => "Levels",
        b"curv" => "Curves",
        b"hue2" => "Hue/Saturation",
        b"brit" => "Brightness/Contrast",
        b"nvrt" => "Invert",
        b"thrs" => "Threshold",
        b"post" => "Posterize",
        b"expA" => "Exposure",
        b"vibA" => "Vibrance",
        b"blnc" => "Color Balance",
        b"mixr" => "Channel Mixer",
        b"grdm" => "Gradient Map",
        b"phfl" => "Photo Filter",
        b"selc" => "Selective Color",
        b"blwh" => "Black & White",
        b"clrL" => "Color Lookup",
        _ => "adjustment",
    }
}

/// Why [`parse`] keeps a block as [`Adjustment::Unsupported`], when there is more to say than
/// "it could not be read" (so far: Color Lookup).
pub fn unreadable_reason(key: &[u8; 4], data: &[u8]) -> Option<String> {
    match key {
        b"clrL" => read_lookup(data).err(),
        _ => None,
    }
}

/// Parses an adjustment block. `cged` is the optional `CgEd` block data
/// (modern brightness/contrast parameters). Levels/Curves store the
/// composite record first, then one per document channel (see [`Channels`]).
pub fn parse(key: &[u8; 4], data: &[u8], cged: Option<&[u8]>, channels: Channels) -> Adjustment {
    // A CMYK Channel Mixer (any record using the fourth, black, source) has no RGB meaning:
    // keep it raw. Mixers written by `write` leave that source at 0 and map in every mode.
    if key == b"mixr" && matches!(channels, Channels::Cmyk | Channels::Other) && (0..).map_while(|r| bei16(data, 4 + r * 10 + 6)).any(|k| k != 0) {
        return unsupported(key, data);
    }
    let mut a = parse_any(key, data, cged);
    match (&mut a, channels) {
        (Adjustment::Levels { space, black, .. }, Channels::Cmyk) | (Adjustment::Levels { space, black, .. }, Channels::Lab) => {
            *space = if channels == Channels::Cmyk { ToneSpace::Cmyk } else { ToneSpace::Lab };
            if channels == Channels::Lab {
                *black = LevelsChannel::default();
            }
        }
        (Adjustment::Curves { space, black, .. }, Channels::Cmyk) => {
            *space = ToneSpace::Cmyk;
            if photocraft_doc::adjust::is_identity_curve(black) {
                black.clear();
            }
        }
        (Adjustment::Curves { space, black, .. }, Channels::Lab) => {
            *space = ToneSpace::Lab;
            black.clear();
        }
        (Adjustment::Levels { black, .. }, _) => *black = LevelsChannel::default(),
        (Adjustment::Curves { black, .. }, _) => black.clear(),
        _ => {}
    }
    match (&mut a, channels) {
        (_, Channels::Rgb | Channels::Cmyk | Channels::Lab) => {}
        (Adjustment::Levels { per_channel, .. }, Channels::Gray) => {
            let g = per_channel[0].clone();
            *per_channel = [g.clone(), g.clone(), g];
        }
        (Adjustment::Curves { per_channel, .. }, Channels::Gray) => {
            let g = per_channel[0].clone();
            *per_channel = [g.clone(), g.clone(), g];
        }
        (Adjustment::Levels { per_channel, .. }, Channels::Other) => *per_channel = Default::default(),
        (Adjustment::Curves { per_channel, .. }, Channels::Other) => {
            let line = || vec![CurvePoint { input: 0.0, output: 0.0 }, CurvePoint { input: 1.0, output: 1.0 }];
            *per_channel = [line(), line(), line()];
        }
        _ => {}
    }
    a
}

fn parse_any(key: &[u8; 4], data: &[u8], cged: Option<&[u8]>) -> Adjustment {
    let parsed = match key {
        b"nvrt" => Some(Adjustment::Invert),
        b"thrs" => be16(data, 0).map(|v| Adjustment::Threshold { level: f32::from(v) / 255.0 }),
        b"post" => be16(data, 0).map(|v| Adjustment::Posterize { levels: u32::from(v) }),
        b"brit" => {
            let modern = cged.and_then(|c| VersionedDescriptor::parse_prefix(c).ok()).and_then(|(v, _)| {
                let d = v.descriptor;
                Some(Adjustment::BrightnessContrast {
                    brightness: desc_num(&d, "Brgh")?,
                    contrast: desc_num(&d, "Cntr")?,
                    legacy: desc_bool(&d, "useLegacy").unwrap_or(false),
                })
            });
            modern
                .or_else(|| Some(Adjustment::BrightnessContrast { brightness: f32::from(bei16(data, 0)?), contrast: f32::from(bei16(data, 2)?), legacy: true }))
        }
        b"hue2" => (|| {
            let colorize = *data.get(2)? != 0;
            let base = if colorize { 4 } else { 10 };
            // Six ranges from byte 16: four range values (degrees) then hue, saturation, lightness.
            let mut ranges = HueRange::defaults();
            for (i, r) in ranges.iter_mut().enumerate() {
                let at = 16 + i * 14;
                let v = |k: usize| bei16(data, at + 2 * k).map(f32::from);
                if let (Some(a), Some(b), Some(c), Some(d), Some(h), Some(s), Some(l)) = (v(0), v(1), v(2), v(3), v(4), v(5), v(6)) {
                    *r = HueRange { hue: h, saturation: s, lightness: l, bounds: HueRange::canonical_bounds([a, b, c, d]) };
                }
            }
            Some(Adjustment::HueSaturation {
                hue: f32::from(bei16(data, base)?),
                saturation: f32::from(bei16(data, base + 2)?),
                lightness: f32::from(bei16(data, base + 4)?),
                colorize,
                ranges,
            })
        })(),
        b"expA" => (|| Some(Adjustment::Exposure { exposure: bef32(data, 2)?, offset: bef32(data, 6)?, gamma: bef32(data, 10)? }))(),
        b"levl" => (|| {
            let m = levels_rec(data, 2)?;
            let r = levels_rec(data, 12)?;
            let g = levels_rec(data, 22)?;
            let b = levels_rec(data, 32)?;
            let k = levels_rec(data, 42).unwrap_or_default();
            Some(Adjustment::Levels { master: m, per_channel: [r, g, b], space: ToneSpace::Rgb, black: k })
        })(),
        b"curv" => parse_curves(data),
        b"selc" => parse_selective(data),
        b"clrL" => parse_lookup(data),
        b"grdm" => parse_gradient_map(data),
        b"vibA" => parse_vibrance(data),
        b"blnc" => parse_color_balance(data),
        b"blwh" => parse_black_white(data),
        b"phfl" => parse_photo_filter(data),
        b"mixr" => parse_channel_mixer(data),
        _ => None,
    };
    parsed.unwrap_or_else(|| unsupported(key, data))
}

fn be32(d: &[u8], at: usize) -> Option<u32> {
    d.get(at..at + 4).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

/// `grdm` (Adobe spec, "Gradient settings"): version (1, or 3 with an interpolation method
/// key), reverse, dither, [method], name, colour stops (location /4096, midpoint %, colour
/// space + four u16 components, 2 bytes), transparency stops, then smoothness (/4096) among
/// noise-gradient fields. Smoothness, midpoints and the method are baked into dense stops like
/// gradient fills. Stops may be in any plain colour space Photoshop writes (RGB, HSB, CMYK, Lab,
/// Grayscale: a CMYK or Lab document stores them in its own space); they are converted to RGB
/// through the colour-managed conversions (the document's CMYK profile when one is in scope).
/// Noise gradients (no colour stops) and stops in other spaces (book colours) stay unsupported.
fn parse_gradient_map(d: &[u8]) -> Option<Adjustment> {
    let version = be16(d, 0)?;
    let reverse = *d.get(2)? != 0;
    let dither = d.get(3).is_some_and(|b| *b != 0);
    let (method, mut at) = match version {
        1 => (None, 4),
        3 => (Some(d.get(4..8)?), 8),
        _ => return None,
    };
    at += 4 + be32(d, at)? as usize * 2;
    let n = usize::from(be16(d, at)?);
    at += 2;
    let mut stops = Vec::with_capacity(n);
    let mut mids = Vec::with_capacity(n);
    for _ in 0..n {
        let loc = (be32(d, at)? as f32 / 4096.0).clamp(0.0, 1.0);
        let mid = be32(d, at + 4)? as f32 / 100.0;
        let space = be16(d, at + 8)?;
        let c = |k: usize| be16(d, at + 10 + 2 * k);
        stops.push((loc, crate::channel_map::decode_known_color(space, [c(0)?, c(1)?, c(2)?, c(3)?])?));
        mids.push(mid);
        at += 20;
    }
    if stops.len() < 2 {
        return None;
    }
    let nt = usize::from(be16(d, at)?);
    at += 2 + nt * 10 + 2; // transparency stops, expansion count
    let smooth = f32::from(be16(d, at)?) / 4096.0;
    // Midpoint k applies to the segment after stop k (in location order).
    let mut order: Vec<usize> = (0..stops.len()).collect();
    order.sort_by(|a, b| stops[*a].0.total_cmp(&stops[*b].0));
    let mids: Vec<f32> = order.iter().skip(1).map(|i| mids[*i]).collect();
    let baked = crate::gradient_bake::bake(stops, &mids, smooth, crate::gradient_bake::Method::from_code(method));
    let mut stops: Vec<(f32, [f32; 3])> = baked.iter().map(|(t, c)| (*t, c.to_rgb())).collect();
    stops.sort_by(|a, b| a.0.total_cmp(&b.0));
    Some(Adjustment::GradientMap { stops, reverse, dither })
}

/// Points an imported `grdm` block at `target` when only Reverse and/or Dither differ (the
/// stops parse unchanged): patches those two bytes and returns `true`, so the file's own stops
/// (in whatever colour space it stored them), smoothness and method survive the save. `false`
/// leaves `data` untouched.
pub(crate) fn retarget_gradient_map(data: &mut [u8], target: &Adjustment, channels: Channels) -> bool {
    let Adjustment::GradientMap { stops, reverse, dither } = target else { return false };
    let same_stops = matches!(parse(b"grdm", data, None, channels), Adjustment::GradientMap { stops: s, .. } if s == *stops);
    match (same_stops, data.get_mut(2..4)) {
        (true, Some(flags)) => {
            flags[0] = u8::from(*reverse);
            flags[1] = u8::from(*dither);
            true
        }
        _ => false,
    }
}

/// `grdm` version 1 for [`Adjustment::GradientMap`] (Classic interpolation, Smoothness 0, so
/// the stops are reproduced exactly).
fn write_gradient_map(stops: &[(f32, [f32; 3])], reverse: bool, dither: bool) -> Vec<u8> {
    let mut v = Vec::new();
    put16(&mut v, 1);
    v.push(u8::from(reverse));
    v.push(u8::from(dither));
    let name: Vec<u16> = "Custom".encode_utf16().chain(std::iter::once(0)).collect();
    v.extend_from_slice(&(name.len() as u32).to_be_bytes());
    for u in name {
        put16(&mut v, u);
    }
    put16(&mut v, stops.len().min(usize::from(u16::MAX)) as u16);
    for (t, c) in stops.iter().take(usize::from(u16::MAX)) {
        v.extend_from_slice(&((t.clamp(0.0, 1.0) * 4096.0).round() as u32).to_be_bytes());
        v.extend_from_slice(&50u32.to_be_bytes());
        put16(&mut v, 0);
        for x in c {
            put16(&mut v, (x.clamp(0.0, 1.0) * 65535.0).round() as u16);
        }
        put16(&mut v, 0); // fourth component
        put16(&mut v, 0);
    }
    put16(&mut v, 2);
    for t in [0u32, 4096] {
        v.extend_from_slice(&t.to_be_bytes());
        v.extend_from_slice(&50u32.to_be_bytes());
        put16(&mut v, 255);
    }
    put16(&mut v, 2); // expansion count
    put16(&mut v, 0); // smoothness
    put16(&mut v, 32); // length
    put16(&mut v, 0); // mode
    v.extend_from_slice(&0u32.to_be_bytes()); // random seed
    put16(&mut v, 0); // showing transparency
    put16(&mut v, 0); // using vector colour
    v.extend_from_slice(&2048u32.to_be_bytes()); // roughness
    put16(&mut v, 3); // colour model
    for x in [0u16, 0, 0, 0, 0x8000, 0x8000, 0x8000, 0x8000] {
        put16(&mut v, x);
    }
    put16(&mut v, 0);
    v
}

/// `selc`: version, method (0 relative / 1 absolute), then 10 CMYK records of i16 percentages;
/// record 0 is reserved, records 1..=9 are reds … blacks.
fn parse_selective(d: &[u8]) -> Option<Adjustment> {
    if be16(d, 0)? != 1 || d.len() < 84 {
        return None;
    }
    let relative = be16(d, 2)? == 0;
    let mut adjustments = [[0.0f32; 4]; 9];
    for (r, rec) in adjustments.iter_mut().enumerate() {
        for (k, v) in rec.iter_mut().enumerate() {
            *v = f32::from(bei16(d, 4 + (r + 1) * 8 + k * 2)?);
        }
    }
    Some(Adjustment::SelectiveColor { relative, adjustments })
}

fn desc_text(d: &Descriptor, key: &str) -> Option<String> {
    match d.get(key)? {
        Value::Text(t) => Some(t.to_string_lossy()),
        _ => None,
    }
}

fn desc_enum<'a>(d: &'a Descriptor, key: &str) -> Option<&'a [u8]> {
    match d.get(key)? {
        Value::Enumerated { value, .. } => Some(value.as_bytes()),
        _ => None,
    }
}

/// `clrL`: version 1 + a versioned descriptor carrying the LUT file itself (`LUT3DFileData`, in
/// the format named by `LUTFormat`). Profile-based lookups (abstract / device link) have no
/// table and stay [`Adjustment::Unsupported`].
fn parse_lookup(d: &[u8]) -> Option<Adjustment> {
    read_lookup(d).ok()
}

/// [`parse_lookup`] with the reason a block can't be read.
fn read_lookup(d: &[u8]) -> Result<Adjustment, String> {
    match be16(d, 0) {
        Some(1) => {}
        Some(v) => return Err(format!("unknown version {v}")),
        None => return Err("the block is truncated".into()),
    }
    let (v, _) = VersionedDescriptor::parse_prefix(d.get(2..).unwrap_or_default()).map_err(|e| format!("unreadable settings ({e})"))?;
    let desc = v.descriptor;
    let bytes = match desc.get("LUT3DFileData") {
        Some(Value::RawData(b)) if !b.is_empty() => b,
        Some(_) => return Err("its LUT data is empty".into()),
        None => return Err("it uses a colour profile (abstract or device link), not a LUT file, which isn't supported yet".into()),
    };
    let (ext, format) = match desc_enum(&desc, "LUTFormat") {
        Some(b"LUTFormat3DL") => ("x.3dl", "3DL"),
        Some(b"LUTFormatLOOK") => ("x.look", "LOOK"),
        _ => ("x.cube", "CUBE"),
    };
    let lut = photocraft_cms::lutfile::parse(ext, bytes).map_err(|e| format!("its {format} LUT can't be read ({e})"))?;
    let name = desc_text(&desc, "LUT3DFileName").or_else(|| desc_text(&desc, "NM  ")).unwrap_or_default();
    Ok(Adjustment::ColorLookup {
        name,
        size: lut.size as u32,
        lut: Some(std::sync::Arc::new(lut.data)),
        tetrahedral: false,
        dither: desc_bool(&desc, "Dthr").unwrap_or(false),
    })
}

/// The descriptor of a `vibA` / `blwh` block: a 4-byte descriptor version (16), then the descriptor.
fn versioned(d: &[u8]) -> Option<Descriptor> {
    VersionedDescriptor::parse_prefix(d).ok().map(|(v, _)| v.descriptor)
}

/// `vibA` (Adobe spec, "Vibrance"): a versioned descriptor with `vibrance` and `Strt`
/// (saturation), both integers in -100..=100. Photoshop omits a key whose value is 0.
fn parse_vibrance(d: &[u8]) -> Option<Adjustment> {
    let desc = versioned(d)?;
    let get = |k: &str| desc_num(&desc, k).unwrap_or(0.0).clamp(-100.0, 100.0);
    Some(Adjustment::Vibrance { vibrance: get("vibrance"), saturation: get("Strt") })
}

/// `blnc` (Adobe spec, "Color Balance"): shadows, midtones and highlights as three i16 each
/// (cyan-red, magenta-green, yellow-blue, -100..=100), then preserve luminosity (1 byte).
fn parse_color_balance(d: &[u8]) -> Option<Adjustment> {
    let tone = |k: usize| -> Option<[f32; 3]> {
        let mut out = [0.0; 3];
        for (i, v) in out.iter_mut().enumerate() {
            *v = f32::from(bei16(d, (k * 3 + i) * 2)?).clamp(-100.0, 100.0);
        }
        Some(out)
    };
    Some(Adjustment::ColorBalance { shadows: tone(0)?, midtones: tone(1)?, highlights: tone(2)?, preserve_luminosity: *d.get(18)? != 0 })
}

/// Photoshop's default Black & White mix (reds, yellows, greens, cyans, blues, magentas) and tint.
const BW_DEFAULT: [f32; 6] = [40.0, 60.0, 40.0, 60.0, 20.0, 80.0];
const BW_KEYS: [&str; 6] = ["Rd  ", "Yllw", "Grn ", "Cyn ", "Bl  ", "Mgnt"];
const BW_TINT: [f32; 3] = [225.0 / 255.0, 211.0 / 255.0, 179.0 / 255.0];

/// An `RGBC` colour (0..255 doubles) read in f64 so `write`'s encoding comes back bit-exact;
/// other colour classes go through [`crate::blocks::color_from_desc`].
fn desc_rgb(d: &Descriptor) -> Option<[f32; 3]> {
    let g = |k: &str| match d.get(k)? {
        Value::Double(v) => Some(*v),
        Value::Integer(v) => Some(f64::from(*v)),
        Value::UnitFloat { value, .. } => Some(*value),
        _ => None,
    };
    if d.class_id.as_bytes() == b"RGBC" && d.get("redFloat").is_none() {
        let c = [g("Rd  ")?, g("Grn ")?, g("Bl  ")?];
        return Some(c.map(|v| (v / 255.0).clamp(0.0, 1.0) as f32));
    }
    crate::blocks::color_from_desc(d).map(|c| c.to_rgb().map(|v| v.clamp(0.0, 1.0)))
}

fn rgb_desc(c: [f32; 3]) -> Descriptor {
    let d = |v: f32| Value::Double(f64::from(v.clamp(0.0, 1.0)) * 255.0);
    Descriptor::new("RGBC").with("Rd  ", d(c[0])).with("Grn ", d(c[1])).with("Bl  ", d(c[2]))
}

/// `blwh` (Adobe spec, "Black and White"): a versioned descriptor with the six colour weights
/// (`Rd  `, `Yllw`, `Grn `, `Cyn `, `Bl  `, `Mgnt`, percent), `useTint` and `tintColor`.
fn parse_black_white(d: &[u8]) -> Option<Adjustment> {
    let desc = versioned(d)?;
    let mut weights = BW_DEFAULT;
    for (w, k) in weights.iter_mut().zip(BW_KEYS) {
        if let Some(v) = desc_num(&desc, k) {
            *w = v.clamp(-200.0, 300.0);
        }
    }
    let tint = desc_bool(&desc, "useTint").unwrap_or(false).then(|| crate::blocks::get_desc(&desc, "tintColor").and_then(desc_rgb).unwrap_or(BW_TINT));
    Some(Adjustment::BlackWhite { weights, tint })
}

/// `phfl` (Adobe spec, "Photo Filter"): version (2 or 3); version 2 has a colour structure
/// (colour space + four u16), version 3 "4 bytes each for XYZ color"; then density (u32 percent)
/// and preserve luminosity (1 byte).
///
/// The spec gives no scale for the XYZ values. They are read as XYZ relative to the D50 white
/// (X / 0.9642, Y, Z / 0.8249) with 1.0 = 32768, so white is stored as (32768, 32768, 32768).
/// That is inferred, not documented: decoded this way, every version 3 colour in the Photoshop
/// CS2 files of #3777 and #4940 is exactly a Photoshop preset (the Yellow filter holds
/// (24191, 25002, 3827) and decodes to #F9E31C, Warming Filter (85) holds (15548, 12029, 0)
/// and decodes to #EC8A00); reading them as absolute XYZ gives #FFE03A and #F18600.
const PHFL_XYZ_ONE: f32 = 32768.0;

fn parse_photo_filter(d: &[u8]) -> Option<Adjustment> {
    let (color, at) = match be16(d, 0)? {
        2 => {
            let space = be16(d, 2)?;
            if !matches!(space, 0 | 2 | 7 | 8) {
                return None;
            }
            let c = [be16(d, 4)?, be16(d, 6)?, be16(d, 8)?, be16(d, 10)?];
            (crate::channel_map::decode_color(space, c).to_rgb(), 12)
        }
        3 => {
            let rel = [be32(d, 2)?, be32(d, 6)?, be32(d, 10)?].map(|v| v as f32 / PHFL_XYZ_ONE);
            // Relative to the white, real colours stay below about 2.0 (Cooling Filter (82)
            // reaches 1.99 on Z); anything past that is not an XYZ colour.
            if rel.iter().any(|v| *v > 2.0) {
                return None;
            }
            // Relative to the D50 white: bring it back to absolute D50 XYZ before converting.
            let xyz = [rel[0] * photocraft_color::convert::D50[0], rel[1], rel[2] * photocraft_color::convert::D50[2]];
            let lin = photocraft_color::convert::mat3_mul(&photocraft_color::convert::XYZ_D50_TO_SRGB, xyz);
            (lin.map(|v| photocraft_color::convert::linear_to_srgb(v.clamp(0.0, 1.0))), 14)
        }
        _ => return None,
    };
    let density = (be32(d, at)? as f32 / 100.0).clamp(0.0, 1.0);
    let preserve_luminosity = *d.get(at + 4)? != 0;
    Some(Adjustment::PhotoFilter { color: color.map(|v| v.clamp(0.0, 1.0)), density, preserve_luminosity })
}

/// `mixr` (Adobe spec, "Channel Mixer"): version (1), monochrome (u16), then one 10-byte record
/// per output channel: four i16 source percentages (red, green, blue, and a fourth used only by
/// CMYK documents) and an i16 constant (percent). With monochrome on, the first record is the
/// gray mix (as ag-psd, MIT, documents).
fn parse_channel_mixer(d: &[u8]) -> Option<Adjustment> {
    if be16(d, 0)? != 1 {
        return None;
    }
    let monochrome = be16(d, 2)? != 0;
    let rec = |r: usize| -> Option<[f32; 4]> {
        let at = 4 + r * 10;
        let v = |k: usize| bei16(d, at + k * 2).map(|x| f32::from(x).clamp(-200.0, 200.0) / 100.0);
        Some([v(0)?, v(1)?, v(2)?, v(4)?])
    };
    let first = rec(0)?;
    // A monochrome block may carry only the gray record.
    let (g, b) = if monochrome { (rec(1).unwrap_or([0.0; 4]), rec(2).unwrap_or([0.0; 4])) } else { (rec(1)?, rec(2)?) };
    Some(Adjustment::ChannelMixer { matrix: [first, g, b], monochrome })
}

fn put16(v: &mut Vec<u8>, x: u16) {
    v.extend_from_slice(&x.to_be_bytes());
}
fn q255(v: f32) -> u16 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u16
}
fn clamp_i16(v: f32) -> i16 {
    v.round().clamp(-32768.0, 32767.0) as i16
}

fn levels_write(v: &mut Vec<u8>, c: &LevelsChannel) {
    put16(v, q255(c.in_black));
    put16(v, q255(c.in_white));
    put16(v, q255(c.out_black));
    put16(v, q255(c.out_white));
    put16(v, (c.gamma * 100.0).round().clamp(1.0, 999.0) as u16);
}

/// Serializes an adjustment into its PSD blocks: `(key, data)` pairs
/// (Brightness/Contrast also writes a `CgEd` descriptor). Every evaluated kind has an
/// encoding; the result is empty only for an [`Adjustment::Unsupported`] with a malformed key.
pub fn write(adj: &Adjustment) -> Vec<([u8; 4], Vec<u8>)> {
    let mut v = Vec::new();
    match adj {
        Adjustment::Invert => return vec![(*b"nvrt", Vec::new())],
        Adjustment::Threshold { level } => {
            put16(&mut v, q255(*level).max(1));
            put16(&mut v, 0);
            return vec![(*b"thrs", v)];
        }
        Adjustment::Posterize { levels } => {
            put16(&mut v, (*levels).clamp(2, 255) as u16);
            put16(&mut v, 0);
            return vec![(*b"post", v)];
        }
        Adjustment::BrightnessContrast { brightness, contrast, legacy } => {
            v.extend_from_slice(&clamp_i16(*brightness).to_be_bytes());
            v.extend_from_slice(&clamp_i16(*contrast).to_be_bytes());
            put16(&mut v, 127);
            v.push(0);
            let d = Descriptor::new("null")
                .with("Vrsn", Value::Integer(1))
                .with("Brgh", Value::Integer(brightness.round() as i32))
                .with("Cntr", Value::Integer(contrast.round() as i32))
                .with("means", Value::Integer(127))
                .with("Lab ", Value::Boolean(false))
                .with("useLegacy", Value::Boolean(*legacy))
                .with("Auto", Value::Boolean(false));
            return vec![(*b"brit", v), (*b"CgEd", VersionedDescriptor::new(d).to_bytes())];
        }
        Adjustment::HueSaturation { hue, saturation, lightness, colorize, ranges } => {
            put16(&mut v, 2);
            v.push(u8::from(*colorize));
            v.push(0);
            for _ in 0..2 {
                for x in [hue, saturation, lightness] {
                    v.extend_from_slice(&clamp_i16(*x).to_be_bytes());
                }
            }
            // Six hue ranges (reds, yellows, greens, cyans, blues, magentas): range values in
            // 0..360, then the range's hue, saturation and lightness.
            for r in ranges {
                for x in r.bounds {
                    put16(&mut v, x.rem_euclid(360.0).round() as u16 % 360);
                }
                for x in [r.hue, r.saturation, r.lightness] {
                    v.extend_from_slice(&clamp_i16(x).to_be_bytes());
                }
            }
            return vec![(*b"hue2", v)];
        }
        Adjustment::Exposure { exposure, offset, gamma } => {
            put16(&mut v, 1);
            for x in [exposure, offset, gamma] {
                v.extend_from_slice(&x.to_be_bytes());
            }
            v.push(1); // color space flag (spec: "1 byte")
            return vec![(*b"expA", v)];
        }
        Adjustment::Levels { master, per_channel, space, black } => {
            put16(&mut v, 2);
            levels_write(&mut v, master);
            for c in per_channel {
                levels_write(&mut v, c);
            }
            let ident = LevelsChannel::default();
            levels_write(&mut v, if *space == ToneSpace::Cmyk { black } else { &ident });
            for _ in 5..29 {
                levels_write(&mut v, &LevelsChannel::default());
            }
            return vec![(*b"levl", v)];
        }
        Adjustment::Curves { master, per_channel, space, black, pencil } => {
            v.push(u8::from(*pencil));
            put16(&mut v, 1);
            let ink = *space == ToneSpace::Cmyk && black.len() >= 2;
            v.extend_from_slice(&(if ink { 0b11111u32 } else { 0b1111 }).to_be_bytes());
            for c in std::iter::once(master).chain(per_channel.iter()).chain(ink.then_some(black)) {
                if *pencil {
                    v.extend(curve_map_outputs(c).iter().map(|&o| u8::try_from(q255(o)).unwrap_or(u8::MAX)));
                    continue;
                }
                put16(&mut v, c.len().min(19) as u16);
                for p in c.iter().take(19) {
                    put16(&mut v, q255(p.output));
                    put16(&mut v, q255(p.input));
                }
            }
            return vec![(*b"curv", v)];
        }
        Adjustment::SelectiveColor { relative, adjustments } => {
            put16(&mut v, 1);
            put16(&mut v, u16::from(!*relative));
            v.extend_from_slice(&[0; 8]);
            for rec in adjustments {
                for x in rec {
                    v.extend_from_slice(&clamp_i16(x.clamp(-100.0, 100.0)).to_be_bytes());
                }
            }
            return vec![(*b"selc", v)];
        }
        Adjustment::ColorLookup { name, lut, size, dither, .. } => {
            // A lookup without a usable table (none chosen yet, or a bad size) renders as the
            // identity, so it is written as a 2³ identity cube rather than dropped.
            let n = *size as usize;
            let cube = lut.as_deref().zip(photocraft_doc::adjust::lut3d_len(*size)).and_then(|(table, len)| table.get(..len)).filter(|_| n <= 256);
            let (file, dither) = match cube {
                Some(cube) => (
                    photocraft_cms::lutfile::LutFile { title: String::new(), size: n, data: cube.to_vec(), domain_min: [0.0; 3], domain_max: [1.0; 3] },
                    *dither,
                ),
                _ => (photocraft_cms::lutfile::LutFile { title: String::new(), ..photocraft_cms::lutfile::LutFile::identity(2) }, false),
            };
            let en =
                |t: &str, val: &str| Value::Enumerated { type_id: photocraft_psd::descriptor::Id::new(t), value: photocraft_psd::descriptor::Id::new(val) };
            let text = |t: &str| Value::Text(photocraft_psd::descriptor::UnicodeString::new_nul(t));
            let d = Descriptor::new("null")
                .with("lookupType", en("colorLookupType", "3DLUT"))
                .with("NM  ", text(name))
                .with("Dthr", Value::Boolean(dither))
                .with("profile", Value::RawData(Vec::new()))
                .with("LUTFormat", en("LUTFormatType", "LUTFormatCUBE"))
                .with("dataOrder", en("colorLookupOrder", "rgbOrder"))
                .with("tableOrder", en("colorLookupOrder", "bgrOrder"))
                .with("LUT3DFileData", Value::RawData(photocraft_cms::lutfile::write_cube(&file).into_bytes()))
                .with("LUT3DFileName", text(name));
            put16(&mut v, 1);
            v.extend_from_slice(&VersionedDescriptor::new(d).to_bytes());
            return vec![(*b"clrL", v)];
        }
        Adjustment::GradientMap { stops, reverse, dither } => {
            // Photoshop needs two stops; fewer are written as the gradient they render as
            // (none: black to white, i.e. the gray value itself; one: that colour throughout).
            let stops = match stops.as_slice() {
                [] => vec![(0.0, [0.0; 3]), (1.0, [1.0; 3])],
                [(_, c)] => vec![(0.0, *c), (1.0, *c)],
                s => s.to_vec(),
            };
            return vec![(*b"grdm", write_gradient_map(&stops, *reverse, *dither))];
        }
        Adjustment::Vibrance { vibrance, saturation } => {
            let d = Descriptor::new("null")
                .with("vibrance", Value::Integer(i32::from(clamp_i16(vibrance.clamp(-100.0, 100.0)))))
                .with("Strt", Value::Integer(i32::from(clamp_i16(saturation.clamp(-100.0, 100.0)))));
            return vec![(*b"vibA", VersionedDescriptor::new(d).to_bytes())];
        }
        Adjustment::ColorBalance { shadows, midtones, highlights, preserve_luminosity } => {
            for x in shadows.iter().chain(midtones).chain(highlights) {
                v.extend_from_slice(&clamp_i16(x.clamp(-100.0, 100.0)).to_be_bytes());
            }
            v.push(u8::from(*preserve_luminosity));
            v.push(0);
            return vec![(*b"blnc", v)];
        }
        Adjustment::BlackWhite { weights, tint } => {
            let mut d = Descriptor::new("null");
            for (w, k) in weights.iter().zip(BW_KEYS) {
                d = d.with(k, Value::Integer(i32::from(clamp_i16(w.clamp(-200.0, 300.0)))));
            }
            let d = d
                .with("useTint", Value::Boolean(tint.is_some()))
                .with("tintColor", Value::Descriptor(rgb_desc(tint.unwrap_or(BW_TINT))))
                .with("bwPresetKind", Value::Integer(1))
                .with("blackAndWhitePresetFileName", Value::Text(photocraft_psd::descriptor::UnicodeString::new_nul("")));
            return vec![(*b"blwh", VersionedDescriptor::new(d).to_bytes())];
        }
        Adjustment::PhotoFilter { color, density, preserve_luminosity } => {
            // Version 2: an RGB colour structure keeps the colour at 16 bits.
            put16(&mut v, 2);
            put16(&mut v, 0);
            for x in color {
                put16(&mut v, (x.clamp(0.0, 1.0) * 65535.0).round() as u16);
            }
            put16(&mut v, 0);
            v.extend_from_slice(&((density.clamp(0.0, 1.0) * 100.0).round() as u32).to_be_bytes());
            v.push(u8::from(*preserve_luminosity));
            // Photoshop requires the 17-byte Photo Filter payload after the version to be
            // padded to a 4-byte boundary INSIDE the tagged block's declared length.
            // A single pad byte made an 18-byte `phfl` block that Photoshop could not open
            // even though PhotoCraft and other tolerant readers accepted it (#1455).
            v.extend_from_slice(&[0, 0, 0]);
            return vec![(*b"phfl", v)];
        }
        Adjustment::ChannelMixer { matrix, monochrome } => {
            put16(&mut v, 1);
            put16(&mut v, u16::from(*monochrome));
            // Red, green, blue, then the gray record (the first row, which monochrome uses).
            for row in matrix.iter().chain(std::iter::once(&matrix[0])) {
                let pct = |x: f32| clamp_i16((x * 100.0).clamp(-200.0, 200.0)).to_be_bytes();
                for x in &row[..3] {
                    v.extend_from_slice(&pct(*x));
                }
                v.extend_from_slice(&[0, 0]);
                v.extend_from_slice(&pct(row[3]));
            }
            return vec![(*b"mixr", v)];
        }
        Adjustment::Unsupported { psd_key, raw } => {
            let k = psd_key.as_bytes();
            if k.len() == 4 {
                return vec![([k[0], k[1], k[2], k[3]], raw.clone())];
            }
        }
    }
    // Only an `Unsupported` block whose key is not four bytes has no encoding.
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_map_v3_methods_and_smoothness() {
        let classic = vec![(0.0, [0.0, 0.0, 1.0]), (1.0, [1.0, 1.0, 0.0])];
        let v1 = write_gradient_map(&classic, false, false);
        // Version 3 inserts the interpolation method after reverse/dither.
        let v3 = |m: &[u8; 4]| {
            let mut v = v1.clone();
            v[1] = 3;
            v.splice(4..4, m.iter().copied());
            v
        };
        let at_t = |data: &[u8], t: f32| match parse(b"grdm", data, None, Channels::Rgb) {
            Adjustment::GradientMap { stops, .. } => {
                let i = stops.windows(2).position(|w| w[0].0 <= t && t <= w[1].0).unwrap();
                let (x, y) = (stops[i], stops[i + 1]);
                let u = (t - x.0) / (y.0 - x.0).max(1e-6);
                std::array::from_fn::<f32, 3, _>(|k| x.1[k] + (y.1[k] - x.1[k]) * u)
            }
            other => panic!("{other:?}"),
        };
        let mid = |data: &[u8]| at_t(data, 0.5);
        let c = mid(&v3(b"Gcls"));
        assert!((c[0] - 0.5).abs() < 1e-3 && (c[2] - 0.5).abs() < 1e-3, "classic = sRGB lerp {c:?}");
        let p = mid(&v3(b"Perc"));
        assert!((p[0] - c[0]).abs() > 0.02, "perceptual differs from classic {p:?}");
        // Smoothness 100 % (4096) bends a three-stop ramp.
        let three = vec![(0.0, [0.0; 3]), (0.25, [1.0, 0.0, 0.0]), (0.75, [0.0, 0.0, 1.0]), (1.0, [1.0; 3])];
        let mut smooth = write_gradient_map(&three, false, false);
        let at = smooth.len() - 2 - 16 - 2 - 4 - 2 - 2 - 4 - 2 - 2 - 2;
        smooth[at..at + 2].copy_from_slice(&4096u16.to_be_bytes());
        let (a, b) = (at_t(&write_gradient_map(&three, false, false), 0.4), at_t(&smooth, 0.4));
        assert!((0..3).any(|k| (a[k] - b[k]).abs() > 1e-3), "{a:?} {b:?}");
        // Noise gradients (no colour stops) stay unsupported.
        assert!(matches!(parse(b"grdm", &[0, 1, 0, 0, 0, 0, 0, 0, 0, 0], None, Channels::Rgb), Adjustment::Unsupported { .. }));
    }

    #[test]
    fn pencil_curves_read_and_write_the_map_form() {
        // Photoshop's pencil-drawn curves: map flag 1, then 256 output bytes per channel.
        let mut block = vec![1, 0, 1, 0, 0, 0, 0b1111];
        let maps: [Vec<u8>; 4] = [
            (0..=255u8).rev().collect(),
            (0..=255u8).map(|k| if k < 128 { 0 } else { 255 }).collect(),
            (0..=255u8).collect(),
            (0..=255u8).map(|k| k / 2).collect(),
        ];
        for m in &maps {
            block.extend_from_slice(m);
        }
        let a = parse(b"curv", &block, None, Channels::Rgb);
        let Adjustment::Curves { master, per_channel, pencil, .. } = &a else { panic!("{a:?}") };
        assert!(*pencil);
        assert_eq!(master.len(), CURVE_MAP_LEN);
        assert_eq!((master[0].output, master[255].output), (1.0, 0.0));
        assert_eq!((per_channel[0][127].output, per_channel[0][128].output), (0.0, 1.0));
        assert_eq!(write(&a)[0].1, block);
    }

    /// A two-stop `grdm` (version 1) whose stops are the colour structures `a` and `b`.
    fn grdm_with(a: (u16, [u16; 4]), b: (u16, [u16; 4])) -> Vec<u8> {
        let mut v = write_gradient_map(&[(0.0, [0.0; 3]), (1.0, [1.0; 3])], false, false);
        // Header 4 + name length 4 + "Custom\0" 14 + stop count 2; 20 bytes per stop.
        for (i, (space, c)) in [a, b].into_iter().enumerate() {
            let at = 24 + 20 * i + 8;
            v[at..at + 2].copy_from_slice(&space.to_be_bytes());
            for (k, x) in c.iter().enumerate() {
                v[at + 2 + 2 * k..at + 4 + 2 * k].copy_from_slice(&x.to_be_bytes());
            }
        }
        v
    }

    fn ends(a: &Adjustment) -> ([f32; 3], [f32; 3]) {
        match a {
            Adjustment::GradientMap { stops, .. } => (stops[0].1, stops[stops.len() - 1].1),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn gradient_map_stops_in_every_colour_space() {
        use photocraft_color::{Color, ColorMode};
        let white_rgb = (0, [65535, 65535, 65535, 0]);
        // CMYK, stored inverted: 65535 = no ink. Full K only.
        let k_only = grdm_with((2, [65535, 65535, 65535, 0]), white_rgb);
        let expected = Color { mode: ColorMode::Cmyk, c: [0.0, 0.0, 0.0, 1.0], alpha: 1.0 }.to_rgb();
        for channels in [Channels::Cmyk, Channels::Rgb, Channels::Lab] {
            let a = parse(b"grdm", &k_only, None, channels);
            let (first, last) = ends(&a);
            assert!(first.iter().zip(expected).all(|(x, y)| (x - y).abs() < 1e-5), "{first:?} vs {expected:?}");
            assert!(first[0] < 0.4 && (first[0] - first[2]).abs() < 0.05, "K-only black is a dark neutral: {first:?}");
            assert_eq!(last, [1.0; 3]);
        }
        // Lab: L 50, a 0, b 0 is a neutral mid grey; L 100 white.
        let lab = parse(b"grdm", &grdm_with((7, [5000, 0, 0, 0]), (7, [10000, 0, 0, 0])), None, Channels::Lab);
        let (first, last) = ends(&lab);
        assert!(first.iter().all(|v| (v - 0.466).abs() < 0.01), "{first:?}");
        assert!(last.iter().all(|v| *v > 0.99), "{last:?}");
        // Lab with chroma: a = +60 is reddish (a and b are signed hundredths).
        let red = parse(b"grdm", &grdm_with((7, [5000, 6000, 0, 0]), white_rgb), None, Channels::Lab);
        let (first, _) = ends(&red);
        assert!(first[0] > first[1] + 0.2, "{first:?}");
        // Grayscale: 0..10000 is ink coverage (10000 = black).
        let gray = parse(b"grdm", &grdm_with((8, [10000, 0, 0, 0]), (8, [0, 0, 0, 0])), None, Channels::Gray);
        assert_eq!(ends(&gray), ([0.0; 3], [1.0; 3]));
        // HSB: hue 240 degrees (2/3 of 65535), full saturation and brightness = blue.
        let hsb = parse(b"grdm", &grdm_with((1, [43690, 65535, 65535, 0]), white_rgb), None, Channels::Rgb);
        let (first, _) = ends(&hsb);
        assert!(first[0] < 1e-3 && first[1] < 1e-3 && (first[2] - 1.0).abs() < 1e-3, "{first:?}");
        // Book colours and unknown ids are not guessed: the block stays raw.
        for space in [3u16, 9, 0xFFFF] {
            let d = grdm_with((space, [1, 2, 3, 4]), white_rgb);
            assert!(matches!(parse(b"grdm", &d, None, Channels::Rgb), Adjustment::Unsupported { .. }), "space {space}");
        }
        // Truncated stops never panic.
        for n in 0..k_only.len() {
            let _ = parse(b"grdm", &k_only[..n], None, Channels::Cmyk);
        }
    }

    #[test]
    fn gradient_map_reverse_and_dither_keep_the_stops_block() {
        let original = grdm_with((2, [65535, 65535, 65535, 0]), (7, [10000, 0, 0, 0]));
        let Adjustment::GradientMap { stops, .. } = parse(b"grdm", &original, None, Channels::Cmyk) else { panic!() };
        let mut data = original.clone();
        let target = Adjustment::GradientMap { stops: stops.clone(), reverse: true, dither: true };
        assert!(retarget_gradient_map(&mut data, &target, Channels::Cmyk));
        assert_eq!(&data[2..4], &[1, 1]);
        assert_eq!(data[4..], original[4..], "the stops keep their colour spaces");
        assert_eq!(parse(b"grdm", &data, None, Channels::Cmyk), target);
        // Edited stops are not patched in.
        let mut edited = stops;
        edited[0].1 = [0.5; 3];
        let mut data2 = original.clone();
        let target = Adjustment::GradientMap { stops: edited, reverse: false, dither: false };
        assert!(!retarget_gradient_map(&mut data2, &target, Channels::Cmyk));
        assert_eq!(data2, original);
        assert!(!retarget_gradient_map(&mut [], &target, Channels::Cmyk));
        assert!(!retarget_gradient_map(&mut data2, &Adjustment::Invert, Channels::Cmyk));
    }

    fn rt(a: Adjustment) {
        let blocks = write(&a);
        assert!(!blocks.is_empty(), "{a:?}");
        let cged = blocks.iter().find(|b| &b.0 == b"CgEd").map(|b| &b.1[..]);
        let back = parse(&blocks[0].0, &blocks[0].1, cged, Channels::Rgb);
        assert_eq!(back, a);
    }

    #[test]
    fn roundtrips() {
        rt(Adjustment::Invert);
        rt(Adjustment::Threshold { level: 128.0 / 255.0 });
        rt(Adjustment::Posterize { levels: 4 });
        rt(Adjustment::BrightnessContrast { brightness: 20.0, contrast: -10.0, legacy: false });
        rt(Adjustment::BrightnessContrast { brightness: -150.0, contrast: 100.0, legacy: true });
        rt(Adjustment::HueSaturation { hue: 30.0, saturation: -20.0, lightness: 5.0, colorize: false, ranges: HueRange::defaults() });
        rt(Adjustment::HueSaturation { hue: 200.0, saturation: 50.0, lightness: 0.0, colorize: true, ranges: HueRange::defaults() });
        let mut ranges = HueRange::defaults();
        ranges[0] = HueRange { hue: 15.0, saturation: -40.0, lightness: 10.0, bounds: [-60.0, -20.0, 10.0, 30.0] };
        ranges[4].saturation = 25.0;
        rt(Adjustment::HueSaturation { hue: -10.0, saturation: 0.0, lightness: 0.0, colorize: false, ranges });
        rt(Adjustment::Exposure { exposure: 1.5, offset: -0.01, gamma: 0.9 });
        let lc = |a: u16, b: u16| LevelsChannel { in_black: f32::from(a) / 255.0, in_white: f32::from(b) / 255.0, gamma: 1.2, out_black: 0.0, out_white: 1.0 };
        rt(Adjustment::Levels {
            master: lc(10, 240),
            per_channel: [lc(0, 255), lc(5, 250), lc(20, 200)],
            space: ToneSpace::Rgb,
            black: LevelsChannel::default(),
        });
        let pts = |v: &[(u8, u8)]| v.iter().map(|&(i, o)| CurvePoint { input: f32::from(i) / 255.0, output: f32::from(o) / 255.0 }).collect::<Vec<_>>();
        rt(Adjustment::Curves {
            master: pts(&[(0, 0), (128, 150), (255, 255)]),
            per_channel: [pts(&[(0, 10), (255, 255)]), pts(&[(0, 0), (255, 245)]), pts(&[(0, 0), (64, 32), (255, 255)])],
            space: ToneSpace::Rgb,
            black: Vec::new(),
            pencil: false,
        });
        // CMYK ink curves and levels keep their black record.
        let ink = Adjustment::Curves {
            master: pts(&[(0, 0), (255, 255)]),
            per_channel: [pts(&[(0, 0), (255, 255)]), pts(&[(0, 30), (255, 255)]), pts(&[(0, 0), (255, 255)])],
            space: ToneSpace::Cmyk,
            black: pts(&[(0, 0), (128, 100), (255, 255)]),
            pencil: false,
        };
        let b = write(&ink);
        assert_eq!(parse(&b[0].0, &b[0].1, None, Channels::Cmyk), ink);
        let inkl = Adjustment::Levels { master: lc(0, 255), per_channel: [lc(0, 255), lc(5, 250), lc(0, 255)], space: ToneSpace::Cmyk, black: lc(30, 255) };
        let b = write(&inkl);
        assert_eq!(parse(&b[0].0, &b[0].1, None, Channels::Cmyk), inkl);
        rt(Adjustment::Unsupported { psd_key: "selc".into(), raw: vec![1, 2, 3] });
        rt(Adjustment::GradientMap { stops: vec![(0.0, [0.0, 0.0, 0.0]), (0.5, [1.0, 0.0, 0.0]), (1.0, [1.0, 1.0, 1.0])], reverse: true, dither: false });
        rt(Adjustment::GradientMap { stops: vec![(0.0, [0.0, 0.0, 0.0]), (1.0, [1.0, 1.0, 1.0])], reverse: false, dither: true });
        rt(Adjustment::SelectiveColor { relative: true, adjustments: std::array::from_fn(|r| [r as f32 * 10.0 - 40.0, 5.0, -100.0, 100.0]) });
        rt(Adjustment::SelectiveColor { relative: false, adjustments: [[0.0; 4]; 9] });
        rt(Adjustment::Vibrance { vibrance: 35.0, saturation: -12.0 });
        rt(Adjustment::Vibrance { vibrance: 0.0, saturation: 0.0 });
        rt(Adjustment::ColorBalance { shadows: [20.0, -10.0, 5.0], midtones: [-15.0, 10.0, 30.0], highlights: [0.0, 5.0, -100.0], preserve_luminosity: true });
        rt(Adjustment::ColorBalance { shadows: [0.0; 3], midtones: [100.0, 0.0, 0.0], highlights: [0.0; 3], preserve_luminosity: false });
        rt(Adjustment::BlackWhite { weights: [40.0, 60.0, 40.0, 60.0, 20.0, 80.0], tint: None });
        rt(Adjustment::BlackWhite { weights: [-200.0, 300.0, 0.0, 10.0, 90.0, 30.0], tint: Some([0.9, 0.7, 0.5]) });
        let q = |v: u16| f32::from(v) / 65535.0;
        rt(Adjustment::PhotoFilter { color: [q(60000), q(30000), q(0)], density: 0.25, preserve_luminosity: true });
        rt(Adjustment::PhotoFilter { color: [q(0), q(40000), q(65535)], density: 1.0, preserve_luminosity: false });
        rt(Adjustment::ChannelMixer { matrix: [[0.5, 0.3, 0.2, 0.0], [0.1, 0.8, 0.1, 0.05], [0.0, 0.2, 0.9, -0.05]], monochrome: false });
        rt(Adjustment::ChannelMixer { matrix: [[0.4, 0.4, 0.2, 0.1], [0.0, 1.0, 0.0, 0.0], [-2.0, 0.0, 2.0, 0.0]], monochrome: true });
        let id = photocraft_cms::lutfile::LutFile::identity(5);
        rt(Adjustment::ColorLookup { name: "Look.cube".into(), lut: Some(std::sync::Arc::new(id.data)), size: 5, tetrahedral: false, dither: true });
    }

    #[test]
    fn selective_color_synthetic_block() {
        // Absolute; reds = (+10, -20, +30, -40), blacks = (0, 0, 0, 25).
        let mut d = vec![0, 1, 0, 1];
        d.extend([0u8; 8]);
        for r in 0..9 {
            let rec: [i16; 4] = match r {
                0 => [10, -20, 30, -40],
                8 => [0, 0, 0, 25],
                _ => [0; 4],
            };
            for x in rec {
                d.extend(x.to_be_bytes());
            }
        }
        let a = parse(b"selc", &d, None, Channels::Rgb);
        let Adjustment::SelectiveColor { relative, adjustments } = &a else { panic!("{a:?}") };
        assert!(!relative);
        assert_eq!(adjustments[0], [10.0, -20.0, 30.0, -40.0]);
        assert_eq!(adjustments[8], [0.0, 0.0, 0.0, 25.0]);
        // Written back byte-exact.
        assert_eq!(write(&a)[0].1, d);
    }

    #[test]
    fn color_lookup_synthetic_block() {
        let cube = b"TITLE \"t\"\nLUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n";
        let en = |t: &str, val: &str| Value::Enumerated { type_id: photocraft_psd::descriptor::Id::new(t), value: photocraft_psd::descriptor::Id::new(val) };
        let desc = Descriptor::new("null")
            .with("lookupType", en("colorLookupType", "3DLUT"))
            .with("NM  ", Value::Text(photocraft_psd::descriptor::UnicodeString::new_nul("Id")))
            .with("Dthr", Value::Boolean(true))
            .with("LUTFormat", en("LUTFormatType", "LUTFormatCUBE"))
            .with("LUT3DFileData", Value::RawData(cube.to_vec()));
        let mut d = vec![0, 1];
        d.extend(VersionedDescriptor::new(desc).to_bytes());
        let a = parse(b"clrL", &d, None, Channels::Rgb);
        let Adjustment::ColorLookup { name, lut: Some(t), size: 2, dither: true, .. } = &a else { panic!("{a:?}") };
        assert_eq!(name, "Id");
        assert_eq!(&t[..6], &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        // A profile-based lookup (no table) is preserved raw.
        let mut p = vec![0, 1];
        p.extend(VersionedDescriptor::new(Descriptor::new("null").with("lookupType", en("colorLookupType", "abstractProfile"))).to_bytes());
        assert!(matches!(parse(b"clrL", &p, None, Channels::Rgb), Adjustment::Unsupported { .. }));
        // Each way a lookup can't be read has a reason for the import warning (#1763).
        assert_eq!(unreadable_reason(b"clrL", &d), None);
        assert!(unreadable_reason(b"clrL", &p).unwrap().contains("colour profile"));
        assert!(unreadable_reason(b"clrL", &[0]).unwrap().contains("truncated"));
        assert!(unreadable_reason(b"clrL", &[0, 2, 0, 0]).unwrap().contains("version 2"));
        assert!(unreadable_reason(b"clrL", &[0, 1, 9]).unwrap().contains("unreadable settings"));
        let bad = Descriptor::new("null").with("LUTFormat", en("LUTFormatType", "LUTFormat3DL")).with("LUT3DFileData", Value::RawData(b"not a lut".to_vec()));
        let mut b = vec![0, 1];
        b.extend(VersionedDescriptor::new(bad).to_bytes());
        assert!(matches!(parse(b"clrL", &b, None, Channels::Rgb), Adjustment::Unsupported { .. }));
        assert!(unreadable_reason(b"clrL", &b).unwrap().contains("3DL LUT can't be read"), "{:?}", unreadable_reason(b"clrL", &b));
        assert_eq!(label(b"clrL"), "Color Lookup");
        assert!(matches!(parse(b"clrL", &[0, 1, 9], None, Channels::Rgb), Adjustment::Unsupported { .. }));
        assert!(matches!(parse(b"selc", &[0, 1, 0, 0], None, Channels::Rgb), Adjustment::Unsupported { .. }));
    }

    #[test]
    fn malformed_falls_back_to_unsupported() {
        assert!(matches!(parse(b"levl", &[0, 2, 1], None, Channels::Rgb), Adjustment::Unsupported { .. }));
        assert!(matches!(parse(b"curv", &[0, 0, 9], None, Channels::Rgb), Adjustment::Unsupported { .. }));
        assert!(matches!(parse(b"thrs", &[], None, Channels::Rgb), Adjustment::Unsupported { .. }));
        assert!(matches!(parse(b"blnc", &[0; 18], None, Channels::Rgb), Adjustment::Unsupported { .. }));
        assert!(matches!(parse(b"vibA", &[0, 0, 0, 15, 0], None, Channels::Rgb), Adjustment::Unsupported { .. }));
        assert!(matches!(parse(b"blwh", &[], None, Channels::Rgb), Adjustment::Unsupported { .. }));
        assert!(matches!(parse(b"mixr", &[0, 2, 0, 0], None, Channels::Rgb), Adjustment::Unsupported { .. }));
        // A CMYK mixer (black source used) stays raw in CMYK documents; our own RGB layout maps.
        let mut cmyk = i16s(&[1, 0]);
        for r in 0..4 {
            cmyk.extend(i16s(&std::array::from_fn::<i16, 5, _>(|k| if k == r { 100 } else { 0 })));
        }
        assert!(matches!(parse(b"mixr", &cmyk, None, Channels::Cmyk), Adjustment::Unsupported { .. }));
        assert!(matches!(parse(b"mixr", &cmyk, None, Channels::Other), Adjustment::Unsupported { .. }));
        let ours = write(&Adjustment::ChannelMixer { matrix: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], monochrome: false });
        assert!(matches!(parse(b"mixr", &ours[0].1, None, Channels::Cmyk), Adjustment::ChannelMixer { .. }));
        assert!(matches!(parse(b"mixr", &ours[0].1, None, Channels::Other), Adjustment::ChannelMixer { .. }));
        assert!(matches!(parse(b"mixr", &[0, 1, 0, 0, 0, 100], None, Channels::Rgb), Adjustment::Unsupported { .. }));
        // Photo Filter: unknown version, HSB colour space, out-of-range version-3 XYZ, truncated.
        assert!(matches!(parse(b"phfl", &[0, 4, 0, 0], None, Channels::Rgb), Adjustment::Unsupported { .. }));
        assert!(matches!(parse(b"phfl", &[0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 1], None, Channels::Rgb), Adjustment::Unsupported { .. }));
        for bad in [[200_000u32, 0, 0], [0, u32::MAX, 0]] {
            let mut xyz = vec![0, 3];
            for x in bad {
                xyz.extend(x.to_be_bytes());
            }
            xyz.extend([0, 0, 0, 25, 1]);
            assert!(matches!(parse(b"phfl", &xyz, None, Channels::Rgb), Adjustment::Unsupported { .. }));
        }
        assert!(matches!(parse(b"phfl", &[0, 2, 0, 0, 0, 0], None, Channels::Rgb), Adjustment::Unsupported { .. }));
        // Arbitrary bytes never panic, whatever the key.
        for key in ADJUSTMENT_KEYS {
            for len in 0..64usize {
                let d: Vec<u8> = (0..len).map(|i| (i * 37 + len) as u8).collect();
                let _ = parse(key, &d, Some(&d), Channels::Rgb);
                let _ = parse(key, &d, None, Channels::Other);
            }
        }
    }

    fn i16s(v: &[i16]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_be_bytes()).collect()
    }

    /// Hand-built blocks laid out per the Adobe spec (no file written by Photoshop involved).
    #[test]
    fn color_balance_golden_block() {
        let mut d = i16s(&[-30, 20, 0, 10, -5, 100, 0, 0, -100]);
        d.extend([1, 0]);
        let a = parse(b"blnc", &d, None, Channels::Rgb);
        assert_eq!(
            a,
            Adjustment::ColorBalance { shadows: [-30.0, 20.0, 0.0], midtones: [10.0, -5.0, 100.0], highlights: [0.0, 0.0, -100.0], preserve_luminosity: true }
        );
        assert_eq!(write(&a), vec![(*b"blnc", d.clone())]);
        // The 19-byte form (no trailing pad) parses the same.
        assert_eq!(parse(b"blnc", &d[..19], None, Channels::Rgb), a);
    }

    #[test]
    fn photo_filter_golden_blocks() {
        // Version 2, RGB colour structure (65535, 32768, 0), density 40 %, preserve luminosity.
        let mut v2 = i16s(&[2, 0]);
        v2.extend([0xff, 0xff, 0x80, 0x00, 0, 0, 0, 0]);
        v2.extend(40u32.to_be_bytes());
        v2.extend([1, 0, 0, 0]);
        assert_eq!(v2.len(), 20, "Photo Filter must occupy 20 bytes");
        assert_eq!(&v2[17..], &[0, 0, 0], "padding is included inside the block length");
        let a = parse(b"phfl", &v2, None, Channels::Rgb);
        let Adjustment::PhotoFilter { color, density, preserve_luminosity: true } = a else { panic!("{a:?}") };
        assert_eq!(color, [1.0, 32768.0 / 65535.0, 0.0]);
        assert_eq!(density, 0.4);
        assert_eq!(write(&a)[0].1, v2, "version 2 RGB is written back byte-exact");
        // Older PhotoCraft files used only one padding byte (18 bytes); keep importing them,
        // but never produce that Photoshop-incompatible length again.
        assert_eq!(parse(b"phfl", &v2[..18], None, Channels::Rgb), a);
        for preserve_luminosity in [false, true] {
            for density in [0.0, 0.14, 1.0] {
                let adjustment = Adjustment::PhotoFilter { color: [0.8, 0.4, 0.2], density, preserve_luminosity };
                let blocks = write(&adjustment);
                assert_eq!(blocks.len(), 1);
                assert_eq!(blocks[0].0, *b"phfl");
                assert_eq!(blocks[0].1.len(), 20);
                assert_eq!(&blocks[0].1[17..], &[0, 0, 0]);
                let Adjustment::PhotoFilter { color, density: read_density, preserve_luminosity: read_preserve } =
                    parse(b"phfl", &blocks[0].1, None, Channels::Rgb)
                else {
                    panic!("written phfl must parse")
                };
                assert!(color.iter().zip([0.8, 0.4, 0.2]).all(|(got, expected)| (got - expected).abs() <= 1.0 / 65535.0));
                assert!((read_density - density).abs() < 0.00001);
                assert_eq!(read_preserve, preserve_luminosity);
            }
        }
        // Version 2, Lab colour structure: L 50, a 0, b 0 is mid gray.
        let mut lab = i16s(&[2, 7, 5000, 0, 0, 0]);
        lab.extend(25u32.to_be_bytes());
        lab.push(0);
        let Adjustment::PhotoFilter { color, density, preserve_luminosity: false } = parse(b"phfl", &lab, None, Channels::Rgb) else { panic!() };
        assert!(color.iter().all(|c| (c - 0.466).abs() < 0.01), "{color:?}");
        assert_eq!(density, 0.25);
        // Version 3: XYZ relative to the D50 white, 1.0 = 32768. White is (32768, 32768, 32768).
        let v3 = phfl_v3([32768, 32768, 32768], 100, true);
        let Adjustment::PhotoFilter { color, density, preserve_luminosity: true } = parse(b"phfl", &v3, None, Channels::Rgb) else { panic!() };
        assert!(color.iter().all(|c| (c - 1.0).abs() < 0.005), "{color:?}");
        assert_eq!(density, 1.0);
        // Photoshop's Warming Filter (85) preset, sRGB #EC8A00: the CS2 file of #4940 stores it
        // relative to the D50 white as (15548, 12029, 0).
        let Adjustment::PhotoFilter { color, density, preserve_luminosity: false } =
            parse(b"phfl", &phfl_v3([15548, 12029, 0], 25, false), None, Channels::Rgb)
        else {
            panic!()
        };
        let expected = [0xecu8, 0x8a, 0x00].map(|v| f32::from(v) / 255.0);
        assert!(color.iter().zip(expected).all(|(got, want)| (got - want).abs() < 1.5 / 255.0), "{color:?}");
        assert_eq!(density, 0.25);
        // The block from #3777 (Photoshop CS2): the Yellow filter preset, #F9E31C, density 63 %.
        let issue = [0, 3, 0, 0, 0x5e, 0x7f, 0, 0, 0x61, 0xaa, 0, 0, 0x0e, 0xf3, 0, 0, 0, 0x3f, 1, 0];
        let Adjustment::PhotoFilter { color, density, preserve_luminosity: true } = parse(b"phfl", &issue, None, Channels::Rgb) else { panic!() };
        let expected = [0xf9u8, 0xe3, 0x1c].map(|v| f32::from(v) / 255.0);
        assert!(color.iter().zip(expected).all(|(got, want)| (got - want).abs() < 1.5 / 255.0), "{color:?}");
        assert_eq!(density, 0.63);
        // Cooling Filter (82) from #4940: a deep blue outside sRGB; its Z stays under the 2.0
        // garbage guard and the colour clamps to (0, 181, 255).
        let Adjustment::PhotoFilter { color: cooling, density, preserve_luminosity: true } =
            parse(b"phfl", &phfl_v3([5327, 9584, 65357], 72, true), None, Channels::Rgb)
        else {
            panic!("Cooling Filter (82) must parse")
        };
        let expected = [0u8, 181, 255].map(|v| f32::from(v) / 255.0);
        assert!(cooling.iter().zip(expected).all(|(got, want)| (got - want).abs() < 1.5 / 255.0), "{cooling:?}");
        assert_eq!(density, 0.72);
        // The 19-byte form (no trailing pad) parses the same.
        assert_eq!(parse(b"phfl", &issue[..19], None, Channels::Rgb), parse(b"phfl", &issue, None, Channels::Rgb));
        // Version 3 is read, never written: an edited filter is saved as version 2 and keeps its colour.
        let a = parse(b"phfl", &issue, None, Channels::Rgb);
        let w = write(&a);
        assert_eq!(&w[0].1[..2], &[0, 2]);
        let Adjustment::PhotoFilter { color: back, .. } = parse(b"phfl", &w[0].1, None, Channels::Rgb) else { panic!() };
        assert!(back.iter().zip(color).all(|(b, c)| (b - c).abs() <= 1.0 / 65535.0));
    }

    /// A version 3 Photo Filter block: version, three u32 XYZ components, density, flag, pad.
    fn phfl_v3(xyz: [u32; 3], density: u32, preserve_luminosity: bool) -> Vec<u8> {
        let mut d = vec![0, 3];
        for x in xyz {
            d.extend(x.to_be_bytes());
        }
        d.extend(density.to_be_bytes());
        d.extend([u8::from(preserve_luminosity), 0]);
        d
    }

    #[test]
    fn channel_mixer_golden_blocks() {
        // RGB: red = 100 % R + 20 % const; green = 50 % R + 50 % G; blue = -50 % G + 150 % B - 10 % const.
        let mut d = i16s(&[1, 0]);
        d.extend(i16s(&[100, 0, 0, 0, 20, 50, 50, 0, 0, 0, 0, -50, 150, 0, -10]));
        let a = parse(b"mixr", &d, None, Channels::Rgb);
        assert_eq!(a, Adjustment::ChannelMixer { matrix: [[1.0, 0.0, 0.0, 0.2], [0.5, 0.5, 0.0, 0.0], [0.0, -0.5, 1.5, -0.1]], monochrome: false });
        // Written with the trailing gray record (a copy of the first row).
        let w = write(&a);
        assert_eq!(w[0].1[..d.len()], d[..]);
        assert_eq!(w[0].1[d.len()..], i16s(&[100, 0, 0, 0, 20])[..]);
        // Monochrome with only the gray record: 30/59/11.
        let mut m = i16s(&[1, 1]);
        m.extend(i16s(&[30, 59, 11, 0, 0]));
        let Adjustment::ChannelMixer { matrix, monochrome: true } = parse(b"mixr", &m, None, Channels::Rgb) else { panic!() };
        assert_eq!(matrix[0], [0.3, 0.59, 0.11, 0.0]);
    }

    fn vdesc(d: Descriptor) -> Vec<u8> {
        VersionedDescriptor::new(d).to_bytes()
    }

    #[test]
    fn vibrance_golden_block() {
        let d = vdesc(Descriptor::new("null").with("vibrance", Value::Integer(-40)).with("Strt", Value::Integer(25)));
        assert_eq!(&d[..4], &[0, 0, 0, 16], "descriptor version 16");
        assert_eq!(parse(b"vibA", &d, None, Channels::Rgb), Adjustment::Vibrance { vibrance: -40.0, saturation: 25.0 });
        // Photoshop omits zero values.
        let only = vdesc(Descriptor::new("null").with("Strt", Value::Integer(10)));
        assert_eq!(parse(b"vibA", &only, None, Channels::Rgb), Adjustment::Vibrance { vibrance: 0.0, saturation: 10.0 });
        // Other descriptor versions are rejected (kept raw).
        let mut v = d.clone();
        v[3] = 17;
        assert!(matches!(parse(b"vibA", &v, None, Channels::Rgb), Adjustment::Unsupported { .. }));
    }

    #[test]
    fn black_white_golden_block() {
        let tint = Descriptor::new("RGBC").with("Rd  ", Value::Double(255.0)).with("Grn ", Value::Double(127.5)).with("Bl  ", Value::Double(0.0));
        let mut desc = Descriptor::new("null");
        for (k, w) in BW_KEYS.iter().zip([10, 20, 30, 40, 50, 60]) {
            desc = desc.with(k, Value::Integer(w));
        }
        let d = vdesc(
            desc.clone().with("useTint", Value::Boolean(true)).with("tintColor", Value::Descriptor(tint.clone())).with("bwPresetKind", Value::Integer(1)),
        );
        let a = parse(b"blwh", &d, None, Channels::Rgb);
        assert_eq!(a, Adjustment::BlackWhite { weights: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0], tint: Some([1.0, 0.5, 0.0]) });
        // useTint off: no tint, even with a stored colour.
        let off = vdesc(desc.with("useTint", Value::Boolean(false)).with("tintColor", Value::Descriptor(tint)));
        assert!(matches!(parse(b"blwh", &off, None, Channels::Rgb), Adjustment::BlackWhite { tint: None, .. }));
        // Missing weights take Photoshop's defaults.
        let empty = vdesc(Descriptor::new("null"));
        assert_eq!(parse(b"blwh", &empty, None, Channels::Rgb), Adjustment::BlackWhite { weights: BW_DEFAULT, tint: None });
    }

    #[test]
    fn degenerate_adjustments_still_get_an_encoding() {
        // Gradient maps with fewer than two stops render as gray (none) or one colour.
        for (stops, want) in
            [(vec![], vec![(0.0, [0.0; 3]), (1.0, [1.0; 3])]), (vec![(0.3, [0.2, 0.4, 0.6])], vec![(0.0, [0.2, 0.4, 0.6]), (1.0, [0.2, 0.4, 0.6])])]
        {
            let b = write(&Adjustment::GradientMap { stops, reverse: true, dither: false });
            let back = parse(&b[0].0, &b[0].1, None, Channels::Rgb);
            let Adjustment::GradientMap { stops, reverse: true, .. } = back else { panic!("{back:?}") };
            assert_eq!(stops.len(), want.len());
            for (s, w) in stops.iter().zip(&want) {
                assert!((s.0 - w.0).abs() < 1e-6 && (0..3).all(|k| (s.1[k] - w.1[k]).abs() < 1e-4), "{s:?} vs {w:?}");
            }
        }
        // A colour lookup without a (usable) table becomes an identity cube, without dither.
        for lut in [None, Some(std::sync::Arc::new(vec![0.5; 7]))] {
            let b = write(&Adjustment::ColorLookup { name: "L".into(), lut, size: 3, tetrahedral: false, dither: true });
            let back = parse(&b[0].0, &b[0].1, None, Channels::Rgb);
            let Adjustment::ColorLookup { lut: Some(t), size: 2, dither: false, name, .. } = back else { panic!("{back:?}") };
            assert_eq!(name, "L");
            assert_eq!(*t, photocraft_cms::lutfile::LutFile::identity(2).data);
        }
        // Every kind except a malformed Unsupported key has an encoding.
        assert!(write(&Adjustment::Unsupported { psd_key: "abc".into(), raw: vec![] }).is_empty());
    }

    #[test]
    fn non_rgb_ignores_channel_records() {
        let lc = LevelsChannel { in_black: 0.1, ..Default::default() };
        let a = Adjustment::Levels {
            master: LevelsChannel::default(),
            per_channel: [lc.clone(), lc.clone(), lc],
            space: ToneSpace::Rgb,
            black: LevelsChannel::default(),
        };
        let b = write(&a);
        match parse(&b[0].0, &b[0].1, None, Channels::Other) {
            Adjustment::Levels { per_channel, .. } => assert_eq!(per_channel, <[LevelsChannel; 3]>::default()),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn gray_uses_first_channel_record_for_all() {
        let lc = LevelsChannel { in_black: 44.0 / 255.0, ..Default::default() };
        let a = Adjustment::Levels {
            master: LevelsChannel::default(),
            per_channel: [lc.clone(), Default::default(), Default::default()],
            space: ToneSpace::Rgb,
            black: LevelsChannel::default(),
        };
        let b = write(&a);
        match parse(&b[0].0, &b[0].1, None, Channels::Gray) {
            Adjustment::Levels { per_channel, .. } => assert_eq!(per_channel, [lc.clone(), lc.clone(), lc]),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn legacy_brit_without_cged() {
        let a = parse(b"brit", &[0, 10, 0xff, 0xf6, 0, 127, 0], None, Channels::Rgb);
        assert_eq!(a, Adjustment::BrightnessContrast { brightness: 10.0, contrast: -10.0, legacy: true });
    }
}
