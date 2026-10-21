//! Creative 3D LUT files for Image › Adjustments › Color Lookup: `.cube` (Adobe/Resolve), `.3dl`
//! (Autodesk Lustre / Flame) and `.look` (SpeedGrade XML), plus a few generated (CC0) looks.
//!
//! Every table is normalised to [`LutFile`]: `size`³ RGB triplets in 0..=1, **red varying
//! fastest** (`((b·size + g)·size + r)·3`). Formats follow their public descriptions: the Adobe
//! "Cube LUT Specification 1.0", the Lustre 3DL layout as documented by OpenColorIO, and the
//! SpeedGrade `.look` XML (hex-encoded little-endian float32 RGB triplets).

/// A parsed 3D LUT.
#[derive(Clone, Debug, PartialEq)]
pub struct LutFile {
    pub title: String,
    pub size: usize,
    /// `size`³ × 3 values, red fastest.
    pub data: Vec<f32>,
    /// The input domain the table expects (`.cube` `DOMAIN_MIN`/`DOMAIN_MAX`,
    /// `LUT_3D_INPUT_RANGE`): an input `c` samples the table at `(c - min) / (max - min)`,
    /// saturating outside. `[0, 1]` is the identity default; applying a table with any other
    /// domain is rejected upstream until Color Lookup carries it through.
    pub domain_min: [f32; 3],
    pub domain_max: [f32; 3],
}

/// Largest edge length accepted (64³ is Photoshop's practical maximum; 129 leaves headroom while
/// keeping the GPU table within texture limits).
pub const MAX_SIZE: usize = 129;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LutError(pub String);

impl std::fmt::Display for LutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for LutError {}

fn err<T>(msg: impl Into<String>) -> Result<T, LutError> {
    Err(LutError(msg.into()))
}

fn checked_data_len(size: usize) -> Result<(usize, usize), LutError> {
    if !(2..=MAX_SIZE).contains(&size) {
        return err(format!("unsupported LUT size {size}"));
    }
    let entries = size.checked_pow(3).ok_or_else(|| LutError(format!("unsupported LUT size {size}")))?;
    let data_len = entries.checked_mul(3).ok_or_else(|| LutError(format!("unsupported LUT size {size}")))?;
    Ok((entries, data_len))
}

impl LutFile {
    /// The identity table of edge `size`.
    pub fn identity(size: usize) -> Self {
        Self::from_fn("Identity", size, |c| c)
    }

    /// Samples `f` on an `size`³ grid.
    pub fn from_fn(title: &str, size: usize, f: impl Fn([f32; 3]) -> [f32; 3]) -> Self {
        let m = (size.max(2) - 1) as f32;
        let mut data = Vec::with_capacity(size * size * size * 3);
        for b in 0..size {
            for g in 0..size {
                for r in 0..size {
                    let o = f([r as f32 / m, g as f32 / m, b as f32 / m]);
                    data.extend(o.map(|v| v.clamp(0.0, 1.0)));
                }
            }
        }
        LutFile { title: title.to_string(), size, data, domain_min: [0.0; 3], domain_max: [1.0; 3] }
    }

    /// Whether the table expects plain 0..1 inputs (the common case; Photoshop's default).
    pub fn domain_is_default(&self) -> bool {
        self.domain_min == [0.0, 0.0, 0.0] && self.domain_max == [1.0, 1.0, 1.0]
    }

    fn check(self) -> Result<Self, LutError> {
        let (entries, data_len) = checked_data_len(self.size)?;
        if self.data.len() != data_len {
            return err(format!("expected {entries} entries, found {}", self.data.len() / 3));
        }
        if self.data.iter().any(|v| !v.is_finite()) {
            return err("non-finite LUT value");
        }
        Ok(self)
    }
}

/// Parses a LUT, picking the format from `file_name`'s extension (falls back to sniffing).
pub fn parse(file_name: &str, bytes: &[u8]) -> Result<LutFile, LutError> {
    let lower = file_name.to_ascii_lowercase();
    let text = String::from_utf8_lossy(bytes);
    if lower.ends_with(".3dl") {
        parse_3dl(&text)
    } else if lower.ends_with(".look") || text.trim_start().starts_with('<') {
        parse_look(&text)
    } else {
        parse_cube(&text)
    }
}

/// Adobe / Resolve `.cube`. 1D LUTs (`LUT_1D_SIZE`) are expanded to a 33³ table.
pub fn parse_cube(text: &str) -> Result<LutFile, LutError> {
    let mut title = String::new();
    let (mut size3, mut size1) = (0usize, 0usize);
    let (mut dmin, mut dmax) = ([0.0f32; 3], [1.0f32; 3]);
    let mut rows: Vec<f32> = Vec::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let key = it.next().unwrap_or("");
        let nums = |it: std::str::SplitWhitespace<'_>| -> Result<[f32; 3], LutError> {
            let v: Vec<f32> = it.filter_map(|s| s.parse().ok()).collect();
            if v.len() == 3 { Ok([v[0], v[1], v[2]]) } else { err(format!("bad line `{line}`")) }
        };
        match key {
            "TITLE" => title = line[5..].trim().trim_matches('"').to_string(),
            "LUT_3D_SIZE" => size3 = it.next().and_then(|s| s.parse().ok()).ok_or_else(|| LutError("bad LUT_3D_SIZE".into()))?,
            "LUT_1D_SIZE" => size1 = it.next().and_then(|s| s.parse().ok()).ok_or_else(|| LutError("bad LUT_1D_SIZE".into()))?,
            "DOMAIN_MIN" => dmin = nums(it)?,
            "DOMAIN_MAX" => dmax = nums(it)?,
            "LUT_3D_INPUT_RANGE" | "LUT_1D_INPUT_RANGE" => {
                let v: Vec<f32> = it.filter_map(|s| s.parse().ok()).collect();
                if v.len() == 2 {
                    dmin = [v[0]; 3];
                    dmax = [v[1]; 3];
                }
            }
            k if k.starts_with(|c: char| c.is_ascii_digit() || c == '-' || c == '+' || c == '.') => {
                let mut v = vec![k.parse::<f32>().map_err(|_| LutError(format!("bad number in `{line}`")))?];
                v.extend(it.filter_map(|s| s.parse::<f32>().ok()));
                if v.len() != 3 {
                    return err(format!("bad data line `{line}`"));
                }
                rows.extend(v);
            }
            // Unknown keywords (e.g. Resolve's LUT_IN_VIDEO_RANGE) are ignored, as the spec allows.
            _ => {}
        }
    }
    let domain_ok = (0..3).all(|k| dmin[k] < dmax[k] && dmin[k].is_finite() && dmax[k].is_finite());
    if (size3 > 0 || size1 >= 2) && !domain_ok {
        return err(format!("DOMAIN_MIN {:?} must lie below DOMAIN_MAX {:?}", dmin, dmax));
    }
    if size3 > 0 {
        return LutFile { title, size: size3, data: rows, domain_min: dmin, domain_max: dmax }.check();
    }
    if size1 >= 2 {
        let expected_rows = size1.checked_mul(3).ok_or_else(|| LutError(format!("unsupported 1D LUT size {size1}")))?;
        if rows.len() != expected_rows {
            return err("no LUT_3D_SIZE in .cube file");
        }
        // The 1D table is resampled onto a 0..1 cube anyway, so its input domain
        // (`DOMAIN_MIN`/`MAX`, `LUT_1D_INPUT_RANGE`) is baked into the resample: an input `v`
        // reads the table at `(v - min) / (max - min)`, saturating outside.
        let curve = |ch: usize, v: f32| {
            let t = (v - dmin[ch]) / (dmax[ch] - dmin[ch]);
            let x = if t.is_finite() { t.clamp(0.0, 1.0) } else { 0.0 } * (size1 - 1) as f32;
            let i = (x.floor() as usize).min(size1 - 2);
            let f = x - i as f32;
            rows[i * 3 + ch] * (1.0 - f) + rows[(i + 1) * 3 + ch] * f
        };
        let mut l = LutFile::from_fn(&title, 33, |c| [curve(0, c[0]), curve(1, c[1]), curve(2, c[2])]);
        l.title = title;
        return l.check();
    }
    err("no LUT_3D_SIZE in .cube file")
}

/// Lustre `.3dl`: an optional shaper line of input values, then integer RGB rows with **blue**
/// varying fastest; the output bit depth is inferred from the largest value.
pub fn parse_3dl(text: &str) -> Result<LutFile, LutError> {
    let mut rows: Vec<[f64; 3]> = Vec::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() || line.starts_with(|c: char| c.is_ascii_alphabetic()) {
            continue;
        }
        let v: Vec<f64> = line.split_whitespace().filter_map(|s| s.parse().ok()).collect();
        if v.len() == 3 {
            rows.push([v[0], v[1], v[2]]);
        }
        // Lines with more than three numbers are the input shaper; it is assumed linear.
    }
    let n = (rows.len() as f64).cbrt().round() as usize;
    if n < 2 || n * n * n != rows.len() {
        return err(format!("{} rows is not a cube", rows.len()));
    }
    let max = rows.iter().flatten().fold(0.0f64, |a, &b| a.max(b));
    let scale = if max <= 1.0 { 1.0 } else { [1023.0, 4095.0, 16383.0, 65535.0].into_iter().find(|&s| max <= s).unwrap_or(max) };
    let mut data = vec![0.0f32; n * n * n * 3];
    for (i, row) in rows.iter().enumerate() {
        let (r, g, b) = (i / (n * n), (i / n) % n, i % n);
        let at = ((b * n + g) * n + r) * 3;
        for k in 0..3 {
            data[at + k] = (row[k] / scale) as f32;
        }
    }
    LutFile { title: String::new(), size: n, data, domain_min: [0.0; 3], domain_max: [1.0; 3] }.check()
}

/// SpeedGrade `.look`: `<size>` and a hex `<data>` string of little-endian float32 RGB triplets
/// (red fastest).
pub fn parse_look(text: &str) -> Result<LutFile, LutError> {
    let tag = |name: &str| -> Option<String> {
        let open = format!("<{name}>");
        let start = text.find(&open)? + open.len();
        let end = text[start..].find(&format!("</{name}>"))? + start;
        Some(text[start..end].trim().trim_matches('"').trim().to_string())
    };
    let size: usize = tag("size").and_then(|s| s.parse().ok()).ok_or_else(|| LutError("no <size> in .look".into()))?;
    let (entries, _) = checked_data_len(size)?;
    let hex: Vec<u8> = tag("data").ok_or_else(|| LutError("no <data> in .look".into()))?.bytes().filter(u8::is_ascii_hexdigit).collect();
    let nib = |c: u8| (c as char).to_digit(16).unwrap_or(0) as u8;
    let bytes: Vec<u8> = hex.as_chunks::<2>().0.iter().map(|p| nib(p[0]) << 4 | nib(p[1])).collect();
    let floats: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let rgba_entries = entries.checked_mul(4).ok_or_else(|| LutError(format!("unsupported LUT size {size}")))?;
    let data = if floats.len() == rgba_entries { floats.as_chunks::<4>().0.iter().flat_map(|c| [c[0], c[1], c[2]]).collect() } else { floats };
    LutFile { title: tag("title").unwrap_or_default(), size, data, domain_min: [0.0; 3], domain_max: [1.0; 3] }.check()
}

/// Writes a `.cube` file (what Photoshop embeds in a Color Lookup layer).
pub fn write_cube(l: &LutFile) -> String {
    let mut s = String::new();
    if !l.title.is_empty() {
        s.push_str(&format!("TITLE \"{}\"\n", l.title.replace('"', "'")));
    }
    s.push_str(&format!("LUT_3D_SIZE {}\n", l.size));
    for c in l.data.as_chunks::<3>().0 {
        s.push_str(&format!("{:.6} {:.6} {:.6}\n", c[0], c[1], c[2]));
    }
    s
}

fn luma(c: [f32; 3]) -> f32 {
    0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
}

fn smooth(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Generated looks shipped with Photocraft (procedural, CC0): `(id, label)`.
pub const BUILTIN: [(&str, &str); 8] = [
    ("warm", "Warm Filter"),
    ("cool", "Cool Filter"),
    ("tealOrange", "Teal & Orange"),
    ("bleachBypass", "Bleach Bypass"),
    ("fadedFilm", "Faded Film"),
    ("dayForNight", "Day for Night"),
    ("monoContrast", "Mono High Contrast"),
    ("crossProcess", "Cross Process"),
];

/// A built-in look by id (33³), or None.
pub fn builtin(id: &str) -> Option<LutFile> {
    let label = BUILTIN.iter().find(|b| b.0 == id)?.1;
    let f: Box<dyn Fn([f32; 3]) -> [f32; 3]> = match id {
        "warm" => Box::new(|c: [f32; 3]| [c[0] * 0.92 + 0.08, c[1] * 0.97 + 0.02, c[2] * 0.85]),
        "cool" => Box::new(|c: [f32; 3]| [c[0] * 0.86, c[1] * 0.97 + 0.01, c[2] * 0.9 + 0.1]),
        "tealOrange" => Box::new(|c: [f32; 3]| {
            // Shadows toward teal, highlights toward orange, skin-ish midtones kept.
            let l = luma(c);
            let t = smooth(l);
            let teal = [0.0, 0.5, 0.55];
            let orange = [1.0, 0.62, 0.3];
            std::array::from_fn(|k| {
                let tint = teal[k] * (1.0 - t) + orange[k] * t;
                c[k] * 0.75 + (tint * l * 1.1).min(1.0) * 0.25 + (smooth(c[k]) - c[k]) * 0.3
            })
        }),
        "bleachBypass" => Box::new(|c: [f32; 3]| {
            let l = luma(c);
            let s = smooth(l);
            std::array::from_fn(|k| {
                let desat = c[k] * 0.45 + l * 0.55;
                desat * 0.5 + s * 0.5
            })
        }),
        "fadedFilm" => Box::new(|c: [f32; 3]| {
            let l = luma(c);
            let tint = [1.0, 0.96, 0.88];
            std::array::from_fn(|k| (0.08 + (c[k] * 0.8 + l * 0.2) * 0.84) * tint[k] + 0.02)
        }),
        "dayForNight" => Box::new(|c: [f32; 3]| {
            let l = luma(c);
            [l * 0.35, l * 0.45 + c[1] * 0.05, l * 0.65 + c[2] * 0.15]
        }),
        "monoContrast" => Box::new(|c: [f32; 3]| [smooth(smooth(luma(c))); 3]),
        _ /* crossProcess */ => Box::new(|c: [f32; 3]| [smooth(c[0]) * 1.05, c[1] * 1.1 - 0.03, c[2] * 0.7 + 0.15]),
    };
    Some(LutFile::from_fn(label, 33, f))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_domain_is_kept_and_validated() {
        let base = "TITLE \"t\"
LUT_3D_SIZE 2
";
        let rows = "0 0 0
0 0 1
0 1 0
0 1 1
1 0 0
1 0 1
1 1 0
1 1 1
";
        // The default domain parses as default.
        let d = parse_cube(&format!("{base}{rows}")).expect("default domain");
        assert!(d.domain_is_default());
        // DOMAIN_MIN/MAX are kept for the caller to act on.
        let d = parse_cube(&format!(
            "DOMAIN_MIN 0.1 0.2 0.3
DOMAIN_MAX 0.9 0.8 0.7
{base}{rows}"
        ))
        .expect("custom domain");
        assert_eq!(d.domain_min, [0.1, 0.2, 0.3]);
        assert_eq!(d.domain_max, [0.9, 0.8, 0.7]);
        assert!(!d.domain_is_default());
        // A degenerate domain is a parse error, not a division by zero later.
        let e = parse_cube(&format!(
            "DOMAIN_MIN 0.5 0.5 0.5
DOMAIN_MAX 0.5 1 1
{base}{rows}"
        ))
        .unwrap_err();
        assert!(e.0.contains("below DOMAIN_MAX"), "{e}");
    }

    #[test]
    fn cube_1d_domain_is_baked_into_the_resample() {
        // An identity ramp over the domain 0.25..0.75: 0.25 maps to 0, 0.5 to 0.5, 0.75 to 1.
        let table = "LUT_1D_SIZE 2\nLUT_1D_INPUT_RANGE 0.25 0.75\n0 0 0\n1 1 1\n";
        let l = parse_cube(table).expect("1D with input range");
        assert!(l.domain_is_default(), "the domain is baked in, so the cube is plain 0..1");
        let at = |r: usize| l.data[r * 3];
        // 33-point grid, red fastest: entry r on the red axis is input r / 32.
        assert!(at(0).abs() < 1e-6, "below the domain saturates to 0");
        assert!((at(16) - 0.5).abs() < 1e-5, "0.5 is the middle of 0.25..0.75: {}", at(16));
        assert!((at(8) - 0.0).abs() < 1e-5, "0.25 maps to 0: {}", at(8));
        assert!((at(24) - 1.0).abs() < 1e-5, "0.75 maps to 1: {}", at(24));
        assert!((at(32) - 1.0).abs() < 1e-6, "above the domain saturates to 1");
        // Per-channel DOMAIN_MIN/MAX work the same way.
        let l = parse_cube("LUT_1D_SIZE 2\nDOMAIN_MIN 0 0 0.5\nDOMAIN_MAX 1 1 1\n0 0 0\n1 1 1\n").expect("1D domain");
        let blue_mid = l.data[(16 * 33 * 33) * 3 + 2];
        assert!(blue_mid.abs() < 1e-5, "blue 0.5 is the bottom of its domain: {blue_mid}");
        // A degenerate 1D domain is rejected like the 3D one.
        let e = parse_cube("LUT_1D_SIZE 2\nLUT_1D_INPUT_RANGE 1 1\n0 0 0\n1 1 1\n").unwrap_err();
        assert!(e.0.contains("below DOMAIN_MAX"), "{e}");
    }

    #[test]
    fn cube_roundtrip_and_identity() {
        let id = LutFile::identity(5);
        let text = write_cube(&LutFile { title: "Id".into(), ..id.clone() });
        let back = parse_cube(&text).unwrap();
        assert_eq!(back.size, 5);
        assert_eq!(back.title, "Id");
        for (a, b) in back.data.iter().zip(&id.data) {
            assert!((a - b).abs() < 1e-5);
        }
        // Red varies fastest: entry 1 is (0.25, 0, 0).
        assert_eq!(&back.data[3..6], &[0.25, 0.0, 0.0]);
    }

    #[test]
    fn cube_1d_expands() {
        let text = "LUT_1D_SIZE 2\n0 0 0\n0.5 1 1\n";
        let l = parse_cube(text).unwrap();
        assert_eq!(l.size, 33);
        let last = &l.data[l.data.len() - 3..];
        assert!((last[0] - 0.5).abs() < 1e-6 && (last[1] - 1.0).abs() < 1e-6);

        let mut larger = String::from("LUT_1D_SIZE 256\n");
        for i in 0..256 {
            let value = i as f32 / 255.0;
            larger.push_str(&format!("{value} {value} {value}\n"));
        }
        assert_eq!(parse_cube(&larger).unwrap().size, 33);
    }

    #[test]
    fn cube_1d_size_multiplication_overflow_is_rejected() {
        let text = format!("LUT_1D_SIZE {}\n", usize::MAX);
        assert!(parse_cube(&text).is_err());
    }

    #[test]
    fn three_dl_blue_fastest_and_bit_depth() {
        let mut text = String::from("0 1023\n");
        for r in 0..2 {
            for g in 0..2 {
                for b in 0..2 {
                    text.push_str(&format!("{} {} {}\n", r * 4095, g * 4095, b * 4095));
                }
            }
        }
        let l = parse_3dl(&text).unwrap();
        assert_eq!(l.size, 2);
        assert_eq!(l.data, LutFile::identity(2).data);
    }

    #[test]
    fn look_hex_floats() {
        let id = LutFile::identity(2);
        let hex: String = id.data.iter().flat_map(|v| v.to_le_bytes()).map(|b| format!("{b:02X}")).collect();
        let xml = format!("<?xml version=\"1.0\"?><look><LUT><size>\"2\"</size><data>\"{hex}\"</data></LUT></look>");
        assert_eq!(parse("a.look", xml.as_bytes()).unwrap().data, id.data);
    }

    #[test]
    fn look_3d_size_exponent_overflow_is_rejected() {
        let text = format!("<look><size>{}</size><data>00</data></look>", usize::MAX);
        assert!(parse_look(&text).is_err());
    }

    #[test]
    fn malformed_rejected() {
        assert!(parse_cube("LUT_3D_SIZE 2\n0 0 0\n").is_err());
        assert!(parse_cube("garbage").is_err());
        assert!(parse_cube("LUT_3D_SIZE 99999\n").is_err());
        assert!(parse_3dl("1 2 3\n4 5 6\n").is_err());
        assert!(parse_look("<look></look>").is_err());
    }

    #[test]
    fn builtins_exist_and_are_in_range() {
        for (id, _) in BUILTIN {
            let l = builtin(id).unwrap();
            assert_eq!(l.data.len(), 33 * 33 * 33 * 3);
            assert!(l.data.iter().all(|v| (0.0..=1.0).contains(v)), "{id}");
        }
        assert!(builtin("nope").is_none());
    }
}
