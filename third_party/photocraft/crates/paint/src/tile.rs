//! Grayscale bitmaps for sampled tips and texture patterns, with a compact serde form.
//!
//! Values are 16-bit (0..65535 ↔ 0..1) so presets stay small; JSON carries them as base64 of the
//! little-endian samples (`"data": "…"`), and also accepts a plain array of 0..1 floats.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A grayscale bitmap, row-major, 1 = full.
///
/// The samples are shared: cloning a tile (brush settings are cloned on every pointer event of a
/// stroke, #4534) copies a pointer, not a sampled tip that may be thousands of pixels across.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GrayTile {
    pub width: u32,
    pub height: u32,
    pub data: std::sync::Arc<[u16]>,
}

impl GrayTile {
    /// Build from a function returning 0..1 values.
    pub fn from_fn(width: u32, height: u32, f: impl Fn(u32, u32) -> f32) -> Self {
        let mut data = Vec::with_capacity(width as usize * height as usize);
        for y in 0..height {
            for x in 0..width {
                data.push(q16(f(x, y)));
            }
        }
        Self { width, height, data: data.into() }
    }
    /// Build from 0..1 floats (`width × height`).
    pub fn from_f32(width: u32, height: u32, v: &[f32]) -> Self {
        assert_eq!(v.len(), width as usize * height as usize, "GrayTile size mismatch");
        Self { width, height, data: v.iter().map(|&x| q16(x)).collect() }
    }
    #[inline]
    pub fn get(&self, x: u32, y: u32) -> f32 {
        f32::from(self.data[(y * self.width + x) as usize]) / 65535.0
    }
    pub fn to_f32(&self) -> Vec<f32> {
        self.data.iter().map(|&v| f32::from(v) / 65535.0).collect()
    }
    pub fn is_valid(&self) -> bool {
        self.width > 0 && self.height > 0 && self.data.len() == self.width as usize * self.height as usize
    }

    /// A box-filtered copy whose larger side is at most `max_side` (the whole tile when it is
    /// already that small), quantised to 8-bit levels so it stores compactly. Thumbnails and the
    /// Brushes panel draw this instead of a full tip that may be thousands of pixels across.
    /// Deterministic: the same tile always gives the same preview. An invalid tile gives 1×1.
    pub fn preview(&self, max_side: u32) -> GrayTile {
        let q8 = |v: u32| ((v + 128) / 257 * 257) as u16;
        if !self.is_valid() {
            return GrayTile { width: 1, height: 1, data: vec![0].into() };
        }
        let (w, h) = (self.width as usize, self.height as usize);
        let max = max_side.max(1) as usize;
        let f = w.max(h).div_ceil(max).max(1);
        let (nw, nh) = (w.div_ceil(f), h.div_ceil(f));
        let mut sum = vec![0u64; nw * nh];
        let mut cnt = vec![0u32; nw * nh];
        for (y, row) in self.data.chunks_exact(w).enumerate() {
            let base = (y / f) * nw;
            for (x, v) in row.iter().enumerate() {
                if let (Some(s), Some(c)) = (sum.get_mut(base + x / f), cnt.get_mut(base + x / f)) {
                    *s += u64::from(*v);
                    *c += 1;
                }
            }
        }
        let data: Vec<u16> = sum.iter().zip(&cnt).map(|(s, c)| q8(if *c > 0 { (*s / u64::from(*c)) as u32 } else { 0 })).collect();
        let data = data.into();
        GrayTile { width: nw as u32, height: nh as u32, data }
    }
}

/// Larger side of the previews kept for tips that are loaded on demand ([`StoredTile`]).
pub const PREVIEW_SIDE: u32 = 96;

/// A tip or texture bitmap kept outside the brush settings by its owner and loaded only when the
/// brush is used (#1843): the desktop preset store keeps imported tips on disk, so a library of a
/// thousand big sampled brushes costs a few small previews in memory, not gigabytes of bitmaps.
///
/// `preview` is a small copy ([`GrayTile::preview`]) for thumbnails and stroke previews; `full`
/// is the bitmap once the owner has loaded it (shared with the owner's cache, never serialised).
/// Brush rendering uses [`StoredTile::bitmap`]: the full bitmap, or the preview when the owner
/// could not load it, so a missing file degrades the stroke instead of failing it.
///
/// Two stored tiles are equal when their key and size are (the key names the content).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StoredTile {
    /// The owner's key for the bitmap (the preset store uses the content hash).
    pub key: String,
    pub width: u32,
    pub height: u32,
    pub preview: std::sync::Arc<GrayTile>,
    #[serde(skip)]
    pub full: Option<std::sync::Arc<GrayTile>>,
}

impl StoredTile {
    /// The full bitmap when loaded, else the preview.
    pub fn bitmap(&self) -> &GrayTile {
        self.full.as_deref().unwrap_or(&self.preview)
    }
    pub fn is_loaded(&self) -> bool {
        self.full.is_some()
    }
}

impl PartialEq for StoredTile {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.width == other.width && self.height == other.height
    }
}

#[inline]
fn q16(v: f32) -> u16 {
    (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16
}

// ---------- serde ----------

#[derive(Serialize)]
struct GrayTileOut<'a> {
    width: u32,
    height: u32,
    data: &'a str,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum DataIn {
    B64(String),
    Floats(Vec<f32>),
}

#[derive(Deserialize)]
struct GrayTileIn {
    width: u32,
    height: u32,
    data: DataIn,
}

impl Serialize for GrayTile {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut bytes = Vec::with_capacity(self.data.len() * 2);
        for v in self.data.iter() {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let enc = b64_encode(&bytes);
        GrayTileOut { width: self.width, height: self.height, data: &enc }.serialize(s)
    }
}

impl<'de> Deserialize<'de> for GrayTile {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let t = GrayTileIn::deserialize(d)?;
        let n = t.width as usize * t.height as usize;
        let data: Vec<u16> = match t.data {
            DataIn::B64(s) => {
                let b = b64_decode(&s).ok_or_else(|| D::Error::custom("invalid base64 in tile data"))?;
                b.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
            }
            DataIn::Floats(v) => v.iter().map(|&x| q16(x)).collect(),
        };
        if n == 0 || data.len() != n {
            return Err(D::Error::custom(format!("tile data has {} samples, expected {}×{}", data.len(), t.width, t.height)));
        }
        Ok(GrayTile { width: t.width, height: t.height, data: data.into() })
    }
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 with padding (the tile serde form; commands taking file bytes).
pub fn b64_encode(b: &[u8]) -> String {
    let mut out = String::with_capacity(b.len().div_ceil(3) * 4);
    for c in b.chunks(3) {
        let n = (u32::from(c[0]) << 16) | (u32::from(*c.get(1).unwrap_or(&0)) << 8) | u32::from(*c.get(2).unwrap_or(&0));
        out.push(B64[(n >> 18) as usize & 63] as char);
        out.push(B64[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 { B64[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if c.len() > 2 { B64[n as usize & 63] as char } else { '=' });
    }
    out
}

/// Decode standard base64 (whitespace ignored); `None` on invalid input.
pub fn b64_decode(s: &str) -> Option<Vec<u8>> {
    let val = |c: u8| -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        } as u32)
    };
    let s: Vec<u8> = s.bytes().filter(|c| !c.is_ascii_whitespace()).collect();
    if !s.len().is_multiple_of(4) {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    for c in s.chunks(4) {
        let pad = c.iter().rev().take_while(|&&x| x == b'=').count();
        let mut n = 0u32;
        for (i, &x) in c.iter().enumerate() {
            n = (n << 6) | if i >= 4 - pad { 0 } else { val(x)? };
        }
        out.push((n >> 16) as u8);
        if pad < 2 {
            out.push((n >> 8) as u8);
        }
        if pad < 1 {
            out.push(n as u8);
        }
    }
    Some(out)
}

// ---------- sampling ----------

/// A float mip pyramid of a [`GrayTile`] for sampling at any scale without aliasing.
#[derive(Clone, Debug)]
pub struct Mips {
    /// (width, height, values), level 0 = full resolution.
    pub levels: Vec<(usize, usize, Vec<f32>)>,
}

impl Mips {
    pub fn new(t: &GrayTile) -> Self {
        let mut levels = vec![(t.width as usize, t.height as usize, t.to_f32())];
        while let Some((w, h, v)) = levels.last() {
            if *w <= 1 && *h <= 1 {
                break;
            }
            let (nw, nh) = ((*w).div_ceil(2), (*h).div_ceil(2));
            let mut nv = vec![0.0f32; nw * nh];
            for y in 0..nh {
                for x in 0..nw {
                    let mut s = 0.0;
                    let mut c = 0.0;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let (sx, sy) = (x * 2 + dx, y * 2 + dy);
                            if sx < *w && sy < *h {
                                s += v[sy * w + sx];
                                c += 1.0;
                            }
                        }
                    }
                    nv[y * nw + x] = s / c;
                }
            }
            levels.push((nw, nh, nv));
        }
        Self { levels }
    }

    /// Mip level for drawing the tile at `px` pixels across its larger side.
    pub fn level_for(&self, px: f32) -> usize {
        let (w, h, _) = &self.levels[0];
        let ratio = (*w.max(h) as f32 / px.max(1.0)).max(1.0);
        (ratio.log2().floor() as usize).min(self.levels.len() - 1)
    }

    /// Bilinear sample at normalised `u, v` in 0..1 (outside = 0).
    #[inline]
    pub fn sample_clamped(&self, level: usize, u: f32, v: f32) -> f32 {
        let (w, h, d) = &self.levels[level];
        let fx = u * *w as f32 - 0.5;
        let fy = v * *h as f32 - 0.5;
        let (x0, y0) = (fx.floor(), fy.floor());
        let (tx, ty) = (fx - x0, fy - y0);
        let (x0, y0) = (x0 as i64, y0 as i64);
        let get = |x: i64, y: i64| if x < 0 || y < 0 || x >= *w as i64 || y >= *h as i64 { 0.0 } else { d[y as usize * w + x as usize] };
        let a = get(x0, y0) + (get(x0 + 1, y0) - get(x0, y0)) * tx;
        let b = get(x0, y0 + 1) + (get(x0 + 1, y0 + 1) - get(x0, y0 + 1)) * tx;
        a + (b - a) * ty
    }
}

/// A tiling float pattern (texture), sampled with wrap-around.
#[derive(Clone, Debug)]
pub struct PatternImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl PatternImage {
    #[inline]
    pub fn sample_wrap(&self, x: f32, y: f32) -> f32 {
        let xi = (x.floor() as i64).rem_euclid(self.width as i64) as usize;
        let yi = (y.floor() as i64).rem_euclid(self.height as i64) as usize;
        self.data[yi * self.width + xi]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cloning a tile (and the brush settings holding it, on every pointer event of a stroke)
    /// shares its samples instead of copying them (#4534).
    #[test]
    fn clones_share_their_samples() {
        let t = GrayTile::from_fn(64, 32, |x, y| ((x + y) % 7) as f32 / 6.0);
        let c = t.clone();
        assert!(std::sync::Arc::ptr_eq(&t.data, &c.data));
        assert_eq!(t, c);
        let brush = crate::brush::BrushSettings { tip: crate::brush::TipShape::Sampled(t.clone()), ..Default::default() };
        let copy = brush.clone();
        let crate::brush::TipShape::Sampled(ct) = &copy.tip else { panic!("sampled") };
        assert!(std::sync::Arc::ptr_eq(&t.data, &ct.data), "the settings' clone shares the tip");
    }

    #[test]
    fn base64_round_trip() {
        for n in 0..10 {
            let b: Vec<u8> = (0..n).map(|i| (i * 37 + 5) as u8).collect();
            assert_eq!(b64_decode(&b64_encode(&b)).unwrap(), b);
        }
    }

    #[test]
    fn tile_serde_round_trip_and_float_form() {
        let t = GrayTile::from_fn(5, 3, |x, y| (x + y) as f32 / 6.0);
        let j = serde_json::to_string(&t).unwrap();
        let back: GrayTile = serde_json::from_str(&j).unwrap();
        assert_eq!(back, t);
        let f: GrayTile = serde_json::from_str(r#"{"width":2,"height":1,"data":[0.0,1.0]}"#).unwrap();
        assert_eq!(f.data, vec![0, 65535].into());
        assert!(serde_json::from_str::<GrayTile>(r#"{"width":2,"height":2,"data":[0.0,1.0]}"#).is_err());
    }

    #[test]
    fn preview_downscales_deterministically_and_tolerates_bad_tiles() {
        let t = GrayTile::from_fn(1000, 400, |x, _| if x < 500 { 1.0 } else { 0.0 });
        let p = t.preview(96);
        assert!(p.is_valid() && p.width <= 96 && p.height <= 96, "{}×{}", p.width, p.height);
        assert_eq!(p, t.preview(96));
        assert!(p.get(0, 0) > 0.99 && p.get(p.width - 1, 0) < 0.01);
        assert!(p.data.iter().all(|v| v % 257 == 0), "8-bit levels");
        // Small tiles keep their size; invalid ones never panic.
        assert_eq!(GrayTile::from_fn(5, 3, |_, _| 0.5).preview(96).width, 5);
        assert!(GrayTile { width: 4, height: 4, data: vec![1].into() }.preview(96).is_valid());
        assert!(GrayTile { width: 0, height: 0, data: vec![].into() }.preview(0).is_valid());
    }

    #[test]
    fn stored_tiles_compare_by_key_and_serialise_loaded_as_sampled() {
        use crate::brush::{Pattern, TipShape};
        use std::sync::Arc;
        let full = GrayTile::from_fn(300, 200, |x, y| ((x + y) % 7) as f32 / 7.0);
        let r = StoredTile { key: "abc".into(), width: 300, height: 200, preview: Arc::new(full.preview(PREVIEW_SIDE)), full: None };
        let loaded = StoredTile { full: Some(Arc::new(full.clone())), ..r.clone() };
        assert_eq!(TipShape::Stored(r.clone()), TipShape::Stored(loaded.clone()));
        assert_ne!(TipShape::Stored(r.clone()), TipShape::Sampled(full.clone()));
        assert_eq!(r.bitmap().width, r.preview.width);
        assert_eq!(loaded.bitmap(), &full);
        // Unloaded: a small reference that reads back as one.
        let j = serde_json::to_value(TipShape::Stored(r.clone())).unwrap();
        assert_eq!(j["stored"]["key"], "abc");
        assert!(j["stored"].get("full").is_none());
        assert_eq!(serde_json::from_value::<TipShape>(j).unwrap(), TipShape::Stored(r.clone()));
        // Loaded: the same JSON as an embedded bitmap.
        assert_eq!(serde_json::to_value(TipShape::Stored(loaded.clone())).unwrap(), serde_json::to_value(TipShape::Sampled(full.clone())).unwrap());
        assert_eq!(serde_json::to_value(Pattern::Stored(loaded)).unwrap(), serde_json::to_value(Pattern::Tile(full)).unwrap());
        assert_eq!(serde_json::to_value(Pattern::Stored(r)).unwrap()["stored"]["width"], 300);
    }

    #[test]
    fn mips_average() {
        let t = GrayTile::from_fn(4, 4, |x, _| if x < 2 { 1.0 } else { 0.0 });
        let m = Mips::new(&t);
        assert_eq!(m.levels.len(), 3);
        assert!((m.levels[2].2[0] - 0.5).abs() < 1e-3);
        assert_eq!(m.level_for(4.0), 0);
        assert_eq!(m.level_for(1.0), 2);
    }
}
