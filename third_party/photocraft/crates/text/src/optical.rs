//! Optical kerning (Photoshop's Character panel › Kerning › Optical): pair spacing computed from
//! the glyph outlines instead of the font's kerning table.
//!
//! Every glyph gets a horizontal *profile*: at [`SAMPLES`] heights across the em, the distance
//! from the start of its advance to its leftmost ink and from its rightmost ink to the end of its
//! advance (em units, size independent, cached per face and glyph). For a pair, the gap at each
//! height is the right distance of the first glyph plus the left distance of the second. Two
//! features describe a pair: its closest approach and the mean white space beyond it (capped,
//! so open counters and overhangs count only so far). The kerning is a linear function of how
//! those differ from the font's own "oo" pair, plus a size term (Photoshop spaces small type
//! looser). The coefficients were fitted (least squares) to Photoshop 2026's optical kerning of
//! 94 pairs in each of Arial, Georgia, Times New Roman and Verdana at 100 pt, and the size term to
//! 12/24/48 pt; the residual is about 8/1000 em RMS for the sans and Georgia, 17 for Times (see
//! the tests). Clean-room: only Photoshop's output positions were observed.

use std::collections::HashMap;

use skrifa::instance::{LocationRef, NormalizedCoord, Size};
use skrifa::outline::{DrawSettings, OutlinePen};
use skrifa::{GlyphId, MetadataProvider};

/// Number of sampled heights per glyph.
pub const SAMPLES: usize = 48;
/// Sampled band, em units above the baseline.
const Y_LO: f32 = -0.25;
const Y_HI: f32 = 1.0;

/// Left/right side distances at each sampled height (em); `None` where the glyph has no ink.
#[derive(Clone, Debug, PartialEq)]
pub struct Profile {
    pub left: [Option<f32>; SAMPLES],
    pub right: [Option<f32>; SAMPLES],
    /// Advance width (em).
    pub advance: f32,
}

impl Profile {
    /// Whether the glyph has any ink (spaces don't).
    pub fn is_empty(&self) -> bool {
        self.left.iter().all(Option::is_none)
    }
}

/// Height of sample `i` (em).
pub fn sample_y(i: usize) -> f32 {
    Y_LO + (Y_HI - Y_LO) * (i as f32 + 0.5) / SAMPLES as f32
}

/// Line segments of a flattened outline (font units).
#[derive(Default)]
struct Flatten {
    segs: Vec<[f32; 4]>,
    start: (f32, f32),
    at: (f32, f32),
}

impl Flatten {
    fn line(&mut self, x: f32, y: f32) {
        self.segs.push([self.at.0, self.at.1, x, y]);
        self.at = (x, y);
    }
}

impl OutlinePen for Flatten {
    fn move_to(&mut self, x: f32, y: f32) {
        self.start = (x, y);
        self.at = (x, y);
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.line(x, y);
    }
    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        let (x0, y0) = self.at;
        for i in 1..=8 {
            let t = i as f32 / 8.0;
            let u = 1.0 - t;
            self.line(u * u * x0 + 2.0 * u * t * cx + t * t * x, u * u * y0 + 2.0 * u * t * cy + t * t * y);
        }
    }
    fn curve_to(&mut self, c0x: f32, c0y: f32, c1x: f32, c1y: f32, x: f32, y: f32) {
        let (x0, y0) = self.at;
        for i in 1..=12 {
            let t = i as f32 / 12.0;
            let u = 1.0 - t;
            let (a, b, c, d) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
            self.line(a * x0 + b * c0x + c * c1x + d * x, a * y0 + b * c0y + c * c1y + d * y);
        }
    }
    fn close(&mut self) {
        let (x, y) = self.start;
        if (x, y) != self.at {
            self.line(x, y);
        }
    }
}

/// Computes the profile of glyph `gid` (`None` if the font or glyph can't be read).
pub fn profile(font: &skrifa::FontRef<'_>, coords: &[NormalizedCoord], gid: u32) -> Option<Profile> {
    let loc = LocationRef::new(coords);
    let upem = f32::from(font.metrics(Size::unscaled(), loc).units_per_em);
    if upem.is_nan() || upem <= 0.0 {
        return None;
    }
    let advance = font.glyph_metrics(Size::unscaled(), loc).advance_width(GlyphId::new(gid)).unwrap_or(0.0) / upem;
    let mut pen = Flatten::default();
    let outline = font.outline_glyphs().get(GlyphId::new(gid))?;
    outline.draw(DrawSettings::unhinted(Size::unscaled(), loc), &mut pen).ok()?;
    let mut left = [None; SAMPLES];
    let mut right = [None; SAMPLES];
    for i in 0..SAMPLES {
        let y = sample_y(i) * upem;
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &[x0, y0, x1, y1] in &pen.segs {
            if (y0 <= y && y < y1) || (y1 <= y && y < y0) {
                let x = x0 + (x1 - x0) * (y - y0) / (y1 - y0);
                lo = lo.min(x);
                hi = hi.max(x);
            }
        }
        if lo.is_finite() && hi.is_finite() {
            left[i] = Some(lo / upem);
            right[i] = Some(advance - hi / upem);
        }
    }
    Some(Profile { left, right, advance })
}

/// Band of heights used for pair gaps (em above the baseline).
const BAND: (f32, f32) = (-0.1, 1.0);
/// How far (em) beyond the closest approach a gap still counts as white space.
const DEPTH: f32 = 0.04;
/// Closest approaches above this (em) stop opening the pair further.
const NEAR: f32 = 0.1;
/// Fitted coefficients (see the module docs): intercept, closest approach, white space beyond
/// it, saturated closest approach.
const C0: f32 = -0.0033;
const C_MIN: f32 = 0.5384;
const C_EXCESS: f32 = 0.5526;
const C_NEAR: f32 = -0.1612;
/// Optical kerning never moves a pair further than this (1/1000 em).
const LIMIT: f32 = 400.0;

/// Shape features of a pair: closest approach and mean white space beyond it (capped at
/// [`DEPTH`]), em. `None` when the glyphs share no inked height inside the band.
pub fn pair_features(l: &Profile, r: &Profile) -> Option<(f32, f32)> {
    let mut gaps = [0.0f32; SAMPLES];
    let mut n = 0;
    for i in 0..SAMPLES {
        let y = sample_y(i);
        if y < BAND.0 || y > BAND.1 {
            continue;
        }
        if let (Some(a), Some(b)) = (l.right[i], r.left[i]) {
            gaps[n] = a + b;
            n += 1;
        }
    }
    let gaps = gaps.get(..n).filter(|g| !g.is_empty())?;
    let min = gaps.iter().copied().fold(f32::INFINITY, f32::min);
    let excess = gaps.iter().map(|g| (g - min).min(DEPTH)).sum::<f32>() / n as f32;
    Some((min, excess))
}

/// Photoshop's optical kerning is looser at small sizes: pairs open by about
/// 11 × ln(95 / size) thousandths of an em below 95 pt (measured at 12, 24, 48, 96 and 100 pt;
/// flat below 15 pt here, where the measurements level off).
pub fn size_adjust(size_pt: f32) -> f32 {
    if !size_pt.is_finite() || size_pt <= 0.0 {
        return 0.0;
    }
    11.0 * (95.0 / size_pt.clamp(15.0, 95.0)).ln()
}

/// Optical kerning (1/1000 em) of a pair with features `f`, for a font whose "oo" pair has
/// features `reference`, at 95 pt and above (see [`size_adjust`]).
pub fn kern_from_features(f: (f32, f32), reference: (f32, f32)) -> f32 {
    let k = C0 + C_MIN * (reference.0 - f.0) + C_EXCESS * (reference.1 - f.1) + C_NEAR * (reference.0.min(NEAR) - f.0.min(NEAR));
    let k = k * 1000.0;
    if k.is_finite() { k.clamp(-LIMIT, LIMIT) } else { 0.0 }
}

/// Per-font cache of glyph profiles and reference features. Keyed by the caller's font id
/// (parley's blob id plus face index) and variation coordinates.
#[derive(Default)]
pub struct Cache {
    profiles: HashMap<(u64, u32), Option<Profile>>,
    reference: HashMap<u64, Option<(f32, f32)>>,
}

impl Cache {
    /// Entries kept before the cache is cleared (bounds memory with many fonts).
    const MAX_PROFILES: usize = 50_000;

    fn font_key(font_id: u64, index: u32, coords: &[NormalizedCoord]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (font_id, index).hash(&mut h);
        for c in coords {
            c.to_bits().hash(&mut h);
        }
        h.finish()
    }

    fn get(&mut self, fk: u64, font: &skrifa::FontRef<'_>, coords: &[NormalizedCoord], gid: u32) -> Option<Profile> {
        if self.profiles.len() > Self::MAX_PROFILES {
            self.profiles.clear();
        }
        self.profiles.entry((fk, gid)).or_insert_with(|| profile(font, coords, gid)).clone()
    }

    /// Features of the font's "oo" pair (its own idea of round-to-round spacing).
    fn reference(&mut self, fk: u64, font: &skrifa::FontRef<'_>, coords: &[NormalizedCoord]) -> Option<(f32, f32)> {
        if let Some(r) = self.reference.get(&fk) {
            return *r;
        }
        let r = ['o', 'O', 'n', 'H'].iter().find_map(|&c| {
            let g = font.charmap().map(c)?;
            let p = self.get(fk, font, coords, g.to_u32())?;
            pair_features(&p, &p)
        });
        self.reference.insert(fk, r);
        r
    }

    /// Optical kerning of glyphs `left`, `right` of one face in 1/1000 em, or `None` when it
    /// doesn't apply (spaces, glyphs that share no inked height, unreadable fonts).
    pub fn pair(&mut self, font_id: u64, data: &[u8], index: u32, coords: &[NormalizedCoord], left: u32, right: u32) -> Option<f32> {
        let font = skrifa::FontRef::from_index(data, index).ok()?;
        let fk = Self::font_key(font_id, index, coords);
        let l = self.get(fk, &font, coords, left)?;
        let r = self.get(fk, &font, coords, right)?;
        let f = pair_features(&l, &r)?;
        Some(kern_from_features(f, self.reference(fk, &font, coords)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Text set in Photoshop 2026 at 100 pt with Kerning › Optical; `PHOTOSHOP` holds the
    /// kerning Photoshop applied after each character (1/1000 em: optical positions minus the
    /// unkerned ones, read from the saved PSDs).
    const SAMPLE: &str = "HHOOHnnoonvAVTAToTyLTWaYoPAFAVaxyrnilHIOHnHoxoxcdbpkkwwAAVVff1471LYTeKOnHeHaHsHtHgHzHkHxHvHwrst";
    #[rustfmt::skip]
    const PHOTOSHOP: [(&str, &str, &[i32]); 4] = [
        ("ArialMT", "/System/Library/Fonts/Supplemental/Arial.ttf", &[-16, -19, -20, -17, -20, -27, -22, -17, -20, -44, -52, -84, 15, -108, -110, -118, -122, -123, -18, -130, 14, -50, -111, -117, -15, -84, -12, -86, -84, -73, -6, 1, -19, -7, -27, -24, -23, -24, -22, -17, -20, -23, -15, -44, -42, -44, -45, -6, -28, -23, -23, -17, 7, 16, -40, 34, -84, 26, -23, 15, -34, -76, -74, -88, -69, -141, 15, -120, -16, -67, -13, -23, -17, -19, -18, -22, -19, -21, -20, -20, -16, -21, -28, -29, -21, -19, -22, -22, -24, -24, -19, -18, -11, -28]),
        ("Georgia", "/System/Library/Fonts/Supplemental/Georgia.ttf", &[-28, -51, -18, -53, -24, -12, -38, -17, -37, -36, -79, -110, 33, -44, -42, -115, -115, -113, -46, -61, 22, -84, -62, -102, -50, -109, 15, -91, -110, -95, 11, 16, 13, 2, -12, -6, -17, -23, -53, -53, -24, -16, -50, -41, -40, -41, -37, -10, -28, -38, -37, 12, 7, 31, -63, 49, -110, 47, -39, 13, -14, -6, -58, -10, -20, -59, 36, -110, -40, -51, -19, -16, -50, -39, -33, -10, -26, -40, -58, -10, -37, -15, -18, -18, -18, 2, -7, -6, -49, -46, -53, 13, -11, -19]),
        ("TimesNewRomanPSMT", "/System/Library/Fonts/Supplemental/Times New Roman.ttf", &[3, -38, -17, -38, 4, 11, -26, -16, -24, -36, -96, -159, -3, -70, -74, -114, -117, -122, -44, -105, 0, -93, -114, -137, -36, -120, 8, -117, -159, -104, -5, -14, 10, 22, -1, -13, 2, -1, -40, -38, 4, 10, -38, -47, -49, -47, -46, -15, -17, -16, -29, 20, 8, 12, -91, 13, -159, 12, -79, -2, 14, -98, -65, -71, -55, -133, -4, -116, -29, -67, -3, 10, -36, -29, -22, -4, -16, -28, -42, -5, -35, -1, 0, -3, 7, 16, 2, 6, -46, -44, -44, 8, -7, -22]),
        ("Verdana", "/System/Library/Fonts/Supplemental/Verdana.ttf", &[-33, -26, -21, -27, -35, -35, -27, -20, -28, -52, -66, -82, 33, -93, -93, -120, -120, -121, -32, -102, 16, -67, -103, -104, -26, -72, -20, -70, -82, -59, -27, 4, -30, -12, -37, -37, -35, -47, -56, -27, -35, -33, -26, -53, -53, -53, -57, -16, -37, -28, -30, -19, 0, -10, -49, 26, -82, 26, -20, 17, -38, -98, -87, -69, -72, -102, 38, -120, -22, -78, -24, -33, -28, -20, -33, -33, -23, -22, -29, -27, -28, -35, -41, -37, -37, -19, -32, -32, -32, -32, -38, -37, 5, -15]),
    ];

    /// The fitted model against Photoshop's optical kerning (macOS system fonts; skipped where
    /// they aren't installed). Reports the error per font; fails if it regresses.
    #[test]
    fn optical_matches_photoshop() {
        let chars: Vec<char> = SAMPLE.chars().collect();
        let (mut sum, mut n) = (0.0f64, 0usize);
        for (name, path, ps) in PHOTOSHOP {
            let Ok(data) = std::fs::read(path) else { continue };
            let mut cache = Cache::default();
            let font = skrifa::FontRef::from_index(&data, 0).unwrap();
            let (mut fs, mut fneg, mut fpos) = (0.0f64, 0usize, 0usize);
            for (i, w) in chars.windows(2).enumerate() {
                let g = |c: char| font.charmap().map(c).unwrap().to_u32();
                let ours = cache.pair(1, &data, 0, &[], g(w[0]), g(w[1])).unwrap_or(0.0);
                let theirs = ps[i] as f32;
                fs += f64::from((ours - theirs).powi(2));
                // Direction agreement on clear cases.
                if theirs < -40.0 && ours > 0.0 {
                    fneg += 1;
                }
                if theirs > 15.0 && ours < -15.0 {
                    fpos += 1;
                }
            }
            let rms = (fs / ps.len() as f64).sqrt();
            eprintln!("{name}: rms {rms:.1}/1000 em over {} pairs", ps.len());
            assert!(rms < 20.0, "{name}: rms {rms}");
            // Times' serif stems (kH, kk) are opened by Photoshop and tightened slightly here.
            assert!(fneg + fpos <= 2, "{name}: {} pairs kerned the wrong way", fneg + fpos);
            sum += fs;
            n += ps.len();
        }
        if n > 0 {
            let rms = (sum / n as f64).sqrt();
            eprintln!("all: rms {rms:.1}/1000 em");
            assert!(rms < 12.5, "rms {rms}");
        }
    }

    /// Writes glyph profiles of macOS system fonts as JSON for fitting [`PARAMS`]
    /// (`OPTICAL_DUMP=<file> cargo test -p photocraft-text optical_dump -- --ignored`).
    #[test]
    #[ignore]
    fn optical_dump() {
        let Some(out) = std::env::var_os("OPTICAL_DUMP") else { return };
        let chars = "HOnovAVTyLWaYPFxrilIcdbpkwf1475eKsgzt";
        let mut json = String::from("{");
        for (name, path) in [
            ("ArialMT", "/System/Library/Fonts/Supplemental/Arial.ttf"),
            ("Georgia", "/System/Library/Fonts/Supplemental/Georgia.ttf"),
            ("TimesNewRomanPSMT", "/System/Library/Fonts/Supplemental/Times New Roman.ttf"),
            ("Verdana", "/System/Library/Fonts/Supplemental/Verdana.ttf"),
        ] {
            let data = std::fs::read(path).unwrap();
            let font = skrifa::FontRef::from_index(&data, 0).unwrap();
            json.push_str(&format!("\"{name}\":{{"));
            for c in chars.chars() {
                let g = font.charmap().map(c).unwrap().to_u32();
                let p = profile(&font, &[], g).unwrap();
                let f = |v: &[Option<f32>]| v.iter().map(|x| x.map_or("null".to_string(), |x| format!("{x:.5}"))).collect::<Vec<_>>().join(",");
                json.push_str(&format!("\"{c}\":{{\"adv\":{},\"l\":[{}],\"r\":[{}]}},", p.advance, f(&p.left), f(&p.right)));
            }
            json.pop();
            json.push_str("},");
        }
        json.pop();
        json.push('}');
        std::fs::write(out, json).unwrap();
    }
}
