//! Canon CR2.
//!
//! CR2 is a TIFF file whose header carries `CR`, the major and minor version
//! and the offset of the raw IFD (IFD3) at bytes 8–15. The raw IFD stores one
//! lossless JPEG (T.81 process 14) whose decoded sample sequence fills the
//! sensor image in vertical slices: the private tag 0xC640 gives the slice
//! count, slice width and last slice width; slice 0 is filled row by row,
//! then slice 1, and so on. sRAW/mRAW (subsampled YCbCr) is not supported.
//!
//! Sensor geometry and the as-shot white balance come from the Canon maker
//! note (an IFD at the MakerNote offset, values relative to the TIFF header):
//! tag 0x00E0 (SensorInfo: sensor size and the image borders) and tag
//! 0x4001 (ColorData, whose as-shot RGGB levels sit at a version-dependent
//! position; candidates are validated before use). The black level is
//! measured on the masked border; the white level is the clipping point
//! found in the data.
//!
//! ## Colour filter phase
//!
//! CR2 has no `CFAPattern` tag, and the phase is not a function of the
//! SensorInfo borders: on every model measured, red sits in the even data
//! columns, but whether the even or the odd data rows hold red varies by model
//! and has no relation to the border offsets (EOS 7D, 550D, 600D, 1200D and
//! 5D Mark II start with a green/blue row with their top border at 56; the 40D,
//! 400D, 450D, 500D, 1000D, 1100D, 5D, 1D Mark III and 1Ds Mark III have an odd
//! top border and the 5D Mark III, 6D, 5DS R, 100D, 650D, 700D, 760D and 1D X an
//! even one, and all of them start with a red/green row). Anchoring a fixed
//! pattern at the image-area origin (or at the sensor origin) is therefore
//! wrong for some models whichever anchor is chosen.
//!
//! So the row phase is measured from the data ([`measured_red_row`]): the two
//! green sites of a 2×2 cell sit on one diagonal, carry the same filter, and
//! so have (a) the smaller difference between diagonal neighbours and (b)
//! nearly equal means, while the red/blue diagonal differs by the scene colour
//! and the white-balance gains. Both cues must agree. When they don't (a flat,
//! black or clipped frame) a small table of Canon model IDs whose data starts
//! with a green/blue row decides, then red/green first (the common case).
//!
//! Sources: Phil Harvey's ExifTool Canon tag documentation (MakerNote tag
//! 0x0010 CanonModelID; 0x00E0 SensorInfo: SensorWidth, SensorHeight,
//! SensorLeft/Top/Right/BottomBorder at indices 1, 2, 5–8), and observation of
//! CC0 samples from raw.pixls.us for the 26 models named above (developed with
//! each candidate phase and compared by eye on test charts and natural scenes).

use crate::error::{RawError, Result};
use crate::sensor::{BlackLevels, Cfa, Rect, Sensor};
use crate::tiff::{Ifd, Tiff, tag};
use crate::{Limits, RawFormat, ljpeg};

const CANON_SENSOR_INFO: u16 = 0x00E0;
const CANON_COLOR_DATA: u16 = 0x4001;
const CANON_MODEL_ID: u16 = 0x0010;

/// Canon model IDs (MakerNote 0x0010) whose raw data starts with a green/blue
/// row (red in the odd data rows), from observed samples. Only consulted when
/// the measurement is inconclusive.
const GREEN_FIRST_MODELS: [(u32, &str); 5] = [
    (0x8000_0218, "EOS 5D Mark II"),
    (0x8000_0250, "EOS 7D"),
    (0x8000_0270, "EOS 550D / Rebel T2i / Kiss X4"),
    (0x8000_0286, "EOS 600D / Rebel T3i / Kiss X5"),
    (0x8000_0327, "EOS 1200D / Rebel T5 / Kiss X70"),
];

/// Which data rows hold red (0 = even, 1 = odd), measured over `area`; `None`
/// when the two cues (see the module docs) disagree or there is too little
/// unclipped signal. Samples every other row pair; red is in the even columns.
pub(crate) fn measured_red_row(data: &[u16], width: usize, area: Rect, black: f32, white: f32) -> Option<usize> {
    // Per column parity p of the upper sample: p = 0 is the diagonal (even row, even col) /
    // (odd row, odd col), green when red rows are odd; p = 1 is the other one.
    let mut diff = [0f64; 2];
    let mut upper = [0f64; 2];
    let mut lower = [0f64; 2];
    let mut n = [0u64; 2];
    let (lo, hi) = (black + 1.0, black + (white - black) * 0.98);
    let x_end = area.x.saturating_add(area.width).min(width).saturating_sub(1);
    let y_end = area.y.saturating_add(area.height).saturating_sub(1);
    let row = |y: usize| data.get(y.checked_mul(width)?..y.checked_add(1)?.checked_mul(width)?);
    let mut y = area.y.saturating_add(area.y & 1);
    while y < y_end {
        let (Some(r0), Some(r1)) = (row(y), y.checked_add(1).and_then(row)) else { break };
        for x in area.x..x_end {
            let (Some(&a), Some(&b)) = (r0.get(x), r1.get(x + 1)) else { break };
            let (a, b) = (f32::from(a), f32::from(b));
            if a < lo || b < lo || a > hi || b > hi {
                continue;
            }
            let p = x & 1;
            diff[p] += f64::from((a - b).abs());
            upper[p] += f64::from(a - black);
            lower[p] += f64::from(b - black);
            n[p] += 1;
        }
        y = y.saturating_add(4);
    }
    if n.iter().any(|&k| k < 256) {
        return None;
    }
    let d = [diff[0] / n[0] as f64, diff[1] / n[1] as f64];
    let rel = [0, 1].map(|p| (upper[p] - lower[p]).abs() / (upper[p] + lower[p]).max(1.0));
    if d[0] < 0.8 * d[1] && rel[0] < rel[1] {
        Some(1)
    } else if d[1] < 0.8 * d[0] && rel[1] < rel[0] {
        Some(0)
    } else {
        None
    }
}

/// The Canon maker note IFD, if present.
pub(crate) fn maker_note(t: &Tiff, ifds: &[Ifd]) -> Option<Ifd> {
    let e = ifds.iter().find_map(|i| i.get(tag::MAKER_NOTE).copied())?;
    t.ifd_at(e.at, 0)
}

/// As-shot white-balance multipliers (R, G, B) from ColorData.
fn as_shot_wb(t: &Tiff, mn: &Ifd) -> Option<[f64; 3]> {
    let e = mn.get(CANON_COLOR_DATA)?;
    // The entry is UNDEFINED: its bytes are a table of 16-bit words (file byte order), and the
    // offsets below count words, not bytes. Reads stay inside the entry's own byte count.
    let bytes = e.unit().checked_mul(e.count as usize)?;
    let word = |off: usize| {
        let at = off.checked_mul(2)?.checked_add(e.at)?;
        (at.checked_add(2)? <= e.at.checked_add(bytes)?).then(|| t.u16_at(at)).flatten().map(f64::from)
    };
    // Word offsets of WB_RGGBLevelsAsShot used by the ColorData versions.
    for off in [0x3F, 0x47, 0x19, 0x22] {
        let (Some(r), Some(g1), Some(g2), Some(b)) = (word(off), word(off + 1), word(off + 2), word(off + 3)) else {
            continue;
        };
        let g = (g1 + g2) / 2.0;
        let plausible = (256.0..16384.0).contains(&g) && (g1 - g2).abs() <= g * 0.05 && (0.25..8.0).contains(&(r / g)) && (0.25..8.0).contains(&(b / g));
        if plausible {
            return Some([r / g, 1.0, b / g]);
        }
    }
    None
}

/// The white (clipping) level: the largest value when a meaningful share of
/// samples piles up at it, else the full range of the declared precision.
pub(crate) fn clip_level(data: &[u16], precision: u32) -> f32 {
    let full = ((1u32 << precision.min(16)) - 1) as f32;
    let step = (data.len() / 2_000_000).max(1);
    let mut max = 0u16;
    for v in data.iter().step_by(step) {
        max = max.max(*v);
    }
    if max == 0 {
        return full;
    }
    let near = data.iter().step_by(step).filter(|&&v| u32::from(v) + u32::from(max / 256) >= u32::from(max)).count();
    let sampled = data.len().div_ceil(step);
    if near * 2000 >= sampled {
        // At least 0.05 % of samples at the top: that is the clipping point.
        // Back off slightly so the clipped plateau maps to white.
        f32::from(max) * 0.995
    } else {
        full
    }
}

/// Mean of a region per 2×2 position.
pub(crate) fn masked_black(data: &[u16], width: usize, area: Rect) -> Option<[f32; 4]> {
    if area.is_empty() || area.width < 2 || area.height < 2 {
        return None;
    }
    let mut sum = [0u64; 4];
    let mut n = [0u64; 4];
    for y in area.y..area.y + area.height {
        for x in area.x..area.x + area.width {
            let v = *data.get(y * width + x)?;
            let k = (y & 1) * 2 + (x & 1);
            sum[k] += u64::from(v);
            n[k] += 1;
        }
    }
    if n.contains(&0) {
        return None;
    }
    Some([0, 1, 2, 3].map(|k| sum[k] as f32 / n[k] as f32))
}

pub(crate) fn decode(t: &Tiff, limits: &Limits) -> Result<Sensor> {
    let ifds = t.all_ifds();
    let ifd0 = t.ifd_at(t.first_ifd, 0).ok_or_else(|| RawError::malformed("CR2 has no IFD0"))?;
    let raw_off = t.u32_at(12).unwrap_or(0) as usize;
    let raw = t.ifd_at(raw_off, 0).ok_or_else(|| RawError::malformed("CR2 raw IFD is missing"))?;
    let off = t.tag_uint(&raw, tag::STRIP_OFFSETS).ok_or_else(|| RawError::malformed("CR2 raw data offset missing"))? as usize;
    let len = t.tag_uint(&raw, tag::STRIP_BYTE_COUNTS).map(|l| l as usize).unwrap_or(t.data.len().saturating_sub(off));
    let src = t.bytes(off, len).or_else(|| t.data.get(off..)).ok_or_else(|| RawError::malformed("CR2 raw data lies outside the file"))?;

    let frame = ljpeg::frame(src).map_err(|e| match e {
        RawError::Unsupported(m) if m.contains("subsampled") => RawError::unsupported("Canon sRAW / mRAW"),
        e => e,
    })?;
    let total = frame.samples().ok_or_else(|| RawError::malformed("CR2 frame too large"))?;
    let slices = t.tag_uints(&raw, tag::CR2_SLICE);
    let (width, widths) = match slices.as_slice() {
        [n, w, last] if *n < 64 && *w > 0 && *last > 0 => {
            let (n, w, last) = (*n as usize, *w as usize, *last as usize);
            let mut ws = vec![w; n];
            ws.push(last);
            (n * w + last, ws)
        }
        _ => {
            let w = frame.width * frame.components;
            (w, vec![w])
        }
    };
    if width == 0 || total % width != 0 {
        return Err(RawError::malformed("CR2 slices do not match the JPEG frame"));
    }
    let height = total / width;
    limits.check(width as u64, height as u64, 2)?;
    let (_, samples) = ljpeg::decode(src, total)?;

    // De-slice.
    let mut data = vec![0u16; width * height];
    let mut pos = 0usize;
    let mut x0 = 0usize;
    for sw in widths {
        for y in 0..height {
            let src = samples.get(pos..pos + sw).ok_or_else(|| RawError::malformed("CR2 slice data is short"))?;
            data[y * width + x0..y * width + x0 + sw].copy_from_slice(src);
            pos += sw;
        }
        x0 += sw;
    }

    let mut warnings = Vec::new();
    let mn = maker_note(t, &ifds);
    let full = Rect::new(0, 0, width, height);
    // SensorInfo: [1] width, [2] height, [5] left, [6] top, [7] right, [8] bottom border (inclusive).
    let info = mn.as_ref().map(|m| t.tag_uints(m, CANON_SENSOR_INFO)).unwrap_or_default();
    let active = match info.get(5..9) {
        Some(&[l, tp, r, b]) if r > l && b > tp => Rect::new(l as usize, tp as usize, (r - l + 1) as usize, (b - tp + 1) as usize).intersect(&full),
        _ => full,
    };
    let active = if active.is_empty() { full } else { active };
    // Black: the masked columns left of the image (skipping a few edge columns).
    let black = if active.x >= 16 {
        // Only the outer half: the columns next to the image area can be partly exposed.
        let area = Rect::new(4, active.y, active.x / 2 - 4, active.height);
        masked_black(&data, width, area)
    } else {
        None
    };
    let black = match black {
        Some(b) => BlackLevels { rows: 2, cols: 2, values: b.to_vec(), delta_h: Vec::new(), delta_v: Vec::new() },
        None => {
            warnings.push("no masked sensor area; black level assumed to be 0".to_string());
            BlackLevels::uniform(0.0)
        }
    };
    let white = clip_level(&data, u32::from(frame.precision));
    // Red is in the even data columns; the red rows are measured (see the module docs).
    let mean_black = black.values.iter().sum::<f32>() / black.values.len().max(1) as f32;
    let red_row = match measured_red_row(&data, width, active, mean_black, white) {
        Some(r) => r,
        None => {
            let id = mn.as_ref().and_then(|m| t.tag_uint(m, CANON_MODEL_ID));
            let known = GREEN_FIRST_MODELS.iter().any(|(m, _)| Some(*m) == id);
            if !known {
                warnings.push("CR2 colour filter phase could not be measured; assumed red/green in the first row".to_string());
            }
            usize::from(known)
        }
    };
    let colors = if red_row == 0 { vec![0, 1, 1, 2] } else { vec![1, 2, 0, 1] };
    let cfa = Cfa { width: 2, height: 2, colors, origin_x: 0, origin_y: 0 };
    let camera_wb = mn.as_ref().and_then(|m| as_shot_wb(t, m));
    // The black-level values index from the active area origin, but the masked
    // measurement used data parity: realign them to active-area parity.
    let black = realign_black(black, active);
    Ok(Sensor {
        format: RawFormat::Cr2,
        make: t.tag_ascii(&ifd0, tag::MAKE),
        model: t.tag_ascii(&ifd0, tag::MODEL),
        width,
        height,
        samples: 1,
        data,
        cfa: Some(cfa),
        linearization: None,
        black,
        white: [white; 3],
        active,
        crop: active,
        color: Default::default(),
        camera_wb,
        orientation: crate::tiff::orientation(t.tag_uint(&ifd0, tag::ORIENTATION)),
        baseline_exposure: 0.0,
        gain_maps: Vec::new(),
        tone_curve: Vec::new(),
        black_point: 0.0,
        warnings,
    })
}

/// Black levels measured per data-parity 2×2 position, re-indexed from the active origin.
pub(crate) fn realign_black(b: BlackLevels, active: Rect) -> BlackLevels {
    if b.rows != 2 || b.cols != 2 || b.values.len() != 4 {
        return b;
    }
    let v = &b.values;
    let values = (0..4)
        .map(|k| {
            let (ax, ay) = (k & 1, k >> 1);
            let (x, y) = ((active.x + ax) & 1, (active.y + ay) & 1);
            v[y * 2 + x]
        })
        .collect();
    BlackLevels { values, ..b }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn red_row_measurement_never_panics_on_odd_geometry() {
        let data = vec![1000u16; 64 * 32];
        for area in [
            Rect::new(0, 0, 0, 0),
            Rect::new(0, 0, 64, 32),
            Rect::new(60, 30, 100, 100),
            Rect::new(usize::MAX - 1, 0, 4, 4),
            Rect::new(0, usize::MAX - 1, 4, 4),
        ] {
            // Flat data: inconclusive, never a guess.
            assert_eq!(measured_red_row(&data, 64, area, 0.0, 4000.0), None, "{area:?}");
        }
        assert_eq!(measured_red_row(&[], 64, Rect::new(0, 0, 64, 32), 0.0, 4000.0), None);
        assert_eq!(measured_red_row(&data, 0, Rect::new(0, 0, 64, 32), 0.0, 4000.0), None);
        assert_eq!(measured_red_row(&data, 64, Rect::new(0, 0, 64, 32), f32::NAN, f32::NAN), None);
    }

    #[test]
    fn red_row_measurement_finds_both_phases() {
        // Grey scene under white-balance gains R 2, B 1.5: green 1000, red 500, blue 667, plus texture.
        for red_row in [0usize, 1] {
            let (w, h) = (64usize, 64usize);
            let data: Vec<u16> = (0..w * h)
                .map(|i| {
                    let (x, y) = (i % w, i / w);
                    let tex = 1.0 + 0.2 * (((x / 3) ^ (y / 5)) & 1) as f32;
                    let c = if (y & 1) == red_row { [0, 1][x & 1] } else { [1, 2][x & 1] };
                    (100.0 + [500.0, 1000.0, 667.0][c] * tex) as u16
                })
                .collect();
            assert_eq!(measured_red_row(&data, w, Rect::new(0, 0, w, h), 100.0, 4000.0), Some(red_row));
        }
    }
}
#[cfg(all(test, feature = "corpus"))]
mod wb_debug {

    /// The corpus PowerShot CR2: ColorData's WB_RGGBLevelsAsShot sits at word offset 0x47 (the
    /// PowerShot SX50 HS's ColorData version), the same words its DNG conversion neutralises
    /// (AsShotNeutral -> WB ≈ [1.66, 1.0, 1.74]; the ColorData words give [1.70, 1.0, 1.69] —
    /// the converter re-derives them through the color matrix). The synthetic twin of this
    /// oracle is `color_data_wb_as_words` above; it runs in every CI.
    #[test]
    fn power_shot_color_data_holds_the_as_shot_wb() {
        let Ok(bytes) = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../corpus/pixls/IMG_4059.CR2")) else {
            panic!("corpus/pixls is fetched for corpus runs: cargo xtask corpus --pixls");
        };
        let s = crate::decode(&bytes, &crate::Limits::default()).expect("decode");
        let wb = s.camera_wb.expect("the PowerShot's as-shot WB is read now");
        assert!((wb[0] - 1.696).abs() < 0.01 && wb[1] == 1.0 && (wb[2] - 1.688).abs() < 0.01, "{wb:?}");
        // The file's white balance is used, not an estimate: no grey-world note in the develop.
        let d = crate::develop_sensor(&s, &crate::DevelopOptions::default()).expect("develop");
        assert!(!d.warnings.iter().any(|w| w.contains("estimated automatically")), "{:?}", d.warnings);
    }
}
#[cfg(test)]
mod wb_words {
    use crate::testgen::Cr2Spec;

    /// ColorData arrives as UNDEFINED bytes that are 16-bit words; the offsets count words.
    /// Regression for the byte-scaled lookup that never found the balance (synthetic twin of
    /// the corpus oracle, runs in every CI).
    #[test]
    fn color_data_wb_as_words() {
        let spec = Cr2Spec {
            width: 8,
            height: 8,
            data: vec![2048u16; 64],
            precision: 12,
            components: 1,
            slices: Vec::new(),
            borders: None,
            wb_rggb: Some([1613, 951, 951, 1605]),
            orientation: 1,
            model_id: None,
        };
        let bytes = spec.build();
        let s = crate::decode(&bytes, &crate::Limits::default()).expect("decode");
        let wb = s.camera_wb.expect("the synthetic as-shot WB is read");
        assert!((wb[0] - 1613.0 / 951.0).abs() < 1e-9 && wb[1] == 1.0 && (wb[2] - 1605.0 / 951.0).abs() < 1e-9, "{wb:?}");
    }

    /// A ColorData entry too short for the offsets reads no balance instead of running past it.
    #[test]
    fn short_color_data_reads_no_wb() {
        let spec = Cr2Spec { wb_rggb: None, ..minimal_spec() };
        let bytes = spec.build();
        let s = crate::decode(&bytes, &crate::Limits::default()).expect("decode");
        assert!(s.camera_wb.is_none());
    }

    fn minimal_spec() -> Cr2Spec {
        Cr2Spec {
            width: 8,
            height: 8,
            data: vec![2048u16; 64],
            precision: 12,
            components: 1,
            slices: Vec::new(),
            borders: None,
            wb_rggb: None,
            orientation: 1,
            model_id: None,
        }
    }
}
