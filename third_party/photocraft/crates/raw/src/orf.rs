//! Olympus ORF with uncompressed sensor data (older Four Thirds bodies such as
//! the E-1 and E-400).
//!
//! The container is TIFF with an `IIRO` / `IIRS` / `MMOR` header; the sensor
//! data is IFD0's strips, 16-bit samples whose top ValidBits bits are
//! significant. The CFA
//! comes from the EXIF CFAPattern. Levels come from the publicly documented
//! Olympus maker-note ImageProcessing tags (ExifTool's Olympus table):
//! WB_RBLevels (0x0100), WB_GLevel (0x011F), BlackLevel2 (0x0600), ValidBits
//! (0x0611) and CropLeft / CropTop / CropWidth / CropHeight (0x0612–0x0615).
//!
//! Later bodies store Olympus' own compressed data, which has no public
//! description and is not decoded.

use crate::error::{RawError, Result};
use crate::sensor::{BlackLevels, JpegLayout, Rect, Sensor, read_plane};
use crate::tiff::{Ifd, Tiff, tag};
use crate::tiffep::{cfa, rggb_by_position};
use crate::{Limits, RawFormat};

const CAMERA_SETTINGS: u16 = 0x2020;
const PREVIEW_IMAGE_START: u16 = 0x0101;
const PREVIEW_IMAGE_LENGTH: u16 = 0x0102;
const IMAGE_PROCESSING: u16 = 0x2040;
const WB_RB_LEVELS: u16 = 0x0100;
const WB_G_LEVEL: u16 = 0x011F;
const BLACK_LEVEL_2: u16 = 0x0600;
const VALID_BITS: u16 = 0x0611;
const CROP: [u16; 4] = [0x0612, 0x0613, 0x0614, 0x0615];

/// A sub-IFD of the Olympus maker note (`sub`: 0x2020 CameraSettings, 0x2040
/// ImageProcessing…), with the TIFF view its offsets are relative to and that
/// view's absolute position in the file. New-style notes (`OLYMPUS\0` + byte
/// order) are relative to the note; old-style ones (`OLYMP\0`) to the file.
fn maker_subifd<'a>(t: &Tiff<'a>, ifds: &[Ifd], sub: u16) -> Option<(Tiff<'a>, Ifd, usize)> {
    let e = ifds.iter().find_map(|i| i.get(tag::MAKER_NOTE).copied())?;
    let head = t.bytes(e.at, 12)?;
    let (view, main, base) = if head.starts_with(b"OLYMPUS\0") {
        let le = match head.get(8..10)? {
            b"II" => true,
            b"MM" => false,
            _ => return None,
        };
        let inner = Tiff { data: t.data.get(e.at..)?, le, first_ifd: 12 };
        let ifd = inner.ifd_at(12, 0)?;
        (inner, ifd, e.at)
    } else if head.starts_with(b"OLYMP\0") {
        (*t, t.ifd_at(e.at.checked_add(8)?, 0)?, 0)
    } else {
        return None;
    };
    let ip = main.get(sub)?;
    let at = match ip.typ {
        4 | 13 => view.uint(ip, 0)? as usize,
        _ => ip.at,
    };
    let ifd = view.ifd_at(at, 0)?;
    Some((view, ifd, base))
}

/// Absolute offset and length of the camera's preview JPEG (maker note
/// CameraSettings PreviewImageStart / PreviewImageLength, 0x0101 / 0x0102).
pub(crate) fn preview_range(t: &Tiff) -> Option<(usize, usize)> {
    let (view, cs, base) = maker_subifd(t, &t.all_ifds(), CAMERA_SETTINGS)?;
    let start = view.tag_uint(&cs, PREVIEW_IMAGE_START)? as usize;
    let len = view.tag_uint(&cs, PREVIEW_IMAGE_LENGTH)? as usize;
    Some((base.checked_add(start)?, len))
}

pub(crate) fn decode(t: &Tiff, limits: &Limits) -> Result<Sensor> {
    let ifds = t.all_ifds();
    let raw = t.ifd_at(t.first_ifd, 0).ok_or_else(|| RawError::malformed("ORF has no IFD0"))?;
    let width = u64::from(t.tag_uint(&raw, tag::IMAGE_WIDTH).unwrap_or(0));
    let height = u64::from(t.tag_uint(&raw, tag::IMAGE_LENGTH).unwrap_or(0));
    // Every ORF, compressed or not, declares its strip byte counts. Missing or unreadable counts
    // mean IFD0 is cut short (a partial download or a damaged file), not the compressed variant:
    // an empty sum must not be mistaken for "fewer than 16 bits per sample".
    let counts = t.tag_uints(&raw, tag::STRIP_BYTE_COUNTS);
    if counts.is_empty() {
        return Err(RawError::malformed("ORF IFD0 has no readable strip byte counts"));
    }
    let stored: u64 = counts.iter().map(|&c| u64::from(c)).sum();
    if t.tag_uint(&raw, tag::COMPRESSION).unwrap_or(1) != 1 || stored < width.saturating_mul(height).saturating_mul(2) {
        return Err(RawError::unsupported("Olympus compressed (or packed) ORF is not decoded"));
    }
    let plane = read_plane(t, &raw, limits, JpegLayout::Flat)?;
    if plane.samples != 1 {
        return Err(RawError::unsupported(format!("ORF with {} samples per pixel", plane.samples)));
    }
    let cfa = cfa(t, &raw, &ifds).ok_or_else(|| RawError::unsupported("ORF: CFA pattern not recorded"))?;
    if !cfa.is_bayer() {
        return Err(RawError::unsupported("ORF: non-Bayer CFA"));
    }
    let full = Rect::new(0, 0, plane.width, plane.height);
    let mut warnings = Vec::new();
    let ip = maker_subifd(t, &ifds, IMAGE_PROCESSING);
    let floats = |tg: u16| ip.as_ref().map(|(v, i, _)| v.tag_floats(i, tg)).unwrap_or_default();

    let black = match rggb_by_position(&cfa, &floats(BLACK_LEVEL_2)) {
        Some(v) => BlackLevels { rows: 2, cols: 2, values: v.to_vec(), delta_h: Vec::new(), delta_v: Vec::new() },
        None => {
            warnings.push("ORF: black level not recorded; assumed 0".to_string());
            BlackLevels::uniform(0.0)
        }
    };
    let mut data = plane.data;
    let white = match floats(VALID_BITS).first() {
        Some(&b) if (8.0..=16.0).contains(&b) => {
            let bits = b as u32;
            let top = (1u32 << bits) - 1;
            // Observed: the E-1 / E-400 store the valid bits left-justified in
            // the 16-bit container (samples are multiples of 16 up to 65520).
            if data.iter().any(|&v| u32::from(v) > top) {
                let shift = 16 - bits;
                data.iter_mut().for_each(|v| *v >>= shift);
            }
            top as f32
        }
        _ => crate::cr2::clip_level(&data, plane.bits),
    };
    let g = floats(WB_G_LEVEL).first().copied().filter(|g| *g > 0.0).unwrap_or(256.0);
    let camera_wb = match floats(WB_RB_LEVELS).as_slice() {
        [r, b, ..] => Some([r / g, 1.0, b / g]).filter(|m| m.iter().all(|v| (0.25..8.0).contains(v))),
        _ => None,
    };
    let crop = match CROP.map(|tg| floats(tg).first().copied()) {
        [Some(x), Some(y), Some(w), Some(h)] if [x, y, w, h].iter().all(|v| (0.0..1e6).contains(v)) => {
            let r = Rect::new(x as usize, y as usize, w as usize, h as usize).intersect(&full);
            if r.is_empty() { full } else { r }
        }
        _ => full,
    };
    Ok(Sensor {
        format: RawFormat::Orf,
        make: ifds.iter().find_map(|i| t.tag_ascii(i, tag::MAKE)),
        model: ifds.iter().find_map(|i| t.tag_ascii(i, tag::MODEL)),
        width: plane.width,
        height: plane.height,
        samples: 1,
        data,
        cfa: Some(cfa),
        linearization: None,
        black,
        white: [white; 3],
        active: full,
        crop,
        color: Default::default(),
        camera_wb,
        orientation: crate::tiff::orientation(t.tag_uint(&raw, tag::ORIENTATION)),
        baseline_exposure: 0.0,
        gain_maps: Vec::new(),
        tone_curve: Vec::new(),
        black_point: 0.0,
        warnings,
    })
}
