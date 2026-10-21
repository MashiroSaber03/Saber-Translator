//! Image › Mode › Multichannel, and conversions out of it.
//!
//! A Multichannel document has no layers: its image is a set of ink channels, kept as spot
//! channels in [`Document::channels`] (value = ink density, 1 = solid; the compositor prints them
//! over white, see `photocraft_compose::multichannel`). Like Photoshop the conversion flattens and
//! turns each colour channel into an ink:
//!
//! * RGB → Cyan, Magenta, Yellow (the red, green and blue data read as ink: density = 1 − value);
//! * CMYK → Cyan, Magenta, Yellow, Black (the plates as they are);
//! * Lab → Alpha 1, Alpha 2, Alpha 3 (lightness, a, b data, printed in black);
//! * Grayscale → Black; Duotone → one channel per ink (its curve applied).
//!
//! Existing alpha channels stay alpha channels. Converting back reinterprets the channel data when
//! the count matches the target (3 → RGB / Lab, 4 → CMYK, 1 → Grayscale), the inverse of the
//! above; otherwise the printed look is converted.

use photocraft_color::{Color, ColorMode, PixelFormat};
use photocraft_doc::{AlphaChannel, Document, Layer, LayerContent};
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

/// Display colours of the process inks (sRGB).
const CYAN: Color = Color::rgb(0.0, 0.682, 0.937);
const MAGENTA: Color = Color::rgb(0.925, 0.0, 0.549);
const YELLOW: Color = Color::rgb(1.0, 0.949, 0.0);
const BLACK: Color = Color::rgb(0.137, 0.122, 0.125);

/// Modes Photoshop converts to Multichannel.
const SOURCES: [ColorMode; 5] = [ColorMode::Rgb, ColorMode::Cmyk, ColorMode::Lab, ColorMode::Grayscale, ColorMode::Duotone];

fn enabled(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    if SOURCES.contains(&d.doc.mode) {
        Ok(())
    } else {
        Err(format!("Multichannel needs an RGB, CMYK, Lab, Grayscale or Duotone image (the document is {:?})", d.doc.mode))
    }
}

/// Native colour planes of the flattened image (opaque over white): one `Vec` per colour
/// channel of the document's pixel format. A lone plain layer is read exactly; anything else
/// goes through the compositor.
fn flattened_planes(doc: &Document) -> Vec<Vec<f32>> {
    let fmt = doc.pixel_format();
    let cc = fmt.mode.color_channels();
    let canvas = doc.bounds();
    let n = canvas.width() as usize * canvas.height() as usize;
    let white = photocraft_raster::from_rgba(&fmt, [1.0, 1.0, 1.0, 1.0]);
    let mut planes = vec![Vec::with_capacity(n); cc];
    let plain = match doc.layers.as_slice() {
        [l] if l.visible
            && l.opacity >= 1.0
            && l.fill_opacity >= 1.0
            && l.mask.is_none()
            && l.vector_mask.is_none()
            && !photocraft_compose::effects::has_effects(l) =>
        {
            match &l.content {
                LayerContent::Raster(s) if s.format().mode == fmt.mode => Some(s),
                _ => None,
            }
        }
        _ => None,
    };
    if let Some(s) = plain {
        let ch = s.channels();
        let alpha = s.format().alpha;
        for px in s.read_region(canvas).chunks_exact(ch) {
            let a = if alpha { px[ch - 1] } else { 1.0 };
            for (c, plane) in planes.iter_mut().enumerate() {
                plane.push(px[c] * a + white[c] * (1.0 - a));
            }
        }
    } else {
        let buf = photocraft_compose::flatten(doc).over_background([1.0, 1.0, 1.0]);
        for p in &buf.px {
            let v = photocraft_raster::from_rgba(&fmt, *p);
            for (c, plane) in planes.iter_mut().enumerate() {
                plane.push(v[c]);
            }
        }
    }
    planes
}

fn channel_surface(doc: &Document, vals: &[f32]) -> Surface {
    let mut s = Surface::new(PixelFormat::new(ColorMode::Grayscale, doc.depth, false));
    let clamped: Vec<f32> = vals.iter().map(|v| v.clamp(0.0, 1.0)).collect();
    s.write_region(doc.bounds(), &clamped);
    s.prune();
    s
}

/// The ink channels (name, display colour, densities) of the flattened document.
fn inks_of(doc: &Document) -> Vec<(String, Color, Vec<f32>)> {
    let planes = flattened_planes(doc);
    let inv = |p: &Vec<f32>| p.iter().map(|v| 1.0 - v).collect::<Vec<f32>>();
    match doc.mode {
        ColorMode::Rgb => {
            [("Cyan", CYAN), ("Magenta", MAGENTA), ("Yellow", YELLOW)].into_iter().zip(&planes).map(|((n, c), p)| (n.to_string(), c, inv(p))).collect()
        }
        ColorMode::Cmyk => [("Cyan", CYAN), ("Magenta", MAGENTA), ("Yellow", YELLOW), ("Black", BLACK)]
            .into_iter()
            .zip(planes)
            .map(|((n, c), p)| (n.to_string(), c, p))
            .collect(),
        ColorMode::Lab => planes.iter().enumerate().map(|(k, p)| (format!("Alpha {}", k + 1), Color::BLACK, inv(p))).collect(),
        ColorMode::Duotone if doc.duotone.as_ref().is_some_and(|d| !d.inks.is_empty()) => doc
            .duotone
            .iter()
            .flat_map(|d| d.inks.iter())
            .map(|ink| (ink.name.clone(), Color::rgb(ink.color[0], ink.color[1], ink.color[2]), planes[0].iter().map(|g| ink.density(1.0 - g)).collect()))
            .collect(),
        _ => vec![("Black".to_string(), Color::BLACK, inv(&planes[0]))],
    }
}

fn to_multichannel(s: &mut Session, _p: &Value) -> Result<Value> {
    let names = s.edit("Multichannel", |doc, active| {
        let inks = inks_of(doc);
        let mut channels: Vec<AlphaChannel> = inks
            .iter()
            .map(|(name, ink, vals)| AlphaChannel { spot: Some((*ink, 0.0)), ..AlphaChannel::new(name.clone(), channel_surface(doc, vals)) })
            .collect();
        // Existing alpha / spot channels follow the inks, converted to the document depth.
        let fmt = PixelFormat::new(ColorMode::Grayscale, doc.depth, false);
        for mut ch in std::mem::take(&mut doc.channels) {
            if ch.surface.format() != fmt {
                ch.surface = ch.surface.convert(fmt);
            }
            channels.push(ch);
        }
        doc.channels = channels;
        doc.layers.clear();
        *active = None;
        doc.mode = ColorMode::Multichannel;
        doc.quick_mask = None;
        doc.color_table = None;
        doc.duotone = None;
        doc.icc_profile = None;
        Ok(inks.into_iter().map(|(n, _, _)| n).collect::<Vec<_>>())
    })?;
    Ok(json!({ "mode": "multichannel", "channels": names }))
}

/// Image › Mode › RGB / CMYK / Lab / Grayscale from a Multichannel document (called by
/// `color_cmds::convert_mode`). Profiles are not involved: Multichannel has none.
pub(crate) fn convert_from(s: &mut Session, mode: ColorMode, _p: &Value) -> Result<Value> {
    if !matches!(mode, ColorMode::Rgb | ColorMode::Cmyk | ColorMode::Lab | ColorMode::Grayscale) {
        return Err(EngineError::Other(format!("a Multichannel document can't be converted to {mode:?}")));
    }
    let how = s.edit("Mode Change", |doc, active| {
        let (inks, others): (Vec<AlphaChannel>, Vec<AlphaChannel>) = std::mem::take(&mut doc.channels).into_iter().partition(|c| c.spot.is_some());
        let canvas = doc.bounds();
        let n = canvas.width() as usize * canvas.height() as usize;
        let reinterpret = matches!((mode, inks.len()), (ColorMode::Rgb | ColorMode::Lab, 3) | (ColorMode::Cmyk, 4) | (ColorMode::Grayscale, 1));
        // The printed look, rendered before the inks leave the document.
        let shown = (!reinterpret).then(|| {
            let mut d = doc.clone();
            d.channels = inks.clone();
            photocraft_compose::flatten(&d)
        });
        doc.mode = mode;
        let fmt = doc.pixel_format();
        let cc = fmt.mode.color_channels();
        let mut data = vec![0.0f32; n * (cc + 1)];
        if let Some(buf) = &shown {
            for (px, p) in data.chunks_exact_mut(cc + 1).zip(&buf.px) {
                px.copy_from_slice(&photocraft_raster::from_rgba(&fmt, [p[0], p[1], p[2], 1.0])[..cc + 1]);
            }
        } else {
            for (k, ch) in inks.iter().enumerate() {
                let vals = ch.surface.read_region(canvas);
                let step = ch.surface.channels();
                for (i, v) in vals.chunks_exact(step).enumerate() {
                    // CMYK plates are ink amounts already; other modes store light.
                    data[i * (cc + 1) + k] = if mode == ColorMode::Cmyk { v[0] } else { 1.0 - v[0] };
                }
            }
            for i in 0..n {
                data[i * (cc + 1) + cc] = 1.0;
            }
        }
        let mut surf = Surface::new(fmt);
        surf.write_region(canvas, &data);
        surf.prune();
        let mut bg = Layer::new("Background", LayerContent::Raster(surf));
        bg.locks.transparency = true;
        bg.locks.position = true;
        *active = Some(bg.id);
        doc.layers = vec![bg];
        doc.channels = others;
        Ok(if reinterpret { "channels" } else { "appearance" })
    })?;
    Ok(json!({ "mode": format!("{mode:?}").to_lowercase(), "from": "multichannel", "converted": how }))
}

/// Names of the ink (spot) channels that make up a Multichannel image.
pub fn ink_names(doc: &Document) -> Vec<String> {
    doc.channels.iter().filter(|c| c.spot.is_some()).map(|c| c.name.clone()).collect()
}

pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: "image.mode.multichannel",
        label: "Multichannel",
        menu: &["Image", "Mode"],
        shortcut: None,
        params: r##"{} (flattens; RGB → Cyan/Magenta/Yellow, CMYK → Cyan/Magenta/Yellow/Black, Lab → Alpha 1–3, Grayscale → Black, Duotone → its inks)"##,
        enabled,
        run: to_multichannel,
        journal: true,
    }]
}

#[cfg(test)]
mod tests;
