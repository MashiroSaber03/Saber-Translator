//! Pixel layers (`Rstr`) and placed images (`ImgN`). A placed image keeps its original file
//! (`DyBm.Bckg` → archive entry `c/N`), which is imported as is; other pixels are planar tiles of
//! 256 bytes × 256 rows per channel, each its own archive entry.

use std::collections::HashMap;

use crate::Error;
use crate::model::{Affine, Image, Pixels, Reader};
use crate::stream::{self, ObjId, Tag, Value};

/// Pixels decoded per layer (the content's bounding box, not the canvas).
const MAX_PIXELS: u64 = 64 << 20;
const TILE: usize = 256;
const TILE_BYTES: usize = TILE * TILE;

pub(crate) fn node_image(r: &mut Reader, id: ObjId, world: Affine) -> Result<Option<Image>, Error> {
    let s = r.s;
    let Some(bitmap) = s.obj(id, b"Bitm") else {
        r.warn("pixel layers without pixels");
        return Ok(None);
    };
    if !s.is(bitmap, b"DyBm") {
        r.warn("pixel layers stored in an unknown way");
        return Ok(None);
    }
    let width = s.int(bitmap, b"BmpW").and_then(|v| u32::try_from(v).ok()).unwrap_or(0);
    let height = s.int(bitmap, b"BmpH").and_then(|v| u32::try_from(v).ok()).unwrap_or(0);
    if width == 0 || height == 0 {
        return Ok(None);
    }
    // A placed image: the original file, byte for byte.
    let has_tiles = s.field(bitmap, b"Sta1").is_some();
    if let Some(name) = s.entry(bitmap, b"Bckg").filter(|n| !n.is_empty())
        && (s.is(id, b"ImgN") || !has_tiles)
    {
        match original(r, name) {
            Ok(bytes) => return Ok(Some(Image { width, height, pixels: Pixels::Encoded(bytes), transform: world })),
            Err(Error::Limit(e)) => return Err(Error::Limit(e)),
            Err(_) => r.warn("an embedded image file that could not be read (its cached pixels were used)"),
        }
    }
    // The content's bounding box, so a mostly empty canvas-sized layer stays small.
    let crop = s
        .field(id, b"BitR")
        .and_then(|v| if let Value::Ints(i) = v { Some(i.clone()) } else { None })
        .filter(|v| v.len() == 4 && v[0] > i64::from(i32::MIN) + 1)
        .map(|v| {
            (
                v[0].clamp(0, i64::from(width)) as u32,
                v[1].clamp(0, i64::from(height)) as u32,
                v[2].clamp(0, i64::from(width)) as u32,
                v[3].clamp(0, i64::from(height)) as u32,
            )
        })
        .unwrap_or((0, 0, width, height));
    if s.field(bitmap, b"Bckg").is_some() && has_tiles && !s.is(id, b"ImgN") {
        r.warn("edited placed images use their stored pixels");
    }
    decode(r, bitmap, crop, world)
}

/// A mask layer (`MRst`): one channel of coverage as grey pixels (white shows, black hides).
pub(crate) fn mask(r: &mut Reader, id: ObjId, world: Affine) -> Result<Option<Image>, Error> {
    let s = r.s;
    let Some(bitmap) = s.obj(id, b"Bitm") else { return Ok(None) };
    let width = s.int(bitmap, b"BmpW").and_then(|v| u32::try_from(v).ok()).unwrap_or(0);
    let height = s.int(bitmap, b"BmpH").and_then(|v| u32::try_from(v).ok()).unwrap_or(0);
    if width == 0 || height == 0 || !matches!(s.enumeration(bitmap, b"Frmt"), Some((6 | 7, _))) {
        r.warn("pixel masks in an unknown format (imported without them)");
        return Ok(None);
    }
    decode(r, bitmap, (0, 0, width, height), world)
}

/// An embedded document or symbol instance (`EmbN`), as the picture of it Affinity caches.
pub(crate) fn embedded(r: &mut Reader, id: ObjId, world: Affine) -> Result<Option<Image>, Error> {
    let s = r.s;
    let Some(cache) = s.obj(id, b"Bitm").and_then(|e| s.obj(e, b"Cach")) else { return Ok(None) };
    let Some(bbox) = s.obj(id, b"Bitm").and_then(|e| s.floats::<4>(e, b"ChBB")) else { return Ok(None) };
    let width = s.int(cache, b"BmpW").and_then(|v| u32::try_from(v).ok()).unwrap_or(0);
    let height = s.int(cache, b"BmpH").and_then(|v| u32::try_from(v).ok()).unwrap_or(0);
    if width == 0 || height == 0 {
        return Ok(None);
    }
    let [x0, y0, x1, y1] = bbox;
    // The instance's origin is the centre of the embedded content (checked against the
    // thumbnails of documents with embedded documents and SVG files).
    let (cx, cy) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
    let place = Affine([(x1 - x0) / f64::from(width), 0.0, 0.0, (y1 - y0) / f64::from(height), x0 - cx, y0 - cy]).then(world);
    decode(r, cache, (0, 0, width, height), place)
}

/// Decode the tiles of a bitmap inside `crop` (pixels), placed by `world` (bitmap pixels to document).
fn decode(r: &mut Reader, bitmap: ObjId, crop: (u32, u32, u32, u32), world: Affine) -> Result<Option<Image>, Error> {
    let s = r.s;
    let (x0, y0, x1, y1) = crop;
    if x1 <= x0 || y1 <= y0 {
        return Ok(None);
    }
    let (w, h) = (x1 - x0, y1 - y0);
    if u64::from(w) * u64::from(h) > MAX_PIXELS {
        r.warn("pixel layers larger than 64 megapixels");
        return Ok(None);
    }
    let format = s.enumeration(bitmap, b"Frmt").map(|(f, _)| f);
    let (channels, bps) = match format {
        Some(0) => (4, 1),
        Some(1) => (4, 2),
        Some(6) => (1, 1),
        Some(7) => (1, 2),
        Some(4) => {
            r.warn("CMYK pixel layers (converted to RGB without the document's colour profile)");
            (5, 1)
        }
        _ => {
            r.warn("pixel layers in grey, Lab or 32-bit formats");
            return Ok(None);
        }
    };
    let mut planes = Vec::with_capacity(channels);
    let mut cache = HashMap::new();
    for c in 1..=channels {
        planes.push(plane(r, bitmap, c, bps, (x0, y0, w, h), &mut cache)?);
    }
    let n = (w as usize) * (h as usize);
    let mut rgba = vec![0u8; n * 4];
    let sample = |p: &Vec<u8>, i: usize| -> f64 {
        if bps == 2 {
            f64::from(u16::from_le_bytes([p.get(2 * i).copied().unwrap_or(0), p.get(2 * i + 1).copied().unwrap_or(0)])) / 65535.0
        } else {
            f64::from(p.get(i).copied().unwrap_or(0)) / 255.0
        }
    };
    for (i, px) in rgba.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let v: Vec<f64> = planes.iter().map(|p| sample(p, i)).collect();
        let (r2, g, b, a) = if channels == 1 {
            (v[0], v[0], v[0], 1.0)
        } else if channels == 5 {
            let k = 1.0 - v[3];
            ((1.0 - v[0]) * k, (1.0 - v[1]) * k, (1.0 - v[2]) * k, v[4])
        } else {
            (v[0], v[1], v[2], v[3])
        };
        px.copy_from_slice(&[r2, g, b, a].map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8));
    }
    let transform = Affine([1.0, 0.0, 0.0, 1.0, f64::from(x0), f64::from(y0)]).then(world);
    Ok(Some(Image { width: w, height: h, pixels: Pixels::Rgba8(rgba), transform }))
}

/// The original image file: entry `c/N` is a stream whose root (`Blck`) holds it as `Data`.
fn original(r: &mut Reader, name: &str) -> Result<Vec<u8>, Error> {
    let bytes = r.archive.read(name)?;
    let s = stream::parse(&bytes)?;
    let root = s.object(s.root).ok_or(Error::Malformed("embedded image"))?;
    match root.get(Tag::of(b"Data")) {
        Some(Value::Blob(b)) if !b.is_empty() => Ok(b.clone()),
        _ => Err(Error::Malformed("embedded image without data")),
    }
}

fn tag(prefix: &[u8; 3], c: usize) -> [u8; 4] {
    [prefix[0], prefix[1], prefix[2], b'0' + c as u8]
}

/// One channel of level 0, cropped to (x0, y0, w, h) pixels; `bps` bytes per sample.
fn plane(
    r: &mut Reader,
    bitmap: ObjId,
    c: usize,
    bps: usize,
    (x0, y0, w, h): (u32, u32, u32, u32),
    cache: &mut HashMap<String, Vec<u8>>,
) -> Result<Vec<u8>, Error> {
    let s = r.s;
    let width = s.int(bitmap, b"BmpW").and_then(|v| usize::try_from(v).ok()).unwrap_or(0);
    let height = s.int(bitmap, b"BmpH").and_then(|v| usize::try_from(v).ok()).unwrap_or(0);
    let tw = s.int(bitmap, &tag(b"TWi", c)).and_then(|v| usize::try_from(v).ok()).unwrap_or_else(|| (width * bps).div_ceil(TILE));
    let th = s.int(bitmap, &tag(b"THi", c)).and_then(|v| usize::try_from(v).ok()).unwrap_or_else(|| height.div_ceil(TILE));
    let states: Vec<u8> = match s.field(bitmap, &tag(b"Sta", c)) {
        Some(Value::Array(a)) => a.iter().map(|v| if let Value::UInt(u) = v { u8::try_from(*u).unwrap_or(255) } else { 255 }).collect(),
        _ => Vec::new(),
    };
    let tiles = s.objs(bitmap, &tag(b"Idx", c));
    let row = (w as usize) * bps;
    let mut out = vec![0u8; row * h as usize];
    let (bx0, bx1) = (x0 as usize * bps, (x0 + w) as usize * bps);
    let (py0, py1) = (y0 as usize, (y0 + h) as usize);
    let mut next = 0usize;
    for (t, state) in states.iter().enumerate().take(tw.saturating_mul(th)) {
        let (ty, tx) = (t / tw.max(1), t % tw.max(1));
        let fill = match state {
            0 | 1 => None,
            2 => Some(0xFF),
            4 => {
                let Some(blck) = tiles.get(next).copied() else { return Err(Error::Malformed("missing pixel tile")) };
                next += 1;
                let (ox, oy) = (tx * TILE, ty * TILE);
                if ox >= bx1 || ox + TILE <= bx0 || oy >= py1 || oy + TILE <= py0 {
                    continue;
                }
                let name = s.entry(blck, b"Data").ok_or(Error::Malformed("pixel tile without data"))?.to_string();
                if !cache.contains_key(&name) {
                    let mut data = r.archive.read(&name)?;
                    // Older files wrap a tile in a one-field stream: header, then a 64 KiB blob.
                    if data.len() == TILE_BYTES + 22 && data.starts_with(b"\x00\xffKS") {
                        data = data.get(21..21 + TILE_BYTES).ok_or(Error::Malformed("pixel tile"))?.to_vec();
                    }
                    if data.len() != TILE_BYTES {
                        return Err(Error::Malformed("pixel tile size"));
                    }
                    cache.insert(name.clone(), data);
                }
                let tile = cache.get(&name).ok_or(Error::Malformed("pixel tile"))?;
                copy(&mut out, row, (bx0, py0, bx1, py1), (ox, oy), |x, y| tile.get(y * TILE + x).copied().unwrap_or(0));
                continue;
            }
            3 => {
                r.warn("32-bit pixel layers");
                None
            }
            5 => {
                r.warn("pixel layers drawn from an embedded image (left empty)");
                None
            }
            _ => return Err(Error::Malformed("unknown pixel tile state")),
        };
        if let Some(v) = fill {
            copy(&mut out, row, (bx0, py0, bx1, py1), (tx * TILE, ty * TILE), |_, _| v);
        }
    }
    Ok(out)
}

/// Copy the overlap of a tile at byte column `ox`, row `oy` into the cropped plane.
fn copy(out: &mut [u8], row: usize, (bx0, py0, bx1, py1): (usize, usize, usize, usize), (ox, oy): (usize, usize), at: impl Fn(usize, usize) -> u8) {
    for y in oy.max(py0)..(oy + TILE).min(py1) {
        for x in ox.max(bx0)..(ox + TILE).min(bx1) {
            if let Some(b) = out.get_mut((y - py0) * row + (x - bx0)) {
                *b = at(x - ox, y - oy);
            }
        }
    }
}
