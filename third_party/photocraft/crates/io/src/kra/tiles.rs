//! Krita's tiled pixel files and the LZF compression inside them.

use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_geom::Rect;
use photocraft_raster::{Interrupt, Surface};

use super::{Result, invalid};
use crate::IoError;

/// The pixel layouts we read, by Krita colour-space id. Samples are straight (unpremultiplied).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Space {
    /// `RGBA`: B, G, R, A bytes.
    Rgba8,
    /// `RGBA16`: B, G, R, A as little-endian u16.
    Rgba16,
    /// `GRAYA`: gray, alpha bytes.
    Gray8,
    /// `GRAYAU16`: gray, alpha as little-endian u16.
    Gray16,
    /// Masks and selections: one byte.
    Alpha8,
}

impl Space {
    pub(crate) fn from_name(name: &str) -> Option<Space> {
        Some(match name {
            "RGBA" => Space::Rgba8,
            "RGBA16" => Space::Rgba16,
            "GRAYA" => Space::Gray8,
            "GRAYAU16" => Space::Gray16,
            _ => return None,
        })
    }
    pub(crate) fn pixel_size(self) -> usize {
        match self {
            Space::Rgba8 => 4,
            Space::Rgba16 => 8,
            Space::Gray8 => 2,
            Space::Gray16 => 4,
            Space::Alpha8 => 1,
        }
    }
    /// The document colour mode and depth for an image in this space.
    pub(crate) fn model(self) -> (ColorMode, SampleType) {
        match self {
            Space::Rgba8 => (ColorMode::Rgb, SampleType::U8),
            Space::Rgba16 => (ColorMode::Rgb, SampleType::U16),
            Space::Gray8 | Space::Alpha8 => (ColorMode::Grayscale, SampleType::U8),
            Space::Gray16 => (ColorMode::Grayscale, SampleType::U16),
        }
    }
    /// The surface format whose bytes [`Self::convert`] writes directly.
    fn native(self) -> PixelFormat {
        match self {
            Space::Alpha8 => PixelFormat::GRAY8,
            s => {
                let (mode, depth) = s.model();
                PixelFormat::new(mode, depth, true)
            }
        }
    }
    /// One interleaved Krita pixel as straight RGBA (0..1).
    fn rgba(self, px: &[u8]) -> [f32; 4] {
        let b = |i: usize| f32::from(px.get(i).copied().unwrap_or(0)) / 255.0;
        let w = |i: usize| f32::from(u16::from_le_bytes([px.get(i).copied().unwrap_or(0), px.get(i + 1).copied().unwrap_or(0)])) / 65535.0;
        match self {
            Space::Rgba8 => [b(2), b(1), b(0), b(3)],
            Space::Rgba16 => [w(4), w(2), w(0), w(6)],
            Space::Gray8 => [b(0), b(0), b(0), b(1)],
            Space::Gray16 => [w(0), w(0), w(0), w(2)],
            Space::Alpha8 => [b(0), b(0), b(0), 1.0],
        }
    }
    /// A default pixel's samples (Krita order, 0..1), for [`Self::to_format`].
    pub(crate) fn normalised(self, px: &[u8]) -> Vec<f32> {
        self.rgba(px).to_vec()
    }
    /// RGBA samples from [`Self::normalised`] as `fmt`'s channels.
    pub(crate) fn to_format(self, rgba: &[f32], fmt: PixelFormat) -> Vec<f32> {
        let p = [0, 1, 2, 3].map(|i| rgba.get(i).copied().unwrap_or(0.0));
        photocraft_raster::from_rgba(&fmt, p)
    }
    /// Interleaved Krita pixels as encoded `fmt` bytes.
    fn convert(self, src: &[u8], fmt: PixelFormat, out: &mut Vec<u8>) {
        out.clear();
        let ps = self.pixel_size();
        if fmt == self.native() {
            for px in src.chunks_exact(ps) {
                match self {
                    Space::Rgba8 => out.extend([2, 1, 0, 3].map(|i| px.get(i).copied().unwrap_or(0))),
                    Space::Rgba16 => {
                        for i in [4, 2, 0, 6] {
                            let v = u16::from_le_bytes([px.get(i).copied().unwrap_or(0), px.get(i + 1).copied().unwrap_or(0)]);
                            out.extend(v.to_ne_bytes());
                        }
                    }
                    Space::Gray16 => {
                        for i in [0, 2] {
                            let v = u16::from_le_bytes([px.get(i).copied().unwrap_or(0), px.get(i + 1).copied().unwrap_or(0)]);
                            out.extend(v.to_ne_bytes());
                        }
                    }
                    Space::Gray8 | Space::Alpha8 => out.extend_from_slice(px),
                }
            }
            return;
        }
        let mut enc = vec![0; fmt.bytes_per_pixel()];
        for px in src.chunks_exact(ps) {
            let v = photocraft_raster::from_rgba(&fmt, self.rgba(px));
            photocraft_raster::encode_pixel(&fmt, &v, &mut enc);
            out.extend_from_slice(&enc);
        }
    }
}

/// LZF decompression (the liblzf format) into exactly `expected` bytes.
pub(crate) fn lzf_decompress(src: &[u8], expected: usize) -> Result<Vec<u8>> {
    let bad = || invalid("corrupt LZF tile data");
    let mut out = Vec::with_capacity(expected);
    let mut ip = 0;
    while ip < src.len() {
        let ctrl = usize::from(*src.get(ip).ok_or_else(bad)?);
        ip += 1;
        if ctrl < 32 {
            let len = ctrl + 1;
            let lit = src.get(ip..ip + len).ok_or_else(bad)?;
            if out.len() + len > expected {
                return Err(bad());
            }
            out.extend_from_slice(lit);
            ip += len;
        } else {
            let mut len = ctrl >> 5;
            if len == 7 {
                len += usize::from(*src.get(ip).ok_or_else(bad)?);
                ip += 1;
            }
            len += 2;
            let back = ((ctrl & 0x1f) << 8) + usize::from(*src.get(ip).ok_or_else(bad)?) + 1;
            ip += 1;
            let start = out.len().checked_sub(back).ok_or_else(bad)?;
            if out.len() + len > expected {
                return Err(bad());
            }
            // Back-references may overlap the bytes they produce, so copy one at a time.
            for i in start..start + len {
                let b = *out.get(i).ok_or_else(bad)?;
                out.push(b);
            }
        }
    }
    if out.len() != expected {
        return Err(invalid(format!("LZF tile data decodes to {} bytes, expected {expected}", out.len())));
    }
    Ok(out)
}

/// A parsed tiled pixel file (`VERSION 2`), borrowing the tile payloads.
pub(crate) struct TileFile<'a> {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) pixel_size: usize,
    tiles: Vec<(i32, i32, &'a [u8])>,
}

fn line<'a>(bytes: &'a [u8], pos: &mut usize) -> Result<&'a str> {
    let rest = bytes.get(*pos..).ok_or_else(|| invalid("truncated tile file"))?;
    let end = rest.iter().take(256).position(|b| *b == b'\n').ok_or_else(|| invalid("truncated or malformed tile file header"))?;
    *pos += end + 1;
    std::str::from_utf8(rest.get(..end).unwrap_or_default()).map_err(|_| invalid("tile file header is not text"))
}

fn header_value(bytes: &[u8], pos: &mut usize, key: &str, max: usize) -> Result<usize> {
    let l = line(bytes, pos)?;
    let v = l.strip_prefix(key).map(str::trim).ok_or_else(|| invalid(format!("tile file: expected {key}, found `{l}`")))?;
    v.parse::<usize>().ok().filter(|v| (1..=max).contains(v)).ok_or_else(|| invalid(format!("tile file: invalid {key} `{v}`")))
}

impl<'a> TileFile<'a> {
    pub(crate) fn parse(bytes: &'a [u8]) -> Result<TileFile<'a>> {
        let mut pos = 0;
        let version = line(bytes, &mut pos)?;
        if version.trim() != "VERSION 2" {
            return Err(IoError::Unsupported(format!("Krita tile file `{}` (only VERSION 2, Krita 2.0 and later)", version.trim())));
        }
        let width = header_value(bytes, &mut pos, "TILEWIDTH", 1024)?;
        let height = header_value(bytes, &mut pos, "TILEHEIGHT", 1024)?;
        let pixel_size = header_value(bytes, &mut pos, "PIXELSIZE", 64)?;
        let l = line(bytes, &mut pos)?;
        let count = l
            .strip_prefix("DATA")
            .map(str::trim)
            .and_then(|v| v.parse::<usize>().ok())
            .ok_or_else(|| invalid(format!("tile file: invalid DATA line `{l}`")))?;
        // Every tile needs at least a header line and a flag byte.
        if count > bytes.len() / 8 + 1 {
            return Err(invalid("tile count exceeds the file size"));
        }
        let mut tiles = Vec::with_capacity(count);
        for _ in 0..count {
            let l = line(bytes, &mut pos)?;
            let mut f = l.split(',');
            let (x, y, method, len) = (f.next(), f.next(), f.next(), f.next());
            let coord = |v: Option<&str>| v.and_then(|v| v.trim().parse::<i32>().ok()).filter(|v| v.unsigned_abs() <= super::MAX_SIDE * 4);
            let (Some(x), Some(y), Some("LZF"), Some(len), None) =
                (coord(x), coord(y), method.map(str::trim), len.and_then(|v| v.trim().parse::<usize>().ok()), f.next())
            else {
                return Err(invalid(format!("tile file: malformed tile header `{l}`")));
            };
            let data = bytes.get(pos..pos.checked_add(len).ok_or_else(|| invalid("tile size overflow"))?).ok_or_else(|| invalid("truncated tile data"))?;
            pos += len;
            tiles.push((x, y, data));
        }
        Ok(TileFile { width, height, pixel_size, tiles })
    }

    /// Bytes the decoded tiles take at `bpp` bytes per pixel.
    pub(crate) fn decoded_bytes(&self, bpp: usize) -> Result<usize> {
        self.tiles.len().checked_mul(self.width * self.height).and_then(|n| n.checked_mul(bpp)).ok_or_else(|| invalid("tile size overflow"))
    }

    /// Writes every tile into `surface` (format `surface.format()`), offset by (`dx`, `dy`).
    pub(crate) fn decode_into(&self, surface: &mut Surface, space: Space, dx: i32, dy: i32, ctl: &Interrupt<'_>) -> Result<()> {
        let fmt = surface.format();
        let n = self.width * self.height;
        let expected = n * self.pixel_size;
        let mut interleaved = vec![0u8; expected];
        let mut out = Vec::new();
        for &(x, y, data) in &self.tiles {
            ctl.check().map_err(|_| IoError::Cancelled)?;
            let (flag, payload) = data.split_first().ok_or_else(|| invalid("empty tile"))?;
            let planar = match flag {
                1 => lzf_decompress(payload, expected)?,
                0 if payload.len() == expected => payload.to_vec(),
                0 => return Err(invalid("raw tile has the wrong size")),
                f => return Err(invalid(format!("unknown tile compression flag {f}"))),
            };
            // Byte planes: byte k of pixel i is at k * n + i.
            for (k, plane) in planar.chunks_exact(n).enumerate() {
                for (i, b) in plane.iter().enumerate() {
                    if let Some(slot) = interleaved.get_mut(i * self.pixel_size + k) {
                        *slot = *b;
                    }
                }
            }
            space.convert(&interleaved, fmt, &mut out);
            let (x0, y0) =
                (x.checked_add(dx).ok_or_else(|| invalid("tile offset overflow"))?, y.checked_add(dy).ok_or_else(|| invalid("tile offset overflow"))?);
            let rect = Rect::from_xywh(x0, y0, self.width as u32, self.height as u32);
            if rect.width() as usize != self.width || rect.height() as usize != self.height || out.len() != n * fmt.bytes_per_pixel() {
                return Err(invalid("tile does not fit its position"));
            }
            surface.write_interleaved(rect, &out);
        }
        Ok(())
    }
}
