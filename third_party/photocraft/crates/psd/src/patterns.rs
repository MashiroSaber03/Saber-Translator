//! Patterns: the global `Patt` / `Pat2` / `Pat3` tagged blocks (8 / 16 / 32-bit documents) and
//! standalone `.pat` files.
//!
//! Layout per the public Photoshop File Formats Specification ("Patterns" and "Virtual Memory
//! Array List"):
//!
//! ```text
//! pattern  := version u32 (1), image mode u32, height i16, width i16,
//!             name (Unicode string), unique id (Pascal string, unpadded),
//!             [indexed: 256 × RGB palette], virtual memory array list
//! VMA list := version u32 (3), length u32, rect (top, left, bottom, right i32),
//!             channel count u32, then count + 2 arrays (user mask, sheet mask)
//! array    := written u32 (0 = absent), length u32 (0 = absent), depth u32, rect,
//!             depth u16, compression u8 (0 raw, 1 PackBits RLE), data
//! ```
//!
//! In the tagged blocks each pattern is prefixed by its u32 length and padded to 4 bytes; a
//! `.pat` file is `8BPT`, version u16 (1), count u32, then the patterns without length prefixes.
//! Photoshop writes the channel count as 24 and only marks the real channels as written; like
//! ag-psd (MIT) we take the first written arrays as the colour channels and the next one as
//! transparency.

use crate::compression::{Compression, MAX_DECODED_BYTES, PlaneLayout, decode_planes, encode_planes};
use crate::error::{PsdError, Result};
use crate::header::Version;

/// Largest pattern edge accepted (Photoshop's own limit is far lower).
const MAX_EDGE: u32 = 30_000;
/// Budget for zero-padded channel planes summed over one pattern. Photoshop writes every channel
/// at the pattern rectangle, so a smaller channel rectangle is only padding; without this cap a
/// few hundred bytes (1x1 channels, a huge pattern rectangle) would allocate up to
/// `MAX_DECODED_BYTES` of zeros. Matches `abr::MAX_TOTAL_BYTES`.
const MAX_PADDED_PATTERN_BYTES: u64 = 512 << 20;
// A PackBits run packet emits at most 128 bytes from a 2-byte packet.
const MAX_RLE_EXPANSION: u64 = 64;

/// One pattern tile with planar, big-endian samples.
#[derive(Debug, Clone, PartialEq)]
pub struct PsdPattern {
    /// PSD colour mode number (1 gray, 2 indexed, 3 RGB, 4 CMYK, 7 multichannel, 8 duotone, 9 Lab).
    pub mode: u32,
    /// Tile width in pixels.
    pub width: u32,
    /// Tile height in pixels.
    pub height: u32,
    /// Name without the trailing NUL (built-ins look like `$$$/Patterns/…=Water`).
    pub name: String,
    /// Unique id (usually a UUID).
    pub id: String,
    /// Indexed-colour palette (768 bytes, RGB triplets).
    pub palette: Option<Vec<u8>>,
    /// Bits per sample: 1, 8, 16 or 32.
    pub depth: u16,
    /// Decoded colour planes (`width × height` samples each).
    pub channels: Vec<Vec<u8>>,
    /// Decoded transparency plane, if any.
    pub alpha: Option<Vec<u8>>,
}

/// Colour channels implied by a PSD colour mode.
pub fn mode_channels(mode: u32) -> usize {
    match mode {
        3 | 9 => 3,
        4 => 4,
        _ => 1,
    }
}

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}

struct PatternChannel<'a> {
    data: &'a [u8],
    compression: Compression,
    source: PlaneLayout,
    destination: PlaneLayout,
    x: usize,
    y: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.p.checked_add(n).ok_or(PsdError::LimitExceeded("pattern length overflow"))?;
        if end > self.b.len() {
            return Err(PsdError::UnexpectedEof { offset: self.p, needed: end - self.b.len() });
        }
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        let s = self.take(2)?;
        Ok(u16::from_be_bytes([s[0], s[1]]))
    }
    fn u32(&mut self) -> Result<u32> {
        let s = self.take(4)?;
        Ok(u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()? as i32)
    }
    fn rect(&mut self) -> Result<[i32; 4]> {
        Ok([self.i32()?, self.i32()?, self.i32()?, self.i32()?])
    }
}

fn read_pattern(r: &mut Rd) -> Result<PsdPattern> {
    let version = r.u32()?;
    if version != 1 {
        return Err(PsdError::invalid(format!("pattern version {version}")));
    }
    let mode = r.u32()?;
    let _h = r.u16()?;
    let _w = r.u16()?;
    let n = r.u32()? as usize;
    if n > 65_536 {
        return Err(PsdError::LimitExceeded("pattern name length"));
    }
    let units: Vec<u16> = r.take(n * 2)?.as_chunks::<2>().0.iter().map(|c| u16::from_be_bytes([c[0], c[1]])).collect();
    let name = String::from_utf16_lossy(&units).trim_end_matches('\0').to_string();
    let idl = r.u8()? as usize;
    let id = String::from_utf8_lossy(r.take(idl)?).to_string();
    let palette = if mode == 2 {
        let p = r.take(768)?.to_vec();
        // Some writers follow the palette with 4 extra bytes (ag-psd skips them).
        if r.b.get(r.p..r.p + 4).is_some_and(|v| v != [0, 0, 0, 3]) {
            r.take(4)?;
        }
        Some(p)
    } else {
        None
    };
    let vver = r.u32()?;
    if vver != 3 {
        return Err(PsdError::invalid(format!("virtual memory array list version {vver}")));
    }
    let len = r.u32()? as usize;
    let body = r.take(len)?;
    let mut v = Rd { b: body, p: 0 };
    let [top, left, bottom, right] = v.rect()?;
    let w = right.checked_sub(left).ok_or(PsdError::LimitExceeded("pattern width overflow"))?;
    let h = bottom.checked_sub(top).ok_or(PsdError::LimitExceeded("pattern height overflow"))?;
    if w <= 0 || h <= 0 || w as u32 > MAX_EDGE || h as u32 > MAX_EDGE {
        return Err(PsdError::LimitExceeded("pattern size"));
    }
    let (w, h) = (w as u32, h as u32);
    let count = v.u32()?;
    if count > 64 {
        return Err(PsdError::LimitExceeded("pattern channel count"));
    }
    // Validate every channel and the total padded size before decoding any plane.
    // Otherwise a later channel can exceed the existing pattern budget only after
    // earlier channels have already allocated several gigabytes of zero padding.
    let mut channels = Vec::new();
    let mut depth = 8u16;
    let mut decoded_bytes = 0u64;
    let mut padded_bytes = 0u64;
    for _ in 0..count + 2 {
        if v.p >= body.len() {
            break;
        }
        if v.u32()? == 0 {
            continue;
        }
        let alen = v.u32()? as usize;
        if alen == 0 {
            continue;
        }
        if alen < 23 {
            return Err(PsdError::invalid("short virtual memory array"));
        }
        let d32 = v.u32()?;
        let [ct, cl, cb, cr] = v.rect()?;
        let d16 = v.u16()?;
        let comp = v.u8()?;
        let data = v.take(alen - 23)?;
        let d = if matches!(d16, 1 | 8 | 16 | 32) { d16 } else { d32 as u16 };
        if !matches!(d, 1 | 8 | 16 | 32) {
            return Err(PsdError::invalid(format!("pattern depth {d}")));
        }
        depth = d;
        if ct < top || cl < left || cb > bottom || cr > right || cb < ct || cr < cl {
            return Err(PsdError::invalid("pattern channel rectangle falls outside pattern rectangle"));
        }
        let cw = usize::try_from(cr.checked_sub(cl).ok_or(PsdError::LimitExceeded("pattern channel width overflow"))?)
            .map_err(|_| PsdError::invalid("negative pattern channel width"))?;
        let ch = usize::try_from(cb.checked_sub(ct).ok_or(PsdError::LimitExceeded("pattern channel height overflow"))?)
            .map_err(|_| PsdError::invalid("negative pattern channel height"))?;
        let channel_layout = PlaneLayout { planes: 1, width: cw, height: ch, depth: d, version: Version::Psd };
        let channel_bytes = channel_layout.total_bytes()?;
        let layout = PlaneLayout { planes: 1, width: w as usize, height: h as usize, depth: d, version: Version::Psd };
        let plane_bytes = layout.total_bytes()?;
        decoded_bytes = decoded_bytes.checked_add(plane_bytes).ok_or(PsdError::LimitExceeded("pattern decoded size overflow"))?;
        if decoded_bytes > MAX_DECODED_BYTES {
            return Err(PsdError::LimitExceeded("pattern data exceeds MAX_DECODED_BYTES"));
        }
        if cw != w as usize || ch != h as usize {
            padded_bytes = padded_bytes.checked_add(plane_bytes).ok_or(PsdError::LimitExceeded("pattern padded size overflow"))?;
            if padded_bytes > MAX_PADDED_PATTERN_BYTES {
                return Err(PsdError::LimitExceeded("padded pattern channels exceed MAX_PADDED_PATTERN_BYTES"));
            }
        }
        let _ = layout.decoded_len()?;
        let compression = if comp == 1 {
            Compression::Rle
        } else if comp == 0 {
            Compression::Raw
        } else {
            Compression::Unknown(u16::from(comp))
        };
        let max_expansion = match compression {
            Compression::Raw => Some(1),
            Compression::Rle => Some(MAX_RLE_EXPANSION),
            Compression::Zip | Compression::ZipPrediction | Compression::Unknown(_) => None,
        };
        if let Some(max_expansion) = max_expansion {
            let expansion_limit = u64::try_from(data.len())
                .map_err(|_| PsdError::LimitExceeded("pattern channel input exceeds address space"))?
                .checked_mul(max_expansion)
                .ok_or(PsdError::LimitExceeded("pattern channel input size overflow"))?;
            if channel_bytes > expansion_limit {
                return Err(PsdError::LimitExceeded("pattern channel exceeds its input-derived size limit"));
            }
        }
        let x = usize::try_from(cl.checked_sub(left).ok_or(PsdError::LimitExceeded("pattern channel x offset overflow"))?)
            .map_err(|_| PsdError::invalid("negative pattern channel x offset"))?;
        let y = usize::try_from(ct.checked_sub(top).ok_or(PsdError::LimitExceeded("pattern channel y offset overflow"))?)
            .map_err(|_| PsdError::invalid("negative pattern channel y offset"))?;
        channels.push(PatternChannel { data, compression, source: channel_layout, destination: layout, x, y });
    }
    let nc = mode_channels(mode);
    if channels.len() < nc {
        return Err(PsdError::invalid("pattern has fewer channels than its colour mode"));
    }
    let mut planes = Vec::new();
    for channel in channels {
        let plane = decode_planes(channel.compression, channel.data, &channel.source)?;
        planes.push(place_channel_plane(plane, &channel.source, channel.x, channel.y, &channel.destination)?);
    }
    let alpha = (planes.len() > nc).then(|| planes.remove(nc));
    planes.truncate(nc);
    Ok(PsdPattern { mode, width: w, height: h, name, id, palette, depth, channels: planes, alpha })
}

fn place_channel_plane(mut plane: Vec<u8>, source: &PlaneLayout, x: usize, y: usize, destination: &PlaneLayout) -> Result<Vec<u8>> {
    if source.planes != 1 || destination.planes != 1 || source.depth != destination.depth {
        return Err(PsdError::invalid("incompatible pattern channel layouts"));
    }
    let x_end = x.checked_add(source.width).ok_or(PsdError::LimitExceeded("pattern channel x offset overflow"))?;
    let y_end = y.checked_add(source.height).ok_or(PsdError::LimitExceeded("pattern channel y offset overflow"))?;
    if x_end > destination.width || y_end > destination.height {
        return Err(PsdError::invalid("pattern channel rectangle falls outside pattern rectangle"));
    }
    let source_len = source.decoded_len()?;
    if plane.len() != source_len {
        return Err(PsdError::invalid("pattern channel payload does not match its rectangle"));
    }
    let destination_len = destination.decoded_len()?;
    if x == 0 && y == 0 && source.width == destination.width && source.height == destination.height {
        // Preserve the old placement's zeroed unused bits without copying a full
        // bitmap plane. Only the last byte of each non-byte-aligned row changes.
        if source.depth == 1 && !source.width.is_multiple_of(8) {
            let mask = u8::MAX << (8 - source.width % 8);
            for row in plane.chunks_exact_mut(source.row_bytes()) {
                if let Some(last) = row.last_mut() {
                    *last &= mask;
                }
            }
        }
        return Ok(plane);
    }
    let mut out = zeroed_pattern_plane(destination_len)?;
    if source.depth == 1 {
        let source_row_bytes = source.row_bytes();
        let destination_row_bytes = destination.row_bytes();
        for row in 0..source.height {
            for col in 0..source.width {
                let source_pos = row
                    .checked_mul(source_row_bytes)
                    .and_then(|v| v.checked_add(col / 8))
                    .ok_or(PsdError::LimitExceeded("pattern channel source offset overflow"))?;
                let destination_col = x.checked_add(col).ok_or(PsdError::LimitExceeded("pattern channel destination offset overflow"))?;
                let destination_pos = y
                    .checked_add(row)
                    .and_then(|v| v.checked_mul(destination_row_bytes))
                    .and_then(|v| v.checked_add(destination_col / 8))
                    .ok_or(PsdError::LimitExceeded("pattern channel destination offset overflow"))?;
                let source_bit = 7 - col % 8;
                let destination_bit = 7 - destination_col % 8;
                let source_byte = plane.get(source_pos).ok_or(PsdError::invalid("pattern channel payload does not match its rectangle"))?;
                let value = (*source_byte >> source_bit) & 1;
                let dst = out.get_mut(destination_pos).ok_or(PsdError::invalid("pattern channel placement exceeds its output plane"))?;
                *dst |= value << destination_bit;
            }
        }
    } else {
        let bytes_per_sample = usize::from(source.depth / 8);
        let source_row_bytes = source.row_bytes();
        let destination_row_bytes = destination.row_bytes();
        let row_bytes = source.width.checked_mul(bytes_per_sample).ok_or(PsdError::LimitExceeded("pattern channel row size overflow"))?;
        let x_bytes = x.checked_mul(bytes_per_sample).ok_or(PsdError::LimitExceeded("pattern channel x offset overflow"))?;
        for row in 0..source.height {
            let source_start = row.checked_mul(source_row_bytes).ok_or(PsdError::LimitExceeded("pattern channel source offset overflow"))?;
            let destination_start = y
                .checked_add(row)
                .and_then(|v| v.checked_mul(destination_row_bytes))
                .and_then(|v| v.checked_add(x_bytes))
                .ok_or(PsdError::LimitExceeded("pattern channel destination offset overflow"))?;
            let source_end = source_start.checked_add(row_bytes).ok_or(PsdError::LimitExceeded("pattern channel source offset overflow"))?;
            let destination_end = destination_start.checked_add(row_bytes).ok_or(PsdError::LimitExceeded("pattern channel destination offset overflow"))?;
            let source_row = plane.get(source_start..source_end).ok_or(PsdError::invalid("pattern channel payload does not match its rectangle"))?;
            let destination_row =
                out.get_mut(destination_start..destination_end).ok_or(PsdError::invalid("pattern channel placement exceeds its output plane"))?;
            destination_row.copy_from_slice(source_row);
        }
    }
    Ok(out)
}

fn zeroed_pattern_plane(len: usize) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(|_| PsdError::LimitExceeded("not enough memory for the padded pattern channel"))?;
    out.resize(len, 0);
    Ok(out)
}

fn write_pattern(p: &PsdPattern, out: &mut Vec<u8>) -> Result<()> {
    if p.width > MAX_EDGE || p.height > MAX_EDGE || p.width > i16::MAX as u32 || p.height > i16::MAX as u32 {
        return Err(PsdError::LimitExceeded("pattern size"));
    }
    out.extend_from_slice(&1u32.to_be_bytes());
    out.extend_from_slice(&p.mode.to_be_bytes());
    out.extend_from_slice(&(p.height as u16).to_be_bytes());
    out.extend_from_slice(&(p.width as u16).to_be_bytes());
    let units: Vec<u16> = p.name.encode_utf16().chain(std::iter::once(0)).collect();
    out.extend_from_slice(&(units.len() as u32).to_be_bytes());
    for u in units {
        out.extend_from_slice(&u.to_be_bytes());
    }
    let id = p.id.as_bytes();
    let idl = id.len().min(255);
    out.push(idl as u8);
    out.extend_from_slice(&id[..idl]);
    if p.mode == 2 {
        let mut pal = p.palette.clone().unwrap_or_default();
        pal.resize(768, 0);
        out.extend_from_slice(&pal);
    }
    let rect = [0i32, 0, p.height as i32, p.width as i32];
    let mut body = Vec::new();
    for v in rect {
        body.extend_from_slice(&v.to_be_bytes());
    }
    const SLOTS: u32 = 24;
    body.extend_from_slice(&SLOTS.to_be_bytes());
    let layout = PlaneLayout { planes: 1, width: p.width as usize, height: p.height as usize, depth: p.depth, version: Version::Psd };
    let array = |plane: &[u8], body: &mut Vec<u8>| -> Result<()> {
        let (comp, data) =
            if p.depth == 8 { (1u8, encode_planes(Compression::Rle, plane, &layout)?) } else { (0u8, encode_planes(Compression::Raw, plane, &layout)?) };
        body.extend_from_slice(&1u32.to_be_bytes());
        body.extend_from_slice(&((23 + data.len()) as u32).to_be_bytes());
        body.extend_from_slice(&u32::from(p.depth).to_be_bytes());
        for v in rect {
            body.extend_from_slice(&v.to_be_bytes());
        }
        body.extend_from_slice(&p.depth.to_be_bytes());
        body.push(comp);
        body.extend_from_slice(&data);
        Ok(())
    };
    let nc = p.channels.len().min(SLOTS as usize);
    for c in &p.channels[..nc] {
        array(c, &mut body)?;
    }
    for _ in nc..SLOTS as usize {
        body.extend_from_slice(&0u32.to_be_bytes());
    }
    // User mask slot: transparency; sheet mask slot: unused.
    match &p.alpha {
        Some(a) => array(a, &mut body)?,
        None => body.extend_from_slice(&0u32.to_be_bytes()),
    }
    body.extend_from_slice(&0u32.to_be_bytes());
    out.extend_from_slice(&3u32.to_be_bytes());
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    out.extend_from_slice(&body);
    Ok(())
}

/// Parses the data of a `Patt` / `Pat2` / `Pat3` global block.
pub fn parse_pattern_block(data: &[u8]) -> Result<Vec<PsdPattern>> {
    let mut r = Rd { b: data, p: 0 };
    let mut out = Vec::new();
    while r.p + 4 <= data.len() {
        let len = r.u32()? as usize;
        if len == 0 {
            break;
        }
        let body = r.take(len)?;
        out.push(read_pattern(&mut Rd { b: body, p: 0 })?);
        let pad = (4 - len % 4) % 4;
        r.p = (r.p + pad).min(data.len());
    }
    Ok(out)
}

/// Serializes patterns as `Patt` / `Pat2` / `Pat3` block data.
pub fn write_pattern_block(patterns: &[PsdPattern]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for p in patterns {
        let mut one = Vec::new();
        write_pattern(p, &mut one)?;
        out.extend_from_slice(&(one.len() as u32).to_be_bytes());
        let pad = (4 - one.len() % 4) % 4;
        out.extend_from_slice(&one);
        out.extend(std::iter::repeat_n(0u8, pad));
    }
    Ok(out)
}

/// Tagged-block key for patterns of a document at `depth` bits.
pub fn block_key(depth: u16) -> [u8; 4] {
    match depth {
        16 => *b"Pat2",
        32 => *b"Pat3",
        _ => *b"Patt",
    }
}

/// Parses a `.pat` pattern file.
pub fn parse_pat_file(data: &[u8]) -> Result<Vec<PsdPattern>> {
    let mut r = Rd { b: data, p: 0 };
    let sig = r.take(4)?;
    if sig != b"8BPT" {
        return Err(PsdError::InvalidSignature { expected: "8BPT", found: [sig[0], sig[1], sig[2], sig[3]] });
    }
    let _version = r.u16()?;
    let count = r.u32()?;
    let mut out = Vec::new();
    for _ in 0..count.min(100_000) {
        out.push(read_pattern(&mut r)?);
    }
    Ok(out)
}

/// Writes a `.pat` pattern file.
pub fn write_pat_file(patterns: &[PsdPattern]) -> Result<Vec<u8>> {
    let mut out = b"8BPT".to_vec();
    out.extend_from_slice(&1u16.to_be_bytes());
    out.extend_from_slice(&(patterns.len() as u32).to_be_bytes());
    for p in patterns {
        write_pattern(p, &mut out)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pat(depth: u16, alpha: bool) -> PsdPattern {
        let (w, h) = (5u32, 3u32);
        let bpp = usize::from(depth / 8);
        let plane = |s: u8| (0..w as usize * h as usize * bpp).map(|i| (i as u8).wrapping_mul(7).wrapping_add(s)).collect::<Vec<u8>>();
        PsdPattern {
            mode: 3,
            width: w,
            height: h,
            name: "Test ✓".into(),
            id: "0bd2d3ba-1234-11d4-8f8f-aabbccddeeff".into(),
            palette: None,
            depth,
            channels: vec![plane(1), plane(2), plane(3)],
            alpha: alpha.then(|| plane(9)),
        }
    }

    #[test]
    fn block_and_pat_file_round_trip_at_every_depth() {
        for depth in [8, 16, 32] {
            for alpha in [false, true] {
                let ps = vec![pat(depth, alpha), PsdPattern { name: "Second".into(), ..pat(depth, false) }];
                let blk = write_pattern_block(&ps).unwrap();
                assert_eq!(blk.len() % 4, 0);
                assert_eq!(parse_pattern_block(&blk).unwrap(), ps);
                let f = write_pat_file(&ps).unwrap();
                assert_eq!(parse_pat_file(&f).unwrap(), ps);
                // Re-serialising is byte-stable.
                assert_eq!(write_pattern_block(&parse_pattern_block(&blk).unwrap()).unwrap(), blk);
            }
        }
    }

    #[test]
    fn malformed_input_errors_without_panicking() {
        let blk = write_pattern_block(&[pat(8, true)]).unwrap();
        for cut in [3, 10, 40, blk.len() - 5] {
            let _ = parse_pattern_block(&blk[..cut]);
        }
        assert!(parse_pat_file(b"8BPS\0\x01").is_err());
        assert!(parse_pattern_block(&[]).unwrap().is_empty());
    }

    fn pattern_vma_rect_offset(pattern_start: usize, p: &PsdPattern) -> usize {
        let name_units = p.name.encode_utf16().count() + 1;
        pattern_start + 16 + name_units * 2 + 1 + p.id.len() + if p.mode == 2 { 768 } else { 0 } + 8
    }

    fn set_i32(bytes: &mut [u8], offset: usize, value: i32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_be_bytes());
    }

    #[test]
    fn hostile_pattern_rectangles_are_rejected_through_all_import_routes() {
        let p = pat(8, false);

        // A subtraction of the hostile endpoints overflowed before rectangle validation.
        let mut global = write_pattern_block(std::slice::from_ref(&p)).unwrap();
        let vma = pattern_vma_rect_offset(4, &p);
        set_i32(&mut global, vma + 4, i32::MIN);
        set_i32(&mut global, vma + 12, i32::MAX);
        assert!(parse_pattern_block(&global).is_err());

        let mut standalone = write_pat_file(std::slice::from_ref(&p)).unwrap();
        let vma = pattern_vma_rect_offset(10, &p);
        set_i32(&mut standalone, vma + 4, i32::MIN);
        set_i32(&mut standalone, vma + 12, i32::MAX);
        assert!(parse_pat_file(&standalone).is_err());

        let mut abr = crate::abr::write_v6(
            1,
            &[crate::abr::AbrSample { id: String::new(), width: 1, height: 1, depth: 8, data: vec![0] }],
            std::slice::from_ref(&p),
            &[],
            true,
        )
        .unwrap();
        let section = abr.windows(4).position(|w| w == b"patt").unwrap();
        let vma = pattern_vma_rect_offset(section + 8, &p) + 4;
        set_i32(&mut abr, vma + 4, i32::MIN);
        set_i32(&mut abr, vma + 12, i32::MAX);
        let parsed = crate::abr::parse(&abr).unwrap();
        assert!(parsed.patterns.is_empty());
        assert!(parsed.warnings.iter().any(|warning| warning.contains("embedded patterns unreadable")));
    }

    #[test]
    fn huge_pattern_rect_with_tiny_channel_payload_is_rejected() {
        // RGB32 needs 10.8 GB; CMYK16 plus alpha needs 9 GB. Both exceed the
        // existing pattern budget, but their first plane fits on 64-bit targets.
        // The complete metadata must be checked before allocating that first plane.
        huge_pattern_rect_is_rejected(pat(32, false));
        let mut cmyk = pat(16, true);
        cmyk.mode = 4;
        cmyk.channels.push(vec![0; 5 * 3 * 2]);
        huge_pattern_rect_is_rejected(cmyk);
    }

    fn huge_pattern_rect_is_rejected(p: PsdPattern) {
        let mut global = write_pattern_block(std::slice::from_ref(&p)).unwrap();
        let vma = pattern_vma_rect_offset(4, &p);
        set_i32(&mut global, vma + 8, 30_000);
        set_i32(&mut global, vma + 12, 30_000);
        assert!(matches!(parse_pattern_block(&global), Err(PsdError::LimitExceeded(_))));

        let mut standalone = write_pat_file(std::slice::from_ref(&p)).unwrap();
        let vma = pattern_vma_rect_offset(10, &p);
        set_i32(&mut standalone, vma + 8, 30_000);
        set_i32(&mut standalone, vma + 12, 30_000);
        assert!(matches!(parse_pat_file(&standalone), Err(PsdError::LimitExceeded(_))));

        let mut abr = crate::abr::write_v6(
            1,
            &[crate::abr::AbrSample { id: String::new(), width: 1, height: 1, depth: 8, data: vec![0] }],
            std::slice::from_ref(&p),
            &[],
            true,
        )
        .unwrap();
        let section = abr.windows(4).position(|w| w == b"patt").unwrap();
        let vma = pattern_vma_rect_offset(section + 8, &p) + 4;
        set_i32(&mut abr, vma + 8, 30_000);
        set_i32(&mut abr, vma + 12, 30_000);
        let parsed = crate::abr::parse(&abr).unwrap();
        assert!(parsed.patterns.is_empty());
        assert!(parsed.warnings.iter().any(|warning| warning.contains("embedded patterns unreadable")));
    }

    #[test]
    fn offset_channel_rectangles_are_placed_inside_the_pattern() {
        for depth in [8, 16, 32] {
            let original = pat(depth, true);
            let mut block = write_pattern_block(std::slice::from_ref(&original)).unwrap();
            let vma = pattern_vma_rect_offset(4, &original);
            set_i32(&mut block, vma, -2);
            set_i32(&mut block, vma + 4, -3);
            set_i32(&mut block, vma + 8, 4);
            set_i32(&mut block, vma + 12, 5);

            let parsed = parse_pattern_block(&block).unwrap();
            let parsed = parsed.first().unwrap();
            assert_eq!((parsed.width, parsed.height), (8, 6));
            let bytes_per_sample = usize::from(depth / 8);
            for (actual, expected) in parsed.channels.iter().zip(&original.channels).chain(parsed.alpha.iter().zip(original.alpha.iter())) {
                for y in 0..6usize {
                    for x in 0..8usize {
                        let at = (y * 8 + x) * bytes_per_sample;
                        let actual_sample = actual.get(at..at + bytes_per_sample).unwrap();
                        if (2..5).contains(&y) && (3..8).contains(&x) {
                            let source_at = ((y - 2) * 5 + (x - 3)) * bytes_per_sample;
                            assert_eq!(actual_sample, expected.get(source_at..source_at + bytes_per_sample).unwrap());
                        } else {
                            assert!(actual_sample.iter().all(|&sample| sample == 0));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn one_bit_offset_channel_placement_preserves_msb_first_pixels() {
        let source = PlaneLayout { planes: 1, width: 5, height: 2, depth: 1, version: Version::Psd };
        let destination = PlaneLayout { planes: 1, width: 8, height: 4, depth: 1, version: Version::Psd };
        let placed = place_channel_plane(vec![0b1010_1000, 0b0101_0000], &source, 2, 1, &destination).unwrap();
        assert_eq!(placed, [0, 0b0010_1010, 0b0001_0100, 0]);
    }

    #[test]
    fn full_size_channels_reuse_the_decoded_allocation() {
        for depth in [1, 8, 16, 32] {
            let layout = PlaneLayout { planes: 1, width: 5, height: 3, depth, version: Version::Psd };
            let plane = vec![0x55; layout.decoded_len().unwrap()];
            let original = plane.as_ptr();
            let placed = place_channel_plane(plane, &layout, 0, 0, &layout).unwrap();
            assert_eq!(placed.as_ptr(), original);
            let expected = if depth == 1 { 0x50 } else { 0x55 };
            assert!(placed.iter().all(|&byte| byte == expected));
        }
    }

    #[test]
    fn large_channel_padding_within_the_existing_budget_is_allowed() {
        let mut p = pat(8, false);
        p.mode = 1;
        let mut block = write_pattern_block(std::slice::from_ref(&p)).unwrap();
        let vma = pattern_vma_rect_offset(4, &p);
        // More than 64 MiB of implicit padding is valid: the fix must not introduce
        // a smaller, padding-specific format limit to avoid the oversized RGB case.
        set_i32(&mut block, vma + 8, 8193);
        set_i32(&mut block, vma + 12, 8193);
        let parsed = parse_pattern_block(&block).unwrap();
        let plane = &parsed[0].channels[0];
        assert_eq!(plane.len(), 8193 * 8193);
        for y in 0..3 {
            assert_eq!(&plane[y * 8193..y * 8193 + 5], &p.channels[0][y * 5..y * 5 + 5]);
        }
        assert_eq!(plane[5], 0);
        assert_eq!(plane[plane.len() - 1], 0);
    }

    #[test]
    fn padded_plane_allocation_failure_is_an_error() {
        // Capacity overflow is deterministic and never asks the OS for huge memory.
        assert!(matches!(zeroed_pattern_plane(usize::MAX), Err(PsdError::LimitExceeded("not enough memory for the padded pattern channel"))));
    }

    #[test]
    fn tiny_channels_in_a_huge_pattern_rect_are_not_zero_padded_without_bound() {
        // 3 channels of 5x3 padded to 30000x30000 would allocate 2.7 GB of zeros.
        let p = pat(8, false);
        let mut global = write_pattern_block(std::slice::from_ref(&p)).unwrap();
        let vma = pattern_vma_rect_offset(4, &p);
        set_i32(&mut global, vma + 8, 30_000);
        set_i32(&mut global, vma + 12, 30_000);
        assert!(matches!(parse_pattern_block(&global), Err(PsdError::LimitExceeded("padded pattern channels exceed MAX_PADDED_PATTERN_BYTES"))));
    }

    #[test]
    fn zero_size_pattern_rectangles_are_rejected() {
        let p = pat(8, false);
        let mut block = write_pattern_block(std::slice::from_ref(&p)).unwrap();
        let vma = pattern_vma_rect_offset(4, &p);
        set_i32(&mut block, vma + 8, 0);
        assert!(parse_pattern_block(&block).is_err());
    }
}
