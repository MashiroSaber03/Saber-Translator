//! Sparse, copy-on-write tiled surfaces in any [`PixelFormat`].
//!
//! A [`Surface`] is an infinite plane of 256×256 tiles. Missing tiles read as the surface's
//! default pixel (transparent for layers, white for "reveal all" masks). Tiles are `Arc`-shared,
//! so cloning a surface is O(tiles) and mutation copies only the touched tiles. That makes
//! undo snapshots, background jobs and autosave cheap.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

mod interrupt;
pub use interrupt::{Cancelled, Interrupt};

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use photocraft_color::{ColorMode, PixelFormat, SampleType, read_sample, write_sample};
use photocraft_geom::{Rect, TILE_SIZE, TileCoord};

/// A checked surface allocation could not be represented or reserved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationError {
    SizeOverflow,
    NotEnoughMemory { bytes: usize },
}

impl fmt::Display for AllocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SizeOverflow => f.write_str("requested surface allocation is too large"),
            Self::NotEnoughMemory { bytes } => write!(f, "not enough memory to allocate {} bytes", bytes),
        }
    }
}

impl std::error::Error for AllocationError {}

/// Pixel storage for one tile: `TILE_SIZE² × bytes_per_pixel`, row-major, interleaved channels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tile {
    data: Box<[u8]>,
}

impl Tile {
    fn try_filled(format: &PixelFormat, pixel: &[u8]) -> Result<Self, AllocationError> {
        let n = (TILE_SIZE * TILE_SIZE) as usize;
        let bytes = n.checked_mul(format.bytes_per_pixel()).ok_or(AllocationError::SizeOverflow)?;
        let mut data = Vec::new();
        data.try_reserve_exact(bytes).map_err(|_| AllocationError::NotEnoughMemory { bytes })?;
        data.resize(bytes, 0);
        if pixel.iter().any(|&b| b != 0) {
            for px in data.chunks_exact_mut(pixel.len()) {
                px.copy_from_slice(pixel);
            }
        }
        Ok(Tile { data: data.into_boxed_slice() })
    }

    /// Clone this tile's pixel buffer with a fallible reservation.
    pub fn try_clone(&self) -> Result<Self, AllocationError> {
        let bytes = self.data.len();
        let mut data = Vec::new();
        data.try_reserve_exact(bytes).map_err(|_| AllocationError::NotEnoughMemory { bytes })?;
        data.extend_from_slice(&self.data);
        Ok(Tile { data: data.into_boxed_slice() })
    }

    fn filled(format: &PixelFormat, pixel: &[u8]) -> Self {
        let n = (TILE_SIZE * TILE_SIZE) as usize;
        let mut data = vec![0u8; n * format.bytes_per_pixel()];
        if pixel.iter().any(|&b| b != 0) {
            for px in data.chunks_exact_mut(pixel.len()) {
                px.copy_from_slice(pixel);
            }
        }
        Tile { data: data.into_boxed_slice() }
    }
    pub fn bytes(&self) -> &[u8] {
        &self.data
    }
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Tile-local half-open bounds `(x0, y0, x1, y1)` of the pixels that differ from the encoded
    /// pixel `dp` (one pixel of the tile's format), or `None` when every pixel equals it (or `dp`
    /// is empty).
    pub fn content_bounds(&self, dp: &[u8]) -> Option<(u16, u16, u16, u16)> {
        let bpp = dp.len();
        if bpp == 0 {
            return None;
        }
        let ts = TILE_SIZE as usize;
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
        for (y, row) in self.data.chunks_exact(ts * bpp).enumerate() {
            let Some(first) = row.chunks_exact(bpp).position(|px| px != dp) else { continue };
            let last = ts - 1 - row.chunks_exact(bpp).rev().position(|px| px != dp).unwrap_or(0);
            x0 = x0.min(first);
            x1 = x1.max(last + 1);
            y0 = y0.min(y);
            y1 = y + 1;
        }
        (x0 != usize::MAX).then_some((x0 as u16, y0 as u16, x1 as u16, y1 as u16))
    }
}

#[derive(Clone, Debug)]
pub struct Surface {
    format: PixelFormat,
    default_pixel: Box<[u8]>,
    tiles: BTreeMap<TileCoord, Arc<Tile>>,
}

/// Tiles of one pixel value met while moving a surface ([`Surface::translated`]): each scanned
/// once, and one tile kept per value so the targets share it.
#[derive(Default)]
struct UniformTiles {
    /// Per source tile (by address): the index of its value's tile in `canonical`, or `None`.
    of: std::collections::HashMap<*const Tile, Option<usize>>,
    canonical: Vec<Arc<Tile>>,
}

impl UniformTiles {
    /// The tile to share for a target reading window `r` of `s`, when every tile under it holds
    /// one and the same pixel value (`r` must lie on existing tiles).
    fn window(&mut self, s: &Surface, r: Rect) -> Option<Arc<Tile>> {
        let bpp = s.format.bytes_per_pixel();
        let mut found = None;
        for c in r.tiles() {
            let t = s.tiles.get(&c)?;
            let k = self.value_of(t, bpp)?;
            if found.is_some_and(|f| f != k) {
                return None;
            }
            found = Some(k);
        }
        self.canonical.get(found?).cloned()
    }

    fn value_of(&mut self, t: &Arc<Tile>, bpp: usize) -> Option<usize> {
        if let Some(k) = self.of.get(&Arc::as_ptr(t)) {
            return *k;
        }
        let mut px = t.data.chunks_exact(bpp);
        let k = px.next().filter(|p0| px.all(|p| p == *p0)).map(|p0| match self.canonical.iter().position(|c| c.data.get(..bpp) == Some(p0)) {
            Some(k) => k,
            None => {
                self.canonical.push(t.clone());
                self.canonical.len() - 1
            }
        });
        self.of.insert(Arc::as_ptr(t), k);
        k
    }
}

impl PartialEq for Surface {
    fn eq(&self, other: &Self) -> bool {
        self.format == other.format && self.default_pixel == other.default_pixel && self.tiles == other.tiles
    }
}

impl Surface {
    /// Reads a region into a newly allocated buffer, returning an error instead of aborting if
    /// the requested pixel buffer cannot be represented or reserved.
    pub fn try_read_region(&self, r: Rect) -> Result<Vec<f32>, AllocationError> {
        let len = (r.width() as usize).checked_mul(r.height() as usize).and_then(|n| n.checked_mul(self.channels())).ok_or(AllocationError::SizeOverflow)?;
        let bytes = len.checked_mul(std::mem::size_of::<f32>()).ok_or(AllocationError::SizeOverflow)?;
        let mut out = Vec::new();
        out.try_reserve_exact(len).map_err(|_| AllocationError::NotEnoughMemory { bytes })?;
        out.resize(len, 0.0);
        self.read_region_into(r, &mut out);
        Ok(out)
    }

    /// Reads a region as encoded interleaved bytes with fallible allocation.
    pub fn try_to_interleaved(&self, r: Rect) -> Result<Vec<u8>, AllocationError> {
        let bpp = self.format.bytes_per_pixel();
        let len = (r.width() as usize).checked_mul(r.height() as usize).and_then(|n| n.checked_mul(bpp)).ok_or(AllocationError::SizeOverflow)?;
        let mut out = Vec::new();
        out.try_reserve_exact(len).map_err(|_| AllocationError::NotEnoughMemory { bytes: len })?;
        out.resize(len, 0);
        let w = r.width() as usize;
        for tc in r.tiles() {
            let tr = tc.rect().intersect(&r);
            let span = tr.width() as usize * bpp;
            for y in tr.y0..tr.y1 {
                let dst = (((y - r.y0) as usize) * w + (tr.x0 - r.x0) as usize) * bpp;
                match self.tiles.get(&tc) {
                    Some(t) => {
                        let src = (((y - tc.ty * TILE_SIZE) as usize) * TILE_SIZE as usize + (tr.x0 - tc.tx * TILE_SIZE) as usize) * bpp;
                        out[dst..dst + span].copy_from_slice(&t.data[src..src + span]);
                    }
                    None => {
                        for px in out[dst..dst + span].chunks_exact_mut(bpp) {
                            px.copy_from_slice(&self.default_pixel);
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// Transparent/zero surface.
    pub fn new(format: PixelFormat) -> Self {
        Self { format, default_pixel: vec![0u8; format.bytes_per_pixel()].into_boxed_slice(), tiles: BTreeMap::new() }
    }

    /// Surface whose untouched pixels read as `pixel` (normalised channel values).
    pub fn with_default(format: PixelFormat, pixel: &[f32]) -> Self {
        let mut s = Self::new(format);
        encode_pixel(&format, pixel, &mut s.default_pixel);
        s
    }

    pub fn format(&self) -> PixelFormat {
        self.format
    }
    pub fn channels(&self) -> usize {
        self.format.channels()
    }
    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }
    pub fn tiles(&self) -> impl Iterator<Item = (&TileCoord, &Arc<Tile>)> {
        self.tiles.iter()
    }
    pub fn tile(&self, c: TileCoord) -> Option<&Arc<Tile>> {
        self.tiles.get(&c)
    }
    pub fn default_pixel(&self) -> Vec<f32> {
        decode_pixel(&self.format, &self.default_pixel)
    }

    /// Mutable access to a tile, allocating it (filled with the default pixel) or un-sharing it.
    pub fn tile_mut(&mut self, c: TileCoord) -> &mut Tile {
        let fmt = self.format;
        let dp = self.default_pixel.clone();
        let arc = self.tiles.entry(c).or_insert_with(|| Arc::new(Tile::filled(&fmt, &dp)));
        Arc::make_mut(arc)
    }

    /// Fallible counterpart to [`Surface::tile_mut`]. Pixel storage is reserved before changing
    /// the surface, and a shared tile is copied only after its replacement has been allocated.
    pub fn try_tile_mut(&mut self, c: TileCoord) -> Result<&mut Tile, AllocationError> {
        if !self.tiles.contains_key(&c) {
            let tile = Arc::new(Tile::try_filled(&self.format, &self.default_pixel)?);
            self.tiles.insert(c, tile);
        } else if self.tiles.get(&c).is_some_and(|t| Arc::strong_count(t) > 1) {
            let original = self.tiles.get(&c).ok_or(AllocationError::SizeOverflow)?;
            let tile = Arc::new(original.try_clone()?);
            self.tiles.insert(c, tile);
        }
        self.tiles.get_mut(&c).and_then(Arc::get_mut).ok_or(AllocationError::SizeOverflow)
    }

    /// Writes a region using fallible pixel-tile allocations. Callers that need transactional
    /// document updates should perform this on their staged document copy and commit on success.
    pub fn try_write_region(&mut self, r: Rect, data: &[f32]) -> Result<(), AllocationError> {
        let n = self.channels();
        let sample = self.format.sample;
        let expected = (r.width() as usize).checked_mul(r.height() as usize).and_then(|v| v.checked_mul(n)).ok_or(AllocationError::SizeOverflow)?;
        if data.len() != expected {
            return Err(AllocationError::SizeOverflow);
        }
        for tc in r.tiles() {
            let tr = tc.rect().intersect(&r);
            let t = self.try_tile_mut(tc)?;
            let span = tr.width() as usize * n;
            for y in tr.y0..tr.y1 {
                let src = (((y - r.y0) as usize) * r.width() as usize + (tr.x0 - r.x0) as usize) * n;
                let dst = (((y - tc.ty * TILE_SIZE) as usize) * TILE_SIZE as usize + (tr.x0 - tc.tx * TILE_SIZE) as usize) * n;
                for (i, v) in data[src..src + span].iter().enumerate() {
                    write_sample(&mut t.data, sample, dst + i, *v);
                }
            }
        }
        Ok(())
    }

    /// Coarse bounds: union of allocated tile rects.
    pub fn tile_bounds(&self) -> Rect {
        self.tiles.keys().fold(Rect::EMPTY, |r, c| r.union(&c.rect()))
    }

    /// Exact bounds of pixels that differ from the default pixel.
    pub fn content_bounds(&self) -> Rect {
        if self.tiles.is_empty() {
            return Rect::EMPTY;
        }
        // Scan tiles from the outside of the tile grid inwards; a tile lying entirely inside the
        // bounds found so far cannot extend them and is skipped (so a full-canvas selection scans
        // roughly its outer ring of tiles instead of every pixel).
        let (mut gx0, mut gy0, mut gx1, mut gy1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
        for c in self.tiles.keys() {
            (gx0, gy0, gx1, gy1) = (gx0.min(c.tx), gy0.min(c.ty), gx1.max(c.tx), gy1.max(c.ty));
        }
        let mut order: Vec<(&TileCoord, &Arc<Tile>)> = self.tiles.iter().collect();
        order.sort_by_key(|(c, _)| (c.tx - gx0).min(gx1 - c.tx).min(c.ty - gy0).min(gy1 - c.ty));
        let mut out = Rect::EMPTY;
        for (c, t) in order {
            let origin = c.rect();
            if !out.is_empty() && out.contains_rect(&origin) {
                continue;
            }
            if let Some((x0, y0, x1, y1)) = t.content_bounds(&self.default_pixel) {
                out = out.union(&Rect::new(x0.into(), y0.into(), x1.into(), y1.into()).translate(origin.x0, origin.y0));
            }
        }
        out
    }

    /// Drop tiles that are entirely the default pixel.
    pub fn prune(&mut self) {
        let dp = self.default_pixel.clone();
        let bpp = dp.len();
        self.tiles.retain(|_, t| t.data.chunks_exact(bpp).any(|px| px != &*dp));
    }

    #[inline]
    fn locate(&self, x: i32, y: i32) -> (TileCoord, usize) {
        let c = TileCoord::containing(x, y);
        let lx = x.rem_euclid(TILE_SIZE) as usize;
        let ly = y.rem_euclid(TILE_SIZE) as usize;
        (c, (ly * TILE_SIZE as usize + lx) * self.format.channels())
    }

    /// Read one pixel as normalised channel values (`channels()` floats).
    pub fn pixel(&self, x: i32, y: i32) -> Vec<f32> {
        let mut out = vec![0.0; self.channels()];
        self.read_pixel(x, y, &mut out);
        out
    }

    pub fn read_pixel(&self, x: i32, y: i32, out: &mut [f32]) {
        let (c, base) = self.locate(x, y);
        let n = self.channels();
        match self.tiles.get(&c) {
            Some(t) => {
                for (i, o) in out.iter_mut().enumerate().take(n) {
                    *o = read_sample(&t.data, self.format.sample, base + i);
                }
            }
            None => {
                for (i, o) in out.iter_mut().enumerate().take(n) {
                    *o = read_sample(&self.default_pixel, self.format.sample, i);
                }
            }
        }
    }

    pub fn write_pixel(&mut self, x: i32, y: i32, px: &[f32]) {
        let (c, base) = self.locate(x, y);
        let n = self.channels();
        let sample = self.format.sample;
        let t = self.tile_mut(c);
        for (i, v) in px.iter().enumerate().take(n) {
            write_sample(&mut t.data, sample, base + i, *v);
        }
    }

    /// One channel of one pixel, without allocating.
    #[inline]
    pub fn sample_channel(&self, x: i32, y: i32, c: usize) -> f32 {
        let (tc, base) = self.locate(x, y);
        match self.tiles.get(&tc) {
            Some(t) => read_sample(&t.data, self.format.sample, base + c),
            None => read_sample(&self.default_pixel, self.format.sample, c),
        }
    }

    /// Channel `c` at `(x0 + i·step, y)` for `i in 0..out.len()`, looking each tile up once per run
    /// (much faster than per-sample [`Surface::sample_channel`] for strided scans).
    pub fn sample_row_strided(&self, y: i32, x0: i32, step: i32, c: usize, out: &mut [f32]) {
        let step = step.max(1);
        let dp = read_sample(&self.default_pixel, self.format.sample, c);
        let n = self.format.channels();
        let ty = y.div_euclid(TILE_SIZE);
        let ly = y.rem_euclid(TILE_SIZE) as usize;
        let mut i = 0;
        while i < out.len() {
            let x = x0 + i as i32 * step;
            let tx = x.div_euclid(TILE_SIZE);
            // Samples that fall in this tile.
            let tile_end = (tx + 1) * TILE_SIZE;
            let count = (((tile_end - x) + step - 1) / step).max(1) as usize;
            let end = (i + count).min(out.len());
            match self.tiles.get(&TileCoord { tx, ty }) {
                None => out[i..end].fill(dp),
                Some(t) => {
                    for (k, o) in out[i..end].iter_mut().enumerate() {
                        let lx = (x + k as i32 * step - tx * TILE_SIZE) as usize;
                        *o = read_sample(&t.data, self.format.sample, (ly * TILE_SIZE as usize + lx) * n + c);
                    }
                }
            }
            i = end;
        }
    }

    /// [`Surface::read_region`] into a reusable buffer (resized to fit).
    pub fn read_region_into(&self, r: Rect, out: &mut Vec<f32>) {
        let n = self.channels();
        let w = r.width() as usize;
        out.clear();
        out.resize(w * r.height() as usize * n, 0.0);
        let dp = self.default_pixel();
        for tc in r.tiles() {
            let tr = tc.rect().intersect(&r);
            let tile = self.tiles.get(&tc);
            for y in tr.y0..tr.y1 {
                let o = (((y - r.y0) as usize) * w + (tr.x0 - r.x0) as usize) * n;
                let span = tr.width() as usize * n;
                let dst = &mut out[o..o + span];
                match tile {
                    Some(t) => {
                        let base = (((y - tc.ty * TILE_SIZE) as usize) * TILE_SIZE as usize + (tr.x0 - tc.tx * TILE_SIZE) as usize) * n;
                        for (i, d) in dst.iter_mut().enumerate() {
                            *d = read_sample(&t.data, self.format.sample, base + i);
                        }
                    }
                    None => {
                        for px in dst.chunks_exact_mut(n) {
                            px.copy_from_slice(&dp);
                        }
                    }
                }
            }
        }
    }

    /// Reads `r` as straight RGBA floats (converting from the surface's
    /// model) into `out` (`w*h` entries), without per-pixel allocation.
    /// Fast paths for RGB(A) and gray(A) at 8/16/32 bits read tile bytes
    /// directly; missing tiles are filled with the default pixel.
    pub fn read_rgba_into(&self, r: Rect, out: &mut [[f32; 4]]) {
        let w = r.width() as usize;
        debug_assert_eq!(out.len(), w * r.height() as usize);
        let fmt = self.format;
        let n = fmt.channels();
        let dp = to_rgba(&fmt, &self.default_pixel());
        for tc in r.tiles() {
            let tr = tc.rect().intersect(&r);
            let tile = self.tiles.get(&tc);
            for y in tr.y0..tr.y1 {
                let o = ((y - r.y0) as usize) * w + (tr.x0 - r.x0) as usize;
                let dst = &mut out[o..o + tr.width() as usize];
                let Some(t) = tile else {
                    dst.fill(dp);
                    continue;
                };
                let base = (((y - tc.ty * TILE_SIZE) as usize) * TILE_SIZE as usize + (tr.x0 - tc.tx * TILE_SIZE) as usize) * n;
                match (fmt.mode, fmt.sample, fmt.alpha) {
                    (ColorMode::Rgb, SampleType::U8, true) => {
                        let src = &t.data[base..base + dst.len() * 4];
                        for (d, s) in dst.iter_mut().zip(src.as_chunks::<4>().0) {
                            *d = [s[0] as f32 / 255.0, s[1] as f32 / 255.0, s[2] as f32 / 255.0, s[3] as f32 / 255.0];
                        }
                    }
                    (ColorMode::Grayscale, SampleType::U8, true) => {
                        let src = &t.data[base..base + dst.len() * 2];
                        for (d, s) in dst.iter_mut().zip(src.as_chunks::<2>().0) {
                            let g = s[0] as f32 / 255.0;
                            *d = [g, g, g, s[1] as f32 / 255.0];
                        }
                    }
                    (ColorMode::Cmyk, SampleType::U8, _) => {
                        // Identical ink bytes share the exact ICC result; alpha is independent.
                        let mut previous: Option<([u8; 4], [f32; 3])> = None;
                        let src = &t.data[base..base + dst.len() * n];
                        for (d, s) in dst.iter_mut().zip(src.chunks_exact(n)) {
                            let ink = [s[0], s[1], s[2], s[3]];
                            let rgb = match previous {
                                Some((key, rgb)) if key == ink => rgb,
                                _ => {
                                    let rgb = photocraft_color::convert::cmyk_to_rgb(ink.map(|v| v as f32 / 255.0));
                                    previous = Some((ink, rgb));
                                    rgb
                                }
                            };
                            *d = [rgb[0], rgb[1], rgb[2], s.get(4).map_or(1.0, |v| *v as f32 / 255.0)];
                        }
                    }
                    _ => {
                        let mut px = [0.0f32; 8];
                        for (i, d) in dst.iter_mut().enumerate() {
                            for (c, v) in px.iter_mut().enumerate().take(n) {
                                *v = read_sample(&t.data, fmt.sample, base + i * n + c);
                            }
                            *d = to_rgba(&fmt, &px[..n]);
                        }
                    }
                }
            }
        }
    }

    /// Reads `r` as straight 8-bit RGBA into `out` (`w*h` entries). RGBA8 surfaces copy tile bytes
    /// directly; other formats convert through [`Surface::read_rgba_into`] one tile row at a time.
    pub fn read_rgba8_into(&self, r: Rect, out: &mut [[u8; 4]]) {
        let w = r.width() as usize;
        debug_assert_eq!(out.len(), w * r.height() as usize);
        let fmt = self.format;
        let rgba8 = matches!((fmt.mode, fmt.sample, fmt.alpha), (ColorMode::Rgb, SampleType::U8, true));
        let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        let dp = to_rgba(&fmt, &self.default_pixel());
        let dp8 = [q(dp[0]), q(dp[1]), q(dp[2]), q(dp[3])];
        let mut row = Vec::new();
        for tc in r.tiles() {
            let tr = tc.rect().intersect(&r);
            let tile = self.tiles.get(&tc);
            for y in tr.y0..tr.y1 {
                let o = ((y - r.y0) as usize) * w + (tr.x0 - r.x0) as usize;
                let dst = &mut out[o..o + tr.width() as usize];
                match tile {
                    None => dst.fill(dp8),
                    Some(t) if rgba8 => {
                        let base = (((y - tc.ty * TILE_SIZE) as usize) * TILE_SIZE as usize + (tr.x0 - tc.tx * TILE_SIZE) as usize) * 4;
                        let len = dst.len();
                        for (d, s) in dst.iter_mut().zip(t.data[base..base + len * 4].as_chunks::<4>().0) {
                            *d = [s[0], s[1], s[2], s[3]];
                        }
                    }
                    Some(_) => {
                        row.resize(dst.len(), [0.0f32; 4]);
                        self.read_rgba_into(Rect::new(tr.x0, y, tr.x1, y + 1), &mut row);
                        for (d, p) in dst.iter_mut().zip(&row) {
                            *d = [q(p[0]), q(p[1]), q(p[2]), q(p[3])];
                        }
                    }
                }
            }
        }
    }

    /// `true` if any allocated tile intersects `r` (else `r` reads as the
    /// default pixel everywhere).
    pub fn has_tiles_in(&self, r: Rect) -> bool {
        r.tiles().any(|tc| self.tiles.contains_key(&tc))
    }

    /// Read a rectangle as interleaved normalised floats (`w*h*channels`).
    pub fn read_region(&self, r: Rect) -> Vec<f32> {
        let n = self.channels();
        let w = r.width() as usize;
        let mut out = vec![0.0f32; w * r.height() as usize * n];
        let dp = self.default_pixel();
        for tc in r.tiles() {
            let tr = tc.rect().intersect(&r);
            let tile = self.tiles.get(&tc);
            for y in tr.y0..tr.y1 {
                for x in tr.x0..tr.x1 {
                    let o = (((y - r.y0) as usize) * w + (x - r.x0) as usize) * n;
                    match tile {
                        Some(t) => {
                            let base = (((y - tc.ty * TILE_SIZE) as usize) * TILE_SIZE as usize + (x - tc.tx * TILE_SIZE) as usize) * n;
                            for i in 0..n {
                                out[o + i] = read_sample(&t.data, self.format.sample, base + i);
                            }
                        }
                        None => out[o..o + n].copy_from_slice(&dp),
                    }
                }
            }
        }
        out
    }

    /// Write interleaved normalised floats into a rectangle.
    pub fn write_region(&mut self, r: Rect, data: &[f32]) {
        let n = self.channels();
        let w = r.width() as usize;
        assert_eq!(data.len(), w * r.height() as usize * n, "region data length mismatch");
        let sample = self.format.sample;
        for tc in r.tiles() {
            let tr = tc.rect().intersect(&r);
            let t = self.tile_mut(tc);
            // Convert contiguous channel samples a row at a time. Dispatching on depth here
            // removes the per-pixel channel loop and lets the compiler vectorise conversion.
            let span = tr.width() as usize * n;
            for y in tr.y0..tr.y1 {
                let o = (((y - r.y0) as usize) * w + (tr.x0 - r.x0) as usize) * n;
                let base = (((y - tc.ty * TILE_SIZE) as usize) * TILE_SIZE as usize + (tr.x0 - tc.tx * TILE_SIZE) as usize) * n;
                let src = &data[o..o + span];
                let bytes = sample.bytes();
                let dst = &mut t.data[base * bytes..(base + span) * bytes];
                match sample {
                    SampleType::U8 => {
                        for (d, v) in dst.iter_mut().zip(src) {
                            *d = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                        }
                    }
                    SampleType::U16 => {
                        for (d, v) in dst.as_chunks_mut::<2>().0.iter_mut().zip(src) {
                            let q = (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
                            d.copy_from_slice(&q.to_ne_bytes());
                        }
                    }
                    SampleType::F32 => {
                        for (d, v) in dst.as_chunks_mut::<4>().0.iter_mut().zip(src) {
                            d.copy_from_slice(&v.to_ne_bytes());
                        }
                    }
                }
            }
        }
    }

    /// Take the tiles covering `r` out of the surface (missing ones filled with the default
    /// pixel), so they can be edited elsewhere, e.g. in parallel: `Arc::unwrap_or_clone` copies a
    /// tile only if an undo snapshot still shares it. Give them back with [`Surface::put_tiles`].
    pub fn take_tiles(&mut self, r: Rect) -> Vec<(TileCoord, Arc<Tile>)> {
        let fmt = self.format;
        let mut blank: Option<Arc<Tile>> = None;
        r.tiles()
            .map(|c| {
                let t = self.tiles.remove(&c).unwrap_or_else(|| blank.get_or_insert_with(|| Arc::new(Tile::filled(&fmt, &self.default_pixel))).clone());
                (c, t)
            })
            .collect()
    }

    /// Fallible counterpart to [`Surface::take_tiles`]. Reserves the complete result and any
    /// default tile before removing a tile, so a reservation failure leaves the surface intact.
    pub fn try_take_tiles(&mut self, r: Rect) -> Result<Vec<(TileCoord, Arc<Tile>)>, AllocationError> {
        let count = r.tiles().count();
        let bytes = count.checked_mul(std::mem::size_of::<(TileCoord, Arc<Tile>)>()).ok_or(AllocationError::SizeOverflow)?;
        let mut out = Vec::new();
        out.try_reserve_exact(count).map_err(|_| AllocationError::NotEnoughMemory { bytes })?;
        let has_missing = r.tiles().any(|c| !self.tiles.contains_key(&c));
        let blank = if has_missing { Some(Arc::new(Tile::try_filled(&self.format, &self.default_pixel)?)) } else { None };
        for c in r.tiles() {
            let tile = self.tiles.remove(&c).or_else(|| blank.clone());
            if let Some(tile) = tile {
                out.push((c, tile));
            }
        }
        Ok(out)
    }

    /// Put tiles back (see [`Surface::take_tiles`]); each replaces the tile at its coordinate.
    pub fn put_tiles(&mut self, tiles: impl IntoIterator<Item = (TileCoord, Arc<Tile>)>) {
        for (c, t) in tiles {
            self.tiles.insert(c, t);
        }
    }

    /// Restore COW sharing after decoding a new version of the same surface.
    /// Changed tiles remain owned by this version, so undo still restores their pixels.
    pub fn reuse_tiles_from(&mut self, previous: &Self) {
        if self.format != previous.format {
            return;
        }
        for (coord, tile) in &mut self.tiles {
            if let Some(old) = previous.tiles.get(coord)
                && old == tile
            {
                *tile = old.clone();
            }
        }
    }

    /// A tile filled with the default pixel (what a missing tile reads as), allocated fallibly.
    pub fn try_default_tile(&self) -> Result<Tile, AllocationError> {
        Tile::try_filled(&self.format, &self.default_pixel)
    }

    /// A tile of this surface's format filled with `px` (normalised channel values). Sharing one
    /// across many coordinates fills them without copying (tiles are copy-on-write).
    pub fn solid_tile(&self, px: &[f32]) -> Arc<Tile> {
        let mut enc = vec![0u8; self.format.bytes_per_pixel()];
        encode_pixel(&self.format, px, &mut enc);
        Arc::new(Tile::filled(&self.format, &enc))
    }

    /// The default pixel's encoded bytes.
    pub fn default_bytes(&self) -> &[u8] {
        &self.default_pixel
    }

    /// Fill a rectangle with one pixel value.
    pub fn fill_rect(&mut self, r: Rect, px: &[f32]) {
        let n = self.channels();
        let mut enc = vec![0u8; self.format.bytes_per_pixel()];
        encode_pixel(&self.format, &px[..n], &mut enc);
        let bpp = enc.len();
        // Fully covered tiles share one Arc: a solid fill costs one tile however large `r` is,
        // and copy-on-write splits a tile off when it is edited (#705).
        let mut full: Option<Arc<Tile>> = None;
        for tc in r.tiles() {
            let tr = tc.rect().intersect(&r);
            if tr == tc.rect() {
                let fmt = self.format;
                let t = full.get_or_insert_with(|| Arc::new(Tile::filled(&fmt, &enc)));
                self.tiles.insert(tc, t.clone());
                continue;
            }
            let t = self.tile_mut(tc);
            for y in tr.y0..tr.y1 {
                let ly = (y - tc.ty * TILE_SIZE) as usize;
                let lx0 = (tr.x0 - tc.tx * TILE_SIZE) as usize;
                let lx1 = (tr.x1 - tc.tx * TILE_SIZE) as usize;
                let row = &mut t.data[(ly * TILE_SIZE as usize + lx0) * bpp..(ly * TILE_SIZE as usize + lx1) * bpp];
                for p in row.chunks_exact_mut(bpp) {
                    p.copy_from_slice(&enc);
                }
            }
        }
    }

    /// Build from interleaved *encoded* bytes of `format` covering `r`.
    pub fn from_interleaved(format: PixelFormat, r: Rect, bytes: &[u8]) -> Self {
        let mut s = Surface::new(format);
        s.write_interleaved(r, bytes);
        s
    }

    /// Write interleaved encoded bytes (same format as the surface) into `r`.
    pub fn write_interleaved(&mut self, r: Rect, bytes: &[u8]) {
        let bpp = self.format.bytes_per_pixel();
        let w = r.width() as usize;
        assert_eq!(bytes.len(), w * r.height() as usize * bpp, "interleaved length mismatch");
        for tc in r.tiles() {
            let tr = tc.rect().intersect(&r);
            let t = self.tile_mut(tc);
            let span = tr.width() as usize * bpp;
            for y in tr.y0..tr.y1 {
                let src = (((y - r.y0) as usize) * w + (tr.x0 - r.x0) as usize) * bpp;
                let dst = (((y - tc.ty * TILE_SIZE) as usize) * TILE_SIZE as usize + (tr.x0 - tc.tx * TILE_SIZE) as usize) * bpp;
                t.data[dst..dst + span].copy_from_slice(&bytes[src..src + span]);
            }
        }
    }

    /// Read `r` as interleaved encoded bytes in the surface's format.
    pub fn to_interleaved(&self, r: Rect) -> Vec<u8> {
        let bpp = self.format.bytes_per_pixel();
        let w = r.width() as usize;
        let mut out = vec![0u8; w * r.height() as usize * bpp];
        for tc in r.tiles() {
            let tr = tc.rect().intersect(&r);
            let span = tr.width() as usize * bpp;
            for y in tr.y0..tr.y1 {
                let dst = (((y - r.y0) as usize) * w + (tr.x0 - r.x0) as usize) * bpp;
                match self.tiles.get(&tc) {
                    Some(t) => {
                        let src = (((y - tc.ty * TILE_SIZE) as usize) * TILE_SIZE as usize + (tr.x0 - tc.tx * TILE_SIZE) as usize) * bpp;
                        out[dst..dst + span].copy_from_slice(&t.data[src..src + span]);
                    }
                    None => {
                        for p in out[dst..dst + span].chunks_exact_mut(bpp) {
                            p.copy_from_slice(&self.default_pixel);
                        }
                    }
                }
            }
        }
        out
    }

    /// The surface moved by whole pixels: its encoded samples copied tile to tile (no float
    /// round trip, no intermediate buffer); moves by whole tiles share the tiles themselves, and
    /// a target tile read entirely from tiles of one single pixel value shares one such tile (the
    /// inside of a solid fill stays one copy-on-write tile however often it moves). A tile it
    /// copies that ends up holding only the default pixel isn't stored.
    /// `content` must hold every pixel that isn't the default one (`content_bounds`, say): only
    /// tiles it reaches are kept.
    pub fn translated(&self, dx: i32, dy: i32, content: Rect) -> Surface {
        let mut out = Surface { format: self.format, default_pixel: self.default_pixel.clone(), tiles: BTreeMap::new() };
        let sources = || self.tiles.iter().filter(move |(c, _)| !c.rect().intersect(&content).is_empty());
        if dx % TILE_SIZE == 0 && dy % TILE_SIZE == 0 {
            let (tx, ty) = (dx / TILE_SIZE, dy / TILE_SIZE);
            out.tiles = sources().map(|(c, t)| (TileCoord::new(c.tx.saturating_add(tx), c.ty.saturating_add(ty)), t.clone())).collect();
            return out;
        }
        let bpp = self.format.bytes_per_pixel();
        let moved = content.translate(dx, dy);
        let targets: std::collections::BTreeSet<TileCoord> = sources().flat_map(|(c, _)| c.rect().translate(dx, dy).intersect(&moved).tiles()).collect();
        let mut uniform = UniformTiles::default();
        let zero = vec![0u8; bpp];
        // One tile row of the default pixel, to compare a finished tile with a row at a time.
        let default_row = self.default_pixel.repeat(TILE_SIZE.unsigned_abs() as usize);
        for dc in targets {
            let dr = dc.rect();
            let sr = dr.translate(dx.saturating_neg(), dy.saturating_neg());
            // Every pixel of the target comes from a source tile (none from the default pixel).
            // `sr` must be the exact preimage: an offset of `i32::MIN` saturates its negation.
            let covered = sr.width() == TILE_SIZE.unsigned_abs()
                && sr.height() == TILE_SIZE.unsigned_abs()
                && sr.translate(dx, dy) == dr
                && content.contains_rect(&sr)
                && sr.tiles().all(|sc| self.tiles.contains_key(&sc));
            if covered && let Some(shared) = uniform.window(self, sr) {
                // A tile of only the default pixel is left out, as `prune` would.
                if shared.data.get(..bpp) != Some(&*self.default_pixel) {
                    out.tiles.insert(dc, shared);
                }
                continue;
            }
            // A covered target is overwritten whole, so skip filling it with the default pixel.
            let mut tile = Tile::filled(&self.format, if covered { &zero } else { &self.default_pixel });
            for sc in sr.intersect(&content).tiles() {
                let Some(st) = self.tiles.get(&sc) else { continue };
                let part = sc.rect().intersect(&sr);
                let span = part.width() as usize * bpp;
                // Offsets inside the tiles (the target's computed wide: an offset near the i32
                // limits saturates `translate`, and such rows are skipped rather than misplaced).
                let at = |x: i64, y: i64| usize::try_from(y * i64::from(TILE_SIZE) + x).ok().map(|i| i * bpp);
                for y in part.y0..part.y1 {
                    let src = at(i64::from(part.x0 - sc.tx * TILE_SIZE), i64::from(y - sc.ty * TILE_SIZE));
                    let dst = at(i64::from(part.x0) + i64::from(dx) - i64::from(dr.x0), i64::from(y) + i64::from(dy) - i64::from(dr.y0));
                    let (Some(src), Some(dst)) = (src, dst) else { continue };
                    if let (Some(d), Some(s)) = (tile.data.get_mut(dst..dst + span), st.data.get(src..src + span)) {
                        d.copy_from_slice(s);
                    }
                }
            }
            // A target that got only default pixels isn't stored, as `prune` would drop it. Checked
            // after the copy, a row at a time: a tile with content stops at its first such row.
            if tile.data.chunks(default_row.len()).any(|row| row != default_row) {
                out.tiles.insert(dc, Arc::new(tile));
            }
        }
        out
    }

    /// Applies `f` to every colour sample (not alpha), including the default pixel: a tone-curve
    /// change such as linearising when a document becomes 32-bit.
    pub fn map_color(&mut self, f: impl Fn(f32) -> f32) {
        let format = self.format;
        let color = format.mode.color_channels();
        let n = format.channels();
        let mut dp = decode_pixel(&format, &self.default_pixel);
        for v in dp.iter_mut().take(color) {
            *v = f(*v);
        }
        encode_pixel(&format, &dp, &mut self.default_pixel);
        let coords: Vec<TileCoord> = self.tiles.keys().copied().collect();
        let px_count = (TILE_SIZE * TILE_SIZE) as usize;
        // A tile shared by several coordinates (a filled background) is mapped once and stays
        // shared (#4531). Keys are the original tiles, all alive until their coordinates are
        // reached, so a freed address can't be mistaken for another original.
        let mut done: std::collections::HashMap<*const Tile, Arc<Tile>> = std::collections::HashMap::new();
        for c in coords {
            let Some(key) = self.tiles.get(&c).map(Arc::as_ptr) else { continue };
            if let Some(mapped) = done.get(&key) {
                self.tiles.insert(c, mapped.clone());
                continue;
            }
            let tile = self.tile_mut(c);
            for i in 0..px_count {
                for k in 0..color {
                    let at = i * n + k;
                    let v = read_sample(&tile.data, format.sample, at);
                    write_sample(&mut tile.data, format.sample, at, f(v));
                }
            }
            if let Some(mapped) = self.tiles.get(&c) {
                done.insert(key, mapped.clone());
            }
        }
    }

    /// Convert to another pixel format (depth and/or model; colour models via [`convert_pixel`]).
    pub fn convert(&self, to: PixelFormat) -> Surface {
        let mut out = Surface::with_default(to, &convert_pixel(&self.format, &to, &self.default_pixel()));
        let from_n = self.channels();
        let mut src = vec![0.0f32; from_n];
        // Each distinct tile converts once; coordinates sharing it share the result (#4531).
        let mut done: std::collections::HashMap<*const Tile, Arc<Tile>> = std::collections::HashMap::new();
        for (c, t) in &self.tiles {
            if let Some(converted) = done.get(&Arc::as_ptr(t)) {
                out.tiles.insert(*c, converted.clone());
                continue;
            }
            let dst = out.tile_mut(*c);
            let px_count = (TILE_SIZE * TILE_SIZE) as usize;
            for i in 0..px_count {
                for (k, v) in src.iter_mut().enumerate() {
                    *v = read_sample(&t.data, self.format.sample, i * from_n + k);
                }
                let px = convert_pixel(&self.format, &to, &src);
                for (k, v) in px.iter().enumerate() {
                    write_sample(&mut dst.data, to.sample, i * to.channels() + k, *v);
                }
            }
            if let Some(converted) = out.tiles.get(c) {
                done.insert(Arc::as_ptr(t), converted.clone());
            }
        }
        out
    }

    /// Convert a surface while reserving each replacement tile fallibly.
    pub fn try_convert(&self, to: PixelFormat) -> Result<Surface, AllocationError> {
        let mut out = Surface::with_default(to, &convert_pixel(&self.format, &to, &self.default_pixel()));
        let from_n = self.channels();
        let mut src = Vec::new();
        src.try_reserve_exact(from_n).map_err(|_| AllocationError::NotEnoughMemory { bytes: from_n * std::mem::size_of::<f32>() })?;
        src.resize(from_n, 0.0f32);
        let mut done: std::collections::HashMap<*const Tile, Arc<Tile>> = std::collections::HashMap::new();
        for (c, t) in &self.tiles {
            if let Some(converted) = done.get(&Arc::as_ptr(t)) {
                out.tiles.insert(*c, converted.clone());
                continue;
            }
            let dst = out.try_tile_mut(*c)?;
            let px_count = (TILE_SIZE * TILE_SIZE) as usize;
            for i in 0..px_count {
                for (k, v) in src.iter_mut().enumerate() {
                    *v = read_sample(&t.data, self.format.sample, i * from_n + k);
                }
                let px = convert_pixel(&self.format, &to, &src);
                for (k, v) in px.iter().enumerate() {
                    write_sample(&mut dst.data, to.sample, i * to.channels() + k, *v);
                }
            }
            if let Some(converted) = out.tiles.get(c) {
                done.insert(Arc::as_ptr(t), converted.clone());
            }
        }
        Ok(out)
    }

    /// Composite-friendly RGBA read (converts from the surface's model).
    pub fn rgba(&self, x: i32, y: i32) -> [f32; 4] {
        let p = self.pixel(x, y);
        to_rgba(&self.format, &p)
    }
}

/// Encode normalised floats into one pixel's bytes.
pub fn encode_pixel(format: &PixelFormat, px: &[f32], out: &mut [u8]) {
    for (i, v) in px.iter().enumerate().take(format.channels()) {
        write_sample(out, format.sample, i, *v);
    }
}

pub fn decode_pixel(format: &PixelFormat, bytes: &[u8]) -> Vec<f32> {
    (0..format.channels()).map(|i| read_sample(bytes, format.sample, i)).collect()
}

/// Convert one pixel between formats via sRGB RGBA (CMYK through the built-in CMYK profile;
/// the engine's mode conversions use the document's profiles instead).
pub fn convert_pixel(from: &PixelFormat, to: &PixelFormat, px: &[f32]) -> Vec<f32> {
    if from.mode == to.mode {
        let n = from.mode.color_channels();
        let mut out: Vec<f32> = px[..n].to_vec();
        if to.alpha {
            out.push(if from.alpha { px[n] } else { 1.0 });
        }
        return out;
    }
    let rgba = to_rgba(from, px);
    from_rgba(to, rgba)
}

pub fn to_rgba(format: &PixelFormat, px: &[f32]) -> [f32; 4] {
    let n = format.mode.color_channels();
    let a = if format.alpha { px[n] } else { 1.0 };
    let rgb = match format.mode {
        ColorMode::Grayscale | ColorMode::Bitmap | ColorMode::Duotone => [px[0]; 3],
        ColorMode::Cmyk => photocraft_color::convert::cmyk_to_rgb([px[0], px[1], px[2], px[3]]),
        ColorMode::Lab => photocraft_color::convert::lab_to_srgb([px[0] * 100.0, px[1] * 255.0 - 128.0, px[2] * 255.0 - 128.0]),
        _ => [px[0], px[1], px[2]],
    };
    [rgb[0], rgb[1], rgb[2], a]
}

pub fn from_rgba(format: &PixelFormat, rgba: [f32; 4]) -> Vec<f32> {
    let rgb = [rgba[0], rgba[1], rgba[2]];
    let mut out: Vec<f32> = match format.mode {
        ColorMode::Grayscale | ColorMode::Bitmap | ColorMode::Duotone => vec![photocraft_color::convert::rgb_to_gray(rgb)],
        ColorMode::Cmyk => photocraft_color::convert::rgb_to_cmyk(rgb).to_vec(),
        ColorMode::Lab => {
            let l = photocraft_color::convert::srgb_to_lab(rgb);
            vec![l[0] / 100.0, (l[1] + 128.0) / 255.0, (l[2] + 128.0) / 255.0]
        }
        _ => rgb.to_vec(),
    };
    if format.alpha {
        out.push(rgba[3]);
    }
    out
}

/// [`from_rgba`] without allocating: writes `format.channels()` values into
/// `out` and returns how many were written.
#[inline]
pub fn from_rgba_into(format: &PixelFormat, rgba: [f32; 4], out: &mut [f32]) -> usize {
    let rgb = [rgba[0], rgba[1], rgba[2]];
    let n = match format.mode {
        ColorMode::Grayscale | ColorMode::Bitmap | ColorMode::Duotone => {
            out[0] = photocraft_color::convert::rgb_to_gray(rgb);
            1
        }
        ColorMode::Cmyk => {
            out[..4].copy_from_slice(&photocraft_color::convert::rgb_to_cmyk(rgb));
            4
        }
        ColorMode::Lab => {
            let l = photocraft_color::convert::srgb_to_lab(rgb);
            out[..3].copy_from_slice(&[l[0] / 100.0, (l[1] + 128.0) / 255.0, (l[2] + 128.0) / 255.0]);
            3
        }
        _ => {
            out[..3].copy_from_slice(&rgb);
            3
        }
    };
    if format.alpha {
        out[n] = rgba[3];
        n + 1
    } else {
        n
    }
}

/// A simple owned RGBA8 image, used for display and thumbnails.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgba8Image {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl Rgba8Image {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height, pixels: vec![0; width as usize * height as usize * 4] }
    }
}

/// Is this sample type able to hold values above 1.0 (HDR)?
pub fn is_hdr(sample: SampleType) -> bool {
    matches!(sample, SampleType::F32)
}

#[cfg(test)]
mod tests {
    /// Distinct tile allocations in `s`.
    fn distinct_tiles(s: &Surface) -> usize {
        let mut seen: Vec<*const Tile> = s.tiles().map(|(_, t)| Arc::as_ptr(t)).collect();
        seen.sort();
        seen.dedup();
        seen.len()
    }

    /// Converting or tone-mapping a surface keeps a shared tile shared (#4531) and gives the
    /// same pixels as converting each pixel: a filled background is one tile across the grid,
    /// plus some distinct tiles.
    #[test]
    fn conversion_keeps_shared_tiles_shared() {
        let mut s = Surface::new(PixelFormat::RGBA8);
        s.fill_rect(Rect::new(0, 0, 2048, 1024), &[0.9, 0.5, 0.2, 1.0]);
        s.fill_rect(Rect::new(10, 10, 20, 20), &[0.1, 0.2, 0.3, 0.5]);
        let shared_before = distinct_tiles(&s);
        assert!(shared_before <= 2 && s.tile_count() == 32, "{shared_before} distinct of {}", s.tile_count());
        let r = Rect::new(0, 0, 2048, 1024);
        for to in [PixelFormat::RGBA16, PixelFormat::RGBA32F] {
            for out in [s.convert(to), s.try_convert(to).expect("memory")] {
                assert_eq!(distinct_tiles(&out), shared_before, "{to:?}: still shared");
                assert_eq!(out.tile_count(), 32);
                let (a, b) = (out.read_region(r), s.read_region(r));
                for (pa, pb) in a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0) {
                    let want = convert_pixel(&s.format(), &to, pb);
                    assert!(pa.iter().zip(&want).all(|(x, y)| (x - y).abs() < 1e-6), "{to:?}: {pa:?} vs {want:?}");
                }
            }
        }
        let mut m = s.clone();
        m.map_color(|v| v * 0.5);
        assert_eq!(distinct_tiles(&m), shared_before, "map_color keeps it shared");
        assert!((m.pixel(1000, 500)[0] - s.pixel(1000, 500)[0] * 0.5).abs() < 1.0 / 255.0);
        assert_eq!(s.pixel(1000, 500), vec![s.pixel(1000, 500)[0], s.pixel(1000, 500)[1], s.pixel(1000, 500)[2], 1.0], "the source is untouched");
    }

    use super::*;
    use photocraft_color::PixelFormat;
    use proptest::prelude::*;

    fn write_region_reference(surface: &mut Surface, r: Rect, data: &[f32]) {
        let n = surface.channels();
        let w = r.width() as usize;
        assert_eq!(data.len(), w * r.height() as usize * n, "region data length mismatch");
        let sample = surface.format.sample;
        for tc in r.tiles() {
            let tr = tc.rect().intersect(&r);
            let t = surface.tile_mut(tc);
            for y in tr.y0..tr.y1 {
                for x in tr.x0..tr.x1 {
                    let o = (((y - r.y0) as usize) * w + (x - r.x0) as usize) * n;
                    let base = (((y - tc.ty * TILE_SIZE) as usize) * TILE_SIZE as usize + (x - tc.tx * TILE_SIZE) as usize) * n;
                    for i in 0..n {
                        write_sample(&mut t.data, sample, base + i, data[o + i]);
                    }
                }
            }
        }
    }

    #[test]
    fn region_writes_match_scalar_bytes_for_every_format() {
        let special = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            2.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::from_bits(0x7fc12345),
            f32::from_bits(0xffc12345),
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            0.5 / 255.0,
            0.5 / 65535.0,
        ];
        for mode in [
            ColorMode::Bitmap,
            ColorMode::Grayscale,
            ColorMode::Indexed,
            ColorMode::Rgb,
            ColorMode::Cmyk,
            ColorMode::Lab,
            ColorMode::Multichannel,
            ColorMode::Duotone,
        ] {
            for sample in SampleType::ALL {
                for alpha in [false, true] {
                    let fmt = PixelFormat::new(mode, sample, alpha);
                    let mut actual = Surface::with_default(fmt, &vec![0.25; fmt.channels()]);
                    actual.write_pixel(900, 900, &vec![0.75; fmt.channels()]);
                    let mut expected = actual.clone();
                    for rect in
                        [Rect::EMPTY, Rect::new(-270, -2, 270, 3), Rect::new(256, 256, 512, 257), Rect::new(-64, -64, 0, 0), Rect::new(255, 250, 258, 270)]
                    {
                        let data: Vec<_> = (0..rect.width() as usize * rect.height() as usize * fmt.channels())
                            .map(|i| {
                                if i % 3 == 0 {
                                    special[i / 3 % special.len()]
                                } else {
                                    // Integer rounding boundaries and deterministic arbitrary f32 bit patterns.
                                    let bits = (i as u32).wrapping_mul(2654435761);
                                    if i % 3 == 1 { ((bits % 65536) as f32 + 0.5) / 65535.0 } else { f32::from_bits(bits) }
                                }
                            })
                            .collect();
                        let snapshot = actual.clone();
                        let snapshot_bytes = snapshot.to_interleaved(rect.inflate(1));
                        actual.write_region(rect, &data);
                        write_region_reference(&mut expected, rect, &data);
                        assert_eq!(actual, expected, "{fmt:?}, {rect:?}");
                        assert_eq!(snapshot.to_interleaved(rect.inflate(1)), snapshot_bytes, "COW snapshot changed");
                        // Repeat against already materialised, unshared tiles too.
                        actual.write_region(rect, &data);
                        assert_eq!(actual, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn from_rgba_into_matches_from_rgba() {
        for fmt in [PixelFormat::RGBA8, PixelFormat::GRAY8, PixelFormat::GRAYA8, PixelFormat::CMYKA8, PixelFormat::new(ColorMode::Lab, SampleType::F32, true)] {
            let mut buf = [0.0f32; 8];
            let n = from_rgba_into(&fmt, [0.2, 0.5, 0.7, 0.4], &mut buf);
            assert_eq!(&buf[..n], &from_rgba(&fmt, [0.2, 0.5, 0.7, 0.4])[..]);
        }
    }

    #[test]
    fn zero_alloc_accessors_match_read_region() {
        for fmt in [PixelFormat::RGBA8, PixelFormat::RGBA16, PixelFormat::RGBA32F, PixelFormat::GRAYA8, PixelFormat::CMYKA8, PixelFormat::GRAY8] {
            let mut s = Surface::with_default(fmt, &vec![0.25; fmt.channels()]);
            let r = Rect::new(-300, -20, 300, 40);
            for (i, (x, y)) in [(-299, -19), (0, 0), (255, 39), (256, 10), (299, 0)].into_iter().enumerate() {
                let px: Vec<f32> = (0..fmt.channels()).map(|c| ((i * 3 + c) % 5) as f32 / 4.0).collect();
                s.write_pixel(x, y, &px);
            }
            let region = s.read_region(r);
            let mut into = vec![9.0; 3];
            s.read_region_into(r, &mut into);
            assert_eq!(region, into);
            let mut rgba = vec![[0.0; 4]; (r.width() * r.height()) as usize];
            s.read_rgba_into(r, &mut rgba);
            let n = fmt.channels();
            for (i, p) in region.chunks_exact(n).enumerate() {
                let want = to_rgba(&fmt, p);
                for c in 0..4 {
                    assert!((rgba[i][c] - want[c]).abs() < 1e-6, "{fmt:?} px {i}");
                }
                let (x, y) = (r.x0 + (i % r.width() as usize) as i32, r.y0 + (i / r.width() as usize) as i32);
                assert_eq!(s.sample_channel(x, y, 0), p[0]);
            }
            assert!(s.has_tiles_in(Rect::new(0, 0, 1, 1)));
            assert!(!s.has_tiles_in(Rect::new(1000, 1000, 1001, 1001)));
        }
    }

    #[test]
    fn fallible_region_reads_report_size_overflow() {
        let surface = Surface::new(PixelFormat::RGBA8);
        let huge = Rect::new(i32::MIN, i32::MIN, i32::MAX, i32::MAX);
        assert_eq!(surface.try_read_region(huge), Err(AllocationError::SizeOverflow));
        assert_eq!(surface.try_to_interleaved(huge), Err(AllocationError::SizeOverflow));
    }

    #[test]
    fn fallible_region_write_rejects_bad_input_without_mutating_surface() {
        let mut surface = Surface::new(PixelFormat::RGBA8);
        let before = surface.clone();
        assert_eq!(surface.try_write_region(Rect::new(0, 0, 2, 2), &[1.0]), Err(AllocationError::SizeOverflow));
        assert_eq!(surface, before);
    }

    #[test]
    fn fill_rect_shares_covered_tiles_and_edits_stay_local() {
        // A 300000² background used to materialize ~360 GB of tiles (#705).
        let mut s = Surface::new(PixelFormat::RGBA8);
        s.fill_rect(Rect::new(0, 0, 300_000, 300_000), &[1.0, 0.5, 0.0, 1.0]);
        let t = |s: &Surface, x, y| s.tile(TileCoord::new(x, y)).unwrap().clone();
        assert!(Arc::ptr_eq(&t(&s, 0, 0), &t(&s, 1000, 900)));
        // 300000 is not a multiple of 256: the edge tile is partial and stays its own.
        let edge = 300_000 / TILE_SIZE;
        assert!(!Arc::ptr_eq(&t(&s, 0, 0), &t(&s, edge, 0)));
        assert_eq!(s.pixel(299_999, 299_999), vec![1.0, 128.0 / 255.0, 0.0, 1.0]);
        assert_eq!(s.pixel(300_000, 0), vec![0.0; 4]);
        s.write_pixel(5, 5, &[0.0, 0.0, 1.0, 1.0]);
        assert_eq!(s.pixel(5, 5), vec![0.0, 0.0, 1.0, 1.0]);
        assert_eq!(s.pixel(300, 5), vec![1.0, 128.0 / 255.0, 0.0, 1.0]);
        assert!(!Arc::ptr_eq(&t(&s, 0, 0), &t(&s, 1, 0)));
    }

    #[test]
    fn empty_surface_reads_default() {
        let s = Surface::new(PixelFormat::RGBA8);
        assert_eq!(s.pixel(10, -500), vec![0.0; 4]);
        let m = Surface::with_default(PixelFormat::GRAY8, &[1.0]);
        assert_eq!(m.pixel(123_456, -9), vec![1.0]);
        assert_eq!(m.tile_count(), 0);
    }

    #[test]
    fn write_read_pixel_every_depth() {
        for sample in SampleType::ALL {
            let f = PixelFormat::RGBA8.with_sample(sample);
            let mut s = Surface::new(f);
            s.write_pixel(-3, 700, &[0.2, 0.4, 0.6, 1.0]);
            let p = s.pixel(-3, 700);
            for (a, b) in p.iter().zip([0.2, 0.4, 0.6, 1.0]) {
                assert!((a - b).abs() <= 1.0 / 255.0, "{sample:?}");
            }
            assert_eq!(s.tile_count(), 1);
            assert_eq!(s.tile_bounds(), TileCoord::containing(-3, 700).rect());
        }
    }

    #[test]
    fn copy_on_write_shares_untouched_tiles() {
        let mut a = Surface::new(PixelFormat::RGBA8);
        a.fill_rect(Rect::new(0, 0, 512, 512), &[1.0, 0.0, 0.0, 1.0]);
        let mut b = a.clone();
        b.write_pixel(10, 10, &[0.0, 1.0, 0.0, 1.0]);
        // a unchanged
        assert_eq!(a.pixel(10, 10), vec![1.0, 0.0, 0.0, 1.0]);
        assert_eq!(b.pixel(10, 10), vec![0.0, 1.0, 0.0, 1.0]);
        // only tile (0,0) was copied
        let shared = a.tiles().filter(|(c, t)| Arc::ptr_eq(t, b.tile(**c).unwrap())).count();
        assert_eq!(shared, 3);
    }

    #[test]
    fn decoded_revisions_reuse_only_identical_tiles_at_every_depth() {
        for sample in SampleType::ALL {
            let format = PixelFormat::RGBA8.with_sample(sample);
            let bounds = Rect::new(0, 0, 512, 512);
            let mut before = Surface::new(format);
            before.fill_rect(bounds, &[1.0, 0.0, 0.0, 1.0]);
            let mut decoded = Surface::from_interleaved(format, bounds, &before.to_interleaved(bounds));
            decoded.write_pixel(10, 10, &[0.0, 1.0, 0.0, 1.0]);
            decoded.reuse_tiles_from(&before);
            assert_eq!(decoded.tiles().filter(|(c, t)| Arc::ptr_eq(t, before.tile(**c).unwrap())).count(), 3);
            assert_eq!(decoded.pixel(10, 10), vec![0.0, 1.0, 0.0, 1.0]);
            decoded.write_pixel(300, 300, &[0.0, 0.0, 1.0, 1.0]);
            assert_eq!(before.pixel(300, 300), vec![1.0, 0.0, 0.0, 1.0]);
            let mut other_format =
                Surface::from_interleaved(format.with_sample(SampleType::U8), bounds, &Surface::new(format.with_sample(SampleType::U8)).to_interleaved(bounds));
            if sample != SampleType::U8 {
                other_format.reuse_tiles_from(&before);
                assert!(other_format.tiles().all(|(c, t)| !Arc::ptr_eq(t, before.tile(*c).unwrap())));
            }
        }
    }

    #[test]
    fn content_bounds_and_prune() {
        let mut s = Surface::new(PixelFormat::RGBA8);
        s.fill_rect(Rect::new(300, 10, 310, 20), &[1.0, 1.0, 1.0, 1.0]);
        s.write_pixel(-5, -5, &[0.0, 0.0, 0.0, 1.0]);
        assert_eq!(s.content_bounds(), Rect::new(-5, -5, 310, 20));
        // erase the single pixel -> its tile becomes default and is pruned
        s.write_pixel(-5, -5, &[0.0, 0.0, 0.0, 0.0]);
        s.prune();
        assert_eq!(s.tile_count(), 1);
        assert_eq!(s.content_bounds(), Rect::new(300, 10, 310, 20));
    }

    #[test]
    fn interleaved_roundtrip_crossing_tiles() {
        let r = Rect::new(-20, 250, 280, 262);
        let bpp = PixelFormat::RGBA16.bytes_per_pixel();
        let bytes: Vec<u8> = (0..r.width() as usize * r.height() as usize * bpp).map(|i| (i * 31 % 251) as u8).collect();
        let s = Surface::from_interleaved(PixelFormat::RGBA16, r, &bytes);
        assert_eq!(s.to_interleaved(r), bytes);
        assert_eq!(s.tile_count(), 6); // x tiles -1..=1, y tiles 0..=1
    }

    #[test]
    fn region_outside_reads_default() {
        let s = Surface::with_default(PixelFormat::GRAY8, &[1.0]);
        let v = s.read_region(Rect::new(0, 0, 3, 3));
        assert_eq!(v, vec![1.0; 9]);
    }

    #[test]
    fn convert_depths_and_models() {
        let mut s = Surface::new(PixelFormat::RGBA8);
        s.write_pixel(1, 1, &[1.0, 1.0, 1.0, 1.0]);
        let g = s.convert(PixelFormat::GRAYA8);
        assert!((g.pixel(1, 1)[0] - 1.0).abs() < 1e-2);
        let f = s.convert(PixelFormat::RGBA32F);
        assert_eq!(f.pixel(1, 1), vec![1.0, 1.0, 1.0, 1.0]);
        let c = s.convert(PixelFormat::CMYKA8);
        let p = c.pixel(1, 1);
        assert!(p[3] < 0.01 && (p[4] - 1.0).abs() < 1e-6);
        let back = c.convert(PixelFormat::RGBA8);
        assert_eq!(back.pixel(1, 1), vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn rgba_from_gray_and_cmyk() {
        let mut g = Surface::new(PixelFormat::GRAYA8);
        g.write_pixel(0, 0, &[0.5, 1.0]);
        let v = g.rgba(0, 0);
        assert!((v[0] - v[2]).abs() < 1e-6 && (v[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn repeated_cmyk_reads_match_individual_pixels_bitwise() {
        let rect = Rect::new(-2, -1, 515, 2);
        for depth in SampleType::ALL {
            for alpha in [false, true] {
                let fmt = PixelFormat::new(ColorMode::Cmyk, depth, alpha);
                let mut s = Surface::with_default(fmt, &[0.2, 0.4, 0.6, 0.3, 0.7]);
                for (i, x) in (-1..514).enumerate() {
                    let c = if i / 2 % 256 == 0 { -0.0 } else { (i / 2 % 256) as f32 / 255.0 };
                    s.write_pixel(x, 0, &[c, 0.4, 0.6, 0.3, (i % 9) as f32 / 8.0]);
                }
                let mut out = vec![[0.0; 4]; (rect.width() * rect.height()) as usize];
                s.read_rgba_into(rect, &mut out);
                for (i, actual) in out.iter().enumerate() {
                    let expected = s.rgba(rect.x0 + (i % rect.width() as usize) as i32, rect.y0 + (i / rect.width() as usize) as i32);
                    assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits), "{depth:?}, alpha={alpha}, pixel={i}");
                }
            }
        }
    }

    #[test]
    fn tile_content_bounds_scans_against_the_given_pixel() {
        let mut s = Surface::new(PixelFormat::GRAY8);
        s.write_pixel(3, 4, &[1.0]);
        s.write_pixel(200, 9, &[1.0]);
        let t = s.tile(TileCoord::new(0, 0)).expect("written tile");
        assert_eq!(t.content_bounds(&[0]), Some((3, 4, 201, 10)));
        assert_eq!(t.content_bounds(&[255]).map(|b| (b.0, b.1)), Some((0, 0)));
        // A malformed pixel is no content, not a panic.
        assert_eq!(t.content_bounds(&[]), None);
    }

    #[test]
    fn translated_matches_reading_the_moved_pixels() {
        // Any offset (whole tiles, negative, across tile edges), every depth, a non-zero default.
        for fmt in [PixelFormat::RGBA8, PixelFormat::RGBA16, PixelFormat::RGBA32F, PixelFormat::GRAY8] {
            let mut s = Surface::with_default(fmt, &vec![0.25; fmt.channels()]);
            let r = Rect::new(-37, 200, 530, 300);
            let v: Vec<f32> = (0..r.width() * r.height() * fmt.channels() as u32).map(|i| (i % 251) as f32 / 250.0).collect();
            s.write_region(r, &v);
            for (dx, dy) in [(0, 0), (256, -512), (1, 0), (-3, 7), (300, -257), (-1000, 999)] {
                let t = s.translated(dx, dy, s.content_bounds());
                let moved = r.translate(dx, dy).inflate(3);
                assert_eq!(t.read_region(moved), s.read_region(moved.translate(-dx, -dy)), "{fmt:?} by ({dx}, {dy})");
                assert_eq!(t.default_pixel(), s.default_pixel());
                assert_eq!(t.to_interleaved(r.translate(dx, dy)), s.to_interleaved(r), "{fmt:?} by ({dx}, {dy}): not an exact copy");
            }
            // Whole tiles: the tiles are shared, not copied.
            let t = s.translated(512, -256, s.content_bounds());
            assert!(s.tiles().zip(t.tiles()).all(|((_, a), (_, b))| Arc::ptr_eq(a, b)));
            // Tiles without content (allocated, default pixels) are left behind.
            let mut padded = s.clone();
            padded.fill_rect(Rect::new(2000, 2000, 2100, 2100), &s.default_pixel());
            assert_eq!(padded.translated(5, 5, s.content_bounds()).tile_count(), s.translated(5, 5, s.content_bounds()).tile_count());
        }
        // A solid fill moved off the tile grid keeps sharing its one tile inside, at every depth
        // and over a non-zero default pixel, and reads exactly like the copied pixels.
        for fmt in [PixelFormat::RGBA8, PixelFormat::RGBA16, PixelFormat::RGBA32F, PixelFormat::GRAY8] {
            let mut s = Surface::with_default(fmt, &vec![0.25; fmt.channels()]);
            let r = Rect::new(-300, -10, 1300, 1030);
            s.fill_rect(r, &vec![0.75; fmt.channels()]);
            let fill = s.tile(TileCoord::new(1, 1)).cloned().expect("an inner fill tile");
            for (dx, dy) in [(10, 7), (-1, 0), (0, 300), (-257, 513)] {
                let t = s.translated(dx, dy, s.content_bounds());
                let moved = r.translate(dx, dy);
                assert_eq!(t.read_region(moved.inflate(3)), s.read_region(moved.inflate(3).translate(-dx, -dy)), "{fmt:?} by ({dx}, {dy})");
                // Targets read only from the fill's whole tiles (x -256..1280, y 0..1024).
                let whole = Rect::new(-256, 0, 1280, 1024).translate(dx, dy);
                let inner: Vec<_> = t.tiles().filter(|(c, _)| whole.contains_rect(&c.rect())).collect();
                assert!(!inner.is_empty() && inner.iter().all(|(_, a)| Arc::ptr_eq(a, &fill)), "{fmt:?} by ({dx}, {dy}): inner tiles copied");
            }
            // Nudged back and forth, the inside keeps sharing one tile (copies of uniform edge
            // tiles count as that value too) and the pixels stay exact.
            let mut t = s.clone();
            for k in 0..8 {
                let d = if k % 2 == 0 { 10 } else { -10 };
                t = t.translated(d, d / 2, t.content_bounds());
            }
            // (Targets whose last window read uniform tiles only: one tile in from the fill's whole
            // tiles on the sides the last move came from.)
            let inner: Vec<_> = t.tiles().filter(|(c, _)| Rect::new(-256, 0, 1024, 768).contains_rect(&c.rect())).map(|(_, a)| a).collect();
            assert!(inner.len() >= 4 && inner.iter().all(|a| Arc::ptr_eq(a, inner[0])), "{fmt:?}: repeated moves copied the inside");
            assert_eq!(t.read_region(r.inflate(3)), s.read_region(r.inflate(3)), "{fmt:?}: back where it started");
        }
        // Two solid fills of different values side by side: windows across the seam are
        // copied (each tile is uniform, but not with the same value), the rest shared.
        for fmt in [PixelFormat::RGBA8, PixelFormat::RGBA32F] {
            let mut s = Surface::new(fmt);
            s.fill_rect(Rect::new(0, 0, 1024, 768), &vec![0.2; fmt.channels()]);
            s.fill_rect(Rect::new(1024, 0, 2048, 768), &vec![0.9; fmt.channels()]);
            let t = s.translated(10, 7, s.content_bounds());
            let all = Rect::new(-20, -20, 2100, 800);
            assert_eq!(t.read_region(all), s.read_region(all.translate(-10, -7)), "{fmt:?}: two fills");
            let seam = t.tile(TileCoord::new(4, 1)).expect("seam tile");
            let (left, right) = (t.tile(TileCoord::new(2, 1)).expect("left"), t.tile(TileCoord::new(6, 1)).expect("right"));
            assert!(!Arc::ptr_eq(seam, left) && !Arc::ptr_eq(seam, right) && !Arc::ptr_eq(left, right), "{fmt:?}: seam aliased");
        }
        // An offset of i32::MIN saturates its negation, so the source window isn't the exact
        // preimage: no shortcut may apply (a zero pre-fill would leak into the output).
        let mut s = Surface::with_default(PixelFormat::GRAY8, &[0.5]);
        s.fill_rect(Rect::new(i32::MAX - 1024, 0, i32::MAX - 512, 512), &[0.25]);
        let (fill, default) = (s.sample_channel(i32::MAX - 600, 10, 0), s.default_pixel()[0]);
        for (dx, dy) in [(i32::MIN, 1), (1, i32::MIN), (i32::MIN, 0)] {
            let t = s.translated(dx, dy, s.content_bounds());
            for (c, _) in t.tiles() {
                assert!(t.read_region(c.rect()).iter().all(|v| *v == fill || *v == default), "({dx}, {dy}): tile {c:?} holds a pre-fill");
            }
        }
        // One tile shared but not uniform: each target is copied, not aliased.
        let mut s = Surface::new(PixelFormat::GRAY8);
        s.write_pixel(3, 4, &[1.0]);
        let odd = s.tile(TileCoord::new(0, 0)).cloned().expect("written tile");
        let grid: Vec<_> = (0..9).map(|i| (TileCoord::new(i % 3, i / 3), odd.clone())).collect();
        s.put_tiles(grid);
        let t = s.translated(10, 7, s.content_bounds());
        let all = Rect::new(-20, -20, 800, 800);
        assert_eq!(t.read_region(all), s.read_region(all.translate(-10, -7)));
        assert!(t.tiles().all(|(_, a)| !Arc::ptr_eq(a, &odd)));
        // Offsets at the limits don't panic.
        let mut s = Surface::new(PixelFormat::GRAY8);
        s.write_pixel(5, 5, &[1.0]);
        let all = Rect::new(i32::MIN, i32::MIN, i32::MAX, i32::MAX);
        let _ = s.translated(i32::MAX, i32::MIN, all);
        let _ = s.translated(i32::MIN + 1, 3, all);
    }

    #[test]
    fn translated_stores_no_tiles_of_only_the_default_pixel() {
        let only_default = |t: &Surface| {
            let dp = t.default_pixel();
            t.tiles().filter(|(c, _)| t.read_region(c.rect()).chunks_exact(dp.len()).all(|p| p == dp.as_slice())).count()
        };
        // Two blocks a few tiles apart, moved off the tile grid and back: the default-pixel edges
        // of their tiles land on tiles of their own, which aren't stored.
        for fmt in [PixelFormat::RGBA8, PixelFormat::RGBA16, PixelFormat::RGBA32F, PixelFormat::GRAY8] {
            for default in [0.0, 0.25] {
                let mut s = Surface::with_default(fmt, &vec![default; fmt.channels()]);
                s.fill_rect(Rect::new(0, 0, 300, 300), &vec![1.0; fmt.channels()]);
                s.fill_rect(Rect::new(2000, 1000, 2300, 1300), &vec![0.5; fmt.channels()]);
                let all = Rect::new(-600, -600, 3000, 2000);
                let mut t = s.clone();
                for (dx, dy) in [(100, 56), (-100, -56), (10, 7), (-3, 40), (130, -90), (-137, 43)] {
                    let moved = t.translated(dx, dy, t.content_bounds());
                    assert_eq!(moved.read_region(all), t.read_region(all.translate(-dx, -dy)), "{fmt:?} over {default} by ({dx}, {dy})");
                    assert_eq!(only_default(&moved), 0, "{fmt:?} over {default} by ({dx}, {dy})");
                    t = moved;
                }
                assert_eq!(t.read_region(all), s.read_region(all), "{fmt:?} over {default}: back where it started");
                assert_eq!(t.tile_count(), s.tile_count(), "{fmt:?} over {default}: back where it started");
            }
        }
        // Stored tiles of the default pixel inside `content`: the targets that would share one of
        // them (#4414's uniform windows) are left out too.
        let mut s = Surface::with_default(PixelFormat::GRAY8, &[0.25]);
        s.fill_rect(Rect::new(0, 0, 1024, 1024), &[0.25]);
        s.write_pixel(600, 600, &[1.0]);
        let content = Rect::new(0, 0, 1024, 1024);
        let t = s.translated(10, 7, content);
        assert_eq!(only_default(&t), 0);
        assert_eq!(t.tile_count(), 1);
        assert_eq!(t.read_region(Rect::new(-20, -20, 1100, 1100)), s.read_region(Rect::new(-30, -27, 1090, 1093)));
    }

    proptest! {
        #[test]
        fn region_write_read_roundtrip(x in -600i32..600, y in -600i32..600, w in 1u32..90, h in 1u32..90, seed in any::<u32>()) {
            let r = Rect::from_xywh(x, y, w, h);
            let n = 4usize;
            let data: Vec<f32> = (0..(w * h) as usize * n).map(|i| ((i as u32).wrapping_mul(2654435761).wrapping_add(seed) % 256) as f32 / 255.0).collect();
            let mut s = Surface::new(PixelFormat::RGBA8);
            s.write_region(r, &data);
            let back = s.read_region(r);
            for (a, b) in back.iter().zip(&data) {
                prop_assert!((a - b).abs() < 1e-6);
            }
            prop_assert_eq!(s.content_bounds().union(&r), r);
        }

        #[test]
        fn clone_mutation_isolated(x in -300i32..300, y in -300i32..300) {
            let mut a = Surface::new(PixelFormat::RGBA8);
            a.fill_rect(Rect::new(-256, -256, 256, 256), &[0.5, 0.5, 0.5, 1.0]);
            let snapshot = a.clone();
            a.write_pixel(x, y, &[1.0, 0.0, 0.0, 1.0]);
            let expect = if Rect::new(-256, -256, 256, 256).contains(x, y) { vec![0.5f32, 0.5, 0.5, 1.0] } else { vec![0.0; 4] };
            let got = snapshot.pixel(x, y);
            for (g, e) in got.iter().zip(&expect) {
                prop_assert!((g - e).abs() <= 1.0 / 255.0);
            }
        }
    }

    #[test]
    fn strided_row_sampling_matches_sample_channel() {
        let mut s = Surface::with_default(PixelFormat::GRAY8, &[0.25]);
        s.fill_rect(Rect::new(-300, 10, 200, 40), &[1.0]);
        s.fill_rect(Rect::new(250, 10, 700, 20), &[0.5]);
        for (y, x0, step) in [(15, -520, 7), (15, -1, 1), (30, 3, 64), (500, 0, 9)] {
            let mut out = vec![0.0; 120];
            s.sample_row_strided(y, x0, step, 0, &mut out);
            for (i, v) in out.iter().enumerate() {
                assert_eq!(*v, s.sample_channel(x0 + i as i32 * step, y, 0), "y={y} x0={x0} step={step} i={i}");
            }
        }
    }

    #[test]
    fn content_bounds_with_tile_skipping() {
        // Interior content only; ring tiles present but all-default (unpruned).
        let mut s = Surface::new(PixelFormat::GRAY8);
        s.fill_rect(Rect::new(0, 0, 1024, 1024), &[0.0]);
        s.fill_rect(Rect::new(300, 310, 700, 720), &[1.0]);
        assert_eq!(s.content_bounds(), Rect::new(300, 310, 700, 720));
        // Full-canvas content plus a far-away speck.
        let mut s = Surface::new(PixelFormat::GRAY8);
        s.fill_rect(Rect::new(0, 0, 2000, 1500), &[1.0]);
        s.fill_rect(Rect::new(-700, 4000, -699, 4001), &[0.5]);
        assert_eq!(s.content_bounds(), Rect::new(-700, 0, 2000, 4001));
    }
}
