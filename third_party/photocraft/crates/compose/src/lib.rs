//! CPU reference compositor.
//!
//! Flattens a [`Document`] layer tree into straight-alpha RGBA (f32) for any rectangle. It encodes
//! Photoshop layer semantics in one place:
//! - blend modes, opacity × fill opacity, visibility
//! - layer masks with density
//! - clipping groups (clipped layers composite *atop* their base)
//! - pass-through vs isolated groups
//! - adjustment layers (applied to the composite below)
//! - fill layers, Dissolve
//!
//! This is the reference the GPU backend (milestone M5) must match within 1/255. Compositing
//! currently happens in display RGB. CMYK layers are read through the document's CMYK profile
//! (the built-in coated CMYK when untagged) and composited in sRGB; mode-native (CMYK/Lab)
//! compositing arrives with ICC in M8.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod adjust;
pub mod advanced;
pub mod bounds;
pub mod effects;
pub mod fill_layout;
pub mod gradient_fill;
pub mod masks;
pub mod multichannel;
pub mod native;
pub mod pattern;
pub mod proxy;
pub mod psblend;
pub mod shape_split;

use photocraft_color::blend::BlendMode;
use photocraft_doc::{Document, Fill, Layer, LayerContent, Pattern};
use photocraft_geom::Rect;
use photocraft_raster::{Rgba8Image, Surface};
use psblend as blend;

pub use advanced::advanced_active;

/// Straight-alpha RGBA float buffer covering a rectangle.
#[derive(Clone, Debug, PartialEq)]
pub struct Buffer {
    pub rect: Rect,
    pub px: Vec<[f32; 4]>,
}

impl Buffer {
    pub fn transparent(rect: Rect) -> Self {
        Self { rect, px: vec![[0.0; 4]; rect.width() as usize * rect.height() as usize] }
    }
    pub fn filled(rect: Rect, c: [f32; 4]) -> Self {
        Self { rect, px: vec![c; rect.width() as usize * rect.height() as usize] }
    }
    #[inline]
    pub fn get(&self, x: i32, y: i32) -> [f32; 4] {
        self.px[((y - self.rect.y0) as usize) * self.rect.width() as usize + (x - self.rect.x0) as usize]
    }
    pub fn to_rgba8(&self) -> Rgba8Image {
        let mut img = Rgba8Image::new(self.rect.width(), self.rect.height());
        let quantize = |out: &mut [[u8; 4]], src: &[[f32; 4]]| {
            for (o, p) in out.iter_mut().zip(src) {
                for (dst, v) in o.iter_mut().zip(p) {
                    *dst = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                }
            }
        };
        let out = img.pixels.as_chunks_mut::<4>().0;
        // In parallel off wasm: a 24 MP composite took 120 ms on one core (the CPU canvas,
        // previews, thumbnails and 8-bit exports all come through here).
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            out.par_chunks_mut(16 * 1024).zip(self.px.par_chunks(16 * 1024)).for_each(|(o, p)| quantize(o, p));
        }
        #[cfg(target_arch = "wasm32")]
        quantize(out, &self.px);
        img
    }
    /// Flatten over an opaque background colour.
    pub fn over_background(&self, bg: [f32; 3]) -> Buffer {
        let mut out = self.clone();
        for p in &mut out.px {
            let a = p[3];
            for i in 0..3 {
                p[i] = p[i] * a + bg[i] * (1.0 - a);
            }
            p[3] = 1.0;
        }
        out
    }
}

/// Tile size for parallel rendering (tile results are independent).
pub const RENDER_TILE: i32 = 256;

/// Composite the whole document over `rect`, in parallel 256² tiles on
/// native targets (single-threaded on wasm).
pub fn render(doc: &Document, rect: Rect) -> Buffer {
    render_tiled(doc, rect, RENDER_TILE)
}

/// Most pixels one [`render`] call will produce (2^30, a 16 GiB buffer; 32768² px). Larger rects
/// (a hostile canvas size is up to 300000 px a side) return an empty buffer instead of aborting
/// on allocation; callers wanting more render in pieces.
pub const MAX_RENDER_PIXELS: u64 = 1 << 30;

/// [`render`] with an explicit tile size (tests check tile independence).
pub fn render_tiled(doc: &Document, rect: Rect, tile: i32) -> Buffer {
    let patterns = pattern::PreparedPatterns::new(&doc.patterns, pattern::PREPARED_PATTERN_BYTES).in_mode(doc.pixel_format().mode);
    let cx = Ctx::for_doc(doc, &patterns);
    render_tiled_with(doc, rect, tile, &cx)
}

fn render_tiled_with(doc: &Document, rect: Rect, tile: i32, cx: &Ctx) -> Buffer {
    let tile = tile.max(1);
    // Lab documents mix Normal blending in CIELAB, as Photoshop does (psblend::LAB_MIX); 32-bit
    // documents don't clip Add / Divide or Hue / Saturation / Color / Luminosity at 1 (psblend::HDR).
    let lab = doc.mode == photocraft_color::ColorMode::Lab;
    let hdr = doc.depth == photocraft_color::SampleType::F32;
    // CMYK layers are read through the document's own CMYK profile (thread-local scope).
    let cmyk = cmyk_space(doc);
    let cmyk = cmyk.as_ref();
    if rect.is_empty() {
        return Buffer::transparent(rect);
    }
    // Refuse rects too large to hold: an empty buffer, not an allocation failure.
    if u64::from(rect.width()) * u64::from(rect.height()) > MAX_RENDER_PIXELS {
        return Buffer::transparent(Rect::EMPTY);
    }
    cx.set_fx_clip(rect);
    if i64::from(rect.width()) <= i64::from(tile) && i64::from(rect.height()) <= i64::from(tile) {
        return photocraft_color::convert::with_cmyk_space(cmyk, || {
            let mut buf = multichannel::backdrop(doc, rect);
            in_blend_space(lab, hdr, || composite_stack_at(&doc.layers, &mut buf, cx, advanced::Scope::ROOT));
            buf
        });
    }
    // Effect maps are built once, here, before any tile needs them (#276).
    prepare_effects(&doc.layers, rect, cx, |f| {
        photocraft_color::convert::with_cmyk_space(cmyk, || in_blend_space(lab, hdr, f));
    });
    let run = |t: Rect| {
        photocraft_color::convert::with_cmyk_space(cmyk, || {
            let mut b = multichannel::backdrop(doc, t);
            in_blend_space(lab, hdr, || composite_stack_at(&doc.layers, &mut b, cx, advanced::Scope::ROOT));
            b
        })
    };
    // Tiles are written straight into the output, one row of tiles (a band) at a time, so the
    // peak is the output plus the tiles in flight, not a second full-size copy.
    let w = rect.width() as usize;
    let mut out = Buffer::transparent(rect);
    let band_tiles = |y0: i32| {
        let y1 = y0.saturating_add(tile).min(rect.y1);
        let mut v = Vec::new();
        let mut x = rect.x0;
        while x < rect.x1 {
            let x1 = x.saturating_add(tile).min(rect.x1);
            v.push(Rect::new(x, y0, x1, y1));
            x = x1;
        }
        v
    };
    let put = |band: &mut [[f32; 4]], y0: i32, part: &Buffer| {
        let pw = part.rect.width() as usize;
        for (row, src) in part.px.chunks_exact(pw).enumerate() {
            let o = ((part.rect.y0 - y0) as usize + row) * w + (part.rect.x0 - rect.x0) as usize;
            band[o..o + pw].copy_from_slice(src);
        }
    };
    let band_len = w * tile as usize;
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        let bands = rect.height().div_ceil(tile as u32) as usize;
        if bands >= 2 * rayon::current_num_threads() {
            // Enough bands to keep every core busy: one band per task, its tiles in turn.
            out.px.par_chunks_mut(band_len).enumerate().for_each(|(i, band)| {
                let y0 = rect.y0 + i as i32 * tile;
                for t in band_tiles(y0) {
                    put(band, y0, &run(t));
                }
            });
        } else {
            // Few, wide bands: the tiles of each band in parallel.
            for (i, band) in out.px.chunks_mut(band_len).enumerate() {
                let y0 = rect.y0 + i as i32 * tile;
                let parts: Vec<Buffer> = band_tiles(y0).into_par_iter().map(run).collect();
                for part in &parts {
                    put(band, y0, part);
                }
            }
        }
    }
    #[cfg(target_arch = "wasm32")]
    for (i, band) in out.px.chunks_mut(band_len).enumerate() {
        let y0 = rect.y0 + i as i32 * tile;
        for t in band_tiles(y0) {
            put(band, y0, &run(t));
        }
    }
    out
}

/// Runs `f` with this thread's document blend flags set ([`psblend::LAB_MIX`], [`psblend::HDR`]),
/// then restores the values they had (also on unwind): a nested rayon job can run another tile
/// on this thread mid-composite, and clearing the flags there broke the outer tile (#1112).
fn in_blend_space<R>(lab: bool, hdr: bool, f: impl FnOnce() -> R) -> R {
    psblend::with_lab_mix(lab, || psblend::with_hdr(hdr, f))
}

/// The document's own CMYK profile for reading its CMYK pixels (`None`: not a CMYK document,
/// untagged, or the built-in coated CMYK). Enter it with `photocraft_color::convert::with_cmyk_space`
/// around code that converts the document's CMYK pixels or colours to RGB.
pub fn cmyk_space(doc: &Document) -> Option<std::sync::Arc<photocraft_color::convert::CmykSpace>> {
    if doc.mode != photocraft_color::ColorMode::Cmyk {
        return None;
    }
    photocraft_color::convert::CmykSpace::for_profile(doc.icc_profile.as_ref())
}

/// Pixels per band of [`render_bands`] (a 14000 px wide band is ~600 rows, ~130 MB of f32).
pub const BAND_PIXELS: u64 = 8 << 20;

/// Composite `rect` in horizontal bands, top to bottom, handing each band to `sink` (tiles inside
/// a band render in parallel). Peak memory is one band instead of the whole composite, so huge
/// documents can be exported or displayed without a full-size float copy. `band_rows` of 0 picks
/// about [`BAND_PIXELS`] per band; bands are whole multiples of [`RENDER_TILE`] rows.
pub fn render_bands<E>(doc: &Document, rect: Rect, band_rows: i32, mut sink: impl FnMut(Buffer) -> Result<(), E>) -> Result<(), E> {
    if rect.is_empty() {
        return Ok(());
    }
    // Keep prepared pixels and compiled vector masks across bands without retaining any rendered band.
    let patterns = pattern::PreparedPatterns::new(&doc.patterns, pattern::PREPARED_PATTERN_BYTES).in_mode(doc.pixel_format().mode);
    let cx = Ctx::for_doc(doc, &patterns);
    let rows = band_rows_for(rect.width(), band_rows);
    let mut y = rect.y0;
    while y < rect.y1 {
        let y1 = y.saturating_add(rows).min(rect.y1);
        sink(render_tiled_with(doc, Rect::new(rect.x0, y, rect.x1, y1), RENDER_TILE, &cx))?;
        y = y1;
    }
    Ok(())
}

/// Band height for [`render_bands`]: `requested` rounded up to whole tiles, or ~[`BAND_PIXELS`].
pub fn band_rows_for(width: u32, requested: i32) -> i32 {
    let want = if requested > 0 { i64::from(requested) } else { (BAND_PIXELS / u64::from(width.max(1))) as i64 };
    let tiles = (want + i64::from(RENDER_TILE) - 1) / i64::from(RENDER_TILE);
    (tiles.clamp(1, 1 << 16) * i64::from(RENDER_TILE)) as i32
}

/// The document's composite as a pixel surface in `fmt` (straight RGBA converted to its model
/// and depth), optionally flattened over an opaque `background`; rendered and written in bands,
/// so no full-size float composite is held. Tiles equal to the default pixel are pruned.
///
/// Fully transparent pixels are written as the default pixel. Converting transparent black
/// leaves colour behind in CMYK (K = 100%) and Lab (neutral a/b = 0.5), and those pixels would
/// otherwise count as content: the result would span the whole canvas.
///
/// A CMYK document of plain Normal layers is composited in its ink values instead
/// ([`native::flatten_cmyk`]), so flattening doesn't round-trip its pixels through RGB (#3778).
pub fn flatten_to_surface(doc: &Document, fmt: photocraft_color::PixelFormat, background: Option<[f32; 3]>) -> Surface {
    composite_to_surface(doc, doc.bounds(), fmt, background)
}

/// [`flatten_to_surface`] over `rect` only (the rest of the surface stays empty).
pub fn composite_to_surface(doc: &Document, rect: Rect, fmt: photocraft_color::PixelFormat, background: Option<[f32; 3]>) -> Surface {
    let paper = background.is_some_and(|bg| bg == [1.0; 3]);
    if (background.is_none() || paper)
        && let Some(s) = native::flatten_cmyk(doc, rect, fmt, paper)
    {
        return s;
    }
    let mut s = Surface::new(fmt);
    let n = fmt.channels();
    let empty = s.default_pixel();
    let mut data = Vec::new();
    // The composite separates back to CMYK through the document's own profile, the one its
    // pixels were read through.
    let cmyk = cmyk_space(doc);
    let _ = render_bands(doc, rect, 0, |band| -> Result<(), ()> {
        let band = match background {
            Some(bg) => band.over_background(bg),
            None => band,
        };
        data.clear();
        data.resize(band.px.len() * n, 0.0);
        photocraft_color::convert::with_cmyk_space(cmyk.as_ref(), || {
            for (p, out) in band.px.iter().zip(data.chunks_exact_mut(n)) {
                if fmt.alpha && (p[3] <= 0.0 || p[3].is_nan()) && empty.len() == n {
                    out.copy_from_slice(&empty);
                } else {
                    photocraft_raster::from_rgba_into(&fmt, *p, out);
                }
            }
        });
        s.write_region(band.rect, &data);
        Ok(())
    });
    s.prune();
    s
}

/// Composite the full canvas.
pub fn flatten(doc: &Document) -> Buffer {
    render(doc, doc.bounds())
}

/// Render an arbitrary subset: a single layer (e.g. for thumbnails), isolated.
pub fn render_layer(layer: &Layer, rect: Rect) -> Buffer {
    let mut buf = Buffer::transparent(rect);
    let patterns = pattern::PreparedPatterns::new(&[], pattern::PREPARED_PATTERN_BYTES);
    composite_stack_at(
        std::slice::from_ref(layer),
        &mut buf,
        &Ctx {
            canvas: rect,
            transfer: adjust::Transfer::Srgb,
            light: photocraft_doc::GlobalLight::default(),
            patterns: &patterns,
            mode: photocraft_color::ColorMode::Rgb,
            depth: photocraft_color::SampleType::F32,
            vector_masks: RenderVectorMasks::default(),
            fx_maps: Default::default(),
            fx_clip: Default::default(),
            doc: None,
        },
        advanced::Scope::ROOT,
    );
    buf
}

/// Documents above this many pixels get thumbnails from a proxy (see [`proxy`]) when that is
/// faithful; smaller ones are reduced from the exact composite.
pub const PROXY_THUMBNAIL_PIXELS: u64 = 16 << 20;

/// Downscaled RGBA8 render of the document (area-averaged) for thumbnails, navigators and
/// histograms, at most `max_side` pixels on its longer side. Large documents are reduced from a
/// downsampled proxy (about twice the thumbnail's size), so a thumbnail of a 200 MP document
/// costs milliseconds and no full-size composite; the rest stream the exact composite in bands.
pub fn thumbnail(doc: &Document, max_side: u32) -> Rgba8Image {
    thumbnail_buffer(doc, max_side).to_rgba8()
}

/// Width and height of [`thumbnail`]`(doc, max_side)`.
pub fn thumbnail_size(doc: &Document, max_side: u32) -> (u32, u32) {
    let b = doc.bounds();
    let longest = b.width().max(b.height()).max(1);
    let scale = (max_side as f32 / longest as f32).min(1.0);
    (((b.width() as f32 * scale).round() as u32).max(1), ((b.height() as f32 * scale).round() as u32).max(1))
}

/// [`thumbnail`] as straight-alpha floats (for callers that colour-convert it first).
pub fn thumbnail_buffer(doc: &Document, max_side: u32) -> Buffer {
    let b = doc.bounds();
    let longest = b.width().max(b.height()).max(1);
    let (w, h) = thumbnail_size(doc, max_side);
    let k = longest / max_side.max(1).saturating_mul(2);
    if k >= 2 && doc.size.area() > PROXY_THUMBNAIL_PIXELS && proxy::proxy_faithful(doc) {
        render_reduced(&proxy::proxy_document(doc, k), w, h)
    } else {
        render_reduced(doc, w, h)
    }
}

/// The document's composite area-averaged (premultiplied) down to `w`×`h` (clamped to the
/// document size), rendered in bands so no full-size composite is held.
pub fn render_reduced(doc: &Document, w: u32, h: u32) -> Buffer {
    render_reduced_in_bands(doc, doc.bounds(), w, h, None, 0)
}

/// [`render_reduced`] of any area of the document plane, `rect`, which may lie past the canvas
/// (layer pixels there are kept until a crop deletes them). The buffer's rect is in output pixels
/// from (0, 0).
pub fn render_reduced_rect(doc: &Document, rect: Rect, w: u32, h: u32) -> Buffer {
    render_reduced_in_bands(doc, rect, w, h, None, 0)
}

/// The part of [`render_reduced`]`(doc, w, h)` that a change to `damage` (document pixels) can
/// affect: the output pixels whose source areas meet it, rendered from just those areas, with the
/// same values the whole reduction gives them. The buffer's rect is in output pixels (empty when
/// `damage` misses the document), so a reduced canvas texture can update only what a stroke touched.
pub fn render_reduced_damage(doc: &Document, w: u32, h: u32, damage: Rect) -> Buffer {
    render_reduced_in_bands(doc, doc.bounds(), w, h, Some(damage), 0)
}

/// [`render_reduced_rect`] (or, with `damage`, [`render_reduced_damage`]) of area `b` with an
/// explicit band height (see [`render_bands`]).
fn render_reduced_in_bands(doc: &Document, b: Rect, w: u32, h: u32, damage: Option<Rect>, band_rows: i32) -> Buffer {
    let (fw, fh) = (b.width() as usize, b.height() as usize);
    let (w, h) = (w.clamp(1, b.width().max(1)) as usize, h.clamp(1, b.height().max(1)) as usize);
    let full = Rect::from_xywh(0, 0, w as u32, h as u32);
    if fw == 0 || fh == 0 {
        return Buffer::transparent(if damage.is_some() { Rect::EMPTY } else { full });
    }
    // Output column / row of each document column / row: output pixel t covers [t·f/n, (t+1)·f/n).
    let span = |t: usize, f: usize, n: usize| (t * f / n, ((t + 1) * f / n).max(t * f / n + 1).min(f));
    // The output pixel whose span holds document column / row `x` (spans are contiguous: n <= f).
    let cell = |x: usize, f: usize, n: usize| ((x + 1) * n).saturating_sub(1) / f;
    // Output pixels to produce and the document area they cover.
    let out = match damage {
        None => full,
        Some(d) => {
            let d = d.intersect(&b);
            if d.is_empty() {
                return Buffer::transparent(Rect::EMPTY);
            }
            let (x0, x1) = ((d.x0 - b.x0) as usize, (d.x1 - b.x0) as usize);
            let (y0, y1) = ((d.y0 - b.y0) as usize, (d.y1 - b.y0) as usize);
            Rect::new(cell(x0, fw, w) as i32, cell(y0, fh, h) as i32, cell(x1 - 1, fw, w) as i32 + 1, cell(y1 - 1, fh, h) as i32 + 1)
        }
    };
    let (ox0, oy0, ow, oh) = (out.x0 as usize, out.y0 as usize, out.width() as usize, out.height() as usize);
    let (sx0, sx1) = (span(ox0, fw, w).0, span(ox0 + ow - 1, fw, w).1);
    let (sy0, sy1) = (span(oy0, fh, h).0, span(oy0 + oh - 1, fh, h).1);
    let src = Rect::new(b.x0 + sx0 as i32, b.y0 + sy0 as i32, b.x0 + sx1 as i32, b.y0 + sy1 as i32);
    if (w, h) == (fw, fh) {
        let mut px = render(doc, src);
        px.rect = out;
        return px;
    }
    let map = |f: usize, n: usize| {
        let mut m = vec![0u32; f];
        for t in 0..n {
            let (a, z) = span(t, f, n);
            m[a..z].fill(t as u32);
        }
        m
    };
    let (cols, rows) = (map(fw, w), map(fh, h));
    let cols = &cols[sx0..sx1];
    let mut acc = vec![[0.0f32; 4]; ow * oh];
    let ok: Result<(), ()> = render_bands(doc, src, band_rows, |band| {
        let (by0, by1) = ((band.rect.y0 - b.y0) as usize, (band.rect.y1 - b.y0) as usize);
        let (t0, t1) = (rows[by0] as usize, rows[by1 - 1] as usize + 1);
        let sw = sx1 - sx0;
        // Each output row sums its source rows in this band (in order, top to bottom).
        let sum_row = |t: usize, out: &mut [[f32; 4]]| {
            let (a, z) = span(t, fh, h);
            for y in a.max(by0)..z.min(by1) {
                let src = &band.px[(y - by0) * sw..(y - by0 + 1) * sw];
                for (p, &c) in src.iter().zip(cols) {
                    let o = &mut out[c as usize - ox0];
                    for i in 0..3 {
                        o[i] += p[i] * p[3];
                    }
                    o[3] += p[3];
                }
            }
        };
        let part = &mut acc[(t0 - oy0) * ow..(t1 - oy0) * ow];
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            part.par_chunks_mut(ow).enumerate().for_each(|(i, out)| sum_row(t0 + i, out));
        }
        #[cfg(target_arch = "wasm32")]
        for (i, out) in part.chunks_mut(ow).enumerate() {
            sum_row(t0 + i, out);
        }
        Ok(())
    });
    debug_assert!(ok.is_ok());
    for (i, p) in acc.iter_mut().enumerate() {
        let ((x0, x1), (y0, y1)) = (span(ox0 + i % ow, fw, w), span(oy0 + i / ow, fh, h));
        let n = ((y1 - y0) * (x1 - x0)).max(1) as f32;
        let a = p[3];
        *p = if a > 0.0 { [p[0] / a, p[1] / a, p[2] / a, a / n] } else { [0.0; 4] };
    }
    Buffer { rect: out, px: acc }
}

type VectorMaskSlot = std::sync::Arc<std::sync::OnceLock<photocraft_vector::CompiledVectorMask>>;

#[derive(Default)]
struct RenderVectorMasks {
    /// Compiled masks by content. Callers pass temporary clones (see `advanced.rs`), so an
    /// address cannot identify a mask: a dropped clone's address is reused by the next one.
    masks: std::sync::Mutex<Vec<(photocraft_doc::VectorMask, VectorMaskSlot)>>,
}

impl RenderVectorMasks {
    fn values(&self, mask: &photocraft_doc::VectorMask, rect: Rect) -> Vec<f32> {
        let slot = {
            let mut masks = self.masks.lock().unwrap_or_else(|e| e.into_inner());
            match masks.iter().find(|(m, _)| m == mask) {
                Some((_, slot)) => slot.clone(),
                None => {
                    let slot = VectorMaskSlot::default();
                    masks.push((mask.clone(), slot.clone()));
                    slot
                }
            }
        };
        // Compilation is sequential. Rendering can enter Rayon and must happen after
        // initialization, so waiting tiles cannot re-enter a slot still being built.
        let compiled = slot.get_or_init(|| photocraft_vector::CompiledVectorMask::new(mask));
        compiled.render(rect)
    }
}

/// Rendering context shared down the tree.
struct Ctx<'a> {
    /// Document canvas: fill layers and gradients are laid out relative to it, never to the render rect.
    canvas: Rect,
    /// Tone transfer used by adjustments that work in linear light.
    transfer: adjust::Transfer,
    /// Global light for layer effects.
    light: photocraft_doc::GlobalLight,
    /// Prepared document patterns, shared across all tiles and bands of this render call.
    patterns: &'a pattern::PreparedPatterns<'a>,
    /// The document's colour mode (channel restrictions name its channels).
    mode: photocraft_color::ColorMode,
    depth: photocraft_color::SampleType,
    vector_masks: RenderVectorMasks,
    /// Effect maps used by this render, by cache key: built before the parallel tiles (see
    /// [`prepare_effects`]) and held for the whole render (or band, for maps clipped to it: the
    /// `bool`), so eviction can't force a rebuild.
    fx_maps: std::sync::Mutex<std::collections::HashMap<u64, (std::sync::Arc<effects::FxMaps>, bool)>>,
    /// The rectangle being rendered (a band of [`render_bands`]); see [`effect_region`].
    fx_clip: std::sync::Mutex<Option<Rect>>,
    /// The document being rendered (deep knockouts reveal its Background); `None` for a lone layer.
    doc: Option<&'a Document>,
}

impl<'a> Ctx<'a> {
    fn for_doc(doc: &'a Document, patterns: &'a pattern::PreparedPatterns<'a>) -> Self {
        Self {
            doc: Some(doc),
            canvas: doc.bounds(),
            transfer: adjust::Transfer::for_document(doc.mode, doc.depth),
            light: doc.global_light,
            patterns,
            mode: doc.mode,
            depth: doc.depth,
            vector_masks: RenderVectorMasks::default(),
            fx_maps: Default::default(),
            fx_clip: Default::default(),
        }
    }

    /// Start rendering `rect`: effect maps clipped to the previous rectangle are let go.
    fn set_fx_clip(&self, rect: Rect) {
        let mut clip = self.fx_clip.lock().unwrap_or_else(|e| e.into_inner());
        if *clip != Some(rect) {
            *clip = Some(rect);
            self.fx_maps.lock().unwrap_or_else(|e| e.into_inner()).retain(|_, (_, clipped)| !*clipped);
        }
    }
}

/// Deepest group nesting [`prepare_effects`] walks (deeper layers build their maps on demand).
const PREPARE_DEPTH: u32 = 64;

/// Build the effect maps of every visible layer in `layers` (groups included) that can reach
/// `rect` before tiles render in parallel, layers concurrently on native targets, each build
/// inside `setting` (the tiles' thread-local colour setting). Tiles then only read finished maps:
/// none builds one while others need it (#276), and none duplicates another's build. Builds never
/// wait on each other (`cached_effect_maps`), so building them in parallel can't deadlock.
fn prepare_effects(layers: &[Layer], rect: Rect, cx: &Ctx, setting: impl Fn(&mut dyn FnMut()) + Sync) {
    let mut todo = Vec::new();
    effect_layers(layers, rect, 0, &mut todo);
    let build = |l: &std::borrow::Cow<Layer>| {
        setting(&mut || {
            let _ = effect_maps(l, rect, cx);
        });
    };
    #[cfg(not(target_arch = "wasm32"))]
    if todo.len() > 1 {
        use rayon::prelude::*;
        todo.par_iter().for_each(build);
        return;
    }
    todo.iter().for_each(build);
}

/// The visible layers of `layers` (groups included) with effects that can reach `rect`.
fn effect_layers<'l>(layers: &'l [Layer], rect: Rect, depth: u32, out: &mut Vec<std::borrow::Cow<'l, Layer>>) {
    let mut base_visible = true;
    for l in layers {
        if !l.clipped {
            base_visible = l.visible;
        }
        // A clipping group is drawn only when its base is.
        if !l.visible || !base_visible {
            continue;
        }
        if effects::has_effects(l) && !empty_in(l, rect) {
            // A mask that hides the effects is left out of the shape they are built from.
            out.push(advanced::effects_source(l));
        }
        if let LayerContent::Group(g) = &l.content
            && depth < PREPARE_DEPTH
        {
            effect_layers(&g.children, rect, depth + 1, out);
        }
    }
}

/// Composite an isolated group's sibling list (bottom→top) onto `backdrop` (its own buffer).
fn composite_stack(layers: &[Layer], backdrop: &mut Buffer, cx: &Ctx) {
    composite_stack_at(layers, backdrop, cx, advanced::Scope::ISOLATED);
}

/// Composite a sibling list (bottom→top) onto `backdrop`; `scope` says where the stack sits (see
/// [`advanced::Scope`]: it decides what knockouts reveal).
fn composite_stack_at(layers: &[Layer], backdrop: &mut Buffer, cx: &Ctx, scope: advanced::Scope) {
    // Shallow knockouts stop at the bottom of this stack: the backdrop it starts from.
    let floor = (!scope.root && advanced::needs_floor(layers)).then(|| backdrop.clone());
    let mut i = 0;
    while i < layers.len() {
        let base = &layers[i];
        // Collect the clipping group: following layers with `clipped = true`.
        let mut j = i + 1;
        while j < layers.len() && layers[j].clipped && !base.clipped {
            j += 1;
        }
        let clipped = &layers[i + 1..j];
        if base.visible {
            let ko = advanced::knockout_target(base, floor.as_ref(), backdrop.rect, cx, scope);
            composite_layer(base, clipped, backdrop, cx, ko.as_deref(), scope);
            if let (LayerContent::Adjustment(_), Some(q)) = (&base.content, adjustment_quantum(cx.depth)) {
                quantize(backdrop, q);
            }
        }
        i = j.max(i + 1);
    }
}

/// Steps per unit of an integer document's samples: Photoshop applies adjustment layers to
/// buffers of the document's depth (8-bit: 255 levels, 16-bit: 32768), so their result is
/// rounded there; a steep curve then amplifies the rounding of its input exactly as in
/// Photoshop (psd-tools adjustment_nested_composition_4: 9.6 → 5.1 % of pixels off;
/// exposure_grayscale passes). Blends stay in float (quantising them too made other files
/// worse).
pub fn adjustment_quantum(depth: photocraft_color::SampleType) -> Option<f32> {
    match depth {
        photocraft_color::SampleType::U8 => Some(255.0),
        photocraft_color::SampleType::U16 => Some(32768.0),
        photocraft_color::SampleType::F32 => None,
    }
}

fn quantize(b: &mut Buffer, q: f32) {
    for p in &mut b.px {
        for v in p.iter_mut() {
            *v = (*v * q + 0.5).floor() / q;
        }
    }
}

/// Deterministic hash for Dissolve (document-coordinate based, so it is stable under tiling).
#[inline]
fn dissolve_noise(x: i32, y: i32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343) ^ (y as u32).wrapping_mul(0xd816_3841) ^ 0x9e37_79b9;
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 12;
    (h & 0xffff) as f32 / 65536.0
}

/// Bounds of a layer's own pixels (union over group children; the canvas
/// for fill layers; an artboard's board).
pub fn layer_bounds(layer: &Layer, canvas: Rect) -> Rect {
    // Transparency Shapes Layer off: effects and knockout cover the whole layer (the canvas).
    if advanced::shapeless(layer) {
        return canvas;
    }
    match &layer.content {
        LayerContent::Group(g) if g.artboard.is_some() => g.artboard.as_ref().map_or(Rect::EMPTY, |a| a.rect),
        LayerContent::Group(g) => g.children.iter().filter(|c| c.visible).fold(Rect::EMPTY, |acc, c| {
            let b = layer_bounds(c, canvas);
            if b.is_empty() {
                acc
            } else if acc.is_empty() {
                b
            } else {
                acc.union(&b)
            }
        }),
        LayerContent::Fill(_) => canvas,
        _ => {
            let b = layer.surface().map_or(Rect::EMPTY, bounds::content_bounds);
            let b = match &layer.content {
                LayerContent::Shape(sh) => {
                    let p = shape_bounds(sh).map(|r| r.intersect(&canvas)).unwrap_or(Rect::EMPTY);
                    if !b.is_empty() && !p.is_empty() {
                        b.union(&p)
                    } else if !p.is_empty() {
                        p
                    } else {
                        b
                    }
                }
                _ => b,
            };
            // Effects follow a filled shape's outline, also where its fill is transparent.
            match effect_outline(layer).and_then(|_| paint_bounds(layer)).filter(|_| effects::has_effects(layer)) {
                Some(p) if !b.is_empty() => b.union(&p),
                Some(p) => p,
                None => b,
            }
        }
    }
}

/// Where a layer's effects take their shape from: [`layer_bounds`], except that a group's
/// children count with their own effects (a child's outside stroke shapes the group's drop
/// shadow and must not be cut off at the child's pixels, #3274).
pub fn effect_source_bounds(layer: &Layer, canvas: Rect) -> Rect {
    match &layer.content {
        LayerContent::Group(g) if g.artboard.is_none() && !advanced::shapeless(layer) => g.children.iter().filter(|c| c.visible).fold(Rect::EMPTY, |acc, c| {
            let m = if effects::has_effects(c) { effects::margin(c) } else { 0 };
            let b = effect_source_bounds(c, canvas);
            let b = if b.is_empty() { b } else { b.inflate(m) };
            if b.is_empty() {
                acc
            } else if acc.is_empty() {
                b
            } else {
                acc.union(&b)
            }
        }),
        _ => layer_bounds(layer, canvas),
    }
}

/// The path a shape layer's effects are shaped by: its fill path. Photoshop builds a filled
/// shape's layer effects from its vector outline, not from its pixels: a stroke runs along the
/// whole path even where a gradient fill fades out (psd-tools stroke-effects). `None` for other
/// layers, shapes without fill, empty and inverted paths, and shapes whose vector stroke reaches
/// past the path (their pixels give the outline: psd-tools double-stroke-effects).
pub fn effect_outline(layer: &Layer) -> Option<&photocraft_doc::vector::Path> {
    let LayerContent::Shape(sh) = &layer.content else { return None };
    let stroke_inside = sh.stroke.as_ref().is_none_or(|s| s.width <= 0.0 || s.align == photocraft_doc::vector::StrokeAlign::Inside);
    (sh.fill.is_some() && stroke_inside && !sh.path.is_empty() && !sh.path.inverted).then_some(&sh.path)
}

/// The shape a layer's effects are built from over `rect` (row-major): its content's alpha
/// (masks applied), completed for shapes with an [`effect_outline`] by the outline's coverage
/// where the fill is see-through. With `l` the alpha and `f` the fill's own opacity (the largest
/// alpha around), the shape is `l + (1 - f) × coverage`: Photoshop's own rasterization where the
/// fill is opaque (it can differ from ours by a fraction of a pixel: psd-tools shape-fx2), the
/// path where the fill fades out (psd-tools stroke-effects).
fn effect_shape(layer: &Layer, rect: Rect, cx: &Ctx) -> Vec<f32> {
    let n = rect.width() as usize * rect.height() as usize;
    // Transparency Shapes Layer off: the effects take the whole layer as their shape (its masks
    // still apply).
    if !layer.advanced.transparency_shapes {
        return mask_vals(layer, rect, cx).unwrap_or_else(|| vec![1.0; n]);
    }
    let Some(path) = effect_outline(layer).filter(|_| n > 0) else {
        return render_content(layer, rect, cx).map(|b| b.px.iter().map(|p| p[3]).collect()).unwrap_or_else(|| vec![0.0; n]);
    };
    // Unmasked: the masks scale the shape, they aren't the fill's transparency.
    let a: Vec<f32> = layer.surface().map(|s| surface_to_buffer(s, rect).px.iter().map(|p| p[3]).collect()).unwrap_or_else(|| vec![0.0; n]);
    let cov = photocraft_vector::path_coverage(path, rect);
    let mv = mask_vals(layer, rect, cx);
    let (w, h) = (rect.width() as usize, rect.height() as usize);
    let mut out = vec![0.0; n];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            // A pixel the outline covers is inside, whatever the fill's opacity.
            let v = if cov[i] >= 1.0 - 1e-3 {
                1.0
            } else {
                let mut f = 0.0f32;
                for yy in y.saturating_sub(1)..(y + 2).min(h) {
                    for xx in x.saturating_sub(1)..(x + 2).min(w) {
                        f = f.max(a[yy * w + xx]);
                    }
                }
                (a[i] + (1.0 - f.min(1.0)) * cov[i]).clamp(0.0, 1.0)
            };
            out[i] = v * mask_k(&mv, i);
        }
    }
    out
}

/// The frame layer-effect gradients and linked patterns are laid out in, when it isn't the
/// layer's pixel bounds: a shape layer's path bounds, truncated to whole pixels. Its rendered
/// pixels can extend past the path (transparent anti-aliasing margin), which Photoshop ignores:
/// psd-tools shape-fx2's 45° overlay spans the 0.92–29.61 path as 0–29 (29 px, centred at 14.5),
/// not the 32 px of pixels.
pub fn paint_bounds(layer: &Layer) -> Option<Rect> {
    let LayerContent::Shape(sh) = &layer.content else { return None };
    let (x0, y0, x1, y1) = sh.path.control_bounds()?;
    let r = Rect::new(x0.floor() as i32, y0.floor() as i32, x1.floor() as i32, y1.floor() as i32);
    (!r.is_empty()).then_some(r)
}

/// The bounding box of a shape layer's vector content: its path control bounds plus stroke width.
pub fn shape_bounds(sh: &photocraft_doc::ShapeLayer) -> Option<Rect> {
    if sh.fill.is_none() && sh.stroke.as_ref().is_none_or(|s| s.width <= 0.0 || s.opacity <= 0.0) {
        return None;
    }
    let (x0, y0, x1, y1) = sh.path.control_bounds()?;
    let stroke_margin = sh.stroke.as_ref().filter(|s| s.opacity > 0.0).map_or(0.0, |s| s.width.ceil()) as i32;
    let r = Rect::new(x0.floor() as i32, y0.floor() as i32, x1.ceil() as i32, y1.ceil() as i32).inflate(stroke_margin);
    (!r.is_empty()).then_some(r)
}

/// The layer's effective mask over `rect` (row-major), read once per tile: the pixel mask
/// times the rasterized vector mask.
fn mask_vals(layer: &Layer, rect: Rect, cx: &Ctx) -> Option<Vec<f32>> {
    // Feathers blur across tiles: read the cached, canvas-wide combined mask.
    if masks::has_feather(layer)
        && let Some(s) = masks::combined_mask(layer, cx.canvas)
    {
        let mut v = Vec::new();
        s.read_region_into(rect, &mut v);
        return Some(v);
    }
    let vector = layer.vector_mask.as_ref().map(|vm| cx.vector_masks.values(vm, rect));
    let Some(m) = layer.mask.as_ref() else { return vector };
    let mut v = Vec::new();
    m.values_into(rect, &mut v);
    if let Some(vm) = vector {
        for (a, b) in v.iter_mut().zip(vm) {
            *a *= b;
        }
    }
    Some(v)
}

#[inline]
fn mask_k(m: &Option<Vec<f32>>, i: usize) -> f32 {
    m.as_ref().map_or(1.0, |v| v[i])
}

/// Render a layer's own content (no blending into the backdrop yet) into an isolated buffer.
/// Returns None for layers that operate on the backdrop (adjustments, pass-through groups).
fn render_content(layer: &Layer, rect: Rect, cx: &Ctx) -> Option<Buffer> {
    let mut buf = match &layer.content {
        LayerContent::Group(g) => {
            let mut b = Buffer::transparent(rect);
            composite_stack(&g.children, &mut b, cx);
            b
        }
        LayerContent::Fill(f) => match &layer.fill_cache {
            // Photoshop's own rendering, valid while the fill is unchanged.
            Some(c) if c.fill == *f => surface_to_buffer(&c.surface, rect),
            _ if cx.mode == photocraft_color::ColorMode::Cmyk => {
                let frame = fill_frame(layer, cx.canvas);
                let native_rect = if matches!(f, Fill::Solid(_)) { Rect::new(0, 0, 1, 1) } else { rect };
                match gradient_fill::render_cmyk_fill(f, native_rect, frame, cx.depth) {
                    Some(s) if matches!(f, Fill::Solid(_)) => Buffer::filled(rect, s.rgba(0, 0)),
                    Some(s) => surface_to_buffer(&s, rect),
                    None => render_fill(f, rect, frame, cx.patterns, adjustment_quantum(cx.depth)),
                }
            }
            _ => render_fill(f, rect, fill_frame(layer, cx.canvas), cx.patterns, adjustment_quantum(cx.depth)),
        },
        LayerContent::Adjustment(_) => return None,
        _ => match layer.surface() {
            Some(s) => surface_to_buffer(s, rect),
            None => Buffer::transparent(rect),
        },
    };
    if let Some(m) = mask_vals(layer, rect, cx) {
        for (p, k) in buf.px.iter_mut().zip(&m) {
            p[3] *= k;
        }
    }
    Some(buf)
}

/// The shape `layer`'s effect maps are built from over `rect` (row-major; for the GPU
/// compositor): its content's alpha (masks applied), joined with the [`effect_outline`] of a
/// filled shape. Zero for adjustment layers.
///
/// See also [`stroke_frame`].
pub fn layer_shape(doc: &Document, layer: &Layer, rect: Rect) -> Vec<f32> {
    let patterns = pattern::PreparedPatterns::new(&doc.patterns, pattern::PREPARED_PATTERN_BYTES).in_mode(doc.pixel_format().mode);
    let cx = Ctx::for_doc(doc, &patterns);
    effect_shape(layer, rect, &cx)
}

/// The frame a gradient stroke `st` of `layer` is laid out in (`effects::FxMaps::stroke_frame`),
/// from the layer's cached effect maps; `None` when `st` isn't a gradient stroke of the layer.
/// For the GPU compositor.
pub fn stroke_frame(doc: &Document, layer: &Layer, st: &photocraft_doc::StrokeFx) -> Option<Rect> {
    if !matches!(st.paint, photocraft_doc::FxPaint::Gradient(_)) || !effects::has_effects(layer) {
        return None;
    }
    let patterns = pattern::PreparedPatterns::new(&doc.patterns, pattern::PREPARED_PATTERN_BYTES).in_mode(doc.pixel_format().mode);
    let cx = Ctx::for_doc(doc, &patterns);
    effect_maps(layer, doc.bounds(), &cx).stroke_frame(st)
}

pub fn surface_to_buffer(s: &Surface, rect: Rect) -> Buffer {
    let mut px = vec![[0.0f32; 4]; rect.width() as usize * rect.height() as usize];
    s.read_rgba_into(rect, &mut px);
    Buffer { rect, px }
}

/// The frame a fill layer's gradient is laid out in ("Align with layer", Photoshop's default):
/// the layer's bounds, i.e. the area its masks reveal: a hide-all pixel mask's painted area, or
/// the vector mask's path; otherwise the canvas.
pub fn fill_frame(layer: &Layer, canvas: Rect) -> Rect {
    let mut frame = canvas;
    // "Align with layer" off: the canvas (the Gradient tool's live gradients).
    if let LayerContent::Fill(Fill::Gradient { align: false, .. }) = &layer.content {
        return frame;
    }
    if let Some(m) = &layer.mask
        && m.enabled
        && m.surface.default_pixel().first().is_some_and(|v| *v <= 0.0)
    {
        let b = bounds::content_bounds(&m.surface).intersect(&canvas);
        if !b.is_empty() {
            frame = b;
        }
    }
    if let Some(vm) = &layer.vector_mask
        && vm.enabled
        && !vm.path.inverted
        && let Some((x0, y0, x1, y1)) = vm.path.control_bounds()
    {
        let b = Rect::new(x0.floor() as i32, y0.floor() as i32, x1.ceil() as i32, y1.ceil() as i32).intersect(&frame);
        if !b.is_empty() {
            frame = b;
        }
    }
    frame
}

/// A fill layer's content over `canvas`, laid out in the frame its masks give it (as the
/// compositor does) but without applying the masks: the pixels a PSD fill layer stores.
pub fn render_fill_content(layer: &Layer, f: &Fill, canvas: Rect, patterns: &[Pattern]) -> Buffer {
    let prepared = pattern::PreparedPatterns::new(patterns, pattern::PREPARED_PATTERN_BYTES);
    render_fill(f, canvas, fill_frame(layer, canvas), &prepared, None)
}

/// `quantum`: the document depth's levels, which a dithered gradient is rounded to
/// ([`gradient_fill::render_quantized`]); `None` keeps it in float.
fn render_fill(f: &Fill, rect: Rect, canvas: Rect, patterns: &pattern::PreparedPatterns<'_>, quantum: Option<f32>) -> Buffer {
    match f {
        Fill::Solid(c) => {
            let rgb = c.to_rgb();
            Buffer::filled(rect, [rgb[0], rgb[1], rgb[2], c.alpha])
        }
        // Gradient geometry relative to the layer's frame, independent of the render rect.
        Fill::Gradient { .. } => Buffer { rect, px: gradient_fill::render_quantized(f, rect, canvas, quantum) },
        // Laid out from the layer's frame when linked; transparent if the pattern is missing.
        Fill::Pattern { name, scale, id, angle, link, phase } => match patterns.get(id, name) {
            Some(tile) => Buffer { rect, px: pattern::render(&tile, &pattern::Placement::new(canvas, *link, *phase, *scale, *angle), rect) },
            None => Buffer::transparent(rect),
        },
    }
}

/// `true` when a pixel-backed layer has nothing to contribute in `rect`:
/// no allocated tiles there, a transparent default, and no effects (which
/// could reach in from outside). Clipped layers depend on the base, so they
/// vanish with it.
fn empty_in(layer: &Layer, rect: Rect) -> bool {
    if effects::has_effects(layer) {
        // Effects reach at most `margin` beyond the layer's pixels (when it is transparent
        // outside them; a group's from its children with their effects): render tiles away from a
        // small text layer skip it entirely.
        let canvas = Rect::new(i32::MIN / 4, i32::MIN / 4, i32::MAX / 4, i32::MAX / 4);
        let bounds = shapeless_stroke_bounds(layer).unwrap_or_else(|| effect_source_bounds(layer, canvas));
        return transparent_outside(layer) && bounds.inflate(effects::margin(layer)).intersect(&rect).is_empty();
    }
    match &layer.content {
        LayerContent::Raster(_) | LayerContent::Text(_) | LayerContent::Shape(_) | LayerContent::Smart(_) => match layer.surface() {
            Some(s) => !s.has_tiles_in(rect) && s.default_pixel().last().is_some_and(|a| *a <= 0.0) && s.format().alpha,
            None => true,
        },
        _ => false,
    }
}

/// For the narrow `tsly=0` case where only local stroke effects are enabled, the effects cannot
/// contribute outside the stroke reach from the layer's pixels. Other `tsly=0` effects can cover
/// the full layer, and advanced blending can change the backdrop independently of those pixels.
fn shapeless_stroke_bounds(layer: &Layer) -> Option<Rect> {
    use photocraft_doc::Knockout;

    if layer.advanced.transparency_shapes
        || layer.advanced.knockout != Knockout::None
        || layer.advanced.blend_interior
        || layer.clipped
        || !layer.blend_if.is_default()
        || layer.mask.as_ref().is_some_and(|m| m.enabled)
        || layer.vector_mask.as_ref().is_some_and(|m| m.enabled)
        || !transparent_outside(layer)
        || !effects::maps_are_local(layer)
    {
        return None;
    }

    if !layer.effects.enabled
        || !layer.effects.items.iter().any(|e| e.enabled())
        || layer.effects.items.iter().any(|e| e.enabled() && !matches!(e, photocraft_doc::Effect::Stroke(_)))
    {
        return None;
    }

    match &layer.content {
        LayerContent::Raster(_) | LayerContent::Text(_) | LayerContent::Shape(_) | LayerContent::Smart(_) => {
            let b = layer.surface().map_or(Rect::EMPTY, bounds::content_bounds);
            // A filled shape's effects follow its outline, also where its fill is transparent (as
            // in `layer_bounds`).
            Some(match effect_outline(layer).and_then(|_| paint_bounds(layer)) {
                Some(p) if !b.is_empty() => b.union(&p),
                Some(p) => p,
                None => b,
            })
        }
        _ => None,
    }
}

/// Blending Options › Channels as per-channel weights over display RGB (1 = the layer's result,
/// 0 = the backdrop's value kept), or `None` when every channel blends. RGB documents map R, G, B
/// directly and a grayscale document's single channel covers all three; other modes composite in
/// display RGB, where a restriction to their own channels has no exact equivalent, so it is
/// ignored there.
pub fn channel_weights(layer: &Layer, mode: photocraft_color::ColorMode) -> Option<[f32; 3]> {
    use photocraft_color::ColorMode as M;
    let x = layer.excluded_channels;
    if x == 0 {
        return None;
    }
    let keep = |bit: u32| if x & (1 << bit) != 0 { 0.0 } else { 1.0 };
    match mode {
        M::Rgb => Some([keep(0), keep(1), keep(2)]),
        M::Grayscale | M::Duotone => Some([keep(0); 3]),
        _ => None,
    }
    .filter(|w| w != &[1.0; 3])
}

/// Put back the backdrop's values in the channels a layer leaves out (`w` from
/// [`channel_weights`]); alpha stays the layer's result.
fn restore_channels(out: &mut Buffer, before: &Buffer, w: [f32; 3]) {
    for (p, b) in out.px.iter_mut().zip(&before.px) {
        for c in 0..3 {
            p[c] = b[c] + (p[c] - b[c]) * w[c];
        }
    }
}

/// Where `layer`'s composite (its pixels and effects; a group's visible children with theirs) can
/// be non-transparent, or `None` when it may draw anywhere (a surface with an opaque default
/// pixel). Outside it, compositing the layer leaves the backdrop unchanged in every blend mode.
/// Adjustment layers count as empty: they only change pixels already there (which is what an
/// isolated group needs; see [`change_bounds`] for the document-level question).
pub fn composite_bounds(layer: &Layer, canvas: Rect) -> Option<Rect> {
    if !transparent_outside(layer) {
        return None;
    }
    let own = match &layer.content {
        LayerContent::Group(g) if g.artboard.is_none() => {
            let mut acc = Rect::EMPTY;
            for c in g.children.iter().filter(|c| c.visible) {
                let b = composite_bounds(c, canvas)?;
                if !b.is_empty() {
                    acc = if acc.is_empty() { b } else { acc.union(&b) };
                }
            }
            acc
        }
        LayerContent::Adjustment(_) => Rect::EMPTY,
        _ => layer_bounds(layer, canvas),
    };
    if own.is_empty() {
        return Some(Rect::EMPTY);
    }
    let m = if effects::has_effects(layer) { effects::margin(layer) } else { 0 };
    Some(own.inflate(m).intersect(&canvas))
}

/// The document pixels that showing, hiding, moving or restyling `layer` can change, or `None`
/// for anywhere: an adjustment layer (and a pass-through group holding one) reaches everything
/// beneath it. Clipped layers above stay within their base's bounds, so they are covered too;
/// effects of groups around the layer can reach further (callers grow the rect by that reach).
pub fn change_bounds(layer: &Layer, canvas: Rect) -> Option<Rect> {
    fn reaches_below(l: &Layer) -> bool {
        match &l.content {
            // Clipped, it changes only its base's pixels.
            LayerContent::Adjustment(_) => !l.clipped,
            LayerContent::Group(g) if l.blend == BlendMode::PassThrough => g.children.iter().any(reaches_below),
            _ => false,
        }
    }
    if matches!(layer.content, LayerContent::Adjustment(_)) || reaches_below(layer) {
        return None;
    }
    composite_bounds(layer, canvas)
}

/// Whether the layer's content is transparent outside its bounds (every surface it draws from
/// has a transparent default pixel), so its effects can't change pixels beyond its bounds grown
/// by their reach.
pub fn transparent_outside(layer: &Layer) -> bool {
    match &layer.content {
        LayerContent::Group(g) => g.children.iter().filter(|c| c.visible).all(transparent_outside),
        // Fill layers cover the canvas (their bounds); adjustments draw nothing of their own.
        LayerContent::Fill(_) | LayerContent::Adjustment(_) => true,
        _ => layer.surface().is_none_or(|s| s.format().alpha && s.default_pixel().last().is_some_and(|a| *a <= 0.0)),
    }
}

/// Whether `layer` replaces everything beneath it in its stack: a visible, unclipped fill layer
/// in Normal mode at full opacity and fill, without masks, effects, channel restrictions or Blend
/// If, whose fill is opaque everywhere (a solid colour, or a gradient whose colour and opacity
/// stops are all opaque). Normal blending at alpha 1 returns the layer's colour exactly, so the
/// layers below can be skipped (the live Gradient tool's preview recomposites the whole canvas on
/// every pointer move).
pub fn occludes_below(layer: &Layer, mode: photocraft_color::ColorMode) -> bool {
    let LayerContent::Fill(f) = &layer.content else { return false };
    let opaque = match f {
        Fill::Solid(c) => c.alpha >= 1.0,
        Fill::Gradient { stops, opacity_stops, .. } => {
            !stops.is_empty() && stops.iter().all(|(_, c)| c.alpha >= 1.0) && opacity_stops.iter().all(|(_, o)| *o >= 1.0)
        }
        Fill::Pattern { .. } => false,
    };
    // A cached render in use (a PSD's stored fill pixels) may not be opaque.
    let cached = layer.fill_cache.as_ref().is_some_and(|c| c.fill == *f);
    opaque
        && !cached
        && layer.visible
        && !layer.clipped
        && layer.blend == photocraft_color::BlendMode::Normal
        && layer.opacity >= 1.0
        && layer.fill_opacity >= 1.0
        && !layer.mask.as_ref().is_some_and(|m| m.enabled)
        && !layer.vector_mask.as_ref().is_some_and(|m| m.enabled)
        && !effects::has_effects(layer)
        && layer.excluded_channels == 0
        && !blend_if_active(layer, mode)
}

/// Whether Blending Options › Blend If changes how `layer` composites in a `mode` document.
/// RGB documents test Gray and R, G, B; grayscale (and duotone) documents their one channel.
/// Other modes composite in display RGB, where ranges over their own channels have no exact
/// equivalent, so (like channel restrictions) the setting round-trips but isn't applied there.
pub fn blend_if_active(layer: &Layer, mode: photocraft_color::ColorMode) -> bool {
    use photocraft_color::ColorMode as M;
    matches!(mode, M::Rgb | M::Grayscale | M::Duotone) && !layer.blend_if.is_default()
}

/// How much of a pixel shows through `layer`'s Blend If ranges, given the layer's own colour
/// (`this`, `None` where the layer has no content of its own there) and the colour beneath it
/// (`under`, `None` where nothing is beneath). Every range multiplies in.
fn blend_if_weight(layer: &Layer, mode: photocraft_color::ColorMode, this: Option<[f32; 4]>, under: Option<[f32; 4]>) -> f32 {
    use photocraft_color::ColorMode as M;
    let bi = &layer.blend_if;
    let mut k = 1.0;
    for (side, px) in [this, under].into_iter().enumerate() {
        let Some(p) = px else { continue };
        let v = |c: usize| blend_if_level(p[c]);
        match mode {
            // Gray is the colour channels' luma (Rec. 601 weights), then R, G, B.
            M::Rgb => {
                let gray = 0.299 * v(0) + 0.587 * v(1) + 0.114 * v(2);
                k *= bi.get(0)[side].weight(gray);
                for c in 0..3 {
                    k *= bi.get(c + 1)[side].weight(v(c));
                }
            }
            // One channel: the PSD spec marks the composite-gray entry irrelevant here, so the
            // channel's own entry carries the setting; both are honoured.
            _ => k *= bi.get(0)[side].weight(v(0)) * bi.get(1)[side].weight(v(0)),
        }
        if k <= 0.0 {
            return 0.0;
        }
    }
    k
}

/// A channel value on Blend If's 0..=255 scale. Compositing can leave an 8- or 16-bit level a few
/// millionths off (a level-20 backdrop composited at partial alpha reads 19.999998), which would
/// put it on the wrong side of a hard point at that level: a value that close to a level is that
/// level. Values of those depths that aren't on a level are at least 1/257 away from one. The GPU
/// compositor (`compose.wgsl` `blend_if_weight`) snaps alike.
fn blend_if_level(x: f32) -> f32 {
    let v = x.clamp(0.0, 1.0) * 255.0;
    let r = v.round();
    if (v - r).abs() < 1e-4 { r } else { v }
}

/// Apply `layer`'s Blend If to a finished composite: `out` (the backdrop with the layer drawn)
/// is mixed back towards `before` (the backdrop without it) where the ranges hide the layer.
/// Mixing premultiplied colour by `k` equals compositing the layer at `k` × its alpha, since
/// every blend mode's source-over result is linear in the source alpha.
fn apply_blend_if(layer: &Layer, before: &Buffer, out: &mut Buffer, cx: &Ctx) {
    // "This Layer" is the layer's own colour; adjustment layers (no content of their own) are
    // judged by their result.
    let own = if matches!(layer.content, LayerContent::Adjustment(_)) { None } else { render_content(layer, out.rect, cx) };
    for (i, (p, b)) in out.px.iter_mut().zip(&before.px).enumerate() {
        let this = match &own {
            Some(o) => Some(o.px[i]).filter(|q| q[3] > 0.0),
            None => Some(*p),
        };
        let under = Some(*b).filter(|q| q[3] > 0.0);
        let k = blend_if_weight(layer, cx.mode, this, under);
        if k >= 1.0 {
            continue;
        }
        let a = b[3] + (p[3] - b[3]) * k;
        *p = if a > 0.0 {
            let c = |c: usize| ((b[c] * b[3] + (p[c] * p[3] - b[c] * b[3]) * k) / a).clamp(0.0, 1.0);
            [c(0), c(1), c(2), a]
        } else {
            [0.0; 4]
        };
    }
}

/// Composite `layer` (plus its clipping group) onto `backdrop`, honouring its channel restrictions
/// and Blend If, and knocking out to `ko` when it has a knockout (see [`advanced`]).
fn composite_layer(layer: &Layer, clipped: &[Layer], backdrop: &mut Buffer, cx: &Ctx, ko: Option<&Buffer>, scope: advanced::Scope) {
    // Blend Clipped Layers as Group off: the base alone, then each clipped layer in its own mode.
    if advanced::clips_individually(layer, clipped) {
        advanced::composite_clipped_individually(layer, clipped, backdrop, cx, ko, scope);
        return;
    }
    let w = channel_weights(layer, cx.mode);
    let blend_if = blend_if_active(layer, cx.mode);
    let before = (w.is_some() || blend_if).then(|| backdrop.clone());
    match ko {
        Some(target) => advanced::knockout(layer, backdrop, target, false, cx, |b| composite_layer_any(layer, clipped, b, cx, scope)),
        None => composite_layer_any(layer, clipped, backdrop, cx, scope),
    }
    if let Some(before) = &before {
        if let Some(w) = w {
            restore_channels(backdrop, before, w);
        }
        if blend_if {
            apply_blend_if(layer, before, backdrop, cx);
        }
    }
}

fn composite_layer_any(layer: &Layer, clipped: &[Layer], backdrop: &mut Buffer, cx: &Ctx, scope: advanced::Scope) {
    if let LayerContent::Group(g) = &layer.content
        && let Some(ab) = &g.artboard
    {
        composite_artboard(layer, ab, clipped, backdrop, cx, scope);
        return;
    }
    composite_layer_plain(layer, clipped, backdrop, cx, scope);
}

/// An artboard: its background and the group, composited only inside the board (contents and
/// effects outside it are clipped away; the backdrop there is untouched).
fn composite_artboard(layer: &Layer, ab: &photocraft_doc::Artboard, clipped: &[Layer], backdrop: &mut Buffer, cx: &Ctx, scope: advanced::Scope) {
    let board = ab.rect.intersect(&backdrop.rect);
    if board.is_empty() {
        return;
    }
    let (bw, w) = (board.width() as usize, backdrop.rect.width() as usize);
    let row0 = |y: i32| (y - backdrop.rect.y0) as usize * w + (board.x0 - backdrop.rect.x0) as usize;
    let mut sub = Buffer::transparent(board);
    for y in board.y0..board.y1 {
        let o = row0(y);
        let so = (y - board.y0) as usize * bw;
        sub.px[so..so + bw].copy_from_slice(&backdrop.px[o..o + bw]);
    }
    if let Some(bg) = ab.background.rgba() {
        blend_into(&mut sub, &Buffer::filled(board, bg), BlendMode::Normal, 1.0);
    }
    composite_layer_plain(layer, clipped, &mut sub, cx, scope);
    for y in board.y0..board.y1 {
        let o = row0(y);
        let so = (y - board.y0) as usize * bw;
        backdrop.px[o..o + bw].copy_from_slice(&sub.px[so..so + bw]);
    }
}

fn composite_layer_plain(layer: &Layer, clipped: &[Layer], backdrop: &mut Buffer, cx: &Ctx, scope: advanced::Scope) {
    let rect = backdrop.rect;
    // A TSL-off stroke base that is transparent here can still have visible clipped siblings;
    // keep the full clipping path for it instead of the stroke-bounds shortcut. Other bases keep
    // the usual skip (their clipped layers vanish with them).
    if (clipped.is_empty() || shapeless_stroke_bounds(layer).is_none()) && empty_in(layer, rect) {
        return;
    }
    let opacity = layer.opacity * layer.fill_opacity;

    // Pass-through groups composite their children straight into the backdrop. Below 100% fill
    // Photoshop renders the group isolated, like Normal (psd-tools passthrough_fill_*: an
    // adjustment inside no longer reaches the layers beneath).
    if let LayerContent::Group(g) = &layer.content
        && layer.blend == BlendMode::PassThrough
        && layer.fill_opacity >= 1.0
        && !effects::has_effects(layer)
    {
        let needs_mix = opacity < 1.0 || layer.mask.is_some() || layer.vector_mask.is_some();
        let has_clipped = clipped.iter().any(|c| c.visible);
        // Children can draw directly unless mixing or clipping needs the original backdrop.
        let before = (needs_mix || has_clipped).then(|| backdrop.clone());
        composite_stack_at(&g.children, backdrop, cx, scope.pass_through());
        if needs_mix && let Some(before) = &before {
            let mv = mask_vals(layer, rect, cx);
            for (i, (p, a)) in backdrop.px.iter_mut().zip(&before.px).enumerate() {
                let k = opacity * mask_k(&mv, i);
                let b = *p;
                // Coverage mixes premultiplied colour; the stored buffer remains straight alpha.
                let wa = a[3] * (1.0 - k);
                let wb = b[3] * k;
                let alpha = wa + wb;
                *p = if alpha > 0.0 {
                    let c = |c: usize| (a[c] * wa + b[c] * wb) / alpha;
                    [c(0), c(1), c(2), alpha]
                } else {
                    [0.0; 4]
                };
            }
        }
        // Layers clipped to a pass-through group sit atop the group's
        // isolated rendering; their effect (isolated result with vs. without
        // them, each placed over the original backdrop) is added to the
        // pass-through result. Exact when the children blend Normal, close
        // otherwise (matches psd-tools clipping-mask3/4/5).
        if has_clipped && let (Some(before), Some(iso)) = (before, render_content(layer, rect, cx)) {
            let mut clipped_iso = iso.clone();
            advanced::composite_clipped(clipped, &mut clipped_iso, cx);
            let mut without = before.clone();
            blend_into(&mut without, &iso, BlendMode::Normal, opacity);
            let mut with = before;
            blend_into(&mut with, &clipped_iso, BlendMode::Normal, opacity);
            for ((p, w), wo) in backdrop.px.iter_mut().zip(&with.px).zip(&without.px) {
                // Work premultiplied so transparent areas stay consistent.
                let pa = p[3];
                let mut pm = [p[0] * pa, p[1] * pa, p[2] * pa, pa];
                for c in 0..3 {
                    pm[c] += w[c] * w[3] - wo[c] * wo[3];
                }
                pm[3] += w[3] - wo[3];
                let a = pm[3].clamp(0.0, 1.0);
                *p = if a > 0.0 { [(pm[0] / a).clamp(0.0, 1.0), (pm[1] / a).clamp(0.0, 1.0), (pm[2] / a).clamp(0.0, 1.0), a] } else { [0.0; 4] };
            }
        }
        return;
    }

    // Adjustment layers transform the backdrop, then blend the result back in.
    if let LayerContent::Adjustment(adj) = &layer.content {
        let mut adjusted = backdrop.clone();
        adjust::apply_in(adj, &mut adjusted, cx.transfer, Some(cx.depth), cx.mode);
        // Clipped layers onto an adjustment are uncommon; they composite atop the adjusted result.
        advanced::composite_clipped(clipped, &mut adjusted, cx);
        let mv = mask_vals(layer, rect, cx);
        for y in rect.y0..rect.y1 {
            for x in rect.x0..rect.x1 {
                let i = ((y - rect.y0) as usize) * rect.width() as usize + (x - rect.x0) as usize;
                let k = opacity * mask_k(&mv, i);
                if k <= 0.0 {
                    continue;
                }
                // Only `adjusted` has changed; this pixel's original backdrop is still in place.
                let b = backdrop.px[i];
                let a = adjusted.px[i];
                let blended = blend::blend_rgb(layer.blend, [b[0], b[1], b[2]], [a[0], a[1], a[2]]);
                backdrop.px[i] = [b[0] + (blended[0] - b[0]) * k, b[1] + (blended[1] - b[1]) * k, b[2] + (blended[2] - b[2]) * k, b[3]];
            }
        }
        return;
    }

    if effects::has_effects(layer) {
        // Layer / Vector Mask Hides Effects: the effects are built without that mask, which then
        // hides the finished layer and effects.
        let src = advanced::effects_source(layer);
        let hide = advanced::hiding_mask(layer, rect, cx);
        let before = hide.as_ref().map(|_| backdrop.clone());
        composite_layer_effects(&src, clipped, backdrop, cx);
        if let (Some(m), Some(b)) = (&hide, &before) {
            advanced::apply_hiding_mask(backdrop, b, m);
        }
        return;
    }
    if let Some((mut content, stroke, mask)) = shape_parts(layer, clipped, rect, cx) {
        advanced::composite_clipped(clipped, &mut content, cx);
        for (i, (p, s)) in content.px.iter_mut().zip(&stroke.px).enumerate() {
            *p = psblend::composite(BlendMode::Normal, *p, *s, 1.0);
            p[3] *= mask_k(&mask, i);
        }
        blend_into_fill(backdrop, &content, layer.blend, layer.opacity, layer.fill_opacity, 1.0);
        return;
    }
    let Some(mut content) = render_content(layer, rect, cx) else { return };
    advanced::composite_clipped(clipped, &mut content, cx);
    blend_into_fill(backdrop, &content, layer.blend, layer.opacity, layer.fill_opacity, text_gamma(layer));
}

/// A layer with effects (and its clipping group) onto `backdrop`.
fn composite_layer_effects(layer: &Layer, clipped: &[Layer], backdrop: &mut Buffer, cx: &Ctx) {
    let rect = backdrop.rect;
    // Neighbourhoods are already captured by the full-region effect maps;
    // content and effect application only need the output rectangle.
    // A stroked shape's vector stroke stays above its clipped layers and interior effects.
    let (mut content, vstroke) = match split_parts(layer, rect, cx) {
        Some((fill, stroke, mask)) => (fill, Some((stroke, mask))),
        None => {
            let Some(content) = render_content(layer, rect, cx) else { return };
            (content, None)
        }
    };
    advanced::composite_clipped(clipped, &mut content, cx);
    let vstroke = vstroke.as_ref().map(|(s, m)| effects::VectorStroke { stroke: s, mask: m.as_deref() });
    let maps = effect_maps(layer, rect, cx);
    effects::composite_with_effects_prepared(
        layer,
        &content,
        backdrop,
        &maps,
        paint_bounds(layer).unwrap_or_else(|| layer_bounds(layer, cx.canvas)),
        cx.patterns,
        vstroke,
    );
}

/// The coverage-mixing gamma of a layer: the text blending gamma
/// ([`psblend::set_text_gamma`], Photoshop's default 1.45) for type layers, else 1.
pub fn text_gamma(layer: &Layer) -> f32 {
    if matches!(layer.content, LayerContent::Text(_)) { psblend::text_gamma() } else { 1.0 }
}

/// A stroked shape layer with visible clipped layers: Photoshop draws the shape's vector stroke
/// above the clipped layers, so the base content is the fill alone and the stroke is laid on top
/// after clipping. See [`split_parts`].
fn shape_parts(layer: &Layer, clipped: &[Layer], rect: Rect, cx: &Ctx) -> Option<(Buffer, Buffer, Option<Vec<f32>>)> {
    if !clipped.iter().any(|c| c.visible) {
        return None;
    }
    split_parts(layer, rect, cx)
}

/// A stroked shape layer's (fill, vector stroke, mask values) over `rect`; `None` for other
/// layers. Fill and stroke are unmasked: the mask applies to them together. Layer effects paint
/// the fill's interior effects beneath the vector stroke (psd-tools stroke-composite: a colour
/// overlay leaves the stroke, the stroke effect covers it).
fn split_parts(layer: &Layer, rect: Rect, cx: &Ctx) -> Option<(Buffer, Buffer, Option<Vec<f32>>)> {
    let LayerContent::Shape(sh) = &layer.content else { return None };
    sh.stroke.as_ref()?;
    let (fs, ss) = shape_split::split(sh, cx.canvas, cx.depth)?;
    Some((surface_to_buffer(&fs, rect), surface_to_buffer(&ss, rect), mask_vals(layer, rect, cx)))
}

// ---------------------------------------------------------------------------------------------
// Layer-effect map cache

/// Order-independent identity of a layer's pixels, masks, effects, Advanced Blending and Blend If
/// and (for groups) children.
/// Pixels are identified by their copy-on-write tile pointers; cache entries pin a clone of the
/// layer so those tiles (and their addresses) stay alive while the entry exists.
fn layer_identity(layer: &Layer, h: &mut std::collections::hash_map::DefaultHasher) {
    use std::hash::{Hash, Hasher};
    fn surface_fp(s: &Surface) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        s.format().hash(&mut h);
        // Missing tiles read the default pixel, so tile identity alone is not pixel identity.
        for sample in s.default_pixel() {
            sample.to_bits().hash(&mut h);
        }
        let tiles = s.tiles().fold(s.tile_count() as u64, |acc, (c, t)| {
            let mut x = (std::sync::Arc::as_ptr(t) as usize as u64) ^ ((c.tx as u64) << 40) ^ ((c.ty as u32 as u64) << 8);
            x = x.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            acc.wrapping_add(x ^ (x >> 29))
        });
        tiles.hash(&mut h);
        h.finish()
    }
    layer.id.0.hash(h);
    // What the maps depend on: `fill_opacity` shades them; layer `opacity` is applied when the
    // maps are composited and `visible` only skips the lookup entirely, so neither belongs in
    // the key (an opacity drag must not rebuild shadows over the whole layer; the GPU's
    // `fx::shape_key` keeps them out for the same reason).
    layer.fill_opacity.to_bits().hash(h);
    match &layer.content {
        LayerContent::Group(g) => {
            for c in &g.children {
                // A styled group's maps come from its children's composite, so how each child
                // composites (unlike the group's own visibility and opacity) shapes them.
                (c.visible, c.opacity.to_bits(), c.clipped).hash(h);
                format!("{:?}", c.blend).hash(h);
                layer_identity(c, h);
            }
            format!("{:?}", g.artboard).hash(h);
        }
        LayerContent::Fill(f) => {
            format!("{f:?}").hash(h);
            // The cached pixels are what shapes the maps while they match the fill: a
            // pixel-only rewrite of the cache (Image › Transform, crop) leaves `f` alone, and
            // the stale maps would keep shadowing the pre-rewrite shape.
            if let Some(c) = &layer.fill_cache
                && c.fill == *f
            {
                surface_fp(&c.surface).hash(h);
            }
        }
        LayerContent::Adjustment(a) => format!("{a:?}").hash(h),
        _ => layer.surface().map_or(0, surface_fp).hash(h),
    }
    if let Some(m) = &layer.mask {
        (surface_fp(&m.surface), m.enabled, m.density.to_bits(), m.feather.to_bits()).hash(h);
    }
    if let Some(vm) = &layer.vector_mask {
        format!("{vm:?}").hash(h);
    }
    format!("{:?}", layer.effects).hash(h);
    // Transparency Shapes Layer picks the shape (content or the whole layer); a group child's
    // Blend If and knockout change the composite a styled group's shape comes from.
    layer.advanced.hash(h);
    layer.blend_if.hash(h);
    h.write_u8(0xfe);
}

struct FxEntry {
    maps: std::sync::Arc<effects::FxMaps>,
    _pin: Layer,
    bytes: usize,
}

/// Global cache of effect maps (bounded by bytes), one slot per layer state. Slots are filled
/// with `set`, never `get_or_init`: nothing may block on a build (see `cached_effect_maps`).
type FxSlot = std::sync::Arc<std::sync::OnceLock<FxEntry>>;
struct FxCache {
    map: std::collections::HashMap<u64, FxSlot>,
    order: std::collections::VecDeque<u64>,
    bytes: usize,
}

/// Default effect-cache budget (Preferences › Performance can change it).
pub const DEFAULT_FX_CACHE_BUDGET: usize = 768 << 20;
static FX_CACHE_BUDGET: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(DEFAULT_FX_CACHE_BUDGET);

/// Set the memory budget (bytes) of the layer-effect map cache; entries over it are evicted
/// oldest-first on the next build.
pub fn set_effect_cache_budget(bytes: usize) {
    FX_CACHE_BUDGET.store(bytes.max(1 << 20), std::sync::atomic::Ordering::Relaxed);
}

/// The current effect-cache budget in bytes.
pub fn effect_cache_budget() -> usize {
    FX_CACHE_BUDGET.load(std::sync::atomic::Ordering::Relaxed)
}

/// Bytes currently held by the effect-map cache.
pub fn effect_cache_bytes() -> usize {
    fx_cache().lock().unwrap_or_else(|e| e.into_inner()).bytes
}

/// Drop every cached effect map (Edit › Purge › All). Returns the bytes released.
pub fn purge_effect_cache() -> usize {
    let mut c = fx_cache().lock().unwrap_or_else(|e| e.into_inner());
    let freed = c.bytes;
    c.map.clear();
    c.order.clear();
    c.bytes = 0;
    freed
}

fn fx_cache() -> &'static std::sync::Mutex<FxCache> {
    static C: std::sync::OnceLock<std::sync::Mutex<FxCache>> = std::sync::OnceLock::new();
    C.get_or_init(|| std::sync::Mutex::new(FxCache { map: Default::default(), order: Default::default(), bytes: 0 }))
}

/// Where `layer`'s effect maps are built (see [`effect_maps`]) and whether that is clipped to the
/// rectangle being rendered rather than the whole region.
///
/// Transparency Shapes Layer off makes the region the whole canvas whatever the layer holds: on a
/// 1920 × 103675 webtoon page each such layer's maps took about 2.4 GB, and a render held them for
/// every layer at once (60 GB with 25 stroked type layers, #1909). When its effects are local
/// ([`effects::maps_are_local`]), such a layer's maps are built over the render rectangle padded
/// by twice the effect reach instead, which gives the same pixels there. Only when `rect`, what
/// the caller composites the layer over, lies within the render rectangle: a group's effect shape
/// (or anything else rendering children over a wider area) gets the whole region.
fn effect_region(layer: &Layer, rect: Rect, cx: &Ctx) -> (Rect, bool) {
    let m = effects::margin(layer);
    let region = effect_source_bounds(layer, cx.canvas).inflate(m).intersect(&cx.canvas.inflate(m));
    let clip = *cx.fx_clip.lock().unwrap_or_else(|e| e.into_inner());
    match clip {
        Some(c) if c.contains_rect(&rect) && advanced::shapeless(layer) && effects::maps_are_local(layer) => {
            let part = region.intersect(&c.inflate(m.saturating_mul(2)));
            (part, part != region)
        }
        _ => (region, false),
    }
}

/// The layer's effect maps over its region (layer bounds grown by the effect reach, within the
/// canvas grown likewise; see [`effect_region`]), built once per layer state and shared by every
/// tile. `rect` is what the caller composites the layer over (see [`effect_region`]).
fn effect_maps(layer: &Layer, rect: Rect, cx: &Ctx) -> std::sync::Arc<effects::FxMaps> {
    use std::hash::{Hash, Hasher};
    let (region, clipped) = effect_region(layer, rect, cx);
    static NO_CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *NO_CACHE.get_or_init(|| std::env::var_os("PHOTOCRAFT_FX_NOCACHE").is_some()) {
        let shape = if region.is_empty() { Vec::new() } else { effect_shape(layer, region, cx) };
        return std::sync::Arc::new(effects::build_maps_prepared(layer, shape, region, &cx.light, &texture_ctx(layer, region, cx), cx.patterns));
    }
    let mut h = std::collections::hash_map::DefaultHasher::new();
    layer_identity(layer, &mut h);
    (region.x0, region.y0, region.x1, region.y1).hash(&mut h);
    (cx.light.angle.to_bits(), cx.light.altitude.to_bits()).hash(&mut h);
    let key = h.finish();
    // Prepared for this render (see `prepare_effects`): no global lookup, no build.
    if let Some((m, _)) = cx.fx_maps.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return m.clone();
    }
    let maps = cached_effect_maps(layer, region, key, cx);
    cx.fx_maps.lock().unwrap_or_else(|e| e.into_inner()).entry(key).or_insert((maps, clipped)).0.clone()
}

/// The layer's effect maps from the global cache, built on a miss.
///
/// Never waits for another thread's build (#276): this can run on a Rayon worker inside a tile,
/// and the build itself runs Rayon work. A worker waiting on that work may steal another tile of
/// the same layer; had it blocked on a once-init held further up its own stack, it would deadlock.
/// Instead a miss builds the maps here and the first finished build is kept. Renders build every
/// layer's maps before their parallel tiles (`prepare_effects`), so tiles rarely miss.
fn cached_effect_maps(layer: &Layer, region: Rect, key: u64, cx: &Ctx) -> std::sync::Arc<effects::FxMaps> {
    let slot = {
        let mut c = fx_cache().lock().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = c.map.get(&key) {
            s.clone()
        } else {
            let s: FxSlot = Default::default();
            c.map.insert(key, s.clone());
            c.order.push_back(key);
            s
        }
    };
    let maps = match slot.get() {
        Some(e) => e.maps.clone(),
        None => {
            let shape: Vec<f32> = if region.is_empty() { Vec::new() } else { effect_shape(layer, region, cx) };
            let maps = effects::build_maps_prepared(layer, shape, region, &cx.light, &texture_ctx(layer, region, cx), cx.patterns);
            let bytes = maps.bytes();
            let maps = std::sync::Arc::new(maps);
            if slot.set(FxEntry { maps: maps.clone(), _pin: layer.clone(), bytes }).is_ok() {
                // Counted exactly once, by the build that filled the slot, and only while the
                // slot is still cached (eviction may have dropped it meanwhile).
                let mut c = fx_cache().lock().unwrap_or_else(|e| e.into_inner());
                if c.map.get(&key).is_some_and(|s| std::sync::Arc::ptr_eq(s, &slot)) {
                    c.bytes += bytes;
                }
            }
            // A concurrent build that finished first wins, so every tile shares one map.
            slot.get().map_or(maps, |e| e.maps.clone())
        }
    };
    // Evict the oldest entries over budget (never the one just used).
    let mut c = fx_cache().lock().unwrap_or_else(|e| e.into_inner());
    let budget = effect_cache_budget();
    while c.bytes > budget && c.order.len() > 1 {
        let Some(old) = c.order.pop_front() else { break };
        if old == key {
            c.order.push_back(old);
            continue;
        }
        if let Some(s) = c.map.remove(&old) {
            c.bytes = c.bytes.saturating_sub(s.get().map_or(0, |e| e.bytes));
        }
    }
    maps
}

fn texture_ctx<'a>(layer: &Layer, region: Rect, cx: &Ctx<'a>) -> effects::TextureCtx<'a> {
    let sb = paint_bounds(layer).unwrap_or_else(|| layer_bounds(layer, cx.canvas));
    effects::TextureCtx { rect: region, patterns: cx.patterns.source(), anchor: layer.effects.reference.unwrap_or((f64::from(sb.x0), f64::from(sb.y0))) }
}

/// Composite `layer` onto `base` restricted to the base's alpha (clipping mask semantics),
/// honouring its channel restrictions and Blend If.
fn composite_atop(layer: &Layer, base: &mut Buffer, cx: &Ctx) {
    let w = channel_weights(layer, cx.mode);
    let blend_if = blend_if_active(layer, cx.mode);
    let before = (w.is_some() || blend_if).then(|| base.clone());
    composite_atop_any(layer, base, cx);
    if let Some(before) = &before {
        if let Some(w) = w {
            restore_channels(base, before, w);
        }
        if blend_if {
            apply_blend_if(layer, before, base, cx);
        }
    }
}

fn composite_atop_any(layer: &Layer, base: &mut Buffer, cx: &Ctx) {
    let rect = base.rect;
    if let LayerContent::Adjustment(adj) = &layer.content {
        let mut adjusted = base.clone();
        adjust::apply_in(adj, &mut adjusted, cx.transfer, Some(cx.depth), cx.mode);
        let mv = mask_vals(layer, rect, cx);
        for (i, p) in base.px.iter_mut().enumerate() {
            let k = layer.opacity * layer.fill_opacity * mask_k(&mv, i);
            let a = adjusted.px[i];
            let bl = blend::blend_rgb(layer.blend, [p[0], p[1], p[2]], [a[0], a[1], a[2]]);
            for c in 0..3 {
                p[c] += (bl[c] - p[c]) * k;
            }
        }
        return;
    }
    if effects::has_effects(layer) {
        // Effects of a clipped layer are clipped to the base too: render
        // them over the base (treated as opaque) and keep the base's alpha.
        // The full-region effect maps already capture their neighbourhoods.
        // A mask that hides the effects applies to the finished result instead.
        let hide = advanced::hiding_mask(layer, rect, cx);
        let src = advanced::effects_source(layer);
        let layer = src.as_ref();
        let (content, vstroke) = match split_parts(layer, rect, cx) {
            Some((fill, stroke, mask)) => (fill, Some((stroke, mask))),
            None => {
                let Some(content) = render_content(layer, rect, cx) else { return };
                (content, None)
            }
        };
        let vstroke = vstroke.as_ref().map(|(s, m)| effects::VectorStroke { stroke: s, mask: m.as_deref() });
        let mut opaque = Buffer { rect, px: base.px.iter().map(|p| [p[0], p[1], p[2], 1.0]).collect() };
        let maps = effect_maps(layer, rect, cx);
        effects::composite_with_effects_prepared(
            layer,
            &content,
            &mut opaque,
            &maps,
            paint_bounds(layer).unwrap_or_else(|| layer_bounds(layer, cx.canvas)),
            cx.patterns,
            vstroke,
        );
        if let Some(m) = &hide {
            let plain = Buffer { rect, px: base.px.iter().map(|p| [p[0], p[1], p[2], 1.0]).collect() };
            advanced::apply_hiding_mask(&mut opaque, &plain, m);
        }
        for (p, o) in base.px.iter_mut().zip(&opaque.px) {
            if p[3] > 0.0 {
                *p = [o[0], o[1], o[2], p[3]];
            }
        }
        return;
    }
    let Some(content) = render_content(layer, rect, cx) else { return };
    let opacity = layer.opacity * layer.fill_opacity;
    let gamma = text_gamma(layer);
    for (i, p) in base.px.iter_mut().enumerate() {
        let alpha = p[3];
        if alpha <= 0.0 {
            continue;
        }
        let s = content.px[i];
        // Blend as if the base were opaque, then keep the base's alpha.
        let r = blend::composite_gamma(layer.blend, [p[0], p[1], p[2], 1.0], s, opacity, gamma);
        *p = [r[0], r[1], r[2], alpha];
    }
}

/// Blend an isolated layer buffer into the backdrop.
fn blend_into(backdrop: &mut Buffer, src: &Buffer, mode: BlendMode, opacity: f32) {
    blend_into_g(backdrop, src, mode, opacity, 1.0);
}

/// [`blend_into_g`] for a layer's own `opacity` and `fill`: Fill is part of the blend for
/// Photoshop's special eight ([`psblend::composite_fill`]), plain coverage otherwise.
fn blend_into_fill(backdrop: &mut Buffer, src: &Buffer, mode: BlendMode, opacity: f32, fill: f32, gamma: f32) {
    if fill >= 1.0 || !psblend::fill_is_special(mode) {
        return blend_into_g(backdrop, src, mode, opacity * fill, gamma);
    }
    for (b, s) in backdrop.px.iter_mut().zip(&src.px) {
        if s[3] > 0.0 {
            *b = psblend::composite_fill(mode, *b, *s, opacity, fill, gamma);
        }
    }
}

/// [`blend_into`] mixing coverage in a `gamma` space (type layers).
fn blend_into_g(backdrop: &mut Buffer, src: &Buffer, mode: BlendMode, opacity: f32, gamma: f32) {
    // The blend space is constant in this loop. Some TLS implementations synchronise even a
    // boolean read, so select the common Normal kernel once instead of once per pixel.
    if mode == BlendMode::Normal && gamma == 1.0 && !psblend::LAB_MIX.with(|l| l.get()) {
        for (i, b) in backdrop.px.iter_mut().enumerate() {
            let s = src.px[i];
            if s[3] <= 0.0 {
                continue;
            }
            *b = psblend::composite_normal_rgb(*b, s, opacity);
        }
        return;
    }
    let rect = backdrop.rect;
    let w = rect.width() as i32;
    for (i, b) in backdrop.px.iter_mut().enumerate() {
        let mut s = src.px[i];
        if s[3] <= 0.0 {
            continue;
        }
        let mut mode = mode;
        if mode == BlendMode::Dissolve {
            let x = rect.x0 + (i as i32 % w);
            let y = rect.y0 + (i as i32 / w);
            s[3] = if dissolve_noise(x, y) < s[3] * opacity { 1.0 } else { 0.0 };
            mode = BlendMode::Normal;
            *b = blend::composite(mode, *b, s, 1.0);
            continue;
        }
        *b = blend::composite_gamma(mode, *b, s, opacity, gamma);
    }
}

#[cfg(test)]
mod stroke_tests;
#[cfg(test)]
mod tests;

#[cfg(test)]
mod advanced_tests;
