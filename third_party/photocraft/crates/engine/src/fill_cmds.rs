//! Edit › Fill… (`edit.fill`): Photoshop's Fill dialog as one command.
//!
//! Contents: Foreground / Background / Color / Content-Aware / Pattern / History / Black /
//! 50% Gray / White, blended with `mode` at `opacity`, optionally preserving the target's
//! transparency. The selection (or the whole canvas) is filled, as one history step ("Fill").
//! Every parameter is validated: a bad one is an error, never a panic.

use photocraft_color::BlendMode;
use photocraft_geom::{Rect, TILE_SIZE};
use photocraft_raster::{Surface, from_rgba_into, to_rgba};
use serde_json::{Value, json};

use crate::commands::{blend_from_str, color_param, layer_param};
use crate::{EngineError, Result, Session};

const CMD: &str = "edit.fill";

/// Every `contents` value, in the dialog's order.
pub const CONTENTS: [&str; 9] = ["foreground", "background", "color", "contentAware", "pattern", "history", "black", "gray", "white"];

fn bad(msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: CMD.into(), msg: msg.into() }
}

/// What a pixel of the filled area gets before blending.
enum Source {
    Color([f32; 4]),
    /// An opaque colour that was picked as CMYK inks: `rgba` is its sRGB stand-in (what other
    /// colour models and blend modes use), `cmyk` the inks a CMYK target gets as they are.
    Cmyk {
        rgba: [f32; 4],
        cmyk: [f32; 4],
    },
    /// Straight RGBA per pixel of `rect` (row-major).
    Pixels(Rect, Vec<[f32; 4]>),
}

impl Source {
    fn at(&self, x: i32, y: i32) -> [f32; 4] {
        match self {
            Source::Color(c) => *c,
            Source::Cmyk { rgba, .. } => *rgba,
            Source::Pixels(r, px) => {
                if !r.contains(x, y) {
                    return [0.0; 4];
                }
                let i = (y - r.y0) as usize * r.width() as usize + (x - r.x0) as usize;
                px.get(i).copied().unwrap_or([0.0; 4])
            }
        }
    }
}

/// The `contents` param; without one, `color` (the old `{"color": …}` form, foreground by default).
fn contents(p: &Value) -> Result<&str> {
    match p.get("contents") {
        None | Some(Value::Null) => Ok("color"),
        Some(Value::String(c)) => {
            CONTENTS.iter().find(|k| k.eq_ignore_ascii_case(c)).copied().ok_or_else(|| bad(format!("unknown contents `{c}` ({})", CONTENTS.join("|"))))
        }
        Some(v) => Err(bad(format!("`contents` must be a string, not {v}"))),
    }
}

fn mode(p: &Value) -> Result<BlendMode> {
    match p.get("mode") {
        None | Some(Value::Null) => Ok(BlendMode::Normal),
        Some(Value::String(m)) => blend_from_str(m).filter(|m| *m != BlendMode::PassThrough).ok_or_else(|| bad(format!("unknown blend mode `{m}`"))),
        Some(v) => Err(bad(format!("`mode` must be a blend mode name, not {v}"))),
    }
}

/// Opacity in percent (0–100) → 0..1.
fn opacity(p: &Value) -> Result<f32> {
    match p.get("opacity") {
        None | Some(Value::Null) => Ok(1.0),
        Some(v) => {
            let o = v.as_f64().filter(|o| o.is_finite()).ok_or_else(|| bad(format!("`opacity` must be a number 0..100, not {v}")))?;
            Ok((o.clamp(0.0, 100.0) / 100.0) as f32)
        }
    }
}

fn flag(p: &Value, k: &str, d: bool) -> Result<bool> {
    match p.get(k) {
        None | Some(Value::Null) => Ok(d),
        Some(Value::Bool(b)) => Ok(*b),
        Some(v) => Err(bad(format!("`{k}` must be true or false, not {v}"))),
    }
}

/// Edit › Fill.
pub fn fill(s: &mut Session, p: &Value) -> Result<Value> {
    let contents = contents(p)?;
    let mode = mode(p)?;
    let opacity = opacity(p)?;
    let preserve = flag(p, "preserveTransparency", false)?;
    let channel = crate::channel_cmds::is_channel_target(p);
    let id = if channel { None } else { Some(layer_param(s, p)?) };
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let doc_bounds = st.doc.bounds();
    let sel = st.doc.selection.clone();
    let area = sel.as_ref().map_or(doc_bounds, |m| m.content_bounds().intersect(&doc_bounds));
    let mut out = json!({ "contents": contents });
    let restore = contents == "history";
    let source = match contents {
        "foreground" => color_source(s, s.tools.foreground),
        "background" => color_source(s, s.tools.background),
        "color" => color_source(s, color_param(p, "color", s.tools.foreground)),
        "black" => Source::Color([0.0, 0.0, 0.0, 1.0]),
        "gray" => Source::Color([128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0, 1.0]),
        "white" => Source::Color([1.0; 4]),
        "pattern" => {
            let pat = crate::pattern_cmds::resolve_param(s, CMD, p)?;
            let (scale, angle, _, phase) = crate::pattern_cmds::placement(p);
            let tile = photocraft_compose::pattern::Tile::new(&pat).ok_or_else(|| bad("the pattern is empty or too large to paint"))?;
            // Photoshop's Fill uses the canvas origin.
            let place = photocraft_compose::pattern::Placement::new(Rect::EMPTY, false, phase, scale, angle);
            out["pattern"] = json!(pat.id);
            Source::Pixels(area, photocraft_compose::pattern::render(&tile, &place, area))
        }
        "history" => {
            let explicit = match p.get("state") {
                None | Some(Value::Null) => None,
                Some(v) => Some(v.as_u64().ok_or_else(|| bad(format!("`state` must be a history state index, not {v}")))? as usize),
            };
            let past = st.history.past_len();
            let source_doc = match explicit {
                Some(i) if i == past => st.doc.clone(),
                Some(i) => st.history.state(i).ok_or_else(|| bad(format!("no history state {i} (0..={past})")))?,
                // The History Brush's default source: the oldest state still held.
                None => st.history.state(0).unwrap_or_else(|| st.doc.clone()),
            };
            let src = crate::channel_cmds::read_target_surface(source_doc.as_ref(), id, p)
                .map_err(|_| EngineError::Other("the target did not exist (as pixels, a mask or a channel) in that history state".into()))?;
            Source::Pixels(area, read_rgba(src, area, "reading Fill's history source")?)
        }
        "contentAware" => {
            let sel = sel.as_ref().ok_or_else(|| EngineError::Other("Content-Aware fill needs a selection".into()))?;
            if channel || crate::commands::is_mask_target(p) {
                return Err(EngineError::Other("Content-Aware fill works on layer pixels only".into()));
            }
            let adapt = flag(p, "colorAdaptation", true)?;
            let lid = id.ok_or_else(|| EngineError::Other("no active layer".into()))?;
            let surf = st.doc.layer(lid).and_then(|l| l.surface()).ok_or_else(|| EngineError::Other("Content-Aware fill needs a pixel layer".into()))?;
            let seed = p.get("seed").and_then(Value::as_u64).unwrap_or(1);
            content_aware(surf, sel, doc_bounds, adapt, seed)?
        }
        _ => return Err(bad(format!("unknown contents `{contents}`"))),
    };
    if area.is_empty() {
        return Ok(out);
    }
    // The document's own CMYK profile converts the fill colour, like the compositor and brush.
    let cmyk = crate::brush_cmds::doc_cmyk_space(s);
    photocraft_color::convert::with_cmyk_space(cmyk.as_ref(), || {
        s.edit("Fill", |doc, _| {
            let sel = doc.selection.clone();
            let (surf, lock) = crate::channel_cmds::target_surface(doc, id, p)?;
            blend_into(surf, area, &source, sel.as_ref(), Blend { mode, opacity, keep_alpha: preserve || lock, restore })?;
            Ok(())
        })
    })?;
    Ok(out)
}

/// A fill colour: the inks it was picked as when it was picked as CMYK (#4196), else plain sRGB.
fn color_source(s: &Session, c: [f32; 4]) -> Source {
    match s.tools.picked_cmyk([c[0], c[1], c[2]]) {
        Some(cmyk) if c[3] >= 1.0 => Source::Cmyk { rgba: c, cmyk },
        _ => Source::Color(c),
    }
}

/// `source`'s opaque colour as samples of `fmt` (alpha last, as in [`from_rgba_into`]), or `None`
/// for a source that is not one colour. A CMYK pick keeps its inks in a CMYK target.
fn encode_color(fmt: &photocraft_color::PixelFormat, source: &Source, enc: &mut [f32; 8]) -> Option<usize> {
    let (rgba, cmyk) = match source {
        Source::Color(c) if c[3] >= 1.0 => (*c, None),
        Source::Cmyk { rgba, cmyk } => (*rgba, Some(*cmyk)),
        _ => return None,
    };
    let n = from_rgba_into(fmt, rgba, enc);
    if let Some(k) = cmyk
        && fmt.mode == photocraft_color::ColorMode::Cmyk
    {
        enc[..4].copy_from_slice(&k);
    }
    Some(n)
}

fn read_rgba(src: &Surface, area: Rect, purpose: &str) -> Result<Vec<[f32; 4]>> {
    let len = (area.width() as usize)
        .checked_mul(area.height() as usize)
        .ok_or_else(|| EngineError::Other(format!("not enough memory for {purpose} (requested size overflowed); the document was not changed")))?;
    let mut px = crate::allocation::filled(len, [0.0f32; 4], purpose)?;
    src.read_rgba_into(area, &mut px);
    Ok(px)
}

/// `src` over `surf` across `area` (Normal, 100 %), each pixel weighted by `limit` when given:
/// a paste into a layer mask or channel, which keeps the pasted pixels' luminosity (#1035).
pub(crate) fn composite_over(surf: &mut Surface, src: &Surface, area: Rect, limit: Option<&Surface>) -> Result<()> {
    let source = Source::Pixels(area, read_rgba(src, area, "pasting pixels")?);
    blend_into(surf, area, &source, limit, Blend { mode: BlendMode::Normal, opacity: 1.0, keep_alpha: false, restore: false })
}

/// `area` of `surf` filled with an opaque `color` (Normal, 100 %), weighted by `sel` when given:
/// Clear on a targeted alpha channel or the Quick Mask (#2734).
pub(crate) fn fill_color(surf: &mut Surface, area: Rect, color: [f32; 4], sel: Option<&Surface>) -> Result<()> {
    blend_into(surf, area, &Source::Color(color), sel, Blend { mode: BlendMode::Normal, opacity: 1.0, keep_alpha: false, restore: false })
}

/// Reuse Edit › Fill's compositing for Edit › Stroke. The stroke's rasterized band
/// is already a grayscale coverage mask; opacity, blend modes and alpha locking
/// must behave identically for both commands at 8/16/32-bit depth.
#[allow(clippy::too_many_arguments)]
pub(crate) fn blend_color_mask(
    surf: &mut Surface,
    area: Rect,
    color: [f32; 4],
    picked: Option<[f32; 4]>,
    coverage: &Surface,
    mode: BlendMode,
    opacity: f32,
    preserve_transparency: bool,
) -> Result<()> {
    let source = match picked {
        Some(cmyk) if color[3] >= 1.0 => Source::Cmyk { rgba: color, cmyk },
        _ => Source::Color(color),
    };
    blend_into(surf, area, &source, Some(coverage), Blend { mode, opacity, keep_alpha: preserve_transparency, restore: false })
}

/// The selection filled from its surroundings (`photocraft_algo::content_aware`), as straight
/// RGBA over the sampling window. Only the selected pixels are used.
fn content_aware(surf: &Surface, sel: &Surface, canvas: Rect, color_adaptation: bool, seed: u64) -> Result<Source> {
    use photocraft_algo::content_aware::{FillOptions, color_level, fill, rotation_level};
    let hole_bounds = sel.content_bounds().intersect(&canvas);
    if hole_bounds.is_empty() {
        return Err(EngineError::Other("the selection is outside the canvas".into()));
    }
    let window = sampling_window(hole_bounds, canvas);
    let fmt = surf.format();
    let n = fmt.channels();
    let (w, h) = window_size(window)?;
    let img = surf.read_region(window);
    let mut hole = vec![false; w * h];
    for (i, hv) in hole.iter_mut().enumerate() {
        *hv = sel.sample_channel(window.x0 + (i % w) as i32, window.y0 + (i / w) as i32, 0) > 0.0;
    }
    if hole.iter().all(|h| *h) {
        return Err(EngineError::Other("Content-Aware fill needs pixels outside the selection to sample from".into()));
    }
    let source = vec![true; w * h];
    let opts = FillOptions {
        color_adaptation: color_level(if color_adaptation { "default" } else { "none" }),
        rotations: rotation_level("none"),
        scale: false,
        mirror: false,
        seed,
    };
    if img.len() != w * h * n {
        return Err(EngineError::Other("could not read the layer pixels".into()));
    }
    let filled = fill(w, h, n, &img, &hole, &source, &opts);
    let px = filled.chunks_exact(n.max(1)).map(|c| to_rgba(&fmt, c)).collect();
    Ok(Source::Pixels(window, px))
}

/// Photoshop's Content-Aware sampling window around the selection bounds `hole`
/// ([`photocraft_algo::content_aware::sampling_window`]): Edit › Fill, the workspace's Rectangular
/// and Auto areas and the retouching tools all copy from within it. Worked out in i64, so a loaded
/// document declaring an absurd selection can't overflow it (#963).
pub(crate) fn sampling_window(hole: Rect, canvas: Rect) -> Rect {
    photocraft_algo::content_aware::sampling_window(hole.intersect(&canvas), canvas)
}

/// The size of a Content-Aware sampling window, or an error for a window too large to read and
/// fill (#963). Same cap as other whole-image operations: refuse absurd windows instead of
/// allocating.
pub(crate) fn window_size(window: Rect) -> Result<(usize, usize)> {
    let (w, h) = (window.width(), window.height());
    if u64::from(w) * u64::from(h) > 400_000_000 {
        return Err(EngineError::Other("the selection is too large for Content-Aware fill".into()));
    }
    Ok((w as usize, h as usize))
}

/// [`Surface::read_region`] of `r`, read in parallel bands and asking `cancelled` before each
/// (`None` once it says so): a Content-Aware window can be tens of megapixels.
pub(crate) fn read_window(surf: &Surface, r: Rect, cancelled: &(dyn Fn() -> bool + Sync)) -> Option<Vec<f32>> {
    use rayon::prelude::*;
    const BAND: usize = 64;
    let n = surf.channels();
    let (w, h) = (r.width() as usize, r.height() as usize);
    let mut out = vec![0.0f32; w * h * n];
    if out.is_empty() {
        return Some(out);
    }
    // Each worker reads its bands through one buffer of its own.
    out.par_chunks_mut(BAND * w * n).enumerate().for_each_init(Vec::new, |buf, (i, chunk)| {
        if cancelled() {
            return;
        }
        let y0 = r.y0 + (i * BAND) as i32;
        let band = Rect::new(r.x0, y0, r.x1, y0 + (chunk.len() / (w * n)) as i32);
        surf.read_region_into(band, buf);
        if buf.len() == chunk.len() {
            chunk.copy_from_slice(buf);
        }
    });
    if cancelled() { None } else { Some(out) }
}

/// Which pixels of `r` a selection (or mask) covers at all: channel 0 above zero, read in
/// parallel bands as [`read_window`] does, without holding the coverage itself.
pub(crate) fn read_window_cover(surf: &Surface, r: Rect, cancelled: &(dyn Fn() -> bool + Sync)) -> Option<Vec<bool>> {
    use rayon::prelude::*;
    const BAND: usize = 64;
    let n = surf.channels().max(1);
    let (w, h) = (r.width() as usize, r.height() as usize);
    let mut out = vec![false; w * h];
    if out.is_empty() {
        return Some(out);
    }
    out.par_chunks_mut(BAND * w).enumerate().for_each_init(Vec::new, |buf, (i, chunk)| {
        if cancelled() {
            return;
        }
        let y0 = r.y0 + (i * BAND) as i32;
        surf.read_region_into(Rect::new(r.x0, y0, r.x1, y0 + (chunk.len() / w) as i32), buf);
        for (o, px) in chunk.iter_mut().zip(buf.chunks_exact(n)) {
            *o = px[0] > 0.0;
        }
    });
    if cancelled() { None } else { Some(out) }
}

/// Channel 0 of [`read_window`] (a selection's or an alpha channel's coverage).
pub(crate) fn read_window_mask(surf: &Surface, r: Rect, cancelled: &(dyn Fn() -> bool + Sync)) -> Option<Vec<f32>> {
    use rayon::prelude::*;
    let n = surf.channels().max(1);
    let all = read_window(surf, r, cancelled)?;
    if n == 1 {
        return Some(all);
    }
    Some(all.par_chunks_exact(n).map(|px| px[0]).collect())
}

/// How the source goes onto the target.
#[derive(Clone, Copy)]
struct Blend {
    mode: BlendMode,
    opacity: f32,
    /// Leave every pixel's alpha as it was (Preserve Transparency, or locked transparency).
    keep_alpha: bool,
    /// Normal mode replaces the pixels, transparency included (History: back to that state).
    restore: bool,
}

/// `d` moved towards `c` by `k`, in premultiplied space (straight RGBA in and out).
fn lerp(d: [f32; 4], c: [f32; 4], k: f32) -> [f32; 4] {
    let a = d[3] * (1.0 - k) + c[3] * k;
    if a <= 0.0 {
        return [0.0; 4];
    }
    let ch = |i: usize| (d[i] * d[3] * (1.0 - k) + c[i] * c[3] * k) / a;
    [ch(0), ch(1), ch(2), a]
}

/// Blend `source` into `surf` over `area`: per pixel, the selection coverage times the opacity,
/// in the blend mode.
/// Works tile by tile in parallel on the encoded pixels (no full-surface f32 round trip).
fn blend_into(surf: &mut Surface, area: Rect, source: &Source, sel: Option<&Surface>, b: Blend) -> Result<()> {
    use rayon::prelude::*;
    let fmt = surf.format();
    // An opaque colour in Normal mode at 100 % that may replace alpha: fully covered tiles become
    // one shared solid tile (copy-on-write), which costs nothing per pixel.
    let mut enc = [0.0f32; 8];
    let solid = match encode_color(&fmt, source, &mut enc) {
        Some(n) if b.mode == BlendMode::Normal && b.opacity >= 1.0 && (!b.keep_alpha || !fmt.alpha) => Some(surf.solid_tile(enc.get(..n).unwrap_or(&[]))),
        _ => None,
    };
    let count = area.tiles().count();
    let done = crate::allocation::capacity(count, "blending pixels")?;
    let tiles = surf
        .try_take_tiles(area)
        .map_err(|e| EngineError::Other(format!("not enough memory while preparing pixel tiles ({e}); the document was not changed")))?;
    let done = std::sync::Mutex::new(done);
    tiles.into_par_iter().try_for_each(|(tc, tile)| -> Result<()> {
        let tr = tc.rect().intersect(&area);
        let mask = sel.map(|m| Mask::of(m, tc));
        if let Some(s) = &solid
            && tr == tc.rect()
            && mask.as_ref().is_none_or(|m| m.full(tr))
        {
            done.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push((tc, s.clone()));
            return Ok(());
        }
        // Nothing selected in this tile (or no opacity): `blend_tile` would leave every pixel as
        // it is, so keep the tile itself rather than a copy the history would hold again (#4517).
        if b.opacity <= 0.0 || mask.as_ref().is_some_and(|m| m.empty(tr)) {
            done.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push((tc, tile));
            return Ok(());
        }
        let mut tile = match std::sync::Arc::try_unwrap(tile) {
            Ok(tile) => tile,
            Err(tile) => tile
                .try_clone()
                .map_err(|e| EngineError::Other(format!("not enough memory while blending a pixel tile ({e}); the document was not changed")))?,
        };
        blend_tile(tile.bytes_mut(), &fmt, tc.rect(), tr, source, mask.as_ref(), b);
        done.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push((tc, std::sync::Arc::new(tile)));
        Ok(())
    })?;
    let done = done.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner);
    surf.put_tiles(done);
    Ok(())
}

/// The selection's coverage over one tile (its channel 0).
pub(crate) struct Mask<'a> {
    bytes: &'a [u8],
    fmt: photocraft_color::PixelFormat,
    origin: Rect,
}

impl<'a> Mask<'a> {
    /// `sel` over the tile at `tc` (its default pixel where it has no tile there).
    pub(crate) fn of(sel: &'a Surface, tc: photocraft_geom::TileCoord) -> Self {
        Mask { bytes: sel.tile(tc).map_or(sel.default_bytes(), |t| t.bytes()), fmt: sel.format(), origin: tc.rect() }
    }
    pub(crate) fn at(&self, x: i32, y: i32) -> f32 {
        let bpp = self.fmt.bytes_per_pixel();
        // A default-pixel mask is one pixel long: every position reads it.
        let i = if self.bytes.len() == bpp { 0 } else { ((y - self.origin.y0) as usize * TILE_SIZE as usize + (x - self.origin.x0) as usize) * bpp };
        self.bytes.get(i..i + bpp).map_or(0.0, |px| photocraft_color::read_sample(px, self.fmt.sample, 0))
    }
    /// Fully selected over `r`?
    pub(crate) fn full(&self, r: Rect) -> bool {
        (r.y0..r.y1).all(|y| (r.x0..r.x1).all(|x| self.at(x, y) >= 1.0))
    }
    /// Whether nothing in `r` is selected.
    pub(crate) fn empty(&self, r: Rect) -> bool {
        (r.y0..r.y1).all(|y| (r.x0..r.x1).all(|x| self.at(x, y) <= 0.0))
    }
}

/// Blend into one tile's encoded pixels (`bytes`, the tile at `origin`) over `tr`.
fn blend_tile(bytes: &mut [u8], fmt: &photocraft_color::PixelFormat, origin: Rect, tr: Rect, source: &Source, mask: Option<&Mask>, b: Blend) {
    let Blend { mode, opacity, keep_alpha, restore } = b;
    let n = fmt.channels();
    let bpp = fmt.bytes_per_pixel();
    let blend = if mode == BlendMode::Dissolve { BlendMode::Normal } else { mode };
    let mut px = [0.0f32; 8];
    let mut enc = [0.0f32; 8];
    // An opaque colour at full coverage in Normal mode just replaces the colour (and the alpha
    // unless it is kept): copy its encoded bytes.
    let opaque = match encode_color(fmt, source, &mut enc) {
        Some(m) if mode == BlendMode::Normal => {
            let mut bytes = vec![0u8; bpp];
            photocraft_raster::encode_pixel(fmt, enc.get(..m.min(n)).unwrap_or(&[]), &mut bytes);
            // Keeping alpha: copy the colour samples only (alpha is the last sample).
            let keep = if keep_alpha && fmt.alpha { bpp - bpp / n.max(1) } else { bpp };
            Some((bytes, keep))
        }
        _ => None,
    };
    for y in tr.y0..tr.y1 {
        let row = (y - origin.y0) as usize * TILE_SIZE as usize;
        for x in tr.x0..tr.x1 {
            let mut k = mask.map_or(1.0, |m| m.at(x, y)) * opacity;
            if k <= 0.0 {
                continue;
            }
            let i = (row + (x - origin.x0) as usize) * bpp;
            let Some(pb) = bytes.get_mut(i..i + bpp) else { continue };
            if k >= 1.0
                && let Some((enc_px, keep)) = &opaque
            {
                if let (Some(dst), Some(src)) = (pb.get_mut(..*keep), enc_px.get(..*keep)) {
                    dst.copy_from_slice(src);
                }
                continue;
            }
            for (c, v) in px.iter_mut().enumerate().take(n) {
                *v = photocraft_color::read_sample(pb, fmt.sample, c);
            }
            let d = to_rgba(fmt, &px[..n.min(8)]);
            let mut c = source.at(x, y);
            if mode == BlendMode::Dissolve {
                // Dissolve: each pixel is either fully painted or untouched, with the coverage as odds.
                k = if photocraft_color::dither_noise(x, y) < c[3] * k { 1.0 } else { 0.0 };
                c[3] = 1.0;
                if k <= 0.0 {
                    continue;
                }
            }
            let mut o = if restore && blend == BlendMode::Normal { lerp(d, c, k) } else { photocraft_color::blend::composite(blend, d, c, k) };
            if keep_alpha || !fmt.alpha {
                o[3] = d[3];
            }
            let m = from_rgba_into(fmt, o, &mut enc);
            photocraft_raster::encode_pixel(fmt, enc.get(..m.min(n)).unwrap_or(&[]), pb);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn session() -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 40, "height": 30, "background": "transparent"})).unwrap();
        s
    }

    fn px(s: &Session, x: i32, y: i32) -> [f32; 4] {
        let st = s.active().unwrap();
        let l = st.doc.layer(st.active_layer.unwrap()).unwrap();
        l.surface().unwrap().rgba(x, y)
    }

    fn close(a: [f32; 4], b: [f32; 4]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1.5 / 255.0)
    }

    #[test]
    fn history_fill_reads_only_the_selection_inside_the_canvas() {
        let mut s = session();
        s.edit("selection", |doc, _| {
            let mut selection = Surface::new(photocraft_color::PixelFormat::GRAY8);
            selection.write_pixel(-2_000_000_000, -2_000_000_000, &[1.0]);
            selection.write_pixel(4, 4, &[1.0]);
            doc.selection = Some(selection);
            Ok(())
        })
        .unwrap();
        // Reading the full sparse selection rectangle overflows the allocation size;
        // only its intersection with this small canvas is meaningful to Fill.
        s.execute(CMD, json!({"contents": "history"})).unwrap();
        assert!(s.undo());
        assert!(s.active().unwrap().doc.selection.as_ref().unwrap().sample_channel(4, 4, 0) > 0.0);
    }

    #[test]
    fn every_constant_contents() {
        let mut s = session();
        s.execute("tools.setColors", json!({"foreground": [1.0, 0.0, 0.0, 1.0], "background": [0.0, 0.0, 1.0, 1.0]})).unwrap();
        for (c, want) in [
            ("foreground", [1.0, 0.0, 0.0, 1.0]),
            ("background", [0.0, 0.0, 1.0, 1.0]),
            ("black", [0.0, 0.0, 0.0, 1.0]),
            ("gray", [128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0, 1.0]),
            ("white", [1.0; 4]),
        ] {
            s.execute(CMD, json!({"contents": c})).unwrap();
            assert!(close(px(&s, 5, 5), want), "{c}: {:?}", px(&s, 5, 5));
        }
        s.execute(CMD, json!({"contents": "color", "color": "#00ff00"})).unwrap();
        assert!(close(px(&s, 5, 5), [0.0, 1.0, 0.0, 1.0]));
        // The old form: a colour, or the foreground.
        s.execute(CMD, json!({"color": "#ffff00"})).unwrap();
        assert!(close(px(&s, 5, 5), [1.0, 1.0, 0.0, 1.0]));
    }

    #[test]
    fn opacity_mode_and_preserve_transparency() {
        let mut s = session();
        s.execute("select.rect", json!({"x": 0, "y": 0, "width": 20, "height": 30})).unwrap();
        s.execute(CMD, json!({"contents": "white"})).unwrap();
        s.execute("select.deselect", json!({})).unwrap();
        // Preserve Transparency: the empty right half stays empty.
        s.execute(CMD, json!({"contents": "black", "opacity": 50, "preserveTransparency": true})).unwrap();
        assert!(close(px(&s, 5, 5), [0.5, 0.5, 0.5, 1.0]), "{:?}", px(&s, 5, 5));
        assert_eq!(px(&s, 30, 5)[3], 0.0);
        // Multiply with white leaves the grey.
        s.execute(CMD, json!({"contents": "white", "mode": "multiply", "preserveTransparency": true})).unwrap();
        assert!(close(px(&s, 5, 5), [0.5, 0.5, 0.5, 1.0]));
        // Without it, the transparent half is filled.
        s.execute(CMD, json!({"contents": "white", "opacity": 100})).unwrap();
        assert_eq!(px(&s, 30, 5), [1.0; 4]);
        // Dissolve paints whole pixels only.
        s.execute(CMD, json!({"contents": "black", "mode": "dissolve", "opacity": 50})).unwrap();
        let st = s.active().unwrap();
        let surf = st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap();
        let blacks = (0..40).flat_map(|x| (0..30).map(move |y| (x, y))).filter(|&(x, y)| surf.rgba(x, y)[0] < 0.01).count();
        assert!(blacks > 300 && blacks < 900, "{blacks}");
        assert!((0..40).all(|x| {
            let v = surf.rgba(x, 3)[0];
            v.is_finite() && !(0.01..=0.99).contains(&v)
        }));
    }

    #[test]
    fn history_pattern_and_content_aware() {
        let mut s = session();
        s.execute(CMD, json!({"contents": "white"})).unwrap();
        s.execute(CMD, json!({"contents": "black"})).unwrap();
        // History: back to state 1 (after the white fill).
        s.execute(CMD, json!({"contents": "history", "state": 1})).unwrap();
        assert_eq!(px(&s, 5, 5), [1.0; 4]);
        // The default source is the oldest state: the empty layer.
        s.execute(CMD, json!({"contents": "history"})).unwrap();
        assert_eq!(px(&s, 5, 5)[3], 0.0);
        assert!(s.execute(CMD, json!({"contents": "history", "state": 999})).is_err());
        let r = s.execute(CMD, json!({"contents": "pattern", "pattern": "Diagonal Lines"})).unwrap();
        assert!(r["pattern"].is_string());
        assert!(px(&s, 5, 5)[3] > 0.0);
        // Content-Aware fills the hole from what is around it.
        s.execute(CMD, json!({"contents": "color", "color": "#336699"})).unwrap();
        s.execute("select.rect", json!({"x": 15, "y": 10, "width": 6, "height": 6})).unwrap();
        s.execute(CMD, json!({"contents": "white"})).unwrap();
        s.execute(CMD, json!({"contents": "contentAware", "colorAdaptation": false})).unwrap();
        assert!(close(px(&s, 17, 12), [0x33 as f32 / 255.0, 0x66 as f32 / 255.0, 0x99 as f32 / 255.0, 1.0]), "{:?}", px(&s, 17, 12));
    }

    #[test]
    fn history_fill_reads_a_source_when_its_pixels_were_locked() {
        // Given a white layer locked in history state 1, then unlocked and filled red.
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 16, "height": 16, "background": "white"})).unwrap();
        s.execute("layer.lockLayers", json!({"pixels": true})).unwrap();
        s.execute("layer.lockLayers", json!({"pixels": false})).unwrap();
        s.execute(CMD, json!({"color": "#ff0000"})).unwrap();

        // When Fill reads the locked historical pixels.
        let result = s.execute(CMD, json!({"contents": "history", "state": 1}));

        // Then it restores white; the historical lock limits edits, not reads.
        assert!(result.is_ok(), "{result:?}");
        assert!(close(px(&s, 8, 8), [1.0; 4]), "{:?}", px(&s, 8, 8));
    }

    #[test]
    fn history_fill_refuses_a_locked_destination_when_source_is_unlocked() {
        // Given an unlocked white source and a red, currently locked destination.
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 16, "height": 16, "background": "white"})).unwrap();
        s.execute(CMD, json!({"color": "#ff0000"})).unwrap();
        s.execute("layer.lockLayers", json!({"pixels": true})).unwrap();

        // When Fill reads the unlocked source and attempts to write to the locked destination.
        let result = s.execute(CMD, json!({"contents": "history", "state": 0}));

        // Then the destination lock still rejects the edit and preserves red pixels.
        assert!(result.is_err());
        assert!(close(px(&s, 8, 8), [1.0, 0.0, 0.0, 1.0]), "{:?}", px(&s, 8, 8));
    }

    /// `cargo test -p photocraft-engine --release --lib fill_cmds::tests::timing_24mp -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn timing_24mp() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 6000, "height": 4000, "background": "white"})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        s.execute(CMD, json!({"contents": "white"})).unwrap();
        let time = |s: &mut Session, label: &str, p: Value| {
            let mut v = Vec::new();
            for i in 0..5 {
                let mut p = p.clone();
                if p.get("contents").is_none() {
                    p["color"] = json!(if i % 2 == 0 { "#336699" } else { "#996633" });
                }
                let t = std::time::Instant::now();
                s.execute(CMD, p).unwrap();
                v.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            v.sort_by(f64::total_cmp);
            println!("{label:<48} median {:>8.1} ms  min {:>8.1} ms", v[2], v[0]);
        };
        time(&mut s, "colour, no selection", json!({}));
        time(&mut s, "colour, preserve transparency", json!({"preserveTransparency": true}));
        time(&mut s, "colour 50% multiply", json!({"opacity": 50, "mode": "multiply"}));
        s.execute("select.rect", json!({"x": 100, "y": 100, "width": 5000, "height": 3000})).unwrap();
        time(&mut s, "colour, 15 MP rectangle selection", json!({}));
        s.execute("select.deselect", json!({})).unwrap();
        // The old per-pixel f32 path, for comparison.
        let st = s.active().unwrap();
        let mut surf = st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().clone();
        let t = std::time::Instant::now();
        crate::pixels::try_fill_surface(&mut surf, Rect::new(0, 0, 6000, 4000), [0.2, 0.4, 0.6, 1.0], None, false).unwrap();
        println!("{:<48} {:>8.1} ms", "fallible fill_surface, no selection", t.elapsed().as_secs_f64() * 1000.0);
    }

    #[test]
    fn bad_params_are_errors() {
        let mut s = session();
        for p in [
            json!({"contents": "plaid"}),
            json!({"contents": 3}),
            json!({"mode": "sideways"}),
            json!({"mode": "passThrough"}),
            json!({"opacity": "lots"}),
            json!({"preserveTransparency": "yes"}),
            json!({"contents": "contentAware"}),
            json!({"contents": "pattern", "pattern": "no such pattern"}),
            json!({"contents": "history", "state": -1}),
            json!({"contents": "contentAware", "colorAdaptation": 1}),
        ] {
            assert!(s.execute(CMD, p.clone()).is_err(), "{p}");
        }
        // Everything selected leaves nothing to sample from.
        s.execute("select.all", json!({})).unwrap();
        assert!(s.execute(CMD, json!({"contents": "contentAware"})).is_err());
        // Out-of-range opacity is clamped.
        s.execute(CMD, json!({"contents": "white", "opacity": 1e300})).unwrap();
    }

    /// A session whose canvas is `size` and whose selection is the given (sparse) rectangles, as
    /// a loaded `.pcraft` can declare (#963): the canvas size is not capped on load and the
    /// selection tiles can lie far apart. Nothing here allocates more than a few tiles.
    fn huge_selection(size: (u32, u32), rects: &[Rect]) -> Session {
        let mut s = session();
        s.execute(CMD, json!({"contents": "color", "color": "#336699"})).unwrap();
        let st = s.active_mut().unwrap();
        let doc = std::sync::Arc::make_mut(&mut st.doc);
        doc.size = photocraft_geom::Size::new(size.0, size.1);
        let mut sel = Surface::new(photocraft_color::PixelFormat::GRAY8);
        for r in rects {
            sel.fill_rect(*r, &[1.0]);
        }
        doc.selection = Some(sel);
        s
    }

    /// Content-Aware fill on a selection whose extent overflows the sampling-window arithmetic
    /// (#963) returns the size error and leaves the document and history untouched.
    #[test]
    fn content_aware_huge_extent_is_an_error() {
        // Largest extent whose `ext * 3` still fits in u32.
        const EDGE: i32 = (u32::MAX / 3) as i32;
        let max = i32::MAX;
        let cases: [(&str, (u32, u32), Vec<Rect>); 7] = [
            // The issue's repro: a second selection tile 5 600 000 tiles to the right.
            ("issue repro", (2_000_000_000, 2_000_000_000), vec![Rect::new(0, 0, 8, 8), Rect::new(5_600_000 * TILE_SIZE, 0, 5_600_000 * TILE_SIZE + 8, 8)]),
            ("wide, just above", (2_000_000_000, 2_000_000_000), vec![Rect::new(0, 0, 1, 1), Rect::new(EDGE, 0, EDGE + 1, 1)]),
            ("tall, just above", (2_000_000_000, 2_000_000_000), vec![Rect::new(0, 0, 1, 1), Rect::new(0, EDGE, 1, EDGE + 1)]),
            ("wide, just below", (2_000_000_000, 2_000_000_000), vec![Rect::new(0, 0, 1, 1), Rect::new(EDGE - 1, 0, EDGE, 1)]),
            ("tall, just below", (2_000_000_000, 2_000_000_000), vec![Rect::new(0, 0, 1, 1), Rect::new(0, EDGE - 1, 1, EDGE)]),
            // The largest canvas a Rect can hold, selected corner to corner, plus a piece off the
            // canvas at the most negative coordinate.
            (
                "extreme coordinates",
                (max as u32, max as u32),
                vec![Rect::new(i32::MIN, i32::MIN, i32::MIN + 1, i32::MIN + 1), Rect::new(0, 0, 1, 1), Rect::new(max - 2, max - 2, max - 1, max - 1)],
            ),
            // A canvas wider than a Rect can hold is treated as empty, never as a panic.
            ("canvas beyond i32", (u32::MAX, u32::MAX), vec![Rect::new(0, 0, 1, 1), Rect::new(max - 2, 0, max - 1, 1)]),
        ];
        for (name, size, rects) in cases {
            let mut s = huge_selection(size, &rects);
            let before = s.active().unwrap().doc.clone();
            let past = s.active().unwrap().history.past_len();
            let r = s.execute(CMD, json!({"contents": "contentAware", "colorAdaptation": false}));
            let err = r.expect_err(name).to_string();
            assert!(!err.contains("internal error"), "{name}: {err}");
            if name != "canvas beyond i32" {
                assert!(err.contains("too large for Content-Aware fill"), "{name}: {err}");
            }
            let st = s.active().unwrap();
            assert!(std::sync::Arc::ptr_eq(&before, &st.doc), "{name}: the document changed");
            assert_eq!(st.history.past_len(), past, "{name}: a history step was recorded");
            assert!(close(px(&s, 5, 5), [0x33 as f32 / 255.0, 0x66 as f32 / 255.0, 0x99 as f32 / 255.0, 1.0]), "{name}");
        }
        // Control: an ordinary selection still fills from its surroundings.
        let mut s = huge_selection((40, 30), &[Rect::new(15, 10, 21, 16)]);
        s.execute(CMD, json!({"contents": "white"})).unwrap();
        s.execute(CMD, json!({"contents": "contentAware", "colorAdaptation": false})).unwrap();
        assert!(close(px(&s, 17, 12), [0x33 as f32 / 255.0, 0x66 as f32 / 255.0, 0x99 as f32 / 255.0, 1.0]), "{:?}", px(&s, 17, 12));
    }

    /// A Stroke, Fill or Clear inside a selection keeps the tiles the selection doesn't reach as
    /// they were (shared with the history), instead of storing a copy of each (#4517).
    #[test]
    fn unselected_tiles_stay_shared() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 1024, "height": 1024, "background": "white"})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        s.execute("edit.fill", json!({"color": "#336699"})).unwrap();
        let id = s.active().unwrap().active_layer.unwrap();
        let surf = |s: &Session| s.active().unwrap().doc.layer(id).unwrap().surface().unwrap().clone();
        for (cmd, params) in [
            ("edit.fill", json!({"color": "#ff0000"})),
            ("edit.stroke", json!({"width": 4, "color": "#00ff00", "location": "inside"})),
            ("edit.clear", json!({})),
        ] {
            // Two small selections in opposite corners (256 px tiles): the selection's bounds
            // span every tile, but only the two corner tiles have anything selected.
            s.execute("select.rect", json!({"x": 20, "y": 20, "width": 100, "height": 100})).unwrap();
            s.execute("select.rect", json!({"x": 900, "y": 900, "width": 100, "height": 100, "mode": "add"})).unwrap();
            let before = surf(&s);
            s.execute(cmd, params).unwrap();
            let after = surf(&s);
            let corners = [photocraft_geom::Rect::new(0, 0, 256, 256), photocraft_geom::Rect::new(768, 768, 1024, 1024)];
            let mut kept = 0;
            for (c, t) in before.tiles() {
                if corners.iter().all(|r| c.rect().intersect(r).is_empty()) {
                    assert!(after.tile(*c).is_some_and(|a| std::sync::Arc::ptr_eq(a, t)), "{cmd}: unselected tile {c:?} was copied");
                    kept += 1;
                }
            }
            assert_eq!(kept, 14, "{cmd}: the 14 tiles between the corners");
            for r in corners {
                assert_ne!(before.read_region(r), after.read_region(r), "{cmd} changed the selected corner {r:?}");
            }
            s.execute("select.deselect", json!({})).unwrap();
        }
    }
}
