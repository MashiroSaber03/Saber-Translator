//! Destructive pixel operations on layers (Image → Adjustments, flips, fills).

use photocraft_compose::{Buffer, adjust};
use photocraft_doc::{Adjustment, Document, Layer, LayerContent};
use photocraft_geom::{Rect, TILE_SIZE};
use photocraft_raster::{Surface, to_rgba};

/// Composite `layers` inside their visible composite bounds (effects included) and write the result directly into a
/// tiled surface. Large merges used to hold both a full-canvas RGBA buffer and a second converted
/// float buffer at once; banded rendering keeps only one compositor band plus the destination.
pub fn composite_layers(solo: &Document, format: photocraft_color::PixelFormat, background: Option<[f32; 3]>) -> Surface {
    let canvas = solo.bounds();
    let area = solo
        .layers
        .iter()
        .filter(|layer| layer.visible)
        // Composite bounds include layer effects (a drop shadow reaches past the pixels); `None`
        // means the layer may draw anywhere.
        .map(|layer| photocraft_compose::composite_bounds(layer, canvas).unwrap_or(canvas))
        .filter(|bounds| !bounds.is_empty())
        .reduce(|a, b| a.union(&b))
        .unwrap_or(Rect::EMPTY)
        .intersect(&canvas);
    if area.is_empty() {
        return Surface::new(format);
    }
    // Also keeps a plain CMYK stack in its ink values (#3778).
    photocraft_compose::composite_to_surface(solo, area, format, background)
}

/// Apply an adjustment destructively to a surface, weighted by an optional selection.
/// Applies `adj` to a surface (any colour model and depth, via straight RGBA) through the
/// selection; `mode` is the document's, for the tone transfer (e.g. Exposure in Grayscale).
/// Works tile by tile in parallel, so a large layer needs one tile's buffers per thread rather
/// than four full-layer copies, and the result is the same as one pass over the whole region.
pub fn adjust_surface(s: &mut Surface, adj: &Adjustment, selection: Option<&Surface>, mode: photocraft_color::ColorMode) {
    use rayon::prelude::*;
    let r = s.content_bounds();
    if r.is_empty() {
        return;
    }
    let fmt = s.format();
    let done: Vec<_> = s
        .take_tiles(r)
        .into_par_iter()
        .flat_map_iter(|(tc, tile)| {
            // A one-tile surface, so `adjusted` reads and writes this tile only.
            let mut one = Surface::new(fmt);
            one.put_tiles([(tc, tile)]);
            let tr = tc.rect().intersect(&r);
            let out = adjusted(&one, tr, adj, selection, mode);
            one.write_region(tr, &out);
            one.take_tiles(tc.rect())
        })
        .collect();
    s.put_tiles(done);
}

#[cfg(test)]
#[path = "flatten_cmyk_tests.rs"]
mod flatten_cmyk_tests;
#[cfg(test)]
#[path = "pixels_tests.rs"]
mod tests;

/// Applies `adj` to a layer mask through the selection. Untouched mask pixels count (they read as
/// the mask's default value), and without a selection the default changes too, since a mask
/// reaches past the canvas: Invert turns a reveal-all mask into a hide-all one (#780).
pub fn adjust_mask(s: &mut Surface, adj: &Adjustment, selection: Option<&Surface>) {
    let mode = photocraft_color::ColorMode::Grayscale;
    if let Some(sel) = selection {
        let r = sel.content_bounds();
        if !r.is_empty() {
            let out = adjusted(s, r, adj, Some(sel), mode);
            s.write_region(r, &out);
            s.prune();
        }
        return;
    }
    let fmt = s.format();
    let unit = Rect::from_xywh(0, 0, 1, 1);
    let mut out = Surface::with_default(fmt, &adjusted(&Surface::with_default(fmt, &s.default_pixel()), unit, adj, None, mode));
    let r = s.tile_bounds();
    if !r.is_empty() {
        out.write_region(r, &adjusted(s, r, adj, None, mode));
    }
    out.prune();
    *s = out;
}

/// `s`'s pixels over `r` (untouched ones included) with `adj` applied through the selection, encoded.
fn adjusted(s: &Surface, r: Rect, adj: &Adjustment, selection: Option<&Surface>, mode: photocraft_color::ColorMode) -> Vec<f32> {
    let fmt = s.format();
    if fmt.mode == mode && matches!(mode, photocraft_color::ColorMode::Lab | photocraft_color::ColorMode::Cmyk) && adjust::stored_channel_adjustment(adj) {
        return adjusted_channels(s, r, adj, selection);
    }
    let n = fmt.channels();
    let raw = s.read_region(r);
    let mut buf = Buffer { rect: r, px: raw.chunks_exact(n).map(|p| to_rgba(&fmt, p)).collect() };
    let orig = buf.clone();
    // 32-bit documents get the float behaviour of an adjustment layer there (Levels doesn't clip);
    // integer depths keep the unrounded, clipped curves.
    let depth = (fmt.sample == photocraft_color::SampleType::F32).then_some(fmt.sample);
    adjust::apply_in(adj, &mut buf, adjust::Transfer::for_document(mode, fmt.sample), depth, mode);
    mix_through_selection(&fmt, r, &orig.px, &buf.px, selection)
}

/// [`adjusted`] for an adjustment Photoshop applies to a Lab or CMYK layer's own channels
/// (`adjust::stored_channels`): exact on the stored values, with no trip through RGB, and mixed
/// through the selection on those values (alpha untouched).
fn adjusted_channels(s: &Surface, r: Rect, adj: &Adjustment, selection: Option<&Surface>) -> Vec<f32> {
    let fmt = s.format();
    let (n, colors) = (fmt.channels(), fmt.mode.color_channels());
    let mut raw = s.read_region(r);
    let w = (r.width() as usize).max(1);
    let mut coverage = vec![1.0f32; w];
    for (j, row) in raw.chunks_mut(w * n).enumerate() {
        if let Some(sel) = selection {
            sel.sample_row_strided(r.y0.saturating_add(j as i32), r.x0, 1, 0, &mut coverage);
        }
        for (px, &k) in row.chunks_exact_mut(n).zip(&coverage) {
            if k <= 0.0 {
                continue;
            }
            let p = to_rgba(&fmt, px);
            let gray = photocraft_color::convert::rgb_to_gray([p[0], p[1], p[2]]);
            let Some(ch) = px.get_mut(..colors) else { continue };
            let mut out = [0.0f32; 4];
            let Some(out) = out.get_mut(..colors) else { continue };
            out.copy_from_slice(ch);
            adjust::stored_channels(adj, fmt.mode, Some(fmt.sample), out, gray);
            for (v, o) in ch.iter_mut().zip(out.iter()) {
                *v += (o - *v) * k;
            }
        }
    }
    raw
}

/// `edited` mixed back over `orig` (straight RGBA over `r`, row-major) by the selection's
/// coverage, encoded in `fmt`: what a destructive edit writes back through a selection.
pub(crate) fn mix_through_selection(
    fmt: &photocraft_color::PixelFormat,
    r: Rect,
    orig: &[[f32; 4]],
    edited: &[[f32; 4]],
    selection: Option<&Surface>,
) -> Vec<f32> {
    let w = (r.width() as usize).max(1);
    let mut out = Vec::with_capacity(orig.len() * fmt.channels());
    // The selection's coverage (channel 0) one row at a time, each of its tiles looked up once
    // per row rather than once per pixel.
    let mut coverage = vec![1.0f32; w];
    for (j, (edited_row, orig_row)) in edited.chunks(w).zip(orig.chunks(w)).enumerate() {
        if let Some(sel) = selection {
            sel.sample_row_strided(r.y0.saturating_add(j as i32), r.x0, 1, 0, &mut coverage);
        }
        for ((a, o), k) in edited_row.iter().zip(orig_row).zip(&coverage) {
            let mixed: [f32; 4] = std::array::from_fn(|c| o[c] + (a[c] - o[c]) * k);
            let mut enc = [0.0f32; 8];
            let m = photocraft_raster::from_rgba_into(fmt, mixed, &mut enc);
            out.extend_from_slice(&enc[..m]);
        }
    }
    out
}

/// Fill (respecting selection coverage) with a straight RGBA colour.
/// Works tile by tile in parallel on the encoded pixels; a fully selected tile of an opaque fill
/// that may set alpha becomes one shared solid tile.
pub(crate) fn try_fill_surface(s: &mut Surface, area: Rect, color: [f32; 4], selection: Option<&Surface>, lock_transparency: bool) -> crate::Result<()> {
    let fmt = s.format();
    // Coverage 1 of an opaque colour replaces the colour (and the alpha, unless it is kept).
    let opaque = color[3] == 1.0 && (!lock_transparency || !fmt.alpha) && selection.is_none_or(|m| m.format().sample != photocraft_color::SampleType::F32);
    let solid = opaque.then(|| {
        let mut enc = [0.0f32; 8];
        let n = photocraft_raster::from_rgba_into(&fmt, [color[0], color[1], color[2], 1.0], &mut enc);
        s.solid_tile(enc.get(..n).unwrap_or(&[]))
    });
    let whole = |full: bool| solid.as_ref().filter(|_| full).map(|t| TileEdit::Put(t.clone()));
    edit_tiles(s, area, selection, "filling pixels", whole, |k, px| {
        if k <= 0.0 {
            return false;
        }
        fill_pixel(&fmt, px, color, k, lock_transparency);
        true
    })
}

/// One pixel of [`try_fill_surface`]: `color` over `px` (encoded) at coverage `k`.
fn fill_pixel(fmt: &photocraft_color::PixelFormat, px: &mut [f32], color: [f32; 4], k: f32, lock_transparency: bool) {
    let d = to_rgba(fmt, px);
    let sa = color[3] * k;
    let oa = sa + d[3] * (1.0 - sa);
    let mut o = [0.0f32; 4];
    if oa > 0.0 {
        for c in 0..3 {
            o[c] = (color[c] * sa + d[c] * d[3] * (1.0 - sa)) / oa;
        }
    }
    o[3] = if lock_transparency { d[3] } else { oa };
    let mut enc = [0.0f32; 8];
    let m = photocraft_raster::from_rgba_into(fmt, o, &mut enc);
    let n = px.len().min(m);
    if let (Some(dst), Some(src)) = (px.get_mut(..n), enc.get(..n)) {
        dst.copy_from_slice(src);
    }
}

/// Clear pixels (make transparent) within the selection.
/// Works tile by tile in parallel on the encoded pixels; a fully selected tile is dropped (it
/// would only hold empty pixels), and tiles the selection leaves alone stay shared with history.
pub(crate) fn try_clear_surface(s: &mut Surface, area: Rect, selection: Option<&Surface>) -> crate::Result<()> {
    let fmt = s.format();
    let n = fmt.channels();
    if !fmt.alpha {
        return Ok(());
    }
    // Cleared pixels become all zeros; when that is also the default pixel, a cleared tile is no tile.
    let empty_default = s.default_bytes().iter().all(|&b| b == 0);
    let whole = |full: bool| (full && empty_default).then_some(TileEdit::Drop);
    edit_tiles(s, area, selection, "erasing pixels", whole, |k, px| {
        let Some(a) = px.get_mut(n - 1) else { return false };
        *a *= 1.0 - k;
        // Fully cleared pixels become the empty pixel, so bounds and tile pruning see them as gone.
        if *a <= 0.0 {
            px.fill(0.0);
        }
        true
    })?;
    s.prune();
    Ok(())
}

/// A whole-tile shortcut of [`edit_tiles`].
enum TileEdit {
    /// Replace the tile with this (shared) one.
    Put(std::sync::Arc<photocraft_raster::Tile>),
    /// Remove the tile (it reads as the default pixel again).
    Drop,
}

/// Edit `s` over `area` tile by tile in parallel. `whole` may handle a tile lying wholly inside
/// `area` and fully selected (`true`) in one go; otherwise `pixel` edits each pixel of the area
/// with its selection coverage (decoded channel values in and out; `false` = left untouched).
/// A tile is copied only when one of its pixels actually changes (tiles are shared with undo
/// states), and pixels are decoded and re-encoded exactly as a whole-region float round trip
/// would, so the result is the same as one pass over `area`.
fn edit_tiles(
    s: &mut Surface,
    area: Rect,
    selection: Option<&Surface>,
    purpose: &str,
    whole: impl Fn(bool) -> Option<TileEdit> + Sync,
    pixel: impl Fn(f32, &mut [f32]) -> bool + Sync,
) -> crate::Result<()> {
    use photocraft_raster::Tile;
    use rayon::prelude::*;
    use std::sync::Arc;
    let fmt = s.format();
    let (n, bpp) = (fmt.channels(), fmt.bytes_per_pixel());
    let oom = |e| crate::EngineError::Other(format!("not enough memory while {purpose} ({e}); the document was not changed"));
    let mut coords = crate::allocation::capacity(area.tiles().count(), purpose)?;
    coords.extend(area.tiles());
    // Untouched pixels read as the default where the surface has no tile.
    let blank = if coords.iter().any(|&tc| s.tile(tc).is_none()) { Some(s.try_default_tile().map_err(oom)?) } else { None };
    let surf = &*s;
    let edits = coords
        .into_par_iter()
        .map(|tc| -> crate::Result<Option<(photocraft_geom::TileCoord, Option<Arc<Tile>>)>> {
            let tr = tc.rect().intersect(&area);
            let mask = selection.map(|m| crate::fill_cmds::Mask::of(m, tc));
            let full = tr == tc.rect() && mask.as_ref().is_none_or(|m| m.full(tr));
            let current = surf.tile(tc);
            match whole(full) {
                Some(TileEdit::Put(t)) => return Ok(Some((tc, Some(t)))),
                Some(TileEdit::Drop) => return Ok(current.map(|_| (tc, None))),
                None => {}
            }
            let Some(src) = current.map(|t| &**t).or(blank.as_ref()) else { return Ok(None) };
            let mut out: Option<Tile> = None;
            let (mut px, mut enc) = ([0.0f32; 8], [0u8; 32]);
            let (Some(px), Some(enc)) = (px.get_mut(..n), enc.get_mut(..bpp)) else { return Ok(None) };
            for y in tr.y0..tr.y1 {
                let row = (y - tc.ty * TILE_SIZE) as usize * TILE_SIZE as usize;
                for x in tr.x0..tr.x1 {
                    let i = (row + (x - tc.tx * TILE_SIZE) as usize) * bpp;
                    let Some(old) = src.bytes().get(i..i + bpp) else { continue };
                    for (c, v) in px.iter_mut().enumerate() {
                        *v = photocraft_color::read_sample(old, fmt.sample, c);
                    }
                    if !pixel(mask.as_ref().map_or(1.0, |m| m.at(x, y)), px) {
                        continue;
                    }
                    photocraft_raster::encode_pixel(&fmt, px, enc);
                    if enc == old {
                        continue;
                    }
                    if out.is_none() {
                        out = Some(src.try_clone().map_err(oom)?);
                    }
                    if let Some(dst) = out.as_mut().and_then(|t| t.bytes_mut().get_mut(i..i + bpp)) {
                        dst.copy_from_slice(enc);
                    }
                }
            }
            Ok(out.map(|t| (tc, Some(Arc::new(t)))))
        })
        .collect::<crate::Result<Vec<_>>>()?;
    for (tc, tile) in edits.into_iter().flatten() {
        match tile {
            Some(t) => s.put_tiles([(tc, t)]),
            None => drop(s.take_tiles(tc.rect())),
        }
    }
    Ok(())
}

/// Remap a surface through a pixel-coordinate mapping (used for flips/rotations).
pub fn remap_surface(s: &Surface, map: impl Fn(i32, i32) -> (i32, i32)) -> Surface {
    let r = s.content_bounds();
    let mut out = Surface::with_default(s.format(), &s.default_pixel());
    let n = s.channels();
    let raw = s.read_region(r);
    let w = r.width() as usize;
    for (i, px) in raw.chunks_exact(n).enumerate() {
        let (x, y) = (r.x0 + (i % w) as i32, r.y0 + (i / w) as i32);
        let (nx, ny) = map(x, y);
        out.write_pixel(nx, ny, px);
    }
    out
}

/// Merge `upper` onto `lower` producing a raster layer (Layer → Merge Down).
pub fn merge_down(doc: &Document, lower: &Layer, upper: &Layer) -> Layer {
    let format = doc.pixel_format();
    let stack = vec![lower.clone(), upper.clone()];
    let mut tmp = Document::new("merge", doc.size, format.mode, format.sample);
    // The pair is read (and a CMYK one separated again) through the document's own profile.
    tmp.icc_profile = doc.icc_profile.clone();
    tmp.layers = stack;
    let s = composite_layers(&tmp, format, None);
    let mut merged = Layer::new(lower.name.clone(), LayerContent::Raster(s));
    merged.id = lower.id;
    merged.blend = lower.blend;
    merged.opacity = 1.0;
    // Merge Down replaces the lower layer, so keep its editing restrictions.
    // In particular, a merged Background must remain opaque and position-locked.
    merged.locks = lower.locks;
    merged
}
