//! A stroked shape layer split into its fill and its vector stroke (rasterised over the canvas),
//! for shapes with clipped layers: Photoshop draws the stroke above the clipped layers. Cached
//! per shape state, so tiles (and the GPU's textures) reuse one rasterisation.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, OnceLock};

use photocraft_color::{PixelFormat, SampleType};
use photocraft_doc::vector::ShapeLayer;
use photocraft_geom::Rect;
use photocraft_raster::Surface;

/// (fill, stroke), last use, and the shape's pixels the stroke was fitted to (kept alive so
/// their tile addresses, part of the key, can't be reused).
type Entry = ((Surface, Surface), u64, Option<Surface>);

fn cache() -> &'static Mutex<(HashMap<u64, Entry>, u64)> {
    static C: OnceLock<Mutex<(HashMap<u64, Entry>, u64)>> = OnceLock::new();
    C.get_or_init(|| Mutex::new((HashMap::new(), 0)))
}

const CAPACITY: usize = 32;
/// Most pixel bytes the split parts may hold (#4530): each entry is two canvas-sized surfaces,
/// so 32 of them at 24 MP could hold several GB outside every memory budget.
#[cfg(not(test))]
const BUDGET: usize = 256 << 20;
/// Small under test, so a few 512² shapes exercise the eviction.
#[cfg(test)]
const BUDGET: usize = 8 << 20;

/// Pixel bytes of one entry's (fill, stroke) parts (the pinned shape pixels are the document's).
fn entry_bytes(parts: &(Surface, Surface)) -> usize {
    [&parts.0, &parts.1].iter().map(|s| s.tiles().map(|(_, t)| t.bytes().len()).sum::<usize>()).sum()
}

/// Drop every cached split (Edit › Purge › All), returning the parts' pixel bytes. Callers may
/// still hold returned surfaces, so this is not a measure of freed process memory.
pub fn purge_cache() -> usize {
    let mut c = cache().lock().unwrap_or_else(|e| e.into_inner());
    let bytes = c.0.values().map(|e| entry_bytes(&e.0)).sum();
    c.0.clear();
    bytes
}

/// (fill only, stroke only) of a stroked shape, rendered over `canvas` in RGBA at `depth`.
/// `None` without a stroke. Where the shape's pixels
/// (Photoshop's rendering, `sh.cache`) reach past our fill, the part the fill doesn't explain is
/// stroke: the stroke follows Photoshop's coverage (psd-tools double-stroke-effects).
pub fn split(sh: &ShapeLayer, canvas: Rect, depth: SampleType) -> Option<(Surface, Surface)> {
    sh.stroke.as_ref()?;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (canvas.x0, canvas.y0, canvas.x1, canvas.y1).hash(&mut h);
    depth.hash(&mut h);
    format!("{:?}{:?}{:?}{:?}", sh.path, sh.fill, sh.stroke, sh.live).hash(&mut h);
    if let Some(c) = &sh.cache {
        format!("{:?}{:?}", c.format(), c.default_pixel()).hash(&mut h);
        for (k, t) in c.tiles() {
            (k.tx, k.ty, std::sync::Arc::as_ptr(t) as usize).hash(&mut h);
        }
    }
    let key = h.finish();
    {
        let mut c = cache().lock().unwrap_or_else(|e| e.into_inner());
        c.1 += 1;
        let tick = c.1;
        if let Some(e) = c.0.get_mut(&key) {
            e.1 = tick;
            return Some(e.0.clone());
        }
    }
    let fmt = PixelFormat::RGBA8.with_sample(depth);
    let fill_only = ShapeLayer { stroke: None, cache: None, ..sh.clone() };
    let stroke_only = ShapeLayer { fill: None, cache: None, ..sh.clone() };
    let mut parts = (photocraft_vector::render_shape(&fill_only, fmt, canvas), photocraft_vector::render_shape(&stroke_only, fmt, canvas));
    if let Some(cache) = &sh.cache {
        fit_stroke(&parts.0, &mut parts.1, cache, canvas);
    }
    let mut c = cache().lock().unwrap_or_else(|e| e.into_inner());
    // Least recently used first, until the new entry fits both the count and the byte budget.
    let new_bytes = entry_bytes(&parts);
    loop {
        let held: usize = c.0.values().map(|e| entry_bytes(&e.0)).sum();
        if c.0.len() < CAPACITY && held.saturating_add(new_bytes) <= BUDGET {
            break;
        }
        let Some(old) = c.0.iter().min_by_key(|e| (e.1).1).map(|e| *e.0) else { break };
        c.0.remove(&old);
    }
    if new_bytes > BUDGET {
        // Larger than the whole budget on its own: use it, don't keep it.
        return Some(parts);
    }
    let tick = c.1;
    c.0.insert(key, (parts.clone(), tick, sh.cache.clone()));
    Some(parts)
}

/// Fits the stroke part to the shape's own pixels: where they reach past the fill, the stroke
/// covers what the fill doesn't (`(pixels - fill) / (1 - fill)`), in the pixels' colour where we
/// drew no stroke, and elsewhere in the colour that, drawn over the fill, gives back the pixels'
/// colour. Our stroke's own colour would be wrong where the file's fill covers more of the pixel
/// than ours (shapes snapped to whole pixels with a stroke thinner than a pixel): the fitted
/// stroke is then mostly the file's fill, and an edge drawn in the stroke's full colour is far
/// too dark.
fn fit_stroke(fill: &Surface, stroke: &mut Surface, cache: &Surface, canvas: Rect) {
    // Where either has pixels (the stroke may reach past the shape's pixels, and vice versa).
    let (cb, sb) = (crate::bounds::content_bounds(cache), crate::bounds::content_bounds(stroke));
    let r = if cb.is_empty() {
        sb
    } else if sb.is_empty() {
        cb
    } else {
        cb.union(&sb)
    }
    .intersect(&canvas);
    if r.is_empty() {
        return;
    }
    let n = r.width() as usize * r.height() as usize;
    let (mut f, mut s, mut c) = (vec![[0.0f32; 4]; n], vec![[0.0f32; 4]; n], vec![[0.0f32; 4]; n]);
    fill.read_rgba_into(r, &mut f);
    stroke.read_rgba_into(r, &mut s);
    cache.read_rgba_into(r, &mut c);
    let mut out = Vec::with_capacity(n * 4);
    let float = stroke.format().sample == SampleType::F32;
    for ((fp, sp), cp) in f.iter().zip(&s).zip(&c) {
        let mut p = *sp;
        if fp[3] < 1.0 - 1e-3 {
            let a = ((cp[3] - fp[3]) / (1.0 - fp[3])).clamp(0.0, 1.0);
            p = if sp[3] <= 0.0 {
                [cp[0], cp[1], cp[2], a]
            } else if a > 1e-3 {
                // Normal over the fill: a·p + f·(1 - a)·fill = pixels (premultiplied).
                let k = fp[3] * (1.0 - a);
                let un = |i: usize| {
                    let v = (cp[3] * cp[i] - k * fp[i]) / a;
                    // Float RGB retains HDR values; integer samples use the normalized range.
                    if float { v } else { v.clamp(0.0, 1.0) }
                };
                [un(0), un(1), un(2), a]
            } else {
                [sp[0], sp[1], sp[2], a]
            };
        }
        out.extend_from_slice(&p);
    }
    stroke.write_region(r, &out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_color::Color;
    use photocraft_doc::Fill;
    use photocraft_doc::vector::{Path, ShapeStroke, Subpath};

    fn shape(x: f64) -> ShapeLayer {
        ShapeLayer {
            path: Path::new(vec![Subpath::polygon(&[(x, 10.0), (x + 400.0, 10.0), (x + 400.0, 500.0), (x, 500.0)])]),
            fill: Some(Fill::Solid(Color::rgb(0.2, 0.4, 0.6))),
            stroke: Some(ShapeStroke { width: 6.0, ..ShapeStroke::default() }),
            live: None,
            cache: None,
            psd_raw: None,
        }
    }

    fn held() -> usize {
        cache().lock().unwrap_or_else(|e| e.into_inner()).0.values().map(|e| entry_bytes(&e.0)).sum()
    }

    /// The split cache stays within its byte budget, evicting the least recently used, and
    /// Edit › Purge › All empties it (#4530).
    #[test]
    fn split_cache_keeps_to_its_budget_and_purges() {
        let canvas = Rect::new(0, 0, 1024, 512);
        for i in 0..6 {
            let parts = split(&shape(f64::from(i) * 10.0), canvas, SampleType::U8).expect("stroked");
            assert!(entry_bytes(&parts) > 0);
            assert!(held() <= BUDGET, "shape {i}: {} bytes held", held());
        }
        // Other tests share the cache, so only what this test can rely on: a purge frees bytes.
        assert!(purge_cache() > 0, "something to purge");
    }
}
