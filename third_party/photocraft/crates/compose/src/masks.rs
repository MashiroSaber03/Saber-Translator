//! A layer's effective mask (pixel mask × vector mask, each feathered) as one surface, for
//! backends that sample masks from textures (the GPU compositor) and for feathered masks on the
//! CPU. Rasterising a vector mask costs a path coverage pass and feathering a blur, so results
//! are cached per mask state; unchanged masks return the same tiles (cheap to clone, and the
//! GPU sees no change). Retention is bounded by both entry count and accounted pixel bytes;
//! [`purge_cache`] drops cached masks and prevents older builds from refilling the cache.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_doc::Layer;
use photocraft_geom::Rect;
use photocraft_raster::Surface;

// A U16 mask keeps substantially smaller cached tiles than F32 while adding at most
// 1/65535 quantization error. Mask values are normalized coverage, so this is finer than
// the final 8-bit display/composite precision.
const FORMAT: PixelFormat = PixelFormat { mode: ColorMode::Grayscale, sample: SampleType::U16, alpha: false };

struct CacheEntry {
    surface: Surface,
    tick: u64,
    bytes: usize,
    // The key contains input tile addresses. Keep those tiles alive until eviction:
    // edits must copy them, and the allocator cannot reuse their addresses for new tiles.
    _pixel_input: Option<Surface>,
}

#[derive(Default)]
struct Cache {
    map: HashMap<u64, CacheEntry>,
    /// Builds in progress, shared by every caller that asks for the same cold mask.
    in_flight: HashMap<u64, Weak<OnceLock<Surface>>>,
    tick: u64,
    bytes: usize,
    generation: u64,
}

impl Cache {
    fn get(&mut self, key: u64) -> Option<Surface> {
        self.tick = self.tick.saturating_add(1);
        let entry = self.map.get_mut(&key)?;
        entry.tick = self.tick;
        Some(entry.surface.clone())
    }

    fn insert(&mut self, key: u64, surface: Surface, pixel_input: Option<Surface>, bytes: usize, budget: usize, generation: u64) {
        // A large result can still be used by its caller without keeping it globally. A purge
        // also invalidates admission by computations that were already running.
        if bytes > budget || generation != self.generation {
            return;
        }
        if let Some(previous) = self.map.remove(&key) {
            self.bytes = self.bytes.saturating_sub(previous.bytes);
        }
        while self.map.len() >= CAPACITY || self.bytes > budget.saturating_sub(bytes) {
            let Some(old) = self.map.iter().min_by_key(|(_, entry)| entry.tick).map(|(&key, _)| key) else { break };
            if let Some(entry) = self.map.remove(&old) {
                self.bytes = self.bytes.saturating_sub(entry.bytes);
            }
        }
        self.bytes = self.bytes.saturating_add(bytes);
        self.tick = self.tick.saturating_add(1);
        self.map.insert(key, CacheEntry { surface, tick: self.tick, bytes, _pixel_input: pixel_input });
    }

    fn clear(&mut self) -> usize {
        let bytes = self.bytes;
        self.map.clear();
        self.bytes = 0;
        self.tick = 0;
        self.generation = self.generation.wrapping_add(1);
        bytes
    }
}

fn cache() -> &'static Mutex<Cache> {
    static C: OnceLock<Mutex<Cache>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(Cache::default()))
}

/// Entries kept (least recently used dropped first).
const CAPACITY: usize = 64;

/// Accounted pixel-byte limit of the combined-mask cache. Five 50 MP U8 input/U16 result pairs
/// fit, while the hard cap still bounds retention for larger documents or more active masks.
pub const CACHE_BUDGET: usize = 768 << 20;

enum CachedMask {
    Hit(Surface),
    /// A build slot, and the cache generation it may fill.
    Pending(Arc<OnceLock<Surface>>, u64),
}

fn lookup(c: &Mutex<Cache>, k: u64) -> CachedMask {
    let mut c = c.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(surface) = c.get(k) {
        return CachedMask::Hit(surface);
    }
    let generation = c.generation;
    // The callers own each slot, so completed builds do not retain an extra
    // surface after their last caller returns. Sweep dead weak slots on misses.
    c.in_flight.retain(|_, slot| slot.strong_count() > 0);
    if let Some(slot) = c.in_flight.get(&k).and_then(Weak::upgrade) {
        return CachedMask::Pending(slot, generation);
    }
    let slot = Arc::new(OnceLock::new());
    c.in_flight.insert(k, Arc::downgrade(&slot));
    CachedMask::Pending(slot, generation)
}

/// The slot's surface, built once by whichever caller gets there first. `build` returns the
/// mask and the pixel-mask input its key was made from (kept alive by the cache entry).
fn finish_mask(c: &Mutex<Cache>, k: u64, slot: &Arc<OnceLock<Surface>>, generation: u64, build: impl FnOnce() -> (Surface, Option<Surface>)) -> Surface {
    slot.get_or_init(|| {
        // Never hold the global cache mutex while rasterizing or feathering.
        let (surface, pixel_input) = build();
        let bytes = entry_bytes(&surface, pixel_input.as_ref());
        let mut c = c.lock().unwrap_or_else(|e| e.into_inner());
        if c.in_flight.get(&k).and_then(Weak::upgrade).is_some_and(|active| Arc::ptr_eq(&active, slot)) {
            c.insert(k, surface.clone(), pixel_input, bytes, CACHE_BUDGET, generation);
        }
        surface
    })
    .clone()
}

fn cached_mask(c: &Mutex<Cache>, k: u64, build: impl FnOnce() -> (Surface, Option<Surface>)) -> Surface {
    match lookup(c, k) {
        CachedMask::Hit(surface) => surface,
        CachedMask::Pending(slot, generation) => finish_mask(c, k, &slot, generation, build),
    }
}

fn surface_bytes(surface: &Surface) -> usize {
    surface.tiles().fold(surface.default_bytes().len(), |bytes, (_, tile)| bytes.saturating_add(tile.bytes().len()))
}

fn entry_bytes(surface: &Surface, pixel_input: Option<&Surface>) -> usize {
    // Charge the whole source as well as the result, including off-canvas tiles. Shared tiles
    // are deliberately charged per entry rather than relying on their current owners.
    surface_bytes(surface).saturating_add(pixel_input.map_or(0, surface_bytes))
}

/// Pixel bytes accounted for by the combined-mask cache. This conservatively includes source
/// tiles and counts tiles shared across entries more than once; it is not process memory use.
pub fn cache_bytes() -> usize {
    cache().lock().unwrap_or_else(|e| e.into_inner()).bytes
}

/// Drop every cached combined mask (Edit › Purge › All), returning its accounted pixel bytes.
/// Callers may still hold returned surfaces, so this is not a measure of freed process memory.
pub fn purge_cache() -> usize {
    cache().lock().unwrap_or_else(|e| e.into_inner()).clear()
}

fn key(layer: &Layer, canvas: Rect) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    layer.id.0.hash(&mut h);
    (canvas.x0, canvas.y0, canvas.x1, canvas.y1).hash(&mut h);
    format!("{:?}", layer.vector_mask).hash(&mut h);
    if let Some(m) = &layer.mask {
        (m.enabled, m.density.to_bits(), m.feather.to_bits()).hash(&mut h);
        format!("{:?}", m.surface.default_pixel()).hash(&mut h);
        for (c, t) in m.surface.tiles() {
            (c.tx, c.ty, Arc::as_ptr(t) as usize).hash(&mut h);
        }
    }
    h.finish()
}

/// Gaussian sigma of a mask feather of `px` pixels (Properties › Feather). Fitted on the
/// psd-tools corpus (layer_mask_data.psd, mask-parameters-no-real-channel.psd: sigma = feather
/// beats 0.33, 0.5, 0.7, 1.3 and 1.6 times it).
pub fn feather_sigma(px: f32) -> f32 {
    if px.is_finite() { (px * FEATHER_SIGMA).clamp(0.0, MAX_FEATHER_SIGMA) } else { 0.0 }
}

const FEATHER_SIGMA: f32 = 1.0;
/// Photoshop's feather tops out at 1000 px.
const MAX_FEATHER_SIGMA: f32 = 1000.0 * FEATHER_SIGMA;

/// Whether `layer` has an enabled mask with a feather (rendered through [`combined_mask`]).
pub fn has_feather(layer: &Layer) -> bool {
    layer.mask.as_ref().is_some_and(|m| m.enabled && feather_sigma(m.feather) > 0.0)
        || layer.vector_mask.as_ref().is_some_and(|v| v.enabled && feather_sigma(v.feather) > 0.0)
}

/// Apply one of the three box passes to a single line. `scratch` preserves the input while
/// the sliding sum writes the output, matching the previous sequential implementation exactly.
fn blur_line(line: &mut [f32], scratch: &mut [f32], r: usize) {
    if line.is_empty() {
        return;
    }
    let Ok(radius) = isize::try_from(r) else { return };
    scratch.copy_from_slice(line);
    let get = |i: isize| scratch[i.clamp(0, scratch.len() as isize - 1) as usize];
    let mut acc: f32 = (-radius..=radius).map(get).sum();
    let n = (2 * r + 1) as f32;
    for (i, dst) in line.iter_mut().enumerate() {
        *dst = acc / n;
        let i = i as isize;
        acc += get(i + radius + 1) - get(i - radius);
    }
}

/// Apply three clamped box passes to independent contiguous rows. Scoped OS threads are used
/// instead of Rayon because mask construction can happen inside a Rayon tile job; nested Rayon
/// work here can deadlock when every worker waits on the same OnceLock (#276).
fn blur_rows_chunk(v: &mut [f32], w: usize, r: usize) {
    let mut scratch = vec![0.0; w];
    for row in v.chunks_mut(w) {
        for _ in 0..3 {
            blur_line(row, &mut scratch, r);
        }
    }
}

fn blur_rows(v: &mut [f32], w: usize, r: usize) {
    if w == 0 || v.is_empty() {
        return;
    }
    let h = v.len() / w;
    let threads = if cfg!(target_arch = "wasm32") || v.len() < 1 << 16 {
        1
    } else {
        std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, 16).min(h.max(1))
    };
    if threads == 1 {
        blur_rows_chunk(v, w, r);
        return;
    }
    let rows_per_worker = h.div_ceil(threads);
    let chunk_len = rows_per_worker.saturating_mul(w);
    std::thread::scope(|scope| {
        for chunk in v.chunks_mut(chunk_len) {
            scope.spawn(move || blur_rows_chunk(chunk, w, r));
        }
    });
}

/// Cache-friendly blocked transpose. It makes the vertical blur contiguous without a Rayon job
/// or unsafe disjoint-column access; the temporary is released before the next mask is built.
fn transpose_into(v: &[f32], out: &mut [f32], w: usize, h: usize) {
    const BLOCK: usize = 32;
    for by in (0..h).step_by(BLOCK) {
        for bx in (0..w).step_by(BLOCK) {
            for y in by..(by + BLOCK).min(h) {
                for x in bx..(bx + BLOCK).min(w) {
                    out[x * h + y] = v[y * w + x];
                }
            }
        }
    }
}

/// Approximate Gaussian blur of a `w`×`h` plane: three box passes per axis (edges clamp, which
/// is exact here since the planes extend into constant mask regions). The axes are processed in
/// parallel on bounded OS threads, preserving each line's floating-point operation order.
fn gaussian(v: &mut [f32], w: usize, h: usize, sigma: f32) {
    if sigma <= 0.0 || w == 0 || h == 0 || v.len() != w.saturating_mul(h) {
        return;
    }
    // Box widths whose three-pass variance matches sigma² (Wells 1986 / Kovesi).
    let ideal = (12.0 * sigma * sigma / 3.0 + 1.0).sqrt();
    let r = (((ideal.floor() as usize) | 1).max(1) - 1) / 2;
    if r == 0 {
        return;
    }
    // Reserve before modifying the input so allocation failure can use the sequential fallback
    // without leaving a half-blurred mask.
    let mut transposed = Vec::new();
    if transposed.try_reserve_exact(v.len()).is_err() {
        gaussian_sequential(v, w, h, r);
        return;
    }
    transposed.resize(v.len(), 0.0);
    blur_rows(v, w, r);
    transpose_into(v, &mut transposed, w, h);
    blur_rows(&mut transposed, h, r);
    transpose_into(&transposed, v, h, w);
}

/// Allocation-light reference path used if parallel transpose storage cannot be reserved.
fn gaussian_sequential(v: &mut [f32], w: usize, h: usize, r: usize) {
    let mut line = Vec::new();
    let mut scratch = Vec::new();
    let len = w.max(h);
    if line.try_reserve_exact(len).is_err() || scratch.try_reserve_exact(len).is_err() {
        return;
    }
    line.resize(len, 0.0);
    scratch.resize(len, 0.0);
    for axis in [false, true] {
        let (lines, line_len) = if axis { (w, h) } else { (h, w) };
        for k in 0..lines {
            for i in 0..line_len {
                line[i] = if axis { v[i * w + k] } else { v[k * w + i] };
            }
            for _ in 0..3 {
                blur_line(&mut line[..line_len], &mut scratch[..line_len], r);
            }
            for i in 0..line_len {
                if axis {
                    v[i * w + k] = line[i];
                } else {
                    v[k * w + i] = line[i];
                }
            }
        }
    }
}

/// The layer's mask values (pixel mask with density × vector mask, each blurred by its feather,
/// exactly as the CPU compositor applies them) as a single-channel f32 surface over `canvas`, or
/// `None` when the layer has no enabled vector mask and no feathered pixel mask (use the pixel
/// mask directly). Values outside the computed area are the surface's default.
pub fn combined_mask(layer: &Layer, canvas: Rect) -> Option<Surface> {
    if !layer.vector_mask.as_ref().is_some_and(|v| v.enabled) && !has_feather(layer) {
        return None;
    }
    Some(cached_mask(cache(), key(layer, canvas), || (build_mask(layer, canvas), layer.mask.as_ref().map(|m| m.surface.clone()))))
}

fn build_mask(layer: &Layer, canvas: Rect) -> Surface {
    let vm = layer.vector_mask.as_ref().filter(|v| v.enabled);
    let pixel = layer.mask.as_ref().filter(|m| m.enabled);
    let (sv, sp) = (vm.map_or(0.0, |v| feather_sigma(v.feather)), pixel.map_or(0.0, |m| feather_sigma(m.feather)));
    let compiled = vm.map(photocraft_vector::CompiledVectorMask::new);
    // Far outside the path the vector mask is constant.
    let far = Rect::from_xywh(canvas.x0 - 1_000_000, canvas.y0 - 1_000_000, 1, 1);
    let v_out = compiled.as_ref().map_or(1.0, |vm| vm.render_sequential(far)[0]);
    let p_def = pixel.map_or(1.0, |m| {
        let d = m.surface.default_pixel().first().copied().unwrap_or(1.0);
        1.0 - m.density * (1.0 - d)
    });
    let grow = |r: Rect, sigma: f32| {
        let m = (sigma * 3.0).ceil() as i32 + 2;
        if r.is_empty() { r } else { Rect::new(r.x0.saturating_sub(m), r.y0.saturating_sub(m), r.x1.saturating_add(m), r.y1.saturating_add(m)) }
    };
    // Where the product can differ from `p_def × v_out`: the path's bounds, plus the pixel
    // mask's painted tiles when the vector mask lets them through outside the path, each grown
    // by its feather.
    let mut area = match vm.and_then(|vm| vm.path.control_bounds()) {
        Some((x0, y0, x1, y1)) => grow(
            Rect::new(
                (x0.floor() as i32).saturating_sub(2),
                (y0.floor() as i32).saturating_sub(2),
                (x1.ceil() as i32).saturating_add(2),
                (y1.ceil() as i32).saturating_add(2),
            ),
            sv,
        ),
        None => Rect::EMPTY,
    };
    if let Some(m) = pixel
        && v_out > 0.0
    {
        let b = grow(m.surface.content_bounds(), sp);
        area = if area.is_empty() {
            b
        } else if b.is_empty() {
            area
        } else {
            area.union(&b)
        };
    }
    let area = area.intersect(&canvas);
    let mut s = Surface::with_default(FORMAT, &[p_def * v_out]);
    if !area.is_empty() {
        let (w, h) = (area.width() as usize, area.height() as usize);
        // A slot initializer must not enter Rayon: another tile requesting the
        // same mask could be stolen onto its stack and wait for its own build.
        let mut v = match &compiled {
            Some(vm) => vm.render_sequential(area),
            None => vec![1.0; w * h],
        };
        gaussian(&mut v, w, h, sv);
        if let Some(m) = pixel {
            let mut pm = Vec::new();
            m.values_into(area, &mut pm);
            gaussian(&mut pm, w, h, sp);
            for (a, b) in v.iter_mut().zip(pm) {
                *a *= b;
            }
        }
        s.write_region(area, &v);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_doc::LayerMask;
    use photocraft_doc::vector::{Knot, Path, Subpath, VectorMask};

    fn pending(c: &Mutex<Cache>, k: u64) -> Arc<OnceLock<Surface>> {
        match lookup(c, k) {
            CachedMask::Pending(slot, _) => slot,
            CachedMask::Hit(_) => panic!("expected a cold mask"),
        }
    }

    /// A built mask (and no pixel-mask input), as `finish_mask`'s build returns it.
    fn test_surface() -> (Surface, Option<Surface>) {
        let mut surface = Surface::new(FORMAT);
        surface.fill_rect(Rect::new(0, 0, 2, 2), &[0.5]);
        (surface, None)
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn cold_callers_share_one_build_without_retaining_a_second_cache() {
        use std::sync::Barrier;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let c = Mutex::new(Cache::default());
        // Acquire all callers' slots before allowing any builder to finish.
        let slots: Vec<_> = (0..8).map(|_| pending(&c, 7)).collect();
        assert!(slots.iter().all(|slot| Arc::ptr_eq(slot, &slots[0])));
        let barrier = Barrier::new(slots.len());
        let builds = AtomicUsize::new(0);
        let outputs = std::thread::scope(|scope| {
            let handles: Vec<_> = slots
                .iter()
                .map(|slot| {
                    let (c, barrier, builds) = (&c, &barrier, &builds);
                    scope.spawn(move || {
                        barrier.wait();
                        finish_mask(c, 7, slot, 0, || {
                            builds.fetch_add(1, Ordering::SeqCst);
                            test_surface()
                        })
                    })
                })
                .collect();
            handles.into_iter().map(|handle| handle.join().unwrap()).collect::<Vec<_>>()
        });
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        assert!(outputs.iter().all(|surface| Arc::ptr_eq(surface.tiles().next().unwrap().1, outputs[0].tiles().next().unwrap().1)));

        // Evicting the retained result cannot make existing callers build again.
        c.lock().unwrap().map.clear();
        let again = finish_mask(&c, 7, &slots[0], 0, || panic!("rebuilt an active slot"));
        assert!(Arc::ptr_eq(again.tiles().next().unwrap().1, outputs[0].tiles().next().unwrap().1));
        drop(slots);
        let next = pending(&c, 8);
        assert_eq!(c.lock().unwrap().in_flight.len(), 1);
        drop(next);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn distinct_keys_build_while_another_key_is_paused() {
        use std::sync::mpsc;
        use std::time::Duration;

        let c = Mutex::new(Cache::default());
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (second_tx, second_rx) = mpsc::channel();
        std::thread::scope(|scope| {
            let c = &c;
            let first = scope.spawn(move || {
                cached_mask(c, 1, || {
                    started_tx.send(()).unwrap();
                    release_rx.recv_timeout(Duration::from_secs(10)).unwrap();
                    test_surface()
                })
            });
            started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            let second = scope.spawn(move || {
                let surface = cached_mask(c, 2, test_surface);
                second_tx.send(surface.sample_channel(0, 0, 0)).unwrap();
            });
            assert!((second_rx.recv_timeout(Duration::from_secs(5)).unwrap() - 0.5).abs() <= 1.0 / f32::from(u16::MAX));
            release_tx.send(()).unwrap();
            first.join().unwrap();
            second.join().unwrap();
        });
    }

    #[test]
    fn obsolete_in_flight_build_does_not_repopulate_the_cache() {
        let c = Mutex::new(Cache::default());
        let old = pending(&c, 1);
        c.lock().unwrap().in_flight.clear();
        let current = pending(&c, 1);
        assert!(!Arc::ptr_eq(&old, &current));
        let old_surface = finish_mask(&c, 1, &old, 0, test_surface);
        assert!(c.lock().unwrap().map.is_empty());
        let current_surface = finish_mask(&c, 1, &current, 0, test_surface);
        assert!(!Arc::ptr_eq(old_surface.tiles().next().unwrap().1, current_surface.tiles().next().unwrap().1));
        assert_eq!(c.lock().unwrap().map.len(), 1);
    }

    #[test]
    fn a_purge_during_a_shared_build_keeps_the_cache_empty() {
        let c = Mutex::new(Cache::default());
        let CachedMask::Pending(slot, generation) = lookup(&c, 1) else { panic!("expected a cold mask") };
        c.lock().unwrap().clear();
        finish_mask(&c, 1, &slot, generation, test_surface);
        assert!(c.lock().unwrap().map.is_empty());
        assert_eq!(c.lock().unwrap().bytes, 0);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn parallel_feathered_pixel_mask_returns_the_same_tiles() {
        let canvas = Rect::new(0, 0, 600, 400);
        let mut layer = Layer::raster("cold shared mask", PixelFormat::RGBA8);
        let mut mask = photocraft_doc::LayerMask::reveal_all();
        mask.surface.fill_rect(canvas, &[0.5]);
        mask.feather = 8.0;
        layer.mask = Some(mask);
        let barrier = std::sync::Barrier::new(8);
        let outputs = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let (layer, barrier) = (&layer, &barrier);
                    scope.spawn(move || {
                        barrier.wait();
                        combined_mask(layer, canvas).unwrap()
                    })
                })
                .collect();
            handles.into_iter().map(|handle| handle.join().unwrap()).collect::<Vec<_>>()
        });
        assert!(outputs.iter().all(|surface| Arc::ptr_eq(surface.tiles().next().unwrap().1, outputs[0].tiles().next().unwrap().1)));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn vector_mask_build_inside_a_small_rayon_pool_completes() {
        use rayon::prelude::*;
        use std::sync::mpsc;
        use std::time::Duration;

        let (tx, rx) = mpsc::channel();
        let handle = std::thread::spawn(move || {
            let canvas = Rect::new(0, 0, 512, 512);
            let mut layer = Layer::raster("rayon vector mask", PixelFormat::RGBA8);
            let mut mask = VectorMask::new(photocraft_vector::shapes::ellipse(8.0, 8.0, 496.0, 496.0));
            mask.feather = 8.0;
            layer.vector_mask = Some(mask);
            let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
            let outputs: Vec<_> = pool.install(|| (0..8).into_par_iter().map(|_| combined_mask(&layer, canvas).unwrap()).collect());
            assert!(outputs.iter().all(|surface| Arc::ptr_eq(surface.tiles().next().unwrap().1, outputs[0].tiles().next().unwrap().1)));
            tx.send(()).unwrap();
        });
        rx.recv_timeout(Duration::from_secs(10)).expect("mask initializer deadlocked inside Rayon");
        handle.join().unwrap();
    }

    fn one_tile(sample: SampleType) -> Surface {
        let mut surface = Surface::with_default(PixelFormat { mode: ColorMode::Grayscale, sample, alpha: false }, &[1.0]);
        surface.fill_rect(Rect::new(0, 0, 1, 1), &[0.0]);
        surface
    }

    #[test]
    fn charges_input_and_output_at_every_depth_including_off_canvas_tiles() {
        let result = one_tile(SampleType::F32);
        for sample in SampleType::ALL {
            let mut input = one_tile(sample);
            input.fill_rect(Rect::new(1024, 1024, 1025, 1025), &[0.0]);
            let tile_pixels = (photocraft_geom::TILE_SIZE * photocraft_geom::TILE_SIZE) as usize;
            let expected = tile_pixels * (SampleType::F32.bytes() + 2 * sample.bytes()) + SampleType::F32.bytes() + sample.bytes();
            let bytes = entry_bytes(&result, Some(&input));
            assert_eq!(bytes, expected, "{sample:?}");
            let mut cache = Cache::default();
            cache.insert(1, result.clone(), None, bytes, bytes * 2, cache.generation);
            cache.insert(2, result.clone(), None, bytes, bytes * 2, cache.generation);
            assert_eq!(cache.bytes, bytes * 2, "shared tiles are conservatively charged per entry");
        }
    }

    #[test]
    fn byte_budget_evicts_the_least_recently_used_entry() {
        let surface = one_tile(SampleType::F32);
        let bytes = entry_bytes(&surface, None);
        let mut cache = Cache::default();
        cache.insert(1, surface.clone(), None, bytes, bytes * 2, cache.generation);
        cache.insert(2, surface.clone(), None, bytes, bytes * 2, cache.generation);
        assert!(cache.get(1).is_some());
        cache.insert(3, surface, None, bytes, bytes * 2, cache.generation);
        assert!(cache.get(2).is_none());
        assert!(cache.get(1).is_some());
        assert!(cache.get(3).is_some());
        assert_eq!(cache.bytes, bytes * 2);
    }

    #[test]
    fn entry_limit_also_bounds_masks_with_no_tiles() {
        let surface = Surface::new(FORMAT);
        let bytes = entry_bytes(&surface, None);
        let mut cache = Cache::default();
        for key in 0..CAPACITY as u64 {
            cache.insert(key, surface.clone(), None, bytes, CACHE_BUDGET, cache.generation);
        }
        assert!(cache.get(0).is_some());
        cache.insert(CAPACITY as u64, surface, None, bytes, CACHE_BUDGET, cache.generation);
        assert_eq!(cache.map.len(), CAPACITY);
        assert!(cache.get(1).is_none());
        assert!(cache.get(0).is_some());
        assert_eq!(cache.bytes, bytes * CAPACITY);
    }

    #[test]
    fn oversized_results_are_usable_without_retention_or_other_eviction() {
        let surface = one_tile(SampleType::F32);
        let bytes = entry_bytes(&surface, None);
        let mut cache = Cache::default();
        let small = Surface::new(FORMAT);
        let small_bytes = entry_bytes(&small, None);
        cache.insert(1, small, None, small_bytes, bytes - 1, cache.generation);
        let tile = surface.tiles().next().unwrap().1.clone();
        let owners = Arc::strong_count(&tile);
        cache.insert(2, surface.clone(), None, bytes, bytes - 1, cache.generation);
        assert!(cache.get(2).is_none());
        assert!(cache.get(1).is_some());
        assert_eq!(cache.bytes, small_bytes);
        assert_eq!(Arc::strong_count(&tile), owners);
        assert_eq!(surface.pixel(0, 0), vec![0.0]);
    }

    #[test]
    fn replacement_and_purge_release_accounted_bytes_and_tile_references() {
        let first = one_tile(SampleType::U8);
        let second = one_tile(SampleType::F32);
        let first_bytes = entry_bytes(&first, None);
        let second_bytes = entry_bytes(&second, None);
        let first_tile = first.tiles().next().unwrap().1.clone();
        let second_tile = second.tiles().next().unwrap().1.clone();
        let mut cache = Cache::default();
        cache.insert(1, first.clone(), None, first_bytes, CACHE_BUDGET, cache.generation);
        assert_eq!(Arc::strong_count(&first_tile), 3);
        cache.insert(1, second.clone(), None, second_bytes, CACHE_BUDGET, cache.generation);
        assert_eq!(cache.bytes, second_bytes);
        assert_eq!(cache.map.len(), 1);
        assert_eq!(Arc::strong_count(&first_tile), 2);
        assert_eq!(Arc::strong_count(&second_tile), 3);
        assert_eq!(cache.clear(), second_bytes);
        assert_eq!(cache.bytes, 0);
        assert!(cache.map.is_empty());
        assert_eq!(Arc::strong_count(&second_tile), 2);
        assert_eq!(cache.clear(), 0);
    }

    #[test]
    fn a_build_started_before_purge_cannot_refill_the_cache() {
        let surface = one_tile(SampleType::F32);
        let bytes = entry_bytes(&surface, None);
        let mut cache = Cache::default();
        assert!(cache.get(1).is_none());
        let generation_before_purge = cache.generation;
        cache.clear();
        cache.insert(1, surface.clone(), None, bytes, CACHE_BUDGET, generation_before_purge);
        assert!(cache.map.is_empty());
        assert_eq!(cache.bytes, 0);
        cache.insert(1, surface, None, bytes, CACHE_BUDGET, cache.generation);
        assert!(cache.get(1).is_some());
    }

    #[test]
    fn pixel_edits_refresh_cached_masks_at_every_depth() {
        let canvas = Rect::new(0, 0, 64, 64);
        for sample in [SampleType::U8, SampleType::U16, SampleType::F32] {
            let mut layer = Layer::raster("edited mask", PixelFormat::RGBA8);
            let mut mask = LayerMask::hide_all();
            mask.surface = Surface::new(PixelFormat { sample, ..FORMAT });
            mask.surface.fill_rect(canvas, &[1.0]);
            mask.feather = 4.0;
            layer.mask = Some(mask);

            let first = combined_mask(&layer, canvas).unwrap();
            assert!((first.sample_channel(32, 32, 0) - 1.0).abs() < 1e-6);
            drop(first);
            // No document/undo snapshot shares the input. Previously this write could
            // reuse the original tile address and return the old cached white mask.
            layer.mask.as_mut().unwrap().surface.fill_rect(canvas, &[0.0]);
            let edited = combined_mask(&layer, canvas).unwrap();
            assert_eq!(edited.sample_channel(32, 32, 0), 0.0, "{sample:?}");

            layer.mask.as_mut().unwrap().surface.fill_rect(canvas, &[0.5]);
            let repainted = combined_mask(&layer, canvas).unwrap();
            assert!((repainted.sample_channel(32, 32, 0) - 0.5).abs() < 0.005, "{sample:?}");
            let warm = combined_mask(&layer, canvas).unwrap();
            assert!(repainted.tiles().zip(warm.tiles()).all(|((_, a), (_, b))| Arc::ptr_eq(a, b)));
        }
    }

    #[test]
    fn parallel_gaussian_matches_the_sequential_reference() {
        let (w, h, sigma): (usize, usize, f32) = (320, 256, 13.0);
        let mut parallel: Vec<_> = (0..w * h).map(|i| ((i * 37 % 1009) as f32) / 1008.0).collect();
        let mut sequential = parallel.clone();
        let ideal = (12.0 * sigma * sigma / 3.0 + 1.0).sqrt();
        let r = (((ideal.floor() as usize) | 1).max(1) - 1) / 2;
        gaussian_sequential(&mut sequential, w, h, r);
        gaussian(&mut parallel, w, h, sigma);
        assert_eq!(parallel, sequential, "parallel blur must retain the exact per-line arithmetic");
    }

    #[test]
    #[ignore = "manual 50 MP sequential-vs-parallel mask blur measurement"]
    fn benchmark_50mp_mask_blur() {
        use std::time::Instant;

        let (w, h, sigma): (usize, usize, f32) = (6336, 7920, 1000.0);
        let ideal = (12.0 * sigma * sigma / 3.0 + 1.0).sqrt();
        let r = (((ideal.floor() as usize) | 1).max(1) - 1) / 2;
        let source: Vec<_> = (0..w * h).map(|i| ((i * 37 % 1009) as f32) / 1008.0).collect();
        let mut sequential = source.clone();
        let start = Instant::now();
        gaussian_sequential(&mut sequential, w, h, r);
        let sequential_time = start.elapsed();
        let mut parallel = source;
        let start = Instant::now();
        gaussian(&mut parallel, w, h, sigma);
        let parallel_time = start.elapsed();
        assert_eq!(parallel, sequential);
        eprintln!("50 MP mask blur: sequential={sequential_time:?}, parallel={parallel_time:?}");
    }

    #[test]
    fn cache_budget_holds_five_50mp_u8_input_u16_result_masks() {
        // Account for whole 256×256 tiles, as entry_bytes does, rather than just logical pixels.
        let (w, h) = (6336usize, 7920usize);
        let tile = photocraft_geom::TILE_SIZE as usize;
        let tile_count = w.div_ceil(tile) * h.div_ceil(tile);
        let tile_pixels = tile * tile;
        let one_entry = tile_count * tile_pixels * (SampleType::U8.bytes() + FORMAT.sample.bytes());
        assert!(one_entry.saturating_mul(5) <= CACHE_BUDGET, "five reported-size masks must fit");

        let mut cache = Cache::default();
        let surface = Surface::new(FORMAT);
        for key in 0..5 {
            cache.insert(key, surface.clone(), None, one_entry, CACHE_BUDGET, cache.generation);
        }
        for key in 0..5 {
            assert!(cache.get(key).is_some(), "mask {key} was evicted before the next plan");
        }
    }

    #[test]
    fn matches_the_cpu_mask_and_is_cached() {
        let mut l = Layer::raster("l", PixelFormat::RGBA8);
        let pts = [(5.0, 5.0), (30.0, 6.0), (20.0, 28.0)];
        let knots = pts.iter().map(|&(x, y)| Knot::corner(x, y)).collect();
        let mut path = Path::default();
        path.subpaths.push(Subpath { knots, closed: true, ..Default::default() });
        let mut vm = VectorMask::new(path);
        vm.density = 0.9;
        l.vector_mask = Some(vm);
        let canvas = Rect::new(0, 0, 40, 32);
        let s = combined_mask(&l, canvas).unwrap();
        let want = photocraft_vector::vector_mask_values(l.vector_mask.as_ref().unwrap(), canvas);
        let mut got = Vec::new();
        s.read_region_into(canvas, &mut got);
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() <= 1.0 / f32::from(u16::MAX));
        }
        let again = combined_mask(&l, canvas).unwrap();
        assert!(s.tiles().zip(again.tiles()).all(|(a, b)| Arc::ptr_eq(a.1, b.1)));
        l.vector_mask = None;
        assert!(combined_mask(&l, canvas).is_none());
    }

    #[test]
    fn hostile_path_coordinates_do_not_overflow() {
        let mut l = Layer::raster("l", PixelFormat::RGBA8);
        let knots = [(-1.0e30, -1.0e30), (1.0e30, 0.0), (0.0, 1.0e30)].iter().map(|&(x, y)| Knot::corner(x, y)).collect();
        let mut path = Path::default();
        path.subpaths.push(Subpath { knots, closed: true, ..Default::default() });
        l.vector_mask = Some(VectorMask::new(path));
        let _ = combined_mask(&l, Rect::new(0, 0, 8, 8));
    }
}
