//! `Surface::content_bounds` with the per-tile scan cached by copy-on-write tile identity, so
//! layer bounds cost a hash lookup per tile instead of a pixel scan per render tile (the CPU
//! compositor asks for an effect layer's bounds in every tile; the GPU planner every frame).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use photocraft_geom::Rect;
use photocraft_raster::{Surface, Tile};

struct Entry {
    tile: Weak<Tile>,
    /// Content bounds inside the tile (tile-local), for one default pixel.
    default: Box<[u8]>,
    bounds: Option<(u16, u16, u16, u16)>,
}

/// The entries, and the size past which dead ones are dropped (see [`next_sweep`]).
struct Cache {
    entries: HashMap<usize, Entry>,
    sweep_at: usize,
}

const MIN_SWEEP: usize = 1 << 16;

/// The size at which to sweep again after a sweep left `survivors` live entries: twice that (at
/// least [`MIN_SWEEP`]), so a cache full of live tiles isn't swept again on every call.
fn next_sweep(survivors: usize) -> usize {
    survivors.saturating_mul(2).max(MIN_SWEEP)
}

fn cache() -> &'static Mutex<Cache> {
    static C: std::sync::OnceLock<Mutex<Cache>> = std::sync::OnceLock::new();
    C.get_or_init(|| Mutex::new(Cache { entries: HashMap::new(), sweep_at: MIN_SWEEP }))
}

/// `Surface::content_bounds`, cached per tile.
pub fn content_bounds(s: &Surface) -> Rect {
    if s.tile_count() == 0 {
        return Rect::EMPTY;
    }
    let fmt = s.format();
    let mut dp = vec![0u8; fmt.bytes_per_pixel()];
    photocraft_raster::encode_pixel(&fmt, &s.default_pixel(), &mut dp);
    let mut c = cache().lock().unwrap_or_else(|e| e.into_inner());
    if c.entries.len() > c.sweep_at {
        c.entries.retain(|_, e| e.tile.strong_count() > 0);
        c.sweep_at = next_sweep(c.entries.len());
    }
    let mut out = Rect::EMPTY;
    for (coord, t) in s.tiles() {
        let key = Arc::as_ptr(t) as usize;
        let hit = c.entries.get(&key).filter(|e| e.tile.upgrade().is_some_and(|u| Arc::ptr_eq(&u, t)) && *e.default == *dp).map(|e| e.bounds);
        let b = match hit {
            Some(b) => b,
            None => {
                let b = t.content_bounds(&dp);
                c.entries.insert(key, Entry { tile: Arc::downgrade(t), default: dp.clone().into_boxed_slice(), bounds: b });
                b
            }
        };
        if let Some((x0, y0, x1, y1)) = b {
            let o = coord.rect();
            let r = Rect::new(o.x0 + x0 as i32, o.y0 + y0 as i32, o.x0 + x1 as i32, o.y0 + y1 as i32);
            out = if out.is_empty() { r } else { out.union(&r) };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sweeps_again_only_past_twice_the_survivors() {
        // A cache of live tiles past the minimum isn't swept on every call (it used to be).
        assert_eq!(next_sweep(0), MIN_SWEEP);
        assert_eq!(next_sweep(MIN_SWEEP / 2), MIN_SWEEP);
        assert_eq!(next_sweep(MIN_SWEEP), 2 * MIN_SWEEP);
        assert_eq!(next_sweep(usize::MAX), usize::MAX);
    }

    #[test]
    fn a_shared_tile_reads_the_same_bounds_at_every_coordinate() {
        let mut s = Surface::new(photocraft_color::PixelFormat::RGBA8);
        s.fill_rect(Rect::new(-300, 10, 1300, 900), &[0.2, 0.4, 0.6, 1.0]);
        assert_eq!(content_bounds(&s), Rect::new(-300, 10, 1300, 900));
        assert_eq!(content_bounds(&s), s.content_bounds(), "cached");
    }
}
