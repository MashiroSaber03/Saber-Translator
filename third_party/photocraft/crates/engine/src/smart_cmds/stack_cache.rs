//! Stages of smart object renders: the placed source and the result after each visible smart
//! filter, keyed by everything that produced them. A re-render starts from the deepest cached
//! stage, so editing filter `i` only reruns the filters from `i` up: hiding or tweaking the top
//! filter of a long stack doesn't rerun the expensive filters under it.

use std::sync::{Mutex, PoisonError};

use photocraft_color::PixelFormat;
use photocraft_doc::{SmartFilter, SmartObject};
use photocraft_geom::Rect;
use photocraft_raster::Surface;

/// blake3 of a stage's inputs: the source contents, the placement and the filters so far.
pub(super) type StageKey = [u8; 32];

#[derive(Clone)]
pub(super) struct Stage {
    pub surface: Surface,
    /// Where filters repeat edge pixels: the canvas ∪ the placed content.
    pub extent: Rect,
}

struct Entry {
    key: StageKey,
    stage: Stage,
    bytes: usize,
}

/// Most recently used last.
static STAGES: Mutex<Vec<Entry>> = Mutex::new(Vec::new());
const ENTRIES: usize = 64;
/// Accounted tile bytes; a stage larger than this on its own isn't kept.
pub const BUDGET: usize = 768 << 20;

/// The first stage: what placing `bytes` (the source contents) through the smart object's
/// transform, warp or perspective gives at pixel format `fmt` on `canvas`.
pub(super) fn placed_key(bytes: &[u8], fmt: PixelFormat, sm: &SmartObject, canvas: Rect) -> StageKey {
    let mut h = blake3::Hasher::new();
    h.update(bytes);
    for v in sm.transform.m {
        h.update(&v.to_bits().to_le_bytes());
    }
    h.update(format!("{fmt:?}|{:?}|{:?}|{:?}|{canvas:?}", sm.warp, sm.perspective, sm.stack_mode).as_bytes());
    *h.finalize().as_bytes()
}

/// The stage after applying `f` to stage `below`.
pub(super) fn filter_key(below: &StageKey, f: &SmartFilter) -> StageKey {
    let mut h = blake3::Hasher::new();
    h.update(below);
    h.update(format!("{}\0{}\0{:?}\0", f.command, f.params, f.blend).as_bytes());
    h.update(&f.opacity.to_bits().to_le_bytes());
    *h.finalize().as_bytes()
}

pub(super) fn get(key: &StageKey) -> Option<Stage> {
    let mut c = STAGES.lock().unwrap_or_else(PoisonError::into_inner);
    let i = c.iter().position(|e| e.key == *key)?;
    let e = c.remove(i);
    let stage = e.stage.clone();
    c.push(e);
    Some(stage)
}

pub(super) fn put(key: StageKey, stage: &Stage) {
    let bytes = surface_bytes(&stage.surface);
    if bytes > BUDGET {
        return;
    }
    let mut c = STAGES.lock().unwrap_or_else(PoisonError::into_inner);
    c.retain(|e| e.key != key);
    c.push(Entry { key, stage: stage.clone(), bytes });
    while c.len() > ENTRIES || c.iter().map(|e| e.bytes).sum::<usize>() > BUDGET {
        c.remove(0);
    }
}

/// Bytes held by cached stages.
pub fn bytes() -> usize {
    STAGES.lock().unwrap_or_else(PoisonError::into_inner).iter().map(|e| e.bytes).sum()
}

/// Drop every cached stage (Edit › Purge › All). Returns the bytes released.
pub fn purge() -> usize {
    let mut c = STAGES.lock().unwrap_or_else(PoisonError::into_inner);
    let freed = c.iter().map(|e| e.bytes).sum();
    c.clear();
    freed
}

fn surface_bytes(s: &Surface) -> usize {
    // Tiles shared with the document (or another stage) are charged anyway: the cache is what
    // keeps them alive once the document moves on.
    s.tiles().fold(s.default_bytes().len(), |n, (_, t)| n.saturating_add(t.bytes().len()))
}
