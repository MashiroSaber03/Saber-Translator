//! Tile references owned by history snapshots. The snapshots keep these addresses alive;
//! COW edits cannot change a retained tile in place. Count occurrences (including aliases
//! within one document), so removing a state releases a tile only after its last reference.
use photocraft_doc::{Document, Surface};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Clone, Debug, Default)]
pub(crate) struct Tiles {
    refs: HashMap<usize, (u128, usize)>,
    bytes: u128,
}

pub(crate) fn surfaces(doc: &Document, mut visit: impl FnMut(&Surface)) {
    for (_, _, layer) in doc.walk() {
        if let Some(surface) = layer.surface() {
            visit(surface);
        }
        if let Some(mask) = &layer.mask {
            visit(&mask.surface);
        }
        if let Some(video) = &layer.video {
            // Undo retains the whole clip, not only the displayed frame. Visit aliases too:
            // Tiles tracks every owner and counts their shared allocations only once.
            for frame in video.frames.iter().chain(video.original_frames.iter().flatten()) {
                visit(frame);
            }
        }
    }
    for channel in doc.channels.iter().chain(&doc.quick_mask) {
        visit(&channel.surface);
    }
    // A pattern fill or layer style copies its pattern into the document.
    for pattern in &doc.patterns {
        visit(&pattern.surface);
    }
    if let Some(selection) = &doc.selection {
        visit(selection);
    }
}

impl Tiles {
    pub(crate) fn add(&mut self, doc: &Document) {
        surfaces(doc, |surface| {
            for (_, tile) in surface.tiles() {
                let count = self.refs.entry(Arc::as_ptr(tile) as usize).or_insert_with(|| {
                    self.bytes += tile.bytes().len() as u128;
                    (0, tile.bytes().len())
                });
                count.0 += 1;
            }
        });
    }

    pub(crate) fn remove(&mut self, doc: &Document) {
        surfaces(doc, |surface| {
            for (_, tile) in surface.tiles() {
                let key = Arc::as_ptr(tile) as usize;
                if let Some(count) = self.refs.get_mut(&key) {
                    count.0 -= 1;
                    if count.0 == 0 {
                        self.bytes -= count.1 as u128;
                        self.refs.remove(&key);
                    }
                }
            }
        });
    }

    /// Retained bytes are incremental; only current tiles need examining below the limit.
    /// Current can change without recording (coalescing), and tile transfers can carry a
    /// different byte length, so a tile-count estimate must not underestimate its memory.
    pub(crate) fn fits(&self, current: &Document, budget: usize) -> bool {
        self.bytes <= budget as u128 && self.bytes_with(current) <= budget as u128
    }

    pub(crate) fn bytes_with(&self, current: &Document) -> u128 {
        let mut total = self.bytes;
        let mut seen = HashSet::new();
        surfaces(current, |surface| {
            for (_, tile) in surface.tiles() {
                let key = Arc::as_ptr(tile) as usize;
                if !self.refs.contains_key(&key) && seen.insert(key) {
                    total += tile.bytes().len() as u128;
                }
            }
        });
        total
    }
}
