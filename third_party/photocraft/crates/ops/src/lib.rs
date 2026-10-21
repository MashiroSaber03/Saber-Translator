//! Undo/redo history.
//!
//! Because pixel tiles are `Arc`-shared copy-on-write (see `photocraft-raster`), a full
//! [`Document`] clone costs O(layers + tiles) pointer copies, not pixel copies. History therefore
//! stores whole-document snapshots per transaction, which is simple, obviously correct, and the
//! same approach Photoshop's History panel exposes to users (one state per step).
//!
//! Memory is bounded by `max_states` plus an approximate byte budget. The current document's
//! tiles and tiles only history holds count against it, including selection masks.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::collections::{HashSet, VecDeque};
use std::sync::Arc;

use photocraft_doc::{Document, LayerId};

mod tiles;

/// The layers a state targeted: the active ("key") layer and every selected layer. Undo and redo
/// bring them back with the state's document, as the reference app does. Selecting layers is not
/// a step of its own, so the session updates the current state's target with
/// [`History::set_current_layers`] just before recording the next step (#1356).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LayerTarget {
    pub active: Option<LayerId>,
    pub selected: Vec<LayerId>,
}

#[derive(Clone, Debug)]
pub struct HistoryState {
    pub label: String,
    pub doc: Arc<Document>,
    /// The layers targeted when this state was created.
    pub layers: LayerTarget,
}

#[derive(Clone, Debug)]
pub struct History {
    /// Past states; the last one is the state *before* the current document.
    undo: VecDeque<HistoryState>,
    redo: Vec<HistoryState>,
    /// Lazily initialized when a byte budget is used; unlimited histories pay no counting cost.
    tiles: Option<tiles::Tiles>,
    pub max_states: usize,
    /// Pixel memory budget in bytes for the current document plus the tiles only history holds
    /// (0 = unlimited). [`History::trim`] drops the oldest states beyond it.
    pub max_bytes: usize,
    /// Label of the step that produced the current document.
    current_label: String,
    /// The layers targeted when the current document was created (opened or edited).
    current_layers: LayerTarget,
}

impl Default for History {
    fn default() -> Self {
        Self::new(50)
    }
}

impl History {
    pub fn new(max_states: usize) -> Self {
        Self {
            undo: VecDeque::new(),
            redo: Vec::new(),
            tiles: None,
            max_states: max_states.max(1),
            max_bytes: 0,
            current_label: "Open".into(),
            current_layers: LayerTarget::default(),
        }
    }

    /// Record that `before` was replaced by a new current document via step `label`, which left
    /// `layers` targeted.
    pub fn record(&mut self, label: impl Into<String>, before: Arc<Document>, layers: LayerTarget) {
        if self.max_bytes == 0 {
            self.tiles = None;
        }
        if let Some(tiles) = &mut self.tiles {
            tiles.add(&before);
        }
        let prev_label = std::mem::replace(&mut self.current_label, label.into());
        let prev_layers = std::mem::replace(&mut self.current_layers, layers);
        self.undo.push_back(HistoryState { label: prev_label, doc: before, layers: prev_layers });
        self.clear_redo();
        while self.undo.len() > self.max_states {
            if let Some(state) = self.undo.pop_front()
                && let Some(tiles) = &mut self.tiles
            {
                tiles.remove(&state.doc);
            }
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }
    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }
    pub fn undo_label(&self) -> Option<&str> {
        self.can_undo().then_some(self.current_label.as_str())
    }
    pub fn redo_label(&self) -> Option<&str> {
        self.redo.last().map(|s| s.label.as_str())
    }
    /// Labels of the redo states, the next redo first (the History panel's greyed rows).
    pub fn redo_labels(&self) -> impl Iterator<Item = &str> {
        self.redo.iter().rev().map(|s| s.label.as_str())
    }

    /// Set the layers the current document targets from its creation on: when it is first
    /// opened, or when its step coalesces or changes the selection after recording.
    pub fn set_current_layers(&mut self, layers: LayerTarget) {
        self.current_layers = layers;
    }

    /// Undo: `current` becomes redoable; returns the document to make current and the layers it
    /// targeted when it was created.
    pub fn undo(&mut self, current: Arc<Document>) -> Option<(Arc<Document>, LayerTarget)> {
        let prev = self.undo.pop_back()?;
        if let Some(tiles) = &mut self.tiles {
            tiles.remove(&prev.doc);
            tiles.add(&current);
        }
        let label = std::mem::replace(&mut self.current_label, prev.label);
        let layers = std::mem::replace(&mut self.current_layers, prev.layers.clone());
        self.redo.push(HistoryState { label, doc: current, layers });
        Some((prev.doc, prev.layers))
    }

    /// Redo: the mirror of [`Self::undo`].
    pub fn redo(&mut self, current: Arc<Document>) -> Option<(Arc<Document>, LayerTarget)> {
        let next = self.redo.pop()?;
        if let Some(tiles) = &mut self.tiles {
            tiles.remove(&next.doc);
            tiles.add(&current);
        }
        let label = std::mem::replace(&mut self.current_label, next.label);
        let layers = std::mem::replace(&mut self.current_layers, next.layers.clone());
        self.undo.push_back(HistoryState { label, doc: current, layers });
        Some((next.doc, next.layers))
    }

    /// Entries for a History panel: past labels oldest→newest, then the current label.
    pub fn entries(&self) -> Vec<String> {
        self.undo.iter().map(|s| s.label.clone()).chain(std::iter::once(self.current_label.clone())).collect()
    }

    /// Document of past entry `i`, indexed like [`History::entries`] (0 = oldest). The last entry is
    /// the current document, which the history does not hold, so it (and any index past it) is `None`.
    pub fn state(&self, i: usize) -> Option<Arc<Document>> {
        self.undo.get(i).map(|s| s.doc.clone())
    }

    /// Number of past states (entries before the current one).
    pub fn past_len(&self) -> usize {
        self.undo.len()
    }

    /// Rename the current step (e.g. after folding several steps into one with [`Self::purge_last`]).
    pub fn set_current_label(&mut self, label: impl Into<String>) {
        self.current_label = label.into();
    }

    /// Forget the most recent undo state and every redo state (Edit › Purge › Undo): the
    /// last step can no longer be undone and its pixels are released.
    pub fn purge_last(&mut self) -> bool {
        self.clear_redo();
        let Some(state) = self.undo.pop_back() else { return false };
        if let Some(tiles) = &mut self.tiles {
            tiles.remove(&state.doc);
        }
        true
    }

    /// Forget the redo states, e.g. after undoing half of a compound step that failed.
    pub fn clear_redo(&mut self) {
        if let Some(tiles) = &mut self.tiles {
            for state in &self.redo {
                tiles.remove(&state.doc);
            }
        }
        self.redo.clear();
    }

    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
        self.tiles = None;
    }

    /// Approximate unique pixel bytes held by history (tiles not shared with `current`).
    pub fn unique_bytes(&self, current: &Document) -> usize {
        let mut seen = HashSet::new();
        tile_bytes(current, &mut seen);
        self.undo.iter().chain(self.redo.iter()).map(|s| tile_bytes(&s.doc, &mut seen)).sum()
    }

    /// Pixel bytes of `current` plus the tiles only history holds (what [`History::trim`]
    /// bounds by [`History::max_bytes`]).
    pub fn pixel_bytes(&self, current: &Document) -> usize {
        if let Some(tiles) = &self.tiles {
            return usize::try_from(tiles.bytes_with(current)).unwrap_or(usize::MAX);
        }
        let mut seen = HashSet::new();
        let own = tile_bytes(current, &mut seen);
        self.undo.iter().chain(self.redo.iter()).fold(own, |n, s| n.saturating_add(tile_bytes(&s.doc, &mut seen)))
    }

    /// Keep pixel memory within [`History::max_bytes`]: the current document's tiles plus the
    /// tiles only history holds (newest states first). The oldest undo states that don't fit are
    /// dropped; the most recent one is always kept so the last step can be undone. Returns how
    /// many states were dropped.
    pub fn trim(&mut self, current: &Document) -> usize {
        if self.max_bytes == 0 || self.undo.len() <= 1 {
            return 0;
        }
        let tiles = self.tiles.get_or_insert_with(|| {
            let mut tiles = tiles::Tiles::default();
            for state in self.undo.iter().chain(&self.redo) {
                tiles.add(&state.doc);
            }
            tiles
        });
        if tiles.fits(current, self.max_bytes) {
            return 0;
        }
        let mut seen = HashSet::new();
        let mut total = tile_bytes(current, &mut seen);
        for s in self.redo.iter().rev() {
            total = total.saturating_add(tile_bytes(&s.doc, &mut seen));
        }
        let mut keep = 0;
        for s in self.undo.iter().rev() {
            total = total.saturating_add(tile_bytes(&s.doc, &mut seen));
            if total > self.max_bytes && keep >= 1 {
                break;
            }
            keep += 1;
        }
        let drop = self.undo.len() - keep;
        for state in self.undo.drain(..drop) {
            tiles.remove(&state.doc);
        }
        drop
    }
}

/// Bytes of the pixel tiles of `doc` (layers, masks, alpha channels, selection) not already in `seen`.
fn tile_bytes(doc: &Document, seen: &mut HashSet<usize>) -> usize {
    let mut n = 0usize;
    tiles::surfaces(doc, |surface| {
        for (_, tile) in surface.tiles() {
            if seen.insert(Arc::as_ptr(tile) as usize) {
                n = n.saturating_add(tile.bytes().len());
            }
        }
    });
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_doc::{Color, ColorMode, Layer, SampleType, Size};

    fn base() -> Document {
        Document::with_background("h", Size::new(64, 64), ColorMode::Rgb, SampleType::U8, Color::WHITE)
    }

    /// Apply an edit the way the engine does: snapshot, mutate a clone, record.
    fn edit(h: &mut History, cur: &mut Arc<Document>, label: &str, f: impl FnOnce(&mut Document)) {
        let before = cur.clone();
        let mut d = (**cur).clone();
        f(&mut d);
        *cur = Arc::new(d);
        h.record(label, before, LayerTarget::default());
    }

    #[test]
    fn undo_redo_cycle() {
        let mut h = History::default();
        let mut cur = Arc::new(base());
        edit(&mut h, &mut cur, "New Layer", |d| {
            d.insert_above(None, Layer::raster("L", d.pixel_format()));
        });
        assert_eq!(cur.layers.len(), 2);
        assert_eq!(h.undo_label(), Some("New Layer"));
        cur = h.undo(cur).unwrap().0;
        assert_eq!(cur.layers.len(), 1);
        assert_eq!(h.redo_label(), Some("New Layer"));
        cur = h.redo(cur).unwrap().0;
        assert_eq!(cur.layers.len(), 2);
        assert!(h.redo(cur.clone()).is_none());
    }

    #[test]
    fn undo_and_redo_return_the_layers_each_state_targeted_when_created() {
        let mut h = History::default();
        let open = Arc::new(base());
        let bg = open.layers[0].id;
        let target = |id: LayerId| LayerTarget { active: Some(id), selected: vec![id] };
        h.set_current_layers(target(bg));
        // New Layer targets the new layer; the background is then selected (not a step) and filled.
        let mut d = (*open).clone();
        let new = d.insert_above(None, Layer::raster("L", d.pixel_format()));
        let layered = Arc::new(d);
        h.record("New Layer", open, target(new));
        let mut d = (*layered).clone();
        d.name = "filled".into();
        h.record("Fill", layered, target(bg));
        let (d, layers) = h.undo(Arc::new(d)).unwrap();
        assert_eq!(layers, target(new), "New Layer's state targeted the new layer when created");
        let (d, layers) = h.undo(d).unwrap();
        assert_eq!(layers, target(bg), "the opened document's target");
        let (d, layers) = h.redo(d).unwrap();
        assert_eq!(layers, target(new));
        let (_, layers) = h.redo(d).unwrap();
        assert_eq!(layers, target(bg), "Fill's state targeted the background");
    }

    #[test]
    fn state_accessor_matches_entries() {
        let mut h = History::default();
        let mut cur = Arc::new(base());
        let open = cur.clone();
        edit(&mut h, &mut cur, "A", |d| d.name = "a".into());
        edit(&mut h, &mut cur, "B", |d| d.name = "b".into());
        assert_eq!(h.entries().len(), h.past_len() + 1);
        assert!(Arc::ptr_eq(&h.state(0).unwrap(), &open));
        assert_eq!(h.state(1).unwrap().name, "a");
        assert!(h.state(2).is_none(), "the current document is not held");
        cur = h.undo(cur).unwrap().0;
        h.undo(cur).unwrap();
        assert_eq!(h.redo_labels().collect::<Vec<_>>(), ["A", "B"], "next redo first");
    }

    #[test]
    fn new_edit_clears_redo() {
        let mut h = History::default();
        let mut cur = Arc::new(base());
        edit(&mut h, &mut cur, "A", |d| d.name = "a".into());
        cur = h.undo(cur).unwrap().0;
        edit(&mut h, &mut cur, "B", |d| d.name = "b".into());
        assert!(!h.can_redo());
        assert_eq!(h.entries(), vec!["Open".to_string(), "B".to_string()]);
    }

    #[test]
    fn max_states_bound() {
        let mut h = History::new(3);
        let mut cur = Arc::new(base());
        for i in 0..10 {
            edit(&mut h, &mut cur, &format!("step {i}"), |d| d.name = format!("{i}"));
        }
        let mut n = 0;
        while let Some((d, _)) = h.undo(cur.clone()) {
            cur = d;
            n += 1;
        }
        assert_eq!(n, 3);
        assert_eq!(cur.name, "6");
    }

    #[test]
    fn byte_budget_drops_oldest_states_but_keeps_one() {
        let mut h = History::new(50);
        let mut cur = Arc::new(base());
        let tile = 256 * 256 * 4;
        // Each step repaints the background's tile: one unique tile per state.
        for i in 0..6 {
            edit(&mut h, &mut cur, &format!("paint {i}"), |d| {
                let id = d.layers[0].id;
                d.layer_mut(id).unwrap().surface_mut().unwrap().write_pixel(1, 1, &[i as f32 / 8.0, 0.0, 0.0, 1.0]);
            });
        }
        assert_eq!(h.trim(&cur), 0, "unlimited by default");
        assert_eq!(h.past_len(), 6);
        // The current tile plus three history tiles fit.
        h.max_bytes = 4 * tile;
        assert_eq!(h.trim(&cur), 3);
        assert_eq!(h.past_len(), 3);
        assert_eq!(h.unique_bytes(&cur), 3 * tile);
        assert_eq!(h.pixel_bytes(&cur), 4 * tile, "the current tile plus history's");
        assert_eq!(h.entries()[0], "paint 2");
        // A budget smaller than one state still keeps the last step undoable.
        h.max_bytes = 1;
        h.trim(&cur);
        assert_eq!(h.past_len(), 1);
        assert!(h.undo(cur.clone()).is_some());
    }

    #[test]
    fn history_shares_untouched_tiles() {
        let mut h = History::default();
        let mut cur = Arc::new(base());
        assert_eq!(h.unique_bytes(&cur), 0);
        // rename only: no pixel changes, history holds no unique tiles
        edit(&mut h, &mut cur, "Rename", |d| d.name = "x".into());
        assert_eq!(h.unique_bytes(&cur), 0);
        // paint one pixel: exactly one tile becomes unique to history
        edit(&mut h, &mut cur, "Paint", |d| {
            let id = d.layers[0].id;
            d.layer_mut(id).unwrap().surface_mut().unwrap().write_pixel(1, 1, &[0.0, 0.0, 0.0, 1.0]);
        });
        assert_eq!(h.unique_bytes(&cur), 256 * 256 * 4);
    }

    #[test]
    fn selection_tiles_count_current_and_history_without_double_counting() {
        use photocraft_doc::{PixelFormat, Surface};
        for depth in SampleType::ALL {
            let mut doc = Document::new("selection", Size::new(300, 64), ColorMode::Rgb, depth);
            let mut selection = Surface::new(PixelFormat::new(ColorMode::Grayscale, depth, false));
            selection.write_pixel(1, 1, &[1.0]);
            selection.write_pixel(257, 1, &[1.0]);
            doc.selection = Some(selection);
            let mut cur = Arc::new(doc);
            let tile = match depth {
                SampleType::U8 => 65_536,
                SampleType::U16 => 131_072,
                SampleType::F32 => 262_144,
            };
            let mut h = History { max_bytes: usize::MAX, ..History::default() };
            assert_eq!(h.pixel_bytes(&cur), 2 * tile, "current selection counts without any history");
            assert_eq!(h.unique_bytes(&cur), 0);
            edit(&mut h, &mut cur, "Rename", |doc| doc.name = "renamed".into());
            assert_eq!(h.pixel_bytes(&cur), 2 * tile, "the shared selection tiles count once");
            edit(&mut h, &mut cur, "Selection", |doc| {
                doc.selection.as_mut().unwrap().write_pixel(1, 1, &[0.5]);
            });
            assert_eq!(h.pixel_bytes(&cur), 3 * tile, "only the changed selection tile is copied");
            assert_eq!(h.unique_bytes(&cur), tile);
            h.trim(&cur);
            assert_counter(&h, &cur);
            let selected_value = cur.selection.as_ref().unwrap().sample_channel(1, 1, 0);
            edit(&mut h, &mut cur, "Deselect", |doc| doc.selection = None);
            assert_eq!(h.pixel_bytes(&cur), 3 * tile, "deselecting leaves the previous masks in history");
            assert_eq!(h.unique_bytes(&cur), 3 * tile);
            assert_counter(&h, &cur);
            cur = h.undo(cur).unwrap().0;
            assert_eq!(cur.selection.as_ref().unwrap().sample_channel(1, 1, 0), selected_value);
            assert_eq!(h.pixel_bytes(&cur), 3 * tile);
            assert_eq!(h.unique_bytes(&cur), tile);
            assert_counter(&h, &cur);
            // Coalescing can change the current mask without recording another state.
            Arc::make_mut(&mut cur).selection.as_mut().unwrap().write_pixel(257, 1, &[0.25]);
            assert_eq!(h.pixel_bytes(&cur), 4 * tile);
            assert_eq!(h.trim(&cur), 0);
            assert_counter(&h, &cur);
            cur = h.redo(cur).unwrap().0;
            assert!(cur.selection.is_none());
            assert_eq!(h.pixel_bytes(&cur), 4 * tile);
            assert_counter(&h, &cur);
        }
    }

    #[test]
    fn selection_byte_budget_drops_old_masks_and_keeps_the_last_undo() {
        use photocraft_doc::{PixelFormat, Surface};
        for depth in SampleType::ALL {
            let mut doc = Document::new("selection", Size::new(64, 64), ColorMode::Rgb, depth);
            doc.selection = Some(Surface::new(PixelFormat::new(ColorMode::Grayscale, depth, false)));
            let mut cur = Arc::new(doc);
            let tile = 256 * 256 * depth.bytes();
            let mut h = History::default();
            for i in 1..=6 {
                edit(&mut h, &mut cur, "Selection", |doc| {
                    doc.selection.as_mut().unwrap().write_pixel(1, 1, &[i as f32 / 8.0]);
                });
            }
            assert_eq!(h.pixel_bytes(&cur), 6 * tile);
            h.max_bytes = 3 * tile;
            assert_eq!(h.trim(&cur), 4);
            assert_eq!(h.past_len(), 2);
            assert_eq!(h.pixel_bytes(&cur), 3 * tile);
            assert_eq!(h.unique_bytes(&cur), 2 * tile);
            assert_counter(&h, &cur);
            h.max_bytes = 1;
            assert_eq!(h.trim(&cur), 1);
            assert_eq!(h.past_len(), 1, "the most recent undo survives even below one mask's size");
            assert_eq!(h.pixel_bytes(&cur), 2 * tile);
            cur = h.undo(cur).unwrap().0;
            assert_counter(&h, &cur);
        }
    }

    fn assert_counter(h: &History, current: &Document) {
        let counter = h.tiles.as_ref().expect("budget initializes the counter");
        let total = tile_bytes(current, &mut HashSet::new()).saturating_add(h.unique_bytes(current));
        assert_eq!(counter.bytes_with(current), total as u128);
        assert_eq!(h.pixel_bytes(current), total);
        let mut seen = HashSet::new();
        let retained: usize = h.undo.iter().chain(&h.redo).map(|s| tile_bytes(&s.doc, &mut seen)).sum();
        let empty = Document::new("empty", Size::new(1, 1), ColorMode::Rgb, SampleType::U8);
        assert_eq!(counter.bytes_with(&empty), retained as u128);
    }

    fn video_document(depth: SampleType, frames: usize) -> Document {
        let mut doc = Document::new("video", Size::new(64, 64), ColorMode::Rgb, depth);
        let mut layer = Layer::raster("Video", doc.pixel_format());
        let footage: Vec<_> = (0..frames)
            .map(|i| {
                let mut surface = photocraft_doc::Surface::new(doc.pixel_format());
                surface.write_pixel(1, 1, &[i as f32 / frames as f32, 0.0, 0.0, 1.0]);
                surface
            })
            .collect();
        layer.content = photocraft_doc::LayerContent::Raster(footage[0].clone());
        layer.video = Some(photocraft_doc::VideoData::new(footage, 24.0));
        doc.layers.push(layer);
        doc
    }

    #[test]
    fn video_memory_counts_originals_and_deduplicates_displayed_frames() {
        for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
            let mut doc = video_document(depth, 3);
            let tile = doc.layers[0].surface().unwrap().tiles().next().unwrap().1.bytes().len();
            let video = doc.layers[0].video.as_mut().unwrap();
            video.original_frames = Some(video.frames.clone());
            // Original footage and the displayed frame alias the stack, except this painted frame.
            video.frames[2].write_pixel(1, 1, &[0.0, 1.0, 0.0, 1.0]);
            let mut cur = Arc::new(doc);
            let mut h = History { max_bytes: usize::MAX, ..History::default() };
            assert_eq!(h.pixel_bytes(&cur), 4 * tile);
            for name in ["A", "B"] {
                edit(&mut h, &mut cur, "Rename", |doc| doc.name = name.into());
                h.trim(&cur);
            }
            assert_counter(&h, &cur);
            assert_eq!(h.pixel_bytes(&cur), 4 * tile);
            assert_eq!(h.unique_bytes(&cur), 0);
            // Rasterizing keeps the displayed frame; the rest of the clip survives only in undo.
            edit(&mut h, &mut cur, "Rasterize", |doc| doc.layers[0].video = None);
            assert_eq!(h.unique_bytes(&cur), 3 * tile);
            assert_eq!(h.pixel_bytes(&cur), 4 * tile);
            assert_counter(&h, &cur);
            cur = h.undo(cur).unwrap().0;
            assert_eq!(h.unique_bytes(&cur), 0);
            assert_counter(&h, &cur);
            cur = h.redo(cur).unwrap().0;
            assert_eq!(h.unique_bytes(&cur), 3 * tile);
            assert_counter(&h, &cur);
            h.clear();
            assert_eq!(h.pixel_bytes(&cur), tile);
        }
    }

    #[test]
    fn video_memory_budget_evicts_deleted_off_playhead_frames() {
        for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
            let mut cur = Arc::new(video_document(depth, 6));
            let tile = cur.layers[0].surface().unwrap().tiles().next().unwrap().1.bytes().len();
            let mut h = History::default();
            for _ in 0..4 {
                edit(&mut h, &mut cur, "Delete Frame", |doc| {
                    doc.layers[0].video.as_mut().unwrap().frames.remove(1);
                });
            }
            assert_eq!(h.pixel_bytes(&cur), 6 * tile);
            assert_eq!(h.unique_bytes(&cur), 4 * tile);
            h.max_bytes = 3 * tile;
            assert_eq!(h.trim(&cur), 3);
            assert_eq!(h.past_len(), 1);
            assert_eq!(h.pixel_bytes(&cur), 3 * tile);
            assert_eq!(h.unique_bytes(&cur), tile);
            assert_counter(&h, &cur);
            cur = h.undo(cur).unwrap().0;
            assert_eq!(cur.layers[0].video.as_ref().unwrap().frames.len(), 3);
            assert_counter(&h, &cur);
            cur = h.redo(cur).unwrap().0;
            assert_eq!(cur.layers[0].video.as_ref().unwrap().frames.len(), 2);
            assert_counter(&h, &cur);
            // A budget below the remaining clip still keeps the latest step undoable.
            h.max_bytes = 1;
            assert_eq!(h.trim(&cur), 0);
            assert!(h.can_undo());
        }
    }

    /// A pattern fill or layer style copies its pattern into the document, so the pattern's tiles
    /// count like layer pixels, shared ones once; undoing that step leaves them to the redo states.
    #[test]
    fn document_patterns_count_as_document_pixels() {
        use photocraft_doc::{Pattern, Surface};
        for depth in SampleType::ALL {
            let tile = 4 * 256 * 256 * depth.bytes();
            let mut cur = Arc::new(Document::new("patterns", Size::new(64, 64), ColorMode::Rgb, depth));
            let mut h = History { max_bytes: usize::MAX, ..History::default() };
            for name in ["A", "B"] {
                edit(&mut h, &mut cur, "Rename", |doc| doc.name = name.into());
                h.trim(&cur);
            }
            let mut surface = Surface::new(cur.pixel_format());
            surface.write_pixel(1, 1, &[0.0, 0.0, 1.0, 1.0]);
            let pattern = Pattern::new("Pattern", surface, 4, 4);
            edit(&mut h, &mut cur, "Pattern Fill", |doc| doc.patterns.push(pattern));
            assert_eq!(h.pixel_bytes(&cur), tile, "the document's pattern");
            assert_eq!(h.unique_bytes(&cur), 0);
            assert_counter(&h, &cur);
            edit(&mut h, &mut cur, "Rename", |doc| doc.name = "C".into());
            assert_eq!(h.pixel_bytes(&cur), tile, "shared with history, it counts once");
            assert_counter(&h, &cur);
            for _ in 0..2 {
                cur = h.undo(cur).unwrap().0;
                assert_counter(&h, &cur);
            }
            assert!(cur.patterns.is_empty());
            assert_eq!(h.unique_bytes(&cur), tile, "only the redo states hold the pattern");
            assert_eq!(h.pixel_bytes(&cur), tile);
            cur = h.redo(cur).unwrap().0;
            assert_eq!(h.unique_bytes(&cur), 0);
            assert_counter(&h, &cur);
            h.clear();
            assert_eq!(h.pixel_bytes(&cur), tile);
        }
    }

    #[test]
    fn counter_tracks_eviction_undo_redo_branching_purge_and_clones() {
        for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
            let mut cur = Arc::new(Document::with_background("h", Size::new(300, 64), ColorMode::Rgb, depth, Color::WHITE));
            Arc::make_mut(&mut cur).selection = Some(photocraft_doc::Surface::new(photocraft_doc::PixelFormat::GRAY8));
            let mut h = History::new(4);
            h.max_bytes = usize::MAX;
            for i in 0..8 {
                edit(&mut h, &mut cur, "Paint", |doc| {
                    doc.layers[0].surface_mut().unwrap().write_pixel(1, 1, &[i as f32 / 10.0, 0.0, 0.0, 1.0]);
                    doc.selection.as_mut().unwrap().write_pixel(1, 1, &[i as f32 / 10.0]);
                });
                h.trim(&cur);
                if i >= 1 {
                    assert_counter(&h, &cur);
                }
            }
            cur = h.undo(cur).unwrap().0;
            assert_counter(&h, &cur);
            cur = h.undo(cur).unwrap().0;
            assert_counter(&h, &cur);
            cur = h.redo(cur).unwrap().0;
            assert_counter(&h, &cur);
            let copy = h.clone();
            edit(&mut h, &mut cur, "Branch", |doc| doc.name = "branch".into());
            assert!(!h.can_redo());
            assert_counter(&h, &cur);
            assert_counter(&copy, &cur);
            cur = h.undo(cur).unwrap().0;
            h.clear_redo();
            assert_counter(&h, &cur);
            h.purge_last();
            assert_counter(&h, &cur);
            h.clear();
            assert!(h.tiles.is_none());
            assert_eq!(h.pixel_bytes(&cur), tile_bytes(&cur, &mut HashSet::new()));
        }
    }

    #[test]
    fn shared_masks_channels_and_repeated_snapshot_references_count_once() {
        use photocraft_doc::{AlphaChannel, LayerMask, PixelFormat, Surface};
        let mut doc = base();
        let mut gray = Surface::new(PixelFormat::GRAY8);
        gray.write_pixel(1, 1, &[1.0]);
        doc.layers[0].mask = Some(LayerMask { surface: gray.clone(), ..LayerMask::reveal_all() });
        doc.channels.push(AlphaChannel::new("alpha", gray.clone()));
        doc.selection = Some(gray.clone());
        doc.quick_mask = Some(AlphaChannel::new("quick", gray));
        doc.layers.push(doc.layers[0].clone());
        let mut cur = Arc::new(doc);
        let bytes = tile_bytes(&cur, &mut HashSet::new());
        let mut h = History::new(50);
        // Counting current separately would double count its tiles; the exact union fits.
        h.max_bytes = bytes;
        for _ in 0..6 {
            h.record("Same", cur.clone(), LayerTarget::default());
            assert_eq!(h.trim(&cur), 0);
        }
        assert_counter(&h, &cur);
        for _ in 0..3 {
            cur = h.undo(cur).unwrap().0;
            assert_counter(&h, &cur);
        }
        h.clear_redo();
        h.purge_last();
        assert_counter(&h, &cur);
        assert_eq!(h.pixel_bytes(&cur), bytes);
    }

    #[test]
    fn threshold_rechecks_coalesced_current_and_changed_budget() {
        let mut h = History::new(50);
        let mut cur = Arc::new(base());
        h.max_bytes = usize::MAX;
        for i in 0..6 {
            edit(&mut h, &mut cur, "Paint", |doc| {
                doc.layers[0].surface_mut().unwrap().write_pixel(1, 1, &[i as f32 / 10.0, 0.0, 0.0, 1.0]);
            });
            h.trim(&cur);
        }
        // A coalesced edit may change current without recording another history state.
        Arc::make_mut(&mut cur).layers[0].surface_mut().unwrap().write_pixel(300, 1, &[1.0; 4]);
        let tile = 256 * 256 * 4;
        h.max_bytes = 4 * tile;
        assert_eq!(h.trim(&cur), 4);
        assert_eq!(h.past_len(), 2);
        assert_counter(&h, &cur);
        cur = h.undo(cur).unwrap().0;
        assert_counter(&h, &cur);
        h.max_bytes = 0;
        edit(&mut h, &mut cur, "Unlimited", |doc| doc.name = "u".into());
        assert!(h.tiles.is_none());
        h.max_bytes = 1;
        h.trim(&cur);
        assert_eq!(h.past_len(), 1, "always retain the most recent undo");
        assert_counter(&h, &cur);
    }

    #[test]
    fn transferred_tiles_cannot_make_the_budget_check_underestimate_current() {
        use photocraft_doc::{PixelFormat, Surface};
        let before = Arc::new(base());
        let mut current = (*before).clone();
        let mut large = Surface::new(PixelFormat::new(ColorMode::Cmyk, SampleType::F32, true));
        large.write_pixel(256, 0, &[1.0; 5]);
        current.layers[0].surface_mut().unwrap().put_tiles(large.take_tiles(photocraft_doc::Rect::from_xywh(256, 0, 1, 1)));
        let mut h = History::new(50);
        h.max_bytes = tile_bytes(&current, &mut HashSet::new()) - 1;
        h.record("A", before.clone(), LayerTarget::default());
        h.record("B", before, LayerTarget::default());
        assert_eq!(h.trim(&current), 1, "current alone exceeds the budget; only the mandatory undo survives");
        assert_counter(&h, &current);
    }
}
