//! The developer tile grid: the active layer's storage tiles over the canvas, turned on with the
//! control channel's `ui.set {"tileGrid": true}` ([`crate::view_cmds::ViewOptions::tile_grid`]).
//! It shows how sparse a layer is, which tiles copy-on-write shares, and which ones `.pcraft`
//! saves once.
//!
//! A layer's pixels live in a sparse [`Surface`] of [`TILE_SIZE`] tiles; cells with no stored
//! tile read as the surface's default pixel and cost nothing. The overlay draws the tile grid
//! faintly over the document, fills every stored tile, and labels it `tx,ty` once it is large
//! enough on screen. Tiles with identical pixels share a hue; a tile whose pixels appear nowhere
//! else is plain blue.
//!
//! Grouping reads every stored tile, so it is cached per document revision and active layer,
//! never redone per frame (AGENTS.md rule 8). Tile identity can't stand in for the revision: a
//! tile nothing else shares is written in place. Each allocation is hashed once, and tiles are
//! grouped by that 64-bit hash of their bytes.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use egui::{Align2, Color32, FontId, Painter, Shape, Stroke};
use photocraft_geom::{Rect as DRect, TILE_SIZE, TileCoord};
use photocraft_raster::{Surface, Tile};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::PhotocraftApp;
use crate::canvas::ViewXform;

/// Outline and fill of a tile whose pixels appear nowhere else in the layer.
const UNIQUE: Color32 = Color32::from_rgb(0x2d, 0x8c, 0xeb);
/// Alpha of a stored tile's fill; low so the image stays readable underneath.
const FILL_ALPHA: u8 = 40;
/// The grid over cells with no stored tile: a translucent mid grey, seen on light and dark pixels.
const EMPTY_LINE: Color32 = Color32::from_rgba_premultiplied(32, 32, 32, 64);
/// Tile labels show once a tile is at least this many points across.
const LABEL_MIN: f32 = 56.0;

/// One stored tile and the group of tiles it shares its pixels with (`None` = unique).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileInfo {
    pub coord: TileCoord,
    pub group: Option<usize>,
}

/// The tiles of one layer, valid for one document revision.
#[derive(Clone)]
struct Cached {
    revision: u64,
    tiles: Arc<Vec<TileInfo>>,
}

/// A tile's allocation as a plain address, so it can cross to the Rayon pool.
fn ptr(t: &Arc<Tile>) -> usize {
    Arc::as_ptr(t) as usize
}

fn content_hash(t: &Tile) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    t.bytes().hash(&mut h);
    h.finish()
}

/// Every stored tile of `s` with its shared-pixels group. Groups are numbered in tile order, so
/// the same surface always gets the same colours. `parallel` hashes on the Rayon pool; pass
/// false while jobs hold it, so the UI thread never waits on them (#1554).
pub fn tile_infos(s: &Surface, parallel: bool) -> Vec<TileInfo> {
    // Hash each distinct allocation once; copy-on-write clones share one.
    let mut by_ptr: HashMap<usize, &Arc<Tile>> = HashMap::new();
    for (_, t) in s.tiles() {
        by_ptr.entry(ptr(t)).or_insert(t);
    }
    let distinct: Vec<(usize, &Arc<Tile>)> = by_ptr.into_iter().collect();
    // rayon is a native-only dependency; the web build hashes serially.
    #[cfg(not(target_arch = "wasm32"))]
    let hashed: Vec<(usize, u64)> = if parallel {
        distinct.par_iter().map(|(p, t)| (*p, content_hash(t))).collect()
    } else {
        distinct.iter().map(|(p, t)| (*p, content_hash(t))).collect()
    };
    #[cfg(target_arch = "wasm32")]
    let hashed: Vec<(usize, u64)> = {
        let _ = parallel;
        distinct.iter().map(|(p, t)| (*p, content_hash(t))).collect()
    };
    let hash_of: HashMap<usize, u64> = hashed.into_iter().collect();
    let mut count: HashMap<u64, usize> = HashMap::new();
    for (_, t) in s.tiles() {
        if let Some(h) = hash_of.get(&ptr(t)) {
            *count.entry(*h).or_default() += 1;
        }
    }
    let mut group_of: HashMap<u64, usize> = HashMap::new();
    s.tiles()
        .map(|(c, t)| {
            let group = hash_of.get(&ptr(t)).filter(|h| count.get(h).is_some_and(|&n| n > 1)).map(|h| {
                let next = group_of.len();
                *group_of.entry(*h).or_insert(next)
            });
            TileInfo { coord: *c, group }
        })
        .collect()
}

/// Hues (0..1) left to unique tiles: shared groups never land in the blues, nor in the
/// blue-violets that read as blue next to them ([`UNIQUE`] is about 0.58).
const BLUE_HUES: (f32, f32) = (0.45, 0.75);
/// Gap kept from either end of the allowed range, as a share of it.
const HUE_MARGIN: f32 = 0.05;

/// The hue of shared group `g`: golden-ratio steps keep neighbouring group numbers far apart,
/// spread over the wheel outside [`BLUE_HUES`] and clear of its edges.
fn group_hue(g: usize) -> f32 {
    let step = (g as f32 * 0.618_034).fract();
    let span = 1.0 - (BLUE_HUES.1 - BLUE_HUES.0);
    (BLUE_HUES.1 + (HUE_MARGIN + step * (1.0 - 2.0 * HUE_MARGIN)) * span).fract()
}

/// The colour of a tile group: blue for unique tiles, an evenly spread hue per shared group.
pub fn group_color(group: Option<usize>) -> Color32 {
    let Some(g) = group else { return UNIQUE };
    egui::ecolor::Hsva::new(group_hue(g), 0.85, 0.95, 1.0).into()
}

/// The tile cells of the grid that touch `r`, as an inclusive (tx0, ty0, tx1, ty1) range.
pub fn cell_range(r: DRect) -> Option<(i32, i32, i32, i32)> {
    if r.is_empty() {
        return None;
    }
    let (a, b) = (TileCoord::containing(r.x0, r.y0), TileCoord::containing(r.x1 - 1, r.y1 - 1));
    Some((a.tx, a.ty, b.tx, b.ty))
}

fn quad(xf: &ViewXform, r: DRect) -> [egui::Pos2; 4] {
    let (x0, y0, x1, y1) = (r.x0 as f32, r.y0 as f32, r.x1 as f32, r.y1 as f32);
    [xf.to_screen(x0, y0), xf.to_screen(x1, y0), xf.to_screen(x1, y1), xf.to_screen(x0, y1)]
}

/// Draw the active layer's tiles of document `idx`. `visible` is the document area in view.
pub fn paint(app: &PhotocraftApp, ctx: &egui::Context, painter: &Painter, xf: &ViewXform, idx: usize, visible: DRect) {
    let Some(st) = app.session.documents().get(idx) else { return };
    let Some(surface) = st.active_layer.and_then(|id| st.doc.layer(id)).and_then(|l| l.surface()) else { return };
    let revision = st.revision;
    let mem = egui::Id::new(("pc-tile-overlay", st.doc.id.0, st.active_layer.map(|l| l.0)));
    let cached = ctx.data(|d| d.get_temp::<Cached>(mem)).filter(|c| c.revision == revision);
    let tiles = match cached {
        Some(c) => c.tiles,
        None => {
            let tiles = Arc::new(tile_infos(surface, !app.session.has_jobs()));
            ctx.data_mut(|d| d.insert_temp(mem, Cached { revision, tiles: tiles.clone() }));
            tiles
        }
    };

    // The grid over the document, so the cells without a stored tile show too.
    let bounds = st.doc.bounds();
    if let Some((tx0, ty0, tx1, ty1)) = cell_range(visible.intersect(&bounds)) {
        let line = Stroke::new(1.0, EMPTY_LINE);
        let edge = |t: i32| t.saturating_mul(TILE_SIZE);
        let (top, bottom) = (bounds.y0.max(edge(ty0)) as f32, bounds.y1.min(edge(ty1.saturating_add(1))) as f32);
        let (left, right) = (bounds.x0.max(edge(tx0)) as f32, bounds.x1.min(edge(tx1.saturating_add(1))) as f32);
        for tx in tx0..=tx1.saturating_add(1) {
            let x = edge(tx).clamp(bounds.x0, bounds.x1) as f32;
            painter.line_segment([xf.to_screen(x, top), xf.to_screen(x, bottom)], line);
        }
        for ty in ty0..=ty1.saturating_add(1) {
            let y = edge(ty).clamp(bounds.y0, bounds.y1) as f32;
            painter.line_segment([xf.to_screen(left, y), xf.to_screen(right, y)], line);
        }
    }

    // Stored tiles, which may lie past the canvas edge.
    let size = TILE_SIZE as f32 * xf.zoom;
    for t in tiles.iter() {
        let r = t.coord.rect();
        if r.intersect(&visible).is_empty() {
            continue;
        }
        let c = group_color(t.group);
        let pts = quad(xf, r).to_vec();
        let fill = Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), FILL_ALPHA);
        painter.add(Shape::convex_polygon(pts.clone(), fill, Stroke::NONE));
        painter.add(Shape::closed_line(pts, Stroke::new(1.0, c)));
        if size >= LABEL_MIN {
            let mid = xf.to_screen(r.x0 as f32 + TILE_SIZE as f32 / 2.0, r.y0 as f32 + TILE_SIZE as f32 / 2.0);
            let label = format!("{},{}", t.coord.tx, t.coord.ty);
            let font = FontId::monospace((size / 8.0).clamp(10.0, 16.0));
            // A dark drop shadow keeps the label readable over light pixels.
            painter.text(mid + egui::vec2(1.0, 1.0), Align2::CENTER_CENTER, &label, font.clone(), Color32::from_black_alpha(180));
            painter.text(mid, Align2::CENTER_CENTER, label, font, Color32::WHITE);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_color::PixelFormat;

    /// The document rect of tile cell (tx, ty).
    fn cell(tx: i32, ty: i32) -> DRect {
        TileCoord::new(tx, ty).rect()
    }

    #[test]
    fn empty_surface_has_no_tiles() {
        assert!(tile_infos(&Surface::new(PixelFormat::RGBA8), true).is_empty());
    }

    #[test]
    fn identical_pixels_share_a_group_and_unique_tiles_have_none() {
        for fmt in [PixelFormat::RGBA8, PixelFormat::RGBA16, PixelFormat::RGBA32F] {
            identical_pixels_share_a_group_in(fmt);
        }
    }

    fn identical_pixels_share_a_group_in(fmt: PixelFormat) {
        let mut s = Surface::new(fmt);
        // Two separately allocated tiles with the same red pixels, as after loading a .pcraft.
        s.fill_rect(cell(0, 0), &[1.0, 0.0, 0.0, 1.0]);
        s.fill_rect(cell(2, 0), &[1.0, 0.0, 0.0, 1.0]);
        // One shared allocation in two cells (copy-on-write), and a lone green tile.
        let blue = s.solid_tile(&[0.0, 0.0, 1.0, 1.0]);
        s.put_tiles([(TileCoord::new(0, 1), blue.clone()), (TileCoord::new(1, 1), blue)]);
        s.fill_rect(cell(1, 0), &[0.0, 1.0, 0.0, 1.0]);
        for parallel in [true, false] {
            let infos = tile_infos(&s, parallel);
            let group = |tx, ty| infos.iter().find(|i| i.coord == TileCoord::new(tx, ty)).map(|i| i.group);
            assert_eq!(infos.len(), 5);
            assert_eq!(group(1, 0), Some(None), "green is unique");
            let (red, blue) = (group(0, 0).flatten(), group(0, 1).flatten());
            assert!(red.is_some() && blue.is_some() && red != blue);
            assert_eq!(group(2, 0).flatten(), red);
            assert_eq!(group(1, 1).flatten(), blue);
        }
    }

    #[test]
    fn groups_get_distinct_colours_and_unique_tiles_stay_blue() {
        assert_eq!(group_color(None), UNIQUE);
        let colours: Vec<Color32> = (0..6).map(|g| group_color(Some(g))).collect();
        for (i, a) in colours.iter().enumerate() {
            assert_ne!(*a, UNIQUE);
            assert!(colours.iter().skip(i + 1).all(|b| b != a));
        }
        // Huge group numbers still map to a colour.
        let _ = group_color(Some(usize::MAX));
    }

    #[test]
    fn shared_groups_never_take_the_blue_of_unique_tiles() {
        for g in (0..10_000).chain([usize::MAX]) {
            let h = group_hue(g);
            assert!((0.0..1.0).contains(&h), "group {g}: hue {h}");
            assert!(!(BLUE_HUES.0..=BLUE_HUES.1).contains(&h), "group {g}: hue {h} is blue");
            // Clear of the range's edges too, where hues still read as blue.
            assert!((h - BLUE_HUES.1).abs() > 0.01 && (h - BLUE_HUES.0).abs() > 0.01, "group {g}: hue {h} is at the edge");
        }
    }

    /// `cargo test --release -p photocraft-ui-egui --lib tile_grouping_36mp -- --ignored --nocapture`
    #[test]
    #[ignore = "benchmark"]
    fn tile_grouping_36mp() {
        let (w, h) = (7360, 4912);
        for fmt in [PixelFormat::RGBA8, PixelFormat::RGBA16] {
            // Every tile different, the worst case: each one is hashed and none share.
            let mut s = Surface::new(fmt);
            let Some((tx1, ty1)) = cell_range(DRect::from_xywh(0, 0, w, h)).map(|(_, _, tx1, ty1)| (tx1, ty1)) else { return };
            for ty in 0..=ty1 {
                for tx in 0..=tx1 {
                    let seed = (tx * 31 + ty * 17) as usize;
                    for (i, b) in s.tile_mut(TileCoord::new(tx, ty)).bytes_mut().iter_mut().enumerate() {
                        *b = (i ^ seed) as u8;
                    }
                }
            }
            for parallel in [true, false] {
                let mut best = f64::MAX;
                for _ in 0..8 {
                    let t = std::time::Instant::now();
                    std::hint::black_box(tile_infos(&s, parallel));
                    best = best.min(t.elapsed().as_secs_f64() * 1000.0);
                }
                let mode = if parallel { "parallel" } else { "serial" };
                println!("36 MP {fmt:?}, {} tiles, {mode}: {best:.1} ms", s.tile_count());
            }
        }
    }

    #[test]
    fn cell_range_covers_the_cells_a_rect_touches() {
        let t = TILE_SIZE;
        // Exactly one cell, then a partial cell past whole ones (a canvas not a multiple of it).
        assert_eq!(cell_range(cell(0, 0)), Some((0, 0, 0, 0)));
        assert_eq!(cell_range(DRect::new(0, 0, 7 * t + t / 2, 4 * t + 1)), Some((0, 0, 7, 4)));
        // One pixel either side of a cell edge, including left of the origin.
        assert_eq!(cell_range(DRect::new(-1, t - 1, t + 1, t + 4)), Some((-1, 0, 1, 1)));
        assert_eq!(cell_range(DRect::EMPTY), None);
    }
}
