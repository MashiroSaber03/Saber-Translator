//! Web slices (Slice tool, Layer › New Layer Based Slice, Save for Web): pure data plus the
//! auto-slice partition and Photoshop-style numbering.
//!
//! A document keeps its *user* slices (drawn with the Slice tool) and *layer-based* slices (they
//! follow a layer's bounds, effects included). *Auto* slices are never stored: they are the
//! rectangles that fill the rest of the canvas, recomputed whenever they are needed, exactly like
//! Photoshop regenerates them after every slice edit.

use photocraft_geom::Rect;
use serde::{Deserialize, Serialize};

use crate::{Document, LayerId};

/// Where a slice comes from (PSD `ESliceOrigin`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SliceOrigin {
    /// Fills the canvas around user and layer slices; regenerated, never edited.
    Auto,
    /// Follows a layer's content bounds (Layer › New Layer Based Slice).
    Layer,
    /// Drawn with the Slice tool (or promoted from an auto slice).
    #[default]
    User,
}

impl SliceOrigin {
    /// PSD resource 1050 code: 0 auto, 1 layer, 2 user.
    pub fn code(self) -> u32 {
        match self {
            SliceOrigin::Auto => 0,
            SliceOrigin::Layer => 1,
            SliceOrigin::User => 2,
        }
    }
    pub fn from_code(c: u32) -> Self {
        match c {
            0 => SliceOrigin::Auto,
            1 => SliceOrigin::Layer,
            _ => SliceOrigin::User,
        }
    }
    pub fn id(self) -> &'static str {
        match self {
            SliceOrigin::Auto => "auto",
            SliceOrigin::Layer => "layer",
            SliceOrigin::User => "user",
        }
    }
}

/// Slice Options › Slice Type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SliceKind {
    /// Exported as an image file.
    #[default]
    Image,
    /// An HTML cell with text (or nothing); no image is written.
    NoImage,
    /// A nested table (ImageReady legacy; exported like Image).
    Table,
}

impl SliceKind {
    /// PSD resource 1050 code: 0 no image, 1 image, 2 table.
    pub fn code(self) -> u32 {
        match self {
            SliceKind::NoImage => 0,
            SliceKind::Image => 1,
            SliceKind::Table => 2,
        }
    }
    pub fn from_code(c: u32) -> Self {
        match c {
            0 => SliceKind::NoImage,
            2 => SliceKind::Table,
            _ => SliceKind::Image,
        }
    }
    pub fn id(self) -> &'static str {
        match self {
            SliceKind::Image => "image",
            SliceKind::NoImage => "noImage",
            SliceKind::Table => "table",
        }
    }
    pub fn from_id(s: &str) -> Option<Self> {
        Some(match s {
            "image" | "img" => SliceKind::Image,
            "noImage" | "none" | "no-image" => SliceKind::NoImage,
            "table" => SliceKind::Table,
            _ => return None,
        })
    }
}

/// One slice and its Slice Options.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Slice {
    /// Stable id (PSD `sliceID`), unique within the document.
    pub id: u32,
    pub group_id: u32,
    pub origin: SliceOrigin,
    /// The layer a layer-based slice follows.
    pub layer: Option<LayerId>,
    /// Slice Options › Name (empty = the generated `<document>_<NN>`).
    pub name: String,
    /// Document pixels. Layer-based slices: the layer's bounds (with effects) plus `outsets`.
    pub rect: Rect,
    pub kind: SliceKind,
    pub url: String,
    pub target: String,
    pub message: String,
    pub alt: String,
    pub cell_text_is_html: bool,
    pub cell_text: String,
    /// HTML alignment of no-image cells (0 default, 1 left, 2 centre, 3 right / top, middle, …).
    pub horizontal_align: u32,
    pub vertical_align: u32,
    /// Slice background (ARGB, 8-bit); None = none.
    pub background: Option<[u8; 4]>,
    /// Layer-based slices: extra margin [top, left, bottom, right].
    pub outsets: [i32; 4],
}

/// The document's slices (PSD resource 1050).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Slices {
    /// Group name (PSD: the base name, usually the document name).
    pub group_name: String,
    /// User and layer-based slices, in creation order (later slices are on top).
    pub list: Vec<Slice>,
}

impl Slices {
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }
    /// A fresh slice id (Photoshop counts from 1; auto slices use ids too, so leave room).
    /// Returns `None` when the stored ids have exhausted the `u32` id space.
    pub fn next_id(&self) -> Option<u32> {
        self.list.iter().map(|s| s.id).max().unwrap_or(0).checked_add(1)
    }
    pub fn get(&self, id: u32) -> Option<&Slice> {
        self.list.iter().find(|s| s.id == id)
    }
    pub fn get_mut(&mut self, id: u32) -> Option<&mut Slice> {
        self.list.iter_mut().find(|s| s.id == id)
    }
}

/// A slice as shown and exported: user/layer slices plus generated auto slices, numbered the way
/// Photoshop numbers them (top-to-bottom, left-to-right, from 1).
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedSlice {
    /// 1-based display number.
    pub number: usize,
    /// The stored slice (None for auto slices).
    pub id: Option<u32>,
    pub origin: SliceOrigin,
    pub rect: Rect,
    /// The slice's name, or the generated `<base>_<NN>`.
    pub name: String,
    pub kind: SliceKind,
}

/// The visible part of every stored slice (later slices cover earlier ones, as in Photoshop),
/// clipped to the canvas.
fn clipped(doc: &Document) -> Vec<(u32, Rect)> {
    let canvas = doc.bounds();
    doc.slices.list.iter().map(|s| (s.id, s.rect.intersect(&canvas))).filter(|(_, r)| !r.is_empty()).collect()
}

/// Rectangles that tile `canvas` minus `taken`. The canvas is cut into a grid along every slice
/// edge; free cells merge into horizontal runs per row band, then runs with identical columns
/// merge down, which yields the few large auto slices Photoshop shows.
pub fn auto_slices(canvas: Rect, taken: &[Rect]) -> Vec<Rect> {
    if canvas.is_empty() {
        return Vec::new();
    }
    let mut xs = vec![canvas.x0, canvas.x1];
    let mut ys = vec![canvas.y0, canvas.y1];
    for r in taken {
        let r = r.intersect(&canvas);
        if r.is_empty() {
            continue;
        }
        xs.extend([r.x0, r.x1]);
        ys.extend([r.y0, r.y1]);
    }
    xs.sort_unstable();
    xs.dedup();
    ys.sort_unstable();
    ys.dedup();
    let covered = |x0: i32, y0: i32| taken.iter().any(|r| x0 >= r.x0 && x0 < r.x1 && y0 >= r.y0 && y0 < r.y1);
    // Horizontal runs per row band.
    let mut runs: Vec<Rect> = Vec::new();
    for yw in ys.windows(2) {
        let mut start: Option<i32> = None;
        for xw in xs.windows(2) {
            let free = !covered(xw[0], yw[0]);
            match (free, start) {
                (true, None) => start = Some(xw[0]),
                (false, Some(s)) => {
                    runs.push(Rect::new(s, yw[0], xw[0], yw[1]));
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(s) = start {
            runs.push(Rect::new(s, yw[0], *xs.last().unwrap_or(&canvas.x1), yw[1]));
        }
    }
    // Merge vertically adjacent runs with the same columns.
    let mut out: Vec<Rect> = Vec::new();
    for r in runs {
        if let Some(prev) = out.iter_mut().find(|p| p.x0 == r.x0 && p.x1 == r.x1 && p.y1 == r.y0) {
            prev.y1 = r.y1;
        } else {
            out.push(r);
        }
    }
    out
}

/// Every slice of `doc` in display order (see [`ResolvedSlice`]). Without user or layer slices
/// the whole canvas is one auto slice named `<base>` alone (as Photoshop's Save for Web names
/// it); with several, generated names are numbered `<base>_NN`.
pub fn resolve(doc: &Document) -> Vec<ResolvedSlice> {
    let base = base_name(doc);
    let stored = clipped(doc);
    let rects: Vec<Rect> = stored.iter().map(|(_, r)| *r).collect();
    let mut all: Vec<(Option<u32>, Rect)> = stored.iter().map(|(id, r)| (Some(*id), *r)).collect();
    all.extend(auto_slices(doc.bounds(), &rects).into_iter().map(|r| (None, r)));
    all.sort_by_key(|(_, r)| (r.y0, r.x0));
    let single = all.len() == 1;
    all.into_iter()
        .enumerate()
        .map(|(i, (id, rect))| {
            let s = id.and_then(|id| doc.slices.get(id));
            let number = i + 1;
            let name = match s.map(|s| s.name.as_str()).filter(|n| !n.is_empty()) {
                Some(n) => n.to_string(),
                None if single => base.clone(),
                None => format!("{base}_{number:02}"),
            };
            ResolvedSlice { number, id, origin: s.map_or(SliceOrigin::Auto, |s| s.origin), rect, name, kind: s.map_or(SliceKind::Image, |s| s.kind) }
        })
        .collect()
}

/// The base of generated slice names: the document name without extension, file-name safe.
pub fn base_name(doc: &Document) -> String {
    let n = doc.name.rsplit_once('.').map_or(doc.name.as_str(), |(a, _)| a);
    let s: String = n.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
    if s.is_empty() { "Untitled".into() } else { s }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ColorMode, SampleType, Size};

    fn area(rs: &[Rect]) -> i64 {
        rs.iter().map(|r| i64::from(r.width()) * i64::from(r.height())).sum()
    }

    #[test]
    fn auto_slices_tile_the_rest_of_the_canvas() {
        let canvas = Rect::new(0, 0, 100, 80);
        let user = [Rect::new(20, 10, 60, 30)];
        let auto = auto_slices(canvas, &user);
        assert_eq!(area(&auto) + area(&user), 100 * 80);
        for (i, a) in auto.iter().enumerate() {
            assert!(a.intersect(&user[0]).is_empty());
            for b in &auto[i + 1..] {
                assert!(a.intersect(b).is_empty(), "{a:?} overlaps {b:?}");
            }
        }
        // Above band, left, right, below band.
        assert_eq!(auto.len(), 4);
        assert_eq!(auto[0], Rect::new(0, 0, 100, 10));
        assert_eq!(auto_slices(canvas, &[]), vec![canvas]);
    }

    #[test]
    fn resolve_numbers_top_to_bottom() {
        let mut d = Document::new("site.psd", Size::new(100, 80), ColorMode::Rgb, SampleType::U8);
        assert_eq!(resolve(&d).len(), 1);
        assert_eq!(resolve(&d)[0].name, "site");
        d.slices.list.push(Slice { id: 1, rect: Rect::new(20, 10, 60, 30), ..Default::default() });
        let r = resolve(&d);
        assert_eq!(r.len(), 5);
        assert_eq!(r[0].name, "site_01");
        let user = r.iter().find(|s| s.id == Some(1)).unwrap();
        assert_eq!(user.number, 3, "after the top band and the left cell");
        assert_eq!(user.origin, SliceOrigin::User);
        d.slices.list[0].name = "logo".into();
        assert!(resolve(&d).iter().any(|s| s.name == "logo"));
    }

    #[test]
    fn exhausted_slice_id_survives_serialization() {
        let slices = Slices { list: vec![Slice { id: u32::MAX, ..Default::default() }], ..Default::default() };
        let value = serde_json::to_value(&slices).unwrap();
        let loaded: Slices = serde_json::from_value(value).unwrap();
        assert_eq!(loaded.next_id(), None);
    }
}
