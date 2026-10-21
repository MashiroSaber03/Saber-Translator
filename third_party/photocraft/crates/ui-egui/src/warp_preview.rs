//! Where the Warp preview samples its mesh: finer where the warp pulls hard.
//!
//! A fixed grid of samples across the whole picture draws a narrow, heavily pulled grid cell with
//! only a few big triangles. Their shear shows as a comb of long wedges along the fold. Here each
//! grid cell gets as many samples as its control points are spread wide, so every triangle stays
//! a few screen pixels across wherever the surface is stretched.

use photocraft_geom::warp::BezierMesh;

/// Target screen pixels per preview segment.
const SEGMENT_PX: f64 = 6.0;
/// Fewest and most segments one grid cell gets along an axis.
const MIN_PER_CELL: usize = 4;
const MAX_PER_CELL: usize = 96;
/// Most samples along either axis for the whole mesh (the triangle count grows with their product).
const MAX_PER_AXIS: usize = 240;

/// Normalised sample positions `(us, vs)` for `mesh` shown at `zoom` screen px per document px.
/// Both include every grid line (0, the knots and 1) in increasing order.
pub fn samples(mesh: &BezierMesh, zoom: f64) -> (Vec<f64>, Vec<f64>) {
    let zoom = if zoom.is_finite() && zoom > 0.0 { zoom } else { 1.0 };
    let (cols, rows) = (mesh.us.len() - 1, mesh.vs.len() - 1);
    let nx = mesh.nx();
    // Screen size of each cell's control points: bounds how far the cell's surface can stretch.
    let mut cu = vec![MIN_PER_CELL; cols];
    let mut cv = vec![MIN_PER_CELL; rows];
    for (b, cv) in cv.iter_mut().enumerate() {
        for (a, cu) in cu.iter_mut().enumerate() {
            let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
            for j in 0..4 {
                for i in 0..4 {
                    if let Some(p) = mesh.points.get((3 * b + j) * nx + 3 * a + i) {
                        for k in 0..2 {
                            lo[k] = lo[k].min(p[k]);
                            hi[k] = hi[k].max(p[k]);
                        }
                    }
                }
            }
            let ext = ((hi[0] - lo[0]).max(hi[1] - lo[1]) * zoom).abs();
            let n = if ext.is_finite() { (ext / SEGMENT_PX).ceil() as usize } else { MIN_PER_CELL };
            let n = n.clamp(MIN_PER_CELL, MAX_PER_CELL);
            *cu = (*cu).max(n);
            *cv = (*cv).max(n);
        }
    }
    (axis(&mesh.us, &mut cu), axis(&mesh.vs, &mut cv))
}

/// Triangle indices for a `row`-wide vertex grid (`pos` and `rest` row by row; `rest` is where each
/// vertex sits with no warp), two triangles per cell. Cells are drawn from the most displaced to the
/// least, so the part of the picture the warp left alone is on top.
///
/// Drawing in plain row order lets the layers of a hard pull (the sheets of an S-shaped fold) cover
/// each other row by row, which shows as a comb along the fold. A fold's layers differ in how far
/// the warp moved them, so ordering by displacement keeps each layer whole, with the tip of the
/// pull underneath and the picture as it was on top.
pub fn triangle_order(pos: &[[f32; 2]], rest: &[[f32; 2]], row: usize) -> Vec<u32> {
    if row < 2 || pos.len() != rest.len() {
        return Vec::new();
    }
    let rows = pos.len() / row;
    let moved = |k: usize| match (pos.get(k), rest.get(k)) {
        (Some(p), Some(r)) => (p[0] - r[0]).hypot(p[1] - r[1]),
        _ => 0.0,
    };
    let mut cells: Vec<(f32, usize)> = Vec::with_capacity(rows * row);
    for j in 0..rows.saturating_sub(1) {
        for i in 0..row - 1 {
            let a = j * row + i;
            cells.push(((moved(a) + moved(a + 1) + moved(a + row) + moved(a + row + 1)) / 4.0, a));
        }
    }
    cells.sort_by(|x, y| y.0.total_cmp(&x.0));
    let mut out = Vec::with_capacity(cells.len() * 6);
    for (_, a) in cells {
        out.extend([a, a + 1, a + row + 1, a, a + row + 1, a + row].map(|k| k as u32));
    }
    out
}

/// Samples along one axis: `counts[k]` segments in knot interval `k`, scaled down together when
/// the total would pass [`MAX_PER_AXIS`].
fn axis(knots: &[f64], counts: &mut [usize]) -> Vec<f64> {
    let total: usize = counts.iter().sum();
    if total > MAX_PER_AXIS {
        for c in counts.iter_mut() {
            *c = (*c * MAX_PER_AXIS / total).max(2);
        }
    }
    let mut out = Vec::with_capacity(counts.iter().sum::<usize>() + 1);
    for (k, n) in counts.iter().enumerate() {
        let (a, b) = (knots[k], knots[k + 1]);
        out.extend((0..*n).map(|i| a + (b - a) * i as f64 / *n as f64));
    }
    out.push(1.0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_narrow_pulled_cell_gets_as_many_samples_as_its_stretch_needs() {
        // A 900 x 1100 mesh with a narrow column (0.45..0.5) pulled 300 px to the left.
        let mut base = BezierMesh::identity([0.0, 0.0, 900.0, 1100.0], 1, 1);
        assert!(base.split_u(0.45) && base.split_u(0.5));
        let mut m = base.clone();
        m.pull(&base.points, 0.47, 0.5, [-300.0, 0.0]);
        let (us, vs) = samples(&m, 1.0);
        let in_narrow = us.iter().filter(|u| **u >= 0.45 && **u < 0.5).count();
        assert!(in_narrow >= 30, "{in_narrow} samples in the narrow cell");
        // Grid lines are samples, and the lists are sorted and span the box.
        for k in [0.0, 0.45, 0.5, 1.0] {
            assert!(us.iter().any(|u| (u - k).abs() < 1e-12), "{k}");
        }
        assert!(us.windows(2).all(|w| w[1] > w[0]) && vs.windows(2).all(|w| w[1] > w[0]));
        assert!(us.len() <= MAX_PER_AXIS + 8 && vs.len() <= MAX_PER_AXIS + 8, "{} x {}", us.len(), vs.len());
    }

    #[test]
    fn the_most_displaced_cells_are_drawn_first() {
        // A 4 x 3 vertex grid; the middle column was pulled 30 px right, so it moved most.
        let rest: Vec<[f32; 2]> = (0..3).flat_map(|j| (0..4).map(move |i| [i as f32 * 10.0, j as f32 * 10.0])).collect();
        let pos: Vec<[f32; 2]> = rest.iter().enumerate().map(|(k, p)| if k % 4 == 1 || k % 4 == 2 { [p[0] + 30.0, p[1]] } else { *p }).collect();
        let idx = triangle_order(&pos, &rest, 4);
        assert_eq!(idx.len(), 2 * 2 * 3 * 3);
        // Cells come in whole pairs of triangles, and displacement never rises along the list.
        let moved = |k: u32| (pos[k as usize][0] - rest[k as usize][0]).abs();
        let per_cell: Vec<f32> = idx.chunks(6).map(|c| c.iter().map(|k| moved(*k)).sum::<f32>()).collect();
        assert!(per_cell.windows(2).all(|w| w[0] >= w[1]), "{per_cell:?}");
        assert!(per_cell[0] > per_cell[per_cell.len() - 1]);
        assert!(triangle_order(&pos, &rest[..5], 4).is_empty());
    }

    #[test]
    fn an_unwarped_mesh_stays_cheap() {
        let m = BezierMesh::identity([0.0, 0.0, 100.0, 100.0], 1, 1);
        let (us, vs) = samples(&m, 1.0);
        assert!(us.len() <= 20 && vs.len() <= 20, "{} x {}", us.len(), vs.len());
    }

    #[test]
    fn a_bad_zoom_still_gives_samples() {
        let m = BezierMesh::identity([0.0, 0.0, 100.0, 100.0], 2, 2);
        for z in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let (us, vs) = samples(&m, z);
            assert!(us.len() >= 3 && vs.len() >= 3);
        }
    }
}
