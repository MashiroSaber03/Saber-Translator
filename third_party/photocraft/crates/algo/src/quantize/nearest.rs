//! Exact nearest colour search. Balanced median splits bound recursion to O(log n).
//! Small palettes use a linear scan; no colour buckets, sampling or approximate distances.

#[derive(Clone, Copy)]
struct Entry {
    point: [f32; 3],
    index: usize,
    axis: usize,
}

pub(super) struct Nearest {
    entries: Vec<Entry>,
    tree: bool,
}

#[inline]
fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

impl Nearest {
    pub(super) fn new(points: impl IntoIterator<Item = [f32; 3]>) -> Self {
        let mut entries: Vec<Entry> = points.into_iter().enumerate().map(|(index, point)| Entry { point, index, axis: 0 }).collect();
        let tree = entries.len() > 64 && entries.iter().all(|e| e.point.iter().all(|v| v.is_finite()));
        if tree {
            partition(&mut entries);
        }
        Self { entries, tree }
    }

    #[inline]
    pub(super) fn find(&self, point: [f32; 3]) -> usize {
        let mut best = (f32::MAX, 0);
        if !self.tree {
            // This slice is still in original order, so strict comparison already breaks ties.
            for entry in &self.entries {
                let d = distance(entry.point, point);
                if d < best.0 {
                    best = (d, entry.index);
                }
            }
        } else if point.iter().all(|v| v.is_finite()) {
            search(&self.entries, point, &mut best);
        }
        // Nonfinite queries against finite tree entries have only NaN/infinite distances;
        // like the original scan, none beats f32::MAX and the fallback remains index zero.
        best.1
    }
}

#[inline]
fn consider(entry: &Entry, point: [f32; 3], best: &mut (f32, usize)) {
    let d = distance(entry.point, point);
    // Original palette order breaks ties, independent of the tree's traversal order.
    if d < best.0 || (d == best.0 && entry.index < best.1) {
        *best = (d, entry.index);
    }
}

fn partition(entries: &mut [Entry]) {
    if entries.len() <= 8 {
        return;
    }
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for e in entries.iter() {
        for ((lo, hi), p) in lo.iter_mut().zip(hi.iter_mut()).zip(e.point) {
            *lo = lo.min(p);
            *hi = hi.max(p);
        }
    }
    let axis = (0..3).max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b]))).unwrap_or(0);
    let mid = entries.len() / 2;
    // `mid` comes from this slice's length, never from image data.
    let (left, pivot, right) = entries.select_nth_unstable_by(mid, |a, b| a.point[axis].total_cmp(&b.point[axis]).then(a.index.cmp(&b.index)));
    pivot.axis = axis;
    partition(left);
    partition(right);
}

fn search(entries: &[Entry], point: [f32; 3], best: &mut (f32, usize)) {
    if entries.len() <= 8 {
        for entry in entries {
            consider(entry, point, best);
        }
        return;
    }
    let (left, rest) = entries.split_at(entries.len() / 2);
    let Some((pivot, right)) = rest.split_first() else { return };
    consider(pivot, point, best);
    let delta = point[pivot.axis] - pivot.point[pivot.axis];
    let (near, far) = if delta <= 0.0 { (left, right) } else { (right, left) };
    if !near.is_empty() {
        search(near, point, best);
    }
    // The splitting plane is a lower bound, using the same f32 arithmetic as distance().
    // Equality must be searched as well: a lower original index may win a distance tie.
    if !far.is_empty() && delta.powi(2) <= best.0 {
        search(far, point, best);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear(points: &[[f32; 3]], query: [f32; 3]) -> usize {
        let mut best = (f32::MAX, 0);
        for (index, point) in points.iter().enumerate() {
            let d = distance(*point, query);
            if d < best.0 {
                best = (d, index);
            }
        }
        best.1
    }

    #[test]
    fn search_matches_linear_oracle_in_rgb_and_lab_with_ties_and_nonfinite_input() {
        let mut seed = 123456789u32;
        let mut random = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / 16777215.0
        };
        for count in [0, 2, 8, 16, 17, 32, 64, 65, 128, 256] {
            for (offset, scale) in [(0.0, 255.0), (-128.0, 256.0)] {
                let points: Vec<[f32; 3]> = (0..count).map(|_| std::array::from_fn(|_| random() * scale + offset)).collect();
                let lookup = Nearest::new(points.iter().copied());
                for _ in 0..3000 {
                    let q = std::array::from_fn(|_| random() * scale + offset);
                    assert_eq!(lookup.find(q), linear(&points, q), "count {count}, {q:?}");
                }
                for q in [[f32::NAN; 3], [f32::INFINITY; 3], [f32::MAX; 3], [0.0; 3]] {
                    assert_eq!(lookup.find(q), linear(&points, q));
                }
            }
        }
        let mut points = vec![[1.0; 3]; 256];
        points[250] = [-1.0; 3];
        let lookup = Nearest::new(points.iter().copied());
        assert_eq!(lookup.find([0.0; 3]), 0, "equidistant entries preserve original order");
        points[0] = [f32::NAN; 3];
        let lookup = Nearest::new(points.iter().copied());
        assert_eq!(lookup.find([0.0; 3]), 1);
    }
}
