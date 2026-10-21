//! Hybrid max-flow / min-cut: Boykov–Kolmogorov and FIFO push/relabel.
//!
//! Implemented from Y. Boykov and V. Kolmogorov, "An Experimental Comparison of Min-Cut/Max-Flow
//! Algorithms for Energy Minimization in Vision", IEEE PAMI 26(9), 2004: two search trees
//! (source tree S, sink tree T) grow towards each other; each S–T contact gives an augmenting
//! path; saturated tree edges create orphans that are re-adopted (or freed) before growing again.
//! Search trees are reused between augmentations, which makes the algorithm very fast on the
//! short-path grid graphs of image segmentation.
//!
//! Large finite graphs with both terminal directions use FIFO push/relabel instead:
//! excess is discharged locally, with periodic exact distances to both terminals.
//! This avoids repeated path augmentation for the many area-cost terminals in Quick Selection.
//! Small graphs and graphs with only one terminal direction retain the BK path.

use std::collections::VecDeque;

mod push_relabel;

const NONE: u32 = u32::MAX;
const TERMINAL: u32 = u32::MAX - 1;
const ORPHAN: u32 = u32::MAX - 2;
const INF_DIST: u32 = u32::MAX;
/// Residual capacities at or below this count as saturated (guards against float round-off
/// leaving tiny positive residuals that would cause long chains of negligible augmentations).
const EPS: f32 = 1e-5;
/// How many steps the max-flow loops take between asks whether to stop.
const CANCEL_EVERY: u32 = 1024;

fn never() -> bool {
    false
}

/// A directed graph with source/sink terminal capacities.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    // Nodes.
    first: Vec<u32>,
    /// Residual terminal capacity: `> 0` from the source, `< 0` to the sink.
    tr_cap: Vec<f32>,
    parent: Vec<u32>,
    is_sink: Vec<bool>,
    ts: Vec<u32>,
    dist: Vec<u32>,
    active: Vec<bool>,
    // Arcs (stored in pairs: arc `a` and its reverse `a ^ 1`).
    head: Vec<u32>,
    next: Vec<u32>,
    r_cap: Vec<f32>,
    flow: f64,
}

impl Graph {
    /// A graph with `nodes` nodes and room for `edges` undirected edge pairs.
    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        Graph {
            first: vec![NONE; nodes],
            tr_cap: vec![0.0; nodes],
            parent: vec![NONE; nodes],
            is_sink: vec![false; nodes],
            ts: vec![0; nodes],
            dist: vec![0; nodes],
            active: vec![false; nodes],
            head: Vec::with_capacity(edges * 2),
            next: Vec::with_capacity(edges * 2),
            r_cap: Vec::with_capacity(edges * 2),
            flow: 0.0,
        }
    }

    pub fn node_count(&self) -> usize {
        self.first.len()
    }

    /// Adds capacities from the source and to the sink (accumulating). Only the difference is
    /// kept per node; the common part is flow that must pass anyway.
    pub fn add_tweights(&mut self, i: usize, cap_source: f32, cap_sink: f32) {
        let delta = self.tr_cap[i];
        let (mut s, mut t) = (cap_source, cap_sink);
        if delta > 0.0 {
            s += delta;
        } else {
            t -= delta;
        }
        self.flow += s.min(t) as f64;
        self.tr_cap[i] = s - t;
    }

    /// Adds an edge `i → j` with capacity `cap` and `j → i` with `rev_cap`.
    pub fn add_edge(&mut self, i: usize, j: usize, cap: f32, rev_cap: f32) {
        debug_assert!(i != j);
        let a = self.head.len() as u32;
        self.head.push(j as u32);
        self.next.push(self.first[i]);
        self.r_cap.push(cap);
        self.first[i] = a;
        self.head.push(i as u32);
        self.next.push(self.first[j]);
        self.r_cap.push(rev_cap);
        self.first[j] = a + 1;
    }

    /// After [`Graph::maxflow`]: is node `i` on the source side of the minimum cut? Nodes
    /// reachable from neither terminal are reported on the sink side (the smallest source set).
    pub fn in_source(&self, i: usize) -> bool {
        self.parent[i] != NONE && !self.is_sink[i]
    }

    /// The flow value found by the last [`Graph::maxflow`].
    pub fn flow(&self) -> f64 {
        self.flow
    }

    fn arcs(&self, i: usize) -> ArcIter<'_> {
        ArcIter { g: self, a: self.first[i] }
    }

    /// Computes the maximum flow (= minimum cut capacity).
    pub fn maxflow(&mut self) -> f64 {
        self.maxflow_with(&never).unwrap_or(self.flow)
    }

    /// [`Graph::maxflow`] that stops when `cancel` returns true (asked every few thousand steps):
    /// `None` then, and the cut is meaningless. Not cancelled, it does exactly what
    /// [`Graph::maxflow`] does.
    pub fn maxflow_with(&mut self, cancel: &dyn Fn() -> bool) -> Option<f64> {
        if self.node_count() >= 4096 {
            let mut source = false;
            let mut sink = false;
            for &capacity in &self.tr_cap {
                if !capacity.is_finite() {
                    return self.bk(cancel);
                }
                source |= capacity > EPS;
                sink |= capacity < -EPS;
            }
            if source && sink && self.r_cap.iter().all(|v| v.is_finite() && *v >= 0.0) {
                return self.push_relabel(cancel);
            }
        }
        self.bk(cancel)
    }

    pub(super) fn maxflow_bk(&mut self) -> f64 {
        self.bk(&never).unwrap_or(self.flow)
    }

    fn bk(&mut self, cancel: &dyn Fn() -> bool) -> Option<f64> {
        let n = self.node_count();
        let mut queue: VecDeque<u32> = VecDeque::new();
        let mut orphans: VecDeque<u32> = VecDeque::new();
        for i in 0..n {
            self.active[i] = false;
            self.ts[i] = 0;
            if self.tr_cap[i].abs() > EPS {
                self.is_sink[i] = self.tr_cap[i] < 0.0;
                self.parent[i] = TERMINAL;
                self.dist[i] = 1;
                self.active[i] = true;
                queue.push_back(i as u32);
            } else {
                self.parent[i] = NONE;
            }
        }
        let mut time = 0u32;
        let mut steps = 0u32;
        while let Some(i) = queue.pop_front() {
            steps = steps.wrapping_add(1);
            if steps.is_multiple_of(CANCEL_EVERY) && cancel() {
                return None;
            }
            let i = i as usize;
            self.active[i] = false;
            if self.parent[i] == NONE {
                continue;
            }
            // Growth stage.
            let mut contact = NONE;
            let sink = self.is_sink[i];
            let mut a = self.first[i];
            while a != NONE {
                let au = a as usize;
                let j = self.head[au] as usize;
                let cap = if sink { self.r_cap[au ^ 1] } else { self.r_cap[au] };
                if cap > EPS {
                    if self.parent[j] == NONE {
                        self.is_sink[j] = sink;
                        self.parent[j] = a ^ 1;
                        self.ts[j] = self.ts[i];
                        self.dist[j] = self.dist[i] + 1;
                        if !self.active[j] {
                            self.active[j] = true;
                            queue.push_back(j as u32);
                        }
                    } else if self.is_sink[j] != sink {
                        // Arc from the S side to the T side.
                        contact = if sink { a ^ 1 } else { a };
                        break;
                    } else if self.ts[j] <= self.ts[i] && self.dist[j] > self.dist[i] {
                        // Heuristic: shorten paths to the terminal.
                        self.parent[j] = a ^ 1;
                        self.ts[j] = self.ts[i];
                        self.dist[j] = self.dist[i] + 1;
                    }
                }
                a = self.next[au];
            }
            if contact == NONE {
                continue;
            }
            // Keep scanning this node after the augmentation.
            if !self.active[i] {
                self.active[i] = true;
                queue.push_front(i as u32);
            }
            time += 1;
            self.augment(contact as usize, &mut orphans);
            // Adoption stage.
            while let Some(o) = orphans.pop_front() {
                self.adopt(o as usize, time, &mut queue, &mut orphans);
            }
        }
        Some(self.flow)
    }

    fn augment(&mut self, m: usize, orphans: &mut VecDeque<u32>) {
        let u = self.head[m ^ 1] as usize;
        let v = self.head[m] as usize;
        // Bottleneck.
        let mut b = self.r_cap[m];
        let mut x = u;
        loop {
            let a = self.parent[x];
            if a == TERMINAL {
                break;
            }
            b = b.min(self.r_cap[a as usize ^ 1]);
            x = self.head[a as usize] as usize;
        }
        b = b.min(self.tr_cap[x]);
        let mut x = v;
        loop {
            let a = self.parent[x];
            if a == TERMINAL {
                break;
            }
            b = b.min(self.r_cap[a as usize]);
            x = self.head[a as usize] as usize;
        }
        b = b.min(-self.tr_cap[x]);
        // Push.
        self.r_cap[m ^ 1] += b;
        self.r_cap[m] -= b;
        let mut x = u;
        loop {
            let a = self.parent[x];
            if a == TERMINAL {
                break;
            }
            let au = a as usize;
            self.r_cap[au] += b;
            self.r_cap[au ^ 1] -= b;
            let up = self.head[au] as usize;
            if self.r_cap[au ^ 1] <= EPS {
                self.parent[x] = ORPHAN;
                orphans.push_back(x as u32);
            }
            x = up;
        }
        self.tr_cap[x] -= b;
        if self.tr_cap[x] <= EPS {
            self.parent[x] = ORPHAN;
            orphans.push_back(x as u32);
        }
        let mut x = v;
        loop {
            let a = self.parent[x];
            if a == TERMINAL {
                break;
            }
            let au = a as usize;
            self.r_cap[au ^ 1] += b;
            self.r_cap[au] -= b;
            let up = self.head[au] as usize;
            if self.r_cap[au] <= EPS {
                self.parent[x] = ORPHAN;
                orphans.push_back(x as u32);
            }
            x = up;
        }
        self.tr_cap[x] += b;
        if self.tr_cap[x] >= -EPS {
            self.parent[x] = ORPHAN;
            orphans.push_back(x as u32);
        }
        self.flow += b as f64;
    }

    fn adopt(&mut self, i: usize, time: u32, queue: &mut VecDeque<u32>, orphans: &mut VecDeque<u32>) {
        let sink = self.is_sink[i];
        let mut best = NONE;
        let mut d_min = INF_DIST;
        let mut a = self.first[i];
        while a != NONE {
            let au = a as usize;
            let cap = if sink { self.r_cap[au] } else { self.r_cap[au ^ 1] };
            let j = self.head[au] as usize;
            if cap > EPS && self.is_sink[j] == sink && self.parent[j] != NONE {
                // Does j's path lead to a terminal?
                let mut d = 0u32;
                let mut k = j;
                loop {
                    if self.ts[k] == time {
                        d = d.saturating_add(self.dist[k]);
                        break;
                    }
                    let pa = self.parent[k];
                    d += 1;
                    if pa == TERMINAL {
                        self.ts[k] = time;
                        self.dist[k] = 1;
                        break;
                    }
                    if pa == ORPHAN {
                        d = INF_DIST;
                        break;
                    }
                    k = self.head[pa as usize] as usize;
                }
                if d < INF_DIST {
                    if d < d_min {
                        best = a;
                        d_min = d;
                    }
                    // Cache distances along the path.
                    let mut k = j;
                    let mut dd = d;
                    while self.ts[k] != time {
                        self.ts[k] = time;
                        self.dist[k] = dd;
                        dd -= 1;
                        k = self.head[self.parent[k] as usize] as usize;
                    }
                }
            }
            a = self.next[au];
        }
        if best != NONE {
            self.parent[i] = best;
            self.ts[i] = time;
            self.dist[i] = d_min + 1;
            return;
        }
        // No parent: i becomes free.
        self.parent[i] = NONE;
        let mut a = self.first[i];
        while a != NONE {
            let au = a as usize;
            let j = self.head[au] as usize;
            let pj = self.parent[j];
            if self.is_sink[j] == sink && pj != NONE {
                let cap = if sink { self.r_cap[au] } else { self.r_cap[au ^ 1] };
                if cap > EPS && !self.active[j] {
                    self.active[j] = true;
                    queue.push_back(j as u32);
                }
                if pj != TERMINAL && pj != ORPHAN && self.head[pj as usize] as usize == i {
                    self.parent[j] = ORPHAN;
                    orphans.push_back(j as u32);
                }
            }
            a = self.next[au];
        }
    }
}

struct ArcIter<'a> {
    g: &'a Graph,
    a: u32,
}

impl Iterator for ArcIter<'_> {
    /// `(neighbour, residual capacity out, residual capacity in)`.
    type Item = (usize, f32, f32);
    fn next(&mut self) -> Option<Self::Item> {
        if self.a == NONE {
            return None;
        }
        let a = self.a as usize;
        self.a = self.g.next[a];
        Some((self.g.head[a] as usize, self.g.r_cap[a], self.g.r_cap[a ^ 1]))
    }
}

impl Graph {
    /// Sum of residual capacities on arcs leaving the source side of the cut (zero after
    /// [`Graph::maxflow`]: no augmenting path remains).
    pub fn residual_out_of_source_side(&self) -> f32 {
        let mut s = 0.0;
        for i in 0..self.node_count() {
            if self.in_source(i) {
                for (j, out, _) in self.arcs(i) {
                    if !self.in_source(j) {
                        s += out;
                    }
                }
            }
        }
        s
    }
}
