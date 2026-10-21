//! FIFO push/relabel with periodic exact distances to both terminals.
//!
//! Implemented from Goldberg & Tarjan, "A New Approach to the Maximum-Flow
//! Problem", JACM 35(4), 1988, with FIFO scheduling and global relabelling.
//! The existing SoA residual graph is reused. Terminal source arcs are saturated
//! into a preflow; `back` keeps their reverse capacity, `excess` uses f64 so local
//! f32 capacity updates do not accumulate excess-rounding drift.
use super::{CANCEL_EVERY, EPS, Graph, NONE, TERMINAL};
use std::collections::VecDeque;

impl Graph {
    #[cfg(test)]
    pub(super) fn maxflow_push_relabel(&mut self) -> f64 {
        self.push_relabel(&super::never).unwrap_or(self.flow)
    }

    /// Push/relabel; `None` when `cancel` stopped it.
    pub(super) fn push_relabel(&mut self, cancel: &dyn Fn() -> bool) -> Option<f64> {
        let n = self.node_count();
        let Some(source_height) = u32::try_from(n).ok().filter(|&n| n < u32::MAX / 2).map(|n| n + 2) else {
            return self.bk(cancel);
        };
        let mut back = vec![0.0f32; n];
        let mut excess = vec![0.0f64; n];
        for i in 0..n {
            self.active[i] = false;
            if self.tr_cap[i] > EPS {
                back[i] = self.tr_cap[i];
                excess[i] = f64::from(self.tr_cap[i]);
                self.tr_cap[i] = 0.0;
            }
        }
        self.global_relabel(source_height, &back);
        let mut queue = VecDeque::new();
        for (i, &e) in excess.iter().enumerate() {
            if e > f64::from(EPS) {
                self.active[i] = true;
                queue.push_back(i)
            }
        }
        let mut work = 0usize;
        let frequency = self.r_cap.len().saturating_mul(2).saturating_add(n);
        let mut steps = 0u32;
        while let Some(i) = queue.pop_front() {
            steps = steps.wrapping_add(1);
            if steps.is_multiple_of(CANCEL_EVERY) && cancel() {
                return None;
            }
            self.active[i] = false;
            while excess[i] > f64::from(EPS) {
                if self.dist[i] == 1 && self.tr_cap[i] < -EPS {
                    let delta = (excess[i] as f32).min(-self.tr_cap[i]);
                    self.tr_cap[i] += delta;
                    excess[i] -= f64::from(delta);
                    self.flow += f64::from(delta);
                    if excess[i] <= f64::from(EPS) {
                        break;
                    }
                }
                if self.dist[i] == source_height + 1 && back[i] > EPS {
                    let delta = (excess[i] as f32).min(back[i]);
                    back[i] -= delta;
                    self.tr_cap[i] += delta;
                    excess[i] -= f64::from(delta);
                    if excess[i] <= f64::from(EPS) {
                        break;
                    }
                }
                let mut a = self.parent[i];
                while a != NONE {
                    let ai = a as usize;
                    let j = self.head[ai] as usize;
                    work += 1;
                    // `checked_add`: two nodes a global relabel left unreachable (`u32::MAX`) must
                    // not count as admissible for each other, or float-stranded excess ping-pongs
                    // between them forever.
                    if self.r_cap[ai] > EPS && self.dist[j].checked_add(1) == Some(self.dist[i]) {
                        let delta = (excess[i] as f32).min(self.r_cap[ai]);
                        self.r_cap[ai] -= delta;
                        self.r_cap[ai ^ 1] += delta;
                        excess[i] -= f64::from(delta);
                        excess[j] += f64::from(delta);
                        if !self.active[j] && excess[j] > f64::from(EPS) {
                            self.active[j] = true;
                            queue.push_back(j);
                        }
                        if excess[i] <= f64::from(EPS) {
                            break;
                        }
                    }
                    a = self.next[ai];
                    self.parent[i] = a;
                }
                if excess[i] <= f64::from(EPS) {
                    break;
                }
                let mut height = if self.tr_cap[i] < -EPS { 0 } else { u32::MAX };
                if back[i] > EPS {
                    height = height.min(source_height)
                }
                let mut a = self.first[i];
                while a != NONE {
                    let ai = a as usize;
                    work += 1;
                    if self.r_cap[ai] > EPS {
                        height = height.min(self.dist[self.head[ai] as usize])
                    }
                    a = self.next[ai];
                }
                if height == u32::MAX {
                    break;
                }
                self.dist[i] = height.saturating_add(1);
                self.parent[i] = self.first[i];
            }
            if work >= frequency {
                self.global_relabel(source_height, &back);
                work = 0;
            }
        }
        // Report the smallest source-side cut, independently of discharge order.
        self.parent.fill(NONE);
        self.is_sink.fill(true);
        for i in 0..n {
            if self.tr_cap[i] > EPS {
                self.parent[i] = TERMINAL;
                self.is_sink[i] = false;
                queue.push_back(i);
            }
        }
        while let Some(i) = queue.pop_front() {
            let mut a = self.first[i];
            while a != NONE {
                let ai = a as usize;
                let j = self.head[ai] as usize;
                if self.r_cap[ai] > EPS && self.parent[j] == NONE {
                    self.parent[j] = TERMINAL;
                    self.is_sink[j] = false;
                    queue.push_back(j);
                }
                a = self.next[ai];
            }
        }
        Some(self.flow)
    }

    fn global_relabel(&mut self, source_height: u32, back: &[f32]) {
        self.dist.fill(u32::MAX);
        self.parent.clone_from(&self.first);
        let mut queue = VecDeque::new();
        for i in 0..self.node_count() {
            if self.tr_cap[i] < -EPS {
                self.dist[i] = 1;
                queue.push_back(i)
            }
        }
        while let Some(i) = queue.pop_front() {
            let mut a = self.first[i];
            while a != NONE {
                let ai = a as usize;
                let j = self.head[ai] as usize;
                if self.r_cap[ai ^ 1] > EPS && self.dist[j] == u32::MAX {
                    self.dist[j] = self.dist[i].saturating_add(1);
                    queue.push_back(j);
                }
                a = self.next[ai];
            }
        }
        // Give sink-unreachable vertices exact distances back to the source.
        // Resetting all of them to the same height can undo discharge progress.
        for (i, &capacity) in back.iter().enumerate() {
            if capacity > EPS && self.dist[i] == u32::MAX {
                self.dist[i] = source_height + 1;
                queue.push_back(i);
            }
        }
        while let Some(i) = queue.pop_front() {
            let mut a = self.first[i];
            while a != NONE {
                let ai = a as usize;
                let j = self.head[ai] as usize;
                if self.r_cap[ai ^ 1] > EPS && self.dist[j] == u32::MAX {
                    self.dist[j] = self.dist[i].saturating_add(1);
                    queue.push_back(j);
                }
                a = self.next[ai];
            }
        }
    }
}

#[cfg(test)]
mod tests;
