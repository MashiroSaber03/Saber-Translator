//! Gaussian mixture colour models with full 3×3 covariances.
//!
//! Fitting: k-means++ seeding (Arthur & Vassilvitskii, "k-means++: The Advantages of Careful
//! Seeding", SODA 2007), Lloyd iterations, then a few EM steps (Dempster, Laird & Rubin 1977).
//! GrabCut's per-iteration update (assign every sample to its most likely component, re-estimate)
//! is [`Gmm::refit`]. Covariances get a small ridge so flat colour areas stay well-conditioned.

use super::Rng;

/// One Gaussian component.
#[derive(Clone, Debug, PartialEq)]
pub struct Component {
    pub weight: f32,
    pub mean: [f32; 3],
    pub cov: [[f32; 3]; 3],
    inv: [[f64; 3]; 3],
    /// `ln(weight) − ½ ln det Σ − 3/2 ln 2π`.
    log_norm: f64,
}

impl Component {
    fn new(weight: f64, mean: [f64; 3], cov: [[f64; 3]; 3], reg: f64) -> Option<Self> {
        let mut c = cov;
        for (i, row) in c.iter_mut().enumerate() {
            row[i] += reg;
        }
        let det = c[0][0] * (c[1][1] * c[2][2] - c[1][2] * c[2][1]) - c[0][1] * (c[1][0] * c[2][2] - c[1][2] * c[2][0])
            + c[0][2] * (c[1][0] * c[2][1] - c[1][1] * c[2][0]);
        if det <= 1e-30 || weight <= 0.0 {
            return None;
        }
        let inv = [
            [(c[1][1] * c[2][2] - c[1][2] * c[2][1]) / det, (c[0][2] * c[2][1] - c[0][1] * c[2][2]) / det, (c[0][1] * c[1][2] - c[0][2] * c[1][1]) / det],
            [(c[1][2] * c[2][0] - c[1][0] * c[2][2]) / det, (c[0][0] * c[2][2] - c[0][2] * c[2][0]) / det, (c[0][2] * c[1][0] - c[0][0] * c[1][2]) / det],
            [(c[1][0] * c[2][1] - c[1][1] * c[2][0]) / det, (c[0][1] * c[2][0] - c[0][0] * c[2][1]) / det, (c[0][0] * c[1][1] - c[0][1] * c[1][0]) / det],
        ];
        let log_norm = weight.ln() - 0.5 * det.ln() - 1.5 * (2.0 * std::f64::consts::PI).ln();
        Some(Component { weight: weight as f32, mean: mean.map(|v| v as f32), cov: c.map(|r| r.map(|v| v as f32)), inv, log_norm })
    }

    /// `ln(weight · N(z; μ, Σ))`.
    #[inline]
    pub fn log_weighted(&self, z: [f32; 3]) -> f64 {
        let d = [(z[0] - self.mean[0]) as f64, (z[1] - self.mean[1]) as f64, (z[2] - self.mean[2]) as f64];
        let m = &self.inv;
        let q = d[0] * (m[0][0] * d[0] + m[0][1] * d[1] + m[0][2] * d[2])
            + d[1] * (m[1][0] * d[0] + m[1][1] * d[1] + m[1][2] * d[2])
            + d[2] * (m[2][0] * d[0] + m[2][1] * d[1] + m[2][2] * d[2]);
        self.log_norm - 0.5 * q
    }
}

/// A Gaussian mixture over RGB (`0..=1`).
#[derive(Clone, Debug, PartialEq)]
pub struct Gmm {
    pub comps: Vec<Component>,
    /// Ridge added to covariance diagonals.
    pub reg: f32,
}

/// Default covariance ridge: about one 8-bit level of standard deviation.
pub const DEFAULT_REG: f32 = 2e-5;

#[derive(Default, Clone, Copy)]
struct Acc {
    n: f64,
    s: [f64; 3],
    ss: [[f64; 3]; 3],
}

impl Acc {
    fn add(&mut self, z: [f32; 3], w: f64) {
        let z = z.map(|v| v as f64);
        self.n += w;
        for i in 0..3 {
            self.s[i] += w * z[i];
            for j in 0..3 {
                self.ss[i][j] += w * z[i] * z[j];
            }
        }
    }
    fn component(&self, total: f64, reg: f32) -> Option<Component> {
        if self.n <= 0.0 {
            return None;
        }
        let m = self.s.map(|v| v / self.n);
        let mut c = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                c[i][j] = self.ss[i][j] / self.n - m[i] * m[j];
            }
        }
        Component::new(self.n / total, m, c, reg as f64)
    }
}

impl Gmm {
    /// Fits `k` components (fewer if there are not enough distinct samples). `None` without samples.
    pub fn fit(samples: &[[f32; 3]], k: usize, reg: f32) -> Option<Gmm> {
        Self::fit_em(samples, k, reg, 3)
    }

    /// [`Gmm::fit`] with an explicit number of EM iterations after k-means.
    pub fn fit_em(samples: &[[f32; 3]], k: usize, reg: f32, em_iters: usize) -> Option<Gmm> {
        if samples.is_empty() {
            return None;
        }
        let k = k.clamp(1, samples.len());
        let centers = kmeans(samples, k, 10);
        let mut labels = vec![0usize; samples.len()];
        for (l, z) in labels.iter_mut().zip(samples) {
            *l = nearest(&centers, *z);
        }
        let mut g = Self::from_labels(samples, &labels, centers.len(), reg)?;
        for _ in 0..em_iters {
            g = g.em_step(samples).unwrap_or(g);
        }
        Some(g)
    }

    /// Component statistics from hard assignments.
    pub fn from_labels(samples: &[[f32; 3]], labels: &[usize], k: usize, reg: f32) -> Option<Gmm> {
        let mut acc = vec![Acc::default(); k];
        for (z, l) in samples.iter().zip(labels) {
            acc[*l].add(*z, 1.0);
        }
        let total = samples.len() as f64;
        let comps: Vec<Component> = acc.iter().filter_map(|a| a.component(total, reg)).collect();
        (!comps.is_empty()).then_some(Gmm { comps, reg })
    }

    /// One EM step (soft responsibilities).
    fn em_step(&self, samples: &[[f32; 3]]) -> Option<Gmm> {
        let k = self.comps.len();
        let mut acc = vec![Acc::default(); k];
        let mut lw = vec![0.0f64; k];
        for z in samples {
            let mut mx = f64::MIN;
            for (c, l) in self.comps.iter().zip(lw.iter_mut()) {
                *l = c.log_weighted(*z);
                mx = mx.max(*l);
            }
            let mut sum = 0.0;
            for l in lw.iter_mut() {
                *l = (*l - mx).exp();
                sum += *l;
            }
            for (a, l) in acc.iter_mut().zip(&lw) {
                let r = l / sum;
                if r > 1e-6 {
                    a.add(*z, r);
                }
            }
        }
        let total = samples.len() as f64;
        let comps: Vec<Component> = acc.iter().filter_map(|a| a.component(total, self.reg)).collect();
        (!comps.is_empty()).then_some(Gmm { comps, reg: self.reg })
    }

    /// GrabCut update: assign each sample to its most likely component and re-estimate.
    pub fn refit(&self, samples: &[[f32; 3]]) -> Option<Gmm> {
        let labels: Vec<usize> = samples.iter().map(|z| self.component_of(*z)).collect();
        Self::from_labels(samples, &labels, self.comps.len(), self.reg)
    }

    /// Index of the most likely component for `z`.
    pub fn component_of(&self, z: [f32; 3]) -> usize {
        let mut best = 0;
        let mut bv = f64::MIN;
        for (i, c) in self.comps.iter().enumerate() {
            let v = c.log_weighted(z);
            if v > bv {
                bv = v;
                best = i;
            }
        }
        best
    }

    /// `ln p(z)`.
    pub fn log_prob(&self, z: [f32; 3]) -> f64 {
        let mut mx = f64::MIN;
        let mut vals = [0.0f64; 16];
        let n = self.comps.len().min(16);
        for (v, c) in vals.iter_mut().zip(&self.comps) {
            *v = c.log_weighted(z);
            mx = mx.max(*v);
        }
        let s: f64 = vals[..n].iter().map(|v| (v - mx).exp()).sum();
        mx + s.ln()
    }

    /// `−ln p(z)`, clamped to a finite range.
    pub fn neg_log(&self, z: [f32; 3]) -> f32 {
        (-self.log_prob(z)).clamp(-50.0, 200.0) as f32
    }
}

fn nearest(centers: &[[f32; 3]], z: [f32; 3]) -> usize {
    let mut best = 0;
    let mut bd = f32::MAX;
    for (i, c) in centers.iter().enumerate() {
        let d = super::d2(*c, z);
        if d < bd {
            bd = d;
            best = i;
        }
    }
    best
}

/// k-means with k-means++ seeding (deterministic seed). Drops empty clusters.
pub fn kmeans(samples: &[[f32; 3]], k: usize, iters: usize) -> Vec<[f32; 3]> {
    let mut rng = Rng::new(0x5EED + samples.len() as u64);
    let mut centers = vec![samples[(rng.next_u64() % samples.len() as u64) as usize]];
    let mut dist: Vec<f32> = samples.iter().map(|z| super::d2(*z, centers[0])).collect();
    while centers.len() < k {
        let total: f64 = dist.iter().map(|d| *d as f64).sum();
        if total <= 1e-12 {
            break;
        }
        let mut t = rng.f32() as f64 * total;
        let mut pick = samples.len() - 1;
        for (i, d) in dist.iter().enumerate() {
            t -= *d as f64;
            if t <= 0.0 {
                pick = i;
                break;
            }
        }
        let c = samples[pick];
        centers.push(c);
        for (d, z) in dist.iter_mut().zip(samples) {
            *d = d.min(super::d2(*z, c));
        }
    }
    for _ in 0..iters {
        let mut sums = vec![([0.0f64; 3], 0usize); centers.len()];
        for z in samples {
            let i = nearest(&centers, *z);
            for (acc, v) in sums[i].0.iter_mut().zip(z) {
                *acc += *v as f64;
            }
            sums[i].1 += 1;
        }
        let new: Vec<[f32; 3]> = sums.iter().filter(|s| s.1 > 0).map(|(s, n)| s.map(|v| (v / *n as f64) as f32)).collect();
        let done = new == centers;
        centers = new;
        if done {
            break;
        }
    }
    centers
}
