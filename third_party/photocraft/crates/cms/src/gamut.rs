//! Gamut checks against a proof (output) profile: Photoshop's View › Gamut Warning.

use crate::clut::Clut;
use crate::pipeline::{Pipeline, Stage};
use crate::profile::Profile;
use crate::transform::link_stages;
use crate::{CmsError, Intent};

/// Default ΔE76 threshold above which a colour counts as out of gamut.
pub const DEFAULT_THRESHOLD: f32 = 4.0;

/// Precomputed out-of-gamut test for colours of `src` against `proof`.
///
/// A colour is out of gamut when sending it to the proof device (relative colorimetric,
/// clipping) and back to Lab moves it by more than the threshold (ΔE76). The ΔE is tabulated
/// over the source device space (33³ grid for 3 channels, 17⁴ for 4, exact otherwise).
#[derive(Debug)]
pub struct GamutCheck {
    inputs: usize,
    grid: Option<Clut>,
    exact: (Pipeline, Pipeline),
    pub threshold: f32,
}

impl GamutCheck {
    pub fn new(src: &Profile, proof: &Profile, threshold: f32) -> Result<GamutCheck, CmsError> {
        let lab = crate::Builtin::LabD50.profile();
        let direct = Pipeline::new(src.channels(), link_stages(src, lab, Intent::RelativeColorimetric, false)?);
        let mut via = link_stages(src, proof, Intent::RelativeColorimetric, false)?;
        via.push(Stage::Clamp01);
        via.extend(link_stages(proof, lab, Intent::RelativeColorimetric, false)?);
        let via = Pipeline::new(src.channels(), via);
        let de = |p: &[f32], o: &mut [f32]| {
            let (mut a, mut b) = ([0.0f32; 16], [0.0f32; 16]);
            direct.eval(p, &mut a);
            via.eval(p, &mut b);
            let d = |i: usize, s: f32| (a[i] - b[i]) * s;
            o[0] = (d(0, 100.0).powi(2) + d(1, 255.0).powi(2) + d(2, 255.0).powi(2)).sqrt();
        };
        let n = src.channels();
        let grid = match n {
            1 => Some(Clut::sample(vec![256], 1, de)),
            3 => Some(Clut::sample(vec![33; 3], 1, de)),
            4 => Some(Clut::sample(vec![17; 4], 1, de)),
            _ => None,
        };
        Ok(GamutCheck { inputs: n, grid, exact: (direct, via), threshold })
    }

    pub fn inputs(&self) -> usize {
        self.inputs
    }

    /// ΔE76 between the colour and its proof round trip.
    pub fn delta_e(&self, px: &[f32]) -> f32 {
        let mut o = [0.0f32; 1];
        match &self.grid {
            Some(g) => {
                let mut c = [0.0f32; 16];
                for (d, s) in c.iter_mut().zip(&px[..self.inputs]) {
                    *d = s.clamp(0.0, 1.0);
                }
                g.eval(&c[..self.inputs], &mut o);
            }
            None => {
                let (mut a, mut b) = ([0.0f32; 16], [0.0f32; 16]);
                self.exact.0.eval(px, &mut a);
                self.exact.1.eval(px, &mut b);
                o[0] = ((a[0] - b[0]) * 100.0).hypot((a[1] - b[1]) * 255.0).hypot((a[2] - b[2]) * 255.0);
            }
        }
        o[0]
    }

    pub fn out_of_gamut(&self, px: &[f32]) -> bool {
        self.delta_e(px) > self.threshold
    }
}
