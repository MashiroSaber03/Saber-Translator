//! All possible sample counts share a bounded rotation table per pin.
use crate::SpinPin;

pub(super) struct Plan {
    rotations: Vec<(f32, f32)>,
}
impl Plan {
    #[cfg(test)]
    pub(super) fn rotation(&self, count: usize, sample: usize) -> Option<(f32, f32)> {
        self.rotations(count)?.get(sample).copied()
    }
    pub(super) fn rotations(&self, count: usize) -> Option<&[(f32, f32)]> {
        if !(2..=64).contains(&count) {
            return None;
        }
        let start = count.checked_mul(count - 1)?.checked_div(2)?.checked_sub(1)?;
        self.rotations.get(start..start.checked_add(count)?)
    }
}
pub(super) fn prepare(pins: &[SpinPin]) -> Option<Vec<Plan>> {
    // Under 2 MiB, even for 120 overlapping pins. Larger lists keep direct sampling.
    if pins.len() > 120 {
        return None;
    }
    let mut plans = Vec::new();
    plans.try_reserve_exact(pins.len()).ok()?;
    for p in pins {
        let theta = p.blur_angle.clamp(0.0, 360.0).to_radians();
        let mut rotations = Vec::new();
        rotations.try_reserve_exact(2079).ok()?;
        for count in 2..=64 {
            for j in 0..count {
                let phi = theta * (j as f32 / (count - 1) as f32 - 0.5);
                rotations.push(phi.sin_cos());
            }
        }
        plans.push(Plan { rotations });
    }
    Some(plans)
}

/// Fixed channels keep the bilinear accumulators in registers. Tap and multiply order
/// stay identical to add_sample_premul; alpha is still multiplied by the tap weight first.
pub(super) fn sample<const N: usize>(src: &crate::image::Image, x: f32, y: f32, clip: photocraft_geom::Rect, alpha: bool, acc: &mut [f32; 8]) {
    if !x.is_finite() || !y.is_finite() || clip.is_empty() {
        return;
    }
    let fx = (x - 0.5).clamp(clip.x0 as f32, (clip.x1 - 1) as f32);
    let fy = (y - 0.5).clamp(clip.y0 as f32, (clip.y1 - 1) as f32);
    let (x0, y0) = (fx.floor() as i32, fy.floor() as i32);
    let (x1, y1) = ((x0 + 1).min(clip.x1 - 1), (y0 + 1).min(clip.y1 - 1));
    let (ax, ay) = (fx - x0 as f32, fy - y0 as f32);
    let w = src.rect.width() as usize;
    let idx = |xx: i32, yy: i32| ((yy - src.rect.y0) as usize * w + (xx - src.rect.x0) as usize) * N;
    for (i, k) in [(idx(x0, y0), (1.0 - ax) * (1.0 - ay)), (idx(x1, y0), ax * (1.0 - ay)), (idx(x0, y1), (1.0 - ax) * ay), (idx(x1, y1), ax * ay)] {
        if k <= 0.0 {
            continue;
        }
        let Some(px) = src.data.get(i..i + N).and_then(|s| s.first_chunk::<N>()) else {
            return;
        };
        let a = if alpha { px.get(N - 1).copied().unwrap_or(0.0) * k } else { k };
        if alpha {
            for (sum, value) in acc.iter_mut().zip(px).take(N - 1) {
                *sum += value * a;
            }
            if let Some(sum) = acc.get_mut(N - 1) {
                *sum += a;
            }
        } else {
            for (sum, value) in acc.iter_mut().zip(px).take(N) {
                *sum += value * a;
            }
        }
    }
}
