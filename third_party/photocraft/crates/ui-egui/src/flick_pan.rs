//! Preferences › Tools › Enable Flick Panning (#204): letting go of a Hand-tool drag (the
//! selected Hand, the Space-held Hand or a middle-button drag) while the pointer is still moving
//! keeps the view gliding that way, slowing smoothly to a stop, as in Photoshop. Any click, new
//! drag or key press stops it. Off, the view stops where the drag ended.
//!
//! [`Flick`] is the pure state (velocity estimate and decay); [`frame`] drives it from one
//! canvas frame's input and returns the screen-space pan to apply.

use egui::{Context, Id, Vec2};

/// Pointer movement this recent (seconds) sets the release velocity.
const WINDOW: f64 = 0.08;
/// Time constant of the exponential friction (seconds): a fast flick stops in under a second.
const TAU: f32 = 0.18;
/// A release slower than this (screen points per second) does not glide.
const MIN_START: f32 = 150.0;
/// The glide stops below this speed.
const MIN_SPEED: f32 = 15.0;
/// Fastest glide (screen points per second); a hostile or glitchy delta can't fling the view away.
const MAX_SPEED: f32 = 6000.0;
/// Longest frame step the glide integrates; a stalled frame doesn't jump the view.
const MAX_DT: f32 = 0.1;

/// One canvas's flick state.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Flick {
    /// Recent drag steps: (time, frame interval, screen delta).
    samples: Vec<(f64, f32, Vec2)>,
    /// A Hand drag was in progress last frame.
    dragging: bool,
    /// The glide velocity in screen points per second, while gliding.
    velocity: Option<Vec2>,
    /// Time of the previous frame.
    last: Option<f64>,
}

impl Flick {
    /// Frame interval since the previous call, clamped to `0..=MAX_DT`; records `now`.
    fn tick(&mut self, now: f64) -> f32 {
        let dt = self.last.map_or(0.0, |l| (now - l) as f32);
        self.last = Some(now);
        if dt.is_finite() { dt.clamp(0.0, MAX_DT) } else { 0.0 }
    }

    /// A Hand drag moved the pointer by `delta` this frame (`dt` after the previous one).
    pub fn track(&mut self, now: f64, dt: f32, delta: Vec2) {
        self.velocity = None;
        self.dragging = true;
        if delta.is_finite() && dt.is_finite() {
            self.samples.push((now, dt, delta));
        }
        self.samples.retain(|s| now - s.0 <= WINDOW);
    }

    /// Pointer velocity over the last [`WINDOW`] before `now`, in screen points per second.
    pub fn release_velocity(&self, now: f64) -> Vec2 {
        let (mut d, mut t) = (Vec2::ZERO, 0.0f32);
        for &(at, dt, delta) in &self.samples {
            if now - at <= WINDOW {
                d += delta;
                t += dt;
            }
        }
        if t <= 1e-4 {
            return Vec2::ZERO;
        }
        let v = d / t;
        if !v.is_finite() {
            return Vec2::ZERO;
        }
        let speed = v.length();
        if speed > MAX_SPEED { v * (MAX_SPEED / speed) } else { v }
    }

    /// The drag ended at `now`: start gliding if `enabled` and the pointer was still moving fast
    /// enough. Returns whether a glide started.
    pub fn release(&mut self, now: f64, enabled: bool) -> bool {
        self.dragging = false;
        let v = self.release_velocity(now);
        self.samples.clear();
        self.velocity = (enabled && v.length() >= MIN_START).then_some(v);
        self.velocity.is_some()
    }

    /// Advance the glide by `dt` seconds: the screen-space distance moved, or `None` when not
    /// gliding (or it just came to rest).
    pub fn step(&mut self, dt: f32) -> Option<Vec2> {
        let v = self.velocity?;
        let dt = if dt.is_finite() { dt.clamp(0.0, MAX_DT) } else { 0.0 };
        // Exact integral of v·e^(−t/τ) over the step, then the decayed velocity.
        let decay = (-dt / TAU).exp();
        let moved = v * (TAU * (1.0 - decay));
        let next = v * decay;
        self.velocity = (next.is_finite() && next.length() >= MIN_SPEED).then_some(next);
        moved.is_finite().then_some(moved)
    }

    /// Stop any glide (a click, new drag or key press).
    pub fn stop(&mut self) {
        self.velocity = None;
        self.samples.clear();
    }

    pub fn gliding(&self) -> bool {
        self.velocity.is_some()
    }
}

/// Drive canvas `id`'s flick for this frame. `hand_drag` is whether a Hand drag moved the view by
/// `delta` (screen points) this frame; `enabled` is the preference. Returns the screen-space
/// distance the glide moves the view this frame (the same sense as a drag delta).
pub fn frame(ctx: &Context, id: Id, hand_drag: bool, delta: Vec2, enabled: bool) -> Option<Vec2> {
    let (now, interrupt) = ctx.input(|i| {
        let key = i.events.iter().any(|e| matches!(e, egui::Event::Key { pressed: true, .. }));
        (i.time, i.pointer.any_pressed() || key)
    });
    let mut flick = ctx.data(|d| d.get_temp::<Flick>(id)).unwrap_or_default();
    let dt = flick.tick(now);
    let moved = if hand_drag {
        flick.track(now, dt, delta);
        None
    } else {
        if flick.dragging {
            flick.release(now, enabled);
        }
        if interrupt || !enabled {
            flick.stop();
        }
        flick.step(dt)
    };
    // A moving drag asks for one more frame, so its release is seen even if nothing else repaints.
    if flick.gliding() || (hand_drag && delta != Vec2::ZERO) {
        ctx.request_repaint();
    }
    ctx.data_mut(|d| d.insert_temp(id, flick));
    moved
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::vec2;

    const DT: f32 = 1.0 / 60.0;

    fn flung(speed_per_frame: f32) -> (Flick, f64) {
        let mut f = Flick::default();
        let mut t = 0.0;
        for _ in 0..10 {
            t += f64::from(DT);
            f.track(t, DT, vec2(speed_per_frame, 0.0));
        }
        (f, t + f64::from(DT))
    }

    #[test]
    fn velocity_comes_from_the_last_moments_of_the_drag() {
        let (f, now) = flung(20.0);
        let v = f.release_velocity(now);
        assert!((v.x - 1200.0).abs() < 1.0, "{v:?}");
        assert_eq!(v.y, 0.0);
        // Held still for a while before letting go: no velocity.
        assert_eq!(f.release_velocity(now + 0.5), Vec2::ZERO);
    }

    #[test]
    fn a_fast_release_glides_and_decays_to_a_stop() {
        let (mut f, now) = flung(20.0);
        assert!(f.release(now, true));
        let mut total = 0.0;
        let mut last = f32::INFINITY;
        let mut frames = 0;
        while let Some(d) = f.step(DT) {
            assert!(d.x > 0.0 && d.x < last, "decelerates: {d:?}");
            last = d.x;
            total += d.x;
            frames += 1;
            assert!(frames < 200, "stops");
        }
        // About 1 s of glide at most, covering about v·τ.
        assert!((20..=70).contains(&frames), "{frames} frames");
        assert!(total > 150.0 && total < 1200.0 * TAU + 1.0, "{total}");
        assert!(!f.gliding());
    }

    #[test]
    fn disabled_or_slow_releases_do_not_glide() {
        let (mut f, now) = flung(20.0);
        assert!(!f.release(now, false));
        assert_eq!(f.step(DT), None);
        let (mut f, now) = flung(1.0);
        assert!(!f.release(now, true), "60 pt/s is a stop, not a flick");
    }

    #[test]
    fn stop_and_hostile_numbers() {
        let (mut f, now) = flung(20.0);
        f.release(now, true);
        f.stop();
        assert_eq!(f.step(DT), None);
        // Huge deltas are clamped; NaN deltas and steps are ignored.
        let (mut f, now) = flung(1.0e9);
        f.track(now - 0.001, DT, vec2(f32::NAN, 0.0));
        assert!(f.release_velocity(now).length() <= MAX_SPEED + 1.0);
        f.release(now, true);
        let d = f.step(f32::NAN);
        assert!(d.is_none_or(|d| d.is_finite()));
        let d = f.step(f32::INFINITY).unwrap_or_default();
        assert!(d.is_finite() && d.length() <= MAX_SPEED * MAX_DT);
    }
}
