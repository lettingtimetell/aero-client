//! The cinematic idle camera: a slow dolly along the walkway with gentle pan, tilt and crane
//! moves on incommensurate periods, so the shot breathes, loops seamlessly (every term is a
//! sine) and never visibly repeats. The first moments ease in from a little further back.

use glam::Vec3;

use super::greenhouse::FLOOR;

/// One frame's camera.
#[derive(Debug, Clone, Copy)]
pub(super) struct Shot {
    pub(super) eye: Vec3,
    pub(super) target: Vec3,
    /// Vertical field of view, radians.
    pub(super) fov_y: f32,
}

/// How long the opening glide takes, seconds.
pub(super) const INTRO_SECONDS: f64 = 2.4;

/// A sine of period `period` seconds, evaluated in `f64` so hours of uptime stay smooth.
fn wave(t: f64, period: f64, phase: f64) -> f32 {
    ((t / period + phase).fract() * std::f64::consts::TAU).sin() as f32
}

/// Smootherstep from 0 to 1 over `0..=1`.
pub(super) fn ease(x: f64) -> f32 {
    let x = x.clamp(0.0, 1.0);
    (x * x * x * (x * (x * 6.0 - 15.0) + 10.0)) as f32
}

/// The camera at `t` seconds since the title screen appeared.
pub(super) fn shot(t: f64) -> Shot {
    let floor = FLOOR as f32;
    let base_eye = Vec3::new(21.6, floor + 3.6, 2.5);
    let base_target = Vec3::new(24.5, floor + 9.0, 40.0);
    let forward = (base_target - base_eye).normalize();
    let right = forward.cross(Vec3::Y).normalize();

    let dolly = wave(t, 97.0, 0.0) * 1.6;
    let truck = wave(t, 71.0, 0.25) * 0.8;
    let crane = wave(t, 53.0, 0.6) * 0.3;
    let pan = wave(t, 83.0, 0.4) * 1.6;
    let tilt = wave(t, 61.0, 0.1) * 0.7;

    let intro = 1.0 - ease(t / INTRO_SECONDS);
    let eye = base_eye
        + forward * dolly
        + right * truck
        + Vec3::Y * crane
        + (-forward * 3.0 + Vec3::Y * 0.7) * intro;
    let target = base_target + forward * dolly + right * pan + Vec3::Y * tilt;
    Shot {
        eye,
        target,
        fov_y: 64f32.to_radians(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shot_is_continuous_and_stays_near_the_walkway() {
        let mut last = shot(0.0);
        for i in 1..20_000 {
            let s = shot(f64::from(i) * 0.05);
            assert!(
                s.eye.distance(last.eye) < 0.2,
                "the camera jumped at step {i}"
            );
            assert!(s.eye.y > FLOOR as f32 + 2.0 && s.eye.y < FLOOR as f32 + 6.0);
            last = s;
        }
    }
}
