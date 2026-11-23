//! Distance attenuation, the listener transform, and stereo panning.
//!
// java: SoundManager.setListener, ISound.AttenuationType, PositionedSound.attenuationType

use mc_math::{DVec3, Vec3};

/// How a sound's volume falls off with distance from the listener.
///
/// 1.8.9 only ever uses these two (`ISound.AttenuationType`) — later versions add more. `None`
/// is used for menu clicks/music and the rider-only minecart loop; `Linear` for ordinary world
/// sounds. [`crate::PlayRequest`] keeps this independent from its optional position, as `ISound`
/// does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttenuationType {
    /// No falloff — full volume regardless of distance (menu sounds, music, GUI clicks).
    None,
    /// Fades linearly to silence at [`fade_distance`], vanilla's only positional mode.
    Linear,
}

impl AttenuationType {
    /// The OpenAL attenuation-model constant vanilla passes to `SoundSystem.newSource`
    /// (`AttenuationType.getTypeInt`). Not consumed by this crate's own math — kept for parity
    /// with the Java source and in case a future backend wants it.
    pub const fn type_int(self) -> i32 {
        match self {
            AttenuationType::None => 0,
            AttenuationType::Linear => 2,
        }
    }
}

/// The distance at which a [`AttenuationType::Linear`] sound fades to silence.
///
/// java: `SoundManager.playSound` — `float f1 = 16.0F; if (f > 1.0F) { f1 *= f; }` where `f` is
/// `ISound.getVolume()`. A louder-than-normal sound (an explosion, `volume > 1.0`) is audible
/// proportionally further away; a quieter one still fades out at the normal 16 blocks.
pub fn fade_distance(sound_volume: f32) -> f32 {
    if sound_volume > 1.0 {
        16.0 * sound_volume
    } else {
        16.0
    }
}

/// # Example
/// ```
/// use mc_audio::{attenuate, AttenuationType};
///
/// assert_eq!(attenuate(0.0, AttenuationType::Linear, 1.0), 1.0);
/// assert_eq!(attenuate(16.0, AttenuationType::Linear, 1.0), 0.0); // fades out by 16 blocks
/// assert_eq!(attenuate(1000.0, AttenuationType::None, 1.0), 1.0); // ambient: never fades
/// ```
///
/// # Note on provenance
///
/// Vanilla never computes this in Java — `SoundManager` only picks an attenuation *model* and
/// a rolloff distance and hands them to `paulscode.sound.SoundSystem`, which drives real OpenAL
/// (`LibraryLWJGLOpenAL`) to do the actual distance falloff every mix frame. That library ships
/// here only as a compiled jar (`jars/libraries/com/paulscode/...`), not decompiled source, so
/// this is a from-first-principles reconstruction of OpenAL's `AL_LINEAR_DISTANCE_CLAMPED`
/// model with the parameters vanilla is known to configure (reference distance `0`, rolloff
/// factor `1`, max distance = [`fade_distance`]) — not a transcription of paulscode's internals.
/// It reproduces the two externally-observable facts about vanilla's behaviour: full volume at
/// the listener, silence at (and beyond) the fade distance, linear in between.
pub fn attenuate(distance: f32, kind: AttenuationType, sound_volume: f32) -> f32 {
    match kind {
        AttenuationType::None => 1.0,
        AttenuationType::Linear => {
            if !distance.is_finite() || distance <= 0.0 {
                return 1.0;
            }
            let max_distance = fade_distance(sound_volume);
            if distance >= max_distance {
                0.0
            } else {
                1.0 - distance / max_distance
            }
        }
    }
}

/// The listener's world position and facing, interpolated the way the render frame is.
///
/// java: `SoundManager.setListener` — this crate keeps its own copy (rather than depending on
/// `mc-render`'s `Camera`, which S19 has no dependency on) built from the same inputs
/// (`EntityPlayer` position/rotation, previous-tick values, `partial_ticks`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Listener {
    /// Eye position in world space.
    pub position: DVec3,
    /// Forward ("at") vector, matching OpenAL's `AL_ORIENTATION` `at` half.
    pub at: Vec3,
    /// Up vector, matching OpenAL's `AL_ORIENTATION` `up` half.
    pub up: Vec3,
}

impl Listener {
    /// A listener at the origin looking down `-Z` with `+Y` up — used before the first player
    /// position is known (e.g. the main menu).
    pub fn origin() -> Self {
        Self {
            position: DVec3::ZERO,
            at: Vec3::new(0.0, 0.0, -1.0),
            up: Vec3::new(0.0, 1.0, 0.0),
        }
    }

    /// Builds the listener transform exactly as `SoundManager.setListener` does: linear
    /// interpolation of position and rotation between the previous and current tick by
    /// `partial_ticks`, eye height added to Y, and the same six-term rotation-matrix expansion
    /// Java uses to turn yaw/pitch into `at`/`up` vectors (rather than, say, a quaternion —
    /// matching it term-for-term keeps this bit-comparable to the Java trig, which matters
    /// since both use the [`mc_math`] sin/cos LUT, not `f32::sin`/`cos`).
    #[allow(clippy::too_many_arguments)]
    pub fn interpolated(
        prev_pos: DVec3,
        pos: DVec3,
        eye_height: f64,
        prev_pitch: f32,
        pitch: f32,
        prev_yaw: f32,
        yaw: f32,
        partial_ticks: f32,
    ) -> Self {
        const DEG_TO_RAD: f32 = 0.017_453_292;
        let f = prev_pitch + (pitch - prev_pitch) * partial_ticks;
        let f1 = prev_yaw + (yaw - prev_yaw) * partial_ticks;
        let position =
            prev_pos + (pos - prev_pos) * partial_ticks as f64 + DVec3::new(0.0, eye_height, 0.0);

        let f2 = mc_math::cos((f1 + 90.0) * DEG_TO_RAD);
        let f3 = mc_math::sin((f1 + 90.0) * DEG_TO_RAD);
        let f4 = mc_math::cos(-f * DEG_TO_RAD);
        let f5 = mc_math::sin(-f * DEG_TO_RAD);
        let f6 = mc_math::cos((-f + 90.0) * DEG_TO_RAD);
        let f7 = mc_math::sin((-f + 90.0) * DEG_TO_RAD);
        let f8 = f2 * f4;
        let f9 = f3 * f4;
        let f10 = f2 * f6;
        let f11 = f3 * f6;

        Self {
            position,
            at: Vec3::new(f8, f5, f9),
            up: Vec3::new(f10, f7, f11),
        }
    }

    /// Distance from the listener to a world position, as the `f32` vanilla's `ISound` interface
    /// works in (`getXPosF`/etc.) — attenuation and panning are both computed at that precision.
    pub fn distance_to(&self, target: DVec3) -> f32 {
        (target - self.position).length() as f32
    }
}

/// A simple stereo pan for `target` relative to `listener`, in `[-1.0, 1.0]` (`-1` = hard left,
/// `1` = hard right, `0` = centred).
///
/// This has no vanilla equivalent to port: real OpenAL performs full 3D spatialisation from the
/// source and listener positions/orientation, which this crate does not attempt to reproduce
/// (see [`attenuate`]'s note on provenance — there is no Java to port here either, since the
/// panning math lives entirely inside OpenAL). This is a deliberately simple addition — the
/// signed component of the direction-to-source along the listener's right vector — good enough
/// to make positional sound audibly left/right without pulling in a full spatialisation model.
pub fn pan(listener: &Listener, target: DVec3) -> f32 {
    let to_source = target - listener.position;
    if to_source.length_squared() < 1e-9 {
        return 0.0;
    }
    let to_source = to_source.as_vec3().normalize_or_zero();
    let right = listener.at.cross(listener.up).normalize_or_zero();
    if right.length_squared() < 1e-9 {
        return 0.0;
    }
    to_source.dot(right).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_is_always_full_volume() {
        for d in [0.0_f32, 1.0, 16.0, 1000.0, f32::INFINITY] {
            assert_eq!(attenuate(d, AttenuationType::None, 1.0), 1.0);
        }
    }

    #[test]
    fn linear_is_full_at_zero_and_silent_at_fade_distance() {
        assert_eq!(attenuate(0.0, AttenuationType::Linear, 1.0), 1.0);
        assert_eq!(attenuate(16.0, AttenuationType::Linear, 1.0), 0.0);
        assert_eq!(attenuate(1000.0, AttenuationType::Linear, 1.0), 0.0);
    }

    #[test]
    fn linear_is_exactly_halfway_at_half_the_fade_distance() {
        let g = attenuate(8.0, AttenuationType::Linear, 1.0);
        assert!((g - 0.5).abs() < 1e-6, "got {g}");
    }

    #[test]
    fn louder_sounds_travel_further() {
        // volume 4.0 -> fade distance 64, so 32 blocks away is still half volume, not silent.
        assert_eq!(fade_distance(4.0), 64.0);
        let g = attenuate(32.0, AttenuationType::Linear, 4.0);
        assert!((g - 0.5).abs() < 1e-6, "got {g}");
    }

    #[test]
    fn quieter_than_normal_does_not_shrink_the_fade_distance() {
        // java: `if (f > 1.0F) f1 *= f;` — only scales *up*, so volume 0.25 still fades at 16.
        assert_eq!(fade_distance(0.25), 16.0);
    }

    #[test]
    fn negative_or_nonfinite_distance_is_treated_as_at_the_listener() {
        assert_eq!(attenuate(-1.0, AttenuationType::Linear, 1.0), 1.0);
        assert_eq!(attenuate(f32::NAN, AttenuationType::Linear, 1.0), 1.0);
    }

    #[test]
    fn listener_faces_forward_at_zero_rotation() {
        let l = Listener::interpolated(DVec3::ZERO, DVec3::ZERO, 1.62, 0.0, 0.0, 0.0, 0.0, 1.0);
        // yaw=0, pitch=0 faces +Z in Minecraft (south), matching `Entity.getLookVec` at the
        // same angles: (-sin(0)*cos(0), -sin(0), cos(0)*cos(0)) = (0, 0, 1).
        assert!(l.at.x.abs() < 1e-4, "{:?}", l.at);
        assert!(l.at.z > 0.99, "{:?}", l.at);
        assert!(l.up.y > 0.99, "{:?}", l.up);
        assert!((l.position.y - 1.62).abs() < 1e-9);
    }

    #[test]
    fn listener_interpolates_position_by_partial_ticks() {
        let l = Listener::interpolated(
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(10.0, 0.0, 0.0),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.5,
        );
        assert!((l.position.x - 5.0).abs() < 1e-9);
    }

    #[test]
    fn pan_is_centered_directly_ahead() {
        let l = Listener::interpolated(DVec3::ZERO, DVec3::ZERO, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let ahead = l.position + DVec3::new(l.at.x as f64, l.at.y as f64, l.at.z as f64) * 5.0;
        let p = pan(&l, ahead);
        assert!(p.abs() < 1e-3, "expected ~0, got {p}");
    }

    #[test]
    fn pan_is_signed_left_vs_right() {
        let l = Listener::interpolated(DVec3::ZERO, DVec3::ZERO, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let right = l.at.cross(l.up).normalize_or_zero();
        let to_the_right =
            l.position + DVec3::new(right.x as f64, right.y as f64, right.z as f64) * 5.0;
        let to_the_left =
            l.position - DVec3::new(right.x as f64, right.y as f64, right.z as f64) * 5.0;
        assert!(pan(&l, to_the_right) > 0.9, "{}", pan(&l, to_the_right));
        assert!(pan(&l, to_the_left) < -0.9, "{}", pan(&l, to_the_left));
    }

    #[test]
    fn pan_at_listener_position_is_centered() {
        let l = Listener::origin();
        assert_eq!(pan(&l, l.position), 0.0);
    }
}
