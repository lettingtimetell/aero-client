//! A scripted camera path, so a replay's *view* is as reproducible as its packets.
//!
//! A deterministic in-world benchmark needs more than a deterministic packet stream, which is
//! only half of one. The other half is where the camera was pointing, because
//! everything the terrain renderer reports — visible sections, quads, draw calls — is a function
//! of the frustum.
//!
//! # Why the path has to *move*
//!
//! §1.6 measures terrain CPU at radius 6 as **0.072 ms with the cull cache warm and 0.142 ms
//! with the camera moving** — very nearly a factor of two. A bench whose camera stands still
//! therefore measures the cached path and nothing else, which is the wrong number to optimise
//! against and the one most likely to flatter a change. Every script here except
//! [`CameraScript::Fixed`] moves the camera and turns it on every single frame.
//!
//! # Why it is closed-form
//!
//! [`CameraPath::pose_at`] is a pure function of the frame index. Nothing is integrated, so
//! there is no accumulated state that could differ between two runs, and frame 12 000 can be
//! asserted directly in a test without stepping to it. It reads no clock: the frame index is
//! supplied by the caller, which increments it once per rendered frame.
//!
//! # The anchor
//!
//! A capture places the player wherever the server placed them, which is not known until
//! `S08PacketPlayerPosLook` arrives. The path is therefore expressed **relative to an anchor**
//! that the driver sets from the first position correction of the session — a deterministic
//! frame, because the packets themselves are deterministic. Until an anchor exists the path
//! produces nothing and the camera behaves exactly as it always did.

use mc_math::DVec3;

/// The eye height the pose's yaw and pitch are computed for.
///
/// [`Client::place_camera`](crate::app::Client) puts the camera at
/// `PhysicsState::eye_pos()`, which is the player's feet plus this. A pose names the *feet*
/// position, so the look angles have to be worked out from a point this far above it or the
/// camera would aim slightly past the anchor. Mirrors `mc_physics::state::PLAYER_EYE_HEIGHT`;
/// the sneaking adjustment is not reproduced because a scripted path never sneaks.
pub const EYE_HEIGHT: f64 = mc_physics::state::PLAYER_EYE_HEIGHT as f64;

/// Default orbit radius, in blocks.
pub const DEFAULT_RADIUS: f64 = 8.0;

/// Default frames per revolution.
///
/// 1800 frames is ten seconds at a 180 fps floor and under a second at the ~2000 fps §1.4
/// measures, so a run of any plausible length sees the camera sweep the whole scene repeatedly
/// rather than crawling across one corner of it.
pub const DEFAULT_PERIOD_FRAMES: u64 = 1_800;

/// Default height of the orbit above the anchor, in blocks.
pub const DEFAULT_HEIGHT: f64 = 3.0;

/// Vertical amplitude of the orbit's bob, in blocks. Non-zero so the camera's Y moves too and
/// the frustum is never merely rotating about a fixed point.
pub const DEFAULT_BOB: f64 = 2.0;

/// Which scripted path a replay drives the camera along.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum CameraScript {
    /// Leave the camera where the session put it.
    ///
    /// This is the **control** for the path itself: with a still camera the terrain phase's
    /// visibility cache never misses, so a run under `Fixed` and a run under [`CameraScript::Orbit`]
    /// on the same capture must report different visible/draw counts. If they did not, the
    /// scripted path would not be doing anything.
    Fixed,
    /// Circle the anchor at a fixed radius, looking inward, bobbing vertically.
    Orbit {
        /// Distance from the anchor, in blocks.
        radius: f64,
        /// Frames per revolution.
        period_frames: u64,
    },
}

impl Default for CameraScript {
    fn default() -> Self {
        CameraScript::Orbit {
            radius: DEFAULT_RADIUS,
            period_frames: DEFAULT_PERIOD_FRAMES,
        }
    }
}

/// One scripted camera position and orientation, in the same units the physics predictor uses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose {
    /// The player's **feet**, which is what `PhysicsState::pos` holds.
    pub pos: DVec3,
    /// java: `Entity.rotationYaw`, degrees.
    pub yaw: f32,
    /// java: `Entity.rotationPitch`, degrees, positive looking down.
    pub pitch: f32,
}

/// A script plus the anchor it is relative to and the frame counter it is evaluated at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraPath {
    script: CameraScript,
    anchor: Option<DVec3>,
    frame: u64,
}

impl CameraPath {
    /// A path that will follow `script` once it has an anchor.
    #[must_use]
    pub fn new(script: CameraScript) -> CameraPath {
        CameraPath {
            script,
            anchor: None,
            frame: 0,
        }
    }

    /// The script in use.
    #[must_use]
    pub fn script(&self) -> CameraScript {
        self.script
    }

    /// The anchor, once one has been set.
    #[must_use]
    pub fn anchor(&self) -> Option<DVec3> {
        self.anchor
    }

    /// How many frames this path has been stepped for.
    #[must_use]
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Whether the path is producing poses yet.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.anchor.is_some() && !matches!(self.script, CameraScript::Fixed)
    }

    /// Fix the point the path is relative to. Ignored after the first call, so a mid-session
    /// teleport cannot move the bench's scene out from under it.
    pub fn set_anchor(&mut self, anchor: DVec3) {
        if self.anchor.is_none() {
            tracing::info!(?anchor, script = ?self.script, "scripted camera path anchored");
            self.anchor = Some(anchor);
        }
    }

    /// The pose at an arbitrary frame index. Pure: same inputs, same output, no clock.
    ///
    /// `None` before an anchor exists or under [`CameraScript::Fixed`], which both mean "leave
    /// the camera alone".
    #[must_use]
    pub fn pose_at(&self, frame: u64) -> Option<Pose> {
        let anchor = self.anchor?;
        let CameraScript::Orbit {
            radius,
            period_frames,
        } = self.script
        else {
            return None;
        };
        let period = period_frames.max(1);
        // `frame % period` first, so a long run cannot lose precision in the division and so
        // frame `n` and frame `n + period` are bit-identical rather than merely close.
        let theta = (frame % period) as f64 / period as f64 * std::f64::consts::TAU;
        let height = DEFAULT_HEIGHT + DEFAULT_BOB * (2.0 * theta).sin();
        let pos = DVec3::new(
            anchor.x + radius * theta.cos(),
            anchor.y + height,
            anchor.z + radius * theta.sin(),
        );

        // Look from the *eye* back at the anchor. java's convention (`Entity.getLook`, and
        // `app::interact::look_vector` here) is `dir = (-sin yaw, -sin pitch, cos yaw)`, so the
        // inverse is `yaw = atan2(-dx, dz)` and `pitch = asin(-dy / |d|)`.
        let eye = DVec3::new(pos.x, pos.y + EYE_HEIGHT, pos.z);
        let d = DVec3::new(anchor.x - eye.x, anchor.y - eye.y, anchor.z - eye.z);
        let horizontal = (d.x * d.x + d.z * d.z).sqrt();
        let yaw = (-d.x).atan2(d.z).to_degrees();
        let pitch = (-d.y).atan2(horizontal).to_degrees();
        Some(Pose {
            pos,
            yaw: yaw as f32,
            pitch: pitch as f32,
        })
    }

    /// The pose for this frame, then advance the counter.
    ///
    /// The counter advances whether or not a pose came out, so that a path anchored late still
    /// has a frame index that means "frames since the session started".
    pub fn step(&mut self) -> Option<Pose> {
        let pose = self.pose_at(self.frame);
        self.frame += 1;
        pose
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn anchored() -> CameraPath {
        let mut path = CameraPath::new(CameraScript::default());
        path.set_anchor(DVec3::new(100.5, 64.0, -200.5));
        path
    }

    #[test]
    fn nothing_is_produced_before_an_anchor_exists() {
        let mut path = CameraPath::new(CameraScript::default());
        assert!(!path.is_active());
        assert_eq!(path.step(), None);
        assert_eq!(path.frame(), 1, "the counter still advances");
        path.set_anchor(DVec3::new(1.0, 2.0, 3.0));
        assert!(path.is_active());
        assert!(path.step().is_some());
    }

    #[test]
    fn the_anchor_is_taken_once_and_never_moved() {
        let mut path = CameraPath::new(CameraScript::default());
        path.set_anchor(DVec3::new(1.0, 2.0, 3.0));
        path.set_anchor(DVec3::new(9.0, 9.0, 9.0));
        assert_eq!(path.anchor(), Some(DVec3::new(1.0, 2.0, 3.0)));
    }

    #[test]
    fn fixed_produces_no_pose_at_all() {
        let mut path = CameraPath::new(CameraScript::Fixed);
        path.set_anchor(DVec3::ZERO);
        assert!(!path.is_active());
        assert_eq!(path.pose_at(0), None);
    }

    /// The property the bench depends on: the pose is a function of the frame index and nothing
    /// else, so two runs that reach frame `n` by any route see the same view.
    #[test]
    fn the_pose_is_a_pure_function_of_the_frame_index() {
        let path = anchored();
        for frame in [0u64, 1, 17, 899, 1799, 1800, 5_400, 1_000_000] {
            assert_eq!(path.pose_at(frame), path.pose_at(frame));
        }
        // Stepping and evaluating directly must agree.
        let mut stepped = anchored();
        for frame in 0..64u64 {
            assert_eq!(stepped.step(), path.pose_at(frame));
        }
    }

    /// A period is a period: frame `n` and frame `n + period` are the same pose, bit for bit.
    #[test]
    fn the_orbit_closes_exactly() {
        let path = anchored();
        assert_eq!(path.pose_at(0), path.pose_at(DEFAULT_PERIOD_FRAMES));
        assert_eq!(path.pose_at(37), path.pose_at(DEFAULT_PERIOD_FRAMES + 37));
    }

    /// §1.6's reason for this module existing: a still camera measures the cull cache. Every
    /// consecutive frame must differ in **both** position and yaw, or the path is not defeating
    /// it.
    #[test]
    fn every_consecutive_frame_moves_and_turns() {
        let path = anchored();
        for frame in 0..600u64 {
            let a = path.pose_at(frame).expect("a pose");
            let b = path.pose_at(frame + 1).expect("a pose");
            assert_ne!(a.pos, b.pos, "frame {frame}: the camera stood still");
            assert_ne!(a.yaw, b.yaw, "frame {frame}: the camera did not turn");
            assert_ne!(a.pos.y, b.pos.y, "frame {frame}: the bob is not bobbing");
        }
    }

    /// The camera has to be looking at the scene, or a bench renders the sky. Reconstructing the
    /// look vector from the yaw/pitch the path produced must point back at the anchor.
    #[test]
    fn the_camera_looks_at_the_anchor() {
        let path = anchored();
        let anchor = path.anchor().expect("anchored");
        for frame in [0u64, 200, 450, 900, 1_350] {
            let pose = path.pose_at(frame).expect("a pose");
            let eye = DVec3::new(pose.pos.x, pose.pos.y + EYE_HEIGHT, pose.pos.z);
            let look = crate::app::interact::look_vector(pose.yaw, pose.pitch);
            let to_anchor = DVec3::new(anchor.x - eye.x, anchor.y - eye.y, anchor.z - eye.z);
            let length = to_anchor.length();
            let unit = DVec3::new(
                to_anchor.x / length,
                to_anchor.y / length,
                to_anchor.z / length,
            );
            let dot = look.x * unit.x + look.y * unit.y + look.z * unit.z;
            // `look_vector` uses vanilla's 65536-entry sine table, which is good to ~1e-4.
            assert!(
                dot > 0.999,
                "frame {frame}: the view is {dot} off the anchor ({pose:?})"
            );
        }
    }

    /// The radius is honoured, so a caller can point the bench at a bigger or smaller sweep.
    #[test]
    fn the_radius_is_the_horizontal_distance_from_the_anchor() {
        let mut path = CameraPath::new(CameraScript::Orbit {
            radius: 20.0,
            period_frames: 400,
        });
        path.set_anchor(DVec3::ZERO);
        for frame in 0..400u64 {
            let pose = path.pose_at(frame).expect("a pose");
            let r = (pose.pos.x * pose.pos.x + pose.pos.z * pose.pos.z).sqrt();
            assert!((r - 20.0).abs() < 1e-9, "frame {frame}: radius {r}");
        }
    }
}
