//! `MobAppearance`'s elder-guardian render handoff.
//!
//! java: `client/particle/MobAppearance.java`. This particle cannot travel through the regular
//! particle atlas: its `renderParticle` creates a full `EntityGuardian`, marks it elder, and
//! renders its model under a camera-relative matrix. The game loop owns the actual call between
//! `EntityPhase::begin` and `finish`, so this module deliberately exposes the fully-resolved
//! submission rather than reaching into `mc-client`'s driver.

use glam::{Mat4, Vec3};

use crate::entity::render::EntityTransform;
use crate::entity::{
    tint, AnimParams, EntityPhase, EntityVertex, LayerBlend, ModelKind, RenderManager,
};
use crate::frame::Camera;

use super::types::{Particle, ParticleKind};

/// java: `RenderGuardian.GUARDIAN_ELDER_TEXTURE`, selected because `MobAppearance` calls
/// `EntityGuardian.setElder()` before its first rendered frame.
pub const MOB_APPEARANCE_TEXTURE: &str = "textures/entity/guardian_elder.png";

/// One ready-to-submit `MobAppearance` render. Its [`submit`](Self::submit) method writes exact
/// guardian model geometry into the normal entity phase; callers must ensure
/// [`MOB_APPEARANCE_TEXTURE`] has been loaded into that phase first.
#[derive(Clone, Debug, PartialEq)]
pub struct MobAppearanceDraw {
    /// Complete camera-relative matrix: `MobAppearance`'s camera-facing transform followed by
    /// `RendererLivingEntity`'s elder-guardian placement transform.
    pub matrix: Mat4,
    /// java: `0.05 + 0.5 * sin(((age + partial) / 30) * PI)`.
    pub alpha: f32,
    /// The particle age fraction used by the X rotation and alpha pulse.
    pub progress: f32,
    /// The custom renderer sets both lightmap coordinates to 240 before drawing the guardian.
    /// EntityPhase has no world-light channel, so this documents that its normal directional
    /// model lighting is intentionally independent of world darkness for this draw.
    pub full_bright: bool,
    /// The guardian starts after construction and is never world-ticked, so it retains the
    /// default neutral animation state. Kept public so the driver handoff is auditable.
    pub params: AnimParams,
}

impl MobAppearanceDraw {
    /// Emits the elder guardian with `MobAppearance`'s blend/depth state. Returns emitted
    /// vertices (zero only if the guardian model itself is unavailable).
    ///
    /// The phase must already contain [`MOB_APPEARANCE_TEXTURE`], normally via
    /// [`EntityPhase::load_texture`]. Missing textures intentionally use EntityPhase's standard
    /// missing-texture fallback rather than dropping a server-triggered apparition.
    pub fn submit(&self, models: &mut RenderManager, phase: &mut EntityPhase) -> usize {
        let mut verts = Vec::<EntityVertex>::new();
        models.emit_with(ModelKind::Guardian, self.matrix, &self.params, &mut verts);
        tint(&mut verts, [1.0, 1.0, 1.0, self.alpha]);
        let count = verts.len();
        phase.push_blended(
            MOB_APPEARANCE_TEXTURE,
            LayerBlend::TranslucentDepthWrite,
            &verts,
        );
        count
    }
}

/// Builds an elder-guardian submission for a live `MOB_APPEARANCE` particle.
///
/// Returns `None` before its first update (when Java's `entity` field is still null), after it
/// dies, or for every other particle kind. This makes the handoff safe to call over the entire
/// particle arena once per render frame.
#[must_use]
pub fn mob_appearance_draw(
    particle: &Particle,
    camera: &Camera,
    partial_ticks: f32,
) -> Option<MobAppearanceDraw> {
    if !matches!(particle.kind, ParticleKind::MobAppearance) || particle.dead || particle.age == 0 {
        return None;
    }

    let progress = (particle.age as f32 + partial_ticks) / particle.max_age.max(1) as f32;
    let alpha = 0.05 + 0.5 * mc_math::sin(progress * core::f32::consts::PI);

    // Exact order from `MobAppearance.renderParticle`'s matrix stack. These coordinates are
    // already camera-relative (the Java renderer executes while the world camera transform is
    // active), which is why the translation starts at the camera origin rather than particle.pos.
    let mut outer = Mat4::from_translation(Vec3::new(0.0, 1.8, 0.0));
    outer *= Mat4::from_rotation_y((180.0 - camera.yaw).to_radians());
    outer *= Mat4::from_rotation_x((60.0 - 150.0 * progress - camera.pitch).to_radians());
    outer *= Mat4::from_translation(Vec3::new(0.0, -0.4, -1.5));
    outer *= Mat4::from_scale(Vec3::splat(0.425_531_92));

    // `RenderGuardian.preRenderCallback` scales its elder entity by 2.35. `emit_with` is the
    // raw-model route, so include the otherwise-normal `RendererLivingEntity` placement here:
    // rotate 180, flip model X/Y, elder scale, and its -1.5078125 living-model lift.
    let elder = crate::entity::entity_matrix(&EntityTransform {
        rel: Vec3::ZERO,
        body_yaw: 0.0,
        scale: 2.35,
        ..EntityTransform::default()
    });
    let mut params = AnimParams::still();
    params.partial_ticks = partial_ticks;

    Some(MobAppearanceDraw {
        matrix: outer * elder,
        alpha,
        progress,
        full_bright: true,
        params,
    })
}

#[cfg(test)]
mod tests {
    use glam::DVec3;
    use mc_math::JavaRandom;

    use super::*;

    #[test]
    fn mob_appearance_waits_one_tick_then_has_exact_pulse_and_camera_transform() {
        let mut rng = JavaRandom::new(44);
        let mut particle = super::super::spawn::mob_appearance(&mut rng, DVec3::ZERO);
        let camera = Camera {
            yaw: 30.0,
            pitch: -10.0,
            ..Camera::default()
        };
        assert!(mob_appearance_draw(&particle, &camera, 0.0).is_none());

        particle.age = 15;
        let draw = mob_appearance_draw(&particle, &camera, 0.0).expect("after first tick");
        assert_eq!(draw.progress, 0.5);
        assert!((draw.alpha - 0.55).abs() < 1e-6, "{}", draw.alpha);
        assert!(draw.full_bright);
        // The outer front translation distinguishes the custom camera-facing apparition from a
        // normal guardian at the packet's world position; camera yaw changes its final matrix.
        let facing_south = mob_appearance_draw(
            &particle,
            &Camera {
                yaw: 0.0,
                pitch: -10.0,
                ..Camera::default()
            },
            0.0,
        )
        .unwrap();
        assert_ne!(draw.matrix, facing_south.matrix);
    }

    #[test]
    fn non_appearances_and_dead_appearances_have_no_entity_handoff() {
        let mut rng = JavaRandom::new(45);
        let normal = super::super::spawn::flame(&mut rng, DVec3::ZERO, DVec3::ZERO);
        assert!(mob_appearance_draw(&normal, &Camera::default(), 0.0).is_none());
        let mut appearance = super::super::spawn::mob_appearance(&mut rng, DVec3::ZERO);
        appearance.age = 1;
        appearance.dead = true;
        assert!(mob_appearance_draw(&appearance, &Camera::default(), 0.0).is_none());
    }

    #[test]
    fn appearance_handoff_builds_the_elder_guardian_mesh_and_applies_its_pulse_alpha() {
        let mut rng = JavaRandom::new(46);
        let mut particle = super::super::spawn::mob_appearance(&mut rng, DVec3::ZERO);
        particle.age = 15;
        let draw = mob_appearance_draw(&particle, &Camera::default(), 0.0).unwrap();

        let mut models = RenderManager::default();
        let mut verts = Vec::<EntityVertex>::new();
        models.emit_with(ModelKind::Guardian, draw.matrix, &draw.params, &mut verts);
        assert!(
            verts.len() > 100,
            "the handoff must emit the full guardian model, not a particle quad"
        );
        tint(&mut verts, [1.0, 1.0, 1.0, draw.alpha]);
        assert!(verts.iter().all(|v| (v.color[3] - 0.55).abs() < 1e-6));
    }
}
