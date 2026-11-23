//! CPU cost of entity submission.
//!
//! **Hologram skip** (`MC_NW_F1_HOLOGRAM_SKIP=0` turns it off). Minigame lobbies, BedWars
//! generators and shop NPCs label things with invisible, empty armour stands, hundreds of them
//! at a time (the user's minigame replay tracks 300-760 entities, 388 of them armour stands). For
//! such a stand `upload_entities` computed the light sample, the 1.8 KB render parameters, the
//! texture key, tints, the placement matrix, the shared model's pose and the three layer
//! dispatches, and then drew nothing. This module decides, from the tracker alone, that a stand
//! draws nothing in that pass, so the pass can move on before any of that work.
//!
//! Why nothing is drawn (`RenderArmorStand` and its renderers, 1.8.9):
//! * `RendererLivingEntity.renderModel`: an invisible entity's body is skipped unless
//!   `!isInvisibleToPlayer(thePlayer)`, which for a non-player is only true for a spectating
//!   viewer (`Entity.isInvisibleToPlayer`). [`crate::app::appearance::BodyDraw::decide`] is that
//!   fork; the caller passes the same `viewer_is_spectator` it uses there.
//! * Its three layers (`LayerBipedArmor`, `LayerHeldItem`, `LayerCustomHead`) each start from an
//!   equipment slot and return at once when it is empty; all five slots are checked.
//! * `Render.doRenderShadowAndFire`: the shadow needs `!isInvisible()` (and a stand's
//!   `shadowSize` is 0); the fire needs `isBurning()`, which is checked.
//! * A stand is an `EntityLivingBase`, not an `EntityLiving`, so it is never leashed; it is no
//!   boss, has no beam, and gets no F3+B box while invisible (`debug_geometry_allowed`).
//! * The F3 entity count and the spectator gates run before the call; the label comes from the
//!   separate nameplate loop, which is untouched.
//!
//! What the skipped work leaves out is the pose of the shared `ModelArmorStand`, which vanilla
//! also performs before its invisibility fork. Nothing reads it: this stand's layers draw
//! nothing, and every other use of the model poses it from that entity's own parameters first;
//! `a_skipped_armor_stand_pose_leaves_no_trace` checks that posing is independent of the pose
//! before it, so dropping one pose changes no later draw.

use mc_data::EntityType;
use mc_world::World;

use crate::sync::entities::TrackedEntity;

/// Kill switch, read once per process.
pub(crate) fn hologram_skip_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("MC_NW_F1_HOLOGRAM_SKIP").is_none_or(|value| value != "0"))
}

/// Whether `upload_entities` would draw nothing at all for this entity: an invisible armour
/// stand with every equipment slot empty that is not burning, seen by a viewer who is not
/// spectating. See the module docs for the vanilla reasoning.
#[must_use]
pub(crate) fn invisible_bare_armor_stand(
    world: &World,
    tracked: &TrackedEntity,
    viewer_is_spectator: bool,
) -> bool {
    tracked.kind.entity_type() == Some(EntityType::ArmorStand)
        && tracked.render.entity.invisible
        && !viewer_is_spectator
        && tracked.equipment.iter().all(Option::is_none)
        && !crate::app::appearance::is_burning(world, tracked)
}

#[cfg(test)]
mod tests {
    use mc_render::entity::{AnimParams, ModelKind, RenderManager};

    /// The whole shared model state (every part's angles, pivots, visibility, flags). `Model` has
    /// no `PartialEq` outside mc-render's own tests; its `Debug` prints every field, floats in
    /// round-trip form.
    fn state(manager: &mut RenderManager) -> String {
        format!("{:?}", manager.model_mut(ModelKind::ArmorStand).model())
    }

    /// Armour-stand poses that differ in everything the stand's `setRotationAngles` writes.
    fn stand_params() -> Vec<AnimParams> {
        let mut out = Vec::new();
        for (arms, no_plate, small, sneaking, yaw) in [
            (false, false, false, false, 0.0_f32),
            (true, true, false, false, 37.5),
            (true, false, true, true, -120.0),
            (false, true, true, false, 300.0),
        ] {
            let mut p = AnimParams::still();
            let mut stand = p.entity.armor_stand.unwrap_or_default();
            stand.show_arms = arms;
            stand.no_base_plate = no_plate;
            stand.rotation_yaw = yaw;
            stand.head = [10.0 + yaw * 0.1, -5.0, 3.0];
            stand.body = [1.0, yaw * 0.05, -2.0];
            stand.left_arm = [-10.0, 0.0, -10.0 + yaw * 0.01];
            stand.right_arm = [-15.0, 0.0, 10.0];
            stand.left_leg = [-1.0, 0.0, -1.0];
            stand.right_leg = [1.0, 0.0, 1.0];
            p.entity.armor_stand = Some(stand);
            p.entity.child = small;
            p.entity.sneaking = sneaking;
            p.entity.invisible = !arms;
            out.push(p);
        }
        out
    }

    /// Posing the shared stand model depends only on the parameters, not on the pose before it,
    /// so a skipped pose (an invisible, empty stand) cannot change how a later stand is drawn.
    #[test]
    fn a_skipped_armor_stand_pose_leaves_no_trace() {
        let params = stand_params();
        for (i, before) in params.iter().enumerate() {
            for (j, after) in params.iter().enumerate() {
                let mut with_skipped = RenderManager::new();
                with_skipped.pose_with(ModelKind::ArmorStand, before);
                with_skipped.pose_with(ModelKind::ArmorStand, after);
                let mut fresh = RenderManager::new();
                fresh.pose_with(ModelKind::ArmorStand, after);
                assert_eq!(
                    state(&mut with_skipped),
                    state(&mut fresh),
                    "pose {j} after pose {i} differs from pose {j} alone"
                );
                // The same for the emitted geometry, which is what a later stand draws.
                let base = mc_render::entity::Mat4::IDENTITY;
                let (mut a, mut b) = (Vec::new(), Vec::new());
                with_skipped.emit_with(ModelKind::ArmorStand, base, after, &mut a);
                fresh.emit_with(ModelKind::ArmorStand, base, after, &mut b);
                assert_eq!(a, b, "geometry of pose {j} after pose {i}");
            }
        }
    }

    /// Negative control: the comparison above can see a pose difference at all.
    #[test]
    fn different_stand_poses_are_distinguishable() {
        let params = stand_params();
        let mut a = RenderManager::new();
        a.pose_with(ModelKind::ArmorStand, &params[1]);
        let mut b = RenderManager::new();
        b.pose_with(ModelKind::ArmorStand, &params[2]);
        assert_ne!(state(&mut a), state(&mut b));
    }
}
