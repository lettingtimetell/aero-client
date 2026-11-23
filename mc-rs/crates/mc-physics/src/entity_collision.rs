//! The pairwise impulse from `Entity.applyEntityCollision`.
//!
//! This is deliberately a pure helper. Vanilla only calls it from the server-side nearby-entity
//! push path for ordinary living entities, while mc-client currently interpolates remote
//! entities. Keeping its inputs and returned velocity delta free of ECS state makes the exact
//! arithmetic available when that tick path is added without inventing a client-side push today.

use mc_math::{abs_max, sqrt_d, DVec3};

/// The entity fields `applyEntityCollision` reads from one participant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EntityCollisionBody {
    /// `posX`, `posY`, and `posZ`. Only X/Z participate in this impulse.
    pub pos: DVec3,
    /// java: `Entity.entityCollisionReduction`.
    pub collision_reduction: f32,
    /// java: `Entity.noClip`.
    pub no_clip: bool,
    /// Whether `riddenByEntity` is non-null.  A mount with a passenger receives no push from
    /// this call; the other participant is still evaluated independently.
    pub ridden_by_entity: bool,
}

impl EntityCollisionBody {
    /// Creates an ordinary collidable entity with no collision-reduction attribute.
    #[must_use]
    pub const fn new(pos: DVec3) -> Self {
        Self {
            pos,
            collision_reduction: 0.0,
            no_clip: false,
            ridden_by_entity: false,
        }
    }
}

/// The two ownership relationships that suppress a pairwise collision.
///
/// These fields are expressed from `other`'s perspective, matching Java's
/// `entityIn.ridingEntity != this && entityIn.riddenByEntity != this` condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EntityCollisionRelation {
    /// `other.ridingEntity == this`.
    pub other_rides_self: bool,
    /// `other.riddenByEntity == this`.
    pub other_is_ridden_by_self: bool,
}

/// The velocity effects of one `this.applyEntityCollision(other)` call.
///
/// A consumer adds these deltas to its participant velocities.  `Entity.addVelocity` in 1.8.9
/// changes motion only; it does **not** set `velocityChanged`, so the two flags are retained as
/// explicit false outputs rather than inventing a network-side consequence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EntityCollisionDelta {
    /// Value to add to `this.motionX/Y/Z`.
    pub self_velocity: DVec3,
    /// Value to add to `other.motionX/Y/Z`.
    pub other_velocity: DVec3,
    /// Whether Java's `this.velocityChanged` becomes true.
    pub self_velocity_changed: bool,
    /// Whether Java's `other.velocityChanged` becomes true.
    pub other_velocity_changed: bool,
}

impl EntityCollisionDelta {
    /// No impulse and no `velocityChanged` writes.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            self_velocity: DVec3::ZERO,
            other_velocity: DVec3::ZERO,
            self_velocity_changed: false,
            other_velocity_changed: false,
        }
    }
}

/// Computes the exact pairwise velocity delta from `Entity.applyEntityCollision`.
///
/// java: `Entity.applyEntityCollision` (`Entity.java:1378`). The `sqrt_double` result narrows to
/// `f32` before it is widened back into the double locals, because that is what
/// `MathHelper.sqrt_double` does. The impulse has no yaw or Y component. Notice that the second
/// reduction applies once to the shared impulse; Java then independently suppresses each output
/// when that participant has a passenger.
#[must_use]
pub fn entity_collision_delta(
    this: EntityCollisionBody,
    other: EntityCollisionBody,
    relation: EntityCollisionRelation,
) -> EntityCollisionDelta {
    if this.no_clip
        || other.no_clip
        || relation.other_rides_self
        || relation.other_is_ridden_by_self
    {
        return EntityCollisionDelta::none();
    }

    let mut dx = other.pos.x - this.pos.x;
    let mut dz = other.pos.z - this.pos.z;
    let distance = abs_max(dx, dz);
    if distance < 0.009_999_999_776_482_582 {
        return EntityCollisionDelta::none();
    }

    // java: `d2 = MathHelper.sqrt_double(d2)`, whose return type is float.
    let distance = f64::from(sqrt_d(distance));
    dx /= distance;
    dz /= distance;
    let mut inverse_distance = 1.0 / distance;
    if inverse_distance > 1.0 {
        inverse_distance = 1.0;
    }
    dx *= inverse_distance;
    dz *= inverse_distance;
    dx *= 0.050_000_000_745_058_06;
    dz *= 0.050_000_000_745_058_06;

    let this_reduction = f64::from(1.0f32 - this.collision_reduction);
    dx *= this_reduction;
    dz *= this_reduction;
    let self_velocity = DVec3::new(-dx, 0.0, -dz);

    EntityCollisionDelta {
        self_velocity: if this.ridden_by_entity {
            DVec3::ZERO
        } else {
            self_velocity
        },
        other_velocity: if other.ridden_by_entity {
            DVec3::ZERO
        } else {
            DVec3::new(dx, 0.0, dz)
        },
        self_velocity_changed: false,
        other_velocity_changed: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(x: f64, y: f64, z: f64) -> EntityCollisionBody {
        EntityCollisionBody::new(DVec3::new(x, y, z))
    }

    #[test]
    fn symmetric_pair_is_pushed_apart_on_x_without_a_y_or_network_flag() {
        let delta = entity_collision_delta(
            body(0.0, 4.0, 0.0),
            body(1.0, -200.0, 0.0),
            EntityCollisionRelation::default(),
        );
        let push = 0.050_000_000_745_058_06;
        assert_eq!(delta.self_velocity, DVec3::new(-push, 0.0, 0.0));
        assert_eq!(delta.other_velocity, DVec3::new(push, 0.0, 0.0));
        assert!(!delta.self_velocity_changed && !delta.other_velocity_changed);
    }

    #[test]
    fn this_reduction_applies_once_to_the_shared_impulse() {
        let mut this = body(0.0, 0.0, 0.0);
        this.collision_reduction = 0.25;
        let mut other = body(1.0, 0.0, 0.0);
        other.collision_reduction = 0.5; // Java does not read this value in this call.
        let delta = entity_collision_delta(this, other, EntityCollisionRelation::default());
        let push = 0.050_000_000_745_058_06;
        assert_eq!(delta.self_velocity.x, -push * 0.75);
        assert_eq!(delta.other_velocity.x, push * 0.75);
        assert_eq!(delta.self_velocity.y, 0.0);
        assert_eq!(delta.other_velocity.z, 0.0);
    }

    #[test]
    fn no_clip_mount_and_passenger_pairs_are_excluded() {
        let this = body(0.0, 0.0, 0.0);
        let other = body(1.0, 0.0, 0.0);
        for (this, other, relation) in [
            (
                EntityCollisionBody {
                    no_clip: true,
                    ..this
                },
                other,
                EntityCollisionRelation::default(),
            ),
            (
                this,
                EntityCollisionBody {
                    no_clip: true,
                    ..other
                },
                EntityCollisionRelation::default(),
            ),
            (
                this,
                other,
                EntityCollisionRelation {
                    other_rides_self: true,
                    ..EntityCollisionRelation::default()
                },
            ),
            (
                this,
                other,
                EntityCollisionRelation {
                    other_is_ridden_by_self: true,
                    ..EntityCollisionRelation::default()
                },
            ),
        ] {
            assert_eq!(
                entity_collision_delta(this, other, relation),
                EntityCollisionDelta::none()
            );
        }
    }

    #[test]
    fn a_passenger_suppresses_only_its_own_participants_push() {
        let this = body(0.0, 0.0, 0.0);
        let other = body(1.0, 0.0, 0.0);
        let this_has_passenger = entity_collision_delta(
            EntityCollisionBody {
                ridden_by_entity: true,
                ..this
            },
            other,
            EntityCollisionRelation::default(),
        );
        assert_eq!(this_has_passenger.self_velocity, DVec3::ZERO);
        assert_eq!(
            this_has_passenger.other_velocity.x,
            0.050_000_000_745_058_06
        );

        let other_has_passenger = entity_collision_delta(
            this,
            EntityCollisionBody {
                ridden_by_entity: true,
                ..other
            },
            EntityCollisionRelation::default(),
        );
        assert_eq!(
            other_has_passenger.self_velocity.x,
            -0.050_000_000_745_058_06
        );
        assert_eq!(other_has_passenger.other_velocity, DVec3::ZERO);
    }

    #[test]
    fn abs_max_threshold_suppresses_near_zero_pairs_and_retains_the_boundary() {
        let this = body(0.0, 0.0, 0.0);
        assert_eq!(
            entity_collision_delta(
                this,
                body(0.009, 0.0, 0.0),
                EntityCollisionRelation::default()
            ),
            EntityCollisionDelta::none()
        );
        assert_ne!(
            entity_collision_delta(
                this,
                body(0.01, 0.0, 0.0),
                EntityCollisionRelation::default()
            )
            .self_velocity,
            DVec3::ZERO,
            "the inclusive threshold still produces motion (but never a velocityChanged write)"
        );
    }
}
