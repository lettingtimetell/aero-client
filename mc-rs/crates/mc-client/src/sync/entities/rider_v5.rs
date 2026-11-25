//! The 1.7.10 seat of the session-owned rider (and, through [`rider_seat_v5`], of a remote one).
//!
//! java17: `Entity.updateRiderPosition` (`entity/Entity.java:1533-1539`) places the rider with
//! `setPosition(posX, posY + getMountedYOffset() + rider.getYOffset(), posZ)`. Three things make
//! the 1.7.10 seat differ from [`EntityTracker::local_rider_position`]'s 1.8.9 one:
//!
//! * the vehicle's `posY` is its box bottom plus its `yOffset` (half the height for boats and
//!   minecarts), [`mc_world::v5::geometry_v5`];
//! * `getMountedYOffset()` is `height * 0.0D - 0.30000001192092896D` for boats and minecarts
//!   (`entity/item/EntityBoat.java:82-85`, `entity/item/EntityMinecart.java:118-121`), otherwise
//!   `height * 0.75D`;
//! * the rider's `getYOffset()` is `yOffset - 0.5F` (`entity/player/EntityPlayer.java:1047-1050`)
//!   and the position is its **eye**, so the caller places it with the 1.7.10 `setPosition`.

use mc_data::EntityType;
use mc_math::{cos, sin, DVec3};
use mc_world::entity::EntityKind;
use mc_world::World;

use super::{current_position, current_rotation, EntityTracker, TrackedEntity};

impl EntityTracker {
    /// The `(posX, posY, posZ)` a 1.7.10 vehicle gives its local rider, `posY` being the
    /// rider's eye; `rider_y_offset` is the rider's `getYOffset()`. `None` when the vehicle is
    /// not tracked or is a kind 1.7.10 does not have.
    pub(crate) fn local_rider_eye_v5(
        &self,
        world: &World,
        network_id: i32,
        rider_y_offset: f64,
    ) -> Option<DVec3> {
        let tracked = self.by_network_id.get(&network_id)?;
        rider_seat_v5(
            world,
            tracked,
            current_position(world, tracked),
            rider_y_offset,
        )
    }
}

/// The `(posX, posY, posZ)` the 1.7.10 vehicle `tracked`, its box bottom at `position`, gives a
/// rider whose `getYOffset()` is `rider_y_offset`. `None` for a kind 1.7.10 does not have. The
/// local rider's seat ([`EntityTracker::local_rider_eye_v5`]) and a remote rider's
/// (`tracker_v5::mounted_position_v5`, which passes the vehicle's place in the current rider
/// pass) are both this.
pub(super) fn rider_seat_v5(
    world: &World,
    tracked: &TrackedEntity,
    position: DVec3,
    rider_y_offset: f64,
) -> Option<DVec3> {
    let vehicle_type = tracked.kind.entity_type()?;
    let geometry = mc_world::v5::geometry_v5(EntityKind::Type(vehicle_type))?;
    let pos_y = position.y + f64::from(geometry.y_offset);
    let mounted = match vehicle_type {
        EntityType::Boat
        | EntityType::MinecartCommandBlock
        | EntityType::MinecartRideable
        | EntityType::MinecartChest
        | EntityType::MinecartFurnace
        | EntityType::MinecartTNT
        | EntityType::MinecartHopper
        // `(double)height * 0.0D - 0.30000001192092896D`: the product is 0 for any height.
        | EntityType::MinecartSpawner => -0.300_000_011_920_928_96,
        _ => f64::from(tracked.height) * 0.75,
    };
    let mut seat = DVec3::new(position.x, pos_y + mounted + rider_y_offset, position.z);
    match vehicle_type {
        EntityType::Boat => {
            // java17: `EntityBoat.updateRiderPosition` (`entity/item/EntityBoat.java:436-444`).
            let (yaw, _) = current_rotation(world, tracked);
            let radians = f64::from(yaw) * core::f64::consts::PI / 180.0;
            seat.x += radians.cos() * 0.4;
            seat.z += radians.sin() * 0.4;
        }
        EntityType::Chicken => {
            // java17: `EntityChicken.updateRiderPosition`
            // (`entity/passive/EntityChicken.java:179-192`).
            let yaw = tracked.render.body_yaw * core::f32::consts::PI / 180.0;
            seat = DVec3::new(
                position.x + f64::from(0.1f32 * sin(yaw)),
                pos_y + f64::from(tracked.height * 0.5) + rider_y_offset + 0.0,
                position.z - f64::from(0.1f32 * cos(yaw)),
            );
        }
        EntityType::EntityHorse => {
            // java17: `EntityHorse.updateRiderPosition`
            // (`entity/passive/EntityHorse.java:1613-1630`): while rearing, the seat moves
            // by the same float offsets as in 1.8.9, added after the rider's y offset.
            let [x, y, z] = tracked.horse_anim.rider_offset(tracked.render.body_yaw);
            if [x, y, z] != [0.0; 3] {
                seat = DVec3::new(
                    position.x + f64::from(x),
                    pos_y + mounted + rider_y_offset + f64::from(y),
                    position.z + f64::from(z),
                );
            }
        }
        _ => {}
    }
    Some(seat)
}
