//! The tracker's packet seams for a 1.7.10 connection.
//!
//! A 1.7.10 entity's `posY` is its box bottom plus `yOffset` (java17 `Entity.setPosition`,
//! `entity/Entity.java:249-257`), and that `posY` is what a 1.7.10 server encodes: the box centre
//! for boats, minecarts, dropped items, XP orbs, primed TNT, falling blocks and ender crystals
//! (`yOffset = height / 2`), the box bottom for everything else. The tracker keeps the client's
//! box-bottom model, so every 1.7.10 position is converted here, as the 1.7.10 handlers and
//! setters compute it:
//!
//! * `serverPosX/Y/Z` keep the wire value (relative moves add to it, as in 1.7.10);
//! * a relative move targets `serverPos / 32`, a teleport `serverPos / 32 + 1/64` on Y with no
//!   dead zone (java17 `NetHandlerPlayClient.handleEntityMovement` / `handleEntityTeleport`,
//!   `client/network/NetHandlerPlayClient.java:482-524`), and the box bottom is that `posY`
//!   minus the entity's `yOffset`, which is what `setPosition` makes of it;
//! * neither packet carries an on-ground flag, so the stored one is left alone;
//! * the setter runs without 1.8.9's teleport flag: 1.7.10's `setPositionAndRotation2` has none,
//!   so a ridden boat eases a teleport like any other move (java17
//!   `entity/item/EntityBoat.java:136-165`, java18's non-teleport branch).
//!
//! A spawn's position is the constructor's, which the dispatcher computes and installs with
//! [`EntityTracker::place_spawn_v5`]. Living entities, players and every other kind have
//! `yOffset` 0, so for them nothing differs from the wire.

use mc_data::EntityType;
use mc_math::{Aabb, BlockPos, DVec3};
use mc_protocol::MetadataEntry;
use mc_world::entity::{EntityKind, Metadata, Position, Rotation};
use mc_world::World;

use super::{
    current_position, current_rotation, merge_registered_metadata, painting_target_position,
    set_target, EntityTracker, TrackedEntity, TrackedKind, Watched, FIXED_POINT_SCALE,
};

/// What a 1.7.10 connection keeps for an entity beyond the 1.8.9 record.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TrackedV5 {
    /// The spawn constructor's width and height ([`mc_world::v5::geometry_v5`]).
    base: (f32, f32),
    /// The last growing age the server sent (an `EntityAgeable`'s watched index 12).
    age: Option<i32>,
    /// A hanging entity's position as java17 `EntityHanging.setDirection` puts it.
    hanging: Option<DVec3>,
}

/// The position java17 `EntityHanging.setDirection` (`entity/EntityHanging.java:39-107`) gives a
/// hanging entity on the wall block `anchor` (1.7.10 hangs from the block it was placed against;
/// 1.8.9 from the block in front of it) facing `direction` (0 south, 1 west, 2 north, 3 east),
/// `width` and `height` in pixels: the block centre, 0.5625 out of the wall, and half a block
/// along the wall and up for a 32 or 64 pixel side, in `float` as Java computes it.
#[must_use]
pub(crate) fn hanging_position_v5(
    anchor: BlockPos,
    direction: i32,
    width: i32,
    height: i32,
) -> DVec3 {
    const OUT_OF_THE_WALL: f32 = 0.5625;
    // `func_70517_b`: 0.5 for 32 and 64 pixels, else 0.
    let centring = |pixels: i32| {
        if pixels == 32 || pixels == 64 {
            0.5_f32
        } else {
            0.0
        }
    };
    let mut x = anchor.x as f32 + 0.5;
    let mut y = anchor.y as f32 + 0.5;
    let mut z = anchor.z as f32 + 0.5;
    match direction {
        0 => {
            z += OUT_OF_THE_WALL;
            x += centring(width);
        }
        1 => {
            x -= OUT_OF_THE_WALL;
            z += centring(width);
        }
        2 => {
            z -= OUT_OF_THE_WALL;
            x -= centring(width);
        }
        3 => {
            x += OUT_OF_THE_WALL;
            z -= centring(width);
        }
        _ => {}
    }
    y += centring(height);
    DVec3::new(f64::from(x), f64::from(y), f64::from(z))
}

/// A 1.7.10 entity's size for this client tick, where its 1.7.10 class sets it every tick on a
/// client: a slime from its size (java17 `EntitySlime.onUpdate:124-128`, `0.6F * size`), an
/// ageable from its growing age (`EntityAgeable.onLivingUpdate:114-117`,
/// [`mc_world::v5::ageable_scale_v5`]); everything else keeps its constructor size. Nothing for a
/// 1.8.9 connection's entities.
pub(crate) fn refresh_living_dimensions_v5(world: &World, tracked: &mut TrackedEntity) {
    let Some(v5) = tracked.v5.as_deref().copied() else {
        return;
    };
    let Some(entity_type) = tracked.kind.entity_type() else {
        return;
    };
    let (width, height) = match entity_type {
        EntityType::Slime | EntityType::LavaSlime => {
            let Some(size) = tracked.with_watched(world, |watched| watched.slime_size()) else {
                return;
            };
            mc_world::v5::slime_dimensions_v5(size)
        }
        EntityType::Skeleton => return,
        _ => {
            let scale = mc_world::v5::ageable_scale_v5(entity_type, v5.age.unwrap_or(0));
            (v5.base.0 * scale, v5.base.1 * scale)
        }
    };
    tracked.width = width;
    tracked.height = height;
}

/// java17 `EntitySkeleton.onLivingUpdate` (`entity/monster/EntitySkeleton.java:156-159`): a
/// remote wither skeleton (type byte 1) takes `0.72F x 2.34F` every tick (1.8.9: 2.535 high); a
/// normal one keeps its box. Nothing for a 1.8.9 connection's entities.
pub(crate) fn refresh_skeleton_dimensions_v5(world: &World, tracked: &mut TrackedEntity) {
    if tracked.v5.is_none() || tracked.kind.entity_type() != Some(EntityType::Skeleton) {
        return;
    }
    if tracked.with_watched(world, |watched| watched.skeleton_type()) == Some(1) {
        (tracked.width, tracked.height) = mc_world::v5::WITHER_SKELETON_DIMENSIONS_V5;
    }
}

/// The position a 1.7.10 hanging entity keeps in place of the 1.8.9 block-anchored one.
#[must_use]
pub(crate) fn hanging_v5(tracked: &TrackedEntity) -> Option<DVec3> {
    tracked.v5.as_ref().and_then(|v5| v5.hanging)
}

/// A move or teleport of a 1.7.10 hanging entity. A vanilla 1.7.10 server sends one rarely: it
/// tracks hanging entities with an update interval of `Integer.MAX_VALUE` but still sends a
/// teleport once 400 watcher changes have passed (java17 `EntityTracker.java:159-161`,
/// `EntityTrackerEntry.java:158-198`). No hanging class overrides the setter in 1.7.10, so it is
/// `Entity.setPositionAndRotation2` (`entity/Entity.java:1589-1611`):
/// `setPosition` with `EntityHanging`'s 0.5 x 0.5 box, `setRotation`, then up out of any block
/// the box (narrowed by 1/32 on X and Z) overlaps. java18's painting and item-frame setters (a
/// block-anchored pose) do not apply.
fn move_hanging_v5(
    world: &mut World,
    tracked: &mut TrackedEntity,
    target: DVec3,
    yaw: f32,
    pitch: f32,
) {
    const HALF_WIDTH: f32 = 0.5 / 2.0;
    const HEIGHT: f32 = 0.5;
    let half = f64::from(HALF_WIDTH);
    let bounds = Aabb::from_bounds(
        target.x - half,
        target.y,
        target.z - half,
        target.x + half,
        target.y + f64::from(HEIGHT),
        target.z + half,
    );
    let mut collisions = Vec::new();
    mc_physics::collision_boxes_in(
        world,
        &bounds.contract(0.031_25, 0.0, 0.031_25),
        &mut collisions,
    );
    let mut position = target;
    if !collisions.is_empty() {
        let top = collisions.iter().fold(0.0, |top, collision| {
            if collision.max.y > top {
                collision.max.y
            } else {
                top
            }
        });
        position.y += top - bounds.min.y;
    }
    if let Some(v5) = tracked.v5.as_mut() {
        v5.hanging = Some(position);
    }
    let (yaw, pitch) = (yaw % 360.0, pitch % 360.0);
    if let Ok(mut current) = world.ecs.get::<&mut Position>(tracked.entity) {
        current.0 = position;
    }
    if let Ok(mut rotation) = world.ecs.get::<&mut Rotation>(tracked.entity) {
        rotation.yaw = yaw;
        rotation.pitch = pitch;
    }
    tracked.target_pos = position;
    tracked.target_yaw = yaw;
    tracked.target_pitch = pitch;
    tracked.interpolation_steps = 0;
}

/// Where a 1.7.10 vehicle at `vehicle_position` (its box bottom in this rider pass) seats a
/// remote rider: the rider's box bottom. `None` for a 1.8.9 connection's vehicle, and for a
/// vehicle 1.7.10 has no seat for (a player), where the 1.8.9 seat applies.
///
/// java17 `Entity.updateRiderPosition` (`entity/Entity.java:1533-1539`):
/// `rider.setPosition(posX, posY + getMountedYOffset() + rider.getYOffset(), posZ)`. The vehicle
/// side (its `posY`, `getMountedYOffset`, the boat's, chicken's and horse's own seats) is the
/// local rider's ([`super::rider_v5::rider_seat_v5`]); the rider side is `getYOffset()`:
/// `yOffset - 0.5F` for a player (`entity/player/EntityPlayer.java:1047-1050`), `yOffset - 0.5D`
/// for a skeleton (`entity/monster/EntitySkeleton.java:368-371`), else `yOffset` (no other
/// 1.7.10 class overrides it; java18's zombie and silverfish offsets do not apply). The
/// rider's `setPosition` then takes its `yOffset` off again.
pub(crate) fn mounted_position_v5(
    world: &World,
    rider: &TrackedEntity,
    vehicle: &TrackedEntity,
    vehicle_position: DVec3,
) -> Option<DVec3> {
    vehicle.v5.as_ref()?;
    let rider_y_offset = y_offset_v5(rider.kind);
    let rider_get_y_offset = match rider.kind {
        TrackedKind::Player => f64::from(rider_y_offset as f32 - 0.5_f32),
        TrackedKind::Type(EntityType::Skeleton) => rider_y_offset - 0.5,
        _ => rider_y_offset,
    };
    let seat =
        super::rider_v5::rider_seat_v5(world, vehicle, vehicle_position, rider_get_y_offset)?;
    Some(DVec3::new(
        seat.x,
        box_bottom(seat.y, rider_y_offset),
        seat.z,
    ))
}

/// The `mc-world` kind of a tracked kind.
#[must_use]
pub(crate) fn entity_kind_v5(kind: TrackedKind) -> EntityKind {
    match kind {
        TrackedKind::Type(t) => EntityKind::Type(t),
        TrackedKind::Player => EntityKind::Player,
        TrackedKind::FishingBobber => EntityKind::FishingBobber,
    }
}

/// `Entity.yOffset` as a 1.7.10 client's spawn constructor leaves it, widened as Java widens it
/// (`(double)this.yOffset`). 0 for every kind but the seven half-height ones, and for kinds
/// 1.7.10 does not have.
#[must_use]
pub(crate) fn y_offset_v5(kind: TrackedKind) -> f64 {
    mc_world::v5::geometry_v5(entity_kind_v5(kind)).map_or(0.0, |g| f64::from(g.y_offset))
}

/// The box bottom a 1.7.10 move gives an entity whose `posY` target is `pos_y`: `setPosition`'s
/// `y - (double)yOffset + (double)ySize`, with `ySize` 0 for every remote entity.
fn box_bottom(pos_y: f64, y_offset: f64) -> f64 {
    pos_y - y_offset
}

/// The box bottom of an object right after its 1.7.10 spawn constructor, for the `posY`
/// argument `y` the handler passes (`serverPosY / 32`, or the raw value for an XP orb).
///
/// java17: `EntityBoat(World, x, y, z)` calls `setPosition(x, y + yOffset, z)`
/// (`entity/item/EntityBoat.java:70-73`), so its box starts at `y` (up to rounding); every other
/// half-height constructor calls `setPosition(x, y, z)` (`EntityMinecart.java:106-109`,
/// `EntityTNTPrimed.java:22-25`, `EntityEnderCrystal.java:27-30`, `EntityItem.java:31-38`,
/// `EntityFallingBlock.java:46-57`, `EntityXPOrb.java:22-27`), so its box starts `yOffset` below.
#[must_use]
pub(crate) fn spawn_box_bottom_v5(kind: EntityType, y: f64) -> f64 {
    let y_offset = y_offset_v5(TrackedKind::Type(kind));
    if kind == EntityType::Boat {
        box_bottom(y + y_offset, y_offset)
    } else {
        box_bottom(y, y_offset)
    }
}

impl EntityTracker {
    /// Make a freshly spawned generation a 1.7.10 one: its constructor size from
    /// [`mc_world::v5::geometry_v5`], its growing age from the spawn's metadata (`None`: the
    /// watcher's default 0), and the size its first client tick gives it.
    pub(crate) fn init_v5(&mut self, world: &World, network_id: i32, age: Option<i32>) -> bool {
        let Some(tracked) = self.by_network_id.get_mut(&network_id) else {
            return false;
        };
        let base = mc_world::v5::geometry_v5(entity_kind_v5(tracked.kind))
            .map_or((tracked.base_width, tracked.base_height), |geometry| {
                (geometry.width, geometry.height)
            });
        tracked.v5 = Some(Box::new(TrackedV5 {
            base,
            age,
            hanging: None,
        }));
        (tracked.base_width, tracked.base_height) = base;
        (tracked.width, tracked.height) = base;
        refresh_living_dimensions_v5(world, tracked);
        refresh_skeleton_dimensions_v5(world, tracked);
        true
    }

    /// Keep the growing age a 1.7.10 `S1C` carried; the client tick sizes the entity from it.
    pub(crate) fn set_age_v5(&mut self, network_id: i32, age: i32) {
        if let Some(v5) = self
            .by_network_id
            .get_mut(&network_id)
            .and_then(|tracked| tracked.v5.as_mut())
        {
            v5.age = Some(age);
        }
    }

    /// Put a 1.7.10 hanging entity where `EntityHanging.setDirection` puts it
    /// ([`hanging_position_v5`]); the tracker then keeps it there instead of deriving the 1.8.9
    /// position from its block. The frame's column is where it now is (0.5625 out of the wall, so
    /// possibly the next column), as for any other spawn ([`Self::rehome_v5`]).
    pub(crate) fn place_hanging_v5(
        &mut self,
        world: &mut World,
        network_id: i32,
        position: DVec3,
    ) -> bool {
        let Some(tracked) = self.by_network_id.get_mut(&network_id) else {
            return false;
        };
        let Some(v5) = tracked.v5.as_mut() else {
            return false;
        };
        v5.hanging = Some(position);
        if let Ok(mut current) = world.ecs.get::<&mut Position>(tracked.entity) {
            current.0 = position;
        }
        tracked.target_pos = position;
        tracked.last_tick_pos = position;
        self.rehome_v5(world, network_id, position);
        true
    }

    /// File a generation under the column of `position`, its new place: resident if that column
    /// is loaded, else queued for it, the rule the tracker's spawn queue applies to every spawn
    /// (see [`Self::readmit_if_loaded_v5`] for how 1.7.10 itself differs).
    fn rehome_v5(&mut self, world: &World, network_id: i32, position: DVec3) {
        let chunk = BlockPos::from_vec3(position).chunk_pos();
        let Some(tracked) = self.by_network_id.get_mut(&network_id) else {
            return;
        };
        if tracked.resident {
            tracked.resident_chunk = chunk;
            if !world.is_chunk_loaded(chunk) {
                self.park_entity(network_id, chunk);
            }
        } else if world.is_chunk_loaded(chunk) {
            self.unqueue_parked(network_id);
            if let Some(tracked) = self.by_network_id.get_mut(&network_id) {
                tracked.resident = true;
                tracked.resident_chunk = chunk;
                self.iteration_order.push(network_id);
            }
        } else {
            self.unqueue_parked(network_id);
            if let Some(tracked) = self.by_network_id.get_mut(&network_id) {
                tracked.resident_chunk = chunk;
            }
            self.pending_by_chunk
                .entry(chunk)
                .or_default()
                .push(network_id);
        }
    }

    /// java17 `NHPC.handleEntityMetadata` on a tracked entity: [`Self::apply_metadata`] entry by
    /// entry, except that a slime's or magma cube's size byte (16) has no watcher callback in
    /// 1.7.10 (`DataWatcher.updateWatchedObjectsFromList` calls `func_145781_i`, which only
    /// `EntityHanging` and `EntityMinecartCommandBlock` override; java17
    /// `entity/DataWatcher.java:322-341`): the byte is stored without 1.8.9's resize, yaw snap
    /// and splash (java18 `EntitySlime.onDataWatcherUpdate`), and the client tick sizes the slime
    /// from it ([`refresh_living_dimensions_v5`]).
    pub(crate) fn apply_metadata_v5(
        &mut self,
        world: &mut World,
        network_id: i32,
        entries: &[MetadataEntry],
    ) -> bool {
        let Some(tracked) = self.by_network_id.get(&network_id) else {
            return false;
        };
        let slime = matches!(
            tracked.kind,
            TrackedKind::Type(EntityType::Slime | EntityType::LavaSlime)
        );
        for entry in entries {
            if !slime || entry.index != 16 {
                if !self.apply_metadata_entry(world, network_id, std::slice::from_ref(entry)) {
                    return false;
                }
                continue;
            }
            let Some(tracked) = self.by_network_id.get_mut(&network_id) else {
                return false;
            };
            let Ok(mut metadata) = world.ecs.get::<&mut Metadata>(tracked.entity) else {
                return false;
            };
            merge_registered_metadata(tracked.kind, &mut metadata, std::slice::from_ref(entry));
            tracked.status_sound_slime_size = Watched::new(tracked.kind, &metadata).slime_size();
        }
        true
    }

    /// `S14`-`S17` for a remote entity: java17 `handleEntityMovement` (`:508-524`). `delta` is
    /// added to the integer server position; `look` is the packet's rotation when it carries one
    /// (else the entity's current one, as in 1.8.9).
    pub(crate) fn apply_relative_move_v5(
        &mut self,
        world: &mut World,
        network_id: i32,
        delta: [i8; 3],
        look: Option<(f32, f32)>,
    ) -> bool {
        let Some(tracked) = self.by_network_id.get_mut(&network_id) else {
            return false;
        };
        for (server, step) in tracked.server_pos.iter_mut().zip(delta) {
            *server = server.wrapping_add(i32::from(step));
        }
        if tracked.kind.entity_type() == Some(EntityType::ItemFrame) {
            tracked.item_frame_anchor = Some(BlockPos::from_vec3(tracked.server_position()));
        }
        let (yaw, pitch) = look.unwrap_or_else(|| current_rotation(world, tracked));
        let server = tracked.server_position();
        let target = DVec3::new(
            server.x,
            box_bottom(server.y, y_offset_v5(tracked.kind)),
            server.z,
        );
        if hanging_v5(tracked).is_some() {
            move_hanging_v5(world, tracked, target, yaw, pitch);
        } else {
            let target = painting_target_position(world, tracked, target);
            set_target(world, tracked, target, yaw, pitch, false);
        }
        self.readmit_if_loaded_v5(world, network_id);
        true
    }

    /// `S18` for a remote entity: java17 `handleEntityTeleport` (`:482-498`). The target is
    /// `y / 32 + 0.015625` whatever the current position; the yaw and pitch are the packet's.
    pub(crate) fn apply_teleport_v5(
        &mut self,
        world: &mut World,
        network_id: i32,
        server_pos: [i32; 3],
        yaw: f32,
        pitch: f32,
    ) -> bool {
        let Some(tracked) = self.by_network_id.get_mut(&network_id) else {
            return false;
        };
        tracked.server_pos = server_pos;
        if tracked.kind.entity_type() == Some(EntityType::ItemFrame) {
            tracked.item_frame_anchor = Some(BlockPos::from_vec3(tracked.server_position()));
        }
        let pos_y = f64::from(server_pos[1]) / FIXED_POINT_SCALE + 0.015_625;
        let target = DVec3::new(
            f64::from(server_pos[0]) / FIXED_POINT_SCALE,
            box_bottom(pos_y, y_offset_v5(tracked.kind)),
            f64::from(server_pos[2]) / FIXED_POINT_SCALE,
        );
        if hanging_v5(tracked).is_some() {
            move_hanging_v5(world, tracked, target, yaw, pitch);
        } else {
            let target = painting_target_position(world, tracked, target);
            set_target(world, tracked, target, yaw, pitch, false);
        }
        self.readmit_if_loaded_v5(world, network_id);
        true
    }

    /// Put a freshly spawned generation where its 1.7.10 constructor put it (`position` is the
    /// box bottom), keeping its wire `serverPos`. The spawn queue follows the new position
    /// ([`Self::rehome_v5`]).
    pub(crate) fn place_spawn_v5(
        &mut self,
        world: &mut World,
        network_id: i32,
        position: DVec3,
    ) -> bool {
        let Some(tracked) = self.by_network_id.get_mut(&network_id) else {
            return false;
        };
        if let Ok(mut current) = world.ecs.get::<&mut Position>(tracked.entity) {
            current.0 = position;
        }
        tracked.target_pos = position;
        tracked.last_tick_pos = position;
        if let Some(boat) = tracked.boat.as_mut() {
            boat.target_pos = position;
        }
        if let Some(minecart) = tracked.minecart.as_mut() {
            minecart.target_pos = position;
        }
        self.rehome_v5(world, network_id, position);
        true
    }

    /// The head yaw of a 1.7.10 `S0F`: java17 `handleSpawnMob` (`:737`) sets
    /// `rotationYawHead` only. `renderYawOffset` keeps its field default, 0 (java17
    /// `EntityLivingBase.<init>` does not set it), where 1.8.9 also turns the body to the head.
    pub(crate) fn apply_spawn_head_look_v5(&mut self, network_id: i32, head_yaw: f32) -> bool {
        let Some(tracked) = self.by_network_id.get_mut(&network_id) else {
            return false;
        };
        tracked.head_yaw = head_yaw;
        tracked.render.head_yaw = head_yaw;
        tracked.render.prev_head_yaw = head_yaw;
        tracked.render.body_yaw = 0.0;
        tracked.render.prev_body_yaw = 0.0;
        tracked.body.prev_render_yaw_head = head_yaw;
        true
    }

    /// A queued generation that a move put into a loaded column joins the world.
    ///
    /// 1.7.10 queues nothing: its client chunk provider reports every column present (java17
    /// `ChunkProviderClient.chunkExists`, `client/multiplayer/ChunkProviderClient.java:33-36`), so
    /// an entity spawned where the server sent no column, such as an XP orb at its raw
    /// coordinates, ticks out there until a move brings it back. This tracker keeps such a
    /// generation in its spawn queue, unticked, and lets it in here, once a move puts it in a
    /// column the player has: what the player sees is the same; the motion it would have gathered
    /// out there is not modelled.
    fn readmit_if_loaded_v5(&mut self, world: &World, network_id: i32) {
        let Some(tracked) = self.by_network_id.get(&network_id) else {
            return;
        };
        if tracked.resident {
            return;
        }
        let chunk = BlockPos::from_vec3(current_position(world, tracked)).chunk_pos();
        if world.is_chunk_loaded(chunk) {
            let _ = self.activate_chunk(world, chunk);
        }
    }
}
