//! The 1.7.10 local-player tick of [`Session`].
//!
//! java17: `EntityClientPlayerMP.onUpdate` (`client/entity/EntityClientPlayerMP.java:68-84`): the
//! player's update, then either `sendMotionUpdates` or, while riding, the C05 + C0C pair. The
//! physics is `mc_physics::v5`; this file is the session glue around it, shaped like the 1.8.9
//! [`Session::tick_player`] with 1.7.10's differences:
//!
//! * no world border, so no border damage and no border wall;
//! * no spectator mode and no render-view entity: the player always reports its movement;
//! * C04/C06 carry the box bottom and the eye stance (`OutboundState::walking_update_v5`);
//! * the horse jump charges on any horse, saddled or not (`EntityPlayerSP.isRidingHorse`,
//!   `client/entity/EntityPlayerSP.java:584-587`).

use mc_data::{BlockId, BlockState};
use mc_math::BlockPos;
use mc_physics::prediction::Predictor;
use mc_physics::state::MoveInput;
use mc_physics::v5::{PlayerV5, DEATH_Y_OFFSET, PLAYER_Y_OFFSET, SLEEP_Y_OFFSET};
use mc_protocol::play::{c2s, C2SPlay};
use mc_world::World;

use super::{flight_toggle_target, Session};
use crate::outbound;

/// java17: `Blocks.bed`.
const BED: BlockId = BlockId(26);

/// java17: the horizontal limit of `World.getBlock` and `World.getBlockMetadata`,
/// `-30000000 <= x, z < 30000000`.
const WORLD_LIMIT: i32 = 30_000_000;

impl Session {
    /// Makes the session's player the 1.7.10 one ([`Predictor::new_v5`]): step height 0.5 and
    /// the eye `posY`, `yOffset2` and movement report fields beside the state. A player the
    /// shared join or respawn path built as a 1.8.9 predictor is converted before its first
    /// 1.7.10 use; an existing 1.7.10 player is left alone.
    pub(crate) fn ensure_predictor_v5(&mut self) {
        if let Some(predictor) = self.predictor.as_mut() {
            if predictor.v5().is_none() {
                *predictor = Predictor::new_v5(predictor.state);
            }
        }
    }

    /// The 1.7.10 body of [`Session::tick_player`]. Returns `None`: the movement packet is a
    /// protocol-5 one, sent with [`Session::send_v5`].
    pub(super) fn tick_player_v5(&mut self, input: &MoveInput) -> Option<C2SPlay> {
        mc_core::profile_scope!("physics.tick_player_v5");
        self.ensure_predictor_v5();

        // A destroyed or replaced vehicle ends the ride before the fork, as in 1.8.9.
        if let Some(vehicle_id) = self.riding.vehicle {
            if self.resolve_vehicle(vehicle_id) != self.riding.vehicle_entity {
                self.clear_riding();
            }
        }
        if let Some(predictor) = self.predictor.as_mut() {
            predictor.state.yaw = input.yaw;
            predictor.state.pitch = input.pitch;
        }
        self.push_server_movement_state();
        if let Some(predictor) = self.predictor.as_mut() {
            // java17 has no spectator game mode (`WorldSettings.GameType`).
            predictor.state.spectator = false;
            predictor.state.spectator_mode = false;
        }
        if self.riding.vehicle.is_some() {
            return self.tick_riding_v5(input);
        }

        // `EntityPlayerSP.onLivingUpdate` sends C13 at each capability mutation
        // (`client/entity/EntityPlayerSP.java:213-225`, `:283-287`), before the walking packets.
        let (flying_before, flight_toggle_edge) = {
            let state = self.predictor.as_ref()?.state;
            (state.flying, flight_toggle_target(&state, input))
        };
        let watched_sprinting = self.local_watched_sprinting();
        let sleeping = self.state.local.sleeping;
        let entity_collisions = self.state.entities.local_sweep_collision_candidates(
            &self.state.world,
            self.state.local.entity_id,
            &self.state.players,
        );
        let (movement, sprinting, sneaking, flying, on_ground) = {
            let predictor = self.predictor.as_mut()?;
            // The local watched flag feeds the tick, as on 1.8.9 (an S1C echo is a flag write).
            predictor.state.sprinting = watched_sprinting;
            let movement = predictor.tick_v5(&self.state.world, input, &entity_collisions)?;
            let state = &predictor.state;
            (
                movement,
                state.sprinting,
                // java17: `EntityPlayerSP.isSneaking()` is the sneak key and not sleeping.
                state.sneaking && !sleeping,
                state.flying,
                state.on_ground,
            )
        };
        self.set_local_watched_sprinting(sprinting);
        let watched_sprinting = self.local_watched_sprinting();
        let entity_id = self.state.local.entity_id;

        let flying_after_toggle = flight_toggle_edge.unwrap_or(flying_before);
        if let Some(target) = flight_toggle_edge {
            self.send_flight_ability_edge(target);
        }
        if flying_after_toggle && on_ground && !flying {
            self.send_flight_ability_edge(false);
        }
        if flying != self.state.local.abilities.flying {
            self.send_flight_ability_edge(flying);
        }

        let mut packets = Vec::with_capacity(3);
        self.outbound.walking_update_v5(
            entity_id,
            movement,
            watched_sprinting,
            sneaking,
            &mut packets,
        );
        for packet in packets {
            self.send_v5(packet);
        }
        None
    }

    /// The riding branch of the 1.7.10 tick: `Entity.updateRidden` runs the player's update
    /// (the `EntityPlayerSP` prologue and its own travel, then the tails), and `onUpdate` sends
    /// C05 + C0C. The seat itself is written after the vehicle's tick, as on 1.8.9.
    fn tick_riding_v5(&mut self, input: &MoveInput) -> Option<C2SPlay> {
        let flight_toggle_edge = {
            let state = self.predictor.as_ref()?.state;
            flight_toggle_target(&state, input)
        };
        let watched_sprinting = self.local_watched_sprinting();
        let entity_collisions = self.state.entities.local_sweep_collision_candidates(
            &self.state.world,
            self.state.local.entity_id,
            &self.state.players,
        );
        let world = &self.state.world;
        let predictor = self.predictor.as_mut()?;
        predictor.state.sprinting = watched_sprinting;
        // java17: the horse block's `var1 = this.movementInput.jump`, last tick's jump key.
        let horse_jump_was_held = predictor.state.prev_jump;
        predictor.mounted_preflight_v5(world, input);
        let flying_before_travel = predictor.state.flying;
        predictor.mounted_travel_v5(world, &entity_collisions);
        let landing_edge = flying_before_travel && !predictor.state.flying;
        // java17: `EntityLivingBase.updateRidden` (`entity/EntityLivingBase.java:1807-1813`).
        predictor.state.fall_distance = 0.0;
        // java17: `EntityPlayer.updateRidden` (`entity/player/EntityPlayer.java:438-464`).
        predictor.state.prev_camera_yaw = predictor.state.camera_yaw;
        predictor.state.camera_yaw = 0.0;
        let state = predictor.state;
        self.set_local_watched_sprinting(state.sprinting);
        if let Some(target) = flight_toggle_edge {
            self.send_flight_ability_edge(target);
        }
        let entity_id = self.state.local.entity_id;
        // java17: `onLivingUpdate`'s horse block (`client/entity/EntityPlayerSP.java:241-277`)
        // sits between the flight toggle and `super.onLivingUpdate()`.
        if let Some(jump) = self.charge_horse_jump_v5(horse_jump_was_held, input.jump) {
            self.pending_horse_jump = self
                .riding
                .vehicle
                .map(|vehicle| (vehicle, jump, input.move_forward_riding(true), input.yaw));
            // java17: `sendHorseJump` sends `(int)(getHorseJumpPower() * 100.0F)`.
            self.send(C2SPlay::EntityAction(c2s::EntityAction::riding_jump(
                entity_id,
                (jump * 100.0) as i32,
            )));
        }
        if landing_edge {
            self.send_flight_ability_edge(false);
        }
        if state.flying != self.state.local.abilities.flying {
            self.send_flight_ability_edge(state.flying);
        }

        // java17: `EntityClientPlayerMP.onUpdate` (`:74-78`).
        let look = C2SPlay::PlayerLook(c2s::PlayerLook {
            yaw: state.yaw,
            pitch: state.pitch,
            on_ground: state.on_ground,
        });
        self.send(look.clone());
        self.send(outbound::steer_vehicle(
            state.move_strafing,
            state.move_forward,
            input.jump,
            input.sneak,
        ));
        Some(look)
    }

    /// One tick of `EntityPlayerSP.onLivingUpdate`'s horse jump charge
    /// (`client/entity/EntityPlayerSP.java:241-277`); the power to send on the release tick.
    ///
    /// The 1.8.9 state machine, with java17's `isRidingHorse()`: riding any `EntityHorse`, with
    /// no saddle test (`client/entity/EntityPlayerSP.java:584-587`).
    fn charge_horse_jump_v5(&mut self, was_jumping: bool, jump: bool) -> Option<f32> {
        if !self.is_riding_horse_v5() {
            self.riding.jump_power = 0.0;
            return None;
        }
        if self.riding.jump_power_counter < 0 {
            self.riding.jump_power_counter += 1;
            if self.riding.jump_power_counter == 0 {
                self.riding.jump_power = 0.0;
            }
        }
        if was_jumping && !jump {
            self.riding.jump_power_counter = -10;
            return Some(self.riding.jump_power);
        }
        if !was_jumping && jump {
            self.riding.jump_power_counter = 0;
            self.riding.jump_power = 0.0;
        } else if was_jumping {
            self.riding.jump_power_counter += 1;
            self.riding.jump_power = if self.riding.jump_power_counter < 10 {
                self.riding.jump_power_counter as f32 * 0.1
            } else {
                0.8 + 2.0 / (self.riding.jump_power_counter - 9) as f32 * 0.1
            };
        }
        None
    }

    /// After the shared 1.8.9 seat write of [`Session::tick_with_held`]: a 1.7.10 rider is
    /// re-seated where 1.7.10 puts it, its eye at `mount.posY + getMountedYOffset() +
    /// getYOffset()` (`entity/Entity.java:1533-1539`), which is 0.15 lower than 1.8.9's seat on a
    /// pig or horse, 0.1 lower in a minecart and 0.15 higher in a boat. A no-op unless riding
    /// with a 1.7.10 player.
    pub(super) fn sync_rider_eye_v5(&mut self) {
        let Some(vehicle_id) = self.riding.vehicle else {
            return;
        };
        let Some(y_offset) = self
            .predictor
            .as_ref()
            .and_then(|predictor| predictor.v5())
            .map(|v5| v5.y_offset())
        else {
            return;
        };
        // java17: `EntityPlayer.getYOffset()`, `(double)(this.yOffset - 0.5F)`.
        let rider_y_offset = f64::from(y_offset - 0.5);
        let Some(seat) =
            self.state
                .entities
                .local_rider_eye_v5(&self.state.world, vehicle_id, rider_y_offset)
        else {
            return;
        };
        if let Some(predictor) = self.predictor.as_mut() {
            predictor.set_position_v5(seat.x, seat.y, seat.z);
        }
    }

    /// java17: `EntityPlayerSP.isRidingHorse()`, `ridingEntity instanceof EntityHorse`.
    fn is_riding_horse_v5(&self) -> bool {
        let Some(vehicle) = self.riding.vehicle else {
            return false;
        };
        self.state
            .entities
            .with_watched(&self.state.world, vehicle, |watched| {
                watched.horse().is_some()
            })
            .unwrap_or(false)
    }

    /// The 1.7.10 end of `EntityPlayer.onDeath` (`entity/player/EntityPlayer.java:583-612`),
    /// after the shared resize and death kick: `setPosition(posX, posY, posZ)` rebuilds the
    /// 0.2 x 0.2 box from the eye (`posY - yOffset + yOffset2`), then `yOffset = 0.1F` (`:610`).
    /// Every later `posY`, the stance a dying player reports until it is removed at `deathTime`
    /// 20, is then `minY + 0.1F - yOffset2`. A no-op without a 1.7.10 player.
    pub(super) fn death_pose_v5(&mut self) {
        let Some(predictor) = self.predictor.as_mut() else {
            return;
        };
        let Some(eye) = predictor.v5().map(PlayerV5::pos_y) else {
            return;
        };
        let (x, z) = (predictor.state.pos.x, predictor.state.pos.z);
        predictor.set_position_v5(x, eye, z);
        predictor.set_y_offset_v5(DEATH_Y_OFFSET);
    }

    /// The 1.7.10 placement of `EntityPlayer.sleepInBedAt` (`entity/player/EntityPlayer.java:
    /// 1198-1284`), over the shared 1.8.9 one: `yOffset = 0.2F` (`:1238`), then
    /// `setPosition(x + dx, (float)y + 0.9375F, z + dz)` (`:1266`, `:1270`) with that y at eye
    /// level, so the box bottom is `y + 0.9375F - 0.2F + yOffset2`, 0.05 above 1.8.9's feet at
    /// `y + 0.6875F` (java18 `EntityPlayer.trySleep`).
    ///
    /// `dx` and `dz` follow the bed's direction (`meta & 3`) whenever `World.blockExists(x, y,
    /// z)` (`world/World.java:297-300`), which on a client is `0 <= y < 256` alone
    /// (`ChunkProviderClient.chunkExists` returns `true`): a bed in a column never sent reads
    /// metadata 0 and faces south, where 1.8.9 centres the player. Outside that height range the
    /// player is centred. A no-op without a 1.7.10 player.
    pub(super) fn enter_bed_v5(&mut self, bed: BlockPos) {
        let Some(predictor) = self.predictor.as_mut() else {
            return;
        };
        if predictor.v5().is_none() {
            return;
        }
        let (dx, dz) = if (0..256).contains(&bed.y) {
            match bed_direction_v5(&self.state.world, bed) {
                0 => (0.5, 0.9),
                1 => (0.1, 0.5),
                2 => (0.5, 0.1),
                _ => (0.9, 0.5),
            }
        } else {
            (0.5, 0.5)
        };
        predictor.set_y_offset_v5(SLEEP_Y_OFFSET);
        predictor.set_position_v5(
            f64::from(bed.x as f32 + dx),
            f64::from(bed.y as f32 + 0.9375),
            f64::from(bed.z as f32 + dz),
        );
    }

    /// The 1.7.10 half of `EntityPlayer.wakeUpPlayer` (`entity/player/EntityPlayer.java:
    /// 1310-1350`), after the shared resize and placement: `resetHeight()` restores `yOffset`
    /// 1.62F (`:849`), and while a bed is still at the sleeping spot the player is placed with
    /// `setPosition(ex + 0.5F, (float)ey + yOffset + 0.1F, ez + 0.5F)` (`:1327`), the eye of a
    /// player standing 0.1 above the exit cell, summed in float. Without a bed there the box
    /// stays where the resize left it and `posY` keeps its in-bed value until the next move
    /// derives one. A no-op without a 1.7.10 player.
    pub(super) fn wake_from_bed_v5(&mut self) {
        let Some(predictor) = self.predictor.as_mut() else {
            return;
        };
        if !predictor.set_y_offset_v5(PLAYER_Y_OFFSET) {
            return;
        }
        let Some(bed) = self.state.local.sleeping_at else {
            return;
        };
        let world = &self.state.world;
        if block_v5(world, bed).block() != BED {
            return;
        }
        let exit = safe_bed_exit_v5(world, bed).unwrap_or_else(|| bed.up());
        predictor.set_position_v5(
            f64::from(exit.x as f32 + 0.5),
            f64::from(exit.y as f32 + PLAYER_Y_OFFSET + 0.1),
            f64::from(exit.z as f32 + 0.5),
        );
    }

    /// The 1.7.10 half of `EntityPlayer.preparePlayerToSpawn` (`entity/player/EntityPlayer.java:
    /// 466-473`) for the player a later JoinGame keeps, after the shared one: `yOffset = 1.62F`,
    /// so a dead or sleeping player's offset goes back, then `Entity.preparePlayerToSpawn`
    /// (`entity/Entity.java:170-189`) places it from the eye with `setPosition(posX, posY,
    /// posZ)` while `posY > 0` (1.8.9 also requires `posY < 256`).
    ///
    /// Java retries one block higher while the new box collides in the player's old world. The
    /// client no longer has that world, so the first placement is the last, as on the 1.8.9
    /// path; for a living, awake player it is where the box already was. A dead or sleeping
    /// player's box drops by the offset change there, and vanilla's retry would lift it out of
    /// the old world's blocks. A no-op without a 1.7.10 player.
    pub(super) fn prepare_reused_player_to_spawn_v5(&mut self) {
        let Some(predictor) = self.predictor.as_mut() else {
            return;
        };
        if !predictor.set_y_offset_v5(PLAYER_Y_OFFSET) {
            return;
        }
        let eye = predictor.v5().map_or(0.0, PlayerV5::pos_y);
        if eye > 0.0 {
            let (x, z) = (predictor.state.pos.x, predictor.state.pos.z);
            predictor.set_position_v5(x, eye, z);
        }
    }
}

/// Whether `World.getBlock` and `World.getBlockMetadata` read the column at all.
fn inside_world_limit(pos: BlockPos) -> bool {
    (-WORLD_LIMIT..WORLD_LIMIT).contains(&pos.x) && (-WORLD_LIMIT..WORLD_LIMIT).contains(&pos.z)
}

/// java17: `World.getBlock` and `World.getBlockMetadata` (`world/World.java:266-290`,
/// `:404-428`): air with metadata 0 outside the horizontal limit or the height range, and in a
/// column never sent (its blank chunk), the stored state otherwise.
fn block_v5(world: &World, pos: BlockPos) -> BlockState {
    if inside_world_limit(pos) {
        world.block_at(pos)
    } else {
        BlockState::AIR
    }
}

/// java17: `BlockDirectional.getDirection(world.getBlockMetadata(x, y, z))`
/// (`block/BlockDirectional.java:14-17`), the low two metadata bits: 0 south, 1 west, 2 north,
/// 3 east. The stored state keeps them for every bed metadata.
fn bed_direction_v5(world: &World, bed: BlockPos) -> u8 {
    block_v5(world, bed).meta() & 3
}

/// java17: `BlockBed.getSafeExitLocation(world, x, y, z, 0)` (`block/BlockBed.java:235-265`):
/// 1.8.9's search order and offsets (`bedDirections`, `:22`), but a cell is free when its feet
/// and head blocks are not opaque (`Material.isOpaque`, so glass, ice, leaves, cactus and cobweb
/// do not block it) where 1.8.9 asks for not solid, standing on 1.7.10's solid top surface.
fn safe_bed_exit_v5(world: &World, bed: BlockPos) -> Option<BlockPos> {
    const BED_DIRECTIONS: [(i32, i32); 4] = [(0, 1), (-1, 0), (0, -1), (1, 0)];
    let (dx, dz) = BED_DIRECTIONS[usize::from(bed_direction_v5(world, bed))];
    for layer in 0..=1 {
        let min_x = bed.x - dx * layer - 1;
        let min_z = bed.z - dz * layer - 1;
        for x in min_x..=min_x + 2 {
            for z in min_z..=min_z + 2 {
                let feet = BlockPos::new(x, bed.y, z);
                if has_solid_top_surface_v5(world, feet.down())
                    && !block_v5(world, feet).props().material.is_opaque()
                    && !block_v5(world, feet.up()).props().material.is_opaque()
                {
                    return Some(feet);
                }
            }
        }
    }
    None
}

/// java17: `World.doesBlockHaveSolidTopSurface` (`world/World.java:2438-2443`) for the block
/// `World.getBlock` reads at `pos` ([`block_v5`]). The rule is mc-data's
/// [`mc_data::v5::has_solid_top_surface_v5`]: 1.8.9's, except that a snow layer's top is solid
/// at eight layers, where 1.8.9's is at seven.
fn has_solid_top_surface_v5(world: &World, pos: BlockPos) -> bool {
    mc_data::v5::has_solid_top_surface_v5(block_v5(world, pos))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::channel::NetHandle;
    use crate::net::{ToGame, ToNet};
    use crate::session::{JoinSettings, Phase};
    use mc_data::block_by_name;
    use mc_math::DVec3;
    use mc_physics::state::PhysicsState;
    use mc_protocol::v5::{c2s as c2s5, s2c as s2c5, C2SPlay5, S2CPlay5};
    use mc_version::GameVersion;
    use tokio::sync::mpsc::{Sender, UnboundedReceiver};

    /// A session in play on a stone floor (top at y 64), its player standing at the origin.
    fn rig(version: GameVersion) -> (Session, Sender<ToGame>, UnboundedReceiver<ToNet>) {
        let (handle, to_game, to_net) = NetHandle::new_pair();
        let mut session = Session::with_version(handle, JoinSettings::default(), version);
        session.phase = Phase::Playing;
        session.state.local.entity_id = 3;
        let stone = BlockState::new(block_by_name("stone").expect("stone"), 0);
        mc_physics::scene::fill(
            &mut session.state.world,
            BlockPos::new(-4, 63, -4),
            BlockPos::new(4, 63, 4),
            stone,
        );
        session.predictor = Some(Predictor::new(PhysicsState::new_player(DVec3::new(
            0.5, 64.0, 0.5,
        ))));
        (session, to_game, to_net)
    }

    /// Applies one 1.7.10 packet the way the tick's drain does.
    fn apply_v5(session: &mut Session, packet: S2CPlay5) {
        session
            .inbound_v5
            .push_back(ToGame::PacketV5(Box::new(packet)));
        assert!(session.apply_next_v5(&mut Vec::new()));
    }

    /// The (box bottom, stance) of every position report in `sent`.
    fn stances(sent: &[C2SPlay5]) -> Vec<(f64, f64)> {
        sent.iter()
            .filter_map(|packet| match packet {
                C2SPlay5::PlayerPosition(p) => Some((p.y, p.stance)),
                C2SPlay5::PlayerPositionAndLook(p) => Some((p.y, p.stance)),
                _ => None,
            })
            .collect()
    }

    fn player_v5(session: &Session) -> (PhysicsState, PlayerV5) {
        let predictor = session.predictor.as_ref().expect("player");
        (predictor.state, *predictor.v5().expect("the 1.7.10 player"))
    }

    fn bed_foot(facing: u8) -> BlockState {
        BlockState::new(block_by_name("bed").expect("bed"), facing)
    }

    /// The movement-family packets queued since the last drain: (1.8.9, protocol 5).
    fn drain(to_net: &mut UnboundedReceiver<ToNet>) -> (Vec<C2SPlay>, Vec<C2SPlay5>) {
        let (mut v47, mut v5) = (Vec::new(), Vec::new());
        while let Ok(command) = to_net.try_recv() {
            match command {
                ToNet::Send(
                    packet @ (C2SPlay::Player(_)
                    | C2SPlay::PlayerPosition(_)
                    | C2SPlay::PlayerLook(_)
                    | C2SPlay::PlayerPositionAndLook(_)
                    | C2SPlay::EntityAction(_)),
                ) => v47.push(packet),
                ToNet::SendV5(packet) => v5.push(packet),
                _ => {}
            }
        }
        (v47, v5)
    }

    #[test]
    fn a_1_7_10_session_ticks_the_1_7_10_player_and_reports_the_eye_stance() {
        let (mut session, _to_game, mut to_net) = rig(GameVersion::V1_7_10);
        session.tick(&MoveInput::default());
        let predictor = session.predictor.as_ref().expect("player");
        let eye = predictor.v5().expect("the 1.7.10 player").pos_y();
        assert_eq!(predictor.state.step_height, 0.5);
        assert_eq!(eye, 64.0 + f64::from(1.62_f32));
        let (v47, v5) = drain(&mut to_net);
        assert!(v47.is_empty(), "no 1.8.9 movement packet: {v47:?}");
        // The first move starts with no motion, so the player only lands on the next tick.
        assert_eq!(
            v5,
            vec![C2SPlay5::PlayerPosition(c2s5::PlayerPosition {
                x: 0.5,
                y: 64.0,
                stance: eye,
                z: 0.5,
                on_ground: false,
            })]
        );

        // Sneaking: the C0B edge, then the C03; the eye drops by the 0.2 x 0.4 offset.
        let sneak = MoveInput {
            sneak: true,
            ..MoveInput::default()
        };
        session.tick(&sneak);
        let (_, v5) = drain(&mut to_net);
        assert_eq!(
            v5,
            vec![
                C2SPlay5::EntityAction(c2s5::EntityAction {
                    entity_id: 3,
                    action_id: 1,
                    aux_data: 0,
                }),
                C2SPlay5::Player(c2s5::Player { on_ground: true }),
            ]
        );
        let predictor = session.predictor.as_ref().expect("player");
        assert_eq!(predictor.v5().expect("v5").pos_y(), 65.539_999_999_105_93);
    }

    /// java17 `EntityPlayer.onDeath` (entity status 3): the 0.2 x 0.2 box rebuilt from the eye,
    /// then `yOffset` 0.1F, so the dying player's reports carry a stance 0.1 above the box.
    #[test]
    fn a_dying_1_7_10_player_reports_a_stance_a_tenth_above_its_box() {
        let (mut session, _to_game, mut to_net) = rig(GameVersion::V1_7_10);
        session.tick(&MoveInput::default());
        session.tick(&MoveInput::default());
        drain(&mut to_net);
        let (_, alive) = player_v5(&session);
        let eye = alive.pos_y();
        assert_eq!(eye, 64.0 + f64::from(1.62_f32));

        apply_v5(
            &mut session,
            S2CPlay5::EntityStatus(s2c5::EntityStatus {
                entity_id: 3,
                opcode: 3,
            }),
        );
        let (state, v5) = player_v5(&session);
        assert_eq!(v5.y_offset(), DEATH_Y_OFFSET);
        assert_eq!(v5.pos_y(), eye, "posY waits for the next move");
        assert_eq!((state.width, state.height), (0.2, 0.2));
        assert_eq!(state.aabb().min.y, eye - f64::from(1.62_f32));

        session.tick(&MoveInput::default());
        let (_, sent) = drain(&mut to_net);
        let reports = stances(&sent);
        let [(y, stance)] = reports.as_slice() else {
            panic!("the death kick moves the player once: {sent:?}");
        };
        assert!(*y > 64.0, "the kick lifts the box: {y}");
        assert_eq!(*stance, y + f64::from(0.1_f32));
        // Control: the living offset would have reported the eye 1.62 up.
        assert_ne!(*stance, y + f64::from(1.62_f32));
    }

    /// java17 `EntityPlayer.sleepInBedAt` and `wakeUpPlayer`: in bed `yOffset` is 0.2F with the
    /// eye at `y + 0.9375F`; waking puts 1.62F back and stands the player 0.1 above the exit
    /// cell, its eye summed in float.
    #[test]
    fn a_1_7_10_player_sleeps_at_eye_level_and_wakes_standing() {
        let (mut session, _to_game, mut to_net) = rig(GameVersion::V1_7_10);
        let bed = BlockPos::new(1, 64, 1);
        // Metadata 0, a south-facing foot: java17 `dz` 0.9.
        mc_physics::scene::fill(&mut session.state.world, bed, bed, bed_foot(0));
        session.tick(&MoveInput::default());
        drain(&mut to_net);

        apply_v5(
            &mut session,
            S2CPlay5::UseBed(s2c5::UseBed {
                player_id: 3,
                x: 1,
                y: 64,
                z: 1,
            }),
        );
        let (state, v5) = player_v5(&session);
        assert_eq!(v5.y_offset(), SLEEP_Y_OFFSET);
        assert_eq!(v5.pos_y(), 64.9375);
        assert_eq!(
            state.pos,
            DVec3::new(1.5, 64.737_499_997_019_77, f64::from(1.0_f32 + 0.9))
        );
        // Control: 1.8.9 puts the feet at `y + 0.6875F`.
        assert_ne!(state.pos.y, 64.6875);

        // Asleep, the player settles on the bed's 9/16 top and reports with the 0.2 offset.
        for _ in 0..30 {
            session.tick(&MoveInput::default());
        }
        let (_, sent) = drain(&mut to_net);
        let reports = stances(&sent);
        assert!(reports.len() >= 2, "the fall and a re-report: {sent:?}");
        for (y, stance) in reports {
            assert_eq!(stance, y + f64::from(0.2_f32));
        }
        assert_eq!(player_v5(&session).0.pos.y, 64.5625);

        // S0B animation 2 wakes it: the first free cell of the exit search is (0, 64, 0).
        apply_v5(
            &mut session,
            S2CPlay5::Animation(s2c5::Animation {
                entity_id: 3,
                animation_type: 2,
            }),
        );
        let (state, v5) = player_v5(&session);
        assert_eq!(v5.y_offset(), PLAYER_Y_OFFSET);
        assert_eq!(v5.pos_y(), f64::from(64.0_f32 + 1.62 + 0.1));
        assert_eq!(v5.pos_y(), 65.720_001_220_703_12);
        assert_eq!(state.pos, DVec3::new(0.5, 64.100_001_215_934_75, 0.5));
        assert_eq!((state.width, state.height), (0.6, 1.8));
        // Control: 1.8.9 stands the feet at `(float)64 + 0.1F`.
        assert_ne!(state.pos.y, f64::from(64.0_f32 + 0.1));
    }

    /// java17 `World.blockExists` is `0 <= y < 256` on a client: a bed in a column never sent
    /// reads metadata 0 (south, `dz` 0.9) where 1.8.9 centres the player; outside the height
    /// range java17 centres it.
    #[test]
    fn a_1_7_10_bed_offset_reads_an_unsent_column_and_centres_out_of_range() {
        let (mut session, _to_game, _to_net) = rig(GameVersion::V1_7_10);
        session.tick(&MoveInput::default());
        apply_v5(
            &mut session,
            S2CPlay5::UseBed(s2c5::UseBed {
                player_id: 3,
                x: 100,
                y: 64,
                z: 100,
            }),
        );
        let (state, _) = player_v5(&session);
        assert_eq!(
            state.pos,
            DVec3::new(100.5, 64.737_499_997_019_77, f64::from(100.0_f32 + 0.9))
        );

        // A protocol-5 bed y is a signed byte: -1 is below the world.
        apply_v5(
            &mut session,
            S2CPlay5::UseBed(s2c5::UseBed {
                player_id: 3,
                x: 1,
                y: -1,
                z: 1,
            }),
        );
        let (state, v5) = player_v5(&session);
        assert_eq!(v5.pos_y(), f64::from(-1.0_f32 + 0.9375));
        assert_eq!(state.pos.x, 1.5);
        assert_eq!(state.pos.z, 1.5);
    }

    /// java17 `BlockBed.getSafeExitLocation`: a cell is free unless its blocks are opaque, so
    /// glass does not block it (1.8.9: not solid, so it does), and a snow layer floor needs
    /// eight layers (1.8.9: seven).
    #[test]
    fn the_1_7_10_bed_exit_passes_glass_and_wants_eight_snow_layers() {
        let (mut session, _to_game, _to_net) = rig(GameVersion::V1_7_10);
        let world = &mut session.state.world;
        let bed = BlockPos::new(1, 64, 1);
        mc_physics::scene::fill(world, bed, bed, bed_foot(0));
        let first = BlockPos::new(0, 64, 0);
        let glass = BlockState::new(block_by_name("glass").expect("glass"), 0);
        mc_physics::scene::fill(world, first, first, glass);
        assert_eq!(safe_bed_exit_v5(world, bed), Some(first));
        assert_eq!(
            super::super::safe_bed_exit(world, bed),
            Some(BlockPos::new(0, 64, 1))
        );

        let snow_layer = block_by_name("snow_layer").expect("snow layer");
        let snow = |layers: u8| BlockState::new(snow_layer, layers - 1);
        mc_physics::scene::fill(world, first, first, BlockState::AIR);
        mc_physics::scene::fill(world, first.down(), first.down(), snow(7));
        assert_eq!(safe_bed_exit_v5(world, bed), Some(BlockPos::new(0, 64, 1)));
        assert_eq!(super::super::safe_bed_exit(world, bed), Some(first));
        mc_physics::scene::fill(world, first.down(), first.down(), snow(8));
        assert_eq!(safe_bed_exit_v5(world, bed), Some(first));
        assert_eq!(
            super::super::safe_bed_exit(world, bed),
            Some(BlockPos::new(0, 64, 1))
        );
    }

    /// java17 `EntityPlayer.preparePlayerToSpawn`: a player a later JoinGame keeps gets
    /// `yOffset` 1.62F back and is placed from its eye.
    #[test]
    fn a_kept_1_7_10_player_gets_the_standing_offset_back() {
        let (mut session, _to_game, _to_net) = rig(GameVersion::V1_7_10);
        session.tick(&MoveInput::default());
        session.tick(&MoveInput::default());
        let (before, alive) = player_v5(&session);
        // Alive and awake: the placement from the eye leaves the box where it was.
        session.prepare_reused_player_to_spawn();
        let (state, v5) = player_v5(&session);
        assert_eq!(v5.y_offset(), PLAYER_Y_OFFSET);
        assert_eq!(state.aabb().min.y, alive.pos_y() - f64::from(1.62_f32));
        assert_eq!(state.aabb().min.y, before.aabb().min.y);

        apply_v5(
            &mut session,
            S2CPlay5::EntityStatus(s2c5::EntityStatus {
                entity_id: 3,
                opcode: 3,
            }),
        );
        assert_eq!(player_v5(&session).1.y_offset(), DEATH_Y_OFFSET);
        session.prepare_reused_player_to_spawn();
        let (state, v5) = player_v5(&session);
        assert_eq!(v5.y_offset(), PLAYER_Y_OFFSET);
        assert_eq!((state.width, state.height), (0.6, 1.8));
    }

    /// The negative control: a 1.8.9 session's player and packets are the 1.8.9 ones.
    #[test]
    fn a_1_8_9_session_keeps_the_1_8_9_tick() {
        let (mut session, _to_game, mut to_net) = rig(GameVersion::V1_8_9);
        session.tick(&MoveInput::default());
        let predictor = session.predictor.as_ref().expect("player");
        assert!(predictor.v5().is_none());
        assert_eq!(predictor.state.step_height, 0.6);
        let (v47, v5) = drain(&mut to_net);
        assert!(v5.is_empty(), "{v5:?}");
        assert!(
            matches!(v47.as_slice(), [C2SPlay::PlayerPosition(_)]),
            "{v47:?}"
        );
        // The rider-eye, death, bed and respawn hooks are no-ops without a 1.7.10 player.
        let before = session.predictor.as_ref().expect("player").state;
        session.sync_rider_eye_v5();
        session.death_pose_v5();
        session.enter_bed_v5(BlockPos::new(0, 64, 0));
        session.state.local.sleeping_at = Some(BlockPos::new(0, 64, 0));
        session.wake_from_bed_v5();
        session.prepare_reused_player_to_spawn_v5();
        let predictor = session.predictor.as_ref().expect("player");
        assert!(predictor.v5().is_none());
        assert_eq!(predictor.state, before);
    }
}
