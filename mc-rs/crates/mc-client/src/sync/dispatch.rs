//! The dispatch: every one of the 74 clientbound Play packets, applied to [`ClientState`].
//!
//! java: `client/network/NetHandlerPlayClient.java` — 2,125 lines of `handleXxx`.
//!
//! # Coverage
//!
//! [`apply`] matches on [`S2CPlay`] **exhaustively**, with no `_ =>` arm. That is the guarantee
//! behind "all 74 handled": adding a variant to `mc-protocol` breaks this file's build rather
//! than silently dropping a packet. `tests/dispatch_coverage.rs` walks all 74 wire ids as a
//! second, runtime check.
//!
//! Five packets arrive here only if a caller feeds them in directly. The connection task consumes
//! `KeepAlive`, `SetCompression` and `Disconnect` itself, and sends `PlayerListHeaderFooter` plus
//! `ResourcePackSend` through its Netty-bypass lane (see [`crate::net::task`]). They are still
//! handled, so the dispatch is usable standalone in tests.
//!
//! # What "handled" means, packet by packet
//!
//! Vanilla's handlers do three kinds of thing. This module keeps them separate:
//!
//! 1. **Mutate world/ECS/tab-list/environment state** — done here, in full.
//! 2. **Call a subsystem S14 does not own** (sound, particles, screens, containers, the
//!    scoreboard overlay) — reported as a [`ClientEvent`] carrying the packet's payload.
//! 3. **Touch something outside this port's scope** (achievements, the demo screens, Realms) —
//!    reported as an event and otherwise ignored, exactly as an unmodded client with those
//!    screens absent would behave.

use mc_data::{BlockState, EntityType};
use mc_math::{Aabb, BlockPos, ChunkPos, DVec3, JavaRandom};
use mc_protocol::play::{s2c, S2CPlay};
use mc_ui::font::component::ChatComponent;
use mc_world::BulkChunkEntry;

use crate::error::ClientError;
use crate::sync::entities::{TrackedKind, FIXED_POINT_SCALE, VELOCITY_SCALE};
use crate::sync::event::{ChatPosition, ClientEvent, ScoreboardEvent, WindowEvent};
use crate::sync::state::{game_mode_from_id, Abilities, ClientState};

/// `EntityArrow.setVelocity`'s spawn-time look angles.
///
/// `NetHandlerPlayClient.handleSpawnObject` first installs the two angle bytes from the packet,
/// then calls `setVelocity` when the object data is positive.  For arrows that override notices
/// their previous angles are still zero and replaces the packet angles with ones derived from
/// the motion vector.  The horizontal length deliberately round-trips through
/// `MathHelper.sqrt_double`'s `float` result before `atan2` sees it, and the degree conversion
/// divides by the bytecode's float-widened pi, not `Math.PI` (`javap -c wq.i(DDD)V`).
/// java: `EntityArrow.setVelocity` (`:157-173`).
#[must_use]
fn arrow_spawn_rotation(velocity: DVec3) -> (f32, f32) {
    crate::sync::entities::projectile_motion_rotation(velocity)
}

/// Decode one `S0FPacketSpawnMob` velocity component.
///
/// `NetHandlerPlayClient.handleSpawnMob` deliberately divides two `float`s and only then casts
/// the result to `double`: `(double)((float)packetIn.getVelocityX() / 8000.0F)`.  This is unlike
/// `S12PacketEntityVelocity` and spawn-object velocity, whose handlers use a `double` divisor.
/// Keeping the narrowing here matters because the resulting motion is retained and integrated
/// by the remote-entity client tick.
/// java: `NetHandlerPlayClient.handleSpawnMob` (`:940-942`).
#[must_use]
fn spawn_mob_velocity_component(component: i16) -> f64 {
    f64::from(component as f32 / 8000.0_f32)
}

/// Whether an S20 key names an attribute registered by a vanilla 1.8.9 living entity.
///
/// `BaseAttributeMap.getAttributeInstanceByName` checks both the unlocalized-name map and the
/// optional description map case-insensitively. The seven names below are the complete vanilla
/// registration set: five shared attributes plus Horse jump strength and Zombie reinforcement
/// chance. Descriptions exist for every entry except attack damage.
#[cfg(test)]
#[must_use]
fn is_registered_vanilla_attribute(key: &str) -> bool {
    const NAMES: &[&str] = &[
        "generic.maxHealth",
        "Max Health",
        "generic.followRange",
        "Follow Range",
        "generic.knockbackResistance",
        "Knockback Resistance",
        "generic.movementSpeed",
        "Movement Speed",
        "generic.attackDamage",
        "horse.jumpStrength",
        "Jump Strength",
        "zombie.spawnReinforcements",
        "Spawn Reinforcements Chance",
    ];
    NAMES.iter().any(|name| key.eq_ignore_ascii_case(name))
}

/// Reproduce the random launch motion from `EntityXPOrb(World, x, y, z, value)`.
///
/// The Java constructor consumes three `Math.random()` doubles, narrows each horizontal
/// expression to `float`, then multiplies by the `float` literal `2.0F`. `Math.random()` is a
/// JVM-global, entropy-seeded stream, so the client uses a stable entity-id seed as its
/// reproducible surrogate; the distribution and narrowing point are the parity-bearing parts.
/// java: `EntityXPOrb.java:25-28`.
#[must_use]
fn xp_orb_spawn_motion(random: &mut JavaRandom) -> DVec3 {
    let horizontal = |random: &mut JavaRandom| {
        f64::from(
            (random.next_f64() * 0.200_000_002_980_232_24 - 0.100_000_001_490_116_12) as f32
                * 2.0_f32,
        )
    };
    let vertical = f64::from((random.next_f64() * 0.2) as f32 * 2.0_f32);
    DVec3::new(horizontal(random), vertical, horizontal(random))
}

/// `EntityFireball`'s constructor normalizes the spawn packet's speed vector into a fixed
/// `0.1` blocks/tick² acceleration. The constructor uses `MathHelper.sqrt_double`, so the
/// length is narrowed to `f32` before the three `f64` divisions.
/// java: `EntityFireball(World, x, y, z, accelX, accelY, accelZ)` (`:60-70`).
#[must_use]
fn fireball_spawn_acceleration(speed: Option<(i16, i16, i16)>) -> mc_world::Acceleration {
    let raw = match speed {
        Some((x, y, z)) => DVec3::new(
            f64::from(x) / VELOCITY_SCALE,
            f64::from(y) / VELOCITY_SCALE,
            f64::from(z) / VELOCITY_SCALE,
        ),
        // `S0E` only serializes speed when object data is positive. Vanilla still constructs
        // an unowned fireball with the default zero fields, and 0/0 faithfully leaves its
        // acceleration NaN rather than silently choosing a direction.
        None => DVec3::ZERO,
    };
    let magnitude = f64::from(mc_math::sqrt_d(
        raw.x * raw.x + raw.y * raw.y + raw.z * raw.z,
    ));
    mc_world::Acceleration(DVec3::new(
        raw.x / magnitude * 0.1,
        raw.y / magnitude * 0.1,
        raw.z / magnitude * 0.1,
    ))
}

/// Whether this spawn-object type reaches `EntityFireball`'s acceleration constructor.
/// java: `NetHandlerPlayClient.handleSpawnObject` types 63, 64 and 66.
#[must_use]
fn is_fireball_spawn(kind: EntityType) -> bool {
    matches!(
        kind,
        EntityType::Fireball | EntityType::SmallFireball | EntityType::WitherSkull
    )
}

/// Vanilla's `EntityMinecart.EnumMinecartType.byNetworkID`.
/// java: `entity/item/EntityMinecart.java`
fn minecart_type(data: i32) -> EntityType {
    match data {
        1 => EntityType::MinecartChest,
        2 => EntityType::MinecartFurnace,
        3 => EntityType::MinecartTNT,
        4 => EntityType::MinecartSpawner,
        5 => EntityType::MinecartHopper,
        6 => EntityType::MinecartCommandBlock,
        // java: `byNetworkID` falls back to RIDEABLE for anything it does not recognise.
        _ => EntityType::MinecartRideable,
    }
}

/// Map an `S0EPacketSpawnObject` type id (a hard-coded table, *not* `EntityList`) onto the
/// `EntityList` type the spawned entity actually has.
///
/// java: the long `if/else if` chain in `handleSpawnObject` (`:297`). `data` selects the
/// minecart variant for type 10, and is otherwise irrelevant here.
///
/// Type 90 is handled directly in [`apply`]: vanilla constructs `EntityFishHook` outside
/// `EntityList` only when its owner id resolves to a player. `None` here therefore means only
/// an object type vanilla leaves as a `null` entity and never adds.
fn object_entity_type(object: mc_data::ObjectType, data: i32) -> Option<EntityType> {
    use mc_data::ObjectType as O;
    Some(match object {
        O::Boat => EntityType::Boat,
        O::ItemStack => EntityType::Item,
        O::Minecart => minecart_type(data),
        O::PrimedTnt => EntityType::PrimedTnt,
        O::EnderCrystal => EntityType::EnderCrystal,
        O::Arrow => EntityType::Arrow,
        O::Snowball => EntityType::Snowball,
        O::ThrownEgg => EntityType::ThrownEgg,
        O::LargeFireball => EntityType::Fireball,
        O::SmallFireball => EntityType::SmallFireball,
        O::ThrownEnderpearl => EntityType::ThrownEnderpearl,
        O::WitherSkull => EntityType::WitherSkull,
        O::FallingBlock => EntityType::FallingSand,
        O::ItemFrame => EntityType::ItemFrame,
        O::EyeOfEnder => EntityType::EyeOfEnderSignal,
        O::ThrownPotion => EntityType::ThrownPotion,
        O::ThrownExpBottle => EntityType::ThrownExpBottle,
        O::FireworkRocket => EntityType::FireworksRocketEntity,
        O::LeashKnot => EntityType::LeashKnot,
        O::ArmorStand => EntityType::ArmorStand,
        O::FishingBobber => return None,
    })
}

/// Whether `handleSpawnObject` zeroes this object type's `data` before the velocity guard.
///
/// java: `NetHandlerPlayClient.handleSpawnObject` (`:297-436`) — nine of the type branches end in
/// `packetIn.func_149002_g(0)`, which mutates the packet so that the
/// `if (packetIn.func_149009_m() > 0) { ... entity.setVelocity(...) }` guard forty lines later
/// sees zero and skips. The three fireball types (63, 64 and 66) take the speed fields through
/// their constructor as **acceleration**, normalized to magnitude `0.1`; motion itself starts at
/// zero. The rest are simply meant to have no initial motion.
///
/// This is exactly the decompiler-artefact hazard the brief warns about: the calls read as dead
/// stores. Evaluating `data > 0` against the *original* value gives a splash potion, a falling
/// block or an item frame the packet's velocity and makes it drift on spawn.
fn velocity_is_zeroed(object_type: i8) -> bool {
    matches!(
        object_type,
        // 63 large fireball, 64 small fireball, 66 wither skull, 70 falling block,
        // 71 item frame, 73 splash potion, 75 XP bottle, 77 leash knot, 90 fishing bobber.
        63 | 64 | 66 | 70 | 71 | 73 | 75 | 77 | 90
    )
}

/// A block-state id straight off the wire.
///
/// `S22`/`S23` carry `Block.BLOCK_STATE_IDS`' packing, which is `block_id << 4 | meta` — the
/// *same* packing as [`BlockState`], and **not** `Block.getStateId`'s `block_id | meta << 12`
/// (which `S0EPacketSpawnObject`'s falling-block data and `S28PacketEffect` use instead). S08
/// confirmed this against `Block.java:1491-1493`; mixing the two is an easy and very visible
/// bug.
///
/// java: the readers resolve the id with `Block.BLOCK_STATE_IDS.getByValue`, which returns null
/// for a negative id, one past the registry, or a meta no state registers. `None` is that null.
/// (No masking: `0x10010` must not alias stone.)
fn registered_block_state(id: i32) -> Option<BlockState> {
    let state = BlockState::from_raw(u16::try_from(id).ok()?);
    state.is_legal().then_some(state)
}

/// java: `WorldProvider.getProviderForDimension` (`WorldProvider.java:198-201`) returns a
/// provider only for the Nether, the Overworld and the End.
fn has_world_provider(dimension: i32) -> bool {
    matches!(dimension, -1..=1)
}

/// The exception `new WorldClient` raises for any other dimension: World's constructor calls
/// `providerIn.getWorldBorder()` on the null provider.
fn no_world_provider(dimension: i32) -> ClientError {
    ClientError::Protocol(mc_protocol::ProtoError::InvalidEnum {
        what: "dimension (null WorldProvider)",
        value: dimension,
    })
}

/// The exception a null S22/S23 state raises: `World.setBlockState` dereferences it
/// (`newState.getBlock()`) after the position check, and the scheduled handler aborts there.
fn unregistered_block_state(id: i32) -> ClientError {
    ClientError::Protocol(mc_protocol::ProtoError::InvalidEnum {
        what: "BLOCK_STATE_IDS entry (null IBlockState)",
        value: id,
    })
}

/// Reconstruct one affected-block coordinate from `S27PacketExplosion`.
///
/// Java evaluates `buf.readByte() + origin` as an `int` expression, whose two's-complement
/// overflow wraps even when the packet's float origin narrowed to `Integer.MIN/MAX_VALUE`.
/// Explicit wrapping keeps debug builds on that same path instead of panicking on a hostile
/// coordinate.
fn explosion_axis_coordinate(origin: i32, offset: i8) -> i32 {
    origin.wrapping_add(i32::from(offset))
}

/// Recover the absolute block list that vanilla's `S27PacketExplosion` reader stores in its
/// `affectedBlockPositions`.  Keeping this conversion available to the event consumer prevents
/// the particle path from accidentally using the packet's offsets as world coordinates.
///
/// Java: `S27PacketExplosion.readPacketData` (`:52-60`) — `(int)posX` truncates toward zero,
/// then each signed byte is promoted to `int` and added with normal two's-complement overflow.
pub(crate) fn explosion_affected_positions(explosion: &s2c::Explosion) -> Vec<BlockPos> {
    let origin = BlockPos::new(explosion.x as i32, explosion.y as i32, explosion.z as i32);
    explosion
        .affected_blocks
        .iter()
        .map(|block| {
            BlockPos::new(
                explosion_axis_coordinate(origin.x, block.dx),
                explosion_axis_coordinate(origin.y, block.dy),
                explosion_axis_coordinate(origin.z, block.dz),
            )
        })
        .collect()
}

/// Snapshot the already-loaded hanging entities that `EntityPainting.onValidSurface` would see
/// while constructing another hanging entity. `World.getEntitiesWithinAABBExcludingEntity` is an
/// insertion-order world query; `EntityTracker::iter` supplies that same resident-only view.
/// The id being spawned is excluded because `WorldClient.addEntityToWorld` removes an old
/// generation before inserting the new one, while Java constructs the replacement first.
fn existing_hanging_boxes(state: &ClientState, excluding_id: i32) -> Vec<Aabb> {
    state
        .entities
        .iter()
        .filter_map(|(network_id, tracked)| {
            if network_id == excluding_id {
                return None;
            }
            let position = state
                .world
                .ecs
                .get::<&mc_world::entity::Position>(tracked.entity)
                .ok()?
                .0;
            match tracked.kind.entity_type()? {
                EntityType::Painting => {
                    let hanging = state
                        .world
                        .ecs
                        .get::<&crate::app::entrender::Hanging>(tracked.entity)
                        .ok()?;
                    Some(crate::app::interact::hanging_bounding_box(
                        position,
                        hanging.facing,
                        hanging.art.size_x,
                        hanging.art.size_y,
                    ))
                }
                EntityType::ItemFrame => {
                    let facing = state
                        .world
                        .ecs
                        .get::<&mc_world::ItemFrameFacing>(tracked.entity)
                        .ok()?;
                    Some(crate::app::interact::hanging_bounding_box(
                        position, facing.0, 12, 12,
                    ))
                }
                EntityType::LeashKnot => {
                    Some(crate::app::interact::leash_knot_bounding_box(position))
                }
                _ => None,
            }
        })
        .collect()
}

/// [`apply`] for a full-column chunk packet whose columns were unpacked off the game thread
/// ([`crate::sync::prebuild`], #15 / NET-1): `columns` holds one column per packet entry, in
/// packet order, built with the `assumed_sky` payload layout.
///
/// Same world effects and the same events, in the same order, as `apply`'s `ChunkData` (full
/// column) and `MapChunkBulk` arms, which must stay in step with this: the only difference is
/// that each column is installed ready-made (`World::install_full_column`) instead of being
/// unpacked here. Anything that does not fit — another packet shape, a column count that does
/// not match, or an `S21` built for a sky-light layout the world no longer has — goes through
/// `apply` unchanged. Pinned by `prebuilt_chunk_packets_apply_exactly_like_plain_ones`.
///
/// # Errors
///
/// Only through the `apply` fallback; installing a ready column cannot fail.
pub fn apply_prebuilt_chunks(
    state: &mut ClientState,
    packet: S2CPlay,
    assumed_sky: bool,
    columns: Vec<mc_world::Chunk>,
    events: &mut Vec<ClientEvent>,
) -> Result<(), ClientError> {
    match packet {
        S2CPlay::ChunkData(chunk)
            if chunk.ground_up_continuous
                && chunk.primary_bit_mask != 0
                && columns.len() == 1
                && assumed_sky == state.world.has_sky_light() =>
        {
            let position = ChunkPos::new(chunk.chunk_x, chunk.chunk_z);
            for column in columns {
                state.world.install_full_column(position, column);
            }
            let _ = state.entities.activate_chunk(&state.world, position);
            events.push(ClientEvent::ChunkLoaded(position));
            Ok(())
        }
        S2CPlay::MapChunkBulk(bulk)
            if bulk.has_sky_light == assumed_sky && columns.len() == bulk.columns.len() =>
        {
            for (entry, column) in bulk.columns.iter().zip(columns) {
                let position = ChunkPos::new(entry.chunk_x, entry.chunk_z);
                state.world.install_full_column(position, column);
            }
            for entry in &bulk.columns {
                let position = ChunkPos::new(entry.chunk_x, entry.chunk_z);
                let _ = state.entities.activate_chunk(&state.world, position);
                events.push(ClientEvent::ChunkLoaded(position));
            }
            Ok(())
        }
        other => apply(state, other, events),
    }
}

/// Apply one decoded clientbound Play packet.
///
/// Events are pushed onto `events` rather than returned, so a caller draining a burst of a
/// thousand packets allocates one vector rather than a thousand.
///
/// # Errors
///
/// [`ClientError::World`] if a chunk payload's length does not match its bitmask, or
/// [`ClientError::Chat`] if a component the server sent does not parse. Both are server faults;
/// neither can panic the client.
#[allow(clippy::too_many_lines)] // 74 arms; splitting it would only hide the exhaustiveness.
pub fn apply(
    state: &mut ClientState,
    packet: S2CPlay,
    events: &mut Vec<ClientEvent>,
) -> Result<(), ClientError> {
    match packet {
        // ---- 0x00 / 0x40 / 0x46: normally consumed by the connection task ----
        S2CPlay::KeepAlive(_) => {}
        S2CPlay::SetCompression(_) => {}
        S2CPlay::Disconnect(s2c::Disconnect { reason }) => {
            return Err(ClientError::Disconnected { reason });
        }

        // ---- lifecycle ----
        // java: handleJoinGame (:277)
        S2CPlay::JoinGame(join) => {
            // java: `new WorldClient` takes `WorldProvider.getProviderForDimension`, null for
            // any id but -1, 0 and 1, and World's constructor dereferences it. Only the
            // PlayerControllerMP has been replaced by then: the NPE aborts the scheduled task
            // before the world, the player and the C15/C17 replies (vanilla never leaves the
            // connecting screen on a first join like this).
            if !has_world_provider(i32::from(join.dimension)) {
                return Err(no_world_provider(i32::from(join.dimension)));
            }
            // java: `Minecraft.loadWorld` builds an EntityPlayerSP only while `thePlayer` is
            // null (the connection's first JoinGame); a mid-connection JoinGame keeps it.
            let keeps_player = state.local.entity_id_known;
            state.reset_world_for_join_with_level_type(i32::from(join.dimension), &join.level_type);
            // **ENTITY-2's D5.** `state.players.clear()` used to be here and vanilla has
            // no such line: `NetHandlerPlayClient.playerInfoMap` is written in exactly
            // three places, all inside `handlePlayerListItem` (`:1626`, `:1630`, `:1635`),
            // and the tab header/footer are dropped only by `Minecraft.loadWorld(null)`
            // (`:2388`) — i.e. when the *connection* ends, which is `PlayerList::clear`'s
            // real caller. A second `S01PacketJoinGame` on one connection is not
            // hypothetical: a proxy sends one on every server switch, and clearing here
            // emptied the tab list until each player happened to be re-announced.
            state.local.entity_id = join.entity_id;
            state.local.game_mode = game_mode_from_id(i32::from(join.game_mode));
            state.local.hardcore = join.hardcore_mode;
            state.local.level_type =
                mc_world::world::canonical_level_type_name(&join.level_type).to_owned();
            state.local.max_players = join.max_players;
            state.local.reduced_debug_info = join.reduced_debug_info;
            state.local.entity_id_known = true;
            if keeps_player {
                // `EntityPlayer.preparePlayerToSpawn` on the kept player: `setHealth` to its own
                // (kept) max health and `deathTime = 0`. Its `hasValidHealth` and hurt timers
                // survive, so the next S06 is compared with this value.
                state.local.health = state.local.max_health;
                state.local.death_time = 0;
            } else {
                // A first join constructs the local EntityPlayerSP. Its first S06 is an initial
                // snapshot, not damage against the default 20 health value.
                state.local.reset_health_pipeline();
            }
            state.env.difficulty = difficulty_from_id(join.difficulty);
            events.push(ClientEvent::JoinedGame(Box::new(join)));
        }
        // java: handleRespawn (:1056) — the world is rebuilt only when the dimension changes.
        S2CPlay::Respawn(respawn) => {
            let changed_world = respawn.dimension != state.local.dimension;
            if changed_world && !has_world_provider(respawn.dimension) {
                // java: handleRespawn clears doneLoadingTerrain (`:1062`), then the null
                // provider throws in the new WorldClient's constructor. The old world, player,
                // game mode and screen all stay; only the cleared latch survives the NPE.
                state.local.done_loading_terrain = false;
                return Err(no_world_provider(respawn.dimension));
            }
            if changed_world {
                state.reset_world_for_dimension_change_with_level_type(
                    respawn.dimension,
                    &respawn.level_type,
                );
                // These are arguments to WorldClient's constructor.  A same-dimension S07
                // creates a new player in the existing WorldClient, so it does not consume
                // either packet field.
                state.env.difficulty = difficulty_from_id(respawn.difficulty);
            }

            // `setDimensionAndSpawnPlayer` does these even when `handleRespawn` kept the old
            // WorldClient: reset its temporary spawn and remove every old remote entity before
            // constructing the fresh local EntityPlayerSP.
            state.env.spawn_position = BlockPos::new(8, 64, 8);
            state.entities.clear();
            state.local.game_mode = game_mode_from_id(i32::from(respawn.game_mode));
            // `setDimensionAndSpawnPlayer` creates a fresh PlayerCapabilities object with Java's
            // defaults. `handleRespawn` then calls `setGameType` once with S07's game mode;
            // Creative intentionally leaves that fresh `isFlying=false` untouched. This is not
            // deferred until an optional S39PacketPlayerAbilities arrives.
            state.local.abilities = Abilities::default();
            state
                .local
                .abilities
                .configure_for_game_mode(state.local.game_mode);
            // `Minecraft.setDimensionAndSpawnPlayer` constructs a new EntityPlayerSP even for a
            // same-dimension death respawn. Its next S06 must therefore take setPlayerSPHealth's
            // first-packet path, with no stale hurt timers from the old life.
            state.local.reset_health_pipeline();
            // java: `Minecraft.setDimensionAndSpawnPlayer` (`Minecraft.java:2445-2447`) copies
            // every watched value from the old EntityPlayerSP into the replacement, and
            // (`:2457`) copies its reduced-debug latch. Those are player-instance values, not
            // S07 fields, so preserve them across both same- and cross-dimension respawns.
            // java: `handleRespawn` resets `doneLoadingTerrain` only inside the
            // dimension-change branch that opens a new `GuiDownloadTerrain`. A same-dimension
            // respawn leaves the latch set and the next S08 must not masquerade as a fresh
            // terrain download completing.
            if changed_world {
                state.local.done_loading_terrain = false;
            }
            events.push(ClientEvent::Respawned {
                dimension: respawn.dimension,
                changed_world,
            });
            events.push(ClientEvent::AbilitiesChanged(state.local.abilities));
        }
        // java: handlePlayerPosLook (:669) — the caller must answer with C06, and the first one
        // after a (re)spawn is what dismisses GuiDownloadTerrain.
        S2CPlay::PlayerPositionLook(look) => {
            events.push(ClientEvent::PositionCorrection(
                mc_physics::prediction::PositionCorrection::from_packet(
                    look.x, look.y, look.z, look.yaw, look.pitch, look.flags,
                ),
            ));
            if !state.local.done_loading_terrain {
                state.local.done_loading_terrain = true;
                events.push(ClientEvent::TerrainDownloaded);
            }
        }

        // ---- chat ----
        // java: handleChat (:849)
        S2CPlay::Chat(chat) => {
            events.push(ClientEvent::Chat {
                component: Box::new(ChatComponent::from_json(&chat.component)?),
                position: ChatPosition::from_raw(chat.position),
            });
        }

        // ---- time, weather, difficulty, border, spawn ----
        // java: handleTimeUpdate (:952)
        S2CPlay::TimeUpdate(time) => {
            state.env.apply_time(time.total_world_time, time.world_time);
            events.push(ClientEvent::TimeUpdated);
        }
        // java: handleSpawnPosition (:959)
        S2CPlay::SpawnPosition(spawn) => {
            state.env.spawn_position = spawn.position;
            events.push(ClientEvent::SpawnPositionChanged(spawn.position));
        }
        // java: handleServerDifficulty (:1540)
        S2CPlay::ServerDifficulty(difficulty) => {
            state.env.difficulty = difficulty.difficulty;
            events.push(ClientEvent::DifficultyChanged(difficulty.difficulty));
        }
        // java: handleWorldBorder (:1558)
        S2CPlay::WorldBorder(border) => {
            state.env.apply_world_border(&border);
            events.push(ClientEvent::WorldBorderChanged);
        }
        // java: handleChangeGameState (:1358) — rain/thunder are ours, the rest is the UI's.
        S2CPlay::ChangeGameState(game_state) => {
            // `MESSAGE_NAMES` is checked before the reason switch. In 1.8.9 its sole entry is
            // reason 0, `tile.bed.notValid`; the float payload is unused.
            if game_state.reason == s2c::GameStateReason::InvalidBed {
                events.push(ClientEvent::Chat {
                    component: Box::new(ChatComponent::translate("tile.bed.notValid", vec![])),
                    // `EntityPlayer.addChatComponentMessage` reaches GuiNewChat directly; it
                    // has no S02 position byte and therefore uses the ordinary chat box.
                    position: ChatPosition::Chat,
                });
            } else if state
                .env
                .apply_game_state(game_state.reason, game_state.value)
            {
                events.push(ClientEvent::WeatherChanged);
            } else if game_state.reason == s2c::GameStateReason::ChangeGameMode {
                // java: `MathHelper.floor_float(f + 0.5F)` — a rounded float, not a byte.
                let id = mc_math::floor_f32(game_state.value + 0.5);
                let mode = game_mode_from_id(id);
                state.local.game_mode = mode;
                events.push(ClientEvent::GameModeChanged(mode));
            } else {
                events.push(ClientEvent::GameState {
                    reason: game_state.reason,
                    value: game_state.value,
                });
            }
        }

        // ---- local player ----
        // java: handleUpdateHealth (:1042)
        S2CPlay::UpdateHealth(health) => {
            state.local.apply_server_health(health.health);
            state.local.food_level = health.food_level;
            state.local.saturation = health.saturation_level;
            events.push(ClientEvent::HealthChanged {
                health: health.health,
                food_level: health.food_level,
                saturation: health.saturation_level,
            });
        }
        // java: handleSetExperience (:1050)
        S2CPlay::SetExperience(experience) => {
            state.local.experience_bar = experience.experience_bar;
            state.local.level = experience.level;
            state.local.total_experience = experience.total_experience;
            events.push(ClientEvent::ExperienceChanged {
                bar: experience.experience_bar,
                level: experience.level,
                total: experience.total_experience,
            });
        }
        // java: handleHeldItemChange (:598) — the range check is vanilla's, not defensive
        // padding: an out-of-range slot is ignored rather than clamped.
        S2CPlay::HeldItemChange(held) => {
            if (0..9).contains(&held.slot) {
                let slot = held.slot as u8;
                state.local.held_slot = slot;
                events.push(ClientEvent::HeldSlotChanged(slot));
            }
        }
        // java: handlePlayerAbilities (:1668)
        S2CPlay::PlayerAbilities(abilities) => {
            let mut decoded =
                Abilities::from_flags(abilities.flags, abilities.fly_speed, abilities.walk_speed);
            // S39 carries four capability flags but not `PlayerCapabilities.allowEdit`; Java
            // mutates those four fields on the existing player rather than replacing its whole
            // capabilities object.
            decoded.allow_edit = state.local.abilities.allow_edit;
            state.local.abilities = decoded;
            events.push(ClientEvent::AbilitiesChanged(decoded));
        }
        // java: handleCamera (:1547)
        S2CPlay::Camera(camera) => events.push(ClientEvent::CameraChanged {
            entity_id: camera.entity_id,
        }),
        // java: handleCombatEvent (:1515)
        S2CPlay::CombatEvent(combat) => {
            events.push(ClientEvent::CombatEvent(Box::new(combat)));
        }

        // ---- entity spawns ----
        // java: handleSpawnPlayer (:530)
        S2CPlay::SpawnPlayer(player) => {
            // `handleSpawnPlayer` constructs `EntityOtherPlayerMP` from the current
            // `NetworkPlayerInfo.getGameProfile()`. Copy the row before inserting the entity;
            // later tab-list replacement/removal must not alter this entity's name or texture
            // properties.
            let player_profile = state.players.get(&player.player_uuid).map(|info| {
                crate::sync::entities::PlayerProfileSnapshot {
                    username: info.username.clone(),
                    properties: info.properties.clone(),
                    trusted_textures: info.trusted_textures.clone(),
                }
            });
            let entity = state.entities.spawn_packet(
                &mut state.world,
                player.entity_id,
                TrackedKind::Player,
                Some(player.player_uuid),
                [player.x, player.y, player.z],
                player.yaw.to_degrees(),
                player.pitch.to_degrees(),
                DVec3::ZERO,
                crate::sync::entities::metadata_for_kind(TrackedKind::Player, &player.metadata),
            );
            let _ = state
                .entities
                .set_player_profile(player.entity_id, entity, player_profile);
            // java: `handleSpawnPlayer` (`:544-551`) seeds
            // `entityotherplayermp.inventory.mainInventory[currentItem]` before adding the
            // entity to the world.  The wire carries only an item id here (not a stack), so the
            // vanilla value is a fresh count-one, zero-damage stack; id zero means an empty
            // hand.  `S04PacketEntityEquipment` normally repeats this value immediately, but
            // that follow-up is not guaranteed for proxy/custom-server traffic and the spawn
            // packet itself must be observable on its own.
            let current_item = (player.current_item != 0).then_some(mc_protocol::ItemStack {
                item_id: player.current_item,
                count: 1,
                damage: 0,
                nbt: None,
            });
            assert!(
                state
                    .entities
                    .apply_equipment(player.entity_id, 0, current_item),
                "the player was inserted into the tracker immediately above"
            );
            events.push(ClientEvent::EntitySpawned {
                network_id: player.entity_id,
                entity,
                kind: TrackedKind::Player,
            });
        }
        // java: handleSpawnMob (:913)
        S2CPlay::SpawnMob(mob) => {
            match mc_data::entity_type(mob.entity_type) {
                Some(kind) => {
                    let entity = state.entities.spawn_packet(
                        &mut state.world,
                        mob.entity_id,
                        TrackedKind::Type(kind),
                        None,
                        [mob.x, mob.y, mob.z],
                        mob.yaw.to_degrees(),
                        mob.pitch.to_degrees(),
                        DVec3::new(
                            spawn_mob_velocity_component(mob.velocity_x),
                            spawn_mob_velocity_component(mob.velocity_y),
                            spawn_mob_velocity_component(mob.velocity_z),
                        ),
                        crate::sync::entities::metadata_for_kind(
                            TrackedKind::Type(kind),
                            &mob.metadata,
                        ),
                    );
                    // java: `handleSpawnMob` (`:925`) — `renderYawOffset = rotationYawHead =
                    // headPitch`, so the *body* yaw at spawn is the head yaw, not `mob.yaw`.
                    // This used to be a plain `apply_head_look`, which left the body
                    // on the packet yaw and made every mob swing round on its first ticks.
                    state
                        .entities
                        .apply_spawn_head_look(mob.entity_id, mob.head_pitch.to_degrees());
                    events.push(ClientEvent::EntitySpawned {
                        network_id: mob.entity_id,
                        entity,
                        kind: TrackedKind::Type(kind),
                    });
                }
                // java: `EntityList.createEntityByID` returns null and the cast to
                // EntityLivingBase would NPE; an unknown id is a broken server, not a crash.
                None => events.push(ClientEvent::EntitySpawnIgnored {
                    network_id: mob.entity_id,
                    reason: "unknown EntityList network id",
                }),
            }
        }
        // java: handleSpawnObject (:297)
        S2CPlay::SpawnObject(object) => {
            if object.object_type == mc_data::ObjectType::FishingBobber as i8 {
                // java: type 90 looks the packet's `data` up *before* clearing it. Unlike the
                // EntityList-backed object branches, `EntityFishHook` is constructed directly
                // and only when that existing object is an `EntityPlayer`.
                let owner_is_player = object.data == state.local.entity_id
                    || state
                        .entities
                        .get(object.data)
                        .is_some_and(|owner| owner.kind == TrackedKind::Player);
                if owner_is_player {
                    let entity = state.entities.spawn_packet(
                        &mut state.world,
                        object.entity_id,
                        TrackedKind::FishingBobber,
                        None,
                        [object.x, object.y, object.z],
                        object.yaw.to_degrees(),
                        object.pitch.to_degrees(),
                        DVec3::ZERO,
                        mc_world::entity::Metadata::new(),
                    );
                    // The owner id is retained for the future RenderFish line seam. It is not
                    // a velocity flag: Java clears the packet's data before its setVelocity
                    // guard, so a hook always starts at zero motion.
                    let _ = state
                        .world
                        .ecs
                        .insert_one(entity, crate::app::entrender::ObjectData(object.data));
                    // `EntityFishHook.angler` is an object reference in Java. Capture the
                    // remote player's ECS generation now; a later id reuse must not retarget an
                    // already-spawned bobber's line.
                    state
                        .entities
                        .set_fishing_bobber_owner(object.entity_id, object.data);
                    events.push(ClientEvent::EntitySpawned {
                        network_id: object.entity_id,
                        entity,
                        kind: TrackedKind::FishingBobber,
                    });
                } else {
                    events.push(ClientEvent::EntitySpawnIgnored {
                        network_id: object.entity_id,
                        reason: "fishing bobber owner is not a player",
                    });
                }
            } else {
                let resolved = mc_data::object_type(object.object_type as u8)
                    .and_then(|o| object_entity_type(o, object.data));
                match resolved {
                    // java: type 70 builds `new EntityFallingBlock(...,
                    // Block.getStateById(data & 65535))`, whose `getStateFromMeta` throws for a
                    // metadata the block rejects: the scheduled handler aborts before the entity
                    // exists (Util.runTask logs it). The falling-block tick and renderer decode
                    // this same data with the panicking form, so such a block must never spawn.
                    Some(EntityType::FallingSand)
                        if mc_data::BlockState::try_from_java_state_id(object.data & 0xFFFF)
                            .is_none() =>
                    {
                        return Err(ClientError::Protocol(
                            mc_protocol::ProtoError::InvalidEnum {
                                what: "falling block state (getStateFromMeta threw)",
                                value: object.data,
                            },
                        ));
                    }
                    Some(kind) => {
                        // java: `entity.setVelocity(...)` runs only when `func_149009_m() > 0` — but
                        // nine of the object types reach that guard with `data` already **zeroed**,
                        // because `handleSpawnObject` calls `packetIn.func_149002_g(0)` inside their
                        // own branch first (`:297-436`). Those `func_149002_g(0)` calls look like
                        // dead stores and are not: a falling block (whose `data` is a block-state id)
                        // or a splash potion (whose `data` is a damage value) must spawn with zero
                        // motion, not with the packet's velocity.
                        let velocity_data = if velocity_is_zeroed(object.object_type) {
                            0
                        } else {
                            object.data
                        };
                        let velocity = match (velocity_data > 0, object.velocity) {
                            (true, Some((x, y, z))) => DVec3::new(
                                f64::from(x) / VELOCITY_SCALE,
                                f64::from(y) / VELOCITY_SCALE,
                                f64::from(z) / VELOCITY_SCALE,
                            ),
                            _ => DVec3::ZERO,
                        };
                        // Java writes the wire angles first, but an arrow's overridden
                        // `setVelocity` immediately replaces both at spawn.  Do this only when the
                        // packet actually supplied the velocity that reaches that call: malformed
                        // zero-data / absent-velocity packets retain their wire orientation just as
                        // the superclass path does.
                        let (yaw, pitch) = if kind == EntityType::Arrow
                            && velocity_data > 0
                            && object.velocity.is_some()
                        {
                            arrow_spawn_rotation(velocity)
                        } else {
                            (object.yaw.to_degrees(), object.pitch.to_degrees())
                        };
                        let entity = state.entities.spawn_packet(
                            &mut state.world,
                            object.entity_id,
                            TrackedKind::Type(kind),
                            None,
                            [object.x, object.y, object.z],
                            yaw,
                            pitch,
                            velocity,
                            mc_world::entity::Metadata::new(),
                        );
                        // java: `data` is a per-type **constructor argument** — the falling block's
                        // `Block.getStateById(data & 65535)`, the item frame's
                        // `EnumFacing.getHorizontal(data)`, the splash potion's damage value, the
                        // arrow's shooter id. It was decoded off the wire and then dropped here.
                        if is_fireball_spawn(kind) {
                            // These branches zero `data` before the ordinary setVelocity guard, so
                            // `velocity` above correctly remains zero. Their original packet speeds
                            // initialized EntityFireball.acceleration* in the constructor first.
                            let _ = state
                                .world
                                .ecs
                                .insert_one(entity, fireball_spawn_acceleration(object.velocity));
                        }
                        // These constructor arguments are distinct from the generic object-data
                        // payload retained below: later render/simulation consumers need typed
                        // semantics, not a packet type switch of their own. They run before the
                        // Java branches zero `data` for the velocity guard.
                        match object.object_type {
                            // java: `new EntityItemFrame(..., EnumFacing.getHorizontal(data))`.
                            // `Facing::from_horizontal` preserves Java's signed remainder/absolute
                            // wrapping for negative data as well as its ordinary 0..=3 inputs.
                            71 => {
                                let _ = state.world.ecs.insert_one(
                                    entity,
                                    mc_world::ItemFrameFacing(mc_math::Facing::from_horizontal(
                                        object.data,
                                    )),
                                );
                            }
                            // java: `new EntityPotion(..., data)`, which builds an ItemStack with
                            // that raw damage value before its branch clears the packet data.
                            73 => {
                                let _ = state
                                    .world
                                    .ecs
                                    .insert_one(entity, mc_world::SplashPotionDamage(object.data));
                            }
                            // java: after the arrow is registered, `data > 0` looks it up and writes
                            // `shootingEntity` only when that object is an EntityLivingBase. A
                            // server uses the arrow's own id for the no-owner case, and no late
                            // resolution occurs if the owner is missing at this instant.
                            60 if object.data > 0 => {
                                let owner_is_living = object.data == state.local.entity_id
                                    || state
                                        .entities
                                        .get(object.data)
                                        .is_some_and(|owner| owner.kind.is_living());
                                if owner_is_living {
                                    let _ = state
                                        .world
                                        .ecs
                                        .insert_one(entity, mc_world::ArrowShooter(object.data));
                                }
                            }
                            _ => {}
                        }
                        let _ = state
                            .world
                            .ecs
                            .insert_one(entity, crate::app::entrender::ObjectData(object.data));
                        events.push(ClientEvent::EntitySpawned {
                            network_id: object.entity_id,
                            entity,
                            kind: TrackedKind::Type(kind),
                        });
                    }
                    None => events.push(ClientEvent::EntitySpawnIgnored {
                        network_id: object.entity_id,
                        reason: "object type has no EntityList counterpart",
                    }),
                }
            }
        }
        // java: handleSpawnExperienceOrb (:447)
        S2CPlay::SpawnExperienceOrb(orb) => {
            // `Math.random()` has no replayable JVM sequence. Seeding the surrogate from the
            // network id keeps every spawn's constructor motion deterministic without borrowing
            // randomness from another entity or changing packet position/XP state.
            let mut random = JavaRandom::new(i64::from(orb.entity_id));
            let spawn_motion = xp_orb_spawn_motion(&mut random);
            let entity = state.entities.spawn_packet(
                &mut state.world,
                orb.entity_id,
                TrackedKind::Type(EntityType::XPOrb),
                None,
                [orb.x, orb.y, orb.z],
                0.0,
                0.0,
                spawn_motion,
                mc_world::entity::Metadata::new(),
            );
            // java: `new EntityXPOrb(world, x, y, z, packetIn.getXPValue())` (`:447`). The value
            // decides `getTextureByXP`'s atlas cell and appears nowhere else on the wire, so
            // dropping it left every orb the same size.
            let _ = state.world.ecs.insert_one(
                entity,
                crate::app::entrender::XpValue(i32::from(orb.xp_value)),
            );
            events.push(ClientEvent::EntitySpawned {
                network_id: orb.entity_id,
                entity,
                kind: TrackedKind::Type(EntityType::XPOrb),
            });
        }
        // java: handleSpawnPainting (:491). The wire carries a BlockPos, and
        // `EntityHanging.updateFacingWithBoundingBox` centres the entity in it.
        S2CPlay::SpawnPainting(painting) => {
            // java: `new EntityPainting(world, blockpos, EnumFacing.getHorizontal(dir), title)`
            // resolves the title against `EnumArt` and then runs
            // `EntityHanging.updateFacingWithBoundingBox` (`EntityHanging.java:41-90`), which is
            // where the entity's real position comes from: the block centre, `0.46875` back off
            // the wall along the facing, plus the even-size centring offsets. This used to be a
            // bare block centre with `title` and `facing` thrown away, which put every painting
            // half-embedded in its wall, at the wrong height for tall art and half a block
            // sideways for wide art.
            // `EntityPainting` owns the RNG draw, not the packet title lookup. `ClientState`
            // does not expose the JVM-created `Entity.rand` seed at this boundary, so use the
            // same per-entity JavaRandom seam as the other constructor surrogates and retain
            // the exact residual: candidate ordering/bounds/consumption are faithful, but the
            // entropy-seeded JVM index cannot be reproduced byte-for-byte.
            let mut painting_random = JavaRandom::new(i64::from(painting.entity_id));
            let hanging_entities = existing_hanging_boxes(state, painting.entity_id);
            let art = mc_render::entity::objects::Art::resolve_for_spawn(
                &state.world,
                painting.position,
                painting.facing,
                &painting.title,
                &mut painting_random,
                &hanging_entities,
            );
            let position = mc_render::entity::objects::hanging_position(
                [
                    painting.position.x,
                    painting.position.y,
                    painting.position.z,
                ],
                painting.facing,
                art.size_x,
                art.size_y,
            );
            let fixed = [
                (position[0] * FIXED_POINT_SCALE).round() as i32,
                (position[1] * FIXED_POINT_SCALE).round() as i32,
                (position[2] * FIXED_POINT_SCALE).round() as i32,
            ];
            let entity = state.entities.spawn_packet(
                &mut state.world,
                painting.entity_id,
                TrackedKind::Type(EntityType::Painting),
                None,
                fixed,
                // java: `rotationYaw = facingDirection.getHorizontalIndex() * 90`.
                mc_render::entity::objects::hanging_yaw(painting.facing),
                0.0,
                DVec3::ZERO,
                mc_world::entity::Metadata::new(),
            );
            // Keep the constructor's selected stream attached to this entity. Paintings do not
            // currently consume it after construction, but preserving the post-selection state
            // keeps the ownership seam correct if a client-local painting effect is added later.
            if let Some(random) = state.entities.particle_random_mut(painting.entity_id) {
                *random = painting_random;
            }
            let _ = state.world.ecs.insert_one(
                entity,
                crate::app::entrender::Hanging {
                    art,
                    facing: painting.facing,
                },
            );
            events.push(ClientEvent::EntitySpawned {
                network_id: painting.entity_id,
                entity,
                kind: TrackedKind::Type(EntityType::Painting),
            });
        }
        // java: handleSpawnGlobalEntity (:463) — lightning only, via addWeatherEffect.
        S2CPlay::SpawnGlobalEntity(global) => {
            if global.entity_type == 1 {
                events.push(ClientEvent::LightningBolt {
                    network_id: global.entity_id,
                    position: DVec3::new(
                        f64::from(global.x_fixed) / 32.0,
                        f64::from(global.y_fixed) / 32.0,
                        f64::from(global.z_fixed) / 32.0,
                    ),
                });
            } else {
                events.push(ClientEvent::EntitySpawnIgnored {
                    network_id: global.entity_id,
                    reason: "unknown global entity type",
                });
            }
        }

        // ---- entity updates ----
        // java: handleDestroyEntities (:654)
        S2CPlay::DestroyEntities(destroy) => {
            // `despawn` per id in order, walking the records once
            // for the whole list (one per id made a long list quadratic).
            let removed = state
                .entities
                .despawn_many(&mut state.world, &destroy.entity_ids);
            for (network_id, removed) in destroy.entity_ids.into_iter().zip(removed) {
                if removed {
                    events.push(ClientEvent::EntityRemoved { network_id });
                }
            }
        }
        // java: handleEntityMovement (:613) — S14/S15/S16/S17 all land here, and all four end
        // `entity.onGround = packetIn.getOnGround();` (`:629`). That flag is the only ground
        // signal a client ever has for a remote entity; dropping it left `EntitySlime`'s squish
        // edges and `EntityChicken`'s wing flap with no input at all.
        // F-P0-client.network-02-17 / F-P1-entity-08-3 / F-P1-entity.monster-07-1.
        S2CPlay::Entity(entity) => {
            if is_local_entity(state, entity.entity_id) {
                state
                    .local
                    .apply_relative_move_packet([0, 0, 0], None, false);
            } else {
                state.entities.apply_relative_move_grounded(
                    &mut state.world,
                    entity.entity_id,
                    [0, 0, 0],
                    None,
                    // java: `S14PacketEntity.readPacketData` (`:32`) reads **only** the entity id, so
                    // `onGround` keeps its Java default of `false` — and `handleEntityMovement`
                    // assigns it anyway. So the bare `0x14` really does clear the flag. It is
                    // unreachable against a vanilla server (nothing in 1.8.9 constructs the base
                    // class; every producer is one of the three subclasses), which is why this is a
                    // literal rather than a wire field.
                    Some(false),
                );
            }
        }
        S2CPlay::EntityRelMove(movement) => {
            if is_local_entity(state, movement.entity_id) {
                state.local.apply_relative_move_packet(
                    [movement.delta_x, movement.delta_y, movement.delta_z],
                    None,
                    movement.on_ground,
                );
            } else {
                state.entities.apply_relative_move_grounded(
                    &mut state.world,
                    movement.entity_id,
                    [movement.delta_x, movement.delta_y, movement.delta_z],
                    None,
                    Some(movement.on_ground),
                );
            }
        }
        S2CPlay::EntityLook(look) => {
            if is_local_entity(state, look.entity_id) {
                state.local.apply_relative_move_packet(
                    [0, 0, 0],
                    Some((look.yaw.to_degrees(), look.pitch.to_degrees())),
                    look.on_ground,
                );
            } else {
                state.entities.apply_relative_move_grounded(
                    &mut state.world,
                    look.entity_id,
                    [0, 0, 0],
                    Some((look.yaw.to_degrees(), look.pitch.to_degrees())),
                    Some(look.on_ground),
                );
            }
        }
        S2CPlay::EntityLookAndRelMove(movement) => {
            if is_local_entity(state, movement.entity_id) {
                state.local.apply_relative_move_packet(
                    [movement.delta_x, movement.delta_y, movement.delta_z],
                    Some((movement.yaw.to_degrees(), movement.pitch.to_degrees())),
                    movement.on_ground,
                );
            } else {
                state.entities.apply_relative_move_grounded(
                    &mut state.world,
                    movement.entity_id,
                    [movement.delta_x, movement.delta_y, movement.delta_z],
                    Some((movement.yaw.to_degrees(), movement.pitch.to_degrees())),
                    Some(movement.on_ground),
                );
            }
        }
        // java: handleEntityTeleport (:566), which also ends `entity.onGround = getOnGround()`.
        S2CPlay::EntityTeleport(teleport) => {
            if is_local_entity(state, teleport.entity_id) {
                state.local.apply_teleport_packet(
                    [teleport.x, teleport.y, teleport.z],
                    teleport.yaw.to_degrees(),
                    teleport.pitch.to_degrees(),
                    teleport.on_ground,
                );
            } else {
                state.entities.apply_teleport_grounded(
                    &mut state.world,
                    teleport.entity_id,
                    [teleport.x, teleport.y, teleport.z],
                    teleport.yaw.to_degrees(),
                    teleport.pitch.to_degrees(),
                    Some(teleport.on_ground),
                );
            }
        }
        // java: handleEntityHeadLook (:637)
        S2CPlay::EntityHeadLook(head) => {
            if is_local_entity(state, head.entity_id) {
                state.local.apply_head_look_packet(head.yaw.to_degrees());
            } else {
                state
                    .entities
                    .apply_head_look(head.entity_id, head.yaw.to_degrees());
            }
        }
        // java: handleEntityVelocity (:501)
        S2CPlay::EntityVelocity(velocity) => {
            // `handleEntityVelocity` calls `setVelocity` on **whatever entity the
            // id names**, and the local player is not in `EntityTracker` — its motion lives in
            // `mc_physics::Predictor`. Without this the client's own knockback packet took
            // `apply_velocity`'s `return false` path every time, so every PvP hit became a
            // rubber-band. `Session::react_to` assigns it at this packet; see `LocalPlayer`.
            if velocity.entity_id == state.local.entity_id {
                state.local.pending_velocity = Some(mc_math::DVec3::new(
                    f64::from(velocity.motion_x) / 8000.0,
                    f64::from(velocity.motion_y) / 8000.0,
                    f64::from(velocity.motion_z) / 8000.0,
                ));
            }
            state.entities.apply_velocity(
                &mut state.world,
                velocity.entity_id,
                [velocity.motion_x, velocity.motion_y, velocity.motion_z],
            );
        }
        // java: handleEntityMetadata (:516)
        S2CPlay::EntityMetadata(metadata) => {
            if metadata.entity_id == state.local.entity_id {
                crate::sync::entities::merge_registered_metadata(
                    TrackedKind::Player,
                    &mut state.local.metadata,
                    &metadata.metadata,
                );
            } else {
                state.entities.apply_metadata(
                    &mut state.world,
                    metadata.entity_id,
                    &metadata.metadata,
                );
            }
        }
        // java: handleEntityStatus (:1024). The local player is deliberately not in
        // `EntityTracker`, so its EntityLivingBase status state must be applied here before the
        // event reaches the remote-entity render path.
        S2CPlay::EntityStatus(status) => {
            if status.entity_id == state.local.entity_id {
                state.local.apply_status(status.opcode);
            } else {
                // Java resolves the entity and invokes its polymorphic status body while this
                // packet is handled. Apply it before a later packet in the same drained batch
                // can destroy or reuse the network id; ClientEvent remains notification-only.
                state
                    .entities
                    .apply_status_in_world(&state.world, status.entity_id, status.opcode);
            }
            events.push(ClientEvent::EntityStatus {
                network_id: status.entity_id,
                opcode: status.opcode,
            });
        }
        // java: handleAnimation (:867). Type 4/5 constructs EntityParticleEmitter immediately
        // while the source object is still resolved. Apply the remote half at the packet
        // boundary so a later destroy/reuse in the same poll batch cannot retarget it; the app
        // driver handles the session-owned local player, which is intentionally outside the
        // tracker.
        S2CPlay::Animation(animation) => {
            if animation.entity_id != state.local.entity_id {
                state.entities.apply_crit_animation_from_packet(
                    &state.world,
                    animation.entity_id,
                    animation.animation_type,
                );
            }
            events.push(ClientEvent::EntityAnimation {
                network_id: animation.entity_id,
                animation: animation.animation_type,
            });
        }
        // java: handleEntityEquipment (:1297)
        S2CPlay::EntityEquipment(equipment) => {
            if is_local_entity(state, equipment.entity_id) {
                // The local player is a real `WorldClient` entity in Java, but deliberately has
                // no `EntityTracker` generation here. Apply valid slots before publishing the
                // event so a same-batch consumer observes Java's immediate mutation.
                let _ = state
                    .local
                    .apply_equipment_packet(equipment.equipment_slot, equipment.item.clone());
            }
            events.push(ClientEvent::EntityEquipment {
                network_id: equipment.entity_id,
                slot: equipment.equipment_slot,
                item: equipment.item,
            });
        }
        // java: handleEntityEffect (:1502). Apply remote state at wire-dispatch time so a later
        // destroy/reuse in the same drained packet batch cannot retarget the effect by numeric id.
        S2CPlay::EntityEffect(effect) => {
            if effect.entity_id != state.local.entity_id {
                state
                    .entities
                    .apply_effect(effect.entity_id, effect.effect_id, effect.amplifier);
            }
            events.push(ClientEvent::EntityEffectAdded(effect));
        }
        // java: handleRemoveEntityEffect (:1606)
        S2CPlay::RemoveEntityEffect(effect) => {
            if effect.entity_id != state.local.entity_id {
                state
                    .entities
                    .remove_effect(effect.entity_id, effect.effect_id as i8);
            }
            events.push(ClientEvent::EntityEffectRemoved {
                network_id: effect.entity_id,
                effect_id: effect.effect_id,
            });
        }
        // java: handleEntityAttach (:966)
        S2CPlay::EntityAttach(attach) => {
            // Remote entities live in `EntityTracker`; the local player remains session-owned
            // and therefore takes the same event through `Session::react_to` below.
            state
                .entities
                .apply_entity_attach(&mut state.world, &attach);
            events.push(ClientEvent::EntityAttached(attach));
        }
        // java: handleEntityProperties (:2051)
        S2CPlay::EntityProperties(properties) => {
            // `handleEntityProperties` ignores a packet whose entity is absent. A known remote
            // living entity receives every name (including a modded/unknown name): the Java
            // fallback is itself an attribute registration path and the resulting snapshot must
            // remain available to a matching client-side consumer. Existing non-living entities
            // still throw before any mutation. This check also prevents a delayed packet for a
            // destroyed id from becoming a stale generation event in the application layer.
            if properties.entity_id != state.local.entity_id {
                let Some(tracked) = state.entities.get(properties.entity_id) else {
                    return Ok(());
                };
                if !tracked.kind.is_living() {
                    return Err(ClientError::EntityPropertiesForNonLiving {
                        entity_id: properties.entity_id,
                    });
                }
            }
            events.push(ClientEvent::EntityProperties(Box::new(properties)));
        }
        // java: handleEntityNBT (:1805). `Entity.clientUpdateEntityNBT` is an empty base hook
        // with no vanilla overrides in 1.8.9, so a resolved or unresolved id has no client-side
        // state or event effect. Keep decoding in mc-protocol, but deliberately discard the
        // decoded packet at this boundary.
        S2CPlay::UpdateEntityNBT(_nbt) => {}
        // java: handleUseBed (:903)
        S2CPlay::UseBed(bed) => events.push(ClientEvent::EnteredBed {
            network_id: bed.player_id,
            bed_position: bed.bed_position,
        }),
        // java: handleCollectItem (:819-844). The pickup sound/particle stay an event for the
        // driver, but `WorldClient.removeEntityFromWorld` happens here immediately rather than
        // waiting for a later S13 destroy packet.
        S2CPlay::CollectItem(collect) => {
            // The Java handler reads the source entity's kind/position/stack and resolves the
            // pickup destination before removing it. Keep the same pre-despawn seam available to
            // the application; an unknown source id intentionally records nothing.
            state
                .entities
                .capture_item_collected_source(&state.world, collect.collected_entity_id);
            if state
                .entities
                .despawn(&mut state.world, collect.collected_entity_id)
            {
                events.push(ClientEvent::EntityRemoved {
                    network_id: collect.collected_entity_id,
                });
            }
            events.push(ClientEvent::ItemCollected {
                collected: collect.collected_entity_id,
                collector: collect.collector_entity_id,
            });
        }

        // ---- chunks and blocks ----
        // java: handleChunkData (:747). Off-thread-unpacked twin: `apply_prebuilt_chunks`.
        S2CPlay::ChunkData(chunk) => {
            let position = ChunkPos::new(chunk.chunk_x, chunk.chunk_z);
            let unloading = chunk.ground_up_continuous && chunk.primary_bit_mask == 0;
            mc_core::profile_scope!("sync.apply_chunk");
            state.world.apply_chunk_data(
                position,
                chunk.ground_up_continuous,
                chunk.primary_bit_mask,
                &chunk.data,
            )?;
            if unloading {
                for network_id in state.entities.unload_in_chunk(&state.world, position) {
                    events.push(ClientEvent::EntityRemoved { network_id });
                }
            } else {
                // `WorldClient.spawnEntityInWorld` queues packet-created objects until the
                // resident column exists. A full/partial chunk payload is the retry boundary;
                // do this after `apply_chunk_data` so the same generation enters the loaded-list
                // order before the ChunkLoaded notification is observed.
                let _ = state.entities.activate_chunk(&state.world, position);
            }
            events.push(if unloading {
                ClientEvent::ChunkUnloaded(position)
            } else {
                ClientEvent::ChunkLoaded(position)
            });
        }
        // java: handleMapChunkBulk (:1337). The packet's own `has_sky_light` flag is
        // authoritative for the dimension — hazard H6. Twin: `apply_prebuilt_chunks`.
        S2CPlay::MapChunkBulk(bulk) => {
            let entries: Vec<BulkChunkEntry<'_>> = bulk
                .columns
                .iter()
                .map(|column| BulkChunkEntry {
                    chunk_x: column.chunk_x,
                    chunk_z: column.chunk_z,
                    primary_bitmask: column.primary_bit_mask,
                    data: &column.data,
                })
                .collect();
            mc_core::profile_scope!("sync.apply_chunk");
            state.world.apply_chunk_bulk(bulk.has_sky_light, &entries)?;
            for column in &bulk.columns {
                let position = ChunkPos::new(column.chunk_x, column.chunk_z);
                let _ = state.entities.activate_chunk(&state.world, position);
                events.push(ClientEvent::ChunkLoaded(position));
            }
        }
        // java: handleBlockChange (:776)
        S2CPlay::BlockChange(change) => {
            // java: invalidateRegionAndSetBlock(pos, null) throws inside setBlockState for a null
            // state, so the old block stays (Util.runTask logs it).
            let new_state = registered_block_state(change.block_state_id)
                .ok_or_else(|| unregistered_block_state(change.block_state_id))?;
            state.world.set_block(change.position, new_state);
        }
        // java: handleMultiBlockChange (:734)
        S2CPlay::MultiBlockChange(change) => {
            let chunk = ChunkPos::new(change.chunk_x, change.chunk_z);
            // java: the entries are set in order, so the first null state aborts the rest of
            // the packet with the earlier entries already applied.
            let mut unregistered = None;
            let changes: Vec<(u16, BlockState)> = change
                .changes
                .iter()
                .map_while(|entry| {
                    let Some(new_state) = registered_block_state(entry.block_state_id) else {
                        unregistered = Some(entry.block_state_id);
                        return None;
                    };
                    // Re-cram into the wire's `x<<12 | z<<8 | y` short, which is the shape
                    // `mc_world::World::apply_multi_block_change` takes.
                    let crammed = (u16::from(entry.rel_x & 0xF) << 12)
                        | (u16::from(entry.rel_z & 0xF) << 8)
                        | u16::from(entry.y);
                    Some((crammed, new_state))
                })
                .collect();
            state.world.apply_multi_block_change(chunk, &changes);
            if let Some(id) = unregistered {
                return Err(unregistered_block_state(id));
            }
        }
        // java: handleExplosion (:1078) — `Explosion.doExplosionB` sets every affected block to
        // air. The origin is `(int)posX`, a **truncation**, not a floor: negative coordinates
        // round towards zero here, and copying `floor` instead misplaces the whole pattern.
        S2CPlay::Explosion(explosion) => {
            for position in explosion_affected_positions(&explosion) {
                state.world.set_block(position, BlockState::AIR);
            }
            events.push(ClientEvent::Explosion(Box::new(explosion)));
        }
        // java: handleBlockAction (:1322)
        S2CPlay::BlockAction(action) => events.push(ClientEvent::BlockAction(action)),
        // java: handleBlockBreakAnim (:1331)
        S2CPlay::BlockBreakAnimation(animation) => {
            events.push(ClientEvent::BlockBreakProgress {
                breaker_id: animation.breaker_id,
                position: animation.position,
                progress: animation.progress,
            });
        }
        // java: handleUpdateTileEntity (:1267)
        S2CPlay::UpdateTileEntity(tile) => events.push(ClientEvent::TileEntityUpdated {
            position: tile.position,
            action: tile.action,
            nbt: tile.nbt.map(Box::new),
        }),
        // java: handleUpdateSign (:1234)
        S2CPlay::UpdateSign(sign) => events.push(ClientEvent::SignUpdated {
            position: sign.position,
            lines: Box::new(sign.lines),
        }),
        // java: handleSignEditorOpen (:1216)
        S2CPlay::SignEditorOpen(open) => {
            events.push(ClientEvent::OpenSignEditor(open.position));
        }

        // ---- effects, sound, particles: S18/S19 ----
        // java: handleEffect (:1441)
        S2CPlay::Effect(effect) => events.push(ClientEvent::Effect(effect)),
        // java: handleSoundEffect (:1695)
        S2CPlay::SoundEffect(sound) => events.push(ClientEvent::SoundEffect(Box::new(sound))),
        // java: handleParticles (:2003)
        S2CPlay::Particles(particles) => {
            events.push(ClientEvent::Particles(Box::new(particles)));
        }

        // ---- tab list ----
        // java: handlePlayerListItem (:1618)
        S2CPlay::PlayerListItem(item) => state.players.apply(&item)?,
        // java: handlePlayerListHeaderFooter (:1600)
        S2CPlay::PlayerListHeaderFooter(header_footer) => {
            state
                .players
                .set_header_footer(&header_footer.header, &header_footer.footer)?;
            events.push(ClientEvent::PlayerListHeaderFooter);
        }

        // ---- containers: S17 ----
        S2CPlay::OpenWindow(mut window) => {
            if matches!(
                ChatComponent::from_json_nullable(&window.window_title),
                Ok(None)
            ) {
                open_window_with_null_title(&mut window)?;
            }
            events.push(window_event(WindowEvent::Open(window)));
        }
        S2CPlay::CloseWindow(window) => {
            events.push(window_event(WindowEvent::Close {
                window_id: window.window_id,
            }));
        }
        S2CPlay::SetSlot(slot) => events.push(window_event(WindowEvent::SetSlot {
            window_id: slot.window_id,
            slot: slot.slot,
            item: slot.item,
        })),
        S2CPlay::WindowItems(items) => events.push(window_event(WindowEvent::Items {
            window_id: items.window_id,
            items: items.items,
        })),
        S2CPlay::WindowProperty(property) => events.push(window_event(WindowEvent::Property {
            window_id: property.window_id,
            index: property.var_index,
            value: property.var_value,
        })),
        S2CPlay::ConfirmTransaction(transaction) => {
            events.push(window_event(WindowEvent::ConfirmTransaction {
                window_id: transaction.window_id,
                action_number: transaction.action_number,
                accepted: transaction.accepted,
            }));
        }

        // ---- scoreboard: S16 ----
        S2CPlay::ScoreboardObjective(objective) => {
            events.push(scoreboard_event(ScoreboardEvent::Objective(objective)));
        }
        S2CPlay::UpdateScore(score) => {
            events.push(scoreboard_event(ScoreboardEvent::Score(score)));
        }
        S2CPlay::DisplayScoreboard(display) => {
            events.push(scoreboard_event(ScoreboardEvent::Display(display)));
        }
        S2CPlay::Teams(teams) => {
            events.push(scoreboard_event(ScoreboardEvent::Teams(Box::new(teams))));
        }

        // ---- everything else ----
        // java: handleTitle (:1565)
        S2CPlay::Title(title) => events.push(ClientEvent::Title(Box::new(title))),
        // java: handleTabComplete (:1683)
        S2CPlay::TabComplete(complete) => {
            events.push(ClientEvent::TabComplete(complete.matches));
        }
        // java: handleCustomPayload (:1822)
        S2CPlay::PluginMessage(message) => events.push(ClientEvent::PluginMessage {
            channel: message.channel,
            data: message.data,
        }),
        // java: handleStatistics (:1458) — Client::react folds this S37 snapshot into the
        // statistics screens and the globally-owned achievement toast.
        S2CPlay::Statistics(statistics) => {
            events.push(ClientEvent::Statistics(statistics.entries));
        }
        // java: handleMaps (:1433)
        S2CPlay::Maps(maps) => {
            // `NetHandlerPlayClient.handleMaps` first loads the persistent client-side
            // `map_<id>` record, applies the packet to it, and only then asks the renderer to
            // refresh its dynamic texture.  Keeping this mutation in dispatch is important:
            // `ClientEvent::Maps` is intentionally a notification for older callers, while the
            // held-map and item-frame render paths read `state.world.map_data` directly.
            //
            // `World::apply_map_packet` validates the rectangle before replacing a record.  A
            // malformed rectangle cannot be represented by Java's fixed 128x128 `MapData`; the
            // protocol layer has already accepted the packet, so preserve the event stream but
            // leave the previous renderable snapshot untouched rather than panicking or
            // partially mutating the map.
            let _ = state.world.apply_map_packet(&maps);
            events.push(ClientEvent::Maps(Box::new(maps)));
        }
        // java: handleResourcePack (:1701) — the caller answers with C19PacketResourcePackStatus.
        S2CPlay::ResourcePackSend(pack) => events.push(ClientEvent::ResourcePack {
            url: pack.url,
            hash: pack.hash,
        }),
    }
    Ok(())
}

fn window_event(event: WindowEvent) -> ClientEvent {
    ClientEvent::Window(Box::new(event))
}

/// `handleOpenWindow` (`NetHandlerPlayClient.java:1092-1128`) with a title Gson resolved to null.
///
/// The decoder accepts the null (`readChatComponent` never throws for it); each branch then
/// decides what it means:
/// - `minecraft:villager`: `NpcMerchant.getDisplayName` falls back to the translated
///   `entity.Villager.name`, so the trade screen opens normally.
/// - a slot-less type (`LocalBlockIntercommunication`): crafting table and anvil draw fixed
///   `I18n` titles and never read it. (`GuiEnchantment` would dereference it while drawing and
///   crash vanilla's render loop; an empty title is the non-crashing equivalent.)
/// - every `InventoryBasic` path (`minecraft:container`, a horse chest and `ContainerLocalMenu`)
///   calls `title.getUnformattedText()` in its constructor. The NullPointerException aborts the
///   scheduled handler before any screen or window id changes, and `Util.runTask` logs it.
fn open_window_with_null_title(window: &mut s2c::OpenWindow) -> Result<(), ClientError> {
    match window.inventory_type.as_str() {
        "minecraft:villager" => {
            window.window_title = r#"{"translate":"entity.Villager.name"}"#.to_owned();
            Ok(())
        }
        "minecraft:container" | "EntityHorse" => Err(null_window_title(window)),
        _ if window.slot_count == 0 => {
            window.window_title = r#"{"text":""}"#.to_owned();
            Ok(())
        }
        _ => Err(null_window_title(window)),
    }
}

fn null_window_title(window: &s2c::OpenWindow) -> ClientError {
    ClientError::Chat(mc_ui::UiError::Component(format!(
        "null title for window {} ({}): InventoryBasic dereferences it",
        window.window_id, window.inventory_type
    )))
}

fn scoreboard_event(event: ScoreboardEvent) -> ClientEvent {
    ClientEvent::Scoreboard(Box::new(event))
}

/// java: `EnumDifficulty.getDifficultyEnum` clamps with `id % 4`, so no id is ever invalid.
fn difficulty_from_id(id: u8) -> s2c::ServerDifficultyLevel {
    match id % 4 {
        0 => s2c::ServerDifficultyLevel::Peaceful,
        1 => s2c::ServerDifficultyLevel::Easy,
        2 => s2c::ServerDifficultyLevel::Normal,
        _ => s2c::ServerDifficultyLevel::Hard,
    }
}

/// Resolve the session-owned player only after a JoinGame has installed its identity.
///
/// `LocalPlayer::default` uses zero as a pre-join sentinel, but Java does not reserve entity id
/// zero. `entity_id_known` distinguishes those cases; the nonzero fallback keeps direct state
/// fixtures that seed a local id without replaying JoinGame compatible with the live seam.
#[inline]
fn is_local_entity(state: &ClientState, network_id: i32) -> bool {
    (state.local.entity_id_known || state.local.entity_id != 0)
        && network_id == state.local.entity_id
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_protocol::{Angle, Packet, PacketBuffer};

    #[test]
    fn vanilla_s20_attribute_names_and_descriptions_are_the_only_registered_keys() {
        for key in [
            "generic.maxHealth",
            "MAX HEALTH",
            "generic.followRange",
            "Follow Range",
            "generic.knockbackResistance",
            "knockback resistance",
            "GENERIC.MOVEMENTSPEED",
            "Movement Speed",
            "generic.attackDamage",
            "horse.jumpStrength",
            "Jump Strength",
            "zombie.spawnReinforcements",
            "Spawn Reinforcements Chance",
        ] {
            assert!(is_registered_vanilla_attribute(key), "rejected {key}");
        }
        assert!(!is_registered_vanilla_attribute("unknown.parityProbe"));
        assert!(!is_registered_vanilla_attribute(
            "generic.movementSpeed.extra"
        ));
    }

    fn drain(state: &mut ClientState, packet: S2CPlay) -> Vec<ClientEvent> {
        let mut events = Vec::new();
        apply(state, packet, &mut events).expect("apply");
        events
    }

    fn join() -> S2CPlay {
        S2CPlay::JoinGame(s2c::JoinGame {
            entity_id: 77,
            hardcore_mode: false,
            game_mode: 1,
            dimension: 0,
            difficulty: 2,
            max_players: 20,
            level_type: "flat".into(),
            reduced_debug_info: false,
        })
    }

    /// java: `WorldProvider.getProviderForDimension` is null for
    /// any id but -1/0/1 and `new WorldClient` throws on it. A dimension-changing S07 then
    /// leaves only `doneLoadingTerrain` cleared, and an S01 changes nothing. Rust used to treat
    /// every unknown id as the Overworld and rebuild the world around the player.
    #[test]
    fn an_unknown_dimension_throws_in_new_world_client_like_java() {
        let respawn = |dimension: i32| {
            S2CPlay::Respawn(s2c::Respawn {
                dimension,
                difficulty: 1,
                game_mode: 0,
                level_type: "default".into(),
            })
        };
        let mut state = ClientState::new();
        drain(&mut state, join());
        state.world.load_empty_chunk(ChunkPos::new(0, 0));
        for dimension in [2, -2, i32::MAX, i32::MIN] {
            state.local.done_loading_terrain = true;
            let mut events = Vec::new();
            assert!(
                apply(&mut state, respawn(dimension), &mut events).is_err(),
                "S07 {dimension}"
            );
            assert!(events.is_empty(), "S07 {dimension}");
            assert_eq!(
                state.local.dimension, 0,
                "S07 {dimension}: same WorldClient"
            );
            assert!(state.world.chunk(ChunkPos::new(0, 0)).is_some());
            assert_eq!(
                state.local.game_mode,
                s2c::GameMode::Creative,
                "setGameType skipped"
            );
            assert!(
                !state.local.done_loading_terrain,
                "cleared before the throw"
            );
        }

        let mut bogus_join = join();
        if let S2CPlay::JoinGame(packet) = &mut bogus_join {
            packet.dimension = 7;
            packet.entity_id = 99;
        }
        let mut events = Vec::new();
        assert!(apply(&mut state, bogus_join, &mut events).is_err());
        assert!(events.is_empty());
        assert_eq!(state.local.entity_id, 77);
        assert!(state.world.chunk(ChunkPos::new(0, 0)).is_some());

        drain(&mut state, respawn(-1));
        assert_eq!(
            state.local.dimension, -1,
            "a provider-backed dimension still loads"
        );
    }

    /// java: `handleChat` with a Gson-null component. Chat and system
    /// lines reach `GuiNewChat.setChatLine`, whose `GuiUtilRenderComponents.splitText(null)`
    /// throws in `Lists.newArrayList` before a line is added; the action bar's
    /// `setRecordPlaying(component.getUnformattedText(), ...)` throws before `recordPlaying`
    /// changes. Util.runTask logs either one and the connection carries on, so the handler is an
    /// error with no event, never a disconnect (the net task forwards it: see task.rs).
    #[test]
    fn a_gson_null_chat_component_is_a_logged_handler_error_with_no_effect() {
        let mut state = ClientState::new();
        for position in [0, 1, 2] {
            for null_json in ["null", "", "[]"] {
                let mut events = Vec::new();
                let result = apply(
                    &mut state,
                    S2CPlay::Chat(s2c::Chat {
                        component: null_json.to_owned(),
                        position,
                    }),
                    &mut events,
                );
                assert!(result.is_err(), "position {position}, {null_json:?}");
                assert!(events.is_empty(), "position {position}, {null_json:?}");
            }
        }
    }

    /// `handleOpenWindow` with a title Gson resolved to null: every
    /// `InventoryBasic` branch throws before opening anything (logged, connection kept); the
    /// villager branch falls back to `entity.Villager.name`; slot-less block GUIs open.
    #[test]
    fn a_gson_null_window_title_follows_each_handle_open_window_branch() {
        let open = |inventory_type: &str, slot_count: u8| {
            S2CPlay::OpenWindow(s2c::OpenWindow {
                window_id: 3,
                inventory_type: inventory_type.to_owned(),
                window_title: "null".to_owned(),
                slot_count,
                entity_id: (inventory_type == "EntityHorse").then_some(9),
            })
        };
        let opened_title = |events: &[ClientEvent]| match events {
            [ClientEvent::Window(window)] => match &**window {
                WindowEvent::Open(packet) => packet.window_title.clone(),
                other => panic!("expected an open-window event, got {other:?}"),
            },
            other => panic!("expected one open-window event, got {other:?}"),
        };

        let mut state = ClientState::new();
        for (inventory_type, slot_count) in [
            ("minecraft:container", 27),
            ("minecraft:chest", 27),
            ("minecraft:furnace", 3),
            ("minecraft:beacon", 1),
            ("EntityHorse", 2),
        ] {
            let mut events = Vec::new();
            assert!(
                apply(&mut state, open(inventory_type, slot_count), &mut events).is_err(),
                "{inventory_type}: InventoryBasic dereferences the null title"
            );
            assert!(events.is_empty(), "{inventory_type} must not open a screen");
        }

        let villager = drain(&mut state, open("minecraft:villager", 3));
        assert_eq!(
            opened_title(&villager),
            r#"{"translate":"entity.Villager.name"}"#
        );
        for slotless in ["minecraft:crafting_table", "minecraft:anvil"] {
            let events = drain(&mut state, open(slotless, 0));
            assert_eq!(opened_title(&events), r#"{"text":""}"#, "{slotless}");
        }

        // A real title is untouched.
        let mut chest = open("minecraft:chest", 27);
        if let S2CPlay::OpenWindow(window) = &mut chest {
            window.window_title = r#"{"text":"Shop"}"#.to_owned();
        }
        assert_eq!(
            opened_title(&drain(&mut state, chest)),
            r#"{"text":"Shop"}"#
        );
    }

    #[test]
    fn local_entity_metadata_merges_into_the_canonical_player_watcher() {
        let mut state = ClientState::new();
        drain(&mut state, join());
        assert_eq!(state.local.entity_id, 77);
        assert!(
            state.entities.get(77).is_none(),
            "local player stays out of remote tracker"
        );

        drain(
            &mut state,
            S2CPlay::EntityMetadata(s2c::EntityMetadata {
                entity_id: 77,
                metadata: vec![mc_protocol::MetadataEntry {
                    index: 17,
                    value: mc_protocol::MetadataValue::Float(6.5),
                }],
            }),
        );
        drain(
            &mut state,
            S2CPlay::EntityMetadata(s2c::EntityMetadata {
                entity_id: 77,
                metadata: vec![
                    mc_protocol::MetadataEntry {
                        index: 10,
                        value: mc_protocol::MetadataValue::Byte(0x7f),
                    },
                    mc_protocol::MetadataEntry {
                        index: 12,
                        value: mc_protocol::MetadataValue::Byte(1),
                    },
                ],
            }),
        );

        assert_eq!(
            state.local.metadata.get(17),
            Some(&mc_world::entity::MetadataValue::Float(6.5)),
            "a later partial S1C must retain omitted watcher indices"
        );
        assert_eq!(
            state.local.metadata.get(10),
            Some(&mc_world::entity::MetadataValue::Byte(0x7f))
        );
        assert_eq!(
            state.local.metadata.get(12),
            None,
            "a Zombie-owned slot is not registered on the local Player watcher"
        );
    }

    /// `S39PacketPlayerAbilities` has no `allowEdit` bit. It must therefore retain the value
    /// installed by the game-mode lifecycle instead of reconstructing default capabilities.
    #[test]
    fn player_abilities_preserve_game_modes_allow_edit() {
        let mut state = ClientState::new();
        state
            .local
            .abilities
            .configure_for_game_mode(s2c::GameMode::Adventure);
        assert!(
            !state.local.abilities.allow_edit,
            "setup: Adventure disables edits"
        );

        drain(
            &mut state,
            S2CPlay::PlayerAbilities(s2c::PlayerAbilities {
                flags: 0x0F,
                fly_speed: 0.25,
                walk_speed: 0.4,
            }),
        );

        assert!(
            !state.local.abilities.allow_edit,
            "S39 does not encode allowEdit"
        );
        assert!(
            state.local.abilities.flying,
            "control: S39 still replaces its own flags"
        );
        assert_eq!(
            (
                state.local.abilities.fly_speed,
                state.local.abilities.walk_speed
            ),
            (0.25, 0.4),
            "control: S39 still replaces its own speeds"
        );

        let mut survival = ClientState::new();
        drain(
            &mut survival,
            S2CPlay::PlayerAbilities(s2c::PlayerAbilities {
                flags: 0,
                fly_speed: 0.05,
                walk_speed: 0.1,
            }),
        );
        assert!(
            survival.local.abilities.allow_edit,
            "negative control: retaining state is not the same as forcing Adventure's false"
        );
    }

    fn spawn_object(
        entity_id: i32,
        object_type: i8,
        data: i32,
        velocity: Option<(i16, i16, i16)>,
    ) -> S2CPlay {
        S2CPlay::SpawnObject(s2c::SpawnObject {
            entity_id,
            object_type,
            x: 0,
            y: 0,
            z: 0,
            pitch: Angle(0),
            yaw: Angle(0),
            data,
            velocity,
        })
    }

    fn install_scoreboard_objective(state: &mut ClientState) {
        state
            .scoreboard
            .apply(&ScoreboardEvent::Objective(s2c::ScoreboardObjective {
                objective_name: "across_worlds".into(),
                mode: s2c::ScoreboardObjectiveMode::Create {
                    display_name: "Across worlds".into(),
                    render_type: s2c::ScoreboardRenderType::Integer,
                },
            }));
    }

    fn spawn_remote_zombie(state: &mut ClientState, entity_id: i32) {
        drain(
            state,
            S2CPlay::SpawnMob(s2c::SpawnMob {
                entity_id,
                entity_type: 54, // EntityList id: zombie.
                x: 0,
                y: 32 * 64,
                z: 0,
                yaw: Angle(0),
                pitch: Angle(0),
                head_pitch: Angle(0),
                velocity_x: 0,
                velocity_y: 0,
                velocity_z: 0,
                metadata: Vec::new(),
            }),
        );
    }

    /// F-P1-entity-06-6: remote potion state is an entity-generation concern, so S1D must mutate
    /// the resolved entity while its packet is dispatched. A later S13 and same-id S0F in the
    /// same network batch must construct a clean swing-duration owner.
    #[test]
    fn remote_s1d_swing_effects_do_not_cross_same_batch_id_reuse() {
        let mut state = ClientState::new();
        // SpawnMob is deferred until its column is resident; this test is about the live swing
        // consumer, so establish the same loaded-column boundary as a real client world.
        state.world.load_empty_chunk(ChunkPos::new(0, 0));
        spawn_remote_zombie(&mut state, 9);

        let events = drain(
            &mut state,
            S2CPlay::EntityEffect(s2c::EntityEffect {
                entity_id: 9,
                effect_id: 3,
                amplifier: 2,
                duration: 200,
                show_particles: true,
            }),
        );
        assert!(matches!(
            events.as_slice(),
            [ClientEvent::EntityEffectAdded(s2c::EntityEffect {
                entity_id: 9,
                effect_id: 3,
                amplifier: 2,
                ..
            })]
        ));
        assert!(state.entities.apply_swing(9));
        state.entities.tick_interpolation(&mut state.world);
        state.entities.tick_interpolation(&mut state.world);
        assert!(
            (state.entities.get(9).unwrap().render.swing_progress - 1.0 / 3.0).abs() < 1e-6,
            "the live remote entity must observe Haste III"
        );

        drain(
            &mut state,
            S2CPlay::DestroyEntities(s2c::DestroyEntities {
                entity_ids: vec![9],
            }),
        );
        spawn_remote_zombie(&mut state, 9);
        assert!(state.entities.apply_swing(9));
        state.entities.tick_interpolation(&mut state.world);
        state.entities.tick_interpolation(&mut state.world);
        assert!(
            (state.entities.get(9).unwrap().render.swing_progress - 1.0 / 6.0).abs() < 1e-6,
            "the replacement generation must use the ordinary six-tick swing"
        );
    }

    /// `handleSpawnMob` is the odd velocity handler: its division is in `float`, then the
    /// rounded result is widened for the entity's `double motionX/Y/Z` fields.  Exercise the
    /// actual signed-i16 packet body first so this cannot silently become a test of a hand-made
    /// `DVec3` rather than protocol-47's S0F fields.
    #[test]
    fn spawn_mob_velocity_widens_after_java_f32_division() {
        let packet = s2c::SpawnMob {
            entity_id: 91,
            entity_type: 54, // zombie
            x: 0,
            y: 0,
            z: 0,
            yaw: Angle(0),
            pitch: Angle(0),
            head_pitch: Angle(0),
            velocity_x: i16::MAX,
            velocity_y: i16::MIN,
            velocity_z: 1,
            metadata: Vec::new(),
        };
        let mut body = PacketBuffer::new();
        packet.write(&mut body);
        let decoded = s2c::SpawnMob::read(&mut PacketBuffer::from_slice(body.as_slice()))
            .expect("the signed S0F velocity words decode");
        assert_eq!(
            (decoded.velocity_x, decoded.velocity_y, decoded.velocity_z),
            (i16::MAX, i16::MIN, 1),
            "S0F carries three signed short velocity components"
        );

        let mut state = ClientState::new();
        drain(&mut state, S2CPlay::SpawnMob(decoded));
        let entity = state
            .entities
            .get(91)
            .expect("known mob was spawned")
            .entity;
        let motion = state
            .world
            .ecs
            .get::<&mc_world::entity::Velocity>(entity)
            .expect("living spawn owns motion")
            .0;

        assert_eq!(
            motion.x.to_bits(),
            f64::from(i16::MAX as f32 / 8000.0_f32).to_bits(),
            "Java narrows S0F's positive extreme before widening"
        );
        assert_eq!(
            motion.y.to_bits(),
            f64::from(i16::MIN as f32 / 8000.0_f32).to_bits(),
            "the same f32 division applies to the negative extreme"
        );
        assert_eq!(motion.z, f64::from(1.0_f32 / 8000.0_f32));

        assert_ne!(
            motion.x.to_bits(),
            (f64::from(i16::MAX) / VELOCITY_SCALE).to_bits(),
            "a direct f64 division is the tempting but non-vanilla implementation"
        );
        assert_ne!(
            motion.y.to_bits(),
            (f64::from(i16::MIN) / VELOCITY_SCALE).to_bits(),
            "negative values must not accidentally take the f64 path either"
        );
    }

    #[test]
    fn collect_item_removes_the_collected_entity_before_reporting_the_pickup() {
        let mut state = ClientState::new();
        let collected = 61;
        let collector = 62;
        spawn_remote_zombie(&mut state, collected);
        spawn_remote_zombie(&mut state, collector);

        let events = drain(
            &mut state,
            S2CPlay::CollectItem(s2c::CollectItem {
                collected_entity_id: collected,
                collector_entity_id: collector,
            }),
        );

        assert!(
            state.entities.get(collected).is_none(),
            "NetHandlerPlayClient removes the collected id in handleCollectItem, not on a later S13"
        );
        assert!(
            state.entities.get(collector).is_some(),
            "the collector is an animation destination, not the removal target"
        );
        assert_eq!(
            events,
            vec![
                ClientEvent::EntityRemoved {
                    network_id: collected,
                },
                ClientEvent::ItemCollected {
                    collected,
                    collector,
                },
            ],
            "the immediate world mutation precedes the driver-owned sound/particle event"
        );
        let source = state
            .entities
            .take_item_collected_source(collected)
            .expect("the source snapshot survives the immediate despawn");
        assert_eq!(source.kind, TrackedKind::Type(EntityType::Zombie));
        assert_eq!(source.position, DVec3::new(0.0, 64.0, 0.0));

        let missing_events = drain(
            &mut state,
            S2CPlay::CollectItem(s2c::CollectItem {
                collected_entity_id: -9,
                collector_entity_id: collector,
            }),
        );
        assert_eq!(
            missing_events,
            vec![ClientEvent::ItemCollected {
                collected: -9,
                collector,
            }],
            "a duplicate/unknown collected id is harmless but still gives the pickup consumer its packet"
        );
        assert!(
            state.entities.take_item_collected_source(-9).is_none(),
            "unknown source ids do not manufacture a render/audio producer"
        );
    }

    /// Java assigns `Entity.onGround = packetIn.getOnGround()` at the end of both
    /// `handleEntityMovement` and `handleEntityTeleport`.  The three S14 subclasses take the
    /// former path, even look-only S16, so all four wire shapes must reach the remote state.
    ///
    /// Negative control: deleting any `Some(packet.on_ground)` at the dispatch seam leaves the
    /// corresponding assertion at its previous value and fails.  The alternating values make a
    /// default `false`, or a one-way `true` latch, fail too.
    #[test]
    fn every_remote_movement_packet_propagates_its_ground_flag() {
        let mut state = ClientState::new();
        let id = 61;
        spawn_remote_zombie(&mut state, id);
        assert!(
            !state.entities.get(id).expect("spawned remote").on_ground,
            "Entity.onGround begins at Java's false default"
        );

        drain(
            &mut state,
            S2CPlay::EntityRelMove(s2c::EntityRelMove {
                entity_id: id,
                delta_x: 1,
                delta_y: 0,
                delta_z: 0,
                on_ground: true,
            }),
        );
        assert!(
            state.entities.get(id).unwrap().on_ground,
            "S15 relative move"
        );

        drain(
            &mut state,
            S2CPlay::EntityLook(s2c::EntityLook {
                entity_id: id,
                yaw: Angle(64),
                pitch: Angle(0),
                on_ground: false,
            }),
        );
        assert!(!state.entities.get(id).unwrap().on_ground, "S16 look-only");

        drain(
            &mut state,
            S2CPlay::EntityLookAndRelMove(s2c::EntityLookAndRelMove {
                entity_id: id,
                delta_x: 0,
                delta_y: 1,
                delta_z: 0,
                yaw: Angle(32),
                pitch: Angle(8),
                on_ground: true,
            }),
        );
        assert!(
            state.entities.get(id).unwrap().on_ground,
            "S17 move-and-look"
        );

        drain(
            &mut state,
            S2CPlay::EntityTeleport(s2c::EntityTeleport {
                entity_id: id,
                x: 32,
                y: 32 * 64,
                z: 0,
                yaw: Angle(0),
                pitch: Angle(0),
                on_ground: false,
            }),
        );
        assert!(!state.entities.get(id).unwrap().on_ground, "S18 teleport");
    }

    /// The local `EntityPlayerSP` is a real `WorldClient` entity in Java, but it intentionally
    /// does not have an `EntityTracker` generation in this port.  These are the packet edges that
    /// used to disappear at that seam: S04's five equipment slots, all S14 wire shapes, S18's
    /// fixed-point/dead-zone target, and S19's head-only yaw.  The remote and unknown ids at the
    /// end are negative controls for the routing predicate, not alternate local-player paths.
    #[test]
    fn local_entity_packets_update_session_state_while_remote_and_unknown_ids_stay_separate() {
        let mut pre_join = ClientState::new();
        drain(
            &mut pre_join,
            S2CPlay::EntityHeadLook(s2c::EntityHeadLook {
                entity_id: 0,
                yaw: Angle::from_degrees(75.0),
            }),
        );
        drain(
            &mut pre_join,
            S2CPlay::EntityEquipment(s2c::EntityEquipment {
                entity_id: 0,
                equipment_slot: 0,
                item: Some(mc_protocol::ItemStack {
                    item_id: 1,
                    count: 1,
                    damage: 0,
                    nbt: None,
                }),
            }),
        );
        assert_eq!(pre_join.local.head_yaw, 0.0);
        assert!(pre_join.local.equipment.iter().all(|item| item.is_none()));

        let mut state = ClientState::new();
        drain(&mut state, join());
        let local_id = state.local.entity_id;

        for (slot, item_id) in (0_i16..=4).zip(300_i32..=304) {
            let item = Some(mc_protocol::ItemStack {
                item_id: item_id as i16,
                count: 1,
                damage: slot,
                nbt: None,
            });
            let events = drain(
                &mut state,
                S2CPlay::EntityEquipment(s2c::EntityEquipment {
                    entity_id: local_id,
                    equipment_slot: slot,
                    item: item.clone(),
                }),
            );
            assert!(matches!(
                events.as_slice(),
                [ClientEvent::EntityEquipment {
                    network_id,
                    slot: event_slot,
                    item: event_item,
                }] if *network_id == local_id && *event_slot == slot && *event_item == item
            ));
            assert_eq!(state.local.equipment[slot as usize], item);
        }

        // Seed the local packet shadow as if the predictor were at a live position.  Dispatch
        // owns only packet bookkeeping; this direct-dispatch test supplies the same current
        // entity fields that Session refreshes before every network packet.
        state.local.server_pos = [32, 64, 96];
        state
            .local
            .sync_entity_packet_shadow(DVec3::new(1.0, 2.0, 3.0), 15.0, -10.0, false);
        drain(
            &mut state,
            S2CPlay::EntityRelMove(s2c::EntityRelMove {
                entity_id: local_id,
                delta_x: 4,
                delta_y: -2,
                delta_z: 8,
                on_ground: true,
            }),
        );
        assert_eq!(state.local.server_pos, [36, 62, 104]);
        assert_eq!(state.local.target_pos, DVec3::new(1.125, 1.9375, 3.25));
        assert_eq!(state.local.target_yaw, 15.0);
        assert_eq!(state.local.target_pitch, -10.0);
        assert_eq!(state.local.interpolation_steps, 3);
        assert!(state.local.on_ground);

        drain(
            &mut state,
            S2CPlay::EntityLook(s2c::EntityLook {
                entity_id: local_id,
                yaw: Angle::from_degrees(90.0),
                pitch: Angle::from_degrees(30.0),
                on_ground: false,
            }),
        );
        assert_eq!(state.local.server_pos, [36, 62, 104]);
        assert_eq!(state.local.target_yaw, 90.0);
        assert!((state.local.target_pitch - 30.0).abs() < 1.0);
        assert!(!state.local.on_ground);

        drain(
            &mut state,
            S2CPlay::EntityLookAndRelMove(s2c::EntityLookAndRelMove {
                entity_id: local_id,
                delta_x: -4,
                delta_y: 2,
                delta_z: -8,
                yaw: Angle::from_degrees(-45.0),
                pitch: Angle::from_degrees(-20.0),
                on_ground: true,
            }),
        );
        assert_eq!(state.local.server_pos, [32, 64, 96]);
        assert_eq!(state.local.target_pos, DVec3::new(1.0, 2.0, 3.0));
        assert!((state.local.target_yaw + 45.0).abs() < 1.0);
        assert!((state.local.target_pitch + 20.0).abs() < 1.0);
        assert!(state.local.on_ground);

        // Strictly inside Java's dead zone keeps the current position while still replacing the
        // target rotation.  Exactly on the X threshold is outside because Java uses `<`, not `<=`.
        state.local.sync_entity_packet_shadow(
            DVec3::new(1.0 + 0.03124, 2.0 + 0.015_624, 3.0 - 0.03124),
            12.0,
            13.0,
            true,
        );
        drain(
            &mut state,
            S2CPlay::EntityTeleport(s2c::EntityTeleport {
                entity_id: local_id,
                x: 32,
                y: 64,
                z: 96,
                yaw: Angle::from_degrees(180.0),
                pitch: Angle::from_degrees(40.0),
                on_ground: false,
            }),
        );
        assert_eq!(
            state.local.target_pos,
            DVec3::new(1.0 + 0.03124, 2.0 + 0.015_624, 3.0 - 0.03124)
        );
        assert!(!state.local.on_ground);

        state.local.sync_entity_packet_shadow(
            DVec3::new(1.0 + 0.03125, 2.0, 3.0),
            12.0,
            13.0,
            true,
        );
        drain(
            &mut state,
            S2CPlay::EntityTeleport(s2c::EntityTeleport {
                entity_id: local_id,
                x: 32,
                y: 64,
                z: 96,
                yaw: Angle::from_degrees(180.0),
                pitch: Angle::from_degrees(40.0),
                on_ground: true,
            }),
        );
        assert_eq!(state.local.target_pos, DVec3::new(1.0, 2.0, 3.0));
        assert!(state.local.on_ground);

        drain(
            &mut state,
            S2CPlay::EntityHeadLook(s2c::EntityHeadLook {
                entity_id: local_id,
                yaw: Angle::from_degrees(-135.0),
            }),
        );
        assert_eq!(state.local.head_yaw, -135.0);

        // Remote packets still use the tracker, and an unknown id remains a no-op.  In
        // particular, an unknown id must not mutate the local player's packet shadow merely
        // because the local player is absent from that tracker.
        let remote_id = 88;
        state.world.load_empty_chunk(ChunkPos::new(0, 0));
        spawn_remote_zombie(&mut state, remote_id);
        drain(
            &mut state,
            S2CPlay::EntityRelMove(s2c::EntityRelMove {
                entity_id: remote_id,
                delta_x: 3,
                delta_y: 0,
                delta_z: 0,
                on_ground: true,
            }),
        );
        assert_eq!(state.entities.get(remote_id).unwrap().server_pos[0], 3);
        assert!(state.entities.get(remote_id).unwrap().on_ground);

        let local_before_unknown = state.local.clone();
        let unknown_id = 999;
        drain(
            &mut state,
            S2CPlay::EntityTeleport(s2c::EntityTeleport {
                entity_id: unknown_id,
                x: 32 * 77,
                y: 32 * 88,
                z: 32 * 99,
                yaw: Angle::from_degrees(20.0),
                pitch: Angle::from_degrees(-10.0),
                on_ground: false,
            }),
        );
        drain(
            &mut state,
            S2CPlay::EntityEquipment(s2c::EntityEquipment {
                entity_id: unknown_id,
                equipment_slot: 0,
                item: Some(mc_protocol::ItemStack {
                    item_id: 1,
                    count: 1,
                    damage: 0,
                    nbt: None,
                }),
            }),
        );
        drain(
            &mut state,
            S2CPlay::EntityHeadLook(s2c::EntityHeadLook {
                entity_id: unknown_id,
                yaw: Angle::from_degrees(70.0),
            }),
        );
        assert_eq!(state.local, local_before_unknown);
        assert!(state.entities.get(unknown_id).is_none());
    }

    #[test]
    fn joined_zero_entity_id_is_local_but_pre_join_zero_is_unknown() {
        let mut state = ClientState::new();
        drain(
            &mut state,
            S2CPlay::EntityHeadLook(s2c::EntityHeadLook {
                entity_id: 0,
                yaw: Angle::from_degrees(75.0),
            }),
        );
        assert_eq!(state.local.head_yaw, 0.0);

        let S2CPlay::JoinGame(mut packet) = join() else {
            unreachable!("join helper must construct S01");
        };
        packet.entity_id = 0;
        drain(&mut state, S2CPlay::JoinGame(packet));
        drain(
            &mut state,
            S2CPlay::EntityHeadLook(s2c::EntityHeadLook {
                entity_id: 0,
                yaw: Angle::from_degrees(75.0),
            }),
        );
        assert_eq!(state.local.head_yaw, Angle::from_degrees(75.0).to_degrees());
    }

    /// **ENTITY-2's D5.** `handleJoinGame` does not touch `playerInfoMap`, and it is not an
    /// omission: the only three writes in `NetHandlerPlayClient` are inside
    /// `handlePlayerListItem`, and the header/footer are dropped only by
    /// `Minecraft.loadWorld(null)`. A second `S01PacketJoinGame` on one connection is what
    /// a proxy sends on every server switch, and the `state.players.clear()` that used to be
    /// in that arm emptied the tab list every time.
    ///
    /// The negative control is the same packet's *other* effects, asserted alongside: the world
    /// really is rebuilt and the local player really is reconfigured, so this is not passing
    /// because the arm stopped doing anything.
    #[test]
    fn a_second_join_game_does_not_empty_the_tab_list() {
        let mut state = ClientState::new();
        drain(&mut state, join());
        drain(
            &mut state,
            S2CPlay::PlayerListItem(s2c::PlayerListItem::AddPlayer(vec![s2c::AddPlayerEntry {
                uuid: uuid::Uuid::from_u128(7),
                username: "Notch".into(),
                properties: Vec::new(),
                game_mode: s2c::GameMode::Survival,
                ping: 12,
                display_name: None,
            }])),
        );
        assert_eq!(state.players.len(), 1);

        drain(
            &mut state,
            S2CPlay::JoinGame(s2c::JoinGame {
                entity_id: 91,
                hardcore_mode: false,
                game_mode: 0,
                dimension: -1,
                difficulty: 1,
                max_players: 20,
                level_type: "default".into(),
                reduced_debug_info: false,
            }),
        );

        assert_eq!(
            state.players.len(),
            1,
            "java: `handleJoinGame` never writes `playerInfoMap`"
        );
        assert_eq!(state.local.entity_id, 91, "control: the arm still ran");
        assert_eq!(
            state.local.level_type, "default",
            "control: the arm still ran"
        );
    }

    #[test]
    fn join_game_configures_the_local_player_and_a_fresh_world() {
        let mut state = ClientState::new();
        let events = drain(&mut state, join());
        assert_eq!(state.local.entity_id, 77);
        assert_eq!(state.local.game_mode, s2c::GameMode::Creative);
        assert_eq!(state.local.level_type, "flat");
        assert_eq!(state.env.difficulty, s2c::ServerDifficultyLevel::Normal);
        assert!(state.world.has_sky_light());
        assert!(matches!(events[0], ClientEvent::JoinedGame(_)));
    }

    #[test]
    fn live_join_and_respawn_select_flat_debug_and_dimension_biome_fallbacks() {
        let probe = BlockPos::new(3, 64, 5);
        let mut state = ClientState::new();

        // S01 FLAT: Java's client WorldInfo has empty generator options, so the provider is the
        // fixed Plains manager. Both an absent column and Chunk::empty's 0xFF sentinel take this
        // provider path; neither is allowed to fall through to the seed-0 graph.
        drain(&mut state, join());
        assert_eq!(
            state.world.biome_source(),
            mc_world::world::BiomeSource::FlatPlains
        );
        assert_eq!(state.world.biome_at(probe), 1);
        state.world.load_empty_chunk(probe.chunk_pos());
        assert_eq!(state.world.biome_at(probe), 1);

        // Same-dimension S07 does not construct a WorldClient in Java. Its differing terrain
        // type is therefore ignored, and the existing fixed provider remains installed.
        drain(
            &mut state,
            S2CPlay::Respawn(s2c::Respawn {
                dimension: 0,
                difficulty: 2,
                game_mode: 0,
                level_type: "default".into(),
            }),
        );
        assert_eq!(state.local.level_type, "flat");
        assert_eq!(
            state.world.biome_source(),
            mc_world::world::BiomeSource::FlatPlains
        );
        assert_eq!(state.world.biome_at(probe), 1);

        // A cross-dimension S07 selects the dimension provider before terrain type. The FLAT
        // label must not leak Nether's fixed Hell biome, including through a fresh sentinel.
        drain(
            &mut state,
            S2CPlay::Respawn(s2c::Respawn {
                dimension: -1,
                difficulty: 2,
                game_mode: 0,
                level_type: "flat".into(),
            }),
        );
        assert_eq!(
            state.world.biome_source(),
            mc_world::world::BiomeSource::Nether
        );
        assert_eq!(state.world.biome_at(probe), 8);
        state.world.load_empty_chunk(probe.chunk_pos());
        assert_eq!(state.world.biome_at(probe), 8);

        // Returning cross-dimension to the surface with DEBUG selects the fixed Plains manager.
        drain(
            &mut state,
            S2CPlay::Respawn(s2c::Respawn {
                dimension: 0,
                difficulty: 2,
                game_mode: 0,
                level_type: "debug_all_block_states".into(),
            }),
        );
        assert_eq!(
            state.world.biome_source(),
            mc_world::world::BiomeSource::DebugPlains
        );
        assert_eq!(state.world.biome_at(probe), 1);

        // Ordinary DEFAULT and End controls retain their pre-existing providers.
        let mut ordinary = ClientState::new();
        drain(
            &mut ordinary,
            S2CPlay::JoinGame(s2c::JoinGame {
                entity_id: 77,
                hardcore_mode: false,
                game_mode: 0,
                dimension: 0,
                difficulty: 2,
                max_players: 20,
                level_type: "default".into(),
                reduced_debug_info: false,
            }),
        );
        assert_eq!(
            ordinary.world.biome_source(),
            mc_world::world::BiomeSource::OverworldSeed0
        );
        assert_eq!(ordinary.world.biome_at(probe), 4);

        drain(
            &mut state,
            S2CPlay::Respawn(s2c::Respawn {
                dimension: 1,
                difficulty: 2,
                game_mode: 0,
                level_type: "debug_all_block_states".into(),
            }),
        );
        assert_eq!(
            state.world.biome_source(),
            mc_world::world::BiomeSource::End
        );
        assert_eq!(state.world.biome_at(probe), 9);
    }

    /// java: `JoinGame`'s gamemode byte packs hardcore into bit 3, and `S01PacketJoinGame`
    /// already unpacks it — a `game_mode` of 3 must be Spectator, not Adventure|hardcore.
    #[test]
    fn spectator_join_is_spectator() {
        let mut state = ClientState::new();
        let mut packet = join();
        if let S2CPlay::JoinGame(ref mut join) = packet {
            join.game_mode = 3;
        }
        drain(&mut state, packet);
        assert_eq!(state.local.game_mode, s2c::GameMode::Spectator);
    }

    /// `handleRespawn` rebuilds `WorldClient` only when the dimension differs, but always calls
    /// `setDimensionAndSpawnPlayer`.  The deliberately mismatched same-dimension packet is the
    /// negative control: its terrain fields must *not* affect the old world, while its game type,
    /// remote-entity purge, initial spawn, and capabilities still must.
    #[test]
    fn respawn_distinguishes_world_recreation_from_every_spawn_player_reset() {
        let mut state = ClientState::new();
        drain(&mut state, join());
        state.local.done_loading_terrain = true;
        state.env.spawn_position = BlockPos::new(1, 2, 3);
        state.local.abilities = Abilities::from_flags(0x02, 0.25, 0.4);
        state.local.reduced_debug_info = true;
        state
            .local
            .metadata
            .set(0, mc_world::entity::MetadataValue::Byte(0x12));
        state
            .local
            .metadata
            .set(17, mc_world::entity::MetadataValue::Float(6.5));
        state.entities.spawn(
            &mut state.world,
            12,
            TrackedKind::Type(EntityType::Zombie),
            None,
            [0, 0, 0],
            0.0,
            0.0,
            DVec3::ZERO,
            mc_world::Metadata::new(),
        );

        let events = drain(
            &mut state,
            S2CPlay::Respawn(s2c::Respawn {
                dimension: 0,
                difficulty: 3,
                game_mode: 3,
                level_type: "largeBiomes".into(),
            }),
        );
        assert!(matches!(
            events[0],
            ClientEvent::Respawned {
                changed_world: false,
                ..
            }
        ));
        assert!(matches!(events[1], ClientEvent::AbilitiesChanged(_)));
        assert!(
            state.entities.is_empty(),
            "every respawn removes old entities"
        );
        assert_eq!(state.env.spawn_position, BlockPos::new(8, 64, 8));
        assert_eq!(state.local.game_mode, s2c::GameMode::Spectator);
        assert_eq!(state.local.level_type, "flat");
        assert!(
            state.local.reduced_debug_info,
            "setDimensionAndSpawnPlayer copies the old reduced-debug latch"
        );
        assert_eq!(
            state.local.metadata.get(0),
            Some(&mc_world::entity::MetadataValue::Byte(0x12)),
            "same-dimension replacement copies old DataWatcher values"
        );
        assert_eq!(
            state.local.metadata.get(17),
            Some(&mc_world::entity::MetadataValue::Float(6.5)),
            "same-dimension replacement copies every watched slot"
        );
        assert_eq!(state.env.difficulty, s2c::ServerDifficultyLevel::Normal);
        assert_eq!(
            state.local.abilities,
            Abilities {
                invulnerable: true,
                flying: true,
                allow_flying: true,
                creative_mode: false,
                allow_edit: false,
                fly_speed: 0.05,
                walk_speed: 0.1,
            },
            "the fresh PlayerCapabilities object is configured before and after S07's game type"
        );
        assert!(
            state.local.done_loading_terrain,
            "same-dimension respawn does not reopen GuiDownloadTerrain"
        );

        // Negative control for the always-clear requirement: there is a fresh remote entity to
        // distinguish the cross-dimension WorldClient replacement from the earlier same-world
        // tracker purge.
        state.entities.spawn(
            &mut state.world,
            13,
            TrackedKind::Type(EntityType::Zombie),
            None,
            [0, 0, 0],
            0.0,
            0.0,
            DVec3::ZERO,
            mc_world::Metadata::new(),
        );

        let events = drain(
            &mut state,
            S2CPlay::Respawn(s2c::Respawn {
                dimension: -1,
                difficulty: 0,
                game_mode: 1,
                level_type: "default".into(),
            }),
        );
        assert!(matches!(
            events[0],
            ClientEvent::Respawned {
                changed_world: true,
                dimension: -1
            }
        ));
        // Hazard H6: the Nether has no sky light, and the flag must follow the dimension.
        assert!(!state.world.has_sky_light());
        assert!(state.entities.is_empty());
        assert_eq!(state.env.spawn_position, BlockPos::new(8, 64, 8));
        assert_eq!(state.local.level_type, "default");
        assert_eq!(state.env.difficulty, s2c::ServerDifficultyLevel::Peaceful);
        assert!(
            state.local.reduced_debug_info,
            "cross-dimension replacement also copies the old reduced-debug latch"
        );
        assert_eq!(
            state.local.metadata.get(0),
            Some(&mc_world::entity::MetadataValue::Byte(0x12)),
            "cross-dimension replacement copies old DataWatcher values"
        );
        assert_eq!(
            state.local.metadata.get(17),
            Some(&mc_world::entity::MetadataValue::Float(6.5)),
            "cross-dimension replacement copies every watched slot"
        );
        assert!(state.local.abilities.creative_mode);
        assert!(state.local.abilities.allow_flying);
        assert!(state.local.abilities.invulnerable);
        assert!(
            !state.local.abilities.flying,
            "a replacement player starts with Java's fresh flying=false capability"
        );
        assert!(
            !state.local.done_loading_terrain,
            "dimension change opens a fresh GuiDownloadTerrain"
        );
    }

    /// Scoreboard packets describe server/session state, while S07 only replaces the
    /// dimension-scoped world.  The old shared `reset_world` path cleared both, making an
    /// existing sidebar disappear during every Nether or End transfer.  Conversely, S01 begins
    /// a fresh server session and must discard the old board.  The two transitions and their
    /// opposite assertions are deliberately adjacent so replacing either lifecycle call with
    /// the other is caught.
    #[test]
    fn dimension_respawn_keeps_the_scoreboard_but_join_game_starts_a_fresh_one() {
        let mut state = ClientState::new();
        drain(&mut state, join());
        install_scoreboard_objective(&mut state);
        assert!(state.scoreboard.objective("across_worlds").is_some());

        drain(
            &mut state,
            S2CPlay::Respawn(s2c::Respawn {
                dimension: -1,
                difficulty: 2,
                game_mode: 0,
                level_type: "default".into(),
            }),
        );
        assert!(
            state.scoreboard.objective("across_worlds").is_some(),
            "S07's dimension swap does not erase the server scoreboard"
        );
        assert_eq!(state.local.dimension, -1, "control: the world did change");

        drain(
            &mut state,
            S2CPlay::JoinGame(s2c::JoinGame {
                entity_id: 91,
                hardcore_mode: false,
                game_mode: 0,
                dimension: 0,
                difficulty: 1,
                max_players: 20,
                level_type: "default".into(),
                reduced_debug_info: false,
            }),
        );
        assert!(
            state.scoreboard.objective("across_worlds").is_none(),
            "S01 starts a fresh board rather than leaking a previous server's objective"
        );
        assert_eq!(
            state.local.entity_id, 91,
            "control: S01 still rebuilt state"
        );
    }

    /// S06's three fields still update together, but health itself must travel through
    /// `EntityPlayerSP.setPlayerSPHealth`: a hit arms the local hurt state while a respawn makes
    /// the next health snapshot quiet again. Negative control: changing the S06 arm back to a
    /// bare assignment leaves `hurt_time` at zero and fails here.
    #[test]
    fn update_health_preserves_the_local_player_sp_damage_lifecycle() {
        let mut state = ClientState::new();
        let packet = |health, food_level, saturation_level| {
            S2CPlay::UpdateHealth(s2c::UpdateHealth {
                health,
                food_level,
                saturation_level,
            })
        };

        drain(&mut state, packet(20.0, 17, 3.5));
        assert!(state.local.has_valid_health);
        assert_eq!(state.local.hurt_time, 0, "the initial S06 is not damage");
        assert_eq!((state.local.food_level, state.local.saturation), (17, 3.5));

        drain(&mut state, packet(13.0, 9, 1.25));
        assert_eq!(state.local.health, 13.0);
        assert_eq!(state.local.last_damage, 7.0);
        assert_eq!(state.local.hurt_resistant_time, 20);
        assert_eq!(state.local.hurt_time, 10);
        assert_eq!((state.local.food_level, state.local.saturation), (9, 1.25));

        drain(
            &mut state,
            S2CPlay::Respawn(s2c::Respawn {
                dimension: 0,
                difficulty: 1,
                game_mode: 0,
                level_type: "default".into(),
            }),
        );
        assert!(!state.local.has_valid_health);
        assert_eq!(state.local.hurt_time, 0);

        drain(&mut state, packet(5.0, 20, 5.0));
        assert_eq!(state.local.health, 5.0);
        assert_eq!(
            state.local.hurt_time, 0,
            "the new life's first S06 is quiet"
        );
    }

    /// S06's health field reaches `EntityPlayerSP.setHealth`, whose Java clamp makes a negative
    /// payload zero before `setPlayerSPHealth` compares it. This live dispatch regression keeps
    /// the malformed-packet edge on the same hurt/death pipeline as a real lethal hit.
    #[test]
    fn update_health_clamps_negative_health_before_the_damage_pipeline() {
        let mut state = ClientState::new();
        let packet = |health| {
            S2CPlay::UpdateHealth(s2c::UpdateHealth {
                health,
                food_level: 20,
                saturation_level: 5.0,
            })
        };

        drain(&mut state, packet(20.0));
        drain(&mut state, packet(-4.0));
        assert_eq!(state.local.health, 0.0);
        assert_eq!(state.local.last_damage, 24.0);
        assert_eq!(state.local.hurt_resistant_time, 20);
        assert_eq!(state.local.hurt_time, 10);
        state.local.tick_living_state();
        assert_eq!(state.local.death_time, 1);

        drain(&mut state, packet(f32::NAN));
        assert!(
            state.local.health.is_nan(),
            "Java's comparison clamp preserves NaN"
        );
        assert!(state.local.last_damage.is_nan());
        state.local.tick_living_state();
        assert_eq!(
            state.local.death_time, 1,
            "NaN is not <= zero in Java's death-update guard"
        );
    }

    /// `handleEntityStatus` looks its id up in `WorldClient`, where the local
    /// `EntityPlayerSP` lives alongside remote entities.  mc-rs keeps that player outside the
    /// tracker, so the dispatch boundary must route its two EntityLivingBase opcodes directly
    /// while retaining the event for the remote render path.  The wrong-id control catches the
    /// tempting but incorrect "all status packets affect LocalPlayer" implementation.
    #[test]
    fn entity_status_reaches_only_the_matching_local_player() {
        let mut state = ClientState::new();
        drain(&mut state, join());
        let status =
            |entity_id, opcode| S2CPlay::EntityStatus(s2c::EntityStatus { entity_id, opcode });

        state.local.max_hurt_time = 37;
        let events = drain(&mut state, status(77, 2));
        assert_eq!(state.local.max_hurt_time, 10);
        assert_eq!(state.local.hurt_time, 10);
        assert!(matches!(
            events.as_slice(),
            [ClientEvent::EntityStatus {
                network_id: 77,
                opcode: 2
            }]
        ));

        state.local.health = 12.0;
        state.local.hurt_time = 4;
        let events = drain(&mut state, status(78, 3));
        assert_eq!(
            state.local.health, 12.0,
            "a remote death does not kill the local player"
        );
        assert_eq!(
            state.local.hurt_time, 4,
            "control: the wrong id made no local mutation"
        );
        assert!(matches!(
            events.as_slice(),
            [ClientEvent::EntityStatus {
                network_id: 78,
                opcode: 3
            }]
        ));

        drain(&mut state, status(77, 3));
        assert_eq!(
            state.local.health, 0.0,
            "opcode 3 is EntityLivingBase.setHealth(0)"
        );
    }

    #[test]
    fn the_first_pos_look_marks_terrain_downloaded_exactly_once() {
        let mut state = ClientState::new();
        drain(&mut state, join());
        let packet = || {
            S2CPlay::PlayerPositionLook(s2c::PlayerPositionLook {
                x: 1.0,
                y: 64.0,
                z: 2.0,
                yaw: 90.0,
                pitch: 0.0,
                flags: 0,
            })
        };
        let first = drain(&mut state, packet());
        assert!(matches!(first[0], ClientEvent::PositionCorrection(_)));
        assert!(matches!(first[1], ClientEvent::TerrainDownloaded));

        let second = drain(&mut state, packet());
        assert_eq!(second.len(), 1, "TerrainDownloaded must fire only once");
    }

    #[test]
    fn a_relative_pos_look_keeps_its_flags() {
        let mut state = ClientState::new();
        let events = drain(
            &mut state,
            S2CPlay::PlayerPositionLook(s2c::PlayerPositionLook {
                x: 0.5,
                y: 0.0,
                z: 0.0,
                yaw: 0.0,
                pitch: 0.0,
                flags: 0x01 | 0x08,
            }),
        );
        match &events[0] {
            ClientEvent::PositionCorrection(correction) => {
                assert!(correction.relative_x);
                assert!(!correction.relative_y);
                assert!(correction.relative_yaw);
                assert!(!correction.relative_pitch);
            }
            other => panic!("expected a correction, got {other:?}"),
        }
    }

    #[test]
    fn block_change_uses_the_block_state_ids_packing() {
        let mut state = ClientState::new();
        // One empty section so the chunk exists.
        let payload = vec![0u8; mc_world::expected_chunk_data_len(0b1, true, true)];
        state
            .world
            .apply_chunk_data(ChunkPos::new(0, 0), true, 0b1, &payload)
            .unwrap();

        // Stone is block id 1, meta 0 -> `1 << 4` = 16 in BLOCK_STATE_IDS packing.
        drain(
            &mut state,
            S2CPlay::BlockChange(s2c::BlockChange {
                position: BlockPos::new(3, 4, 5),
                block_state_id: 16,
            }),
        );
        assert_eq!(
            state.world.block_at(BlockPos::new(3, 4, 5)),
            BlockState::from_raw(16)
        );
        assert_eq!(
            state.world.block_at(BlockPos::new(3, 4, 5)).block().0,
            1,
            "block id 1 = stone"
        );
    }

    /// java: `BLOCK_STATE_IDS.getByValue` is null for a negative
    /// id, an id past the registry and an unregistered meta. S23's handler then throws inside
    /// `World.setBlockState`, so the old block stays; S22 applies its entries in order and stops
    /// at the first null. Rust used to mask the id to 16 bits (`0x10010` became stone) and to
    /// store unregistered ids, which read back as air.
    #[test]
    fn an_unregistered_block_state_id_leaves_the_block_like_a_null_state() {
        let mut state = ClientState::new();
        let payload = vec![0u8; mc_world::expected_chunk_data_len(0b1, true, true)];
        state
            .world
            .apply_chunk_data(ChunkPos::new(0, 0), true, 0b1, &payload)
            .unwrap();
        let pos = BlockPos::new(3, 4, 5);
        let glass = BlockState::from_raw(20 << 4);
        drain(
            &mut state,
            S2CPlay::BlockChange(s2c::BlockChange {
                position: pos,
                block_state_id: 20 << 4,
            }),
        );
        assert_eq!(state.world.block_at(pos), glass);

        // Stone (block 1) registers metas 0..=6 only; 0x10010 masks to stone meta 0.
        assert!(!BlockState::from_raw((1 << 4) | 7).is_legal());
        for id in [-1, 0x1_0010, (1 << 4) | 7, i32::MAX] {
            let mut events = Vec::new();
            let result = apply(
                &mut state,
                S2CPlay::BlockChange(s2c::BlockChange {
                    position: pos,
                    block_state_id: id,
                }),
                &mut events,
            );
            assert!(
                result.is_err(),
                "{id:#x}: the null state throws in the handler"
            );
            assert_eq!(
                state.world.block_at(pos),
                glass,
                "{id:#x} must not change it"
            );
        }

        // S22: the entry before the null is applied, the ones after it are not.
        let entry = |rel_x: u8, block_state_id: i32| s2c::MultiBlockChangeEntry {
            rel_x,
            y: 4,
            rel_z: 5,
            block_state_id,
        };
        let mut events = Vec::new();
        let result = apply(
            &mut state,
            S2CPlay::MultiBlockChange(s2c::MultiBlockChange {
                chunk_x: 0,
                chunk_z: 0,
                changes: vec![entry(1, 1 << 4), entry(2, 0x1_0010), entry(3, 1 << 4)],
            }),
            &mut events,
        );
        assert!(result.is_err());
        assert_eq!(state.world.block_at(BlockPos::new(1, 4, 5)).block().0, 1);
        assert_eq!(
            state.world.block_at(BlockPos::new(2, 4, 5)),
            BlockState::AIR
        );
        assert_eq!(
            state.world.block_at(BlockPos::new(3, 4, 5)),
            glass,
            "entries after the null are never reached"
        );
    }

    #[test]
    fn block_change_is_packet_pure_for_farmland_until_authoritative_dirt_arrives() {
        let mut state = ClientState::new();
        state.world.load_empty_chunk(ChunkPos::new(0, 0));
        let farmland = BlockPos::new(3, 63, 5);
        let above = BlockPos::new(3, 64, 5);
        state
            .world
            .set_block(farmland, BlockState::from_raw(60 << 4));

        drain(
            &mut state,
            S2CPlay::BlockChange(s2c::BlockChange {
                position: above,
                block_state_id: 1 << 4,
            }),
        );
        assert_eq!(state.world.block_at(above).block().0, 1, "stone arrived");
        assert_eq!(
            state.world.block_at(farmland),
            BlockState::from_raw(60 << 4),
            "Java's S23 client handler direct-writes the changed block and does not run BlockFarmland neighbor callbacks"
        );

        drain(
            &mut state,
            S2CPlay::BlockChange(s2c::BlockChange {
                position: farmland,
                block_state_id: 3 << 4,
            }),
        );
        assert_eq!(
            state.world.block_at(farmland),
            BlockState::from_raw(3 << 4),
            "the later authoritative farmland-to-dirt packet owns the conversion"
        );
    }

    #[test]
    fn block_change_overwrites_a_locally_predicted_open_fence_gate() {
        let mut state = ClientState::new();
        state.world.load_empty_chunk(ChunkPos::new(0, 0));
        let gate = BlockPos::new(3, 64, 5);
        let closed = BlockState::from_raw(107 << 4);
        let predicted_open = BlockState::from_raw((107 << 4) | 4);
        state.world.set_block(gate, predicted_open);

        drain(
            &mut state,
            S2CPlay::BlockChange(s2c::BlockChange {
                position: gate,
                block_state_id: i32::from(closed.id),
            }),
        );

        assert_eq!(
            state.world.block_at(gate),
            closed,
            "the authoritative S23 state must replace the local open-bit prediction"
        );
    }

    #[test]
    fn block_changes_overwrite_predicted_lit_redstone_without_replaying_the_click() {
        let mut state = ClientState::new();
        state.world.load_empty_chunk(ChunkPos::new(0, 0));
        let pos = BlockPos::new(3, 64, 5);
        let unlit = BlockState::from_raw(73 << 4);
        let lit = BlockState::from_raw(74 << 4);

        state.world.set_block(pos, unlit);
        assert!(state.world.activate_redstone_ore_click(pos));
        assert_eq!(state.world.block_at(pos), lit);
        let events = drain(
            &mut state,
            S2CPlay::BlockChange(s2c::BlockChange {
                position: pos,
                block_state_id: i32::from(unlit.id),
            }),
        );
        assert!(events.is_empty());
        assert_eq!(
            state.world.block_at(pos),
            unlit,
            "S23 owns the final state and does not replay local activation"
        );

        assert!(state.world.activate_redstone_ore_click(pos));
        assert_eq!(state.world.block_at(pos), lit);
        let events = drain(
            &mut state,
            S2CPlay::MultiBlockChange(s2c::MultiBlockChange {
                chunk_x: 0,
                chunk_z: 0,
                changes: vec![s2c::MultiBlockChangeEntry {
                    rel_x: 3,
                    y: 64,
                    rel_z: 5,
                    block_state_id: i32::from(unlit.id),
                }],
            }),
        );
        assert!(events.is_empty());
        assert_eq!(
            state.world.block_at(pos),
            unlit,
            "S22 owns the final state and does not replay local activation"
        );
    }

    #[test]
    fn multi_block_change_recrams_the_position_short() {
        let mut state = ClientState::new();
        let payload = vec![0u8; mc_world::expected_chunk_data_len(0b1, true, true)];
        state
            .world
            .apply_chunk_data(ChunkPos::new(2, -1), true, 0b1, &payload)
            .unwrap();

        drain(
            &mut state,
            S2CPlay::MultiBlockChange(s2c::MultiBlockChange {
                chunk_x: 2,
                chunk_z: -1,
                changes: vec![s2c::MultiBlockChangeEntry {
                    rel_x: 3,
                    y: 7,
                    rel_z: 5,
                    block_state_id: 16,
                }],
            }),
        );
        assert_eq!(
            state.world.block_at(BlockPos::new(2 * 16 + 3, 7, -16 + 5)),
            BlockState::from_raw(16)
        );
    }

    /// java: `Explosion.doExplosionB` clears every affected block, relative to `(int)posX`.
    #[test]
    fn explosion_clears_its_affected_blocks_around_a_truncated_origin() {
        let mut state = ClientState::new();
        let affected = BlockPos::new(4, 5, 6);
        // The explosion mutates the client world's existing chunk.  Load the target column
        // before placing the control block: `WorldClient.setBlockState` cannot make an unloaded
        // column materialize, and using a decoded all-ones payload couples this coordinate test
        // to palette/blank-chunk representation details.
        state.world.load_empty_chunk(ChunkPos::new(0, 0));
        assert!(state.world.is_chunk_loaded(ChunkPos::new(0, 0)));
        state
            .world
            .set_block(affected, BlockState::from_raw(1 << 4)); // stone, meta 0
        assert_ne!(state.world.block_at(affected), BlockState::AIR);

        drain(
            &mut state,
            S2CPlay::Explosion(s2c::Explosion {
                x: 4.9,
                y: 5.9,
                z: 6.9,
                strength: 4.0,
                affected_blocks: vec![s2c::ExplosionAffectedBlock {
                    dx: 0,
                    dy: 0,
                    dz: 0,
                }],
                motion_x: 0.0,
                motion_y: 0.0,
                motion_z: 0.0,
            }),
        );
        assert_eq!(
            state.world.block_at(affected),
            BlockState::AIR,
            "(int)4.9 is 4, so the origin block is the one cleared"
        );
    }

    /// A zero-offset-only fixture cannot distinguish the packet origin from an affected block.
    /// Keep a nonzero target plus two live decoys so dropping or misapplying the relative list
    /// changes an observable world cell, and retain the original packet for the particle sink.
    #[test]
    fn explosion_nonzero_offset_clears_only_the_target_and_preserves_event_offsets() {
        let mut state = ClientState::new();
        let origin = BlockPos::new(8, 64, 8);
        let target = BlockPos::new(10, 65, 7);
        let decoy = BlockPos::new(9, 64, 8);
        let stone = BlockState::from_raw(1 << 4);
        state.world.load_empty_chunk(ChunkPos::new(0, 0));
        state.world.set_block(origin, stone);
        state.world.set_block(target, stone);
        state.world.set_block(decoy, stone);

        let events = drain(
            &mut state,
            S2CPlay::Explosion(s2c::Explosion {
                x: 8.9,
                y: 64.9,
                z: 8.9,
                strength: 4.0,
                affected_blocks: vec![s2c::ExplosionAffectedBlock {
                    dx: 2,
                    dy: 1,
                    dz: -1,
                }],
                motion_x: 0.0,
                motion_y: 0.0,
                motion_z: 0.0,
            }),
        );

        assert_eq!(state.world.block_at(target), BlockState::AIR);
        assert_eq!(state.world.block_at(origin), stone);
        assert_eq!(state.world.block_at(decoy), stone);
        let [ClientEvent::Explosion(packet)] = events.as_slice() else {
            panic!("the live handler must preserve one explosion event");
        };
        assert_eq!((packet.x, packet.y, packet.z), (8.9, 64.9, 8.9));
        assert_eq!(packet.affected_blocks.len(), 1);
        let offset = &packet.affected_blocks[0];
        assert_eq!((offset.dx, offset.dy, offset.dz), (2, 1, -1));
    }

    /// Java oracle for `int j1 = buf.readByte() + j`: promote the signed byte to `int`, add in
    /// two's-complement arithmetic, and retain the low 32 bits. Cover every possible relative
    /// byte at both overflow boundaries as well as ordinary coordinates.
    #[test]
    fn explosion_axis_addition_matches_java_for_all_signed_offsets() {
        for origin in [i32::MIN, i32::MIN + 1, -1, 0, 1, i32::MAX - 1, i32::MAX] {
            for offset in i8::MIN..=i8::MAX {
                let java = (i64::from(origin) + i64::from(offset)) as i32;
                assert_eq!(
                    explosion_axis_coordinate(origin, offset),
                    java,
                    "origin {origin}, offset {offset}"
                );
            }
        }
    }

    /// Reach the live handler with all three narrowed origins at an integer boundary. The
    /// reconstructed positions are outside loaded chunks and therefore no-op, exactly as in
    /// `WorldClient`; the important contract is that packet handling completes without a debug
    /// overflow panic.
    #[test]
    fn hostile_explosion_coordinates_wrap_without_panicking() {
        let mut state = ClientState::new();
        let events = drain(
            &mut state,
            S2CPlay::Explosion(s2c::Explosion {
                x: f64::from(f32::MAX),
                y: f64::from(f32::MIN),
                z: f64::from(f32::MAX),
                strength: 4.0,
                affected_blocks: vec![s2c::ExplosionAffectedBlock {
                    dx: i8::MAX,
                    dy: i8::MIN,
                    dz: 1,
                }],
                motion_x: 0.0,
                motion_y: 0.0,
                motion_z: 0.0,
            }),
        );

        assert!(matches!(events.as_slice(), [ClientEvent::Explosion(_)]));
    }

    #[test]
    fn chunk_data_with_an_empty_bitmask_unloads_the_column() {
        let mut state = ClientState::new();
        let payload = vec![0u8; mc_world::expected_chunk_data_len(0b1, true, true)];
        let events = drain(
            &mut state,
            S2CPlay::ChunkData(mc_protocol::chunk::ChunkData {
                chunk_x: 5,
                chunk_z: 5,
                ground_up_continuous: true,
                primary_bit_mask: 0b1,
                data: payload,
            }),
        );
        assert!(matches!(events[0], ClientEvent::ChunkLoaded(_)));
        assert!(state.world.is_chunk_loaded(ChunkPos::new(5, 5)));

        let events = drain(
            &mut state,
            S2CPlay::ChunkData(mc_protocol::chunk::ChunkData {
                chunk_x: 5,
                chunk_z: 5,
                ground_up_continuous: true,
                primary_bit_mask: 0,
                data: Vec::new(),
            }),
        );
        assert!(matches!(events[0], ClientEvent::ChunkUnloaded(_)));
        assert!(!state.world.is_chunk_loaded(ChunkPos::new(5, 5)));
    }

    /// #15 / NET-1: a chunk packet whose columns were unpacked off-thread must leave the client in
    /// exactly the state, with exactly the events, of the same packet through `apply` — including
    /// queued entities activated by the load, the dirty set, and the sky-mismatch fallback.
    #[test]
    fn prebuilt_chunk_packets_apply_exactly_like_plain_ones() {
        fn payload(mask: u16, has_sky: bool, salt: u16) -> Vec<u8> {
            let sections = mask.count_ones() as usize;
            let mut data = Vec::new();
            for cell in 0..(sections * 4096) as u16 {
                let raw: u16 = match (cell ^ salt) % 13 {
                    0 => 1 << 4,
                    1 => 54 << 4,
                    2 => (9 << 4) | 3,
                    _ => 0,
                };
                data.extend_from_slice(&raw.to_le_bytes());
            }
            let light_arrays = if has_sky { 2 } else { 1 };
            data.resize(data.len() + sections * 2048 * light_arrays, 0x4C);
            data.resize(data.len() + 256, 4);
            data
        }
        fn zombie(entity_id: i32, chunk: ChunkPos) -> S2CPlay {
            S2CPlay::SpawnMob(s2c::SpawnMob {
                entity_id,
                entity_type: 54,
                x: (chunk.x * 16 + 8) * 32,
                y: 64 * 32,
                z: (chunk.z * 16 + 8) * 32,
                yaw: Angle(0),
                pitch: Angle(0),
                head_pitch: Angle(0),
                velocity_x: 0,
                velocity_y: 0,
                velocity_z: 0,
                metadata: Vec::new(),
            })
        }
        /// What the background thread would build; `None` where it would fail (a payload too
        /// short for the assumed layout), which never reaches `apply_prebuilt_chunks`.
        fn columns(packet: &S2CPlay, has_sky: bool) -> Option<Vec<mc_world::Chunk>> {
            let built = |mask: u16, data: &[u8]| {
                mc_world::Chunk::from_full_column_packet(mask, has_sky, data).ok()
            };
            match packet {
                S2CPlay::ChunkData(chunk) => {
                    Some(vec![built(chunk.primary_bit_mask, &chunk.data)?])
                }
                S2CPlay::MapChunkBulk(bulk) => bulk
                    .columns
                    .iter()
                    .map(|column| built(column.primary_bit_mask, &column.data))
                    .collect(),
                _ => unreachable!(),
            }
        }

        for world_sky in [true, false] {
            let a = ChunkPos::new(5, -2);
            let b = ChunkPos::new(6, -2);
            let c = ChunkPos::new(-9, 4);
            let s21 = S2CPlay::ChunkData(mc_protocol::chunk::ChunkData {
                chunk_x: a.x,
                chunk_z: a.z,
                ground_up_continuous: true,
                primary_bit_mask: 0b1001,
                data: payload(0b1001, world_sky, 3),
            });
            let s26 = S2CPlay::MapChunkBulk(mc_protocol::chunk::MapChunkBulk {
                has_sky_light: !world_sky, // layout only: the dimension stays the world's
                columns: [(b, 0b110u16), (c, 0)]
                    .into_iter()
                    .map(|(pos, mask)| mc_protocol::chunk::ChunkBulkColumn {
                        chunk_x: pos.x,
                        chunk_z: pos.z,
                        primary_bit_mask: mask,
                        data: payload(mask, !world_sky, 5),
                    })
                    .collect(),
            });
            // (packet, layout the background unpack assumed)
            let cases = [
                (s21.clone(), world_sky),
                (s26.clone(), !world_sky),
                // An S21 built for the other dimension falls back to `apply`.
                (s21.clone(), !world_sky),
            ];
            for (case, (packet, assumed_sky)) in cases.into_iter().enumerate() {
                let fresh = || {
                    let mut state = ClientState::new();
                    state.world.set_has_sky_light(world_sky);
                    let mut events = Vec::new();
                    for (id, pos) in [(40, a), (41, b), (42, c)] {
                        apply(&mut state, zombie(id, pos), &mut events).expect("spawn");
                    }
                    let _ = state.world.drain_dirty_sections();
                    state
                };
                let mut plain = fresh();
                let mut plain_events = Vec::new();
                apply(&mut plain, packet.clone(), &mut plain_events).expect("plain apply");

                let Some(built) = columns(&packet, assumed_sky) else {
                    continue;
                };
                let mut prebuilt = fresh();
                let mut prebuilt_events = Vec::new();
                apply_prebuilt_chunks(
                    &mut prebuilt,
                    packet.clone(),
                    assumed_sky,
                    built,
                    &mut prebuilt_events,
                )
                .expect("prebuilt apply");

                let context = format!("world sky {world_sky}, case {case}");
                assert_eq!(
                    format!("{prebuilt_events:?}"),
                    format!("{plain_events:?}"),
                    "{context}"
                );
                for pos in [a, b, c] {
                    assert_eq!(
                        format!("{:?}", prebuilt.world.chunk(pos)),
                        format!("{:?}", plain.world.chunk(pos)),
                        "{context}, column {pos:?}"
                    );
                }
                assert_eq!(
                    prebuilt.world.drain_dirty_sections(),
                    plain.world.drain_dirty_sections(),
                    "{context}"
                );
                assert_eq!(prebuilt.entities.len(), plain.entities.len(), "{context}");
            }
        }
    }

    #[test]
    fn chunk_unload_removes_current_column_entities_in_order_before_the_chunk_event() {
        fn zombie(entity_id: i32, x: i32, z: i32) -> S2CPlay {
            S2CPlay::SpawnMob(s2c::SpawnMob {
                entity_id,
                entity_type: 54,
                x: x * 32,
                y: 64 * 32,
                z: z * 32,
                yaw: Angle(0),
                pitch: Angle(0),
                head_pitch: Angle(0),
                velocity_x: 0,
                velocity_y: 0,
                velocity_z: 0,
                metadata: Vec::new(),
            })
        }

        let mut state = ClientState::new();
        let column = ChunkPos::new(5, 5);
        state.world.load_empty_chunk(column);

        // Deliberately non-numeric insertion order: a surviving rider, its doomed vehicle, an
        // entity that entered the column after spawning elsewhere, and an unrelated survivor.
        drain(&mut state, zombie(30, 97, 81));
        drain(&mut state, zombie(20, 81, 81));
        drain(&mut state, zombie(10, -31, -31));
        drain(&mut state, zombie(40, -31, -31));
        drain(
            &mut state,
            S2CPlay::EntityAttach(s2c::EntityAttach {
                entity_id: 30,
                vehicle_entity_id: 20,
                attach_type: s2c::AttachType::Vehicle,
            }),
        );

        let doomed_vehicle = state.entities.get(20).expect("doomed vehicle").entity;
        let moved = state.entities.get(10).expect("moved entity").entity;
        state
            .world
            .ecs
            .get::<&mut mc_world::entity::Position>(moved)
            .expect("live position")
            .0 = DVec3::new(82.0, 64.0, 82.0);

        let non_unload = drain(
            &mut state,
            S2CPlay::ChunkData(mc_protocol::chunk::ChunkData {
                chunk_x: column.x,
                chunk_z: column.z,
                ground_up_continuous: true,
                primary_bit_mask: 1,
                data: vec![0; mc_world::expected_chunk_data_len(1, true, true)],
            }),
        );
        assert!(matches!(
            non_unload.as_slice(),
            [ClientEvent::ChunkLoaded(pos)] if *pos == column
        ));
        assert!(state.entities.get(20).is_some());
        assert!(state.entities.get(10).is_some());
        assert_eq!(
            state
                .entities
                .get(30)
                .expect("rider remains mounted")
                .riding_entity(),
            Some(20),
            "a nonempty S21 must not run the chunk-unload entity path"
        );
        assert!(state
            .world
            .ecs
            .get::<&mc_world::entity::Position>(doomed_vehicle)
            .is_ok());

        let events = drain(
            &mut state,
            S2CPlay::ChunkData(mc_protocol::chunk::ChunkData {
                chunk_x: column.x,
                chunk_z: column.z,
                ground_up_continuous: true,
                primary_bit_mask: 0,
                data: Vec::new(),
            }),
        );

        assert!(matches!(
            events.as_slice(),
            [
                ClientEvent::EntityRemoved { network_id: 20 },
                ClientEvent::EntityRemoved { network_id: 10 },
                ClientEvent::ChunkUnloaded(pos)
            ] if *pos == column
        ));
        // `WorldClient.onEntityRemoved` requeues a live object when its chunk unloads.  The
        // loaded-list view is suppressed, but the id map and ECS generation survive so a later
        // load can re-admit the same object instead of manufacturing a replacement.
        assert!(state.entities.get(20).is_some());
        assert!(state.entities.get(10).is_some());
        assert!(!state.entities.is_resident(20));
        assert!(!state.entities.is_resident(10));
        assert!(state.entities.get(40).is_some());
        assert!(state
            .world
            .ecs
            .get::<&mc_world::entity::Position>(doomed_vehicle)
            .is_ok());
        assert!(state
            .world
            .ecs
            .get::<&mc_world::entity::Position>(moved)
            .is_ok());
        assert_eq!(
            state
                .entities
                .get(30)
                .expect("rider survives")
                .riding_entity(),
            Some(20),
            "chunk unload parks the vehicle generation; Java clears the mount only when the
             deferred unloaded-entity list is actually removed"
        );
    }

    #[test]
    fn a_short_chunk_payload_is_an_error_not_a_panic() {
        let mut state = ClientState::new();
        let mut events = Vec::new();
        assert!(matches!(
            apply(
                &mut state,
                S2CPlay::ChunkData(mc_protocol::chunk::ChunkData {
                    chunk_x: 0,
                    chunk_z: 0,
                    ground_up_continuous: true,
                    primary_bit_mask: 0b1111,
                    data: vec![0u8; 3],
                }),
                &mut events,
            ),
            Err(ClientError::World(_))
        ));
    }

    #[test]
    fn spawn_player_puts_a_remote_player_in_the_ecs() {
        use crate::sync::entities::RemotePlayer;
        let mut state = ClientState::new();
        let uuid = uuid::Uuid::from_u128(0x1234);
        let events = drain(
            &mut state,
            S2CPlay::SpawnPlayer(s2c::SpawnPlayer {
                entity_id: 11,
                player_uuid: uuid,
                x: 32 * 8,
                y: 32 * 70,
                z: 32 * -3,
                yaw: Angle::from_degrees(90.0),
                pitch: Angle(0),
                current_item: 264,
                metadata: Vec::new(),
            }),
        );
        assert!(matches!(events[0], ClientEvent::EntitySpawned { .. }));
        let tracked = state.entities.get(11).expect("tracked");
        assert_eq!(tracked.kind, TrackedKind::Player);
        assert_eq!(tracked.uuid, Some(uuid));
        let marker = state
            .world
            .ecs
            .get::<&RemotePlayer>(tracked.entity)
            .expect("player marker");
        assert_eq!(marker.uuid, uuid);
        let position = state
            .world
            .ecs
            .get::<&mc_world::entity::Position>(tracked.entity)
            .unwrap()
            .0;
        assert_eq!(position, DVec3::new(8.0, 70.0, -3.0));
        assert_eq!(
            tracked.equipment[0],
            Some(mc_protocol::ItemStack {
                item_id: 264,
                count: 1,
                damage: 0,
                nbt: None,
            }),
            "SpawnPlayer.current_item seeds the held slot before any S04 repeat"
        );
    }

    #[test]
    fn spawn_player_snapshots_profile_across_tab_replacement_and_removal() {
        let mut state = ClientState::new();
        let uuid = uuid::Uuid::from_u128(0x4455);
        let first_property = s2c::ProfileProperty {
            name: "textures".to_owned(),
            value: "first-textures".to_owned(),
            signature: Some("first-signature".to_owned()),
        };
        drain(
            &mut state,
            S2CPlay::PlayerListItem(s2c::PlayerListItem::AddPlayer(vec![s2c::AddPlayerEntry {
                uuid,
                username: "before".to_owned(),
                properties: vec![first_property.clone()],
                game_mode: s2c::GameMode::Spectator,
                ping: 12,
                display_name: None,
            }])),
        );
        drain(
            &mut state,
            S2CPlay::SpawnPlayer(s2c::SpawnPlayer {
                entity_id: 31,
                player_uuid: uuid,
                x: 0,
                y: 32 * 64,
                z: 0,
                yaw: Angle(0),
                pitch: Angle(0),
                current_item: 0,
                metadata: Vec::new(),
            }),
        );
        let first = state.entities.get(31).unwrap().player_profile().unwrap();
        assert_eq!(first.username, "before");
        assert_eq!(first.properties, vec![first_property]);

        // ADD_PLAYER replaces the mutable tab row, but not the GameProfile copied into the
        // already-existing EntityOtherPlayerMP.
        drain(
            &mut state,
            S2CPlay::PlayerListItem(s2c::PlayerListItem::AddPlayer(vec![s2c::AddPlayerEntry {
                uuid,
                username: "after".to_owned(),
                properties: vec![s2c::ProfileProperty {
                    name: "textures".to_owned(),
                    value: "replacement-textures".to_owned(),
                    signature: Some("replacement-signature".to_owned()),
                }],
                game_mode: s2c::GameMode::Survival,
                ping: 99,
                display_name: None,
            }])),
        );
        assert_eq!(state.players.get(&uuid).unwrap().username, "after");
        let tracked = state.entities.get(31).unwrap();
        assert_eq!(tracked.player_username(), Some("before"));

        // REMOVE_PLAYER removes only the tab row; the spawned entity remains fully renderable
        // from its own profile snapshot.
        drain(
            &mut state,
            S2CPlay::PlayerListItem(s2c::PlayerListItem::RemovePlayer(vec![uuid])),
        );
        assert!(state.players.get(&uuid).is_none());
        let tracked = state.entities.get(31).unwrap();
        assert_eq!(tracked.player_username(), Some("before"));
        assert_eq!(tracked.player_profile().unwrap().properties.len(), 1);
    }

    #[test]
    fn entity_nbt_is_a_vanilla_noop_for_remote_unknown_and_local_ids() {
        use mc_nbt::{Nbt, NbtCompound};

        let mut state = ClientState::new();
        drain(&mut state, join());
        spawn_remote_zombie(&mut state, 41);

        let remote_before = state.entities.get(41).unwrap().clone();
        let local_before = state.local.clone();
        let mut events = vec![ClientEvent::TerrainDownloaded];
        let mut tag = NbtCompound::new();
        tag.insert("Marker", Nbt::String("ignored".to_owned()));
        apply(
            &mut state,
            S2CPlay::UpdateEntityNBT(s2c::UpdateEntityNBT {
                entity_id: 41,
                tag: Some(tag.clone()),
            }),
            &mut events,
        )
        .expect("apply known entity NBT");
        assert_eq!(events, vec![ClientEvent::TerrainDownloaded]);
        let remote_after = state.entities.get(41).unwrap();
        assert_eq!(remote_after.entity, remote_before.entity);
        assert_eq!(remote_after.kind, remote_before.kind);
        assert_eq!(remote_after.server_pos, remote_before.server_pos);
        assert_eq!(remote_after.target_pos, remote_before.target_pos);
        assert_eq!(
            remote_after.interpolation_steps,
            remote_before.interpolation_steps
        );
        assert_eq!(remote_after.equipment, remote_before.equipment);

        let mut unknown = NbtCompound::new();
        unknown.insert("Marker", Nbt::String("also ignored".to_owned()));
        apply(
            &mut state,
            S2CPlay::UpdateEntityNBT(s2c::UpdateEntityNBT {
                entity_id: 999,
                tag: Some(unknown),
            }),
            &mut events,
        )
        .expect("apply unknown entity NBT");
        assert_eq!(events, vec![ClientEvent::TerrainDownloaded]);
        assert!(state.entities.get(999).is_none());

        let local_id = state.local.entity_id;
        apply(
            &mut state,
            S2CPlay::UpdateEntityNBT(s2c::UpdateEntityNBT {
                entity_id: local_id,
                tag: Some(tag),
            }),
            &mut events,
        )
        .expect("apply local entity NBT");
        assert_eq!(events, vec![ClientEvent::TerrainDownloaded]);
        assert_eq!(state.local, local_before);
    }

    #[test]
    fn initial_spawns_filter_metadata_after_resolving_the_entity_kind() {
        let mut state = ClientState::new();
        drain(
            &mut state,
            S2CPlay::SpawnPlayer(s2c::SpawnPlayer {
                entity_id: 11,
                player_uuid: uuid::Uuid::nil(),
                x: 0,
                y: 64 * 32,
                z: 0,
                yaw: Angle(0),
                pitch: Angle(0),
                current_item: 0,
                metadata: vec![
                    mc_protocol::MetadataEntry {
                        index: 10,
                        value: mc_protocol::MetadataValue::Byte(0x41),
                    },
                    mc_protocol::MetadataEntry {
                        index: 12,
                        value: mc_protocol::MetadataValue::Byte(1),
                    },
                ],
            }),
        );
        let player_entity = state.entities.get(11).expect("player").entity;
        let player_metadata = state
            .world
            .ecs
            .get::<&mc_world::entity::Metadata>(player_entity)
            .expect("player metadata");
        assert_eq!(
            player_metadata.get(10),
            Some(&mc_world::entity::MetadataValue::Byte(0x41))
        );
        assert_eq!(
            player_metadata.get(12),
            None,
            "Zombie slot rejected for Player"
        );
        drop(player_metadata);

        drain(
            &mut state,
            S2CPlay::SpawnMob(s2c::SpawnMob {
                entity_id: 12,
                entity_type: 54,
                x: 0,
                y: 64 * 32,
                z: 0,
                yaw: Angle(0),
                pitch: Angle(0),
                head_pitch: Angle(0),
                velocity_x: 0,
                velocity_y: 0,
                velocity_z: 0,
                metadata: vec![
                    mc_protocol::MetadataEntry {
                        index: 12,
                        value: mc_protocol::MetadataValue::Byte(1),
                    },
                    mc_protocol::MetadataEntry {
                        index: 10,
                        value: mc_protocol::MetadataValue::Byte(0x41),
                    },
                ],
            }),
        );
        let zombie_entity = state.entities.get(12).expect("zombie").entity;
        let zombie_metadata = state
            .world
            .ecs
            .get::<&mc_world::entity::Metadata>(zombie_entity)
            .expect("zombie metadata");
        assert_eq!(
            zombie_metadata.get(12),
            Some(&mc_world::entity::MetadataValue::Byte(1))
        );
        assert_eq!(
            zombie_metadata.get(10),
            None,
            "Player slot rejected for Zombie"
        );
    }

    /// S11's constructor keeps the wire position and XP value untouched, but seeds the newly
    /// created orb with its own random outward/upward motion. The exact seed is only a stable
    /// test surrogate for Java's unseeded `Math.random()` stream; the constructor formula and its
    /// float narrowing are the behavior under test.
    #[test]
    fn spawn_xp_orb_retains_wire_state_and_gets_constructor_motion() {
        let mut state = ClientState::new();
        let entity_id = 37;
        let packet = s2c::SpawnExperienceOrb {
            entity_id,
            x: 32 * 3,
            y: 32 * 70,
            z: 32 * -2,
            xp_value: 17,
        };
        let events = drain(&mut state, S2CPlay::SpawnExperienceOrb(packet));
        assert!(matches!(
            events.as_slice(),
            [ClientEvent::EntitySpawned {
                network_id: 37,
                kind: TrackedKind::Type(EntityType::XPOrb),
                ..
            }]
        ));

        let tracked = state.entities.get(entity_id).expect("XP orb is tracked");
        assert_eq!(tracked.server_pos, [96, 2240, -64]);
        assert_eq!(
            state
                .world
                .ecs
                .get::<&mc_world::entity::Position>(tracked.entity)
                .expect("spawn position")
                .0,
            DVec3::new(3.0, 70.0, -2.0),
            "constructor motion must not perturb the packet position"
        );
        assert_eq!(
            state
                .world
                .ecs
                .get::<&crate::app::entrender::XpValue>(tracked.entity)
                .expect("XP value")
                .0,
            17,
            "constructor motion must not replace the packet XP value"
        );

        let mut random = JavaRandom::new(i64::from(entity_id));
        let expected = xp_orb_spawn_motion(&mut random);
        let motion = state
            .world
            .ecs
            .get::<&mc_world::entity::Velocity>(tracked.entity)
            .expect("constructor motion")
            .0;
        assert_eq!(motion, expected);
        assert_ne!(motion, DVec3::ZERO, "zero-motion mutant must be observable");
    }

    /// The three constructor draws have Java's bounded distribution: X/Z are symmetric and Y is
    /// upward-only. Sweeping stable seeds supplies both horizontal signs and catches a formula
    /// that accidentally drops the random draw or reuses one axis.
    #[test]
    fn xp_orb_constructor_motion_has_java_bounds_and_horizontal_signs() {
        let mut saw_negative_x = false;
        let mut saw_positive_x = false;
        let mut saw_negative_z = false;
        let mut saw_positive_z = false;
        for seed in 0_i64..512 {
            let mut random = JavaRandom::new(seed);
            let motion = xp_orb_spawn_motion(&mut random);
            assert!(motion.x >= -0.200_000_02 && motion.x <= 0.200_000_02);
            assert!(motion.z >= -0.200_000_02 && motion.z <= 0.200_000_02);
            assert!(motion.y >= 0.0 && motion.y <= 0.400_000_01);
            saw_negative_x |= motion.x < 0.0;
            saw_positive_x |= motion.x > 0.0;
            saw_negative_z |= motion.z < 0.0;
            saw_positive_z |= motion.z > 0.0;
        }
        assert!(
            saw_negative_x && saw_positive_x,
            "X must scatter both directions"
        );
        assert!(
            saw_negative_z && saw_positive_z,
            "Z must scatter both directions"
        );
    }

    #[test]
    fn spawn_player_zero_current_item_clears_the_held_slot() {
        let mut state = ClientState::new();
        drain(
            &mut state,
            S2CPlay::SpawnPlayer(s2c::SpawnPlayer {
                entity_id: 12,
                player_uuid: uuid::Uuid::from_u128(0x5678),
                x: 0,
                y: 0,
                z: 0,
                yaw: Angle(0),
                pitch: Angle(0),
                current_item: 0,
                metadata: Vec::new(),
            }),
        );
        assert_eq!(
            state.entities.get(12).expect("tracked").equipment[0],
            None,
            "Java clears the fresh player's selected slot when currentItem is zero"
        );
    }

    /// java: type 70 throws in `Block.getStateById` for a metadata the block
    /// rejects (powered rail 6), so no falling block exists. Spawning it anyway let its tracker
    /// tick and renderer decode the same data through the panicking form: a crash one S0E could
    /// cause (found by the live-client packet fuzz).
    #[test]
    fn a_falling_block_with_rejected_metadata_never_spawns() {
        let golden_rail = i32::from(mc_data::block_by_name("golden_rail").unwrap().0);
        let spawn = |entity_id: i32, data: i32| {
            S2CPlay::SpawnObject(s2c::SpawnObject {
                entity_id,
                object_type: 70,
                x: 0,
                y: 0,
                z: 0,
                pitch: Angle(0),
                yaw: Angle(0),
                data,
                velocity: (data > 0).then_some((0, 0, 0)),
            })
        };
        let mut state = ClientState::new();
        let mut events = Vec::new();
        assert!(apply(&mut state, spawn(5, golden_rail | (6 << 12)), &mut events).is_err());
        assert!(events.is_empty());
        assert!(state.entities.get(5).is_none());

        drain(&mut state, spawn(6, 12));
        let tracked = state
            .entities
            .get(6)
            .expect("a legal state (sand) still spawns");
        assert_eq!(tracked.kind, TrackedKind::Type(EntityType::FallingSand));
    }

    #[test]
    fn spawn_object_resolves_the_object_id_space_not_the_entity_list_one() {
        let mut state = ClientState::new();
        // 70 is a falling block as an *object*; EntityList id 70 is nothing at all.
        drain(
            &mut state,
            S2CPlay::SpawnObject(s2c::SpawnObject {
                entity_id: 3,
                object_type: 70,
                x: 0,
                y: 0,
                z: 0,
                pitch: Angle(0),
                yaw: Angle(0),
                data: 1,
                velocity: Some((8000, 0, 0)),
            }),
        );
        let tracked = state.entities.get(3).expect("tracked");
        assert_eq!(tracked.kind, TrackedKind::Type(EntityType::FallingSand));
        let velocity = state
            .world
            .ecs
            .get::<&mc_world::entity::Velocity>(tracked.entity)
            .unwrap()
            .0;
        // This used to assert `1.0` with the comment "data > 0, so the
        // velocity applies", which is the misreading the finding is about: the falling block's
        // branch of `handleSpawnObject` ends in `packetIn.func_149002_g(0)` (`:403-407`) *before*
        // the `if (packetIn.func_149009_m() > 0)` guard at `:425`, so by the time the guard runs
        // `data` is zero and the velocity is skipped. `data` for a falling block is a block-state
        // id, not a "has velocity" flag, and it is almost always nonzero.
        assert_eq!(
            velocity.x, 0.0,
            "type 70 zeroes its data before the velocity guard"
        );
        // A type that is *not* on the zeroing list still takes the velocity — 60, an arrow, whose
        // `data` is the shooter's entity id.
        drain(
            &mut state,
            S2CPlay::SpawnObject(s2c::SpawnObject {
                entity_id: 4,
                object_type: 60,
                x: 0,
                y: 0,
                z: 0,
                pitch: Angle(0),
                yaw: Angle(0),
                data: 1,
                velocity: Some((8000, 0, 0)),
            }),
        );
        let arrow = state.entities.get(4).expect("tracked");
        let velocity = state
            .world
            .ecs
            .get::<&mc_world::entity::Velocity>(arrow.entity)
            .unwrap()
            .0;
        assert_eq!(velocity.x, 1.0, "data > 0 and type 60 is not zeroed");
    }

    /// `EntityArrow.setVelocity` discards `S0E`'s quantised look angles at spawn and derives
    /// them from motion, while the otherwise identical snowball path retains the wire angles.
    #[test]
    fn an_arrow_spawn_uses_motion_look_angles_but_a_snowball_keeps_the_wire_angles() {
        let mut state = ClientState::new();
        let wire_yaw = Angle::from_degrees(-135.0);
        let wire_pitch = Angle::from_degrees(-45.0);
        let velocity = (4321, -2345, 6789);

        drain(
            &mut state,
            S2CPlay::SpawnObject(s2c::SpawnObject {
                entity_id: 41,
                object_type: 60, // arrow
                x: 0,
                y: 0,
                z: 0,
                pitch: wire_pitch,
                yaw: wire_yaw,
                data: 41,
                velocity: Some(velocity),
            }),
        );
        let arrow = state.entities.get(41).expect("arrow tracked");
        let (arrow_yaw, arrow_pitch) = {
            let rotation = state
                .world
                .ecs
                .get::<&mc_world::entity::Rotation>(arrow.entity)
                .expect("arrow rotation");
            (rotation.yaw, rotation.pitch)
        };
        let (expected_yaw, expected_pitch) = arrow_spawn_rotation(DVec3::new(
            f64::from(velocity.0) / VELOCITY_SCALE,
            f64::from(velocity.1) / VELOCITY_SCALE,
            f64::from(velocity.2) / VELOCITY_SCALE,
        ));
        assert_eq!(arrow_yaw.to_bits(), expected_yaw.to_bits());
        assert_eq!(arrow_pitch.to_bits(), expected_pitch.to_bits());
        assert_ne!(arrow_yaw.to_bits(), wire_yaw.to_degrees().to_bits());
        assert_ne!(arrow_pitch.to_bits(), wire_pitch.to_degrees().to_bits());

        // Negative control: `EntityThrowable` inherits Entity.setVelocity, which does not
        // recalculate yaw/pitch.  A snowball has the same wire shape and velocity guard.
        drain(
            &mut state,
            S2CPlay::SpawnObject(s2c::SpawnObject {
                entity_id: 42,
                object_type: 61, // snowball
                x: 0,
                y: 0,
                z: 0,
                pitch: wire_pitch,
                yaw: wire_yaw,
                data: 42,
                velocity: Some(velocity),
            }),
        );
        let snowball = state.entities.get(42).expect("snowball tracked");
        let rotation = state
            .world
            .ecs
            .get::<&mc_world::entity::Rotation>(snowball.entity)
            .expect("snowball rotation");
        assert_eq!(rotation.yaw.to_bits(), wire_yaw.to_degrees().to_bits());
        assert_eq!(rotation.pitch.to_bits(), wire_pitch.to_degrees().to_bits());
    }

    /// The three `EntityFireball` object branches consume S0E's speed fields as normalized
    /// acceleration, whereas an ordinary thrown object keeps those fields as motion.
    #[test]
    fn fireball_spawns_preserve_normalized_acceleration_and_non_fireballs_do_not() {
        let mut state = ClientState::new();
        let speed = (1600, -3200, 6400);
        let expected = fireball_spawn_acceleration(Some(speed)).0;

        for (network_id, object_type, kind) in [
            (61, 63, EntityType::Fireball),
            (62, 64, EntityType::SmallFireball),
            (63, 66, EntityType::WitherSkull),
        ] {
            drain(
                &mut state,
                S2CPlay::SpawnObject(s2c::SpawnObject {
                    entity_id: network_id,
                    object_type,
                    x: 0,
                    y: 0,
                    z: 0,
                    pitch: Angle(0),
                    yaw: Angle(0),
                    data: 7, // a shooter id: makes S0E carry the three speed shorts
                    velocity: Some(speed),
                }),
            );
            let entity = state
                .entities
                .get(network_id)
                .expect("fireball tracked")
                .entity;
            assert_eq!(
                state.entities.get(network_id).unwrap().kind,
                TrackedKind::Type(kind)
            );
            let acceleration = state
                .world
                .ecs
                .get::<&mc_world::Acceleration>(entity)
                .expect("fireball acceleration")
                .0;
            assert_eq!(acceleration.x.to_bits(), expected.x.to_bits());
            assert_eq!(acceleration.y.to_bits(), expected.y.to_bits());
            assert_eq!(acceleration.z.to_bits(), expected.z.to_bits());
            let velocity = state
                .world
                .ecs
                .get::<&mc_world::entity::Velocity>(entity)
                .expect("fireball velocity")
                .0;
            assert_eq!(
                velocity,
                DVec3::ZERO,
                "setVelocity is skipped after data zeroing"
            );
        }

        // An unowned fireball has data 0, so S0E has no speed fields. The Java constructor still
        // divides the all-zero vector by `sqrt_double(0)`, preserving three NaN fields.
        drain(
            &mut state,
            S2CPlay::SpawnObject(s2c::SpawnObject {
                entity_id: 64,
                object_type: 63,
                x: 0,
                y: 0,
                z: 0,
                pitch: Angle(0),
                yaw: Angle(0),
                data: 0,
                velocity: None,
            }),
        );
        let unowned = state.entities.get(64).expect("unowned fireball").entity;
        let acceleration = state
            .world
            .ecs
            .get::<&mc_world::Acceleration>(unowned)
            .expect("unowned acceleration")
            .0;
        assert!(acceleration.x.is_nan() && acceleration.y.is_nan() && acceleration.z.is_nan());

        // Negative control: a snowball receives the same optional triples through the ordinary
        // setVelocity path and must not gain a fireball-only acceleration component.
        drain(
            &mut state,
            S2CPlay::SpawnObject(s2c::SpawnObject {
                entity_id: 65,
                object_type: 61,
                x: 0,
                y: 0,
                z: 0,
                pitch: Angle(0),
                yaw: Angle(0),
                data: 7,
                velocity: Some(speed),
            }),
        );
        let snowball = state.entities.get(65).expect("snowball tracked").entity;
        assert!(state
            .world
            .ecs
            .get::<&mc_world::Acceleration>(snowball)
            .is_err());
        assert_eq!(
            state
                .world
                .ecs
                .get::<&mc_world::entity::Velocity>(snowball)
                .expect("snowball velocity")
                .0,
            DVec3::new(0.2, -0.4, 0.8),
        );
    }

    /// java: minecart variant comes from the object-data field, via `EnumMinecartType`.
    #[test]
    fn minecart_variants_come_from_the_object_data_field() {
        assert_eq!(minecart_type(0), EntityType::MinecartRideable);
        assert_eq!(minecart_type(1), EntityType::MinecartChest);
        assert_eq!(minecart_type(6), EntityType::MinecartCommandBlock);
        assert_eq!(minecart_type(99), EntityType::MinecartRideable);
    }

    /// `S0E.data` is a constructor argument, not one generic "extra data" field. This pins
    /// the three values whose consumers live beyond spawn: item-frame facing, potion damage,
    /// and an arrow's resolved living shooter.
    #[test]
    fn spawn_object_data_keeps_typed_frame_potion_and_arrow_state_with_java_sign_semantics() {
        let mut state = ClientState::new();
        // Make a living owner that is already in the world's id map, the same precondition the
        // Java `getEntityByID(data) instanceof EntityLivingBase` check needs.
        state.entities.spawn(
            &mut state.world,
            900,
            TrackedKind::Type(EntityType::Zombie),
            None,
            [0, 0, 0],
            0.0,
            0.0,
            DVec3::ZERO,
            mc_world::Metadata::new(),
        );

        // EnumFacing.getHorizontal uses S-W-N-E order and a signed remainder then abs: zero is
        // South while -1 is West, not a rejected/down-facing value.
        drain(&mut state, spawn_object(41, 71, 0, Some((8000, 0, 0))));
        drain(&mut state, spawn_object(42, 71, -1, Some((8000, 0, 0))));
        let zero_frame = state.entities.get(41).unwrap().entity;
        let negative_frame = state.entities.get(42).unwrap().entity;
        assert_eq!(
            *state
                .world
                .ecs
                .get::<&mc_world::ItemFrameFacing>(zero_frame)
                .unwrap(),
            mc_world::ItemFrameFacing(mc_math::Facing::South)
        );
        assert_eq!(
            *state
                .world
                .ecs
                .get::<&mc_world::ItemFrameFacing>(negative_frame)
                .unwrap(),
            mc_world::ItemFrameFacing(mc_math::Facing::West)
        );
        assert_eq!(
            state
                .world
                .ecs
                .get::<&mc_world::Velocity>(negative_frame)
                .unwrap()
                .0,
            DVec3::ZERO,
            "item-frame data is zeroed before the shared velocity guard"
        );

        // EntityPotion hands packet data to ItemStack verbatim, even at zero or negative, then
        // clears it before the shared velocity path.
        drain(&mut state, spawn_object(43, 73, 0, Some((8000, 0, 0))));
        drain(
            &mut state,
            spawn_object(44, 73, -16_385, Some((8000, 0, 0))),
        );
        let zero_potion = state.entities.get(43).unwrap().entity;
        let negative_potion = state.entities.get(44).unwrap().entity;
        assert_eq!(
            *state
                .world
                .ecs
                .get::<&mc_world::SplashPotionDamage>(zero_potion)
                .unwrap(),
            mc_world::SplashPotionDamage(0)
        );
        assert_eq!(
            *state
                .world
                .ecs
                .get::<&mc_world::SplashPotionDamage>(negative_potion)
                .unwrap(),
            mc_world::SplashPotionDamage(-16_385)
        );
        assert_eq!(
            state
                .world
                .ecs
                .get::<&mc_world::Velocity>(negative_potion)
                .unwrap()
                .0,
            DVec3::ZERO
        );

        // Positive data selects a *currently present living* owner. Missing, self/nonliving,
        // zero and negative values all leave shootingEntity null; only positive data gets the
        // ordinary velocity call.
        drain(&mut state, spawn_object(45, 60, 900, Some((8000, 0, 0))));
        drain(&mut state, spawn_object(46, 60, 901, Some((8000, 0, 0))));
        drain(&mut state, spawn_object(47, 60, 0, Some((8000, 0, 0))));
        drain(&mut state, spawn_object(48, 60, -1, Some((8000, 0, 0))));
        let owned_arrow = state.entities.get(45).unwrap().entity;
        let missing_owner_arrow = state.entities.get(46).unwrap().entity;
        let zero_arrow = state.entities.get(47).unwrap().entity;
        let negative_arrow = state.entities.get(48).unwrap().entity;
        assert_eq!(
            *state
                .world
                .ecs
                .get::<&mc_world::ArrowShooter>(owned_arrow)
                .unwrap(),
            mc_world::ArrowShooter(900)
        );
        for arrow in [missing_owner_arrow, zero_arrow, negative_arrow] {
            assert!(state
                .world
                .ecs
                .get::<&mc_world::ArrowShooter>(arrow)
                .is_err());
        }
        assert_eq!(
            state
                .world
                .ecs
                .get::<&mc_world::Velocity>(owned_arrow)
                .unwrap()
                .0,
            DVec3::new(1.0, 0.0, 0.0)
        );
        for arrow in [zero_arrow, negative_arrow] {
            assert_eq!(
                state.world.ecs.get::<&mc_world::Velocity>(arrow).unwrap().0,
                DVec3::ZERO
            );
        }
    }

    #[test]
    fn fishing_bobbers_only_spawn_for_player_owners_and_zero_the_spawn_velocity() {
        let mut state = ClientState::new();
        state.local.entity_id = 7;
        let events = drain(
            &mut state,
            S2CPlay::SpawnObject(s2c::SpawnObject {
                entity_id: 4,
                object_type: 90,
                x: 32,
                y: 0,
                z: 0,
                pitch: Angle(0),
                yaw: Angle(0),
                data: 7,
                velocity: Some((8000, 0, 0)),
            }),
        );
        assert!(matches!(
            events[0],
            ClientEvent::EntitySpawned {
                network_id: 4,
                kind: TrackedKind::FishingBobber,
                ..
            }
        ));
        let bobber = state.entities.get(4).expect("local-player bobber");
        assert_eq!(bobber.kind, TrackedKind::FishingBobber);
        assert_eq!(
            *state
                .world
                .ecs
                .get::<&mc_world::EntityKind>(bobber.entity)
                .unwrap(),
            mc_world::EntityKind::FishingBobber
        );
        assert_eq!(
            state
                .world
                .ecs
                .get::<&mc_world::Velocity>(bobber.entity)
                .unwrap()
                .0,
            DVec3::ZERO,
            "type 90 clears data before the generic setVelocity guard"
        );
        assert_eq!(
            state
                .world
                .ecs
                .get::<&crate::app::entrender::ObjectData>(bobber.entity)
                .unwrap()
                .0,
            7,
            "the player id remains available for the future RenderFish line"
        );

        // The owner may also be a remote `EntityPlayer`, which is the tracked-player variant;
        // the local-player fast path above must not accidentally be the only accepted one.
        state.entities.spawn(
            &mut state.world,
            9,
            TrackedKind::Player,
            Some(uuid::Uuid::nil()),
            [0, 0, 0],
            0.0,
            0.0,
            DVec3::ZERO,
            mc_world::entity::Metadata::new(),
        );
        let events = drain(
            &mut state,
            S2CPlay::SpawnObject(s2c::SpawnObject {
                entity_id: 6,
                object_type: 90,
                x: 0,
                y: 0,
                z: 0,
                pitch: Angle(0),
                yaw: Angle(0),
                data: 9,
                velocity: None,
            }),
        );
        assert!(matches!(
            events[0],
            ClientEvent::EntitySpawned {
                network_id: 6,
                kind: TrackedKind::FishingBobber,
                ..
            }
        ));

        // Negative control: type 90's word is an owner id, not merely nonzero data.  A zombie
        // is present but fails Java's `instanceof EntityPlayer` gate, so no hook is constructed.
        drain(
            &mut state,
            S2CPlay::SpawnMob(s2c::SpawnMob {
                entity_id: 8,
                entity_type: 54,
                x: 0,
                y: 0,
                z: 0,
                yaw: Angle(0),
                pitch: Angle(0),
                head_pitch: Angle(0),
                velocity_x: 0,
                velocity_y: 0,
                velocity_z: 0,
                metadata: Vec::new(),
            }),
        );
        let events = drain(
            &mut state,
            S2CPlay::SpawnObject(s2c::SpawnObject {
                entity_id: 5,
                object_type: 90,
                x: 0,
                y: 0,
                z: 0,
                pitch: Angle(0),
                yaw: Angle(0),
                data: 8,
                velocity: Some((8000, 0, 0)),
            }),
        );
        assert!(matches!(
            events[0],
            ClientEvent::EntitySpawnIgnored { network_id: 5, .. }
        ));
        assert!(state.entities.get(5).is_none());
    }

    #[test]
    fn an_unknown_mob_id_is_ignored_rather_than_fatal() {
        let mut state = ClientState::new();
        let events = drain(
            &mut state,
            S2CPlay::SpawnMob(s2c::SpawnMob {
                entity_id: 9,
                entity_type: 3, // not an EntityList id
                x: 0,
                y: 0,
                z: 0,
                yaw: Angle(0),
                pitch: Angle(0),
                head_pitch: Angle(0),
                velocity_x: 0,
                velocity_y: 0,
                velocity_z: 0,
                metadata: Vec::new(),
            }),
        );
        assert!(matches!(
            events[0],
            ClientEvent::EntitySpawnIgnored { network_id: 9, .. }
        ));
    }

    /// `handleChangeGameState` rounds its float payload with
    /// `MathHelper.floor_float(value + 0.5F)` before `GameType.getByID`.  In particular, the
    /// final half-open interval for Adventure ends at `2.5`, where Spectator starts; treating
    /// the payload as a byte would make the latter endpoint Adventure instead.
    ///
    /// The four values cover every normal game mode and the `2.49`/`2.5` pair is the negative
    /// control for removing Java's round-to-nearest step.
    #[test]
    fn game_state_reason_three_rounds_into_every_game_mode() {
        for (value, expected) in [
            (0.0, s2c::GameMode::Survival),
            (1.0, s2c::GameMode::Creative),
            (2.49, s2c::GameMode::Adventure),
            (2.5, s2c::GameMode::Spectator),
        ] {
            let mut state = ClientState::new();
            let events = drain(
                &mut state,
                S2CPlay::ChangeGameState(s2c::ChangeGameState {
                    reason: s2c::GameStateReason::ChangeGameMode,
                    value,
                }),
            );
            assert_eq!(state.local.game_mode, expected, "payload {value}");
            assert!(
                matches!(events.as_slice(), [ClientEvent::GameModeChanged(mode)] if *mode == expected)
            );
        }
    }

    #[test]
    fn weather_reasons_reach_the_environment() {
        let mut state = ClientState::new();
        let events = drain(
            &mut state,
            S2CPlay::ChangeGameState(s2c::ChangeGameState {
                reason: s2c::GameStateReason::BeginRaining,
                value: 0.0,
            }),
        );
        assert!(matches!(events[0], ClientEvent::WeatherChanged));
        assert!(state.env.is_raining());
    }

    /// `S2BPacketChangeGameState.MESSAGE_NAMES` has one entry, at index zero. Vanilla emits it
    /// before switching on the reason, so the invalid-bed packet is a chat event rather than an
    /// opaque GameState event; its float is deliberately ignored.
    #[test]
    fn game_state_invalid_bed_emits_the_vanilla_translation() {
        let mut state = ClientState::new();
        let events = drain(
            &mut state,
            S2CPlay::ChangeGameState(s2c::ChangeGameState {
                reason: s2c::GameStateReason::InvalidBed,
                value: f32::NAN,
            }),
        );
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            ClientEvent::Chat { component, position: ChatPosition::Chat }
                if **component == ChatComponent::translate("tile.bed.notValid", vec![])
        ));
    }

    /// `WorldSettings.GameType` declares `NOT_SET(-1, "")`, and `getByID` returns the matching
    /// enum before its unknown-id `SURVIVAL` fallback. S2B reason 3 rounds the float first, so
    /// `-1.0F` is a live dispatch case that must remain `NotSet`; changing the shared mapper to
    /// treat every negative id as Survival would reproduce the audit's false finding.
    /// java: `WorldSettings.java:135-141,203-214` and `NetHandlerPlayClient.java:1362-1383`.
    #[test]
    fn game_state_reason_three_preserves_java_explicit_not_set_id() {
        let mut state = ClientState::new();
        let events = drain(
            &mut state,
            S2CPlay::ChangeGameState(s2c::ChangeGameState {
                reason: s2c::GameStateReason::ChangeGameMode,
                value: -1.0,
            }),
        );
        assert_eq!(state.local.game_mode, s2c::GameMode::NotSet);
        assert!(matches!(
            events.as_slice(),
            [ClientEvent::GameModeChanged(s2c::GameMode::NotSet)]
        ));
    }

    /// The remaining presentation-only reasons are intentionally forwarded losslessly. In
    /// particular, demo-message selection in vanilla compares the float itself (not its rounded
    /// game-mode-style integer), so a host must be able to distinguish 101 from its neighbours.
    #[test]
    fn game_state_presentation_reasons_preserve_the_exact_float_for_the_host() {
        let cases = [
            (s2c::GameStateReason::EnterCredits, -0.0_f32),
            (s2c::GameStateReason::DemoMessage, 101.000_01_f32),
            (s2c::GameStateReason::ArrowHittingPlayer, f32::NAN),
            (s2c::GameStateReason::PlayMobAppearance, 1.0_f32),
        ];

        for (reason, value) in cases {
            let mut state = ClientState::new();
            let events = drain(
                &mut state,
                S2CPlay::ChangeGameState(s2c::ChangeGameState { reason, value }),
            );
            assert!(matches!(
                &events[..],
                [ClientEvent::GameState { reason: actual_reason, value: actual_value }]
                    if *actual_reason == reason && actual_value.to_bits() == value.to_bits()
            ));
        }
    }

    #[test]
    fn maps_dispatch_updates_the_world_record_and_keeps_packet_order() {
        let mut state = ClientState::new();
        let events = drain(
            &mut state,
            S2CPlay::Maps(s2c::Maps {
                map_id: -7,
                scale: 2,
                icons: vec![s2c::MapIcon {
                    icon_type: 1,
                    x: -12,
                    z: 34,
                    direction: 9,
                }],
                update: Some(s2c::MapPixelUpdate {
                    width: 2,
                    height: 2,
                    min_x: 4,
                    min_y: 5,
                    data: vec![10, 11, 12, 13],
                }),
            }),
        );

        let map = state.world.map_data(-7).expect("S34 creates map_-7");
        assert_eq!(map.scale, 2);
        assert_eq!(map.colors[4 + 5 * mc_world::MAP_SIZE], 10);
        assert_eq!(map.colors[5 + 5 * mc_world::MAP_SIZE], 11);
        assert_eq!(map.colors[4 + 6 * mc_world::MAP_SIZE], 12);
        assert_eq!(map.colors[5 + 6 * mc_world::MAP_SIZE], 13);
        assert_eq!(map.decorations.len(), 1);
        assert_eq!(map.decorations[0].key, "icon-0");
        assert_eq!(map.decorations[0].icon_type, 1);
        assert!(matches!(events.as_slice(), [ClientEvent::Maps(packet)]
            if packet.map_id == -7));

        // Java replaces scale/decorations on an icon-only packet while retaining the prior
        // pixels. This second packet also proves dispatch is not merely forwarding an event.
        drain(
            &mut state,
            S2CPlay::Maps(s2c::Maps {
                map_id: -7,
                scale: 4,
                icons: vec![],
                update: None,
            }),
        );
        let map = state.world.map_data(-7).expect("map record remains live");
        assert_eq!(map.scale, 4);
        assert!(map.decorations.is_empty());
        assert_eq!(map.colors[5 + 6 * mc_world::MAP_SIZE], 13);
    }

    #[test]
    fn malformed_map_rectangle_does_not_create_or_partially_replace_state() {
        let mut state = ClientState::new();
        drain(
            &mut state,
            S2CPlay::Maps(s2c::Maps {
                map_id: 91,
                scale: 1,
                icons: vec![],
                update: Some(s2c::MapPixelUpdate {
                    width: 2,
                    height: 2,
                    min_x: 127,
                    min_y: 127,
                    data: vec![1, 2, 3, 4],
                }),
            }),
        );
        assert!(state.world.map_data(91).is_none());
    }

    #[test]
    fn an_out_of_range_held_slot_is_ignored() {
        let mut state = ClientState::new();
        state.local.held_slot = 4;
        let events = drain(
            &mut state,
            S2CPlay::HeldItemChange(s2c::HeldItemChange { slot: 9 }),
        );
        assert!(events.is_empty());
        assert_eq!(state.local.held_slot, 4);

        drain(
            &mut state,
            S2CPlay::HeldItemChange(s2c::HeldItemChange { slot: 8 }),
        );
        assert_eq!(state.local.held_slot, 8);
    }

    #[test]
    fn a_play_state_disconnect_becomes_a_typed_error() {
        let mut state = ClientState::new();
        let mut events = Vec::new();
        let reason = r#"{"text":"bye"}"#;
        match apply(
            &mut state,
            S2CPlay::Disconnect(s2c::Disconnect {
                reason: reason.into(),
            }),
            &mut events,
        ) {
            Err(ClientError::Disconnected { reason: got }) => assert_eq!(got, reason),
            other => panic!("expected Disconnected, got {other:?}"),
        }
    }

    #[test]
    fn malformed_chat_json_is_an_error_not_a_panic() {
        let mut state = ClientState::new();
        let mut events = Vec::new();
        assert!(matches!(
            apply(
                &mut state,
                S2CPlay::Chat(s2c::Chat {
                    component: "{oh no".into(),
                    position: 0,
                }),
                &mut events,
            ),
            Err(ClientError::Chat(_))
        ));
    }

    /// S05 and S41 both mutate the world-scoped state at the packet boundary and then publish
    /// their notification in the same order the wire delivered them.  The static dead-wire
    /// audit listed these fields as write-only; this exercises the live `ClientState` path and
    /// gives callers a stable event sequence when a proxy sends several updates in one burst.
    #[test]
    fn spawn_position_and_difficulty_keep_live_state_and_event_order() {
        let mut state = ClientState::new();
        let first_spawn = BlockPos::new(i32::MIN, 0, i32::MAX);
        let second_spawn = BlockPos::new(-17, 255, 23);

        let mut events = Vec::new();
        apply(
            &mut state,
            S2CPlay::SpawnPosition(s2c::SpawnPosition {
                position: first_spawn,
            }),
            &mut events,
        )
        .expect("S05 dispatch");
        apply(
            &mut state,
            S2CPlay::ServerDifficulty(s2c::ServerDifficulty {
                difficulty: s2c::ServerDifficultyLevel::Hard,
            }),
            &mut events,
        )
        .expect("S41 dispatch");
        apply(
            &mut state,
            S2CPlay::SpawnPosition(s2c::SpawnPosition {
                position: second_spawn,
            }),
            &mut events,
        )
        .expect("second S05 dispatch");

        assert_eq!(state.spawn_position(), second_spawn);
        assert_eq!(state.env.difficulty, s2c::ServerDifficultyLevel::Hard);
        assert_eq!(
            events,
            vec![
                ClientEvent::SpawnPositionChanged(first_spawn),
                ClientEvent::DifficultyChanged(s2c::ServerDifficultyLevel::Hard),
                ClientEvent::SpawnPositionChanged(second_spawn),
            ],
            "state notifications retain packet order rather than coalescing a burst"
        );
    }

    /// `S19` is an extensible byte at the protocol edge.  Vanilla ignores an opcode no entity
    /// class understands; it must not accidentally turn an unknown local status into hurt or
    /// death state, while the packet event remains available to a future consumer.
    #[test]
    fn unknown_local_entity_status_is_forwarded_without_mutating_health() {
        let mut state = ClientState::new();
        drain(&mut state, join());
        state.local.health = 12.0;
        state.local.hurt_time = 4;
        state.local.hurt_resistant_time = 9;
        let local_id = state.local.entity_id;

        let events = drain(
            &mut state,
            S2CPlay::EntityStatus(s2c::EntityStatus {
                entity_id: local_id,
                opcode: 127,
            }),
        );

        assert_eq!(state.local.health, 12.0);
        assert_eq!(state.local.hurt_time, 4);
        assert_eq!(state.local.hurt_resistant_time, 9);
        assert!(matches!(
            events.as_slice(),
            [ClientEvent::EntityStatus {
                network_id: 77,
                opcode: 127
            }]
        ));
    }

    #[test]
    fn difficulty_ids_wrap_the_way_enum_difficulty_does() {
        let expected = [
            s2c::ServerDifficultyLevel::Peaceful,
            s2c::ServerDifficultyLevel::Easy,
            s2c::ServerDifficultyLevel::Normal,
            s2c::ServerDifficultyLevel::Hard,
        ];
        for id in 0..=u8::MAX {
            assert_eq!(
                difficulty_from_id(id),
                expected[usize::from(id % 4)],
                "malformed/extended difficulty id {id} still follows Java's modulo lookup"
            );
        }
    }

    /// `S3FPacketCustomPayload` is an opaque transport at dispatch time: the game-thread
    /// consumer owns channel-specific parsing and must see every channel in wire order,
    /// including malformed bytes that only that consumer can classify.
    #[test]
    fn custom_payload_channels_are_forwarded_opaque_in_wire_order() {
        let mut state = ClientState::new();
        let mut events = Vec::new();
        for (channel, data) in [
            ("MC|TrList", vec![0, 0, 0, 7, 0]),
            ("MC|Brand", vec![0xff, 0x00]),
            ("MC|BOpen", vec![0xde, 0xad]),
        ] {
            apply(
                &mut state,
                S2CPlay::PluginMessage(s2c::PluginMessage {
                    channel: channel.to_owned(),
                    data: data.clone(),
                }),
                &mut events,
            )
            .expect("custom payload dispatch");
        }
        assert_eq!(
            events,
            vec![
                ClientEvent::PluginMessage {
                    channel: "MC|TrList".into(),
                    data: vec![0, 0, 0, 7, 0],
                },
                ClientEvent::PluginMessage {
                    channel: "MC|Brand".into(),
                    data: vec![0xff, 0x00],
                },
                ClientEvent::PluginMessage {
                    channel: "MC|BOpen".into(),
                    data: vec![0xde, 0xad],
                },
            ]
        );
    }
}
