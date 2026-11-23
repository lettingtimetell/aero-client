//! Full-column chunk payloads unpacked off the game thread, installed on it in packet order.
//!
//! #15 / C1-4 / NET-1. A join, respawn or teleport delivers hundreds of `S21`/`S26` columns, and
//! unpacking each one — the per-cell decode, a 12 KB section allocation per present section and
//! the heightmap descent — used to run inside `Session::poll_all` on the game thread, landing as
//! multi-millisecond frames right after the terrain screen closes.
//!
//! The network task now wraps every full-column chunk packet in a [`PrebuildingChunkPacket`] and
//! queues the unpack on one background thread **as the packet is decoded**. The wrapper travels
//! through the same inbound queue, in the same position the plain packet would have. When the
//! game thread reaches it, [`PrebuildingChunkPacket::take`] hands back the packet plus:
//!
//! * the unpacked columns, if the background thread finished first — the game thread then only
//!   installs them (`World::install_full_column`), with every other effect of `apply` unchanged;
//! * nothing, if the background thread has not finished it — the game thread claims it and runs
//!   today's `apply` inline, so a busy, starved or missing worker costs what it did before (plus,
//!   only while the worker is mid-unpack, one copy of the packet, since the worker still reads it).
//!
//! The game thread **never waits** for the worker: both sides hold the slot lock only for O(1)
//! moves, never across an unpack, so a worker preempted by the OS cannot stall a frame.
//!
//! Parity: inbound apply order is unchanged (same queue, same slot, installed inside the same
//! `apply` step); nothing outbound is generated, delayed or reordered — the network task only
//! adds an `Arc` and a channel send per chunk packet, so the KeepAlive echo is not delayed; the
//! world is bit-identical (same function, same bytes; an `S21` built with a sky-light layout the
//! world no longer has, a failed build or a panicking one all fall back to today's path, which
//! reproduces today's error or panic exactly).

use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use mc_protocol::play::S2CPlay;
use mc_world::Chunk;

/// A full-column chunk packet whose columns a background thread may already be unpacking.
pub struct PrebuildingChunkPacket {
    /// The sky-light payload layout the unpack used: `S26`'s own flag, or for `S21` the
    /// dimension the network task last saw in `JoinGame`/`Respawn`. An `S21` is installed only
    /// if the world agrees when its turn comes.
    assumed_sky: bool,
    slot: Mutex<Slot>,
}

enum Slot {
    /// Not unpacked yet (queued, or being read by the background thread through its own `Arc`).
    /// The packet stays here so the game thread can always take it immediately.
    Pending(Arc<S2CPlay>),
    /// Unpacked by the background thread: one column per entry in packet order, or `None` when
    /// the unpack failed (the game thread then reproduces the failure on today's path). The
    /// worker has dropped its `Arc` before publishing, so taking the packet does not copy it.
    Built(Arc<S2CPlay>, Option<Vec<Chunk>>),
    /// Handed to the game thread.
    Taken,
}

impl std::fmt::Debug for PrebuildingChunkPacket {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PrebuildingChunkPacket")
            .field("assumed_sky", &self.assumed_sky)
            .finish_non_exhaustive()
    }
}

/// The background unpack thread, started on first use. `None` if it could not be started, in
/// which case chunk packets travel unwrapped exactly as before.
fn unpack_worker() -> Option<&'static Mutex<Sender<Arc<PrebuildingChunkPacket>>>> {
    static WORKER: OnceLock<Option<Mutex<Sender<Arc<PrebuildingChunkPacket>>>>> = OnceLock::new();
    WORKER
        .get_or_init(|| {
            let (sender, jobs) = channel::<Arc<PrebuildingChunkPacket>>();
            std::thread::Builder::new()
                .name("mc-chunk-unpack".to_owned())
                .spawn(move || {
                    for job in jobs {
                        // Only this queue still holds it: the session is gone, skip the work.
                        if Arc::strong_count(&job) > 1 {
                            job.unpack();
                        }
                    }
                })
                .ok()?;
            Some(Mutex::new(sender))
        })
        .as_ref()
}

/// The columns of a full-column chunk packet, unpacked with `has_sky_light`'s layout. `None` for
/// any other packet and for any payload `fill_from_packet` rejects.
fn unpack_columns(packet: &S2CPlay, has_sky_light: bool) -> Option<Vec<Chunk>> {
    match packet {
        S2CPlay::ChunkData(chunk) if chunk.ground_up_continuous && chunk.primary_bit_mask != 0 => {
            Chunk::from_full_column_packet(chunk.primary_bit_mask, has_sky_light, &chunk.data)
                .ok()
                .map(|column| vec![column])
        }
        S2CPlay::MapChunkBulk(bulk) => bulk
            .columns
            .iter()
            .map(|column| {
                Chunk::from_full_column_packet(column.primary_bit_mask, has_sky_light, &column.data)
                    .ok()
            })
            .collect(),
        _ => None,
    }
}

impl PrebuildingChunkPacket {
    /// Wrap `packet` and queue its unpack if it is a full-column chunk packet (`S21` with
    /// `ground_up_continuous` and a non-zero bitmask, or any `S26`); otherwise hand it back.
    ///
    /// `dimension_has_sky` is the sky-light state of the dimension the stream is in, as far as
    /// the caller has seen (`JoinGame`/`Respawn`); only an `S21` uses it.
    ///
    /// # Errors
    ///
    /// Gives `packet` back, untouched, when it is not one of those shapes or no background thread
    /// is available; send it as a plain packet then.
    // Handing the packet back by value is the point: boxing it would add an allocation on the
    // path every non-chunk packet takes, to save a move.
    #[allow(clippy::result_large_err)]
    pub fn spawn(packet: S2CPlay, dimension_has_sky: bool) -> Result<Arc<Self>, S2CPlay> {
        let assumed_sky = match &packet {
            S2CPlay::ChunkData(chunk)
                if chunk.ground_up_continuous && chunk.primary_bit_mask != 0 =>
            {
                Some(dimension_has_sky)
            }
            S2CPlay::MapChunkBulk(bulk) => Some(bulk.has_sky_light),
            _ => None,
        };
        let Some(assumed_sky) = assumed_sky else {
            return Err(packet);
        };
        let Some(worker) = unpack_worker() else {
            return Err(packet);
        };
        let job = Arc::new(Self {
            assumed_sky,
            slot: Mutex::new(Slot::Pending(Arc::new(packet))),
        });
        let sent = worker
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .send(Arc::clone(&job))
            .is_ok();
        if sent {
            return Ok(job);
        }
        let unsent = std::mem::replace(
            &mut *job.slot.lock().unwrap_or_else(PoisonError::into_inner),
            Slot::Taken,
        );
        match unsent {
            Slot::Pending(packet) | Slot::Built(packet, _) => Err(Arc::unwrap_or_clone(packet)),
            Slot::Taken => unreachable!("an unsent job cannot have been taken"),
        }
    }

    /// The sky-light layout the unpack used; see [`crate::sync::dispatch::apply_prebuilt_chunks`].
    #[must_use]
    pub fn assumed_sky(&self) -> bool {
        self.assumed_sky
    }

    /// Background thread: unpack the columns unless the game thread already claimed the packet.
    /// The slot lock is held only to clone the `Arc` and to publish, never during the unpack.
    fn unpack(&self) {
        let packet = match &*self.slot.lock().unwrap_or_else(PoisonError::into_inner) {
            Slot::Pending(packet) => Arc::clone(packet),
            Slot::Built(..) | Slot::Taken => return,
        };
        // A panic here would otherwise strand the slot; publish "no columns" instead, and the
        // game thread's own `apply` behaves exactly as it always did.
        // Contained, so the panic hook does not record this recovered panic as the game's crash
        // (a real one repeats on the game thread, with that thread's context).
        let columns = mc_core::panic_scope::contain(|| unpack_columns(&packet, self.assumed_sky))
            .ok()
            .flatten();
        // Released before publishing, so a game thread that sees `Built` owns the only reference.
        drop(packet);
        let mut slot = self.slot.lock().unwrap_or_else(PoisonError::into_inner);
        if matches!(*slot, Slot::Pending(_)) {
            let Slot::Pending(packet) = std::mem::replace(&mut *slot, Slot::Taken) else {
                unreachable!("checked under the same lock");
            };
            *slot = Slot::Built(packet, columns);
        }
    }

    /// Game thread, exactly once, at this packet's turn in the inbound queue: the packet, plus its
    /// unpacked columns when the background thread finished them first. `None` only if called
    /// twice.
    ///
    /// Never waits on the unpack: an unfinished one is abandoned (the worker discards its result)
    /// and the packet is applied inline, copied only if the worker is reading it at that moment.
    pub fn take(&self) -> Option<(S2CPlay, Option<Vec<Chunk>>)> {
        let taken = std::mem::replace(
            &mut *self.slot.lock().unwrap_or_else(PoisonError::into_inner),
            Slot::Taken,
        );
        match taken {
            Slot::Built(packet, columns) => Some((Arc::unwrap_or_clone(packet), columns)),
            Slot::Pending(packet) => Some((Arc::unwrap_or_clone(packet), None)),
            Slot::Taken => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_protocol::chunk::{ChunkBulkColumn, ChunkData, MapChunkBulk};

    fn full_column(chunk_x: i32, mask: u16, has_sky: bool) -> ChunkData {
        let sections = mask.count_ones() as usize;
        let mut data = Vec::new();
        for cell in 0..sections * 4096 {
            let raw: u16 = if cell % 5 == 0 { 1 << 4 } else { 0 };
            data.extend_from_slice(&raw.to_le_bytes());
        }
        let light_arrays = if has_sky { 2 } else { 1 };
        data.resize(data.len() + sections * 2048 * light_arrays, 0x77);
        data.resize(data.len() + 256, 1);
        ChunkData {
            chunk_x,
            chunk_z: 7,
            ground_up_continuous: true,
            primary_bit_mask: mask,
            data,
        }
    }

    fn wait_built(job: &PrebuildingChunkPacket) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while std::time::Instant::now() < deadline {
            if matches!(
                *job.slot.lock().unwrap_or_else(PoisonError::into_inner),
                Slot::Built(..)
            ) {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        panic!("the background unpack never finished");
    }

    /// Only full-column shapes are wrapped; an unload (`S21` full with bitmask 0), a partial
    /// `S21` and any other packet go back untouched.
    #[test]
    fn only_full_column_chunk_packets_are_prebuilt() {
        let unload = ChunkData {
            primary_bit_mask: 0,
            ..full_column(0, 0, true)
        };
        assert!(PrebuildingChunkPacket::spawn(S2CPlay::ChunkData(unload), true).is_err());
        let partial = ChunkData {
            ground_up_continuous: false,
            ..full_column(0, 1, true)
        };
        assert!(PrebuildingChunkPacket::spawn(S2CPlay::ChunkData(partial), true).is_err());
        let full = PrebuildingChunkPacket::spawn(S2CPlay::ChunkData(full_column(0, 1, true)), true);
        assert!(full.is_ok());
    }

    /// The background result is the same column `Chunk::from_full_column_packet` builds, and the
    /// packet itself comes back unchanged.
    #[test]
    fn a_finished_unpack_hands_back_the_packet_and_its_columns() {
        let packet = S2CPlay::MapChunkBulk(MapChunkBulk {
            has_sky_light: false,
            columns: [(1, 0b11), (2, 0), (3, 0b1000_0000_0000_0001)]
                .into_iter()
                .map(|(x, mask)| {
                    let column = full_column(x, mask, false);
                    ChunkBulkColumn {
                        chunk_x: column.chunk_x,
                        chunk_z: column.chunk_z,
                        primary_bit_mask: column.primary_bit_mask,
                        data: column.data,
                    }
                })
                .collect(),
        });
        let expected = format!("{packet:?}");
        let job = PrebuildingChunkPacket::spawn(packet, true).expect("S26 is prebuilt");
        assert!(!job.assumed_sky(), "S26 uses its own layout flag");
        wait_built(&job);
        let (packet, columns) = job.take().expect("first take");
        assert_eq!(format!("{packet:?}"), expected);
        let columns = columns.expect("well-formed payloads unpack");
        let S2CPlay::MapChunkBulk(bulk) = &packet else {
            unreachable!()
        };
        assert_eq!(columns.len(), bulk.columns.len());
        for (column, built) in bulk.columns.iter().zip(&columns) {
            let direct =
                Chunk::from_full_column_packet(column.primary_bit_mask, false, &column.data)
                    .expect("well-formed");
            assert_eq!(format!("{built:?}"), format!("{direct:?}"));
        }
        assert!(job.take().is_none(), "a packet is handed over once");
    }

    /// A payload the unpack rejects comes back without columns, so the game thread replays it on
    /// today's path and reports today's error.
    #[test]
    fn a_failed_unpack_hands_back_the_packet_without_columns() {
        let mut short = full_column(4, 0b1, true);
        short.data.truncate(100);
        let job = PrebuildingChunkPacket::spawn(S2CPlay::ChunkData(short), true).expect("wrapped");
        wait_built(&job);
        let (packet, columns) = job.take().expect("first take");
        assert!(columns.is_none());
        assert!(matches!(packet, S2CPlay::ChunkData(ref chunk) if chunk.data.len() == 100));
    }

    /// Suspect 1. java: `Minecraft.runGameLoop` runs every queued handler before
    /// the frame's ticks, one packet at a time, and `handleChunkData`, `handleBlockChange`,
    /// `handleMultiBlockChange` and `doPreChunk(false)` write `WorldClient` directly, which is
    /// what `World.getCollidingBoundingBoxes` reads. Through the live `Session::poll_all`, with
    /// two columns already unpacked off-thread: each prebuilt column is installed at its own slot
    /// (so an S23 queued after it lands on top of it, and one queued before a later full column
    /// is wiped by it), S22 and the unload follow in order, and collision sees all of it by the
    /// time the drain returns.
    #[test]
    fn world_packets_reach_collision_in_packet_order_through_poll_all() {
        use crate::net::channel::NetHandle;
        use crate::net::ToGame;
        use crate::session::{JoinSettings, Session};
        use mc_data::BlockState;
        use mc_math::{Aabb, BlockPos, ChunkPos};
        use mc_protocol::play::s2c;

        let (handle, to_game, _to_net) = NetHandle::new_pair();
        let mut session = Session::new(handle, JoinSettings::default());
        // `full_column` packs stone (raw 1 << 4) into every fifth cell and puts the column at
        // chunk z = 7: local (0,0,0) and (5,0,0) are stone, local (1,0,0) and (0,0,1) are air.
        let p = BlockPos::new(0, 0, 112);
        let kept = BlockPos::new(5, 0, 112);
        let s22 = BlockPos::new(1, 0, 112);
        let q = BlockPos::new(16, 0, 113);
        let stone = BlockState::new(mc_data::BlockId(1), 0);
        let glass = BlockState::new(mc_data::BlockId(20), 0);

        let prebuilt = |chunk_x| {
            PrebuildingChunkPacket::spawn(S2CPlay::ChunkData(full_column(chunk_x, 1, true)), true)
                .expect("a full column is prebuilt")
        };
        let job_a = prebuilt(0);
        let job_b = prebuilt(1);
        // Make the off-thread result the one that is installed, not the inline fallback.
        wait_built(&job_a);
        wait_built(&job_b);

        let unload = ChunkData {
            primary_bit_mask: 0,
            data: Vec::new(),
            ..full_column(2, 0, true)
        };
        for message in [
            // Columns B and C resident first, through the plain inline path.
            ToGame::Packet(S2CPlay::ChunkData(full_column(1, 1, true))),
            ToGame::Packet(S2CPlay::ChunkData(full_column(2, 1, true))),
            ToGame::ChunkPacket(job_a),
            ToGame::Packet(S2CPlay::BlockChange(s2c::BlockChange {
                position: p,
                block_state_id: 0,
            })),
            ToGame::Packet(S2CPlay::BlockChange(s2c::BlockChange {
                position: q,
                block_state_id: i32::from(glass.id),
            })),
            ToGame::ChunkPacket(job_b),
            ToGame::Packet(S2CPlay::MultiBlockChange(s2c::MultiBlockChange {
                chunk_x: 0,
                chunk_z: 7,
                changes: vec![s2c::MultiBlockChangeEntry {
                    rel_x: 1,
                    y: 0,
                    rel_z: 0,
                    block_state_id: i32::from(glass.id),
                }],
            })),
            ToGame::Packet(S2CPlay::ChunkData(unload)),
        ] {
            to_game.try_send(message).expect("inbound queue");
        }
        session.poll_all();

        let world = &session.state().world;
        let boxes_at = |pos: BlockPos| {
            let cell = Aabb::from_bounds(
                f64::from(pos.x),
                f64::from(pos.y),
                f64::from(pos.z),
                f64::from(pos.x) + 1.0,
                f64::from(pos.y) + 1.0,
                f64::from(pos.z) + 1.0,
            );
            let mut out = Vec::new();
            mc_physics::collision::collision_boxes_in(
                world,
                &cell.contract(0.01, 0.01, 0.01),
                &mut out,
            );
            out
        };
        assert_eq!(
            world.block_at(kept),
            stone,
            "the prebuilt column A is installed"
        );
        assert_eq!(boxes_at(kept).len(), 1, "and collidable before any tick");
        assert_eq!(
            world.block_at(p),
            BlockState::AIR,
            "the S23 after A lands on top of it"
        );
        assert!(boxes_at(p).is_empty());
        assert_eq!(world.block_at(s22), glass, "the S22 entry is applied");
        assert_eq!(boxes_at(s22).len(), 1);
        assert_eq!(
            world.block_at(q),
            BlockState::AIR,
            "the S23 before the second full B column is wiped by it"
        );
        assert!(
            !world.is_chunk_loaded(ChunkPos::new(2, 7)),
            "the S21 unload removes C"
        );
        assert!(boxes_at(BlockPos::new(32, 0, 112)).is_empty());
    }

    /// L8 follow-up (HIGH): bridging over the void. A vanilla 1.8 server sends a lone chunk as
    /// `S21PacketChunkData(chunk, true, 65535)` (`EntityPlayerMP.onUpdate`), and an all-air
    /// column extracts to bitmask 0, which the client treats as an unload
    /// (`NetHandlerPlayClient.handleChunkData:751-756`): void columns beside an island are never
    /// resident. `World.setBlockState`/`getBlockState` then go to `ChunkProviderClient`'s one
    /// shared `blankChunk`, an `EmptyChunk` that inherits both, so the predicted bridge block and
    /// the server's S23 for it land there, collide, and show at the same local cell of every
    /// absent column, through further unloads, until a new world. The old `World::set_block`
    /// dropped both writes and the player fell through the bridge the server had placed.
    #[test]
    fn a_bridge_over_a_void_column_lands_in_the_blank_chunk_and_holds_the_player() {
        use crate::app::interact::{
            can_place_block_on_side, item_block_delegate_use, BlockUseResult,
        };
        use crate::sync::dispatch::apply;
        use crate::sync::event::ClientEvent;
        use crate::sync::state::ClientState;
        use mc_data::BlockState;
        use mc_math::{Aabb, BlockPos, ChunkPos, DVec3, Facing};
        use mc_physics::{tick_player, MoveInput, PhysicsState};
        use mc_protocol::play::s2c;

        fn lone_void_s21(state: &mut ClientState, events: &mut Vec<ClientEvent>) {
            let all_air = ChunkData {
                chunk_x: 1,
                chunk_z: 0,
                ground_up_continuous: true,
                primary_bit_mask: 0,
                data: Vec::new(),
            };
            apply(state, S2CPlay::ChunkData(all_air), events).expect("an unload applies");
        }
        /// Three ticks of a player standing at the bridge's middle, from rest.
        fn stand_on_bridge(world: &mc_world::World) -> PhysicsState {
            let mut player = PhysicsState::new_player(DVec3::new(16.5, 65.0, 8.5));
            for _ in 0..3 {
                tick_player(world, &mut player, &MoveInput::default());
            }
            player
        }

        let mut state = ClientState::new();
        let mut events = Vec::new();
        // The island: a resident column whose east edge block is (15, 64, 8).
        let edge = BlockPos::new(15, 64, 8);
        let stone = BlockState::new(mc_data::BlockId(1), 0);
        state.world.load_empty_chunk(edge.chunk_pos());
        state.world.set_block(edge, stone);
        // The void column east of it arrives alone and all air.
        let void_column = ChunkPos::new(1, 0);
        lone_void_s21(&mut state, &mut events);
        assert!(!state.world.is_chunk_loaded(void_column));

        // The prediction: standing on the edge, click its east face with wool.
        let wool_stack = mc_protocol::ItemStack {
            item_id: 35,
            count: 64,
            damage: 14,
            nbt: None,
        };
        let wool = BlockState::new(mc_data::BlockId(35), 14);
        let target = edge.east();
        let blockers = [PhysicsState::new_player(DVec3::new(15.5, 65.0, 8.5)).aabb()];
        assert!(can_place_block_on_side(
            &state.world,
            edge,
            Facing::East,
            &wool_stack,
            &blockers
        ));
        let BlockUseResult::Consumed {
            placement: Some(placement),
            ..
        } = item_block_delegate_use(
            &state.world,
            edge,
            Facing::East,
            0.5,
            -90.0,
            &wool_stack,
            true,
            &blockers,
        )
        else {
            panic!("the bridge block is predicted");
        };
        assert_eq!((placement.pos, placement.state), (target, wool));
        // `Client::on_player_right_click`'s local write.
        state.world.set_block(placement.pos, placement.state);
        // The server's S23 for it, and for the next block of the bridge.
        let next = target.east();
        for position in [target, next] {
            let change = s2c::BlockChange {
                position,
                block_state_id: i32::from(wool.id),
            };
            apply(&mut state, S2CPlay::BlockChange(change), &mut events).expect("S23 applies");
        }
        assert!(
            !state.world.is_chunk_loaded(void_column),
            "the blank chunk holds the bridge"
        );
        assert_eq!(state.world.block_at(target), wool);
        assert_eq!(state.world.block_at(next), wool);
        // `World.rayTraceBlocks` reads `getBlockState` too: the crosshair finds the bridge block,
        // so the next click extends the bridge from it.
        let looking_down = crate::app::interact::raytrace_blocks(
            &state.world,
            DVec3::new(16.5, 66.62, 8.5),
            DVec3::new(16.5, 61.0, 8.5),
        )
        .and_then(crate::app::interact::RayTrace::as_block)
        .expect("the ray stops on the bridge");
        assert_eq!((looking_down.pos, looking_down.face), (target, Facing::Up));

        // Walk onto it: the absent column's gravity pin pulls at -0.1 and the wool stops it.
        let held = stand_on_bridge(&state.world);
        assert_eq!(held.pos.y, 65.0, "the bridge holds the player");
        assert!(held.on_ground);
        // Negative control, the old behaviour: `set_block` returned early for an absent column,
        // i.e. a world where neither write exists. The same three ticks fall through.
        let mut old = mc_world::World::new();
        old.load_empty_chunk(edge.chunk_pos());
        old.set_block(edge, stone);
        let fell = stand_on_bridge(&old);
        assert!(
            fell.pos.y < 65.0 && !fell.on_ground,
            "without it the player falls"
        );

        // The same local cell of every other absent column: (-80, 64, 40) is local (0, 64, 8).
        let mirror = BlockPos::new(-80, 64, 40);
        assert_eq!(state.world.block_at(mirror), wool);
        let mut boxes = Vec::new();
        let mirror_cell = Aabb::from_bounds(-79.9, 64.1, 40.1, -79.1, 64.9, 40.9);
        mc_physics::collision::collision_boxes_in(&state.world, &mirror_cell, &mut boxes);
        assert_eq!(boxes.len(), 1, "and collides there");

        // Another lone all-air S21 for the column (an unload) leaves the bridge alone.
        lone_void_s21(&mut state, &mut events);
        assert_eq!(state.world.block_at(target), wool);
        assert_eq!(stand_on_bridge(&state.world).pos.y, 65.0);

        // A new world (JoinGame builds a new WorldClient) has a fresh blank chunk.
        let join = s2c::JoinGame {
            entity_id: 1,
            hardcore_mode: false,
            game_mode: 0,
            dimension: 0,
            difficulty: 2,
            max_players: 20,
            level_type: "default".into(),
            reduced_debug_info: false,
        };
        apply(&mut state, S2CPlay::JoinGame(join), &mut events).expect("JoinGame applies");
        assert_eq!(state.world.block_at(target), BlockState::AIR);
        assert_eq!(state.world.block_at(mirror), BlockState::AIR);
    }
}
