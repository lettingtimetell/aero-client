//! The client's lighting seam.
//!
//! Owned by the lighting author. These are `impl Client` methods, kept out of `driver.rs`
//! because every Wave-1 author is editing that file at once; the fields they reach are
//! `pub(crate)` on [`Client`] for exactly this reason.
//!
//! | method | java |
//! |---|---|
//! | [`Client::drop_column_meshes`] | `WorldClient.doPreChunk(x, z, false)`'s `markBlockRangeForRenderUpdate` (`:166`) |
//! | [`Client::drop_all_chunk_meshes`] | `Minecraft.loadWorld` → `RenderGlobal.setWorldAndLoadRenderers` |
//! | [`Client::tick_client_light`] | `ChunkProviderClient.unloadQueuedChunks` (`:109`) + `WorldClient.updateBlocks` (`:120`) |

#[cfg(test)]
use std::collections::HashSet;

use mc_math::{BlockPos, ChunkPos, ChunkSectionPos, DVec3};

use crate::app::driver::Client;

/// How many active columns [`Client::tick_client_light`] relights per tick.
///
/// java: `WorldClient.updateBlocks` hands **10** newly-active columns per tick to
/// `playMoodSoundAndCheckLight` (`:143-148`). The separate
/// `ChunkProviderClient.unloadQueuedChunks` pass walks every loaded column for gap repair; its
/// 5 ms wall-clock cutoff is intentionally represented by a deterministic per-column policy in
/// `mc_world::World::tick_chunk_gap_light`, while this constant remains the relight budget only.
///
/// `enqueueRelightChecks` advances at most 8 of a chunk's 4096 slots per visited column.
pub const LIGHT_COLUMNS_PER_TICK: usize = 10;

/// One player's `activeChunkSet` centre: java `MathHelper.floor_double(posX / 16.0D)` (and Z).
/// `floor_f64` is the exact port, including its wrap below `i32::MIN` (an S08 can place the
/// player anywhere).
fn player_chunk_centre(position: DVec3) -> ChunkPos {
    ChunkPos::new(
        mc_math::floor_f64(position.x / 16.0),
        mc_math::floor_f64(position.z / 16.0),
    )
}

/// Build `World.activeChunkSet` around every client-side player, the literal Java way.
///
/// The live tick hands only the centres to `World::tick_chunk_relight_active_around`, which
/// caches the same set (#6); this is the reference the equivalence tests compare against.
#[cfg(test)]
fn active_player_chunks(positions: &[DVec3], render_distance: i32) -> HashSet<ChunkPos> {
    let mut active = HashSet::new();
    for position in positions {
        let centre = player_chunk_centre(*position);
        for dx in -render_distance..=render_distance {
            for dz in -render_distance..=render_distance {
                active.insert(ChunkPos::new(centre.x + dx, centre.z + dz));
            }
        }
    }
    active
}

/// Select Java's one random player and one `floor(pos) + nextInt(11) - 5` light sample.
fn sampled_player_light_position(
    positions: &[DVec3],
    random: &mut mc_math::JavaRandom,
) -> Option<BlockPos> {
    let bound = i32::try_from(positions.len()).ok()?;
    if bound == 0 {
        return None;
    }
    let player = positions.get(random.next_i32_bound(bound) as usize)?;
    // java: `MathHelper.floor_double(posX) + rand.nextInt(11) - 5` is wrapping `int` arithmetic;
    // an S08 can put the player where `floor_double` saturates. Draws stay in x, y, z order.
    Some(BlockPos::new(
        mc_math::floor_f64(player.x)
            .wrapping_add(random.next_i32_bound(11))
            .wrapping_sub(5),
        mc_math::floor_f64(player.y)
            .wrapping_add(random.next_i32_bound(11))
            .wrapping_sub(5),
        mc_math::floor_f64(player.z)
            .wrapping_add(random.next_i32_bound(11))
            .wrapping_sub(5),
    ))
}

impl Client {
    /// Drop one chunk column's 16 GPU meshes. java: `WorldClient.doPreChunk(x, z, false)` calls
    /// `unloadChunk` **and then** `markBlockRangeForRenderUpdate` over the whole column, which
    /// is what makes the departing geometry disappear.
    ///
    /// Nothing consumed `ClientEvent::ChunkUnloaded`, so a column that left view
    /// distance kept rendering as a frozen shell until `TerrainRenderer::evict_to_budget`
    /// happened to reach it — a trail of ghost terrain behind the player, and resident GPU
    /// memory the frame budget was paying for.
    ///
    /// `mc_world::World::unload_chunk` marks the column and its eight neighbours dirty (the
    /// other half), so the neighbours re-mesh against the new hole; this half is the
    /// part only the renderer can do, because an unloaded column produces no mesh at all and so
    /// can never be *overwritten* by one.
    pub(crate) fn drop_column_meshes(&mut self, pos: ChunkPos) {
        for y in 0..16 {
            let section = ChunkSectionPos::new(pos.x, y, pos.z);
            self.mesh_queue.cancel(section);
            self.uploaded_sections.remove(&section);
            // A completed upload may still be waiting for its frame deadline and therefore
            // have no uploaded_sections entry. Cancel it as well as resident air/geometry.
            if let Some(render) = &self.render {
                render.terrain.borrow_mut().remove(section);
            }
        }
    }

    /// Drop every chunk mesh the renderer holds, without touching the session.
    ///
    /// `NetHandlerPlayClient.handleRespawn` (`:1060-1068`) builds a whole new
    /// `WorldClient` on a dimension change and calls `gameController.loadWorld(...)`, which
    /// reaches `RenderGlobal.setWorldAndLoadRenderers` and drops every `RenderChunk`. Here
    /// `ClientState::reset_world` replaced the world but nothing told the renderer, so going
    /// through a Nether portal left the overworld's geometry resident and drawn wherever the new
    /// dimension had not yet supplied a section at the same `ChunkSectionPos` — a patchwork of
    /// two worlds that never resolves above the Nether roof.
    ///
    /// This is the mesh-teardown half of `Client::unload_world`, split out because a respawn
    /// keeps the session, the chat and the camera path that `unload_world` drops.
    pub(crate) fn drop_all_chunk_meshes(&mut self) {
        self.mesh_queue.reset_world();
        // The cached precipitation columns describe the world that is going away.
        self.precip_key = None;
        // Particles are positioned in the world that just went away.
        // java: `RenderGlobal.setWorldAndLoadRenderers` → `EffectRenderer.clearEffects`.
        self.particles.clear();
        if let Some(render) = &self.render {
            let mut terrain = render.terrain.borrow_mut();
            terrain.cancel_all_pending_uploads();
            // `remove` for each, in this set's order; done in one step when
            // the set is every resident section (the usual case). Then the
            // mesh-revision tombstones go too: the queue was reset above, so no result meshed
            // for the old world can arrive any more.
            terrain.remove_world_sections(&self.uploaded_sections);
            self.uploaded_sections.clear();
        } else {
            self.uploaded_sections.clear();
        }
    }

    /// One tick of the client's periodic light repair. Returns the number of columns consumed by
    /// the ten-column relight budget, `0` when there is no world.
    ///
    /// Vanilla runs `recheckGaps` for every loaded column in
    /// `ChunkProviderClient.unloadQueuedChunks`, then separately gives at most ten newly-active
    /// columns to `enqueueRelightChecks` in `WorldClient.updateBlocks`. Keep those calls in that
    /// order: sharing one ten-column budget leaves flagged sky gaps unvisited for seconds at
    /// ordinary view distances.
    pub(crate) fn tick_client_light(&mut self) -> usize {
        mc_core::profile_scope!("client.tick_client_light");
        let Some(session) = self.session.as_ref() else {
            return 0;
        };
        let state = session.state();
        let mut player_positions = Vec::new();
        if let Some(predictor) = session.predictor() {
            player_positions.push(predictor.state.pos);
        }
        player_positions.extend(
            state
                .entities
                .iter()
                .filter(|(_, tracked)| tracked.kind == crate::sync::entities::TrackedKind::Player)
                .map(|(_, tracked)| tracked.render_position(&state.world, 1.0)),
        );
        // `activeChunkSet` is the union of the `(2r+1)²` squares around these centres; the world
        // caches it on (centres, r) instead of rebuilding it with 12k-22k inserts per tick.
        let centres: Vec<ChunkPos> = player_positions
            .iter()
            .map(|position| player_chunk_centre(*position))
            .collect();
        let radius = self.settings.render_distance_chunks.clamp(2, 32);
        let sample = sampled_player_light_position(&player_positions, &mut self.light_check_rng);

        let session = self.session.as_mut().expect("session checked above");
        let world = &mut session.state_mut().world;
        world.tick_chunk_gap_light();
        if let Some(sample) = sample {
            world.check_light(sample);
        }
        world.tick_chunk_relight_active_around(&centres, radius, LIGHT_COLUMNS_PER_TICK)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_columns_floor_negative_player_coordinates_and_union_overlaps() {
        let active = active_player_chunks(
            &[DVec3::new(-0.1, 64.0, -0.1), DVec3::new(16.1, 70.0, 0.1)],
            1,
        );
        assert!(active.contains(&ChunkPos::new(-2, -2)));
        assert!(active.contains(&ChunkPos::new(2, 1)));
        assert!(active.contains(&ChunkPos::new(0, 0)));
        assert_eq!(
            active.len(),
            16,
            "overlapping squares form a set, not duplicate work"
        );
        assert!(!active.contains(&ChunkPos::new(30, 30)));
    }

    #[test]
    fn player_light_sample_uses_one_player_draw_then_three_bounded_offsets() {
        let positions = [
            DVec3::new(-0.25, 64.99, 16.0),
            DVec3::new(31.75, 5.1, -32.01),
        ];
        let seed = 0x1234_5678;
        let mut expected = mc_math::JavaRandom::new(seed);
        let player = positions[expected.next_i32_bound(2) as usize];
        let want = BlockPos::new(
            player.x.floor() as i32 + expected.next_i32_bound(11) - 5,
            player.y.floor() as i32 + expected.next_i32_bound(11) - 5,
            player.z.floor() as i32 + expected.next_i32_bound(11) - 5,
        );
        let mut actual = mc_math::JavaRandom::new(seed);
        assert_eq!(
            sampled_player_light_position(&positions, &mut actual),
            Some(want)
        );
        assert_eq!(
            actual, expected,
            "the live helper consumes exactly four draws"
        );

        assert_eq!(
            sampled_player_light_position(&[], &mut actual),
            None,
            "an empty playerEntities list consumes no RNG"
        );
        assert_eq!(actual, expected);
    }

    /// An S08 can place the player where `MathHelper.floor_double` saturates
    /// (or, below `i32::MIN`, wraps to `MAX`). Java's `floor + nextInt(11) - 5` then wraps as
    /// `int` arithmetic; the old `floor() as i32 +` panicked in debug builds.
    #[test]
    fn player_light_sample_wraps_at_saturated_positions_like_java_int() {
        let positions = [DVec3::new(f64::MAX, -1.0e300, f64::NAN)];
        let seed = 0x0bad_5eed;
        let mut expected = mc_math::JavaRandom::new(seed);
        let _player = expected.next_i32_bound(1);
        let want = BlockPos::new(
            i32::MAX
                .wrapping_add(expected.next_i32_bound(11))
                .wrapping_sub(5),
            // `(int)-1e300` is MIN, and `-1e300 < MIN` makes floor_double return MIN - 1 = MAX.
            i32::MAX
                .wrapping_add(expected.next_i32_bound(11))
                .wrapping_sub(5),
            expected.next_i32_bound(11) - 5,
        );
        let mut actual = mc_math::JavaRandom::new(seed);
        assert_eq!(
            sampled_player_light_position(&positions, &mut actual),
            Some(want)
        );
        assert_eq!(actual, expected);
        assert_eq!(
            player_chunk_centre(DVec3::new(-1.0e300, 0.0, f64::NAN)),
            ChunkPos::new(i32::MAX, 0),
            "floor_double(-1e300 / 16) wraps to MAX; NaN casts to 0"
        );
    }

    #[test]
    fn live_client_light_tick_uses_the_sample_and_active_only_relight_driver() {
        // Split on the test module itself: the file also has `#[cfg(test)]` items above it (a
        // test-only import and the old hashed active-set rebuild kept as an oracle).
        let production = include_str!("relight.rs")
            .split("#[cfg(test)]\nmod tests")
            .next()
            .expect("production relight module");
        assert!(production.contains("world.check_light(sample);"));
        assert!(production.contains(
            "world.tick_chunk_relight_active_around(&centres, radius, LIGHT_COLUMNS_PER_TICK)"
        ));
        assert!(
            !production.contains("world.tick_chunk_relight(LIGHT_COLUMNS_PER_TICK)"),
            "the all-loaded compatibility queue must not own the live client budget"
        );
    }
}
