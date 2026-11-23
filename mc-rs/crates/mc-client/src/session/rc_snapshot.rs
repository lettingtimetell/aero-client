//! Exact copies of a replay session, for replay checkpoints.
//!
//! A checkpoint holds a whole [`Session`] as it was at one moment of a replay; a seek restores it
//! and plays forward from there, and must reach the same frame as playing from the start. So
//! every field is copied. The struct literal below names each one: a field added to `Session`
//! later stops this file compiling until someone decides how a copy carries it, which is the
//! point (a field silently left out would make seeks drift from playback).
//!
//! Only a session fed by a replay is ever copied ([`Session::is_driven`]): its network handle is
//! a queue with no socket, and a copy gets a fresh, empty one. Copies are taken between records,
//! after every pushed packet was applied, so there is nothing in the old queue to carry over.

use mc_world::EcsCloner;

use super::Session;
use crate::net::NetHandle;
use crate::sync::ClientState;

impl ClientState {
    /// An exact, independent copy (the world through [`mc_world::World::try_clone_with`]), or
    /// `None` when the world's entity arena holds a component type `cloner` does not know.
    pub(crate) fn rc_try_clone(&self, cloner: &EcsCloner) -> Option<ClientState> {
        Some(ClientState {
            world: self.world.try_clone_with(cloner)?,
            entities: self.entities.clone(),
            players: self.players.clone(),
            scoreboard: self.scoreboard.clone(),
            env: self.env.clone(),
            local: self.local.clone(),
        })
    }
}

impl Session {
    /// An exact copy of a replay session with a fresh, empty driven network handle. `None` for
    /// a live session or when the world cannot be copied. The caller takes it only after every
    /// pushed packet was applied (the playback's `packets_pending` is false).
    pub(crate) fn rc_try_clone(&self, cloner: &EcsCloner) -> Option<Session> {
        if !self.handle.is_driven() {
            return None;
        }
        Some(Session {
            handle: NetHandle::driven(),
            state: self.state.rc_try_clone(cloner)?,
            outbound: self.outbound.clone(),
            predictor: self.predictor.clone(),
            local_render: self.local_render.clone(),
            local_swing_progress_int: self.local_swing_progress_int,
            local_swinging: self.local_swinging,
            local_swing_animation_end: self.local_swing_animation_end,
            phase: self.phase.clone(),
            settings: self.settings.clone(),
            last_client_settings: self.last_client_settings.clone(),
            login: self.login.clone(),
            effects: self.effects.clone(),
            riding: self.riding,
            pending_horse_jump: self.pending_horse_jump,
            pending_horse_mount: self.pending_horse_mount,
            controller_updated: self.controller_updated,
            local_particle_random: self.local_particle_random.clone(),
            local_fire: self.local_fire,
            local_running_in_water: self.local_running_in_water,
            terrain_download_ticks: self.terrain_download_ticks,
            local_spawn_point: self.local_spawn_point,
            local_spawn_forced: self.local_spawn_forced,
            local_added_to_chunk: self.local_added_to_chunk,
            view_target: self.view_target,
            world_generation: self.world_generation,
            hit_particles: self.hit_particles.clone(),
            tapped_raw: None,
        })
    }
}
