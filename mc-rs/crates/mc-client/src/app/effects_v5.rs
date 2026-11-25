//! The world events (`S28PacketEffect`) a 1.7.10 session plays differently from 1.8.9's table.
//!
//! java17 `RenderGlobal.playAuxSFX` (`client/renderer/RenderGlobal.java:2128-2372`) against
//! java18's (`:2160-2360`): the same ids, sounds and particles but for these:
//!
//! - 1000, 1001 and 1002 (dispenser and bow clicks) sound at the block's corner,
//!   `playSound(x, y, z, ...)`, where java18 adds 0.5 on every axis (`playSoundAtPos`);
//! - 2006, the dust of an entity landing from a fall, exists only in 1.7.10. A 1.7.10 server sends
//!   it with `data = ceil(fallDistance - 3)` (java17 `EntityLivingBase.updateFallState`,
//!   `entity/EntityLivingBase.java:170-172`) where a 1.8.9 server sends block-dust particles;
//! - 1003 (a door) picks `random.door_open` or `random.door_close` at random in 1.7.10
//!   (`:2158-2168`) where java18 plays `random.door_open`; the 1.8.9 table is kept for it, since
//!   the sound list this client plays (1.8.9's) gives `random.door_open` both door files, which
//!   sounds the same.
//!
//! The dispatcher already drops 1006, which 1.7.10 does not have. Everything else is the 1.8.9
//! table's.

use mc_math::{BlockPos, DVec3};
use mc_protocol::play::s2c;

use super::driver::Client;

impl Client {
    /// Play `packet` the 1.7.10 way if a 1.7.10 session's table differs from 1.8.9's for it;
    /// `false` (nothing done) otherwise, and always for a 1.8.9 session.
    pub(crate) fn react_to_effect_v5(&mut self, packet: &s2c::Effect) -> bool {
        if packet.disable_relative_volume || !matches!(packet.effect_id, 1000..=1002 | 2006) {
            return false;
        }
        if !self.plays_1_7_10() {
            return false;
        }
        let pos = packet.position;
        let corner = DVec3::new(f64::from(pos.x), f64::from(pos.y), f64::from(pos.z));
        match packet.effect_id {
            1000 => self.play_sound_with_distance_delay(corner, "random.click", 1.0, 1.0, false),
            1001 => self.play_sound_with_distance_delay(corner, "random.click", 1.0, 1.2, false),
            1002 => self.play_sound_with_distance_delay(corner, "random.bow", 1.0, 1.2, false),
            _ => self.fall_dust_v5(pos, packet.data),
        }
        true
    }

    /// java17 `RenderGlobal.playAuxSFX` case 2006 (`:2346-2370`): unless the block at `pos` is
    /// air, `(int)(150 * d)` block-dust particles of that block from the top centre of the block,
    /// `d = min(0.2F + data / 15.0F, 10.0F)` capped at 2.5, each thrown along a random angle on
    /// the ring with the world's random stream (the shared effect stream here, as the 1.8.9
    /// table's cases draw from).
    fn fall_dust_v5(&mut self, pos: BlockPos, data: i32) {
        let Some(state) = self
            .session
            .as_ref()
            .map(|session| session.state().world.block_at(pos))
        else {
            return;
        };
        if mc_data::block_props(state).material == mc_data::Material::Air {
            return;
        }
        let mut strength = f64::from((0.2_f32 + data as f32 / 15.0).min(10.0));
        if strength > 2.5 {
            strength = 2.5;
        }
        let count = (150.0 * strength) as i32;
        let origin = DVec3::new(
            f64::from(pos.x as f32 + 0.5),
            f64::from(pos.y as f32 + 1.0),
            f64::from(pos.z as f32 + 0.5),
        );
        let args = [state.to_java_state_id()];
        for _ in 0..count {
            let (angle, spread) = {
                let rng = self.particles.rng_mut();
                let angle = random_float_clamp(rng, 0.0, std::f32::consts::PI * 2.0);
                (angle, f64::from(random_float_clamp(rng, 0.75, 1.0)))
            };
            let up = 0.200_000_002_980_232_24 + strength / 100.0;
            let reach = spread * spread * (strength + 0.2);
            let speed = DVec3::new(
                f64::from(mc_math::cos(angle) * 0.2) * reach,
                up,
                f64::from(mc_math::sin(angle) * 0.2) * reach,
            );
            self.spawn_effect_particle(
                mc_render::particle::packet::BLOCK_DUST,
                origin,
                speed,
                &args,
            );
        }
    }
}

/// java: `MathHelper.randomFloatClamp(rand, min, max)`: `min` when `min >= max`, else
/// `rand.nextFloat() * (max - min) + min`.
fn random_float_clamp(rng: &mut mc_math::JavaRandom, min: f32, max: f32) -> f32 {
    if min >= max {
        min
    } else {
        rng.next_f32() * (max - min) + min
    }
}

#[cfg(test)]
mod tests {
    use mc_version::GameVersion;

    use super::*;
    use crate::app::config::{ClientConfig, LoginMode};
    use crate::net::NetHandle;
    use crate::session::{JoinSettings, Session};
    use crate::sync::ClientEvent;

    /// A client on the real assets with an offline session of `version` whose column (0, 0) is
    /// loaded and holds stone at (0, 1, 0). `None` (skip) without the vanilla jar.
    fn client_with_stone(version: GameVersion) -> Option<(Client, tempfile::TempDir)> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("skip: no vanilla jar in this checkout");
            return None;
        }
        let mut client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        let (handle, _to_game, _to_net) = NetHandle::new_pair();
        let mut session = Session::with_version(handle, JoinSettings::default(), version);
        let state = session.state_mut();
        let data = vec![0u8; mc_world::expected_chunk_data_len(0b1, true, true)];
        crate::sync::dispatch::apply(
            state,
            mc_protocol::play::S2CPlay::ChunkData(mc_protocol::chunk::ChunkData {
                chunk_x: 0,
                chunk_z: 0,
                ground_up_continuous: true,
                primary_bit_mask: 0b1,
                data,
            }),
            &mut Vec::new(),
        )
        .expect("an all-air column");
        state.world.set_block(
            BlockPos::new(0, 1, 0),
            mc_data::BlockState::from_raw(1 << 4),
        );
        client.session = Some(session);
        Some((client, dir))
    }

    fn effect(effect_id: i32, y: i32) -> ClientEvent {
        ClientEvent::Effect(s2c::Effect {
            effect_id,
            position: BlockPos::new(0, y, 0),
            data: 1,
            disable_relative_volume: false,
        })
    }

    /// java17 `RenderGlobal.playAuxSFX` case 2006: dust from the stone the entity landed on,
    /// none from air; java18 has no case 2006 (the control) and spawns nothing.
    #[test]
    fn fall_dust_is_a_1_7_10_event() {
        let Some((mut client, _dir)) = client_with_stone(GameVersion::V1_7_10) else {
            return;
        };
        client.react(&effect(2006, 2));
        assert!(client.particles.is_empty(), "air: no dust");
        client.react(&effect(2006, 1));
        assert!(!client.particles.is_empty(), "stone: dust");

        let Some((mut control, _dir)) = client_with_stone(GameVersion::V1_8_9) else {
            return;
        };
        control.react(&effect(2006, 1));
        assert!(control.particles.is_empty(), "1.8.9 has no event 2006");
    }

    /// `MathHelper.randomFloatClamp` keeps `min` for an empty range and stays in `[min, max)`.
    #[test]
    fn random_float_clamp_is_javas() {
        let mut rng = mc_math::JavaRandom::new(7);
        assert_eq!(random_float_clamp(&mut rng, 1.0, 1.0), 1.0);
        for _ in 0..100 {
            let value = random_float_clamp(&mut rng, 0.75, 1.0);
            assert!((0.75..1.0).contains(&value));
        }
    }
}
