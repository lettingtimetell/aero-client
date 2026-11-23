//! The shader packs' client half: pack exclusivity, the per-frame
//! inputs ([`PackFrame`]), the Classic pack's settings, and the shared core's phases.
//!
//! * **Exclusivity.** The three packs (`shaders_classic`, `shaders_dreamy`, `shaders_underwater`)
//!   are ordinary mods; turning one on turns the other two off here, whatever turned it on (the
//!   menu, a profile load, a hand-edited settings file). When several are on at once with none
//!   newly switched on (a settings file), the first in `PackId::ALL` order stays on.
//! * **The core** ([`ShaderCore`], `mc_render::shaderpack`) is created with the client and shared
//!   with every pack phase. Each frame while a pack is switched on this file fills a
//!   [`PackFrame`] from the session and hands it over; the Dreamy and Underwater packs drive
//!   their own phases from their markers below, after that.
//! * **Classic** is the core features alone (shadows, sun shading, water, waving plants, bloom,
//!   tone map) plus the light rays, configured from the `shaders_classic` options.
//! * **The mesher.** Sun shading and the material flags (waving, water) need a different mesh;
//!   [`Client::shader_pack_mesh_settings`] folds the running pack's needs into the mesher settings
//!   beside Tall Grass, so a change re-meshes the world once through the existing path.
//!
//! Cost with every pack off: one `u64` compare per frame (the settings generation) and one
//! `Option` test. Nothing is allocated on the GPU, no phase records, the mesher is vanilla.

use mc_mods::mods::shaders_classic::SHADOW_RESOLUTIONS;
use mc_render::entity::EntityPhase;
use mc_render::shaderpack::{
    order, BloomSettings, CoreConfig, EyeMedium, FinalPhase, FrameInputs, LightingSettings,
    PackFrame, PackId, PackMesh, RaysPhase, ShaderCore, ShadowPhase, ShadowSettings, SkyGlowPhase,
    ToneCurve, TonemapSettings, WaterSettings, WavingSettings,
};
use mc_render::sky::fog::{CameraMedium, FogState};
use mc_render::terrain::{MeshSettings, TerrainRenderer};
use mc_render::{Camera, Gpu, PhaseList};
use std::cell::RefCell;
use std::rc::Rc;

use super::super::{Client, Shared, Ui};

/// The Classic pack's mod id.
const CLASSIC: &str = "shaders_classic";

/// What the client last applied for the shader packs.
#[derive(Default)]
pub(crate) struct ShaderPacksState {
    /// The settings generation last read. 0 = never (the store's generation starts at 1).
    seen: u64,
    /// Which packs were switched on at the last sync, in `PackId::ALL` order.
    was_on: [bool; 3],
    /// The pack switched on after exclusivity.
    active: Option<PackId>,
    /// The shared core, cloned into every pack phase.
    core: ShaderCore,
    /// The core's final pass (bloom, tone map), kept to free its targets when packs stop.
    final_phase: Option<Shared<FinalPhase>>,
    /// Classic's light rays (the rays, composited at `order::WORLD`).
    classic_rays: Option<Shared<RaysPhase>>,
    /// Classic's sun glow over the sky (`order::WORLD`).
    classic_glow: Option<Shared<SkyGlowPhase>>,
    /// Whether a pack rendered last frame.
    was_running: bool,
    /// The mesher mode last folded into the mesh settings.
    mesh: PackMesh,
    /// Frames counted while a pack is on.
    frame_index: u32,
    /// The Time Changer is on (read with the settings): it holds the clock the packs light by.
    time_changer_on: bool,
}

impl ShaderPacksState {
    /// The shared core (the Dreamy and Underwater packs clone it into their phases at their
    /// install markers).
    pub(crate) fn core(&self) -> &ShaderCore {
        &self.core
    }
}

/// The Classic options as core features.
pub(crate) fn classic_config(s: &mc_mods::ModSettings) -> CoreConfig {
    let pct = |option: &str| s.f32(CLASSIC, option) / 100.0;
    let shadows = s.bool(CLASSIC, "shadows").then(|| ShadowSettings {
        distance: s.f32(CLASSIC, "shadow_distance"),
        resolution: SHADOW_RESOLUTIONS[s
            .choice(CLASSIC, "shadow_quality")
            .min(SHADOW_RESOLUTIONS.len() - 1)],
        softness: pct("shadow_softness") * 6.0,
        strength: pct("shadow_strength"),
        cutout_casters: s.bool(CLASSIC, "plant_shadows"),
        entity_casters: s.bool(CLASSIC, "player_shadows"),
    });
    let lighting = s.bool(CLASSIC, "sun_shading").then(|| LightingSettings {
        sun_share: pct("sun_strength"),
        warmth: pct("sunset_warmth"),
        night_blue: pct("moonlight_tint"),
        shade_floor: pct("shade_brightness"),
    });
    let water = s.bool(CLASSIC, "water").then(|| WaterSettings {
        waves: pct("water_waves"),
        reflections: s.bool(CLASSIC, "water_reflections"),
        clarity: pct("water_clarity"),
        tint: [1.0, 1.0, 1.0],
    });
    let plants = s.bool(CLASSIC, "waving_plants");
    let leaves = s.bool(CLASSIC, "waving_leaves");
    let waving = (plants || leaves).then(|| WavingSettings {
        plants,
        leaves,
        strength: pct("waving_strength"),
        speed: 1.0,
    });
    let bloom = s.bool(CLASSIC, "bloom").then(|| BloomSettings {
        strength: pct("bloom_strength"),
        ..BloomSettings::default()
    });
    let curve = match s.choice(CLASSIC, "tone_curve") {
        1 => ToneCurve::Aces,
        2 => ToneCurve::Linear,
        _ => ToneCurve::Filmic,
    };
    let tonemap = Some(TonemapSettings {
        curve,
        exposure: s.f32(CLASSIC, "exposure"),
        contrast: pct("contrast"),
        saturation: pct("saturation"),
        vibrance: pct("vibrance"),
    });
    CoreConfig {
        sun_tilt: s.f32(CLASSIC, "sun_path_tilt"),
        shadows,
        lighting,
        water,
        waving,
        bloom,
        tonemap,
    }
}

/// Classic's light-ray strength, 0..1 (0 = off).
pub(crate) fn classic_rays(s: &mc_mods::ModSettings) -> f32 {
    if s.bool(CLASSIC, "light_rays") {
        s.f32(CLASSIC, "light_rays_strength") / 100.0
    } else {
        0.0
    }
}

/// Exclusivity: which pack stays on, given which are on now and which were on before. A pack
/// switched on since last time wins; otherwise the first one on, in `PackId::ALL` order.
pub(crate) fn exclusive_pack(on: [bool; 3], was_on: [bool; 3]) -> Option<usize> {
    (0..3)
        .find(|&i| on[i] && !was_on[i])
        .or_else(|| (0..3).find(|&i| on[i]))
}

impl Client {
    /// Register the core's phases (shadow map before the sky, bloom and tone
    /// map at `POST`) and attach the core to the terrain renderer. Runs from `Client::install`,
    /// so also after a resource-pack reload; the next frame re-syncs everything.
    pub(crate) fn install_shader_packs(
        &mut self,
        gpu: &Gpu,
        phases: &mut PhaseList,
        terrain: &Rc<RefCell<TerrainRenderer>>,
        entities: &Shared<EntityPhase>,
    ) {
        let _ = gpu;
        let core = self.tw_shaders.core.clone();
        terrain.borrow_mut().set_shader_core(Some(core.clone()));
        let mut shadow = ShadowPhase::new(core.clone(), Rc::clone(terrain));
        // Entities cast into a per-frame copy of the cached terrain map (only while the running
        // pack has entity shadows on; the entity phase is not recording at this point).
        let entities = entities.clone();
        shadow.set_entity_casters(Box::new(move |gpu, enc, target| {
            entities.with(|phase| {
                phase.record_pack_shadow(gpu, enc, &target.view, &target.uniform);
            });
        }));
        phases.insert(order::SHADOW, Box::new(shadow));
        let glow = Shared::new(SkyGlowPhase::new(core.clone(), PackId::Classic));
        phases.insert(order::WORLD, Box::new(glow.clone()));
        self.tw_shaders.classic_glow = Some(glow);
        let rays = Shared::new(RaysPhase::new(core.clone(), PackId::Classic));
        phases.insert(order::WORLD, Box::new(rays.clone()));
        self.tw_shaders.classic_rays = Some(rays);
        let final_phase = Shared::new(FinalPhase::new(core));
        phases.insert(order::FINAL, Box::new(final_phase.clone()));
        self.tw_shaders.final_phase = Some(final_phase);
        // Register your phases here (one line each), after the core's.
        self.install_tw_dreamy(phases, &self.tw_shaders.core().clone());
        self.tw_shaders.seen = 0;
    }

    /// Exclusivity and settings (when the settings generation moved), then,
    /// while a pack is switched on, this frame's [`PackFrame`] and each pack's own frame work.
    /// With every pack off: one `u64` compare and one `Option` test.
    pub(crate) fn shader_packs_frame(&mut self, gpu: &Gpu, camera: &Camera, partial_ticks: f32) {
        let generation = self.mods.settings.generation();
        if self.tw_shaders.seen != generation {
            self.tw_shaders.seen = generation;
            self.sync_shader_packs();
        }
        if self.tw_shaders.active.is_none() {
            if self.tw_shaders.was_running {
                self.stop_shader_packs();
            }
            return;
        }
        let frame = self.shader_pack_frame_inputs(gpu, camera, partial_ticks);
        self.tw_shaders.core.set_frame(gpu, frame);
        self.tw_shaders.frame_index = self.tw_shaders.frame_index.wrapping_add(1);
        // Your per-frame call goes here (after exclusivity and the PackFrame).
        self.tw_dreamy_core_frame(gpu);
        let running = self.tw_shaders.core.any_running();
        if self.tw_shaders.was_running && !running {
            self.release_shader_pack_targets();
        }
        self.tw_shaders.was_running = running;
        self.sync_shader_pack_mesh();
    }

    /// Exclusivity, the active pack, and Classic's config.
    fn sync_shader_packs(&mut self) {
        self.tw_shaders.time_changer_on = self.mods.settings.enabled("time_changer");
        let on = PackId::ALL.map(|pack| self.mods.settings.enabled(pack.mod_id()));
        let keep = exclusive_pack(on, self.tw_shaders.was_on);
        for (i, pack) in PackId::ALL.into_iter().enumerate() {
            if on[i] && Some(i) != keep {
                self.mods.settings.set_enabled(pack.mod_id(), false);
            }
        }
        let mut now = [false; 3];
        if let Some(i) = keep {
            now[i] = true;
        }
        self.tw_shaders.was_on = now;
        let active = keep.map(|i| PackId::ALL[i]);
        self.tw_shaders.active = active;
        self.tw_shaders.core.set_active(active);
        if active == Some(PackId::Classic) {
            let config = classic_config(&self.mods.settings);
            self.tw_shaders.core.set_config(config);
        }
        let rays = if active == Some(PackId::Classic) {
            classic_rays(&self.mods.settings)
        } else {
            0.0
        };
        if let Some(phase) = &self.tw_shaders.classic_rays {
            phase.with(|phase| phase.set_strength(rays));
        }
        let glow = if active == Some(PackId::Classic) {
            self.mods.settings.f32(CLASSIC, "sun_glow") / 100.0
        } else {
            0.0
        };
        if let Some(phase) = &self.tw_shaders.classic_glow {
            phase.with(|phase| phase.set_strength(glow));
        }
        // The generation moved again if a pack was switched off above; take that as seen.
        self.tw_shaders.seen = self.mods.settings.generation();
    }

    /// This frame's [`PackFrame`].
    fn shader_pack_frame_inputs(
        &self,
        gpu: &Gpu,
        camera: &Camera,
        partial_ticks: f32,
    ) -> PackFrame {
        let in_world = self.shader_pack_world_on_screen();
        let Some(render) = &self.render else {
            return PackFrame {
                in_world: false,
                ..PackFrame::default()
            };
        };
        let (environment, inputs, fog_distance) = render
            .fog
            .with(|phase| (phase.environment, phase.inputs, phase.distance()));
        let fog = FogState::compute(&environment, inputs, partial_ticks);
        let eye_medium = match inputs.medium {
            CameraMedium::Water => EyeMedium::Water,
            CameraMedium::Lava => EyeMedium::Lava,
            CameraMedium::Air | CameraMedium::Cloud => EyeMedium::Air,
        };
        PackFrame::compute(&FrameInputs {
            environment: &environment,
            biome_temperature: inputs.biome_temperature,
            fog_color: fog.color,
            fog: fog.mode,
            fog_distance,
            eye_medium,
            render_distance_chunks: self.settings.render_distance_chunks.clamp(2, 32) as u32,
            camera,
            width: gpu.config.width,
            height: gpu.config.height,
            partial_ticks,
            clock_held: self.shader_pack_clock_held(),
            // The client's render clock: the wall clock in normal play
            // (as before), the replay's time when one is watched or exported.
            seconds: self.render_clock().seconds as f32,
            // The per-frame dither and grain seed: in a replay (and its
            // export) a function of the replay's time (100 µs steps, so it still changes every
            // frame), the same at any frame rate and in every run; otherwise the frame count.
            frame_index: {
                let clock = self.render_clock();
                if clock.replay {
                    (clock.seconds * 10_000.0) as u32
                } else {
                    self.tw_shaders.frame_index
                }
            },
            in_world,
            sun_tilt: self.tw_shaders.core.config().sun_tilt,
        })
    }

    /// Whether the world clock the packs light by is held: the Time Changer shows a fixed time,
    /// or the server stopped the daylight cycle (a negative time in its time packets). The packs'
    /// sun and moon then stand still instead of swinging with the partial tick. Two field reads.
    pub(crate) fn shader_pack_clock_held(&self) -> bool {
        self.tw_shaders.time_changer_on
            || self
                .session
                .as_ref()
                .is_some_and(|session| session.state().env.daylight_cycle_frozen())
    }

    /// Whether a world is on screen: a session whose player exists and is past the
    /// downloading-terrain screen. Menus and loading screens get no pack.
    fn shader_pack_world_on_screen(&self) -> bool {
        !matches!(self.ui, Ui::Loading)
            && self.session.as_ref().is_some_and(|session| {
                session.predictor().is_some() && !session.terrain_download_active()
            })
    }

    /// Every pack switched off: free what the core and its phases hold and go back to vanilla
    /// meshing.
    fn stop_shader_packs(&mut self) {
        self.tw_shaders.core.set_active(None);
        self.tw_shaders.core.release_unused();
        self.release_shader_pack_targets();
        self.tw_shaders.was_running = false;
        self.sync_shader_pack_mesh();
    }

    fn release_shader_pack_targets(&mut self) {
        if let Some(phase) = &self.tw_shaders.final_phase {
            phase.with(FinalPhase::release_if_idle);
        }
        if let Some(phase) = &self.tw_shaders.classic_rays {
            phase.with(RaysPhase::release_if_idle);
        }
        if let Some(phase) = &self.tw_shaders.classic_glow {
            phase.with(SkyGlowPhase::release_if_idle);
        }
    }

    /// Fold the running pack's mesher needs into the mesh settings when they change. Packs need a
    /// pack mesh while switched on and single-sampled, in a world or not (so joining a world
    /// never re-meshes twice).
    fn sync_shader_pack_mesh(&mut self) {
        let core = &self.tw_shaders.core;
        let wanted = match core.active() {
            Some(_) if self.settings.msaa_samples <= 1 => core.config().mesh(),
            _ => PackMesh::VANILLA,
        };
        if wanted != self.tw_shaders.mesh {
            self.tw_shaders.mesh = wanted;
            self.sync_mesh_settings();
        }
    }

    /// The shader packs' mesher mode, folded into `wanted` beside the
    /// other mods' mesher switches. Vanilla (`PackMesh::VANILLA`) with every pack off.
    pub(crate) fn shader_pack_mesh_settings(&self, wanted: MeshSettings) -> MeshSettings {
        MeshSettings {
            pack: self.tw_shaders.mesh,
            ..wanted
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::config::{ClientConfig, LoginMode};
    use crate::net::ToGame;
    use crate::session::{JoinSettings, Session};
    use mc_protocol::play::{s2c, S2CPlay};

    /// A client on the real assets, or `None` (skip) without the vanilla jar. The `TempDir` is
    /// the game dir and must outlive the client.
    fn offline_client() -> Option<(Client, tempfile::TempDir)> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("SKIP: no vanilla jar in this checkout (set MC_JARS_DIR)");
            return None;
        }
        let client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        Some((client, dir))
    }

    /// The packs' held-clock rule ([`Client::shader_pack_clock_held`]): the clock the packs light
    /// by is held while the Time Changer shows a fixed time, or while the server has stopped the
    /// daylight cycle (a negative world time in its time packets, as vanilla 1.8.9's
    /// `WorldClient.setWorldTime` reads it), and runs otherwise. Driven the way the client gets
    /// there: the Time Changer through the mod settings and the settings sync a frame runs when
    /// they change, the server's cycle through time packets the session handles.
    #[test]
    fn the_packs_clock_is_held_by_the_time_changer_or_a_stopped_daylight_cycle() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let time_changer = |client: &mut Client, on: bool| {
            client.mods.settings.set_enabled("time_changer", on);
            client.sync_shader_packs();
        };
        // Neither: the Time Changer is off by default, and there is no world yet.
        time_changer(&mut client, false);
        assert!(!client.shader_pack_clock_held(), "nothing holds the clock");
        time_changer(&mut client, true);
        assert!(client.shader_pack_clock_held(), "the Time Changer holds it");
        time_changer(&mut client, false);
        assert!(
            !client.shader_pack_clock_held(),
            "the Time Changer is off again"
        );

        // A world whose server runs the daylight cycle, stops it, then runs it again.
        let (handle, to_game, _to_net) = crate::net::channel::NetHandle::new_pair();
        client.session = Some(Session::new(handle, JoinSettings::default()));
        let send = |client: &mut Client, packet: S2CPlay| {
            to_game
                .try_send(ToGame::Packet(packet))
                .expect("inbound channel");
            client.poll();
        };
        send(
            &mut client,
            S2CPlay::JoinGame(s2c::JoinGame {
                entity_id: 27,
                hardcore_mode: false,
                game_mode: 0,
                dimension: 0,
                difficulty: 2,
                max_players: 20,
                level_type: "default".into(),
                reduced_debug_info: false,
            }),
        );
        let time = |world_time: i64| {
            S2CPlay::TimeUpdate(s2c::TimeUpdate {
                total_world_time: 120_000,
                world_time,
            })
        };
        send(&mut client, time(6000));
        assert!(!client.shader_pack_clock_held(), "a running server clock");
        send(&mut client, time(-6000));
        assert!(
            client.shader_pack_clock_held(),
            "a server-stopped cycle holds it"
        );
        time_changer(&mut client, true);
        assert!(client.shader_pack_clock_held(), "both hold it");
        time_changer(&mut client, false);
        assert!(client.shader_pack_clock_held(), "the server still holds it");
        send(&mut client, time(6001));
        assert!(
            !client.shader_pack_clock_held(),
            "the server runs the cycle again"
        );
    }

    #[test]
    fn a_newly_switched_on_pack_wins_and_load_order_breaks_ties() {
        // Nothing on.
        assert_eq!(exclusive_pack([false; 3], [false; 3]), None);
        // Classic on, then Dreamy switched on: Dreamy wins.
        assert_eq!(
            exclusive_pack([true, true, false], [true, false, false]),
            Some(1)
        );
        // Underwater switched on while Dreamy was on.
        assert_eq!(
            exclusive_pack([false, true, true], [false, true, false]),
            Some(2)
        );
        // A settings file with two on and no history: the first in order.
        assert_eq!(exclusive_pack([false, true, true], [false; 3]), Some(1));
        // Turning the active one off leaves nothing.
        assert_eq!(exclusive_pack([false; 3], [true, false, false]), None);
    }

    #[test]
    fn the_classic_defaults_are_the_pvp_tuning() {
        let settings = mc_mods::ModSettings::new();
        assert!(!settings.enabled(CLASSIC), "off by default");
        let config = classic_config(&settings).sanitized();
        let shadows = config.shadows.expect("shadows on");
        assert_eq!(shadows.resolution, 2048);
        assert_eq!(shadows.distance, 96.0);
        let lighting = config.lighting.expect("sun shading on");
        // 40% linear is about 66% of the sunlit brightness on screen: shaded terrain stays easy
        // to read (players are never darkened by shade at all).
        assert!(lighting.shade_floor >= 0.35, "shade stays readable");
        let bloom = config.bloom.expect("bloom on");
        assert!(bloom.strength <= 0.2, "bloom never blinds");
        assert!(config.water.is_some() && config.waving.is_some() && config.tonemap.is_some());
        assert!(!config.mesh().is_vanilla());
    }
}
