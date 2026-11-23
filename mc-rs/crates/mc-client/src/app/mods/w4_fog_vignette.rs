//! The Visual window's **Fog** and **Vignette** mods. Both
//! default on, and on is vanilla.
//!
//! Hooks, both `&self` because the driver calls them inside `render.*.with(..)` closures while
//! it holds a shared borrow of `self.render`:
//! * [`Client::mod_fog_inputs`] in `driver.rs::drive_atmosphere`, right after
//!   every other `FogInputs` field is written. The renderer turns the two flags into the falloff
//!   in one place (`mc_render::sky::fog::world_pass_fog_mode`), and every fog consumer is fed
//!   from those inputs: the screen-space fog pass (terrain, entities, particles, weather), the
//!   clouds (the `fog_state` handed to `CloudsPhase::set_fog`), the sky pass (the inputs copied
//!   into `SkyPhase::fog_inputs`) and the clear colour. This one line is the whole client side
//!   of Fog.
//! * [`Client::mod_vignette`] in `drive_atmosphere`'s overlay closure:
//!   `OverlayPhase::ordinary_vignette`.
//!
//! The settings snapshot lives in `Cell`s and is refreshed once per settings generation. With
//! both mods at their vanilla default a hook costs one `u64` compare and a few `bool` stores; no
//! allocation, lock, hashing or formatting. Nothing here sends a packet: both mods change only
//! what is drawn.

use std::cell::Cell;

use mc_mods::mods::fog;
use mc_mods::ModSettings;
use mc_render::post::OverlayPhase;
use mc_render::sky::FogInputs;

use super::super::Client;

/// The two mods' settings, in the renderer's terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct W4Settings {
    /// `fog` on: vanilla's render-distance fog ([`FogInputs::distance_fog`]).
    distance_fog: bool,
    /// `fog`'s "Clear water and lava" ([`FogInputs::clear_liquids`]). Read whether the mod is on
    /// or off: the menu shows it in both states and its label promises it works in both.
    clear_liquids: bool,
    /// `vignette` on: vanilla's brightness vignette ([`OverlayPhase::ordinary_vignette`]).
    vignette: bool,
}

impl Default for W4Settings {
    /// Vanilla, and the renderer's own defaults.
    fn default() -> Self {
        Self {
            distance_fog: true,
            clear_liquids: false,
            vignette: true,
        }
    }
}

impl W4Settings {
    fn read(s: &ModSettings) -> Self {
        Self {
            distance_fog: s.enabled("fog"),
            clear_liquids: s.bool("fog", fog::CLEAR_LIQUIDS),
            vignette: s.enabled("vignette"),
        }
    }
}

/// The client state: a typed snapshot of the two mods' settings.
#[derive(Debug, Default)]
pub(crate) struct W4State {
    /// The settings generation `settings` was read at; `0` = never (the store starts at 1).
    seen: Cell<u64>,
    settings: Cell<W4Settings>,
}

impl W4State {
    /// The snapshot, re-read only when the settings changed since the last call.
    fn get(&self, s: &ModSettings) -> W4Settings {
        let generation = s.generation();
        if self.seen.get() != generation {
            self.seen.set(generation);
            self.settings.set(W4Settings::read(s));
        }
        self.settings.get()
    }
}

impl Client {
    /// Fog: write the mod's two flags into this frame's fog inputs.
    pub(crate) fn mod_fog_inputs(&self, inputs: &mut FogInputs) {
        let settings = self.w4.get(&self.mods.settings);
        inputs.distance_fog = settings.distance_fog;
        inputs.clear_liquids = settings.clear_liquids;
    }

    /// Vignette: whether the ordinary darkening draws this frame. The
    /// world-border warning is never affected.
    pub(crate) fn mod_vignette(&self, phase: &mut OverlayPhase) {
        phase.ordinary_vignette = self.w4.get(&self.mods.settings).vignette;
    }
}

#[cfg(test)]
mod tests {
    use mc_render::sky::fog::world_pass_fog_mode;
    use mc_render::sky::{CameraMedium, FogMode, FogState};
    use mc_render::{Camera, Gpu, GpuOptions, PhaseList};

    use super::*;

    fn settings() -> ModSettings {
        ModSettings::new()
    }

    /// A fresh install is vanilla: both mods on, liquids not cleared.
    #[test]
    fn defaults_are_vanilla() {
        let s = settings();
        assert!(s.enabled("fog") && s.enabled("vignette"));
        assert_eq!(W4Settings::read(&s), W4Settings::default());
        let mut inputs = FogInputs::default();
        let before = format!("{inputs:?}");
        inputs.distance_fog = W4Settings::read(&s).distance_fog;
        inputs.clear_liquids = W4Settings::read(&s).clear_liquids;
        assert_eq!(
            format!("{inputs:?}"),
            before,
            "the renderer's defaults are vanilla too"
        );
    }

    #[test]
    fn the_snapshot_follows_each_switch_and_rereads_only_on_change() {
        let mut s = settings();
        let state = W4State::default();
        assert_eq!(state.get(&s), W4Settings::default());

        s.set_enabled("fog", false);
        assert!(!state.get(&s).distance_fog);
        s.set_bool("fog", fog::CLEAR_LIQUIDS, true);
        assert!(state.get(&s).clear_liquids);
        s.set_enabled("vignette", false);
        assert_eq!(
            state.get(&s),
            W4Settings {
                distance_fog: false,
                clear_liquids: true,
                vignette: false,
            }
        );
        // "Clear water and lava" is read with the mod on too.
        s.set_enabled("fog", true);
        assert_eq!(
            state.get(&s),
            W4Settings {
                distance_fog: true,
                clear_liquids: true,
                vignette: false,
            }
        );

        // An unchanged generation never re-reads: a planted stale value proves the early return.
        state.settings.set(W4Settings::default());
        assert_eq!(state.get(&s), W4Settings::default());
        s.set_enabled("vignette", true);
        assert_eq!(
            state.get(&s),
            W4Settings {
                distance_fog: true,
                clear_liquids: true,
                vignette: true,
            }
        );
    }

    /// Every hook is called at its site, once, and the fog hook sits after the last `FogInputs`
    /// write and before the state that feeds the clouds, the sky and the clear colour is taken.
    #[test]
    fn every_hook_is_wired_at_its_driver_site() {
        let driver = include_str!("../driver.rs");
        for hook in [
            "self.mod_fog_inputs(&mut phase.inputs);",
            "self.mod_vignette(phase);",
            "w4: Default::default(),",
        ] {
            assert_eq!(driver.matches(hook).count(), 1, "{hook}");
        }
        let atmosphere = driver.find("fn drive_atmosphere(").unwrap();
        let temperature = atmosphere
            + driver[atmosphere..]
                .find("phase.inputs.biome_temperature = temp;")
                .unwrap();
        let fog_hook = driver
            .find("self.mod_fog_inputs(&mut phase.inputs);")
            .unwrap();
        let state = atmosphere
            + driver[atmosphere..]
                .find("FogState::compute(&phase.environment, phase.inputs, partial_ticks),")
                .unwrap();
        let clouds = atmosphere
            + driver[atmosphere..]
                .find(".with(|phase| phase.set_fog(Some((fog_state, fog_distance))));")
                .unwrap();
        let sky = atmosphere
            + driver[atmosphere..]
                .find("let sky_fog_inputs = render.fog.with(|phase| phase.inputs);")
                .unwrap();
        assert!(
            temperature < fog_hook && fog_hook < state && state < clouds && clouds < sky,
            "the flags reach the fog pass, the clouds and the sky in this frame"
        );
        let vignette = atmosphere
            + driver[atmosphere..]
                .find("phase.set_vignette_brightness(vignette_brightness);")
                .unwrap();
        let vignette_hook = driver.find("self.mod_vignette(phase);").unwrap();
        assert!(vignette < vignette_hook);
        assert!(
            driver[vignette..vignette_hook].find("});").is_none(),
            "same overlay closure"
        );
    }

    fn offline_client() -> Option<(Client, tempfile::TempDir)> {
        use crate::app::config::{ClientConfig, LoginMode};
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
        let client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        Some((client, dir))
    }

    /// What `drive_atmosphere` handed the installed phases this frame.
    #[derive(Debug, PartialEq)]
    struct Fed {
        fog_inputs: String,
        fog_pass: bool,
        world_mode: FogMode,
        sky_distance_fog: bool,
        sky_clear_liquids: bool,
        clear_color: [u64; 3],
        ordinary_vignette: bool,
    }

    fn drive(client: &mut Client, gpu: &Gpu) -> Fed {
        let camera = Camera::default();
        client.drive_atmosphere(gpu, &camera, 0.5);
        let render = client.render.as_ref().expect("installed");
        let (fog_inputs, fog_pass, world_mode) = render.fog.with(|phase| {
            use mc_render::RenderPhase as _;
            (
                format!("{:?}", phase.inputs),
                phase.enabled(),
                world_pass_fog_mode(phase.inputs),
            )
        });
        let (sky_distance_fog, sky_clear_liquids) = render.sky.with(|phase| {
            (
                phase.fog_inputs.distance_fog,
                phase.fog_inputs.clear_liquids,
            )
        });
        let clear = client.pending_clear_color;
        Fed {
            fog_inputs,
            fog_pass,
            world_mode,
            sky_distance_fog,
            sky_clear_liquids,
            clear_color: [clear.r.to_bits(), clear.g.to_bits(), clear.b.to_bits()],
            ordinary_vignette: render.overlay.with(|phase| phase.ordinary_vignette),
        }
    }

    /// The real driver on a real installed client: the defaults feed vanilla inputs, Fog off
    /// reaches the fog pass, the clouds' state and the sky inputs in the same frame without
    /// touching the clear colour, Vignette off reaches the overlay, and switching both back on
    /// is the vanilla frame again.
    #[test]
    fn a_real_client_feeds_the_flags_to_every_fog_consumer_and_back() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let Ok(gpu) = Gpu::headless(
            32,
            32,
            GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        ) else {
            eprintln!("skip: no wgpu adapter available on this machine");
            return;
        };
        let mut phases = PhaseList::new();
        client.install(&gpu, &mut phases, 32, 32);

        let vanilla = drive(&mut client, &gpu);
        assert!(vanilla.fog_pass, "vanilla air fog draws");
        assert!(matches!(vanilla.world_mode, FogMode::Linear { .. }));
        assert!(!vanilla.world_mode.is_clear());
        assert!(vanilla.sky_distance_fog && !vanilla.sky_clear_liquids);
        assert!(vanilla.ordinary_vignette);
        assert_eq!(
            vanilla.fog_inputs,
            format!(
                "{:?}",
                render_inputs_with(&client, |inputs| {
                    inputs.distance_fog = true;
                    inputs.clear_liquids = false;
                })
            ),
            "with the mods at their defaults the inputs are the renderer's vanilla defaults"
        );

        client.mods.settings.set_enabled("fog", false);
        client.mods.settings.set_enabled("vignette", false);
        let off = drive(&mut client, &gpu);
        assert!(!off.fog_pass, "no fog pass at all for an air eye");
        assert!(off.world_mode.is_clear());
        assert!(
            !off.sky_distance_fog,
            "the sky gets the same inputs (and keeps its fade)"
        );
        assert_eq!(
            off.clear_color, vanilla.clear_color,
            "the colour never changes"
        );
        assert!(!off.ordinary_vignette);
        // The clouds were handed the same state as the fog pass computes from these inputs.
        let clouds_state = render_state(&client);
        assert!(clouds_state.mode.is_clear());

        client
            .mods
            .settings
            .set_bool("fog", fog::CLEAR_LIQUIDS, true);
        let cleared = drive(&mut client, &gpu);
        assert!(cleared.sky_clear_liquids);

        client.mods.settings.set_enabled("fog", true);
        client.mods.settings.set_enabled("vignette", true);
        client
            .mods
            .settings
            .set_bool("fog", fog::CLEAR_LIQUIDS, false);
        assert_eq!(
            drive(&mut client, &gpu),
            vanilla,
            "back on is vanilla again"
        );
    }

    /// The fog pass's current inputs after `edit`.
    fn render_inputs_with(client: &Client, edit: impl FnOnce(&mut FogInputs)) -> FogInputs {
        let mut inputs = client
            .render
            .as_ref()
            .expect("installed")
            .fog
            .with(|phase| phase.inputs);
        edit(&mut inputs);
        inputs
    }

    /// The world-pass state the driver computed for the clouds this frame (recomputed from the
    /// fog pass's own fields, which is exactly what the driver does).
    fn render_state(client: &Client) -> FogState {
        client
            .render
            .as_ref()
            .expect("installed")
            .fog
            .with(|phase| phase.state())
    }

    /// Water and lava keep their thick fog unless "Clear water and lava" is on, whatever the
    /// main switch says; blindness always keeps its fog.
    #[test]
    fn the_inputs_select_the_documented_falloff() {
        let client_inputs =
            |distance_fog: bool, clear_liquids: bool, medium: CameraMedium| FogInputs {
                medium,
                distance_fog,
                clear_liquids,
                ..FogInputs::default()
            };
        let vanilla = |medium| world_pass_fog_mode(client_inputs(true, false, medium));
        for medium in [CameraMedium::Water, CameraMedium::Lava, CameraMedium::Cloud] {
            assert_eq!(
                world_pass_fog_mode(client_inputs(false, false, medium)),
                vanilla(medium),
                "{medium:?} keeps its fog with Fog off"
            );
        }
        assert!(world_pass_fog_mode(client_inputs(false, false, CameraMedium::Air)).is_clear());
        assert_eq!(
            world_pass_fog_mode(client_inputs(true, true, CameraMedium::Water)),
            vanilla(CameraMedium::Air)
        );
        assert!(world_pass_fog_mode(client_inputs(false, true, CameraMedium::Lava)).is_clear());
        let blind = FogInputs {
            blindness: 1.0,
            ..client_inputs(false, true, CameraMedium::Water)
        };
        assert_eq!(
            world_pass_fog_mode(blind),
            world_pass_fog_mode(FogInputs {
                blindness: 1.0,
                ..FogInputs::default()
            }),
            "blindness fog always stays"
        );
    }
}
