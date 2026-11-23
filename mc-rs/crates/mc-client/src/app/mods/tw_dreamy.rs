//! The Dreamy shader pack's client half (`shaders_dreamy`).
//!
//! The render half is `mc_render::shaderpack::dreamy`: [`DreamyAtmospherePhase`] at
//! `shaderpack::order::WORLD` and [`DreamyFinishPhase`] at `shaderpack::order::FINAL`, both gated
//! on the shared [`ShaderCore`] running the Dreamy pack (switched on with exclusivity, a
//! world on screen, no MSAA). This file:
//!
//! * registers the two phases ([`Client::install_tw_dreamy`], from `Client::install`);
//! * on a settings-generation change, runs the preset logic (picking a preset writes its look
//!   into the look options; moving a look option by hand switches the preset to Custom) and maps
//!   the menu's values onto [`DreamySettings`] ([`dreamy_settings`], pure, tested);
//! * every frame while the pack is on: eases the ground fog's top toward "a set distance below
//!   your feet" (Below you mode) and advances the fog's wind drift.
//!
//! With the mod off a frame costs one `u64` compare and one `bool` test here, and both phases
//! report `enabled() == false` with no GPU resources.

use mc_math::DVec3;
use mc_mods::mods::shaders_dreamy::{self as desc, ids, Look, CUSTOM, LOOKS, LOOK_IDS};
use mc_mods::{ModSettings, OptionValue};
use mc_render::shaderpack::dreamy::{
    DreamyAtmospherePhase, DreamyFinishPhase, DreamySettings, FogSettings, RaySettings,
};
use mc_render::shaderpack::grade::{GradeParams, Wheel};
use mc_render::shaderpack::ground_fog::{FogLayer, FogResolution};
use mc_render::shaderpack::{
    order, CoreConfig, LightingSettings, PackId, ShaderCore, ShadowSettings,
};
use mc_render::{Gpu, PhaseList};

use super::super::{Client, Shared};

/// How quickly the fog's top follows your height in Below you mode: seconds to close ~63 % of the
/// gap. Jumps, bridging and falls never move it visibly; a new island height or map does, slowly.
const FOG_EASE_SECONDS: f64 = 8.0;
/// A height change bigger than this (a teleport, a new map, a respawn) moves the fog at once.
const FOG_SNAP_BLOCKS: f64 = 48.0;
/// Fog drift at 100 %, blocks per second.
const MAX_WIND: f64 = 3.0;

/// The pack's client state.
#[derive(Default)]
pub(crate) struct TwDreamyState {
    /// The settings generation last read (0 = never; the store starts at 1).
    seen: u64,
    /// The mod is switched on.
    on: bool,
    /// The preset last seen, `None` before the first sync (the loaded values are adopted).
    preset_seen: Option<usize>,
    /// The look options last seen, to tell a hand edit from a preset pick.
    look_seen: Option<[OptionValue; 27]>,
    /// The settings in effect.
    settings: Option<DreamySettings>,
    /// Below you mode (else World height).
    below_you: bool,
    /// Blocks between the feet and the fog's top (Below you).
    fog_depth: f64,
    /// The fog's top in World height mode.
    fog_level: f64,
    /// Fog drift speed, blocks per second.
    wind_speed: f64,
    /// The eased fog top, world y.
    fog_top: Option<f64>,
    /// The fog's accumulated drift.
    wind: DVec3,
    /// The render clock at the last frame, seconds.
    last_seconds: Option<f32>,
    /// The render clock's epoch at the last frame: a jump (a seek, a
    /// replay opened, an export pass) restarts the fog's ease from scratch.
    clock_epoch: Option<u32>,
    /// The pack ran last frame (its phases may hold GPU resources).
    was_running: bool,
    /// The shared-core features the pack asks for ([`dreamy_core_config`]), from the settings.
    core_config: CoreConfig,
    /// The settings generation `core_config` was computed at (0 = never).
    core_config_seen: u64,
    /// The shared shader core, from install.
    core: Option<ShaderCore>,
    atmosphere: Option<Shared<DreamyAtmospherePhase>>,
    finish: Option<Shared<DreamyFinishPhase>>,
}

/// `0xAARRGGBB` to display RGB in 0..1.
fn argb_rgb(argb: u32) -> [f32; 3] {
    let channel = |shift: u32| ((argb >> shift) & 0xFF) as f32 / 255.0;
    [channel(16), channel(8), channel(0)]
}

/// The look options' current values, in [`LOOK_IDS`] order.
fn read_look(settings: &ModSettings) -> [OptionValue; 27] {
    std::array::from_fn(|i| {
        settings
            .value(desc::ID, LOOK_IDS[i])
            .cloned()
            .unwrap_or(OptionValue::Action)
    })
}

/// Write a preset's look into the look options.
fn write_look(settings: &mut ModSettings, look: &Look) {
    for (id, value) in LOOK_IDS.iter().zip(look.values()) {
        settings.set_value(desc::ID, id, value);
    }
}

/// The menu's values as the render phases want them. Pure: the PvP tests run it on every preset.
pub(crate) fn dreamy_settings(settings: &ModSettings) -> DreamySettings {
    let id = desc::ID;
    let f = |option: &str| settings.f32(id, option);
    let pct = |option: &str| settings.f32(id, option) / 100.0;
    let rgb = |option: &str| argb_rgb(settings.color(id, option));
    let grade = GradeParams {
        exposure: f(ids::EXPOSURE),
        temperature: f(ids::TEMPERATURE) / 100.0,
        tint: f(ids::TINT) / 100.0,
        contrast: pct(ids::CONTRAST),
        saturation: pct(ids::SATURATION),
        vibrance: pct(ids::VIBRANCE),
        hue_shift: f(ids::HUE_SHIFT),
        lift: Wheel {
            hue: f(ids::SHADOWS_HUE),
            strength: pct(ids::SHADOWS_STRENGTH),
        },
        gamma: Wheel {
            hue: f(ids::MIDTONES_HUE),
            strength: pct(ids::MIDTONES_STRENGTH),
        },
        gain: Wheel {
            hue: f(ids::HIGHLIGHTS_HUE),
            strength: pct(ids::HIGHLIGHTS_STRENGTH),
        },
        split_shadow: rgb(ids::SPLIT_SHADOWS),
        split_highlight: rgb(ids::SPLIT_HIGHLIGHTS),
        split_balance: f(ids::SPLIT_BALANCE) / 100.0,
        split_strength: pct(ids::SPLIT_STRENGTH),
        gradient: [
            rgb(ids::GRADIENT_DARK),
            rgb(ids::GRADIENT_MID),
            rgb(ids::GRADIENT_LIGHT),
        ],
        gradient_mix: pct(ids::GRADIENT_MIX),
        fade: pct(ids::FADE),
    };
    let fog = settings.bool(id, ids::GROUND_FOG).then(|| {
        let softness = f(ids::FOG_SOFTNESS);
        FogSettings {
            layer: FogLayer {
                // Set per frame from the height mode.
                top: 0.0,
                softness,
                density: pct(ids::FOG_DENSITY) * 0.3,
                billows: 0.5 * softness,
                wisps: 0.55,
                start: f(ids::FOG_START),
                steps: 10,
                resolution: if settings.choice(id, ids::FOG_QUALITY) == 0 {
                    FogResolution::Half
                } else {
                    FogResolution::Quarter
                },
            },
            color: (settings.choice(id, ids::FOG_COLOR_MODE) != desc::FOG_FOLLOW_SKY)
                .then(|| rgb(ids::FOG_COLOR)),
            light: pct(ids::FOG_SUN_GLOW),
        }
    });
    let rays = settings.bool(id, ids::LIGHT_RAYS).then(|| RaySettings {
        strength: pct(ids::RAYS_STRENGTH),
        length: pct(ids::RAYS_LENGTH),
    });
    DreamySettings {
        grade,
        haze: pct(ids::HAZE),
        haze_start: f(ids::HAZE_START),
        sky_tint: rgb(ids::SKY_TINT),
        sky_tint_strength: pct(ids::SKY_TINT_STRENGTH),
        fog,
        rays,
        glow: pct(ids::GLOW),
        glow_radius: pct(ids::GLOW_RADIUS),
        halation: pct(ids::HALATION),
        vignette: pct(ids::VIGNETTE),
        grain: pct(ids::GRAIN),
        chromatic: pct(ids::CHROMATIC),
    }
}

/// The preset logic, on its own so it can be tested without a client: returns what to remember.
/// Picking a named preset writes its look; editing a look option by hand while a named preset is
/// picked switches to Custom. The first call adopts whatever was loaded.
fn run_presets(
    settings: &mut ModSettings,
    preset_seen: Option<usize>,
    look_seen: Option<&[OptionValue; 27]>,
) -> (usize, [OptionValue; 27]) {
    let preset = settings.choice(desc::ID, ids::PRESET);
    let mut look = read_look(settings);
    if let (Some(before), Some(look_before)) = (preset_seen, look_seen) {
        if preset != before && preset < CUSTOM {
            write_look(settings, &LOOKS[preset]);
            look = read_look(settings);
        } else if preset == before && preset < CUSTOM && look != *look_before {
            settings.set_choice(desc::ID, ids::PRESET, CUSTOM);
        }
    }
    (settings.choice(desc::ID, ids::PRESET), look)
}

/// The shared-core features the Dreamy pack asks for (the core): sun shading and soft
/// shadows, lighter than Classic's (shadow strength 35 % against 85 %, shade floor 60 % against
/// 45 %) so players in shade stay clear, on the same tilted sun path. Sanitized, so it compares
/// equal to what the core holds.
pub(crate) fn dreamy_core_config(settings: &ModSettings) -> CoreConfig {
    let id = desc::ID;
    let on = settings.enabled(id);
    let shadows = (on && settings.bool(id, ids::SOFT_SHADOWS)).then(|| ShadowSettings {
        distance: 96.0,
        resolution: 2048,
        softness: 3.0,
        strength: settings.f32(id, ids::SHADOW_STRENGTH) / 100.0,
        cutout_casters: true,
        // Players cast shadows; they never receive them.
        entity_casters: true,
    });
    let lighting = (on && settings.bool(id, ids::SUNLIGHT)).then_some(LightingSettings {
        sun_share: 0.5,
        warmth: 1.3,
        night_blue: 1.2,
        shade_floor: 0.6,
    });
    let tilt = if shadows.is_some() || lighting.is_some() {
        25.0
    } else {
        0.0
    };
    CoreConfig {
        sun_tilt: tilt,
        shadows,
        lighting,
        ..CoreConfig::OFF
    }
    .sanitized()
}

impl Client {
    /// Register the Dreamy pack's two phases on the shared core. Runs from
    /// `Client::install`, so also after a resource-pack reload, which rebuilds every phase; the
    /// new phases get the settings on the next frame.
    pub(crate) fn install_tw_dreamy(&mut self, phases: &mut PhaseList, core: &ShaderCore) {
        let atmosphere = Shared::new(DreamyAtmospherePhase::new(core.clone()));
        let finish = Shared::new(DreamyFinishPhase::new(core.clone()));
        phases.insert(order::WORLD, Box::new(atmosphere.clone()));
        phases.insert(order::FINAL, Box::new(finish.clone()));
        let state = std::mem::take(&mut self.tw_dreamy);
        self.tw_dreamy = TwDreamyState {
            core: Some(core.clone()),
            atmosphere: Some(atmosphere),
            finish: Some(finish),
            // Keep the preset memory and the eased fog across a reload.
            preset_seen: state.preset_seen,
            look_seen: state.look_seen,
            fog_top: state.fog_top,
            wind: state.wind,
            ..TwDreamyState::default()
        };
    }

    /// In the `shader_packs_frame`, right after the core's exclusivity and
    /// `PackFrame` and before its mesher sync and the frame: while Dreamy is the active pack, hand
    /// the core exactly the features Dreamy wants, every frame (`set_config` compares), so nothing
    /// of another pack's config (Classic's bloom, tone map, stronger shadows) carries over. When
    /// that changed the config (a pack switch or a settings change), the core's per-frame state
    /// (light colours, shadow map) is derived again at once, so not even the switch frame shows
    /// the old config.
    pub(crate) fn tw_dreamy_core_frame(&mut self, gpu: &Gpu) {
        let Some(core) = self.tw_dreamy.core.clone() else {
            return;
        };
        if core.active() != Some(PackId::Dreamy) {
            return;
        }
        let generation = self.mods.settings.generation();
        if self.tw_dreamy.core_config_seen != generation {
            self.tw_dreamy.core_config_seen = generation;
            self.tw_dreamy.core_config = dreamy_core_config(&self.mods.settings);
        }
        if core.config() != self.tw_dreamy.core_config {
            core.set_config(self.tw_dreamy.core_config);
            core.set_frame(gpu, core.frame());
        }
    }

    /// Once per frame, among the per-frame mod hooks: settings on a generation change, then (pack
    /// on) the fog's height and drift.
    pub(crate) fn tw_dreamy_frame(&mut self, partial_ticks: f32) {
        let generation = self.mods.settings.generation();
        if self.tw_dreamy.seen != generation {
            self.sync_tw_dreamy();
            self.tw_dreamy.seen = self.mods.settings.generation();
        }
        if !self.tw_dreamy.on {
            return;
        }
        let Some(core) = self.tw_dreamy.core.clone() else {
            return;
        };
        // The core features are handed over earlier in the frame (`tw_dreamy_core_frame`).
        if !core.is_running(PackId::Dreamy) {
            // No world on screen, MSAA, or another pack: a disabled phase is never recorded, so
            // free what the phases built here, once.
            if self.tw_dreamy.was_running {
                self.tw_dreamy.was_running = false;
                self.tw_dreamy.last_seconds = None;
                self.release_tw_dreamy();
            }
            return;
        }
        self.tw_dreamy.was_running = true;
        // `PackFrame.seconds` is the client's render clock. When it jumps,
        // the fog's ease starts again from its target (deterministic in replays and exports).
        let clock = self.render_clock();
        if self.tw_dreamy.clock_epoch != Some(clock.epoch) {
            self.tw_dreamy.clock_epoch = Some(clock.epoch);
            self.tw_dreamy.last_seconds = None;
            self.tw_dreamy.fog_top = None;
        }
        let seconds = core.frame().seconds;
        let dt = self
            .tw_dreamy
            .last_seconds
            .map_or(0.0, |last| f64::from((seconds - last).clamp(0.0, 0.25)));
        self.tw_dreamy.last_seconds = Some(seconds);

        let fog_on = self
            .tw_dreamy
            .settings
            .is_some_and(|settings| settings.fog.is_some());
        if !fog_on {
            return;
        }
        let target = if self.tw_dreamy.below_you {
            let feet = self
                .session
                .as_ref()
                .and_then(|session| session.render_view_position(partial_ticks));
            match feet {
                Some(feet) => feet.y - self.tw_dreamy.fog_depth,
                None => return,
            }
        } else {
            self.tw_dreamy.fog_level
        };
        let top = match self.tw_dreamy.fog_top {
            Some(top) if (target - top).abs() <= FOG_SNAP_BLOCKS && self.tw_dreamy.below_you => {
                top + (target - top) * (1.0 - (-dt / FOG_EASE_SECONDS).exp())
            }
            _ => target,
        };
        self.tw_dreamy.fog_top = Some(top);
        let speed = self.tw_dreamy.wind_speed;
        if clock.replay {
            // A replay (and its export): the drift is a function of the
            // replay's time alone, the same at any frame rate and in every run.
            self.tw_dreamy.wind = DVec3::new(1.0, 0.12, 0.35) * (speed * clock.seconds);
        } else {
            self.tw_dreamy.wind += DVec3::new(1.0, 0.12, 0.35) * (speed * dt);
        }
        let wind = self.tw_dreamy.wind;
        // A custom sky stays in view below the horizon: no fog sea on it from above the fog, as
        // far as its layers cover the sky (the last frame's draw).
        let keep_sky = self.custom_sky_cover(core.frame().dimension);
        if let Some(atmosphere) = &self.tw_dreamy.atmosphere {
            atmosphere.with(|phase| {
                phase.set_fog_top(top as f32);
                phase.set_wind(wind);
                phase.set_keep_sky(keep_sky);
            });
        }
    }

    /// Read the mod's settings (only when the generation moved): the preset logic, then push.
    fn sync_tw_dreamy(&mut self) {
        let on = self.mods.settings.enabled(desc::ID);
        self.tw_dreamy.on = on;
        if !on {
            // Off: the phases drop their resources; the presets are remembered as they are.
            self.tw_dreamy.settings = None;
            self.tw_dreamy.last_seconds = None;
            self.push_tw_dreamy(DreamySettings::default());
            return;
        }
        let (preset, look) = run_presets(
            &mut self.mods.settings,
            self.tw_dreamy.preset_seen,
            self.tw_dreamy.look_seen.as_ref(),
        );
        self.tw_dreamy.preset_seen = Some(preset);
        self.tw_dreamy.look_seen = Some(look);
        let settings = &self.mods.settings;
        let id = desc::ID;
        self.tw_dreamy.below_you = settings.choice(id, ids::FOG_HEIGHT_MODE) == desc::FOG_BELOW_YOU;
        self.tw_dreamy.fog_depth = f64::from(settings.f32(id, ids::FOG_DEPTH));
        self.tw_dreamy.fog_level = f64::from(settings.f32(id, ids::FOG_LEVEL));
        self.tw_dreamy.wind_speed = f64::from(settings.f32(id, ids::FOG_WIND)) / 100.0 * MAX_WIND;
        let dreamy = dreamy_settings(settings);
        self.tw_dreamy.settings = Some(dreamy);
        self.push_tw_dreamy(dreamy);
    }

    fn release_tw_dreamy(&self) {
        if let Some(atmosphere) = &self.tw_dreamy.atmosphere {
            atmosphere.with(DreamyAtmospherePhase::release);
        }
        if let Some(finish) = &self.tw_dreamy.finish {
            finish.with(DreamyFinishPhase::release);
        }
    }

    fn push_tw_dreamy(&self, settings: DreamySettings) {
        if let Some(atmosphere) = &self.tw_dreamy.atmosphere {
            atmosphere.with(|phase| phase.set_settings(settings));
        }
        if let Some(finish) = &self.tw_dreamy.finish {
            finish.with(|phase| phase.set_settings(settings));
        }
    }

    /// Whether the Dreamy pack's phases hold GPU resources: `(atmosphere, finish)`. For tools
    /// and tests that drive a client directly (`tests/tw_dreamy_pictures.rs`): off must hold
    /// none.
    pub fn dreamy_pack_allocated(&self) -> (bool, bool) {
        (
            self.tw_dreamy
                .atmosphere
                .as_ref()
                .is_some_and(|phase| phase.with(|p| p.allocated())),
            self.tw_dreamy
                .finish
                .as_ref()
                .is_some_and(|phase| phase.with(|p| p.allocated())),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_render::shaderpack::grade::{linear_to_oklab, srgb_to_linear};
    use mc_render::shaderpack::ground_fog::FogLayer;

    fn settings_with(look: &Look) -> ModSettings {
        let mut settings = ModSettings::new();
        settings.set_enabled(desc::ID, true);
        write_look(&mut settings, look);
        settings
    }

    fn oklab(display: [f32; 3]) -> mc_math::Vec3 {
        linear_to_oklab(srgb_to_linear(mc_math::Vec3::from(display)))
    }

    fn rgb8(r: u8, g: u8, b: u8) -> [f32; 3] {
        [r, g, b].map(|v| f32::from(v) / 255.0)
    }

    /// Team red and blue as they reach the screen in Minecraft 1.8: chat and nametag colours
    /// (§c/§9, §4/§1), wool, and dyed leather armour.
    fn team_pairs() -> [([f32; 3], [f32; 3], &'static str); 4] {
        [
            (rgb8(255, 85, 85), rgb8(85, 85, 255), "nametag red / blue"),
            (rgb8(170, 0, 0), rgb8(0, 0, 170), "dark red / dark blue"),
            (rgb8(150, 52, 48), rgb8(46, 56, 141), "red / blue wool"),
            (rgb8(176, 46, 38), rgb8(60, 68, 170), "red / blue leather"),
        ]
    }

    /// Whether a grade keeps team red red and blue blue (OKLab a and b keep their order with a
    /// clear margin) and at least half of their difference: `Err` names the first failure.
    fn keeps_teams_apart(grade: &GradeParams) -> Result<(), String> {
        for (red, blue, what) in team_pairs() {
            let (r0, b0) = (oklab(red), oklab(blue));
            let (r1, b1) = (oklab(grade.apply(red)), oklab(grade.apply(blue)));
            if r1.y - b1.y <= 0.08 {
                return Err(format!("{what}: a {} vs {}", r1.y, b1.y));
            }
            if r1.z - b1.z <= 0.08 {
                return Err(format!("{what}: b {} vs {}", r1.z, b1.z));
            }
            let (before, after) = ((r0 - b0).length(), (r1 - b1).length());
            if after <= 0.5 * before {
                return Err(format!("{what}: {after} of {before}"));
            }
        }
        Ok(())
    }

    /// PvP: under every preset, team red stays red and blue stays blue. The check itself rejects
    /// grades that do hide teams (a full gradient map, no saturation): the negative control.
    #[test]
    fn every_preset_keeps_team_red_and_blue_apart() {
        for (index, look) in LOOKS.iter().enumerate() {
            let grade = dreamy_settings(&settings_with(look)).grade;
            if let Err(why) = keeps_teams_apart(&grade) {
                panic!("{}: {why}", desc::PRESETS[index]);
            }
        }
        let full_map = GradeParams {
            gradient: [[0.1, 0.0, 0.3], [0.8, 0.3, 0.5], [1.0, 0.9, 0.7]],
            gradient_mix: 1.0,
            ..GradeParams::IDENTITY
        };
        assert!(
            keeps_teams_apart(&full_map).is_err(),
            "a full gradient map hides teams"
        );
        let grey = GradeParams {
            saturation: 0.0,
            ..GradeParams::IDENTITY
        };
        assert!(
            keeps_teams_apart(&grey).is_err(),
            "black and white hides teams"
        );
    }

    /// PvP: white nametag text stays bright and black backgrounds stay dark under every preset.
    #[test]
    fn every_preset_keeps_nametag_contrast() {
        for (index, look) in LOOKS.iter().enumerate() {
            let grade = dreamy_settings(&settings_with(look)).grade;
            let white = oklab(grade.apply([1.0; 3])).x;
            let black = oklab(grade.apply([0.0; 3])).x;
            let name = desc::PRESETS[index];
            assert!(white > 0.85, "{name}: white L {white}");
            assert!(black < 0.3, "{name}: black L {black}");
        }
    }

    /// The defaults are the Dreamy preset with fog, rays, haze and the lens on.
    #[test]
    fn the_defaults_map_onto_a_full_pack() {
        let mut settings = ModSettings::new();
        settings.set_enabled(desc::ID, true);
        let dreamy = dreamy_settings(&settings);
        assert!(dreamy.atmosphere_on() && dreamy.finish_on());
        assert!(dreamy.fog.is_some() && dreamy.rays.is_some());
        assert_eq!(dreamy.chromatic, 0.0);
        assert!(!dreamy.grade.is_identity());
        // The shared core: soft shadows lighter than Classic's, sunlight, no core bloom or tone
        // map (Dreamy has its own glow and grade).
        let core = dreamy_core_config(&settings);
        let shadows = core.shadows.expect("soft shadows on by default");
        assert!(shadows.strength <= 0.4, "{shadows:?}");
        assert!(core.lighting.is_some_and(|l| l.shade_floor >= 0.55));
        assert!(core.bloom.is_none() && core.tonemap.is_none() && core.water.is_none());
        assert_eq!(
            core,
            core.sanitized(),
            "stored sanitized so it compares with the core's"
        );
        // Off: nothing.
        settings.set_enabled(desc::ID, false);
        assert_eq!(dreamy_core_config(&settings), CoreConfig::OFF.sanitized());
    }

    /// Switching straight from Classic to Dreamy hands the shared core exactly Dreamy's features
    /// in that same frame: nothing of Classic's config (bloom, tone map, its stronger shadows)
    /// carries over, not even for one frame. Skips without the vanilla jar or a GPU.
    #[test]
    fn a_switch_from_classic_hands_over_exactly_dreamys_core_config_in_the_same_frame() {
        use crate::app::config::{ClientConfig, LoginMode};
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("SKIP: no vanilla jar in this checkout");
            return;
        }
        let gpu = match mc_render::Gpu::headless(8, 8, mc_render::GpuOptions::default()) {
            Ok(gpu) => gpu,
            Err(mc_render::RenderError::NoAdapter) => {
                eprintln!("SKIP: no wgpu adapter");
                return;
            }
            Err(error) => panic!("headless context: {error}"),
        };
        let mut client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        let mut phases = PhaseList::new();
        client.install(&gpu, &mut phases, 8, 8);
        let mut camera = mc_render::Camera::default();
        client.mods.settings.set_enabled("shaders_classic", true);
        client.shader_packs_frame(&gpu, &camera, 1.0);
        let core = client.tw_shaders.core().clone();
        assert_eq!(core.active(), Some(PackId::Classic));
        assert!(core.config().tonemap.is_some(), "Classic runs its tone map");
        // Switch to Dreamy: exclusivity turns Classic off, and in this very frame the core holds
        // Dreamy's features and nothing else.
        client.mods.settings.set_enabled(desc::ID, true);
        client.shader_packs_frame(&gpu, &camera, 1.0);
        assert_eq!(core.active(), Some(PackId::Dreamy));
        assert!(!client.mods.settings.enabled("shaders_classic"));
        let wanted = dreamy_core_config(&client.mods.settings);
        assert_eq!(core.config(), wanted);
        assert!(core.config().tonemap.is_none() && core.config().bloom.is_none());
        // And it stays that way frame after frame (the glue's full frame included).
        for _ in 0..3 {
            client.shader_packs_frame(&gpu, &camera, 1.0);
            client.tw_dreamy_frame(1.0);
            assert_eq!(core.config(), wanted);
        }
        camera.update();
    }

    /// Picking a preset writes its look; a hand edit switches to Custom; the first sync adopts
    /// what was loaded; Custom never writes anything.
    #[test]
    fn presets_set_the_sliders_and_hand_edits_make_custom() {
        let mut settings = ModSettings::new();
        settings.set_enabled(desc::ID, true);
        // First sync adopts the loaded values even if they differ from the preset.
        settings.set_f32(desc::ID, ids::GLOW, 77.0);
        let (preset, look) = run_presets(&mut settings, None, None);
        assert_eq!(preset, 0);
        assert_eq!(settings.f32(desc::ID, ids::GLOW), 77.0);
        // Pick Dusk: its look lands in the sliders.
        settings.set_choice(desc::ID, ids::PRESET, 1);
        let (preset, look) = run_presets(&mut settings, Some(preset), Some(&look));
        assert_eq!(preset, 1);
        assert_eq!(look, LOOKS[1].values());
        assert_eq!(
            settings.f32(desc::ID, ids::TEMPERATURE),
            LOOKS[1].temperature
        );
        // Nothing changed: nothing happens.
        let (preset, look) = run_presets(&mut settings, Some(preset), Some(&look));
        assert_eq!(preset, 1);
        // A hand edit: Custom, and the edit stays.
        settings.set_f32(desc::ID, ids::SATURATION, 150.0);
        let (preset, look) = run_presets(&mut settings, Some(preset), Some(&look));
        assert_eq!(preset, CUSTOM);
        assert_eq!(settings.f32(desc::ID, ids::SATURATION), 150.0);
        // Under Custom, edits never switch anything.
        settings.set_f32(desc::ID, ids::FADE, 40.0);
        let (preset, _) = run_presets(&mut settings, Some(preset), Some(&look));
        assert_eq!(preset, CUSTOM);
        assert_eq!(settings.f32(desc::ID, ids::FADE), 40.0);
    }

    /// PvP: with the default ground fog in Below you mode, a player standing at your height is
    /// never fogged at fighting range or on the next island, and one 8 blocks below you (an
    /// island lower down) keeps nearly all of their contrast. (The first, exponential, profile
    /// failed this: its tail put 33 % fog on the lower player at 30 blocks; the fog's fade now
    /// ends at a ceiling, `FogLayer::ceiling`, under the feet.)
    #[test]
    fn default_fog_never_hides_players_at_your_height() {
        let mut settings = ModSettings::new();
        settings.set_enabled(desc::ID, true);
        let dreamy = dreamy_settings(&settings);
        let fog = dreamy.fog.expect("fog on by default");
        let feet = 74.0_f32;
        let depth = settings.f32(desc::ID, ids::FOG_DEPTH);
        let layer = FogLayer {
            top: feet - depth,
            ..fog.layer
        };
        let eye = feet + 1.62;
        for distance in [3.0_f32, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0] {
            // The target's body centre, level and 8 blocks lower.
            let level = layer.transmittance(eye, mc_math::Vec3::new(distance, -0.7, 0.0));
            assert!(level > 0.99, "level at {distance}: {level}");
            let lower = layer.transmittance(eye, mc_math::Vec3::new(distance, -8.7, 0.0));
            assert!(lower > 0.85, "8 below at {distance}: {lower}");
        }
    }
}
