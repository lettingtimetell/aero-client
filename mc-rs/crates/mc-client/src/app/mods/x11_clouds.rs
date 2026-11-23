//! The client half of `volumetric_clouds`.
//!
//! The renderer is `mc_render::sky::volumetric` (see its module docs for the technique, the cost
//! and how a shader pack would plug in). This file decides when the clouds draw and feeds them:
//!
//! * **Off** (the default): the two phase slots `install` registered stay disabled and nothing is
//!   allocated. The frame hook is one `u64` compare and two `bool` tests. Turning the mod off
//!   gives every GPU object back once.
//! * **On:** the pipelines are built on a worker thread (also on the title screen, so they are
//!   ready when a world loads). Vanilla's clouds keep drawing until they are, and on any frame
//!   the volumetric clouds do not draw: no world, the Nether or the End, the Clouds video option
//!   at Off (which hides both, as vanilla's own Off does), the eye in water or lava, Blindness,
//!   or anaglyph 3D (there vanilla's clouds are fogged or colour-converted the vanilla way).
//! * **Drawing:** vanilla's cloud phase is switched off for the frame (`drive_atmosphere` sets
//!   its mode from the video option every frame, so nothing needs restoring), and the light
//!   follows vanilla's time of day and weather (`in_game_look`), with the Time Changer included
//!   (it drives the same `Environment`).
//! * **Height**. "World height" puts the bases at vanilla's
//!   cloud height, as before. "Above you" (the default) keeps them "Distance above you" over the
//!   eye through [`BaseFollower`]: a slow follower, quick-ish to rise (a big climb), very slow to
//!   sink (falls), that snaps on a teleport (a new map, a respawn) and never lets the layer come
//!   within half the distance of the eye. Jumps and bridging move the bases by a fraction of a
//!   block, which nobody can see 60 blocks away. The noise moves with the layer
//!   (`CloudLayer::in_game_at`), so the same clouds simply sit higher or lower.
//!
//! Visual only: nothing here sends a packet.

use mc_math::DVec3;
use mc_mods::mods::volumetric_clouds::{DISTANCE_ABOVE, HEIGHT};
use mc_render::sky::volumetric::{self, CloudLayer, CloudQuality, VolumetricCloudsPhase};
use mc_render::sky::CloudMode;
use mc_render::{Camera, Gpu, PhaseList};

use super::super::Client;
use super::tw_replay_export::RenderClock;

/// The mod's id in the catalog and the settings file.
const MOD_ID: &str = "volumetric_clouds";
/// How fast the clouds drift at 100 % wind, in blocks per second (vanilla's flat clouds drift at
/// 0.6). Toward -X, like vanilla's.
const WIND_BLOCKS_PER_SECOND: f64 = 2.4;
/// Longest frame gap the wind integrates over (a hitch or a minimised window does not jump the
/// clouds).
const MAX_FRAME_GAP_S: f64 = 0.25;

/// "Above you": seconds for the bases to cover 63 % of a climb.
const RISE_SECONDS: f64 = 4.0;
/// "Above you": seconds for the bases to cover 63 % of a drop. Slow on purpose: a fall (or a
/// drop into the void) should not visibly pull the sky down.
const SINK_SECONDS: f64 = 20.0;
/// "Above you": a move of the eye this far within one frame is a teleport (a new map, a respawn):
/// the bases go straight to their place. The trace forgets its history at the same distance.
const TELEPORT_BLOCKS: f64 = 64.0;
/// "Above you": however fast the eye climbs, it stays at least this share of the distance under
/// the bases (and at least [`MIN_CLEARANCE`] blocks), so it can never reach the layer.
const CLEARANCE_SHARE: f64 = 0.5;
/// The smallest clearance between the eye and the bases in "Above you" mode, in blocks.
const MIN_CLEARANCE: f64 = 12.0;
/// A change of the bases this large between two frames (a teleport, a mode switch) also drops
/// the accumulated frames, which show the old layer.
const HISTORY_RESET_BLOCKS: f32 = 8.0;

/// The "Height" option.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum HeightMode {
    /// A set distance over the eye, eased ([`BaseFollower`]).
    #[default]
    AboveYou,
    /// Vanilla's cloud height.
    World,
}

/// "Above you": where the cloud bases are, following the eye slowly.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct BaseFollower {
    /// The eased bases' height, `None` until the first frame in a world.
    base: Option<f64>,
    /// The eye at the last frame (to tell a teleport from movement).
    last_eye: Option<DVec3>,
}

impl BaseFollower {
    /// One frame `dt` seconds after the last: ease toward `distance` above `eye` and return the
    /// bases' height for this frame.
    fn follow(&mut self, eye: DVec3, distance: f64, dt: f64) -> f64 {
        let target = eye.y + distance;
        let teleported = self
            .last_eye
            .is_some_and(|last| last.distance(eye) > TELEPORT_BLOCKS);
        self.last_eye = Some(eye);
        let eased = match self.base {
            Some(base) if !teleported => {
                let seconds = if target > base {
                    RISE_SECONDS
                } else {
                    SINK_SECONDS
                };
                base + (target - base) * (1.0 - (-dt.max(0.0) / seconds).exp())
            }
            _ => target,
        };
        // Never let the layer come down to the eye: a fast climb pushes it up at once.
        let clearance = (distance * CLEARANCE_SHARE)
            .max(MIN_CLEARANCE)
            .min(distance);
        let base = eased.max(eye.y + clearance);
        self.base = Some(base);
        base
    }
}

/// The client's side of the volumetric clouds (`Client::x11`).
#[derive(Debug, Default)]
pub(crate) struct X11State {
    /// The settings generation last read; 0 = never (the store starts at 1).
    seen: u64,
    /// The mod is on.
    enabled: bool,
    quality: CloudQuality,
    /// `0..=1`.
    coverage: f32,
    /// Wind multiplier, `0..=3`.
    wind: f32,
    /// The renderer behind the two slots `install` registered.
    phase: Option<std::rc::Rc<std::cell::RefCell<VolumetricCloudsPhase>>>,
    /// Something was requested since the mod was last off (so turning it off releases once).
    live: bool,
    /// How far the wind has carried the clouds, in blocks (x, z).
    wind_offset: [f64; 2],
    /// Seconds of slow drift through the noise (the clouds change shape as they move).
    drift: f64,
    /// The render clock at the last frame (seconds) and its epoch: the
    /// wind follows the client's render clock (the wall clock in normal play, the replay's time
    /// when one is watched or exported).
    last_seconds: Option<f64>,
    clock_epoch: Option<u32>,
    /// "Height".
    height: HeightMode,
    /// "Distance above you", in blocks.
    distance_above: f32,
    /// "Above you": the eased bases.
    follower: BaseFollower,
    /// The bases the layer had last frame (to drop the history on a big change).
    drawn_bottom: Option<f32>,
}

impl X11State {
    fn sync(&mut self, settings: &mc_mods::ModSettings) {
        let generation = settings.generation();
        if self.seen == generation {
            return;
        }
        self.seen = generation;
        self.enabled = settings.enabled(MOD_ID);
        self.quality = CloudQuality::from_index(settings.choice(MOD_ID, "quality"));
        self.coverage = settings.f32(MOD_ID, "coverage") / 100.0;
        self.wind = settings.f32(MOD_ID, "wind") / 100.0;
        self.height = match settings.choice(MOD_ID, HEIGHT) {
            1 => HeightMode::World,
            _ => HeightMode::AboveYou,
        };
        self.distance_above = settings.f32(MOD_ID, DISTANCE_ABOVE);
    }

    /// Advance the wind by the render-clock time since the last frame; returns that time
    /// (seconds, capped like the wind).
    ///
    /// In normal play the render clock is the wall clock, so this is what
    /// it always was. In a replay (and its video export) the offset and drift are functions of
    /// the replay's time alone, the same at any export frame rate and in every run, and a jump of
    /// the clock (a seek, an export pass) also restarts the "Above you" ease from its target.
    fn blow(&mut self, clock: RenderClock) -> f64 {
        if self.clock_epoch != Some(clock.epoch) {
            self.clock_epoch = Some(clock.epoch);
            self.last_seconds = None;
            if clock.replay {
                // A replay jumped (a seek, an export pass): start the eased bases, the history and
                // the jitter afresh, so the frames from here on depend on the replay alone.
                self.follower = BaseFollower::default();
                if let Some(phase) = &self.phase {
                    phase.borrow_mut().restart_sequence();
                }
            }
        }
        let dt = self.last_seconds.map_or(0.0, |last| {
            (clock.seconds - last).clamp(0.0, MAX_FRAME_GAP_S)
        });
        self.last_seconds = Some(clock.seconds);
        let wind = f64::from(self.wind);
        if clock.replay {
            self.wind_offset[0] = WIND_BLOCKS_PER_SECOND * wind * clock.seconds;
            self.drift = wind * clock.seconds;
        } else {
            self.wind_offset[0] += WIND_BLOCKS_PER_SECOND * wind * dt;
            self.drift += wind * dt;
        }
        dt
    }

    /// This frame's layer for an eye at `eye`, `dt` seconds after the last frame, with vanilla's
    /// cloud height `cloud_height`.
    fn layer(&mut self, eye: DVec3, cloud_height: f32, dt: f64) -> CloudLayer {
        match self.height {
            HeightMode::World => {
                self.follower = BaseFollower::default();
                CloudLayer::in_game(cloud_height)
            }
            HeightMode::AboveYou => {
                let base = self
                    .follower
                    .follow(eye, f64::from(self.distance_above), dt);
                CloudLayer::in_game_at(base as f32, cloud_height)
            }
        }
    }
}

/// Whether the volumetric clouds draw: in a world, in the overworld (vanilla's
/// `isSurfaceWorld()` gate for clouds), with the Clouds video option not at Off.
fn draws_in(dimension: Option<i32>, clouds_option: i32) -> bool {
    dimension == Some(0) && clouds_option != 0
}

/// Whether this frame's view hands the sky back to vanilla's clouds: the eye in water or lava, or
/// Blindness. Vanilla's clouds are drawn under the world fog there (so they all but vanish);
/// the volumetric clouds sit behind the screen-space fog's sky test and would shine through it.
/// Anaglyph 3D too: vanilla converts its cloud colours for it, the volumetric clouds do not.
/// (Vanilla's in-cloud fog band, `CameraMedium::Cloud`, is not a reason: the volumetric layer is
/// its own cloud.)
fn view_wants_vanilla(
    medium: mc_render::sky::CameraMedium,
    blindness: f32,
    anaglyph: bool,
) -> bool {
    use mc_render::sky::CameraMedium;
    matches!(medium, CameraMedium::Water | CameraMedium::Lava) || blindness > 0.0 || anaglyph
}

impl Client {
    /// Register the two (disabled) slots. Allocates nothing on the GPU. Also
    /// runs on a resource reload, which rebuilds every phase: the old renderer goes with them and
    /// a mod that is on builds a new one off-thread (vanilla's clouds meanwhile).
    pub(crate) fn install_volumetric_clouds(&mut self, phases: &mut PhaseList) {
        self.x11.phase = Some(VolumetricCloudsPhase::install(phases));
        self.x11.live = false;
    }

    /// Decide whether the volumetric clouds draw this frame and feed them.
    /// Runs after `drive_atmosphere`, so it can switch vanilla's clouds off for the frame. Off:
    /// one `u64` compare and two `bool` tests.
    pub(crate) fn volumetric_clouds_frame(
        &mut self,
        gpu: &Gpu,
        camera: &Camera,
        partial_ticks: f32,
    ) {
        self.x11.sync(&self.mods.settings);
        if !self.x11.enabled {
            if self.x11.live {
                self.x11.live = false;
                self.x11.last_seconds = None;
                if let Some(phase) = &self.x11.phase {
                    phase.borrow_mut().release();
                }
            }
            return;
        }
        self.x11.live = true;
        let clock = self.render_clock();
        let dt = self.x11.blow(clock);
        let dimension = self
            .session
            .as_ref()
            .map(|session| session.state().local.dimension);
        if dimension.is_none() {
            // No world: the next one starts its "Above you" bases afresh.
            self.x11.follower = BaseFollower::default();
            self.x11.drawn_bottom = None;
        }
        let (true, Some(render)) = (self.x11.phase.is_some(), &self.render) else {
            return;
        };
        // `drive_atmosphere` has filled the fog's inputs for this frame.
        let (medium, blindness) = render
            .fog
            .with(|fog| (fog.inputs.medium, fog.inputs.blindness));
        let draw = draws_in(dimension, self.settings.clouds)
            && !view_wants_vanilla(medium, blindness, self.settings.anaglyph);
        let sky = draw.then(|| {
            let (environment, cloud_height) = render
                .clouds
                .with(|clouds| (clouds.environment, clouds.cloud_height));
            // "Above you" (the default) or vanilla's "World height".
            (environment, self.x11.layer(camera.pos, cloud_height, dt))
        });
        let Some(phase) = &self.x11.phase else {
            return;
        };
        let mut phase = phase.borrow_mut();
        phase.request(gpu);
        let Some((environment, layer)) = sky else {
            phase.set_frame(false, camera.pos.y);
            return;
        };
        let biome_temperature = render.sky.with(|sky| sky.biome_temperature);
        let mut look = volumetric::in_game_look(
            &environment,
            biome_temperature,
            partial_ticks,
            self.x11.coverage,
        );
        look.wind = self.x11.wind_offset;
        look.drift = self.x11.drift;
        phase.look = look;
        // A teleport or a mode switch moves the bases at once: the accumulated frames show the
        // old layer.
        if self
            .x11
            .drawn_bottom
            .is_some_and(|bottom| (bottom - layer.bottom).abs() > HISTORY_RESET_BLOCKS)
        {
            phase.reset_history();
        }
        self.x11.drawn_bottom = Some(layer.bottom);
        phase.layer = layer;
        phase.quality = self.x11.quality;
        if phase.set_frame(true, camera.pos.y) {
            // Ours draw this frame: vanilla's step aside. `drive_atmosphere` sets their mode
            // from the video option again next frame.
            render.clouds.with(|clouds| clouds.mode = CloudMode::Off);
        }
    }

    /// Whether the volumetric clouds draw this frame (tests, F3).
    #[cfg(test)]
    pub(crate) fn volumetric_clouds_drawing(&self) -> bool {
        self.x11
            .phase
            .as_ref()
            .is_some_and(|phase| phase.borrow().drawing())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::app::config::{ClientConfig, LoginMode};

    #[test]
    fn they_draw_only_in_the_overworld_with_clouds_on() {
        assert!(draws_in(Some(0), 1));
        assert!(draws_in(Some(0), 2));
        assert!(
            !draws_in(Some(0), 0),
            "the Clouds video option at Off hides them too"
        );
        assert!(!draws_in(Some(-1), 2), "no clouds in the Nether");
        assert!(!draws_in(Some(1), 2), "no clouds in the End");
        assert!(!draws_in(None, 2), "no world");
    }

    #[test]
    fn water_lava_blindness_and_anaglyph_hand_the_sky_back_to_vanilla() {
        use mc_render::sky::CameraMedium;
        assert!(!view_wants_vanilla(CameraMedium::Air, 0.0, false));
        assert!(
            !view_wants_vanilla(CameraMedium::Cloud, 0.0, false),
            "vanilla's in-cloud fog band is not a reason"
        );
        assert!(view_wants_vanilla(CameraMedium::Water, 0.0, false));
        assert!(view_wants_vanilla(CameraMedium::Lava, 0.0, false));
        assert!(
            view_wants_vanilla(CameraMedium::Air, 0.4, false),
            "blindness"
        );
        assert!(view_wants_vanilla(CameraMedium::Air, 0.0, true), "anaglyph");
    }

    /// The render clock of normal play (the wall clock) at `seconds`.
    fn wall(seconds: f64) -> RenderClock {
        RenderClock {
            seconds,
            epoch: 0,
            replay: false,
        }
    }

    #[test]
    fn the_wind_follows_the_wall_clock_and_its_option() {
        let mut state = X11State {
            wind: 1.0,
            ..X11State::default()
        };
        let start = 1234.5;
        state.blow(wall(start));
        assert_eq!(
            state.wind_offset, [0.0; 2],
            "the first frame only starts the clock"
        );
        state.blow(wall(start + 0.1));
        assert!((state.wind_offset[0] - WIND_BLOCKS_PER_SECOND * 0.1).abs() < 1e-9);
        // A long hitch moves the clouds by at most a quarter second's worth.
        state.blow(wall(start + 10.0));
        assert!(state.wind_offset[0] < WIND_BLOCKS_PER_SECOND * 0.36);
        // No wind: they stand still.
        state.wind = 0.0;
        let before = (state.wind_offset, state.drift);
        state.blow(wall(start + 11.0));
        assert_eq!((state.wind_offset, state.drift), before);
    }

    /// In a replay or its export the clouds' wind is a function of the
    /// replay's time alone: the same at 60 and at 1000 fps, whatever came before.
    #[test]
    fn in_a_replay_the_wind_is_a_function_of_replay_time() {
        let replay = |seconds, epoch| RenderClock {
            seconds,
            epoch,
            replay: true,
        };
        let at = |steps: u32, history: f64| {
            let mut state = X11State {
                wind: 1.5,
                ..X11State::default()
            };
            // Whatever the clouds did before (normal play, other replay moments)...
            state.blow(wall(history));
            state.blow(wall(history + 0.2));
            // ...an export from 3.0 s to 4.0 s in `steps` frames ends at the same place.
            for i in 0..=steps {
                state.blow(replay(3.0 + f64::from(i) / f64::from(steps), 1));
            }
            (state.wind_offset, state.drift)
        };
        let (wind60, drift60) = at(60, 10.0);
        let (wind1000, drift1000) = at(1000, 9876.0);
        assert_eq!(wind60, wind1000);
        assert_eq!(drift60, drift1000);
        assert!((wind60[0] - WIND_BLOCKS_PER_SECOND * 1.5 * 4.0).abs() < 1e-9);
    }

    // ---- "Height" ------------------------------------

    const FRAME: f64 = 1.0 / 60.0;

    /// One vanilla jump from `ground` at 60 fps (3 frames per tick, interpolated):
    /// `EntityLivingBase.jump` 0.42 up, gravity 0.08, drag 0.98.
    fn jump_from(ground: f64) -> Vec<f64> {
        let (mut y, mut vy) = (0.0f64, 0.42f64);
        let mut ticks = vec![0.0];
        while y + vy > 0.0 {
            y += vy;
            vy = (vy - 0.08) * 0.98;
            ticks.push(y);
        }
        ticks.push(0.0);
        let mut out = Vec::new();
        for pair in ticks.windows(2) {
            for f in 0..3 {
                out.push(ground + pair[0] + (pair[1] - pair[0]) * f64::from(f) / 3.0);
            }
        }
        out
    }

    fn follow(follower: &mut BaseFollower, eyes: impl IntoIterator<Item = f64>) -> Vec<f64> {
        eyes.into_iter()
            .map(|y| follower.follow(DVec3::new(0.5, y, 0.5), 60.0, FRAME))
            .collect()
    }

    /// "Above you": jumps, bunny hopping, bridging and falls barely move the bases; a big climb
    /// and a teleport do; the eye never gets within the clearance, however fast it rises.
    #[test]
    fn above_you_follows_climbs_and_new_maps_but_not_jumps_or_falls() {
        let mut follower = BaseFollower::default();
        assert_eq!(
            follow(&mut follower, [70.0]),
            [130.0],
            "the first frame: in place"
        );
        let jump = jump_from(70.0);
        assert!(jump.iter().any(|y| *y > 71.2), "a real jump arc");
        let moved = |bases: &[f64], from: f64| {
            bases
                .iter()
                .map(|b| (b - from).abs())
                .fold(0.0f64, f64::max)
        };
        assert!(
            moved(&follow(&mut follower, jump.clone()), 130.0) < 0.15,
            "one jump"
        );
        // Bunny hopping: the follower rises faster than it sinks, so it settles toward the top
        // of the arc, still under a block (60 blocks away, about half a degree).
        let hops: Vec<f64> = (0..10).flat_map(|_| jump.clone()).collect();
        assert!(
            moved(&follow(&mut follower, hops), 130.0) < 1.0,
            "ten jumps in a row"
        );
        // Bridging: standing at one height for 10 s; the bases ease back down, slowly.
        let settled = follow(&mut follower, std::iter::repeat_n(70.0, 600));
        assert!(moved(&settled, 130.0) < 1.0);
        assert!(settled.last().unwrap() < settled.first().unwrap());

        // A fall of 20 blocks in two seconds, then a second standing: about 2 blocks.
        let mut follower = BaseFollower::default();
        follow(&mut follower, [90.0]);
        let fall = (0..120).map(|i| 90.0 - 20.0 * f64::from(i) / 119.0);
        let bases = follow(&mut follower, fall.chain(std::iter::repeat_n(70.0, 60)));
        assert!(moved(&bases, 150.0) < 2.5, "a fall: {:?}", bases.last());

        // A climb of 40 blocks over 20 s: the bases follow it most of the way.
        let mut follower = BaseFollower::default();
        follow(&mut follower, [70.0]);
        let climb = (0..1200).map(|i| 70.0 + 40.0 * f64::from(i) / 1199.0);
        let bases = follow(&mut follower, climb);
        let last = *bases.last().unwrap();
        assert!(
            (160.0..=170.0).contains(&last),
            "a big climb moves the clouds: {last}"
        );

        // A fast climb (20 blocks a second for 3 s): never closer than 30 blocks (half of 60).
        let mut follower = BaseFollower::default();
        follow(&mut follower, [70.0]);
        let rocket: Vec<f64> = (0..180).map(|i| 70.0 + f64::from(i) / 3.0).collect();
        let bases = follow(&mut follower, rocket.clone());
        for (eye, base) in rocket.iter().zip(&bases) {
            assert!(base - eye >= 30.0 - 1e-9, "eye {eye}, bases {base}");
        }

        // A teleport to a new map: straight to 60 above it.
        let mut follower = BaseFollower::default();
        follow(&mut follower, [70.0]);
        let bases: Vec<f64> = [(0.5, 70.0, 0.5), (900.0, 101.0, -40.0)]
            .into_iter()
            .map(|(x, y, z)| follower.follow(DVec3::new(x, y, z), 60.0, FRAME))
            .collect();
        assert_eq!(bases, [130.0, 161.0]);
    }

    /// "World height" is exactly the layer the mod drew before this change (vanilla's height, the
    /// noise unshifted), whatever the eye does; "Above you" is the same layer moved up, noise and
    /// all.
    #[test]
    fn world_height_is_the_old_layer_and_above_you_moves_it_whole() {
        let mut state = X11State {
            height: HeightMode::World,
            distance_above: 60.0,
            ..X11State::default()
        };
        for y in [64.0, 127.9, 300.0] {
            assert_eq!(
                state.layer(DVec3::new(0.0, y, 0.0), 128.0, FRAME),
                CloudLayer::in_game(128.0)
            );
        }
        state.height = HeightMode::AboveYou;
        let layer = state.layer(DVec3::new(0.0, 72.0, 0.0), 128.0, FRAME);
        assert_eq!(layer.bottom, 132.0);
        assert_eq!(layer.top - layer.bottom, 120.0);
        assert_eq!(layer.noise_shift, 4.0, "the noise moves with the layer");
        assert_eq!(
            CloudLayer {
                bottom: 128.0,
                top: 248.0,
                noise_shift: 0.0,
                ..layer
            },
            CloudLayer::in_game(128.0),
            "nothing else changes"
        );
    }

    fn headless_client() -> Option<(Client, tempfile::TempDir)> {
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

    fn headless_gpu() -> Option<Gpu> {
        match Gpu::headless(
            64,
            64,
            mc_render::GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        ) {
            Ok(gpu) => Some(gpu),
            Err(mc_render::RenderError::NoAdapter) => {
                eprintln!("SKIP: no wgpu adapter on this machine");
                None
            }
            Err(error) => panic!("headless GPU: {error}"),
        }
    }

    /// Off by default and allocates nothing; on, it builds (also without a world, so it is ready
    /// for one) but never draws or hides vanilla's clouds without a world; off again, it gives
    /// everything back.
    #[test]
    fn off_allocates_nothing_and_on_builds_without_touching_vanillas_clouds() {
        let Some((mut client, _dir)) = headless_client() else {
            return;
        };
        let Some(gpu) = headless_gpu() else { return };
        let mut phases = mc_render::PhaseList::new();
        client.install(&gpu, &mut phases, 64, 64);
        let names: Vec<_> = phases.names().collect();
        assert!(names.contains(&volumetric::EARLY_PHASE_NAME));
        assert!(names.contains(&volumetric::LATE_PHASE_NAME));
        let camera = Camera::default();
        let allocated = |client: &Client| client.x11.phase.as_ref().unwrap().borrow().allocated();
        let vanilla_mode = |client: &Client| {
            client
                .render
                .as_ref()
                .unwrap()
                .clouds
                .with(|clouds| clouds.mode)
        };

        // Off (the default).
        assert!(!client.mods.settings.enabled(MOD_ID));
        let before = vanilla_mode(&client);
        for _ in 0..3 {
            client.volumetric_clouds_frame(&gpu, &camera, 0.0);
        }
        assert!(!allocated(&client), "nothing allocated while off");
        assert!(!client.volumetric_clouds_drawing());
        assert_eq!(vanilla_mode(&client), before, "vanilla's clouds untouched");

        // On, without a world: the pipelines build, nothing draws, vanilla's clouds stay.
        client.mods.settings.set_enabled(MOD_ID, true);
        client.volumetric_clouds_frame(&gpu, &camera, 0.0);
        assert!(allocated(&client), "on: the pipelines are being built");
        assert!(client
            .x11
            .phase
            .as_ref()
            .unwrap()
            .borrow_mut()
            .wait_ready(Duration::from_secs(60)));
        client.volumetric_clouds_frame(&gpu, &camera, 0.0);
        assert!(!client.volumetric_clouds_drawing(), "no world, no clouds");
        assert_eq!(vanilla_mode(&client), before);

        // Off again: everything given back.
        client.mods.settings.set_enabled(MOD_ID, false);
        client.volumetric_clouds_frame(&gpu, &camera, 0.0);
        assert!(!allocated(&client), "turning the mod off frees everything");
    }

    /// A session with no socket behind it that has joined the overworld (`JoinGame`).
    fn joined_overworld() -> (crate::session::Session, impl Sized) {
        let (handle, to_game, to_net) = crate::net::channel::NetHandle::new_pair();
        let mut session =
            crate::session::Session::new(handle, crate::session::JoinSettings::default());
        let join = mc_protocol::play::s2c::JoinGame {
            entity_id: 3,
            hardcore_mode: false,
            game_mode: 0,
            dimension: 0,
            difficulty: 2,
            max_players: 20,
            level_type: "default".into(),
            reduced_debug_info: false,
        };
        to_game
            .try_send(crate::net::ToGame::Packet(
                mc_protocol::play::S2CPlay::JoinGame(join),
            ))
            .expect("join channel");
        assert!(session
            .poll_all()
            .iter()
            .any(|event| matches!(event, crate::sync::ClientEvent::JoinedGame(_))));
        (session, (to_game, to_net))
    }

    /// In a world, through the real `Client::frame` (`drive_atmosphere`, then the hook): with
    /// the mod on, vanilla's clouds are off for the frame and ours draw from the early slot (the
    /// eye is below the cloud bases), and the installed phase list renders without a validation
    /// error. The Clouds video option at Off hides both; the mod off gives vanilla's back at once
    /// and frees everything.
    #[test]
    fn in_a_world_ours_replace_vanillas_and_off_gives_them_back() {
        let Some((mut client, _dir)) = headless_client() else {
            return;
        };
        let Some(mut gpu) = headless_gpu() else {
            return;
        };
        gpu.device.on_uncaptured_error(std::sync::Arc::new(|e| {
            panic!("wgpu validation error: {e}")
        }));
        let mut phases = mc_render::PhaseList::new();
        client.install(&gpu, &mut phases, 64, 64);
        let (session, _channels) = joined_overworld();
        client.session = Some(session);
        client.ui = super::super::super::Ui::InGame;
        client.settings.clouds = 2;
        client.mods.settings.set_enabled(MOD_ID, true);
        let vanilla_mode = |client: &Client| {
            client
                .render
                .as_ref()
                .unwrap()
                .clouds
                .with(|clouds| clouds.mode)
        };
        let mut camera = Camera::default();
        client.frame(&gpu, &mut camera, 0.0);
        assert!(client
            .x11
            .phase
            .as_ref()
            .unwrap()
            .borrow_mut()
            .wait_ready(Duration::from_secs(60)));
        client.frame(&gpu, &mut camera, 0.0);
        assert!(
            client.volumetric_clouds_drawing(),
            "ours draw in the overworld"
        );
        assert_eq!(
            vanilla_mode(&client),
            CloudMode::Off,
            "vanilla's stepped aside"
        );
        assert_eq!(
            client.x11.phase.as_ref().unwrap().borrow().drawing_order(),
            Some(volumetric::EARLY_ORDER)
        );
        for _ in 0..3 {
            let target = gpu.acquire().expect("offscreen frame");
            phases.render(&gpu, target.view(), &camera, 0.0);
        }
        let _ = gpu.read_pixels().expect("readback");

        // The Clouds video option at Off: neither kind draws.
        client.settings.clouds = 0;
        client.frame(&gpu, &mut camera, 0.0);
        assert!(!client.volumetric_clouds_drawing());
        assert_eq!(vanilla_mode(&client), CloudMode::Off);

        // The mod off: vanilla's fancy clouds again, and nothing of ours left.
        client.settings.clouds = 2;
        client.mods.settings.set_enabled(MOD_ID, false);
        client.frame(&gpu, &mut camera, 0.0);
        assert!(!client.volumetric_clouds_drawing());
        assert_eq!(vanilla_mode(&client), CloudMode::Fancy);
        assert!(!client.x11.phase.as_ref().unwrap().borrow().allocated());

        // H7: java `GameSettings.shouldRenderClouds()` is `renderDistanceChunks >= 4 ? clouds :
        // 0`, so 2 and 3 chunks draw no vanilla clouds; 4 draws them again.
        let render_distance = client.settings.render_distance_chunks;
        client.settings.render_distance_chunks = 3;
        client.frame(&gpu, &mut camera, 0.0);
        assert_eq!(vanilla_mode(&client), CloudMode::Off);
        client.settings.render_distance_chunks = 4;
        client.frame(&gpu, &mut camera, 0.0);
        assert_eq!(vanilla_mode(&client), CloudMode::Fancy);
        client.settings.render_distance_chunks = render_distance;
    }
}
