//! The Underwater shader pack's client half (`shaders_underwater`).
//!
//! The effect is `mc_render::shaderpack::underwater`: a world phase at
//! [`WORLD_ORDER`] (shafts + composite) and the marine snow at [`SNOW_ORDER`]. This file moves
//! the mod's settings into them and decides, every frame, whether the pack draws:
//!
//! * the mod is on, a world is on screen, no MSAA;
//! * Everywhere mode: the Overworld (no water in the Nether, no sky in the End); Only under water:
//!   the camera's eye is in water;
//! * never while blind (vanilla's blindness fog must keep hiding things, and the pack's bright
//!   surface would show silhouettes against it) and never with the eye in lava.
//!
//! Three driver hooks, each one line:
//! * [`Client::install_tw_underwater`] from `Client::install` (also after a resource reload);
//! * [`Client::tw_underwater_fog_inputs`] inside `drive_atmosphere`, right after the Fog mod's
//!   hook: when the pack draws it owns the distance fade, so vanilla's world fog is switched off
//!   (`distance_fog = false`, and `clear_liquids` under water, the pack's own absorption keeping
//!   at least vanilla's visibility). Blindness fog is untouched. The decision is made here, on the
//!   frame's real medium, so a dive never shows one frame of double fog;
//! * [`Client::tw_underwater_frame`] among the per-frame mod hooks, after `drive_atmosphere`: the
//!   settings (on a generation change), the frame inputs, the translucent terrain's per-draw fog
//!   (water colour) and, under water, vanilla's overlay texture (replaced unless the option keeps
//!   it).
//!
//! With the mod off the hooks cost one `bool` test each and the phases are disabled (no GPU
//! objects).

use std::cell::Cell;

use mc_data::Material;
use mc_math::{BlockPos, ChunkPos, DVec3, Vec3};
use mc_mods::mods::shaders_underwater as desc;
use mc_render::post::overlay::OverlayKind;
use mc_render::shaderpack::underwater::{
    absorption, rgb_from_argb, water_glow, UnderwaterFrame, UnderwaterMode, UnderwaterPhase,
    UnderwaterQuality, UnderwaterSettings, UnderwaterSnowPhase, OCCLUSION_SIZE, SNOW_ORDER,
    WORLD_ORDER,
};
use mc_render::shaderpack::{CoreConfig, PackId};
use mc_render::sky::fog::{
    world_pass_fog_mode, CameraMedium, FogDistance, FogInputs, FogMode, FogState,
};
use mc_render::{Camera, Gpu, PhaseList};

use super::super::{Client, Shared, Ui};

/// How far above the eye the virtual sea surface sits in Everywhere mode, in blocks.
pub(crate) const VIRTUAL_DEPTH: f64 = 24.0;
/// The virtual surface never comes closer than this above the eye.
const MIN_VIRTUAL_DEPTH: f64 = 8.0;
/// How quickly the virtual surface follows the eye (seconds to close ~63% of the gap): jumps,
/// bridging and falls never move it visibly; a new map or a long climb does, slowly.
const SURFACE_EASE_SECONDS: f64 = 10.0;
/// How far up a real water column is scanned for its surface, in blocks.
const SURFACE_SCAN: i32 = 48;
/// Rows of the light-occlusion map refreshed per game tick (two chunk rows): the whole map is
/// refreshed every 6 ticks, so a placed or broken block shows in its shadow within a third of a
/// second, for about 24 chunk lookups a tick.
const OCCLUSION_ROWS_PER_TICK: i32 = 32;

/// One above the highest block in a chunk column that keeps sunlight out: vanilla's sky-light
/// height map, then down through water, ice and glass, which let the light through.
fn column_top(chunk: &mc_world::Chunk, x: u8, z: u8) -> u8 {
    let mut h = chunk.height_value(x, z).clamp(0, 256);
    while h > 0 {
        let material = chunk.block_at(x, (h - 1) as u8, z).props().material;
        match material {
            Material::Water
            | Material::Ice
            | Material::PackedIce
            | Material::Glass
            | Material::Air => h -= 1,
            _ => break,
        }
    }
    h.min(255) as u8
}

/// The pack's client state.
#[derive(Default)]
pub(crate) struct TwUnderwaterState {
    /// The settings generation last read (0 = never; the store starts at 1).
    seen: u64,
    /// The mod is switched on.
    on: bool,
    /// The settings in effect.
    settings: UnderwaterSettings,
    /// Keep vanilla's water overlay texture under water.
    vanilla_overlay: bool,
    /// The `preset` choice last seen, to tell a preset pick from a slider change.
    preset_seen: Option<usize>,
    /// MSAA is on (from the GPU at install): the pack never draws then.
    msaa: bool,
    /// The two phases, shared with the `PhaseList`. `None` until `Client::install`.
    world: Option<Shared<UnderwaterPhase>>,
    snow: Option<Shared<UnderwaterSnowPhase>>,
    /// This frame's decision, made by the fog hook (which only has `&self`).
    drawing: Cell<bool>,
    /// The eye is in water this frame (fog hook).
    eye_in_water: Cell<bool>,
    /// Vanilla's exponential fog density for this eye in water, before the pack cleared it.
    vanilla_density: Cell<f32>,
    /// What the phases were last told (only changes are pushed).
    pushed_active: bool,
    /// The eased virtual surface, world y.
    surface_y: Option<f64>,
    /// Wave time in seconds (speed-scaled, integrated so a speed change never jumps).
    wave_clock: f64,
    /// Game time of the last frame, seconds.
    last_game_seconds: Option<f64>,
    /// The light-occlusion map's corner (world x, z), chunk aligned.
    occlusion_origin: Option<(i32, i32)>,
    /// The next map row to refresh.
    occlusion_row: i32,
    /// The game tick the map was last refreshed on.
    occlusion_tick: u64,
}

impl TwUnderwaterState {
    /// Whether the pack draws this frame (for tests).
    #[cfg(test)]
    pub(crate) fn drawing(&self) -> bool {
        self.drawing.get()
    }
}

/// The pack's settings as the phases want them.
fn read_settings(s: &mc_mods::ModSettings) -> UnderwaterSettings {
    let id = desc::ID;
    UnderwaterSettings {
        mode: if s.choice(id, "mode") == desc::MODE_UNDER_WATER {
            UnderwaterMode::OnlyUnderWater
        } else {
            UnderwaterMode::Everywhere
        },
        water_color: rgb_from_argb(s.color(id, "water_color")),
        clarity: s.f32(id, "clarity"),
        clear_range: s.f32(id, "clear_range"),
        red_fade: s.f32(id, "red_fade") / 100.0,
        depth_darkening: s.f32(id, "depth_darkening") / 100.0,
        surface_sky: s.bool(id, "surface_sky"),
        caustics: s.bool(id, "caustics"),
        caustic_strength: s.f32(id, "caustic_strength") / 100.0,
        caustic_size: s.f32(id, "caustic_size"),
        wave_speed: s.f32(id, "wave_speed") / 100.0,
        light_shafts: s.bool(id, "light_shafts"),
        shaft_strength: s.f32(id, "shaft_strength") / 100.0,
        marine_snow: s.bool(id, "marine_snow"),
        snow_amount: s.f32(id, "snow_amount") / 100.0,
        refraction: s.bool(id, "refraction"),
        refraction_strength: s.f32(id, "refraction_strength") / 100.0,
        underwater_clarity: s.f32(id, "underwater_clarity"),
        quality: UnderwaterQuality::from_index(s.choice(id, "quality")),
    }
}

/// Keep the `preset` choice and its four settings in step: picking a preset writes its values;
/// changing one of the values (or loading a file whose values match no preset) shows `Custom`.
/// Returns whether anything was written.
fn sync_preset(s: &mut mc_mods::ModSettings, seen: Option<usize>) -> bool {
    let id = desc::ID;
    let preset = s.choice(id, "preset");
    let matching = desc::matching_preset(
        s.color(id, "water_color"),
        s.f32(id, "clarity"),
        s.f32(id, "red_fade"),
        s.f32(id, "depth_darkening"),
    );
    let picked = seen.is_some_and(|previous| previous != preset);
    if picked && preset < desc::PRESETS.len() {
        if matching == Some(preset) {
            return false;
        }
        let p = desc::PRESETS[preset];
        s.set_color(id, "water_color", p.water_color);
        s.set_f32(id, "clarity", p.clarity);
        s.set_f32(id, "red_fade", p.red_fade);
        s.set_f32(id, "depth_darkening", p.depth_darkening);
        return true;
    }
    if preset != desc::PRESET_CUSTOM && matching != Some(preset) {
        return s.set_choice(id, "preset", matching.unwrap_or(desc::PRESET_CUSTOM));
    }
    false
}

/// Move the occlusion map's contents for a corner that moved `dx`, `dz` blocks (both under the
/// map's size): the new cell `(row, col)` takes the old cell `(row + dz, col + dx)`. Cells with no
/// old source keep stale values; the caller reads those strips next.
fn shift_map(map: &mut [u8], dx: i32, dz: i32) {
    let size = OCCLUSION_SIZE as i32;
    let n = OCCLUSION_SIZE as usize;
    let lo = dx.max(0) as usize;
    let hi = (size + dx.min(0)) as usize;
    let dest = (-dx).max(0) as usize;
    let mut shift_row = |row: i32| {
        let src = row + dz;
        if (0..size).contains(&src) {
            let (from, to) = (src as usize * n, row as usize * n);
            map.copy_within(from + lo..from + hi, to + dest);
        }
    };
    // Rows move toward the side they come from first, so no source is overwritten before use.
    if dz >= 0 {
        (0..size).for_each(&mut shift_row);
    } else {
        (0..size).rev().for_each(&mut shift_row);
    }
}

/// Linear `0..1` to display-encoded, the 2.2 curve the pack uses everywhere.
fn display(c: f32) -> f32 {
    c.max(0.0).powf(1.0 / 2.2)
}

impl Client {
    /// The mod settings, for tools and tests that drive a client directly (the shader packs'
    /// picture harness, `tests/tw_underwater_pictures.rs`). A change takes effect on the next
    /// frame, exactly like a change made in the mod menu.
    pub fn mod_settings_mut(&mut self) -> &mut mc_mods::ModSettings {
        &mut self.mods.settings
    }

    /// Register the two phases. Runs from `Client::install`, so also after a resource reload
    /// (which rebuilds every phase); the next frame pushes the settings again.
    pub(crate) fn install_tw_underwater(&mut self, gpu: &Gpu, phases: &mut PhaseList) {
        let samples = gpu.sample_count();
        let world = Shared::new(UnderwaterPhase::new(samples));
        let snow = Shared::new(UnderwaterSnowPhase::new(samples));
        phases.insert(WORLD_ORDER, Box::new(world.clone()));
        phases.insert(SNOW_ORDER, Box::new(snow.clone()));
        self.tw_underwater = TwUnderwaterState {
            msaa: samples > 1,
            world: Some(world),
            snow: Some(snow),
            ..TwUnderwaterState::default()
        };
    }

    /// Whether a world is on screen: a session whose player exists and is past the
    /// downloading-terrain screen.
    fn tw_underwater_in_world(&self) -> bool {
        !matches!(self.ui, Ui::Loading)
            && self.session.as_ref().is_some_and(|session| {
                session.predictor().is_some() && !session.terrain_download_active()
            })
    }

    /// The fog hook (`drive_atmosphere`, after the Fog mod's): decide whether the pack draws this
    /// frame and, if it does, hand the distance fade to it. One `bool` test while the mod is off.
    pub(crate) fn tw_underwater_fog_inputs(&self, inputs: &mut FogInputs) {
        let state = &self.tw_underwater;
        if !state.on {
            state.drawing.set(false);
            return;
        }
        let in_water = inputs.medium == CameraMedium::Water;
        let blind = inputs.blindness > 0.0 || inputs.blindness_duration_ticks.is_some();
        let overworld = !inputs.dimension_always_fogs;
        let mode_ok = match state.settings.mode {
            UnderwaterMode::Everywhere => overworld || in_water,
            UnderwaterMode::OnlyUnderWater => in_water,
        };
        let drawing = mode_ok
            && !state.msaa
            && !blind
            && inputs.medium != CameraMedium::Lava
            && self.tw_underwater_in_world();
        state.drawing.set(drawing);
        state.eye_in_water.set(in_water);
        if !drawing {
            return;
        }
        // Vanilla's own falloff for this eye, before the pack changes it: under water it is the
        // visibility floor (0 when the Fog mod's clear water already made it clear).
        let density = match world_pass_fog_mode(*inputs) {
            FogMode::Exp { density } if in_water => density,
            _ => 0.0,
        };
        state.vanilla_density.set(density);
        inputs.distance_fog = false;
        inputs.clear_liquids = true;
    }

    /// The per-frame hook: settings, inputs, translucent fog and the overlay. Runs after
    /// `drive_atmosphere`. One `u64` compare and one `bool` test while the mod is off.
    pub(crate) fn tw_underwater_frame(&mut self, camera: &Camera, partial_ticks: f32) {
        let generation = self.mods.settings.generation();
        if self.tw_underwater.seen != generation {
            self.tw_underwater_sync();
        }
        if !self.tw_underwater.on {
            return;
        }
        // The core keeps the last pack's features when another pack takes over: hand it
        // this pack's (none: the Underwater pack is its own passes), so Classic's shadows or tone
        // map never run under it. A compare when nothing changed.
        let core = self.tw_shaders.core();
        if core.active() == Some(PackId::Underwater) {
            core.set_config(CoreConfig::OFF);
        }
        let drawing = self.tw_underwater.drawing.get() && self.tw_underwater_in_world();
        if drawing != self.tw_underwater.pushed_active {
            self.tw_underwater.pushed_active = drawing;
            if let Some(world) = &self.tw_underwater.world {
                world.with(|phase| phase.set_active(drawing));
            }
            if let Some(snow) = &self.tw_underwater.snow {
                snow.with(|phase| phase.set_active(drawing));
            }
        }
        if !drawing {
            self.tw_underwater.surface_y = None;
            self.tw_underwater.last_game_seconds = None;
            return;
        }
        let mut frame = self.tw_underwater_inputs(camera, partial_ticks);
        frame.occlusion_origin = self.tw_underwater_occlusion(frame.eye);
        let state = &self.tw_underwater;
        if let Some(world) = &state.world {
            world.with(|phase| phase.set_frame(frame));
        }
        if let Some(snow) = &state.snow {
            snow.with(|phase| phase.set_frame(frame));
        }
        let Some(render) = &self.render else {
            return;
        };
        // Water surfaces, stained glass and ice fog themselves per draw (after the pack's
        // composite): give them the pack's water colour and an absorption close to its green
        // channel, so a far lake fades like the ground around it.
        let sigma = absorption(&state.settings, &frame);
        let glow = water_glow(&state.settings, &frame);
        let fog = FogState {
            color: Vec3::new(display(glow.x), display(glow.y), display(glow.z)),
            mode: FogMode::Exp { density: sigma.y },
        };
        render
            .terrain
            .borrow_mut()
            .set_translucent_fog(Some((fog, FogDistance::Radial)));
        // Under water the pack's own look replaces vanilla's overlay texture unless kept.
        if frame.eye_in_water && !state.vanilla_overlay {
            render.overlay.with(|overlay| {
                if overlay.kind == OverlayKind::Water {
                    overlay.kind = OverlayKind::None;
                }
            });
        }
    }

    /// Read the settings (the generation moved): presets, the switch, the phases.
    fn tw_underwater_sync(&mut self) {
        let seen = self.tw_underwater.preset_seen;
        sync_preset(&mut self.mods.settings, seen);
        let s = &self.mods.settings;
        self.tw_underwater.preset_seen = Some(s.choice(desc::ID, "preset"));
        self.tw_underwater.seen = s.generation();
        let on = s.enabled(desc::ID);
        let settings = read_settings(s).sanitized();
        let state = &mut self.tw_underwater;
        state.on = on;
        state.settings = settings;
        state.vanilla_overlay = s.bool(desc::ID, "vanilla_overlay");
        if !on {
            state.drawing.set(false);
            state.pushed_active = false;
            // The phase frees its occlusion map when switched off: read all of it next time.
            state.occlusion_origin = None;
        }
        if let Some(world) = &state.world {
            world.with(|phase| {
                phase.set_settings(settings);
                phase.set_enabled(on);
            });
        }
        if let Some(snow) = &state.snow {
            snow.with(|phase| {
                phase.set_settings(settings);
                phase.set_enabled(on);
            });
        }
    }

    /// This frame's inputs for the phases.
    fn tw_underwater_inputs(&mut self, camera: &Camera, partial_ticks: f32) -> UnderwaterFrame {
        let eye = camera.pos;
        let in_water = self.tw_underwater.eye_in_water.get();
        // Wave time: the game clock (ticks + partial tick), so a replay export at a fixed
        // timestep animates exactly; a jump in the clock (a new session) never jumps the waves.
        let game_seconds = (self.tick as f64 + f64::from(partial_ticks)) / 20.0;
        let delta = self
            .tw_underwater
            .last_game_seconds
            .map_or(0.0, |last| (game_seconds - last).clamp(0.0, 0.25));
        self.tw_underwater.last_game_seconds = Some(game_seconds);
        self.tw_underwater.wave_clock += delta * f64::from(self.tw_underwater.settings.wave_speed);
        if self.tw_underwater.wave_clock > 1.0e6 {
            // Every loop the pack uses divides 16 s; keep the f64 small and exact.
            self.tw_underwater.wave_clock %= 16.0 * 36.0 * 25.0;
        }
        let surface_height = if in_water {
            self.tw_underwater.surface_y = None;
            self.real_water_surface(eye)
        } else {
            let target = eye.y + VIRTUAL_DEPTH;
            let ease = 1.0 - (-delta / SURFACE_EASE_SECONDS).exp();
            let current = self.tw_underwater.surface_y.unwrap_or(target);
            let eased = (current + (target - current) * ease).max(eye.y + MIN_VIRTUAL_DEPTH);
            self.tw_underwater.surface_y = Some(eased);
            (eased - eye.y) as f32
        };
        let (environment, inputs) = self
            .render
            .as_ref()
            .map(|render| render.fog.with(|phase| (phase.environment, phase.inputs)))
            .unwrap_or_default();
        // The air's fog colour (the sky at the horizon), whatever the eye is in.
        let air = FogState::compute(
            &environment,
            FogInputs {
                medium: CameraMedium::Air,
                ..inputs
            },
            partial_ticks,
        );
        let sky = environment.sky_color(inputs.biome_temperature, partial_ticks);
        // A held clock's sun stands still: no swing with the partial tick (the light shafts would
        // follow it every frame).
        let sun_partial_ticks =
            if self.shader_pack_clock_held() && mc_render::shaderpack::shadow::steady_shading() {
                0.0
            } else {
                partial_ticks
            };
        UnderwaterFrame {
            eye,
            eye_in_water: in_water,
            surface_height,
            wave_clock: self.tw_underwater.wave_clock,
            celestial_angle: environment.celestial_angle(sun_partial_ticks),
            sun_brightness: environment.sun_brightness(partial_ticks),
            rain: environment.weather.rain_strength,
            thunder: environment.weather.thunder_strength,
            sky_color: sky.to_array(),
            fog_color: air.color.to_array(),
            render_distance: self.settings.render_distance_chunks.clamp(2, 32) as f32 * 16.0,
            vanilla_density: self.tw_underwater.vanilla_density.get(),
            occlusion_origin: None,
        }
    }

    /// Keep the light-occlusion map around the eye current: [`OCCLUSION_ROWS_PER_TICK`] rows per
    /// game tick, chunk by chunk (one lookup per chunk, not per column); when the window moves a
    /// chunk the map is shifted in place and only the newly exposed strips are read (a full read
    /// is ~37,000 columns, about half a millisecond: too much for one frame). Uploads only when
    /// a column changed. Returns the map's corner.
    fn tw_underwater_occlusion(&mut self, eye: DVec3) -> Option<(i32, i32)> {
        let size = OCCLUSION_SIZE as i32;
        let origin = (
            ((eye.x.floor() as i32).div_euclid(16) - 6) * 16,
            ((eye.z.floor() as i32).div_euclid(16) - 6) * 16,
        );
        let tick = self.tick;
        let state = &mut self.tw_underwater;
        let moved = state
            .occlusion_origin
            .map(|old| (origin.0 - old.0, origin.1 - old.1));
        let fresh_tick = state.occlusion_tick != tick;
        if moved == Some((0, 0)) && !fresh_tick {
            return Some(origin);
        }
        state.occlusion_origin = Some(origin);
        // Chunk rows (in map rows, multiples of 16) and chunk columns to read this frame.
        let (rows, cols): ((i32, i32), (i32, i32)) = match moved {
            Some((0, 0)) => {
                state.occlusion_tick = tick;
                let start = state.occlusion_row;
                state.occlusion_row = (start + OCCLUSION_ROWS_PER_TICK) % size;
                ((start, start + OCCLUSION_ROWS_PER_TICK), (0, 0))
            }
            Some((dx, dz)) if dx.abs() < size && dz.abs() < size => {
                // The exposed strips: the rows (all columns) and columns (all rows) the shift
                // brought in. The rest keeps its values, moved to its new place below.
                let rows = if dz > 0 { (size - dz, size) } else { (0, -dz) };
                let cols = if dx > 0 { (size - dx, size) } else { (0, -dx) };
                (rows, cols)
            }
            _ => ((0, size), (0, 0)),
        };
        let (Some(session), Some(world_phase)) = (&self.session, &self.tw_underwater.world) else {
            return Some(origin);
        };
        let world = &session.state().world;
        world_phase.with(|phase| {
            let map = phase.occlusion_map();
            let mut changed = false;
            if let Some((dx, dz)) =
                moved.filter(|&(dx, dz)| (dx, dz) != (0, 0) && dx.abs() < size && dz.abs() < size)
            {
                shift_map(map, dx, dz);
                changed = true;
            }
            let mut read = |chunk_row: i32, chunk_col: i32| {
                let cz = (origin.1 + chunk_row) >> 4;
                let chunk = world.chunk(ChunkPos::new((origin.0 + chunk_col) >> 4, cz));
                for lz in 0..16u8 {
                    let row = (chunk_row + i32::from(lz)) as usize * OCCLUSION_SIZE as usize;
                    for lx in 0..16u8 {
                        let top = chunk.map_or(0, |c| column_top(c, lx, lz));
                        let cell = &mut map[row + chunk_col as usize + usize::from(lx)];
                        if *cell != top {
                            *cell = top;
                            changed = true;
                        }
                    }
                }
            };
            for chunk_row in (rows.0..rows.1).step_by(16) {
                for chunk_col in (0..size).step_by(16) {
                    read(chunk_row, chunk_col);
                }
            }
            for chunk_col in (cols.0..cols.1).step_by(16) {
                for chunk_row in (0..size).step_by(16) {
                    if !(rows.0..rows.1).contains(&chunk_row) {
                        read(chunk_row, chunk_col);
                    }
                }
            }
            if changed {
                phase.mark_occlusion_changed();
            }
        });
        Some(origin)
    }

    /// Blocks from the eye up to the top of the water column it is in (at least half a block).
    fn real_water_surface(&self, eye: DVec3) -> f32 {
        let Some(session) = &self.session else {
            return 4.0;
        };
        let world = &session.state().world;
        let start = BlockPos::from_vec3(eye);
        let mut top = f64::from(start.y + 1);
        for dy in 1..=SURFACE_SCAN {
            let pos = BlockPos::new(start.x, start.y + dy, start.z);
            let material = mc_data::block_props(world.block_at(pos)).material;
            if material != Material::Water {
                break;
            }
            top = f64::from(pos.y + 1);
        }
        ((top - eye.y) as f32).max(0.5)
    }

    /// The world phase, for tests and the screenshot tool.
    #[cfg(test)]
    pub(crate) fn tw_underwater_world<R>(
        &self,
        f: impl FnOnce(&mut UnderwaterPhase) -> R,
    ) -> Option<R> {
        self.tw_underwater
            .world
            .as_ref()
            .map(|phase| phase.with(|phase| f(phase)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_map_to_the_phase_defaults() {
        let settings = mc_mods::ModSettings::new();
        assert_eq!(read_settings(&settings), UnderwaterSettings::default());
        assert!(!settings.enabled(desc::ID), "a shader pack is opt-in");
    }

    #[test]
    fn picking_a_preset_writes_it_and_touching_a_value_makes_it_custom() {
        let mut s = mc_mods::ModSettings::new();
        let id = desc::ID;
        // A fresh install: the default preset and its values agree; nothing is written.
        assert!(!sync_preset(&mut s, None));
        let seen = Some(s.choice(id, "preset"));
        // Pick Deep sea: its values are written.
        s.set_choice(id, "preset", 2);
        assert!(sync_preset(&mut s, seen));
        assert_eq!(s.f32(id, "clarity"), desc::PRESETS[2].clarity);
        assert_eq!(s.color(id, "water_color"), desc::PRESETS[2].water_color);
        let seen = Some(s.choice(id, "preset"));
        assert!(!sync_preset(&mut s, seen), "stable once written");
        // Drag a slider: the choice becomes Custom, and the value stays.
        s.set_f32(id, "clarity", 200.0);
        assert!(sync_preset(&mut s, seen));
        assert_eq!(s.choice(id, "preset"), desc::PRESET_CUSTOM);
        assert_eq!(s.f32(id, "clarity"), 200.0);
        // Custom never overwrites anything.
        let seen = Some(s.choice(id, "preset"));
        assert!(!sync_preset(&mut s, seen));
        assert_eq!(s.f32(id, "clarity"), 200.0);
        // Values that happen to match a preset again (typed back by hand) show that preset only
        // when the choice is not Custom; Custom stays the user's.
        let p = desc::PRESETS[1];
        s.set_color(id, "water_color", p.water_color);
        s.set_f32(id, "clarity", p.clarity);
        s.set_f32(id, "red_fade", p.red_fade);
        s.set_f32(id, "depth_darkening", p.depth_darkening);
        assert!(!sync_preset(&mut s, seen));
        assert_eq!(s.choice(id, "preset"), desc::PRESET_CUSTOM);
    }

    /// The in-place shift equals reading every cell again from the moved window.
    #[test]
    fn shifting_the_occlusion_map_matches_a_fresh_read() {
        let n = OCCLUSION_SIZE as i32;
        // A "world": the value of world column (x, z), and a map of the window at a corner.
        let world = |x: i32, z: i32| ((x * 7 + z * 13).rem_euclid(251)) as u8;
        let window = |ox: i32, oz: i32| -> Vec<u8> {
            (0..n)
                .flat_map(|r| (0..n).map(move |c| world(ox + c, oz + r)))
                .collect()
        };
        for (dx, dz) in [(16, 0), (-16, 0), (0, 16), (0, -16), (16, -32), (-48, 16)] {
            let mut map = window(0, 0);
            shift_map(&mut map, dx, dz);
            let fresh = window(dx, dz);
            for r in 0..n {
                for c in 0..n {
                    let exposed = !(0..n).contains(&(r + dz)) || !(0..n).contains(&(c + dx));
                    let i = (r * n + c) as usize;
                    if !exposed {
                        assert_eq!(map[i], fresh[i], "({dx}, {dz}) at row {r}, col {c}");
                    }
                }
            }
        }
    }

    /// A client on the real assets, or `None` (skip) without the vanilla jar.
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
            eprintln!("SKIP: no vanilla jar in this checkout");
            return None;
        }
        let client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        Some((client, dir))
    }

    /// JoinGame then the first S08 (closes the terrain screen). Keep both channel ends alive.
    fn join_world(
        client: &mut Client,
    ) -> (
        tokio::sync::mpsc::Sender<crate::net::ToGame>,
        tokio::sync::mpsc::UnboundedReceiver<crate::net::ToNet>,
    ) {
        use crate::net::ToGame;
        use mc_protocol::play::{s2c, S2CPlay};
        let (handle, to_game, to_net) = crate::net::channel::NetHandle::new_pair();
        client.session = Some(crate::session::Session::new(
            handle,
            crate::session::JoinSettings::default(),
        ));
        let join = S2CPlay::JoinGame(s2c::JoinGame {
            entity_id: 27,
            hardcore_mode: false,
            game_mode: 0,
            dimension: 0,
            difficulty: 2,
            max_players: 20,
            level_type: "default".into(),
            reduced_debug_info: false,
        });
        to_game.try_send(ToGame::Packet(join)).expect("join");
        client.poll();
        let spawn = S2CPlay::PlayerPositionLook(s2c::PlayerPositionLook {
            x: 8.5,
            y: 70.0,
            z: 8.5,
            yaw: 0.0,
            pitch: 0.0,
            flags: 0,
        });
        to_game.try_send(ToGame::Packet(spawn)).expect("spawn");
        client.poll();
        (to_game, to_net)
    }

    /// The real driver: the pack takes vanilla's distance fade while it draws (and only then),
    /// steps aside while the player is blind, does nothing in Only under water with the eyes in
    /// air, and switched off hands everything back and holds no GPU objects.
    #[test]
    fn a_real_client_hands_the_distance_fade_to_the_pack_and_takes_it_back() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let Ok(gpu) = mc_render::Gpu::headless(
            64,
            64,
            mc_render::GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        ) else {
            eprintln!("SKIP: no wgpu adapter available on this machine");
            return;
        };
        let mut phases = PhaseList::new();
        client.install(&gpu, &mut phases, 64, 64);
        let _channels = join_world(&mut client);
        let mut camera = Camera::default();
        let mut frames = |client: &mut Client, n: usize| {
            for _ in 0..n {
                client.frame(&gpu, &mut camera, 0.0);
            }
        };
        let fog = |client: &Client| {
            client
                .render
                .as_ref()
                .expect("installed")
                .fog
                .with(|p| p.inputs)
        };
        let active = |client: &Client| client.tw_underwater_world(|p| p.is_active());

        frames(&mut client, 2);
        assert!(!client.tw_underwater.drawing(), "off by default");
        assert!(fog(&client).distance_fog && !fog(&client).clear_liquids);
        assert_eq!(active(&client), Some(false));

        client.mods.settings.set_enabled(desc::ID, true);
        frames(&mut client, 2);
        assert!(
            client.tw_underwater.drawing(),
            "Everywhere draws in a world"
        );
        assert!(!fog(&client).distance_fog && fog(&client).clear_liquids);
        assert_eq!(active(&client), Some(true));
        assert!(phases.scene_mode());

        client.active_effects.insert(
            15,
            super::super::super::TrackedEffect {
                amplifier: 0,
                duration: 200,
                duration_max: false,
                show_particles: true,
            },
        );
        frames(&mut client, 1);
        assert!(
            !client.tw_underwater.drawing(),
            "blind: vanilla's fog rules"
        );
        assert!(fog(&client).distance_fog);
        assert_eq!(active(&client), Some(false));
        client.active_effects.clear();

        client
            .mods
            .settings
            .set_choice(desc::ID, "mode", desc::MODE_UNDER_WATER);
        frames(&mut client, 2);
        assert!(!client.tw_underwater.drawing(), "eyes in air");
        assert!(fog(&client).distance_fog);
        assert!(!phases.scene_mode());

        client
            .mods
            .settings
            .set_choice(desc::ID, "mode", desc::MODE_EVERYWHERE);
        frames(&mut client, 2);
        assert!(client.tw_underwater.drawing());
        client.mods.settings.set_enabled(desc::ID, false);
        frames(&mut client, 2);
        assert!(!client.tw_underwater.drawing());
        assert!(fog(&client).distance_fog && !fog(&client).clear_liquids);
        assert_eq!(active(&client), Some(false));
        assert_eq!(client.tw_underwater_world(|p| p.allocated()), Some(false));
        assert!(!phases.scene_mode());
    }

    #[test]
    fn a_loaded_file_whose_values_match_no_preset_shows_custom() {
        let mut s = mc_mods::ModSettings::new();
        s.set_f32(desc::ID, "clarity", 64.0);
        assert!(sync_preset(&mut s, None));
        assert_eq!(s.choice(desc::ID, "preset"), desc::PRESET_CUSTOM);
    }

    #[test]
    fn the_options_reach_the_phase_settings() {
        let mut s = mc_mods::ModSettings::new();
        let id = desc::ID;
        s.set_choice(id, "mode", desc::MODE_UNDER_WATER);
        s.set_f32(id, "caustic_strength", 80.0);
        s.set_bool(id, "light_shafts", false);
        s.set_bool(id, "refraction", true);
        s.set_choice(id, "quality", 2);
        let r = read_settings(&s);
        assert_eq!(r.mode, UnderwaterMode::OnlyUnderWater);
        assert!((r.caustic_strength - 0.8).abs() < 1e-6);
        assert!(!r.light_shafts);
        assert!(r.refraction);
        assert_eq!(r.quality, UnderwaterQuality::High);
    }
}
