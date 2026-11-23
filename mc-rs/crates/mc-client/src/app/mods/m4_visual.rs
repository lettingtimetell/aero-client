//! The Visual window's Fullbright, Time Changer and Block Overlay,
//! and the PvP window's Crosshair, Fire Height and Shiny Pots.
//!
//! The hooks are `&self` methods because the driver calls them while it holds shared borrows of
//! `self.render` (the fire hook runs inside a phase closure). The settings snapshot therefore
//! lives in `Cell`s and is refreshed lazily, once per settings generation. With every M4 mod off
//! a hook costs one `u64` compare and one `Option` read.
//!
//! Hooks, by mod:
//! * `fullbright`: [`Client::mod_lightmap`] in `drive_atmosphere`, and
//!   [`Client::mod_tile_fullbright`] in `tiles.rs`;
//! * `time_changer`: [`Client::mod_display_world_time`] at the three `SkyEnvironment` builds;
//! * `fire_height`: [`Client::mod_fire_overlay`] in `drive_atmosphere`;
//! * `block_overlay`: [`Client::mod_selection_style`] in `drive_block_overlay`;
//! * `crosshair`, `shiny_pots`: [`Client::mod_m4_frame`] in `Client::frame`.
//!
//! Nothing here sends a packet or touches the world: every hook changes only what is drawn.

use std::cell::Cell;

use mc_mods::mods::{crosshair, fullbright, shiny_pots, time_changer};
use mc_mods::ModSettings;
use mc_render::post::overlay::{FireOverlayStyle, FIRST_PERSON_FIRE_ALPHA};
use mc_render::post::OverlayPhase;
use mc_render::sky::LightmapParams;
use mc_render::terrain::SelectionStyle;
use mc_ui::modvisual::{CustomCrosshair, ShinyPotsArea};

use super::super::Client;

/// Minecraft ticks in one day.
const TICKS_PER_DAY: i64 = 24_000;
/// Potion id 12: fire resistance.
const FIRE_RESISTANCE: i8 = 12;
/// Fire Height's "Hide with Fire Resistance" shows the overlay again for the last 5 seconds of
/// the effect, as a warning.
const FIRE_RESISTANCE_WARNING_TICKS: i32 = 100;

/// The Fire Height mod's settings.
#[derive(Debug, Clone, Copy, PartialEq)]
struct FireHeight {
    style: FireOverlayStyle,
    hide_with_fire_resistance: bool,
}

impl FireHeight {
    /// Whether the overlay draws at all, given the ticks left on the local player's Fire
    /// Resistance: not at opacity 0, and not while more than five seconds of Fire Resistance
    /// remain when "Hide with Fire Resistance" is on.
    fn shows(&self, fire_resistance_ticks: Option<i32>) -> bool {
        let resisted = self.hide_with_fire_resistance
            && fire_resistance_ticks.is_some_and(|ticks| ticks > FIRE_RESISTANCE_WARNING_TICKS);
        self.style.alpha > 0.0 && !resisted
    }
}

/// The client state: a typed snapshot of its six mods' settings. Each field is `None`
/// while its mod is off.
#[derive(Debug, Default)]
pub(crate) struct M4State {
    /// The settings generation the snapshot was read at; `0` = never (the store starts at 1).
    seen: Cell<u64>,
    /// Fullbright's brightness in percent (`100..=1500`; 1500 is Full).
    brightness: Cell<Option<f32>>,
    /// Time Changer's time of day in ticks (`0..24000`).
    time_of_day: Cell<Option<i64>>,
    fire: Cell<Option<FireHeight>>,
    selection: Cell<Option<SelectionStyle>>,
    crosshair: Cell<Option<CustomCrosshair>>,
    shiny_pots: Cell<Option<ShinyPotsArea>>,
}

impl M4State {
    /// Re-read the settings if they changed since the last call. While nothing changed (and so
    /// with every M4 mod off), this compare is the whole cost.
    fn sync(&self, s: &ModSettings) {
        let generation = s.generation();
        if self.seen.get() == generation {
            return;
        }
        self.seen.set(generation);
        self.brightness.set(
            s.enabled("fullbright")
                .then(|| s.f32("fullbright", "brightness")),
        );
        self.time_of_day.set(
            s.enabled("time_changer")
                .then(|| time_changer::time_of_day_ticks(s.f32("time_changer", "time"))),
        );
        self.fire.set(s.enabled("fire_height").then(|| FireHeight {
            style: FireOverlayStyle {
                height_offset: s.f32("fire_height", "height"),
                // 100% is vanilla's 0.9.
                alpha: FIRST_PERSON_FIRE_ALPHA * s.f32("fire_height", "opacity") / 100.0,
            },
            hide_with_fire_resistance: s.bool("fire_height", "hide_fire_resistance"),
        }));
        self.selection.set(s.enabled("block_overlay").then(|| {
            SelectionStyle::from_argb(
                s.color("block_overlay", "outline_color"),
                s.f32("block_overlay", "outline_width"),
                s.color("block_overlay", "fill_color"),
            )
        }));
        self.crosshair.set(s.enabled("crosshair").then(|| {
            CustomCrosshair {
                length: s.f32("crosshair", "length").round() as i32,
                gap: s.f32("crosshair", "gap").round() as i32,
                thickness: s.f32("crosshair", "thickness").round() as i32,
                dot: s.bool("crosshair", "dot"),
                color: (s.choice("crosshair", "color_mode") == crosshair::COLOR_MODE_CUSTOM)
                    .then(|| s.color("crosshair", "color")),
                show_in_f5: s.bool("crosshair", "show_in_f5"),
            }
        }));
        self.shiny_pots.set(s.enabled("shiny_pots").then(|| {
            if s.choice("shiny_pots", "area") == shiny_pots::AREA_WHOLE_SLOT {
                ShinyPotsArea::WholeSlot
            } else {
                ShinyPotsArea::ItemBox
            }
        }));
    }
}

/// Fullbright's lightmap: Full switches the table to all-white (the path the `options.txt`
/// `fullbright` key also takes); a lower value replaces the video gamma with `brightness / 100`,
/// which vanilla's gamma lerp extrapolates past 1.0.
fn lightmap_with_brightness(params: LightmapParams, brightness: f32) -> LightmapParams {
    if brightness >= fullbright::BRIGHTNESS_FULL {
        LightmapParams {
            fullbright: true,
            ..params
        }
    } else {
        LightmapParams {
            gamma: brightness / 100.0,
            ..params
        }
    }
}

/// Time Changer's world time: the real day count with the chosen time of day. Keeping the day
/// keeps the server's moon phase (`(time / 24000) % 8`).
fn display_world_time(real: i64, time_of_day: i64) -> i64 {
    real.wrapping_sub(real.rem_euclid(TICKS_PER_DAY))
        .wrapping_add(time_of_day)
}

impl Client {
    /// The snapshot, refreshed if the settings moved.
    fn m4_synced(&self) -> &M4State {
        self.m4.sync(&self.mods.settings);
        &self.m4
    }

    /// Fullbright: this frame's lightmap parameters with the mod's
    /// brightness. Every lightmap owner (terrain, particles, tile entities, entities, the hand)
    /// gets the result. Off: `params` unchanged.
    pub(crate) fn mod_lightmap(&self, params: LightmapParams) -> LightmapParams {
        match self.m4_synced().brightness.get() {
            None => params,
            Some(brightness) => lightmap_with_brightness(params, brightness),
        }
    }

    /// Fullbright's Full for the tile entities drawn with a CPU brightness (chests, signs, skulls,
    /// banners, beacons), which the all-white table cannot reach. Their brightness never follows
    /// gamma, so only Full changes them.
    pub(crate) fn mod_tile_fullbright(&self) -> bool {
        self.m4_synced()
            .brightness
            .get()
            .is_some_and(|brightness| brightness >= fullbright::BRIGHTNESS_FULL)
    }

    /// Time Changer: the world time the sky, fog, clouds, stars and lightmap render with. The
    /// server's time, the clock item and F3 keep using the real one. Off: `real`.
    pub(crate) fn mod_display_world_time(&self, real: i64) -> i64 {
        match self.m4_synced().time_of_day.get() {
            None => real,
            Some(time_of_day) => display_world_time(real, time_of_day),
        }
    }

    /// Fire Height right after vanilla's `phase.fire` is set: moves and
    /// fades the first-person fire, or hides it (opacity 0, or Fire Resistance with more than
    /// five seconds left when that option is on). Off: vanilla's quads (`fire_style = None`).
    pub(crate) fn mod_fire_overlay(&self, phase: &mut OverlayPhase) {
        let Some(fire) = self.m4_synced().fire.get() else {
            phase.fire_style = None;
            return;
        };
        phase.fire_style = Some(fire.style);
        let fire_resistance = self
            .active_effects
            .get(&FIRE_RESISTANCE)
            .map(|effect| effect.duration);
        if !fire.shows(fire_resistance) {
            phase.fire = false;
        }
    }

    /// Block Overlay: the selection box's fill and wide outline.
    /// Off: `None`, vanilla's one-pixel line. The box and its gates are vanilla's either way.
    pub(crate) fn mod_selection_style(&self) -> Option<SelectionStyle> {
        self.m4_synced().selection.get()
    }

    /// Crosshair and Shiny Pots (before the UI is built): hand mc-ui
    /// this frame's values. Two stores; `None` for both is vanilla.
    pub(crate) fn mod_m4_frame(&mut self) {
        let m4 = self.m4_synced();
        let (custom_crosshair, glint_area) = (m4.crosshair.get(), m4.shiny_pots.get());
        self.hud.custom_crosshair = custom_crosshair;
        mc_ui::modvisual::set_shiny_pots(glint_area);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> LightmapParams {
        LightmapParams {
            gamma: 0.5,
            fullbright: false,
            ..LightmapParams::default()
        }
    }

    #[test]
    fn fullbright_is_the_white_table_at_full_and_a_gamma_override_below() {
        let full = lightmap_with_brightness(params(), fullbright::BRIGHTNESS_FULL);
        assert!(full.fullbright);
        assert_eq!(full.gamma, 0.5, "Full leaves the video gamma alone");
        let boosted = lightmap_with_brightness(params(), 800.0);
        assert!(!boosted.fullbright);
        assert_eq!(boosted.gamma, 8.0);
        assert_eq!(
            LightmapParams {
                gamma: 0.5,
                ..boosted
            },
            params(),
            "nothing else changes"
        );
    }

    #[test]
    fn the_display_time_keeps_the_real_day_and_so_the_moon_phase() {
        assert_eq!(display_world_time(0, 6_000), 6_000);
        assert_eq!(display_world_time(23_999, 6_000), 6_000);
        assert_eq!(
            display_world_time(5 * 24_000 + 17_000, 18_000),
            5 * 24_000 + 18_000
        );
        assert_eq!(display_world_time(5 * 24_000, 0), 5 * 24_000);
        // Absurd inputs wrap instead of panicking.
        let _ = display_world_time(i64::MIN, 23_999);
        let _ = display_world_time(i64::MAX, 23_999);
    }

    #[test]
    fn the_snapshot_is_read_once_per_generation() {
        let mut settings = ModSettings::new();
        let state = M4State::default();
        state.sync(&settings);
        assert_eq!(state.brightness.get(), None);
        assert_eq!(state.selection.get(), None);
        assert_eq!(state.crosshair.get(), None);
        assert_eq!(state.shiny_pots.get(), None);
        assert_eq!(state.fire.get(), None);
        assert_eq!(state.time_of_day.get(), None);

        settings.set_enabled("fullbright", true);
        settings.set_enabled("time_changer", true);
        settings.set_enabled("fire_height", true);
        settings.set_enabled("block_overlay", true);
        settings.set_enabled("crosshair", true);
        settings.set_enabled("shiny_pots", true);
        state.sync(&settings);
        assert_eq!(state.brightness.get(), Some(fullbright::BRIGHTNESS_FULL));
        assert_eq!(state.time_of_day.get(), Some(6_000), "noon by default");
        let fire = state.fire.get().expect("fire height on");
        assert_eq!(fire.style.height_offset, -0.3);
        assert_eq!(fire.style.alpha, FIRST_PERSON_FIRE_ALPHA);
        assert!(!fire.hide_with_fire_resistance);
        assert_eq!(
            state.selection.get(),
            Some(SelectionStyle::from_argb(0x9900_0000, 3.0, 0x2600_0000))
        );
        assert_eq!(
            state.crosshair.get(),
            Some(CustomCrosshair {
                length: 4,
                gap: 2,
                thickness: 1,
                dot: false,
                color: None,
                show_in_f5: true,
            })
        );
        assert_eq!(state.shiny_pots.get(), Some(ShinyPotsArea::ItemBox));

        settings.set_choice("crosshair", "color_mode", crosshair::COLOR_MODE_CUSTOM);
        settings.set_color("crosshair", "color", 0x80FF_0000);
        settings.set_choice("shiny_pots", "area", shiny_pots::AREA_WHOLE_SLOT);
        settings.set_f32("fire_height", "opacity", 50.0);
        settings.set_f32("time_changer", "time", 0.0);
        state.sync(&settings);
        assert_eq!(state.crosshair.get().unwrap().color, Some(0x80FF_0000));
        assert_eq!(state.shiny_pots.get(), Some(ShinyPotsArea::WholeSlot));
        assert!((state.fire.get().unwrap().style.alpha - 0.45).abs() < 1e-6);
        assert_eq!(state.time_of_day.get(), Some(18_000), "midnight");

        // An unchanged generation never re-reads: a stale value proves the early return.
        state.brightness.set(Some(123.0));
        state.sync(&settings);
        assert_eq!(state.brightness.get(), Some(123.0));
        settings.set_enabled("fullbright", false);
        state.sync(&settings);
        assert_eq!(state.brightness.get(), None);
    }

    #[test]
    fn fire_hides_at_zero_opacity_and_while_fire_resistance_lasts_if_asked() {
        let fire = FireHeight {
            style: FireOverlayStyle {
                height_offset: -0.3,
                alpha: FIRST_PERSON_FIRE_ALPHA,
            },
            hide_with_fire_resistance: false,
        };
        assert!(fire.shows(None));
        assert!(fire.shows(Some(6_000)), "the option is off");
        let hide = FireHeight {
            hide_with_fire_resistance: true,
            ..fire
        };
        assert!(hide.shows(None));
        assert!(!hide.shows(Some(101)));
        assert!(
            hide.shows(Some(100)),
            "the last 5 seconds show it again as a warning"
        );
        let invisible = FireHeight {
            style: FireOverlayStyle {
                alpha: 0.0,
                ..fire.style
            },
            ..fire
        };
        assert!(!invisible.shows(None));
    }

    /// Every hook is called at its site, once. A merge that drops one fails here.
    #[test]
    fn every_hook_is_wired_at_its_driver_site() {
        let driver = include_str!("../driver.rs");
        assert_eq!(
            driver
                .matches("world_time: self.mod_display_world_time(env.client_world_time()),")
                .count(),
            3,
            "the three SkyEnvironment builds (fog light, environment, particles)"
        );
        assert!(!driver.contains("world_time: env.client_world_time(),"));
        for hook in [
            "let params = self.mod_lightmap(params);",
            "phase.fire = fire_overlay;\n            self.mod_fire_overlay(phase);",
            "let selection_style = self.mod_selection_style();",
            "phase.selection_style = selection_style;",
            "self.mod_m4_frame();",
            "m4: Default::default(),",
        ] {
            assert_eq!(driver.matches(hook).count(), 1, "{hook}");
        }
        let literal = driver
            .find("fullbright: self.settings.fullbright,")
            .unwrap();
        let hook = driver
            .find("let params = self.mod_lightmap(params);")
            .unwrap();
        let upload = driver
            .find("render.terrain.borrow_mut().set_lightmap_params(gpu, params);")
            .unwrap();
        assert!(
            literal < hook && hook < upload,
            "fullbright edits the params every owner gets"
        );
        let frame = driver.find("self.mod_m4_frame();").unwrap();
        assert!(
            driver[frame..].contains("self.build_ui(gpu, partial_ticks);"),
            "the crosshair and glint are handed to mc-ui before the UI is built"
        );
        assert!(include_str!("../tiles.rs")
            .contains("let fullbright = self.settings.fullbright || self.mod_tile_fullbright();"));
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

    const MODS: [&str; 6] = [
        "fullbright",
        "time_changer",
        "fire_height",
        "block_overlay",
        "crosshair",
        "shiny_pots",
    ];

    /// The hooks on a real client: identity while the mods are off, the settings while they are
    /// on, and identity again once they are switched off.
    #[test]
    fn a_real_client_is_vanilla_with_the_mods_off_and_again_after_they_were_on() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let gpu = mc_render::Gpu::headless(
            8,
            8,
            mc_render::GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        )
        .ok();
        let mut overlay = gpu.as_ref().map(OverlayPhase::new);
        let params = LightmapParams {
            gamma: 0.25,
            ..LightmapParams::default()
        };
        let assert_vanilla = |client: &mut Client, overlay: &mut Option<OverlayPhase>| {
            assert_eq!(client.mod_lightmap(params), params);
            assert!(!client.mod_tile_fullbright());
            assert_eq!(client.mod_display_world_time(30_123), 30_123);
            assert_eq!(client.mod_selection_style(), None);
            client.mod_m4_frame();
            assert_eq!(client.hud.custom_crosshair, None);
            assert_eq!(mc_ui::modvisual::shiny_pots(), None);
            if let Some(phase) = overlay {
                phase.fire = true;
                client.mod_fire_overlay(phase);
                assert!(phase.fire && phase.fire_style.is_none());
            }
        };
        assert_vanilla(&mut client, &mut overlay);

        for id in MODS {
            client.mods.settings.set_enabled(id, true);
        }
        assert!(client.mod_lightmap(params).fullbright);
        assert!(client.mod_tile_fullbright());
        assert_eq!(client.mod_display_world_time(30_123), 24_000 + 6_000);
        assert!(client.mod_selection_style().is_some());
        client.mod_m4_frame();
        assert!(client.hud.custom_crosshair.is_some());
        assert_eq!(mc_ui::modvisual::shiny_pots(), Some(ShinyPotsArea::ItemBox));
        if let Some(phase) = &mut overlay {
            phase.fire = true;
            client.mod_fire_overlay(phase);
            assert!(phase.fire);
            assert_eq!(phase.fire_style.map(|s| s.height_offset), Some(-0.3));
            client
                .mods
                .settings
                .set_bool("fire_height", "hide_fire_resistance", true);
            client.active_effects.insert(
                FIRE_RESISTANCE,
                super::super::super::TrackedEffect {
                    amplifier: 0,
                    duration: 600,
                    duration_max: false,
                    show_particles: true,
                },
            );
            client.mod_fire_overlay(phase);
            assert!(!phase.fire, "hidden while fire resistance lasts");
            client.active_effects.clear();
        }

        for id in MODS {
            client.mods.settings.set_enabled(id, false);
        }
        assert_vanilla(&mut client, &mut overlay);
    }
}
