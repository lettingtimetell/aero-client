//! World-rendering mods, client half.
//!
//! * **Sky** (`sky`, World window, on by default: on is vanilla). [`Client::mod_sky`] in `driver.rs::drive_atmosphere` writes the switch and its two options into
//!   the sky phase (`mc_render::sky::dome::SkyPhase::{vanilla_sky, vanilla_sun_and_moon,
//!   vanilla_stars}`). Off: no sky plane, sunrise glow, sun, moon, stars, horizon/void planes or
//!   End box, so a Custom Sky pack's layers stand alone over the fog colour.
//!
//! The settings snapshot lives in `Cell`s (the hook runs inside a `render.sky.with(..)` closure,
//! under a shared borrow of `self.render`) and is re-read once per settings generation. At the
//! vanilla default the hook costs one `u64` compare and three `bool` stores per frame: no
//! allocation, lock, hashing or formatting. Nothing here sends a packet.
//!
//! The lane's other items live in the renderer and need no client state: the item-frame map
//! atlas (`mc_render::entity::map_atlas`, wired in `app/entrender.rs`), the end portal's back-face
//! cull (`mc_render::entity::tilepass`), translucent terrain and the vignette.

use std::cell::Cell;

use mc_mods::mods::sky;
use mc_mods::ModSettings;
use mc_render::sky::SkyPhase;

use super::super::Client;

/// The Sky mod's settings, in the renderer's terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SkySettings {
    /// `sky` on: [`SkyPhase::vanilla_sky`].
    vanilla_sky: bool,
    /// "Sun and moon": [`SkyPhase::vanilla_sun_and_moon`].
    sun_and_moon: bool,
    /// "Stars": [`SkyPhase::vanilla_stars`].
    stars: bool,
}

impl Default for SkySettings {
    /// Vanilla, and the sky phase's own defaults.
    fn default() -> Self {
        Self {
            vanilla_sky: true,
            sun_and_moon: true,
            stars: true,
        }
    }
}

impl SkySettings {
    fn read(s: &ModSettings) -> Self {
        Self {
            vanilla_sky: s.enabled(sky::DESCRIPTOR.id),
            sun_and_moon: s.bool(sky::DESCRIPTOR.id, sky::SUN_AND_MOON),
            stars: s.bool(sky::DESCRIPTOR.id, sky::STARS),
        }
    }
}

/// The client state.
#[derive(Debug, Default)]
pub(crate) struct X7State {
    /// The settings generation `sky` was read at; `0` = never (the store starts at 1).
    seen: Cell<u64>,
    sky: Cell<SkySettings>,
}

impl X7State {
    /// The Sky snapshot, re-read only when the settings changed since the last call.
    fn sky(&self, s: &ModSettings) -> SkySettings {
        let generation = s.generation();
        if self.seen.get() != generation {
            self.seen.set(generation);
            self.sky.set(SkySettings::read(s));
        }
        self.sky.get()
    }
}

impl Client {
    /// Sky: which vanilla sky parts draw this frame.
    pub(crate) fn mod_sky(&self, phase: &mut SkyPhase) {
        let settings = self.x7.sky(&self.mods.settings);
        phase.vanilla_sky = settings.vanilla_sky;
        phase.vanilla_sun_and_moon = settings.sun_and_moon;
        phase.vanilla_stars = settings.stars;
    }

    /// Whether vanilla would draw its vignette this frame, apart from
    /// the Fancy-graphics gate the callers apply.
    ///
    /// java: `renderVignette` (the brightness vignette and the world-border warning) runs inside
    /// `GuiIngame.renderGameOverlay`, which `EntityRenderer.updateCameraAndRender` calls only
    /// `if (!gameSettings.hideGUI || currentScreen != null)`. So with F1 in game there is no
    /// vignette at all, and `prevVignetteBrightness` pauses, as with Fast graphics. Behind an
    /// open screen it still draws.
    pub(crate) fn vignette_overlay_drawn(&self) -> bool {
        !self.hide_gui || !matches!(self.ui, super::super::Ui::InGame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh install is vanilla, and so are the renderer's defaults.
    #[test]
    fn defaults_are_vanilla() {
        let s = ModSettings::new();
        assert!(s.enabled("sky"));
        assert_eq!(SkySettings::read(&s), SkySettings::default());
        assert_eq!(X7State::default().sky(&s), SkySettings::default());
    }

    #[test]
    fn the_snapshot_follows_each_switch_and_rereads_only_on_change() {
        let mut s = ModSettings::new();
        let state = X7State::default();
        s.set_enabled("sky", false);
        assert_eq!(
            state.sky(&s),
            SkySettings {
                vanilla_sky: false,
                sun_and_moon: true,
                stars: true,
            }
        );
        s.set_enabled("sky", true);
        s.set_bool("sky", sky::SUN_AND_MOON, false);
        s.set_bool("sky", sky::STARS, false);
        assert_eq!(
            state.sky(&s),
            SkySettings {
                vanilla_sky: true,
                sun_and_moon: false,
                stars: false,
            }
        );
        // An unchanged generation never re-reads: a planted stale value proves the early return.
        state.sky.set(SkySettings::default());
        assert_eq!(state.sky(&s), SkySettings::default());
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

    /// java: `renderGameOverlay` (and so `renderVignette`) runs `if (!hideGUI || currentScreen
    /// != null)`.
    #[test]
    fn f1_in_game_hides_the_vignette_but_an_open_screen_keeps_it() {
        use super::super::super::Ui;
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        client.ui = Ui::InGame;
        client.hide_gui = false;
        assert!(client.vignette_overlay_drawn());
        client.hide_gui = true;
        assert!(
            !client.vignette_overlay_drawn(),
            "F1 in game: no overlay, no vignette"
        );
        client.ui = Ui::Loading;
        assert!(
            client.vignette_overlay_drawn(),
            "a screen is open: the overlay draws"
        );
        let driver = include_str!("../driver.rs");
        assert_eq!(
            driver
                .matches("self.settings.fancy_graphics && self.vignette_overlay_drawn()")
                .count(),
            2,
            "both border-warning sites"
        );
        let brightness = driver
            .split_once("let vignette_brightness = (self.settings.fancy_graphics")
            .map(|(_, rest)| &rest[..rest.find(".then_some(").unwrap()])
            .expect("the vignette brightness gate");
        assert!(brightness.contains("self.vignette_overlay_drawn()"));
    }

    /// The hook is called once, in `drive_atmosphere`, and the state is initialised once.
    #[test]
    fn the_hook_is_wired_at_its_driver_site() {
        let driver = include_str!("../driver.rs");
        for needle in [
            "render.sky.with(|phase| self.mod_sky(phase));",
            "x7: Default::default(),",
        ] {
            assert_eq!(driver.matches(needle).count(), 1, "{needle}");
        }
        let atmosphere = driver.find("fn drive_atmosphere(").unwrap();
        let hook = driver
            .find("render.sky.with(|phase| self.mod_sky(phase));")
            .unwrap();
        let next_fn = atmosphere + 1 + driver[atmosphere + 1..].find("\n    fn ").unwrap();
        assert!(
            atmosphere < hook && hook < next_fn,
            "inside drive_atmosphere"
        );
    }
}
