//! The client half of the ASCII Super Secret Setting, and the
//! settings snapshot for 1.7 Animations' "1.7 item edges" (`translucent_held`, pushed into the
//! first-person phase by `driver.rs`; the rendering is
//! `mc_render::pipeline::old_item_edges`).
//!
//! The render half is [`AsciiPhase`] (`mc_render::post::ascii`) at [`AsciiPhase::ORDER`],
//! registered by [`Client::install_rfx`]. The `super_secret_settings` mod's "Shader" choice
//! "ASCII" (index [`ASCII_CHOICE`]) turns it on, with "Character size" and "Character colors" as
//! its style. Vanilla's own groups keep going through `x10_blur.rs`: choosing ASCII there maps to
//! no vanilla group, so `x10_blur.rs` clears any group that was loaded, exactly as "Off" does.
//!
//! The effect runs while a world is on screen through the local player's eyes, as vanilla's
//! Super Secret shaders do; the main menu and a spectated mob's view are left alone.
//!
//! The font sheet (`textures/font/ascii.png` from the active packs) is read once when the effect
//! first needs it after an install (a resource reload builds a new phase, which asks again).
//!
//! **Off costs nothing:** a frame costs one `u64` compare and a `bool` here, the phase reports
//! itself disabled, never asks for the scene copy, and owns no GPU object (`release` drops them
//! the moment the choice or the mod is switched off).

use mc_mods::mods::super_secret_settings::ASCII_CHOICE;
use mc_render::post::ascii::{AsciiColors, AsciiPhase, AsciiSize};
use mc_render::PhaseList;

use super::super::{Client, Shared};

/// What the client last read from the mod and pushed into the phase.
#[derive(Default)]
pub(crate) struct RfxState {
    /// The settings generation last read. 0 = never (the store's generation starts at 1).
    seen: u64,
    /// The mod is on and its choice is ASCII.
    ascii_on: bool,
    /// 1.7 Animations is on with its "1.7 item edges" option (`translucent_held`): pushed into
    /// the first-person phase every frame (`mc_render::pipeline::old_item_edges`).
    pub(crate) old_item_edges: bool,
    /// The phase, shared with the `PhaseList`. `None` until `Client::install`.
    phase: Option<Shared<AsciiPhase>>,
}

/// The mod's ASCII settings: on, size and colours.
fn ascii_settings(settings: &mc_mods::ModSettings) -> (bool, AsciiSize, AsciiColors) {
    const MOD: &str = "super_secret_settings";
    let on = settings.enabled(MOD) && settings.choice(MOD, "shader") == ASCII_CHOICE;
    let size = AsciiSize::from_index(settings.choice(MOD, "ascii_size"));
    let colors = AsciiColors::from_index(settings.choice(MOD, "ascii_colors"));
    (on, size, colors)
}

/// 1.7 Animations' "1.7 item edges": the mod on and its `translucent_held` option on.
fn old_item_edges(settings: &mc_mods::ModSettings) -> bool {
    settings.enabled("old_animations") && settings.bool("old_animations", "translucent_held")
}

impl Client {
    /// Register the ASCII phase. Runs from `Client::install`, so also after a resource-pack
    /// reload, which rebuilds every phase (and the new phase asks for the new pack's font).
    pub(crate) fn install_rfx(&mut self, phases: &mut PhaseList) {
        let phase = Shared::new(AsciiPhase::new());
        phases.insert(AsciiPhase::ORDER, Box::new(phase.clone()));
        self.rfx.phase = Some(phase);
        // Push the settings into the new phase on the next frame.
        self.rfx.seen = 0;
    }

    /// Once per frame from `Client::frame`. Off: one generation compare and a `bool`.
    pub(crate) fn rfx_frame(&mut self) {
        let generation = self.mods.settings.generation();
        if self.rfx.seen != generation {
            self.rfx.seen = generation;
            let (on, size, colors) = ascii_settings(&self.mods.settings);
            self.rfx.ascii_on = on;
            self.rfx.old_item_edges = old_item_edges(&self.mods.settings);
            if let Some(phase) = &self.rfx.phase {
                phase.with(|phase| {
                    phase.set_style(size, colors);
                    if !on {
                        // Off: no pipeline, no ramp, no target. The next frame on rebuilds them.
                        phase.release();
                    }
                });
            }
        }
        if !self.rfx.ascii_on {
            return;
        }
        let Some(phase) = &self.rfx.phase else {
            return;
        };
        let world = self.session.as_ref().is_some_and(|session| {
            matches!(
                session.render_view_target(),
                crate::session::RenderViewTarget::Local
            )
        });
        let wants_font = phase.with(|phase| {
            phase.set_active(world);
            world && phase.needs_font()
        });
        if wants_font {
            let location = mc_assets::ResourceLocation::minecraft("textures/font/ascii.png");
            let font = match self.assets.resources.load_png_rgba(&location) {
                Ok((rgba, width, height, _)) => Some((rgba, width, height)),
                Err(error) => {
                    tracing::warn!(%error, "ASCII: no font sheet; using the built-in glyphs");
                    None
                }
            };
            phase.with(|phase| {
                phase.set_font(
                    font.as_ref()
                        .map(|(rgba, width, height)| (rgba.as_slice(), *width, *height)),
                );
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_is_on_only_for_its_own_choice_with_the_mod_on() {
        let mut settings = mc_mods::ModSettings::default();
        let (on, size, colors) = ascii_settings(&settings);
        assert!(!on, "the mod is off by default");
        assert_eq!(size, AsciiSize::Medium);
        assert_eq!(colors, AsciiColors::Color);
        settings.set_enabled("super_secret_settings", true);
        assert!(!ascii_settings(&settings).0, "Shader is Off by default");
        settings.set_choice("super_secret_settings", "shader", ASCII_CHOICE);
        assert!(ascii_settings(&settings).0);
        // A vanilla group is not ASCII.
        settings.set_choice("super_secret_settings", "shader", 3);
        assert!(!ascii_settings(&settings).0);
        settings.set_choice("super_secret_settings", "shader", ASCII_CHOICE);
        settings.set_choice("super_secret_settings", "ascii_size", 2);
        settings.set_choice("super_secret_settings", "ascii_colors", 1);
        assert_eq!(
            ascii_settings(&settings),
            (true, AsciiSize::Large, AsciiColors::White)
        );
        settings.set_enabled("super_secret_settings", false);
        assert!(!ascii_settings(&settings).0);
    }

    /// A client on the real assets, or `None` (skip) without the vanilla jar. The `TempDir` is
    /// the game dir and must outlive the client.
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

    /// JoinGame, then the first S08. Keep both channel ends alive.
    fn join_world(
        client: &mut Client,
    ) -> (
        tokio::sync::mpsc::Sender<crate::net::ToGame>,
        tokio::sync::mpsc::UnboundedReceiver<crate::net::ToNet>,
    ) {
        use crate::net::ToGame;
        use crate::session::{JoinSettings, Session};
        let (handle, to_game, to_net) = crate::net::channel::NetHandle::new_pair();
        client.session = Some(Session::new(handle, JoinSettings::default()));
        to_game
            .try_send(ToGame::Packet(mc_protocol::play::S2CPlay::JoinGame(
                mc_protocol::play::s2c::JoinGame {
                    entity_id: 27,
                    hardcore_mode: false,
                    game_mode: 0,
                    dimension: 0,
                    difficulty: 2,
                    max_players: 20,
                    level_type: "default".into(),
                    reduced_debug_info: false,
                },
            )))
            .expect("join channel");
        client.poll();
        to_game
            .try_send(ToGame::Packet(
                mc_protocol::play::S2CPlay::PlayerPositionLook(
                    mc_protocol::play::s2c::PlayerPositionLook {
                        x: 8.5,
                        y: 70.0,
                        z: 8.5,
                        yaw: 0.0,
                        pitch: 0.0,
                        flags: 0,
                    },
                ),
            ))
            .expect("position channel");
        client.poll();
        (to_game, to_net)
    }

    /// The phase follows the choice and the world; off releases it; the font is read once.
    #[test]
    fn ascii_runs_only_in_a_world_and_off_releases_everything() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let mut phases = PhaseList::new();
        client.install_rfx(&mut phases);
        let names: Vec<_> = phases.names().collect();
        assert_eq!(names, [AsciiPhase::NAME]);
        let phase = client.rfx.phase.clone().expect("installed");

        // Off by default: nothing opts in.
        client.rfx_frame();
        assert!(!phases.scene_mode());
        assert!(!phase.with(|p| p.is_active()));

        // ASCII chosen, but no world (the title screen): still nothing.
        client
            .mods
            .settings
            .set_enabled("super_secret_settings", true);
        client
            .mods
            .settings
            .set_choice("super_secret_settings", "shader", ASCII_CHOICE);
        client
            .mods
            .settings
            .set_choice("super_secret_settings", "ascii_colors", 2);
        client.rfx_frame();
        assert!(!phases.scene_mode(), "no world, no effect");
        assert!(
            phase.with(|p| p.needs_font()),
            "the font is read only when needed"
        );

        // In a world: on, styled, with the pack's font.
        let (_to_game, _to_net) = join_world(&mut client);
        client.rfx_frame();
        assert!(phases.scene_mode());
        assert!(phase.with(|p| p.is_active() && !p.needs_font()));
        assert_eq!(
            phase.with(|p| p.style()),
            (AsciiSize::Medium, AsciiColors::Green)
        );

        // Another choice: off, and everything released.
        client
            .mods
            .settings
            .set_choice("super_secret_settings", "shader", 1);
        client.rfx_frame();
        assert!(!phases.scene_mode());
        assert!(phase.with(|p| !p.is_active() && !p.pipelines_built() && p.needs_font()));
    }

    #[test]
    fn old_item_edges_need_the_mod_and_follow_the_option() {
        let mut settings = mc_mods::ModSettings::default();
        assert!(
            !old_item_edges(&settings),
            "1.7 Animations is off by default"
        );
        settings.set_enabled("old_animations", true);
        assert!(
            old_item_edges(&settings),
            "the option defaults on with the mod"
        );
        settings.set_bool("old_animations", "translucent_held", false);
        assert!(!old_item_edges(&settings));
    }
}
