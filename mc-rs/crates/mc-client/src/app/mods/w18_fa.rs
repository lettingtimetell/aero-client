//! Bug report: "symbols in the chat dont render with a background shadow like
//! text".
//!
//! A symbol is any glyph `ascii.png` does not have (hearts, stars, swords, check marks, arrows),
//! drawn from a unicode page. While the unicode font is off, vanilla 1.8.9 draws a symbol's
//! shadow exactly under the symbol (`FontRenderer.renderStringAtPos`, FontRenderer.java:479-494),
//! where the glyph covers it, so none shows. The Chat mod's "Symbol shadows" (on by default; it
//! works with the Chat mod off) keeps it one pixel down and right, like a letter's, everywhere
//! text has a shadow: chat, the Tab list, titles, tooltips, item names, the boss bar, and the
//! mods' text with their shadow on. Off is vanilla. The font half is
//! [`mc_ui::font::FontRenderer::set_symbol_shadows`].
//!
//! Hook: In `build_ui`, once per frame: a `u64` compare and a `bool`
//! compare. A language or Force Unicode Font change can replace the font renderer (`relayout`),
//! so the option is compared with the live renderer every frame rather than pushed once.

use mc_mods::mods::chat::SYMBOL_SHADOWS;
use mc_mods::ModSettings;

use super::super::Client;
use super::x2_chat::CHAT;

/// The state on the client (`Client::w18_fa`).
#[derive(Debug, Default)]
pub(crate) struct FaState {
    /// The settings generation `wanted` was read at (0 = never; the store starts at 1).
    seen: u64,
    /// "Symbol shadows".
    wanted: bool,
}

impl FaState {
    /// The option's value, re-read only when the settings changed.
    fn wanted(&mut self, settings: &ModSettings) -> bool {
        if self.seen != settings.generation() {
            self.seen = settings.generation();
            self.wanted = settings.bool(CHAT, SYMBOL_SHADOWS);
        }
        self.wanted
    }
}

impl Client {
    /// Keep the font's symbol shadows in step with the option.
    pub(crate) fn w18_fa_symbol_shadows(&mut self) {
        let wanted = self.w18_fa.wanted(&self.mods.settings);
        if self.font.symbol_shadows() != wanted {
            self.font.set_symbol_shadows(wanted);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::config::{ClientConfig, LoginMode};
    use mc_ui::font::{FontRenderer, TextMesh};

    /// A client on the real assets. `None` = skip without the vanilla jar (`MC_JARS_DIR`).
    fn client() -> Option<(Client, tempfile::TempDir)> {
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

    /// Where a shadowed heart's two quads go: (shadow pass, main pass) top-left corners.
    fn heart(client: &mut Client) -> ([f32; 2], [f32; 2]) {
        let mut mesh = TextMesh::new();
        client
            .font
            .draw_string(&mut mesh, "\u{2764}", 10.0, 20.0, -1, true);
        assert_eq!(
            mesh.glyphs.len(),
            2,
            "the jar's unicode page 27 has the heart"
        );
        (mesh.glyphs[0].pos[0], mesh.glyphs[1].pos[0])
    }

    #[test]
    fn the_option_reaches_the_font_with_the_chat_mod_off() {
        let Some((mut client, _dir)) = client() else {
            return;
        };
        assert!(
            !client.mods.settings.enabled(CHAT),
            "Chat is off by default"
        );
        assert!(
            client.mods.settings.bool(CHAT, SYMBOL_SHADOWS),
            "on by default"
        );
        assert!(
            !client.font.symbol_shadows(),
            "the font itself starts vanilla"
        );
        assert_eq!(
            heart(&mut client),
            ([10.0, 20.0], [10.0, 20.0]),
            "vanilla: under it"
        );

        client.w18_fa_symbol_shadows();
        assert!(client.font.symbol_shadows());
        assert_eq!(
            heart(&mut client),
            ([11.0, 21.0], [10.0, 20.0]),
            "like a letter's"
        );

        client.mods.settings.set_bool(CHAT, SYMBOL_SHADOWS, false);
        client.w18_fa_symbol_shadows();
        assert!(!client.font.symbol_shadows(), "off is vanilla");
        assert_eq!(heart(&mut client), ([10.0, 20.0], [10.0, 20.0]));
    }

    /// `relayout` replaces the renderer when the unicode font is switched; the next frame's hook
    /// gives the new renderer the option again although no setting changed.
    #[test]
    fn a_rebuilt_font_gets_the_option_again() {
        let Some((mut client, _dir)) = client() else {
            return;
        };
        client.w18_fa_symbol_shadows();
        assert!(client.font.symbol_shadows());
        client.font =
            FontRenderer::load(&client.assets.resources, false, false).expect("rebuilt font");
        assert!(!client.font.symbol_shadows());
        client.w18_fa_symbol_shadows();
        assert!(client.font.symbol_shadows());
    }

    /// The hook runs in `build_ui`, before any of the frame's 2D text is laid out.
    #[test]
    fn build_ui_runs_the_hook() {
        let driver = include_str!("../driver.rs");
        let body = driver
            .split_once("fn build_ui(&mut self, gpu: &Gpu, partial_ticks: f32)")
            .expect("build_ui")
            .1;
        let hook = body
            .find("self.w18_fa_symbol_shadows();")
            .expect("the hook");
        let first_draw = body
            .find("self.draw_ingame_overlay(")
            .expect("the HUD draw");
        assert!(hook < first_draw, "the hook runs before the HUD is drawn");
    }
}
