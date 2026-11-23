//! The game's own options over a replay.
//!
//! The game's general settings (chat and the rest) can be opened while watching a replay.
//! The viewer's **Options** button (a cog next to Hide, or Ctrl+,) opens vanilla's options screen
//! over the replay, exactly the one the pause menu opens in a game: Video settings, Chat
//! settings, Controls, Sound, Skin customisation, Language, Resource packs. The replay keeps
//! playing behind it (or stays paused), so a change shows at once, and Done (or Esc) on the
//! options screen comes back to the replay and its controls. Sub-screens go back to the options
//! screen first, as in vanilla (the client's `OptionsParent` chain, which gained `Replay`).

use mc_ui::screens::options::OptionsScreen;

use super::super::{Client, OptionsParent, Ui};

impl Client {
    /// The viewer's Options button: vanilla's options screen over the replay; Done comes back.
    pub(crate) fn nw_rd_open_replay_options(&mut self) {
        if !self.replay_is_active() || self.export_running() {
            return;
        }
        let in_world = self.session.is_some();
        let difficulty = self.current_server_difficulty();
        // The replay's world is shown behind it, as for the pause menu's options in a game.
        self.options_parents.push(OptionsParent::Replay);
        self.ui = Ui::Options(Box::new(OptionsScreen::new_with_difficulty(
            in_world, difficulty,
        )));
    }
}

/// Through the real client: a real replay file opened by the engine, the options screen
/// over it, a sub-screen and back, and Done back to watching. Needs the vanilla jar (skips without
/// it; set `MC_JARS_DIR`).
#[cfg(test)]
mod tests {
    use std::time::Duration;

    use mc_replay::{ReplayMeta, StreamId, Writer, WriterOptions};

    use super::super::super::{Client, Ui};
    use crate::app::config::{ClientConfig, LoginMode};

    fn offline_client() -> Option<(Client, tempfile::TempDir)> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("skip: no vanilla jar in this checkout (set MC_JARS_DIR)");
            return None;
        }
        let client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        Some((client, dir))
    }

    #[test]
    fn options_open_over_a_replay_and_done_comes_back_to_it() {
        let Some((mut client, dir)) = offline_client() else {
            return;
        };
        let path = dir.path().join("game.aeroreplay");
        let mut meta = ReplayMeta {
            title: "SkyWars".to_owned(),
            ..ReplayMeta::default()
        };
        let mut writer = Writer::create(&path, &meta, WriterOptions::default()).expect("writer");
        for i in 0..10u64 {
            writer.push(i * 100_000, StreamId::Packet, &[0x02, 0, 0]);
        }
        meta.duration = 900_000;
        writer
            .finish(meta)
            .wait(Duration::from_secs(20))
            .expect("the file is written");

        // Not watching: the button does nothing.
        client.nw_rd_open_replay_options();
        assert!(!matches!(client.ui, Ui::Options(_)));

        client.replay_open(&path).expect("the replay opens");
        client.ui = Ui::InGame;
        client.nw_rd_open_replay_options();
        assert!(
            matches!(client.ui, Ui::Options(_)),
            "the options screen is up over the replay"
        );
        assert!(client.replay_is_active(), "and the replay is still open");

        // Chat settings (a sub-screen) and back: the options screen again, as in vanilla.
        client.handle_action(mc_ui::screens::Action::OpenOptionsPage(
            mc_ui::screens::options::OptionsPage::Chat,
        ));
        assert!(matches!(client.ui, Ui::OptionList(_, _)), "chat settings");
        client.go_back();
        assert!(matches!(client.ui, Ui::Options(_)), "back on the options");
        // Done: back to watching, with the replay still open.
        client.go_back();
        assert!(matches!(client.ui, Ui::InGame), "back to the replay");
        assert!(client.replay_is_active());
        assert!(client.options_parents.is_empty(), "the chain is unwound");
    }
}
