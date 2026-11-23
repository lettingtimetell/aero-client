//! Client-lifecycle parity with vanilla 1.8.9.
//!
//! Not a mod: these are vanilla behaviors that the frame-driven screen model did not reach. They
//! live here so the huge shared driver only carries one-line hooks.
//!
//! * **Resize.** `Minecraft.resize` (a window resize or a fullscreen switch) calls
//!   `currentScreen.onResize`, which clears `buttonList` and runs `initGui` again. This client
//!   lays every screen out each frame, so layout needs nothing; the two `initGui` bodies with a
//!   packet side effect are handled here: `GuiStats` and `GuiAchievements` re-send
//!   `C16PacketClientStatus(REQUEST_STATS)`, and `GuiStats` also goes back to loading.
//! * **The command-block editor.** `GuiCommandBlock` holds the live `CommandBlockLogic` by
//!   reference; the editor here is handed that logic before each input pass.

use mc_protocol::play::{c2s, C2SPlay};
use mc_ui::screens::sign::CommandBlockTarget;

use super::super::{Client, Ui};

impl Client {
    /// Mirror the live `CommandBlockLogic` into an open
    /// `GuiCommandBlock` before its input pass (see `CommandBlockScreen::sync_live_logic`): the
    /// input that follows (Done, the O/X toggle) and the next layout pass then read what
    /// vanilla's screen would read from the object it holds.
    ///
    /// * A command block: its tile state, which an `S35` action 2 rewrites in place
    ///   (`TileEntityCommandBlock.readFromNBT`); the editor's own toggle, Done and Cancel write
    ///   `trackOutput` back into it.
    /// * A command minecart: the logic its watcher callbacks maintain (`onDataWatcherUpdate`
    ///   23/24). Its `trackOutput` is written only by this editor, so the screen's own bit is
    ///   the live one.
    ///
    /// A tile or entity that is gone leaves the editor on the logic it last saw, like Java's
    /// orphaned object. Any other screen costs one `match`; this one a lookup and a few string
    /// comparisons, allocating only when a value changed.
    pub(crate) fn w18_pb_sync_command_block_logic(&mut self) {
        let Ui::CommandBlock(screen) = &mut self.ui else {
            return;
        };
        let Some(session) = &self.session else {
            return;
        };
        let state = session.state();
        let language = &self.assets.language;
        match screen.target() {
            CommandBlockTarget::Block { position } => {
                if let Some(logic) = state
                    .world
                    .tile_entity(position)
                    .and_then(mc_world::TileEntity::command_block)
                {
                    screen.sync_live_logic(
                        &logic.command,
                        logic.track_output,
                        logic.last_output.as_deref(),
                        language,
                    );
                }
            }
            CommandBlockTarget::Minecart { entity_id } => {
                if let Some(logic) = state
                    .entities
                    .get(entity_id)
                    .and_then(|tracked| tracked.command_minecart_callback_state())
                {
                    let track_output = screen.track_output();
                    screen.sync_live_logic(
                        &logic.command,
                        track_output,
                        Some(&logic.last_output_json),
                        language,
                    );
                }
            }
        }
    }

    /// java: `Minecraft.resize` -> `GuiScreen.onResize` ->
    /// `setWorldAndResolution` -> `initGui`, for the screens whose `initGui` sends a packet.
    ///
    /// * `GuiStats.initGui` (`:55-60`): `doesGuiPauseGame = true`, then `C16 REQUEST_STATS`.
    /// * `GuiAchievements.initGui` (`:63-68`): `C16 REQUEST_STATS` first, then its Done button;
    ///   `loadingAchievements` is left as it was, so the map stays on screen.
    ///
    /// Called by the shell's resize callback, which fires once per real size change, as
    /// `Minecraft.checkWindowResize` does once per frame when `Display` reports a new size. Both
    /// screens are only reachable from the pause menu, so a session exists whenever they are up;
    /// without one there is nothing to send and nothing changes. Every other screen: one `match`.
    pub(crate) fn w18_pb_screen_resized(&mut self) {
        if self.session.is_none() {
            return;
        }
        let request_stats = match &mut self.ui {
            Ui::Stats(screen) => {
                screen.restart_loading();
                true
            }
            Ui::Achievements(_) => true,
            _ => false,
        };
        if request_stats {
            if let Some(session) = &mut self.session {
                session.send(C2SPlay::ClientStatus(c2s::ClientStatus {
                    action: c2s::ClientStatusAction::RequestStats,
                }));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::config::{ClientConfig, LoginMode};
    use crate::session::{JoinSettings, Session};
    use crate::sync::ClientEvent;
    use mc_ui::screens::Action;

    fn offline_client() -> Option<(Client, tempfile::TempDir)> {
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

    /// A socketless session; keep the receiver alive, it is where sent packets go.
    fn attach_session(
        client: &mut Client,
    ) -> (
        tokio::sync::mpsc::Sender<crate::net::ToGame>,
        tokio::sync::mpsc::UnboundedReceiver<crate::net::ToNet>,
    ) {
        let (handle, to_game, to_net) = crate::net::channel::NetHandle::new_pair();
        client.session = Some(Session::new(handle, JoinSettings::default()));
        (to_game, to_net)
    }

    fn stats_requests(
        to_net: &mut tokio::sync::mpsc::UnboundedReceiver<crate::net::ToNet>,
    ) -> usize {
        let mut count = 0;
        while let Ok(message) = to_net.try_recv() {
            if let crate::net::ToNet::Send(C2SPlay::ClientStatus(c2s::ClientStatus {
                action: c2s::ClientStatusAction::RequestStats,
            })) = message
            {
                count += 1;
            }
        }
        count
    }

    fn statistics(value: i32) -> ClientEvent {
        ClientEvent::Statistics(vec![mc_protocol::play::s2c::StatisticEntry {
            stat_id: "stat.mineBlock.minecraft.stone".to_owned(),
            value,
        }])
    }

    /// java: `GuiStats.initGui` runs again on a resize: one more `C16 REQUEST_STATS`, the
    /// loading text again, and the reply rebuilds the lists on the General tab.
    #[test]
    fn a_resize_over_the_statistics_screen_requests_them_again_and_reloads() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let (_to_game, mut to_net) = attach_session(&mut client);
        client.handle_action(Action::OpenStats);
        assert_eq!(stats_requests(&mut to_net), 1, "initGui on open");
        client.react(&statistics(3));
        let Ui::Stats(screen) = &mut client.ui else {
            panic!("OpenStats opens GuiStats");
        };
        assert!(!screen.loading);
        screen.tab = mc_ui::screens::stats::StatsTab::Blocks;
        assert_eq!(screen.current_rows().len(), 1);

        client.w18_pb_screen_resized();
        assert_eq!(stats_requests(&mut to_net), 1, "initGui on resize");
        let Ui::Stats(screen) = &client.ui else {
            panic!("a resize keeps the same screen");
        };
        assert!(screen.loading, "back to Downloading statistics");
        assert_eq!(screen.tab, mc_ui::screens::stats::StatsTab::General);

        client.react(&statistics(5));
        let Ui::Stats(screen) = &client.ui else {
            unreachable!()
        };
        assert!(
            !screen.loading,
            "the reply's doneLoading rebuilds the lists"
        );
        assert_eq!(screen.tab, mc_ui::screens::stats::StatsTab::General);
    }

    /// java: `GuiAchievements.initGui` re-sends `C16` on a resize but leaves
    /// `loadingAchievements` alone, so the map stays up. Other screens send nothing.
    #[test]
    fn a_resize_over_achievements_requests_stats_and_other_screens_send_nothing() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let (_to_game, mut to_net) = attach_session(&mut client);
        client.handle_action(Action::OpenAchievements);
        assert_eq!(stats_requests(&mut to_net), 1);
        client.react(&statistics(1));
        client.w18_pb_screen_resized();
        assert_eq!(stats_requests(&mut to_net), 1);
        let Ui::Achievements(screen) = &client.ui else {
            panic!("a resize keeps the achievements screen");
        };
        assert!(
            !screen.loading,
            "loadingAchievements is not reset by initGui"
        );

        // Controls: the pause menu and the in-game HUD have no such initGui side effect.
        client.ui = Ui::Pause(Box::default());
        client.w18_pb_screen_resized();
        client.ui = Ui::InGame;
        client.w18_pb_screen_resized();
        assert_eq!(stats_requests(&mut to_net), 0);
    }

    fn command_block_nbt(
        position: mc_math::BlockPos,
        command: &str,
        track_output: bool,
        last_output: Option<&str>,
    ) -> mc_nbt::NbtCompound {
        use mc_nbt::Nbt;
        let mut nbt = mc_nbt::NbtCompound::new();
        nbt.insert("id", Nbt::String("Control".to_owned()));
        nbt.insert("x", Nbt::Int(position.x));
        nbt.insert("y", Nbt::Int(position.y));
        nbt.insert("z", Nbt::Int(position.z));
        nbt.insert("Command", Nbt::String(command.to_owned()));
        nbt.insert("SuccessCount", Nbt::Int(0));
        nbt.insert("TrackOutput", Nbt::Byte(i8::from(track_output)));
        if let Some(output) = last_output {
            nbt.insert("LastOutput", Nbt::String(output.to_owned()));
        }
        nbt
    }

    /// Lay the open screen out at a window size, as the frame's UI pass does.
    fn layout(client: &mut Client, width: u32, height: u32) {
        let ctx = mc_ui::screen_stack::ScreenCtx::new(client.resolution(width, height), 0.0);
        if let Ui::CommandBlock(screen) = &mut client.ui {
            screen.init(&ctx, &client.assets.language, &client.font);
        }
    }

    fn editor(client: &Client) -> &mc_ui::screens::sign::CommandBlockScreen {
        match &client.ui {
            Ui::CommandBlock(screen) => screen,
            _ => panic!("the command block editor is open"),
        }
    }

    /// java: `GuiCommandBlock` holds the block's live logic. An
    /// S35 that rewrites it while the editor is open shows only after a resize re-runs
    /// `initGui`, and Done carries the logic's live `trackOutput`.
    #[test]
    fn an_s35_under_the_command_block_editor_shows_after_a_resize_and_rides_on_done() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let (_to_game, _to_net) = attach_session(&mut client);
        let position = mc_math::BlockPos::new(3, 65, 4);
        let mut world = mc_render::terrain::fixture::blank_world(1);
        mc_render::terrain::fixture::place(&mut world, position, "minecraft:command_block");
        client.session.as_mut().expect("session").state_mut().world = world;
        let old = command_block_nbt(position, "say old", true, Some(r#"{"text":"old"}"#));
        client.react_to_tile_entity_update(position, 2, Some(&old));
        let state = client
            .session
            .as_ref()
            .and_then(|session| {
                session
                    .state()
                    .world
                    .tile_entity(position)
                    .and_then(mc_world::TileEntity::command_block)
                    .cloned()
            })
            .expect("the S35 built the command block's logic");
        client.ui = Ui::CommandBlock(Box::new(
            mc_ui::screens::sign::CommandBlockScreen::from_state(
                CommandBlockTarget::Block { position },
                &state,
                &client.assets.language,
            ),
        ));
        client.w18_pb_sync_command_block_logic();
        layout(&mut client, 854, 480);
        assert_eq!(editor(&client).command(), "say old");
        assert_eq!(editor(&client).previous_output_text_field().text(), "old");

        let new = command_block_nbt(position, "say new", true, Some(r#"{"text":"new"}"#));
        client.react_to_tile_entity_update(position, 2, Some(&new));
        client.w18_pb_sync_command_block_logic();
        layout(&mut client, 854, 480);
        assert_eq!(
            editor(&client).command(),
            "say old",
            "no initGui, no change"
        );
        assert_eq!(editor(&client).previous_output_text_field().text(), "old");
        layout(&mut client, 640, 360);
        assert_eq!(editor(&client).command(), "say new", "the resize's initGui");
        assert_eq!(editor(&client).previous_output_text_field().text(), "new");

        // The server turns output tracking off: still "O" on screen, but Done sends `false`.
        let off = command_block_nbt(position, "say new", false, None);
        client.react_to_tile_entity_update(position, 2, Some(&off));
        client.w18_pb_sync_command_block_logic();
        layout(&mut client, 640, 360);
        assert_eq!(editor(&client).track_output_button().text, "O");
        let font = &client.font;
        let clipboard = &mut client.clipboard;
        let Ui::CommandBlock(screen) = &mut client.ui else {
            unreachable!()
        };
        screen.key_typed(
            None,
            mc_ui::widget::keys::ENTER,
            mc_ui::widget::keys::Modifiers::NONE,
            clipboard,
            font,
        );
        assert_eq!(
            screen.take_actions(),
            [
                mc_ui::screens::sign::CommandBlockAction::Send {
                    target: CommandBlockTarget::Block { position },
                    command: "say new".to_owned(),
                    track_output: false,
                },
                mc_ui::screens::sign::CommandBlockAction::ClearLastOutput,
                mc_ui::screens::sign::CommandBlockAction::Close,
            ]
        );
    }
}
