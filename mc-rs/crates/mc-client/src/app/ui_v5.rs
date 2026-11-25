//! The client's 1.7.10 screens and HUD: which container screen a window opens, the rules the
//! session's own inventory window predicts with, and (further below) the chat, sign, book,
//! tab-list, sidebar, resource-pack and camera behaviour of a 1.7.10 session.
//!
//! A child of `app::driver` (declared there), so it reaches `Client`'s private fields. Each hook
//! is one line at its 1.8.9 site and leaves a 1.8.9 session exactly as before; the 1.7.10 rules
//! themselves live in `mc-ui` (`screens/container/v5.rs` and its siblings) where they are
//! UI-only.

use mc_protocol::play::s2c;
use mc_ui::screen_stack::ScreenCtx;
use mc_ui::screens::chat::ChatLayout;
use mc_ui::screens::container::ContainerScreen;
use mc_ui::screens::hud::Sidebar;
use mc_ui::screens::tab_list::{TabCellV5, TabGridV5};
use mc_ui::widget::DrawList;
use mc_version::GameVersion;

use mc_ui::screens::multiplayer::ResourceMode;

use super::{AsyncMessage, Client, PendingServerResourcePack, Ui};
use crate::app::assets;
use crate::session::Session;
use crate::sync::scoreboard;

/// What a 1.7.10 session's smooth camera filters hold between the tick's and the frame's
/// smooth-camera steps ([`Client::smooth_camera_kept_v5`]).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) enum SmoothCameraV5 {
    /// Off, kept as the smooth camera setting left them (at rest before it was ever on).
    #[default]
    Kept,
    /// On: through the setting (with or without the Zoom mod), or the Zoom mod alone.
    On {
        /// The Zoom mod alone: the setting is off.
        zoom_only: bool,
    },
    /// Off after the Zoom mod alone: cleared by 1.8.9's rules.
    Cleared,
}

/// The local player's latency from the session's player list, for the Ping HUD. 1.7.10's list
/// is keyed by name (java17 `NetHandlerPlayClient.handlePlayerListItem` keeps `GuiPlayerInfo`s
/// by `S38PacketPlayerListItem`'s name), so the player is found by their username; 1.8.9's by
/// UUID.
pub(super) fn session_latency_ms(session: &Session) -> Option<i32> {
    match session.version() {
        GameVersion::V1_8_9 => session.state().latency_ms(),
        GameVersion::V1_7_10 => {
            let state = session.state();
            state
                .players_v5
                .get(&state.local.username)
                .map(|info| info.ping_ms)
        }
    }
}

impl Client {
    /// The screen a server's window opens: 1.7.10's shapes and rules in a 1.7.10 session
    /// ([`ContainerScreen::for_open_window_v5`]), else the 1.8.9 screen.
    pub(super) fn container_screen_for_open_window(
        &self,
        packet: &s2c::OpenWindow,
        title: String,
    ) -> Option<ContainerScreen> {
        if self.plays_1_7_10() {
            ContainerScreen::for_open_window_v5(packet, title)
        } else {
            ContainerScreen::for_open_window(packet, title)
        }
    }

    /// Where `handleJoinGame` builds the new player's container: window 0 (the player's
    /// inventory) predicts clicks with the session's version's rules, and the 1.7.10 tab list's
    /// cells start over.
    pub(super) fn join_game_ui_v5(&mut self) {
        let version = self
            .session
            .as_ref()
            .map_or(GameVersion::V1_8_9, |session| session.version());
        self.player_container.set_game_version(version);
        self.tab_list.clear_v5();
    }

    /// java17 `GuiIngame.renderGameOverlay:388-483`: the 1.7.10 player list, while the list
    /// key is held in a 1.7.10 session (`true`: drawn, the 1.8.9 list is not). The cells are
    /// rebuilt only when the player list, the scoreboard or the maximum player count changed.
    pub(super) fn draw_tab_list_v5(&mut self, d: &mut DrawList, ctx: &ScreenCtx) -> bool {
        if !self.plays_1_7_10() {
            return false;
        }
        let Some(session) = self.session.as_mut() else {
            return true;
        };
        let state = session.state_mut();
        let max_players = state.local.max_players;
        let key = (
            state.players_v5.revision(),
            state.scoreboard.revision(),
            max_players,
        );
        let font = &self.font;
        self.tab_list.refresh_v5(key, || {
            let grid = TabGridV5::for_max_players(max_players);
            let objective = state
                .scoreboard
                .objective_in_display_slot(scoreboard::SLOT_LIST)
                .map(|objective| objective.name.clone());
            let players: Vec<(String, i32)> = state
                .players_v5
                .iter()
                .take(usize::from(max_players))
                .map(|info| (info.name.clone(), info.ping_ms))
                .collect();
            players
                .into_iter()
                .map(|(player, ping_ms)| {
                    let name = state.scoreboard.format_player_name(&player);
                    // `getValueFromObjective` (it creates a zero score) runs only when the score
                    // has room, as in java17.
                    let score = objective
                        .as_deref()
                        .filter(|_| grid.has_score_room(font.string_width(&name)))
                        .and_then(|objective| {
                            state.scoreboard.value_from_objective(&player, objective)
                        })
                        .map(|points| format!("\u{a7}e{points}"));
                    TabCellV5 {
                        name,
                        score,
                        ping_ms,
                    }
                })
                .collect()
        });
        self.tab_list
            .draw_v5(d, &mut self.font, ctx.width, max_players);
        true
    }

    /// java17 `GuiChat.mouseClicked`'s component half, for a press on the chat (or the bed
    /// screen's chat) in a 1.7.10 session: `true` when the press was a component click, which is
    /// then done (see `mc_ui::screens::chat`'s `component_clicked_v5`). A 1.8.9 session, another
    /// button or another screen: `false`, nothing done.
    pub(super) fn chat_click_v5(&mut self, button: i32, ctx: &ScreenCtx, shift: bool) -> bool {
        if button != 0 || !self.plays_1_7_10() || !matches!(self.ui, Ui::Chat(_) | Ui::Sleep(_)) {
            return false;
        }
        let (x, y) = self.cursor;
        let layout = ChatLayout::from_settings(&self.settings);
        let piece = self
            .chat
            .clicked_piece_v5(x, y, ctx, &layout, &self.font)
            .map(|(text, event)| (text.to_owned(), event.clone()));
        let piece = piece.as_ref().map(|(text, event)| (text.as_str(), event));
        let chat_links = self.settings.chat_links;
        let action = match &mut self.ui {
            Ui::Chat(screen) => {
                screen.init(ctx);
                screen.component_clicked_v5(piece, shift, chat_links, &self.font)
            }
            Ui::Sleep(screen) => {
                screen.init(ctx, &self.assets.language);
                screen
                    .chat
                    .component_clicked_v5(piece, shift, chat_links, &self.font)
            }
            _ => None,
        };
        let Some(action) = action else {
            return false;
        };
        self.handle_action(action);
        true
    }

    /// The press's chat style for the 1.8.9 component rules: none in a 1.7.10 session, whose
    /// rules [`Client::chat_click_v5`] already applied.
    pub(super) fn chat_style_for_v47_rules<T>(&self, style: Option<T>) -> Option<T> {
        style.filter(|_| !self.plays_1_7_10())
    }

    /// java17 `GuiChat.submitChatMessage`: a command run from a chat component goes into the
    /// sent history first, in a 1.7.10 session (1.8.9 runs it without recording it).
    pub(super) fn record_run_command_v5(&mut self, command: &str) {
        if self.plays_1_7_10() {
            self.chat.add_sent(command);
        }
    }

    /// A screen about to open, with a 1.7.10 session's rules where its editor has them: the
    /// sign's 15-character lines, the book's 13-line plain pages, the command block's missing
    /// track-output toggle. Every other screen, and every screen of a 1.8.9 session, unchanged.
    pub(super) fn screen_for_session_version(&self, mut screen: Option<Ui>) -> Option<Ui> {
        if !self.plays_1_7_10() {
            return screen;
        }
        let version = GameVersion::V1_7_10;
        match screen.as_mut() {
            Some(Ui::EditSign(editor)) => editor.set_game_version(version),
            Some(Ui::Book(book)) => book.set_game_version(version),
            Some(Ui::CommandBlock(editor)) => editor.set_game_version(version),
            _ => {}
        }
        screen
    }

    /// java17 `GuiIngame.renderGameOverlay:372` + `renderScoreboard`: the sidebar slot's
    /// objective (1.7.10 has no team-colour slots), shown only with at most 15 scores, `#`
    /// names included ([`Sidebar::from_scores_v5`]). Built when the scoreboard changes, like the
    /// 1.8.9 snapshot.
    pub(super) fn sidebar_snapshot_v5(&self) -> Option<Sidebar> {
        let board = &self.session.as_ref()?.state().scoreboard;
        let objective = board.objective_in_display_slot(scoreboard::SLOT_SIDEBAR)?;
        let scores: Vec<(String, i32)> = board
            .sorted_scores(&objective.name)
            .into_iter()
            .map(|(name, points)| (name.to_owned(), points))
            .collect();
        Sidebar::from_scores_v5(objective.display_name.clone(), &scores, |name| {
            board.format_player_name(name)
        })
    }

    /// java17 `NetHandlerPlayClient.handleCustomPayload`'s `MC|RPack` arm
    /// (`client/network/NetHandlerPlayClient.java:1436-1466`): a server entry set to enabled
    /// downloads the pack; one set to prompt (or no entry) asks; one set to disabled ignores it.
    /// Nothing is ever answered to the server (1.7.10 has no resource-pack status packet).
    /// `false` in a 1.8.9 session, which handles the request itself.
    pub(super) fn react_to_resource_pack_v5(&mut self, url: &str) -> bool {
        if !self.plays_1_7_10() {
            return false;
        }
        match self.current_server_resource_mode {
            ResourceMode::Enabled => self.obtain_resource_pack_v5(url),
            ResourceMode::Disabled => {}
            ResourceMode::Prompt => {
                // A second request replaces the visible prompt (`displayGuiScreen`), orphaning
                // the first one's answer, as in 1.8.9.
                let parent = match std::mem::replace(&mut self.ui, Ui::Loading) {
                    Ui::ConfirmServerResourcePack(_, parent, _) => *parent,
                    other => other,
                };
                let pending = PendingServerResourcePack {
                    url: url.to_owned(),
                    hash: String::new(),
                };
                self.show_server_resource_pack_prompt(pending, parent);
            }
        }
        true
    }

    /// The 1.7.10 prompt's `confirmClicked`: **either** answer sets the server entry to enabled
    /// and saves the server list; only "yes" downloads this pack; then the screen closes,
    /// closing the download's progress screen too (the download goes on). `false` in a 1.8.9
    /// session.
    pub(super) fn finish_resource_pack_choice_v5(&mut self, confirmed: bool) -> bool {
        if !self.plays_1_7_10() {
            return false;
        }
        let current = std::mem::replace(&mut self.ui, Ui::Loading);
        let Ui::ConfirmServerResourcePack(_, _, pending) = current else {
            unreachable!("resource-pack choice is only finished from its own screen")
        };
        self.current_server_resource_mode = ResourceMode::Enabled;
        self.persist_current_server_resource_mode();
        if confirmed {
            self.obtain_resource_pack_v5(&pending.url);
        }
        self.drop_server_resource_pack_progress();
        self.show_screen(None);
        true
    }

    /// 1.7.10 sends no resource-pack status: `true` when this session must not.
    pub(super) fn suppresses_resource_pack_status_v5(&self) -> bool {
        self.plays_1_7_10()
    }

    /// Whether the smooth camera's filters are to be left exactly as they are, at the tick's or
    /// the frame's smooth-camera step (`smooth`: on for this step). java17
    /// `EntityRenderer.updateRenderer` (`:235-244`) filters only while smooth camera is on and
    /// has no `else`, and `updateCameraAndRender` (`:955-982`) passes the mouse delta through
    /// while it is off without clearing the accumulated deltas; nothing resets the filters
    /// (java17 `MouseFilter` has no reset). So in a 1.7.10 session, after the setting is turned
    /// off and on again, the old filter state keeps turning the camera. 1.8.9 clears the filters
    /// every tick and the accumulators every frame while it is off.
    ///
    /// That rule is the setting's only. The Zoom mod's "smooth camera while zoomed" is not
    /// vanilla (1.7.10 has no zoom) and behaves exactly as in a 1.8.9 session: what it leaves
    /// is cleared, and a zoom that starts while the setting's state is kept starts from 1.8.9's
    /// rest state. So a zoom never turns the camera, and the rotation the server is sent, on its
    /// own. Always `false` in a 1.8.9 session.
    pub(super) fn smooth_camera_kept_v5(&mut self, smooth: bool) -> bool {
        if !self.plays_1_7_10() {
            return false;
        }
        let zoom_only = !self.settings.smooth_camera;
        let (next, kept) = match (self.smooth_camera_v5, smooth) {
            (SmoothCameraV5::Kept, true) if zoom_only => {
                // 1.8.9's two "off" arms, once: the filters and their output, the accumulators.
                self.smooth_camera.tick(self.look_sensitivity(), false);
                let _ = self.smooth_camera.apply(0.0, 0.0, 0.0, false);
                (SmoothCameraV5::On { zoom_only }, false)
            }
            (_, true) => (SmoothCameraV5::On { zoom_only }, false),
            (SmoothCameraV5::On { zoom_only: true } | SmoothCameraV5::Cleared, false) => {
                (SmoothCameraV5::Cleared, false)
            }
            (SmoothCameraV5::On { zoom_only: false } | SmoothCameraV5::Kept, false) => {
                (SmoothCameraV5::Kept, true)
            }
        };
        self.smooth_camera_v5 = next;
        kept
    }

    /// java17 `ResourcePackRepository.obtainResourcePack` (`client/resources/
    /// ResourcePackRepository.java:159-203`): only a URL whose file name (before any `?`) ends in
    /// `.zip` is fetched. The current server pack is dropped without a reload (`func_148529_f`,
    /// `:205-209`, only forgets it, so it stays drawn until the new one is installed), and the
    /// pack is requested with no hash and the `X-Minecraft-Version: 1.7.10` header, then
    /// installed with one reload. The download itself is 1.8.9's (its `legacy` cache slot),
    /// which reports back through [`AsyncMessage::ResourcePackDownloaded`] like a 1.8.9 one.
    ///
    /// One client-local difference: java17 `HttpUtil.downloadResourcePack` (`util/HttpUtil.java:
    /// 172-186`) keeps a file of the same name whose length equals the response's
    /// Content-Length instead of reading the body; this client reads the body every time.
    fn obtain_resource_pack_v5(&mut self, url: &str) {
        let name = url.rsplit('/').next().unwrap_or(url);
        let name = name.split('?').next().unwrap_or(name);
        if !name.ends_with(".zip") {
            return;
        }
        let replacing_progress = self.server_pack_progress.is_some();
        self.cancel_server_resource_pack_download();
        if replacing_progress {
            self.finish_server_resource_pack_progress();
        }
        let _ = assets::clear_active_server_resource_pack(&self.config);
        let hash = String::new();
        self.pending_server_resource_pack_hash = Some(hash.clone());
        self.begin_server_resource_pack_progress();
        let config = self.config.clone();
        let url = url.to_owned();
        let headers = assets::ServerResourcePackHeaders {
            username: self.auth.as_ref().map(|session| session.username.clone()),
            uuid: self.auth.as_ref().map(|session| session.uuid_undashed()),
            version: self.current_game_version(),
        };
        let tx = self.async_tx.clone();
        let task = self.runtime.handle().spawn(async move {
            let result = assets::download_server_resource_pack_with_proxy(
                &config,
                &url,
                &hash,
                &headers,
                config.proxy.as_ref(),
            )
            .await;
            let _ = tx.send(AsyncMessage::ResourcePackDownloaded {
                hash,
                result: Box::new(result),
            });
        });
        self.pending_server_resource_pack_abort = Some(task.abort_handle());
    }
}

#[cfg(test)]
#[path = "ui_v5_tests.rs"]
mod tests;
