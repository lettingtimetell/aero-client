//! The menu's Interface window: the client half of `boss_bar`, `scoreboard`, `chat`
//! and `centered_inventory`.
//!
//! Everything reads one generation-gated snapshot of the four mods ([`M2State`]), so with all
//! four off every hook below is a `u64` compare plus a few `bool` tests, and every vanilla path
//! runs unchanged:
//!
//! * [`Client::m2_sync`], the snapshot refresh, which pushes the chat options
//!   into the log when they change.
//! * [`Client::m2_frame`], sync, then the centred layout of an open
//!   inventory.
//! * [`Client::m2_before_vanilla_hud`] detaches the vanilla boss bar
//!   and consumes its frame.
//! * [`M2State::sidebar_detached`] gates the vanilla sidebar draw.
//! * [`M2State::fill`], boss and sidebar for the two HUD elements.
//! * Top of `mouse_clicked`: [`Client::m2_chat_copy_click`], right-click copies a chat line.
//!
//! The boss bar's 100-frame lifetime keeps counting exactly as vanilla's draw counts it
//! (`Hud::consume_boss_frame`), whether the element shows, hides or moves the bar.

use mc_mods::ModSettings;
use mc_ui::font::component::{ChatComponent, ChatStyle};
use mc_ui::font::TextFormat;
use mc_ui::modhud::HudData;
use mc_ui::screen_stack::ScreenCtx;
use mc_ui::screens::chat::{ChatLayout, ChatMods};
use mc_ui::screens::hud::Hud;
use mc_ui::widget::Clipboard as _;

use super::super::{Client, SidebarCache, Ui};

/// The chat id of the "Copied to clipboard" notice: non-zero, so each copy replaces the last
/// notice in place (vanilla's `deleteChatLine` rule) and it is never stacked or stamped.
const COPY_NOTICE_ID: i32 = 0x4D32_C0B1;

/// The state on the client (`Client::m2`).
#[derive(Debug, Default)]
pub(crate) struct M2State {
    /// The settings generation the snapshot was taken at; 0 = never (the store starts at 1).
    seen: u64,
    boss_bar: bool,
    scoreboard: bool,
    /// `chat` is enabled and its "Right-click to copy" is on.
    chat_copy: bool,
    /// The options the chat log should hold: vanilla's while the mod is off.
    chat_mods: ChatMods,
    centered_inventory: bool,
    inventory_show_effects: bool,
    /// The centred layout was pushed to an open inventory, so switching the mod off restores
    /// vanilla's once.
    inventory_applied: bool,
    /// This frame's `Hud::consume_boss_frame` result while the boss bar is detached.
    boss_frame: bool,
    /// The sidebar cache `(revision, username)` the HUD data was last filled from, `Some(None)`
    /// for "no cache".
    sidebar_seen: Option<Option<(u64, String)>>,
}

impl M2State {
    /// Refresh the snapshot when the settings generation moved. Returns whether it did.
    fn sync(&mut self, settings: &ModSettings) -> bool {
        if self.seen == settings.generation() {
            return false;
        }
        self.seen = settings.generation();
        self.boss_bar = settings.enabled("boss_bar");
        self.scoreboard = settings.enabled("scoreboard");
        let chat = settings.enabled("chat");
        self.chat_copy = chat && settings.bool("chat", "copy");
        self.chat_mods = if chat {
            ChatMods {
                stack: settings.bool("chat", "stack"),
                timestamps: u8::try_from(settings.choice("chat", "timestamps")).unwrap_or(0),
                background: settings.f32("chat", "background_opacity") / 100.0,
                shadow: settings.bool("chat", "text_shadow"),
                raise: if settings.bool("chat", "raise") {
                    ChatMods::RAISE
                } else {
                    0
                },
            }
        } else {
            ChatMods::default()
        };
        self.centered_inventory = settings.enabled("centered_inventory");
        self.inventory_show_effects = settings.bool("centered_inventory", "show_effects");
        true
    }

    /// The `scoreboard` mod draws the sidebar, so the vanilla draw is skipped.
    pub(crate) fn sidebar_detached(&self) -> bool {
        self.scoreboard
    }

    /// `(centered, show_effects)`: the layout an open survival or creative inventory is given
    /// (`set_effect_layout`), vanilla's `(false, true)` while `centered_inventory` is off. The
    /// replay recorder stores it and the replay viewer lays recorded inventories out with it.
    pub(crate) fn inventory_layout(&self) -> (bool, bool) {
        if self.centered_inventory {
            (true, self.inventory_show_effects)
        } else {
            (false, true)
        }
    }

    /// In `fill_hud_data`: the boss bar (name copied only when it changed) and
    /// the sidebar (cloned only when the client's sidebar cache was rebuilt). Takes the client's
    /// fields one by one, so it composes with the other lanes' fills of the same `data`.
    pub(super) fn fill(
        &mut self,
        data: &mut HudData,
        hud: &Hud,
        sidebar_cache: Option<&SidebarCache>,
    ) {
        if self.boss_bar {
            let name = hud.boss_name().unwrap_or("");
            data.boss.shown = self.boss_frame;
            data.boss.health_scale = hud.boss_health_scale();
            if data.boss.name != name {
                data.boss.name.clear();
                data.boss.name.push_str(name);
            }
        }
        if self.scoreboard {
            let unchanged = match (&self.sidebar_seen, sidebar_cache) {
                (Some(Some((revision, username))), Some(cache)) => {
                    *revision == cache.revision && *username == cache.username
                }
                (Some(None), None) => true,
                _ => false,
            };
            if !unchanged {
                data.sidebar = sidebar_cache.and_then(|cache| cache.sidebar.clone());
                data.sidebar_revision = data.sidebar_revision.wrapping_add(1);
                self.sidebar_seen =
                    Some(sidebar_cache.map(|cache| (cache.revision, cache.username.clone())));
            }
        }
    }
}

impl Client {
    /// And the start of every other M2 hook: one `u64` compare unless the
    /// settings changed, when the chat options are pushed into the log.
    pub(crate) fn m2_sync(&mut self) {
        if self.m2.sync(&self.mods.settings) {
            self.chat.set_mods(self.m2.chat_mods);
        }
    }

    /// Sync, then keep an open survival or creative inventory centred while
    /// `centered_inventory` is on. The screens apply it to every later effect update and
    /// re-init themselves, so this only has to reach a newly opened screen before its draw.
    pub(crate) fn m2_frame(&mut self) {
        self.m2_sync();
        let on = self.m2.centered_inventory;
        if !on && !self.m2.inventory_applied {
            return;
        }
        let (centered, show_effects) = self.m2.inventory_layout();
        match &mut self.ui {
            Ui::Container(screen) => screen.set_effect_layout(centered, show_effects),
            Ui::Creative(screen) => screen.set_effect_layout(centered, show_effects),
            _ => {}
        }
        self.m2.inventory_applied = on;
    }

    /// Just before `Hud::draw_with_spectator_and_crosshair`: while
    /// `boss_bar` is on the vanilla bar is detached and this consumes its rendered frame, the
    /// countdown vanilla's draw would have run.
    pub(crate) fn m2_before_vanilla_hud(&mut self) {
        self.m2_sync();
        let detached = self.m2.boss_bar;
        self.hud.set_boss_bar_detached(detached);
        if detached {
            self.m2.boss_frame = self.hud.consume_boss_frame();
        }
    }

    /// Top of `mouse_clicked`: with chat open (or the bed screen's chat), a right-click on a
    /// line copies its text, without formatting codes, timestamp or stack counter, and posts a
    /// grey notice. Vanilla's chat ignores button 1, so nothing else reacts to the press.
    pub(crate) fn m2_chat_copy_click(&mut self, button: i32, ctx: &ScreenCtx) {
        if button != 1 || !self.m2.chat_copy || !matches!(self.ui, Ui::Chat(_) | Ui::Sleep(_)) {
            return;
        }
        let layout = ChatLayout::from_settings(&self.settings);
        let (x, y) = self.cursor;
        let Some(text) = self.chat.copy_text_at(x, y, ctx, &layout) else {
            return;
        };
        self.clipboard.set(&text);
        let notice = ChatComponent::text("Copied to clipboard").with_style(ChatStyle {
            color: Some(TextFormat::Gray),
            ..ChatStyle::default()
        });
        self.chat.add_message(
            notice,
            COPY_NOTICE_ID,
            self.tick as i32,
            &layout,
            &self.font,
            &self.assets.language,
            true,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::config::{ClientConfig, LoginMode};
    use mc_ui::screens::container::ContainerScreen;

    /// A real offline client over a temporary game dir (skips without the vanilla jar).
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

    #[test]
    fn m2_snapshot_is_vanilla_while_off_and_follows_the_options() {
        let mut settings = ModSettings::new();
        let mut state = M2State::default();
        assert!(state.sync(&settings));
        assert!(!state.sync(&settings), "same generation: no work");
        assert_eq!(state.chat_mods, ChatMods::default());
        assert!(!state.boss_bar && !state.sidebar_detached() && !state.chat_copy);

        settings.set_enabled("chat", true);
        state.sync(&settings);
        assert_eq!(
            state.chat_mods,
            ChatMods {
                stack: true,
                ..ChatMods::default()
            },
            "enabled at its defaults: only stacking differs from vanilla"
        );
        assert!(state.chat_copy);
        settings.set_choice("chat", "timestamps", 2);
        settings.set_f32("chat", "background_opacity", 40.0);
        settings.set_bool("chat", "text_shadow", false);
        settings.set_bool("chat", "raise", true);
        state.sync(&settings);
        assert_eq!(
            state.chat_mods,
            ChatMods {
                stack: true,
                timestamps: 2,
                background: 0.4,
                shadow: false,
                raise: ChatMods::RAISE,
            }
        );
        settings.set_enabled("chat", false);
        state.sync(&settings);
        assert_eq!(state.chat_mods, ChatMods::default());
        assert!(!state.chat_copy);

        settings.set_enabled("scoreboard", true);
        settings.set_enabled("boss_bar", true);
        state.sync(&settings);
        assert!(state.sidebar_detached() && state.boss_bar);
    }

    #[test]
    fn hooks_drive_the_chat_log_boss_lifetime_and_inventory_layout() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        // Off: nothing changes.
        client.m2_sync();
        assert_eq!(client.chat.mods(), ChatMods::default());
        client.hud.set_boss_status(50.0, 100.0, "Wither", true);
        client.m2_before_vanilla_hud();
        assert!(!client.hud.boss_bar_detached());
        assert_eq!(
            client.hud.boss_status_bar_time(),
            100,
            "vanilla's draw counts, not us"
        );

        // Chat on: the log gets the options at the next sync.
        client.mods.settings.set_enabled("chat", true);
        client.m2_sync();
        assert!(client.chat.mods().stack);

        // Boss bar on: detached, and the frame is consumed exactly once per HUD frame.
        client.mods.settings.set_enabled("boss_bar", true);
        client.m2_before_vanilla_hud();
        assert!(client.hud.boss_bar_detached());
        assert_eq!(client.hud.boss_status_bar_time(), 99);
        assert!(client.m2.boss_frame);
        client.m2.fill(
            &mut client.mods.hud_data,
            &client.hud,
            client.sidebar_cache.as_ref(),
        );
        assert_eq!(client.mods.hud_data.boss.name, "Wither");
        assert!(client.mods.hud_data.boss.shown);
        assert!((client.mods.hud_data.boss.health_scale - 0.5).abs() < 1e-6);
        client.mods.settings.set_enabled("boss_bar", false);
        client.m2_before_vanilla_hud();
        assert!(
            !client.hud.boss_bar_detached(),
            "off again: vanilla draws it"
        );

        // Centred inventory: an open survival inventory with an effect is centred, and switching
        // the mod off restores vanilla's shift.
        let effect = mc_ui::screens::hud::ActiveEffect {
            icon_index: 0,
            name: "Speed".to_owned(),
            duration: "1:00".to_owned(),
            beneficial: true,
        };
        let mut screen = ContainerScreen::player(0);
        screen.set_active_effects(vec![effect]);
        let ctx = ScreenCtx::new(
            mc_ui::widget::ScaledResolution::new(854, 480, 2, false),
            0.0,
        );
        use mc_ui::screen_stack::Screen as _;
        screen.init(&ctx);
        let shifted = screen.gui_left();
        client.ui = Ui::Container(Box::new(screen));
        client.mods.settings.set_enabled("centered_inventory", true);
        client.m2_frame();
        let Ui::Container(screen) = &client.ui else {
            unreachable!()
        };
        assert_eq!(screen.gui_left(), (ctx.width - 176) / 2);
        assert_ne!(screen.gui_left(), shifted);
        client
            .mods
            .settings
            .set_enabled("centered_inventory", false);
        client.m2_frame();
        let Ui::Container(screen) = &client.ui else {
            unreachable!()
        };
        assert_eq!(screen.gui_left(), shifted);
    }

    /// Chat's timestamps over a replay: a line that arrives while the
    /// session runs on a recording's clock (`Environment::set_replay_clock`, which the replay
    /// player sets) is stamped with that clock, not the time it is watched. A live line keeps
    /// the wall clock (the control).
    #[test]
    fn replayed_chat_is_stamped_with_the_recordings_clock() {
        use mc_protocol::play::{s2c, S2CPlay};
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        client.mods.settings.set_enabled("chat", true);
        client.mods.settings.set_choice("chat", "timestamps", 2);
        client.mods.settings.set_bool("chat", "stack", false);
        client.m2_sync();
        let (handle, to_game, _to_net) = crate::net::channel::NetHandle::new_pair();
        client.session = Some(crate::session::Session::new(
            handle,
            crate::session::JoinSettings::default(),
        ));
        let send = |packet: S2CPlay| {
            to_game
                .try_send(crate::net::ToGame::Packet(packet))
                .expect("channel");
        };
        send(S2CPlay::JoinGame(s2c::JoinGame {
            entity_id: 1,
            hardcore_mode: false,
            game_mode: 0,
            dimension: 0,
            difficulty: 2,
            max_players: 20,
            level_type: "default".into(),
            reduced_debug_info: false,
        }));
        client.poll();
        let chat = |text: &str| {
            S2CPlay::Chat(s2c::Chat {
                component: format!("{{\"text\":\"{text}\"}}"),
                position: 0,
            })
        };
        let newest = |client: &Client| mc_ui::font::strip_formatting(&client.chat.lines()[0].text);
        let stamp =
            |(hour, minute): (u32, u32), text: &str| format!("[{hour:02}:{minute:02}] {text}");
        let now_ms = || {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(0))
        };

        // A replay: the recording's clock, 11 h 23 min away from any wall clock it shares a
        // stamp with.
        let recorded = now_ms() - 41_000_000;
        client
            .session
            .as_mut()
            .expect("session")
            .state_mut()
            .env
            .set_replay_clock(Some(recorded));
        send(chat("from the recording"));
        client.poll();
        let expected = stamp(
            mc_ui::screens::chat::stamp_clock_at(recorded),
            "from the recording",
        );
        assert_eq!(newest(&client), expected);

        // Live: the wall clock.
        client
            .session
            .as_mut()
            .expect("session")
            .state_mut()
            .env
            .set_replay_clock(None);
        let before = now_ms();
        send(chat("live"));
        client.poll();
        let after = now_ms();
        let line = newest(&client);
        let live = |ms| stamp(mc_ui::screens::chat::stamp_clock_at(ms), "live");
        assert!(line == live(before) || line == live(after), "{line}");
        assert_ne!(
            line,
            stamp(mc_ui::screens::chat::stamp_clock_at(recorded), "live")
        );
    }
}
