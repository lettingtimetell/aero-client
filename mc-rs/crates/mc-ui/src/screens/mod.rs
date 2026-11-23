//! Every screen the player can see. S16 owns everything in this module except
//! [`container`], which is S17's.
//!
//! | Module | Screen |
//! |---|---|
//! | [`common`] | backgrounds, hovering text, the unicode-font rule |
//! | [`clipboard`] | the system-clipboard seam S15 left open |
//! | [`menu`] | the main menu, its splash text and the panorama |
//! | [`multiplayer`] | the server list, `servers.dat`, ping display |
//! | [`connect`] | direct connect, add/edit server, connecting, disconnected |
//! | [`options`] | video, controls, sound, chat, language, resource packs |
//! | [`hud`] | the in-game HUD: hotbar, health, hunger, armour, XP, crosshair, effects |
//! | [`chat`] | the chat log and the chat input screen |
//! | [`debug`] | the F3 overlay |
//! | [`gameover`] | the death screen |
//! | [`pause`] | the in-game menu |
//! | [`login`] | the Microsoft device-code login screen |
//! | [`stats`] | statistics and achievements |
//! | [`tab_list`] | the Tab player-list overlay |
//! | [`container`] | inventory and containers (S17) |
//!
//! # Why screens are not all `dyn Screen`
//!
//! S15's [`Screen`](crate::screen_stack::Screen) trait hands a screen a
//! [`DrawList`](crate::widget::DrawList), a font and a
//! [`ScreenCtx`](crate::screen_stack::ScreenCtx) — and nothing else. That is right for a screen
//! whose state is entirely its own, and every such screen here does implement it. But the
//! server list needs `servers.dat`, the options screens need
//! [`GameSettings`](mc_render::settings::GameSettings), the HUD and F3 need
//! [`World`](mc_world::World), and every screen needs
//! [`Language`](mc_assets::Language). Reaching those through the trait would mean either a
//! shared mutable graph — the `Rc<RefCell<>>` the arena-index rule exists to forbid — or a context
//! struct big enough to be its own coupling problem.
//!
//! So the rule is: **state a screen does not own arrives as an explicit parameter, and what the
//! screen wants done about it leaves as an [`Action`]**. Nothing here holds a back-pointer to
//! its parent the way vanilla's `parentScreen` does. That also makes every screen driveable
//! from a unit test with no window, no GPU and no network, which is most of why the tests below
//! exist at all.

// The Accounts screen and the title screen's account corner.
pub mod accounts;
// The Aero title screen, its title font and phase.
pub mod aero_title;
pub mod book;
pub mod chat;
pub mod clipboard;
pub mod common;
pub mod connect;
pub mod container;
pub mod debug;
pub mod demo;
pub mod gameover;
pub mod hud;
pub mod login;
pub mod menu;
// The mod menu (menu key: Right Shift), its glass phase and the HUD
// editor.
pub mod modmenu;
pub mod multiplayer;
pub mod options;
pub mod pause;
// The replay browser and viewer controls.
pub mod replay;
pub mod sign;
pub mod sleep;
pub mod spectator;
pub mod stats;
pub mod tab_list;
pub mod yes_no;

use multiplayer::ServerEntry;

/// What a screen wants the application to do.
///
/// Vanilla's screens call `mc.displayGuiScreen(...)` directly, which is what makes them
/// impossible to test and impossible to reuse. Returning an intent instead keeps the screen a
/// pure function of its inputs.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Action {
    /// The event was handled and nothing else should happen.
    #[default]
    None,
    /// Close this screen and show whatever was underneath.
    Back,
    /// Close every screen and return to the game. java: `displayGuiScreen(null)`
    CloseAll,
    /// Quit the game. java: `mc.shutdown()`
    Quit,
    /// Open the multiplayer server list.
    OpenMultiplayer,
    /// Open the mod menu over the current screen (the "Mods"
    /// buttons on the pause menu and the Aero title screen). Not vanilla.
    OpenModMenu,
    /// Open the options root screen.
    OpenOptions,
    /// Open one of the options sub-screens.
    OpenOptionsPage(options::OptionsPage),
    /// Press GuiOptions' deliberately unlabelled shader/sound cycler (button id 8675309).
    PressSuperSecretSettings,
    /// Open the direct-connect screen.
    OpenDirectConnect,
    /// Open the add-server screen with a blank entry.
    OpenAddServer,
    /// Open the add-server screen pre-filled for editing.
    OpenEditServer(Box<ServerEntry>),
    /// Ask the player to confirm deleting the named server.
    ConfirmDeleteServer(String),
    /// The add/edit screen produced a finished entry; add it or write it back.
    AcceptServer(Box<ServerEntry>),
    /// Begin connecting to a server.
    Connect(Box<ServerEntry>),
    /// The server list changed and should be written back to `servers.dat`.
    SaveServers,
    /// `options.txt` changed and should be written back.
    SaveSettings,
    /// Something changed that invalidates every laid-out screen — a GUI scale or font switch.
    Relayout,
    /// Leave the current server. java: `GuiIngameMenu` button 1
    Disconnect,
    /// Ask the server to respawn the dead player, then return to the game.
    ///
    /// java: `GuiGameOver.actionPerformed` case 0 — `thePlayer.respawnPlayer()` followed by
    /// `displayGuiScreen(null)`. `respawnPlayer` is `C16PacketClientStatus(PERFORM_RESPAWN)` and
    /// **the server will not move a dead client without it**, so this is the only exit from
    /// [`gameover::GameOverScreen`] that keeps the session.
    Respawn,
    /// Cancel an in-progress connection.
    CancelConnect,
    /// Open the statistics screen.
    OpenStats,
    /// Open the achievements screen.
    OpenAchievements,
    /// Start (or restart) the Microsoft device-code login.
    BeginLogin,
    /// Send a chat message or command to the server.
    SendChat(String),
    /// Send chat from `GuiSleepMP` without closing the sleep screen.
    SendSleepChat(String),
    /// Ask the server to wake the local player from bed.
    StopSleeping,
    /// Ask the server to complete the partial command in the chat box.
    // java: C14PacketTabComplete
    RequestTabComplete(String),
    /// Open a URL in the system browser, after the player confirmed it.
    OpenUrl(String),
    /// Copy text to the system clipboard (currently the login screen's Copy code button).
    ///
    /// `copy_to_clipboard` is not a 1.8.9 chat-component click action; chat components use
    /// [`crate::font::component::ClickAction`], whose six actions mirror vanilla's
    /// `ClickEvent.Action` enum. This app-level intent remains useful for non-chat UI.
    CopyToClipboard(String),
    /// Run a command from a chat component's `click.run_command`.
    RunCommand(String),
    /// Replace the chat input with a component's `click.suggest_command` text.
    SuggestCommand(String),
    /// Open the resource-pack folder in the system file browser (the pack screen's "Open Pack
    /// Folder" button).
    ///
    /// Its own intent, not an [`Action::OpenUrl`] with a reserved value:
    /// that value is also what a server's `open_url` chat or book link carries, which then opened
    /// the folder with no confirmation where vanilla refuses the scheme-less URL.
    OpenResourcePackFolder,
    /// The Accounts screen and the title screen's account corner (no vanilla
    /// equivalent): open the screen, add, switch to or remove an account.
    Accounts(accounts::AccountsAction),
}

impl Action {
    /// Whether this is [`Action::None`].
    #[must_use]
    pub fn is_none(&self) -> bool {
        matches!(self, Action::None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_action_is_do_nothing() {
        assert!(Action::default().is_none());
        assert!(!Action::Back.is_none());
    }
}
