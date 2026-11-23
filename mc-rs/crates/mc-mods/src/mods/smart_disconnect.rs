//! Confirm Disconnect: the pause menu's Disconnect asks before it leaves the server.
//!
//! Keep the `id` stable (it is the settings-file key, kept from the
//! mod's first name when the display name changed); the `category`
//! line must name the mod's home in `crate::menu_layout`, which decides where the menu lists it
//! (a test keeps the two in step).
//!
//! **Off by default**, and off is vanilla `GuiIngameMenu`: Disconnect leaves at once. On, the
//! button opens a question in vanilla `GuiYesNo`'s layout ("Are you sure you want to
//! disconnect?"), whose two buttons sit on a row of their own and wake up after
//! [`ARM_TICKS`] ticks, so a double click or a stray second click can never answer it. Only its
//! **Disconnect** leaves, through exactly the teardown the pause menu's button runs (the channel
//! is closed, no packet is sent); **Cancel** goes back to the pause menu and Escape back to the
//! game. Joining another server from the pause menu's server list (where a double click on a row
//! joins) asks the same question first; its **Join Server** runs the unchanged join. While either
//! question asks, nothing changes on the wire: the game keeps ticking and talking to the server
//! exactly as it does under the pause menu. A replay has no server to lose, so leaving one never
//! asks.
//!
//! The pause menu's question lives in `mc_ui::screens::pause::PauseScreen`; the client half (and
//! the join question) is `crates/mc-client/src/app/mods/w18_fb.rs`.

use crate::category::Category;
use crate::descriptor::ModDescriptor;

/// The mod's id. A settings-file key: never rename it.
pub const ID: &str = "smart_disconnect";

/// Ticks (1/20 s) before the question's buttons accept a click: half a second, the usual double
/// click time. Vanilla arms its own leave-the-game question on the death screen the same way
/// (`GuiGameOver` calls `GuiYesNo.setButtonDelay(20)`).
pub const ARM_TICKS: i32 = 10;

/// The catalog entry.
pub static DESCRIPTOR: ModDescriptor = ModDescriptor {
    id: ID,
    name: "Confirm Disconnect",
    category: Category::Interface,
    description: "Disconnect in the pause menu, or joining another server from its server list, \
                  asks first, so a stray click can't end your game.",
    default_enabled: false,
    options: &[],
    hud_elements: &[],
};
