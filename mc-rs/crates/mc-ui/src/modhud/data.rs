//! [`HudData`]: the per-frame snapshot of game state the mod HUD elements read.
//!
//! The client fills it in `mc-client/src/app/mods.rs` (`Client::fill_hud_data`), and **only**
//! on frames where at least one mod HUD element is enabled, so it costs nothing while every HUD
//! mod is off. Fields are updated in place: item stacks are only re-cloned when they changed,
//! and `effects` reuses its allocation.
//!
//! # Adding a field (lanes)
//!
//! Add it under your lane's marker below, give it a cheap `Default`, and fill it under the same
//! marker in `fill_hud_data`. Keep the type plain data (no references, no locks).

use mc_protocol::ItemStack;

/// One active potion effect on the local player.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HudEffect {
    /// Vanilla potion id (`Potion.potionTypes` index): 1 speed, 2 slowness, ... 23 saturation.
    pub id: u8,
    /// Zero-based level: 0 is "I", 1 is "II".
    pub amplifier: u8,
    /// Remaining ticks (20 per second).
    pub duration_ticks: i32,
    /// The server sent the 32767-tick cap: vanilla shows `**:**`.
    pub duration_max: bool,
}

impl HudEffect {
    /// Vanilla's `Potion.getDurationString`: `m:ss`, or `**:**` for a capped duration.
    #[must_use]
    pub fn duration_text(&self) -> String {
        if self.duration_max {
            return "**:**".to_owned();
        }
        let seconds = self.duration_ticks.max(0) / 20;
        format!("{}:{:02}", seconds / 60, seconds % 60)
    }
}

/// The last attack that hit an entity, for the reach display.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct HudReach {
    /// Eye-to-hitbox distance in blocks.
    pub distance: f64,
    /// [`HudData::now_ms`] clock when it landed.
    pub at_ms: u64,
}

/// What the toggle-sprint HUD text should say. The client sets it; the HUD element draws it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SprintHudState {
    /// Nothing to show.
    #[default]
    Off,
    /// Sprinting because the key is physically held.
    SprintHeld,
    /// Sprinting because sprint is toggled on.
    SprintToggled,
    /// Sneaking because the key is physically held.
    SneakHeld,
    /// Sneaking because sneak is toggled on.
    SneakToggled,
}

/// Pressed state of the bindings the keystrokes mod draws. Read from the **key bindings** (so a
/// remapped key lights up the right box), sampled once per frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HudKeys {
    /// `key.forward`.
    pub forward: bool,
    /// `key.left`.
    pub left: bool,
    /// `key.back`.
    pub back: bool,
    /// `key.right`.
    pub right: bool,
    /// `key.jump`.
    pub jump: bool,
    /// `key.sneak`.
    pub sneak: bool,
    /// `key.attack` (left mouse by default).
    pub attack: bool,
    /// `key.use` (right mouse by default).
    pub use_item: bool,
}

/// Everything the mod HUD elements read, refreshed once per frame while any is enabled.
#[derive(Debug, Clone, Default)]
pub struct HudData {
    /// Milliseconds since the client started (monotonic). Use it for fades and timeouts.
    pub now_ms: u64,
    /// Client tick counter (20 Hz).
    pub tick: u64,

    // ---- filled by F --------------------------------------------------------------------
    /// Frames presented in the last whole second (the F3 number).
    pub fps: u32,
    /// Left clicks in the last second. Shared by `cps` and `keystrokes` (filled by M1).
    pub cps_left: u32,
    /// Right clicks in the last second (filled by M1).
    pub cps_right: u32,
    /// Latency to the server from the tab list, `None` before it arrives.
    pub ping_ms: Option<i32>,
    /// Feet position.
    pub position: [f64; 3],
    /// Yaw in degrees (vanilla: 0 = south, 90 = west).
    pub yaw: f32,
    /// Pitch in degrees (positive looks down).
    pub pitch: f32,
    /// Helmet, chestplate, leggings, boots.
    pub armor: [Option<ItemStack>; 4],
    /// The selected hotbar stack.
    pub held: Option<ItemStack>,
    /// Active effects on the local player, sorted by potion id.
    pub effects: Vec<HudEffect>,
    /// The last melee hit, for the reach display (filled by M1).
    pub reach: Option<HudReach>,
    /// Toggle-sprint text state (filled by M3).
    pub sprint: SprintHudState,
    /// Binding states for keystrokes, sampled per frame (filled by F, only while keystrokes is
    /// enabled).
    pub keys: HudKeys,

    // ---- lane fields: add yours under your marker -----------------------------------------
    /// Arrows (item 262) across the 36 main inventory slots. Filled by M1 while `armor_status`
    /// or `item_info` is enabled (a held bow shows it).
    pub arrow_count: u32,
    /// The held stack's item and damage summed across the 36 main slots, the held stack
    /// included (M1, same gate as `arrow_count`).
    pub held_total: u32,
    /// Vanilla's held-item name timer, `GuiIngame.remainingHighlightTicks` (M1: `item_info`'s
    /// "fade with item name").
    pub held_highlight_ticks: i32,
    /// The local player is a spectator: there is no hotbar, so `armor_status` and `item_info`
    /// hide (M1).
    pub spectator: bool,
    /// The server set reduced debug info: F3 hides coordinates, and so does `coordinates` (M1).
    pub reduced_debug_info: bool,
    /// Counted attack and use presses since the client started, `[attack, use]`: the presses
    /// `cps_left`/`cps_right` count. Keystrokes compares them frame to frame, so a click that
    /// is released within one frame still lights its key for that frame (M1).
    pub click_presses: [u64; 2],
    /// Keystrokes labels for forward, left, back and right: the bound key's name when it is one
    /// to three characters long, else empty (draw W, A, S, D). Rebuilt only when a binding
    /// changes (M1).
    pub move_key_labels: [String; 4],
    /// The boss bar for the `boss_bar` element: whether vanilla would draw it this frame, the
    /// name and the health. Filled by M2 only while `boss_bar` is enabled; the name is copied
    /// only when it changed.
    pub boss: crate::modhud::elements::boss_bar::HudBoss,
    /// The sidebar for the `scoreboard` element, `None` when no objective is shown. Filled by
    /// M2 only while `scoreboard` is enabled, and re-cloned only when the client's sidebar
    /// cache was rebuilt.
    pub sidebar: Option<crate::screens::hud::Sidebar>,
    /// Bumped every time [`HudData::sidebar`] is replaced, so the element re-measures only then.
    pub sidebar_revision: u64,
    /// What the toggle-sprint text says. Set every frame by M3 (`m3_controls.rs`) while
    /// `toggle_sprint` is on. (`sprint` above lacks flying, riding and vanilla sprint, so the
    /// element reads this instead.)
    pub sprint_status: crate::modhud::elements::toggle_sprint::SprintStatus,
    /// The Spotify widget's track, state and Segoe UI renderer. Filled by
    /// the client (`w9_spotify.rs`) only while `spotify` is on; strings only when they change.
    pub spotify: crate::modhud::elements::spotify::SpotifyHud,
    /// The overlay message and its colour this frame, for the `action_bar` element. Filled by the client (`x3_screens.rs`) only while `action_bar` is on; the
    /// text is copied only when it changed.
    pub action_bar: crate::modhud::elements::action_bar::HudActionBar,
    /// The title, subtitle and fade alpha this frame, for the `titles` element (X3). Filled only
    /// while `titles` is on; strings are copied only when they changed.
    pub titles: crate::modhud::elements::titles::HudTitles,
}

impl HudData {
    /// The compass direction vanilla's F3 screen prints for `yaw`: `"south"`, `"west"`,
    /// `"north"` or `"east"`, plus the axis you are looking along.
    ///
    /// java: `MathHelper.floor_double(yaw * 4 / 360 + 0.5) & 3` indexes
    /// `EnumFacing.getHorizontal`.
    #[must_use]
    pub fn facing(&self) -> (&'static str, &'static str) {
        let index = ((f64::from(self.yaw) * 4.0 / 360.0 + 0.5).floor() as i64) & 3;
        match index {
            0 => ("south", "+Z"),
            1 => ("west", "-X"),
            2 => ("north", "-Z"),
            _ => ("east", "+X"),
        }
    }

    /// The block the player's feet are in.
    #[must_use]
    pub fn block_position(&self) -> [i32; 3] {
        self.position.map(|v| v.floor() as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn facing_matches_vanilla_quadrants() {
        let at = |yaw: f32| {
            HudData {
                yaw,
                ..HudData::default()
            }
            .facing()
            .0
        };
        assert_eq!(at(0.0), "south");
        assert_eq!(at(90.0), "west");
        assert_eq!(at(180.0), "north");
        assert_eq!(at(-90.0), "east");
        assert_eq!(at(44.0), "south");
        assert_eq!(at(46.0), "west");
        assert_eq!(at(-720.0 + 180.0), "north");
    }

    #[test]
    fn effect_durations_format_like_vanilla() {
        let effect = HudEffect {
            id: 1,
            amplifier: 0,
            duration_ticks: 20 * 95,
            duration_max: false,
        };
        assert_eq!(effect.duration_text(), "1:35");
        let capped = HudEffect {
            duration_max: true,
            ..effect
        };
        assert_eq!(capped.duration_text(), "**:**");
    }
}
