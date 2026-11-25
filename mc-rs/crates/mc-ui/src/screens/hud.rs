//! The in-game HUD: hotbar, health, armour, hunger, air, XP, crosshair, effects.
//!
//! java: `client/gui/GuiIngame.java`
//!
//! # Where the numbers come from
//!
//! [`PlayerHud`] is the whole input. `mc-world`'s ECS components stop at position, velocity,
//! rotation, kind and metadata — health, food, armour and XP arrive on the wire as
//! `S06PacketUpdateHealth`, `S1FPacketSetExperience` and `DataWatcher` entries and are S14's to
//! decode — so the HUD takes a plain snapshot struct rather than reaching into the world. That
//! keeps every layout rule below testable from a literal.
//!
//! # The heart row arithmetic
//!
//! Vanilla's heart layout is the fiddliest part of the whole HUD and every constant matters:
//!
//! * `rows = ceil((max_health + absorption) / 2 / 10)` — how many rows of ten hearts.
//! * `row_pitch = max(10 - (rows - 2), 3)` — rows *overlap* once there are more than two, down
//!   to a floor of 3 pixels.
//! * the armour bar sits at `base - (rows - 1) * row_pitch - 10`, i.e. it is pushed up by the
//!   whole heart stack.
//!
//! A port that hard-codes ten hearts looks right until the player drinks a health-boost potion.

use crate::font::{FontRenderer, TextSink};
use crate::screen_stack::ScreenCtx;
use crate::screens::container::render::{
    draw_count_text, draw_durability_bar, draw_item_icon_parts, draw_item_icon_tinted,
    ItemIconParts,
};
use crate::screens::container::stack_ext::{self, ItemStack};
use crate::widget::draw::{DrawList, UiTexture, UnblendedText};
use crate::widget::draw_centered_string;
use crate::widget::Transform;
use std::cell::Cell;

// Minecraft 1.7.10's sidebar rules: a child module, for `Sidebar`'s private state.
#[path = "hud_v5.rs"]
mod v5;

/// `textures/gui/container/inventory.png` — where the potion icons live.
pub const INVENTORY_TEXTURE: &str = "textures/gui/container/inventory.png";

/// What the player's first-person targeting pass says is under the reticle.
///
/// This deliberately describes the three *semantic* predicates `GuiIngame.showCrosshair` uses,
/// rather than exposing a block id or an entity id to the HUD.  Deciding whether an entity is an
/// `EntityLivingBase` or an `EntityItemFrame`, and whether a block's loaded tile entity implements
/// `IInventory`, belongs to the client/world layer which owns those objects.
// java: GuiIngame.java:513-543 (`showCrosshair`)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CrosshairTarget {
    /// No relevant target, including an ordinary non-living entity or non-inventory block.
    #[default]
    None,
    /// `pointedEntity instanceof EntityLivingBase`.
    LivingEntity,
    /// `pointedEntity instanceof EntityItemFrame` (the second branch that updates
    /// `Minecraft.pointedEntity` in `EntityRenderer.getMouseOver`).
    ItemFrame,
    /// `objectMouseOver` is a block whose *loaded tile entity* is an `IInventory`.
    InventoryBlock,
}

/// Inputs to vanilla's `GuiIngame.showCrosshair` predicate.
///
/// The default is the ordinary, non-spectator path, where vanilla always draws the crosshair.
/// A spectator must supply the semantic result of its current target through
/// [`CrosshairTarget`]; `hasTileEntity` alone is intentionally insufficient (a sign is the
/// important negative control).  `third_person` is retained in the state snapshot for callers
/// which already collect the camera mode, but vanilla's `showCrosshair()` does not inspect it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CrosshairState {
    /// The camera mode collected by the client.  This is intentionally not a visibility gate:
    /// Java's `GuiIngame.showCrosshair()` has no `thirdPersonView` branch.
    pub third_person: bool,
    /// `gameSettings.showDebugInfo`.
    pub show_debug_info: bool,
    /// `thePlayer.hasReducedDebug()`.
    pub has_reduced_debug: bool,
    /// `gameSettings.reducedDebugInfo`.
    pub reduced_debug_info: bool,
    /// `thePlayer.isSpectator()`.
    pub spectator: bool,
    /// The currently targeted entity/block, reduced to `showCrosshair`'s relevant predicates.
    pub target: CrosshairTarget,
}

impl CrosshairState {
    /// Java `GuiIngame.showCrosshair` exactly.
    #[must_use]
    pub const fn shows_crosshair(self) -> bool {
        if self.show_debug_info && !self.has_reduced_debug && !self.reduced_debug_info {
            return false;
        }
        !self.spectator
            || matches!(
                self.target,
                CrosshairTarget::LivingEntity
                    | CrosshairTarget::ItemFrame
                    | CrosshairTarget::InventoryBlock
            )
    }
}

/// One active potion effect, as the HUD needs it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveEffect {
    /// `Potion.getStatusIconIndex()`, the cell in the 8-wide icon grid.
    pub icon_index: i32,
    /// The translated effect name, amplifier suffix included.
    pub name: String,
    /// Remaining duration, formatted as `m:ss`.
    pub duration: String,
    /// Whether the effect is beneficial. Vanilla tints the icon by this in later versions; in
    /// 1.8.9 it only decides nothing, but S21 may want it.
    pub beneficial: bool,
}

/// How many rows the sidebar shows at most. java: `if (list.size() > 15)` in `renderScoreboard`.
pub const SIDEBAR_MAX_ROWS: usize = 15;

/// The scoreboard sidebar, as [`Hud::draw_scoreboard`] wants it.
///
/// java: the `ScoreObjective` + `Scoreboard.getSortedScores(objective)` pair that
/// `GuiIngame.renderScoreboard` is handed. The names arrive **already team-formatted** —
/// `ScorePlayerTeam.formatPlayerName` needs the scoreboard, which lives in `mc-client`, and
/// `mc-ui` deliberately takes state it does not own as a parameter (see this module's siblings).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Sidebar {
    /// The objective's display name, drawn centred on the header plate.
    pub title: String,
    /// `(formatted name, points)`, **ascending by points** — `getSortedScores` order.
    pub rows: Vec<(String, i32)>,
    /// How many leading rows the 15-row cap drops. See [`Sidebar::visible_rows`].
    skip: usize,
}

impl Sidebar {
    /// The rows that are actually drawn: the bottom-15 slice of `rows`, with vanilla's `#`
    /// filter already applied by [`Sidebar::from_scores`].
    ///
    /// java:
    /// ```text
    /// if (list.size() > 15) collection = Iterables.skip(list, collection.size() - 15);
    /// else                  collection = list;
    /// ```
    /// Note `collection.size()` there is the **unfiltered** count, so when `#`-prefixed rows were
    /// dropped vanilla skips more than it needs to and shows *fewer* than 15. That is reproduced
    /// rather than corrected — see [`Sidebar::from_scores`], which records the unfiltered count.
    #[must_use]
    pub fn visible_rows(&self) -> &[(String, i32)] {
        let skip = self.skip.min(self.rows.len());
        &self.rows[skip..]
    }

    /// Build from `getSortedScores` output, applying vanilla's filter and its 15-row cap.
    ///
    /// `scores` carries the **raw** player names, ascending by points, and `format` is the
    /// caller's `ScorePlayerTeam.formatPlayerName`. The two are separate because vanilla filters
    /// on the raw name and formats afterwards: a row named `#hidden` is dropped even if its
    /// team's prefix would have moved the `#` off the front. java: the `Predicate<Score>` at
    /// `renderScoreboard:556` — `getPlayerName() != null && !startsWith("#")` — which is how
    /// servers park bookkeeping entries on a board without showing them.
    #[must_use]
    pub fn from_scores(
        title: impl Into<String>,
        scores: &[(String, i32)],
        mut format: impl FnMut(&str) -> String,
    ) -> Sidebar {
        let unfiltered = scores.len();
        let rows: Vec<(String, i32)> = scores
            .iter()
            .filter(|(name, _)| !name.starts_with('#'))
            .map(|(name, points)| (format(name), *points))
            .collect();
        let skip = if rows.len() > SIDEBAR_MAX_ROWS {
            unfiltered.saturating_sub(SIDEBAR_MAX_ROWS)
        } else {
            0
        };
        Sidebar {
            title: title.into(),
            rows,
            skip,
        }
    }
}

/// A snapshot of everything the HUD draws.
#[derive(Debug, Clone, PartialEq)]
pub struct PlayerHud {
    /// Current health in half-hearts' worth of float, 0..=`max_health`.
    pub health: f32,
    /// `SharedMonsterAttributes.maxHealth`, normally 20.
    pub max_health: f32,
    /// Absorption hearts (golden), added on top.
    pub absorption: f32,
    /// `getTotalArmorValue()`, 0..=20.
    pub armor: i32,
    /// Food level, 0..=20.
    pub food: i32,
    /// Saturation. At zero the food icons jitter.
    pub saturation: f32,
    /// Remaining air, 0..=300. Only drawn while the head is underwater.
    pub air: i32,
    /// Whether the player's head is in water.
    pub underwater: bool,
    /// XP bar fill, 0..=1.
    pub experience: f32,
    /// XP level. Drawn only when above zero.
    pub experience_level: i32,
    /// `EntityPlayerSP.horseJumpPower` while mounted on a saddled horse. `Some` replaces the XP
    /// bar for this frame; `None` selects the ordinary experience path.
    pub horse_jump_power: Option<f32>,
    /// The selected hotbar slot, 0..=8.
    pub selected_slot: i32,
    /// The nine hotbar stacks, `mainInventory[0..9]`. `None` is vanilla's `null` slot.
    ///
    /// This is the player's *own* inventory, which arrives as `S30PacketWindowItems` for window
    /// 0 and is S17's [`Container`](crate::screens::container::Container) to hold — the HUD only
    /// needs a read-only snapshot of the first nine slots, so it takes one rather than borrowing
    /// a container it does not own.
    // java: GuiIngame.java:1039 — `player.inventory.mainInventory[index]`
    pub hotbar: [Option<ItemStack>; 9],
    /// Pickup-pop animation ticks remaining for each visible hotbar slot.
    ///
    /// In Java this lives on each `ItemStack` as `animationsToGo`; the UI snapshot keeps the
    /// nine values beside the stacks so a caller can copy the client inventory's 36..44 slots
    /// without making the renderer know about `ItemStack`'s mutable animation field. Values are
    /// rendered with the current frame's [`ScreenCtx::partial_ticks`] and should be decremented
    /// by the inventory owner once per client tick.
    // java: ItemStack.animationsToGo; GuiIngame.java:1037-1060
    pub animations_to_go: [i32; 9],
    /// Whether the player is in survival or adventure mode, which is what shows the stat bars.
    pub survival: bool,
    /// `EntityPlayer.sleepTimer`: `0` while awake, ramps through `1..=100` while sleeping, then
    /// advances through `101..=109` during the non-immediate wake fade before returning to zero.
    pub sleep_timer: i32,
    /// Whether the world is hardcore, which swaps the heart artwork.
    pub hardcore: bool,
    /// Whether poison is active (green hearts).
    pub poisoned: bool,
    /// Whether wither is active (black hearts).
    pub withered: bool,
    /// Whether hunger is active (green drumsticks).
    pub hungry: bool,
    /// Whether regeneration is active, which lifts one heart per tick.
    pub regenerating: bool,
    /// Active potion effects.
    pub effects: Vec<ActiveEffect>,
    /// Draw the potion list on the HUD.
    ///
    /// **Off by default, because 1.8.9 does not do this**: vanilla draws active effects only on
    /// the inventory screen (`InventoryEffectRenderer.drawActivePotionEffects`). The geometry is
    /// implemented here so both callers share it; turning it on is an addition, not a port.
    pub show_effect_overlay: bool,
}

impl Default for PlayerHud {
    fn default() -> Self {
        Self {
            health: 20.0,
            max_health: 20.0,
            absorption: 0.0,
            armor: 0,
            food: 20,
            saturation: 5.0,
            air: 300,
            underwater: false,
            experience: 0.0,
            experience_level: 0,
            horse_jump_power: None,
            selected_slot: 0,
            hotbar: [None, None, None, None, None, None, None, None, None],
            animations_to_go: [0; 9],
            survival: true,
            sleep_timer: 0,
            hardcore: false,
            poisoned: false,
            withered: false,
            hungry: false,
            regenerating: false,
            effects: Vec::new(),
            show_effect_overlay: false,
        }
    }
}

/// How many rows of ten hearts the health bar needs.
// java: GuiIngame.java:648 — `MathHelper.ceiling_float_int((f + f1) / 2.0F / 10.0F)`
#[must_use]
pub fn heart_rows(max_health: f32, absorption: f32) -> i32 {
    mc_math::ceil_f32((max_health + absorption) / 2.0 / 10.0)
}

/// The vertical pitch between heart rows, which shrinks as rows are added.
// java: GuiIngame.java:649 — `Math.max(10 - (l1 - 2), 3)`
#[must_use]
pub fn heart_row_pitch(rows: i32) -> i32 {
    // Java's wrapping `int` arithmetic: a hostile absorption amount makes
    // `rows` wrap to `Integer.MIN_VALUE` (see `mc_math::ceil_f32`), which must not panic in a
    // build with overflow checks.
    10i32.wrapping_sub(rows.wrapping_sub(2)).max(3)
}

/// The y of the armour bar (and the air bubbles), given the heart geometry.
// java: GuiIngame.java:650 — `k1 - (l1 - 1) * i2 - 10`
#[must_use]
pub fn armor_row_y(base_y: i32, rows: i32, pitch: i32) -> i32 {
    base_y
        .wrapping_sub(rows.wrapping_sub(1).wrapping_mul(pitch))
        .wrapping_sub(10)
}

/// The three armour sprites: full, half, empty. java: `u = 34 / 25 / 16`, `v = 9`
#[must_use]
pub fn armor_sprite_u(slot: i32, armor: i32) -> i32 {
    if slot * 2 + 1 < armor {
        34
    } else if slot * 2 + 1 == armor {
        25
    } else {
        16
    }
}

/// The u offset of the *filled* heart sprite family for the current status effects.
// java: GuiIngame.java:689 — 16 base, +36 poisoned, +72 withered
#[must_use]
pub fn heart_family_u(poisoned: bool, withered: bool) -> i32 {
    if poisoned {
        16 + 36
    } else if withered {
        16 + 72
    } else {
        16
    }
}

/// The v row of the heart sprites: hardcore worlds use the sixth row.
// java: GuiIngame.java:723 — `k4 = 5` when hardcore, then `9 * k4`
#[must_use]
pub fn heart_row_v(hardcore: bool) -> i32 {
    if hardcore {
        45
    } else {
        0
    }
}

/// The most hearts [`Hud::draw_player_stats`] draws in one frame: four
/// times the 1024 that the largest legitimate max health plus absorption gives. Rows past it
/// would sit far above the top of the screen at the 3-pixel minimum row pitch anyway.
pub const MAX_DRAWN_HEARTS: i32 = 4096;

/// How many air bubbles are full and how many are popping.
// java: GuiIngame.java:877
#[must_use]
pub fn air_bubbles(air: i32) -> (i32, i32) {
    let full = mc_math::ceil_f64(f64::from(air - 2) * 10.0 / 300.0);
    let popping = mc_math::ceil_f64(f64::from(air) * 10.0 / 300.0) - full;
    (full, popping)
}

/// Kill switch for the pass-ordered hotbar items
/// ([`Hud::draw_hotbar_items_in_order`]) and Tab list rows (`TabListOverlay::draw`):
/// `MC_NW_F3_PASSES=0` draws them element by element, as before. Read once per process.
#[must_use]
pub fn nw_f3_passes_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var_os("MC_NW_F3_PASSES").is_none_or(|value| value != "0")
    });
    *ON
}

/// The item-icon scale for one frame of vanilla's hotbar pickup animation.
///
/// `ItemStack.animationsToGo` is an integer countdown, but the renderer subtracts the frame's
/// partial tick before deriving the two scales. Returning `None` for a non-positive (or NaN)
/// remainder preserves Java's `if (f > 0.0F)` guard and gives callers a cheap CPU-side seam to
/// test without constructing a GPU frame.
// java: GuiIngame.java:1043-1053
#[must_use]
pub fn hotbar_item_pop_scale(animations_to_go: i32, partial_ticks: f32) -> Option<(f32, f32)> {
    let remaining = animations_to_go as f32 - partial_ticks;
    if !matches!(
        remaining.partial_cmp(&0.0),
        Some(std::cmp::Ordering::Greater)
    ) {
        return None;
    }
    let f1 = 1.0 + remaining / 5.0;
    Some((1.0 / f1, (f1 + 1.0) / 2.0))
}

/// Build the temporary matrix that vanilla applies around a hotbar slot's icon.
///
/// Keeping the pivot construction separate from the item draw makes the Java GL sequence
/// directly testable while ensuring every icon path (flat, block, or glint) shares the same
/// anchor.
// java: GuiIngame.java:1048-1051
#[must_use]
fn hotbar_item_pop_transform(x: i32, y: i32, scale_x: f32, scale_y: f32) -> Transform {
    let anchor_x = (x + 8) as f32;
    let anchor_y = (y + 12) as f32;
    Transform::translate(anchor_x, anchor_y)
        .then_scale_xy(scale_x, scale_y)
        .then_translate(-anchor_x, -anchor_y)
}

/// Vanilla's full-screen sleep-fade colour for an `EntityPlayer.sleepTimer` value.
///
/// The two ramps are deliberately asymmetric in duration: falling asleep reaches alpha 220 over
/// 100 ticks; `wakeUpPlayer(false, ...)` resets the timer to 100 and the following entity ticks
/// expose 101 through 109, fading out in ten steps before 110 is replaced with zero. The RGB is
/// Java's decimal literal `1052704` (`0x101020`).
// java: GuiIngame.java:194-211
#[must_use]
pub fn sleep_overlay_color(sleep_timer: i32) -> Option<u32> {
    if sleep_timer <= 0 {
        return None;
    }
    let mut fade = sleep_timer as f32 / 100.0;
    if fade > 1.0 {
        fade = 1.0 - (sleep_timer - 100) as f32 / 10.0;
    }
    let alpha = (220.0 * fade) as i32;
    Some(((alpha << 24) as u32) | 0x0010_1020)
}

/// The interpolated portal time that may be submitted to the post overlay this frame.
///
/// Vanilla's `GuiIngame.renderGameOverlay` only calls `renderPortal` when the local player's
/// active-potion map does **not** contain `Potion.confusion` (the nausea effect). Java's
/// `isPotionActive` is a map-presence check, so the amplifier and remaining duration do not
/// change this gate. `None` also represents the ordinary `f <= 0` render guard; callers should
/// write zero to the render phase when it is absent so a previously submitted overlay cannot
/// remain visible.
// java: GuiIngame.java:152-160
#[must_use]
pub fn portal_overlay_time(time_in_portal: f32, nausea_amplifier: Option<i8>) -> Option<f32> {
    if nausea_amplifier.is_some()
        || !matches!(
            time_in_portal.partial_cmp(&0.0),
            Some(std::cmp::Ordering::Greater)
        )
    {
        None
    } else {
        Some(time_in_portal)
    }
}

/// The per-frame HUD RNG seed. Java multiplies as `int` and widens the wrapped result to `long`.
// java: GuiIngame.java:637 — `(long)(this.updateCounter * 312871)`
fn hud_random_seed(update_counter: i32) -> i64 {
    i64::from(update_counter.wrapping_mul(312_871))
}

/// The in-game HUD.
#[derive(Debug, Clone)]
pub struct Hud {
    /// Ticks since the HUD was created, for the food jitter and the regeneration bounce.
    pub update_counter: i32,
    /// `GuiIngame.playerHealth`, the most recently observed ceiling-rounded health.
    player_health: i32,
    /// `GuiIngame.lastPlayerHealth`, retained while the heart-change flash runs.
    last_player_health: i32,
    /// `GuiIngame.lastSystemTime`, represented in the caller's monotonic milliseconds.
    last_health_system_time_ms: u64,
    /// `GuiIngame.healthUpdateCounter`; its relation to `update_counter` selects every other
    /// three-tick phase of the health flash.
    health_update_counter: i64,
    /// The flash predicate sampled before this frame updates `health_update_counter`.
    health_flash_active: bool,
    /// "1.7 Animations"' `old_health`: skip the white "ghost" hearts that
    /// show the health a hit took. The container still flashes. `false` (vanilla) by default.
    pub hide_lost_health_hearts: bool,
    /// The name of the item just selected, shown above the hotbar for 40 ticks.
    ///
    /// Already formatted: [`Hud::update_held_item`]'s caller resolves the display name and
    /// prefixes `§o` for a custom-named stack, exactly as `renderSelectedItem` does.
    pub held_item_name: String,
    /// How long that name stays up. java: `remainingHighlightTicks`
    pub highlight_ticks: i32,
    /// The stack the popup is currently about, so a *change* can be detected.
    // java: GuiIngame.java:73 — `highlightingItemStack`
    pub highlighting: Option<ItemStack>,
    /// The overlay message above the hotbar — the action bar, and the "Now playing" line a
    /// jukebox raises.
    ///
    /// **This is a different field from [`Hud::held_item_name`] with a different lifetime**, and
    /// conflating the two was a real defect: an action-bar message set by a server was
    /// overwritten by the held-item name on the very next tick. Vanilla keeps `recordPlaying`
    /// (60 ticks, drawn at `height - 68`) apart from `highlightingItemStack` (40 ticks, drawn at
    /// `height - 59`), and so does this.
    // java: GuiIngame.java:60 — `recordPlaying`
    pub record_playing: String,
    /// Remaining ticks of the overlay message. java: `recordPlayingUpFor`, set to 60
    pub record_ticks: i32,
    /// Whether the message came from a jukebox, which rainbow-cycles its colour.
    // java: GuiIngame.java:64 — `recordIsPlaying`
    pub record_is_playing: bool,
    /// The `/title` text, if any.
    pub title: String,
    /// The `/title` subtitle.
    pub subtitle: String,
    /// Remaining ticks of the title animation.
    pub title_timer: i32,
    /// Fade-in ticks. java: `titleFadeIn`, default 10
    pub title_fade_in: i32,
    /// Hold ticks. java: `titleDisplayTime`, default 70
    pub title_display: i32,
    /// Fade-out ticks. java: `titleFadeOut`, default 20
    pub title_fade_out: i32,
    /// The last boss accepted by the entity render pass. Java keeps this in the global
    /// `BossStatus` singleton, where the last renderer in one frame overwrites earlier bosses.
    boss_name: Option<String>,
    /// `BossStatus.healthScale`; intentionally not clamped because Java trusts the entity values.
    boss_health_scale: f32,
    /// Rendered-frame lifetime. `renderBossHealth` decrements this inside the draw call, not in
    /// `GuiIngame.updateTick`, so a 100-frame status remains visible on frame 100.
    boss_status_bar_time: Cell<i32>,
    /// Withers request the boss lightmap modifier while dragons do not.
    boss_has_color_modifier: bool,
    /// The `boss_bar` mod: the bar is painted by a mod HUD element, so
    /// [`Hud::draw_with_spectator_and_crosshair`] skips it. The caller then owns the lifetime
    /// countdown through [`Hud::consume_boss_frame`]. `false` (the default) is vanilla.
    boss_bar_detached: bool,
    /// The `crosshair` mod: drawn instead of the vanilla sprite when `Some`. The client
    /// sets it from the mod settings; `None` (the default) is vanilla's draw, untouched.
    pub custom_crosshair: Option<crate::modvisual::CustomCrosshair>,
    /// The `action_bar` mod: the overlay message is painted by a mod HUD element,
    /// so [`Hud::draw_with_spectator_and_crosshair`] skips it. Its lifetime still runs in
    /// [`Hud::tick`]. `false` (the default) is vanilla.
    pub action_bar_detached: bool,
    /// The `titles` mod: the title and subtitle are painted by a mod HUD element,
    /// so the vanilla draw skips them. The timers are untouched. `false` (the default) is vanilla.
    pub titles_detached: bool,
    /// Optional independent resolution for the hotbar, stats, XP and attached captions.
    /// The crosshair, boss bar, titles and other overlays retain the caller's resolution.
    pub hotbar_resolution: Option<crate::widget::ScaledResolution>,
}

impl Default for Hud {
    /// Construct the same zeroed runtime state as Java's `GuiIngame` constructor, including its
    /// default title timings.  `Hud` is a public snapshot-facing type, so callers are allowed to
    /// use either `Hud::new()` or `Hud::default()`; both must produce a title that can actually
    /// be displayed.
    fn default() -> Self {
        Self {
            update_counter: 0,
            player_health: 0,
            last_player_health: 0,
            last_health_system_time_ms: 0,
            health_update_counter: 0,
            health_flash_active: false,
            hide_lost_health_hearts: false,
            held_item_name: String::new(),
            highlight_ticks: 0,
            highlighting: None,
            record_playing: String::new(),
            record_ticks: 0,
            record_is_playing: false,
            title: String::new(),
            subtitle: String::new(),
            title_timer: 0,
            title_fade_in: 10,
            title_display: 70,
            title_fade_out: 20,
            boss_name: None,
            boss_health_scale: 0.0,
            boss_status_bar_time: Cell::new(0),
            boss_has_color_modifier: false,
            boss_bar_detached: false,
            custom_crosshair: None,
            action_bar_detached: false,
            titles_detached: false,
            hotbar_resolution: None,
        }
    }
}

impl Hud {
    /// A HUD with vanilla's default title timings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Reproduce `BossStatus.setBossStatus` for one accepted boss render.
    // java: BossStatus.java:10-16
    pub fn set_boss_status(
        &mut self,
        health: f32,
        max_health: f32,
        name: impl Into<String>,
        has_color_modifier: bool,
    ) {
        self.boss_health_scale = health / max_health;
        self.boss_status_bar_time.set(100);
        self.boss_name = Some(name.into());
        self.boss_has_color_modifier = has_color_modifier;
    }

    /// Clear world-owned boss state when the client installs or drops a world.
    pub fn clear_boss_status(&mut self) {
        self.boss_name = None;
        self.boss_health_scale = 0.0;
        self.boss_status_bar_time.set(0);
        self.boss_has_color_modifier = false;
    }

    /// Remaining rendered frames in the current boss status.
    #[must_use]
    pub fn boss_status_bar_time(&self) -> i32 {
        self.boss_status_bar_time.get()
    }

    /// The current last-writer boss label.
    #[must_use]
    pub fn boss_name(&self) -> Option<&str> {
        self.boss_name.as_deref()
    }

    /// The raw current/max boss-health ratio.
    #[must_use]
    pub fn boss_health_scale(&self) -> f32 {
        self.boss_health_scale
    }

    /// Whether the current last-writer boss requests the lightmap colour modifier.
    #[must_use]
    pub fn boss_has_color_modifier(&self) -> bool {
        self.boss_has_color_modifier
    }

    /// Hand the boss bar to a mod HUD element (`true`) or draw it here (`false`,
    /// vanilla). While detached, the caller must call [`Hud::consume_boss_frame`] once per
    /// rendered HUD frame in place of the draw.
    pub fn set_boss_bar_detached(&mut self, detached: bool) {
        self.boss_bar_detached = detached;
    }

    /// Whether the boss bar is painted by a mod HUD element (see [`Hud::set_boss_bar_detached`]).
    #[must_use]
    pub fn boss_bar_detached(&self) -> bool {
        self.boss_bar_detached
    }

    /// The lifetime half of [`Hud::draw_boss_health`] without the paint: consume one rendered
    /// frame and return whether vanilla would draw the bar on this frame. Exactly the same
    /// countdown, so a hidden or moved bar expires on the same frame as vanilla's.
    pub fn consume_boss_frame(&self) -> bool {
        if self.boss_name.is_none() {
            return false;
        }
        let remaining = self.boss_status_bar_time.get();
        if remaining <= 0 {
            return false;
        }
        self.boss_status_bar_time.set(remaining - 1);
        true
    }

    /// One client tick.
    // java: GuiIngame.java:1068 (updateTick)
    pub fn tick(&mut self) {
        mc_core::profile_scope!("hud.tick");
        self.update_counter = self.update_counter.wrapping_add(1);
        // java: GuiIngame.java:1070 — the overlay message counts down in `updateTick`, before
        // the held-item block below.
        if self.record_ticks > 0 {
            self.record_ticks -= 1;
        }
        if self.highlight_ticks > 0 {
            self.highlight_ticks -= 1;
        }
        if self.title_timer > 0 {
            self.title_timer -= 1;
            if self.title_timer == 0 {
                self.title.clear();
                self.subtitle.clear();
            }
        }
    }

    /// Forget the damage flash and when health last changed, as a fresh `GuiIngame` has them.
    ///
    /// A replay runs the HUD's tick counter on the recording's
    /// ticks, so a flash counter left from before it opened (or from later in the replay, before
    /// a backward seek) sat far ahead of the counter and blinked the hearts for hundreds of
    /// ticks that did not blink live.
    pub fn reset_health_flash(&mut self) {
        self.health_update_counter = 0;
        self.health_flash_active = false;
        self.last_health_system_time_ms = 0;
    }

    /// Observe the health state used by `GuiIngame.renderPlayerStats` for its damaged-heart
    /// animation. Call once immediately before each HUD render with the local player's live
    /// `hurtResistantTime` and a monotonic millisecond clock.
    // java: GuiIngame.java:614-636
    pub fn observe_player_health(&mut self, health: f32, hurt_resistant_time: i32, now_ms: u64) {
        let health = mc_math::ceil_f32(health);
        self.health_flash_active = self.health_update_counter > i64::from(self.update_counter)
            && (self.health_update_counter - i64::from(self.update_counter)) / 3 % 2 == 1;

        if health < self.player_health && hurt_resistant_time > 0 {
            self.last_health_system_time_ms = now_ms;
            // Java adds while `updateCounter` is an int, then widens its wrapped result to long.
            self.health_update_counter = i64::from(self.update_counter.wrapping_add(20));
        } else if health > self.player_health && hurt_resistant_time > 0 {
            self.last_health_system_time_ms = now_ms;
            self.health_update_counter = i64::from(self.update_counter.wrapping_add(10));
        }

        if now_ms.wrapping_sub(self.last_health_system_time_ms) > 1_000 {
            self.player_health = health;
            self.last_player_health = health;
            self.last_health_system_time_ms = now_ms;
        }

        self.player_health = health;
    }

    /// Feed the currently held stack, which is what raises and re-raises the name popup.
    ///
    /// `display_name` is the already-resolved, already-formatted label (`getDisplayName()`, with
    /// `§o` prepended when the stack `hasDisplayName()`); resolving it needs a
    /// [`Language`](mc_assets::Language) and the stack's NBT, neither of which the HUD owns.
    /// [`tooltip::build`](crate::screens::container::tooltip::build)`(stack, false)[0]` is
    /// exactly that string — it is the same `getDisplayName()` call, italic prefix included.
    ///
    /// # Contract: call this *after* [`Hud::tick`], once per client tick
    ///
    /// Vanilla does all of this inside one `updateTick` block, decrement included. Here the
    /// decrement stays in [`Hud::tick`] — so a caller with no inventory feed still sees the
    /// popup fade — and this method only does the two *assignments*. The net per-tick effect is
    /// identical in every branch, which
    /// `the_split_between_tick_and_update_held_item_matches_vanillas_single_block` pins:
    /// vanilla decrements only on the unchanged-stack branch, and on the other two branches it
    /// overwrites the counter with `0` or `40` anyway, so an extra earlier decrement is invisible.
    // java: GuiIngame.java:1089 (updateTick)
    pub fn update_held_item(&mut self, current: Option<&ItemStack>, display_name: &str) {
        match current {
            None => self.highlight_ticks = 0,
            Some(cur) => {
                if !self
                    .highlighting
                    .as_ref()
                    .is_some_and(|prev| Self::same_highlight(cur, prev))
                {
                    self.highlight_ticks = 40;
                }
                self.held_item_name = display_name.to_owned();
            }
        }
        self.highlighting = current.cloned();
    }

    /// Whether two stacks count as "the same item" for the name popup.
    ///
    /// **Vanilla quirk, reproduced:** the metadata comparison is skipped entirely for a
    /// *damageable* item — `itemstack.isItemStackDamageable() || metadata == metadata` — so
    /// chipping a sword's durability does not re-raise its name, while chipping a
    /// non-damageable subtype (a different wool colour, say) does. The short-circuit tests the
    /// **new** stack's damageability, not the old one's.
    // java: GuiIngame.java:1097
    #[must_use]
    fn same_highlight(current: &ItemStack, previous: &ItemStack) -> bool {
        current.item_id == previous.item_id
            && stack_ext::tags_equal(Some(current), Some(previous))
            && (stack_ext::is_stack_damageable(current) || current.damage == previous.damage)
    }

    /// Raise the overlay message above the hotbar for 60 ticks.
    ///
    /// `is_playing` is vanilla's jukebox flag: it rainbow-cycles the colour. A chat packet with
    /// position `2` (the action bar) passes `false`, which is `handleChat`'s own argument.
    // java: GuiIngame.java:1118 (setRecordPlaying)
    pub fn set_record_playing(&mut self, message: impl Into<String>, is_playing: bool) {
        self.record_playing = message.into();
        self.record_ticks = 60;
        self.record_is_playing = is_playing;
    }

    /// The overlay message's colour and alpha at `partial_ticks`, or `None` when invisible.
    ///
    /// **Vanilla quirk, reproduced:** the alpha divisor is `20` while the counter starts at
    /// `60`, so the message is fully opaque for its first 40 ticks and only then fades — and the
    /// `> 8` cut-off means it vanishes a little before the counter reaches zero.
    // java: GuiIngame.java:245
    #[must_use]
    pub fn record_color(&self, partial_ticks: f32) -> Option<i32> {
        if self.record_ticks <= 0 {
            return None;
        }
        let remaining = self.record_ticks as f32 - partial_ticks;
        let alpha = ((remaining * 255.0 / 20.0) as i32).min(255);
        if alpha <= 8 {
            return None;
        }
        let rgb = if self.record_is_playing {
            mc_math::hsv_to_rgb(remaining / 50.0, 0.7, 0.6) & 0x00FF_FFFF
        } else {
            0x00FF_FFFF
        };
        // java: `l + (l1 << 24 & -16777216)` — the shift is masked, not clamped.
        Some(rgb + ((alpha << 24) & 0xFF00_0000_u32 as i32))
    }

    /// Show a `/title` and subtitle together.
    ///
    /// This convenience entry point predates the packet's five distinct actions. Production
    /// packet handling should use [`Self::set_title`], [`Self::set_subtitle`],
    /// [`Self::set_title_times`], [`Self::clear_title`], and [`Self::reset_title`] so the odd
    /// branch behaviour of `GuiIngame.displayTitle` remains observable.
    pub fn display_title(&mut self, title: impl Into<String>, subtitle: impl Into<String>) {
        self.subtitle = subtitle.into();
        self.set_title(title);
    }

    /// Apply the `TITLE` action, retaining the previously supplied subtitle.
    // java: GuiIngame.displayTitle:1133-1137
    pub fn set_title(&mut self, title: impl Into<String>) {
        self.title = title.into();
        self.title_timer = self.title_duration();
    }

    /// Apply the `SUBTITLE` action without starting or restarting the timer.
    // java: GuiIngame.displayTitle:1138-1141
    pub fn set_subtitle(&mut self, subtitle: impl Into<String>) {
        self.subtitle = subtitle.into();
    }

    /// Apply the `TIMES` action. Negative fields retain their previous values.
    ///
    /// **Vanilla quirk, reproduced:** `handleTitle` passes a `TIMES` packet to
    /// `displayTitle(null, null, fadeIn, stay, fadeOut)`, and that method's *first* branch is
    /// `title == null && subTitle == null && all three < 0`. So three negative times are not a
    /// no-op: they clear the title, the subtitle and the timer, and keep the old timings.
    // java: GuiIngame.displayTitle:1127-1132, :1144-1162; NetHandlerPlayClient.handleTitle:1589
    pub fn set_title_times(&mut self, fade_in: i32, display: i32, fade_out: i32) {
        if fade_in < 0 && display < 0 && fade_out < 0 {
            self.title.clear();
            self.subtitle.clear();
            self.title_timer = 0;
            return;
        }
        if fade_in >= 0 {
            self.title_fade_in = fade_in;
        }
        if display >= 0 {
            self.title_display = display;
        }
        if fade_out >= 0 {
            self.title_fade_out = fade_out;
        }
        if self.title_timer > 0 {
            self.title_timer = self.title_duration();
        }
    }

    /// Apply the packet's `CLEAR` action.
    ///
    /// Vanilla routes this through `displayTitle(null, null, 0, 0, 0)`: the strings remain in
    /// memory, all three timings become zero, and an active timer becomes zero. This lasting
    /// zero-timing quirk is intentional.
    pub fn clear_title(&mut self) {
        self.set_title_times(0, 0, 0);
    }

    /// Apply the packet's `RESET` action, including its counter-intuitive subtitle retention.
    ///
    /// `handleTitle` first calls `displayTitle("", "", -1, -1, -1)`. The non-null title wins
    /// that method's `else if` chain, so the subtitle argument is ignored and the old timing sum
    /// starts again. Only then are the default timings restored.
    pub fn reset_title(&mut self) {
        self.title.clear();
        self.title_timer = self.title_duration();
        self.title_fade_in = 10;
        self.title_display = 70;
        self.title_fade_out = 20;
    }

    /// Java `int` addition wraps before the title duration is assigned or cast to `float`.
    fn title_duration(&self) -> i32 {
        self.title_fade_in
            .wrapping_add(self.title_display)
            .wrapping_add(self.title_fade_out)
    }

    /// The title's alpha at `partial_ticks`, or `None` when it is invisible.
    // java: GuiIngame.java:280
    #[must_use]
    pub fn title_alpha(&self, partial_ticks: f32) -> Option<i32> {
        if self.title_timer <= 0 {
            return None;
        }
        let remaining = self.title_timer as f32 - partial_ticks;
        let mut alpha = 255.0;
        if self.title_timer > self.title_fade_out.wrapping_add(self.title_display) {
            let elapsed = self.title_duration() as f32 - remaining;
            alpha = elapsed * 255.0 / self.title_fade_in as f32;
        }
        if self.title_timer <= self.title_fade_out {
            alpha = remaining * 255.0 / self.title_fade_out as f32;
        }
        let alpha = (alpha as i32).clamp(0, 255);
        (alpha > 8).then_some(alpha)
    }

    /// Draw the ordinary first-person crosshair.
    ///
    /// This compatibility entry point has the default, non-spectator [`CrosshairState`].  A
    /// world-aware caller must use [`Self::draw_crosshair_for`] so spectator targeting and the
    /// debug-info gate are not lost at the UI boundary.
    // java: GuiIngame.java:179
    pub fn draw_crosshair(&self, d: &mut DrawList, ctx: &ScreenCtx) {
        self.draw_crosshair_for(d, ctx, CrosshairState::default());
    }

    /// Draw the crosshair when [`CrosshairState::shows_crosshair`] permits it.
    ///
    /// Vanilla temporarily switches to `ONE_MINUS_DST_COLOR, ONE_MINUS_SRC_COLOR` for exactly
    /// this sprite.  [`DrawList::blit_invert`] carries that state in the batch key, so adjacent
    /// HUD sprites remain ordinary alpha blended.
    // java: GuiIngame.java:179-185, :338-355
    pub fn draw_crosshair_for(&self, d: &mut DrawList, ctx: &ScreenCtx, state: CrosshairState) {
        if !state.shows_crosshair() {
            return;
        }
        // The `crosshair` mod: the custom crosshair replaces the sprite, behind the same
        // `showCrosshair` gate, plus its own third-person switch.
        if let Some(custom) = &self.custom_crosshair {
            if custom.show_in_f5 || !state.third_person {
                custom.draw(d, ctx);
            }
            return;
        }
        d.blit_invert(
            UiTexture::icons(),
            ctx.width / 2 - 7,
            ctx.height / 2 - 7,
            0,
            0,
            16,
            16,
        );
    }

    /// Draw the one-slot boss bar and consume one rendered frame of its lifetime.
    // java: GuiIngame.java:901-925
    pub fn draw_boss_health(&self, d: &mut DrawList, font: &mut FontRenderer, ctx: &ScreenCtx) {
        let Some(name) = self.boss_name.as_deref() else {
            return;
        };
        let remaining = self.boss_status_bar_time.get();
        if remaining <= 0 {
            return;
        }
        self.boss_status_bar_time.set(remaining - 1);
        paint_boss_bar(
            d,
            font,
            ctx.width / 2,
            2,
            name,
            self.boss_health_scale,
            BossBarStyle::VANILLA,
        );
    }

    /// Draw the hotbar frame and its selection cursor.
    // java: GuiIngame.java:365 (renderTooltip)
    pub fn draw_hotbar(&self, d: &mut DrawList, ctx: &ScreenCtx, state: &PlayerHud) {
        let tex = UiTexture::widgets();
        let cx = ctx.width / 2;
        d.blit(tex, cx - 91, ctx.height - 22, 0, 0, 182, 22);
        d.blit(
            tex,
            cx - 91 - 1 + state.selected_slot.clamp(0, 8) * 20,
            ctx.height - 22 - 1,
            0,
            22,
            24,
            22,
        );
    }

    /// Where hotbar item `slot` draws its icon.
    // java: GuiIngame.java:385
    #[must_use]
    pub fn hotbar_item_pos(ctx: &ScreenCtx, slot: i32) -> (i32, i32) {
        (ctx.width / 2 - 90 + slot * 20 + 2, ctx.height - 16 - 3)
    }

    /// Draw the nine hotbar stacks on top of the frame [`Hud::draw_hotbar`] laid down.
    ///
    /// `atlas_texture` is whatever [`UiTexture`] key the caller uploaded
    /// `mc_assets::BakedAssets::atlas`'s pixels under — the same key S17's container screens
    /// pass to [`draw_item_in_slot`], because it is literally the same call. Empty slots draw
    /// nothing, exactly as `renderHotbarItem`'s `itemstack != null` guard does.
    ///
    /// `renderHotbarItem`'s pop animation squashes a slot for a few ticks after its stack changes.
    /// The animation uses `scale(1/f1, (f1 + 1)/2, 1)` about the slot centre. It is fed by
    /// [`PlayerHud::animations_to_go`], and `DrawList`'s transform stack keeps the overlay text
    /// and durability bar outside that temporary matrix.
    // java: GuiIngame.java:383 (renderTooltip's slot loop) and :1037 (renderHotbarItem)
    pub fn draw_hotbar_items(
        &self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        atlas_texture: &UiTexture,
        ctx: &ScreenCtx,
        state: &PlayerHud,
    ) {
        self.draw_hotbar_items_in_order(d, font, atlas_texture, ctx, state, nw_f3_passes_enabled());
    }

    /// [`Self::draw_hotbar_items`] with an explicit draw order. With `passes`,
    /// every icon is drawn, then every enchantment glint, then every overlay (durability bar,
    /// stack count), instead of the three slot by slot: three UI batches instead of up to ~12
    /// (each icon/overlay switch is a texture and depth-window change), the same pixels.
    ///
    /// Why the pixels are the same: per slot, its own icon, glint and overlays still come in that
    /// order, and no slot's geometry reaches another slot's icon or overlays except as below.
    /// Icons are 16 px in 20 px slots (the pop animation only narrows an icon horizontally,
    /// `1/f1`, and stretches it vertically, where no other slot is); the durability bar spans
    /// `x+2..x+15`; the count is right-aligned at `x+17` (its shadow at `x+18`), so it can only
    /// reach *earlier* slots, leftwards, with four or more digits, and those slots' icons and
    /// glints come before it in both orders. The glint's `GL_EQUAL` test finds its own icon's
    /// depth either way. `tests/nw_f3_draw_passes.rs` renders hotbars both ways and compares the
    /// frames byte for byte.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_hotbar_items_in_order(
        &self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        atlas_texture: &UiTexture,
        ctx: &ScreenCtx,
        state: &PlayerHud,
        passes: bool,
    ) {
        if passes {
            for parts in [ItemIconParts::Model, ItemIconParts::Glint] {
                for slot in 0..9 {
                    let Some(stack) = state.hotbar[slot as usize].as_ref() else {
                        continue;
                    };
                    let (x, y) = Self::hotbar_item_pos(ctx, slot);
                    let animation = state.animations_to_go[slot as usize];
                    let white = 0xFFFF_FFFF;
                    match hotbar_item_pop_scale(animation, ctx.partial_ticks) {
                        Some((scale_x, scale_y)) => {
                            d.with_transform(
                                hotbar_item_pop_transform(x, y, scale_x, scale_y),
                                |d| {
                                    draw_item_icon_parts(
                                        d,
                                        atlas_texture,
                                        x,
                                        y,
                                        stack,
                                        white,
                                        parts,
                                    );
                                },
                            );
                        }
                        None => draw_item_icon_parts(d, atlas_texture, x, y, stack, white, parts),
                    }
                }
            }
            for slot in 0..9 {
                let Some(stack) = state.hotbar[slot as usize].as_ref() else {
                    continue;
                };
                let (x, y) = Self::hotbar_item_pos(ctx, slot);
                // java: renderItemOverlays is after the animation matrix is popped.
                draw_durability_bar(d, x, y, stack);
                draw_count_text(d, font, x, y, stack);
            }
            return;
        }
        for slot in 0..9 {
            let Some(stack) = state.hotbar[slot as usize].as_ref() else {
                continue;
            };
            let (x, y) = Self::hotbar_item_pos(ctx, slot);
            let animation = state.animations_to_go[slot as usize];
            if let Some((scale_x, scale_y)) = hotbar_item_pop_scale(animation, ctx.partial_ticks) {
                d.with_transform(hotbar_item_pop_transform(x, y, scale_x, scale_y), |d| {
                    draw_item_icon_tinted(d, atlas_texture, x, y, stack, 0xFFFF_FFFF)
                });
            } else {
                draw_item_icon_tinted(d, atlas_texture, x, y, stack, 0xFFFF_FFFF);
            }

            // java: renderItemOverlays is after the animation matrix is popped.
            draw_durability_bar(d, x, y, stack);
            draw_count_text(d, font, x, y, stack);
        }
    }

    /// Draw the XP bar and the level number.
    // java: GuiIngame.java:414 (renderExpBar)
    pub fn draw_experience(
        &self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        state: &PlayerHud,
    ) {
        if !state.survival {
            return;
        }
        let tex = UiTexture::icons();
        let x = ctx.width / 2 - 91;
        let y = ctx.height - 32 + 3;
        d.blit(tex, x, y, 0, 64, 182, 5);
        // Vanilla trusts the server's raw S1F float; values outside [0, 1] intentionally draw
        // outside the usual 183-pixel range.
        let filled = (state.experience * 183.0) as i32;
        if filled > 0 {
            d.blit(tex, x, y, 0, 69, filled, 5);
        }
        if state.experience_level > 0 {
            let text = state.experience_level.to_string();
            let lx = ctx.width / 2;
            let ly = ctx.height - 31 - 4;
            // Vanilla draws four black copies offset by one pixel, then the green centre — a
            // hand-rolled outline, not a drop shadow.
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                draw_outline_string(d, font, &text, lx + dx, ly + dy, 0);
            }
            draw_outline_string(d, font, &text, lx, ly, 8_453_920);
        }
    }

    /// Draw the mounted-horse jump bar in the XP bar's slot.
    // java: GuiIngame.java:438-455 (renderHorseJumpBar)
    pub fn draw_horse_jump_bar(&self, d: &mut DrawList, ctx: &ScreenCtx, jump_power: f32) {
        let tex = UiTexture::icons();
        let x = ctx.width / 2 - 91;
        let y = ctx.height - 32 + 3;
        d.blit(tex, x, y, 0, 84, 182, 5);
        let filled = (jump_power * 183.0) as i32;
        if filled > 0 {
            d.blit(tex, x, y, 0, 89, filled, 5);
        }
    }

    /// Select Java's mutually exclusive mounted-horse or experience bottom bar.
    pub fn draw_experience_or_horse(
        &self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        state: &PlayerHud,
    ) {
        if let Some(jump_power) = state.horse_jump_power {
            self.draw_horse_jump_bar(d, ctx, jump_power);
        } else {
            self.draw_experience(d, font, ctx, state);
        }
    }

    /// Draw the armour, health, hunger and air rows.
    // java: GuiIngame.java:609 (renderPlayerStats)
    pub fn draw_player_stats(&self, d: &mut DrawList, ctx: &ScreenCtx, state: &PlayerHud) {
        if !state.survival {
            return;
        }
        let tex = UiTexture::icons();
        let left = ctx.width / 2 - 91;
        let right = ctx.width / 2 + 91;
        let base_y = ctx.height - 39;
        let rows = heart_rows(state.max_health, state.absorption);
        let pitch = heart_row_pitch(rows);
        let top_y = armor_row_y(base_y, rows, pitch);

        if state.armor > 0 {
            for slot in 0..10 {
                d.blit(
                    tex,
                    left + slot * 8,
                    top_y,
                    armor_sprite_u(slot, state.armor),
                    9,
                    9,
                    9,
                );
            }
        }

        // Hearts, drawn from the last one backwards so earlier ones overlap later ones.
        let health = mc_math::ceil_f32(state.health);
        // java: `updateCounter % ceiling_float_int(maxHealth + 5)`. A NaN
        // max health (an S20 attribute base value) makes the divisor 0, which throws in Java and
        // panicked here: a crash on the next Regeneration effect. No heart is lifted instead
        // (with a NaN max health no heart is drawn either).
        let regen_lift = if state.regenerating {
            self.update_counter
                .checked_rem(mc_math::ceil_f32(state.max_health + 5.0))
                .unwrap_or(-1)
        } else {
            -1
        };
        let family = heart_family_u(state.poisoned, state.withered);
        let v = heart_row_v(state.hardcore);
        // The absorption amount is a raw float from the player's metadata
        // (DataWatcher index 17), so a server can claim 2e9 and make this loop draw a billion
        // hearts per frame: vanilla freezes, and here the draw list grew until memory ran out.
        // Legitimate state stays far below the cap: max health is clamped to 1024 and an
        // Absorption 255 effect adds 1024 more, 1024 hearts in all, which draw exactly as before.
        let total =
            mc_math::ceil_f32((state.max_health + state.absorption) / 2.0).min(MAX_DRAWN_HEARTS);
        let mut absorption_left = state.absorption;
        let mut rng = mc_math::JavaRandom::new(hud_random_seed(self.update_counter));
        for index in (0..total).rev() {
            let row = mc_math::ceil_f32((index + 1) as f32 / 10.0) - 1;
            let x = left + (index % 10) * 8;
            let mut y = base_y - row * pitch;
            if health <= 4 {
                // java: `j4 += this.rand.nextInt(2)` — the low-health shake.
                y += rng.next_i32_bound(2);
            }
            if index == regen_lift {
                y -= 2;
            }
            // The container: `16 + k3 * 9`, where k3 is the damage flash (not modelled — it is
            // driven by `hurtResistantTime`, which is S14's to surface).
            d.blit(
                tex,
                x,
                y,
                16 + if self.health_flash_active { 9 } else { 0 },
                v,
                9,
                9,
            );
            // `hide_lost_health_hearts` is false unless the mod sets it.
            if self.health_flash_active && !self.hide_lost_health_hearts {
                if index * 2 + 1 < self.last_player_health {
                    d.blit(tex, x, y, family + 54, v, 9, 9);
                } else if index * 2 + 1 == self.last_player_health {
                    d.blit(tex, x, y, family + 63, v, 9, 9);
                }
            }
            if absorption_left > 0.0 {
                let u = if absorption_left == state.absorption && state.absorption % 2.0 == 1.0 {
                    family + 153
                } else {
                    family + 144
                };
                d.blit(tex, x, y, u, v, 9, 9);
                absorption_left -= 2.0;
            } else if index * 2 + 1 < health {
                d.blit(tex, x, y, family + 36, v, 9, 9);
            } else if index * 2 + 1 == health {
                d.blit(tex, x, y, family + 45, v, 9, 9);
            }
        }

        // Hunger, drawn right to left.
        let food_family = if state.hungry { 16 + 36 } else { 16 };
        let food_container = if state.hungry { 13 } else { 0 };
        for i in 0..10 {
            let mut y = base_y;
            // java: `updateCounter % (foodLevel * 3 + 1)` in wrapping
            // `int` arithmetic. The S06 food level is any `int`, and 1431655765 wraps the
            // divisor to 0, which throws in Java and panicked here; that one value never shakes.
            let shake_period = state.food.wrapping_mul(3).wrapping_add(1);
            if state.saturation <= 0.0 && self.update_counter.checked_rem(shake_period) == Some(0) {
                y = base_y + (rng.next_i32_bound(3) - 1);
            }
            let x = right - i * 8 - 9;
            d.blit(tex, x, y, 16 + food_container * 9, 27, 9, 9);
            if i * 2 + 1 < state.food {
                d.blit(tex, x, y, food_family + 36, 27, 9, 9);
            } else if i * 2 + 1 == state.food {
                d.blit(tex, x, y, food_family + 45, 27, 9, 9);
            }
        }

        if state.underwater {
            let (full, popping) = air_bubbles(state.air);
            for i in 0..full + popping {
                let u = if i < full { 16 } else { 25 };
                d.blit(tex, right - i * 8 - 9, top_y, u, 18, 9, 9);
            }
        }
    }

    /// Draw the full-screen sleep fade, if the player's live timer is nonzero.
    ///
    /// This is a solid GUI rectangle rather than a world/post-processing effect. In vanilla it
    /// is emitted after player stats and before the XP bar, so the latter remains visible above
    /// the fade.
    // java: GuiIngame.java:194-211
    pub fn draw_sleep_overlay(&self, d: &mut DrawList, ctx: &ScreenCtx, sleep_timer: i32) {
        if let Some(color) = sleep_overlay_color(sleep_timer) {
            d.rect(0, 0, ctx.width, ctx.height, color);
        }
    }

    /// Where a potion icon and its two text lines go, relative to `(x, y)`.
    ///
    /// Vanilla uses a 33-pixel pitch for at most five effects, then compresses the whole column
    /// into 132 pixels. The first effect starts at `y`; the icon is at `x + 6` and the name at
    /// `x + 10 + 18`.
    // java: InventoryEffectRenderer.java:60 (drawActivePotionEffects)
    #[must_use]
    pub fn effect_row_y(y: i32, index: usize, effect_count: usize) -> i32 {
        let pitch = if effect_count > 5 {
            132 / (effect_count as i32 - 1)
        } else {
            33
        };
        y + index as i32 * pitch
    }

    /// Draw the active-effect list. See [`PlayerHud::show_effect_overlay`].
    pub fn draw_effects(
        &self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        x: i32,
        y: i32,
        effects: &[ActiveEffect],
    ) {
        let tex = UiTexture::gui(INVENTORY_TEXTURE);
        for (i, e) in effects.iter().enumerate() {
            let row_y = Self::effect_row_y(y, i, effects.len());
            d.blit(&tex, x, row_y, 0, 166, 140, 32);
            if e.icon_index >= 0 {
                d.blit(
                    &tex,
                    x + 6,
                    row_y + 7,
                    (e.icon_index % 8) * 18,
                    198 + (e.icon_index / 8) * 18,
                    18,
                    18,
                );
            }
            font.draw_string_with_shadow(
                d,
                &e.name,
                (x + 10 + 18) as f32,
                (row_y + 6) as f32,
                0x00FF_FFFF,
            );
            font.draw_string_with_shadow(
                d,
                &e.duration,
                (x + 10 + 18) as f32,
                (row_y + 6 + 10) as f32,
                0x007F_7F7F,
            );
        }
    }

    /// The scoreboard sidebar. java: `GuiIngame.renderScoreboard` (`:551`).
    ///
    /// `sidebar.rows` must arrive in `Scoreboard.getSortedScores` order — **ascending** by points
    /// — because the row walk below is vanilla's: it takes the *tail* of the list and steps
    /// upward from `height / 2 + rows * 9 / 3`, which is what puts the highest score at the top.
    ///
    /// Every number here is vanilla's: the `0x50000000` row plate, the `0x60000000` header plate
    /// with a second `0x50000000` sliver under it, the `0x20FFFFFF` **shadowless** text, the
    /// 3-pixel right margin, and the box widened to the widest `"<name>: §c<points>"` — measured
    /// with the colon and the score, then drawn as two separate strings.
    pub fn draw_scoreboard(
        &self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        sidebar: &Sidebar,
    ) {
        mc_core::profile_scope!("hud.sidebar");
        let rows = sidebar.visible_rows();
        if rows.is_empty() {
            return;
        }
        let width = sidebar_text_width(font, sidebar, true);
        let margin = 3;
        let content_height = rows.len() as i32 * font.font_height();
        let baseline = ctx.height / 2 + content_height / 3;
        let left = ctx.width - width - margin;
        // The painting itself is shared with the `scoreboard` mod's HUD element, so
        // a fix to the vanilla draw reaches both.
        paint_sidebar(
            d,
            font,
            sidebar,
            left,
            baseline,
            width,
            SidebarStyle::VANILLA,
        );
    }

    /// Draw everything.
    ///
    /// `atlas_texture` names the uploaded block/item atlas the hotbar's icons sample. Pass
    /// `None` before assets are loaded: the hotbar then draws its frame and selection cursor and
    /// no items, which is the pre-S21 state, not a supported look.
    pub fn draw(
        &self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        atlas_texture: Option<&UiTexture>,
        ctx: &ScreenCtx,
        state: &PlayerHud,
    ) {
        self.draw_with_spectator(d, font, atlas_texture, ctx, state, false);
    }

    /// Draw the HUD with the ordinary player-only layers replaced by the spectator bar.
    ///
    /// This method does not draw the spectator bar itself: the caller owns its menu lifetime and
    /// draws it after this method. Passing `true` suppresses exactly the four branches vanilla
    /// replaces in spectator mode: the item hotbar, player stats, XP, and held-item caption.
    /// Crosshair (subject to the default `showCrosshair` snapshot), action-bar message,
    /// title/subtitle, and optional effect overlay remain visible.
    ///
    /// [`Hud::draw`] remains the non-spectator convenience entry point so callers which do not
    /// have a game-mode concept keep their existing behaviour.
    // java: GuiIngame.java:162-169, :190-196, :213-235
    pub fn draw_with_spectator(
        &self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        atlas_texture: Option<&UiTexture>,
        ctx: &ScreenCtx,
        state: &PlayerHud,
        spectator: bool,
    ) {
        self.draw_with_spectator_and_crosshair(
            d,
            font,
            atlas_texture,
            ctx,
            state,
            spectator,
            CrosshairState {
                spectator,
                ..CrosshairState::default()
            },
        );
    }

    /// Draw the HUD with a caller-supplied vanilla `showCrosshair` snapshot.
    ///
    /// The ordinary [`Self::draw_with_spectator`] entry point remains source-compatible for
    /// callers that only know the game mode; it supplies the conservative spectator/no-target
    /// default.  The in-game client uses this method after its render-time mouse-over pass so
    /// ItemFrame, inventory-block, F3 and reduced-debug inputs reach the crosshair gate.
    // java: GuiIngame.java:162-185, :513-543
    #[allow(clippy::too_many_arguments)]
    pub fn draw_with_spectator_and_crosshair(
        &self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        atlas_texture: Option<&UiTexture>,
        ctx: &ScreenCtx,
        state: &PlayerHud,
        spectator: bool,
        crosshair: CrosshairState,
    ) {
        self.draw_layers_below_debug(d, font, atlas_texture, ctx, state, spectator, crosshair);
        self.draw_layers_above_debug(d, font, ctx, state);
    }

    /// The part of `renderGameOverlay` that comes before `GuiOverlayDebug.renderDebugInfo`
    /// (`:240-243`): crosshair, boss bar, hotbar, player stats, sleep overlay, XP or horse-jump
    /// bar, and the held-item name.
    ///
    /// A client that draws F3 draws it between this and [`Self::draw_layers_above_debug`], which
    /// is vanilla's order: the action bar, the title, the sidebar, chat and the Tab list are all
    /// painted over the debug text. [`Self::draw_with_spectator_and_crosshair`] is the two halves
    /// back to back.
    // java: GuiIngame.java:162-238
    #[allow(clippy::too_many_arguments)]
    pub fn draw_layers_below_debug(
        &self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        atlas_texture: Option<&UiTexture>,
        ctx: &ScreenCtx,
        state: &PlayerHud,
        spectator: bool,
        crosshair: CrosshairState,
    ) {
        mc_core::profile_scope!("hud.draw");
        self.draw_crosshair_for(
            d,
            ctx,
            CrosshairState {
                spectator,
                ..crosshair
            },
        );
        // java: renderGameOverlay draws boss health immediately after the crosshair and before
        // player stats. It is visible in spectator mode too.
        // A detached bar is painted (and its lifetime consumed) by the `boss_bar`
        // mod instead; `false` is vanilla and the only state a client without the mod sees.
        if !self.boss_bar_detached {
            self.draw_boss_health(d, font, ctx);
        }
        let hotbar_ctx = ScreenCtx::new(
            self.hotbar_resolution.unwrap_or(ctx.resolution),
            ctx.partial_ticks,
        );
        let separate_hotbar = hotbar_ctx.resolution != ctx.resolution;
        if separate_hotbar {
            d.push_transform(hotbar_ctx.resolution.transform_to(ctx.resolution));
        }
        if !spectator {
            self.draw_hotbar(d, &hotbar_ctx, state);
            if let Some(atlas) = atlas_texture {
                self.draw_hotbar_items(d, font, atlas, &hotbar_ctx, state);
            }
            self.draw_player_stats(d, &hotbar_ctx, state);
        }
        if separate_hotbar {
            d.pop_transform();
        }
        self.draw_sleep_overlay(d, ctx, state.sleep_timer);
        if separate_hotbar {
            d.push_transform(hotbar_ctx.resolution.transform_to(ctx.resolution));
        }
        if !spectator {
            self.draw_experience_or_horse(d, font, &hotbar_ctx, state);
        }
        if !spectator && self.highlight_ticks > 0 && !self.held_item_name.is_empty() {
            let alpha = ((self.highlight_ticks as f32 * 256.0 / 10.0) as i32).min(255);
            if alpha > 0 {
                let name = &self.held_item_name;
                let x = (hotbar_ctx.width - font.string_width(name)) / 2;
                // java: `j += 14` when `PlayerControllerMP.shouldDrawHUD()` is false.
                let y = hotbar_ctx.height - 59 + if state.survival { 0 } else { 14 };
                font.draw_string_with_shadow(
                    d,
                    name,
                    x as f32,
                    y as f32,
                    0x00FF_FFFF | (alpha << 24),
                );
            }
        }
        if separate_hotbar {
            d.pop_transform();
        }
    }

    /// The part of `renderGameOverlay` between F3 and the sidebar: the action bar
    /// (`recordPlaying`, `:245-275`) and the title and subtitle (`:277-316`), followed by the
    /// optional effect overlay. See [`Self::draw_layers_below_debug`].
    // java: GuiIngame.java:245-316
    pub fn draw_layers_above_debug(
        &self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        state: &PlayerHud,
    ) {
        mc_core::profile_scope!("hud.messages");
        let hotbar_ctx = ScreenCtx::new(
            self.hotbar_resolution.unwrap_or(ctx.resolution),
            ctx.partial_ticks,
        );
        // java: GuiIngame.java:245 — `translate(i / 2, j - 68, 0)` then a string drawn at
        // `(-width / 2, -4)`, i.e. centred on `height - 72`. No shadow.
        // A detached message is painted by the `action_bar` mod element.
        if let Some(color) = self
            .record_color(ctx.partial_ticks)
            .filter(|_| !self.action_bar_detached)
        {
            if !self.record_playing.is_empty() {
                let separate_hotbar = hotbar_ctx.resolution != ctx.resolution;
                if separate_hotbar {
                    d.push_transform(hotbar_ctx.resolution.transform_to(ctx.resolution));
                }
                let message = &self.record_playing;
                let x = hotbar_ctx.width / 2 - font.string_width(message) / 2;
                font.draw_string(
                    d,
                    message,
                    x as f32,
                    (hotbar_ctx.height - 72) as f32,
                    color,
                    false,
                );
                if separate_hotbar {
                    d.pop_transform();
                }
            }
        }
        // Detached titles are painted by the `titles` mod element.
        if let Some(alpha) = self
            .title_alpha(ctx.partial_ticks)
            .filter(|_| !self.titles_detached)
        {
            let color = 0x00FF_FFFF | (alpha << 24);
            // C2-8: borrow; these were cloned every frame while shown.
            let title = &self.title;
            let subtitle = &self.subtitle;
            // java: translate(w/2, h/2), then two independent push/scale/draw/pop pairs.
            let anchor = Transform::translate((ctx.width / 2) as f32, (ctx.height / 2) as f32);
            d.push_transform(anchor.then_scale(4.0));
            draw_centered_string(d, font, title, 0, -10, color);
            d.pop_transform();
            d.push_transform(anchor.then_scale(2.0));
            draw_centered_string(d, font, subtitle, 0, 5, color);
            d.pop_transform();
        }
        if state.show_effect_overlay && !state.effects.is_empty() {
            self.draw_effects(d, font, ctx.width - 150, 0, &state.effects);
        }
    }
}

/// `EnumChatFormatting.RED + points` — `"§c"` and the decimal score — formatted into `buf`, so
/// the sidebar's per-row score string costs no heap allocation. Same bytes as `format!`.
pub(crate) fn red_score_text(buf: &mut [u8; 16], points: i32) -> &str {
    use std::io::Write as _;
    let len = {
        let mut cursor = std::io::Cursor::new(&mut buf[..]);
        write!(cursor, "\u{a7}c{points}").expect("`§c` and any i32 fit in 16 bytes");
        cursor.position() as usize
    };
    std::str::from_utf8(&buf[..len]).expect("formatted from a UTF-8 literal and ASCII digits")
}

/// Which halves of the boss bar [`paint_boss_bar`] draws, and the name's shadow.
/// [`BossBarStyle::VANILLA`] is `renderBossHealth`; the `boss_bar` mod passes its
/// options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BossBarStyle {
    /// The 182-pixel health bar.
    pub bar: bool,
    /// The boss name above it.
    pub text: bool,
    /// Draw the name with the font's drop shadow.
    pub shadow: bool,
}

impl BossBarStyle {
    /// `GuiIngame.renderBossHealth` exactly.
    pub const VANILLA: BossBarStyle = BossBarStyle {
        bar: true,
        text: true,
        shadow: true,
    };
}

/// Paint the boss bar centred on `center_x`, the name's top at `top` and the bar 10 pixels
/// below it. [`Hud::draw_boss_health`] calls this with `(width / 2, 2)` and
/// [`BossBarStyle::VANILLA`]; the `boss_bar` mod's HUD element calls it at its own origin.
// java: GuiIngame.java:901-925
#[allow(clippy::too_many_arguments)]
pub fn paint_boss_bar(
    d: &mut DrawList,
    font: &mut FontRenderer,
    center_x: i32,
    top: i32,
    name: &str,
    health_scale: f32,
    style: BossBarStyle,
) {
    if style.bar {
        let tex = UiTexture::icons();
        let width = 182;
        let x = center_x - width / 2;
        let y = top + 10;
        let filled = (health_scale * 183.0) as i32;
        // Vanilla submits the backdrop twice. Preserve the duplicate draw even though the two
        // identical alpha-blended quads look redundant.
        d.blit(tex, x, y, 0, 74, width, 5);
        d.blit(tex, x, y, 0, 74, width, 5);
        if filled > 0 {
            d.blit(tex, x, y, 0, 79, filled, 5);
        }
    }
    if style.text {
        font.draw_string(
            d,
            name,
            (center_x - font.string_width(name) / 2) as f32,
            top as f32,
            0x00FF_FFFF,
            style.shadow,
        );
    }
}

/// How [`paint_sidebar`] styles the board. [`SidebarStyle::VANILLA`] is `renderScoreboard`; the
/// `scoreboard` mod passes its options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SidebarStyle {
    /// Draw the red score column (and measure it into the width, see [`sidebar_text_width`]).
    pub numbers: bool,
    /// Multiplier on the plates' alpha, `0.0..=1.0`; `1.0` is vanilla's `0x50`/`0x60`, and a
    /// plate whose alpha comes out zero is not drawn at all.
    pub background: f32,
    /// Draw the text with the font's drop shadow. The text is then fully opaque, so the shadow
    /// is the normal dark one rather than a `0x20`-alpha ghost.
    pub shadow: bool,
}

impl SidebarStyle {
    /// `GuiIngame.renderScoreboard` exactly.
    pub const VANILLA: SidebarStyle = SidebarStyle {
        numbers: true,
        background: 1.0,
        shadow: false,
    };
}

/// The sidebar's text colour. java: `drawString(..., 553648127)` in `renderScoreboard`, i.e.
/// `0x20FFFFFF`. Vanilla draws it with blending off, so it shows opaque: [`paint_sidebar`] sends
/// it through [`UnblendedText`].
const SIDEBAR_TEXT_COLOR: i32 = 0x20FF_FFFF;

/// The sidebar's text width: the widest of the title and every visible row, a row measured as
/// one `"<name>: §c<points>"` string with `numbers`, the name alone without.
///
/// java: `int i = getStringWidth(objective.getDisplayName())`, then widened by each row's
/// *combined* `name + ": " + RED + points`, even though the two halves are drawn apart. Both
/// strings are measured without heap allocation (C2-1): the combined width is walked over its
/// parts, and the score is formatted into a stack buffer.
#[must_use]
pub fn sidebar_text_width(font: &FontRenderer, sidebar: &Sidebar, numbers: bool) -> i32 {
    let mut width = font.string_width(&sidebar.title);
    let mut score_buf = [0u8; 16];
    for (name, points) in sidebar.visible_rows() {
        let row = if numbers {
            let score = red_score_text(&mut score_buf, *points);
            font.string_width_of_parts(&[name.as_str(), ": ", score])
        } else {
            font.string_width(name)
        };
        width = width.max(row);
    }
    width
}

/// Paint the sidebar with its text column at `left` and the bottom of its lowest row at
/// `baseline` (vanilla's `l1` and `j1`); `width` is [`sidebar_text_width`] for the same
/// `numbers` choice. The plates span `left - 2 .. left + width + 2` horizontally and
/// `baseline - (rows + 1) * 9 - 1 .. baseline` vertically.
///
/// [`Hud::draw_scoreboard`] calls this at vanilla's position with [`SidebarStyle::VANILLA`]; the
/// `scoreboard` mod's HUD element calls it at its own origin, so the two share every pixel rule.
///
/// Vanilla interleaves the draws row by row upwards from `baseline` (plate, name, score, and on
/// the topmost row the header plate, the sliver under it and the title). This paints every plate
/// first and then every string. No plate overlaps another row's text: each row plate is its own
/// row's band, and the header and sliver still precede the title (whose underline or shadow
/// reaches the sliver). So the pixels are the same, and the board is two batches rather than two
/// per row, now that the vanilla text is unblended.
///
/// java: every `drawString` in `renderScoreboard` follows a `Gui.drawRect`, which ends with
/// `disableBlend()`, so the `0x20FFFFFF` text is written opaque under `FontRenderer`'s alpha
/// test: [`UnblendedText`]. The mod's `shadow` style draws opaque white with the ordinary
/// blend and shadow instead.
#[allow(clippy::too_many_arguments)]
pub fn paint_sidebar(
    d: &mut DrawList,
    font: &mut FontRenderer,
    sidebar: &Sidebar,
    left: i32,
    baseline: i32,
    width: i32,
    style: SidebarStyle,
) {
    let rows = sidebar.visible_rows();
    let Some(top_row) = i32::try_from(rows.len()).ok().filter(|&n| n > 0) else {
        return;
    };
    let font_height = font.font_height();
    let right = left + width + 2;
    let row_plate = plate_color(0x50, style.background);
    let header_plate = plate_color(0x60, style.background);
    // java: `if (j == collection.size())`: the *last* row drawn is the topmost one, and the
    // objective title's plate is stamped on top of it.
    let top = baseline - top_row * font_height;
    if row_plate != 0 {
        for row in 1..=top_row {
            let y = baseline - row * font_height;
            d.rect(left - 2, y, right, y + font_height, row_plate);
        }
    }
    if header_plate != 0 {
        d.rect(
            left - 2,
            top - font_height - 1,
            right,
            top - 1,
            header_plate,
        );
    }
    if row_plate != 0 {
        d.rect(left - 2, top - 1, right, top, row_plate);
    }
    if style.shadow {
        let text = (0xFFFF_FFFF_u32 as i32, true);
        paint_sidebar_text(d, font, sidebar, left, baseline, width, style.numbers, text);
    } else {
        let text = (SIDEBAR_TEXT_COLOR, false);
        let mut unblended = UnblendedText(d);
        paint_sidebar_text(
            &mut unblended,
            font,
            sidebar,
            left,
            baseline,
            width,
            style.numbers,
            text,
        );
    }
}

/// The strings of [`paint_sidebar`], in vanilla's order: each row's name and score upwards from
/// `baseline`, then the title. `(color, shadow)` is passed to every `drawString`.
#[allow(clippy::too_many_arguments)]
fn paint_sidebar_text(
    sink: &mut dyn TextSink,
    font: &mut FontRenderer,
    sidebar: &Sidebar,
    left: i32,
    baseline: i32,
    width: i32,
    numbers: bool,
    (color, shadow): (i32, bool),
) {
    let rows = sidebar.visible_rows();
    let font_height = font.font_height();
    let right = left + width + 2;
    let mut score_buf = [0u8; 16];
    let mut y = baseline;
    for (name, points) in rows {
        y -= font_height;
        font.draw_string(sink, name, left as f32, y as f32, color, shadow);
        if numbers {
            let score = red_score_text(&mut score_buf, *points);
            let score_x = right - font.string_width(score);
            font.draw_string(sink, score, score_x as f32, y as f32, color, shadow);
        }
    }
    let title_x = left + width / 2 - font.string_width(&sidebar.title) / 2;
    let title_y = (y - font_height) as f32;
    font.draw_string(sink, &sidebar.title, title_x as f32, title_y, color, shadow);
}

/// A black plate of vanilla alpha `alpha` scaled by `factor` (clamped to `0..=1`). At `1.0` this
/// is exactly `alpha << 24`.
fn plate_color(alpha: u32, factor: f32) -> u32 {
    let factor = if factor.is_finite() {
        factor.clamp(0.0, 1.0)
    } else {
        1.0
    };
    ((alpha as f32 * factor).round() as u32) << 24
}

/// `drawCenteredString` without the shadow, for the XP level's hand-rolled outline.
fn draw_outline_string(
    d: &mut DrawList,
    font: &mut FontRenderer,
    text: &str,
    x: i32,
    y: i32,
    color: i32,
) {
    let x = x - font.string_width(text) / 2;
    font.draw_string(d, text, x as f32, y as f32, color, false);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::draw::{ICONS_TEXTURE, WIDGETS_TEXTURE};
    use crate::widget::test_support::uniform_font;
    use crate::widget::ScaledResolution;

    #[test]
    fn independent_hotbar_scales_bottom_group_without_scaling_the_crosshair() {
        let global = ScreenCtx::new(ScaledResolution::new(1920, 1080, 3, false), 0.0);
        let small = ScaledResolution::new(1920, 1080, 1, false);
        let player = PlayerHud::default();
        let mut hud = Hud::new();
        let mut vanilla = DrawList::new();
        hud.draw(&mut vanilla, &mut uniform_font(), None, &global, &player);
        hud.hotbar_resolution = Some(global.resolution);
        let mut same = DrawList::new();
        hud.draw(&mut same, &mut uniform_font(), None, &global, &player);
        assert_eq!(
            same.vertices(),
            vanilla.vertices(),
            "Default preserves every vertex"
        );
        hud.hotbar_resolution = Some(small);
        let mut independent = DrawList::new();
        hud.draw(
            &mut independent,
            &mut uniform_font(),
            None,
            &global,
            &player,
        );
        let crosshair = |d: &DrawList| {
            let first = &d.batches()[0];
            d.vertices()[first.range.clone()].to_vec()
        };
        assert_eq!(crosshair(&vanilla), crosshair(&independent));
        let widget_width = |d: &DrawList| {
            let batch = d
                .batches()
                .iter()
                .find(|b| b.texture == UiTexture::widgets().clone())
                .unwrap();
            let frame = &d.vertices()[batch.range.start..batch.range.start + 6];
            let left = frame.iter().map(|v| v.pos[0]).fold(f32::INFINITY, f32::min);
            let right = frame
                .iter()
                .map(|v| v.pos[0])
                .fold(f32::NEG_INFINITY, f32::max);
            right - left
        };
        assert!((widget_width(&vanilla) - 182.0).abs() < 0.001);
        assert!((widget_width(&independent) * 3.0 - 182.0).abs() < 0.001);
        // Closing/changing the module must not leak a transform into later overlays.
        independent.rect(1, 2, 3, 4, 0xFFFFFFFF);
        assert_eq!(
            independent.vertices()[independent.vertices().len() - 6].pos,
            [1.0, 2.0]
        );
    }

    #[test]
    fn red_score_text_matches_format_for_every_edge_value() {
        let mut buf = [0u8; 16];
        for points in [0, 1, -1, 9, 10, 99, -100, 123_456, i32::MAX, i32::MIN] {
            assert_eq!(red_score_text(&mut buf, points), format!("\u{a7}c{points}"));
        }
    }

    fn ctx() -> ScreenCtx {
        ScreenCtx::new(ScaledResolution::new(854, 480, 1, false), 0.0)
    }

    #[test]
    fn boss_status_is_last_writer_and_visible_for_exactly_one_hundred_rendered_frames() {
        let mut hud = Hud::new();
        hud.set_boss_status(25.0, 100.0, "Dragon A", false);
        hud.set_boss_status(90.0, 100.0, "Wither B", true);
        assert_eq!(hud.boss_name(), Some("Wither B"));
        assert_eq!(hud.boss_health_scale(), 0.9);
        assert!(hud.boss_has_color_modifier());
        assert_eq!(hud.boss_status_bar_time(), 100);

        let c = ctx();
        for frame in 1..=100 {
            let mut draw = DrawList::new();
            hud.draw_boss_health(&mut draw, &mut uniform_font(), &c);
            let icon_vertices: Vec<_> = draw
                .batches()
                .iter()
                .filter(|batch| batch.texture == UiTexture::gui(ICONS_TEXTURE))
                .flat_map(|batch| draw.vertices()[batch.range.clone()].iter())
                .collect();
            assert_eq!(icon_vertices.len(), 18, "boss quads on frame {frame}");
            assert_eq!(hud.boss_status_bar_time(), 100 - frame);

            if frame == 1 {
                assert_eq!(icon_vertices[..6], icon_vertices[6..12]);
                let fill_right = icon_vertices[12..]
                    .iter()
                    .map(|vertex| vertex.pos[0])
                    .fold(f32::NEG_INFINITY, f32::max);
                assert_eq!(fill_right, (c.width / 2 - 91 + 164) as f32);
            }
        }

        let mut expired = DrawList::new();
        hud.draw_boss_health(&mut expired, &mut uniform_font(), &c);
        assert!(
            expired.is_empty(),
            "frame 101 emits no boss geometry or name"
        );
        assert_eq!(hud.boss_status_bar_time(), 0);
    }

    #[test]
    fn boss_bar_is_between_crosshair_and_player_hud_layers() {
        let mut hud = Hud::new();
        hud.set_boss_status(1.0, 2.0, "Boss", false);
        let mut draw = DrawList::new();
        hud.draw(
            &mut draw,
            &mut uniform_font(),
            None,
            &ctx(),
            &PlayerHud::default(),
        );

        let icons = UiTexture::gui(ICONS_TEXTURE);
        let widgets = UiTexture::gui(WIDGETS_TEXTURE);
        let first_ordinary_icons = draw
            .batches()
            .iter()
            .position(|batch| {
                batch.texture == icons && batch.blend == crate::widget::draw::UiBlend::Alpha
            })
            .expect("boss icons after inverse-blended crosshair");
        let hotbar = draw
            .batches()
            .iter()
            .position(|batch| batch.texture == widgets)
            .expect("hotbar after boss bar");
        assert!(first_ordinary_icons < hotbar);
    }

    /// Hand-derived from `GuiIngame.renderPlayerStats`, not from this implementation:
    /// `l1 = ceil((f + f1) / 2 / 10)`, `i2 = max(10 - (l1 - 2), 3)`.
    #[test]
    fn the_heart_row_geometry_matches_the_java_formulas() {
        // 20 max health = 10 hearts = 1 row, pitch 10 - (1 - 2) = 11.
        assert_eq!(heart_rows(20.0, 0.0), 1);
        assert_eq!(heart_row_pitch(1), 11);
        // 40 max health = 2 rows, pitch 10.
        assert_eq!(heart_rows(40.0, 0.0), 2);
        assert_eq!(heart_row_pitch(2), 10);
        // 60 = 3 rows, pitch 9.
        assert_eq!(heart_rows(60.0, 0.0), 3);
        assert_eq!(heart_row_pitch(3), 9);
        // Absorption counts towards the row total.
        assert_eq!(heart_rows(20.0, 20.0), 2);
        // The pitch bottoms out at 3, never lower, however many rows there are.
        assert_eq!(heart_row_pitch(9), 3);
        assert_eq!(heart_row_pitch(40), 3);
        // 21 health is still 2 rows: ceil(21/2/10) = ceil(1.05) = 2.
        assert_eq!(heart_rows(21.0, 0.0), 2);
    }

    #[test]
    fn the_armour_row_is_pushed_up_by_the_whole_heart_stack() {
        // One row: base - 0 * 11 - 10.
        assert_eq!(armor_row_y(100, 1, 11), 90);
        // Three rows at pitch 9: base - 2 * 9 - 10.
        assert_eq!(armor_row_y(100, 3, 9), 72);
    }

    #[test]
    fn the_armour_sprite_picks_full_half_and_empty_at_the_right_points() {
        // 5 armour points: slots 0 and 1 full, slot 2 half, the rest empty.
        assert_eq!(armor_sprite_u(0, 5), 34);
        assert_eq!(armor_sprite_u(1, 5), 34);
        assert_eq!(armor_sprite_u(2, 5), 25);
        assert_eq!(armor_sprite_u(3, 5), 16);
        assert_eq!(armor_sprite_u(0, 0), 16);
        assert_eq!(armor_sprite_u(9, 20), 34);
    }

    /// Oracle values produced by the 1.8.9 Java float expression in
    /// `GuiIngame.renderGameOverlay`, not by this helper. The post-100 assertions cover every
    /// reachable wake frame in addition to pinning both endpoints and Java's float truncation at
    /// 99. The wake values are intentionally not derived from their tempting sleep-ramp mirrors:
    /// Java float rounding makes timer 106 produce alpha 87 while timer 40 produces 88.
    #[test]
    fn sleep_overlay_color_uses_vanillas_hundred_tick_in_and_ten_tick_out_curve() {
        assert_eq!(sleep_overlay_color(0), None);
        assert_eq!(sleep_overlay_color(1), Some(0x0210_1020));
        assert_eq!(sleep_overlay_color(50), Some(0x6E10_1020));
        assert_eq!(sleep_overlay_color(99), Some(0xD910_1020));
        assert_eq!(sleep_overlay_color(100), Some(0xDC10_1020));
        assert_eq!(sleep_overlay_color(101), Some(0xC610_1020));
        assert_eq!(sleep_overlay_color(105), Some(0x6E10_1020));
        assert_eq!(sleep_overlay_color(109), Some(0x1610_1020));
        let wake_frames = [
            0xC610_1020,
            0xB010_1020,
            0x9A10_1020,
            0x8410_1020,
            0x6E10_1020,
            0x5710_1020,
            0x4210_1020,
            0x2B10_1020,
            0x1610_1020,
        ];
        for (timer, expected) in (101..=109).zip(wake_frames) {
            assert_eq!(sleep_overlay_color(timer), Some(expected), "timer {timer}");
        }
    }

    /// `GuiIngame.renderGameOverlay` suppresses the portal pass for any present nausea effect;
    /// `isPotionActive` does not inspect the effect's amplifier or duration.
    #[test]
    fn portal_overlay_is_suppressed_by_nausea_presence_not_strength() {
        assert_eq!(portal_overlay_time(0.75, None), Some(0.75));
        assert_eq!(portal_overlay_time(0.0, None), None);
        assert_eq!(portal_overlay_time(-0.25, None), None);
        assert_eq!(portal_overlay_time(0.75, Some(-1)), None);
        assert_eq!(portal_overlay_time(0.75, Some(0)), None);
        assert_eq!(portal_overlay_time(0.75, Some(3)), None);
    }

    /// `GuiIngame` draws player stats, then the sleep rectangle, then the XP bar. Inspecting the
    /// actual batch order and full-screen solid vertices prevents a colour-only helper from
    /// passing while the live HUD path is disconnected or layered incorrectly.
    #[test]
    fn sleep_overlay_covers_the_screen_after_stats_but_before_experience() {
        let hud = Hud::new();
        let c = ctx();
        let state = PlayerHud {
            sleep_timer: 100,
            experience: 0.5,
            ..PlayerHud::default()
        };
        let mut draw = DrawList::new();
        hud.draw(&mut draw, &mut uniform_font(), None, &c, &state);

        let sleep_batch = draw
            .batches()
            .iter()
            .position(|batch| {
                batch.texture == UiTexture::Solid
                    && draw.vertices()[batch.range.clone()]
                        .iter()
                        .any(|vertex| vertex.pos == [c.width as f32, c.height as f32])
            })
            .expect("full-screen sleep rectangle");
        let sleep_vertices = &draw.vertices()[draw.batches()[sleep_batch].range.clone()];
        assert_eq!(sleep_vertices[0].pos, [0.0, 0.0]);
        assert_eq!(
            sleep_vertices[0].color,
            [16.0 / 255.0, 16.0 / 255.0, 32.0 / 255.0, 220.0 / 255.0]
        );

        let icons = UiTexture::gui(ICONS_TEXTURE);
        assert!(
            draw.batches()[..sleep_batch]
                .iter()
                .any(|batch| batch.texture == icons),
            "crosshair/player stats precede sleep"
        );
        assert!(
            draw.batches()[sleep_batch + 1..]
                .iter()
                .any(|batch| batch.texture == icons),
            "XP bar follows sleep and remains visible above it"
        );

        let mut awake = DrawList::new();
        hud.draw(
            &mut awake,
            &mut uniform_font(),
            None,
            &c,
            &PlayerHud::default(),
        );
        assert!(
            awake
                .batches()
                .iter()
                .all(|batch| batch.texture != UiTexture::Solid),
            "timer zero emits no transparent placeholder rectangle"
        );
    }

    #[test]
    fn the_heart_family_and_row_follow_the_status_effects() {
        assert_eq!(heart_family_u(false, false), 16);
        assert_eq!(heart_family_u(true, false), 52);
        assert_eq!(heart_family_u(false, true), 88);
        // Poison wins over wither, exactly as vanilla's if/else-if does.
        assert_eq!(heart_family_u(true, true), 52);
        assert_eq!(heart_row_v(false), 0);
        assert_eq!(heart_row_v(true), 45);
    }

    /// Hand-derived from `ceiling_double_int((air - 2) * 10 / 300)` and the difference form.
    #[test]
    fn air_bubbles_drain_from_ten_to_zero() {
        assert_eq!(air_bubbles(300), (10, 0));
        assert_eq!(air_bubbles(0), (0, 0));
        // 150 air: ceil(148 * 10 / 300) = ceil(4.933) = 5 full, ceil(5) - 5 = 0 popping.
        assert_eq!(air_bubbles(150), (5, 0));
        // 152: ceil(5.0) = 5 full, ceil(5.066) = 6, so one is popping.
        assert_eq!(air_bubbles(152), (5, 1));
        assert_eq!(air_bubbles(2), (0, 1));
    }

    #[test]
    fn the_hotbar_and_crosshair_land_on_vanillas_pixels() {
        let hud = Hud::new();
        let c = ctx();
        let mut d = DrawList::new();
        let state = PlayerHud {
            selected_slot: 3,
            ..PlayerHud::default()
        };
        hud.draw_crosshair(&mut d, &c);
        // 16×16 at (w/2 - 7, h/2 - 7).
        assert_eq!(
            d.vertices()[0].pos,
            [(854 / 2 - 7) as f32, (480 / 2 - 7) as f32]
        );
        assert_eq!(d.vertices()[0].uv, [0.0, 0.0]);
        assert_eq!(
            d.batches()[0].blend,
            crate::widget::draw::UiBlend::Invert,
            "GuiIngame changes blend state only for the crosshair"
        );
        assert!(
            d.batches()[0].alpha_test,
            "GuiIngame enables alphaFunc(GREATER, 0.1) around the crosshair"
        );

        let mut d = DrawList::new();
        hud.draw_hotbar(&mut d, &c, &state);
        let v = d.vertices();
        assert_eq!(v[0].pos, [(854 / 2 - 91) as f32, (480 - 22) as f32]);
        // The selection cursor is 24 wide, one pixel up and left of slot 3's start.
        assert_eq!(v[6].pos, [(854 / 2 - 92 + 60) as f32, (480 - 23) as f32]);
        assert_eq!(v[6].uv, [0.0, 22.0 / 256.0]);

        assert_eq!(Hud::hotbar_item_pos(&c, 0), (854 / 2 - 88, 480 - 19));
        assert_eq!(Hud::hotbar_item_pos(&c, 8), (854 / 2 - 88 + 160, 480 - 19));
    }

    #[test]
    fn crosshair_visibility_is_the_exact_spectator_target_predicate() {
        let ordinary = CrosshairState::default();
        assert!(ordinary.shows_crosshair());
        // Java's showCrosshair() never consults thirdPersonView; keeping this true is a
        // deliberate negative control against importing a later-version camera rule.
        assert!(CrosshairState {
            third_person: true,
            ..ordinary
        }
        .shows_crosshair());

        // Spectators see the reticle only over living entities, ItemFrames, or an actual
        // IInventory tile; a sign/non-living entity is intentionally the same `None` negative
        // control.
        for target in [
            CrosshairTarget::None,
            CrosshairTarget::LivingEntity,
            CrosshairTarget::ItemFrame,
            CrosshairTarget::InventoryBlock,
        ] {
            assert_eq!(
                CrosshairState {
                    spectator: true,
                    target,
                    ..CrosshairState::default()
                }
                .shows_crosshair(),
                matches!(
                    target,
                    CrosshairTarget::LivingEntity
                        | CrosshairTarget::ItemFrame
                        | CrosshairTarget::InventoryBlock
                ),
                "unexpected spectator crosshair result for {target:?}"
            );
        }
    }

    #[test]
    fn crosshair_debug_gate_matches_vanillas_reduced_debug_exceptions() {
        let debug = CrosshairState {
            show_debug_info: true,
            ..CrosshairState::default()
        };
        assert!(!debug.shows_crosshair(), "F3 hides the ordinary crosshair");

        assert!(CrosshairState {
            has_reduced_debug: true,
            ..debug
        }
        .shows_crosshair());
        assert!(CrosshairState {
            reduced_debug_info: true,
            ..debug
        }
        .shows_crosshair());

        // The debug check runs before spectator targeting: an unexceptioned F3 must still hide
        // the reticle even when a living entity is under the cursor.
        assert!(!CrosshairState {
            spectator: true,
            target: CrosshairTarget::LivingEntity,
            ..debug
        }
        .shows_crosshair());
        assert!(CrosshairState {
            spectator: true,
            target: CrosshairTarget::LivingEntity,
            has_reduced_debug: true,
            ..debug
        }
        .shows_crosshair());
    }

    #[test]
    fn hidden_crosshair_opens_no_invert_batch() {
        let hud = Hud::new();
        let mut draw = DrawList::new();
        hud.draw_crosshair_for(
            &mut draw,
            &ctx(),
            CrosshairState {
                spectator: true,
                target: CrosshairTarget::None,
                ..CrosshairState::default()
            },
        );
        assert!(draw.vertices().is_empty());
        assert!(draw.batches().is_empty());

        let mut draw = DrawList::new();
        hud.draw_crosshair_for(
            &mut draw,
            &ctx(),
            CrosshairState {
                show_debug_info: true,
                ..CrosshairState::default()
            },
        );
        assert!(
            draw.vertices().is_empty(),
            "F3 must not open an invert batch"
        );
        assert!(draw.batches().is_empty());
    }

    #[test]
    fn a_full_health_bar_is_ten_containers_and_ten_filled_hearts() {
        let hud = Hud::new();
        let c = ctx();
        let mut d = DrawList::new();
        hud.draw_player_stats(&mut d, &c, &PlayerHud::default());
        // 10 containers + 10 filled hearts + 10 food containers + 10 filled food = 40 quads.
        assert_eq!(d.vertices().len() / 6, 40);
        // The leftmost heart sits at (w/2 - 91, h - 39).
        assert!(d
            .vertices()
            .iter()
            .any(|v| v.pos == [(854 / 2 - 91) as f32, (480 - 39) as f32]));
    }

    #[test]
    fn half_a_heart_uses_the_half_sprite_and_missing_hearts_use_none() {
        let hud = Hud::new();
        let c = ctx();
        let mut d = DrawList::new();
        // 9 health = 4 full hearts and one half.
        hud.draw_player_stats(
            &mut d,
            &c,
            &PlayerHud {
                health: 9.0,
                food: 20,
                ..PlayerHud::default()
            },
        );
        // Match on each quad's *first* vertex: the right edge of the u = 52 sprite is also
        // 61/256, so a naive filter over every vertex counts the full hearts twice over.
        let count = |u: i32, v: i32| {
            d.vertices()
                .chunks(6)
                .filter(|q| {
                    (q[0].uv[0] - u as f32 / 256.0).abs() < 1e-6
                        && (q[0].uv[1] - v as f32 / 256.0).abs() < 1e-6
                })
                .count()
        };
        assert_eq!(count(16 + 45, 0), 1, "exactly one half heart");
        assert_eq!(count(16 + 36, 0), 4, "and four full ones");
        assert_eq!(
            count(16 + 36, 27),
            10,
            "and ten full drumsticks, on their own row"
        );
    }

    /// Two server values made a HUD divisor zero, which panicked (vanilla
    /// throws `ArithmeticException` there): a NaN max-health attribute while Regeneration is
    /// active, and the S06 food level 1431655765, whose `food * 3 + 1` wraps to 0.
    #[test]
    fn hostile_health_and_food_values_cannot_divide_by_zero() {
        let mut hud = Hud::new();
        let c = ctx();
        for counter in 0..4 {
            hud.update_counter = counter;
            let mut d = DrawList::new();
            hud.draw_player_stats(
                &mut d,
                &c,
                &PlayerHud {
                    max_health: f32::NAN,
                    regenerating: true,
                    ..PlayerHud::default()
                },
            );
            let mut d = DrawList::new();
            hud.draw_player_stats(
                &mut d,
                &c,
                &PlayerHud {
                    food: 1_431_655_765,
                    saturation: 0.0,
                    ..PlayerHud::default()
                },
            );
        }
        // Real values keep Java's period: food 20 shakes every 61 counter steps.
        assert_eq!(20i32.wrapping_mul(3).wrapping_add(1), 61);
    }

    /// A raw absorption of 2e9 asked for a billion hearts per frame; the
    /// draw list now stops at `MAX_DRAWN_HEARTS`, and the largest legitimate state is untouched.
    #[test]
    fn a_hostile_absorption_value_draws_a_bounded_number_of_hearts() {
        let hud = Hud::new();
        let c = ctx();
        let mut d = DrawList::new();
        let started = std::time::Instant::now();
        hud.draw_player_stats(
            &mut d,
            &c,
            &PlayerHud {
                absorption: 2.0e9,
                ..PlayerHud::default()
            },
        );
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
        // Each heart is a container plus a golden heart; ten food slots add twenty quads.
        assert_eq!(d.vertices().len() / 6, 2 * MAX_DRAWN_HEARTS as usize + 20);
        const { assert!((1024 + 1024) / 2 < MAX_DRAWN_HEARTS) };
        // Past `i32` the heart count wraps negative exactly as Java's does, and nothing is drawn
        // for it, with no overflow panic in a build that checks.
        for absorption in [1.0e30f32, -1.0e30, f32::INFINITY, f32::NAN] {
            let mut d = DrawList::new();
            hud.draw_player_stats(
                &mut d,
                &c,
                &PlayerHud {
                    absorption,
                    ..PlayerHud::default()
                },
            );
            assert!(d.vertices().len() / 6 <= 2 * MAX_DRAWN_HEARTS as usize + 20);
        }
    }

    #[test]
    fn absorption_hearts_are_drawn_from_the_golden_family() {
        let hud = Hud::new();
        let c = ctx();
        let mut d = DrawList::new();
        hud.draw_player_stats(
            &mut d,
            &c,
            &PlayerHud {
                absorption: 4.0,
                ..PlayerHud::default()
            },
        );
        // 12 hearts total now (ceil((20+4)/2) = 12), so two rows.
        let gold_u = (16 + 144) as f32 / 256.0;
        let gold = d
            .vertices()
            .iter()
            .filter(|v| (v.uv[0] - gold_u).abs() < 1e-6)
            .count()
            / 3;
        assert_eq!(gold, 2, "two golden hearts for four absorption points");
    }

    #[test]
    fn damage_flash_uses_blank_containers_and_previous_health_ghosts() {
        let mut hud = Hud::new();
        // The first stable frame initialises GuiIngame's `playerHealth` / `lastPlayerHealth`.
        hud.observe_player_health(20.0, 0, 1_001);
        // The hit frame arms `healthUpdateCounter`, but Java sampled `flag` before doing so.
        hud.observe_player_health(16.0, 20, 1_002);
        assert!(
            !hud.health_flash_active,
            "a hit never flashes on its arming render"
        );
        for _ in 0..3 {
            hud.tick();
        }
        hud.observe_player_health(16.0, 17, 1_152);
        assert!(
            hud.health_flash_active,
            "the 17-tick phase is the first visible flash"
        );

        let mut d = DrawList::new();
        hud.draw_player_stats(
            &mut d,
            &ctx(),
            &PlayerHud {
                health: 16.0,
                ..PlayerHud::default()
            },
        );
        let quads_at = |u: i32, v: i32| {
            d.vertices()
                .chunks(6)
                .filter(|quad| {
                    (quad[0].uv[0] - u as f32 / 256.0).abs() < 1e-6
                        && (quad[0].uv[1] - v as f32 / 256.0).abs() < 1e-6
                })
                .count()
        };
        assert_eq!(
            quads_at(25, 0),
            10,
            "flash swaps all ten heart containers to u = 25"
        );
        assert_eq!(
            quads_at(16 + 54, 0),
            10,
            "lastPlayerHealth's ten full hearts remain as the white ghost pass"
        );
    }

    /// 1.7 Animations' `old_health`: the containers still flash, but the
    /// white ghost hearts (`u = family + 54 / 63`) are not drawn.
    #[test]
    fn hidden_lost_health_hearts_keep_the_container_flash_only() {
        let mut hud = Hud::new();
        hud.observe_player_health(20.0, 0, 1_001);
        hud.observe_player_health(15.0, 20, 1_002);
        for _ in 0..3 {
            hud.tick();
        }
        hud.observe_player_health(15.0, 17, 1_152);
        assert!(hud.health_flash_active);
        hud.hide_lost_health_hearts = true;

        let mut d = DrawList::new();
        hud.draw_player_stats(
            &mut d,
            &ctx(),
            &PlayerHud {
                health: 15.0,
                ..PlayerHud::default()
            },
        );
        let quads_at = |u: i32| {
            d.vertices()
                .chunks(6)
                .filter(|quad| {
                    (quad[0].uv[0] - u as f32 / 256.0).abs() < 1e-6 && quad[0].uv[1].abs() < 1e-6
                })
                .count()
        };
        assert_eq!(quads_at(25), 10, "the containers still flash");
        assert_eq!(quads_at(16 + 54) + quads_at(16 + 63), 0, "no ghost hearts");
        assert_eq!(quads_at(16 + 36), 7, "the seven full hearts left");
        assert_eq!(quads_at(16 + 45), 1, "and the half heart");
    }

    #[test]
    fn health_change_without_hurt_resistance_does_not_arm_the_flash() {
        let mut hud = Hud::new();
        hud.observe_player_health(20.0, 0, 1_001);
        hud.observe_player_health(8.0, 0, 1_002);
        for _ in 0..9 {
            hud.tick();
            hud.observe_player_health(8.0, 0, 1_002);
            assert!(
                !hud.health_flash_active,
                "the hurt-resistance gate is what distinguishes a visual damage event"
            );
        }
    }

    #[test]
    fn creative_mode_draws_no_stat_bars_and_no_xp() {
        let hud = Hud::new();
        let c = ctx();
        let mut font = uniform_font();
        let mut d = DrawList::new();
        let state = PlayerHud {
            survival: false,
            experience: 0.5,
            experience_level: 30,
            ..PlayerHud::default()
        };
        hud.draw_player_stats(&mut d, &c, &state);
        hud.draw_experience(&mut d, &mut font, &c, &state);
        assert!(d.is_empty());
    }

    #[test]
    fn the_xp_bar_fill_is_one_hundred_and_eighty_three_wide_at_full() {
        let hud = Hud::new();
        let c = ctx();
        let mut font = uniform_font();
        let mut d = DrawList::new();
        hud.draw_experience(
            &mut d,
            &mut font,
            &c,
            &PlayerHud {
                experience: 1.0,
                ..PlayerHud::default()
            },
        );
        let x = 854 / 2 - 91;
        let y = 480 - 32 + 3;
        // Track then fill.
        assert_eq!(d.vertices()[0].pos, [x as f32, y as f32]);
        let fill_right = d.vertices()[6..12]
            .iter()
            .map(|v| v.pos[0])
            .fold(f32::MIN, f32::max);
        assert_eq!(fill_right, (x + 183) as f32);

        // Empty bar: the track only.
        let mut d = DrawList::new();
        hud.draw_experience(&mut d, &mut font, &c, &PlayerHud::default());
        assert_eq!(d.vertices().len(), 6);
    }

    #[test]
    fn the_xp_bar_preserves_out_of_range_server_fractions() {
        let hud = Hud::new();
        let c = ctx();
        let mut font = uniform_font();
        let x = c.width / 2 - 91;

        let mut over = DrawList::new();
        hud.draw_experience(
            &mut over,
            &mut font,
            &c,
            &PlayerHud {
                experience: 2.0,
                ..PlayerHud::default()
            },
        );
        let fill_right = over.vertices()[6..12]
            .iter()
            .map(|v| v.pos[0])
            .fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(fill_right, (x + 366) as f32);

        let mut negative = DrawList::new();
        hud.draw_experience(
            &mut negative,
            &mut font,
            &c,
            &PlayerHud {
                experience: -1.0,
                ..PlayerHud::default()
            },
        );
        assert_eq!(
            negative.vertices().len(),
            6,
            "negative fill draws only the track"
        );
    }

    #[test]
    fn the_mounted_horse_bar_replaces_xp_and_uses_the_charge_texels() {
        let hud = Hud::new();
        let c = ctx();
        let mut font = uniform_font();
        let mut d = DrawList::new();
        hud.draw_experience_or_horse(
            &mut d,
            &mut font,
            &c,
            &PlayerHud {
                survival: false,
                experience: 1.0,
                experience_level: 30,
                horse_jump_power: Some(0.5),
                ..PlayerHud::default()
            },
        );

        assert_eq!(d.vertices().len(), 12, "track plus one charge fill");
        let x = c.width / 2 - 91;
        let fill_right = d.vertices()[6..12]
            .iter()
            .map(|vertex| vertex.pos[0])
            .fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(fill_right, (x + 91) as f32);
        assert!((d.vertices()[0].uv[1] - 84.0 / 256.0).abs() < f32::EPSILON);
        assert!((d.vertices()[6].uv[1] - 89.0 / 256.0).abs() < f32::EPSILON);
        assert!(
            d.batches()
                .iter()
                .all(|batch| !matches!(batch.texture, UiTexture::Font(_))),
            "the XP level is suppressed while mounted",
        );
    }

    #[test]
    fn the_hud_rng_seed_wraps_at_the_same_frame_as_java_int_multiplication() {
        assert_eq!(hud_random_seed(6_863), 2_147_233_673);
        assert_eq!(hud_random_seed(6_864), -2_147_420_752);
        assert_ne!(hud_random_seed(6_864), i64::from(6_864) * 312_871);
    }

    #[test]
    fn the_xp_level_is_drawn_five_times_for_its_outline() {
        let hud = Hud::new();
        let c = ctx();
        let mut font = uniform_font();
        let mut d = DrawList::new();
        hud.draw_experience(
            &mut d,
            &mut font,
            &c,
            &PlayerHud {
                experience_level: 7,
                ..PlayerHud::default()
            },
        );
        let glyphs = d
            .batches()
            .iter()
            .filter(|b| matches!(b.texture, UiTexture::Font(_)))
            .map(|b| b.range.len() / 6)
            .sum::<usize>();
        assert_eq!(glyphs, 5, "four black offsets plus the green centre");
    }

    #[test]
    fn the_title_fade_curve_matches_vanillas_three_phases() {
        let mut hud = Hud::new();
        hud.display_title("Hello", "there");
        assert_eq!(hud.title_timer, 100);
        // Just displayed: elapsed = 0, so alpha = 0 — below the >8 gate.
        assert_eq!(hud.title_alpha(0.0), None);
        // Five ticks in: elapsed = 5, alpha = 5 * 255 / 10 = 127.
        hud.title_timer = 95;
        assert_eq!(hud.title_alpha(0.0), Some(127));
        // Mid-hold: neither branch fires, so full opacity.
        hud.title_timer = 50;
        assert_eq!(hud.title_alpha(0.0), Some(255));
        // Fading out: remaining = 10, alpha = 10 * 255 / 20 = 127.
        hud.title_timer = 10;
        assert_eq!(hud.title_alpha(0.0), Some(127));
        hud.title_timer = 0;
        assert_eq!(hud.title_alpha(0.0), None);
    }

    #[test]
    fn default_hud_has_the_same_title_timings_as_the_java_constructor() {
        let mut hud = Hud::default();
        assert_eq!(
            (hud.title_fade_in, hud.title_display, hud.title_fade_out),
            (10, 70, 20)
        );
        hud.set_title("default-constructed");
        assert_eq!(hud.title_timer, 100);
        assert_eq!(hud.title_alpha(5.0), Some(127));
    }

    #[test]
    fn a_title_clears_itself_when_its_timer_runs_out() {
        let mut hud = Hud::new();
        hud.display_title("A", "B");
        for _ in 0..100 {
            hud.tick();
        }
        assert!(hud.title.is_empty() && hud.subtitle.is_empty());
        assert_eq!(hud.title_timer, 0);
    }

    #[test]
    fn the_five_title_actions_match_gui_ingames_branch_order() {
        let mut hud = Hud::new();
        hud.set_subtitle("old subtitle");
        assert_eq!(
            hud.title_timer, 0,
            "a subtitle alone does not start a title"
        );

        hud.set_title_times(4, 30, 6);
        assert_eq!(hud.title_timer, 0, "TIMES does not start an idle title");
        hud.set_title("main");
        assert_eq!(hud.title_timer, 40);
        assert_eq!(hud.subtitle, "old subtitle");

        hud.set_title_times(-1, 50, -1);
        assert_eq!(
            (hud.title_fade_in, hud.title_display, hud.title_fade_out),
            (4, 50, 6)
        );
        assert_eq!(hud.title_timer, 60, "TIMES restarts an active title");

        hud.clear_title();
        assert_eq!(hud.title_timer, 0);
        assert_eq!(
            (hud.title_fade_in, hud.title_display, hud.title_fade_out),
            (0, 0, 0)
        );
        assert_eq!((&*hud.title, &*hud.subtitle), ("main", "old subtitle"));
        hud.set_title("invisible after clear");
        assert_eq!(hud.title_timer, 0, "CLEAR leaves zero timings behind");

        hud.set_title_times(3, 20, 7);
        hud.set_subtitle("retained by reset");
        hud.reset_title();
        assert!(hud.title.is_empty());
        assert_eq!(hud.subtitle, "retained by reset");
        assert_eq!(hud.title_timer, 30, "RESET first starts the old timing sum");
        assert_eq!(
            (hud.title_fade_in, hud.title_display, hud.title_fade_out),
            (10, 70, 20)
        );
    }

    #[test]
    fn a_times_packet_with_three_negative_values_clears_like_display_titles_first_branch() {
        // java: handleTitle(TIMES) -> displayTitle(null, null, -1, -1, -1), whose first branch
        // empties both strings and zeroes the timer; the timings themselves are untouched.
        let mut hud = Hud::new();
        hud.set_title_times(4, 30, 6);
        hud.set_subtitle("sub");
        hud.set_title("main");
        assert_eq!(hud.title_timer, 40);
        hud.set_title_times(-1, -1, -1);
        assert_eq!(hud.title_timer, 0);
        assert!(hud.title.is_empty() && hud.subtitle.is_empty());
        assert_eq!(hud.title_alpha(0.0), None);
        assert_eq!(
            (hud.title_fade_in, hud.title_display, hud.title_fade_out),
            (4, 30, 6),
            "the clearing branch never reaches the setters"
        );
        // Any non-negative field takes the setter branch instead, as before.
        hud.set_title("again");
        hud.set_title_times(-1, -1, 0);
        assert_eq!(hud.title, "again");
        assert_eq!(hud.title_timer, 34);
    }

    #[test]
    fn hostile_title_times_wrap_like_java_int_addition() {
        let mut hud = Hud::new();
        hud.set_title_times(i32::MAX, i32::MAX, i32::MAX);
        hud.set_title("overflow");
        assert_eq!(hud.title_timer, i32::MAX.wrapping_mul(3));
    }

    #[test]
    fn effect_rows_start_at_gui_top_and_compress_only_after_five() {
        assert_eq!(Hud::effect_row_y(10, 0, 5), 10);
        assert_eq!(Hud::effect_row_y(10, 1, 5), 43);
        assert_eq!(Hud::effect_row_y(10, 4, 5), 142);

        // Six effects use `132 / (6 - 1) = 26`, keeping the final row within 132 pixels.
        assert_eq!(Hud::effect_row_y(10, 0, 6), 10);
        assert_eq!(Hud::effect_row_y(10, 1, 6), 36);
        assert_eq!(Hud::effect_row_y(10, 5, 6), 140);
    }

    #[test]
    fn title_and_subtitle_use_vanillas_independent_four_and_two_times_transforms() {
        let mut hud = Hud::new();
        hud.display_title("A", "B");
        hud.title_timer = 50;
        let c = ctx();
        let mut font = uniform_font();
        let mut d = DrawList::new();
        hud.draw(&mut d, &mut font, None, &c, &PlayerHud::default());

        let glyphs: Vec<&[crate::widget::UiVertex]> = d
            .batches()
            .iter()
            .filter(|batch| matches!(batch.texture, UiTexture::Font(_)))
            .flat_map(|batch| d.vertices()[batch.range.clone()].chunks_exact(6))
            .collect();
        assert_eq!(
            glyphs.len(),
            4,
            "one shadow and one foreground glyph per line"
        );

        let bounds = |quad: &[crate::widget::UiVertex]| {
            let min_x = quad.iter().map(|v| v.pos[0]).fold(f32::INFINITY, f32::min);
            let max_x = quad
                .iter()
                .map(|v| v.pos[0])
                .fold(f32::NEG_INFINITY, f32::max);
            let min_y = quad.iter().map(|v| v.pos[1]).fold(f32::INFINITY, f32::min);
            (min_x, max_x, min_y)
        };
        let title = bounds(glyphs[1]);
        let subtitle = bounds(glyphs[3]);
        assert_eq!(title.2, (c.height / 2 - 40) as f32);
        assert_eq!(subtitle.2, (c.height / 2 + 10) as f32);
        assert!(
            ((title.1 - title.0) - 2.0 * (subtitle.1 - subtitle.0)).abs() < 0.001,
            "the title must be exactly twice the subtitle's screen-space width"
        );
        assert_eq!(d.transform_depth(), 0, "the title transforms must not leak");
    }

    #[test]
    fn the_effect_overlay_is_off_by_default_because_one_point_eight_has_none() {
        let hud = Hud::new();
        let c = ctx();
        let mut font = uniform_font();
        let state = PlayerHud {
            effects: vec![ActiveEffect {
                icon_index: 10,
                name: "Speed II".into(),
                duration: "1:30".into(),
                beneficial: true,
            }],
            ..PlayerHud::default()
        };
        let mut d = DrawList::new();
        hud.draw(&mut d, &mut font, None, &c, &state);
        let has_inventory_tex = d
            .batches()
            .iter()
            .any(|b| matches!(&b.texture, UiTexture::Resource(r) if r.path == INVENTORY_TEXTURE));
        assert!(
            !has_inventory_tex,
            "vanilla 1.8.9 draws no HUD effect icons"
        );

        let mut d = DrawList::new();
        hud.draw(
            &mut d,
            &mut font,
            None,
            &c,
            &PlayerHud {
                show_effect_overlay: true,
                ..state
            },
        );
        assert!(d.batches().iter().any(|b| {
            matches!(&b.texture, UiTexture::Resource(r) if r.path == INVENTORY_TEXTURE)
        }));
    }

    /// A damaged diamond sword. No atlas is installed in a unit test, so `item_model_for`
    /// returns the empty model and the *icon* draws nothing — but `renderItemOverlays`'
    /// durability bar is pure geometry, which makes it the position probe here. The GPU test
    /// `tests/hud_items.rs` is what proves the icon's own texels.
    fn damaged_sword() -> ItemStack {
        ItemStack {
            item_id: 276, // minecraft:diamond_sword, max_damage 1561
            count: 1,
            damage: 780,
            nbt: None,
        }
    }

    #[test]
    fn an_empty_hotbar_draws_no_items_at_all() {
        let hud = Hud::new();
        let c = ctx();
        let mut font = uniform_font();
        let mut d = DrawList::new();
        hud.draw_hotbar_items(
            &mut d,
            &mut font,
            &UiTexture::gui("mc-ui/test-atlas"),
            &c,
            &PlayerHud::default(),
        );
        assert!(
            d.is_empty(),
            "a null slot draws nothing, as vanilla's guard does"
        );
    }

    #[test]
    fn every_occupied_hotbar_slot_draws_at_its_vanilla_position() {
        let hud = Hud::new();
        let c = ctx();
        let mut font = uniform_font();
        let mut state = PlayerHud::default();
        state.hotbar[0] = Some(damaged_sword());
        state.hotbar[8] = Some(damaged_sword());

        let mut d = DrawList::new();
        hud.draw_hotbar_items(
            &mut d,
            &mut font,
            &UiTexture::gui("mc-ui/test-atlas"),
            &c,
            &state,
        );
        // Two slots × (durability backing + coloured foreground) = 4 rects.
        assert_eq!(d.vertices().len() / 6, 6);
        let bar_left = |slot: i32| (Hud::hotbar_item_pos(&c, slot).0 + 2) as f32;
        let bar_top = |slot: i32| (Hud::hotbar_item_pos(&c, slot).1 + 13) as f32;
        assert_eq!(d.vertices()[0].pos, [bar_left(0), bar_top(0)]);
        assert_eq!(d.vertices()[18].pos, [bar_left(8), bar_top(8)]);
        // 20 pixels of pitch between slot 0 and slot 8, eight slots apart.
        assert_eq!(bar_left(8) - bar_left(0), 160.0);
    }

    /// The Java source computes the countdown remainder before the scale and only enters the
    /// matrix branch for a strictly positive remainder. The NaN case is included because Java's
    /// `f > 0.0F` comparison is false for it, unlike a loose `<= 0` translation.
    #[test]
    fn hotbar_pickup_pop_scale_matches_the_java_float_branch() {
        fn java(animations_to_go: i32, partial_ticks: f32) -> Option<(f32, f32)> {
            let f = animations_to_go as f32 - partial_ticks;
            if !matches!(f.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
                return None;
            }
            let f1 = 1.0 + f / 5.0;
            Some((1.0 / f1, (f1 + 1.0) / 2.0))
        }

        for (ticks, partial) in [
            (5, 0.0),
            (5, 0.25),
            (2, 0.5),
            (1, 1.0),
            (0, 0.0),
            (-1, 0.0),
            (5, 5.0),
            (5, f32::NAN),
        ] {
            assert_eq!(
                hotbar_item_pop_scale(ticks, partial),
                java(ticks, partial),
                "animationsToGo={ticks}, partialTicks={partial:?}"
            );
        }
    }

    /// The GL sequence is a scale about `(x + 8, y + 12)`, not about the slot's top-left. This
    /// CPU oracle bites both the old static path and a tempting origin-scaled transcription.
    #[test]
    fn hotbar_pickup_pop_is_pivoted_at_the_java_slot_centre() {
        let (x, y) = (117, 283);
        let (scale_x, scale_y) = hotbar_item_pop_scale(5, 0.0).unwrap();
        let transform = hotbar_item_pop_transform(x, y, scale_x, scale_y);
        let anchor = [(x + 8) as f32, (y + 12) as f32];
        assert_eq!(transform.apply(anchor), anchor);

        let corner = [x as f32, y as f32];
        let expected = [
            anchor[0] + (corner[0] - anchor[0]) * scale_x,
            anchor[1] + (corner[1] - anchor[1]) * scale_y,
        ];
        let actual = transform.apply(corner);
        assert!(
            (actual[0] - expected[0]).abs() < 1e-6,
            "{actual:?} vs {expected:?}"
        );
        assert!(
            (actual[1] - expected[1]).abs() < 1e-6,
            "{actual:?} vs {expected:?}"
        );
    }

    /// Java pops the icon matrix before drawing count/durability overlays. A counted, damaged
    /// stack is a CPU-only probe for that ordering even when no atlas has been installed.
    #[test]
    fn hotbar_pickup_pop_leaves_overlays_pixel_aligned() {
        let hud = Hud::new();
        let c = ctx();
        let mut font = uniform_font();
        let mut state = PlayerHud::default();
        state.animations_to_go[0] = 5;
        let mut stack = damaged_sword();
        stack.count = 2;
        state.hotbar[0] = Some(stack);

        let mut d = DrawList::new();
        hud.draw_hotbar_items(
            &mut d,
            &mut font,
            &UiTexture::gui("mc-ui/test-atlas"),
            &c,
            &state,
        );
        let (x, y) = Hud::hotbar_item_pos(&c, 0);
        assert_eq!(
            d.transform_depth(),
            0,
            "the temporary GL matrix must be popped"
        );
        assert_eq!(d.vertices()[0].pos, [(x + 2) as f32, (y + 13) as f32]);
    }

    #[test]
    fn the_full_hud_draws_hotbar_items_only_when_it_is_given_an_atlas() {
        let hud = Hud::new();
        let c = ctx();
        let mut font = uniform_font();
        let mut state = PlayerHud::default();
        state.hotbar[4] = Some(damaged_sword());

        let mut without = DrawList::new();
        hud.draw(&mut without, &mut font, None, &c, &state);
        let mut with = DrawList::new();
        hud.draw(
            &mut with,
            &mut font,
            Some(&UiTexture::gui("mc-ui/test-atlas")),
            &c,
            &state,
        );
        assert_eq!(
            with.vertices().len() - without.vertices().len(),
            18,
            "the one occupied slot adds exactly its three durability rects"
        );
    }

    #[test]
    fn spectator_draw_suppresses_only_the_ordinary_player_layers() {
        let c = ctx();
        let atlas = UiTexture::gui("mc-ui/test-atlas");

        let mut populated_hud = Hud::new();
        populated_hud.held_item_name = "Diamond Sword".into();
        populated_hud.highlight_ticks = 40;
        populated_hud.set_record_playing("Spectating", false);
        let mut populated = PlayerHud {
            health: 3.0,
            armor: 20,
            food: 2,
            experience: 0.75,
            experience_level: 30,
            selected_slot: 8,
            ..PlayerHud::default()
        };
        populated.hotbar[8] = Some(damaged_sword());

        let mut baseline_hud = populated_hud.clone();
        baseline_hud.held_item_name.clear();
        baseline_hud.highlight_ticks = 0;
        let baseline = PlayerHud {
            survival: false,
            selected_slot: 0,
            ..PlayerHud::default()
        };

        let mut populated_font = uniform_font();
        let mut populated_draw = DrawList::new();
        let spectator_crosshair = CrosshairState {
            spectator: true,
            target: CrosshairTarget::LivingEntity,
            ..CrosshairState::default()
        };
        populated_hud.draw_with_spectator_and_crosshair(
            &mut populated_draw,
            &mut populated_font,
            Some(&atlas),
            &c,
            &populated,
            true,
            spectator_crosshair,
        );
        let mut baseline_font = uniform_font();
        let mut baseline_draw = DrawList::new();
        baseline_hud.draw_with_spectator_and_crosshair(
            &mut baseline_draw,
            &mut baseline_font,
            None,
            &c,
            &baseline,
            true,
            spectator_crosshair,
        );

        assert_eq!(
            populated_draw.vertices(),
            baseline_draw.vertices(),
            "hotbar contents/selection, survival stats, XP, and held caption must be inert"
        );
        let resource_vertices = populated_draw
            .batches()
            .iter()
            .filter(|batch| matches!(batch.texture, UiTexture::Resource(_)))
            .map(|batch| batch.range.len())
            .sum::<usize>();
        assert_eq!(
            resource_vertices, 6,
            "only the crosshair resource quad remains"
        );
        assert!(
            populated_draw
                .batches()
                .iter()
                .any(|batch| matches!(batch.texture, UiTexture::Font(_))),
            "the action-bar overlay is still rendered"
        );
    }

    /// Vanilla's `updateTick` decrements only on the unchanged-stack branch and *assigns* on the
    /// other two, so moving the decrement into [`Hud::tick`] cannot change the outcome. Driven
    /// against a literal transcription of the Java block rather than against this port.
    #[test]
    fn the_split_between_tick_and_update_held_item_matches_vanillas_single_block() {
        /// java: GuiIngame.java:1089, transcribed as one block the way vanilla has it.
        fn vanilla(
            remaining: &mut i32,
            highlighting: &mut Option<ItemStack>,
            current: Option<&ItemStack>,
        ) {
            match current {
                None => *remaining = 0,
                Some(cur) => {
                    let same = highlighting
                        .as_ref()
                        .is_some_and(|prev| Hud::same_highlight(cur, prev));
                    if same {
                        if *remaining > 0 {
                            *remaining -= 1;
                        }
                    } else {
                        *remaining = 40;
                    }
                }
            }
            *highlighting = current.cloned();
        }

        let stone = ItemStack {
            item_id: 1,
            count: 1,
            damage: 0,
            nbt: None,
        };
        let dirt = ItemStack {
            item_id: 3,
            count: 1,
            damage: 0,
            nbt: None,
        };
        // A script that exercises all three branches, including the empty-hand reset.
        let script: Vec<Option<&ItemStack>> = vec![
            Some(&stone),
            Some(&stone),
            Some(&stone),
            Some(&dirt),
            Some(&dirt),
            None,
            None,
            Some(&stone),
        ];

        let mut hud = Hud::new();
        let (mut remaining, mut highlighting) = (0, None);
        for (tick, current) in script.into_iter().enumerate() {
            hud.tick();
            hud.update_held_item(current, "name");
            vanilla(&mut remaining, &mut highlighting, current);
            assert_eq!(
                hud.highlight_ticks, remaining,
                "tick {tick} diverged from the Java block"
            );
            assert_eq!(hud.highlighting, highlighting);
        }
    }

    /// The metadata comparison is short-circuited for damageable items, so a sword losing
    /// durability keeps its popup counting down instead of restarting it.
    #[test]
    fn damaging_a_sword_does_not_re_raise_its_name_but_recolouring_wool_does() {
        let mut hud = Hud::new();
        hud.update_held_item(Some(&damaged_sword()), "Diamond Sword");
        assert_eq!(hud.highlight_ticks, 40);
        hud.highlight_ticks = 5;
        let mut worse = damaged_sword();
        worse.damage += 1;
        hud.update_held_item(Some(&worse), "Diamond Sword");
        assert_eq!(hud.highlight_ticks, 5, "damage alone is not a new item");

        // Wool (id 35) is not damageable, so its metadata *is* compared.
        let white = ItemStack {
            item_id: 35,
            count: 1,
            damage: 0,
            nbt: None,
        };
        let red = ItemStack {
            item_id: 35,
            count: 1,
            damage: 14,
            nbt: None,
        };
        hud.update_held_item(Some(&white), "White Wool");
        hud.highlight_ticks = 5;
        hud.update_held_item(Some(&red), "Red Wool");
        assert_eq!(hud.highlight_ticks, 40);
        assert_eq!(hud.held_item_name, "Red Wool");
    }

    #[test]
    fn an_empty_hand_clears_the_popup_immediately() {
        let mut hud = Hud::new();
        hud.update_held_item(Some(&damaged_sword()), "Diamond Sword");
        assert_eq!(hud.highlight_ticks, 40);
        hud.update_held_item(None, "");
        assert_eq!(hud.highlight_ticks, 0);
        assert!(hud.highlighting.is_none());
    }

    #[test]
    fn the_held_item_name_fades_over_ten_ticks_and_stops_at_forty() {
        let mut hud = Hud::new();
        hud.held_item_name = "Diamond Sword".into();
        hud.highlight_ticks = 40;
        let c = ctx();
        let mut font = uniform_font();
        let mut d = DrawList::new();
        hud.draw(&mut d, &mut font, None, &c, &PlayerHud::default());
        let glyphs = d
            .batches()
            .iter()
            .filter(|b| matches!(b.texture, UiTexture::Font(_)))
            .count();
        assert!(glyphs > 0, "the name is drawn while the timer runs");

        hud.highlight_ticks = 0;
        let mut d = DrawList::new();
        hud.draw(&mut d, &mut font, None, &c, &PlayerHud::default());
        let glyphs = d
            .batches()
            .iter()
            .filter(|b| matches!(b.texture, UiTexture::Font(_)))
            .count();
        assert_eq!(glyphs, 0, "and not once it has expired");
    }

    #[test]
    fn creative_held_item_names_shift_down_fourteen_pixels() {
        let mut hud = Hud::new();
        hud.held_item_name = "Stick".into();
        hud.highlight_ticks = 40;
        let c = ctx();

        let top = |survival| {
            let mut font = uniform_font();
            let mut d = DrawList::new();
            hud.draw(
                &mut d,
                &mut font,
                None,
                &c,
                &PlayerHud {
                    survival,
                    ..PlayerHud::default()
                },
            );
            d.batches()
                .iter()
                .filter(|b| matches!(b.texture, UiTexture::Font(_)))
                .flat_map(|b| d.vertices()[b.range.clone()].iter())
                .map(|v| v.pos[1])
                .fold(f32::INFINITY, f32::min)
        };
        assert_eq!(top(true), (c.height - 59) as f32);
        assert_eq!(top(false), (c.height - 45) as f32);
    }

    /// **The regression this field exists for.** The action bar and the held-item name are two
    /// different vanilla fields; when they shared one, driving the held-item popup per tick
    /// overwrote every action-bar message the server sent on the very next tick.
    #[test]
    fn the_action_bar_and_the_held_item_name_do_not_overwrite_each_other() {
        let mut hud = Hud::new();
        hud.set_record_playing("§aYou joined the game", false);
        // A tick of held-item feeding, which is what `GuiIngame.updateTick` does every tick.
        hud.tick();
        let compass = ItemStack {
            item_id: 345, // minecraft:compass
            count: 1,
            damage: 0,
            nbt: None,
        };
        hud.update_held_item(Some(&compass), "Compass");
        assert_eq!(hud.record_playing, "§aYou joined the game");
        assert_eq!(hud.held_item_name, "Compass");
        assert_eq!(hud.record_ticks, 59);
        assert_eq!(hud.highlight_ticks, 40);
    }

    /// java: the alpha divisor is `20` while the counter starts at `60`, so the message is fully
    /// opaque for its first 40 ticks and then fades. Deriving the alpha from `60` instead — the
    /// obvious "simplification" — would make it translucent from the first frame.
    #[test]
    fn the_action_bar_holds_full_alpha_for_forty_ticks_then_fades() {
        let mut hud = Hud::new();
        hud.set_record_playing("hello", false);
        assert_eq!(hud.record_color(0.0), Some(0xFFFF_FFFFu32 as i32));
        for _ in 0..20 {
            hud.tick();
        }
        assert_eq!(hud.record_ticks, 40);
        assert_eq!(
            hud.record_color(0.0),
            Some(0xFFFF_FFFFu32 as i32),
            "still 255"
        );
        for _ in 0..21 {
            hud.tick();
        }
        assert_eq!(hud.record_ticks, 19);
        // 19 * 255 / 20 = 242.
        assert_eq!(hud.record_color(0.0), Some(0xF2FF_FFFFu32 as i32));
        // java: `> 8` — the message disappears before the counter does.
        for _ in 0..19 {
            hud.tick();
        }
        assert_eq!(hud.record_ticks, 0);
        assert_eq!(hud.record_color(0.0), None);
    }

    /// The jukebox branch cycles hue; the action-bar branch is plain white. Only the flag
    /// distinguishes them, so a port that ignored it would rainbow every server message.
    #[test]
    fn only_the_jukebox_branch_cycles_colour() {
        let mut plain = Hud::new();
        plain.set_record_playing("x", false);
        let mut record = Hud::new();
        record.set_record_playing("x", true);
        assert_eq!(plain.record_color(0.0).unwrap() & 0x00FF_FFFF, 0x00FF_FFFF);
        assert_ne!(record.record_color(0.0).unwrap() & 0x00FF_FFFF, 0x00FF_FFFF);
        assert_eq!(
            record.record_color(0.0).unwrap() & 0xFF00_0000u32 as i32,
            0xFF00_0000u32 as i32,
            "the alpha is the same in both branches"
        );
    }

    #[test]
    fn the_action_bar_draws_above_the_held_item_line() {
        let mut hud = Hud::new();
        hud.set_record_playing("bar", false);
        let c = ctx();
        let mut font = uniform_font();
        let mut d = DrawList::new();
        hud.draw(&mut d, &mut font, None, &c, &PlayerHud::default());
        let glyphs = d
            .batches()
            .iter()
            .filter(|b| matches!(b.texture, UiTexture::Font(_)))
            .count();
        assert!(glyphs > 0, "the overlay message is drawn");

        // java: `translate(i / 2, j - 68, 0)` then `drawString(.., -w / 2, -4, ..)`.
        let vertices = d.vertices().to_vec();
        let top = d
            .batches()
            .iter()
            .filter(|b| matches!(b.texture, UiTexture::Font(_)))
            .flat_map(|b| vertices[b.range.clone()].iter())
            .map(|v| v.pos[1])
            .fold(f32::INFINITY, f32::min);
        assert!(
            (top - (c.height - 72) as f32).abs() < 1.0,
            "expected the overlay message at height - 72, got {top}"
        );
    }

    /// java: `renderGameOverlay` draws F3 (`:240-243`) between the held-item name (`:226-233`)
    /// and the action bar (`:245`)/title (`:277`). The client draws F3 between
    /// `draw_layers_below_debug` and `draw_layers_above_debug`, so the overlay messages must be in
    /// the upper half and everything else in the lower one, and the two halves must add up to
    /// the single-call entry point.
    #[test]
    fn the_overlay_messages_are_the_half_drawn_after_f3() {
        let mut hud = Hud::new();
        hud.set_record_playing("bar", false);
        hud.display_title("Big", "small");
        for _ in 0..5 {
            hud.tick();
        }
        hud.held_item_name = "Sword".to_owned();
        hud.highlight_ticks = 40;
        let c = ctx();
        let state = PlayerHud::default();
        let mut whole = DrawList::new();
        hud.draw(&mut whole, &mut uniform_font(), None, &c, &state);

        let mut below = DrawList::new();
        hud.draw_layers_below_debug(
            &mut below,
            &mut uniform_font(),
            None,
            &c,
            &state,
            false,
            CrosshairState::default(),
        );
        let mut above = DrawList::new();
        hud.draw_layers_above_debug(&mut above, &mut uniform_font(), &c, &state);
        let mut halves = DrawList::new();
        hud.draw_layers_below_debug(
            &mut halves,
            &mut uniform_font(),
            None,
            &c,
            &state,
            false,
            CrosshairState::default(),
        );
        hud.draw_layers_above_debug(&mut halves, &mut uniform_font(), &c, &state);
        assert_eq!(halves, whole, "the two halves are the whole HUD");

        let font_ys = |d: &DrawList| -> Vec<f32> {
            let vertices = d.vertices();
            d.batches()
                .iter()
                .filter(|b| matches!(b.texture, UiTexture::Font(_)))
                .flat_map(|b| vertices[b.range.clone()].iter().map(|v| v.pos[1]))
                .collect()
        };
        let upper = font_ys(&above);
        assert!(
            upper
                .iter()
                .any(|&y| (y - (c.height - 72) as f32).abs() < 1.0),
            "the action bar is drawn after F3"
        );
        assert!(
            upper.iter().any(|&y| y < (c.height / 2) as f32),
            "the title is drawn after F3"
        );
        let lower = font_ys(&below);
        assert!(
            lower
                .iter()
                .any(|&y| (y - (c.height - 59) as f32).abs() < 1.0),
            "the held-item name is drawn before F3"
        );
        assert!(
            !upper
                .iter()
                .any(|&y| (y - (c.height - 59) as f32).abs() < 1.0),
            "the held-item name is not in the half drawn over F3"
        );
    }

    /// The `action_bar` / `titles` mods detach exactly their own text from the
    /// vanilla draw, and the default (attached) draw is the untouched vanilla one.
    #[test]
    fn detached_action_bar_and_titles_are_skipped_and_nothing_else_changes() {
        let draw = |action_bar: bool, titles: bool, record: bool, title: bool| {
            let mut hud = Hud::new();
            if record {
                hud.set_record_playing("bar text", false);
            }
            if title {
                hud.display_title("Big", "small");
            }
            // Past the title's fade-in start (alpha 0 on its first tick).
            for _ in 0..5 {
                hud.tick();
            }
            hud.action_bar_detached = action_bar;
            hud.titles_detached = titles;
            let mut d = DrawList::new();
            hud.draw(
                &mut d,
                &mut uniform_font(),
                None,
                &ctx(),
                &PlayerHud::default(),
            );
            d
        };
        assert!(!Hud::new().action_bar_detached && !Hud::new().titles_detached);
        let bare = draw(false, false, false, false);
        // Detaching with nothing to draw changes nothing.
        assert_eq!(draw(true, true, false, false), bare);
        // A detached message or title draws exactly what the HUD draws without it.
        assert_eq!(draw(true, false, true, false), bare);
        assert_eq!(draw(false, true, false, true), bare);
        assert_eq!(draw(true, true, true, true), bare);
        // Attached, both really draw (more glyph quads than the bare HUD).
        assert!(draw(false, false, true, false).vertices().len() > bare.vertices().len());
        assert!(draw(false, false, false, true).vertices().len() > bare.vertices().len());
        // Detaching one leaves the other.
        assert_eq!(
            draw(true, false, true, true),
            draw(false, false, false, true)
        );
        assert_eq!(
            draw(false, true, true, true),
            draw(false, false, true, false)
        );
    }

    // ---- the scoreboard sidebar ----------------------------

    fn scores(pairs: &[(&str, i32)]) -> Vec<(String, i32)> {
        pairs.iter().map(|(n, p)| ((*n).to_owned(), *p)).collect()
    }

    /// java: the `Predicate<Score>` at `renderScoreboard:556` drops `#`-prefixed rows, and it
    /// tests the **raw** name — before `formatPlayerName` could move the `#` off the front.
    #[test]
    fn hash_prefixed_rows_are_hidden_and_the_rest_are_team_formatted() {
        let sidebar = Sidebar::from_scores(
            "Kills",
            &scores(&[("#total", 99), ("bob", 3), ("ann", 7)]),
            |name| format!("[R] {name}"),
        );
        assert_eq!(
            sidebar.visible_rows(),
            [("[R] bob".to_owned(), 3), ("[R] ann".to_owned(), 7)]
        );
        assert_eq!(sidebar.title, "Kills");
    }

    /// java: `if (list.size() > 15) collection = Iterables.skip(list, collection.size() - 15)`.
    /// `collection.size()` is the **unfiltered** count, so a board carrying hidden `#` rows shows
    /// fewer than 15. Reproduced deliberately — a port that "fixed" it would show a row vanilla
    /// does not.
    #[test]
    fn the_fifteen_row_cap_keeps_the_top_of_the_board_and_reproduces_vanillas_skip() {
        let many: Vec<(String, i32)> = (0..20).map(|i| (format!("p{i:02}"), i)).collect();
        let sidebar = Sidebar::from_scores("Kills", &many, str::to_owned);
        let rows = sidebar.visible_rows();
        assert_eq!(rows.len(), 15);
        assert_eq!(rows[0].1, 5, "the bottom five scores are dropped");
        assert_eq!(rows[14].1, 19, "the highest score is last, i.e. drawn top");

        // Now with two hidden rows in the mix: 20 raw, 18 shown, and vanilla still skips
        // `20 - 15 = 5` of them, leaving 13 rather than 15.
        let mut mixed = many.clone();
        mixed.insert(0, ("#a".to_owned(), -2));
        mixed.insert(1, ("#b".to_owned(), -1));
        let sidebar = Sidebar::from_scores("Kills", &mixed, str::to_owned);
        assert_eq!(
            sidebar.visible_rows().len(),
            13,
            "22 raw, 20 shown, skip 22 - 15 = 7 → 13 rows, not 15. This *is* vanilla."
        );
        assert_eq!(sidebar.visible_rows()[0].1, 7);
    }

    /// A short board is drawn whole, with no skipping at all.
    #[test]
    fn a_short_board_draws_every_row() {
        let sidebar = Sidebar::from_scores("Kills", &scores(&[("a", 1), ("b", 2)]), str::to_owned);
        assert_eq!(sidebar.visible_rows().len(), 2);
        let empty = Sidebar::from_scores("Kills", &[], str::to_owned);
        assert!(empty.visible_rows().is_empty());
    }

    /// The geometry, hand-derived from `renderScoreboard` rather than from this implementation:
    /// rows walk **upward** from `height / 2 + rows * 9 / 3`, the box is right-aligned with a
    /// 3-pixel margin, and the header plate sits above the topmost row.
    #[test]
    fn the_sidebar_lands_on_vanillas_pixels() {
        let mut font = uniform_font();
        let c = ctx();
        let hud = Hud::new();
        let sidebar = Sidebar::from_scores(
            "Kills",
            &scores(&[("a", 1), ("b", 2), ("c", 3)]),
            str::to_owned,
        );
        let mut d = DrawList::new();
        hud.draw_scoreboard(&mut d, &mut font, &c, &sidebar);

        // Three row plates, plus the header plate and its one-pixel sliver.
        let vertices = d.vertices().to_vec();
        let plate_tops: Vec<f32> = d
            .batches()
            .iter()
            .filter(|b| matches!(b.texture, UiTexture::Solid))
            .flat_map(|b| vertices[b.range.clone()].iter())
            .map(|v| v.pos[1])
            .collect();
        assert!(!plate_tops.is_empty(), "the plates must be drawn");

        // java: `j1 = scaledHeight / 2 + i1 / 3` with `i1 = 3 * 9`, so the *first* row's top is
        // `j1 - 1 * 9`.
        let baseline = c.height / 2 + (3 * 9) / 3;
        let first_row_top = (baseline - 9) as f32;
        assert!(
            plate_tops
                .iter()
                .any(|y| (*y - first_row_top).abs() < 0.001),
            "expected a row plate at y = {first_row_top}, got {plate_tops:?}"
        );
        // The topmost row is `baseline - 3 * 9`, and the header plate is one font height above it.
        let top_row = (baseline - 27) as f32;
        assert!(plate_tops.iter().any(|y| (*y - top_row).abs() < 0.001));
        assert!(
            plate_tops
                .iter()
                .any(|y| (*y - (top_row - 9.0 - 1.0)).abs() < 0.001),
            "the objective title's plate sits a font height above the top row"
        );

        // Right-aligned: nothing may be drawn past `width - 3 + 2`.
        let right = d
            .vertices()
            .iter()
            .map(|v| v.pos[0])
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (right - (c.width - 3 + 2) as f32).abs() < 0.001,
            "the box's right edge is width - 3 + 2, got {right}"
        );

        // An empty sidebar draws absolutely nothing rather than an empty plate.
        let mut empty = DrawList::new();
        hud.draw_scoreboard(&mut empty, &mut font, &c, &Sidebar::default());
        assert!(empty.batches().is_empty());
    }
}
