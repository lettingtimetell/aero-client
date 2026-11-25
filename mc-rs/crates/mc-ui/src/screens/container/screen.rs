//! [`ContainerScreen`]: the on-screen GUI for every server-synced container kind. java:
//! `client/gui/inventory/GuiContainer.java` plus its nine concrete subclasses (`GuiInventory`,
//! `GuiChest`, `GuiDispenser`, `GuiFurnace`, `GuiBrewingStand`, `GuiHopper`, `GuiEnchantment`,
//! `GuiRepair`, `GuiBeacon`).
//!
//! This is the piece that turns S17's tested backend into something a player can look at and
//! click. It wires together, per frame:
//!
//! ```text
//!   ContainerArt::draw_background   panel art from textures/gui/container/*.png + dynamic overlays
//!   render::draw_item_in_slot       every slot's stack, count and durability bar
//!   the 0x80ffffff hover gradient   over the 16x16 slot under the cursor
//!   ContainerArt::draw_foreground   the container's and the player's inventory titles
//!   render::draw_item_in_slot       the cursor stack, at (mouseX - 8, mouseY - 8)
//!   tooltip_box::draw(tooltip::build(..))
//! ```
//!
//! and, per event, `interaction::MouseState` -> [`net::Container::click`](super::net) ->
//! [`ContainerScreen::take_outgoing`].
//!
//! # Why one struct instead of nine
//!
//! Java gets one subclass per container because `drawGuiContainerBackgroundLayer` is abstract
//! and every subclass overrides it. The overrides differ only in *which* texture is blitted and
//! *what* dynamic art sits on top, so this port keeps one [`ContainerScreen`] carrying a
//! [`ContainerArt`] enum: the shared `GuiContainer` behaviour (slot loop, hover, tooltip, mouse
//! routing) exists once, and the per-kind part is a `match` in one place.
//!
//! # What is *not* reproduced
//!
//! - **The rotating 3-D player model**'s live limb-swing animation and equipped layers (armour,
//!   held item, cape) — the panel itself (`GuiInventory.drawEntityOnScreen`) is drawn, by
//!   [`super::player_preview`]; see that module's doc for exactly what is and is not reproduced
//!   about it.
//! - **The enchanting table's animated book** (`ModelBook` rendered through a second projection
//!   matrix) is still render-owned; its CPU spring state is exposed by
//!   [`EnchantingBookAnimation`] / [`EnchantingBookPose`]. The **Standard Galactic rune text**
//!   still needs `mc.standardGalacticFontRenderer` — a third font sheet S15 does not load. The
//!   galley plates, level pips, hover state and level numbers *are* drawn.
//! - The drag-splitting preview and the parallel touchscreen `clickedSlot`/`draggedStack`/
//!   `returningStack` gesture are both reproduced. The touchscreen return is GUI-only: it flies
//!   back to the source slot for 100 ms and never creates a world `EntityItem`.

use std::collections::BTreeSet;

use glam::Vec3;
use mc_assets::Language;
use mc_math::math_helper::truncate_f64_to_i32;
use mc_math::random::JavaRandom;
use mc_protocol::play::{c2s, s2c};
use mc_render::entity::{
    tilepass::{enchant_book_screen_model_matrix, project_enchant_book_vertex},
    EntityVertex, SkinModel, TileModels,
};
use mc_world::MapData;

use crate::font::{FontRenderer, TextMesh, TextSink};
use crate::screen_stack::{Screen, ScreenAction, ScreenCtx};
use crate::screens::hud::{ActiveEffect, INVENTORY_TEXTURE};
use crate::widget::keys::Modifiers;
use crate::widget::{keys, DrawList, MemoryClipboard, TextField, UiTexture};

use super::interaction::{MouseState, SlotTarget};
use super::model::Container;

// The 1.7.10 screens' rules (enchanting gate and plates, furnace properties, anvil field): a
// child module, for this type's private state.
#[path = "screen_v5.rs"]
mod v5;
use super::player_preview::{HorsePreviewSnapshot, PlayerPreviewRenderer, PlayerPreviewSnapshot};
use super::stack_ext::{self, ItemStack};
use super::tooltip_box::{self, WORLD_DIM_BOTTOM, WORLD_DIM_TOP};
use super::{layouts, render, tooltip};

/// The hover highlight vanilla paints over the slot under the cursor, `-2130706433`. java:
/// `GuiContainer.java:134` — `drawGradientRect(x, y, x + 16, y + 16, -2130706433, -2130706433)`.
pub const SLOT_HIGHLIGHT: u32 = 0x80FF_FFFF;

/// The dark grey vanilla draws container titles in, `4210752`. java: every
/// `drawGuiContainerForegroundLayer`'s `fontRendererObj.drawString(..., 4210752)`.
pub const TITLE_COLOR: i32 = 0x0040_4040;

/// The lighter grey the beacon's two column headings use, `14737632`. java:
/// `GuiBeacon.java:171`.
pub const BEACON_HEADING_COLOR: i32 = 0x00E0_E0E0;

/// java: `ItemStack.isItemEnchantable` -> `Item.isItemTool(stack) && !isItemEnchanted()`.
///
/// `ItemBook` is the one useful override: a single ordinary book is a tool even though its
/// registry stack limit is 64. `ItemEnchantedBook` overrides the predicate to false. The only
/// other override, `ItemFishingRod`, delegates straight back to `Item`.
fn is_item_enchantable(stack: &ItemStack) -> bool {
    let Some(props) = stack_ext::try_props(stack) else {
        return false;
    };
    let is_item_tool = match props.name {
        "minecraft:book" => stack.count == 1,
        "minecraft:enchanted_book" => false,
        _ => props.max_stack_size == 1 && props.damageable,
    };
    is_item_tool && !tooltip::is_item_enchanted(stack)
}

/// The bright green an enchanting-table offer's level number is drawn in when the offer can be
/// taken, `8453920`. java: `GuiEnchantment`'s `i2 = 8453920`.
///
/// Named rather than inline because it was mis-transcribed as `0x0081_00A0` — a purple, and the
/// most conspicuous element of that screen. The decimal is pinned by a
/// test so the hex can never drift from the java again.
pub const ENCHANT_LEVEL_COLOR: i32 = 0x0080_FF20;

/// The dark green the same number uses when the offer cannot be taken, `4226832`. java:
/// `GuiEnchantment`'s `i2 = 4226832`. Was `0x0040_7830`.
pub const ENCHANT_LEVEL_DISABLED_COLOR: i32 = 0x0040_7F10;

/// The bright green the anvil's "Enchantment Cost: N" label uses, `8453920`. java:
/// `GuiRepair.java:77` — `int i = 8453920`. The same decimal as
/// [`ENCHANT_LEVEL_COLOR`], but a separate java constant in a separate class, so it is named
/// separately here too.
pub const ANVIL_COST_COLOR: i32 = 0x0080_FF20;

/// The red the anvil's cost label turns when the repair is too expensive **or** unaffordable,
/// `16736352`. java: `GuiRepair.java:83,92` — `i = 16736352` in both arms.
pub const ANVIL_COST_DENIED_COLOR: i32 = 0x00FF_6060;

/// LWJGL scan code for `Q`, vanilla's default "drop item" binding. `widget::keys` (S15, frozen)
/// carries only the keys its own widgets need, so the container-specific ones are named here.
pub const KEY_DROP: i32 = 16;
/// LWJGL scan code for `E`, vanilla's default "open inventory" binding.
pub const KEY_INVENTORY: i32 = 18;
/// LWJGL scan code for the `1` key. `2..=9` follow contiguously, hence `KEY_HOTBAR_1 + n`.
pub const KEY_HOTBAR_1: i32 = 2;
/// LWJGL scan code for `T`, vanilla's default "open chat" binding.
pub const KEY_CHAT: i32 = 20;

/// The mouse button vanilla's default "pick block" binding resolves to.
///
/// java: `GuiContainer.mouseClicked` compares `mouseButton == keyBindPickBlock.getKeyCode() +
/// 100`; a mouse binding stores `button - 100`, and pick-block defaults to the middle button
/// (`2`), so the comparison is against `2`.
pub const DEFAULT_PICK_BLOCK_BUTTON: i32 = 2;

/// The *keyboard* scan code vanilla's default pick-block binding resolves to — which is none.
///
/// java: `KeyBinding.getKeyCode()` for a mouse binding is `button - 100`, so the default middle
/// button is `-98`, and `GuiContainer.keyTyped`'s `keyCode == keyBindPickBlock.getKeyCode()`
/// branch is therefore **inert** with the default bindings: no keyboard scan code is negative.
///
/// This constant exists because a single `pick_block_button` field was compared against *both*
/// a mouse button index and an LWJGL scan code. LWJGL scan code `2` is the
/// `1` key, so with A07-01 landed, pressing `1` over a slot while holding a stack would have put
/// a spurious mode-3 `ClickWindow` on the wire and stepped the transaction counter past what the
/// server expects.
pub const DEFAULT_PICK_BLOCK_KEY: i32 = -98;

/// The live key-binding snapshot used by container GUIs.
///
/// Java reads `GameSettings` directly from `GuiContainer`, so rebinding an action takes effect
/// without reconstructing the screen. This value keeps the UI independent from the host input
/// backend while preserving that live handoff at each event boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContainerBindings {
    /// LWJGL scan code for the inventory close action.
    pub inventory_key: i32,
    /// LWJGL scan code for the drop action.
    pub drop_key: i32,
    /// LWJGL scan codes for hotbar slots zero through eight.
    pub hotbar_keys: [i32; 9],
    /// LWJGL scan code for keyboard-bound pick block (negative while mouse-bound).
    pub pick_block_key: i32,
    /// Zero-based mouse button for pick block, or `-1` while keyboard-bound/unbound.
    pub pick_block_button: i32,
    /// LWJGL code of the chat binding (`button - 100` while mouse-bound). Off its search tab,
    /// `GuiContainerCreative.keyTyped` switches to search instead of handling the key.
    pub chat_key: i32,
}

impl Default for ContainerBindings {
    fn default() -> Self {
        Self {
            inventory_key: KEY_INVENTORY,
            drop_key: KEY_DROP,
            hotbar_keys: std::array::from_fn(|slot| KEY_HOTBAR_1 + slot as i32),
            pick_block_key: DEFAULT_PICK_BLOCK_KEY,
            pick_block_button: DEFAULT_PICK_BLOCK_BUTTON,
            chat_key: KEY_CHAT,
        }
    }
}

/// One vanilla client tick in milliseconds, used by [`ContainerScreen::tick`] to advance the
/// double-click clock when the host has not set a real one.
const TICK_MS: u64 = 50;

/// `textures/gui/container/` prefix for every panel texture this module blits.
fn container_texture(name: &str) -> UiTexture {
    UiTexture::gui(&format!("textures/gui/container/{name}"))
}

/// The block/item atlas key items are drawn from.
///
/// [`render::draw_item_in_slot`] samples `mc_assets::BakedAssets::atlas`, which the host uploads
/// under a [`UiTexture`] key of its own choosing; this is the default that key defaults to.
#[must_use]
pub fn default_atlas_texture() -> UiTexture {
    UiTexture::gui("textures/atlas/blocks.png")
}

// ---------------------------------------------------------------- per-kind dynamic art

/// A furnace's fire and progress arrow. java: `GuiFurnace.getBurnLeftScaled` /
/// `getCookProgressScaled`, fed by `S31PacketWindowProperty` (`TileEntityFurnace.getField`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FurnaceArt {
    /// Field 0: ticks of fuel left.
    pub burn_time: i32,
    /// Field 1: how long the current fuel item burns for in total.
    pub current_item_burn_time: i32,
    /// Field 2: ticks the current item has been smelting.
    pub cook_time: i32,
    /// Field 3: ticks a full smelt takes.
    pub total_cook_time: i32,
}

impl FurnaceArt {
    /// java: `TileEntityFurnace.isBurning`.
    #[must_use]
    pub const fn is_burning(&self) -> bool {
        self.burn_time > 0
    }

    /// java: `GuiFurnace.getBurnLeftScaled` — note the `i == 0 -> 200` fallback, which is what
    /// keeps the flame from dividing by zero on the first tick after a window opens.
    #[must_use]
    pub const fn burn_left_scaled(&self, pixels: i32) -> i32 {
        let denominator = if self.current_item_burn_time == 0 {
            200
        } else {
            self.current_item_burn_time
        };
        self.burn_time * pixels / denominator
    }

    /// java: `GuiFurnace.getCookProgressScaled`.
    #[must_use]
    pub const fn cook_progress_scaled(&self, pixels: i32) -> i32 {
        if self.total_cook_time != 0 && self.cook_time != 0 {
            self.cook_time * pixels / self.total_cook_time
        } else {
            0
        }
    }
}

/// A brewing stand's bubble column and brew arrow. java: `GuiBrewingStand`, field 0 =
/// `brewTime` counting **down** from 400.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BrewingStandArt {
    /// Field 0: ticks left in the current brew, `0` when idle.
    pub brew_time: i32,
}

impl BrewingStandArt {
    /// The arrow's filled height in pixels, `0..=28`. java: `(int)(28.0F * (1.0F - k / 400.0F))`.
    #[must_use]
    pub fn arrow_height(&self) -> i32 {
        if self.brew_time <= 0 {
            return 0;
        }
        (28.0 * (1.0 - self.brew_time as f32 / 400.0)) as i32
    }

    /// The bubble column's height, from vanilla's seven-entry lookup on `brewTime / 2 % 7`.
    /// java: `GuiBrewingStand`'s `switch (i1)` — deliberately a table, not a formula, because
    /// the steps are not evenly spaced (`29, 24, 20, 16, 11, 6, 0`).
    #[must_use]
    pub fn bubble_height(&self) -> i32 {
        if self.brew_time <= 0 {
            return 0;
        }
        const STEPS: [i32; 7] = [29, 24, 20, 16, 11, 6, 0];
        STEPS[(self.brew_time / 2 % 7) as usize]
    }
}

/// An enchanting table's three offers. java: `GuiEnchantment` +
/// `ContainerEnchantment.updateProgressBar` (0..2 levels, 3 xp seed, 4..6 enchantment ids).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnchantingArt {
    /// Fields 0..2: the XP level each of the three offers costs, `0` for "no offer".
    pub enchant_levels: [i32; 3],
    /// Field 3: the seed vanilla uses to generate the rune text (not rendered — see the module
    /// doc) and the clue enchantment.
    pub xp_seed: i32,
    /// Fields 4..6: the hinted enchantment id/level, `-1` for "unknown".
    pub enchantment_ids: [i32; 3],
    /// The player's current XP level, for the "can I afford this" test. Not a window property —
    /// the host supplies it.
    pub player_level: i32,
    /// Whether the player is in creative mode, which bypasses the level/lapis gates.
    pub creative: bool,
}

impl Default for EnchantingArt {
    fn default() -> Self {
        Self {
            enchant_levels: [0; 3],
            xp_seed: 0,
            // java: `ContainerEnchantment` initialises every clue to `-1`. A derived default
            // made an untouched row look like Protection level 0 as soon as the hover tooltip
            // gained a real reader for this field.
            enchantment_ids: [-1; 3],
            player_level: 0,
            creative: false,
        }
    }
}

impl EnchantingArt {
    /// The three source names vanilla hands to its Standard Galactic font renderer.
    ///
    /// `GuiEnchantment` reseeds `EnchantmentNameParts` from the signed, widened `xpSeed` once
    /// per background draw, then generates a name for **every** row before checking whether that
    /// row has an offer. Keeping that sequence here makes field 3's otherwise opaque value
    /// inspectable and gives a future `ascii_sga.png` renderer its exact text source. It does
    /// not draw with the ordinary font: vanilla uses the separate `ascii_sga.png` sheet, which
    /// is not loaded by the current font API.
    // java: GuiEnchantment.java:172-181; EnchantmentNameParts.java:17-35.
    #[must_use]
    pub fn rune_offer_names(&self) -> [String; 3] {
        const PARTS: [&str; 55] = [
            "the",
            "elder",
            "scrolls",
            "klaatu",
            "berata",
            "niktu",
            "xyzzy",
            "bless",
            "curse",
            "light",
            "darkness",
            "fire",
            "air",
            "earth",
            "water",
            "hot",
            "dry",
            "cold",
            "wet",
            "ignite",
            "snuff",
            "embiggen",
            "twist",
            "shorten",
            "stretch",
            "fiddle",
            "destroy",
            "imbue",
            "galvanize",
            "enchant",
            "free",
            "limited",
            "range",
            "of",
            "towards",
            "inside",
            "sphere",
            "cube",
            "self",
            "other",
            "ball",
            "mental",
            "physical",
            "grow",
            "shrink",
            "demon",
            "elemental",
            "spirit",
            "animal",
            "creature",
            "beast",
            "humanoid",
            "undead",
            "fresh",
            "stale",
        ];

        let mut random = JavaRandom::new(i64::from(self.xp_seed));
        std::array::from_fn(|_| {
            let words = random.next_i32_bound(2) + 3;
            (0..words)
                .map(|_| PARTS[random.next_i32_bound(PARTS.len() as i32) as usize])
                .collect::<Vec<_>>()
                .join(" ")
        })
    }

    /// Whether offer `index` is currently affordable, which is what picks between vanilla's
    /// lit and dimmed plate art. java: the
    /// `(k < l + 1 || experienceLevel < l1) && !isCreativeMode` test in
    /// `GuiEnchantment.drawGuiContainerBackgroundLayer`, where `k` is the lapis count.
    #[must_use]
    pub fn is_affordable(&self, index: usize, lapis: i32) -> bool {
        let Some(&level) = self.enchant_levels.get(index) else {
            return false;
        };
        if level == 0 {
            return false;
        }
        // java: `k < l + 1` (lapis short of the offer number) or `experienceLevel < l1`.
        self.creative || (lapis > index as i32 && self.player_level >= level)
    }

    /// Client-side half of `ContainerEnchantment.enchantItem`. On a remote world vanilla does
    /// not spend levels/lapis or mutate the item here; it only decides whether
    /// `GuiEnchantment.mouseClicked` may send `C11PacketEnchantItem`.
    #[must_use]
    pub fn can_activate(&self, index: usize, lapis: i32, has_item: bool) -> bool {
        let Some(&level) = self.enchant_levels.get(index) else {
            return false;
        };
        has_item
            && level > 0
            && (self.creative
                || (lapis > index as i32
                    && self.player_level > index as i32
                    && self.player_level >= level))
    }
}

/// The CPU-side state of `GuiEnchantment.func_147068_g`, which drives the book above the table.
///
/// The actual `ModelBook` projection is a render-owned concern, but the screen owns the six
/// values that feed it: the page-flip spring (`field_147071_v`/`field_147082_x`/
/// `field_147081_y`) and the opening spring (`field_147080_z`/`field_147076_A`). Keeping this
/// state here makes the tick ordering and the interpolated values available to the renderer
/// without pretending that a 2-D `DrawList` can draw the 3-D model itself.
///
/// java: `GuiEnchantment.java:289-334`, `:125-166`; `ModelBook.setRotationAngles:38-50`.
#[derive(Debug, Clone, PartialEq)]
pub struct EnchantingBookAnimation {
    /// `field_147073_u`: ticks since the screen was created.
    pub ticks: i32,
    /// `field_147071_v`: current page-flip spring position.
    pub page_flip: f32,
    /// `field_147069_w`: page-flip position at the start of the current tick.
    pub page_flip_prev: f32,
    /// `field_147082_x`: page-flip spring target.
    pub page_flip_target: f32,
    /// `field_147081_y`: page-flip spring velocity.
    pub page_flip_velocity: f32,
    /// `field_147080_z`: current book opening amount, clamped to `0..=1`.
    pub spread: f32,
    /// `field_147076_A`: book opening amount at the start of the current tick.
    pub spread_prev: f32,
    random: JavaRandom,
    last_item: Option<ItemStack>,
}

/// The interpolated values a `ModelBook` draw needs for one frame.
///
/// `page_flip_right` and `page_flip_left` already include vanilla's `+0.25`/`+0.75`,
/// fractional reduction, `*1.6 - 0.3`, and `0..=1` clamp. They are the `limbSwingAmount` and
/// `ageInTicks` arguments passed to `ModelBook.render`; `spread` is its `netHeadYaw` argument.
/// `limb_swing` is the screen tick counter plus the frame partial. It is **not** the model's
/// `limbSwing`: `GuiEnchantment.java:162` passes a literal `0.0F` there, so the GUI book has no
/// breathing term.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnchantingBookPose {
    /// Interpolated book opening amount (`f2`).
    pub spread: f32,
    /// Right flipping-page amount (`f3`).
    pub page_flip_right: f32,
    /// Left flipping-page amount (`f4`).
    pub page_flip_left: f32,
    /// `GuiEnchantment`'s `field_147073_u + partialTicks`. Vanilla never feeds this counter to
    /// `ModelBook`; it is kept for callers that read it.
    pub limb_swing: f32,
}

impl Default for EnchantingBookAnimation {
    fn default() -> Self {
        Self {
            ticks: 0,
            page_flip: 0.0,
            page_flip_prev: 0.0,
            page_flip_target: 0.0,
            page_flip_velocity: 0.0,
            spread: 0.0,
            spread_prev: 0.0,
            // Java's no-arg Random is process/time seeded. A fixed seed keeps this state
            // deterministic for headless callers; the seed only chooses the same kind of
            // cosmetic target perturbation and is not part of the server-visible protocol.
            random: JavaRandom::new(0),
            last_item: None,
        }
    }
}

impl EnchantingBookAnimation {
    /// Advance one `GuiEnchantment.updateScreen` tick.
    ///
    /// The target reroll happens only when `ItemStack.areItemStacksEqual` says slot 0 changed;
    /// it may consume several pairs of `nextInt(4)` until the target is more than one unit from
    /// the current position. The opening spring then advances from the offer array, and the
    /// page spring is updated last, in the same order as Java.
    pub fn tick(&mut self, item: Option<&ItemStack>, enchant_levels: &[i32; 3]) {
        if !stack_ext::stacks_equal(item, self.last_item.as_ref()) {
            self.last_item = item.cloned();
            loop {
                let delta = self.random.next_i32_bound(4) - self.random.next_i32_bound(4);
                self.page_flip_target += delta as f32;
                if self.page_flip > self.page_flip_target + 1.0
                    || self.page_flip < self.page_flip_target - 1.0
                {
                    break;
                }
            }
        }

        self.ticks = self.ticks.wrapping_add(1);
        self.page_flip_prev = self.page_flip;
        self.spread_prev = self.spread;

        let has_offer = enchant_levels.iter().any(|level| *level != 0);
        self.spread += if has_offer { 0.2 } else { -0.2 };
        self.spread = clamp_unit(self.spread);

        let mut spring = (self.page_flip_target - self.page_flip) * 0.4;
        spring = spring.clamp(-0.2, 0.2);
        self.page_flip_velocity += (spring - self.page_flip_velocity) * 0.9;
        self.page_flip += self.page_flip_velocity;
    }

    /// Interpolate the spring values for a render frame, matching `GuiEnchantment`'s float math.
    #[must_use]
    pub fn pose(&self, partial_ticks: f32) -> EnchantingBookPose {
        let spread = self.spread_prev + (self.spread - self.spread_prev) * partial_ticks;
        let page_base =
            self.page_flip_prev + (self.page_flip - self.page_flip_prev) * partial_ticks;
        EnchantingBookPose {
            spread,
            page_flip_right: book_page_flip(page_base + 0.25),
            page_flip_left: book_page_flip(page_base + 0.75),
            limb_swing: self.ticks as f32 + partial_ticks,
        }
    }
}

/// Java's `MathHelper.clamp_float(value, 0, 1)` without changing its NaN behaviour.
fn clamp_unit(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

/// `GuiEnchantment.drawGuiContainerBackgroundLayer`'s page-flip reduction and clamp.
fn book_page_flip(value: f32) -> f32 {
    let whole = truncate_f64_to_i32(f64::from(value));
    clamp_unit((value - whole as f32) * 1.6 - 0.3)
}

/// An anvil's XP cost and its "no valid output" cross. java: `GuiRepair` +
/// `ContainerRepair.updateProgressBar` (field 0 = `maximumCost`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AnvilArt {
    /// Field 0: the XP levels the repair costs, `0` when there is nothing to repair.
    pub maximum_cost: i32,
    /// Whether the player is in creative mode, which lifts the 40-level cap.
    pub creative: bool,
    /// Last known player XP level. `None` means the host has not synchronized it yet, in which
    /// case the visual affordability conjunct is left permissive rather than inventing level 0.
    pub player_level: Option<i32>,
    /// The contents of the rename box. Kept as a cheap public mirror for hosts that already
    /// inspect the art; [`ContainerScreen`] owns the live [`TextField`] and mutation queue.
    pub rename_text: String,
}

impl AnvilArt {
    /// java: `GuiRepair.drawGuiContainerForegroundLayer` — 40 levels or more is "Too Expensive!"
    /// unless the player is in creative mode.
    #[must_use]
    pub const fn is_too_expensive(&self) -> bool {
        self.maximum_cost >= 40 && !self.creative
    }

    /// The cost label vanilla paints for the current output state, or `None` when its second
    /// cascade arm suppresses the label. Keeping this decision as data lets hosts and tests
    /// observe the exact same result the renderer consumes.
    #[must_use]
    pub fn cost_label(&self, container: &Container) -> Option<AnvilCostLabel> {
        if self.maximum_cost <= 0 {
            return None;
        }
        if self.is_too_expensive() {
            return Some(AnvilCostLabel {
                text: "Too Expensive!".to_owned(),
                color: ANVIL_COST_DENIED_COLOR,
            });
        }
        container.slot_stack(2)?;
        let affordable = self.creative
            || self
                .player_level
                .is_none_or(|level| level >= self.maximum_cost);
        Some(AnvilCostLabel {
            text: format!("Enchantment Cost: {}", self.maximum_cost),
            color: if affordable {
                ANVIL_COST_COLOR
            } else {
                ANVIL_COST_DENIED_COLOR
            },
        })
    }
}

/// The resolved text and RGB colour of an anvil's XP-cost label.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnvilCostLabel {
    /// Localized label text. Currently the exact 1.8.9 en_US strings used by this screen.
    pub text: String,
    /// Vanilla RGB colour: bright green when affordable, red when denied.
    pub color: i32,
}

/// Legacy villager-trade state carried by `GuiMerchant`. Offers arrive out-of-band on the
/// `MC|TrList` plugin channel; only the selected index is sent back on `MC|TrSel`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MerchantArt {
    /// Server-provided offers in wire order.
    pub recipes: Vec<super::model::MerchantRecipe>,
    /// Current `ContainerMerchant.currentRecipeIndex`.
    pub selected: usize,
}

impl MerchantArt {
    /// The currently selected offer, if the server has sent any offers.
    #[must_use]
    pub fn selected_recipe(&self) -> Option<&super::model::MerchantRecipe> {
        self.recipes.get(self.selected)
    }

    /// Select an offer and return the exact legacy `MC|TrSel` payload.
    #[must_use]
    pub fn select(&mut self, index: usize) -> Option<Vec<u8>> {
        if index >= self.recipes.len() {
            return None;
        }
        self.selected = index;
        Some((index as i32).to_be_bytes().to_vec())
    }

    /// Java's next-offer arrow (`GuiMerchant`'s + button).
    #[must_use]
    // This public name mirrors the vanilla arrow operation and is not an iterator adapter.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<Vec<u8>> {
        self.select(self.selected.saturating_add(1))
    }

    /// Java's previous-offer arrow (`GuiMerchant`'s - button).
    #[must_use]
    pub fn previous(&mut self) -> Option<Vec<u8>> {
        self.selected
            .checked_sub(1)
            .and_then(|index| self.select(index))
    }
}

/// A beacon's pyramid level and chosen effects. java: `GuiBeacon` + `TileEntityBeacon`'s three
/// fields (0 levels, 1 primary effect, 2 secondary effect).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BeaconArt {
    /// Field 0: the pyramid's height, `0..=4`. The client-side
    /// `ContainerLocalMenu.getField` default is zero before a property update arrives.
    pub levels: i32,
    /// Field 1: the chosen primary effect's potion id, `0` for none.
    pub primary: i32,
    /// Field 2: the chosen secondary effect's potion id, `0` for none.
    pub secondary: i32,
}

/// java: `TileEntityBeacon.effectsList` — which potion effects each pyramid tier unlocks.
/// Tier 3 (index 3) is the secondary column, which also gains a "double the primary" entry.
pub const BEACON_EFFECTS: [&[u8]; 4] = [&[1, 3], &[11, 8], &[5], &[10]];

/// `TileEntityBeacon.func_183001_h`: only the six effects the beacon itself can grant are
/// retained when a server updates either selected-effect field.  This is deliberately a
/// whitelist, not merely an in-range potion check: poison, blindness, and every unregistered
/// id become the no-selection sentinel just as they do in vanilla.
// java: TileEntityBeacon.java:173-184
#[must_use]
pub const fn beacon_effect_or_none(effect_id: i32) -> i32 {
    match effect_id {
        1 | 3 | 5 | 8 | 10 | 11 => effect_id,
        _ => 0,
    }
}

/// java: `Potion.getStatusIconIndex()` for the six effects a beacon can grant.
///
/// `mc_data::PotionEffect::status_icon_index` now
/// carries this for all 23 potions, extracted from the real `Potion` class, so the beacon no
/// longer needs its own transcription of the six it happens to use.
#[must_use]
pub fn beacon_status_icon_index(potion_id: u8) -> Option<i32> {
    mc_data::potion(potion_id)?.status_icon_index
}

/// One of the beacon's 22x22 buttons, positioned in **window-relative** pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BeaconButton {
    /// Left edge, relative to `gui_left`.
    pub x: i32,
    /// Top edge, relative to `gui_top`.
    pub y: i32,
    /// What the button does.
    pub kind: BeaconButtonKind,
    /// Whether the button can be pressed. java: `GuiButton.enabled`.
    pub enabled: bool,
    /// Whether it is the currently chosen effect. java: `GuiBeacon.Button.field_146142_r`.
    pub selected: bool,
}

/// What a [`BeaconButton`] does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeaconButtonKind {
    /// Apply the chosen effects and close. java: button id `-1`, sends `MC|Beacon`.
    Confirm,
    /// Close without applying. java: button id `-2`.
    Cancel,
    /// Choose an effect. java: button id `tier << 8 | potion_id`.
    Power {
        /// The pyramid tier row this button sits in, `0..=3`. `3` is the secondary column.
        tier: u8,
        /// The potion id this button grants.
        potion: u8,
    },
}

impl BeaconArt {
    /// Resolve every button the beacon would build from its current fields. `ContainerScreen`
    /// snapshots the returned power buttons at vanilla's `buttonsNotDrawn` boundary; keeping the
    /// pure builder here also gives direct art renderers the same one-frame layout contract.
    // java: GuiBeacon.java:56 (updateScreen)
    #[must_use]
    pub fn buttons(&self, has_payment: bool) -> Vec<BeaconButton> {
        let mut out = Vec::with_capacity(8);
        out.push(BeaconButton {
            x: 164,
            y: 107,
            kind: BeaconButtonKind::Confirm,
            enabled: has_payment && self.primary > 0,
            selected: false,
        });
        out.push(BeaconButton {
            x: 190,
            y: 107,
            kind: BeaconButtonKind::Cancel,
            enabled: true,
            selected: false,
        });
        if self.levels < 0 {
            return out;
        }

        for (tier, row) in BEACON_EFFECTS.iter().enumerate().take(3) {
            let count = row.len() as i32;
            let span = count * 22 + (count - 1) * 2;
            for (column, &potion) in row.iter().enumerate() {
                out.push(BeaconButton {
                    x: 76 + column as i32 * 24 - span / 2,
                    y: 22 + tier as i32 * 25,
                    kind: BeaconButtonKind::Power {
                        tier: tier as u8,
                        potion,
                    },
                    enabled: (tier as i32) < self.levels,
                    selected: i32::from(potion) == self.primary,
                });
            }
        }

        // The secondary column: `effectsList[3]` plus one extra slot that doubles the primary.
        let secondary = BEACON_EFFECTS[3];
        let count = secondary.len() as i32 + 1;
        let span = count * 22 + (count - 1) * 2;
        for (column, &potion) in secondary.iter().enumerate() {
            out.push(BeaconButton {
                x: 167 + column as i32 * 24 - span / 2,
                y: 47,
                kind: BeaconButtonKind::Power { tier: 3, potion },
                enabled: 3 < self.levels,
                selected: i32::from(potion) == self.secondary,
            });
        }
        if self.primary > 0 {
            let potion = u8::try_from(self.primary).unwrap_or(0);
            out.push(BeaconButton {
                x: 167 + (count - 1) * 24 - span / 2,
                y: 47,
                kind: BeaconButtonKind::Power { tier: 3, potion },
                enabled: 3 < self.levels,
                selected: self.primary == self.secondary,
            });
        }
        out
    }
}

// ---------------------------------------------------------------- the art enum

/// Which vanilla `GuiContainer` subclass a [`ContainerScreen`] is: the panel texture, its size,
/// and whatever dynamic art the kind draws over it.
#[derive(Debug, Clone, PartialEq)]
pub enum ContainerArt {
    /// java: `GuiInventory` over `inventory.png`.
    Player,
    /// java: `GuiChest` over `generic_54.png`; `rows` is `lowerInv.getSizeInventory() / 9`.
    Chest {
        /// How many 9-wide rows the container half has, `1..=6`.
        rows: u8,
    },
    /// java: `GuiCrafting` over `crafting_table.png`.
    ///
    /// Carries no dynamic state: `ContainerWorkbench` sends no window property, and its result
    /// slot is filled by the client-side `CraftingManager` adapter in `model.rs`, not by the
    /// server.
    Crafting,
    /// java: `GuiHopper` over `hopper.png`.
    Hopper,
    /// java: `GuiDispenser` over `dispenser.png` (droppers reuse it).
    Dispenser,
    /// java: `GuiFurnace` over `furnace.png`.
    Furnace(FurnaceArt),
    /// java: `GuiBrewingStand` over `brewing_stand.png`.
    BrewingStand(BrewingStandArt),
    /// java: `GuiEnchantment` over `enchanting_table.png`.
    Enchanting(EnchantingArt),
    /// java: `GuiRepair` over `anvil.png`.
    Anvil(AnvilArt),
    /// java: `GuiBeacon` over `beacon.png`. The only kind that is not 176 wide.
    Beacon(BeaconArt),
    /// java: `GuiMerchant` over `villager.png`.
    Merchant(MerchantArt),
    /// java: `GuiScreenHorseInventory` over `horse.png`. `chested` selects the extra 3x5
    /// animal-inventory grid used by donkeys, mules, and llamas.
    Horse {
        /// Whether the entity supplied the fifteen additional chest slots.
        chested: bool,
    },
}

impl ContainerArt {
    /// The panel texture this kind blits.
    #[must_use]
    pub fn texture(&self) -> UiTexture {
        container_texture(match self {
            ContainerArt::Player => "inventory.png",
            ContainerArt::Chest { .. } => "generic_54.png",
            ContainerArt::Crafting => "crafting_table.png",
            ContainerArt::Hopper => "hopper.png",
            ContainerArt::Dispenser => "dispenser.png",
            ContainerArt::Furnace(_) => "furnace.png",
            ContainerArt::BrewingStand(_) => "brewing_stand.png",
            ContainerArt::Enchanting(_) => "enchanting_table.png",
            ContainerArt::Anvil(_) => "anvil.png",
            ContainerArt::Beacon(_) => "beacon.png",
            ContainerArt::Merchant(_) => "villager.png",
            ContainerArt::Horse { .. } => "horse.png",
        })
    }

    /// java: `GuiContainer.xSize`, `176` for everything except the beacon.
    #[must_use]
    pub const fn x_size(&self) -> i32 {
        match self {
            ContainerArt::Beacon(_) => 230,
            _ => 176,
        }
    }

    /// java: `GuiContainer.ySize`. `GuiChest` computes it as `114 + rows * 18`
    /// (`222 - 108 + rows * 18`); the hopper pins 133 and the beacon 219.
    #[must_use]
    pub const fn y_size(&self) -> i32 {
        match self {
            ContainerArt::Chest { rows } => 114 + *rows as i32 * 18,
            ContainerArt::Hopper => 133,
            ContainerArt::Beacon(_) => 219,
            _ => 166,
        }
    }

    /// A stable screen name for [`Screen::name`] and the F3 overlay.
    #[must_use]
    pub const fn screen_name(&self) -> &'static str {
        match self {
            ContainerArt::Player => "container.player",
            ContainerArt::Chest { .. } => "container.chest",
            ContainerArt::Crafting => "container.crafting",
            ContainerArt::Hopper => "container.hopper",
            ContainerArt::Dispenser => "container.dispenser",
            ContainerArt::Furnace(_) => "container.furnace",
            ContainerArt::BrewingStand(_) => "container.brewing_stand",
            ContainerArt::Enchanting(_) => "container.enchanting_table",
            ContainerArt::Anvil(_) => "container.anvil",
            ContainerArt::Beacon(_) => "container.beacon",
            ContainerArt::Merchant(_) => "container.merchant",
            ContainerArt::Horse { .. } => "container.horse",
        }
    }

    /// java: `GuiScreen.allowUserInput` — whether movement keys still reach the player. Only the
    /// inventory-shaped screens set it; `GuiChest`/`GuiHopper` explicitly clear it.
    #[must_use]
    pub const fn allows_user_input(&self) -> bool {
        matches!(self, ContainerArt::Player)
    }

    /// Apply an `S31PacketWindowProperty`. Unknown indices are ignored rather than panicking —
    /// this is server-driven input.
    // java: Container.updateProgressBar and each subclass's override.
    pub fn apply_window_property(&mut self, var_index: i16, var_value: i16) {
        let value = i32::from(var_value);
        match self {
            ContainerArt::Furnace(f) => match var_index {
                0 => f.burn_time = value,
                1 => f.current_item_burn_time = value,
                2 => f.cook_time = value,
                3 => f.total_cook_time = value,
                _ => {}
            },
            ContainerArt::BrewingStand(b) => {
                if var_index == 0 {
                    b.brew_time = value;
                }
            }
            ContainerArt::Enchanting(e) => match var_index {
                0..=2 => e.enchant_levels[var_index as usize] = value,
                3 => e.xp_seed = value,
                4..=6 => e.enchantment_ids[var_index as usize - 4] = value,
                _ => {}
            },
            ContainerArt::Anvil(a) => {
                if var_index == 0 {
                    a.maximum_cost = value;
                }
            }
            ContainerArt::Beacon(b) => match var_index {
                0 => b.levels = value,
                1 => b.primary = beacon_effect_or_none(value),
                2 => b.secondary = beacon_effect_or_none(value),
                _ => {}
            },
            ContainerArt::Player
            | ContainerArt::Crafting
            | ContainerArt::Chest { .. }
            | ContainerArt::Hopper
            | ContainerArt::Dispenser
            | ContainerArt::Merchant(_)
            | ContainerArt::Horse { .. } => {}
        }
    }

    /// java: each subclass's `drawGuiContainerBackgroundLayer`. `left`/`right` are absolute
    /// screen pixels (`guiLeft`/`guiTop`), matching vanilla's own untranslated background pass.
    pub fn draw_background(&self, d: &mut DrawList, left: i32, top: i32, container: &Container) {
        self.draw_background_inner(d, left, top, container, true, None, true);
        if matches!(self, ContainerArt::Beacon(_)) {
            // The standalone art API has no screen-owned atlas override. Keep its historical
            // default texture contract while the live screen passes its configured atlas below.
            draw_beacon_payment_items_at(d, &default_atlas_texture(), left, top);
        }
    }

    /// Draw the panel and every dynamic overlay except enchanting offer plates.
    ///
    /// `GuiEnchantment` paints its second-projection book between the panel texture and the
    /// offer plates.  The ordinary background helper predates that 3-D pass and intentionally
    /// keeps its all-in-one contract for direct callers, so the live screen uses this narrow split
    /// seam to preserve Java's command order.
    fn draw_background_before_enchanting_book(
        &self,
        d: &mut DrawList,
        left: i32,
        top: i32,
        container: &Container,
        hovered_enchant_plate: Option<usize>,
    ) {
        self.draw_background_inner(d, left, top, container, false, hovered_enchant_plate, false);
    }

    /// Draw the three enchanting offer plates after the book's second projection.
    fn draw_enchanting_offers(
        &self,
        d: &mut DrawList,
        left: i32,
        top: i32,
        container: &Container,
        hovered_enchant_plate: Option<usize>,
        mut runes: Option<&mut [TextMesh; 3]>,
    ) {
        // A 1.7.10 table's plates have no lapis (`screen_v5.rs`).
        let hovered = hovered_enchant_plate;
        if self.draw_enchanting_offers_v5(d, (left, top), container, hovered, runes.as_deref_mut())
        {
            return;
        }
        let ContainerArt::Enchanting(e) = self else {
            return;
        };
        let tex = self.texture();
        let lapis = container.slot_stack(1).map_or(0, |s| s.count);
        for row in 0..3usize {
            let x = left + 60;
            let y = top + 14 + 19 * row as i32;
            let level = e.enchant_levels[row];
            if level == 0 {
                d.blit(&tex, x, y, 0, 185, 108, 19);
                continue;
            }
            if e.is_affordable(row, lapis) {
                let plate_v = if hovered_enchant_plate == Some(row) {
                    204
                } else {
                    166
                };
                d.blit(&tex, x, y, 0, plate_v, 108, 19);
                d.blit(&tex, x + 1, y + 1, 16 * row as i32, 223, 16, 16);
            } else {
                d.blit(&tex, x, y, 0, 185, 108, 19);
                d.blit(&tex, x + 1, y + 1, 16 * row as i32, 239, 16, 16);
            }
            if let Some(rows) = runes.as_deref_mut() {
                let mesh = std::mem::take(&mut rows[row]);
                for glyph in mesh.glyphs {
                    d.glyph(glyph);
                }
                for bar in mesh.bars {
                    d.bar(bar);
                }
            }
        }
    }

    // This helper mirrors Java's art-layer seam; keeping its arguments explicit preserves the
    // existing direct-render call sites and their command ordering.
    #[allow(clippy::too_many_arguments)]
    fn draw_background_inner(
        &self,
        d: &mut DrawList,
        left: i32,
        top: i32,
        container: &Container,
        draw_beacon_buttons: bool,
        hovered_enchant_plate: Option<usize>,
        draw_enchanting_offers: bool,
    ) {
        let tex = self.texture();
        let (w, h) = (self.x_size(), self.y_size());
        match self {
            ContainerArt::Chest { rows } => {
                // The chest sheet holds the container half at v=0 and the player half at v=126,
                // so the two are blitted separately and the seam moves with `rows`.
                let container_height = i32::from(*rows) * 18 + 17;
                d.blit(&tex, left, top, 0, 0, w, container_height);
                d.blit(&tex, left, top + container_height, 0, 126, w, 96);
            }
            _ => d.blit(&tex, left, top, 0, 0, w, h),
        }

        match self {
            ContainerArt::Furnace(f) => {
                if f.is_burning() {
                    let k = f.burn_left_scaled(13);
                    d.blit(&tex, left + 56, top + 36 + 12 - k, 176, 12 - k, 14, k + 1);
                }
                let progress = f.cook_progress_scaled(24);
                d.blit(&tex, left + 79, top + 34, 176, 14, progress + 1, 16);
            }
            ContainerArt::BrewingStand(b) => {
                let arrow = b.arrow_height();
                if arrow > 0 {
                    d.blit(&tex, left + 97, top + 16, 176, 0, 9, arrow);
                }
                let bubbles = b.bubble_height();
                if bubbles > 0 {
                    d.blit(
                        &tex,
                        left + 65,
                        top + 14 + 29 - bubbles,
                        185,
                        29 - bubbles,
                        12,
                        bubbles,
                    );
                }
            }
            ContainerArt::Enchanting(_) => {}
            ContainerArt::Anvil(_) => {
                // The rename box art has a "filled" and an "empty" row stacked below the panel.
                let v = h + if container.slot_stack(0).is_some() {
                    0
                } else {
                    16
                };
                d.blit(&tex, left + 59, top + 20, 0, v, 110, 16);
                let has_input =
                    container.slot_stack(0).is_some() || container.slot_stack(1).is_some();
                if has_input && container.slot_stack(2).is_none() {
                    // The red cross: "these two items cannot be combined".
                    d.blit(&tex, left + 99, top + 45, w, 0, 28, 21);
                }
            }
            ContainerArt::Merchant(merchant) => {
                draw_merchant_buttons(d, &tex, left, top, merchant);
                // java: GuiMerchant draws the disabled cross over each buy slot when the
                // selected offer cannot be completed; the offer item previews themselves are
                // drawn after the shared slot pass by ContainerScreen.
                if merchant
                    .selected_recipe()
                    .is_some_and(|recipe| recipe.disabled || recipe.uses >= recipe.max_uses)
                {
                    d.blit(&tex, left + 83, top + 21, 212, 0, 28, 21);
                    d.blit(&tex, left + 83, top + 51, 212, 0, 28, 21);
                }
            }
            ContainerArt::Horse { chested } => {
                // java: GuiScreenHorseInventory draws this 90x54 overlay only for a chested
                // mount. The armor-capability overlay is entity-state dependent and is supplied
                // by the horse slot background when available; the packet itself carries only
                // the inventory size, so do not invent it for donkeys/llamas here.
                if *chested {
                    d.blit(&tex, left + 79, top + 17, 0, h, 90, 54);
                }
            }
            ContainerArt::Beacon(b) if draw_beacon_buttons => {
                for button in b.buttons(container.slot_stack(0).is_some()) {
                    draw_beacon_button(d, &tex, left, top, button, false);
                }
            }
            ContainerArt::Beacon(_) => {}
            ContainerArt::Player
            | ContainerArt::Crafting
            | ContainerArt::Chest { .. }
            | ContainerArt::Hopper
            | ContainerArt::Dispenser => {}
        }
        if draw_enchanting_offers && matches!(self, ContainerArt::Enchanting(_)) {
            self.draw_enchanting_offers(d, left, top, container, hovered_enchant_plate, None);
        }
    }

    /// java: each subclass's `drawGuiContainerForegroundLayer`. Vanilla runs this inside the
    /// `translate(guiLeft, guiTop)` block, so its literal coordinates are window-relative; they
    /// are offset here instead.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_foreground(
        &self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        left: i32,
        top: i32,
        container: &Container,
        title: &str,
        player_title: &str,
        language: Option<&Language>,
    ) {
        let x_size = self.x_size();
        let y_size = self.y_size();
        let player_y = top + y_size - 96 + 2;
        let centered = |font: &FontRenderer, s: &str| left + x_size / 2 - font.string_width(s) / 2;

        match self {
            ContainerArt::Player => {
                font.draw_string(
                    d,
                    title,
                    (left + 86) as f32,
                    (top + 16) as f32,
                    TITLE_COLOR,
                    false,
                );
            }
            // java: `GuiCrafting.drawGuiContainerForegroundLayer` draws the *localised*
            // `container.crafting` at `(28, 6)` and ignores the window title entirely -- see
            // `ContainerScreen::crafting_table`'s doc for why the packet's title never reaches
            // this screen.
            ContainerArt::Crafting => {
                font.draw_string(
                    d,
                    title,
                    (left + 28) as f32,
                    (top + 6) as f32,
                    TITLE_COLOR,
                    false,
                );
                font.draw_string(
                    d,
                    player_title,
                    (left + 8) as f32,
                    player_y as f32,
                    TITLE_COLOR,
                    false,
                );
            }
            ContainerArt::Chest { .. } | ContainerArt::Hopper | ContainerArt::Horse { .. } => {
                font.draw_string(
                    d,
                    title,
                    (left + 8) as f32,
                    (top + 6) as f32,
                    TITLE_COLOR,
                    false,
                );
                font.draw_string(
                    d,
                    player_title,
                    (left + 8) as f32,
                    player_y as f32,
                    TITLE_COLOR,
                    false,
                );
            }
            ContainerArt::Dispenser | ContainerArt::Furnace(_) | ContainerArt::BrewingStand(_) => {
                let x = centered(font, title);
                font.draw_string(d, title, x as f32, (top + 6) as f32, TITLE_COLOR, false);
                font.draw_string(
                    d,
                    player_title,
                    (left + 8) as f32,
                    player_y as f32,
                    TITLE_COLOR,
                    false,
                );
            }
            ContainerArt::Enchanting(e) => {
                font.draw_string(
                    d,
                    title,
                    (left + 12) as f32,
                    (top + 5) as f32,
                    TITLE_COLOR,
                    false,
                );
                font.draw_string(
                    d,
                    player_title,
                    (left + 8) as f32,
                    player_y as f32,
                    TITLE_COLOR,
                    false,
                );
                let lapis = container.slot_stack(1).map_or(0, |s| s.count);
                let lapis = v5::offer_lapis_v5(container, lapis);
                for row in 0..3usize {
                    let level = e.enchant_levels[row];
                    if level == 0 {
                        continue;
                    }
                    let text = level.to_string();
                    // java: `drawStringWithShadow(s1, j1 + 86 - width, j + 16 + 19*l + 7, i2)`
                    // where `j1 = guiLeft + 80`. The colour is `8453920` when the offer can be
                    // taken and `4226832` when it cannot.
                    let x = left + 80 + 86 - font.string_width(&text);
                    let y = top + 16 + 19 * row as i32 + 7;
                    // These two were transcribed inline as `0x0081_00A0` (a *purple*)
                    // and `0x0040_7830`, neither of which is the decimal constant the comment
                    // above names.
                    let color = if e.is_affordable(row, lapis) {
                        ENCHANT_LEVEL_COLOR
                    } else {
                        ENCHANT_LEVEL_DISABLED_COLOR
                    };
                    font.draw_string_with_shadow(d, &text, x as f32, y as f32, color);
                }
            }
            // java: `GuiRepair` draws exactly one heading -- the hard-coded `container.repair`
            // -- and, alone among the container GUIs, **no** player-inventory title.
            ContainerArt::Anvil(a) => {
                let title = language.map_or(title, |language| language.tr("container.repair"));
                font.draw_string(
                    d,
                    title,
                    (left + 60) as f32,
                    (top + 6) as f32,
                    TITLE_COLOR,
                    false,
                );
                if let Some(label) = a.cost_label(container) {
                    let text = language.map_or_else(
                        || label.text.clone(),
                        |language| {
                            if a.is_too_expensive() {
                                language.tr("container.repair.expensive").to_owned()
                            } else {
                                language
                                    .format("container.repair.cost", &[&a.maximum_cost.to_string()])
                            }
                        },
                    );
                    let x = left + x_size - 8 - font.string_width(&text);
                    let y = top + 67;
                    if font.unicode_flag() {
                        // java: GuiRepair's `getUnicodeFlag()` branch (`drawRect` twice),
                        // relative to the translated foreground origin. `x` and `y` are already
                        // absolute screen coordinates here, so only the right edge needs the
                        // panel's absolute left offset.
                        d.rect(x - 3, y - 2, left + x_size - 7, y + 10, 0xFF00_0000);
                        d.rect(x - 2, y - 1, left + x_size - 8, y + 9, 0xFF3B_3B3B);
                    } else {
                        // java: the three offset passes vanilla uses in place of a real shadow.
                        let shadow = 0xFF00_0000_u32 as i32 | ((label.color & 16_579_836) >> 2);
                        font.draw_string(d, &text, x as f32, (y + 1) as f32, shadow, false);
                        font.draw_string(d, &text, (x + 1) as f32, y as f32, shadow, false);
                        font.draw_string(d, &text, (x + 1) as f32, (y + 1) as f32, shadow, false);
                    }
                    font.draw_string(d, &text, x as f32, y as f32, label.color, false);
                }
            }
            ContainerArt::Beacon(_) => {
                let primary = language.map_or("Primary Power", |language| {
                    language.tr("tile.beacon.primary")
                });
                let secondary = language.map_or("Secondary Power", |language| {
                    language.tr("tile.beacon.secondary")
                });
                font.draw_string_with_shadow(
                    d,
                    primary,
                    (left + 62 - font.string_width(primary) / 2) as f32,
                    (top + 10) as f32,
                    BEACON_HEADING_COLOR,
                );
                font.draw_string_with_shadow(
                    d,
                    secondary,
                    (left + 169 - font.string_width(secondary) / 2) as f32,
                    (top + 10) as f32,
                    BEACON_HEADING_COLOR,
                );
            }
            ContainerArt::Merchant(_) => {
                let x = centered(font, title);
                font.draw_string(d, title, x as f32, (top + 6) as f32, TITLE_COLOR, false);
                font.draw_string(
                    d,
                    player_title,
                    (left + 8) as f32,
                    player_y as f32,
                    TITLE_COLOR,
                    false,
                );
            }
        }
    }
}

/// java: `GuiBeacon.Button.drawButton` — one 22x22 plate from `beacon.png`'s `v = 219` strip,
/// with the effect icon (or the tick/cross glyph) blitted 2 pixels inside it.
fn draw_beacon_button(
    d: &mut DrawList,
    beacon_tex: &UiTexture,
    left: i32,
    top: i32,
    button: BeaconButton,
    hovered: bool,
) {
    let mut u = 0;
    if !button.enabled {
        u += 44;
    } else if button.selected {
        u += 22;
    } else if hovered {
        u += 66;
    }
    let (x, y) = (left + button.x, top + button.y);
    d.blit(beacon_tex, x, y, u, 219, 22, 22);
    match button.kind {
        BeaconButtonKind::Confirm => d.blit(beacon_tex, x + 2, y + 2, 90, 220, 18, 18),
        BeaconButtonKind::Cancel => d.blit(beacon_tex, x + 2, y + 2, 112, 220, 18, 18),
        BeaconButtonKind::Power { potion, .. } => {
            if let Some(icon) = beacon_status_icon_index(potion) {
                // java: the potion status icons live on `inventory.png`, not `beacon.png`.
                let sheet = container_texture("inventory.png");
                d.blit(
                    &sheet,
                    x + 2,
                    y + 2,
                    icon % 8 * 18,
                    198 + icon / 8 * 18,
                    18,
                    18,
                );
            }
        }
    }
}

/// java: `GuiBeacon.drawGuiContainerBackgroundLayer`'s four decorative payment stacks. They are
/// rendered independently of the payment slot contents, with no count or durability overlay.
fn draw_beacon_payment_items_at(d: &mut DrawList, atlas: &UiTexture, left: i32, top: i32) {
    const PAYMENT_ITEMS: [(i16, i32); 4] = [
        (388, 0),  // emerald
        (264, 22), // diamond
        (266, 44), // gold ingot
        (265, 66), // iron ingot
    ];
    for (item_id, x_offset) in PAYMENT_ITEMS {
        let stack = ItemStack {
            item_id,
            count: 1,
            damage: 0,
            nbt: None,
        };
        render::draw_item_icon_tinted_at_z_level(
            d,
            atlas,
            left + 42 + x_offset,
            top + 109,
            &stack,
            0xFFFF_FFFF,
            render::GUI_SLOT_ITEM_RENDER_Z_LEVEL,
        );
    }
}

fn draw_merchant_buttons(
    d: &mut DrawList,
    texture: &UiTexture,
    left: i32,
    top: i32,
    merchant: &MerchantArt,
) {
    let buttons = [
        (17, merchant.selected > 0, false),
        (
            147,
            merchant.selected.saturating_add(1) < merchant.recipes.len(),
            true,
        ),
    ];
    for (x, enabled, next) in buttons {
        let u = if enabled { 176 } else { 200 };
        let v = if next { 0 } else { 19 };
        d.blit(texture, left + x, top + 23, u, v, 12, 19);
    }
}

// ---------------------------------------------------------------- the screen

#[derive(Debug, Clone, PartialEq, Eq)]
struct DragPreview {
    mode: i32,
    slots: BTreeSet<u16>,
}

#[derive(Debug, Clone, PartialEq, Default)]
struct TouchscreenGesture {
    clicked_slot: Option<u16>,
    dragged_stack: Option<ItemStack>,
    right_click: bool,
    hold_slot: Option<u16>,
    hold_started_ms: u64,
}

#[derive(Debug, Clone, PartialEq)]
struct ReturningStack {
    stack: ItemStack,
    start: (i32, i32),
    destination_slot: u16,
    started_ms: u64,
}

/// The two fixed `RenderHelper.enableStandardItemLighting` directions used by the 3-D entity and
/// tile passes.  The UI has no fragment lighting shader, so the same diffuse factor is folded into
/// each book vertex before it enters [`DrawList`].
const BOOK_LIGHT0: Vec3 = Vec3::new(0.2, 1.0, -0.7);
const BOOK_LIGHT1: Vec3 = Vec3::new(-0.2, 1.0, 0.7);

#[must_use]
fn book_lighting(normal: [f32; 3]) -> f32 {
    let n = Vec3::from_array(normal).normalize_or_zero();
    0.4 + 0.6 * n.dot(BOOK_LIGHT0.normalize()).max(0.0)
        + 0.6 * n.dot(BOOK_LIGHT1.normalize()).max(0.0)
}

#[must_use]
fn book_color(vertex: &EntityVertex) -> [f32; 4] {
    let light = book_lighting(vertex.normal);
    [
        (light * vertex.color[0]).clamp(0.0, 1.0),
        (light * vertex.color[1]).clamp(0.0, 1.0),
        (light * vertex.color[2]).clamp(0.0, 1.0),
        vertex.color[3].clamp(0.0, 1.0),
    ]
}

/// Draw the projected `ModelBook` triangles under vanilla's GL state for the GUI book.
///
/// java: `GuiEnchantment.drawGuiContainerBackgroundLayer:113-169` runs inside
/// `GuiContainer.drawScreen` *before* that method's `disableDepth()`, so the book is drawn with
/// the depth test (`GL_LEQUAL`) and depth writes on, and with `GL_CULL_FACE`/`GL_BACK` still set
/// by `EntityRenderer.renderWorldPass` (nothing on the GUI path turns it off). Both are needed:
/// the covers, spine and flipping pages are zero-depth boxes whose two Z quads are coplanar with
/// opposite windings, and the page blocks sit between the covers, so a closed book is only cover.
/// The former far-to-near painter sort by mean depth, with no cull, let page triangles whose
/// centroid was nearer draw over the cover — the reported page poking out of the closed book.
///
/// Each vertex's depth is `ndc_z + 1`, i.e. twice the GL window depth. Before this draw,
/// vanilla's world-background gradient and the panel blit wrote window depth 0.5 under the whole
/// screen (`z = -2000` in `ortho(0, w, h, 0, 1000, 3000)`), so book fragments deeper than 0.5 fail
/// `GL_LEQUAL`. Doubling puts that 0.5 on the UI's far plane, where the GPU's depth clip drops
/// exactly those fragments; the mapping is monotonic, so the book's own ordering is unchanged.
/// Nothing in the UI pass writes depth under the book beforehand (panel and backdrop are overlay
/// batches), so the book only tests against itself and that plane.
fn draw_projected_enchanting_book(
    d: &mut DrawList,
    texture: &UiTexture,
    vertices: &[EntityVertex],
    scaled_width: i32,
    scaled_height: i32,
) {
    use crate::widget::draw::{UiBlend, UiCull, UiDepth};

    d.with_render_state(UiDepth::Item, UiCull::Back, |d| {
        'triangles: for chunk in vertices.chunks_exact(3) {
            let mut pos = [[0.0; 2]; 3];
            let mut depth = [0.0; 3];
            for (i, vertex) in chunk.iter().enumerate() {
                let Some([x, y, ndc_z]) = project_enchant_book_vertex(
                    Vec3::from_array(vertex.pos),
                    scaled_width,
                    scaled_height,
                ) else {
                    continue 'triangles;
                };
                pos[i] = [x, y];
                depth[i] = ndc_z + 1.0;
            }
            d.triangle_with_blend_depth(
                texture,
                UiBlend::Alpha,
                depth,
                pos,
                [chunk[0].uv, chunk[1].uv, chunk[2].uv],
                [
                    book_color(&chunk[0]),
                    book_color(&chunk[1]),
                    book_color(&chunk[2]),
                ],
            );
        }
    });
}

#[derive(Debug)]
enum OrderedContainerAction {
    ClickWindow(c2s::ClickWindow),
    EnchantItem(c2s::EnchantItem),
    /// `GuiRepair.renameItem`'s `C17PacketCustomPayload("MC|ItemName", ...)`.
    /// It shares the click stream because vanilla sends it from inside `Container.slotClick`,
    /// before that click's C0E, as well as from `keyTyped`.
    AnvilName(String),
}

/// The `MC|ItemName` payload: `new PacketBuffer(...).writeString(name)`.
/// java: `GuiRepair.renameItem`.
fn anvil_name_message(name: &str) -> c2s::PluginMessage {
    let mut payload = mc_protocol::PacketBuffer::new();
    payload.write_string(name);
    c2s::PluginMessage {
        channel: "MC|ItemName".to_owned(),
        data: payload.as_slice().to_vec(),
    }
}

/// One built item tooltip and the complete input it was built from (C2-13).
#[derive(Debug)]
struct TooltipCache {
    stack: ItemStack,
    advanced: bool,
    language_generation: u64,
    map_data: Option<MapData>,
    lines: Vec<String>,
}

/// A live container GUI: a [`Container`] plus everything needed to draw it and click it.
///
/// Construct one with the matching helper ([`ContainerScreen::player`],
/// [`ContainerScreen::chest`], ...) or from the server's `S2DPacketOpenWindow` with
/// [`ContainerScreen::for_open_window`], push it onto a
/// [`ScreenStack`](crate::screen_stack::ScreenStack), and drain
/// [`take_ordered_outgoing`](Self::take_ordered_outgoing) once per frame to get the packets to send.
#[derive(Debug)]
pub struct ContainerScreen {
    /// The server-synced window. Apply `SetSlot`/`WindowItems`/`ConfirmTransaction` to this
    /// directly (see [`super::net`]).
    pub container: Container,
    /// Which vanilla screen this is, and its dynamic art state.
    pub art: ContainerArt,
    /// The container's own title, already resolved from `OpenWindow`'s chat component.
    pub title: String,
    /// The player-inventory heading, vanilla's `container.inventory` ("Inventory").
    pub player_title: String,
    /// The [`UiTexture`] key the host uploaded `mc_assets::BakedAssets::atlas` under.
    pub atlas: UiTexture,
    /// Vanilla's F3+H "advanced tooltips" setting.
    pub advanced_tooltips: bool,
    /// Active translations used by metadata-aware item tooltip names.
    language: Option<Language>,
    /// Bumped by [`Self::set_language`], so the tooltip cache can tell translation tables apart
    /// without comparing them.
    language_generation: u64,
    /// C2-13: the last item tooltip built, with every input it was built from. `RefCell`
    /// because [`Self::hovered_tooltip`] takes `&self`.
    tooltip_cache: std::cell::RefCell<Option<TooltipCache>>,
    /// Whether the player is in creative mode, which unlocks click modes 3 and drag mode 2.
    pub creative: bool,
    /// Current inventory/drop/hotbar/pick-block bindings, refreshed by the host before input.
    bindings: ContainerBindings,
    /// Milliseconds since an arbitrary epoch, for the 250 ms double-click window. Set it from
    /// the host's real clock with [`set_clock_ms`](Self::set_clock_ms); left alone it advances
    /// one tick (50 ms) per [`Screen::tick`], which is accurate enough for double-click
    /// detection but is not a wall clock.
    pub clock_ms: u64,
    mouse: MouseState,
    /// NOT VANILLA 1.8.9: the mouse button whose press the hotbar addition in
    /// [`Self::mouse_clicked_with_mods`] consumed, so its release is swallowed too.
    hotbar_mouse_button: Option<i32>,
    /// Live `GameSettings.touchscreen` snapshot. This selects Java's parallel tap/drag gesture
    /// owner rather than the desktop drag-splitting state machine.
    touchscreen: bool,
    touch: TouchscreenGesture,
    returning_stack: Option<ReturningStack>,
    /// `InventoryEffectRenderer`'s live effect collection. It only affects
    /// [`ContainerArt::Player`]; server-opened containers retain ordinary centring.
    active_effects: Vec<ActiveEffect>,
    /// The `centered_inventory` mod: keep the player inventory centred while effects
    /// are active instead of vanilla's shift to the right. `false` is vanilla.
    effects_centered: bool,
    /// The `centered_inventory` mod with "show effects" off: skip the effect panels.
    /// `false` is vanilla.
    effects_hidden: bool,
    /// Last scaled width supplied to [`Screen::init`], retained because `GuiInventory` calls
    /// `updateActivePotionEffects()` every tick and can therefore move `guiLeft` without a
    /// resize.
    screen_width: i32,
    gui_left: i32,
    gui_top: i32,
    hovered_slot: Option<u16>,
    /// The live `map_<id>` record resolved by the client immediately before drawing. Only the
    /// id and scale affect `ItemMap.addInformation`, so pixel updates do not replace this copy.
    tooltip_map_data: Option<MapData>,
    /// The enchanting offer under the cursor. Kept separately from `hovered_slot` because the
    /// three 108x17 offer plates are not inventory slots at all.
    hovered_enchant_offer: Option<usize>,
    /// The selected merchant recipe preview under the cursor. These three rendered stacks are
    /// not backed by container slots, but vanilla still routes them through the ordinary item
    /// tooltip path.
    hovered_merchant_preview: Option<ItemStack>,
    /// `GuiEnchantment`'s CPU-side opening/page-flip spring. The render owner consumes its
    /// interpolated [`EnchantingBookPose`]; this screen keeps the tick state because the Java
    /// update runs from `GuiEnchantment.updateScreen` rather than the tile renderer.
    enchanting_book: EnchantingBookAnimation,
    /// Cached `ModelBook` owner for the enchanting screen's second projection.
    ///
    /// java: `GuiEnchantment.MODEL_BOOK` is a static model.  The UI keeps the same mutable model
    /// between frames so `ModelBook.setRotationAngles` remains a pose update rather than a
    /// reconstruction at every draw.
    enchanting_book_models: TileModels,
    /// One-frame Standard Galactic glyph meshes, prepared by the live host's dedicated font
    /// renderer and consumed between each enchanting plate/icon and the ordinary slot pass.
    enchanting_runes: [TextMesh; 3],
    /// GUI-side mirror of `MouseState`'s desktop drag set. `MouseState` owns packet production;
    /// this copy owns only the pre-release rendering that vanilla stores on `GuiContainer`.
    drag_preview: Option<DragPreview>,
    /// `GuiBeacon.buttonsNotDrawn`'s result: power buttons are captured once after field 0 first
    /// becomes non-negative, then survive later window-property changes until `initGui` re-arms
    /// the snapshot. Confirm/cancel remain live and are not stored here.
    beacon_power_buttons: Option<Vec<BeaconButton>>,
    /// The beacon button under the cursor during the most recent draw. Vanilla stores this on
    /// each `GuiButton` while drawing and uses the first hovered button for its foreground
    /// tooltip.
    hovered_beacon_button: Option<BeaconButton>,
    /// C0E/C11 actions in the exact order their GUI callbacks invoked them. Java enqueues both
    /// packet types immediately, so keeping separate category queues would reorder an
    /// enchanting-screen slot click followed by an enchantment action.
    outgoing_actions: Vec<OrderedContainerAction>,
    /// Plugin payloads produced by container controls, currently `MC|Beacon`. The host drains
    /// these alongside ordinary click packets because a screen has no network session.
    outgoing_plugins: Vec<c2s::PluginMessage>,
    /// The last close this screen asked for was `displayGuiScreen(null)`
    /// called by the GUI itself, not `EntityPlayerSP.closeScreen()`: vanilla sends **no**
    /// `C0DPacketCloseWindow` for it. See [`Self::take_closes_without_close_window`].
    closes_without_close_window: bool,
    /// `GuiRepair.nameField`. `None` on every non-anvil screen.
    anvil_name: Option<TextField>,
    /// The slot-0 display name installed by `GuiRepair.sendSlotContents`. Needed by
    /// `renameItem`'s "unchanged ordinary name becomes the empty string" wire rule.
    anvil_input_name: Option<String>,
    /// Whether slot 0 already carried `display.Name` when it was synchronized.
    anvil_input_has_custom_name: bool,
    /// The slot-0 stack the name field was last synchronized from: `Container.inventoryItemStacks`'
    /// entry for slot 0, which `detectAndSendChanges` compares with `areItemStacksEqual` before
    /// calling `GuiRepair.sendSlotContents(0, ...)`. Starts empty, which is what `initGui`'s
    /// `onCraftGuiOpened` reports for a fresh anvil.
    anvil_synced_input: Option<ItemStack>,
    // Accepted name-field keys and non-empty slot-0 synchronizations now queue
    // their `MC|ItemName` in `outgoing_actions` (`OrderedContainerAction::AnvilName`).
    clipboard: MemoryClipboard,
    /// java: `GuiInventory`'s implicit `mc.thePlayer` — the CPU-projected 3-D player preview
    /// drawn on the Inventory panel only ([`ContainerArt::Player`]). See
    /// [`super::player_preview`].
    player_preview: PlayerPreviewRenderer,
    /// Which skin model (arm width) the preview poses. Set from outside with
    /// [`Self::set_player_preview_skin`]; harmless (and never drawn) for every non-`Player` art.
    player_skin: SkinModel,
    /// Ready generation-specific skin texture. `None` preserves the UUID-model Steve/Alex
    /// fallback while profile pixels are absent or still downloading.
    player_skin_texture: Option<UiTexture>,
    /// Live armour/held-item/cape/pose snapshot for the inventory preview.
    player_preview_snapshot: PlayerPreviewSnapshot,
    /// java: `ageInTicks`'s driver for the preview — ticks since this screen was constructed,
    /// combined with [`ScreenCtx::partial_ticks`] at draw time. Advanced by [`Screen::tick`].
    preview_age_ticks: u32,
    /// java: `GuiInventory`'s `oldMouseX`/`oldMouseY` — the *previous* frame's mouse, which is
    /// deliberately what the preview's look-at reads, one frame stale, matching vanilla exactly.
    preview_prev_mouse: (i32, i32),
    /// The tracked mount id retained from `OpenWindow.entity_id`.
    horse_entity_id: Option<i32>,
    /// Live horse state accepted only when its entity id matches `horse_entity_id`.
    horse_preview_snapshot: Option<HorsePreviewSnapshot>,
    /// The `scrollable_tooltips` mod's state. Off by default, when the tooltip
    /// draw is exactly vanilla's; the client turns it on with `set_options`.
    pub tooltip_scroll: super::tooltip_scroll::TooltipScroll,
    /// The left edge a 1.7.10 inventory keeps until its next layout (`screen_v5.rs`).
    init_shift_v5: super::v5::InitShiftV5,
}

impl ContainerScreen {
    /// Wrap an already-built [`Container`].
    #[must_use]
    pub fn new(container: Container, art: ContainerArt, title: impl Into<String>) -> Self {
        let anvil_name = matches!(art, ContainerArt::Anvil(_)).then(|| {
            let mut field = TextField::new(0, 62, 24, 103, 12);
            field.set_max_length(30);
            field.background = false;
            field.enabled_color = -1;
            field.disabled_color = -1;
            field.enabled = false;
            field
        });
        Self {
            container,
            art,
            title: title.into(),
            player_title: "Inventory".to_string(),
            atlas: default_atlas_texture(),
            advanced_tooltips: false,
            language: None,
            language_generation: 0,
            tooltip_cache: std::cell::RefCell::new(None),
            creative: false,
            bindings: ContainerBindings::default(),
            clock_ms: 0,
            mouse: MouseState::new(),
            hotbar_mouse_button: None,
            touchscreen: false,
            touch: TouchscreenGesture::default(),
            returning_stack: None,
            active_effects: Vec::new(),
            effects_centered: false,
            effects_hidden: false,
            screen_width: 0,
            gui_left: 0,
            gui_top: 0,
            hovered_slot: None,
            tooltip_map_data: None,
            hovered_enchant_offer: None,
            hovered_merchant_preview: None,
            enchanting_book: EnchantingBookAnimation::default(),
            enchanting_book_models: TileModels::new(false),
            enchanting_runes: std::array::from_fn(|_| TextMesh::new()),
            drag_preview: None,
            beacon_power_buttons: None,
            hovered_beacon_button: None,
            outgoing_actions: Vec::new(),
            outgoing_plugins: Vec::new(),
            closes_without_close_window: false,
            anvil_name,
            anvil_input_name: None,
            anvil_input_has_custom_name: false,
            anvil_synced_input: None,
            clipboard: MemoryClipboard::default(),
            player_preview: PlayerPreviewRenderer::new(),
            player_skin: SkinModel::Classic,
            player_skin_texture: None,
            player_preview_snapshot: PlayerPreviewSnapshot::default(),
            preview_age_ticks: 0,
            preview_prev_mouse: (0, 0),
            horse_entity_id: None,
            horse_preview_snapshot: None,
            tooltip_scroll: super::tooltip_scroll::TooltipScroll::default(),
            init_shift_v5: super::v5::InitShiftV5::default(),
        }
    }

    /// Install the host's current rebindable container actions.
    pub fn set_bindings(&mut self, bindings: ContainerBindings) {
        self.bindings = bindings;
    }

    /// Refresh Java's live `GameSettings.touchscreen` flag at the screen boundary.
    pub fn set_touchscreen(&mut self, touchscreen: bool) {
        self.touchscreen = touchscreen;
        if !touchscreen {
            self.touch = TouchscreenGesture::default();
        }
    }

    /// Sets which skin model (arm width) the Inventory-tab player preview poses. Only meaningful
    /// while [`Self::art`] is [`ContainerArt::Player`] — every other art never draws the preview
    /// at all. java: has no equivalent; vanilla's preview always poses `mc.thePlayer` directly
    /// and never needs to be told which skin that is. See [`super::player_preview`].
    pub fn set_player_preview_skin(&mut self, model: SkinModel) {
        self.player_skin = model;
    }

    /// Sets the ready dynamic skin texture used by the player-inventory preview.
    pub fn set_player_preview_texture(&mut self, texture: Option<UiTexture>) {
        self.player_skin_texture = texture;
    }

    /// Replace the complete live player snapshot consumed by the inventory preview. The
    /// snapshot is copied as one unit so armour, held item, cape and animation cannot tear.
    pub fn set_player_preview_snapshot(&mut self, snapshot: PlayerPreviewSnapshot) {
        self.player_preview_snapshot = snapshot;
    }

    /// The mount id retained from the horse open-window packet, if this is a horse screen.
    #[must_use]
    pub const fn horse_entity_id(&self) -> Option<i32> {
        self.horse_entity_id
    }

    /// Feed one live horse preview snapshot. Stale snapshots from a recycled network id are
    /// discarded at the screen boundary, before they can affect pose, saddle, reins, chest or
    /// armour rendering.
    pub fn set_horse_preview_snapshot(&mut self, snapshot: Option<HorsePreviewSnapshot>) {
        self.horse_preview_snapshot =
            snapshot.filter(|state| self.horse_entity_id == Some(state.entity_id));
        if let Some(state) = self.horse_preview_snapshot.as_ref() {
            self.container
                .set_horse_can_wear_armor(state.state.horse_type == 0);
        }
    }

    /// Replace the local player's active-effect list used by `GuiInventory`'s
    /// `InventoryEffectRenderer` base class.
    ///
    /// The host already resolves the translated name, amplifier suffix, icon and duration into
    /// [`ActiveEffect`] for the HUD. Supplying that same snapshot here avoids a second potion
    /// registry while preserving vanilla's inventory-only effect list. The origin is refreshed
    /// immediately as well as from [`Screen::tick`], so a packet processed between ticks cannot
    /// produce one stale frame.
    pub fn set_active_effects(&mut self, effects: Vec<ActiveEffect>) {
        self.active_effects = effects;
        self.update_active_potion_effects();
    }

    /// The effect snapshot currently rendered beside the player inventory.
    #[must_use]
    pub fn active_effects(&self) -> &[ActiveEffect] {
        &self.active_effects
    }

    /// `centered_inventory` mod: `centered` keeps the player inventory centred while
    /// effects are active (vanilla shifts it right), and `show_effects == false` skips the effect
    /// panels. `(false, true)` is vanilla. Applied at once, and by every later effect update.
    pub fn set_effect_layout(&mut self, centered: bool, show_effects: bool) {
        self.effects_centered = centered;
        self.effects_hidden = !show_effects;
        self.update_active_potion_effects();
    }

    /// Return the enchanting table's interpolated book pose for the current render frame.
    ///
    /// The pose is `None` for every other container kind. A caller that owns the 3-D model can
    /// pass the result to the equivalent of `ModelBook.render`; no 2-D vertices are emitted here.
    #[must_use]
    pub fn enchanting_book_pose(&self, partial_ticks: f32) -> Option<EnchantingBookPose> {
        matches!(&self.art, ContainerArt::Enchanting(_))
            .then(|| self.enchanting_book.pose(partial_ticks))
    }

    /// `GuiBeacon.updateScreen`: confirm is refreshed every tick, while power buttons are built
    /// only once behind `buttonsNotDrawn && levels >= 0`.
    fn beacon_buttons_for_draw(&mut self) -> Option<Vec<BeaconButton>> {
        let ContainerArt::Beacon(art) = &self.art else {
            return None;
        };
        let mut current = art.buttons(self.container.slot_stack(0).is_some());
        let power_buttons = current.split_off(2);
        if self.beacon_power_buttons.is_none() && art.levels >= 0 {
            self.beacon_power_buttons = Some(power_buttons);
        }
        if let Some(snapshot) = &self.beacon_power_buttons {
            current.extend_from_slice(snapshot);
        }
        Some(current)
    }

    fn draw_beacon_payment_items(&self, d: &mut DrawList) {
        if !matches!(&self.art, ContainerArt::Beacon(_)) {
            return;
        }
        draw_beacon_payment_items_at(d, &self.atlas, self.gui_left, self.gui_top);
    }

    fn draw_beacon_buttons(&mut self, d: &mut DrawList, mouse: (i32, i32)) {
        self.hovered_beacon_button = None;
        let Some(buttons) = self.beacon_buttons_for_draw() else {
            return;
        };
        let tex = self.art.texture();
        let hovered = buttons
            .iter()
            .copied()
            .find(|button| self.beacon_button_contains(*button, mouse.0, mouse.1));
        self.hovered_beacon_button = hovered;
        for button in buttons {
            draw_beacon_button(
                d,
                &tex,
                self.gui_left,
                self.gui_top,
                button,
                hovered == Some(button),
            );
        }
    }

    fn beacon_button_contains(&self, button: BeaconButton, x: i32, y: i32) -> bool {
        x >= self.gui_left + button.x
            && y >= self.gui_top + button.y
            && x < self.gui_left + button.x + 22
            && y < self.gui_top + button.y + 22
    }

    fn beacon_button_tooltip(&self) -> Option<Vec<String>> {
        let button = self.hovered_beacon_button?;
        let text = match button.kind {
            BeaconButtonKind::Confirm => self.language.as_ref().map_or_else(
                || "Done".to_owned(),
                |language| language.tr("gui.done").to_owned(),
            ),
            BeaconButtonKind::Cancel => self.language.as_ref().map_or_else(
                || "Cancel".to_owned(),
                |language| language.tr("gui.cancel").to_owned(),
            ),
            BeaconButtonKind::Power { tier, potion } => {
                let info = mc_data::potion(potion)?;
                let mut text = self.language.as_ref().map_or_else(
                    || info.display_name.to_owned(),
                    |language| language.tr(info.name).to_owned(),
                );
                // java: GuiBeacon.PowerButton.drawButtonForegroundLayer appends this literal
                // suffix for the secondary tier except regeneration.
                if tier >= 3 && potion != 10 {
                    text.push_str(" II");
                }
                text
            }
        };
        Some(vec![text])
    }

    // java: InventoryEffectRenderer.java:35-48 (updateActivePotionEffects)
    fn update_active_potion_effects(&mut self) {
        self.gui_left = if self.art == ContainerArt::Player
            && !self.active_effects.is_empty()
            && !self.effects_centered
        {
            160 + (self.screen_width - self.art.x_size() - 200) / 2
        } else {
            (self.screen_width - self.art.x_size()) / 2
        };
        // 1.7.10 keeps the position its `initGui` chose (`screen_v5.rs`).
        self.keep_init_gui_left_v5();
    }

    // java: InventoryEffectRenderer.java:71-112 (drawActivePotionEffects)
    fn draw_active_potion_effects(&self, d: &mut DrawList, font: &mut FontRenderer) {
        if self.art != ContainerArt::Player || self.active_effects.is_empty() || self.effects_hidden
        {
            return;
        }
        let texture = UiTexture::gui(INVENTORY_TEXTURE);
        let x = self.gui_left - 124;
        let mut y = self.gui_top;
        let pitch = if self.active_effects.len() > 5 {
            132 / (self.active_effects.len() as i32 - 1)
        } else {
            33
        };
        for effect in &self.active_effects {
            d.blit(&texture, x, y, 0, 166, 140, 32);
            if effect.icon_index >= 0 {
                d.blit(
                    &texture,
                    x + 6,
                    y + 7,
                    (effect.icon_index % 8) * 18,
                    198 + (effect.icon_index / 8) * 18,
                    18,
                    18,
                );
            }
            font.draw_string_with_shadow(
                d,
                &effect.name,
                (x + 28) as f32,
                (y + 6) as f32,
                0x00FF_FFFF,
            );
            font.draw_string_with_shadow(
                d,
                &effect.duration,
                (x + 28) as f32,
                (y + 16) as f32,
                0x007F_7F7F,
            );
            y += pitch;
        }
    }

    /// Refresh translations used by item names and the player-inventory heading.
    pub fn set_language(&mut self, language: &Language) {
        self.language = Some(language.clone());
        self.language_generation = self.language_generation.wrapping_add(1);
        self.player_title = language.tr("container.inventory").to_owned();
        if matches!(self.art, ContainerArt::Anvil(_)) {
            // java: GuiRepair does not consult the server window title here; it resolves this
            // key from the active language on every draw.
            self.title = language.tr("container.repair").to_owned();
        }
    }

    /// Synchronize `GuiRepair.nameField` from slot 0, exactly as
    /// `GuiRepair.sendSlotContents(..., slotInd = 0, stack)` does.
    ///
    /// The host supplies `display_name` because resolving `ItemStack.getDisplayName()` is an
    /// asset/language operation. `None` clears and disables the field without producing a
    /// request. `Some` installs the name, enables editing, and queues the normalized value for
    /// `MC|ItemName`, just like vanilla's trailing `if (stack != null) renameItem()`.
    /// `has_custom_name` is the exact `ItemStack.hasDisplayName()` input to that normalization.
    pub fn sync_anvil_input_name(
        &mut self,
        display_name: Option<&str>,
        has_custom_name: bool,
        font: &FontRenderer,
    ) {
        let input = self.container.slot_stack(0).cloned();
        self.send_anvil_slot_contents(input, display_name, has_custom_name, Some(font));
    }

    /// `GuiRepair.sendSlotContents(container, 0, stack)`: install the name, enable the field and
    /// call `renameItem` (one queued `MC|ItemName`), or clear and disable it for an empty slot.
    /// `input` becomes the `detectAndSendChanges` snapshot of slot 0.
    ///
    /// With no font (a slot click; `Screen::mouse_clicked` has none) the field's
    /// scroll window is laid out by its next font-bearing call, which gives the same result.
    fn send_anvil_slot_contents(
        &mut self,
        input: Option<ItemStack>,
        display_name: Option<&str>,
        has_custom_name: bool,
        font: Option<&FontRenderer>,
    ) {
        let Some(field) = self.anvil_name.as_mut() else {
            return;
        };
        self.anvil_synced_input = input;
        let text = display_name.unwrap_or("");
        match font {
            Some(font) => field.set_text(text, font),
            None => field.set_text_deferred_layout(text),
        }
        match display_name {
            Some(name) => {
                field.enabled = true;
                self.anvil_input_name = Some(name.to_owned());
                self.anvil_input_has_custom_name = has_custom_name;
                if let ContainerArt::Anvil(art) = &mut self.art {
                    art.rename_text = field.text();
                }
                // `updateRepairOutput` compares with this same
                // `getDisplayName()`, in the player's language.
                let resolved = self
                    .anvil_synced_input
                    .clone()
                    .map(|stack| (stack, name.to_owned()));
                self.container.set_anvil_input_display_name(resolved);
                self.queue_anvil_name();
            }
            None => {
                field.enabled = false;
                self.anvil_input_name = None;
                self.anvil_input_has_custom_name = false;
                if let ContainerArt::Anvil(art) = &mut self.art {
                    art.rename_text.clear();
                }
                self.container.set_anvil_input_display_name(None);
                self.container.set_anvil_name("");
            }
        }
    }

    /// The anvil half of the `detectAndSendChanges` that `Container.slotClick`
    /// triggers: every input-slot mutation is an `InventoryBasic.markDirty`, whose
    /// `ContainerRepair.onCraftMatrixChanged` compares slot 0 with its last-sent snapshot and,
    /// when it differs, calls `GuiRepair.sendSlotContents(0, stack)`. That `renameItem` puts
    /// `MC|ItemName` on the wire **before** `PlayerControllerMP.windowClick` sends the click's
    /// C0E, so [`Self::dispatch`] runs this between the local prediction and the packet.
    ///
    /// `slot0` is slot 0 as the click left it: a release can apply several clicks before the
    /// screen sees them (a drag, a shift-double-click sweep), and each gets its own check.
    ///
    /// `ItemStack.getDisplayName()` resolves through the screen's own language (the host sets it
    /// on open, see [`Self::set_language`]). The wire value does not depend on it: an ordinary
    /// name normalizes to `""` and a `display.Name` is language-free.
    fn detect_anvil_slot_change(&mut self, slot0: Option<ItemStack>) {
        if self.anvil_name.is_none() || slot0 == self.anvil_synced_input {
            return;
        }
        let resolved = slot0.as_ref().map(|stack| match &self.language {
            Some(language) => tooltip::display_name_localized(stack, language),
            None => tooltip::display_name_localized(stack, &Language::default()),
        });
        match resolved {
            Some((name, custom)) => self.send_anvil_slot_contents(slot0, Some(&name), custom, None),
            None => self.send_anvil_slot_contents(None, None, false, None),
        }
    }

    /// Whether anvil slot 0 differs from the stack the name field was last synchronized from,
    /// so the host must call [`Self::sync_anvil_input_name`]. Always `false` off an anvil.
    ///
    /// java: `Container.detectAndSendChanges` calls `GuiRepair.sendSlotContents(…, 0, …)`
    /// only when `!ItemStack.areItemStacksEqual(snapshot, slot0)`. The renamed slot-2 output
    /// the server answers every `MC|ItemName` with leaves slot 0 alone, so typing survives it.
    #[must_use]
    pub fn anvil_input_needs_sync(&self) -> bool {
        self.anvil_name.is_some()
            && self.container.slot_stack(0) != self.anvil_synced_input.as_ref()
    }

    /// Current visible anvil name, if this is an anvil screen.
    #[must_use]
    pub fn anvil_rename_text(&self) -> Option<String> {
        self.anvil_name.as_ref().map(TextField::text)
    }

    /// Whether the anvil name field currently accepts edits. Vanilla disables it while slot 0
    /// is empty.
    #[must_use]
    pub fn anvil_name_enabled(&self) -> bool {
        self.anvil_name.as_ref().is_some_and(|field| field.enabled)
    }

    /// Drain only the queued `MC|ItemName` names, leaving every other action in order. Prefer
    /// [`Self::take_ordered_outgoing`] when sending to a session: it carries these as
    /// `MC|ItemName` packets in their place among the C0E/C11 actions.
    pub fn take_anvil_name_changes(&mut self) -> Vec<String> {
        let actions = std::mem::take(&mut self.outgoing_actions);
        let mut names = Vec::new();
        let mut remaining = Vec::with_capacity(actions.len());
        for action in actions {
            match action {
                OrderedContainerAction::AnvilName(name) => names.push(name),
                other => remaining.push(other),
            }
        }
        self.outgoing_actions = remaining;
        names
    }

    /// `GuiRepair.renameItem`: normalize the field text, set the container's name and queue the
    /// `MC|ItemName` in the ordered click stream.
    fn queue_anvil_name(&mut self) {
        let Some(field) = self.anvil_name.as_ref() else {
            return;
        };
        let text = field.text();
        let normalized = if !self.anvil_input_has_custom_name
            && self.anvil_input_name.as_deref() == Some(text.as_str())
        {
            String::new()
        } else {
            text
        };
        self.container.set_anvil_name(normalized.clone());
        self.outgoing_actions
            .push(OrderedContainerAction::AnvilName(normalized));
    }

    /// The always-open player inventory. java: `GuiInventory`.
    #[must_use]
    pub fn player(window_id: u8) -> Self {
        Self::new(
            layouts::player_container(window_id),
            ContainerArt::Player,
            "Crafting",
        )
    }

    /// java: `GuiCrafting`.
    ///
    /// The title is a **parameter and not the packet's**: `GuiCrafting` always draws
    /// `I18n.format("container.crafting")`, and the `S2DPacketOpenWindow` for a crafting table
    /// carries the block's display name, which vanilla discards. `for_open_window` passes the
    /// localised string, never `pkt.window_title`.
    #[must_use]
    pub fn crafting_table(window_id: u8, title: impl Into<String>) -> Self {
        Self::new(
            layouts::crafting_table_container(window_id),
            ContainerArt::Crafting,
            title,
        )
    }

    /// A chest of `rows` rows. `3` is a single/ender chest, `6` a double chest, and `0` is
    /// vanilla's answer for a plugin menu of fewer than nine slots. java: `GuiChest`.
    #[must_use]
    pub fn chest(window_id: u8, rows: u8, title: impl Into<String>) -> Self {
        Self::new(
            layouts::chest_container(window_id, rows),
            ContainerArt::Chest { rows },
            title,
        )
    }

    /// A hopper or hopper-minecart. java: `GuiHopper`.
    #[must_use]
    pub fn hopper(window_id: u8, slot_count: u8, title: impl Into<String>) -> Self {
        Self::new(
            layouts::hopper_container(window_id, slot_count),
            ContainerArt::Hopper,
            title,
        )
    }

    /// A dispenser or dropper. java: `GuiDispenser`.
    #[must_use]
    pub fn dispenser(window_id: u8, title: impl Into<String>) -> Self {
        Self::new(
            layouts::dispenser_container(window_id),
            ContainerArt::Dispenser,
            title,
        )
    }

    /// A furnace. java: `GuiFurnace`.
    #[must_use]
    pub fn furnace(window_id: u8, title: impl Into<String>) -> Self {
        Self::new(
            layouts::furnace_container(window_id),
            ContainerArt::Furnace(FurnaceArt::default()),
            title,
        )
    }

    /// A brewing stand. java: `GuiBrewingStand`.
    #[must_use]
    pub fn brewing_stand(window_id: u8, title: impl Into<String>) -> Self {
        Self::new(
            layouts::brewing_stand_container(window_id),
            ContainerArt::BrewingStand(BrewingStandArt::default()),
            title,
        )
    }

    /// An enchanting table. java: `GuiEnchantment`.
    #[must_use]
    pub fn enchanting_table(window_id: u8, title: impl Into<String>) -> Self {
        Self::new(
            layouts::enchanting_table_container(window_id),
            ContainerArt::Enchanting(EnchantingArt {
                enchantment_ids: [-1; 3],
                ..EnchantingArt::default()
            }),
            title,
        )
    }

    /// An anvil. java: `GuiRepair`.
    #[must_use]
    pub fn anvil(window_id: u8, title: impl Into<String>) -> Self {
        Self::new(
            layouts::anvil_container(window_id),
            ContainerArt::Anvil(AnvilArt::default()),
            title,
        )
    }

    /// A beacon. java: `GuiBeacon`.
    #[must_use]
    pub fn beacon(window_id: u8, title: impl Into<String>) -> Self {
        Self::new(
            layouts::beacon_container(window_id),
            // java: `ContainerLocalMenu.getField` returns zero until a server property
            // populates the field; `GuiBeacon` therefore builds its initially-disabled power
            // buttons on the first update tick.
            ContainerArt::Beacon(BeaconArt::default()),
            title,
        )
    }

    /// A villager trading screen. Offers are populated later by `MC|TrList`.
    #[must_use]
    pub fn merchant(window_id: u8, title: impl Into<String>) -> Self {
        let container = super::net::container_for_open_window(&s2c::OpenWindow {
            window_id,
            inventory_type: "minecraft:villager".to_owned(),
            window_title: String::new(),
            slot_count: 3,
            entity_id: None,
        })
        .expect("villager dispatch builds a merchant container");
        Self::new(
            container,
            ContainerArt::Merchant(MerchantArt::default()),
            title,
        )
    }

    /// A horse, donkey, mule, or llama inventory. java: `GuiScreenHorseInventory`.
    ///
    /// The wire packet carries the entity id and reports either two slots (an ordinary
    /// un-chested horse) or seventeen (a chested mount). The host-facing constructor takes the
    /// already-resolved `chested` bit; [`Self::for_open_window`] derives it from the packet.
    #[must_use]
    pub fn horse(window_id: u8, chested: bool, title: impl Into<String>) -> Self {
        let slot_count = if chested { 17 } else { 2 };
        let container = super::net::container_for_open_window(&s2c::OpenWindow {
            window_id,
            inventory_type: "EntityHorse".to_owned(),
            window_title: String::new(),
            slot_count,
            entity_id: Some(0),
        })
        .expect("valid horse dispatch builds a horse container");
        let mut screen = Self::new(container, ContainerArt::Horse { chested }, title);
        screen.horse_entity_id = Some(0);
        screen
    }

    /// Build the right screen for a server `S2DPacketOpenWindow`, or `None` for a window type
    /// this section does not cover — the same dispatch (and the same coverage) as
    /// [`super::net::container_for_open_window`], which this delegates its type list to.
    ///
    /// `title` is the already-resolved window title: `OpenWindow.window_title` is a chat
    /// component and resolving it needs a `Language`, which this crate's screens do not carry.
    #[must_use]
    pub fn for_open_window(pkt: &s2c::OpenWindow, title: impl Into<String>) -> Option<Self> {
        let title = title.into();
        // The container half is [`super::net::container_for_open_window`]'s job and is asked
        // first, so the two dispatches cannot drift: a type it declines has no screen either.
        // (A07-07 made its last arm a chest fallback rather than `None`; this arm follows.)
        let container = super::net::container_for_open_window(pkt)?;
        let mut screen = match pkt.inventory_type.as_str() {
            // java: `EntityPlayerSP.displayGui` hard-codes `I18n.format("container.crafting")`
            // through `GuiCrafting` and never looks at the packet's title. Passing `title` here
            // would draw the block's display name where vanilla draws "Crafting".
            "minecraft:crafting_table" if pkt.slot_count == 0 => {
                Self::new(container, ContainerArt::Crafting, "Crafting")
            }
            "minecraft:hopper" => Self::new(container, ContainerArt::Hopper, title),
            "minecraft:dispenser" | "minecraft:dropper" => {
                Self::new(container, ContainerArt::Dispenser, title)
            }
            "minecraft:furnace" => Self::new(
                container,
                ContainerArt::Furnace(FurnaceArt::default()),
                title,
            ),
            "minecraft:brewing_stand" => Self::new(
                container,
                ContainerArt::BrewingStand(BrewingStandArt::default()),
                title,
            ),
            "minecraft:enchanting_table" if pkt.slot_count == 0 => Self::new(
                container,
                ContainerArt::Enchanting(EnchantingArt {
                    enchantment_ids: [-1; 3],
                    ..EnchantingArt::default()
                }),
                title,
            ),
            // java: `GuiRepair.drawGuiContainerForegroundLayer:73` draws
            // `I18n.format("container.repair")` -- "Repair & Name" -- and never the packet's
            // title, exactly like `GuiCrafting`. The server *does* send one
            // (`BlockAnvil.Anvil.getDisplayName()` is `tile.anvil.name` = "Anvil"), and vanilla
            // discards it; passing `title` here drew "Anvil", and let a server rename a heading
            // vanilla makes immovable.
            "minecraft:anvil" if pkt.slot_count == 0 => Self::new(
                container,
                ContainerArt::Anvil(AnvilArt::default()),
                "Repair & Name",
            ),
            "minecraft:beacon" => {
                Self::new(container, ContainerArt::Beacon(BeaconArt::default()), title)
            }
            "minecraft:villager" => Self::new(
                container,
                ContainerArt::Merchant(MerchantArt::default()),
                title,
            ),
            "EntityHorse" => Self::new(
                container,
                ContainerArt::Horse {
                    chested: pkt.slot_count > 2,
                },
                title,
            ),
            // java: `displayGUIChest`'s default. `"minecraft:chest"`, `"minecraft:container"` and
            // every unrecognised slotted guiId all draw the chest panel.
            // java: `GuiChest`'s `inventoryRows = lowerInv.getSizeInventory() / 9` -- integer
            // truncation with **no** floor, so a 1..8-slot plugin menu really does get zero
            // container rows and a player half 18px higher.
            _ => Self::chest(pkt.window_id, pkt.slot_count / 9, title),
        };
        if pkt.inventory_type == "EntityHorse" {
            // `container_for_open_window` has already rejected a missing id and malformed slot
            // count. Retain the exact id for the state/snapshot freshness gate.
            screen.horse_entity_id = pkt.entity_id;
        }
        Some(screen)
    }

    /// The window's left edge, `(width - xSize) / 2`. java: `GuiContainer.guiLeft`.
    #[must_use]
    pub const fn gui_left(&self) -> i32 {
        self.gui_left
    }

    /// The window's top edge, `(height - ySize) / 2`. java: `GuiContainer.guiTop`.
    #[must_use]
    pub const fn gui_top(&self) -> i32 {
        self.gui_top
    }

    /// The slot the cursor was over on the last [`Screen::draw`], if any. java:
    /// `GuiContainer.theSlot`.
    #[must_use]
    pub const fn hovered_slot(&self) -> Option<u16> {
        self.hovered_slot
    }

    /// Set the double-click clock from the host's real monotonic milliseconds.
    pub fn set_clock_ms(&mut self, ms: u64) {
        self.clock_ms = ms;
    }

    /// Take every `ClickWindow` packet still queued, preserving its order relative to other C0E
    /// actions. Prefer [`Self::take_ordered_outgoing`] when sending to a session: this legacy
    /// category drain remains for callers that only inspect slot-click packets.
    pub fn take_outgoing(&mut self) -> Vec<c2s::ClickWindow> {
        let actions = std::mem::take(&mut self.outgoing_actions);
        let mut clicks = Vec::new();
        let mut remaining = Vec::with_capacity(actions.len());
        for action in actions {
            match action {
                OrderedContainerAction::ClickWindow(packet) => clicks.push(packet),
                other => remaining.push(other),
            }
        }
        self.outgoing_actions = remaining;
        clicks
    }

    /// Take every `C11PacketEnchantItem` still queued, preserving its order relative to other C11
    /// actions. Prefer [`Self::take_ordered_outgoing`] when sending to a session.
    pub fn take_enchant_outgoing(&mut self) -> Vec<c2s::EnchantItem> {
        let actions = std::mem::take(&mut self.outgoing_actions);
        let mut enchants = Vec::new();
        let mut remaining = Vec::with_capacity(actions.len());
        for action in actions {
            match action {
                OrderedContainerAction::EnchantItem(packet) => enchants.push(packet),
                other => remaining.push(other),
            }
        }
        self.outgoing_actions = remaining;
        enchants
    }

    /// Take C0E slot clicks, C11 enchant actions and anvil `MC|ItemName` payloads in the exact
    /// order their Java callbacks invoked them. `GuiContainer.handleMouseClick`,
    /// `GuiEnchantment.mouseClicked` and `GuiRepair.renameItem` each enqueue immediately;
    /// draining separate vectors would batch all C0E before C11, and a slot
    /// click's `MC|ItemName` after the C0E that `slotClick` sends it ahead of.
    pub fn take_ordered_outgoing(&mut self) -> Vec<mc_protocol::play::C2SPlay> {
        std::mem::take(&mut self.outgoing_actions)
            .into_iter()
            .map(|action| match action {
                OrderedContainerAction::ClickWindow(packet) => {
                    mc_protocol::play::C2SPlay::ClickWindow(packet)
                }
                OrderedContainerAction::EnchantItem(packet) => {
                    mc_protocol::play::C2SPlay::EnchantItem(packet)
                }
                OrderedContainerAction::AnvilName(name) => {
                    mc_protocol::play::C2SPlay::PluginMessage(anvil_name_message(&name))
                }
            })
            .collect()
    }

    /// Take plugin payloads produced by container controls, in click order. The host should
    /// encode these as serverbound `PluginMessage` packets (`MC|Beacon` and `MC|TrSel`).
    pub fn take_plugin_outgoing(&mut self) -> Vec<c2s::PluginMessage> {
        std::mem::take(&mut self.outgoing_plugins)
    }

    /// Whether the close this screen just returned is vanilla's
    /// `displayGuiScreen(null)` from inside the GUI (`GuiBeacon.actionPerformed` Done/Cancel,
    /// `GuiContainer.mouseClicked`'s touchscreen outside press) rather than
    /// `EntityPlayerSP.closeScreen()`. The host must then run the local `onGuiClosed` hook with
    /// the carried stack still installed and send **no** `C0DPacketCloseWindow`. Reading clears
    /// the flag.
    pub fn take_closes_without_close_window(&mut self) -> bool {
        std::mem::take(&mut self.closes_without_close_window)
    }

    /// Apply a villager `MC|TrList` payload. Returns `false` when this is not a matching merchant
    /// window; malformed payloads are returned to the host for logging/connection policy.
    ///
    /// java: `NetHandlerPlayClient.handleCustomPayload` reads the four-byte window id before it
    /// checks either `currentScreen instanceof GuiMerchant` or the player's open-container id.
    /// Once that header is present, a foreign window (or a non-merchant screen) is a silent
    /// no-op: vanilla never attempts to decode its recipe bytes. Keep that ordering here so a
    /// truncated payload for another window cannot poison the active merchant, while a truncated
    /// header still reports the same `IOException` the Java handler logs.
    pub fn apply_merchant_plugin(&mut self, data: &[u8]) -> Result<bool, mc_protocol::ProtoError> {
        let mut header = mc_protocol::PacketBuffer::from_slice(data);
        let window_id = header.read_i32()?;
        if !matches!(&self.art, ContainerArt::Merchant(_))
            || window_id != i32::from(self.container.window_id)
        {
            return Ok(false);
        }

        // Decode only after the Java-equivalent screen/window gate. The parser intentionally
        // starts at byte zero because the packet's recipe list includes the same header.
        let (_, recipes) = super::net::merchant_recipes_from_plugin(data)?;
        let ContainerArt::Merchant(merchant) = &mut self.art else {
            return Ok(false);
        };
        let page = merchant.selected;
        merchant.recipes = recipes;
        merchant.selected = 0;
        let first = merchant.recipes.first().cloned();
        self.hovered_merchant_preview = None;
        // A 1.7.10 window keeps its page and its output (`screen_v5.rs`).
        if !self.apply_merchant_list_v5(page) {
            self.container.set_merchant_recipe(first);
        }
        Ok(true)
    }

    /// Select a merchant offer and queue the exact `MC|TrSel` payload.
    pub fn select_merchant_recipe(&mut self, index: usize) -> bool {
        let (data, recipe) = match &mut self.art {
            ContainerArt::Merchant(merchant) => {
                let data = merchant.select(index);
                (data, merchant.selected_recipe().cloned())
            }
            _ => (None, None),
        };
        let Some(data) = data else {
            return false;
        };
        self.hovered_merchant_preview = None;
        // A 1.7.10 container takes the whole list (`screen_v5.rs`).
        if !self.sync_merchant_offers_v5() {
            self.container.set_merchant_recipe(recipe);
        }
        self.outgoing_plugins.push(c2s::PluginMessage {
            channel: "MC|TrSel".to_owned(),
            data,
        });
        true
    }

    /// Apply an `S31PacketWindowProperty` addressed at this window; ignored otherwise.
    pub fn apply_window_property(&mut self, pkt: &s2c::WindowProperty) {
        // A 1.7.10 furnace numbers its properties differently (`screen_v5.rs`).
        if self.apply_window_property_v5(pkt) {
            return;
        }
        if pkt.window_id == self.container.window_id {
            self.art.apply_window_property(pkt.var_index, pkt.var_value);
            // Deliberately NOT mirrored onto `Container::maximum_cost`, even though that is the
            // same `ContainerRepair` field `0`: an anvil's only `S31PacketWindowProperty` is the
            // one `onCraftGuiOpened` sends at open, carrying `0`, and it is never refreshed
            // (`ContainerRepair.java:389-393`, which is the whole of that container's
            // `sendProgressBarUpdate` surface). Feeding a permanently-stale `0` into
            // `Container::can_take_stack` would refuse every anvil take. See that field's doc.
        }
    }

    /// Apply a server `S2FPacketSetSlot` to this screen's container and refresh the client-visible
    /// enchanting offers when the table inventory changes.
    ///
    /// java: `NetHandlerPlayClient.handleSetSlot` calls `Container.putStackInSlot`, whose
    /// `Slot.putStack` marks the backing inventory dirty. For `ContainerEnchantment`, that is
    /// the two-slot table inventory, so a server update to either slot re-enters
    /// `onCraftMatrixChanged`; its client-side invalid-input branch clears both offer arrays.
    /// The cursor sentinel (`windowId == -1`) and foreign windows do not touch that inventory.
    pub fn apply_set_slot(&mut self, pkt: &s2c::SetSlot) {
        self.container.apply_set_slot(pkt);
        if self.container.take_enchanting_input_change() {
            self.clear_invalid_enchanting_offers();
        }
    }

    /// Apply a server `S30PacketWindowItems` and refresh the client-visible enchanting offers.
    ///
    /// java: `NetHandlerPlayClient.handleWindowItems` calls `Container.putStacks`, which invokes
    /// the enchanting table inventory's dirty callback while replacing its slots. A normal
    /// packet always includes slot zero; the non-empty guard also preserves the no-op behavior
    /// of the loop for a malformed empty list.
    pub fn apply_window_items(&mut self, pkt: &s2c::WindowItems) {
        self.container.apply_window_items(pkt);
        if self.container.take_enchanting_input_change() {
            self.clear_invalid_enchanting_offers();
        }
    }

    /// Tell the container the player's XP level, as `S1FPacketSetExperience` last reported it.
    ///
    /// java: `EntityPlayer.experienceLevel`, read by the anvil output slot's `canTakeStack`
    /// (`ContainerRepair.java:73`) and by `GuiRepair`'s red-cost test (`GuiRepair.java:90`).
    /// Until a host calls this the level conjunct is treated as satisfied rather than as `0` --
    /// see [`Container::set_player_experience_level`]'s field doc.
    pub fn set_player_experience_level(&mut self, level: i32) {
        self.container.set_player_experience_level(level);
        match &mut self.art {
            ContainerArt::Enchanting(e) => e.player_level = level,
            ContainerArt::Anvil(a) => a.player_level = Some(level),
            ContainerArt::Furnace(_)
            | ContainerArt::BrewingStand(_)
            | ContainerArt::Beacon(_)
            | ContainerArt::Player
            | ContainerArt::Crafting
            | ContainerArt::Chest { .. }
            | ContainerArt::Hopper
            | ContainerArt::Dispenser
            | ContainerArt::Merchant(_)
            | ContainerArt::Horse { .. } => {}
        }
    }

    /// Feed one client-world `MapData.scale` into the live crafting predictor. The host calls
    /// this after a map packet updates its world record; the container then refreshes a pending
    /// 3x3 map-extending result immediately.
    pub fn set_map_scale(&mut self, map_id: i32, scale: Option<i8>) {
        self.container.set_map_scale(map_id, scale);
    }

    /// Return the signed metadata ids of filled-map stacks that can be visible on this frame.
    ///
    /// The application uses this bounded list to fetch only the corresponding world records;
    /// the screen never receives or retains the complete world's map registry. The cursor and the
    /// selected merchant offer are included because vanilla renders both through the same item
    /// renderer as ordinary container slots.
    #[must_use]
    pub fn map_ids(&self) -> Vec<i32> {
        let mut ids = BTreeSet::new();
        let mut collect = |stack: &ItemStack| {
            if stack_ext::try_props(stack).is_some_and(|props| props.name == "minecraft:filled_map")
            {
                ids.insert(i32::from(stack.damage));
            }
        };
        for stack in self
            .container
            .slots()
            .iter()
            .filter_map(|slot| self.container.slot_stack(usize::from(slot.id)))
        {
            collect(stack);
        }
        if let Some(stack) = self.container.cursor() {
            collect(stack);
        }
        if let ContainerArt::Merchant(merchant) = &self.art {
            if let Some(recipe) = merchant.selected_recipe() {
                collect(&recipe.buy_a);
                if let Some(stack) = &recipe.buy_b {
                    collect(stack);
                }
                collect(&recipe.sell);
            }
        }
        ids.into_iter().collect()
    }

    /// Retain the host-facing map-record seam without feeding live pixels into `GuiContainer`
    /// slots. Java's `RenderItem` uses the static `map_filled` item model here; live records are
    /// still supplied separately through [`Self::draw_with_map_data`] for advanced tooltips.
    ///
    /// The argument is intentionally ignored. Keeping this no-op preserves the application
    /// boundary used by hosts that also drive the creative screen, whose renderer has a distinct
    /// dynamic-map path.
    pub fn set_map_render_data(&mut self, _records: Vec<MapData>) {}

    /// Draw with the live tooltip record and the compatibility map-record feed used by the host.
    /// `GuiContainer` slot imagery remains static; [`Self::draw_with_map_data`] is the path that
    /// supplies the record to advanced filled-map tooltips.
    pub fn draw_with_map_records(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        mouse: (i32, i32),
        tooltip_map_data: Option<&MapData>,
        records: Vec<MapData>,
    ) {
        self.set_map_render_data(records);
        self.draw_with_map_data(d, font, ctx, mouse, tooltip_map_data);
    }

    /// Live container draw using Java's independent `standardGalacticFontRenderer` for the
    /// enchanting offer names while retaining the ordinary renderer for titles and level digits.
    // The public signature is the stable application seam for separate ordinary/SGA renderers;
    // collapsing it would change the host API and Java renderer ordering.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_with_map_records_and_sga(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        sga_font: &mut FontRenderer,
        ctx: &ScreenCtx,
        mouse: (i32, i32),
        tooltip_map_data: Option<&MapData>,
        records: Vec<MapData>,
    ) {
        self.prepare_enchanting_runes(sga_font, mouse);
        self.draw_with_map_records(d, font, ctx, mouse, tooltip_map_data, records);
    }

    fn prepare_enchanting_runes(
        &mut self,
        sga_font: &mut FontRenderer,
        (mouse_x, mouse_y): (i32, i32),
    ) {
        for row in &mut self.enchanting_runes {
            row.clear();
        }
        let ContainerArt::Enchanting(mut art) = self.art else {
            return;
        };
        art.creative = self.creative;
        let names = art.rune_offer_names();
        let lapis = self.container.slot_stack(1).map_or(0, |stack| stack.count);
        let lapis = v5::offer_lapis_v5(&self.container, lapis);
        let hovered = self.enchant_plate_at_position(mouse_x, mouse_y);
        for (row, name) in names.iter().enumerate() {
            if art.enchant_levels[row] == 0 {
                continue;
            }
            let color = if !art.is_affordable(row, lapis) {
                (6_839_882 & 16_711_422) >> 1
            } else if hovered == Some(row) {
                16_777_088
            } else {
                6_839_882
            };
            sga_font.draw_split_string(
                &mut self.enchanting_runes[row],
                name,
                self.gui_left + 80,
                self.gui_top + 16 + 19 * row as i32,
                86,
                color,
            );
        }
    }

    /// Vanilla's slot/point hit test, borders and all.
    ///
    /// java: `GuiContainer.isPointInRegion` — note the `- 1` / `+ 1`: the region tested is one
    /// pixel **larger** on every side than the rectangle passed in, so a 16x16 slot is actually
    /// hoverable across 18x18 pixels and two adjacent slots' hit boxes overlap by two columns.
    /// The slot loop resolves that overlap by taking the *first* match in slot order.
    #[must_use]
    pub fn is_point_in_region(
        &self,
        left: i32,
        top: i32,
        width: i32,
        height: i32,
        point_x: i32,
        point_y: i32,
    ) -> bool {
        let x = point_x - self.gui_left;
        let y = point_y - self.gui_top;
        x >= left - 1 && x < left + width + 1 && y >= top - 1 && y < top + height + 1
    }

    /// The first slot whose 18x18 hit box contains the point. java:
    /// `GuiContainer.getSlotAtPosition`.
    #[must_use]
    pub fn slot_at_position(&self, x: i32, y: i32) -> Option<u16> {
        self.container
            .slots()
            .iter()
            .find(|s| self.is_point_in_region(s.x, s.y, 16, 16, x, y))
            .map(|s| s.id)
    }

    /// Whether the point is off the window panel entirely. java: `GuiContainer.java:372`'s
    /// `flag1`.
    #[must_use]
    pub fn is_outside_window(&self, x: i32, y: i32) -> bool {
        x < self.gui_left
            || y < self.gui_top
            || x >= self.gui_left + self.art.x_size()
            || y >= self.gui_top + self.art.y_size()
    }

    /// Resolve a screen point to the three-way target [`MouseState`] consumes.
    ///
    /// **Outside wins over a slot.** Because [`is_point_in_region`](Self::is_point_in_region)
    /// grows every slot by a pixel, a slot flush with the panel edge is "hovered" at a point
    /// that is already outside the window; vanilla computes `l` from the slot first and then
    /// unconditionally overwrites it with `-999` when `flag1` holds, so the outside case wins.
    #[must_use]
    pub fn target_at(&self, x: i32, y: i32) -> SlotTarget {
        if self.is_outside_window(x, y) {
            SlotTarget::Outside
        } else if let Some(id) = self.slot_at_position(x, y) {
            SlotTarget::Slot(id)
        } else {
            SlotTarget::Inside
        }
    }

    /// The raw 108x19 plate hit test inside `GuiEnchantment.drawGuiContainerBackgroundLayer`.
    /// This intentionally does not use [`Self::is_point_in_region`]: the later tooltip pass does,
    /// with height 17 and its one-pixel expansion, but the plate hover uses ordinary bounds.
    fn enchant_plate_at_position(&self, x: i32, y: i32) -> Option<usize> {
        if !matches!(&self.art, ContainerArt::Enchanting(_)) {
            return None;
        }
        let local_x = x - (self.gui_left + 60);
        (0..3usize).find(|row| {
            let local_y = y - (self.gui_top + 14 + 19 * *row as i32);
            (0..108).contains(&local_x) && (0..19).contains(&local_y)
        })
    }

    /// Return the selected recipe stack rendered at a merchant preview position. The preview
    /// coordinates are the exact `GuiMerchant` item positions and use the same expanded
    /// `isPointInRegion` bounds as ordinary item tooltips.
    fn merchant_preview_at_position(&self, x: i32, y: i32) -> Option<&ItemStack> {
        let ContainerArt::Merchant(merchant) = &self.art else {
            return None;
        };
        let recipe = merchant.selected_recipe()?;
        if self.is_point_in_region(36, 24, 16, 16, x, y) {
            Some(&recipe.buy_a)
        } else if self.is_point_in_region(62, 24, 16, 16, x, y) {
            recipe.buy_b.as_ref()
        } else if self.is_point_in_region(120, 24, 16, 16, x, y) {
            Some(&recipe.sell)
        } else {
            None
        }
    }

    /// Run the client-visible half of `ContainerEnchantment.onCraftMatrixChanged` after a
    /// predicted click mutates the table inventory. The offer calculation is server-only, but
    /// clearing stale offers for an empty or non-enchantable input is deliberately outside
    /// Java's `worldPointer.isRemote` guard.
    fn clear_invalid_enchanting_offers(&mut self) {
        let should_clear = matches!(&self.art, ContainerArt::Enchanting(_))
            && self
                .container
                .slot_stack(0)
                .is_none_or(|stack| !is_item_enchantable(stack));
        if should_clear {
            let ContainerArt::Enchanting(art) = &mut self.art else {
                unreachable!("the match above established the art variant");
            };
            art.enchant_levels = [0; 3];
            art.enchantment_ids = [-1; 3];
        }
    }

    /// Consume a server packet's table-inventory edge before a frame is painted. This keeps the
    /// live `Container::apply_set_slot`/`apply_window_items` path faithful even when the host
    /// updates the model directly, as `NetHandlerPlayClient` does in Java.
    fn refresh_invalid_enchanting_offers(&mut self) {
        if self.container.take_enchanting_input_change() {
            self.clear_invalid_enchanting_offers();
        }
    }

    /// Run one click triple through the container and queue the packet it produces.
    fn dispatch(&mut self, slot_id: i32, button: u8, mode: u8) {
        self.container.set_creative_mode(self.creative);
        let enchanting_input_before = self.container.slot_stack(0).cloned();
        let packet = self.container.click(slot_id, button, mode, self.creative);
        if self.container.slot_stack(0) != enchanting_input_before.as_ref() {
            self.clear_invalid_enchanting_offers();
        }
        // java: `PlayerControllerMP.windowClick` runs `slotClick` (whose
        // `detectAndSendChanges` reaches `GuiRepair.renameItem`) before it sends the C0E.
        if self.anvil_name.is_some() {
            self.detect_anvil_slot_change(self.container.slot_stack(0).cloned());
        }
        self.outgoing_actions
            .push(OrderedContainerAction::ClickWindow(packet));
    }

    /// java: `GuiContainer.checkHotbarKeys` (`:718`). With an empty cursor over a hovered slot,
    /// a code bound to `key.hotbar.N` swaps that slot with hotbar slot `N` (click mode 2).
    /// Returns whether it fired.
    fn check_hotbar_keys(&mut self, key: i32) -> bool {
        let Some(slot) = self.hovered_slot else {
            return false;
        };
        if self.container.cursor().is_some() {
            return false;
        }
        let Some(hotbar) = self
            .bindings
            .hotbar_keys
            .iter()
            .position(|bound| *bound == key)
        else {
            return false;
        };
        self.dispatch(i32::from(slot), hotbar as u8, 2);
        true
    }

    fn touchscreen_clicked(
        &mut self,
        target: SlotTarget,
        button: i32,
        pick_block: bool,
    ) -> Option<ScreenAction> {
        if !self.touchscreen || !(button == 0 || button == 1 || pick_block) {
            return None;
        }
        if target == SlotTarget::Outside && self.container.cursor().is_none() {
            return Some(ScreenAction::CloseAll);
        }
        if target != SlotTarget::Inside {
            self.touch = match target {
                SlotTarget::Slot(slot_id)
                    if self.container.slot_stack(usize::from(slot_id)).is_some() =>
                {
                    TouchscreenGesture {
                        clicked_slot: Some(slot_id),
                        dragged_stack: None,
                        right_click: button == 1,
                        hold_slot: None,
                        hold_started_ms: 0,
                    }
                }
                SlotTarget::Slot(_) | SlotTarget::Outside | SlotTarget::Inside => {
                    TouchscreenGesture::default()
                }
            };
        }
        Some(ScreenAction::None)
    }

    fn touchscreen_released(&mut self, target: SlotTarget, button: i32, x: i32, y: i32) -> bool {
        let Some(source) = self.touch.clicked_slot else {
            return false;
        };
        if button != 0 && button != 1 {
            return true;
        }

        let destination = match target {
            SlotTarget::Slot(slot_id) => Some(slot_id),
            SlotTarget::Inside | SlotTarget::Outside => None,
        };
        if self.touch.dragged_stack.is_none() && destination != Some(source) {
            self.touch.dragged_stack = self.container.slot_stack(usize::from(source)).cloned();
        }
        let dragged = self.touch.dragged_stack.clone();
        let acceptable = destination
            .zip(dragged.as_ref())
            .is_some_and(|(slot_id, stack)| {
                self.container.slot_stack_accepts_touch(slot_id, stack)
            });

        if let (Some(destination), Some(dragged)) = (destination, dragged.as_ref()) {
            if acceptable {
                self.dispatch(i32::from(source), button as u8, 0);
                self.dispatch(i32::from(destination), 0, 0);
                if self.container.cursor().is_some() {
                    self.dispatch(i32::from(source), button as u8, 0);
                    if self.container.cursor().is_some() {
                        self.returning_stack = Some(ReturningStack {
                            stack: dragged.clone(),
                            start: (x, y),
                            destination_slot: source,
                            started_ms: self.clock_ms,
                        });
                    } else {
                        self.returning_stack = None;
                    }
                } else {
                    self.returning_stack = None;
                }
            } else {
                self.returning_stack = Some(ReturningStack {
                    stack: dragged.clone(),
                    start: (x, y),
                    destination_slot: source,
                    started_ms: self.clock_ms,
                });
            }
        } else if let Some(dragged) = dragged {
            self.returning_stack = Some(ReturningStack {
                stack: dragged,
                start: (x, y),
                destination_slot: source,
                started_ms: self.clock_ms,
            });
        }

        self.touch = TouchscreenGesture::default();
        self.drag_preview = None;
        true
    }

    fn touchscreen_dragged(&mut self, target: SlotTarget, button: i32) -> bool {
        let Some(source) = self.touch.clicked_slot else {
            return false;
        };
        if !self.touchscreen || (button != 0 && button != 1) {
            return self.touchscreen;
        }
        let destination = match target {
            SlotTarget::Slot(slot_id) => Some(slot_id),
            SlotTarget::Inside | SlotTarget::Outside => None,
        };
        if self.touch.dragged_stack.is_none() && destination != Some(source) {
            self.touch.dragged_stack = self.container.slot_stack(usize::from(source)).cloned();
        }

        let can_place_one = destination
            .zip(self.touch.dragged_stack.as_ref())
            .is_some_and(|(slot_id, stack)| {
                stack.count > 1 && self.container.slot_stack_accepts_touch(slot_id, stack)
            });
        if !can_place_one {
            self.touch.hold_slot = None;
            return true;
        }

        let destination = destination.expect("the acceptance check established a slot");
        if self.touch.hold_slot != Some(destination) {
            self.touch.hold_slot = Some(destination);
            self.touch.hold_started_ms = self.clock_ms;
            return true;
        }
        if self.clock_ms.saturating_sub(self.touch.hold_started_ms) > 500 {
            self.dispatch(i32::from(source), 0, 0);
            self.dispatch(i32::from(destination), 1, 0);
            self.dispatch(i32::from(source), 0, 0);
            self.touch.hold_started_ms = self.clock_ms.saturating_add(750);
            if let Some(stack) = self.touch.dragged_stack.as_mut() {
                stack.count -= 1;
            }
        }
        true
    }

    fn returning_stack_position(&mut self) -> Option<(ItemStack, i32, i32)> {
        let returning = self.returning_stack.as_ref()?;
        let elapsed = self.clock_ms.saturating_sub(returning.started_ms);
        if elapsed >= 100 {
            self.returning_stack = None;
            return None;
        }
        let destination = self
            .container
            .slots()
            .get(usize::from(returning.destination_slot))?;
        let destination = (self.gui_left + destination.x, self.gui_top + destination.y);
        let progress = elapsed as f32 / 100.0;
        let x = returning.start.0 + ((destination.0 - returning.start.0) as f32 * progress) as i32;
        let y = returning.start.1 + ((destination.1 - returning.start.1) as f32 * progress) as i32;
        Some((returning.stack.clone(), x, y))
    }

    /// [`Screen::mouse_clicked`], plus the one thing the trait's signature cannot carry:
    /// whether shift is held.
    ///
    /// java: `GuiContainer.mouseClicked` (`:359`) reads shift from the **static**
    /// `GuiScreen.isShiftKeyDown()`, so vanilla's own method needs no parameter for it.
    /// [`Screen::mouse_clicked`] has no [`Modifiers`] argument, and the only two call sites that
    /// existed passed a hard-coded `false` — so click mode `1` (shift-transfer, the single
    /// most-used inventory gesture) was unreachable from a running client no matter how
    /// completely [`MouseState`] implemented it.
    ///
    /// A host that knows the real modifier state calls this; [`Screen::mouse_clicked`] forwards
    /// here with [`Modifiers::NONE`], which is the honest translation of "this caller does not
    /// know" rather than a claim that shift was up.
    pub fn mouse_clicked_with_mods(
        &mut self,
        x: i32,
        y: i32,
        button: i32,
        mods: Modifiers,
    ) -> ScreenAction {
        // NOT VANILLA 1.8.9, a deliberate addition. 1.8.9 calls
        // `checkHotbarKeys` only from `keyTyped` (`GuiContainer.java:699`), so a mouse button
        // bound to `key.hotbar.N` did nothing over a slot. Here it does what its keyboard
        // counterpart does: the same mode-2 C0E through the same `check_hotbar_keys`, and the
        // press (and its release) does nothing else. Delete this block, the release check in
        // `mouse_released_with_mods` and `hotbar_mouse_button` to restore 1.8.9 exactly.
        self.hotbar_mouse_button = None;
        if self.check_hotbar_keys(button - 100) {
            self.hotbar_mouse_button = Some(button);
            return ScreenAction::None;
        }
        if button == 0 {
            let local_x = x - self.gui_left;
            let local_y = y - self.gui_top;
            if let ContainerArt::Merchant(merchant) = &self.art {
                let selected = merchant.selected;
                let next = (147..159).contains(&local_x) && (23..42).contains(&local_y);
                let previous = (17..29).contains(&local_x) && (23..42).contains(&local_y);
                if next || previous {
                    let changed = if next {
                        self.select_merchant_recipe(selected.saturating_add(1))
                    } else {
                        selected
                            .checked_sub(1)
                            .is_some_and(|index| self.select_merchant_recipe(index))
                    };
                    if changed {
                        return ScreenAction::None;
                    }
                }
            }
        }
        // java: `GuiScreen.mouseClicked` dispatches buttonList before GuiContainer's slot
        // handling. Beacon controls occupy ordinary panel space, so they must be checked first
        // or every power/Done/Cancel press is silently treated as an inside-window no-op.
        if button == 0 {
            if let Some(beacon_button) = self.beacon_buttons_for_draw().and_then(|buttons| {
                buttons
                    .into_iter()
                    .find(|candidate| self.beacon_button_contains(*candidate, x, y))
            }) {
                if beacon_button.enabled {
                    match beacon_button.kind {
                        BeaconButtonKind::Confirm => {
                            let ContainerArt::Beacon(art) = &self.art else {
                                unreachable!("beacon button implies beacon art");
                            };
                            let mut data = Vec::with_capacity(8);
                            data.extend_from_slice(&art.primary.to_be_bytes());
                            data.extend_from_slice(&art.secondary.to_be_bytes());
                            self.outgoing_plugins.push(c2s::PluginMessage {
                                channel: "MC|Beacon".to_owned(),
                                data,
                            });
                            // java: actionPerformed(-1) calls displayGuiScreen(null), which invokes
                            // onGuiClosed but does not send C0D itself.
                            self.closes_without_close_window = true;
                            return ScreenAction::Pop;
                        }
                        BeaconButtonKind::Cancel => {
                            // java: actionPerformed(-2) only closes; GuiContainer's close hook does
                            // the client-side container cleanup and no explicit C0D is sent here.
                            self.closes_without_close_window = true;
                            return ScreenAction::Pop;
                        }
                        BeaconButtonKind::Power { tier, potion } => {
                            if let ContainerArt::Beacon(art) = &mut self.art {
                                if tier < 3 {
                                    art.primary = i32::from(potion);
                                } else {
                                    art.secondary = i32::from(potion);
                                }
                            }
                            // java: the power-button action clears buttonList, calls initGui, then
                            // updateScreen; this re-arms the one-shot power-button snapshot.
                            self.beacon_power_buttons = None;
                        }
                    }
                    return ScreenAction::None;
                }
            }
        }
        let target = self.target_at(x, y);
        let pick_block = button == self.bindings.pick_block_button;
        let clock = self.clock_ms;
        let touch_action = self.touchscreen_clicked(target, button, pick_block);
        if touch_action == Some(ScreenAction::CloseAll) {
            // java: `GuiContainer.mouseClicked:385-389` closes with `displayGuiScreen(null)`.
            self.closes_without_close_window = true;
            return ScreenAction::CloseAll;
        }
        if touch_action.is_none() {
            let starts_drag = self.drag_preview.is_none()
                && self.container.cursor().is_some()
                && (button == 0 || button == 1 || pick_block)
                && target != SlotTarget::Inside;
            if let Some((slot_id, used_button, mode)) = self.mouse.mouse_clicked(
                &self.container,
                target,
                button,
                pick_block,
                mods.shift,
                clock,
            ) {
                self.dispatch(slot_id, used_button, mode);
            } else if starts_drag {
                self.drag_preview = Some(DragPreview {
                    mode: if button == 0 {
                        0
                    } else if button == 1 {
                        1
                    } else {
                        2
                    },
                    slots: BTreeSet::new(),
                });
            }
            // A 1.7.10 pick-block drag previews with the limit it reuses (`screen_v5.rs`).
            self.sync_drag_preview_v5();
        }
        // java: `GuiRepair.mouseClicked` calls the container implementation first, then gives
        // the same press to its name field. The draw-time caret metrics make this exact without
        // adding a font parameter to the shared screen click trait.
        if let Some(field) = self.anvil_name.as_mut() {
            field.mouse_clicked_cached(x, y, button);
        }

        // java: `GuiEnchantment.mouseClicked` calls `super.mouseClicked` first, then checks the
        // three raw 108x19 plate rectangles. Deliberately no mouse-button filter: vanilla's
        // loop sits outside `GuiContainer`'s left/right/pick-block guard, so any button can
        // activate an eligible offer. `ContainerEnchantment.enchantItem` is a pure predicate on
        // the client; the server owns all item, lapis and XP mutation.
        if let Some(row) = self.enchant_plate_at_position(x, y) {
            let lapis = self.container.slot_stack(1).map_or(0, |stack| stack.count);
            let has_item = self.container.slot_stack(0).is_some();
            let can_activate = match &mut self.art {
                ContainerArt::Enchanting(art) => {
                    art.creative = self.creative;
                    art.can_activate(row, lapis, has_item)
                }
                _ => false,
            };
            // A 1.7.10 table's rule (`screen_v5.rs`).
            let can_activate = self.enchant_gate_v5(row).unwrap_or(can_activate);
            if can_activate {
                self.outgoing_actions
                    .push(OrderedContainerAction::EnchantItem(c2s::EnchantItem {
                        window_id: self.container.window_id,
                        button: row as u8,
                    }));
            }
        }
        ScreenAction::None
    }

    /// [`Screen::mouse_released`] with the real modifier state. See
    /// [`Self::mouse_clicked_with_mods`] — java: `GuiContainer.mouseReleased` (`:515`) reads
    /// `isShiftKeyDown()` too, for the shift-double-click sweep at `:535`.
    pub fn mouse_released_with_mods(&mut self, x: i32, y: i32, button: i32, mods: Modifiers) {
        // NOT VANILLA 1.8.9: see the hotbar block in `mouse_clicked_with_mods`.
        if self.hotbar_mouse_button == Some(button) {
            self.hotbar_mouse_button = None;
            return;
        }
        let target = self.target_at(x, y);
        if self.touchscreen_released(target, button, x, y) {
            return;
        }
        let pick_block = button == self.bindings.pick_block_button;
        let enchanting_input_before = self.container.slot_stack(0).cloned();
        // Unlike the press path, `MouseState::mouse_released` applies each click to the
        // container as it produces it — java calls `handleMouseClick` inline and both the
        // shift-double-click sweep and the method's `lastClickTime` tail read the state those
        // clicks leave behind. See [`super::interaction`]'s module doc.
        // Each click's slot 0 as that click left it, so an anvil can put its
        // `MC|ItemName` ahead of exactly the C0E whose `slotClick` changed slot 0.
        let anvil = self.anvil_name.is_some();
        let mut clicks = Vec::new();
        self.mouse.mouse_released_with(
            &mut self.container,
            target,
            button,
            pick_block,
            mods.shift,
            self.creative,
            &mut |container, packet| {
                let slot0 = anvil.then(|| container.slot_stack(0).cloned()).flatten();
                clicks.push((packet, slot0));
            },
        );
        for (packet, slot0) in clicks {
            if anvil {
                self.detect_anvil_slot_change(slot0);
            }
            self.outgoing_actions
                .push(OrderedContainerAction::ClickWindow(packet));
        }
        if self.container.slot_stack(0) != enchanting_input_before.as_ref() {
            self.clear_invalid_enchanting_offers();
        }
        self.drag_preview = None;
    }

    fn enchantment_level_token(level: i16) -> String {
        match level {
            1 => "I".to_string(),
            2 => "II".to_string(),
            3 => "III".to_string(),
            4 => "IV".to_string(),
            5 => "V".to_string(),
            6 => "VI".to_string(),
            7 => "VII".to_string(),
            8 => "VIII".to_string(),
            9 => "IX".to_string(),
            10 => "X".to_string(),
            _ => format!("enchantment.level.{level}"),
        }
    }

    /// `GuiEnchantment.drawScreen`'s offer tooltip. `I18n` resolves the clue, enchantment,
    /// lapis, and level strings from the active language; the English branches below are only
    /// the no-language fallback used by headless callers.
    fn enchanting_offer_tooltip(&self, row: usize) -> Option<Vec<String>> {
        let ContainerArt::Enchanting(art) = &self.art else {
            return None;
        };
        let level_cost = *art.enchant_levels.get(row)?;
        let clue = *art.enchantment_ids.get(row)?;
        if level_cost <= 0 || clue < 0 {
            return None;
        }

        let mut lines = Vec::new();
        let enchantment_id = (clue & 255) as u8;
        let enchantment_level = ((clue & 65_280) >> 8) as i16;
        if let Some(info) = mc_data::enchantment(enchantment_id) {
            let enchantment_name = self.language.as_ref().map_or_else(
                || {
                    format!(
                        "{} {}",
                        info.display_name,
                        Self::enchantment_level_token(enchantment_level)
                    )
                },
                |language| {
                    let level_key = format!("enchantment.level.{enchantment_level}");
                    format!(
                        "{} {}",
                        language.format(info.name, &[]),
                        language.format(&level_key, &[])
                    )
                },
            );
            let clue = self.language.as_ref().map_or_else(
                || format!("{enchantment_name} . . . ?"),
                |language| language.format("container.enchant.clue", &[&enchantment_name]),
            );
            lines.push(format!("\u{a7}f\u{a7}o{clue}"));
        }

        if !art.creative {
            lines.push(String::new());
            if art.player_level < level_cost {
                lines.push(format!("\u{a7}cLevel Requirement: {level_cost}"));
            } else {
                let payment = row as i32 + 1;
                let lapis = self.container.slot_stack(1).map_or(0, |stack| stack.count);
                let colour = if lapis >= payment { '7' } else { 'c' };
                let payment_text = payment.to_string();
                let lapis_text = self.language.as_ref().map_or_else(
                    || {
                        if payment == 1 {
                            "1 Lapis Lazuli".to_string()
                        } else {
                            format!("{payment} Lapis Lazuli")
                        }
                    },
                    |language| {
                        language.format(
                            if payment == 1 {
                                "container.enchant.lapis.one"
                            } else {
                                "container.enchant.lapis.many"
                            },
                            &[&payment_text],
                        )
                    },
                );
                let levels_text = self.language.as_ref().map_or_else(
                    || {
                        if payment == 1 {
                            "1 Enchantment Level".to_string()
                        } else {
                            format!("{payment} Enchantment Levels")
                        }
                    },
                    |language| {
                        language.format(
                            if payment == 1 {
                                "container.enchant.level.one"
                            } else {
                                "container.enchant.level.many"
                            },
                            &[&payment_text],
                        )
                    },
                );
                lines.push(format!("\u{a7}{colour}{lapis_text}"));
                lines.push(format!("\u{a7}7{levels_text}"));
            }
        }
        Some(lines)
    }

    fn item_tooltip(&self, stack: &ItemStack) -> Vec<String> {
        // C2-13: a shop keeps the cursor on an item, so the same tooltip was rebuilt, recoloured
        // and re-measured every frame. `build_item_tooltip` reads exactly the stack, the
        // advanced flag, the language and the map record, so reuse its result while all four are
        // unchanged (the language through its generation; the rest by value).
        if let Some(cache) = self.tooltip_cache.borrow().as_ref() {
            if cache.advanced == self.advanced_tooltips
                && cache.language_generation == self.language_generation
                && cache.stack == *stack
                && cache.map_data == self.tooltip_map_data
            {
                return cache.lines.clone();
            }
        }
        let lines = self.build_item_tooltip(stack);
        *self.tooltip_cache.borrow_mut() = Some(TooltipCache {
            stack: stack.clone(),
            advanced: self.advanced_tooltips,
            language_generation: self.language_generation,
            map_data: self.tooltip_map_data.clone(),
            lines: lines.clone(),
        });
        lines
    }

    /// The uncached `ItemStack.getTooltip` + rarity recolouring behind [`Self::item_tooltip`].
    fn build_item_tooltip(&self, stack: &ItemStack) -> Vec<String> {
        let lines = self.language.as_ref().map_or_else(
            || {
                tooltip::build_with_map_data(
                    stack,
                    self.advanced_tooltips,
                    self.tooltip_map_data.as_ref(),
                )
            },
            |language| {
                tooltip::build_localized_with_map_data(
                    stack,
                    self.advanced_tooltips,
                    language,
                    self.tooltip_map_data.as_ref(),
                )
            },
        );
        tooltip_box::recolour_tooltip_lines(lines, tooltip::rarity(stack))
    }

    /// The tooltip lines for whatever the cursor is over, or `None`.
    ///
    /// java: the `inventory.getItemStack() == null && theSlot != null && theSlot.getHasStack()`
    /// guard at the end of `drawScreen` — a tooltip never shows while a stack is held.
    #[must_use]
    pub fn hovered_tooltip(&self) -> Option<Vec<String>> {
        if let Some(row) = self.hovered_enchant_offer {
            if let Some(lines) = self.enchanting_offer_tooltip(row) {
                return Some(lines);
            }
        }
        // java: GuiBeacon.drawGuiContainerForegroundLayer asks the first hovered button for
        // its tooltip before GuiContainer's held-stack/item tooltip pass.
        if let Some(lines) = self.beacon_button_tooltip() {
            return Some(lines);
        }
        // java: `GuiMerchant.drawScreen` paints these custom preview tooltips after
        // `GuiContainer.drawScreen`; unlike the base slot-tooltip branch, it does not test the
        // carried cursor stack.
        if let Some(stack) = self.hovered_merchant_preview.as_ref() {
            return Some(self.item_tooltip(stack));
        }
        // `GuiEnchantment.drawScreen` adds its offer tooltip *after* `super.drawScreen`, and
        // unlike the base slot-tooltip branch it has no cursor-stack guard. Therefore this
        // check belongs after the enchanting branch, not before it.
        if self.container.cursor().is_some() {
            return None;
        }
        let id = self.hovered_slot?;
        let stack = self.container.slot_stack(id as usize)?;
        Some(self.item_tooltip(stack))
    }

    /// Returns the filled-map id currently eligible for a slot tooltip at `mouse`.
    ///
    /// The driver asks this after `init` but before drawing, so the live world's `map_<id>`
    /// record reaches the same frame's `ItemMap.addInformation` call.
    #[must_use]
    pub fn hovered_map_id_at(&self, (mouse_x, mouse_y): (i32, i32)) -> Option<i32> {
        if !self.advanced_tooltips || self.container.cursor().is_some() {
            return None;
        }
        self.container
            .slots()
            .iter()
            .find(|slot| self.is_point_in_region(slot.x, slot.y, 16, 16, mouse_x, mouse_y))
            .and_then(|slot| self.container.slot_stack(usize::from(slot.id)))
            .or_else(|| self.merchant_preview_at_position(mouse_x, mouse_y))
            .filter(|stack| {
                stack_ext::try_props(stack)
                    .is_some_and(|props| props.name == "minecraft:filled_map")
            })
            .map(|stack| i32::from(stack.damage))
    }

    /// Draw this screen with the current `map_<id>` record for its eligible hovered stack.
    /// Ordinary [`Screen::draw`] callers keep the no-record branch used by unit tests and
    /// headless UI hosts.
    pub fn draw_with_map_data(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        mouse: (i32, i32),
        map_data: Option<&MapData>,
    ) {
        let current = self
            .tooltip_map_data
            .as_ref()
            .map(|data| (data.id, data.scale));
        let incoming = map_data.map(|data| (data.id, data.scale));
        if current != incoming {
            self.tooltip_map_data = map_data.cloned();
        }
        <Self as Screen>::draw(self, d, font, ctx, mouse);
    }

    fn draw_item_with_count_override(
        &self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        x: i32,
        y: i32,
        stack: &ItemStack,
        override_count: Option<i32>,
    ) {
        self.draw_item_with_count_override_at_z_level(
            d,
            font,
            x,
            y,
            stack,
            override_count,
            render::GUI_SLOT_ITEM_RENDER_Z_LEVEL,
        );
    }

    // Java's render-item seam passes the complete state explicitly; retain that shape so slot,
    // cursor and returning-stack paths keep identical z-level and texture ordering.
    #[allow(clippy::too_many_arguments)]
    fn draw_item_with_count_override_at_z_level(
        &self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        x: i32,
        y: i32,
        stack: &ItemStack,
        override_count: Option<i32>,
        item_render_z_level: f32,
    ) {
        if let Some(count) = override_count {
            // java: GuiContainer.drawSlot always reaches RenderItem's ordinary item-model path;
            // a filled map's live MapData is not consulted for inventory/cursor icons.
            render::draw_item_icon_tinted_at_z_level(
                d,
                &self.atlas,
                x,
                y,
                stack,
                0xFFFF_FFFF,
                item_render_z_level,
            );
            render::draw_durability_bar(d, x, y, stack);
            let text = format!("\u{a7}e{count}");
            render::draw_count_text_with_override(d, font, x, y, stack, Some(&text));
        } else {
            render::draw_item_in_slot_at_z_level(
                d,
                &self.atlas,
                font,
                x,
                y,
                stack,
                item_render_z_level,
            );
        }
    }

    fn draw_merchant_recipe(&self, d: &mut DrawList, font: &mut FontRenderer) {
        let Some(recipe) = (match &self.art {
            ContainerArt::Merchant(merchant) => merchant.selected_recipe(),
            _ => None,
        }) else {
            return;
        };
        self.draw_item_with_count_override(
            d,
            font,
            self.gui_left + 36,
            self.gui_top + 24,
            &recipe.buy_a,
            None,
        );
        if let Some(second) = &recipe.buy_b {
            self.draw_item_with_count_override(
                d,
                font,
                self.gui_left + 62,
                self.gui_top + 24,
                second,
                None,
            );
        }
        self.draw_item_with_count_override(
            d,
            font,
            self.gui_left + 120,
            self.gui_top + 24,
            &recipe.sell,
            None,
        );
    }

    /// Draw the live enchanting-table book through the Java second-projection equivalent.
    ///
    /// The animation inputs come from [`Self::enchanting_book_pose`], while `TileModels` supplies
    /// the existing CPU `ModelBook` geometry. Keeping this call in the screen's background path
    /// places the book after the panel texture and before the offer plates, matching
    /// `GuiEnchantment.drawGuiContainerBackgroundLayer`.
    fn draw_enchanting_book(&mut self, d: &mut DrawList, ctx: &ScreenCtx) {
        let Some(pose) = self.enchanting_book_pose(ctx.partial_ticks) else {
            return;
        };
        let base = enchant_book_screen_model_matrix(pose.spread);
        let mut vertices = Vec::new();
        let texture = self.enchanting_book_models.emit_book_with_age(
            base,
            // java: `GuiEnchantment.java:162` — `MODEL_BOOK.render(null, 0.0F, f3, f4, f2, 0.0F,
            // 0.0625F)`. The GUI's `limbSwing` is a literal 0, so its book never breathes; only
            // the table's TESR passes `tickCount + partialTicks` there.
            0.0,
            (pose.page_flip_right, pose.page_flip_left),
            pose.spread,
            &mut vertices,
        );
        draw_projected_enchanting_book(
            d,
            &UiTexture::gui(texture),
            &vertices,
            ctx.width,
            ctx.height,
        );
    }

    fn drag_preview_for_slot(&self, slot_id: u16) -> Option<(ItemStack, Option<i32>, i32)> {
        let preview = self.drag_preview.as_ref()?;
        if !preview.slots.contains(&slot_id) {
            return None;
        }
        self.container
            .drag_preview_stack(slot_id, preview.mode, preview.slots.len())
    }

    fn drag_preview_remnant(&self) -> Option<i32> {
        let preview = self.drag_preview.as_ref()?;
        let cursor = self.container.cursor()?;
        let mut remaining = cursor.count;
        for slot_id in &preview.slots {
            let (_, _, consumed) =
                self.container
                    .drag_preview_stack(*slot_id, preview.mode, preview.slots.len())?;
            remaining -= consumed;
        }
        Some(remaining)
    }

    fn refresh_drag_preview(&mut self) {
        let Some(preview) = self.drag_preview.as_ref() else {
            return;
        };
        let mode = preview.mode;
        let count = preview.slots.len();
        let ids: Vec<u16> = preview.slots.iter().copied().collect();
        let rejected: Vec<u16> = ids
            .into_iter()
            .filter(|slot_id| {
                self.container
                    .drag_preview_stack(*slot_id, mode, count)
                    .is_none()
            })
            .collect();
        if let Some(preview) = self.drag_preview.as_mut() {
            for slot_id in rejected {
                preview.slots.remove(&slot_id);
            }
        }
    }

    /// Draw one slot: its in-progress drag candidate (when any), placeholder sprite, item and
    /// hover highlight in vanilla's order.
    ///
    /// java: `GuiContainer.drawSlot` (`:277-306`). Two things there that are easy to lose:
    ///
    /// * a non-null `getSlotTexture()` on an **empty** slot blits the placeholder and sets
    ///   `flag1 = true`, which skips the rest of the method — so an empty armour slot gets no
    ///   hover highlight either;
    /// * the highlight (`drawRect(..., -2130706433)`) is drawn **before**
    ///   `renderItemAndEffectIntoGUI`, so it sits *under* the item. Drawing it afterwards
    ///   washes a hovered item out with 50% white.
    fn draw_slot(
        &self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        x: i32,
        y: i32,
        slot: super::slot::Slot,
        hovered: bool,
    ) {
        let preview_member = self
            .drag_preview
            .as_ref()
            .is_some_and(|preview| preview.slots.contains(&slot.id));
        if preview_member
            && self
                .drag_preview
                .as_ref()
                .is_some_and(|preview| preview.slots.len() == 1)
        {
            // java: `if (dragSplittingSlots.size() == 1) return;`
            return;
        }

        let preview = self.drag_preview_for_slot(slot.id);
        let touch_source =
            self.touch.clicked_slot == Some(slot.id) && self.touch.dragged_stack.is_some();
        let touch_hidden = touch_source && !self.touch.right_click;
        let touch_half = (touch_source && self.touch.right_click)
            .then(|| {
                let mut stack = self.container.slot_stack(usize::from(slot.id))?.clone();
                // java: `itemstack.stackSize /= 2` deliberately floors the source half.
                stack.count /= 2;
                Some(stack)
            })
            .flatten();
        let stack = if touch_hidden {
            None
        } else {
            preview
                .as_ref()
                .map(|(stack, _, _)| stack)
                .or(touch_half.as_ref())
                .or_else(|| self.container.slot_stack(usize::from(slot.id)))
        };
        let mut placeholder = false;
        if stack.is_none() && !touch_hidden {
            if let Some(name) = slot.background_sprite {
                let uv = super::slot::background_sprite_uv(name);
                d.quad_rect(
                    &self.atlas,
                    (x as f32, y as f32, (x + 16) as f32, (y + 16) as f32),
                    (uv.min_u, uv.min_v, uv.max_u, uv.max_v),
                    0xFFFF_FFFF,
                );
                placeholder = true; // java: `flag1 = true` inside `drawSlot`
            }
        }
        if preview.is_some() && !placeholder {
            // java: the drag-preview `flag` draws this rect inside `drawSlot`, below the item.
            d.gradient_rect(x, y, x + 16, y + 16, SLOT_HIGHLIGHT, SLOT_HIGHLIGHT);
        }
        if !placeholder && !touch_hidden {
            if let Some(stack) = stack {
                let override_count = preview.as_ref().and_then(|(_, count, _)| *count);
                self.draw_item_with_count_override(d, font, x, y, stack, override_count);
            }
        }
        if hovered {
            // This highlight is *outside* Java's `drawSlot`: `drawScreen` calls `drawSlot`
            // first and then paints the hovered rectangle, including over placeholder sprites.
            d.gradient_rect(x, y, x + 16, y + 16, SLOT_HIGHLIGHT, SLOT_HIGHLIGHT);
        }
    }
}

impl Screen for ContainerScreen {
    fn name(&self) -> &'static str {
        self.art.screen_name()
    }

    // java: GuiContainer.java:88 (initGui)
    fn init(&mut self, ctx: &ScreenCtx) {
        self.screen_width = ctx.width;
        self.init_shift_v5.laid_out(ctx.width, ctx.height);
        self.update_active_potion_effects();
        self.gui_top = (ctx.height - self.art.y_size()) / 2;
        if matches!(&self.art, ContainerArt::Beacon(_)) {
            // java: `GuiBeacon.initGui` sets `buttonsNotDrawn = true`; the next update rebuilds
            // against the then-current fields, including after a resize or local power press.
            self.beacon_power_buttons = None;
        }
        if let Some(field) = self.anvil_name.as_mut() {
            // java: `new GuiTextField(..., guiLeft + 62, guiTop + 24, 103, 12)`.
            field.rect.x = self.gui_left + 62;
            field.rect.y = self.gui_top + 24;
            field.rect.width = 103;
            field.rect.height = 12;
        }
    }

    fn tick(&mut self) {
        self.clock_ms = self.clock_ms.saturating_add(TICK_MS);
        self.preview_age_ticks = self.preview_age_ticks.wrapping_add(1);
        self.update_active_potion_effects();
        if let ContainerArt::Enchanting(art) = &self.art {
            // java: GuiEnchantment.updateScreen -> func_147068_g. The model update belongs to
            // the screen tick, while the tile/book renderer only consumes the resulting pose.
            self.enchanting_book
                .tick(self.container.slot_stack(0), &art.enchant_levels);
        }
        if let Some(field) = self.anvil_name.as_mut() {
            field.tick();
        }
    }

    // java: GuiContainer.java:99 (drawScreen)
    fn draw(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        (mouse_x, mouse_y): (i32, i32),
    ) {
        self.refresh_invalid_enchanting_offers();
        d.gradient_rect(0, 0, ctx.width, ctx.height, WORLD_DIM_TOP, WORLD_DIM_BOTTOM);
        self.container.set_creative_mode(self.creative);
        // java reads `mc.thePlayer.capabilities.isCreativeMode` afresh at each use site
        // (`GuiEnchantment.drawGuiContainerForegroundLayer`, `GuiRepair.drawGuiContainer-
        // ForegroundLayer`). Here the flag is owned by the screen and the host assigns the
        // public field directly, so the art's own copy -- which has no other writer anywhere in
        // the workspace -- has to be refreshed from it before it is read.
        match &mut self.art {
            ContainerArt::Enchanting(e) => e.creative = self.creative,
            ContainerArt::Anvil(a) => {
                a.creative = self.creative;
                if let Some(cost) = self.container.maximum_cost {
                    a.maximum_cost = cost;
                }
            }
            ContainerArt::Furnace(_)
            | ContainerArt::BrewingStand(_)
            | ContainerArt::Beacon(_)
            | ContainerArt::Player
            | ContainerArt::Crafting
            | ContainerArt::Chest { .. }
            | ContainerArt::Hopper
            | ContainerArt::Dispenser
            | ContainerArt::Merchant(_)
            | ContainerArt::Horse { .. } => {}
        }
        let hovered_enchant_plate = self.enchant_plate_at_position(mouse_x, mouse_y);
        self.art.draw_background_before_enchanting_book(
            d,
            self.gui_left,
            self.gui_top,
            &self.container,
            hovered_enchant_plate,
        );
        if let Some(horse) = self.horse_preview_snapshot.as_ref() {
            if matches!(self.art, ContainerArt::Horse { .. }) && horse.state.horse_type == 0 {
                // java: GuiScreenHorseInventory's armor slot background is drawn only when
                // EntityHorse.canWearArmor() is true (ordinary horses, not donkeys/llamas).
                let tex = self.art.texture();
                let h = self.art.y_size();
                d.blit(
                    &tex,
                    self.gui_left + 7,
                    self.gui_top + 35,
                    0,
                    h + 54,
                    18,
                    18,
                );
            }
        }
        self.draw_enchanting_book(d, ctx);
        self.art.draw_enchanting_offers(
            d,
            self.gui_left,
            self.gui_top,
            &self.container,
            hovered_enchant_plate,
            Some(&mut self.enchanting_runes),
        );
        self.draw_beacon_payment_items(d);
        self.draw_beacon_buttons(d, (mouse_x, mouse_y));

        self.refresh_drag_preview();
        self.hovered_enchant_offer = match &self.art {
            ContainerArt::Enchanting(art) => (0..3usize).find(|row| {
                art.enchant_levels[*row] > 0
                    && art.enchantment_ids[*row] >= 0
                    && self.is_point_in_region(60, 14 + 19 * *row as i32, 108, 17, mouse_x, mouse_y)
            }),
            _ => None,
        };

        // java: `GuiInventory.drawGuiContainerBackgroundLayer` — drawn right after the panel
        // texture, before any slot. `oldMouseX`/`oldMouseY` (not this frame's `mouse_x`/
        // `mouse_y`) is vanilla's own one-frame-stale read; see `preview_prev_mouse`'s doc.
        if self.art == ContainerArt::Player {
            let (px, py) = (self.gui_left + 51, self.gui_top + 75);
            let (old_x, old_y) = self.preview_prev_mouse;
            self.player_preview.draw_with_snapshot(
                d,
                px,
                py,
                30,
                (px - old_x) as f32,
                (py - 50 - old_y) as f32,
                self.player_skin,
                self.preview_age_ticks as f32 + ctx.partial_ticks,
                self.player_skin_texture.as_ref(),
                &self.player_preview_snapshot,
                &self.atlas,
            );
        }

        if matches!(self.art, ContainerArt::Horse { .. }) {
            if let Some(horse) = self.horse_preview_snapshot.as_ref() {
                let (px, py) = (self.gui_left + 51, self.gui_top + 60);
                self.player_preview.draw_horse(
                    d,
                    px,
                    py,
                    17,
                    (px - mouse_x) as f32,
                    (py - 50 - mouse_y) as f32,
                    self.preview_age_ticks as f32 + ctx.partial_ticks,
                    horse,
                );
            }
        }

        let mut hovered = None;
        for index in 0..self.container.slots().len() {
            let slot = self.container.slots()[index];
            let is_hovered = hovered.is_none()
                && slot.can_be_hovered()
                && self.is_point_in_region(slot.x, slot.y, 16, 16, mouse_x, mouse_y);
            if is_hovered {
                hovered = Some(slot.id);
            }
            let (x, y) = (self.gui_left + slot.x, self.gui_top + slot.y);
            self.draw_slot(d, font, x, y, slot, is_hovered);
        }
        self.hovered_slot = hovered;
        self.hovered_merchant_preview =
            self.merchant_preview_at_position(mouse_x, mouse_y).cloned();
        self.draw_merchant_recipe(d, font);

        self.art.draw_foreground(
            d,
            font,
            self.gui_left,
            self.gui_top,
            &self.container,
            &self.title,
            &self.player_title,
            self.language.as_ref(),
        );

        // java: `draggedStack` replaces the ordinary cursor while a touchscreen gesture is live.
        // It sits eight pixels lower, and a right-click gesture displays the ceil half while the
        // source slot displays the floor half in `draw_slot` above.
        let touch_dragged = self.touch.dragged_stack.as_ref();
        if let Some(stack) = touch_dragged.or_else(|| self.container.cursor()) {
            let mut stack: ItemStack = stack.clone();
            let override_count = if touch_dragged.is_some() && self.touch.right_click {
                stack.count = (stack.count + 1) / 2;
                None
            } else if touch_dragged.is_none()
                && self
                    .drag_preview
                    .as_ref()
                    .is_some_and(|preview| preview.slots.len() > 1)
            {
                self.drag_preview_remnant().map(|remaining| {
                    // java: `itemstack.stackSize =
                    // dragSplittingRemnant`, unclamped. A full-stack (middle-button) sweep can
                    // ask for more than is held, and `renderItemOverlayIntoGUI` then prints the
                    // negative size in red, not a red 0.
                    stack.count = remaining;
                    (remaining == 0).then_some(0)
                })
            } else {
                None
            }
            .flatten();
            self.draw_item_with_count_override_at_z_level(
                d,
                font,
                mouse_x - 8,
                mouse_y - if touch_dragged.is_some() { 16 } else { 8 },
                &stack,
                override_count,
                render::GUI_CURSOR_ITEM_RENDER_Z_LEVEL,
            );
        }

        // java: `returningStack` is painted after the held stack and before the tooltip. The
        // interpolation coordinates are absolute here because DrawList has no translated matrix.
        if let Some((stack, x, y)) = self.returning_stack_position() {
            self.draw_item_with_count_override_at_z_level(
                d,
                font,
                x,
                y,
                &stack,
                None,
                render::GUI_CURSOR_ITEM_RENDER_Z_LEVEL,
            );
        }

        // Through the `scrollable_tooltips` mod: exactly `tooltip_box::draw`
        // while it is off.
        match self.hovered_tooltip() {
            Some(lines) => self
                .tooltip_scroll
                .draw(d, font, &lines, mouse_x, mouse_y, ctx.width, ctx.height),
            None => self.tooltip_scroll.clear(),
        }

        // java: `GuiRepair.drawScreen` draws the field after `super.drawScreen`, including after
        // the held-stack tooltip. Its background is disabled, so the anvil texture supplies the
        // plate while the widget supplies text, selection and caret.
        if let Some(field) = self.anvil_name.as_mut() {
            field.draw(d, font);
        }

        // java call order: `InventoryEffectRenderer.drawScreen` invokes
        // `GuiContainer.drawScreen` first, then paints this list. `GuiInventory.drawScreen`
        // updates `oldMouseX/Y` only after both have returned.
        self.draw_active_potion_effects(d, font);

        // java: `GuiInventory.drawScreen` sets `oldMouseX`/`oldMouseY` *after* `super.drawScreen`
        // (everything above) has already run, so this frame's preview read last frame's mouse —
        // reproduced by updating only now, at the very end of this frame's own draw.
        self.preview_prev_mouse = (mouse_x, mouse_y);
    }

    // java: GuiContainer.java:359 (mouseClicked)
    fn mouse_clicked(&mut self, x: i32, y: i32, button: i32) -> ScreenAction {
        self.mouse_clicked_with_mods(x, y, button, Modifiers::NONE)
    }

    // java: GuiContainer.java:511 (mouseReleased)
    fn mouse_released(&mut self, x: i32, y: i32, button: i32) {
        self.mouse_released_with_mods(x, y, button, Modifiers::NONE);
    }

    // java: GuiContainer.java:465 (mouseClickMove)
    fn mouse_dragged(&mut self, x: i32, y: i32, _button: i32, _held_ms: u64) {
        let target = self.target_at(x, y);
        if self.touchscreen_dragged(target, _button) {
            return;
        }
        self.mouse.drag_move(&self.container, target);
        let Some(slot_id) = (match target {
            SlotTarget::Slot(slot_id) => Some(slot_id),
            SlotTarget::Inside | SlotTarget::Outside => None,
        }) else {
            return;
        };
        let Some(cursor) = self.container.cursor() else {
            return;
        };
        let Some(preview) = self.drag_preview.as_mut() else {
            return;
        };
        if cursor.count > preview.slots.len() as i32
            && self.container.slot_accepts_for_drag(slot_id, cursor)
        {
            preview.slots.insert(slot_id);
        }
    }

    // java: GuiContainer.java:686 (keyTyped)
    fn key_typed(&mut self, _typed: Option<char>, key: i32, mods: Modifiers) -> ScreenAction {
        // java: `closeScreen()` does not return from `keyTyped`; the hotbar and slot-action
        // checks below still run when a close key deliberately collides with one of those
        // bindings. Keep the close action until after those producers so the caller applies one
        // `Pop` after every packet this key event generated, rather than closing twice or
        // dropping the later producer.
        let closes = key == keys::ESCAPE || key == self.bindings.inventory_key;

        // java: `checkHotbarKeys` runs first and only fires with an empty cursor. Its boolean
        // result is intentionally ignored by `keyTyped`, so a colliding drop/pick binding still
        // gets its own subsequent check below.
        self.check_hotbar_keys(key);
        if let Some(slot) = self.hovered_slot {
            if self.container.slot_stack(slot as usize).is_some() {
                // This is a **keyboard** comparison and must not reuse
                // [`ContainerBindings::pick_block_button`], which is a mouse button index. With
                // the vanilla default binding `pick_block_key` is `-98` and this branch never
                // fires, exactly as in java.
                if key == self.bindings.pick_block_key {
                    self.dispatch(i32::from(slot), 0, 3);
                } else if key == self.bindings.drop_key {
                    self.dispatch(i32::from(slot), u8::from(mods.ctrl), 4);
                }
            }
        }

        if closes {
            ScreenAction::Pop
        } else {
            ScreenAction::None
        }
    }

    fn key_typed_with_font(
        &mut self,
        typed: Option<char>,
        key: i32,
        mods: Modifiers,
        font: &FontRenderer,
    ) -> ScreenAction {
        let consumed = self
            .anvil_name
            .as_mut()
            .is_some_and(|field| field.key_typed(typed, key, mods, &mut self.clipboard, font));
        if consumed {
            if let (Some(field), ContainerArt::Anvil(art)) = (&self.anvil_name, &mut self.art) {
                art.rename_text = field.text();
            }
            // java calls `renameItem()` whenever `textboxKeyTyped` returns true, including
            // navigation and clipboard keys whose accepted action did not change the string.
            self.queue_anvil_name();
            ScreenAction::None
        } else {
            self.key_typed(typed, key, mods)
        }
    }

    /// java: `GuiContainer.doesGuiPauseGame()` returns `false` — and the multiplayer-only rule means there is
    /// no integrated server to pause anyway.
    fn pauses_game(&self) -> bool {
        false
    }

    fn allows_user_input(&self) -> bool {
        self.art.allows_user_input()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::font::FontTexture;
    use crate::widget::test_support::{uniform_font, uniform_font_with_texture};
    use crate::widget::ScaledResolution;

    fn ctx() -> ScreenCtx {
        ScreenCtx::new(ScaledResolution::new(427, 240, 1, false), 0.0)
    }

    fn stack(name: &str, count: i32) -> ItemStack {
        let id = mc_data::item_by_name(name).expect("known item");
        ItemStack {
            item_id: id.0 as i16,
            count,
            damage: 0,
            nbt: None,
        }
    }

    fn opened(mut s: ContainerScreen) -> ContainerScreen {
        s.init(&ctx());
        s
    }

    #[test]
    fn gui_left_and_top_are_vanillas_centering_arithmetic() {
        let s = opened(ContainerScreen::chest(1, 3, "Chest"));
        assert_eq!(s.art.x_size(), 176);
        assert_eq!(s.art.y_size(), 114 + 3 * 18);
        assert_eq!(s.gui_left(), (427 - 176) / 2);
        assert_eq!(s.gui_top(), (240 - 168) / 2);
    }

    #[test]
    fn player_inventory_forwards_and_clears_the_ready_skin_texture() {
        let custom = UiTexture::Resource(mc_assets::ResourceLocation::minecraft(
            "textures/entity/skin/player/profile-2.png",
        ));
        let mut screen = opened(ContainerScreen::player(0));
        screen.set_player_preview_texture(Some(custom.clone()));

        let mut ready = DrawList::new();
        screen.draw(&mut ready, &mut uniform_font(), &ctx(), (0, 0));
        assert!(ready.batches().iter().any(|batch| batch.texture == custom));

        screen.set_player_preview_texture(None);
        let mut fallback = DrawList::new();
        screen.draw(&mut fallback, &mut uniform_font(), &ctx(), (0, 0));
        assert!(!fallback
            .batches()
            .iter()
            .any(|batch| batch.texture == custom));
        assert!(fallback
            .batches()
            .iter()
            .any(|batch| { batch.texture == UiTexture::gui("textures/entity/steve.png") }));
    }

    fn effects(count: usize) -> Vec<ActiveEffect> {
        (0..count)
            .map(|index| ActiveEffect {
                icon_index: -1,
                name: format!("Effect {index}"),
                duration: "1:23".to_owned(),
                beneficial: true,
            })
            .collect()
    }

    /// java: `InventoryEffectRenderer.updateActivePotionEffects` runs at init and from
    /// `GuiInventory.updateScreen` every tick. The non-player control proves this is not a
    /// blanket shift applied to every `GuiContainer` subclass.
    #[test]
    fn active_effects_shift_only_the_player_inventory_and_track_live_changes() {
        let mut player = ContainerScreen::player(0);
        player.set_active_effects(effects(1));
        player.init(&ctx());
        assert_eq!(player.gui_left(), 160 + (427 - 176 - 200) / 2);

        player.set_active_effects(Vec::new());
        player.tick();
        assert_eq!(player.gui_left(), (427 - 176) / 2);

        player.set_active_effects(effects(1));
        player.tick();
        assert_eq!(player.gui_left(), 160 + (427 - 176 - 200) / 2);

        let mut chest = ContainerScreen::chest(1, 3, "Chest");
        chest.set_active_effects(effects(1));
        chest.init(&ctx());
        assert_eq!(chest.gui_left(), (427 - 176) / 2);
    }

    /// `centered_inventory`: with the mod's layout the player inventory stays at
    /// vanilla's plain centring whatever the effects, the panels still sit at `guiLeft - 124`
    /// unless hidden, and `(false, true)` restores vanilla's shift.
    #[test]
    fn the_centered_inventory_layout_keeps_gui_left_centred_and_can_hide_the_panels() {
        let centred = (427 - 176) / 2;
        let shifted = 160 + (427 - 176 - 200) / 2;
        let mut player = ContainerScreen::player(0);
        player.set_active_effects(effects(2));
        player.init(&ctx());
        assert_eq!(player.gui_left(), shifted);
        player.set_effect_layout(true, true);
        assert_eq!(player.gui_left(), centred, "applied at once");
        player.tick();
        player.set_active_effects(effects(3));
        player.init(&ctx());
        assert_eq!(player.gui_left(), centred, "and by every later update");

        let plates = |screen: &ContainerScreen| {
            let mut draw = DrawList::new();
            screen.draw_active_potion_effects(&mut draw, &mut uniform_font());
            let texture = UiTexture::gui(INVENTORY_TEXTURE);
            draw.batches()
                .iter()
                .filter(|batch| batch.texture == texture)
                .map(|batch| draw.vertices()[batch.range.start].pos)
                .collect::<Vec<_>>()
        };
        let shown = plates(&player);
        assert_eq!(shown.len(), 3);
        assert_eq!(shown[0], [(centred - 124) as f32, player.gui_top() as f32]);
        player.set_effect_layout(true, false);
        assert!(plates(&player).is_empty(), "show effects off: no panels");
        player.set_effect_layout(false, true);
        assert_eq!(player.gui_left(), shifted, "vanilla again");
        assert_eq!(plates(&player).len(), 3);
    }

    /// java: `InventoryEffectRenderer.drawActivePotionEffects` starts row zero at `guiTop`,
    /// uses 33px through five effects, then uses integer pitch `132 / (count - 1)` (so the
    /// covered distance can be just under 132px after truncation). Inspecting the actual
    /// inventory-texture batches means the assertion covers rendered plates rather than only a
    /// layout helper.
    #[test]
    fn active_effect_plates_use_vanillas_origin_and_compressed_pitch() {
        fn plate_origins(count: usize) -> (ContainerScreen, Vec<[f32; 2]>) {
            let mut screen = ContainerScreen::player(0);
            screen.set_active_effects(effects(count));
            screen.init(&ctx());
            let mut draw = DrawList::new();
            screen.draw_active_potion_effects(&mut draw, &mut uniform_font());
            let texture = UiTexture::gui(INVENTORY_TEXTURE);
            let origins = draw
                .batches()
                .iter()
                .filter(|batch| batch.texture == texture)
                .map(|batch| draw.vertices()[batch.range.start].pos)
                .collect();
            (screen, origins)
        }

        let (five, five_rows) = plate_origins(5);
        assert_eq!(five_rows.len(), 5);
        assert_eq!(
            five_rows[0],
            [(five.gui_left() - 124) as f32, five.gui_top() as f32]
        );
        assert_eq!(five_rows[4][1], (five.gui_top() + 4 * 33) as f32);

        let (six, six_rows) = plate_origins(6);
        assert_eq!(six_rows.len(), 6);
        assert_eq!(
            six_rows[0],
            [(six.gui_left() - 124) as f32, six.gui_top() as f32]
        );
        assert_eq!(six_rows[1][1], (six.gui_top() + 132 / 5) as f32);
        assert_eq!(six_rows[5][1], (six.gui_top() + 5 * (132 / 5)) as f32);
    }

    #[test]
    fn every_kind_reports_vanillas_own_panel_size() {
        let cases: &[(ContainerArt, i32, i32)] = &[
            (ContainerArt::Player, 176, 166),
            (ContainerArt::Chest { rows: 3 }, 176, 168),
            (ContainerArt::Chest { rows: 6 }, 176, 222),
            (ContainerArt::Hopper, 176, 133),
            (ContainerArt::Dispenser, 176, 166),
            (ContainerArt::Furnace(FurnaceArt::default()), 176, 166),
            (
                ContainerArt::BrewingStand(BrewingStandArt::default()),
                176,
                166,
            ),
            (ContainerArt::Enchanting(EnchantingArt::default()), 176, 166),
            (ContainerArt::Anvil(AnvilArt::default()), 176, 166),
            (ContainerArt::Beacon(BeaconArt::default()), 230, 219),
            (ContainerArt::Merchant(MerchantArt::default()), 176, 166),
            (ContainerArt::Horse { chested: false }, 176, 166),
        ];
        for (art, w, h) in cases {
            assert_eq!((art.x_size(), art.y_size()), (*w, *h), "{art:?}");
        }
    }

    /// The `- 1`/`+ 1` in `isPointInRegion` is not a rounding convenience: it makes every slot's
    /// hit box 18x18 and is why the two-pixel gutter between adjacent slots is still clickable.
    #[test]
    fn a_slots_hit_box_is_eighteen_wide_not_sixteen() {
        let s = opened(ContainerScreen::chest(1, 3, "Chest"));
        let slot = s.container.slots()[0];
        let (bx, by) = (s.gui_left() + slot.x, s.gui_top() + slot.y);
        assert!(s.is_point_in_region(slot.x, slot.y, 16, 16, bx - 1, by - 1));
        assert!(s.is_point_in_region(slot.x, slot.y, 16, 16, bx + 16, by + 16));
        assert!(!s.is_point_in_region(slot.x, slot.y, 16, 16, bx - 2, by));
        assert!(!s.is_point_in_region(slot.x, slot.y, 16, 16, bx + 17, by));
    }

    #[test]
    fn slot_hit_testing_finds_the_slot_under_the_cursor() {
        let s = opened(ContainerScreen::chest(1, 3, "Chest"));
        let slot = s.container.slots()[13];
        let (x, y) = (s.gui_left() + slot.x + 8, s.gui_top() + slot.y + 8);
        assert_eq!(s.slot_at_position(x, y), Some(13));
        assert_eq!(s.target_at(x, y), SlotTarget::Slot(13));
    }

    #[test]
    fn a_point_on_the_panel_art_is_inside_but_on_no_slot() {
        let s = opened(ContainerScreen::chest(1, 3, "Chest"));
        // The 4-pixel margin at the panel's very top-left is artwork, never a slot.
        assert_eq!(
            s.target_at(s.gui_left() + 1, s.gui_top() + 1),
            SlotTarget::Inside
        );
    }

    #[test]
    fn a_point_off_the_panel_is_the_outside_sentinel() {
        let s = opened(ContainerScreen::chest(1, 3, "Chest"));
        assert_eq!(
            s.target_at(s.gui_left() - 1, s.gui_top() + 20),
            SlotTarget::Outside
        );
        assert_eq!(s.target_at(0, 0), SlotTarget::Outside);
    }

    /// The two rules interact: a slot whose expanded hit box pokes outside the panel is still
    /// "outside" for click purposes, because vanilla overwrites `l` with `-999` afterwards.
    #[test]
    fn outside_beats_a_slot_whose_expanded_hit_box_pokes_over_the_panel_edge() {
        let s = opened(ContainerScreen::hopper(1, 5, "Item Hopper"));
        // The hopper's hotbar row sits at y = 109..125, and ySize is 133 — so no slot reaches
        // the bottom edge. Use the left edge instead: slot x = 8, so x - 1 = 7 is still on the
        // panel. Build the case artificially at the very top-left corner of the panel instead.
        let slot = s.container.slots()[5]; // first player-inventory slot, x = 8
        let x = s.gui_left() + slot.x - 1;
        assert!(s.is_point_in_region(slot.x, slot.y, 16, 16, x, s.gui_top() + slot.y));
        assert!(!s.is_outside_window(x, s.gui_top() + slot.y));
        // ...and one pixel further left is off the panel, where the slot check is discarded.
        assert!(s.is_outside_window(s.gui_left() - 1, s.gui_top() + slot.y));
        assert_eq!(
            s.target_at(s.gui_left() - 1, s.gui_top() + slot.y),
            SlotTarget::Outside
        );
    }

    #[test]
    fn clicking_a_stacked_slot_queues_one_packet_and_picks_the_stack_up() {
        let mut s = opened(ContainerScreen::chest(1, 3, "Chest"));
        s.container.set_slot_contents(0, Some(stack("stone", 10)));
        let slot = s.container.slots()[0];
        let (x, y) = (s.gui_left() + slot.x + 8, s.gui_top() + slot.y + 8);
        s.mouse_clicked(x, y, 0);
        let out = s.take_outgoing();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].slot_id, 0);
        assert_eq!(out[0].mode, 0);
        assert_eq!(out[0].window_id, 1);
        assert_eq!(s.container.cursor().map(|c| c.count), Some(10));
        assert!(s.take_outgoing().is_empty(), "draining is destructive");
    }

    /// java: `GuiContainer.mouseReleased`'s touchscreen direct-replacement arm. A right press
    /// takes half of eight, while the empty armour destination passes
    /// `Container.canAddItemToSlot`'s compatibility-only test but rejects the real placement.
    /// The final right click can return only one item to the source, so three remain on the
    /// cursor and start `returningStack`. A normal empty inventory target would clear the cursor
    /// and (correctly) produce no animation.
    #[test]
    fn touchscreen_direct_replacement_returns_the_remainder_for_exactly_one_hundred_ms() {
        let mut s = opened(ContainerScreen::player(0));
        s.set_touchscreen(true);
        s.set_clock_ms(1_000);
        // Player-container ids 5..=8 are armour; 9 is the first ordinary inventory slot.
        let source_slot = 9usize;
        let destination_slot = 5usize;
        s.container
            .set_slot_contents(source_slot, Some(stack("stone", 8)));
        let point = |screen: &ContainerScreen, id: usize| {
            let slot = screen.container.slots()[id];
            (
                screen.gui_left() + slot.x + 8,
                screen.gui_top() + slot.y + 8,
            )
        };
        let source = point(&s, source_slot);
        let release = point(&s, destination_slot);

        assert_eq!(
            s.mouse_clicked_with_mods(source.0, source.1, 1, Modifiers::NONE),
            ScreenAction::None
        );
        assert!(s.take_outgoing().is_empty(), "touch press sends no C0E");
        s.mouse_released_with_mods(release.0, release.1, 1, Modifiers::NONE);

        let packets = s.take_outgoing();
        assert_eq!(packets.len(), 3, "source, destination, source");
        assert_eq!(
            s.container.slot_stack(source_slot).map(|stack| stack.count),
            Some(5)
        );
        assert_eq!(s.container.slot_stack(destination_slot), None);
        assert_eq!(s.container.cursor().map(|stack| stack.count), Some(3));
        let returning = s
            .returning_stack
            .as_ref()
            .expect("cursor remainder flies back");
        assert_eq!(
            returning.stack.count, 8,
            "Java retains the dragged snapshot"
        );
        assert_eq!(returning.start, release);
        assert_eq!(returning.destination_slot, source_slot as u16);

        let destination_slot = s.container.slots()[source_slot];
        let destination = (
            s.gui_left() + destination_slot.x,
            s.gui_top() + destination_slot.y,
        );
        assert_eq!(
            s.returning_stack_position().map(|(_, x, y)| (x, y)),
            Some(release)
        );
        s.set_clock_ms(1_050);
        assert_eq!(
            s.returning_stack_position().map(|(_, x, y)| (x, y)),
            Some((
                release.0 + (destination.0 - release.0) / 2,
                release.1 + (destination.1 - release.1) / 2,
            ))
        );
        s.set_clock_ms(1_099);
        assert!(s.returning_stack_position().is_some());
        s.set_clock_ms(1_100);
        assert_eq!(s.returning_stack_position(), None);
        assert_eq!(s.returning_stack, None);
    }

    #[test]
    fn touchscreen_invalid_target_returns_without_packets_while_desktop_never_animates() {
        let prepare = || {
            let mut screen = opened(ContainerScreen::chest(1, 3, "Chest"));
            screen
                .container
                .set_slot_contents(0, Some(stack("stone", 8)));
            screen
                .container
                .set_slot_contents(1, Some(stack("dirt", 1)));
            screen
        };
        let point = |screen: &ContainerScreen, id: usize| {
            let slot = screen.container.slots()[id];
            (
                screen.gui_left() + slot.x + 8,
                screen.gui_top() + slot.y + 8,
            )
        };

        let mut touch = prepare();
        touch.set_touchscreen(true);
        let source = point(&touch, 0);
        let incompatible = point(&touch, 1);
        touch.mouse_clicked(source.0, source.1, 0);
        touch.mouse_dragged(incompatible.0, incompatible.1, 0, 0);
        assert_eq!(
            touch.touch.dragged_stack.as_ref().map(|stack| stack.count),
            Some(8)
        );
        touch.mouse_released(incompatible.0, incompatible.1, 0);
        assert!(touch.take_outgoing().is_empty());
        assert_eq!(
            touch
                .returning_stack
                .as_ref()
                .map(|value| value.stack.count),
            Some(8)
        );
        assert_eq!(
            touch.container.slot_stack(0).map(|stack| stack.count),
            Some(8)
        );
        assert_eq!(
            touch.container.slot_stack(1).map(|stack| stack.count),
            Some(1)
        );

        let mut desktop = prepare();
        let source = point(&desktop, 0);
        desktop.mouse_clicked(source.0, source.1, 0);
        desktop.mouse_released(source.0, source.1, 0);
        assert_eq!(desktop.take_outgoing().len(), 1);
        assert_eq!(desktop.returning_stack, None);
    }

    #[test]
    fn a_press_on_the_panel_art_queues_nothing() {
        let mut s = opened(ContainerScreen::chest(1, 3, "Chest"));
        s.mouse_clicked(s.gui_left() + 1, s.gui_top() + 1, 0);
        assert!(s.take_outgoing().is_empty());
    }

    #[test]
    fn a_full_drag_across_three_slots_produces_start_three_adds_and_an_end() {
        let mut s = opened(ContainerScreen::chest(1, 3, "Chest"));
        s.container.set_cursor(Some(stack("stone", 9)));
        let point = |s: &ContainerScreen, id: usize| {
            let slot = s.container.slots()[id];
            (s.gui_left() + slot.x + 8, s.gui_top() + slot.y + 8)
        };
        let (x0, y0) = point(&s, 0);
        s.mouse_clicked(x0, y0, 0);
        assert!(
            s.take_outgoing().is_empty(),
            "a drag sends nothing on press"
        );
        for id in [0usize, 1, 2] {
            let (x, y) = point(&s, id);
            s.mouse_dragged(x, y, 0, 10);
        }
        let (x2, y2) = point(&s, 2);
        s.mouse_released(x2, y2, 0);
        let out = s.take_outgoing();
        assert_eq!(out.len(), 5, "start + three adds + end: {out:?}");
        assert_eq!((out[0].slot_id, out[0].mode), (-999, 5));
        assert_eq!((out[4].slot_id, out[4].mode), (-999, 5));
        assert_eq!(s.container.slot_stack(0).map(|s| s.count), Some(3));
        assert_eq!(s.container.slot_stack(1).map(|s| s.count), Some(3));
        assert_eq!(s.container.slot_stack(2).map(|s| s.count), Some(3));
        assert_eq!(s.container.cursor(), None);
    }

    /// java: `GuiContainer.drawSlot:236-270` and `updateDragSplitting:309-336`. Before the
    /// preview port, every assertion below had the wire state available but `draw_slot` still
    /// emitted nothing for the swept empty cells.
    #[test]
    fn a_drag_previews_the_split_in_each_swept_slot_and_shrinks_the_cursor() {
        let mut s = opened(ContainerScreen::chest(1, 3, "Chest"));
        s.container.set_cursor(Some(stack("stone", 10)));
        let point = |s: &ContainerScreen, id: usize| {
            let slot = s.container.slots()[id];
            (s.gui_left() + slot.x + 8, s.gui_top() + slot.y + 8)
        };

        let (x0, y0) = point(&s, 0);
        s.mouse_clicked(x0, y0, 0);
        s.mouse_dragged(x0, y0, 0, 10);
        let slot0 = s.container.slots()[0];
        let mut one = DrawList::new();
        s.draw_slot(&mut one, &mut uniform_font(), 0, 0, slot0, false);
        assert!(
            one.vertices().is_empty(),
            "vanilla returns early while the drag set contains exactly one slot"
        );

        let (x1, y1) = point(&s, 1);
        s.mouse_dragged(x1, y1, 0, 10);
        let (preview, override_count, consumed) = s.drag_preview_for_slot(0).unwrap();
        assert_eq!(preview.count, 5);
        assert_eq!(override_count, None);
        assert_eq!(consumed, 5);
        assert_eq!(s.drag_preview_remnant(), Some(0));

        let mut drawn = DrawList::new();
        s.draw_slot(&mut drawn, &mut uniform_font(), 0, 0, slot0, false);
        assert!(drawn.vertices().len() > 6, "preview shade plus count text");
        assert!(drawn.vertices()[..6]
            .iter()
            .all(|vertex| { (vertex.color[3] - 128.0 / 255.0).abs() < 1e-3 }));

        // Negative control: the same empty slot outside a drag draws nothing.
        s.drag_preview = None;
        let mut plain = DrawList::new();
        s.draw_slot(&mut plain, &mut uniform_font(), 0, 0, slot0, false);
        assert!(plain.vertices().is_empty());
    }

    /// The computed amount can exceed both the item's own maximum and a slot's limit. Vanilla
    /// clips it and supplies the clipped number as yellow override text instead of silently
    /// drawing the ordinary count.
    #[test]
    fn a_drag_preview_reports_the_yellow_count_override_when_a_slot_caps_it() {
        let mut s = opened(ContainerScreen::chest(1, 3, "Chest"));
        s.container.set_slot_contents(0, Some(stack("stone", 63)));
        s.container.set_cursor(Some(stack("stone", 64)));
        s.drag_preview = Some(DragPreview {
            mode: 0,
            slots: [0, 1].into_iter().collect(),
        });
        let (preview, override_count, consumed) = s.drag_preview_for_slot(0).unwrap();
        assert_eq!(preview.count, 64);
        assert_eq!(override_count, Some(64));
        assert_eq!(consumed, 1);
        assert_eq!(s.drag_preview_remnant(), Some(31));
    }

    /// java: `GuiContainer.drawScreen:158-167` sets the held stack's
    /// size to `dragSplittingRemnant` unclamped. A middle-button (full-stack) sweep of 5 over
    /// three slots previews 64 each, so the remnant is 5 - 192 = -187, which
    /// `renderItemOverlayIntoGUI` prints in red as "-187" (four glyphs), not as a red "0".
    #[test]
    fn a_full_stack_sweep_shows_the_negative_remnant_in_red() {
        let mut s = opened(ContainerScreen::player(0));
        s.container.set_cursor(Some(stack("stone", 5)));
        let point = |s: &ContainerScreen, id: usize| {
            let slot = s.container.slots()[id];
            (s.gui_left() + slot.x + 8, s.gui_top() + slot.y + 8)
        };
        let (x, y) = point(&s, 36);
        let _ = s.mouse_clicked_with_mods(x, y, DEFAULT_PICK_BLOCK_BUTTON, Modifiers::NONE);
        for id in [36, 37, 38] {
            let (x, y) = point(&s, id);
            s.mouse_dragged(x, y, DEFAULT_PICK_BLOCK_BUTTON, 10);
        }
        assert_eq!(s.drag_preview_remnant(), Some(5 - 3 * 64));
        let mut d = DrawList::new();
        // Parked over the panel's top edge, off every slot and the player preview.
        s.draw(
            &mut d,
            &mut uniform_font(),
            &ctx(),
            (s.gui_left() + 100, s.gui_top() + 2),
        );
        // `§c` is 0xFF5555; each glyph is one six-vertex quad (its shadow is darker).
        let red = d
            .vertices()
            .iter()
            .filter(|v| {
                (v.color[0] - 1.0).abs() < 1e-3
                    && (v.color[1] - 85.0 / 255.0).abs() < 1e-3
                    && (v.color[2] - 85.0 / 255.0).abs() < 1e-3
            })
            .count();
        assert_eq!(red, 4 * 6, "the held stack's red \"-187\"");
    }

    /// `AnvilArt::creative` / `EnchantingArt::creative` had **no writer anywhere in the
    /// workspace** -- the host assigns `ContainerScreen::creative` (the field the click state
    /// machine reads) and the art's own copy stayed `false` forever, so `is_too_expensive`
    /// reported "Too Expensive!" to a creative player and `is_affordable` refused every
    /// enchantment offer they could actually take. java reads
    /// `mc.thePlayer.capabilities.isCreativeMode` afresh at each use site.
    #[test]
    fn drawing_refreshes_the_arts_creative_flag_from_the_screens() {
        let mut s = opened(ContainerScreen::anvil(1, "Repair"));
        s.creative = true;
        s.apply_window_property(&s2c::WindowProperty {
            window_id: 1,
            var_index: 0,
            var_value: 41,
        });
        let ContainerArt::Anvil(before) = &s.art else {
            unreachable!()
        };
        assert!(
            before.is_too_expensive(),
            "negative control: stale flag, so 41 levels still reads as too expensive"
        );
        s.draw(&mut DrawList::new(), &mut uniform_font(), &ctx(), (0, 0));
        let ContainerArt::Anvil(after) = &s.art else {
            unreachable!()
        };
        assert!(
            !after.is_too_expensive(),
            "java: the 40-level cap does not apply in creative"
        );
    }

    /// The anvil's `S31PacketWindowProperty` must reach the art and **not** the click state
    /// machine: vanilla sends it once, at open, with `0`, and never refreshes it
    /// (`ContainerRepair.java:389-393`). Wiring it into `Container::can_take_stack` would pin
    /// the gate shut for the whole session.
    #[test]
    fn the_anvil_window_property_feeds_the_art_but_never_the_click_gate() {
        let mut s = opened(ContainerScreen::anvil(1, "Repair"));
        s.container.set_slot_contents(2, Some(stack("stone", 1)));
        s.apply_window_property(&s2c::WindowProperty {
            window_id: 1,
            var_index: 0,
            var_value: 0,
        });
        let ContainerArt::Anvil(a) = &s.art else {
            unreachable!()
        };
        assert_eq!(a.maximum_cost, 0, "the art gets it (and renders no price)");
        assert_eq!(s.container.maximum_cost, None, "the gate must not");
        assert!(
            s.container.can_take_stack(2, false),
            "an anvil take is still allowed while no local repair cost is known"
        );
    }

    /// ... and once something *does* compute the cost, the gate and the XP level both work.
    #[test]
    fn a_computed_repair_cost_and_the_players_level_gate_the_anvil_output() {
        let mut s = opened(ContainerScreen::anvil(1, "Repair"));
        s.container.set_slot_contents(2, Some(stack("stone", 1)));
        s.container.maximum_cost = Some(7);
        s.set_player_experience_level(3);
        assert!(!s.container.can_take_stack(2, false), "3 levels < 7");
        s.set_player_experience_level(7);
        assert!(s.container.can_take_stack(2, false));
    }

    /// The whole crafting-table path, end to end: the packet a vanilla server actually sends
    /// (three-argument `S2DPacketOpenWindow`, so `slot_count == 0`) has to produce a 46-slot
    /// `ContainerWorkbench` screen. Before this, `for_open_window` returned `None` and the
    /// driver logged "unsupported window type", leaving the server holding a window the client
    /// could never send a `C0DPacketCloseWindow` for.
    #[test]
    fn a_zero_slot_crafting_table_packet_opens_a_workbench_screen() {
        let pkt = s2c::OpenWindow {
            window_id: 7,
            inventory_type: "minecraft:crafting_table".to_string(),
            window_title: "{\"text\":\"Crafting Table\"}".to_string(),
            slot_count: 0,
            entity_id: None,
        };
        let s = opened(ContainerScreen::for_open_window(&pkt, "ignored").expect("must open"));
        assert_eq!(s.art, ContainerArt::Crafting);
        assert_eq!(s.container.slots().len(), 46);
        assert_eq!(s.container.window_id, 7);
        assert_eq!(s.art.x_size(), 176);
        assert_eq!(s.art.y_size(), 166);
        assert!(
            !s.art.allows_user_input(),
            "java: only GuiInventory and GuiContainerCreative set allowUserInput"
        );
        // java: `GuiCrafting` draws `I18n.format("container.crafting")` and never the packet's
        // title, which for a crafting table is the block's display name.
        assert_eq!(s.title, "Crafting");
    }

    #[test]
    fn villager_window_routes_trlist_and_trsel_without_becoming_a_chest() {
        let pkt = s2c::OpenWindow {
            window_id: 8,
            inventory_type: "minecraft:villager".to_owned(),
            window_title: "{\"text\":\"Farmer\"}".to_owned(),
            slot_count: 3,
            entity_id: Some(42),
        };
        let mut s = opened(ContainerScreen::for_open_window(&pkt, "Farmer").expect("merchant"));
        assert!(matches!(s.art, ContainerArt::Merchant(_)));
        assert_eq!(
            s.container.kind,
            super::super::model::ContainerKind::Merchant
        );
        assert_eq!(s.container.slots().len(), 39);

        let buy = stack("wheat", 20);
        let sell = stack("emerald", 1);
        let mut buf = mc_protocol::PacketBuffer::new();
        buf.write_i32(8);
        buf.write_u8(2);
        for _ in 0..2 {
            mc_protocol::item::ItemStack::write(Some(&buy), &mut buf);
            mc_protocol::item::ItemStack::write(Some(&sell), &mut buf);
            buf.write_bool(false);
            buf.write_bool(false);
            buf.write_i32(0);
            buf.write_i32(7);
        }
        assert!(s.apply_merchant_plugin(buf.as_slice()).expect("decode"));
        let selected = match &s.art {
            ContainerArt::Merchant(merchant) => merchant.selected,
            _ => usize::MAX,
        };
        assert_eq!(selected, 0);

        s.mouse_clicked(s.gui_left() + 149, s.gui_top() + 25, 0);
        let plugin = s.take_plugin_outgoing();
        assert_eq!(plugin.len(), 1);
        assert_eq!(plugin[0].channel, "MC|TrSel");
        assert_eq!(plugin[0].data, 1_i32.to_be_bytes().to_vec());

        // Java reads the header even when it is for another window, but never touches that
        // window's recipe list. A malformed foreign body is therefore a silent no-op; only the
        // header itself is malformed enough to reach the handler's caught IOException.
        let mut foreign = mc_protocol::PacketBuffer::new();
        foreign.write_i32(99);
        foreign.write_u8(1);
        foreign.write_u8(0xFF);
        assert!(!s
            .apply_merchant_plugin(foreign.as_slice())
            .expect("foreign body is not decoded"));
        assert!(s.apply_merchant_plugin(&[0, 0, 0]).is_err());

        // A non-merchant screen with this window id follows the same gate: its body is not
        // decoded, but a four-byte header is still consumed successfully.
        let mut chest = opened(ContainerScreen::chest(8, 3, "Chest"));
        assert!(!chest
            .apply_merchant_plugin(foreign.as_slice())
            .expect("non-merchant body is not decoded"));

        // A matching merchant body is decoded atomically. Truncating it must not replace the
        // already-installed offers or selected recipe.
        let before = s.art.clone();
        let mut malformed = mc_protocol::PacketBuffer::new();
        malformed.write_i32(8);
        malformed.write_u8(1);
        assert!(s.apply_merchant_plugin(malformed.as_slice()).is_err());
        assert_eq!(s.art, before);
    }

    #[test]
    fn merchant_recipe_previews_share_the_item_tooltip_path() {
        let mut s = opened(ContainerScreen::merchant(8, "Farmer"));
        let ContainerArt::Merchant(merchant) = &mut s.art else {
            unreachable!()
        };
        merchant.recipes.push(super::super::model::MerchantRecipe {
            buy_a: stack("wheat", 20),
            buy_b: Some(stack("coal", 3)),
            sell: stack("emerald", 1),
            uses: 0,
            max_uses: 7,
            disabled: false,
        });

        for (x, expected) in [
            (36, stack("wheat", 20)),
            (62, stack("coal", 3)),
            (120, stack("emerald", 1)),
        ] {
            s.draw(
                &mut DrawList::new(),
                &mut uniform_font(),
                &ctx(),
                (s.gui_left() + x + 8, s.gui_top() + 24 + 8),
            );
            assert_eq!(
                s.hovered_tooltip(),
                Some(tooltip_box::recolour_tooltip_lines(
                    tooltip::build(&expected, false),
                    tooltip::rarity(&expected),
                )),
                "merchant preview at x={x} must expose its exact stack tooltip"
            );
        }

        s.container.set_cursor(Some(stack("stone", 1)));
        assert!(
            s.hovered_tooltip().is_some(),
            "GuiMerchant's custom preview tooltip runs after the base cursor guard"
        );
        s.container.set_cursor(None);

        s.draw(
            &mut DrawList::new(),
            &mut uniform_font(),
            &ctx(),
            (s.gui_left() + 80, s.gui_top() + 32),
        );
        assert!(s.hovered_tooltip().is_none(), "outside all three previews");

        let mut empty_refresh = mc_protocol::PacketBuffer::new();
        empty_refresh.write_i32(8);
        empty_refresh.write_u8(0);
        assert!(s
            .apply_merchant_plugin(empty_refresh.as_slice())
            .expect("zero-offer refresh"));
        assert!(s.hovered_tooltip().is_none(), "refresh clears stale hover");
        let ContainerArt::Merchant(merchant) = &s.art else {
            unreachable!()
        };
        assert!(merchant.recipes.is_empty());
        assert!(s.container.slot_stack(2).is_none());
    }

    /// java: `ContainerWorkbench.transferStackInSlot` -- the result slot merges into `10..46`
    /// **reversed** (hotbar first), the 3x3 grid merges into `10..46` forwards.
    #[test]
    fn shift_clicking_the_workbench_result_fills_the_hotbar_first() {
        let mut s = opened(ContainerScreen::crafting_table(1, "Crafting"));
        s.container.set_slot_contents(0, Some(stack("stone", 4)));
        let packet = s.container.click(0, 0, 1, false);
        assert_eq!(packet.mode, 1);
        // Reverse merge starts at the end: slot 45 is the last hotbar slot.
        assert_eq!(s.container.slot_stack(45), Some(&stack("stone", 4)));
        assert_eq!(s.container.slot_stack(0), None);

        // A grid cell merges forwards, so it lands in the first main-inventory slot.
        s.container.set_slot_contents(5, Some(stack("dirt", 3)));
        s.container.click(5, 0, 1, false);
        assert_eq!(s.container.slot_stack(10), Some(&stack("dirt", 3)));
    }

    /// java: `ContainerWorkbench.canMergeSlot` excludes `craftResult`, exactly as
    /// `ContainerPlayer` does, so a double-click never sweeps items out of the result slot.
    #[test]
    fn the_workbench_result_slot_is_excluded_from_double_click_collection() {
        let s = opened(ContainerScreen::crafting_table(1, "Crafting"));
        assert!(!s.container.can_merge_slot(0));
        for id in 1..46u16 {
            assert!(s.container.can_merge_slot(id), "slot {id}");
        }
    }

    #[test]
    fn a_hotbar_key_over_a_hovered_slot_sends_mode_two() {
        let mut s = opened(ContainerScreen::chest(1, 3, "Chest"));
        s.container.set_slot_contents(0, Some(stack("stone", 4)));
        let mut d = DrawList::new();
        let mut font = uniform_font();
        let slot = s.container.slots()[0];
        let point = (s.gui_left() + slot.x + 8, s.gui_top() + slot.y + 8);
        s.draw(&mut d, &mut font, &ctx(), point);
        assert_eq!(s.hovered_slot(), Some(0));
        s.key_typed(Some('3'), KEY_HOTBAR_1 + 2, Modifiers::NONE);
        let out = s.take_outgoing();
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].slot_id, out[0].used_button, out[0].mode), (0, 2, 2));
    }

    /// NOT VANILLA 1.8.9 (the hotbar block in `mouse_clicked_with_mods`): a mouse button bound
    /// to `key.hotbar.3`, pressed over a hovered slot, puts the keyboard key's exact C0E bytes
    /// on the wire. Its press and release do nothing else: even bound to the left button, the
    /// press does not also pick the stack up and the release neither places nor collects. With
    /// a stack on the cursor `checkHotbarKeys` declines and the press stays vanilla's.
    #[test]
    fn a_mouse_bound_hotbar_key_sends_the_keyboard_keys_exact_click() {
        fn wire(packets: &[mc_protocol::play::C2SPlay]) -> Vec<(i32, Vec<u8>)> {
            packets
                .iter()
                .map(|packet| {
                    let mut body = mc_protocol::PacketBuffer::new();
                    packet.write(&mut body);
                    (packet.id(), body.as_slice().to_vec())
                })
                .collect()
        }
        let hovered_chest = |bindings: ContainerBindings| {
            let mut s = opened(ContainerScreen::chest(1, 3, "Chest"));
            s.set_bindings(bindings);
            s.container.set_slot_contents(0, Some(stack("stone", 4)));
            let slot = s.container.slots()[0];
            let point = (s.gui_left() + slot.x + 8, s.gui_top() + slot.y + 8);
            s.draw(&mut DrawList::new(), &mut uniform_font(), &ctx(), point);
            assert_eq!(s.hovered_slot(), Some(0));
            (s, point)
        };
        let contents = |s: &ContainerScreen| {
            (0..s.container.slots().len())
                .map(|index| s.container.slot_stack(index).cloned())
                .collect::<Vec<_>>()
        };

        let (mut keyboard, _) = hovered_chest(ContainerBindings::default());
        keyboard.key_typed(Some('3'), KEY_HOTBAR_1 + 2, Modifiers::NONE);
        let expected = wire(&keyboard.take_ordered_outgoing());
        assert_eq!(expected.len(), 1, "control: one mode-2 C0E");

        for button in [4, 0] {
            // A mouse binding is stored as `button - 100`.
            let mut bindings = ContainerBindings::default();
            bindings.hotbar_keys[2] = button - 100;
            let (mut mouse, (x, y)) = hovered_chest(bindings);
            assert_eq!(
                mouse.mouse_clicked_with_mods(x, y, button, Modifiers::NONE),
                ScreenAction::None
            );
            mouse.mouse_dragged(x, y, button, 0);
            mouse.mouse_released_with_mods(x, y, button, Modifiers::NONE);
            assert_eq!(
                wire(&mouse.take_ordered_outgoing()),
                expected,
                "button {button}"
            );
            assert_eq!(contents(&mouse), contents(&keyboard), "button {button}");
            assert!(mouse.container.cursor().is_none(), "button {button}");
        }

        let mut bindings = ContainerBindings::default();
        bindings.hotbar_keys[2] = 4 - 100;
        let (mut carrying, (x, y)) = hovered_chest(bindings);
        carrying.container.set_cursor(Some(stack("dirt", 1)));
        carrying.mouse_clicked_with_mods(x, y, 4, Modifiers::NONE);
        carrying.mouse_released_with_mods(x, y, 4, Modifiers::NONE);
        assert!(carrying.take_outgoing().iter().all(|click| click.mode != 2));
    }

    /// java: `GuiRepair.sendSlotContents` resets the name field only when
    /// `Container.detectAndSendChanges` finds slot 0 itself changed (`areItemStacksEqual`).
    /// The renamed slot-2 output that answers every `MC|ItemName`, or any other window packet,
    /// must not look like that, or the typed name is reset after every keystroke.
    #[test]
    fn only_a_slot_zero_change_asks_the_host_to_resync_the_anvil_name() {
        let mut s = opened(ContainerScreen::anvil(1, "Repair & Name"));
        let font = uniform_font();
        assert!(
            !s.anvil_input_needs_sync(),
            "initGui already reported the empty slot 0"
        );

        s.container.set_slot_contents(0, Some(stack("apple", 1)));
        assert!(s.anvil_input_needs_sync());
        s.sync_anvil_input_name(Some("Apple"), false, &font);
        assert!(!s.anvil_input_needs_sync());

        s.anvil_name.as_mut().unwrap().set_focused(true);
        Screen::key_typed_with_font(&mut s, Some('!'), 0, Modifiers::NONE, &font);
        assert_eq!(s.anvil_rename_text().as_deref(), Some("Apple!"));
        // The server's answer to MC|ItemName (the renamed slot-2 output), then a property.
        s.container.set_slot_contents(2, Some(stack("apple", 1)));
        s.apply_window_property(&s2c::WindowProperty {
            window_id: 1,
            var_index: 0,
            var_value: 1,
        });
        assert!(!s.anvil_input_needs_sync(), "slot 0 did not change");
        assert_eq!(s.anvil_rename_text().as_deref(), Some("Apple!"));

        // Any real slot-0 change does, including a count change and emptying it.
        s.container.set_slot_contents(0, Some(stack("apple", 2)));
        assert!(s.anvil_input_needs_sync());
        s.sync_anvil_input_name(Some("Apple"), false, &font);
        s.container.set_slot_contents(0, None);
        assert!(s.anvil_input_needs_sync());
        s.sync_anvil_input_name(None, false, &font);
        assert!(!s.anvil_input_needs_sync());

        assert!(!opened(ContainerScreen::chest(1, 3, "Chest")).anvil_input_needs_sync());
    }

    /// java: `GuiRepair` fills the field with `getDisplayName()` in the
    /// player's language and `updateRepairOutput` compares against that same name. In French a
    /// plain apple reads "Pomme"; retyping it as the English "Apple" is a one-level rename.
    #[test]
    fn retyping_a_localized_anvil_name_in_english_is_a_rename() {
        let mut s = opened(ContainerScreen::anvil(1, "Repair & Name"));
        let font = uniform_font();
        s.container.set_slot_contents(0, Some(stack("apple", 1)));
        s.sync_anvil_input_name(Some("Pomme"), false, &font);
        assert_eq!(s.take_anvil_name_changes(), [String::new()]);
        assert_eq!(
            s.container.maximum_cost,
            Some(0),
            "an untouched name costs nothing"
        );
        assert_eq!(s.container.slot_stack(2), None);

        s.anvil_name.as_mut().unwrap().set_focused(true);
        Screen::key_typed_with_font(&mut s, None, keys::A, Modifiers::CTRL, &font);
        for typed in "Apple".chars() {
            Screen::key_typed_with_font(&mut s, Some(typed), 0, Modifiers::NONE, &font);
        }
        assert_eq!(s.anvil_rename_text().as_deref(), Some("Apple"));
        assert_eq!(
            s.take_anvil_name_changes().last().map(String::as_str),
            Some("Apple")
        );
        assert_eq!(s.container.maximum_cost, Some(1));
        let output = s.container.slot_stack(2).expect("the rename output");
        assert_eq!(
            output
                .nbt
                .as_ref()
                .and_then(|root| root.get_compound("display"))
                .and_then(|display| display.get_str("Name")),
            Some("Apple")
        );
    }

    #[test]
    fn the_drop_key_over_a_filled_slot_sends_mode_four_and_ctrl_drops_the_whole_stack() {
        let mut s = opened(ContainerScreen::chest(1, 3, "Chest"));
        s.container.set_slot_contents(0, Some(stack("stone", 4)));
        let mut d = DrawList::new();
        let mut font = uniform_font();
        let slot = s.container.slots()[0];
        s.draw(
            &mut d,
            &mut font,
            &ctx(),
            (s.gui_left() + slot.x + 8, s.gui_top() + slot.y + 8),
        );
        s.key_typed(Some('q'), KEY_DROP, Modifiers::NONE);
        s.key_typed(Some('q'), KEY_DROP, Modifiers::CTRL);
        let out = s.take_outgoing();
        assert_eq!(out.len(), 2);
        assert_eq!((out[0].mode, out[0].used_button), (4, 0));
        assert_eq!((out[1].mode, out[1].used_button), (4, 1));
    }

    /// `MouseState` has always implemented mode 1 completely; the screen
    /// passed a hard-coded `false` for `shift`, so no caller could ever reach it. These two
    /// assertions are the same click with the modifier down and up.
    ///
    /// java: `GuiContainer.mouseClicked` (`:359`) picks `clickType == 1` from
    /// `isShiftKeyDown()`, and `Container.slotClick`'s `mode == 1` arm is the whole
    /// `transferStackInSlot` path.
    #[test]
    fn holding_shift_turns_a_slot_click_into_a_mode_one_transfer() {
        let mut s = opened(ContainerScreen::chest(1, 3, "Chest"));
        s.container.set_slot_contents(0, Some(stack("stone", 10)));
        let slot = s.container.slots()[0];
        let (x, y) = (s.gui_left() + slot.x + 8, s.gui_top() + slot.y + 8);

        s.mouse_clicked_with_mods(x, y, 0, Modifiers::SHIFT);
        let out = s.take_outgoing();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].mode, 1, "shift-click is click mode 1");
        assert_eq!(out[0].slot_id, 0);
        assert_eq!(out[0].window_id, 1);
        assert_eq!(
            s.container.cursor(),
            None,
            "a transfer never puts the stack on the cursor"
        );
        assert!(
            s.container.slot_stack(0).is_none(),
            "the chest slot must have been emptied into the player inventory"
        );
        assert!(
            // 27 chest slots, then the player's 36.
            (27..63).any(|id| s.container.slot_stack(id).is_some()),
            "…and the stack must have landed somewhere in the player's own slots"
        );
    }

    /// java: `GuiContainer.mouseReleased`'s last branch (`:626`) — with a stack on the cursor,
    /// the release itself is the click, and `flag1` (shift, and not the `-999` sentinel) picks
    /// mode 1 over mode 0. This is the release-path half; the press path above is the
    /// other.
    #[test]
    fn releasing_a_held_stack_with_shift_down_sends_mode_one_not_mode_zero() {
        let mut s = opened(ContainerScreen::chest(1, 3, "Chest"));
        s.container.set_cursor(Some(stack("stone", 4)));
        let slot = s.container.slots()[6];
        let (x, y) = (s.gui_left() + slot.x + 8, s.gui_top() + slot.y + 8);

        // A press with a stack held only *arms* a drag; nothing is sent until the release.
        assert_eq!(
            s.mouse_clicked_with_mods(x, y, 0, Modifiers::SHIFT),
            ScreenAction::None
        );
        assert!(s.take_outgoing().is_empty());

        s.mouse_released_with_mods(x, y, 0, Modifiers::SHIFT);
        let out = s.take_outgoing();
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].slot_id, out[0].mode), (6, 1));
    }

    #[test]
    fn the_same_click_without_shift_is_still_an_ordinary_mode_zero_pickup() {
        let mut s = opened(ContainerScreen::chest(1, 3, "Chest"));
        s.container.set_slot_contents(0, Some(stack("stone", 10)));
        let slot = s.container.slots()[0];
        let (x, y) = (s.gui_left() + slot.x + 8, s.gui_top() + slot.y + 8);
        s.mouse_clicked_with_mods(x, y, 0, Modifiers::NONE);
        let out = s.take_outgoing();
        assert_eq!(out[0].mode, 0);
        assert_eq!(s.container.cursor().map(|c| c.count), Some(10));
    }

    /// The trait method has no [`Modifiers`] to read, so it must state what it assumes rather
    /// than silently claiming shift was up for a caller that did know better. Pinned so a future
    /// change to the trait signature has to come past this test.
    #[test]
    fn the_modifier_free_trait_method_behaves_exactly_like_modifiers_none() {
        let make = || {
            let mut s = opened(ContainerScreen::chest(1, 3, "Chest"));
            s.container.set_slot_contents(0, Some(stack("stone", 10)));
            s
        };
        let mut through_trait = make();
        let mut through_mods = make();
        let slot = through_trait.container.slots()[0];
        let (x, y) = (
            through_trait.gui_left() + slot.x + 8,
            through_trait.gui_top() + slot.y + 8,
        );
        through_trait.mouse_clicked(x, y, 0);
        through_mods.mouse_clicked_with_mods(x, y, 0, Modifiers::NONE);
        assert_eq!(
            through_trait.take_outgoing(),
            through_mods.take_outgoing(),
            "the trait method must forward, not diverge"
        );
    }

    /// java: `mouseReleased` (`:515`) reads `isShiftKeyDown()` for the shift-double-click sweep
    /// at `:535`, which emits **one mode-1 click per matching slot in the same inventory** —
    /// unlike the plain double click, which emits a single mode 6.
    #[test]
    fn a_shift_double_click_sweeps_the_backing_inventory_with_mode_one_clicks() {
        let mut s = opened(ContainerScreen::chest(1, 3, "Chest"));
        // The sweep only has anything to do while the *origin* slot still holds its stack after
        // the first shift-click, so the player's own 36 slots are packed with full stacks of a
        // different item and `transferStackInSlot` has nowhere to put the stone.
        for id in 27..63usize {
            s.container.set_slot_contents(id, Some(stack("dirt", 64)));
        }
        for id in 0..3usize {
            s.container.set_slot_contents(id, Some(stack("stone", 8)));
        }
        let slot = s.container.slots()[0];
        let (x, y) = (s.gui_left() + slot.x + 8, s.gui_top() + slot.y + 8);

        // The first press records `shiftClickedSlot`; the second, inside the 250 ms window, is
        // the double click whose *release* runs the sweep. The clock starts at a minute in, not
        // at zero, so `now - last_click_time` cannot be confused with `now - 0` -- see
        // `interaction::tests::an_ignored_release_preserves_the_double_click_clock_on_a_realistic_clock`.
        s.set_clock_ms(60_000);
        s.mouse_clicked_with_mods(x, y, 0, Modifiers::SHIFT);
        s.mouse_released_with_mods(x, y, 0, Modifiers::SHIFT);
        s.set_clock_ms(60_100);
        s.mouse_clicked_with_mods(x, y, 0, Modifiers::SHIFT);
        assert!(
            s.container.slot_stack(0).is_some(),
            "the setup is only meaningful while the transfer had nowhere to go"
        );
        let _ = s.take_outgoing();
        s.mouse_released_with_mods(x, y, 0, Modifiers::SHIFT);

        let out = s.take_outgoing();
        assert_eq!(
            out.iter().map(|p| (p.slot_id, p.mode)).collect::<Vec<_>>(),
            vec![(0, 1), (1, 1), (2, 1)],
            "one mode-1 transfer per matching slot in the same backing inventory. java's loop is
             `for (Slot slot2 : this.inventorySlots.inventorySlots)` with five conditions, none
             of which is `slot2 != slot`, so the double-clicked slot is swept too"
        );
    }

    /// And the *unshifted* double click is still vanilla's single mode 6, so the sweep above
    /// cannot be mistaken for a change in the plain path.
    #[test]
    fn an_unshifted_double_click_is_still_one_mode_six_collect() {
        let mut s = opened(ContainerScreen::chest(1, 3, "Chest"));
        s.container.set_slot_contents(0, Some(stack("stone", 8)));
        let slot = s.container.slots()[0];
        let (x, y) = (s.gui_left() + slot.x + 8, s.gui_top() + slot.y + 8);
        s.mouse_clicked_with_mods(x, y, 0, Modifiers::NONE);
        s.mouse_released_with_mods(x, y, 0, Modifiers::NONE);
        s.set_clock_ms(100);
        s.mouse_clicked_with_mods(x, y, 0, Modifiers::NONE);
        let _ = s.take_outgoing();
        s.mouse_released_with_mods(x, y, 0, Modifiers::NONE);
        let out = s.take_outgoing();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].mode, 6);
    }

    /// java: `GuiEnchantment` draws the offer's level number with
    /// `i2 = 8453920` when it can be taken and `4226832` when it cannot. Both were transcribed
    /// to the wrong hex, one of them to a different hue family entirely. Pinning the decimals
    /// is the check that could not drift: they are copied straight out of the java.
    #[test]
    fn the_enchant_offer_colours_are_the_decimals_the_java_writes() {
        assert_eq!(ENCHANT_LEVEL_COLOR, 8_453_920);
        assert_eq!(ENCHANT_LEVEL_DISABLED_COLOR, 4_226_832);
        // …and the affordable one is a green, not the purple it used to be.
        let (r, g, b) = (
            (ENCHANT_LEVEL_COLOR >> 16) & 0xFF,
            (ENCHANT_LEVEL_COLOR >> 8) & 0xFF,
            ENCHANT_LEVEL_COLOR & 0xFF,
        );
        assert!(g > r && g > b, "green must dominate: {r:02x}{g:02x}{b:02x}");
    }

    /// java: `GuiEnchantment.drawGuiContainerBackgroundLayer` uses a raw `0 <= x < 108`,
    /// `0 <= y < 19` test and selects the texture row at v=204 only for an affordable offer.
    /// Inspect the production draw list rather than only the coordinate helper.
    #[test]
    fn an_affordable_enchant_offer_uses_the_hover_plate_inside_exact_raw_bounds() {
        fn row_zero_plate_v(screen: &mut ContainerScreen, mouse: (i32, i32)) -> i32 {
            let plate_pos = [
                (screen.gui_left() + 60) as f32,
                (screen.gui_top() + 14) as f32,
            ];
            let mut draw = DrawList::new();
            screen.draw(&mut draw, &mut uniform_font(), &ctx(), mouse);
            let vertex = draw
                .vertices()
                .iter()
                .find(|vertex| vertex.pos == plate_pos && vertex.uv[0] == 0.0)
                .expect("row-zero enchant plate top-left vertex");
            (vertex.uv[1] * 256.0).round() as i32
        }

        let mut screen = opened(ContainerScreen::enchanting_table(1, "Enchant"));
        screen.creative = true;
        let ContainerArt::Enchanting(art) = &mut screen.art else {
            unreachable!();
        };
        art.enchant_levels = [5, 0, 0];
        let origin = (screen.gui_left() + 60, screen.gui_top() + 14);

        assert_eq!(
            row_zero_plate_v(&mut screen, (origin.0 + 107, origin.1 + 18)),
            204
        );
        assert_eq!(
            row_zero_plate_v(&mut screen, (origin.0 + 108, origin.1 + 18)),
            166,
            "the right edge is exclusive"
        );
        assert_eq!(
            row_zero_plate_v(&mut screen, (origin.0 + 107, origin.1 + 19)),
            166,
            "the bottom edge belongs to the next row, not row zero"
        );
    }

    /// LWJGL scan code `2` is the `1` key *and* was the value
    /// `pick_block_button` held as a mouse index, so one field answered two different questions.
    /// With a stack on the cursor the hotbar branch is skipped (`checkHotbarKeys` needs an empty
    /// one), and the key then fell straight into the mode-3 dispatch.
    ///
    /// Since A07-01 landed that stray packet reaches the server and steps the transaction
    /// counter, so this is now a desync rather than a harmless no-op. java: `keyTyped` compares
    /// against `keyBindPickBlock.getKeyCode()`, which is `-98` for a mouse binding.
    #[test]
    fn the_one_key_over_a_slot_with_a_full_cursor_sends_nothing() {
        let mut s = opened(ContainerScreen::chest(1, 3, "Chest"));
        s.container.set_slot_contents(0, Some(stack("stone", 4)));
        s.container.set_cursor(Some(stack("dirt", 1)));
        let slot = s.container.slots()[0];
        s.draw(
            &mut DrawList::new(),
            &mut uniform_font(),
            &ctx(),
            (s.gui_left() + slot.x + 8, s.gui_top() + slot.y + 8),
        );
        assert_eq!(s.hovered_slot(), Some(0));

        s.key_typed(Some('1'), KEY_HOTBAR_1, Modifiers::NONE);
        assert!(
            s.take_outgoing().is_empty(),
            "the `1` key must not be read as the pick-block *mouse* button"
        );
        assert_eq!(
            DEFAULT_PICK_BLOCK_KEY, -98,
            "java: a mouse binding's `getKeyCode()` is `button - 100`, which no key event carries"
        );

        // A host that really did rebind pick-block to a keyboard key still gets the branch.
        s.bindings.pick_block_key = KEY_HOTBAR_1;
        s.container.set_cursor(None);
        s.key_typed(Some('1'), KEY_HOTBAR_1, Modifiers::NONE);
        let out = s.take_outgoing();
        assert_eq!(
            out.len(),
            1,
            "an empty cursor takes `checkHotbarKeys` first"
        );
        assert_eq!(out[0].mode, 2, "…which is the hotbar swap, not pick block");
    }

    #[test]
    fn escape_and_the_inventory_key_both_pop_the_screen() {
        let mut s = opened(ContainerScreen::chest(1, 3, "Chest"));
        assert_eq!(
            s.key_typed(None, keys::ESCAPE, Modifiers::NONE),
            ScreenAction::Pop
        );
        assert_eq!(
            s.key_typed(None, KEY_INVENTORY, Modifiers::NONE),
            ScreenAction::Pop
        );
    }

    #[test]
    fn close_binding_still_emits_colliding_hotbar_then_drop_once() {
        let mut hotbar_keys = ContainerBindings::default().hotbar_keys;
        hotbar_keys[2] = 40;
        let bindings = ContainerBindings {
            inventory_key: 40,
            drop_key: 40,
            hotbar_keys,
            ..Default::default()
        };

        let mut s = opened(ContainerScreen::chest(1, 3, "Chest"));
        s.set_bindings(bindings);
        // Use a player-inventory source so Java's mode-2 swap leaves the hovered slot filled
        // with the former hotbar stack; that makes the following drop check observable.
        let source_slot = 27;
        s.container
            .set_slot_contents(source_slot, Some(stack("stone", 4)));
        let hotbar_slot = s.container.slots().len() - 9 + 2;
        s.container
            .set_slot_contents(hotbar_slot, Some(stack("dirt", 2)));
        let slot = s.container.slots()[source_slot];
        s.draw(
            &mut DrawList::new(),
            &mut uniform_font(),
            &ctx(),
            (s.gui_left() + slot.x + 8, s.gui_top() + slot.y + 8),
        );
        assert_eq!(s.hovered_slot(), Some(source_slot as u16));

        // java: closeScreen() runs first, but keyTyped then ignores checkHotbarKeys' success and
        // reaches the filled-slot drop branch. The Rust caller applies this one Pop after it
        // drains both queued C0E actions, so there is no second close action to apply.
        assert_eq!(
            s.key_typed(None, bindings.inventory_key, Modifiers::NONE),
            ScreenAction::Pop
        );
        let packets = s.take_ordered_outgoing();
        let actions = packets
            .iter()
            .map(|packet| match packet {
                mc_protocol::play::C2SPlay::ClickWindow(packet) => {
                    (packet.slot_id, packet.used_button, packet.mode)
                }
                other => panic!("unexpected packet from close-key collision: {other:?}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            actions,
            vec![(source_slot as i16, 2, 2), (source_slot as i16, 0, 4)]
        );

        // Escape remains a single close action when there is no producer bound to it; the first
        // close key must not leave a phantom duplicate packet behind for the next key event.
        assert_eq!(
            s.key_typed(None, keys::ESCAPE, Modifiers::NONE),
            ScreenAction::Pop
        );
        assert!(s.take_ordered_outgoing().is_empty());
    }

    #[test]
    fn escape_close_binding_still_emits_colliding_hotbar_then_ctrl_drop() {
        let mut bindings = ContainerBindings::default();
        bindings.hotbar_keys[1] = keys::ESCAPE;
        bindings.drop_key = keys::ESCAPE;

        let mut s = opened(ContainerScreen::chest(1, 3, "Chest"));
        s.set_bindings(bindings);
        let source_slot = 27;
        s.container
            .set_slot_contents(source_slot, Some(stack("stone", 4)));
        let hotbar_slot = s.container.slots().len() - 9 + 1;
        s.container
            .set_slot_contents(hotbar_slot, Some(stack("dirt", 2)));
        let slot = s.container.slots()[source_slot];
        s.draw(
            &mut DrawList::new(),
            &mut uniform_font(),
            &ctx(),
            (s.gui_left() + slot.x + 8, s.gui_top() + slot.y + 8),
        );

        assert_eq!(
            s.key_typed(None, keys::ESCAPE, Modifiers::CTRL),
            ScreenAction::Pop
        );
        let packets = s.take_outgoing();
        assert_eq!(packets.len(), 2);
        assert_eq!((packets[0].mode, packets[0].used_button), (2, 1));
        assert_eq!((packets[1].mode, packets[1].used_button), (4, 1));
    }

    #[test]
    fn rebound_container_actions_accept_new_keys_and_reject_former_defaults() {
        fn screen_with_hover(bindings: ContainerBindings) -> ContainerScreen {
            let mut screen = opened(ContainerScreen::chest(1, 3, "Chest"));
            screen.set_bindings(bindings);
            screen
                .container
                .set_slot_contents(0, Some(stack("stone", 4)));
            let slot = screen.container.slots()[0];
            screen.draw(
                &mut DrawList::new(),
                &mut uniform_font(),
                &ctx(),
                (
                    screen.gui_left() + slot.x + 8,
                    screen.gui_top() + slot.y + 8,
                ),
            );
            screen
        }

        let mut bindings = ContainerBindings {
            inventory_key: 30,
            ..Default::default()
        };
        bindings.drop_key = 31;
        bindings.hotbar_keys[2] = 32;
        bindings.pick_block_key = 33;
        bindings.pick_block_button = 4;

        let mut close = screen_with_hover(bindings);
        assert_eq!(
            close.key_typed(None, KEY_INVENTORY, Modifiers::NONE),
            ScreenAction::None,
            "the former inventory default must stop closing the screen"
        );
        assert_eq!(
            close.key_typed(None, bindings.inventory_key, Modifiers::NONE),
            ScreenAction::Pop
        );

        for (new_key, old_key, expected_mode, expected_button) in [
            (bindings.hotbar_keys[2], KEY_HOTBAR_1 + 2, 2, 2),
            (bindings.pick_block_key, DEFAULT_PICK_BLOCK_KEY, 3, 0),
            (bindings.drop_key, KEY_DROP, 4, 0),
        ] {
            let mut screen = screen_with_hover(bindings);
            screen.key_typed(None, old_key, Modifiers::NONE);
            assert!(
                screen.take_outgoing().is_empty(),
                "former default {old_key} must be inert"
            );
            screen.key_typed(None, new_key, Modifiers::NONE);
            let packets = screen.take_outgoing();
            assert_eq!(packets.len(), 1, "rebound key {new_key} must dispatch");
            assert_eq!(packets[0].mode, expected_mode);
            assert_eq!(packets[0].used_button, expected_button);
        }

        let mut mouse_pick = screen_with_hover(bindings);
        let slot = mouse_pick.container.slots()[0];
        mouse_pick.mouse_clicked_with_mods(
            mouse_pick.gui_left() + slot.x + 8,
            mouse_pick.gui_top() + slot.y + 8,
            bindings.pick_block_button,
            Modifiers::NONE,
        );
        assert_eq!(mouse_pick.take_outgoing()[0].mode, 3);
    }

    #[test]
    fn drawing_marks_the_hovered_slot_and_emits_a_highlight_quad() {
        let mut s = opened(ContainerScreen::chest(1, 3, "Chest"));
        let mut font = uniform_font();
        let slot = s.container.slots()[7];
        let point = (s.gui_left() + slot.x + 3, s.gui_top() + slot.y + 3);

        let mut hovered = DrawList::new();
        s.draw(&mut hovered, &mut font, &ctx(), point);
        assert_eq!(s.hovered_slot(), Some(7));

        let mut not_hovered = DrawList::new();
        s.draw(&mut not_hovered, &mut font, &ctx(), (0, 0));
        assert_eq!(s.hovered_slot(), None);

        assert_eq!(
            hovered.vertices().len() - not_hovered.vertices().len(),
            6,
            "exactly one extra quad: the 16x16 hover gradient"
        );
    }

    /// C2-13: the tooltip cache is invisible. Every answer equals a fresh build, and a change
    /// of stack, advanced flag, language or map record is picked up on the very next call.
    #[test]
    fn cached_item_tooltips_equal_fresh_builds_and_follow_every_input() {
        let mut s = opened(ContainerScreen::chest(1, 3, "Chest"));
        let sword = stack("diamond_sword", 1);
        let mut worn = sword.clone();
        worn.damage = 100;
        let check = |s: &ContainerScreen, stack: &ItemStack, what: &str| {
            let fresh = s.build_item_tooltip(stack);
            assert_eq!(s.item_tooltip(stack), fresh, "{what}: first call");
            assert_eq!(s.item_tooltip(stack), fresh, "{what}: cached call");
            fresh
        };
        let plain = check(&s, &sword, "plain");
        s.advanced_tooltips = true;
        let advanced = check(&s, &sword, "advanced flag");
        assert_ne!(
            plain, advanced,
            "advanced tooltips add the id/durability lines"
        );
        let worn_lines = check(&s, &worn, "different damage");
        assert_ne!(advanced, worn_lines, "durability differs");
        s.set_language(&Language::new("en_US"));
        check(&s, &sword, "language set");
        let mut map = stack("filled_map", 1);
        map.damage = 37;
        let mut data = MapData::new(37);
        s.tooltip_map_data = Some(data.clone());
        let scale_zero = check(&s, &map, "map record");
        data.scale = 3;
        s.tooltip_map_data = Some(data);
        let scale_three = check(&s, &map, "map scale");
        assert_ne!(scale_zero, scale_three, "the scale line follows the record");
    }

    #[test]
    fn filled_map_tooltip_still_uses_the_live_map_record() {
        let mut s = opened(ContainerScreen::chest(1, 3, "Chest"));
        let mut map = stack("filled_map", 1);
        map.damage = 37;
        s.container.set_slot_contents(0, Some(map));
        s.advanced_tooltips = true;
        let slot = s.container.slots()[0];
        let mouse = (s.gui_left() + slot.x + 8, s.gui_top() + slot.y + 8);
        let mut data = MapData::new(37);
        data.scale = 3;

        s.draw_with_map_data(
            &mut DrawList::new(),
            &mut uniform_font(),
            &ctx(),
            mouse,
            Some(&data),
        );

        assert_eq!(s.hovered_map_id_at(mouse), Some(37));
        assert!(
            s.hovered_tooltip()
                .expect("filled map tooltip")
                .iter()
                .any(|line| line.ends_with("Scaling at 1:8")),
            "the live record's scale must reach ItemMap.addInformation"
        );
    }

    #[test]
    fn gui_container_item_calls_separate_slot_cursor_and_returning_depths() {
        let screen = opened(ContainerScreen::chest(1, 3, "Chest"));
        // A builtin chest keeps this caller test independent of the optional stitched item atlas
        // used by synthetic/headless screens while still exercising the real item depth route.
        let stone = stack("chest", 1);

        // GuiContainer.drawSlot sets itemRender.zLevel=100; drawItemStack (used by both the
        // carried cursor and returningStack) sets it to 200. RenderItem adds +50 in both calls,
        // so the same model must retain a fixed 0.05 depth separation per vertex.
        let mut slot = DrawList::new();
        screen.draw_item_with_count_override(&mut slot, &mut uniform_font(), 10, 10, &stone, None);
        let mut cursor = DrawList::new();
        screen.draw_item_with_count_override_at_z_level(
            &mut cursor,
            &mut uniform_font(),
            10,
            10,
            &stone,
            None,
            render::GUI_CURSOR_ITEM_RENDER_Z_LEVEL,
        );
        let mut returning = DrawList::new();
        screen.draw_item_with_count_override_at_z_level(
            &mut returning,
            &mut uniform_font(),
            10,
            10,
            &stone,
            None,
            render::GUI_CURSOR_ITEM_RENDER_Z_LEVEL,
        );

        assert!(!slot.depths().is_empty(), "slot emits item geometry");
        assert_eq!(slot.depths().len(), cursor.depths().len());
        assert_eq!(cursor.depths(), returning.depths());
        for (slot_depth, cursor_depth) in slot.depths().iter().zip(cursor.depths()) {
            assert!(
                (*slot_depth - *cursor_depth - 0.05).abs() < 1e-6,
                "cursor/returning is nearer by Java's 50 z units: {slot_depth} vs {cursor_depth}"
            );
        }

        // Reversed submission must keep the nearer cursor depth instead of relying on painter
        // order; the final ordinary rectangle also proves the item state window was restored.
        let mut reversed = DrawList::new();
        screen.draw_item_with_count_override_at_z_level(
            &mut reversed,
            &mut uniform_font(),
            10,
            10,
            &stone,
            None,
            render::GUI_CURSOR_ITEM_RENDER_Z_LEVEL,
        );
        screen.draw_item_with_count_override(
            &mut reversed,
            &mut uniform_font(),
            10,
            10,
            &stone,
            None,
        );
        assert_eq!(&reversed.depths()[..cursor.depths().len()], cursor.depths());
        assert_eq!(&reversed.depths()[cursor.depths().len()..], slot.depths());
        reversed.rect(0, 0, 1, 1, 0xFFFF_FFFF);
        assert_eq!(
            reversed.batches().last().map(|batch| batch.depth),
            Some(crate::widget::UiDepth::Overlay)
        );
    }

    #[test]
    fn drawing_a_filled_map_slot_uses_the_static_item_model_even_with_live_map_data() {
        let mut s = opened(ContainerScreen::chest(1, 3, "Chest"));
        let mut stack = stack("filled_map", 1);
        stack.damage = -7;
        s.container.set_slot_contents(0, Some(stack));
        let mut data = MapData::new(-7);
        data.colors[0] = 8 * 4 + 2;
        data.decorations.push(mc_world::MapDecoration {
            key: "icon-0".to_owned(),
            icon_type: 1,
            x: 0,
            z: 0,
            direction: 0,
        });
        s.set_map_render_data(vec![data]);

        let mut draw = DrawList::new();
        s.draw(&mut draw, &mut uniform_font(), &ctx(), (0, 0));
        let map_texture = crate::map::dynamic_texture_key(-7);
        let icons_texture = UiTexture::gui(crate::map::MAP_ICONS_TEXTURE);
        assert!(!draw
            .batches()
            .iter()
            .any(|batch| batch.texture == map_texture));
        assert!(!draw
            .batches()
            .iter()
            .any(|batch| batch.texture == icons_texture));

        // The focused slot path must be byte-identical to the ordinary static item-model route,
        // not merely free of the dynamic map texture. This catches a partial fix that leaves a
        // map-specific icon branch in the count/overlay helper.
        let mut actual = DrawList::new();
        s.draw_item_with_count_override(
            &mut actual,
            &mut uniform_font(),
            10,
            20,
            s.container.slot_stack(0).expect("filled map slot"),
            None,
        );
        let mut expected = DrawList::new();
        render::draw_item_in_slot_at_z_level(
            &mut expected,
            &s.atlas,
            &mut uniform_font(),
            10,
            20,
            s.container.slot_stack(0).expect("filled map slot"),
            render::GUI_SLOT_ITEM_RENDER_Z_LEVEL,
        );
        assert_eq!(
            actual, expected,
            "GuiContainer slots use RenderItem's static model"
        );
        assert_eq!(s.map_ids(), vec![-7]);
    }

    #[test]
    fn a_tooltip_is_suppressed_while_a_stack_is_held() {
        let mut s = opened(ContainerScreen::chest(1, 3, "Chest"));
        s.container.set_slot_contents(0, Some(stack("apple", 1)));
        let mut font = uniform_font();
        let slot = s.container.slots()[0];
        let point = (s.gui_left() + slot.x + 8, s.gui_top() + slot.y + 8);
        s.draw(&mut DrawList::new(), &mut font, &ctx(), point);
        assert!(s.hovered_tooltip().is_some(), "hovering a filled slot");

        s.container.set_cursor(Some(stack("stone", 1)));
        assert!(
            s.hovered_tooltip().is_none(),
            "vanilla hides the tooltip while the cursor holds a stack"
        );
    }

    #[test]
    fn an_empty_slot_has_no_tooltip() {
        let mut s = opened(ContainerScreen::chest(1, 3, "Chest"));
        let mut font = uniform_font();
        let slot = s.container.slots()[0];
        s.draw(
            &mut DrawList::new(),
            &mut font,
            &ctx(),
            (s.gui_left() + slot.x + 8, s.gui_top() + slot.y + 8),
        );
        assert_eq!(s.hovered_slot(), Some(0));
        assert!(s.hovered_tooltip().is_none());
    }

    #[test]
    fn open_window_dispatch_builds_the_matching_screen() {
        let pkt = s2c::OpenWindow {
            window_id: 4,
            inventory_type: "minecraft:brewing_stand".into(),
            window_title: "{\"text\":\"Brewing Stand\"}".into(),
            slot_count: 4,
            entity_id: None,
        };
        let s = ContainerScreen::for_open_window(&pkt, "Brewing Stand").expect("known type");
        assert_eq!(s.name(), "container.brewing_stand");
        assert_eq!(s.container.slots().len(), 40);
        let horse = s2c::OpenWindow {
            window_id: 5,
            inventory_type: "EntityHorse".into(),
            window_title: "{\"text\":\"Horse\"}".into(),
            slot_count: 2,
            entity_id: Some(42),
        };
        let horse = ContainerScreen::for_open_window(&horse, "Horse").expect("horse");
        assert_eq!(horse.name(), "container.horse");
        assert_eq!(horse.container.slots().len(), 38);
        assert_eq!(horse.art, ContainerArt::Horse { chested: false });
        assert_eq!(horse.horse_entity_id(), Some(42));

        let missing_entity = s2c::OpenWindow {
            entity_id: None,
            ..horse_packet()
        };
        assert!(ContainerScreen::for_open_window(&missing_entity, "Horse").is_none());
    }

    #[test]
    fn horse_preview_rejects_recycled_entity_ids_and_accepts_live_state() {
        let packet = s2c::OpenWindow {
            window_id: 5,
            inventory_type: "EntityHorse".into(),
            window_title: "{\"text\":\"Horse\"}".into(),
            slot_count: 2,
            entity_id: Some(42),
        };
        let mut screen = ContainerScreen::for_open_window(&packet, "Horse").expect("horse");
        let stale = HorsePreviewSnapshot {
            entity_id: 41,
            state: mc_render::entity::models::passive::HorseRenderState {
                saddled: true,
                ..Default::default()
            },
            ..Default::default()
        };
        screen.set_horse_preview_snapshot(Some(stale));
        assert!(screen.horse_preview_snapshot.is_none());

        let live = HorsePreviewSnapshot {
            entity_id: 42,
            state: mc_render::entity::models::passive::HorseRenderState {
                saddled: true,
                ..Default::default()
            },
            ..Default::default()
        };
        screen.set_horse_preview_snapshot(Some(live.clone()));
        assert_eq!(screen.horse_preview_snapshot, Some(live));

        let mut draw = DrawList::new();
        screen.draw(&mut draw, &mut uniform_font(), &ctx(), (0, 0));
        assert!(
            !draw.vertices().is_empty(),
            "the accepted live horse reaches the generic emitter"
        );
    }

    /// java: `NetHandlerPlayClient.handleOpenWindow` followed by
    /// `EntityPlayerSP.displayGUIChest`/`displayGui`.  This is the screen-side counterpart to
    /// the wire table in `net.rs`: every vanilla id must retain its dedicated art and its
    /// packet-derived slot shape, rather than silently becoming a generic chest.
    #[test]
    fn every_vanilla_open_window_id_reaches_the_matching_screen_art() {
        let packet = |inventory_type: &str, slot_count: u8| s2c::OpenWindow {
            window_id: 41,
            inventory_type: inventory_type.to_owned(),
            window_title: format!(r#"{{"text":"{inventory_type}"}}"#),
            slot_count,
            entity_id: None,
        };
        let cases = [
            ("minecraft:container", 54, "container.chest", 90),
            ("minecraft:chest", 27, "container.chest", 63),
            ("minecraft:crafting_table", 0, "container.crafting", 46),
            ("minecraft:furnace", 3, "container.furnace", 39),
            ("minecraft:dispenser", 9, "container.dispenser", 45),
            ("minecraft:dropper", 9, "container.dispenser", 45),
            (
                "minecraft:enchanting_table",
                0,
                "container.enchanting_table",
                38,
            ),
            ("minecraft:brewing_stand", 4, "container.brewing_stand", 40),
            ("minecraft:beacon", 1, "container.beacon", 37),
            ("minecraft:hopper", 5, "container.hopper", 41),
            ("minecraft:anvil", 0, "container.anvil", 39),
            ("minecraft:villager", 3, "container.merchant", 39),
        ];
        for (inventory_type, slot_count, expected_name, expected_slots) in cases {
            let screen = ContainerScreen::for_open_window(
                &packet(inventory_type, slot_count),
                "server title",
            )
            .unwrap_or_else(|| panic!("{inventory_type} must open"));
            assert_eq!(screen.name(), expected_name, "{inventory_type}: art");
            assert_eq!(
                screen.container.slots().len(),
                expected_slots,
                "{inventory_type}: slots"
            );
            assert_eq!(
                screen.container.window_id, 41,
                "{inventory_type}: window id"
            );
        }

        for (slot_count, chested, expected_slots) in [(2, false, 38), (17, true, 53)] {
            let mut horse = packet("EntityHorse", slot_count);
            horse.entity_id = Some(99);
            let screen = ContainerScreen::for_open_window(&horse, "horse").expect("horse");
            assert_eq!(screen.name(), "container.horse");
            assert_eq!(screen.container.slots().len(), expected_slots);
            assert_eq!(screen.art, ContainerArt::Horse { chested });
        }
    }

    fn horse_packet() -> s2c::OpenWindow {
        s2c::OpenWindow {
            window_id: 5,
            inventory_type: "EntityHorse".into(),
            window_title: "{\"text\":\"Horse\"}".into(),
            slot_count: 2,
            entity_id: Some(42),
        }
    }

    // ------------------------------------------------------------ dynamic art

    #[test]
    fn furnace_fire_and_arrow_scale_the_way_vanilla_scales_them() {
        let mut f = FurnaceArt::default();
        assert!(!f.is_burning());
        assert_eq!(f.cook_progress_scaled(24), 0);
        // A zero `currentItemBurnTime` falls back to 200 rather than dividing by zero.
        f.burn_time = 100;
        assert!(f.is_burning());
        assert_eq!(f.burn_left_scaled(13), 100 * 13 / 200);
        f.current_item_burn_time = 1600;
        assert_eq!(f.burn_left_scaled(13), 100 * 13 / 1600);
        f.cook_time = 100;
        f.total_cook_time = 200;
        assert_eq!(f.cook_progress_scaled(24), 12);
    }

    #[test]
    fn window_properties_route_to_the_right_furnace_field() {
        let mut art = ContainerArt::Furnace(FurnaceArt::default());
        for (index, value) in [(0i16, 11i16), (1, 22), (2, 33), (3, 44), (9, 55)] {
            art.apply_window_property(index, value);
        }
        let ContainerArt::Furnace(f) = art else {
            panic!("kind changed")
        };
        assert_eq!(
            f,
            FurnaceArt {
                burn_time: 11,
                current_item_burn_time: 22,
                cook_time: 33,
                total_cook_time: 44,
            },
            "index 9 is unknown and must be ignored, not panic"
        );
    }

    /// Finding F-P0-network.play.server-12-3 covered the apparently dead `S31` fields 3..6.
    /// In 1.8.9 they are not a generic range: furnace field 3 is the full smelt duration, while
    /// enchanting fields 3..6 are the XP seed and the three packed clue ids. Exercise the live
    /// screen packet edge, each value writer, each reset/removal value, the visible readers, and
    /// the window/kind gates together so an accept-all or storage-only implementation cannot pass.
    #[test]
    fn window_properties_3_to_6_reach_their_specialized_live_consumers() {
        let mut furnace = opened(ContainerScreen::furnace(7, "Furnace"));
        furnace.apply_window_property(&s2c::WindowProperty {
            window_id: 7,
            var_index: 2,
            var_value: 100,
        });
        furnace.apply_window_property(&s2c::WindowProperty {
            window_id: 7,
            var_index: 3,
            var_value: 200,
        });
        let ContainerArt::Furnace(art) = &furnace.art else {
            unreachable!();
        };
        assert_eq!(art.total_cook_time, 200);
        assert_eq!(art.cook_progress_scaled(24), 12);

        // A later field update replaces the duration, and zero is the vanilla idle/reset value.
        furnace.apply_window_property(&s2c::WindowProperty {
            window_id: 7,
            var_index: 3,
            var_value: 400,
        });
        let ContainerArt::Furnace(art) = &furnace.art else {
            unreachable!();
        };
        assert_eq!(art.cook_progress_scaled(24), 6);
        furnace.apply_window_property(&s2c::WindowProperty {
            window_id: 7,
            var_index: 3,
            var_value: 0,
        });
        for index in 4..=6 {
            furnace.apply_window_property(&s2c::WindowProperty {
                window_id: 7,
                var_index: index,
                var_value: 99,
            });
        }
        let ContainerArt::Furnace(art) = &furnace.art else {
            unreachable!();
        };
        assert_eq!(art.total_cook_time, 0);
        assert_eq!(art.cook_progress_scaled(24), 0);

        // A foreign window must not overwrite the active specialized state.
        furnace.apply_window_property(&s2c::WindowProperty {
            window_id: 8,
            var_index: 3,
            var_value: 200,
        });
        let ContainerArt::Furnace(art) = &furnace.art else {
            unreachable!();
        };
        assert_eq!(art.total_cook_time, 0);

        let mut enchanting = opened(ContainerScreen::enchanting_table(9, "Enchant"));
        for (index, value) in [
            (3, 12_345),
            (4, 16),
            (5, 16 | (2 << 8)),
            (6, 16 | (3 << 8)),
            (1, 12),
        ] {
            enchanting.apply_window_property(&s2c::WindowProperty {
                window_id: 9,
                var_index: index,
                var_value: value,
            });
        }
        let ContainerArt::Enchanting(art) = &enchanting.art else {
            unreachable!();
        };
        assert_eq!(art.xp_seed, 12_345);
        assert_eq!(art.enchantment_ids, [16, 16 | (2 << 8), 16 | (3 << 8)]);
        assert_eq!(art.enchant_levels[1], 12);
        assert_ne!(
            art.rune_offer_names(),
            EnchantingArt::default().rune_offer_names(),
            "field 3 must feed the rune-name consumer"
        );
        assert!(
            enchanting.enchanting_offer_tooltip(1).is_some(),
            "field 5 must feed the visible clue tooltip"
        );

        // -1 is ContainerEnchantment's unknown-clue sentinel; it removes the row's clue rather
        // than leaving stale text available to the hover consumer. A zero seed is also a real
        // update, not a missing packet.
        enchanting.apply_window_property(&s2c::WindowProperty {
            window_id: 9,
            var_index: 5,
            var_value: -1,
        });
        enchanting.apply_window_property(&s2c::WindowProperty {
            window_id: 9,
            var_index: 3,
            var_value: 0,
        });
        let ContainerArt::Enchanting(art) = &enchanting.art else {
            unreachable!();
        };
        assert_eq!(art.xp_seed, 0);
        assert_eq!(art.enchantment_ids[1], -1);
        assert!(enchanting.enchanting_offer_tooltip(1).is_none());

        enchanting.apply_window_property(&s2c::WindowProperty {
            window_id: 10,
            var_index: 6,
            var_value: 16,
        });
        let ContainerArt::Enchanting(art) = &enchanting.art else {
            unreachable!();
        };
        // The packet addressed window 10 was ignored, so the original packed clue remains:
        // `16 | (3 << 8) == 784`.
        assert_eq!(art.enchantment_ids[2], 16 | (3 << 8));
    }

    /// Vanilla's bubble column is a seven-entry lookup table with uneven steps, not a formula;
    /// a linear approximation would animate visibly differently.
    #[test]
    fn brewing_bubbles_follow_vanillas_uneven_seven_step_table() {
        let heights: Vec<i32> = (0..7)
            .map(|i| {
                BrewingStandArt {
                    brew_time: i * 2 + 2,
                }
                .bubble_height()
            })
            .collect();
        assert_eq!(heights, vec![24, 20, 16, 11, 6, 0, 29]);
        assert_eq!(BrewingStandArt { brew_time: 0 }.bubble_height(), 0);
    }

    #[test]
    fn the_brew_arrow_fills_as_the_timer_counts_down_from_four_hundred() {
        assert_eq!(BrewingStandArt { brew_time: 400 }.arrow_height(), 0);
        assert_eq!(BrewingStandArt { brew_time: 200 }.arrow_height(), 14);
        assert_eq!(BrewingStandArt { brew_time: 1 }.arrow_height(), 27);
        assert_eq!(BrewingStandArt { brew_time: 0 }.arrow_height(), 0);
    }

    #[test]
    fn an_enchant_offer_needs_both_enough_lapis_and_enough_levels() {
        let e = EnchantingArt {
            enchant_levels: [1, 5, 30],
            player_level: 10,
            ..EnchantingArt::default()
        };
        assert!(e.is_affordable(0, 3), "1 lapis, level 1");
        assert!(e.is_affordable(1, 3), "2 lapis, level 5");
        assert!(!e.is_affordable(2, 3), "level 30 > player level 10");
        assert!(!e.is_affordable(1, 1), "offer 2 needs 2 lapis, only 1 held");
        let creative = EnchantingArt {
            creative: true,
            ..e
        };
        assert!(creative.is_affordable(2, 0), "creative bypasses both gates");
        assert!(
            !creative.is_affordable(0, 0) || creative.enchant_levels[0] != 0,
            "a zero-level offer is never affordable"
        );
    }

    #[test]
    fn a_zero_level_enchant_offer_is_never_affordable_even_in_creative() {
        let e = EnchantingArt {
            enchant_levels: [0, 0, 0],
            creative: true,
            player_level: 99,
            ..EnchantingArt::default()
        };
        for row in 0..3 {
            assert!(!e.is_affordable(row, 64));
        }
    }

    /// `GuiEnchantment` reseeds one Java `Random` from the signed `xpSeed`, consumes its
    /// 3-or-4-word name for all three rows, and only then branches on `enchantLevels[row]`.
    /// These fixed outputs bite a wrong RNG, unsigned seed widening, a wrong word order/count,
    /// or skipping the zero-level row's otherwise invisible draw-time consumption.
    #[test]
    fn enchant_rune_names_match_vanillas_seeded_three_offer_sequence() {
        let seed_zero = EnchantingArt {
            xp_seed: 0,
            ..EnchantingArt::default()
        };
        assert_eq!(
            seed_zero.rune_offer_names(),
            [
                "galvanize ignite imbue beast",
                "elemental xyzzy stretch",
                "scrolls scrolls earth imbue",
            ]
        );

        // Negative control for treating S31's signed short/int value as an unsigned seed.
        let negative_seed = EnchantingArt {
            xp_seed: -1,
            ..EnchantingArt::default()
        };
        assert_eq!(
            negative_seed.rune_offer_names(),
            [
                "ball water creature",
                "shorten range shorten free",
                "klaatu twist air",
            ]
        );
        assert_ne!(
            negative_seed.rune_offer_names(),
            seed_zero.rune_offer_names()
        );

        // A missing offer still consumes its name before Java checks `l1 == 0`; otherwise rows
        // below it would drift for the same seed when a server changes only the offer levels.
        let with_middle_offer_absent = EnchantingArt {
            xp_seed: 12_345,
            enchant_levels: [1, 0, 30],
            ..EnchantingArt::default()
        };
        assert_eq!(
            with_middle_offer_absent.rune_offer_names(),
            [
                "beast xyzzy grow",
                "stale ball cold sphere",
                "undead undead inside",
            ]
        );
    }

    #[test]
    fn enchanting_offer_draw_uses_sga_batches_before_ascii_level_text() {
        let mut screen = eligible_enchanting_screen();
        let left = screen.gui_left;
        let top = screen.gui_top;
        let mouse = (left + 61, top + 15);
        let mut draw = DrawList::new();
        let mut ordinary = uniform_font();
        let mut sga = uniform_font_with_texture(FontTexture::StandardGalactic);
        screen.draw_with_map_records_and_sga(
            &mut draw,
            &mut ordinary,
            &mut sga,
            &ctx(),
            mouse,
            None,
            Vec::new(),
        );

        let sga_batches: Vec<_> = draw
            .batches()
            .iter()
            .enumerate()
            .filter(|(_, batch)| batch.texture == UiTexture::Font(FontTexture::StandardGalactic))
            .collect();
        assert_eq!(sga_batches.len(), 3, "one rune batch per offer row");
        let expected_colors = [16_777_088, 6_839_882, 6_839_882];
        for (row, (_, batch)) in sga_batches.iter().enumerate() {
            let first = draw.vertices()[batch.range.start];
            assert_eq!(
                first.pos,
                [left as f32 + 80.0, (top + 16 + 19 * row as i32) as f32]
            );
            assert_eq!(
                first.color,
                crate::widget::unpack_argb(expected_colors[row] as u32 | 0xFF00_0000)
            );
        }
        let first_sga = sga_batches[0].0;
        let last_sga = sga_batches[2].0;
        assert!(draw.batches()[..first_sga].iter().any(|batch| {
            batch.texture == UiTexture::gui("textures/gui/container/enchanting_table.png")
        }));
        assert!(draw.batches()[last_sga + 1..]
            .iter()
            .any(|batch| batch.texture == UiTexture::Font(FontTexture::Ascii)));
    }

    /// `GuiEnchantment.func_147068_g` is the screen-side part of the animated book. The render
    /// owner is deliberately not involved here: these controls pin Java's seeded target reroll,
    /// spring ordering, open/close clamp and negative-value page reduction before a pose is
    /// handed to `ModelBook`.
    #[test]
    fn enchanting_book_animation_matches_java_spring_and_pose_math() {
        let item = stack("diamond_sword", 1);
        let mut book = EnchantingBookAnimation::default();

        // Java Random(0): the first target loop consumes (2,3), then (0,2), yielding -3.
        // The item-change branch must run only once, before the page spring update.
        book.tick(Some(&item), &[1, 0, 0]);
        assert_eq!(book.ticks, 1);
        assert_eq!(book.page_flip_target, -3.0);
        assert!((book.page_flip - (-0.18)).abs() < 1e-6);
        assert!((book.page_flip_velocity - (-0.18)).abs() < 1e-6);
        assert_eq!(book.spread, 0.2);
        assert_eq!(book.page_flip_prev, 0.0);
        assert_eq!(book.spread_prev, 0.0);

        // An equal snapshot does not reroll the target. A count change is part of Java's
        // areItemStacksEqual and therefore does reroll it (the next target is -2 for Random(0)).
        book.tick(Some(&item), &[1, 0, 0]);
        assert_eq!(book.page_flip_target, -3.0);
        let changed_count = stack("diamond_sword", 2);
        book.tick(Some(&changed_count), &[1, 0, 0]);
        assert_eq!(book.page_flip_target, -2.0);

        // Reset and run the deterministic sequence used by the Java oracle through four open
        // and four close ticks. The latter reaches a negative page value, where a plain Rust
        // float-to-int cast would disagree with MathHelper.truncateDoubleToInt's +1024 trick.
        let mut book = EnchantingBookAnimation::default();
        for _ in 0..4 {
            book.tick(Some(&item), &[1, 0, 0]);
        }
        for _ in 0..4 {
            book.tick(Some(&item), &[0, 0, 0]);
        }
        assert_eq!(book.ticks, 8);
        assert!((book.page_flip - (-1.5777779)).abs() < 1e-6);
        assert!((book.spread - 2.9802322e-8).abs() < 1e-6);
        let pose = book.pose(0.5);
        assert!((pose.spread - 0.10000003).abs() < 1e-6);
        assert!((pose.page_flip_right - 0.9355555).abs() < 1e-6);
        assert!((pose.page_flip_left - 0.13555545).abs() < 1e-6);

        // Once offers disappear, Java keeps moving the opening spring toward zero and clamps it.
        book.tick(Some(&item), &[0, 0, 0]);
        assert_eq!(book.spread, 0.0);
    }

    #[test]
    fn enchanting_book_pose_is_screen_owned_but_absent_for_other_containers() {
        let mut enchanting = opened(ContainerScreen::enchanting_table(1, "Enchant"));
        let item = stack("diamond_sword", 1);
        enchanting.container.set_slot_contents(0, Some(item));
        if let ContainerArt::Enchanting(art) = &mut enchanting.art {
            art.enchant_levels[0] = 1;
        }
        enchanting.tick();
        let pose = enchanting
            .enchanting_book_pose(0.5)
            .expect("enchanting screens expose the book pose");
        assert!((pose.spread - 0.1).abs() < 1e-6);

        let chest = opened(ContainerScreen::chest(1, 3, "Chest"));
        assert!(chest.enchanting_book_pose(0.5).is_none());
    }

    #[test]
    fn enchanting_screen_draw_consumes_book_pose_between_panel_and_offer_plates() {
        let mut screen = opened(ContainerScreen::enchanting_table(1, "Enchant"));
        screen
            .container
            .set_slot_contents(0, Some(stack("diamond_sword", 1)));
        if let ContainerArt::Enchanting(art) = &mut screen.art {
            art.enchant_levels = [1, 5, 30];
            art.player_level = 30;
        }
        screen.tick();

        let mut draw = DrawList::new();
        let mut font = uniform_font();
        let frame = ctx();
        screen.draw(&mut draw, &mut font, &frame, (0, 0));

        let book_texture = UiTexture::gui("textures/entity/enchanting_table_book.png");
        let book_batch = draw
            .batches()
            .iter()
            .find(|batch| batch.texture == book_texture)
            .expect("the enchanting screen submits the ModelBook texture");
        assert_eq!(
            book_batch.range.len(),
            252,
            "seven ModelBook parts are projected"
        );
        let book_vertices = &draw.vertices()[book_batch.range.clone()];
        assert!(book_vertices.iter().all(|vertex| {
            vertex.pos.iter().all(|value| value.is_finite())
                && vertex.uv.iter().all(|value| value.is_finite())
                && vertex.color.iter().all(|value| value.is_finite())
        }));
        assert!(book_vertices.iter().any(|vertex| {
            (0.0..=frame.width as f32).contains(&vertex.pos[0])
                && (0.0..=frame.height as f32).contains(&vertex.pos[1])
        }));

        let panel_texture = UiTexture::gui("textures/gui/container/enchanting_table.png");
        let panel_batches: Vec<_> = draw
            .batches()
            .iter()
            .enumerate()
            .filter(|(_, batch)| batch.texture == panel_texture)
            .map(|(index, _)| index)
            .collect();
        let book_batch_index = draw
            .batches()
            .iter()
            .position(|batch| batch.texture == book_texture)
            .expect("book batch index");
        assert!(panel_batches.iter().any(|index| *index < book_batch_index));
        assert!(panel_batches.iter().any(|index| *index > book_batch_index));

        // A closed offer state still emits the book, and the next open tick changes its projected
        // geometry. This bites a draw path that merely records the pose without consuming it.
        let before = book_vertices.to_vec();
        screen.tick();
        let mut next = DrawList::new();
        let mut next_font = uniform_font();
        screen.draw(&mut next, &mut next_font, &frame, (0, 0));
        let next_batch = next
            .batches()
            .iter()
            .find(|batch| batch.texture == book_texture)
            .expect("book remains submitted after the next tick");
        assert!(before
            .iter()
            .zip(&next.vertices()[next_batch.range.clone()])
            .any(|(old, new)| old.pos != new.pos));
    }

    fn eligible_enchanting_screen() -> ContainerScreen {
        let mut screen = opened(ContainerScreen::enchanting_table(3, "Enchant"));
        screen
            .container
            .set_slot_contents(0, Some(stack("diamond_sword", 1)));
        screen.container.set_slot_contents(1, Some(stack("dye", 3)));
        let ContainerArt::Enchanting(art) = &mut screen.art else {
            unreachable!();
        };
        art.enchant_levels = [1, 5, 30];
        art.player_level = 30;
        screen
    }

    /// java: `GuiEnchantment.mouseClicked` runs after the ordinary container click and its row
    /// loop has no mouse-button guard. The plate is panel art rather than a slot, so successful
    /// local action is exactly one queued C11 and no C0E prediction.
    #[test]
    fn an_eligible_enchant_plate_queues_c11_for_any_mouse_button() {
        let mut screen = eligible_enchanting_screen();
        let point = (screen.gui_left() + 60, screen.gui_top() + 14 + 19);
        screen.mouse_clicked_with_mods(point.0, point.1, 2, Modifiers::NONE);

        assert!(
            screen.take_outgoing().is_empty(),
            "a plate click is not C0E"
        );
        assert_eq!(
            screen.take_enchant_outgoing(),
            vec![c2s::EnchantItem {
                window_id: 3,
                button: 1,
            }]
        );
        assert_eq!(
            screen.container.slot_stack(1).map(|stack| stack.count),
            Some(3),
            "the remote-world predicate must leave lapis mutation to the server"
        );
    }

    /// Java queues C0E/C11 at each callback site, so a same-frame slot click followed by an
    /// enchanting plate click must retain that interleaving on the eventual session drain.
    #[test]
    fn slot_and_enchant_actions_share_one_invocation_order_queue() {
        let mut screen = eligible_enchanting_screen();
        // Use a player-inventory slot so the slot click does not invalidate the enchanting input
        // before the plate callback runs. `dispatch` is the same local prediction seam used by
        // mouse/key input and always allocates the next transaction id.
        screen.dispatch(2, 0, 0);
        let point = (screen.gui_left() + 60, screen.gui_top() + 14 + 19);
        screen.mouse_clicked_with_mods(point.0, point.1, 2, Modifiers::NONE);

        let packets = screen.take_ordered_outgoing();
        assert_eq!(packets.len(), 2);
        assert!(matches!(
            packets[0],
            mc_protocol::play::C2SPlay::ClickWindow(_)
        ));
        assert!(matches!(
            packets[1],
            mc_protocol::play::C2SPlay::EnchantItem(_)
        ));
    }

    /// Exercise every conjunct in client-side `ContainerEnchantment.enchantItem`, including the
    /// less-obvious `experienceLevel >= row + 1` check, then prove creative bypasses only the
    /// payment gates (not positive-offer/item existence).
    #[test]
    fn enchant_activation_rejects_unaffordable_or_invalid_rows() {
        let mut screen = eligible_enchanting_screen();
        let row = |screen: &ContainerScreen, index: i32| {
            (screen.gui_left() + 61, screen.gui_top() + 15 + 19 * index)
        };

        screen.container.set_slot_contents(1, Some(stack("dye", 1)));
        let point = row(&screen, 1);
        screen.mouse_clicked(point.0, point.1, 0);
        assert!(
            screen.take_enchant_outgoing().is_empty(),
            "row 2 needs two lapis"
        );

        screen.container.set_slot_contents(1, Some(stack("dye", 3)));
        let ContainerArt::Enchanting(art) = &mut screen.art else {
            unreachable!();
        };
        art.player_level = 1;
        screen.mouse_clicked(point.0, point.1, 0);
        assert!(
            screen.take_enchant_outgoing().is_empty(),
            "row 2 needs level >= 2 and >= 5"
        );

        let ContainerArt::Enchanting(art) = &mut screen.art else {
            unreachable!();
        };
        art.player_level = 30;
        art.enchant_levels[1] = 0;
        screen.mouse_clicked(point.0, point.1, 0);
        assert!(
            screen.take_enchant_outgoing().is_empty(),
            "zero means no offer"
        );

        let ContainerArt::Enchanting(art) = &mut screen.art else {
            unreachable!();
        };
        art.enchant_levels[1] = 5;
        screen.container.set_slot_contents(0, None);
        screen.mouse_clicked(point.0, point.1, 0);
        assert!(
            screen.take_enchant_outgoing().is_empty(),
            "slot 0 must contain an item"
        );

        screen.creative = true;
        screen
            .container
            .set_slot_contents(0, Some(stack("diamond_sword", 1)));
        screen.container.set_slot_contents(1, None);
        let ContainerArt::Enchanting(art) = &mut screen.art else {
            unreachable!();
        };
        art.player_level = 0;
        screen.mouse_clicked(point.0, point.1, 7);
        assert_eq!(screen.take_enchant_outgoing().len(), 1);
    }

    /// The click rectangles are raw `< 108`, not `isPointInRegion`'s one-pixel expansion.
    #[test]
    fn enchant_plate_clicks_do_not_grow_past_the_raw_rectangle() {
        let mut screen = eligible_enchanting_screen();
        let left = screen.gui_left() + 60;
        let top = screen.gui_top() + 14;
        for point in [(left - 1, top), (left + 108, top), (left, top + 3 * 19)] {
            screen.mouse_clicked(point.0, point.1, 0);
        }
        assert!(screen.take_enchant_outgoing().is_empty());
    }

    #[test]
    fn untouched_enchanting_clues_start_unknown_not_as_protection_zero() {
        assert_eq!(EnchantingArt::default().enchantment_ids, [-1; 3]);
    }

    /// java: `ContainerEnchantment.onCraftMatrixChanged` keeps its random offer calculation
    /// server-side, but its invalid-input `else` clears both arrays on the client too. Exercise
    /// the real screen dispatch path so this cannot pass with only a detached helper implemented.
    #[test]
    fn a_predicted_invalid_enchanting_input_clears_stale_offers_immediately() {
        fn seed_offers(screen: &mut ContainerScreen) {
            let ContainerArt::Enchanting(art) = &mut screen.art else {
                panic!("enchanting screen");
            };
            art.enchant_levels = [5, 12, 30];
            art.enchantment_ids = [16, 16 | (2 << 8), 16 | (3 << 8)];
        }

        let mut removed = opened(ContainerScreen::enchanting_table(1, "Enchant"));
        removed
            .container
            .set_slot_contents(0, Some(stack("diamond_sword", 1)));
        seed_offers(&mut removed);
        let item_slot = removed.container.slots()[0];
        removed.mouse_clicked_with_mods(
            removed.gui_left() + item_slot.x + 8,
            removed.gui_top() + item_slot.y + 8,
            0,
            Modifiers::NONE,
        );
        let ContainerArt::Enchanting(art) = &removed.art else {
            unreachable!();
        };
        assert_eq!(art.enchant_levels, [0; 3]);
        assert_eq!(art.enchantment_ids, [-1; 3]);

        let mut stone = opened(ContainerScreen::enchanting_table(1, "Enchant"));
        stone.container.set_cursor(Some(stack("stone", 1)));
        seed_offers(&mut stone);
        let item_slot = stone.container.slots()[0];
        let point = (
            stone.gui_left() + item_slot.x + 8,
            stone.gui_top() + item_slot.y + 8,
        );
        stone.mouse_clicked(point.0, point.1, 0);
        stone.mouse_dragged(point.0, point.1, 0, 10);
        stone.mouse_released(point.0, point.1, 0);
        assert_eq!(
            stone
                .container
                .slot_stack(0)
                .map(|stack| stack_ext::props(stack).name),
            Some("minecraft:stone")
        );
        let ContainerArt::Enchanting(art) = &stone.art else {
            unreachable!();
        };
        assert_eq!(art.enchant_levels, [0; 3]);
        assert_eq!(art.enchantment_ids, [-1; 3]);
    }

    /// The negative control covers the branch that must *not* clear the server's offers and the
    /// two 1.8.9 overrides that make the predicate less obvious than `damageable`.
    #[test]
    fn a_predicted_enchantable_input_preserves_offers_and_matches_item_overrides() {
        use mc_nbt::{Nbt, NbtCompound};

        let mut screen = opened(ContainerScreen::enchanting_table(1, "Enchant"));
        screen.container.set_cursor(Some(stack("diamond_sword", 1)));
        let ContainerArt::Enchanting(art) = &mut screen.art else {
            unreachable!();
        };
        art.enchant_levels = [5, 12, 30];
        art.enchantment_ids = [16, 16 | (2 << 8), 16 | (3 << 8)];
        screen.dispatch(0, 0, 0);
        let ContainerArt::Enchanting(art) = &screen.art else {
            unreachable!();
        };
        assert_eq!(art.enchant_levels, [5, 12, 30]);
        assert_eq!(art.enchantment_ids, [16, 16 | (2 << 8), 16 | (3 << 8)]);

        // `onCraftMatrixChanged` is an InventoryBasic callback, not a blanket post-click hook:
        // moving an unrelated player-inventory stack must not rewrite the arrays by itself.
        let mut unrelated = opened(ContainerScreen::enchanting_table(1, "Enchant"));
        unrelated
            .container
            .set_slot_contents(2, Some(stack("stone", 1)));
        let ContainerArt::Enchanting(art) = &mut unrelated.art else {
            unreachable!();
        };
        art.enchant_levels = [5, 12, 30];
        art.enchantment_ids = [16, 16 | (2 << 8), 16 | (3 << 8)];
        unrelated.dispatch(2, 0, 0);
        let ContainerArt::Enchanting(art) = &unrelated.art else {
            unreachable!();
        };
        assert_eq!(art.enchant_levels, [5, 12, 30]);
        assert_eq!(art.enchantment_ids, [16, 16 | (2 << 8), 16 | (3 << 8)]);

        assert!(is_item_enchantable(&stack("book", 1)));
        assert!(!is_item_enchantable(&stack("book", 2)));
        assert!(!is_item_enchantable(&stack("enchanted_book", 1)));

        let mut enchanted = stack("diamond_sword", 1);
        let mut root = NbtCompound::new();
        root.insert("ench", Nbt::list(Vec::new()));
        enchanted.nbt = Some(root);
        assert!(!is_item_enchantable(&enchanted));
    }

    /// java: `NetHandlerPlayClient.handleSetSlot` -> `Container.putStackInSlot` -> the table
    /// inventory's `markDirty`. A server correction can therefore invalidate offers without a
    /// local click; routing only through `Container::apply_set_slot` leaves stale plates and
    /// clues visible. Cursor and foreign-window packets are deliberately no-ops for this hook.
    #[test]
    fn server_set_slot_resync_clears_only_invalid_enchanting_table_input() {
        fn seed_offers(screen: &mut ContainerScreen) {
            let ContainerArt::Enchanting(art) = &mut screen.art else {
                panic!("enchanting screen");
            };
            art.enchant_levels = [5, 12, 30];
            art.enchantment_ids = [16, 16 | (2 << 8), 16 | (3 << 8)];
        }

        let mut invalid = opened(ContainerScreen::enchanting_table(1, "Enchant"));
        invalid
            .container
            .set_slot_contents(0, Some(stack("diamond_sword", 1)));
        seed_offers(&mut invalid);
        invalid.apply_set_slot(&s2c::SetSlot {
            window_id: 1,
            slot: 0,
            item: Some(stack("stone", 1)),
        });
        let ContainerArt::Enchanting(art) = &invalid.art else {
            unreachable!();
        };
        assert_eq!(art.enchant_levels, [0; 3]);
        assert_eq!(art.enchantment_ids, [-1; 3]);

        // Matching-window slot 1 is part of the same InventoryBasic callback, even though it
        // cannot itself make an item enchantable. This control catches a hook that watches slot 0
        // only and would leave stale offers after the table's lapis slot is synchronized.
        let mut lapis = opened(ContainerScreen::enchanting_table(1, "Enchant"));
        lapis
            .container
            .set_slot_contents(0, Some(stack("stone", 1)));
        seed_offers(&mut lapis);
        lapis.apply_set_slot(&s2c::SetSlot {
            window_id: 1,
            slot: 1,
            item: Some(stack("dye", 1)),
        });
        let ContainerArt::Enchanting(art) = &lapis.art else {
            unreachable!();
        };
        assert_eq!(art.enchant_levels, [0; 3]);
        assert_eq!(art.enchantment_ids, [-1; 3]);

        // A valid replacement preserves the server's offers, just like the client-side
        // `onCraftMatrixChanged` branch does when slot 0 is enchantable.
        let mut valid = opened(ContainerScreen::enchanting_table(1, "Enchant"));
        valid
            .container
            .set_slot_contents(0, Some(stack("stone", 1)));
        seed_offers(&mut valid);
        valid.apply_set_slot(&s2c::SetSlot {
            window_id: 1,
            slot: 0,
            item: Some(stack("diamond_sword", 1)),
        });
        let ContainerArt::Enchanting(art) = &valid.art else {
            unreachable!();
        };
        assert_eq!(art.enchant_levels, [5, 12, 30]);
        assert_eq!(art.enchantment_ids, [16, 16 | (2 << 8), 16 | (3 << 8)]);

        // The live host currently feeds packet state into `screen.container` directly. The
        // model edge is consumed at the next frame, so this path remains faithful without a
        // client-driver-specific callback.
        let mut direct = opened(ContainerScreen::enchanting_table(1, "Enchant"));
        direct
            .container
            .set_slot_contents(0, Some(stack("diamond_sword", 1)));
        seed_offers(&mut direct);
        direct.container.apply_set_slot(&s2c::SetSlot {
            window_id: 1,
            slot: 0,
            item: Some(stack("stone", 1)),
        });
        let ContainerArt::Enchanting(art) = &direct.art else {
            unreachable!();
        };
        assert_eq!(art.enchant_levels, [5, 12, 30]);
        direct.draw(&mut DrawList::new(), &mut uniform_font(), &ctx(), (0, 0));
        let ContainerArt::Enchanting(art) = &direct.art else {
            unreachable!();
        };
        assert_eq!(art.enchant_levels, [0; 3]);
        assert_eq!(art.enchantment_ids, [-1; 3]);

        // `windowId == -1` is the cursor stack, and a different numbered window is unrelated;
        // neither may trigger the table callback or erase currently valid offers.
        seed_offers(&mut valid);
        valid.apply_set_slot(&s2c::SetSlot {
            window_id: -1,
            slot: 0,
            item: Some(stack("stone", 1)),
        });
        valid.apply_set_slot(&s2c::SetSlot {
            window_id: 2,
            slot: 0,
            item: Some(stack("stone", 1)),
        });
        let ContainerArt::Enchanting(art) = &valid.art else {
            unreachable!();
        };
        assert_eq!(art.enchant_levels, [5, 12, 30]);
        assert_eq!(art.enchantment_ids, [16, 16 | (2 << 8), 16 | (3 << 8)]);
    }

    /// java: `NetHandlerPlayClient.handleWindowItems` -> `Container.putStacks` invokes the same
    /// table-inventory dirty callback while a bulk resync replaces the slots. The window gate and
    /// empty-list guard mirror the packet handler's loop and keep foreign/no-op packets inert.
    #[test]
    fn server_window_items_resync_clears_invalid_enchanting_input() {
        fn seed_offers(screen: &mut ContainerScreen) {
            let ContainerArt::Enchanting(art) = &mut screen.art else {
                panic!("enchanting screen");
            };
            art.enchant_levels = [5, 12, 30];
            art.enchantment_ids = [16, 16 | (2 << 8), 16 | (3 << 8)];
        }

        let mut screen = opened(ContainerScreen::enchanting_table(1, "Enchant"));
        screen
            .container
            .set_slot_contents(0, Some(stack("diamond_sword", 1)));
        seed_offers(&mut screen);
        let mut items = vec![None; screen.container.slots().len()];
        items[0] = Some(stack("stone", 1));
        screen.apply_window_items(&s2c::WindowItems {
            window_id: 1,
            items,
        });
        let ContainerArt::Enchanting(art) = &screen.art else {
            unreachable!();
        };
        assert_eq!(art.enchant_levels, [0; 3]);
        assert_eq!(art.enchantment_ids, [-1; 3]);

        // As with SetSlot, a caller may feed the packet to the model member directly; the draw
        // boundary consumes the same edge and applies the client-side clear.
        let mut direct = opened(ContainerScreen::enchanting_table(1, "Enchant"));
        direct
            .container
            .set_slot_contents(0, Some(stack("diamond_sword", 1)));
        seed_offers(&mut direct);
        let mut direct_items = vec![None; direct.container.slots().len()];
        direct_items[0] = Some(stack("stone", 1));
        direct.container.apply_window_items(&s2c::WindowItems {
            window_id: 1,
            items: direct_items,
        });
        direct.draw(&mut DrawList::new(), &mut uniform_font(), &ctx(), (0, 0));
        let ContainerArt::Enchanting(art) = &direct.art else {
            unreachable!();
        };
        assert_eq!(art.enchant_levels, [0; 3]);
        assert_eq!(art.enchantment_ids, [-1; 3]);

        // A packet for another window must not overwrite slots or clear the active offers.
        seed_offers(&mut screen);
        let mut foreign_items = vec![None; screen.container.slots().len()];
        foreign_items[0] = Some(stack("stone", 1));
        screen.apply_window_items(&s2c::WindowItems {
            window_id: 2,
            items: foreign_items,
        });
        let ContainerArt::Enchanting(art) = &screen.art else {
            unreachable!();
        };
        assert_eq!(art.enchant_levels, [5, 12, 30]);
        assert_eq!(art.enchantment_ids, [16, 16 | (2 << 8), 16 | (3 << 8)]);

        // A matching empty list performs no `putStackInSlot` calls and therefore cannot invoke
        // the callback on its own.
        screen.apply_window_items(&s2c::WindowItems {
            window_id: 1,
            items: Vec::new(),
        });
        let ContainerArt::Enchanting(art) = &screen.art else {
            unreachable!();
        };
        assert_eq!(art.enchant_levels, [5, 12, 30]);
        assert_eq!(art.enchantment_ids, [16, 16 | (2 << 8), 16 | (3 << 8)]);
    }

    /// java: `GuiEnchantment.drawScreen:232-304`. This drives the real expanded 108x17 row hit
    /// box through `draw`, then reads the tooltip the screen would paint. Before this port,
    /// `enchantment_ids` had no reader and the result was `None` for every row.
    #[test]
    fn hovering_an_enchant_offer_shows_the_clue_and_payment_requirements() {
        let mut s = opened(ContainerScreen::enchanting_table(1, "Enchant"));
        s.container.set_slot_contents(1, Some(stack("dye", 2)));
        let ContainerArt::Enchanting(art) = &mut s.art else {
            unreachable!();
        };
        art.enchant_levels[1] = 12;
        art.enchantment_ids[1] = 16 | (3 << 8); // Sharpness III
        art.player_level = 30;

        let mouse = (s.gui_left() + 60, s.gui_top() + 14 + 19 + 8);
        let mut d = DrawList::new();
        s.draw(&mut d, &mut uniform_font(), &ctx(), mouse);
        assert_eq!(
            s.hovered_tooltip(),
            Some(vec![
                "\u{a7}f\u{a7}oSharpness III . . . ?".to_string(),
                String::new(),
                "\u{a7}72 Lapis Lazuli".to_string(),
                "\u{a7}72 Enchantment Levels".to_string(),
            ])
        );
        s.container.set_cursor(Some(stack("stone", 1)));
        assert!(
            s.hovered_tooltip().is_some(),
            "GuiEnchantment draws its offer tooltip after the base cursor guard"
        );
        s.container.set_cursor(None);

        let ContainerArt::Enchanting(art) = &mut s.art else {
            unreachable!();
        };
        art.player_level = 5;
        assert_eq!(
            s.enchanting_offer_tooltip(1).unwrap()[2],
            "\u{a7}cLevel Requirement: 12"
        );

        // Negative controls: outside the offer plate, and an offer whose clue is unknown.
        s.hovered_enchant_offer = None;
        assert!(s.hovered_tooltip().is_none());
        let ContainerArt::Enchanting(art) = &mut s.art else {
            unreachable!();
        };
        art.enchantment_ids[1] = -1;
        assert!(s.enchanting_offer_tooltip(1).is_none());
    }

    /// The tooltip is a separate `drawScreen` pass: it uses the expanded 108x17
    /// `isPointInRegion` bounds (not the background plate's raw 108x19 bounds), and I18n owns
    /// every non-literal fragment. These branch controls cover unavailable lapis, unavailable
    /// XP, and creative mode, where Java shows the clue but no payment lines.
    #[test]
    fn enchant_offer_tooltip_uses_localized_strings_and_vanilla_payment_gates() {
        let mut s = opened(ContainerScreen::enchanting_table(1, "Enchant"));
        let mut language = Language::new("test");
        language.merge_lang_str(concat!(
            "enchantment.damage.all=Keen\n",
            "enchantment.level.3=III\n",
            "container.enchant.clue=Clue: %s\n",
            "container.enchant.lapis.many=%d Blue Gems\n",
            "container.enchant.level.many=%d Costs\n",
        ));
        s.set_language(&language);
        let ContainerArt::Enchanting(art) = &mut s.art else {
            unreachable!();
        };
        art.enchant_levels[2] = 30;
        art.enchantment_ids[2] = 16 | (3 << 8);
        art.player_level = 30;

        let left = s.gui_left() + 60;
        let top = s.gui_top() + 14 + 2 * 19;
        s.draw(
            &mut DrawList::new(),
            &mut uniform_font(),
            &ctx(),
            (left - 1, top - 1),
        );
        assert_eq!(
            s.hovered_tooltip(),
            Some(vec![
                "\u{a7}f\u{a7}oClue: Keen III".to_string(),
                String::new(),
                "\u{a7}c3 Blue Gems".to_string(),
                "\u{a7}73 Costs".to_string(),
            ]),
            "one-pixel expansion is part of GuiContainer.isPointInRegion"
        );
        s.draw(
            &mut DrawList::new(),
            &mut uniform_font(),
            &ctx(),
            (left + 109, top),
        );
        assert!(
            s.hovered_tooltip().is_none(),
            "the expanded right edge is exclusive"
        );

        if let ContainerArt::Enchanting(art) = &mut s.art {
            art.player_level = 29;
        }
        assert_eq!(
            s.enchanting_offer_tooltip(2),
            Some(vec![
                "\u{a7}f\u{a7}oClue: Keen III".to_string(),
                String::new(),
                "\u{a7}cLevel Requirement: 30".to_string(),
            ]),
            "insufficient XP hides both lapis and level-cost lines"
        );
        if let ContainerArt::Enchanting(art) = &mut s.art {
            art.creative = true;
        }
        assert_eq!(
            s.enchanting_offer_tooltip(2),
            Some(vec!["\u{a7}f\u{a7}oClue: Keen III".to_string()]),
            "creative suppresses the blank separator and all payment requirements"
        );
    }

    #[test]
    fn the_anvil_cross_only_appears_with_an_input_and_no_output() {
        let mut s = opened(ContainerScreen::anvil(1, "Repair & Name"));
        let mut base = DrawList::new();
        s.art.draw_background(&mut base, 0, 0, &s.container);
        let empty = base.vertices().len();

        s.container.set_slot_contents(0, Some(stack("stone", 1)));
        let mut with_input = DrawList::new();
        s.art.draw_background(&mut with_input, 0, 0, &s.container);
        assert_eq!(
            with_input.vertices().len(),
            empty + 6,
            "one extra quad: the red cross"
        );

        s.container.set_slot_contents(2, Some(stack("stone", 1)));
        let mut with_output = DrawList::new();
        s.art.draw_background(&mut with_output, 0, 0, &s.container);
        assert_eq!(
            with_output.vertices().len(),
            empty,
            "an output slot removes the cross again"
        );
    }

    /// java: `GuiRepair.initGui:47-51`, `sendSlotContents:200-212` and `renameItem:137-148`.
    /// This covers both mutation controls: an empty slot disables without sending, while an
    /// ordinary input enables and immediately sends the empty-string normalization rather than
    /// redundantly assigning its default display name.
    #[test]
    fn anvil_slot_zero_sync_controls_the_name_field_and_wire_value() {
        let mut s = opened(ContainerScreen::anvil(1, "Repair & Name"));
        let font = uniform_font();
        let field = s.anvil_name.as_ref().expect("anvil field");
        assert_eq!(field.rect.x, s.gui_left() + 62);
        assert_eq!(field.rect.y, s.gui_top() + 24);
        assert_eq!((field.rect.width, field.rect.height), (103, 12));
        assert_eq!(field.max_length(), 30);
        assert!(!field.background);
        assert!(!s.anvil_name_enabled());
        assert!(s.take_anvil_name_changes().is_empty());

        s.sync_anvil_input_name(Some("Apple"), false, &font);
        assert!(s.anvil_name_enabled());
        assert_eq!(s.anvil_rename_text().as_deref(), Some("Apple"));
        assert_eq!(s.take_anvil_name_changes(), vec![String::new()]);

        // A custom display name is not blanked, even when the visible field still equals it.
        s.sync_anvil_input_name(Some("Pomme"), true, &font);
        assert_eq!(s.take_anvil_name_changes(), vec!["Pomme"]);

        s.sync_anvil_input_name(None, false, &font);
        assert!(!s.anvil_name_enabled());
        assert_eq!(s.anvil_rename_text().as_deref(), Some(""));
        assert!(
            s.take_anvil_name_changes().is_empty(),
            "null slot 0 never calls renameItem"
        );

        // Negative control: the same API is inert on every other container kind.
        let mut chest = opened(ContainerScreen::chest(1, 3, "Chest"));
        chest.sync_anvil_input_name(Some("Apple"), false, &font);
        assert_eq!(chest.anvil_rename_text(), None);
        assert!(chest.take_anvil_name_changes().is_empty());
    }

    /// `textboxKeyTyped` returning true is the trigger, not a text-difference comparison.
    /// Therefore Left and Ctrl+C send just like a printable edit; a disabled field consumes the
    /// same keys but cannot mutate its contents.
    #[test]
    fn every_consumed_anvil_key_queues_the_normalized_name() {
        let mut s = opened(ContainerScreen::anvil(1, "Repair & Name"));
        let font = uniform_font();
        s.sync_anvil_input_name(Some("Apple"), false, &font);
        s.take_anvil_name_changes();
        s.anvil_name.as_mut().unwrap().set_focused(true);

        assert_eq!(
            Screen::key_typed_with_font(&mut s, Some('!'), 0, Modifiers::NONE, &font),
            ScreenAction::None
        );
        assert_eq!(s.anvil_rename_text().as_deref(), Some("Apple!"));
        assert_eq!(s.take_anvil_name_changes(), vec!["Apple!"]);

        Screen::key_typed_with_font(&mut s, None, keys::LEFT, Modifiers::NONE, &font);
        assert_eq!(
            s.take_anvil_name_changes(),
            vec!["Apple!"],
            "a consumed navigation key still invokes renameItem"
        );
        Screen::key_typed_with_font(&mut s, None, keys::C, Modifiers::CTRL, &font);
        assert_eq!(s.take_anvil_name_changes(), vec!["Apple!"]);

        s.sync_anvil_input_name(None, false, &font);
        s.anvil_name.as_mut().unwrap().set_focused(true);
        Screen::key_typed_with_font(&mut s, Some('X'), 0, Modifiers::NONE, &font);
        assert_eq!(s.anvil_rename_text().as_deref(), Some(""));
        assert_eq!(
            s.take_anvil_name_changes(),
            vec![String::new()],
            "GuiTextField consumes allowed chars while disabled and GuiRepair still calls renameItem"
        );
    }

    #[test]
    fn clicking_the_anvil_name_plate_focuses_the_live_field() {
        let mut s = opened(ContainerScreen::anvil(1, "Repair & Name"));
        let font = uniform_font();
        s.sync_anvil_input_name(Some("Apple"), false, &font);
        s.take_anvil_name_changes();
        // Drawing records exact glyph advances for the fontless screen-click seam.
        s.draw(&mut DrawList::new(), &mut uniform_font(), &ctx(), (0, 0));
        let x = s.gui_left() + 62 + 2;
        let y = s.gui_top() + 24 + 2;
        s.mouse_clicked_with_mods(x, y, 0, Modifiers::NONE);
        assert!(s.anvil_name.as_ref().unwrap().is_focused());
        assert!(s.take_outgoing().is_empty(), "the plate is not a slot");

        s.mouse_clicked_with_mods(s.gui_left() + 5, s.gui_top() + 5, 0, Modifiers::NONE);
        assert!(!s.anvil_name.as_ref().unwrap().is_focused());
    }

    #[test]
    fn the_anvil_cost_api_is_the_same_three_arm_cascade_the_renderer_uses() {
        let mut s = opened(ContainerScreen::anvil(1, "Repair & Name"));
        if let ContainerArt::Anvil(art) = &mut s.art {
            art.maximum_cost = 5;
        }
        let ContainerArt::Anvil(art) = &s.art else {
            unreachable!()
        };
        assert_eq!(art.cost_label(&s.container), None, "empty output hides it");

        s.container.set_slot_contents(2, Some(stack("stone", 1)));
        assert_eq!(
            s.container.maximum_cost, None,
            "the live wire leaves this unknown"
        );
        s.set_player_experience_level(30);
        let ContainerArt::Anvil(art) = &s.art else {
            unreachable!()
        };
        assert_eq!(
            art.cost_label(&s.container),
            Some(AnvilCostLabel {
                text: "Enchantment Cost: 5".to_owned(),
                color: ANVIL_COST_COLOR,
            })
        );

        s.set_player_experience_level(1);
        let ContainerArt::Anvil(art) = &s.art else {
            unreachable!()
        };
        assert_eq!(
            art.cost_label(&s.container).unwrap().color,
            ANVIL_COST_DENIED_COLOR
        );
        if let ContainerArt::Anvil(art) = &mut s.art {
            art.maximum_cost = 40;
        }
        let ContainerArt::Anvil(art) = &s.art else {
            unreachable!()
        };
        assert_eq!(art.cost_label(&s.container).unwrap().text, "Too Expensive!");
        if let ContainerArt::Anvil(art) = &mut s.art {
            art.creative = true;
        }
        let ContainerArt::Anvil(art) = &s.art else {
            unreachable!()
        };
        assert_eq!(
            art.cost_label(&s.container).unwrap().text,
            "Enchantment Cost: 40"
        );
    }

    /// java: `GuiContainer.drawSlot:277-291` — an empty slot with a non-null
    /// `getSlotTexture()` blits the placeholder and sets `flag1`, which skips the item render.
    /// The hover rectangle is outside `drawSlot`: `drawScreen:127-139` paints it afterwards,
    /// including over the silhouette. The four player armour slots are the only slots with one.
    #[test]
    fn an_empty_armour_slot_draws_its_silhouette_then_the_hover_highlight() {
        let s = opened(ContainerScreen::player(0));
        for (i, name) in [
            "items/empty_armor_slot_helmet",
            "items/empty_armor_slot_chestplate",
            "items/empty_armor_slot_leggings",
            "items/empty_armor_slot_boots",
        ]
        .iter()
        .enumerate()
        {
            assert_eq!(
                s.container.slots()[5 + i].background_sprite,
                Some(*name),
                "armour slot {i}"
            );
        }
        let armour = s.container.slots()[5];
        let mut font = uniform_font();

        let mut plain = DrawList::new();
        s.draw_slot(&mut plain, &mut font, 0, 0, armour, false);
        assert_eq!(plain.vertices().len(), 6, "one placeholder quad");

        let mut hovered = DrawList::new();
        s.draw_slot(&mut hovered, &mut font, 0, 0, armour, true);
        assert_eq!(
            hovered.vertices().len(),
            12,
            "one placeholder quad followed by one highlight quad"
        );

        // Negative control: an ordinary empty slot has no placeholder, so hovering it *does*
        // add the highlight. Without this the test would also pass against "never highlight".
        let plain_slot = s.container.slots()[9];
        assert_eq!(plain_slot.background_sprite, None);
        let mut d = DrawList::new();
        s.draw_slot(&mut d, &mut font, 0, 0, plain_slot, false);
        assert_eq!(d.vertices().len(), 0);
        let mut d = DrawList::new();
        s.draw_slot(&mut d, &mut font, 0, 0, plain_slot, true);
        assert_eq!(d.vertices().len(), 6, "the highlight quad");
    }

    #[test]
    fn an_occupied_armour_slot_suppresses_its_empty_slot_silhouette() {
        let mut s = opened(ContainerScreen::player(0));
        s.container
            .set_slot_contents(5, Some(stack("iron_helmet", 1)));
        let slot = s.container.slots()[5];
        let mut d = DrawList::new();
        // No atlas is installed in this unit-test fixture, and a count of one emits no text;
        // any vertices here therefore prove the empty-slot ghost was drawn over an item.
        s.draw_slot(&mut d, &mut uniform_font(), 0, 0, slot, false);
        assert_eq!(
            d.vertices().len(),
            0,
            "an occupied armour slot must not draw its empty-slot ghost"
        );
    }

    /// java: `GuiContainer.drawScreen:127-139` calls `drawSlot(slot)` first and paints the
    /// 50%-white hover rectangle afterwards. The separate drag-preview rectangle is the one
    /// inside `drawSlot` and therefore below its preview item.
    #[test]
    fn the_hover_highlight_is_drawn_over_the_item() {
        let mut s = opened(ContainerScreen::chest(1, 3, "Chest"));
        // A count of 5 so the stack draws its count text: `mc_assets` has no installed atlas
        // in a unit test, so the icon itself contributes no quads and the text is what proves
        // the item half ran at all.
        s.container.set_slot_contents(4, Some(stack("stone", 5)));
        let slot = s.container.slots()[4];
        let mut d = DrawList::new();
        s.draw_slot(&mut d, &mut uniform_font(), 0, 0, slot, true);
        let v = d.vertices();
        assert!(v.len() > 6, "highlight plus the item's count text");
        for (i, vert) in v.iter().rev().take(6).enumerate() {
            assert!(
                (vert.color[3] - 128.0 / 255.0).abs() < 1e-3,
                "tail vertex {i} should be the 0x80ffffff highlight, got {:?}",
                vert.color
            );
        }
        assert!(
            v[..v.len() - 6]
                .iter()
                .any(|vert| (vert.color[3] - 1.0).abs() < 1e-3),
            "the opaque item precedes the highlight"
        );
    }

    /// java: `GuiRepair.drawGuiContainerForegroundLayer:75-114` is a three-arm cascade, and
    /// only the first arm was ported. The second (`!getSlot(2).getHasStack() -> flag = false`)
    /// suppresses the label entirely; the third
    /// (`!getSlot(2).canTakeStack(player) -> i = 16736352`) reddens it.
    ///
    /// Colours are read straight off the emitted vertices, so the assertion is on what is
    /// actually painted rather than on a helper's return value.
    #[test]
    fn the_anvil_cost_label_runs_vanillas_three_arm_cascade() {
        fn foreground(s: &ContainerScreen) -> DrawList {
            let mut d = DrawList::new();
            s.art.draw_foreground(
                &mut d,
                &mut uniform_font(),
                0,
                0,
                &s.container,
                "Repair & Name",
                "Inventory",
                None,
            );
            d
        }
        fn has_color(d: &DrawList, rgb: i32) -> bool {
            let want = [
                ((rgb >> 16) & 0xFF) as f32 / 255.0,
                ((rgb >> 8) & 0xFF) as f32 / 255.0,
                (rgb & 0xFF) as f32 / 255.0,
            ];
            d.vertices().iter().any(|v| {
                (v.color[0] - want[0]).abs() < 1e-3
                    && (v.color[1] - want[1]).abs() < 1e-3
                    && (v.color[2] - want[2]).abs() < 1e-3
            })
        }

        let mut s = opened(ContainerScreen::anvil(1, "Repair & Name"));
        let title_only = foreground(&s).vertices().len();
        assert!(!has_color(&foreground(&s), ANVIL_COST_COLOR), "cost is 0");

        // Arm 2: a cost, but nothing in the output slot -> nothing drawn beyond the title.
        s.apply_window_property(&s2c::WindowProperty {
            window_id: 1,
            var_index: 0,
            var_value: 5,
        });
        assert!(s.container.slot_stack(2).is_none());
        assert_eq!(
            foreground(&s).vertices().len(),
            title_only,
            "an empty output slot suppresses the whole label"
        );

        // Default arm: an output, a known and affordable cost -> green.
        s.container.set_slot_contents(2, Some(stack("stone", 1)));
        s.container.maximum_cost = Some(5);
        s.set_player_experience_level(30);
        let d = foreground(&s);
        assert!(d.vertices().len() > title_only, "the label is drawn");
        assert!(has_color(&d, ANVIL_COST_COLOR), "affordable is green");
        assert!(!has_color(&d, ANVIL_COST_DENIED_COLOR));

        // Arm 3: same cost, not enough levels -> same text, red.
        let affordable_len = foreground(&s).vertices().len();
        s.set_player_experience_level(1);
        let d = foreground(&s);
        assert_eq!(
            d.vertices().len(),
            affordable_len,
            "arm 3 recolours, it does not change the text"
        );
        assert!(
            has_color(&d, ANVIL_COST_DENIED_COLOR),
            "unaffordable is red"
        );
        assert!(!has_color(&d, ANVIL_COST_COLOR));

        // Arm 1: >= 40 levels and not creative -> the "Too Expensive!" text, also red.
        s.apply_window_property(&s2c::WindowProperty {
            window_id: 1,
            var_index: 0,
            var_value: 40,
        });
        s.set_player_experience_level(99);
        s.container.maximum_cost = Some(40);
        let d = foreground(&s);
        assert!(has_color(&d, ANVIL_COST_DENIED_COLOR));
    }

    /// java: `GuiRepair:73` draws `I18n.format("container.repair")` and never the packet's
    /// title. The 1.8.9 en_US table has `container.repair=Repair & Name`, while the title the
    /// server actually sends for an anvil is `tile.anvil.name=Anvil`.
    #[test]
    fn an_anvil_window_ignores_the_title_the_server_sent() {
        let mut s = ContainerScreen::for_open_window(
            &s2c::OpenWindow {
                window_id: 4,
                inventory_type: "minecraft:anvil".into(),
                window_title: String::new(),
                slot_count: 0,
                entity_id: None,
            },
            "Anvil",
        )
        .expect("an anvil screen");
        assert_eq!(s.title, "Repair & Name");

        let mut language = Language::new("fr_FR");
        language.merge_lang_str("container.repair=Réparer et nommer\n");
        s.set_language(&language);
        assert_eq!(s.title, "Réparer et nommer");
    }

    /// The lower caption belongs to the local player inventory, not the server window title.
    /// Java obtains it from `InventoryPlayer.getName()` (`container.inventory`) for every
    /// non-player container screen.
    #[test]
    fn player_inventory_heading_follows_the_active_language() {
        let mut s = ContainerScreen::chest(1, 3, "Coffre");
        let mut language = Language::new("fr_FR");
        language.merge_lang_str("container.inventory=Inventaire\n");

        s.set_language(&language);

        assert_eq!(s.player_title, "Inventaire");
        assert_eq!(
            s.title, "Coffre",
            "the server container title stays separate"
        );
    }

    #[test]
    fn forty_levels_is_too_expensive_unless_creative() {
        assert!(!AnvilArt {
            maximum_cost: 39,
            ..AnvilArt::default()
        }
        .is_too_expensive());
        assert!(AnvilArt {
            maximum_cost: 40,
            ..AnvilArt::default()
        }
        .is_too_expensive());
        assert!(!AnvilArt {
            maximum_cost: 99,
            creative: true,
            ..AnvilArt::default()
        }
        .is_too_expensive());
    }

    #[test]
    fn the_beacon_shows_only_confirm_and_cancel_before_the_server_reports_a_level() {
        let b = BeaconArt {
            levels: -1,
            ..BeaconArt::default()
        };
        let buttons = b.buttons(false);
        assert_eq!(buttons.len(), 2);
        assert_eq!(buttons[0].kind, BeaconButtonKind::Confirm);
        assert!(!buttons[0].enabled, "no payment, no primary effect");
        assert_eq!(buttons[1].kind, BeaconButtonKind::Cancel);
        assert!(buttons[1].enabled);
    }

    /// `TileEntityBeacon.setField` never lets arbitrary `S31` values become selected powers.
    /// Exercise both fields because their identical whitelist is easy to apply to only primary.
    #[test]
    fn beacon_window_properties_reject_non_beacon_effects() {
        let mut art = ContainerArt::Beacon(BeaconArt::default());
        for effect in [1, 3, 5, 8, 10, 11] {
            art.apply_window_property(1, effect);
            art.apply_window_property(2, effect);
            let ContainerArt::Beacon(beacon) = art else {
                unreachable!();
            };
            assert_eq!(beacon.primary, i32::from(effect));
            assert_eq!(beacon.secondary, i32::from(effect));
            art = ContainerArt::Beacon(beacon);
        }

        for invalid in [-1, 0, 2, 4, 6, 7, 9, 12, 255] {
            art.apply_window_property(1, invalid);
            art.apply_window_property(2, invalid);
            let ContainerArt::Beacon(beacon) = art else {
                unreachable!();
            };
            assert_eq!(beacon.primary, 0, "primary id {invalid}");
            assert_eq!(beacon.secondary, 0, "secondary id {invalid}");
            art = ContainerArt::Beacon(beacon);
        }
    }

    #[test]
    fn a_full_beacon_pyramid_offers_every_effect_and_centres_each_row() {
        let b = BeaconArt {
            levels: 4,
            primary: 1,
            secondary: 0,
        };
        let buttons = b.buttons(true);
        let powers: Vec<_> = buttons
            .iter()
            .filter(|b| matches!(b.kind, BeaconButtonKind::Power { .. }))
            .collect();
        // 2 + 2 + 1 tier buttons, 1 secondary (regeneration), plus the "double primary" slot.
        assert_eq!(powers.len(), 7);
        assert!(buttons[0].enabled, "payment present and a primary chosen");
        // java: `76 + col * 24 - span/2` centres a two-wide row on x = 76.
        let tier0: Vec<i32> = powers
            .iter()
            .filter(|b| matches!(b.kind, BeaconButtonKind::Power { tier: 0, .. }))
            .map(|b| b.x)
            .collect();
        assert_eq!(tier0, vec![53, 77]);
        assert!(
            powers
                .iter()
                .any(|b| b.selected && b.kind == BeaconButtonKind::Power { tier: 0, potion: 1 }),
            "the chosen primary is drawn selected"
        );
    }

    #[test]
    fn beacon_buttons_above_the_pyramid_level_are_disabled() {
        let b = BeaconArt {
            levels: 1,
            ..BeaconArt::default()
        };
        for button in b.buttons(true) {
            if let BeaconButtonKind::Power { tier, .. } = button.kind {
                assert_eq!(button.enabled, tier < 1, "tier {tier}");
            }
        }
    }

    /// java: `GuiBeacon.updateScreen` clears `buttonsNotDrawn` after the first non-negative
    /// field-0 snapshot. Later S31 updates still refresh confirm, but do not mutate any existing
    /// PowerButton until `initGui` re-arms the builder.
    #[test]
    fn beacon_power_buttons_freeze_until_init_rearms_the_snapshot() {
        fn property(screen: &mut ContainerScreen, index: i16, value: i16) {
            screen.apply_window_property(&s2c::WindowProperty {
                window_id: 1,
                var_index: index,
                var_value: value,
            });
        }
        fn power(buttons: &[BeaconButton], tier: u8, potion: u8) -> &BeaconButton {
            buttons
                .iter()
                .find(|button| button.kind == BeaconButtonKind::Power { tier, potion })
                .expect("power button")
        }

        let mut screen = opened(ContainerScreen::beacon(1, "Beacon"));
        assert_eq!(
            screen.beacon_buttons_for_draw().unwrap().len(),
            8,
            "field 0 defaults to zero, so the latched power set is present but disabled"
        );

        // Java latches the initially-disabled power set if field 0 is still at its
        // ContainerLocalMenu default when the first update runs. A late property must not make
        // those buttons appear until initGui (resize or a local power-button press) re-arms it.
        property(&mut screen, 0, 1);
        property(&mut screen, 1, 1);
        let first = screen.beacon_buttons_for_draw().unwrap();
        assert_eq!(
            first.len(),
            8,
            "late field updates do not rebuild the latched list"
        );
        assert!(!power(&first, 0, 1).enabled);

        screen.init(&ctx());
        let first = screen.beacon_buttons_for_draw().unwrap();
        assert!(power(&first, 0, 1).enabled);
        assert!(!power(&first, 1, 11).enabled);
        assert!(power(&first, 0, 1).selected);

        property(&mut screen, 0, 4);
        property(&mut screen, 1, 3);
        screen
            .container
            .set_slot_contents(0, Some(stack("emerald", 1)));
        let frozen = screen.beacon_buttons_for_draw().unwrap();
        assert!(frozen[0].enabled, "confirm is refreshed from live fields");
        assert!(
            !power(&frozen, 1, 11).enabled,
            "the level-four update does not live-enable a captured button"
        );
        assert!(
            power(&frozen, 0, 1).selected,
            "the primary update does not rewrite captured selection"
        );
        assert!(!power(&frozen, 0, 3).selected);

        screen.init(&ctx());
        let rebuilt = screen.beacon_buttons_for_draw().unwrap();
        assert!(power(&rebuilt, 1, 11).enabled);
        assert!(power(&rebuilt, 0, 3).selected);
        assert!(!power(&rebuilt, 0, 1).selected);
    }

    /// java: `GuiBeacon` dispatches the 22x22 button list before `GuiContainer`'s slot path;
    /// confirm writes two big-endian ints to `MC|Beacon`, while a disabled power button is an
    /// ordinary no-op. The edge assertions are the negative control for an inclusive hit box.
    #[test]
    fn beacon_buttons_use_half_open_hitboxes_and_emit_the_exact_confirm_payload() {
        let mut screen = opened(ContainerScreen::beacon(1, "Beacon"));
        screen.art = ContainerArt::Beacon(BeaconArt {
            levels: 4,
            primary: 0,
            secondary: 0,
        });
        screen
            .container
            .set_slot_contents(0, Some(stack("emerald", 1)));

        let tier_zero = {
            let ContainerArt::Beacon(art) = &screen.art else {
                unreachable!();
            };
            art.buttons(true)
                .into_iter()
                .find(|button| button.kind == BeaconButtonKind::Power { tier: 0, potion: 1 })
                .expect("speed power button")
        };
        let x = screen.gui_left() + tier_zero.x;
        let y = screen.gui_top() + tier_zero.y;
        assert!(screen.beacon_button_contains(tier_zero, x, y));
        assert!(screen.beacon_button_contains(tier_zero, x + 21, y + 21));
        assert!(!screen.beacon_button_contains(tier_zero, x + 22, y));
        assert!(!screen.beacon_button_contains(tier_zero, x, y + 22));

        assert_eq!(
            screen.mouse_clicked_with_mods(x + 1, y + 1, 0, Modifiers::NONE),
            ScreenAction::None
        );
        let ContainerArt::Beacon(art) = &screen.art else {
            unreachable!();
        };
        assert_eq!(art.primary, 1, "a power click selects the primary effect");
        assert!(screen.take_plugin_outgoing().is_empty());

        // `GuiScreen.mouseClicked` dispatches its button list only for mouse button zero. A
        // right-click on another power plate must not change the selected effect.
        let tier_zero_other = BeaconArt {
            levels: 4,
            primary: 1,
            secondary: 0,
        }
        .buttons(true)
        .into_iter()
        .find(|button| button.kind == BeaconButtonKind::Power { tier: 0, potion: 3 })
        .expect("haste power button");
        assert_eq!(
            screen.mouse_clicked_with_mods(
                screen.gui_left() + tier_zero_other.x + 1,
                screen.gui_top() + tier_zero_other.y + 1,
                1,
                Modifiers::NONE,
            ),
            ScreenAction::None
        );
        let ContainerArt::Beacon(art) = &screen.art else {
            unreachable!();
        };
        assert_eq!(
            art.primary, 1,
            "right-click does not dispatch beacon buttons"
        );
        assert!(screen.take_plugin_outgoing().is_empty());

        // A power above the current pyramid level is visible but disabled, just like Java's
        // GuiBeacon.PowerButton. It must not select an effect or fall through to a slot click.
        screen.art = ContainerArt::Beacon(BeaconArt {
            levels: 1,
            primary: 1,
            secondary: 0,
        });
        screen.beacon_power_buttons = None;
        let disabled = {
            let ContainerArt::Beacon(art) = &screen.art else {
                unreachable!();
            };
            art.buttons(true)
                .into_iter()
                .find(|button| {
                    button.kind
                        == BeaconButtonKind::Power {
                            tier: 1,
                            potion: 11,
                        }
                })
                .expect("resistance power button")
        };
        assert!(!disabled.enabled);
        assert_eq!(
            screen.mouse_clicked_with_mods(
                screen.gui_left() + disabled.x + 1,
                screen.gui_top() + disabled.y + 1,
                0,
                Modifiers::NONE,
            ),
            ScreenAction::None
        );
        let ContainerArt::Beacon(art) = &screen.art else {
            unreachable!();
        };
        assert_eq!(art.primary, 1);
        assert!(screen.take_plugin_outgoing().is_empty());

        let confirm = {
            let ContainerArt::Beacon(art) = &screen.art else {
                unreachable!();
            };
            art.buttons(true)
                .into_iter()
                .find(|button| button.kind == BeaconButtonKind::Confirm)
                .expect("confirm button")
        };
        screen.art = ContainerArt::Beacon(BeaconArt {
            levels: 4,
            primary: 1,
            secondary: 0,
        });
        screen.beacon_power_buttons = None;
        assert_eq!(
            screen.mouse_clicked_with_mods(
                screen.gui_left() + confirm.x + 1,
                screen.gui_top() + confirm.y + 1,
                0,
                Modifiers::NONE,
            ),
            ScreenAction::Pop
        );
        assert_eq!(
            screen.take_plugin_outgoing(),
            vec![c2s::PluginMessage {
                channel: "MC|Beacon".to_owned(),
                data: vec![0, 0, 0, 1, 0, 0, 0, 0],
            }]
        );
        assert!(
            screen.take_outgoing().is_empty(),
            "GuiBeacon confirm closes without a C0D slot packet"
        );

        let mut cancelled = opened(ContainerScreen::beacon(1, "Beacon"));
        let cancel = BeaconArt::default()
            .buttons(false)
            .into_iter()
            .find(|button| button.kind == BeaconButtonKind::Cancel)
            .expect("cancel button");
        assert_eq!(
            cancelled.mouse_clicked_with_mods(
                cancelled.gui_left() + cancel.x + 1,
                cancelled.gui_top() + cancel.y + 1,
                0,
                Modifiers::NONE,
            ),
            ScreenAction::Pop
        );
        assert!(cancelled.take_plugin_outgoing().is_empty());
    }

    /// java: `GuiBeacon.drawGuiContainerForegroundLayer` asks the first hovered button for its
    /// tooltip. Power names come from `Potion.getName`, and only tier three's non-regeneration
    /// entries receive the literal ` II` suffix.
    #[test]
    fn beacon_hover_tooltips_localize_power_names_and_clear_outside_the_button_list() {
        let mut screen = opened(ContainerScreen::beacon(1, "Beacon"));
        screen.art = ContainerArt::Beacon(BeaconArt {
            levels: 4,
            primary: 1,
            secondary: 0,
        });
        let mut language = Language::new("test");
        language.merge_lang_str(concat!(
            "container.inventory=Inventory\n",
            "gui.done=Finish\n",
            "gui.cancel=Abort\n",
            "potion.moveSpeed=Velocity\n",
            "potion.regeneration=Regen\n",
            "tile.beacon.primary=Primary Power\n",
            "tile.beacon.secondary=Secondary Power\n",
        ));
        screen.set_language(&language);

        let buttons = BeaconArt {
            levels: 4,
            primary: 1,
            secondary: 0,
        }
        .buttons(false);
        let speed_two = buttons
            .iter()
            .copied()
            .find(|button| button.kind == BeaconButtonKind::Power { tier: 3, potion: 1 })
            .expect("double-speed power button");
        let point = (
            screen.gui_left() + speed_two.x + 1,
            screen.gui_top() + speed_two.y + 1,
        );
        screen.draw_beacon_buttons(&mut DrawList::new(), point);
        assert_eq!(screen.hovered_beacon_button, Some(speed_two));
        assert_eq!(
            screen.hovered_tooltip(),
            Some(vec!["Velocity II".to_owned()])
        );

        let regeneration = buttons
            .iter()
            .copied()
            .find(|button| {
                button.kind
                    == BeaconButtonKind::Power {
                        tier: 3,
                        potion: 10,
                    }
            })
            .expect("regeneration power button");
        screen.draw_beacon_buttons(
            &mut DrawList::new(),
            (
                screen.gui_left() + regeneration.x + 1,
                screen.gui_top() + regeneration.y + 1,
            ),
        );
        assert_eq!(
            screen.hovered_tooltip(),
            Some(vec!["Regen".to_owned()]),
            "regeneration is the tier-three exception to the II suffix"
        );

        let confirm = buttons
            .iter()
            .copied()
            .find(|button| button.kind == BeaconButtonKind::Confirm)
            .expect("confirm button");
        screen.draw_beacon_buttons(
            &mut DrawList::new(),
            (
                screen.gui_left() + confirm.x + 1,
                screen.gui_top() + confirm.y + 1,
            ),
        );
        assert_eq!(screen.hovered_tooltip(), Some(vec!["Finish".to_owned()]));

        // The four-pixel gap between Done and Cancel is not a button. A stale hovered button must
        // not leak its tooltip into the next foreground pass.
        screen.draw_beacon_buttons(
            &mut DrawList::new(),
            (screen.gui_left() + 186, screen.gui_top() + 108),
        );
        assert_eq!(screen.hovered_beacon_button, None);
        assert_eq!(screen.hovered_tooltip(), None);
    }

    /// java: `GuiRepair` replaces the three fake shadow glyph passes with two solid rectangles
    /// when the Unicode font is active. The coordinates must be translated exactly once from
    /// the foreground origin; the ASCII branch is the negative control.
    #[test]
    fn unicode_anvil_cost_backdrop_uses_absolute_panel_coordinates() {
        let mut screen = opened(ContainerScreen::anvil(1, "Repair & Name"));
        screen
            .container
            .set_slot_contents(2, Some(stack("stone", 1)));
        if let ContainerArt::Anvil(art) = &mut screen.art {
            art.maximum_cost = 5;
        }

        let left = 23;
        let top = 31;
        let mut unicode_font = uniform_font();
        unicode_font.set_unicode_flag(true);
        let mut unicode_draw = DrawList::new();
        screen.art.draw_foreground(
            &mut unicode_draw,
            &mut unicode_font,
            left,
            top,
            &screen.container,
            "Repair & Name",
            "Inventory",
            None,
        );
        let solid = unicode_draw
            .batches()
            .iter()
            .find(|batch| batch.texture == UiTexture::Solid)
            .expect("Unicode anvil cost backdrop");
        let vertices = &unicode_draw.vertices()[solid.range.clone()];
        assert_eq!(vertices.len(), 12, "two rectangles, no fake shadow glyphs");
        assert_eq!(vertices[0].pos, [180.0, 96.0]);
        assert_eq!(vertices[6].pos, [181.0, 97.0]);

        let mut ascii_font = uniform_font();
        let mut ascii_draw = DrawList::new();
        screen.art.draw_foreground(
            &mut ascii_draw,
            &mut ascii_font,
            left,
            top,
            &screen.container,
            "Repair & Name",
            "Inventory",
            None,
        );
        assert!(
            ascii_draw
                .batches()
                .iter()
                .all(|batch| batch.texture != UiTexture::Solid),
            "ASCII uses the three text shadow passes instead of the Unicode plate"
        );
    }

    #[test]
    fn the_six_beacon_effects_carry_vanillas_own_status_icon_indices() {
        // java: Potion.setIconIndex(col, row) stores row * 8 + col.
        assert_eq!(beacon_status_icon_index(1), Some(0)); // speed        (0, 0)
        assert_eq!(beacon_status_icon_index(3), Some(2)); // haste        (2, 0)
        assert_eq!(beacon_status_icon_index(5), Some(4)); // strength     (4, 0)
        assert_eq!(beacon_status_icon_index(8), Some(10)); // jump boost  (2, 1)
        assert_eq!(beacon_status_icon_index(10), Some(7)); // regen       (7, 0)
        assert_eq!(beacon_status_icon_index(11), Some(14)); // resistance (6, 1)
                                                            // The function now delegates to `mc_data::PotionEffect::status_icon_index` rather than a
                                                            // beacon-only transcription, so it answers for *any* registered potion, not just the six
                                                            // the beacon grants — poison (19) has a real HUD icon (6), it just is not a beacon tier.
        assert_eq!(beacon_status_icon_index(19), Some(6));
        // `0` is not a registered potion id at all; `heal` (6) is registered but instant, so
        // `Potion.hasStatusIcon()` is false for it. Both correctly resolve to no icon.
        assert_eq!(beacon_status_icon_index(0), None);
        assert_eq!(beacon_status_icon_index(6), None);
        for row in BEACON_EFFECTS {
            for &id in row {
                assert!(beacon_status_icon_index(id).is_some(), "potion {id}");
                assert!(mc_data::potion(id).is_some(), "potion {id} is registered");
            }
        }
    }

    #[test]
    fn a_chest_blits_its_two_halves_with_the_seam_moving_with_the_row_count() {
        for rows in [1u8, 3, 6] {
            let s = ContainerScreen::chest(1, rows, "Chest");
            let mut d = DrawList::new();
            s.art.draw_background(&mut d, 0, 0, &s.container);
            assert_eq!(d.vertices().len(), 12, "two blits");
            let bottom = d
                .vertices()
                .iter()
                .map(|v| v.pos[1])
                .fold(f32::MIN, f32::max);
            assert_eq!(
                bottom,
                (i32::from(rows) * 18 + 17 + 96) as f32,
                "the panel's height has to equal ySize + 2 for {rows} rows"
            );
            assert_eq!(s.art.y_size(), i32::from(rows) * 18 + 114);
        }
    }

    #[test]
    fn the_panel_texture_names_match_the_vanilla_jars_own_paths() {
        let expected = [
            (ContainerArt::Player, "inventory.png"),
            (ContainerArt::Chest { rows: 3 }, "generic_54.png"),
            (ContainerArt::Hopper, "hopper.png"),
            (ContainerArt::Dispenser, "dispenser.png"),
            (ContainerArt::Furnace(FurnaceArt::default()), "furnace.png"),
            (
                ContainerArt::BrewingStand(BrewingStandArt::default()),
                "brewing_stand.png",
            ),
            (
                ContainerArt::Enchanting(EnchantingArt::default()),
                "enchanting_table.png",
            ),
            (ContainerArt::Anvil(AnvilArt::default()), "anvil.png"),
            (ContainerArt::Beacon(BeaconArt::default()), "beacon.png"),
            (ContainerArt::Horse { chested: false }, "horse.png"),
        ];
        for (art, name) in expected {
            assert_eq!(
                art.texture(),
                UiTexture::gui(&format!("textures/gui/container/{name}")),
                "{art:?}"
            );
        }
    }

    #[test]
    fn only_the_player_inventory_lets_movement_keys_through() {
        assert!(ContainerScreen::player(0).allows_user_input());
        assert!(!ContainerScreen::chest(1, 3, "Chest").allows_user_input());
        assert!(!ContainerScreen::hopper(1, 5, "Hopper").allows_user_input());
        assert!(!ContainerScreen::horse(1, false, "Horse").allows_user_input());
        assert!(!ContainerScreen::player(0).pauses_game());
    }

    // Packet order for every local anvil interaction, against the Java chain
    // `PlayerControllerMP.windowClick` (`slotClick`, then `addToSendQueue(C0E)`) →
    // `Container.slotClick` → `Slot.putStack`/`decrStackSize` → `InventoryBasic.markDirty` →
    // `ContainerRepair.onCraftMatrixChanged` → `Container.detectAndSendChanges` →
    // `GuiRepair.sendSlotContents(0, stack)` → `renameItem` → `C17 MC|ItemName`.

    /// One anvil packet as the server reads it.
    #[derive(Debug, PartialEq)]
    enum AnvilWire {
        /// `C17PacketCustomPayload("MC|ItemName", writeString(name))`.
        Name(String),
        /// `C0EPacketClickWindow`'s `(slotId, usedButton, mode)`.
        Click(i16, u8, u8),
    }

    fn name(text: &str) -> AnvilWire {
        AnvilWire::Name(text.to_owned())
    }

    /// Drain the screen exactly as the host does and decode each packet's wire body.
    fn anvil_wire(s: &mut ContainerScreen) -> Vec<AnvilWire> {
        s.take_ordered_outgoing()
            .into_iter()
            .map(|packet| match packet {
                mc_protocol::play::C2SPlay::PluginMessage(message) => {
                    assert_eq!(message.channel, "MC|ItemName");
                    let mut body = mc_protocol::PacketBuffer::from_slice(&message.data);
                    let text = body.read_string(32_767).expect("a writeString body");
                    assert_eq!(body.remaining(), 0, "nothing after the string");
                    AnvilWire::Name(text)
                }
                mc_protocol::play::C2SPlay::ClickWindow(click) => {
                    AnvilWire::Click(click.slot_id, click.used_button, click.mode)
                }
                other => panic!("unexpected anvil packet {other:?}"),
            })
            .collect()
    }

    /// An open anvil whose language resolves the apple, as the host's does on `OpenWindow`.
    fn english_anvil() -> ContainerScreen {
        let mut s = opened(ContainerScreen::anvil(1, "Repair & Name"));
        let mut language = Language::default();
        language.merge_lang_str("item.apple.name=Apple\n");
        s.set_language(&language);
        s
    }

    /// A press and release over `slot`, a second apart so no two are a double-click.
    fn click_anvil_slot(s: &mut ContainerScreen, slot: usize, button: i32, mods: Modifiers) {
        s.clock_ms += 1_000;
        let at = s.container.slots()[slot];
        let (x, y) = (s.gui_left() + at.x + 8, s.gui_top() + at.y + 8);
        assert_eq!(s.target_at(x, y), SlotTarget::Slot(slot as u16));
        s.mouse_clicked_with_mods(x, y, button, mods);
        s.mouse_released_with_mods(x, y, button, mods);
    }

    fn type_into_anvil(s: &mut ContainerScreen, typed: char, font: &FontRenderer) {
        s.anvil_name.as_mut().unwrap().set_focused(true);
        Screen::key_typed_with_font(s, Some(typed), 0, Modifiers::NONE, font);
    }

    fn named_apple(text: &str) -> ItemStack {
        let mut display = mc_nbt::NbtCompound::new();
        display.insert("Name", mc_nbt::Nbt::String(text.to_owned()));
        let mut root = mc_nbt::NbtCompound::new();
        root.insert("display", mc_nbt::Nbt::Compound(display));
        ItemStack {
            nbt: Some(root),
            ..stack("apple", 1)
        }
    }

    /// The bug report: "mc-rs sends the name packet after the click instead of before it".
    /// Placing a stack into slot 0 sends `MC|ItemName` first. The server's echo of slot 0 then
    /// equals the snapshot, so it sends nothing more (before the fix, that echo was the only
    /// trigger, one round trip after the C0E).
    #[test]
    fn w7_putting_an_item_in_sends_the_name_before_the_click() {
        let mut s = english_anvil();
        s.container.set_cursor(Some(stack("apple", 1)));
        click_anvil_slot(&mut s, 0, 0, Modifiers::NONE);
        assert_eq!(anvil_wire(&mut s), [name(""), AnvilWire::Click(0, 0, 0)]);
        assert!(s.anvil_name_enabled());
        assert_eq!(s.anvil_rename_text().as_deref(), Some("Apple"));
        assert_eq!(s.container.slot_stack(0), Some(&stack("apple", 1)));

        s.container.apply_set_slot(&s2c::SetSlot {
            window_id: 1,
            slot: 0,
            item: Some(stack("apple", 1)),
        });
        assert!(!s.anvil_input_needs_sync(), "the echo is no change");
        assert!(anvil_wire(&mut s).is_empty());
    }

    /// `ItemStack.hasDisplayName()` keeps a custom name on the wire instead of `""`.
    #[test]
    fn w7_a_named_item_sends_its_custom_name_before_the_click() {
        let mut s = english_anvil();
        s.container.set_cursor(Some(named_apple("Sword of Doom")));
        click_anvil_slot(&mut s, 0, 0, Modifiers::NONE);
        assert_eq!(
            anvil_wire(&mut s),
            [name("Sword of Doom"), AnvilWire::Click(0, 0, 0)]
        );
        assert_eq!(s.anvil_rename_text().as_deref(), Some("Sword of Doom"));
    }

    /// Taking the item out: `sendSlotContents(0, null)` clears and disables the field and has
    /// no `renameItem`, so the C0E is the only packet.
    #[test]
    fn w7_taking_the_item_out_sends_only_the_click() {
        let mut s = english_anvil();
        s.container.set_cursor(Some(stack("apple", 1)));
        click_anvil_slot(&mut s, 0, 0, Modifiers::NONE);
        let _ = anvil_wire(&mut s);

        click_anvil_slot(&mut s, 0, 0, Modifiers::NONE);
        assert_eq!(anvil_wire(&mut s), [AnvilWire::Click(0, 0, 0)]);
        assert_eq!(s.container.cursor(), Some(&stack("apple", 1)));
        assert!(!s.anvil_name_enabled());
        assert_eq!(s.anvil_rename_text().as_deref(), Some(""));
    }

    /// Typing: one `MC|ItemName` per key `textboxKeyTyped` accepts, and none per click it did
    /// not cause. A click after typing keeps Java's invocation order in the one stream.
    #[test]
    fn w7_typing_then_clicking_keeps_javas_order() {
        let font = uniform_font();
        let mut s = english_anvil();
        s.container.set_cursor(Some(stack("apple", 1)));
        click_anvil_slot(&mut s, 0, 0, Modifiers::NONE);
        let _ = anvil_wire(&mut s);

        type_into_anvil(&mut s, '!', &font);
        type_into_anvil(&mut s, '?', &font);
        // Slot 1 is not slot 0: `detectAndSendChanges` finds no slot-0 change, no rename.
        s.container.set_slot_contents(5, Some(stack("dirt", 1)));
        click_anvil_slot(&mut s, 5, 0, Modifiers::SHIFT);
        assert_eq!(
            anvil_wire(&mut s),
            [name("Apple!"), name("Apple!?"), AnvilWire::Click(5, 0, 1)]
        );
        assert_eq!(s.container.slot_stack(1), Some(&stack("dirt", 1)));
        assert_eq!(
            s.anvil_rename_text().as_deref(),
            Some("Apple!?"),
            "typing survives"
        );
    }

    /// Shift-clicking into the anvil fills slot 0 inside `transferStackInSlot`, so the name
    /// precedes the mode-1 C0E. Shift-clicking it back out empties slot 0: the click alone.
    #[test]
    fn w7_shift_clicks_in_and_out_follow_the_same_rule() {
        let mut s = english_anvil();
        s.container.set_slot_contents(3, Some(stack("apple", 1)));
        click_anvil_slot(&mut s, 3, 0, Modifiers::SHIFT);
        assert_eq!(anvil_wire(&mut s), [name(""), AnvilWire::Click(3, 0, 1)]);
        assert_eq!(s.container.slot_stack(0), Some(&stack("apple", 1)));

        click_anvil_slot(&mut s, 0, 0, Modifiers::SHIFT);
        assert_eq!(anvil_wire(&mut s), [AnvilWire::Click(0, 0, 1)]);
        assert_eq!(s.container.slot_stack(0), None);
        assert!(!s.anvil_name_enabled());
    }

    /// Taking part of slot 0 is still a slot-0 change: vanilla resets the typed name to the
    /// item's own (`setText(getDisplayName())`) and renames before the C0E.
    #[test]
    fn w7_taking_half_of_slot_zero_resets_the_name_and_resends_it_first() {
        let font = uniform_font();
        let mut s = english_anvil();
        s.container.set_cursor(Some(stack("apple", 2)));
        click_anvil_slot(&mut s, 0, 0, Modifiers::NONE);
        type_into_anvil(&mut s, '!', &font);
        assert_eq!(
            anvil_wire(&mut s),
            [name(""), AnvilWire::Click(0, 0, 0), name("Apple!")]
        );

        click_anvil_slot(&mut s, 0, 1, Modifiers::NONE);
        assert_eq!(anvil_wire(&mut s), [name(""), AnvilWire::Click(0, 1, 0)]);
        assert_eq!(s.container.slot_stack(0), Some(&stack("apple", 1)));
        assert_eq!(s.anvil_rename_text().as_deref(), Some("Apple"));
    }

    /// Taking the output consumes the inputs in `onPickupFromSlot`; slot 0 becomes empty, so
    /// again only the C0E is sent.
    #[test]
    fn w7_taking_the_output_sends_only_the_click() {
        let font = uniform_font();
        let mut s = english_anvil();
        s.set_player_experience_level(30);
        s.container.set_cursor(Some(stack("apple", 1)));
        click_anvil_slot(&mut s, 0, 0, Modifiers::NONE);
        type_into_anvil(&mut s, '!', &font);
        let _ = anvil_wire(&mut s);
        assert!(
            s.container.slot_stack(2).is_some(),
            "the rename has an output"
        );

        click_anvil_slot(&mut s, 2, 0, Modifiers::NONE);
        assert_eq!(anvil_wire(&mut s), [AnvilWire::Click(2, 0, 0)]);
        assert_eq!(s.container.slot_stack(0), None);
        let taken = s.container.cursor().and_then(|stack| {
            let display = stack.nbt.as_ref()?.get_compound("display")?;
            display.get_str("Name").map(str::to_owned)
        });
        assert_eq!(
            taken.as_deref(),
            Some("Apple!"),
            "the renamed output is on the cursor"
        );
        assert!(!s.anvil_name_enabled());
    }

    /// The font-free reset a slot click makes is laid out exactly as `setText` would have been,
    /// by the next draw, so a long name shows its tail like vanilla's field.
    #[test]
    fn w7_a_click_reset_lays_out_the_field_like_set_text() {
        let font = uniform_font();
        let long = "An Extremely Long Custom Apple Name!";
        let mut eager = english_anvil();
        let mut clicked = english_anvil();
        eager
            .container
            .set_slot_contents(0, Some(named_apple(long)));
        eager.sync_anvil_input_name(Some(long), true, &font);
        clicked.container.set_cursor(Some(named_apple(long)));
        click_anvil_slot(&mut clicked, 0, 0, Modifiers::NONE);
        clicked.draw(&mut DrawList::new(), &mut uniform_font(), &ctx(), (0, 0));

        let field = |s: &ContainerScreen| {
            let f = s.anvil_name.as_ref().unwrap();
            (f.text(), f.cursor_position(), f.scroll_offset())
        };
        assert_eq!(field(&clicked), field(&eager));
        assert!(
            field(&clicked).2 > 0,
            "a 30-unit name scrolls in the 103-px box"
        );
        // `setText` truncated it to 30 units, so the wire carries the truncated custom name.
        assert_eq!(
            anvil_wire(&mut clicked),
            [name(&long[..30]), AnvilWire::Click(0, 0, 0)]
        );
    }
}
