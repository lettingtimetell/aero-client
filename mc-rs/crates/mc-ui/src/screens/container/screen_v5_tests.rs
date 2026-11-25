//! The 1.7.10 container screens: the middle-button drag's packets (the creative inventory's
//! too), the enchanting table's `C11` gate, the furnace's properties, the anvil's name field and
//! the inventories' potion shift, each against 1.8.9.

use mc_protocol::play::c2s;
use mc_protocol::v5::s2c as s2c5;
use mc_version::GameVersion;

use super::*;
use crate::screen_stack::{Screen, ScreenCtx};
use crate::screens::container::interaction::{MouseState, SlotTarget};
use crate::screens::container::layouts;
use crate::screens::container::stack_ext::ItemStack;
use crate::screens::container::v5::open_window_v47;
use crate::screens::container::CreativeScreen;
use crate::widget::keys::Modifiers;
use crate::widget::ScaledResolution;

const STONE: i16 = 1;
const MIDDLE: i32 = 2;

fn stone(count: i32) -> ItemStack {
    ItemStack {
        item_id: STONE,
        count,
        damage: 0,
        nbt: None,
    }
}

fn triples(packets: &[c2s::ClickWindow]) -> Vec<(i16, u8, u8)> {
    packets
        .iter()
        .map(|p| (p.slot_id, p.used_button, p.mode))
        .collect()
}

/// One `GuiContainer` drag: press `button` over `slots[0]`, move over every slot, release over
/// the last. Returns the packets the release sent.
fn drag(
    mouse: &mut MouseState,
    container: &mut Container,
    button: i32,
    pick_block: bool,
    slots: &[u16],
) -> Vec<c2s::ClickWindow> {
    let first = SlotTarget::Slot(slots[0]);
    assert!(mouse
        .mouse_clicked(container, first, button, pick_block, false, 1_000)
        .is_none());
    for &slot in slots {
        mouse.drag_move(container, SlotTarget::Slot(slot));
    }
    let last = SlotTarget::Slot(*slots.last().unwrap());
    mouse.mouse_released(container, last, button, pick_block, false, false)
}

#[test]
fn a_fresh_screens_middle_drag_splits_with_the_left_drags_buttons_in_1_7_10() {
    // Survival, window 0, slots 9 and 10 empty, 4 stone on the cursor: press the pick-block
    // button over 9, move over 9 and 10, release over 10.
    let mut c = layouts::player_container(0).with_game_version(GameVersion::V1_7_10);
    c.set_cursor(Some(stone(4)));
    let packets = drag(&mut MouseState::new(), &mut c, MIDDLE, true, &[9, 10]);
    assert_eq!(
        triples(&packets),
        [(-999, 0, 5), (9, 1, 5), (10, 1, 5), (-999, 2, 5)]
    );
    assert!(packets.iter().all(|p| p.clicked_item.is_none()));
    assert_eq!(c.slot_stack(9), Some(&stone(2)));
    assert_eq!(c.slot_stack(10), Some(&stone(2)));
    assert!(c.cursor().is_none());

    // 1.8.9: the creative-fill buttons 8/9/10, refused in survival; nothing moves.
    let mut c = layouts::player_container(0);
    c.set_cursor(Some(stone(4)));
    let packets = drag(&mut MouseState::new(), &mut c, MIDDLE, true, &[9, 10]);
    assert_eq!(
        triples(&packets),
        [(-999, 8, 5), (9, 9, 5), (10, 9, 5), (-999, 10, 5)]
    );
    assert!(c.slot_stack(9).is_none() && c.slot_stack(10).is_none());
    assert_eq!(c.cursor(), Some(&stone(4)));
}

#[test]
fn a_middle_drag_after_a_right_drag_reuses_the_right_drags_buttons_in_1_7_10() {
    let run = |version: GameVersion| {
        let mut c = layouts::player_container(0).with_game_version(version);
        c.set_cursor(Some(stone(4)));
        let mut mouse = MouseState::new();
        let right = drag(&mut mouse, &mut c, 1, false, &[20, 21]);
        assert_eq!(
            triples(&right),
            [(-999, 4, 5), (20, 5, 5), (21, 5, 5), (-999, 6, 5)]
        );
        assert_eq!(c.cursor(), Some(&stone(2)));
        let middle = drag(&mut mouse, &mut c, MIDDLE, true, &[9, 10]);
        (triples(&middle), c)
    };
    let (middle, c) = run(GameVersion::V1_7_10);
    assert_eq!(middle, [(-999, 4, 5), (9, 5, 5), (10, 5, 5), (-999, 6, 5)]);
    assert_eq!(c.slot_stack(9), Some(&stone(1)));
    assert_eq!(c.slot_stack(10), Some(&stone(1)));
    assert!(c.cursor().is_none());
    // 1.8.9 (negative control): 8/9/10, refused, the cursor keeps its two.
    let (middle, c) = run(GameVersion::V1_8_9);
    assert_eq!(middle, [(-999, 8, 5), (9, 9, 5), (10, 9, 5), (-999, 10, 5)]);
    assert!(c.slot_stack(9).is_none());
    assert_eq!(c.cursor(), Some(&stone(2)));
}

fn ctx() -> ScreenCtx {
    ScreenCtx::new(ScaledResolution::new(427, 240, 1, false), 0.0)
}

/// The C10s a drag sent, as (slot, count).
type SlotCounts = Vec<(i16, Option<i32>)>;

/// The creative inventory's Inventory tab, 4 stone on the cursor: a right drag over slots 20 and
/// 21, then a middle (pick-block) drag over 9 and 10, through the screen's own mouse handlers.
/// Returns the C10s of each drag and the cursor left.
fn creative_right_then_middle_drag(
    version: GameVersion,
) -> (SlotCounts, SlotCounts, Option<ItemStack>) {
    use crate::screens::container::creative::CreativeTab;
    use crate::screens::container::creative_screen::CreativeSlot;

    let container = layouts::player_container(0).with_game_version(version);
    let mut screen = CreativeScreen::over(container);
    screen.init(&ctx());
    screen.set_tab(CreativeTab::Inventory.index().expect("a tab"));
    screen.player.set_cursor(Some(stone(4)));
    let point = |screen: &CreativeScreen, id: u16| {
        let (_, x, y) = screen
            .slot_rects()
            .into_iter()
            .find(|(slot, _, _)| *slot == CreativeSlot::Player(id))
            .expect("the Inventory tab shows the main inventory");
        (screen.gui_left() + x + 8, screen.gui_top() + y + 8)
    };
    let mut drag = |button: i32, slots: &[u16]| {
        let (x, y) = point(&screen, slots[0]);
        screen.mouse_clicked(x, y, button);
        for &slot in slots {
            let (x, y) = point(&screen, slot);
            screen.mouse_dragged(x, y, button, 0);
        }
        let (x, y) = point(&screen, *slots.last().unwrap());
        screen.mouse_released(x, y, button);
        screen
            .take_outgoing()
            .into_iter()
            .map(|packet| (packet.slot_id, packet.stack.map(|stack| stack.count)))
            .collect::<Vec<_>>()
    };
    let right = drag(1, &[20, 21]);
    let middle = drag(MIDDLE, &[9, 10]);
    (right, middle, screen.player.cursor().cloned())
}

/// java17 `GuiContainer.mouseClicked` sets the drag limit for the left and right buttons only,
/// so the creative screen's middle drag reuses the right drag's one-per-slot split; 1.8.9's
/// middle drag fills each slot with a full stack (creative drag mode 2, which charges the
/// cursor a full stack per slot and so empties it).
#[test]
fn a_1_7_10_creative_middle_drag_reuses_the_right_drags_split() {
    let (right, middle, cursor) = creative_right_then_middle_drag(GameVersion::V1_7_10);
    assert_eq!(right, [(20, Some(1)), (21, Some(1))]);
    assert_eq!(middle, [(9, Some(1)), (10, Some(1))]);
    assert_eq!(cursor, None);
    // 1.8.9 (the negative control).
    let (right, middle, cursor) = creative_right_then_middle_drag(GameVersion::V1_8_9);
    assert_eq!(right, [(20, Some(1)), (21, Some(1))]);
    assert_eq!(middle, [(9, Some(64)), (10, Some(64))]);
    assert_eq!(cursor, None);
}

fn open_v5(inventory_type: u8, slot_count: u8) -> ContainerScreen {
    let lowered = open_window_v47(&s2c5::OpenWindow {
        window_id: 5,
        inventory_type,
        title: String::new(),
        slot_count,
        use_provided_title: false,
        entity_id: 0,
    })
    .expect("a 1.7.10 window");
    let mut screen = ContainerScreen::for_open_window_v5(&lowered, "t").expect("a screen");
    screen.init(&ctx());
    screen
}

fn property(window_id: u8, var_index: i16, var_value: i16) -> s2c::WindowProperty {
    s2c::WindowProperty {
        window_id,
        var_index,
        var_value,
    }
}

fn sword() -> ItemStack {
    ItemStack {
        item_id: 276,
        count: 1,
        damage: 0,
        nbt: None,
    }
}

/// Click enchanting plate `row` and return the C11s it queued.
fn click_plate(screen: &mut ContainerScreen, row: i32) -> Vec<c2s::EnchantItem> {
    let x = screen.gui_left() + 61;
    let y = screen.gui_top() + 15 + 19 * row;
    screen.mouse_clicked_with_mods(x, y, 0, Modifiers::default());
    screen.take_enchant_outgoing()
}

#[test]
fn a_1_7_10_table_enchants_without_lapis() {
    // Sword in slot 0, property 0 = 3, player level 5, click option 0.
    let mut screen = open_v5(4, 9);
    screen.container.set_slot_contents(0, Some(sword()));
    screen.apply_window_property(&property(5, 0, 3));
    screen.set_player_experience_level(5);
    let sent = click_plate(&mut screen, 0);
    assert_eq!(
        sent,
        [c2s::EnchantItem {
            window_id: 5,
            button: 0
        }]
    );
    // 1.8.9: the same window with no lapis sends nothing.
    let mut screen = ContainerScreen::enchanting_table(5, "t");
    screen.init(&ctx());
    screen.container.set_slot_contents(0, Some(sword()));
    screen.apply_window_property(&property(5, 0, 3));
    screen.set_player_experience_level(5);
    assert!(click_plate(&mut screen, 0).is_empty());
}

#[test]
fn the_1_7_10_gate_is_level_against_cost_only() {
    let mut screen = open_v5(4, 9);
    screen.container.set_slot_contents(0, Some(sword()));
    // Row 1 costs 1: 1.8.9 would also want `experienceLevel >= 2`; 1.7.10 does not.
    screen.apply_window_property(&property(5, 1, 1));
    screen.apply_window_property(&property(5, 2, 30));
    screen.set_player_experience_level(1);
    assert_eq!(click_plate(&mut screen, 1).len(), 1);
    // Too expensive, no offer, or no item: nothing.
    assert!(click_plate(&mut screen, 2).is_empty());
    assert!(click_plate(&mut screen, 0).is_empty());
    screen.creative = true;
    assert_eq!(click_plate(&mut screen, 2).len(), 1);
    screen.creative = false;
    screen.container.set_slot_contents(0, None);
    assert!(click_plate(&mut screen, 1).is_empty());
}

#[test]
fn a_1_7_10_furnace_reads_cook_burn_and_item_burn_time() {
    let mut screen = open_v5(2, 3);
    screen.apply_window_property(&property(5, 0, 100));
    screen.apply_window_property(&property(5, 1, 40));
    screen.apply_window_property(&property(5, 2, 1600));
    let ContainerArt::Furnace(art) = screen.art else {
        panic!("a furnace");
    };
    assert_eq!(
        (art.cook_time, art.burn_time, art.current_item_burn_time),
        (100, 40, 1600)
    );
    assert_eq!(art.cook_progress_scaled(24), 12, "100 * 24 / 200");
    // 1.8.9: property 0 is the burn time.
    let mut screen = ContainerScreen::furnace(5, "t");
    screen.apply_window_property(&property(5, 0, 100));
    let ContainerArt::Furnace(art) = screen.art else {
        panic!("a furnace");
    };
    assert_eq!((art.burn_time, art.cook_time), (100, 0));
}

fn one_effect() -> Vec<crate::screens::hud::ActiveEffect> {
    vec![crate::screens::hud::ActiveEffect {
        icon_index: -1,
        name: "Speed".to_owned(),
        duration: "1:23".to_owned(),
        beneficial: true,
    }]
}

/// What the client does with an open inventory: a tick (effects first, as the client hands
/// them over), then a frame's two layouts (`init` from the input pass and from the draw).
trait Frames {
    fn run(&mut self, effects: &[crate::screens::hud::ActiveEffect], ctx: &ScreenCtx);
    fn left(&self) -> i32;
    fn set_layout(&mut self, centered: bool);
}

impl Frames for ContainerScreen {
    fn run(&mut self, effects: &[crate::screens::hud::ActiveEffect], ctx: &ScreenCtx) {
        for _ in 0..3 {
            self.set_active_effects(effects.to_vec());
            self.tick();
            self.init(ctx);
            self.init(ctx);
        }
    }
    fn left(&self) -> i32 {
        self.gui_left()
    }
    fn set_layout(&mut self, centered: bool) {
        self.set_effect_layout(centered, true);
    }
}

impl Frames for CreativeScreen {
    fn run(&mut self, effects: &[crate::screens::hud::ActiveEffect], ctx: &ScreenCtx) {
        for _ in 0..3 {
            self.set_active_effects(effects.to_vec());
            self.tick();
            self.init(ctx);
            self.init(ctx);
        }
    }
    fn left(&self) -> i32 {
        self.gui_left()
    }
    fn set_layout(&mut self, centered: bool) {
        self.set_effect_layout(centered, true);
    }
}

/// java17 `InventoryEffectRenderer.initGui`: the survival and creative inventories shift for
/// effects only when laid out (opened or resized); 1.8.9 follows them every tick. The
/// `centered_inventory` mod applies at once in both.
fn shifts_for_effects_only_when_laid_out(
    open: impl Fn(GameVersion) -> Box<dyn Frames>,
    x_size: i32,
) {
    let centred = |width: i32| (width - x_size) / 2;
    let shifted = |width: i32| 160 + (width - x_size - 200) / 2;
    let wide = ScreenCtx::new(ScaledResolution::new(480, 270, 1, false), 0.0);
    let none = Vec::new();
    let one = one_effect();
    for version in [GameVersion::V1_8_9, GameVersion::V1_7_10] {
        let v17 = version.is_1_7_10();
        let mut screen = open(version);
        screen.run(&none, &ctx());
        assert_eq!(screen.left(), centred(427), "{version}");
        // An effect starts while the inventory is open: frames and ticks go by.
        screen.run(&one, &ctx());
        let open_left = if v17 { centred(427) } else { shifted(427) };
        assert_eq!(screen.left(), open_left, "{version}");
        // A new size is a new layout, which takes the effects into account ...
        screen.run(&one, &wide);
        assert_eq!(screen.left(), shifted(480), "{version}");
        // ... and 1.7.10 keeps that shift after they end.
        screen.run(&none, &wide);
        let ended = if v17 { shifted(480) } else { centred(480) };
        assert_eq!(screen.left(), ended, "{version}");
        // The mod's centring, on and off again, applies at once in both versions.
        screen.set_layout(true);
        screen.run(&one, &wide);
        assert_eq!(screen.left(), centred(480), "{version}");
        screen.set_layout(false);
        screen.run(&one, &wide);
        assert_eq!(screen.left(), shifted(480), "{version}");
    }
}

#[test]
fn a_1_7_10_inventory_shifts_for_effects_only_when_laid_out() {
    shifts_for_effects_only_when_laid_out(
        |version| {
            let container = layouts::player_container(0).with_game_version(version);
            Box::new(ContainerScreen::new(
                container,
                ContainerArt::Player,
                "Crafting",
            ))
        },
        176,
    );
}

#[test]
fn the_1_7_10_creative_inventory_shifts_for_effects_only_when_laid_out() {
    shifts_for_effects_only_when_laid_out(
        |version| {
            let container = layouts::player_container(0).with_game_version(version);
            Box::new(CreativeScreen::over(container))
        },
        crate::screens::container::creative_screen::X_SIZE,
    );
}

#[test]
fn the_1_7_10_anvil_name_field_takes_forty_characters() {
    for (mut screen, max) in [(open_v5(8, 9), 40), (ContainerScreen::anvil(5, "t"), 30)] {
        screen.init(&ctx());
        let field = screen.anvil_name.as_ref().expect("an anvil");
        assert_eq!(field.max_length(), max);
    }
}
