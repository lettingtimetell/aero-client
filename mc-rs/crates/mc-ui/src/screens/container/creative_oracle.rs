//! The creative inventory click oracle's Rust half (test only).
//!
//! `tests/fixtures/w21_ci_creative/creative_gui_diff.txt` was written by
//! `tests/javaref/CreativeGuiDiffRef.java`, which runs the real recompiled 1.8.9
//! `GuiContainerCreative` (its mouse and key handlers, `GuiContainer`'s state machine,
//! `ContainerCreative`, `ContainerPlayer`, `Container.slotClick` and the `CreativeCrafting`
//! listener) and records, after every input, each `C10PacketCreativeInventoryAction` the client
//! sends, the held stack, the 45 player slots and the open tab. This replays the same inputs
//! through [`CreativeScreen`] and compares every line: packets in order, and the state.
//!
//! Regenerate with `tests/javaref/generate-creative-gui-diff.ps1` (needs a JDK and the
//! recompiled vanilla classes).

use super::creative::{tab_column, tab_is_first_row};
use super::creative_screen::{CreativeScreen, CreativeSlot};
use super::layouts;
use super::model::Container;
use super::stack_ext::ItemStack;
use crate::font::FontRenderer;
use crate::screen_stack::{Screen, ScreenCtx};
use crate::widget::keys::Modifiers;
use crate::widget::test_support::uniform_font;
use crate::widget::{DrawList, ScaledResolution};

const FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/w21_ci_creative/creative_gui_diff.txt"
));

fn ctx() -> ScreenCtx {
    // The Java harness calls `setWorldAndResolution(mc, 427, 240)`.
    ScreenCtx::new(ScaledResolution::new(427, 240, 1, false), 0.0)
}

fn stack(name: &str, count: &str, damage: &str) -> ItemStack {
    let id = mc_data::item_by_name(name).unwrap_or_else(|| panic!("unknown item {name}"));
    ItemStack {
        item_id: id.0 as i16,
        count: count.parse().expect("count"),
        damage: damage.parse().expect("damage"),
        nbt: None,
    }
}

/// `<item id>x<count>@<damage>`, or `-`: the Java harness's `fmt`.
fn fmt(stack: Option<&ItemStack>) -> String {
    stack.map_or_else(
        || "-".to_owned(),
        |s| format!("{}x{}@{}", s.item_id, s.count, s.damage),
    )
}

fn mods(shift: bool, ctrl: bool) -> Modifiers {
    Modifiers {
        shift,
        ctrl,
        ..Modifiers::NONE
    }
}

fn centre(s: &CreativeScreen, slot: CreativeSlot) -> (i32, i32) {
    let (_, x, y) = s
        .slot_rects()
        .into_iter()
        .find(|(candidate, _, _)| *candidate == slot)
        .unwrap_or_else(|| panic!("{slot:?} is not on tab {}", s.selected_tab()));
    (s.gui_left() + x + 8, s.gui_top() + y + 8)
}

/// The Java harness's `at`: a cell's centre, or one of the fixed points.
fn at(s: &CreativeScreen, target: &str) -> (i32, i32) {
    let (left, top) = (s.gui_left(), s.gui_top());
    match target {
        "outside" => (left - 20, top + 40),
        "panel" => (left + 171, top + 100),
        "trash" => centre(s, CreativeSlot::Trash),
        _ if target.starts_with("tab") => {
            let index: u8 = target[3..].parse().expect("tab index");
            let column = tab_column(index);
            let mut x = 28 * column;
            if column == 5 {
                x = 195 - 28 + 2;
            } else if column > 0 {
                x += column;
            }
            let y = if tab_is_first_row(index) { -32 } else { 136 };
            (left + x + 14, top + y + 16)
        }
        _ if target.starts_with('g') => centre(
            s,
            CreativeSlot::Grid(target[1..].parse().expect("grid cell")),
        ),
        _ => centre(
            s,
            CreativeSlot::Player(target[1..].parse().expect("slot id")),
        ),
    }
}

fn state(s: &CreativeScreen) -> String {
    let mut out = format!("state {}", fmt(s.player.cursor()));
    for slot in 0..45 {
        out.push(' ');
        out.push_str(&fmt(s.player.slot_stack(slot)));
    }
    out.push_str(&format!(" tab{}", s.selected_tab()));
    out
}

fn sent(s: &mut CreativeScreen) -> String {
    let mut out = "sent".to_owned();
    for packet in s.take_outgoing() {
        out.push_str(&format!(
            " {}:{}",
            packet.slot_id,
            fmt(packet.stack.as_ref())
        ));
    }
    out
}

/// One scenario's client: the window-0 container before the screen opens, then the screen.
struct Replay {
    player: Container,
    tab: u8,
    screen: Option<CreativeScreen>,
    font: FontRenderer,
}

impl Replay {
    fn new() -> Self {
        Self {
            player: layouts::player_container(0),
            tab: 0,
            screen: None,
            font: uniform_font(),
        }
    }

    fn open(&mut self, player: Container) {
        let mut screen = CreativeScreen::over(player);
        screen.init(&ctx());
        screen.set_tab(self.tab);
        self.screen = Some(screen);
    }

    fn input(&mut self, words: &[&str]) {
        match words[0] {
            "open" => self.open(self.player.clone()),
            "reopen" => {
                // java: `onGuiClosed` (the held stack, none here, is dropped) and a new screen
                // over the same `ContainerPlayer`, on the static `selectedTabIndex`.
                let old = self.screen.take().expect("an open screen");
                self.tab = old.selected_tab();
                let mut player = old.player;
                player.set_cursor(None);
                self.open(player);
            }
            "press" | "release" | "move" => {
                let s = self.screen.as_mut().expect("an open screen");
                let (x, y) = at(s, words[1]);
                let button: i32 = words[2].parse().expect("button");
                let ms: u64 = words[words.len() - 1].parse().expect("ms");
                // `move <target> <button> <ms>` has no shift column.
                let held = mods(words.len() == 5 && words[3] == "1", false);
                s.set_clock_ms(ms);
                match words[0] {
                    "press" => {
                        let _ = s.mouse_clicked_with_mods(x, y, button, held);
                    }
                    "release" => s.mouse_released_with_mods(x, y, button, held),
                    _ => s.mouse_dragged(x, y, button, 0),
                }
            }
            "key" => {
                let s = self.screen.as_mut().expect("an open screen");
                let (x, y) = at(s, words[3]);
                s.set_clock_ms(words[4].parse().expect("ms"));
                // java: `theSlot` is whatever the last `drawScreen` found under the mouse.
                s.draw(&mut DrawList::new(), &mut self.font, &ctx(), (x, y));
                let key: i32 = words[1].parse().expect("key code");
                let _ = s.key_typed_with_font(None, key, mods(false, words[2] == "1"), &self.font);
            }
            "server" => {
                let s = self.screen.as_mut().expect("an open screen");
                let id: usize = words[1].parse().expect("slot");
                let item = (words[2] != "-").then(|| stack(words[2], words[3], words[4]));
                // java: `inventoryContainer.putStackInSlot`, an S2F write: the slot, not the
                // listener's mirror.
                s.player.set_slot_contents(id, item);
            }
            other => panic!("unknown input {other}"),
        }
    }
}

/// Every input of every scenario sends the same C10 packets, in the same order, and leaves the
/// same held stack, slots and tab as the real 1.8.9 client.
#[test]
fn every_creative_click_matches_the_real_client_packet_for_packet() {
    let mut replay = Replay::new();
    let mut scenario = String::new();
    let mut failures = Vec::new();
    let mut inputs = 0;
    let mut lines = FIXTURE.lines().filter(|line| !line.starts_with('#'));
    while let Some(line) = lines.next() {
        let words: Vec<&str> = line.split(' ').collect();
        match words[0] {
            "scenario" => {
                scenario = words[1].to_owned();
                replay = Replay::new();
            }
            "end" => {}
            "tab" => replay.tab = words[1].parse().expect("tab"),
            "slot" => replay.player.set_slot_contents(
                words[1].parse().expect("slot"),
                Some(stack(words[2], words[3], words[4])),
            ),
            "cursor" => replay
                .player
                .set_cursor(Some(stack(words[1], words[2], words[3]))),
            _ => {
                replay.input(&words);
                inputs += 1;
                let expected_sent = lines.next().expect("a sent line");
                let expected_state = lines.next().expect("a state line");
                let s = replay.screen.as_mut().expect("an open screen");
                let (got_sent, got_state) = (sent(s), state(s));
                if got_sent != expected_sent {
                    failures.push(format!(
                        "{scenario} / {line}\n  java: {expected_sent}\n  rust: {got_sent}"
                    ));
                }
                if got_state != expected_state {
                    failures.push(format!(
                        "{scenario} / {line}\n  java: {expected_state}\n  rust: {got_state}"
                    ));
                }
            }
        }
    }
    assert!(inputs > 100, "the fixture has {inputs} inputs");
    assert!(
        failures.is_empty(),
        "{} differences from the real client:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
