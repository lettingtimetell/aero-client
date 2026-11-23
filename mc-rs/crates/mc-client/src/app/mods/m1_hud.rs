//! HUD mods, client half: the numbers behind `cps`, `keystrokes`, `reach_display`,
//! `armor_status`, `item_info` and `coordinates`. The elements that draw them live in
//! `mc-ui/src/modhud/elements/`; `fps`, `ping` and `potion_status` need nothing from here.
//!
//! * **Clicks** (`cps`, `keystrokes`): every physical mouse press that reaches the game (no
//!   screen open, the vanilla rule) and matches the attack or use binding goes into a one-second
//!   ring, stamped with its receipt time ([`M1State::stamp`] moves it onto the client clock).
//! * **Reach** (`reach_display`): when `click_mouse` attacks an entity, the distance from the
//!   eye the tick's pick was cast from to the point the pick hit.
//! * **Per-frame facts** the elements cannot see for themselves: spectator mode, reduced debug
//!   info, vanilla's held-item name timer, the arrows carried and the held item's total.
//! * **Keystrokes labels**: the movement bindings' key names, checked at 20 Hz and republished
//!   only when a binding changed.
//!
//! The hooks are one line each: [`Client::m1_mouse_press`],
//! [`Client::m1_note_attack`], [`Client::m1_tick`], [`Client::m1_frame`] in `driver.rs`, and
//! [`M1State::publish`] in `fill_hud_data`. With every M1 mod off each one costs a settings
//! generation compare and a `bool` test.
//!
//! # Why `publish` takes `&mut HudData` and not `&mut Client`
//!
//! `fill_hud_data` holds `data = &mut self.mods.hud_data` across every lane's fill marker, so a
//! `&mut self` call at M1's marker would stop compiling the moment a later lane writes
//! `data.x = ...` below it. Everything that needs a `&self` method (`is_spectator`, `held_item`)
//! is gathered by `m1_frame`, which runs earlier in the same frame (`Client::frame`, before
//! `build_ui` draws the HUD), and `publish` only copies `self.m1` into `data`: two disjoint
//! fields.

use mc_math::DVec3;
use mc_mods::ModSettings;
use mc_protocol::ItemStack;
use mc_render::input::Input;
use mc_ui::modhud::{HudData, HudReach};

use super::super::{Client, Ui};

/// A press leaves the CPS count once it is this old.
const CPS_WINDOW_MS: u64 = 1000;
/// Ring capacity per button: far above any human click rate. When full, the oldest press is
/// overwritten.
const RING: usize = 64;
/// Vanilla's arrow item.
const ARROW: i16 = 262;
/// The player container's 36 main inventory slots: 27 storage, then the 9 hotbar slots.
const MAIN_SLOTS: std::ops::RangeInclusive<usize> = 9..=44;

/// The click times of one button over the last second.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ClickRing {
    times: [u64; RING],
    head: usize,
    len: usize,
}

impl Default for ClickRing {
    fn default() -> Self {
        ClickRing {
            times: [0; RING],
            head: 0,
            len: 0,
        }
    }
}

impl ClickRing {
    fn push(&mut self, at_ms: u64) {
        self.times[(self.head + self.len) % RING] = at_ms;
        if self.len == RING {
            self.head = (self.head + 1) % RING;
        } else {
            self.len += 1;
        }
    }

    /// Drop the presses at least a second old at `now_ms` and count the rest.
    fn count_at(&mut self, now_ms: u64) -> u32 {
        while self.len > 0 && now_ms.saturating_sub(self.times[self.head]) >= CPS_WINDOW_MS {
            self.head = (self.head + 1) % RING;
            self.len -= 1;
        }
        self.len as u32
    }
}

/// The state on the client (`Client::m1`).
#[derive(Debug, Default)]
pub(crate) struct M1State {
    /// The settings generation the flags below were read at; 0 = never.
    seen: u64,
    /// `cps` or `keystrokes`: presses are recorded.
    clicks_on: bool,
    /// `reach_display`.
    reach_on: bool,
    /// `armor_status` or `item_info`: arrows and the held item's total are counted.
    gear_on: bool,
    /// `keystrokes`: movement labels are kept.
    keystrokes_on: bool,
    /// `coordinates` or the gear mods: the per-frame facts are gathered.
    frame_on: bool,
    /// Attack, use.
    clicks: [ClickRing; 2],
    /// Counted attack and use presses since start (`HudData::click_presses`).
    presses: [u64; 2],
    /// The newest stamp handed out; stamps never go backwards.
    last_stamp: u64,
    /// The input clock's offset from the client clock, best estimate so far.
    clock_offset: Option<i64>,
    reach: Option<HudReach>,
    spectator: bool,
    reduced_debug_info: bool,
    highlight_ticks: i32,
    arrow_count: u32,
    held_total: u32,
    /// The movement bindings the labels were built from; `None` = rebuild.
    move_inputs: Option<[Option<Input>; 4]>,
    labels: [String; 4],
    /// `labels` changed since `publish` last copied them.
    labels_dirty: bool,
}

impl M1State {
    /// Re-read which M1 mods are on. One `u64` compare while nothing changed.
    fn sync(&mut self, settings: &ModSettings) {
        if self.seen == settings.generation() {
            return;
        }
        self.seen = settings.generation();
        self.keystrokes_on = settings.enabled("keystrokes");
        self.clicks_on = self.keystrokes_on || settings.enabled("cps");
        self.reach_on = settings.enabled("reach_display");
        self.gear_on = settings.enabled("armor_status") || settings.enabled("item_info");
        self.frame_on = self.gear_on || settings.enabled("coordinates");
        if !self.keystrokes_on {
            // Rebuilt from the bindings when keystrokes comes back on.
            self.move_inputs = None;
        }
    }

    /// Move a receipt time on the input clock (`TimedMouseEvent::at_ms`: milliseconds since the
    /// shell's `InputState` was created) onto the client clock (`Client::started`, which
    /// `HudData::now_ms` reads), `now_ms` being the client clock as the event is handled.
    ///
    /// The two clocks differ by a fixed offset. An event is always handled after it arrived, so
    /// `now_ms - at_ms` overestimates that offset by the handling delay, and the smallest value
    /// seen is the best estimate. A stamp is never later than `now_ms` and never earlier than
    /// the stamp before it.
    fn stamp(&mut self, at_ms: u64, now_ms: u64) -> u64 {
        let observed = now_ms as i64 - at_ms as i64;
        let offset = match self.clock_offset {
            Some(offset) if offset <= observed => offset,
            _ => {
                self.clock_offset = Some(observed);
                observed
            }
        };
        let stamp = (at_ms as i64 + offset).clamp(0, now_ms as i64) as u64;
        self.last_stamp = stamp.max(self.last_stamp);
        self.last_stamp
    }

    /// One counted press of `slot` (0 attack, 1 use) at client time `at_ms`.
    fn record(&mut self, slot: usize, at_ms: u64) {
        self.clicks[slot].push(at_ms);
        self.presses[slot] = self.presses[slot].wrapping_add(1);
    }

    /// Copy this frame's M1 numbers into `data`. Called from `fill_hud_data`, after `data.now_ms` is set, only while a HUD element is enabled.
    pub(crate) fn publish(&mut self, data: &mut HudData) {
        if self.clicks_on {
            data.cps_left = self.clicks[0].count_at(data.now_ms);
            data.cps_right = self.clicks[1].count_at(data.now_ms);
            data.click_presses = self.presses;
        }
        if self.reach_on {
            data.reach = self.reach;
        }
        if self.frame_on {
            data.spectator = self.spectator;
            data.reduced_debug_info = self.reduced_debug_info;
            data.held_highlight_ticks = self.highlight_ticks;
            data.arrow_count = self.arrow_count;
            data.held_total = self.held_total;
        }
        if self.labels_dirty {
            self.labels_dirty = false;
            for (to, from) in data.move_key_labels.iter_mut().zip(&self.labels) {
                to.clone_from(from);
            }
        }
    }
}

/// `|hit_vec - eye|`: how far from the eye the look ray met the target's collision-border
/// expanded box (vanilla's `objectMouseOver.hitVec`, `0` when the eye is inside it).
#[must_use]
fn reach_distance(eye: DVec3, hit_vec: DVec3) -> f64 {
    mc_math::vec::distance_to(hit_vec, eye)
}

/// `(arrows, held total)` over the 36 main inventory slots: every arrow, and every stack with
/// the held stack's item and damage (the held stack included).
#[must_use]
fn inventory_counts<'a>(
    slot: impl Fn(usize) -> Option<&'a ItemStack>,
    held: Option<&ItemStack>,
) -> (u32, u32) {
    let (mut arrows, mut total) = (0_u32, 0_u32);
    for stack in MAIN_SLOTS.filter_map(slot) {
        let count = u32::try_from(stack.count).unwrap_or(0);
        if stack.item_id == ARROW {
            arrows = arrows.saturating_add(count);
        }
        if held.is_some_and(|held| held.item_id == stack.item_id && held.damage == stack.damage) {
            total = total.saturating_add(count);
        }
    }
    (arrows, total)
}

/// The keystrokes label for a binding: its key name when that is 1 to 3 characters (`W`,
/// `UP`), else empty so the element draws its default letter.
fn short_label(out: &mut String, input: Option<Input>) {
    out.clear();
    if let Some(input) = input.filter(|input| input.is_bound()) {
        let name = input.display_name();
        if (1..=3).contains(&name.chars().count()) {
            out.push_str(&name);
        }
    }
}

impl Client {
    /// A physical mouse press, `at_ms` on the input clock. Counted when no
    /// screen is open (vanilla: an open screen consumes mouse events before the bindings see
    /// them) and the button is what attack or use is bound to.
    pub(crate) fn m1_mouse_press(&mut self, button: u16, at_ms: u64) {
        self.m1.sync(&self.mods.settings);
        if !self.m1.clicks_on || !matches!(self.ui, Ui::InGame) {
            return;
        }
        use mc_render::input::keys as bind;
        let pressed = Some(Input::Mouse(button));
        let bound = |name: &str| self.settings.keys.get(name).map(|b| b.binding()) == pressed;
        let slot = if bound(bind::ATTACK) {
            0
        } else if bound(bind::USE_ITEM) {
            1
        } else {
            return;
        };
        let now_ms = self.started.elapsed().as_millis() as u64;
        let at_ms = self.m1.stamp(at_ms, now_ms);
        self.m1.record(slot, at_ms);
    }

    /// `click_mouse` attacks the entity this tick's pick hit at `hit_vec`.
    pub(crate) fn m1_note_attack(&mut self, hit_vec: DVec3) {
        self.m1.sync(&self.mods.settings);
        if !self.m1.reach_on {
            return;
        }
        // The eye `update_hit_result` cast the pick from (`getMouseOver(1.0F)`). The player has
        // not moved since: the pick ran at the start of this tick, before the input.
        let Some(eye) = self
            .session
            .as_ref()
            .and_then(|session| session.render_view_eye_position(1.0))
        else {
            return;
        };
        self.m1.reach = Some(HudReach {
            distance: reach_distance(eye, hit_vec),
            at_ms: self.started.elapsed().as_millis() as u64,
        });
    }

    /// The keystrokes movement labels, rebuilt only when one of the
    /// four movement bindings changed.
    pub(crate) fn m1_tick(&mut self) {
        self.m1.sync(&self.mods.settings);
        if !self.m1.keystrokes_on {
            return;
        }
        use mc_render::input::keys as bind;
        let inputs = [bind::FORWARD, bind::LEFT, bind::BACK, bind::RIGHT]
            .map(|name| self.settings.keys.get(name).map(|b| b.binding()));
        if self.m1.move_inputs == Some(inputs) {
            return;
        }
        self.m1.move_inputs = Some(inputs);
        for (label, input) in self.m1.labels.iter_mut().zip(inputs) {
            short_label(label, input);
        }
        self.m1.labels_dirty = true;
    }

    /// The facts the elements need that only `&self` methods know. Runs in
    /// `Client::frame` before `build_ui`, so `publish` copies this frame's values.
    pub(crate) fn m1_frame(&mut self) {
        self.m1.sync(&self.mods.settings);
        if !self.m1.frame_on {
            return;
        }
        self.m1.spectator = self.is_spectator();
        // F3's own test: the server's flag or the "Reduced Debug Info" chat option.
        self.m1.reduced_debug_info = self.debug_overlay_reduced();
        self.m1.highlight_ticks = self.hud.highlight_ticks;
        if self.m1.gear_on {
            let (arrows, total) =
                inventory_counts(|i| self.player_container.slot_stack(i), self.held_item());
            self.m1.arrow_count = arrows;
            self.m1.held_total = total;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::config::{ClientConfig, LoginMode};
    use crate::app::interact::{EntityHit, HitResult};

    fn stack(item_id: i16, damage: i16, count: i32) -> ItemStack {
        ItemStack {
            item_id,
            count,
            damage,
            nbt: None,
        }
    }

    #[test]
    fn the_cps_window_is_one_second_and_the_ring_never_grows() {
        let mut ring = ClickRing::default();
        for t in [0, 100, 200, 999] {
            ring.push(t);
        }
        assert_eq!(ring.count_at(999), 4);
        assert_eq!(
            ring.count_at(1000),
            3,
            "a press exactly 1000 ms old has left"
        );
        assert_eq!(ring.count_at(1998), 1);
        assert_eq!(ring.count_at(1999), 0);
        // 100 presses inside 100 ms (a macro, or a test): the newest 64 are kept.
        for t in 0..100 {
            ring.push(5000 + t);
        }
        assert_eq!(ring.count_at(5100), RING as u32);
        assert_eq!(ring.times[ring.head], 5036, "the oldest kept press");
        assert_eq!(ring.count_at(6036), 63);
    }

    #[test]
    fn stamps_move_the_input_clock_onto_the_client_clock() {
        let mut state = M1State::default();
        // The shell's input clock started 250 ms after the client's. The first press is
        // handled 8 ms after it arrived: the estimate starts 8 ms high.
        assert_eq!(state.stamp(1000, 1258), 1258);
        // A press handled 1 ms after arrival tightens it to 251.
        assert_eq!(state.stamp(2000, 2251), 2251);
        // Two presses handled together in a late frame keep their own receipt times.
        assert_eq!(state.stamp(3000, 3300), 3251);
        assert_eq!(state.stamp(3040, 3300), 3291);
        // An input clock that started *before* the client's (a negative offset) works too.
        let mut early = M1State::default();
        assert_eq!(early.stamp(1500, 1003), 1003);
        // Handled sooner after arrival: the offset tightens from -497 to -500.
        assert_eq!(early.stamp(1600, 1100), 1100);
        assert_eq!(
            early.stamp(1700, 1250),
            1200,
            "receipt time, not handling time"
        );
    }

    #[test]
    fn reach_is_eye_to_hit_point_and_counts_cover_the_main_inventory() {
        // Vanilla's `Vec3.distanceTo` goes through `MathHelper.sqrt_double`, which returns a
        // float: the same f32-precision number the pick compared against 3.0.
        let eye = DVec3::new(0.5, 65.62, 0.5);
        let hit = DVec3::new(0.5, 65.62, 3.37);
        assert!((reach_distance(eye, hit) - 2.87).abs() < 1e-6);
        assert_eq!(reach_distance(eye, hit), f64::from(2.87_f32));
        // 3-4-5 in the horizontal plane.
        assert!((reach_distance(eye, eye + DVec3::new(1.8, 0.0, 2.4)) - 3.0).abs() < 1e-6);
        assert_eq!(
            reach_distance(eye, eye),
            0.0,
            "eye inside the box: vanilla's hitVec = eye"
        );

        let mut slots: Vec<Option<ItemStack>> = vec![None; 45];
        slots[5] = Some(stack(ARROW, 0, 64)); // an armour slot: not counted
        slots[9] = Some(stack(ARROW, 0, 16));
        slots[20] = Some(stack(368, 0, 4)); // ender pearls
        slots[36] = Some(stack(368, 0, 3)); // hotbar: the held stack
        slots[44] = Some(stack(ARROW, 0, 7));
        slots[30] = Some(stack(368, 1, 9)); // same id, other damage: another item
        let held = slots[36].clone();
        let (arrows, total) =
            inventory_counts(|i| slots.get(i).and_then(Option::as_ref), held.as_ref());
        assert_eq!((arrows, total), (23, 7));
        assert_eq!(
            inventory_counts(|i| slots.get(i).and_then(Option::as_ref), None),
            (23, 0)
        );
    }

    #[test]
    fn short_labels_keep_one_to_three_characters() {
        use winit::keyboard::KeyCode;
        let mut out = String::from("stale");
        short_label(&mut out, Some(Input::Key(KeyCode::KeyW)));
        assert_eq!(out, "W");
        short_label(&mut out, Some(Input::Key(KeyCode::ArrowUp)));
        assert_eq!(out, "UP");
        short_label(&mut out, Some(Input::Key(KeyCode::ShiftLeft)));
        assert_eq!(out, "", "LSHIFT is too long: the default letter is drawn");
        short_label(&mut out, Some(Input::Unbound));
        assert_eq!(out, "");
        short_label(&mut out, None);
        assert_eq!(out, "");
    }

    // ---- with a real client ------------------------------------------------------------------

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

    /// A joined session (the predictor exists) with the player's feet at `feet`. Keep the
    /// returned receiver alive: it is where the session's packets go.
    fn join(
        client: &mut Client,
        feet: DVec3,
    ) -> tokio::sync::mpsc::UnboundedReceiver<crate::net::ToNet> {
        let (handle, to_game, to_net) = crate::net::channel::NetHandle::new_pair();
        let mut session =
            crate::session::Session::new(handle, crate::session::JoinSettings::default());
        to_game
            .try_send(crate::net::ToGame::Packet(
                mc_protocol::play::S2CPlay::JoinGame(mc_protocol::play::s2c::JoinGame {
                    entity_id: 1,
                    hardcore_mode: false,
                    game_mode: 0,
                    dimension: 0,
                    difficulty: 2,
                    max_players: 20,
                    level_type: "default".into(),
                    reduced_debug_info: false,
                }),
            ))
            .expect("channel");
        session.poll(16);
        let predictor = session
            .predictor_mut()
            .expect("JoinGame builds the predictor");
        predictor.state.pos = feet;
        predictor.state.prev_pos = feet;
        client.session = Some(session);
        to_net
    }

    #[test]
    fn presses_count_in_game_only_and_follow_the_bindings() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        client.ui = Ui::InGame;
        // Off: nothing is recorded.
        client.m1_mouse_press(0, 10);
        assert_eq!(client.m1.presses, [0, 0]);

        client.mods.settings.set_enabled("cps", true);
        client.m1_mouse_press(0, 20);
        client.m1_mouse_press(0, 30);
        client.m1_mouse_press(1, 40);
        client.m1_mouse_press(2, 50); // pick block: neither attack nor use
        assert_eq!(client.m1.presses, [2, 1]);
        // A screen consumes the click.
        client.ui = Ui::Loading;
        client.m1_mouse_press(0, 60);
        assert_eq!(client.m1.presses, [2, 1]);
        client.ui = Ui::InGame;
        // A remapped attack counts its own button, not the left one.
        client
            .settings
            .keys
            .get_mut(mc_render::input::keys::ATTACK)
            .unwrap()
            .set_binding(Input::Mouse(4));
        client.m1_mouse_press(0, 70);
        client.m1_mouse_press(4, 80);
        assert_eq!(client.m1.presses, [3, 1]);

        // The frame hook and F's fill publish the counts.
        client.m1_frame();
        client.fill_hud_data();
        let data = &client.mods.hud_data;
        assert_eq!((data.cps_left, data.cps_right), (3, 1));
        assert_eq!(data.click_presses, [3, 1]);
    }

    #[test]
    fn an_attack_records_the_eye_to_hit_distance() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let _net = join(&mut client, DVec3::new(0.5, 64.0, 0.5));
        let eye_y = 64.0 + f64::from(mc_physics::state::PLAYER_EYE_HEIGHT);
        let hit_vec = DVec3::new(0.5, eye_y, 3.37);
        // Off: nothing is recorded.
        client.m1_note_attack(hit_vec);
        assert!(client.m1.reach.is_none());

        client.mods.settings.set_enabled("reach_display", true);
        client.m1_note_attack(hit_vec);
        let reach = client.m1.reach.expect("recorded");
        assert!((reach.distance - 2.87).abs() < 1e-6, "{reach:?}");

        // Through the real click path: `click_mouse` on an entity hit reaches the hook.
        client.m1.reach = None;
        client.hit = HitResult::Entity(EntityHit {
            network_id: 7,
            hit_vec: DVec3::new(0.5, eye_y, 2.5),
        });
        client.click_mouse();
        let reach = client.m1.reach.expect("click_mouse calls m1_note_attack");
        assert!((reach.distance - 2.0).abs() < 1e-6, "{reach:?}");
        client.m1_frame();
        client.fill_hud_data();
        assert_eq!(client.mods.hud_data.reach, Some(reach));
    }

    #[test]
    fn frame_facts_and_labels_reach_the_hud_data() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let _net = join(&mut client, DVec3::new(0.0, 64.0, 0.0));
        client.mods.settings.set_enabled("armor_status", true);
        client.mods.settings.set_enabled("keystrokes", true);
        client
            .player_container
            .set_slot_contents(10, Some(stack(ARROW, 0, 20)));
        client
            .player_container
            .set_slot_contents(40, Some(stack(ARROW, 0, 3)));
        client.hud.highlight_ticks = 17;
        client
            .settings
            .keys
            .get_mut(mc_render::input::keys::FORWARD)
            .unwrap()
            .set_binding(Input::Key(winit::keyboard::KeyCode::ArrowUp));
        client.m1_tick();
        client.m1_frame();
        client.fill_hud_data();
        let data = &client.mods.hud_data;
        assert_eq!(data.arrow_count, 23);
        assert_eq!(data.held_highlight_ticks, 17);
        assert!(!data.spectator);
        assert_eq!(
            data.move_key_labels,
            ["UP", "A", "S", "D"].map(String::from),
            "the bound key names"
        );
    }
}
