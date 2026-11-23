//! "1.7 Animations": the client half.
//!
//! The settings snapshot, the hooks `driver.rs` calls, and the smooth-sneak camera state. The
//! matrices live in [`mc_render::pipeline::old_animations`]. Every sub-option is visual: the
//! only input-side hook, `punch_during_use`, starts the local swing through the packet-free
//! [`Client::start_first_person_swing`] and sends nothing (see the packet test below).
//!
//! Cost with the mod off: one `u64` generation compare per hook call, a few `bool` reads and
//! one `Option` store per frame. The third-person switches live in mc-render thread-locals that
//! are written only when the settings change.

use mc_render::pipeline::old_animations::{self, OldAnimations};
use mc_render::Camera;

use super::super::Client;
use crate::app::interact::{self, HitResult};

/// The settings-file key.
const MOD_ID: &str = "old_animations";

/// `ItemCarrotOnAStick`'s 1.8.9 id. With the fishing rod it is one of the two items whose
/// `Item.shouldRotateAroundWhenRendering()` is true.
const CARROT_ON_A_STICK_ITEM_ID: i16 = 398;

/// The state on [`Client`] (`self.m5`).
#[derive(Debug, Default)]
pub(crate) struct M5State {
    /// The settings generation this snapshot was taken at; 0 = never synced.
    seen: u64,
    /// The first-person switches, or `None` when the mod or all four of them are off.
    first_person: Option<OldAnimations>,
    punch_during_use: bool,
    sword_block_3p: bool,
    armor_red: bool,
    smooth_sneak: bool,
    old_health: bool,
    sneak: SmoothSneak,
    /// Visual-only swings started by `punch_during_use`; read by tests.
    punches: u64,
}

impl M5State {
    /// Refresh the snapshot; `true` when the generation moved (the caller then pushes the
    /// switches that live outside this struct).
    fn sync(&mut self, settings: &mc_mods::ModSettings) -> bool {
        if self.seen == settings.generation() {
            return false;
        }
        self.seen = settings.generation();
        let on = settings.enabled(MOD_ID);
        let option = |id: &str| on && settings.bool(MOD_ID, id);
        let first_person = OldAnimations {
            blockhit: option("blockhit"),
            item_positions: option("item_positions"),
            bow: option("bow"),
            rod: option("rod"),
            rotate_around: false,
        };
        self.first_person = (first_person != OldAnimations::default()).then_some(first_person);
        self.punch_during_use = option("punch_during_use");
        self.sword_block_3p = option("sword_block_3p");
        self.armor_red = option("armor_red");
        self.smooth_sneak = option("smooth_sneak");
        self.old_health = option("old_health");
        if !self.smooth_sneak {
            self.sneak = SmoothSneak::default();
        }
        true
    }
}

/// `smooth_sneak`: the render camera's eye height eased per tick.
///
/// Per tick `h` follows the physics eye height: a crouch (`target < h`) snaps at the tick, an
/// un-sneak closes half the gap (0.04, 0.02, 0.01 ...). Frames draw `lerp(prev, h, partial)`.
/// Only the render camera reads it; the pick ray, the sky, sound and physics keep vanilla's
/// eye (none of them reads `Camera::pos`).
#[derive(Debug, Default, Clone, Copy)]
struct SmoothSneak {
    /// `(prev, h)`, or `None` to snap to the next target seen.
    eye: Option<(f32, f32)>,
    /// The client tick `eye` was last advanced to.
    tick: u64,
    /// The predictor's correction count at that tick (a teleport snaps).
    corrections: u64,
}

/// Advance `(prev, h)` by `ticks` client ticks toward `target`.
fn advance_sneak_eye(eye: (f32, f32), target: f32, ticks: u64) -> (f32, f32) {
    let (mut prev, mut h) = eye;
    for _ in 0..ticks.min(20) {
        prev = h;
        h = if target < h {
            target
        } else {
            h + (target - h) * 0.5
        };
    }
    (prev, h)
}

/// Whether a held stack is drawn turned round (`Item.shouldRotateAroundWhenRendering`).
fn rotates_around(stack: &mc_protocol::ItemStack) -> bool {
    stack.item_id == interact::FISHING_ROD_ITEM_ID || stack.item_id == CARROT_ON_A_STICK_ITEM_ID
}

impl Client {
    /// Sync the snapshot and, on a change, push the switches that live elsewhere: the
    /// mc-render thread-locals (third-person sword block, armour red) and the HUD's ghost-heart
    /// flag.
    fn m5_sync(&mut self) {
        if self.m5.sync(&self.mods.settings) {
            old_animations::set_third_person(self.m5.sword_block_3p, self.m5.armor_red);
            self.hud.hide_lost_health_hearts = self.m5.old_health;
        }
    }

    /// `driver.rs` `Client::frame`, right after `place_camera` (the first M5 hook of a frame):
    /// sync the settings, then apply the smooth-sneak eye height to the render camera.
    pub(crate) fn m5_frame_camera(&mut self, camera: &mut Camera, partial_ticks: f32) {
        self.m5_sync();
        if !self.m5.smooth_sneak {
            return;
        }
        let tick = self.tick;
        let Some((target, corrections)) = self.session.as_ref().and_then(|session| {
            if !session.render_view_is_local() || session.state().local.sleeping {
                return None;
            }
            let predictor = session.predictor()?;
            let state = &predictor.state;
            Some((
                (state.eye_pos().y - state.pos.y) as f32,
                predictor.corrections,
            ))
        }) else {
            self.m5.sneak.eye = None;
            return;
        };
        let sneak = &mut self.m5.sneak;
        let eye = match sneak.eye {
            Some(eye) if sneak.corrections == corrections => {
                advance_sneak_eye(eye, target, tick.wrapping_sub(sneak.tick))
            }
            _ => (target, target),
        };
        sneak.eye = Some(eye);
        sneak.tick = tick;
        sneak.corrections = corrections;
        let (prev, h) = eye;
        let smoothed = prev + (h - prev) * partial_ticks;
        camera.pos.y += f64::from(smoothed - target);
    }

    /// `drive_first_person`: this frame's first-person switches.
    ///
    /// `latched` is the phase's `item_to_render`. The rod bit is classified only while the
    /// latched model is the current stack; during an equip transition the outgoing model keeps
    /// the previous frame's bit (`previous`), as the model itself stays latched until the swap.
    pub(crate) fn m5_first_person(
        &self,
        latched: Option<u64>,
        previous: Option<OldAnimations>,
    ) -> Option<OldAnimations> {
        let config = self.m5.first_person?;
        // `latched` is now the latched stack *object's* container identity (Java's
        // `itemToRender` reference); while it is the held slot's object, classify that stack.
        let rotate_around = if latched.is_some() && latched == self.held_item_identity() {
            self.first_person_latch.value().is_some_and(rotates_around)
        } else {
            previous.is_some_and(|previous| previous.rotate_around)
        };
        Some(OldAnimations {
            rotate_around,
            ..config
        })
    }

    /// The per-tick dig hook (`Minecraft.sendClickBlockToController`), before vanilla's
    /// "not using an item" gate: `punch_during_use`. Holding attack on a block while blocking,
    /// eating, drinking or drawing a bow shows the swing and the crack particles, as 1.7 did.
    ///
    /// Visual only. It calls the packet-free [`Client::start_first_person_swing`] (the local arm
    /// and the F5 body) and the client-side hit particles, and nothing that sends: no
    /// `C0APacketAnimation`, no `C07` dig, no `C09` slot sync. The vanilla body then runs
    /// unchanged and returns at its use gate, because an item is in use.
    pub(crate) fn m5_punch_during_use(&mut self, left_click: bool) {
        self.m5_sync();
        if !self.m5.punch_during_use
            || !left_click
            || !self.controller.is_using_item()
            || !self.controller.left_click_ready()
        {
            return;
        }
        let HitResult::Block(hit) = self.hit else {
            return;
        };
        let Some(session) = self.session.as_ref() else {
            return;
        };
        if session.state().world.block_at(hit.pos).props().is_air() {
            return;
        }
        mc_render::particle::packet::add_block_hit_effects(
            &mut self.particles,
            &session.state().world,
            hit.pos,
            hit.face,
            &crate::app::particles::AtlasIcons,
        );
        self.start_first_person_swing();
        self.m5.punches += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::config::{ClientConfig, LoginMode};
    use crate::app::driver::HOTBAR_SLOT_OFFSET;
    use crate::net::{ToGame, ToNet};
    use crate::session::{JoinSettings, Session};
    use mc_math::{BlockPos, DVec3};
    use mc_protocol::play::C2SPlay;
    use tokio::sync::mpsc::UnboundedReceiver;

    #[test]
    fn smooth_sneak_snaps_down_and_eases_up() {
        // Crouch: the target drops below h, so h snaps on the next tick (frames ramp it).
        assert_eq!(advance_sneak_eye((1.62, 1.62), 1.54, 1), (1.62, 1.54));
        // Stand up: half the gap per tick.
        let (prev, h) = advance_sneak_eye((1.54, 1.54), 1.62, 1);
        assert!((prev - 1.54).abs() < 1e-6);
        assert!((h - 1.58).abs() < 1e-6);
        let (prev, h) = advance_sneak_eye((prev, h), 1.62, 1);
        assert!((prev - 1.58).abs() < 1e-6 && (h - 1.60).abs() < 1e-6);
        // Several ticks between two frames advance several steps; no tick, no change.
        let (_, h) = advance_sneak_eye((1.54, 1.54), 1.62, 3);
        assert!((h - 1.61).abs() < 1e-6);
        assert_eq!(advance_sneak_eye((1.5, 1.6), 1.62, 0), (1.5, 1.6));
    }

    #[test]
    fn only_the_rod_and_the_carrot_on_a_stick_rotate_around() {
        assert_eq!(
            mc_data::item_by_name("minecraft:carrot_on_a_stick"),
            Some(mc_data::ItemId(CARROT_ON_A_STICK_ITEM_ID as u16))
        );
        assert_eq!(
            mc_data::item_by_name("minecraft:fishing_rod"),
            Some(mc_data::ItemId(interact::FISHING_ROD_ITEM_ID as u16))
        );
        let stack = |item_id| mc_protocol::ItemStack {
            item_id,
            count: 1,
            damage: 0,
            nbt: None,
        };
        assert!(rotates_around(&stack(346)));
        assert!(rotates_around(&stack(398)));
        assert!(!rotates_around(&stack(276)), "a diamond sword");
        assert!(!rotates_around(&stack(261)), "a bow");
    }

    #[test]
    fn snapshot_follows_the_switches_and_is_empty_while_off() {
        let mut settings = mc_mods::ModSettings::default();
        let mut state = M5State::default();
        assert!(state.sync(&settings));
        assert!(!state.sync(&settings), "unchanged generation: no work");
        assert_eq!(state.first_person, None);
        assert!(!state.punch_during_use && !state.sword_block_3p && !state.armor_red);
        assert!(!state.smooth_sneak && !state.old_health);

        settings.set_enabled(MOD_ID, true);
        assert!(state.sync(&settings));
        assert_eq!(
            state.first_person,
            Some(OldAnimations {
                blockhit: true,
                item_positions: true,
                bow: true,
                rod: true,
                rotate_around: false,
            })
        );
        assert!(state.punch_during_use && state.sword_block_3p && state.armor_red);
        assert!(state.smooth_sneak);
        assert!(!state.old_health, "old_health defaults off");

        for id in ["blockhit", "item_positions", "bow", "rod"] {
            settings.set_bool(MOD_ID, id, false);
        }
        state.sync(&settings);
        assert_eq!(
            state.first_person, None,
            "all four first-person switches off"
        );

        settings.set_enabled(MOD_ID, false);
        settings.set_bool(MOD_ID, "old_health", true);
        state.sync(&settings);
        assert!(
            !state.old_health && !state.armor_red,
            "the mod switch gates every option"
        );
    }

    /// The hooks stay wired where they must be, and this lane's production code never names a
    /// packet-sending API (the runtime proof is the packet test below).
    #[test]
    fn hooks_are_wired_and_the_lane_never_names_a_packet_api() {
        let driver = include_str!("../driver.rs")
            .split_once("#[cfg(test)]\nmod tests {")
            .expect("driver keeps its inline test module")
            .0;
        let (_, click) = driver
            .split_once("fn send_click_block_to_controller(&mut self, left_click: bool) {")
            .expect("the per-tick dig");
        let punch = click
            .find("self.m5_punch_during_use(left_click);")
            .expect("punch hook");
        let gate = click
            .find("if !self.controller.may_damage_block() {")
            .expect("vanilla gate");
        assert!(
            punch < gate,
            "the visual swing runs before vanilla's use gate returns"
        );
        for hook in [
            "self.m5_frame_camera(camera, partial_ticks);",
            "self.m5_first_person(phase.render_state.item_to_render, phase.old_animations);",
            "mc_render::pipeline::old_animations::third_person_held_item_matrix(",
            "mc_render::pipeline::old_animations::hurt_flash_combines(layer)",
            "pub(crate) m5: mods::m5_animations::M5State,",
        ] {
            assert!(driver.contains(hook), "missing hook: {hook}");
        }

        let production = include_str!("m5_animations.rs")
            .split_once("#[cfg(test)]")
            .expect("this file has a test module")
            .0;
        for packet_api in [
            "swing_arm",
            "player_action",
            "damage_block",
            "click_block",
            "reset_block_removing",
            "sync_held_item",
            "send(",
        ] {
            assert!(
                !production.contains(packet_api),
                "M5 must not call {packet_api}"
            );
        }
    }

    // ---- punch_during_use: the packet proof ------------------------------------------------

    /// The client, its game dir (which must outlive it) and the network side of its session.
    type Fixture = (Client, tempfile::TempDir, UnboundedReceiver<ToNet>);

    fn diamond_sword() -> mc_protocol::ItemStack {
        mc_protocol::ItemStack {
            item_id: 276,
            count: 1,
            damage: 0,
            nbt: None,
        }
    }

    /// A client on the real assets (`None` = skip without the vanilla jar), in a world with
    /// stone at `(3, 65, 4)` under the crosshair, blocking with a diamond sword in hotbar slot 0.
    fn client_blocking_at_stone() -> Option<Fixture> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("SKIP: no vanilla jar in this checkout");
            return None;
        }
        let mut client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        let pos = BlockPos::new(3, 65, 4);
        let mut world = mc_render::terrain::fixture::blank_world(1);
        mc_render::terrain::fixture::place(&mut world, pos, "minecraft:stone");
        let (handle, to_game, mut to_net) = crate::net::channel::NetHandle::new_pair();
        let mut session = Session::new(handle, JoinSettings::default());
        to_game
            .try_send(ToGame::Packet(mc_protocol::play::S2CPlay::JoinGame(
                mc_protocol::play::s2c::JoinGame {
                    entity_id: 1,
                    hardcore_mode: false,
                    game_mode: 0,
                    dimension: 0,
                    difficulty: 2,
                    max_players: 20,
                    level_type: "default".into(),
                    reduced_debug_info: false,
                },
            )))
            .expect("channel");
        session.poll(16);
        session.state_mut().world = world;
        client.session = Some(session);
        client.player_hud.selected_slot = 0;
        client
            .player_container
            .set_slot_contents(HOTBAR_SLOT_OFFSET, Some(diamond_sword()));
        client.refresh_hotbar();
        client.hit = HitResult::Block(interact::BlockHit {
            pos,
            face: mc_math::Facing::Up,
            hit_vec: DVec3::new(3.5, 66.0, 4.5),
        });
        // Sword block: `EntityPlayer.setItemInUse(stack, 72000)`.
        client.controller.set_item_in_use(diamond_sword(), 72_000);
        drop(to_game);
        let _ = sent(&mut to_net);
        Some((client, dir, to_net))
    }

    fn sent(to_net: &mut UnboundedReceiver<ToNet>) -> Vec<C2SPlay> {
        let mut out = Vec::new();
        while let Ok(message) = to_net.try_recv() {
            if let ToNet::Send(packet) = message {
                out.push(packet);
            }
        }
        out
    }

    /// Twenty ticks of "attack held on stone while blocking", the way the driver calls the
    /// per-tick dig (`send_click_block_to_controller(attack_held)`), with the controller's
    /// per-tick counters advancing between them. Returns every packet sent.
    fn hold_attack_while_blocking(
        client: &mut Client,
        to_net: &mut UnboundedReceiver<ToNet>,
    ) -> Vec<C2SPlay> {
        let mut all = Vec::new();
        for _ in 0..20 {
            client.send_click_block_to_controller(true);
            client.controller.tick();
            all.extend(sent(to_net));
        }
        all
    }

    /// The rule for every visual mod, proven for the one hook that observes input: holding
    /// attack on a block while blocking sends exactly what vanilla sends (nothing: the dig is
    /// gated off while an item is in use) — zero C0A swings, zero C07 digs, zero C09 slot syncs —
    /// while the visual swing does start.
    #[test]
    fn punch_during_use_swings_locally_and_sends_no_packet() {
        let Some((mut vanilla, _vanilla_dir, mut vanilla_net)) = client_blocking_at_stone() else {
            return;
        };
        let Some((mut modded, _modded_dir, mut modded_net)) = client_blocking_at_stone() else {
            return;
        };
        modded.mods.settings.set_enabled(MOD_ID, true);
        assert!(vanilla.controller.is_using_item() && modded.controller.is_using_item());

        let vanilla_packets = hold_attack_while_blocking(&mut vanilla, &mut vanilla_net);
        let modded_packets = hold_attack_while_blocking(&mut modded, &mut modded_net);

        for packets in [&vanilla_packets, &modded_packets] {
            let swings = packets
                .iter()
                .filter(|p| matches!(p, C2SPlay::Animation(_)))
                .count();
            let digs = packets
                .iter()
                .filter(|p| matches!(p, C2SPlay::PlayerDigging(_)))
                .count();
            let slot_syncs = packets
                .iter()
                .filter(|p| matches!(p, C2SPlay::HeldItemChange(_)))
                .count();
            assert_eq!((swings, digs, slot_syncs), (0, 0, 0), "{packets:?}");
        }
        assert_eq!(
            modded_packets, vanilla_packets,
            "the mod must not change the packet stream"
        );
        assert_eq!(vanilla.m5.punches, 0, "mod off: no visual swing");
        assert_eq!(
            modded.m5.punches, 20,
            "mod on: the swing is requested every held tick"
        );
        assert!(
            modded.particles.len() > vanilla.particles.len(),
            "the crack particles show"
        );

        // Not using an item: the hook stands aside and vanilla's own dig (with its real C07s
        // and C0A swings) is byte-identical with the mod on or off.
        vanilla.controller.stop_using_item();
        modded.controller.stop_using_item();
        let vanilla_dig = hold_attack_while_blocking(&mut vanilla, &mut vanilla_net);
        let modded_dig = hold_attack_while_blocking(&mut modded, &mut modded_net);
        assert!(!vanilla_dig.is_empty(), "vanilla digs the stone");
        assert_eq!(modded_dig, vanilla_dig);
        assert_eq!(
            modded.m5.punches, 20,
            "no extra visual swing while not using an item"
        );

        // The sub-option off: no visual swing either.
        modded.controller.set_item_in_use(diamond_sword(), 72_000);
        modded
            .mods
            .settings
            .set_bool(MOD_ID, "punch_during_use", false);
        let _ = hold_attack_while_blocking(&mut modded, &mut modded_net);
        assert_eq!(modded.m5.punches, 20);
    }

    /// The client-level hooks: the sync pushes the switches that live outside `M5State` (the
    /// mc-render thread-locals and the HUD flag), and the first-person hook latches the rod bit
    /// only for the model that is actually latched.
    #[test]
    fn client_sync_pushes_switches_and_latches_the_rod_with_the_model() {
        let Some((mut client, _dir, _net)) = client_blocking_at_stone() else {
            return;
        };
        client.m5_sync();
        assert_eq!(client.m5_first_person(None, None), None, "mod off: vanilla");
        assert!(!old_animations::third_person_sword_block() && !old_animations::armor_red());
        assert!(!client.hud.hide_lost_health_hearts);

        client.mods.settings.set_enabled(MOD_ID, true);
        client.mods.settings.set_bool(MOD_ID, "old_health", true);
        client.m5_sync();
        assert!(old_animations::third_person_sword_block() && old_animations::armor_red());
        assert!(client.hud.hide_lost_health_hearts);

        let rod = mc_protocol::ItemStack {
            item_id: interact::FISHING_ROD_ITEM_ID,
            ..diamond_sword()
        };
        // The held slot's object and the first-person latch on it.
        client
            .player_container
            .set_slot_contents(36, Some(rod.clone()));
        let rod_object = client.held_item_identity().expect("a held rod");
        client.first_person_latch.latch(Some((rod_object, &rod)));
        let latched = client
            .m5_first_person(Some(rod_object), None)
            .expect("mod on");
        assert!(latched.rotate_around && latched.blockhit && latched.item_positions);
        // Mid equip transition (the outgoing model is still latched): keep its bit.
        let outgoing = OldAnimations {
            rotate_around: false,
            ..latched
        };
        let kept = client
            .m5_first_person(Some(rod_object + 1), Some(outgoing))
            .expect("mod on");
        assert!(!kept.rotate_around);
        client
            .player_container
            .set_slot_contents(36, Some(diamond_sword()));
        let sword_object = client.held_item_identity().expect("a held sword");
        client
            .first_person_latch
            .latch(Some((sword_object, &diamond_sword())));
        assert!(
            !client
                .m5_first_person(Some(sword_object), Some(latched))
                .expect("on")
                .rotate_around
        );

        client.mods.settings.set_enabled(MOD_ID, false);
        client.m5_sync();
        assert_eq!(
            client.m5_first_person(Some(sword_object), Some(latched)),
            None
        );
        assert!(!old_animations::third_person_sword_block() && !old_animations::armor_red());
        assert!(!client.hud.hide_lost_health_hearts);
    }
}
