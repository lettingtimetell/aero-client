//! S21e: every key and mouse button, from a raw device event to the packet it produces.
//!
//! # Why this file exists
//!
//! A human play-tested this client against a live 1.8 server and reported: *"no game input is
//! working except for WASD and mouse movement"*. That was exactly right. `driver.rs` read
//! [`mc_render::InputState`] once per tick to build a `MoveInput` and did nothing else with it —
//! no key ever reached a screen, no mouse button ever reached a button, and the GUI cursor was
//! hard-coded to the centre of the screen. Fourteen bindings did nothing at all.
//!
//! These tests drive the **shipping** entry points — `Client::handle_game_input` and
//! `Client::handle_screen_input` — with real `InputState` / `KeyBindings` values, against the
//! same loopback double `smoke.rs` uses, and assert on the packets the peer receives. They are
//! deliberately packet-level: "the screen changed" would pass for a client that changed the
//! screen and told the server nothing.
//!
//! java: `Minecraft.runTick`'s two event loops (`:1838` and `:1904`), `clickMouse` (`:1522`),
//! `rightClickMouse` (`:1570`) and `sendClickBlockToController` (`:1496`).

mod common;

use std::time::Duration;

use common::{accept_and_join, config, gpu, pump, pump_until, CHUNK, DIG_TARGET, FLOOR_Y, SPAWN};
use mc_client::app::Client;
use mc_math::BlockPos;
use mc_physics::state::MoveInput;
use mc_protocol::packet::Packet;
use mc_protocol::play::{c2s, s2c};
use mc_render::input::Input;
use mc_render::{Camera, InputState, KeyBindings, PhaseList};
use mc_ui::screens::multiplayer::ServerEntry;
use mc_ui::screens::Action;
use tokio::net::TcpListener;
use winit::event::MouseButton;
use winit::keyboard::KeyCode;

const WIDTH: u32 = 854;
const HEIGHT: u32 = 480;

/// Everything a test needs to drive the client for one input.
struct Rig {
    client: Client,
    gpu: mc_render::Gpu,
    phases: PhaseList,
    camera: Camera,
    /// The loopback peer, still running on the client's own runtime.
    peer: tokio::task::JoinHandle<common::RefPeer>,
    /// Where the player is looking, re-applied on every tick. See [`Rig::look`].
    look: MoveInput,
}

impl Rig {
    fn pump(&mut self) {
        let look = self.look;
        self.pump_with(&look);
    }

    fn pump_with(&mut self, input: &MoveInput) {
        pump(
            &mut self.client,
            &mut self.gpu,
            &mut self.phases,
            &mut self.camera,
            input,
        );
    }

    /// Point the crosshair, and keep it pointed: `Predictor::tick` takes its rotation from the
    /// `MoveInput` it is handed, so a test that sets `state.pitch` once and then ticks with
    /// `MoveInput::default()` is looking at the horizon again one tick later.
    fn look(&mut self, yaw: f32, pitch: f32) {
        self.look = MoveInput {
            yaw,
            pitch,
            ..MoveInput::default()
        };
        if let Some(predictor) = self
            .client
            .session_mut()
            .and_then(mc_client::Session::predictor_mut)
        {
            predictor.state.yaw = yaw;
            predictor.state.pitch = pitch;
        }
        self.pump();
        // `common::pump` drives the public headless tick rather than `ShellApp::tick`, so it does
        // not run the shell-only interaction sample. Reproduce that sample through the same
        // public shipping picker. This controlled world has no pickable entities, so the empty
        // candidate list is exact rather than a shortcut.
        let hit = self
            .client
            .session()
            .and_then(|session| {
                session.predictor().map(|predictor| {
                    mc_client::app::interact::pick(
                        session.world(),
                        predictor.state.eye_pos(),
                        mc_client::app::interact::look_vector(yaw, pitch),
                        4.5,
                        false,
                        std::iter::empty::<mc_client::app::interact::Pickable>(),
                    )
                })
            })
            .unwrap_or(mc_client::app::interact::HitResult::Miss);
        self.client.hit = hit;
    }

    /// Deliver one tick of rebindable-action input, exactly as `ShellApp::tick` does.
    ///
    /// The leading empty call pays off any `KeyBinding.unPressAllKeys()` debt a screen opening
    /// left behind — java: `displayGuiScreen` → `setIngameNotInFocus`.
    fn tap(&mut self, input: Input) {
        let state = InputState::new();
        let mut keys = KeyBindings::vanilla();
        self.client.handle_game_input(&state, &mut keys);
        keys.set_state(input, true);
        self.client.handle_game_input(&state, &mut keys);
    }

    /// Hold a binding down for `ticks` ticks, pumping between each.
    fn hold(&mut self, input: Input, ticks: usize) {
        let mut state = InputState::new();
        let mut keys = KeyBindings::vanilla();
        self.client.handle_game_input(&state, &mut keys);
        match input {
            Input::Key(code) => {
                state.on_key(code, true, false);
            }
            Input::Mouse(button) => {
                state.on_mouse_button(mouse_of(button), true);
            }
            Input::RawLwjgl(code) => panic!("cannot synthesize raw LWJGL input {code}"),
            Input::Unbound => {}
        }
        keys.set_state(input, true);
        for _ in 0..ticks {
            self.client.handle_game_input(&state, &mut keys);
            self.pump();
        }
        // java: `sendClickBlockToController(false)` when the button comes up.
        keys.set_state(input, false);
    }

    /// Press and release a raw key, the way [`Client::handle_raw_key`] delivers it — the same
    /// path `ShellApp::on_key_event` drives from a real `winit::event::KeyEvent`, which this
    /// test cannot construct itself (see that method's doc comment for why).
    ///
    /// No `typed` character is supplied, so `handle_screen_input` falls back to the US-QWERTY
    /// table for it, exactly as it would for a real event winit gave no text for.
    fn key(&mut self, code: KeyCode) {
        self.client.handle_raw_key(code, true, false, None);
        self.client
            .handle_screen_input(&mut InputState::new(), WIDTH, HEIGHT);
        self.client.handle_raw_key(code, false, false, None);
        self.client
            .handle_screen_input(&mut InputState::new(), WIDTH, HEIGHT);
    }

    /// Click at a point in **GUI** pixels.
    fn click(&mut self, gx: i32, gy: i32) {
        let resolution = self.client.resolution(WIDTH, HEIGHT);
        let px = f64::from(gx) * f64::from(WIDTH) / f64::from(resolution.scaled_width());
        let py = f64::from(gy) * f64::from(HEIGHT) / f64::from(resolution.scaled_height());
        let mut state = InputState::new();
        state.on_cursor_moved(px, py);
        state.on_mouse_button(MouseButton::Left, true);
        self.client.handle_screen_input(&mut state, WIDTH, HEIGHT);
        state.on_mouse_button(MouseButton::Left, false);
        self.client.handle_screen_input(&mut state, WIDTH, HEIGHT);
    }

    /// Turn the wheel by `delta` notches during one render frame, then run the next tick's input
    /// section, which is where vanilla applies it: LWJGL queues the event until `runTick`'s
    /// mouse loop (`Minecraft.java:1838-1881`), after `updateController`. The client queues the
    /// notches the same way, so the slot moves on that tick.
    fn wheel(&mut self, delta: f32) {
        let mut state = InputState::new();
        state.on_scroll(winit::event::MouseScrollDelta::LineDelta(0.0, delta));
        self.client.handle_screen_input(&mut state, WIDTH, HEIGHT);
        let mut keys = KeyBindings::vanilla();
        self.client.handle_game_input(&InputState::new(), &mut keys);
    }

    /// Close the socket and hand back everything the peer saw.
    fn finish(mut self) -> Vec<(i32, Vec<u8>)> {
        self.client.disconnect();
        for _ in 0..10 {
            self.pump();
        }
        let handle = self.client.runtime().handle().clone();
        let peer = handle.block_on(async {
            tokio::time::timeout(Duration::from_secs(10), self.peer)
                .await
                .expect("the peer must finish")
                .expect("the peer task must not panic")
        });
        peer.seen
    }
}

fn mouse_of(button: u16) -> MouseButton {
    match button {
        0 => MouseButton::Left,
        1 => MouseButton::Right,
        _ => MouseButton::Middle,
    }
}

/// Every serverbound packet of one id the peer saw, decoded.
fn all_of<P: Packet>(seen: &[(i32, Vec<u8>)]) -> Vec<P> {
    seen.iter()
        .filter(|(id, _)| *id == P::ID)
        .map(|(_, body)| common::decode::<P>(body))
        .collect()
}

/// Join the loopback double and stop on the tick the world is ready.
fn rig() -> Option<Rig> {
    let gpu = gpu(WIDTH, HEIGHT)?;
    let dir = Box::leak(Box::new(tempfile::tempdir().expect("tempdir")));
    let config = config(dir.path(), "Dev")?;
    let mut client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
    let mut phases = PhaseList::new();
    client.install(&gpu, &mut phases, WIDTH, HEIGHT);
    let mut camera = mc_client::app::driver::menu_camera();

    let listener = client
        .runtime()
        .handle()
        .block_on(TcpListener::bind("127.0.0.1:0"))
        .expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let peer = client.runtime().handle().spawn(serve(listener));

    client.handle_action(Action::Connect(Box::new(ServerEntry::new(
        "Loopback",
        format!("127.0.0.1:{port}"),
    ))));

    let mut rig = Rig {
        client,
        gpu,
        phases,
        camera,
        peer,
        look: MoveInput::default(),
    };
    pump_until(
        &mut rig.client,
        &mut rig.gpu,
        &mut rig.phases,
        &mut rig.camera,
        &MoveInput::default(),
        Duration::from_secs(30),
        "the client to reach the world",
        |client| {
            client.ui.name() == "in_game"
                && client
                    .session()
                    .is_some_and(|s| s.world().is_chunk_loaded(CHUNK))
        },
    );
    let _ = &mut camera;
    Some(rig)
}

/// The peer: join, fill window 0 the way a hub server does, then read until the client leaves.
async fn serve(listener: TcpListener) -> common::RefPeer {
    let mut peer = accept_and_join(listener, 42).await;

    // java: `S30PacketWindowItems` for window 0 is how every server fills the player's own
    // inventory. 45 slots: 1 craft result, 4 grid, 4 armour, 27 main, 9 hotbar.
    let mut items: Vec<Option<mc_protocol::ItemStack>> = vec![None; 45];
    items[36] = Some(mc_protocol::ItemStack {
        item_id: 345, // minecraft:compass — the item the play-test report was about
        count: 1,
        damage: 0,
        nbt: None,
    });
    items[38] = Some(mc_protocol::ItemStack {
        item_id: 276, // minecraft:diamond_sword
        count: 1,
        damage: 0,
        nbt: None,
    });
    peer.send(&s2c::WindowItems {
        window_id: 0,
        items,
    })
    .await;

    peer.record_until_closed().await;
    peer
}

/// **The report's first line, end to end.** Fourteen bindings, each asserted on the packet it
/// puts on the wire or the screen it opens — not on an internal flag.
#[test]
fn every_vanilla_binding_reaches_the_action_it_names() {
    let Some(mut rig) = rig() else { return };

    // Let the inventory arrive.
    pump_until(
        &mut rig.client,
        &mut rig.gpu,
        &mut rig.phases,
        &mut rig.camera,
        &MoveInput::default(),
        Duration::from_secs(10),
        "window 0's items",
        |client| client.player_hud.hotbar[0].is_some(),
    );

    // ---- hotbar 1..9 -----------------------------------------------------------------
    // java: `Minecraft.runTick:2075` — `keyBindsHotbar[l].isPressed()` sets `currentItem = l`.
    for (key, slot) in [
        (KeyCode::Digit3, 2),
        (KeyCode::Digit9, 8),
        (KeyCode::Digit1, 0),
    ] {
        rig.tap(Input::Key(key));
        assert_eq!(
            rig.client.player_hud.selected_slot, slot,
            "{key:?} did not select slot {slot}"
        );
        // Selection itself is local-only; the next 20 Hz controller update lazily emits C09.
        rig.pump();
    }

    // ---- the wheel -------------------------------------------------------------------
    // java: scrolling **up** moves *down* the hotbar index, and both ends wrap.
    rig.wheel(1.0);
    assert_eq!(rig.client.player_hud.selected_slot, 8, "wrapped 0 -> 8");
    rig.pump();
    rig.wheel(-1.0);
    assert_eq!(rig.client.player_hud.selected_slot, 0);
    rig.pump();

    // ---- E: the inventory ------------------------------------------------------------
    rig.tap(Input::Key(KeyCode::KeyE));
    assert_eq!(rig.client.ui.name(), "container");
    // java: `GuiContainer.keyTyped` closes on the inventory key as well as Escape.
    rig.key(KeyCode::KeyE);
    assert_eq!(rig.client.ui.name(), "in_game");

    // ---- Q: drop ---------------------------------------------------------------------
    rig.tap(Input::Key(KeyCode::KeyQ));

    // ---- T and / : chat --------------------------------------------------------------
    rig.tap(Input::Key(KeyCode::KeyT));
    assert_eq!(rig.client.ui.name(), "chat");
    rig.key(KeyCode::Escape);
    assert_eq!(rig.client.ui.name(), "in_game");
    rig.tap(Input::Key(KeyCode::Slash));
    assert_eq!(rig.client.ui.name(), "chat");
    // Typing goes through the *raw* keyboard, which is the half `KeyBindings` cannot serve.
    for code in [KeyCode::KeyH, KeyCode::KeyI] {
        rig.key(code);
    }
    rig.key(KeyCode::Enter);
    assert_eq!(rig.client.ui.name(), "in_game", "Enter sends and closes");

    // ---- Escape: the pause menu ------------------------------------------------------
    rig.key(KeyCode::Escape);
    assert_eq!(rig.client.ui.name(), "pause");
    rig.key(KeyCode::Escape);
    assert_eq!(rig.client.ui.name(), "in_game");

    // ---- F3: the debug overlay -------------------------------------------------------
    assert!(!rig.client.debug.visible);
    rig.key(KeyCode::F3);
    assert!(rig.client.debug.visible, "F3 did not open the F3 overlay");
    rig.key(KeyCode::F3);
    assert!(!rig.client.debug.visible);

    // ---- F5: perspective -------------------------------------------------------------
    // java: `++thirdPersonView; if (thirdPersonView > 2) thirdPersonView = 0;`
    assert_eq!(rig.client.third_person_view, 0);
    for expected in [1, 2, 0] {
        rig.tap(Input::Key(KeyCode::F5));
        assert_eq!(rig.client.third_person_view, expected);
    }

    // ---- F11: fullscreen -------------------------------------------------------------
    // `toggle_fullscreen` now has a real effect on the OS window too: the shell polls
    // `ShellApp::wants_fullscreen` every frame and reconciles it (`mc_render::shell`'s
    // `apply_fullscreen`). This rig never opens a real window, though, so only the setting
    // itself is observable from here — the window-level half is unverified by this test; see
    // `mc_render::shell`'s `fullscreen_is_only_reconciled_when_the_request_actually_changes`
    // for the part of it that is provable without a real window.
    let before = rig.client.settings.fullscreen;
    rig.key(KeyCode::F11);
    assert_eq!(rig.client.settings.fullscreen, !before);
    rig.key(KeyCode::F11);
    assert_eq!(rig.client.settings.fullscreen, before);

    // ---- what actually went out ------------------------------------------------------
    let seen = rig.finish();

    let held: Vec<c2s::HeldItemChange> = all_of(&seen);
    let slots: Vec<i16> = held.iter().map(|p| p.slot_id).collect();
    assert!(
        slots.contains(&2) && slots.contains(&8) && slots.contains(&0),
        "the number keys must reach the server as C09PacketHeldItemChange: {slots:?}"
    );
    // java: `syncCurrentPlayItem` suppresses a resend for the slot the server already knows.
    for pair in slots.windows(2) {
        assert_ne!(pair[0], pair[1], "a redundant held-item change went out");
    }

    let digging: Vec<c2s::PlayerDigging> = all_of(&seen);
    assert!(
        digging
            .iter()
            .any(|p| p.status == c2s::PlayerDiggingStatus::DropItem),
        "Q must send C07PacketPlayerDigging(DROP_ITEM): {:?}",
        digging.iter().map(|p| p.status).collect::<Vec<_>>()
    );

    let chat: Vec<c2s::ChatMessage> = all_of(&seen);
    assert_eq!(
        chat.iter().map(|c| c.message.as_str()).collect::<Vec<_>>(),
        ["/hi"],
        "the command key must pre-fill the slash and the typed characters must reach the wire"
    );

    let status: Vec<c2s::ClientStatus> = all_of(&seen);
    assert!(
        status
            .iter()
            .any(|s| s.action == c2s::ClientStatusAction::OpenInventoryAchievement),
        "java: opening the inventory sends C16PacketClientStatus(OPEN_INVENTORY_ACHIEVEMENT)"
    );
}

/// **Left-click.** Holding attack on a block must produce exactly one START and one STOP, with
/// the swing animation alongside — the whole `clickMouse` → `sendClickBlockToController` →
/// `onPlayerDamageBlock` chain, driven from a mouse button rather than by calling `OutboundState`
/// directly (which is what `smoke.rs` did, and which is why this defect survived it).
#[test]
fn holding_the_attack_button_digs_the_block_under_the_crosshair() {
    let Some(mut rig) = rig() else { return };

    // Look straight down at the floor block the peer sent, from standing height.
    rig.look(0.0, 90.0);

    let hit = rig
        .client
        .hit
        .block()
        .expect("the crosshair must be on the floor");
    assert_eq!(
        hit.pos,
        BlockPos::new(SPAWN.0 as i32, FLOOR_Y as i32, SPAWN.2 as i32),
        "looking straight down must pick the block underfoot"
    );
    assert_eq!(hit.face, mc_math::Facing::Up);
    assert_eq!(hit.pos, DIG_TARGET);

    // Stone is 1.5 hardness; bare-handed on the ground that is 151 ticks (see
    // `app::interact`'s `survival_digging_accumulates_to_one_stop`). 220 clears it with room for
    // the five-tick `blockHitDelay` the first `clickBlock` does not spend.
    // 1 START on the press edge, then one damage application per tick; the STOP lands on the
    // 151st. Held past that, `blockHitDelay` waits five ticks and vanilla starts the *next* dig
    // - correct, and why this stops at 155 rather than running on and counting two of each.
    rig.hold(Input::MOUSE_LEFT, 155);

    let seen = rig.finish();
    let digging: Vec<c2s::PlayerDigging> = all_of(&seen);
    let starts: Vec<_> = digging
        .iter()
        .filter(|p| p.status == c2s::PlayerDiggingStatus::StartDestroyBlock)
        .collect();
    let stops: Vec<_> = digging
        .iter()
        .filter(|p| p.status == c2s::PlayerDiggingStatus::StopDestroyBlock)
        .collect();
    assert_eq!(
        starts.len(),
        1,
        "java: one START per dig, not one per tick — {:?}",
        digging.iter().map(|p| p.status).collect::<Vec<_>>()
    );
    assert_eq!(stops.len(), 1, "and exactly one STOP when it breaks");
    assert_eq!(starts[0].position, DIG_TARGET);
    assert_eq!(stops[0].position, DIG_TARGET);
    assert_eq!(starts[0].face, mc_math::Facing::Up);

    let swings: Vec<c2s::Animation> = all_of(&seen);
    assert!(
        !swings.is_empty(),
        "java: `thePlayer.swingItem()` on every successful onPlayerDamageBlock"
    );
}

/// A held attack whose next real pick is `Miss` must abort the old target exactly once.
///
/// java: `sendClickBlockToController(true)` falls into its non-block arm after
/// `EntityRenderer.getMouseOver` replaces `objectMouseOver` with a miss. That arm calls
/// `PlayerControllerMP.resetBlockRemoving`; keeping the old controller target would allow a
/// later STOP for a block the crosshair left.
#[test]
fn held_attack_recomputed_to_miss_aborts_the_old_target_once() {
    let Some(mut rig) = rig() else { return };

    // `rig` stops when the chunk is resident; on a fast loopback that can be the provisional
    // empty column created at join just before the full S21 payload is applied. Wait for the
    // actual dig target so this test's START edge cannot race the packet reader.
    pump_until(
        &mut rig.client,
        &mut rig.gpu,
        &mut rig.phases,
        &mut rig.camera,
        &MoveInput::default(),
        Duration::from_secs(10),
        "the floor block",
        |client| {
            client
                .session()
                .is_some_and(|session| !session.world().block_at(DIG_TARGET).props().is_air())
        },
    );
    rig.look(0.0, 90.0);
    assert_eq!(
        rig.client.hit.block().map(|hit| hit.pos),
        Some(DIG_TARGET),
        "the control must begin on the floor block"
    );

    let mut state = InputState::new();
    let mut keys = KeyBindings::vanilla();
    rig.client.handle_game_input(&state, &mut keys);
    state.on_mouse_button(MouseButton::Left, true);
    keys.set_state(Input::MOUSE_LEFT, true);
    rig.client.handle_game_input(&state, &mut keys);
    rig.pump();

    // Keep the button genuinely held while the shipping pick path moves to open sky.
    rig.look(0.0, -90.0);
    assert_eq!(rig.client.hit, mc_client::app::interact::HitResult::Miss);
    rig.client.handle_game_input(&state, &mut keys);
    assert!(
        rig.client.controller.hitting_block().is_none(),
        "the first held-Miss input must clear the active target before idle/release"
    );
    rig.pump();

    // An idle held-Miss tick and the later release are both inert: neither may duplicate the
    // ABORT or resurrect the old target as a STOP.
    rig.client.handle_game_input(&state, &mut keys);
    rig.pump();
    state.on_mouse_button(MouseButton::Left, false);
    keys.set_state(Input::MOUSE_LEFT, false);
    rig.client.handle_game_input(&state, &mut keys);
    rig.pump();

    let seen = rig.finish();
    let digging: Vec<c2s::PlayerDigging> = all_of(&seen);
    let starts: Vec<_> = digging
        .iter()
        .filter(|packet| packet.status == c2s::PlayerDiggingStatus::StartDestroyBlock)
        .collect();
    let aborts: Vec<_> = digging
        .iter()
        .filter(|packet| packet.status == c2s::PlayerDiggingStatus::AbortDestroyBlock)
        .collect();
    let stops: Vec<_> = digging
        .iter()
        .filter(|packet| packet.status == c2s::PlayerDiggingStatus::StopDestroyBlock)
        .collect();
    assert_eq!(starts.len(), 1, "one press edge starts exactly one dig");
    assert_eq!(starts[0].position, DIG_TARGET);
    assert_eq!(
        aborts.len(),
        1,
        "the first held-Miss tick aborts exactly once"
    );
    assert_eq!(aborts[0].position, DIG_TARGET);
    assert!(
        stops.is_empty(),
        "the abandoned target must never later STOP"
    );
}

/// **Right-click.** With a block under the crosshair the use button sends
/// `C08PacketPlayerBlockPlacement` against that face; with nothing under it, the no-target form
/// — which is what makes a hub compass open its menu.
#[test]
fn the_use_button_places_against_a_face_and_uses_the_item_at_thin_air() {
    let Some(mut rig) = rig() else { return };

    pump_until(
        &mut rig.client,
        &mut rig.gpu,
        &mut rig.phases,
        &mut rig.camera,
        &MoveInput::default(),
        Duration::from_secs(10),
        "window 0's items",
        |client| client.player_hud.hotbar[0].is_some(),
    );

    // Down at the floor.
    rig.look(0.0, 90.0);
    rig.tap(Input::MOUSE_RIGHT);

    // Straight up at open sky: nothing within reach.
    rig.look(0.0, -90.0);
    assert_eq!(rig.client.hit, mc_client::app::interact::HitResult::Miss);
    // java: `rightClickDelayTimer = 4` — the second click has to wait out the first.
    for _ in 0..5 {
        rig.pump();
    }
    rig.tap(Input::MOUSE_RIGHT);

    let seen = rig.finish();
    let placements: Vec<c2s::PlayerBlockPlacement> = all_of(&seen);
    assert!(
        placements.len() >= 2,
        "expected a targeted placement and an untargeted use, got {}",
        placements.len()
    );
    let targeted = placements
        .iter()
        .find(|p| p.position == DIG_TARGET)
        .expect("java: onPlayerRightClick sends the hit block and face");
    assert_eq!(targeted.placed_block_direction, 1, "EnumFacing.UP");
    // java: `C08PacketPlayerBlockPlacement.writePacketData` truncates each axis to sixteenths.
    assert_eq!(targeted.cursor_y, 16, "the hit was on the top face");

    let untargeted = placements
        .iter()
        .find(|p| p.position == BlockPos::new(-1, -1, -1))
        .expect("java: sendUseItem's no-target constructor is (-1,-1,-1) / direction 255");
    assert_eq!(untargeted.placed_block_direction, 255);
    assert_eq!(
        untargeted.stack.as_ref().map(|s| s.item_id),
        Some(345),
        "the held compass must be attached, or a hub plugin cannot tell which item was used"
    );
}

/// **The report's third line.** Window 0's `S30PacketWindowItems` reaches the HUD's hotbar, and
/// the inventory screen shows the same stacks.
#[test]
fn window_zero_fills_both_the_hotbar_and_the_inventory_screen() {
    let Some(mut rig) = rig() else { return };
    pump_until(
        &mut rig.client,
        &mut rig.gpu,
        &mut rig.phases,
        &mut rig.camera,
        &MoveInput::default(),
        Duration::from_secs(10),
        "window 0's items",
        |client| client.player_hud.hotbar[0].is_some(),
    );

    assert_eq!(
        rig.client.player_hud.hotbar[0].as_ref().map(|s| s.item_id),
        Some(345),
        "slot 36 of window 0 is mainInventory[0], i.e. hotbar slot 0"
    );
    assert_eq!(
        rig.client.player_hud.hotbar[2].as_ref().map(|s| s.item_id),
        Some(276)
    );
    assert!(
        rig.client.player_hud.hotbar[1].is_none(),
        "slot 37 was empty and must stay empty"
    );
    // Reading `items[0..9]` instead of `items[36..45]` would put the crafting grid here.
    for slot in 0..9 {
        assert_eq!(
            rig.client.player_container.slot_stack(36 + slot),
            rig.client.player_hud.hotbar[slot].as_ref()
        );
    }

    // The held-item name popup is driven from the same feed. Java's
    // `GuiIngame.renderSelectedItem` uses `ItemStack.getDisplayName()`, which is the plain
    // localized item name here; tooltip formatting (including a trailing reset) is not part of
    // this popup's string contract.
    assert_eq!(rig.client.hud.held_item_name, "Compass");
    assert!(rig.client.hud.highlight_ticks > 0);

    // And the icons rasterise, which needs the atlas uploaded under the key S17 picks.
    rig.pump();
    let pixels = rig.gpu.read_pixels().expect("readback");
    let resolution = rig.client.resolution(WIDTH, HEIGHT);
    let (sw, sh) = (resolution.scaled_width(), resolution.scaled_height());
    let scale_x = f64::from(WIDTH) / f64::from(sw);
    let scale_y = f64::from(HEIGHT) / f64::from(sh);
    let mut coloured = 0;
    // java: GuiIngame.java:385 — hotbar slot 0's 16x16 icon box.
    for dy in 0..16 {
        for dx in 0..16 {
            let gx = sw / 2 - 90 + 2 + dx;
            let gy = sh - 16 - 3 + dy;
            let px = ((f64::from(gx) + 0.5) * scale_x) as usize;
            let py = ((f64::from(gy) + 0.5) * scale_y) as usize;
            let at = (py * WIDTH as usize + px) * 4;
            if let Some(rgba) = pixels.get(at..at + 4) {
                if rgba[3] > 0 && (rgba[0] != rgba[1] || rgba[1] != rgba[2]) {
                    coloured += 1;
                }
            }
        }
    }
    assert!(
        coloured > 20,
        "the compass icon did not rasterise into hotbar slot 0 ({coloured} coloured pixels)"
    );

    rig.finish();
}

/// **The report's second line.** A health packet of zero raises the death screen on the next
/// tick, the buttons ignore the first second, and the Respawn button sends
/// `C16PacketClientStatus(PERFORM_RESPAWN)`.
///
/// This is the half of the check that runs with no server, so the behaviour stays in the green
/// gate; a live-server run checks the rest.
#[test]
fn zero_health_raises_the_death_screen_and_respawn_asks_the_server_to_revive() {
    let Some(mut rig) = rig() else { return };

    rig.client
        .session_mut()
        .expect("session")
        .state_mut()
        .local
        .health = 0.0;
    assert!(rig.client.player_is_dead());
    // java: `runTick:1773` raises it on the tick, not in the packet handler.
    rig.pump();
    assert_eq!(rig.client.ui.name(), "game_over");

    // java: `GuiGameOver.initGui` disables every button; `updateScreen` re-enables at tick 20.
    let resolution = rig.client.resolution(WIDTH, HEIGHT);
    let respawn = (
        resolution.scaled_width() / 2,
        resolution.scaled_height() / 4 + 72 + 10,
    );
    rig.click(respawn.0, respawn.1);
    assert_eq!(
        rig.client.ui.name(),
        "game_over",
        "the click that killed you must not also dismiss the screen"
    );
    for _ in 0..25 {
        rig.pump();
    }
    rig.click(respawn.0, respawn.1);
    // GuiGameOver sends C16 then displayGuiScreen(null). Minecraft substitutes another
    // GuiGameOver while health is still zero; this peer has not sent S07 Respawn yet.
    assert_eq!(rig.client.ui.name(), "game_over");

    for _ in 0..5 {
        rig.pump();
    }
    let seen = rig.finish();
    let status: Vec<c2s::ClientStatus> = all_of(&seen);
    assert!(
        status
            .iter()
            .any(|s| s.action == c2s::ClientStatusAction::PerformRespawn),
        "without C16PacketClientStatus(PERFORM_RESPAWN) the server keeps the player dead \
         forever, which is worse than showing no screen at all: {:?}",
        status.iter().map(|s| s.action).collect::<Vec<_>>()
    );
}

/// Opening a screen releases every held key, so walking into the inventory does not walk you
/// out of the world behind it. java: `displayGuiScreen` → `setIngameNotInFocus` →
/// `KeyBinding.unPressAllKeys()`.
#[test]
fn opening_a_screen_releases_the_movement_keys() {
    let Some(mut rig) = rig() else { return };

    let mut state = InputState::new();
    state.on_key(KeyCode::KeyW, true, false);
    let mut keys = KeyBindings::vanilla();
    keys.set_state(Input::Key(KeyCode::KeyW), true);
    assert!(keys.is_down(mc_render::input::keys::FORWARD));

    // Open the inventory the way the shell would, then run one more tick of game input.
    keys.set_state(Input::Key(KeyCode::KeyE), true);
    rig.client.handle_game_input(&state, &mut keys);
    assert_eq!(rig.client.ui.name(), "container");
    rig.client.handle_game_input(&state, &mut keys);
    assert!(
        !keys.is_down(mc_render::input::keys::FORWARD),
        "the forward key must be released when a screen opens"
    );

    rig.finish();
}

/// **The menu was unclickable too.** `build_ui` handed every screen `(width / 2, height / 2)` as
/// the mouse position and nothing ever called `mouse_clicked`, so the main menu's buttons could
/// neither highlight nor be pressed. Needs no server: the failure was entirely local.
#[test]
fn clicking_the_main_menu_actually_presses_its_buttons() {
    let Some(gpu) = gpu(WIDTH, HEIGHT) else {
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    // vanilla's buttons at vanilla's coordinates: the vanilla title screen (the
    // Aero one's own clicks are tested in `app/mods/aero_title.rs`).
    common::vanilla_title_screen(dir.path());
    let Some(config) = config(dir.path(), "Dev") else {
        eprintln!("skip: no vanilla jar in this checkout");
        return;
    };
    let mut client = Client::new(config, &mut |_, _| {}).expect("assets");
    let mut phases = PhaseList::new();
    client.install(&gpu, &mut phases, WIDTH, HEIGHT);
    let mut camera = mc_client::app::driver::menu_camera();
    assert_eq!(client.ui.name(), "main_menu");

    // The screen has to be laid out before it can be hit-tested; `build_ui` does that.
    client.frame(&gpu, &mut camera, 0.0);

    // java: `GuiMainMenu.initGui` — `j = height / 4 + 48`, Multiplayer at `j + 24`, 200x20.
    let resolution = client.resolution(WIDTH, HEIGHT);
    let gx = resolution.scaled_width() / 2;
    let gy = resolution.scaled_height() / 4 + 48 + 24 + 10;
    let px = f64::from(gx) * f64::from(WIDTH) / f64::from(resolution.scaled_width());
    let py = f64::from(gy) * f64::from(HEIGHT) / f64::from(resolution.scaled_height());

    let mut state = InputState::new();
    state.on_cursor_moved(px, py);
    state.on_mouse_button(MouseButton::Left, true);
    client.handle_screen_input(&mut state, WIDTH, HEIGHT);
    assert_eq!(
        client.ui.name(),
        "multiplayer",
        "clicking Multiplayer must open the server list"
    );

    // A click well away from every button must do nothing, or the test above proves only that
    // *some* click changes the screen.
    state.on_mouse_button(MouseButton::Left, false);
    client.handle_screen_input(&mut state, WIDTH, HEIGHT);
    state.on_cursor_moved(2.0, 2.0);
    state.on_mouse_button(MouseButton::Left, true);
    client.handle_screen_input(&mut state, WIDTH, HEIGHT);
    assert_eq!(client.ui.name(), "multiplayer");
}

/// F5 must move the camera, not just a counter. java: `EntityRenderer.orientCamera` (`:1119`)
/// pulls the eye back four blocks along the look vector, and the front view (`thirdPersonView ==
/// 2`) additionally turns it around.
///
/// **Named gap, asserted here so it cannot be forgotten:** the player's own body is *not* drawn
/// in either third-person view, because S14 tracks only remote players in the ECS. The camera is
/// right; there is nothing to look at.
#[test]
fn the_perspective_key_moves_the_camera_off_the_players_eyes() {
    let Some(mut rig) = rig() else { return };
    rig.look(0.0, 0.0);

    let eye = rig.camera.pos;
    let yaw = rig.camera.yaw;

    rig.tap(Input::Key(KeyCode::F5));
    rig.pump();
    assert_eq!(rig.client.third_person_view, 1);
    let behind = rig.camera.pos;
    // Yaw 0 faces +Z, so the third-person camera must sit at *lower* Z than the eyes.
    assert!(
        behind.z < eye.z - 0.5,
        "the camera did not pull back: {eye:?} -> {behind:?}"
    );
    assert!(
        (behind - eye).length() <= 4.0 + 1e-6,
        "java: thirdPersonDistance is 4.0 and the wall raytrace only ever shortens it"
    );
    assert_eq!(rig.camera.yaw, yaw, "the back view keeps the player's yaw");

    rig.tap(Input::Key(KeyCode::F5));
    rig.pump();
    assert_eq!(rig.client.third_person_view, 2);
    let front = rig.camera.pos;
    assert!(
        front.z > eye.z + 0.5,
        "the front view must sit in front of the player: {eye:?} -> {front:?}"
    );
    // java: `f -= 180.0F` for view 2.
    assert!(
        (rig.camera.yaw - (yaw + 180.0)).abs() < 1e-3,
        "{}",
        rig.camera.yaw
    );

    rig.tap(Input::Key(KeyCode::F5));
    rig.pump();
    assert_eq!(rig.client.third_person_view, 0);
    assert!(
        (rig.camera.pos - eye).length() < 1e-6,
        "first person must put the camera back on the eyes"
    );

    rig.finish();
}

/// The four bindings that *were* already wired, pinned so the report's table is evidence rather
/// than a claim: jump, sneak and sprint all reach the physics and the wire through
/// `app::input::move_input` → `Predictor` → `OutboundState::walking_update`, which is why the
/// play-test found WASD working when nothing else was.
///
/// java: `MovementInputFromOptions.updatePlayerMoveState` plus `EntityPlayerSP.onLivingUpdate`'s
/// two `C0BPacketEntityAction` edges.
#[test]
fn jump_sneak_and_sprint_were_already_wired_and_still_are() {
    let Some(mut rig) = rig() else { return };

    let ground = rig
        .client
        .session()
        .expect("session")
        .predictor()
        .expect("predictor")
        .state
        .pos
        .y;

    // Jump: the player must leave the floor.
    let jump = MoveInput {
        jump: true,
        ..MoveInput::default()
    };
    let mut peak = ground;
    for _ in 0..8 {
        rig.pump_with(&jump);
        peak = peak.max(
            rig.client
                .session()
                .expect("session")
                .predictor()
                .expect("predictor")
                .state
                .pos
                .y,
        );
    }
    assert!(peak > ground + 0.3, "the jump key did not lift the player");

    // Sneak, then sprint: each must produce exactly one edge packet, not one per tick.
    let sneak = MoveInput {
        sneak: true,
        ..MoveInput::default()
    };
    for _ in 0..10 {
        rig.pump_with(&sneak);
    }
    let sprint = MoveInput {
        forward: true,
        sprint: true,
        ..MoveInput::default()
    };
    for _ in 0..10 {
        rig.pump_with(&sprint);
    }

    let seen = rig.finish();
    let actions: Vec<c2s::EntityAction> = all_of(&seen);
    let kinds: Vec<c2s::EntityActionType> = actions.iter().map(|a| a.action).collect();
    assert_eq!(
        kinds
            .iter()
            .filter(|k| **k == c2s::EntityActionType::StartSneaking)
            .count(),
        1,
        "java: the sneak edge is sent once, not every tick — {kinds:?}"
    );
    assert_eq!(
        kinds
            .iter()
            .filter(|k| **k == c2s::EntityActionType::StopSneaking)
            .count(),
        1
    );
    assert_eq!(
        kinds
            .iter()
            .filter(|k| **k == c2s::EntityActionType::StartSprinting)
            .count(),
        1,
        "{kinds:?}"
    );
}
