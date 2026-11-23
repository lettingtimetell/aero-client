//! The serverbound container protocol, end to end: `C0EPacketClickWindow`,
//! `C0DPacketCloseWindow` and `C07PacketPlayerDigging(RELEASE_USE_ITEM)`.
//!
//! # Why this file exists
//!
//! `mc-ui`'s container state machine was
//! complete — all seven click modes, the three-phase drag protocol, transaction ids — and
//! *entirely unreachable*: `ContainerScreen::take_outgoing()` had no caller in `mc-client`, the
//! `shift` argument into `MouseState` was a hard-coded `false` at both call sites, `c2s::CloseWindow`
//! had no constructor anywhere in the crate, and `OutboundState::release_use_item` was dead code.
//! Every one of those is invisible to a test that asserts on client-side state, because the
//! client-side state was already right — it was the *wire* that was empty. So every assertion
//! here is on bytes the peer received.
//!
//! # What this proves and what it does not
//!
//! Same disclosure as `tests/common/mod.rs` and `tests/input_dispatch.rs`, which this reuses
//! wholesale: the far side is a **loopback double**, not a Minecraft server. Its framing is
//! written independently of `mc_protocol::codec`; its packet *bodies* are encoded with
//! `mc-protocol`'s own `Packet::write`, so this proves the client's dispatch, state machine and
//! outbound path, and does **not** prove that a real vanilla 1.8.9 server accepts these bytes.
//! There is no network and no Minecraft server in this environment.
//!
//! java: `client/gui/inventory/GuiContainer.java`'s `mouseClicked`/`mouseReleased`/`keyTyped`,
//! `client/multiplayer/PlayerControllerMP.java`'s `windowClick` (`:534`) and
//! `onStoppedUsingItem` (`:576`), and `client/entity/EntityPlayerSP.closeScreen` (`:330`).

mod common;

use std::time::Duration;

use common::{accept_and_join, config, gpu, pump, pump_until, CHUNK};
use mc_client::app::driver::Ui;
use mc_client::app::Client;
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

/// The window id the scripted peer opens its chest under. Deliberately **not** `0` and not `1`,
/// so a client that hard-coded either would be caught.
const CHEST_WINDOW: u8 = 7;
/// A 3-row chest: 27 container slots, then the player's own 36.
const CHEST_SLOTS: u8 = 27;
/// `minecraft:stone`.
const STONE: i16 = 1;
/// `minecraft:bow`.
const BOW: i16 = 261;
/// `minecraft:arrow`.
const ARROW: i16 = 262;

/// What the peer does after the join handshake.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Script {
    /// Fill window `0` and then just listen. No container is ever opened by the server.
    Plain,
    /// Fill window `0`, then `S2DPacketOpenWindow` a chest and fill it.
    Chest,
    /// As [`Script::Chest`], but the *server* closes the window as soon as it sees the client's
    /// first `C0EPacketClickWindow`.
    ChestThenServerClose,
}

struct Rig {
    client: Client,
    gpu: mc_render::Gpu,
    phases: PhaseList,
    camera: Camera,
    peer: tokio::task::JoinHandle<common::RefPeer>,
    look: MoveInput,
}

impl Rig {
    fn pump(&mut self) {
        let look = self.look;
        pump(
            &mut self.client,
            &mut self.gpu,
            &mut self.phases,
            &mut self.camera,
            &look,
        );
    }

    fn wait(&mut self, what: &str, done: impl FnMut(&Client) -> bool) {
        let look = self.look;
        pump_until(
            &mut self.client,
            &mut self.gpu,
            &mut self.phases,
            &mut self.camera,
            &look,
            Duration::from_secs(20),
            what,
            done,
        );
    }

    /// Point the crosshair and keep it pointed. See `input_dispatch.rs`'s identical helper.
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
    }

    /// The centre of a slot of the open container, in **GUI** pixels.
    ///
    /// Read off the live screen rather than recomputed here on purpose: a test that duplicated
    /// `guiLeft = (width - xSize) / 2` would agree with a client that got the centring wrong.
    fn slot_point(&self, slot_id: u16) -> (i32, i32) {
        let Ui::Container(screen) = &self.client.ui else {
            panic!("no container screen is open");
        };
        let slot = *screen
            .container
            .slots()
            .iter()
            .find(|s| s.id == slot_id)
            .expect("the container must have that slot");
        (
            screen.gui_left() + slot.x + 8,
            screen.gui_top() + slot.y + 8,
        )
    }

    /// Press and release a mouse button at a point in GUI pixels, optionally with shift held.
    ///
    /// The shift edge goes through [`Client::handle_raw_key`], which is the same path
    /// `ShellApp::on_key_event` drives — it is what populates the `Modifiers` the container
    /// screen now reads.
    fn click_at(&mut self, (gx, gy): (i32, i32), button: MouseButton, shift: bool) {
        if shift {
            self.client
                .handle_raw_key(KeyCode::ShiftLeft, true, false, None);
        }
        let resolution = self.client.resolution(WIDTH, HEIGHT);
        let px = f64::from(gx) * f64::from(WIDTH) / f64::from(resolution.scaled_width());
        let py = f64::from(gy) * f64::from(HEIGHT) / f64::from(resolution.scaled_height());
        let mut state = InputState::new();
        state.on_cursor_moved(px, py);
        state.on_mouse_button(button, true);
        self.client.handle_screen_input(&mut state, WIDTH, HEIGHT);
        state.on_mouse_button(button, false);
        self.client.handle_screen_input(&mut state, WIDTH, HEIGHT);
        if shift {
            self.client
                .handle_raw_key(KeyCode::ShiftLeft, false, false, None);
            self.client
                .handle_screen_input(&mut InputState::new(), WIDTH, HEIGHT);
        }
    }

    /// A raw key press+release, as `input_dispatch.rs` does it.
    fn key(&mut self, code: KeyCode) {
        self.client.handle_raw_key(code, true, false, None);
        self.client
            .handle_screen_input(&mut InputState::new(), WIDTH, HEIGHT);
        self.client.handle_raw_key(code, false, false, None);
        self.client
            .handle_screen_input(&mut InputState::new(), WIDTH, HEIGHT);
    }

    /// One tick of a rebindable action going down and back up.
    fn tap(&mut self, input: Input) {
        let state = InputState::new();
        let mut keys = KeyBindings::vanilla();
        self.client.handle_game_input(&state, &mut keys);
        keys.set_state(input, true);
        self.client.handle_game_input(&state, &mut keys);
    }

    /// Hold a binding for `ticks` ticks and then **let it go**, delivering one more
    /// `handle_game_input` with the button up.
    ///
    /// `input_dispatch.rs`'s `hold` stops one call short of that, which is fine for digging
    /// (whose STOP is produced by the damage accumulator, not by the release) and useless here:
    /// the release edge *is* what this file is about. java: `runTick:2125` — `if
    /// (!keyBindUseItem.isKeyDown()) playerController.onStoppedUsingItem(...)`.
    fn hold_then_release(&mut self, input: Input, ticks: usize) {
        let mut down = InputState::new();
        let mut keys = KeyBindings::vanilla();
        // Pays off the `KeyBinding.unPressAllKeys()` debt any screen opening left behind.
        self.client.handle_game_input(&down, &mut keys);
        match input {
            Input::Key(code) => {
                down.on_key(code, true, false);
            }
            Input::Mouse(button) => {
                down.on_mouse_button(mouse_of(button), true);
            }
            Input::RawLwjgl(code) => panic!("cannot synthesize raw LWJGL input {code}"),
            Input::Unbound => {}
        }
        keys.set_state(input, true);
        for _ in 0..ticks {
            self.client.handle_game_input(&down, &mut keys);
            self.pump();
        }
        keys.set_state(input, false);
        self.client.handle_game_input(&InputState::new(), &mut keys);
        self.pump();
    }

    /// Close the socket and hand back every frame the peer saw, **in order**.
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

/// The index of the first frame with this packet id, in arrival order.
fn first_index<P: Packet>(seen: &[(i32, Vec<u8>)]) -> Option<usize> {
    seen.iter().position(|(id, _)| *id == P::ID)
}

fn stack(item_id: i16, count: i32) -> mc_protocol::ItemStack {
    mc_protocol::ItemStack {
        item_id,
        count,
        damage: 0,
        nbt: None,
    }
}

/// Window `0`, filled the way a real server fills it: a bow and a stack of arrows in the hotbar.
fn player_window() -> Vec<Option<mc_protocol::ItemStack>> {
    // 45 slots: 1 craft result, 4 grid, 4 armour, 27 main, 9 hotbar. `mainInventory[0]` is 36.
    let mut items = vec![None; 45];
    items[36] = Some(stack(BOW, 1));
    items[37] = Some(stack(ARROW, 32));
    items
}

async fn serve(listener: TcpListener, script: Script) -> common::RefPeer {
    let mut peer = accept_and_join(listener, 42).await;
    peer.send(&s2c::WindowItems {
        window_id: 0,
        items: player_window(),
    })
    .await;

    if script != Script::Plain {
        // java: `S2DPacketOpenWindow` → `NetHandlerPlayClient.handleOpenWindow` → `GuiChest`.
        peer.send(&s2c::OpenWindow {
            window_id: CHEST_WINDOW,
            inventory_type: "minecraft:chest".to_owned(),
            window_title: r#"{"text":"Loopback Chest"}"#.to_owned(),
            slot_count: CHEST_SLOTS,
            entity_id: None,
        })
        .await;
        // The chest's own 27 slots, then the player's 36 — a `GuiChest`'s container is both.
        let mut items = vec![None; CHEST_SLOTS as usize + 36];
        items[0] = Some(stack(STONE, 10));
        items[1] = Some(stack(STONE, 10));
        peer.send(&s2c::WindowItems {
            window_id: CHEST_WINDOW,
            items,
        })
        .await;
    }

    if script == Script::ChestThenServerClose {
        // java: `S2EPacketCloseWindow` → `handleCloseWindow` (`:1314`) →
        // `closeScreenAndDropStack()`, which sends **nothing** back.
        peer.read_until(c2s::ClickWindow::ID, |_| true).await;
        peer.send(&s2c::CloseWindow {
            window_id: CHEST_WINDOW,
        })
        .await;
    }

    peer.record_until_closed().await;
    peer
}

fn rig(script: Script) -> Option<Rig> {
    let gpu = gpu(WIDTH, HEIGHT)?;
    let dir = Box::leak(Box::new(tempfile::tempdir().expect("tempdir")));
    let config = config(dir.path(), "Dev")?;
    let mut client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
    let mut phases = PhaseList::new();
    client.install(&gpu, &mut phases, WIDTH, HEIGHT);
    let camera = mc_client::app::driver::menu_camera();

    let listener = client
        .runtime()
        .handle()
        .block_on(TcpListener::bind("127.0.0.1:0"))
        .expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let peer = client.runtime().handle().spawn(serve(listener, script));

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
    rig.wait("the client to reach the world", |client| {
        client.ui.name() == "in_game"
            && client
                .session()
                .is_some_and(|s| s.world().is_chunk_loaded(CHUNK))
    });
    Some(rig)
}

/// Pump until the server-pushed chest is on screen with its stacks in it.
fn open_chest(rig: &mut Rig) {
    rig.wait("the server's chest window", |client| {
        matches!(&client.ui, Ui::Container(screen)
            if screen.container.window_id == CHEST_WINDOW
                && screen.container.slot_stack(0).is_some())
    });
    // One more frame so `Screen::init` has centred the panel; `slot_point` reads `guiLeft`.
    rig.pump();
}

// ---------------------------------------------------------------------------------------------
// The click actually leaves the machine
// ---------------------------------------------------------------------------------------------

/// Clicking a stack in a server-opened chest must put
/// exactly one `C0EPacketClickWindow` on the wire, carrying the window the *server* chose, the
/// slot clicked, mode `0`, and the transaction id the container allocated.
///
/// java: `GuiContainer.handleMouseClick` → `PlayerControllerMP.windowClick` (`:534`) —
/// `slotClick` locally **and** `addToSendQueue(new C0EPacketClickWindow(...))`.
#[test]
fn clicking_a_slot_in_a_server_opened_chest_sends_one_click_window() {
    let Some(mut rig) = rig(Script::Chest) else {
        return;
    };
    open_chest(&mut rig);

    let point = rig.slot_point(0);
    rig.click_at(point, MouseButton::Left, false);
    rig.pump();

    // The local prediction still has to be right, or the packet is describing a lie.
    let Ui::Container(screen) = &rig.client.ui else {
        panic!("the chest must still be open");
    };
    assert_eq!(
        screen.container.cursor().map(|s| s.count),
        Some(10),
        "a plain left click picks the stack up onto the cursor"
    );
    assert!(screen.container.slot_stack(0).is_none());

    let seen = rig.finish();
    let clicks: Vec<c2s::ClickWindow> = all_of(&seen);
    assert_eq!(
        clicks.len(),
        1,
        "one press, one packet — the release is `ignoreMouseUp`"
    );
    let click = &clicks[0];
    assert_eq!(click.window_id, CHEST_WINDOW, "the server's window id");
    assert_eq!(click.slot_id, 0);
    assert_eq!(click.used_button, 0, "the left button");
    assert_eq!(click.mode, 0, "an ordinary pickup");
    assert_eq!(
        click.action_number, 1,
        "java: `getNextTransactionID` starts at 1"
    );
    // java: `slotClick` returns a *copy of the slot's stack as it was before the click*, and
    // `windowClick` puts exactly that into the packet.
    assert_eq!(click.clicked_item, Some(stack(STONE, 10)));
}

/// The same gesture with shift held is click mode `1` — `transferStackInSlot` — and
/// the stack ends up in the player's own inventory rather than on the cursor.
///
/// java: `GuiContainer.mouseClicked:359` picks `clickType == 1` from `isShiftKeyDown()`. The
/// `shift` argument was a hard-coded `false`, so this packet could not exist.
#[test]
fn shift_clicking_a_slot_sends_click_mode_one() {
    let Some(mut rig) = rig(Script::Chest) else {
        return;
    };
    open_chest(&mut rig);

    let point = rig.slot_point(0);
    rig.click_at(point, MouseButton::Left, true);
    rig.pump();

    let Ui::Container(screen) = &rig.client.ui else {
        panic!("the chest must still be open");
    };
    assert!(
        screen.container.cursor().is_none(),
        "a transfer never touches the cursor"
    );
    assert!(screen.container.slot_stack(0).is_none());
    assert!(
        (CHEST_SLOTS as usize..CHEST_SLOTS as usize + 36)
            .any(|slot| screen.container.slot_stack(slot).is_some()),
        "the stack must have moved into the player's own slots"
    );

    let seen = rig.finish();
    let clicks: Vec<c2s::ClickWindow> = all_of(&seen);
    assert_eq!(clicks.len(), 1);
    assert_eq!(
        clicks[0].mode, 1,
        "shift-click is mode 1; mode 0 means the modifier never arrived"
    );
    assert_eq!(clicks[0].slot_id, 0);
    assert_eq!(clicks[0].window_id, CHEST_WINDOW);
}

// ---------------------------------------------------------------------------------------------
// The close
// ---------------------------------------------------------------------------------------------

/// Escape closes the screen *and* tells the server, with the server's own window id
/// — and the close lands **after** the click that preceded it, or the server would process a
/// `C0E` against a window it had already torn down.
///
/// java: `GuiContainer.keyTyped:696` → `EntityPlayerSP.closeScreen:330` →
/// `C0DPacketCloseWindow(openContainer.windowId)`, then `displayGuiScreen(null)`.
#[test]
fn escape_sends_close_window_after_the_clicks_it_follows() {
    let Some(mut rig) = rig(Script::Chest) else {
        return;
    };
    open_chest(&mut rig);

    let point = rig.slot_point(0);
    rig.click_at(point, MouseButton::Left, false);
    rig.key(KeyCode::Escape);
    assert_eq!(rig.client.ui.name(), "in_game", "Escape leaves the screen");
    rig.pump();

    let seen = rig.finish();
    let closes: Vec<c2s::CloseWindow> = all_of(&seen);
    assert_eq!(closes.len(), 1, "exactly one close");
    assert_eq!(closes[0].window_id, CHEST_WINDOW);

    let click_at = first_index::<c2s::ClickWindow>(&seen).expect("the click must have gone out");
    let close_at = first_index::<c2s::CloseWindow>(&seen).expect("the close must have gone out");
    assert!(
        click_at < close_at,
        "the click has to reach the server before the window is torn down \
         (click at frame {click_at}, close at frame {close_at})"
    );
}

/// **A07-03's window-`0` case, which the brief called out as the dangerous one.**
///
/// Vanilla really does send `C0DPacketCloseWindow(0)` when the player closes their own
/// inventory: `EntityPlayer`'s `openContainer` starts as `inventoryContainer`
/// (`EntityPlayer.java:182`), a `ContainerPlayer` whose `windowId` is `Container`'s default `0`.
/// Suppressing it as a "special case" would leave the server's `openContainer` pointing at a
/// window the client has left.
#[test]
fn closing_the_players_own_inventory_sends_close_window_zero() {
    let Some(mut rig) = rig(Script::Plain) else {
        return;
    };
    rig.wait("window 0's items", |client| {
        client.player_hud.hotbar[0].is_some()
    });

    rig.tap(Input::Key(KeyCode::KeyE));
    assert_eq!(rig.client.ui.name(), "container");
    // java: `GuiContainer.keyTyped` closes on the inventory key as well as on Escape.
    rig.key(KeyCode::KeyE);
    assert_eq!(rig.client.ui.name(), "in_game");
    rig.pump();

    let seen = rig.finish();
    let closes: Vec<c2s::CloseWindow> = all_of(&seen);
    assert_eq!(
        closes.iter().map(|c| c.window_id).collect::<Vec<_>>(),
        vec![0]
    );
}

/// **The other half of A07-03: a wrong close is worse than none.** A close the *server*
/// initiated must not be echoed back — `handleCloseWindow` (`:1314`) calls
/// `closeScreenAndDropStack()`, not `closeScreen()`, so no packet leaves the client.
///
/// Answering one would make the server tear down whatever window it opened next.
#[test]
fn a_server_initiated_close_is_never_answered() {
    let Some(mut rig) = rig(Script::ChestThenServerClose) else {
        return;
    };
    open_chest(&mut rig);

    // The click is what cues the peer to send its `S2EPacketCloseWindow`.
    let point = rig.slot_point(0);
    rig.click_at(point, MouseButton::Left, false);
    rig.wait("the server's close to land", |client| {
        client.ui.name() == "in_game"
    });

    let seen = rig.finish();
    assert!(
        !all_of::<c2s::ClickWindow>(&seen).is_empty(),
        "the cue click itself must have gone out"
    );
    assert!(
        all_of::<c2s::CloseWindow>(&seen).is_empty(),
        "the client answered a server-initiated close: {:?}",
        all_of::<c2s::CloseWindow>(&seen)
    );
}

// ---------------------------------------------------------------------------------------------
// Item use is released
// ---------------------------------------------------------------------------------------------

/// Holding the use button with a bow draws it; letting go must send
/// `C07PacketPlayerDigging(RELEASE_USE_ITEM)`, which is the packet that actually looses the
/// arrow. `OutboundState::release_use_item` built exactly that packet and had no caller.
///
/// The test also pins the two things vanilla's `isUsingItem()` branch suppresses while the bow
/// is drawn: the four-tick right-click repeat (`runTick:2163`'s third clause) must not re-send
/// `sendUseItem`, and the attack binding must not swing.
///
/// java: `Minecraft.runTick:2123` and `PlayerControllerMP.onStoppedUsingItem` (`:576`).
#[test]
fn drawing_a_bow_and_letting_go_sends_release_use_item() {
    let Some(mut rig) = rig(Script::Plain) else {
        return;
    };
    rig.wait("the bow to reach the hotbar", |client| {
        client.player_hud.hotbar[0].as_ref().map(|s| s.item_id) == Some(BOW)
    });
    // Straight up at open sky: `getMouseOver` misses, so the use button reaches `sendUseItem`
    // rather than `onPlayerRightClick`.
    rig.look(0.0, -90.0);
    assert_eq!(rig.client.hit, mc_client::app::interact::HitResult::Miss);

    // Twelve ticks is three times the four-tick `rightClickDelayTimer`, so a client that kept
    // repeating `rightClickMouse` while drawn would send three extra placements.
    rig.hold_then_release(Input::MOUSE_RIGHT, 12);

    let seen = rig.finish();
    let digging: Vec<c2s::PlayerDigging> = all_of(&seen);
    assert_eq!(
        digging
            .iter()
            .filter(|p| p.status == c2s::PlayerDiggingStatus::ReleaseUseItem)
            .count(),
        1,
        "exactly one RELEASE_USE_ITEM: {:?}",
        digging.iter().map(|p| p.status).collect::<Vec<_>>()
    );
    // java: `BlockPos.ORIGIN` and `EnumFacing.DOWN`, both ignored by the server for this status
    // but both on the wire.
    let release = digging
        .iter()
        .find(|p| p.status == c2s::PlayerDiggingStatus::ReleaseUseItem)
        .expect("checked above");
    assert_eq!(release.position, mc_math::BlockPos::new(0, 0, 0));
    assert_eq!(release.face, mc_math::Facing::Down);

    let placements: Vec<c2s::PlayerBlockPlacement> = all_of(&seen);
    assert_eq!(
        placements.len(),
        1,
        "the draw is one `sendUseItem`; the repeat is gated on `!isUsingItem()`"
    );
    assert_eq!(
        placements[0].stack.as_ref().map(|s| s.item_id),
        Some(BOW),
        "the drawn item has to be attached, or the server cannot tell what was used"
    );
}

/// The release only fires for items that actually *have* a use duration. A stone block's
/// right-click is `getMaxItemUseDuration() == 0`, so nothing is ever in use and letting go sends
/// no `RELEASE_USE_ITEM` — it does keep re-firing `sendUseItem` every four ticks, which is
/// exactly what vanilla does with a non-timed item.
///
/// java: `Item.getMaxItemUseDuration` (`Item.java:373`) returns `0` unless overridden, and only
/// five classes override it.
#[test]
fn a_non_timed_item_never_produces_a_release() {
    let Some(mut rig) = rig(Script::Plain) else {
        return;
    };
    rig.wait("the bow to reach the hotbar", |client| {
        client.player_hud.hotbar[0].as_ref().map(|s| s.item_id) == Some(BOW)
    });
    // Slot 1 holds the arrows, which are an `ItemGeneric`: no use duration.
    rig.tap(Input::Key(KeyCode::Digit2));
    assert_eq!(rig.client.player_hud.selected_slot, 1);
    rig.look(0.0, -90.0);

    rig.hold_then_release(Input::MOUSE_RIGHT, 12);

    let seen = rig.finish();
    let digging: Vec<c2s::PlayerDigging> = all_of(&seen);
    assert!(
        !digging
            .iter()
            .any(|p| p.status == c2s::PlayerDiggingStatus::ReleaseUseItem),
        "nothing was in use, so nothing can be released"
    );
    assert!(
        all_of::<c2s::PlayerBlockPlacement>(&seen).len() > 1,
        "a non-timed item keeps repeating every four ticks, unlike a drawn bow"
    );
}
