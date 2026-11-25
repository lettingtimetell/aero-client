//! The client's 1.7.10 send sites: when a screen's events are handled and its packets go out,
//! and the book screen's payload in its 1.7.10 form.
//!
//! A child of `app::driver` (declared there), so it reaches `Client`'s private fields. Each hook
//! is called from one line at its 1.8.9 site and does nothing on a 1.8.9 connection; the
//! session's half is `crate::session`'s `sends_v5` module, the packets `crate::outbound::v5`.
//!
//! # The screen phase
//!
//! java17's `runTick` hands the open screen its queued mouse and key events inside the tick, at
//! `currentScreen.handleInput()` (`client/Minecraft.java:1603-1607`): after `updateController`
//! has applied the tick's packets (`:1569`) and after the death and sleep screens
//! (`:1579-1593`). A 1.8.9 session handles them in the shell's frame, as before. On a 1.7.10
//! connection the frame only queues them, one batch per frame with that frame's pointer
//! ([`Client::queue_screen_input_v5`]), and the tick's screen phase hands each batch to the same
//! `Client::handle_screen_input` ([`Client::tick_screen_input_v5`]). So a click on a window the
//! tick's packets closed sends nothing, a click reads the items those packets set, and a click
//! leaves in the first tick after it arrives. Opening a screen or losing the focus drops what
//! was queued before it, as it drops the rest of a frame's input in 1.8.9 (the input epoch
//! `prepare_modal_input` advances).
//!
//! Only vanilla's screens and keys work that way. The client's own UI, which has no vanilla
//! counterpart, keeps 1.8.9's per-frame handling in a 1.7.10 session too: its screens (the mod
//! menu with its HUD editor, the replay browser, the accounts screen, the join question), its
//! widgets over the chat (the media widget, the client's own links, the right-click copy), the
//! zoom's wheel, and the mouse buttons with no screen up (the game reads its bindings, not these
//! events; what is left is the client's click counter). Each event is routed when it is first
//! seen ([`Client::aero_takes_screen_event_v5`]); a packet the client's UI sends still waits
//! for the tick.

use mc_protocol::ItemStack;
use mc_render::{InputState, MouseEventKind, TimedMouseEvent};
use mc_ui::screen_stack::ScreenCtx;
use mc_ui::screens::chat::ChatLayout;
use winit::event::{MouseButton, MouseScrollDelta};

use super::{Client, Ui};
use crate::app::input::{merge_screen_events, KeyPress, ScreenInputEvent};
use crate::outbound::v5 as build;
use crate::session::Session;

/// One frame's screen input on a 1.7.10 connection, waiting for the tick's screen phase.
#[derive(Debug)]
struct Batch {
    /// The pointer in window pixels, as `InputState::cursor` gave it.
    cursor: (f64, f64),
    /// The framebuffer size the pointer is measured in.
    size: (u32, u32),
    /// `Client::input_epoch` when queued.
    epoch: u64,
    keys: Vec<KeyPress>,
    mouse: Vec<TimedMouseEvent>,
}

/// The screen input a 1.7.10 connection's frames queue for the tick.
#[derive(Debug, Default)]
pub(super) struct ScreenInputV5 {
    /// Oldest first.
    batches: Vec<Batch>,
    /// The last frame's framebuffer size, for the events the tick takes itself.
    size: (u32, u32),
    /// The pointer and size of the last batch: a frame with no event and the pointer where it was
    /// queues nothing.
    last: Option<((f64, f64), (u32, u32))>,
    /// The input a batch is handed to `handle_screen_input` in (kept for the next replay).
    replay: InputState,
    /// One bit per mouse button whose press the client's own UI took in the frame: its release
    /// goes the same way.
    pressed_now: u32,
    /// The press handed to `handle_screen_input` on its own (kept, so it allocates once).
    one_key: Vec<KeyPress>,
    /// Test only: handle a 1.7.10 session's events in the frame, as a 1.8.9 one does (the
    /// negative control of the tests that need the screen phase).
    #[cfg(test)]
    pub(super) at_frame_time: bool,
}

impl ScreenInputV5 {
    #[cfg(test)]
    fn at_frame_time(&self) -> bool {
        self.at_frame_time
    }

    #[cfg(not(test))]
    fn at_frame_time(&self) -> bool {
        false
    }

    /// Whether the press of `button` was the client's own UI's.
    fn pressed_now(&self, button: u16) -> bool {
        1u32.checked_shl(u32::from(button))
            .is_some_and(|bit| self.pressed_now & bit != 0)
    }

    /// Note an edge of `button`: a press the client's UI took (`now`), or anything else.
    fn note_button(&mut self, button: u16, pressed: bool, now: bool) {
        let Some(bit) = 1u32.checked_shl(u32::from(button)) else {
            return;
        };
        if pressed && now {
            self.pressed_now |= bit;
        } else {
            self.pressed_now &= !bit;
        }
    }
}

/// The winit button `InputState` records as LWJGL button `index` (the inverse of
/// `mc_render::Input::from`).
fn mouse_button(index: u16) -> MouseButton {
    match index {
        0 => MouseButton::Left,
        1 => MouseButton::Right,
        2 => MouseButton::Middle,
        3 => MouseButton::Back,
        4 => MouseButton::Forward,
        other => MouseButton::Other(other),
    }
}

/// Queue `event` on `input` as it arrived: its button or wheel, receipt time and sequence.
fn push_mouse_event(input: &mut InputState, event: &TimedMouseEvent) {
    match event.kind {
        MouseEventKind::Button { button, pressed } => {
            input.on_mouse_button_at_ms_with_sequence(
                mouse_button(button),
                pressed,
                event.at_ms,
                event.sequence,
            );
        }
        MouseEventKind::Wheel { delta } => input.on_scroll_at_ms_with_sequence(
            MouseScrollDelta::LineDelta(0.0, delta),
            event.at_ms,
            event.sequence,
        ),
    }
}

impl Client {
    /// The shell frame's screen pass on a 1.7.10 connection, ahead of 1.8.9's
    /// `handle_screen_input` in `ShellApp::frame`: hand the client's own UI its events now, as
    /// 1.8.9's pass does; queue the rest with the frame's pointer for the tick's screen phase;
    /// and run the pass's pointer half (the hover, the layout a new window needs, and the
    /// per-frame drag of the screens that poll it, java17 `GuiButton.drawButton`). `false`,
    /// having done nothing else, on a 1.8.9 connection: the caller runs its pass as before.
    pub(crate) fn queue_screen_input_v5(
        &mut self,
        input: &mut InputState,
        width: u32,
        height: u32,
    ) -> bool {
        self.screen_input_v5.size = (width, height);
        if !self.screen_input_in_tick_v5() {
            // What a 1.7.10 connection queued before it ended goes first.
            self.replay_screen_input_v5();
            return false;
        }
        self.route_screen_input_v5(input, (width, height), true);
        // The pass with nothing queued is its pointer half. A container's drag reacts to a move
        // event (java17 `GuiScreen.handleMouseInput`'s `mouseClickMove`), so this frame's move
        // waits for the tick with the rest: the pass must not see it as one.
        let drag_window = self.screen_drag_cursor_window;
        self.sample_screen_cursor(input, width, height);
        self.screen_drag_cursor_window = self.cursor_window;
        self.hold_screen_packets_v5(true);
        self.handle_screen_input(input, width, height);
        self.hold_screen_packets_v5(false);
        self.screen_drag_cursor_window = drag_window;
        true
    }

    /// The tick's screen phase, `currentScreen.handleInput()` (`ShellApp::tick`, after the death
    /// and sleep screens and before the screen update): on a 1.7.10 connection, route the events
    /// no frame has taken yet as the frame would (with the pointer now), then hand every batch,
    /// oldest first, to `handle_screen_input`. A packet the frame's pass held goes out first.
    /// Nothing on a 1.8.9 connection.
    pub(crate) fn tick_screen_input_v5(&mut self, input: &mut InputState) {
        self.release_screen_packets_v5();
        if self.diagnostic_progress.is_some() {
            return;
        }
        let size = self.screen_input_v5.size;
        if self.screen_input_in_tick_v5() && size.0 > 0 && size.1 > 0 {
            self.route_screen_input_v5(input, size, false);
        }
        self.replay_screen_input_v5();
    }

    /// Whether this connection's screen events wait for the tick: a live 1.7.10 session whose
    /// ticks the shell runs (not a replay being watched, not the startup load). It runs every
    /// frame and every tick, so the version goes first: a 1.8.9 session stops there.
    fn screen_input_in_tick_v5(&self) -> bool {
        self.plays_1_7_10()
            && !self.session.as_ref().is_some_and(Session::is_driven)
            && !self.screen_input_v5.at_frame_time()
            && self.diagnostic_progress.is_none()
            && !self.is_loading()
            && !self.tw_rb_owns_ticks()
    }

    /// Take the presses the keyboard holds and the mouse events `input` holds, in the order they
    /// arrived: the client's own UI's are handled now, one at a time, as 1.8.9's pass handles
    /// them ([`Client::handle_aero_screen_event_v5`], its packets held when `hold`); the rest are
    /// queued with `input`'s pointer, measured in `size`, for the screen phase.
    fn route_screen_input_v5(&mut self, input: &mut InputState, size: (u32, u32), hold: bool) {
        let keys = self.keyboard.drain_presses();
        let mouse = input.drain_all_mouse_events();
        let cursor = input.cursor();
        let epoch = self.input_epoch;
        let (mut queued_keys, mut queued_mouse) = (Vec::new(), Vec::new());
        if !keys.is_empty() || !mouse.is_empty() {
            // The routing hit-tests the pointer the pass would.
            self.sample_screen_cursor(input, size.0, size.1);
            for event in merge_screen_events(keys, mouse) {
                // As in 1.8.9's pass: once a screen opens or the focus goes, the rest is dropped.
                if self.input_epoch != epoch {
                    break;
                }
                let now = self.aero_takes_screen_event_v5(&event, size);
                if let ScreenInputEvent::Mouse(TimedMouseEvent {
                    kind: MouseEventKind::Button { button, pressed },
                    ..
                }) = event
                {
                    self.screen_input_v5.note_button(button, pressed, now);
                }
                match event {
                    _ if now => self.handle_aero_screen_event_v5(event, cursor, size, hold),
                    ScreenInputEvent::Key(press) => queued_keys.push(press),
                    ScreenInputEvent::Mouse(event) => queued_mouse.push(event),
                }
            }
        }
        // Without an event, only a move over an open container matters (its `mouseClickMove`);
        // a container that opens later starts a new input epoch, which drops this batch anyway.
        let container = matches!(self.ui, Ui::Container(_) | Ui::Creative(_));
        let pointer = (cursor, size);
        let state = &mut self.screen_input_v5;
        let events = !queued_keys.is_empty() || !queued_mouse.is_empty();
        if !events && (!container || state.last == Some(pointer)) {
            return;
        }
        state.last = Some(pointer);
        state.batches.push(Batch {
            cursor,
            size,
            epoch,
            keys: queued_keys,
            mouse: queued_mouse,
        });
    }

    /// Whether `event` is for the client's own UI, which keeps 1.8.9's per-frame handling: every
    /// event while one of its screens is up; the media widget's and the client's own links'
    /// left clicks and any other button over the chat (vanilla's chat reacts to the left button
    /// only; the client copies a line on the right one); the zoom's wheel; and a mouse button
    /// with no screen up (the game reads its bindings, not this event). A release goes where its
    /// press went. Everything else is vanilla's and waits for the tick.
    fn aero_takes_screen_event_v5(&self, event: &ScreenInputEvent, size: (u32, u32)) -> bool {
        let aeros_screen = matches!(
            self.ui,
            Ui::ModMenu(_) | Ui::ReplayBrowser(_) | Ui::Accounts(_) | Ui::ConfirmLeaveServer(..)
        );
        if aeros_screen || self.replay_view_wants_cursor() {
            return true;
        }
        let ScreenInputEvent::Mouse(event) = event else {
            return false;
        };
        match event.kind {
            MouseEventKind::Wheel { .. } => self.zoom_takes_wheel(),
            MouseEventKind::Button {
                button,
                pressed: false,
            } => self.screen_input_v5.pressed_now(button),
            MouseEventKind::Button { button, .. } => match self.ui {
                Ui::InGame => true,
                Ui::Chat(_) | Ui::Sleep(_) if button != 0 => true,
                Ui::Chat(_) | Ui::Sleep(_) => {
                    self.w9_takes_press(0) || self.link_takes_click_v5(size)
                }
                _ => false,
            },
        }
    }

    /// Whether a left click at the pointer lands on one of the client's own chat links
    /// ([`Client::x2_takes_chat_click`]), hit-tested as `mouse_clicked` does.
    fn link_takes_click_v5(&self, size: (u32, u32)) -> bool {
        let ctx = ScreenCtx::new(self.screen_resolution(size.0, size.1), 0.0);
        let layout = ChatLayout::from_settings(&self.settings);
        let (x, y) = self.cursor;
        let style = self.chat.style_at(x, y, &ctx, &layout, true, &self.font);
        Self::x2_takes_chat_click(style, self.keyboard.modifiers().shift)
    }

    /// One event of the client's own UI through `handle_screen_input` on its own, now, at the
    /// frame's pointer: 1.8.9's pass for that event. A packet it sends waits for the tick when
    /// `hold` (in the frame).
    fn handle_aero_screen_event_v5(
        &mut self,
        event: ScreenInputEvent,
        cursor: (f64, f64),
        size: (u32, u32),
        hold: bool,
    ) {
        let mut replay = std::mem::take(&mut self.screen_input_v5.replay);
        replay.clear_held();
        replay.on_cursor_moved(cursor.0, cursor.1);
        match event {
            ScreenInputEvent::Key(press) => {
                let state = &mut self.screen_input_v5;
                state.one_key.push(press);
                self.keyboard.requeue_presses(&mut state.one_key);
            }
            ScreenInputEvent::Mouse(event) => push_mouse_event(&mut replay, &event),
        }
        // A queued container batch measures its moves from the last replayed pointer.
        let drag_window = self.screen_drag_cursor_window;
        if hold {
            self.hold_screen_packets_v5(true);
        }
        self.handle_screen_input(&mut replay, size.0, size.1);
        if hold {
            self.hold_screen_packets_v5(false);
        }
        self.screen_drag_cursor_window = drag_window;
        self.screen_input_v5.replay = replay;
    }

    /// Hand every queued batch, oldest first, to `handle_screen_input` with its own pointer.
    /// Dropped: a batch from before a screen opened or the focus went (an older input epoch), and
    /// a batch that only moved the pointer while no container is open (only a container reads a
    /// move).
    fn replay_screen_input_v5(&mut self) {
        if self.screen_input_v5.batches.is_empty() {
            return;
        }
        // Presses no batch holds came after all of them: set aside, then queued again.
        let mut later = self.keyboard.drain_presses();
        let mut batches = std::mem::take(&mut self.screen_input_v5.batches);
        let mut replay = std::mem::take(&mut self.screen_input_v5.replay);
        for mut batch in batches.drain(..) {
            if batch.epoch != self.input_epoch {
                continue;
            }
            let events = !batch.keys.is_empty() || !batch.mouse.is_empty();
            if !events && !matches!(self.ui, Ui::Container(_) | Ui::Creative(_)) {
                continue;
            }
            replay.clear_held();
            replay.on_cursor_moved(batch.cursor.0, batch.cursor.1);
            for event in &batch.mouse {
                push_mouse_event(&mut replay, event);
            }
            self.keyboard.requeue_presses(&mut batch.keys);
            self.handle_screen_input(&mut replay, batch.size.0, batch.size.1);
        }
        // Both allocations stay for the next replay.
        batches.append(&mut self.screen_input_v5.batches);
        self.screen_input_v5.batches = batches;
        self.screen_input_v5.replay = replay;
        self.keyboard.requeue_presses(&mut later);
    }

    /// Hold the session's packets during the frame's pointer pass on a 1.7.10 connection: the
    /// tick's screen phase sends them ([`Client::tick_screen_input_v5`]).
    fn hold_screen_packets_v5(&mut self, hold: bool) {
        if let Some(session) = &mut self.session {
            session.hold_screen_packets_v5(hold);
        }
    }

    /// Send what [`Client::hold_screen_packets_v5`] held, in order.
    fn release_screen_packets_v5(&mut self) {
        if let Some(session) = &mut self.session {
            session.release_screen_packets_v5();
        }
    }

    /// After each event of the screen pass, on a 1.7.10 connection: send the clicks a key made
    /// in a container (`GuiContainer.keyTyped`'s hotbar swap, pick and drop, which the screen
    /// only queues), so they keep their place among the pass's packets instead of waiting for
    /// the tick's backstop drain, which runs after the tick's game input.
    pub(super) fn drain_key_clicks_v5(&mut self) {
        if self.plays_1_7_10() {
            self.drain_container_clicks();
        }
    }

    /// The book `GuiScreenBook.sendBookToServer` sends and keeps in the hand, on this
    /// connection: on 1.7.10 a signed book's pages as typed
    /// ([`build::signed_book_pages_as_typed`]); otherwise `stack` as the screen built it. `None`
    /// when a page of a signed book is not in the form the screen builds: the book is not sent,
    /// since its pages would reach the server as JSON.
    pub(super) fn book_as_sent_v5(&self, channel: &str, stack: ItemStack) -> Option<ItemStack> {
        if !self.plays_1_7_10() || channel != "MC|BSign" {
            return Some(stack);
        }
        let typed = build::signed_book_pages_as_typed(&stack);
        if typed.is_none() {
            tracing::error!("not sending a signed book: a page is not a JSON string literal");
        }
        typed
    }
}

#[cfg(test)]
mod tests {
    use mc_nbt::Nbt;
    use mc_protocol::connection::LoginOutcome;
    use mc_protocol::play::{s2c as s2c47, C2SPlay, S2CPlay};
    use mc_protocol::v5::{s2c as s2c5, C2SPlay5, NbtBudget, PacketBufferV5, S2CPlay5};
    use mc_protocol::PacketBuffer;
    use mc_render::{Camera, GameSettings, ShellApp, TickCtx};
    use mc_ui::screen_stack::ScreenCtx;
    use mc_ui::screens::book::BookScreen;
    use mc_ui::screens::container::v5::open_window_v47;
    use mc_version::GameVersion;
    use tokio::sync::mpsc::{Sender, UnboundedReceiver};
    use winit::keyboard::KeyCode;

    use super::*;
    use crate::app::config::{ClientConfig, LoginMode};
    use crate::app::input::RawKeyboard;
    use crate::net::{NetHandle, ToGame, ToNet};
    use crate::session::{JoinSettings, Session};
    use crate::sync::WindowEvent;

    type Joined = (
        Client,
        tempfile::TempDir,
        Sender<ToGame>,
        UnboundedReceiver<ToNet>,
    );

    const WINDOW: (u32, u32) = (854, 480);

    /// A client with an offline `version` session that has joined a world (entity 27), on
    /// channels the test holds; `None` (a skip) without the vanilla jar.
    fn joined(version: GameVersion) -> Option<Joined> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("skip: no vanilla jar in this checkout (set MC_JARS_DIR)");
            return None;
        }
        let mut client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        let (handle, to_game, to_net) = NetHandle::new_pair();
        client.session = Some(Session::with_version(
            handle,
            JoinSettings::default(),
            version,
        ));
        to_game
            .try_send(ToGame::Connected(LoginOutcome {
                uuid: uuid::Uuid::from_u128(7),
                username: "Dev".to_owned(),
                compression_threshold: None,
                encrypted: false,
            }))
            .expect("room");
        let join = if version.is_1_7_10() {
            ToGame::PacketV5(Box::new(S2CPlay5::JoinGame(s2c5::JoinGame {
                entity_id: 27,
                hardcore_mode: false,
                game_mode: 0,
                dimension: 0,
                difficulty: 2,
                max_players: 20,
                level_type: "flat".to_owned(),
            })))
        } else {
            ToGame::Packet(S2CPlay::JoinGame(s2c47::JoinGame {
                entity_id: 27,
                hardcore_mode: false,
                game_mode: 0,
                dimension: 0,
                difficulty: 2,
                max_players: 20,
                level_type: "flat".to_owned(),
                reduced_debug_info: false,
            }))
        };
        to_game.try_send(join).expect("room");
        client.poll_in_place();
        client.drain_v5_before_world(&mut Camera::default());
        assert!(
            client
                .session
                .as_ref()
                .is_some_and(|session| session.phase().is_live()),
            "joined"
        );
        Some((client, dir, to_game, to_net))
    }

    /// The shell frame's screen pass, as `ShellApp::frame` runs it with no diagnostic run
    /// ([`the_shell_runs_the_screen_phase_hooks`]).
    fn frame(client: &mut Client, input: &mut InputState) {
        frame_at(client, input, WINDOW);
    }

    /// [`frame`] in a `(width, height)` window.
    fn frame_at(client: &mut Client, input: &mut InputState, (width, height): (u32, u32)) {
        if !client.queue_screen_input_v5(input, width, height) {
            client.handle_screen_input(input, width, height);
        }
    }

    /// One iteration's tick, as the shell runs it.
    fn tick(client: &mut Client, input: &mut InputState) {
        let mut camera = Camera::default();
        let mut settings = GameSettings::default();
        ShellApp::before_ticks(client, &mut camera);
        ShellApp::tick(
            client,
            &mut TickCtx {
                input,
                settings: &mut settings,
                camera: &mut camera,
                tick: 1,
            },
        );
    }

    fn drain(to_net: &mut UnboundedReceiver<ToNet>) -> Vec<ToNet> {
        std::iter::from_fn(|| to_net.try_recv().ok()).collect()
    }

    /// The container clicks among `sent`, as (window, slot).
    fn clicks(sent: &[ToNet]) -> Vec<(i32, i16)> {
        sent.iter()
            .filter_map(|message| match message {
                ToNet::Send(C2SPlay::ClickWindow(click)) => {
                    Some((i32::from(click.window_id), click.slot_id))
                }
                ToNet::SendV5(C2SPlay5::ClickWindow(click)) => {
                    Some((i32::from(click.window_id), click.slot_id))
                }
                _ => None,
            })
            .collect()
    }

    /// The C0D packets among `sent`.
    fn closes(sent: &[ToNet]) -> usize {
        sent.iter()
            .filter(|message| {
                matches!(
                    message,
                    ToNet::Send(C2SPlay::CloseWindow(_)) | ToNet::SendV5(C2SPlay5::CloseWindow(_))
                )
            })
            .count()
    }

    /// A chest as window `id`, the way the client opens the server's S2D.
    fn open_chest(client: &mut Client, id: u8) {
        let chest = open_window_v47(&s2c5::OpenWindow {
            window_id: id,
            inventory_type: 0,
            title: "Chest".to_owned(),
            slot_count: 27,
            use_provided_title: true,
            entity_id: 0,
        })
        .expect("a chest");
        client.react_to_window(&WindowEvent::Open(chest));
        assert!(matches!(client.ui, Ui::Container(_)), "the chest is open");
    }

    /// A left click off the open window's panel (slot -999), pressed and released in two frames.
    fn click_outside(client: &mut Client, input: &mut InputState) {
        input.on_cursor_moved(3.0, 3.0);
        frame(client, input);
        input.on_mouse_button(MouseButton::Left, true);
        frame(client, input);
        input.on_mouse_button(MouseButton::Left, false);
        frame(client, input);
    }

    /// The server's S2E for window `id`, which a 1.7.10 session applies in the tick.
    fn close_from_server(to_game: &Sender<ToGame>, id: u8) {
        to_game
            .try_send(ToGame::PacketV5(Box::new(S2CPlay5::CloseWindow(
                s2c5::CloseWindow { window_id: id },
            ))))
            .expect("room");
    }

    /// The code of the function whose first line is `signature`, without its comments and
    /// whitespace (a commented-out hook counts as deleted).
    fn body_of(source: &str, signature: &str) -> String {
        let start = source
            .find(signature)
            .unwrap_or_else(|| panic!("`{signature}` is gone"));
        let rest = &source[start..];
        let end = rest.find("\n    }\n").expect("the function's end");
        rest[..end]
            .lines()
            .map(|line| line.split_once("//").map_or(line, |(code, _)| code))
            .flat_map(str::split_whitespace)
            .collect()
    }

    /// The shell runs what these tests and the 1.7.10 oracle drive: `ShellApp::frame` passes its
    /// screen input to [`Client::queue_screen_input_v5`] ahead of 1.8.9's pass, and
    /// `ShellApp::tick` runs the screen phase after the death and sleep screens and before the
    /// pre-world drain and the game input. Deleting either hook fails here.
    #[test]
    fn the_shell_runs_the_screen_phase_hooks() {
        let driver = include_str!("driver.rs");
        let frame = body_of(
            driver,
            "    fn frame(&mut self, ctx: &mut mc_render::FrameCtx<'_>) {",
        );
        assert!(
            frame.contains(
                "ifself.diagnostic_progress.is_none()\
                 &&!self.queue_screen_input_v5(ctx.input,width,height)\
                 {self.handle_screen_input(ctx.input,width,height);}"
            ),
            "the frame's screen pass"
        );
        assert_eq!(frame.matches("handle_screen_input(").count(), 1);
        let tick = body_of(
            driver,
            "    fn tick(&mut self, ctx: &mut mc_render::TickCtx<'_>) {",
        );
        let at = |what: &str| tick.find(what).unwrap_or_else(|| panic!("{what} is gone"));
        let in_world = at("self.drain_v5_in_world(ctx.camera);");
        let death = at("self.tick_death_and_sleep_screens();");
        let screen = at("self.tick_screen_input_v5(ctx.input);");
        let pre_world = at("self.drain_v5_before_world(ctx.camera);");
        let game_input = at("self.handle_game_input(");
        assert!(in_world < death && death < screen && screen < pre_world && pre_world < game_input);
        assert_eq!(tick.matches("tick_screen_input_v5(").count(), 1);
    }

    #[test]
    fn mouse_buttons_map_back_to_their_lwjgl_index() {
        for index in 0..16 {
            assert_eq!(
                mc_render::Input::from(mouse_button(index)),
                mc_render::Input::Mouse(index)
            );
        }
    }

    #[test]
    fn requeued_presses_go_ahead_of_later_ones() {
        let mut keyboard = RawKeyboard::new();
        keyboard.push(KeyCode::KeyA, true, false, Some('a'));
        keyboard.push(KeyCode::KeyB, true, false, Some('b'));
        let mut taken = keyboard.drain_presses();
        keyboard.push(KeyCode::KeyC, true, false, Some('c'));
        keyboard.requeue_presses(&mut taken);
        assert!(taken.is_empty());
        let order: Vec<KeyCode> = keyboard.drain_presses().iter().map(|p| p.code).collect();
        assert_eq!(order, [KeyCode::KeyA, KeyCode::KeyB, KeyCode::KeyC]);
    }

    /// A 1.7.10 click leaves in the next tick's screen phase, not in the frame; a 1.8.9 one in
    /// the frame, as before, and nothing is queued for it.
    #[test]
    fn a_1_7_10_click_leaves_in_the_next_tick_and_a_1_8_9_one_in_the_frame() {
        for version in [GameVersion::V1_7_10, GameVersion::V1_8_9] {
            let Some((mut client, _dir, _to_game, mut to_net)) = joined(version) else {
                return;
            };
            let mut input = InputState::new();
            open_chest(&mut client, 3);
            drain(&mut to_net);
            click_outside(&mut client, &mut input);
            let in_frame = clicks(&drain(&mut to_net));
            tick(&mut client, &mut input);
            let in_tick = clicks(&drain(&mut to_net));
            if version.is_1_7_10() {
                assert_eq!((in_frame, in_tick), (vec![], vec![(3, -999)]), "{version}");
            } else {
                assert_eq!((in_frame, in_tick), (vec![(3, -999)], vec![]), "{version}");
                assert!(client.screen_input_v5.batches.is_empty());
            }
        }
    }

    /// java17 hands a screen its events after the tick's packets, and only if it is still open:
    /// a click queued before the server's S2E closes the window sends nothing. Handled in the
    /// frame instead (the negative control), it goes out for the window the server is closing.
    #[test]
    fn a_click_queued_before_the_server_closes_its_window_is_never_sent() {
        for at_frame_time in [false, true] {
            let Some((mut client, _dir, to_game, mut to_net)) = joined(GameVersion::V1_7_10) else {
                return;
            };
            client.screen_input_v5.at_frame_time = at_frame_time;
            let mut input = InputState::new();
            open_chest(&mut client, 3);
            drain(&mut to_net);
            click_outside(&mut client, &mut input);
            let mut sent = drain(&mut to_net);
            close_from_server(&to_game, 3);
            tick(&mut client, &mut input);
            sent.extend(drain(&mut to_net));
            assert!(matches!(client.ui, Ui::InGame), "the server closed it");
            let expected = if at_frame_time {
                vec![(3, -999)]
            } else {
                vec![]
            };
            assert_eq!(clicks(&sent), expected, "in the frame: {at_frame_time}");
        }
    }

    /// The same for an Escape: once the server has closed the window it reaches the game, which
    /// opens the game menu (java17's keyboard loop), and no C0D goes out. Handled in the frame,
    /// it closes the window itself with a C0D.
    #[test]
    fn an_escape_queued_before_the_server_closes_its_window_sends_no_close() {
        for at_frame_time in [false, true] {
            let Some((mut client, _dir, to_game, mut to_net)) = joined(GameVersion::V1_7_10) else {
                return;
            };
            client.screen_input_v5.at_frame_time = at_frame_time;
            let mut input = InputState::new();
            open_chest(&mut client, 3);
            drain(&mut to_net);
            client.handle_raw_key(KeyCode::Escape, true, false, None);
            frame(&mut client, &mut input);
            client.handle_raw_key(KeyCode::Escape, false, false, None);
            let mut sent = drain(&mut to_net);
            close_from_server(&to_game, 3);
            tick(&mut client, &mut input);
            sent.extend(drain(&mut to_net));
            if at_frame_time {
                assert_eq!(closes(&sent), 1);
            } else {
                assert_eq!(closes(&sent), 0);
                assert!(matches!(client.ui, Ui::Pause(_)), "the game menu");
            }
        }
    }

    /// The bed screen the tick's packets close gets none of its queued input either: a chat line
    /// typed just before the server's wake-up (animation 2) in that tick is never sent; handled
    /// in the frame, it goes out first (java17 closes the screen before `handleInput`,
    /// `client/Minecraft.java:1590-1592`).
    #[test]
    fn chat_typed_on_the_bed_screen_before_a_wake_up_in_that_tick_is_never_sent() {
        for at_frame_time in [false, true] {
            let Some((mut client, _dir, to_game, mut to_net)) = joined(GameVersion::V1_7_10) else {
                return;
            };
            client.screen_input_v5.at_frame_time = at_frame_time;
            client
                .session
                .as_mut()
                .expect("joined")
                .state_mut()
                .local
                .sleeping = true;
            client.open_sleep_screen();
            assert!(matches!(client.ui, Ui::Sleep(_)), "the bed screen");
            drain(&mut to_net);
            let mut input = InputState::new();
            for (code, typed) in [
                (KeyCode::KeyH, Some('h')),
                (KeyCode::KeyI, Some('i')),
                (KeyCode::Enter, None),
            ] {
                client.handle_raw_key(code, true, false, typed);
                client.handle_raw_key(code, false, false, None);
            }
            frame(&mut client, &mut input);
            let mut sent = drain(&mut to_net);
            to_game
                .try_send(ToGame::PacketV5(Box::new(S2CPlay5::Animation(
                    s2c5::Animation {
                        entity_id: 27,
                        animation_type: 2,
                    },
                ))))
                .expect("room");
            tick(&mut client, &mut input);
            sent.extend(drain(&mut to_net));
            assert!(
                !matches!(client.ui, Ui::Sleep(_)),
                "awake, the bed screen closed"
            );
            let lines: Vec<&str> = sent
                .iter()
                .filter_map(|message| match message {
                    ToNet::Send(C2SPlay::ChatMessage(chat)) => Some(chat.message.as_str()),
                    _ => None,
                })
                .collect();
            let expected: &[&str] = if at_frame_time { &["hi"] } else { &[] };
            assert_eq!(lines, expected, "in the frame: {at_frame_time}");
        }
    }

    /// The client's own UI keeps 1.8.9's per-frame handling in a 1.7.10 session: a click on a
    /// mod menu switch turns the mod on in its frame, before any tick, and nothing is queued.
    /// (A vanilla screen's click waits for the tick:
    /// `a_1_7_10_click_leaves_in_the_next_tick_and_a_1_8_9_one_in_the_frame`.)
    #[test]
    fn a_mod_menu_click_takes_effect_in_its_frame_in_a_1_7_10_session() {
        let Some((mut client, _dir, _to_game, _to_net)) = joined(GameVersion::V1_7_10) else {
            return;
        };
        const SIZE: (u32, u32) = (1280, 720);
        let t0 = std::time::Instant::now();
        client.open_mod_menu(t0);
        let mut input = InputState::new();
        // The first frame samples the pointer and the GUI geometry; the layout records where
        // the switches are once the opening animation is over.
        frame_at(&mut client, &mut input, SIZE);
        client
            .layout_mod_menu(
                SIZE,
                &mut mc_ui::widget::DrawList::new(),
                t0 + std::time::Duration::from_secs(1),
            )
            .expect("the menu is on screen");
        let fps = client.mods.settings.index_of("fps").expect("the FPS mod");
        let Ui::ModMenu(menu) = &client.ui else {
            panic!("the mod menu, got {}", client.ui.name());
        };
        let (x, y) = menu.switch_rect(fps).expect("the FPS switch").center();
        assert!(!client.mods.settings.enabled("fps"));

        input.on_cursor_moved(f64::from(x), f64::from(y));
        input.on_mouse_button(MouseButton::Left, true);
        input.on_mouse_button(MouseButton::Left, false);
        frame_at(&mut client, &mut input, SIZE);
        assert!(
            client.mods.settings.enabled("fps"),
            "on in the click's frame"
        );
        assert!(
            client.screen_input_v5.batches.is_empty(),
            "nothing waits for the tick"
        );
    }

    /// The window pixel `client` maps to the GUI point where `found` holds, searched upward over
    /// the window's lower left quarter (where the chat is).
    fn window_point_where(
        client: &Client,
        found: impl Fn(&Client, (i32, i32)) -> bool,
    ) -> Option<(f64, f64)> {
        let resolution = client.screen_resolution(WINDOW.0, WINDOW.1);
        (WINDOW.1 / 2..WINDOW.1)
            .rev()
            .flat_map(|py| (0..WINDOW.0 / 2).map(move |px| (f64::from(px), f64::from(py))))
            .find(|&(px, py)| found(client, resolution.cursor_to_gui(px, py)))
    }

    /// Over the chat: the client's right-click copy (no vanilla meaning) takes effect in the
    /// frame, while a left click on a `/spawn` link, vanilla's, sends its C01 in the next tick.
    #[test]
    fn over_the_chat_the_copy_is_per_frame_and_a_link_click_waits_for_the_tick() {
        use mc_ui::font::component::{ChatComponent, ChatStyle, ClickAction, ClickEvent};
        use mc_ui::screens::chat::ChatScreen;

        let Some((mut client, _dir, _to_game, mut to_net)) = joined(GameVersion::V1_7_10) else {
            return;
        };
        client.settings.chat_links = true;
        client.mods.settings.set_enabled("chat", true);
        client
            .mods
            .settings
            .set_value("chat", "copy", mc_mods::OptionValue::Bool(true));
        client.m2_sync();
        let name = ChatComponent::text("Steve").with_style(ChatStyle {
            click_event: Some(ClickEvent {
                action: ClickAction::RunCommand,
                value: "/spawn".to_owned(),
            }),
            ..ChatStyle::default()
        });
        let layout = ChatLayout::from_settings(&client.settings);
        client.chat.add_message(
            ChatComponent::text("<a> ").with_extra(name),
            0,
            0,
            &layout,
            &client.font,
            &client.assets.language,
            true,
        );
        client.ui = Ui::Chat(Box::new(ChatScreen::new("", 0, &client.font)));
        let ctx = ScreenCtx::new(client.screen_resolution(WINDOW.0, WINDOW.1), 0.0);
        let on_link = window_point_where(&client, |client, (x, y)| {
            client
                .chat
                .style_at(x, y, &ctx, &layout, true, &client.font)
                .is_some_and(|style| style.click_event.is_some())
        })
        .expect("a pixel over the link");
        drain(&mut to_net);
        let mut input = InputState::new();
        input.on_cursor_moved(on_link.0, on_link.1);

        input.on_mouse_button(MouseButton::Left, true);
        input.on_mouse_button(MouseButton::Left, false);
        frame(&mut client, &mut input);
        let commands = |sent: &[ToNet]| -> Vec<String> {
            sent.iter()
                .filter_map(|message| match message {
                    ToNet::Send(C2SPlay::ChatMessage(chat)) => Some(chat.message.clone()),
                    _ => None,
                })
                .collect()
        };
        assert_eq!(
            commands(&drain(&mut to_net)),
            Vec::<String>::new(),
            "not in the frame"
        );
        tick(&mut client, &mut input);
        assert_eq!(commands(&drain(&mut to_net)), ["/spawn"], "in the tick");

        // The copy's notice is a new line: the click comes last, as it moves the link up.
        let lines = client.chat.lines().len();
        input.on_mouse_button(MouseButton::Right, true);
        input.on_mouse_button(MouseButton::Right, false);
        frame(&mut client, &mut input);
        assert_eq!(
            client.chat.lines().len(),
            lines + 1,
            "the copy notice, in the frame"
        );
    }

    /// The routing itself: the client's screens and a mouse button with no screen up are the
    /// client's; keys, a plain wheel notch in game and a container's click are vanilla's.
    #[test]
    fn the_routing_gives_vanilla_its_screens_and_keys_and_the_client_its_own() {
        let Some((mut client, _dir, _to_game, _to_net)) = joined(GameVersion::V1_7_10) else {
            return;
        };
        let press = |pressed| {
            ScreenInputEvent::Mouse(TimedMouseEvent {
                kind: MouseEventKind::Button { button: 0, pressed },
                at_ms: 0,
                sequence: 0,
            })
        };
        let wheel = ScreenInputEvent::Mouse(TimedMouseEvent {
            kind: MouseEventKind::Wheel { delta: 1.0 },
            at_ms: 0,
            sequence: 0,
        });
        let mut keyboard = RawKeyboard::new();
        keyboard.push(KeyCode::Escape, true, false, None);
        let key = ScreenInputEvent::Key(keyboard.drain_presses()[0]);

        client.ui = Ui::InGame;
        assert!(client.aero_takes_screen_event_v5(&press(true), WINDOW));
        assert!(
            !client.aero_takes_screen_event_v5(&key, WINDOW),
            "Esc in game"
        );
        assert!(
            !client.aero_takes_screen_event_v5(&wheel, WINDOW),
            "the hotbar"
        );
        open_chest(&mut client, 3);
        assert!(!client.aero_takes_screen_event_v5(&press(true), WINDOW));
        assert!(!client.aero_takes_screen_event_v5(&key, WINDOW));
        client.open_mod_menu(std::time::Instant::now());
        for event in [press(true), press(false), key, wheel] {
            assert!(
                client.aero_takes_screen_event_v5(&event, WINDOW),
                "{event:?}"
            );
        }
    }

    /// A window the server opens over the one clicked in drops the queued click, as opening a
    /// screen drops the unread input: no C0E for the replaced window (nor for the new one).
    #[test]
    fn a_click_queued_before_the_server_replaces_its_window_is_dropped() {
        let Some((mut client, _dir, to_game, mut to_net)) = joined(GameVersion::V1_7_10) else {
            return;
        };
        let mut input = InputState::new();
        open_chest(&mut client, 3);
        drain(&mut to_net);
        click_outside(&mut client, &mut input);
        to_game
            .try_send(ToGame::PacketV5(Box::new(S2CPlay5::OpenWindow(
                s2c5::OpenWindow {
                    window_id: 4,
                    inventory_type: 0,
                    title: "Chest".to_owned(),
                    slot_count: 27,
                    use_provided_title: true,
                    entity_id: 0,
                },
            ))))
            .expect("room");
        tick(&mut client, &mut input);
        assert!(
            matches!(&client.ui, Ui::Container(screen) if screen.container.window_id == 4),
            "the server's new window"
        );
        assert_eq!(clicks(&drain(&mut to_net)), []);
    }

    /// Signing on the real book screen of a 1.7.10 session sends `MC|BSign` with the pages as
    /// typed, though the screen builds 1.8.9's JSON pages.
    #[test]
    fn signing_on_the_book_screen_sends_the_typed_pages() {
        let Some((mut client, _dir, _to_game, mut to_net)) = joined(GameVersion::V1_7_10) else {
            return;
        };
        let book = ItemStack {
            item_id: 386,
            count: 1,
            damage: 0,
            nbt: None,
        };
        client.show_screen(Some(Ui::Book(Box::new(BookScreen::new(book, "Dev", true)))));
        drain(&mut to_net);
        let ctx = ScreenCtx::new(client.resolution(WINDOW.0, WINDOW.1), 0.0);
        let typed = "a < b & c's \"q\" \u{e9}";
        let mods = mc_ui::widget::keys::Modifiers::default();
        let Ui::Book(screen) = &mut client.ui else {
            panic!("the book screen");
        };
        screen.init(&ctx, &client.assets.language);
        for ch in typed.chars() {
            screen.key_typed(Some(ch), 0, mods, &client.clipboard, &client.font);
        }
        assert_eq!(screen.current_page(), typed);
        // "Sign", the title, then "Sign and Close" in the same place.
        let (x, y) = (ctx.width / 2 - 50, 4 + 192 + 10);
        screen.mouse_clicked(x, y, 0, &client.font, &client.assets.language);
        assert!(screen.is_signing());
        for ch in "Notes".chars() {
            screen.key_typed(Some(ch), 0, mods, &client.clipboard, &client.font);
        }
        screen.mouse_clicked(x, y, 0, &client.font, &client.assets.language);
        client.drain_book_actions();

        let sent = drain(&mut to_net);
        let stack = sent
            .iter()
            .find_map(|message| match message {
                ToNet::SendV5(C2SPlay5::PluginMessage(message))
                    if message.channel == "MC|BSign" =>
                {
                    let mut payload = PacketBuffer::from_slice(&message.data);
                    payload
                        .read_item_v5(&mut NbtBudget::per_packet())
                        .expect("a protocol-5 slot")
                }
                _ => None,
            })
            .expect("MC|BSign");
        assert_eq!(stack.item_id, 387);
        let tag = stack.nbt.as_ref().expect("the book's tag");
        assert_eq!(
            tag.get_list("pages"),
            Some(&[Nbt::String(typed.to_owned())][..])
        );
        assert_eq!(tag.get("title"), Some(&Nbt::String("Notes".to_owned())));
    }
}
