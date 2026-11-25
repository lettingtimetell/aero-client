//! Turning raw input into game intent.
//!
//! java: `util/MovementInputFromOptions.java` (the movement half) and the key handling in
//! `Minecraft.runTick` (`:1741`) — hotbar selection, the sneak/sprint edges, and the
//! attack/use buttons.
//!
//! Nothing here touches the network or the world: it reads the player's live [`KeyBindings`] and
//! produces [`MoveInput`], which is exactly what S13's `Predictor::tick` wants. That makes it
//! testable by feeding binding edges and reading the struct back.

use std::collections::HashSet;

use mc_physics::state::MoveInput;
use mc_render::input::{keys, lwjgl_of_keycode};
use mc_render::{Input, InputState, KeyBindings, TimedMouseEvent};
use mc_ui::screens::container::ContainerBindings;
use mc_ui::widget::keys::Modifiers;
use winit::keyboard::KeyCode;

/// Build one tick's movement intent.
///
/// java: `MovementInputFromOptions.updatePlayerMoveState`, which reads the four direction keys,
/// jump and sneak, and applies the sneak slowdown to the *analogue* values. The slowdown is not
/// applied here: S13's `tick_player` owns it (`PhysicsState::sneaking` feeds
/// `moveEntityWithHeading`'s `0.3` multiplier), and doing it twice would halve the speed.
///
/// `yaw`/`pitch` come from the camera rather than from the keys, because vanilla's
/// `EntityPlayerSP` carries rotation on the entity and the movement code reads it there.
///
/// `using_item` is likewise not a key. java: `EntityPlayerSP.onLivingUpdate` (`:788`) applies the
/// `0.2F` movement scale while `EntityPlayer.isUsingItem()` — an item is actively being *consumed*
/// (eating, drinking, drawing a bow, blocking), which is not the same fact as the use-item button
/// being held down this instant. The distinction matters: the button is released the moment a bow
/// is loosed, while the draw itself lasted many ticks. The caller passes
/// [`PlayerController::is_using_item`](crate::app::interact::PlayerController::is_using_item),
/// which is the state Wave 0 already tracks; [`using_item`] (the raw button) is deliberately
/// *not* the value to pass here.
#[must_use]
pub fn move_input(bindings: &KeyBindings, yaw: f32, pitch: f32, using_item: bool) -> MoveInput {
    let down = |name: &str| bindings.is_down(name);
    MoveInput {
        forward: down(keys::FORWARD),
        back: down(keys::BACK),
        left: down(keys::LEFT),
        right: down(keys::RIGHT),
        jump: down(keys::JUMP),
        sneak: down(keys::SNEAK),
        // java: both sprint branches in `EntityPlayerSP.onLivingUpdate` read
        // `keyBindSprint.isKeyDown()` directly. Their existing `moveForward >= 0.8F` and
        // `!isSneaking()` conditions decide whether sprint engages; pre-filtering the binding
        // state here duplicates those gates and gives this field different semantics from vanilla.
        sprint: down(keys::SPRINT),
        yaw,
        pitch,
        using_item,
    }
}

/// Whether the attack (dig) button is held. java: `keyBindAttack`.
#[must_use]
pub fn attacking(bindings: &KeyBindings) -> bool {
    bindings.is_down(keys::ATTACK)
}

/// Whether the use-item (place) button is held. java: `keyBindUseItem`.
#[must_use]
pub fn using_item(bindings: &KeyBindings) -> bool {
    bindings.is_down(keys::USE_ITEM)
}

/// The hotbar slot a number key selects, if any, as a **0-based** index.
///
/// java: `Minecraft.runTick`'s
/// `for (int j = 0; j < 9; ++j) if (this.gameSettings.keyBindsHotbar[j].isPressed())`, whose
/// binding *names* are 1-based (`key.hotbar.1`) while the slot it assigns to
/// `inventory.currentItem` is 0-based. S10's `keys::hotbar` takes the 1-based name, so the
/// conversion happens here rather than at every call site.
#[must_use]
pub fn hotbar_selection(bindings: &mut KeyBindings) -> Option<u8> {
    let mut selected = None;
    for name in 1u8..=9 {
        // java: the loop does not break after an `isPressed()` result.  Every binding gets one
        // press consumed, and the final assignment therefore makes the highest pressed slot win.
        if bindings.take_pressed(&keys::hotbar(name)) {
            selected = Some(name - 1);
        }
    }
    selected
}

/// Snapshot the live game settings into the backend-neutral codes container screens consume.
///
/// Keyboard events arrive as positive LWJGL scan codes. Mouse pick-block arrives separately as
/// a zero-based button index, matching Java's `getKeyCode() + 100` comparison.
#[must_use]
pub fn container_bindings(bindings: &KeyBindings) -> ContainerBindings {
    let action = |name: &str| {
        bindings
            .get(name)
            .map_or(Input::Unbound, |binding| binding.binding())
    };
    let pick = action(keys::PICK_ITEM);
    let pick_block_button = match pick {
        Input::Mouse(button) => i32::from(button),
        Input::RawLwjgl(code) if code < 0 => code + 100,
        _ => -1,
    };
    ContainerBindings {
        inventory_key: action(keys::INVENTORY).to_lwjgl(),
        drop_key: action(keys::DROP).to_lwjgl(),
        hotbar_keys: std::array::from_fn(|slot| action(&keys::hotbar(slot as u8 + 1)).to_lwjgl()),
        pick_block_key: pick.to_lwjgl(),
        pick_block_button,
        chat_key: action(keys::CHAT).to_lwjgl(),
    }
}

/// Apply a mouse-wheel scroll to the held slot.
///
/// java: `Minecraft.runTick` (`:1900`) — `Mouse.getEventDWheel()` is clamped to ±1, **inverted**
/// relative to the slot index (scrolling up decrements), and wraps at both ends.
#[must_use]
pub fn scroll_hotbar(current: u8, scroll: f32) -> u8 {
    if scroll == 0.0 {
        return current;
    }
    let step: i32 = if scroll > 0.0 { -1 } else { 1 };
    let mut slot = i32::from(current) + step;
    while slot < 0 {
        slot += 9;
    }
    while slot > 8 {
        slot -= 9;
    }
    slot as u8
}

// -------------------------------------------------------------------------------------------
// Raw keyboard and mouse edges
// -------------------------------------------------------------------------------------------

/// One key going down. java: what `while (Keyboard.next())` hands `GuiScreen.handleKeyboardInput`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyPress {
    /// Monotonic receipt sequence shared with mouse events by the live shell.
    pub sequence: u64,
    /// The physical key.
    pub code: KeyCode,
    /// Its LWJGL 2 scancode, which is what [`mc_ui::screen_stack::Screen::key_typed`] takes.
    pub lwjgl: i32,
    /// The character it produced, if any. java: `Keyboard.getEventCharacter()`.
    ///
    /// Sourced from winit's own `KeyEvent::text` — the OS's answer for the user's actual
    /// keyboard layout — via [`RawKeyboard::push`]. Only falls back to [`char_for`]'s
    /// US-QWERTY guess when winit reports no text for the press at all.
    pub typed: Option<char>,
    /// Modifiers held at the time.
    pub mods: Modifiers,
    /// Whether this press is an OS auto-repeat rather than a fresh down edge.
    pub repeat: bool,
}

/// One buffered GUI input event, ordered exactly as the window shell received it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScreenInputEvent {
    /// Keyboard press, including text-entry repeats.
    Key(KeyPress),
    /// Mouse button or wheel event.
    Mouse(TimedMouseEvent),
}

impl ScreenInputEvent {
    /// Shared shell receipt sequence.
    #[must_use]
    pub const fn sequence(self) -> u64 {
        match self {
            Self::Key(event) => event.sequence,
            Self::Mouse(event) => event.sequence,
        }
    }
}

/// Merge independently buffered device queues back into the shell's exact receipt order.
#[must_use]
pub fn merge_screen_events(
    keys: Vec<KeyPress>,
    mouse: Vec<TimedMouseEvent>,
) -> Vec<ScreenInputEvent> {
    let mut events = keys
        .into_iter()
        .map(ScreenInputEvent::Key)
        .chain(mouse.into_iter().map(ScreenInputEvent::Mouse))
        .collect::<Vec<_>>();
    events.sort_by_key(|event| event.sequence());
    events
}

/// Keyboard **edges** and modifier state, fed by [`mc_render::ShellApp::on_key_event`].
///
/// # What this replaced
///
/// `ShellApp` used to expose only polled key *state*: `InputState::key_down` answers "is this
/// key held **now**", never "did it just change". Escape, F3 and every character typed into a
/// text field are not rebindable actions — [`KeyBindings::take_pressed`] is a queue only for
/// those — so the previous version of this type recovered edges itself, by diffing that held
/// state once per frame against ~130 watched physical keys, and it guessed the typed character
/// from a hard-coded US-QWERTY table because winit's own `KeyEvent::text` never reached this
/// crate at all.
///
/// That had two honest, undocumented-in-vanilla costs: a key pressed *and released* inside one
/// frame was missed entirely (a real gap at the 250+ FPS this client runs at, since a poll only
/// sees the two ends of that frame, never what happened between them), and every non-US layout
/// typed the US character into chat regardless of the user's real keyboard.
///
/// # Now
///
/// [`RawKeyboard::push`] is called once per raw event, in event order, from
/// [`mc_render::ShellApp::on_key_event`] — there is no frame-boundary poll left to lose an edge
/// inside, because every edge arrives the instant it happens — and winit's `KeyEvent::text` is
/// threaded straight through instead of guessed. [`RawKeyboard::drain_presses`] is what
/// `handle_screen_input` calls once per frame to collect everything queued since the last call.
#[derive(Debug, Clone, Default)]
pub struct RawKeyboard {
    /// Presses queued since the last `drain_presses`, oldest first.
    queued: Vec<KeyPress>,
    /// The half of each modifier pair (`ControlLeft`/`ControlRight`, etc.) currently held. Only
    /// these eight keys need live tracking here: every other key's *edge* goes straight to
    /// `queued` instead of being reconstructed from held state, and nothing else reads held
    /// state through this type. The Super pair supplies vanilla's macOS Command-as-Control rule.
    held_mods: HashSet<KeyCode>,
    /// Deterministic sequence source for direct/headless callers. The live shell supplies its
    /// own counter shared with mouse ingress through [`Self::push_with_sequence`].
    next_sequence: u64,
}

impl RawKeyboard {
    /// A tracker with nothing held and nothing queued.
    #[must_use]
    pub fn new() -> RawKeyboard {
        RawKeyboard::default()
    }

    /// Modifier state as of the most recent event. java: `GuiScreen.isCtrlKeyDown`/`isShiftKeyDown`.
    #[must_use]
    pub fn modifiers(&self) -> Modifiers {
        self.modifiers_for_platform(cfg!(target_os = "macos"))
    }

    /// The platform argument keeps vanilla's OS branch directly testable on every build host.
    fn modifiers_for_platform(&self, macos: bool) -> Modifiers {
        Modifiers {
            ctrl: if macos {
                self.held_mods.contains(&KeyCode::SuperLeft)
                    || self.held_mods.contains(&KeyCode::SuperRight)
            } else {
                self.held_mods.contains(&KeyCode::ControlLeft)
                    || self.held_mods.contains(&KeyCode::ControlRight)
            },
            shift: self.held_mods.contains(&KeyCode::ShiftLeft)
                || self.held_mods.contains(&KeyCode::ShiftRight),
            alt: self.held_mods.contains(&KeyCode::AltLeft)
                || self.held_mods.contains(&KeyCode::AltRight),
        }
    }

    /// Forget every held modifier and drop anything queued but not yet drained.
    ///
    /// java: `KeyBinding.unPressAllKeys` on focus loss. Also called when a screen opens, so a
    /// key physically still held across the transition cannot deliver a stale edge to whatever
    /// just opened.
    pub fn clear(&mut self) {
        self.held_mods.clear();
        self.queued.clear();
    }

    /// Record one raw keyboard event. Called from [`mc_render::ShellApp::on_key_event`].
    ///
    /// Every event updates the modifier state — including releases and repeats — so
    /// [`RawKeyboard::modifiers`] is always current. Every **press** that maps to an LWJGL
    /// scancode is queued, including OS auto-repeat; the screen dispatcher admits repeats only
    /// for screens where vanilla enables LWJGL repeat events. Nothing in the GUI layer acts on a
    /// key-up, exactly as `GuiScreen.keyTyped` is only called for `getEventKeyState()`.
    ///
    /// `typed` should be winit's `KeyEvent::text`, narrowed to its first `char`. `None` falls
    /// back to [`char_for`]'s US-QWERTY table.
    pub fn push(&mut self, code: KeyCode, pressed: bool, repeat: bool, typed: Option<char>) {
        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.wrapping_add(1);
        self.push_with_sequence(code, pressed, repeat, typed, sequence);
    }

    /// Record one raw keyboard event with the shell's cross-device receipt sequence.
    pub fn push_with_sequence(
        &mut self,
        code: KeyCode,
        pressed: bool,
        repeat: bool,
        typed: Option<char>,
        sequence: u64,
    ) {
        self.push_with_sequence_for_platform(
            code,
            pressed,
            repeat,
            typed,
            sequence,
            cfg!(target_os = "macos"),
        );
    }

    fn push_with_sequence_for_platform(
        &mut self,
        code: KeyCode,
        pressed: bool,
        repeat: bool,
        typed: Option<char>,
        sequence: u64,
        macos: bool,
    ) {
        if matches!(
            code,
            KeyCode::ControlLeft
                | KeyCode::ControlRight
                | KeyCode::ShiftLeft
                | KeyCode::ShiftRight
                | KeyCode::AltLeft
                | KeyCode::AltRight
                | KeyCode::SuperLeft
                | KeyCode::SuperRight
        ) {
            if pressed {
                self.held_mods.insert(code);
            } else {
                self.held_mods.remove(&code);
            }
        }
        if !pressed {
            return;
        }
        let Some(lwjgl) = lwjgl_of_keycode(code) else {
            return;
        };
        let mods = self.modifiers_for_platform(macos);
        self.queued.push(KeyPress {
            sequence,
            code,
            lwjgl,
            typed: typed.or_else(|| char_for(code, mods.shift)),
            mods,
            repeat,
        });
    }

    /// Queue a deterministic platform-specific event for cross-platform integration tests.
    #[cfg(test)]
    pub(crate) fn push_for_platform_for_test(
        &mut self,
        code: KeyCode,
        pressed: bool,
        typed: Option<char>,
        macos: bool,
    ) {
        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.wrapping_add(1);
        self.push_with_sequence_for_platform(code, pressed, false, typed, sequence, macos);
    }

    /// Take every press queued since the last call, oldest first.
    pub fn drain_presses(&mut self) -> Vec<KeyPress> {
        std::mem::take(&mut self.queued)
    }

    /// Queue `presses`, taken earlier by [`Self::drain_presses`], again ahead of anything queued
    /// since, oldest first, and leave `presses` empty. The held modifiers stay as they are; each
    /// press keeps the snapshot it was queued with. A 1.7.10 session takes a frame's presses at
    /// the frame and hands them back at its tick (`app/send_sites_v5.rs`).
    pub(crate) fn requeue_presses(&mut self, presses: &mut Vec<KeyPress>) {
        self.queued.splice(0..0, presses.drain(..));
    }
}

/// The character a physical key produces on a **US QWERTY** layout — the *fallback*
/// [`RawKeyboard::push`] uses only when winit reports no `KeyEvent::text` for a press.
///
/// java: `Keyboard.getEventCharacter()`, which LWJGL takes from the OS and which therefore
/// honours the user's real layout. winit's `KeyEvent::text` is the same kind of value and is now
/// the primary source, via [`mc_render::ShellApp::on_key_event`] — this table only stands in for
/// the rare press winit resolves no text for. **Falling back to this table is still a
/// deviation**: on an AZERTY or Dvorak keyboard, a key winit gave no text for would type the US
/// character for that physical position rather than nothing. Movement is unaffected — WASD is
/// bound on physical keys in 1.8.9 too (see S10's `input.rs`).
#[must_use]
pub fn char_for(code: KeyCode, shift: bool) -> Option<char> {
    let (lower, upper) = match code {
        KeyCode::KeyA => ('a', 'A'),
        KeyCode::KeyB => ('b', 'B'),
        KeyCode::KeyC => ('c', 'C'),
        KeyCode::KeyD => ('d', 'D'),
        KeyCode::KeyE => ('e', 'E'),
        KeyCode::KeyF => ('f', 'F'),
        KeyCode::KeyG => ('g', 'G'),
        KeyCode::KeyH => ('h', 'H'),
        KeyCode::KeyI => ('i', 'I'),
        KeyCode::KeyJ => ('j', 'J'),
        KeyCode::KeyK => ('k', 'K'),
        KeyCode::KeyL => ('l', 'L'),
        KeyCode::KeyM => ('m', 'M'),
        KeyCode::KeyN => ('n', 'N'),
        KeyCode::KeyO => ('o', 'O'),
        KeyCode::KeyP => ('p', 'P'),
        KeyCode::KeyQ => ('q', 'Q'),
        KeyCode::KeyR => ('r', 'R'),
        KeyCode::KeyS => ('s', 'S'),
        KeyCode::KeyT => ('t', 'T'),
        KeyCode::KeyU => ('u', 'U'),
        KeyCode::KeyV => ('v', 'V'),
        KeyCode::KeyW => ('w', 'W'),
        KeyCode::KeyX => ('x', 'X'),
        KeyCode::KeyY => ('y', 'Y'),
        KeyCode::KeyZ => ('z', 'Z'),
        KeyCode::Digit0 => ('0', ')'),
        KeyCode::Digit1 => ('1', '!'),
        KeyCode::Digit2 => ('2', '@'),
        KeyCode::Digit3 => ('3', '#'),
        KeyCode::Digit4 => ('4', '$'),
        KeyCode::Digit5 => ('5', '%'),
        KeyCode::Digit6 => ('6', '^'),
        KeyCode::Digit7 => ('7', '&'),
        KeyCode::Digit8 => ('8', '*'),
        KeyCode::Digit9 => ('9', '('),
        KeyCode::Space => (' ', ' '),
        KeyCode::Minus => ('-', '_'),
        KeyCode::Equal => ('=', '+'),
        KeyCode::BracketLeft => ('[', '{'),
        KeyCode::BracketRight => (']', '}'),
        KeyCode::Backslash => ('\\', '|'),
        KeyCode::Semicolon => (';', ':'),
        KeyCode::Quote => ('\'', '"'),
        KeyCode::Backquote => ('`', '~'),
        KeyCode::Comma => (',', '<'),
        KeyCode::Period => ('.', '>'),
        KeyCode::Slash => ('/', '?'),
        // The numpad ignores Shift for the digits, as it does in LWJGL.
        KeyCode::Numpad0 => ('0', '0'),
        KeyCode::Numpad1 => ('1', '1'),
        KeyCode::Numpad2 => ('2', '2'),
        KeyCode::Numpad3 => ('3', '3'),
        KeyCode::Numpad4 => ('4', '4'),
        KeyCode::Numpad5 => ('5', '5'),
        KeyCode::Numpad6 => ('6', '6'),
        KeyCode::Numpad7 => ('7', '7'),
        KeyCode::Numpad8 => ('8', '8'),
        KeyCode::Numpad9 => ('9', '9'),
        KeyCode::NumpadDecimal => ('.', '.'),
        KeyCode::NumpadAdd => ('+', '+'),
        KeyCode::NumpadSubtract => ('-', '-'),
        KeyCode::NumpadMultiply => ('*', '*'),
        KeyCode::NumpadDivide => ('/', '/'),
        _ => return None,
    };
    Some(if shift { upper } else { lower })
}

/// A mouse button going down or coming up, in GUI pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseEvent {
    /// java: `GuiScreen.mouseClicked`
    Press(i32),
    /// java: `GuiScreen.mouseReleased`
    Release(i32),
}

/// Mouse-button **edges**, recovered by polling — [`ShellApp`](mc_render::ShellApp) has no mouse
/// event hook the way it now has [`ShellApp::on_key_event`](mc_render::ShellApp::on_key_event)
/// for the keyboard, so this still diffs [`InputState`]'s held button state once per frame the
/// way [`RawKeyboard`] used to.
///
/// Only the three buttons vanilla names (`0` left, `1` right, `2` middle) are tracked, because
/// `KeyBinding`'s `button - 100` encoding is what the options screen writes and it never offers
/// more.
#[derive(Debug, Clone, Default)]
pub struct MouseTracker {
    down: [bool; 3],
}

impl MouseTracker {
    /// A tracker with nothing held.
    #[must_use]
    pub fn new() -> MouseTracker {
        MouseTracker::default()
    }

    /// Whether a button is held as of the last poll.
    #[must_use]
    pub fn is_down(&self, button: i32) -> bool {
        usize::try_from(button)
            .ok()
            .and_then(|i| self.down.get(i))
            .copied()
            .unwrap_or(false)
    }

    /// Forget every held button, e.g. when a screen opens.
    pub fn clear(&mut self) {
        self.down = [false; 3];
    }

    /// Diff against `input` and report the edges, presses before releases per button index.
    pub fn poll(&mut self, input: &InputState) -> Vec<MouseEvent> {
        let mut events = Vec::new();
        for button in 0..3usize {
            let now = input.button_down(button as u16);
            if now != self.down[button] {
                self.down[button] = now;
                events.push(if now {
                    MouseEvent::Press(button as i32)
                } else {
                    MouseEvent::Release(button as i32)
                });
            }
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn press(input: &mut InputState, code: KeyCode) {
        input.on_key(code, true, false);
    }

    fn press_binding(bindings: &mut KeyBindings, input: Input) {
        bindings.set_state(input, true);
    }

    #[test]
    fn the_wasd_keys_produce_vanillas_movement_intent() {
        let mut bindings = KeyBindings::vanilla();
        assert_eq!(
            move_input(&bindings, 0.0, 0.0, false),
            MoveInput::default(),
            "nothing held is the default"
        );

        press_binding(&mut bindings, Input::Key(KeyCode::KeyW));
        press_binding(&mut bindings, Input::Key(KeyCode::KeyA));
        press_binding(&mut bindings, Input::Key(KeyCode::Space));
        let moved = move_input(&bindings, 90.0, -12.5, false);
        assert!(moved.forward && moved.left && moved.jump);
        assert!(!moved.back && !moved.right && !moved.sneak);
        assert_eq!((moved.yaw, moved.pitch), (90.0, -12.5));
        // S13's own helpers must agree with the flags, since they are what the physics reads.
        assert_eq!(moved.move_forward(), 1.0);
        assert_eq!(moved.move_strafe(), 1.0);
    }

    /// java: `EntityPlayerSP.onLivingUpdate` reads `keyBindSprint.isKeyDown()` directly in both
    /// sprint branches. Forward/sneak are separate conditions at those call sites, so this input
    /// field must remain the live binding state under every combination.
    #[test]
    fn sprint_flag_is_the_live_binding_state_without_preconditions() {
        let mut bindings = KeyBindings::vanilla();

        press_binding(&mut bindings, Input::Key(KeyCode::ControlLeft));
        assert!(
            move_input(&bindings, 0.0, 0.0, false).sprint,
            "the live sprint binding is held even before the downstream forward gate"
        );

        press_binding(&mut bindings, Input::Key(KeyCode::KeyW));
        assert!(move_input(&bindings, 0.0, 0.0, false).sprint);

        press_binding(&mut bindings, Input::Key(KeyCode::ShiftLeft));
        let sneaking = move_input(&bindings, 0.0, 0.0, false);
        assert!(sneaking.sneak);
        assert!(
            sneaking.sprint,
            "sneaking is a downstream sprint gate, not part of the binding state"
        );

        bindings.set_state(Input::Key(KeyCode::ControlLeft), false);
        assert!(
            !move_input(&bindings, 0.0, 0.0, false).sprint,
            "releasing only the sprint key clears the raw state"
        );
    }

    #[test]
    fn attack_and_use_read_the_mouse_buttons() {
        let mut bindings = KeyBindings::vanilla();
        assert!(!attacking(&bindings));
        assert!(!using_item(&bindings));

        press_binding(&mut bindings, Input::MOUSE_LEFT);
        assert!(attacking(&bindings));
        assert!(!using_item(&bindings));

        press_binding(&mut bindings, Input::MOUSE_RIGHT);
        assert!(using_item(&bindings));
    }

    /// `unPressAllKeys` clears `KeyBinding.pressed`, not the operating system's physical-key
    /// state.  A raw W/Control/mouse button still held across a screen transition must therefore
    /// be inert until it produces a fresh binding edge after focus returns.
    #[test]
    fn unpress_all_beats_still_held_raw_movement_sprint_and_mouse_inputs() {
        let mut bindings = KeyBindings::vanilla();
        let mut raw = InputState::new();
        press(&mut raw, KeyCode::KeyW);
        press(&mut raw, KeyCode::ControlLeft);
        raw.on_mouse_button(winit::event::MouseButton::Left, true);
        raw.on_mouse_button(winit::event::MouseButton::Right, true);

        assert!(raw.is_down(Input::Key(KeyCode::KeyW)));
        assert!(raw.is_down(Input::Key(KeyCode::ControlLeft)));
        assert!(raw.is_down(Input::MOUSE_LEFT));
        assert!(raw.is_down(Input::MOUSE_RIGHT));
        assert!(
            !move_input(&bindings, 0.0, 0.0, false).forward
                && !move_input(&bindings, 0.0, 0.0, false).sprint
                && !attacking(&bindings)
                && !using_item(&bindings),
            "negative control: physical state alone must not bypass KeyBinding.pressed"
        );

        for input in [
            Input::Key(KeyCode::KeyW),
            Input::Key(KeyCode::ControlLeft),
            Input::MOUSE_LEFT,
            Input::MOUSE_RIGHT,
        ] {
            press_binding(&mut bindings, input);
        }
        assert!(move_input(&bindings, 0.0, 0.0, false).forward);
        assert!(move_input(&bindings, 0.0, 0.0, false).sprint);
        assert!(attacking(&bindings) && using_item(&bindings));

        bindings.unpress_all();
        assert!(raw.is_down(Input::Key(KeyCode::KeyW)) && raw.is_down(Input::MOUSE_LEFT));
        assert_eq!(move_input(&bindings, 0.0, 0.0, false), MoveInput::default());
        assert!(!attacking(&bindings) && !using_item(&bindings));
    }

    /// The binding names are 1-based (`key.hotbar.1`) and the slot index is 0-based; getting
    /// that conversion wrong silently shifts every hotbar key by one and loses slot 9 entirely.
    #[test]
    fn the_number_keys_pick_a_zero_based_hotbar_slot() {
        let mut bindings = KeyBindings::vanilla();
        assert_eq!(hotbar_selection(&mut bindings), None);

        for (key, expected) in [
            (KeyCode::Digit1, 0u8),
            (KeyCode::Digit4, 3),
            (KeyCode::Digit9, 8),
        ] {
            press_binding(&mut bindings, Input::Key(key));
            assert_eq!(hotbar_selection(&mut bindings), Some(expected), "{key:?}");
        }

        // Vanilla consumes every queued slot press in its ascending loop, so the highest pressed
        // key is the final assignment.  The lower press must be consumed too.
        press_binding(&mut bindings, Input::Key(KeyCode::Digit7));
        press_binding(&mut bindings, Input::Key(KeyCode::Digit2));
        assert_eq!(hotbar_selection(&mut bindings), Some(6));
        assert_eq!(hotbar_selection(&mut bindings), None);
    }

    #[test]
    fn unpress_all_discards_a_hotbar_press_while_the_raw_number_key_stays_held() {
        let mut bindings = KeyBindings::vanilla();
        let mut raw = InputState::new();
        press(&mut raw, KeyCode::Digit9);
        press_binding(&mut bindings, Input::Key(KeyCode::Digit9));
        bindings.unpress_all();

        assert!(raw.is_down(Input::Key(KeyCode::Digit9)));
        assert_eq!(hotbar_selection(&mut bindings), None);
    }

    #[test]
    fn container_binding_snapshot_separates_keyboard_and_mouse_pick_inputs() {
        let mut bindings = KeyBindings::vanilla();
        bindings
            .get_mut(keys::INVENTORY)
            .unwrap()
            .set_binding(Input::Key(KeyCode::KeyR));
        bindings
            .get_mut(keys::DROP)
            .unwrap()
            .set_binding(Input::Key(KeyCode::KeyG));
        bindings
            .get_mut(&keys::hotbar(3))
            .unwrap()
            .set_binding(Input::Key(KeyCode::KeyZ));
        bindings
            .get_mut(keys::PICK_ITEM)
            .unwrap()
            .set_binding(Input::Mouse(4));

        let snapshot = container_bindings(&bindings);
        assert_eq!(
            snapshot.inventory_key,
            lwjgl_of_keycode(KeyCode::KeyR).unwrap()
        );
        assert_eq!(snapshot.drop_key, lwjgl_of_keycode(KeyCode::KeyG).unwrap());
        assert_eq!(
            snapshot.hotbar_keys[2],
            lwjgl_of_keycode(KeyCode::KeyZ).unwrap()
        );
        assert_eq!(snapshot.pick_block_key, -96);
        assert_eq!(snapshot.pick_block_button, 4);

        bindings
            .get_mut(keys::PICK_ITEM)
            .unwrap()
            .set_binding(Input::Key(KeyCode::KeyP));
        let keyboard_pick = container_bindings(&bindings);
        assert_eq!(
            keyboard_pick.pick_block_key,
            lwjgl_of_keycode(KeyCode::KeyP).unwrap()
        );
        assert_eq!(keyboard_pick.pick_block_button, -1);
    }

    /// java: scrolling **up** moves *down* the hotbar index, and both ends wrap.
    #[test]
    fn scrolling_the_hotbar_is_inverted_and_wraps() {
        assert_eq!(scroll_hotbar(4, 0.0), 4, "no scroll changes nothing");
        assert_eq!(scroll_hotbar(4, 1.0), 3);
        assert_eq!(scroll_hotbar(4, -1.0), 5);
        assert_eq!(scroll_hotbar(0, 1.0), 8, "scrolling up from 0 wraps to 8");
        assert_eq!(
            scroll_hotbar(8, -1.0),
            0,
            "scrolling down from 8 wraps to 0"
        );
        // A big wheel delta is still one step, as vanilla clamps `getEventDWheel` to a sign.
        assert_eq!(scroll_hotbar(4, 120.0), 3);
    }

    // ------------------------------------------------------------------ raw edges

    #[test]
    fn the_raw_keyboard_reports_a_press_exactly_once() {
        let mut raw = RawKeyboard::new();
        assert!(raw.drain_presses().is_empty());

        raw.push(KeyCode::Escape, true, false, None);
        let events = raw.drain_presses();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].code, KeyCode::Escape);
        assert_eq!(
            events[0].lwjgl,
            mc_ui::widget::keys::ESCAPE,
            "the screens take LWJGL scancodes"
        );
        assert!(
            raw.drain_presses().is_empty(),
            "a press is only ever queued once, and draining empties the queue"
        );

        raw.push(KeyCode::Escape, false, false, None);
        assert!(raw.drain_presses().is_empty(), "releases are not queued");
        raw.push(KeyCode::Escape, true, false, None);
        assert_eq!(raw.drain_presses().len(), 1, "re-pressing queues again");
    }

    /// OS auto-repeat remains distinguishable until the live UI decides whether its current
    /// screen enabled LWJGL repeat events. Gameplay bindings still discard it in `InputState`.
    #[test]
    fn an_os_auto_repeat_is_queued_with_its_repeat_bit() {
        let mut raw = RawKeyboard::new();
        raw.push(KeyCode::KeyA, true, false, Some('a'));
        raw.drain_presses();
        raw.push(KeyCode::KeyA, true, true, Some('a'));
        let repeated = raw.drain_presses();
        assert_eq!(repeated.len(), 1);
        assert!(repeated[0].repeat);
        assert_eq!(repeated[0].typed, Some('a'));
    }

    #[test]
    fn independently_buffered_devices_merge_by_shell_receipt_sequence() {
        use mc_render::{MouseEventKind, TimedMouseEvent};

        let mut raw = RawKeyboard::new();
        raw.push_with_sequence(KeyCode::KeyA, true, false, Some('a'), 1);
        let mouse = vec![
            TimedMouseEvent {
                kind: MouseEventKind::Button {
                    button: 0,
                    pressed: true,
                },
                at_ms: 10,
                sequence: 0,
            },
            TimedMouseEvent {
                kind: MouseEventKind::Wheel { delta: 1.0 },
                at_ms: 11,
                sequence: 2,
            },
        ];

        let merged = merge_screen_events(raw.drain_presses(), mouse);
        assert_eq!(
            merged
                .iter()
                .copied()
                .map(ScreenInputEvent::sequence)
                .collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert!(matches!(merged[0], ScreenInputEvent::Mouse(_)));
        assert!(matches!(merged[1], ScreenInputEvent::Key(_)));
        assert!(matches!(merged[2], ScreenInputEvent::Mouse(_)));
    }

    /// The problem the old diff-per-frame design could not solve: a key pressed *and* released
    /// before the next poll was invisible, because polling only ever sees the two ends of a
    /// frame. Feeding events as they happen has no such gap to fall into.
    #[test]
    fn a_press_and_release_inside_one_frame_still_produces_the_press() {
        let mut raw = RawKeyboard::new();
        raw.push(KeyCode::KeyG, true, false, Some('g'));
        raw.push(KeyCode::KeyG, false, false, None);
        // Nothing has drained yet, so both events landed "inside one frame" from the caller's
        // point of view — and the press already made it into the queue regardless.
        let events = raw.drain_presses();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].typed, Some('g'));
    }

    /// java: `GuiScreen.isCtrlKeyDown` checks LWJGL 219/220 (Command) on macOS and 29/157
    /// (Control) everywhere else. Exercise both platform branches here so this remains covered
    /// even when the test host is not a Mac.
    #[test]
    fn ctrl_modifier_matches_vanillas_platform_branch_for_all_four_keys() {
        for code in [KeyCode::ControlLeft, KeyCode::ControlRight] {
            let mut raw = RawKeyboard::new();
            raw.push(code, true, false, None);
            assert!(raw.modifiers_for_platform(false).ctrl, "{code:?}");
            assert!(
                !raw.modifiers_for_platform(true).ctrl,
                "macOS must not treat {code:?} as GuiScreen control"
            );
            raw.push(code, false, false, None);
            assert!(!raw.modifiers_for_platform(false).ctrl, "{code:?} release");
        }

        for code in [KeyCode::SuperLeft, KeyCode::SuperRight] {
            let mut raw = RawKeyboard::new();
            raw.push(code, true, false, None);
            assert!(raw.modifiers_for_platform(true).ctrl, "{code:?}");
            assert!(
                !raw.modifiers_for_platform(false).ctrl,
                "non-macOS must not treat {code:?} as GuiScreen control"
            );
            raw.push(code, false, false, None);
            assert!(!raw.modifiers_for_platform(true).ctrl, "{code:?} release");
        }

        let mut control = RawKeyboard::new();
        control.push(KeyCode::ControlLeft, true, false, None);
        assert_eq!(control.modifiers().ctrl, !cfg!(target_os = "macos"));

        let mut command = RawKeyboard::new();
        command.push(KeyCode::SuperLeft, true, false, None);
        assert_eq!(command.modifiers().ctrl, cfg!(target_os = "macos"));
    }

    /// Every key vanilla could name (java: the range `Keyboard.getEventKey()` could return)
    /// round-trips through `push` to its LWJGL scancode — the same guarantee the old watch list
    /// gave, without needing one.
    #[test]
    fn every_key_vanilla_can_name_round_trips_through_push() {
        let mut raw = RawKeyboard::new();
        let mut count = 0;
        for lwjgl in 1..=223 {
            let Some(code) = mc_render::input::keycode_of_lwjgl(lwjgl) else {
                continue;
            };
            raw.push(code, true, false, None);
            let events = raw.drain_presses();
            assert_eq!(events.len(), 1, "{code:?}");
            assert_eq!(events[0].lwjgl, lwjgl);
            count += 1;
        }
        assert!(count > 100, "{count}");
    }

    /// Shift is read *after* it is folded into the modifier state, so pressing Shift and a
    /// letter "in the same frame" still types the capital — now because the two events are
    /// handled in the order they actually happened, rather than folded together and re-derived.
    #[test]
    fn modifiers_apply_to_a_key_pressed_in_the_same_frame() {
        let mut raw = RawKeyboard::new();
        raw.push(KeyCode::ShiftLeft, true, false, None);
        raw.push(KeyCode::KeyA, true, false, None);
        let events = raw.drain_presses();
        let a = events
            .iter()
            .find(|e| e.code == KeyCode::KeyA)
            .expect("A was pressed");
        assert_eq!(a.typed, Some('A'));
        assert!(a.mods.shift);
    }

    /// The fix for non-US layouts: winit's own text wins over the US-QWERTY guess whenever it
    /// has one, even when the guess would have produced a different character entirely.
    #[test]
    fn winits_own_text_overrides_the_us_layout_guess() {
        let mut raw = RawKeyboard::new();
        // `char_for` would resolve this physical key to '5' on a US layout. A French AZERTY
        // layout's OS-level text for the same physical key is '(' — the event's `typed` must win.
        raw.push(KeyCode::Digit5, true, false, Some('('));
        assert_eq!(raw.drain_presses()[0].typed, Some('('));
    }

    /// When winit reports no text — the case [`char_for`] exists for — the US-QWERTY table is
    /// still the fallback, and for keys it has no answer for either, `typed` stays `None`.
    #[test]
    fn no_winit_text_falls_back_to_the_us_layout_table() {
        let mut raw = RawKeyboard::new();
        raw.push(KeyCode::KeyZ, true, false, None);
        assert_eq!(raw.drain_presses()[0].typed, Some('z'));

        raw.push(KeyCode::F3, true, false, None);
        assert_eq!(raw.drain_presses()[0].typed, None);
    }

    /// `clear` drops both the modifier state and anything queued but not yet drained — e.g. a
    /// screen opening mid-frame must not deliver a stale press to whatever just opened.
    #[test]
    fn clear_forgets_modifiers_and_drops_the_queue() {
        let mut raw = RawKeyboard::new();
        raw.push(KeyCode::ShiftLeft, true, false, None);
        raw.push(KeyCode::KeyA, true, false, None);
        raw.clear();
        assert!(!raw.modifiers().shift);
        assert!(raw.drain_presses().is_empty());
    }

    #[test]
    fn the_us_layout_map_covers_what_chat_needs() {
        assert_eq!(char_for(KeyCode::KeyZ, false), Some('z'));
        assert_eq!(char_for(KeyCode::KeyZ, true), Some('Z'));
        assert_eq!(char_for(KeyCode::Digit1, false), Some('1'));
        assert_eq!(char_for(KeyCode::Digit1, true), Some('!'));
        assert_eq!(char_for(KeyCode::Slash, false), Some('/'));
        assert_eq!(char_for(KeyCode::Space, true), Some(' '));
        // The numpad ignores Shift, as LWJGL does.
        assert_eq!(char_for(KeyCode::Numpad7, true), Some('7'));
        // Keys that produce no character must stay `None`, or a text field would swallow them.
        assert_eq!(char_for(KeyCode::Escape, false), None);
        assert_eq!(char_for(KeyCode::Enter, false), None);
        assert_eq!(char_for(KeyCode::Backspace, false), None);
        assert_eq!(char_for(KeyCode::F3, false), None);
        assert_eq!(char_for(KeyCode::ArrowLeft, false), None);
        // Every character it does produce must be one chat accepts.
        for lwjgl in 1..=223 {
            let Some(code) = mc_render::input::keycode_of_lwjgl(lwjgl) else {
                continue;
            };
            for shift in [false, true] {
                if let Some(c) = char_for(code, shift) {
                    assert!(
                        mc_ui::widget::keys::is_allowed_character(c),
                        "{code:?} shift={shift} produced {c:?}, which chat rejects"
                    );
                }
            }
        }
    }

    #[test]
    fn the_mouse_tracker_reports_both_edges_once() {
        let mut mouse = MouseTracker::new();
        let mut input = InputState::new();
        assert!(mouse.poll(&input).is_empty());

        input.on_mouse_button(winit::event::MouseButton::Left, true);
        assert_eq!(mouse.poll(&input), [MouseEvent::Press(0)]);
        assert!(mouse.poll(&input).is_empty(), "held is not a new press");
        assert!(mouse.is_down(0));

        input.on_mouse_button(winit::event::MouseButton::Right, true);
        assert_eq!(mouse.poll(&input), [MouseEvent::Press(1)]);

        input.on_mouse_button(winit::event::MouseButton::Left, false);
        assert_eq!(mouse.poll(&input), [MouseEvent::Release(0)]);
        assert!(!mouse.is_down(0));
        assert!(mouse.is_down(1));

        mouse.clear();
        assert!(!mouse.is_down(1));
    }

    /// A rebound key must be honoured: the mapping goes through `KeyBindings`, not through
    /// hard-coded scancodes.
    #[test]
    fn rebinding_forward_moves_the_forward_input() {
        let mut bindings = KeyBindings::vanilla();
        assert!(bindings.set_binding(keys::FORWARD, mc_render::input::Input::Key(KeyCode::KeyI)));

        let mut input = InputState::new();
        press(&mut input, KeyCode::KeyW);
        assert!(
            !move_input(&bindings, 0.0, 0.0, false).forward,
            "the old key must stop working"
        );

        press_binding(&mut bindings, Input::Key(KeyCode::KeyI));
        assert!(move_input(&bindings, 0.0, 0.0, false).forward);
    }
}
