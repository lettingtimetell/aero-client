//! Keyboard, mouse, cursor grab, and the rebindable key-binding system.
//!
//! java: `client/settings/KeyBinding.java`, `MouseHelper.java`, and the input half of
//! `EntityRenderer.updateCameraAndRender`.
//!
//! Two layers live here:
//!
//! * [`InputState`] — raw device state for this frame: which physical keys and mouse buttons are
//!   down, accumulated raw mouse motion, scroll, focus, and cursor grab.
//! * [`KeyBindings`] — the vanilla layer on top: named, rebindable actions with conflict
//!   detection, stored in [`GameSettings`](crate::settings::GameSettings) and persisted to
//!   `options.txt` as `key_key.forward:17` lines.
//!
//! Bindings are keyed on *physical* keys ([`winit::keyboard::KeyCode`]), which is what vanilla's
//! LWJGL scancodes were: WASD stays where it is on an AZERTY keyboard, exactly as in 1.8.9.

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

use winit::event::MouseButton;
use winit::keyboard::KeyCode;

/// Something a [`KeyBinding`] can be bound to.
///
/// 1.8.9 stores both in a single `int`: mouse buttons are negative (`button - 100`, so left is
/// `-100`), `0` means unbound, and anything positive is an LWJGL 2 keyboard scancode. That
/// encoding is preserved on disk so `options.txt` stays interchangeable with vanilla's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Input {
    /// A physical keyboard key.
    Key(KeyCode),
    /// A mouse button, by index: 0 = left, 1 = right, 2 = middle.
    Mouse(u16),
    /// A vanilla/LWJGL code for which winit has no physical-key equivalent.
    ///
    /// Retaining this integer keeps `options.txt` round-trips lossless even though the binding
    /// cannot fire through winit.
    RawLwjgl(i32),
    /// Not bound to anything. java: key code `0`
    Unbound,
}

impl Input {
    /// Left mouse button — vanilla's default attack/mine binding (`-100`).
    pub const MOUSE_LEFT: Self = Self::Mouse(0);
    /// Right mouse button — vanilla's default use-item binding (`-99`).
    pub const MOUSE_RIGHT: Self = Self::Mouse(1);
    /// Middle mouse button — vanilla's default pick-block binding (`-98`).
    pub const MOUSE_MIDDLE: Self = Self::Mouse(2);

    /// Encode as the `int` vanilla writes to `options.txt`.
    #[must_use]
    pub fn to_lwjgl(self) -> i32 {
        match self {
            Self::Key(code) => lwjgl_of_keycode(code).unwrap_or(0),
            Self::Mouse(button) => i32::from(button) - 100,
            Self::RawLwjgl(code) => code,
            Self::Unbound => 0,
        }
    }

    /// Decode a vanilla `options.txt` key code. Unknown scancodes retain their raw integer so a
    /// newer or modded `options.txt` can be saved without silently erasing the binding.
    #[must_use]
    pub fn from_lwjgl(code: i32) -> Self {
        match code {
            0 => Self::Unbound,
            c if c < 0 => {
                let button = c + 100;
                u16::try_from(button).map_or(Self::RawLwjgl(c), Self::Mouse)
            }
            c => keycode_of_lwjgl(c).map_or(Self::RawLwjgl(c), Self::Key),
        }
    }

    /// Human-readable name for the controls screen (S16), using LWJGL's exact key names.
    #[must_use]
    pub fn display_name(self) -> String {
        match self {
            Self::Key(code) => {
                lwjgl_name_of_keycode(code).map_or_else(|| format!("{code:?}"), str::to_owned)
            }
            Self::Mouse(0) => "Button 1".into(),
            Self::Mouse(1) => "Button 2".into(),
            Self::Mouse(2) => "Button 3".into(),
            Self::Mouse(n) => format!("Button {}", n + 1),
            Self::RawLwjgl(code) if code < 0 => format!("Button {}", code + 101),
            Self::RawLwjgl(code) => lwjgl_name(code).unwrap_or("null").to_owned(),
            Self::Unbound => "NONE".into(),
        }
    }

    /// Whether this input is bound to anything. Unbound bindings never conflict.
    #[must_use]
    pub fn is_bound(self) -> bool {
        self != Self::Unbound
    }
}

impl From<MouseButton> for Input {
    fn from(b: MouseButton) -> Self {
        match b {
            MouseButton::Left => Self::Mouse(0),
            MouseButton::Right => Self::Mouse(1),
            MouseButton::Middle => Self::Mouse(2),
            MouseButton::Back => Self::Mouse(3),
            MouseButton::Forward => Self::Mouse(4),
            MouseButton::Other(n) => Self::Mouse(n),
        }
    }
}

impl From<KeyCode> for Input {
    fn from(k: KeyCode) -> Self {
        Self::Key(k)
    }
}

/// One rebindable action.
///
/// java: `client/settings/KeyBinding.java`. The static registry and `IntHashMap` from the Java
/// become [`KeyBindings`]; the press-counting semantics are preserved exactly, because GUI code
/// depends on [`KeyBinding::take_pressed`] consuming one queued press per call.
#[derive(Debug, Clone)]
pub struct KeyBinding {
    description: String,
    category: String,
    default: Input,
    bound: Input,
    pressed: bool,
    press_time: u32,
}

impl KeyBinding {
    /// A binding named `description` (a `key.*` translation key), defaulting to `default`,
    /// filed under `category` (a `key.categories.*` translation key).
    #[must_use]
    pub fn new(description: &str, default: Input, category: &str) -> Self {
        Self {
            description: description.to_owned(),
            category: category.to_owned(),
            default,
            bound: default,
            pressed: false,
            press_time: 0,
        }
    }

    /// Translation key identifying this action, e.g. `key.forward`. Also its `options.txt` key,
    /// written as `key_key.forward`.
    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Translation key of the controls-screen group, e.g. `key.categories.movement`.
    #[must_use]
    pub fn category(&self) -> &str {
        &self.category
    }

    /// What this action is currently bound to.
    #[must_use]
    pub fn binding(&self) -> Input {
        self.bound
    }

    /// What vanilla binds this action to out of the box.
    #[must_use]
    pub fn default_binding(&self) -> Input {
        self.default
    }

    /// Rebind without changing the held or queued state.
    ///
    /// java: `KeyBinding.setKeyCode` only assigns `keyCode`; the controls screen rebuilds the
    /// static lookup table separately and does not call `unpressKey` here.
    pub fn set_binding(&mut self, input: Input) {
        self.bound = input;
    }

    /// Restore the vanilla default.
    pub fn reset(&mut self) {
        self.set_binding(self.default);
    }

    /// Is the key held right now? For per-tick polling (movement).
    /// java: `KeyBinding.isKeyDown`
    #[must_use]
    pub fn is_down(&self) -> bool {
        self.pressed
    }

    /// Consume one queued press. Returns `true` once per physical press, so a single tap fires
    /// exactly one action even if several ticks elapse before it is read.
    /// java: `KeyBinding.isPressed`
    pub fn take_pressed(&mut self) -> bool {
        if self.press_time == 0 {
            false
        } else {
            self.press_time -= 1;
            true
        }
    }

    /// Drop held state and any queued presses. java: `KeyBinding.unpressKey`
    pub fn unpress(&mut self) {
        self.pressed = false;
        self.press_time = 0;
    }

    /// Queued presses not yet taken by [`Self::take_pressed`] (java: `pressTime`). The replay recorder writes them down each tick.
    #[must_use]
    pub fn pending_presses(&self) -> u32 {
        self.press_time
    }

    /// Set the held flag and the queued presses directly. Replay playback
    /// re-drives a tick's recorded bindings through the same input code.
    pub fn force_state(&mut self, held: bool, presses: u32) {
        self.pressed = held;
        self.press_time = presses;
    }
}

/// Translation keys of the vanilla bindings, so callers are not writing bare strings.
///
pub mod keys {
    /// Mine / attack. Default: left mouse button.
    pub const ATTACK: &str = "key.attack";
    /// Use item / place block. Default: right mouse button.
    pub const USE_ITEM: &str = "key.use";
    /// Walk forwards. Default: `W`.
    pub const FORWARD: &str = "key.forward";
    /// Strafe left. Default: `A`.
    pub const LEFT: &str = "key.left";
    /// Walk backwards. Default: `S`.
    pub const BACK: &str = "key.back";
    /// Strafe right. Default: `D`.
    pub const RIGHT: &str = "key.right";
    /// Jump. Default: space.
    pub const JUMP: &str = "key.jump";
    /// Sneak. Default: left shift.
    pub const SNEAK: &str = "key.sneak";
    /// Sprint. Default: left control.
    pub const SPRINT: &str = "key.sprint";
    /// Drop the held item. Default: `Q`.
    pub const DROP: &str = "key.drop";
    /// Open the inventory. Default: `E`.
    pub const INVENTORY: &str = "key.inventory";
    /// Open chat. Default: `T`.
    pub const CHAT: &str = "key.chat";
    /// Show the player list. Default: tab.
    pub const PLAYER_LIST: &str = "key.playerlist";
    /// Pick block. Default: middle mouse button.
    pub const PICK_ITEM: &str = "key.pickItem";
    /// Open chat with a leading slash. Default: `/`.
    pub const COMMAND: &str = "key.command";
    /// Take a screenshot. Default: `F2`.
    pub const SCREENSHOT: &str = "key.screenshot";
    /// Cycle first/third person. Default: `F5`.
    pub const TOGGLE_PERSPECTIVE: &str = "key.togglePerspective";
    /// Toggle cinematic camera. Unbound by default.
    pub const SMOOTH_CAMERA: &str = "key.smoothCamera";
    /// Toggle fullscreen. Default: `F11`.
    pub const FULLSCREEN: &str = "key.fullscreen";
    /// Spectator outlines. Unbound by default.
    pub const SPECTATOR_OUTLINES: &str = "key.spectatorOutlines";
    /// Start or stop the legacy Twitch stream. Default: `F6`.
    pub const STREAM_START_STOP: &str = "key.streamStartStop";
    /// Pause or resume the legacy Twitch stream. Default: `F7`.
    pub const STREAM_PAUSE_UNPAUSE: &str = "key.streamPauseUnpause";
    /// Run a legacy Twitch commercial. Unbound by default.
    pub const STREAM_COMMERCIAL: &str = "key.streamCommercial";
    /// Toggle the legacy Twitch microphone. Unbound by default.
    pub const STREAM_TOGGLE_MIC: &str = "key.streamToggleMic";
    /// Hotbar slot `n`, 1-based; `hotbar(1)` is `key.hotbar.1`.
    #[must_use]
    pub fn hotbar(slot: u8) -> String {
        format!("key.hotbar.{slot}")
    }

    /// [`hotbar`] without the allocation, for per-tick lookups: slots 1–9 are the nine names
    /// vanilla registers. Any other slot names no binding, so it maps to `""`, which no binding
    /// has either; a lookup misses exactly as it would with `hotbar(slot)`'s string.
    #[must_use]
    pub fn hotbar_static(slot: u8) -> &'static str {
        const NAMES: [&str; 9] = [
            "key.hotbar.1",
            "key.hotbar.2",
            "key.hotbar.3",
            "key.hotbar.4",
            "key.hotbar.5",
            "key.hotbar.6",
            "key.hotbar.7",
            "key.hotbar.8",
            "key.hotbar.9",
        ];
        slot.checked_sub(1)
            .and_then(|index| NAMES.get(usize::from(index)))
            .copied()
            .unwrap_or("")
    }
}

/// Construction order of the fields in `GameSettings.java:127-151`.
///
/// This is deliberately separate from `GameSettings.keyBindings` order: vanilla uses the former
/// when its collision-overwriting input hash is rebuilt, and the latter for display/persistence.
fn vanilla_registration_priority(description: &str) -> u8 {
    use keys as k;

    match description {
        k::FORWARD => 0,
        k::LEFT => 1,
        k::BACK => 2,
        k::RIGHT => 3,
        k::JUMP => 4,
        k::SNEAK => 5,
        k::SPRINT => 6,
        k::INVENTORY => 7,
        k::USE_ITEM => 8,
        k::DROP => 9,
        k::ATTACK => 10,
        k::PICK_ITEM => 11,
        k::CHAT => 12,
        k::PLAYER_LIST => 13,
        k::COMMAND => 14,
        k::SCREENSHOT => 15,
        k::TOGGLE_PERSPECTIVE => 16,
        k::SMOOTH_CAMERA => 17,
        k::FULLSCREEN => 18,
        k::SPECTATOR_OUTLINES => 19,
        k::STREAM_START_STOP => 20,
        k::STREAM_PAUSE_UNPAUSE => 21,
        k::STREAM_COMMERCIAL => 22,
        k::STREAM_TOGGLE_MIC => 23,
        _ => description
            .strip_prefix("key.hotbar.")
            .and_then(|slot| slot.parse::<u8>().ok())
            .filter(|slot| (1..=9).contains(slot))
            .map_or(0, |slot| 23 + slot),
    }
}

/// The full set of rebindable actions, with conflict detection.
///
/// java: the static `keybindArray` / `hash` / `keybindSet` in `KeyBinding.java`, made into a
/// value rather than global mutable state.
#[derive(Debug, Clone, Default)]
pub struct KeyBindings {
    list: Vec<KeyBinding>,
}

impl KeyBindings {
    /// The 33 vanilla bindings in `GameSettings.keyBindings` order.
    ///
    /// java: `GameSettings.keyBindings`
    #[must_use]
    pub fn vanilla() -> Self {
        use keys as k;
        const MOVE: &str = "key.categories.movement";
        const GAMEPLAY: &str = "key.categories.gameplay";
        const INV: &str = "key.categories.inventory";
        const MULTI: &str = "key.categories.multiplayer";
        const MISC: &str = "key.categories.misc";
        const STREAM: &str = "key.categories.stream";

        let mut list = vec![
            KeyBinding::new(k::ATTACK, Input::MOUSE_LEFT, GAMEPLAY),
            KeyBinding::new(k::USE_ITEM, Input::MOUSE_RIGHT, GAMEPLAY),
            KeyBinding::new(k::FORWARD, Input::Key(KeyCode::KeyW), MOVE),
            KeyBinding::new(k::LEFT, Input::Key(KeyCode::KeyA), MOVE),
            KeyBinding::new(k::BACK, Input::Key(KeyCode::KeyS), MOVE),
            KeyBinding::new(k::RIGHT, Input::Key(KeyCode::KeyD), MOVE),
            KeyBinding::new(k::JUMP, Input::Key(KeyCode::Space), MOVE),
            KeyBinding::new(k::SNEAK, Input::Key(KeyCode::ShiftLeft), MOVE),
            KeyBinding::new(k::SPRINT, Input::Key(KeyCode::ControlLeft), MOVE),
            KeyBinding::new(k::DROP, Input::Key(KeyCode::KeyQ), GAMEPLAY),
            KeyBinding::new(k::INVENTORY, Input::Key(KeyCode::KeyE), INV),
            KeyBinding::new(k::CHAT, Input::Key(KeyCode::KeyT), MULTI),
            KeyBinding::new(k::PLAYER_LIST, Input::Key(KeyCode::Tab), MULTI),
            KeyBinding::new(k::PICK_ITEM, Input::MOUSE_MIDDLE, GAMEPLAY),
            KeyBinding::new(k::COMMAND, Input::Key(KeyCode::Slash), MULTI),
            KeyBinding::new(k::SCREENSHOT, Input::Key(KeyCode::F2), MISC),
            KeyBinding::new(k::TOGGLE_PERSPECTIVE, Input::Key(KeyCode::F5), MISC),
            KeyBinding::new(k::SMOOTH_CAMERA, Input::Unbound, MISC),
            KeyBinding::new(k::STREAM_START_STOP, Input::Key(KeyCode::F6), STREAM),
            KeyBinding::new(k::STREAM_PAUSE_UNPAUSE, Input::Key(KeyCode::F7), STREAM),
            KeyBinding::new(k::STREAM_COMMERCIAL, Input::Unbound, STREAM),
            KeyBinding::new(k::STREAM_TOGGLE_MIC, Input::Unbound, STREAM),
            KeyBinding::new(k::FULLSCREEN, Input::Key(KeyCode::F11), MISC),
            KeyBinding::new(k::SPECTATOR_OUTLINES, Input::Unbound, MISC),
        ];

        const HOTBAR: [KeyCode; 9] = [
            KeyCode::Digit1,
            KeyCode::Digit2,
            KeyCode::Digit3,
            KeyCode::Digit4,
            KeyCode::Digit5,
            KeyCode::Digit6,
            KeyCode::Digit7,
            KeyCode::Digit8,
            KeyCode::Digit9,
        ];
        for (i, code) in HOTBAR.iter().enumerate() {
            list.push(KeyBinding::new(
                &keys::hotbar(i as u8 + 1),
                Input::Key(*code),
                INV,
            ));
        }

        Self { list }
    }

    /// Look a binding up by its translation key.
    #[must_use]
    pub fn get(&self, description: &str) -> Option<&KeyBinding> {
        self.list.iter().find(|b| b.description == description)
    }

    /// Mutable lookup by translation key.
    pub fn get_mut(&mut self, description: &str) -> Option<&mut KeyBinding> {
        self.list.iter_mut().find(|b| b.description == description)
    }

    /// Is this action's key held? `false` for an unknown action, so callers need no `unwrap`.
    #[must_use]
    pub fn is_down(&self, description: &str) -> bool {
        self.get(description).is_some_and(KeyBinding::is_down)
    }

    /// Consume one queued press for this action. `false` for an unknown action.
    pub fn take_pressed(&mut self, description: &str) -> bool {
        self.get_mut(description)
            .is_some_and(KeyBinding::take_pressed)
    }

    /// Every binding, in controls-screen order.
    pub fn iter(&self) -> std::slice::Iter<'_, KeyBinding> {
        self.list.iter()
    }

    /// Distinct categories. Java exposes this collection as a `HashSet`; the controls screen
    /// obtains its visible sorted headings from the separately sorted binding array.
    #[must_use]
    pub fn categories(&self) -> HashSet<&str> {
        self.list.iter().map(|b| b.category.as_str()).collect()
    }

    /// Rebind an action. Returns `false` if no such action exists.
    ///
    /// Vanilla permits creating a conflict and only paints it red; so do we. Query
    /// [`KeyBindings::is_conflicting`] after rebinding to decide what to draw.
    pub fn set_binding(&mut self, description: &str, input: Input) -> bool {
        match self.get_mut(description) {
            Some(b) => {
                b.set_binding(input);
                true
            }
            None => false,
        }
    }

    /// Restore every vanilla default. java: `GuiControls` "Reset Keys"
    pub fn reset_all(&mut self) {
        for b in &mut self.list {
            b.reset();
        }
    }

    /// Does this action share its key with another action?
    #[must_use]
    pub fn is_conflicting(&self, description: &str) -> bool {
        let Some(target) = self.get(description) else {
            return false;
        };
        if !target.bound.is_bound() {
            return false;
        }
        self.list.iter().filter(|b| b.bound == target.bound).count() > 1
    }

    /// Every group of two or more actions sharing one key, as translation keys.
    ///
    /// Unbound actions never appear: any number of actions may be unbound at once.
    #[must_use]
    pub fn conflicts(&self) -> Vec<Vec<&str>> {
        let mut by_input: HashMap<Input, Vec<&str>> = HashMap::new();
        for b in &self.list {
            if b.bound.is_bound() {
                by_input
                    .entry(b.bound)
                    .or_default()
                    .push(b.description.as_str());
            }
        }
        let mut groups: Vec<Vec<&str>> = by_input
            .into_values()
            .filter(|group| group.len() > 1)
            .collect();
        groups.sort();
        groups
    }

    /// Apply a raw press/release to every action bound to `input`.
    ///
    /// A press both marks the action held and queues one [`KeyBinding::take_pressed`]. Key-repeat
    /// events must be filtered out before calling this, or a held key would queue a press per
    /// repeat — vanilla only enables LWJGL repeat events inside text fields.
    ///
    /// java: `KeyBinding.setKeyBindState` + `KeyBinding.onTick`
    pub fn set_state(&mut self, input: Input, pressed: bool) {
        if !input.is_bound() {
            return;
        }
        // Vanilla dispatches through an IntHashMap that is rebuilt in construction order. Its
        // `addKey` overwrites on collision, so exactly the last-constructed binding receives the
        // edge. Construction order differs from `GameSettings.keyBindings` display/save order.
        if let Some(b) = self
            .list
            .iter_mut()
            .filter(|b| b.bound == input)
            .max_by_key(|b| vanilla_registration_priority(&b.description))
        {
            b.pressed = pressed;
            if pressed {
                b.press_time += 1;
            }
        }
    }

    /// Release everything and drop queued presses — on focus loss, or when a screen opens.
    /// Apply a raw key or mouse edge only while the current screen accepts game input.
    ///
    /// java: `KeyBinding.setKeyBindState` and `KeyBinding.onTick` in `Minecraft.runTick`, which
    /// see an event only while no screen is open: an open screen's `handleInput()` drains the
    /// LWJGL queues first, whatever its `allowUserInput`.
    ///
    /// Opening a screen must first call [`Self::unpress_all`]. Once it has, discarding *both*
    /// press and release edges while input is disallowed is intentional: a held physical key must
    /// not become a held game binding merely because the screen closes. The raw device state can
    /// still be maintained independently by [`InputState`].
    pub fn set_state_when_input_allowed(
        &mut self,
        input: Input,
        pressed: bool,
        input_allowed: bool,
    ) {
        if input_allowed {
            self.set_state(input, pressed);
        }
    }

    /// Release every held binding and discard queued presses when focus is lost or a screen opens.
    /// java: `KeyBinding.unPressAllKeys`
    pub fn unpress_all(&mut self) {
        for b in &mut self.list {
            b.unpress();
        }
    }

    /// Mark `description` held again when `is_down` says its bound input is
    /// physically down, so every key is re-read as a screen closes (Modern Keybinds). 1.8.9 has no such step: `unPressAllKeys` on open and
    /// nothing on close.
    ///
    /// Only the held flag is set, never a queued press, so [`Self::take_pressed`] (vanilla's
    /// `isPressed`) cannot fire for it. The input goes to the binding [`Self::set_state`] would
    /// give its edges, so two actions sharing one key behave as they do for a real press.
    /// Returns whether the binding was re-pressed.
    pub fn repress_if_held(&mut self, description: &str, is_down: impl Fn(Input) -> bool) -> bool {
        let Some(input) = self.get(description).map(KeyBinding::binding) else {
            return false;
        };
        if !input.is_bound() || !is_down(input) {
            return false;
        }
        let winner = self
            .list
            .iter_mut()
            .filter(|b| b.bound == input)
            .max_by_key(|b| vanilla_registration_priority(&b.description));
        match winner {
            Some(b) if b.description == description => {
                b.pressed = true;
                true
            }
            _ => false,
        }
    }
}

/// How the cursor behaves while the game has focus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CursorMode {
    /// Cursor visible and free — a screen is open. java: `Mouse.setGrabbed(false)`
    Free,
    /// Cursor hidden and locked to the window — in-game look control.
    /// java: `Mouse.setGrabbed(true)`
    Grabbed,
}

/// Where vanilla puts the pointer when it releases the mouse for a screen, in winit's physical
/// window coordinates (origin top-left).
///
/// java: `MouseHelper.ungrabMouseCursor` calls
/// `Mouse.setCursorPosition(Display.getWidth() / 2, Display.getHeight() / 2)`. That pair is in
/// LWJGL's bottom-left frame, and LWJGL 2.9.4's `WindowsDisplay.setCursorPosition` warps to
/// `(rect.left + x, rect.bottom - 1 - y)` (checked in its bytecode), so the row from the top is
/// `height - 1 - height / 2`: one row above `height / 2` when the height is even. Both divisions
/// truncate, as Java's do. A zero-sized (minimised) window maps to `(0, 0)`.
#[must_use]
pub fn cursor_release_point(width: u32, height: u32) -> (u32, u32) {
    (width / 2, height.saturating_sub(1) - height / 2)
}

/// Maximum event age kept by [`InputState::drain_mouse_events`]. The client's screen path uses
/// [`InputState::drain_all_mouse_events`] instead: vanilla's `GuiScreen.handleInput` has no age
/// gate.
pub const MOUSE_EVENT_MAX_AGE_MS: u64 = 200;

/// One lossless mouse input kind retained in receipt order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MouseEventKind {
    /// A numbered mouse-button edge. The bool is `true` for press.
    Button {
        /// Vanilla/LWJGL button index.
        button: u16,
        /// `true` for a press and `false` for a release.
        pressed: bool,
    },
    /// One wheel event in notch-equivalents; events are never aggregated in this queue.
    Wheel {
        /// Signed notch-equivalent delta for this single event.
        delta: f32,
    },
}

/// One mouse event timestamped when winit delivered it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimedMouseEvent {
    /// Event payload.
    pub kind: MouseEventKind,
    /// Milliseconds since this [`InputState`] was created.
    pub at_ms: u64,
    /// Monotonic receipt sequence, useful for biting order controls.
    pub sequence: u64,
}

/// Raw per-frame device state.
///
/// The shell feeds this from winit events; the game reads it once per tick. Mouse motion is
/// accumulated from `DeviceEvent::MouseMotion` — raw, unaccelerated, and unaffected by the
/// desktop pointer hitting a screen edge, which is what vanilla gets from LWJGL's grabbed mouse.
#[derive(Debug, Clone)]
pub struct InputState {
    keys: HashSet<KeyCode>,
    buttons: HashSet<u16>,
    /// Raw motion accumulated between frame reads.  Winit exposes this as `f64`, while
    /// LWJGL's `Mouse.getDX/getDY` expose the integer count for the whole interval since the
    /// previous read.  Keep the raw value until [`Self::take_mouse_delta`] so multiple fractional
    /// events in one frame are quantized together, rather than truncating each event separately.
    mouse_delta: (f64, f64),
    scroll: f32,
    mouse_events: VecDeque<TimedMouseEvent>,
    next_mouse_sequence: u64,
    mouse_epoch: Instant,
    cursor: (f64, f64),
    cursor_mode: Option<CursorMode>,
    focused: bool,
    /// The window's inner size in physical pixels as the shell last reported it; `(0, 0)` until
    /// it does. Only the release re-centre in [`Self::request_cursor`] reads it.
    window_size: (u32, u32),
    /// A focused grabbed → free transition happened and the shell has not warped the OS pointer
    /// to [`cursor_release_point`] yet. See [`Self::take_cursor_recentre`].
    cursor_recentre: bool,
    /// Raw Mouse Input is OFF: grabbed look reads OS pointer positions instead of
    /// raw device counts. See [`Self::set_os_pointer_look`].
    os_pointer_look: bool,
    /// The pointer position the next OS-pointer delta is measured from (LWJGL's
    /// `WindowsMouse.last_x/last_y`); `None` until the first position after a grab or a warp.
    pointer_anchor: Option<(f64, f64)>,
    /// A frame read non-zero OS-pointer motion, so the shell must warp the pointer back to the
    /// window centre (LWJGL's `WindowsMouse.poll` calling `centerCursor`).
    pointer_recentre: bool,
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            keys: HashSet::new(),
            buttons: HashSet::new(),
            mouse_delta: (0.0, 0.0),
            scroll: 0.0,
            mouse_events: VecDeque::new(),
            next_mouse_sequence: 0,
            mouse_epoch: Instant::now(),
            cursor: (0.0, 0.0),
            cursor_mode: None,
            focused: false,
            window_size: (0, 0),
            cursor_recentre: false,
            os_pointer_look: false,
            pointer_anchor: None,
            pointer_recentre: false,
        }
    }
}

impl InputState {
    /// Fresh, unfocused, ungrabbed state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Is this physical key held?
    #[must_use]
    pub fn key_down(&self, code: KeyCode) -> bool {
        self.keys.contains(&code)
    }

    /// Is this mouse button held? 0 = left, 1 = right, 2 = middle.
    #[must_use]
    pub fn button_down(&self, button: u16) -> bool {
        self.buttons.contains(&button)
    }

    /// Is this [`Input`] held? [`Input::Unbound`] is never held.
    #[must_use]
    pub fn is_down(&self, input: Input) -> bool {
        match input {
            Input::Key(c) => self.key_down(c),
            Input::Mouse(b) => self.button_down(b),
            Input::RawLwjgl(_) => false,
            Input::Unbound => false,
        }
    }

    /// Does the window have keyboard focus?
    #[must_use]
    pub fn focused(&self) -> bool {
        self.focused
    }

    /// Cursor position in physical pixels, for 2D hit-testing in screens.
    #[must_use]
    pub fn cursor(&self) -> (f64, f64) {
        self.cursor
    }

    /// Is look input currently enabled by the requested cursor mode?
    ///
    /// This is the mode requested from the window shell, not an acknowledgement that the
    /// backend accepted `set_cursor_grab`.  Winit reports that platform operation's failure at
    /// the window boundary; there is no truthful success/failure value for [`InputState`] to
    /// manufacture here.
    #[must_use]
    pub fn grabbed(&self) -> bool {
        self.cursor_mode == Some(CursorMode::Grabbed)
    }

    /// Request a cursor mode. The shell applies it to the window on the next event pump; the
    /// request is idempotent, so calling this every tick is fine. Entering [`CursorMode::Grabbed`]
    /// clears pending relative motion at the same boundary as Java's
    /// `MouseHelper.grabMouseCursor`, preventing a first grabbed frame from inheriting desktop
    /// motion. Leaving it [`CursorMode::Free`] deliberately preserves a pending delta: Java's
    /// `ungrabMouseCursor` recenters/releases the OS cursor but does not zero its public delta
    /// fields. The next grab clears it before look input can consume it.
    ///
    /// Leaving [`CursorMode::Grabbed`] for [`CursorMode::Free`] while the window has focus is
    /// vanilla's `ungrabMouseCursor` re-centre: [`Self::cursor`] moves to
    /// [`cursor_release_point`] at once, and the shell warps the OS pointer there when it
    /// releases the grab ([`Self::take_cursor_recentre`]). LWJGL's `Mouse.setCursorPosition`
    /// updates `Mouse.getX/getY` synchronously, so the new screen's first input pass and first
    /// draw read the centre; waiting for the OS to echo the warp back as a `CursorMoved` would
    /// hand them the pointer's last in-game position instead. Only a grabbed → free edge
    /// re-centres, as `setIngameNotInFocus` does only `if (this.inGameHasFocus)`: the first
    /// screen at startup and screen-to-screen changes keep the pointer where it is. An unfocused
    /// window (the Alt-Tab release in [`Self::on_focus`]) never moves the pointer, because it is
    /// in another application's hands by then.
    ///
    /// The OS warp is a separate best-effort window operation. If a backend rejects it, that
    /// failure cannot be observed or repaired by this raw-input layer; the shell must still
    /// release/show the cursor and report the platform limitation.
    pub fn request_cursor(&mut self, mode: CursorMode) {
        if self.cursor_mode != Some(mode) {
            let was_grabbed = self.cursor_mode == Some(CursorMode::Grabbed);
            self.cursor_mode = Some(mode);
            match mode {
                CursorMode::Grabbed => {
                    // Discard motion accumulated while ungrabbed, or the first grabbed frame
                    // snaps the view by however far the pointer travelled across the desktop.
                    self.mouse_delta = (0.0, 0.0);
                    self.cursor_recentre = false;
                }
                CursorMode::Free => {
                    if was_grabbed && self.focused {
                        self.cursor_recentre = true;
                        let (width, height) = self.window_size;
                        if width > 0 && height > 0 {
                            let (x, y) = cursor_release_point(width, height);
                            self.cursor = (f64::from(x), f64::from(y));
                        }
                    }
                }
            }
        }
    }

    /// Whether the shell must warp the OS pointer to [`cursor_release_point`] as it releases the
    /// grab, resetting the request. Set by a focused grabbed → free edge in
    /// [`Self::request_cursor`]; cleared by the next grab.
    pub fn take_cursor_recentre(&mut self) -> bool {
        std::mem::take(&mut self.cursor_recentre)
    }

    /// Record the window's inner size in physical pixels, as winit reports it. The shell calls
    /// this for every size it sees, so a release can re-centre [`Self::cursor`] without waiting
    /// for a frame.
    pub fn set_window_size(&mut self, width: u32, height: u32) {
        self.window_size = (width, height);
    }

    /// Choose the grabbed look source (Raw Mouse Input). `false`, the default and
    /// what mc-rs always did, accumulates `DeviceEvent::MouseMotion` (`WM_INPUT` device counts,
    /// no OS acceleration). `true` is vanilla LWJGL 2's Windows path instead: grabbed motion is
    /// the OS pointer's movement between `CursorMoved` positions, so Windows' pointer speed and
    /// "Enhance pointer precision" apply, and the shell warps the pointer back to the centre
    /// after every frame that read motion.
    ///
    /// Idempotent; the client calls it every frame. A change drops any pending delta and the
    /// anchor, so switching never jerks the view.
    pub fn set_os_pointer_look(&mut self, on: bool) {
        if self.os_pointer_look != on {
            self.os_pointer_look = on;
            self.mouse_delta = (0.0, 0.0);
            self.pointer_anchor = None;
            self.pointer_recentre = false;
        }
    }

    /// Whether grabbed look reads the OS pointer ([`Self::set_os_pointer_look`]).
    #[must_use]
    pub fn os_pointer_look(&self) -> bool {
        self.os_pointer_look
    }

    /// Whether the shell must warp the pointer back to the centre now, resetting the request.
    /// Set by [`Self::take_mouse_delta`] when the OS-pointer look path read motion.
    pub fn take_pointer_recentre(&mut self) -> bool {
        std::mem::take(&mut self.pointer_recentre)
    }

    /// The shell warped the pointer to `(x, y)` (physical pixels): measure the next OS-pointer
    /// delta from there, as LWJGL's `centerCursor` resets `last_x/last_y`.
    pub fn pointer_recentred(&mut self, x: f64, y: f64) {
        self.pointer_anchor = Some((x, y));
    }

    /// Currently requested cursor mode, if one has been requested.
    #[must_use]
    pub fn cursor_mode(&self) -> Option<CursorMode> {
        self.cursor_mode
    }

    /// Take the accumulated raw mouse motion, resetting it. Positive `y` is downwards.
    pub fn take_mouse_delta(&mut self) -> (i32, i32) {
        let (dx, dy) = std::mem::replace(&mut self.mouse_delta, (0.0, 0.0));
        if self.os_pointer_look && (dx != 0.0 || dy != 0.0) {
            // java (LWJGL 2): `WindowsMouse.poll` re-centres after a read with motion.
            self.pointer_recentre = true;
        }
        (truncated_mouse_count(dx), truncated_mouse_count(dy))
    }

    /// Take the accumulated scroll, resetting it. One notch is ±1.0.
    pub fn take_scroll(&mut self) -> f32 {
        std::mem::replace(&mut self.scroll, 0.0)
    }

    /// Drain fresh mouse events in exact receipt order.
    pub fn drain_mouse_events(&mut self) -> Vec<TimedMouseEvent> {
        let now_ms = self.mouse_epoch.elapsed().as_millis() as u64;
        self.drain_mouse_events_at_ms(now_ms)
    }

    /// Drain every queued mouse event in receipt order, whatever its
    /// age: the screen input path. java: `GuiScreen.handleInput` runs
    /// `while (Mouse.next()) handleMouseInput()` with no age gate. `Minecraft.runTick`'s only
    /// 200 ms test (`getSystemTime() - systemTime`, the time since the previous tick ended, not
    /// an event's age) guards its no-screen wheel/refocus branch, and an open screen has already
    /// drained the queue before that loop runs.
    pub fn drain_all_mouse_events(&mut self) -> Vec<TimedMouseEvent> {
        self.mouse_events.drain(..).collect()
    }

    /// Hand every queued **wheel** event's delta to `on_wheel`, oldest first, and remove them,
    /// leaving button events queued in their order for the screen path.
    ///
    /// java: with no screen open, `Minecraft.runTick`'s own `while (Mouse.next())` loop
    /// (`:1838-1881`) consumes the wheel at the start of the tick, after `updateController` and
    /// before the keyboard loop, so a notch that arrived before a frame's ticks lands in that
    /// frame's first tick. The client calls this from its tick (H3 N1). No
    /// allocation: `VecDeque::retain` visits in order and keeps the survivors' order.
    pub fn drain_wheel_events(&mut self, mut on_wheel: impl FnMut(f32)) {
        self.mouse_events.retain(|event| match event.kind {
            MouseEventKind::Wheel { delta } => {
                on_wheel(delta);
                false
            }
            MouseEventKind::Button { .. } => true,
        });
    }

    /// Deterministic-clock variant used by CPU controls.
    pub fn drain_mouse_events_at_ms(&mut self, now_ms: u64) -> Vec<TimedMouseEvent> {
        self.mouse_events
            .drain(..)
            .filter(|event| now_ms.saturating_sub(event.at_ms) <= MOUSE_EVENT_MAX_AGE_MS)
            .collect()
    }

    /// Convert accumulated integer mouse counts into the two `float` arguments vanilla passes to
    /// `Entity.setAngles`, applying the sensitivity curve and resetting the accumulator.
    ///
    /// The curve is cubic, not linear: `f = sensitivity * 0.6 + 0.2` then `f³ * 8`, and the
    /// returned f2/f3 deliberately do **not** include `Entity.setAngles`' `0.15D`. That operation
    /// must be applied while adding to the existing rotation in `f64`, with one final narrowing.
    ///
    /// java: `EntityRenderer.updateCameraAndRender:1097-1106`
    pub fn look_delta_raw(&mut self, sensitivity: f32, invert_y: bool) -> (f32, f32) {
        let (dx, dy) = self.take_mouse_delta();
        let f = sensitivity * 0.6 + 0.2;
        let scale = f * f * f * 8.0;
        let yaw = dx as f32 * scale;
        // LWJGL's deltaY points up and winit's points down, so vanilla's negation is already
        // applied by the coordinate system; inverting the setting flips it back.
        let pitch = dy as f32 * scale * if invert_y { -1.0 } else { 1.0 };
        (yaw, pitch)
    }

    /// Convert mouse input to a standalone degree delta.
    ///
    /// Player-camera callers needing exact `Entity.setAngles` accumulation should use
    /// [`InputState::look_delta_raw`] and apply `0.15D` while adding to the current rotation. This
    /// compatibility helper is appropriate only where no existing rotation participates.
    pub fn look_delta(&mut self, sensitivity: f32, invert_y: bool) -> (f32, f32) {
        let (yaw, pitch) = self.look_delta_raw(sensitivity, invert_y);
        // Entity.setAngles widens the f32 mouse argument to double before multiplying by its
        // `0.15D` literal. Folding 0.15 into the f32 sensitivity scale changes the low bit.
        (
            (f64::from(yaw) * 0.15) as f32,
            (f64::from(pitch) * 0.15) as f32,
        )
    }

    /// Record a key press or release. `repeat` events are ignored so that holding a key does not
    /// queue one [`KeyBinding::take_pressed`] per repeat.
    ///
    /// Returns the [`Input`] that changed state, or `None` for an ignored repeat, so the caller
    /// can forward it to [`KeyBindings::set_state`].
    pub fn on_key(&mut self, code: KeyCode, pressed: bool, repeat: bool) -> Option<Input> {
        if repeat {
            return None;
        }
        if pressed {
            self.keys.insert(code);
        } else {
            self.keys.remove(&code);
        }
        Some(Input::Key(code))
    }

    /// Record a mouse button press or release.
    pub fn on_mouse_button(&mut self, button: MouseButton, pressed: bool) -> Input {
        let at_ms = self.mouse_epoch.elapsed().as_millis() as u64;
        self.on_mouse_button_at_ms(button, pressed, at_ms)
    }

    /// Record a mouse button edge with the shell's cross-device receipt sequence.
    pub fn on_mouse_button_with_sequence(
        &mut self,
        button: MouseButton,
        pressed: bool,
        sequence: u64,
    ) -> Input {
        let at_ms = self.mouse_epoch.elapsed().as_millis() as u64;
        self.on_mouse_button_at_ms_with_sequence(button, pressed, at_ms, sequence)
    }

    /// Deterministic-clock mouse-button ingress used by CPU controls.
    pub fn on_mouse_button_at_ms(
        &mut self,
        button: MouseButton,
        pressed: bool,
        at_ms: u64,
    ) -> Input {
        let sequence = self.next_mouse_sequence;
        self.next_mouse_sequence = self.next_mouse_sequence.wrapping_add(1);
        self.on_mouse_button_at_ms_with_sequence(button, pressed, at_ms, sequence)
    }

    /// Deterministic-clock and externally sequenced mouse-button ingress.
    pub fn on_mouse_button_at_ms_with_sequence(
        &mut self,
        button: MouseButton,
        pressed: bool,
        at_ms: u64,
        sequence: u64,
    ) -> Input {
        let input = Input::from(button);
        if let Input::Mouse(index) = input {
            if pressed {
                self.buttons.insert(index);
            } else {
                self.buttons.remove(&index);
            }
            self.push_mouse_event_with_sequence(
                MouseEventKind::Button {
                    button: index,
                    pressed,
                },
                at_ms,
                sequence,
            );
        }
        input
    }

    /// Accumulate raw mouse motion. Ignored unless the cursor is grabbed, matching vanilla's
    /// `inGameHasFocus` gate — a screen-relative cursor drives [`InputState::cursor`] instead.
    pub fn on_raw_motion(&mut self, dx: f64, dy: f64) {
        // With Raw Mouse Input off, look comes from `on_cursor_moved` instead.
        if self.grabbed() && !self.os_pointer_look {
            // LWJGL's Mouse.getDX/getDY return the integer count accumulated since the previous
            // read.  Do not narrow each winit event independently: e.g. two +0.6 events must
            // produce one count when the frame reads the accumulator.
            self.mouse_delta.0 += dx;
            self.mouse_delta.1 += dy;
        }
    }

    /// Accumulate a scroll event. Pixel deltas are divided down to notch-equivalents.
    pub fn on_scroll(&mut self, delta: winit::event::MouseScrollDelta) {
        let at_ms = self.mouse_epoch.elapsed().as_millis() as u64;
        self.on_scroll_at_ms(delta, at_ms);
    }

    /// Accumulate one scroll event with the shell's cross-device receipt sequence.
    pub fn on_scroll_with_sequence(
        &mut self,
        delta: winit::event::MouseScrollDelta,
        sequence: u64,
    ) {
        let at_ms = self.mouse_epoch.elapsed().as_millis() as u64;
        self.on_scroll_at_ms_with_sequence(delta, at_ms, sequence);
    }

    /// Deterministic-clock wheel ingress used by CPU controls.
    pub fn on_scroll_at_ms(&mut self, delta: winit::event::MouseScrollDelta, at_ms: u64) {
        let sequence = self.next_mouse_sequence;
        self.next_mouse_sequence = self.next_mouse_sequence.wrapping_add(1);
        self.on_scroll_at_ms_with_sequence(delta, at_ms, sequence);
    }

    /// Deterministic-clock and externally sequenced wheel ingress.
    pub fn on_scroll_at_ms_with_sequence(
        &mut self,
        delta: winit::event::MouseScrollDelta,
        at_ms: u64,
        sequence: u64,
    ) {
        let delta = match delta {
            winit::event::MouseScrollDelta::LineDelta(_, y) => y,
            winit::event::MouseScrollDelta::PixelDelta(p) => (p.y / 120.0) as f32,
        };
        self.scroll += delta;
        self.push_mouse_event_with_sequence(MouseEventKind::Wheel { delta }, at_ms, sequence);
    }

    /// Record the cursor position, in physical pixels.
    ///
    /// While grabbed with Raw Mouse Input off ([`Self::set_os_pointer_look`]), the movement from
    /// the previous position is also the look delta, as LWJGL 2's `WindowsMouse.handleMouseMoved`
    /// accumulates `x - last_x`.
    pub fn on_cursor_moved(&mut self, x: f64, y: f64) {
        self.cursor = (x, y);
        if self.os_pointer_look && self.grabbed() {
            if let Some((last_x, last_y)) = self.pointer_anchor {
                self.mouse_delta.0 += x - last_x;
                self.mouse_delta.1 += y - last_y;
            }
            self.pointer_anchor = Some((x, y));
        } else {
            // A position from outside the grab is never a look delta's starting point.
            self.pointer_anchor = None;
        }
    }

    /// Record a focus change. Losing focus clears all held state — otherwise alt-tabbing while
    /// walking leaves the player sprinting forever.
    pub fn on_focus(&mut self, focused: bool) {
        self.focused = focused;
        if !focused {
            self.clear_held();
            self.request_cursor(CursorMode::Free);
        }
    }

    /// Release every key and button and discard accumulated motion.
    pub fn clear_held(&mut self) {
        self.keys.clear();
        self.buttons.clear();
        self.mouse_delta = (0.0, 0.0);
        self.scroll = 0.0;
        self.mouse_events.clear();
        self.pointer_anchor = None;
    }

    fn push_mouse_event_with_sequence(&mut self, kind: MouseEventKind, at_ms: u64, sequence: u64) {
        self.mouse_events.push_back(TimedMouseEvent {
            kind,
            at_ms,
            sequence,
        });
    }
}

/// Convert one frame's accumulated raw motion to the signed integer count exposed by LWJGL.
///
/// The normal path is the exact Java-visible operation: discard the fractional part toward zero.
/// Finite values outside the signed 32-bit range are reduced modulo `2^32` to retain the wrapping
/// behavior of the old integer accumulator at the synthetic overflow boundary. NaN contributes
/// no count, matching Rust's float-to-integer conversion for the prior event-wise bridge.
fn truncated_mouse_count(value: f64) -> i32 {
    if value.is_nan() {
        return 0;
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            i32::MIN
        } else {
            i32::MAX
        };
    }

    let value = value.trunc();
    if (f64::from(i32::MIN)..=f64::from(i32::MAX)).contains(&value) {
        return value as i32;
    }

    // Rust's float-to-int cast saturates, but the integer `Mouse.getDX` accumulator wraps.  The
    // modulo path only matters for values a real device cannot produce in a frame.
    value.rem_euclid(4_294_967_296.0) as u32 as i32
}

/// LWJGL 2 scancode ↔ winit physical key. Vanilla's `options.txt` speaks the left column.
const LWJGL_KEYS: &[(i32, KeyCode)] = &[
    (1, KeyCode::Escape),
    (2, KeyCode::Digit1),
    (3, KeyCode::Digit2),
    (4, KeyCode::Digit3),
    (5, KeyCode::Digit4),
    (6, KeyCode::Digit5),
    (7, KeyCode::Digit6),
    (8, KeyCode::Digit7),
    (9, KeyCode::Digit8),
    (10, KeyCode::Digit9),
    (11, KeyCode::Digit0),
    (12, KeyCode::Minus),
    (13, KeyCode::Equal),
    (14, KeyCode::Backspace),
    (15, KeyCode::Tab),
    (16, KeyCode::KeyQ),
    (17, KeyCode::KeyW),
    (18, KeyCode::KeyE),
    (19, KeyCode::KeyR),
    (20, KeyCode::KeyT),
    (21, KeyCode::KeyY),
    (22, KeyCode::KeyU),
    (23, KeyCode::KeyI),
    (24, KeyCode::KeyO),
    (25, KeyCode::KeyP),
    (26, KeyCode::BracketLeft),
    (27, KeyCode::BracketRight),
    (28, KeyCode::Enter),
    (29, KeyCode::ControlLeft),
    (30, KeyCode::KeyA),
    (31, KeyCode::KeyS),
    (32, KeyCode::KeyD),
    (33, KeyCode::KeyF),
    (34, KeyCode::KeyG),
    (35, KeyCode::KeyH),
    (36, KeyCode::KeyJ),
    (37, KeyCode::KeyK),
    (38, KeyCode::KeyL),
    (39, KeyCode::Semicolon),
    (40, KeyCode::Quote),
    (41, KeyCode::Backquote),
    (42, KeyCode::ShiftLeft),
    (43, KeyCode::Backslash),
    (44, KeyCode::KeyZ),
    (45, KeyCode::KeyX),
    (46, KeyCode::KeyC),
    (47, KeyCode::KeyV),
    (48, KeyCode::KeyB),
    (49, KeyCode::KeyN),
    (50, KeyCode::KeyM),
    (51, KeyCode::Comma),
    (52, KeyCode::Period),
    (53, KeyCode::Slash),
    (54, KeyCode::ShiftRight),
    (55, KeyCode::NumpadMultiply),
    (56, KeyCode::AltLeft),
    (57, KeyCode::Space),
    (58, KeyCode::CapsLock),
    (59, KeyCode::F1),
    (60, KeyCode::F2),
    (61, KeyCode::F3),
    (62, KeyCode::F4),
    (63, KeyCode::F5),
    (64, KeyCode::F6),
    (65, KeyCode::F7),
    (66, KeyCode::F8),
    (67, KeyCode::F9),
    (68, KeyCode::F10),
    (69, KeyCode::NumLock),
    (70, KeyCode::ScrollLock),
    (71, KeyCode::Numpad7),
    (72, KeyCode::Numpad8),
    (73, KeyCode::Numpad9),
    (74, KeyCode::NumpadSubtract),
    (75, KeyCode::Numpad4),
    (76, KeyCode::Numpad5),
    (77, KeyCode::Numpad6),
    (78, KeyCode::NumpadAdd),
    (79, KeyCode::Numpad1),
    (80, KeyCode::Numpad2),
    (81, KeyCode::Numpad3),
    (82, KeyCode::Numpad0),
    (83, KeyCode::NumpadDecimal),
    (86, KeyCode::IntlBackslash),
    (87, KeyCode::F11),
    (88, KeyCode::F12),
    (100, KeyCode::F13),
    (101, KeyCode::F14),
    (102, KeyCode::F15),
    (103, KeyCode::F16),
    (104, KeyCode::F17),
    (105, KeyCode::F18),
    (113, KeyCode::F19),
    (112, KeyCode::KanaMode),
    (121, KeyCode::Convert),
    (123, KeyCode::NonConvert),
    (125, KeyCode::IntlYen),
    (141, KeyCode::NumpadEqual),
    (148, KeyCode::Lang1),
    (156, KeyCode::NumpadEnter),
    (157, KeyCode::ControlRight),
    (179, KeyCode::NumpadComma),
    (181, KeyCode::NumpadDivide),
    (183, KeyCode::PrintScreen),
    (184, KeyCode::AltRight),
    (197, KeyCode::Pause),
    (199, KeyCode::Home),
    (200, KeyCode::ArrowUp),
    (201, KeyCode::PageUp),
    (203, KeyCode::ArrowLeft),
    (205, KeyCode::ArrowRight),
    (207, KeyCode::End),
    (208, KeyCode::ArrowDown),
    (209, KeyCode::PageDown),
    (210, KeyCode::Insert),
    (211, KeyCode::Delete),
    (219, KeyCode::SuperLeft),
    (220, KeyCode::SuperRight),
    (221, KeyCode::ContextMenu),
    (222, KeyCode::Power),
    (223, KeyCode::Sleep),
];

/// Exact names returned by LWJGL 2.9.4's `Keyboard.getKeyName`.
#[must_use]
fn lwjgl_name(code: i32) -> Option<&'static str> {
    Some(match code {
        0 => "NONE",
        1 => "ESCAPE",
        2 => "1",
        3 => "2",
        4 => "3",
        5 => "4",
        6 => "5",
        7 => "6",
        8 => "7",
        9 => "8",
        10 => "9",
        11 => "0",
        12 => "MINUS",
        13 => "EQUALS",
        14 => "BACK",
        15 => "TAB",
        16 => "Q",
        17 => "W",
        18 => "E",
        19 => "R",
        20 => "T",
        21 => "Y",
        22 => "U",
        23 => "I",
        24 => "O",
        25 => "P",
        26 => "LBRACKET",
        27 => "RBRACKET",
        28 => "RETURN",
        29 => "LCONTROL",
        30 => "A",
        31 => "S",
        32 => "D",
        33 => "F",
        34 => "G",
        35 => "H",
        36 => "J",
        37 => "K",
        38 => "L",
        39 => "SEMICOLON",
        40 => "APOSTROPHE",
        41 => "GRAVE",
        42 => "LSHIFT",
        43 => "BACKSLASH",
        44 => "Z",
        45 => "X",
        46 => "C",
        47 => "V",
        48 => "B",
        49 => "N",
        50 => "M",
        51 => "COMMA",
        52 => "PERIOD",
        53 => "SLASH",
        54 => "RSHIFT",
        55 => "MULTIPLY",
        56 => "LMENU",
        57 => "SPACE",
        58 => "CAPITAL",
        59 => "F1",
        60 => "F2",
        61 => "F3",
        62 => "F4",
        63 => "F5",
        64 => "F6",
        65 => "F7",
        66 => "F8",
        67 => "F9",
        68 => "F10",
        69 => "NUMLOCK",
        70 => "SCROLL",
        71 => "NUMPAD7",
        72 => "NUMPAD8",
        73 => "NUMPAD9",
        74 => "SUBTRACT",
        75 => "NUMPAD4",
        76 => "NUMPAD5",
        77 => "NUMPAD6",
        78 => "ADD",
        79 => "NUMPAD1",
        80 => "NUMPAD2",
        81 => "NUMPAD3",
        82 => "NUMPAD0",
        83 => "DECIMAL",
        87 => "F11",
        88 => "F12",
        100 => "F13",
        101 => "F14",
        102 => "F15",
        103 => "F16",
        104 => "F17",
        105 => "F18",
        112 => "KANA",
        113 => "F19",
        121 => "CONVERT",
        123 => "NOCONVERT",
        125 => "YEN",
        141 => "NUMPADEQUALS",
        144 => "CIRCUMFLEX",
        145 => "AT",
        146 => "COLON",
        147 => "UNDERLINE",
        148 => "KANJI",
        149 => "STOP",
        150 => "AX",
        151 => "UNLABELED",
        156 => "NUMPADENTER",
        157 => "RCONTROL",
        167 => "SECTION",
        179 => "NUMPADCOMMA",
        181 => "DIVIDE",
        183 => "SYSRQ",
        184 => "RMENU",
        196 => "FUNCTION",
        197 => "PAUSE",
        199 => "HOME",
        200 => "UP",
        201 => "PRIOR",
        203 => "LEFT",
        205 => "RIGHT",
        207 => "END",
        208 => "DOWN",
        209 => "NEXT",
        210 => "INSERT",
        211 => "DELETE",
        218 => "CLEAR",
        219 => "LMETA",
        220 => "RMETA",
        221 => "APPS",
        222 => "POWER",
        223 => "SLEEP",
        _ => return None,
    })
}

#[must_use]
fn lwjgl_name_of_keycode(key: KeyCode) -> Option<&'static str> {
    lwjgl_of_keycode(key).and_then(lwjgl_name)
}

/// LWJGL 2 scancode → winit physical key, or `None` for a scancode we have no equivalent for.
#[must_use]
pub fn keycode_of_lwjgl(code: i32) -> Option<KeyCode> {
    LWJGL_KEYS.iter().find(|(c, _)| *c == code).map(|(_, k)| *k)
}

/// winit physical key → LWJGL 2 scancode, or `None` for a key 1.8.9 could not express.
#[must_use]
pub fn lwjgl_of_keycode(key: KeyCode) -> Option<i32> {
    LWJGL_KEYS.iter().find(|(_, k)| *k == key).map(|(c, _)| *c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lwjgl_codes_round_trip() {
        for &(code, key) in LWJGL_KEYS {
            assert_eq!(lwjgl_of_keycode(key), Some(code));
            assert_eq!(keycode_of_lwjgl(code), Some(key));
            assert_eq!(Input::from_lwjgl(code), Input::Key(key));
            assert_eq!(Input::Key(key).to_lwjgl(), code);
        }
    }

    #[test]
    fn lwjgl_table_has_no_duplicates() {
        let codes: HashSet<i32> = LWJGL_KEYS.iter().map(|(c, _)| *c).collect();
        let keys: HashSet<KeyCode> = LWJGL_KEYS.iter().map(|(_, k)| *k).collect();
        assert_eq!(codes.len(), LWJGL_KEYS.len());
        assert_eq!(keys.len(), LWJGL_KEYS.len());
    }

    #[test]
    fn mouse_buttons_use_vanillas_negative_encoding() {
        // java: GameSettings — attack -100, use -99, pickItem -98
        assert_eq!(Input::MOUSE_LEFT.to_lwjgl(), -100);
        assert_eq!(Input::MOUSE_RIGHT.to_lwjgl(), -99);
        assert_eq!(Input::MOUSE_MIDDLE.to_lwjgl(), -98);
        assert_eq!(Input::from_lwjgl(-100), Input::MOUSE_LEFT);
        assert_eq!(Input::from_lwjgl(-99), Input::MOUSE_RIGHT);
        assert_eq!(Input::from_lwjgl(-98), Input::MOUSE_MIDDLE);
    }

    #[test]
    fn zero_is_unbound_and_unknown_codes_round_trip_losslessly() {
        assert_eq!(Input::from_lwjgl(0), Input::Unbound);
        assert_eq!(Input::Unbound.to_lwjgl(), 0);
        for code in [
            113, 144, 145, 146, 147, 149, 150, 151, 167, 196, 218, 9999, -101,
        ] {
            assert_eq!(Input::from_lwjgl(code).to_lwjgl(), code, "code {code}");
        }
        assert_eq!(Input::from_lwjgl(113), Input::Key(KeyCode::F19));
        assert_eq!(Input::from_lwjgl(144), Input::RawLwjgl(144));
        assert_eq!(Input::from_lwjgl(-101), Input::RawLwjgl(-101));
    }

    #[test]
    fn display_names_are_lwjgl_names_not_winit_debug_names() {
        assert_eq!(Input::Key(KeyCode::KeyW).display_name(), "W");
        assert_eq!(Input::Key(KeyCode::Space).display_name(), "SPACE");
        assert_eq!(Input::Key(KeyCode::ControlLeft).display_name(), "LCONTROL");
        assert_eq!(Input::Key(KeyCode::Digit1).display_name(), "1");
        assert_eq!(Input::Key(KeyCode::PageUp).display_name(), "PRIOR");
        assert_eq!(Input::RawLwjgl(144).display_name(), "CIRCUMFLEX");
        assert_eq!(Input::RawLwjgl(9999).display_name(), "null");
        assert_eq!(Input::RawLwjgl(-101).display_name(), "Button 0");
        assert_eq!(Input::Unbound.display_name(), "NONE");
    }

    #[test]
    fn vanilla_defaults_match_the_java_source() {
        // java: GameSettings.java lines 127-151
        let b = KeyBindings::vanilla();
        let expect = [
            (keys::FORWARD, 17),
            (keys::LEFT, 30),
            (keys::BACK, 31),
            (keys::RIGHT, 32),
            (keys::JUMP, 57),
            (keys::SNEAK, 42),
            (keys::SPRINT, 29),
            (keys::INVENTORY, 18),
            (keys::USE_ITEM, -99),
            (keys::DROP, 16),
            (keys::ATTACK, -100),
            (keys::PICK_ITEM, -98),
            (keys::CHAT, 20),
            (keys::PLAYER_LIST, 15),
            (keys::COMMAND, 53),
            (keys::SCREENSHOT, 60),
            (keys::TOGGLE_PERSPECTIVE, 63),
            (keys::SMOOTH_CAMERA, 0),
            (keys::STREAM_START_STOP, 64),
            (keys::STREAM_PAUSE_UNPAUSE, 65),
            (keys::STREAM_COMMERCIAL, 0),
            (keys::STREAM_TOGGLE_MIC, 0),
            (keys::FULLSCREEN, 87),
            (keys::SPECTATOR_OUTLINES, 0),
        ];
        for (name, code) in expect {
            let binding = b.get(name).unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(binding.binding().to_lwjgl(), code, "{name}");
        }
        for slot in 1..=9u8 {
            let name = keys::hotbar(slot);
            assert_eq!(
                b.get(&name).unwrap().binding().to_lwjgl(),
                i32::from(slot) + 1,
                "{name}"
            );
        }
    }

    /// The allocation-free table must name exactly what `hotbar` names, and a slot outside 1–9
    /// must miss every binding just as `hotbar`'s string does.
    #[test]
    fn hotbar_static_matches_hotbar_and_misses_identically() {
        let b = KeyBindings::vanilla();
        for slot in 1..=9u8 {
            assert_eq!(keys::hotbar_static(slot), keys::hotbar(slot));
        }
        for slot in [0u8, 10, 11, 255] {
            assert!(b.get(&keys::hotbar(slot)).is_none(), "slot {slot}");
            assert!(b.get(keys::hotbar_static(slot)).is_none(), "slot {slot}");
        }
    }

    #[test]
    fn press_is_consumed_once_per_physical_press() {
        let mut b = KeyBindings::vanilla();
        b.set_state(Input::Key(KeyCode::KeyW), true);
        assert!(b.is_down(keys::FORWARD));
        assert!(b.take_pressed(keys::FORWARD));
        assert!(!b.take_pressed(keys::FORWARD));
        assert!(b.is_down(keys::FORWARD));
        b.set_state(Input::Key(KeyCode::KeyW), false);
        assert!(!b.is_down(keys::FORWARD));
    }

    #[test]
    fn queued_presses_accumulate_across_ticks() {
        let mut b = KeyBindings::vanilla();
        for _ in 0..3 {
            b.set_state(Input::Key(KeyCode::KeyE), true);
            b.set_state(Input::Key(KeyCode::KeyE), false);
        }
        assert!(b.take_pressed(keys::INVENTORY));
        assert!(b.take_pressed(keys::INVENTORY));
        assert!(b.take_pressed(keys::INVENTORY));
        assert!(!b.take_pressed(keys::INVENTORY));
    }

    #[test]
    fn rebinding_moves_the_action_and_preserves_held_and_queued_state() {
        let mut b = KeyBindings::vanilla();
        b.set_state(Input::Key(KeyCode::KeyW), true);
        assert!(b.set_binding(keys::FORWARD, Input::Key(KeyCode::ArrowUp)));
        assert!(b.is_down(keys::FORWARD));
        assert!(b.take_pressed(keys::FORWARD));
        assert!(!b.take_pressed(keys::FORWARD));

        // The preserved held flag belongs to the binding, not its old physical input. Java's
        // `setKeyCode` leaves it set until a later edge or `unPressAllKeys` clears it.
        b.set_state(Input::Key(KeyCode::KeyW), true);
        assert!(b.is_down(keys::FORWARD));
        b.set_state(Input::Key(KeyCode::ArrowUp), false);
        assert!(!b.is_down(keys::FORWARD));
        b.set_state(Input::Key(KeyCode::ArrowUp), true);
        assert!(b.is_down(keys::FORWARD));
        assert!(!b.set_binding("key.nonexistent", Input::Unbound));
    }

    #[test]
    fn conflicts_are_detected_and_unbound_never_conflicts() {
        let mut b = KeyBindings::vanilla();
        assert!(b.conflicts().is_empty(), "vanilla defaults must not clash");
        assert!(!b.is_conflicting(keys::SMOOTH_CAMERA)); // both spectator+smooth are unbound
        assert!(!b.is_conflicting(keys::SPECTATOR_OUTLINES));

        b.set_binding(keys::SPRINT, Input::Key(KeyCode::KeyW));
        assert!(b.is_conflicting(keys::SPRINT));
        assert!(b.is_conflicting(keys::FORWARD));
        assert!(!b.is_conflicting(keys::BACK));
        let groups = b.conflicts();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 2);
        assert!(groups[0].contains(&keys::FORWARD));
        assert!(groups[0].contains(&keys::SPRINT));

        b.reset_all();
        assert!(b.conflicts().is_empty());
    }

    #[test]
    fn one_key_drives_only_vanillas_last_registered_action() {
        let mut b = KeyBindings::vanilla();
        b.set_binding(keys::SPRINT, Input::Key(KeyCode::KeyW));
        b.set_state(Input::Key(KeyCode::KeyW), true);
        assert!(!b.is_down(keys::FORWARD));
        assert!(b.is_down(keys::SPRINT));
        assert!(!b.take_pressed(keys::FORWARD));
        assert!(b.take_pressed(keys::SPRINT));

        // `keyBindings` list order starts Attack, Use, but Java field construction order is
        // Use, Attack. This catches both "drive every match" and a tempting Vec::rfind mutant.
        b.set_binding(keys::USE_ITEM, Input::MOUSE_LEFT);
        b.set_state(Input::MOUSE_LEFT, true);
        assert!(b.is_down(keys::ATTACK));
        assert!(!b.is_down(keys::USE_ITEM));
    }

    #[test]
    fn unpress_all_clears_held_and_queued() {
        let mut b = KeyBindings::vanilla();
        b.set_state(Input::Key(KeyCode::Space), true);
        b.unpress_all();
        assert!(!b.is_down(keys::JUMP));
        assert!(!b.take_pressed(keys::JUMP));
    }

    #[test]
    fn disallowed_screen_input_cannot_rearm_a_binding_after_unpress() {
        let mut b = KeyBindings::vanilla();

        // Opening a normal GuiScreen calls KeyBinding.unPressAllKeys. Its own key-event loop
        // consumes later W/E edges, so neither a held movement binding nor an isPressed queue
        // may reappear before the screen closes.
        b.set_state(Input::Key(KeyCode::KeyW), true);
        b.set_state(Input::Key(KeyCode::KeyE), true);
        b.unpress_all();
        b.set_state_when_input_allowed(Input::Key(KeyCode::KeyW), true, false);
        b.set_state_when_input_allowed(Input::Key(KeyCode::KeyE), true, false);
        b.set_state_when_input_allowed(Input::Key(KeyCode::KeyW), false, false);
        b.set_state_when_input_allowed(Input::Key(KeyCode::KeyE), false, false);

        assert!(!b.is_down(keys::FORWARD));
        assert!(!b.take_pressed(keys::INVENTORY));

        // Once game input is allowed again (no screen open), re-presses must reach the same
        // KeyBinding latch and queue.
        b.set_state_when_input_allowed(Input::Key(KeyCode::KeyW), true, true);
        b.set_state_when_input_allowed(Input::Key(KeyCode::KeyE), true, true);
        assert!(b.is_down(keys::FORWARD));
        assert!(b.take_pressed(keys::INVENTORY));
    }

    #[test]
    fn sensitivity_curve_matches_vanilla_at_default() {
        let mut input = InputState::new();
        input.request_cursor(CursorMode::Grabbed);
        input.on_raw_motion(100.0, 0.0);
        // f = 0.5*0.6+0.2 = 0.5 ; f^3*8 = 1.0 ; *0.15 deg = 15 degrees for 100 counts.
        let (yaw, pitch) = input.look_delta(0.5, false);
        assert!((yaw - 15.0).abs() < 1e-4, "{yaw}");
        assert_eq!(pitch, 0.0);
    }

    #[test]
    fn entity_angle_scale_uses_the_java_double_literal() {
        let mut input = InputState::new();
        input.request_cursor(CursorMode::Grabbed);
        input.on_raw_motion(3.0, 3.0);
        let (yaw, pitch) = input.look_delta(0.5, false);
        assert_eq!(yaw.to_bits(), 0x3ee6_6666, "yaw={yaw:?}");
        assert_eq!(pitch.to_bits(), 0x3ee6_6666, "pitch={pitch:?}");
    }

    #[test]
    fn raw_look_delta_is_exactly_the_float_passed_to_set_angles() {
        let mut input = InputState::new();
        input.request_cursor(CursorMode::Grabbed);
        input.on_raw_motion(3.0, -4.0);
        // At sensitivity 0.5 the cubic multiplier is exactly 1.0. EntityRenderer passes these
        // floats to setAngles; the latter applies 0.15D while adding to existing rotation.
        assert_eq!(input.look_delta_raw(0.5, false), (3.0, -4.0));
    }

    #[test]
    fn sensitivity_curve_is_cubic_not_linear() {
        let mut input = InputState::new();
        input.request_cursor(CursorMode::Grabbed);
        input.on_raw_motion(100.0, 0.0);
        let (full, _) = input.look_delta(1.0, false);
        // f = 0.8 ; 0.8^3*8 = 4.096 ; *0.15*100 = 61.44 degrees. Linear would give 30.
        assert!((full - 61.44).abs() < 1e-3, "{full}");
    }

    #[test]
    fn invert_y_flips_pitch_only() {
        let mut input = InputState::new();
        input.request_cursor(CursorMode::Grabbed);
        input.on_raw_motion(10.0, 10.0);
        let (yaw_a, pitch_a) = input.look_delta(0.5, false);
        input.on_raw_motion(10.0, 10.0);
        let (yaw_b, pitch_b) = input.look_delta(0.5, true);
        assert_eq!(yaw_a, yaw_b);
        assert_eq!(pitch_a, -pitch_b);
        assert!(pitch_a > 0.0, "mouse down must look down");
    }

    #[test]
    fn motion_is_ignored_while_ungrabbed() {
        let mut input = InputState::new();
        input.on_raw_motion(500.0, 500.0);
        assert_eq!(input.take_mouse_delta(), (0, 0));
        input.request_cursor(CursorMode::Grabbed);
        input.on_raw_motion(1.0, 2.0);
        assert_eq!(input.take_mouse_delta(), (1, 2));
    }

    #[test]
    fn grabbing_discards_stale_motion() {
        let mut input = InputState::new();
        input.request_cursor(CursorMode::Grabbed);
        input.on_raw_motion(1000.0, 1000.0);
        input.request_cursor(CursorMode::Free);
        input.request_cursor(CursorMode::Grabbed);
        assert_eq!(input.take_mouse_delta(), (0, 0));
    }

    #[test]
    fn release_preserves_pending_delta_but_the_next_grab_discards_it() {
        let mut input = InputState::new();
        input.request_cursor(CursorMode::Grabbed);
        input.on_raw_motion(7.0, -3.0);

        // MouseHelper.ungrabMouseCursor only recentres/releases the OS cursor. It leaves the
        // public delta fields untouched; preserving this boundary also keeps a backend recenter
        // failure from inventing a different delta policy in the raw-input layer.
        input.request_cursor(CursorMode::Free);
        assert_eq!(input.take_mouse_delta(), (7, -3));

        // MouseHelper.grabMouseCursor zeros the fields before the first in-game frame. A stale
        // value must not survive a free -> grabbed transition, regardless of how the cursor was
        // recentered by the platform.
        input.request_cursor(CursorMode::Grabbed);
        assert_eq!(input.take_mouse_delta(), (0, 0));
    }

    #[test]
    fn fractional_motion_accumulates_before_frame_extraction_and_resets() {
        let mut input = InputState::new();
        input.request_cursor(CursorMode::Grabbed);

        // LWJGL reads the whole interval at once: two sub-count events combine before the
        // integer delta is exposed.  Y remains positive-down in the winit-facing state.
        input.on_raw_motion(0.75, -0.75);
        input.on_raw_motion(0.75, -0.75);
        assert_eq!(input.take_mouse_delta(), (1, -1));
        assert_eq!(input.take_mouse_delta(), (0, 0));

        // Negative accumulation must truncate toward zero after summing, not per event.
        input.on_raw_motion(-0.75, 0.75);
        input.on_raw_motion(-0.75, 0.75);
        assert_eq!(input.take_mouse_delta(), (-1, 1));

        // A sub-count residue is consumed/reset with the frame; it is not banked into the next
        // call, matching the integer-valued Mouse.getDX/getDY read boundary.
        input.on_raw_motion(0.75, -0.75);
        assert_eq!(input.take_mouse_delta(), (0, 0));
        assert_eq!(input.take_mouse_delta(), (0, 0));
    }

    #[test]
    fn raw_motion_total_truncates_toward_zero() {
        let mut input = InputState::new();
        input.request_cursor(CursorMode::Grabbed);

        // Java/LWJGL exposes signed `int` counts. Winit exposes raw events as f64, so the bridge
        // discards the fractional portion toward zero when the frame consumes the total.
        input.on_raw_motion(1.99, -1.99);
        input.on_raw_motion(-2.99, 2.99);
        assert_eq!(input.take_mouse_delta(), (-1, 1));
    }

    #[test]
    fn raw_motion_count_accumulation_wraps_like_java_int() {
        let mut input = InputState::new();
        input.request_cursor(CursorMode::Grabbed);

        // Retaining Java's two's-complement wrap matters at this otherwise pathological boundary:
        // saturating would freeze a following count instead.
        input.on_raw_motion(f64::from(i32::MAX), f64::from(i32::MIN));
        input.on_raw_motion(1.0, -1.0);
        assert_eq!(input.take_mouse_delta(), (i32::MIN, i32::MAX));
    }

    #[test]
    fn key_repeat_events_are_ignored() {
        let mut input = InputState::new();
        assert!(input.on_key(KeyCode::KeyW, true, false).is_some());
        assert!(input.on_key(KeyCode::KeyW, true, true).is_none());
        assert!(input.key_down(KeyCode::KeyW));
    }

    #[test]
    fn losing_focus_releases_everything() {
        let mut input = InputState::new();
        input.on_focus(true);
        input.on_key(KeyCode::KeyW, true, false);
        input.on_mouse_button(MouseButton::Left, true);
        input.request_cursor(CursorMode::Grabbed);
        input.on_raw_motion(4.0, -5.0);
        input.on_scroll(winit::event::MouseScrollDelta::LineDelta(0.0, 2.0));
        input.on_mouse_button_at_ms(MouseButton::Right, true, 1);
        input.on_focus(false);
        assert!(!input.key_down(KeyCode::KeyW));
        assert!(!input.button_down(0));
        assert!(!input.grabbed());
        assert_eq!(input.take_mouse_delta(), (0, 0));
        assert_eq!(input.take_scroll(), 0.0);
        assert!(input.drain_mouse_events_at_ms(1).is_empty());
    }

    #[test]
    fn the_shell_releases_the_grab_before_it_warps_on_windows_and_survives_a_warp_failure() {
        // A synthetic winit Window cannot exercise OS cursor errors or clipping, so pin the
        // shell's order here. On Windows the release must come first: winit 0.30 implements
        // `Locked` there as a 1x1 `ClipCursor` rect at the pointer, and `SetCursorPos` is
        // clamped to an active clip while still returning success. Warping first (LWJGL's
        // order) left the pointer wherever the last grab caught it: the "cursor doesn't go back
        // to the middle" bug of 2026-11-07. Wayland only warps a locked pointer, so the other
        // platforms keep the warp before the release. The cursor is shown last either way.
        //
        // `apply_cursor` now acts on a `CursorPlan`
        // (`shell::cursor_plan`), so the free arm is `CursorPlan::Free { recentre }` and commits
        // `Some(CursorMode::Free)`. The plan clears `recentre` for an unfocused window (no warp
        // while another application has the pointer), and the arm drops a clip winit cannot
        // drop for an inactive window (`os_cursor::release_clip`) before the release.
        let source = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/shell.rs"));
        let apply = source
            .split_once("fn apply_cursor(")
            .expect("apply_cursor")
            .1
            .split_once("\n}")
            .expect("apply_cursor body")
            .0;
        let free = apply
            .split_once("CursorPlan::Free { recentre } =>")
            .expect("free cursor arm")
            .1
            .split_once("state.applied_cursor = Some(CursorMode::Free)")
            .expect("cursor state commit")
            .0;
        let other_warp = free
            .find("#[cfg(not(windows))]")
            .expect("non-Windows platforms warp while still locked");
        let unclip = free
            .find("os_cursor::release_clip()")
            .expect("an unfocused release drops winit's leftover clip");
        let release = free
            .find("state.window.set_cursor_grab(CursorGrabMode::None)")
            .expect("the free arm releases the grab");
        let windows_warp = free
            .find("#[cfg(windows)]")
            .expect("Windows warps after the release");
        let visible = free
            .find("state.window.set_cursor_visible(true)")
            .expect("free cursor is shown after release");
        assert!(other_warp < release && release < windows_warp && windows_warp < visible);
        assert!(unclip < release, "the clip goes before winit's release");
        assert!(free.contains("if !focused && state.applied_cursor == Some(CursorMode::Grabbed)"));
        assert_eq!(
            free.matches("warp_to_release_point(&state.window)").count(),
            2,
            "one warp per platform branch, both behind the recentre request"
        );
        assert_eq!(
            free.matches("if recentre {").count(),
            2,
            "both warps wait for the recentre request"
        );
        // The plan never asks an unfocused window to re-centre.
        let plan = source
            .split_once("fn cursor_plan(")
            .expect("cursor_plan")
            .1
            .split_once("\n}")
            .expect("cursor_plan body")
            .0;
        assert!(plan.contains("CursorPlan::Free { recentre: false }"));
        let warp = source
            .split_once("fn warp_to_release_point(")
            .expect("warp helper")
            .1
            .split_once("\n}")
            .expect("warp helper body")
            .0;
        assert!(warp.contains("if let Err(e) = window.set_cursor_position(centre)"));
        assert!(warp.contains(
            "tracing::warn!(error = %e, \"cursor re-centre unavailable while ungrabbing\")"
        ));
    }

    /// Vanilla's ungrab target, in winit's top-left frame. `WindowsDisplay.setCursorPosition`
    /// warps to `rect.bottom - 1 - y` for LWJGL's `y = height / 2`.
    #[test]
    fn the_release_point_is_lwjgls_centre_flipped_into_the_top_left_frame() {
        assert_eq!(
            cursor_release_point(854, 480),
            (427, 239),
            "even height: one row up"
        );
        assert_eq!(
            cursor_release_point(855, 481),
            (427, 240),
            "odd sizes truncate"
        );
        assert_eq!(cursor_release_point(1920, 1080), (960, 539));
        assert_eq!(cursor_release_point(1, 1), (0, 0));
        assert_eq!(
            cursor_release_point(0, 0),
            (0, 0),
            "a minimised window cannot underflow"
        );
        assert_eq!(
            cursor_release_point(u32::MAX, u32::MAX),
            (i32::MAX as u32, i32::MAX as u32),
            "every winit size stays representable as the shell's i32 position"
        );
    }

    /// A focused screen open (grabbed → free) moves the GUI cursor to the release point at once,
    /// before any `CursorMoved` echo of the OS warp, and asks the shell for exactly one warp.
    #[test]
    fn a_focused_release_recentres_the_gui_cursor_and_requests_one_warp() {
        let mut input = InputState::new();
        input.on_focus(true);
        input.set_window_size(854, 480);
        input.request_cursor(CursorMode::Grabbed);
        // Wherever the hidden pointer was last reported while playing.
        input.on_cursor_moved(12.0, 470.0);
        assert!(
            !input.take_cursor_recentre(),
            "grabbing never asks for a warp"
        );

        input.request_cursor(CursorMode::Free);
        assert_eq!(input.cursor(), (427.0, 239.0));
        assert!(input.take_cursor_recentre());
        assert!(
            !input.take_cursor_recentre(),
            "the request is consumed once"
        );

        // The OS echo of the warp, and later real motion, keep driving the GUI cursor.
        input.on_cursor_moved(427.0, 239.0);
        input.on_cursor_moved(500.0, 300.0);
        assert_eq!(input.cursor(), (500.0, 300.0));

        // A repeated Free request is not another edge.
        input.request_cursor(CursorMode::Free);
        assert_eq!(input.cursor(), (500.0, 300.0));
        assert!(!input.take_cursor_recentre());
    }

    /// `setIngameNotInFocus` only ungrabs `if (this.inGameHasFocus)`: the first screen at startup
    /// and screen-to-screen changes leave the pointer alone, and so does a release while another
    /// application has focus.
    #[test]
    fn only_a_focused_grabbed_to_free_edge_recentres() {
        // Startup: the first request is Free with nothing grabbed before it.
        let mut input = InputState::new();
        input.on_focus(true);
        input.set_window_size(854, 480);
        input.on_cursor_moved(5.0, 6.0);
        input.request_cursor(CursorMode::Free);
        assert_eq!(input.cursor(), (5.0, 6.0));
        assert!(!input.take_cursor_recentre());

        // Alt-Tab: `on_focus(false)` releases the grab but must not move the pointer.
        input.request_cursor(CursorMode::Grabbed);
        input.on_focus(false);
        assert!(!input.grabbed());
        assert_eq!(input.cursor(), (5.0, 6.0));
        assert!(!input.take_cursor_recentre());

        // An unfocused grabbed → free edge (the 500 ms lost-focus pause) does not either.
        input.request_cursor(CursorMode::Grabbed);
        input.request_cursor(CursorMode::Free);
        assert_eq!(input.cursor(), (5.0, 6.0));
        assert!(!input.take_cursor_recentre());
    }

    /// The warp request lives until the shell takes it, even across a later grab → free edge in
    /// the same frame (a screen closed by a tick, another opened by the input pass: the window
    /// never saw the grab, but the GUI cursor moved); a grab cancels a request nobody took.
    #[test]
    fn the_warp_request_survives_until_taken_and_a_grab_cancels_it() {
        let mut input = InputState::new();
        input.on_focus(true);
        input.request_cursor(CursorMode::Grabbed);
        input.on_cursor_moved(40.0, 50.0);
        // No size reported yet: the GUI cursor cannot be placed, but the shell still warps.
        input.request_cursor(CursorMode::Free);
        assert_eq!(input.cursor(), (40.0, 50.0));
        assert!(input.take_cursor_recentre());

        // Release, grab, release again before the shell runs: one warp, at the new centre.
        input.set_window_size(855, 481);
        input.request_cursor(CursorMode::Grabbed);
        input.request_cursor(CursorMode::Free);
        input.request_cursor(CursorMode::Grabbed);
        input.request_cursor(CursorMode::Free);
        assert_eq!(input.cursor(), (427.0, 240.0));
        assert!(input.take_cursor_recentre());
        assert!(!input.take_cursor_recentre());

        // A release followed by a grab that the shell never saw: nothing to warp.
        input.request_cursor(CursorMode::Grabbed);
        input.request_cursor(CursorMode::Free);
        input.request_cursor(CursorMode::Grabbed);
        assert!(
            !input.take_cursor_recentre(),
            "a grab cancels the pending warp"
        );
    }

    #[test]
    fn scroll_accumulates_and_drains() {
        let mut input = InputState::new();
        input.on_scroll(winit::event::MouseScrollDelta::LineDelta(0.0, 1.0));
        input.on_scroll(winit::event::MouseScrollDelta::LineDelta(0.0, -3.0));
        assert!((input.take_scroll() + 2.0).abs() < 1e-6);
        assert_eq!(input.take_scroll(), 0.0);
    }

    /// The tick-time wheel drain (H3) takes the wheel events one per OS
    /// message, in order and unmerged, and leaves the button edges queued, in order, for the
    /// screen path.
    #[test]
    fn the_wheel_drain_takes_each_wheel_event_and_keeps_the_button_edges() {
        let mut input = InputState::new();
        input.on_mouse_button_at_ms(MouseButton::Left, true, 0);
        input.on_scroll_at_ms(winit::event::MouseScrollDelta::LineDelta(0.0, 2.0), 1);
        input.on_mouse_button_at_ms(MouseButton::Left, false, 2);
        input.on_scroll_at_ms(winit::event::MouseScrollDelta::LineDelta(0.0, -0.25), 3);

        let mut wheels = Vec::new();
        input.drain_wheel_events(|delta| wheels.push(delta));
        assert_eq!(wheels, [2.0, -0.25], "one call per event, never merged");

        let left = input.drain_all_mouse_events();
        assert_eq!(
            left.iter().map(|event| event.kind).collect::<Vec<_>>(),
            [
                MouseEventKind::Button {
                    button: 0,
                    pressed: true,
                },
                MouseEventKind::Button {
                    button: 0,
                    pressed: false,
                },
            ]
        );
        let mut none = 0;
        input.drain_wheel_events(|_| none += 1);
        assert_eq!(none, 0);
    }

    #[test]
    fn timed_mouse_queue_preserves_edges_wheels_extra_buttons_and_age_boundary() {
        let mut input = InputState::new();
        input.on_mouse_button_at_ms(MouseButton::Left, true, 0);
        input.on_mouse_button_at_ms(MouseButton::Other(7), true, 1);
        input.on_mouse_button_at_ms(MouseButton::Left, false, 2);
        input.on_scroll_at_ms(winit::event::MouseScrollDelta::LineDelta(0.0, 1.0), 3);
        input.on_scroll_at_ms(winit::event::MouseScrollDelta::LineDelta(0.0, -2.0), 4);

        let events = input.drain_mouse_events_at_ms(200);
        assert_eq!(
            events
                .iter()
                .map(|event| event.sequence)
                .collect::<Vec<_>>(),
            [0, 1, 2, 3, 4]
        );
        assert_eq!(
            events.iter().map(|event| event.kind).collect::<Vec<_>>(),
            [
                MouseEventKind::Button {
                    button: 0,
                    pressed: true,
                },
                MouseEventKind::Button {
                    button: 7,
                    pressed: true,
                },
                MouseEventKind::Button {
                    button: 0,
                    pressed: false,
                },
                MouseEventKind::Wheel { delta: 1.0 },
                MouseEventKind::Wheel { delta: -2.0 },
            ]
        );

        input.on_mouse_button_at_ms(MouseButton::Right, true, 0);
        input.on_mouse_button_at_ms(MouseButton::Right, false, 1);
        assert_eq!(
            input
                .drain_mouse_events_at_ms(201)
                .into_iter()
                .map(|event| event.kind)
                .collect::<Vec<_>>(),
            [MouseEventKind::Button {
                button: 1,
                pressed: false,
            }],
            "age 200 is admitted and age 201 is dropped"
        );

        for _ in 0..512 {
            input.on_scroll_at_ms(winit::event::MouseScrollDelta::LineDelta(0.0, 1.0), 0);
        }
        input.on_scroll_at_ms(winit::event::MouseScrollDelta::LineDelta(0.0, -1.0), 201);
        assert_eq!(
            input.drain_mouse_events_at_ms(201).len(),
            1,
            "a stale flood cannot starve the fresh tail event"
        );

        input.on_mouse_button_at_ms(MouseButton::Middle, true, 202);
        input.on_focus(false);
        assert!(input.drain_mouse_events_at_ms(202).is_empty());
    }

    #[test]
    fn categories_cover_every_binding() {
        let b = KeyBindings::vanilla();
        assert!(b.categories().contains("key.categories.movement"));
        assert!(b.categories().contains("key.categories.stream"));
        assert_eq!(b.categories().len(), 6);
        assert_eq!(b.iter().count(), 33);
    }

    /// Raw Mouse Input OFF: grabbed look reads the OS pointer's movement and asks
    /// the shell for a re-centre after each read with motion. ON, the default, is the raw path
    /// exactly as before: pointer positions never look and nothing is ever warped.
    #[test]
    fn os_pointer_look_reads_pointer_movement_instead_of_raw_motion() {
        let mut input = InputState::new();
        input.request_cursor(CursorMode::Grabbed);
        assert!(!input.os_pointer_look(), "raw input is the default");
        input.on_cursor_moved(100.0, 100.0);
        input.on_cursor_moved(130.0, 90.0);
        input.on_raw_motion(5.0, -2.0);
        assert_eq!(input.take_mouse_delta(), (5, -2));
        assert!(
            !input.take_pointer_recentre(),
            "the raw path never warps the pointer"
        );

        // OS pointer: raw counts are ignored and movement from the anchor is the delta.
        input.on_raw_motion(1.0, 1.0);
        input.set_os_pointer_look(true);
        assert_eq!(
            input.take_mouse_delta(),
            (0, 0),
            "switching drops the pending delta"
        );
        assert!(!input.take_pointer_recentre());
        input.on_raw_motion(40.0, 40.0);
        input.on_cursor_moved(200.0, 150.0); // the first position only sets the anchor
        input.on_cursor_moved(212.0, 141.0);
        input.on_cursor_moved(215.0, 140.0);
        assert_eq!(input.take_mouse_delta(), (15, -10));
        assert!(
            input.take_pointer_recentre(),
            "a read with motion asks for the warp"
        );
        assert!(!input.take_pointer_recentre(), "once");
        // The shell warps to the centre; the warp's own echo adds nothing.
        input.pointer_recentred(427.0, 239.0);
        input.on_cursor_moved(427.0, 239.0);
        assert_eq!(input.take_mouse_delta(), (0, 0));
        assert!(!input.take_pointer_recentre(), "no motion, no warp");
        input.on_cursor_moved(420.0, 245.0);
        assert_eq!(input.take_mouse_delta(), (-7, 6));

        // Ungrabbed (a screen): the pointer only moves the GUI cursor, and the next grab does
        // not measure from a position the pointer had outside it.
        input.request_cursor(CursorMode::Free);
        input.on_cursor_moved(10.0, 10.0);
        assert_eq!(input.cursor(), (10.0, 10.0));
        assert_eq!(input.take_mouse_delta(), (0, 0));
        input.request_cursor(CursorMode::Grabbed);
        input.on_cursor_moved(300.0, 300.0);
        assert_eq!(
            input.take_mouse_delta(),
            (0, 0),
            "a fresh grab starts a fresh anchor"
        );

        // Back to raw: pointer movement no longer looks.
        input.set_os_pointer_look(false);
        input.on_cursor_moved(50.0, 50.0);
        input.on_cursor_moved(80.0, 80.0);
        input.on_raw_motion(3.0, 0.0);
        assert_eq!(input.take_mouse_delta(), (3, 0));
        assert!(!input.take_pointer_recentre());
    }

    /// Re-pressing restores only the held flag: the press counter that
    /// `isPressed` reads stays empty, an unheld key stays released, and a key shared by two
    /// actions goes to the one vanilla's dispatch would pick.
    #[test]
    fn repress_sets_the_held_flag_only_for_physically_held_inputs() {
        let mut b = KeyBindings::vanilla();
        let mut raw = InputState::new();
        raw.on_key(KeyCode::KeyW, true, false);
        b.set_state(Input::Key(KeyCode::KeyW), true);
        b.unpress_all();
        assert!(!b.is_down(keys::FORWARD), "a screen opened: unpressed");

        let held = |input: Input| raw.is_down(input);
        assert!(b.repress_if_held(keys::FORWARD, held));
        assert!(b.is_down(keys::FORWARD));
        assert!(
            !b.take_pressed(keys::FORWARD),
            "no queued press: an edge binding on this key could not fire"
        );
        assert!(!b.repress_if_held(keys::BACK, held), "S is not held");
        assert!(!b.is_down(keys::BACK));
        assert!(!b.repress_if_held("key.nonexistent", held));

        // E held across the close: the inventory binding is never re-fired by the held flag.
        raw.on_key(KeyCode::KeyE, true, false);
        let held = |input: Input| raw.is_down(input);
        assert!(b.repress_if_held(keys::INVENTORY, held));
        assert!(!b.take_pressed(keys::INVENTORY));

        // W bound to both forward and sprint: vanilla's dispatch gives W's edges to sprint
        // (registered later), so forward is not re-pressed and sprint is.
        let mut shared = KeyBindings::vanilla();
        shared.set_binding(keys::SPRINT, Input::Key(KeyCode::KeyW));
        assert!(!shared.repress_if_held(keys::FORWARD, held));
        assert!(shared.repress_if_held(keys::SPRINT, held));
        assert!(shared.is_down(keys::SPRINT) && !shared.is_down(keys::FORWARD));

        // Unbound never re-presses.
        let mut unbound = KeyBindings::vanilla();
        unbound.set_binding(keys::FORWARD, Input::Unbound);
        assert!(!unbound.repress_if_held(keys::FORWARD, |_| true));
    }
}
