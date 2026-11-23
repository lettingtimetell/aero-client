//! Generic screen behavior: focus, keyboard navigation, and modal behaviour.
//!
//! java: `client/gui/GuiScreen.java`, `client/Minecraft.displayGuiScreen`
//!
//! [`Screen`] and [`ScreenCtx`] are production-facing low-level contracts for individual screens.
//! [`ScreenStack`] is deliberately retained only as a generic library/test harness. The live
//! `mc-client` runtime dispatches its [`crate::screens`] through the `Ui` enum because screen
//! actions also own client session, network, inventory, and resource-pack side effects; replacing
//! that explicit dispatch with this stack would require rewriting those seams and would not make
//! the vanilla parity path more faithful.
//!
//! # Why a stack, when vanilla has one field
//!
//! Vanilla keeps a single `Minecraft.currentScreen` and each screen remembers its own
//! `parentScreen` so it can put the old one back. That works, but it means "go back" is
//! reimplemented (and occasionally mis-implemented) in every screen, and an overlay that wants
//! to draw *over* a live screen — the in-game HUD under a chat box, a confirmation dialog over
//! the options menu — has nowhere to live.
//!
//! [`ScreenStack`] makes the relationship explicit:
//!
//! * **Drawing** walks the stack bottom-to-top, starting at the topmost [`Screen::opaque`]
//!   screen. Everything above it draws on top.
//! * **Input** goes to the top screen. It falls through to the next one down only when the top
//!   screen is not [`Screen::modal`], which is how a HUD overlay coexists with the world.
//! * **Escape** pops, and popping the last screen leaves the stack empty — which is vanilla's
//!   `displayGuiScreen(null)`, i.e. back to the game.
//!
//! # Focus and keyboard navigation
//!
//! 1.8.9 has *no* keyboard navigation: buttons are mouse-only and only `GuiTextField` has a
//! focus concept. [`FocusRing`] adds the missing piece — Tab / Shift-Tab / arrows to move
//! between widgets and Enter or Space to activate — without changing how the mouse behaves, so
//! a screen that ignores it behaves exactly like vanilla.

use crate::font::FontRenderer;
use crate::widget::draw::DrawList;
use crate::widget::keys::{self, Modifiers};
use crate::widget::ScaledResolution;

/// What a screen wants the stack to do after handling an event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScreenAction {
    /// Nothing; the event was handled (or ignored) and the stack is unchanged.
    #[default]
    None,
    /// Close this screen.
    Pop,
    /// Close every screen, returning to the game.
    CloseAll,
}

/// Everything a screen is handed for one frame or one event.
#[derive(Debug, Clone, Copy)]
pub struct ScreenCtx {
    /// The current scaled surface. `width`/`height` below are its integer size.
    pub resolution: ScaledResolution,
    /// Scaled width, i.e. `resolution.scaled_width()`.
    pub width: i32,
    /// Scaled height.
    pub height: i32,
    /// Fractional tick for animation, `0.0..1.0`.
    pub partial_ticks: f32,
}

impl ScreenCtx {
    /// Build from a resolution.
    #[must_use]
    pub fn new(resolution: ScaledResolution, partial_ticks: f32) -> Self {
        Self {
            resolution,
            width: resolution.scaled_width(),
            height: resolution.scaled_height(),
            partial_ticks,
        }
    }
}

/// One screen.
///
/// Every method has a default, so a screen implements only what it uses. The stack never calls
/// [`Screen::init`] twice without an intervening resize or push.
pub trait Screen {
    /// A stable name, for the F3 overlay and for [`ScreenStack::find`].
    fn name(&self) -> &'static str {
        "screen"
    }

    /// Lay out widgets. Called on push and on every resize.
    // java: GuiScreen.java:575 (initGui)
    fn init(&mut self, _ctx: &ScreenCtx) {}

    /// Advance animations. Called once per client tick.
    // java: GuiScreen.java:654 (updateScreen)
    fn tick(&mut self) {}

    /// Draw. Called once per frame, bottom-to-top.
    // java: GuiScreen.java:87 (drawScreen)
    fn draw(
        &mut self,
        _d: &mut DrawList,
        _font: &mut FontRenderer,
        _ctx: &ScreenCtx,
        _mouse: (i32, i32),
    ) {
    }

    /// A key was pressed. `typed` is the produced character, `key` an LWJGL scan code.
    ///
    /// The default handles Escape by popping, which is what `GuiScreen.keyTyped` does.
    ///
    /// This is handed no [`FontRenderer`], which is fine for every screen that does not hold a
    /// [`TextField`](crate::widget::TextField) — pick this one. A screen that does should
    /// override [`Screen::key_typed_with_font`] instead, so it can re-measure the field in the
    /// same call rather than queueing the key for [`Screen::draw`] to apply next frame.
    /// [`ScreenStack`] always calls `key_typed_with_font`, whose default just forwards here, so
    /// implementing only this method is never observably different from today.
    // java: GuiScreen.java:104 (keyTyped)
    fn key_typed(&mut self, _typed: Option<char>, key: i32, _mods: Modifiers) -> ScreenAction {
        if key == keys::ESCAPE {
            ScreenAction::Pop
        } else {
            ScreenAction::None
        }
    }

    /// [`Screen::key_typed`], but also handed the font a [`TextField`](crate::widget::TextField)
    /// needs to re-measure its visible window after an edit.
    ///
    /// [`ScreenStack::key_typed`] calls this — never plain [`Screen::key_typed`] — so a screen
    /// holding a text field has everything it needs to apply a keystroke immediately instead of
    /// queueing it for the next [`Screen::draw`] (the one-frame-lag workaround
    /// `CreativeScreen` used before this method existed). The default just forwards to
    /// [`Screen::key_typed`] and drops the font, so a screen with nothing to measure never has to
    /// know this method exists.
    // java: no equivalent — `GuiScreen` always had `fontRendererObj` to reach for; this closes
    // the gap `ScreenStack` opened by not handing `key_typed` one directly.
    fn key_typed_with_font(
        &mut self,
        typed: Option<char>,
        key: i32,
        mods: Modifiers,
        font: &FontRenderer,
    ) -> ScreenAction {
        let _ = font;
        self.key_typed(typed, key, mods)
    }

    /// Give a screen a chance to route a keyboard event through a [`FocusRing`].
    ///
    /// The return value distinguishes "this screen has no focus ring" (`None`) from "the ring
    /// consumed the event" (`Some(action)`). That distinction matters for a non-modal overlay:
    /// only an event the overlay actually consumes may stop the stack falling through. Screens
    /// that remain mouse-only, including all untouched vanilla ports, keep the default.
    fn focus_key_typed(&mut self, _key: i32, _mods: Modifiers) -> Option<ScreenAction> {
        None
    }

    /// A mouse button went down, in GUI pixels.
    // java: GuiScreen.java:499 (mouseClicked)
    fn mouse_clicked(&mut self, _x: i32, _y: i32, _button: i32) -> ScreenAction {
        ScreenAction::None
    }

    /// A mouse button came up.
    // java: GuiScreen.java:520 (mouseReleased)
    fn mouse_released(&mut self, _x: i32, _y: i32, _button: i32) {}

    /// The mouse moved with a button held.
    // java: GuiScreen.java:533 (mouseClickMove)
    fn mouse_dragged(&mut self, _x: i32, _y: i32, _button: i32, _held_ms: u64) {}

    /// The wheel turned. Positive is "away from the user".
    fn mouse_wheel(&mut self, _x: i32, _y: i32, _delta: f32) {}

    /// The screen is being removed.
    // java: GuiScreen.java:661 (onGuiClosed)
    fn closed(&mut self) {}

    /// Whether this screen swallows input rather than letting it reach the screen below.
    ///
    /// Default `true`, which is every vanilla screen. A HUD overlay returns `false`.
    fn modal(&self) -> bool {
        true
    }

    /// Whether this screen covers whatever is under it, so nothing below need be drawn.
    ///
    /// Default `false`: vanilla's screens draw a *translucent* gradient over the world, so the
    /// thing below still has to be painted.
    fn opaque(&self) -> bool {
        false
    }

    /// Whether the game clock stops while this screen is up. Multiplayer ignores it (the multiplayer-only rule: there is no integrated server to pause), but the F3 overlay reports it.
    // java: GuiScreen.java:708 (doesGuiPauseGame)
    fn pauses_game(&self) -> bool {
        true
    }

    /// Whether movement keys still reach the player while this screen is up — vanilla's
    /// `allowUserInput`, which the player-inventory and creative-container screens set. Chat
    /// inherits `false` from `GuiScreen` in 1.8.9.
    fn allows_user_input(&self) -> bool {
        false
    }
}

/// A stack of screens, innermost last.
#[derive(Default)]
#[doc(hidden)]
pub struct ScreenStack {
    screens: Vec<Box<dyn Screen>>,
    resolution: Option<ScaledResolution>,
}

impl std::fmt::Debug for ScreenStack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScreenStack")
            .field(
                "screens",
                &self.screens.iter().map(|s| s.name()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl ScreenStack {
    /// An empty stack — "in the game, no screen open".
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// How many screens are open.
    #[must_use]
    pub fn len(&self) -> usize {
        self.screens.len()
    }

    /// Whether no screen is open.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.screens.is_empty()
    }

    /// The screens' names, bottom to top.
    #[must_use]
    pub fn names(&self) -> Vec<&'static str> {
        self.screens.iter().map(|s| s.name()).collect()
    }

    /// Whether a screen with this name is anywhere in the stack.
    #[must_use]
    pub fn find(&self, name: &str) -> bool {
        self.screens.iter().any(|s| s.name() == name)
    }

    /// The top screen.
    #[must_use]
    pub fn top(&self) -> Option<&dyn Screen> {
        self.screens.last().map(std::convert::AsRef::as_ref)
    }

    /// The top screen, mutably.
    pub fn top_mut(&mut self) -> Option<&mut (dyn Screen + 'static)> {
        self.screens.last_mut().map(std::convert::AsMut::as_mut)
    }

    /// Remember the resolution so pushes and resizes can call [`Screen::init`].
    pub fn set_resolution(&mut self, resolution: ScaledResolution) {
        self.resolution = Some(resolution);
        let ctx = ScreenCtx::new(resolution, 0.0);
        for screen in &mut self.screens {
            screen.init(&ctx);
        }
    }

    /// Push a screen and initialise it.
    pub fn push(&mut self, mut screen: Box<dyn Screen>) {
        if let Some(res) = self.resolution {
            screen.init(&ScreenCtx::new(res, 0.0));
        }
        self.screens.push(screen);
    }

    /// Pop the top screen, calling [`Screen::closed`] on it.
    pub fn pop(&mut self) -> Option<Box<dyn Screen>> {
        let mut screen = self.screens.pop()?;
        screen.closed();
        Some(screen)
    }

    /// Replace the whole stack with one screen. Vanilla's `displayGuiScreen`.
    pub fn replace(&mut self, screen: Box<dyn Screen>) {
        self.clear();
        self.push(screen);
    }

    /// Close everything, innermost first.
    pub fn clear(&mut self) {
        while self.pop().is_some() {}
    }

    /// The index of the lowest screen that has to be drawn: the topmost opaque one, or 0.
    #[must_use]
    pub fn first_visible(&self) -> usize {
        self.screens.iter().rposition(|s| s.opaque()).unwrap_or(0)
    }

    /// Whether the screen below the top one still receives input.
    #[must_use]
    pub fn top_is_modal(&self) -> bool {
        self.top().is_none_or(Screen::modal)
    }

    /// Whether the game clock should stop.
    #[must_use]
    pub fn pauses_game(&self) -> bool {
        self.top().is_some_and(Screen::pauses_game)
    }

    /// Whether movement keys still reach the player.
    #[must_use]
    pub fn allows_user_input(&self) -> bool {
        self.top().is_none_or(Screen::allows_user_input)
    }

    /// Tick every open screen, top-down.
    pub fn tick(&mut self) {
        for screen in self.screens.iter_mut().rev() {
            screen.tick();
        }
    }

    /// Draw from the topmost opaque screen upwards.
    pub fn draw(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        mouse: (i32, i32),
    ) {
        let from = self.first_visible();
        for screen in &mut self.screens[from..] {
            screen.draw(d, font, ctx, mouse);
        }
    }

    /// Deliver a key press, falling through non-modal screens.
    ///
    /// `font` reaches every screen through [`Screen::key_typed_with_font`], so one holding a
    /// [`TextField`](crate::widget::TextField) can re-measure it within this same call.
    pub fn key_typed(
        &mut self,
        typed: Option<char>,
        key: i32,
        mods: Modifiers,
        font: &FontRenderer,
    ) {
        let mut index = self.screens.len();
        while index > 0 {
            index -= 1;
            if let Some(action) = self.screens[index].focus_key_typed(key, mods) {
                let modal = self.screens[index].modal();
                // `Some(None)` is a consumed focus event, not "no-op, keep falling through".
                // This distinction is what lets a non-modal HUD own Tab/Enter without leaking
                // those keys to the world screen underneath it.
                let consumed = matches!(action, ScreenAction::None);
                if self.apply(index, action) || modal || consumed {
                    return;
                }
                continue;
            }
            let action = self.screens[index].key_typed_with_font(typed, key, mods, font);
            let modal = self.screens[index].modal();
            if self.apply(index, action) || modal {
                return;
            }
        }
    }

    /// Deliver a mouse press, falling through non-modal screens.
    pub fn mouse_clicked(&mut self, x: i32, y: i32, button: i32) {
        let mut index = self.screens.len();
        while index > 0 {
            index -= 1;
            let action = self.screens[index].mouse_clicked(x, y, button);
            let modal = self.screens[index].modal();
            if self.apply(index, action) || modal {
                return;
            }
        }
    }

    /// Deliver a mouse release to the top screen only.
    pub fn mouse_released(&mut self, x: i32, y: i32, button: i32) {
        if let Some(top) = self.top_mut() {
            top.mouse_released(x, y, button);
        }
    }

    /// Deliver a drag to the top screen only.
    pub fn mouse_dragged(&mut self, x: i32, y: i32, button: i32, held_ms: u64) {
        if let Some(top) = self.top_mut() {
            top.mouse_dragged(x, y, button, held_ms);
        }
    }

    /// Deliver a wheel event to the top screen only.
    pub fn mouse_wheel(&mut self, x: i32, y: i32, delta: f32) {
        if let Some(top) = self.top_mut() {
            top.mouse_wheel(x, y, delta);
        }
    }

    /// Apply a screen's requested action. Returns whether the stack changed.
    fn apply(&mut self, index: usize, action: ScreenAction) -> bool {
        match action {
            ScreenAction::None => false,
            ScreenAction::Pop => {
                if index < self.screens.len() {
                    let mut screen = self.screens.remove(index);
                    screen.closed();
                }
                true
            }
            ScreenAction::CloseAll => {
                self.clear();
                true
            }
        }
    }
}

/// Keyboard focus over a fixed number of widgets.
///
/// Not a vanilla concept — 1.8.9's GUI is mouse-only — so this is additive. It tracks an index
/// into whatever list of widgets the screen keeps, skips the ones it is told are unfocusable,
/// and turns Tab / Shift-Tab / arrow keys into moves.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FocusRing {
    focused: Option<usize>,
    len: usize,
}

impl FocusRing {
    /// A ring over `len` widgets with nothing focused.
    #[must_use]
    pub fn new(len: usize) -> Self {
        Self { focused: None, len }
    }

    /// Change the widget count, dropping focus if it no longer exists.
    pub fn set_len(&mut self, len: usize) {
        self.len = len;
        if self.focused.is_some_and(|i| i >= len) {
            self.focused = None;
        }
    }

    /// How many widgets the ring covers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the ring covers nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The focused widget, if any.
    #[must_use]
    pub fn focused(&self) -> Option<usize> {
        self.focused
    }

    /// Focus a specific widget, or clear focus with `None`.
    pub fn focus(&mut self, index: Option<usize>) {
        self.focused = index.filter(|i| *i < self.len);
    }

    /// Move focus by `delta`, wrapping, skipping widgets `focusable` rejects.
    ///
    /// Returns the new focus. With nothing focusable this leaves focus alone rather than
    /// looping forever.
    pub fn advance(&mut self, delta: isize, focusable: impl Fn(usize) -> bool) -> Option<usize> {
        if self.len == 0 || delta == 0 {
            return self.focused;
        }
        let len = self.len as isize;
        let start = match self.focused {
            Some(i) => i as isize,
            // Tab with nothing focused enters at the first widget (or the last, going back).
            None if delta > 0 => -1,
            None => 0,
        };
        for step in 1..=len {
            let candidate = (start + delta * step).rem_euclid(len) as usize;
            if focusable(candidate) {
                self.focused = Some(candidate);
                return self.focused;
            }
        }
        self.focused
    }

    /// Interpret a key as a focus move. Returns whether it was consumed.
    ///
    /// Tab and Down move forward, Shift-Tab and Up move back. Left/Right are deliberately
    /// **not** bound: a focused slider or text field wants them.
    pub fn key_typed(
        &mut self,
        key: i32,
        mods: Modifiers,
        focusable: impl Fn(usize) -> bool,
    ) -> bool {
        match key {
            keys::TAB => {
                self.advance(if mods.shift { -1 } else { 1 }, focusable);
                true
            }
            keys::DOWN => {
                self.advance(1, focusable);
                true
            }
            keys::UP => {
                self.advance(-1, focusable);
                true
            }
            _ => false,
        }
    }

    /// Whether `key` activates the focused widget: Enter, numpad Enter, or Space.
    #[must_use]
    pub fn is_activate(key: i32) -> bool {
        matches!(key, keys::ENTER | keys::NUMPAD_ENTER | keys::SPACE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[derive(Default)]
    struct Log {
        events: Vec<String>,
    }

    struct Probe {
        name: &'static str,
        modal: bool,
        opaque: bool,
        log: Rc<RefCell<Log>>,
        on_key: ScreenAction,
        on_click: ScreenAction,
    }

    impl Probe {
        fn new(name: &'static str, log: &Rc<RefCell<Log>>) -> Self {
            Self {
                name,
                modal: true,
                opaque: false,
                log: Rc::clone(log),
                on_key: ScreenAction::None,
                on_click: ScreenAction::None,
            }
        }
        fn note(&self, what: &str) {
            self.log
                .borrow_mut()
                .events
                .push(format!("{}:{what}", self.name));
        }
    }

    impl Screen for Probe {
        fn name(&self) -> &'static str {
            self.name
        }
        fn init(&mut self, _ctx: &ScreenCtx) {
            self.note("init");
        }
        fn tick(&mut self) {
            self.note("tick");
        }
        fn draw(
            &mut self,
            _d: &mut DrawList,
            _f: &mut FontRenderer,
            _c: &ScreenCtx,
            _m: (i32, i32),
        ) {
            self.note("draw");
        }
        fn key_typed(&mut self, _t: Option<char>, _k: i32, _m: Modifiers) -> ScreenAction {
            self.note("key");
            self.on_key
        }
        fn mouse_clicked(&mut self, _x: i32, _y: i32, _b: i32) -> ScreenAction {
            self.note("click");
            self.on_click
        }
        fn mouse_released(&mut self, _x: i32, _y: i32, _b: i32) {
            self.note("release");
        }
        fn mouse_dragged(&mut self, _x: i32, _y: i32, _b: i32, _held_ms: u64) {
            self.note("drag");
        }
        fn mouse_wheel(&mut self, _x: i32, _y: i32, _delta: f32) {
            self.note("wheel");
        }
        fn closed(&mut self) {
            self.note("closed");
        }
        fn modal(&self) -> bool {
            self.modal
        }
        fn opaque(&self) -> bool {
            self.opaque
        }
    }

    fn stack() -> (ScreenStack, Rc<RefCell<Log>>) {
        let mut s = ScreenStack::new();
        s.set_resolution(ScaledResolution::new(854, 480, 0, false));
        (s, Rc::new(RefCell::new(Log::default())))
    }

    fn events(log: &Rc<RefCell<Log>>) -> Vec<String> {
        log.borrow().events.clone()
    }

    #[test]
    fn pushing_initialises_and_popping_closes() {
        let (mut s, log) = stack();
        s.push(Box::new(Probe::new("a", &log)));
        assert_eq!(events(&log), ["a:init"]);
        assert_eq!(s.len(), 1);
        s.pop();
        assert_eq!(events(&log), ["a:init", "a:closed"]);
        assert!(s.is_empty());
    }

    #[test]
    fn a_resize_reinitialises_every_open_screen() {
        let (mut s, log) = stack();
        s.push(Box::new(Probe::new("a", &log)));
        s.push(Box::new(Probe::new("b", &log)));
        s.set_resolution(ScaledResolution::new(1920, 1080, 0, false));
        assert_eq!(events(&log), ["a:init", "b:init", "a:init", "b:init"]);
    }

    #[test]
    fn drawing_starts_at_the_topmost_opaque_screen() {
        let (mut s, log) = stack();
        s.push(Box::new(Probe::new("world", &log)));
        let mut menu = Probe::new("menu", &log);
        menu.opaque = true;
        s.push(Box::new(menu));
        s.push(Box::new(Probe::new("dialog", &log)));
        assert_eq!(s.first_visible(), 1);

        let mut d = DrawList::new();
        let mut f = crate::widget::test_support::uniform_font();
        let ctx = ScreenCtx::new(ScaledResolution::new(854, 480, 0, false), 0.0);
        log.borrow_mut().events.clear();
        s.draw(&mut d, &mut f, &ctx, (0, 0));
        assert_eq!(events(&log), ["menu:draw", "dialog:draw"]);
    }

    #[test]
    fn a_modal_screen_swallows_input() {
        let (mut s, log) = stack();
        s.push(Box::new(Probe::new("under", &log)));
        s.push(Box::new(Probe::new("over", &log)));
        log.borrow_mut().events.clear();
        s.key_typed(
            None,
            42,
            Modifiers::NONE,
            &crate::widget::test_support::uniform_font(),
        );
        assert_eq!(events(&log), ["over:key"]);
        s.mouse_clicked(1, 1, 0);
        assert_eq!(events(&log), ["over:key", "over:click"]);
    }

    /// Pointer methods are intentionally separate in the vanilla API. This matrix is a biting
    /// control for a hand-dispatch mutant: all three event kinds must reach the top screen, and
    /// none may leak to a modal screen underneath it.
    #[test]
    fn pointer_dispatch_has_one_topmost_owner() {
        let (mut s, log) = stack();
        let mut under = Probe::new("under", &log);
        under.modal = false;
        s.push(Box::new(under));
        s.push(Box::new(Probe::new("over", &log)));
        log.borrow_mut().events.clear();
        s.mouse_released(1, 1, 0);
        s.mouse_dragged(1, 1, 0, 16);
        s.mouse_wheel(1, 1, 1.0);
        assert_eq!(events(&log), ["over:release", "over:drag", "over:wheel"]);
    }

    #[test]
    fn a_non_modal_overlay_lets_input_through() {
        let (mut s, log) = stack();
        s.push(Box::new(Probe::new("under", &log)));
        let mut hud = Probe::new("hud", &log);
        hud.modal = false;
        s.push(Box::new(hud));
        log.borrow_mut().events.clear();
        s.key_typed(
            None,
            42,
            Modifiers::NONE,
            &crate::widget::test_support::uniform_font(),
        );
        assert_eq!(events(&log), ["hud:key", "under:key"]);
        assert!(!s.top_is_modal());
    }

    #[test]
    fn fall_through_stops_as_soon_as_a_screen_acts() {
        let (mut s, log) = stack();
        s.push(Box::new(Probe::new("under", &log)));
        let mut hud = Probe::new("hud", &log);
        hud.modal = false;
        hud.on_key = ScreenAction::Pop;
        s.push(Box::new(hud));
        log.borrow_mut().events.clear();
        s.key_typed(
            None,
            42,
            Modifiers::NONE,
            &crate::widget::test_support::uniform_font(),
        );
        assert_eq!(events(&log), ["hud:key", "hud:closed"]);
        assert_eq!(s.names(), ["under"]);
    }

    #[test]
    fn escape_pops_by_default_and_empties_the_stack() {
        let (mut s, log) = stack();
        s.push(Box::new(Probe::new("a", &log)));
        s.push(Box::new(Probe::new("b", &log)));
        // The probe overrides key_typed, so use a screen that does not.
        struct Plain;
        impl Screen for Plain {
            fn name(&self) -> &'static str {
                "plain"
            }
        }
        s.push(Box::new(Plain));
        assert_eq!(s.len(), 3);
        s.key_typed(
            None,
            keys::ESCAPE,
            Modifiers::NONE,
            &crate::widget::test_support::uniform_font(),
        );
        assert_eq!(s.names(), ["a", "b"]);
    }

    #[test]
    fn close_all_empties_the_stack_innermost_first() {
        let (mut s, log) = stack();
        s.push(Box::new(Probe::new("a", &log)));
        let mut b = Probe::new("b", &log);
        b.on_click = ScreenAction::CloseAll;
        s.push(Box::new(b));
        log.borrow_mut().events.clear();
        s.mouse_clicked(0, 0, 0);
        assert!(s.is_empty());
        assert_eq!(events(&log), ["b:click", "b:closed", "a:closed"]);
    }

    #[test]
    fn replace_swaps_the_whole_stack() {
        let (mut s, log) = stack();
        s.push(Box::new(Probe::new("a", &log)));
        s.push(Box::new(Probe::new("b", &log)));
        s.replace(Box::new(Probe::new("c", &log)));
        assert_eq!(s.names(), ["c"]);
        assert!(s.find("c"));
        assert!(!s.find("a"));
    }

    #[test]
    fn ticks_go_top_down_to_every_screen() {
        let (mut s, log) = stack();
        s.push(Box::new(Probe::new("a", &log)));
        s.push(Box::new(Probe::new("b", &log)));
        log.borrow_mut().events.clear();
        s.tick();
        assert_eq!(events(&log), ["b:tick", "a:tick"]);
    }

    #[test]
    fn an_empty_stack_answers_the_flags_like_the_bare_game() {
        let s = ScreenStack::new();
        assert!(s.top().is_none());
        assert!(s.top_is_modal(), "nothing to fall through to");
        assert!(!s.pauses_game());
        assert!(
            s.allows_user_input(),
            "no screen means the player has control"
        );
        assert_eq!(s.first_visible(), 0);
    }

    // ---------------------------------------------------------------- focus

    #[test]
    fn tab_walks_focus_forward_and_wraps() {
        let mut ring = FocusRing::new(3);
        assert_eq!(ring.focused(), None);
        assert_eq!(ring.advance(1, |_| true), Some(0));
        assert_eq!(ring.advance(1, |_| true), Some(1));
        assert_eq!(ring.advance(1, |_| true), Some(2));
        assert_eq!(ring.advance(1, |_| true), Some(0), "wraps");
    }

    #[test]
    fn shift_tab_walks_back_and_enters_at_the_end() {
        let mut ring = FocusRing::new(3);
        assert_eq!(ring.advance(-1, |_| true), Some(2));
        assert_eq!(ring.advance(-1, |_| true), Some(1));
        assert_eq!(ring.advance(-1, |_| true), Some(0));
        assert_eq!(ring.advance(-1, |_| true), Some(2));
    }

    #[test]
    fn unfocusable_widgets_are_skipped() {
        let mut ring = FocusRing::new(4);
        let ok = |i: usize| i == 1 || i == 3;
        assert_eq!(ring.advance(1, ok), Some(1));
        assert_eq!(ring.advance(1, ok), Some(3));
        assert_eq!(ring.advance(1, ok), Some(1));
        assert_eq!(ring.advance(-1, ok), Some(3));
    }

    #[test]
    fn a_ring_with_nothing_focusable_terminates() {
        let mut ring = FocusRing::new(5);
        assert_eq!(ring.advance(1, |_| false), None);
        ring.focus(Some(2));
        assert_eq!(ring.advance(1, |_| false), Some(2), "focus is left alone");
    }

    #[test]
    fn an_empty_ring_is_inert() {
        let mut ring = FocusRing::new(0);
        assert!(ring.is_empty());
        assert_eq!(ring.advance(1, |_| true), None);
        ring.focus(Some(0));
        assert_eq!(ring.focused(), None, "out of range focus is refused");
    }

    #[test]
    fn shrinking_the_ring_drops_a_now_invalid_focus() {
        let mut ring = FocusRing::new(4);
        ring.focus(Some(3));
        ring.set_len(2);
        assert_eq!(ring.focused(), None);
        ring.focus(Some(1));
        ring.set_len(2);
        assert_eq!(ring.focused(), Some(1), "still in range");
    }

    /// A ring is useful only when the production stack asks the screen about it. This probe is a
    /// deliberately tiny screen-level adapter: a mutant that leaves `ScreenStack::key_typed`
    /// on the old direct-dispatch path will fail because Tab never reaches the ring and Enter
    /// never activates its focused widget.
    #[test]
    fn stack_drives_focus_ring_before_normal_key_dispatch() {
        struct FocusProbe {
            ring: FocusRing,
            log: Rc<RefCell<Vec<&'static str>>>,
            modal: bool,
        }
        impl Screen for FocusProbe {
            fn name(&self) -> &'static str {
                "focus"
            }
            fn focus_key_typed(&mut self, key: i32, mods: Modifiers) -> Option<ScreenAction> {
                if self.ring.key_typed(key, mods, |_| true) {
                    self.log.borrow_mut().push("focus");
                    return Some(ScreenAction::None);
                }
                if FocusRing::is_activate(key) {
                    if self.ring.focused().is_some() {
                        self.log.borrow_mut().push("activate");
                    }
                    return Some(ScreenAction::None);
                }
                None
            }
            fn key_typed(
                &mut self,
                _typed: Option<char>,
                _key: i32,
                _mods: Modifiers,
            ) -> ScreenAction {
                self.log.borrow_mut().push("normal");
                ScreenAction::None
            }
            fn modal(&self) -> bool {
                self.modal
            }
        }

        let (mut stack, _) = stack();
        let log = Rc::new(RefCell::new(Vec::new()));
        let font = crate::widget::test_support::uniform_font();
        stack.push(Box::new(FocusProbe {
            ring: FocusRing::new(2),
            log: Rc::clone(&log),
            modal: true,
        }));
        stack.key_typed(None, keys::TAB, Modifiers::NONE, &font);
        stack.key_typed(None, keys::ENTER, Modifiers::NONE, &font);
        assert_eq!(*log.borrow(), ["focus", "activate"]);
    }

    #[test]
    fn a_non_modal_focus_ring_owns_its_consumed_key() {
        struct FocusOverlay {
            ring: FocusRing,
            log: Rc<RefCell<Vec<&'static str>>>,
        }
        impl Screen for FocusOverlay {
            fn focus_key_typed(&mut self, key: i32, mods: Modifiers) -> Option<ScreenAction> {
                self.ring
                    .key_typed(key, mods, |_| true)
                    .then_some(ScreenAction::None)
            }
            fn modal(&self) -> bool {
                false
            }
            fn key_typed(
                &mut self,
                _typed: Option<char>,
                _key: i32,
                _mods: Modifiers,
            ) -> ScreenAction {
                self.log.borrow_mut().push("overlay-normal");
                ScreenAction::None
            }
        }
        struct Under {
            log: Rc<RefCell<Vec<&'static str>>>,
        }
        impl Screen for Under {
            fn key_typed(
                &mut self,
                _typed: Option<char>,
                _key: i32,
                _mods: Modifiers,
            ) -> ScreenAction {
                self.log.borrow_mut().push("under");
                ScreenAction::None
            }
        }

        let (mut stack, _) = stack();
        let log = Rc::new(RefCell::new(Vec::new()));
        stack.push(Box::new(Under {
            log: Rc::clone(&log),
        }));
        stack.push(Box::new(FocusOverlay {
            ring: FocusRing::new(1),
            log: Rc::clone(&log),
        }));
        stack.key_typed(
            None,
            keys::TAB,
            Modifiers::NONE,
            &crate::widget::test_support::uniform_font(),
        );
        assert!(
            log.borrow().is_empty(),
            "a consumed focus key must stop fallthrough"
        );
    }

    #[test]
    fn the_key_bindings_are_tab_and_the_vertical_arrows() {
        let mut ring = FocusRing::new(3);
        assert!(ring.key_typed(keys::TAB, Modifiers::NONE, |_| true));
        assert_eq!(ring.focused(), Some(0));
        assert!(ring.key_typed(keys::DOWN, Modifiers::NONE, |_| true));
        assert_eq!(ring.focused(), Some(1));
        assert!(ring.key_typed(keys::UP, Modifiers::NONE, |_| true));
        assert_eq!(ring.focused(), Some(0));
        assert!(ring.key_typed(keys::TAB, Modifiers::SHIFT, |_| true));
        assert_eq!(ring.focused(), Some(2));
        // Horizontal arrows belong to whatever is focused.
        assert!(!ring.key_typed(keys::LEFT, Modifiers::NONE, |_| true));
        assert!(!ring.key_typed(keys::RIGHT, Modifiers::NONE, |_| true));
        assert_eq!(ring.focused(), Some(2));
    }

    #[test]
    fn enter_and_space_activate() {
        assert!(FocusRing::is_activate(keys::ENTER));
        assert!(FocusRing::is_activate(keys::NUMPAD_ENTER));
        assert!(FocusRing::is_activate(keys::SPACE));
        assert!(!FocusRing::is_activate(keys::TAB));
        assert!(!FocusRing::is_activate(keys::ESCAPE));
    }
}
