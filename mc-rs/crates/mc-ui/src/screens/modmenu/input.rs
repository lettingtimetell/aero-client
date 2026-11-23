//! Input: mouse buttons, the pointer, the wheel, keys and the clipboard, and what a click on each
//! hit target does.
//!
//! A click on a mod's row selects it
//! (its switch still toggles it; a right click pins it), typing goes to the search, ↑/↓ move
//! within a window and Tab between the windows, Enter opens, Space toggles, ←/→ change the
//! focused setting, and Esc steps back one level at a time.

use mc_mods::menu_rules;
use mc_mods::{ModEvent, ModSettings, OptionKind, OptionValue};

use super::details::nav_option;
use super::state::{Drag, Mode, NavRow, Pane, Target, View};
use super::widgets::{argb_to_hsva, display_step, set_slider_from_x, still_armed};
use super::{KeyOutcome, MenuInput, ModMenuScreen};
use crate::widget::keys::{self, Clipboard, Modifiers};

// LWJGL key codes the menu reacts to.
pub(super) const KEY_ESCAPE: i32 = 1;
pub(super) const KEY_BACK: i32 = 14;
pub(super) const KEY_TAB: i32 = 15;
pub(super) const KEY_RETURN: i32 = 28;
pub(super) const KEY_SPACE: i32 = 57;
pub(super) const KEY_UP: i32 = 200;
pub(super) const KEY_LEFT: i32 = 203;
pub(super) const KEY_RIGHT: i32 = 205;
pub(super) const KEY_DOWN: i32 = 208;

/// The longest search, in characters.
pub(super) const SEARCH_MAX_CHARS: usize = 40;

impl ModMenuScreen {
    // ---- input -----------------------------------------------------------------------

    /// A mouse button went down at the physical-pixel point `(x, y)`.
    pub fn mouse_pressed(&mut self, x: f32, y: f32, button: u8, input: &mut MenuInput<'_>) {
        self.cursor = (x, y);
        self.keyboard = false;
        if self.closing_since.is_some() {
            return;
        }
        if let Some((m, o)) = self.capture {
            // Keybind capture: any non-left button binds (vanilla encoding: button - 100); a
            // left click cancels, so clicking elsewhere backs out.
            if button != 0 {
                input
                    .settings
                    .set_value_at(m, o, OptionValue::Key(i32::from(button) - 100));
            }
            self.capture = None;
            return;
        }
        if self.menu_key_capture {
            // The menu key is keyboard-only: any click backs out of the capture.
            self.menu_key_capture = false;
            return;
        }
        if self.mode == Mode::HudEditor {
            self.editor_pressed(x, y, button, input);
            return;
        }
        self.text_focus = None;
        self.name_focus = None;
        // The Profiles view's fields too (a click on one refocuses it).
        self.profiles_blur();
        self.search_selected = false;
        let Some(target) = self.hit(x, y) else {
            return;
        };
        match (target, button) {
            (Target::Slider(m, o, track), 0) => {
                self.drag = Some(Drag::Slider(m, o, track));
                self.last_slider = Some((m, o));
                self.focus_option(m, o);
                set_slider_from_x(input.settings, m, o, track, x);
            }
            (Target::ColorSlider(m, o, channel, track), 0) => {
                self.drag = Some(Drag::ColorSlider(m, o, channel, track));
                self.set_color_channel(input.settings, m, o, channel, track, x);
            }
            _ => self.pressed = Some((target, button)),
        }
    }

    /// A mouse button went up.
    pub fn mouse_released(&mut self, x: f32, y: f32, button: u8, input: &mut MenuInput<'_>) {
        self.cursor = (x, y);
        if self.mode == Mode::HudEditor {
            self.editor_released(button, input);
            return;
        }
        if matches!(self.drag, Some(Drag::Slider(..) | Drag::ColorSlider(..))) && button == 0 {
            self.drag = None;
            return;
        }
        let Some((target, pressed_button)) = self.pressed.take() else {
            return;
        };
        if pressed_button != button || self.hit(x, y) != Some(target) {
            return;
        }
        self.activate(target, button, input);
    }

    /// The cursor is at `(x, y)` (physical pixels); called once per frame before `build`, it
    /// drives drags. A pointer that moved hides the keyboard focus ring.
    pub fn pointer_moved(&mut self, x: f32, y: f32, input: &mut MenuInput<'_>) {
        if (x - self.cursor.0).abs() > 0.5 || (y - self.cursor.1).abs() > 0.5 {
            self.keyboard = false;
        }
        self.cursor = (x, y);
        match self.drag {
            Some(Drag::Slider(m, o, track)) => set_slider_from_x(input.settings, m, o, track, x),
            Some(Drag::ColorSlider(m, o, channel, track)) => {
                self.set_color_channel(input.settings, m, o, channel, track, x);
            }
            Some(Drag::Hud { .. }) => self.editor_dragged(x, y, input),
            None => {
                if self.mode == Mode::HudEditor {
                    let (gx, gy) = (x / input.gui_scale, y / input.gui_scale);
                    self.hud_hover = input.hud.hit(gx, gy);
                    // The hovered element is the one the wheel resizes.
                    if let Some(slot) = self.hud_hover {
                        self.hud_grip = Some((slot, (gx, gy)));
                    }
                }
            }
        }
    }

    /// The wheel turned `delta` notches (positive is up) at `(x, y)`.
    pub fn scrolled(&mut self, x: f32, y: f32, delta: f32, input: &mut MenuInput<'_>) {
        if self.mode == Mode::HudEditor {
            let (gx, gy) = (x / input.gui_scale, y / input.gui_scale);
            // The element under the pointer, or the one the wheel is
            // still gripping: an element that shrinks out from under the pointer keeps
            // answering the wheel, so it can be scrolled back up.
            let Some(slot) = self.scroll_target(gx, gy, input.hud) else {
                self.hud_grip = None;
                return;
            };
            let step = if input.shift { 0.05 } else { 0.1 };
            if let Some(info) = input.hud.slots().get(slot) {
                let (m, e) = (info.mod_index, info.element_index);
                if let Some(current) = input.settings.placement_at(m, e) {
                    // Scale about the anchor: anchor and offset stay, so the element's own
                    // resolve (custom pivots included) keeps its pinned point still. The
                    // placement clamps the scale to `HUD_MIN_SCALE..=HUD_MAX_SCALE`.
                    let scale = ((current.scale + delta.signum() * step) * 20.0).round() / 20.0;
                    input.settings.set_placement_at(m, e, current.scaled(scale));
                }
            }
            self.hud_last = Some(slot);
            self.hud_grip = Some((slot, (gx, gy)));
            self.hud_wheel_at = Some(input.now);
            return;
        }
        let window = self
            .hits
            .iter()
            .rev()
            .find_map(|(rect, target)| match target {
                Target::Window(w) if rect.contains(x, y) => Some(*w),
                _ => None,
            });
        if let Some(w) = window.filter(|&w| w < self.scroll_target.len()) {
            self.scroll_target[w] =
                (self.scroll_target[w] - delta * 42.0).clamp(0.0, self.max_scroll[w]);
        }
    }

    /// A key went down. `typed` is the character it produced, if any.
    ///
    /// In order: a keybind capture takes the key, then the menu-key capture, then a focused
    /// text field; in the HUD editor Esc goes back and the menu key closes. Otherwise Esc steps
    /// back (the colour editor, the search, window 3 to window 2) and then closes; the menu key
    /// closes (unless it types a character into a search in progress); Tab and ↑/↓ move the
    /// keyboard focus, Enter opens, Space toggles, ←/→ change the focused setting (else the
    /// hovered or last-dragged slider); Backspace and characters edit the search.
    #[allow(clippy::too_many_lines)]
    pub fn key(
        &mut self,
        lwjgl: i32,
        typed: Option<char>,
        input: &mut MenuInput<'_>,
    ) -> KeyOutcome {
        let printable = typed.filter(|c| !c.is_control());
        if self.menu_key_capture {
            self.menu_key_capture = false;
            match lwjgl {
                KEY_ESCAPE => {}
                // Never unbound (nothing else opens the menu): Backspace restores Right Shift.
                KEY_BACK => {
                    input.settings.set_menu_key(mc_mods::DEFAULT_MENU_KEY);
                }
                code => {
                    input.settings.set_menu_key(code);
                }
            }
            return KeyOutcome::Handled;
        }
        if let Some(m) = self.name_focus {
            match lwjgl {
                KEY_ESCAPE | KEY_TAB => self.name_focus = None,
                KEY_RETURN => {
                    self.name_focus = None;
                    self.save_profile(m, input);
                }
                KEY_BACK => {
                    self.profile_name.pop();
                }
                _ => {
                    if let Some(c) = printable {
                        if self.profile_name.chars().count() < mc_mods::PROFILE_NAME_MAX {
                            self.profile_name.push(c);
                        }
                    }
                }
            }
            return KeyOutcome::Handled;
        }
        if let Some((m, o)) = self.capture {
            match lwjgl {
                KEY_ESCAPE => {}
                KEY_BACK => {
                    input.settings.set_value_at(m, o, OptionValue::Key(0));
                }
                code => {
                    input.settings.set_value_at(m, o, OptionValue::Key(code));
                }
            }
            self.capture = None;
            return KeyOutcome::Handled;
        }
        if let Some((m, o)) = self.text_focus {
            match lwjgl {
                KEY_ESCAPE | KEY_RETURN | KEY_TAB => self.text_focus = None,
                KEY_BACK => {
                    if let Some(OptionValue::Text(text)) = input.settings.value_at(m, o).cloned() {
                        let mut text = text.into_owned();
                        text.pop();
                        input
                            .settings
                            .set_value_at(m, o, OptionValue::Text(text.into()));
                    }
                }
                _ => {
                    if let Some(c) = typed.filter(|c| !c.is_control()) {
                        if let Some(OptionValue::Text(text)) =
                            input.settings.value_at(m, o).cloned()
                        {
                            let mut text = text.into_owned();
                            text.push(c);
                            input
                                .settings
                                .set_value_at(m, o, OptionValue::Text(text.into()));
                        }
                    }
                }
            }
            return KeyOutcome::Handled;
        }
        if self.mode == Mode::HudEditor {
            match lwjgl {
                // The menu key closes the whole menu from the editor too; Esc goes back a level.
                code if code == input.settings.menu_key() => return KeyOutcome::Close,
                KEY_ESCAPE => self.leave_editor(input.now),
                KEY_UP | KEY_DOWN | KEY_LEFT | KEY_RIGHT => {
                    let step = if input.shift { 10.0 } else { 1.0 };
                    let (dx, dy) = match lwjgl {
                        KEY_UP => (0.0, -step),
                        KEY_DOWN => (0.0, step),
                        KEY_LEFT => (-step, 0.0),
                        _ => (step, 0.0),
                    };
                    self.nudge(dx, dy, input);
                }
                _ => {}
            }
            return KeyOutcome::Handled;
        }
        // Keys a view handles itself go here.
        if let Some(outcome) = self.profiles_key(lwjgl, typed, input) {
            return outcome;
        }
        match lwjgl {
            KEY_ESCAPE if self.color_editor.is_some() => {
                self.color_editor = None;
                KeyOutcome::Handled
            }
            KEY_ESCAPE if !self.search.is_empty() => {
                self.clear_search();
                KeyOutcome::Handled
            }
            KEY_ESCAPE if self.return_to_hud => {
                self.enter_editor(input.now);
                KeyOutcome::Handled
            }
            KEY_ESCAPE if self.keyboard && self.pane == Pane::Details => {
                self.focus_pane(Pane::List);
                KeyOutcome::Handled
            }
            KEY_ESCAPE => KeyOutcome::Close,
            // A printable menu key (say M) types into a search in progress instead.
            code if code == input.settings.menu_key()
                && !(printable.is_some() && !self.search.is_empty()) =>
            {
                KeyOutcome::Close
            }
            KEY_TAB => {
                let next = if input.shift { 2 } else { 1 };
                self.focus_pane(Pane::ALL[(self.pane.index() + next) % Pane::ALL.len()]);
                KeyOutcome::Handled
            }
            KEY_UP | KEY_DOWN => {
                self.search_selected = false;
                self.move_cursor(if lwjgl == KEY_UP { -1 } else { 1 }, input);
                KeyOutcome::Handled
            }
            KEY_RETURN => {
                self.search_selected = false;
                self.press_enter(input);
                KeyOutcome::Handled
            }
            KEY_SPACE if self.search.is_empty() => {
                self.press_space(input);
                KeyOutcome::Handled
            }
            KEY_LEFT | KEY_RIGHT => {
                self.search_selected = false;
                let steps = if input.shift { 10 } else { 1 };
                let steps = if lwjgl == KEY_LEFT { -steps } else { steps };
                if !self.step_focused(steps, input) {
                    self.step_slider(steps, input.settings);
                }
                KeyOutcome::Handled
            }
            KEY_BACK => {
                if self.search_selected {
                    self.clear_search();
                } else {
                    self.search.pop();
                }
                KeyOutcome::Handled
            }
            keys::DELETE if self.search_selected => {
                self.clear_search();
                KeyOutcome::Handled
            }
            _ => {
                if let Some(c) = printable {
                    self.insert_search(c.encode_utf8(&mut [0; 4]));
                }
                KeyOutcome::Handled
            }
        }
    }

    /// Handle clipboard shortcuts before ordinary character input. Other fields retain their
    /// existing paste priority; Ctrl+A/C only apply while typing would reach the search.
    /// Key captures, the HUD editor and closing screens never access the clipboard.
    pub fn clipboard_shortcut(
        &mut self,
        lwjgl: i32,
        modifiers: Modifiers,
        clipboard: &mut dyn Clipboard,
        input: &mut MenuInput<'_>,
    ) -> bool {
        if modifiers.is_ctrl_combo(lwjgl, keys::V) && self.takes_paste() {
            self.paste(&clipboard.get(), input);
            // A failed/empty clipboard read must not type a literal V or clear a selection.
            return true;
        }
        if !self.typing_to_search() {
            return false;
        }
        if modifiers.is_ctrl_combo(lwjgl, keys::A) {
            self.search_selected = !self.search.is_empty();
            return true;
        }
        if modifiers.is_ctrl_combo(lwjgl, keys::C) {
            if self.search_selected && !self.search.is_empty() {
                clipboard.set(&self.search);
            }
            return true;
        }
        false
    }

    pub(super) fn clear_search(&mut self) {
        self.search.clear();
        self.search_selected = false;
        self.narrow = None;
    }

    /// Insert whole Unicode characters, replacing a Ctrl+A selection only once an allowed
    /// character is available. The search's original leading-space and length rules apply.
    pub(super) fn insert_search(&mut self, text: &str) {
        let mut len = if self.search_selected {
            0
        } else {
            self.search.chars().count()
        };
        let mut inserted = String::new();
        for c in text.chars().filter(|c| !c.is_control()) {
            if len >= SEARCH_MAX_CHARS {
                break;
            }
            if c == ' ' && len == 0 {
                continue;
            }
            inserted.push(c);
            len += 1;
        }
        if !inserted.is_empty() {
            if self.search_selected {
                self.clear_search();
            }
            self.search.push_str(&inserted);
        }
    }

    /// Whether Ctrl+V would paste now (the windows are up and no key capture is waiting for
    /// a key: a capture takes Ctrl+V as a key).
    #[must_use]
    pub fn takes_paste(&self) -> bool {
        self.mode == Mode::Menu
            && self.closing_since.is_none()
            && self.capture.is_none()
            && !self.menu_key_capture
    }

    /// Ctrl+V (the client reads the clipboard): `text` goes where typing would go: the focused
    /// text option (a chat macro), the profile name, else the search. Line breaks and tabs
    /// become spaces, other control characters are dropped, and each field keeps its own
    /// length limit. Returns whether something took it (nothing does in the HUD editor or
    /// during a key capture).
    pub fn paste(&mut self, text: &str, input: &mut MenuInput<'_>) -> bool {
        if !self.takes_paste() {
            return false;
        }
        let clean: String = text
            .trim_end_matches(['\r', '\n'])
            .chars()
            .filter_map(|c| match c {
                '\r' | '\n' | '\t' => Some(' '),
                c if c.is_control() => None,
                c => Some(c),
            })
            .collect();
        if clean.is_empty() {
            return false;
        }
        if let Some((m, o)) = self.text_focus {
            if let Some(OptionValue::Text(old)) = input.settings.value_at(m, o).cloned() {
                let mut joined = old.into_owned();
                joined.push_str(&clean);
                // The store truncates to the option's `max_len`.
                input
                    .settings
                    .set_value_at(m, o, OptionValue::Text(joined.into()));
            }
            return true;
        }
        // A focused Profiles field takes the paste.
        if self.profiles_paste(&clean) {
            return true;
        }
        if self.name_focus.is_none() {
            self.insert_search(&clean);
            return true;
        }
        let (field, max) = (&mut self.profile_name, mc_mods::PROFILE_NAME_MAX);
        for c in clean.chars() {
            if field.chars().count() >= max {
                break;
            }
            if !(c == ' ' && field.is_empty()) {
                field.push(c);
            }
        }
        true
    }

    /// Whether a typed character goes to the search right now: no text field or capture has
    /// focus and the windows are up.
    pub(super) fn typing_to_search(&self) -> bool {
        self.mode == Mode::Menu
            && self.closing_since.is_none()
            && self.text_focus.is_none()
            && self.name_focus.is_none()
            && !self.profiles_typing()
            && self.capture.is_none()
            && !self.menu_key_capture
    }

    // ---- the keyboard --------------------------------------------------------

    /// Give `pane` the keyboard, on the row that stands for what it shows (window 1: the open
    /// list; window 2: the selected mod; window 3: its first row).
    pub(super) fn focus_pane(&mut self, pane: Pane) {
        self.pane = pane;
        self.keyboard = true;
        let p = pane.index();
        let rows = &self.nav[p];
        let current = match pane {
            Pane::Sidebar => {
                let active = if self.searching() {
                    self.narrow.unwrap_or(View::AllMods)
                } else {
                    self.view.unwrap_or(View::MySetup)
                };
                rows.iter().position(|row| row.enter == Target::Nav(active))
            }
            Pane::List => rows.iter().position(|row| {
                matches!(row.enter, Target::ModRow(m) | Target::Found(m)
                    if Some(m) == self.selected)
            }),
            Pane::Details => None,
        };
        let valid = self.cursors[p].filter(|&c| c < rows.len());
        self.cursors[p] = current.or(valid).or((!rows.is_empty()).then_some(0));
        self.reveal_nav = Some(pane);
    }

    /// ↑/↓: the keyboard's row in its window moves by `delta`. In windows 1 and 2 what is
    /// picked follows it (the list, the mod in window 3). Up from window 1's first entry
    /// reaches the search field.
    fn move_cursor(&mut self, delta: i32, input: &mut MenuInput<'_>) {
        let p = self.pane.index();
        let len = self.nav[p].len();
        self.keyboard = true;
        if len == 0 {
            return;
        }
        let next = match self.cursors[p] {
            Some(0) if delta < 0 && self.pane == Pane::Sidebar => {
                self.cursors[p] = None;
                return;
            }
            Some(c) => {
                let c = i64::try_from(c).unwrap_or(0) + i64::from(delta);
                usize::try_from(c.clamp(0, i64::try_from(len - 1).unwrap_or(0))).unwrap_or(0)
            }
            None if delta < 0 => len - 1,
            None => 0,
        };
        self.cursors[p] = Some(next);
        self.reveal_nav = Some(self.pane);
        let row = self.nav[p][next];
        match (self.pane, row.enter) {
            (Pane::Sidebar, Target::Nav(view)) => self.open_view(view, input.settings),
            (Pane::List, Target::ModRow(m)) => self.select(m),
            (Pane::List, Target::Found(m)) => self.open_found(m, input),
            _ => {}
        }
        // Keep the keyboard where it was: opening a view or a mod must not move it.
        self.pane = Pane::ALL[p];
        self.cursors[p] = Some(next);
    }

    /// Enter: window 1 moves into the list, window 2 opens the mod (or a search result) in
    /// window 3, window 3 does what a click on the focused row does. While typing a search,
    /// Enter opens the best result.
    fn press_enter(&mut self, input: &mut MenuInput<'_>) {
        self.keyboard = true;
        let list = Pane::List.index();
        // While searching with no row of window 2 under the keyboard, Enter opens the search's
        // best result (the ranking: "zoom key" opens Zoom at its key).
        if self.searching() && self.cursors[list].is_none() {
            let typing =
                self.pane == Pane::Sidebar && self.cursors[Pane::Sidebar.index()].is_none();
            if typing || self.pane == Pane::List {
                if let Some(best) = self.best_target() {
                    self.enter_row(
                        NavRow {
                            enter: best,
                            space: None,
                            span: (0.0, 0.0),
                        },
                        input,
                    );
                }
                return;
            }
        }
        match self.pane {
            Pane::Sidebar => {
                if let Some(row) = self.cursor_row(Pane::Sidebar) {
                    self.activate(row.enter, 0, input);
                }
                self.focus_pane(Pane::List);
            }
            Pane::List => {
                let row = self
                    .cursor_row(Pane::List)
                    .or_else(|| self.nav[list].first().copied());
                if let Some(row) = row {
                    self.enter_row(row, input);
                }
            }
            Pane::Details => {
                if let Some(row) = self.cursor_row(Pane::Details) {
                    self.activate(row.enter, 0, input);
                    self.pane = Pane::Details;
                }
            }
        }
    }

    /// Enter on a window-2 row: a mod or a search result opens in window 3 (which takes the
    /// keyboard); anything else (a lane's row) is clicked.
    fn enter_row(&mut self, row: NavRow, input: &mut MenuInput<'_>) {
        self.activate(row.enter, 0, input);
        if matches!(row.enter, Target::ModRow(_) | Target::Found(_)) {
            self.focus_pane(Pane::Details);
        }
    }

    /// Space: toggles the focused row's switch (in window 2 a mod, in window 3 a setting), or
    /// the selected mod when no row has the keyboard.
    fn press_space(&mut self, input: &mut MenuInput<'_>) {
        let target = self
            .cursor_row(self.pane)
            .and_then(|row| row.space)
            .or_else(|| match self.pane {
                Pane::List | Pane::Sidebar => self.selected.map(Target::Switch),
                Pane::Details => None,
            });
        if let Some(target) = target {
            let pane = self.pane;
            self.activate(target, 0, input);
            self.pane = pane;
        }
    }

    /// ←/→ on window 3's focused row: a slider moves `steps` steps, a choice to its next or
    /// previous value, a switch off or on. Returns whether a row took it.
    fn step_focused(&mut self, steps: i32, input: &mut MenuInput<'_>) -> bool {
        if self.pane != Pane::Details || !self.keyboard {
            return false;
        }
        let Some(row) = self.cursor_row(Pane::Details) else {
            return false;
        };
        let settings = &mut *input.settings;
        match row.enter {
            Target::Switch(m) => {
                settings.set_enabled_at(m, steps > 0);
            }
            Target::Toggle(m, o) => {
                settings.set_value_at(m, o, OptionValue::Bool(steps > 0));
            }
            Target::Choice(m, o) => {
                let catalog = settings.catalog();
                let OptionKind::Choice { choices } = catalog[m].options[o].kind else {
                    return false;
                };
                if let Some(OptionValue::Choice(index)) = settings.value_at(m, o).cloned() {
                    let n = choices.len().max(1);
                    let next = if steps > 0 {
                        (index + 1) % n
                    } else {
                        (index + n - 1) % n
                    };
                    settings.set_value_at(m, o, OptionValue::Choice(next));
                }
            }
            Target::Label(m, o) => {
                if !self.step_slider_at(m, o, steps, settings) {
                    return false;
                }
            }
            _ => return false,
        }
        true
    }

    /// The row that has the keyboard in `pane`.
    fn cursor_row(&self, pane: Pane) -> Option<NavRow> {
        let p = pane.index();
        self.cursors[p].and_then(|c| self.nav[p].get(c)).copied()
    }

    /// Put window 3's keyboard row on setting `o` of mod `m` (a click there).
    fn focus_option(&mut self, m: usize, o: usize) {
        let details = Pane::Details.index();
        if let Some(index) = self.nav[details]
            .iter()
            .position(|row| nav_option(row.enter) == Some((m, o)))
        {
            self.cursors[details] = Some(index);
        }
        self.pane = Pane::Details;
    }

    /// The slider the arrow keys move: the one under the pointer, else the last one pressed.
    fn arrow_slider(&self) -> Option<(usize, usize)> {
        self.hits
            .iter()
            .rev()
            .find_map(|(rect, target)| match target {
                Target::Slider(m, o, _) if rect.contains(self.cursor.0, self.cursor.1) => {
                    Some((*m, *o))
                }
                _ => None,
            })
            .or(self.last_slider)
    }

    /// Move the arrow keys' slider `steps` of its step (continuous sliders: of the step their
    /// badge shows). The settings store clamps and snaps the result.
    fn step_slider(&mut self, steps: i32, settings: &mut ModSettings) {
        if let Some((m, o)) = self.arrow_slider() {
            self.step_slider_at(m, o, steps, settings);
        }
    }

    /// Move slider `o` of mod `m` by `steps` steps. Returns whether it is a slider.
    fn step_slider_at(
        &mut self,
        m: usize,
        o: usize,
        steps: i32,
        settings: &mut ModSettings,
    ) -> bool {
        let Some(option) = settings.catalog().get(m).and_then(|d| d.options.get(o)) else {
            return false;
        };
        let step = match option.kind {
            OptionKind::Slider { step, decimals, .. } => display_step(step, decimals),
            OptionKind::SliderLabeled { min, max, step, .. } => {
                if step > 0.0 {
                    step
                } else {
                    (max - min) / 100.0
                }
            }
            _ => return false,
        };
        if let Some(OptionValue::Float(v)) = settings.value_at(m, o).cloned() {
            settings.set_value_at(m, o, OptionValue::Float(v + step * steps as f32));
            self.last_slider = Some((m, o));
        }
        true
    }

    // ---- what a click does ----------------------------------------------------------------

    /// Open `view` in window 2 (from window 1). While searching it narrows the results instead
    /// (All mods: every result); Keybinds and Profiles end the search. Window 3 keeps its mod
    /// while the new list has it, else it shows the list's first mod.
    pub(super) fn open_view(&mut self, view: View, settings: &ModSettings) {
        let list = Pane::List.index();
        if self.searching() && !matches!(view, View::Keybinds | View::Profiles) {
            self.narrow = (view != View::AllMods).then_some(view);
            self.scroll_target[list] = 0.0;
            self.cursors[list] = None;
            return;
        }
        if self.searching() {
            self.clear_search();
        }
        self.set_view(view);
        let own_window = matches!(view, View::Keybinds | View::Profiles);
        let keep = self
            .selected
            .is_some_and(|m| own_window || self.view_lists(settings, view, m));
        if !keep {
            if let Some(first) = self.first_in_view(settings) {
                self.select(first);
            }
        }
    }

    /// Open a search result's mod in window 3, its matching settings flashed and the first of
    /// them scrolled into view.
    fn open_found(&mut self, m: usize, input: &MenuInput<'_>) {
        let options = self.found.options_of(m);
        self.select(m);
        self.reveal_option = options.first().copied();
        self.flash_options = options;
        self.flash_since = Some(input.now);
    }

    /// Put setting `o` of mod `m` back to its default (a paired row's partner too: a chat
    /// macro's key and text reset together).
    fn reset_option(&mut self, m: usize, o: usize, settings: &mut ModSettings) {
        let Some(descriptor) = settings.catalog().get(m).copied() else {
            return;
        };
        let Some(option) = descriptor.options.get(o) else {
            return;
        };
        settings.set_value_at(m, o, option.default.clone());
        for paired in menu_rules::PAIRED
            .iter()
            .filter(|paired| paired.mod_id == descriptor.id)
        {
            for (left, right) in paired.pairs {
                if *left == option.id {
                    if let Some(r) = descriptor.option_index(right) {
                        settings.set_value_at(m, r, descriptor.options[r].default.clone());
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    pub(super) fn activate(&mut self, target: Target, button: u8, input: &mut MenuInput<'_>) {
        let catalog = input.settings.catalog();
        let right = button == 1;
        let left = button == 0;
        match target {
            Target::Nav(view) => {
                if left {
                    self.open_view(view, input.settings);
                    self.pane = Pane::Sidebar;
                    let sidebar = Pane::Sidebar.index();
                    self.cursors[sidebar] = self.nav[sidebar]
                        .iter()
                        .position(|row| row.enter == Target::Nav(view));
                }
            }
            Target::ModRow(m) if right => {
                let on = !input.settings.pinned_at(m);
                input.settings.set_pinned_at(m, on);
            }
            Target::ModRow(m) => {
                if left {
                    self.select(m);
                    self.pane = Pane::List;
                    let list = Pane::List.index();
                    self.cursors[list] = self.nav[list]
                        .iter()
                        .position(|row| row.enter == Target::ModRow(m));
                }
            }
            Target::Found(m) => {
                if left {
                    self.open_found(m, input);
                    self.pane = Pane::List;
                }
            }
            Target::Switch(m) => {
                if left {
                    let on = !input.settings.enabled_at(m);
                    input.settings.set_enabled_at(m, on);
                }
            }
            Target::Pin(m) => {
                if left {
                    let on = !input.settings.pinned_at(m);
                    input.settings.set_pinned_at(m, on);
                }
            }
            Target::Label(m, o) | Target::Toggle(m, o) if right => {
                self.reset_option(m, o, input.settings);
                self.focus_option(m, o);
            }
            Target::Label(m, o) => self.focus_option(m, o),
            Target::ResetOption(m, o) => {
                if left {
                    self.reset_option(m, o, input.settings);
                }
            }
            Target::Tab(m, o, choice) => {
                if left {
                    input
                        .settings
                        .set_value_at(m, o, OptionValue::Choice(choice));
                    self.focus_option(m, o);
                }
            }
            Target::Group(m, g) => {
                if left {
                    if let Some(at) = self.toggled_groups.iter().position(|&k| k == (m, g)) {
                        self.toggled_groups.remove(at);
                    } else {
                        self.toggled_groups.push((m, g));
                    }
                }
            }
            Target::Toggle(m, o) => {
                if let Some(OptionValue::Bool(on)) = input.settings.value_at(m, o).cloned() {
                    input.settings.set_value_at(m, o, OptionValue::Bool(!on));
                }
                self.focus_option(m, o);
            }
            Target::Choice(m, o) => {
                if let (Some(OptionValue::Choice(index)), OptionKind::Choice { choices }) = (
                    input.settings.value_at(m, o).cloned(),
                    catalog[m].options[o].kind,
                ) {
                    let n = choices.len().max(1);
                    let next = if right {
                        (index + n - 1) % n
                    } else {
                        (index + 1) % n
                    };
                    input.settings.set_value_at(m, o, OptionValue::Choice(next));
                }
                self.focus_option(m, o);
            }
            Target::Key(m, o) => {
                if left {
                    self.capture = Some((m, o));
                }
            }
            Target::Text(m, o) => self.text_focus = Some((m, o)),
            Target::Button(m, o) => input.events.push(ModEvent::Button {
                mod_id: catalog[m].id,
                option_id: catalog[m].options[o].id,
            }),
            Target::Swatch(m, o) => {
                if self.color_editor == Some((m, o)) {
                    self.color_editor = None;
                } else {
                    self.color_editor = Some((m, o));
                    if let Some(OptionValue::Color(argb)) = input.settings.value_at(m, o) {
                        self.color_hsv = argb_to_hsva(*argb);
                    }
                }
                self.focus_option(m, o);
            }
            Target::Preset(m, o, preset) => {
                let alpha_allowed = matches!(
                    catalog[m].options[o].kind,
                    OptionKind::Color { alpha: true }
                );
                let current_alpha = match input.settings.value_at(m, o) {
                    Some(OptionValue::Color(argb)) => argb & 0xFF00_0000,
                    _ => 0xFF00_0000,
                };
                let argb = if alpha_allowed {
                    (preset & 0x00FF_FFFF) | current_alpha
                } else {
                    preset
                };
                input.settings.set_value_at(m, o, OptionValue::Color(argb));
                self.color_hsv = argb_to_hsva(argb);
            }
            Target::DynCycle(m, row) => input.events.push(ModEvent::RowCycle {
                mod_id: catalog[m].id,
                row,
                forward: !right,
            }),
            Target::DynAction(m, row, action) => {
                if left {
                    input.events.push(ModEvent::RowAction {
                        mod_id: catalog[m].id,
                        row,
                        action,
                    });
                }
            }
            Target::EditHud => {
                if left && !self.no_world {
                    self.enter_editor(input.now);
                }
            }
            Target::HudSettings(m) => {
                if left {
                    self.leave_editor(input.now);
                    self.return_to_hud = true;
                    self.clear_search();
                    self.select(m);
                    self.pane = Pane::Details;
                }
            }
            Target::Done => self.leave_editor(input.now),
            Target::ResetAll => self.reset_all(input),
            Target::SearchField => {
                // Typing goes to the search again (a click elsewhere already dropped any field
                // focus); a right click clears it.
                if right {
                    self.clear_search();
                }
                self.pane = Pane::Sidebar;
                self.cursors[Pane::Sidebar.index()] = None;
            }
            Target::SearchClear => self.clear_search(),
            Target::MenuKey => {
                if left {
                    self.menu_key_capture = true;
                }
            }
            Target::ResetMod(m) => {
                if left {
                    let armed = self
                        .reset_armed
                        .is_some_and(|(armed_m, at)| armed_m == m && still_armed(at, input.now));
                    if armed {
                        input.settings.reset_options_at(m);
                        self.reset_armed = None;
                        self.color_editor = None;
                    } else {
                        self.reset_armed = Some((m, input.now));
                    }
                }
            }
            Target::ProfileName(m) => self.name_focus = Some(m),
            Target::ProfileSave(m) => {
                if left {
                    self.save_profile(m, input);
                }
            }
            Target::ProfileLoad(m, p) => {
                if left {
                    input.settings.load_profile_at(m, p);
                    self.delete_armed = None;
                }
            }
            Target::ProfileDelete(m, p) => {
                if left {
                    let armed = self.delete_armed.is_some_and(|(am, ap, at)| {
                        (am, ap) == (m, p) && still_armed(at, input.now)
                    });
                    if armed {
                        input.settings.delete_profile_at(m, p);
                        self.delete_armed = None;
                    } else {
                        self.delete_armed = Some((m, p, input.now));
                    }
                }
            }
            // The targets.
            Target::MoveOnScreen(m) => {
                if left && !self.no_world {
                    self.activate_move_on_screen(m, input);
                }
            }
            Target::Keys(target) => self.activate_keys(target, button, input),
            Target::Profiles(target) => self.activate_profiles(target, button, input),
            Target::Extras(target) => self.activate_extras(target, button, input),
            Target::Window(_) | Target::Slider(..) | Target::ColorSlider(..) => {}
        }
    }
}
