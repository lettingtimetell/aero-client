//! A single-line editable text field.
//!
//! java: `client/gui/GuiTextField.java`
//!
//! The awkward part of this widget is `lineScrollOffset`: the field shows a *window* onto the
//! text, and vanilla recomputes that window inside `setSelectionPos`, which every cursor move
//! funnels through. Getting it wrong shows up as the caret walking off the end of the box, so
//! the port keeps vanilla's exact sequence of adjustments.
//!
//! # Indices and limits
//!
//! Cursor, selection, and scroll positions are UTF-16 code-unit indices, matching Java's
//! `String.length()` and `substring` APIs. The field stores those units directly instead of
//! indexing a Rust `String`, which keeps every edit boundary safe even when Java would split a
//! supplementary character into an unpaired surrogate. Public text-returning APIs necessarily
//! use `String::from_utf16_lossy`: valid scalar text round-trips exactly, while an intentionally
//! split surrogate is exposed as `U+FFFD` because Rust strings cannot represent lone surrogates.

use crate::font::FontRenderer;
use crate::widget::draw::{DrawList, Rect};
use crate::widget::keys::{self, filter_allowed_characters, Clipboard, Modifiers};
use std::fmt;
use std::sync::Arc;

/// Enabled text colour, `0xE0E0E0`.
pub const ENABLED_COLOR: i32 = 14_737_632;
/// Disabled text colour, `0x707070`.
pub const DISABLED_COLOR: i32 = 7_368_816;
/// The 1-pixel outline colour, `0xFFA0A0A0` as a signed int.
pub const BORDER_COLOR: u32 = 0xFFA0_A0A0;
/// The box fill, opaque black.
pub const BACKGROUND_COLOR: u32 = 0xFF00_0000;
/// The block caret's colour, `0xFFD0D0D0`.
pub const CARET_COLOR: u32 = 0xFFD0_D0D0;
/// The selection highlight source. Vanilla uses this opaque blue with `GL_XOR`, rather than an
/// alpha-blended translucent rectangle.
pub const SELECTION_COLOR: u32 = 0xFF00_00FF;

/// What [`TextField::draw`] recorded about its visible window.
///
/// Enough to place the caret from a later click with **no** [`FontRenderer`] in hand — see
/// [`TextField::mouse_clicked_cached`] for why that matters. Every number here is a measurement
/// the font already made during the draw; nothing is re-derived from a width table, so this
/// cannot drift from hazard **H9**'s baked widths.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CaretMetrics {
    /// `line_scroll_offset` at the moment of the draw.
    scroll_offset: i32,
    /// `text.len()` at the moment of the draw.
    text_len: i32,
    /// `inner_width()` at the moment of the draw, so a resized box is detected as stale.
    inner_width: i32,
    /// Cumulative pixel advance after each UTF-16 unit of the visible window. `advances[0]` is
    /// always `0`, and `advances.len() - 1` is the number of visible units.
    advances: Vec<i32>,
    /// `trim_to_width_dir(text, inner_width, reverse)` length — how far vanilla scrolls the
    /// window back when the caret lands on the first visible character.
    back_len: i32,
}

impl CaretMetrics {
    /// Number of UTF-16 units in the visible window.
    fn visible_len(&self) -> i32 {
        self.advances.len() as i32 - 1
    }

    /// Replay `trim_to_width(visible, width)`'s accept/stop rule against the cached advances.
    ///
    /// The loop it mirrors guards on "accumulated width so far `< width`" *before* each
    /// character and breaks on "accumulated width `> width`" after it, which is not the same as
    /// a single `<=` comparison when a zero-width glyph is involved.
    // java: FontRenderer.java:708 (trimStringToWidth)
    fn units_fitting(&self, width: i32) -> i32 {
        let mut n = 0;
        for k in 1..self.advances.len() {
            if self.advances[k - 1] >= width || self.advances[k] > width {
                break;
            }
            n = k;
        }
        n as i32
    }
}

#[derive(Clone)]
struct TextValidator(Arc<dyn Fn(&str) -> bool + Send + Sync>);

impl fmt::Debug for TextValidator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("TextValidator(..)")
    }
}

type TextResponderCallback = dyn Fn(i32, &str) + Send + Sync;

#[derive(Clone)]
struct TextResponder(Arc<TextResponderCallback>);

impl fmt::Debug for TextResponder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("TextResponder(..)")
    }
}

fn utf16_len(units: &[u16]) -> i32 {
    units.len() as i32
}

fn utf16_prefix(text: &str, max_units: i32) -> Vec<u16> {
    text.encode_utf16()
        .take(max_units.max(0) as usize)
        .collect()
}

fn units_to_string(units: &[u16]) -> String {
    String::from_utf16_lossy(units)
}

fn filter_allowed_units(units: &[u16]) -> Vec<u16> {
    units
        .iter()
        .copied()
        .filter(|unit| *unit >= 32 && *unit != 127 && *unit != 167)
        .collect()
}

/// An editable single-line text box.
#[derive(Debug, Clone)]
pub struct TextField {
    /// The id reported to the screen's change handler.
    pub id: i32,
    /// Position and size in GUI pixels.
    pub rect: Rect,
    /// Raw Java UTF-16 code units. Keeping these rather than `char`s lets the safe Rust port
    /// represent the half-surrogate state reachable through Java's cursor/delete operations.
    text: Vec<u16>,
    max_length: i32,
    cursor_counter: i32,
    /// Whether the 1-pixel border and black fill are drawn.
    pub background: bool,
    /// Whether clicking elsewhere clears focus.
    pub can_lose_focus: bool,
    focused: bool,
    /// A disabled field renders greyed out and ignores typing.
    pub enabled: bool,
    /// A hidden field draws nothing.
    pub visible: bool,
    line_scroll_offset: i32,
    cursor_position: i32,
    selection_end: i32,
    /// Text colour when enabled.
    pub enabled_color: i32,
    /// Text colour when disabled.
    pub disabled_color: i32,
    caret_metrics: Option<CaretMetrics>,
    validator: Option<TextValidator>,
    responder: Option<TextResponder>,
    /// A [`set_text_deferred_layout`](Self::set_text_deferred_layout) edit whose
    /// visible window still has to be laid out by [`finish_layout`](Self::finish_layout).
    layout_pending: bool,
}

impl TextField {
    /// A field at `(x, y)` of `width`×`height` GUI pixels.
    // java: GuiTextField.java:60
    #[must_use]
    pub fn new(id: i32, x: i32, y: i32, width: i32, height: i32) -> Self {
        Self {
            id,
            rect: Rect::new(x, y, width, height),
            text: Vec::new(),
            max_length: 32,
            cursor_counter: 0,
            background: true,
            can_lose_focus: true,
            focused: false,
            enabled: true,
            visible: true,
            line_scroll_offset: 0,
            cursor_position: 0,
            selection_end: 0,
            enabled_color: ENABLED_COLOR,
            disabled_color: DISABLED_COLOR,
            caret_metrics: None,
            validator: None,
            responder: None,
            layout_pending: false,
        }
    }

    /// The field's contents.
    #[must_use]
    pub fn text(&self) -> String {
        units_to_string(&self.text)
    }

    /// Length in Java UTF-16 code units (the unit used by cursor positions).
    #[must_use]
    pub fn len(&self) -> i32 {
        self.utf16_len()
    }

    /// Whether the field is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Length in Java `char`s, i.e. UTF-16 code units.
    #[must_use]
    pub fn utf16_len(&self) -> i32 {
        utf16_len(&self.text)
    }

    /// Install the predicate that must accept a candidate before an edit is committed.
    ///
    /// Like vanilla, this affects [`set_text`](Self::set_text), [`write_text`](Self::write_text),
    /// and [`delete_from_cursor`](Self::delete_from_cursor). Rejection leaves every field of the
    /// widget unchanged.
    // java: GuiTextField.java:121 (setValidator)
    pub fn set_validator<F>(&mut self, validator: F)
    where
        F: Fn(&str) -> bool + Send + Sync + 'static,
    {
        self.validator = Some(TextValidator(Arc::new(validator)));
    }

    /// Restore vanilla's default always-true validator.
    pub fn clear_validator(&mut self) {
        self.validator = None;
    }

    /// Register the change callback used by `GuiPageButtonList.GuiResponder` in vanilla.
    ///
    /// It fires after every accepted `writeText` or `deleteFromCursor`, including an accepted
    /// edit whose candidate happens to equal the old text. Programmatic `setText` does not fire
    /// it.
    // java: GuiTextField.java:70 (func_175207_a)
    pub fn set_responder<F>(&mut self, responder: F)
    where
        F: Fn(i32, &str) + Send + Sync + 'static,
    {
        self.responder = Some(TextResponder(Arc::new(responder)));
    }

    /// Remove the registered change callback.
    pub fn clear_responder(&mut self) {
        self.responder = None;
    }

    fn accepts(&self, candidate: &str) -> bool {
        self.validator
            .as_ref()
            .is_none_or(|validator| (validator.0)(candidate))
    }

    fn notify_responder(&self) {
        if let Some(responder) = &self.responder {
            (responder.0)(self.id, &self.text());
        }
    }

    /// Replace the contents, truncating to [`max_length`](Self::max_length) and moving the
    /// cursor to the end.
    // java: GuiTextField.java:86 (setText)
    pub fn set_text(&mut self, text: &str, font: &FontRenderer) {
        // An earlier font-free `setText` is laid out first, as vanilla's was.
        self.finish_layout(font);
        if !self.accepts(text) {
            return;
        }
        self.text = utf16_prefix(text, self.max_length);
        self.set_cursor_position(self.len(), font);
    }

    /// [`set_text`](Self::set_text) for a caller with **no [`FontRenderer`]**.
    ///
    /// The text, caret and selection change at once; only the visible window
    /// (`lineScrollOffset`), which needs glyph widths, waits for
    /// [`finish_layout`](Self::finish_layout). Finishing gives exactly what `set_text` would
    /// have, because `setSelectionPos` reads nothing but the text, the clamped scroll offset and
    /// the box width. The caller must finish before any other edit of this field.
    ///
    /// `ContainerScreen` needs this because a slot click resets the anvil's name inside
    /// `Container.slotClick`, and `Screen::mouse_clicked` is handed no font.
    // java: GuiTextField.java:86 (setText)
    pub fn set_text_deferred_layout(&mut self, text: &str) {
        if !self.accepts(text) {
            return;
        }
        self.text = utf16_prefix(text, self.max_length);
        let len = self.len();
        self.cursor_position = len;
        self.selection_end = len;
        // `setSelectionPos`'s first, font-free step.
        self.line_scroll_offset = self.line_scroll_offset.min(len);
        self.caret_metrics = None;
        self.layout_pending = true;
    }

    /// Lay out a pending
    /// [`set_text_deferred_layout`](Self::set_text_deferred_layout): `setText`'s trailing
    /// `setCursorPositionEnd`. A no-op when nothing is pending.
    pub fn finish_layout(&mut self, font: &FontRenderer) {
        if std::mem::take(&mut self.layout_pending) {
            self.set_cursor_position(self.cursor_position, font);
        }
    }

    /// The character limit. Vanilla's default is 32.
    #[must_use]
    pub fn max_length(&self) -> i32 {
        self.max_length
    }

    /// Set the character limit, truncating any excess immediately.
    // java: GuiTextField.java:639
    pub fn set_max_length(&mut self, max: i32) {
        self.max_length = max;
        if self.utf16_len() > max {
            self.text.truncate(max.max(0) as usize);
            let new_len = self.len();
            self.cursor_position = self.cursor_position.clamp(0, new_len);
            self.selection_end = self.selection_end.clamp(0, new_len);
            self.line_scroll_offset = self.line_scroll_offset.clamp(0, new_len);
        }
        self.caret_metrics = None;
    }

    /// Where the caret is.
    #[must_use]
    pub fn cursor_position(&self) -> i32 {
        self.cursor_position
    }

    /// The other end of the selection. Equal to the cursor when nothing is selected.
    #[must_use]
    pub fn selection_end(&self) -> i32 {
        self.selection_end
    }

    /// The first visible UTF-16 code-unit index.
    #[must_use]
    pub fn scroll_offset(&self) -> i32 {
        self.line_scroll_offset
    }

    /// Whether the field has keyboard focus.
    #[must_use]
    pub fn is_focused(&self) -> bool {
        self.focused
    }

    /// Give or take focus. Gaining focus restarts the caret blink.
    // java: GuiTextField.java:697
    pub fn set_focused(&mut self, focused: bool) {
        if focused && !self.focused {
            self.cursor_counter = 0;
        }
        self.focused = focused;
    }

    /// Advance the caret blink. Call once per client tick.
    // java: GuiTextField.java:78
    pub fn tick(&mut self) {
        self.cursor_counter = self.cursor_counter.wrapping_add(1);
    }

    /// Whether the caret is currently in its visible half of the blink.
    #[must_use]
    pub fn caret_visible(&self) -> bool {
        self.focused && self.cursor_counter / 6 % 2 == 0
    }

    /// Usable text width: the box minus 4 pixels of padding on each side when the background
    /// is drawn.
    // java: GuiTextField.java:731 (getWidth)
    #[must_use]
    pub fn inner_width(&self) -> i32 {
        if self.background {
            self.rect.width - 8
        } else {
            self.rect.width
        }
    }

    /// The selected text, or `""`.
    // java: GuiTextField.java:114
    #[must_use]
    pub fn selected_text(&self) -> String {
        let (i, j) = self.selection_range();
        units_to_string(&self.text[i..j])
    }

    fn selection_range(&self) -> (usize, usize) {
        let a = self.cursor_position.min(self.selection_end).max(0) as usize;
        let b = self.cursor_position.max(self.selection_end).max(0) as usize;
        (a.min(self.text.len()), b.min(self.text.len()))
    }

    /// Insert `text` at the cursor, replacing the selection.
    ///
    /// Disallowed characters are filtered out first, and the insertion is truncated to whatever
    /// room [`max_length`](Self::max_length) leaves.
    // java: GuiTextField.java:129 (writeText)
    pub fn write_text(&mut self, text: &str, font: &FontRenderer) {
        let filtered = filter_allowed_characters(text);
        let units = filtered.encode_utf16().collect::<Vec<_>>();
        self.write_utf16_units(&units, font);
    }

    /// Insert already-separated Java UTF-16 units at the cursor, replacing the selection.
    ///
    /// This is the narrow completion seam needed by [`GuiChat`]: Java's
    /// `StringUtils.getCommonPrefix` can end on one half of a supplementary character, and the
    /// subsequent `GuiTextField.writeText` keeps that lone surrogate. Rust callers normally use
    /// [`write_text`](Self::write_text); this raw form preserves the same unit boundary while
    /// applying the same `ChatAllowedCharacters` filter.
    // java: GuiTextField.java:129 (writeText), after GuiChat.java:330
    pub(crate) fn write_utf16_units(&mut self, units: &[u16], font: &FontRenderer) {
        let filtered = filter_allowed_units(units);
        let (i, j) = self.selection_range();
        // java: maxStringLength - text.length() - (i - j); (i - j) is <= 0, so this is
        // "free space + the selection about to be replaced".
        let selected_units = utf16_len(&self.text[i..j]);
        let room = (self.max_length - self.utf16_len() + selected_units).max(0);
        let inserted = filtered
            .into_iter()
            .take(room.max(0) as usize)
            .collect::<Vec<_>>();
        let taken = inserted.len();

        let mut next: Vec<u16> = Vec::with_capacity(self.text.len() + taken);
        next.extend_from_slice(&self.text[..i]);
        next.extend_from_slice(&inserted);
        if j < self.text.len() {
            next.extend_from_slice(&self.text[j..]);
        }
        let candidate = units_to_string(&next);
        if !self.accepts(&candidate) {
            return;
        }
        self.text = next;
        self.set_cursor_position(i as i32 + taken as i32, font);
        self.notify_responder();
    }

    /// Delete `count` words forward (positive) or backward (negative) from the cursor.
    // java: GuiTextField.java:175 (deleteWords)
    pub fn delete_words(&mut self, count: i32, font: &FontRenderer) {
        if self.text.is_empty() {
            return;
        }
        if self.selection_end != self.cursor_position {
            self.write_text("", font);
        } else {
            let target = self.nth_word_from_cursor(count);
            self.delete_from_cursor(target - self.cursor_position, font);
        }
    }

    /// Delete `count` characters forward (positive) or backward (negative), or the selection.
    // java: GuiTextField.java:193 (deleteFromCursor)
    pub fn delete_from_cursor(&mut self, count: i32, font: &FontRenderer) {
        if self.text.is_empty() {
            return;
        }
        if self.selection_end != self.cursor_position {
            self.write_text("", font);
            return;
        }
        let backwards = count < 0;
        let i = if backwards {
            self.cursor_position + count
        } else {
            self.cursor_position
        };
        let j = if backwards {
            self.cursor_position
        } else {
            self.cursor_position + count
        };
        let mut next: Vec<u16> = Vec::new();
        if i >= 0 {
            next.extend_from_slice(&self.text[..(i as usize).min(self.text.len())]);
        }
        if j < self.len() {
            next.extend_from_slice(&self.text[(j.max(0) as usize).min(self.text.len())..]);
        }
        let candidate = units_to_string(&next);
        if !self.accepts(&candidate) {
            return;
        }
        self.text = next;
        self.caret_metrics = None;
        if backwards {
            self.move_cursor_by(count, font);
        }
        self.notify_responder();
    }

    /// Index of the `n`-th word boundary from the cursor.
    // java: GuiTextField.java:244 (getNthWordFromCursor)
    #[must_use]
    pub fn nth_word_from_cursor(&self, n: i32) -> i32 {
        self.nth_word_from_pos(n, self.cursor_position)
    }

    /// Index of the `n`-th word boundary from `pos`. Negative `n` searches backwards.
    // java: GuiTextField.java:257 (func_146197_a, skipSpaces = true)
    #[must_use]
    pub fn nth_word_from_pos(&self, n: i32, pos: i32) -> i32 {
        self.word_from_pos(n, pos, true)
    }

    /// Index of the `n`-th word boundary from `pos`, optionally skipping runs of spaces.
    ///
    /// Normal Ctrl+word navigation uses `skip_spaces = true`; chat completion uses `false` so
    /// an empty prefix after a space stays empty.
    // java: GuiTextField.java:257 (func_146197_a)
    #[must_use]
    pub fn word_from_pos(&self, n: i32, pos: i32, skip_spaces: bool) -> i32 {
        let mut i = pos.clamp(0, self.len());
        let backwards = n < 0;
        for _ in 0..n.saturating_abs() {
            if backwards {
                while skip_spaces && i > 0 && self.text[i as usize - 1] == u16::from(b' ') {
                    i -= 1;
                }
                while i > 0 && self.text[i as usize - 1] != u16::from(b' ') {
                    i -= 1;
                }
            } else {
                let len = self.len();
                i = match self.text[i.min(len) as usize..]
                    .iter()
                    .position(|c| *c == u16::from(b' '))
                {
                    Some(off) => {
                        let mut k = i + off as i32;
                        while skip_spaces && k < len && self.text[k as usize] == u16::from(b' ') {
                            k += 1;
                        }
                        k
                    }
                    None => len,
                };
            }
        }
        i
    }

    /// Move the caret relative to the **selection anchor**, clearing the selection.
    ///
    /// Note it is relative to `selectionEnd`, not to `cursorPosition` — that is what makes
    /// pressing Left with a selection active collapse it to the left edge.
    // java: GuiTextField.java:302 (moveCursorBy)
    pub fn move_cursor_by(&mut self, delta: i32, font: &FontRenderer) {
        self.set_cursor_position(self.selection_end + delta, font);
    }

    /// Put the caret at `pos`, clamped, and collapse the selection onto it.
    // java: GuiTextField.java:310 (setCursorPosition)
    pub fn set_cursor_position(&mut self, pos: i32, font: &FontRenderer) {
        self.cursor_position = pos.clamp(0, self.len());
        self.set_selection_pos(self.cursor_position, font);
    }

    /// Move the selection anchor, recomputing the visible window.
    // java: GuiTextField.java:739 (setSelectionPos)
    pub fn set_selection_pos(&mut self, pos: i32, font: &FontRenderer) {
        // Every text edit and every cursor move funnels through here, so this is the one place
        // the draw-time caret cache has to be dropped.
        self.caret_metrics = None;
        let i = self.len();
        let pos = pos.clamp(0, i);
        self.selection_end = pos;

        if self.line_scroll_offset > i {
            self.line_scroll_offset = i;
        }
        let width = self.inner_width();
        let tail = units_to_string(&self.text[self.line_scroll_offset.max(0) as usize..]);
        let visible = font.trim_to_width(&tail, width);
        let k = visible.encode_utf16().count() as i32 + self.line_scroll_offset;

        if pos == self.line_scroll_offset {
            let back = font.trim_to_width_dir(&self.text(), width, true);
            self.line_scroll_offset -= back.encode_utf16().count() as i32;
        }
        if pos > k {
            self.line_scroll_offset += pos - k;
        } else if pos <= self.line_scroll_offset {
            // java writes `lineScrollOffset -= lineScrollOffset - pos`, which is just `= pos`.
            self.line_scroll_offset = pos;
        }
        self.line_scroll_offset = self.line_scroll_offset.clamp(0, i);
    }

    /// Handle a key press. Returns whether the field consumed it.
    ///
    /// `typed` is the character the key produced, if any; `key` is an LWJGL scan code (see
    /// [`crate::widget::keys`]).
    // java: GuiTextField.java:337 (textboxKeyTyped)
    pub fn key_typed(
        &mut self,
        typed: Option<char>,
        key: i32,
        mods: Modifiers,
        clipboard: &mut dyn Clipboard,
        font: &FontRenderer,
    ) -> bool {
        self.finish_layout(font);
        if !self.focused {
            return false;
        }

        if mods.is_ctrl_combo(key, keys::A) {
            self.set_cursor_position(self.len(), font);
            self.set_selection_pos(0, font);
            return true;
        }
        if mods.is_ctrl_combo(key, keys::C) {
            clipboard.set(&self.selected_text());
            return true;
        }
        if mods.is_ctrl_combo(key, keys::V) {
            if self.enabled {
                let pasted = clipboard.get();
                self.write_text(&pasted, font);
            }
            return true;
        }
        if mods.is_ctrl_combo(key, keys::X) {
            clipboard.set(&self.selected_text());
            if self.enabled {
                self.write_text("", font);
            }
            return true;
        }

        match key {
            keys::BACKSPACE => {
                if self.enabled {
                    if mods.ctrl {
                        self.delete_words(-1, font);
                    } else {
                        self.delete_from_cursor(-1, font);
                    }
                }
                true
            }
            keys::HOME => {
                if mods.shift {
                    self.set_selection_pos(0, font);
                } else {
                    self.set_cursor_position(0, font);
                }
                true
            }
            keys::LEFT => {
                if mods.shift {
                    if mods.ctrl {
                        let p = self.nth_word_from_pos(-1, self.selection_end);
                        self.set_selection_pos(p, font);
                    } else {
                        self.set_selection_pos(self.selection_end - 1, font);
                    }
                } else if mods.ctrl {
                    let p = self.nth_word_from_cursor(-1);
                    self.set_cursor_position(p, font);
                } else {
                    self.move_cursor_by(-1, font);
                }
                true
            }
            keys::RIGHT => {
                if mods.shift {
                    if mods.ctrl {
                        let p = self.nth_word_from_pos(1, self.selection_end);
                        self.set_selection_pos(p, font);
                    } else {
                        self.set_selection_pos(self.selection_end + 1, font);
                    }
                } else if mods.ctrl {
                    let p = self.nth_word_from_cursor(1);
                    self.set_cursor_position(p, font);
                } else {
                    self.move_cursor_by(1, font);
                }
                true
            }
            keys::END => {
                if mods.shift {
                    self.set_selection_pos(self.len(), font);
                } else {
                    self.set_cursor_position(self.len(), font);
                }
                true
            }
            keys::DELETE => {
                if self.enabled {
                    if mods.ctrl {
                        self.delete_words(1, font);
                    } else {
                        self.delete_from_cursor(1, font);
                    }
                }
                true
            }
            _ => match typed {
                Some(c) if keys::is_allowed_character(c) => {
                    if self.enabled {
                        let s = c.to_string();
                        self.write_text(&s, font);
                    }
                    true
                }
                _ => false,
            },
        }
    }

    /// Handle a mouse press, taking or dropping focus and placing the caret.
    // java: GuiTextField.java:499 (mouseClicked)
    pub fn mouse_clicked(&mut self, x: i32, y: i32, button: i32, font: &FontRenderer) {
        self.finish_layout(font);
        let inside = self.rect.contains(x, y);
        if self.can_lose_focus {
            self.set_focused(inside);
        }
        if self.focused && inside && button == 0 {
            let mut i = x - self.rect.x;
            if self.background {
                i -= 4;
            }
            let tail = units_to_string(&self.text[self.line_scroll_offset.max(0) as usize..]);
            let visible = font.trim_to_width(&tail, self.inner_width());
            let hit = font.trim_to_width(&visible, i).encode_utf16().count() as i32;
            self.set_cursor_position(hit + self.line_scroll_offset, font);
        }
    }

    /// Whether [`draw`](Self::draw) has run since the last edit, so
    /// [`mouse_clicked_cached`](Self::mouse_clicked_cached) can place the caret exactly.
    ///
    /// False for a field that has never been drawn, one edited since its last draw, or one
    /// whose box has been resized since.
    #[must_use]
    pub fn caret_metrics_are_current(&self) -> bool {
        self.caret_metrics.as_ref().is_some_and(|m| {
            m.text_len == self.len()
                && m.scroll_offset == self.line_scroll_offset
                && m.inner_width == self.inner_width()
                && self.line_scroll_offset <= self.len()
        })
    }

    /// [`mouse_clicked`](Self::mouse_clicked) **without a [`FontRenderer`]**.
    ///
    /// `Screen::mouse_clicked` is handed no font, which is why `CreativeScreen` cannot place a
    /// caret by clicking (recorded as a cross-section request from S17). It does not need one:
    /// the only measurements a click consumes are of text that was already on screen, and
    /// [`draw`](Self::draw) — which *does* get a font — records exactly those. This replays
    /// vanilla's arithmetic against that recording, so the caret lands on the same character
    /// `mouse_clicked` would have chosen, not an approximation of it.
    ///
    /// Focus is taken or dropped either way. The return value is whether the **caret** was
    /// placed: `false` means the field had not been drawn since its last edit, so a caller that
    /// does have a font should fall back to [`mouse_clicked`](Self::mouse_clicked).
    // java: GuiTextField.java:499 (mouseClicked)
    pub fn mouse_clicked_cached(&mut self, x: i32, y: i32, button: i32) -> bool {
        let inside = self.rect.contains(x, y);
        if self.can_lose_focus {
            self.set_focused(inside);
        }
        if !(self.focused && inside && button == 0) {
            // Vanilla places no caret in this case either, so nothing was missed.
            return true;
        }
        if !self.caret_metrics_are_current() {
            return false;
        }
        let Some(m) = self.caret_metrics.clone() else {
            return false;
        };
        let mut i = x - self.rect.x;
        if self.background {
            i -= 4;
        }
        let hit = m.units_fitting(i);
        self.cursor_position = (hit + m.scroll_offset).clamp(0, self.len());
        self.set_selection_pos_cached(self.cursor_position, &m);
        true
    }

    /// [`set_selection_pos`](Self::set_selection_pos) driven by a draw-time recording.
    ///
    /// A line-for-line replay of the font-backed version, with the visible UTF-16-unit count and the
    /// reverse trim read out of [`CaretMetrics`] instead of re-measured. Only ever called with
    /// metrics whose `scroll_offset` still matches the live one, so the two see the same window.
    // java: GuiTextField.java:739 (setSelectionPos)
    fn set_selection_pos_cached(&mut self, pos: i32, m: &CaretMetrics) {
        let i = self.len();
        let pos = pos.clamp(0, i);
        self.selection_end = pos;

        if self.line_scroll_offset > i {
            self.line_scroll_offset = i;
        }
        let k = m.visible_len() + self.line_scroll_offset;

        if pos == self.line_scroll_offset {
            self.line_scroll_offset -= m.back_len;
        }
        if pos > k {
            self.line_scroll_offset += pos - k;
        } else if pos <= self.line_scroll_offset {
            self.line_scroll_offset = pos;
        }
        self.line_scroll_offset = self.line_scroll_offset.clamp(0, i);
        // The window may have moved, so the recording no longer describes what is on screen.
        self.caret_metrics = None;
    }

    /// The text currently inside the visible window.
    #[must_use]
    pub fn visible_text(&self, font: &FontRenderer) -> String {
        let tail = units_to_string(&self.text[self.line_scroll_offset.max(0) as usize..]);
        font.trim_to_width(&tail, self.inner_width())
    }

    /// Draw the box, its text, the caret and the selection highlight.
    // java: GuiTextField.java:525 (drawTextBox)
    pub fn draw(&mut self, d: &mut DrawList, font: &mut FontRenderer) {
        self.finish_layout(font);
        if !self.visible {
            return;
        }
        if self.background {
            d.rect(
                self.rect.x - 1,
                self.rect.y - 1,
                self.rect.right() + 1,
                self.rect.bottom() + 1,
                BORDER_COLOR,
            );
            d.rect(
                self.rect.x,
                self.rect.y,
                self.rect.right(),
                self.rect.bottom(),
                BACKGROUND_COLOR,
            );
        }

        let color = if self.enabled {
            self.enabled_color
        } else {
            self.disabled_color
        };
        let j = self.cursor_position - self.line_scroll_offset;
        let mut k = self.selection_end - self.line_scroll_offset;
        let visible = self.visible_text(font);
        let visible_units: Vec<u16> = visible.encode_utf16().collect();
        let vis_len = visible_units.len() as i32;
        self.record_caret_metrics(&visible_units, font);
        let caret_in_view = j >= 0 && j <= vis_len;
        let blink = self.caret_visible() && caret_in_view;
        let text_x = if self.background {
            self.rect.x + 4
        } else {
            self.rect.x
        };
        let text_y = if self.background {
            self.rect.y + (self.rect.height - 8) / 2
        } else {
            self.rect.y
        };
        let mut pen = text_x;

        if k > vis_len {
            k = vis_len;
        }

        if !visible_units.is_empty() {
            let head: String = if caret_in_view {
                units_to_string(&visible_units[..j.max(0) as usize])
            } else {
                visible.clone()
            };
            pen = font.draw_string_with_shadow(d, &head, pen as f32, text_y as f32, color);
        }

        let block_caret = self.cursor_position < self.len() || self.utf16_len() >= self.max_length;
        let mut caret_x = pen;
        if !caret_in_view {
            caret_x = if j > 0 {
                text_x + self.rect.width
            } else {
                text_x
            };
        } else if block_caret {
            caret_x = pen - 1;
            pen -= 1;
        }

        if !visible_units.is_empty() && caret_in_view && j < vis_len {
            let tail = units_to_string(&visible_units[j.max(0) as usize..]);
            pen = font.draw_string_with_shadow(d, &tail, pen as f32, text_y as f32, color);
        }

        if blink {
            if block_caret {
                d.rect(
                    caret_x,
                    text_y - 1,
                    caret_x + 1,
                    text_y + 1 + font.font_height(),
                    CARET_COLOR,
                );
            } else {
                font.draw_string_with_shadow(d, "_", caret_x as f32, text_y as f32, color);
            }
        }

        if k != j {
            let selected = units_to_string(&visible_units[..k.clamp(0, vis_len) as usize]);
            let other = text_x + font.string_width(&selected);
            self.draw_selection(
                d,
                caret_x,
                text_y - 1,
                other - 1,
                text_y + 1 + font.font_height(),
            );
        }
        let _ = pen;
    }

    /// Snapshot the visible window's geometry for [`mouse_clicked_cached`].
    ///
    /// Called from [`draw`](Self::draw) with the window it just laid out, so the recording is by
    /// construction the same measurement the pixels came from.  The width walk deliberately
    /// mirrors `FontRenderer::trim_to_width_dir`: formatting codes are zero-width, and bold adds
    /// one pixel per UTF-16 unit.  A raw `char_width` sum would make a cached click disagree with
    /// the font-backed path for programmatically supplied `\u{00a7}`-formatted text.
    fn record_caret_metrics(&mut self, visible_units: &[u16], font: &FontRenderer) {
        let mut advances = Vec::with_capacity(visible_units.len() + 1);
        let mut acc = 0;
        let mut after_section = false;
        let mut bold = false;
        advances.push(0);
        for &unit in visible_units {
            let unit_width = font.unit_width(unit);
            let mut advance = 0;
            if after_section {
                after_section = false;
                if unit == u16::from(b'l') || unit == u16::from(b'L') {
                    bold = true;
                } else if unit == u16::from(b'r') || unit == u16::from(b'R') {
                    bold = false;
                }
            } else if unit_width < 0 {
                after_section = true;
            } else {
                advance = unit_width;
                if bold {
                    advance += 1;
                }
            }
            acc += advance;
            advances.push(acc);
        }
        let inner_width = self.inner_width();
        self.caret_metrics = Some(CaretMetrics {
            scroll_offset: self.line_scroll_offset,
            text_len: self.len(),
            inner_width,
            advances,
            back_len: font
                .trim_to_width_dir(&self.text(), inner_width, true)
                .encode_utf16()
                .count() as i32,
        });
    }

    // java: GuiTextField.java:597 (drawCursorVertical)
    fn draw_selection(&self, d: &mut DrawList, x0: i32, y0: i32, x1: i32, y1: i32) {
        let right_limit = self.rect.right();
        let a = x0.min(x1).min(right_limit);
        let b = x0.max(x1).min(right_limit);
        let (t, bt) = (y0.min(y1), y0.max(y1));
        d.rect_xor_blue(a, t, b, bt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::keys::MemoryClipboard;
    use std::sync::Mutex;

    /// A font whose every glyph is 5 wide, so widths in these tests are `5 * chars`.
    fn font() -> FontRenderer {
        crate::widget::test_support::uniform_font()
    }

    fn field(font: &FontRenderer) -> TextField {
        let mut t = TextField::new(1, 10, 20, 100, 20);
        t.set_focused(true);
        t.set_max_length(64);
        t.set_text("hello world", font);
        t
    }

    #[test]
    fn typing_inserts_at_the_cursor() {
        let f = font();
        let mut t = field(&f);
        let mut clip = MemoryClipboard::default();
        t.set_cursor_position(5, &f);
        assert!(t.key_typed(Some('!'), 0, Modifiers::NONE, &mut clip, &f));
        assert_eq!(t.text(), "hello! world");
        assert_eq!(t.cursor_position(), 6);
    }

    #[test]
    fn disallowed_characters_are_rejected_not_inserted() {
        let f = font();
        let mut t = field(&f);
        let mut clip = MemoryClipboard::default();
        assert!(!t.key_typed(Some('§'), 0, Modifiers::NONE, &mut clip, &f));
        assert!(!t.key_typed(Some('\n'), 0, Modifiers::NONE, &mut clip, &f));
        assert_eq!(t.text(), "hello world");
    }

    #[test]
    fn an_unfocused_field_swallows_nothing() {
        let f = font();
        let mut t = field(&f);
        let mut clip = MemoryClipboard::default();
        t.set_focused(false);
        assert!(!t.key_typed(Some('x'), 0, Modifiers::NONE, &mut clip, &f));
        assert_eq!(t.text(), "hello world");
    }

    #[test]
    fn max_length_truncates_both_writes_and_the_limit_itself() {
        let f = font();
        let mut t = TextField::new(0, 0, 0, 100, 20);
        t.set_focused(true);
        t.set_text("abcdefgh", &f);
        t.set_max_length(4);
        assert_eq!(t.text(), "abcd");
        t.set_cursor_position(4, &f);
        t.write_text("xyz", &f);
        assert_eq!(t.text(), "abcd", "no room left");
        t.set_max_length(6);
        t.write_text("xyz", &f);
        assert_eq!(t.text(), "abcdxy", "only two characters fit");
    }

    #[test]
    fn max_length_is_measured_in_utf16_code_units() {
        let f = font();
        let mut t = TextField::new(0, 0, 0, 100, 20);
        t.set_max_length(100);
        t.set_text(&"😀".repeat(60), &f);
        assert_eq!(t.text().chars().count(), 50);
        assert_eq!(t.utf16_len(), 100);

        // Replacing the two UTF-16 units of the emoji frees two units. The end position is 3:
        // `a` occupies 0..1, the emoji 1..3, and `b` 3..4.
        t.set_max_length(4);
        t.set_text("a😀b", &f);
        t.set_cursor_position(1, &f);
        t.set_selection_pos(3, &f);
        t.write_text("XY", &f);
        assert_eq!(t.text(), "aXYb");
        assert_eq!(t.utf16_len(), 4);
    }

    #[test]
    fn non_bmp_cursor_and_selection_positions_are_utf16_units() {
        let f = font();
        let mut t = TextField::new(0, 0, 0, 100, 20);
        t.set_focused(true);
        t.set_max_length(64);
        t.set_text("a😀b", &f);
        assert_eq!(t.len(), 4, "a + surrogate pair + b");
        assert_eq!(t.cursor_position(), 4);

        // Java arrows advance one UTF-16 char at a time, so the two middle positions are
        // reachable even though Rust exposes the intact pair when it is returned as a String.
        let mut clip = MemoryClipboard::default();
        t.key_typed(None, keys::LEFT, Modifiers::NONE, &mut clip, &f);
        assert_eq!(t.cursor_position(), 3, "before b");
        t.key_typed(None, keys::LEFT, Modifiers::NONE, &mut clip, &f);
        assert_eq!(t.cursor_position(), 2, "between the surrogate halves");
        t.key_typed(None, keys::LEFT, Modifiers::NONE, &mut clip, &f);
        assert_eq!(t.cursor_position(), 1, "before the emoji");

        t.set_cursor_position(1, &f);
        t.set_selection_pos(3, &f);
        assert_eq!(t.selected_text(), "😀");
        t.set_selection_pos(2, &f);
        assert_eq!(
            t.selected_text(),
            "�",
            "a half-surrogate is lossy at the Rust API edge"
        );
    }

    #[test]
    fn non_bmp_delete_and_paste_never_slice_a_rust_string() {
        let f = font();
        let mut t = TextField::new(0, 0, 0, 100, 20);
        t.set_focused(true);
        t.set_max_length(64);
        t.set_text("a😀b", &f);
        let mut clip = MemoryClipboard::default();

        // Backspace/Delete are Java-character operations. Each half can be removed separately,
        // leaving a safe raw UTF-16 unit behind until the second operation removes it.
        t.set_cursor_position(3, &f);
        t.key_typed(None, keys::BACKSPACE, Modifiers::NONE, &mut clip, &f);
        assert_eq!(t.cursor_position(), 2);
        assert_eq!(t.text(), "a�b");
        t.key_typed(None, keys::BACKSPACE, Modifiers::NONE, &mut clip, &f);
        assert_eq!(t.cursor_position(), 1);
        assert_eq!(t.text(), "ab");

        t.set_text("a😀b", &f);
        t.set_cursor_position(1, &f);
        t.set_selection_pos(3, &f);
        clip.set("XY😀");
        t.key_typed(None, keys::V, Modifiers::CTRL, &mut clip, &f);
        assert_eq!(t.text(), "aXY😀b");
        assert_eq!(t.cursor_position(), 5, "a + XY + emoji = five UTF-16 units");
    }

    #[test]
    fn non_bmp_mouse_and_scroll_positions_count_utf16_units() {
        let f = font();
        let mut t = TextField::new(0, 0, 0, 48, 20); // inner width 40 px
        t.set_focused(true);
        t.set_max_length(64);
        t.set_text("😀abcdefghijklmnop", &f);
        assert_eq!(
            t.cursor_position(),
            18,
            "two UTF-16 units for the leading emoji"
        );
        assert!(t.scroll_offset() > 0);
        assert!(t.visible_text(&f).encode_utf16().count() <= 10);

        // The synthetic font has no emoji glyph, so a click just into the first visible glyph
        // still passes over the zero-width surrogate pair and lands on a UTF-16 boundary.
        t.set_text("😀abc", &f);
        t.mouse_clicked(5, 4, 0, &f);
        assert_eq!(t.cursor_position(), 2, "click after the zero-width emoji");

        // The font-free path must use the same UTF-16-unit advances as the font-backed path.
        let mut backed = TextField::new(0, 0, 0, 48, 20);
        backed.set_focused(true);
        backed.set_max_length(64);
        backed.set_text("😀abc", &f);
        let mut cached = backed.clone();
        let mut draw_font = font();
        let mut d = DrawList::new();
        cached.draw(&mut d, &mut draw_font);
        for x in 0..=24 {
            let mut a = backed.clone();
            let mut b = cached.clone();
            a.mouse_clicked(x, 4, 0, &f);
            assert!(b.mouse_clicked_cached(x, 4, 0));
            assert_eq!(
                (a.cursor_position(), a.selection_end(), a.scroll_offset()),
                (b.cursor_position(), b.selection_end(), b.scroll_offset()),
                "emoji click at x={x}"
            );
        }
    }

    /// The font-free `setText` plus its later layout is exactly
    /// [`TextField::set_text`], scroll window included, from every earlier scroll position.
    /// Drawing, typing and a font-backed `set_text` all finish a pending layout first.
    #[test]
    fn a_deferred_set_text_finishes_exactly_like_set_text() {
        let f = font();
        let starts = ["", "short", "a much longer line than the box shows"];
        let texts = [
            String::new(),
            "Apple".to_owned(),
            "Enchanted Golden Apple".to_owned(),
            "x".repeat(40),
        ];
        let state = |t: &TextField| {
            (
                t.text(),
                t.cursor_position(),
                t.selection_end(),
                t.scroll_offset(),
            )
        };
        for start in starts {
            for position in 0..=start.len() as i32 {
                for text in &texts {
                    let mut eager = TextField::new(0, 0, 0, 48, 20); // inner width 40 px
                    eager.set_max_length(30);
                    eager.set_text(start, &f);
                    eager.set_cursor_position(position, &f);
                    let mut deferred = eager.clone();
                    let mut drawn = eager.clone();
                    eager.set_text(text, &f);

                    deferred.set_text_deferred_layout(text);
                    assert_eq!(deferred.text(), eager.text(), "the text changes at once");
                    assert_eq!(deferred.cursor_position(), eager.cursor_position());
                    deferred.finish_layout(&f);
                    assert_eq!(
                        state(&deferred),
                        state(&eager),
                        "{start:?}@{position} {text:?}"
                    );

                    drawn.set_text_deferred_layout(text);
                    drawn.draw(&mut DrawList::new(), &mut font());
                    assert_eq!(state(&drawn), state(&eager), "draw lays it out");
                }
            }
        }
        // A later font-backed `set_text` sees the first one laid out, as vanilla's would.
        let mut eager = TextField::new(0, 0, 0, 48, 20);
        eager.set_max_length(30);
        let mut deferred = eager.clone();
        eager.set_text(&texts[3], &f);
        eager.set_text("Apple", &f);
        deferred.set_text_deferred_layout(&texts[3]);
        deferred.set_text("Apple", &f);
        assert_eq!(state(&deferred), state(&eager));
    }

    #[test]
    fn backspace_and_delete_remove_one_character() {
        let f = font();
        let mut t = field(&f);
        let mut clip = MemoryClipboard::default();
        t.set_cursor_position(5, &f);
        t.key_typed(None, keys::BACKSPACE, Modifiers::NONE, &mut clip, &f);
        assert_eq!(t.text(), "hell world");
        assert_eq!(t.cursor_position(), 4);
        t.key_typed(None, keys::DELETE, Modifiers::NONE, &mut clip, &f);
        assert_eq!(t.text(), "hellworld");
        assert_eq!(t.cursor_position(), 4, "delete leaves the caret alone");
    }

    #[test]
    fn forward_delete_leaves_the_visible_window_untouched() {
        let f = font();
        let mut t = TextField::new(0, 0, 0, 48, 20);
        t.set_max_length(64);
        t.set_text("The quick brown fox", &f);
        t.cursor_position = 6;
        t.selection_end = 6;
        t.line_scroll_offset = 6;

        t.delete_from_cursor(1, &f);
        assert_eq!(t.text(), "The quck brown fox");
        assert_eq!(t.cursor_position(), 6);
        assert_eq!(t.selection_end(), 6);
        assert_eq!(t.scroll_offset(), 6);
    }

    #[test]
    fn deleting_at_the_edges_is_a_no_op_not_a_panic() {
        let f = font();
        let mut t = field(&f);
        t.set_cursor_position(0, &f);
        t.delete_from_cursor(-1, &f);
        assert_eq!(t.text(), "hello world");
        t.set_cursor_position(t.len(), &f);
        t.delete_from_cursor(1, &f);
        assert_eq!(t.text(), "hello world");
        // An empty field ignores both.
        let mut e = TextField::new(0, 0, 0, 50, 20);
        e.delete_from_cursor(-1, &f);
        e.delete_from_cursor(1, &f);
        assert!(e.is_empty());
    }

    #[test]
    fn selection_uses_vanillas_opaque_blue_xor_batch() {
        let mut f = font();
        let mut t = field(&f);
        t.set_cursor_position(0, &f);
        t.set_selection_pos(5, &f);
        let mut d = DrawList::new();
        t.draw(&mut d, &mut f);

        let batch = d.batches().last().expect("selection emits a batch");
        assert_eq!(batch.blend, crate::widget::draw::UiBlend::Xor);
        assert_eq!(batch.texture, crate::widget::draw::UiTexture::Solid);
        assert_eq!(
            d.vertices()[batch.range.start].color,
            [0.0, 0.0, 1.0, 1.0],
            "GuiTextField sets opaque blue before GL_XOR"
        );
        assert_eq!(SELECTION_COLOR, 0xFF00_00FF);
    }

    #[test]
    fn word_navigation_skips_runs_of_spaces() {
        let f = font();
        let mut t = TextField::new(0, 0, 0, 200, 20);
        t.set_max_length(64);
        t.set_text("one  two three", &f);
        assert_eq!(t.nth_word_from_pos(1, 0), 5, "past 'one' and both spaces");
        assert_eq!(t.nth_word_from_pos(1, 5), 9);
        assert_eq!(t.nth_word_from_pos(1, 9), 14, "no more spaces: the end");
        assert_eq!(t.nth_word_from_pos(-1, 14), 9);
        assert_eq!(t.nth_word_from_pos(-1, 9), 5);
        assert_eq!(t.nth_word_from_pos(-1, 5), 0);
        assert_eq!(t.nth_word_from_pos(-1, 0), 0);
        assert_eq!(t.nth_word_from_pos(2, 0), 9, "n applies twice");
    }

    #[test]
    fn word_navigation_can_preserve_spaces_for_chat_completion() {
        let f = font();
        let mut t = TextField::new(0, 0, 0, 200, 20);
        t.set_max_length(64);
        t.set_text("/tp No", &f);
        assert_eq!(t.word_from_pos(-1, 4, false), 4);
        assert_eq!(t.word_from_pos(-1, 4, true), 0);
        assert_eq!(t.word_from_pos(1, 0, false), 3);
        assert_eq!(t.word_from_pos(1, 0, true), 4);
    }

    #[test]
    fn rejected_candidates_leave_all_edit_state_untouched() {
        let f = font();
        let mut t = TextField::new(0, 0, 0, 48, 20);
        t.set_max_length(64);
        t.set_text("a.b", &f);
        t.set_cursor_position(2, &f);
        t.set_validator(|candidate| !candidate.contains("..") && !candidate.is_empty());

        let before = (
            t.text(),
            t.cursor_position(),
            t.selection_end(),
            t.scroll_offset(),
        );
        t.write_text(".", &f);
        assert_eq!(
            (
                t.text(),
                t.cursor_position(),
                t.selection_end(),
                t.scroll_offset(),
            ),
            before
        );

        t.set_text("", &f);
        assert_eq!(t.text(), "a.b", "setText validates before truncating");

        t.clear_validator();
        t.set_text("x", &f);
        t.set_validator(|candidate| !candidate.is_empty());
        t.delete_from_cursor(-1, &f);
        assert_eq!(t.text(), "x", "a rejected deletion is also atomic");
        assert_eq!(t.cursor_position(), 1);
    }

    #[test]
    fn responder_fires_only_for_accepted_user_edits() {
        let f = font();
        let calls = Arc::new(Mutex::new(Vec::new()));
        let observed = Arc::clone(&calls);
        let mut t = TextField::new(7, 0, 0, 100, 20);
        t.set_responder(move |id, text| {
            observed.lock().unwrap().push((id, text.to_owned()));
        });

        t.set_text("a", &f);
        assert!(calls.lock().unwrap().is_empty(), "setText does not notify");
        t.write_text("b", &f);
        t.delete_from_cursor(1, &f); // accepted no-op at the end still notifies in Java
        t.set_validator(|candidate| !candidate.ends_with('!'));
        t.write_text("!", &f);

        assert_eq!(
            calls.lock().unwrap().as_slice(),
            &[(7, "ab".to_owned()), (7, "ab".to_owned())]
        );
    }

    #[test]
    fn ctrl_backspace_deletes_a_word() {
        let f = font();
        let mut t = field(&f);
        let mut clip = MemoryClipboard::default();
        t.set_cursor_position(t.len(), &f);
        t.key_typed(None, keys::BACKSPACE, Modifiers::CTRL, &mut clip, &f);
        assert_eq!(t.text(), "hello ");
    }

    #[test]
    fn select_all_copy_and_paste_round_trip() {
        let f = font();
        let mut t = field(&f);
        let mut clip = MemoryClipboard::default();
        assert!(t.key_typed(None, keys::A, Modifiers::CTRL, &mut clip, &f));
        assert_eq!(t.selected_text(), "hello world");
        assert!(t.key_typed(None, keys::C, Modifiers::CTRL, &mut clip, &f));
        assert_eq!(clip.get(), "hello world");
        assert!(t.key_typed(None, keys::X, Modifiers::CTRL, &mut clip, &f));
        assert_eq!(t.text(), "", "cut clears the selection");
        assert!(t.key_typed(None, keys::V, Modifiers::CTRL, &mut clip, &f));
        assert_eq!(t.text(), "hello world");
    }

    #[test]
    fn a_disabled_field_still_copies_but_never_edits() {
        let f = font();
        let mut t = field(&f);
        let mut clip = MemoryClipboard::default();
        t.enabled = false;
        t.key_typed(None, keys::A, Modifiers::CTRL, &mut clip, &f);
        assert!(t.key_typed(None, keys::C, Modifiers::CTRL, &mut clip, &f));
        assert_eq!(clip.get(), "hello world");
        assert!(t.key_typed(None, keys::X, Modifiers::CTRL, &mut clip, &f));
        assert_eq!(t.text(), "hello world", "cut copies but does not delete");
        assert!(t.key_typed(Some('z'), 0, Modifiers::NONE, &mut clip, &f));
        assert_eq!(t.text(), "hello world");
    }

    #[test]
    fn shift_arrow_extends_the_selection_and_plain_arrow_collapses_it() {
        let f = font();
        let mut t = field(&f);
        let mut clip = MemoryClipboard::default();
        t.set_cursor_position(0, &f);
        for _ in 0..5 {
            t.key_typed(None, keys::RIGHT, Modifiers::SHIFT, &mut clip, &f);
        }
        assert_eq!(t.selected_text(), "hello");
        assert_eq!(t.cursor_position(), 0);
        assert_eq!(t.selection_end(), 5);
        // A plain arrow moves relative to the *anchor*, collapsing the selection.
        t.key_typed(None, keys::LEFT, Modifiers::NONE, &mut clip, &f);
        assert_eq!(t.cursor_position(), 4);
        assert_eq!(t.selected_text(), "");
    }

    #[test]
    fn home_and_end_jump_to_the_edges() {
        let f = font();
        let mut t = field(&f);
        let mut clip = MemoryClipboard::default();
        t.key_typed(None, keys::HOME, Modifiers::NONE, &mut clip, &f);
        assert_eq!(t.cursor_position(), 0);
        t.key_typed(None, keys::END, Modifiers::NONE, &mut clip, &f);
        assert_eq!(t.cursor_position(), 11);
        t.key_typed(None, keys::HOME, Modifiers::SHIFT, &mut clip, &f);
        assert_eq!(t.selected_text(), "hello world");
    }

    #[test]
    fn typing_over_a_selection_replaces_it() {
        let f = font();
        let mut t = field(&f);
        let mut clip = MemoryClipboard::default();
        t.set_cursor_position(0, &f);
        t.set_selection_pos(5, &f);
        t.key_typed(Some('X'), 0, Modifiers::NONE, &mut clip, &f);
        assert_eq!(t.text(), "X world");
    }

    #[test]
    fn clicking_places_the_caret_at_the_nearest_character_boundary() {
        let f = font();
        let mut t = field(&f);
        // Glyphs are 5 wide, the box is at x = 10 with 4 pixels of padding: character n starts
        // at 14 + 5n.
        t.mouse_clicked(14, 25, 0, &f);
        assert_eq!(t.cursor_position(), 0);
        t.mouse_clicked(14 + 12, 25, 0, &f);
        assert_eq!(t.cursor_position(), 2, "12 px fits two 5-px glyphs");
        t.mouse_clicked(1000, 25, 0, &f);
        assert!(t.cursor_position() > 0);
    }

    #[test]
    fn clicking_outside_drops_focus_unless_forbidden() {
        let f = font();
        let mut t = field(&f);
        t.mouse_clicked(0, 0, 0, &f);
        assert!(!t.is_focused());
        t.can_lose_focus = false;
        t.set_focused(true);
        t.mouse_clicked(0, 0, 0, &f);
        assert!(t.is_focused());
    }

    #[test]
    fn the_visible_window_follows_the_caret_off_the_right_edge() {
        let f = font();
        let mut t = TextField::new(0, 0, 0, 48, 20); // inner width 40 -> 8 glyphs
        t.set_max_length(64);
        t.set_focused(true);
        t.set_text("abcdefghijklmnop", &f);
        // The caret lands at the end, so the window has scrolled to show the tail.
        assert_eq!(t.cursor_position(), 16);
        assert!(t.scroll_offset() > 0, "offset {}", t.scroll_offset());
        assert_eq!(t.visible_text(&f).chars().count(), 8);
        assert!(t.text().ends_with(&t.visible_text(&f)));
        // Jump home and the window follows back.
        t.set_cursor_position(0, &f);
        assert_eq!(t.scroll_offset(), 0);
        assert_eq!(t.visible_text(&f), "abcdefgh");
    }

    #[test]
    fn caret_blink_is_six_ticks_on_six_off() {
        let mut t = TextField::new(0, 0, 0, 50, 20);
        t.set_focused(true);
        assert!(t.caret_visible());
        for _ in 0..6 {
            t.tick();
        }
        assert!(!t.caret_visible());
        for _ in 0..6 {
            t.tick();
        }
        assert!(t.caret_visible());
        t.set_focused(false);
        assert!(!t.caret_visible(), "an unfocused field has no caret");
    }

    #[test]
    fn drawing_a_populated_field_emits_geometry_without_panicking() {
        let f = font();
        let mut font = font();
        let mut t = field(&f);
        let mut d = DrawList::new();
        t.draw(&mut d, &mut font);
        assert!(!d.is_empty());
        // Border, fill, then glyphs.
        assert!(d.batches().len() >= 2);

        // With a selection, and with the caret at the end, and empty: none of these panic.
        t.set_cursor_position(0, &f);
        t.set_selection_pos(5, &f);
        d.clear();
        t.draw(&mut d, &mut font);
        t.set_text("", &f);
        d.clear();
        t.draw(&mut d, &mut font);
        t.background = false;
        d.clear();
        t.draw(&mut d, &mut font);
    }

    /// **The proof for the font-free click.**
    ///
    /// Not "the cached path looks right" — every pixel column of every one of these fields is
    /// clicked twice, once through [`TextField::mouse_clicked`] with a real font and once
    /// through [`TextField::mouse_clicked_cached`] with none, and the two fields must end in
    /// *identical* states: caret, selection anchor and scroll window.
    #[test]
    fn the_font_free_click_agrees_with_the_font_backed_one_on_every_pixel_column() {
        for mut f in [
            crate::widget::test_support::uniform_font(),
            crate::widget::test_support::ragged_font(),
        ] {
            for text in [
                "",
                "a",
                "hello world",
                "abcdefghijklmnopqrstuvwxyz0123456789",
                "!\"#$%&'()*+,-./ mixed WIDTHS here",
                "                ",
            ] {
                for width in [24i32, 48, 100, 9] {
                    for start_cursor in [0i32, 3, 1000] {
                        let mut a = TextField::new(0, 10, 20, width, 20);
                        a.set_max_length(64);
                        a.set_focused(true);
                        a.set_text(text, &f);
                        a.set_cursor_position(start_cursor, &f);
                        let mut b = a.clone();

                        // `b` gets a draw (which is where a real screen would fill the cache).
                        let mut d = DrawList::new();
                        b.draw(&mut d, &mut f);
                        assert!(
                            b.caret_metrics_are_current(),
                            "drawing should leave the cache usable"
                        );

                        for x in (a.rect.x - 6)..=(a.rect.right() + 6) {
                            let mut a = a.clone();
                            let mut b = b.clone();
                            let y = a.rect.y + 4;
                            a.mouse_clicked(x, y, 0, &f);
                            assert!(
                                b.mouse_clicked_cached(x, y, 0),
                                "the cache should still be current at x={x}"
                            );
                            assert_eq!(
                                (a.cursor_position(), a.selection_end(), a.scroll_offset()),
                                (b.cursor_position(), b.selection_end(), b.scroll_offset()),
                                "text {text:?} width {width} column {x}"
                            );
                            assert_eq!(a.is_focused(), b.is_focused());
                        }
                    }
                }
            }
        }
    }

    /// Formatting codes are part of the Java string index even though they consume no pixels;
    /// bold also changes each following glyph's advance.  The cached click must replay those
    /// rules rather than treating every Rust `char` as an ordinary glyph.
    #[test]
    fn the_font_free_click_replays_formatting_code_widths() {
        let f = crate::widget::test_support::uniform_font();
        for text in [
            "§lbold §rplain",
            "§cblue §lbright§r normal",
            "prefix §lwide",
        ] {
            let mut a = TextField::new(0, 10, 20, 48, 20);
            a.set_max_length(64);
            a.set_focused(true);
            a.set_text(text, &f);
            let mut b = a.clone();

            let mut draw_font = crate::widget::test_support::uniform_font();
            let mut d = DrawList::new();
            b.draw(&mut d, &mut draw_font);
            assert!(b.caret_metrics_are_current());

            for x in (a.rect.x - 6)..=(a.rect.right() + 6) {
                let mut backed = a.clone();
                let mut cached = b.clone();
                let y = backed.rect.y + 4;
                backed.mouse_clicked(x, y, 0, &f);
                assert!(cached.mouse_clicked_cached(x, y, 0));
                assert_eq!(
                    (
                        backed.cursor_position(),
                        backed.selection_end(),
                        backed.scroll_offset(),
                    ),
                    (
                        cached.cursor_position(),
                        cached.selection_end(),
                        cached.scroll_offset(),
                    ),
                    "formatted text {text:?} at column {x}"
                );
            }
        }
    }

    /// The one thing a cache cannot do is answer for text it never saw.
    #[test]
    fn the_cache_reports_itself_stale_after_an_edit_a_resize_or_no_draw_at_all() {
        let mut f = crate::widget::test_support::ragged_font();
        let mut t = TextField::new(0, 10, 20, 100, 20);
        t.set_max_length(64);
        t.set_focused(true);
        t.set_text("hello world", &f);
        assert!(
            !t.caret_metrics_are_current(),
            "a field that has never been drawn has nothing cached"
        );
        assert!(
            !t.mouse_clicked_cached(20, 24, 0),
            "and says so instead of guessing"
        );

        let mut d = DrawList::new();
        t.draw(&mut d, &mut f);
        assert!(t.caret_metrics_are_current());

        // An edit invalidates it.
        let mut edited = t.clone();
        edited.write_text("!", &f);
        assert!(!edited.caret_metrics_are_current());

        // So does moving the caret, which can move the window.
        let mut moved = t.clone();
        moved.set_cursor_position(0, &f);
        assert!(!moved.caret_metrics_are_current());

        // So does resizing the box.
        let mut resized = t.clone();
        resized.rect.width = 40;
        assert!(!resized.caret_metrics_are_current());
        assert!(!resized.mouse_clicked_cached(20, 24, 0));

        // And a successful cached click consumes it, because the window may have scrolled.
        assert!(t.mouse_clicked_cached(20, 24, 0));
        assert!(!t.caret_metrics_are_current());
    }

    /// Focus is font-free, so it is handled whether or not the caret can be placed — a click
    /// outside must still be able to *drop* focus on a field that has never been drawn.
    #[test]
    fn a_stale_cached_click_still_moves_focus() {
        let f = crate::widget::test_support::uniform_font();
        let mut t = field(&f);
        assert!(t.is_focused());
        assert!(
            t.mouse_clicked_cached(0, 0, 0),
            "a click outside places no caret in vanilla either"
        );
        assert!(!t.is_focused());

        t.can_lose_focus = false;
        t.set_focused(true);
        t.mouse_clicked_cached(0, 0, 0);
        assert!(t.is_focused());
    }

    /// A click that scrolls the window is the interesting case: vanilla's `setSelectionPos` has
    /// a branch that runs a *reverse* trim over the whole string when the caret lands on the
    /// first visible character, and the cached path has to take it too.
    // java: GuiTextField.java:739
    #[test]
    fn clicking_the_first_visible_character_scrolls_the_window_back_like_vanilla() {
        let mut f = crate::widget::test_support::ragged_font();
        let mut a = TextField::new(0, 0, 0, 48, 20);
        a.set_max_length(64);
        a.set_focused(true);
        a.set_text("abcdefghijklmnopqrstuvwxyz", &f);
        assert!(a.scroll_offset() > 0, "the window has scrolled to the tail");
        let mut b = a.clone();

        let mut d = DrawList::new();
        b.draw(&mut d, &mut f);

        // x = 0 is the first visible column, i.e. `pos == lineScrollOffset`.
        a.mouse_clicked(0, 4, 0, &f);
        assert!(b.mouse_clicked_cached(0, 4, 0));
        assert_eq!(a.scroll_offset(), b.scroll_offset());
        assert_eq!(a.cursor_position(), b.cursor_position());
        assert!(
            b.scroll_offset() < a.max_length(),
            "the window really did move"
        );
    }

    #[test]
    fn inner_width_accounts_for_the_border() {
        let mut t = TextField::new(0, 0, 0, 100, 20);
        assert_eq!(t.inner_width(), 92);
        t.background = false;
        assert_eq!(t.inner_width(), 100);
    }
}
