//! LWJGL key codes and modifier state.
//!
//! java: `org.lwjgl.input.Keyboard`, used raw throughout `client/gui/`
//!
//! Vanilla's GUI code compares against LWJGL scan codes (`case 203:` is left-arrow), and S10's
//! [`mc_render::Input`] already speaks the same numbering — `Input::from_lwjgl` /
//! `Input::to_lwjgl` are the bridge from winit. Naming them here keeps the widget code readable
//! without inventing a second key enumeration for the UI layer to translate through.

/// Escape — closes the screen.
pub const ESCAPE: i32 = 1;
/// Backspace.
pub const BACKSPACE: i32 = 14;
/// Tab — moves focus forward (Shift+Tab moves it back).
pub const TAB: i32 = 15;
/// Return.
pub const ENTER: i32 = 28;
/// `A`, for Ctrl+A.
pub const A: i32 = 30;
/// Space.
pub const SPACE: i32 = 57;
/// `X`, for Ctrl+X.
pub const X: i32 = 45;
/// `C`, for Ctrl+C.
pub const C: i32 = 46;
/// `V`, for Ctrl+V.
pub const V: i32 = 47;
/// Numpad Enter.
pub const NUMPAD_ENTER: i32 = 156;
/// Home.
pub const HOME: i32 = 199;
/// Up arrow.
pub const UP: i32 = 200;
/// Left arrow.
pub const LEFT: i32 = 203;
/// Right arrow.
pub const RIGHT: i32 = 205;
/// End.
pub const END: i32 = 207;
/// Down arrow.
pub const DOWN: i32 = 208;
/// Delete.
pub const DELETE: i32 = 211;

/// Which modifier keys are held.
///
/// Vanilla reads these from LWJGL's global keyboard state at the moment a key event is handled
/// (`GuiScreen.isCtrlKeyDown` and friends); passing them explicitly keeps the widgets pure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Modifiers {
    /// Either Control (or, on macOS, either Command — vanilla substitutes the same way).
    pub ctrl: bool,
    /// Either Shift.
    pub shift: bool,
    /// Either Alt.
    pub alt: bool,
}

impl Modifiers {
    /// No modifiers held.
    pub const NONE: Self = Self {
        ctrl: false,
        shift: false,
        alt: false,
    };
    /// Control only.
    pub const CTRL: Self = Self {
        ctrl: true,
        shift: false,
        alt: false,
    };
    /// Shift only.
    pub const SHIFT: Self = Self {
        ctrl: false,
        shift: true,
        alt: false,
    };

    /// Vanilla's "is this the Ctrl+`key` combo?" test: Control down, Shift and Alt **up**.
    // java: GuiScreen.java:765 (isKeyComboCtrlX and friends)
    #[must_use]
    pub const fn is_ctrl_combo(self, pressed: i32, wanted: i32) -> bool {
        pressed == wanted && self.ctrl && !self.shift && !self.alt
    }
}

/// Whether a character may be typed into chat or a text field.
///
/// Everything from `U+0020` up except the section sign and `DEL`.
// java: util/ChatAllowedCharacters.java:10
#[must_use]
pub const fn is_allowed_character(c: char) -> bool {
    c as u32 != 167 && c as u32 >= 32 && c as u32 != 127
}

/// Drop every character [`is_allowed_character`] rejects.
// java: util/ChatAllowedCharacters.java:16
#[must_use]
pub fn filter_allowed_characters(input: &str) -> String {
    input.chars().filter(|c| is_allowed_character(*c)).collect()
}

/// Somewhere to copy text to and paste text from.
///
/// Vanilla goes straight to `java.awt.Toolkit`; the seam exists so headless tests (and the
/// eventual winit-side implementation in S16) can swap it out. A failed read returns `""`,
/// exactly as vanilla's swallowed exception does.
pub trait Clipboard {
    /// The clipboard's current text, or `""` if there is none.
    fn get(&self) -> String;
    /// Replace the clipboard's text. Vanilla ignores empty strings.
    fn set(&mut self, text: &str);
}

/// An in-process clipboard: the default, and what the tests use.
#[derive(Debug, Clone, Default)]
pub struct MemoryClipboard(String);

impl Clipboard for MemoryClipboard {
    fn get(&self) -> String {
        self.0.clone()
    }
    fn set(&mut self, text: &str) {
        // java: GuiScreen.setClipboardString ignores empty input.
        if !text.is_empty() {
            self.0 = text.to_owned();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctrl_combos_require_ctrl_alone() {
        assert!(Modifiers::CTRL.is_ctrl_combo(A, A));
        assert!(!Modifiers::CTRL.is_ctrl_combo(C, A), "wrong key");
        assert!(!Modifiers::NONE.is_ctrl_combo(A, A), "no ctrl");
        assert!(
            !Modifiers {
                ctrl: true,
                shift: true,
                alt: false
            }
            .is_ctrl_combo(A, A),
            "shift disqualifies it"
        );
        assert!(
            !Modifiers {
                ctrl: true,
                shift: false,
                alt: true
            }
            .is_ctrl_combo(A, A),
            "alt disqualifies it"
        );
    }

    #[test]
    fn allowed_characters_exclude_control_codes_and_the_section_sign() {
        assert!(is_allowed_character('a'));
        assert!(is_allowed_character(' '));
        assert!(is_allowed_character('\u{e9}'));
        assert!(!is_allowed_character('\n'));
        assert!(!is_allowed_character('\t'));
        assert!(!is_allowed_character('\u{7f}'), "DEL");
        assert!(!is_allowed_character('§'));
        assert_eq!(filter_allowed_characters("a\nb§c\u{7f}"), "abc");
    }

    #[test]
    fn the_memory_clipboard_ignores_empty_writes() {
        let mut c = MemoryClipboard::default();
        assert_eq!(c.get(), "");
        c.set("hello");
        assert_eq!(c.get(), "hello");
        c.set("");
        assert_eq!(c.get(), "hello", "vanilla skips empty copies");
    }
}
