//! The system clipboard seam.
//!
//! java: `client/gui/GuiScreen.java` (`getClipboardString` / `setClipboardString`, which call
//! `java.awt.Toolkit.getDefaultToolkit().getSystemClipboard()` and swallow every exception)
//!
//! # Why this is a seam and not a `winit` call
//!
//! S15 shipped [`Clipboard`] plus an in-memory stub and handed S16
//! the job of implementing it "over winit". **winit does not have a clipboard API.** As of
//! 0.30.13 the only mention of one in the whole crate is
//! `// TODO(kchibisov): Remove `cfg` when the clipboard will be added.` in `event_loop.rs`;
//! there is no `Window::clipboard`, no `ClipboardEvent`, nothing. Clipboard access on the
//! desktop is a separate crate (`arboard`, `copypasta`, `clipboard-win`), and every one of them
//! needs platform FFI — which `mc-ui` cannot host, because the crate is `#![forbid(unsafe_code)]`
//! (the no-`unsafe` rule exempts only `mc-render`).
//!
//! So what lands here is the honest shape of the problem:
//!
//! * [`SystemClipboard`] is a [`Clipboard`] that delegates to two
//!   caller-supplied closures. `mc-client` — which already has `anyhow` and no `forbid(unsafe)`
//!   restriction on its dependency graph — adds `arboard` and wires it in with three lines.
//! * Vanilla's error behaviour is reproduced *here*, not in the backend: a failed read is `""`
//!   and an empty write is ignored, so a backend that returns `None` behaves exactly as
//!   `java.awt`'s swallowed `Exception` does.
//! * [`NullClipboard`] is what a headless build (and every test that does not care) uses.
//!
//! See §13 of the port plan for the cross-section request that closes this properly.

use crate::widget::keys::Clipboard;

/// A [`Clipboard`] backed by two caller-supplied closures.
///
/// The closures are the platform boundary. `get` returning `None` and `set` returning `false`
/// both mean "the platform refused", and both are swallowed exactly as vanilla swallows
/// `java.awt`'s exceptions.
///
/// ```
/// use std::cell::RefCell;
/// use mc_ui::screens::clipboard::SystemClipboard;
/// use mc_ui::widget::Clipboard;
///
/// // Stand-in for `arboard::Clipboard`, which lives in `mc-client`.
/// let backing = RefCell::new(String::new());
/// let mut c = SystemClipboard::new(
///     || Some(backing.borrow().clone()),
///     |t| {
///         *backing.borrow_mut() = t.to_owned();
///         true
///     },
/// );
/// c.set("hello");
/// assert_eq!(c.get(), "hello");
/// ```
pub struct SystemClipboard<G, S> {
    get: G,
    set: S,
    last_error: Option<&'static str>,
}

impl<G, S> std::fmt::Debug for SystemClipboard<G, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SystemClipboard")
            .field("last_error", &self.last_error)
            .finish_non_exhaustive()
    }
}

impl<G, S> SystemClipboard<G, S>
where
    G: Fn() -> Option<String>,
    S: FnMut(&str) -> bool,
{
    /// Wrap a platform backend.
    pub fn new(get: G, set: S) -> Self {
        Self {
            get,
            set,
            last_error: None,
        }
    }

    /// Why the last operation failed, if it did. Vanilla logs and moves on; so does this.
    #[must_use]
    pub fn last_error(&self) -> Option<&'static str> {
        self.last_error
    }
}

impl<G, S> Clipboard for SystemClipboard<G, S>
where
    G: Fn() -> Option<String>,
    S: FnMut(&str) -> bool,
{
    fn get(&self) -> String {
        // java: GuiScreen.getClipboardString catches Exception and returns "".
        (self.get)().unwrap_or_default()
    }

    fn set(&mut self, text: &str) {
        // java: GuiScreen.setClipboardString ignores empty input before touching the toolkit.
        if text.is_empty() {
            return;
        }
        if (self.set)(text) {
            self.last_error = None;
        } else {
            self.last_error = Some("the platform clipboard refused the write");
        }
    }
}

/// A clipboard that is always empty and discards every write.
///
/// What a headless run gets. Copy still *appears* to work from the widget's point of view —
/// vanilla behaves the same way when `java.awt` is unavailable.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullClipboard;

impl Clipboard for NullClipboard {
    fn get(&self) -> String {
        String::new()
    }
    fn set(&mut self, _text: &str) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn a_backend_read_failure_becomes_the_empty_string_not_a_panic() {
        let c = SystemClipboard::new(|| None, |_| true);
        assert_eq!(c.get(), "", "vanilla swallows the exception");
    }

    #[test]
    fn an_empty_write_never_reaches_the_backend() {
        let hits = RefCell::new(0);
        let mut c = SystemClipboard::new(
            || Some(String::new()),
            |_| {
                *hits.borrow_mut() += 1;
                true
            },
        );
        c.set("");
        assert_eq!(*hits.borrow(), 0);
        c.set("x");
        assert_eq!(*hits.borrow(), 1);
    }

    #[test]
    fn a_refused_write_is_recorded_but_not_fatal() {
        let mut c = SystemClipboard::new(|| Some(String::new()), |_| false);
        assert!(c.last_error().is_none());
        c.set("nope");
        assert!(c.last_error().is_some());
    }

    #[test]
    fn a_round_trip_through_the_backend_preserves_the_text() {
        let backing = RefCell::new(String::new());
        let mut c = SystemClipboard::new(
            || Some(backing.borrow().clone()),
            |t| {
                *backing.borrow_mut() = t.to_owned();
                true
            },
        );
        c.set("/tp 0 64 0");
        assert_eq!(c.get(), "/tp 0 64 0");
    }

    #[test]
    fn the_null_clipboard_is_inert() {
        let mut c = NullClipboard;
        c.set("anything");
        assert_eq!(c.get(), "");
    }

    /// The `TextField` widget S15 shipped drives the trait, so this checks the seam actually
    /// fits the consumer rather than only its own shape.
    #[test]
    fn a_text_field_pastes_through_the_system_clipboard() {
        use crate::widget::keys::{self, Modifiers};
        use crate::widget::TextField;

        let font = crate::widget::test_support::uniform_font();
        let backing = RefCell::new("pasted".to_owned());
        let mut clip = SystemClipboard::new(
            || Some(backing.borrow().clone()),
            |t| {
                *backing.borrow_mut() = t.to_owned();
                true
            },
        );
        let mut field = TextField::new(0, 0, 0, 200, 20);
        field.set_focused(true);
        field.key_typed(None, keys::V, Modifiers::CTRL, &mut clip, &font);
        assert_eq!(field.text(), "pasted");
    }
}
