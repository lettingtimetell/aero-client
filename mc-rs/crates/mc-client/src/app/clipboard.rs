//! The real system clipboard.
//!
//! java: `GuiScreen.getClipboardString`/`setClipboardString`, which go through
//! `java.awt.Toolkit.getDefaultToolkit().getSystemClipboard()` and **swallow every exception** —
//! a headless or busy clipboard silently reads as `""` and writes are dropped.
//!
//! S15 defined the [`Clipboard`] trait, S16 built `screens::clipboard::SystemClipboard` as a
//! pair of closures with vanilla's swallow-the-exception behaviour already on that side, and
//! recorded that the closures themselves cannot live in `mc-ui`: winit 0.30 has no
//! clipboard API, every desktop clipboard crate needs platform FFI, and `mc-ui` is
//! `#![forbid(unsafe_code)]` (the no-`unsafe` rule exempts only `mc-render`). `mc-client` can host it
//! because the FFI is inside the dependency, not in this crate — this file is still
//! `unsafe`-free.

use std::cell::RefCell;
use std::sync::{Mutex, MutexGuard, OnceLock};

use copypasta::{ClipboardContext, ClipboardProvider};
use mc_ui::widget::Clipboard;

/// Serialises every call into the clipboard back-end.
///
/// **This is not defensive tidiness; it is a fix for a real crash.** `copypasta 0.10` resolves
/// to `clipboard-win 3.1.1` on Windows, and opening or reading the clipboard from two threads at
/// once reliably aborts the process with `STATUS_HEAP_CORRUPTION` (`0xC0000374`). Found by this
/// module's own test suite: the tests below pass one at a time and killed the whole test binary
/// when run in parallel, which is cargo's default. There is exactly one OS clipboard, so
/// serialising costs nothing — vanilla likewise goes through a single
/// `Toolkit.getSystemClipboard()`.
static CLIPBOARD_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

/// Take the back-end lock, ignoring poisoning: a panic in another thread's clipboard call says
/// nothing about whether the OS clipboard is usable now. Also taken by the screenshot `[COPY]`
/// image write (`app/mods/x2_screenshot.rs`), which runs on the encoder thread.
pub(crate) fn lock() -> MutexGuard<'static, ()> {
    CLIPBOARD_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The OS clipboard, or a no-op when the platform has none.
///
/// The context is behind a [`RefCell`] because [`Clipboard::get`] takes `&self` while
/// `copypasta`'s `get_contents` takes `&mut self`. This is not the world/entity graph, so
/// the arena-index rule does not apply; there is exactly one of these, owned by the client, and it is
/// never shared across threads.
pub struct SystemClipboard {
    context: Option<RefCell<ClipboardContext>>,
    last_error: RefCell<Option<String>>,
}

impl std::fmt::Debug for SystemClipboard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SystemClipboard")
            .field("available", &self.context.is_some())
            .field("last_error", &self.last_error.borrow())
            .finish()
    }
}

impl Default for SystemClipboard {
    fn default() -> Self {
        SystemClipboard::new()
    }
}

impl SystemClipboard {
    /// Open the OS clipboard. Never fails: a platform without one degrades to a clipboard that
    /// reads empty and drops writes, which is what vanilla does when AWT is headless.
    #[must_use]
    pub fn new() -> SystemClipboard {
        let _guard = lock();
        match ClipboardContext::new() {
            Ok(context) => SystemClipboard {
                context: Some(RefCell::new(context)),
                last_error: RefCell::new(None),
            },
            Err(error) => {
                tracing::warn!(%error, "no system clipboard; copy and paste will do nothing");
                SystemClipboard {
                    context: None,
                    last_error: RefCell::new(Some(error.to_string())),
                }
            }
        }
    }

    /// Whether a real clipboard is behind this.
    #[must_use]
    pub fn available(&self) -> bool {
        self.context.is_some()
    }

    /// The most recent failure, for the F3 overlay or a crash report.
    #[must_use]
    pub fn last_error(&self) -> Option<String> {
        self.last_error.borrow().clone()
    }
}

impl Clipboard for SystemClipboard {
    /// java: `getClipboardString` returns `""` on any `Exception`.
    fn get(&self) -> String {
        let Some(context) = &self.context else {
            return String::new();
        };
        let Ok(mut context) = context.try_borrow_mut() else {
            return String::new();
        };
        let _guard = lock();
        match context.get_contents() {
            Ok(text) => {
                *self.last_error.borrow_mut() = None;
                text
            }
            Err(error) => {
                // An empty clipboard is an `Err` on Windows, so this is the common path and must
                // not be noisy.
                tracing::trace!(%error, "clipboard read failed");
                *self.last_error.borrow_mut() = Some(error.to_string());
                String::new()
            }
        }
    }

    /// java: `setClipboardString` ignores an empty string and swallows any `Exception`.
    fn set(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        let Some(context) = &self.context else {
            return;
        };
        let Ok(mut context) = context.try_borrow_mut() else {
            return;
        };
        let _guard = lock();
        match context.set_contents(text.to_owned()) {
            Ok(()) => *self.last_error.borrow_mut() = None,
            Err(error) => {
                tracing::warn!(%error, "clipboard write failed");
                *self.last_error.borrow_mut() = Some(error.to_string());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The OS clipboard is one value for the whole machine. [`lock`] serialises single calls, but
    /// a test that writes and then reads back needs the clipboard to itself in between: without
    /// this, the many-threads test's writes landed between the round trip's write and read (3 of
    /// 200 runs of this module at four test threads read "thread 2 round 7" and the like). Every
    /// test here that writes the real clipboard holds this for its whole body.
    static OS_CLIPBOARD_TESTS: Mutex<()> = Mutex::new(());

    fn os_clipboard_to_this_test() -> MutexGuard<'static, ()> {
        OS_CLIPBOARD_TESTS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// A real round trip through the OS clipboard. This is the test that makes the §13 request
    /// closed rather than merely implemented — the bytes go out to Windows and come back.
    ///
    /// Skipped rather than failed where no clipboard exists (a headless Linux CI box), because
    /// the absence of a clipboard is a property of the machine, not a bug in this file.
    #[test]
    fn a_real_round_trip_through_the_os_clipboard() {
        let _alone = os_clipboard_to_this_test();
        let mut clipboard = SystemClipboard::new();
        if !clipboard.available() {
            eprintln!("skip: no system clipboard on this machine");
            return;
        }
        // Unicode and a `§` code, because chat and text fields both carry them.
        let payload = "mc-rs \u{a7}6clipboard \u{4e00} test";
        clipboard.set(payload);
        assert_eq!(clipboard.get(), payload);
        assert_eq!(clipboard.last_error(), None);

        // java: `setClipboardString` returns early on an empty string, so the old contents stay.
        clipboard.set("");
        assert_eq!(
            clipboard.get(),
            payload,
            "an empty write must not clear the clipboard"
        );
    }

    /// Vanilla never lets the clipboard throw into the GUI. Reading from an unavailable
    /// clipboard must be an empty string, not a panic.
    #[test]
    fn an_unavailable_clipboard_reads_empty_and_swallows_writes() {
        let mut clipboard = SystemClipboard {
            context: None,
            last_error: RefCell::new(Some("no clipboard".to_owned())),
        };
        assert!(!clipboard.available());
        assert_eq!(clipboard.get(), "");
        clipboard.set("anything");
        assert_eq!(clipboard.get(), "");
        assert_eq!(clipboard.last_error().as_deref(), Some("no clipboard"));
    }

    /// It must satisfy the trait S15 froze, since that is the only thing `mc-ui` will accept.
    #[test]
    fn it_is_usable_as_the_ui_crates_clipboard_trait_object() {
        let mut clipboard = SystemClipboard::new();
        let object: &mut dyn Clipboard = &mut clipboard;
        object.set("");
        let _ = object.get();
    }

    /// The regression test for [`CLIPBOARD_LOCK`]. Without the lock this aborts the whole test
    /// binary with `STATUS_HEAP_CORRUPTION` rather than failing an assertion, so it is written
    /// to hammer the back-end from several threads at once on purpose.
    #[test]
    fn concurrent_use_from_many_threads_does_not_corrupt_the_heap() {
        let _alone = os_clipboard_to_this_test();
        if !SystemClipboard::new().available() {
            eprintln!("skip: no system clipboard on this machine");
            return;
        }
        let handles: Vec<_> = (0..8)
            .map(|index| {
                std::thread::spawn(move || {
                    for round in 0..8 {
                        let mut clipboard = SystemClipboard::new();
                        clipboard.set(&format!("thread {index} round {round}"));
                        let _ = clipboard.get();
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("no thread may panic");
        }
    }
}
