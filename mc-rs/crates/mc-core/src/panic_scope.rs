//! Panics the code recovers from.
//!
//! The game binary installs a panic hook that turns a panic on **any** thread into the run's
//! crash report and stops the game (java: `Minecraft.crashed`, which the run loop checks every
//! iteration). A hook runs before the unwinding that `catch_unwind` stops, so it cannot tell a
//! panic somebody will catch from one nobody will. Work that is allowed to fail on its own (one
//! chunk packet unpacked off-thread, one skin decoded, one background preparation) runs inside
//! [`contain`]: the hook sees [`is_contained`] and leaves the crash report alone, the caller gets
//! the panic back as an `Err`, and the game goes on, as vanilla does when a download or a sound
//! thread throws and logs.
//!
//! Containment is per thread and nests. Code outside any [`contain`] call is not contained.

use std::cell::Cell;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

thread_local! {
    static DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Run `work`, catching a panic in it as an `Err` that the crash-report hook ignores.
///
/// The closure is treated as unwind safe: like every other recovery site in the client, the
/// caller must not reuse state the panic may have left half-updated (the usual shape is a pure
/// function of owned inputs whose result is simply dropped on failure).
///
/// # Errors
///
/// The panic payload, if `work` panicked.
pub fn contain<R>(work: impl FnOnce() -> R) -> std::thread::Result<R> {
    let _scope = Scope::enter();
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(work))
}

/// Run a thread's whole body contained, for a thread whose owner learns of a panic from
/// `JoinHandle::join` (an encoder thread, a file-finishing thread: the game waits for it and
/// reports a panic there as a failed export, then goes on).
///
/// A panic is passed on with `resume_unwind`, which does not run the panic hook a second time:
/// the owner's `join` still returns it exactly as before, but the hook saw it inside the scope
/// and did not take it for the run's crash. Use it as the thread's outermost call.
pub fn contain_for_join<R>(work: impl FnOnce() -> R) -> R {
    match contain(work) {
        Ok(value) => value,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

/// Whether the current thread is inside [`contain`] (or polling a [`Contained`] future). The
/// panic hook asks this first.
#[must_use]
pub fn is_contained() -> bool {
    DEPTH.with(|depth| depth.get() > 0)
}

/// A future whose every poll runs contained, for async work a runtime catches on its own (a
/// task's `JoinHandle` reports the panic as an error). The panic still ends the task; it just is
/// not the game's crash.
#[derive(Debug)]
pub struct Contained<F> {
    inner: Pin<Box<F>>,
}

impl<F: Future> Contained<F> {
    /// Wrap `future`.
    pub fn new(future: F) -> Self {
        Self {
            inner: Box::pin(future),
        }
    }
}

impl<F: Future> Future for Contained<F> {
    type Output = F::Output;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let _scope = Scope::enter();
        self.inner.as_mut().poll(cx)
    }
}

/// One level of containment on this thread, released on drop (including while unwinding).
struct Scope;

impl Scope {
    fn enter() -> Self {
        DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
        Scope
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[test]
    fn contain_returns_the_value_or_the_panic() {
        assert_eq!(contain(|| 7).expect("no panic"), 7);
        let caught = contain(|| -> u32 { panic!("decode failed") });
        let payload = caught.expect_err("the panic comes back as an error");
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"decode failed"));
        assert!(
            !is_contained(),
            "the scope ends with the call, panic or not"
        );
    }

    /// What the crash hook sees: contained while the work runs (the hook runs before unwinding),
    /// not contained before or after. Negative control: a bare `catch_unwind` is not contained.
    #[test]
    fn the_hook_sees_containment_only_inside_contain() {
        let seen = Arc::new(AtomicU32::new(0));
        let contained_panics = Arc::new(AtomicU32::new(0));
        let thread = {
            let seen = Arc::clone(&seen);
            let contained_panics = Arc::clone(&contained_panics);
            std::thread::spawn(move || {
                // A hook is process-wide; this one only counts panics from this test's thread.
                let me = std::thread::current().id();
                let previous = std::panic::take_hook();
                std::panic::set_hook(Box::new(move |info| {
                    if std::thread::current().id() == me {
                        seen.fetch_add(1, Ordering::SeqCst);
                        if is_contained() {
                            contained_panics.fetch_add(1, Ordering::SeqCst);
                        }
                    } else {
                        previous(info);
                    }
                }));
                assert!(!is_contained());
                let _ = contain(|| panic!("inside"));
                let _ = std::panic::catch_unwind(|| panic!("bare catch_unwind"));
                let nested = contain(|| contain(|| panic!("nested")).is_err());
                assert_eq!(nested.ok(), Some(true));
                assert!(!is_contained());
                // The hook stays installed: it hands every other thread's panic to the previous
                // hook, and this thread's id is never reused.
            })
        };
        thread
            .join()
            .expect("the test thread itself does not panic");
        assert_eq!(seen.load(Ordering::SeqCst), 3);
        assert_eq!(
            contained_panics.load(Ordering::SeqCst),
            2,
            "the bare catch_unwind is the one uncontained panic"
        );
    }

    /// A thread whose body runs through `contain_for_join`: contained at the moment it panics
    /// (what the hook checks, before any unwinding), and its owner still gets the panic, with
    /// its payload, from `join`. Negative control: the same body spawned bare is not contained.
    #[test]
    fn a_thread_contained_for_join_still_hands_its_panic_to_join() {
        use std::sync::atomic::AtomicBool;
        let contained_at_panic = Arc::new(AtomicBool::new(false));
        let seen = Arc::clone(&contained_at_panic);
        let thread = std::thread::spawn(move || {
            contain_for_join(move || -> u32 {
                seen.store(is_contained(), Ordering::SeqCst);
                panic!("encoder failed");
            })
        });
        let payload = thread.join().expect_err("the owner still sees the panic");
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"encoder failed"));
        assert!(contained_at_panic.load(Ordering::SeqCst));

        let bare_seen = Arc::new(AtomicBool::new(true));
        let seen = Arc::clone(&bare_seen);
        let bare = std::thread::spawn(move || {
            seen.store(is_contained(), Ordering::SeqCst);
            panic!("encoder failed");
        });
        assert!(bare.join().is_err());
        assert!(
            !bare_seen.load(Ordering::SeqCst),
            "a bare thread's panic is the run's crash"
        );

        let fine = std::thread::spawn(|| contain_for_join(|| 5_u32));
        assert_eq!(fine.join().expect("no panic"), 5);
    }

    #[test]
    fn a_contained_future_is_contained_while_polled() {
        struct Probe;
        impl Future for Probe {
            type Output = bool;
            fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<bool> {
                Poll::Ready(is_contained())
            }
        }
        let waker = std::task::Waker::noop();
        let mut cx = Context::from_waker(waker);
        let mut wrapped = Contained::new(Probe);
        assert_eq!(
            Pin::new(&mut wrapped).poll(&mut cx),
            Poll::Ready(true),
            "polled inside the scope"
        );
        assert!(!is_contained(), "and released after the poll");
        let mut bare = Probe;
        assert_eq!(Pin::new(&mut bare).poll(&mut cx), Poll::Ready(false));
    }
}
