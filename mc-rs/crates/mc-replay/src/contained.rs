//! Helper threads that recover from their own panic.
//!
//! The client's panic hook records the first panic on any thread as the run's crash report
//! (shown, and the launcher's error dialog, at the next quit), unless the thread is inside
//! `mc_core::panic_scope` when it panics. This crate's helper threads each have a failure state
//! their owner already shows or works around: the writer's ("Recording failed"), the
//! microphone's and the voice output's error, desktop audio's status. A panic on one of them
//! becomes that failure, and the game goes on.

/// Run a helper thread's `body` contained; if it panics, hand `failed` the message
/// `"the <what> stopped after an internal error"` (the panic's own message is already in the log:
/// the hook prints it).
pub(crate) fn run_helper(what: &str, body: impl FnOnce(), failed: impl FnOnce(String)) {
    if mc_core::panic_scope::contain(body).is_err() {
        failed(format!("the {what} stopped after an internal error"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Contained at the moment of the panic (what the hook checks, before unwinding), and the
    /// failure reaches the owner's state. A body that ends normally reports nothing.
    #[test]
    fn a_panicking_helper_is_contained_and_reports_its_failure() {
        let mut contained_at_panic = false;
        let mut reported = None;
        run_helper(
            "microphone",
            || {
                contained_at_panic = mc_core::panic_scope::is_contained();
                panic!("device call failed");
            },
            |message| reported = Some(message),
        );
        assert!(contained_at_panic);
        assert_eq!(
            reported.as_deref(),
            Some("the microphone stopped after an internal error")
        );
        assert!(
            !mc_core::panic_scope::is_contained(),
            "the scope ends with the call"
        );

        let mut ran = false;
        run_helper("microphone", || ran = true, |_| panic!("nothing failed"));
        assert!(ran);
    }
}
