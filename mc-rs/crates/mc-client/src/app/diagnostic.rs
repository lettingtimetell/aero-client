//! Progress state for the explicitly requested deterministic replay diagnostic.
//!
//! The driver owns the frame boundary and calls [`DiagnosticProgress::advance`] once after each
//! simulation/render-preparation frame. A skipped surface acquisition invalidates the shell
//! trace separately. This module has no clock, renderer handle, or mesh queue access:
//! readiness is one boolean supplied by the caller at the fixed warmup boundary.

use super::config::DiagnosticSimulation;

/// Stable failure reported when the diagnostic barrier is not ready at the exact warmup boundary.
const READINESS_BARRIER_FAILURE: &str =
    "diagnostic warmup reached its boundary before the renderer was ready";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProgressState {
    Waiting,
    Frozen,
    Failed,
}

/// Pure per-session progress for the frozen steady-state diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DiagnosticProgress {
    warmup_frames: u64,
    warmup_frames_elapsed: u64,
    state: ProgressState,
}

impl DiagnosticProgress {
    /// Start a progress machine with the configured, fixed warmup length.
    pub(super) fn new(simulation: DiagnosticSimulation) -> Self {
        Self {
            warmup_frames: simulation.warmup_frames,
            warmup_frames_elapsed: 0,
            state: ProgressState::Waiting,
        }
    }

    /// Advance one simulation frame after replay exhaustion, if the caller supplies that boundary.
    ///
    /// Frames before replay exhaustion do not count. Once the fixed boundary is reached, the
    /// readiness barrier is evaluated exactly once: readiness freezes the diagnostic, while an
    /// unready renderer fails it permanently. There is no extra waiting for meshes after failure.
    pub(super) fn advance(&mut self, exhausted: bool, ready: bool) {
        if !exhausted || self.state != ProgressState::Waiting {
            return;
        }

        if self.warmup_frames_elapsed < self.warmup_frames {
            self.warmup_frames_elapsed += 1;
        }
        if self.warmup_frames_elapsed != self.warmup_frames {
            return;
        }

        self.state = if ready {
            ProgressState::Frozen
        } else {
            ProgressState::Failed
        };
    }

    /// Whether the exact warmup boundary passed the readiness barrier.
    #[must_use]
    pub(super) fn is_frozen(&self) -> bool {
        self.state == ProgressState::Frozen
    }

    /// The stable barrier reason after a failed boundary, or `None` while waiting/frozen.
    #[must_use]
    pub(super) fn failure(&self) -> Option<&'static str> {
        (self.state == ProgressState::Failed).then_some(READINESS_BARRIER_FAILURE)
    }

    /// Number of simulation frames counted after replay exhaustion.
    #[must_use]
    pub(super) fn warmup_frames_elapsed(&self) -> u64 {
        self.warmup_frames_elapsed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn progress(warmup_frames: u64) -> DiagnosticProgress {
        DiagnosticProgress::new(DiagnosticSimulation {
            seed: 7,
            warmup_frames,
        })
    }

    #[test]
    fn frames_before_replay_exhaustion_do_not_count_or_freeze() {
        let mut progress = progress(2);

        progress.advance(false, true);
        progress.advance(false, true);
        progress.advance(false, true);

        assert_eq!(progress.warmup_frames_elapsed(), 0);
        assert!(!progress.is_frozen());
        assert_eq!(progress.failure(), None);
    }

    #[test]
    fn readiness_freezes_at_the_exact_warmup_boundary() {
        let mut progress = progress(2);

        progress.advance(true, true);
        assert_eq!(progress.warmup_frames_elapsed(), 1);
        assert!(!progress.is_frozen());

        progress.advance(true, true);
        assert_eq!(progress.warmup_frames_elapsed(), 2);
        assert!(progress.is_frozen());
        assert_eq!(progress.failure(), None);
    }

    #[test]
    fn an_unready_boundary_fails_without_an_implicit_extension() {
        let mut progress = progress(2);

        progress.advance(true, true);
        progress.advance(true, false);
        let reason = progress
            .failure()
            .expect("the exact boundary must apply the readiness barrier");
        assert_eq!(reason, READINESS_BARRIER_FAILURE);
        assert!(!progress.is_frozen());
        assert_eq!(progress.warmup_frames_elapsed(), 2);

        progress.advance(true, true);
        assert_eq!(progress.failure(), Some(reason));
        assert_eq!(progress.warmup_frames_elapsed(), 2);
    }

    #[test]
    fn frozen_state_is_stable_and_does_not_count_later_frames() {
        let mut progress = progress(1);
        progress.advance(true, true);
        progress.advance(false, false);
        progress.advance(true, false);

        assert!(progress.is_frozen());
        assert_eq!(progress.failure(), None);
        assert_eq!(progress.warmup_frames_elapsed(), 1);
    }

    #[test]
    fn failed_state_is_stable_and_keeps_the_static_reason() {
        let mut progress = progress(1);
        progress.advance(true, false);
        let reason = progress.failure().expect("failed boundary");

        progress.advance(false, true);
        progress.advance(true, true);

        assert!(!progress.is_frozen());
        assert_eq!(progress.failure(), Some(reason));
        assert_eq!(progress.warmup_frames_elapsed(), 1);
    }

    #[test]
    fn zero_warmup_checks_readiness_on_the_first_exhausted_frame() {
        let mut ready = progress(0);
        ready.advance(false, true);
        assert!(!ready.is_frozen());
        ready.advance(true, true);
        assert!(ready.is_frozen());
        assert_eq!(ready.warmup_frames_elapsed(), 0);

        let mut blocked = progress(0);
        blocked.advance(true, false);
        assert_eq!(blocked.failure(), Some(READINESS_BARRIER_FAILURE));
    }
}
