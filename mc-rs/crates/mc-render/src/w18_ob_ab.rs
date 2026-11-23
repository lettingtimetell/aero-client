//! A within-run A/B interleave for the switches: a
//! measurement aid.
//!
//! `MC_W18_OB_AB_INTERLEAVE=<name>[,<name>...]` (read once) turns the named candidates off on
//! every other presented frame and leaves them on on the frames between, so one benchmark run
//! compares on and off frames that show nearly the same scene under the same background load.
//! Names: `decouple`, `clear_fold`, `depth_read_pass`, `ui_ring`, `camera_ring`, `direct_draw`,
//! and `control` (turns nothing off: the A/A control of the method).
//!
//! Alternating frames is exact for a candidate that changes only the work inside a frame (the
//! camera and UI rings: compare `render_ms`). A candidate that moves work to the present thread
//! or changes the loop's timing (decouple, the pass merges, direct draw) shifts part of its
//! effect onto the next frame's wait, so measure those with separate runs.
//!
//! The shell flips the parity once per presented frame. The frame drawn while
//! [`presented_frames`] is odd has the named candidates off; in a frame trace that is every
//! frame with an even `frame_index` (the first present only anchors the trace's clock).
//!
//! Unset (always, for players) the mask is empty and every check is one cached load of it.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;

/// The environment variable naming the candidates to interleave.
pub const AB_INTERLEAVE_ENV: &str = "MC_W18_OB_AB_INTERLEAVE";

/// `MC_W18_OB_DECOUPLE`.
pub const DECOUPLE: u32 = 1 << 0;
/// `MC_W18_OB_CLEAR_FOLD`.
pub const CLEAR_FOLD: u32 = 1 << 1;
/// `MC_W18_OB_DEPTH_READ_PASS`.
pub const DEPTH_READ_PASS: u32 = 1 << 2;
/// `MC_W18_OB_UI_RING`.
pub const UI_RING: u32 = 1 << 4;
/// `MC_W18_OB_CAMERA_RING`.
pub const CAMERA_RING: u32 = 1 << 5;
/// `MC_W18_OB_DIRECT_DRAW`.
pub const DIRECT_DRAW: u32 = 1 << 6;
/// Turns nothing off.
pub const CONTROL: u32 = 1 << 7;

const NAMES: [(&str, u32); 7] = [
    ("decouple", DECOUPLE),
    ("clear_fold", CLEAR_FOLD),
    ("depth_read_pass", DEPTH_READ_PASS),
    ("ui_ring", UI_RING),
    ("camera_ring", CAMERA_RING),
    ("direct_draw", DIRECT_DRAW),
    ("control", CONTROL),
];

static OFF_FRAME: AtomicBool = AtomicBool::new(false);
static PRESENTED: AtomicU64 = AtomicU64::new(0);

fn mask() -> u32 {
    static MASK: OnceLock<u32> = OnceLock::new();
    *MASK.get_or_init(|| {
        let Some(value) = std::env::var_os(AB_INTERLEAVE_ENV) else {
            return 0;
        };
        let value = value.to_string_lossy();
        let mut mask = 0;
        for name in value
            .split(',')
            .map(str::trim)
            .filter(|name| !name.is_empty())
        {
            match NAMES.iter().find(|(known, _)| *known == name) {
                Some((_, bit)) => mask |= bit,
                None => tracing::warn!(name, "{AB_INTERLEAVE_ENV}: unknown candidate; ignored"),
            }
        }
        if mask != 0 {
            tracing::warn!(%value, "{AB_INTERLEAVE_ENV}: interleaving (measurement aid)");
        }
        mask
    })
}

/// Whether candidate `bit` is turned off for the frame being drawn.
#[inline]
#[must_use]
pub fn off(bit: u32) -> bool {
    mask() & bit != 0 && OFF_FRAME.load(Ordering::Relaxed)
}

/// Frames presented so far while the interleave is on (zero otherwise).
#[must_use]
pub fn presented_frames() -> u64 {
    PRESENTED.load(Ordering::Relaxed)
}

/// The shell presented a frame: flip the parity for the next one. Nothing while unset.
pub(crate) fn frame_presented() {
    if mask() == 0 {
        return;
    }
    let presented = PRESENTED.fetch_add(1, Ordering::Relaxed) + 1;
    OFF_FRAME.store(presented % 2 == 1, Ordering::Relaxed);
}
