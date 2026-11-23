//! Main-thread CPU candidates for busy lobbies and tick frames.
//!
//! Every candidate is on by default behind its own kill switch, read once per process:
//! `MC_W18_OA_<NAME>=0` restores the previous code path exactly. Each candidate only skips work
//! whose result is provably identical to one already computed, so the frame and the simulation
//! stay identical; the proofs are next to each candidate and in the lane report.
//!
//! The switches are read once per frame or per tick by their callers, except `head_profiles`,
//! which is read once per drawn player head (a cached `bool` and one relaxed atomic load).

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::OnceLock;

/// Frame-domain candidate: the nameplate walk reuses the living walk's culling answers.
pub(crate) const PLATE_GATES: u32 = 1;
/// Frame-domain candidate: a player head reads its owner straight out of the item's NBT instead
/// of building a `GameProfile` copy of it every frame.
pub(crate) const HEAD_PROFILES: u32 = 2;
/// Tick-domain candidate: the tile index's every-tick rescan of the player's 3x3 columns skips a
/// column whose sections are provably the ones it last scanned.
pub(crate) const NEAR_RESCAN: u32 = 1 << 16;

/// Candidates measured on alternate *frames* by the interleave aid.
const FRAME_DOMAIN: u32 = PLATE_GATES | HEAD_PROFILES;
/// Candidates measured on alternate *pairs of ticks* by the interleave aid.
const TICK_DOMAIN: u32 = NEAR_RESCAN;

/// Reads `MC_W18_OA_<name>` once: anything but `0` (or unset) leaves the candidate on.
fn read_switch(name: &str) -> bool {
    std::env::var_os(name).is_none_or(|value| value != "0")
}

/// `MC_W18_OA_PLATE_GATES=0` makes the nameplate walk recompute the
/// section and frustum gates for every entity, as before.
pub(crate) fn plate_gates() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| read_switch("MC_W18_OA_PLATE_GATES")) && !ab_off(PLATE_GATES)
}

/// `MC_W18_OA_HEAD_PROFILES=0` builds the owner `GameProfile` of every
/// drawn player head from its item NBT each frame, as before.
pub(crate) fn head_profiles() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| read_switch("MC_W18_OA_HEAD_PROFILES")) && !ab_off(HEAD_PROFILES)
}

/// `MC_W18_OA_NEAR_RESCAN=0` rescans the player's 3x3 columns every tick
/// whatever happened to them, as before.
pub(crate) fn near_rescan() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| read_switch("MC_W18_OA_NEAR_RESCAN")) && !ab_off(NEAR_RESCAN)
}

// ---- per-frame scratch --------------------------------------------------------------------

/// The per-frame scratch on `Client`: reused, so a frame allocates
/// nothing once it has grown.
#[derive(Debug, Default)]
pub(crate) struct OaScratch {
    /// One [`GateMemo`] byte per entity of the living walk, in walk order.
    pub(crate) gate_memo: Vec<u8>,
}

/// What the living walk learned about one entity's culling gates.
///
/// Both walks of `upload_entities` iterate the same tracker inside one shared borrow of the
/// session, and both ask `entrender::entity_section_reached` and
/// `entrender::should_render_tracked_with_leash` with identical arguments (the same render
/// stack, world, tracked record, frustum, render-view position, partial tick, local vehicle and
/// leash holder). Those are pure reads of state that nothing between the two walks can change,
/// so the second walk may take the first walk's answer instead of recomputing it.
pub(crate) struct GateMemo;

impl GateMemo {
    /// The living walk never reached this entity's gates (fishing hook, first-person view
    /// entity): the nameplate walk computes them itself, as before.
    pub(crate) const UNKNOWN: u8 = 0;
    /// `entity_section_reached` was false: the nameplate walk skips the entity.
    pub(crate) const SECTION_MISSED: u8 = 1;
    /// The section was reached but the leash-aware frustum gate said no: skipped.
    pub(crate) const OUT_OF_VIEW: u8 = 2;
    /// The section was reached and the frustum gate said yes: both calls are skipped (the
    /// render-distance caps still run, as before).
    pub(crate) const IN_VIEW: u8 = 3;
}

// ---- measurement aid: within-run on/off interleave ---------------------------------------
//
// `MC_W18_OA_AB_INTERLEAVE=<names>` (comma-separated candidate names, e.g. `PLATE_GATES`) turns
// the named candidates off on alternate lockstep steps of a `--replay` bench run: frame-domain
// candidates on every odd step, tick-domain candidates on alternate pairs of ticks (a busy
// replay can load odd and even ticks differently, so each arm gets one of each). So
// on-minus-off is measured on adjacent frames or ticks under the same machine load. Unset
// (every normal run), the mask is zero and a switch read
// costs one relaxed atomic load more than its cached `bool`.

static OFF_THIS_FRAME: AtomicU32 = AtomicU32::new(0);

fn interleaved() -> u32 {
    static MASK: OnceLock<u32> = OnceLock::new();
    *MASK.get_or_init(|| {
        let value = std::env::var("MC_W18_OA_AB_INTERLEAVE").unwrap_or_default();
        value.split(',').fold(0, |mask, name| {
            mask | match name.trim() {
                "PLATE_GATES" => PLATE_GATES,
                "HEAD_PROFILES" => HEAD_PROFILES,
                "NEAR_RESCAN" => NEAR_RESCAN,
                _ => 0,
            }
        })
    })
}

#[inline]
fn ab_off(candidate: u32) -> bool {
    OFF_THIS_FRAME.load(Ordering::Relaxed) & candidate != 0
}

/// Called once at the top of `Client::frame`: with the interleave aid set, picks this frame's
/// arms from the replay's lockstep step. A no-op otherwise.
pub(crate) fn ab_frame(session: Option<&crate::session::Session>) {
    let mask = interleaved();
    if mask == 0 {
        return;
    }
    let step = session
        .and_then(crate::session::Session::replay_source)
        .map_or(0, |source| source.virtual_micros() / 1000);
    let mut off = 0;
    if step % 2 == 1 {
        off |= mask & FRAME_DOMAIN;
    }
    // A step's tick runs before its frame, so this frame sets the arm of the next step's tick:
    // tick `k` (the frame at step `50 * k`) runs with these candidates off when `(k / 2)` is odd.
    if ((step + 1) / 100) % 2 == 1 {
        off |= mask & TICK_DOMAIN;
    }
    OFF_THIS_FRAME.store(off, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    /// The text between a call's `(` and its matching `)`, whitespace removed, with the nameplate
    /// walk's spelling of the world (`&session.state().world`) written as the living walk's local.
    fn normalised_arguments(after_open: &str) -> String {
        let mut depth = 1;
        let mut end = after_open.len();
        for (index, ch) in after_open.char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = index;
                        break;
                    }
                }
                _ => {}
            }
        }
        after_open[..end]
            .split_whitespace()
            .collect::<String>()
            .replace("&session.state().world", "world")
    }

    /// The memo is exact only while both walks of `upload_entities` ask
    /// the two gates the same question inside the one session borrow: same arguments, the living
    /// walk first. This keeps a later edit to one walk from silently desynchronising them.
    #[test]
    fn both_entity_walks_ask_the_gates_the_same_question() {
        let source = include_str!("../driver.rs");
        let production = source
            .split_once("#[cfg(test)]\nmod tests {")
            .expect("driver keeps its inline test module")
            .0;
        let upload = production
            .split_once(
                "fn upload_entities(&mut self, gpu: &Gpu, camera: &Camera, partial_ticks: f32)",
            )
            .expect("the entity upload")
            .1;
        let upload = upload
            .split_once("fn drive_atmosphere(")
            .expect("the next function")
            .0;
        // The living walk's `world` is the session's world, which the nameplate walk names.
        assert!(upload.contains("let world = &session.state().world;"));
        for call in [
            "entrender::entity_section_reached(",
            "entrender::should_render_tracked_with_leash(",
        ] {
            let arguments: Vec<String> = upload
                .match_indices(call)
                .map(|(at, _)| normalised_arguments(&upload[at + call.len()..]))
                .collect();
            assert_eq!(arguments.len(), 2, "{call} must have one call per walk");
            assert_eq!(
                arguments[0], arguments[1],
                "{call} arguments differ between walks"
            );
        }
        let written = upload
            .find("gate_memo.push(")
            .expect("the living walk writes");
        let read = upload
            .find(".get(memo_slot)")
            .expect("the nameplate walk reads");
        let plates = upload
            .find("// <<W18-OA:plate-gates>> The same walk as the living pass's")
            .expect("the nameplate walk");
        assert!(written < plates && plates < read);
        // Negative control: the comparison sees a changed argument.
        assert_ne!(
            normalised_arguments("render, world, tracked, false)"),
            normalised_arguments("render, world, tracked, true)")
        );
    }
}
