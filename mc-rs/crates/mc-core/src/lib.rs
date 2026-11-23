#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Conventions shared by every `mc-*` crate.
//!
//! Deliberately tiny. Four things live here and nothing else should:
//!
//! * [`profile_scope!`] — the replacement for `net/minecraft/profiler/`.
//! * [`profile`] (feature `subscriber`) — the layer that *aggregates* those scopes, which is
//!   consumed by snapshots, explicit dumps, and optional windowed stderr reports.
//! * [`init_tracing`] (feature `subscriber`) — the single place a subscriber is installed.
//! * [`FastMap`] / [`FastSet`] — the hasher every hot-path map in the workspace should use.
//!
//! Errors are **not** centralised here. Each crate
//! defines its own `thiserror` enum, and `anyhow` appears only in `mc-client`.

/// Re-exported so [`profile_scope!`] expands correctly in crates that do not themselves
/// depend on `tracing`.
pub use tracing;

/// The `BuildHasher` behind [`FastMap`] and [`FastSet`]: foldhash's fast variant, seeded per
/// process *and* per map.
///
/// `std`'s default is SipHash-1-3, which costs roughly 10–20 ns per small key before the probe —
/// the dominant part of a lookup keyed by `ChunkPos`, `BlockPos`, `i32` or a short `String`.
/// foldhash costs about 1–2 ns for those keys.
///
/// **Why seeded rather than a fixed Fx hash.** Most hot keys are chosen by the server: entity
/// ids, chunk and block positions, player UUIDs, team and score-holder names. With a fixed,
/// public hash function a hostile server can precompute thousands of keys that share one bucket
/// chain and turn every lookup linear. A per-process random seed makes that infeasible at the
/// same speed class (foldhash's "minimally DoS-resistant" guarantee); a fixed Fx would save well
/// under a nanosecond per key on top of this.
///
/// **Iteration order.** Like `std::collections::hash_map::RandomState`, every instance draws a
/// fresh seed, so iteration order differs between maps and between runs. No code may depend on
/// it — exactly the contract the std maps already impose. Code that must reproduce Java's
/// `HashMap` order (attribute modifiers, sidebar ties) computes Java's hash itself and does not
/// go through a Rust hasher. Use [`std::collections::BTreeMap`] or an explicit sort where an
/// order is observable.
pub type FastHasher = foldhash::fast::RandomState;

/// A `HashMap` with [`FastHasher`]. Construct with `FastMap::default()`, or
/// `FastMap::with_capacity_and_hasher(n, Default::default())`; [`FastMapExt`] also restores
/// `new()` / `with_capacity(n)` when imported.
pub type FastMap<K, V> = std::collections::HashMap<K, V, FastHasher>;

/// A `HashSet` with [`FastHasher`]. Construct with `FastSet::default()`; see [`FastMap`].
pub type FastSet<T> = std::collections::HashSet<T, FastHasher>;

/// `FastMap::new()` / `FastMap::with_capacity(n)`, which `std` only defines for its own hasher.
pub use foldhash::HashMapExt as FastMapExt;
/// `FastSet::new()` / `FastSet::with_capacity(n)`, which `std` only defines for its own hasher.
pub use foldhash::HashSetExt as FastSetExt;

#[cfg(feature = "subscriber")]
pub mod profile;

/// Atomic, flushed replacement of small files such as settings ([`durable::write_atomic`]).
pub mod durable;

/// Panics the code recovers from, which the crash hook must ignore ([`panic_scope::contain`]).
pub mod panic_scope;

/// Opens a profiling span covering the rest of the enclosing block.
///
/// This replaces vanilla's manual `startSection` / `endSection` stack: the span closes when
/// the block ends, so mismatched pairs are not possible.
///
/// ```
/// fn mesh_section() {
///     mc_core::profile_scope!("terrain.mesh");
///     // ... work is attributed to the span until the function returns
/// }
/// ```
// java: Profiler.java:startSection
#[macro_export]
macro_rules! profile_scope {
    ($name:expr) => {
        let _mc_profile_span = $crate::tracing::span!($crate::tracing::Level::TRACE, $name);
        let _mc_profile_guard = _mc_profile_span.enter();
    };
}

/// The non-blocking stdout writer [`init_tracing`] installs.
#[cfg(feature = "subscriber")]
pub mod logwriter;

/// Sign-in tokens registered as they arrive, and their removal from logs,
/// crash reports and panic messages. Not behind `subscriber`: `mc-auth` registers its tokens.
pub mod secrets;

/// Installs the process-wide `tracing` subscriber.
///
/// Call exactly once, from `mc-client`'s `main`. Verbosity comes from `RUST_LOG`
/// (for example `RUST_LOG=mc_render=debug,mc_protocol=trace`) and defaults to `info`.
///
/// Lines are written by a background thread ([`logwriter`]); call [`flush_logs`] before
/// printing anything that must follow them and before the process exits.
///
/// When [`profile::ENABLE_ENV`] is set, [`profile::ProfileLayer`] is installed **alongside** the
/// formatting layer, each with its own filter: the profiler sees every `mc_*` span regardless of
/// `RUST_LOG`, and `RUST_LOG` keeps deciding what is printed. Without the split, reading the
/// `TRACE` spans would mean printing every `TRACE` log line in the workspace, which is both
/// unreadable and slow enough to change the thing being measured.
#[cfg(feature = "subscriber")]
pub fn init_tracing() {
    use tracing_subscriber::{filter, fmt, prelude::*, EnvFilter};

    let env = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    // `Option<Layer>` is itself a `Layer`, so `None` costs nothing: the layer is not merely
    // filtered out, it is absent, and the global max level stays wherever `RUST_LOG` put it.
    let profiler = profile::enabled().then(|| {
        // Spans only. Events keep `Interest::never()` from this layer, so enabling the profiler
        // does not turn every `trace!` callsite in the workspace into a dispatched event.
        profile::ProfileLayer::new().with_filter(filter::filter_fn(
            |meta: &tracing::Metadata<'_>| meta.is_span() && meta.target().starts_with("mc_"),
        ))
    });
    // Formatted lines go to stdout from a background thread: a console or pipe
    // that stalls must not stall the frame. Same bytes, same order; see `logwriter`.
    let writer = logwriter::NonBlockingStdout::install();
    tracing_subscriber::registry()
        .with(fmt::layer().with_writer(writer).with_filter(env))
        .with(profiler)
        .init();
    if profile::enabled() {
        profile::start();
    }
}

/// Wait (at most two seconds) until every log line emitted so far has reached stdout.
///
/// `mc-client`'s `main` calls this before a crash report is printed (that report goes to
/// stdout too, and must come after the lines that led to it) and before returning, because
/// the writer thread does not outlive `main`. A no-op when [`init_tracing`] was not called.
#[cfg(feature = "subscriber")]
pub fn flush_logs() {
    logwriter::flush(std::time::Duration::from_secs(2));
}

/// Calls [`flush_logs`] when dropped. Bind one (`let _flush_logs = FlushLogsOnDrop;`) right
/// after [`init_tracing`] in a `main` that can return early through `?`, so every exit path
/// drains the background writer.
#[cfg(feature = "subscriber")]
#[derive(Debug, Default)]
pub struct FlushLogsOnDrop;

#[cfg(feature = "subscriber")]
impl Drop for FlushLogsOnDrop {
    fn drop(&mut self) {
        flush_logs();
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn profile_scope_expands_and_nests() {
        crate::profile_scope!("outer");
        {
            crate::profile_scope!("inner");
        }
    }

    /// The aliases are drop-in `HashMap`/`HashSet`s: same API, `Default`, `collect`, `Clone`
    /// with the same contents, and the `Ext` traits restore `new()`.
    #[test]
    fn fast_map_and_set_behave_like_std() {
        use crate::{FastMap, FastMapExt, FastSet, FastSetExt};

        let mut map: FastMap<(i32, i32), u32> = FastMap::default();
        for x in -40..40 {
            for z in -40..40 {
                map.insert((x, z), (x * 1000 + z) as u32);
            }
        }
        assert_eq!(map.len(), 6400);
        assert_eq!(map.get(&(-7, 39)), Some(&((-7 * 1000 + 39) as u32)));
        let cloned = map.clone();
        assert_eq!(cloned, map);
        assert!(map.remove(&(0, 0)).is_some());
        assert!(!map.contains_key(&(0, 0)));

        let set: FastSet<String> = ["a", "b", "a"].iter().map(|s| (*s).to_owned()).collect();
        assert_eq!(set.len(), 2);
        assert!(set.contains("b"));

        let mut empty: FastMap<i32, i32> = FastMap::new();
        empty.insert(1, 2);
        let mut sized: FastSet<i32> = FastSet::with_capacity(16);
        sized.insert(3);
        assert!(sized.capacity() >= 16);
        assert_eq!(empty[&1], 2);
    }

    /// Two maps holding the same keys must not be assumed to iterate alike — the same contract
    /// as `std`'s `RandomState`. Sorting restores a deterministic order.
    #[test]
    fn fast_map_order_is_unspecified_but_sortable() {
        let a: crate::FastSet<i32> = (0..256).collect();
        let b: crate::FastSet<i32> = (0..256).rev().collect();
        let mut a: Vec<i32> = a.into_iter().collect();
        let mut b: Vec<i32> = b.into_iter().collect();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
        assert_eq!(a, (0..256).collect::<Vec<_>>());
    }
}
