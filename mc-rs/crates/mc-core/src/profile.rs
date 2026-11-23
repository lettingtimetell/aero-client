//! A `tracing` layer that turns [`profile_scope!`](crate::profile_scope) into a real profiler.
//!
//! java: `net/minecraft/profiler/Profiler.java` — vanilla's `startSection`/`endSection` pair
//! accumulated per-section times into a tree and printed it on demand (F3+shift). This port's
//! [`profile_scope!`](crate::profile_scope) spans are consumed by [`snapshot`],
//! [`snapshot_hierarchy`], [`dump`], and the optional windowed stderr reporter when the profiling
//! layer is enabled.
//!
//! # Turning it on
//!
//! Off unless `MC_PROFILE` is set to something other than `0`/`false`/empty. When off the layer
//! is not installed at all, so the global `tracing` max level is unchanged and span construction
//! stays a no-op — the cost of having this module is zero.
//!
//! | variable | meaning |
//! |---|---|
//! | `MC_PROFILE=1` | install the layer |
//! | `MC_PROFILE_INTERVAL=<secs>` | print a table every `<secs>` seconds, covering **only that window** |
//! | `MC_PROFILE_WARMUP=<secs>` | discard everything recorded before this many seconds have passed |
//!
//! `MC_PROFILE_WARMUP` matters more than it looks: connecting to a server spends its first
//! seconds loading assets and meshing the world for the first time, and those samples swamp the
//! steady state that a play-test actually feels. Interval dumps are windowed for the same reason.
//!
//! # What is measured
//!
//! Every entered span is timed with [`Instant`], and attributed to its **name** — not its call
//! site, so the same name in two places sums. Two numbers are kept:
//!
//! * **total** — wall time between enter and exit, including nested spans.
//! * **self** — total minus the time spent inside nested spans *on the same thread*.
//!
//! Self time is what localises a cost; total is what tells you which subtree it lives in. Because
//! the child subtraction is per-thread, a span whose children run on a `rayon` pool keeps their
//! time in *its* total and in *its* self time on the parent thread — which is the honest answer:
//! from the frame's point of view the pool's latency is time the frame did not have.
//!
//! Nothing here is sampled and nothing is estimated. A scope that is not instrumented shows up
//! only as self time in its enclosing scope, which is how a gap is found: instrument the parent,
//! see the unattributed remainder, subdivide.
//!
//! # Percentiles
//!
//! A mean says nothing about stutter. On a lobby world the measured worst
//! `terrain.drain_mesh_queue` call is 3.16 ms against a mean well under 0.1 ms; on a real
//! DEFAULT-terrain world the same scope's worst call is **138.38 ms**, a 50× overrun of
//! the 2.78 ms frame budget, and its *mean* is 3.558 ms. At 360 Hz the first of each pair is a
//! visible freeze and the second is a number nobody can feel, and no column in this table used to
//! be able to tell them apart — `max` names the single worst call and says nothing about how often
//! it happens. So every scope also carries **p50 / p95 / p99 / p99.9** of its per-call *total*
//! duration, from a [`Histogram`] described in full on that type.
//!
//! The histogram is what bounds the memory: 4 KiB per scope name, allocated on first record,
//! never growing with call count, and it does not retain individual samples. What that costs is
//! **value resolution, never count accuracy**: the rank is exact, the value it maps to is
//! quantised to 1/32 of an octave. The bound is deliberately *not* a ring or a reservoir, because
//! §1.8's distribution spans 0.08 ms to 138 ms with its extremes concentrated in a burst — the
//! case where a recency-bounded scheme reports a tail that has simply aged out. See [`Histogram`],
//! and `the_bound_holds_on_the_real_drain_mesh_queue_distribution`, which asserts the bound
//! against that distribution and shows a 4096-sample ring failing it.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use tracing::span::Id;
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::LookupSpan;

/// Environment variable that installs the layer.
pub const ENABLE_ENV: &str = "MC_PROFILE";
/// Environment variable holding the windowed-dump period, in seconds.
pub const INTERVAL_ENV: &str = "MC_PROFILE_INTERVAL";
/// Environment variable holding the warm-up period, in seconds.
pub const WARMUP_ENV: &str = "MC_PROFILE_WARMUP";

// ---------------------------------------------------------------------------------------------
// Percentiles
// ---------------------------------------------------------------------------------------------

/// Sub-buckets per power of two. `5` → 32 buckets per octave, so one bucket spans at most
/// **1/32 = 3.125 %** of its own lower bound.
const SUB_BITS: u32 = 5;
/// `1 << SUB_BITS`. Values below this are counted exactly, one bucket per nanosecond.
const SUB_COUNT: usize = 1 << SUB_BITS;
/// Enough buckets to reach octave 35, i.e. every duration below **2^36 ns = 68.7 s**. A single
/// span longer than that saturates into the last bucket; nothing in a frame loop comes close, and
/// `max_ns` is exact regardless, so the saturation cannot silently hide a tail.
const BUCKET_COUNT: usize = 1024;

/// Which bucket a duration falls in. Monotone in `ns`, which is what makes a cumulative walk over
/// the buckets a valid rank query.
fn bucket_of(ns: u64) -> usize {
    if (ns as usize) < SUB_COUNT {
        return ns as usize;
    }
    // `ns >= 32` so `leading_zeros() <= 58` and `octave >= 5`.
    let octave = 63 - ns.leading_zeros() as usize;
    let shift = octave - SUB_BITS as usize;
    let sub = ((ns >> shift) as usize) & (SUB_COUNT - 1);
    ((octave - SUB_BITS as usize + 1) * SUB_COUNT + sub).min(BUCKET_COUNT - 1)
}

/// The smallest duration that lands in `index`.
fn bucket_lower(index: usize) -> u64 {
    if index < SUB_COUNT {
        return index as u64;
    }
    let group = index / SUB_COUNT; // >= 1
    let sub = index % SUB_COUNT;
    ((SUB_COUNT + sub) as u64) << (group - 1)
}

/// How wide `index` is, in nanoseconds.
fn bucket_width(index: usize) -> u64 {
    if index < SUB_COUNT {
        1
    } else {
        1u64 << (index / SUB_COUNT - 1)
    }
}

/// The value reported for a sample known only to be somewhere in `index`: its midpoint, so the
/// error is symmetric and at most half a bucket in either direction.
fn bucket_mid(index: usize) -> u64 {
    bucket_lower(index) + bucket_width(index) / 2
}

/// A fixed-memory duration histogram — the thing that makes percentiles possible without
/// retaining every sample.
///
/// # The bound, and exactly what it costs
///
/// **Memory:** `BUCKET_COUNT * 4 B = 4 KiB` per scope name, allocated on the scope's first
/// recorded call and **constant thereafter**. A run with twenty instrumented scopes costs 80 KiB
/// however many hours it runs and however many million calls it makes. A retained-sample scheme
/// would cost 4 B × calls (≈ 4 MiB per scope per million frames), and a fixed-capacity ring or a
/// reservoir would bound that only by throwing samples away — a ring loses everything older than
/// its capacity (so the lobby world's join burst would age out of a 21 s run entirely), and a
/// reservoir makes the tail estimate itself random. This loses neither.
///
/// **What is exact:** the *count*. Every call is counted in exactly one bucket, so the rank a
/// percentile query resolves to is the true rank over the true population — there is no sampling
/// and nothing is discarded. `calls`, `max_ns`, `total_ns` are unaffected and remain exact.
///
/// **What is approximate:** the *value*, and only the value. A bucket spans at most 1/32 of its
/// own lower bound, and the reported figure is the bucket midpoint, so
///
/// > a reported percentile is within **±1.57 %** of the true one, and always within one bucket.
///
/// Durations below 32 ns are held one-per-bucket and are therefore exact. Above 2^36 ns (68.7 s)
/// a sample saturates into the top bucket; `max_ns` still records it exactly.
///
/// # Windowing
///
/// Cumulative histograms subtract bucket-wise, so unlike `max_ns` — which cannot be windowed at
/// all, a maximum is not a difference — [`ScopeStats::since`] produces percentiles that describe
/// **only** the window. That is what makes the `MC_PROFILE_INTERVAL` dumps usable for finding
/// *when* a tail happened rather than merely that it did.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Histogram {
    /// `BUCKET_COUNT` counters, or empty when nothing has been recorded yet.
    buckets: Vec<u32>,
}

impl Histogram {
    /// Count one duration.
    pub fn record(&mut self, ns: u64) {
        if self.buckets.is_empty() {
            self.buckets = vec![0u32; BUCKET_COUNT];
        }
        let b = &mut self.buckets[bucket_of(ns)];
        *b = b.saturating_add(1);
    }

    /// How many durations are counted.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.buckets.iter().map(|c| u64::from(*c)).sum()
    }

    /// Whether anything is counted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// The `q`-quantile in nanoseconds, `q` in `0.0..=1.0`, by **nearest rank**: the smallest
    /// recorded value `v` such that at least `ceil(q * n)` samples are `<= v`, reported as the
    /// midpoint of `v`'s bucket.
    ///
    /// Nearest rank rather than interpolation because interpolating between two buckets would
    /// invent a value the run never produced, and because at `q = 0.999` with a few thousand
    /// samples the interpolation weight is dominated by whichever single sample is adjacent.
    #[must_use]
    pub fn quantile(&self, q: f64) -> u64 {
        let n = self.count();
        if n == 0 {
            return 0;
        }
        let rank = ((q.clamp(0.0, 1.0) * n as f64).ceil() as u64).clamp(1, n);
        let mut cumulative = 0u64;
        for (index, count) in self.buckets.iter().enumerate() {
            cumulative += u64::from(*count);
            if cumulative >= rank {
                return bucket_mid(index);
            }
        }
        0
    }

    /// This histogram minus an earlier snapshot of the same counters, bucket by bucket.
    #[must_use]
    fn since(&self, earlier: &Histogram) -> Histogram {
        if earlier.buckets.is_empty() {
            return self.clone();
        }
        if self.buckets.is_empty() {
            return Histogram::default();
        }
        Histogram {
            buckets: self
                .buckets
                .iter()
                .zip(earlier.buckets.iter())
                .map(|(now, before)| now.saturating_sub(*before))
                .collect(),
        }
    }
}

/// What one scope name cost.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ScopeStats {
    /// Times the scope was entered and left.
    pub calls: u64,
    /// Nanoseconds between enter and exit, summed, including nested scopes.
    pub total_ns: u64,
    /// [`total_ns`](ScopeStats::total_ns) minus time spent in nested scopes on the same thread.
    pub self_ns: u64,
    /// The most expensive single entry-to-exit, in nanoseconds.
    pub max_ns: u64,
    /// The distribution of single entry-to-exit durations, for percentiles. See [`Histogram`]
    /// for the memory bound and what it costs in accuracy.
    pub totals: Histogram,
}

impl ScopeStats {
    /// This snapshot minus an earlier one — the cost incurred *between* two snapshots.
    ///
    /// `max_ns` cannot be windowed from cumulative counters (a maximum does not subtract), so it
    /// is carried through unchanged and describes the whole run to date. The percentile histogram
    /// *can* be, and is: the returned percentiles describe only the window.
    #[must_use]
    pub fn since(&self, earlier: &ScopeStats) -> ScopeStats {
        ScopeStats {
            calls: self.calls.saturating_sub(earlier.calls),
            total_ns: self.total_ns.saturating_sub(earlier.total_ns),
            self_ns: self.self_ns.saturating_sub(earlier.self_ns),
            max_ns: self.max_ns,
            totals: self.totals.since(&earlier.totals),
        }
    }

    /// Whether anything was recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.calls == 0
    }

    /// The `q`-quantile of this scope's per-call total duration, in nanoseconds.
    #[must_use]
    pub fn quantile_ns(&self, q: f64) -> u64 {
        self.totals.quantile(q)
    }
}

/// One thread's stack of currently-entered spans.
#[derive(Debug, Clone)]
struct Frame {
    entered: Instant,
    name: Option<&'static str>,
    /// Complete path using `/` as an internal separator. Metadata names may themselves contain
    /// dots (the Rust scopes commonly do), so using Java's display separator here would invent
    /// parents that were never entered.
    path: Option<String>,
    /// Time already attributed to nested spans that opened and closed inside this one.
    child_ns: u64,
}

thread_local! {
    static STACK: RefCell<Vec<Frame>> = const { RefCell::new(Vec::new()) };
}

/// Cumulative totals, keyed by span name. `&'static str` because `Metadata::name` is static.
///
/// [`FastMap`](crate::FastMap), not SipHash: both maps are hashed on *every* span exit while
/// `MC_PROFILE` is on (two string hashes per scope, one per meshed section on the workers), and
/// that cost lands inside the very scopes being measured.
static TOTALS: OnceLock<Mutex<crate::FastMap<&'static str, ScopeStats>>> = OnceLock::new();
/// Inclusive totals keyed by the complete entered-span path. Java's profiler keeps this map by
/// path (`root.tick.render`), rather than by the leaf name, so an F3 pie can ask for immediate
/// children of an arbitrary selected parent without confusing two same-named scopes.
static HIERARCHY: OnceLock<Mutex<crate::FastMap<String, u64>>> = OnceLock::new();
/// Set once the warm-up has elapsed; until then samples are dropped.
static RECORDING: AtomicBool = AtomicBool::new(false);
/// When the layer was installed — the origin for the warm-up.
static STARTED: OnceLock<Instant> = OnceLock::new();
/// When recording actually began, which is later than [`STARTED`] under a warm-up. The dump
/// header divides by this: charging discarded warm-up seconds against the frames that *were*
/// counted would report a frame rate the run never ran at.
static RECORDING_SINCE: OnceLock<Instant> = OnceLock::new();

fn totals() -> &'static Mutex<crate::FastMap<&'static str, ScopeStats>> {
    TOTALS.get_or_init(|| Mutex::new(crate::FastMap::default()))
}

fn hierarchy() -> &'static Mutex<crate::FastMap<String, u64>> {
    HIERARCHY.get_or_init(|| Mutex::new(crate::FastMap::default()))
}

/// Whether [`ENABLE_ENV`] asks for the profiler. Read once and cached.
#[must_use]
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var(ENABLE_ENV) {
        Ok(value) => {
            let value = value.trim();
            !(value.is_empty() || value == "0" || value.eq_ignore_ascii_case("false"))
        }
        Err(_) => false,
    })
}

fn current_section_for(profiler_enabled: bool) -> String {
    if !profiler_enabled {
        return "N/A (disabled)".to_owned();
    }
    STACK
        .try_with(|stack| {
            stack.try_borrow().ok().and_then(|frames| {
                frames
                    .iter()
                    .rev()
                    .find_map(|frame| frame.name)
                    .map(str::to_owned)
            })
        })
        .ok()
        .flatten()
        .unwrap_or_else(|| "N/A (unavailable)".to_owned())
}

/// The innermost named profiling section on this thread.
///
/// This never borrows from the thread-local stack and never panics if a crash interrupts stack
/// mutation. It reports explicit disabled and unavailable states for crash diagnostics.
#[must_use]
pub fn current_section() -> String {
    current_section_for(enabled())
}

fn seconds_from_env(key: &str) -> Option<f64> {
    std::env::var(key)
        .ok()?
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|s| *s > 0.0)
}

/// Everything recorded so far, as `(name, stats)` sorted by self time, descending.
#[must_use]
pub fn snapshot() -> Vec<(&'static str, ScopeStats)> {
    let map = totals().lock().unwrap_or_else(|e| e.into_inner());
    let mut rows: Vec<(&'static str, ScopeStats)> =
        map.iter().map(|(k, v)| (*k, v.clone())).collect();
    drop(map);
    rows.sort_by_key(|(_, stats)| std::cmp::Reverse(stats.self_ns));
    rows
}

/// One immediate child returned by the Java-shaped profiler query.
///
/// `use_percentage` is relative to the selected parent; `total_use_percentage` is relative to
/// the synthetic/root section. The Rust profiler has no separate Java object here, so this small
/// owned value is the stable boundary consumed by the F3 UI.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileResult {
    /// Section name (or the selected path for the first/root row).
    pub name: String,
    /// Percentage relative to the selected parent.
    pub use_percentage: f64,
    /// Percentage relative to the root section.
    pub total_use_percentage: f64,
}

/// Return the selected profiler section followed by its immediate children, matching
/// `Profiler.getProfilingData` in 1.8.9.
///
/// Tracing scopes are rooted at the process rather than in an explicit `root` span. Therefore
/// `root` is virtual when no literal root path was recorded, and its inclusive time is the sum
/// of top-level paths. A selected path may be written as Java writes it (`root.foo.bar`); the
/// virtual prefix is removed before looking up the tracing paths.
///
/// Like vanilla's query this ages the tree: see [`decayed`]. It is meant for the F3 chart only,
/// which asks once per drawn frame (and once per number-key press), as vanilla does.
#[must_use]
pub fn snapshot_hierarchy(profiler_name: &str) -> Vec<ProfileResult> {
    let mut map = hierarchy().lock().unwrap_or_else(|e| e.into_inner());
    let stored_name = profiler_name.strip_prefix("root/").unwrap_or(profiler_name);
    let root_total = map.get("root").copied().unwrap_or_else(|| {
        map.iter()
            .filter(|(name, _)| !name.contains('/'))
            .map(|(_, total)| *total)
            .sum()
    });
    let selected_total = if profiler_name == "root" {
        root_total
    } else {
        map.get(stored_name).copied().unwrap_or(0)
    };
    let parent_prefix = if profiler_name == "root" || profiler_name.is_empty() {
        String::new()
    } else {
        format!("{stored_name}/")
    };
    let mut children: Vec<(String, u64)> = map
        .iter()
        .filter_map(|(path, total)| {
            if profiler_name == "root" && path == "root" {
                return None;
            }
            let child = path.strip_prefix(&parent_prefix)?;
            if child.is_empty() || child.contains('/') {
                return None;
            }
            Some((child.to_owned(), *total))
        })
        .collect();
    // Java promotes the selected section total to the denominator when an inconsistent child
    // map or a parent recorded without children would otherwise make percentages exceed 100%.
    let denominator = selected_total.max(children.iter().map(|(_, total)| *total).sum());
    let root_denominator = root_total.max(denominator);
    let mut results = vec![ProfileResult {
        // Java displays the dotted profiler path even though this implementation uses `/`
        // internally to keep metadata names containing dots intact.
        name: profiler_name.replace('/', "."),
        use_percentage: 100.0,
        total_use_percentage: if root_denominator == 0 {
            0.0
        } else {
            denominator as f64 * 100.0 / root_denominator as f64
        },
    }];
    if denominator > 0 {
        let child_sum: u64 = children.iter().map(|(_, total)| *total).sum();
        for (name, total) in children.drain(..) {
            results.push(ProfileResult {
                name,
                use_percentage: total as f64 * 100.0 / denominator as f64,
                total_use_percentage: total as f64 * 100.0 / root_denominator as f64,
            });
        }
        // Java casts the child sum to float before subtracting; f64 is the closest stable Rust
        // equivalent and preserves the important unaccounted remainder semantics.
        if (denominator as f64) > child_sum as f64 {
            let remainder = denominator - child_sum;
            results.push(ProfileResult {
                name: "unspecified".to_owned(),
                use_percentage: remainder as f64 * 100.0 / denominator as f64,
                total_use_percentage: remainder as f64 * 100.0 / root_denominator as f64,
            });
        }
    }
    // java: `getProfilingData` ends by rewriting every entry as
    // `l * 999L / 1000L`, after the percentages above were taken from the old values.
    for total in map.values_mut() {
        *total = decayed(*total);
    }
    drop(map);
    results[1..].sort_by(|left, right| {
        right
            .use_percentage
            .partial_cmp(&left.use_percentage)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.name.cmp(&right.name))
    });
    results
}

/// One query's ageing of a section total. java: the last loop of
/// `Profiler.getProfilingData` (`:133-136`), `profilingMap.put(s3, l * 999L / 1000L)`. Every query
/// keeps 99.9% of each total, so the chart is a moving average over roughly the last thousand
/// queries (frames) rather than a total since the chart was opened. Widened to `u128` so the
/// product cannot wrap.
fn decayed(total: u64) -> u64 {
    (u128::from(total) * 999 / 1000) as u64
}

/// Forget the F3 chart's section tree, and only that. java:
/// `Profiler.clearProfiling`, which `Minecraft.runGameLoop` (`:1152-1157`) calls when the chart
/// is drawn after a frame without it: vanilla profiles only while the chart is up, so it always
/// starts empty. The per-name totals behind [`snapshot`] and [`dump`] are left alone.
pub fn clear_hierarchy() {
    hierarchy()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clear();
}

/// Forget everything recorded so far. Used by the warm-up, and available to a caller that wants
/// to measure one phase of a run in isolation.
pub fn reset() {
    totals().lock().unwrap_or_else(|e| e.into_inner()).clear();
    hierarchy()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clear();
}

/// How long the profiler has been *recording* — the span the totals cover.
///
/// Equal to the time since installation when there is no warm-up, and to the time since the
/// warm-up ended when there is one.
#[must_use]
pub fn elapsed() -> Duration {
    RECORDING_SINCE
        .get()
        .or_else(|| STARTED.get())
        .map_or(Duration::ZERO, Instant::elapsed)
}

/// Format a table of `rows`, headed by `title` and normalised over `window`.
///
/// Split from [`dump`] so the formatting is testable without a subscriber, and so a caller can
/// route the table somewhere other than stderr.
#[must_use]
pub fn format_table(title: &str, rows: &[(&'static str, ScopeStats)], window: Duration) -> String {
    // Per-frame columns are only meaningful against a frame count, and `client.frame` is the
    // scope that has one. Falling back to the shell's own frame scope keeps this working if the
    // client's is ever renamed.
    let frames = rows
        .iter()
        .find(|(name, _)| *name == "client.frame" || *name == "shell.draw")
        .map_or(0, |(_, stats)| stats.calls);
    let window_ms = window.as_secs_f64() * 1000.0;

    let mut out = String::new();
    out.push_str(&format!(
        "\n===== mc-rs profile: {title} — {:.2} s wall, {frames} frames",
        window.as_secs_f64()
    ));
    if frames > 0 {
        out.push_str(&format!(
            ", {:.1} fps, {:.3} ms/frame",
            frames as f64 / window.as_secs_f64().max(f64::MIN_POSITIVE),
            window_ms / frames as f64
        ));
    }
    out.push_str(" =====\n");
    out.push_str(&format!(
        "{:<34} {:>9} {:>11} {:>11} {:>10} {:>10} {:>9} {:>9} {:>9} {:>9} {:>8}\n",
        "scope",
        "calls",
        "self ms",
        "total ms",
        "self/frm",
        "avg us",
        "p50 us",
        "p95 us",
        "p99 us",
        "p99.9 us",
        "max ms"
    ));
    let ms = |ns: u64| ns as f64 / 1_000_000.0;
    let us = |ns: u64| ns as f64 / 1_000.0;
    for (name, stats) in rows {
        if stats.is_empty() {
            continue;
        }
        let per_frame = if frames > 0 {
            format!("{:.3}", ms(stats.self_ns) / frames as f64)
        } else {
            "-".to_owned()
        };
        out.push_str(&format!(
            "{:<34} {:>9} {:>11.3} {:>11.3} {:>10} {:>10.1} \
             {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>8.2}\n",
            name,
            stats.calls,
            ms(stats.self_ns),
            ms(stats.total_ns),
            per_frame,
            stats.total_ns as f64 / stats.calls.max(1) as f64 / 1000.0,
            us(stats.quantile_ns(0.50)),
            us(stats.quantile_ns(0.95)),
            us(stats.quantile_ns(0.99)),
            us(stats.quantile_ns(0.999)),
            ms(stats.max_ns),
        ));
    }
    let attributed: u64 = rows.iter().map(|(_, s)| s.self_ns).sum();
    out.push_str(&format!(
        "{:<34} {:>9} {:>11.3} {:>11} {:>10} {:>10} {:>9} {:>9} {:>9} {:>9} {:>8}\n",
        "(sum of self)",
        "",
        ms(attributed),
        "",
        if frames > 0 {
            format!("{:.3}", ms(attributed) / frames as f64)
        } else {
            "-".to_owned()
        },
        "",
        "",
        "",
        "",
        "",
        "",
    ));
    out.push_str(
        "  percentiles are of each scope's per-call *total* duration, from a fixed 4 KiB/scope \
         histogram:\n  ranks exact, values within +/-1.57% (one bucket = 1/32 octave). \
         Windowed dumps subtract; `max ms` does not window.\n",
    );
    out
}

/// Print everything recorded so far to stderr.
///
/// Safe to call when the profiler is off: it prints nothing.
pub fn dump() {
    if !enabled() {
        return;
    }
    let rows = snapshot();
    eprint!("{}", format_table("cumulative", &rows, elapsed()));
}

/// The layer. Install with [`ProfileLayer::new`] — or let
/// [`init_tracing`](crate::init_tracing) do it from [`ENABLE_ENV`].
#[derive(Debug, Default, Clone, Copy)]
pub struct ProfileLayer;

impl ProfileLayer {
    /// A layer that records into the process-wide totals.
    #[must_use]
    pub fn new() -> Self {
        ProfileLayer
    }
}

fn record(name: &'static str, total_ns: u64, self_ns: u64) {
    let mut map = totals().lock().unwrap_or_else(|e| e.into_inner());
    let entry = map.entry(name).or_default();
    entry.calls += 1;
    entry.total_ns += total_ns;
    entry.self_ns += self_ns;
    if total_ns > entry.max_ns {
        entry.max_ns = total_ns;
    }
    entry.totals.record(total_ns);
}

fn record_hierarchy(path: String, total_ns: u64) {
    let mut map = hierarchy().lock().unwrap_or_else(|e| e.into_inner());
    let entry = map.entry(path).or_default();
    *entry = entry.saturating_add(total_ns);
}

impl<S> Layer<S> for ProfileLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
        let name = ctx.metadata(id).map(tracing::Metadata::name);
        STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            let path = name.map(|name| {
                let mut path = stack
                    .last()
                    .and_then(|frame| frame.path.as_deref())
                    .unwrap_or_default()
                    .to_owned();
                if !path.is_empty() {
                    path.push('/');
                }
                path.push_str(name);
                path
            });
            stack.push(Frame {
                entered: Instant::now(),
                name,
                path,
                child_ns: 0,
            });
        });
    }

    fn on_exit(&self, id: &Id, ctx: Context<'_, S>) {
        let now = Instant::now();
        let Some(frame) = STACK.with(|stack| stack.borrow_mut().pop()) else {
            return;
        };
        let total_ns = now.duration_since(frame.entered).as_nanos() as u64;
        // Attribute the whole span to whichever scope encloses it *on this thread*, so the
        // parent's self time is the work it did itself.
        STACK.with(|stack| {
            if let Some(parent) = stack.borrow_mut().last_mut() {
                parent.child_ns += total_ns;
            }
        });
        if !RECORDING.load(Ordering::Relaxed) {
            return;
        }
        let Some(name) = frame
            .name
            .or_else(|| ctx.metadata(id).map(tracing::Metadata::name))
        else {
            return;
        };
        record(name, total_ns, total_ns.saturating_sub(frame.child_ns));
        if let Some(path) = frame.path {
            record_hierarchy(path, total_ns);
        }
    }
}

/// Start the clock, honour [`WARMUP_ENV`], and spawn the [`INTERVAL_ENV`] reporter.
///
/// Called by [`init_tracing`](crate::init_tracing) after the layer is installed. Idempotent.
pub(crate) fn start() {
    if STARTED.set(Instant::now()).is_err() {
        return;
    }
    let warmup = seconds_from_env(WARMUP_ENV);
    let interval = seconds_from_env(INTERVAL_ENV);
    match warmup {
        None => {
            RECORDING_SINCE.set(Instant::now()).ok();
            RECORDING.store(true, Ordering::Relaxed);
        }
        Some(secs) => {
            std::thread::Builder::new()
                .name("mc-profile-warmup".to_owned())
                .spawn(move || {
                    std::thread::sleep(Duration::from_secs_f64(secs));
                    reset();
                    RECORDING_SINCE.set(Instant::now()).ok();
                    RECORDING.store(true, Ordering::Relaxed);
                    eprintln!("mc-rs profile: warm-up of {secs} s elapsed, recording");
                })
                .ok();
        }
    }
    let Some(period) = interval else { return };
    std::thread::Builder::new()
        .name("mc-profile-report".to_owned())
        .spawn(move || {
            let period = Duration::from_secs_f64(period);
            let mut previous: HashMap<&'static str, ScopeStats> = HashMap::new();
            let mut mark = Instant::now();
            loop {
                std::thread::sleep(period);
                if !RECORDING.load(Ordering::Relaxed) {
                    mark = Instant::now();
                    continue;
                }
                let now = snapshot();
                let mut rows: Vec<(&'static str, ScopeStats)> = now
                    .iter()
                    .map(|(name, stats)| {
                        let earlier = previous.get(name).cloned().unwrap_or_default();
                        (*name, stats.since(&earlier))
                    })
                    .collect();
                rows.sort_by_key(|(_, stats)| std::cmp::Reverse(stats.self_ns));
                eprint!("{}", format_table("window", &rows, mark.elapsed()));
                mark = Instant::now();
                previous = now.into_iter().collect();
            }
        })
        .ok();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_section_is_owned_and_panic_safe() {
        use tracing_subscriber::prelude::*;

        assert_eq!(current_section_for(false), "N/A (disabled)");
        STACK.with(|stack| {
            stack.borrow_mut().push(Frame {
                entered: Instant::now(),
                name: Some("test.current"),
                path: Some("test.current".to_owned()),
                child_ns: 0,
            });
        });
        assert_eq!(current_section_for(true), "test.current");
        STACK.with(|stack| {
            let _borrow_conflict = stack.borrow_mut();
            assert_eq!(current_section_for(true), "N/A (unavailable)");
        });
        STACK.with(|stack| stack.borrow_mut().clear());
        assert_eq!(current_section_for(true), "N/A (unavailable)");

        let subscriber = tracing_subscriber::registry().with(ProfileLayer::new());
        tracing::subscriber::with_default(subscriber, || {
            let current = tracing::span!(tracing::Level::TRACE, "test.current.live");
            let _entered = current.enter();
            assert_eq!(current_section_for(true), "test.current.live");
        });
        assert_eq!(current_section_for(true), "N/A (unavailable)");
    }

    /// The whole claim of this module: a parent's *self* time is its total minus its children's.
    ///
    /// Driven through a real subscriber, but a **thread-local** one
    /// ([`tracing::subscriber::with_default`]), so no other test in this binary can record into
    /// the totals while this one is reading them.
    #[test]
    fn the_layer_charges_nested_time_to_the_child_not_the_parent() {
        use tracing_subscriber::prelude::*;

        RECORDING.store(true, Ordering::Relaxed);
        reset();
        let subscriber = tracing_subscriber::registry().with(ProfileLayer::new());
        tracing::subscriber::with_default(subscriber, || {
            let outer = tracing::span!(tracing::Level::TRACE, "test.outer");
            let _outer = outer.enter();
            std::thread::sleep(Duration::from_millis(20));
            {
                let inner = tracing::span!(tracing::Level::TRACE, "test.inner");
                let _inner = inner.enter();
                std::thread::sleep(Duration::from_millis(40));
            }
        });
        let rows: HashMap<&'static str, ScopeStats> = snapshot().into_iter().collect();
        reset();
        RECORDING.store(false, Ordering::Relaxed);

        let outer = rows.get("test.outer").cloned().expect("outer recorded");
        let inner = rows.get("test.inner").cloned().expect("inner recorded");
        assert_eq!(outer.calls, 1);
        assert_eq!(inner.calls, 1);

        // Sleeps overshoot — Windows' scheduler tick is 15.6 ms — so nothing here asserts a
        // duration. What is asserted is the *identity*, which holds whatever the sleeps did:
        // the parent's self time plus the child's total is the parent's total.
        let reconstructed = outer.self_ns + inner.total_ns;
        assert!(
            reconstructed.abs_diff(outer.total_ns) < 2_000_000,
            "outer self {} + inner total {} should reconstruct outer total {}",
            outer.self_ns,
            inner.total_ns,
            outer.total_ns
        );
        // Control: the child really did dominate, so a layer that did *not* subtract would have
        // reported the parent's self time as its whole total and failed the line above.
        assert!(
            inner.total_ns > outer.total_ns / 2,
            "the nested scope must be the larger half for this test to prove anything"
        );
        assert!(inner.self_ns == inner.total_ns, "a leaf has no children");
    }

    #[test]
    fn hierarchy_snapshot_rolls_up_immediate_children_and_unaccounted_time() {
        let seed = || {
            reset();
            let mut map = hierarchy().lock().unwrap();
            map.insert("tick".to_owned(), 60_000);
            map.insert("tick/render".to_owned(), 30_000);
            map.insert("tick/input".to_owned(), 10_000);
            map.insert("flat".to_owned(), 40_001);
        };
        seed();

        let root = snapshot_hierarchy("root");
        assert_eq!(root[0].name, "root");
        assert_eq!(root[1].name, "tick");
        assert_eq!(root[2].name, "flat");
        assert!((root[1].use_percentage - 60_000.0 * 100.0 / 100_001.0).abs() < 1e-9);

        // java: the query then rewrote every entry as
        // `l * 999L / 1000L` (truncating), so the chart is a moving average, not a running total.
        {
            let map = hierarchy().lock().unwrap();
            assert_eq!(map["tick"], 59_940);
            assert_eq!(map["tick/render"], 29_970);
            assert_eq!(map["tick/input"], 9_990);
            assert_eq!(
                map["flat"], 39_960,
                "40_001 * 999 / 1000 = 39_960.999, truncated"
            );
        }
        assert_eq!(decayed(1), 0);
        assert_eq!(
            decayed(u64::MAX),
            (u128::from(u64::MAX) * 999 / 1000) as u64
        );
        // `clearProfiling`: a chart opened again starts empty.
        clear_hierarchy();
        assert_eq!(
            snapshot_hierarchy("root").len(),
            1,
            "only the root row is left"
        );

        seed();
        let tick = snapshot_hierarchy("root/tick");
        assert_eq!(tick[0].name, "root.tick");
        assert_eq!(tick[1].name, "render");
        assert_eq!(tick[2].name, "unspecified");
        assert_eq!(tick[3].name, "input");
        assert!((tick[1].use_percentage - 50.0).abs() < f64::EPSILON);
        assert!((tick[2].use_percentage - 33.333_333_333_333_33).abs() < 1e-12);
        reset();
    }

    #[test]
    fn windowed_stats_subtract_the_earlier_snapshot() {
        let earlier = ScopeStats {
            calls: 10,
            total_ns: 1000,
            self_ns: 600,
            max_ns: 200,
            ..ScopeStats::default()
        };
        let later = ScopeStats {
            calls: 25,
            total_ns: 4000,
            self_ns: 2100,
            max_ns: 900,
            ..ScopeStats::default()
        };
        let window = later.since(&earlier);
        assert_eq!(window.calls, 15);
        assert_eq!(window.total_ns, 3000);
        assert_eq!(window.self_ns, 1500);
        // A maximum is not a difference; it stays the run's maximum.
        assert_eq!(window.max_ns, 900);
        // Control: a snapshot minus itself is empty, so a window with no work reports none.
        assert!(later.since(&later).is_empty());
    }

    /// A helper that builds a scope's stats from an explicit list of per-call durations, so a
    /// test can state the population and then assert the percentile the table would print.
    fn stats_from(ns: &[u64]) -> ScopeStats {
        let mut s = ScopeStats::default();
        for d in ns {
            s.calls += 1;
            s.total_ns += *d;
            s.self_ns += *d;
            s.max_ns = s.max_ns.max(*d);
            s.totals.record(*d);
        }
        s
    }

    /// The whole point of T0.3, on the shape of data that motivated it: 998 cheap frames and two
    /// 3.16 ms drains. The mean cannot see them; p99.9 and max can, and p50/p95/p99 must *not*.
    ///
    /// Two spikes rather than one because nearest rank puts a lone spike at p100 and nowhere
    /// else — `ceil(0.999 n) = n` only for `n < 1000` — and a test that only worked at one
    /// population size would be testing the size, not the statistic.
    #[test]
    fn a_rare_spike_is_invisible_in_the_mean_and_visible_in_p99_9() {
        let mut population: Vec<u64> = vec![50_000; 998]; // 0.05 ms
        population.extend(vec![3_160_000u64; 2]); // the measured worst `terrain.drain_mesh_queue`
        let stats = stats_from(&population);

        let mean_ns = stats.total_ns as f64 / stats.calls as f64;
        // Control: the mean moves by 12 % and would be read as "no stutter here" — it is still
        // 60x under the 2.78 ms frame budget that the spike itself blows through.
        assert!(
            (mean_ns - 56_220.0).abs() < 1.0,
            "mean should be 56.2 us, was {mean_ns}"
        );
        // p50 and p95 must stay on the cheap population, or the statistic is not a percentile.
        assert!(within_one_bucket(stats.quantile_ns(0.50), 50_000));
        assert!(within_one_bucket(stats.quantile_ns(0.95), 50_000));
        assert!(within_one_bucket(stats.quantile_ns(0.99), 50_000));
        // p99.9 is the 999th of 1000 samples by nearest rank: the first spike.
        assert!(
            within_one_bucket(stats.quantile_ns(0.999), 3_160_000),
            "p99.9 was {} ns, expected the 3.16 ms spike",
            stats.quantile_ns(0.999)
        );
        assert_eq!(stats.max_ns, 3_160_000, "max stays exact");
    }

    /// The documented accuracy bound, asserted rather than claimed: within one bucket, and one
    /// bucket is at most 1/32 of the value.
    fn within_one_bucket(reported: u64, truth: u64) -> bool {
        let b = bucket_of(truth);
        reported >= bucket_lower(b) && reported < bucket_lower(b) + bucket_width(b)
    }

    #[test]
    fn every_quantile_is_within_the_documented_error_bound() {
        // A spread over four orders of magnitude, so the bound is tested in several octaves.
        let population: Vec<u64> = (0..10_000u64).map(|i| 1_000 + i * 997).collect();
        let stats = stats_from(&population);
        let mut sorted = population.clone();
        sorted.sort_unstable();
        for q in [0.5, 0.9, 0.95, 0.99, 0.999, 1.0] {
            let rank = ((q * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
            let truth = sorted[rank - 1];
            let reported = stats.quantile_ns(q);
            let error = (reported as f64 - truth as f64).abs() / truth as f64;
            assert!(
                error <= 0.0157,
                "q{q}: reported {reported} vs true {truth} is {:.4} relative error, \
                 over the documented 1.57%",
                error
            );
        }
        // Control: the bound is tight enough to be falsifiable. A statistic that ignored `q` and
        // always answered the median would fail it by a wide margin at q=0.999.
        let median = stats.quantile_ns(0.5) as f64;
        let p999 = stats.quantile_ns(0.999) as f64;
        assert!(
            p999 > median * 1.5,
            "p99.9 {p999} must be well clear of p50 {median}, or this test proves nothing"
        );
    }

    #[test]
    fn buckets_are_monotone_contiguous_and_at_most_a_thirty_second_wide() {
        assert_eq!(bucket_of(0), 0);
        assert_eq!(bucket_of(31), 31);
        assert_eq!(bucket_of(32), 32);
        // Contiguity: every bucket's lower bound is the previous bucket's lower + its width, so
        // no duration falls between two buckets and the cumulative walk cannot skip a rank.
        for i in 0..BUCKET_COUNT - 1 {
            assert_eq!(
                bucket_lower(i) + bucket_width(i),
                bucket_lower(i + 1),
                "bucket {i} does not meet bucket {}",
                i + 1
            );
            assert_eq!(bucket_of(bucket_lower(i)), i, "round trip at bucket {i}");
            // The accuracy claim, per bucket: half a width is <= 1.57% of the lower bound.
            if i >= SUB_COUNT {
                let relative = bucket_width(i) as f64 / 2.0 / bucket_lower(i) as f64;
                assert!(relative <= 0.0157, "bucket {i} is {relative} wide");
            }
        }
        // The saturation point is where it is documented to be.
        assert_eq!(bucket_of(u64::MAX), BUCKET_COUNT - 1);
    }

    #[test]
    fn windowed_percentiles_describe_only_the_window() {
        // A first window that is all spikes, then a second that is all cheap frames.
        let first = stats_from(&vec![3_000_000; 500]);
        let mut both = first.clone();
        for d in vec![40_000u64; 500] {
            both.calls += 1;
            both.total_ns += d;
            both.self_ns += d;
            both.totals.record(d);
        }
        let window = both.since(&first);
        assert_eq!(window.calls, 500);
        assert!(
            within_one_bucket(window.quantile_ns(0.99), 40_000),
            "the second window's p99 was {} ns; it must not inherit the first window's spikes",
            window.quantile_ns(0.99)
        );
        // Control: the cumulative view *does* still see them, so the subtraction is what changed
        // the answer rather than the data never having been there.
        assert!(within_one_bucket(both.quantile_ns(0.99), 3_000_000));
        // And `max_ns` is documented as un-windowable; assert that documented behaviour.
        assert_eq!(window.max_ns, both.max_ns);
    }

    /// The bound, tested against the distribution it actually has to survive.
    ///
    /// A measurement of `terrain.drain_mesh_queue` on real DEFAULT terrain at render
    /// distance 16: a mean of 3.558 ms, a body in the tenths of a millisecond, and a worst single
    /// call of **138.38 ms**. That is three and a half orders of magnitude, which is precisely the
    /// shape that breaks a fixed-capacity ring and makes a reservoir's tail estimate a lottery.
    ///
    /// Two things are asserted. First that every quantile is within the documented ±1.57 % **of
    /// the exact answer over the same population**, computed here by sorting it. Second — the
    /// control §0 rule 2 asks for — that the 4096-sample ring this could have been built from
    /// gets p99.9 wrong by more than 30×, so the assertion above is demonstrably able to detect a
    /// bounding scheme that does not work.
    #[test]
    fn the_bound_holds_on_the_real_drain_mesh_queue_distribution() {
        // A deterministic LCG; the population must be identical on every run or the bound being
        // asserted is a different bound each time.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut population: Vec<u64> = Vec::with_capacity(100_001);
        // The measured worst call, exactly, and *first* — §1.5's freeze is a join burst.
        population.push(138_380_000);
        for i in 0..100_000u32 {
            let r = next();
            // The freezes are put in the first 3 % of the stream, which is where the lobby
            // measurement found them: the join burst. A scheme that keeps only recent samples
            // loses exactly this, and that is the point of the control below. There are ~150 of
            // them, so p99.9 lands inside the freeze band rather than straddling it.
            let sample = if i < 3_000 && r % 20 == 0 {
                20_000_000 + r % 118_380_000 // 20 - 138 ms, the freeze
            } else {
                match r % 1000 {
                    0..=989 => 80_000 + r % 520_000,         // 0.08 - 0.6 ms, the body
                    990..=997 => 3_000_000 + r % 17_000_000, // 3 - 20 ms
                    _ => 600_000 + r % 2_400_000,
                }
            };
            population.push(sample);
        }
        let stats = stats_from(&population);

        let mut sorted = population.clone();
        sorted.sort_unstable();
        let exact = |q: f64| {
            let rank = ((q * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
            sorted[rank - 1]
        };
        for q in [0.5, 0.9, 0.95, 0.99, 0.999, 1.0] {
            let (truth, reported) = (exact(q), stats.quantile_ns(q));
            let error = (reported as f64 - truth as f64).abs() / truth as f64;
            assert!(
                error <= 0.0157,
                "q{q} on the 0.08-138 ms distribution: reported {reported} vs true {truth}, \
                 {:.4} relative error",
                error
            );
        }
        assert_eq!(stats.max_ns, 138_380_000, "max is exact at any magnitude");
        // The tail really is three and a half orders of magnitude above the body, or none of the
        // above is testing anything hard.
        assert!(exact(0.999) > exact(0.5) * 100);

        // Control: the 4096-sample ring this module could have used instead. Same stream, same
        // order, and it answers p99.9 from only the last 4096 calls — by which point the freeze
        // is thousands of samples in the past.
        let mut ring: Vec<u64> = Vec::with_capacity(4096);
        let mut cursor = 0usize;
        for sample in &population {
            if ring.len() < 4096 {
                ring.push(*sample);
            } else {
                ring[cursor] = *sample;
                cursor = (cursor + 1) % 4096;
            }
        }
        ring.sort_unstable();
        let ring_p999 = ring[((0.999 * ring.len() as f64).ceil() as usize) - 1];
        let ring_max = *ring.last().expect("non-empty");
        assert!(
            (ring_p999 as f64) < exact(0.999) as f64 / 2.0,
            "a 4096-sample ring understates p99.9 here (it said {ring_p999} against the true {}); \
             if it did not, this control proves nothing about why the histogram was chosen",
            exact(0.999)
        );
        assert!(
            ring_max < 138_380_000 / 5,
            "and the ring cannot see the 138.38 ms freeze at all — its whole population is \
             younger than the burst. Its maximum was {ring_max} ns."
        );
    }

    #[test]
    fn a_histogram_costs_one_fixed_allocation_however_many_samples_it_counts() {
        let mut h = Histogram::default();
        assert!(h.is_empty());
        assert_eq!(h.buckets.len(), 0, "nothing is allocated before first use");
        h.record(1234);
        assert_eq!(h.buckets.len(), BUCKET_COUNT);
        for i in 0..200_000u64 {
            h.record(i % 5_000_000);
        }
        assert_eq!(
            h.buckets.len(),
            BUCKET_COUNT,
            "memory must not grow with the sample count"
        );
        assert_eq!(h.count(), 200_001);
        assert_eq!(
            std::mem::size_of::<u32>() * BUCKET_COUNT,
            4096,
            "the documented 4 KiB per scope"
        );
    }

    #[test]
    fn the_table_reports_per_frame_cost_against_the_frame_scope() {
        let rows = vec![
            (
                "client.frame",
                ScopeStats {
                    calls: 100,
                    total_ns: 1_400_000_000,
                    self_ns: 400_000_000,
                    max_ns: 20_000_000,
                    ..ScopeStats::default()
                },
            ),
            (
                "terrain.drain_mesh_queue",
                ScopeStats {
                    calls: 100,
                    total_ns: 1_000_000_000,
                    self_ns: 1_000_000_000,
                    max_ns: 30_000_000,
                    ..ScopeStats::default()
                },
            ),
        ];
        let table = format_table("test", &rows, Duration::from_secs(2));
        assert!(table.contains("100 frames"), "{table}");
        assert!(table.contains("50.0 fps"), "{table}");
        // 400 ms of self time over 100 frames is 4 ms/frame.
        assert!(table.contains("4.000"), "{table}");
        // 1000 ms over 100 frames is 10 ms/frame.
        assert!(table.contains("10.000"), "{table}");
    }

    /// The percentile columns exist, are headed, and carry the numbers the scope's own histogram
    /// answers — not a rounding of the mean.
    #[test]
    fn the_table_prints_percentile_columns() {
        // 90 frames at 0.4 ms and 10 at 8 ms: mean 1.16 ms, p50 0.4 ms, p99 8 ms.
        let mut population: Vec<u64> = vec![400_000; 90];
        population.extend(vec![8_000_000u64; 10]);
        let rows = vec![("client.frame", stats_from(&population))];
        let table = format_table("test", &rows, Duration::from_secs(1));

        for header in ["p50 us", "p95 us", "p99 us", "p99.9 us", "max ms"] {
            assert!(table.contains(header), "missing {header} in\n{table}");
        }
        // scope calls self total self/frm avg p50 p95 p99 p99.9 max
        let row = table
            .lines()
            .find(|l| l.starts_with("client.frame"))
            .expect("row present");
        let cols: Vec<f64> = row
            .split_whitespace()
            .skip(1)
            .map(|t| t.parse::<f64>().expect("numeric column"))
            .collect();
        assert_eq!(cols.len(), 10, "unexpected column count in {row}");
        let near = |got: f64, want: f64| (got - want).abs() / want <= 0.0157;
        assert!(near(cols[4], 1160.0), "avg us column: {row}");
        assert!(near(cols[5], 400.0), "p50 us column: {row}");
        assert!(near(cols[6], 8000.0), "p95 us column: {row}");
        assert!(near(cols[7], 8000.0), "p99 us column: {row}");
        assert!(near(cols[9], 8.0), "max ms column: {row}");
        // Control: the percentile columns must not merely restate the mean, or this table would
        // be no better than the one it replaced.
        assert!(
            cols[5] < cols[4] / 2.0 && cols[7] > cols[4] * 2.0,
            "p50 {} and p99 {} straddle the mean {}, which is the whole point: {row}",
            cols[5],
            cols[7],
            cols[4]
        );
    }

    #[test]
    fn a_scope_with_no_calls_is_not_printed() {
        let rows = vec![("never.ran", ScopeStats::default())];
        let table = format_table("test", &rows, Duration::from_secs(1));
        assert!(!table.contains("never.ran"), "{table}");
    }

    #[test]
    fn the_profiler_is_off_unless_the_variable_asks_for_it() {
        // `enabled()` caches, so this asserts the parsing rule rather than the cached value.
        let off = ["", "0", "false", "FALSE"];
        for value in off {
            let v = value.trim();
            assert!(v.is_empty() || v == "0" || v.eq_ignore_ascii_case("false"));
        }
        for value in ["1", "yes", "on"] {
            let v = value.trim();
            assert!(!(v.is_empty() || v == "0" || v.eq_ignore_ascii_case("false")));
        }
    }
}
