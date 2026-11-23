//! Background meshing on a `rayon` pool, driven by `mc_world`'s dirty set.
//!
//! Vanilla runs `ChunkRenderDispatcher` with a small pool of `ChunkRenderWorker` threads and a
//! bounded queue, and rebuilds the section the player is standing in *synchronously* so that
//! placing a block never shows a one-frame hole. Both behaviours are reproduced here; the
//! difference is that [`mesh_section`](super::mesh_section) is a pure function, so there is no
//! lock around the world and no `RegionRenderCache` handoff — `rayon` just maps over the dirty
//! set with a shared `&World`.
//
// java: client/renderer/chunk/ChunkRenderDispatcher.java, ChunkRenderWorker.java,
//       client/renderer/RenderGlobal.java:915-935 (the "build near" synchronous path)

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use mc_data::RenderLayer;
use mc_math::{ChunkSectionPos, DVec3};
use mc_world::World;
use rayon::prelude::*;

use super::cull_face::MeshSettings;
#[cfg(test)]
use super::mesher::mesh_section_with;
use super::mesher::{mesh_section_with_alternatives, section_is_absent, ChunkMesh};
use super::vertex::VERTEX_SIZE;

/// Compatibility limit for callers that still request a count-based drain.
///
/// The live client no longer uses a fixed count: it calls [`MeshQueue::drain_for_frame`], which
/// follows `EntityRenderer`'s nanosecond deadline. This remains public for benches and for small
/// deterministic callers that explicitly want to ask for exactly `N` sections.
///
/// java: `ChunkRenderDispatcher.queueChunkUpdates`' bounded queue is **not** a per-frame section
/// count; do not use this constant as a rendering policy.
pub const DEFAULT_BUDGET: usize = 32;

/// The number of compile tasks accepted by vanilla's pending dispatcher queue.
///
/// java: `ChunkRenderDispatcher.queueChunkUpdates` is an
/// `ArrayBlockingQueue<>(100)`. The queue is deliberately bounded independently from the
/// frame budget: a frame may build fewer than 100 sections, but a caller must never be able to
/// turn a render-distance invalidation into an unbounded allocation.
pub const MAX_PENDING: usize = 100;

/// The number of vanilla terrain worker threads, and the fewest this client runs.
///
/// java: `ChunkRenderDispatcher` starts two daemon `ChunkRenderWorker`s. This pool stays separate
/// from Rayon's process-wide pool so its size is decided here; see [`default_pool_size`].
pub const TERRAIN_WORKER_COUNT: usize = 2;

/// How many threads mesh sections. `MC_RS_TERRAIN_WORKERS` overrides [`default_pool_size`], to
/// measure other sizes.
fn terrain_pool_size() -> usize {
    std::env::var("MC_RS_TERRAIN_WORKERS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&count| count > 0)
        .unwrap_or_else(|| {
            default_pool_size(std::thread::available_parallelism().map_or(1, usize::from))
        })
}

/// A quarter of the logical CPUs, between vanilla's two workers and four.
///
/// Measured on an 8-core, 16-thread CPU loading a 17,424-section world: two workers took 12.03 s,
/// four took 6.44 s, and six took 5.82 s with noticeably worse slowest frames while loading. Uploads
/// run in dispatch order whichever worker finishes first, so this changes how soon terrain appears,
/// not what it looks like. A smaller machine keeps vanilla's two, leaving room for the main thread.
fn default_pool_size(logical_cpus: usize) -> usize {
    (logical_cpus / 4).clamp(TERRAIN_WORKER_COUNT, 4)
}

/// A completed section job that is older than this is still valid, but is marked slow so a
/// caller can account for a worker that missed the frame-shaped meshing target.  Meshing itself
/// is not forcibly killed halfway through a 16³ walk: doing so would lose the section's blank/
/// visibility result and would make a later re-mesh race the cancelled worker.  Cancellation is
/// cooperative at the task boundary instead.
pub const DEFAULT_SLOW_MESH_DEADLINE: Duration = Duration::from_millis(50);

/// Monotonic identity attached to one queued section rebuild.
///
/// This is process-wide rather than queue-local so resetting the client's queue on world unload
/// cannot make a new world's revision look older than a renderer's tombstone for the old world.
pub type MeshRevision = u64;

static NEXT_MESH_REVISION: AtomicU64 = AtomicU64::new(1);

pub(crate) fn next_mesh_revision() -> MeshRevision {
    // Zero is reserved for the compatibility APIs that accept an unversioned `ChunkMesh`.
    let revision = NEXT_MESH_REVISION.fetch_add(1, AtomicOrdering::Relaxed);
    if revision == 0 {
        NEXT_MESH_REVISION.fetch_add(1, AtomicOrdering::Relaxed)
    } else {
        revision
    }
}

const NANOS_PER_SECOND: u64 = 1_000_000_000;
const VANILLA_MIN_FPS: u32 = 60;
const VANILLA_FRAME_FRACTION: u64 = 4;

/// The CPU time available to terrain work in one rendered frame.
///
/// java: `EntityRenderer.updateCameraAndRender` computes
/// `j = max(min(debugFPS, limitFramerate), 60)`, then
/// `max(1_000_000_000 / j / 4 - elapsedSinceFrameStart, 0)`. The divisions are deliberately
/// integer divisions in both implementations: at 60 FPS the whole slice is 4,166,666 ns, not a
/// rounded 4.166 ms floating-point value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshFrameBudget {
    /// Effective FPS divisor used by Java's integer expression.
    effective_fps: u32,
    /// The complete quarter-frame slice before prior frame work is subtracted.
    allotted_nanos: u64,
    /// Nanoseconds already spent before the terrain deadline was computed.
    spent_nanos: u64,
    /// Nanoseconds left for budgeted terrain work; zero means the deadline is already due.
    remaining_nanos: u64,
}

impl MeshFrameBudget {
    /// Compute Java 1.8.9's terrain deadline budget from the FPS readings available to the client.
    ///
    /// `debug_fps` is the last completed `Minecraft.debugFPS` window. `limit_framerate` is the
    /// raw `GameSettings.limitFramerate` integer, including the `260`/negative unlimited forms.
    /// `spent_nanos` is the `System.nanoTime() - nanoTime` work that happened before
    /// `EntityRenderer` computes its deadline.
    #[must_use]
    pub fn vanilla(debug_fps: u32, limit_framerate: i32, spent_nanos: u64) -> Self {
        // Java's Math.min is evaluated before Math.max. A non-positive max-FPS setting therefore
        // still lands on the 60-FPS floor rather than becoming an invalid divisor.
        let effective_fps = i64::from(debug_fps)
            .min(i64::from(limit_framerate))
            .max(i64::from(VANILLA_MIN_FPS)) as u32;
        let allotted_nanos = NANOS_PER_SECOND / u64::from(effective_fps) / VANILLA_FRAME_FRACTION;
        let remaining_nanos = allotted_nanos.saturating_sub(spent_nanos);
        Self {
            effective_fps,
            allotted_nanos,
            spent_nanos,
            remaining_nanos,
        }
    }

    /// Java's effective FPS divisor (`max(min(debugFPS, limitFramerate), 60)`).
    #[must_use]
    pub fn effective_fps(self) -> u32 {
        self.effective_fps
    }

    /// The unspent quarter-frame slice before earlier frame work is deducted.
    #[must_use]
    pub fn allotted_nanos(self) -> u64 {
        self.allotted_nanos
    }

    /// Work already elapsed when Java computes the deadline.
    #[must_use]
    pub fn spent_nanos(self) -> u64 {
        self.spent_nanos
    }

    /// Nanoseconds left for deadline-bound terrain work.
    #[must_use]
    pub fn remaining_nanos(self) -> u64 {
        self.remaining_nanos
    }
}

/// The render-thread handoff from one background terrain rebuild.
///
/// `ChunkMesh` deliberately remains the frozen geometry type used by the CPU mesher and the
/// upload planner.  This wrapper carries the lifecycle data that was missing when a bare mesh
/// crossed the worker boundary: an old result can now be discarded after cancellation or a
/// newer rebuild, while `slow` lets the client observe a worker that ran past its diagnostic
/// deadline without throwing away a valid blank/tile/visibility result.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshTaskResult {
    mesh: ChunkMesh,
    revision: MeshRevision,
    elapsed: Duration,
    slow: bool,
    order: u64,
}

impl MeshTaskResult {
    fn new(
        mesh: ChunkMesh,
        revision: MeshRevision,
        elapsed: Duration,
        slow_mesh_deadline: Duration,
        order: u64,
    ) -> Self {
        Self {
            mesh,
            revision,
            elapsed,
            slow: elapsed > slow_mesh_deadline,
            order,
        }
    }

    /// Section this result was built for.
    #[must_use]
    pub fn pos(&self) -> ChunkSectionPos {
        self.mesh.pos
    }

    /// Queue revision captured when this task was dispatched.
    #[must_use]
    pub fn revision(&self) -> MeshRevision {
        self.revision
    }

    /// CPU duration of the worker's section walk.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.elapsed
    }

    /// Whether the worker exceeded [`DEFAULT_SLOW_MESH_DEADLINE`] (or the configured override).
    #[must_use]
    pub fn is_slow(&self) -> bool {
        self.slow
    }

    /// Whether this result is the blank/all-air section result.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mesh.is_empty()
    }

    /// Borrow the completed mesh without dropping its lifecycle metadata.
    #[must_use]
    pub fn mesh(&self) -> &ChunkMesh {
        &self.mesh
    }

    /// Consume the handoff and return the frozen mesh payload.
    #[must_use]
    pub fn into_mesh(self) -> ChunkMesh {
        self.mesh
    }

    /// Internal compatibility constructor for APIs that predate revisioned worker results.
    pub(crate) fn unversioned(mesh: ChunkMesh) -> Self {
        Self {
            order: 0,
            revision: 0,
            elapsed: Duration::ZERO,
            slow: false,
            mesh,
        }
    }

    /// Tests: a result carrying `revision`, as a worker would send it.
    #[cfg(test)]
    pub(crate) fn with_revision(mesh: ChunkMesh, revision: MeshRevision) -> Self {
        Self {
            revision,
            ..Self::unversioned(mesh)
        }
    }

    fn order(&self) -> u64 {
        self.order
    }
}

/// The persistent two-thread pool corresponding to vanilla's `Chunk Batcher` workers.
///
/// `mesh_section_with_alternatives` borrows its world, so a detached job cannot borrow the live
/// client world across frames. [`World::render_snapshot`](mc_world::World::render_snapshot)
/// gives each dispatch one owned, immutable world value; the pool then shares that snapshot
/// across its workers. Unlike a count-based main-thread loop, selection is still governed by
/// the frame deadline and each section is executed by the dedicated terrain workers.
fn terrain_mesh_pool() -> &'static rayon::ThreadPool {
    TERRAIN_MESH_POOL.get_or_init(|| {
        terrain_pool_builder(terrain_pool_size())
            .build()
            .expect("terrain worker pool must be constructible")
    })
}

/// [`terrain_mesh_pool`]'s pool, once something has meshed.
static TERRAIN_MESH_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

/// Have each terrain worker return its allocator's freed pages to the OS
/// (`crate::w21_mb::collect_on_thread`), as a background job queued behind whatever it is doing.
/// The game thread frees the meshes the workers allocated, and those pages are only reclaimed by
/// the worker that owns them, which after a world is torn down may not run again for a long time.
/// Nothing happens if no section was ever meshed (the pool is not created for this), and the
/// caller never waits.
pub(crate) fn collect_worker_heaps() {
    if let Some(pool) = TERRAIN_MESH_POOL.get() {
        pool.spawn_broadcast(|_| {
            let _ = crate::w21_mb::collect_on_thread();
        });
    }
}

/// The terrain pool's configuration, separate so a test can build the same pool.
///
/// A job given to `ThreadPool::spawn` that panics goes to the pool's panic
/// handler, and without one rayon **aborts the process**: a mesher bug on one odd block made
/// the client vanish with no crash report, because the report the game's panic hook had just
/// recorded (`BlockMeshingPanic`, vanilla's "Tesselating block in world") was never shown. The
/// handler keeps the worker alive; the game sees the recorded report on its next frame and
/// stops with it, which is what vanilla's `ChunkRenderWorker` does through `Minecraft.crashed`.
fn terrain_pool_builder(threads: usize) -> rayon::ThreadPoolBuilder {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|index| format!("Terrain Batcher {index}"))
        .panic_handler(|_payload| {
            MESH_WORKER_PANICS.fetch_add(1, AtomicOrdering::Relaxed);
            tracing::error!("a terrain mesh worker panicked; the section was not meshed");
        })
}

/// Terrain mesh jobs that panicked since start (see [`terrain_pool_builder`]).
static MESH_WORKER_PANICS: AtomicU64 = AtomicU64::new(0);

/// How many terrain mesh jobs have panicked on the worker pool since the process started. The
/// panic itself is the game's crash report (the panic hook records it); this count is for
/// diagnostics and tests.
#[must_use]
pub fn mesh_worker_panics() -> u64 {
    MESH_WORKER_PANICS.load(AtomicOrdering::Relaxed)
}

/// Per-frame lifecycle counters and pending-age snapshots for an opt-in diagnostic run.
///
/// Ordinary queue operation leaves this state disabled: no age map is allocated, no frame index
/// is read, and no event counters are updated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MeshQueueDiagnostics {
    /// Number of sections in the active bounded queue at the snapshot.
    pub active: u64,
    /// Number of sections retained after an active-queue offer failed.
    pub deferred: u64,
    /// Number of sections currently owned by worker tasks.
    pub in_flight: u64,
    /// New logical rebuilds admitted to the active or deferred queues this frame.
    pub enqueued: u64,
    /// New rebuilds placed in the deferred queue because the active queue was full.
    pub enqueue_overflow: u64,
    /// Actual deferred-to-active membership transitions this frame.
    pub deferred_promoted: u64,
    /// Tasks handed to the background dispatcher this frame.
    pub dispatched: u64,
    /// Sections built through the synchronous drain path this frame.
    pub synchronous_built: u64,
    /// Worker results accepted by the queue this frame.
    pub completed: u64,
    /// Accepted worker results whose build duration exceeded the slow threshold.
    pub slow_completed: u64,
    /// Logical rebuilds cancelled this frame.
    pub cancelled: u64,
    /// Worker results rejected because their in-flight or queue revision was stale.
    pub stale_result_rejected: u64,
    /// Oldest active age in diagnostic frames, or `None` when an active age is unknown.
    pub oldest_active_age_frames: Option<u64>,
    /// Oldest deferred age in diagnostic frames, or `None` when a deferred age is unknown.
    pub oldest_deferred_age_frames: Option<u64>,
    /// Oldest in-flight age in diagnostic frames, or `None` when an in-flight age is unknown.
    pub oldest_in_flight_age_frames: Option<u64>,
}

/// Error returned when a diagnostic frame index does not advance monotonically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshQueueDiagnosticsError {
    /// The caller supplied a frame index that is not greater than the previous one.
    NonMonotonicFrameIndex {
        /// The last frame index accepted by the queue.
        previous: u64,
        /// The rejected frame index supplied by the caller.
        current: u64,
    },
}

impl std::fmt::Display for MeshQueueDiagnosticsError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonMonotonicFrameIndex { previous, current } => write!(
                formatter,
                "diagnostic frame index must increase: previous={previous}, current={current}"
            ),
        }
    }
}

impl std::error::Error for MeshQueueDiagnosticsError {}

#[derive(Debug, Default)]
struct MeshQueueDiagnosticsState {
    frame_index: Option<u64>,
    age_start: HashMap<ChunkSectionPos, u64>,
    enqueued: u64,
    enqueue_overflow: u64,
    deferred_promoted: u64,
    dispatched: u64,
    synchronous_built: u64,
    completed: u64,
    slow_completed: u64,
    cancelled: u64,
    stale_result_rejected: u64,
}

fn oldest_age<'a, I>(
    ages: &HashMap<ChunkSectionPos, u64>,
    current_frame: Option<u64>,
    positions: I,
) -> Option<u64>
where
    I: Iterator<Item = &'a ChunkSectionPos>,
{
    let current_frame = current_frame?;
    let mut oldest = None;
    for pos in positions {
        let start = ages.get(pos)?;
        let age = current_frame.saturating_sub(*start);
        oldest = Some(oldest.map_or(age, |current: u64| current.max(age)));
    }
    oldest
}

/// The pending re-mesh queue.
///
/// Feed it from [`World::drain_dirty_sections`](mc_world::World::drain_dirty_sections) once per
/// tick and drain it once per frame. The active queue is capped at [`MAX_PENDING`] to match
/// Java's bounded dispatcher; overflow remains in the deferred set so a dirty burst cannot lose
/// invalidations.
///
/// The queue uses hash membership indexes for O(n) admission and nearest-section selection for
/// bounded drains. The optional diagnostic state records lifecycle transitions separately and is
/// absent from ordinary queue instances.
#[derive(Debug)]
pub struct MeshQueue {
    pending: Vec<ChunkSectionPos>,
    /// Membership index for `pending`. Kept in step with it on every mutation; the invariant is
    /// asserted by `the_membership_index_matches_the_queue`.
    queued: HashSet<ChunkSectionPos>,
    /// Dirty sections observed after the Java-shaped pending queue filled. A failed direct
    /// [`Self::enqueue`] offer is retained here as well as reported to its caller: this is the
    /// queue-local equivalent of Java leaving the failed task in `RenderGlobal.chunksToUpdate`,
    /// and it prevents a Rust caller that ignores the old unit return value from losing work.
    ///
    /// The world dirty-set path has no selective reinsert API, so it has to retain overflow here
    /// or a full queue would silently lose a render invalidation when `drain_dirty_sections`
    /// clears the world set.
    deferred: Vec<ChunkSectionPos>,
    /// Membership index for `deferred`; kept separate from `queued` so `len()` remains the
    /// caller-visible pending queue size and never exceeds [`MAX_PENDING`].
    deferred_set: HashSet<ChunkSectionPos>,
    settings: MeshSettings,
    allow_block_alternatives: bool,
    /// Latest lifecycle revision known for each section.  A revision remains present after
    /// cancellation so a result already on the worker's stack cannot become uploadable again.
    revisions: HashMap<ChunkSectionPos, MeshRevision>,
    /// Tasks currently owned by the two-worker dispatcher.  They are deliberately separate from
    /// `pending`: a frame may hand work to the workers and return while those tasks are still
    /// compiling, exactly as `ChunkRenderDispatcher.updateChunkLater` does.
    in_flight: HashMap<ChunkSectionPos, InFlightMeshTask>,
    /// Blank results for dirty sections that had no block storage when they were admitted (an
    /// unloaded column, or a section the server never sent). They never take a `pending` or
    /// `in_flight` slot, a snapshot or a worker; the next poll or drain hands them over, still
    /// gated on their revision so a later dirty event or cancellation supersedes them.
    absent_ready: Vec<AbsentReady>,
    /// `absent_ready`'s length at which [`Self::compact_absent_ready`] next
    /// drops the results that later events superseded.
    absent_compact_at: usize,
    result_rx: Receiver<MeshTaskResult>,
    result_tx: Sender<MeshTaskResult>,
    next_order: u64,
    slow_mesh_deadline: Duration,
    diagnostics: Option<MeshQueueDiagnosticsState>,
}

/// A blank result waiting in [`MeshQueue::absent_ready`]: the section, its
/// revision and its admission order. Its [`ChunkMesh::empty`] is built when it is handed over,
/// so a waiting one costs 32 bytes instead of a whole [`MeshTaskResult`] (about 700, most of it
/// the 512-byte occlusion snapshot): a replay seek without frames can leave hundreds of
/// thousands of them waiting.
#[derive(Debug, Clone, Copy)]
struct AbsentReady {
    pos: ChunkSectionPos,
    revision: MeshRevision,
    order: u64,
}

/// The smallest [`MeshQueue::absent_ready`] compaction threshold, more than a
/// join burst on a void map normally leaves between two frames.
const ABSENT_COMPACT_MIN: usize = 4096;

#[derive(Debug)]
struct InFlightMeshTask {
    revision: MeshRevision,
    cancelled: Arc<AtomicBool>,
}

impl Default for MeshQueue {
    fn default() -> Self {
        let (result_tx, result_rx) = mpsc::channel();
        Self {
            pending: Vec::new(),
            queued: HashSet::new(),
            deferred: Vec::new(),
            deferred_set: HashSet::new(),
            settings: MeshSettings::default(),
            allow_block_alternatives: true,
            revisions: HashMap::new(),
            in_flight: HashMap::new(),
            absent_ready: Vec::new(),
            absent_compact_at: ABSENT_COMPACT_MIN,
            result_rx,
            result_tx,
            next_order: 0,
            slow_mesh_deadline: DEFAULT_SLOW_MESH_DEADLINE,
            diagnostics: None,
        }
    }
}

impl MeshQueue {
    /// An empty queue with vanilla's default graphics settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable the bounded lifecycle diagnostics.
    ///
    /// Enabling after work already exists intentionally leaves those positions un-stamped; their
    /// age remains `None` until they leave the queue. Disabling drops all diagnostic state.
    pub fn set_diagnostics_enabled(&mut self, enabled: bool) {
        match (enabled, self.diagnostics.is_some()) {
            (true, false) => self.diagnostics = Some(MeshQueueDiagnosticsState::default()),
            (false, true) => self.diagnostics = None,
            _ => {}
        }
    }

    /// Begin a diagnostic frame and reset that frame's event counters.
    ///
    /// The caller owns the logical frame index. It must increase strictly while diagnostics are
    /// enabled; no host clock is consulted. Calling this while diagnostics are disabled is a
    /// successful no-op so ordinary clients can share a guarded integration path.
    pub fn begin_diagnostic_frame(
        &mut self,
        frame_index: u64,
    ) -> Result<(), MeshQueueDiagnosticsError> {
        let Some(diagnostics) = self.diagnostics.as_mut() else {
            return Ok(());
        };
        if let Some(previous) = diagnostics.frame_index {
            if frame_index <= previous {
                return Err(MeshQueueDiagnosticsError::NonMonotonicFrameIndex {
                    previous,
                    current: frame_index,
                });
            }
        }
        diagnostics.frame_index = Some(frame_index);
        diagnostics.enqueued = 0;
        diagnostics.enqueue_overflow = 0;
        diagnostics.deferred_promoted = 0;
        diagnostics.dispatched = 0;
        diagnostics.synchronous_built = 0;
        diagnostics.completed = 0;
        diagnostics.slow_completed = 0;
        diagnostics.cancelled = 0;
        diagnostics.stale_result_rejected = 0;
        Ok(())
    }

    /// Snapshot queue lengths, this-frame lifecycle events, and known pending ages.
    #[must_use]
    pub fn diagnostic_stats(&self) -> Option<MeshQueueDiagnostics> {
        let diagnostics = self.diagnostics.as_ref()?;
        Some(MeshQueueDiagnostics {
            active: self.pending.len() as u64,
            deferred: self.deferred.len() as u64,
            in_flight: self.in_flight.len() as u64,
            enqueued: diagnostics.enqueued,
            enqueue_overflow: diagnostics.enqueue_overflow,
            deferred_promoted: diagnostics.deferred_promoted,
            dispatched: diagnostics.dispatched,
            synchronous_built: diagnostics.synchronous_built,
            completed: diagnostics.completed,
            slow_completed: diagnostics.slow_completed,
            cancelled: diagnostics.cancelled,
            stale_result_rejected: diagnostics.stale_result_rejected,
            oldest_active_age_frames: oldest_age(
                &diagnostics.age_start,
                diagnostics.frame_index,
                self.pending.iter(),
            ),
            oldest_deferred_age_frames: oldest_age(
                &diagnostics.age_start,
                diagnostics.frame_index,
                self.deferred.iter(),
            ),
            oldest_in_flight_age_frames: oldest_age(
                &diagnostics.age_start,
                diagnostics.frame_index,
                self.in_flight.keys(),
            ),
        })
    }

    fn diagnostic_admitted(&mut self, pos: ChunkSectionPos, overflow: bool) {
        let Some(diagnostics) = self.diagnostics.as_mut() else {
            return;
        };
        diagnostics.enqueued += 1;
        if overflow {
            diagnostics.enqueue_overflow += 1;
        }
        if let Some(frame_index) = diagnostics.frame_index {
            diagnostics.age_start.insert(pos, frame_index);
        }
    }

    fn diagnostic_cancelled(&mut self, pos: ChunkSectionPos) {
        let Some(diagnostics) = self.diagnostics.as_mut() else {
            return;
        };
        diagnostics.cancelled += 1;
        diagnostics.age_start.remove(&pos);
    }

    fn diagnostic_promoted(&mut self) {
        if let Some(diagnostics) = self.diagnostics.as_mut() {
            diagnostics.deferred_promoted += 1;
        }
    }

    fn diagnostic_dispatched(&mut self, pos: ChunkSectionPos) {
        let Some(diagnostics) = self.diagnostics.as_mut() else {
            return;
        };
        diagnostics.dispatched += 1;
        // A caller that enabled diagnostics after admission has an unknown age; do not invent
        // frame zero. A known stamp naturally survives the active -> in-flight transition.
        if diagnostics.frame_index.is_none() {
            diagnostics.age_start.remove(&pos);
        }
    }

    fn diagnostic_synchronous_built<I>(&mut self, positions: I)
    where
        I: IntoIterator<Item = ChunkSectionPos>,
    {
        let Some(diagnostics) = self.diagnostics.as_mut() else {
            return;
        };
        for pos in positions {
            diagnostics.synchronous_built += 1;
            diagnostics.age_start.remove(&pos);
        }
    }

    fn diagnostic_completed(&mut self, result: &MeshTaskResult) {
        let Some(diagnostics) = self.diagnostics.as_mut() else {
            return;
        };
        diagnostics.completed += 1;
        if result.is_slow() {
            diagnostics.slow_completed += 1;
        }
        diagnostics.age_start.remove(&result.pos());
    }

    fn diagnostic_stale_result(&mut self) {
        if let Some(diagnostics) = self.diagnostics.as_mut() {
            diagnostics.stale_result_rejected += 1;
        }
    }

    /// Replace the graphics settings future meshes are built with. Changing these invalidates
    /// every existing mesh, exactly as toggling fancy graphics does in vanilla.
    pub fn set_settings(&mut self, settings: MeshSettings) {
        self.settings = settings;
        self.cancel_in_flight_tasks();
    }

    /// The settings meshes are currently built with.
    #[must_use]
    pub fn settings(&self) -> MeshSettings {
        self.settings
    }

    /// Set whether weighted blockstates choose a deterministic position alternative.
    ///
    /// Disabling this makes future meshes use the baked base candidate, matching vanilla's
    /// `allowBlockAlternatives` option. Existing uploaded sections still need to be enqueued by
    /// the caller, just like changes to [`Self::set_settings`].
    pub fn set_allow_block_alternatives(&mut self, allow: bool) {
        self.allow_block_alternatives = allow;
        self.cancel_in_flight_tasks();
    }

    /// Whether future meshes use position-weighted block alternatives.
    #[must_use]
    pub fn allow_block_alternatives(&self) -> bool {
        self.allow_block_alternatives
    }

    /// Override the diagnostic threshold used to mark a completed worker result as slow.
    ///
    /// The threshold never cancels a walk: a slow result can still contain the only correct
    /// blank-section/tile-neighbour snapshot for this revision, and dropping it would create a
    /// permanent hole until another dirty event arrives.
    pub fn set_slow_mesh_deadline(&mut self, deadline: Duration) {
        self.slow_mesh_deadline = deadline;
    }

    /// The current slow-mesh diagnostic threshold.
    #[must_use]
    pub fn slow_mesh_deadline(&self) -> Duration {
        self.slow_mesh_deadline
    }

    /// Latest revision known for `pos`, if it has ever been queued or cancelled.
    #[must_use]
    pub fn revision(&self, pos: ChunkSectionPos) -> Option<MeshRevision> {
        self.revisions.get(&pos).copied()
    }

    /// Number of section rebuilds currently running on the terrain workers, plus blank
    /// absent-section results waiting for the next poll. Both are handed-off work whose result
    /// has not been returned yet, so "nothing in flight" still means "nothing outstanding".
    #[must_use]
    pub fn in_flight_len(&self) -> usize {
        self.in_flight.len() + self.absent_ready.len()
    }

    /// The sections whose rebuild is running on the terrain workers:
    /// what [`Self::set_settings`] cancels. A caller re-meshing for new settings queues these
    /// again (a section meshed for the first time has no buffer to be re-meshed from).
    #[must_use]
    pub fn in_flight_positions(&self) -> Vec<ChunkSectionPos> {
        self.in_flight.keys().copied().collect()
    }

    fn allocate_revision(&mut self, pos: ChunkSectionPos) -> MeshRevision {
        let revision = next_mesh_revision();
        self.revisions.insert(pos, revision);
        revision
    }

    fn cancel_in_flight_tasks(&mut self) {
        let positions: Vec<_> = self.in_flight.keys().copied().collect();
        for pos in positions {
            if let Some(task) = self.in_flight.remove(&pos) {
                task.cancelled.store(true, AtomicOrdering::Relaxed);
                self.allocate_revision(pos);
                self.diagnostic_cancelled(pos);
            }
        }
    }

    fn cancel_in_flight(&mut self, pos: ChunkSectionPos) -> bool {
        let Some(task) = self.in_flight.remove(&pos) else {
            return false;
        };
        task.cancelled.store(true, AtomicOrdering::Relaxed);
        true
    }

    /// Cancel every queued/deferred/in-flight rebuild, used when the owning client world is
    /// replaced.  Existing revisions remain tombstones so a worker that races the reset cannot
    /// send an old-world result into a newly-created queue slot.
    pub fn cancel_all(&mut self) -> usize {
        let mut positions: HashSet<_> = self.pending.iter().copied().collect();
        positions.extend(self.deferred.iter().copied());
        positions.extend(self.in_flight.keys().copied());
        positions.extend(self.absent_ready.iter().map(|ready| ready.pos));
        let count = positions.len();
        for pos in positions {
            self.cancel(pos);
        }
        // Each one now carries a superseded revision; drop them rather than poll them later.
        self.absent_ready.clear();
        count
    }

    /// Reset all world-owned queue state while preserving client queue options and the optional
    /// diagnostic frame/counter state.
    ///
    /// This is the renderer-side half of a JoinGame/dimension replacement. Existing queued,
    /// deferred, and in-flight work is cooperatively cancelled first; then the membership,
    /// revisions, ordering, and result channel are replaced wholesale so an old worker cannot
    /// reinsert a result into the new world's queue. Diagnostic counters remain in the current
    /// logical frame (including cancellations caused by this reset), while all pending ages are
    /// discarded because no age may cross a world boundary. The default diagnostics-disabled
    /// path remains allocation-free apart from the fresh empty queue/channel state.
    pub fn reset_world(&mut self) {
        self.cancel_all();

        let (result_tx, result_rx) = mpsc::channel();
        self.pending = Vec::new();
        self.queued = HashSet::new();
        self.deferred = Vec::new();
        self.deferred_set = HashSet::new();
        self.revisions = HashMap::new();
        self.in_flight = HashMap::new();
        self.absent_ready = Vec::new();
        self.absent_compact_at = ABSENT_COMPACT_MIN;
        self.result_rx = result_rx;
        self.result_tx = result_tx;
        self.next_order = 0;
        if let Some(diagnostics) = self.diagnostics.as_mut() {
            diagnostics.age_start.clear();
        }
    }

    /// Take everything `world` has marked dirty since the last call and retain it for meshing.
    ///
    /// The first [`MAX_PENDING`] unique sections occupy the Java-shaped active queue. Remaining
    /// sections are held in [`Self::deferred`] because [`World::drain_dirty_sections`] clears the
    /// world-owned set atomically and has no selective reinsert operation. They are promoted as
    /// active slots become available, so a render-distance invalidation cannot disappear merely
    /// because more than 100 sections changed in one tick.
    ///
    /// Returns the number of newly retained sections (active plus deferred). Duplicate sections
    /// already in either set are ignored.
    ///
    /// java: `RenderGlobal.markBlocksForUpdate` feeding `chunksToUpdate`.
    pub fn enqueue_dirty(&mut self, world: &mut World) -> usize {
        mc_core::profile_scope!("terrain.enqueue_dirty");
        self.retry_deferred();
        let mut retained = 0;
        for pos in world.drain_dirty_sections() {
            if self.queued.contains(&pos) || self.deferred_set.contains(&pos) {
                continue;
            }
            // A dirty event supersedes a worker that is already compiling this section.  Keep
            // the old task's cancellation token, but assign the replacement a new revision so
            // whichever result reaches the render thread first cannot overwrite the newer world.
            if self.in_flight.contains_key(&pos) {
                self.cancel_in_flight(pos);
                self.diagnostic_cancelled(pos);
            }
            let revision = self.allocate_revision(pos);
            // #2 / JOBS-1: a section with no block storage (its column is not loaded, or the
            // server never sent it) can only build `ChunkMesh::empty`, which is exactly what
            // `dispatch_one` would send after a pending slot, an in-flight slot and a snapshot.
            // On a void map that is most of every join burst. The blank result is ready now and
            // still revisioned, so a later dirty event (the section gains storage) or a cancel
            // supersedes it, and the client still turns an unloaded column's result into a
            // removal. Neighbours are unaffected: their own dirty events queue them as before.
            if section_is_absent(world, pos) {
                self.diagnostic_admitted(pos, false);
                self.next_order = self.next_order.wrapping_add(1);
                if self.absent_ready.len() >= self.absent_compact_at {
                    self.compact_absent_ready();
                }
                // Its blank mesh is built when it is handed over.
                self.absent_ready.push(AbsentReady {
                    pos,
                    revision,
                    order: self.next_order,
                });
                retained += 1;
                continue;
            }
            if self.pending.len() < MAX_PENDING {
                self.queued.insert(pos);
                self.pending.push(pos);
                self.diagnostic_admitted(pos, false);
            } else {
                self.deferred_set.insert(pos);
                self.deferred.push(pos);
                self.diagnostic_admitted(pos, true);
            }
            retained += 1;
        }
        retained
    }

    /// Offer one section to the bounded active queue, ignoring duplicates.
    ///
    /// `true` means the section is already active or was accepted into the active queue.
    /// `false` is the caller-visible Java `BlockingQueue.offer` failure: the section remains in
    /// the deferred retry set and the caller may retry/requeue it after a later drain. Retaining
    /// it here also protects older Rust callers that ignored the former unit return value.
    pub fn enqueue(&mut self, pos: ChunkSectionPos) -> bool {
        if self.queued.contains(&pos) {
            return true;
        }
        if self.deferred_set.contains(&pos) {
            if self.pending.len() >= MAX_PENDING {
                return false;
            }
            self.promote_deferred_position(pos);
            return true;
        }
        if self.in_flight.contains_key(&pos) {
            self.cancel_in_flight(pos);
            self.diagnostic_cancelled(pos);
        }
        self.allocate_revision(pos);
        if self.pending.len() >= MAX_PENDING {
            self.deferred_set.insert(pos);
            self.deferred.push(pos);
            self.diagnostic_admitted(pos, true);
            return false;
        }
        self.queued.insert(pos);
        self.pending.push(pos);
        self.diagnostic_admitted(pos, false);
        true
    }

    /// Retry deferred dirty sections in deterministic insertion order.
    ///
    /// This is also run by [`Self::enqueue_dirty`] and by the priority-taking methods. It is
    /// public so a caller that explicitly owns a frame/tick boundary can make the retry visible
    /// without manufacturing a second world dirty event.
    pub fn retry_deferred(&mut self) -> usize {
        let room = MAX_PENDING.saturating_sub(self.pending.len());
        let promote = room.min(self.deferred.len());
        let promoted_positions: Vec<_> = self.deferred.drain(..promote).collect();
        for pos in promoted_positions {
            self.deferred_set.remove(&pos);
            if self.queued.insert(pos) {
                self.pending.push(pos);
                self.diagnostic_promoted();
            }
        }
        promote
    }

    /// Promote one specific deferred offer when a caller retries it after a queue drain.
    fn promote_deferred_position(&mut self, pos: ChunkSectionPos) -> bool {
        let Some(index) = self.deferred.iter().position(|queued| *queued == pos) else {
            return false;
        };
        self.deferred.remove(index);
        self.deferred_set.remove(&pos);
        self.queued.insert(pos);
        self.pending.push(pos);
        self.diagnostic_promoted();
        true
    }

    /// Cancel a pending or deferred section.
    ///
    /// This is the queue-local equivalent of Java's finished compile task removing itself from
    /// `queueChunkUpdates`. A deferred section is cancelled too, so a stale dirty notification
    /// cannot resurrect work after its render chunk was unloaded.
    pub fn cancel(&mut self, pos: ChunkSectionPos) -> bool {
        let active = self.queued.remove(&pos);
        if active {
            self.pending.retain(|queued| *queued != pos);
        }
        let deferred = self.deferred_set.remove(&pos);
        if deferred {
            self.deferred.retain(|queued| *queued != pos);
        }
        let running = self.cancel_in_flight(pos);
        // Keep a tombstone even when no queue slot was visible.  The worker may have completed
        // between `try_recv` and this call, and its result must still be rejected by revision.
        self.allocate_revision(pos);
        if active || deferred || running {
            self.diagnostic_cancelled(pos);
            self.retry_deferred();
        }
        active || deferred || running
    }

    /// How many sections are waiting in the active bounded queue.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// How many dirty sections are waiting for an active queue slot.
    #[must_use]
    pub fn deferred_len(&self) -> usize {
        self.deferred.len()
    }

    /// Whether no active or deferred section is waiting.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
            && self.deferred.is_empty()
            && self.in_flight.is_empty()
            && self.absent_ready.is_empty()
    }

    /// Drop the blank results that a later dirty event or a cancel superseded
    /// (their revision is no longer the section's), keeping the rest in admission order: exactly
    /// what [`Self::take_absent_ready`] would drop and keep, so what a drain hands over is
    /// unchanged. Frames drain this list; a run without frames (a replay export's first seek,
    /// the headless seek bench) re-dirties the same absent sections every tick and piled up one
    /// stale result per section per tick: 1.73 million (about 1.2 GB) half way through an
    /// hour-long replay. The threshold doubles with what
    /// survives, so this stays amortised O(1) per push.
    fn compact_absent_ready(&mut self) {
        let before = self.absent_ready.len();
        let revisions = &self.revisions;
        self.absent_ready
            .retain(|ready| revisions.get(&ready.pos).copied() == Some(ready.revision));
        if let Some(diagnostics) = self.diagnostics.as_mut() {
            diagnostics.stale_result_rejected += (before - self.absent_ready.len()) as u64;
        }
        self.absent_compact_at = (self.absent_ready.len() * 2).max(ABSENT_COMPACT_MIN);
    }

    /// Hand over the blank absent-section results that are still current, in admission order.
    fn take_absent_ready(&mut self) -> Vec<MeshTaskResult> {
        let mut ready = std::mem::take(&mut self.absent_ready);
        let mut current = Vec::with_capacity(ready.len());
        for waiting in ready.drain(..) {
            if self.revisions.get(&waiting.pos).copied() == Some(waiting.revision) {
                // Exactly the result admission used to build and store.
                let result = MeshTaskResult::new(
                    ChunkMesh::empty(waiting.pos),
                    waiting.revision,
                    Duration::ZERO,
                    self.slow_mesh_deadline,
                    waiting.order,
                );
                self.diagnostic_completed(&result);
                current.push(result);
            } else {
                self.diagnostic_stale_result();
            }
        }
        // Keep the allocation for the next burst.
        self.absent_ready = ready;
        self.absent_compact_at = ABSENT_COMPACT_MIN;
        current
    }

    /// Mesh up to `budget` sections in parallel, nearest to the camera first.
    ///
    /// This explicit-count entry point is retained for deterministic benches and callers that
    /// have their own scheduling policy. The live client uses [`Self::drain_for_frame`], because
    /// vanilla's policy is a time deadline rather than a fixed section count.
    pub fn drain(&mut self, world: &World, camera: DVec3, budget: usize) -> Vec<ChunkMesh> {
        mc_core::profile_scope!("terrain.drain_mesh_queue");
        // Blank absent-section results never entered the queue, so they are not part of the
        // `budget`; they cost no meshing.
        let mut done: Vec<ChunkMesh> = self
            .take_absent_ready()
            .into_iter()
            .map(MeshTaskResult::into_mesh)
            .collect();
        let batch = self.take_nearest(camera, budget);
        if batch.is_empty() {
            return done;
        }
        self.diagnostic_synchronous_built(batch.iter().copied());
        done.extend(self.mesh_batch(world, batch));
        done
    }

    /// Mesh the queue under Java 1.8.9's frame-rate-shaped deadline.
    ///
    /// `debug_fps` and `limit_framerate` are the same two values used by
    /// `EntityRenderer.updateCameraAndRender`. The deadline starts at this call because the
    /// caller does not expose the outer `runGameLoop` timestamp; use
    /// [`Self::drain_for_frame_with_elapsed`] when that earlier frame work is available.
    ///
    /// Sections in the camera's ±16-block neighbourhood are built synchronously before the
    /// deadline-bound queue, matching `RenderGlobal.setupTerrain`'s unbudgeted `updateChunkNow`
    /// branch. Remaining work is taken nearest-first in two-section worker batches until the
    /// deadline is due.
    pub fn drain_for_frame(
        &mut self,
        world: &World,
        camera: DVec3,
        debug_fps: u32,
        limit_framerate: i32,
    ) -> Vec<ChunkMesh> {
        self.drain_for_frame_with_elapsed(world, camera, debug_fps, limit_framerate, 0)
    }

    /// [`Self::drain_for_frame`] with explicit pre-render elapsed time.
    ///
    /// `spent_nanos` is subtracted using the exact Java integer divisions before the local
    /// deadline is made. It is public so a shell integration with an outer frame timestamp can
    /// preserve the last few nanoseconds of the vanilla contract without changing queue state.
    pub fn drain_for_frame_with_elapsed(
        &mut self,
        world: &World,
        camera: DVec3,
        debug_fps: u32,
        limit_framerate: i32,
        spent_nanos: u64,
    ) -> Vec<ChunkMesh> {
        mc_core::profile_scope!("terrain.drain_mesh_frame");
        let budget = MeshFrameBudget::vanilla(debug_fps, limit_framerate, spent_nanos);
        let frame_start = Instant::now();
        let deadline = frame_start + Duration::from_nanos(budget.remaining_nanos());
        let settings = self.settings;
        let allow_block_alternatives = self.allow_block_alternatives;

        // Java's setupTerrain does this before updateChunks and deliberately does not charge the
        // near-camera compile to `finishTimeNano`. Keep it separate from the budgeted batches.
        let near = self.take_near(camera);
        self.diagnostic_synchronous_built(near.iter().copied());
        // `updateChunkNow` uses the dispatcher's dedicated render worker directly on the render
        // thread; do the same for this small, unbudgeted neighbourhood instead of sending it to
        // the background pool used by `updateChunkLater`.
        let mut done: Vec<ChunkMesh> = self
            .take_absent_ready()
            .into_iter()
            .map(MeshTaskResult::into_mesh)
            .collect();
        done.extend(near.into_iter().map(|pos| {
            mesh_section_with_alternatives(world, pos, settings, allow_block_alternatives)
        }));

        while Instant::now() < deadline {
            let batch = self.take_nearest(camera, TERRAIN_WORKER_COUNT);
            if batch.is_empty() {
                break;
            }
            self.diagnostic_synchronous_built(batch.iter().copied());
            done.extend(mesh_batch(world, batch, settings, allow_block_alternatives));
        }
        done
    }

    /// Dispatch terrain work without waiting for a worker to finish.
    ///
    /// This is the live-client equivalent of Java's `updateChunkNow`/`updateChunkLater` split at
    /// the dispatcher boundary.  The old [`Self::drain_for_frame_with_elapsed`] API remains a
    /// synchronous compatibility seam for deterministic callers and benches; the client uses
    /// this method so a slow 16³ walk cannot stall the frame.  Completed results from an earlier
    /// frame are returned first.  Results dispatched by this frame remain owned by the queue and
    /// are polled on a later frame, even if a very fast worker happens to finish before this
    /// method returns; that one-frame handoff is what keeps the frame boundary non-blocking and
    /// deterministic.
    pub fn dispatch_for_frame(
        &mut self,
        world: &World,
        camera: DVec3,
        debug_fps: u32,
        limit_framerate: i32,
    ) -> Vec<MeshTaskResult> {
        self.dispatch_for_frame_with_elapsed(world, camera, debug_fps, limit_framerate, 0)
    }

    /// [`Self::dispatch_for_frame`] with explicit pre-render elapsed time.
    ///
    /// `spent_nanos` consumes the same absolute Java-shaped quarter-frame slice as the upload
    /// caller; it does not make this method wait for a worker.
    pub fn dispatch_for_frame_with_elapsed(
        &mut self,
        world: &World,
        camera: DVec3,
        debug_fps: u32,
        limit_framerate: i32,
        spent_nanos: u64,
    ) -> Vec<MeshTaskResult> {
        mc_core::profile_scope!("terrain.dispatch_mesh_frame");
        let done = self.poll_completed();
        let snapshot = if self.pending.is_empty() && self.deferred.is_empty() {
            None
        } else {
            // The only world value a detached task receives is this owned snapshot.  In
            // particular, no `&World` borrow escapes this call and no worker can observe a
            // half-mutated live chunk column.
            Some(Arc::new(self.dispatch_snapshot(world)))
        };

        let Some(snapshot) = snapshot else {
            return done;
        };

        // Java has no snapshot step to pay for, so the budget starts after it. Counting it would
        // let a slow snapshot spend the whole slice and leave one batch per frame.
        let budget = MeshFrameBudget::vanilla(debug_fps, limit_framerate, spent_nanos);
        let deadline = Instant::now() + Duration::from_nanos(budget.remaining_nanos());

        // Keep Java's unbudgeted near-camera preference, but dispatch it cooperatively.  A
        // pathological section is still allowed to finish and retain its blank/tile result; the
        // frame itself is never held hostage by that section.
        let near = self.take_near(camera);
        for pos in near {
            self.dispatch_one(Arc::clone(&snapshot), pos);
        }

        // The deadline governs how much work this frame hands to the dispatcher, not how long it
        // waits.  As in `RenderGlobal.updateChunks`, the first later task is offered even when
        // the deadline is already due; subsequent offers stop at the absolute boundary.  Each
        // worker task is one section, matching Java's compile-task lifecycle and making
        // cancellation/revision checks atomic at the section boundary.  The Java capacity applies
        // to tasks already handed to the dispatcher (the two worker-owned tasks plus its bounded
        // waiting queue), so a full `pending` list must still be drained into `in_flight`.
        let mut dispatched_any = false;
        while self.in_flight.len() < MAX_PENDING && (!dispatched_any || Instant::now() <= deadline)
        {
            let batch = self.take_nearest(camera, TERRAIN_WORKER_COUNT);
            if batch.is_empty() {
                break;
            }
            for pos in batch {
                self.dispatch_one(Arc::clone(&snapshot), pos);
            }
            dispatched_any = true;
        }

        done
    }

    /// Poll completed worker results without blocking.
    ///
    /// Results are ordered by dispatch sequence rather than worker completion time.  Rayon is
    /// intentionally free to finish either section first, while uploads retain the queue's
    /// nearest-first ordering and therefore the same deterministic blank/tile replacement order
    /// as the synchronous compatibility path.
    pub fn poll_completed(&mut self) -> Vec<MeshTaskResult> {
        let mut completed = Vec::new();
        while let Ok(result) = self.result_rx.try_recv() {
            let current = self.in_flight.get(&result.pos());
            if current.is_some_and(|task| task.revision == result.revision)
                && self.revisions.get(&result.pos()).copied() == Some(result.revision)
            {
                self.in_flight.remove(&result.pos());
                self.diagnostic_completed(&result);
                completed.push(result);
            } else {
                self.diagnostic_stale_result();
            }
            // A result from an older task is deliberately dropped.  Its worker may have raced a
            // cancel/re-enqueue, but it never reaches the renderer.
        }
        if !self.absent_ready.is_empty() {
            completed.extend(self.take_absent_ready());
        }
        completed.sort_by_key(MeshTaskResult::order);
        self.retry_deferred();
        completed
    }

    fn dispatch_snapshot(&self, world: &World) -> World {
        let queued = self.pending.len().saturating_add(self.deferred.len());
        // Conservative crossover from component experiments. Large bursts keep the
        // original broad snapshot and exactly the original budget/admission behavior.
        if queued <= 8 && world.loaded_chunk_count() >= 256 {
            let sections: Vec<_> = self
                .pending
                .iter()
                .chain(self.deferred.iter())
                .copied()
                .collect();
            world.render_snapshot_for_sections(&sections)
        } else {
            world.render_snapshot()
        }
    }

    fn dispatch_one(&mut self, snapshot: Arc<World>, pos: ChunkSectionPos) {
        let Some(revision) = self.revisions.get(&pos).copied() else {
            // `take_nearest` only returns queued entries, but keeping this guard makes the
            // cancellation invariant explicit if a caller mutates queue state in a future API.
            if let Some(diagnostics) = self.diagnostics.as_mut() {
                diagnostics.age_start.remove(&pos);
            }
            return;
        };
        if let Some(previous) = self.in_flight.remove(&pos) {
            previous.cancelled.store(true, AtomicOrdering::Relaxed);
        }
        let cancelled = Arc::new(AtomicBool::new(false));
        let worker_cancelled = Arc::clone(&cancelled);
        let sender = self.result_tx.clone();
        let settings = self.settings;
        let allow_block_alternatives = self.allow_block_alternatives;
        let slow_mesh_deadline = self.slow_mesh_deadline;
        self.next_order = self.next_order.wrapping_add(1);
        let order = self.next_order;
        self.in_flight.insert(
            pos,
            InFlightMeshTask {
                revision,
                cancelled,
            },
        );
        self.diagnostic_dispatched(pos);
        // A section the server never sent has nothing to walk, and most of a loaded world is sky.
        // Its result takes the same one-frame handoff, without a trip through the worker pool.
        if section_is_absent(&snapshot, pos) {
            let result = MeshTaskResult::new(
                ChunkMesh::empty(pos),
                revision,
                Duration::ZERO,
                slow_mesh_deadline,
                order,
            );
            let _ = sender.send(result);
            return;
        }
        terrain_mesh_pool().spawn(move || {
            if worker_cancelled.load(AtomicOrdering::Relaxed) {
                return;
            }
            let started = Instant::now();
            let mesh =
                mesh_section_with_alternatives(&snapshot, pos, settings, allow_block_alternatives);
            if worker_cancelled.load(AtomicOrdering::Relaxed) {
                return;
            }
            let elapsed = started.elapsed();
            let result = MeshTaskResult::new(mesh, revision, elapsed, slow_mesh_deadline, order);
            let _ = sender.send(result);
        });
    }

    /// Remove the `budget` sections nearest `camera` from the queue, nearest first.
    ///
    /// Split out of [`drain`](MeshQueue::drain) so the ordering rule can be tested without a
    /// world to mesh, and so a caller with its own worker pool can use the same policy.
    ///
    /// java: `RenderGlobal` sorts its update list so near chunks rebuild first. The full sort is
    /// deliberately **not** reproduced: only the `budget` sections that are about to be built
    /// need to be in order, and the queue behind them is re-examined next frame anyway from a
    /// camera that has moved. Selecting the nearest `budget` is O(n); sorting all of them is
    /// O(n log n) every frame over a queue that can hold ten thousand entries.
    pub fn take_nearest(&mut self, camera: DVec3, budget: usize) -> Vec<ChunkSectionPos> {
        self.promote_deferred_nearest(camera, budget);
        let take = budget.min(self.pending.len());
        if take == 0 {
            return Vec::new();
        }
        let key = |p: &ChunkSectionPos| distance_sq(*p, camera);
        // Distance alone is **not a total order**: a stationary camera makes exact ties routine
        // (every section on a shell around the eye has the same `distance_sq`), and which of two
        // tied sections is selected then depends on the order they arrived in — which comes from
        // `World::drain_dirty_sections`' `HashSet` and therefore from `RandomState`. Two runs of
        // the same capture drew the same scene from differently-grouped upload batches, so
        // per-frame draw and bind counts differed. Breaking ties on the position itself makes the
        // comparator total and the batch a pure function of (queue contents, camera).
        // Found by T0.1's replay determinism harness.
        if take < self.pending.len() {
            self.pending
                .select_nth_unstable_by(take - 1, |a, b| key(a).total_cmp(&key(b)).then(a.cmp(b)));
        }
        let mut batch: Vec<ChunkSectionPos> = self.pending.drain(..take).collect();
        batch.sort_by(|a, b| key(a).total_cmp(&key(b)).then(a.cmp(b)));
        for pos in &batch {
            self.queued.remove(pos);
        }
        batch
    }

    /// Remove all queued sections whose origins are within Java's unbudgeted ±16-block window.
    ///
    /// `RenderGlobal.isPositionInRenderChunk` compares a section-aligned integer feet origin with
    /// the candidate section origin on each axis. The terrain-only client seam has the interpolated
    /// camera position, so [`position_in_near_window`] floors that proxy to the same section grid;
    /// first-person horizontal coordinates are identical to feet and eye/F5 displacement remains
    /// the unavoidable approximation until the client passes the raw view-entity feet position.
    fn take_near(&mut self, camera: DVec3) -> Vec<ChunkSectionPos> {
        self.promote_deferred_nearest(camera, MAX_PENDING);
        if self.pending.is_empty() {
            return Vec::new();
        }
        let mut near = Vec::new();
        let mut far = Vec::with_capacity(self.pending.len());
        for pos in self.pending.drain(..) {
            if position_in_near_window(pos, camera) {
                self.queued.remove(&pos);
                near.push(pos);
            } else {
                far.push(pos);
            }
        }
        self.pending = far;
        near.sort_by(|a, b| {
            distance_sq(*a, camera)
                .total_cmp(&distance_sq(*b, camera))
                .then(a.cmp(b))
        });
        near
    }

    /// Run one selected batch on the dedicated terrain workers.
    fn mesh_batch(&self, world: &World, batch: Vec<ChunkSectionPos>) -> Vec<ChunkMesh> {
        mesh_batch(world, batch, self.settings, self.allow_block_alternatives)
    }

    /// Mesh one section immediately on the calling thread, removing it from the queue.
    ///
    /// java: `RenderGlobal.setupTerrain`'s "build near" branch — the section containing the
    /// camera is never allowed to lag a frame behind.
    pub fn mesh_now(&mut self, world: &World, pos: ChunkSectionPos) -> ChunkMesh {
        self.cancel(pos);
        self.diagnostic_synchronous_built([pos]);
        mesh_section_with_alternatives(world, pos, self.settings, self.allow_block_alternatives)
    }

    /// Promote the nearest deferred sections into the priority window needed by a take.
    ///
    /// Deferred work is the Rust equivalent of the caller retaining a failed Java `offer` in
    /// `chunksToUpdate`. Choosing by the same camera distance used for [`Self::take_nearest`]
    /// prevents a large dirty burst from making a newly-visible near section wait behind the
    /// far end of the burst.
    fn promote_deferred_nearest(&mut self, camera: DVec3, needed: usize) -> usize {
        let target = needed.min(MAX_PENDING);
        let mut promoted = 0;
        while promoted < target {
            let Some(deferred_index) = self
                .deferred
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    distance_sq(**a, camera)
                        .total_cmp(&distance_sq(**b, camera))
                        .then(a.cmp(b))
                })
                .map(|(index, _)| index)
            else {
                break;
            };

            let Some(active_index) = self
                .pending
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    distance_sq(**a, camera)
                        .total_cmp(&distance_sq(**b, camera))
                        .then(a.cmp(b))
                })
                .map(|(index, _)| index)
            else {
                let pos = self.deferred.remove(deferred_index);
                self.deferred_set.remove(&pos);
                self.queued.insert(pos);
                self.pending.push(pos);
                self.diagnostic_promoted();
                promoted += 1;
                continue;
            };

            if self.pending.len() >= MAX_PENDING {
                let deferred_pos = self.deferred[deferred_index];
                let active_pos = self.pending[active_index];
                let order = distance_sq(deferred_pos, camera)
                    .total_cmp(&distance_sq(active_pos, camera))
                    .then(deferred_pos.cmp(&active_pos));
                if order != Ordering::Less {
                    break;
                }
                self.pending.swap_remove(active_index);
                self.queued.remove(&active_pos);
                self.deferred_set.insert(active_pos);
                self.deferred.push(active_pos);
            }

            let pos = self.deferred.remove(deferred_index);
            self.deferred_set.remove(&pos);
            self.queued.insert(pos);
            self.pending.push(pos);
            self.diagnostic_promoted();
            promoted += 1;
        }
        promoted
    }

    #[cfg(test)]
    fn total_retained(&self) -> usize {
        self.pending.len() + self.deferred.len()
    }
}

// -------------------------------------------------------------------------------------------
// S23: upload batching and the GPU memory budget
// -------------------------------------------------------------------------------------------

/// A cap on how many bytes of terrain vertex data may live on the GPU at once.
///
/// "Chunk mesh upload batching + GPU memory budget". Vanilla has no equivalent —
/// `RenderGlobal` sizes its `RenderChunk` array from the render distance and trusts that
/// `renderDistanceChunks` × 16 sections of display lists fit, which on 1.8.9 hardware they did.
/// A cap is worth having anyway: mesh size is a function of *content*, not of section count, and
/// a dense build can be an order of magnitude heavier per section than open terrain.
///
/// The default is deliberately generous. It is a **backstop against pathological content**, not
/// a working set target; a budget tight enough to evict during ordinary play would thrash.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuBudget {
    limit_bytes: u64,
}

impl GpuBudget {
    /// 512 MiB. At the 20-byte [`TerrainVertex`](super::vertex::TerrainVertex) stride (T1.1;
    /// it was 28) that is roughly 6.7 million quads — several times what render distance 16 of
    /// dense terrain produces, so ordinary play never reaches it. The figure is worth revisiting
    /// now that the stride has moved.
    pub const DEFAULT_BYTES: u64 = 512 * 1024 * 1024;

    /// A budget of `limit_bytes`.
    #[must_use]
    pub fn new(limit_bytes: u64) -> Self {
        GpuBudget { limit_bytes }
    }

    /// The cap in bytes.
    #[must_use]
    pub fn limit(self) -> u64 {
        self.limit_bytes
    }
}

impl Default for GpuBudget {
    fn default() -> Self {
        GpuBudget::new(GpuBudget::DEFAULT_BYTES)
    }
}

/// What [`GpuResidency`] knows about the terrain working set, for the F3 overlay and for logs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GpuMemoryReport {
    /// Sections with vertex data resident.
    pub sections: usize,
    /// Bytes of vertex data resident.
    pub bytes: u64,
    /// The configured cap.
    pub limit: u64,
    /// Sections evicted since the accountant was created.
    pub evicted: u64,
}

impl GpuMemoryReport {
    /// Fraction of the budget in use, `0.0..` (can exceed 1.0 between an insert and the eviction
    /// pass that answers it).
    #[must_use]
    pub fn used_fraction(&self) -> f64 {
        if self.limit == 0 {
            return 0.0;
        }
        self.bytes as f64 / self.limit as f64
    }
}

/// [`GpuMemoryReport`] plus what happened *since the last time it was taken*.
///
/// Added, where the chunk pipeline never reaches steady state and three
/// explanations fit the same symptom. A cumulative report cannot tell them apart; a windowed one
/// can, in a single run:
///
/// | observation | conclusion |
/// |---|---|
/// | `peak_bytes` well under `limit`, `evicted == 0`, queue still busy | not the budget — a re-queue or dirty-set defect |
/// | `bytes` climbing window on window at a render distance whose working set is fixed | an accounting leak in [`GpuResidency`] |
/// | `peak_bytes` at `limit` with `evicted > 0` every window | a real capacity problem, which T1.1 attacks |
///
/// `peak_bytes` is the load-bearing field: `bytes` sampled once per window can sit under the cap
/// while the working set crosses it repeatedly inside the window, which is exactly the shape
/// eviction churn would have.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GpuMemoryWindow {
    /// Instantaneous state at the moment the window was taken.
    pub now: GpuMemoryReport,
    /// The high-water mark of resident bytes *within* the window.
    pub peak_bytes: u64,
    /// Sections evicted within the window.
    pub evicted: u64,
    /// Sections inserted or replaced within the window — the meshing pipeline's output rate, so
    /// "the queue is still busy" is a number in the same table rather than an inference.
    pub inserted: u64,
    /// Wall time the window covered, filled in by the caller that owns the clock.
    pub seconds: f64,
}

/// Tracks how much GPU vertex memory each resident section costs and decides what to drop.
///
/// This is pure bookkeeping — it holds no `wgpu` handles and frees nothing itself. The renderer
/// owns the buffers; this owns the *policy*, which is why it can be tested exhaustively without
/// a GPU.
///
/// Eviction is **farthest from the camera first**, which is the only ordering that is safe
/// without knowing what is on screen: a distant section is the one most likely to leave the view
/// frustum next, and re-meshing it later costs one queue entry.
#[derive(Debug, Clone, Default)]
pub struct GpuResidency {
    budget: GpuBudget,
    sections: HashMap<ChunkSectionPos, u64>,
    bytes: u64,
    evicted: u64,
    /// High-water mark of `bytes` since [`GpuResidency::take_window`] last reset it.
    window_peak: u64,
    /// Evictions since the last [`GpuResidency::take_window`].
    window_evicted: u64,
    /// Inserts since the last [`GpuResidency::take_window`].
    window_inserted: u64,
}

impl GpuResidency {
    /// An empty accountant with `budget`.
    #[must_use]
    pub fn new(budget: GpuBudget) -> Self {
        GpuResidency {
            budget,
            sections: HashMap::new(),
            bytes: 0,
            evicted: 0,
            window_peak: 0,
            window_evicted: 0,
            window_inserted: 0,
        }
    }

    /// Record that `pos` now costs `bytes` on the GPU, replacing any previous figure for it.
    pub fn insert(&mut self, pos: ChunkSectionPos, bytes: u64) {
        if let Some(old) = self.sections.insert(pos, bytes) {
            self.bytes -= old;
        }
        self.bytes += bytes;
        self.window_inserted += 1;
        self.window_peak = self.window_peak.max(self.bytes);
    }

    /// Record that `pos` is no longer resident.
    pub fn remove(&mut self, pos: ChunkSectionPos) {
        if let Some(old) = self.sections.remove(&pos) {
            self.bytes -= old;
        }
    }

    /// Make room for `sections` entries in total, so a world load does not
    /// rehash this map mid-load. Accounting is unchanged.
    pub fn reserve(&mut self, sections: usize) {
        self.sections
            .reserve(sections.saturating_sub(self.sections.len()));
    }

    /// Forget everything, keeping the budget.
    pub fn clear(&mut self) {
        self.sections.clear();
        self.bytes = 0;
    }

    /// Bytes currently resident.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Whether the working set is over budget.
    #[must_use]
    pub fn is_over_budget(&self) -> bool {
        self.bytes > self.budget.limit()
    }

    /// A snapshot for the debug overlay.
    #[must_use]
    pub fn report(&self) -> GpuMemoryReport {
        GpuMemoryReport {
            sections: self.sections.len(),
            bytes: self.bytes,
            limit: self.budget.limit(),
            evicted: self.evicted,
        }
    }

    /// Everything that happened since this was last called, and resets the window counters.
    ///
    /// See [`GpuMemoryWindow`] for what the three numbers are for. `seconds` is left at zero —
    /// the caller owns the clock.
    pub fn take_window(&mut self) -> GpuMemoryWindow {
        let w = GpuMemoryWindow {
            now: self.report(),
            // A window in which nothing moved still has a peak: the standing working set.
            peak_bytes: self.window_peak.max(self.bytes),
            evicted: self.window_evicted,
            inserted: self.window_inserted,
            seconds: 0.0,
        };
        self.window_peak = self.bytes;
        self.window_evicted = 0;
        self.window_inserted = 0;
        w
    }

    /// The sections to drop, farthest from `camera` first, to get back under budget — and
    /// forgets them, so the caller only has to free the buffers.
    ///
    /// Returns empty when already within budget, so calling it every frame is free.
    pub fn evict_to_budget(&mut self, camera: DVec3) -> Vec<ChunkSectionPos> {
        if !self.is_over_budget() {
            return Vec::new();
        }
        let mut by_distance: Vec<(ChunkSectionPos, u64)> =
            self.sections.iter().map(|(p, b)| (*p, *b)).collect();
        by_distance.sort_by(|a, b| {
            distance_sq(b.0, camera)
                .total_cmp(&distance_sq(a.0, camera))
                // Ties broken by position so the choice is deterministic; two sections at the
                // same distance must not depend on `HashMap` iteration order.
                .then_with(|| b.0.cmp(&a.0))
        });
        let mut dropped = Vec::new();
        for (pos, bytes) in by_distance {
            if !self.is_over_budget() {
                break;
            }
            self.sections.remove(&pos);
            self.bytes -= bytes;
            self.evicted += 1;
            self.window_evicted += 1;
            dropped.push(pos);
        }
        dropped
    }
}

/// One section's slice inside a batched upload buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UploadSlice {
    /// Which section this run of vertices belongs to.
    pub pos: ChunkSectionPos,
    /// Byte offset of its first vertex within [`UploadBatch::bytes`].
    pub offset: u64,
    /// Length in bytes.
    pub len: u64,
    /// Quads in this run (`len / (4 * VERTEX_SIZE)`).
    pub quads: u32,
}

/// Many sections' geometry for one render layer, concatenated into a single buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UploadBatch {
    /// Which of vanilla's four layers this batch is for.
    pub layer: RenderLayer,
    /// The concatenated vertex bytes, ready for one `create_buffer_init`/`write_buffer`.
    pub bytes: Vec<u8>,
    /// Where each section's vertices start. In the order they were laid down.
    pub slices: Vec<UploadSlice>,
}

impl UploadBatch {
    /// Total bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether this batch carries no geometry.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

/// A batched upload: at most one GPU buffer per render layer instead of one per section per
/// layer.
///
/// "Chunk mesh upload batching". Before it,
/// [`TerrainRenderer::upload`](super::phase::TerrainRenderer::upload) created a fresh
/// `wgpu::Buffer` for every non-empty layer of every section — up to four allocations per
/// section, and a whole render-distance-12 load is thousands of them. Concatenating per layer
/// turns that into four, and gives every section a byte offset into a shared buffer.
///
/// [`TerrainRenderer::upload_all`](super::phase::TerrainRenderer::upload_all) consumes this.
/// Measured on 324 real meshed sections: **486 `wgpu::Buffer`s → 4**, and on the client's actual
/// batch size (`DEFAULT_BUDGET` sections per frame) 486 → 44. The byte offset is also what let
/// the draw loop stop rebinding a vertex buffer per section — it passes the offset as
/// `draw_indexed`'s `base_vertex` instead — which took terrain's per-frame CPU share from
/// 0.084 ms to 0.055 ms at 112 draws. It is *not* enough for a draw-call **reduction**: merging
/// two sections into one `draw_indexed` needs the per-section offset inside the vertex stream,
/// not merely inside the buffer. See the S23 handoff for the measurement that decided against it.
///
/// This type is the **plan**, not the upload: it is pure data, so the packing rule can be
/// tested exhaustively with no GPU. The renderer consumes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UploadPlan {
    /// One entry per layer that had any geometry, in vanilla's ordinal order.
    pub batches: Vec<UploadBatch>,
    /// Sections that did not fit under the budget and were left for a later frame.
    pub deferred: Vec<ChunkSectionPos>,
    /// Total bytes across every batch.
    pub total_bytes: u64,
    /// How many separate GPU buffers the unbatched path would have created for the same input.
    /// The whole point of the batch is that `batches.len()` is smaller than this.
    pub buffers_unbatched: usize,
}

/// `wgpu::COPY_BUFFER_ALIGNMENT`. Every slice offset is a multiple of this, so a batch can be
/// bound with `buffer.slice(offset..)` on any backend.
const COPY_ALIGNMENT: u64 = 4;

impl UploadPlan {
    /// Group `meshes` into one batch per render layer, stopping once `budget` bytes are packed.
    ///
    /// Sections are taken in the order given, so a caller that already sorted them nearest-first
    /// (which [`MeshQueue::drain`] does) gets the near ones uploaded and the far ones
    /// [`deferred`](UploadPlan::deferred). A single section is never split across frames: it is
    /// packed whole or deferred whole, because half a section's geometry is a visible hole.
    #[must_use]
    pub fn plan(meshes: &[ChunkMesh], budget: GpuBudget) -> UploadPlan {
        // The stride is a multiple of 4 (asserted in `vertex.rs`), so per-section runs are
        // already aligned; the max() keeps that true if the vertex format ever changes.
        debug_assert_eq!(
            (4 * VERTEX_SIZE as u64) % COPY_ALIGNMENT,
            0,
            "a quad's byte run must stay copy-aligned"
        );
        let mut batches: Vec<UploadBatch> = Vec::new();
        let mut deferred = Vec::new();
        let mut total: u64 = 0;
        let mut unbatched = 0usize;

        // Size each layer's byte vector before anything is copied into it. Concatenation is the
        // one thing batching *adds* to the upload path — the per-section path hands each mesh's
        // bytes straight to `create_buffer_init` — so letting a 7 MiB `Vec` grow from empty,
        // recopying everything written so far a dozen times, would be paying for the batch twice
        // over. Measured: it is the difference between the batched upload of 100 real sections
        // costing 4.8 ms and costing 2.7 ms. The bound is the whole input, which over-reserves
        // when the budget defers something; that costs address space, not copies.
        let mut per_layer = [0usize; 4];
        for mesh in meshes {
            for (layer, buffer) in mesh.layers() {
                per_layer[layer_ordinal(layer) as usize] += buffer.byte_size();
            }
        }

        for mesh in meshes {
            let cost = mesh.byte_size() as u64;
            if cost == 0 {
                // An all-air section still has a visibility matrix the culling walk reads, but no
                // bytes; it costs nothing and is never deferred.
                continue;
            }
            if total + cost > budget.limit() {
                deferred.push(mesh.pos);
                continue;
            }
            total += cost;
            for (layer, buffer) in mesh.layers() {
                if buffer.is_empty() {
                    continue;
                }
                unbatched += 1;
                let batch = match batches.iter_mut().position(|b| b.layer == layer) {
                    Some(i) => &mut batches[i],
                    None => {
                        batches.push(UploadBatch {
                            layer,
                            bytes: Vec::with_capacity(per_layer[layer_ordinal(layer) as usize]),
                            slices: Vec::new(),
                        });
                        batches.last_mut().expect("just pushed")
                    }
                };
                let offset = batch.bytes.len() as u64;
                let data = buffer.as_bytes();
                batch.bytes.extend_from_slice(data);
                batch.slices.push(UploadSlice {
                    pos: mesh.pos,
                    offset,
                    len: data.len() as u64,
                    quads: buffer.quad_count() as u32,
                });
            }
        }
        // Vanilla's ordinal order, so the batches come out in the order the phases draw them.
        batches.sort_by_key(|b| layer_ordinal(b.layer));
        UploadPlan {
            batches,
            deferred,
            total_bytes: total,
            buffers_unbatched: unbatched,
        }
    }

    /// Every section this plan uploads, with its byte cost — what to feed [`GpuResidency`].
    pub fn residency_costs(&self) -> impl Iterator<Item = (ChunkSectionPos, u64)> + '_ {
        let mut costs: HashMap<ChunkSectionPos, u64> = HashMap::new();
        for batch in &self.batches {
            for slice in &batch.slices {
                *costs.entry(slice.pos).or_default() += slice.len;
            }
        }
        costs.into_iter()
    }
}

/// java: `EnumWorldBlockLayer`'s ordinal.
fn layer_ordinal(layer: RenderLayer) -> u8 {
    match layer {
        RenderLayer::Solid => 0,
        RenderLayer::CutoutMipped => 1,
        RenderLayer::Cutout => 2,
        RenderLayer::Translucent => 3,
    }
}

/// Squared distance from `camera` to a section's centre.
#[must_use]
pub fn distance_sq(pos: ChunkSectionPos, camera: DVec3) -> f32 {
    let o = pos.origin();
    let dx = f64::from(o.x) + 8.0 - camera.x;
    let dy = f64::from(o.y) + 8.0 - camera.y;
    let dz = f64::from(o.z) + 8.0 - camera.z;
    (dx * dx + dy * dy + dz * dz) as f32
}

/// Whether a section origin falls in the inclusive camera-neighbourhood window Java builds
/// synchronously in `RenderGlobal.setupTerrain`.
fn position_in_near_window(pos: ChunkSectionPos, camera: DVec3) -> bool {
    let origin = pos.origin();
    // Java first floors the interpolated feet position to a block and then rounds it down to a
    // section origin (`new BlockPos(d3, d4, d5)` followed by `floor_double(v / 16) * 16`).
    // Comparing raw camera coordinates would incorrectly drop the -16 section for a camera at
    // block coordinate 8, even though Java's blockpos origin is 0 and both +/-16 neighbours fit.
    let feet_origin = |coordinate: f64| (coordinate / 16.0).floor() * 16.0;
    (f64::from(origin.x) - feet_origin(camera.x)).abs() <= 16.0
        && (f64::from(origin.y) - feet_origin(camera.y)).abs() <= 16.0
        && (f64::from(origin.z) - feet_origin(camera.z)).abs() <= 16.0
}

/// Mesh one selected batch on the fixed two-worker terrain pool.
fn mesh_batch(
    world: &World,
    batch: Vec<ChunkSectionPos>,
    settings: MeshSettings,
    allow_block_alternatives: bool,
) -> Vec<ChunkMesh> {
    terrain_mesh_pool().install(|| {
        batch
            .into_par_iter()
            .map(|pos| {
                mesh_section_with_alternatives(world, pos, settings, allow_block_alternatives)
            })
            .collect()
    })
}

/// Mesh many sections in parallel with no queue in between — the bulk path a first chunk load
/// or a render-distance change takes.
#[must_use]
pub fn mesh_all(
    world: &World,
    sections: &[ChunkSectionPos],
    settings: MeshSettings,
) -> Vec<ChunkMesh> {
    mesh_all_with_alternatives(world, sections, settings, true)
}

/// [`mesh_all`] with an explicit `allowBlockAlternatives` value for bulk rebuild callers.
#[must_use]
pub fn mesh_all_with_alternatives(
    world: &World,
    sections: &[ChunkSectionPos],
    settings: MeshSettings,
    allow_block_alternatives: bool,
) -> Vec<ChunkMesh> {
    terrain_mesh_pool().install(|| {
        sections
            .par_iter()
            .map(|pos| {
                mesh_section_with_alternatives(world, *pos, settings, allow_block_alternatives)
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::fixture;
    use mc_math::BlockPos;

    /// Set in the child process of the negative control below.
    const ABORT_CHILD_ENV: &str = "MC_W21_BS_RAYON_ABORT_CHILD";

    /// A panicking mesh job no longer takes the process down: the handler
    /// counts it and the same worker goes on to run the next job.
    #[test]
    fn a_panicking_terrain_job_is_handled_and_the_worker_survives() {
        let pool = terrain_pool_builder(1).build().expect("pool");
        let before = mesh_worker_panics();
        pool.spawn(|| panic!("deliberate mesh job panic (test)"));
        let (done, finished) = mpsc::channel();
        pool.spawn(move || {
            let _ = done.send(std::thread::current().name().map(str::to_owned));
        });
        let name = finished
            .recv_timeout(Duration::from_secs(10))
            .expect("the worker ran the next job after the panic");
        assert_eq!(name.as_deref(), Some("Terrain Batcher 0"));
        assert!(
            mesh_worker_panics() > before,
            "the panic reached the handler"
        );
    }

    /// Negative control, in a child process because it kills its process: the same pool
    /// without the handler (the old configuration) aborts on the first panicking job.
    #[test]
    fn without_the_handler_a_panicking_job_aborts_the_process() {
        if std::env::var_os(ABORT_CHILD_ENV).is_some() {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool");
            pool.spawn(|| panic!("deliberate mesh job panic (abort child)"));
            std::thread::sleep(Duration::from_secs(10));
            return; // not reached: rayon aborts first
        }
        let output = std::process::Command::new(std::env::current_exe().expect("test exe"))
            .args([
                "--exact",
                "terrain::jobs::tests::without_the_handler_a_panicking_job_aborts_the_process",
                "--test-threads=1",
            ])
            .env(ABORT_CHILD_ENV, "1")
            .output()
            .expect("run the child");
        assert!(!output.status.success(), "the child must not survive");
        // A failed assertion exits with 101; an abort does not.
        assert_ne!(output.status.code(), Some(101), "{output:?}");
    }

    #[test]
    fn sparse_snapshot_retains_deferred_neighbours_and_uses_full_world_for_bursts() {
        let mut world = World::new();
        for x in 0..17 {
            for z in 0..17 {
                world.load_empty_chunk(mc_math::ChunkPos::new(x, z));
            }
        }
        let mut queue = MeshQueue::new();
        queue.enqueue(ChunkSectionPos::new(2, 4, 2));
        // Deferred entries can be promoted during dispatch and must already be covered.
        let deferred = ChunkSectionPos::new(12, 4, 12);
        queue.deferred.push(deferred);
        queue.deferred_set.insert(deferred);
        let snapshot = queue.dispatch_snapshot(&world);
        assert_eq!(snapshot.loaded_chunk_count(), 18);
        for centre in [2, 12] {
            for dx in -1..=1 {
                for dz in -1..=1 {
                    assert!(snapshot
                        .chunk(mc_math::ChunkPos::new(centre + dx, centre + dz))
                        .is_some());
                }
            }
        }
        for x in 3..10 {
            queue.enqueue(ChunkSectionPos::new(x, 4, 2));
        }
        assert_eq!(queue.dispatch_snapshot(&world).loaded_chunk_count(), 289);
        let mut small = World::new();
        for x in 0..5 {
            for z in 0..5 {
                small.load_empty_chunk(mc_math::ChunkPos::new(x, z));
            }
        }
        let mut one = MeshQueue::new();
        one.enqueue(ChunkSectionPos::new(2, 4, 2));
        assert_eq!(one.dispatch_snapshot(&small).loaded_chunk_count(), 25);
    }

    #[test]
    fn vanilla_frame_budget_preserves_integer_deadline_semantics() {
        // The three values exercise Java's 60-FPS floor, a normal configured cap, and the
        // literal 260 "unlimited" cap. These are the exact integer divisions in
        // EntityRenderer.updateCameraAndRender, not rounded duration guesses.
        let floor = MeshFrameBudget::vanilla(0, 260, 0);
        assert_eq!(floor.effective_fps(), 60);
        assert_eq!(floor.allotted_nanos(), 4_166_666);
        assert_eq!(floor.remaining_nanos(), 4_166_666);

        let capped = MeshFrameBudget::vanilla(240, 120, 0);
        assert_eq!(capped.effective_fps(), 120);
        assert_eq!(capped.allotted_nanos(), 2_083_333);

        let unlimited = MeshFrameBudget::vanilla(300, 260, 0);
        assert_eq!(unlimited.effective_fps(), 260);
        assert_eq!(unlimited.allotted_nanos(), 961_538);

        // A pre-render overrun cannot produce a negative duration. This is the branch a mutant
        // that uses ordinary subtraction would either wrap or expose as a huge deadline.
        let overrun = MeshFrameBudget::vanilla(240, 120, 2_083_334);
        assert_eq!(overrun.remaining_nanos(), 0);
    }

    #[test]
    fn terrain_budget_changes_with_framerate_and_spent_frame_time() {
        let low = MeshFrameBudget::vanilla(60, 260, 0);
        let high = MeshFrameBudget::vanilla(240, 260, 0);
        assert_ne!(low.remaining_nanos(), high.remaining_nanos());
        assert_eq!(low.remaining_nanos(), 4_166_666);
        assert_eq!(high.remaining_nanos(), 1_041_666);

        let half = MeshFrameBudget::vanilla(60, 260, 2_000_000);
        assert_eq!(half.remaining_nanos(), 2_166_666);
        assert!(half.remaining_nanos() < low.remaining_nanos());
    }

    #[test]
    fn near_sections_ignore_an_expired_budget_but_far_work_stays_queued() {
        let world = fixture::blank_world(0);
        let camera = DVec3::new(8.0, 8.0, 8.0);
        let near = ChunkSectionPos::new(0, 0, 0);
        let far = ChunkSectionPos::new(3, 0, 0);
        let mut queue = MeshQueue::new();
        queue.enqueue(far);
        queue.enqueue(near);

        // Java's build-near branch is intentionally unbudgeted. The elapsed value consumes the
        // whole 60-FPS slice, so a fixed-count or budget-only mutant leaves `near` behind.
        let done = queue.drain_for_frame_with_elapsed(&world, camera, 0, 260, 4_166_666);
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].pos, near);
        assert_eq!(queue.len(), 1);
        assert_eq!(queue.take_nearest(camera, 1), vec![far]);
    }

    #[test]
    fn near_window_rounds_camera_to_java_section_origin_before_comparing() {
        let world = fixture::blank_world(0);
        let camera = DVec3::new(8.0, 8.0, 8.0);
        let left = ChunkSectionPos::new(-1, 0, 0);
        let centre = ChunkSectionPos::new(0, 0, 0);
        let right = ChunkSectionPos::new(1, 0, 0);
        let far = ChunkSectionPos::new(2, 0, 0);
        let mut queue = MeshQueue::new();
        for pos in [far, right, left, centre] {
            queue.enqueue(pos);
        }

        // Java floors feet x=8 to block x=8 and then section origin x=0. Therefore origins
        // -16, 0 and +16 are all within 16 blocks; raw-coordinate comparison would wrongly omit
        // the left section because |-16 - 8| is 24.
        let done = queue.drain_for_frame_with_elapsed(&world, camera, 0, 260, 4_166_666);
        let mut positions: Vec<_> = done.into_iter().map(|mesh| mesh.pos).collect();
        positions.sort();
        assert_eq!(positions, vec![left, centre, right]);
        assert_eq!(queue.take_nearest(camera, 1), vec![far]);
    }

    #[test]
    fn full_pending_queue_dispatches_into_workers_even_without_near_sections() {
        let world = fixture::blank_world(0);
        let camera = DVec3::new(8.0, 8.0, 8.0);
        let mut queue = MeshQueue::new();
        for i in 0..MAX_PENDING {
            queue.enqueue(ChunkSectionPos::new(100 + i as i32, 0, 100));
        }

        // With the Java deadline already expired, updateChunks still offers its first later
        // compile task.  A queue-cap implementation that combines pending entries with worker
        // owned entries incorrectly sees 100/100 here and never dispatches anything.
        let done = queue.dispatch_for_frame_with_elapsed(
            &world,
            camera,
            0,
            260,
            MeshFrameBudget::vanilla(0, 260, 0).allotted_nanos(),
        );
        assert!(done.is_empty());
        assert_eq!(queue.len(), MAX_PENDING - TERRAIN_WORKER_COUNT);
        assert_eq!(queue.in_flight_len(), TERRAIN_WORKER_COUNT);
        queue.cancel_all();
    }

    #[test]
    fn the_dirty_set_drives_the_queue() {
        let mut w = fixture::blank_world(0);
        // Loading a chunk marks all sixteen of its sections dirty — and, The
        // sixteen of each of its eight neighbours too. java: `handleChunkData`'s
        // `markBlockRangeForRenderUpdate(x<<4, 0, z<<4, (x<<4)+15, 256, (z<<4)+15)`, which
        // `RenderGlobal` expands by one block on every axis before bucketing by 16. Column A's
        // faces on a shared boundary were built against "unloaded = air" and have to be redone
        // once B arrives.
        //
        // The loaded column's sixteen sections are stored (all air) and queue; the eight
        // unloaded neighbours' 128 positions have no storage, so they are answered with a blank
        // result at admission and take no queue slot (#2/JOBS-1).
        let mut q = MeshQueue::new();
        assert_eq!(q.enqueue_dirty(&mut w), 9 * 16);
        assert_eq!(q.len(), 16);
        assert_eq!(q.deferred_len(), 0);
        assert_eq!(q.in_flight_len(), 8 * 16);
        assert!(w.drain_dirty_sections().is_empty());

        // The bounded queue still retains overflow from the dirty set: nine loaded columns are
        // 144 stored sections, plus the sixteen unloaded columns around them.
        let mut wide = fixture::blank_world(1);
        let mut q_wide = MeshQueue::new();
        let admitted = q_wide.enqueue_dirty(&mut wide);
        assert!(admitted >= 25 * 16);
        assert_eq!(q_wide.len(), MAX_PENDING);
        assert_eq!(q_wide.deferred_len(), 9 * 16 - MAX_PENDING);
        assert_eq!(q_wide.in_flight_len(), admitted - 9 * 16);

        // A single block change re-dirties the section it is in, plus whatever the light
        // update touched. Placing the first opaque block in an all-air column moves
        // the heightmap from 0 to 21, and `Chunk.relightBlock` then rewrites the sky column all
        // the way down — `markBlocksDirtyVertical(x, z, 21, 0)`, which is sections 0 and 1.
        fixture::place(&mut w, BlockPos::new(1, 20, 1), "minecraft:stone");
        let mut q2 = MeshQueue::new();
        q2.enqueue_dirty(&mut w);
        assert_eq!(q2.len(), 2);
        assert!(q2.pending.contains(&ChunkSectionPos::new(0, 1, 0)));
        assert!(q2.pending.contains(&ChunkSectionPos::new(0, 0, 0)));
    }

    #[test]
    fn absent_sections_resolve_at_admission_and_stay_revision_gated() {
        let camera = DVec3::new(8.0, 8.0, 8.0);
        let mut w = fixture::blank_world(0);
        let mut q = MeshQueue::new();
        q.enqueue_dirty(&mut w);
        let east = ChunkSectionPos::new(1, 4, 0);
        // Not adjacent to the east column, so loading that column does not re-dirty it.
        let west = ChunkSectionPos::new(-1, 4, 0);
        assert!(section_is_absent(&w, east) && section_is_absent(&w, west));
        assert!(!q.is_empty(), "a ready absent result is outstanding work");

        // Cancelling supersedes a ready blank result exactly as it does a running task.
        q.cancel(west);

        // The east column arrives with storage before the blank result was polled: its new
        // dirty events queue real rebuilds and the older blank answers must not upload.
        fixture::load_blank_chunk(&mut w, mc_math::ChunkPos::new(1, 0));
        q.enqueue_dirty(&mut w);
        assert!(q.pending.contains(&east) || q.deferred.contains(&east));

        let ready = q.poll_completed();
        assert!(!ready.is_empty());
        assert!(ready.iter().all(MeshTaskResult::is_empty));
        assert!(ready
            .iter()
            .all(|r| r.pos() != west && r.pos().chunk_pos() != east.chunk_pos()));
        assert!(ready.iter().all(|r| section_is_absent(&w, r.pos())));
        assert!(ready
            .iter()
            .all(|r| q.revision(r.pos()) == Some(r.revision())));
        // Delivered once, in admission order.
        assert!(ready
            .windows(2)
            .all(|pair| pair[0].order() < pair[1].order()));
        assert!(q.poll_completed().is_empty());
        q.cancel_all();
        assert!(q.is_empty());

        // The synchronous seam hands the same blank results over outside the mesh budget.
        let mut w2 = fixture::blank_world(0);
        let mut q2 = MeshQueue::new();
        q2.enqueue_dirty(&mut w2);
        let blank = q2.drain(&w2, camera, 0);
        assert_eq!(blank.len(), 8 * 16);
        assert!(blank.iter().all(|mesh| *mesh == ChunkMesh::empty(mesh.pos)));
        assert_eq!(
            q2.len(),
            16,
            "the stored sections still wait for the budget"
        );
    }

    /// A run without frames (a replay export's first seek)
    /// re-dirties the same absent sections every tick. The blank results each event supersedes
    /// must not pile up, and what a drain then hands over must be exactly what the old queue
    /// handed over (same sections, same admission order).
    #[test]
    fn superseded_absent_results_do_not_pile_up_without_frames() {
        const TICKS: usize = 20_000;
        // A block at the east edge of the only loaded column: its dirty box reaches into the
        // absent column next to it every time.
        let edge = BlockPos::new(15, 64, 8);
        let run = |compact: bool| {
            let mut w = fixture::blank_world(0);
            let mut q = MeshQueue::new();
            if !compact {
                // The old behaviour, as the negative control.
                q.absent_compact_at = usize::MAX;
            }
            q.enqueue_dirty(&mut w);
            let mut peak = 0;
            for _ in 0..TICKS {
                w.mark_dirty_box(edge);
                q.enqueue_dirty(&mut w);
                peak = peak.max(q.absent_ready.len());
            }
            let handed: Vec<(ChunkSectionPos, u64)> = q
                .take_absent_ready()
                .iter()
                .map(|result| (result.pos(), result.order()))
                .collect();
            (peak, handed)
        };
        let (old_peak, old_handed) = run(false);
        let (new_peak, new_handed) = run(true);
        assert!(
            old_peak > TICKS,
            "the control: the old queue keeps every superseded result ({old_peak})"
        );
        assert!(
            new_peak <= 2 * ABSENT_COMPACT_MIN,
            "superseded blank results piled up: {new_peak}"
        );
        assert!(!new_handed.is_empty());
        assert_eq!(
            new_handed, old_handed,
            "a drain hands over exactly what it did before"
        );
    }

    #[test]
    fn enqueue_is_idempotent() {
        let mut q = MeshQueue::new();
        q.enqueue(ChunkSectionPos::new(1, 2, 3));
        q.enqueue(ChunkSectionPos::new(1, 2, 3));
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn bounded_queue_rejects_101st_offer_and_accepts_it_after_a_drain() {
        let mut q = MeshQueue::new();
        for i in 0..MAX_PENDING {
            assert!(q.enqueue(ChunkSectionPos::new(i as i32, 0, 0)));
        }
        let rejected = ChunkSectionPos::new(MAX_PENDING as i32, 0, 0);
        assert!(!q.enqueue(rejected));
        assert_eq!(q.len(), MAX_PENDING);
        assert_eq!(q.deferred_len(), 1);

        let drained = q.take_nearest(DVec3::ZERO, 1);
        assert_eq!(drained.len(), 1);
        assert_eq!(q.len(), MAX_PENDING - 1);
        assert!(q.enqueue(rejected));
        assert_eq!(q.len(), MAX_PENDING);
        assert_eq!(q.deferred_len(), 0);
    }

    #[test]
    fn pending_cap_excludes_worker_owned_tasks_like_java() {
        let mut q = MeshQueue::new();
        for i in 0..MAX_PENDING {
            assert!(q.enqueue(ChunkSectionPos::new(i as i32, 0, 0)));
        }

        // Java's ArrayBlockingQueue counts only tasks still waiting in the dispatcher queue;
        // a worker has already removed its task before it starts compiling.  Model one such
        // worker-owned task, free one pending slot, and verify a new offer succeeds rather than
        // being deferred behind an (incorrectly combined) pending+in-flight cap.
        let running = ChunkSectionPos::new(-1, 0, 0);
        let revision = next_mesh_revision();
        q.revisions.insert(running, revision);
        q.in_flight.insert(
            running,
            InFlightMeshTask {
                revision,
                cancelled: Arc::new(AtomicBool::new(false)),
            },
        );
        assert_eq!(q.take_nearest(DVec3::ZERO, 1).len(), 1);

        let offered = ChunkSectionPos::new(1_000, 0, 0);
        assert!(q.enqueue(offered));
        assert_eq!(q.len(), MAX_PENDING);
        assert_eq!(q.deferred_len(), 0);
        assert_eq!(q.in_flight_len(), 1);
    }

    #[test]
    fn cancelled_worker_result_is_rejected_after_finish_race() {
        let pos = ChunkSectionPos::new(4, 2, -3);
        let mut q = MeshQueue::new();
        let revision = next_mesh_revision();
        q.revisions.insert(pos, revision);
        q.in_flight.insert(
            pos,
            InFlightMeshTask {
                revision,
                cancelled: Arc::new(AtomicBool::new(false)),
            },
        );

        // Java's finish runnable removes a task from queueChunkUpdates, but a worker may already
        // have crossed the take() boundary. The completion must remain unuploadable after that
        // race; the Rust tombstone/revision gate is the equivalent boundary.
        q.result_tx
            .send(MeshTaskResult::new(
                ChunkMesh::empty(pos),
                revision,
                Duration::ZERO,
                DEFAULT_SLOW_MESH_DEADLINE,
                1,
            ))
            .expect("test result receiver remains live");
        assert!(q.cancel(pos));
        assert!(q.revision(pos).is_some_and(|current| current > revision));
        assert!(q.poll_completed().is_empty());
        assert_eq!(q.in_flight_len(), 0);
    }

    #[test]
    fn duplicate_offer_does_not_consume_capacity() {
        let mut q = MeshQueue::new();
        let pos = ChunkSectionPos::new(7, 2, -3);
        assert!(q.enqueue(pos));
        assert!(q.enqueue(pos));
        assert_eq!(q.len(), 1);

        for i in 0..(MAX_PENDING - 1) {
            assert!(q.enqueue(ChunkSectionPos::new(i as i32, 4, 1)));
        }
        assert_eq!(q.len(), MAX_PENDING);
        assert!(q.enqueue(pos));
        assert_eq!(q.len(), MAX_PENDING);
    }

    #[test]
    fn cancellation_frees_a_slot_and_removes_deferred_work() {
        let mut q = MeshQueue::new();
        for i in 0..MAX_PENDING {
            assert!(q.enqueue(ChunkSectionPos::new(i as i32, 0, 0)));
        }
        let active = ChunkSectionPos::new(12, 0, 0);
        assert!(q.cancel(active));
        assert_eq!(q.len(), MAX_PENDING - 1);
        assert!(!q.cancel(active));
        assert!(q.enqueue(ChunkSectionPos::new(500, 0, 0)));

        let deferred = ChunkSectionPos::new(900, 0, 0);
        q.deferred.push(deferred);
        q.deferred_set.insert(deferred);
        assert!(q.cancel(deferred));
        assert_eq!(q.deferred_len(), 0);
        assert!(!q.is_empty());
    }

    #[test]
    fn reset_world_replaces_work_preserves_options_and_keeps_diagnostic_frame() {
        let old_pos = ChunkSectionPos::new(2, 3, 4);
        let new_pos = ChunkSectionPos::new(-5, 1, 6);
        let settings = MeshSettings {
            fancy_graphics: false,
            ambient_occlusion: false,
            ..MeshSettings::default()
        };
        let mut queue = MeshQueue::new();
        queue.set_settings(settings);
        queue.set_allow_block_alternatives(false);
        queue.set_slow_mesh_deadline(Duration::from_millis(3));
        queue.set_diagnostics_enabled(true);
        queue
            .begin_diagnostic_frame(10)
            .expect("first diagnostic frame");
        assert!(queue.enqueue(old_pos));
        let old_revision = queue.revision(old_pos).expect("old revision");
        let old_sender = queue.result_tx.clone();

        queue.reset_world();

        assert_eq!(queue.settings(), settings);
        assert!(!queue.allow_block_alternatives());
        assert_eq!(queue.slow_mesh_deadline(), Duration::from_millis(3));
        assert!(queue.is_empty());
        assert!(queue.revision(old_pos).is_none());
        assert!(
            old_sender
                .send(MeshTaskResult::new(
                    ChunkMesh::empty(old_pos),
                    old_revision,
                    Duration::ZERO,
                    DEFAULT_SLOW_MESH_DEADLINE,
                    0,
                ))
                .is_err(),
            "reset must drop the old receiver so an old worker cannot reinsert a result"
        );

        // The cancellation and post-reset admission share the same logical frame. The old age
        // is gone, the new age starts at zero, and both admissions remain in this frame's count.
        assert!(queue.enqueue(new_pos));
        let stats = queue
            .diagnostic_stats()
            .expect("diagnostics remain enabled");
        assert_eq!(stats.enqueued, 2);
        assert_eq!(stats.cancelled, 1);
        assert_eq!(stats.active, 1);
        assert_eq!(stats.oldest_active_age_frames, Some(0));
        assert_eq!(stats.oldest_deferred_age_frames, None);
        assert_eq!(stats.oldest_in_flight_age_frames, None);

        assert!(matches!(
            queue.begin_diagnostic_frame(10),
            Err(MeshQueueDiagnosticsError::NonMonotonicFrameIndex {
                previous: 10,
                current: 10
            })
        ));
        queue
            .begin_diagnostic_frame(11)
            .expect("next diagnostic frame");
        let next = queue
            .diagnostic_stats()
            .expect("diagnostics remain enabled");
        assert_eq!(next.enqueued, 0);
        assert_eq!(next.cancelled, 0);
        assert_eq!(next.oldest_active_age_frames, Some(1));

        let mut ordinary = MeshQueue::new();
        ordinary.enqueue(old_pos);
        ordinary.reset_world();
        assert!(ordinary.diagnostic_stats().is_none());
        assert!(ordinary.is_empty());
    }

    #[test]
    fn block_alternatives_default_on_and_can_be_disabled() {
        let mut q = MeshQueue::new();
        assert!(
            q.allow_block_alternatives(),
            "vanilla's option defaults to true"
        );
        q.set_allow_block_alternatives(false);
        assert!(!q.allow_block_alternatives());
        q.set_allow_block_alternatives(true);
        assert!(q.allow_block_alternatives());
    }

    #[test]
    fn drain_respects_the_budget_and_meshes_nearest_first() {
        let mut w = fixture::blank_world(1);
        fixture::place(&mut w, BlockPos::new(1, 20, 1), "minecraft:stone");
        let mut q = MeshQueue::new();
        for y in 0..16 {
            q.enqueue(ChunkSectionPos::new(0, y, 0));
        }
        let camera = DVec3::new(8.0, 8.0, 8.0);
        let done = q.drain(&w, camera, 4);
        assert_eq!(done.len(), 4);
        assert_eq!(q.len(), 12);
        // Section y=0 contains the camera and must be first.
        assert_eq!(done[0].pos, ChunkSectionPos::new(0, 0, 0));
    }

    #[test]
    fn parallel_meshing_matches_the_serial_result_exactly() {
        // The whole point of `mesh_section` being pure: rayon must not change the output.
        let mut w = fixture::blank_world(1);
        let stone = fixture::state_of("minecraft:stone");
        fixture::fill(
            &mut w,
            BlockPos::new(0, 16, 0),
            BlockPos::new(15, 20, 15),
            stone,
        );
        let sections: Vec<_> = (0..4).map(|y| ChunkSectionPos::new(0, y, 0)).collect();
        let parallel = mesh_all(&w, &sections, MeshSettings::default());
        let serial: Vec<_> = sections
            .iter()
            .map(|p| mesh_section_with(&w, *p, MeshSettings::default()))
            .collect();
        assert_eq!(parallel, serial);
    }

    #[test]
    fn mesh_now_removes_the_section_from_the_queue() {
        let w = fixture::blank_world(0);
        let mut q = MeshQueue::new();
        let pos = ChunkSectionPos::new(0, 3, 0);
        q.enqueue(pos);
        q.enqueue(ChunkSectionPos::new(0, 4, 0));
        let mesh = q.mesh_now(&w, pos);
        assert_eq!(mesh.pos, pos);
        assert_eq!(q.len(), 1);
        // Re-enqueueing it must work: the membership index has to have been cleared too, or the
        // section would be silently un-rebuildable for the rest of the session.
        q.enqueue(pos);
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn queue_settings_reach_fluid_culling_on_both_mesh_paths() {
        // A water side against a leaf is the narrowest settings-sensitive fluid case: fancy
        // leaves are non-opaque, while fast leaves are opaque.  The visible side is emitted
        // double-sided, so Fancy has exactly eight more translucent vertices.  Exercise both
        // the rayon queue and the synchronous near-camera path; both must use `self.settings`.
        let mut world = fixture::blank_world(0);
        let pos = ChunkSectionPos::new(0, 1, 0);
        fixture::place(&mut world, BlockPos::new(1, 20, 1), "minecraft:water");
        fixture::place(&mut world, BlockPos::new(2, 20, 1), "minecraft:leaves");
        let fancy = MeshSettings {
            fancy_graphics: true,
            ambient_occlusion: true,
            ..MeshSettings::default()
        };
        let fast = MeshSettings {
            fancy_graphics: false,
            ambient_occlusion: true,
            ..MeshSettings::default()
        };

        let queued = |settings| {
            let mut queue = MeshQueue::new();
            queue.set_settings(settings);
            queue.enqueue(pos);
            queue
                .drain(&world, DVec3::new(8.0, 24.0, 8.0), 1)
                .pop()
                .expect("the enqueued section must be meshed")
        };
        let immediate = |settings| {
            let mut queue = MeshQueue::new();
            queue.set_settings(settings);
            queue.mesh_now(&world, pos)
        };

        let fancy_queued = queued(fancy).translucent.vertices().len();
        let fast_queued = queued(fast).translucent.vertices().len();
        let fancy_immediate = immediate(fancy).translucent.vertices().len();
        let fast_immediate = immediate(fast).translucent.vertices().len();
        assert_eq!(fancy_queued, fast_queued + 8);
        assert_eq!(fancy_immediate, fast_immediate + 8);
        assert_eq!(fancy_queued, fancy_immediate);
        assert_eq!(fast_queued, fast_immediate);
    }

    // ---- S23: queue cost, batching, budget -------------------------------------------------

    /// The reference implementation the queue used before S23: a linear membership scan and a
    /// full sort. Kept in the tests so the replacements can be checked against it for *behaviour*
    /// — the point of the change was cost, and a cheaper algorithm that answers differently is
    /// not an optimisation.
    fn reference_take_nearest(
        pending: &mut Vec<ChunkSectionPos>,
        camera: DVec3,
        budget: usize,
    ) -> Vec<ChunkSectionPos> {
        pending.sort_by(|a, b| distance_sq(*a, camera).total_cmp(&distance_sq(*b, camera)));
        let take = budget.min(pending.len());
        pending.drain(..take).collect()
    }

    #[test]
    fn the_worker_pool_scales_with_the_cpu_between_vanillas_two_and_four() {
        let sizes = [1, 2, 4, 8, 11, 12, 15, 16, 32, 128].map(default_pool_size);
        assert_eq!(sizes, [2, 2, 2, 2, 2, 3, 3, 4, 4, 4]);
    }

    #[test]
    fn take_nearest_returns_what_the_old_full_sort_returned() {
        let camera = DVec3::new(3.5, 40.0, -7.25);
        let positions: Vec<ChunkSectionPos> = (0..500i32)
            .map(|i| ChunkSectionPos::new(i % 17 - 8, (i / 17) % 16, i % 23 - 11))
            .collect();
        for budget in [0usize, 1, 2, 7, 32, 499, 500, 501] {
            let mut q = MeshQueue::new();
            for p in &positions {
                q.enqueue(*p);
            }
            let mine = q.take_nearest(camera, budget);
            let mut reference = positions.clone();
            reference.dedup();
            let mut reference: Vec<ChunkSectionPos> = {
                let mut seen = HashSet::new();
                positions
                    .iter()
                    .copied()
                    .filter(|p| seen.insert(*p))
                    .collect()
            };
            let expected = reference_take_nearest(&mut reference, camera, budget.min(MAX_PENDING));
            // Distances, not identities: two sections equidistant from the camera are
            // interchangeable and `select_nth_unstable_by` may pick either. What must match is
            // that no farther section is taken before a nearer one.
            let d = |v: &[ChunkSectionPos]| -> Vec<f32> {
                v.iter().map(|p| distance_sq(*p, camera)).collect()
            };
            assert_eq!(d(&mine), d(&expected), "budget {budget}");
            assert_eq!(q.total_retained(), reference.len(), "budget {budget}");
        }
    }

    #[test]
    fn the_membership_index_matches_the_queue() {
        // `queued` and `pending` are two representations of one set; if they drift, a section
        // either never rebuilds or rebuilds forever.
        let mut q = MeshQueue::new();
        let w = fixture::blank_world(0);
        for i in 0..50i32 {
            q.enqueue(ChunkSectionPos::new(i % 5, i % 16, i % 3));
        }
        assert_eq!(q.pending.len(), q.queued.len());
        let taken = q.take_nearest(DVec3::new(0.0, 0.0, 0.0), 7);
        assert_eq!(taken.len(), 7);
        assert_eq!(q.pending.len(), q.queued.len());
        for p in &taken {
            assert!(!q.queued.contains(p));
        }
        q.mesh_now(&w, q.pending[0]);
        assert_eq!(q.pending.len(), q.queued.len());
        // Everything still queued can be re-enqueued without growing the queue.
        let before = q.len();
        let all: Vec<_> = q.pending.clone();
        for p in all {
            q.enqueue(p);
        }
        assert_eq!(q.len(), before);
    }

    fn mesh_of(quads_per_layer: [usize; 4], pos: ChunkSectionPos) -> ChunkMesh {
        use super::super::vertex::TerrainVertex;
        let mut mesh = ChunkMesh::empty(pos);
        for (i, count) in quads_per_layer.iter().enumerate() {
            let buffer = match i {
                0 => &mut mesh.solid,
                1 => &mut mesh.cutout_mipped,
                2 => &mut mesh.cutout,
                _ => &mut mesh.translucent,
            };
            for q in 0..*count {
                let v = TerrainVertex::new(
                    [q as f32, i as f32, pos.y as f32],
                    [0.0; 2],
                    [0; 4],
                    [0; 4],
                );
                buffer.push_quad([v; 4]);
            }
        }
        mesh
    }

    #[test]
    fn an_upload_plan_is_one_buffer_per_layer_and_loses_nothing() {
        let meshes = vec![
            mesh_of([3, 0, 1, 2], ChunkSectionPos::new(0, 0, 0)),
            mesh_of([5, 2, 0, 0], ChunkSectionPos::new(1, 0, 0)),
            mesh_of([0, 0, 0, 4], ChunkSectionPos::new(2, 0, 0)),
        ];
        let plan = UploadPlan::plan(&meshes, GpuBudget::new(u64::MAX));

        // Today: one buffer per non-empty layer per section.
        assert_eq!(plan.buffers_unbatched, 3 + 2 + 1);
        // Batched: one per layer that had anything, in vanilla's ordinal order.
        assert_eq!(
            plan.batches.iter().map(|b| b.layer).collect::<Vec<_>>(),
            vec![
                RenderLayer::Solid,
                RenderLayer::CutoutMipped,
                RenderLayer::Cutout,
                RenderLayer::Translucent
            ]
        );
        assert_eq!(plan.batches.len(), 4);
        assert!(plan.deferred.is_empty());

        // Nothing is lost or duplicated: every section's bytes appear exactly once, at the
        // offset the plan claims, and the total matches the meshes' own accounting.
        let total: u64 = meshes.iter().map(|m| m.byte_size() as u64).sum();
        assert_eq!(plan.total_bytes, total);
        for batch in &plan.batches {
            let mut expected_offset = 0u64;
            for slice in &batch.slices {
                assert_eq!(slice.offset, expected_offset);
                let mesh = meshes.iter().find(|m| m.pos == slice.pos).expect("section");
                let source = mesh
                    .layers()
                    .find(|(l, _)| *l == batch.layer)
                    .expect("layer")
                    .1;
                assert_eq!(slice.len as usize, source.as_bytes().len());
                assert_eq!(slice.quads, source.quad_count() as u32);
                let start = slice.offset as usize;
                let end = start + slice.len as usize;
                assert_eq!(&batch.bytes[start..end], source.as_bytes());
                expected_offset = end as u64;
            }
            assert_eq!(batch.bytes.len() as u64, expected_offset);
            // Every slice must be bindable: `buffer.slice(offset..)` requires 4-byte alignment.
            assert!(batch.slices.iter().all(|s| s.offset % 4 == 0));
        }
    }

    #[test]
    fn an_empty_section_costs_no_buffer_at_all() {
        let meshes = vec![
            ChunkMesh::empty(ChunkSectionPos::new(0, 9, 0)),
            mesh_of([1, 0, 0, 0], ChunkSectionPos::new(0, 1, 0)),
        ];
        let plan = UploadPlan::plan(&meshes, GpuBudget::new(u64::MAX));
        assert_eq!(plan.batches.len(), 1);
        assert_eq!(plan.buffers_unbatched, 1);
        assert!(plan.deferred.is_empty(), "an air section is never deferred");
    }

    #[test]
    fn the_budget_defers_whole_sections_in_the_order_given() {
        let one = mesh_of([10, 0, 0, 0], ChunkSectionPos::new(0, 0, 0)).byte_size() as u64;
        let meshes = vec![
            mesh_of([10, 0, 0, 0], ChunkSectionPos::new(0, 0, 0)),
            mesh_of([10, 0, 0, 0], ChunkSectionPos::new(1, 0, 0)),
            mesh_of([10, 0, 0, 0], ChunkSectionPos::new(2, 0, 0)),
        ];
        let plan = UploadPlan::plan(&meshes, GpuBudget::new(one * 2));
        assert_eq!(plan.total_bytes, one * 2);
        assert_eq!(plan.deferred, vec![ChunkSectionPos::new(2, 0, 0)]);
        // A deferred section contributes nothing at all — not a partial run.
        for batch in &plan.batches {
            assert!(batch.slices.iter().all(|s| s.pos.x != 2));
        }
        // Control: with room, none is deferred and all three land.
        let roomy = UploadPlan::plan(&meshes, GpuBudget::new(one * 3));
        assert!(roomy.deferred.is_empty());
        assert_eq!(roomy.total_bytes, one * 3);
    }

    #[test]
    fn residency_costs_add_up_to_the_plans_total() {
        let meshes = vec![
            mesh_of([3, 1, 0, 2], ChunkSectionPos::new(0, 0, 0)),
            mesh_of([1, 0, 4, 0], ChunkSectionPos::new(0, 1, 0)),
        ];
        let plan = UploadPlan::plan(&meshes, GpuBudget::new(u64::MAX));
        let costs: HashMap<ChunkSectionPos, u64> = plan.residency_costs().collect();
        assert_eq!(costs.len(), 2);
        assert_eq!(costs.values().sum::<u64>(), plan.total_bytes);
        for mesh in &meshes {
            assert_eq!(costs[&mesh.pos], mesh.byte_size() as u64);
        }
    }

    #[test]
    fn residency_evicts_the_farthest_sections_and_only_when_over_budget() {
        let camera = DVec3::new(0.0, 0.0, 0.0);
        let mut r = GpuResidency::new(GpuBudget::new(300));
        for x in 0..4i32 {
            r.insert(ChunkSectionPos::new(x, 0, 0), 100);
        }
        assert_eq!(r.bytes(), 400);
        assert!(r.is_over_budget());
        let dropped = r.evict_to_budget(camera);
        assert_eq!(dropped, vec![ChunkSectionPos::new(3, 0, 0)]);
        assert_eq!(r.bytes(), 300);
        assert!(!r.is_over_budget());

        // Control: within budget it evicts nothing, so the eviction above was the budget doing
        // its job rather than the accountant dropping things unconditionally.
        assert!(r.evict_to_budget(camera).is_empty());
        assert_eq!(r.report().sections, 3);
        assert_eq!(r.report().evicted, 1);
        assert!((r.report().used_fraction() - 1.0).abs() < 1e-9);

        // Replacing a section's cost updates the total rather than adding to it.
        r.insert(ChunkSectionPos::new(0, 0, 0), 10);
        assert_eq!(r.bytes(), 210);
        r.remove(ChunkSectionPos::new(0, 0, 0));
        assert_eq!(r.bytes(), 200);
        assert_eq!(r.report().sections, 2);
    }

    #[test]
    fn the_memory_window_separates_a_busy_queue_from_a_biting_budget() {
        // The instrument the churn diagnosis needs: a cumulative report cannot tell "the budget
        // evicts every window" from "the budget never fires and the queue refills anyway". This
        // asserts the windowed one can, by producing both shapes and showing the numbers differ.
        let camera = DVec3::new(0.0, 0.0, 0.0);

        // Shape 1: busy but well under the cap. Uploads happen every window, nothing is evicted,
        // and the peak stays far below the limit.
        let mut r = GpuResidency::new(GpuBudget::new(10_000));
        for x in 0..5i32 {
            r.insert(ChunkSectionPos::new(x, 0, 0), 100);
        }
        assert!(r.evict_to_budget(camera).is_empty());
        let w = r.take_window();
        assert_eq!(w.inserted, 5, "the queue was busy");
        assert_eq!(w.evicted, 0, "and the budget never fired");
        assert_eq!(w.peak_bytes, 500);
        assert_eq!(w.now.bytes, 500);
        assert!(w.now.used_fraction() < 0.1);

        // The window resets: a quiet window reports no traffic, but still reports the standing
        // working set as its peak rather than zero.
        let quiet = r.take_window();
        assert_eq!((quiet.inserted, quiet.evicted), (0, 0));
        assert_eq!(quiet.peak_bytes, 500);
        assert_eq!(quiet.now.sections, 5);

        // Shape 2: the budget bites. Same traffic, a cap that cannot hold it.
        let mut r = GpuResidency::new(GpuBudget::new(250));
        for x in 0..5i32 {
            r.insert(ChunkSectionPos::new(x, 0, 0), 100);
        }
        let dropped = r.evict_to_budget(camera);
        assert!(!dropped.is_empty());
        let w = r.take_window();
        assert_eq!(w.inserted, 5);
        assert_eq!(w.evicted, dropped.len() as u64);
        assert!(w.evicted > 0, "the budget must have fired");
        // And `peak_bytes` is the whole point: `bytes` sampled now is under the cap, but the
        // working set crossed it inside the window, which is what a sampled figure would hide.
        assert!(
            w.now.bytes <= 250,
            "sampled after eviction: {}",
            w.now.bytes
        );
        assert_eq!(
            w.peak_bytes, 500,
            "the peak inside the window is not hidden"
        );
    }

    #[test]
    fn eviction_is_deterministic_for_equidistant_sections() {
        // Two sections the same distance away must not be chosen between by `HashMap` iteration
        // order, or the same frame produces different results on different runs.
        let camera = DVec3::new(8.0, 8.0, 8.0);
        let mut first = None;
        for _ in 0..8 {
            let mut r = GpuResidency::new(GpuBudget::new(100));
            for (x, z) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                r.insert(ChunkSectionPos::new(x, 0, z), 50);
            }
            let dropped = r.evict_to_budget(camera);
            match &first {
                None => first = Some(dropped),
                Some(expected) => assert_eq!(&dropped, expected),
            }
        }
    }

    #[test]
    fn a_zero_budget_defers_everything_and_never_panics() {
        let meshes = vec![mesh_of([4, 0, 0, 0], ChunkSectionPos::new(0, 0, 0))];
        let plan = UploadPlan::plan(&meshes, GpuBudget::new(0));
        assert_eq!(plan.total_bytes, 0);
        assert!(plan.batches.is_empty());
        assert_eq!(plan.deferred.len(), 1);
        let mut r = GpuResidency::new(GpuBudget::new(0));
        r.insert(ChunkSectionPos::new(0, 0, 0), 1);
        assert_eq!(r.evict_to_budget(DVec3::new(0.0, 0.0, 0.0)).len(), 1);
        assert_eq!(r.report().used_fraction(), 0.0);
    }

    #[test]
    fn queue_diagnostics_track_admission_overflow_and_known_age() {
        let mut queue = MeshQueue::new();
        queue.set_diagnostics_enabled(true);
        queue.begin_diagnostic_frame(10).unwrap();
        for x in 0..MAX_PENDING as i32 {
            assert!(queue.enqueue(ChunkSectionPos::new(x, 0, 0)));
        }
        let deferred = ChunkSectionPos::new(MAX_PENDING as i32, 0, 0);
        assert!(!queue.enqueue(deferred));

        let stats = queue.diagnostic_stats().unwrap();
        assert_eq!(stats.enqueued, (MAX_PENDING + 1) as u64);
        assert_eq!(stats.enqueue_overflow, 1);
        assert_eq!(stats.deferred_promoted, 0);
        assert_eq!(stats.oldest_active_age_frames, Some(0));
        assert_eq!(stats.oldest_deferred_age_frames, Some(0));

        queue.begin_diagnostic_frame(12).unwrap();
        assert!(queue.cancel(ChunkSectionPos::new(0, 0, 0)));
        let stats = queue.diagnostic_stats().unwrap();
        assert_eq!(stats.deferred_promoted, 1);
        assert_eq!(stats.cancelled, 1);
        assert_eq!(stats.oldest_deferred_age_frames, None);
        assert_eq!(stats.oldest_active_age_frames, Some(2));
    }

    #[test]
    fn duplicate_admission_does_not_reset_age_or_consume_counter() {
        let pos = ChunkSectionPos::new(3, 0, 0);
        let mut queue = MeshQueue::new();
        queue.set_diagnostics_enabled(true);
        queue.begin_diagnostic_frame(10).unwrap();
        assert!(queue.enqueue(pos));
        assert!(queue.enqueue(pos));
        assert_eq!(queue.diagnostic_stats().unwrap().enqueued, 1);

        queue.begin_diagnostic_frame(13).unwrap();
        assert!(queue.enqueue(pos));
        let stats = queue.diagnostic_stats().unwrap();
        assert_eq!(stats.enqueued, 0);
        assert_eq!(stats.oldest_active_age_frames, Some(3));
    }

    #[test]
    fn superseding_in_flight_work_counts_cancel_and_fresh_admission() {
        let pos = ChunkSectionPos::new(5, 1, 0);
        let mut queue = MeshQueue::new();
        queue.set_diagnostics_enabled(true);
        queue.begin_diagnostic_frame(20).unwrap();
        assert!(queue.enqueue(pos));
        let revision = queue.revisions[&pos];
        queue.queued.remove(&pos);
        queue.pending.retain(|queued| *queued != pos);
        queue.in_flight.insert(
            pos,
            InFlightMeshTask {
                revision,
                cancelled: Arc::new(AtomicBool::new(false)),
            },
        );

        queue.begin_diagnostic_frame(21).unwrap();
        assert!(queue.enqueue(pos));
        let stats = queue.diagnostic_stats().unwrap();
        assert_eq!(stats.cancelled, 1);
        assert_eq!(stats.enqueued, 1);
        assert_eq!(stats.oldest_active_age_frames, Some(0));
    }

    #[test]
    fn enabling_diagnostics_late_keeps_existing_age_unknown() {
        let mut queue = MeshQueue::new();
        let existing = ChunkSectionPos::new(1, 0, 0);
        let new = ChunkSectionPos::new(2, 0, 0);
        assert!(queue.enqueue(existing));
        queue.set_diagnostics_enabled(true);
        queue.begin_diagnostic_frame(20).unwrap();
        assert!(queue.enqueue(new));
        assert_eq!(
            queue.diagnostic_stats().unwrap().oldest_active_age_frames,
            None
        );
        assert!(queue.cancel(existing));
        assert_eq!(
            queue.diagnostic_stats().unwrap().oldest_active_age_frames,
            Some(0)
        );
    }

    #[test]
    fn diagnostic_frame_indices_must_advance_strictly() {
        let mut queue = MeshQueue::new();
        queue.set_diagnostics_enabled(true);
        queue.begin_diagnostic_frame(4).unwrap();
        assert_eq!(
            queue.begin_diagnostic_frame(4),
            Err(MeshQueueDiagnosticsError::NonMonotonicFrameIndex {
                previous: 4,
                current: 4
            })
        );
        assert_eq!(
            queue.begin_diagnostic_frame(3),
            Err(MeshQueueDiagnosticsError::NonMonotonicFrameIndex {
                previous: 4,
                current: 3
            })
        );
    }

    #[test]
    fn synchronous_drain_counts_builds_without_worker_events() {
        let world = fixture::blank_world(0);
        let pos = ChunkSectionPos::new(0, 0, 0);
        let mut queue = MeshQueue::new();
        queue.set_diagnostics_enabled(true);
        queue.begin_diagnostic_frame(1).unwrap();
        assert!(queue.enqueue(pos));
        assert_eq!(queue.drain(&world, DVec3::ZERO, 1).len(), 1);

        let stats = queue.diagnostic_stats().unwrap();
        assert_eq!(stats.synchronous_built, 1);
        assert_eq!(stats.dispatched, 0);
        assert_eq!(stats.completed, 0);
        assert_eq!(stats.oldest_active_age_frames, None);
    }

    #[test]
    fn diagnostic_worker_result_counters_use_revision_gate_without_spawning() {
        let pos = ChunkSectionPos::new(4, 2, -3);
        let mut queue = MeshQueue::new();
        queue.set_diagnostics_enabled(true);
        queue.begin_diagnostic_frame(7).unwrap();
        let revision = next_mesh_revision();
        queue.revisions.insert(pos, revision + 1);
        queue.in_flight.insert(
            pos,
            InFlightMeshTask {
                revision,
                cancelled: Arc::new(AtomicBool::new(false)),
            },
        );
        queue
            .result_tx
            .send(MeshTaskResult::new(
                ChunkMesh::empty(pos),
                revision,
                Duration::ZERO,
                DEFAULT_SLOW_MESH_DEADLINE,
                1,
            ))
            .unwrap();
        assert!(queue.poll_completed().is_empty());
        assert_eq!(queue.diagnostic_stats().unwrap().stale_result_rejected, 1);
    }

    #[test]
    fn diagnostic_worker_completion_counts_slow_results_without_spawning() {
        let pos = ChunkSectionPos::new(8, 1, 2);
        let mut queue = MeshQueue::new();
        queue.set_diagnostics_enabled(true);
        queue.begin_diagnostic_frame(9).unwrap();
        let revision = next_mesh_revision();
        queue.revisions.insert(pos, revision);
        queue.in_flight.insert(
            pos,
            InFlightMeshTask {
                revision,
                cancelled: Arc::new(AtomicBool::new(false)),
            },
        );
        queue
            .result_tx
            .send(MeshTaskResult::new(
                ChunkMesh::empty(pos),
                revision,
                DEFAULT_SLOW_MESH_DEADLINE + Duration::from_millis(1),
                DEFAULT_SLOW_MESH_DEADLINE,
                1,
            ))
            .unwrap();
        assert_eq!(queue.poll_completed().len(), 1);
        let stats = queue.diagnostic_stats().unwrap();
        assert_eq!(stats.completed, 1);
        assert_eq!(stats.slow_completed, 1);
    }
}
