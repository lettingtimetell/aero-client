//! The deterministic in-world benchmark, and the test that proves it is deterministic.
//!
//! T0.1's verification is stated in the plan as: *"two runs of the same capture produce identical
//! section counts, quad counts and draw counts; frame times differ only by noise."* That is
//! asserted here rather than eyeballed, because "identical" is exactly the kind of claim that
//! degrades into "close enough" the first time nobody checks it.
//!
//! # What the two runs share and what they do not
//!
//! Both runs happen **in one process, against one GPU adapter, with one set of baked assets** —
//! §0 rule 1's requirement, and the reason this is a test rather than two invocations of the
//! binary. Each run builds a fresh [`Client`] and a fresh `PhaseList`, so nothing but the file
//! carries over.
//!
//! # The controls
//!
//! Per §0 rule 2, an equality assertion is worthless unless it could have failed. Two are run:
//!
//! * **A moving camera against a fixed one.** §1.6 measures terrain CPU at 0.072 ms with a
//!   stationary camera and 0.142 ms with a moving one, because the cull cache never misses when
//!   nothing moves. A third run with `--replayCamera fixed` must therefore produce a *different*
//!   per-frame count sequence. If it did not, the comparison would be blind to the camera and
//!   the equality above would be measuring nothing.
//! * **A different cadence.** The same capture stepped at a different rate delivers its packets
//!   on different frames, so its count sequence must differ too.
//!
//! # A non-determinism this found, which is not in the replay
//!
//! The orbit configuration — the one T0.1 specifies — is bit-identical run to run and process to
//! process. The **pinned-camera control is not**, and the difference is instructive enough to
//! record next to the test that found it:
//!
//! ```text
//! fixed vs fixed (two runs, one process), frames differing out of 2000 — observed instances:
//!   lobby capture:   loaded 0 · visible 14 · draws 4 · quads    5 · binds 1904
//!   lobby capture:   loaded 0 · visible 12 · draws 6 · quads 1905 · binds   26
//!   terrain capture: loaded 0 · visible  9 · draws 4 · quads    4 · binds   20
//! ```
//!
//! `loaded_sections` is stable — the same world arrives either way — and everything downstream
//! of the *order* it arrives in is not. The size of the effect varies between invocations, which
//! is itself the signature of a hash-order dependency rather than a systematic difference; it is
//! largest on the lobby capture, whose 49 columns around a symmetric spawn produce far more exact
//! distance ties than 1089 columns of real terrain do.
//!
//! The chain is:
//!
//! 1. `mc_world::World::drain_dirty_sections` returns a **`HashSet`**, so the order sections are
//!    enqueued in is `RandomState`'s, i.e. different on every `HashSet` instance;
//! 2. `MeshQueue::take_nearest` orders that queue by `distance_sq` alone, which is **not a total
//!    order** — a stationary camera makes exact ties the common case, not the rare one;
//! 3. so which sections land in a given frame's batch, and therefore which sections end up
//!    sharing an upload buffer, follows the hash order. `vertex_buffer_binds` is a direct count
//!    of that grouping, and `quads` moves whenever a tie falls across the per-frame mesh budget
//!    so that a section is built one frame earlier or later.
//!
//! A moving camera hides it because every section is then at a distinct distance and the order
//! becomes total again — so this run's determinism is **empirical, not structural**. Both files
//! are outside this work's ownership; the one-line fixes are in the report.
//!
//! # The scene these numbers come from
//!
//! `captures/spigot-1.8.8-perfworld-vd16.mccap`, a 34-second session against §1.8's survival
//! server (seed `1234567890`, `view-distance=16`, anti-xray off). It delivers **1089 chunk
//! columns and 5884 non-vacant sections** — 46× the 127 the lobby fixtures manage, and the same
//! order as the ~6 500 §1.8 measured live. The lobby fixtures still work as input and are the
//! fallback when the terrain capture is absent; `MC_REPLAY_CAPTURE` selects any file.
//!
//! Note that the *client's* render distance does not change what a capture holds: the server sends
//! on its own `view-distance`, so a capture recorded with the client at rd 8 carries the same
//! 1089 columns. Render distance is a replay-time knob, which is exactly what makes rd 8 and
//! rd 16 comparable below — same bytes, same world, different frustum.
//!
//! # What this does **not** prove
//!
//! These runs are in the **test profile**, not the `lto="fat"` release binary §1 measured, so the
//! frame times below are internally comparable (run A against run B, on one machine, in one
//! process) and are *not* comparable with §1.3–§1.8. They also share the machine with whatever
//! else is building at the time, which shows up in `max` and not in `p50`.
//!
//! Nor does a deterministic replay prove the *client* is deterministic in general — only that it
//! is for this capture, at these two render distances, with a moving camera. The pinned-camera
//! control above is the standing counter-example.

mod common;

use std::sync::{PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::Instant;

use common::{config, gpu};
use mc_client::app::campath::{CameraScript, DEFAULT_PERIOD_FRAMES};
use mc_client::app::Client;
use mc_client::net::ReplaySummary;
use mc_physics::state::MoveInput;
use mc_render::{Camera, Gpu, PhaseList};

const WIDTH: u32 = 854;
const HEIGHT: u32 = 480;

/// Frames per run. `MC_REPLAY_FRAMES` overrides it for a longer manual measurement.
///
/// Long enough that the camera completes a whole revolution of the default
/// [`DEFAULT_PERIOD_FRAMES`] orbit — a run shorter than one period would only ever see one arc,
/// and a determinism check over one arc is a weaker check.
const FRAMES: usize = 2_000;

fn frames() -> usize {
    std::env::var("MC_REPLAY_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(FRAMES)
}

/// Frames for the budget experiment, which needs enough of them for the **control** to finish
/// draining its queue — 1089 columns × 16 sections ÷ `MESH_BUDGET` = 2178 frames, so anything at
/// or below that cannot tell "settled" from "still filling".
fn budget_experiment_frames() -> usize {
    frames().max(3_000)
}

/// The render distance the main determinism assertion runs at.
///
/// 8 and 16 are both measured; 8 is the one the headline test uses because it is the cheaper run,
/// **not** because it is the safe one. An earlier measurement suggested rd 8
/// and rd 12 settled while rd 16 thrashed; that was since retracted — its rd 8
/// and rd 12 control windows were 0.33 s and 4.21 s, far too short, and on a fully generated
/// world **every** render distance thrashes. So neither render distance here is a "quiet" one,
/// and the `mesh_queue` column below is what says so rather than an assumption either way.
const RENDER_DISTANCE: i32 = 8;

/// Captured microseconds per frame.
///
/// The default cadence is 1000 µs/frame, which plays a 44-second capture over 44 000 frames —
/// right for a real bench and far too slow for a test. 40 000 µs/frame plays the same capture in
/// 1100 frames. The cadence does not affect *whether* the replay is deterministic, only how much
/// of it fits in [`FRAMES`]; the control below re-runs at a different value and shows that
/// changing it changes the result.
const STEP_MICROS: u64 = 40_000;

/// A frame this slow is part of a run's tail, reported (not asserted) next to the frame times.
const SLOW_FRAME_MS: f64 = 200.0;

/// Keeps the timed comparison off a machine that a sibling test's replay is also using.
///
/// Every [`Client`] in this process meshes on the same process-wide terrain pool (two to four
/// workers by CPU count), and all of them share the CPU and the GPU. libtest runs this file's
/// tests in parallel, so the siblings used to start their replays alongside run A of
/// [`two_runs_of_the_same_capture_render_the_same_scene`] and were finished by run B: in every
/// gate from 2026-11-09 to 2026-11-18, A's mean frame time came out 27.5-31.9% above B's with
/// identical counts. In one reproduction, A's frames 1-56 took 27 s while the siblings were
/// starting up, and no frame of B took over 200 ms. Alone, the means differed by 0.8-4.3%.
/// So the timed test takes this lock exclusively and every other replay test shares it: the
/// siblings still overlap each other, never the measurement.
static REPLAY_MACHINE: RwLock<()> = RwLock::new(());

/// For the test whose frame times are compared: no other replay runs in this process meanwhile.
fn replay_machine_alone() -> RwLockWriteGuard<'static, ()> {
    REPLAY_MACHINE
        .write()
        .unwrap_or_else(PoisonError::into_inner)
}

/// For every other replay test: they may overlap each other, but not the timed test.
fn replay_machine_shared() -> RwLockReadGuard<'static, ()> {
    REPLAY_MACHINE
        .read()
        .unwrap_or_else(PoisonError::into_inner)
}

/// One frame's terrain numbers, as `TerrainStats` reports them after the draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FrameCounts {
    loaded_sections: usize,
    visible_sections: usize,
    draw_calls: usize,
    quads: usize,
    vertex_buffer_binds: usize,
    /// Sections still waiting in the active queue at the end of this frame. Async worker-owned
    /// sections are recorded separately below; `Client::mesh_queue_len` intentionally reports
    /// only this pending count.
    mesh_queue: usize,
    /// Dirty sections retained after the active queue filled. This separates deferred invalidation
    /// pressure from worker-owned work in the asynchronous lifecycle.
    mesh_deferred: usize,
    /// Section rebuilds currently owned by the two background terrain workers. A full active
    /// queue with this non-zero means worker backpressure, not necessarily repeated invalidation.
    mesh_in_flight: usize,
}

/// Everything one run produced.
struct Run {
    counts: Vec<FrameCounts>,
    /// Wall-clock milliseconds per `poll + tick + frame + render`.
    millis: Vec<f64>,
    summary: ReplaySummary,
    uploaded_sections: usize,
    chunks: usize,
    anchored: bool,
    /// `TerrainRenderer::gpu_memory` at the end of the run: resident sections, resident bytes,
    /// the cap, and how many sections the budget has evicted since the run started. The last two
    /// are what the budget-churn hypothesis (below) stands or falls on.
    memory: mc_render::terrain::jobs::GpuMemoryReport,
}

impl Run {
    /// The counts, deduplicated to the distinct states the run passed through. Two runs that
    /// agree here agree frame by frame as well (the vectors are compared directly), but a
    /// failure prints something a human can read instead of 2000 lines.
    fn distinct(&self) -> Vec<FrameCounts> {
        let mut out: Vec<FrameCounts> = Vec::new();
        for c in &self.counts {
            if out.last() != Some(c) {
                out.push(*c);
            }
        }
        out
    }

    /// Frames on which the resident-section count went **down**.
    ///
    /// `TerrainRenderer` only drops a section when the GPU budget evicts it, so a non-zero count
    /// here is the direct signature of the budget-churn hypothesis, and a zero
    /// is the direct absence of it. Reported rather than inferred from frame times.
    fn evictions(&self) -> usize {
        self.counts
            .windows(2)
            .filter(|w| w[1].loaded_sections < w[0].loaded_sections)
            .count()
    }

    fn peak(&self) -> FrameCounts {
        FrameCounts {
            loaded_sections: self
                .counts
                .iter()
                .map(|c| c.loaded_sections)
                .max()
                .unwrap_or(0),
            visible_sections: self
                .counts
                .iter()
                .map(|c| c.visible_sections)
                .max()
                .unwrap_or(0),
            draw_calls: self.counts.iter().map(|c| c.draw_calls).max().unwrap_or(0),
            quads: self.counts.iter().map(|c| c.quads).max().unwrap_or(0),
            vertex_buffer_binds: self
                .counts
                .iter()
                .map(|c| c.vertex_buffer_binds)
                .max()
                .unwrap_or(0),
            mesh_queue: self.counts.iter().map(|c| c.mesh_queue).max().unwrap_or(0),
            mesh_deferred: self
                .counts
                .iter()
                .map(|c| c.mesh_deferred)
                .max()
                .unwrap_or(0),
            mesh_in_flight: self
                .counts
                .iter()
                .map(|c| c.mesh_in_flight)
                .max()
                .unwrap_or(0),
        }
    }

    /// The churn signal, read from all retained mesh work: how many frames ended with active,
    /// deferred, or worker-owned sections, and the last frame index on which that was true.
    ///
    /// A client that settles empties the queue and never refills it, so `pending_frames` is a
    /// prefix and `last_pending` is well before the end of the run. A client whose queue refills
    /// as fast as it drains has `pending_frames == frames` and `last_pending == frames - 1`.
    fn queue_pressure(&self) -> (usize, usize) {
        let pending = self
            .counts
            .iter()
            .filter(|c| c.mesh_queue + c.mesh_deferred + c.mesh_in_flight > 0)
            .count();
        let last = self
            .counts
            .iter()
            .rposition(|c| c.mesh_queue + c.mesh_deferred + c.mesh_in_flight > 0)
            .map_or(0, |i| i + 1);
        (pending, last)
    }
}

/// How many frames each individual counter differs on, field by field.
///
/// A whole-vector `assert_eq!` says "these runs differ" and nothing else. Naming the field is
/// what turns a failure into a finding.
fn field_diffs(a: &[FrameCounts], b: &[FrameCounts]) -> Vec<(&'static str, usize)> {
    /// One named counter and the accessor that reads it.
    type Field = (&'static str, fn(&FrameCounts) -> usize);
    let fields: [Field; 8] = [
        ("loaded_sections", |c| c.loaded_sections),
        ("visible_sections", |c| c.visible_sections),
        ("draw_calls", |c| c.draw_calls),
        ("quads", |c| c.quads),
        ("vertex_buffer_binds", |c| c.vertex_buffer_binds),
        ("mesh_queue", |c| c.mesh_queue),
        ("mesh_deferred", |c| c.mesh_deferred),
        ("mesh_in_flight", |c| c.mesh_in_flight),
    ];
    fields
        .into_iter()
        .map(|(name, get)| {
            let differing = a.iter().zip(b).filter(|(x, y)| get(x) != get(y)).count();
            (name, differing)
        })
        .collect()
}

/// Mean, sd, min, max, n and percentiles — §0 rules 3 and 4, which ask for both.
#[derive(Debug, Clone, Copy)]
struct Spread {
    n: usize,
    mean: f64,
    sd: f64,
    min: f64,
    max: f64,
    p50: f64,
    p95: f64,
    p99: f64,
    p999: f64,
}

impl Spread {
    fn of(samples: &[f64]) -> Spread {
        assert!(!samples.is_empty());
        let n = samples.len();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let variance = samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n as f64;
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        let at = |p: f64| -> f64 {
            let index = ((p * n as f64).ceil() as usize).clamp(1, n) - 1;
            sorted[index]
        };
        Spread {
            n,
            mean,
            sd: variance.sqrt(),
            min: sorted[0],
            max: sorted[n - 1],
            p50: at(0.50),
            p95: at(0.95),
            p99: at(0.99),
            p999: at(0.999),
        }
    }

    fn line(&self, label: &str) -> String {
        format!(
            "{label}: n={} mean={:.4} sd={:.4} min={:.4} p50={:.4} p95={:.4} p99={:.4} \
             p99.9={:.4} max={:.4} ms",
            self.n, self.mean, self.sd, self.min, self.p50, self.p95, self.p99, self.p999, self.max
        )
    }
}

/// The capture to replay: `MC_REPLAY_CAPTURE` if set, otherwise the committed **terrain**
/// recording, falling back to the lobby fixture.
///
/// `captures/spigot-1.8.8-perfworld-vd16.mccap` is a 34-second session against the survival
/// server set up for these measurements — seed `1234567890`, `view-distance=16`, anti-xray off. It
/// delivers **1089 columns and 5884 non-vacant sections** where the lobby fixtures deliver 127,
/// which is the difference between a working instrument and a useful one.
fn capture_path() -> Option<std::path::PathBuf> {
    if let Some(path) = std::env::var_os("MC_REPLAY_CAPTURE") {
        let path = std::path::PathBuf::from(path);
        return path.is_file().then_some(path);
    }
    let parity = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(1)?
        .join("mc-parity");
    [
        parity
            .join("captures")
            .join("spigot-1.8.8-perfworld-vd16.mccap"),
        parity
            .join("tests")
            .join("fixtures")
            .join("spigot-1.8.8-offline-binary-session.mccap"),
    ]
    .into_iter()
    .find(|p| p.is_file())
}

/// One whole run: build a client, replay the capture for [`frames`] frames, record every frame.
// Keep the independent experiment controls explicit at each comparison's call site.
#[allow(clippy::too_many_arguments)]
fn run(
    gpu: &mut Gpu,
    capture: &std::path::Path,
    camera_script: CameraScript,
    step: u64,
    render_distance: i32,
    budget_bytes: Option<u64>,
    total: usize,
    deterministic_mesh: bool,
) -> Run {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = config(dir.path(), "BenchBot").expect("the checkout's 1.8.9 jar");
    config.replay = Some(capture.to_path_buf());
    config.replay_step_micros = step;
    config.replay_camera = camera_script;

    let mut client = Client::new(config, &mut |_, _| {}).expect("start");
    // Set before `install`, which sizes the terrain phase from it, and before `connect`, which
    // puts it in the (unsent) `ClientSettings`. There is no `options.txt` in the scratch game
    // directory, so without this the run would use `GameSettings`' own default and the test
    // would silently stop being a rd-8 measurement if that default ever moved.
    client.settings.render_distance_chunks = render_distance;

    let mut phases = PhaseList::new();
    client.install(gpu, &mut phases, WIDTH, HEIGHT);
    client.set_replay_deterministic_mesh(deterministic_mesh);
    if let Some(bytes) = budget_bytes {
        client
            .render()
            .expect("installed")
            .terrain
            .borrow_mut()
            .set_gpu_budget(mc_render::terrain::jobs::GpuBudget::new(bytes));
    }
    let mut camera = Camera::default();
    // `Client::new` loads everything up front, so `finish_loading` — which is what honours
    // `--server` — never runs. The connect is therefore explicit here; the host and port are
    // ignored, because `open_connection` takes the replay branch.
    client.connect("replay", 0);

    let input = MoveInput::default();
    let mut counts = Vec::with_capacity(total);
    let mut millis = Vec::with_capacity(total);
    for _ in 0..total {
        let started = Instant::now();
        client.poll();
        client.tick(&input);
        client.frame(gpu, &mut camera, 0.0);
        let frame = gpu.acquire().expect("acquire offscreen frame");
        phases.render(gpu, frame.view(), &camera, 0.0);
        millis.push(started.elapsed().as_secs_f64() * 1e3);

        let stats = client.render().expect("installed").terrain.borrow().stats();
        counts.push(FrameCounts {
            loaded_sections: stats.loaded_sections,
            visible_sections: stats.visible_sections,
            draw_calls: stats.draw_calls,
            quads: stats.quads,
            vertex_buffer_binds: stats.vertex_buffer_binds,
            mesh_queue: client.mesh_queue_len(),
            mesh_deferred: client.mesh_queue_deferred_len(),
            mesh_in_flight: client.mesh_queue_in_flight_len(),
        });
    }

    let session = client.session().expect("a replay session");
    let summary = session
        .replay_source()
        .expect("a capture-backed handle")
        .summary()
        .clone();
    let chunks = session.world().loaded_chunk_count();
    let memory = client
        .render()
        .expect("installed")
        .terrain
        .borrow()
        .gpu_memory();
    Run {
        counts,
        millis,
        summary,
        uploaded_sections: client.uploaded_section_count(),
        chunks,
        anchored: client.camera_path().is_some_and(|p| p.anchor().is_some()),
        memory,
    }
}

#[test]
fn two_runs_of_the_same_capture_render_the_same_scene() {
    let _machine = replay_machine_alone();
    let Some(capture) = capture_path() else {
        eprintln!("skip: no .mccap fixture in this checkout");
        return;
    };
    let Some(mut gpu) = gpu(WIDTH, HEIGHT) else {
        return;
    };
    if config(std::path::Path::new("."), "x").is_none() {
        eprintln!("skip: no vanilla jar in this checkout");
        return;
    }

    let orbit = CameraScript::default();
    let started = Instant::now();
    // This assertion intentionally requests the deterministic compatibility seam. Other replay
    // tests leave it disabled so they exercise the production asynchronous worker lifecycle.
    let a = run(
        &mut gpu,
        &capture,
        orbit,
        STEP_MICROS,
        RENDER_DISTANCE,
        None,
        frames(),
        true,
    );
    let b = run(
        &mut gpu,
        &capture,
        orbit,
        STEP_MICROS,
        RENDER_DISTANCE,
        None,
        frames(),
        true,
    );
    let elapsed = started.elapsed();

    println!("capture: {}", capture.display());
    println!("         {}", a.summary.line());
    println!(
        "run A: {} frames, {} chunk columns loaded, {} sections resident, camera anchored={}",
        a.counts.len(),
        a.chunks,
        a.uploaded_sections,
        a.anchored
    );
    println!(
        "run B: {} frames, {} chunk columns loaded, {} sections resident, camera anchored={}",
        b.counts.len(),
        b.chunks,
        b.uploaded_sections,
        b.anchored
    );
    println!("peak A: {:?}", a.peak());
    println!("peak B: {:?}", b.peak());
    // Two scalars a human can compare at a glance, and that a diff would move: the number of
    // distinct count-states the run passed through (i.e. how much the scene actually changed),
    // and the total quads submitted across every frame.
    for (label, r) in [("A", &a), ("B", &b)] {
        let (pending, last) = r.queue_pressure();
        println!(
            "F5 mesh work {label}: retained on {pending} of {} frames; last non-empty frame {last}; peak pending/deferred/in-flight {}/{}/{}",
            r.counts.len(),
            r.peak().mesh_queue,
            r.peak().mesh_deferred,
            r.peak().mesh_in_flight,
        );
    }
    println!(
        "states A/B: {} / {}   total quads A/B: {} / {}   total draws A/B: {} / {}",
        a.distinct().len(),
        b.distinct().len(),
        a.counts.iter().map(|c| c.quads).sum::<usize>(),
        b.counts.iter().map(|c| c.quads).sum::<usize>(),
        a.counts.iter().map(|c| c.draw_calls).sum::<usize>(),
        b.counts.iter().map(|c| c.draw_calls).sum::<usize>(),
    );
    let spread_a = Spread::of(&a.millis);
    let spread_b = Spread::of(&b.millis);
    println!("{}", spread_a.line("frame time A"));
    println!("{}", spread_b.line("frame time B"));
    println!(
        "both runs in {:.1} s; mean differs by {:.1}% ({:.4} ms), median by {:.1}% ({:.4} ms)",
        elapsed.as_secs_f64(),
        (spread_a.mean - spread_b.mean).abs() / spread_a.mean * 100.0,
        (spread_a.mean - spread_b.mean).abs(),
        (spread_a.p50 - spread_b.p50).abs() / spread_a.p50 * 100.0,
        (spread_a.p50 - spread_b.p50).abs()
    );
    // Where each run's slow frames sit: a tail in one run only moves the mean, not the median.
    for (label, r) in [("A", &a), ("B", &b)] {
        let slow: Vec<usize> = (0..r.millis.len())
            .filter(|&i| r.millis[i] > SLOW_FRAME_MS)
            .collect();
        println!(
            "frames over {SLOW_FRAME_MS} ms {label}: {} (first {:?}, last {:?}), {:.1} s in total",
            slow.len(),
            slow.first(),
            slow.last(),
            // `abs`: an empty float sum is -0.0, which would print as "-0.0 s".
            slow.iter().map(|&i| r.millis[i]).sum::<f64>().abs() / 1e3
        );
    }

    // ---- floors, so "identical" cannot come to mean "identically empty" ----
    assert!(
        a.summary.replayed > 100,
        "the capture must actually drive a session: {}",
        a.summary.line()
    );
    assert!(
        a.peak().loaded_sections > 0 && a.peak().quads > 0 && a.peak().draw_calls > 0,
        "nothing was ever drawn, so an equality over the counts proves nothing: {:?}",
        a.peak()
    );
    assert!(
        a.anchored && b.anchored,
        "the scripted camera never anchored, so the runs did not move the camera at all"
    );

    // ---- T0.1's verification ----
    assert_eq!(
        a.summary, b.summary,
        "the same file decoded to two different packet sets"
    );
    assert_eq!(a.chunks, b.chunks, "different amounts of world arrived");
    assert_eq!(
        a.uploaded_sections, b.uploaded_sections,
        "different numbers of sections ended up resident on the GPU"
    );
    if a.counts != b.counts {
        let first = a
            .counts
            .iter()
            .zip(&b.counts)
            .position(|(x, y)| x != y)
            .expect("the vectors differ");
        panic!(
            "the two runs diverged at frame {first}:\n  A = {:?}\n  B = {:?}\n\
             A passed through {} distinct states, B through {}",
            a.counts[first],
            b.counts[first],
            a.distinct().len(),
            b.distinct().len()
        );
    }

    // ---- the frame-time half of the claim, stated as a number ----
    // "Only by noise" is not a property of the client, it is a property of this rig, so it is
    // reported rather than asserted tightly. The one thing worth asserting is that the two
    // distributions are the *same* distribution — a 25% gap in the mean would mean the second
    // run did different work, which the count equality above should already have caught. Both
    // runs hold `REPLAY_MACHINE` alone, so no sibling test's replay lands in one run and not the
    // other; the slow-frame lines above show it if anything else does.
    let gap = (spread_a.mean - spread_b.mean).abs() / spread_a.mean.max(spread_b.mean);
    assert!(
        gap < 0.25,
        "the two runs' mean frame times differ by {:.1}%, which is not noise: {} / {}",
        gap * 100.0,
        spread_a.line("A"),
        spread_b.line("B")
    );
}

/// **The control.** An equality assertion has to be able to fail.
///
/// Two things are changed, one at a time, and each must move the numbers:
///
/// * pinning the camera — §1.6's cull-cache effect, and the reason
///   [`CameraScript`]'s default is an orbit;
/// * changing the cadence — which moves every packet onto a different frame.
///
/// If either of these produced the identical count sequence, the test above would be comparing
/// something that does not depend on the run at all.
#[test]
fn the_counts_respond_to_the_camera_and_to_the_cadence() {
    let _machine = replay_machine_shared();
    let Some(capture) = capture_path() else {
        eprintln!("skip: no .mccap fixture in this checkout");
        return;
    };
    let Some(mut gpu) = gpu(WIDTH, HEIGHT) else {
        return;
    };
    if config(std::path::Path::new("."), "x").is_none() {
        eprintln!("skip: no vanilla jar in this checkout");
        return;
    }

    let orbit = run(
        &mut gpu,
        &capture,
        CameraScript::default(),
        STEP_MICROS,
        RENDER_DISTANCE,
        None,
        frames(),
        false,
    );
    let fixed = run(
        &mut gpu,
        &capture,
        CameraScript::Fixed,
        STEP_MICROS,
        RENDER_DISTANCE,
        None,
        frames(),
        false,
    );
    let fixed2 = run(
        &mut gpu,
        &capture,
        CameraScript::Fixed,
        STEP_MICROS,
        RENDER_DISTANCE,
        None,
        frames(),
        false,
    );
    let faster = run(
        &mut gpu,
        &capture,
        CameraScript::default(),
        STEP_MICROS / 4,
        RENDER_DISTANCE,
        None,
        frames(),
        false,
    );

    println!("orbit  peak {:?}", orbit.peak());
    println!("fixed  peak {:?}", fixed.peak());
    println!("fixed2 peak {:?}", fixed2.peak());
    println!("faster peak {:?}", faster.peak());
    // Recorded, not asserted. See this file's header: two pinned-camera runs are *not*
    // bit-identical, and the reason lives in two crates this work does not own. Printing the
    // per-field breakdown is what makes the claim checkable by whoever fixes it.
    println!(
        "fixed vs fixed2, frames differing per field: {:?}",
        field_diffs(&fixed.counts, &fixed2.counts)
    );
    println!(
        "orbit vs fixed,  frames differing per field: {:?}",
        field_diffs(&orbit.counts, &fixed.counts)
    );
    println!("{}", Spread::of(&orbit.millis).line("frame time orbit"));
    println!("{}", Spread::of(&fixed.millis).line("frame time fixed"));

    assert!(
        orbit.peak().quads > 0,
        "no control is possible on an empty scene"
    );
    assert_ne!(
        orbit.counts, fixed.counts,
        "a moving camera and a pinned one produced the identical frame-by-frame counts, so \
         the scripted path is not reaching the renderer and the equality test above is blind"
    );
    assert_ne!(
        orbit.counts, faster.counts,
        "the same capture at a quarter of the cadence produced the identical frame-by-frame \
         counts, so the counts do not depend on the replay at all"
    );
    // ... but the *world* the two cadences build is the same one, just reached at different
    // frames. This is what separates "the cadence changed the schedule" from "the cadence
    // changed the scene", and it is the sharper of the two statements.
    assert_eq!(
        orbit.summary, faster.summary,
        "the same file must decode to the same packets whatever the cadence"
    );

    // The camera really did complete revolutions rather than crawling along one arc.
    assert!(
        frames() as u64 >= DEFAULT_PERIOD_FRAMES,
        "a run shorter than one orbit is a weaker determinism check"
    );
}

/// Render distance **16**, measured and reported rather than asserted.
///
/// The real-terrain finding: at rd 16 on real terrain the client re-meshes its
/// working set ~47 times over four minutes and never settles, with a worst `drain_mesh_queue`
/// call of 138 ms. The leading hypothesis is GPU-budget eviction churn. If that is right, a
/// replay at rd 16 has a timing-dependent loop inside it and cannot be expected to reproduce
/// frame for frame — through no fault of the replay source.
///
/// So this test asserts only what must hold either way — **the same world arrives**, packet for
/// packet and section for section — and prints the per-field divergence of everything downstream.
/// A run that comes back with zero divergence is evidence *against* F5's hypothesis and should be
/// said so out loud; a run that diverges tells whoever fixes F5 exactly which counter to watch.
///
/// **What it came back with, 2026-08-16, 12 000 frames per run** (`MC_REPLAY_FRAMES=12000`):
///
/// ```text
/// rd 16 A vs B, frames differing per field: loaded 0 · visible 0 · draws 0 · quads 0 · binds 0
/// rd 16 frames on which resident sections fell: A 0 / B 0 over 12000
/// rd 16 GPU terrain memory: 5165 sections, 312.8 MiB resident of 512 MiB cap (61.1 %),
///                           0 sections evicted over the whole run
/// ```
///
/// Bit-identical, and the budget never fired. **Under replay the client settles**: at rd 8 the
/// queue drains once — 2178 frames × 8 = 17 424 sections, exactly the working set — and is then
/// empty for every one of the remaining 9 821 frames. That is the opposite of F5's live
/// signature, on the same world and the same packets.
///
/// It does not exonerate the client, because
/// `the_gpu_budget_is_what_decides_whether_the_mesh_queue_ever_settles` shows the churn is real
/// and reproducible the moment the budget binds. It relocates the question to **how many bytes
/// were resident in the live run**, which is now one `TerrainRenderer::gpu_memory()` call away.
///
/// One confound has to be stated: §1.8 measured F5 on the *pre-change* binary, where
/// `TerrainVertex` was 28 bytes. It is 20 today (T1.1 stage 1), so the scene that sits at 61 % of
/// the cap here would have sat at 312.8 × 28/20 = **437.9 MiB, 85.5 %** — the same scene, much
/// closer to the edge. Re-running against the 28-byte control format
/// (`TerrainOptions::wide_vertices`, which `mc-render` already carries) is the experiment that
/// would settle it, and it needs one line of plumbing in a crate this work does not own.
#[test]
fn at_render_distance_16_the_replay_is_measured_not_assumed() {
    let _machine = replay_machine_shared();
    let Some(capture) = capture_path() else {
        eprintln!("skip: no .mccap fixture in this checkout");
        return;
    };
    let Some(mut gpu) = gpu(WIDTH, HEIGHT) else {
        return;
    };
    if config(std::path::Path::new("."), "x").is_none() {
        eprintln!("skip: no vanilla jar in this checkout");
        return;
    }

    let orbit = CameraScript::default();
    let a = run(
        &mut gpu,
        &capture,
        orbit,
        STEP_MICROS,
        16,
        None,
        frames(),
        false,
    );
    let b = run(
        &mut gpu,
        &capture,
        orbit,
        STEP_MICROS,
        16,
        None,
        frames(),
        false,
    );

    println!("rd 16 peak A: {:?}", a.peak());
    println!("rd 16 peak B: {:?}", b.peak());
    println!(
        "rd 16 totals A/B: quads {} / {}   draws {} / {}   resident sections {} / {}",
        a.counts.iter().map(|c| c.quads).sum::<usize>(),
        b.counts.iter().map(|c| c.quads).sum::<usize>(),
        a.counts.iter().map(|c| c.draw_calls).sum::<usize>(),
        b.counts.iter().map(|c| c.draw_calls).sum::<usize>(),
        a.uploaded_sections,
        b.uploaded_sections,
    );
    println!(
        "rd 16 A vs B, frames differing per field: {:?}",
        field_diffs(&a.counts, &b.counts)
    );
    println!(
        "rd 16 frames on which resident sections fell (GPU-budget evictions): A {} / B {} \
         over {} frames",
        a.evictions(),
        b.evictions(),
        a.counts.len()
    );
    for (label, r) in [("A", &a), ("B", &b)] {
        let (pending, last) = r.queue_pressure();
        println!(
            "rd 16 F5 mesh work {label}: retained on {pending} of {} frames; last non-empty frame {last}; peak pending/deferred/in-flight {}/{}/{}",
            r.counts.len(),
            r.peak().mesh_queue,
            r.peak().mesh_deferred,
            r.peak().mesh_in_flight,
        );
    }
    for (label, run) in [("A", &a), ("B", &b)] {
        let m = run.memory;
        println!(
            "rd 16 GPU terrain memory {label}: {} sections, {:.1} MiB resident of {:.0} MiB cap \
             ({:.1} %), {} sections evicted over the whole run",
            m.sections,
            m.bytes as f64 / 1_048_576.0,
            m.limit as f64 / 1_048_576.0,
            m.used_fraction() * 100.0,
            m.evicted,
        );
    }
    println!("{}", Spread::of(&a.millis).line("rd 16 frame time A"));
    println!("{}", Spread::of(&b.millis).line("rd 16 frame time B"));
    println!("rd 16 identical frame for frame: {}", a.counts == b.counts);

    // What must hold whatever F5 turns out to be: the file decodes the same way and delivers the
    // same world. If *this* diverged the replay source itself would be at fault.
    assert_eq!(a.summary, b.summary);
    assert_eq!(a.chunks, b.chunks, "different amounts of world arrived");
    assert!(
        a.peak().quads > 0 && a.peak().draw_calls > 0,
        "nothing was drawn at rd 16: {:?}",
        a.peak()
    );
}

/// **What a deterministic replay is actually for: an experiment on the real-terrain finding.**
///
/// F5's live signature is `mesh_section` calls pinned at `8 × frames` — `MESH_BUDGET` — with no
/// decay, at every render distance, on a world where nothing moves. The leading hypothesis is
/// GPU-budget churn: the budget evicts, `Client::upload_terrain` re-enqueues what was evicted,
/// and the queue refills exactly as fast as it drains.
///
/// A replay is the right instrument for that question precisely because it takes the *server*
/// out of it: the same 1089 columns arrive on the same frames every time, so anything that keeps
/// happening afterwards is the client's own doing.
///
/// This runs the same capture twice at the same render distance, changing **one** thing: the GPU
/// vertex budget. The control is the default 512 MiB, under which the scene fits. The treatment
/// is a budget deliberately set below what the scene needs. If the hypothesis is right, the
/// control's mesh queue drains once and stays empty and the treatment's never empties.
/// Both runs explicitly use fixed-count meshing and uploads, so worker throughput cannot
/// masquerade as memory-budget churn. The cadence and render-distance tests above measure
/// the ordinary asynchronous path separately.
#[test]
fn the_gpu_budget_is_what_decides_whether_the_mesh_queue_ever_settles() {
    let _machine = replay_machine_shared();
    let Some(capture) = capture_path() else {
        eprintln!("skip: no .mccap fixture in this checkout");
        return;
    };
    let Some(mut gpu) = gpu(WIDTH, HEIGHT) else {
        return;
    };
    if config(std::path::Path::new("."), "x").is_none() {
        eprintln!("skip: no vanilla jar in this checkout");
        return;
    }

    const TIGHT_BYTES: u64 = 128 * 1024 * 1024;
    let orbit = CameraScript::default();
    let roomy = run(
        &mut gpu,
        &capture,
        orbit,
        STEP_MICROS,
        RENDER_DISTANCE,
        None,
        budget_experiment_frames(),
        true,
    );
    let tight = run(
        &mut gpu,
        &capture,
        orbit,
        STEP_MICROS,
        RENDER_DISTANCE,
        Some(TIGHT_BYTES),
        budget_experiment_frames(),
        true,
    );

    for (label, r) in [("512 MiB (control)", &roomy), ("128 MiB (tight)", &tight)] {
        let (pending, last) = r.queue_pressure();
        let m = r.memory;
        println!(
            "budget {label}: retained mesh work non-empty on {pending} of {} frames; last non-empty \
             frame {last}; peak pending/deferred/in-flight {}/{}/{}; {:.1} MiB resident of {:.0} MiB; {} sections evicted; \
             {} frames on which resident sections fell",
            r.counts.len(),
            r.peak().mesh_queue,
            r.peak().mesh_deferred,
            r.peak().mesh_in_flight,
            m.bytes as f64 / 1_048_576.0,
            m.limit as f64 / 1_048_576.0,
            m.evicted,
            r.evictions(),
        );
        println!("  {}", Spread::of(&r.millis).line("frame time"));
    }

    // The control must have settled, or there is no experiment: a queue that was still draining
    // its first pass when the run ended says nothing about whether it refills.
    let (_, roomy_last) = roomy.queue_pressure();
    let (tight_pending, tight_last) = tight.queue_pressure();
    assert!(
        roomy_last < roomy.counts.len(),
        "the control never emptied its mesh queue in {} frames, so this run cannot distinguish \
         'settled' from 'thrashing' — raise MC_REPLAY_FRAMES",
        roomy.counts.len()
    );
    assert_eq!(
        roomy.memory.evicted, 0,
        "the control evicted sections, so it is not a control"
    );

    // The treatment: the budget is the only thing that changed.
    assert!(
        tight.memory.evicted > 0,
        "the tight budget never evicted anything, so it was not tight enough to test the \
         hypothesis: {:.1} MiB resident of {:.0} MiB",
        tight.memory.bytes as f64 / 1_048_576.0,
        tight.memory.limit as f64 / 1_048_576.0
    );
    // Frame 1 is legitimately empty — nothing has been enqueued yet — so the claim is "still
    // full at the end, and essentially never empty in between", not a bare equality.
    assert_eq!(
        tight_last,
        tight.counts.len(),
        "under a budget it cannot fit in, the mesh queue was empty by the end of the run; the \
         churn hypothesis predicts it never empties"
    );
    assert!(
        tight_pending * 100 >= tight.counts.len() * 99,
        "the mesh queue emptied on {} of {} frames under the tight budget, which is not the \
         'refills as fast as it drains' the hypothesis predicts (last non-empty {tight_last})",
        tight.counts.len() - tight_pending,
        tight.counts.len()
    );
}

/// A `--replay` that names something unreadable must behave like a failed connection, not a
/// panic and not a silent main menu.
#[test]
fn an_unreadable_capture_is_a_disconnect_not_a_crash() {
    let _machine = replay_machine_shared();
    let Some(mut gpu) = gpu(64, 64) else {
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let Some(mut config) = config(dir.path(), "BenchBot") else {
        eprintln!("skip: no vanilla jar in this checkout");
        return;
    };
    config.replay = Some(dir.path().join("does-not-exist.mccap"));

    let mut client = Client::new(config, &mut |_, _| {}).expect("start");
    let mut phases = PhaseList::new();
    client.install(&gpu, &mut phases, 64, 64);
    let mut camera = Camera::default();
    client.connect("replay", 0);

    assert!(client.session().is_none(), "no session can have opened");
    // The client must still render, which is the part a panic would take away.
    for _ in 0..4 {
        client.poll();
        client.frame(&gpu, &mut camera, 0.0);
        let frame = gpu.acquire().expect("acquire");
        phases.render(&gpu, frame.view(), &camera, 0.0);
    }
    assert!(client.running());
}
