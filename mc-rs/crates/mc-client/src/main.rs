#![forbid(unsafe_code)]

//! The game binary.
//!
//! java: `client/main/Main.java` (120 lines) plus `Minecraft.run` (`:400`) — parse the launch
//! options, build the game, run it, and turn anything that escapes into a crash report.
//!
//! `anyhow` is used here and nowhere else in the workspace: every library call below
//! already returns a typed error, and this level only needs to add context and print.
//!
//! # The shape of `Minecraft.run`, kept
//!
//! ```text
//! run() {
//!     try { startGame() } catch { crash("Initializing game"); return }
//!     while (running) { runGameLoop() }         ← throws are caught and reported
//!     finally { shutdownMinecraftApplet() }
//! }
//! ```
//!
//! Two differences, both deliberate. Vanilla calls `System.exit` from inside
//! `displayCrashReport`, which runs *after* its `finally` — here the exit code is returned so
//! that `Drop` runs for the GPU, the tokio runtime and the unsaved settings. And a Rust panic is
//! not an exception: it is caught by the hook installed in [`mc_client::app::crash`], which
//! records it for the run loop to notice, exactly as `Minecraft.crashed` sets `hasCrashed`.

use std::process::ExitCode;

use anyhow::Context;
use mc_client::app::crash::{self, CrashExit, CrashReport};
use mc_client::app::{Client, ClientConfig};
use mc_render::shell::{Shell, ShellConfig};

/// The process-wide allocator.
///
/// A crowded BedWars frame makes thousands of short-lived allocations (UI draw lists, font
/// scratch, nameplates, entity scratch) and the mesh workers allocate per section; Windows'
/// default process heap takes its lock for every block the low-fragmentation front end does not
/// serve (16 KiB and up) and hands blocks of about a megabyte and up straight back to
/// `VirtualFree`, so the largest per-frame `Vec`s re-fault their pages every frame. mimalloc's
/// thread-local free lists and retained segments avoid both. No `unsafe` is written here:
/// the `GlobalAlloc` impl is the crate's, and the attribute's generated shims are compiler
/// expansion, so `#![forbid(unsafe_code)]` above still holds.
///
/// Only the binary may name an allocator; the library crates, their tests and benches keep the
/// system one (this file's own unit tests do get it). The default-on `mimalloc` feature is the
/// switch for an A/B (`--no-default-features --features offline` is the system-allocator arm).
/// Nothing observable changes except resident memory: the allocator retains freed segments for
/// reuse, so F3's `Mem:` reads somewhat higher.
#[cfg(all(feature = "mimalloc", not(feature = "dhat-heap")))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Measurement builds only (`--features dhat-heap`): dhat counts every allocation, writes
/// `dhat-heap.json` when [`run`] returns, and `run` prints the allocations per frame next to
/// the frame-time line — the number every allocation-reuse patch (ALLOC-*) should move. dhat
/// takes over from mimalloc while the feature is on; its own `unsafe impl` lives in the crate.
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static GLOBAL: dhat::Alloc = dhat::Alloc;

/// A game-loop failure whose report was enriched while the client was still alive.
///
/// Vanilla's `Minecraft.run` keeps `Minecraft` alive through
/// `addGraphicsAndWorldToCrashReport` on every post-start failure.  Returning a plain
/// `anyhow::Error` here would drop the client before that enrichment, which makes the report lose
/// the current screen/world and also made the panic path print a second, bare report.  The report
/// is deliberately returned *after* shutdown so the caller can present it without holding a GPU.
struct RunFailure {
    report: CrashReport,
}

impl RunFailure {
    fn new(report: CrashReport) -> Self {
        Self { report }
    }
}

/// The small seam that lets the error-report path be tested without constructing a GPU client.
/// Production uses [`Client::describe`]; the test double below proves the callback is called
/// before the shared `Game directory` detail is appended.
trait CrashReportContext {
    fn describe_crash(&self, report: &mut CrashReport);
}

impl CrashReportContext for Client {
    fn describe_crash(&self, report: &mut CrashReport) {
        self.describe(report);
    }
}

fn add_run_error_context<D: CrashReportContext + ?Sized>(
    report: &mut CrashReport,
    client: Option<&D>,
    adapter_facts: Option<&mc_render::AdapterFacts>,
    game_dir: &std::path::Path,
) {
    if let Some(client) = client {
        client.describe_crash(report);
    } else {
        // java: the `startGame` catch creates this category before calling the common crash
        // enrichment method.  There is no Client to describe when boot itself failed.
        report.make_category("Initialization");
    }
    if let Some(facts) = adapter_facts {
        report.add_adapter_facts(facts);
    }
    report
        .category()
        .add("Game directory", game_dir.display().to_string());
}

/// Run the event loop, turning a panic that escapes it into the report the panic hook recorded.
///
/// A panic on the main thread inside a frame unwinds into the window
/// library, which ends the loop and resumes the panic out of `Shell::run`. Uncaught, it left
/// `main` with exit code 101: no shutdown, and the report the hook had just recorded was never
/// written, which made the most common crash the one without a crash report. java:
/// `Minecraft.run` catches every `Throwable` from the loop and shows its crash report.
/// A panic the hook did not record (no hook installed) keeps Rust's own behaviour.
fn run_catching_panics<R>(event_loop: impl FnOnce() -> R) -> Result<R, Box<CrashReport>> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(event_loop)) {
        Ok(value) => Ok(value),
        Err(payload) => match crash::take_recorded_crash() {
            Some(report) => Err(Box::new(report)),
            None => std::panic::resume_unwind(payload),
        },
    }
}

fn shutdown_client(client: &mut Client) {
    let report = client.shutdown();
    if !report.is_clean() {
        for (step, error) in &report.failures {
            tracing::error!(?step, %error, "shutdown step failed");
        }
    }
}

fn main() -> ExitCode {
    // Before any thread or graphics, audio or video library loads a DLL:
    // the current directory (the shareable `game` folder) and `PATH` leave the DLL search.
    let safe_dll_search = mc_render::dll_search::harden_dll_search_order();
    // The update helper and the update self-test are this same exe started
    // with their own flags (`mc_update::helper`): they run and exit before anything of the game.
    // A normal command line returns `None` at once.
    if let Some(code) = mc_update::helper::dispatch(std::env::args_os()) {
        return ExitCode::from(code);
    }
    mc_core::init_tracing();
    tracing::debug!(safe_dll_search, "DLL search order");
    // Log lines are written by a background thread. A panic that unwinds out of the event loop
    // skips every explicit `flush_logs()` below; this guard still drains the queue on that path.
    let _flush_logs = mc_core::FlushLogsOnDrop;
    // java: `Minecraft.crashed(CrashReport)` — record and let the loop notice. Installed before
    // anything else so a panic during option parsing is still a report.
    crash::install_panic_hook();
    // A just-installed update counts this start; one that never reached its
    // title screen in three starts hands itself to the update helper's rollback and exits.
    // Outside a portable install: one cached file check, nothing else.
    if mc_update::helper::on_client_start() == mc_update::helper::StartAction::Exit {
        mc_core::flush_logs();
        return ExitCode::SUCCESS;
    }

    let config = match ClientConfig::from_env() {
        Ok(config) => with_default_assets(config),
        Err(message) => {
            mc_core::flush_logs();
            eprintln!("mc-rs: {message}");
            return ExitCode::from(2);
        }
    };
    if config.version_jar.as_ref().is_none_or(|jar| !jar.is_file()) {
        tracing::warn!("no Minecraft 1.8.9 jar: pass --versionJar or set MC_JARS_DIR");
    }
    let game_dir = config.game_dir.clone();

    let exit = match run(config) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            let RunFailure { mut report } = *error;
            // Log lines are written by a background thread (`mc_core::logwriter`); let the
            // ones that led here reach the console before the report that follows them.
            mc_core::flush_logs();
            // Keep this as the only panic-report presentation.  The old path displayed the
            // enriched report here and then returned an anyhow error, causing main to print a
            // second bare report after it.
            match crash::display_crash_report(&mut report, &game_dir) {
                CrashExit::Saved => ExitCode::from(255),
                CrashExit::NotSaved => ExitCode::from(254),
            }
        }
    };
    // The writer thread does not outlive `main`: drain it so `stopped` and any shutdown
    // failures are not lost.
    mc_core::flush_logs();
    exit
}

/// Where the client finds the vanilla 1.8.9 files that the command line did not name.
///
/// A development build (debug assertions on) also tries the repository checkout it was built
/// from, as the tests do. An optimised build, which is every client that ships, never does: that
/// path is compiled in and names a folder of the PC that built the client. It is told instead: the launchers pass `--assetsDir` and
/// `--versionJar`, and the benchmark and PGO tools set `MC_JARS_DIR`.
#[cfg(debug_assertions)]
fn with_default_assets(config: ClientConfig) -> ClientConfig {
    config.with_discovered_assets()
}

#[cfg(not(debug_assertions))]
fn with_default_assets(config: ClientConfig) -> ClientConfig {
    config.with_assets_from_env()
}

/// Build the client and run it. java: `Minecraft.run`.
fn run(config: ClientConfig) -> std::result::Result<(), Box<RunFailure>> {
    mc_core::profile_scope!("run");

    // java: `startGame` creates the display (`:488`) *before* `refreshResources` (`:497`). Only
    // the pack stack, the language and the font are loaded here — everything the loading screen
    // itself needs — and the expensive stages run one per frame inside the shell, with the window
    // open and `LoadingScreenRenderer`'s port on screen. `--server` is honoured when that
    // finishes, not before, or `GuiConnecting` would hide the progress bar.
    let game_dir = config.game_dir.clone();
    let mut client = match Client::new_deferred(config).context("Initializing game") {
        Ok(client) => client,
        Err(error) => {
            let mut report = CrashReport::from_error("Initializing game", &error);
            add_run_error_context::<Client>(&mut report, None, None, &game_dir);
            return Err(Box::new(RunFailure::new(report)));
        }
    };

    // Measurement builds only: counts every allocation from here on (the boot above is
    // excluded, so totals divide sensibly by frames); dhat writes `dhat-heap.json` when this
    // is dropped, i.e. when `run` returns.
    #[cfg(feature = "dhat-heap")]
    let _dhat = dhat::Profiler::new_heap();

    let shell = ShellConfig {
        title: "Minecraft 1.8.9 (mc-rs)".to_owned(),
        width: client.config.width,
        height: client.config.height,
        max_frames: client.config.max_frames,
        ..ShellConfig::default()
    };
    let settings = client.settings.clone();

    // `Shell::run`'s error is the renderer's rich, inline diagnostic type (mc-render allows this
    // lint for the same reason); the closure only passes it through.
    #[allow(clippy::result_large_err)]
    let summary = match run_catching_panics(|| Shell::run(shell, settings, &mut client)) {
        Ok(summary) => summary,
        Err(report) => {
            let mut report = *report;
            mc_core::profile::dump();
            // The frame stopped part-way, so describing and shutting down are best effort: a
            // second panic here must not lose the first one's report. java:
            // `addGraphicsAndWorldToCrashReport` catches its own failures the same way.
            let _ = mc_core::panic_scope::contain(|| client.describe(&mut report));
            let _ = mc_core::panic_scope::contain(|| shutdown_client(&mut client));
            return Err(Box::new(RunFailure::new(report)));
        }
    };
    // The loop has stopped, so the totals are final. No-op unless `MC_PROFILE` is set; placed
    // before the error handling below so a run that ends in a crash still reports where its time
    // went, which is exactly the run you most want the profile for.
    mc_core::profile::dump();

    // A stage of the deferred load failed. `ShellApp::frame` cannot return an error, so it is
    // recorded and the loop asked to stop; this is where it becomes an exit code.
    // java: `Minecraft.crashed(CrashReport)` followed by the `hasCrashed` check at `:422`.
    let load_failure = client
        .fatal_error()
        .map(|error| anyhow::anyhow!("{error}").context("Loading assets"));
    let adapter_facts = match &summary {
        Ok(summary) => summary.adapter_facts.clone(),
        Err(error) => error.adapter_facts().cloned(),
    };

    // java: the `hasCrashed` check at `:422` — a panic on any thread is reported here, with the
    // client's own state attached the way `addGraphicsAndWorldToCrashReport` attaches it. This
    // must happen before shutdown unloads the world and replaces the useful live state.
    // The shell's exit hook leaves the shutdown to this function when a
    // report is coming (`Client::ending_with_a_report`), so the world is still loaded here. Both
    // steps are contained: a second panic must not lose the report being written.
    if let Some(mut report) = crash::take_recorded_crash() {
        let _ = mc_core::panic_scope::contain(|| client.describe(&mut report));
        if let Some(facts) = &adapter_facts {
            report.add_adapter_facts(facts);
        }
        let _ = mc_core::panic_scope::contain(|| shutdown_client(&mut client));
        return Err(Box::new(RunFailure::new(report)));
    }

    if let Some(error) = load_failure {
        let mut report = CrashReport::from_error("Unexpected error", &error);
        let _ = mc_core::panic_scope::contain(|| {
            add_run_error_context(
                &mut report,
                Some(&client),
                adapter_facts.as_ref(),
                &game_dir,
            );
        });
        let _ = mc_core::panic_scope::contain(|| shutdown_client(&mut client));
        return Err(Box::new(RunFailure::new(report)));
    }

    let summary = match summary.context("Rendering the game") {
        Ok(summary) => summary,
        Err(error) => {
            let mut report = CrashReport::from_error("Unexpected error", &error);
            // Contained like the branches above (a lost device ends
            // here).
            let _ = mc_core::panic_scope::contain(|| {
                add_run_error_context(
                    &mut report,
                    Some(&client),
                    adapter_facts.as_ref(),
                    &game_dir,
                );
            });
            let _ = mc_core::panic_scope::contain(|| shutdown_client(&mut client));
            return Err(Box::new(RunFailure::new(report)));
        }
    };
    // java: `Minecraft.run`'s `finally { shutdownMinecraftApplet(); }`. The error branches above
    // do this after enriching their report; the successful path has no report to preserve.
    shutdown_client(&mut client);
    // `ms_per_frame` sits next to `fps` deliberately. FPS is a reciprocal, so at the rates this
    // client runs at it compresses exactly what a frame budget is measured in: 1000 → 2000 fps and
    // 2000 → 4000 fps read as the same doubling and are 0.5 ms and 0.25 ms of reclaimed budget.
    // `frame_time` carries the tail, which no average of either kind can.
    tracing::info!(
        frames = summary.frames,
        ticks = summary.ticks,
        fps = summary.average_fps,
        ms_per_frame = summary.frame_time.mean_ms,
        p99_ms = summary.frame_time.p99_ms,
        max_ms = summary.frame_time.max_ms,
        adapter = summary.adapter,
        present_mode = ?summary.present_mode,
        "stopped"
    );
    // The full distribution goes to stderr rather than into the tracing line: it is nine numbers
    // and it is the thing an A/B run is read off, so it should be one legible row and not a
    // structured-log field soup.
    eprintln!("mc-rs frame time: {}", summary.frame_time);
    // CPU side of input latency per presented frame (`RunSummary::input_to_present`), so the
    // serial and pipelined loops can be compared on it.
    eprintln!("mc-rs input-to-present: {}", summary.input_to_present);
    #[cfg(feature = "dhat-heap")]
    {
        let heap = dhat::HeapStats::get();
        eprintln!(
            "mc-rs allocations: {} blocks / {} bytes in total, {:.1} blocks per frame",
            heap.total_blocks,
            heap.total_bytes,
            heap.total_blocks as f64 / summary.frames.max(1) as f64
        );
    }
    if !summary.gpu.is_empty() {
        eprint!("{}", summary.gpu.format_table());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Spy;

    fn adapter_facts() -> mc_render::AdapterFacts {
        mc_render::AdapterFacts {
            renderer: "Synthetic Renderer".to_owned(),
            vendor: "0x1234".to_owned(),
            backend: "Vulkan".to_owned(),
            driver: "Synthetic Driver (1.2.3)".to_owned(),
            device: "0x5678 (DiscreteGpu)".to_owned(),
        }
    }

    impl CrashReportContext for Spy {
        fn describe_crash(&self, report: &mut CrashReport) {
            report
                .make_category("Affected client")
                .add("Marker", "describe-called");
        }
    }

    #[test]
    fn post_start_error_context_keeps_client_details_before_game_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut report = CrashReport::new("Unexpected error", "TestError: boom");
        let facts = adapter_facts();
        add_run_error_context(&mut report, Some(&Spy), Some(&facts), dir.path());
        let text = report.complete_report();

        let described = text
            .find("-- Affected client --")
            .expect("post-start reports keep client context");
        let system = text
            .find("-- System Details --")
            .expect("system details remain present");
        assert!(
            described < system,
            "client context must precede system details"
        );
        assert!(text.contains("\tMarker: describe-called"), "{text}");
        assert!(text.contains("\tRenderer: Synthetic Renderer"), "{text}");
        assert!(text.contains("\tVendor: 0x1234"), "{text}");
        assert!(text.contains("\tBackend: Vulkan"), "{text}");
        assert!(
            text.contains("\tDriver: Synthetic Driver (1.2.3)"),
            "{text}"
        );
        assert!(text.contains("\tDevice: 0x5678 (DiscreteGpu)"), "{text}");
        assert!(text.contains(&format!("\tGame directory: {}", dir.path().display())));
        assert!(!text.contains("-- Initialization --"), "{text}");
    }

    /// A panic escaping the event loop comes back as the hook's report
    /// (written to `crash-reports/` by `main`), and a normal loop's value passes through. The
    /// negative control is the old code: nothing caught the panic, so the process ended with 101
    /// and the recorded report was never taken.
    #[test]
    fn a_panic_escaping_the_event_loop_becomes_the_recorded_report() {
        crash::install_panic_hook();
        let _ = crash::take_recorded_crash();
        assert_eq!(run_catching_panics(|| 7).ok(), Some(7));

        let report = run_catching_panics(|| -> u32 { panic!("a frame exploded (test)") })
            .expect_err("the panic is caught");
        let text = report.complete_report();
        assert!(text.contains("a frame exploded (test)"), "{text}");
        assert!(
            !crash::has_crashed(),
            "the report was taken, not left for later"
        );
        let _ = std::panic::take_hook();
    }

    #[test]
    fn boot_error_context_has_initialization_category_and_game_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut report = CrashReport::new("Initializing game", "AssetError: missing");
        add_run_error_context::<Spy>(&mut report, None, None, dir.path());
        let text = report.complete_report();

        assert!(text.contains("-- Initialization --\nDetails:"), "{text}");
        assert!(text.contains(&format!("\tGame directory: {}", dir.path().display())));
        for row in ["Renderer", "Vendor", "Backend", "Driver", "Device"] {
            assert!(!text.contains(&format!("\t{row}:")), "{text}");
        }
    }
}
