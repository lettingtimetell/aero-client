//! Render-side memory.
//!
//! Switches (environment variables, read once; `=0` restores the earlier behaviour exactly):
//! - [`LAZY_OUTLINE_ENV`]: the spectator outline's screen-sized targets are made the first time
//!   the outline is drawn instead of at install (about 25 MiB of GPU memory at 1080p that most
//!   sessions never use).
//! - [`SCENE_RELEASE_ENV`]: the frame's scene snapshot is dropped after a few seconds without a
//!   phase that reads it (it stayed for the whole session once the title screen made it).
//! - [`GLASS_RELEASE_ENV`]: the mod menu's glass drops its blur chain when the menu closes.
//! - [`SHARED_SHADOW_ENV`]: the block-atlas owners that share one GPU texture share one raw CPU
//!   copy of it too (five copies of the atlas mip chain before).
//! - [`TRIM_ENV`]: long-lived buffers built once are sized exactly (the greenhouse's vertices).
//! - [`SMALL_FIRST_PAGE_ENV`]: the terrain vertex arena's first page starts at 32 MiB where it
//!   can grow, instead of 128 MiB (a SkyWars map holds about 11 MiB of terrain).
//! - [`GPU_BLOCKS_ENV`]: the size of the GPU allocator's memory blocks (see there).
//!
//! `start_memory_report` is a measurement aid. With [`MEMDUMP_ENV`] set to a folder, a
//! background thread appends the process's memory counters and the GPU allocator's report,
//! grouped by resource label, to `<folder>/mr-mem-<pid>.txt` every [`MEMDUMP_SECS_ENV`] seconds
//! (default 1). Unset (always, for players): the variable is read once when the device is created
//! and nothing else happens; no frame does any work for it.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// `MC_W21_MR_LAZY_OUTLINE=0`: build the spectator outline's mask and post-chain targets at the
/// window size when the entity phase installs them, as before. On (default): they are built at
/// 1x1 and grown to the window size the first frame the outline is drawn (spectator mode with
/// the highlight key held), then kept.
pub const LAZY_OUTLINE_ENV: &str = "MC_W21_MR_LAZY_OUTLINE";

/// Whether [`LAZY_OUTLINE_ENV`] is on (default). Read once.
#[must_use]
pub fn lazy_outline_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| switch_on(LAZY_OUTLINE_ENV))
}

/// `MC_W21_MR_GLASS_RELEASE=0`: the mod menu's glass keeps its screen-sized blur chain after the
/// menu closes, as before. On (default): the chain is dropped when the menu closes and made
/// again on the next open (every level is rewritten from the scene each frame before the glass
/// samples it, so the picture is the same).
pub const GLASS_RELEASE_ENV: &str = "MC_W21_MR_GLASS_RELEASE";

/// Whether [`GLASS_RELEASE_ENV`] is on (default). Read once.
#[must_use]
pub fn glass_release_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| switch_on(GLASS_RELEASE_ENV))
}

/// `MC_W21_MR_SMALL_FIRST_PAGE=0`: the terrain vertex arena opens its first page at the full
/// page size (128 MiB), as before. On (default), where that page can grow (native multi-draw) it
/// opens at 32 MiB and doubles as the world fills it; see `terrain::phase`.
pub const SMALL_FIRST_PAGE_ENV: &str = "MC_W21_MR_SMALL_FIRST_PAGE";

/// Whether [`SMALL_FIRST_PAGE_ENV`] is on (default). Read once.
#[must_use]
pub fn small_first_page_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| switch_on(SMALL_FIRST_PAGE_ENV))
}

/// `MC_W21_MR_TRIM=0`: keep the growth slack of long-lived buffers built once (the title
/// screen greenhouse's vertices), as before. On (default): they are sized exactly. Same bytes.
pub const TRIM_ENV: &str = "MC_W21_MR_TRIM";

/// Whether [`TRIM_ENV`] is on (default). Read once.
#[must_use]
pub fn trim_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| switch_on(TRIM_ENV))
}

/// `MC_W21_MR_SHARED_SHADOW=0`: every block-atlas owner keeps its own raw copy of the atlas mip
/// chain (its anaglyph "shadow"), as before. On (default): the owners that share one GPU atlas
/// texture (`atlas_transfer::share_atlas_shadow`) share one shadow too.
pub const SHARED_SHADOW_ENV: &str = "MC_W21_MR_SHARED_SHADOW";

/// Whether [`SHARED_SHADOW_ENV`] is on (default). Read once.
#[must_use]
pub fn shared_shadow_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| switch_on(SHARED_SHADOW_ENV))
}

/// `MC_W21_MR_SCENE_RELEASE=0`: keep the frame's scene snapshot (a screen-sized texture made the
/// first time a phase reads the scene: the title screen's glass and finish, the mod menu's
/// glass, menu blur, post effects) for good, as before. On (default): it is dropped after
/// [`SCENE_RELEASE_IDLE_FRAMES`] frames in a row without such a phase, and made again, exactly
/// as the first time, when one returns.
pub const SCENE_RELEASE_ENV: &str = "MC_W21_MR_SCENE_RELEASE";

/// Frames in a row without a scene-reading phase after which the snapshot is dropped: a few
/// seconds at 60 fps, well under a second uncapped. Long enough that closing and reopening a
/// menu does not remake it every time.
pub const SCENE_RELEASE_IDLE_FRAMES: u32 = 240;

/// Whether [`SCENE_RELEASE_ENV`] is on (default). Read once.
#[must_use]
pub fn scene_release_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| switch_on(SCENE_RELEASE_ENV))
}

/// Per frame: `used` says whether this frame uses `held`. A frame that
/// uses it resets `idle`; otherwise, while something is held, `idle` counts up and at
/// `idle_limit` the value is dropped if `releasable` agrees (and the switch is on). With nothing
/// held (the usual case in game) this is one `Option` test.
pub(crate) fn release_when_idle<T>(
    used: bool,
    held: &mut Option<T>,
    idle: &mut u32,
    idle_limit: u32,
    releasable: impl FnOnce(&T) -> bool,
) {
    if used {
        *idle = 0;
        return;
    }
    let Some(value) = held.as_ref() else {
        return;
    };
    *idle = idle.saturating_add(1);
    if *idle >= idle_limit && scene_release_enabled() && releasable(value) {
        *held = None;
        *idle = 0;
    }
}

/// `MC_W21_MR_GPU_BLOCKS`: how big the blocks are that the GPU memory allocator reserves and
/// sub-allocates resources from. Unset, `perf` or `0`: wgpu's `MemoryHints::Performance`, as
/// before (device blocks of 128-256 MiB, host-visible blocks of 64-128 MiB). `usage`:
/// `MemoryHints::MemoryUsage` (8-64 MiB and 4-32 MiB). `<min>-<max>` (MiB, e.g. `16-64`):
/// `MemoryHints::Manual` with those device block sizes (host blocks half as big). Read once,
/// when the device is created.
///
/// Every reserved block counts against the process's commit charge as well as the GPU's memory.
/// `usage` starts far lower (the SkyWars bench reserved ~210 MiB instead of 448 MiB) but frees
/// and re-makes small blocks on every server switch, and the driver kept that commit: over a
/// few reconnects it climbed back to about the `perf` level (the reconnect harness), so
/// `perf` stays the default. Most of the in-game difference came from the terrain arena's first
/// page forcing a second 256 MiB block, which [`SMALL_FIRST_PAGE_ENV`] removes with big blocks.
pub const GPU_BLOCKS_ENV: &str = "MC_W21_MR_GPU_BLOCKS";

/// The allocator hints for a new device (see [`GPU_BLOCKS_ENV`]).
#[must_use]
pub fn memory_hints() -> wgpu::MemoryHints {
    let value = std::env::var(GPU_BLOCKS_ENV).unwrap_or_default();
    parse_memory_hints(value.trim()).unwrap_or(wgpu::MemoryHints::Performance)
}

/// [`GPU_BLOCKS_ENV`]'s value, or `None` for an empty or unreadable one (the default).
fn parse_memory_hints(value: &str) -> Option<wgpu::MemoryHints> {
    const MIB: u64 = 1024 * 1024;
    match value {
        "0" | "perf" => Some(wgpu::MemoryHints::Performance),
        "usage" => Some(wgpu::MemoryHints::MemoryUsage),
        _ => {
            let (min, max) = value.split_once('-')?;
            let min = min.trim().parse::<u64>().ok()?.checked_mul(MIB)?;
            let max = max.trim().parse::<u64>().ok()?.checked_mul(MIB)?;
            (min > 0 && min <= max).then_some(wgpu::MemoryHints::Manual {
                suballocated_device_memory_block_size: min..max,
            })
        }
    }
}

/// A switch is on unless its variable is exactly `0` (surrounding whitespace ignored).
fn switch_on(name: &str) -> bool {
    !matches!(std::env::var(name).as_deref().map(str::trim), Ok("0"))
}

/// The folder the memory report goes to (unset or empty: no report).
pub const MEMDUMP_ENV: &str = "MC_W21_MR_MEMDUMP";
/// Seconds between two reports (default 1, at least 0.25).
pub const MEMDUMP_SECS_ENV: &str = "MC_W21_MR_MEMDUMP_SECS";

/// Start the report thread for `device` when [`MEMDUMP_ENV`] names a folder. At most one per
/// process (the first device wins). The thread keeps a handle to the device and runs until the
/// process exits; it only exists in measurement runs.
pub(crate) fn start_memory_report(device: &wgpu::Device) {
    static STARTED: AtomicBool = AtomicBool::new(false);
    let Some(dir) = std::env::var_os(MEMDUMP_ENV).filter(|v| !v.is_empty()) else {
        return;
    };
    if STARTED.swap(true, Ordering::Relaxed) {
        return;
    }
    let secs = std::env::var(MEMDUMP_SECS_ENV)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|s| s.is_finite())
        .unwrap_or(1.0)
        .max(0.25);
    let path = PathBuf::from(dir).join(format!("mr-mem-{}.txt", std::process::id()));
    let device = device.clone();
    let started = Instant::now();
    let spawned = std::thread::Builder::new()
        .name("mr-memory-report".to_owned())
        .spawn(move || loop {
            std::thread::sleep(Duration::from_secs_f64(secs));
            let text = report(&device, started.elapsed());
            let written = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .and_then(|mut file| file.write_all(text.as_bytes()));
            if let Err(error) = written {
                tracing::warn!(%error, "memory report: could not write; stopping");
                return;
            }
        });
    if let Err(error) = spawned {
        tracing::warn!(%error, "memory report: could not start its thread");
    }
}

/// One record: a header line with the process counters and the allocator's totals, then one
/// line per resource label (bytes, allocations), largest first.
fn report(device: &wgpu::Device, elapsed: Duration) -> String {
    let mut out = String::new();
    let process = process_memory();
    let _ = write!(
        out,
        "@ t={:.3} working_set={} private={} peak_working_set={}",
        elapsed.as_secs_f64(),
        process.working_set,
        process.private,
        process.peak_working_set
    );
    let Some(gpu) = device.generate_allocator_report() else {
        out.push_str(" gpu=unavailable\n");
        return out;
    };
    let _ = writeln!(
        out,
        " gpu_allocated={} gpu_reserved={} gpu_allocations={} gpu_blocks={}",
        gpu.total_allocated_bytes,
        gpu.total_reserved_bytes,
        gpu.allocations.len(),
        gpu.blocks.len()
    );
    // One line per allocator block: its size and the largest allocation in it, so device-local
    // and host-visible blocks can be told apart by what they hold.
    for block in &gpu.blocks {
        let largest = gpu
            .allocations
            .get(block.allocations.clone())
            .and_then(|list| list.iter().max_by_key(|allocation| allocation.size));
        let _ = writeln!(
            out,
            "  block {:>12} {:>6}  {}",
            block.size,
            block.allocations.len(),
            largest.map_or("", |allocation| allocation.name.as_str())
        );
    }
    let mut by_label: HashMap<&str, (u64, u32)> = HashMap::new();
    for allocation in &gpu.allocations {
        let entry = by_label.entry(allocation.name.as_str()).or_default();
        entry.0 += allocation.size;
        entry.1 += 1;
    }
    let mut rows: Vec<(&str, (u64, u32))> = by_label.into_iter().collect();
    rows.sort_by(|a, b| b.1 .0.cmp(&a.1 .0).then(a.0.cmp(b.0)));
    for (label, (bytes, count)) in rows {
        let _ = writeln!(out, "  {bytes:>12} {count:>6}  {label}");
    }
    out
}

/// The process's working set and private commit, in bytes.
#[derive(Debug, Clone, Copy, Default)]
struct ProcessMemory {
    working_set: u64,
    private: u64,
    peak_working_set: u64,
}

#[cfg(windows)]
fn process_memory() -> ProcessMemory {
    /// `PROCESS_MEMORY_COUNTERS_EX`.
    #[repr(C)]
    #[derive(Default)]
    struct Counters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
        private_usage: usize,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(process: isize, counters: *mut Counters, cb: u32) -> i32;
    }

    let mut counters = Counters {
        cb: std::mem::size_of::<Counters>() as u32,
        ..Counters::default()
    };
    // SAFETY: a plain Win32 query into a correctly sized, initialised struct that outlives the
    // call; the pseudo-handle of the current process needs no closing.
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    if ok == 0 {
        return ProcessMemory::default();
    }
    ProcessMemory {
        working_set: counters.working_set_size as u64,
        private: counters.private_usage as u64,
        peak_working_set: counters.peak_working_set_size as u64,
    }
}

#[cfg(not(windows))]
fn process_memory() -> ProcessMemory {
    ProcessMemory::default()
}

/// Test helpers shared by this lane's tests in other modules.
#[cfg(test)]
pub(crate) mod test_support {
    /// A headless context for memory tests, or `None` (printed as a skip) without an adapter.
    pub(crate) fn gpu() -> Option<crate::Gpu> {
        match crate::Gpu::headless(8, 8, crate::GpuOptions::default()) {
            Ok(gpu) => Some(gpu),
            Err(error) => {
                eprintln!("SKIP: no headless GPU ({error})");
                None
            }
        }
    }

    /// The vanilla jar as a one-pack resource manager, or `None` (printed as a skip).
    pub(crate) fn vanilla_resources() -> Option<mc_assets::ResourceManager> {
        let dir = match std::env::var_os("MC_JARS_DIR") {
            Some(dir) => std::path::PathBuf::from(dir),
            None => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .ancestors()
                .nth(3)?
                .join("MCP-919-main")
                .join("jars"),
        };
        let jar = dir.join("versions").join("1.8.9").join("1.8.9.jar");
        if !jar.is_file() {
            eprintln!("SKIP: {} not found; set MC_JARS_DIR", jar.display());
            return None;
        }
        let mut resources = mc_assets::ResourceManager::new();
        resources.push_pack(Box::new(
            mc_assets::ZipResourcePack::open(&jar).expect("open 1.8.9.jar"),
        ));
        Some(resources)
    }

    /// Bytes the device's allocator holds for resources labelled `label`, or `None` when the
    /// backend gives no report.
    pub(crate) fn allocated_bytes(device: &wgpu::Device, label: &str) -> Option<u64> {
        let report = device.generate_allocator_report()?;
        Some(
            report
                .allocations
                .iter()
                .filter(|allocation| allocation.name == label)
                .map(|allocation| allocation.size)
                .sum(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `MC_W21_MR_GPU_BLOCKS` values: the two presets, a MiB range, and anything else ignored
    /// (the default applies).
    #[test]
    fn the_gpu_blocks_switch_reads_presets_and_mib_ranges() {
        assert!(matches!(
            parse_memory_hints("perf"),
            Some(wgpu::MemoryHints::Performance)
        ));
        assert!(matches!(
            parse_memory_hints("0"),
            Some(wgpu::MemoryHints::Performance)
        ));
        assert!(matches!(
            parse_memory_hints("usage"),
            Some(wgpu::MemoryHints::MemoryUsage)
        ));
        match parse_memory_hints("16-64") {
            Some(wgpu::MemoryHints::Manual {
                suballocated_device_memory_block_size: range,
            }) => assert_eq!(range, (16 << 20)..(64 << 20)),
            other => panic!("{other:?}"),
        }
        for bad in [
            "",
            "fast",
            "64-16",
            "0-8",
            "8-",
            "-8",
            "99999999999999-99999999999999",
        ] {
            assert!(parse_memory_hints(bad).is_none(), "{bad}");
        }
    }

    /// The report reads real counters and groups the allocator's report by label: a labelled
    /// 1 MiB buffer shows up as its own row.
    #[test]
    fn the_report_lists_a_labelled_buffer_under_its_label() {
        let Ok(gpu) = crate::Gpu::headless(8, 8, crate::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter");
            return;
        };
        let _buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("w21-mr report probe"),
            size: 1 << 20,
            usage: wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });
        let text = report(&gpu.device, Duration::from_secs(1));
        let first = text.lines().next().expect("a header line");
        assert!(first.starts_with("@ t=1.000 working_set="), "{first}");
        if gpu.device.generate_allocator_report().is_none() {
            eprintln!("skip: this backend has no allocator report");
            return;
        }
        let row = text
            .lines()
            .find(|line| {
                !line.trim_start().starts_with("block ") && line.ends_with("  w21-mr report probe")
            })
            .unwrap_or_else(|| panic!("no row for the probe buffer:\n{text}"));
        let bytes: u64 = row.split_whitespace().next().unwrap().parse().unwrap();
        assert!(bytes >= 1 << 20, "{row}");
        #[cfg(windows)]
        {
            let memory = process_memory();
            assert!(memory.working_set > 0 && memory.private > 0, "{memory:?}");
        }
    }
}
