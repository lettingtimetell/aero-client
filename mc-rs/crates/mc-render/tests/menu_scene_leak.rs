//! The leak probe for the Aero title background (aero-title 2026-11-07 follow-up).
//!
//! Ignored: one "visit" per cycle on a headless 256x192 GPU (the client leak test's size), then
//! the working set and private commit (`GetProcessMemoryInfo`) and the bytes live in Rust's heap
//! (a counting global allocator) per cycle.
//!
//! ```text
//! LEAK_MODE=keep cargo test -p mc-render --test menu_scene_leak -- --ignored --nocapture
//! ```
//!
//! `LEAK_MODE`: `scene` (build, render 4 frames, drop: the old per-visit way), `keep` (one
//! scene, phases in and out: the client's way now), `release` (`keep` plus
//! `release_gpu_memory` every cycle), `nodraw`, `frames1`, raw `buffer` / `shadow` replicas,
//! `empty` (the noise floor). `LEAK_CYCLES`, `LEAK_FRAMES` (frames per visit), `LEAK_ALLOC=1`
//! (print wgpu's allocator report), `LEAK_VERBOSE=1` (timestamps).
//!
//! What it found (2026-11-07): `scene` grew private commit by ~23 MB over the first ~14 visits
//! and then levelled off, with wgpu's allocator report and the Rust heap unchanged every visit:
//! a warm-up in the GPU driver's own memory, set off by building and freeing the scene's GPU
//! objects per visit. `keep` warms up once and stays flat, hence the client parks the scene.

use std::path::{Path, PathBuf};

use mc_render::menu_scene::{self, MenuScene};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, RenderError};

fn jar() -> Option<PathBuf> {
    let jars = std::env::var_os("MC_JARS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .ancestors()
                .nth(3)
                .expect("repository root")
                .join("MCP-919-main")
                .join("jars")
        });
    let jar = jars.join("versions").join("1.8.9").join("1.8.9.jar");
    jar.is_file().then_some(jar)
}

/// Counts the bytes live in Rust's allocator, so growth can be split into "Rust heap" (wgpu,
/// naga, this crate) and "everything else" (the GPU driver's own allocations).
struct Counting;

static LIVE: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

// SAFETY: forwards every call to the system allocator unchanged and only counts sizes.
unsafe impl std::alloc::GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        let p = unsafe { std::alloc::System.alloc(layout) };
        if !p.is_null() {
            LIVE.fetch_add(layout.size() as i64, std::sync::atomic::Ordering::Relaxed);
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        unsafe { std::alloc::System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size() as i64, std::sync::atomic::Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new: usize) -> *mut u8 {
        let p = unsafe { std::alloc::System.realloc(ptr, layout, new) };
        if !p.is_null() {
            LIVE.fetch_add(
                new as i64 - layout.size() as i64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        p
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// `PROCESS_MEMORY_COUNTERS`.
#[cfg(target_os = "windows")]
#[repr(C)]
#[derive(Default)]
struct ProcessMemoryCounters {
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
}

#[cfg(target_os = "windows")]
#[link(name = "kernel32")]
extern "system" {
    fn GetCurrentProcess() -> isize;
    fn K32GetProcessMemoryInfo(
        process: isize,
        counters: *mut ProcessMemoryCounters,
        cb: u32,
    ) -> i32;
}

/// (working set, private commit) in bytes, read directly (no subprocess).
#[cfg(target_os = "windows")]
fn memory() -> (u64, u64) {
    let mut c = ProcessMemoryCounters {
        cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
        ..ProcessMemoryCounters::default()
    };
    // SAFETY: a plain Win32 query into a correctly sized, initialised struct.
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    assert_ne!(ok, 0, "GetProcessMemoryInfo failed");
    (c.working_set_size as u64, c.pagefile_usage as u64)
}

#[cfg(not(target_os = "windows"))]
fn memory() -> (u64, u64) {
    let rss = resident().unwrap_or(0);
    (rss, rss)
}

/// The working set in bytes, the way `mc_client::app::crash::process_memory_bytes` reads it.
#[allow(dead_code)]
fn resident() -> Option<u64> {
    #[cfg(target_os = "windows")]
    {
        let pid = std::process::id();
        let output = std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
            .output()
            .ok()?;
        let text = String::from_utf8_lossy(&output.stdout);
        let field = text.rsplit('"').nth(1)?;
        let digits: String = field.chars().filter(char::is_ascii_digit).collect();
        digits.parse::<u64>().ok().map(|kib| kib * 1024)
    }
    #[cfg(not(target_os = "windows"))]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        let line = status.lines().find(|l| l.starts_with("VmRSS:"))?;
        let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kib * 1024)
    }
}

fn frame(gpu: &mut Gpu, phases: &mut PhaseList) {
    let target = gpu.acquire().expect("offscreen frame");
    phases.render(gpu, target.view(), &Camera::default(), 0.0);
}

#[test]
#[ignore = "resident-set probe; run by hand"]
fn menu_scene_cycles() {
    let Some(jar) = jar() else {
        eprintln!("skip: no 1.8.9 jar");
        return;
    };
    let mut gpu = match Gpu::headless(
        256,
        192,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(g) => g,
        Err(RenderError::NoAdapter) => return,
        Err(e) => panic!("headless context: {e}"),
    };
    gpu.device.on_uncaptured_error(std::sync::Arc::new(|e| {
        panic!("wgpu validation error: {e}")
    }));
    let mode = std::env::var("LEAK_MODE").unwrap_or_else(|_| "scene".to_owned());
    let cycles: usize = std::env::var("LEAK_CYCLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(15);
    // Bake and mesh once, outside the measurement, as the client's prewarm does.
    menu_scene::prepare_stats(&jar).expect("prepare");
    let mut phases = PhaseList::new();
    let mut samples = Vec::new();
    let mut rust_live: Vec<i64> = Vec::new();
    let verbose = std::env::var_os("LEAK_VERBOSE").is_some();
    let started = std::time::Instant::now();
    let stamp = |what: &str, cycle: usize| {
        if verbose {
            eprintln!(
                "{:8.3}s cycle {cycle}: {what}",
                started.elapsed().as_secs_f64()
            );
        }
    };
    // Raw replicas of each per-visit resource, for bisecting.
    let vertex_bytes = vec![7u8; 127_677 * 4 * 32];
    // `keep`: one scene for the whole run, only its phases come and go (the client's way since
    // the leak follow-up); `release`: the same, but its big GPU memory is released every cycle.
    let mut kept = matches!(mode.as_str(), "keep" | "release")
        .then(|| MenuScene::new(&gpu, &jar).expect("menu scene"));
    for cycle in 0..3 + cycles {
        stamp("new", cycle);
        match mode.as_str() {
            "scene" | "nodraw" | "frames1" | "keep" | "release" => {
                let mut fresh = None;
                let scene = match kept.as_mut() {
                    Some(scene) => scene,
                    None => fresh.insert(MenuScene::new(&gpu, &jar).expect("menu scene")),
                };
                stamp("built", cycle);
                if mode != "nodraw" {
                    phases.insert(menu_scene::BACKGROUND_ORDER, scene.background_phase());
                    phases.insert(menu_scene::FINISH_ORDER, scene.finish_phase());
                    let frames = if mode == "frames1" {
                        1
                    } else {
                        std::env::var("LEAK_FRAMES")
                            .ok()
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(4)
                    };
                    for i in 0..frames {
                        scene.update(4.0 + f64::from(i) * 0.1, 256, 192);
                        frame(&mut gpu, &mut phases);
                    }
                    phases.remove_named(menu_scene::BACKGROUND_PHASE_NAME);
                    phases.remove_named(menu_scene::FINISH_PHASE_NAME);
                }
                if mode == "release" {
                    scene.release_gpu_memory();
                    assert!(scene.gpu_memory_released());
                }
                drop(fresh);
            }
            "buffer" => {
                let buffer = wgpu::util::DeviceExt::create_buffer_init(
                    &gpu.device,
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("probe vertices"),
                        contents: &vertex_bytes,
                        usage: wgpu::BufferUsages::VERTEX,
                    },
                );
                frame(&mut gpu, &mut phases);
                drop(buffer);
            }
            "shadow" => {
                let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("probe shadow"),
                    size: wgpu::Extent3d {
                        width: 4096,
                        height: 4096,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth16Unorm,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                let mut enc = gpu
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                drop(enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                }));
                gpu.queue.submit([enc.finish()]);
                frame(&mut gpu, &mut phases);
                drop((view, texture));
            }
            _ => {
                for _ in 0..4 {
                    frame(&mut gpu, &mut phases);
                }
            }
        }
        // The client keeps rendering after leaving the title screen.
        for _ in 0..6 {
            frame(&mut gpu, &mut phases);
        }
        stamp("frames done", cycle);
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        stamp("polled", cycle);
        if cycle >= 2 {
            samples.push(memory());
            rust_live.push(LIVE.load(std::sync::atomic::Ordering::Relaxed));
        }
        if std::env::var_os("LEAK_ALLOC").is_some() {
            if let Some(report) = gpu.device.generate_allocator_report() {
                let names: std::collections::BTreeMap<&str, usize> = report
                    .allocations
                    .iter()
                    .fold(Default::default(), |mut m, a| {
                        *m.entry(a.name.as_str()).or_default() += 1;
                        m
                    });
                eprintln!(
                    "cycle {cycle}: gpu-allocator {} allocations, {} blocks, {:.1} MB allocated, {:.1} MB reserved; by name {names:?}",
                    report.allocations.len(),
                    report.blocks.len(),
                    report.total_allocated_bytes as f64 / 1_048_576.0,
                    report.total_reserved_bytes as f64 / 1_048_576.0,
                );
            }
            let c = gpu.device.get_internal_counters();
            eprintln!("cycle {cycle}: counters {:?}", c.hal);
        }
        stamp("sampled", cycle);
    }
    let live_deltas: Vec<i64> = rust_live.windows(2).map(|w| (w[1] - w[0]) / 1024).collect();
    println!(
        "mode {mode} rust heap live: {:.1} MB -> {:.1} MB = {:.0} KB/cycle; per-cycle KB: {live_deltas:?}",
        rust_live[0] as f64 / 1_048_576.0,
        *rust_live.last().expect("samples") as f64 / 1_048_576.0,
        (rust_live.last().expect("samples") - rust_live[0]) as f64 / cycles as f64 / 1024.0,
    );
    for (what, pick) in [("working set", 0usize), ("private", 1)] {
        let value = |s: &(u64, u64)| if pick == 0 { s.0 } else { s.1 };
        let first = value(&samples[0]);
        let last = value(samples.last().expect("samples"));
        let per_cycle = (last as f64 - first as f64) / cycles as f64 / 1024.0;
        let deltas: Vec<i64> = samples
            .windows(2)
            .map(|w| (value(&w[1]) as i64 - value(&w[0]) as i64) / 1024)
            .collect();
        println!(
            "mode {mode} {what}: {:.1} MB -> {:.1} MB over {cycles} cycles = {per_cycle:.0} KB/cycle; per-cycle KB: {deltas:?}",
            first as f64 / 1_048_576.0,
            last as f64 / 1_048_576.0,
        );
    }
    stamp("test body done", 0);
}
