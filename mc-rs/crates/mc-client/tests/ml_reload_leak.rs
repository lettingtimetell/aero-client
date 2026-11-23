//! A resource-pack reload must not leak a baked-asset set.
//!
//! `mc_assets::model::HotSwap` publishes the block and item model registries at `&'static`
//! lifetime, so a reload cannot free the generation it replaces. Before this change every reload
//! leaked one more set: the leak harness (`tests/ml_leak_harness.rs`, scenario `packs`)
//! measured **+13.1 MB of live heap and +28,600 blocks per reload, every reload**, about 12 MB of
//! it the baked models, item models and sprites, plus two 256 KiB colormaps. A server resource
//! pack reloads on join and again on leave, so that was ~26 MB per visit to such a server.
//!
//! Now a reload with the inputs of a bake published before (the bake cache's own identity of a
//! clean bake) publishes that bake again, and equal colormaps are reused the same way: the leak
//! is one set per distinct pack stack, not one per reload. This test reloads the same stack
//! several times and checks both the registries' published counts and the live heap.
//!
//! Its own binary: the registries are process-wide, and another test building a client in the
//! same process would publish generations of its own in the middle of the measurement.

mod common;
mod ml_leak_support;

use std::time::Duration;

use common::{config, gpu, pump_no_tick};
use mc_client::app::Client;
use mc_render::{Camera, PhaseList};
use ml_leak_support::TracingAlloc;

#[global_allocator]
static GLOBAL: TracingAlloc = TracingAlloc;

const WIDTH: u32 = 256;
const HEIGHT: u32 = 192;
const RELOADS: usize = 4;

/// Live-heap growth allowed over [`RELOADS`] reloads of the same packs. One leaked set is about
/// 12 MB, so a single leaked reload already fails this; what remains is per-reload churn that
/// settles (texture CPU copies replaced, caches refilled), measured at under 1 MB.
const MAX_GROWTH_BYTES: i64 = 6 * 1024 * 1024;

#[test]
fn reloading_the_same_packs_reuses_the_published_bake() {
    let Some(mut gpu) = gpu(WIDTH, HEIGHT) else {
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let Some(config) = config(dir.path(), "Dev") else {
        eprintln!("skip: no vanilla jar in this checkout");
        return;
    };
    let mut client = Client::new(config, &mut |_, _| {}).expect("start");
    let mut phases = PhaseList::new();
    client.install(&gpu, &mut phases, WIDTH, HEIGHT);
    let mut camera = Camera::default();
    let mut frames = |client: &mut Client, gpu: &mut mc_render::Gpu, phases: &mut PhaseList| {
        for _ in 0..3 {
            pump_no_tick(client, gpu, phases, &mut camera);
            std::thread::sleep(Duration::from_millis(30));
        }
    };
    frames(&mut client, &mut gpu, &mut phases);

    // The first reload of a fresh game directory may bake (and cache) where start-up read the
    // cache, or the reverse; from then on the inputs are fixed.
    client
        .reload_resource_packs(Some(&gpu), Some(&mut phases), &mut |_, _| {})
        .expect("reload");
    frames(&mut client, &mut gpu, &mut phases);
    let generation = mc_assets::model::installed_generation();
    let bakes = mc_assets::model::distinct_generations();
    let colormaps = mc_render::terrain::tint::distinct_colormap_sets();
    let live = ml_leak_support::live_bytes();

    for _ in 0..RELOADS {
        client
            .reload_resource_packs(Some(&gpu), Some(&mut phases), &mut |_, _| {})
            .expect("reload");
        frames(&mut client, &mut gpu, &mut phases);
    }
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let growth = ml_leak_support::live_bytes() - live;
    println!(
        "{RELOADS} reloads: live heap {:+.1} KB; baked sets published {bakes} -> {}, colormap \
         sets {colormaps} -> {}",
        growth as f64 / 1024.0,
        mc_assets::model::distinct_generations(),
        mc_render::terrain::tint::distinct_colormap_sets(),
    );

    // Every reload still took effect (the generation moved), exactly as before...
    assert_eq!(
        mc_assets::model::installed_generation(),
        generation + RELOADS as u64,
        "each reload must still publish a generation"
    );
    // ...but none of them published (and leaked) another copy of the same bake.
    assert_eq!(
        mc_assets::model::distinct_generations(),
        bakes,
        "reloading the same packs leaked another baked-asset set"
    );
    assert_eq!(
        mc_render::terrain::tint::distinct_colormap_sets(),
        colormaps,
        "reloading the same packs leaked another pair of colormaps"
    );
    assert!(
        growth < MAX_GROWTH_BYTES,
        "the live heap grew {growth} bytes over {RELOADS} reloads of the same packs"
    );
}
