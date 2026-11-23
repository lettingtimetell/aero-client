//! S18 acceptance tests: real GPU rendering (headless readback) and a throughput check.
//!
//! Everything CPU-side (spawn formulas, tick physics, collision, packet decode, density gating)
//! has unit/integration coverage beside the code in `src/particle/**`. This file is for the
//! things that only a real `wgpu` device can prove: that a billboard actually lands where the
//! math says it should, that the two alpha buckets really do differ in `depthMask` the way
//! `EffectRenderer.renderParticles` does, and that batching by texture really does collapse many
//! particles into one draw call. It also loads the real vanilla jar (skipping cleanly when it is
//! not on disk) for one end-to-end texture-asset test, mirroring `s11_mesher.rs`'s precedent.
//!
//! Every GPU test skips itself (with a printed reason) when this machine has no adapter, per the
//! same convention `tests/headless.rs` uses.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use glam::DVec3;
use mc_math::{BlockPos, ChunkPos, JavaRandom};
use mc_render::particle::{
    phase::ParticlePhase,
    spawn,
    system::ParticleSystem,
    types::{Particle, ParticleIcon},
};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_world::{SkyBlock, World};

fn gpu(width: u32, height: u32) -> Option<Gpu> {
    match Gpu::headless(
        width,
        height,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(gpu) => Some(gpu),
        Err(RenderError::NoAdapter) => {
            eprintln!("skip: no wgpu adapter available on this machine");
            None
        }
        Err(e) => panic!("headless context creation failed: {e}"),
    }
}

fn pixel(pixels: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let i = ((y * width + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
}

fn close(actual: [u8; 4], expected: [u8; 4], tolerance: u8) -> bool {
    actual
        .iter()
        .zip(expected.iter())
        .all(|(a, b)| a.abs_diff(*b) <= tolerance)
}

fn solid_atlas(w: u32, h: u32, rgba: [u8; 4]) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for _ in 0..(w * h) {
        out.extend_from_slice(&rgba);
    }
    out
}

/// Build the full-bright resident column these colour/blend probes need.  `World::new()` has no
/// loaded chunks, and Java's `Entity.getBrightnessForRender` consequently returns packed light
/// zero; that is the right live-world result but would make a red/green blend assertion measure
/// the lightmap's dark texel instead of the particle pipeline.  Keep the scene's lighting explicit
/// so the expected colours exercise billboard output and alpha ordering.
fn full_bright_world() -> World {
    let mut world = World::new();
    world.load_empty_chunk(ChunkPos::new(0, 0));
    for z in 0..16 {
        let pos = BlockPos::new(0, 0, z);
        world.set_light_for(SkyBlock::Sky, pos, 15);
        world.set_light_for(SkyBlock::Block, pos, 15);
    }
    world
}

// ---- billboard placement ------------------------------------------------------------------

#[test]
fn a_particle_billboard_renders_at_its_world_position_and_camera_facing_colour() {
    let Some(mut gpu) = gpu(64, 64) else { return };
    let mut phase = ParticlePhase::new(&gpu);
    phase.install_misc_atlas(&gpu, &solid_atlas(16, 16, [255, 0, 0, 255]), 16, 16);

    let world = full_bright_world();
    let cam = Camera::default();
    let mut rng = JavaRandom::new(1);
    let mut p = spawn::generic(&mut rng, cam.pos + cam.look_vec() * 5.0, DVec3::ZERO);
    p.motion = DVec3::ZERO;
    p.scale = 8.0; // large enough to cover the centre pixel regardless of exact FOV rounding

    phase.upload(&gpu, &[p], &cam, 0.0, &world);

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 0.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    }; // blue background
    phases.insert(PhaseOrder::PARTICLES, Box::new(phase));

    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &cam, 0.0);
    let pixels = gpu.read_pixels().expect("readback");

    let center = pixel(&pixels, 64, 32, 32);
    assert!(
        close(center, [255, 0, 0, 255], 12),
        "centre should be the red particle: {center:?}"
    );
    let corner = pixel(&pixels, 64, 1, 1);
    assert!(
        close(corner, [0, 0, 255, 255], 12),
        "corner should still be the blue background: {corner:?}"
    );
}

#[test]
fn a_particle_behind_the_camera_is_not_drawn() {
    let Some(mut gpu) = gpu(32, 32) else { return };
    let mut phase = ParticlePhase::new(&gpu);
    phase.install_misc_atlas(&gpu, &solid_atlas(16, 16, [255, 0, 0, 255]), 16, 16);

    let world = World::new();
    let cam = Camera::default();
    let mut rng = JavaRandom::new(2);
    // Behind the camera: `-look_vec` instead of `+look_vec`.
    let mut p = spawn::generic(&mut rng, cam.pos - cam.look_vec() * 5.0, DVec3::ZERO);
    p.motion = DVec3::ZERO;
    p.scale = 8.0;

    phase.upload(&gpu, &[p], &cam, 0.0, &world);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 0.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    };
    phases.insert(PhaseOrder::PARTICLES, Box::new(phase));

    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &cam, 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    let center = pixel(&pixels, 32, 16, 16);
    assert!(
        close(center, [0, 0, 255, 255], 12),
        "nothing should be behind the camera: {center:?}"
    );
}

// ---- alpha-bucket depth behaviour --------------------------------------------------------

#[test]
fn translucent_particles_do_not_occlude_each_other_depth_write_is_off() {
    // java: `EffectRenderer.renderParticles` — `case 0: GlStateManager.depthMask(false);`. Two
    // half-alpha particles on the same ray, nearer drawn first: if depth *write* were (bugly) on
    // for this bucket, the nearer one's depth write would fail the farther one's depth test and
    // it would never draw. Correct behaviour double-blends both over the background.
    let Some(mut gpu) = gpu(32, 32) else { return };
    let mut phase = ParticlePhase::new(&gpu);
    phase.install_misc_atlas(&gpu, &solid_atlas(16, 16, [255, 255, 255, 255]), 16, 16);

    let world = full_bright_world();
    let cam = Camera::default();
    let mut rng = JavaRandom::new(3);

    let mut nearer = spawn::generic(&mut rng, cam.pos + cam.look_vec() * 4.0, DVec3::ZERO);
    nearer.motion = DVec3::ZERO;
    nearer.scale = 12.0;
    nearer.color = [1.0, 0.0, 0.0, 0.5];

    let mut farther = spawn::generic(&mut rng, cam.pos + cam.look_vec() * 8.0, DVec3::ZERO);
    farther.motion = DVec3::ZERO;
    farther.scale = 12.0;
    farther.color = [0.0, 1.0, 0.0, 0.5];

    // Order matters: nearer drawn first exercises the depth-write-must-be-off requirement.
    phase.upload(&gpu, &[nearer, farther], &cam, 0.0, &world);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    phases.insert(PhaseOrder::PARTICLES, Box::new(phase));
    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &cam, 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    let center = pixel(&pixels, 32, 16, 16);

    // Expected: red(0.5) over black -> (127,0,0); green(0.5) over that -> (64,127,0).
    assert!(
        close(center, [64, 127, 0, 255], 20),
        "both particles should have blended (double-blend): {center:?}"
    );
    assert!(
        center[1] > 60,
        "the farther green particle must not have been depth-occluded: {center:?}"
    );
}

#[test]
fn a_translucent_particle_in_front_of_an_opaque_one_still_blends() {
    // Opaque bucket renders *after* translucent (java: `j` loop order 0 then 1), and always
    // writes an alpha of 1.0's worth of colour on top — so this asserts the *ordering*, not a
    // literal depth occlusion, matches `EffectRenderer`'s two-pass structure.
    let Some(mut gpu) = gpu(32, 32) else { return };
    let mut phase = ParticlePhase::new(&gpu);
    phase.install_misc_atlas(&gpu, &solid_atlas(16, 16, [255, 255, 255, 255]), 16, 16);

    let world = full_bright_world();
    let cam = Camera::default();
    let mut rng = JavaRandom::new(4);

    let mut translucent = spawn::generic(&mut rng, cam.pos + cam.look_vec() * 4.0, DVec3::ZERO);
    translucent.motion = DVec3::ZERO;
    translucent.scale = 12.0;
    translucent.color = [1.0, 1.0, 1.0, 0.25];

    let mut opaque = spawn::generic(&mut rng, cam.pos + cam.look_vec() * 8.0, DVec3::ZERO);
    opaque.motion = DVec3::ZERO;
    opaque.scale = 12.0;
    opaque.color = [0.0, 1.0, 0.0, 1.0];

    phase.upload(&gpu, &[translucent, opaque], &cam, 0.0, &world);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    phases.insert(PhaseOrder::PARTICLES, Box::new(phase));
    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &cam, 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    let center = pixel(&pixels, 32, 16, 16);

    // Translucent white(0.25) over black -> (64,64,64); opaque green drawn on top at alpha 1
    // fully replaces it -> (0, 255, 0).
    assert!(
        close(center, [0, 255, 0, 255], 12),
        "opaque draws last and wins: {center:?}"
    );
}

// ---- batching --------------------------------------------------------------------------------

#[test]
fn many_particles_sharing_a_texture_collapse_into_one_draw_call() {
    let Some(gpu) = gpu(16, 16) else { return };
    let mut phase = ParticlePhase::new(&gpu);
    phase.install_misc_atlas(&gpu, &solid_atlas(16, 16, [255, 255, 255, 255]), 16, 16);
    let world = World::new();
    let cam = Camera::default();
    let mut rng = JavaRandom::new(5);
    let particles: Vec<_> = (0..300)
        .map(|_| spawn::smoke(&mut rng, DVec3::new(0.0, 70.0, 0.0), DVec3::ZERO, 1.0))
        .collect();
    phase.upload(&gpu, &particles, &cam, 0.0, &world);
    assert_eq!(phase.batch_count(), 1);
    assert_eq!(phase.vertex_count(), 300 * 6);
}

#[test]
fn particles_on_different_layers_produce_separate_batches() {
    let Some(gpu) = gpu(16, 16) else { return };
    let mut phase = ParticlePhase::new(&gpu);
    phase.install_misc_atlas(&gpu, &solid_atlas(16, 16, [255, 255, 255, 255]), 16, 16);
    phase.install_block_atlas(
        &gpu,
        &mc_assets::Atlas {
            texture: solid_atlas(16, 16, [128, 64, 0, 255]),
            mipmaps: Vec::new(),
            w: 16,
            h: 16,
        },
    );
    let world = World::new();
    let cam = Camera::default();
    let mut rng = JavaRandom::new(6);
    let icon = ParticleIcon {
        min_u: 0.0,
        max_u: 1.0,
        min_v: 0.0,
        max_v: 1.0,
    };
    let mut particles = Vec::new();
    for _ in 0..10 {
        particles.push(spawn::smoke(
            &mut rng,
            DVec3::new(0.0, 70.0, 0.0),
            DVec3::ZERO,
            1.0,
        ));
        particles.push(spawn::digging(
            &mut rng,
            DVec3::new(0.0, 70.0, 0.0),
            DVec3::ZERO,
            icon,
            spawn::DEFAULT_DIGGING_TINT,
            false,
            mc_math::BlockPos::new(0, 70, 0),
        ));
    }
    phase.upload(&gpu, &particles, &cam, 0.0, &world);
    // Misc (translucent, since alpha stays 1.0 here it is actually Opaque) and Terrain each get
    // their own contiguous run even though the input alternates between them.
    assert_eq!(
        phase.batch_count(),
        2,
        "one batch per (texture, alpha-bucket) group, not per particle"
    );
}

// ---- real vanilla asset end-to-end ------------------------------------------------------------

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

fn vanilla_jar() -> Option<PathBuf> {
    let p = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    p.is_file().then_some(p)
}

#[test]
fn the_real_particles_png_loads_and_renders_a_recognisable_flame() {
    let Some(jar) = vanilla_jar() else {
        eprintln!("SKIP: jars/versions/1.8.9/1.8.9.jar not found; set MC_JARS_DIR");
        return;
    };
    let Some(mut gpu) = gpu(64, 64) else { return };

    let mut rm = mc_assets::ResourceManager::new();
    rm.push_pack(Box::new(
        mc_assets::ZipResourcePack::open(&jar).expect("open 1.8.9.jar"),
    ));
    let bytes = rm
        .load(&mc_assets::ResourceLocation::parse("textures/particle/particles.png").unwrap())
        .expect("load particles.png");
    let img = image::load_from_memory(&bytes)
        .expect("decode particles.png")
        .to_rgba8();
    let (w, h) = img.dimensions();
    // 1.8.9's particles.png is 128x128 (8px cells in a 16x16 grid), not the 256x256 this test
    // originally assumed — the UV math is resolution-independent (it works in fractions of the
    // atlas), so this is just correcting the test's expectation to reality.
    assert_eq!(
        (w, h),
        (128, 128),
        "vanilla's particle atlas is a 128x128, 16x16-cell grid"
    );

    let mut phase = ParticlePhase::new(&gpu);
    phase.install_misc_atlas(&gpu, img.as_raw(), w, h);

    let world = World::new();
    let cam = Camera::default();
    let mut rng = JavaRandom::new(9);
    let mut p = spawn::flame(&mut rng, cam.pos + cam.look_vec() * 4.0, DVec3::ZERO);
    p.motion = DVec3::ZERO;
    // `Particle::scale` is *not* what `Flame`'s render-time size ramp reads (see the doc comment
    // on `Particle::kind`) — it reads `peak_scale` out of the `ParticleKind::Flame` variant
    // itself, spawn-time-fixed and decoupled from `scale` on purpose. Setting `p.scale` here
    // would silently do nothing.
    p.kind = mc_render::particle::ParticleKind::Flame { peak_scale: 6.0 };

    phase.upload(&gpu, &[p], &cam, 0.0, &world);
    let mut phases = PhaseList::new();
    // Blue background, deliberately far from the flame texture's own yellow/orange/black/
    // transparent palette in every channel — and distinct even after the particle's own light
    // boost darkens it (a freshly spawned flame has `age = 0`, so its light boost starts at 0
    // and this test's empty `World` reports no world light either, so the flame draws *dark*,
    // not bright). What this test checks is "the real PNG decoded and its alpha mask produced a
    // recognisable non-rectangular silhouette", not "the flame looks visually hot".
    const BACKGROUND: [u8; 4] = [0, 0, 255, 255];
    phases.clear_color = wgpu::Color {
        r: 0.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    };
    phases.insert(PhaseOrder::PARTICLES, Box::new(phase));
    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &cam, 0.0);
    let pixels = gpu.read_pixels().expect("readback");

    // Count centre-area pixels that changed from the background at all (blend or full alpha-test
    // discard both leave a trace unless the pixel was fully transparent in the source PNG), and
    // separately require that at least *some* of those pixels are darker than the background (a
    // black/near-black flame outline, since blue's channels are far from black/yellow either
    // way) — together these rule out both "nothing rendered" and "rendered a solid blue no-op".
    let mut changed = 0;
    let mut darker = 0;
    for y in 20..44 {
        for x in 20..44 {
            let px = pixel(&pixels, 64, x, y);
            if !close(px, BACKGROUND, 8) {
                changed += 1;
                if px[2] < BACKGROUND[2] - 8 {
                    darker += 1;
                }
            }
        }
    }
    assert!(
        changed > 20,
        "expected a visible flame silhouette near the centre, got {changed}/576 changed pixels"
    );
    assert!(
        darker > 5,
        "expected some genuinely dark (non-blue) flame texels, got {darker}"
    );
}

// ---- throughput --------------------------------------------------------------------------

/// A small, *fixed*-size scene (always this many particles, independent of whatever `count` the
/// real per-iteration measurement is using) for
/// [`throughput_at_a_realistic_particle_count`]'s calibration draw.
///
/// Deliberately non-empty. An earlier version of this calibration used a literal empty particle
/// slice, reasoning that "cheapest possible, count-independent" was the safest baseline — but
/// live measurement showed that was structurally the wrong shape of cheap: with zero particles,
/// `ParticlePhase::upload` never populates `self.buffer` (see its early `return` when
/// `self.verts.is_empty()`), so `RenderPhase::record`'s `let Some(buffer) = &self.buffer else {
/// return }` bails before ever binding a pipeline or issuing a draw call. That measured only the
/// shared clear pass, not the pipeline-bind-and-draw path a real particle frame actually takes —
/// and it showed: an empty-scene calibration sat consistently at a few hundred microseconds while
/// an *isolated* real draw was observed at 10.9ms in the same run (§13/S21→S23 handoff), i.e. the
/// two were measuring different things, not the same thing under the same load. 16 particles is
/// enough to produce one non-empty batch — the real draw call path runs — while staying cheap
/// enough that its own cost still sits near the timer's noise floor.
fn calibration_particles(world: &World) -> Vec<Particle> {
    let mut system = ParticleSystem::with_seed(0xCA11_B7A7);
    for i in 0..16 {
        let pos = DVec3::new(f64::from(i), 70.0, 0.0);
        let particle = spawn::smoke(system.rng_mut(), pos, DVec3::ZERO, 1.0);
        system.spawn(world, particle);
    }
    system.tick(world);
    system.particles().to_vec()
}

/// One upload → record → submit → GPU-wait cycle, timed. Shared by the real per-count
/// measurement and the calibration draw in
/// [`throughput_at_a_realistic_particle_count`] so both go through the *exact* same code path —
/// same atlas, same render target, same `PollType::wait_indefinitely` — and differ only in
/// `particles`. Returns the elapsed wall time and the batch count the real phase produced (purely
/// diagnostic; kept so the `eprintln!` below still reports it).
fn time_particle_draw(
    gpu: &Gpu,
    cam: &Camera,
    world: &World,
    particles: &[Particle],
) -> (Duration, usize) {
    let mut phase = ParticlePhase::new(gpu);
    phase.install_misc_atlas(gpu, &solid_atlas(16, 16, [255, 255, 255, 255]), 16, 16);

    // Warm up allocations (buffer growth) before timing, same reasoning s11_perf.rs uses.
    phase.upload(gpu, particles, cam, 0.0, world);

    let start = Instant::now();
    phase.upload(gpu, particles, cam, 0.0, world);
    let batches = phase.batch_count();
    let mut phases = PhaseList::new();
    phases.insert(PhaseOrder::PARTICLES, Box::new(phase));
    let target = wgpu::TextureDescriptor {
        label: Some("throughput target"),
        size: wgpu::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: mc_render::HEADLESS_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    };
    let tex = gpu.device.create_texture(&target);
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    let _submission = phases.render(gpu, &view, cam, 0.0);
    // Blocks until this (and any other pending) submission has actually finished on the GPU, so
    // the measurement includes real device time, not just command recording.
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    (start.elapsed(), batches)
}

/// Not a rigorous benchmark (debug profile, one machine, one run) — see `s11_perf.rs` for the
/// precedent this follows. It exists to put a real, honestly-measured number next to the
/// section's "<1 ms/frame at vanilla density" target rather than asserting it unverified.
///
/// §13/S21→S23: this used to assert a flat `frame_time < 50ms`, which tripped under concurrent
/// GPU load from other work sharing the machine (61ms observed twice) despite passing
/// 3/3 in isolation — a fixed wall-clock budget can't distinguish "this machine is busy right
/// now" from "particle rendering regressed". Fixed by measuring a **calibration draw**
/// ([`calibration_particles`], through the identical upload/record/submit/GPU-wait path — see
/// [`time_particle_draw`]) bracketing the real draw — once immediately before, once immediately
/// after — and asserting the real draw is only some bounded multiple of the *worse* of the two.
/// This follows the precedent in `crates/mc-client/tests/leak.rs` (measuring growth relative to a
/// same-run baseline) and `mc-core/src/profile.rs`'s test
/// `the_layer_charges_nested_time_to_the_child_not_the_parent` (asserting a *relationship*
/// between two same-run timings rather than trusting either absolute wall-clock number). Because
/// both draws are dominated by the same fixed GPU submit+wait overhead and are measured close in
/// time, contention on the machine inflates both roughly together, so the *ratio* stays
/// meaningful even when neither absolute number is trustworthy on a loaded box. Bracketing instead
/// of a single leading calibration sample is deliberate: a live run of this exact fix (single
/// leading sample) still failed once, with a calibration draw of 185µs immediately followed by a
/// real draw of 10.9ms — a load spike that landed on the real draw's submission and missed the
/// calibration draw's, a few hundred microseconds earlier, entirely. Sampling after the real draw
/// too roughly doubles the chance a genuine, sustained load spike touches at least one
/// calibration sample instead of none.
#[test]
fn throughput_at_a_realistic_particle_count() {
    let Some(gpu) = gpu(256, 256) else { return };
    let world = World::new();
    let cam = Camera::default();

    // Prime the device before any timing starts: this machine's very first submit+GPU-wait of
    // the whole process pays a one-time cold-start tax (observed here at ~18-20ms, consistent
    // across repeated runs — plausibly a GPU power-state ramp-up) that has nothing to do with
    // particle rendering. Left untimed, it would otherwise land on whichever measurement happens
    // to go first (the first count's leading calibration draw) and make that one calibration
    // reading wildly non-representative of "current load" — exactly the kind of noise this fix is
    // meant to filter out, not reintroduce. Same "warm up before timing" idiom as the per-count
    // warm-up upload below, just at the device level instead of per-`ParticlePhase`.
    let _ = time_particle_draw(&gpu, &cam, &world, &calibration_particles(&world));

    // "Vanilla density": vanilla's own hard cap is 4000 *per bucket* (32000 absolute), but that
    // is a safety valve, not a realistic play scene. A few hundred simultaneous particles
    // (breaking blocks steadily, standing in rain, a nearby lava pool) is a representative heavy
    // case; 2000 is thrown in as a stress figure well above normal play.
    for &count in &[200usize, 1000, 2000] {
        let mut system = ParticleSystem::with_seed(1);
        for i in 0..count {
            let pos = DVec3::new((i % 20) as f64, 70.0, (i / 20) as f64);
            let particle = match i % 4 {
                0 => spawn::smoke(system.rng_mut(), pos, DVec3::ZERO, 1.0),
                1 => spawn::flame(system.rng_mut(), pos, DVec3::ZERO),
                2 => spawn::crit(system.rng_mut(), pos, DVec3::ZERO, 1.0, false),
                _ => spawn::explode(system.rng_mut(), pos, DVec3::ZERO),
            };
            system.spawn(&world, particle);
        }

        let tick_start = Instant::now();
        system.tick(&world);
        let tick_time = tick_start.elapsed();

        // Calibration brackets the real draw — once immediately before, once immediately after —
        // on a small *fixed*-size scene (see [`calibration_particles`]) measured on this run, so
        // it reflects whatever load (GPU contention from another test, OS scheduling noise) is
        // actually happening right now rather than a number baked in at review time. The worse of
        // the two guards against a load spike narrow enough to miss one calibration sample but
        // wide enough to hit the real draw (see the doc comment above for the run that motivated
        // this).
        let calibration = calibration_particles(&world);
        let (calibration_before, _) = time_particle_draw(&gpu, &cam, &world, &calibration);
        let (frame_time, batches) = time_particle_draw(&gpu, &cam, &world, system.particles());
        let (calibration_after, _) = time_particle_draw(&gpu, &cam, &world, &calibration);
        let calibration_time = calibration_before.max(calibration_after);

        eprintln!(
            "particles={count}: tick={:.3}ms calibration={:.3}/{:.3}ms \
             upload+record+submit+gpu-wait={:.3}ms batches={batches}",
            tick_time.as_secs_f64() * 1000.0,
            calibration_before.as_secs_f64() * 1000.0,
            calibration_after.as_secs_f64() * 1000.0,
            frame_time.as_secs_f64() * 1000.0,
        );

        // Relative ceiling, not an absolute one (see the doc comment above for why). On an idle
        // machine, calibration's fixed 16-particle submit+GPU-wait and a real 2000-particle one
        // are both sub-millisecond here — the fixed submission/wait cost dominates over the
        // actual per-particle work in this range — so in the common case the ratio sits close to
        // 1x. 15x leaves generous headroom for legitimate per-particle scaling (more particles
        // really is more vertex data and more overdraw) while still failing loudly on the kind of
        // order-of-magnitude regression this test exists to catch (the same bar the old absolute
        // 50ms ceiling was set to catch, just measured relatively now).
        //
        // The 20ms floor is sized off a *real* observation, not a guess: a live run of an
        // earlier, single-leading-calibration-sample version of this fix (see the doc comment
        // above) hit an isolated 10.9ms real draw against a normal ~0.2ms calibration — a single
        // submission's worth of load-spike jitter, not sustained contention, so no multiplier on
        // a normal calibration reading would have covered it without a floor doing the real work.
        // 20ms comfortably covers that observed spike with margin while staying well under the
        // old 50ms ceiling this replaces (so a regression of "particles now cost 25ms+ regardless
        // of load" still trips it).
        const MULTIPLIER: f64 = 15.0;
        const FLOOR: Duration = Duration::from_millis(20);
        let ceiling = calibration_time.mul_f64(MULTIPLIER).max(FLOOR);
        assert!(
            frame_time <= ceiling,
            "{count} particles took {frame_time:?} against same-run calibration draws of \
             {calibration_before:?} (before) / {calibration_after:?} (after) (ceiling {ceiling:?} \
             = max({MULTIPLIER}x calibration, {FLOOR:?})) — that's a disproportionate slowdown \
             relative to this run's own GPU/OS load, not just load itself"
        );
    }
}
