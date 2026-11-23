//! S23: batched chunk-mesh upload, the GPU memory budget, and the cached culling walk.
//!
//! "Chunk mesh upload batching + GPU memory budget" and the two remaining hotspots
//! of "Profile the frame loop; fix the top 5 hotspots".
//!
//! # The claim this file exists to make
//!
//! Batching changes the GPU state materially: instead of one `wgpu::Buffer` per section per
//! layer, bound whole and drawn from index 0, a section is now a *range* inside a buffer it
//! shares with up to a hundred others, reached through `draw_indexed`'s `base_vertex`, with the
//! chunk-offset buffer bound once per layer and selected with `first_instance`. Every one of
//! those is a way to render the wrong thing.
//!
//! So the load-bearing assertion is not "it is faster". It is
//! `the_batched_path_renders_byte_identical_pixels_to_the_per_section_path`: the same scene,
//! through both upload paths, read back and compared **byte for byte** across the whole
//! framebuffer. The controls that make that assertion mean something are asserted alongside it:
//!
//! * the two paths really did produce different GPU states (the buffer counts differ by an order
//!   of magnitude), so the frames are not identical because the code is the same; and
//! * the frame is not blank (a large fraction of pixels differ from the clear colour), so they
//!   are not identical because nothing drew.
//!
//! Without both, "byte-identical" is satisfied by two black screens.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use mc_math::{BlockPos, ChunkSectionPos, DVec3};
use mc_render::terrain::jobs::{GpuBudget, UploadPlan};
use mc_render::terrain::phase::MAX_U16_QUADS;
use mc_render::terrain::vertex::{Buffer, TerrainVertex, PACKED_VERTEX_SIZE, VERTEX_SIZE};
use mc_render::terrain::{
    fixture, mesh_section, ChunkMesh, TerrainOptions, TerrainPhase, TerrainRenderer,
};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, RenderError};
use mc_world::World;
use wgpu::util::DeviceExt;

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

/// Bake and install the vanilla assets once for the whole binary. `None` means "skip".
fn vanilla_atlas() -> Option<&'static mc_assets::Atlas> {
    static ATLAS: OnceLock<Option<mc_assets::Atlas>> = OnceLock::new();
    ATLAS
        .get_or_init(|| {
            let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
            if !jar.is_file() {
                eprintln!("skip: 1.8.9.jar not found; set MC_JARS_DIR");
                return None;
            }
            let mut rm = mc_assets::ResourceManager::new();
            rm.push_pack(Box::new(
                mc_assets::ZipResourcePack::open(&jar).expect("open 1.8.9.jar"),
            ));
            let (assets, report) = mc_assets::bake_all(&rm, 2048).expect("bake_all");
            assert!(report.is_clean());
            let atlas = assets.atlas.clone();
            mc_assets::model::install(assets);
            mc_render::terrain::tint::load_colormaps(&rm).expect("colormaps");
            Some(atlas)
        })
        .as_ref()
}

fn gpu() -> Option<Gpu> {
    match Gpu::headless(
        WIDTH,
        HEIGHT,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(g) => {
            g.device.on_uncaptured_error(std::sync::Arc::new(|e| {
                panic!("wgpu validation error: {e}")
            }));
            Some(g)
        }
        Err(RenderError::NoAdapter) => {
            eprintln!("skip: no wgpu adapter available on this machine");
            None
        }
        Err(e) => panic!("headless context: {e}"),
    }
}

/// A three-chunk-square scene with all four render layers populated, so a batch really does
/// carry more than one section per layer and the concatenation order matters.
fn scene() -> World {
    let mut w = fixture::blank_world(1);
    let stone = fixture::state_of("minecraft:stone");
    let grass = fixture::state_of("minecraft:grass");
    let glass = fixture::state_of("minecraft:glass");
    let leaves = fixture::state_of("minecraft:leaves");
    let water = fixture::state_of("minecraft:water");
    let log = fixture::state_of("minecraft:log");

    fixture::fill(
        &mut w,
        BlockPos::new(-16, 16, -16),
        BlockPos::new(31, 18, 31),
        stone,
    );
    fixture::fill(
        &mut w,
        BlockPos::new(-16, 19, -16),
        BlockPos::new(31, 19, 31),
        grass,
    );
    // A pool, a glass wall, and trees, spread across chunk boundaries so several sections
    // contribute to every layer's batch.
    fixture::fill(
        &mut w,
        BlockPos::new(-4, 19, -4),
        BlockPos::new(6, 19, 6),
        water,
    );
    fixture::fill(
        &mut w,
        BlockPos::new(-10, 20, 12),
        BlockPos::new(20, 23, 12),
        glass,
    );
    for (tx, tz) in [(-8, -8), (12, 4), (4, 22), (22, 20)] {
        fixture::fill(
            &mut w,
            BlockPos::new(tx, 20, tz),
            BlockPos::new(tx, 24, tz),
            log,
        );
        fixture::fill(
            &mut w,
            BlockPos::new(tx - 2, 25, tz - 2),
            BlockPos::new(tx + 2, 26, tz + 2),
            leaves,
        );
    }
    w
}

/// Two stained-glass walls at different depths, in different sections, whose projections from
/// [`camera`] overlap: blue at z = 6 (section z 0) in front of red at z = 20 (section z 1), both
/// spanning x sections -1..=1. `TRANSLUCENT` draws with depth writes off, so where they overlap
/// the blend depends on which section is submitted first. The plain [`scene`]'s only translucent
/// geometry is a flat pool, which no submission order can change (glass is `CUTOUT` in 1.8.9).
fn overlap_scene() -> World {
    let mut w = fixture::blank_world(1);
    let stained_glass = |meta| {
        let block = mc_data::block_by_name("minecraft:stained_glass").expect("stained_glass");
        mc_data::BlockState::new(block, meta)
    };
    fixture::fill(
        &mut w,
        BlockPos::new(-16, 16, -16),
        BlockPos::new(31, 18, 31),
        fixture::state_of("minecraft:stone"),
    );
    fixture::fill(
        &mut w,
        BlockPos::new(-16, 19, -16),
        BlockPos::new(31, 19, 31),
        fixture::state_of("minecraft:grass"),
    );
    // EnumDyeColor metadata: 14 red, 11 blue.
    fixture::fill(
        &mut w,
        BlockPos::new(-10, 20, 20),
        BlockPos::new(20, 30, 20),
        stained_glass(14),
    );
    fixture::fill(
        &mut w,
        BlockPos::new(-10, 20, 6),
        BlockPos::new(20, 30, 6),
        stained_glass(11),
    );
    w
}

/// Every section the scene occupies, in a fixed order.
fn all_sections() -> Vec<ChunkSectionPos> {
    (-1..=1)
        .flat_map(|x| {
            (-1..=1).flat_map(move |z| (0..3).map(move |y| ChunkSectionPos::new(x, y, z)))
        })
        .collect()
}

fn meshes(world: &World) -> Vec<ChunkMesh> {
    all_sections()
        .into_iter()
        .map(|pos| mesh_section(world, pos))
        .collect()
}

fn camera() -> Camera {
    let mut c = Camera {
        pos: DVec3::new(8.0, 34.0, -22.0),
        yaw: 0.0,
        pitch: 22.0,
        fov: 70.0,
        near: 0.05,
        far: 256.0,
        aspect: WIDTH as f32 / HEIGHT as f32,
        ..Camera::default()
    };
    c.update();
    c
}

const CLEAR: wgpu::Color = wgpu::Color {
    r: 0.10,
    g: 0.10,
    b: 0.10,
    a: 1.0,
};

/// How a [`render`] call should be set up. Everything not named here is identical between them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Setup {
    /// `upload_all` (one buffer per layer) rather than `upload` per section.
    batched: bool,
    /// Draw through the pre-T1.1 28-byte vertex layout.
    packed_vertices: bool,
    /// Push one oversized layer through the renderer and drop it again before the real upload,
    /// so the shared index buffer is on its `Uint32` fallback when the scene is drawn. T1.2.
    force_u32_indices: bool,
    /// Use native grouped indirect draws when the adapter supports them.
    multi_draw_indirect: bool,
    /// Draw [`overlap_scene`] instead of [`scene`], so translucent submission order shows.
    overlapping_translucent: bool,
    /// The order control: submit translucent sections near to far, the reverse of vanilla.
    translucent_near_to_far: bool,
    /// Upload through the live arena path (`upload_all_with_deadline_retaining`) with these
    /// `(first page bytes, page 0 growth limit)`, instead of the batch plan.
    live_arena: Option<(u64, u64)>,
}

impl Setup {
    const PER_SECTION: Setup = Setup {
        batched: false,
        packed_vertices: false,
        force_u32_indices: false,
        multi_draw_indirect: false,
        overlapping_translucent: false,
        translucent_near_to_far: false,
        live_arena: None,
    };
    const BATCHED: Setup = Setup {
        batched: true,
        packed_vertices: false,
        force_u32_indices: false,
        multi_draw_indirect: false,
        overlapping_translucent: false,
        translucent_near_to_far: false,
        live_arena: None,
    };
    const BATCHED_MDI: Setup = Setup {
        batched: true,
        packed_vertices: false,
        force_u32_indices: false,
        multi_draw_indirect: true,
        overlapping_translucent: false,
        translucent_near_to_far: false,
        live_arena: None,
    };
}

/// Render the scene and hand back the pixels plus what the renderer counted.
fn render(gpu: &mut Gpu, atlas: &mc_assets::Atlas, setup: Setup) -> Frame {
    let world = if setup.overlapping_translucent {
        overlap_scene()
    } else {
        scene()
    };
    let cam = camera();

    let mut phases = PhaseList::new();
    phases.clear_color = CLEAR;
    let shared = TerrainPhase::install_with(
        gpu,
        atlas,
        TerrainOptions {
            packed_vertices: setup.packed_vertices,
            ..TerrainOptions::default()
        },
        &mut phases,
    );
    {
        let mut r = shared.borrow_mut();
        r.set_multi_draw_indirect(setup.multi_draw_indirect);
        r.set_translucent_near_to_far_for_test(setup.translucent_near_to_far);
        r.render_distance = 8;
        if setup.force_u32_indices {
            // Uploading a layer over the `u16` ceiling widens the shared index buffer, and the
            // widening is sticky, so removing the section again leaves the renderer drawing the
            // real scene through `Uint32` indices with nothing else changed.
            let far = ChunkSectionPos::new(64, 0, 64);
            r.upload(gpu, oversized_mesh(far, MAX_U16_QUADS + 1), cam.pos);
            r.remove(far);
            assert_eq!(r.index_format(), wgpu::IndexFormat::Uint32);
        }
        if let Some((page_bytes, grow_limit)) = setup.live_arena {
            r.set_arena_page_bytes_for_test(page_bytes, grow_limit);
            let deadline = Instant::now() + Duration::from_secs(3600);
            let result =
                r.upload_all_with_deadline_retaining(gpu, meshes(&world), cam.pos, deadline);
            assert!(result.retained_positions().is_empty(), "{setup:?}");
        } else if setup.batched {
            let deferred = r.upload_all(gpu, meshes(&world), cam.pos);
            assert!(
                deferred.is_empty(),
                "the default budget must fit this scene"
            );
        } else {
            for mesh in meshes(&world) {
                r.upload(gpu, mesh, cam.pos);
            }
        }
    }

    let frame = gpu.acquire().expect("acquire");
    let index = phases.render(gpu, frame.view(), &cam, 0.0);
    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: None,
        })
        .expect("poll");
    drop(frame);
    let pixels = gpu.read_pixels().expect("read pixels");

    let r = shared.borrow();
    Frame {
        pixels,
        buffers: r.gpu_buffer_count(),
        draw_calls: r.stats().draw_calls,
        draw_items: r.stats().draw_items,
        multi_draw_indirect: r.multi_draw_indirect_enabled(),
        binds: r.stats().vertex_buffer_binds,
        visible: r.stats().visible_sections,
        quads: r.stats().quads,
        resident_bytes: r.gpu_memory().bytes,
        index_format: r.index_format(),
    }
}

/// A mesh whose solid layer holds `quads` quads, laid out as a flat grid inside the section so
/// the geometry is real rather than degenerate. Used to reach the `u16` index ceiling, which no
/// natural fixture gets near (a dense section is ~814 quads).
fn oversized_mesh(pos: ChunkSectionPos, quads: u32) -> ChunkMesh {
    let side = (quads as f64).sqrt().ceil() as u32; // 128 for 16 384
    let step = 16.0 / side as f32;
    // A real sprite, not (0, 0): the solid layer still runs `if texel.a <= 0.0 { discard; }`,
    // and the atlas corner is not guaranteed opaque.
    let sprite = mc_assets::sprite(&mc_assets::ResourceLocation::minecraft(
        "textures/blocks/stone",
    ))
    .uv;
    let uv = [
        (sprite.min_u + sprite.max_u) * 0.5,
        (sprite.min_v + sprite.max_v) * 0.5,
    ];
    let mut solid = Buffer::with_capacity(quads as usize);
    for i in 0..quads {
        let x = f32::from((i % side) as u16) * step;
        let z = f32::from((i / side) as u16) * step;
        let v = |x: f32, z: f32| TerrainVertex::new([x, 8.0, z], uv, [255; 4], [240, 0, 0, 0]);
        solid.push_quad([
            v(x, z),
            v(x, z + step),
            v(x + step, z + step),
            v(x + step, z),
        ]);
    }
    assert_eq!(solid.quad_count() as u32, quads);
    ChunkMesh {
        solid,
        ..ChunkMesh::empty(pos)
    }
}

/// **The T1.1 stage-2 gating measurement.** Is `create_buffer_init` bytes-bound?
///
/// The prediction was "upload buffer creation 100.0 → ~57 ms **if it is purely
/// bytes-bound** — that 'if' is the thing to check first, because it is also the thing that would
/// make stage 2 not worth its risk." The `frame_loop` bench cannot answer it: it measures one
/// stride per build, so before/after is a sequential A/B across two binaries on a machine with
/// other work on it.
///
/// This is the §0 rule 1 version — same process, interleaved, on identical byte layouts. It
/// allocates the *same quad count* at the three strides that matter (28 B pre-T1.1, 20 B stage 1,
/// 16 B stage 2), in the four-buffer-per-layer shape the batched upload actually uses, cycling
/// A/B/C/A/B/C so drift affects all three equally.
///
/// It prints rather than asserting a rate — a wall-clock throughput is not a property to pin — but
/// it *does* assert the shape of the answer, so the reported conclusion cannot silently invert.
#[test]
fn gpu_buffer_creation_is_bytes_bound_not_call_bound() {
    let Some(gpu) = gpu() else { return };

    // The radius-12 fixture, and its measured per-layer quad split.
    const QUADS: usize = 1_520_798;
    const SPLIT: [f64; 4] = [0.6528, 0.0526, 0.0316, 0.2630];
    const ROUNDS: usize = 12;
    let strides: [(usize, &str); 3] = [
        (28, "28 B  pre-T1.1"),
        (20, "20 B  stage 1"),
        (16, "16 B  stage 2"),
    ];

    // One zeroed source buffer, sliced per allocation, so the timed region is exactly the
    // `create_buffer_init` calls and never a host allocation.
    let max_layer_bytes = (QUADS as f64 * SPLIT[0]).ceil() as usize * 4 * 28;
    let src = vec![0u8; max_layer_bytes];

    let mut samples: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut totals = [0usize; 3];
    for round in 0..ROUNDS {
        for (i, (stride, _)) in strides.iter().enumerate() {
            let mut total = 0usize;
            let t = std::time::Instant::now();
            let mut keep = Vec::with_capacity(4);
            for share in SPLIT {
                let bytes = (QUADS as f64 * share).ceil() as usize * 4 * stride;
                total += bytes;
                keep.push(
                    gpu.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("stride probe"),
                            contents: &src[..bytes],
                            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        }),
                );
            }
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            // Explicit destroy + poll, outside the timed region: `drop` alone leaves the
            // allocation for the device to reclaim whenever it gets round to it, and twelve
            // rounds of that exhausts VRAM (measured — this test OOM'd until it did this).
            for b in &keep {
                b.destroy();
            }
            drop(keep);
            let _ = gpu.device.poll(wgpu::PollType::Poll);
            totals[i] = total;
            // Discard the first round: the first large allocation of the process pays for
            // driver heap growth, which is a one-off and not part of the question.
            if round > 0 {
                samples[i].push(ms);
            }
        }
    }

    let stat = |v: &[f64]| {
        let n = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n;
        let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        let min = v.iter().copied().fold(f64::INFINITY, f64::min);
        let max = v.iter().copied().fold(0.0f64, f64::max);
        let mut sorted = v.to_vec();
        sorted.sort_by(f64::total_cmp);
        let p50 = sorted[sorted.len() / 2];
        (mean, sd, min, max, p50)
    };

    println!(
        "\ncreate_buffer_init, 4 buffers, {QUADS} quads, {} rounds interleaved A/B/C:",
        ROUNDS - 1
    );
    let mut mean_of = [0.0f64; 3];
    let mut p50_of = [0.0f64; 3];
    for (i, (_, label)) in strides.iter().enumerate() {
        let (mean, sd, min, max, p50) = stat(&samples[i]);
        mean_of[i] = mean;
        p50_of[i] = p50;
        let mib = totals[i] as f64 / (1024.0 * 1024.0);
        println!(
            "  {label}  {mib:8.2} MiB  mean {mean:7.2} ms  sd {sd:5.2}  min {min:7.2}  \
             p50 {p50:7.2}  max {max:7.2}  n = {}   =>  {:.3} GiB/s at the mean",
            samples[i].len(),
            mib / 1024.0 / (mean / 1000.0)
        );
    }
    let byte_ratio_20 = totals[1] as f64 / totals[0] as f64;
    let byte_ratio_16 = totals[2] as f64 / totals[0] as f64;
    println!(
        "  bytes      28->20 {byte_ratio_20:.3}   28->16 {byte_ratio_16:.3}\n  \
         time mean  28->20 {:.3}   28->16 {:.3}\n  \
         time p50   28->20 {:.3}   28->16 {:.3}\n  \
         a purely bytes-bound allocator puts the time rows on the bytes row",
        mean_of[1] / mean_of[0],
        mean_of[2] / mean_of[0],
        p50_of[1] / p50_of[0],
        p50_of[2] / p50_of[0],
    );

    // Only the end-to-end direction is asserted, and deliberately **not** the 28 -> 20 -> 16
    // ordering: the run-to-run sd here is 20-40 % of the mean, and the 20 B and 16 B points sit
    // inside each other's spread. Pinning an ordering that noise can flip would be a flaky test
    // dressed up as a measurement. What must not be allowed to invert silently is the reported
    // conclusion itself — that allocating fewer bytes costs less time at all.
    assert!(mean_of.iter().all(|m| *m > 0.0));
    assert!(
        mean_of[2] < mean_of[0] && p50_of[2] < p50_of[0],
        "43 % fewer bytes did not cost less time: means {mean_of:?}, p50s {p50_of:?}"
    );
}

struct Frame {
    pixels: Vec<u8>,
    buffers: usize,
    draw_calls: usize,
    draw_items: usize,
    multi_draw_indirect: bool,
    binds: usize,
    visible: usize,
    quads: usize,
    resident_bytes: u64,
    index_format: wgpu::IndexFormat,
}

impl Frame {
    /// Pixels differing from the clear colour, so "the frame drew something" is a number.
    fn drawn_pixels(&self) -> usize {
        let clear = [
            (CLEAR.r * 255.0).round() as u8,
            (CLEAR.g * 255.0).round() as u8,
            (CLEAR.b * 255.0).round() as u8,
        ];
        self.pixels
            .chunks_exact(4)
            .filter(|p| {
                p[0].abs_diff(clear[0]) > 2
                    || p[1].abs_diff(clear[1]) > 2
                    || p[2].abs_diff(clear[2]) > 2
            })
            .count()
    }
}

/// **The load-bearing test of the batching work.**
#[test]
fn the_batched_path_renders_byte_identical_pixels_to_the_per_section_path() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(mut gpu) = gpu() else { return };

    let per_section = render(&mut gpu, atlas, Setup::PER_SECTION);
    let batched = render(&mut gpu, atlas, Setup::BATCHED);

    // Control 1: the two runs really are different GPU states. Per section, every non-empty
    // layer of every section is its own buffer; batched, it is one buffer per layer for the
    // whole upload. If these were equal the equality below would prove nothing.
    assert!(
        per_section.buffers >= 4 * batched.buffers,
        "the two paths must differ in GPU state: {} buffers per-section vs {} batched",
        per_section.buffers,
        batched.buffers
    );
    assert!(
        batched.buffers <= 4,
        "batched must be at most one buffer per render layer, got {}",
        batched.buffers
    );

    // Control 2: something actually drew. Two blank frames are byte-identical too.
    let drawn = per_section.drawn_pixels();
    let total = (WIDTH * HEIGHT) as usize;
    assert!(
        drawn > total / 10,
        "only {drawn} of {total} pixels drew; the comparison would be vacuous"
    );
    assert!(
        per_section.visible > 4 && per_section.quads > 1000,
        "scene is too thin: {} sections, {} quads",
        per_section.visible,
        per_section.quads
    );

    // The claim.
    assert_eq!(
        per_section.pixels.len(),
        batched.pixels.len(),
        "framebuffers must be the same size"
    );
    let differing = per_section
        .pixels
        .iter()
        .zip(&batched.pixels)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        differing,
        0,
        "batching changed {differing} of {} framebuffer bytes",
        per_section.pixels.len()
    );

    // And the two paths agree on everything else they report, so the equality is not the result
    // of one of them quietly drawing less.
    assert_eq!(per_section.visible, batched.visible);
    assert_eq!(per_section.draw_calls, batched.draw_calls);
    assert_eq!(per_section.quads, batched.quads);
    assert_eq!(per_section.resident_bytes, batched.resident_bytes);
    assert!(
        per_section.resident_bytes > 0,
        "the residency accountant must have seen the upload"
    );

    // The draw loop got cheaper in a way that is counted rather than asserted: per section every
    // draw rebound two vertex buffers; batched, each layer binds the offsets buffer once and the
    // shared geometry buffer once.
    assert!(
        batched.binds < per_section.binds,
        "batched {} binds vs per-section {}",
        batched.binds,
        per_section.binds
    );
    println!(
        "per-section: {} buffers, {} draws, {} vertex-buffer binds\n\
         batched:     {} buffers, {} draws, {} vertex-buffer binds\n\
         {} visible sections, {} quads, {} drawn pixels, {} resident bytes",
        per_section.buffers,
        per_section.draw_calls,
        per_section.binds,
        batched.buffers,
        batched.draw_calls,
        batched.binds,
        batched.visible,
        batched.quads,
        drawn,
        batched.resident_bytes
    );
}

/// Draw one frame through `phases` and read it back.
fn draw_frame(gpu: &mut Gpu, phases: &mut PhaseList, cam: &Camera) -> Vec<u8> {
    let frame = gpu.acquire().expect("acquire");
    let index = phases.render(gpu, frame.view(), cam, 0.0);
    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: None,
        })
        .expect("poll");
    drop(frame);
    gpu.read_pixels().expect("read pixels")
}

/// Live uploads share arena pages, so a replaced or removed section gives its range back and a
/// later section reuses it. A range handed out wrongly, or a draw plan kept past a replacement,
/// would draw one section with another's vertices. So churn the arena hard, drawing part way
/// through, and the final frame must still match a clean upload of the same meshes byte for byte.
#[test]
fn the_live_arena_renders_byte_identical_pixels_after_heavy_churn() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(mut gpu) = gpu() else { return };

    for setup in [
        Setup::BATCHED,
        Setup::BATCHED_MDI,
        Setup {
            packed_vertices: true,
            ..Setup::BATCHED
        },
    ] {
        let clean = render(&mut gpu, atlas, setup);
        let world = scene();
        let cam = camera();
        let sections = all_sections();
        let filler = |index: usize| ChunkSectionPos::new(64 + index as i32, 0, 64);
        let live_upload = |r: &mut TerrainRenderer, gpu: &Gpu, meshes: Vec<ChunkMesh>| {
            let deadline = Instant::now() + Duration::from_secs(3600);
            let count = meshes.len();
            let result = r.upload_all_with_deadline_retaining(gpu, meshes, cam.pos, deadline);
            assert_eq!(result.uploaded_positions().len(), count, "{setup:?}");
        };

        let mut phases = PhaseList::new();
        phases.clear_color = CLEAR;
        let shared = TerrainPhase::install_with(
            &gpu,
            atlas,
            TerrainOptions {
                packed_vertices: setup.packed_vertices,
                ..TerrainOptions::default()
            },
            &mut phases,
        );
        {
            let mut r = shared.borrow_mut();
            r.set_multi_draw_indirect(setup.multi_draw_indirect);
            r.render_distance = 8;
            // Visible sections with placeholder geometry, interleaved with out-of-view fillers of
            // varied sizes, so the ranges that are freed below are scattered through the page.
            for (index, pos) in sections.iter().enumerate() {
                let visible = oversized_mesh(*pos, 40 + (index as u32 * 37) % 900);
                let hidden = oversized_mesh(filler(index), 40 + (index as u32 * 53) % 900);
                live_upload(&mut r, &gpu, vec![visible, hidden]);
            }
        }
        // A frame now builds the walk and the draw plan, which the replacements must invalidate.
        let placeholder = draw_frame(&mut gpu, &mut phases, &cam);
        assert_ne!(
            placeholder, clean.pixels,
            "the churn must really change the frame"
        );

        {
            let mut r = shared.borrow_mut();
            for index in (0..sections.len()).step_by(2) {
                r.remove(filler(index));
            }
            for (index, pos) in sections.iter().enumerate().rev() {
                let resized = oversized_mesh(*pos, 40 + (index as u32 * 91) % 1500);
                live_upload(&mut r, &gpu, vec![resized]);
            }
        }
        let resized = draw_frame(&mut gpu, &mut phases, &cam);
        assert_ne!(
            resized, placeholder,
            "replacing the geometry must change the frame"
        );

        {
            let mut r = shared.borrow_mut();
            for index in (1..sections.len()).step_by(2) {
                r.remove(filler(index));
            }
        }
        draw_frame(&mut gpu, &mut phases, &cam);

        // No removal between that frame and this one: the uploads alone must invalidate what the
        // frame cached.
        {
            let mut r = shared.borrow_mut();
            let mut real = meshes(&world);
            real.reverse();
            for mesh in real {
                live_upload(&mut r, &gpu, vec![mesh]);
            }
        }
        let pixels = draw_frame(&mut gpu, &mut phases, &cam);

        let r = shared.borrow();
        assert_eq!(
            r.gpu_buffer_count(),
            1,
            "live sections must share one arena page: {setup:?}"
        );
        assert_eq!(r.stats().visible_sections, clean.visible, "{setup:?}");
        assert_eq!(r.stats().quads, clean.quads, "{setup:?}");
        assert_eq!(r.stats().draw_items, clean.draw_items, "{setup:?}");
        assert_eq!(r.gpu_memory().bytes, clean.resident_bytes, "{setup:?}");
        assert!(
            clean.drawn_pixels() > (WIDTH * HEIGHT) as usize / 10,
            "{setup:?}"
        );
        let differing = pixels
            .iter()
            .zip(&clean.pixels)
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(pixels.len(), clean.pixels.len(), "{setup:?}");
        assert_eq!(
            differing, 0,
            "{setup:?}: churn changed {differing} framebuffer bytes"
        );
    }
}

/// Native multi-draw is an optional backend capability. When present, adjacent sections sharing
/// one uploaded vertex buffer are submitted through one indirect call, while the logical section
/// items and pixels remain unchanged. Unsupported adapters retain the direct path.
#[test]
fn native_indirect_path_preserves_pixels_and_reduces_api_draw_calls() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(mut gpu) = gpu() else { return };

    let direct = render(&mut gpu, atlas, Setup::BATCHED);
    let indirect = render(&mut gpu, atlas, Setup::BATCHED_MDI);
    if !indirect.multi_draw_indirect {
        eprintln!("skip: adapter lacks native multi-draw indexed indirect support");
        return;
    }

    assert_eq!(
        direct.pixels, indirect.pixels,
        "indirect submission changed pixels"
    );
    assert_eq!(direct.visible, indirect.visible);
    assert_eq!(direct.quads, indirect.quads);
    assert_eq!(direct.draw_items, indirect.draw_items);
    println!(
        "native MDI: {} direct API draws -> {} indirect API draws; {} logical items; {} -> {} vertex-buffer binds",
        direct.draw_calls,
        indirect.draw_calls,
        indirect.draw_items,
        direct.binds,
        indirect.binds,
    );
    assert!(
        indirect.draw_calls < direct.draw_calls,
        "native indirect path did not reduce API calls: {} direct vs {} indirect",
        direct.draw_calls,
        indirect.draw_calls
    );
}

/// 03-terrain-walk-6 put `TRANSLUCENT` on the native multi-draw path. Vanilla submits it far to
/// near (`RenderGlobal.renderBlockLayer` walks `renderInfos` backwards) with depth writes off, so
/// a multi-draw must execute its commands in that order. [`overlap_scene`]'s two walls sit in
/// different sections of one arena page, so one indirect call carries both. The control draws the
/// same frame near to far and must differ: without it, equal pixels could just mean the fixture
/// cannot see translucent order at all.
#[test]
fn native_indirect_path_keeps_translucent_far_to_near_order() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(mut gpu) = gpu() else { return };
    let overlap = Setup {
        overlapping_translucent: true,
        ..Setup::BATCHED
    };

    let direct = render(&mut gpu, atlas, overlap);
    let reversed = render(
        &mut gpu,
        atlas,
        Setup {
            translucent_near_to_far: true,
            ..overlap
        },
    );
    assert!(direct.drawn_pixels() > (WIDTH * HEIGHT) as usize / 10);
    assert_eq!(direct.quads, reversed.quads);
    let order_sensitive = direct
        .pixels
        .chunks_exact(4)
        .zip(reversed.pixels.chunks_exact(4))
        .filter(|(a, b)| a != b)
        .count();
    println!("near-to-far control changed {order_sensitive} pixels");
    assert!(
        order_sensitive > (WIDTH * HEIGHT) as usize / 100,
        "control: drawing the walls near to far changed only {order_sensitive} pixels, so this \
         fixture cannot see translucent submission order"
    );

    let indirect = render(
        &mut gpu,
        atlas,
        Setup {
            multi_draw_indirect: true,
            ..overlap
        },
    );
    if !indirect.multi_draw_indirect {
        eprintln!("skip: adapter lacks native multi-draw indexed indirect support");
        return;
    }
    assert!(
        indirect.draw_calls < direct.draw_calls,
        "the walls were not grouped into shared indirect calls: {} direct vs {} indirect",
        direct.draw_calls,
        indirect.draw_calls
    );
    assert_eq!(direct.draw_items, indirect.draw_items);
    assert_eq!(
        direct.pixels, indirect.pixels,
        "native multi-draw changed translucent submission order"
    );
}

/// Phase 3b: live uploads that outgrow the first arena page grow it in place. The scene drawn
/// from one grown page, and from several pages once growth is capped, is byte-identical to one
/// page that never grew, through direct draws and native multi-draw; one grown page also keeps
/// the multi-draw call count of a page that never grew.
#[test]
fn a_grown_arena_page_renders_byte_identical_pixels() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(mut gpu) = gpu() else { return };
    for multi_draw_indirect in [false, true] {
        let live = |pages| Setup {
            live_arena: Some(pages),
            multi_draw_indirect,
            ..Setup::PER_SECTION
        };
        let reference = render(&mut gpu, atlas, live((128 << 20, 1 << 30)));
        let grown = render(&mut gpu, atlas, live((64 << 10, 1 << 30)));
        let capped = render(&mut gpu, atlas, live((64 << 10, 128 << 10)));
        assert!(reference.drawn_pixels() > (WIDTH * HEIGHT) as usize / 10);
        assert_eq!(reference.quads, grown.quads);
        assert!(
            grown.pixels == reference.pixels,
            "grown page changed pixels (mdi={multi_draw_indirect})"
        );
        assert!(
            capped.pixels == reference.pixels,
            "capped pages changed pixels (mdi={multi_draw_indirect})"
        );
        assert_eq!(
            grown.draw_calls, reference.draw_calls,
            "mdi={multi_draw_indirect}"
        );
        println!(
            "mdi={multi_draw_indirect}: draw calls one page {}, grown {}, capped {}",
            reference.draw_calls, grown.draw_calls, capped.draw_calls
        );
    }
}

/// The MDI command plan is allowed to persist across still frames, but a resident replacement
/// or visibility change must rebuild it. The final A/B draw keeps the framebuffer comparison
/// load-bearing: caching cannot trade stale first-instance slots or base vertices for fewer CPU
/// allocations.
#[test]
fn native_indirect_plan_cache_rebuilds_for_visibility_and_replacement() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(mut gpu) = gpu() else { return };
    let world = scene();
    let cam = camera();
    let mut phases = PhaseList::new();
    phases.clear_color = CLEAR;
    let shared = TerrainPhase::install(&gpu, atlas, &mut phases);
    {
        let mut renderer = shared.borrow_mut();
        renderer.set_multi_draw_indirect(true);
        renderer.upload_all(&gpu, meshes(&world), cam.pos);
    }

    let mut draw = |gpu: &mut Gpu, cam: &Camera| {
        let frame = gpu.acquire().expect("acquire");
        let submission = phases.render(gpu, frame.view(), cam, 0.0);
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .expect("poll");
        drop(frame);
        let pixels = gpu.read_pixels().expect("read pixels");
        let renderer = shared.borrow();
        (pixels, renderer.stats(), renderer.draw_plan_cache_counts())
    };

    let (initial_pixels, initial, initial_cache) = draw(&mut gpu, &cam);
    let (still_pixels, still, still_cache) = draw(&mut gpu, &cam);
    assert_eq!(still_pixels.len(), (WIDTH * HEIGHT * 4) as usize);
    assert_eq!(
        still_pixels, initial_pixels,
        "a still frame must retain pixels"
    );
    assert_eq!(still, initial, "a still frame must retain draw contents");
    assert!(
        still_cache.1 > initial_cache.1,
        "a still frame must hit the cached layer plans: {initial_cache:?} -> {still_cache:?}"
    );

    {
        let mut renderer = shared.borrow_mut();
        renderer.smart_culling = false;
        renderer.render_distance = 1;
    }
    let (distance_mdi_pixels, after_distance, distance_cache) = draw(&mut gpu, &cam);
    assert!(
        after_distance.visible_sections > 0
            && after_distance.visible_sections < initial.visible_sections,
        "changing render distance must change the visible set: {} -> {}",
        initial.visible_sections,
        after_distance.visible_sections
    );
    assert!(
        distance_cache.0 > still_cache.0,
        "visibility changes must rebuild layer plans: {still_cache:?} -> {distance_cache:?}"
    );

    shared.borrow_mut().set_multi_draw_indirect(false);
    let (distance_direct_pixels, _, distance_direct_cache) = draw(&mut gpu, &cam);
    assert_eq!(
        distance_mdi_pixels, distance_direct_pixels,
        "MDI and direct paths must agree after visibility changes"
    );
    assert!(distance_direct_cache.0 > distance_cache.0);
    shared.borrow_mut().set_multi_draw_indirect(true);

    {
        let mut renderer = shared.borrow_mut();
        renderer.smart_culling = true;
        renderer.render_distance = 8;
    }
    let (_, restored, restored_cache) = draw(&mut gpu, &cam);
    assert_eq!(restored.visible_sections, initial.visible_sections);
    assert!(restored_cache.0 > distance_direct_cache.0);

    let removed_pos = ChunkSectionPos::new(0, 1, -1);
    shared.borrow_mut().remove(removed_pos);
    let (_, _after_remove, removed_cache) = draw(&mut gpu, &cam);
    assert!(
        removed_cache.0 > restored_cache.0,
        "a resident removal must rebuild layer plans: {restored_cache:?} -> {removed_cache:?}"
    );
    let replacement = meshes(&world)
        .into_iter()
        .find(|mesh| mesh.pos == removed_pos)
        .expect("replacement section");
    shared.borrow_mut().upload(&gpu, replacement, cam.pos);
    let (mdi_pixels, _, replaced_cache) = draw(&mut gpu, &cam);
    assert!(
        replaced_cache.0 > removed_cache.0,
        "a resident replacement must rebuild layer plans: {removed_cache:?} -> {replaced_cache:?}"
    );

    shared.borrow_mut().set_multi_draw_indirect(false);
    let (direct_pixels, _, direct_cache) = draw(&mut gpu, &cam);
    assert_eq!(
        mdi_pixels, direct_pixels,
        "cached MDI changed replacement pixels"
    );
    assert!(
        direct_cache.0 > replaced_cache.0,
        "switching to the direct fallback must rebuild the plan key"
    );
}

/// **The load-bearing test of T1.1 stage 1** — `TerrainVertex` 28 B → 20 B.
///
/// Same harness, same two controls. The packed format quantises position to 1/2048 of a block
/// and UV to 1/32768 of the atlas. It is an explicitly lossy regression control; the faithful
/// 28-byte layout is the production contract because fluids and generated-model UVs are off-grid.
///
/// A sound A/B asks for an interleaved same-process control. A whole-format change
/// cannot be interleaved — a pipeline has one vertex layout — so the control is
/// `TerrainOptions::packed_vertices` keeps that control path runnable. The comparison verifies
/// that both paths render the same amount of geometry while proving the control's quantisation is
/// observable, so a future accidental return to packed production vertices cannot hide.
#[test]
fn the_faithful_vertex_format_and_packed_control_both_render_the_scene() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(mut gpu) = gpu() else { return };

    let packed = render(
        &mut gpu,
        atlas,
        Setup {
            packed_vertices: true,
            ..Setup::BATCHED
        },
    );
    let faithful = render(&mut gpu, atlas, Setup::BATCHED);

    // Control 1: the two runs really are different GPU states — different vertex bytes resident,
    // in exactly the 28:20 ratio the change is about. If these were equal the equality below
    // would prove nothing.
    assert_eq!(
        faithful.resident_bytes * PACKED_VERTEX_SIZE as u64,
        packed.resident_bytes * VERTEX_SIZE as u64,
        "the two formats must differ in vertex bytes: {} faithful vs {} packed",
        faithful.resident_bytes,
        packed.resident_bytes
    );
    assert!(
        packed.resident_bytes < faithful.resident_bytes,
        "{} packed is not smaller than {} faithful",
        packed.resident_bytes,
        faithful.resident_bytes
    );

    // Control 2: something actually drew. Two blank frames are byte-identical too.
    let drawn = faithful.drawn_pixels();
    let total = (WIDTH * HEIGHT) as usize;
    assert!(
        drawn >= 40_000,
        "only {drawn} of {total} pixels drew; the comparison would be vacuous"
    );
    assert_eq!(packed.drawn_pixels(), drawn, "both frames drew the same");

    // The packed control is intentionally lossy. The source-level vertex regression proves that
    // off-grid inputs decode to different bits; this framebuffer check makes the same contract
    // visible end to end without inventing a tolerance as a production equivalence claim.
    assert_eq!(faithful.pixels.len(), packed.pixels.len());
    let differing = faithful
        .pixels
        .iter()
        .zip(&packed.pixels)
        .filter(|(a, b)| a != b)
        .count();
    let worst = faithful
        .pixels
        .iter()
        .zip(&packed.pixels)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    assert!(
        differing > 0 && worst > 0,
        "packed control unexpectedly matched faithful pixels exactly: {differing} differing bytes, \
         worst channel delta {worst}"
    );

    // And neither path quietly drew less.
    assert_eq!(faithful.visible, packed.visible);
    assert_eq!(faithful.draw_calls, packed.draw_calls);
    assert_eq!(faithful.quads, packed.quads);
    println!(
        "vertex format 28 B -> 20 B: {} -> {} resident bytes ({:.1} %), {} quads, {} visible \
         sections, {drawn} drawn pixels, {differing} of {} framebuffer bytes differ (worst {worst})",
        faithful.resident_bytes,
        packed.resident_bytes,
        100.0 * packed.resident_bytes as f64 / faithful.resident_bytes as f64,
        packed.quads,
        packed.visible,
        packed.pixels.len()
    );
}

/// **T1.2's mandatory guard.** A layer over the `u16` ceiling must take the `Uint32` path.
///
/// Without this the saving is a latent corruption bug: a 16 384-quad layer's last index is
/// 65 539, which a `u16` buffer would wrap to 3, drawing that section's geometry out of its own
/// first vertices.
#[test]
fn a_layer_over_the_u16_index_ceiling_falls_back_to_uint32() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(mut gpu) = gpu() else { return };

    assert_eq!(MAX_U16_QUADS, 16_383, "4 * 16383 + 3 == u16::MAX");

    // Control: at the ceiling exactly, `Uint16` is still correct and still chosen. Without this
    // half a renderer that always picked `Uint32` would pass the assertion below.
    let mut at_ceiling = TerrainRenderer::new(&gpu, atlas);
    assert_eq!(at_ceiling.index_format(), wgpu::IndexFormat::Uint16);
    at_ceiling.upload(
        &gpu,
        oversized_mesh(ChunkSectionPos::new(0, 0, 0), MAX_U16_QUADS),
        DVec3::ZERO,
    );
    assert_eq!(
        at_ceiling.index_format(),
        wgpu::IndexFormat::Uint16,
        "{MAX_U16_QUADS} quads still fit a u16 index"
    );

    // One quad more, and the fallback must fire.
    let mut over = TerrainRenderer::new(&gpu, atlas);
    over.upload(
        &gpu,
        oversized_mesh(ChunkSectionPos::new(0, 0, 0), MAX_U16_QUADS + 1),
        DVec3::ZERO,
    );
    assert_eq!(
        over.index_format(),
        wgpu::IndexFormat::Uint32,
        "{} quads must not be indexed with a u16",
        MAX_U16_QUADS + 1
    );

    // The upgrade is sticky: a small batch afterwards must not drop back to `Uint16`, or the
    // still-resident oversized section would be drawn through a wrapping index buffer.
    over.upload_all(&gpu, meshes(&scene()), DVec3::ZERO);
    assert_eq!(over.index_format(), wgpu::IndexFormat::Uint32);

    // And it renders. A format chosen correctly but bound wrongly draws nothing (or garbage),
    // so the fallback path is exercised on the GPU rather than merely selected.
    let mut phases = PhaseList::new();
    phases.clear_color = CLEAR;
    // Back-face culling off: this test is about the index buffer, and which way a synthetic
    // grid of quads happens to wind is not the claim.
    let shared = TerrainPhase::install_with(
        &gpu,
        atlas,
        TerrainOptions {
            cull_backfaces: false,
            ..TerrainOptions::default()
        },
        &mut phases,
    );
    let mut cam = camera();
    cam.pos = DVec3::new(8.0, 18.0, 8.0);
    cam.pitch = 89.0;
    cam.update();
    shared.borrow_mut().render_distance = 8;
    shared.borrow_mut().upload(
        &gpu,
        oversized_mesh(ChunkSectionPos::new(0, 0, 0), MAX_U16_QUADS + 1),
        cam.pos,
    );
    assert_eq!(
        shared.borrow().index_format(),
        wgpu::IndexFormat::Uint32,
        "the renderer under test must be on the fallback"
    );
    let frame = gpu.acquire().expect("acquire");
    let index = phases.render(&gpu, frame.view(), &cam, 0.0);
    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: None,
        })
        .expect("poll");
    drop(frame);
    let pixels = gpu.read_pixels().expect("read pixels");
    let drawn = Frame {
        pixels,
        buffers: 0,
        draw_calls: 0,
        draw_items: 0,
        multi_draw_indirect: false,
        binds: 0,
        visible: 0,
        quads: 0,
        resident_bytes: 0,
        index_format: wgpu::IndexFormat::Uint32,
    }
    .drawn_pixels();
    assert!(
        drawn >= 20_000,
        "the 16 384-quad plane drew only {drawn} pixels through the Uint32 fallback"
    );
    assert_eq!(shared.borrow().stats().quads, (MAX_U16_QUADS + 1) as usize);
    println!(
        "u16 ceiling {MAX_U16_QUADS} quads; {} quads -> Uint32, {drawn} pixels drawn",
        MAX_U16_QUADS + 1
    );
}

/// T1.2 byte-identity: halving the index buffer must not change a pixel.
///
/// The control is a renderer forced onto the `Uint32` path by an oversized section that is then
/// removed — same scene, same vertices, same draws, only the index width differs.
#[test]
fn uint16_and_uint32_index_buffers_render_the_same_frame() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(mut gpu) = gpu() else { return };

    let u16_path = render(&mut gpu, atlas, Setup::BATCHED);
    let u32_path = render(
        &mut gpu,
        atlas,
        Setup {
            force_u32_indices: true,
            ..Setup::BATCHED
        },
    );

    // Control 1: the two really did bind different index formats.
    assert_eq!(u16_path.index_format, wgpu::IndexFormat::Uint16);
    assert_eq!(u32_path.index_format, wgpu::IndexFormat::Uint32);
    // Control 2: the frame is not blank.
    let drawn = u16_path.drawn_pixels();
    assert!(drawn >= 40_000, "only {drawn} pixels drew");

    let differing = u16_path
        .pixels
        .iter()
        .zip(&u32_path.pixels)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        differing,
        0,
        "the index width changed {differing} of {} framebuffer bytes",
        u16_path.pixels.len()
    );
    assert_eq!(u16_path.quads, u32_path.quads);
    assert_eq!(u16_path.draw_calls, u32_path.draw_calls);
    println!(
        "index buffer Uint32 -> Uint16: {} quads, {} draws, {drawn} drawn pixels, 0 of {} bytes \
         differ",
        u16_path.quads,
        u16_path.draw_calls,
        u16_path.pixels.len()
    );
}

/// The buffer-count claim on its own, against the number the plan records (150 → 4 on 100 real
/// meshed sections), so the headline figure is reproduced rather than quoted.
#[test]
fn batching_collapses_one_buffer_per_section_per_layer_into_one_per_layer() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(gpu) = gpu() else { return };
    let world = scene();
    let all = meshes(&world);

    let plan = UploadPlan::plan(&all, GpuBudget::new(u64::MAX));
    assert!(
        plan.buffers_unbatched > 20,
        "the fixture must have enough non-empty layers to be worth batching, got {}",
        plan.buffers_unbatched
    );
    assert_eq!(plan.batches.len(), 4, "all four layers are populated");

    let mut per_section = TerrainRenderer::new(&gpu, atlas);
    for mesh in all.clone() {
        per_section.upload(&gpu, mesh, DVec3::ZERO);
    }
    let mut batched = TerrainRenderer::new(&gpu, atlas);
    batched.upload_all(&gpu, all.clone(), DVec3::ZERO);

    assert_eq!(
        per_section.gpu_buffer_count(),
        plan.buffers_unbatched,
        "the per-section path is exactly the plan's `buffers_unbatched`"
    );
    assert_eq!(batched.gpu_buffer_count(), 4);
    assert_eq!(
        per_section.loaded_sections(),
        batched.loaded_sections(),
        "and both hold the same sections"
    );
    assert_eq!(
        per_section.gpu_memory().bytes,
        batched.gpu_memory().bytes,
        "and the same bytes"
    );
    println!(
        "{} sections: {} buffers unbatched -> {} batched ({} bytes)",
        all.len(),
        per_section.gpu_buffer_count(),
        batched.gpu_buffer_count(),
        batched.gpu_memory().bytes
    );
}

/// The budget: over it, the farthest sections are dropped; under it, nothing is. The control is
/// the second half — without it, an accountant that dropped everything unconditionally would
/// pass the first half.
#[test]
fn the_gpu_budget_evicts_the_farthest_sections_and_only_when_over_it() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(mut gpu) = gpu() else { return };
    let world = scene();
    let all = meshes(&world);
    let cam = camera();

    let mut phases = PhaseList::new();
    let shared = TerrainPhase::install(&gpu, atlas, &mut phases);
    let loaded;
    let full_bytes;
    {
        let mut r = shared.borrow_mut();
        r.render_distance = 8;
        r.upload_all(&gpu, all.clone(), cam.pos);
        loaded = r.loaded_sections();
        full_bytes = r.gpu_memory().bytes;
        assert!(full_bytes > 0);
        assert_eq!(r.gpu_memory().limit, GpuBudget::DEFAULT_BYTES);
        assert_eq!(r.gpu_memory().evicted, 0, "nothing evicted under the cap");
    }

    // One frame under the default cap: the eviction pass runs and does nothing.
    let frame = gpu.acquire().expect("acquire");
    let index = phases.render(&gpu, frame.view(), &cam, 0.0);
    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: None,
        })
        .expect("poll");
    drop(frame);
    assert_eq!(
        shared.borrow().loaded_sections(),
        loaded,
        "a frame under budget must evict nothing"
    );
    assert_eq!(shared.borrow().gpu_memory().evicted, 0);

    // Now halve the cap and run one more frame.
    shared
        .borrow_mut()
        .set_gpu_budget(GpuBudget::new(full_bytes / 2));
    let frame = gpu.acquire().expect("acquire");
    let index = phases.render(&gpu, frame.view(), &cam, 0.0);
    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: None,
        })
        .expect("poll");
    drop(frame);

    // Every dropped section must be *reported*, exactly once. A caller that tracks what it
    // uploaded — `mc-client` does — would otherwise believe an evicted section is still resident
    // and never ask for it again, which is a permanent hole rather than a one-frame one.
    let reported = shared.borrow_mut().take_evicted();
    assert_eq!(
        reported.len() as u64,
        shared.borrow().gpu_memory().evicted,
        "every evicted section must be reported"
    );
    for pos in &reported {
        assert!(
            shared.borrow().visibility(*pos).is_none(),
            "{pos:?} was reported as evicted but is still resident"
        );
    }
    assert!(
        shared.borrow_mut().take_evicted().is_empty(),
        "the report must drain, or a caller would re-queue the same sections every frame"
    );

    let r = shared.borrow();
    assert!(r.gpu_memory().evicted > 0, "the cap must have bitten");
    assert!(
        r.gpu_memory().bytes <= full_bytes / 2,
        "{} bytes still resident against a {} cap",
        r.gpu_memory().bytes,
        full_bytes / 2
    );
    assert!(
        r.loaded_sections() < loaded,
        "sections must actually have been dropped: {} of {}",
        r.loaded_sections(),
        loaded
    );
    // Farthest-first: the section the camera is nearest must survive.
    let nearest = ChunkSectionPos::new(0, 1, 0);
    assert!(
        r.visibility(nearest).is_some(),
        "eviction must be farthest-first, but the nearest section went"
    );
    println!(
        "budget {} bytes: {} of {} sections survive, {} evicted",
        full_bytes / 2,
        r.loaded_sections(),
        loaded,
        r.gpu_memory().evicted
    );
}

/// A packing budget too small for the whole batch defers whole sections, and the renderer says
/// which — so the caller can requeue them instead of rendering a hole.
#[test]
fn a_batch_over_the_packing_budget_defers_whole_sections() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(gpu) = gpu() else { return };
    let world = scene();
    let all = meshes(&world);
    let total: u64 = all.iter().map(|m| m.byte_size() as u64).sum();

    let mut r = TerrainRenderer::new(&gpu, atlas);
    r.set_gpu_budget(GpuBudget::new(total / 2));
    let deferred = r.upload_all(&gpu, all.clone(), DVec3::ZERO);
    assert!(!deferred.is_empty(), "half the bytes cannot all fit");
    for pos in &deferred {
        assert!(
            r.visibility(*pos).is_none(),
            "{pos:?} was deferred but is resident anyway"
        );
    }
    assert!(
        r.gpu_memory().bytes <= total / 2,
        "the packing budget must be respected: {} > {}",
        r.gpu_memory().bytes,
        total / 2
    );

    // Control: with room, nothing is deferred and every section lands.
    let mut roomy = TerrainRenderer::new(&gpu, atlas);
    assert!(roomy.upload_all(&gpu, all.clone(), DVec3::ZERO).is_empty());
    assert_eq!(roomy.loaded_sections(), all.len());
}

/// The culling-walk cache, driven through the real renderer rather than through `cull.rs`'s own
/// unit tests: an unchanged frame must reuse the walk, and every way the world can change must
/// force a fresh one.
#[test]
fn the_renderer_reuses_the_culling_walk_only_while_nothing_it_depends_on_moved() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(mut gpu) = gpu() else { return };
    let world = scene();
    let mut cam = camera();

    let mut phases = PhaseList::new();
    let shared = TerrainPhase::install(&gpu, atlas, &mut phases);
    shared.borrow_mut().render_distance = 8;
    shared
        .borrow_mut()
        .upload_all(&gpu, meshes(&world), cam.pos);

    let draw = |gpu: &mut Gpu, phases: &mut PhaseList, cam: &Camera| {
        let frame = gpu.acquire().expect("acquire");
        let index = phases.render(gpu, frame.view(), cam, 0.0);
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: None,
            })
            .expect("poll");
        drop(frame);
    };

    // Four identical frames: one walk, three cache hits.
    for _ in 0..4 {
        draw(&mut gpu, &mut phases, &cam);
    }
    let (hits, walks) = shared.borrow().cull_cache_counts();
    assert_eq!((hits, walks), (3, 1), "a still camera must walk once");
    let still_visible = shared.borrow().stats().visible_sections;
    assert!(still_visible > 0);

    // The camera turns without moving: the frustum changed, so the walk must be redone.
    cam.yaw += 90.0;
    cam.update();
    draw(&mut gpu, &mut phases, &cam);
    assert_eq!(
        shared.borrow().cull_cache_counts(),
        (3, 2),
        "a turn re-walks"
    );

    // The camera moves by a hundredth of a block — inside the same section, which a
    // section-granularity cache key would wrongly treat as unchanged.
    cam.pos.x += 0.01;
    cam.update();
    draw(&mut gpu, &mut phases, &cam);
    assert_eq!(
        shared.borrow().cull_cache_counts(),
        (3, 3),
        "a sub-section move re-walks"
    );

    // A chunk unloads.
    draw(&mut gpu, &mut phases, &cam);
    assert_eq!(shared.borrow().cull_cache_counts(), (4, 3), "control: hit");
    shared.borrow_mut().remove(ChunkSectionPos::new(0, 1, 0));
    draw(&mut gpu, &mut phases, &cam);
    assert_eq!(
        shared.borrow().cull_cache_counts(),
        (4, 4),
        "an unloaded chunk re-walks"
    );

    // A chunk loads (or a dirty section is re-meshed and re-uploaded, which is the same call).
    shared.borrow_mut().upload(
        &gpu,
        mesh_section(&world, ChunkSectionPos::new(0, 1, 0)),
        cam.pos,
    );
    draw(&mut gpu, &mut phases, &cam);
    assert_eq!(
        shared.borrow().cull_cache_counts(),
        (4, 5),
        "a re-uploaded section re-walks"
    );

    // Removing a section that was never resident changes nothing, so it must not invalidate.
    shared.borrow_mut().remove(ChunkSectionPos::new(99, 9, 99));
    draw(&mut gpu, &mut phases, &cam);
    assert_eq!(
        shared.borrow().cull_cache_counts(),
        (5, 5),
        "a no-op removal must not invalidate"
    );

    // The render distance is an input too.
    shared.borrow_mut().render_distance = 4;
    draw(&mut gpu, &mut phases, &cam);
    assert_eq!(shared.borrow().cull_cache_counts(), (5, 6));
}

/// The re-sort path writes into a *shared* buffer at an offset now. A wrong offset would corrupt
/// a neighbouring section's vertices, which is invisible in a still frame and catastrophic in
/// motion — so this asserts a moved camera's re-sorted frame is exactly what a freshly uploaded
/// one is.
#[test]
fn re_sorting_translucent_geometry_in_a_shared_buffer_matches_a_fresh_upload() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(mut gpu) = gpu() else { return };
    let world = scene();

    // The camera the frame is finally rendered from.
    let mut moved = camera();
    moved.pos = DVec3::new(30.0, 34.0, 40.0);
    moved.yaw = 200.0;
    moved.update();

    // (a) uploaded (and therefore sorted) for a *different* camera, then re-sorted for `moved`.
    let mut phases = PhaseList::new();
    phases.clear_color = CLEAR;
    let shared = TerrainPhase::install(&gpu, atlas, &mut phases);
    {
        let mut r = shared.borrow_mut();
        r.render_distance = 8;
        r.upload_all(&gpu, meshes(&world), camera().pos);
        r.resort_translucent(&gpu, moved.pos);
    }
    let frame = gpu.acquire().expect("acquire");
    let index = phases.render(&gpu, frame.view(), &moved, 0.0);
    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: None,
        })
        .expect("poll");
    drop(frame);
    let resorted = gpu.read_pixels().expect("read pixels");

    // (b) uploaded straight for `moved`, which sorts on the way in.
    let mut phases = PhaseList::new();
    phases.clear_color = CLEAR;
    let shared = TerrainPhase::install(&gpu, atlas, &mut phases);
    {
        let mut r = shared.borrow_mut();
        r.render_distance = 8;
        r.upload_all(&gpu, meshes(&world), moved.pos);
    }
    let frame = gpu.acquire().expect("acquire");
    let index = phases.render(&gpu, frame.view(), &moved, 0.0);
    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: None,
        })
        .expect("poll");
    drop(frame);
    let fresh = gpu.read_pixels().expect("read pixels");

    let differing = resorted.iter().zip(&fresh).filter(|(a, b)| a != b).count();
    assert_eq!(
        differing, 0,
        "an in-place re-sort into a shared batch buffer must land exactly where a fresh upload \
         does ({differing} bytes differ)"
    );
}
