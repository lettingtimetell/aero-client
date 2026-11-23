//! The shader packs' shading holds still on the GPU.
//!
//! A still scene (a grass floor with stone pillars of several heights and a floating slab, so
//! there are shadow edges cast from near and far) is drawn through the real terrain variant and
//! shadow phase, with this frame's `PackFrame` built by `PackFrame::compute` from vanilla's clock
//! exactly as the client builds it: three frames per tick at partial ticks 0, 1/3 and 2/3, on the
//! packs' sun path tilted 25 degrees. Every frame is compared with the one before: a pixel
//! "jumps" when its brightness moves by 3 levels or more.
//!
//! * A held clock (the Time Changer; a server that stops the daylight cycle) draws the same
//!   picture on every frame.
//! * A running clock moves the shading only in the small steps of the map's direction
//!   quantisation; between those steps nothing moves.
//!
//! `MC_W22_SHD_PNG_DIR=<dir>` writes the worst step of each timeline as a strip (before, after,
//! difference x8). `MC_W22_SHD_STABLE=0` runs the previous quantisation (the counts it
//! prints are the "before" numbers).

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use glam::{DVec3, Vec3};
use mc_math::{BlockPos, ChunkSectionPos};
use mc_render::shaderpack::{
    order, CoreConfig, EyeMedium, FrameInputs, LightingSettings, PackFrame, PackId, ShaderCore,
    ShadowPhase, ShadowSettings,
};
use mc_render::sky::fog::{FogDistance, FogMode};
use mc_render::sky::Environment;
use mc_render::terrain::{
    fixture, mesh_section_with, ChunkMesh, MeshSettings, TerrainOptions, TerrainPhase,
    TerrainRenderer,
};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, RenderError};
use mc_world::World;

const WIDTH: u32 = 480;
const HEIGHT: u32 = 270;
const FRAMES_PER_TICK: u32 = 3;
/// A pixel jumps when its brightness moves by at least this many levels between two frames.
const JUMP: f32 = 3.0;
/// The packs' sun-path tilt (Classic's default, Dreamy's fixed value).
const TILT: f32 = 25.0;

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

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

/// A grass floor, pillars from 2 to 12 blocks tall, and a slab floating 8 blocks up.
fn scene() -> World {
    let mut w = fixture::blank_world(1);
    let stone = fixture::state_of("minecraft:stone");
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
        fixture::state_of("minecraft:grass"),
    );
    let mut height = 2;
    for x in (-8..=24).step_by(8) {
        for z in (0..=24).step_by(8) {
            fixture::fill(
                &mut w,
                BlockPos::new(x, 20, z),
                BlockPos::new(x + 1, 19 + height, z),
                stone,
            );
            height = height % 12 + 2;
        }
    }
    fixture::fill(
        &mut w,
        BlockPos::new(2, 28, 12),
        BlockPos::new(6, 28, 15),
        stone,
    );
    w
}

fn sections() -> Vec<ChunkSectionPos> {
    (-1..=1)
        .flat_map(|x| {
            (-1..=1).flat_map(move |z| (0..3).map(move |y| ChunkSectionPos::new(x, y, z)))
        })
        .collect()
}

fn meshes(world: &World, settings: MeshSettings) -> Vec<ChunkMesh> {
    sections()
        .into_iter()
        .map(|pos| mesh_section_with(world, pos, settings))
        .collect()
}

fn camera() -> Camera {
    let mut c = Camera {
        pos: DVec3::new(8.3, 36.0, -12.7),
        yaw: 0.0,
        pitch: 42.0,
        fov: 70.0,
        near: 0.05,
        far: 256.0,
        aspect: WIDTH as f32 / HEIGHT as f32,
        ..Camera::default()
    };
    c.update();
    c
}

fn classic(resolution: u32) -> CoreConfig {
    CoreConfig {
        sun_tilt: TILT,
        shadows: Some(ShadowSettings {
            resolution,
            ..ShadowSettings::default()
        }),
        lighting: Some(LightingSettings::default()),
        ..CoreConfig::OFF
    }
}

struct Bench {
    gpu: Gpu,
    phases: PhaseList,
    terrain: std::rc::Rc<std::cell::RefCell<TerrainRenderer>>,
    core: ShaderCore,
    cam: Camera,
    frame_index: u32,
}

impl Bench {
    fn new(atlas: &mc_assets::Atlas, gpu: Gpu, resolution: u32) -> Bench {
        let cam = camera();
        let mut phases = PhaseList::new();
        let terrain =
            TerrainPhase::install_with(&gpu, atlas, TerrainOptions::default(), &mut phases);
        let core = ShaderCore::new();
        terrain.borrow_mut().set_shader_core(Some(core.clone()));
        phases.insert(
            order::SHADOW,
            Box::new(ShadowPhase::new(core.clone(), terrain.clone())),
        );
        core.set_active(Some(PackId::Classic));
        let config = classic(resolution);
        core.set_config(config);
        let mesh = MeshSettings {
            pack: config.mesh(),
            ..MeshSettings::default()
        };
        let mut bench = Bench {
            gpu,
            phases,
            terrain,
            core,
            cam,
            frame_index: 0,
        };
        // A first frame so the core runs before the upload (the mesh mode is the pack's).
        bench.set_frame(3_000, 0.0, false);
        {
            let mut r = bench.terrain.borrow_mut();
            r.render_distance = 8;
            let deferred = r.upload_all(&bench.gpu, meshes(&scene(), mesh), bench.cam.pos);
            assert!(deferred.is_empty());
        }
        bench
    }

    fn set_frame(&mut self, world_time: i64, partial_ticks: f32, clock_held: bool) {
        let env = Environment {
            world_time,
            ..Environment::default()
        };
        let frame = PackFrame::compute(&FrameInputs {
            environment: &env,
            biome_temperature: 0.8,
            fog_color: Vec3::ONE,
            fog: FogMode::CLEAR,
            fog_distance: FogDistance::Planar,
            eye_medium: EyeMedium::Air,
            render_distance_chunks: 8,
            camera: &self.cam,
            width: WIDTH,
            height: HEIGHT,
            partial_ticks,
            clock_held,
            seconds: 0.0,
            frame_index: self.frame_index,
            in_world: true,
            sun_tilt: self.core.config().sun_tilt,
        });
        self.frame_index = self.frame_index.wrapping_add(1);
        self.core.set_frame(&self.gpu, frame);
    }

    fn render(&mut self, world_time: i64, partial_ticks: f32, clock_held: bool) -> Vec<u8> {
        self.set_frame(world_time, partial_ticks, clock_held);
        let frame = self.gpu.acquire().expect("acquire");
        let index = self
            .phases
            .render(&self.gpu, frame.view(), &self.cam, partial_ticks);
        self.gpu
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: None,
            })
            .expect("poll");
        drop(frame);
        self.gpu.read_pixels().expect("read pixels")
    }

    fn shadow_renders(&self) -> u64 {
        self.terrain.borrow().shadow_pass_stats().1
    }
}

fn luma(p: &[u8]) -> f32 {
    0.2126 * f32::from(p[0]) + 0.7152 * f32::from(p[1]) + 0.0722 * f32::from(p[2])
}

fn jumps(a: &[u8], b: &[u8]) -> (u32, f32) {
    let mut count = 0;
    let mut max = 0.0_f32;
    for (p, q) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let d = (luma(p) - luma(q)).abs();
        if d >= JUMP {
            count += 1;
        }
        max = max.max(d);
    }
    (count, max)
}

/// Before, after and the difference (x8), side by side at 2x.
fn save_strip(name: &str, before: &[u8], after: &[u8]) {
    let Some(dir) = std::env::var_os("MC_W22_SHD_PNG_DIR") else {
        return;
    };
    let (w, h) = (WIDTH as usize, HEIGHT as usize);
    let scale = 2;
    let (sw, sh) = (w * 3 * scale, h * scale);
    let mut strip = vec![0_u8; sw * sh * 4];
    for tile in 0..3 {
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                let px: [u8; 4] = match tile {
                    0 => [before[i], before[i + 1], before[i + 2], 255],
                    1 => [after[i], after[i + 1], after[i + 2], 255],
                    _ => {
                        let d = |c: usize| {
                            ((f32::from(after[i + c]) - f32::from(before[i + c])).abs() * 8.0)
                                .min(255.0) as u8
                        };
                        [d(0), d(1), d(2), 255]
                    }
                };
                for dy in 0..scale {
                    for dx in 0..scale {
                        let o = ((y * scale + dy) * sw + tile * w * scale + x * scale + dx) * 4;
                        strip[o..o + 4].copy_from_slice(&px);
                    }
                }
            }
        }
    }
    let path = PathBuf::from(dir).join(format!("{name}.png"));
    image::save_buffer(&path, &strip, sw as u32, sh as u32, image::ColorType::Rgba8)
        .expect("save png");
    eprintln!("wrote {}", path.display());
}

/// What a timeline did: per step (frame to frame) the jumping pixels, the worst step's frames,
/// and the shadow-map draws.
struct Timeline {
    steps: Vec<(u32, f32)>,
    worst: Option<(usize, Vec<u8>, Vec<u8>)>,
    draws: u64,
}

impl Timeline {
    fn run(bench: &mut Bench, start: i64, ticks: i64, held: bool) -> Timeline {
        let before = bench.shadow_renders();
        let mut previous: Option<Vec<u8>> = None;
        let mut steps = Vec::new();
        let mut worst: Option<(usize, Vec<u8>, Vec<u8>)> = None;
        let mut worst_count = 0;
        for tick in 0..ticks {
            let time = if held { start } else { start + tick };
            for f in 0..FRAMES_PER_TICK {
                let pt = f as f32 / FRAMES_PER_TICK as f32;
                let frame = bench.render(time, pt, held);
                if let Some(prev) = previous.as_ref() {
                    let step = jumps(prev, &frame);
                    if step.0 > worst_count {
                        worst_count = step.0;
                        worst = Some((steps.len(), prev.clone(), frame.clone()));
                    }
                    steps.push(step);
                }
                previous = Some(frame);
            }
        }
        Timeline {
            steps,
            worst,
            draws: bench.shadow_renders() - before,
        }
    }

    fn moving_steps(&self) -> usize {
        self.steps.iter().filter(|s| s.0 > 0).count()
    }

    fn max_jumps(&self) -> u32 {
        self.steps.iter().map(|s| s.0).max().unwrap_or(0)
    }

    fn total(&self) -> u64 {
        self.steps.iter().map(|s| u64::from(s.0)).sum()
    }

    fn report(&self, label: &str) -> String {
        let largest = self.steps.iter().map(|s| s.1).fold(0.0_f32, f32::max);
        format!(
            "{label:<34} {} steps: {} with jumping pixels, at most {} jumping ({:.2}% of the \
             frame), {} in all, largest change {largest:.1} levels; {} map draws",
            self.steps.len(),
            self.moving_steps(),
            self.max_jumps(),
            100.0 * f64::from(self.max_jumps()) / f64::from(WIDTH * HEIGHT),
            self.total(),
            self.draws,
        )
    }
}

fn mode() -> &'static str {
    if std::env::var("MC_W22_SHD_STABLE").is_ok_and(|v| v.trim() == "0") {
        "old"
    } else {
        "new"
    }
}

/// A held clock draws the same picture every frame, at every partial tick: at noon and at the
/// clock times whose one-tick swing straddled a direction step (09:15, 21:30 at a 25 degree tilt).
#[test]
fn a_held_clock_draws_the_same_picture_every_frame() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(gpu) = gpu() else { return };
    let mut bench = Bench::new(atlas, gpu, 2048);
    let mut lines = Vec::new();
    let mut moving = Vec::new();
    for (label, time) in [("12:00", 6_000), ("09:15", 3_250), ("21:30", 15_500)] {
        // Settle on the held time first (the first frames build the map).
        for _ in 0..3 {
            bench.render(time, 0.0, true);
        }
        let run = Timeline::run(&mut bench, time, 30, true);
        lines.push(run.report(&format!("{} held {label}", mode())));
        if let Some((i, before, after)) = &run.worst {
            save_strip(
                &format!("gpu_{}_held_{}_step{i}", mode(), &label[..2]),
                before,
                after,
            );
        }
        if run.max_jumps() > 0 {
            moving.push(format!("{label}: {}", run.max_jumps()));
        }
    }
    eprintln!("{}", lines.join("\n"));
    assert!(
        moving.is_empty(),
        "a held clock moves the shading: {}",
        moving.join(", ")
    );
}

/// A running clock (the sun crawling, 60 fps): the shading changes only on the frames where the
/// map takes its next direction step, never in between (no slide, no snap back).
#[test]
fn a_running_clock_moves_the_shading_only_in_its_steps() {
    let Some(atlas) = vanilla_atlas() else { return };
    let mut lines = Vec::new();
    let mut sliding = Vec::new();
    for resolution in [2048, 4096] {
        let Some(gpu) = gpu() else { return };
        let mut bench = Bench::new(atlas, gpu, resolution);
        for (label, start) in [("09:00", 3_000), ("15:30", 9_500)] {
            for _ in 0..3 {
                bench.render(start, 0.0, false);
            }
            let run = Timeline::run(&mut bench, start, 200, false);
            lines.push(run.report(&format!("{} running {label} {resolution}", mode())));
            if let Some((i, before, after)) = &run.worst {
                save_strip(
                    &format!(
                        "gpu_{}_running_{}_{resolution}_step{i}",
                        mode(),
                        &label[..2]
                    ),
                    before,
                    after,
                );
            }
            // Only the steps where the map was redrawn may move anything.
            if run.moving_steps() as u64 > run.draws {
                sliding.push(format!(
                    "{label} {resolution}: {} moving steps for {} map draws",
                    run.moving_steps(),
                    run.draws
                ));
            }
        }
    }
    eprintln!("{}", lines.join("\n"));
    assert!(
        sliding.is_empty(),
        "the shading moves between redraws: {}",
        sliding.join("; ")
    );
}
