//! Pictures of the falling-block lighting fix.
//!
//! java: `RenderFallingBlock.doRender` lights the block through
//! `BlockModelRenderer.renderModel`, whose smooth path needs
//! `Minecraft.isAmbientOcclusionEnabled()`. mc-rs used to light a falling block with the vanilla
//! defaults (smooth lighting on) whatever the setting, so with smooth lighting off a falling block
//! kept dark ambient-occlusion corners its terrain twin did not have. This draws a terrain sand
//! block and a falling sand entity side by side, in the same surroundings (a floor, a wall behind,
//! a pillar beside each), once with `ao:0` and once with `ao:2` in `options.txt`, on the SkyWars
//! benchmark capture's world, and saves both frames.
//!
//! ```text
//! MC_W18_PA_PNG_DIR=<dir> MC_JARS_DIR=<jars> cargo test -p mc-client --test w18_pa_pictures \
//!     -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Facing south, the falling block is on the left and its terrain twin on the right. Besides the
//! pictures (and a 2x crop, `ao:0` above `ao:2`) it checks that the falling block's pixels follow
//! the setting, which they did not before the fix, and that its front face matches its twin's.
//! The second check also caught the falling block being drawn through the lit entity pipeline:
//! java's `RenderFallingBlock` calls `disableLighting()`, so the entity diffuse light must not
//! darken its sides.

use std::path::{Path, PathBuf};

use mc_client::app::campath::CameraScript;
use mc_client::app::config::LoginMode;
use mc_client::sync::TrackedKind;
use mc_client::{Client, ClientConfig};
use mc_data::{BlockState, EntityType};
use mc_math::{BlockPos, DVec3};
use mc_physics::state::MoveInput;
use mc_render::{Camera, Gpu, GpuOptions, PhaseList};

const WIDTH: u32 = 960;
const HEIGHT: u32 = 540;
/// Ticks to run before building the stage: the join and the chunks have settled by then.
const SETTLE_TICKS: u32 = 120;
/// The SkyWars benchmark capture, relative to this crate.
const CAPTURE: &str =
    "../../optimization/scratch/skywars-scene-2026-11-05/captures/sw-mega-doubles-mid.mccap";

struct Rig {
    client: Client,
    gpu: Gpu,
    phases: PhaseList,
    camera: Camera,
    _dir: tempfile::TempDir,
}

impl Rig {
    fn new(ao: i32) -> Option<Rig> {
        let capture = Path::new(env!("CARGO_MANIFEST_DIR")).join(CAPTURE);
        if !capture.is_file() {
            eprintln!("SKIP: no SkyWars capture at {}", capture.display());
            return None;
        }
        let dir = tempfile::tempdir().ok()?;
        // Smooth lighting is baked into the terrain meshes, so it is set before the client
        // starts, exactly as a player's `options.txt` does.
        std::fs::write(dir.path().join("options.txt"), format!("ao:{ao}\n")).ok()?;
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("BenchBot".to_owned()),
            replay: Some(capture),
            replay_step_micros: 50_000,
            replay_camera: CameraScript::Fixed,
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("SKIP: no vanilla jar (set MC_JARS_DIR)");
            return None;
        }
        let mut client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        assert_eq!(
            client.settings.ambient_occlusion, ao,
            "options.txt reached the settings"
        );
        let gpu = Gpu::headless(
            WIDTH,
            HEIGHT,
            GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        )
        .ok()?;
        let mut phases = PhaseList::new();
        client.install(&gpu, &mut phases, WIDTH, HEIGHT);
        client.connect("replay.invalid", 25565);
        let mut camera = Camera::default();
        camera.set_viewport(WIDTH, HEIGHT);
        Some(Rig {
            client,
            gpu,
            phases,
            camera,
            _dir: dir,
        })
    }

    fn render(&mut self) -> Vec<u8> {
        self.client.frame(&self.gpu, &mut self.camera, 1.0);
        let frame = self.gpu.acquire().expect("acquire");
        self.phases
            .render(&self.gpu, frame.view(), &self.camera, 1.0);
        let pixels = frame.read_pixels(&self.gpu).expect("read back");
        frame.present(&self.gpu.queue);
        pixels
    }

    fn tick(&mut self) {
        self.client.poll();
        self.client.tick(&MoveInput::default());
        let _ = self.render();
    }

    fn viewer(&self) -> DVec3 {
        self.client
            .session()
            .and_then(|s| s.predictor())
            .map(|p| p.state.pos)
            .expect("the viewer exists")
    }

    fn pose(&mut self, feet: DVec3, yaw: f32, pitch: f32) {
        let session = self.client.session_mut().expect("a session");
        let predictor = session.predictor_mut().expect("a player");
        predictor.state.pos = feet;
        predictor.state.prev_pos = feet;
        predictor.state.yaw = yaw;
        predictor.state.pitch = pitch;
    }

    /// A stone stage in open air: a floor, a wall behind, a pillar on the outer side of each
    /// slot, a sand block in the left slot (terrain) and a falling sand entity in the right one.
    /// Both slots have the same neighbours, so with equal lighting rules they look the same.
    fn build_stage(&mut self, origin: BlockPos) -> (BlockPos, BlockPos) {
        let stone = mc_data::block_by_name("stone")
            .expect("stone")
            .default_state();
        let sand = mc_data::block_by_name("sand")
            .expect("sand")
            .default_state();
        let terrain = origin.add(-2, 1, 0);
        let falling = origin.add(2, 1, 0);
        let session = self.client.session_mut().expect("a session");
        let state = session.state_mut();
        let world = &mut state.world;
        for x in -5..=5 {
            for z in -3..=2 {
                for y in 0..=4 {
                    world.set_block(origin.add(x, y, z), BlockState::AIR);
                }
                world.set_block(origin.add(x, 0, z), stone);
            }
            for y in 1..=3 {
                world.set_block(origin.add(x, y, 1), stone);
            }
        }
        for slot in [terrain, falling] {
            let outer = if slot.x < origin.x { -1 } else { 1 };
            for y in 0..=1 {
                world.set_block(slot.add(outer, y, 0), stone);
            }
        }
        world.set_block(terrain, sand);
        (terrain, falling)
    }

    /// The falling sand entity, spawned after the tick that enqueued the stage for meshing so
    /// that no client tick moves it.
    fn spawn_falling(&mut self, falling: BlockPos) {
        let sand = mc_data::block_by_name("sand")
            .expect("sand")
            .default_state();
        let session = self.client.session_mut().expect("a session");
        let state = session.state_mut();
        let entity = state.entities.spawn(
            &mut state.world,
            900_001,
            TrackedKind::Type(EntityType::FallingSand),
            None,
            [falling.x * 32 + 16, falling.y * 32, falling.z * 32 + 16],
            0.0,
            0.0,
            DVec3::ZERO,
            mc_world::Metadata::new(),
        );
        state
            .world
            .ecs
            .insert_one(
                entity,
                mc_client::app::entrender::ObjectData(sand.to_java_state_id()),
            )
            .expect("object data");
    }
}

/// Mean RGB of a square of pixels around (`cx`, `cy`).
fn patch(pixels: &[u8], cx: u32, cy: u32, half: u32) -> [f64; 3] {
    let mut sum = [0.0; 3];
    let mut n = 0.0;
    for y in cy - half..=cy + half {
        for x in cx - half..=cx + half {
            let i = ((y * WIDTH + x) * 4) as usize;
            for c in 0..3 {
                sum[c] += f64::from(pixels[i + c]);
            }
            n += 1.0;
        }
    }
    sum.map(|s| s / n)
}

fn shoot(ao: i32, out: &Path) -> Option<Vec<u8>> {
    let mut rig = Rig::new(ao)?;
    for _ in 0..SETTLE_TICKS {
        rig.tick();
    }
    let me = rig.viewer();
    let origin = BlockPos::new(
        me.x.floor() as i32,
        me.y.floor() as i32 + 12,
        me.z.floor() as i32 + 6,
    );
    let (terrain, falling) = rig.build_stage(origin);
    eprintln!("ao:{ao} viewer {me:?} stage {origin:?} terrain {terrain:?} falling {falling:?}");
    // Stand north of the stage (facing south, +Z), a little above it. Facing south, +X (the
    // falling slot) is on the left of the picture and the terrain slot on the right.
    let feet = DVec3::new(
        f64::from(origin.x) + 0.5,
        f64::from(origin.y) + 1.5,
        f64::from(origin.z) - 4.5,
    );
    // The edits reach the mesh queue in the next client tick (`enqueue_dirty`).
    rig.pose(feet, 0.0, 22.0);
    rig.tick();
    rig.spawn_falling(falling);
    let mut pixels = Vec::new();
    // Let the edited sections re-mesh (renders only: the replay does not advance).
    for _ in 0..240 {
        rig.pose(feet, 0.0, 22.0);
        pixels = rig.render();
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let path = out.join(format!("falling_block_ao{ao}.png"));
    image::save_buffer(&path, &pixels, WIDTH, HEIGHT, image::ColorType::Rgba8).expect("png");
    eprintln!("wrote {}", path.display());
    Some(pixels)
}

#[test]
#[ignore = "writes PNGs of a real frame; run by hand (see the module docs)"]
fn falling_block_lighting_pictures() {
    let Some(out) = std::env::var_os("MC_W18_PA_PNG_DIR").map(PathBuf::from) else {
        eprintln!("SKIP: set MC_W18_PA_PNG_DIR");
        return;
    };
    std::fs::create_dir_all(&out).expect("png dir");
    let Some(flat) = shoot(0, &out) else { return };
    let Some(smooth) = shoot(2, &out) else { return };
    // The interiors of the two front faces (falling block left, terrain block right).
    let falling_face = (322, 252);
    let terrain_face = (628, 250);
    let mut falling_means = Vec::new();
    for (label, pixels) in [("ao:0", &flat), ("ao:2", &smooth)] {
        let falling = patch(pixels, falling_face.0, falling_face.1, 24);
        let terrain = patch(pixels, terrain_face.0, terrain_face.1, 24);
        eprintln!("{label}: falling face {falling:?} terrain face {terrain:?}");
        // Same block, same surroundings, same lighting rules: the faces match up to the sand
        // texture's own variation between the two (mirrored) sample squares. Before the
        // unlit-pipeline fix the falling face was ~0.74x its twin (the entity diffuse light).
        for c in 0..3 {
            let ratio = falling[c] / terrain[c];
            assert!(
                (0.93..=1.07).contains(&ratio),
                "{label}: the falling block matches its terrain twin (channel {c} ratio {ratio})"
            );
        }
        falling_means.push(falling);
    }
    // Before the fix the falling block was lit with smooth lighting whatever the setting, so its
    // pixels were the same in both frames.
    let change: f64 = (0..3)
        .map(|c| (falling_means[0][c] - falling_means[1][c]).abs())
        .sum();
    assert!(
        change > 6.0,
        "the falling block follows the setting: {falling_means:?}"
    );

    // A 2x crop of both blocks, `ao:0` above `ao:2`.
    let (x0, y0, w, h) = (230_u32, 150_u32, 500_u32, 170_u32);
    let mut crop = vec![0_u8; (w * 2 * h * 4 * 4) as usize];
    for (row, pixels) in [&flat, &smooth].into_iter().enumerate() {
        for y in 0..h * 2 {
            for x in 0..w * 2 {
                let src = (((y0 + y / 2) * WIDTH + x0 + x / 2) * 4) as usize;
                let dst = (((row as u32 * h * 2 + y) * w * 2 + x) * 4) as usize;
                crop[dst..dst + 4].copy_from_slice(&pixels[src..src + 4]);
            }
        }
    }
    let path = out.join("falling_block_crops.png");
    image::save_buffer(&path, &crop, w * 2, h * 4, image::ColorType::Rgba8).expect("png");
    eprintln!("wrote {}", path.display());
}
