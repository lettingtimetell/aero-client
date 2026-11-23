//! The Classic shader pack's picture and timing harness.
//!
//! Drives the real client over the SkyWars benchmark capture (`sw-mega-doubles-mid`, Dwarf
//! Fortress, 40 players) on a headless GPU, one tick per frame, then poses the camera and writes
//! PNGs with the pack off and on: noon, sunset and night; a player at 5 and at 30 blocks; a pool of
//! water set into the client's world beside the viewer. The rig is the (`tw_underwater_pictures.rs`), copied so the two lanes never edit one file.
//!
//! ```text
//! MC_TW_SA_PNG_DIR=<dir> MC_JARS_DIR=<jars> cargo test -p mc-client --test tw_classic_pictures \
//!     -- --ignored --nocapture --test-threads=1
//! MC_GPU_TIMING=1 ... classic_gpu_timings          (GPU time per phase at 1080p)
//! ```
//!
//! It also checks on the real frame that switching the pack off again restores vanilla's frame
//! (only the wall-clock enchantment glint differs) and that the pack off never asks for the scene
//! copy. Exact byte identity of the pack-off terrain is proven by `mc-render`'s
//! `tests/sa_pack_terrain.rs`.

use std::path::{Path, PathBuf};

use mc_client::app::campath::CameraScript;
use mc_client::app::config::LoginMode;
use mc_client::sync::TrackedKind;
use mc_client::{Client, ClientConfig};
use mc_data::{BlockId, BlockState};
use mc_math::{BlockPos, DVec3};
use mc_physics::state::MoveInput;
use mc_render::{Camera, Gpu, GpuOptions, PhaseList};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
/// Ticks to run before posing: the join, the chunks and the players have settled by 280.
const SETTLE_TICKS: u32 = 285;
/// The SkyWars benchmark capture, relative to this crate.
const CAPTURE: &str =
    "../../optimization/scratch/skywars-scene-2026-11-05/captures/sw-mega-doubles-mid.mccap";
const ID: &str = "shaders_classic";

struct Rig {
    client: Client,
    gpu: Gpu,
    phases: PhaseList,
    camera: Camera,
    out: PathBuf,
    width: u32,
    height: u32,
    _dir: tempfile::TempDir,
}

/// Where the camera stands and looks.
#[derive(Debug, Clone, Copy)]
struct Pose {
    feet: DVec3,
    yaw: f32,
    pitch: f32,
}

impl Pose {
    /// Stand at `feet` and look at `target`.
    fn looking_at(feet: DVec3, target: DVec3) -> Pose {
        let eye = feet + DVec3::new(0.0, 1.62, 0.0);
        let d = target - eye;
        let horizontal = (d.x * d.x + d.z * d.z).sqrt();
        Pose {
            feet,
            yaw: (-d.x).atan2(d.z).to_degrees() as f32,
            pitch: (-d.y).atan2(horizontal).to_degrees() as f32,
        }
    }
}

impl Rig {
    fn new(width: u32, height: u32) -> Option<Rig> {
        let out = PathBuf::from(std::env::var_os("MC_TW_SA_PNG_DIR")?);
        std::fs::create_dir_all(&out).ok()?;
        let capture = Path::new(env!("CARGO_MANIFEST_DIR")).join(CAPTURE);
        if !capture.is_file() {
            eprintln!("SKIP: no SkyWars capture at {}", capture.display());
            return None;
        }
        let dir = tempfile::tempdir().ok()?;
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
        let gpu = Gpu::headless(
            width,
            height,
            GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        )
        .ok()?;
        let mut phases = PhaseList::new();
        client.install(&gpu, &mut phases, width, height);
        client.connect("replay.invalid", 25565);
        let mut camera = Camera::default();
        camera.set_viewport(width, height);
        Some(Rig {
            client,
            gpu,
            phases,
            camera,
            out,
            width,
            height,
            _dir: dir,
        })
    }

    /// One frame without a tick and without a poll, so the scene holds still.
    fn render(&mut self) -> Vec<u8> {
        self.client.frame(&self.gpu, &mut self.camera, 1.0);
        let frame = self.gpu.acquire().expect("acquire");
        self.phases
            .render(&self.gpu, frame.view(), &self.camera, 1.0);
        let pixels = frame.read_pixels(&self.gpu).expect("read back");
        frame.present(&self.gpu.queue);
        pixels
    }

    /// One poll, one tick and one frame.
    fn tick(&mut self) {
        self.client.poll();
        self.client.tick(&MoveInput::default());
        self.client.frame(&self.gpu, &mut self.camera, 1.0);
        let frame = self.gpu.acquire().expect("acquire");
        self.phases
            .render(&self.gpu, frame.view(), &self.camera, 1.0);
        frame.present(&self.gpu.queue);
    }

    fn pose(&mut self, pose: Pose) {
        let session = self.client.session_mut().expect("a session");
        let predictor = session.predictor_mut().expect("a player");
        predictor.state.pos = pose.feet;
        predictor.state.yaw = pose.yaw;
        predictor.state.pitch = pose.pitch;
    }

    fn settings<R>(&mut self, edit: impl FnOnce(&mut mc_mods::ModSettings) -> R) {
        let _ = edit(self.client.mod_settings_mut());
    }

    /// Let a mesher-mode change land: the pack's mesh (flat shade, flags) re-meshes the world over
    /// a few hundred frames (the mesh queue's per-frame budget). Renders only, so the replay does
    /// not advance and the scene holds still.
    fn settle_meshes(&mut self) {
        for _ in 0..300 {
            self.render();
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    /// Render a few frames (settings reach the phases one frame late) and save the last one.
    fn shot(&mut self, name: &str, pose: Pose) -> Vec<u8> {
        let mut pixels = Vec::new();
        for _ in 0..6 {
            self.pose(pose);
            pixels = self.render();
        }
        let path = self.out.join(format!("{name}.png"));
        image::save_buffer(
            &path,
            &pixels,
            self.width,
            self.height,
            image::ColorType::Rgba8,
        )
        .expect("write png");
        eprintln!("wrote {}", path.display());
        pixels
    }

    fn viewer(&self) -> DVec3 {
        self.client
            .session()
            .and_then(|s| s.predictor())
            .map(|p| p.state.pos)
            .expect("the viewer exists")
    }

    /// The tracked remote players' feet positions.
    fn players(&self) -> Vec<DVec3> {
        let session = self.client.session().expect("a session");
        let world = &session.state().world;
        session
            .state()
            .entities
            .iter()
            .filter(|(_, e)| e.kind == TrackedKind::Player)
            .map(|(_, e)| e.render_position(world, 1.0))
            .collect()
    }

    fn solid(&self, pos: DVec3) -> bool {
        let session = self.client.session().expect("a session");
        let state = session.state().world.block_at(BlockPos::from_vec3(pos));
        state.props().blocks_movement()
    }

    /// A pose `distance` blocks from `target` (horizontally, same height) with nothing solid in
    /// between, looking at its chest. Tries 32 directions.
    fn pose_near(&self, target: DVec3, distance: f64) -> Option<Pose> {
        let chest = target + DVec3::new(0.0, 1.2, 0.0);
        for k in 0..32 {
            let angle = f64::from(k) / 32.0 * std::f64::consts::TAU;
            let dir = DVec3::new(angle.cos(), 0.0, angle.sin());
            let feet = target + dir * distance;
            let eye = feet + DVec3::new(0.0, 1.62, 0.0);
            let blocked = (0..=(distance * 4.0) as i32).any(|i| {
                let t = f64::from(i) / (distance * 4.0);
                let p = eye + (chest - eye) * t;
                t > 0.05 && t < 0.9 && self.solid(p)
            });
            if !blocked && !self.solid(eye) && !self.solid(feet + DVec3::new(0.0, 0.5, 0.0)) {
                return Some(Pose::looking_at(feet, chest));
            }
        }
        None
    }

    /// Sink a pool: the top `depth` blocks of the ground in a square around `center` become
    /// still water.
    fn pool(&mut self, center: BlockPos, half: i32, depth: i32) {
        let session = self.client.session_mut().expect("a session");
        let world = &mut session.state_mut().world;
        let water = BlockState::new(BlockId(9), 0);
        for x in -half..=half {
            for z in -half..=half {
                for y in 0..depth {
                    world.set_block(
                        BlockPos::new(center.x + x, center.y - y, center.z + z),
                        water,
                    );
                }
                // Clear anything standing on the pool.
                for y in 1..4 {
                    let p = BlockPos::new(center.x + x, center.y + y, center.z + z);
                    if !world.block_at(p).props().is_air() {
                        world.set_block(p, BlockState::AIR);
                    }
                }
            }
        }
    }
}

fn time(s: &mut mc_mods::ModSettings, hours: f32) {
    s.set_enabled("time_changer", true);
    s.set_f32("time_changer", "time", hours);
}

/// The share of pixels that differ between two frames of the same pose.
fn differing(a: &[u8], b: &[u8]) -> f64 {
    let differ = a
        .chunks_exact(4)
        .zip(b.chunks_exact(4))
        .filter(|(p, q)| p[..3] != q[..3])
        .count();
    differ as f64 / (a.len() / 4) as f64
}

#[test]
#[ignore = "writes PNGs of the real SkyWars scene; run by hand (see the module docs)"]
fn classic_pictures() {
    let Some(mut rig) = Rig::new(WIDTH, HEIGHT) else {
        return;
    };
    for _ in 0..SETTLE_TICKS {
        rig.tick();
    }
    let me = rig.viewer();
    eprintln!("viewer feet {me:?}");
    // Across the map toward mid (+Z), from a little above the plaza.
    let vista = Pose {
        feet: me + DVec3::new(0.0, 6.0, -10.0),
        yaw: 0.0,
        pitch: 10.0,
    };
    // Toward the west, where the sun sets.
    let west = Pose {
        feet: me + DVec3::new(6.0, 5.0, 0.0),
        yaw: 90.0,
        pitch: 4.0,
    };

    for (label, hours) in [("noon", 12.0), ("sunset", 18.4), ("night", 0.0)] {
        rig.settings(|s| {
            time(s, hours);
            s.set_enabled(ID, false);
        });
        rig.tick();
        rig.tick();
        rig.settle_meshes();
        let off = rig.shot(&format!("{label}_off_vista"), vista);
        assert!(
            !rig.phases.scene_mode(),
            "the pack off never asks for the scene copy"
        );
        rig.shot(&format!("{label}_off_west"), west);
        rig.settings(|s| s.set_enabled(ID, true));
        rig.settle_meshes();
        let on = rig.shot(&format!("{label}_on_vista"), vista);
        assert!(rig.phases.scene_mode(), "the pack draws");
        rig.shot(&format!("{label}_on_west"), west);
        let changed = differing(&off, &on);
        eprintln!("{label}: pack vs vanilla {changed:.4}");
        assert!(changed > 0.5, "the pack changes the picture");
    }

    // Players at 5 and 30 blocks, at noon.
    rig.settings(|s| time(s, 12.0));
    rig.tick();
    rig.tick();
    let players = rig.players();
    eprintln!("{} tracked players", players.len());
    for (distance, label) in [(5.0, "player05"), (30.0, "player30")] {
        let Some(pose) = players
            .iter()
            .find_map(|feet| rig.pose_near(*feet, distance))
        else {
            eprintln!("no clear view of a player at {distance} blocks");
            continue;
        };
        rig.settings(|s| s.set_enabled(ID, false));
        rig.settle_meshes();
        rig.shot(&format!("noon_off_{label}"), pose);
        rig.settings(|s| s.set_enabled(ID, true));
        rig.settle_meshes();
        let with = rig.shot(&format!("noon_on_{label}"), pose);
        if distance < 10.0 {
            // Player shadows off: the difference is only the players' cast shadows (and glint).
            rig.settings(|s| s.set_bool(ID, "player_shadows", false));
            let without = rig.shot(&format!("noon_on_{label}_no_player_shadows"), pose);
            eprintln!(
                "{label}: player shadows change {:.4} of the frame",
                differing(&with, &without)
            );
            rig.settings(|s| s.set_bool(ID, "player_shadows", true));
        }
    }

    // Waving plants: two frames 0.6 s apart differ (beyond the glint) only with waving on.
    {
        let pose = vista;
        let pair = |rig: &mut Rig| {
            let a = rig.shot("noon_on_vista_wave_a", pose);
            std::thread::sleep(std::time::Duration::from_millis(600));
            let b = rig.shot("noon_on_vista_wave_b", pose);
            differing(&a, &b)
        };
        rig.settings(|s| s.set_enabled(ID, true));
        rig.settle_meshes();
        let waving = pair(&mut rig);
        rig.settings(|s| {
            s.set_bool(ID, "waving_plants", false);
            s.set_bool(ID, "waving_leaves", false);
        });
        rig.settle_meshes();
        let still = pair(&mut rig);
        eprintln!("waving: frames 0.6 s apart differ {waving:.4} waving, {still:.4} still");
        assert!(waving > still + 0.002, "plants and leaves move");
        rig.settings(|s| {
            s.set_bool(ID, "waving_plants", true);
            s.set_bool(ID, "waving_leaves", true);
        });
    }

    // Off, on, off again at one tick (renders only): the last frame is vanilla's again, bar the
    // wall-clock glint, and the pack off never asks for the scene copy.
    rig.settings(|s| s.set_enabled(ID, false));
    rig.settle_meshes();
    let first = rig.shot("noon_off_vista_before", vista);
    let glint = differing(&first, &rig.shot("noon_off_vista_before_2", vista));
    rig.settings(|s| s.set_enabled(ID, true));
    rig.settle_meshes();
    rig.shot("noon_on_vista_between", vista);
    rig.settings(|s| s.set_enabled(ID, false));
    rig.settle_meshes();
    let again = rig.shot("noon_off_vista_again", vista);
    let d = differing(&first, &again);
    eprintln!("noon: off again vs off {d:.4} (vanilla vs vanilla {glint:.4})");
    assert!(
        d <= glint.max(0.002) * 1.5 + 0.002,
        "switching the pack off restores vanilla's frame ({d})"
    );
    assert!(!rig.phases.scene_mode());

    // Exclusivity on the real client: switching Dreamy on switches Classic off, and Underwater
    // then switches Dreamy off.
    rig.settings(|s| s.set_enabled(ID, true));
    rig.render();
    rig.settings(|s| s.set_enabled("shaders_dreamy", true));
    rig.render();
    rig.render();
    let (classic, dreamy) = {
        let s = rig.client.mod_settings_mut();
        (s.enabled(ID), s.enabled("shaders_dreamy"))
    };
    assert!(!classic && dreamy, "Dreamy on turns Classic off");
    rig.settings(|s| s.set_enabled("shaders_underwater", true));
    rig.render();
    rig.render();
    let (dreamy, underwater) = {
        let s = rig.client.mod_settings_mut();
        (s.enabled("shaders_dreamy"), s.enabled("shaders_underwater"))
    };
    assert!(!dreamy && underwater, "Underwater on turns Dreamy off");
    rig.settings(|s| {
        s.set_enabled("shaders_underwater", false);
        s.set_enabled(ID, false);
    });
    rig.settle_meshes();
    eprintln!("exclusivity: ok");

    // Water: a pool sunk into the plaza beside the viewer, seen low across its surface.
    let ground = BlockPos::from_vec3(me + DVec3::new(0.0, -1.0, 6.0));
    rig.pool(ground, 4, 2);
    rig.settle_meshes();
    let edge = DVec3::new(
        f64::from(ground.x) + 0.5,
        f64::from(ground.y) + 1.0,
        f64::from(ground.z) - 6.5,
    );
    let water = Pose {
        feet: edge,
        yaw: 0.0,
        pitch: 18.0,
    };
    for (label, hours) in [("noon", 12.0), ("sunset", 18.4)] {
        rig.settings(|s| {
            time(s, hours);
            s.set_enabled(ID, false);
        });
        rig.tick();
        rig.settle_meshes();
        rig.shot(&format!("{label}_off_water"), water);
        rig.settings(|s| s.set_enabled(ID, true));
        rig.settle_meshes();
        rig.shot(&format!("{label}_on_water"), water);
    }
}

/// GPU time of the pack at 1080p on the real SkyWars scene, from timestamp queries. Prints the
/// frame's mean GPU time with the pack off and on (same pose, no ticks) and the pack's own
/// phases. Timings are provisional.
#[test]
#[ignore = "GPU timing run; needs MC_GPU_TIMING=1 (see the module docs)"]
fn classic_gpu_timings() {
    if !mc_render::context::gpu_timing_requested() {
        eprintln!("SKIP: set MC_GPU_TIMING=1");
        return;
    }
    let Some(mut rig) = Rig::new(1920, 1080) else {
        return;
    };
    for _ in 0..SETTLE_TICKS {
        rig.tick();
    }
    let me = rig.viewer();
    let vista = Pose {
        feet: me + DVec3::new(0.0, 6.0, -10.0),
        yaw: 0.0,
        pitch: 10.0,
    };
    let frames = 240;
    let measure = |rig: &mut Rig, label: &str| {
        rig.settle_meshes();
        for _ in 0..30 {
            rig.pose(vista);
            rig.render();
        }
        let start = rig.phases.gpu_timings();
        for _ in 0..frames {
            rig.pose(vista);
            rig.render();
        }
        let end = rig.phases.gpu_timings();
        let frame_ms = (end.frame_ns.total_ns - start.frame_ns.total_ns) as f64
            / (end.frame_ns.calls - start.frame_ns.calls).max(1) as f64
            / 1.0e6;
        eprintln!("[{label}] gpu.frame mean {frame_ms:.4} ms over {frames} frames");
        for (name, stats) in &end.scopes {
            let before = start
                .scopes
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, s)| *s)
                .unwrap_or_default();
            let calls = stats.calls - before.calls;
            if calls == 0 {
                continue;
            }
            let ms = (stats.total_ns - before.total_ns) as f64 / calls as f64 / 1.0e6;
            if name.contains("shaderpack") || name.starts_with("terrain") || ms > 0.02 {
                eprintln!("[{label}]   {name:<40} {ms:.4} ms ({calls} frames)");
            }
        }
        frame_ms
    };
    rig.settings(|s| {
        time(s, 15.0);
        s.set_enabled(ID, false);
    });
    rig.tick();
    let off = measure(&mut rig, "off");
    rig.settings(|s| s.set_enabled(ID, true));
    let on = measure(&mut rig, "on (defaults)");
    rig.settings(|s| {
        s.set_choice(ID, "shadow_quality", 2);
        s.set_f32(ID, "shadow_distance", 192.0);
    });
    let high = measure(&mut rig, "on (4096 shadows, 192 blocks)");
    rig.settings(|s| {
        s.set_choice(ID, "shadow_quality", 1);
        s.set_f32(ID, "shadow_distance", 96.0);
        s.set_enabled(ID, false);
    });
    let off_again = measure(&mut rig, "off again");
    eprintln!(
        "pack cost (provisional): defaults {:+.4} ms, high shadows {:+.4} ms; off vs off again \
         {:+.4} ms",
        on - off,
        high - off,
        off_again - off
    );
}
