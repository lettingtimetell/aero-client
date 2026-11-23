//! The Underwater shader pack's picture and timing harness.
//!
//! Drives the real client over the SkyWars benchmark capture (`sw-mega-doubles-mid`, Dwarf
//! Fortress, 40 players) on a headless GPU, one tick per frame, then poses the camera and writes
//! PNGs with the pack off and on: noon, sunset and night; both modes; a player at 5 and at 30
//! blocks (with the water presets at 30); looking up; and the eye really under water (a pool of
//! still water set into the client's world around the camera, which the mesher picks up like any
//! block change).
//!
//! ```text
//! MC_TW_SC_PNG_DIR=<dir> MC_JARS_DIR=<jars> cargo test -p mc-client --test tw_underwater_pictures \
//!     -- --ignored --nocapture --test-threads=1
//! MC_GPU_TIMING=1 ... skywars_gpu_timings          (GPU time per phase at 1080p)
//! ```
//!
//! It also checks the off path on the real frame: with the pack off, and in "Only under water"
//! with the eye in air, no phase asks for the scene copy and only the enchantment glint (which
//! follows the wall clock) differs from vanilla. Exact byte identity is proven by the synthetic
//! GPU test in `mc-render/src/shaderpack/underwater.rs`.

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
const ID: &str = "shaders_underwater";

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
        let out = PathBuf::from(std::env::var_os("MC_TW_SC_PNG_DIR")?);
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

    /// One frame without a tick and without a poll, so the replay feeds no packets and the scene
    /// holds still (only the enchantment glint, which follows the wall clock, still moves).
    fn render(&mut self) -> Vec<u8> {
        self.client.frame(&self.gpu, &mut self.camera, 1.0);
        let frame = self.gpu.acquire().expect("acquire");
        self.phases
            .render(&self.gpu, frame.view(), &self.camera, 1.0);
        let pixels = frame.read_pixels(&self.gpu).expect("read back");
        frame.present(&self.gpu.queue);
        pixels
    }

    /// One poll, one tick and one frame. Re-meshing and the sky's time of day happen in the tick.
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

    /// Render (ticking now and then, which is when dirty sections are queued) until the mesher
    /// is idle for a few frames: every chunk in view is meshed and uploaded, so later frames of
    /// one pose differ only by the enchantment glint. At most a minute.
    fn settle(&mut self, pose: Pose) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        let mut quiet = 0;
        let mut frames = 0u32;
        let busy = |client: &Client| {
            client.mesh_queue_len()
                + client.mesh_queue_deferred_len()
                + client.mesh_queue_in_flight_len()
                + client
                    .render()
                    .map_or(0, |r| r.terrain.borrow().pending_ready_upload_count())
        };
        while std::time::Instant::now() < deadline && quiet < 5 {
            self.pose(pose);
            self.render();
            frames += 1;
            if frames.is_multiple_of(20) {
                self.tick();
            }
            quiet = if busy(&self.client) == 0 {
                quiet + 1
            } else {
                0
            };
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        eprintln!(
            "settled after {frames} frames (mesher backlog {})",
            busy(&self.client)
        );
    }

    fn settings<R>(&mut self, edit: impl FnOnce(&mut mc_mods::ModSettings) -> R) {
        let _ = edit(self.client.mod_settings_mut());
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

    /// Fill the air in a box around `center` with still water.
    fn flood(&mut self, center: BlockPos, half: i32, below: i32, above: i32) {
        let session = self.client.session_mut().expect("a session");
        let world = &mut session.state_mut().world;
        let water = BlockState::new(BlockId(9), 0);
        for x in -half..=half {
            for z in -half..=half {
                for y in -below..=above {
                    let p = BlockPos::new(center.x + x, center.y + y, center.z + z);
                    if world.block_at(p).props().is_air() {
                        world.set_block(p, water);
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

/// The process-wide caustic bake (the pack starts it on a worker thread; wait for it here so the
/// first picture already has caustics).
fn wait_for_caustics() {
    let _ = mc_render::shaderpack::caustics::Caustics::default().wait();
}

/// A 3x nearest-neighbour enlargement of the frame's middle third (for tiny details: snow).
fn zoom(out: &Path, name: &str, frame: &[u8], width: u32, height: u32) {
    let (cw, ch) = (width / 3, height / 3);
    let (x0, y0) = (width / 3, height / 6);
    let mut big = Vec::with_capacity((cw * ch * 9 * 4) as usize);
    for y in 0..ch * 3 {
        for x in 0..cw * 3 {
            let i = (((y0 + y / 3) * width + x0 + x / 3) * 4) as usize;
            big.extend_from_slice(&frame[i..i + 4]);
        }
    }
    let path = out.join(format!("{name}.png"));
    image::save_buffer(&path, &big, cw * 3, ch * 3, image::ColorType::Rgba8).expect("write");
}

/// Where two frames differ: a mask PNG (red, brighter = larger difference, the frame dimmed
/// elsewhere) and a printed 8x6 grid of the share of differing pixels per cell.
fn diff_report(out: &Path, name: &str, a: &[u8], b: &[u8], width: u32, height: u32) {
    let mut mask = Vec::with_capacity(a.len());
    let (cols, rows) = (8usize, 6usize);
    let mut cells = vec![(0u32, 0u32); cols * rows];
    let mut max = 0u32;
    for (i, (p, q)) in a.chunks_exact(4).zip(b.chunks_exact(4)).enumerate() {
        let d: u32 = (0..3).map(|c| u32::from(p[c].abs_diff(q[c]))).sum();
        max = max.max(d);
        let (x, y) = (i % width as usize, i / width as usize);
        let cell = &mut cells[(y * rows / height as usize) * cols + x * cols / width as usize];
        cell.1 += 1;
        if d > 0 {
            cell.0 += 1;
            mask.extend_from_slice(&[(d * 16).min(255) as u8, 0, 0, 255]);
        } else {
            mask.extend_from_slice(&[p[0] / 4, p[1] / 4, p[2] / 4, 255]);
        }
    }
    let path = out.join(format!("{name}.png"));
    image::save_buffer(&path, &mask, width, height, image::ColorType::Rgba8).expect("write");
    eprintln!("{name}: max channel-sum difference {max}; per cell (% differing):");
    for r in 0..rows {
        let line: Vec<String> = (0..cols)
            .map(|c| {
                let (d, n) = cells[r * cols + c];
                format!("{:5.1}", 100.0 * f64::from(d) / f64::from(n.max(1)))
            })
            .collect();
        eprintln!("  {}", line.join(" "));
    }
}

/// The share of pixels that differ between two frames of the same pose. Two vanilla frames
/// differ only where an enchantment glint is (it follows the wall clock: the hotbar's and the
/// hand's Sharpness sword, players' enchanted gear), a few percent of the frame at most; the pack
/// changes nearly every pixel.
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
fn skywars_pictures() {
    let Some(mut rig) = Rig::new(WIDTH, HEIGHT) else {
        return;
    };
    // Vanilla's HUD vignette eases 1% toward its target on every rendered frame, so two frames
    // of one pose never match at the edges; the Vignette mod off removes it (the pack never
    // touches it either way). What still differs is the glint, which follows the wall clock.
    rig.settings(|s| s.set_enabled("vignette", false));
    for _ in 0..SETTLE_TICKS {
        rig.tick();
    }
    let me = rig.viewer();
    eprintln!("viewer feet {me:?}");
    // The view across the map toward mid (+Z), slightly down.
    let vista = Pose {
        feet: me + DVec3::new(0.0, 6.0, -10.0),
        yaw: 0.0,
        pitch: 8.0,
    };
    let up = Pose {
        pitch: -62.0,
        ..vista
    };
    rig.settle(vista);

    for (label, hours) in [("noon", 12.0), ("sunset", 18.3), ("night", 0.0)] {
        rig.settings(|s| {
            time(s, hours);
            s.set_enabled(ID, false);
        });
        // The sky environment (and with it the Time Changer's time) is refreshed in the tick.
        rig.tick();
        rig.tick();
        rig.settle(vista);
        let off = rig.shot(&format!("{label}_off_vista"), vista);
        assert!(
            !rig.phases.scene_mode(),
            "the pack off never asks for the scene copy"
        );
        let off_2 = rig.shot(&format!("{label}_off_vista_2"), vista);
        let glint = differing(&off, &off_2);
        if label == "noon" {
            let (w, h) = (rig.width, rig.height);
            diff_report(&rig.out, "diff_noon_off_vs_off2", &off, &off_2, w, h);
        }
        rig.settings(|s| {
            s.set_enabled(ID, true);
            s.set_choice(ID, "mode", 0);
        });
        wait_for_caustics();
        let on = rig.shot(&format!("{label}_on_vista"), vista);
        assert!(rig.phases.scene_mode(), "the pack draws");
        let changed = differing(&off, &on);
        eprintln!("{label}: vanilla vs vanilla {glint:.4}, pack vs vanilla {changed:.4}");
        assert!(changed > 0.5, "the pack changes the picture");
        rig.shot(&format!("{label}_on_up"), up);
        if label == "noon" {
            // Refraction wobble on (off by default).
            rig.settings(|s| s.set_bool(ID, "refraction", true));
            rig.shot("noon_on_vista_wobble", vista);
            rig.settings(|s| s.set_bool(ID, "refraction", false));
        }
        if label == "night" || label == "noon" {
            // Marine snow at 100%, and a 3x crop of the middle of the frame to see the specks.
            rig.settings(|s| s.set_f32(ID, "snow_amount", 100.0));
            let snow = rig.shot(&format!("{label}_on_vista_snow100"), vista);
            let (w, h) = (rig.width, rig.height);
            zoom(
                &rig.out,
                &format!("{label}_on_vista_snow100_zoom"),
                &snow,
                w,
                h,
            );
            rig.settings(|s| s.set_f32(ID, "snow_amount", 30.0));
            let snow = rig.shot(&format!("{label}_on_vista_snow30"), vista);
            zoom(
                &rig.out,
                &format!("{label}_on_vista_snow30_zoom"),
                &snow,
                w,
                h,
            );
        }
        if label == "noon" {
            // Only under water, eye in air: the pack off, exactly: no scene copy, and only the
            // glint's pixels differ.
            rig.settings(|s| s.set_choice(ID, "mode", 1));
            let only = rig.shot("noon_onlywater_air_vista", vista);
            assert!(
                !rig.phases.scene_mode(),
                "no scene copy while the eye is in air"
            );
            let d = differing(&off, &only);
            eprintln!("noon: only-under-water in air vs off {d:.4}");
            let (w, h) = (rig.width, rig.height);
            diff_report(&rig.out, "diff_noon_off_vs_onlywater", &off, &only, w, h);
            assert!(
                d <= glint.max(0.002) * 1.5 + 0.002,
                "only the glint differs ({d})"
            );
            rig.settings(|s| s.set_choice(ID, "mode", 0));
        }
    }
    rig.settings(|s| time(s, 12.0));
    rig.tick();
    rig.tick();

    // Players at 5 and 30 blocks (and the water presets on the far one).
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
        rig.shot(&format!("noon_off_{label}"), pose);
        rig.settings(|s| s.set_enabled(ID, true));
        rig.shot(&format!("noon_on_{label}"), pose);
        if distance > 10.0 {
            for (preset, name) in [(1, "ocean"), (2, "deepsea"), (0, "lagoon")] {
                rig.settings(|s| s.set_choice(ID, "preset", preset));
                rig.shot(&format!("noon_on_{label}_{name}"), pose);
            }
        }
    }

    // Off, on, off at one scene state: the last frame is the first (bar the glint), and no
    // phase asks for the scene copy any more.
    rig.settings(|s| s.set_enabled(ID, false));
    rig.settle(vista);
    let before = rig.shot("noon_off_vista_before", vista);
    let glint = differing(&before, &rig.shot("noon_off_vista_before_2", vista));
    rig.settings(|s| s.set_enabled(ID, true));
    rig.shot("noon_on_vista_between", vista);
    rig.settings(|s| s.set_enabled(ID, false));
    let again = rig.shot("noon_off_vista_again", vista);
    let d = differing(&before, &again);
    eprintln!("noon: off again vs off {d:.4} (glint {glint:.4})");
    assert!(
        d <= glint.max(0.002) * 1.5 + 0.002,
        "switching the pack off restores vanilla's frame ({d})"
    );
    assert!(!rig.phases.scene_mode());

    // The eye really under water: a pool around the viewer. Re-meshing is driven by the tick's
    // dirty set, and the meshes arrive from the worker threads a few frames later.
    let pool = BlockPos::from_vec3(me + DVec3::new(0.0, 3.0, -4.0));
    rig.flood(pool, 7, 4, 4);
    for _ in 0..4 {
        rig.tick();
    }
    let under = Pose {
        feet: DVec3::new(
            f64::from(pool.x) + 0.5,
            f64::from(pool.y) - 1.0,
            f64::from(pool.z) + 0.5,
        ),
        yaw: 0.0,
        pitch: 4.0,
    };
    let under_up = Pose {
        pitch: -55.0,
        ..under
    };
    rig.settle(under);
    rig.settings(|s| s.set_enabled(ID, false));
    rig.shot("noon_vanilla_underwater", under);
    rig.shot("noon_vanilla_underwater_up", under_up);
    rig.settings(|s| {
        s.set_enabled(ID, true);
        s.set_choice(ID, "mode", 0);
    });
    rig.shot("noon_on_underwater", under);
    rig.shot("noon_on_underwater_up", under_up);
    rig.settings(|s| s.set_choice(ID, "mode", 1));
    rig.shot("noon_onlywater_underwater", under);
}

/// GPU time of the pack at 1080p on the real SkyWars scene, from timestamp queries. Prints the
/// frame's mean GPU time with the pack off and on (same pose, no ticks, so only the pack
/// differs) and the pack's own phases. Timings are provisional.
#[test]
#[ignore = "GPU timing run; needs MC_GPU_TIMING=1 (see the module docs)"]
fn skywars_gpu_timings() {
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
        pitch: 8.0,
    };
    let frames = 240;
    let measure = |rig: &mut Rig, label: &str| {
        // Warm up (pipelines, the bake, scene targets), then measure from a fresh baseline.
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
            if name.contains("underwater") || name.starts_with("gpu.") || ms > 0.02 {
                eprintln!("[{label}]   {name:<40} {ms:.4} ms ({calls} frames)");
            }
        }
        frame_ms
    };
    rig.settings(|s| {
        time(s, 12.0);
        s.set_enabled(ID, false);
    });
    rig.tick();
    let off = measure(&mut rig, "off");
    rig.settings(|s| s.set_enabled(ID, true));
    wait_for_caustics();
    let on = measure(&mut rig, "on (defaults)");
    rig.settings(|s| s.set_choice(ID, "quality", 2));
    let high = measure(&mut rig, "on (High quality)");
    rig.settings(|s| {
        s.set_choice(ID, "quality", 1);
        s.set_enabled(ID, false);
    });
    let off_again = measure(&mut rig, "off again");
    eprintln!(
        "pack cost (provisional): defaults {:+.4} ms, High {:+.4} ms; off vs off again {:+.4} ms",
        on - off,
        high - off,
        off_again - off
    );
}
