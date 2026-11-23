//! The Dreamy pack's picture and timing harness.
//!
//! Drives the real client over the SkyWars benchmark capture (`sw-mega-doubles-mid`, Dwarf
//! Fortress, 40 players) on a headless GPU, then poses the camera and writes PNGs with the pack off
//! and on: noon, sunset and night; a vista over the void, the sunset horizon, the island's edge
//! (the fog sea), a high view, the east side and the sky; a player at 5 and at 30 blocks; every
//! preset; ground fog on and off. An integration test, so it only needs the library to build.
//!
//! ```text
//! MC_TW_SB_PNG_DIR=<dir> MC_JARS_DIR=<jars> cargo test -p mc-client --test tw_dreamy_pictures \
//!     -- --ignored --nocapture --test-threads=1 skywars_pictures
//! MC_GPU_TIMING=1 MC_JARS_DIR=<jars> cargo test -p mc-client --test tw_dreamy_pictures \
//!     -- --ignored --nocapture skywars_gpu_timings
//! ```
//!
//! It also checks, on the real frame, that switching the pack off after it ran renders
//! byte-identically to never having switched it on, with nothing left allocated.

use std::path::{Path, PathBuf};

use mc_client::app::campath::CameraScript;
use mc_client::app::config::LoginMode;
use mc_client::sync::TrackedKind;
use mc_client::{Client, ClientConfig};
use mc_math::{BlockPos, DVec3};
use mc_mods::mods::shaders_dreamy::{self as desc, ids, PRESETS};
use mc_physics::state::MoveInput;
use mc_render::{Camera, Gpu, GpuOptions, PhaseList};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
/// Ticks to run before posing: the join, the chunks and the players have settled by 280.
const SETTLE_TICKS: u32 = 285;
/// The SkyWars benchmark capture, relative to this crate.
const CAPTURE: &str =
    "../../optimization/scratch/skywars-scene-2026-11-05/captures/sw-mega-doubles-mid.mccap";
const ID: &str = desc::ID;

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

struct Rig {
    client: Client,
    gpu: Gpu,
    phases: PhaseList,
    camera: Camera,
    out: Option<PathBuf>,
    width: u32,
    height: u32,
    _dir: tempfile::TempDir,
}

impl Rig {
    fn new(width: u32, height: u32, out: Option<PathBuf>) -> Option<Rig> {
        if let Some(out) = &out {
            std::fs::create_dir_all(out).ok()?;
        }
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

    /// One frame without a tick and without a poll: the scene holds still.
    fn render(&mut self) -> Vec<u8> {
        self.client.frame(&self.gpu, &mut self.camera, 1.0);
        let frame = self.gpu.acquire().expect("acquire");
        self.phases
            .render(&self.gpu, frame.view(), &self.camera, 1.0);
        let pixels = frame.read_pixels(&self.gpu).expect("read back");
        frame.present(&self.gpu.queue);
        pixels
    }

    /// A frame that is not read back.
    fn draw(&mut self) {
        self.client.frame(&self.gpu, &mut self.camera, 1.0);
        let frame = self.gpu.acquire().expect("acquire");
        self.phases
            .render(&self.gpu, frame.view(), &self.camera, 1.0);
        frame.present(&self.gpu.queue);
    }

    /// One poll, one tick and one frame: re-meshing and the sky's time of day happen in the tick.
    fn tick(&mut self) {
        self.client.poll();
        self.client.tick(&MoveInput::default());
        self.draw();
    }

    fn pose(&mut self, pose: Pose) {
        let session = self.client.session_mut().expect("a session");
        let predictor = session.predictor_mut().expect("a player");
        predictor.state.pos = pose.feet;
        predictor.state.prev_pos = pose.feet;
        predictor.state.yaw = pose.yaw;
        predictor.state.pitch = pose.pitch;
    }

    fn settings<R>(&mut self, edit: impl FnOnce(&mut mc_mods::ModSettings) -> R) {
        let _ = edit(self.client.mod_settings_mut());
    }

    /// Apply a new time of day: it takes effect in a tick (the replay moves on by those ticks).
    fn retime(&mut self, hours: f32, pose: Pose) {
        self.settings(|s| time(s, hours));
        for _ in 0..3 {
            self.pose(pose);
            self.tick();
        }
    }

    /// Settle after a settings change with frames only (the scene holds still, so an off/on
    /// pair shows the same moment; the phases pick the settings up and the mesher switches mode
    /// within these frames), then keep the last frame; save it when a name is given and an
    /// output folder is set.
    fn shot(&mut self, name: Option<&str>, pose: Pose) -> Vec<u8> {
        for _ in 0..60 {
            self.pose(pose);
            self.draw();
        }
        self.pose(pose);
        let pixels = self.render();
        if let (Some(name), Some(out)) = (name, &self.out) {
            let path = out.join(format!("{name}.png"));
            image::save_buffer(
                &path,
                &pixels,
                self.width,
                self.height,
                image::ColorType::Rgba8,
            )
            .expect("write png");
            eprintln!("wrote {}", path.display());
        }
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

    /// Standing a few blocks inside the island's nearest edge, looking out over the void and
    /// down: the view that shows the fog sea. Scans 32 directions for the first column with
    /// nothing under it for 40 blocks.
    fn edge_pose(&self, me: DVec3) -> Option<Pose> {
        let mut best: Option<(f64, DVec3)> = None;
        for k in 0..32 {
            let angle = f64::from(k) / 32.0 * std::f64::consts::TAU;
            let dir = DVec3::new(angle.cos(), 0.0, angle.sin());
            for step in 4..80 {
                // The rim: open void here and for 8 more blocks out (not a hole in a floor).
                let empty = (0..=8).step_by(4).all(|more| {
                    let p = me + dir * f64::from(step + more);
                    (0..40).all(|down| !self.solid(p - DVec3::new(0.0, f64::from(down), 0.0)))
                });
                if empty {
                    if best.is_none_or(|(d, _)| f64::from(step) < d) {
                        best = Some((f64::from(step), dir));
                    }
                    break;
                }
            }
        }
        let (distance, dir) = best?;
        let feet = me + dir * (distance - 4.0).max(0.0) + DVec3::new(0.0, 3.0, 0.0);
        Some(Pose {
            feet,
            yaw: (-dir.x).atan2(dir.z).to_degrees() as f32,
            pitch: 30.0,
        })
    }

    /// A pose `distance` blocks from `target` (horizontally, same height) with nothing solid in
    /// between, looking at its chest: on ground if one of 32 directions allows it, else floating
    /// (the islands end within 30 blocks in most directions).
    fn pose_near(&self, target: DVec3, distance: f64) -> Option<Pose> {
        let chest = target + DVec3::new(0.0, 1.2, 0.0);
        for need_ground in [true, false] {
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
                let grounded = self.solid(feet - DVec3::new(0.0, 0.5, 0.0));
                let clear = !self.solid(eye) && !self.solid(feet + DVec3::new(0.0, 0.5, 0.0));
                if !blocked && clear && (grounded || !need_ground) {
                    return Some(Pose::looking_at(feet, chest));
                }
            }
        }
        None
    }
}

fn time(s: &mut mc_mods::ModSettings, hours: f32) {
    s.set_enabled("time_changer", true);
    s.set_f32("time_changer", "time", hours);
}

/// The pack at its defaults (the Dreamy preset), on or off. The fog's top is pinned with World
/// height mode at exactly the default distance below the viewer's real feet: what Below you mode
/// settles to while standing on the island, without its easing following the posed camera (the
/// poses float the camera above the island).
fn pack(s: &mut mc_mods::ModSettings, on: bool, feet_y: f64) {
    s.reset_options(ID);
    s.set_enabled(ID, on);
    let depth = f64::from(s.f32(ID, ids::FOG_DEPTH));
    s.set_choice(ID, ids::FOG_HEIGHT_MODE, desc::FOG_WORLD_HEIGHT);
    s.set_f32(ID, ids::FOG_LEVEL, (feet_y - depth).round() as f32);
}

/// Frames side by side as one RGBA image.
fn side_by_side(frames: &[&[u8]], width: u32, height: u32) -> Vec<u8> {
    let row = (width * 4) as usize;
    let mut out = Vec::with_capacity(frames.len() * frames[0].len());
    for y in 0..height as usize {
        for frame in frames {
            out.extend_from_slice(&frame[y * row..(y + 1) * row]);
        }
    }
    out
}

fn save_row(rig: &Rig, name: &str, frames: &[&[u8]]) {
    let Some(out) = &rig.out else {
        return;
    };
    let pixels = side_by_side(frames, rig.width, rig.height);
    let path = out.join(format!("{name}.png"));
    image::save_buffer(
        &path,
        &pixels,
        rig.width * frames.len() as u32,
        rig.height,
        image::ColorType::Rgba8,
    )
    .expect("write png");
    eprintln!("wrote {}", path.display());
}

#[test]
#[ignore = "writes PNGs of the real SkyWars scene; run by hand (see the module docs)"]
fn skywars_pictures() {
    let Some(out) = std::env::var_os("MC_TW_SB_PNG_DIR").map(PathBuf::from) else {
        eprintln!("SKIP: set MC_TW_SB_PNG_DIR");
        return;
    };
    let Some(mut rig) = Rig::new(WIDTH, HEIGHT, Some(out)) else {
        return;
    };
    for _ in 0..SETTLE_TICKS {
        rig.tick();
    }
    let me = rig.viewer();
    eprintln!("viewer at {me:?}");

    // Views. The far island and the sunset are west (yaw 90); the edge view is computed.
    let vista = Pose {
        feet: me + DVec3::new(0.0, 6.0, 0.0),
        yaw: 90.0,
        pitch: 14.0,
    };
    let sunset_view = Pose {
        feet: me + DVec3::new(0.0, 2.0, 0.0),
        yaw: 90.0,
        pitch: -4.0,
    };
    let edge = rig.edge_pose(me).unwrap_or(Pose {
        feet: me + DVec3::new(0.0, 4.0, 0.0),
        yaw: -90.0,
        pitch: 38.0,
    });
    eprintln!("edge pose {edge:?}");
    let high = Pose {
        feet: me + DVec3::new(12.0, 22.0, 0.0),
        yaw: 90.0,
        pitch: 32.0,
    };
    let east = Pose {
        feet: me + DVec3::new(0.0, 3.0, 0.0),
        yaw: -90.0,
        pitch: 10.0,
    };
    let up = Pose {
        feet: me,
        yaw: -90.0,
        pitch: -50.0,
    };

    // Off vs on (defaults) at noon, sunset and night, for each view.
    for (label, hours) in [("noon", 12.0), ("sunset", 18.5), ("night", 22.0)] {
        for (view, pose) in [
            ("vista", vista),
            ("west", sunset_view),
            ("edge", edge),
            ("high", high),
            ("east", east),
            ("up", up),
        ] {
            rig.settings(|s| pack(s, false, me.y));
            rig.retime(hours, pose);
            let off = rig.shot(None, pose);
            rig.settings(|s| pack(s, true, me.y));
            let on = rig.shot(Some(&format!("{label}_{view}_on")), pose);
            save_row(&rig, &format!("{label}_{view}_off_vs_on"), &[&off, &on]);
        }
    }

    // Players at 5 and 30 blocks (pack off vs on), at noon and sunset.
    let nearest = rig.players().into_iter().min_by(|a, b| {
        (*a - me)
            .length()
            .partial_cmp(&(*b - me).length())
            .expect("finite")
    });
    for (label, hours) in [("noon", 12.0), ("sunset", 18.5)] {
        let Some(target) = nearest else {
            eprintln!("no player to photograph");
            break;
        };
        for distance in [5.0, 30.0] {
            let Some(pose) = rig.pose_near(target, distance) else {
                eprintln!("no clear pose {distance} blocks from the player");
                continue;
            };
            rig.settings(|s| pack(s, false, me.y));
            rig.retime(hours, pose);
            let off = rig.shot(None, pose);
            rig.settings(|s| pack(s, true, me.y));
            let on = rig.shot(None, pose);
            save_row(
                &rig,
                &format!("{label}_player_{distance:.0}_off_vs_on"),
                &[&off, &on],
            );
        }
    }

    // Every preset, at sunset (west) and noon (vista), as one strip each (left to right in menu
    // order) plus single frames.
    for (label, hours, pose) in [("sunset", 18.5, sunset_view), ("noon", 12.0, vista)] {
        let mut strip = Vec::new();
        rig.retime(hours, pose);
        for (index, name) in PRESETS.iter().enumerate().take(PRESETS.len() - 1) {
            rig.settings(|s| {
                pack(s, true, me.y);
                s.set_choice(ID, ids::PRESET, index);
            });
            let slug = name.to_lowercase().replace(' ', "_");
            strip.push(rig.shot(Some(&format!("preset_{label}_{index}_{slug}")), pose));
        }
        let first: Vec<&[u8]> = strip.iter().take(4).map(Vec::as_slice).collect();
        let second: Vec<&[u8]> = strip.iter().skip(4).map(Vec::as_slice).collect();
        save_row(&rig, &format!("presets_{label}_1"), &first);
        save_row(&rig, &format!("presets_{label}_2"), &second);
    }

    // Ground fog on vs off (pack on), sunset high view and noon edge view.
    for (label, hours, pose) in [("sunset_high", 18.5, high), ("noon_edge", 12.0, edge)] {
        rig.settings(|s| {
            pack(s, true, me.y);
            s.set_bool(ID, ids::GROUND_FOG, false);
        });
        rig.retime(hours, pose);
        let off = rig.shot(None, pose);
        rig.settings(|s| s.set_bool(ID, ids::GROUND_FOG, true));
        let on = rig.shot(None, pose);
        save_row(&rig, &format!("fog_{label}_off_vs_on"), &[&off, &on]);
    }

    // Off after on is exactly never-on (nothing left behind). Grain and the fog's drift do not
    // matter here: with the pack off neither runs.
    rig.settings(|s| pack(s, false, me.y));
    rig.retime(12.0, vista);
    let never = rig.shot(None, vista);
    rig.settings(|s| pack(s, true, me.y));
    let _ = rig.shot(None, vista);
    rig.settings(|s| pack(s, false, me.y));
    let after = rig.shot(None, vista);
    let differing = never
        .chunks_exact(4)
        .zip(after.chunks_exact(4))
        .filter(|(a, b)| a != b)
        .count();
    eprintln!("off-after-on vs never-on: {differing} pixels differ");
    assert_eq!(rig.client.dreamy_pack_allocated(), (false, false));
}

/// Which effect lights the night sky: the night "up" view with the pack on, then with each effect
/// switched off in turn. Frames only after the settle (the scene holds still), saved as one strip:
/// off, all on, no rays, no fog, no haze, no sky tint, no glow.
#[test]
#[ignore = "diagnostic pictures; run by hand (see the module docs)"]
fn night_sky_diagnostic() {
    let Some(out) = std::env::var_os("MC_TW_SB_PNG_DIR").map(PathBuf::from) else {
        eprintln!("SKIP: set MC_TW_SB_PNG_DIR");
        return;
    };
    let Some(mut rig) = Rig::new(640, 360, Some(out)) else {
        return;
    };
    for _ in 0..SETTLE_TICKS {
        rig.tick();
    }
    let me = rig.viewer();
    let pose = Pose {
        feet: me,
        yaw: -90.0,
        pitch: -50.0,
    };
    rig.settings(|s| {
        time(s, 22.0);
        pack(s, false, me.y);
    });
    rig.pose(pose);
    rig.tick();
    let frames = |rig: &mut Rig| {
        for _ in 0..60 {
            rig.pose(pose);
            rig.draw();
        }
        rig.pose(pose);
        rig.render()
    };
    let off = frames(&mut rig);
    let mut strip = vec![off];
    type Edit = fn(&mut mc_mods::ModSettings);
    let variants: [(&str, Edit); 6] = [
        ("all on", |_| {}),
        ("no rays", |s| {
            s.set_bool(ID, ids::LIGHT_RAYS, false);
        }),
        ("no fog", |s| {
            s.set_bool(ID, ids::GROUND_FOG, false);
        }),
        ("no haze", |s| {
            s.set_f32(ID, ids::HAZE, 0.0);
        }),
        ("no sky tint", |s| {
            s.set_f32(ID, ids::SKY_TINT_STRENGTH, 0.0);
        }),
        ("no glow", |s| {
            s.set_f32(ID, ids::GLOW, 0.0);
            s.set_f32(ID, ids::HALATION, 0.0);
        }),
    ];
    for (label, edit) in variants {
        rig.settings(|s| {
            pack(s, true, me.y);
            edit(s);
        });
        eprintln!("rendering: {label}");
        strip.push(frames(&mut rig));
    }
    let refs: Vec<&[u8]> = strip.iter().map(Vec::as_slice).collect();
    save_row(&rig, "diag_night_up", &refs);
}

/// The top-left of the frame (left 60 %, top 70 %) as bytes, for exact comparisons: away from the
/// held sword and the hotbar's sword icon, whose enchantment glint follows the wall clock.
fn left_part(pixels: &[u8], width: u32, height: u32) -> Vec<u8> {
    let row = (width * 4) as usize;
    let keep = (width * 6 / 10 * 4) as usize;
    (0..(height * 7 / 10) as usize)
        .flat_map(|y| pixels[y * row..y * row + keep].iter().copied())
        .collect()
}

/// Switching the pack off after it ran leaves the real frame exactly as if it had never been on:
/// no tick runs after settling (the scene holds still), and once the mesher has gone back to
/// vanilla meshing (the pack's sunlight changes the mesh mode) the frame is byte-identical away
/// from the glinting sword, with nothing allocated.
#[test]
#[ignore = "drives the real SkyWars scene; run by hand (see the module docs)"]
fn off_after_on_is_never_on() {
    let Some(mut rig) = Rig::new(640, 360, None) else {
        return;
    };
    for _ in 0..SETTLE_TICKS {
        rig.tick();
    }
    let me = rig.viewer();
    // Terrain (re-meshed by the pack's sunlight) and sky, and no players: their enchanted gear
    // glints with the wall clock, so a view with players never holds still.
    let pose = Pose {
        feet: me + DVec3::new(0.0, 2.0, 0.0),
        yaw: -90.0,
        pitch: -35.0,
    };
    rig.settings(|s| {
        time(s, 12.0);
        pack(s, false, me.y);
    });
    // The time of day applies in a tick: one, then frames only from here on.
    rig.pose(pose);
    rig.tick();
    let settle = |rig: &mut Rig| {
        let mut last = Vec::new();
        for frame in 0..900 {
            rig.pose(pose);
            let now = left_part(&rig.render(), rig.width, rig.height);
            if now == last {
                return (now, frame);
            }
            if frame % 150 == 149 {
                // Where the frame still changes: count and bounding box of the moving pixels.
                let keep = (rig.width * 6 / 10) as usize;
                let (mut count, mut min, mut max) = (0, (usize::MAX, usize::MAX), (0, 0));
                for (i, (a, b)) in now.chunks_exact(4).zip(last.chunks_exact(4)).enumerate() {
                    if a != b {
                        count += 1;
                        let (x, y) = (i % keep, i / keep);
                        min = (min.0.min(x), min.1.min(y));
                        max = (max.0.max(x), max.1.max(y));
                    }
                }
                eprintln!("frame {frame}: {count} pixels moved, box {min:?}..{max:?}");
            }
            last = now;
        }
        (last, 900)
    };
    let (never, frames) = settle(&mut rig);
    eprintln!("never-on frame stable after {frames} frames");
    rig.settings(|s| pack(s, true, me.y));
    for _ in 0..120 {
        rig.pose(pose);
        rig.draw();
    }
    let on = left_part(&rig.render(), rig.width, rig.height);
    assert_ne!(on, never, "the pack was not on");
    rig.settings(|s| pack(s, false, me.y));
    let mut matched = None;
    for frame in 0..900 {
        rig.pose(pose);
        let now = left_part(&rig.render(), rig.width, rig.height);
        if now == never {
            matched = Some(frame);
            break;
        }
    }
    eprintln!("off-after-on matched never-on after {matched:?} frames");
    assert!(matched.is_some(), "the frame never went back to never-on");
    assert_eq!(rig.client.dreamy_pack_allocated(), (false, false));
}

#[test]
#[ignore = "GPU timing run; needs MC_GPU_TIMING=1 (see the docs)"]
fn skywars_gpu_timings() {
    if !mc_render::context::gpu_timing_requested() {
        eprintln!("SKIP: set MC_GPU_TIMING=1");
        return;
    }
    let Some(mut rig) = Rig::new(1920, 1080, None) else {
        return;
    };
    for _ in 0..SETTLE_TICKS {
        rig.tick();
    }
    let me = rig.viewer();
    // A heavy view for the pack: the fog sea and the low sun's rays both on screen.
    let pose = Pose {
        feet: me + DVec3::new(12.0, 22.0, 0.0),
        yaw: 90.0,
        pitch: 12.0,
    };
    let frames = 240;
    let measure = |rig: &mut Rig, label: &str| {
        for _ in 0..3 {
            rig.pose(pose);
            rig.tick();
        }
        for _ in 0..40 {
            rig.pose(pose);
            rig.draw();
        }
        let start = rig.phases.gpu_timings();
        for _ in 0..frames {
            rig.pose(pose);
            rig.draw();
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
            if name.contains("dreamy")
                || name.contains("shaderpack")
                || name.starts_with("gpu.")
                || ms > 0.02
            {
                eprintln!("[{label}]   {name:<40} {ms:.4} ms ({calls} frames)");
            }
        }
        frame_ms
    };
    rig.settings(|s| {
        time(s, 18.5);
        pack(s, false, me.y);
    });
    let off = measure(&mut rig, "off");
    rig.settings(|s| pack(s, true, me.y));
    let on = measure(&mut rig, "on (defaults)");
    rig.settings(|s| {
        s.set_bool(ID, ids::SOFT_SHADOWS, false);
        s.set_bool(ID, ids::SUNLIGHT, false);
    });
    let own = measure(&mut rig, "on, without the shared shadows and sunlight");
    rig.settings(|s| s.set_choice(ID, ids::FOG_QUALITY, 1));
    let fast = measure(&mut rig, "on, own passes, fog quality Fast");
    eprintln!(
        "SUMMARY 1080p (provisional): off {off:.4} ms, on {on:.4} ms (+{:.4}), own passes only \
         {own:.4} ms (+{:.4}), fast fog {fast:.4} ms",
        on - off,
        own - off
    );
}

/// The colour grade switched to neutral (the pack's atmosphere and glow stay as they are).
fn neutral_grade(s: &mut mc_mods::ModSettings) {
    for (option, value) in [
        (ids::EXPOSURE, 0.0),
        (ids::CONTRAST, 100.0),
        (ids::SATURATION, 100.0),
        (ids::VIBRANCE, 0.0),
        (ids::TEMPERATURE, 0.0),
        (ids::TINT, 0.0),
        (ids::HUE_SHIFT, 0.0),
        (ids::SHADOWS_STRENGTH, 0.0),
        (ids::MIDTONES_STRENGTH, 0.0),
        (ids::HIGHLIGHTS_STRENGTH, 0.0),
        (ids::SPLIT_STRENGTH, 0.0),
        (ids::FADE, 0.0),
        (ids::GRADIENT_MIX, 0.0),
    ] {
        s.set_f32(ID, option, value);
    }
}

/// Frames for judging the white balance (run by hand): the pack at its defaults without vignette
/// or grain, once with a neutral grade (what the grade's table receives:
/// `wb_<view>_input.png`), once per white-balance setting with only that in the grade
/// (`wb_<view>_t<temperature>_n<tint>.png`), and the whole default look (`wb_<view>_look.png`).
/// An outside tool can then grade the input frame with any white balance and compare.
#[test]
#[ignore = "writes PNGs of the real SkyWars scene; run by hand (see the module docs)"]
fn white_balance_frames() {
    let Some(out) = std::env::var_os("MC_TW_SB_PNG_DIR").map(PathBuf::from) else {
        eprintln!("SKIP: set MC_TW_SB_PNG_DIR");
        return;
    };
    let Some(mut rig) = Rig::new(960, 540, Some(out)) else {
        return;
    };
    for _ in 0..SETTLE_TICKS {
        rig.tick();
    }
    let me = rig.viewer();
    let vista = Pose {
        feet: me + DVec3::new(0.0, 6.0, 0.0),
        yaw: 90.0,
        pitch: 14.0,
    };
    let sunset_view = Pose {
        feet: me + DVec3::new(0.0, 2.0, 0.0),
        yaw: 90.0,
        pitch: -4.0,
    };
    let calm = |s: &mut mc_mods::ModSettings| {
        pack(s, true, me.y);
        s.set_f32(ID, ids::VIGNETTE, 0.0);
        s.set_f32(ID, ids::GRAIN, 0.0);
    };
    let settings: [(f32, f32); 9] = [
        (8.0, 3.0),
        (20.0, 8.0),
        (-24.0, -3.0),
        (50.0, 0.0),
        (-50.0, 0.0),
        (0.0, 50.0),
        (0.0, -50.0),
        (100.0, 0.0),
        (-100.0, 0.0),
    ];
    for (label, hours, pose) in [("noon", 12.0, vista), ("sunset", 18.5, sunset_view)] {
        rig.settings(calm);
        rig.retime(hours, pose);
        rig.settings(neutral_grade);
        let _ = rig.shot(Some(&format!("wb_{label}_input")), pose);
        for (temperature, tint) in settings {
            rig.settings(|s| {
                neutral_grade(s);
                s.set_f32(ID, ids::TEMPERATURE, temperature);
                s.set_f32(ID, ids::TINT, tint);
            });
            let name = format!("wb_{label}_t{temperature}_n{tint}");
            let _ = rig.shot(Some(&name), pose);
        }
        rig.settings(calm);
        let _ = rig.shot(Some(&format!("wb_{label}_look")), pose);
    }
}
