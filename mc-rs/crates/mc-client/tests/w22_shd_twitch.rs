//! How steady the shader packs' lighting is from one frame to the next.
//!
//! Shading that "twitches" is light or shadow that jumps between frames while nothing in the
//! scene moves. This harness holds a still scene (the SkyWars benchmark capture after its settle:
//! every entity removed, no packets read any more, the camera fixed, the HUD hidden, clouds off)
//! and drives the world clock by hand, three frames per tick at partial ticks 0, 1/3 and 2/3 (a
//! 60 fps client). For every frame it counts the pixels whose brightness jumped since the frame
//! before. Timelines:
//!
//! * `running`: the Time Changer off and the daylight cycle running (the sun crawls about 0.015
//!   degrees a tick). A steady renderer changes the picture by a fraction of a level per frame;
//!   a twitch is a frame where many pixels jump at once.
//! * `tcHHMM`: the Time Changer on at a clock time (the clock is held, only the partial tick
//!   moves). A steady renderer draws the same picture every frame.
//! * `srvHHMM`: the Time Changer off, the server holding the clock (it stopped the daylight
//!   cycle, as the SkyWars capture's server does). Also the same picture every frame.
//!
//! Each timeline runs once with no pack (the control: what vanilla itself animates, such as
//! water textures, is masked out of the pack's count) and once per pack.
//!
//! ```text
//! MC_NO_SKIN_DOWNLOADS=1 MC_W22_SHD_PNG_DIR=<dir> MC_JARS_DIR=<jars> \
//!     cargo test -p mc-client --test w22_shd_twitch -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Optional: `MC_W22_SHD_PACKS=classic,dreamy,underwater` (which packs), `MC_W22_SHD_TIMES=
//! 3.5,12` (Time Changer clock times, hours), `MC_W22_SHD_SERVER=9.25` (server-held clock times),
//! `MC_W22_SHD_TICKS=80` (ticks of the running timeline), `MC_W22_SHD_YAW` / `_PITCH` (the
//! view), `MC_W22_SHD_TAG=before` (a prefix for the output files). `MC_W22_SHD_STABLE=0` runs
//! the packs without the shading stability fixes (the "before" numbers).

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use mc_client::app::campath::CameraScript;
use mc_client::app::config::LoginMode;
use mc_client::{Client, ClientConfig};
use mc_math::DVec3;
use mc_physics::state::MoveInput;
use mc_render::{Camera, Gpu, GpuOptions, PhaseList};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
/// Ticks of the capture to play before the scene is held: the join, the chunks and the players
/// have settled by 280.
const SETTLE_TICKS: u32 = 285;
/// The SkyWars benchmark capture, relative to this crate.
const CAPTURE: &str =
    "../../optimization/scratch/skywars-scene-2026-11-05/captures/sw-mega-doubles-mid.mccap";
/// Frames per tick (60 fps).
const FRAMES_PER_TICK: u32 = 3;
/// A pixel jumps when its brightness moves by at least this many levels (of 255) between two
/// consecutive frames.
const JUMP: f32 = 3.0;
/// A big jump.
const BIG_JUMP: f32 = 10.0;
/// The world time the running timeline starts at: 09:00.
const RUNNING_START: i64 = 3_000;
/// Ticks run before each timeline is measured.
const WARM_UP_TICKS: u32 = 60;

struct Rig {
    client: Client,
    gpu: Gpu,
    phases: PhaseList,
    camera: Camera,
    out: PathBuf,
    tag: String,
    _dir: tempfile::TempDir,
}

/// Where the held camera looks.
#[derive(Debug, Clone, Copy)]
struct View {
    yaw: f32,
    pitch: f32,
}

/// One frame-to-frame comparison.
#[derive(Debug, Clone, Copy, Default)]
struct Step {
    /// Pixels that jumped by `JUMP` or more.
    jumps: u32,
    /// Pixels that jumped by `BIG_JUMP` or more.
    big: u32,
    /// Mean absolute brightness change over the compared pixels, levels.
    mean: f64,
    /// The largest change, levels.
    max: f32,
}

/// What one timeline did.
struct Series {
    name: String,
    steps: Vec<Step>,
    /// Per pixel: how many steps it jumped in.
    heat: Vec<u16>,
    /// The worst step: (index, frame before, frame after).
    worst: Option<(usize, Vec<u8>, Vec<u8>)>,
    /// Shadow-map renders during the timeline.
    shadow_renders: u64,
    /// The world time the render environment held at the end.
    end_time: i64,
}

impl Series {
    fn frames_with_jumps(&self, at_least: u32) -> usize {
        self.steps.iter().filter(|s| s.jumps >= at_least).count()
    }

    fn total_jumps(&self) -> u64 {
        self.steps.iter().map(|s| u64::from(s.jumps)).sum()
    }

    fn max_jumps(&self) -> u32 {
        self.steps.iter().map(|s| s.jumps).max().unwrap_or(0)
    }

    fn max_big(&self) -> u32 {
        self.steps.iter().map(|s| s.big).max().unwrap_or(0)
    }

    fn summary(&self) -> String {
        let n = self.steps.len().max(1);
        let mean: f64 = self.steps.iter().map(|s| s.mean).sum::<f64>() / n as f64;
        format!(
            "{:<34} steps {:>4}  frames with >=50 jumping px {:>4}  >=500 {:>4}  max jumping px \
             {:>7} (>= {BIG_JUMP} levels: {:>6})  total {:>9}  mean |d| {:.4}  shadow renders \
             {:>4}  end time {}",
            self.name,
            self.steps.len(),
            self.frames_with_jumps(50),
            self.frames_with_jumps(500),
            self.max_jumps(),
            self.max_big(),
            self.total_jumps(),
            mean,
            self.shadow_renders,
            self.end_time,
        )
    }
}

fn luma(p: &[u8]) -> f32 {
    0.2126 * f32::from(p[0]) + 0.7152 * f32::from(p[1]) + 0.0722 * f32::from(p[2])
}

/// Compare two frames over the pixels `mask` keeps (`None`: all).
fn compare(a: &[u8], b: &[u8], mask: Option<&[bool]>, heat: &mut [u16]) -> Step {
    let mut step = Step::default();
    let mut sum = 0.0_f64;
    let mut count = 0_u64;
    for (i, (p, q)) in a.chunks_exact(4).zip(b.chunks_exact(4)).enumerate() {
        if mask.is_some_and(|m| !m[i]) {
            continue;
        }
        let d = (luma(p) - luma(q)).abs();
        sum += f64::from(d);
        count += 1;
        if d >= JUMP {
            step.jumps += 1;
            heat[i] = heat[i].saturating_add(1);
        }
        if d >= BIG_JUMP {
            step.big += 1;
        }
        step.max = step.max.max(d);
    }
    step.mean = sum / count.max(1) as f64;
    step
}

impl Rig {
    fn new() -> Option<Rig> {
        let out = PathBuf::from(std::env::var_os("MC_W22_SHD_PNG_DIR")?);
        // The capture's players would otherwise have their skins fetched.
        if std::env::var_os("MC_NO_SKIN_DOWNLOADS").is_none_or(|v| v != "1") {
            eprintln!("SKIP: set MC_NO_SKIN_DOWNLOADS=1 (no skin downloads from a test)");
            return None;
        }
        std::fs::create_dir_all(&out).ok()?;
        let capture = Path::new(env!("CARGO_MANIFEST_DIR")).join(CAPTURE);
        if !capture.is_file() {
            eprintln!("SKIP: no SkyWars capture at {}", capture.display());
            return None;
        }
        let dir = tempfile::tempdir().ok()?;
        // Clouds drift with the tick counter: off, so the sky holds still.
        std::fs::write(dir.path().join("options.txt"), "renderClouds:false\n").ok()?;
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
        let tag = std::env::var("MC_W22_SHD_TAG").unwrap_or_else(|_| "run".to_owned());
        Some(Rig {
            client,
            gpu,
            phases,
            camera,
            out,
            tag,
            _dir: dir,
        })
    }

    /// One frame at `partial_ticks`, read back.
    fn render(&mut self, partial_ticks: f32) -> Vec<u8> {
        self.client
            .frame(&self.gpu, &mut self.camera, partial_ticks);
        let frame = self.gpu.acquire().expect("acquire");
        self.phases
            .render(&self.gpu, frame.view(), &self.camera, partial_ticks);
        let pixels = frame.read_pixels(&self.gpu).expect("read back");
        frame.present(&self.gpu.queue);
        pixels
    }

    /// One frame without a readback.
    fn draw(&mut self, partial_ticks: f32) {
        self.client
            .frame(&self.gpu, &mut self.camera, partial_ticks);
        let frame = self.gpu.acquire().expect("acquire");
        self.phases
            .render(&self.gpu, frame.view(), &self.camera, partial_ticks);
        frame.present(&self.gpu.queue);
    }

    /// One poll, one tick and one frame of the capture.
    fn replay_tick(&mut self) {
        self.client.poll();
        self.client.tick(&MoveInput::default());
        self.draw(1.0);
    }

    /// One tick with no packets: the clock and the client's own animations move, nothing else.
    fn tick(&mut self) {
        self.client.tick(&MoveInput::default());
    }

    /// Remove every tracked entity (players, items, arrows): only the terrain and the sky stay.
    fn clear_entities(&mut self) {
        let session = self.client.session_mut().expect("a session");
        let state = session.state_mut();
        let ids: Vec<i32> = state.entities.iter().map(|(id, _)| id).collect();
        for id in ids {
            state.entities.despawn(&mut state.world, id);
        }
    }

    fn hold(&mut self, view: View) {
        let session = self.client.session_mut().expect("a session");
        let predictor = session.predictor_mut().expect("a player");
        predictor.state.yaw = view.yaw;
        predictor.state.pitch = view.pitch;
        predictor.state.vel = DVec3::ZERO;
    }

    fn settings<R>(&mut self, edit: impl FnOnce(&mut mc_mods::ModSettings) -> R) {
        let _ = edit(self.client.mod_settings_mut());
    }

    /// Set the session's clock (as a time packet would): `world_time` negative freezes the cycle.
    fn set_clock(&mut self, world_time: i64) {
        let session = self.client.session_mut().expect("a session");
        let env = &mut session.state_mut().env;
        let total = env.total_world_time;
        env.apply_time(total, world_time);
    }

    /// The world time the render environment holds (the Time Changer's, if it is on).
    fn render_time(&self) -> i64 {
        self.client
            .render()
            .map_or(0, |r| r.fog.with(|phase| phase.environment.world_time))
    }

    fn shadow_renders(&self) -> u64 {
        self.client
            .render()
            .map_or(0, |r| r.terrain.borrow().shadow_pass_stats().1)
    }

    /// Let a mesher-mode change land (re-meshing runs over a few hundred frames).
    fn settle_meshes(&mut self) {
        for _ in 0..300 {
            self.draw(0.0);
        }
    }

    /// Before a timeline: the clock set so it reaches `RUNNING_START` when the timeline starts
    /// (when it runs), or held by the server at `server_held` (a negative time, as a server that
    /// stops the daylight cycle sends it), the meshes settled, then `WARM_UP_TICKS` ticks so
    /// vanilla's per-tick easing (fog brightness, lightmap) has settled on the timeline's time.
    fn warm_up(&mut self, view: View, server_held: Option<i64>) {
        self.set_clock(
            server_held.map_or(RUNNING_START - i64::from(WARM_UP_TICKS) - 1, |time| -time),
        );
        self.tick();
        self.settle_meshes();
        for _ in 0..WARM_UP_TICKS {
            self.tick();
            self.hold(view);
            self.draw(0.0);
        }
    }

    fn save(&self, name: &str, pixels: &[u8], width: u32, height: u32) {
        let path = self.out.join(format!("{}_{name}.png", self.tag));
        image::save_buffer(&path, pixels, width, height, image::ColorType::Rgba8)
            .expect("write png");
        eprintln!("wrote {}", path.display());
    }

    /// Run a timeline: `ticks` ticks of `FRAMES_PER_TICK` frames. `mask` (when given) keeps only
    /// the pixels vanilla itself held still on the same timeline.
    fn run(&mut self, name: &str, ticks: u32, view: View, mask: Option<&[bool]>) -> Series {
        let pixels = (WIDTH * HEIGHT) as usize;
        let mut series = Series {
            name: name.to_owned(),
            steps: Vec::new(),
            heat: vec![0; pixels],
            worst: None,
            shadow_renders: 0,
            end_time: 0,
        };
        let renders_before = self.shadow_renders();
        let mut previous: Option<Vec<u8>> = None;
        let mut worst_jumps = 0;
        for _ in 0..ticks {
            self.tick();
            for f in 0..FRAMES_PER_TICK {
                self.hold(view);
                let pt = f as f32 / FRAMES_PER_TICK as f32;
                let frame = self.render(pt);
                if let Some(before) = previous.as_ref() {
                    let step = compare(before, &frame, mask, &mut series.heat);
                    if step.jumps > worst_jumps {
                        worst_jumps = step.jumps;
                        series.worst = Some((series.steps.len(), before.clone(), frame.clone()));
                    }
                    series.steps.push(step);
                }
                previous = Some(frame);
            }
        }
        series.shadow_renders = self.shadow_renders() - renders_before;
        series.end_time = self.render_time();
        if let Some(last) = previous {
            self.save(&format!("{name}_last"), &last, WIDTH, HEIGHT);
        }
        series
    }

    /// The time series as text, the heat map and the worst step's strip.
    fn report(&self, series: &Series) {
        let mut csv = String::from("step,jumps,big,mean,max\n");
        for (i, s) in series.steps.iter().enumerate() {
            let _ = writeln!(csv, "{i},{},{},{:.5},{:.2}", s.jumps, s.big, s.mean, s.max);
        }
        let path = self.out.join(format!("{}_{}.csv", self.tag, series.name));
        std::fs::write(&path, csv).expect("write csv");
        // Heat map: black never jumped, red to yellow jumped in more and more steps.
        let most = series.heat.iter().copied().max().unwrap_or(0).max(1);
        let mut heat = Vec::with_capacity(series.heat.len() * 4);
        for &h in &series.heat {
            let t = f32::from(h) / f32::from(most);
            let (r, g) = if h == 0 {
                (0.0, 0.0)
            } else {
                (255.0 * (0.4 + 0.6 * t.sqrt()), 255.0 * t)
            };
            heat.extend_from_slice(&[r as u8, g as u8, 0, 255]);
        }
        self.save(&format!("{}_heat", series.name), &heat, WIDTH, HEIGHT);
        if let Some((index, before, after)) = &series.worst {
            self.save_strip(&format!("{}_worst{index}", series.name), before, after);
        }
    }

    /// Before, after and the difference (x8) of the 320x180 box with the most change, at 2x.
    fn save_strip(&self, name: &str, before: &[u8], after: &[u8]) {
        let (w, h) = (WIDTH as usize, HEIGHT as usize);
        let (bw, bh) = (320_usize, 180_usize);
        // The box with the most absolute change, on a 40-pixel grid.
        let diff: Vec<f32> = before
            .chunks_exact(4)
            .zip(after.chunks_exact(4))
            .map(|(p, q)| (luma(p) - luma(q)).abs())
            .collect();
        let mut best = (0, 0, -1.0_f32);
        for y0 in (0..=h - bh).step_by(40) {
            for x0 in (0..=w - bw).step_by(40) {
                let mut sum = 0.0;
                for y in y0..y0 + bh {
                    for x in x0..x0 + bw {
                        sum += diff[y * w + x];
                    }
                }
                if sum > best.2 {
                    best = (x0, y0, sum);
                }
            }
        }
        let (x0, y0, _) = best;
        let scale = 2;
        let (sw, sh) = (bw * 3 * scale, bh * scale);
        let mut strip = vec![0_u8; sw * sh * 4];
        for tile in 0..3 {
            for y in 0..bh {
                for x in 0..bw {
                    let i = ((y0 + y) * w + x0 + x) * 4;
                    let px: [u8; 4] = match tile {
                        0 => [before[i], before[i + 1], before[i + 2], 255],
                        1 => [after[i], after[i + 1], after[i + 2], 255],
                        _ => {
                            let d = |c: usize| {
                                (f32::from(after[i + c]) - f32::from(before[i + c])).abs() * 8.0
                            };
                            [
                                d(0).min(255.0) as u8,
                                d(1).min(255.0) as u8,
                                d(2).min(255.0) as u8,
                                255,
                            ]
                        }
                    };
                    for dy in 0..scale {
                        for dx in 0..scale {
                            let sx = tile * bw * scale + x * scale + dx;
                            let sy = y * scale + dy;
                            let o = (sy * sw + sx) * 4;
                            strip[o..o + 4].copy_from_slice(&px);
                        }
                    }
                }
            }
        }
        self.save(
            &format!("{name}_strip_x{x0}_y{y0}"),
            &strip,
            sw as u32,
            sh as u32,
        );
    }
}

fn time_changer(s: &mut mc_mods::ModSettings, hours: Option<f32>) {
    match hours {
        Some(hours) => {
            s.set_enabled("time_changer", true);
            s.set_f32("time_changer", "time", hours);
        }
        None => {
            s.set_enabled("time_changer", false);
        }
    }
}

/// Switch to one pack (or none) with the animations that move by design held still: waving
/// plants and leaves off, flat water, no film grain or fog wind, still caustics, no marine snow.
fn pack(s: &mut mc_mods::ModSettings, which: Option<&str>) {
    for id in ["shaders_classic", "shaders_dreamy", "shaders_underwater"] {
        s.set_enabled(id, false);
    }
    s.set_bool("shaders_classic", "waving_plants", false);
    s.set_bool("shaders_classic", "waving_leaves", false);
    s.set_f32("shaders_classic", "water_waves", 0.0);
    s.set_f32("shaders_dreamy", "grain", 0.0);
    s.set_f32("shaders_dreamy", "fog_wind", 0.0);
    s.set_f32("shaders_underwater", "wave_speed", 0.0);
    s.set_bool("shaders_underwater", "marine_snow", false);
    if let Some(id) = which {
        s.set_enabled(id, true);
    }
}

fn pack_id(name: &str) -> Option<&'static str> {
    match name {
        "classic" => Some("shaders_classic"),
        "classic4096" => Some("shaders_classic"),
        "dreamy" => Some("shaders_dreamy"),
        "underwater" => Some("shaders_underwater"),
        _ => None,
    }
}

#[test]
#[ignore = "renders the real SkyWars scene for minutes; run by hand (see the module docs)"]
fn shading_holds_still() {
    let Some(mut rig) = Rig::new() else {
        return;
    };
    for _ in 0..SETTLE_TICKS {
        rig.replay_tick();
    }
    rig.client.set_hide_gui_for_capture(true);
    rig.clear_entities();
    let feet = rig
        .client
        .session()
        .and_then(|s| s.predictor())
        .map(|p| (p.state.pos, p.state.on_ground))
        .expect("a player");
    eprintln!("viewer feet {:?} on ground {}", feet.0, feet.1);
    if let Some(session) = rig.client.session() {
        let env = &session.state().env;
        eprintln!(
            "the capture's server clock: world time {} (daylight cycle {})",
            env.world_time,
            if env.daylight_cycle_frozen() {
                "stopped"
            } else {
                "running"
            }
        );
    }
    let packs: Vec<String> = std::env::var("MC_W22_SHD_PACKS")
        .unwrap_or_else(|_| "classic,classic4096,dreamy,underwater".to_owned())
        .split(',')
        .map(str::to_owned)
        .collect();
    let times: Vec<f32> = std::env::var("MC_W22_SHD_TIMES")
        .unwrap_or_else(|_| "3.5,6,9.25,12,15".to_owned())
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    let running_ticks: u32 = std::env::var("MC_W22_SHD_TICKS")
        .ok()
        .and_then(|t| t.parse().ok())
        .unwrap_or(80);
    // West over the plaza, a little down: castle walls, their shadows and open ground.
    let view = View {
        yaw: std::env::var("MC_W22_SHD_YAW")
            .ok()
            .and_then(|t| t.parse().ok())
            .unwrap_or(90.0),
        pitch: std::env::var("MC_W22_SHD_PITCH")
            .ok()
            .and_then(|t| t.parse().ok())
            .unwrap_or(20.0),
    };
    // Let the last particles die and the held pose settle.
    rig.settings(|s| time_changer(s, Some(12.0)));
    for _ in 0..60 {
        rig.tick();
        rig.hold(view);
        rig.draw(0.0);
    }

    // Clock times a server holds (it stopped the daylight cycle), Time Changer off.
    let server_times: Vec<f32> = std::env::var("MC_W22_SHD_SERVER")
        .unwrap_or_else(|_| "9.25".to_owned())
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    let label = |prefix: &str, hours: f32| {
        let minutes = (hours * 60.0).round() as i32;
        format!("{prefix}{:02}{:02}", minutes / 60, minutes % 60)
    };
    // Timelines: (name, Time Changer hours, the server's held world time, ticks); neither = the
    // running clock.
    let mut timelines: Vec<(String, Option<f32>, Option<i64>, u32)> =
        vec![("running".to_owned(), None, None, running_ticks)];
    for t in &times {
        timelines.push((label("tc", *t), Some(*t), None, 30));
    }
    for t in &server_times {
        let held = mc_mods::mods::time_changer::time_of_day_ticks(*t);
        timelines.push((label("srv", *t), None, Some(held), 30));
    }

    let mut lines = Vec::new();
    for (timeline, hours, server_held, ticks) in &timelines {
        // The control: no pack, same timeline. What it animates is masked out below.
        rig.settings(|s| {
            pack(s, None);
            time_changer(s, *hours);
        });
        rig.warm_up(view, *server_held);
        let control = rig.run(&format!("{timeline}_vanilla"), *ticks, view, None);
        let mask: Vec<bool> = control.heat.iter().map(|&h| h == 0).collect();
        let masked = mask.iter().filter(|m| !**m).count();
        eprintln!("{}", control.summary());
        eprintln!("  vanilla moves {masked} pixels by itself: masked out of the packs' counts");
        lines.push(control.summary());
        rig.report(&control);
        for name in &packs {
            let Some(id) = pack_id(name) else {
                eprintln!("unknown pack {name}");
                continue;
            };
            rig.settings(|s| {
                pack(s, Some(id));
                time_changer(s, *hours);
                let quality = if name == "classic4096" { 2 } else { 1 };
                s.set_choice("shaders_classic", "shadow_quality", quality);
            });
            rig.warm_up(view, *server_held);
            let series = rig.run(&format!("{timeline}_{name}"), *ticks, view, Some(&mask));
            eprintln!("{}", series.summary());
            lines.push(series.summary());
            rig.report(&series);
        }
    }
    let summary = lines.join("\n");
    std::fs::write(
        rig.out.join(format!("{}_summary.txt", rig.tag)),
        format!("{summary}\n"),
    )
    .expect("write summary");
    eprintln!("---- summary ----\n{summary}");
}
