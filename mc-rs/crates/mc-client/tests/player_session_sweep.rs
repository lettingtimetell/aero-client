//! A player's session with the real client, headless: every screen, every mod on and off,
//! window sizes and GUI scales, packs and shader packs, the F keys, screenshots, replays, idle.
//!
//! Each test drives the real `Client` through `ShellApp` in the window loop's order
//! (`before_ticks`, the ticks, `frame`, `finalize_camera`, the phases, `after_render`), with key
//! and mouse edges fed the way the shell feeds them (`handle_raw_key` plus the key bindings,
//! `InputState` for the cursor and buttons). wgpu's default error handler panics on a validation
//! error, so a GPU misuse fails the test as it would crash the game. Every warning and error the
//! client logs is collected with the step that was running and printed in a summary, and frames
//! slower than `SWEEP_HITCH_MS` (250 by default) are listed with their step.
//!
//! Ignored (GPU, vanilla jar). Run one at a time:
//!
//! ```text
//! MC_JARS_DIR=<jars> cargo test -p mc-client --test player_session_sweep -- --ignored \
//!     --nocapture --test-threads=1 <name>
//! ```
//!
//! Environment: `SWEEP_PNG_DIR` (a picture of every step), `SWEEP_GAME_DIR` (the game folder; a
//! temporary one otherwise), `SWEEP_REPLAYS` (a folder of `.aeroreplay` files for
//! `replays_sweep`), `SWEEP_CAPTURE` (a `.mccap` session; the benchmark capture by default),
//! `SWEEP_IDLE_SECS` (the idle tests, 120 by default), `SWEEP_SEED` (the monkey's seed),
//! `MC_GPU_ADAPTER` (another GPU).
//!
//! It never reaches the network: skin and cape downloads are off, the world comes from a capture
//! file, and the random clicks only land on screens with no sign-in, server, link or folder
//! button (anything else that opens is closed again before the next click).

mod common;

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use mc_client::app::campath::CameraScript;
use mc_client::app::config::LoginMode;
use mc_client::{Client, ClientConfig};
use mc_mods::{OptionKind, OptionValue};
use mc_render::{
    Camera, FrameCtx, FrameStats, GameSettings, Gpu, InputState, PhaseList, RenderError, ShellApp,
    TickCtx,
};
use mc_ui::screens::options::OptionsPage;
use mc_ui::screens::Action;
use winit::event::MouseButton;
use winit::keyboard::KeyCode;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
/// The benchmark capture, relative to this crate (absent in a public checkout: those tests skip).
const CAPTURE: &str =
    "../../optimization/scratch/skywars-scene-2026-11-05/captures/sw-mega-doubles-mid.mccap";
/// Screens where a random click can start a sign-in, a connection, open a link or a folder, or
/// leave what is being swept. The monkey closes them before its next click.
const NOT_FOR_MONKEYS: &[&str] = &[
    "login",
    "connecting",
    "multiplayer",
    "direct_connect",
    "add_server",
    "accounts",
    "resource_packs",
    "confirm_open_link",
    "confirm_server_resource_pack",
    "confirm_leave_server",
    "replay_browser",
    "disconnected",
];

// ---------------------------------------------------------------------------------------------
// Logs: every WARN and ERROR, with the step that was running
// ---------------------------------------------------------------------------------------------

static LOGGED: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());
static STEP: Mutex<String> = Mutex::new(String::new());

fn set_step(name: &str) {
    if let Ok(mut step) = STEP.lock() {
        step.clear();
        step.push_str(name);
    }
    println!("-- step: {name}");
}

fn step() -> String {
    STEP.lock().map(|step| step.clone()).unwrap_or_default()
}

struct Collect;

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for Collect {
    fn on_event(&self, event: &tracing::Event<'_>, _: tracing_subscriber::layer::Context<'_, S>) {
        let meta = event.metadata();
        let mut line = format!("{} {}:", meta.level(), meta.target());
        event.record(&mut Fields(&mut line));
        if let Ok(mut logged) = LOGGED.lock() {
            logged.push((step(), line));
        }
    }
}

struct Fields<'a>(&'a mut String);

impl tracing::field::Visit for Fields<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        use std::fmt::Write as _;
        let _ = write!(self.0, " {}={:?}", field.name(), value);
    }
}

fn init_logs() {
    use tracing_subscriber::filter::LevelFilter;
    use tracing_subscriber::prelude::*;
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _ = tracing_subscriber::registry()
            .with(tracing_subscriber::fmt::layer().with_filter(LevelFilter::INFO))
            .with(Collect.with_filter(LevelFilter::WARN))
            .try_init();
    });
}

/// The environment's own messages, not the client's.
fn benign(line: &str) -> bool {
    line.contains("Loader Message") || line.contains("objects: (type: INSTANCE")
}

fn print_log_summary() {
    let logged = LOGGED
        .lock()
        .map(|logged| logged.clone())
        .unwrap_or_default();
    let mut count = 0;
    println!("== warnings and errors by step");
    for (step, line) in logged.iter().filter(|(_, line)| !benign(line)) {
        count += 1;
        let short: String = line.chars().take(400).collect();
        println!("  [{step}] {short}");
    }
    println!("== {count} warnings and errors");
}

// ---------------------------------------------------------------------------------------------
// The rig
// ---------------------------------------------------------------------------------------------

struct Rig {
    client: Client,
    gpu: Gpu,
    phases: PhaseList,
    camera: Camera,
    input: InputState,
    settings: GameSettings,
    stats: FrameStats,
    ticks: u64,
    frames: u64,
    /// Frames per game tick (one tick runs on every `per_tick`th frame).
    per_tick: u32,
    phase: u32,
    /// Whether frames run `before_ticks` and ticks (false holds the scene still).
    live: bool,
    /// Whether live frames run `before_ticks` (the capture's packets, async results). Off, the
    /// world still ticks but nothing new arrives: the scene settles and stays.
    packets: bool,
    png: Option<PathBuf>,
    shots: u32,
    hitch: Duration,
    hitches: Vec<(String, f64)>,
    width: u32,
    height: u32,
    game_dir: PathBuf,
    rng: u64,
    _dir: Option<tempfile::TempDir>,
}

fn game_dir(name: &str) -> (Option<tempfile::TempDir>, PathBuf) {
    match std::env::var_os("SWEEP_GAME_DIR") {
        Some(root) => {
            let path = PathBuf::from(root).join(name);
            std::fs::create_dir_all(&path).expect("create the game folder");
            (None, path)
        }
        None => {
            let dir = tempfile::tempdir().expect("tempdir");
            let path = dir.path().to_path_buf();
            (Some(dir), path)
        }
    }
}

fn capture_path() -> Option<PathBuf> {
    let path = std::env::var_os("SWEEP_CAPTURE")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join(CAPTURE));
    if path.is_file() {
        Some(path)
    } else {
        eprintln!("skip: no capture at {}", path.display());
        None
    }
}

impl Rig {
    /// The client on its title screen (`world: false`) or in the capture's world.
    fn new(name: &str, world: bool, prepare: impl FnOnce(&Path)) -> Option<Rig> {
        init_logs();
        set_step(&format!("{name}: start"));
        let capture = if world { Some(capture_path()?) } else { None };
        let (dir, game_dir) = game_dir(name);
        prepare(&game_dir);
        // The folders `--gameDir` puts in the game folder (the defaults are relative to the
        // working directory).
        let mut config = ClientConfig {
            game_dir: game_dir.clone(),
            resource_pack_dir: game_dir.join("resourcepacks"),
            assets_dir: game_dir.join("assets"),
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|jar| jar.is_file()) {
            eprintln!("skip: no vanilla jar (set MC_JARS_DIR)");
            return None;
        }
        let per_tick = 3;
        if let Some(capture) = capture {
            config.replay = Some(capture);
            config.replay_step_micros = 50_000 / u64::from(per_tick);
            config.replay_camera = CameraScript::Fixed;
        }
        let gpu = common::gpu(WIDTH, HEIGHT)?;
        let mut client = Client::new(config, &mut |_, _| {}).expect("start");
        client.set_skin_downloads(false);
        let mut phases = PhaseList::new();
        client.install(&gpu, &mut phases, WIDTH, HEIGHT);
        let mut camera = Camera::default();
        camera.set_viewport(WIDTH, HEIGHT);
        let mut input = InputState::new();
        input.set_window_size(WIDTH, HEIGHT);
        println!("adapter: {}", gpu.adapter_info().name);
        let png = std::env::var_os("SWEEP_PNG_DIR").map(|root| PathBuf::from(root).join(name));
        if let Some(png) = &png {
            std::fs::create_dir_all(png).expect("png folder");
        }
        let hitch_ms = std::env::var("SWEEP_HITCH_MS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(250);
        let seed = std::env::var("SWEEP_SEED")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0x5eed_2026_1122);
        let settings = client.settings.clone();
        let mut rig = Rig {
            client,
            gpu,
            phases,
            camera,
            input,
            settings,
            stats: FrameStats::new(),
            ticks: 0,
            frames: 0,
            per_tick,
            phase: 0,
            live: true,
            packets: true,
            png,
            shots: 0,
            hitch: Duration::from_millis(hitch_ms),
            hitches: Vec::new(),
            width: WIDTH,
            height: HEIGHT,
            game_dir,
            rng: seed,
            _dir: dir,
        };
        if world {
            set_step(&format!("{name}: join"));
            rig.client.connect("replay.invalid", 25565);
            let deadline = Instant::now() + Duration::from_secs(120);
            while !(rig.client.ui.name() == "in_game" && rig.ticks > 200) {
                rig.frame();
                assert!(
                    Instant::now() < deadline,
                    "the capture's world never came up (ui {})",
                    rig.client.ui.name()
                );
            }
        } else {
            rig.pump(30);
        }
        Some(rig)
    }

    /// One iteration of the window loop. Returns the frame's pixels when `read`.
    fn frame_inner(&mut self, read: bool) -> Option<Vec<u8>> {
        let started = Instant::now();
        let mut partial = 0.0;
        if self.live {
            if self.packets {
                ShellApp::before_ticks(&mut self.client, &mut self.camera);
            }
            self.phase += 1;
            if self.phase >= self.per_tick {
                self.phase = 0;
                self.ticks += 1;
                ShellApp::tick(
                    &mut self.client,
                    &mut TickCtx {
                        input: &mut self.input,
                        settings: &mut self.settings,
                        camera: &mut self.camera,
                        tick: self.ticks,
                    },
                );
            }
            partial = self.phase as f32 / self.per_tick as f32;
        }
        ShellApp::frame(
            &mut self.client,
            &mut FrameCtx {
                gpu: &self.gpu,
                phases: &mut self.phases,
                camera: &mut self.camera,
                input: &mut self.input,
                settings: &self.settings,
                partial_ticks: partial,
                stats: &self.stats,
            },
        );
        ShellApp::finalize_camera(&mut self.client, &mut self.camera);
        let frame = match self.gpu.acquire() {
            Ok(frame) => frame,
            // A minimised window: nothing to draw into (the shell skips the frame the same way).
            Err(RenderError::FrameUnavailable) => {
                self.gpu.flush_staged_writes();
                return None;
            }
            Err(error) => panic!("[{}] acquire failed: {error}", step()),
        };
        self.phases
            .render(&self.gpu, frame.view(), &self.camera, partial);
        ShellApp::after_render(&mut self.client, &self.gpu, &frame);
        let pixels = read.then(|| frame.read_pixels(&self.gpu).expect("read back"));
        frame.present(&self.gpu.queue);
        self.stats.frame();
        self.frames += 1;
        let took = started.elapsed();
        if took > self.hitch {
            self.hitches.push((step(), took.as_secs_f64() * 1000.0));
        }
        pixels
    }

    fn frame(&mut self) {
        let _ = self.frame_inner(false);
    }

    fn pump(&mut self, frames: usize) {
        for _ in 0..frames {
            self.frame();
        }
    }

    /// Frames until the mesh queue has nothing queued, running or deferred (at most 3000), then
    /// 30 more so the last meshes are uploaded and drawn.
    fn settle_meshes(&mut self) {
        for _ in 0..3000 {
            if self.client.mesh_queue_len() == 0
                && self.client.mesh_queue_in_flight_len() == 0
                && self.client.mesh_queue_deferred_len() == 0
            {
                break;
            }
            self.frame();
        }
        self.pump(30);
    }

    /// The step's picture, `<n>-<label>.png` (only with `SWEEP_PNG_DIR`).
    fn shot(&mut self, label: &str) {
        let Some(dir) = self.png.clone() else {
            self.frame();
            return;
        };
        let Some(pixels) = self.frame_inner(true) else {
            return;
        };
        self.shots += 1;
        let path = dir.join(format!("{:03}-{label}.png", self.shots));
        let (width, height) = (self.gpu.config.width, self.gpu.config.height);
        if pixels.len() == (width * height * 4) as usize {
            image::save_buffer(&path, &pixels, width, height, image::ColorType::Rgba8)
                .expect("write png");
        }
    }

    /// Save pixels read earlier at the current size (only with `SWEEP_PNG_DIR`).
    fn save_png(&mut self, label: &str, pixels: &[u8]) {
        let Some(dir) = self.png.clone() else {
            return;
        };
        let (width, height) = (self.gpu.config.width, self.gpu.config.height);
        if pixels.len() == (width * height * 4) as usize {
            self.shots += 1;
            let path = dir.join(format!("{:03}-{label}.png", self.shots));
            image::save_buffer(&path, pixels, width, height, image::ColorType::Rgba8)
                .expect("write png");
        }
    }

    /// The frame's pixels (the scene held still while reading is the caller's choice).
    fn pixels(&mut self) -> Vec<u8> {
        loop {
            if let Some(pixels) = self.frame_inner(true) {
                return pixels;
            }
        }
    }

    // --- input, as the shell delivers it ---

    fn key(&mut self, code: KeyCode, pressed: bool, typed: Option<char>) {
        self.client.handle_raw_key(code, pressed, false, typed);
        if let Some(input) = self.input.on_key(code, pressed, false) {
            let allowed = ShellApp::game_input_allowed(&self.client);
            self.settings
                .keys
                .set_state_when_input_allowed(input, pressed, allowed);
        }
    }

    fn tap(&mut self, code: KeyCode) {
        self.key(code, true, None);
        self.pump(2);
        self.key(code, false, None);
        self.pump(2);
    }

    /// A chord: `held` down, `code` tapped, `held` up.
    fn chord(&mut self, held: KeyCode, code: KeyCode) {
        self.key(held, true, None);
        self.frame();
        self.tap(code);
        self.key(held, false, None);
        self.pump(2);
    }

    fn type_text(&mut self, text: &str) {
        for c in text.chars() {
            let code = key_for(c);
            self.key(code, true, Some(c));
            self.frame();
            self.key(code, false, None);
        }
        self.frame();
    }

    fn move_to(&mut self, x: f64, y: f64) {
        self.input.on_cursor_moved(x, y);
    }

    fn button(&mut self, button: MouseButton, pressed: bool) {
        let input = self.input.on_mouse_button(button, pressed);
        let allowed = ShellApp::game_input_allowed(&self.client);
        self.settings
            .keys
            .set_state_when_input_allowed(input, pressed, allowed);
    }

    fn click_at(&mut self, x: f64, y: f64, button: MouseButton) {
        self.move_to(x, y);
        self.frame();
        self.button(button, true);
        self.frame();
        self.button(button, false);
        self.pump(2);
    }

    fn scroll(&mut self, lines: f32) {
        self.input
            .on_scroll(winit::event::MouseScrollDelta::LineDelta(0.0, lines));
        self.pump(2);
    }

    // --- window ---

    /// A window resize as the shell applies it: the GPU's targets, the camera, the phases, then
    /// the app.
    fn resize(&mut self, width: u32, height: u32) {
        let rebuilt = self.gpu.resize(width, height);
        let (w, h) = (self.gpu.config.width, self.gpu.config.height);
        self.camera.set_viewport(w, h);
        self.camera.update();
        self.input.set_window_size(width, height);
        if rebuilt {
            self.phases.resize(&self.gpu, w, h);
            ShellApp::resized(&mut self.client, &self.gpu, w, h);
        }
        self.width = w;
        self.height = h;
        self.pump(3);
    }

    /// Select `packs` (folder or zip names in the game folder's `resourcepacks`, first on top, as
    /// the resource packs screen's Done does) and reload.
    fn select_packs(&mut self, packs: &[&str]) {
        self.client.settings.resource_packs = packs.iter().map(|name| (*name).to_owned()).collect();
        let mut progress = |_: &str, _: i32| {};
        if let Err(error) = self.client.reload_resource_packs(
            Some(&self.gpu),
            Some(&mut self.phases),
            &mut progress,
        ) {
            println!("   reload with {packs:?} failed: {error}");
        }
        self.settle_meshes();
    }

    /// The GUI scale the video settings' button sets (0 = auto).
    fn gui_scale(&mut self, scale: i32) {
        self.client.settings.gui_scale = scale;
        self.client.handle_action(Action::Relayout);
        self.pump(3);
    }

    fn act(&mut self, action: Action) {
        self.client.handle_action(action);
        self.pump(4);
    }

    fn ui(&self) -> &'static str {
        self.client.ui.name()
    }

    /// Back out of anything a random click must not touch.
    fn guard(&mut self) {
        for _ in 0..4 {
            if !NOT_FOR_MONKEYS.contains(&self.ui()) {
                return;
            }
            let name = self.ui();
            if name == "connecting" {
                self.client.handle_action(Action::CancelConnect);
            } else {
                self.tap(KeyCode::Escape);
            }
            if self.ui() == name {
                self.client.handle_action(Action::Back);
                self.pump(2);
            }
        }
        assert!(
            !NOT_FOR_MONKEYS.contains(&self.ui()),
            "[{}] could not leave {}",
            step(),
            self.ui()
        );
    }

    fn next(&mut self) -> u64 {
        // xorshift64*: the same clicks every run for a seed.
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        self.rng.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Random clicks, wheel turns and keys on whatever screen is open, never staying on a
    /// screen in [`NOT_FOR_MONKEYS`]. Stops early if the screen leaves `keep` (when given).
    fn monkey(&mut self, actions: usize, keep: Option<&[&str]>) {
        for n in 0..actions {
            let roll = self.next() % 100;
            let x = (self.next() % u64::from(self.width.max(1))) as f64;
            let y = (self.next() % u64::from(self.height.max(1))) as f64;
            match roll {
                0..=69 => self.click_at(x, y, MouseButton::Left),
                70..=79 => self.click_at(x, y, MouseButton::Right),
                80..=87 => {
                    self.move_to(x, y);
                    let lines = if self.next().is_multiple_of(2) {
                        1.0
                    } else {
                        -1.0
                    };
                    self.scroll(lines);
                }
                88..=91 => {
                    // A drag: press, move, release.
                    self.move_to(x, y);
                    self.frame();
                    self.button(MouseButton::Left, true);
                    for step in 1..=6 {
                        self.move_to(x + f64::from(step) * 9.0, y + f64::from(step) * 3.0);
                        self.frame();
                    }
                    self.button(MouseButton::Left, false);
                    self.pump(2);
                }
                _ => {
                    const KEYS: [KeyCode; 8] = [
                        KeyCode::KeyA,
                        KeyCode::Digit1,
                        KeyCode::Backspace,
                        KeyCode::ArrowDown,
                        KeyCode::ArrowUp,
                        KeyCode::Tab,
                        KeyCode::Space,
                        KeyCode::Delete,
                    ];
                    let code = KEYS[(self.next() % KEYS.len() as u64) as usize];
                    self.tap(code);
                }
            }
            // A now-playing widget controls the desktop's media player: never leave it on.
            if self.client.mod_settings_mut().enabled("spotify") {
                self.client.mod_settings_mut().set_enabled("spotify", false);
            }
            self.guard();
            if let Some(keep) = keep {
                if !keep.contains(&self.ui()) {
                    println!("   monkey left {:?} after {} actions", keep, n + 1);
                    return;
                }
            }
        }
    }

    fn report(&mut self, label: &str) {
        let _ = self.gpu.device.poll(wgpu::PollType::wait_indefinitely());
        let (bytes, count) = self
            .gpu
            .device
            .generate_allocator_report()
            .map_or((0, 0), |report| {
                (report.total_allocated_bytes, report.allocations.len())
            });
        let (private, working_set) = mc_render::w21_mb::process_memory()
            .map_or((0.0, 0.0), |memory| {
                (memory.private as f64 / 1e6, memory.working_set as f64 / 1e6)
            });
        println!(
            "== {label:40} ui {:22} frames {:6} gpu {:7.1} MiB in {count:5}; private {private:7.1} \
             MB; working set {working_set:7.1} MB",
            self.ui(),
            self.frames,
            bytes as f64 / 1_048_576.0,
        );
    }

    fn finish(mut self) {
        set_step("shutdown");
        let settings = self.settings.clone();
        ShellApp::shutdown(&mut self.client, &settings);
        println!(
            "== hitches over {} ms: {}",
            self.hitch.as_millis(),
            self.hitches.len()
        );
        for (step, ms) in self.hitches.iter().take(60) {
            println!("  [{step}] {ms:.1} ms");
        }
        print_log_summary();
    }
}

/// The physical key a typed character comes from (US layout; anything else rides on `KeyA`
/// with its own character, which is how the client reads text).
fn key_for(c: char) -> KeyCode {
    match c.to_ascii_lowercase() {
        'a' => KeyCode::KeyA,
        'b' => KeyCode::KeyB,
        'c' => KeyCode::KeyC,
        'd' => KeyCode::KeyD,
        'e' => KeyCode::KeyE,
        'g' => KeyCode::KeyG,
        'h' => KeyCode::KeyH,
        'i' => KeyCode::KeyI,
        'l' => KeyCode::KeyL,
        'm' => KeyCode::KeyM,
        'n' => KeyCode::KeyN,
        'o' => KeyCode::KeyO,
        'p' => KeyCode::KeyP,
        'r' => KeyCode::KeyR,
        's' => KeyCode::KeyS,
        't' => KeyCode::KeyT,
        'u' => KeyCode::KeyU,
        'w' => KeyCode::KeyW,
        'y' => KeyCode::KeyY,
        '0'..='9' => KeyCode::Digit1,
        ' ' => KeyCode::Space,
        '/' => KeyCode::Slash,
        _ => KeyCode::KeyA,
    }
}

/// How many pixels differ between two frames.
fn differing_pixels(a: &[u8], b: &[u8]) -> usize {
    a.chunks_exact(4)
        .zip(b.chunks_exact(4))
        .filter(|(a, b)| a != b)
        .count()
}

/// How many pixels differ by more than `tolerance` in a colour channel (alpha ignored: the
/// window shows colour only). The world keeps ticking between two frames of the mod sweep, so
/// the sun and the fog drift by a level or two; a mod that left something behind differs by far
/// more.
fn visibly_differing_pixels(a: &[u8], b: &[u8], tolerance: u8) -> usize {
    a.chunks_exact(4)
        .zip(b.chunks_exact(4))
        .filter(|(a, b)| (0..3).any(|channel| a[channel].abs_diff(b[channel]) > tolerance))
        .count()
}

// ---------------------------------------------------------------------------------------------
// The sweeps
// ---------------------------------------------------------------------------------------------

/// Window sizes a player meets: the default, small, tall, wide, tiny, minimised, big.
const SIZES: [(u32, u32); 7] = [
    (854, 480),
    (640, 360),
    (720, 1280),
    (2560, 720),
    (64, 48),
    (0, 0),
    (1920, 1080),
];

fn sizes_and_scales(rig: &mut Rig, label: &str) {
    for (width, height) in SIZES {
        set_step(&format!("{label}: resize {width}x{height}"));
        rig.resize(width, height);
        rig.pump(3);
    }
    rig.resize(WIDTH, HEIGHT);
    for scale in [1, 2, 3, 4, 7, 0] {
        set_step(&format!("{label}: gui scale {scale}"));
        rig.gui_scale(scale);
    }
}

/// Every screen the title screen reaches without a world, at every size and GUI scale, with
/// random clicks on the safe ones; F1, F2 and F3 on the title; the vanilla title screen too.
#[test]
#[ignore = "GPU + vanilla jar; a sweep"]
fn title_screens_sweep() {
    let Some(mut rig) = Rig::new("title", false, |_| {}) else {
        return;
    };
    rig.report("title");
    rig.shot("title");
    sizes_and_scales(&mut rig, "title");
    for code in [
        KeyCode::F3,
        KeyCode::F3,
        KeyCode::F1,
        KeyCode::F1,
        KeyCode::F2,
    ] {
        set_step(&format!("title: {code:?}"));
        rig.tap(code);
        rig.pump(6);
    }
    rig.shot("title-after-keys");
    let pages = [
        ("options", Action::OpenOptions),
        ("video", Action::OpenOptionsPage(OptionsPage::Video)),
        ("controls", Action::OpenOptionsPage(OptionsPage::Controls)),
        ("sound", Action::OpenOptionsPage(OptionsPage::Sound)),
        ("chat", Action::OpenOptionsPage(OptionsPage::Chat)),
        ("language", Action::OpenOptionsPage(OptionsPage::Language)),
        ("packs", Action::OpenOptionsPage(OptionsPage::ResourcePacks)),
        ("snooper", Action::OpenOptionsPage(OptionsPage::Snooper)),
        (
            "skin",
            Action::OpenOptionsPage(OptionsPage::SkinCustomisation),
        ),
        ("multiplayer", Action::OpenMultiplayer),
        ("add-server", Action::OpenAddServer),
        ("direct-connect", Action::OpenDirectConnect),
        (
            "accounts",
            Action::Accounts(mc_ui::screens::accounts::AccountsAction::Open),
        ),
        ("mods", Action::OpenModMenu),
        ("stats", Action::OpenStats),
        ("achievements", Action::OpenAchievements),
    ];
    for (label, action) in pages {
        set_step(&format!("title > {label}: open"));
        rig.act(action);
        println!("   {label}: ui {}", rig.ui());
        rig.shot(label);
        let opened = rig.ui();
        for (width, height) in [(640, 360), (1920, 1080), (WIDTH, HEIGHT)] {
            set_step(&format!("title > {label}: resize {width}x{height}"));
            rig.resize(width, height);
        }
        for scale in [1, 4, 0] {
            set_step(&format!("title > {label}: gui scale {scale}"));
            rig.gui_scale(scale);
        }
        if !NOT_FOR_MONKEYS.contains(&opened) {
            set_step(&format!("title > {label}: monkey"));
            rig.monkey(60, Some(&[opened]));
            rig.shot(&format!("{label}-after-monkey"));
        }
        set_step(&format!("title > {label}: escape"));
        for _ in 0..4 {
            if matches!(rig.ui(), "main_menu") {
                break;
            }
            rig.tap(KeyCode::Escape);
            rig.pump(20);
        }
        if rig.ui() != "main_menu" {
            println!(
                "   escape did not reach the title from {label}: ui {}",
                rig.ui()
            );
            rig.client.show_screen(None);
            rig.pump(20);
        }
        rig.report(&format!("after {label}"));
    }
    set_step("title: vanilla title screen");
    rig.client
        .mod_settings_mut()
        .set_enabled("aero_title", false);
    rig.pump(30);
    rig.shot("vanilla-title");
    sizes_and_scales(&mut rig, "vanilla title");
    set_step("title: aero title again");
    rig.client
        .mod_settings_mut()
        .set_enabled("aero_title", true);
    rig.pump(30);
    rig.shot("aero-title-again");
    rig.report("end");
    let screenshots = rig.game_dir.join("screenshots");
    let count = std::fs::read_dir(&screenshots).map_or(0, |dir| dir.count());
    println!(
        "== screenshots written: {count} in {}",
        screenshots.display()
    );
    rig.finish();
}

/// In the capture's world: the HUD and its F keys, every screen a player opens in game, random
/// clicks in the inventory and the options, chat, sizes and GUI scales, and the mod menu.
#[test]
#[ignore = "GPU + vanilla jar + the benchmark capture; a sweep"]
fn world_tour() {
    let Some(mut rig) = Rig::new("world", true, |_| {}) else {
        return;
    };
    rig.report("joined");
    rig.shot("joined");
    for (label, code) in [
        ("f3", KeyCode::F3),
        ("f1", KeyCode::F1),
        ("f1 again", KeyCode::F1),
        ("f5 back", KeyCode::F5),
        ("f5 front", KeyCode::F5),
        ("f5 first", KeyCode::F5),
        ("f2 screenshot", KeyCode::F2),
        ("f11 fullscreen", KeyCode::F11),
        ("f11 window", KeyCode::F11),
    ] {
        set_step(&format!("world: {label}"));
        rig.tap(code);
        rig.pump(10);
        println!(
            "   {label}: wants fullscreen {:?}",
            ShellApp::wants_fullscreen(&rig.client)
        );
        rig.shot(&label.replace(' ', "-"));
    }
    for (label, code) in [
        ("f3+b hitboxes", KeyCode::KeyB),
        ("f3+g chunk borders", KeyCode::KeyG),
        ("f3+h advanced tooltips", KeyCode::KeyH),
        ("f3+t reload packs", KeyCode::KeyT),
        ("f3+a reload chunks", KeyCode::KeyA),
        ("f3+p pause on focus", KeyCode::KeyP),
        ("f3+b again", KeyCode::KeyB),
        ("f3+g again", KeyCode::KeyG),
        ("f3+h again", KeyCode::KeyH),
    ] {
        set_step(&format!("world: {label}"));
        rig.chord(KeyCode::F3, code);
        rig.pump(20);
        // F3+T also opens the chat (T is the chat key, as in vanilla): close it again.
        if rig.ui() != "in_game" {
            println!("   {label}: ui {}", rig.ui());
            rig.tap(KeyCode::Escape);
            rig.pump(4);
        }
    }
    rig.shot("after-f3-chords");
    // The F3 overlay stays on for the next steps (the HUD at every size).
    rig.tap(KeyCode::F3);
    sizes_and_scales(&mut rig, "world");
    rig.tap(KeyCode::F3);
    rig.report("sizes");
    // Hotbar, the wheel, the player list.
    set_step("world: hotbar and wheel");
    for code in [KeyCode::Digit2, KeyCode::Digit9, KeyCode::Digit1] {
        rig.tap(code);
    }
    for _ in 0..12 {
        rig.scroll(1.0);
    }
    set_step("world: player list");
    rig.key(KeyCode::Tab, true, None);
    rig.pump(10);
    rig.shot("player-list");
    rig.key(KeyCode::Tab, false, None);
    rig.pump(4);
    // The inventory, with random clicks in it.
    set_step("world: inventory");
    rig.tap(KeyCode::KeyE);
    rig.pump(6);
    println!("   inventory: ui {}", rig.ui());
    rig.shot("inventory");
    set_step("world: inventory monkey");
    rig.monkey(80, Some(&["container"]));
    rig.shot("inventory-after-monkey");
    for _ in 0..3 {
        if rig.ui() == "in_game" {
            break;
        }
        rig.tap(KeyCode::Escape);
    }
    // Chat: a message and a command, both to a capture that sends nothing anywhere.
    set_step("world: chat");
    rig.tap(KeyCode::KeyT);
    rig.pump(4);
    rig.type_text("hello from the sweep");
    rig.shot("chat-typing");
    rig.tap(KeyCode::Enter);
    rig.pump(4);
    set_step("world: command");
    rig.tap(KeyCode::Slash);
    rig.pump(4);
    rig.type_text("help");
    rig.tap(KeyCode::Tab);
    rig.tap(KeyCode::Enter);
    rig.pump(4);
    set_step("world: chat scrolled");
    rig.tap(KeyCode::KeyT);
    for _ in 0..10 {
        rig.scroll(1.0);
    }
    rig.shot("chat-scrolled");
    rig.tap(KeyCode::Escape);
    rig.pump(4);
    // The pause menu and every options page over the world, with random clicks.
    set_step("world: pause");
    rig.tap(KeyCode::Escape);
    rig.pump(6);
    println!("   pause: ui {}", rig.ui());
    rig.shot("pause");
    let pages = [
        ("options", Action::OpenOptions),
        ("video", Action::OpenOptionsPage(OptionsPage::Video)),
        ("controls", Action::OpenOptionsPage(OptionsPage::Controls)),
        ("chat", Action::OpenOptionsPage(OptionsPage::Chat)),
        ("language", Action::OpenOptionsPage(OptionsPage::Language)),
        (
            "skin",
            Action::OpenOptionsPage(OptionsPage::SkinCustomisation),
        ),
        ("stats", Action::OpenStats),
        ("achievements", Action::OpenAchievements),
    ];
    for (label, action) in pages {
        set_step(&format!("world > {label}"));
        rig.act(action);
        let opened = rig.ui();
        println!("   {label}: ui {opened}");
        rig.shot(&format!("world-{label}"));
        rig.resize(854, 480);
        rig.gui_scale(4);
        rig.resize(WIDTH, HEIGHT);
        rig.gui_scale(0);
        if !NOT_FOR_MONKEYS.contains(&opened) {
            set_step(&format!("world > {label}: monkey"));
            rig.monkey(50, Some(&[opened]));
        }
        set_step(&format!("world > {label}: back"));
        for _ in 0..5 {
            if matches!(rig.ui(), "in_game" | "pause") {
                break;
            }
            rig.tap(KeyCode::Escape);
            rig.pump(6);
        }
        if rig.client.session().is_none() {
            println!("   the session ended during {label}");
            break;
        }
        if rig.ui() == "in_game" {
            rig.tap(KeyCode::Escape);
            rig.pump(6);
        }
    }
    rig.report("options pages");
    // The mod menu over the world: open, random clicks, close; then the key again.
    if rig.client.session().is_some() {
        if rig.ui() != "in_game" {
            rig.client.show_screen(None);
            rig.pump(6);
        }
        set_step("world: mod menu (Right Shift)");
        rig.tap(KeyCode::ShiftRight);
        rig.pump(20);
        println!("   mod menu: ui {}", rig.ui());
        rig.shot("mod-menu");
        set_step("world: mod menu monkey");
        rig.monkey(300, Some(&["mod_menu"]));
        rig.shot("mod-menu-after-monkey");
        set_step("world: mod menu resized");
        rig.resize(854, 480);
        rig.resize(1920, 1080);
        rig.resize(WIDTH, HEIGHT);
        rig.tap(KeyCode::Escape);
        rig.pump(30);
        rig.report("mod menu");
        // Whatever the monkey switched on stays on: play a while with it.
        set_step("world: playing with the monkey's settings");
        rig.pump(300);
        rig.shot("after-mod-menu");
    }
    rig.report("end");
    let screenshots = rig.game_dir.join("screenshots");
    let count = std::fs::read_dir(&screenshots).map_or(0, |dir| dir.count());
    println!("== screenshots written: {count}");
    rig.finish();
}

/// The replay viewer over copies of real recordings (`SWEEP_REPLAYS`, a folder of `.aeroreplay`
/// files, copied into the game folder's library with a nested folder and an odd folder name):
/// the browser from the title screen's Replays button, driven by keys only (its buttons open
/// the system file browser), then each recording played at several speeds, jumped forward and
/// back, through the three cameras, with F1, F3 and F5, the mod menu over it, sizes and GUI
/// scales, then closed; the memory after each.
#[test]
#[ignore = "GPU + vanilla jar + SWEEP_REPLAYS; a sweep"]
fn replays_sweep() {
    use mc_client::app::driver::ReplayCameraMode;
    let Some(source) = std::env::var_os("SWEEP_REPLAYS").map(PathBuf::from) else {
        eprintln!("skip: set SWEEP_REPLAYS to a folder of .aeroreplay files");
        return;
    };
    let mut files: Vec<PathBuf> = std::fs::read_dir(&source)
        .expect("the replay folder")
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "aeroreplay"))
        .collect();
    files.sort();
    if files.is_empty() {
        eprintln!("skip: no .aeroreplay in SWEEP_REPLAYS");
        return;
    }
    let library = std::cell::RefCell::new(Vec::new());
    let Some(mut rig) = Rig::new("replays", false, |dir| {
        let replays = dir.join("replays");
        let nested = replays.join("Season 1").join("Finals");
        let odd = replays.join("odd \u{2713} name (copy) [2] \u{202e}x");
        std::fs::create_dir_all(&nested).expect("nested folder");
        std::fs::create_dir_all(&odd).expect("odd folder");
        for file in &files {
            let to = replays.join(file.file_name().expect("file name"));
            std::fs::copy(file, &to).expect("copy a replay");
            library.borrow_mut().push(to);
        }
        std::fs::copy(&files[0], nested.join("finals.aeroreplay")).expect("copy");
        std::fs::copy(&files[files.len() - 1], odd.join("x.aeroreplay")).expect("copy");
    }) else {
        return;
    };
    rig.report("title");
    // The loading screen covers the title (and takes the clicks) for about a second.
    let shown = Instant::now();
    while shown.elapsed() < Duration::from_secs(4) {
        rig.frame();
    }
    // The title screen's Replays button (the second button of the Aero title at 1280x720).
    set_step("replays: the browser from the title");
    rig.click_at(198.0, 226.0, MouseButton::Left);
    rig.pump(30);
    println!("   after the Replays button: ui {}", rig.ui());
    if rig.ui() == "replay_browser" {
        rig.shot("browser");
        for (label, keys) in [
            ("down", vec![KeyCode::ArrowDown, KeyCode::ArrowDown]),
            ("folders view", vec![KeyCode::KeyF]),
            ("into a folder", vec![KeyCode::ArrowDown, KeyCode::Enter]),
            ("deeper", vec![KeyCode::ArrowDown, KeyCode::ArrowRight]),
            ("up", vec![KeyCode::Backspace]),
            ("up again", vec![KeyCode::ArrowLeft, KeyCode::Backspace]),
            ("sort menu", vec![KeyCode::KeyO]),
            (
                "sort pick",
                vec![KeyCode::ArrowDown, KeyCode::ArrowDown, KeyCode::Enter],
            ),
            ("list view", vec![KeyCode::KeyF]),
            (
                "end of list",
                vec![KeyCode::End, KeyCode::PageUp, KeyCode::Home],
            ),
        ] {
            set_step(&format!("replays: browser {label}"));
            for key in keys {
                rig.tap(key);
            }
            rig.pump(10);
            rig.shot(&format!("browser-{}", label.replace(' ', "-")));
        }
        for (width, height) in [(640, 360), (1920, 1080), (WIDTH, HEIGHT)] {
            set_step(&format!("replays: browser resize {width}x{height}"));
            rig.resize(width, height);
        }
        for scale in [1, 4, 0] {
            set_step(&format!("replays: browser gui scale {scale}"));
            rig.gui_scale(scale);
        }
        rig.shot("browser-after-sizes");
        set_step("replays: browser escape");
        rig.tap(KeyCode::Escape);
        rig.pump(10);
        println!("   after Escape: ui {}", rig.ui());
        if rig.ui() != "main_menu" {
            rig.client.show_screen(None);
            rig.pump(10);
        }
    }
    // The browser's listing moves recordings left flat in `replays\` into date folders (the
    // library's `organize`): find each copy again by name.
    let mut found = Vec::new();
    let mut folders = vec![rig.game_dir.join("replays")];
    while let Some(folder) = folders.pop() {
        for entry in std::fs::read_dir(&folder).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                folders.push(path);
            } else if path.extension().is_some_and(|ext| ext == "aeroreplay") {
                found.push(path);
            }
        }
    }
    let library: Vec<PathBuf> = library
        .into_inner()
        .into_iter()
        .filter_map(|copy| {
            let name = copy.file_name()?.to_owned();
            found
                .iter()
                .find(|path| path.file_name() == Some(name.as_os_str()))
                .cloned()
        })
        .collect();
    println!("   recordings after the browser: {}", library.len());
    for (index, path) in library.iter().enumerate() {
        let label = format!("replay {index}");
        set_step(&format!("{label}: open"));
        rig.client.replay_open(path).expect("open the replay");
        rig.pump(60);
        rig.report(&format!("{label}: opened"));
        rig.shot(&format!("replay{index}-opened"));
        let (start, end) = rig.client.replay_view_range();
        let span = end.saturating_sub(start);
        set_step(&format!("{label}: play 1x, 4x, 0.25x"));
        rig.client.replay_play();
        rig.pump(120);
        rig.client.replay_set_speed(4.0);
        rig.pump(120);
        rig.client.replay_set_speed(0.25);
        rig.pump(60);
        rig.client.replay_pause();
        for (step, at) in [
            ("jump to the middle", start + span / 2),
            ("jump back to a quarter", start + span / 4),
            ("jump to the end", end),
            ("jump to the start", start),
            ("jump to 90%", start + span / 10 * 9),
        ] {
            set_step(&format!("{label}: {step}"));
            rig.client.replay_seek(at);
            rig.pump(40);
        }
        rig.shot(&format!("replay{index}-after-jumps"));
        for (name, mode) in [
            ("third person", ReplayCameraMode::ThirdPerson),
            ("free", ReplayCameraMode::Free),
            ("pov", ReplayCameraMode::Pov),
        ] {
            set_step(&format!("{label}: camera {name}"));
            rig.client.replay_set_camera_mode(mode);
            rig.client.replay_play();
            rig.pump(60);
            rig.shot(&format!("replay{index}-camera-{}", name.replace(' ', "-")));
        }
        for code in [
            KeyCode::F1,
            KeyCode::F1,
            KeyCode::F3,
            KeyCode::F5,
            KeyCode::F5,
            KeyCode::F5,
            KeyCode::F3,
            KeyCode::Space,
            KeyCode::ArrowRight,
            KeyCode::ArrowLeft,
            KeyCode::Space,
        ] {
            set_step(&format!("{label}: {code:?}"));
            rig.tap(code);
            rig.pump(6);
        }
        set_step(&format!("{label}: mod menu over the replay"));
        rig.tap(KeyCode::ShiftRight);
        rig.pump(20);
        rig.shot(&format!("replay{index}-mod-menu"));
        rig.tap(KeyCode::Escape);
        rig.pump(20);
        set_step(&format!("{label}: sizes and scales"));
        rig.resize(854, 480);
        rig.gui_scale(4);
        rig.resize(WIDTH, HEIGHT);
        rig.gui_scale(0);
        rig.pump(30);
        set_step(&format!("{label}: close"));
        rig.client.replay_close();
        rig.pump(30);
        rig.report(&format!("{label}: closed, ui"));
    }
    rig.finish();
}

/// The replay viewer positioned frame by frame the way a video export positions it
/// (`replay_render_at`): a second from the start, then 30 s in (held there, as the export's
/// settle frames hold it), then frame by frame from there; the time each call takes and the
/// checkpoint restores it made. `SWEEP_REPLAY_FILE` names the recording (the first of
/// `SWEEP_REPLAYS` otherwise).
#[test]
#[ignore = "GPU + vanilla jar + a recording; timings"]
fn replay_render_at_steps() {
    let path = std::env::var_os("SWEEP_REPLAY_FILE")
        .map(PathBuf::from)
        .or_else(|| {
            let dir = std::env::var_os("SWEEP_REPLAYS")?;
            let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
                .ok()?
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| path.extension().is_some_and(|ext| ext == "aeroreplay"))
                .collect();
            files.sort();
            files.into_iter().next()
        });
    let Some(path) = path else {
        eprintln!("skip: set SWEEP_REPLAY_FILE or SWEEP_REPLAYS");
        return;
    };
    let Some(mut rig) = Rig::new("render-at", false, |_| {}) else {
        return;
    };
    rig.client.replay_open(&path).expect("open the replay");
    rig.pump(30);
    let (start, _) = rig.client.replay_view_range();
    let frame_us = 33_333;
    let at30 = start + 30_000_000;
    let steps: Vec<(&str, mc_replay::ReplayTime)> = (0..30)
        .map(|k| ("from the start", start + k * frame_us))
        .chain((0..10).map(|_| ("held at 30 s", at30)))
        .chain((0..30).map(|k| ("from 30 s", at30 + k * frame_us)))
        .chain((0..10).map(|_| ("held at 5 s", start + 5_000_000)))
        .chain((0..10).map(|k| ("from 5 s", start + 5_000_000 + k * frame_us)))
        .collect();
    let mut slow = 0;
    for (label, time) in steps {
        set_step(&format!("render at: {label}"));
        let restored = rig
            .client
            .replay_checkpoint_stats()
            .map_or(0, |stats| stats.restored);
        let started = Instant::now();
        rig.client.replay_render_at(time);
        let took = started.elapsed();
        rig.frame();
        let restored_now = rig
            .client
            .replay_checkpoint_stats()
            .map_or(0, |stats| stats.restored);
        if took > Duration::from_millis(100) {
            slow += 1;
        }
        println!(
            "   {label:15} t={:>9} us: {:7.1} ms, checkpoint restores +{}, replay time {}",
            time - start,
            took.as_secs_f64() * 1e3,
            restored_now - restored,
            rig.client.replay_time().saturating_sub(start)
        );
    }
    println!("== calls over 100 ms: {slow}");
    rig.client.replay_close();
    rig.finish();
}

/// The multiplayer list over a `servers.dat` with odd entries: long, formatted, right-to-left
/// and empty names; empty, half-written and out-of-range addresses; and 300 more rows. Every
/// address is an IP literal on this machine (loopback, nothing listening), so the list's pings
/// end at once and nothing leaves the PC (no host name, so no DNS either). Opened, pinged,
/// scrolled by keys and the wheel, resized and rescaled, closed.
#[test]
#[ignore = "GPU + vanilla jar; a sweep"]
fn multiplayer_list_sweep() {
    use mc_ui::screens::multiplayer::{ServerEntry, ServerList};
    let Some(mut rig) = Rig::new("servers", false, |dir| {
        let mut list = ServerList::new();
        let long = "x".repeat(500);
        let odd = [
            ("", "127.0.0.1:1"),
            ("\u{a7}k\u{a7}l\u{a7}cFormatted \u{a7}r", "127.0.0.1:2"),
            ("\u{202e}right to left\u{202c}", "[::1]:3"),
            (long.as_str(), "127.0.0.1:65535"),
            ("port zero", "127.0.0.1:0"),
            ("port too big", "127.0.0.1:99999"),
            ("negative port", "127.0.0.1:-1"),
            ("ipv6 port too big", "[::1]:70000"),
            ("only colons", "::"),
            ("unspecified", "0.0.0.0:4"),
            ("\u{1F600}\u{1F600}\u{1F600}", "127.0.0.1:5"),
        ];
        // No entry may fall back to the default port: that port takes a DNS SRV lookup (as in
        // vanilla) and would reach a server on this machine if one were running.
        for (name, address) in odd {
            list.add(ServerEntry::new(name, address));
        }
        for n in 0..300 {
            list.add(ServerEntry::new(
                format!("Row {n}"),
                format!("127.0.0.1:{}", 20_000 + n),
            ));
        }
        let mut bytes = Vec::new();
        list.write(&mut bytes).expect("encode servers.dat");
        std::fs::write(dir.join("servers.dat"), bytes).expect("write servers.dat");
    }) else {
        return;
    };
    set_step("servers: open the list");
    let opened = Instant::now();
    rig.act(Action::OpenMultiplayer);
    rig.pump(120);
    println!("   ui {}", rig.ui());
    rig.shot("list");
    // One click on the first row's name selects it (the icon's right half would join). Then the
    // moves a player makes: Shift+Up on the top row (nothing), Shift+Down three times (the row
    // moves, servers.dat is saved), the arrows past both ends (the selection clears).
    set_step("servers: select and move");
    rig.click_at(700.0, 150.0, MouseButton::Left);
    rig.chord(KeyCode::ShiftLeft, KeyCode::ArrowUp);
    for _ in 0..3 {
        rig.chord(KeyCode::ShiftLeft, KeyCode::ArrowDown);
    }
    for _ in 0..5 {
        rig.tap(KeyCode::ArrowDown);
    }
    for _ in 0..12 {
        rig.tap(KeyCode::ArrowUp);
    }
    rig.shot("list-moved");
    set_step("servers: wait for the pings");
    while opened.elapsed() < Duration::from_secs(12) {
        rig.pump(10);
    }
    rig.shot("list-settled");
    set_step("servers: refresh");
    rig.tap(KeyCode::F5);
    rig.pump(30);
    println!("   after F5: ui {}", rig.ui());
    set_step("servers: drag the scrollbar to the bottom");
    rig.move_to(1108.0, 150.0);
    rig.frame();
    rig.button(MouseButton::Left, true);
    for step in 0..20 {
        rig.move_to(1108.0, 150.0 + f64::from(step) * 30.0);
        rig.frame();
    }
    rig.button(MouseButton::Left, false);
    rig.pump(30);
    rig.shot("list-bottom");
    rig.click_at(700.0, 360.0, MouseButton::Left);
    for _ in 0..3 {
        rig.tap(KeyCode::ArrowDown);
    }
    for _ in 0..2 {
        rig.chord(KeyCode::ShiftLeft, KeyCode::ArrowDown);
    }
    for (label, code, times) in [
        ("down", KeyCode::ArrowDown, 40),
        ("up", KeyCode::ArrowUp, 10),
        ("end", KeyCode::End, 1),
        ("home", KeyCode::Home, 1),
    ] {
        set_step(&format!("servers: {label}"));
        for _ in 0..times {
            rig.tap(code);
        }
    }
    set_step("servers: wheel");
    rig.move_to(640.0, 360.0);
    for _ in 0..30 {
        rig.scroll(-1.0);
    }
    rig.shot("list-scrolled");
    sizes_and_scales(&mut rig, "servers");
    rig.pump(120);
    rig.shot("list-after-sizes");
    set_step("servers: close");
    rig.tap(KeyCode::Escape);
    rig.pump(20);
    println!("   after Escape: ui {}", rig.ui());
    let saved = std::fs::read(rig.game_dir.join("servers.dat")).expect("servers.dat");
    let saved = ServerList::read(&saved).expect("servers.dat still reads");
    let names: Vec<&str> = saved.entries().iter().map(|e| e.name.as_str()).collect();
    println!(
        "   servers.dat: {} entries; first five {:?}",
        names.len(),
        &names[..5.min(names.len())]
    );
    assert_eq!(names.len(), 311, "no row lost or added");
    rig.finish();
}

/// Every vanilla video option (and the client's own on that page) through its values and back,
/// changed while the Video Settings screen is open as a player does, with the world still (no
/// packet arrives): back at its value the option must draw the frame from before. The same check
/// as `every_mod_on_and_off`, for the settings that are not mods.
#[test]
#[ignore = "GPU + vanilla jar + the benchmark capture; a sweep"]
fn video_options_round_trip() {
    let Some(mut rig) = Rig::new("video", true, |dir| {
        std::fs::write(dir.join("options.txt"), "particles:2\nrenderClouds:false\n")
            .expect("write options.txt");
        std::fs::write(
            dir.join(mc_mods::SETTINGS_FILE_NAME),
            r#"{"format":1,"mods":{"vignette":{"enabled":false}}}"#,
        )
        .expect("write the mod settings");
    }) else {
        return;
    };
    mc_render::entity::item::pin_animation_clock(Some(1_000_000));
    rig.packets = false;
    set_step("video: settle");
    rig.pump(300);
    rig.settle_meshes();
    let mut left_behind: Vec<(&str, usize)> = Vec::new();
    fn close(rig: &mut Rig) {
        for _ in 0..4 {
            if rig.ui() == "in_game" {
                return;
            }
            rig.tap(KeyCode::Escape);
            rig.pump(6);
        }
        rig.client.show_screen(None);
        rig.pump(6);
    }
    macro_rules! round_trip {
        ($field:ident, $values:expr) => {{
            let name = stringify!($field);
            let original = rig.client.settings.$field.clone();
            let before = rig.pixels();
            for value in $values {
                set_step(&format!("video: {name} = {value:?}"));
                rig.act(Action::OpenOptionsPage(OptionsPage::Video));
                rig.client.settings.$field = value;
                rig.pump(6);
                close(&mut rig);
                rig.pump(30);
            }
            set_step(&format!("video: {name} back to {original:?}"));
            rig.act(Action::OpenOptionsPage(OptionsPage::Video));
            rig.client.settings.$field = original;
            rig.pump(6);
            close(&mut rig);
            rig.pump(90);
            rig.settle_meshes();
            rig.pump(30);
            let after = rig.pixels();
            let visibly = visibly_differing_pixels(&before, &after, 8);
            if visibly > 0 {
                println!("   {name}: {visibly} pixels visibly differ after the round trip");
                rig.save_png(&format!("{name}-before"), &before);
                rig.save_png(&format!("{name}-after"), &after);
                left_behind.push((name, visibly));
            }
        }};
    }
    let flip = |value: bool| [!value, value];
    let fancy = rig.client.settings.fancy_graphics;
    let bobbing = rig.client.settings.view_bobbing;
    let shadows = rig.client.settings.entity_shadows;
    let alternatives = rig.client.settings.allow_block_alternatives;
    let fullbright = rig.client.settings.fullbright;
    round_trip!(render_distance_chunks, [2, 32, 5]);
    round_trip!(fancy_graphics, flip(fancy));
    round_trip!(ambient_occlusion, [0, 1, 2]);
    round_trip!(mipmap_levels, [0, 4, 2]);
    round_trip!(gamma, [0.0, 1.0, 0.5]);
    round_trip!(clouds, [1, 2, 0]);
    round_trip!(particles, [0, 1]);
    round_trip!(view_bobbing, flip(bobbing));
    round_trip!(entity_shadows, flip(shadows));
    round_trip!(allow_block_alternatives, flip(alternatives));
    round_trip!(fov, [30.0, 110.0, 90.0]);
    round_trip!(anaglyph, [true, false]);
    round_trip!(fbo_enable, [false, true]);
    round_trip!(use_vbo, [false, true]);
    round_trip!(msaa_samples, [2, 4, 8, 1]);
    round_trip!(fullbright, flip(fullbright));
    round_trip!(entity_render_distance, [8, 64, 0]);
    round_trip!(player_render_distance, [8, 64, 0]);
    rig.report("video");
    mc_render::entity::item::pin_animation_clock(None);
    rig.finish();
    println!("== video options whose round trip left the frame different: {left_behind:?}");
    assert!(
        left_behind.is_empty(),
        "video options back at their value draw a different frame: {left_behind:?}"
    );
}

/// Languages that take the font's other paths: right to left (the bidi reorder and Arabic
/// shaping run on every string drawn), the unicode font (CJK, Thai), the joke languages, and the
/// forced unicode font. In each: the title's screens at two sizes, then in game the HUD with
/// every HUD mod on, F3, the player list, chat with typed text, the inventory and the mod menu.
#[test]
#[ignore = "GPU + vanilla jar + the benchmark capture; a sweep"]
fn languages_sweep() {
    const LANGUAGES: [&str; 9] = [
        "ar_SA", "he_IL", "fa_IR", "zh_CN", "ja_JP", "th_TH", "en_PT", "lol_US", "qya_AA",
    ];
    let set_language = |rig: &mut Rig, code: &str, unicode: bool| {
        set_step(&format!(
            "languages: {code}{}",
            if unicode { " + unicode" } else { "" }
        ));
        rig.client.settings.language = code.to_owned();
        rig.client.settings.force_unicode_font = unicode;
        rig.act(Action::Relayout);
        rig.pump(6);
    };
    if let Some(mut rig) = Rig::new("languages-title", false, |_| {}) {
        let pages = [
            Action::OpenOptions,
            Action::OpenOptionsPage(OptionsPage::Video),
            Action::OpenOptionsPage(OptionsPage::Controls),
            Action::OpenOptionsPage(OptionsPage::Language),
            Action::OpenOptionsPage(OptionsPage::ResourcePacks),
            Action::OpenMultiplayer,
            Action::OpenModMenu,
            Action::Accounts(mc_ui::screens::accounts::AccountsAction::Open),
            Action::OpenStats,
            Action::OpenAchievements,
        ];
        for (n, code) in LANGUAGES.iter().enumerate() {
            set_language(&mut rig, code, n == LANGUAGES.len() - 1);
            rig.shot(&format!("title-{code}"));
            for page in pages.clone() {
                rig.act(page);
                let opened = rig.ui();
                set_step(&format!("languages: {code} {opened}"));
                rig.resize(640, 360);
                rig.resize(WIDTH, HEIGHT);
                if *code == "ar_SA" || *code == "zh_CN" {
                    rig.shot(&format!("{code}-{opened}"));
                }
                for _ in 0..4 {
                    if rig.ui() == "main_menu" {
                        break;
                    }
                    rig.tap(KeyCode::Escape);
                    rig.pump(10);
                }
                if rig.ui() != "main_menu" {
                    rig.client.show_screen(None);
                    rig.pump(10);
                }
            }
        }
        rig.finish();
    }
    let Some(mut rig) = Rig::new("languages-world", true, |_| {}) else {
        return;
    };
    for descriptor in mc_mods::catalog::all() {
        if !descriptor.hud_elements.is_empty() && descriptor.id != "spotify" {
            rig.client
                .mod_settings_mut()
                .set_enabled(descriptor.id, true);
        }
    }
    for (n, code) in LANGUAGES.iter().enumerate() {
        set_language(&mut rig, code, n == 0);
        rig.tap(KeyCode::F3);
        rig.pump(10);
        rig.key(KeyCode::Tab, true, None);
        rig.pump(10);
        rig.shot(&format!("hud-{code}"));
        rig.key(KeyCode::Tab, false, None);
        rig.tap(KeyCode::F3);
        rig.tap(KeyCode::KeyT);
        rig.type_text("abc (123) [x] 12:30 50% d-e");
        rig.pump(6);
        rig.shot(&format!("chat-{code}"));
        rig.tap(KeyCode::Escape);
        rig.tap(KeyCode::KeyE);
        rig.pump(10);
        rig.tap(KeyCode::Escape);
        rig.tap(KeyCode::ShiftRight);
        rig.pump(20);
        if *code == "ar_SA" || *code == "zh_CN" {
            rig.shot(&format!("mods-{code}"));
        }
        rig.tap(KeyCode::Escape);
        rig.pump(10);
        if rig.ui() != "in_game" {
            rig.client.show_screen(None);
            rig.pump(10);
        }
    }
    set_language(&mut rig, "en_US", false);
    rig.finish();
}

/// Every mod with a HUD element on (and the rest of the HUD: F3, chat open, the player list),
/// at every window size and GUI scale, then with the mod menu's HUD editor open and random drags
/// in it, at every size again: layout arithmetic at its extremes (debug builds check overflow).
#[test]
#[ignore = "GPU + vanilla jar + the benchmark capture; a sweep"]
fn hud_at_every_size() {
    let Some(mut rig) = Rig::new("hud", true, |_| {}) else {
        return;
    };
    for descriptor in mc_mods::catalog::all() {
        if !descriptor.hud_elements.is_empty() && descriptor.id != "spotify" {
            rig.client
                .mod_settings_mut()
                .set_enabled(descriptor.id, true);
        }
    }
    rig.pump(30);
    rig.shot("hud-all");
    rig.tap(KeyCode::F3);
    sizes_and_scales(&mut rig, "hud with F3");
    rig.tap(KeyCode::F3);
    rig.key(KeyCode::Tab, true, None);
    sizes_and_scales(&mut rig, "hud with the player list");
    rig.key(KeyCode::Tab, false, None);
    rig.tap(KeyCode::KeyT);
    sizes_and_scales(&mut rig, "hud with chat open");
    rig.tap(KeyCode::Escape);
    rig.pump(10);
    // The HUD editor: the mod menu's Edit HUD button (bottom centre at 1280x720).
    set_step("hud: the HUD editor");
    rig.tap(KeyCode::ShiftRight);
    rig.pump(30);
    rig.click_at(640.0, 650.0, MouseButton::Left);
    rig.pump(20);
    println!("   after Edit HUD: ui {}", rig.ui());
    rig.shot("hud-editor");
    set_step("hud: drags in the HUD editor");
    rig.monkey(120, Some(&["mod_menu"]));
    rig.shot("hud-editor-after-drags");
    sizes_and_scales(&mut rig, "hud editor");
    rig.shot("hud-editor-after-sizes");
    rig.tap(KeyCode::Escape);
    rig.pump(10);
    rig.tap(KeyCode::Escape);
    rig.pump(30);
    rig.shot("hud-after-editor");
    rig.finish();
}

/// Pipelines made and dropped again and again: Super Secret Settings through every shader,
/// then each shader pack and anti-aliasing mode on and off, `SWEEP_CYCLES` times (4 by
/// default), with the process's memory after each cycle. Growth that keeps coming back cycle
/// after cycle is a leak; growth that stops after the first cycle is a cache filling once.
#[test]
#[ignore = "GPU + vanilla jar + the benchmark capture; memory"]
fn pipeline_churn() {
    let cycles: usize = std::env::var("SWEEP_CYCLES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4);
    let Some(mut rig) = Rig::new("churn", true, |_| {}) else {
        return;
    };
    rig.report("joined");
    let shaders = match mc_mods::catalog::find("super_secret_settings")
        .and_then(|descriptor| descriptor.option("shader"))
        .map(|option| option.kind)
    {
        Some(OptionKind::Choice { choices }) => choices.len(),
        _ => 0,
    };
    for cycle in 1..=cycles {
        set_step(&format!("churn: cycle {cycle}"));
        rig.client
            .mod_settings_mut()
            .set_enabled("super_secret_settings", true);
        for shader in 0..shaders {
            rig.client.mod_settings_mut().set_value(
                "super_secret_settings",
                "shader",
                OptionValue::Choice(shader),
            );
            rig.pump(3);
        }
        rig.client
            .mod_settings_mut()
            .set_enabled("super_secret_settings", false);
        for pack in ["shaders_classic", "shaders_dreamy", "shaders_underwater"] {
            rig.client.mod_settings_mut().set_enabled(pack, true);
            rig.pump(6);
            rig.client.mod_settings_mut().set_enabled(pack, false);
            rig.pump(3);
        }
        rig.client
            .mod_settings_mut()
            .set_enabled("anti_aliasing", true);
        for mode in 0..3 {
            rig.client.mod_settings_mut().set_value(
                "anti_aliasing",
                "mode",
                OptionValue::Choice(mode),
            );
            rig.pump(6);
        }
        rig.client
            .mod_settings_mut()
            .set_enabled("anti_aliasing", false);
        rig.pump(30);
        rig.report(&format!("after cycle {cycle}"));
    }
    rig.finish();
}

/// A folder resource pack with `textures` (path under `assets/minecraft/textures/`, size, RGBA)
/// and extra raw files (path under the pack, bytes).
fn write_pack(
    root: &Path,
    name: &str,
    mcmeta: &str,
    textures: &[(&str, u32, [u8; 4])],
    raw: &[(&str, &[u8])],
) {
    let pack = root.join(name);
    std::fs::create_dir_all(&pack).expect("pack folder");
    std::fs::write(pack.join("pack.mcmeta"), mcmeta).expect("pack.mcmeta");
    for (path, size, rgba) in textures {
        let file = pack.join("assets/minecraft/textures").join(path);
        std::fs::create_dir_all(file.parent().expect("parent")).expect("texture folder");
        let image = image::RgbaImage::from_pixel(*size, *size, image::Rgba(*rgba));
        image.save(&file).expect("write texture");
    }
    for (path, bytes) in raw {
        let file = pack.join(path);
        std::fs::create_dir_all(file.parent().expect("parent")).expect("folder");
        std::fs::write(&file, bytes).expect("write file");
    }
}

/// Resource packs switched in game, the way the resource packs screen's Done reloads them:
/// a recolour pack, a high-resolution one, a broken one (bad JSON, a PNG that is not one, a
/// model and a blockstate of garbage, an odd language file), combinations, the packs in
/// `SWEEP_PACKS` (a folder of real packs, copied once into the game folder), each shader pack
/// over them, and back to none; a picture after each and the warnings collected.
#[test]
#[ignore = "GPU + vanilla jar + the benchmark capture; a sweep"]
fn packs_sweep() {
    let real: Vec<String> = std::env::var_os("SWEEP_PACKS")
        .map(PathBuf::from)
        .and_then(|dir| std::fs::read_dir(dir).ok())
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| entry.file_name().to_str().map(str::to_owned))
        .collect();
    let Some(mut rig) = Rig::new("packs", true, |dir| {
        let packs = dir.join("resourcepacks");
        let meta = r#"{"pack":{"pack_format":1,"description":"sweep"}}"#;
        write_pack(
            &packs,
            "sweep-recolor",
            meta,
            &[
                ("blocks/stone.png", 16, [200, 40, 40, 255]),
                ("blocks/cobblestone.png", 16, [40, 40, 200, 255]),
                ("blocks/grass_top.png", 16, [250, 250, 250, 255]),
            ],
            &[],
        );
        write_pack(
            &packs,
            "sweep-hd",
            meta,
            &[
                ("blocks/stone.png", 128, [120, 120, 140, 255]),
                ("blocks/stonebrick.png", 256, [90, 90, 90, 255]),
                ("items/diamond_sword.png", 64, [0, 255, 255, 255]),
            ],
            &[],
        );
        write_pack(
            &packs,
            "sweep-broken",
            "{\"pack\":{\"pack_format\":1,\"description\":",
            &[],
            &[
                (
                    "assets/minecraft/textures/blocks/stone.png",
                    b"not a png at all".as_slice(),
                ),
                (
                    "assets/minecraft/textures/blocks/dirt.png",
                    [0x89, b'P', b'N', b'G', 0, 0].as_slice(),
                ),
                (
                    "assets/minecraft/models/block/stone.json",
                    b"{\"parent\":{{{".as_slice(),
                ),
                (
                    "assets/minecraft/blockstates/stone.json",
                    b"[1,2,3]".as_slice(),
                ),
                (
                    "assets/minecraft/lang/en_US.lang",
                    b"key.attack=\xff\xfe\n=\n\n%s%s%s=x".as_slice(),
                ),
                ("assets/minecraft/sounds.json", b"{\"x\":5}".as_slice()),
            ],
        );
        // The same garbage behind a valid pack.mcmeta, so the pack loads and its files are read.
        write_pack(
            &packs,
            "sweep-broken-inside",
            meta,
            &[],
            &[
                (
                    "assets/minecraft/textures/blocks/stone.png",
                    b"not a png at all".as_slice(),
                ),
                (
                    "assets/minecraft/textures/blocks/dirt.png",
                    [0x89, b'P', b'N', b'G', 0, 0].as_slice(),
                ),
                (
                    "assets/minecraft/textures/blocks/sand.png.mcmeta",
                    b"{\"animation\":{\"frametime\":-5,\"frames\":[99,-1]}}".as_slice(),
                ),
                (
                    "assets/minecraft/models/block/stone.json",
                    b"{\"parent\":{{{".as_slice(),
                ),
                (
                    "assets/minecraft/models/block/cube.json",
                    b"{\"elements\":[{\"from\":[0,0,0],\"to\":[1e9,-1e9,16]}]}".as_slice(),
                ),
                (
                    "assets/minecraft/blockstates/stone.json",
                    b"[1,2,3]".as_slice(),
                ),
                (
                    "assets/minecraft/lang/en_US.lang",
                    b"key.attack=\xff\xfe\n=\n\n%s%s%s=x".as_slice(),
                ),
                ("assets/minecraft/sounds.json", b"{\"x\":5}".as_slice()),
                (
                    "assets/minecraft/mcpatcher/ctm/stone/stone.properties",
                    b"method=ctm\ntiles=0-99999\nmatchBlocks=1\n".as_slice(),
                ),
                (
                    "assets/minecraft/mcpatcher/sky/world0/sky1.properties",
                    b"startFadeIn=25:00\nsource=../../../../x.png\nblend=???\n".as_slice(),
                ),
            ],
        );
        // Real packs, copied once (they can be large): the copy is kept between runs.
        if let Some(source) = std::env::var_os("SWEEP_PACKS").map(PathBuf::from) {
            for name in &real {
                let to = packs.join(name);
                if !to.exists() {
                    copy_tree(&source.join(name), &to);
                }
            }
        }
    }) else {
        return;
    };
    rig.settle_meshes();
    rig.shot("vanilla");
    let mut sets: Vec<Vec<&str>> = vec![
        vec!["sweep-recolor"],
        vec!["sweep-hd"],
        vec!["sweep-broken"],
        vec!["sweep-broken-inside"],
        vec![
            "sweep-recolor",
            "sweep-hd",
            "sweep-broken",
            "sweep-broken-inside",
        ],
        vec!["missing-pack"],
    ];
    for name in &real {
        sets.push(vec![name.as_str()]);
        sets.push(vec!["sweep-recolor", name.as_str()]);
    }
    for (index, set) in sets.iter().enumerate() {
        set_step(&format!("packs: {set:?}"));
        let started = Instant::now();
        rig.select_packs(set);
        println!("   {set:?}: {:.1} s", started.elapsed().as_secs_f64());
        rig.pump(30);
        rig.report(&format!("packs {index}"));
        rig.shot(&format!("packs-{index}"));
    }
    // The last real pack (or the high-resolution one) with its connected textures and custom sky
    // on, then under each shader pack.
    let last = real.last().map_or("sweep-hd", String::as_str);
    for (step, packs) in [
        ("real", vec![last]),
        ("broken", vec!["sweep-broken-inside", last]),
    ] {
        set_step(&format!(
            "packs: connected textures and custom sky with the {step} pack"
        ));
        rig.select_packs(&packs);
        rig.client
            .mod_settings_mut()
            .set_enabled("connected_textures", true);
        rig.client
            .mod_settings_mut()
            .set_enabled("custom_sky", true);
        rig.pump(30);
        rig.settle_meshes();
        rig.shot(&format!("ctm-sky-{step}"));
        rig.client
            .mod_settings_mut()
            .set_enabled("connected_textures", false);
        rig.client
            .mod_settings_mut()
            .set_enabled("custom_sky", false);
        rig.pump(30);
        rig.settle_meshes();
    }
    rig.select_packs(&[last]);
    for shader in ["shaders_classic", "shaders_dreamy", "shaders_underwater"] {
        set_step(&format!("packs: {shader} over {last}"));
        rig.client.mod_settings_mut().set_enabled(shader, true);
        rig.pump(60);
        rig.settle_meshes();
        rig.shot(shader);
        set_step(&format!("packs: F3+T under {shader}"));
        rig.chord(KeyCode::F3, KeyCode::KeyT);
        rig.pump(10);
        if rig.ui() != "in_game" {
            rig.tap(KeyCode::Escape);
        }
        rig.settle_meshes();
        rig.client.mod_settings_mut().set_enabled(shader, false);
        rig.pump(30);
    }
    set_step("packs: back to none");
    rig.select_packs(&[]);
    rig.pump(30);
    rig.report("packs: none");
    rig.shot("none");
    rig.finish();
}

/// Copy a folder tree (a real resource pack) file by file.
fn copy_tree(from: &Path, to: &Path) {
    if from.is_file() {
        std::fs::copy(from, to).expect("copy a pack file");
        return;
    }
    std::fs::create_dir_all(to).expect("pack folder");
    for entry in std::fs::read_dir(from).into_iter().flatten().flatten() {
        copy_tree(&entry.path(), &to.join(entry.file_name()));
    }
}

/// One mod switched away from its default and on, each option to its extremes and every
/// choice, then reset and back to its default (`rest`: on or off at the end instead).
fn sweep_mod(
    rig: &mut Rig,
    descriptor: &'static mc_mods::ModDescriptor,
    pass: &str,
    rest: Option<bool>,
) {
    let id = descriptor.id;
    set_step(&format!("mods/{pass}: {id} on"));
    let default_on = rig.client.mod_settings_mut().enabled(id);
    rig.client.mod_settings_mut().set_enabled(id, !default_on);
    rig.pump(8);
    rig.client.mod_settings_mut().set_enabled(id, true);
    rig.pump(8);
    for option in descriptor.options {
        let values: Vec<OptionValue> = match option.kind {
            OptionKind::Toggle => vec![OptionValue::Bool(true), OptionValue::Bool(false)],
            OptionKind::Slider { min, max, .. } | OptionKind::SliderLabeled { min, max, .. } => {
                vec![
                    OptionValue::Float(min),
                    OptionValue::Float(max),
                    OptionValue::Float((min + max) * 0.5),
                ]
            }
            OptionKind::Choice { choices } => (0..choices.len()).map(OptionValue::Choice).collect(),
            OptionKind::Color { .. } => vec![
                OptionValue::Color(0x0000_0000),
                OptionValue::Color(0xFFFF_00FF),
            ],
            OptionKind::Text { max_len } => vec![
                OptionValue::Text("".into()),
                OptionValue::Text(
                    "W\u{e9}\u{300}\u{1F600}\u{202E}\u{a7}l\u{a7}k%s%n{}"
                        .chars()
                        .cycle()
                        .take(max_len.max(1))
                        .collect::<String>()
                        .into(),
                ),
            ],
            // Keys, hidden state and buttons: left alone (a key bound here would act on the
            // sweep's own key presses; buttons open other screens).
            _ => Vec::new(),
        };
        for value in values {
            set_step(&format!("mods/{pass}: {id}.{} = {value:?}", option.id));
            rig.client
                .mod_settings_mut()
                .set_value(id, option.id, value);
            rig.pump(4);
        }
    }
    set_step(&format!("mods/{pass}: {id} reset and default"));
    rig.client.mod_settings_mut().reset_options(id);
    rig.pump(4);
    rig.client
        .mod_settings_mut()
        .set_enabled(id, rest.unwrap_or(descriptor.default_enabled));
    rig.pump(6);
}

/// `SWEEP_MODS=<id>,<id>`: what `every_mod_on_and_off` flags, looked at closer. Each listed mod
/// is swept as there (the world ticking, no packet arriving), then the frame is compared with
/// the one before the sweep after 30, 100, 300 and 600 more ticks, with pictures: a difference
/// that fades is smoothing settling, one that stays is state left behind.
#[test]
#[ignore = "GPU + vanilla jar + the benchmark capture; pictures"]
fn mod_toggle_pictures() {
    let wanted = std::env::var("SWEEP_MODS").unwrap_or_default();
    let Some(mut rig) = Rig::new("toggle", true, |dir| {
        std::fs::write(dir.join("options.txt"), "particles:2\nrenderClouds:false\n")
            .expect("write options.txt");
    }) else {
        return;
    };
    mc_render::entity::item::pin_animation_clock(Some(1_000_000));
    rig.packets = false;
    set_step("toggle: settle");
    rig.pump(300);
    for id in wanted.split(',').filter(|id| !id.is_empty()) {
        let Some(descriptor) = mc_mods::catalog::find(id) else {
            println!("   no mod {id}");
            continue;
        };
        let before = rig.pixels();
        rig.save_png(&format!("{id}-before"), &before);
        sweep_mod(&mut rig, descriptor, "toggle", None);
        let mut ticks = 0;
        for until in [30, 100, 300, 600] {
            set_step(&format!("toggle: {id} +{until} ticks"));
            rig.pump(((until - ticks) * rig.per_tick) as usize);
            ticks = until;
            println!(
                "   {id} +{until} ticks: mesh queue {} queued, {} running, {} deferred",
                rig.client.mesh_queue_len(),
                rig.client.mesh_queue_in_flight_len(),
                rig.client.mesh_queue_deferred_len()
            );
            let after = rig.pixels();
            println!(
                "   {id} +{until} ticks: {} pixels visibly differ",
                visibly_differing_pixels(&before, &after, 8)
            );
            rig.save_png(&format!("{id}-after-{until}"), &after);
        }
    }
    mc_render::entity::item::pin_animation_clock(None);
    rig.finish();
}

/// A setup profile as the mod menu loads one (`ModSettings::apply_setup`, every value in one
/// step): every mod on at once (a hand-edited or shared profile can hold what the menu would not
/// let a player click together, such as every shader pack on), every option at its maximum, every
/// other mod on, each followed by the setup from before. Back on the original setup the frame
/// must be the one from before (the world is still, no packet arrives).
#[test]
#[ignore = "GPU + vanilla jar + the benchmark capture; a sweep"]
fn profile_switches() {
    let Some(mut rig) = Rig::new("profiles", true, |dir| {
        std::fs::write(dir.join("options.txt"), "particles:2\nrenderClouds:false\n")
            .expect("write options.txt");
        std::fs::write(
            dir.join(mc_mods::SETTINGS_FILE_NAME),
            r#"{"format":1,"mods":{"vignette":{"enabled":false}}}"#,
        )
        .expect("write the mod settings");
    }) else {
        return;
    };
    mc_render::entity::item::pin_animation_clock(Some(1_000_000));
    rig.packets = false;
    set_step("profiles: settle");
    rig.pump(300);
    rig.settle_meshes();
    let original = rig.client.mod_settings_mut().clone();
    let catalog = mc_mods::catalog::all();
    // The desktop media session mod stays off (it would reach the PC's media player).
    let setup = |on: &dyn Fn(usize) -> bool, maxed: bool| {
        let mut setup = original.clone();
        for (n, descriptor) in catalog.iter().enumerate() {
            let id = descriptor.id;
            setup.set_enabled(id, id != "spotify" && on(n));
            if !maxed || id == "spotify" {
                continue;
            }
            for option in descriptor.options {
                let value = match option.kind {
                    OptionKind::Toggle => OptionValue::Bool(true),
                    OptionKind::Slider { max, .. } | OptionKind::SliderLabeled { max, .. } => {
                        OptionValue::Float(max)
                    }
                    OptionKind::Choice { choices } => {
                        OptionValue::Choice(choices.len().saturating_sub(1))
                    }
                    _ => continue,
                };
                setup.set_value(id, option.id, value);
            }
        }
        setup
    };
    let setups = [
        ("every mod on", setup(&|_| true, false)),
        ("every option at its maximum", setup(&|_| true, true)),
        ("every other mod on", setup(&|n| n % 2 == 0, false)),
        ("the other half on", setup(&|n| n % 2 == 1, true)),
    ];
    let before = rig.pixels();
    rig.save_png("before", &before);
    let mut left_behind: Vec<(&str, usize)> = Vec::new();
    for (label, profile) in &setups {
        let label: &str = label;
        set_step(&format!("profiles: load {label}"));
        let changed = rig.client.mod_settings_mut().apply_setup(profile);
        rig.pump(60);
        rig.settle_meshes();
        rig.pump(30);
        rig.shot(label);
        println!("   {label}: changed {changed}");
        set_step(&format!("profiles: back from {label}"));
        rig.client.mod_settings_mut().apply_setup(&original);
        rig.pump(90);
        rig.settle_meshes();
        rig.pump(30);
        let after = rig.pixels();
        let visibly = visibly_differing_pixels(&before, &after, 8);
        println!(
            "   back from {label}: {} pixels differ, {visibly} visibly",
            differing_pixels(&before, &after)
        );
        if visibly > 0 {
            left_behind.push((label, visibly));
            rig.save_png(&format!("after-{label}"), &after);
        }
    }
    rig.report("profiles");
    mc_render::entity::item::pin_animation_clock(None);
    rig.finish();
    assert!(
        left_behind.is_empty(),
        "the original setup draws a different frame after: {left_behind:?}"
    );
}

/// Every mod in the catalog on, each option to its extremes and every choice, then off, with
/// the world ticking but no packet arriving (the scene settles and stays); afterwards each mod
/// back at its default must draw the same frame as before its sweep (anything else is state the
/// mod left behind). A second pass does the same with the capture's packets flowing (no picture
/// comparison).
#[test]
#[ignore = "GPU + vanilla jar + the benchmark capture; a sweep"]
fn every_mod_on_and_off() {
    let Some(mut rig) = Rig::new("mods", true, |dir| {
        // Fewer moving parts: minimal particles, no clouds (they move every tick) and no vignette
        // (it eases toward the light level for hundreds of frames).
        std::fs::write(dir.join("options.txt"), "particles:2\nrenderClouds:false\n")
            .expect("write options.txt");
        std::fs::write(
            dir.join(mc_mods::SETTINGS_FILE_NAME),
            r#"{"format":1,"mods":{"vignette":{"enabled":false}}}"#,
        )
        .expect("write the mod settings");
    }) else {
        return;
    };
    // The scene settles and stays: the world keeps ticking (the mods' tick hooks run) but no
    // packet arrives; the animation clock is pinned.
    mc_render::entity::item::pin_animation_clock(Some(1_000_000));
    rig.packets = false;
    set_step("mods: settle");
    rig.pump(300);
    rig.shot("before");
    let catalog = mc_mods::catalog::all();
    // Per mod, in the still pass: (id, pixels differing from the frame before its sweep).
    let mut left_behind: Vec<(&str, usize)> = Vec::new();
    for pass in ["still", "ticking"] {
        rig.packets = pass == "ticking";
        for descriptor in catalog {
            let id = descriptor.id;
            if pass == "still" {
                rig.settle_meshes();
            }
            let before = (pass == "still").then(|| rig.pixels());
            if id == "spotify" {
                // Reads and controls the desktop's media session: on and off only, no options.
                set_step(&format!("mods/{pass}: {id} on"));
                rig.client.mod_settings_mut().set_enabled(id, true);
                rig.pump(10);
                rig.client.mod_settings_mut().set_enabled(id, false);
                rig.pump(4);
                continue;
            }
            // The vanilla vignette eases toward the light level by 1% a frame: after a mod that
            // changed the light it is still easing several hundred frames later. It rests off in
            // the still pass (its own sweep still runs) so that easing is not read as a leftover.
            let rest = (pass == "still" && id == "vignette").then_some(false);
            sweep_mod(&mut rig, descriptor, pass, rest);
            if let Some(before) = before {
                // The world has not moved: the mod back at its default must draw the same frame
                // (meshes rebuilt by a mesher option get frames to land).
                rig.pump(90);
                rig.settle_meshes();
                let after = rig.pixels();
                let differing = differing_pixels(&before, &after);
                let visibly = visibly_differing_pixels(&before, &after, 8);
                if differing > 0 {
                    println!(
                        "   {id}: {differing} pixels differ after its sweep, {visibly} visibly"
                    );
                }
                if visibly > 0 {
                    left_behind.push((id, visibly));
                    rig.save_png(&format!("{id}-before"), &before);
                    rig.save_png(&format!("{id}-after"), &after);
                }
            }
        }
        rig.report(&format!("after the {pass} pass"));
    }
    mc_render::entity::item::pin_animation_clock(None);
    rig.finish();
    println!("== mods whose sweep left the frame different: {left_behind:?}");
    assert!(
        left_behind.is_empty(),
        "mods back at their defaults draw a different frame: {left_behind:?}"
    );
}
